from __future__ import annotations

import copy
import json
import math

import pytest

from eval.opensource.omniaid_metrics import (
    DEFAULT_BOOTSTRAP_SEED,
    FIXED_THRESHOLD,
    SCHEMA_VERSION,
    SCORE_SEMANTICS,
    THRESHOLD_OPERATOR,
    omniaid_detection_metrics_strict,
    summarize_omniaid_pair_slice,
    summarize_omniaid_results,
)


def _expected(
    sample_id: str,
    task_id: str,
    kind: str,
    *,
    domain: str = "lodging",
) -> dict:
    return {
        "sample_id": sample_id,
        "task_id": task_id,
        "kind": kind,
        "label": int(kind == "forged"),
        "domain": domain,
    }


def _result(
    expected: dict,
    probability: float,
    *,
    visibility: str = "full",
    fraction: float = 1.0,
    status: str = "ok",
    peak_cuda_memory_bytes: int | None = None,
) -> dict:
    result = {
        "id": expected["sample_id"],
        "task_id": expected["task_id"],
        "kind": expected["kind"],
        "label": expected["label"],
        "domain": expected["domain"],
        "status": status,
        "edit_visibility": visibility,
        "edit_visible_gt_fraction": fraction,
        "latency_ms": 1.25,
        "peak_cuda_memory_bytes": peak_cuda_memory_bytes,
    }
    if status == "ok":
        result.update(
            {
                "ai_score": probability,
                "fake_probability": probability,
                "score": probability,
            }
        )
    else:
        result.update(
            {
                "ai_score": None,
                "fake_probability": None,
                "score": None,
            }
        )
    return result


def _fixture() -> tuple[list[dict], list[dict]]:
    expected = [
        _expected("a-real", "a", "real"),
        _expected("a-forged", "a", "forged"),
        _expected("b-real", "b", "real", domain="restaurant"),
        _expected("b-forged", "b", "forged", domain="restaurant"),
    ]
    rows = [
        _result(expected[0], 0.1),
        _result(expected[1], 0.9),
        _result(
            expected[2],
            0.8,
            visibility="partial",
            fraction=0.25,
        ),
        _result(
            expected[3],
            0.5,
            visibility="partial",
            fraction=0.25,
        ),
    ]
    return rows, expected


def _pairs(rows: list[dict]) -> list[dict]:
    return [
        {"real": rows[0], "forged": rows[1]},
        {"real": rows[2], "forged": rows[3]},
    ]


def test_frozen_probability_contract_and_default_seed() -> None:
    assert FIXED_THRESHOLD == 0.5
    assert THRESHOLD_OPERATOR == ">"
    assert DEFAULT_BOOTSTRAP_SEED == 20260724
    assert SCHEMA_VERSION == "omniaid_detection_summary_v1"


def test_detection_is_probability_bounded_and_strictly_greater_than_half() -> None:
    result = omniaid_detection_metrics_strict(
        [
            {"status": "ok", "label": 0, "ai_score": 0.0},
            {"status": "ok", "label": 0, "fake_probability": 0.5},
            {"status": "ok", "label": 1, "ai_score": 0.5},
            {"status": "ok", "label": 1, "ai_score": 1.0},
        ]
    )

    assert result["threshold"] == 0.5
    assert result["threshold_operator"] == ">"
    assert (result["tp"], result["fp"], result["fn"], result["tn"]) == (
        1,
        0,
        1,
        2,
    )
    assert result["score_key"] == "ai_score"
    assert result["score_aliases"] == ["ai_score", "fake_probability"]
    assert result["score_semantics"] == SCORE_SEMANTICS


@pytest.mark.parametrize(
    ("score", "message"),
    [
        (math.nan, "not finite"),
        (math.inf, "not finite"),
        (-1e-9, r"outside \[0, 1\]"),
        (1.000000001, r"outside \[0, 1\]"),
        (True, "real number"),
        ("0.5", "real number"),
        (None, "real number"),
    ],
)
def test_detection_rejects_bad_probability(
    score: object,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        omniaid_detection_metrics_strict(
            [{"status": "ok", "label": 1, "ai_score": score}]
        )


def test_detection_rejects_probability_alias_drift() -> None:
    with pytest.raises(ValueError, match="fake_probability differs"):
        omniaid_detection_metrics_strict(
            [
                {
                    "status": "ok",
                    "label": 1,
                    "ai_score": 0.75,
                    "fake_probability": 0.7500001,
                }
            ]
        )


@pytest.mark.parametrize("threshold", [0.0, 0.5000000001, 1.0])
def test_nonreleased_threshold_is_rejected_as_omniaid(
    threshold: float,
) -> None:
    with pytest.raises(ValueError, match="OmniAID uses fixed threshold 0.5"):
        omniaid_detection_metrics_strict([], threshold=threshold)


def test_pair_slice_is_complete_pair_bootstrapped_and_deterministic() -> None:
    rows, _ = _fixture()
    pairs = _pairs(rows)
    first = summarize_omniaid_pair_slice(pairs, iterations=80, seed=13)
    second = summarize_omniaid_pair_slice(pairs, iterations=80, seed=13)

    assert first == second
    assert first["pairs"] == 2
    assert first["images"] == 4
    assert first["bootstrap_unit"] == "task_id_pair"
    assert first["bootstrap_samples"] == 80
    assert first["paired_ranking"]["wins"] == 1
    assert first["paired_ranking"]["losses"] == 1
    assert first["paired_ranking"]["ties"] == 0
    assert first["image_confusion_at_0_5"] == {
        "tp": 1,
        "fp": 1,
        "fn": 1,
        "tn": 1,
    }
    assert first["score_semantics"] == SCORE_SEMANTICS


def test_pair_slice_rejects_identity_and_visibility_drift() -> None:
    rows, _ = _fixture()
    pairs = _pairs(rows)

    wrong_domain = copy.deepcopy(pairs)
    wrong_domain[0]["forged"]["domain"] = "restaurant"
    with pytest.raises(ValueError, match="mismatched domains"):
        summarize_omniaid_pair_slice(wrong_domain, iterations=2)

    wrong_visibility = copy.deepcopy(pairs)
    wrong_visibility[0]["forged"]["edit_visibility"] = "partial"
    wrong_visibility[0]["forged"]["edit_visible_gt_fraction"] = 0.25
    with pytest.raises(ValueError, match="mismatched edit_visibility"):
        summarize_omniaid_pair_slice(wrong_visibility, iterations=2)

    bad_category = copy.deepcopy(pairs)
    bad_category[0]["real"]["edit_visibility"] = "tiny"
    with pytest.raises(ValueError, match="invalid category"):
        summarize_omniaid_pair_slice(bad_category, iterations=2)


def test_summary_has_omniaid_schema_domains_visibility_and_coverage() -> None:
    rows, expected = _fixture()
    summary = summarize_omniaid_results(
        rows,
        expected,
        bootstrap_samples=30,
        seed=7,
    )

    assert summary["schema_version"] == "omniaid_detection_summary_v1"
    assert "ufd" not in json.dumps(summary, sort_keys=True).lower()
    assert summary["task_scope"]["valid_for_t1"] is True
    assert summary["task_scope"]["valid_for_t2"] is False
    assert summary["task_scope"]["primary_score"] == "ai_score"
    assert summary["task_scope"]["score_semantics"] == SCORE_SEMANTICS
    assert summary["coverage"] == {
        "expected_images": 4,
        "physical_result_rows": 4,
        "result_images": 4,
        "valid_images": 4,
        "error_images": 0,
        "missing_images": 0,
        "coverage_fraction": 1.0,
        "valid_fraction": 1.0,
        "is_complete": True,
    }
    assert summary["paired_coverage"]["complete_valid_pairs"] == 2
    assert set(summary["by_domain"]) == {"lodging", "restaurant"}
    assert set(summary["by_edit_visibility"]) == {"full", "partial"}
    assert summary["pair_bootstrap"]["score_semantics"] == SCORE_SEMANTICS
    assert all(
        value["score_semantics"] == SCORE_SEMANTICS
        for value in summary["by_domain"].values()
    )


def test_summary_uses_last_physical_retry_and_counts_all_attempts() -> None:
    rows, expected = _fixture()
    error = _result(
        expected[1],
        0.0,
        status="error",
    )
    history = [*rows, error, copy.deepcopy(rows[1])]
    summary = summarize_omniaid_results(
        history,
        expected,
        bootstrap_samples=8,
    )

    assert summary["coverage"]["physical_result_rows"] == 6
    assert summary["coverage"]["result_images"] == 4
    assert summary["coverage"]["valid_images"] == 4
    assert summary["coverage"]["is_complete"] is True
    assert summary["paired_coverage"]["complete_valid_pairs"] == 2


def test_summary_validates_bad_old_retry_not_only_latest() -> None:
    rows, expected = _fixture()
    invalid_old = copy.deepcopy(rows[0])
    invalid_old["ai_score"] = -0.1
    invalid_old["fake_probability"] = -0.1
    invalid_old["score"] = -0.1

    with pytest.raises(ValueError, match=r"outside \[0, 1\]"):
        summarize_omniaid_results(
            [invalid_old, *rows],
            expected,
            bootstrap_samples=4,
        )


def test_summary_last_error_breaks_pair_and_complete_coverage() -> None:
    rows, expected = _fixture()
    history = [
        *rows,
        _result(expected[1], 0.0, status="error"),
    ]
    summary = summarize_omniaid_results(
        history,
        expected,
        bootstrap_samples=5,
    )

    assert summary["coverage"]["error_images"] == 1
    assert summary["coverage"]["is_complete"] is False
    assert summary["paired_coverage"]["complete_valid_pairs"] == 1
    assert summary["paired_coverage"]["unpaired_valid_images"] == 1


def test_summary_rejects_visibility_fraction_category_mismatch() -> None:
    rows, expected = _fixture()
    rows[0]["edit_visibility"] = "full"
    rows[0]["edit_visible_gt_fraction"] = 0.2

    with pytest.raises(ValueError, match="category fraction mismatch"):
        summarize_omniaid_results(
            rows,
            expected,
            bootstrap_samples=4,
        )


def test_summary_is_deterministic_for_fixed_complete_pair_seed() -> None:
    rows, expected = _fixture()
    first = summarize_omniaid_results(
        rows,
        expected,
        bootstrap_samples=100,
        seed=DEFAULT_BOOTSTRAP_SEED,
    )
    second = summarize_omniaid_results(
        rows,
        expected,
        bootstrap_samples=100,
        seed=DEFAULT_BOOTSTRAP_SEED,
    )

    assert first == second
