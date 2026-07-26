from __future__ import annotations

import copy

import pytest

from eval.opensource.community_forensics_metrics import (
    FIXED_THRESHOLD,
    THRESHOLD_OPERATOR,
    community_forensics_detection_metrics_strict,
    summarize_community_forensics_pair_slice,
    summarize_community_forensics_results,
)


def _expected(
    sample_id: str,
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


def _result(
    expected: dict,
    score: float,
    *,
    visibility: str = "full",
    fraction: float = 1.0,
) -> dict:
    return {
        "id": expected["sample_id"],
        "task_id": expected["task_id"],
        "kind": expected["kind"],
        "label": expected["label"],
        "domain": expected["domain"],
        "status": "ok",
        "ai_score": score,
        "edit_visibility": visibility,
        "edit_visible_gt_fraction": fraction,
        "latency_ms": 1.0,
        "peak_cuda_memory_bytes": None,
    }


def _two_pairs() -> tuple[list[dict], list[dict]]:
    expected = [
        _expected("r1", "t1", "real"),
        _expected("f1", "t1", "forged"),
        _expected("r2", "t2", "real", "restaurant"),
        _expected("f2", "t2", "forged", "restaurant"),
    ]
    rows = [
        _result(expected[0], 0.1),
        _result(expected[1], 0.9),
        _result(expected[2], 0.5, visibility="none", fraction=0.0),
        _result(expected[3], 0.5, visibility="none", fraction=0.0),
    ]
    return expected, rows


def test_frozen_threshold_contract() -> None:
    assert FIXED_THRESHOLD == 0.5
    assert THRESHOLD_OPERATOR == ">"


def test_detection_uses_strict_greater_than_threshold() -> None:
    rows = [
        {"status": "ok", "label": 0, "ai_score": 0.5},
        {"status": "ok", "label": 1, "ai_score": 0.5},
    ]
    detection = community_forensics_detection_metrics_strict(rows)
    assert detection["tp"] == 0
    assert detection["fp"] == 0
    assert detection["fn"] == 1
    assert detection["tn"] == 1
    assert detection["threshold_operator"] == ">"


@pytest.mark.parametrize("score", [-0.1, 1.1, float("nan"), float("inf")])
def test_detection_rejects_invalid_probabilities(score: float) -> None:
    with pytest.raises(ValueError):
        community_forensics_detection_metrics_strict(
            [{"status": "ok", "label": 1, "ai_score": score}]
        )


def test_detection_rejects_nonreleased_threshold() -> None:
    with pytest.raises(ValueError, match="fixed threshold"):
        community_forensics_detection_metrics_strict([], threshold=0.4)


def test_pair_slice_is_deterministic_and_pair_bootstrapped() -> None:
    expected, rows = _two_pairs()
    pairs = [
        {"real": rows[0], "forged": rows[1]},
        {"real": rows[2], "forged": rows[3]},
    ]
    first = summarize_community_forensics_pair_slice(
        pairs,
        iterations=40,
        seed=7,
    )
    second = summarize_community_forensics_pair_slice(
        pairs,
        iterations=40,
        seed=7,
    )
    assert first == second
    assert first["bootstrap_unit"] == "task_id_pair"
    assert first["pairs"] == 2
    assert first["paired_ranking"]["wins"] == 1
    assert first["paired_ranking"]["ties"] == 1


def test_summary_has_method_specific_schema_and_t1_only_scope() -> None:
    expected, rows = _two_pairs()
    summary = summarize_community_forensics_results(
        rows,
        expected,
        bootstrap_samples=20,
        seed=4,
    )
    assert (
        summary["schema_version"]
        == "community_forensics_detection_summary_v1"
    )
    assert summary["coverage"]["is_complete"] is True
    assert summary["paired_coverage"]["complete_valid_pairs"] == 2
    assert summary["task_scope"]["valid_for_t1"] is True
    assert summary["task_scope"]["valid_for_t2"] is False
    assert set(summary["by_domain"]) == {"lodging", "restaurant"}
    assert set(summary["by_edit_visibility"]) == {"full", "none"}


def test_summary_uses_latest_physical_retry() -> None:
    expected, rows = _two_pairs()
    retry = copy.deepcopy(rows[1])
    retry["status"] = "error"
    retry["ai_score"] = None
    history = [*rows, retry, rows[1]]
    summary = summarize_community_forensics_results(
        history,
        expected,
        bootstrap_samples=10,
    )
    assert summary["coverage"]["valid_images"] == 4
    assert summary["coverage"]["physical_result_rows"] == 6


def test_summary_rejects_visibility_fraction_mismatch() -> None:
    expected, rows = _two_pairs()
    rows[0]["edit_visibility"] = "full"
    rows[0]["edit_visible_gt_fraction"] = 0.2
    with pytest.raises(ValueError, match="category fraction mismatch"):
        summarize_community_forensics_results(
            rows,
            expected,
            bootstrap_samples=5,
        )


def test_summary_rejects_result_identity_tamper() -> None:
    expected, rows = _two_pairs()
    rows[0]["task_id"] = "other"
    with pytest.raises(ValueError, match="task_id"):
        summarize_community_forensics_results(
            rows,
            expected,
            bootstrap_samples=5,
        )


def test_summary_reports_incomplete_errors_without_bootstrapping_pair() -> None:
    expected, rows = _two_pairs()
    rows[1]["status"] = "error"
    rows[1]["ai_score"] = None
    summary = summarize_community_forensics_results(
        rows,
        expected,
        bootstrap_samples=5,
    )
    assert summary["coverage"]["is_complete"] is False
    assert summary["coverage"]["error_images"] == 1
    assert summary["paired_coverage"]["complete_valid_pairs"] == 1
