from __future__ import annotations

import copy

import pytest

from eval.opensource.spai_metrics import (
    FIXED_THRESHOLD,
    THRESHOLD_OPERATOR,
    spai_detection_metrics_strict,
    summarize_spai_pair_slice,
    summarize_spai_results,
)


def _expected(sample_id: str, task_id: str, kind: str, domain: str) -> dict:
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
    visibility: str,
    fraction: float,
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
        _expected("r1", "t1", "real", "lodging"),
        _expected("f1", "t1", "forged", "lodging"),
        _expected("r2", "t2", "real", "restaurant"),
        _expected("f2", "t2", "forged", "restaurant"),
    ]
    rows = [
        _result(expected[0], 0.1, visibility="full", fraction=1.0),
        _result(expected[1], 0.9, visibility="full", fraction=1.0),
        _result(expected[2], 0.5, visibility="none", fraction=0.0),
        _result(expected[3], 0.5, visibility="none", fraction=0.0),
    ]
    return expected, rows


def test_frozen_threshold_is_strict_greater_than() -> None:
    assert FIXED_THRESHOLD == 0.5
    assert THRESHOLD_OPERATOR == ">"
    detection = spai_detection_metrics_strict(
        [
            {"status": "ok", "label": 0, "ai_score": 0.5},
            {"status": "ok", "label": 1, "ai_score": 0.5},
        ]
    )
    assert (detection["tp"], detection["fp"]) == (0, 0)
    assert (detection["fn"], detection["tn"]) == (1, 1)


@pytest.mark.parametrize("score", [-0.1, 1.1, float("nan"), float("inf")])
def test_detection_rejects_invalid_probability(score: float) -> None:
    with pytest.raises(ValueError):
        spai_detection_metrics_strict(
            [{"status": "ok", "label": 1, "ai_score": score}]
        )


def test_pair_slice_bootstrap_is_deterministic() -> None:
    _, rows = _two_pairs()
    pairs = [
        {"real": rows[0], "forged": rows[1]},
        {"real": rows[2], "forged": rows[3]},
    ]
    first = summarize_spai_pair_slice(pairs, iterations=30, seed=9)
    second = summarize_spai_pair_slice(pairs, iterations=30, seed=9)
    assert first == second
    assert first["bootstrap_unit"] == "task_id_pair"
    assert first["paired_ranking"]["wins"] == 1
    assert first["paired_ranking"]["ties"] == 1


def test_summary_is_t1_only_and_stratified() -> None:
    expected, rows = _two_pairs()
    summary = summarize_spai_results(
        rows,
        expected,
        bootstrap_samples=20,
        seed=3,
    )
    assert summary["schema_version"] == "spai_detection_summary_v1"
    assert summary["coverage"]["is_complete"] is True
    assert summary["task_scope"]["valid_for_t1"] is True
    assert summary["task_scope"]["valid_for_t2"] is False
    assert set(summary["by_domain"]) == {"lodging", "restaurant"}
    assert set(summary["by_edit_visibility"]) == {"full", "none"}


def test_summary_preserves_retry_history_and_rejects_identity_tamper() -> None:
    expected, rows = _two_pairs()
    error = copy.deepcopy(rows[1])
    error.update({"status": "error", "ai_score": None})
    summary = summarize_spai_results(
        [*rows, error, rows[1]],
        expected,
        bootstrap_samples=5,
    )
    assert summary["coverage"]["physical_result_rows"] == 6
    tampered = copy.deepcopy(rows)
    tampered[0]["task_id"] = "wrong"
    with pytest.raises(ValueError, match="task_id"):
        summarize_spai_results(
            tampered,
            expected,
            bootstrap_samples=5,
        )
