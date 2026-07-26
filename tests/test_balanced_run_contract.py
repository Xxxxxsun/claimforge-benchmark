from __future__ import annotations

import copy
import hashlib
import math
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import SimpleNamespace

import pytest

from eval.opensource.balanced_run_contract import (
    RESULT_SCHEMA_VERSION,
    RUN_DATASET_CONTRACT_SCHEMA_VERSION,
    ContractError,
    IncompleteCoverageError,
    ScoreSpec,
    build_result_identity,
    build_run_dataset_contract,
    index_latest_attempts,
    require_complete_coverage,
    selected_ids_sha256,
    summarize_coverage,
)
from eval.opensource.canonical_release import Capability, SelectionSpec
from eval.opensource.common import stable_json


SCHEMA = "claimforge_balanced250_canonical_v1"
DATASET_ID = "balanced-fixture"
RUN_ID = "fixture-run"
FINGERPRINT = "f" * 64


def _rows_sha256(rows: list[dict]) -> str:
    payload = "".join(f"{stable_json(row)}\n" for row in rows).encode()
    return hashlib.sha256(payload).hexdigest()


class FakeCapability(Enum):
    WHOLE_IMAGE = "whole_image_t1"
    LOCALIZATION = "localization_t2"

    @property
    def conditions(self) -> tuple[str, ...]:
        if self is FakeCapability.WHOLE_IMAGE:
            return (
                "real",
                "local_mouse",
                "fullframe_cat",
            )
        return ("real", "local_mouse")

    @property
    def valid_for_t1(self) -> bool:
        return self is FakeCapability.WHOLE_IMAGE

    @property
    def valid_for_t2(self) -> bool:
        return self is FakeCapability.LOCALIZATION


@dataclass(frozen=True)
class FakeSelectionSpec:
    capability: FakeCapability
    conditions: tuple[str, ...] | None = None
    per_condition_limit: int | None = None
    sample_id: str | None = None
    pair_limit: int | None = None


def _row(rank: int, condition: str) -> dict:
    semantics = {
        "real": (
            "real",
            "authentic",
            "real",
            0,
            "all_zero",
        ),
        "local_mouse": (
            "local_splice",
            "local_insertion",
            "forged",
            1,
            "exact_diff",
        ),
        "fullframe_cat": (
            "full_frame_conditional_edit",
            "conditional_full_frame_edit",
            "forged",
            1,
            "not_applicable",
        ),
    }
    family, scope, kind, label, gt_kind = semantics[condition]
    return {
        "schema_version": SCHEMA,
        "dataset_id": DATASET_ID,
        "sample_id": f"sample-{rank}",
        "rank": rank,
        "condition": condition,
        "condition_family": family,
        "manipulation_scope": scope,
        "normalized_task_id": f"normalized-{rank}",
        "task_id": f"task-{rank}",
        "kind": kind,
        "label": label,
        "domain": "lodging" if rank % 2 == 0 else "restaurant",
        "gt_mask_kind": gt_kind,
        "canonical_path": f"release/images/sample-{rank}.jpg",
        "canonical_sha256": f"{rank + 1:x}" * 64,
        "width": 640 + rank,
        "height": 480 + rank,
    }


def _rows() -> list[dict]:
    return [
        _row(0, "real"),
        _row(1, "local_mouse"),
        _row(2, "fullframe_cat"),
    ]


def _release(tmp_path: Path, rows: list[dict] | None = None) -> SimpleNamespace:
    inputs = copy.deepcopy(rows if rows is not None else _rows())
    panel = [
        {"sample_id": row["sample_id"], "condition": row["condition"]}
        for row in inputs
    ]
    source_pairs = [
        {
            "real_sample_id": inputs[0]["sample_id"],
            "forged_sample_id": row["sample_id"],
            "condition": row["condition"],
        }
        for row in inputs
        if row["kind"] == "forged"
    ]
    ledgers = {
        "inputs": {
            "path": "release/inputs.jsonl",
            "sha256": _rows_sha256(inputs),
            "rows": len(inputs),
        },
        "panel": {
            "path": "release/panel.jsonl",
            "sha256": _rows_sha256(panel),
            "rows": len(panel),
        },
        "source_pairs": {
            "path": "release/source_pairs.jsonl",
            "sha256": _rows_sha256(source_pairs),
            "rows": len(source_pairs),
        },
    }
    ledger_views = {
        name: SimpleNamespace(
            name=name,
            path=tmp_path / value["path"],
            sha256=value["sha256"],
            rows=value["rows"],
        )
        for name, value in ledgers.items()
    }
    return SimpleNamespace(
        repo_root=tmp_path,
        manifest_path=tmp_path / "release/manifest.json",
        manifest_sha256="a" * 64,
        manifest={
            "schema_version": SCHEMA,
            "dataset_id": DATASET_ID,
            "contract_sha256": "b" * 64,
            "ledgers": ledgers,
        },
        schema_version=SCHEMA,
        dataset_id=DATASET_ID,
        release_kind="balanced250",
        contract_sha256="b" * 64,
        inputs_ledger=ledger_views["inputs"],
        inputs=inputs,
        panel_ledger=ledger_views["panel"],
        panel=panel,
        source_pairs_ledger=ledger_views["source_pairs"],
        source_pairs=source_pairs,
        legacy_pairs_ledger=None,
        legacy_pairs=None,
    )


def _score_spec() -> ScoreSpec:
    return ScoreSpec(
        key="ai_score",
        direction="higher_means_fake",
        fixed_threshold=0.5,
        threshold_operator=">",
    )


def _ok_attempt(row: dict, score: float = 0.75) -> dict:
    return {
        **build_result_identity(
            row,
            run_id=RUN_ID,
            run_manifest_fingerprint=FINGERPRINT,
        ),
        "status": "ok",
        "valid_for_metrics": True,
        "ai_score": score,
    }


def _error_attempt(row: dict) -> dict:
    return {
        **build_result_identity(
            row,
            run_id=RUN_ID,
            run_manifest_fingerprint=FINGERPRINT,
        ),
        "status": "error",
        "valid_for_metrics": False,
        "ai_score": None,
        "error_type": "RuntimeError",
        "error_message": "fixture error",
    }


def test_result_identity_v2_propagates_condition_fields_without_pair_rank():
    row = _row(1, "local_mouse")
    identity = build_result_identity(
        row,
        run_id=RUN_ID,
        run_manifest_fingerprint=FINGERPRINT,
    )

    assert identity == {
        "schema_version": RESULT_SCHEMA_VERSION,
        "run_id": RUN_ID,
        "run_manifest_fingerprint": FINGERPRINT,
        "dataset_id": DATASET_ID,
        "id": "sample-1",
        "sample_id": "sample-1",
        "rank": 1,
        "condition": "local_mouse",
        "condition_family": "local_splice",
        "manipulation_scope": "local_insertion",
        "normalized_task_id": "normalized-1",
        "task_id": "task-1",
        "kind": "forged",
        "label": 1,
        "domain": "restaurant",
        "gt_mask_kind": "exact_diff",
        "input_path": "release/images/sample-1.jpg",
        "input_sha256": "2" * 64,
        "input_width": 641,
        "input_height": 481,
    }
    assert "pair_rank" not in identity


def test_result_identity_rejects_pair_rank_and_condition_semantic_drift():
    with_pair = {**_row(0, "real"), "pair_rank": 0}
    with pytest.raises(ContractError, match="must not contain pair_rank"):
        build_result_identity(
            with_pair,
            run_id=RUN_ID,
            run_manifest_fingerprint=FINGERPRINT,
        )

    wrong_scope = {**_row(2, "fullframe_cat"), "gt_mask_kind": "exact_diff"}
    with pytest.raises(ContractError, match="requires gt_mask_kind"):
        build_result_identity(
            wrong_scope,
            run_id=RUN_ID,
            run_manifest_fingerprint=FINGERPRINT,
        )


def test_score_spec_is_strict_and_uses_the_frozen_operator():
    spec = _score_spec()
    assert spec.as_dict() == {
        "key": "ai_score",
        "direction": "higher_means_fake",
        "fixed_threshold": 0.5,
        "threshold_operator": ">",
    }
    assert spec.decision(0.50001) is True
    assert spec.decision(0.5) is False

    with pytest.raises(ContractError, match="requires > or >="):
        ScoreSpec("score", "higher_means_fake", 0.5, "<")
    with pytest.raises(ContractError, match="finite"):
        ScoreSpec("score", "higher_means_fake", math.nan, ">")
    with pytest.raises(ContractError, match="unsupported score direction"):
        ScoreSpec("score", "unknown", 0.5, ">")
    with pytest.raises(ContractError, match="reserved"):
        ScoreSpec("label", "higher_means_fake", 0.5, ">")


def test_dataset_contract_binds_release_ledgers_capability_and_selection(
    tmp_path: Path,
):
    release = _release(tmp_path)
    selection = FakeSelectionSpec(FakeCapability.WHOLE_IMAGE)
    contract = build_run_dataset_contract(
        release,
        selection,
        release.inputs,
        score_spec=_score_spec(),
    )
    value = contract.as_dict()

    assert value["schema_version"] == RUN_DATASET_CONTRACT_SCHEMA_VERSION
    assert value["release"] == {
        "schema_version": SCHEMA,
        "release_kind": "balanced250",
        "dataset_id": DATASET_ID,
        "manifest_path": "release/manifest.json",
        "manifest_sha256": "a" * 64,
        "contract_sha256": "b" * 64,
    }
    assert value["ledgers"] == release.manifest["ledgers"]
    assert value["capability"] == {
        "name": "whole_image_t1",
        "conditions": ["real", "local_mouse", "fullframe_cat"],
        "valid_for_t1": True,
        "valid_for_t2": False,
    }
    assert value["selection"] == {
        "spec": {
            "capability": "whole_image_t1",
            "conditions": None,
            "per_condition_limit": None,
            "sample_id": None,
            "pair_limit": None,
        },
        "selected_images": 3,
        "selected_ids_sha256": selected_ids_sha256(
            ["sample-0", "sample-1", "sample-2"]
        ),
        "counts_by_condition": {
            "real": 1,
            "local_mouse": 1,
            "fullframe_cat": 1,
        },
    }
    assert value["score_spec"] == _score_spec().as_dict()


def test_dataset_contract_serializes_all_selection_spec_fields(tmp_path: Path):
    release = _release(tmp_path)
    selection = FakeSelectionSpec(
        FakeCapability.WHOLE_IMAGE,
        conditions=("local_mouse",),
        per_condition_limit=1,
    )
    contract = build_run_dataset_contract(
        release,
        selection,
        [release.inputs[1]],
        score_spec=_score_spec(),
    )
    assert contract.as_dict()["selection"]["spec"] == {
        "capability": "whole_image_t1",
        "conditions": ["local_mouse"],
        "per_condition_limit": 1,
        "sample_id": None,
        "pair_limit": None,
    }

    single = FakeSelectionSpec(
        FakeCapability.WHOLE_IMAGE,
        sample_id="sample-2",
    )
    contract = build_run_dataset_contract(
        release,
        single,
        [release.inputs[2]],
        score_spec=_score_spec(),
    )
    assert contract.selection.sample_id == "sample-2"


def test_dataset_contract_accepts_frozen_canonical_selection_api(tmp_path: Path):
    release = _release(tmp_path)
    spec = SelectionSpec(
        capability=Capability.WHOLE_IMAGE_T1,
        conditions=("real",),
        per_condition_limit=1,
    )
    contract = build_run_dataset_contract(
        release,
        spec,
        [release.inputs[0]],
        score_spec=_score_spec(),
    )
    assert contract.capability.name == "whole_image_t1"
    assert contract.capability.conditions == (
        "real",
        "local_mouse",
        "local_cat",
        "local_trash_can",
        "fullframe_mouse",
        "fullframe_cat",
        "fullframe_trash_can",
    )
    assert contract.selection.conditions == ("real",)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda release: release.manifest.update(dataset_id="other"),
            "dataset_id drifted",
        ),
        (
            lambda release: setattr(release.inputs_ledger, "rows", 99),
            "materialized row count drifted",
        ),
        (
            lambda release: release.manifest["ledgers"]["inputs"].update(
                sha256="9" * 64
            ),
            "SHA-256 drifted",
        ),
        (
            lambda release: release.inputs[0].update(task_id="polluted"),
            "materialized content drifted",
        ),
    ],
)
def test_dataset_contract_rejects_release_binding_drift(
    tmp_path: Path,
    mutation,
    message: str,
):
    release = _release(tmp_path)
    mutation(release)
    with pytest.raises(ContractError, match=message):
        build_run_dataset_contract(
            release,
            FakeSelectionSpec(FakeCapability.WHOLE_IMAGE),
            release.inputs,
            score_spec=_score_spec(),
        )


def test_dataset_contract_rejects_selection_and_score_drift(tmp_path: Path):
    release = _release(tmp_path)

    changed = copy.deepcopy(release.inputs[1])
    changed["task_id"] = "other-task"
    with pytest.raises(ContractError, match="identity drifted"):
        build_run_dataset_contract(
            release,
            FakeSelectionSpec(FakeCapability.WHOLE_IMAGE),
            [release.inputs[0], changed],
            score_spec=_score_spec(),
        )

    with pytest.raises(ContractError, match="unique rank order"):
        build_run_dataset_contract(
            release,
            FakeSelectionSpec(FakeCapability.WHOLE_IMAGE),
            [release.inputs[1], release.inputs[0]],
            score_spec=_score_spec(),
        )

    changed_gt = copy.deepcopy(release.inputs)
    changed_gt[1]["gt_mask_path"] = "release/masks/changed.png"
    with pytest.raises(
        ContractError,
        match="content does not exactly match",
    ):
        build_run_dataset_contract(
            release,
            FakeSelectionSpec(FakeCapability.WHOLE_IMAGE),
            changed_gt,
            score_spec=_score_spec(),
        )

    with pytest.raises(ContractError, match="must not set pair_limit"):
        build_run_dataset_contract(
            release,
            FakeSelectionSpec(FakeCapability.WHOLE_IMAGE, pair_limit=1),
            release.inputs,
            score_spec=_score_spec(),
        )

    with pytest.raises(ContractError, match="has no score spec"):
        build_run_dataset_contract(
            release,
            FakeSelectionSpec(FakeCapability.WHOLE_IMAGE),
            release.inputs,
            score_spec=None,
        )

    local_rows = release.inputs[:2]
    t2_contract = build_run_dataset_contract(
        release,
        FakeSelectionSpec(FakeCapability.LOCALIZATION),
        local_rows,
        score_spec=None,
    )
    assert t2_contract.score_spec is None
    with pytest.raises(ContractError, match="must not declare"):
        build_run_dataset_contract(
            release,
            FakeSelectionSpec(FakeCapability.LOCALIZATION),
            local_rows,
            score_spec=_score_spec(),
        )


def test_dataset_contract_rejects_a_formal_selection_subset(tmp_path: Path):
    release = _release(tmp_path)
    with pytest.raises(
        ContractError,
        match="exactly materialize the selection spec",
    ):
        build_run_dataset_contract(
            release,
            FakeSelectionSpec(FakeCapability.WHOLE_IMAGE),
            [release.inputs[0]],
            score_spec=_score_spec(),
        )


def test_latest_attempt_is_last_wins_and_complete_retry_history_is_audited():
    rows = _rows()
    attempts = [
        _ok_attempt(rows[0]),
        _error_attempt(rows[1]),
        _ok_attempt(rows[1], 0.8),
        _ok_attempt(rows[2], 0.9),
    ]
    latest = index_latest_attempts(
        rows,
        attempts,
        run_id=RUN_ID,
        run_manifest_fingerprint=FINGERPRINT,
        score_spec=_score_spec(),
    )

    assert latest.physical_attempts == 4
    assert latest.superseded_attempts == 1
    assert latest.attempts_by_sample_id["sample-1"] == 2
    assert latest.latest_by_sample_id["sample-1"]["status"] == "ok"
    assert latest.pending_sample_ids() == ()

    coverage = summarize_coverage(latest)
    assert coverage.as_dict() == {
        "expected_images": 3,
        "physical_attempts": 4,
        "result_images": 3,
        "valid_images": 3,
        "error_images": 0,
        "missing_images": 0,
        "superseded_attempts": 1,
        "coverage_fraction": 1.0,
        "success_fraction": 1.0,
        "is_complete": True,
        "counts_by_condition": {
            "real": {
                "expected_images": 1,
                "result_images": 1,
                "valid_images": 1,
                "error_images": 0,
                "missing_images": 0,
            },
            "local_mouse": {
                "expected_images": 1,
                "result_images": 1,
                "valid_images": 1,
                "error_images": 0,
                "missing_images": 0,
            },
            "fullframe_cat": {
                "expected_images": 1,
                "result_images": 1,
                "valid_images": 1,
                "error_images": 0,
                "missing_images": 0,
            },
        },
    }
    require_complete_coverage(coverage)


def test_latest_error_overrides_success_and_coverage_fails_closed():
    rows = _rows()
    latest = index_latest_attempts(
        rows,
        [
            _ok_attempt(rows[0]),
            _ok_attempt(rows[1]),
            _error_attempt(rows[1]),
        ],
        run_id=RUN_ID,
        run_manifest_fingerprint=FINGERPRINT,
        score_spec=_score_spec(),
    )
    assert latest.pending_sample_ids() == ("sample-1", "sample-2")
    assert latest.pending_sample_ids(retry_errors=False) == ("sample-2",)

    coverage = summarize_coverage(latest)
    assert coverage.valid_images == 1
    assert coverage.error_images == 1
    assert coverage.missing_images == 1
    assert coverage.is_complete is False
    assert coverage.as_dict()["counts_by_condition"]["fullframe_cat"][
        "missing_images"
    ] == 1
    with pytest.raises(IncompleteCoverageError, match="incomplete Balanced250"):
        require_complete_coverage(coverage)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("id", "other", "id does not match"),
        ("rank", 99, "rank identity drifted"),
        ("dataset_id", "other-dataset", "dataset_id identity drifted"),
        ("condition", "local_cat", "condition identity drifted"),
        ("input_sha256", "9" * 64, "input_sha256 identity drifted"),
        ("run_id", "other-run", "run_id identity drifted"),
        (
            "run_manifest_fingerprint",
            "e" * 64,
            "run_manifest_fingerprint identity drifted",
        ),
    ],
)
def test_every_physical_attempt_rejects_identity_and_run_drift(
    field: str,
    value,
    message: str,
):
    rows = _rows()
    attempt = _ok_attempt(rows[0])
    attempt[field] = value
    with pytest.raises(ContractError, match=message):
        index_latest_attempts(
            rows,
            [attempt],
            run_id=RUN_ID,
            run_manifest_fingerprint=FINGERPRINT,
            score_spec=_score_spec(),
        )


def test_attempt_index_rejects_pair_rank_foreign_ids_and_invalid_scores():
    rows = _rows()

    with_pair = {**_ok_attempt(rows[0]), "pair_rank": 0}
    with pytest.raises(ContractError, match="must not contain pair_rank"):
        index_latest_attempts(
            rows,
            [with_pair],
            run_id=RUN_ID,
            run_manifest_fingerprint=FINGERPRINT,
            score_spec=_score_spec(),
        )

    foreign = _ok_attempt(rows[0])
    foreign["sample_id"] = "foreign"
    foreign["id"] = "foreign"
    with pytest.raises(ContractError, match="unexpected sample_id"):
        index_latest_attempts(
            rows,
            [foreign],
            run_id=RUN_ID,
            run_manifest_fingerprint=FINGERPRINT,
            score_spec=_score_spec(),
        )

    missing_score = _ok_attempt(rows[0])
    missing_score.pop("ai_score")
    with pytest.raises(ContractError, match="ai_score"):
        index_latest_attempts(
            rows,
            [missing_score],
            run_id=RUN_ID,
            run_manifest_fingerprint=FINGERPRINT,
            score_spec=_score_spec(),
        )

    nonfinite = _ok_attempt(rows[0], math.inf)
    with pytest.raises(ContractError, match="finite"):
        index_latest_attempts(
            rows,
            [nonfinite],
            run_id=RUN_ID,
            run_manifest_fingerprint=FINGERPRINT,
            score_spec=_score_spec(),
        )

    invalid_validity = _ok_attempt(rows[0])
    invalid_validity["valid_for_metrics"] = False
    with pytest.raises(
        ContractError,
        match="valid_for_metrics/status mismatch",
    ):
        index_latest_attempts(
            rows,
            [invalid_validity],
            run_id=RUN_ID,
            run_manifest_fingerprint=FINGERPRINT,
            score_spec=_score_spec(),
        )


def test_selected_id_hash_is_order_sensitive_and_rejects_duplicates():
    first = selected_ids_sha256(["a", "b"])
    assert first == selected_ids_sha256(["a", "b"])
    assert first != selected_ids_sha256(["b", "a"])
    with pytest.raises(ContractError, match="duplicates"):
        selected_ids_sha256(["a", "a"])
