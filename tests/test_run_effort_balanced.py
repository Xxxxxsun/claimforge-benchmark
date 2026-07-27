from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from eval.opensource import run_effort_balanced as runner
from eval.opensource.canonical_release import (
    BALANCED_CONDITIONS,
    BALANCED_DATASET_ID,
    BALANCED_SCHEMA,
    CanonicalRelease,
    LedgerView,
    load_canonical_release,
)
from eval.opensource.common import read_jsonl


REPO_ROOT = Path(__file__).resolve().parents[1]
FORMAL_MANIFEST = Path("outputs/opensource/balanced250_v1/manifest.json")


@pytest.fixture(scope="module")
def formal_release() -> CanonicalRelease:
    return load_canonical_release(
        REPO_ROOT,
        FORMAL_MANIFEST,
        verify_files=False,
    )


def test_runner_root_and_source_provenance_are_frozen():
    assert runner.DEFAULT_RESULTS_DIR == runner.legacy.DEFAULT_RESULTS_DIR
    assert "eval/opensource/analyze_effort_run.py" in (runner.ADAPTER_SOURCE_PATHS)
    assert "eval/opensource/maskclip_metrics.py" not in (runner.ADAPTER_SOURCE_PATHS)
    contract = runner.adapter_source_contract(REPO_ROOT)
    assert tuple(contract) == runner.ADAPTER_SOURCE_PATHS
    for relative, binding in contract.items():
        path = REPO_ROOT / relative
        assert binding == {
            "path": relative,
            "bytes": path.stat().st_size,
            "sha256": runner.sha256_file(path),
        }


def test_formal_selection_is_the_exact_1775_score_cache(
    formal_release: CanonicalRelease,
):
    spec, selected = runner.select_mode_inputs(
        formal_release,
        mode="formal",
        per_condition_limit=None,
        sample_id=None,
    )
    assert spec.capability.value == "whole_image_t1"
    assert len(selected) == 1775
    assert Counter(row["condition"] for row in selected) == runner.FORMAL_COUNTS
    assert [row["sample_id"] for row in selected] == [
        row["sample_id"] for row in formal_release.inputs
    ]


def test_smoke_selection_is_panel_priority_and_condition_balanced(
    formal_release: CanonicalRelease,
):
    spec, selected = runner.select_mode_inputs(
        formal_release,
        mode="smoke",
        per_condition_limit=5,
        sample_id=None,
    )
    assert spec.per_condition_limit == 5
    assert len(selected) == 35
    assert Counter(row["condition"] for row in selected) == {
        condition: 5 for condition in BALANCED_CONDITIONS
    }
    assert all(row["panel"] is True for row in selected)
    panel_first_five = {
        condition: [
            row["sample_id"]
            for row in formal_release.panel
            if row["condition"] == condition
        ][:5]
        for condition in BALANCED_CONDITIONS
    }
    actual = {
        condition: [
            row["sample_id"] for row in selected if row["condition"] == condition
        ]
        for condition in BALANCED_CONDITIONS
    }
    assert actual == panel_first_five
    assert [row["rank"] for row in selected] == sorted(row["rank"] for row in selected)
    contract = runner.build_run_dataset_contract(
        formal_release,
        spec,
        selected,
        score_spec=runner.SCORE_SPEC,
    )
    assert contract.selection.selected_images == 35
    assert dict(contract.selection.counts_by_condition) == {
        condition: 5 for condition in BALANCED_CONDITIONS
    }


def test_single_selection_and_mode_arguments_are_strict(
    formal_release: CanonicalRelease,
):
    wanted = formal_release.inputs[-1]["sample_id"]
    spec, selected = runner.select_mode_inputs(
        formal_release,
        mode="single",
        per_condition_limit=None,
        sample_id=wanted,
    )
    assert spec.sample_id == wanted
    assert [row["sample_id"] for row in selected] == [wanted]

    with pytest.raises(ValueError, match="formal mode"):
        runner.select_mode_inputs(
            formal_release,
            mode="formal",
            per_condition_limit=1,
            sample_id=None,
        )
    with pytest.raises(ValueError, match="exactly 5"):
        runner.select_mode_inputs(
            formal_release,
            mode="smoke",
            per_condition_limit=0,
            sample_id=None,
        )
    with pytest.raises(ValueError, match="exactly 5"):
        runner.select_mode_inputs(
            formal_release,
            mode="smoke",
            per_condition_limit=4,
            sample_id=None,
        )
    with pytest.raises(ValueError, match="requires --sample-id"):
        runner.select_mode_inputs(
            formal_release,
            mode="single",
            per_condition_limit=None,
            sample_id=None,
        )


def test_v2_result_identity_has_balanced_fields_and_no_pair_rank(
    formal_release: CanonicalRelease,
):
    local = next(
        row for row in formal_release.inputs if row["condition"] == "local_cat"
    )
    identity = runner.result_identity(
        local,
        repo_root=REPO_ROOT,
        run_id="identity-test",
        run_manifest_fingerprint="f" * 64,
        valid_for_metrics=True,
    )
    for key in (
        "sample_id",
        "rank",
        "condition",
        "condition_family",
        "manipulation_scope",
        "normalized_task_id",
        "task_id",
        "kind",
        "label",
        "domain",
        "gt_mask_kind",
        "valid_for_metrics",
    ):
        assert key in identity
    assert identity["schema_version"] == "opensource_result_v2"
    assert identity["edit_visibility"] == "full"
    assert identity["edit_visible_gt_fraction"] == 1.0
    assert "pair_rank" not in identity

    fullframe = next(
        row for row in formal_release.inputs if row["condition"] == "fullframe_cat"
    )
    identity = runner.result_identity(
        fullframe,
        repo_root=REPO_ROOT,
        run_id="identity-test",
        run_manifest_fingerprint="f" * 64,
        valid_for_metrics=False,
    )
    assert identity["edit_visibility"] == "not_applicable"
    assert identity["edit_visible_gt_fraction"] is None
    assert identity["valid_for_metrics"] is False


def test_attempt_gate_recursively_rejects_t2_claims_but_allows_negations(
    formal_release: CanonicalRelease,
):
    row = formal_release.inputs[0]
    attempt = {
        **runner.result_identity(
            row,
            repo_root=REPO_ROOT,
            run_id="attempt-gate-test",
            run_manifest_fingerprint="e" * 64,
            valid_for_metrics=True,
        ),
        "status": "ok",
        "audit": {
            "valid_for_t2": False,
            "localization_output": None,
        },
    }
    runner._validate_runner_attempt(
        attempt,
        input_row=row,
        repo_root=REPO_ROOT,
        run_id="attempt-gate-test",
        run_manifest_fingerprint="e" * 64,
    )

    forbidden_claims = (
        {"t2": {"valid": True}},
        {"audit": {"localisation_heatmap": [0.1]}},
        {"audit": {"mask_path": "mask.npy"}},
        {"audit": [{"predicted_mask_sha256": "f" * 64}]},
        {"audit": {"pixel_metrics": {"pixel_auroc": 0.9}}},
        {"audit": {"joint_score": 0.8}},
    )
    for claim in forbidden_claims:
        contaminated = {**attempt, **claim}
        with pytest.raises(ValueError, match="T2 payload"):
            runner._validate_runner_attempt(
                contaminated,
                input_row=row,
                repo_root=REPO_ROOT,
                run_id="attempt-gate-test",
                run_manifest_fingerprint="e" * 64,
            )


def _minimal_row(tmp_path: Path) -> dict:
    return {
        "schema_version": BALANCED_SCHEMA,
        "dataset_id": BALANCED_DATASET_ID,
        "rank": 0,
        "sample_id": "a" * 24,
        "condition": "real",
        "condition_family": "real",
        "manipulation_scope": "authentic",
        "normalized_task_id": "fixture-task",
        "task_id": "fixture-task",
        "kind": "real",
        "label": 0,
        "domain": "lodging",
        "gt_mask_kind": "all_zero",
        "canonical_path": (
            "outputs/opensource/balanced250_v1/images/" f"{'a' * 24}.jpg"
        ),
        "canonical_sha256": "1" * 64,
        "width": 8,
        "height": 6,
        "panel": True,
    }


def _minimal_release(tmp_path: Path) -> CanonicalRelease:
    row = _minimal_row(tmp_path)
    panel_rows = (
        {
            "sample_id": row["sample_id"],
            "condition": row["condition"],
        },
    )
    ledger_records = {
        "inputs": {
            "path": "outputs/fixture/inputs.jsonl",
            "sha256": runner._rows_sha256((row,)),
            "rows": 1,
        },
        "panel": {
            "path": "outputs/fixture/panel.jsonl",
            "sha256": runner._rows_sha256(panel_rows),
            "rows": 1,
        },
        "source_pairs": {
            "path": "outputs/fixture/source_pairs.jsonl",
            "sha256": runner._rows_sha256(()),
            "rows": 0,
        },
    }

    def ledger(name: str) -> LedgerView:
        value = ledger_records[name]
        return LedgerView(
            name=name,
            path=REPO_ROOT / value["path"],
            sha256=value["sha256"],
            rows=value["rows"],
        )

    return CanonicalRelease(
        repo_root=REPO_ROOT,
        manifest_path=REPO_ROOT / "outputs/fixture/manifest.json",
        manifest_sha256="5" * 64,
        manifest={
            "schema_version": BALANCED_SCHEMA,
            "dataset_id": BALANCED_DATASET_ID,
            "contract_sha256": "6" * 64,
            "ledgers": ledger_records,
        },
        schema_version=BALANCED_SCHEMA,
        dataset_id=BALANCED_DATASET_ID,
        release_kind="balanced250",
        contract_sha256="6" * 64,
        inputs_ledger=ledger("inputs"),
        inputs=(row,),
        panel_ledger=ledger("panel"),
        panel=panel_rows,
        source_pairs_ledger=ledger("source_pairs"),
        source_pairs=(),
        legacy_pairs_ledger=None,
        legacy_pairs=(),
    )


def _scoring(score: float = 0.75) -> dict:
    margin = float(np.log(score / (1.0 - score)))
    return {
        "class_logits": [0.0, margin],
        "raw_logit_margin": margin,
        "fake_probability": score,
        "probability": score,
        "ai_score": score,
        "score": score,
        "score_semantics": runner.legacy.SCORE_SEMANTICS,
        "classification_decision": True,
        "classification_threshold": 0.5,
        "classification_threshold_operator": ">",
        "classification": {
            "decision": True,
            "threshold": 0.5,
            "operator": ">",
        },
        "t1": {
            "valid": True,
            "score": score,
            "decision": True,
        },
        "manual_replay": {
            "head_logits_exact": True,
            "softmax_dtype": "float32",
            "fake_class_index": 1,
        },
    }


def _patch_cpu_run(
    monkeypatch,
    release: CanonicalRelease,
    events: list[str],
    artifacts_root: Path,
):
    source = {"commit": runner.legacy.MODEL_SOURCE_COMMIT}
    assets = {
        "checkpoint": {
            "id": runner.legacy.CHECKPOINT["id"],
            "sha256": runner.legacy.CHECKPOINT["sha256"],
        },
    }
    model_audit = {"strict_load": True}
    golden = {"status": "passed", "kind": "fixture"}
    preflight = {
        "schema_version": "effort_balanced_cpu_preflight_v1",
        "status": "passed",
        "source": source,
        "assets": assets,
        "model_audit": model_audit,
        "runtime": {"device": "cpu"},
        "runtime_golden": golden,
        "accelerator_model_forwards": 0,
        "balanced250_model_scores_computed": 0,
        "cuda_initialized_before": False,
        "cuda_initialized_after": False,
    }
    monkeypatch.setattr(
        runner,
        "load_canonical_release",
        lambda *_args, **_kwargs: release,
    )
    monkeypatch.setattr(runner, "DEFAULT_RESULTS_DIR", artifacts_root.parent)
    monkeypatch.setattr(runner, "DEFAULT_ARTIFACTS_DIR", artifacts_root)

    def run_preflight(**_kwargs):
        events.append("cpu_preflight")
        return preflight

    def verify_assets(*_args, **_kwargs):
        events.append("verify_assets")
        return assets, object(), object()

    def configure_runtime(device_text: str):
        events.append(f"configure:{device_text}")
        return SimpleNamespace(type="cpu"), {
            "device": device_text,
            "seed": runner.legacy.MODEL_SEED,
        }

    monkeypatch.setattr(runner, "run_cpu_preflight", run_preflight)
    monkeypatch.setattr(runner.legacy, "verify_source", lambda _root: source)
    monkeypatch.setattr(runner.legacy, "verify_assets", verify_assets)
    monkeypatch.setattr(runner.legacy, "configure_runtime", configure_runtime)
    monkeypatch.setattr(
        runner.legacy,
        "_build_model",
        lambda *_args, **_kwargs: (SimpleNamespace(), model_audit),
    )
    monkeypatch.setattr(
        runner.legacy,
        "validate_runtime_golden",
        lambda *_args, **_kwargs: golden,
    )
    monkeypatch.setattr(
        runner,
        "adapter_source_contract",
        lambda _root: {
            path: {
                "path": path,
                "bytes": index + 1,
                "sha256": f"{index + 1:064x}",
            }
            for index, path in enumerate(runner.ADAPTER_SOURCE_PATHS)
        },
    )
    monkeypatch.setattr(
        runner.legacy,
        "preprocess_image",
        lambda *_args, **_kwargs: (
            np.zeros((3, 224, 224), dtype=np.float32),
            {
                "native_width": 8,
                "native_height": 6,
            },
        ),
    )
    monkeypatch.setattr(
        runner.legacy,
        "infer_one",
        lambda *_args, **_kwargs: (
            _scoring(),
            np.arange(
                runner.legacy.FEATURE_DIMENSION,
                dtype=np.float32,
            ),
            np.asarray([0.0, float(np.log(3.0))], dtype=np.float32),
            None,
            1.25,
        ),
    )


def _single_args(tmp_path: Path, *, resume: bool = False) -> list[str]:
    result = [
        "--repo-root",
        str(REPO_ROOT),
        "--mode",
        "single",
        "--sample-id",
        "a" * 24,
        "--run-id",
        "effort-balanced-single-test",
        "--results-dir",
        str(tmp_path),
        "--artifacts-dir",
        str(tmp_path / "artifact-output"),
        "--device",
        "cpu",
    ]
    if resume:
        result.append("--resume")
    return result


def test_cpu_single_run_writes_v2_contract_result_and_replay_artifact(
    tmp_path: Path,
    monkeypatch,
):
    release = _minimal_release(tmp_path)
    events: list[str] = []
    _patch_cpu_run(
        monkeypatch,
        release,
        events,
        tmp_path / "artifact-output",
    )

    assert runner.main(_single_args(tmp_path)) == 0
    assert events[:3] == ["cpu_preflight", "verify_assets", "configure:cpu"]

    run_dir = tmp_path / "effort-balanced-single-test"
    manifest = json.loads((run_dir / "manifest.json").read_text())
    summary = json.loads((run_dir / "summary.json").read_text())
    results = read_jsonl(run_dir / "results.jsonl")
    assert manifest["schema_version"] == runner.RUN_MANIFEST_SCHEMA
    assert manifest["status"] == "complete"
    assert manifest["fingerprint"] == runner._fingerprint(manifest["immutable"])
    assert (
        manifest["immutable"]["dataset_contract"]["selection"]["selected_images"] == 1
    )
    assert set(manifest["immutable"]["adapter_sources"]) == set(
        runner.ADAPTER_SOURCE_PATHS
    )
    preflight = manifest["immutable"]["cpu_preflight"]["report"]
    assert preflight["cuda_initialized_before"] is False
    assert preflight["cuda_initialized_after"] is False
    assert summary["summary_kind"] == "runtime_coverage_only"
    assert summary["scientific_metrics"] is None
    assert summary["coverage"]["is_complete"] is True

    assert len(results) == 1
    result = results[0]
    assert result["schema_version"] == runner.RESULT_SCHEMA_VERSION
    assert result["condition"] == "real"
    assert result["valid_for_metrics"] is True
    assert "pair_rank" not in result
    assert result["ai_score"] == 0.75
    artifact_path = Path(result["artifact_path"])
    assert artifact_path.is_file()
    assert result["artifact_bytes"] == artifact_path.stat().st_size == 4_640
    with np.load(artifact_path, allow_pickle=False) as payload:
        assert set(payload.files) == {"pooler_output", "class_logits"}
        assert payload["pooler_output"].shape == (runner.legacy.FEATURE_DIMENSION,)
        assert payload["pooler_output"].dtype == np.float32
        assert payload["class_logits"].shape == (runner.legacy.CLASS_COUNT,)
        assert payload["class_logits"].dtype == np.float32


def test_resume_is_append_safe_last_wins_and_replays_latest_artifact(
    tmp_path: Path,
    monkeypatch,
):
    release = _minimal_release(tmp_path)
    events: list[str] = []
    _patch_cpu_run(
        monkeypatch,
        release,
        events,
        tmp_path / "artifact-output",
    )
    assert runner.main(_single_args(tmp_path)) == 0

    replayed: list[str] = []

    def validate_resume(row, **_kwargs):
        replayed.append(str(row["sample_id"]))

    monkeypatch.setattr(runner.legacy, "_validate_resume_row", validate_resume)
    monkeypatch.setattr(
        runner.legacy,
        "infer_one",
        lambda *_args, **_kwargs: pytest.fail("resume must not re-run inference"),
    )
    assert runner.main(_single_args(tmp_path, resume=True)) == 0
    assert replayed == ["a" * 24]

    run_dir = tmp_path / "effort-balanced-single-test"
    assert len(read_jsonl(run_dir / "results.jsonl")) == 1
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["execution"]["resume_skips"] == 1
    assert manifest["execution"]["new_successes"] == 0
    assert manifest["execution"]["superseded_attempts"] == 0


def test_inference_error_is_recorded_with_invalid_metrics_and_incomplete_gate(
    tmp_path: Path,
    monkeypatch,
):
    release = _minimal_release(tmp_path)
    events: list[str] = []
    _patch_cpu_run(
        monkeypatch,
        release,
        events,
        tmp_path / "artifact-output",
    )
    monkeypatch.setattr(
        runner.legacy,
        "infer_one",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("inference failed")
        ),
    )

    assert runner.main(_single_args(tmp_path)) == 2
    run_dir = tmp_path / "effort-balanced-single-test"
    result = read_jsonl(run_dir / "results.jsonl")[0]
    summary = json.loads((run_dir / "summary.json").read_text())
    assert result["status"] == "error"
    assert result["valid_for_metrics"] is False
    assert "ai_score" not in result
    assert summary["status"] == "incomplete"
    assert summary["coverage"]["error_images"] == 1
    assert summary["coverage"]["is_complete"] is False


def test_preflight_mode_is_cpu_only_and_does_not_load_dataset(
    monkeypatch,
    capsys,
):
    calls: list[str] = []
    monkeypatch.setattr(
        runner,
        "run_cpu_preflight",
        lambda **_kwargs: calls.append("cpu") or {"status": "passed"},
    )
    monkeypatch.setattr(
        runner,
        "load_canonical_release",
        lambda *_args, **_kwargs: pytest.fail("preflight loaded dataset"),
    )
    assert runner.main(["--mode", "preflight"]) == 0
    assert calls == ["cpu"]
    assert '"status": "passed"' in capsys.readouterr().out

    with pytest.raises(ValueError, match="preflight accepts no"):
        runner.main(["--mode", "preflight", "--device", "cuda:0"])
