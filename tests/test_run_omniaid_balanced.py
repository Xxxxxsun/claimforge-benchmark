from __future__ import annotations

import json
import os
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from eval.opensource import run_omniaid_balanced as runner
from eval.opensource.canonical_release import (
    BALANCED_CONDITIONS,
    CanonicalRelease,
    load_canonical_release,
)
from eval.opensource.common import stable_json


REPO_ROOT = Path(__file__).resolve().parents[1]
FORMAL_MANIFEST = Path("outputs/opensource/balanced250_v1/manifest.json")


@pytest.fixture(scope="module")
def formal_release() -> CanonicalRelease:
    return load_canonical_release(
        REPO_ROOT,
        FORMAL_MANIFEST,
        verify_files=False,
    )


def _arrays(*, score_at_threshold: bool = True) -> dict[str, np.ndarray]:
    logits = (
        np.zeros(2, dtype=np.float32)
        if score_at_threshold
        else np.asarray([1.0, -1.0], dtype=np.float32)
    )
    return {
        "pooler_output": np.zeros(
            runner.legacy.FEATURE_DIMENSION,
            dtype=np.float32,
        ),
        "class_logits": logits,
        "routing_feature": np.zeros(
            runner.legacy.FEATURE_DIMENSION,
            dtype=np.float32,
        ),
        "semantic_top_k_indices": np.asarray([0, 1], dtype=np.int64),
        "semantic_top_k_gates": np.asarray(
            [0.5, 0.5],
            dtype=np.float32,
        ),
        "final_gates": np.asarray(
            [0.5, 0.5, 0.0, 0.0, 0.0, 1.0],
            dtype=np.float32,
        ),
    }


def _scoring(arrays: dict[str, np.ndarray]) -> dict:
    logits = arrays["class_logits"]
    indices = arrays["semantic_top_k_indices"]
    gates = arrays["semantic_top_k_gates"]
    final = arrays["final_gates"]
    shifted = np.exp(logits - np.max(logits), dtype=np.float32)
    probability = float(np.float32(shifted[1] / np.sum(shifted, dtype=np.float32)))
    decision = probability > runner.legacy.CLASSIFICATION_THRESHOLD
    return {
        "class_logits": [float(value) for value in logits.tolist()],
        "raw_logit_margin": float(np.float32(logits[1] - logits[0])),
        "fake_probability": probability,
        "probability": probability,
        "ai_score": probability,
        "score": probability,
        "score_semantics": runner.legacy.SCORE_SEMANTICS,
        "routing_mode": "Auto (Router)",
        "semantic_expert_names": [
            "Human",
            "Animal",
            "Object",
            "Scene",
            "Anime",
        ],
        "artifact_expert_name": "Artifact",
        "semantic_top_k_indices": [int(value) for value in indices.tolist()],
        "semantic_top_k_gates": [float(value) for value in gates.tolist()],
        "final_expert_gates": [float(value) for value in final.tolist()],
        "semantic_gate_sum": float(
            final[: runner.legacy.SEMANTIC_EXPERT_COUNT].sum(dtype=np.float32)
        ),
        "final_gate_sum": float(final.sum(dtype=np.float32)),
        "classification_decision": decision,
        "classification_threshold": (runner.legacy.CLASSIFICATION_THRESHOLD),
        "classification_threshold_operator": (
            runner.legacy.CLASSIFICATION_THRESHOLD_OPERATOR
        ),
        "classification": {
            "decision": decision,
            "threshold": runner.legacy.CLASSIFICATION_THRESHOLD,
            "operator": runner.legacy.CLASSIFICATION_THRESHOLD_OPERATOR,
        },
        "t1": {
            "valid": True,
            "score": probability,
            "decision": decision,
        },
        "manual_replay": {
            "head_logits_exact": True,
            "softmax_dtype": "float32",
            "fake_class_index": 1,
            "router_scatter_exact": True,
        },
    }


def test_frozen_ids_task_and_artifact_contracts():
    assert runner.DEFAULT_FORMAL_RUN_ID == (
        "omniaid_dino_v2_mirage_auto_balanced250_v1_" "full1775_20260727"
    )
    assert runner.DEFAULT_SMOKE_RUN_ID_A.endswith("smoke5x7_a_20260727")
    assert runner.DEFAULT_SMOKE_RUN_ID_B.endswith("smoke5x7_b_20260727")
    assert runner.TASK_SCOPE["valid_for_t1"] is True
    assert runner.TASK_SCOPE["valid_for_t2"] is False
    assert runner.TASK_SCOPE["localization_output"] is None
    assert runner.ARTIFACT_FILE_BYTES == 9848
    assert tuple(runner.ARTIFACT_CONTRACT["keys"]) == tuple(
        runner.legacy.ARTIFACT_SCHEMA
    )
    assert runner.SCORE_SPEC.as_dict() == {
        "key": "ai_score",
        "direction": "higher_means_fake",
        "fixed_threshold": 0.5,
        "threshold_operator": ">",
    }
    assert runner.MODEL_CONTRACT["license"]["commercial_clearance"] is False
    assert "eval/opensource/analyze_omniaid_balanced.py" in (
        runner.ADAPTER_SOURCE_PATHS
    )


def test_formal_selection_is_exact_1775_and_frozen(
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
    assert Counter(row["condition"] for row in selected) == (runner.FORMAL_COUNTS)
    assert runner._rows_sha256(selected) == (runner.FORMAL_SELECTED_ROWS_SHA256)
    assert (
        runner.selected_ids_sha256(row["sample_id"] for row in selected)
        == runner.FORMAL_SELECTED_IDS_SHA256
    )
    assert all("pair_rank" not in row for row in selected)
    assert runner.visibility_census(selected) == {
        "full": 750,
        "not_applicable": 1025,
    }


def test_smoke_selection_is_exact_panel_balanced_and_frozen(
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
    assert (
        runner.selected_ids_sha256(row["sample_id"] for row in selected)
        == runner.SMOKE5X7_SELECTED_IDS_SHA256
    )
    for invalid_limit in (1, 4, 6, 250, True):
        with pytest.raises(ValueError, match="exactly 5"):
            runner.select_mode_inputs(
                formal_release,
                mode="smoke",
                per_condition_limit=invalid_limit,
                sample_id=None,
            )


def test_single_selection_and_run_ids_are_strict(
    formal_release: CanonicalRelease,
):
    sample_id = formal_release.inputs[-1]["sample_id"]
    spec, selected = runner.select_mode_inputs(
        formal_release,
        mode="single",
        per_condition_limit=None,
        sample_id=sample_id,
    )
    assert spec.sample_id == sample_id
    assert [row["sample_id"] for row in selected] == [sample_id]
    parser = runner._build_parser()
    assert (
        runner._resolve_run_id(parser.parse_args(["--mode", "formal"]))
        == runner.DEFAULT_FORMAL_RUN_ID
    )
    assert (
        runner._resolve_run_id(
            parser.parse_args(["--mode", "smoke", "--smoke-replicate", "a"])
        )
        == runner.DEFAULT_SMOKE_RUN_ID_A
    )
    assert (
        runner._resolve_run_id(
            parser.parse_args(["--mode", "smoke", "--smoke-replicate", "b"])
        )
        == runner.DEFAULT_SMOKE_RUN_ID_B
    )
    with pytest.raises(ValueError, match="requires --smoke-replicate"):
        runner._resolve_run_id(parser.parse_args(["--mode", "smoke"]))
    with pytest.raises(ValueError, match="explicit --run-id"):
        runner._resolve_run_id(
            parser.parse_args(["--mode", "single", "--sample-id", sample_id])
        )


def test_balanced_identity_visibility_and_no_pair_rank(
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
    assert identity["schema_version"] == runner.RESULT_SCHEMA_VERSION
    assert identity["edit_visibility"] == "full"
    assert identity["edit_visible_gt_fraction"] == 1.0
    assert identity["task_scope"]["valid_for_t2"] is False
    assert identity["score_semantics"] == runner.legacy.SCORE_SEMANTICS
    assert "pair_rank" not in identity

    for condition, expected_basis in (
        ("real", "authentic_input_has_all_zero_GT"),
        (
            "fullframe_cat",
            "conditional_full_frame_edit_has_no_local_GT",
        ),
    ):
        row = next(
            item for item in formal_release.inputs if item["condition"] == condition
        )
        visibility = runner.visibility_diagnostic(row)
        assert visibility["edit_visibility"] == "not_applicable"
        assert visibility["edit_visible_gt_fraction"] is None
        assert visibility["edit_visibility_evidence"]["basis"] == expected_basis


def test_recursive_t2_guard_allows_only_explicit_negations():
    allowed = {
        "task_scope": {
            "valid_for_t2": False,
            "localization_output": None,
            "native_dense_output": False,
        },
        "gt_mask_kind": "exact_diff",
    }
    assert runner.forbidden_t2_claims(allowed) == set()
    forbidden = (
        {"valid_for_t2": True},
        {"localization_output": "heatmap.npy"},
        {"native_dense_output": True},
        {"nested": {"predicted_mask": [[1]]}},
        {"nested": [{"pixel_metrics": {"pixel_auroc": 0.9}}]},
        {"joint_score": 0.8},
        {"t2": {"valid": True}},
    )
    for value in forbidden:
        assert runner.forbidden_t2_claims(value)


def test_artifact_is_canonical_deterministic_and_round_trips(tmp_path: Path):
    arrays = _arrays()
    root = tmp_path / "omniaid-run"
    first = runner.artifact_path(root, "a" * 24)
    second = runner.artifact_path(root, "b" * 24)
    runner.persist_artifact(first, arrays)
    runner.persist_artifact(second, arrays)
    assert first.read_bytes() == second.read_bytes()
    assert first.stat().st_size == runner.ARTIFACT_FILE_BYTES
    loaded, record = runner.validate_artifact(
        first,
        repo_root=tmp_path,
    )
    assert record["artifact_bytes"] == 9848
    assert record["artifact_keys"] == list(runner.legacy.ARTIFACT_SCHEMA)
    assert record["artifact_path"].endswith(f"artifacts/{'a' * 24}.npz")
    for key in arrays:
        assert np.array_equal(loaded[key], arrays[key])


def test_artifact_rejects_tamper_extra_inventory_and_symlink(
    tmp_path: Path,
):
    arrays = _arrays()
    root = tmp_path / "omniaid-run"
    path = runner.artifact_path(root, "a" * 24)
    runner.persist_artifact(path, arrays)
    payload = bytearray(path.read_bytes())
    payload[-1] ^= 1
    path.write_bytes(payload)
    with pytest.raises(ValueError):
        runner.validate_artifact(path, repo_root=tmp_path)

    runner.persist_artifact(path, arrays)
    extra = path.parent / "extra.npz"
    extra.write_bytes(path.read_bytes())
    with pytest.raises(ValueError, match="inventory mismatch"):
        runner.validate_artifact_inventory(
            artifact_root=root,
            latest_by_sample_id={},
            repo_root=tmp_path,
        )
    extra.unlink()
    path.unlink()
    target = path.parent / "target.npz"
    target.write_bytes(runner._npz_bytes(arrays))
    path.symlink_to(target)
    with pytest.raises(FileNotFoundError):
        runner.validate_artifact(path, repo_root=tmp_path)


def test_score_contract_preserves_strict_threshold_and_router():
    arrays = _arrays()
    score = _scoring(arrays)
    assert score["ai_score"] == 0.5
    assert score["classification_decision"] is False
    runner._validate_score_payload(score, arrays=arrays)
    contaminated = dict(score)
    contaminated["classification_decision"] = True
    with pytest.raises(ValueError, match="semantics"):
        runner._validate_score_payload(contaminated, arrays=arrays)
    contaminated = dict(score)
    contaminated["t2"] = {"valid": True}
    with pytest.raises(ValueError, match="T2"):
        runner._validate_score_payload(contaminated, arrays=arrays)

    arrays = _arrays()
    arrays["semantic_top_k_indices"] = np.asarray([0, 2], dtype=np.int64)
    arrays["semantic_top_k_gates"] = np.asarray(
        [0.4042757749557495, 0.5957242846488953],
        dtype=np.float32,
    )
    arrays["final_gates"] = np.asarray(
        [0.4042757749557495, 0.0, 0.5957242846488953, 0.0, 0.0, 1.0],
        dtype=np.float32,
    )
    official_style = _scoring(arrays)
    assert official_style["semantic_gate_sum"] == 1.0
    runner._validate_score_payload(official_style, arrays=arrays)


def test_cpu_ok_result_preserves_legacy_api_and_local_artifact(
    tmp_path: Path,
    formal_release: CanonicalRelease,
):
    import torch

    assert torch.cuda.is_initialized() is False
    row = formal_release.inputs[0]
    arrays = _arrays()
    image, preprocess = runner.legacy.preprocess_image(
        REPO_ROOT / row["canonical_path"]
    )
    assert image.shape == (3, 448, 448)

    head = torch.nn.Linear(runner.legacy.FEATURE_DIMENSION, 2)
    with torch.no_grad():
        head.weight.zero_()
        head.bias.zero_()

    class Router(torch.nn.Module):
        def forward(self, routing):
            count = routing.shape[0]
            return {
                "top_k_indices": torch.tensor(
                    [[0, 1]],
                    dtype=torch.int64,
                    device=routing.device,
                ).repeat(count, 1),
                "top_k_gates": torch.tensor(
                    [[0.5, 0.5]],
                    dtype=torch.float32,
                    device=routing.device,
                ).repeat(count, 1),
            }

    model = SimpleNamespace(head=head, gating_network=Router())
    result = runner._build_ok_result(
        input_row=row,
        repo_root=REPO_ROOT,
        artifact_root=tmp_path / "local-artifacts",
        run_id="cpu-ok-test",
        fingerprint="d" * 64,
        model=model,
        device=torch.device("cpu"),
        scoring=_scoring(arrays),
        arrays=arrays,
        preprocess=preprocess,
        preprocess_latency_ms=0.0,
        latency_ms=0.0,
        peak_cuda_memory_bytes=None,
    )
    assert result["status"] == "ok"
    assert result["ai_score"] == result["fake_probability"] == 0.5
    assert result["classification_decision"] is False
    assert result["artifact_bytes"] == 9848
    assert result["feature_semantics"] == runner.legacy.FEATURE_SEMANTICS
    assert result["routing_feature_semantics"] == (
        runner.legacy.ROUTING_FEATURE_SEMANTICS
    )
    assert "pair_rank" not in result
    assert runner.forbidden_t2_claims(result) == set()
    assert torch.cuda.is_initialized() is False


def test_error_result_nulls_every_score_and_has_no_t2(
    tmp_path: Path,
    formal_release: CanonicalRelease,
):
    row = formal_release.inputs[0]
    result = runner._build_error_result(
        input_row=row,
        repo_root=REPO_ROOT,
        artifact_root=tmp_path / "artifacts",
        run_id="error-test",
        fingerprint="e" * 64,
        error=RuntimeError("fixture failure"),
    )
    assert result["status"] == "error"
    assert result["valid_for_metrics"] is False
    for key in (
        "class_logits",
        "raw_logit_margin",
        "fake_probability",
        "probability",
        "ai_score",
        "score",
        "classification_decision",
        "peak_cuda_memory_bytes",
    ):
        assert result[key] is None
    assert result["latency_ms"] == 0.0
    assert runner.forbidden_t2_claims(result) == set()


def test_history_allows_error_retries_but_rejects_after_success(
    monkeypatch: pytest.MonkeyPatch,
    formal_release: CanonicalRelease,
    tmp_path: Path,
):
    row = formal_release.inputs[0]
    common = runner.build_result_identity(
        row,
        run_id="history-test",
        run_manifest_fingerprint="c" * 64,
    )
    error = {
        **common,
        "status": "error",
        "valid_for_metrics": False,
        "ai_score": None,
    }
    success = {
        **common,
        "status": "ok",
        "valid_for_metrics": True,
        "ai_score": 0.5,
    }
    monkeypatch.setattr(
        runner,
        "_validate_runner_attempt",
        lambda *args, **kwargs: None,
    )
    runner.validate_attempt_history(
        selected=[row],
        attempts=[error, error],
        repo_root=REPO_ROOT,
        artifact_root=tmp_path,
        run_id="history-test",
        run_manifest_fingerprint="c" * 64,
    )
    with pytest.raises(ValueError, match="after success"):
        runner.validate_attempt_history(
            selected=[row],
            attempts=[success, error],
            repo_root=REPO_ROOT,
            artifact_root=tmp_path,
            run_id="history-test",
            run_manifest_fingerprint="c" * 64,
        )
    with pytest.raises(ValueError, match="after success"):
        runner.validate_attempt_history(
            selected=[row],
            attempts=[success, success],
            repo_root=REPO_ROOT,
            artifact_root=tmp_path,
            run_id="history-test",
            run_manifest_fingerprint="c" * 64,
        )


def test_strict_json_readers_reject_duplicate_nonfinite_and_noncanonical(
    tmp_path: Path,
):
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"a":1,"a":2}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        runner._load_json_strict(duplicate, "duplicate")
    nonfinite = tmp_path / "nonfinite.json"
    nonfinite.write_text('{"a": NaN}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="non-finite"):
        runner._load_json_strict(nonfinite, "nonfinite")
    noncanonical = tmp_path / "rows.jsonl"
    noncanonical.write_text('{"b": 2, "a": 1}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="canonical"):
        runner._read_jsonl_strict(noncanonical, "rows")
    canonical = tmp_path / "canonical.jsonl"
    row = {"b": 2, "a": 1}
    canonical.write_text(f"{stable_json(row)}\n", encoding="utf-8")
    assert runner._read_jsonl_strict(canonical, "canonical") == [row]


def test_artifact_policy_is_gitignored_and_path_is_exact():
    root = REPO_ROOT / "outputs/opensource/omniaid" / runner.DEFAULT_FORMAL_RUN_ID
    policy = runner.artifact_policy_contract(
        repo_root=REPO_ROOT,
        artifact_root=root,
    )
    assert policy["storage"] == "local_only"
    assert policy["gitignored"] is True
    assert policy["artifact_root"] == (
        f"outputs/opensource/omniaid/{runner.DEFAULT_FORMAL_RUN_ID}"
    )
    assert runner.artifact_path(root, "f" * 24) == (
        root / "artifacts" / f"{'f' * 24}.npz"
    )


def test_preflight_mode_never_loads_dataset_or_configures_accelerator(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    calls: list[str] = []

    def fake_preflight(**kwargs):
        calls.append("cpu_preflight")
        return {"status": "passed", "cuda_used": False}

    monkeypatch.setattr(runner, "run_cpu_preflight", fake_preflight)
    monkeypatch.setattr(
        runner,
        "load_canonical_release",
        lambda *args, **kwargs: pytest.fail("dataset was loaded"),
    )
    monkeypatch.setattr(
        runner,
        "configure_runtime",
        lambda *args, **kwargs: pytest.fail("accelerator was configured"),
    )
    args = runner._build_parser().parse_args(["--mode", "preflight", "--device", "cpu"])
    assert runner.run(args) == 0
    assert calls == ["cpu_preflight"]
    assert json.loads(capsys.readouterr().out)["cuda_used"] is False


def test_runtime_contract_fails_closed_on_malformed_python_record():
    with pytest.raises(ValueError, match="key set"):
        runner.validate_runtime_contract({"device": "cpu"})


_HEAVY_CPU_PREFLIGHT = (
    os.environ.get("CLAIMFORGE_OMNIAID_CPU_PREFLIGHT_TEST") == "1"
    and Path(runner.DEFAULT_CHECKPOINT).is_file()
    and Path(os.path.abspath(sys.executable))
    == Path(os.path.abspath(runner.FROZEN_PYTHON_EXECUTABLE))
)


@pytest.mark.skipif(
    not _HEAVY_CPU_PREFLIGHT,
    reason=(
        "set CLAIMFORGE_OMNIAID_CPU_PREFLIGHT_TEST=1 and use the frozen "
        "OmniAID venv to execute the 3.2GB CPU golden"
    ),
)
def test_real_cpu_preflight_is_byte_exact_and_never_initializes_cuda(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    import torch

    empty_prefix = tmp_path / "empty-pycache"
    empty_prefix.mkdir()
    monkeypatch.setattr(
        runner,
        "FROZEN_PYTHONPYCACHEPREFIX",
        empty_prefix,
    )
    monkeypatch.setenv("PYTHONHASHSEED", runner.FROZEN_PYTHONHASHSEED)
    monkeypatch.setenv("PYTHONDONTWRITEBYTECODE", "1")
    monkeypatch.setenv("NO_ALBUMENTATIONS_UPDATE", "1")
    monkeypatch.setenv("PYTHONPYCACHEPREFIX", str(empty_prefix))
    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    monkeypatch.setattr(sys, "pycache_prefix", str(empty_prefix))
    assert torch.cuda.is_initialized() is False
    report = runner.run_cpu_preflight(
        repo_root=REPO_ROOT,
        source_root=Path(runner.DEFAULT_SOURCE_ROOT),
        space_root=Path(runner.DEFAULT_SPACE_ROOT),
        checkpoint_path=Path(runner.DEFAULT_CHECKPOINT),
        omniaid_config_path=Path(runner.DEFAULT_OMNIAID_CONFIG),
    )
    assert report["status"] == "passed"
    assert report["cuda_used"] is False
    assert report["cuda_tensor_operations"] is False
    assert report["dataset_manifest_loaded"] is False
    assert report["balanced_golden"]["artifact"] == {
        "artifact_bytes": 9848,
        "artifact_sha256": runner.CPU_GOLDEN_ARTIFACT_SHA256,
        "array_sha256": runner.CPU_GOLDEN_ARRAY_SHA256,
    }
    assert report["balanced_golden"]["repeat_byte_exact"] is True
    runner._validate_preflight_report(
        report,
        source=report["source"],
        assets=report["assets"],
    )
    assert torch.cuda.is_initialized() is False
