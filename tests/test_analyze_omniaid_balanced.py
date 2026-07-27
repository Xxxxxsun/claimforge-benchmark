from __future__ import annotations

import inspect
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from eval.opensource import analyze_omniaid_balanced as analyzer
from eval.opensource import run_omniaid_balanced as runner
from eval.opensource.balanced_run_contract import (
    ScoreSpec,
    build_result_identity,
    build_run_dataset_contract,
)
from eval.opensource.canonical_release import load_canonical_release
from eval.opensource.common import sha256_file, stable_json


REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_MANIFEST = REPO_ROOT / "outputs/opensource/balanced250_v1/manifest.json"


@pytest.fixture(scope="module")
def release() -> Any:
    return load_canonical_release(
        REPO_ROOT,
        DATASET_MANIFEST,
        verify_files=False,
    )


def _arrays(
    *,
    official_style_gates: bool = False,
) -> dict[str, np.ndarray]:
    if official_style_gates:
        indices = np.asarray([0, 2], dtype=np.int64)
        gates = np.asarray(
            [0.4042757749557495, 0.5957242846488953],
            dtype=np.float32,
        )
        final = np.asarray(
            [
                0.4042757749557495,
                0.0,
                0.5957242846488953,
                0.0,
                0.0,
                1.0,
            ],
            dtype=np.float32,
        )
    else:
        indices = np.asarray([0, 1], dtype=np.int64)
        gates = np.asarray([0.5, 0.5], dtype=np.float32)
        final = np.asarray(
            [0.5, 0.5, 0.0, 0.0, 0.0, 1.0],
            dtype=np.float32,
        )
    return {
        "pooler_output": np.zeros(
            analyzer.legacy.FEATURE_DIMENSION,
            dtype=np.float32,
        ),
        "class_logits": np.zeros(2, dtype=np.float32),
        "routing_feature": np.zeros(
            analyzer.legacy.FEATURE_DIMENSION,
            dtype=np.float32,
        ),
        "semantic_top_k_indices": indices,
        "semantic_top_k_gates": gates,
        "final_gates": final,
    }


def _score_fields(arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    logits = arrays["class_logits"]
    score = 0.5
    decision = False
    indices = arrays["semantic_top_k_indices"]
    gates = arrays["semantic_top_k_gates"]
    final = arrays["final_gates"]
    return {
        "class_logits": [float(value) for value in logits.tolist()],
        "raw_logit_margin": float(np.float32(logits[1] - logits[0])),
        "fake_probability": score,
        "probability": score,
        "ai_score": score,
        "score": score,
        "score_semantics": analyzer.legacy.SCORE_SEMANTICS,
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
        "semantic_gate_sum": float(final[:5].sum(dtype=np.float32)),
        "final_gate_sum": float(final.sum(dtype=np.float32)),
        "classification_decision": decision,
        "classification_threshold": 0.5,
        "classification_threshold_operator": ">",
        "classification": {
            "decision": decision,
            "threshold": 0.5,
            "operator": ">",
        },
        "t1": {
            "valid": True,
            "score": score,
            "decision": decision,
        },
        "manual_replay": {
            "head_logits_exact": True,
            "softmax_dtype": "float32",
            "fake_class_index": 1,
            "router_scatter_exact": True,
        },
    }


def _artifact_row(
    repo_root: Path,
    *,
    run_id: str,
    sample_id: str,
    arrays: dict[str, np.ndarray] | None = None,
) -> tuple[dict[str, Any], analyzer.OmniAIDArtifact]:
    values = _arrays() if arrays is None else arrays
    artifact_dir = repo_root / analyzer.DEFAULT_ARTIFACTS_DIR / run_id / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    path = artifact_dir / f"{sample_id}.npz"
    path.write_bytes(analyzer._npz_bytes(values))
    relative = path.relative_to(repo_root).as_posix()
    hashes = {key: analyzer._array_sha256(value) for key, value in values.items()}
    row = {
        "sample_id": sample_id,
        "status": "ok",
        "run_id": run_id,
        "run_manifest_fingerprint": "a" * 64,
        "config_fingerprint": "a" * 64,
        "completed_at": "2026-07-27T00:00:00+00:00",
        "preprocess_latency_ms": 1.0,
        "latency_ms": 2.0,
        "peak_cuda_memory_bytes": None,
        "artifact_path": relative,
        "artifact_sha256": sha256_file(path),
        "artifact_bytes": analyzer.ARTIFACT_FILE_BYTES,
        "artifact_keys": list(analyzer.ARTIFACT_SCHEMA),
        "artifact_paths": {"omniaid_npz": relative},
        "artifact_array_sha256": hashes,
        "feature_shape": [analyzer.legacy.FEATURE_DIMENSION],
        "feature_dtype": "float32",
        "feature_semantics": analyzer.legacy.FEATURE_SEMANTICS,
        "feature_array_sha256": hashes["pooler_output"],
        "class_logits_shape": [2],
        "class_logits_dtype": "float32",
        "class_logits_array_sha256": hashes["class_logits"],
        "routing_feature_shape": [analyzer.legacy.FEATURE_DIMENSION],
        "routing_feature_dtype": "float32",
        "routing_feature_semantics": (analyzer.legacy.ROUTING_FEATURE_SEMANTICS),
        "semantic_top_k_indices_shape": [2],
        "semantic_top_k_indices_dtype": "int64",
        "semantic_top_k_gates_shape": [2],
        "semantic_top_k_gates_dtype": "float32",
        "final_gates_shape": [6],
        "final_gates_dtype": "float32",
        **_score_fields(values),
    }
    artifact = analyzer._load_npz_artifact(
        row=row,
        sample_id=sample_id,
        repo_root=repo_root,
        artifact_dir=artifact_dir,
    )
    return row, artifact


def test_analyzer_contract_is_bound_to_frozen_runner_and_legacy_path():
    assert analyzer._assert_runner_contract_exports() is runner
    assert analyzer.AUDIT_SCHEMA_VERSION == ("omniaid_balanced_replay_audit_v2")
    assert analyzer.SMOKE_COMPARISON_SCHEMA_VERSION == (
        "omniaid_balanced_smoke_comparison_v2"
    )
    assert analyzer.METRICS_SCHEMA_VERSION == ("balanced250_t1_summary_v1")
    assert analyzer.ARTIFACT_FILE_BYTES == 9848
    assert tuple(analyzer.ARTIFACT_SCHEMA) == tuple(runner.legacy.ARTIFACT_SCHEMA)
    assert analyzer.legacy_audit.__name__.endswith("analyze_omniaid_run")
    assert inspect.signature(analyzer.analyze).parameters["replay"].default is True


def test_formal_and_smoke_selections_are_independently_frozen(release):
    formal_spec, formal = analyzer._formal_selection(release)
    smoke_spec, smoke = analyzer._smoke_selection(release)
    assert formal_spec.capability.value == "whole_image_t1"
    assert len(formal) == 1775
    assert analyzer._rows_sha256(formal) == (analyzer.FORMAL_SELECTED_ROWS_SHA256)
    assert (
        analyzer.selected_ids_sha256(row["sample_id"] for row in formal)
        == analyzer.FORMAL_SELECTED_IDS_SHA256
    )
    assert smoke_spec.per_condition_limit == 5
    assert len(smoke) == 35
    assert (
        analyzer.selected_ids_sha256(row["sample_id"] for row in smoke)
        == analyzer.SMOKE5X7_SELECTED_IDS_SHA256
    )
    assert all("pair_rank" not in row for row in formal + smoke)


def test_visibility_and_t2_boundary_are_independent(release):
    _, selected = analyzer._formal_selection(release)
    assert analyzer._visibility_census(selected) == {
        "full": 750,
        "not_applicable": 1025,
    }
    local = next(row for row in selected if row["condition"] == "local_cat")
    fullframe = next(row for row in selected if row["condition"] == "fullframe_cat")
    assert analyzer._visibility_diagnostic(local)["edit_visible_gt_fraction"] == 1.0
    assert (
        analyzer._visibility_diagnostic(fullframe)["edit_visible_gt_fraction"] is None
    )
    allowed = {
        "valid_for_t2": False,
        "localization_output": None,
        "native_dense_output": False,
        "gt_mask_kind": "exact_diff",
    }
    assert analyzer.forbidden_t2_claims(allowed) == set()
    for forbidden in (
        {"valid_for_t2": True},
        {"localization_output": "map.npy"},
        {"predicted_mask": [[1]]},
        {"nested": {"mask_path": "mask.npy"}},
        {"nested": [{"heatmap_sha256": "a" * 64}]},
        {"pixel_metrics": {"iou": 0.5}},
        {"joint_score": 0.5},
        {"pair_rank": 1},
        {"t2": True},
    ):
        assert analyzer.forbidden_t2_claims(forbidden)


def test_load_run_rejects_symlinked_run_directory(tmp_path: Path):
    repo_root = tmp_path / "repo"
    results_root = repo_root / analyzer.DEFAULT_RESULTS_DIR
    artifacts_root = repo_root / analyzer.DEFAULT_ARTIFACTS_DIR
    target = results_root / "actual"
    target.mkdir(parents=True)
    artifacts_root.mkdir(parents=True)
    (results_root / analyzer.DEFAULT_FORMAL_RUN_ID).symlink_to(
        target,
        target_is_directory=True,
    )
    with pytest.raises(ValueError, match="missing or unsafe"):
        analyzer._load_run(
            repo_root=repo_root,
            results_dir=results_root,
            artifacts_dir=artifacts_root,
            run_id=analyzer.DEFAULT_FORMAL_RUN_ID,
            mode="formal",
        )


def test_score_contract_is_float32_strict_at_threshold():
    arrays = _arrays(official_style_gates=True)
    row = _score_fields(arrays)
    evidence = analyzer._validate_score_payload(
        row,
        sample_id="fixture",
        arrays=arrays,
    )
    assert evidence["score"] == 0.5
    assert evidence["decision"] is False
    assert row["semantic_gate_sum"] == 1.0
    changed = dict(row)
    changed["classification_decision"] = True
    with pytest.raises(ValueError, match="classification_decision"):
        analyzer._validate_score_payload(
            changed,
            sample_id="fixture",
            arrays=arrays,
        )


def test_artifact_is_exact_canonical_9848_byte_six_array_npz(
    tmp_path: Path,
):
    root = tmp_path / "repo"
    root.mkdir()
    row, artifact = _artifact_row(
        root,
        run_id="run-a",
        sample_id="a" * 24,
    )
    assert artifact.file_bytes == 9848
    assert artifact.path.stat().st_size == 9848
    assert list(artifact.arrays) == list(analyzer.ARTIFACT_SCHEMA)
    assert artifact.path.read_bytes() == analyzer._npz_bytes(artifact.arrays)
    assert row["artifact_array_sha256"] == dict(artifact.array_sha256)


def test_artifact_rejects_tamper_extra_and_symlink(tmp_path: Path):
    root = tmp_path / "repo"
    root.mkdir()
    row, artifact = _artifact_row(
        root,
        run_id="run-a",
        sample_id="b" * 24,
    )
    payload = bytearray(artifact.path.read_bytes())
    payload[-1] ^= 1
    artifact.path.write_bytes(payload)
    with pytest.raises(ValueError):
        analyzer._load_npz_artifact(
            row=row,
            sample_id="b" * 24,
            repo_root=root,
            artifact_dir=artifact.path.parent,
        )

    row, artifact = _artifact_row(
        root,
        run_id="run-b",
        sample_id="c" * 24,
    )
    extra = artifact.path.parent / "extra.npz"
    extra.write_bytes(artifact.path.read_bytes())
    with pytest.raises(ValueError, match="coverage"):
        analyzer.validate_artifact_inventory(
            latest_results=[row],
            repo_root=root,
            artifact_root=artifact.path.parent.parent,
            artifact_dir=artifact.path.parent,
        )
    extra.unlink()
    target = artifact.path.parent / "target.npz"
    artifact.path.replace(target)
    artifact.path.symlink_to(target)
    with pytest.raises(ValueError, match="symlink"):
        analyzer._load_npz_artifact(
            row=row,
            sample_id="c" * 24,
            repo_root=root,
            artifact_dir=artifact.path.parent,
        )


def test_artifact_inventory_requires_only_artifacts_directory(
    tmp_path: Path,
):
    root = tmp_path / "repo"
    root.mkdir()
    row, artifact = _artifact_row(
        root,
        run_id="inventory",
        sample_id="d" * 24,
    )
    loaded = analyzer.validate_artifact_inventory(
        latest_results=[row],
        repo_root=root,
        artifact_root=artifact.path.parent.parent,
        artifact_dir=artifact.path.parent,
    )
    assert set(loaded) == {"d" * 24}
    (artifact.path.parent.parent / "unexpected.txt").write_text(
        "x",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="extra entries"):
        analyzer.validate_artifact_inventory(
            latest_results=[row],
            repo_root=root,
            artifact_root=artifact.path.parent.parent,
            artifact_dir=artifact.path.parent,
        )


def test_smoke_immutable_projection_normalizes_run_specific_policy():
    left = {key: None for key in analyzer.EXPECTED_IMMUTABLE_KEYS}
    right = {key: None for key in analyzer.EXPECTED_IMMUTABLE_KEYS}
    left.update(
        {
            "run_id": analyzer.DEFAULT_SMOKE_RUN_ID_A,
            "outputs": {"artifact_root": "run-a"},
            "artifact_policy": {"artifact_root": "run-a", "gitignored": True},
        }
    )
    right.update(
        {
            "run_id": analyzer.DEFAULT_SMOKE_RUN_ID_B,
            "outputs": {"artifact_root": "run-b"},
            "artifact_policy": {"artifact_root": "run-b", "gitignored": True},
        }
    )
    assert stable_json(analyzer._smoke_immutable_projection(left)) == stable_json(
        analyzer._smoke_immutable_projection(right)
    )


def test_35_image_smoke_comparison_is_bit_exact(tmp_path: Path):
    root = tmp_path / "repo"
    root.mkdir()
    reference_rows = []
    replay_rows = []
    reference_artifacts = {}
    replay_artifacts = {}
    for index in range(analyzer.SMOKE_IMAGES):
        sample_id = f"{index:024x}"
        left, left_artifact = _artifact_row(
            root,
            run_id="smoke-a",
            sample_id=sample_id,
        )
        right, right_artifact = _artifact_row(
            root,
            run_id="smoke-b",
            sample_id=sample_id,
        )
        reference_rows.append(left)
        replay_rows.append(right)
        reference_artifacts[sample_id] = left_artifact
        replay_artifacts[sample_id] = right_artifact
    comparison = analyzer.compare_computational_results(
        reference_rows=reference_rows,
        replay_rows=replay_rows,
        reference_artifacts=reference_artifacts,
        replay_artifacts=replay_artifacts,
    )
    assert comparison["images_compared"] == 35
    assert comparison["npz_file_bytes_exact"] is True
    assert comparison["all_six_arrays_exact"] is True
    replay_rows[0] = dict(replay_rows[0])
    replay_rows[0]["ai_score"] = 0.6
    with pytest.raises(ValueError):
        analyzer.compare_computational_results(
            reference_rows=reference_rows,
            replay_rows=replay_rows,
            reference_artifacts=reference_artifacts,
            replay_artifacts=replay_artifacts,
        )


def test_persisted_head_softmax_router_replays_on_cpu(tmp_path: Path):
    import torch

    assert torch.cuda.is_initialized() is False
    root = tmp_path / "repo"
    root.mkdir()
    row, artifact = _artifact_row(
        root,
        run_id="cpu-replay",
        sample_id="e" * 24,
    )
    head = torch.nn.Linear(analyzer.legacy.FEATURE_DIMENSION, 2)
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
    evidence = analyzer.replay_persisted_head_softmax_router(
        latest_results=[row],
        artifacts={"e" * 24: artifact},
        model=model,
        device=torch.device("cpu"),
    )
    assert evidence["artifacts_replayed"] == 1
    assert set(evidence["maximum_absolute_difference"].values()) == {0.0}
    assert torch.cuda.is_initialized() is False


def test_formal_1775_metrics_use_shared_balanced_path(release):
    spec, selected = analyzer._formal_selection(release)
    contract = build_run_dataset_contract(
        release,
        spec,
        selected,
        score_spec=ScoreSpec(
            key="ai_score",
            direction="higher_means_fake",
            fixed_threshold=0.5,
            threshold_operator=">",
        ),
    )
    run_id = analyzer.DEFAULT_FORMAL_RUN_ID
    fingerprint = "f" * 64
    results = [
        {
            **build_result_identity(
                row,
                run_id=run_id,
                run_manifest_fingerprint=fingerprint,
            ),
            "status": "ok",
            "valid_for_metrics": True,
            "ai_score": 0.1 if row["label"] == 0 else 0.9,
        }
        for row in selected
    ]
    bundle = SimpleNamespace(
        mode="formal",
        selected=tuple(selected),
        release=release,
        latest_results=tuple(results),
        run_id=run_id,
        fingerprint=fingerprint,
        contract=contract,
    )
    metrics = analyzer.recompute_metrics(bundle)
    assert metrics["schema_version"] == analyzer.METRICS_SCHEMA_VERSION
    assert metrics["coverage"]["is_complete"] is True
    assert (
        metrics["primary"]["all_conditions_macro"]["overall"]["auroc"]["estimate"]
        == 1.0
    )
    with pytest.raises(ValueError, match="iterations=1000"):
        analyzer.recompute_metrics(bundle, iterations=10)


def test_fresh_model_replay_requires_and_covers_all_1775(
    monkeypatch: pytest.MonkeyPatch,
    release,
):
    _, selected = analyzer._formal_selection(release)
    arrays = _arrays()
    score = _score_fields(arrays)
    preprocess = {"fixture": "independent-preprocess"}
    latest = tuple(
        {
            "sample_id": row["sample_id"],
            "preprocess": preprocess,
            **score,
        }
        for row in selected
    )
    artifact = analyzer.OmniAIDArtifact(
        sample_id="shared",
        path=Path("/tmp/not-read.npz"),
        file_sha256="0" * 64,
        file_bytes=9848,
        array_sha256={
            key: analyzer._array_sha256(value) for key, value in arrays.items()
        },
        arrays=arrays,
    )
    bundle = SimpleNamespace(
        mode="formal",
        selected=tuple(selected),
        latest_results=latest,
        artifacts={row["sample_id"]: artifact for row in selected},
    )
    monkeypatch.setattr(
        analyzer,
        "_canonical_input_path",
        lambda row, repo_root: Path("/tmp/fixture.jpg"),
    )
    monkeypatch.setattr(
        analyzer.legacy_audit,
        "_preprocess",
        lambda path: (np.zeros((3, 448, 448), dtype=np.float32), preprocess),
    )
    monkeypatch.setattr(
        analyzer.legacy_audit,
        "_fresh_forward",
        lambda model, device, image: (
            arrays,
            {"score": 0.5, "raw_logit_margin": 0.0},
        ),
    )
    evidence, rows = analyzer.replay_model(
        bundle,
        repo_root=REPO_ROOT,
        model=object(),
        device=SimpleNamespace(type="cpu"),
    )
    assert evidence["complete_model_forward_passes"] == 1775
    assert evidence["six_array_sets_compared"] == 1775
    assert len(rows) == 1775


def test_independent_model_builder_uses_legacy_scientific_path(
    monkeypatch: pytest.MonkeyPatch,
):
    called = []

    class Model:
        def parameters(self):
            return []

    evidence = {
        "strict_load": True,
        "state_entries": analyzer.legacy.CHECKPOINT["tensor_count"],
        "state_elements": analyzer.legacy.CHECKPOINT["state_elements"],
        "svd_modules": analyzer.legacy.SVD_MODULE_COUNT,
        "parameter_count": 507_041_863,
        "base_weights_downloaded": False,
        "eval_mode": True,
    }

    def fake_build(state, config, device, space_root):
        called.append((state, config, device, space_root))
        return Model(), evidence

    monkeypatch.setattr(
        analyzer.legacy_audit,
        "_build_model",
        fake_build,
    )
    model, observed = analyzer._build_independent_model(
        state={"state": 1},
        config={"config": 1},
        device=SimpleNamespace(type="cpu"),
        space_root=Path("/tmp/space"),
    )
    assert isinstance(model, Model)
    assert observed is evidence
    assert len(called) == 1


def test_official_golden_uses_independent_preprocess_and_forward(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    arrays = _arrays()
    hashes = {key: analyzer._array_sha256(value) for key, value in arrays.items()}
    preprocess_by_path = {}
    score_by_path = {}
    cases = []
    for frozen in analyzer.legacy.GOLDEN_CASES:
        relative = str(frozen["path"])
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(relative.encode())
        preprocess = {
            "decoded_rgb_sha256": frozen["decoded_rgb_sha256"],
            "resized_rgb_sha256": frozen["resized_rgb_sha256"],
            "tensor_sha256": frozen["tensor_sha256"],
        }
        probability = float(frozen["official_service_probability"])
        preprocess_by_path[relative] = preprocess
        score_by_path[relative] = probability
        cases.append(
            {
                "path": relative,
                "input_sha256": frozen["sha256"],
                "preprocess": preprocess,
                "array_sha256": hashes,
                "logits": [0.0, 0.0],
                "final_expert_gates": [
                    0.5,
                    0.5,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                ],
                "fake_probability": probability,
                "repeat_all_arrays_exact": True,
                "observed_official_service_probability": probability,
                "official_service_probability_abs_diff": 0.0,
                "frozen_runtime_logit_max_abs_diff": None,
                "frozen_runtime_probability_abs_diff": None,
                "frozen_runtime_gate_max_abs_diff": None,
            }
        )
    real_sha256_file = analyzer.sha256_file
    monkeypatch.setattr(
        analyzer,
        "sha256_file",
        lambda path: next(
            (
                str(frozen["sha256"])
                for frozen in analyzer.legacy.GOLDEN_CASES
                if Path(path).as_posix().endswith(str(frozen["path"]))
            ),
            real_sha256_file(Path(path)),
        ),
    )
    monkeypatch.setattr(
        analyzer.legacy_audit,
        "_preprocess",
        lambda path: (
            Path(path).relative_to(tmp_path).as_posix(),
            preprocess_by_path[Path(path).relative_to(tmp_path).as_posix()],
        ),
    )
    monkeypatch.setattr(
        analyzer.legacy_audit,
        "_fresh_forward",
        lambda model, device, image: (
            arrays,
            {
                "score": score_by_path[image],
                "raw_logit_margin": 0.0,
            },
        ),
    )
    evidence = analyzer.audit_official_golden(
        model=object(),
        device=SimpleNamespace(type="cpu"),
        space_root=tmp_path,
        recorded={
            "status": "passed",
            "kind": analyzer.OFFICIAL_GOLDEN_KIND,
            "device_family": "cpu",
            "runtime_abs_tolerance": (analyzer.legacy.GOLDEN_RUNTIME_ABS_TOLERANCE),
            "official_service_abs_tolerance": (
                analyzer.legacy.GOLDEN_SERVICE_ABS_TOLERANCE
            ),
            "official_service_observed_at": "2026-07-25",
            "cases": cases,
            "mouse_model_scores_computed": 0,
        },
    )
    assert evidence["cases_audited"] == 4
    assert evidence["all_recorded_arrays_exact"] is True


def test_standard_output_paths_and_smoke_report_are_frozen(
    tmp_path: Path,
):
    run_dir = tmp_path / "results" / analyzer.DEFAULT_FORMAL_RUN_ID
    metrics = run_dir / "balanced250_metrics.json"
    audit = run_dir / "independent_audit.json"
    assert analyzer._validate_formal_output_scope(
        run_dir=run_dir,
        metrics_output_path=metrics,
        audit_output_path=audit,
    ) == {"metrics": metrics, "audit": audit}
    with pytest.raises(ValueError, match="must be"):
        analyzer._validate_formal_output_scope(
            run_dir=run_dir,
            metrics_output_path=tmp_path / "elsewhere.json",
            audit_output_path=audit,
        )
    report = analyzer._smoke_comparison_default_path(
        results_dir=tmp_path,
        reference_run_id=analyzer.DEFAULT_SMOKE_RUN_ID_A,
        replay_run_id=analyzer.DEFAULT_SMOKE_RUN_ID_B,
    )
    assert report.parent == tmp_path / "_reports"
    assert report.name.startswith(analyzer.SMOKE_COMPARISON_SCHEMA_VERSION)


def test_strict_json_and_output_target_protection(tmp_path: Path):
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"a":1,"a":2}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        analyzer._load_json(duplicate, "duplicate")
    rows = tmp_path / "rows.jsonl"
    rows.write_text('{"b": 2, "a": 1}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="canonical"):
        analyzer._read_jsonl_strict(rows, "rows")
    protected = tmp_path / "evidence.json"
    protected.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="overwrite evidence"):
        analyzer._validate_output_targets(
            {"audit": protected},
            protected_files=[protected],
            protected_dirs=[],
        )


def test_final_bundle_verification_rejects_adapter_source_changed_during_analysis(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    repo_root = tmp_path / "repo"
    recorded_sources = {}
    for index, relative in enumerate(analyzer.EXPECTED_ADAPTER_SOURCE_PATHS):
        path = repo_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"frozen source {index}\n", encoding="utf-8")
        recorded_sources[relative] = {
            "path": relative,
            "bytes": path.stat().st_size,
            "sha256": analyzer.sha256_file(path),
        }

    evidence_paths = {}
    evidence_snapshot = {
        "artifact_inventory_sha256": "a" * 64,
        "dataset_manifest_sha256": "b" * 64,
    }
    for key in (
        "manifest_sha256",
        "results_sha256",
        "expected_inputs_sha256",
        "runtime_summary_sha256",
    ):
        path = tmp_path / f"{key}.json"
        path.write_text("{}\n", encoding="utf-8")
        evidence_paths[key] = path
        evidence_snapshot[key] = analyzer.sha256_file(path)

    release = SimpleNamespace(manifest_sha256="b" * 64)
    contract = SimpleNamespace(as_dict=lambda: {"frozen": True})
    bundle = SimpleNamespace(
        evidence_snapshot=evidence_snapshot,
        manifest_path=evidence_paths["manifest_sha256"],
        results_path=evidence_paths["results_sha256"],
        expected_path=evidence_paths["expected_inputs_sha256"],
        summary_path=evidence_paths["runtime_summary_sha256"],
        latest_results=(),
        artifact_root=tmp_path / "artifact-root",
        artifact_dir=tmp_path / "artifact-root" / "artifacts",
        release=release,
        selected=(),
        contract=contract,
        immutable={"adapter_sources": recorded_sources},
        mode="formal",
    )
    monkeypatch.setattr(
        analyzer,
        "validate_artifact_inventory",
        lambda **kwargs: {},
    )
    monkeypatch.setattr(
        analyzer,
        "_artifact_inventory_sha256",
        lambda artifacts: "a" * 64,
    )
    monkeypatch.setattr(
        analyzer,
        "_rebuild_contract",
        lambda **kwargs: (release, [], contract),
    )

    analyzer._verify_bundle_unchanged(bundle, repo_root=repo_root)
    changed = repo_root / "eval/opensource/analyze_omniaid_balanced.py"
    changed.write_text("changed during replay\n", encoding="utf-8")
    with pytest.raises(ValueError, match="adapter source changed"):
        analyzer._verify_bundle_unchanged(bundle, repo_root=repo_root)


def test_main_defaults_to_fresh_replay_and_writes_standard_outputs(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    calls = []
    monkeypatch.setattr(
        analyzer,
        "_assert_runner_contract_exports",
        lambda: runner,
    )

    def fake_analyze(**kwargs):
        calls.append(kwargs)
        return {"status": "replay_audit_passed"}

    monkeypatch.setattr(analyzer, "analyze", fake_analyze)
    assert (
        analyzer.main(
            [
                "--repo-root",
                str(REPO_ROOT),
                "--device",
                "cpu",
            ]
        )
        == 0
    )
    assert calls[0]["replay"] is True
    run_dir = REPO_ROOT / analyzer.DEFAULT_RESULTS_DIR / analyzer.DEFAULT_FORMAL_RUN_ID
    assert calls[0]["metrics_output_path"] == (run_dir / "balanced250_metrics.json")
    assert calls[0]["audit_output_path"] == (run_dir / "independent_audit.json")
    assert json.loads(capsys.readouterr().out)["status"] == ("replay_audit_passed")
