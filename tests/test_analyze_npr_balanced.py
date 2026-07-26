from __future__ import annotations

import dataclasses
import hashlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from PIL import Image

from eval.opensource import analyze_npr_balanced as analyzer
from eval.opensource.balanced_run_contract import (
    ScoreSpec,
    build_run_dataset_contract,
    selected_ids_sha256,
)
from eval.opensource.canonical_release import (
    Capability,
    SelectionSpec,
    load_canonical_release,
    select_inputs,
)
from eval.opensource.common import stable_json


REPO_ROOT = Path(__file__).resolve().parents[1]
BALANCED_MANIFEST = (
    REPO_ROOT / "outputs/opensource/balanced250_v1/manifest.json"
)


def _success_score_payload(
    *,
    sample_id: str = "sample",
    condition: str = "real",
    raw_logit: float = -1000.0,
    probability: float = 0.0,
) -> dict[str, Any]:
    decision = probability > 0.5
    return {
        "sample_id": sample_id,
        "condition": condition,
        "raw_logit": raw_logit,
        "probability": probability,
        "ai_score": probability,
        "score": probability,
        "score_semantics": (
            "official_float32_sigmoid_probability_higher_is_fake"
        ),
        "classification_decision": decision,
        "classification_threshold": 0.5,
        "classification_threshold_operator": ">",
        "classification": {
            "raw_logit": raw_logit,
            "probability": probability,
            "ai_score": probability,
            "score": probability,
            "threshold": 0.5,
            "threshold_operator": ">",
            "decision": decision,
            "semantics": (
                "official_float32_sigmoid_probability_higher_is_fake"
            ),
        },
        "t1": {
            "raw_logit": raw_logit,
            "probability": probability,
            "ai_score": probability,
            "score": probability,
            "threshold": 0.5,
            "threshold_operator": ">",
            "decision": decision,
            "policy": "official_NPR_AIGC_float32_sigmoid",
        },
        "manual_replay": {
            "raw_logit": raw_logit,
            "probability": probability,
            "ai_score": probability,
            "classification_decision": decision,
            "model_forward_calls": 1,
            "fc_hook_calls": 1,
            "official_logit_exact_match": True,
            "official_probability_exact_match": True,
        },
    }


def _release_and_contract() -> tuple[Any, tuple[dict[str, Any], ...], Any]:
    release = load_canonical_release(
        REPO_ROOT, BALANCED_MANIFEST, verify_files=False
    )
    spec = SelectionSpec(capability=Capability.WHOLE_IMAGE_T1)
    selected = tuple(select_inputs(release, spec))
    contract = build_run_dataset_contract(
        release, spec, selected, score_spec=analyzer._score_spec()
    )
    return release, selected, contract


def test_primary_and_raw_score_contracts_are_distinct() -> None:
    assert analyzer._score_spec() == ScoreSpec(
        key="ai_score",
        direction="higher_means_fake",
        fixed_threshold=0.5,
        threshold_operator=">",
    )
    assert analyzer._raw_logit_score_spec() == ScoreSpec(
        key="raw_logit",
        direction="higher_means_fake",
        fixed_threshold=0.0,
        threshold_operator=">",
    )
    assert analyzer.PREPROCESS_PROFILE == (
        "author_documented_aigcdetect_native_even_trim_completion"
    )


@pytest.mark.parametrize(
    ("width", "height", "expected"),
    [
        (400, 400, (400, 400)),
        (401, 400, (400, 400)),
        (400, 401, (400, 400)),
        (1285, 1137, (1284, 1136)),
    ],
)
def test_effective_native_size(
    width: int,
    height: int,
    expected: tuple[int, int],
) -> None:
    assert analyzer.effective_native_size(width, height) == expected


@pytest.mark.parametrize(
    ("width", "height"),
    [(1, 10), (10, 1), (0, 2), (True, 2), (2, False)],
)
def test_effective_native_size_rejects_invalid(
    width: int,
    height: int,
) -> None:
    with pytest.raises(ValueError, match="above one"):
        analyzer.effective_native_size(width, height)


def test_static_score_validation_accepts_sigmoid_underflow() -> None:
    row = _success_score_payload(raw_logit=-1000.0, probability=0.0)
    analyzer._validate_score_payload(row, sample_id="sample")


def test_static_score_validation_accepts_positive_raw_rounded_to_half() -> None:
    row = _success_score_payload(raw_logit=1e-8, probability=0.5)
    analyzer._validate_score_payload(row, sample_id="sample")
    diagnostic = analyzer._boundary_and_saturation_diagnostic(
        [
            {
                "sample_id": f"{condition}-{index}",
                "condition": condition,
                "raw_logit": 1e-8 if index == 0 else -1000.0,
                "ai_score": 0.5 if index == 0 else 0.0,
            }
            for condition in analyzer.FORMAL_COUNTS
            for index in range(analyzer.FORMAL_COUNTS[condition])
        ]
    )
    mismatch = diagnostic["decision_boundary_disagreements"][
        "raw_positive_probability_not_above_half"
    ]
    assert mismatch["count"] == len(analyzer.FORMAL_COUNTS)
    assert diagnostic["official_decision_authority"] == {
        "score_key": "ai_score",
        "threshold": 0.5,
        "operator": ">",
    }


def test_static_score_validation_rejects_alias_drift() -> None:
    row = _success_score_payload(raw_logit=2.0, probability=0.75)
    row["classification_decision"] = False
    with pytest.raises(ValueError, match="alias"):
        analyzer._validate_score_payload(row, sample_id="sample")


def test_strict_json_rejects_duplicate_and_nonfinite() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        analyzer._json_loads('{"a":1,"a":2}', "payload")
    with pytest.raises(ValueError, match="non-finite"):
        analyzer._json_loads('{"a":NaN}', "payload")


def test_strict_jsonl_requires_canonical_newline(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl"
    path.write_text('{"b": 2, "a": 1}', encoding="utf-8")
    with pytest.raises(ValueError, match="final newline"):
        analyzer._read_jsonl_strict(path, "rows")
    path.write_text('{"b": 2, "a": 1}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="canonical"):
        analyzer._read_jsonl_strict(path, "rows")
    row = {"a": 1, "b": 2}
    path.write_text(f"{stable_json(row)}\n", encoding="utf-8")
    assert analyzer._read_jsonl_strict(path, "rows") == [row]


def test_safe_repo_path_rejects_traversal_and_symlink(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    inside = root / "inside.txt"
    inside.write_text("ok", encoding="utf-8")
    assert analyzer._safe_repo_path(
        "inside.txt", repo_root=root, label="inside"
    ) == inside.resolve()
    with pytest.raises(ValueError, match="traversing"):
        analyzer._safe_repo_path(
            "../outside.txt", repo_root=root, label="outside"
        )
    link = root / "link"
    link.symlink_to(inside)
    with pytest.raises(ValueError, match="symlink"):
        analyzer._safe_repo_path("link", repo_root=root, label="link")


def test_visibility_contract_for_real_and_fullframe() -> None:
    release, selected, _contract = _release_and_contract()
    del release
    real = next(row for row in selected if row["condition"] == "real")
    fullframe = next(
        row for row in selected if row["condition"] == "fullframe_cat"
    )
    real_value = analyzer._independent_visibility_diagnostic(
        real, repo_root=REPO_ROOT
    )
    fullframe_value = analyzer._independent_visibility_diagnostic(
        fullframe, repo_root=REPO_ROOT
    )
    assert real_value["edit_visibility"] == "not_applicable"
    assert real_value["edit_visibility_evidence"]["gt_mask_kind"] == "all_zero"
    assert fullframe_value["edit_visibility"] == "not_applicable"
    assert fullframe_value["edit_visibility_evidence"]["gt_mask_kind"] == (
        "not_applicable"
    )
    analyzer._reject_unsupported_claims(real_value, "real result")
    analyzer._reject_unsupported_claims(fullframe_value, "fullframe result")


def test_real_local_visibility_structure_is_not_model_localization() -> None:
    _release, selected, _contract = _release_and_contract()
    local = next(
        row for row in selected if row["condition"] == "local_cat"
    )
    diagnostic = analyzer._independent_visibility_diagnostic(
        local, repo_root=REPO_ROOT
    )
    evidence = diagnostic["edit_visibility_evidence"]
    assert evidence["pixel_center_mapping"] == (
        analyzer._INPUT_PIXEL_CENTER_MAPPING
    )
    assert evidence["gt_mask_kind"] == "exact_diff"
    analyzer._reject_unsupported_claims(diagnostic, "local result")

    with pytest.raises(ValueError, match="unsupported"):
        analyzer._reject_unsupported_claims(
            {
                "model_output": {
                    "pixel_center_mapping": (
                        analyzer._INPUT_PIXEL_CENTER_MAPPING
                    )
                }
            },
            "payload",
        )
    with pytest.raises(ValueError, match="unsupported"):
        analyzer._reject_unsupported_claims(
            {
                "edit_visibility_evidence": {
                    "nested_model_output": {
                        "pixel_center_mapping": (
                            analyzer._INPUT_PIXEL_CENTER_MAPPING
                        )
                    }
                }
            },
            "payload",
        )
    with pytest.raises(ValueError, match="unsupported"):
        analyzer._reject_unsupported_claims(
            {
                "edit_visibility_evidence": {
                    "pixel_center_mapping": "model_dense_pixel_mapping",
                    "gt_mask_kind": "exact_diff",
                }
            },
            "payload",
        )
    with pytest.raises(ValueError, match="unsupported"):
        analyzer._reject_unsupported_claims(
            {
                "edit_visibility_evidence": {
                    "PIXEL_CENTER_MAPPING": (
                        analyzer._INPUT_PIXEL_CENTER_MAPPING
                    ),
                    "gt_mask_kind": "exact_diff",
                }
            },
            "payload",
        )
    with pytest.raises(ValueError, match="unsupported"):
        analyzer._reject_unsupported_claims(
            {
                "edit_visibility_evidence": {
                    "pixel_center_mapping": (
                        analyzer._INPUT_PIXEL_CENTER_MAPPING
                    ),
                    "gt_mask_kind": "model_predicted_mask",
                }
            },
            "payload",
        )


def test_formal_selection_and_visibility_census() -> None:
    _release, selected, contract = _release_and_contract()
    assert len(selected) == analyzer.FORMAL_IMAGES
    assert contract.selection.selected_ids_sha256 == (
        analyzer.FORMAL_SELECTED_IDS_SHA256
    )
    assert analyzer._validate_formal_visibility_census(
        selected, repo_root=REPO_ROOT
    ) == analyzer.EXPECTED_LOCAL_VISIBILITY


def test_dual_odd_cpu_golden_preprocess_hashes() -> None:
    torch = pytest.importorskip("torch")
    path = REPO_ROOT / analyzer.CPU_GOLDEN_INPUT_PATH
    prepared = analyzer.preprocess_image(path, torch_module=torch)
    assert prepared.audit["profile"] == analyzer.PREPROCESS_PROFILE
    assert prepared.audit["decoded_size"] == [1285, 1137]
    assert prepared.audit["effective_size"] == [1284, 1136]
    assert prepared.audit["trim_right"] == 1
    assert prepared.audit["trim_bottom"] == 1
    assert prepared.audit["decoded_rgb_sha256"] == (
        analyzer.CPU_GOLDEN_DECODED_RGB_SHA256
    )
    assert prepared.audit["tensor_sha256"] == (
        analyzer.CPU_GOLDEN_TENSOR_SHA256
    )
    assert prepared.audit["npr_residual_sha256"] == (
        analyzer.CPU_GOLDEN_RESIDUAL_SHA256
    )


def test_independent_source_hf_checkpoint_asset_audit() -> None:
    source, assets, state, module = analyzer.verify_assets(
        source_root=analyzer.DEFAULT_SOURCE_ROOT,
        hf_source_root=analyzer.DEFAULT_HF_SOURCE_ROOT,
        checkpoint_path=analyzer.DEFAULT_CHECKPOINT,
    )
    assert source["commit"] == analyzer.legacy.MODEL_SOURCE_COMMIT
    assert source["hf_space"]["commit"] == analyzer.legacy.HF_SPACE_COMMIT
    assert source["hf_space"]["deployment_mode_defect"][
        "calls_model_eval"
    ] is False
    assert assets["bundle_sha256"] == (
        analyzer.EXPECTED_ASSET_BUNDLE_SHA256
    )
    assert len(state) == analyzer.legacy.CHECKPOINT["state_entries"]
    assert callable(module.resnet50)


def test_final_runner_contract_runtime_and_local_policy_align() -> None:
    runner = analyzer._assert_runner_contract_exports()
    assert analyzer._expected_local_artifact_policy(REPO_ROOT) == (
        runner._local_artifact_policy(REPO_ROOT)
    )
    device, runtime = runner.configure_runtime(
        "cpu", seed=analyzer.EXPECTED_RUNTIME_SEED
    )
    assert str(device) == "cpu"
    assert analyzer._validate_runtime_contract(
        runtime, label="test runtime"
    ) == runtime
    drifted = dict(runtime)
    drifted["autocast"] = True
    with pytest.raises(ValueError, match="deterministic numerical"):
        analyzer._validate_runtime_contract(
            drifted, label="drifted runtime"
        )


def test_manifest_rejects_schema_before_trusting_nested_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_runner = SimpleNamespace(
        _valid_run_id=lambda value: value,
    )
    monkeypatch.setattr(
        analyzer, "_assert_runner_contract_exports", lambda: fake_runner
    )
    with pytest.raises(ValueError, match="key set"):
        analyzer._validate_manifest(
            manifest={},
            repo_root=REPO_ROOT,
            run_id="run",
            expected_mode="formal",
        )
    manifest = {
        "schema_version": "wrong",
        "run_id": "run",
        "status": "complete",
        "started_at": "start",
        "completed_at": "end",
        "fingerprint": "0" * 64,
        "immutable": {},
        "dataset": {},
        "outputs": {},
        "execution": {},
    }
    with pytest.raises(ValueError, match="identity/status"):
        analyzer._validate_manifest(
            manifest=manifest,
            repo_root=REPO_ROOT,
            run_id="run",
            expected_mode="formal",
        )


def test_smoke_selection_id_hash_is_frozen() -> None:
    release = load_canonical_release(
        REPO_ROOT, BALANCED_MANIFEST, verify_files=False
    )
    selected = select_inputs(
        release,
        SelectionSpec(
            capability=Capability.WHOLE_IMAGE_T1,
            per_condition_limit=analyzer.SMOKE_PER_CONDITION,
        ),
    )
    assert len(selected) == analyzer.SMOKE_IMAGES
    assert selected_ids_sha256(
        str(row["sample_id"]) for row in selected
    ) == analyzer.SMOKE_SELECTED_IDS_SHA256


def test_raw_logit_diagnostic_reuses_exact_selection_and_bootstrap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    release, selected, contract = _release_and_contract()
    rows = tuple(
        {
            "sample_id": str(row["sample_id"]),
            "condition": str(row["condition"]),
            "raw_logit": -1000.0,
            "ai_score": 0.0,
        }
        for row in selected
    )
    coverage = {
        "inputs": 1775,
        "panel": 1750,
        "source_pairs": 1500,
        "results": 1775,
        "is_complete": True,
    }
    bootstrap = {
        "iterations": analyzer.BOOTSTRAP_ITERATIONS,
        "seed": analyzer.BOOTSTRAP_SEED,
        "primary_unit": "clusters",
        "secondary_unit": "clusters",
        "ci": "two_sided_95_percentile",
    }
    official_contract_sha = "1" * 64
    official = {
        "schema_version": analyzer.METRICS_SCHEMA_VERSION,
        "dataset_schema_version": release.schema_version,
        "dataset_id": release.dataset_id,
        "run_id": "formal",
        "run_manifest_fingerprint": "f" * 64,
        "run_dataset_contract_sha256": official_contract_sha,
        "bootstrap": bootstrap,
        "coverage": coverage,
    }
    calls: list[Any] = []

    def fake_summary(
        inputs: Any,
        panel: Any,
        source_pairs: Any,
        results: Any,
        **kwargs: Any,
    ) -> dict[str, Any]:
        calls.append(
            (inputs, panel, source_pairs, results, dict(kwargs))
        )
        raw_contract = kwargs["run_dataset_contract"]
        assert raw_contract.score_spec == analyzer._raw_logit_score_spec()
        assert raw_contract.selection == contract.selection
        return {
            **official,
            "run_dataset_contract_sha256": "2" * 64,
            "score_contract": {
                "score_key": "raw_logit",
                "direction": "higher_is_forged",
                "fixed_threshold": 0.0,
                "fixed_threshold_operator": ">",
                "tpr_at_fpr_5_percent_threshold_operator": ">",
                "target_fpr": 0.05,
                "fpr_quantile_method": "higher",
            },
        }

    monkeypatch.setattr(
        analyzer, "summarize_balanced250_t1", fake_summary
    )
    bundle = SimpleNamespace(
        mode="formal",
        selected=selected,
        contract=contract,
        release=release,
        latest_results=rows,
        run_id="formal",
        fingerprint="f" * 64,
    )
    diagnostic = analyzer.recompute_raw_logit_diagnostic(
        bundle, official_metrics=official
    )
    assert len(calls) == 1
    assert calls[0][4]["iterations"] == analyzer.BOOTSTRAP_ITERATIONS
    assert calls[0][4]["seed"] == analyzer.BOOTSTRAP_SEED
    assert diagnostic["official_probability_result_remains_primary"] is True
    assert diagnostic["must_not_replace_official_fixed_threshold_result"] is True
    proof = diagnostic["same_selection_and_bootstrap_proof"]
    assert proof["selected_images"] == analyzer.FORMAL_IMAGES
    assert proof["selected_ids_sha256"] == analyzer.FORMAL_SELECTED_IDS_SHA256
    assert proof["official_dataset_contract_sha256"] == official_contract_sha
    assert proof["raw_diagnostic_dataset_contract_sha256"] == "2" * 64


def test_raw_logit_diagnostic_rejects_seed_or_iteration_drift() -> None:
    bundle = SimpleNamespace(mode="formal", selected=[{}] * 1775)
    with pytest.raises(ValueError, match="iterations=1000"):
        analyzer.recompute_raw_logit_diagnostic(
            bundle, iterations=999, seed=analyzer.BOOTSTRAP_SEED
        )
    with pytest.raises(ValueError, match="seed=20260726"):
        analyzer.recompute_raw_logit_diagnostic(
            bundle, iterations=1000, seed=1
        )


def _feature_row(
    *,
    root: Path,
    run_id: str,
    sample_id: str,
    values: np.ndarray,
) -> tuple[dict[str, Any], analyzer.FeatureArtifact]:
    feature_dir = root / "outputs/opensource/npr" / run_id / "features"
    feature_dir.mkdir(parents=True)
    path = feature_dir / f"{sample_id}.npy"
    np.save(path, values, allow_pickle=False)
    file_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    array_sha = analyzer._array_sha256(values)
    relative = path.relative_to(root).as_posix()
    row = {
        **_success_score_payload(
            sample_id=sample_id, raw_logit=-1.0, probability=0.25
        ),
        "npr_feature": {
            "relative_path": relative,
            "sha256": file_sha,
            "file_bytes": path.stat().st_size,
            "array_sha256": array_sha,
            "dtype": "float32",
            "shape": [analyzer.FEATURE_DIMENSION],
            "nbytes": analyzer.FEATURE_NBYTES,
            "finite": True,
            "semantics": (
                "official_fc1_input_after_adaptive_global_average_pool"
            ),
            "visibility": "local_only_gitignored_output",
        },
        "npr_feature_path": relative,
        "npr_feature_sha256": file_sha,
        "npr_feature_array_sha256": array_sha,
        "npr_feature_shape": [analyzer.FEATURE_DIMENSION],
        "npr_feature_dtype": "float32",
        "npr_feature_nbytes": analyzer.FEATURE_NBYTES,
        "npr_feature_semantics": (
            "official_fc1_input_after_adaptive_global_average_pool"
        ),
        "artifact_paths": {"npr_feature_npy": relative},
    }
    artifact = analyzer._feature_artifact(
        row=row,
        sample_id=sample_id,
        repo_root=root,
        feature_dir=feature_dir,
    )
    return row, artifact


def test_feature_artifact_validates_canonical_bytes(
    tmp_path: Path,
) -> None:
    values = np.arange(analyzer.FEATURE_DIMENSION, dtype=np.float32)
    row, artifact = _feature_row(
        root=tmp_path,
        run_id="run",
        sample_id="sample",
        values=values,
    )
    assert artifact.file_bytes == 2176
    assert np.array_equal(artifact.array, values)
    row["npr_feature"]["dtype"] = "float64"
    with pytest.raises(ValueError, match="metadata"):
        analyzer._feature_artifact(
            row=row,
            sample_id="sample",
            repo_root=tmp_path,
            feature_dir=artifact.path.parent,
        )


def test_feature_inventory_rejects_extras(tmp_path: Path) -> None:
    values = np.zeros(analyzer.FEATURE_DIMENSION, dtype=np.float32)
    row, artifact = _feature_row(
        root=tmp_path,
        run_id="run",
        sample_id="sample",
        values=values,
    )
    (artifact.path.parent / "extra.npy").write_bytes(artifact.path.read_bytes())
    with pytest.raises(ValueError, match="inventory mismatch"):
        analyzer.validate_feature_inventory(
            latest_results=[row],
            repo_root=tmp_path,
            feature_dir=artifact.path.parent,
        )


def test_smoke_result_comparison_requires_bit_exact(
    tmp_path: Path,
) -> None:
    left_row, left_artifact = _feature_row(
        root=tmp_path / "left",
        run_id="a",
        sample_id="sample",
        values=np.zeros(analyzer.FEATURE_DIMENSION, dtype=np.float32),
    )
    right_row, right_artifact = _feature_row(
        root=tmp_path / "right",
        run_id="b",
        sample_id="sample",
        values=np.zeros(analyzer.FEATURE_DIMENSION, dtype=np.float32),
    )
    for row, run_id in ((left_row, "a"), (right_row, "b")):
        row.update(
            {
                "run_id": run_id,
                "run_manifest_fingerprint": (
                    "a" * 64 if run_id == "a" else "b" * 64
                ),
                "completed_at": "now",
                "latency_ms": 1.0,
                "preprocess_latency_ms": 1.0,
                "peak_cuda_memory_bytes": None,
            }
        )
    report = analyzer.compare_computational_results(
        reference_rows=[left_row],
        replay_rows=[right_row],
        reference_features={"sample": left_artifact},
        replay_features={"sample": right_artifact},
    )
    assert report["raw_logit_abs_tolerance"] == 0.0
    assert report["probability_abs_tolerance"] == 0.0
    right_row["ai_score"] = np.nextafter(
        np.float32(0.25), np.float32(1.0)
    ).item()
    with pytest.raises(ValueError, match="alias|projection|bit-exact"):
        analyzer.compare_computational_results(
            reference_rows=[left_row],
            replay_rows=[right_row],
            reference_features={"sample": left_artifact},
            replay_features={"sample": right_artifact},
        )


def test_unsupported_t2_pair_and_dense_claims_are_rejected() -> None:
    for payload in (
        {"pair_rank": 1},
        {"t2": {"score": 1}},
        {"valid_for_t2": True},
        {"localization_output": "heatmap"},
    ):
        with pytest.raises(ValueError, match="unsupported"):
            analyzer._reject_unsupported_claims(payload, "payload")
    analyzer._reject_unsupported_claims(
        {
            "valid_for_t2": False,
            "native_dense_output": False,
            "localization_output": None,
        },
        "payload",
    )


def test_output_collision_protects_run_and_feature_evidence(
    tmp_path: Path,
) -> None:
    protected = tmp_path / "manifest.json"
    protected.write_text("{}", encoding="utf-8")
    feature_dir = tmp_path / "features"
    feature_dir.mkdir()
    with pytest.raises(ValueError, match="overwrite"):
        analyzer._validate_output_targets(
            {"audit": protected},
            repo_root=tmp_path,
            protected_files=[protected],
            protected_dirs=[feature_dir],
        )
    with pytest.raises(ValueError, match="artifact inventory"):
        analyzer._validate_output_targets(
            {"audit": feature_dir / "audit.json"},
            repo_root=tmp_path,
            protected_files=[protected],
            protected_dirs=[feature_dir],
        )


def test_short_smoke_comparison_name_is_bounded_and_stable() -> None:
    first = analyzer._short_comparison_name("a" * 160, "b" * 160)
    second = analyzer._short_comparison_name("a" * 160, "b" * 160)
    reverse = analyzer._short_comparison_name("b" * 160, "a" * 160)
    assert first == second
    assert first != reverse
    assert len(first) < 80


def test_summary_and_execution_bind_same_device_replay_count() -> None:
    _release, selected, contract = _release_and_contract()
    runner = analyzer._assert_runner_contract_exports()
    coverage = {"is_complete": True}
    summary = {
        "schema_version": analyzer.EXPECTED_RUNTIME_SUMMARY_SCHEMA,
        "summary_kind": "runtime_coverage_only",
        "scientific_metrics": None,
        "scientific_metrics_owner": "analyze_npr_balanced.py",
        "run_id": "run",
        "run_manifest_fingerprint": "f" * 64,
        "status": "complete",
        "mode": "formal",
        "model": analyzer.legacy.MODEL_NAME,
        "model_slug": analyzer.legacy.MODEL_SLUG,
        "preprocess_profile": analyzer.PREPROCESS_PROFILE,
        "score_spec": analyzer._score_spec().as_dict(),
        "raw_logit_diagnostic": runner.MODEL_CONTRACT[
            "raw_logit_diagnostic"
        ],
        "dataset_contract": contract.as_dict(),
        "odd_dimension_counts": runner.odd_dimension_counts(selected),
        "coverage": coverage,
        "same_device_feature_head_sigmoid_replays": len(selected),
        "generated_at": "now",
    }
    analyzer._validate_summary(
        summary=summary,
        bundle_mode="formal",
        run_id="run",
        fingerprint="f" * 64,
        contract=contract,
        coverage=coverage,
        odd_dimension_counts=runner.odd_dimension_counts(selected),
        raw_logit_diagnostic=runner.MODEL_CONTRACT[
            "raw_logit_diagnostic"
        ],
        expected_replays=len(selected),
    )
    execution = {
        "new_successes": len(selected),
        "resume_skips": 0,
        "new_errors": 0,
        "physical_result_rows": len(selected),
        "latest_result_rows": len(selected),
        "superseded_attempts": 0,
        "same_device_feature_head_sigmoid_replays": len(selected),
    }
    manifest = {"execution": execution}
    analyzer._validate_execution(
        manifest=manifest,
        selected_images=len(selected),
        physical_rows=len(selected),
        latest_rows=len(selected),
    )
    execution["same_device_feature_head_sigmoid_replays"] -= 1
    with pytest.raises(ValueError, match="accounting"):
        analyzer._validate_execution(
            manifest=manifest,
            selected_images=len(selected),
            physical_rows=len(selected),
            latest_rows=len(selected),
        )


def test_physical_history_allows_error_retries_but_rejects_post_success() -> None:
    analyzer._validate_physical_attempt_history(
        [
            {"sample_id": "a", "status": "error"},
            {"sample_id": "a", "status": "error"},
            {"sample_id": "a", "status": "ok"},
            {"sample_id": "b", "status": "error"},
        ]
    )
    with pytest.raises(ValueError, match="duplicate successful"):
        analyzer._validate_physical_attempt_history(
            [
                {"sample_id": "a", "status": "ok"},
                {"sample_id": "a", "status": "ok"},
            ]
        )
    with pytest.raises(ValueError, match="after success"):
        analyzer._validate_physical_attempt_history(
            [
                {"sample_id": "a", "status": "ok"},
                {"sample_id": "a", "status": "error"},
            ]
        )


def test_dataclass_replace_changes_only_raw_score_spec() -> None:
    _release, _selected, contract = _release_and_contract()
    raw = dataclasses.replace(
        contract, score_spec=analyzer._raw_logit_score_spec()
    )
    assert raw.score_spec == analyzer._raw_logit_score_spec()
    assert dataclasses.replace(raw, score_spec=contract.score_spec) == contract
