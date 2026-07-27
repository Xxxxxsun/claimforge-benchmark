from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from eval.opensource import analyze_spai_balanced as analyzer
from eval.opensource import run_spai_balanced as runner
from eval.opensource.common import sha256_file, stable_json


def _score_fields(
    *,
    sample_id: str = "sample",
    raw_logit: float = 0.0,
    probability: float = 0.5,
) -> dict[str, Any]:
    decision = bool(probability > analyzer.legacy.CLASSIFICATION_THRESHOLD)
    classification = {
        "raw_logit": raw_logit,
        "probability": probability,
        "ai_score": probability,
        "score": probability,
        "decision": decision,
        "threshold": analyzer.legacy.CLASSIFICATION_THRESHOLD,
        "threshold_operator": (analyzer.legacy.CLASSIFICATION_THRESHOLD_OPERATOR),
        "semantics": analyzer.legacy.SCORE_SEMANTICS,
    }
    t1 = {key: value for key, value in classification.items() if key != "semantics"}
    t1["policy"] = analyzer.legacy.T1_POLICY
    manual = {
        "raw_logit": raw_logit,
        "probability": probability,
        "ai_score": probability,
        "classification_decision": decision,
        "model_forward_calls": 1,
        "to_kv_hook_calls": 1,
        "attention_hook_calls": 1,
        "norm_hook_calls": 1,
        "official_attention_exact_match": True,
        "official_aggregated_exact_match": True,
        "official_feature_exact_match": True,
        "official_logit_exact_match": True,
        "official_probability_exact_match": True,
        "sca_replay": True,
        "norm_replay": True,
        "complete_mlp_replay": True,
    }
    return {
        "sample_id": sample_id,
        "raw_logit": raw_logit,
        "probability": probability,
        "ai_score": probability,
        "score": probability,
        "score_semantics": analyzer.legacy.SCORE_SEMANTICS,
        "classification_decision": decision,
        "classification_threshold": (analyzer.legacy.CLASSIFICATION_THRESHOLD),
        "classification_threshold_operator": (
            analyzer.legacy.CLASSIFICATION_THRESHOLD_OPERATOR
        ),
        "classification": classification,
        "t1": t1,
        "manual_replay": manual,
    }


def _artifact_record(
    *,
    repo_root: Path,
    run_id: str,
    sample_id: str,
    prefix: str,
    directory: str,
    array: np.ndarray,
    semantics: str,
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    path = (
        repo_root
        / "outputs"
        / "opensource"
        / "spai"
        / run_id
        / directory
        / f"{sample_id}.npy"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(analyzer._npy_bytes(array))
    relative = path.relative_to(repo_root).as_posix()
    file_sha = sha256_file(path)
    array_sha = analyzer._array_sha256(array)
    record = {
        "relative_path": relative,
        "sha256": file_sha,
        "file_bytes": path.stat().st_size,
        "array_sha256": array_sha,
        "dtype": "float32",
        "shape": list(array.shape),
        "nbytes": array.nbytes,
        "finite": True,
        "semantics": semantics,
        "allow_pickle": False,
    }
    aliases = {
        f"{prefix}_path": relative,
        f"{prefix}_sha256": file_sha,
        f"{prefix}_array_sha256": array_sha,
        f"{prefix}_shape": list(array.shape),
        f"{prefix}_dtype": "float32",
        f"{prefix}_nbytes": array.nbytes,
        f"{prefix}_semantics": semantics,
    }
    return record, aliases, path


def _artifact_row(
    repo_root: Path,
    *,
    run_id: str = "run-a",
    sample_id: str = "sample",
    patch_count: int = 4,
    offset: float = 0.0,
) -> tuple[dict[str, Any], analyzer.SampleArtifacts]:
    patch = np.ascontiguousarray(
        np.arange(
            patch_count * analyzer.FEATURE_DIMENSION,
            dtype=np.float32,
        ).reshape(patch_count, analyzer.FEATURE_DIMENSION)
        + np.float32(offset)
    )
    feature = np.ascontiguousarray(
        np.arange(analyzer.FEATURE_DIMENSION, dtype=np.float32) + np.float32(offset)
    )
    attention = np.ascontiguousarray(
        np.arange(
            analyzer.ATTENTION_HEADS * patch_count,
            dtype=np.float32,
        ).reshape(analyzer.ATTENTION_HEADS, patch_count)
        + np.float32(offset)
    )
    row = {
        **_score_fields(sample_id=sample_id),
        "preprocess": {"geometry": {"effective_patch_count": patch_count}},
        "attention_is_diagnostic_not_t2": True,
    }
    paths: dict[str, str] = {}
    artifacts: list[analyzer.ArrayArtifact] = []
    for prefix, directory, array, semantics, path_key in (
        (
            "spai_patch_features",
            "patch_features",
            patch,
            analyzer.PATCH_FEATURE_SEMANTICS,
            "spai_patch_features_npy",
        ),
        (
            "spai_feature",
            "feature",
            feature,
            analyzer.FEATURE_SEMANTICS,
            "spai_feature_npy",
        ),
        (
            "spai_attention",
            "attention",
            attention,
            analyzer.ATTENTION_SEMANTICS,
            "spai_attention_npy",
        ),
    ):
        record, aliases, path = _artifact_record(
            repo_root=repo_root,
            run_id=run_id,
            sample_id=sample_id,
            prefix=prefix,
            directory=directory,
            array=array,
            semantics=semantics,
        )
        row[prefix] = record
        row.update(aliases)
        paths[path_key] = record["relative_path"]
        artifacts.append(
            analyzer.ArrayArtifact(
                kind=prefix,
                sample_id=sample_id,
                path=path,
                file_sha256=record["sha256"],
                file_bytes=record["file_bytes"],
                array_sha256=record["array_sha256"],
                array=array,
            )
        )
    row["artifact_paths"] = paths
    row["feature_array_sha256"] = row["spai_feature_array_sha256"]
    return row, analyzer.SampleArtifacts(*artifacts)


def _smoke_row(
    repo_root: Path,
    *,
    run_id: str,
    sample_id: str = "sample",
) -> tuple[dict[str, Any], analyzer.SampleArtifacts]:
    row, artifacts = _artifact_row(
        repo_root,
        run_id=run_id,
        sample_id=sample_id,
    )
    row.update(
        {
            "run_id": run_id,
            "run_manifest_fingerprint": "a" * 64,
            "config_fingerprint": "a" * 64,
            "completed_at": "2026-07-26T00:00:00Z",
            "preprocess_latency_ms": 1.0,
            "latency_ms": 2.0,
            "peak_cuda_memory_bytes": 0,
        }
    )
    return row, artifacts


def test_runner_exports_are_independently_pinned() -> None:
    assert analyzer._assert_runner_contract_exports() is runner
    assert (
        frozenset(runner.IMMUTABLE_CONFIG_KEYS)
        == analyzer.EXPECTED_IMMUTABLE_CONFIG_KEYS
    )
    assert (
        tuple(runner.CPU_OFFICIAL_GOLDEN_CASES)
        == analyzer.EXPECTED_CPU_OFFICIAL_GOLDEN_CASES
    )
    assert (
        runner.FORMAL_SELECTED_ROWS_SHA256
        == analyzer.EXPECTED_FORMAL_SELECTED_ROWS_SHA256
    )
    assert (
        runner.FORMAL_SELECTED_IDS_SHA256
        == analyzer.EXPECTED_FORMAL_SELECTED_IDS_SHA256
    )
    assert (
        runner.SMOKE5X7_SELECTED_IDS_SHA256
        == analyzer.EXPECTED_SMOKE5X7_SELECTED_IDS_SHA256
    )


def test_method_boundary_is_t1_only() -> None:
    assert runner.TASK_SCOPE == {
        "primary_task": "T1_whole_image_AIGC_detection",
        "valid_for_t1": True,
        "valid_for_t2": False,
        "localization_output": None,
        "native_dense_output": False,
    }
    assert runner.ARTIFACT_CONTRACT["attention"]["valid_for_t2"] is False
    assert "not_localization" in analyzer.ATTENTION_SEMANTICS


def test_frozen_selection_hashes_and_counts() -> None:
    assert runner.FORMAL_SELECTED_IDS_SHA256 == (
        "e4418d86461f889e4a4423f26aab63243e6f63a435a49624881c34979b812e41"
    )
    assert runner.SMOKE5X7_SELECTED_IDS_SHA256 == (
        "b420bc581386a540b742d917d60d007f0e5522b6cca43fa217797944c40667e5"
    )
    assert sum(runner.FORMAL_COUNTS.values()) == analyzer.FORMAL_IMAGES


def test_strict_json_rejects_duplicate_and_nonfinite() -> None:
    with pytest.raises(ValueError, match="duplicate JSON key"):
        analyzer._json_loads('{"x":1,"x":2}', "test")
    with pytest.raises(ValueError, match="non-finite"):
        analyzer._json_loads('{"x":NaN}', "test")
    with pytest.raises(ValueError, match="scalar type"):
        analyzer._require_exact_json({"rank": True}, {"rank": 1}, "test")
    with pytest.raises(ValueError, match="changed"):
        analyzer._require_exact_json(-0.0, 0.0, "test")


def test_strict_jsonl_requires_canonical_newline(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl"
    path.write_text('{"b":2,"a":1}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="canonical"):
        analyzer._read_jsonl_strict(path, "rows")
    path.write_text(stable_json({"a": 1, "b": 2}), encoding="utf-8")
    with pytest.raises(ValueError, match="final newline"):
        analyzer._read_jsonl_strict(path, "rows")


@pytest.mark.parametrize(
    "key",
    [
        "pair_rank",
        "localization",
        "attention_map",
        "attention_map_path",
        "attention_mask",
        "attention_mask_path",
        "mask",
        "pixel_metrics",
        "t2_score",
        "heatmap_value",
        "predicted_mask",
        "joint_metrics",
        "s_joint",
    ],
)
def test_recursive_t2_localization_claims_are_rejected(key: str) -> None:
    value: Any = {"outer": [{"nested": {key: 1}}]}
    with pytest.raises(ValueError, match="unsupported SPAI claim"):
        analyzer._reject_unsupported_claims(value, "payload")


def test_false_null_scope_and_diagnostic_attention_are_allowed() -> None:
    analyzer._reject_unsupported_claims(
        {
            "valid_for_t2": False,
            "native_dense_output": False,
            "localization_output": None,
            "t2": None,
            "attention_is_diagnostic_not_t2": True,
            "edit_visibility_evidence": {"mask_positive_pixels": 1},
        },
        "payload",
    )


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("valid_for_t2", True),
        ("native_dense_output", 0),
        ("localization_output", {}),
        ("attention_is_diagnostic_not_t2", False),
    ],
)
def test_invalid_scope_declarations_fail(key: str, value: Any) -> None:
    with pytest.raises(ValueError):
        analyzer._reject_unsupported_claims({key: value}, "payload")


@pytest.mark.parametrize(
    ("probability", "decision"),
    [(0.5, False), (0.5000001, True), (0.0, False), (1.0, True)],
)
def test_score_contract_uses_strict_greater_than_threshold(
    probability: float,
    decision: bool,
) -> None:
    row = _score_fields(probability=probability)
    analyzer._validate_score_payload(row, sample_id="sample")
    assert row["classification_decision"] is decision


@pytest.mark.parametrize(
    "mutation",
    [
        lambda row: row.__setitem__("ai_score", 0.1),
        lambda row: row.__setitem__("score", 0.1),
        lambda row: row.__setitem__("classification_decision", True),
        lambda row: row["manual_replay"].__setitem__("model_forward_calls", 2),
        lambda row: row["manual_replay"].__setitem__(
            "official_feature_exact_match", False
        ),
        lambda row: row["manual_replay"].__setitem__("model_forward_calls", True),
        lambda row: row["classification"].__setitem__("raw_logit", True),
        lambda row: row["manual_replay"].__setitem__("extra", True),
    ],
)
def test_score_alias_or_manual_replay_tamper_fails(mutation: Any) -> None:
    row = _score_fields()
    mutation(row)
    with pytest.raises(ValueError):
        analyzer._validate_score_payload(row, sample_id="sample")


def test_nested_numeric_bool_cannot_alias_one() -> None:
    row = _score_fields(raw_logit=1.0)
    row["classification"]["raw_logit"] = True
    with pytest.raises(ValueError):
        analyzer._validate_score_payload(row, sample_id="sample")


def test_safe_repo_path_rejects_traversal_and_symlink(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="traversing"):
        analyzer._safe_repo_path(
            "../outside",
            repo_root=tmp_path,
            label="path",
        )
    target = tmp_path / "target"
    target.write_text("x", encoding="utf-8")
    link = tmp_path / "link"
    link.symlink_to(target)
    with pytest.raises(ValueError, match="symlink"):
        analyzer._safe_repo_path(
            "link",
            repo_root=tmp_path,
            label="path",
        )


def test_resolve_run_dir_rejects_final_directory_symlink(
    tmp_path: Path,
) -> None:
    results = tmp_path / "results"
    target = results / "target"
    target.mkdir(parents=True)
    (results / "linked-run").symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        analyzer._resolve_run_dir(results, "linked-run")


def test_three_artifact_inventory_validates_canonical_arrays(
    tmp_path: Path,
) -> None:
    row, _artifacts = _artifact_row(tmp_path)
    root = tmp_path / "outputs" / "opensource" / "spai" / "run-a"
    validated = analyzer.validate_artifact_inventory(
        latest_results=[row],
        repo_root=tmp_path,
        artifact_root=root,
        run_id="run-a",
    )
    assert set(validated) == {"sample"}
    assert validated["sample"].patch_features.array.shape == (
        4,
        analyzer.FEATURE_DIMENSION,
    )
    assert validated["sample"].attention.array.shape == (
        analyzer.ATTENTION_HEADS,
        4,
    )


@pytest.mark.parametrize(
    ("record", "field", "value"),
    [
        ("spai_patch_features", "dtype", "float64"),
        ("spai_feature", "finite", False),
        ("spai_attention", "allow_pickle", True),
        ("spai_attention", "semantics", "localization"),
        ("spai_feature", "shape", [1, 1096]),
    ],
)
def test_artifact_metadata_tamper_fails(
    tmp_path: Path,
    record: str,
    field: str,
    value: Any,
) -> None:
    row, _artifacts = _artifact_row(tmp_path)
    row[record][field] = value
    root = tmp_path / "outputs" / "opensource" / "spai" / "run-a"
    with pytest.raises(ValueError):
        analyzer.validate_artifact_inventory(
            latest_results=[row],
            repo_root=tmp_path,
            artifact_root=root,
            run_id="run-a",
        )


def test_artifact_file_tamper_and_extra_inventory_fail(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    row, artifacts = _artifact_row(tmp_path)
    root = tmp_path / "outputs" / "opensource" / "spai" / "run-a"
    artifacts.feature.path.write_bytes(artifacts.feature.path.read_bytes() + b"x")
    original_load = analyzer.np.load

    def guarded_load(candidate, *args, **kwargs):
        if Path(candidate) == artifacts.feature.path:
            pytest.fail("oversized NPY must be rejected before np.load")
        return original_load(candidate, *args, **kwargs)

    monkeypatch.setattr(
        analyzer.np,
        "load",
        guarded_load,
    )
    with pytest.raises(ValueError, match="size|hash"):
        analyzer.validate_artifact_inventory(
            latest_results=[row],
            repo_root=tmp_path,
            artifact_root=root,
            run_id="run-a",
        )
    row, _artifacts = _artifact_row(tmp_path)
    (root / "attention" / "extra.npy").write_bytes(b"x")
    with pytest.raises(ValueError, match="inventory"):
        analyzer.validate_artifact_inventory(
            latest_results=[row],
            repo_root=tmp_path,
            artifact_root=root,
            run_id="run-a",
        )


@pytest.mark.parametrize(
    ("difference", "tolerance", "passes"),
    [
        (1e-6, 1e-5, True),
        (1.01e-5, 1e-5, False),
        (1e-7, 1e-6, True),
        (1.01e-6, 1e-6, False),
    ],
)
def test_array_tolerance_boundary(
    difference: float,
    tolerance: float,
    passes: bool,
) -> None:
    expected = np.zeros((1,), dtype=np.float32)
    actual = np.array([difference], dtype=np.float32)
    if passes:
        analyzer._require_array_close(
            actual,
            expected,
            label="array",
            tolerance=tolerance,
        )
    else:
        with pytest.raises(ValueError, match="differs"):
            analyzer._require_array_close(
                actual,
                expected,
                label="array",
                tolerance=tolerance,
            )


def test_array_comparison_rejects_nonfinite_values() -> None:
    expected = np.zeros((2,), dtype=np.float32)
    for value in (np.nan, np.inf, -np.inf):
        actual = expected.copy()
        actual[0] = value
        with pytest.raises(ValueError, match="finite|shape/dtype"):
            analyzer._require_array_close(
                actual,
                expected,
                label="array",
                tolerance=1.0,
            )


def test_exact_smoke_comparison_checks_all_three_artifacts(
    tmp_path: Path,
) -> None:
    left_row, left_artifacts = _smoke_row(tmp_path, run_id="smoke-a")
    right_row, right_artifacts = _smoke_row(tmp_path, run_id="smoke-b")
    report = analyzer.compare_computational_results(
        reference_rows=[left_row],
        replay_rows=[right_row],
        reference_artifacts={"sample": left_artifacts},
        replay_artifacts={"sample": right_artifacts},
    )
    assert report["exact_computational_projection"] is True
    assert report["independent_artifact_paths"] is True
    assert not any(report["maximum_absolute_differences"].values())


def test_smoke_comparison_rejects_score_or_array_difference(
    tmp_path: Path,
) -> None:
    left_row, left_artifacts = _smoke_row(tmp_path, run_id="smoke-a")
    right_row, right_artifacts = _smoke_row(tmp_path, run_id="smoke-b")
    right_row["preprocess"]["changed"] = True
    with pytest.raises(ValueError, match="differs"):
        analyzer.compare_computational_results(
            reference_rows=[left_row],
            replay_rows=[right_row],
            reference_artifacts={"sample": left_artifacts},
            replay_artifacts={"sample": right_artifacts},
        )


def test_execution_schema_and_replay_counts_are_exact() -> None:
    execution = {
        "new_successes": 35,
        "resume_skips": 0,
        "new_errors": 0,
        "physical_result_rows": 35,
        "latest_result_rows": 35,
        "superseded_attempts": 0,
        "same_device_artifact_replays_before_execution": 0,
        "same_device_artifact_replays_final": 35,
    }
    analyzer._validate_execution(
        manifest={"execution": execution},
        selected_images=35,
        physical_rows=35,
        latest_rows=35,
    )
    execution["extra"] = 0
    with pytest.raises(ValueError, match="key set"):
        analyzer._validate_execution(
            manifest={"execution": execution},
            selected_images=35,
            physical_rows=35,
            latest_rows=35,
        )
    execution.pop("extra")
    execution["new_successes"] = 34
    execution["resume_skips"] = 1
    execution["same_device_artifact_replays_before_execution"] = 0
    with pytest.raises(ValueError, match="replay coverage"):
        analyzer._validate_execution(
            manifest={"execution": execution},
            selected_images=35,
            physical_rows=35,
            latest_rows=35,
        )


def test_execution_device_gate_uses_device_specific_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cpu_nested = {"cpu": True}
    cpu_preflight = {"official_golden": cpu_nested}
    monkeypatch.setattr(
        analyzer,
        "_validate_cpu_official_golden",
        lambda value, assets: value,
    )
    monkeypatch.setattr(
        analyzer,
        "_validate_cuda_official_golden",
        lambda value, assets: value,
    )
    cpu = {
        "performed_after_explicit_device_configuration_before_scoring": True,
        "cross_device_bit_equality_required": False,
        "report": {
            "status": "passed",
            "device": "cpu",
            "reference_device": "cpu",
            "gate": "frozen_CPU_bit_repeat_regression",
            "cross_device_bit_equality_required": False,
            "report": cpu_nested,
        },
    }
    analyzer._validate_execution_device_golden(
        cpu,
        runtime={"device": "cpu"},
        cpu_preflight=cpu_preflight,
        assets={},
    )
    cuda = copy.deepcopy(cpu)
    cuda["report"].update(
        {
            "device": "cuda:0",
            "reference_device": "cuda",
            "gate": ("released_CUDA_highest_no_TF32_implementation_regression"),
            "report": {"cuda": True},
        }
    )
    analyzer._validate_execution_device_golden(
        cuda,
        runtime={"device": "cuda:0"},
        cpu_preflight=cpu_preflight,
        assets={},
    )
    cuda["cross_device_bit_equality_required"] = True
    with pytest.raises(ValueError, match="wrapper"):
        analyzer._validate_execution_device_golden(
            cuda,
            runtime={"device": "cuda:0"},
            cpu_preflight=cpu_preflight,
            assets={},
        )


def test_runtime_is_independently_pinned_against_runner_codrift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _device, runtime = runner.configure_runtime(
        "cpu",
        seed=analyzer.EXPECTED_RUNTIME_SEED,
    )
    analyzer._validate_runtime_contract(runtime, label="runtime")
    monkeypatch.setattr(
        runner,
        "validate_runtime_contract",
        lambda value, label="runtime": value,
    )
    changed = copy.deepcopy(runtime)
    changed["versions"]["numpy"] = "changed"
    with pytest.raises(ValueError, match="package pins"):
        analyzer._validate_runtime_contract(changed, label="runtime")


def test_runtime_summary_schema_is_exact() -> None:
    contract = SimpleNamespace(as_dict=lambda: {"contract": True})
    coverage = {"valid_images": 35, "is_complete": True}
    execution = {
        "same_device_artifact_replays_before_execution": 0,
        "same_device_artifact_replays_final": 35,
    }
    summary = {
        "schema_version": analyzer.EXPECTED_RUNTIME_SUMMARY_SCHEMA,
        "summary_kind": "runtime_coverage_only",
        "scientific_metrics": None,
        "scientific_metrics_owner": "analyze_spai_balanced.py",
        "run_id": "smoke-a",
        "run_manifest_fingerprint": "a" * 64,
        "status": "complete",
        "mode": "smoke",
        "model": analyzer.legacy.MODEL_NAME,
        "model_slug": analyzer.legacy.MODEL_SLUG,
        "preprocess_profile": analyzer.PREPROCESS_PROFILE,
        "score_spec": analyzer._score_spec().as_dict(),
        "dataset_contract": contract.as_dict(),
        "selection_visibility_census": {"diagnostic": True},
        **execution,
        "coverage": coverage,
        "generated_at": "2026-07-26T00:00:00Z",
    }
    analyzer._validate_summary(
        summary=summary,
        bundle_mode="smoke",
        run_id="smoke-a",
        fingerprint="a" * 64,
        contract=contract,
        selection_visibility={"diagnostic": True},
        coverage=coverage,
        execution=execution,
    )
    summary["extra"] = True
    with pytest.raises(ValueError, match="key set"):
        analyzer._validate_summary(
            summary=summary,
            bundle_mode="smoke",
            run_id="smoke-a",
            fingerprint="a" * 64,
            contract=contract,
            selection_visibility={"diagnostic": True},
            coverage=coverage,
            execution=execution,
        )


def test_assets_checkpoint_schema_and_golden_records_are_exact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint_path = tmp_path / "checkpoint.pth"
    checkpoint_path.write_bytes(b"checkpoint")
    checkpoint_sha = sha256_file(checkpoint_path)
    monkeypatch.setattr(
        analyzer,
        "EXPECTED_CHECKPOINT_BYTES",
        checkpoint_path.stat().st_size,
    )
    monkeypatch.setattr(
        analyzer,
        "EXPECTED_CHECKPOINT_SHA256",
        checkpoint_sha,
    )
    monkeypatch.setitem(
        analyzer.legacy.CHECKPOINT,
        "bytes",
        checkpoint_path.stat().st_size,
    )
    monkeypatch.setitem(
        analyzer.legacy.CHECKPOINT,
        "sha256",
        checkpoint_sha,
    )
    items: list[dict[str, Any]] = []
    for index in range(323):
        numel = analyzer.legacy.CHECKPOINT["state_elements"] - 1 if index == 0 else 0
        items.append(
            {
                "key": f"float.{index}",
                "shape": [numel],
                "dtype": "torch.float32",
                "numel": numel,
                "sha256": f"{index:064x}",
            }
        )
    items.append(
        {
            "key": "integer",
            "shape": [1],
            "dtype": "torch.int64",
            "numel": 1,
            "sha256": "f" * 64,
        }
    )
    items_sha = hashlib.sha256(stable_json(items).encode()).hexdigest()
    monkeypatch.setattr(
        analyzer,
        "EXPECTED_CHECKPOINT_SCHEMA_SHA256",
        items_sha,
    )
    monkeypatch.setitem(
        analyzer.legacy.CHECKPOINT,
        "schema_items_sha256",
        items_sha,
    )
    schema = {
        "tensor_count": 324,
        "state_elements": analyzer.legacy.CHECKPOINT["state_elements"],
        "dtype_counts": {"torch.float32": 323, "torch.int64": 1},
        "items_sha256": items_sha,
        "items": items,
        "embedded_config_minimum_patches": 1,
        "released_inference_config_minimum_patches": 4,
        "embedded_config_is_historical_not_restored": True,
    }
    checkpoint = {
        **analyzer.legacy.CHECKPOINT,
        "path": str(checkpoint_path),
        "actual_bytes": checkpoint_path.stat().st_size,
        "actual_sha256": checkpoint_sha,
        "serialization_safety": {
            "weights_only": True,
            "pickle_executed": False,
            "safe_global_allowlist": ["yacs.config.CfgNode"],
            "loader": "torch.load(map_location=cpu, weights_only=True)",
        },
        "schema": schema,
    }
    golden_assets = []
    for frozen in analyzer.legacy.GOLDEN_CASES:
        path = analyzer.DEFAULT_GOLDEN_ROOT / str(frozen["relative_path"])
        golden_assets.append(
            {
                **frozen,
                "path": str(path),
                "bytes": path.stat().st_size,
            }
        )
    assets = {
        "checkpoint": checkpoint,
        "golden_assets": golden_assets,
    }
    assert analyzer._validate_assets_contract(assets) == assets
    for path in (
        ("extra",),
        ("checkpoint", "extra"),
        ("checkpoint", "schema", "extra"),
        ("golden_assets", 0, "extra"),
    ):
        changed = copy.deepcopy(assets)
        target: Any = changed
        for component in path[:-1]:
            target = target[component]
        target[path[-1]] = True
        with pytest.raises(ValueError):
            analyzer._validate_assets_contract(changed)


def test_cpu_model_load_schema_rejects_extras_and_wrong_class() -> None:
    value = {
        "config": {
            "sid_approach": "freq_restoration",
            "resolution_mode": "arbitrary",
            "required_normalization": "positive_0_1",
            "image_patch_size": 224,
            "patch_stride": 224,
            "minimum_patches": 4,
            "feature_extraction_batch": 400,
            "num_classes": 2,
            "attention_heads": 12,
            "attention_embed_dimension": (analyzer.legacy.ATTENTION_EMBED_DIMENSION),
            "original_resolution": True,
        },
        "load": {
            "strict": True,
            "full_state_coverage": True,
            "missing_keys": [],
            "unexpected_keys": [],
            "loaded_tensor_exact_match": True,
        },
        "model": {
            "class": "spai.models.sid.PatchBasedMFViT",
            "state_tensors": analyzer.legacy.CHECKPOINT["tensor_count"],
            "state_elements": analyzer.legacy.CHECKPOINT["state_elements"],
            "feature_dimension": 1096,
            "attention_heads": 12,
            "eval": True,
        },
        "network": {
            "allowed": False,
            "attempts": {
                "urllib_urlopen": 0,
                "socket_create_connection": 0,
                "socket_connect": 0,
                "torch_hub_load": 0,
                "torch_hub_load_state_dict_from_url": 0,
            },
        },
    }
    analyzer._validate_cpu_model_load(value)
    changed = copy.deepcopy(value)
    changed["extra"] = True
    with pytest.raises(ValueError, match="model-load"):
        analyzer._validate_cpu_model_load(changed)
    changed = copy.deepcopy(value)
    changed["model"]["class"] = "other.Model"
    with pytest.raises(ValueError, match="model-load"):
        analyzer._validate_cpu_model_load(changed)
    mutations = (
        ("network", "attempts", "urllib_urlopen", False),
        ("network", "allowed", 0),
        ("load", "strict", 1),
        ("model", "eval", 1),
        ("config", "original_resolution", 1),
    )
    for mutation in mutations:
        changed = copy.deepcopy(value)
        target: Any = changed
        for key in mutation[:-2]:
            target = target[key]
        target[mutation[-2]] = mutation[-1]
        with pytest.raises(ValueError, match="model-load"):
            analyzer._validate_cpu_model_load(changed)


def test_cpu_preflight_balanced_golden_is_exact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch

    repo_root = Path(__file__).resolve().parents[1]
    image_path = repo_root / runner.CPU_GOLDEN_INPUT_PATH
    preprocess = analyzer.legacy_audit.preprocess_image(
        image_path,
        torch_module=torch,
    ).audit
    artifacts = {
        "patch_features": {
            "array_sha256": runner.CPU_GOLDEN_PATCH_ARRAY_SHA256,
            "file_sha256": runner.CPU_GOLDEN_PATCH_FILE_SHA256,
            "shape": [4, 1096],
            "dtype": "float32",
            "nbytes": 4 * 1096 * 4,
            "file_bytes": 17664,
        },
        "feature": {
            "array_sha256": runner.CPU_GOLDEN_FEATURE_ARRAY_SHA256,
            "file_sha256": runner.CPU_GOLDEN_FEATURE_FILE_SHA256,
            "shape": [1096],
            "dtype": "float32",
            "nbytes": 1096 * 4,
            "file_bytes": 4512,
        },
        "attention": {
            "array_sha256": runner.CPU_GOLDEN_ATTENTION_ARRAY_SHA256,
            "file_sha256": runner.CPU_GOLDEN_ATTENTION_FILE_SHA256,
            "shape": [12, 4],
            "dtype": "float32",
            "nbytes": 12 * 4 * 4,
            "file_bytes": 320,
        },
    }
    manual = {
        "raw_logit": runner.CPU_GOLDEN_RAW_LOGIT,
        "probability": runner.CPU_GOLDEN_PROBABILITY,
        "ai_score": runner.CPU_GOLDEN_PROBABILITY,
        "classification_decision": False,
        "model_forward_calls": 1,
        "to_kv_hook_calls": 1,
        "attention_hook_calls": 1,
        "norm_hook_calls": 1,
        "official_attention_exact_match": True,
        "official_aggregated_exact_match": True,
        "official_feature_exact_match": True,
        "official_logit_exact_match": True,
        "official_probability_exact_match": True,
        "sca_replay": True,
        "norm_replay": True,
        "complete_mlp_replay": True,
    }
    balanced = {
        "sample_id": runner.CPU_GOLDEN_SAMPLE_ID,
        "input_path": runner.CPU_GOLDEN_INPUT_PATH,
        "image_sha256": runner.CPU_GOLDEN_IMAGE_SHA256,
        "input_width": 640,
        "input_height": 640,
        "preprocess": preprocess,
        **artifacts,
        "raw_logit": runner.CPU_GOLDEN_RAW_LOGIT,
        "probability": runner.CPU_GOLDEN_PROBABILITY,
        "ai_score": runner.CPU_GOLDEN_PROBABILITY,
        "classification_decision": False,
        "manual_replay": manual,
        "peak_cuda_memory_bytes": 0,
        "repeat_patch_features_file_sha256": (runner.CPU_GOLDEN_PATCH_FILE_SHA256),
        "repeat_feature_file_sha256": (runner.CPU_GOLDEN_FEATURE_FILE_SHA256),
        "repeat_attention_file_sha256": (runner.CPU_GOLDEN_ATTENTION_FILE_SHA256),
        "repeat_raw_logit": runner.CPU_GOLDEN_RAW_LOGIT,
        "repeat_probability": runner.CPU_GOLDEN_PROBABILITY,
        "repeat_byte_exact": True,
    }
    preprocess_gate = {
        "status": "passed",
        "official_transform": "test",
        "cases": [],
    }
    source: dict[str, Any] = {"source": True}
    assets: dict[str, Any] = {"assets": True}
    report = {
        "schema_version": analyzer.EXPECTED_CPU_PREFLIGHT_SCHEMA,
        "status": "passed",
        "source": source,
        "assets": assets,
        "model_load": {},
        "runtime": {},
        "official_preprocess_equivalence": preprocess_gate,
        "official_golden": {},
        "balanced_golden": balanced,
        "cuda_used": False,
        "cuda_tensor_operations": False,
        "cuda_initialized_before_cpu_model_load": False,
        "cuda_initialized_after_cpu_forwards": False,
        "dataset_manifest_loaded": False,
    }
    wrapper = {
        "performed_before_dataset_and_accelerator_configuration": True,
        "report": report,
    }
    monkeypatch.setattr(
        runner,
        "_validate_preflight_report",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        analyzer,
        "_validate_runtime_contract",
        lambda *_args, **_kwargs: {"device": "cpu"},
    )
    monkeypatch.setattr(
        analyzer,
        "_validate_cpu_model_load",
        lambda _value: None,
    )
    monkeypatch.setattr(
        analyzer,
        "_validate_cpu_official_golden",
        lambda _value, assets: {},
    )
    monkeypatch.setattr(
        analyzer.legacy,
        "validate_official_preprocess_equivalence",
        lambda: preprocess_gate,
    )
    analyzer._validate_cpu_preflight(
        wrapper,
        repo_root=repo_root,
        source=source,
        assets=assets,
    )
    for mutation in (
        "extra",
        "repeat",
        "hook_bool",
        "peak_bool",
        "artifact_bool",
    ):
        changed = copy.deepcopy(wrapper)
        if mutation == "extra":
            changed["report"]["balanced_golden"]["extra"] = True
        elif mutation == "repeat":
            changed["report"]["balanced_golden"]["repeat_feature_file_sha256"] = (
                "0" * 64
            )
        elif mutation == "hook_bool":
            changed["report"]["balanced_golden"]["manual_replay"][
                "model_forward_calls"
            ] = True
        elif mutation == "peak_bool":
            changed["report"]["balanced_golden"]["peak_cuda_memory_bytes"] = False
        else:
            changed["report"]["balanced_golden"]["feature"]["finite"] = 1
        with pytest.raises(ValueError, match="Balanced golden"):
            analyzer._validate_cpu_preflight(
                changed,
                repo_root=repo_root,
                source=source,
                assets=assets,
            )


def test_persisted_patch_and_feature_replays(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch

    row, sample = _artifact_row(tmp_path)

    class FakeModel:
        def parameters(self) -> tuple[Any, ...]:
            return ()

        def cls_head(self, value: Any) -> Any:
            return torch.zeros((value.shape[0], 1), dtype=torch.float32)

    monkeypatch.setattr(
        analyzer,
        "_configure_exact_recorded_runtime",
        lambda **_kwargs: (torch.device("cpu"), {"device": "cpu"}),
    )
    monkeypatch.setattr(
        analyzer,
        "_load_independent_model",
        lambda **_kwargs: (
            FakeModel(),
            {"commit": analyzer.EXPECTED_FROZEN_SOURCE_COMMIT},
            {"items_sha256": analyzer.EXPECTED_CHECKPOINT_SCHEMA_SHA256},
            {"weights_only": True},
        ),
    )
    monkeypatch.setattr(
        analyzer.legacy_audit,
        "_replay_patch_artifact",
        lambda _model, _patch: SimpleNamespace(
            normalized_feature=torch.from_numpy(sample.feature.array),
            attention=torch.from_numpy(sample.attention.array),
            raw_logit=torch.tensor(0.0, dtype=torch.float32),
        ),
    )
    report = analyzer.replay_persisted_artifacts(
        latest_results=[row],
        artifacts={"sample": sample},
        source_root=tmp_path,
        checkpoint_path=tmp_path / "checkpoint",
        device_text="cpu",
        recorded_runtime={},
    )
    assert report["images_replayed"] == 1
    assert report["patch_sca_norm_complete_mlp_replays"] == 1
    assert report["normalized_feature_complete_mlp_replays"] == 1
    assert not any(report["maximum_absolute_differences"].values())
    bad_feature = sample.feature.array.copy()
    bad_feature[0] = np.nan
    monkeypatch.setattr(
        analyzer.legacy_audit,
        "_replay_patch_artifact",
        lambda _model, _patch: SimpleNamespace(
            normalized_feature=torch.from_numpy(bad_feature),
            attention=torch.from_numpy(sample.attention.array),
            raw_logit=torch.tensor(0.0, dtype=torch.float32),
        ),
    )
    with pytest.raises(ValueError, match="finite"):
        analyzer.replay_persisted_artifacts(
            latest_results=[row],
            artifacts={"sample": sample},
            source_root=tmp_path,
            checkpoint_path=tmp_path / "checkpoint",
            device_text="cpu",
            recorded_runtime={},
        )
    monkeypatch.setattr(
        analyzer.legacy_audit,
        "_replay_patch_artifact",
        lambda _model, _patch: SimpleNamespace(
            normalized_feature=torch.from_numpy(sample.feature.array),
            attention=torch.from_numpy(sample.attention.array),
            raw_logit=torch.tensor(float("nan"), dtype=torch.float32),
        ),
    )
    with pytest.raises(ValueError, match="finite"):
        analyzer.replay_persisted_artifacts(
            latest_results=[row],
            artifacts={"sample": sample},
            source_root=tmp_path,
            checkpoint_path=tmp_path / "checkpoint",
            device_text="cpu",
            recorded_runtime={},
        )


def test_fresh_full_model_replay_checks_every_stage_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch

    monkeypatch.setattr(analyzer, "FORMAL_IMAGES", 1)
    sample_id = "sample"
    input_path = tmp_path / "input.jpg"
    input_path.write_bytes(b"canonical")
    canonical = {
        "sample_id": sample_id,
        "canonical_path": "input.jpg",
        "canonical_sha256": sha256_file(input_path),
    }
    row, sample = _artifact_row(tmp_path, sample_id=sample_id)
    preprocess = {"geometry": {"effective_patch_count": 4}}
    row["preprocess"] = preprocess

    class FakeModel:
        def parameters(self) -> tuple[Any, ...]:
            return ()

    monkeypatch.setattr(
        analyzer,
        "_configure_exact_recorded_runtime",
        lambda **_kwargs: (torch.device("cpu"), {"device": "cpu"}),
    )
    monkeypatch.setattr(
        analyzer,
        "_load_independent_model",
        lambda **_kwargs: (
            FakeModel(),
            {"commit": analyzer.EXPECTED_FROZEN_SOURCE_COMMIT},
            {"items_sha256": analyzer.EXPECTED_CHECKPOINT_SCHEMA_SHA256},
            {"weights_only": True},
        ),
    )
    monkeypatch.setattr(
        analyzer.legacy_audit,
        "preprocess_image",
        lambda _path, torch_module: SimpleNamespace(
            tensor=torch.zeros((3, 224, 224), dtype=torch.float32),
            audit=preprocess,
        ),
    )
    monkeypatch.setattr(
        analyzer,
        "_independent_visibility_diagnostic",
        lambda _row, repo_root: {},
    )
    monkeypatch.setattr(
        analyzer.legacy_audit,
        "_forward_with_evidence",
        lambda *_args, **_kwargs: SimpleNamespace(
            patch_features=torch.from_numpy(sample.patch_features.array),
            normalized_feature=torch.from_numpy(sample.feature.array),
            attention=torch.from_numpy(sample.attention.array),
            raw_logit=torch.tensor(0.0, dtype=torch.float32),
        ),
    )
    source_root = tmp_path / "source"
    source_root.mkdir()
    checkpoint = tmp_path / "checkpoint"
    checkpoint.write_bytes(b"checkpoint")
    bundle = SimpleNamespace(
        selected=(canonical,),
        latest_results=(row,),
        immutable={
            "runtime": {"device": "cpu"},
            "source": {
                "root": str(source_root),
                "commit": analyzer.EXPECTED_FROZEN_SOURCE_COMMIT,
            },
            "assets": {"checkpoint": {"path": str(checkpoint)}},
        },
        release=SimpleNamespace(repo_root=tmp_path),
        artifacts={sample_id: sample},
    )
    report = analyzer.replay_model(
        bundle,
        source_root=source_root,
        checkpoint_path=checkpoint,
        device_text="cpu",
    )
    assert report["images_replayed"] == 1
    assert report["full_fft_vit_srs_sca_norm_mlp_forward_per_input"] is True
    assert report["persisted_artifact_only_replay"] is False
    assert not any(report["maximum_absolute_differences"].values())
    bad_attention = sample.attention.array.copy()
    bad_attention[0, 0] = np.nan
    monkeypatch.setattr(
        analyzer.legacy_audit,
        "_forward_with_evidence",
        lambda *_args, **_kwargs: SimpleNamespace(
            patch_features=torch.from_numpy(sample.patch_features.array),
            normalized_feature=torch.from_numpy(sample.feature.array),
            attention=torch.from_numpy(bad_attention),
            raw_logit=torch.tensor(0.0, dtype=torch.float32),
        ),
    )
    with pytest.raises(ValueError, match="finite"):
        analyzer.replay_model(
            bundle,
            source_root=source_root,
            checkpoint_path=checkpoint,
            device_text="cpu",
        )


def test_feature_head_near_threshold_decision_must_match(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch

    row, sample = _artifact_row(tmp_path)
    row.update(
        _score_fields(
            raw_logit=-1e-7,
            probability=0.5,
        )
    )

    class FakeModel:
        def parameters(self) -> tuple[Any, ...]:
            return ()

        def cls_head(self, value: Any) -> Any:
            return torch.full(
                (value.shape[0], 1),
                1e-7,
                dtype=torch.float32,
            )

    monkeypatch.setattr(
        analyzer,
        "_configure_exact_recorded_runtime",
        lambda **_kwargs: (torch.device("cpu"), {"device": "cpu"}),
    )
    monkeypatch.setattr(
        analyzer,
        "_load_independent_model",
        lambda **_kwargs: (
            FakeModel(),
            {"commit": analyzer.EXPECTED_FROZEN_SOURCE_COMMIT},
            {"items_sha256": analyzer.EXPECTED_CHECKPOINT_SCHEMA_SHA256},
            {"weights_only": True},
        ),
    )
    monkeypatch.setattr(
        analyzer.legacy_audit,
        "_replay_patch_artifact",
        lambda _model, _patch: SimpleNamespace(
            normalized_feature=torch.from_numpy(sample.feature.array),
            attention=torch.from_numpy(sample.attention.array),
            raw_logit=torch.tensor(-1e-7, dtype=torch.float32),
        ),
    )
    monkeypatch.setattr(
        analyzer.legacy_audit,
        "_float32_sigmoid",
        lambda *_args: pytest.fail("CPU sigmoid helper must not be used"),
    )
    with pytest.raises(ValueError, match="decision"):
        analyzer.replay_persisted_artifacts(
            latest_results=[row],
            artifacts={"sample": sample},
            source_root=tmp_path,
            checkpoint_path=tmp_path / "checkpoint",
            device_text="cpu",
            recorded_runtime={},
        )


def test_output_collision_and_run_inventory_poisoning_fail(
    tmp_path: Path,
) -> None:
    protected = tmp_path / "manifest.json"
    protected.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="collide"):
        analyzer._validate_output_targets(
            {"metrics": tmp_path / "x", "audit": tmp_path / "x"},
            protected_files=[],
            protected_dirs=[],
        )
    with pytest.raises(ValueError, match="overwrite"):
        analyzer._validate_output_targets(
            {"metrics": protected},
            protected_files=[protected],
            protected_dirs=[],
        )
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    bundle = SimpleNamespace(run_dir=run_dir)
    with pytest.raises(ValueError, match="another run"):
        analyzer._validate_formal_output_locations(
            bundle=bundle,
            metrics_output_path=run_dir / "wrong.json",
            audit_output_path=None,
        )
    analyzer._validate_formal_output_locations(
        bundle=bundle,
        metrics_output_path=run_dir / "balanced250_metrics.json",
        audit_output_path=run_dir / "independent_audit.json",
    )
    third_run = tmp_path / "third-run"
    third_run.mkdir()
    with pytest.raises(ValueError, match="another run"):
        analyzer._validate_formal_output_locations(
            bundle=bundle,
            metrics_output_path=third_run / "manifest.json",
            audit_output_path=None,
        )
    analyzer._validate_formal_output_locations(
        bundle=bundle,
        metrics_output_path=tmp_path / "_reports" / "metrics.json",
        audit_output_path=None,
    )
    reports_named_run = tmp_path / "reports"
    reports_named_run.mkdir()
    with pytest.raises(ValueError, match="another run"):
        analyzer._validate_formal_output_locations(
            bundle=bundle,
            metrics_output_path=reports_named_run / "manifest.json",
            audit_output_path=None,
        )
    outside = tmp_path / "outside"
    outside.mkdir()
    linked = tmp_path / "linked-output"
    linked.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        analyzer._validate_formal_output_locations(
            bundle=bundle,
            metrics_output_path=linked / "metrics.json",
            audit_output_path=None,
        )
    with pytest.raises(ValueError, match="results root"):
        analyzer._validate_formal_output_locations(
            bundle=bundle,
            metrics_output_path=tmp_path.parent / "external-metrics.json",
            audit_output_path=None,
        )


def test_verified_json_write_and_tamper_detection(tmp_path: Path) -> None:
    path = tmp_path / "report.json"
    value = {"schema_version": "test", "ok": True}
    analyzer._write_json_verified(path, value, label="report")
    expected = analyzer._json_artifact_sha256(value)
    analyzer._verify_json_artifact(
        path,
        expected_sha256=expected,
        label="report",
    )
    path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="changed"):
        analyzer._verify_json_artifact(
            path,
            expected_sha256=expected,
            label="report",
        )


def test_metrics_require_frozen_formal_parameters() -> None:
    bundle = SimpleNamespace(selected=())
    with pytest.raises(ValueError, match="1,775"):
        analyzer.recompute_metrics(bundle)


def test_metrics_use_only_shared_balanced250_t1_summary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(analyzer, "FORMAL_IMAGES", 1)
    observed: dict[str, Any] = {}

    def fake_summary(*args: Any, **kwargs: Any) -> dict[str, Any]:
        observed["args"] = args
        observed["kwargs"] = kwargs
        return {
            "schema_version": analyzer.METRICS_SCHEMA_VERSION,
            "coverage": {"is_complete": True},
            "bootstrap": {
                "iterations": analyzer.BOOTSTRAP_ITERATIONS,
                "seed": analyzer.BOOTSTRAP_SEED,
            },
        }

    monkeypatch.setattr(analyzer, "summarize_balanced250_t1", fake_summary)
    contract = SimpleNamespace()
    bundle = SimpleNamespace(
        selected=({"sample_id": "sample"},),
        release=SimpleNamespace(
            inputs=("inputs",),
            panel=("panel",),
            source_pairs=("pairs",),
        ),
        latest_results=({"sample_id": "sample"},),
        run_id="formal",
        fingerprint="a" * 64,
        contract=contract,
    )
    metrics = analyzer.recompute_metrics(bundle)
    assert metrics["schema_version"] == analyzer.METRICS_SCHEMA_VERSION
    assert observed["kwargs"]["run_dataset_contract"] is contract
    assert observed["kwargs"]["iterations"] == analyzer.BOOTSTRAP_ITERATIONS


def test_parser_defaults_to_frozen_assets() -> None:
    args = analyzer._build_parser().parse_args([])
    assert args.source_root == runner.DEFAULT_SOURCE_ROOT
    assert args.checkpoint == runner.DEFAULT_CHECKPOINT
    assert args.golden_root == runner.DEFAULT_GOLDEN_ROOT
    assert args.bootstrap_iterations == analyzer.BOOTSTRAP_ITERATIONS
    assert args.bootstrap_seed == analyzer.BOOTSTRAP_SEED


def test_smoke_comparison_default_output_is_order_bound(
    tmp_path: Path,
) -> None:
    first = analyzer._resolve_smoke_comparison_output(
        requested_output=None,
        repo_root=tmp_path,
        results_dir=tmp_path,
        reference_run_id="a",
        replay_run_id="b",
    )
    second = analyzer._resolve_smoke_comparison_output(
        requested_output=None,
        repo_root=tmp_path,
        results_dir=tmp_path,
        reference_run_id="b",
        replay_run_id="a",
    )
    assert first != second
    assert first.parent == tmp_path / "_reports"
    assert hashlib.sha256(stable_json(["a", "b"]).encode()).hexdigest() in (first.name)
