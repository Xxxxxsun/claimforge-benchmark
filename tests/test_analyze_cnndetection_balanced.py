from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from eval.opensource import analyze_cnndetection_balanced as analyzer
from eval.opensource.common import stable_json


def _save_feature(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        np.save(handle, value, allow_pickle=False)


def _row(root: Path, sample_id: str, score: float = 0.75) -> dict:
    feature_path = root / "features" / f"{sample_id}.npy"
    feature = np.arange(2048, dtype=np.float32)
    _save_feature(feature_path, feature)
    logit = float(np.log(score / (1.0 - score)))
    aliases = {
        "raw_logit": logit,
        "probability": score,
        "ai_score": score,
        "score": score,
        "threshold": 0.5,
        "threshold_operator": ">",
        "decision": True,
        "semantics": "official_float32_sigmoid_uncalibrated_fake_score",
    }
    return {
        "schema_version": "opensource_result_v2",
        "run_id": "run-a",
        "run_manifest_fingerprint": "a" * 64,
        "config_fingerprint": "a" * 64,
        "dataset_id": "dataset",
        "id": sample_id,
        "sample_id": sample_id,
        "rank": 0,
        "condition": "real",
        "condition_family": "real",
        "manipulation_scope": "authentic",
        "normalized_task_id": "normalized",
        "task_id": "task",
        "kind": "real",
        "label": 0,
        "domain": "lodging",
        "gt_mask_kind": "all_zero",
        "input_path": f"inputs/{sample_id}.jpg",
        "input_sha256": "b" * 64,
        "input_width": 640,
        "input_height": 480,
        "status": "ok",
        "valid_for_metrics": True,
        "completed_at": "2026-07-26T00:00:00+00:00",
        "model": "CNNDetection",
        "model_slug": "cnndetection_blur_jpg_prob0_1",
        "preprocess_profile": (
            "official_recommended_native_rgb_no_resize_no_crop"
        ),
        "checkpoint_id": "CNNDetection-BlurJPEG0.1@official-dropbox",
        "task_scope": {
            "valid_for_t1": True,
            "valid_for_t2": False,
            "native_dense_output": False,
        },
        "edit_visibility": "not_applicable",
        "edit_visible_gt_fraction": None,
        "edit_visibility_evidence": {
            "basis": "authentic_input",
            "preprocess_profile": (
                "official_recommended_native_rgb_no_resize_no_crop"
            ),
        },
        "preprocess": {"profile": "fixture"},
        "preprocess_latency_ms": 1.25,
        "cnndetection_feature_path": (
            feature_path.relative_to(root).as_posix()
        ),
        "cnndetection_feature_sha256": hashlib.sha256(
            feature_path.read_bytes()
        ).hexdigest(),
        "cnndetection_feature_bytes": feature_path.stat().st_size,
        "cnndetection_feature_shape": [2048],
        "cnndetection_feature_dtype": "float32",
        "cnndetection_feature_semantics": (
            "official_fc_input_after_adaptive_global_average_pool"
        ),
        "raw_logit": logit,
        "probability": score,
        "ai_score": score,
        "score": score,
        "score_semantics": (
            "official_float32_sigmoid_uncalibrated_fake_score"
        ),
        "calibrated_probability": False,
        "classification_decision": True,
        "classification_threshold": 0.5,
        "classification_threshold_operator": ">",
        "classification": dict(aliases),
        "t1": {
            **aliases,
            "policy": (
                "official_CNNDetection_float32_sigmoid_strict_gt_0_5"
            ),
        },
        "manual_replay": {
            "raw_logit": logit,
            "probability": score,
            "ai_score": score,
            "classification_decision": True,
            "model_forward_calls": 1,
            "fc_hook_calls": 1,
            "official_logit_exact_match": True,
            "official_score_exact_match": True,
        },
        "latency_ms": 2.5,
        "peak_cuda_memory_bytes": 1234,
    }


def test_strict_jsonl_rejects_noncanonical_and_missing_newline(tmp_path: Path):
    path = tmp_path / "rows.jsonl"
    path.write_text('{"z": 1, "a": 2}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="not canonical"):
        analyzer._read_jsonl_strict(path, "rows")
    path.write_text(stable_json({"a": 2, "z": 1}), encoding="utf-8")
    with pytest.raises(ValueError, match="final newline"):
        analyzer._read_jsonl_strict(path, "rows")


def test_feature_inventory_validates_hash_shape_dtype_and_exact_files(
    tmp_path: Path,
):
    row = _row(tmp_path, "sample")
    artifacts = analyzer.validate_feature_inventory(
        latest_results=[row],
        repo_root=tmp_path,
        feature_dir=tmp_path / "features",
    )
    assert artifacts["sample"].array.shape == (2048,)
    assert artifacts["sample"].array.dtype == np.float32

    (tmp_path / "features" / "extra.npy").write_bytes(b"extra")
    with pytest.raises(ValueError, match="feature inventory mismatch"):
        analyzer.validate_feature_inventory(
            latest_results=[row],
            repo_root=tmp_path,
            feature_dir=tmp_path / "features",
        )


def test_feature_inventory_rejects_tampered_artifact(tmp_path: Path):
    row = _row(tmp_path, "sample")
    path = tmp_path / row["cnndetection_feature_path"]
    _save_feature(path, np.ones(2048, dtype=np.float32))
    with pytest.raises(ValueError, match="artifact hash mismatch"):
        analyzer.validate_feature_inventory(
            latest_results=[row],
            repo_root=tmp_path,
            feature_dir=tmp_path / "features",
        )


def test_feature_inventory_requires_persisted_byte_size(tmp_path: Path):
    row = _row(tmp_path, "sample")
    del row["cnndetection_feature_bytes"]
    with pytest.raises(ValueError, match="byte-size metadata"):
        analyzer.validate_feature_inventory(
            latest_results=[row],
            repo_root=tmp_path,
            feature_dir=tmp_path / "features",
        )


def test_score_payload_rejects_nonfinite_and_decision_drift(tmp_path: Path):
    row = _row(tmp_path, "sample")
    broken = copy.deepcopy(row)
    broken["ai_score"] = float("nan")
    with pytest.raises(ValueError, match="not finite"):
        analyzer._validate_score_payload(broken, sample_id="sample")
    broken = copy.deepcopy(row)
    broken["classification_decision"] = False
    with pytest.raises(ValueError, match="decision"):
        analyzer._validate_score_payload(broken, sample_id="sample")


def test_recursive_t2_claim_gate_allows_null_declaration_only():
    analyzer._reject_t2_claims(
        {
            "task_scope": {
                "valid_for_t2": False,
                "localization_output": None,
                "joint_output": None,
            }
        },
        "payload",
    )
    for payload in (
        {"nested": [{"score_map_sha256": "a" * 64}]},
        {"localization_output": {"shape": [1, 1]}},
        {"joint_score": 0.5},
    ):
        with pytest.raises(ValueError, match="unsupported"):
            analyzer._reject_t2_claims(payload, "payload")


def test_runtime_summary_requires_every_formal_field():
    coverage = {"is_complete": True}
    summary = {
        "schema_version": analyzer.balanced_runner.RUNTIME_SUMMARY_SCHEMA,
        "summary_kind": "runtime_coverage_only",
        "scientific_metrics": None,
        "scientific_metrics_owner": "analyze_cnndetection_balanced.py",
        "run_id": "run",
        "run_manifest_fingerprint": "a" * 64,
        "status": "complete",
        "mode": "formal",
        "model": analyzer.legacy_runner.MODEL_NAME,
        "model_slug": analyzer.legacy_runner.MODEL_SLUG,
        "score_spec": analyzer.balanced_runner.SCORE_SPEC.as_dict(),
        "dataset_contract": {"contract": True},
        "coverage": coverage,
        "generated_at": "2026-07-26T00:00:00+00:00",
    }
    contract = type(
        "Contract",
        (),
        {"as_dict": lambda self: {"contract": True}},
    )()
    analyzer._validate_runtime_summary(
        summary=summary,
        run_id="run",
        fingerprint="a" * 64,
        mode="formal",
        contract=contract,
        coverage=coverage,
    )
    del summary["coverage"]
    with pytest.raises(ValueError, match="coverage mismatch"):
        analyzer._validate_runtime_summary(
            summary=summary,
            run_id="run",
            fingerprint="a" * 64,
            mode="formal",
            contract=contract,
            coverage=coverage,
        )


def test_exact_projection_ignores_run_identity_and_requires_feature_bytes(
    tmp_path: Path,
):
    left_root = tmp_path / "left"
    right_root = tmp_path / "right"
    left = _row(left_root, "sample")
    right = _row(right_root, "sample")
    right["run_id"] = "run-b"
    right["run_manifest_fingerprint"] = "c" * 64
    right["config_fingerprint"] = "c" * 64
    right["completed_at"] = "2026-07-26T01:00:00+00:00"
    right["preprocess_latency_ms"] = 99.0
    right["latency_ms"] = 100.0
    right["peak_cuda_memory_bytes"] = 9999
    left_features = analyzer.validate_feature_inventory(
        latest_results=[left],
        repo_root=left_root,
        feature_dir=left_root / "features",
    )
    right_features = analyzer.validate_feature_inventory(
        latest_results=[right],
        repo_root=right_root,
        feature_dir=right_root / "features",
    )
    report = analyzer.compare_computational_results(
        reference_rows=[left],
        replay_rows=[right],
        reference_features=left_features,
        replay_features=right_features,
        exact=True,
    )
    assert report["images_compared"] == 1
    assert report["exact_computational_projection"] is True

    changed = np.arange(2048, dtype=np.float32)
    changed[0] = 999
    right_path = right_root / right["cnndetection_feature_path"]
    _save_feature(right_path, changed)
    right["cnndetection_feature_sha256"] = hashlib.sha256(
        right_path.read_bytes()
    ).hexdigest()
    right["cnndetection_feature_bytes"] = right_path.stat().st_size
    changed_features = analyzer.validate_feature_inventory(
        latest_results=[right],
        repo_root=right_root,
        feature_dir=right_root / "features",
    )
    with pytest.raises(ValueError, match="feature bytes differ"):
        analyzer.compare_computational_results(
            reference_rows=[left],
            replay_rows=[right],
            reference_features=left_features,
            replay_features=changed_features,
            exact=True,
        )


def test_exact_smoke_comparison_rejects_any_extra_field_difference(
    tmp_path: Path,
):
    left_root = tmp_path / "left"
    right_root = tmp_path / "right"
    left = _row(left_root, "sample")
    right = _row(right_root, "sample")
    right["run_id"] = "run-b"
    right["run_manifest_fingerprint"] = "c" * 64
    right["config_fingerprint"] = "c" * 64
    right["unexpected_computation"] = {"value": 1}
    left_features = analyzer.validate_feature_inventory(
        latest_results=[left],
        repo_root=left_root,
        feature_dir=left_root / "features",
    )
    right_features = analyzer.validate_feature_inventory(
        latest_results=[right],
        repo_root=right_root,
        feature_dir=right_root / "features",
    )
    with pytest.raises(ValueError, match="full computational row differs"):
        analyzer.compare_computational_results(
            reference_rows=[left],
            replay_rows=[right],
            reference_features=left_features,
            replay_features=right_features,
            exact=True,
        )


def test_projection_rejects_missing_duplicate_and_nonfinite(tmp_path: Path):
    left_root = tmp_path / "left"
    right_root = tmp_path / "right"
    left = _row(left_root, "sample")
    right = _row(right_root, "sample")
    left_features = analyzer.validate_feature_inventory(
        latest_results=[left],
        repo_root=left_root,
        feature_dir=left_root / "features",
    )
    right_features = analyzer.validate_feature_inventory(
        latest_results=[right],
        repo_root=right_root,
        feature_dir=right_root / "features",
    )
    with pytest.raises(ValueError, match="duplicate sample_id"):
        analyzer.compare_computational_results(
            reference_rows=[left],
            replay_rows=[right, right],
            reference_features=left_features,
            replay_features=right_features,
            exact=True,
        )
    with pytest.raises(ValueError, match="coverage differs"):
        analyzer.compare_computational_results(
            reference_rows=[left],
            replay_rows=[{**right, "sample_id": "other", "id": "other"}],
            reference_features=left_features,
            replay_features=right_features,
            exact=True,
        )


def _manifest_fixture(tmp_path: Path) -> dict:
    sources = {}
    for relative in analyzer.balanced_runner.ADAPTER_SOURCE_PATHS:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative, encoding="utf-8")
        sources[relative] = {
            "path": relative,
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    source = {
        "repo_url": analyzer.legacy_runner.MODEL_REPO_URL,
        "root": str((tmp_path / "source").resolve()),
        "commit": analyzer.legacy_runner.MODEL_SOURCE_COMMIT,
        "paper_era_stable_commit": (
            analyzer.legacy_runner.PAPER_ERA_STABLE_COMMIT
        ),
        "tracked_dirty": False,
        "source_files": analyzer.legacy_runner.SOURCE_FILES,
        "core_inference_byte_identical_to_paper_era_commit": True,
    }
    checkpoint = analyzer.legacy_runner.CHECKPOINT
    asset = {
        **checkpoint,
        "path": str((tmp_path / checkpoint["filename"]).resolve()),
        "serialization_safety": {
            "weights_only": True,
            "weights_only_load_succeeded": True,
            "static_unsafe_global_scan": {
                "supported": True,
                "unsafe_globals": [],
            },
            "unrestricted_pickle_used": False,
        },
        "schema": {
            "outer_type": "dict",
            "outer_keys": list(checkpoint["outer_keys"]),
            "model_type": "OrderedDict",
            "state_entries": checkpoint["state_entries"],
            "state_elements": checkpoint["state_elements"],
            "state_payload_sha256": checkpoint["state_payload_sha256"],
            "conv1_weight_shape": [64, 3, 7, 7],
            "fc_weight_shape": [
                1,
                analyzer.legacy_runner.FEATURE_DIMENSION,
            ],
            "fc_bias_shape": [1],
            "optimizer_state_entries": checkpoint[
                "optimizer_state_entries"
            ],
            "optimizer_param_groups": checkpoint[
                "optimizer_param_groups"
            ],
            "total_steps": checkpoint["total_steps"],
            "strict_model_load": True,
        },
    }
    runtime = {
        "device": "cpu",
        "python": "3.12.0",
        "platform": "fixture",
        "torch": "2.0",
        "torchvision": "0.1",
        "pillow": "10",
        "numpy": "2",
        "seed": analyzer.legacy_runner.MODEL_SEED,
        "dtype": "float32",
        "batch_size": 1,
        "autocast": False,
        "deterministic_algorithms_enabled": True,
        "deterministic_algorithms_warn_only": False,
        "cublas_workspace_config": ":4096:8",
        "cudnn": {
            "enabled": True,
            "benchmark": False,
            "deterministic": True,
            "allow_tf32": False,
        },
        "matmul_allow_tf32": False,
    }
    golden_cases = []
    for profile in (
        analyzer.legacy_runner.PRIMARY_PROFILE,
        analyzer.legacy_runner.PAPER_CROP_PROFILE,
    ):
        for filename in ("real.png", "fake.png"):
            golden_cases.append(
                {
                    "profile": profile,
                    "filename": filename,
                    "input_sha256": "1" * 64,
                    "tensor_sha256": "2" * 64,
                    "raw_logit": 0.0,
                    "fake_score": 0.5,
                    "classification_decision": False,
                    "feature_sha256": "3" * 64,
                }
            )
    run_id = "formal"
    outputs = {
        "results_path": f"results/{run_id}/results.jsonl",
        "expected_inputs_path": (
            f"results/{run_id}/expected_inputs.jsonl"
        ),
        "summary_path": f"results/{run_id}/summary.json",
        "feature_dir": f"results/{run_id}/features",
    }
    immutable = {
        "schema_version": analyzer.balanced_runner.RUN_CONFIG_SCHEMA,
        "run_id": run_id,
        "mode": "formal",
        "adapter_sources": sources,
        "model": {
            "name": analyzer.legacy_runner.MODEL_NAME,
            "slug": analyzer.legacy_runner.MODEL_SLUG,
            "architecture": analyzer.legacy_runner.MODEL_ARCH,
            "repository": analyzer.legacy_runner.MODEL_REPO_URL,
            "source_commit": analyzer.legacy_runner.MODEL_SOURCE_COMMIT,
            "checkpoint_id": analyzer.legacy_runner.CHECKPOINT["id"],
            "checkpoint_sha256": analyzer.legacy_runner.CHECKPOINT["sha256"],
            "checkpoint_bytes": analyzer.legacy_runner.CHECKPOINT["bytes"],
        },
        "preprocess": {
            "profile": analyzer.legacy_runner.PRIMARY_PROFILE,
            "contract": analyzer.legacy_runner.PREPROCESS_PROFILES[
                analyzer.legacy_runner.PRIMARY_PROFILE
            ],
            "batch_size": 1,
            "test_time_blur_or_jpeg": False,
        },
        "score_spec": analyzer.balanced_runner.SCORE_SPEC.as_dict(),
        "task_scope": {
            "primary_task": "T1_whole_image_AIGC_detection",
            "valid_for_t1": True,
            "valid_for_t2": False,
            "localization_output": None,
        },
        "dataset_contract": {"fixture": True},
        "selected_rows_sha256": "4" * 64,
        "selected_ids_sha256": "5" * 64,
        "source": source,
        "asset": asset,
        "runtime": runtime,
        "cpu_preflight": {
            "performed_before_accelerator_configuration": True,
            "report": {
                "schema_version": "cnndetection_preflight_v1",
                "status": "passed",
                "source": source,
                "asset": asset,
                "runtime": runtime,
                "golden_reference_runtime": {},
                "golden_cases": golden_cases,
                "cuda_used": False,
                "mouse_inference_run": False,
            },
        },
        "artifact_contract": {
            "feature": {
                "format": "NumPy .npy, allow_pickle=False",
                "shape": [analyzer.legacy_runner.FEATURE_DIMENSION],
                "dtype": "float32",
                "semantics": (
                    "official_fc_input_after_adaptive_global_average_pool"
                ),
                "exact_fc_and_sigmoid_replay": True,
            },
        },
        "outputs": outputs,
    }
    fingerprint = hashlib.sha256(
        stable_json(immutable).encode("utf-8")
    ).hexdigest()
    return {
        "schema_version": analyzer.MANIFEST_SCHEMA_VERSION,
        "run_id": run_id,
        "status": "complete",
        "started_at": "2026-07-26T00:00:00+00:00",
        "completed_at": "2026-07-26T00:01:00+00:00",
        "fingerprint": fingerprint,
        "immutable": immutable,
        "dataset": {
            "contract": {"fixture": True},
            "manifest_path": "release/manifest.json",
            "manifest_sha256": "6" * 64,
            "expected_inputs_path": outputs["expected_inputs_path"],
            "expected_inputs_sha256": "7" * 64,
            "selected_images": 1775,
        },
        "outputs": {
            **outputs,
            "results_sha256": "8" * 64,
            "summary_sha256": "9" * 64,
            "feature_files": 1775,
        },
        "execution": {
            "new_successes": 1775,
            "resume_skips": 0,
            "new_errors": 0,
            "physical_result_rows": 1775,
            "latest_result_rows": 1775,
            "superseded_attempts": 0,
        },
    }


def _refingerprint(manifest: dict) -> None:
    manifest["fingerprint"] = hashlib.sha256(
        stable_json(manifest["immutable"]).encode("utf-8")
    ).hexdigest()


def test_manifest_fingerprint_and_immutable_are_fail_closed(tmp_path: Path):
    manifest = _manifest_fixture(tmp_path)
    actual, returned = analyzer._validate_manifest(
        manifest=manifest,
        repo_root=tmp_path,
        run_id="formal",
    )
    assert actual == manifest["fingerprint"]
    assert returned == manifest["immutable"]
    manifest["fingerprint"] = "0" * 64
    with pytest.raises(ValueError, match="does not bind"):
        analyzer._validate_manifest(
            manifest=manifest,
            repo_root=tmp_path,
            run_id="formal",
        )

    drifted = _manifest_fixture(tmp_path)
    drifted["immutable"]["task_scope"]["extra"] = True
    _refingerprint(drifted)
    with pytest.raises(ValueError, match="task_scope"):
        analyzer._validate_manifest(
            manifest=drifted,
            repo_root=tmp_path,
            run_id="formal",
        )


def test_adapter_sources_require_exact_keys_and_hash_metadata(tmp_path: Path):
    manifest = _manifest_fixture(tmp_path)
    sources = manifest["immutable"]["adapter_sources"]
    missing_key = next(iter(sources))
    del sources[missing_key]
    _refingerprint(manifest)
    with pytest.raises(ValueError, match="key set mismatch"):
        analyzer._validate_manifest(
            manifest=manifest,
            repo_root=tmp_path,
            run_id="formal",
        )

    manifest = _manifest_fixture(tmp_path)
    sources = manifest["immutable"]["adapter_sources"]
    sources[next(iter(sources))].pop("sha256")
    _refingerprint(manifest)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        analyzer._validate_manifest(
            manifest=manifest,
            repo_root=tmp_path,
            run_id="formal",
        )


def test_cli_defaults_to_formal_metrics_and_independent_audit(
    monkeypatch,
    tmp_path: Path,
    capsys,
):
    called = {}

    def fake_analyze(**kwargs):
        called.update(kwargs)
        return {"status": "replay_audit_passed"}

    monkeypatch.setattr(analyzer, "analyze", fake_analyze)
    assert analyzer.main(
        [
            "--repo-root",
            str(tmp_path),
            "--results-dir",
            "results",
            "--run-id",
            "formal",
            "--skip-model-replay",
        ]
    ) == 0
    assert called["metrics_output_path"].name == "balanced250_metrics.json"
    assert called["audit_output_path"].name == "independent_audit.json"
    assert called["replay"] is False
    assert json.loads(capsys.readouterr().out)["status"] == (
        "replay_audit_passed"
    )


def test_run_id_and_resolved_run_path_are_fail_closed(tmp_path: Path):
    results_root = tmp_path / "results"
    results_root.mkdir()
    with pytest.raises(ValueError, match="run-id"):
        analyzer._resolve_run_dir(results_root, "../escape")

    outside = tmp_path / "outside"
    outside.mkdir()
    (results_root / "linked").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="escapes results root"):
        analyzer._resolve_run_dir(results_root, "linked")


def test_cli_rejects_unsafe_comparison_run_id_before_dispatch(
    monkeypatch,
    tmp_path: Path,
):
    called = False

    def fake_compare(**_kwargs):
        nonlocal called
        called = True
        return {}

    monkeypatch.setattr(analyzer, "compare_smoke_runs", fake_compare)
    with pytest.raises(ValueError, match="run-id"):
        analyzer.main(
            [
                "--repo-root",
                str(tmp_path),
                "--results-dir",
                "results",
                "--run-id",
                "smoke-a",
                "--compare-smoke-run-id",
                "../escape",
            ]
        )
    assert called is False


def test_cli_smoke_comparison_writes_default_evidence(
    monkeypatch,
    tmp_path: Path,
    capsys,
):
    called = {}

    def fake_compare(**kwargs):
        called.update(kwargs)
        return {"status": "deterministic_smoke_comparison_passed"}

    monkeypatch.setattr(analyzer, "compare_smoke_runs", fake_compare)
    assert analyzer.main(
        [
            "--repo-root",
            str(tmp_path),
            "--results-dir",
            "results",
            "--run-id",
            "smoke-a",
            "--compare-smoke-run-id",
            "smoke-b",
        ]
    ) == 0
    assert called["reference_run_id"] == "smoke-a"
    assert called["replay_run_id"] == "smoke-b"
    assert called["output_path"].name == (
        "smoke-a__vs__smoke-b_comparison.json"
    )
    assert json.loads(capsys.readouterr().out)["status"] == (
        "deterministic_smoke_comparison_passed"
    )
