from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from eval.opensource import analyze_effort_balanced as analyzer
from eval.opensource.canonical_release import load_canonical_release
from eval.opensource.common import stable_json


REPO_ROOT = Path(__file__).resolve().parents[1]


def _runtime(device: str = "cpu") -> dict:
    return {
        "python": "3.12.3",
        "torch": analyzer.legacy.TORCH_VERSION,
        "torchvision": analyzer.legacy.TORCHVISION_VERSION,
        "transformers": analyzer.legacy.TRANSFORMERS_VERSION,
        "numpy": analyzer.legacy.NUMPY_VERSION,
        "opencv": analyzer.legacy.OPENCV_VERSION,
        "device": device,
        "cuda_available": False,
        "cuda_device_name": None if device == "cpu" else "fixture GPU",
        "cublas_workspace_config": ":4096:8",
        "deterministic_algorithms": True,
        "cudnn_enabled": True,
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
        "allow_tf32_matmul": False,
        "allow_tf32_cudnn": False,
        "float32_matmul_precision": "highest",
        "autocast": False,
        "model_dtype": "float32",
        "batch_size": 1,
        "cpu_threads": 16,
        "seed": 20260724,
    }


def _golden(device_family: str = "cpu") -> dict:
    cases = []
    for frozen in analyzer.legacy.GOLDEN_CASES:
        cases.append(
            {
                "path": frozen["path"],
                "input_sha256": frozen["sha256"],
                "preprocess": {},
                "logits": list(frozen[f"{device_family}_logits"]),
                "fake_probability": frozen[f"{device_family}_probability"],
                "feature_sha256": "a" * 64,
                "repeat_feature_max_abs_diff": 0.0,
                "repeat_logit_max_abs_diff": 0.0,
                "frozen_runtime_logit_max_abs_diff": 0.0,
                "frozen_runtime_probability_abs_diff": 0.0,
                "frozen_cpu_cuda_logit_max_abs_diff": 0.0,
            }
        )
    return {
        "status": "passed",
        "kind": ("repository_fixture_runtime_regression_not_author_published_golden"),
        "device_family": device_family,
        "runtime_abs_tolerance": analyzer.legacy.GOLDEN_RUNTIME_ABS_TOLERANCE,
        "cpu_cuda_abs_tolerance": (analyzer.legacy.GOLDEN_CPU_CUDA_ABS_TOLERANCE),
        "cases": cases,
        "mouse_model_scores_computed": 0,
    }


def _preflight(
    source: dict,
    assets: dict,
    model_audit: dict,
) -> dict:
    return {
        "performed_before_accelerator_configuration": True,
        "report": {
            "schema_version": analyzer.CPU_PREFLIGHT_SCHEMA,
            "status": "passed",
            "source": source,
            "assets": assets,
            "model_audit": model_audit,
            "runtime": _runtime(),
            "runtime_golden": _golden(),
            "accelerator_model_forwards": 0,
            "balanced250_model_scores_computed": 0,
            "cuda_initialized_before": False,
            "cuda_initialized_after": False,
        },
    }


def _score_payload(logits: np.ndarray) -> dict:
    logits = np.ascontiguousarray(logits, dtype=np.float32)
    score = analyzer._float32_softmax_class1(logits.tolist())
    margin = float(np.float32(logits[1] - logits[0]))
    decision = score > analyzer.legacy.CLASSIFICATION_THRESHOLD
    return {
        "class_logits": logits.tolist(),
        "raw_logit_margin": margin,
        "fake_probability": score,
        "probability": score,
        "ai_score": score,
        "score": score,
        "score_semantics": analyzer.legacy.SCORE_SEMANTICS,
        "classification_decision": decision,
        "classification_threshold": analyzer.legacy.CLASSIFICATION_THRESHOLD,
        "classification_threshold_operator": (
            analyzer.legacy.CLASSIFICATION_THRESHOLD_OPERATOR
        ),
        "classification": {
            "decision": decision,
            "threshold": analyzer.legacy.CLASSIFICATION_THRESHOLD,
            "operator": analyzer.legacy.CLASSIFICATION_THRESHOLD_OPERATOR,
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
        },
    }


def _save_npz(path: Path, feature: np.ndarray, logits: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        np.savez(
            handle,
            pooler_output=np.ascontiguousarray(feature, dtype=np.float32),
            class_logits=np.ascontiguousarray(logits, dtype=np.float32),
        )


def _artifact_row(
    repo_root: Path,
    artifact_dir: Path,
    *,
    sample_id: str = "sample",
    feature: np.ndarray | None = None,
    logits: np.ndarray | None = None,
) -> dict:
    feature_value = (
        np.arange(analyzer.FEATURE_SHAPE[0], dtype=np.float32)
        if feature is None
        else np.ascontiguousarray(feature, dtype=np.float32)
    )
    logits_value = (
        np.asarray([0.25, -0.5], dtype=np.float32)
        if logits is None
        else np.ascontiguousarray(logits, dtype=np.float32)
    )
    path = artifact_dir / f"{sample_id}.npz"
    _save_npz(path, feature_value, logits_value)
    assert path.stat().st_size == analyzer.NPZ_FILE_BYTES
    relative = path.relative_to(repo_root).as_posix()
    return {
        "sample_id": sample_id,
        "artifact_path": relative,
        "artifact_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "artifact_bytes": path.stat().st_size,
        "artifact_keys": ["pooler_output", "class_logits"],
        "artifact_paths": {"effort_npz": relative},
        "feature_shape": list(feature_value.shape),
        "feature_dtype": "float32",
        "feature_semantics": analyzer.legacy.FEATURE_SEMANTICS,
        "feature_array_sha256": analyzer._array_sha256(feature_value),
        "class_logits_shape": list(logits_value.shape),
        "class_logits_dtype": "float32",
        "class_logits_array_sha256": analyzer._array_sha256(logits_value),
        **_score_payload(logits_value),
    }


def _artifact_object(
    tmp_path: Path,
    *,
    sample_id: str,
    feature: np.ndarray,
    logits: np.ndarray,
) -> analyzer.EffortArtifact:
    path = tmp_path / f"{sample_id}.npz"
    return analyzer.EffortArtifact(
        sample_id=sample_id,
        path=path,
        relative_path=path.name,
        file_sha256="a" * 64,
        file_bytes=analyzer.NPZ_FILE_BYTES,
        feature_array_sha256=analyzer._array_sha256(feature),
        logits_array_sha256=analyzer._array_sha256(logits),
        feature=np.ascontiguousarray(feature, dtype=np.float32),
        logits=np.ascontiguousarray(logits, dtype=np.float32),
    )


def _bundle(
    tmp_path: Path,
    *,
    row: dict,
    artifact: analyzer.EffortArtifact,
    mode: str = "smoke",
) -> analyzer.RunBundle:
    placeholder = tmp_path / "placeholder"
    return analyzer.RunBundle(
        run_id="run-a",
        mode=mode,
        fingerprint="f" * 64,
        run_dir=tmp_path,
        manifest_path=placeholder,
        results_path=placeholder,
        expected_path=placeholder,
        summary_path=placeholder,
        manifest={},
        immutable={"runtime": _runtime()},
        release=SimpleNamespace(repo_root=tmp_path),
        selected=({},),
        contract=SimpleNamespace(),
        physical_results=(row,),
        latest_results=(row,),
        coverage={"is_complete": True},
        artifact_dir=tmp_path,
        artifacts={artifact.sample_id: artifact},
    )


def test_strict_json_rejects_duplicate_nonfinite_and_noncanonical(
    tmp_path: Path,
):
    path = tmp_path / "rows.jsonl"
    path.write_text('{"a":1,"a":2}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate JSON key"):
        analyzer._read_jsonl_strict(path, "fixture")
    path.write_text('{"a":NaN}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="non-finite"):
        analyzer._read_jsonl_strict(path, "fixture")
    path.write_text('{"b": 2, "a": 1}\\n'.replace("\\n", "\n"), encoding="utf-8")
    with pytest.raises(ValueError, match="canonical JSONL"):
        analyzer._read_jsonl_strict(path, "fixture")
    canonical = {"a": 1, "b": 2}
    path.write_text(f"{stable_json(canonical)}\n", encoding="utf-8")
    assert analyzer._read_jsonl_strict(path, "fixture") == [canonical]


def test_cpu_preflight_requires_real_cuda_uninitialized_fields():
    source, assets, model_audit = (
        {"commit": "x"},
        {"checkpoint": {}},
        {"strict_load": True},
    )
    value = _preflight(source, assets, model_audit)
    analyzer._validate_cpu_preflight(
        value,
        source=source,
        assets=assets,
        model_audit=model_audit,
    )
    for field in ("cuda_initialized_before", "cuda_initialized_after"):
        tampered = copy.deepcopy(value)
        tampered["report"][field] = True
        with pytest.raises(ValueError, match="CPU preflight evidence"):
            analyzer._validate_cpu_preflight(
                tampered,
                source=source,
                assets=assets,
                model_audit=model_audit,
            )
    missing = copy.deepcopy(value)
    del missing["report"]["cuda_initialized_before"]
    with pytest.raises(ValueError, match="key set"):
        analyzer._validate_cpu_preflight(
            missing,
            source=source,
            assets=assets,
            model_audit=model_audit,
        )


def test_visibility_census_is_exact_for_balanced250():
    release = load_canonical_release(
        REPO_ROOT,
        Path("outputs/opensource/balanced250_v1/manifest.json"),
        verify_files=False,
    )
    census = analyzer.visibility_census(release.inputs)
    assert census["all_local"] == {
        "full": 750,
        "partial": 0,
        "none": 0,
        "total": 750,
        "mean_edit_visible_gt_fraction": 1.0,
    }
    assert census["not_applicable_images"] == 1025
    assert "pair_rank" not in json.dumps(census)


def test_score_payload_recomputes_two_class_float32_softmax():
    logits = np.asarray([0.25, -0.5], dtype=np.float32)
    row = _score_payload(logits)
    analyzer._validate_score_payload(
        row,
        sample_id="sample",
        artifact_logits=logits,
    )
    for mutation in (
        {"score": row["score"] + 1e-6},
        {"raw_logit_margin": row["raw_logit_margin"] + 1e-6},
        {"class_logits": [0.25, -0.25]},
        {"manual_replay": {}},
    ):
        with pytest.raises(ValueError):
            analyzer._validate_score_payload(
                {**row, **mutation},
                sample_id="sample",
                artifact_logits=logits,
            )


def test_score_payload_allows_only_the_frozen_cpu_cuda_softmax_ulp_gap():
    logits = np.asarray(
        [2.7427079677581787, -3.2405028343200684],
        dtype=np.float32,
    )
    row = _score_payload(logits)
    recorded_cuda_score = 0.002514382591471076
    assert (
        abs(recorded_cuda_score - row["ai_score"])
        < analyzer.STATIC_CPU_SOFTMAX_ABS_TOLERANCE
    )
    for key in ("fake_probability", "probability", "ai_score", "score"):
        row[key] = recorded_cuda_score
    row["t1"]["score"] = recorded_cuda_score
    analyzer._validate_score_payload(
        row,
        sample_id="cuda-softmax-fixture",
        artifact_logits=logits,
    )

    invalid = copy.deepcopy(row)
    invalid_score = recorded_cuda_score + (
        2.0 * analyzer.STATIC_CPU_SOFTMAX_ABS_TOLERANCE
    )
    for key in ("fake_probability", "probability", "ai_score", "score"):
        invalid[key] = invalid_score
    invalid["t1"]["score"] = invalid_score
    with pytest.raises(ValueError, match="static CPU softmax sanity"):
        analyzer._validate_score_payload(
            invalid,
            sample_id="cuda-softmax-fixture",
            artifact_logits=logits,
        )


def test_npz_artifact_contract_and_inventory_are_fail_closed(
    tmp_path: Path,
):
    artifact_dir = tmp_path / "outputs" / "artifacts"
    row = _artifact_row(tmp_path, artifact_dir)
    artifacts = analyzer.validate_artifact_inventory(
        latest_results=[row],
        repo_root=tmp_path,
        artifact_dir=artifact_dir,
    )
    assert np.array_equal(
        artifacts["sample"].feature,
        np.arange(analyzer.FEATURE_SHAPE[0], dtype=np.float32),
    )
    assert artifacts["sample"].logits.shape == analyzer.LOGIT_SHAPE

    tampered = copy.deepcopy(row)
    tampered["class_logits_array_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="class_logits_array_sha256"):
        analyzer.validate_artifact_inventory(
            latest_results=[tampered],
            repo_root=tmp_path,
            artifact_dir=artifact_dir,
        )
    (artifact_dir / "extra.npz").write_bytes(b"extra")
    with pytest.raises(ValueError, match="inventory mismatch"):
        analyzer.validate_artifact_inventory(
            latest_results=[row],
            repo_root=tmp_path,
            artifact_dir=artifact_dir,
        )


def test_artifact_rejects_embedded_logit_drift(tmp_path: Path):
    artifact_dir = tmp_path / "outputs" / "artifacts"
    row = _artifact_row(tmp_path, artifact_dir)
    row["class_logits"][1] = float(
        np.float32(row["class_logits"][1] + np.float32(0.125))
    )
    with pytest.raises(ValueError, match="embedded/artifact logits"):
        analyzer.validate_artifact_inventory(
            latest_results=[row],
            repo_root=tmp_path,
            artifact_dir=artifact_dir,
        )


def test_smoke_compare_ignores_npz_container_identity_only(
    tmp_path: Path,
):
    feature = np.arange(analyzer.FEATURE_SHAPE[0], dtype=np.float32)
    logits = np.asarray([0.0, 1.0], dtype=np.float32)
    base = {
        "sample_id": "sample",
        "status": "ok",
        "valid_for_metrics": True,
        **_score_payload(logits),
        "run_id": "run-a",
        "run_manifest_fingerprint": "a" * 64,
        "config_fingerprint": "a" * 64,
        "completed_at": "time-a",
        "preprocess_latency_ms": 1.0,
        "latency_ms": 2.0,
        "peak_cuda_memory_bytes": 3,
        "artifact_path": "a.npz",
        "artifact_sha256": "a" * 64,
        "artifact_paths": {"effort_npz": "a.npz"},
    }
    replay = {
        **base,
        "run_id": "run-b",
        "run_manifest_fingerprint": "b" * 64,
        "config_fingerprint": "b" * 64,
        "completed_at": "time-b",
        "preprocess_latency_ms": 9.0,
        "latency_ms": 10.0,
        "peak_cuda_memory_bytes": 11,
        "artifact_path": "b.npz",
        "artifact_sha256": "b" * 64,
        "artifact_paths": {"effort_npz": "b.npz"},
    }
    left = _artifact_object(
        tmp_path,
        sample_id="sample",
        feature=feature,
        logits=logits,
    )
    right = _artifact_object(
        tmp_path,
        sample_id="sample",
        feature=feature.copy(),
        logits=logits.copy(),
    )
    report = analyzer.compare_computational_results(
        reference_rows=[base],
        replay_rows=[replay],
        reference_artifacts={"sample": left},
        replay_artifacts={"sample": right},
    )
    assert report["images_compared"] == 1
    assert report["npz_file_sha256_ignored_due_to_zip_timestamps"] is True

    changed_feature = feature.copy()
    changed_feature[0] += np.float32(1.0)
    changed = _artifact_object(
        tmp_path,
        sample_id="sample",
        feature=changed_feature,
        logits=logits,
    )
    with pytest.raises(ValueError, match="artifact arrays differ"):
        analyzer.compare_computational_results(
            reference_rows=[base],
            replay_rows=[replay],
            reference_artifacts={"sample": left},
            replay_artifacts={"sample": changed},
        )
    score_tamper = copy.deepcopy(replay)
    score_tamper["ai_score"] += 0.01
    with pytest.raises(ValueError):
        analyzer.compare_computational_results(
            reference_rows=[base],
            replay_rows=[score_tamper],
            reference_artifacts={"sample": left},
            replay_artifacts={"sample": right},
        )


def test_head_replay_uses_1024_to_2_linear_and_softmax(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    import torch

    feature = np.zeros(analyzer.FEATURE_SHAPE, dtype=np.float32)
    feature[0] = 2.0
    weight = torch.zeros(
        analyzer.LOGIT_SHAPE[0],
        analyzer.FEATURE_SHAPE[0],
        dtype=torch.float32,
    )
    weight[1, 0] = 0.5
    bias = torch.zeros(analyzer.LOGIT_SHAPE[0], dtype=torch.float32)
    logits = np.asarray([0.0, 1.0], dtype=np.float32)
    row = {"sample_id": "sample", **_score_payload(logits)}
    artifact = _artifact_object(
        tmp_path,
        sample_id="sample",
        feature=feature,
        logits=logits,
    )
    bundle = _bundle(tmp_path, row=row, artifact=artifact)
    monkeypatch.setattr(
        analyzer,
        "_configure_exact_recorded_runtime",
        lambda **_kwargs: (torch.device("cpu"), _runtime()),
    )
    report = analyzer.replay_linear_head(
        bundle,
        state={"head.weight": weight, "head.bias": bias},
        device_text="cpu",
    )
    assert report["features_replayed"] == 1
    assert report["class_count"] == 2
    assert report["max_head_logit_abs_difference"] == 0.0

    bias[1] = 0.25
    with pytest.raises(ValueError, match="head logits differ"):
        analyzer.replay_linear_head(
            bundle,
            state={"head.weight": weight, "head.bias": bias},
            device_text="cpu",
        )


def test_runtime_gate_rejects_cross_device_before_configuration(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        analyzer.legacy_audit,
        "_configure_runtime",
        lambda *_args: pytest.fail("must reject before runtime configuration"),
    )
    with pytest.raises(ValueError, match="exactly match"):
        analyzer._configure_exact_recorded_runtime(
            device_text="cuda:0",
            recorded_runtime=_runtime("cpu"),
        )


def test_fresh_replay_checks_every_image_feature_logit_and_head(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    import torch

    input_path = tmp_path / "image.jpg"
    input_path.write_bytes(b"fixture")
    preprocess = {"fixture": True}
    feature = np.zeros(analyzer.FEATURE_SHAPE, dtype=np.float32)
    feature[0] = 2.0
    logits = np.asarray([0.0, 1.0], dtype=np.float32)
    row = {
        "sample_id": "sample",
        "preprocess": preprocess,
        **_score_payload(logits),
    }
    artifact = _artifact_object(
        tmp_path,
        sample_id="sample",
        feature=feature,
        logits=logits,
    )
    bundle = _bundle(
        tmp_path,
        row=row,
        artifact=artifact,
        mode="formal",
    )
    bundle = analyzer.RunBundle(
        **{
            **bundle.__dict__,
            "selected": (
                {
                    "sample_id": "sample",
                    "canonical_path": "image.jpg",
                    "canonical_sha256": hashlib.sha256(b"fixture").hexdigest(),
                },
            ),
            "immutable": {
                "runtime": _runtime(),
                "model_audit": {},
                "runtime_golden": {},
            },
        }
    )

    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.head = torch.nn.Linear(
                analyzer.FEATURE_SHAPE[0],
                analyzer.LOGIT_SHAPE[0],
            )
            with torch.no_grad():
                self.head.weight.zero_()
                self.head.bias.zero_()
                self.head.weight[1, 0] = 0.5

        def forward(self, _tensor):
            value = torch.from_numpy(feature).reshape(1, -1)
            return self.head(value), value

    monkeypatch.setattr(analyzer, "FORMAL_IMAGES", 1)
    monkeypatch.setattr(
        analyzer,
        "_configure_exact_recorded_runtime",
        lambda **_kwargs: (torch.device("cpu"), _runtime()),
    )
    monkeypatch.setattr(
        analyzer.legacy_audit,
        "_build_model",
        lambda *_args, **_kwargs: (Toy().eval(), {}),
    )
    monkeypatch.setattr(
        analyzer,
        "_validate_independent_model_audit",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        analyzer,
        "_replay_runtime_golden_independently",
        lambda **_kwargs: {"status": "passed"},
    )
    monkeypatch.setattr(
        analyzer.legacy_audit,
        "_preprocess",
        lambda _path: (
            np.zeros((3, 224, 224), dtype=np.float32),
            preprocess,
        ),
    )
    report = analyzer.replay_model(
        bundle,
        source_root=tmp_path,
        state={},
        config={},
        device_text="cpu",
    )
    assert report["images_replayed"] == 1
    assert report["max_feature_abs_difference"] == 0.0
    assert report["max_class_logit_abs_difference"] == 0.0

    drift = feature.copy()
    drift[1] = 1.0
    drift_artifact = _artifact_object(
        tmp_path,
        sample_id="sample",
        feature=drift,
        logits=logits,
    )
    drift_bundle = analyzer.RunBundle(
        **{
            **bundle.__dict__,
            "artifacts": {"sample": drift_artifact},
        }
    )
    with pytest.raises(ValueError, match="fresh feature differs"):
        analyzer.replay_model(
            drift_bundle,
            source_root=tmp_path,
            state={},
            config={},
            device_text="cpu",
        )


def test_output_scope_is_canonical_and_traversal_closed(tmp_path: Path):
    bundle = SimpleNamespace(
        run_dir=tmp_path / "results" / "run",
        manifest_path=tmp_path / "results" / "run" / "manifest.json",
        results_path=tmp_path / "results" / "run" / "results.jsonl",
        expected_path=tmp_path / "results" / "run" / "expected.jsonl",
        summary_path=tmp_path / "results" / "run" / "summary.json",
    )
    metrics, audit = analyzer._formal_output_paths(
        bundle=bundle,
        repo_root=tmp_path,
        metrics_output_path=None,
        audit_output_path=None,
    )
    assert metrics.name == "balanced250_metrics.json"
    assert audit.name == "independent_audit.json"
    with pytest.raises(ValueError, match="canonical files"):
        analyzer._formal_output_paths(
            bundle=bundle,
            repo_root=tmp_path,
            metrics_output_path=Path("../escaped.json"),
            audit_output_path=None,
        )
