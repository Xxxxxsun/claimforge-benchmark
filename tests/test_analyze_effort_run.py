from __future__ import annotations

import copy
import json
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn

from eval.opensource import analyze_effort_run as analyzer
from eval.opensource.common import (
    atomic_write_json,
    atomic_write_jsonl,
    read_jsonl,
    sha256_file,
)
from eval.opensource.effort_metrics import summarize_effort_results


class _TinyEffort(nn.Module):
    """Small deterministic substitute for independent full-model forwards."""

    def __init__(self) -> None:
        super().__init__()
        self.head = nn.Linear(1024, 2, dtype=torch.float32)
        with torch.no_grad():
            self.head.weight.zero_()
            self.head.bias.zero_()
            self.head.weight[0, 0] = -1.0
            self.head.weight[1, 0] = 1.0

    def forward(
        self,
        image: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mean = image.mean(dim=(1, 2, 3))
        feature = torch.zeros(
            (image.shape[0], 1024),
            dtype=torch.float32,
            device=image.device,
        )
        feature[:, 0] = mean
        return self.head(feature), feature


@dataclass
class _TinyRun:
    repo_root: Path
    run_dir: Path
    output_path: Path
    expected_path: Path
    results_path: Path
    summary_path: Path
    manifest_path: Path
    expected: list[dict]
    rows: list[dict]
    model: _TinyEffort


def _write_artifact(
    path: Path,
    feature: np.ndarray,
    logits: np.ndarray,
    **extras: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        pooler_output=np.ascontiguousarray(feature, dtype=np.float32),
        class_logits=np.ascontiguousarray(logits, dtype=np.float32),
        **extras,
    )


def _build_tiny_run(tmp_path: Path) -> _TinyRun:
    repo_root = tmp_path / "repo"
    run_dir = repo_root / "results" / "tiny-effort"
    image_dir = repo_root / "inputs"
    artifact_dir = run_dir / "artifacts"
    image_dir.mkdir(parents=True)
    artifact_dir.mkdir(parents=True)

    model = _TinyEffort().eval()
    expected: list[dict] = []
    rows: list[dict] = []
    config = {
        "source_commit": analyzer.FROZEN_SOURCE_COMMIT,
        "preprocess_profile": analyzer.FROZEN_PREPROCESS_PROFILE,
        "score_semantics": analyzer.FROZEN_SCORE_SEMANTICS,
        "classification_threshold": analyzer.THRESHOLD,
        "checkpoint_and_protocol_frozen_before_mouse_scores": True,
        "bootstrap_samples": 5,
        "bootstrap_seed": 20260724,
    }
    config_fingerprint = analyzer._fingerprint(config)
    runtime_golden = {
        "status": "passed",
        "kind": "synthetic_non_mouse_test_fixture",
        "mouse_model_scores_computed": 0,
    }

    specifications = (
        ("tiny-real", "real", 0, 0),
        ("tiny-forged", "forged", 1, 255),
    )
    for rank, (sample_id, kind, label, level) in enumerate(specifications):
        rgb = np.full((9, 13, 3), level, dtype=np.uint8)
        image_path = image_dir / f"{sample_id}.png"
        Image.fromarray(rgb, mode="RGB").save(image_path)
        canonical_path = str(image_path.relative_to(repo_root))
        expected_row = {
            "sample_id": sample_id,
            "task_id": "tiny-task",
            "pair_rank": 0,
            "rank": rank,
            "kind": kind,
            "label": label,
            "domain": "synthetic",
            "canonical_path": canonical_path,
            "canonical_sha256": sha256_file(image_path),
            "width": 13,
            "height": 9,
        }
        expected.append(expected_row)

        image, preprocess = analyzer._preprocess(image_path)
        with torch.inference_mode():
            logits_tensor, feature_tensor = model(
                torch.from_numpy(image).unsqueeze(0)
            )
            probability = torch.softmax(logits_tensor, dim=1)[:, 1]
            margin = logits_tensor[:, 1] - logits_tensor[:, 0]
        feature = np.ascontiguousarray(
            feature_tensor[0].numpy(),
            dtype=np.float32,
        )
        logits = np.ascontiguousarray(
            logits_tensor[0].numpy(),
            dtype=np.float32,
        )
        artifact_path = artifact_dir / f"{sample_id}.npz"
        _write_artifact(artifact_path, feature, logits)
        score = float(probability[0].item())
        row = {
            "id": sample_id,
            "sample_id": sample_id,
            "task_id": "tiny-task",
            "pair_rank": 0,
            "rank": rank,
            "kind": kind,
            "label": label,
            "domain": "synthetic",
            "input_path": canonical_path,
            "input_sha256": sha256_file(image_path),
            "input_width": 13,
            "input_height": 9,
            "model": "Effort",
            "model_slug": analyzer.FROZEN_MODEL_SLUG,
            "checkpoint_id": analyzer.FROZEN_CHECKPOINT_ID,
            "preprocess_profile": analyzer.FROZEN_PREPROCESS_PROFILE,
            "score_semantics": analyzer.FROZEN_SCORE_SEMANTICS,
            "classification_threshold": analyzer.THRESHOLD,
            "classification_threshold_operator": ">",
            "config_fingerprint": config_fingerprint,
            "edit_visibility": "full",
            "edit_visible_gt_fraction": 1.0,
            "edit_visibility_evidence": {
                "definition": "whole_canvas_direct_resize",
                "crop": None,
            },
            "valid_for_t1": True,
            "valid_for_t2": False,
            "status": "ok",
            "valid_for_metrics": True,
            "preprocess": preprocess,
            "artifact_path": str(artifact_path.relative_to(repo_root)),
            "artifact_sha256": sha256_file(artifact_path),
            "artifact_keys": ["pooler_output", "class_logits"],
            "feature_shape": [1024],
            "feature_dtype": "float32",
            "feature_array_sha256": analyzer._array_sha256(feature),
            "class_logits_shape": [2],
            "class_logits_dtype": "float32",
            "class_logits_array_sha256": analyzer._array_sha256(logits),
            "class_logits": [float(value) for value in logits.tolist()],
            "raw_logit_margin": float(margin[0].item()),
            "fake_probability": score,
            "probability": score,
            "ai_score": score,
            "score": score,
            "classification_decision": score > 0.5,
            "latency_ms": 1.0,
            "peak_cuda_memory_bytes": None,
        }
        rows.append(row)

    expected_path = run_dir / "expected_inputs.jsonl"
    results_path = run_dir / "results.jsonl"
    summary_path = run_dir / "summary.json"
    manifest_path = run_dir / "run_manifest.json"
    output_path = run_dir / "independent_audit.json"
    atomic_write_jsonl(expected_path, expected)
    atomic_write_jsonl(results_path, rows)

    summary = summarize_effort_results(
        rows,
        expected,
        threshold=0.5,
        bootstrap_samples=5,
        seed=20260724,
    )
    summary.update(
        {
            "run_id": run_dir.name,
            "model": "Effort",
            "model_slug": analyzer.FROZEN_MODEL_SLUG,
            "checkpoint_id": analyzer.FROZEN_CHECKPOINT_ID,
            "checkpoint_sha256": analyzer.FROZEN_CHECKPOINT_SHA256,
            "preprocess_profile": analyzer.FROZEN_PREPROCESS_PROFILE,
            "config_fingerprint": config_fingerprint,
            "runtime_golden_status": runtime_golden["status"],
            "runtime_golden_fingerprint": analyzer._fingerprint(
                runtime_golden
            ),
            "generated_at": "2026-07-25T00:00:00Z",
        }
    )
    atomic_write_json(summary_path, summary)

    manifest = {
        "schema_version": "effort_detection_run_manifest_v1",
        "run_id": run_dir.name,
        "status": "complete",
        "config": config,
        "config_fingerprint": config_fingerprint,
        "runtime": {"device": "cpu"},
        "runtime_golden": runtime_golden,
        "dataset": {
            "selected_images": 2,
            "expected_inputs_sha256": sha256_file(expected_path),
        },
        "outputs": {
            "results_sha256": sha256_file(results_path),
            "summary_sha256": sha256_file(summary_path),
            "artifact_files": 2,
        },
        "task_scope": {
            "valid_for_t1": True,
            "valid_for_t2": False,
        },
    }
    atomic_write_json(manifest_path, manifest)
    return _TinyRun(
        repo_root=repo_root,
        run_dir=run_dir,
        output_path=output_path,
        expected_path=expected_path,
        results_path=results_path,
        summary_path=summary_path,
        manifest_path=manifest_path,
        expected=expected,
        rows=rows,
        model=model,
    )


def _install_independent_mocks(
    case: _TinyRun,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        analyzer,
        "_verify_source",
        lambda _path: {
            "commit": analyzer.FROZEN_SOURCE_COMMIT,
            "tracked_dirty": False,
            "tracked_license_files": [],
            "files": {},
        },
    )
    monkeypatch.setattr(
        analyzer,
        "_load_assets",
        lambda _checkpoint, _config: (
            OrderedDict(),
            {"vision_config": {}},
            {
                "checkpoint_sha256": analyzer.FROZEN_CHECKPOINT_SHA256,
                "checkpoint_tensor_count": 681,
                "weights_only": True,
                "unsafe_globals": [],
            },
        ),
    )
    monkeypatch.setattr(
        analyzer,
        "_configure_runtime",
        lambda device_text: (
            torch.device(device_text),
            {"device": str(torch.device(device_text)), "synthetic": True},
        ),
    )
    monkeypatch.setattr(
        analyzer,
        "_build_model",
        lambda _state, _config, _device: (
            case.model,
            {
                "strict_load": True,
                "missing_keys": [],
                "unexpected_keys": [],
                "svd_modules": 96,
            },
        ),
    )


def _audit(
    case: _TinyRun,
    monkeypatch: pytest.MonkeyPatch,
) -> dict:
    _install_independent_mocks(case, monkeypatch)
    return analyzer._audit_run(
        repo_root=case.repo_root,
        run_dir=case.run_dir,
        source_root=case.repo_root / "source",
        checkpoint_path=case.repo_root / "checkpoint.pth",
        config_path=case.repo_root / "config.json",
        device_text="cpu",
        output_path=case.output_path,
    )


def _manifest(case: _TinyRun) -> dict:
    return json.loads(case.manifest_path.read_text(encoding="utf-8"))


def _summary(case: _TinyRun) -> dict:
    return json.loads(case.summary_path.read_text(encoding="utf-8"))


def _rewrite_results_and_sync_hash(
    case: _TinyRun,
    rows: list[dict],
) -> None:
    atomic_write_jsonl(case.results_path, rows)
    manifest = _manifest(case)
    manifest["outputs"]["results_sha256"] = sha256_file(case.results_path)
    atomic_write_json(case.manifest_path, manifest)


def _rewrite_summary_and_sync_hash(
    case: _TinyRun,
    summary: dict,
) -> None:
    atomic_write_json(case.summary_path, summary)
    manifest = _manifest(case)
    manifest["outputs"]["summary_sha256"] = sha256_file(case.summary_path)
    atomic_write_json(case.manifest_path, manifest)


def test_synthetic_happy_path_audits_every_layer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _build_tiny_run(tmp_path)
    audit = _audit(case, monkeypatch)

    assert audit["schema_version"] == "effort_independent_audit_v1"
    assert audit["status"] == "ok"
    assert audit["run_id"] == "tiny-effort"
    assert audit["coverage"] == {
        "expected_images": 2,
        "physical_result_rows": 2,
        "latest_images": 2,
        "fresh_full_model_forwards": 2,
        "artifact_replays": 2,
        "complete_pairs": 1,
    }
    assert audit["replay"] == {
        "max_abs_feature_diff": 0.0,
        "max_abs_class_logit_diff": 0.0,
        "max_abs_head_replay_diff": 0.0,
        "max_abs_probability_diff": 0.0,
        "max_abs_margin_diff": 0.0,
        "all_decisions_exact": True,
        "all_preprocess_records_exact": True,
        "summary_exact_recompute": True,
    }
    assert audit["hashes"]["manifest_sha256"] == sha256_file(
        case.manifest_path
    )
    assert audit["hashes"]["results_sha256"] == sha256_file(case.results_path)
    assert audit["hashes"]["summary_sha256"] == sha256_file(case.summary_path)
    assert audit["hashes"]["expected_inputs_sha256"] == sha256_file(
        case.expected_path
    )
    assert set(audit["hashes"]["result_fingerprints"]) == {
        "tiny-real",
        "tiny-forged",
    }
    assert json.loads(case.output_path.read_text(encoding="utf-8")) == audit
    analyzer._reject_t2(audit, path="audit")


@pytest.mark.parametrize(
    ("target", "message"),
    [
        ("results_sha256", "Effort results SHA-256 mismatch"),
        ("summary_sha256", "Effort summary SHA-256 mismatch"),
    ],
)
def test_manifest_output_hash_tamper_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target: str,
    message: str,
) -> None:
    case = _build_tiny_run(tmp_path)
    manifest = _manifest(case)
    manifest["outputs"][target] = "0" * 64
    atomic_write_json(case.manifest_path, manifest)

    with pytest.raises(ValueError, match=message):
        _audit(case, monkeypatch)


def test_expected_inputs_hash_and_count_tamper_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _build_tiny_run(tmp_path)
    manifest = _manifest(case)
    manifest["dataset"]["expected_inputs_sha256"] = "0" * 64
    atomic_write_json(case.manifest_path, manifest)
    with pytest.raises(ValueError, match="expected inputs SHA-256 mismatch"):
        _audit(case, monkeypatch)

    case = _build_tiny_run(tmp_path / "count")
    manifest = _manifest(case)
    manifest["dataset"]["selected_images"] = 3
    atomic_write_json(case.manifest_path, manifest)
    with pytest.raises(ValueError, match="expected image count mismatch"):
        _audit(case, monkeypatch)


def test_config_fingerprint_and_pinned_fields_are_independently_checked(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _build_tiny_run(tmp_path)
    manifest = _manifest(case)
    manifest["config"]["bootstrap_seed"] = 1
    atomic_write_json(case.manifest_path, manifest)
    with pytest.raises(ValueError, match="config fingerprint mismatch"):
        _audit(case, monkeypatch)

    case = _build_tiny_run(tmp_path / "pin")
    manifest = _manifest(case)
    manifest["config"]["source_commit"] = "0" * 40
    manifest["config_fingerprint"] = analyzer._fingerprint(
        manifest["config"]
    )
    atomic_write_json(case.manifest_path, manifest)
    with pytest.raises(ValueError, match="source pin mismatch"):
        _audit(case, monkeypatch)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("schema_version", "wrong", "manifest schema mismatch"),
        ("status", "incomplete", "manifest is not complete"),
        ("run_id", "other", "manifest run_id mismatch"),
    ],
)
def test_manifest_identity_tamper_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: str,
    message: str,
) -> None:
    case = _build_tiny_run(tmp_path)
    manifest = _manifest(case)
    manifest[field] = value
    atomic_write_json(case.manifest_path, manifest)
    with pytest.raises(ValueError, match=message):
        _audit(case, monkeypatch)


def test_source_and_checkpoint_drift_propagate_as_hard_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _build_tiny_run(tmp_path)
    _install_independent_mocks(case, monkeypatch)

    def reject_source(_path: Path) -> dict:
        raise ValueError("independent Effort source commit mismatch")

    monkeypatch.setattr(
        analyzer,
        "_verify_source",
        reject_source,
    )
    with pytest.raises(ValueError, match="source commit mismatch"):
        analyzer._audit_run(
            repo_root=case.repo_root,
            run_dir=case.run_dir,
            source_root=case.repo_root / "source",
            checkpoint_path=case.repo_root / "checkpoint.pth",
            config_path=case.repo_root / "config.json",
            device_text="cpu",
            output_path=case.output_path,
        )

    case = _build_tiny_run(tmp_path / "checkpoint")
    _install_independent_mocks(case, monkeypatch)

    def reject_checkpoint(
        _checkpoint: Path,
        _config: Path,
    ) -> tuple[OrderedDict, dict, dict]:
        raise ValueError("independent Effort checkpoint byte size mismatch")

    monkeypatch.setattr(
        analyzer,
        "_load_assets",
        reject_checkpoint,
    )
    with pytest.raises(ValueError, match="checkpoint byte size mismatch"):
        analyzer._audit_run(
            repo_root=case.repo_root,
            run_dir=case.run_dir,
            source_root=case.repo_root / "source",
            checkpoint_path=case.repo_root / "checkpoint.pth",
            config_path=case.repo_root / "config.json",
            device_text="cpu",
            output_path=case.output_path,
        )


def test_private_source_and_asset_guards_reject_drift_before_loading(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(analyzer, "_git_value", lambda *_args: "0" * 40)
    with pytest.raises(ValueError, match="source commit mismatch"):
        analyzer._verify_source(tmp_path)

    checkpoint = tmp_path / "tiny.pth"
    config = tmp_path / "config.json"
    checkpoint.write_bytes(b"not-the-pinned-checkpoint")
    config.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="checkpoint byte size mismatch"):
        analyzer._load_assets(checkpoint, config)


def test_safe_npz_loader_accepts_only_exact_finite_float32_contract(
    tmp_path: Path,
) -> None:
    path = tmp_path / "valid.npz"
    feature = np.arange(1024, dtype=np.float32)
    logits = np.asarray([-1.0, 1.0], dtype=np.float32)
    _write_artifact(path, feature, logits)
    loaded_feature, loaded_logits = analyzer._load_artifact(path)
    np.testing.assert_array_equal(loaded_feature, feature)
    np.testing.assert_array_equal(loaded_logits, logits)


@pytest.mark.parametrize(
    "variant",
    ["extra", "shape", "dtype", "nonfinite", "object"],
)
def test_safe_npz_loader_rejects_malformed_artifacts(
    tmp_path: Path,
    variant: str,
) -> None:
    path = tmp_path / f"{variant}.npz"
    feature: np.ndarray = np.zeros(1024, dtype=np.float32)
    logits: np.ndarray = np.zeros(2, dtype=np.float32)
    extras: dict[str, np.ndarray] = {}
    if variant == "extra":
        extras["unexpected"] = np.zeros(1, dtype=np.float32)
    elif variant == "shape":
        feature = np.zeros(1023, dtype=np.float32)
    elif variant == "dtype":
        logits = np.zeros(2, dtype=np.float64)
    elif variant == "nonfinite":
        logits[0] = np.inf
    elif variant == "object":
        feature = np.asarray([object()] * 1024, dtype=object)
    np.savez(
        path,
        pooler_output=feature,
        class_logits=logits,
        **extras,
    )
    with pytest.raises(ValueError):
        analyzer._load_artifact(path)


def test_artifact_path_traversal_and_duplicate_reference_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _build_tiny_run(tmp_path)
    rows = copy.deepcopy(case.rows)
    outside = case.repo_root / "outside.npz"
    original = case.run_dir / "artifacts" / "tiny-real.npz"
    outside.write_bytes(original.read_bytes())
    rows[0]["artifact_path"] = str(outside.relative_to(case.repo_root))
    rows[0]["artifact_sha256"] = sha256_file(outside)
    _rewrite_results_and_sync_hash(case, rows)
    with pytest.raises(ValueError, match="artifact escapes run"):
        _audit(case, monkeypatch)

    case = _build_tiny_run(tmp_path / "duplicate")
    rows = copy.deepcopy(case.rows)
    for key in (
        "artifact_path",
        "artifact_sha256",
        "feature_array_sha256",
        "class_logits_array_sha256",
        "class_logits",
    ):
        rows[1][key] = copy.deepcopy(rows[0][key])
    _rewrite_results_and_sync_hash(case, rows)
    with pytest.raises(ValueError, match="artifact name mismatch"):
        _audit(case, monkeypatch)


def test_extra_artifact_file_and_manifest_count_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _build_tiny_run(tmp_path)
    _write_artifact(
        case.run_dir / "artifacts" / "extra.npz",
        np.zeros(1024, dtype=np.float32),
        np.zeros(2, dtype=np.float32),
    )
    with pytest.raises(ValueError, match="missing/extra files"):
        _audit(case, monkeypatch)

    case = _build_tiny_run(tmp_path / "count")
    manifest = _manifest(case)
    manifest["outputs"]["artifact_files"] = 3
    atomic_write_json(case.manifest_path, manifest)
    with pytest.raises(ValueError, match="artifact count mismatch"):
        _audit(case, monkeypatch)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("task_id", "other-task"),
        ("model", "Other"),
        ("checkpoint_id", "wrong"),
        ("preprocess_profile", "cubic"),
        ("valid_for_t2", True),
    ],
)
def test_result_identity_tamper_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
) -> None:
    case = _build_tiny_run(tmp_path)
    rows = copy.deepcopy(case.rows)
    rows[0][field] = value
    _rewrite_results_and_sync_hash(case, rows)
    with pytest.raises(ValueError, match=f"identity {field} mismatch"):
        _audit(case, monkeypatch)


def test_preprocess_tamper_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _build_tiny_run(tmp_path)
    rows = copy.deepcopy(case.rows)
    rows[0]["preprocess"]["tensor_sha256"] = "0" * 64
    _rewrite_results_and_sync_hash(case, rows)
    with pytest.raises(ValueError, match="preprocess mismatch"):
        _audit(case, monkeypatch)


def test_feature_and_logit_artifact_tamper_fail_fresh_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _build_tiny_run(tmp_path)
    rows = copy.deepcopy(case.rows)
    artifact = case.run_dir / "artifacts" / "tiny-real.npz"
    feature, logits = analyzer._load_artifact(artifact)
    feature[1] = 1.0
    _write_artifact(artifact, feature, logits)
    rows[0]["artifact_sha256"] = sha256_file(artifact)
    rows[0]["feature_array_sha256"] = analyzer._array_sha256(feature)
    _rewrite_results_and_sync_hash(case, rows)
    with pytest.raises(ValueError, match="fresh replay differs"):
        _audit(case, monkeypatch)

    case = _build_tiny_run(tmp_path / "logit")
    rows = copy.deepcopy(case.rows)
    artifact = case.run_dir / "artifacts" / "tiny-real.npz"
    feature, logits = analyzer._load_artifact(artifact)
    logits[0] += np.float32(0.25)
    _write_artifact(artifact, feature, logits)
    rows[0]["artifact_sha256"] = sha256_file(artifact)
    rows[0]["class_logits_array_sha256"] = analyzer._array_sha256(logits)
    rows[0]["class_logits"] = [float(value) for value in logits.tolist()]
    _rewrite_results_and_sync_hash(case, rows)
    with pytest.raises(ValueError, match="fresh replay differs"):
        _audit(case, monkeypatch)


def test_feature_hash_and_embedded_logits_tamper_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _build_tiny_run(tmp_path)
    rows = copy.deepcopy(case.rows)
    rows[0]["feature_array_sha256"] = "0" * 64
    _rewrite_results_and_sync_hash(case, rows)
    with pytest.raises(ValueError, match="feature hash mismatch"):
        _audit(case, monkeypatch)

    case = _build_tiny_run(tmp_path / "embedded")
    rows = copy.deepcopy(case.rows)
    rows[0]["class_logits"][0] += 0.5
    _rewrite_results_and_sync_hash(case, rows)
    with pytest.raises(ValueError, match="embedded logits mismatch"):
        _audit(case, monkeypatch)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("ai_score", 0.25, "probability replay differs"),
        ("fake_probability", 0.25, "score alias fake_probability drifted"),
        ("raw_logit_margin", 123.0, "margin replay differs"),
        ("classification_decision", None, "decision mismatch"),
    ],
)
def test_probability_margin_alias_and_decision_tamper_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
    message: str,
) -> None:
    case = _build_tiny_run(tmp_path)
    rows = copy.deepcopy(case.rows)
    rows[0][field] = value
    _rewrite_results_and_sync_hash(case, rows)
    with pytest.raises(ValueError, match=message):
        _audit(case, monkeypatch)


def test_summary_tamper_with_updated_hash_still_fails_recompute(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _build_tiny_run(tmp_path)
    summary = _summary(case)
    summary["detection"]["accuracy_at_0_5"] = 0.123
    _rewrite_summary_and_sync_hash(case, summary)
    with pytest.raises(ValueError, match="summary does not exactly recompute"):
        _audit(case, monkeypatch)


@pytest.mark.parametrize("container", ["manifest", "summary", "result"])
def test_nested_t2_tamper_is_rejected_everywhere(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    container: str,
) -> None:
    case = _build_tiny_run(tmp_path)
    invented = {"outer": [{"inner": {"pixel_ap": 0.9}}]}
    if container == "manifest":
        manifest = _manifest(case)
        manifest["invented"] = invented
        atomic_write_json(case.manifest_path, manifest)
    elif container == "summary":
        summary = _summary(case)
        summary["invented"] = invented
        _rewrite_summary_and_sync_hash(case, summary)
    else:
        rows = copy.deepcopy(case.rows)
        rows[0]["invented"] = invented
        _rewrite_results_and_sync_hash(case, rows)

    with pytest.raises(ValueError, match="invents Effort T2 fields"):
        _audit(case, monkeypatch)


def test_physical_duplicate_expected_id_and_extra_result_id_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _build_tiny_run(tmp_path)
    expected = copy.deepcopy(case.expected)
    expected[1]["sample_id"] = expected[0]["sample_id"]
    atomic_write_jsonl(case.expected_path, expected)
    manifest = _manifest(case)
    manifest["dataset"]["expected_inputs_sha256"] = sha256_file(
        case.expected_path
    )
    atomic_write_json(case.manifest_path, manifest)
    with pytest.raises(ValueError, match="expected IDs are duplicated"):
        _audit(case, monkeypatch)

    case = _build_tiny_run(tmp_path / "extra")
    rows = copy.deepcopy(case.rows)
    extra = copy.deepcopy(rows[0])
    extra["id"] = "unexpected"
    rows.append(extra)
    _rewrite_results_and_sync_hash(case, rows)
    with pytest.raises(ValueError, match="latest result coverage mismatch"):
        _audit(case, monkeypatch)


def test_main_rejects_run_and_output_path_traversal(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="run-id escapes"):
        analyzer.main(
            [
                "--results-dir",
                str(tmp_path),
                "--run-id",
                "../escape",
                "--device",
                "cpu",
            ]
        )
