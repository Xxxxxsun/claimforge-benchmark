from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from eval.opensource.common import sha256_file
from eval.opensource import run_community_forensics as runner


@pytest.fixture(scope="module")
def official_model() -> tuple:
    source, assets, state = runner.verify_assets(
        source_root=runner.DEFAULT_SOURCE_ROOT,
        model_root=runner.DEFAULT_MODEL_ROOT,
        processor_root=runner.DEFAULT_PROCESSOR_ROOT,
    )
    device, runtime = runner.configure_runtime("cpu")
    model, model_audit = runner.load_model(state=state, device=device)
    return model, device, source, assets, runtime, model_audit


def test_frozen_primary_contract_constants() -> None:
    assert runner.MODEL_SOURCE_COMMIT == (
        "ee5b71d43db0f3779e1edd64ee927b13f2dd6ad4"
    )
    assert runner.EVAL_SINGLE_COMMIT == (
        "5e52ed690bdbd609f9bb1705c4c80d11872a05bd"
    )
    assert runner.MODEL_HF_REVISION == (
        "6076002bf0d9dd37537f965ee2f06f826c333b61"
    )
    assert runner.PROCESSOR_HF_REVISION == (
        "3540a3f0d688f8bf492a8aed48613b891f88047e"
    )
    assert runner.CHECKPOINT["bytes"] == 87_262_324
    assert runner.CHECKPOINT["sha256"] == (
        "b89f36275f3bf5e2b040eee36597a8f19db051bff9a473a9cf7b2466284fb387"
    )
    assert runner.FEATURE_DIMENSION == 384
    assert runner.MODEL_INPUT_SIZE == 384
    assert runner.RESIZE_SHORT_SIDE == 440


def test_asset_contract_is_full_safe_state(official_model: tuple) -> None:
    _, _, source, assets, runtime, audit = official_model
    assert source["commit"] == runner.MODEL_SOURCE_COMMIT
    assert source["eval_single"]["commit"] == runner.EVAL_SINGLE_COMMIT
    checkpoint = assets["checkpoint"]
    assert checkpoint["serialization_safety"]["pickle_executed"] is False
    assert checkpoint["schema"]["tensor_count"] == 152
    assert checkpoint["schema"]["state_elements"] == 21_811_969
    assert audit["load"]["strict"] is True
    assert audit["load"]["full_state_coverage"] is True
    assert audit["load"]["missing_keys"] == []
    assert audit["load"]["unexpected_keys"] == []
    assert audit["construction"]["pretrained"] is False
    flags = audit["construction"]["attention_block_flags"]
    assert flags
    assert len(set(flags)) == 1
    assert audit["construction"]["fused_attention"] is flags[0]
    assert not any(audit["network"]["attempts"].values())
    assert runtime["cudnn_enabled"] is False
    assert runtime["cuda_matmul_allow_tf32"] is False
    assert runtime["cudnn_allow_tf32"] is False
    assert runtime["deterministic_algorithms"] is True
    assert runtime["float32_matmul_precision"] == "highest"


def test_official_five_image_golden_gate(official_model: tuple) -> None:
    model, device, *_ = official_model
    golden = runner.validate_official_golden(
        model=model,
        device=device,
        source_root=runner.DEFAULT_SOURCE_ROOT,
    )
    assert golden["status"] == "passed"
    assert len(golden["cases"]) == 5
    assert all(case["passed"] for case in golden["cases"])
    assert max(
        case["absolute_difference"] for case in golden["cases"]
    ) < 1e-9


def test_official_golden_gate_rejects_probability_drift(
    official_model: tuple,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, device, *_ = official_model
    changed = [dict(value) for value in runner.GOLDEN_CASES]
    changed[0]["probability"] = float(changed[0]["probability"]) - 0.01
    monkeypatch.setattr(runner, "GOLDEN_CASES", tuple(changed))
    with pytest.raises(ValueError, match="golden mismatch"):
        runner.validate_official_golden(
            model=model,
            device=device,
            source_root=runner.DEFAULT_SOURCE_ROOT,
        )


@pytest.mark.parametrize(
    ("width", "height", "resized", "crop_start"),
    [
        (1000, 500, [880, 440], [248, 28]),
        (500, 1000, [440, 880], [28, 248]),
        (1024, 1024, [440, 440], [28, 28]),
    ],
)
def test_preprocess_geometry_matches_torchvision(
    width: int,
    height: int,
    resized: list[int],
    crop_start: list[int],
) -> None:
    geometry = runner.compute_preprocess_geometry(width, height)
    assert geometry["resize"]["destination_size"] == resized
    assert geometry["center_crop"]["start_xy"] == crop_start
    assert geometry["center_crop"]["size"] == [384, 384]


def test_preprocess_is_rgb_float32_and_deterministic(tmp_path: Path) -> None:
    pixels = np.arange(37 * 61 * 3, dtype=np.uint8).reshape(37, 61, 3)
    path = tmp_path / "input.png"
    Image.fromarray(pixels, mode="RGB").save(path)
    first, first_audit = runner.preprocess_image(path)
    second, second_audit = runner.preprocess_image(path)
    assert first.shape == (3, 384, 384)
    assert first.dtype == np.float32
    assert np.array_equal(first, second)
    assert first_audit == second_audit
    assert first_audit["tensor_sha256"] == runner._array_sha256(first)


def test_classifier_replay_has_complete_exact_aliases() -> None:
    import torch

    classifier = torch.nn.Linear(384, 1)
    with torch.no_grad():
        classifier.weight.zero_()
        classifier.bias.zero_()
    feature = torch.zeros((1, 384), dtype=torch.float32)
    output = classifier(feature)
    result = runner.replay_classifier(output, feature, classifier)
    assert result["raw_logit"] == 0.0
    assert result["probability"] == 0.5
    assert result["score"] == result["ai_score"] == 0.5
    assert result["classification_decision"] is False
    assert result["classification"]["decision"] is False
    assert result["t1"]["policy"] == runner.T1_POLICY
    assert result["manual_replay"]["official_logit_exact_match"] is True


def test_infer_one_captures_one_384d_feature(
    official_model: tuple,
) -> None:
    model, device, *_ = official_model
    path = runner.DEFAULT_SOURCE_ROOT / "test_images" / "00000274.png"
    image, _ = runner.preprocess_image(path)
    scoring, feature, peak, latency = runner.infer_one(
        model,
        device,
        image,
    )
    assert feature.shape == (384,)
    assert feature.dtype == np.float32
    assert np.isfinite(feature).all()
    assert scoring["manual_replay"]["classifier_hook_calls"] == 1
    assert scoring["score"] == scoring["probability"]
    assert peak is None
    assert latency >= 0.0


def test_npy_file_hash_and_raw_array_hash_are_distinct(tmp_path: Path) -> None:
    feature = np.arange(384, dtype=np.float32)
    path = tmp_path / "feature.npy"
    runner._atomic_save_npy(path, feature)
    file_digest = sha256_file(path)
    raw_digest = runner._array_sha256(feature)
    assert file_digest != raw_digest
    assert np.array_equal(np.load(path, allow_pickle=False), feature)


def _resume_fixture(tmp_path: Path) -> tuple:
    import torch

    image_path = tmp_path / "input.png"
    Image.new("RGB", (500, 300), color=(20, 40, 60)).save(image_path)
    digest = sha256_file(image_path)
    canonical = {
        "sample_id": "sample-safe",
        "task_id": "task-1",
        "pair_rank": 1,
        "rank": 1,
        "kind": "forged",
        "label": 1,
        "domain": "lodging",
        "canonical_path": str(image_path),
        "canonical_sha256": digest,
        "width": 500,
        "height": 300,
    }
    visibility = {
        "edit_visibility": "full",
        "edit_visible_gt_fraction": 1.0,
        "edit_visibility_evidence": {"test": True},
    }
    fingerprint = "a" * 64
    identity = runner._result_identity(
        canonical,
        repo_root=tmp_path,
        visibility=visibility,
        config_fingerprint=fingerprint,
    )
    _, preprocess = runner.preprocess_image(image_path)
    classifier = torch.nn.Linear(384, 1)
    with torch.no_grad():
        classifier.weight.fill_(0.001)
        classifier.bias.fill_(0.25)
    feature = np.zeros(384, dtype=np.float32)
    tensor = torch.from_numpy(feature).unsqueeze(0)
    scoring = runner.replay_classifier(
        classifier(tensor),
        tensor,
        classifier,
    )
    feature_path = tmp_path / "features" / "sample-safe.npy"
    runner._atomic_save_npy(feature_path, feature)
    relative = str(feature_path.relative_to(tmp_path))
    row = {
        **identity,
        "status": "ok",
        "valid_for_metrics": True,
        "preprocess": preprocess,
        "preprocess_latency_ms": 1.0,
        "latency_ms": 2.0,
        "peak_cuda_memory_bytes": None,
        "commfor_feature_path": relative,
        "commfor_feature_sha256": sha256_file(feature_path),
        "commfor_feature_array_sha256": runner._array_sha256(feature),
        "feature_array_sha256": runner._array_sha256(feature),
        "commfor_feature_shape": [384],
        "commfor_feature_dtype": "float32",
        "commfor_feature_semantics": runner.FEATURE_SEMANTICS,
        "artifact_paths": {"commfor_feature_npy": relative},
        **scoring,
    }
    return row, identity, classifier, feature_path, fingerprint


def test_resume_replays_preprocess_feature_head_and_sigmoid(
    tmp_path: Path,
) -> None:
    import torch

    row, identity, classifier, _, fingerprint = _resume_fixture(tmp_path)
    runner._validate_resume_row(
        row,
        expected=identity,
        repo_root=tmp_path,
        run_dir=tmp_path,
        config_fingerprint=fingerprint,
        classifier=classifier,
        device=torch.device("cpu"),
    )


def test_resume_rejects_score_tamper(tmp_path: Path) -> None:
    import torch

    row, identity, classifier, _, fingerprint = _resume_fixture(tmp_path)
    tampered = copy.deepcopy(row)
    tampered["ai_score"] += 0.01
    with pytest.raises(ValueError):
        runner._validate_resume_row(
            tampered,
            expected=identity,
            repo_root=tmp_path,
            run_dir=tmp_path,
            config_fingerprint=fingerprint,
            classifier=classifier,
            device=torch.device("cpu"),
        )


def test_resume_rejects_self_consistent_feature_tamper(
    tmp_path: Path,
) -> None:
    import torch

    row, identity, classifier, path, fingerprint = _resume_fixture(tmp_path)
    feature = np.load(path, allow_pickle=False)
    feature[0] = 1.0
    runner._atomic_save_npy(path, feature)
    tampered = copy.deepcopy(row)
    tampered["commfor_feature_sha256"] = sha256_file(path)
    raw = runner._array_sha256(feature)
    tampered["commfor_feature_array_sha256"] = raw
    tampered["feature_array_sha256"] = raw
    with pytest.raises(ValueError, match="does not exactly replay"):
        runner._validate_resume_row(
            tampered,
            expected=identity,
            repo_root=tmp_path,
            run_dir=tmp_path,
            config_fingerprint=fingerprint,
            classifier=classifier,
            device=torch.device("cpu"),
        )


def test_resume_rejects_preprocess_hash_tamper(tmp_path: Path) -> None:
    import torch

    row, identity, classifier, _, fingerprint = _resume_fixture(tmp_path)
    tampered = copy.deepcopy(row)
    tampered["preprocess"]["tensor_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="does not replay exactly"):
        runner._validate_resume_row(
            tampered,
            expected=identity,
            repo_root=tmp_path,
            run_dir=tmp_path,
            config_fingerprint=fingerprint,
            classifier=classifier,
            device=torch.device("cpu"),
        )


@pytest.mark.parametrize(
    "value",
    ["..", ".", "../escape", "a/b", r"a\\b", " space", ""],
)
def test_safe_component_rejects_traversal(value: str) -> None:
    with pytest.raises(ValueError, match="safe"):
        runner._safe_component(value, label="test")


def test_select_inputs_rejects_traversal_sample_id() -> None:
    with pytest.raises(ValueError, match="safe"):
        runner.select_inputs([], None, "../sample")


def test_load_release_rejects_canonical_pin_drift(tmp_path: Path) -> None:
    manifest = {
        **runner.CANONICAL_RELEASE,
        "dataset_id": "substituted",
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="dataset_id"):
        runner.load_release(tmp_path, path)


def test_frozen_visibility_census_rejects_incomplete_mapping() -> None:
    with pytest.raises(ValueError, match="census changed"):
        runner.validate_frozen_visibility_census(
            {
                "task": {
                    "edit_visibility": "full",
                    "edit_visible_gt_fraction": 1.0,
                }
            }
        )


def test_adapter_contract_includes_transitive_metrics() -> None:
    root = Path(runner.__file__).resolve().parents[2]
    contract = runner.adapter_contract(root)
    assert "eval/opensource/run_community_forensics.py" in contract
    assert "eval/opensource/community_forensics_metrics.py" in contract
    assert "eval/opensource/ufd_metrics.py" in contract
    assert all(
        len(value["sha256"]) == 64 for value in contract.values()
    )


def test_model_construction_audit_changes_config_fingerprint() -> None:
    common = {
        "adapter": {},
        "runtime": {"device": "cpu"},
        "release": {
            "schema_version": runner.CANONICAL_RELEASE["schema_version"],
            "dataset_id": runner.CANONICAL_RELEASE["dataset_id"],
            "inputs_sha256": runner.CANONICAL_RELEASE["inputs_sha256"],
        },
        "selected": [],
        "source_audit": {
            "commit": runner.MODEL_SOURCE_COMMIT,
            "eval_single": {"commit": runner.EVAL_SINGLE_COMMIT},
        },
        "asset_audit": {
            "checkpoint": {
                "actual_sha256": runner.CHECKPOINT["sha256"],
                "schema": {"items_sha256": "1" * 64},
            },
            "processor": {"revision": runner.PROCESSOR_HF_REVISION},
        },
        "golden": {"status": "passed"},
        "device_text": "cpu",
        "pair_limit": None,
        "sample_id": None,
        "bootstrap_samples": 10,
        "bootstrap_seed": 3,
    }
    first = runner._run_config(
        **common,
        model_audit={"construction": {"fused_attention": True}},
    )
    second = runner._run_config(
        **common,
        model_audit={"construction": {"fused_attention": False}},
    )
    assert runner._manifest_fingerprint(first) != (
        runner._manifest_fingerprint(second)
    )
