from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import os
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from eval.opensource import run_bfree_balanced as runner
from eval.opensource.balanced_run_contract import (
    build_run_dataset_contract,
    selected_ids_sha256,
)
from eval.opensource.canonical_release import (
    BALANCED_CONDITIONS,
    Capability,
    load_canonical_release,
)
from eval.opensource.common import stable_json


REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def balanced_release():
    return load_canonical_release(
        REPO_ROOT,
        runner.DEFAULT_DATASET_MANIFEST,
        verify_files=True,
    )


@pytest.fixture(scope="module")
def formal_selection(balanced_release):
    _spec, selected = runner.select_mode_inputs(
        balanced_release,
        mode="formal",
        per_condition_limit=None,
        sample_id=None,
    )
    return selected


@pytest.fixture(scope="module")
def formal_visibility(formal_selection):
    return runner.selection_visibility_census(
        formal_selection,
        repo_root=REPO_ROOT,
    )


@pytest.fixture(scope="module")
def cpu_preflight():
    report = runner.run_cpu_preflight(
        repo_root=REPO_ROOT,
        source_root=runner.DEFAULT_SOURCE_ROOT,
        weights_dir=runner.DEFAULT_WEIGHTS_DIR,
        weights_zip=runner.DEFAULT_WEIGHTS_ZIP,
    )
    source, assets, state = runner.verify_assets(
        source_root=runner.DEFAULT_SOURCE_ROOT,
        weights_dir=runner.DEFAULT_WEIGHTS_DIR,
        weights_zip=runner.DEFAULT_WEIGHTS_ZIP,
    )
    try:
        runner._validate_preflight_report(
            report,
            source=source,
            assets=assets,
        )
    finally:
        del state
        gc.collect()
    return report


def _real_input(
    root: Path,
    *,
    sample_id: str = "0" * 24,
) -> dict:
    relative = Path("inputs") / f"{sample_id}.jpg"
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (640, 480), color=(10, 20, 30)).save(
        path,
        format="JPEG",
        quality=95,
    )
    return {
        "dataset_id": "test-balanced250",
        "sample_id": sample_id,
        "rank": 0,
        "condition": "real",
        "condition_family": "real",
        "manipulation_scope": "authentic",
        "normalized_task_id": "lodging_000_slot_001",
        "task_id": "lodging_000_slot_001",
        "kind": "real",
        "label": 0,
        "domain": "lodging",
        "gt_mask_kind": "all_zero",
        "gt_mask_path": None,
        "gt_mask_sha256": None,
        "gt_positive_pixels": 0,
        "canonical_path": relative.as_posix(),
        "canonical_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "canonical_bytes": path.stat().st_size,
        "width": 640,
        "height": 480,
        "panel": True,
    }


def _scoring(
    raw_logit: float = 1.0,
    crop_logits: np.ndarray | None = None,
) -> dict:
    crop = (
        np.full(runner.legacy.CROP_COUNT, raw_logit, dtype=np.float32)
        if crop_logits is None
        else np.ascontiguousarray(crop_logits, dtype=np.float32)
    )
    raw = float(raw_logit)
    probability = runner.legacy.bfree_fake_probability_float32(raw)
    decision = raw > runner.legacy.CLASSIFICATION_THRESHOLD
    classification = {
        "raw_logit": raw,
        "ai_score": raw,
        "fake_probability": probability,
        "decision": decision,
        "threshold": runner.legacy.CLASSIFICATION_THRESHOLD,
        "threshold_operator": (
            runner.legacy.CLASSIFICATION_THRESHOLD_OPERATOR
        ),
        "semantics": runner.legacy.SCORE_SEMANTICS,
    }
    t1 = {
        key: value
        for key, value in classification.items()
        if key != "semantics"
    }
    t1["policy"] = runner.legacy.T1_POLICY
    crop_values = [float(value) for value in crop.tolist()]
    return {
        "raw_logit": raw,
        "ai_score": raw,
        "score": raw,
        "fake_probability": probability,
        "crop_logits": crop_values,
        "score_semantics": runner.legacy.SCORE_SEMANTICS,
        "classification_decision": decision,
        "classification_threshold": runner.legacy.CLASSIFICATION_THRESHOLD,
        "classification_threshold_operator": (
            runner.legacy.CLASSIFICATION_THRESHOLD_OPERATOR
        ),
        "classification": classification,
        "t1": t1,
        "manual_replay": {
            "crop_logits": crop_values,
            "raw_logit": raw,
            "ai_score": raw,
            "official_crop_logits_exact_match": True,
            "official_mean_exact_match": True,
            "model_forward_calls": 1,
            "classifier_hook_calls": 1,
        },
    }


def _ok_result(
    root: Path,
    *,
    sample_id: str = "0" * 24,
    features: np.ndarray | None = None,
    crop_logits: np.ndarray | None = None,
    raw_logit: float = 1.0,
) -> tuple[dict, dict, Path]:
    input_row = _real_input(root, sample_id=sample_id)
    artifact_dir = (
        root
        / runner.DEFAULT_ARTIFACTS_DIR
        / "run-a"
        / "bfree_artifacts"
    )
    artifact_dir.mkdir(parents=True, exist_ok=True)
    feature_array = (
        np.zeros(
            (runner.legacy.CROP_COUNT, runner.legacy.FEATURE_DIMENSION),
            dtype=np.float32,
        )
        if features is None
        else np.ascontiguousarray(features, dtype=np.float32)
    )
    crop_array = (
        np.full(runner.legacy.CROP_COUNT, raw_logit, dtype=np.float32)
        if crop_logits is None
        else np.ascontiguousarray(crop_logits, dtype=np.float32)
    )
    _image, preprocess = runner.legacy.preprocess_image(
        root / input_row["canonical_path"]
    )
    result = runner._build_ok_result(
        input_row=input_row,
        repo_root=root,
        run_id="run-a",
        fingerprint="a" * 64,
        asset_bundle_sha256=runner.EXPECTED_ASSET_BUNDLE_SHA256,
        artifact_dir=artifact_dir,
        scoring=_scoring(raw_logit, crop_array),
        features=feature_array,
        crop_logits=crop_array,
        preprocess=preprocess,
        preprocess_latency_ms=1.0,
        latency_ms=2.0,
        peak_cuda_memory_bytes=0,
    )
    return result, input_row, artifact_dir


def _run_args(**overrides):
    values = {
        "repo_root": REPO_ROOT,
        "dataset_manifest": runner.DEFAULT_DATASET_MANIFEST,
        "source_root": runner.DEFAULT_SOURCE_ROOT,
        "weights_dir": runner.DEFAULT_WEIGHTS_DIR,
        "weights_zip": runner.DEFAULT_WEIGHTS_ZIP,
        "results_dir": runner.DEFAULT_RESULTS_DIR,
        "artifacts_dir": runner.DEFAULT_ARTIFACTS_DIR,
        "run_id": None,
        "mode": "formal",
        "per_condition_limit": None,
        "sample_id": None,
        "device": None,
        "seed": runner.DEFAULT_SEED,
        "resume": False,
        "fail_fast": False,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_frozen_runner_runtime_and_run_id_contracts():
    assert runner.RUN_MANIFEST_SCHEMA == "bfree_balanced_run_manifest_v2"
    assert runner.RUN_CONFIG_SCHEMA == "bfree_balanced_run_config_v2"
    assert runner.RUNTIME_SUMMARY_SCHEMA == (
        "bfree_balanced_runtime_summary_v2"
    )
    assert runner.CPU_PREFLIGHT_SCHEMA == "bfree_balanced_cpu_preflight_v1"
    assert runner.DEFAULT_FORMAL_RUN_ID == (
        "bfree_dino2reg4_balanced250_v1_full1775_20260727"
    )
    assert runner.DEFAULT_SMOKE_RUN_ID_A == (
        "bfree_dino2reg4_balanced250_v1_smoke5x7_a_r3_20260727"
    )
    assert runner.DEFAULT_SMOKE_RUN_ID_B == (
        "bfree_dino2reg4_balanced250_v1_smoke5x7_b_r3_20260727"
    )
    assert runner.DEFAULT_SEED == runner.legacy.MODEL_SEED == 20260725
    assert runner.FROZEN_PYTHONHASHSEED == "0"
    assert runner.FROZEN_PYTHON_EXECUTABLE == Path(
        "/root/.cache/claimforge/venvs/bfree/bin/python"
    )
    assert runner.FROZEN_PYTHONPYCACHEPREFIX == Path(
        "/root/.cache/claimforge/pycache/bfree-balanced-v2-empty"
    )
    assert runner.FROZEN_PYVENV_CONFIG_SHA256 == (
        "1ee492ad073827f75ebf74bf270e554ee23a28ee44756d616218d4bd6e40c6cc"
    )
    assert runner.SCORE_SPEC.as_dict() == {
        "key": "ai_score",
        "direction": "higher_means_fake",
        "fixed_threshold": 0.0,
        "threshold_operator": ">",
    }
    assert runner.TASK_SCOPE == {
        "primary_task": "T1_whole_image_AIGC_detection",
        "valid_for_t1": True,
        "valid_for_t2": False,
        "localization_output": None,
        "native_dense_output": False,
    }


def test_model_preprocess_and_npz_contracts_are_exact():
    assert runner.MODEL_CONTRACT["source_commit"] == (
        "c6a9f898782fb466b29af01f21960b67415afb0e"
    )
    assert runner.MODEL_CONTRACT["checkpoint_sha256"] == (
        "5948ca78f4d94e820c250d24cdf155035b4a85960443800bfe6bb7f06bffe947"
    )
    assert runner.EXPECTED_ASSET_BUNDLE_SHA256 == (
        "58859ff170ba42edd9c13bfcbc0094513de227d7001e5a261f7c37dd69db8349"
    )
    assert runner.PREPROCESS_CONTRACT["resize"] is False
    assert runner.PREPROCESS_CONTRACT["crop_size_pixels"] == 504
    assert runner.PREPROCESS_CONTRACT["crop_count"] == 5
    assert runner.ARTIFACT_CONTRACT["keys"] == [
        "features",
        "crop_logits",
    ]
    assert runner.ARTIFACT_CONTRACT["file_bytes"] == 15_904
    assert runner.ARTIFACT_CONTRACT["features"]["shape"] == [5, 768]
    assert runner.ARTIFACT_CONTRACT["features"]["nbytes"] == 15_360
    assert runner.ARTIFACT_CONTRACT["crop_logits"]["shape"] == [5]
    assert runner.ARTIFACT_CONTRACT["crop_logits"]["nbytes"] == 20
    assert runner.ARTIFACT_CONTRACT["zip_members"] == {
        "features.npy": {
            "compress_type": zipfile.ZIP_STORED,
            "file_size": 15_488,
            "compress_size": 15_488,
        },
        "crop_logits.npy": {
            "compress_type": zipfile.ZIP_STORED,
            "file_size": 148,
            "compress_size": 148,
        },
    }


def test_adapter_source_inventory_is_dynamic_and_hash_bound():
    evidence = runner.adapter_source_contract(REPO_ROOT)
    assert set(evidence) == set(runner.ADAPTER_SOURCE_PATHS)
    assert "eval/opensource/run_bfree_balanced.py" in evidence
    assert "eval/opensource/analyze_bfree_balanced.py" in evidence
    for relative, record in evidence.items():
        path = REPO_ROOT / relative
        assert record == {
            "path": relative,
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }


def test_formal_selection_is_exact(formal_selection):
    assert len(formal_selection) == 1775
    assert Counter(row["condition"] for row in formal_selection) == Counter(
        runner.FORMAL_COUNTS
    )
    assert [row["rank"] for row in formal_selection] == list(range(1775))
    assert all("pair_rank" not in row for row in formal_selection)
    assert runner._rows_sha256(formal_selection) == (
        "6b5128909eeffdbd88e61af02ca1bd191cb5460f94a23b47c87ebd0172e6d12c"
    )
    assert selected_ids_sha256(
        row["sample_id"] for row in formal_selection
    ) == "e4418d86461f889e4a4423f26aab63243e6f63a435a49624881c34979b812e41"


def test_smoke_selection_is_five_per_condition(balanced_release):
    spec, selected = runner.select_mode_inputs(
        balanced_release,
        mode="smoke",
        per_condition_limit=None,
        sample_id=None,
    )
    assert spec.capability is Capability.WHOLE_IMAGE_T1
    assert len(selected) == 35
    assert Counter(row["condition"] for row in selected) == Counter(
        {condition: 5 for condition in BALANCED_CONDITIONS}
    )
    assert [row["rank"] for row in selected] == sorted(
        row["rank"] for row in selected
    )
    assert selected_ids_sha256(
        row["sample_id"] for row in selected
    ) == "b420bc581386a540b742d917d60d007f0e5522b6cca43fa217797944c40667e5"


@pytest.mark.parametrize(
    ("mode", "limit", "sample_id", "pattern"),
    [
        ("formal", 1, None, "does not accept"),
        ("formal", None, "0" * 24, "does not accept"),
        ("smoke", 0, None, r"\[1, 250\]"),
        ("smoke", 251, None, r"\[1, 250\]"),
        ("smoke", True, None, r"\[1, 250\]"),
        ("single", None, None, "requires"),
        ("single", 1, "0" * 24, "does not accept"),
        ("bad", None, None, "unsupported"),
    ],
)
def test_mode_selector_rejects_invalid_combinations(
    balanced_release,
    mode,
    limit,
    sample_id,
    pattern,
):
    with pytest.raises(ValueError, match=pattern):
        runner.select_mode_inputs(
            balanced_release,
            mode=mode,
            per_condition_limit=limit,
            sample_id=sample_id,
        )


def test_single_selection(balanced_release):
    expected = balanced_release.inputs[123]
    spec, selected = runner.select_mode_inputs(
        balanced_release,
        mode="single",
        per_condition_limit=None,
        sample_id=expected["sample_id"],
    )
    assert spec.sample_id == expected["sample_id"]
    assert selected == [expected]


def test_formal_visibility_and_geometry_census_are_frozen(
    formal_visibility,
):
    assert formal_visibility["by_condition"] == {
        condition: runner.LOCAL_VISIBILITY_CENSUS[condition]
        for condition in (
            "local_mouse",
            "local_cat",
            "local_trash_can",
        )
    }
    assert formal_visibility["all_local"] == (
        runner.LOCAL_VISIBILITY_CENSUS["all_local"]
    )
    assert formal_visibility["not_applicable_images"] == 1025
    assert formal_visibility["replicate_wrap_total"] == 165
    assert formal_visibility["replicate_wrap_by_condition"] == (
        runner.FORMAL_GEOMETRY_CENSUS["replicate_wrap_by_condition"]
    )
    assert formal_visibility["distinct_crop_starts_all"] == {
        "1": 165,
        "3": 7,
        "5": 1603,
    }
    assert formal_visibility["distinct_crop_starts_by_condition"] == (
        runner.FORMAL_GEOMETRY_CENSUS[
            "distinct_crop_starts_by_condition"
        ]
    )
    assert formal_visibility["role"] == (
        "input_condition_stratum_not_model_localization"
    )
    assert json.loads(stable_json(formal_visibility)) == formal_visibility


def test_real_and_fullframe_visibility_are_not_applicable(
    balanced_release,
):
    real = next(
        row for row in balanced_release.inputs if row["condition"] == "real"
    )
    fullframe = next(
        row
        for row in balanced_release.inputs
        if row["condition"] == "fullframe_mouse"
    )
    for row in (real, fullframe):
        value = runner._visibility_diagnostic(row, repo_root=REPO_ROOT)
        assert value["edit_visibility"] == "not_applicable"
        assert value["edit_visible_gt_fraction"] is None
        assert value["edit_visibility_evidence"]["gt_mask_kind"] == (
            row["gt_mask_kind"]
        )
        assert value["edit_visibility_evidence"]["role"] == (
            "input_condition_stratum_not_model_localization"
        )


def test_local_visibility_is_exact_input_evidence(balanced_release):
    local = next(
        row
        for row in balanced_release.inputs
        if row["condition"] == "local_cat"
    )
    value = runner._visibility_diagnostic(local, repo_root=REPO_ROOT)
    evidence = value["edit_visibility_evidence"]
    assert value["edit_visibility"] in {"full", "partial", "none"}
    assert 0.0 <= value["edit_visible_gt_fraction"] <= 1.0
    assert evidence["gt_mask_kind"] == "exact_diff"
    assert evidence["basis"].startswith("exact_diff_positive_pixels")
    assert evidence["role"] == (
        "input_condition_stratum_not_model_localization"
    )
    assert evidence["visible_positive_pixels"] <= evidence["positive_pixels"]


@pytest.mark.parametrize(
    "value",
    [
        "",
        ".",
        "..",
        "../x",
        "/tmp/x",
        "x/y",
        "x\\y",
        "a" * 161,
        True,
    ],
)
def test_run_id_safety(value):
    with pytest.raises(ValueError, match="run-id"):
        runner._valid_run_id(value)
    assert runner._valid_run_id("safe-run_1.0") == "safe-run_1.0"


def test_safe_repo_file_rejects_traversal_and_symlink(tmp_path):
    row = _real_input(tmp_path)
    path = runner._safe_repo_file(
        row["canonical_path"],
        repo_root=tmp_path,
        label="input",
    )
    assert path.is_file()
    with pytest.raises(ValueError, match="canonical"):
        runner._safe_repo_file(
            "../escape.jpg",
            repo_root=tmp_path,
            label="input",
        )
    target = tmp_path / "target.jpg"
    target.write_bytes(b"x")
    link = tmp_path / "link.jpg"
    link.symlink_to(target)
    with pytest.raises(ValueError, match="symlink"):
        runner._safe_repo_file(
            "link.jpg",
            repo_root=tmp_path,
            label="input",
        )


def test_recursive_same_json_comparison_rejects_bool_integer_aliases():
    assert runner._same_json_type_and_value(
        {"a": [False, None, {"b": 1}]},
        {"a": [False, None, {"b": 1}]},
    )
    assert not runner._same_json_type_and_value(
        {"a": [True]},
        {"a": [1]},
    )
    assert not runner._same_json_type_and_value(
        {"a": {"b": False}},
        {"a": {"b": 0}},
    )
    with pytest.raises(ValueError, match="changed"):
        runner._require_same_json({"a": True}, {"a": 1}, "identity")


def test_ok_result_and_exact_npz_contract(tmp_path):
    result, input_row, artifact_dir = _ok_result(tmp_path)
    path = artifact_dir / f"{result['sample_id']}.npz"
    assert result["schema_version"] == "opensource_result_v2"
    assert result["status"] == "ok"
    assert result["valid_for_metrics"] is True
    assert result["ai_score"] == result["raw_logit"] == 1.0
    assert result["task_scope"]["valid_for_t2"] is False
    assert result["edit_visibility"] == "not_applicable"
    assert result["feature_shape"] == [5, 768]
    assert result["crop_logits_shape"] == [5]
    assert path.stat().st_size == 15_904
    with zipfile.ZipFile(path) as archive:
        assert archive.namelist() == ["features.npy", "crop_logits.npy"]
        assert {
            info.filename: (
                info.compress_type,
                info.file_size,
                info.compress_size,
            )
            for info in archive.infolist()
        } == {
            "features.npy": (zipfile.ZIP_STORED, 15_488, 15_488),
            "crop_logits.npy": (zipfile.ZIP_STORED, 148, 148),
        }
    runner._validate_runner_attempt(
        result,
        input_row=input_row,
        repo_root=tmp_path,
        run_id="run-a",
        run_manifest_fingerprint="a" * 64,
    )
    round_trip = json.loads(stable_json(result))
    assert runner._same_json_type_and_value(round_trip, result)
    runner._validate_runner_attempt(
        round_trip,
        input_row=input_row,
        repo_root=tmp_path,
        run_id="run-a",
        run_manifest_fingerprint="a" * 64,
    )


def test_raw_logit_zero_uses_strict_greater_than(tmp_path):
    result, _input, _artifacts = _ok_result(
        tmp_path,
        raw_logit=0.0,
        crop_logits=np.zeros(5, dtype=np.float32),
    )
    assert result["raw_logit"] == 0.0
    assert result["fake_probability"] == 0.5
    assert result["classification_decision"] is False
    assert result["classification_threshold"] == 0.0
    assert result["classification_threshold_operator"] == ">"
    changed = copy.deepcopy(result)
    changed["classification_threshold_operator"] = ">="
    with pytest.raises(ValueError, match="semantics"):
        runner._validate_score_payload(
            changed,
            sample_id=result["sample_id"],
        )


def test_score_nested_bool_integer_tampering_is_rejected(tmp_path):
    result, _input, _artifacts = _ok_result(tmp_path)
    changed = copy.deepcopy(result)
    changed["manual_replay"]["official_mean_exact_match"] = 1
    with pytest.raises(ValueError, match="semantics"):
        runner._validate_score_payload(
            changed,
            sample_id=result["sample_id"],
        )
    changed = copy.deepcopy(result)
    changed["classification"]["decision"] = 1
    with pytest.raises(ValueError, match="semantics"):
        runner._validate_score_payload(
            changed,
            sample_id=result["sample_id"],
        )


def test_artifact_tampering_and_bool_shape_alias_are_rejected(tmp_path):
    result, _input, artifact_dir = _ok_result(tmp_path)
    changed = copy.deepcopy(result)
    changed["crop_logits_shape"] = [True]
    with pytest.raises(ValueError, match="alias"):
        runner._validate_artifact(
            changed,
            sample_id=result["sample_id"],
            repo_root=tmp_path,
            run_id="run-a",
        )
    path = artifact_dir / f"{result['sample_id']}.npz"
    path.write_bytes(path.read_bytes() + b"x")
    with pytest.raises(ValueError, match="byte size"):
        runner._validate_artifact(
            result,
            sample_id=result["sample_id"],
            repo_root=tmp_path,
            run_id="run-a",
        )


def test_object_or_extra_member_npz_is_rejected(tmp_path):
    features = np.zeros((5, 768), dtype=np.float32)
    logits = np.zeros(5, dtype=np.float32)
    extra = tmp_path / "extra.npz"
    np.savez(extra, features=features, crop_logits=logits, extra=np.zeros(1))
    with pytest.raises(ValueError, match="byte size|members"):
        runner._validate_npz_structure(
            extra.read_bytes(),
            sample_id="sample",
        )
    objects = tmp_path / "objects.npz"
    np.savez(
        objects,
        features=np.array([object()], dtype=object),
        crop_logits=logits,
    )
    with pytest.raises(ValueError, match="byte size|members"):
        runner._validate_npz_structure(
            objects.read_bytes(),
            sample_id="sample",
        )


def test_failed_result_construction_removes_orphan_npz(
    tmp_path,
    monkeypatch,
):
    row = _real_input(tmp_path)
    artifact_dir = (
        tmp_path
        / runner.DEFAULT_ARTIFACTS_DIR
        / "run-a"
        / "bfree_artifacts"
    )
    artifact_dir.mkdir(parents=True)
    features = np.zeros((5, 768), dtype=np.float32)
    crop_logits = np.ones(5, dtype=np.float32)
    _image, preprocess = runner.legacy.preprocess_image(
        tmp_path / row["canonical_path"]
    )

    def reject(*_args, **_kwargs):
        raise ValueError("forced self-validation failure")

    monkeypatch.setattr(runner, "_validate_runner_attempt", reject)
    with pytest.raises(ValueError, match="forced"):
        runner._build_ok_result(
            input_row=row,
            repo_root=tmp_path,
            run_id="run-a",
            fingerprint="a" * 64,
            asset_bundle_sha256=runner.EXPECTED_ASSET_BUNDLE_SHA256,
            artifact_dir=artifact_dir,
            scoring=_scoring(1.0, crop_logits),
            features=features,
            crop_logits=crop_logits,
            preprocess=preprocess,
            preprocess_latency_ms=1.0,
            latency_ms=2.0,
            peak_cuda_memory_bytes=0,
        )
    assert list(artifact_dir.iterdir()) == []


def test_preexisting_artifact_is_never_overwritten(tmp_path):
    row = _real_input(tmp_path)
    artifact_dir = (
        tmp_path
        / runner.DEFAULT_ARTIFACTS_DIR
        / "run-a"
        / "bfree_artifacts"
    )
    artifact_dir.mkdir(parents=True)
    path = artifact_dir / f"{row['sample_id']}.npz"
    path.write_bytes(b"do-not-overwrite")
    _image, preprocess = runner.legacy.preprocess_image(
        tmp_path / row["canonical_path"]
    )
    with pytest.raises(FileExistsError, match="already exists"):
        runner._build_ok_result(
            input_row=row,
            repo_root=tmp_path,
            run_id="run-a",
            fingerprint="a" * 64,
            asset_bundle_sha256=runner.EXPECTED_ASSET_BUNDLE_SHA256,
            artifact_dir=artifact_dir,
            scoring=_scoring(),
            features=np.zeros((5, 768), dtype=np.float32),
            crop_logits=np.ones(5, dtype=np.float32),
            preprocess=preprocess,
            preprocess_latency_ms=1.0,
            latency_ms=2.0,
            peak_cuda_memory_bytes=0,
        )
    assert path.read_bytes() == b"do-not-overwrite"


def test_same_device_five_crop_head_replay(tmp_path):
    import torch

    result, _input, _artifacts = _ok_result(tmp_path)
    state = {
        "head.weight": torch.zeros(
            (1, runner.legacy.FEATURE_DIMENSION),
            dtype=torch.float32,
        ),
        "head.bias": torch.tensor([1.0], dtype=torch.float32),
    }
    replayed = runner._validate_latest_artifact_head_replay(
        latest_by_sample_id={result["sample_id"]: result},
        repo_root=tmp_path,
        run_id="run-a",
        state=state,
        device=torch.device("cpu"),
    )
    assert replayed == 1
    changed = copy.deepcopy(result)
    changed["raw_logit"] = 0.5
    with pytest.raises(ValueError, match="head replay mismatch"):
        runner._validate_latest_artifact_head_replay(
            latest_by_sample_id={result["sample_id"]: changed},
            repo_root=tmp_path,
            run_id="run-a",
            state=state,
            device=torch.device("cpu"),
        )


def test_error_result_has_explicit_null_model_and_artifact_fields(tmp_path):
    row = _real_input(tmp_path)
    try:
        raise RuntimeError("boom")
    except RuntimeError as error:
        result = runner._build_error_result(
            input_row=row,
            repo_root=tmp_path,
            run_id="run-a",
            fingerprint="a" * 64,
            asset_bundle_sha256=runner.EXPECTED_ASSET_BUNDLE_SHA256,
            error=error,
        )
    assert result["status"] == "error"
    assert result["valid_for_metrics"] is False
    assert result["error_type"] == "RuntimeError"
    assert result["error"] == "boom"
    assert result["traceback"]
    assert set(runner._ERROR_NULL_FIELDS) <= set(result)
    assert all(result[key] is None for key in runner._ERROR_NULL_FIELDS)
    assert result["task_scope"]["valid_for_t2"] is False
    assert result["edit_visibility"] == "not_applicable"
    runner._validate_runner_attempt(
        json.loads(stable_json(result)),
        input_row=row,
        repo_root=tmp_path,
        run_id="run-a",
        run_manifest_fingerprint="a" * 64,
    )


def test_attempt_history_rejects_success_duplicates_and_post_success():
    runner._validate_physical_attempt_history(
        [
            {"sample_id": "a", "status": "error"},
            {"sample_id": "a", "status": "error"},
            {"sample_id": "a", "status": "ok"},
        ]
    )
    with pytest.raises(ValueError, match="duplicate successful"):
        runner._validate_physical_attempt_history(
            [
                {"sample_id": "a", "status": "ok"},
                {"sample_id": "a", "status": "ok"},
            ]
        )
    with pytest.raises(ValueError, match="after success"):
        runner._validate_physical_attempt_history(
            [
                {"sample_id": "a", "status": "ok"},
                {"sample_id": "a", "status": "error"},
            ]
        )


def test_artifact_inventory_is_exact_and_rejects_orphans(tmp_path):
    result, _input, artifact_dir = _ok_result(tmp_path)
    latest = {result["sample_id"]: result}
    assert runner._validate_artifact_inventory(
        artifact_dir=artifact_dir,
        latest_by_sample_id=latest,
        repo_root=tmp_path,
        run_id="run-a",
    ) == 1
    (artifact_dir / "orphan.npz").write_bytes(b"x")
    with pytest.raises(ValueError, match="inventory mismatch"):
        runner._validate_artifact_inventory(
            artifact_dir=artifact_dir,
            latest_by_sample_id=latest,
            repo_root=tmp_path,
            run_id="run-a",
        )


def test_output_directories_are_disjoint_and_resume_safe(tmp_path):
    run_dir = tmp_path / "results" / "run-a"
    artifact_root = tmp_path / runner.DEFAULT_ARTIFACTS_DIR / "run-a"
    directory = runner._prepare_output_directories(
        repo_root=tmp_path,
        run_dir=run_dir,
        artifact_root=artifact_root,
        resume=False,
    )
    assert directory == artifact_root / "bfree_artifacts"
    (run_dir / "manifest.json").write_text("{}", encoding="utf-8")
    (run_dir / "expected_inputs.jsonl").write_text("{}\n", encoding="utf-8")
    assert runner._prepare_output_directories(
        repo_root=tmp_path,
        run_dir=run_dir,
        artifact_root=artifact_root,
        resume=True,
    ) == directory
    (run_dir / "balanced250_metrics.json").write_text(
        "{}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="forbidden after analyzer outputs"):
        runner._prepare_output_directories(
            repo_root=tmp_path,
            run_dir=run_dir,
            artifact_root=artifact_root,
            resume=True,
        )
    (run_dir / "balanced250_metrics.json").unlink()
    with pytest.raises(ValueError, match="disjoint"):
        runner._prepare_output_directories(
            repo_root=tmp_path,
            run_dir=run_dir,
            artifact_root=run_dir,
            resume=True,
        )


def test_run_directory_allowlist(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    for name in ("manifest.json", "expected_inputs.jsonl"):
        (run_dir / name).write_text("{}\n", encoding="utf-8")
    runner._validate_run_dir_inventory(
        run_dir,
        allow_missing_results=True,
    )
    for name in (
        "results.jsonl",
        "summary.json",
        "balanced250_metrics.json",
        "independent_audit.json",
    ):
        (run_dir / name).write_text("{}\n", encoding="utf-8")
    runner._validate_run_dir_inventory(
        run_dir,
        allow_missing_results=False,
    )
    (run_dir / "unexpected.bin").write_bytes(b"x")
    with pytest.raises(ValueError, match="unexpected"):
        runner._validate_run_dir_inventory(
            run_dir,
            allow_missing_results=False,
        )


def test_strict_json_rejects_duplicates_constants_and_noncanonical_jsonl(
    tmp_path,
):
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"a":1,"a":2}', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        runner._load_json_strict(duplicate, "duplicate")
    constant = tmp_path / "constant.json"
    constant.write_text('{"a":NaN}', encoding="utf-8")
    with pytest.raises(ValueError, match="non-finite"):
        runner._load_json_strict(constant, "constant")
    jsonl = tmp_path / "rows.jsonl"
    jsonl.write_text('{"b": 2, "a": 1}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="canonical"):
        runner._read_jsonl_strict(jsonl, "rows")
    jsonl.write_text('{"a":1}', encoding="utf-8")
    with pytest.raises(ValueError, match="final newline"):
        runner._read_jsonl_strict(jsonl, "rows")
    row = {"a": 1, "b": 2}
    jsonl.write_text(f"{stable_json(row)}\n", encoding="utf-8")
    assert runner._read_jsonl_strict(jsonl, "rows") == [row]


@pytest.mark.parametrize(
    "payload",
    [
        {"pair_rank": 1},
        {"outer": [{"valid_for_t2": True}]},
        {"outer": {"native_dense_output": 0}},
        {"localization_output": "map"},
        {"pixel_ap": 0.5},
        {"nested": {"heatmap_path": "x.npy"}},
        {"nested": [{"predicted_mask": None}]},
    ],
)
def test_t2_pair_and_joint_claims_are_recursively_rejected(payload):
    with pytest.raises(ValueError, match="unsupported"):
        runner._reject_unsupported_claims(payload)
    runner._reject_unsupported_claims(runner.TASK_SCOPE)
    runner._reject_unsupported_claims(
        {
            "nested": {
                "valid_for_t2": False,
                "localization_output": None,
                "gt_mask_kind": "exact_diff",
            }
        }
    )


def test_source_and_assets_are_strict_and_complete():
    source, assets, state = runner.verify_assets(
        source_root=runner.DEFAULT_SOURCE_ROOT,
        weights_dir=runner.DEFAULT_WEIGHTS_DIR,
        weights_zip=runner.DEFAULT_WEIGHTS_ZIP,
    )
    try:
        assert source["commit"] == runner.legacy.MODEL_SOURCE_COMMIT
        assert source["tracked_dirty"] is False
        assert assets["bundle_sha256"] == (
            runner.EXPECTED_ASSET_BUNDLE_SHA256
        )
        assert assets["zip"]["verified_sha256"] == (
            runner.legacy.OFFICIAL_ZIP["sha256"]
        )
        checkpoint = assets["checkpoint"]
        assert checkpoint["sha256"] == (
            runner.legacy.CHECKPOINT["sha256"]
        )
        assert checkpoint["bytes"] == 346_171_370
        assert checkpoint["schema"]["tensor_count"] == 177
        assert checkpoint["schema"]["state_elements"] == 86_526_721
        assert checkpoint["schema"]["schema_sha256"] == (
            "e4bb9ddd115309740a70235152b7376e2c8299bb90baf243809f2a5e1665f524"
        )
        assert len(state) == 177
    finally:
        del state
        gc.collect()


def test_cpu_preflight_runs_official_four_and_balanced_gates(cpu_preflight):
    import torch

    assert cpu_preflight["status"] == "passed"
    assert cpu_preflight["cuda_used"] is False
    assert cpu_preflight["cuda_tensor_operations"] is False
    assert cpu_preflight["dataset_manifest_loaded"] is False
    assert torch.cuda.is_initialized() is False
    official = cpu_preflight["official_golden"]
    assert official["status"] == "passed"
    assert len(official["cases"]) == 4
    assert all(case["passed"] is True for case in official["cases"])
    assert all(case["repeat_bit_identical"] is True for case in official["cases"])
    golden = cpu_preflight["balanced_golden"]
    assert golden["sample_id"] == runner.CPU_GOLDEN_SAMPLE_ID
    assert golden["artifact_sha256"] == runner.CPU_GOLDEN_ARTIFACT_SHA256
    assert golden["artifact_bytes"] == 15_904
    assert golden["feature_array_sha256"] == (
        runner.CPU_GOLDEN_FEATURE_ARRAY_SHA256
    )
    assert golden["crop_logits_array_sha256"] == (
        runner.CPU_GOLDEN_CROP_LOGITS_ARRAY_SHA256
    )
    assert golden["raw_logit"] == runner.CPU_GOLDEN_RAW_LOGIT
    assert golden["fake_probability"] == runner.CPU_GOLDEN_FAKE_PROBABILITY
    assert golden["repeat_byte_exact"] is True


def test_preflight_validation_rejects_scalar_type_and_value_tamper(
    cpu_preflight,
):
    source = cpu_preflight["source"]
    assets = cpu_preflight["assets"]
    changed = copy.deepcopy(cpu_preflight)
    changed["balanced_golden"]["artifact_bytes"] = True
    with pytest.raises(ValueError, match="scalar"):
        runner._validate_preflight_report(
            changed,
            source=source,
            assets=assets,
        )
    changed = copy.deepcopy(cpu_preflight)
    changed["model_load"]["network"]["attempts"]["urllib_urlopen"] = False
    with pytest.raises(ValueError, match="model"):
        runner._validate_preflight_report(
            changed,
            source=source,
            assets=assets,
        )
    changed = copy.deepcopy(cpu_preflight)
    changed["cuda_used"] = True
    with pytest.raises(ValueError, match="provenance"):
        runner._validate_preflight_report(
            changed,
            source=source,
            assets=assets,
        )


def test_runtime_contract_is_frozen_cpu_and_startup_isolated():
    import torch

    assert torch.cuda.is_initialized() is False
    device, runtime = runner.configure_runtime("cpu")
    assert str(device) == "cpu"
    assert torch.cuda.is_initialized() is False
    assert runtime["python"]["executable"] == str(
        runner.FROZEN_PYTHON_EXECUTABLE
    )
    assert runtime["seed"] == 20260725
    assert runtime["process_environment"] == {
        "PYTHONHASHSEED": "0",
        "PYTHONDONTWRITEBYTECODE": "1",
        "NO_ALBUMENTATIONS_UPDATE": "1",
        "PYTHONPYCACHEPREFIX": str(runner.FROZEN_PYTHONPYCACHEPREFIX),
        "python_dont_write_bytecode": True,
        "sys_pycache_prefix": str(runner.FROZEN_PYTHONPYCACHEPREFIX),
        "pycache_prefix_initially_empty": True,
    }
    assert runtime["packages"]["timm"] == "1.0.12"
    assert runtime["cudnn"] == {
        "enabled": True,
        "benchmark": False,
        "deterministic": True,
        "allow_tf32": False,
    }
    runner.validate_runtime_contract(runtime)
    changed = copy.deepcopy(runtime)
    changed["batch_size"] = True
    with pytest.raises(ValueError, match="numerical"):
        runner.validate_runtime_contract(changed)
    changed = copy.deepcopy(runtime)
    changed["process_environment"]["python_dont_write_bytecode"] = 1
    with pytest.raises(ValueError, match="numerical"):
        runner.validate_runtime_contract(changed)


def test_startup_contract_rejects_wrong_hash_seed(monkeypatch):
    monkeypatch.setenv("PYTHONHASHSEED", str(runner.DEFAULT_SEED))
    with pytest.raises(RuntimeError, match="PYTHONHASHSEED=0"):
        runner._startup_isolation_contract()


def test_immutable_config_is_json_roundtrip_safe_and_binds_all_gates(
    balanced_release,
    cpu_preflight,
):
    spec, selected = runner.select_mode_inputs(
        balanced_release,
        mode="single",
        per_condition_limit=None,
        sample_id=runner.CPU_GOLDEN_SAMPLE_ID,
    )
    contract = build_run_dataset_contract(
        balanced_release,
        spec,
        selected,
        score_spec=runner.SCORE_SPEC,
    )
    run_dir = REPO_ROOT / "results/opensource/bfree/test-run"
    artifact_dir = (
        REPO_ROOT
        / runner.DEFAULT_ARTIFACTS_DIR
        / "test-run"
        / "bfree_artifacts"
    )
    immutable = runner.build_immutable_run_config(
        repo_root=REPO_ROOT,
        run_id="test-run",
        mode="single",
        dataset_contract=contract.as_dict(),
        selected=selected,
        selection_visibility=runner.selection_visibility_census(
            selected,
            repo_root=REPO_ROOT,
        ),
        adapter_sources=runner.adapter_source_contract(REPO_ROOT),
        source=cpu_preflight["source"],
        assets=cpu_preflight["assets"],
        runtime=cpu_preflight["runtime"],
        cpu_preflight=cpu_preflight,
        execution_model_load=cpu_preflight["model_load"],
        execution_official_golden=cpu_preflight["official_golden"],
        run_dir=run_dir,
        results_path=run_dir / "results.jsonl",
        expected_inputs_path=run_dir / "expected_inputs.jsonl",
        summary_path=run_dir / "summary.json",
        artifact_dir=artifact_dir,
    )
    assert set(immutable) == runner.IMMUTABLE_CONFIG_KEYS
    assert immutable["task_scope"]["valid_for_t2"] is False
    assert immutable["dataset_contract"]["capability"]["name"] == (
        "whole_image_t1"
    )
    assert immutable["cpu_preflight"][
        "performed_before_dataset_manifest_load"
    ] is True
    assert immutable["cpu_preflight"][
        "performed_before_accelerator_configuration"
    ] is True
    assert immutable["local_artifact_policy"]["gitignored"] is True
    assert len(runner._fingerprint(immutable)) == 64
    round_trip = json.loads(stable_json(immutable))
    assert runner._same_json_type_and_value(round_trip, immutable)


def test_parser_modes_and_defaults():
    parser = runner._build_parser()
    args = parser.parse_args([])
    assert args.mode == "formal"
    assert args.device is None
    assert args.seed == 20260725
    assert args.results_dir == runner.DEFAULT_RESULTS_DIR
    assert args.artifacts_dir == runner.DEFAULT_ARTIFACTS_DIR
    smoke = parser.parse_args(
        ["--mode", "smoke", "--run-id", "smoke-a"]
    )
    assert smoke.mode == "smoke"
    assert smoke.run_id == "smoke-a"


def test_result_identity_never_invents_pair_rank(tmp_path):
    row = _real_input(tmp_path)
    identity = runner.result_identity(
        row,
        repo_root=tmp_path,
        run_id="run-a",
        run_manifest_fingerprint="a" * 64,
        asset_bundle_sha256=runner.EXPECTED_ASSET_BUNDLE_SHA256,
        valid_for_metrics=True,
    )
    assert identity["schema_version"] == "opensource_result_v2"
    assert "pair_rank" not in identity
    assert identity["task_scope"] == {
        "valid_for_t1": True,
        "valid_for_t2": False,
        "native_dense_output": False,
    }


def test_preflight_cli_rejects_run_and_selection_options():
    with pytest.raises(ValueError, match="preflight accepts"):
        runner.run(
            _run_args(
                mode="preflight",
                run_id="bad",
                device="cpu",
            )
        )
    with pytest.raises(ValueError, match="preflight accepts"):
        runner.run(
            _run_args(
                mode="preflight",
                per_condition_limit=5,
                device="cpu",
            )
        )


def test_cpu_preflight_precedes_dataset_load_and_accelerator(
    monkeypatch,
):
    events: list[str] = []

    def fake_preflight(**_kwargs):
        events.append("cpu_preflight")
        return {}

    def fake_assets(**_kwargs):
        events.append("assets")
        return {}, {}, {}

    def fake_validate(*_args, **_kwargs):
        events.append("preflight_validation")

    class DatasetOpened(RuntimeError):
        pass

    def fake_dataset(*_args, **_kwargs):
        events.append("dataset_load")
        raise DatasetOpened

    def forbidden_accelerator(*_args, **_kwargs):
        raise AssertionError("accelerator configured before dataset sentinel")

    monkeypatch.setattr(runner, "run_cpu_preflight", fake_preflight)
    monkeypatch.setattr(runner, "verify_assets", fake_assets)
    monkeypatch.setattr(runner, "_validate_preflight_report", fake_validate)
    monkeypatch.setattr(runner, "load_canonical_release", fake_dataset)
    monkeypatch.setattr(runner, "configure_runtime", forbidden_accelerator)
    with pytest.raises(DatasetOpened):
        runner.run(_run_args(run_id="ordering-test", device="cuda:0"))
    assert events == [
        "cpu_preflight",
        "assets",
        "preflight_validation",
        "dataset_load",
    ]


def test_mode_and_seed_fail_before_preflight(monkeypatch):
    monkeypatch.setattr(
        runner,
        "run_cpu_preflight",
        lambda **_kwargs: pytest.fail("preflight must not run"),
    )
    with pytest.raises(ValueError, match="seed"):
        runner.run(_run_args(seed=True))
    with pytest.raises(ValueError, match="explicit --run-id"):
        runner.run(_run_args(mode="smoke"))


def test_artifact_root_is_fixed_to_gitignored_location(monkeypatch):
    monkeypatch.setattr(
        runner,
        "run_cpu_preflight",
        lambda **_kwargs: pytest.fail("preflight must not run"),
    )
    with pytest.raises(ValueError, match="artifacts-dir must be exactly"):
        runner.run(
            _run_args(
                artifacts_dir=Path("results/opensource/bfree-artifacts"),
            )
        )
