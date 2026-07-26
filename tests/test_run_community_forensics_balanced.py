from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from eval.opensource import run_community_forensics_balanced as runner
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
def verified_assets():
    return runner.verify_assets(
        source_root=runner.DEFAULT_SOURCE_ROOT,
        model_root=runner.DEFAULT_MODEL_ROOT,
        processor_root=runner.DEFAULT_PROCESSOR_ROOT,
    )


@pytest.fixture(scope="module")
def cpu_preflight():
    report = runner.run_cpu_preflight(
        repo_root=REPO_ROOT,
        source_root=runner.DEFAULT_SOURCE_ROOT,
        model_root=runner.DEFAULT_MODEL_ROOT,
        processor_root=runner.DEFAULT_PROCESSOR_ROOT,
    )
    source, assets, _state = runner.verify_assets(
        source_root=runner.DEFAULT_SOURCE_ROOT,
        model_root=runner.DEFAULT_MODEL_ROOT,
        processor_root=runner.DEFAULT_PROCESSOR_ROOT,
    )
    runner._validate_preflight_report(
        report,
        source=source,
        assets=assets,
    )
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


def _processed(
    raw_logit: float = 1.0,
    probability: float = 0.75,
) -> dict:
    decision = probability > runner.legacy.CLASSIFICATION_THRESHOLD
    classification = {
        "raw_logit": float(raw_logit),
        "probability": float(probability),
        "ai_score": float(probability),
        "score": float(probability),
        "decision": decision,
        "threshold": runner.legacy.CLASSIFICATION_THRESHOLD,
        "threshold_operator": runner.legacy.CLASSIFICATION_THRESHOLD_OPERATOR,
        "semantics": runner.legacy.SCORE_SEMANTICS,
    }
    t1 = {
        key: value for key, value in classification.items() if key != "semantics"
    }
    t1["policy"] = runner.legacy.T1_POLICY
    return {
        "raw_logit": float(raw_logit),
        "probability": float(probability),
        "ai_score": float(probability),
        "score": float(probability),
        "score_semantics": runner.legacy.SCORE_SEMANTICS,
        "classification_decision": decision,
        "classification_threshold": runner.legacy.CLASSIFICATION_THRESHOLD,
        "classification_threshold_operator": (
            runner.legacy.CLASSIFICATION_THRESHOLD_OPERATOR
        ),
        "classification": classification,
        "t1": t1,
        "manual_replay": {
            "raw_logit": float(raw_logit),
            "probability": float(probability),
            "ai_score": float(probability),
            "classification_decision": decision,
            "official_logit_exact_match": True,
            "official_probability_exact_match": True,
            "model_forward_calls": 1,
            "classifier_hook_calls": 1,
        },
    }


def _ok_result(
    root: Path,
    *,
    sample_id: str = "0" * 24,
    processed: dict | None = None,
    feature: np.ndarray | None = None,
) -> tuple[dict, dict, Path]:
    input_row = _real_input(root, sample_id=sample_id)
    feature_dir = (
        root
        / runner.DEFAULT_ARTIFACTS_DIR
        / "run-a"
        / "commfor_features"
    )
    feature_dir.mkdir(parents=True, exist_ok=True)
    value = (
        np.linspace(
            -1.0,
            1.0,
            runner.legacy.FEATURE_DIMENSION,
            dtype=np.float32,
        )
        if feature is None
        else np.ascontiguousarray(feature)
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
        feature_dir=feature_dir,
        processed=_processed() if processed is None else processed,
        feature=value,
        preprocess=preprocess,
        preprocess_latency_ms=1.0,
        latency_ms=2.0,
        peak_cuda_memory_bytes=0,
    )
    return result, input_row, feature_dir


def test_frozen_contracts_and_runtime_identity():
    assert runner.RUN_MANIFEST_SCHEMA == (
        "community_forensics_balanced_run_manifest_v2"
    )
    assert runner.RUN_CONFIG_SCHEMA == (
        "community_forensics_balanced_run_config_v2"
    )
    assert runner.RUNTIME_SUMMARY_SCHEMA == (
        "community_forensics_balanced_runtime_summary_v2"
    )
    assert runner.CPU_PREFLIGHT_SCHEMA == (
        "community_forensics_balanced_cpu_preflight_v1"
    )
    assert runner.FROZEN_PROFILE == "official_highres_resize440_centercrop384"
    assert runner.DEFAULT_SEED == 100
    assert runner.FROZEN_PYVENV_CONFIG_SHA256 == (
        "7a40b0582b3525537e9e005348ceec3a23259899af45afc367014c7acbdf91f4"
    )
    assert runner.FROZEN_PYTHON_EXECUTABLE == Path(
        "/root/.cache/claimforge/venvs/"
        "community-forensics-balanced-nightly20250627/bin/python"
    )
    assert runner.SCORE_SPEC.as_dict() == {
        "key": "ai_score",
        "direction": "higher_means_fake",
        "fixed_threshold": 0.5,
        "threshold_operator": ">",
    }
    assert runner.FORMAL_SELECTED_ROWS_SHA256 == (
        "6b5128909eeffdbd88e61af02ca1bd191cb5460f94a23b47c87ebd0172e6d12c"
    )
    assert runner.FORMAL_SELECTED_IDS_SHA256 == (
        "e4418d86461f889e4a4423f26aab63243e6f63a435a49624881c34979b812e41"
    )
    assert runner.SMOKE5X7_SELECTED_IDS_SHA256 == (
        "b420bc581386a540b742d917d60d007f0e5522b6cca43fa217797944c40667e5"
    )
    assert runner.TASK_SCOPE == {
        "primary_task": "T1_whole_image_AIGC_detection",
        "valid_for_t1": True,
        "valid_for_t2": False,
        "localization_output": None,
        "native_dense_output": False,
    }


def test_model_preprocess_and_artifact_contracts_are_exact():
    assert runner.MODEL_CONTRACT["source_commit"] == (
        "ee5b71d43db0f3779e1edd64ee927b13f2dd6ad4"
    )
    assert runner.MODEL_CONTRACT["eval_single_commit"] == (
        "5e52ed690bdbd609f9bb1705c4c80d11872a05bd"
    )
    assert runner.EXPECTED_ASSET_BUNDLE_SHA256 == (
        "810a7592a82f09cbf638985e9c59eed9ebd2c3ff28ebab97f348bfd3c69b7fb3"
    )
    assert runner.PREPROCESS_CONTRACT["resize"]["short_side"] == 440
    assert runner.PREPROCESS_CONTRACT["center_crop"] == [384, 384]
    assert runner.ARTIFACT_CONTRACT["feature"] == {
        "format": "NumPy .npy, allow_pickle=False",
        "shape": [384],
        "dtype": "float32",
        "nbytes": 1536,
        "finite": True,
        "semantics": (
            "timm_vit_forward_head_pre_logits_classifier_input"
        ),
        "allow_pickle": False,
        "exact_head_and_sigmoid_replay_on_recorded_device": True,
        "visibility": "local_only_gitignored_output",
    }


def test_adapter_source_inventory_is_hash_bound():
    evidence = runner.adapter_source_contract(REPO_ROOT)
    assert set(evidence) == set(runner.ADAPTER_SOURCE_PATHS)
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
    from eval.opensource.balanced_run_contract import selected_ids_sha256

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
    from eval.opensource.balanced_run_contract import selected_ids_sha256

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


def test_formal_visibility_census_is_frozen(formal_visibility):
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
    assert formal_visibility["role"] == (
        "input_condition_diagnostic_not_model_localization"
    )


def test_real_and_fullframe_visibility_are_not_applicable(
    balanced_release,
):
    real = next(row for row in balanced_release.inputs if row["condition"] == "real")
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


def test_local_visibility_is_input_evidence(balanced_release):
    local = next(
        row
        for row in balanced_release.inputs
        if row["condition"] == "local_cat"
    )
    value = runner._visibility_diagnostic(local, repo_root=REPO_ROOT)
    assert value["edit_visibility"] in {"full", "partial", "none"}
    assert 0.0 <= value["edit_visible_gt_fraction"] <= 1.0
    assert value["edit_visibility_evidence"]["gt"]["profile_id"] == (
        runner.FROZEN_PROFILE
    )


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


def test_ok_result_and_feature_contract(tmp_path):
    result, input_row, feature_dir = _ok_result(tmp_path)
    assert result["schema_version"] == "opensource_result_v2"
    assert result["status"] == "ok"
    assert result["valid_for_metrics"] is True
    assert result["ai_score"] == result["probability"] == 0.75
    assert result["task_scope"]["valid_for_t2"] is False
    assert result["edit_visibility"] == "not_applicable"
    assert result["commfor_feature_shape"] == [384]
    assert result["commfor_feature_nbytes"] == 1536
    assert len(list(feature_dir.glob("*.npy"))) == 1
    runner._validate_runner_attempt(
        result,
        input_row=input_row,
        repo_root=tmp_path,
        run_id="run-a",
        run_manifest_fingerprint="a" * 64,
    )


def test_probability_alias_and_range_validation(tmp_path):
    result, _input, _features = _ok_result(tmp_path)
    changed = copy.deepcopy(result)
    changed["ai_score"] = 0.5
    with pytest.raises(ValueError, match="aliases"):
        runner._validate_score_payload(changed, sample_id=result["sample_id"])
    changed = copy.deepcopy(result)
    changed["probability"] = 1.1
    with pytest.raises(ValueError, match=r"outside \[0,1\]"):
        runner._validate_score_payload(changed, sample_id=result["sample_id"])


def test_feature_tampering_is_rejected(tmp_path):
    result, _input, _features = _ok_result(tmp_path)
    path = tmp_path / result["commfor_feature_path"]
    payload = bytearray(path.read_bytes())
    payload[-1] ^= 1
    path.write_bytes(payload)
    with pytest.raises(ValueError, match="metadata/hash|array"):
        runner._validate_feature_artifact(
            result,
            sample_id=result["sample_id"],
            repo_root=tmp_path,
            run_id="run-a",
        )


def test_same_device_feature_head_replay(tmp_path):
    import torch

    feature = np.zeros(runner.legacy.FEATURE_DIMENSION, dtype=np.float32)
    raw = 1.0
    probability = float(torch.sigmoid(torch.tensor(raw, dtype=torch.float32)))
    result, _input, _features = _ok_result(
        tmp_path,
        processed=_processed(raw, probability),
        feature=feature,
    )
    state = {
        "vit.head.weight": torch.zeros(
            (1, runner.legacy.FEATURE_DIMENSION),
            dtype=torch.float32,
        ),
        "vit.head.bias": torch.tensor([raw], dtype=torch.float32),
    }
    replayed = runner._validate_latest_feature_head_replay(
        latest_by_sample_id={result["sample_id"]: result},
        repo_root=tmp_path,
        run_id="run-a",
        state=state,
        device=torch.device("cpu"),
    )
    assert replayed == 1
    result["probability"] = 0.1
    with pytest.raises(ValueError, match="head replay mismatch"):
        runner._validate_latest_feature_head_replay(
            latest_by_sample_id={result["sample_id"]: result},
            repo_root=tmp_path,
            run_id="run-a",
            state=state,
            device=torch.device("cpu"),
        )


def test_error_result_is_t1_only(tmp_path):
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
    assert "ai_score" not in result
    assert result["error_type"] == "RuntimeError"


def test_attempt_history_rejects_success_duplicates_and_post_success():
    runner._validate_physical_attempt_history(
        [
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


def test_feature_inventory_rejects_extra_files(tmp_path):
    result, _input, feature_dir = _ok_result(tmp_path)
    assert runner._validate_feature_inventory(
        feature_dir=feature_dir,
        latest_by_sample_id={result["sample_id"]: result},
        repo_root=tmp_path,
        run_id="run-a",
    ) == 1
    (feature_dir / "extra.npy").write_bytes(b"x")
    with pytest.raises(ValueError, match="inventory mismatch"):
        runner._validate_feature_inventory(
            feature_dir=feature_dir,
            latest_by_sample_id={result["sample_id"]: result},
            repo_root=tmp_path,
            run_id="run-a",
        )


def test_output_directories_are_disjoint_and_resume_safe(tmp_path):
    run_dir = tmp_path / "results" / "run-a"
    feature_root = (
        tmp_path / runner.DEFAULT_ARTIFACTS_DIR / "run-a"
    )
    directory = runner._prepare_output_directories(
        repo_root=tmp_path,
        run_dir=run_dir,
        feature_root=feature_root,
        resume=False,
    )
    assert directory == feature_root / "commfor_features"
    (run_dir / "manifest.json").write_text("{}", encoding="utf-8")
    (run_dir / "expected_inputs.jsonl").write_text("{}\n", encoding="utf-8")
    assert runner._prepare_output_directories(
        repo_root=tmp_path,
        run_dir=run_dir,
        feature_root=feature_root,
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
            feature_root=feature_root,
            resume=True,
        )
    (run_dir / "balanced250_metrics.json").unlink()
    with pytest.raises(ValueError, match="disjoint"):
        runner._prepare_output_directories(
            repo_root=tmp_path,
            run_dir=run_dir,
            feature_root=run_dir,
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
    for name in ("results.jsonl", "summary.json", "balanced250_metrics.json",
                 "independent_audit.json"):
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
    row = {"a": 1, "b": 2}
    jsonl.write_text(f"{stable_json(row)}\n", encoding="utf-8")
    assert runner._read_jsonl_strict(jsonl, "rows") == [row]


@pytest.mark.parametrize(
    "payload",
    [
        {"pair_rank": 1},
        {"valid_for_t2": True},
        {"native_dense_output": True},
        {"localization_output": "map"},
        {"pixel_ap": 0.5},
        {"heatmap_path": "x.npy"},
    ],
)
def test_t2_and_pair_claims_are_rejected(payload):
    with pytest.raises(ValueError, match="unsupported"):
        runner._reject_unsupported_claims(payload)
    runner._reject_unsupported_claims(runner.TASK_SCOPE)


def test_assets_are_strict_and_complete(verified_assets):
    source, assets, state = verified_assets
    assert source["commit"] == runner.legacy.MODEL_SOURCE_COMMIT
    assert source["tracked_dirty"] is False
    assert assets["bundle_sha256"] == runner.EXPECTED_ASSET_BUNDLE_SHA256
    checkpoint = assets["checkpoint"]
    assert checkpoint["actual_sha256"] == runner.legacy.CHECKPOINT["sha256"]
    assert checkpoint["actual_bytes"] == 87_262_324
    assert checkpoint["schema"]["tensor_count"] == 152
    assert checkpoint["schema"]["state_elements"] == 21_811_969
    assert checkpoint["schema"]["items_sha256"] == (
        runner.EXPECTED_CHECKPOINT_SCHEMA_SHA256
    )
    assert len(state) == 152


def test_cpu_preflight_has_two_independent_gates(cpu_preflight):
    assert cpu_preflight["status"] == "passed"
    assert cpu_preflight["cuda_used"] is False
    assert cpu_preflight["dataset_manifest_loaded"] is False
    official = cpu_preflight["official_golden"]
    assert len(official["cases"]) == 5
    assert max(case["absolute_difference"] for case in official["cases"]) == (
        pytest.approx(4.2596481897305694e-11, rel=0.0, abs=1e-16)
    )
    golden = cpu_preflight["balanced_golden"]
    assert golden["sample_id"] == runner.CPU_GOLDEN_SAMPLE_ID
    assert golden["feature_array_sha256"] == (
        runner.CPU_GOLDEN_FEATURE_ARRAY_SHA256
    )
    assert golden["feature_file_sha256"] == (
        runner.CPU_GOLDEN_FEATURE_FILE_SHA256
    )
    assert golden["raw_logit"] == runner.CPU_GOLDEN_RAW_LOGIT
    assert golden["probability"] == runner.CPU_GOLDEN_PROBABILITY
    assert golden["repeat_byte_exact"] is True


def test_preflight_validation_rejects_tamper(cpu_preflight):
    source = cpu_preflight["source"]
    assets = cpu_preflight["assets"]
    changed = copy.deepcopy(cpu_preflight)
    changed["balanced_golden"]["probability"] += 1e-12
    with pytest.raises(ValueError, match="probability"):
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


def test_runtime_contract_is_frozen_cpu():
    device, runtime = runner.configure_runtime("cpu")
    assert str(device) == "cpu"
    assert runtime["python"]["executable"] == str(
        runner.FROZEN_PYTHON_EXECUTABLE
    )
    assert runtime["venv"]["include_system_site_packages"] is True
    assert runtime["packages"]["timm"] == "1.0.15"
    assert runtime["cudnn"]["enabled"] is False
    runner.validate_runtime_contract(runtime)
    changed = copy.deepcopy(runtime)
    changed["packages"]["timm"] = "different"
    with pytest.raises(ValueError, match="packages"):
        runner.validate_runtime_contract(changed)


def test_immutable_config_binds_selection_and_outputs(
    balanced_release,
    cpu_preflight,
    tmp_path,
):
    spec, selected = runner.select_mode_inputs(
        balanced_release,
        mode="single",
        per_condition_limit=None,
        sample_id=runner.CPU_GOLDEN_SAMPLE_ID,
    )
    from eval.opensource.balanced_run_contract import build_run_dataset_contract

    contract = build_run_dataset_contract(
        balanced_release,
        spec,
        selected,
        score_spec=runner.SCORE_SPEC,
    )
    source = cpu_preflight["source"]
    assets = cpu_preflight["assets"]
    _device, runtime = runner.configure_runtime("cpu")
    run_dir = REPO_ROOT / "results/opensource/community_forensics/test-run"
    feature_dir = (
        REPO_ROOT
        / runner.DEFAULT_ARTIFACTS_DIR
        / "test-run"
        / "commfor_features"
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
        source=source,
        assets=assets,
        runtime=runtime,
        cpu_preflight=cpu_preflight,
        run_dir=run_dir,
        results_path=run_dir / "results.jsonl",
        expected_inputs_path=run_dir / "expected_inputs.jsonl",
        summary_path=run_dir / "summary.json",
        feature_dir=feature_dir,
    )
    assert set(immutable) == runner.IMMUTABLE_CONFIG_KEYS
    assert immutable["task_scope"]["valid_for_t2"] is False
    assert immutable["dataset_contract"]["capability"]["name"] == (
        "whole_image_t1"
    )
    assert immutable["local_artifact_policy"]["gitignored"] is True
    assert len(runner._fingerprint(immutable)) == 64


def test_parser_modes_and_defaults():
    parser = runner._build_parser()
    args = parser.parse_args([])
    assert args.mode == "formal"
    assert args.device is None
    assert args.seed == 100
    assert args.results_dir == runner.DEFAULT_RESULTS_DIR
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


def test_preflight_cli_rejects_mutating_options(monkeypatch):
    args = argparse.Namespace(
        repo_root=REPO_ROOT,
        dataset_manifest=runner.DEFAULT_DATASET_MANIFEST,
        source_root=runner.DEFAULT_SOURCE_ROOT,
        model_root=runner.DEFAULT_MODEL_ROOT,
        processor_root=runner.DEFAULT_PROCESSOR_ROOT,
        results_dir=runner.DEFAULT_RESULTS_DIR,
        artifacts_dir=runner.DEFAULT_ARTIFACTS_DIR,
        run_id="bad",
        mode="preflight",
        per_condition_limit=None,
        sample_id=None,
        device="cpu",
        seed=100,
        resume=False,
        fail_fast=False,
    )
    with pytest.raises(ValueError, match="preflight accepts"):
        runner.run(args)
