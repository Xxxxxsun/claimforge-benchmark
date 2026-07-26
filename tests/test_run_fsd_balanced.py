from __future__ import annotations

import argparse
import copy
import hashlib
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from eval.opensource import run_fsd_balanced as runner
from eval.opensource.canonical_release import (
    BALANCED_CONDITIONS,
    Capability,
    load_canonical_release,
)
from eval.opensource.common import read_jsonl


REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def balanced_release():
    return load_canonical_release(
        REPO_ROOT,
        runner.DEFAULT_DATASET_MANIFEST,
        verify_files=True,
    )


def _real_input(
    root: Path,
    *,
    sample_id: str = "0" * 24,
) -> dict:
    relative = Path("inputs") / f"{sample_id}.jpg"
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"synthetic-jpeg")
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
        "width": 32,
        "height": 32,
        "panel": True,
    }


def _processed(ai_score: float = 3.0) -> dict:
    raw = (
        runner.legacy.TRAIN_MEAN
        - float(ai_score) * runner.legacy.TRAIN_STD
    )
    z_score = (
        raw - runner.legacy.TRAIN_MEAN
    ) / runner.legacy.TRAIN_STD
    score = -z_score
    decision = score > runner.legacy.AI_SCORE_THRESHOLD
    released = z_score < runner.legacy.RELEASED_Z_THRESHOLD
    return {
        "raw_likelihood": raw,
        "released_z_score": z_score,
        "ai_score": score,
        "released_is_fake": released,
        "classification_decision": decision,
        "manual_replay": {
            "raw_likelihood": raw,
            "released_z_score": z_score,
            "ai_score": score,
            "released_is_fake": released,
            "classification_decision": decision,
            "official_raw_exact_match": True,
            "official_z_exact_match": True,
            "compute_fsd_calls": 1,
        },
    }


def _ok_result(
    root: Path,
    *,
    sample_id: str = "0" * 24,
    ai_score: float = 3.0,
    descriptor: np.ndarray | None = None,
) -> tuple[dict, dict, Path]:
    input_row = _real_input(root, sample_id=sample_id)
    descriptor_dir = (
        root
        / runner.DEFAULT_ARTIFACTS_DIR
        / "run-a"
        / "raw_descriptors"
    )
    descriptor_dir.mkdir(parents=True, exist_ok=True)
    value = (
        np.arange(runner.legacy.FSD_DIMENSION, dtype=np.float64)
        if descriptor is None
        else descriptor
    )
    result = runner._build_ok_result(
        input_row=input_row,
        repo_root=root,
        run_id="run-a",
        fingerprint="a" * 64,
        weights_bundle_sha256=runner.EXPECTED_WEIGHTS_BUNDLE_SHA256,
        descriptor_dir=descriptor_dir,
        processed=_processed(ai_score),
        descriptor=value,
        preprocess=runner.legacy.compute_preprocess_geometry(32, 32),
        preprocess_latency_ms=1.0,
        latency_ms=2.0,
        peak_cuda_memory_bytes=None,
    )
    return result, input_row, descriptor_dir


def test_frozen_contract_and_adapter_inventory():
    assert runner.RUN_MANIFEST_SCHEMA == "fsd_balanced_run_manifest_v2"
    assert runner.RUN_CONFIG_SCHEMA == "fsd_balanced_run_config_v2"
    assert runner.RUNTIME_SUMMARY_SCHEMA == "fsd_balanced_runtime_summary_v2"
    assert runner.DEFAULT_FORMAL_RUN_ID == (
        "fsd_v1_2_0_official_balanced250_v1_full1775_20260726"
    )
    assert runner.DEFAULT_SEED == 20260726
    assert runner.FROZEN_RUNTIME_VERSIONS["scikit-learn"] == "1.8.0"
    assert runner.SCORE_SPEC.as_dict() == {
        "key": "ai_score",
        "direction": "higher_means_fake",
        "fixed_threshold": 2.0,
        "threshold_operator": ">",
    }
    assert runner.ADAPTER_SOURCE_PATHS == (
        "eval/__init__.py",
        "eval/opensource/__init__.py",
        "eval/opensource/run_fsd_balanced.py",
        "eval/opensource/analyze_fsd_balanced.py",
        "eval/opensource/run_fsd.py",
        "eval/opensource/analyze_fsd_run.py",
        "eval/opensource/canonical_release.py",
        "eval/opensource/balanced_run_contract.py",
        "eval/opensource/balanced250_metrics.py",
        "eval/opensource/common.py",
    )
    evidence = runner.adapter_source_contract(REPO_ROOT)
    assert set(evidence) == set(runner.ADAPTER_SOURCE_PATHS)
    for relative, record in evidence.items():
        path = REPO_ROOT / relative
        assert record == {
            "path": relative,
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }


def test_formal_selection_is_exact_whole_image_t1(balanced_release):
    spec, selected = runner.select_mode_inputs(
        balanced_release,
        mode="formal",
        per_condition_limit=None,
        sample_id=None,
    )
    assert spec.capability is Capability.WHOLE_IMAGE_T1
    assert len(selected) == 1775
    assert [row["sample_id"] for row in selected] == [
        row["sample_id"] for row in balanced_release.inputs
    ]
    assert Counter(row["condition"] for row in selected) == Counter(
        runner.FORMAL_COUNTS
    )
    assert all("pair_rank" not in row for row in selected)


def test_smoke_selection_is_panel_priority_exact_5x7(balanced_release):
    spec, selected = runner.select_mode_inputs(
        balanced_release,
        mode="smoke",
        per_condition_limit=5,
        sample_id=None,
    )
    assert spec.capability is Capability.WHOLE_IMAGE_T1
    assert len(selected) == 35
    assert Counter(row["condition"] for row in selected) == {
        condition: 5 for condition in BALANCED_CONDITIONS
    }
    selected_ids = {row["sample_id"] for row in selected}
    expected_ids = set()
    counts = Counter()
    for row in balanced_release.panel:
        condition = row["condition"]
        if counts[condition] < 5:
            expected_ids.add(row["sample_id"])
            counts[condition] += 1
    assert selected_ids == expected_ids
    assert [row["rank"] for row in selected] == sorted(
        row["rank"] for row in selected
    )


def test_selection_modes_and_run_id_are_strict(balanced_release):
    sample_id = balanced_release.inputs[0]["sample_id"]
    _spec, selected = runner.select_mode_inputs(
        balanced_release,
        mode="single",
        per_condition_limit=None,
        sample_id=sample_id,
    )
    assert [row["sample_id"] for row in selected] == [sample_id]
    invalid = (
        ("formal", 5, None),
        ("formal", None, sample_id),
        ("smoke", 5, sample_id),
        ("single", 1, sample_id),
        ("single", None, None),
        ("other", None, None),
    )
    for mode, limit, selected_id in invalid:
        with pytest.raises(ValueError):
            runner.select_mode_inputs(
                balanced_release,
                mode=mode,
                per_condition_limit=limit,
                sample_id=selected_id,
            )
    for value in ("", ".", "..", "../escape", "a/b", "a b", "x" * 161):
        with pytest.raises(ValueError):
            runner._valid_run_id(value)


def test_cpu_runtime_is_exact_and_invalid_device_seed_fail_closed():
    device, runtime = runner.configure_runtime(
        "cpu",
        seed=runner.DEFAULT_SEED,
    )
    assert str(device) == "cpu"
    assert runtime["packages"]["torch"]["version"] == "2.10.0+cu128"
    assert runtime["deterministic_algorithms_enabled"] is True
    assert runtime["cudnn"]["benchmark"] is False
    assert runner.validate_runtime_contract(runtime) is runtime
    broken = copy.deepcopy(runtime)
    broken["autocast"] = True
    with pytest.raises(ValueError, match="numerical contract"):
        runner.validate_runtime_contract(broken)
    with pytest.raises(ValueError, match="seed"):
        runner.configure_runtime("cpu", seed=1)
    for device_text in ("cuda", "cuda:-1", "mps", ""):
        with pytest.raises(ValueError):
            runner.configure_runtime(device_text)


def test_visibility_local_crop_and_real_fullframe_na(
    balanced_release,
):
    local = next(
        row
        for row in balanced_release.inputs
        if row["sample_id"] == runner.CPU_GOLDEN_SAMPLE_ID
    )
    diagnostic = runner._visibility_diagnostic(local, repo_root=REPO_ROOT)
    assert diagnostic["edit_visibility"] == "none"
    assert diagnostic["edit_visible_gt_fraction"] == 0.0
    assert (
        diagnostic["edit_visibility_evidence"]["gt"][
            "visible_positive_pixel_centers"
        ]
        == 0
    )
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


def _exact_golden_record() -> dict:
    return {
        "sample_id": runner.CPU_GOLDEN_SAMPLE_ID,
        "input_path": runner.CPU_GOLDEN_INPUT_PATH,
        "image_sha256": runner.CPU_GOLDEN_IMAGE_SHA256,
        "input_width": 1800,
        "input_height": 1350,
        "preprocess": runner.legacy.compute_preprocess_geometry(1800, 1350),
        "descriptor_file_sha256": (
            runner.CPU_GOLDEN_DESCRIPTOR_FILE_SHA256
        ),
        "descriptor_file_bytes": 7808,
        "descriptor_array_sha256": (
            runner.CPU_GOLDEN_DESCRIPTOR_ARRAY_SHA256
        ),
        "descriptor_shape": [runner.legacy.FSD_DIMENSION],
        "descriptor_dtype": "float64",
        "descriptor_nbytes": 7680,
        "raw_likelihood": runner.CPU_GOLDEN_RAW_LIKELIHOOD,
        "released_z_score": runner.CPU_GOLDEN_RELEASED_Z_SCORE,
        "ai_score": runner.CPU_GOLDEN_AI_SCORE,
        "classification_decision": False,
        "released_is_fake": False,
        "full_image_forward": True,
        "compute_fsd_calls": 1,
        "repeat_descriptor_file_sha256": (
            runner.CPU_GOLDEN_DESCRIPTOR_FILE_SHA256
        ),
        "repeat_descriptor_file_bytes": 7808,
        "repeat_descriptor_array_sha256": (
            runner.CPU_GOLDEN_DESCRIPTOR_ARRAY_SHA256
        ),
        "repeat_raw_likelihood": runner.CPU_GOLDEN_RAW_LIKELIHOOD,
        "repeat_released_z_score": runner.CPU_GOLDEN_RELEASED_Z_SCORE,
        "repeat_ai_score": runner.CPU_GOLDEN_AI_SCORE,
        "repeat_classification_decision": False,
        "repeat_released_is_fake": False,
        "repeat_full_image_forward": True,
        "repeat_compute_fsd_calls": 1,
        "repeat_byte_exact": True,
    }


def test_cpu_preflight_requires_exact_two_forward_evidence():
    _device, runtime = runner.configure_runtime("cpu")
    source = {"commit": runner.legacy.MODEL_SOURCE_COMMIT}
    weights = {
        "bundle_sha256": runner.EXPECTED_WEIGHTS_BUNDLE_SHA256,
        "release_tag": runner.legacy.RELEASE_TAG,
    }
    report = {
        "schema_version": runner.CPU_PREFLIGHT_SCHEMA,
        "status": "passed",
        "source": source,
        "weights": weights,
        "runtime": runtime,
        "golden": _exact_golden_record(),
        "cuda_used": False,
        "cuda_tensor_operations": False,
        "dataset_manifest_loaded": False,
    }
    runner._validate_preflight_report(
        report,
        source=source,
        weights=weights,
    )
    mutations = (
        ("report", "cuda_tensor_operations", True),
        ("report", "dataset_manifest_loaded", True),
        ("golden", "repeat_descriptor_file_sha256", "0" * 64),
        ("golden", "repeat_descriptor_array_sha256", "1" * 64),
        ("golden", "repeat_raw_likelihood", 0.0),
        ("golden", "repeat_compute_fsd_calls", 0),
        ("golden", "repeat_full_image_forward", False),
        ("golden", "repeat_byte_exact", False),
    )
    for location, field, replacement in mutations:
        broken = copy.deepcopy(report)
        target = broken if location == "report" else broken["golden"]
        target[field] = replacement
        with pytest.raises(ValueError, match="preflight|golden"):
            runner._validate_preflight_report(
                broken,
                source=source,
                weights=weights,
            )
    broken = copy.deepcopy(report)
    del broken["golden"]["repeat_ai_score"]
    with pytest.raises(ValueError, match="golden key set"):
        runner._validate_preflight_report(
            broken,
            source=source,
            weights=weights,
        )


def test_run_cpu_preflight_executes_two_full_image_forwards(
    monkeypatch,
    tmp_path: Path,
):
    image_path = tmp_path / runner.CPU_GOLDEN_INPUT_PATH
    image_path.parent.mkdir(parents=True)
    image_path.write_bytes(b"golden")
    runtime = {"device": "cpu"}
    monkeypatch.setattr(
        runner,
        "configure_runtime",
        lambda *_args, **_kwargs: ("cpu", runtime),
    )
    source = {"commit": runner.legacy.MODEL_SOURCE_COMMIT}
    weights = {
        "bundle_sha256": runner.EXPECTED_WEIGHTS_BUNDLE_SHA256,
        "release_tag": runner.legacy.RELEASE_TAG,
    }
    monkeypatch.setattr(
        runner.legacy,
        "load_detector",
        lambda **_kwargs: (
            object(),
            "cpu",
            {"source": source, "weights": weights},
        ),
    )
    calls = []

    def fake_infer(*_args):
        calls.append("full")
        return (
            _processed(runner.CPU_GOLDEN_AI_SCORE),
            np.zeros(runner.legacy.FSD_DIMENSION, dtype=np.float64),
            None,
            1.0,
        )

    monkeypatch.setattr(runner.legacy, "infer_one", fake_infer)
    monkeypatch.setattr(
        runner,
        "_validate_golden_forward",
        lambda *_args, **_kwargs: (b"same", "a" * 64),
    )
    monkeypatch.setattr(
        runner,
        "sha256_file",
        lambda path: (
            runner.CPU_GOLDEN_IMAGE_SHA256
            if path == image_path
            else hashlib.sha256(path.read_bytes()).hexdigest()
        ),
    )
    report = runner.run_cpu_preflight(
        repo_root=tmp_path,
        source_root=Path("/source"),
        weights_dir=Path("/weights"),
    )
    assert calls == ["full", "full"]
    assert report["cuda_used"] is False
    assert report["cuda_tensor_operations"] is False
    assert report["golden"]["repeat_full_image_forward"] is True
    assert report["golden"]["repeat_compute_fsd_calls"] == 1


@pytest.mark.parametrize(
    "payload",
    [
        {"pair_rank": 0},
        {"valid_for_t2": True},
        {"native_dense_output": True},
        {"t2_metrics": {}},
        {"pixel_ap": 0.5},
        {"localization": {}},
        {"dense_output_path": "dense.npy"},
        {"heatmap": []},
        {"mask": []},
        {"joint_score": 0.1},
    ],
)
def test_claim_gate_rejects_pair_t2_joint_and_dense(payload):
    with pytest.raises(ValueError, match="unsupported"):
        runner._reject_unsupported_claims(payload)
    runner._reject_unsupported_claims(
        {
            "valid_for_t2": False,
            "native_dense_output": False,
            "localization_output": None,
            "pixel_center_mapping": "official geometry diagnostic",
        }
    )


def test_result_keyset_score_aliases_and_frozen_preprocess_are_exact(
    tmp_path: Path,
):
    result, input_row, _descriptor_dir = _ok_result(tmp_path)
    runner._validate_runner_attempt(
        result,
        input_row=input_row,
        repo_root=tmp_path,
        run_id="run-a",
        run_manifest_fingerprint="a" * 64,
    )
    assert "pair_rank" not in result
    assert result["task_scope"]["valid_for_t2"] is False
    assert "pixel_center_mapping" in result["preprocess"]
    assert result["descriptor"]["shape"] == [960]
    assert result["descriptor"]["dtype"] == "float64"

    broken = copy.deepcopy(result)
    broken["weights_bundle_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="weights bundle"):
        runner._validate_runner_attempt(
            broken,
            input_row=input_row,
            repo_root=tmp_path,
            run_id="run-a",
            run_manifest_fingerprint="a" * 64,
        )
    broken = copy.deepcopy(result)
    broken["extra"] = True
    with pytest.raises(ValueError, match="key set"):
        runner._validate_runner_attempt(
            broken,
            input_row=input_row,
            repo_root=tmp_path,
            run_id="run-a",
            run_manifest_fingerprint="a" * 64,
        )


def test_strict_threshold_equality_is_not_fake(tmp_path: Path):
    result, input_row, _descriptor_dir = _ok_result(
        tmp_path,
        ai_score=2.0,
    )
    assert result["classification_decision"] is False
    assert result["released_is_fake"] is False
    runner._validate_runner_attempt(
        result,
        input_row=input_row,
        repo_root=tmp_path,
        run_id="run-a",
        run_manifest_fingerprint="a" * 64,
    )


def test_descriptor_inventory_rejects_tamper_extra_nan_and_symlink(
    tmp_path: Path,
):
    result, input_row, descriptor_dir = _ok_result(tmp_path / "tamper")
    descriptor_path = (
        tmp_path
        / "tamper"
        / result["descriptor"]["relative_path"]
    )
    descriptor_path.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="metadata/hash"):
        runner._validate_runner_attempt(
            result,
            input_row=input_row,
            repo_root=tmp_path / "tamper",
            run_id="run-a",
            run_manifest_fingerprint="a" * 64,
        )


def _run_args(root: Path, *, resume: bool = False) -> argparse.Namespace:
    return argparse.Namespace(
        repo_root=root,
        dataset_manifest=runner.DEFAULT_DATASET_MANIFEST,
        source_root=Path("/source"),
        weights_dir=Path("/weights"),
        results_dir=Path("results/opensource/fsd"),
        artifacts_dir=runner.DEFAULT_ARTIFACTS_DIR,
        run_id="single-run",
        mode="single",
        per_condition_limit=None,
        sample_id="0" * 24,
        device="cpu",
        seed=runner.DEFAULT_SEED,
        resume=resume,
        fail_fast=False,
    )


def _patch_single_run(
    monkeypatch: pytest.MonkeyPatch,
    root: Path,
    *,
    inference_error: BaseException | None = None,
):
    input_row = _real_input(root)
    contract_dict = {
        "schema_version": "test_dataset_contract_v2",
        "selection": {"selected_images": 1},
    }

    class Contract:
        def as_dict(self):
            return contract_dict

    release = SimpleNamespace(
        manifest_sha256="d" * 64,
        repo_root=root,
    )
    monkeypatch.setattr(
        runner,
        "load_canonical_release",
        lambda *_args, **_kwargs: release,
    )
    monkeypatch.setattr(
        runner,
        "select_mode_inputs",
        lambda *_args, **_kwargs: (object(), [input_row]),
    )
    monkeypatch.setattr(
        runner,
        "build_run_dataset_contract",
        lambda *_args, **_kwargs: Contract(),
    )
    source = {"commit": runner.legacy.MODEL_SOURCE_COMMIT}
    weights = {
        "bundle_sha256": runner.EXPECTED_WEIGHTS_BUNDLE_SHA256,
        "release_tag": runner.legacy.RELEASE_TAG,
    }
    preflight = {
        "schema_version": runner.CPU_PREFLIGHT_SCHEMA,
        "status": "passed",
        "source": source,
        "weights": weights,
    }
    monkeypatch.setattr(
        runner,
        "run_cpu_preflight",
        lambda **_kwargs: preflight,
    )
    monkeypatch.setattr(
        runner.legacy,
        "_verify_static_contract",
        lambda **_kwargs: (source, weights),
    )
    monkeypatch.setattr(
        runner,
        "_validate_preflight_report",
        lambda *_args, **_kwargs: None,
    )
    device = SimpleNamespace(type="cpu")
    runtime = {"device": "cpu", "frozen": True}
    monkeypatch.setattr(
        runner,
        "configure_runtime",
        lambda *_args, **_kwargs: (device, runtime),
    )
    monkeypatch.setattr(
        runner,
        "validate_runtime_contract",
        lambda value, **_kwargs: value,
    )
    monkeypatch.setattr(
        runner,
        "adapter_source_contract",
        lambda _root: {
            "adapter.py": {
                "path": "adapter.py",
                "bytes": 1,
                "sha256": "e" * 64,
            }
        },
    )
    load_calls = []

    def fake_load(**_kwargs):
        load_calls.append("load")
        return (
            object(),
            device,
            {"source": source, "weights": weights},
        )

    monkeypatch.setattr(runner.legacy, "load_detector", fake_load)

    def fake_infer(*_args):
        if inference_error is not None:
            raise inference_error
        return (
            _processed(),
            np.arange(runner.legacy.FSD_DIMENSION, dtype=np.float64),
            None,
            2.0,
        )

    monkeypatch.setattr(runner.legacy, "infer_one", fake_infer)
    return input_row, contract_dict, load_calls


def test_mocked_single_run_writes_exact_v2_runtime_artifacts(
    monkeypatch,
    tmp_path: Path,
):
    input_row, contract, load_calls = _patch_single_run(
        monkeypatch,
        tmp_path,
    )
    assert runner.run(_run_args(tmp_path)) == 0
    assert load_calls == ["load"]
    run_dir = tmp_path / "results/opensource/fsd/single-run"
    manifest = runner._load_json_strict(
        run_dir / "manifest.json",
        "manifest",
    )
    assert set(manifest) == {
        "schema_version",
        "run_id",
        "status",
        "started_at",
        "completed_at",
        "fingerprint",
        "immutable",
        "dataset",
        "outputs",
        "execution",
    }
    assert manifest["status"] == "complete"
    assert set(manifest["immutable"]) == runner.IMMUTABLE_CONFIG_KEYS
    assert manifest["immutable"]["dataset_contract"] == contract
    assert manifest["immutable"]["task_scope"] == runner.TASK_SCOPE
    assert manifest["immutable"]["preprocess"] == runner.PREPROCESS_CONTRACT
    assert manifest["immutable"]["artifact_contract"] == (
        runner.ARTIFACT_CONTRACT
    )
    assert manifest["outputs"]["descriptor_files"] == 1
    rows = read_jsonl(run_dir / "results.jsonl")
    assert len(rows) == 1
    row = rows[0]
    assert row["sample_id"] == input_row["sample_id"]
    assert row["schema_version"] == runner.RESULT_SCHEMA_VERSION
    assert "pair_rank" not in row
    assert row["descriptor"]["file_bytes"] == 7808
    assert row["raw_descriptor_array_sha256"] == (
        row["descriptor"]["array_sha256"]
    )
    summary = runner._load_json_strict(
        run_dir / "summary.json",
        "summary",
    )
    assert summary["summary_kind"] == "runtime_coverage_only"
    assert summary["scientific_metrics"] is None
    assert summary["scientific_metrics_owner"] == (
        "analyze_fsd_balanced.py"
    )
    assert summary["coverage"]["is_complete"] is True


def test_resume_validates_history_and_skips_complete_without_loading(
    monkeypatch,
    tmp_path: Path,
):
    _input, _contract, load_calls = _patch_single_run(monkeypatch, tmp_path)
    args = _run_args(tmp_path)
    assert runner.run(args) == 0
    assert load_calls == ["load"]
    results_path = (
        tmp_path / "results/opensource/fsd/single-run/results.jsonl"
    )
    before = results_path.read_bytes()
    args.resume = True
    load_calls.clear()
    assert runner.run(args) == 0
    assert load_calls == []
    assert results_path.read_bytes() == before
    manifest = runner._load_json_strict(
        results_path.parent / "manifest.json",
        "manifest",
    )
    assert manifest["execution"]["resume_skips"] == 1
    assert manifest["execution"]["new_successes"] == 0
    assert manifest["execution"]["physical_result_rows"] == 1


def test_inference_error_appends_invalid_attempt_and_returns_incomplete(
    monkeypatch,
    tmp_path: Path,
):
    _patch_single_run(
        monkeypatch,
        tmp_path,
        inference_error=RuntimeError("inference failed"),
    )
    assert runner.run(_run_args(tmp_path)) == 2
    run_dir = tmp_path / "results/opensource/fsd/single-run"
    rows = read_jsonl(run_dir / "results.jsonl")
    assert len(rows) == 1
    assert rows[0]["status"] == "error"
    assert rows[0]["valid_for_metrics"] is False
    assert "descriptor" not in rows[0]
    assert not any(
        (
            tmp_path
            / runner.DEFAULT_ARTIFACTS_DIR
            / "single-run"
            / "raw_descriptors"
        ).iterdir()
    )
    manifest = runner._load_json_strict(
        run_dir / "manifest.json",
        "manifest",
    )
    assert manifest["status"] == "incomplete"
    assert manifest["execution"]["new_errors"] == 1
    assert manifest["outputs"]["descriptor_files"] == 0


def test_preflight_mode_never_loads_dataset_or_requested_accelerator(
    monkeypatch,
    tmp_path: Path,
    capsys,
):
    source = {"commit": runner.legacy.MODEL_SOURCE_COMMIT}
    weights = {
        "bundle_sha256": runner.EXPECTED_WEIGHTS_BUNDLE_SHA256,
        "release_tag": runner.legacy.RELEASE_TAG,
    }
    report = {
        "schema_version": runner.CPU_PREFLIGHT_SCHEMA,
        "status": "passed",
        "source": source,
        "weights": weights,
    }
    monkeypatch.setattr(
        runner,
        "run_cpu_preflight",
        lambda **_kwargs: report,
    )
    monkeypatch.setattr(
        runner.legacy,
        "_verify_static_contract",
        lambda **_kwargs: (source, weights),
    )
    monkeypatch.setattr(
        runner,
        "_validate_preflight_report",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        runner,
        "load_canonical_release",
        lambda *_args, **_kwargs: pytest.fail("dataset was loaded"),
    )
    args = _run_args(tmp_path)
    args.mode = "preflight"
    args.run_id = None
    args.sample_id = None
    args.device = "cpu"
    assert runner.run(args) == 0
    assert '"status": "passed"' in capsys.readouterr().out

    result, input_row, descriptor_dir = _ok_result(tmp_path / "extra")
    (descriptor_dir / "extra.npy").write_bytes(b"extra")
    with pytest.raises(ValueError, match="inventory mismatch"):
        runner._validate_descriptor_inventory(
            descriptor_dir=descriptor_dir,
            latest_by_sample_id={input_row["sample_id"]: result},
            repo_root=tmp_path / "extra",
            run_id="run-a",
        )

    with pytest.raises(ValueError, match="descriptor"):
        _ok_result(
            tmp_path / "nan",
            descriptor=np.full(
                runner.legacy.FSD_DIMENSION,
                np.nan,
                dtype=np.float64,
            ),
        )

    result, input_row, _descriptor_dir = _ok_result(tmp_path / "symlink")
    path = tmp_path / "symlink" / result["descriptor"]["relative_path"]
    target = tmp_path / "symlink" / "target.npy"
    path.replace(target)
    path.symlink_to(target)
    with pytest.raises(ValueError, match="symlink"):
        runner._validate_runner_attempt(
            result,
            input_row=input_row,
            repo_root=tmp_path / "symlink",
            run_id="run-a",
            run_manifest_fingerprint="a" * 64,
        )
