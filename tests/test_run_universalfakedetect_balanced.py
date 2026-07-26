from __future__ import annotations

import argparse
import copy
import hashlib
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from eval.opensource import run_universalfakedetect_balanced as runner
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
    Image.new("RGB", (32, 32), color=(10, 20, 30)).save(
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
        "width": 32,
        "height": 32,
        "panel": True,
    }


def _processed(
    raw_logit: float = 1.0,
    *,
    probability: float | None = None,
) -> dict:
    if probability is None:
        probability = runner.legacy._float32_sigmoid(raw_logit)
    decision = probability > runner.legacy.CLASSIFICATION_THRESHOLD
    return {
        "raw_logit": float(raw_logit),
        "probability": probability,
        "ai_score": probability,
        "classification_decision": decision,
        "manual_replay": {
            "raw_logit": float(raw_logit),
            "probability": probability,
            "ai_score": probability,
            "classification_decision": decision,
            "official_logit_exact_match": True,
            "official_probability_exact_match": True,
            "model_forward_calls": 1,
            "fc_hook_calls": 1,
        },
    }


def _ok_result(
    root: Path,
    *,
    sample_id: str = "0" * 24,
    raw_logit: float = 1.0,
    probability: float | None = None,
    feature: np.ndarray | None = None,
) -> tuple[dict, dict, Path]:
    input_row = _real_input(root, sample_id=sample_id)
    feature_dir = root / runner.DEFAULT_ARTIFACTS_DIR / "run-a" / "clip_features"
    feature_dir.mkdir(parents=True, exist_ok=True)
    value = (
        np.linspace(
            -1.0,
            1.0,
            runner.legacy.FEATURE_DIMENSION,
            dtype=np.float32,
        )
        if feature is None
        else feature
    )
    image_path = root / input_row["canonical_path"]
    _tensor, preprocess = runner.legacy.preprocess_image(
        image_path,
        runner.FROZEN_PROFILE,
    )
    result = runner._build_ok_result(
        input_row=input_row,
        repo_root=root,
        run_id="run-a",
        fingerprint="a" * 64,
        asset_bundle_sha256=runner.EXPECTED_ASSET_BUNDLE_SHA256,
        feature_dir=feature_dir,
        processed=_processed(raw_logit, probability=probability),
        feature=value,
        preprocess=preprocess,
        preprocess_latency_ms=1.0,
        latency_ms=2.0,
        peak_cuda_memory_bytes=0,
    )
    return result, input_row, feature_dir


def test_frozen_contract_and_adapter_inventory():
    assert runner.RUN_MANIFEST_SCHEMA == (
        "universalfakedetect_balanced_run_manifest_v2"
    )
    assert runner.RUN_CONFIG_SCHEMA == ("universalfakedetect_balanced_run_config_v2")
    assert runner.RUNTIME_SUMMARY_SCHEMA == (
        "universalfakedetect_balanced_runtime_summary_v2"
    )
    assert runner.CPU_PREFLIGHT_SCHEMA == (
        "universalfakedetect_balanced_cpu_preflight_v1"
    )
    assert runner.FROZEN_PROFILE == ("current_head_native_center_crop224")
    assert runner.DEFAULT_SEED == 20260726
    assert runner.FROZEN_PYTHON_EXECUTABLE == Path(
        "/root/.cache/claimforge/venvs/" "ufd-balanced-torch2.8.0/bin/python"
    )
    assert runner.SCORE_SPEC.as_dict() == {
        "key": "ai_score",
        "direction": "higher_means_fake",
        "fixed_threshold": 0.5,
        "threshold_operator": ">",
    }
    assert runner.ARTIFACT_CONTRACT["feature"] == {
        "format": "NumPy .npy, allow_pickle=False",
        "shape": [768],
        "dtype": "float32",
        "nbytes": 3072,
        "finite": True,
        "semantics": ("official_CLIP_encode_image_output_before_linear_head"),
        "allow_pickle": False,
        "exact_fc_and_sigmoid_replay": True,
    }
    assert runner.SOURCE_CHECKPOINT_DRIFT["checkpoint_era_profile_executed"] is False
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
    expected_ids = set()
    counts = Counter()
    for row in balanced_release.panel:
        condition = row["condition"]
        if counts[condition] < 5:
            expected_ids.add(row["sample_id"])
            counts[condition] += 1
    assert {row["sample_id"] for row in selected} == expected_ids
    assert [row["rank"] for row in selected] == sorted(row["rank"] for row in selected)


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
    assert runtime["packages"]["torch"]["version"] == "2.8.0+cu128"
    assert runtime["packages"]["torchvision"]["version"] == ("0.23.0+cu128")
    assert runtime["packages"]["setuptools"] == "75.8.0"
    assert runtime["packages"]["tqdm"] == "4.67.1"
    assert runtime["venv"]["include_system_site_packages"] is False
    assert runtime["venv"]["pyvenv_cfg_sha256"] == (runner.FROZEN_PYVENV_CONFIG_SHA256)
    assert runtime["deterministic_algorithms_enabled"] is True
    assert runtime["cudnn"]["benchmark"] is False
    assert runner.validate_runtime_contract(runtime) is runtime
    broken = copy.deepcopy(runtime)
    broken["autocast"] = True
    with pytest.raises(ValueError, match="numerical contract"):
        runner.validate_runtime_contract(broken)
    broken = copy.deepcopy(runtime)
    broken["venv"]["include_system_site_packages"] = True
    with pytest.raises(ValueError, match="clean-environment"):
        runner.validate_runtime_contract(broken)
    with pytest.raises(ValueError, match="seed"):
        runner.configure_runtime("cpu", seed=1)
    for device_text in ("cuda", "cuda:-1", "mps", ""):
        with pytest.raises(ValueError):
            runner.configure_runtime(device_text)


def test_visibility_local_crop_and_real_fullframe_na(balanced_release):
    local = next(
        row
        for row in balanced_release.inputs
        if row["sample_id"] == runner.CPU_GOLDEN_SAMPLE_ID
    )
    diagnostic = runner._visibility_diagnostic(local, repo_root=REPO_ROOT)
    assert diagnostic["edit_visibility"] == "none"
    assert diagnostic["edit_visible_gt_fraction"] == 0.0
    assert (
        diagnostic["edit_visibility_evidence"]["gt"]["visible_positive_pixel_centers"]
        == 0
    )
    real = next(row for row in balanced_release.inputs if row["condition"] == "real")
    fullframe = next(
        row for row in balanced_release.inputs if row["condition"] == "fullframe_mouse"
    )
    for row in (real, fullframe):
        value = runner._visibility_diagnostic(row, repo_root=REPO_ROOT)
        assert value["edit_visibility"] == "not_applicable"
        assert value["edit_visible_gt_fraction"] is None


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


def test_result_keyset_scores_feature_and_preprocess_are_exact(
    tmp_path: Path,
):
    result, input_row, _feature_dir = _ok_result(tmp_path)
    runner._validate_runner_attempt(
        result,
        input_row=input_row,
        repo_root=tmp_path,
        run_id="run-a",
        run_manifest_fingerprint="a" * 64,
    )
    assert result["schema_version"] == runner.RESULT_SCHEMA_VERSION
    assert "pair_rank" not in result
    assert result["task_scope"]["valid_for_t2"] is False
    assert result["preprocess_profile"] == runner.FROZEN_PROFILE
    assert result["clip_feature"]["shape"] == [768]
    assert result["clip_feature"]["dtype"] == "float32"
    assert result["clip_feature_nbytes"] == 3072
    assert result["clip_feature_array_sha256"] == (
        result["clip_feature"]["array_sha256"]
    )

    broken = copy.deepcopy(result)
    broken["asset_bundle_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="asset bundle"):
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
    result, input_row, _feature_dir = _ok_result(
        tmp_path,
        raw_logit=0.0,
    )
    assert result["probability"] == 0.5
    assert result["classification_decision"] is False
    runner._validate_runner_attempt(
        result,
        input_row=input_row,
        repo_root=tmp_path,
        run_id="run-a",
        run_manifest_fingerprint="a" * 64,
    )


def test_float32_sigmoid_tiny_positive_rounds_to_threshold(
    tmp_path: Path,
):
    result, input_row, _feature_dir = _ok_result(
        tmp_path,
        raw_logit=1e-8,
    )
    assert result["raw_logit"] > 0.0
    assert result["probability"] == 0.5
    assert result["classification_decision"] is False
    runner._validate_runner_attempt(
        result,
        input_row=input_row,
        repo_root=tmp_path,
        run_id="run-a",
        run_manifest_fingerprint="a" * 64,
    )


def test_cuda_sigmoid_value_is_not_recomputed_on_cpu(tmp_path: Path):
    raw_logit = 2.6707892417907715
    cuda_probability = 0.9352807402610779
    cpu_probability = runner.legacy._float32_sigmoid(raw_logit)
    assert cuda_probability != cpu_probability
    assert abs(cuda_probability - cpu_probability) > (
        runner.legacy.RESUME_CPU_SIGMOID_ABS_TOLERANCE
    )
    result, input_row, _feature_dir = _ok_result(
        tmp_path,
        raw_logit=raw_logit,
        probability=cuda_probability,
    )
    runner._validate_runner_attempt(
        result,
        input_row=input_row,
        repo_root=tmp_path,
        run_id="run-a",
        run_manifest_fingerprint="a" * 64,
    )


@pytest.mark.parametrize("probability", [-0.1, 1.1])
def test_probability_must_be_in_unit_interval(
    tmp_path: Path,
    probability: float,
):
    with pytest.raises(ValueError, match=r"outside \[0,1\]"):
        _ok_result(
            tmp_path,
            raw_logit=0.0,
            probability=probability,
        )


def test_feature_inventory_rejects_tamper_extra_nan_and_symlink(
    tmp_path: Path,
):
    result, input_row, feature_dir = _ok_result(tmp_path / "tamper")
    feature_path = tmp_path / "tamper" / result["clip_feature"]["relative_path"]
    feature_path.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="metadata/hash"):
        runner._validate_runner_attempt(
            result,
            input_row=input_row,
            repo_root=tmp_path / "tamper",
            run_id="run-a",
            run_manifest_fingerprint="a" * 64,
        )

    result, input_row, feature_dir = _ok_result(tmp_path / "extra")
    (feature_dir / "extra.npy").write_bytes(b"extra")
    with pytest.raises(ValueError, match="inventory mismatch"):
        runner._validate_feature_inventory(
            feature_dir=feature_dir,
            latest_by_sample_id={input_row["sample_id"]: result},
            repo_root=tmp_path / "extra",
            run_id="run-a",
        )

    with pytest.raises(ValueError, match="feature"):
        _ok_result(
            tmp_path / "nan",
            feature=np.full(
                runner.legacy.FEATURE_DIMENSION,
                np.nan,
                dtype=np.float32,
            ),
        )

    result, input_row, _feature_dir = _ok_result(tmp_path / "symlink")
    path = tmp_path / "symlink" / result["clip_feature"]["relative_path"]
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


def test_head_replay_requires_and_uses_explicit_torch_device(
    monkeypatch,
    tmp_path: Path,
):
    import torch

    result, _input_row, _feature_dir = _ok_result(tmp_path)
    weight = torch.zeros(
        (1, runner.legacy.FEATURE_DIMENSION),
        dtype=torch.float32,
    )
    weight[0, 0] = 1.0
    state = {
        "weight": weight,
        "bias": torch.tensor([2.0], dtype=torch.float32),
    }
    latest = {str(result["sample_id"]): result}
    replay_devices = []
    original_linear = torch.nn.functional.linear

    def capture_linear(feature, replay_weight, replay_bias):
        replay_devices.append(
            (
                feature.device,
                replay_weight.device,
                replay_bias.device,
            )
        )
        return original_linear(feature, replay_weight, replay_bias)

    monkeypatch.setattr(torch.nn.functional, "linear", capture_linear)
    with pytest.raises(ValueError, match="explicit configured torch device"):
        runner._validate_latest_feature_head_replay(
            latest_by_sample_id=latest,
            repo_root=tmp_path,
            run_id="run-a",
            head_state=state,
            device="cpu",
        )
    assert (
        runner._validate_latest_feature_head_replay(
            latest_by_sample_id=latest,
            repo_root=tmp_path,
            run_id="run-a",
            head_state=state,
            device=torch.device("cpu"),
        )
        == 1
    )
    assert replay_devices == [
        (
            torch.device("cpu"),
            torch.device("cpu"),
            torch.device("cpu"),
        )
    ]


def test_output_directories_reject_collisions_and_nonempty_roots(
    tmp_path: Path,
):
    same = tmp_path / "same"
    with pytest.raises(ValueError, match="disjoint"):
        runner._prepare_output_directories(
            repo_root=tmp_path,
            run_dir=same,
            feature_root=same,
            resume=False,
        )
    run_dir = tmp_path / "results" / "run"
    run_dir.mkdir(parents=True)
    (run_dir / "foreign").write_text("collision", encoding="utf-8")
    with pytest.raises(FileExistsError, match="run directory"):
        runner._prepare_output_directories(
            repo_root=tmp_path,
            run_dir=run_dir,
            feature_root=tmp_path / "artifacts" / "run",
            resume=False,
        )


def test_run_cpu_preflight_executes_two_full_image_forwards(
    monkeypatch,
    tmp_path: Path,
):
    import torch

    image_path = tmp_path / runner.CPU_GOLDEN_INPUT_PATH
    image_path.parent.mkdir(parents=True)
    image_path.write_bytes(b"golden")
    runtime = {"device": "cpu"}
    monkeypatch.setattr(
        runner,
        "configure_runtime",
        lambda *_args, **_kwargs: ("cpu", runtime),
    )
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    source = {"commit": runner.legacy.MODEL_SOURCE_COMMIT}
    assets = {
        "bundle_sha256": runner.EXPECTED_ASSET_BUNDLE_SHA256,
    }
    load_audit = {"source": source, "assets": assets}
    monkeypatch.setattr(
        runner.legacy,
        "load_model",
        lambda **_kwargs: (object(), "cpu", load_audit),
    )
    preprocess = runner._expected_golden_preprocess()
    monkeypatch.setattr(
        runner.legacy,
        "preprocess_image",
        lambda *_args, **_kwargs: (
            np.zeros((3, 224, 224), dtype=np.float32),
            preprocess,
        ),
    )
    calls = []

    def fake_infer(*_args):
        calls.append("full")
        return (
            {
                "raw_logit": runner.CPU_GOLDEN_RAW_LOGIT,
                "probability": runner.CPU_GOLDEN_PROBABILITY,
                "ai_score": runner.CPU_GOLDEN_PROBABILITY,
                "classification_decision": False,
                "manual_replay": {},
            },
            np.zeros(runner.legacy.FEATURE_DIMENSION, dtype=np.float32),
            0,
            1.0,
        )

    monkeypatch.setattr(runner.legacy, "infer_one", fake_infer)
    monkeypatch.setattr(
        runner,
        "_validate_golden_forward",
        lambda *_args, **_kwargs: (b"x" * 3200, "a" * 64),
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
        head_checkpoint=Path("/head"),
        backbone_checkpoint=Path("/backbone"),
    )
    assert calls == ["full", "full"]
    assert report["cuda_used"] is False
    assert report["cuda_tensor_operations"] is False
    assert report["cuda_initialized_before_cpu_model_load"] is False
    assert report["cuda_initialized_after_cpu_forwards"] is False
    assert report["golden"]["repeat_full_image_forward"] is True
    assert report["golden"]["repeat_model_forward_calls"] == 1
    assert report["golden"]["repeat_fc_hook_calls"] == 1

    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    with pytest.raises(RuntimeError, match="already initialized"):
        runner.run_cpu_preflight(
            repo_root=tmp_path,
            source_root=Path("/source"),
            head_checkpoint=Path("/head"),
            backbone_checkpoint=Path("/backbone"),
        )
    cuda_states = iter((False, True))
    monkeypatch.setattr(
        torch.cuda,
        "is_initialized",
        lambda: next(cuda_states),
    )
    with pytest.raises(RuntimeError, match="initialized CUDA"):
        runner.run_cpu_preflight(
            repo_root=tmp_path,
            source_root=Path("/source"),
            head_checkpoint=Path("/head"),
            backbone_checkpoint=Path("/backbone"),
        )


def _run_args(root: Path, *, resume: bool = False) -> argparse.Namespace:
    return argparse.Namespace(
        repo_root=root,
        dataset_manifest=runner.DEFAULT_DATASET_MANIFEST,
        source_root=Path("/source"),
        head_checkpoint=Path("/head"),
        backbone_checkpoint=Path("/backbone"),
        results_dir=runner.DEFAULT_RESULTS_DIR,
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
    import torch

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
    assets = {
        "bundle_sha256": runner.EXPECTED_ASSET_BUNDLE_SHA256,
    }
    preflight = {
        "schema_version": runner.CPU_PREFLIGHT_SCHEMA,
        "status": "passed",
        "source": source,
        "assets": assets,
    }
    monkeypatch.setattr(
        runner,
        "run_cpu_preflight",
        lambda **_kwargs: preflight,
    )
    monkeypatch.setattr(
        runner.legacy,
        "_verify_asset_contract",
        lambda **_kwargs: (
            source,
            assets,
            {
                "weight": torch.nn.functional.one_hot(
                    torch.tensor([0]),
                    num_classes=runner.legacy.FEATURE_DIMENSION,
                ).to(dtype=torch.float32),
                "bias": torch.tensor([2.0], dtype=torch.float32),
            },
        ),
    )
    monkeypatch.setattr(
        runner,
        "_validate_preflight_report",
        lambda *_args, **_kwargs: None,
    )
    device = torch.device("cpu")
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
            {"source": source, "assets": assets},
        )

    monkeypatch.setattr(runner.legacy, "load_model", fake_load)

    def fake_infer(*_args):
        if inference_error is not None:
            raise inference_error
        return (
            _processed(),
            np.linspace(
                -1.0,
                1.0,
                runner.legacy.FEATURE_DIMENSION,
                dtype=np.float32,
            ),
            0,
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
    run_dir = tmp_path / runner.DEFAULT_RESULTS_DIR / "single-run"
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
    assert manifest["immutable"]["preprocess"] == (runner.PREPROCESS_CONTRACT)
    assert manifest["immutable"]["artifact_contract"] == (runner.ARTIFACT_CONTRACT)
    assert manifest["outputs"]["feature_files"] == 1
    rows = read_jsonl(run_dir / "results.jsonl")
    assert len(rows) == 1
    row = rows[0]
    assert row["sample_id"] == input_row["sample_id"]
    assert row["schema_version"] == runner.RESULT_SCHEMA_VERSION
    assert "pair_rank" not in row
    assert row["clip_feature"]["file_bytes"] == 3200
    assert row["clip_feature_array_sha256"] == (row["clip_feature"]["array_sha256"])
    summary = runner._load_json_strict(
        run_dir / "summary.json",
        "summary",
    )
    assert summary["summary_kind"] == "runtime_coverage_only"
    assert summary["scientific_metrics"] is None
    assert summary["scientific_metrics_owner"] == (
        "analyze_universalfakedetect_balanced.py"
    )
    assert summary["preprocess_profile"] == runner.FROZEN_PROFILE
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
        tmp_path / runner.DEFAULT_RESULTS_DIR / "single-run" / "results.jsonl"
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


def _rewrite_finalized_result_evidence(
    *,
    run_dir: Path,
    rows: list[dict],
) -> bytes:
    results_path = run_dir / "results.jsonl"
    manifest_path = run_dir / "manifest.json"
    runner.atomic_write_jsonl(results_path, rows)
    manifest = runner._load_json_strict(manifest_path, "manifest")
    manifest["outputs"]["results_sha256"] = runner.sha256_file(results_path)
    runner.atomic_write_json(manifest_path, manifest)
    return manifest_path.read_bytes()


def test_resume_rejects_rehashed_feature_that_does_not_replay_head(
    monkeypatch,
    tmp_path: Path,
):
    _patch_single_run(monkeypatch, tmp_path)
    args = _run_args(tmp_path)
    assert runner.run(args) == 0
    run_dir = tmp_path / runner.DEFAULT_RESULTS_DIR / "single-run"
    rows = runner._read_jsonl_strict(
        run_dir / "results.jsonl",
        "results",
    )
    row = rows[0]
    feature_path = tmp_path / row["clip_feature"]["relative_path"]
    tampered = np.zeros(
        runner.legacy.FEATURE_DIMENSION,
        dtype=np.float32,
    )
    runner.legacy._atomic_save_npy(feature_path, tampered)
    feature_record = runner._feature_record(
        feature=tampered,
        feature_path=feature_path,
        repo_root=tmp_path,
    )
    row["clip_feature"] = feature_record
    row["clip_feature_path"] = feature_record["relative_path"]
    row["clip_feature_sha256"] = feature_record["sha256"]
    row["clip_feature_array_sha256"] = feature_record["array_sha256"]
    row["clip_feature_shape"] = feature_record["shape"]
    row["clip_feature_dtype"] = feature_record["dtype"]
    row["clip_feature_nbytes"] = feature_record["nbytes"]
    row["clip_feature_semantics"] = feature_record["semantics"]
    row["artifact_paths"] = {
        "clip_feature_npy": feature_record["relative_path"],
    }
    manifest_before = _rewrite_finalized_result_evidence(
        run_dir=run_dir,
        rows=rows,
    )
    args.resume = True
    with pytest.raises(
        ValueError,
        match="independent head replay logit mismatch",
    ):
        runner.run(args)
    assert (run_dir / "manifest.json").read_bytes() == manifest_before


def test_resume_rejects_alias_consistent_probability_drift(
    monkeypatch,
    tmp_path: Path,
):
    _patch_single_run(monkeypatch, tmp_path)
    args = _run_args(tmp_path)
    assert runner.run(args) == 0
    run_dir = tmp_path / runner.DEFAULT_RESULTS_DIR / "single-run"
    rows = runner._read_jsonl_strict(
        run_dir / "results.jsonl",
        "results",
    )
    row = rows[0]
    drifted_probability = row["probability"] + 5e-8
    row["probability"] = drifted_probability
    row["ai_score"] = drifted_probability
    row["score"] = drifted_probability
    for nested in ("classification", "t1"):
        row[nested]["probability"] = drifted_probability
        row[nested]["ai_score"] = drifted_probability
        row[nested]["score"] = drifted_probability
    row["manual_replay"]["probability"] = drifted_probability
    row["manual_replay"]["ai_score"] = drifted_probability
    manifest_before = _rewrite_finalized_result_evidence(
        run_dir=run_dir,
        rows=rows,
    )
    args.resume = True
    with pytest.raises(
        ValueError,
        match="independent head replay probability mismatch",
    ):
        runner.run(args)
    assert (run_dir / "manifest.json").read_bytes() == manifest_before


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
    run_dir = tmp_path / runner.DEFAULT_RESULTS_DIR / "single-run"
    rows = read_jsonl(run_dir / "results.jsonl")
    assert len(rows) == 1
    assert rows[0]["status"] == "error"
    assert rows[0]["valid_for_metrics"] is False
    assert "clip_feature" not in rows[0]
    assert not any(
        (
            tmp_path / runner.DEFAULT_ARTIFACTS_DIR / "single-run" / "clip_features"
        ).iterdir()
    )
    manifest = runner._load_json_strict(
        run_dir / "manifest.json",
        "manifest",
    )
    assert manifest["status"] == "incomplete"
    assert manifest["execution"]["new_errors"] == 1
    assert manifest["outputs"]["feature_files"] == 0


def test_preflight_mode_never_loads_dataset_or_accelerator(
    monkeypatch,
    tmp_path: Path,
    capsys,
):
    source = {"commit": runner.legacy.MODEL_SOURCE_COMMIT}
    assets = {
        "bundle_sha256": runner.EXPECTED_ASSET_BUNDLE_SHA256,
    }
    report = {
        "schema_version": runner.CPU_PREFLIGHT_SCHEMA,
        "status": "passed",
        "source": source,
        "assets": assets,
    }
    monkeypatch.setattr(
        runner,
        "run_cpu_preflight",
        lambda **_kwargs: report,
    )
    monkeypatch.setattr(
        runner.legacy,
        "_verify_asset_contract",
        lambda **_kwargs: (source, assets, {}),
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
