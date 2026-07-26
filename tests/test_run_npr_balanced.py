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

from eval.opensource import run_npr_balanced as runner
from eval.opensource.canonical_release import (
    BALANCED_CONDITIONS,
    Capability,
    load_canonical_release,
)
from eval.opensource.common import read_jsonl, stable_json


REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def balanced_release():
    return load_canonical_release(
        REPO_ROOT,
        runner.DEFAULT_DATASET_MANIFEST,
        verify_files=True,
    )


@pytest.fixture(scope="module")
def audited_checkpoint_state():
    if (
        not runner.DEFAULT_SOURCE_ROOT.is_dir()
        or not runner.DEFAULT_HF_SOURCE_ROOT.is_dir()
        or not runner.DEFAULT_CHECKPOINT.is_file()
    ):
        pytest.skip("frozen NPR source/checkpoint assets are unavailable")
    _source, _assets, state, _module = runner.legacy.verify_assets(
        source_root=runner.DEFAULT_SOURCE_ROOT,
        hf_source_root=runner.DEFAULT_HF_SOURCE_ROOT,
        checkpoint_path=runner.DEFAULT_CHECKPOINT,
    )
    return state


def _float32_sigmoid(raw_logit: float) -> float:
    import torch

    return float(
        torch.sigmoid(
            torch.tensor(raw_logit, dtype=torch.float32),
        ).item()
    )


def _score_payload(
    raw_logit: float = 1.0,
    *,
    probability: float | None = None,
) -> dict:
    if probability is None:
        probability = _float32_sigmoid(raw_logit)
    decision = probability > runner.legacy.CLASSIFICATION_THRESHOLD
    semantics = "official_float32_sigmoid_probability_higher_is_fake"
    return {
        "raw_logit": float(raw_logit),
        "probability": probability,
        "ai_score": probability,
        "score": probability,
        "score_semantics": semantics,
        "classification_decision": decision,
        "classification_threshold": runner.legacy.CLASSIFICATION_THRESHOLD,
        "classification_threshold_operator": (
            runner.legacy.CLASSIFICATION_THRESHOLD_OPERATOR
        ),
        "classification": {
            "raw_logit": float(raw_logit),
            "probability": probability,
            "ai_score": probability,
            "score": probability,
            "threshold": runner.legacy.CLASSIFICATION_THRESHOLD,
            "threshold_operator": (runner.legacy.CLASSIFICATION_THRESHOLD_OPERATOR),
            "decision": decision,
            "semantics": semantics,
        },
        "t1": {
            "raw_logit": float(raw_logit),
            "probability": probability,
            "ai_score": probability,
            "score": probability,
            "threshold": runner.legacy.CLASSIFICATION_THRESHOLD,
            "threshold_operator": (runner.legacy.CLASSIFICATION_THRESHOLD_OPERATOR),
            "decision": decision,
            "policy": "official_NPR_AIGC_float32_sigmoid",
        },
        "manual_replay": {
            "raw_logit": float(raw_logit),
            "probability": probability,
            "ai_score": probability,
            "classification_decision": decision,
            "model_forward_calls": 1,
            "fc_hook_calls": 1,
            "official_logit_exact_match": True,
            "official_probability_exact_match": True,
        },
    }


def _processed(
    raw_logit: float = 1.0,
    *,
    probability: float | None = None,
) -> dict:
    return _score_payload(raw_logit, probability=probability)


def _real_input(
    root: Path,
    *,
    sample_id: str = "0" * 24,
    size: tuple[int, int] = (32, 32),
) -> dict:
    relative = Path("inputs") / f"{sample_id}.jpg"
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=(10, 20, 30)).save(
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
        "width": size[0],
        "height": size[1],
        "panel": True,
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
    feature_dir = root / runner.DEFAULT_ARTIFACTS_DIR / "run-a" / "features"
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
    _tensor, preprocess = runner._preprocess_image(image_path)
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


def test_frozen_contract_runtime_and_adapter_inventory():
    assert runner.RUN_MANIFEST_SCHEMA == "npr_balanced_run_manifest_v2"
    assert runner.RUN_CONFIG_SCHEMA == "npr_balanced_run_config_v2"
    assert runner.RUNTIME_SUMMARY_SCHEMA == "npr_balanced_runtime_summary_v2"
    assert runner.CPU_PREFLIGHT_SCHEMA == "npr_balanced_cpu_preflight_v1"
    assert runner.FROZEN_PROFILE == (
        "author_documented_aigcdetect_native_even_trim_completion"
    )
    assert runner.DEFAULT_SEED == runner.legacy.MODEL_SEED == 100
    assert runner.FROZEN_PYTHON_EXECUTABLE == Path(
        "/root/.cache/claimforge/venvs/" "npr-balanced-torch2.8.0/bin/python"
    )
    assert runner.FROZEN_RUNTIME_VERSIONS == {
        "python": "3.12.3",
        "torch": "2.8.0+cu128",
        "torch_distribution": "2.8.0+cu128",
        "torchvision": "0.23.0+cu128",
        "torchvision_distribution": "0.23.0+cu128",
        "numpy": "2.2.6",
        "Pillow": "11.1.0",
        "scikit-learn": "1.8.0",
        "scipy": "1.17.1",
        "joblib": "1.5.3",
        "threadpoolctl": "3.6.0",
        "setuptools": "75.8.0",
    }
    assert runner.FROZEN_PYVENV_CONFIG_SHA256 == (
        "35470b7542154bebe1a55dac3c8760e7638711ff9b166285694e5156186acd06"
    )
    assert runner.SCORE_SPEC.as_dict() == {
        "key": "ai_score",
        "direction": "higher_means_fake",
        "fixed_threshold": 0.5,
        "threshold_operator": ">",
    }
    assert runner.ARTIFACT_CONTRACT["feature"] == {
        "format": "NumPy .npy, allow_pickle=False",
        "shape": [runner.legacy.FEATURE_DIMENSION],
        "dtype": "float32",
        "nbytes": runner.legacy.FEATURE_DIMENSION * 4,
        "finite": True,
        "semantics": ("official_fc1_input_after_adaptive_global_average_pool"),
        "allow_pickle": False,
        "exact_fc1_and_sigmoid_replay_on_recorded_device": True,
        "visibility": "local_only_gitignored_output",
    }
    assert runner.TASK_SCOPE == {
        "primary_task": "T1_whole_image_AIGC_detection",
        "valid_for_t1": True,
        "valid_for_t2": False,
        "localization_output": None,
        "native_dense_output": False,
    }
    assert runner.ADAPTER_SOURCE_PATHS == (
        ".gitignore",
        "eval/__init__.py",
        "eval/opensource/__init__.py",
        "eval/opensource/run_npr_balanced.py",
        "eval/opensource/analyze_npr_balanced.py",
        "eval/opensource/run_npr.py",
        "eval/opensource/analyze_npr_run.py",
        "eval/opensource/canonical_release.py",
        "eval/opensource/balanced_run_contract.py",
        "eval/opensource/balanced250_metrics.py",
        "eval/opensource/common.py",
        "eval/opensource/npr_metrics.py",
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


def test_preprocess_contract_states_author_documented_completion():
    serialized = stable_json(runner.PREPROCESS_CONTRACT)
    assert runner.PREPROCESS_CONTRACT["profile"] == runner.FROZEN_PROFILE
    assert "author_documented" in serialized
    assert "official_aigcdetect_native_even_trim" not in serialized
    assert runner.PREPROCESS_CONTRACT["resize"] is None
    assert runner.PREPROCESS_CONTRACT["crop"] is None
    assert runner.PREPROCESS_CONTRACT["batch_size"] == 1
    assert runner.PREPROCESS_CONTRACT["trim_bottom_if_height_odd"] is True
    assert runner.PREPROCESS_CONTRACT["trim_right_if_width_odd"] is True


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
    expected_ids: set[str] = set()
    counts: Counter[str] = Counter()
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


def test_formal_odd_dimension_census_and_native_even_trim(balanced_release):
    counts = {
        "odd_width": sum(int(row["width"]) % 2 for row in balanced_release.inputs),
        "odd_height": sum(int(row["height"]) % 2 for row in balanced_release.inputs),
        "both_odd": sum(
            int(row["width"]) % 2 == 1 and int(row["height"]) % 2 == 1
            for row in balanced_release.inputs
        ),
        "either_odd": sum(
            int(row["width"]) % 2 == 1 or int(row["height"]) % 2 == 1
            for row in balanced_release.inputs
        ),
    }
    assert counts == {
        "odd_width": 81,
        "odd_height": 376,
        "both_odd": 13,
        "either_odd": 444,
    }
    assert counts == runner.FORMAL_ODD_DIMENSION_COUNTS
    assert runner.EXPECTED_ODD_DIMENSION_IMAGES == 444
    assert runner.effective_native_size(5, 3) == (4, 2)
    assert runner.effective_native_size(6, 3) == (6, 2)
    assert runner.effective_native_size(5, 4) == (4, 4)
    assert runner.effective_native_size(6, 4) == (6, 4)
    with pytest.raises(ValueError, match="exceed one pixel"):
        runner.effective_native_size(1, 2)


def test_preprocess_is_exact_legacy_tensor_with_renamed_audit_profile(
    tmp_path: Path,
):
    rgb = np.arange(45, dtype=np.uint8).reshape(3, 5, 3)
    path = tmp_path / "odd.png"
    Image.fromarray(rgb).save(path)
    expected_tensor, legacy_audit = runner.legacy.preprocess_image(path)
    actual_tensor, audit = runner._preprocess_image(path)
    import torch

    assert torch.equal(actual_tensor, expected_tensor)
    assert audit == {**legacy_audit, "profile": runner.FROZEN_PROFILE}
    assert audit["decoded_size"] == [5, 3]
    assert audit["effective_size"] == [4, 2]
    assert audit["trim_right"] == 1
    assert audit["trim_bottom"] == 1
    assert audit["tensor_shape"] == [3, 2, 4]
    assert audit["npr_residual_shape"] == [3, 2, 4]


def test_dual_odd_cpu_golden_constants_and_schema_are_exact():
    assert runner.CPU_GOLDEN_SAMPLE_ID == "7aeae0f17050bf766257b47d"
    assert runner.CPU_GOLDEN_INPUT_PATH == (
        "outputs/opensource/balanced250_v1/images/" "7aeae0f17050bf766257b47d.jpg"
    )
    assert runner.CPU_GOLDEN_IMAGE_SHA256 == (
        "21bfef64a1863cda43e122846c6cde1c40d97adcc33f813409f8204732e5093b"
    )
    assert runner.CPU_GOLDEN_DECODED_RGB_SHA256 == (
        "51e0da2279f209abe1b8349f0d215df0b2a989291c6ab02026b8b8b40e76a3e3"
    )
    assert runner.CPU_GOLDEN_TENSOR_SHA256 == (
        "1ff28c34fdfdf89c8a684d99b71fbe78bf3e582ab1c5b0c9192b4ff18fd04640"
    )
    assert runner.CPU_GOLDEN_RESIDUAL_SHA256 == (
        "84613a075de69e102fa62b71d99e7cca5f638774ae8b667928f91dcc6e8e9715"
    )
    assert runner.CPU_GOLDEN_FEATURE_ARRAY_SHA256 == (
        "521a7bfbd00dbfee27d21271649c0d24b100842bd64b23f44dc422e9acddbbd4"
    )
    assert runner.CPU_GOLDEN_FEATURE_FILE_SHA256 == (
        "6c3fc67b81c69bac75159dcfffd56d127393546a93fefca5d664c34c355bb8a4"
    )
    assert runner.CPU_GOLDEN_RAW_LOGIT == -84.44386291503906
    assert runner.CPU_GOLDEN_PROBABILITY == 2.120783389925294e-37
    preprocess = runner._expected_golden_preprocess()
    assert preprocess["decoded_size"] == [1285, 1137]
    assert preprocess["effective_size"] == [1284, 1136]
    assert preprocess["trim_bottom"] == 1
    assert preprocess["trim_right"] == 1
    assert preprocess["decoded_rgb_sha256"] == (runner.CPU_GOLDEN_DECODED_RGB_SHA256)
    assert preprocess["tensor_sha256"] == runner.CPU_GOLDEN_TENSOR_SHA256
    assert preprocess["npr_residual_sha256"] == (runner.CPU_GOLDEN_RESIDUAL_SHA256)
    processed = runner._expected_golden_processed()
    assert processed["raw_logit"] == runner.CPU_GOLDEN_RAW_LOGIT
    assert processed["probability"] == runner.CPU_GOLDEN_PROBABILITY
    assert processed["ai_score"] == runner.CPU_GOLDEN_PROBABILITY
    assert processed["classification_decision"] is False
    assert processed["manual_replay"]["model_forward_calls"] == 1
    assert processed["manual_replay"]["fc_hook_calls"] == 1


def _exact_golden_record() -> dict:
    return {
        "sample_id": runner.CPU_GOLDEN_SAMPLE_ID,
        "input_path": runner.CPU_GOLDEN_INPUT_PATH,
        "image_sha256": runner.CPU_GOLDEN_IMAGE_SHA256,
        "input_width": 1285,
        "input_height": 1137,
        "effective_width": 1284,
        "effective_height": 1136,
        "trim_bottom": 1,
        "trim_right": 1,
        "preprocess": runner._expected_golden_preprocess(),
        "tensor_sha256": runner.CPU_GOLDEN_TENSOR_SHA256,
        "npr_residual_sha256": runner.CPU_GOLDEN_RESIDUAL_SHA256,
        "feature_file_sha256": runner.CPU_GOLDEN_FEATURE_FILE_SHA256,
        "feature_file_bytes": 2176,
        "feature_array_sha256": runner.CPU_GOLDEN_FEATURE_ARRAY_SHA256,
        "feature_shape": [runner.legacy.FEATURE_DIMENSION],
        "feature_dtype": "float32",
        "feature_nbytes": 2048,
        "raw_logit": runner.CPU_GOLDEN_RAW_LOGIT,
        "probability": runner.CPU_GOLDEN_PROBABILITY,
        "ai_score": runner.CPU_GOLDEN_PROBABILITY,
        "classification_decision": False,
        "full_image_forward": True,
        "model_forward_calls": 1,
        "fc_hook_calls": 1,
        "repeat_feature_file_sha256": (runner.CPU_GOLDEN_FEATURE_FILE_SHA256),
        "repeat_feature_file_bytes": 2176,
        "repeat_feature_array_sha256": (runner.CPU_GOLDEN_FEATURE_ARRAY_SHA256),
        "repeat_raw_logit": runner.CPU_GOLDEN_RAW_LOGIT,
        "repeat_probability": runner.CPU_GOLDEN_PROBABILITY,
        "repeat_ai_score": runner.CPU_GOLDEN_PROBABILITY,
        "repeat_classification_decision": False,
        "repeat_full_image_forward": True,
        "repeat_model_forward_calls": 1,
        "repeat_fc_hook_calls": 1,
        "repeat_byte_exact": True,
    }


def test_run_cpu_preflight_executes_two_full_image_forwards(
    monkeypatch,
    tmp_path: Path,
):
    import torch

    image_path = tmp_path / runner.CPU_GOLDEN_INPUT_PATH
    image_path.parent.mkdir(parents=True)
    image_path.write_bytes(b"golden")
    device = torch.device("cpu")
    runtime = {"device": "cpu"}
    monkeypatch.setattr(
        runner,
        "configure_runtime",
        lambda *_args, **_kwargs: (device, runtime),
    )
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    source = {"commit": runner.legacy.MODEL_SOURCE_COMMIT}
    assets = {
        "bundle_sha256": runner.EXPECTED_ASSET_BUNDLE_SHA256,
    }
    monkeypatch.setattr(
        runner,
        "verify_assets",
        lambda **_kwargs: (
            source,
            assets,
            {"fc1.weight": object(), "fc1.bias": object()},
            object(),
        ),
    )
    monkeypatch.setattr(
        runner,
        "_preprocess_image",
        lambda _path: (
            np.zeros((3, 2, 2), dtype=np.float32),
            runner._expected_golden_preprocess(),
        ),
    )
    monkeypatch.setattr(
        runner,
        "_load_model",
        lambda **_kwargs: (object(), {"model_mode": "eval"}),
    )
    calls = []

    def fake_infer(**_kwargs):
        calls.append("full")
        return (
            _processed(
                runner.CPU_GOLDEN_RAW_LOGIT,
                probability=runner.CPU_GOLDEN_PROBABILITY,
            ),
            np.zeros(
                runner.legacy.FEATURE_DIMENSION,
                dtype=np.float32,
            ),
            0,
            1.0,
        )

    monkeypatch.setattr(runner, "_infer_one", fake_infer)
    monkeypatch.setattr(
        runner,
        "_validate_golden_forward",
        lambda *_args, **_kwargs: (
            b"x" * 2176,
            runner.CPU_GOLDEN_FEATURE_ARRAY_SHA256,
        ),
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
        hf_source_root=Path("/hf-source"),
        checkpoint_path=Path("/checkpoint"),
    )
    assert calls == ["full", "full"]
    assert report["cuda_used"] is False
    assert report["cuda_tensor_operations"] is False
    assert report["cuda_initialized_before_cpu_model_load"] is False
    assert report["cuda_initialized_after_cpu_forwards"] is False
    assert report["golden"]["trim_bottom"] == 1
    assert report["golden"]["trim_right"] == 1
    assert report["golden"]["repeat_full_image_forward"] is True
    assert report["golden"]["repeat_model_forward_calls"] == 1
    assert report["golden"]["repeat_fc_hook_calls"] == 1

    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    with pytest.raises(RuntimeError, match="initialized before"):
        runner.run_cpu_preflight(
            repo_root=tmp_path,
            source_root=Path("/source"),
            hf_source_root=Path("/hf-source"),
            checkpoint_path=Path("/checkpoint"),
        )


def test_cpu_runtime_is_exact_and_invalid_device_seed_fail_closed():
    import torch

    assert torch.cuda.is_initialized() is False
    device, runtime = runner.configure_runtime(
        "cpu",
        seed=runner.DEFAULT_SEED,
    )
    assert str(device) == "cpu"
    assert runtime["packages"]["torch"]["version"] == "2.8.0+cu128"
    assert runtime["packages"]["torchvision"]["version"] == "0.23.0+cu128"
    assert runtime["packages"]["numpy"] == "2.2.6"
    assert runtime["packages"]["Pillow"] == "11.1.0"
    assert runtime["packages"]["setuptools"] == "75.8.0"
    assert runtime["venv"]["include_system_site_packages"] is False
    assert runtime["venv"]["pyvenv_cfg_sha256"] == (runner.FROZEN_PYVENV_CONFIG_SHA256)
    assert runtime["process_environment"] == {
        "PYTHONHASHSEED": "100",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPYCACHEPREFIX": str(runner.FROZEN_PYTHONPYCACHEPREFIX),
        "python_dont_write_bytecode": True,
        "sys_pycache_prefix": str(runner.FROZEN_PYTHONPYCACHEPREFIX),
        "pycache_prefix_initially_empty": True,
    }
    assert runtime["batch_size"] == 1
    assert runtime["autocast"] is False
    assert runtime["deterministic_algorithms_enabled"] is True
    assert runtime["cudnn"]["benchmark"] is False
    assert torch.cuda.is_initialized() is False
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


def test_startup_isolation_fails_closed_on_environment_drift(monkeypatch):
    assert (
        runner._startup_isolation_contract()["pycache_prefix_initially_empty"] is True
    )
    monkeypatch.setenv("PYTHONHASHSEED", "1")
    with pytest.raises(RuntimeError, match="startup isolation"):
        runner._startup_isolation_contract()


def test_verified_source_inventory_ignores_only_bytecode_cache(
    monkeypatch,
    tmp_path: Path,
):
    monkeypatch.setattr(
        runner,
        "_git_status_lines",
        lambda _root: ["?? networks/__pycache__/resnet.cpython-312.pyc"],
    )
    assert runner._validate_source_inventory(tmp_path) == {
        "tracked_and_non_cache_untracked_clean": True,
        "untracked_bytecode_caches_ignored": 1,
        "bytecode_cache_execution": False,
        "loader": "compile_verified_utf8_source_bytes_no_pyc",
    }
    monkeypatch.setattr(
        runner,
        "_git_status_lines",
        lambda _root: ["?? networks/resnet.py.backup"],
    )
    with pytest.raises(ValueError, match="inventory drifted"):
        runner._validate_source_inventory(tmp_path)


def test_formal_visibility_census_and_nonlocal_not_applicable(
    balanced_release,
):
    census = {
        condition: Counter()
        for condition in (
            "local_mouse",
            "local_cat",
            "local_trash_can",
        )
    }
    for row in balanced_release.inputs:
        diagnostic = runner._visibility_diagnostic(
            row,
            repo_root=REPO_ROOT,
        )
        condition = row["condition"]
        if condition in census:
            census[condition][diagnostic["edit_visibility"]] += 1
            assert 0.0 < diagnostic["edit_visible_gt_fraction"] <= 1.0
            evidence = diagnostic["edit_visibility_evidence"]
            assert evidence["gt_mask_kind"] == "exact_diff"
            assert evidence["visible_positive_pixel_centers"] > 0
            assert (
                evidence["visible_positive_pixel_centers"]
                <= evidence["total_positive_pixels"]
            )
        else:
            assert diagnostic["edit_visibility"] == "not_applicable"
            assert diagnostic["edit_visible_gt_fraction"] is None
            assert diagnostic["edit_visibility_evidence"]["gt_mask_kind"] in (
                "all_zero",
                "not_applicable",
            )
    assert {
        condition: {
            category: counts[category] for category in ("full", "partial", "none")
        }
        for condition, counts in census.items()
    } == {
        "local_mouse": {"full": 250, "partial": 0, "none": 0},
        "local_cat": {"full": 239, "partial": 11, "none": 0},
        "local_trash_can": {"full": 211, "partial": 39, "none": 0},
    }


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
            "edit_visibility_evidence": {
                "basis": "input geometry diagnostic only",
            },
        }
    )


def test_score_accepts_finite_raw_logit_with_float32_sigmoid_underflow():
    payload = _score_payload(
        runner.CPU_GOLDEN_RAW_LOGIT,
        probability=runner.CPU_GOLDEN_PROBABILITY,
    )
    assert np.isfinite(payload["raw_logit"])
    assert 0.0 <= payload["probability"] <= 1.0
    runner._validate_score_payload(payload, sample_id="golden")

    underflow = _score_payload(-1_000.0, probability=0.0)
    assert underflow["raw_logit"] == -1_000.0
    assert underflow["probability"] == 0.0
    assert underflow["classification_decision"] is False
    runner._validate_score_payload(underflow, sample_id="underflow")


@pytest.mark.parametrize("probability", [-0.1, 1.1])
def test_probability_must_be_in_unit_interval(probability: float):
    payload = _score_payload(0.0, probability=probability)
    with pytest.raises(ValueError, match=r"outside \[0, ?1\]"):
        runner._validate_score_payload(payload, sample_id="range")


@pytest.mark.parametrize("raw_logit", [float("nan"), float("inf"), -float("inf")])
def test_raw_logit_must_remain_finite(raw_logit: float):
    payload = _score_payload(raw_logit, probability=0.5)
    with pytest.raises(ValueError, match="finite"):
        runner._validate_score_payload(payload, sample_id="raw")


def test_score_validation_does_not_apply_a_cross_device_cpu_sigmoid_gate():
    raw_logit = 2.6707892417907715
    hypothetical_cuda_probability = 0.9352807402610779
    cpu_probability = _float32_sigmoid(raw_logit)
    assert hypothetical_cuda_probability != cpu_probability
    payload = _score_payload(
        raw_logit,
        probability=hypothetical_cuda_probability,
    )
    runner._validate_score_payload(payload, sample_id="cuda-origin")


def test_strict_threshold_equality_is_not_fake():
    payload = _score_payload(0.0, probability=0.5)
    assert payload["classification_decision"] is False
    runner._validate_score_payload(payload, sample_id="threshold")


def test_result_keyset_score_feature_and_preprocess_are_exact(
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
    assert result["preprocess"]["profile"] == runner.FROZEN_PROFILE
    assert result["npr_feature"]["shape"] == [runner.legacy.FEATURE_DIMENSION]
    assert result["npr_feature"]["dtype"] == "float32"
    assert result["npr_feature_nbytes"] == (runner.legacy.FEATURE_DIMENSION * 4)
    assert result["npr_feature_array_sha256"] == (result["npr_feature"]["array_sha256"])

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


def test_feature_inventory_rejects_tamper_extra_nan_and_symlink(
    tmp_path: Path,
):
    result, input_row, _feature_dir = _ok_result(tmp_path / "tamper")
    feature_path = tmp_path / "tamper" / result["npr_feature"]["relative_path"]
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
    path = tmp_path / "symlink" / result["npr_feature"]["relative_path"]
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
    audited_checkpoint_state,
):
    import torch

    feature = np.zeros(
        runner.legacy.FEATURE_DIMENSION,
        dtype=np.float32,
    )
    feature[0] = 1.0
    with torch.inference_mode():
        raw_logit = float(
            torch.nn.functional.linear(
                torch.from_numpy(feature).reshape(1, -1),
                audited_checkpoint_state["fc1.weight"],
                audited_checkpoint_state["fc1.bias"],
            )
            .reshape(())
            .item()
        )
        probability = _float32_sigmoid(raw_logit)
    result, _input_row, _feature_dir = _ok_result(
        tmp_path,
        raw_logit=raw_logit,
        probability=probability,
        feature=feature,
    )
    latest = {str(result["sample_id"]): result}
    replay_devices = []
    original_linear = torch.nn.functional.linear

    def capture_linear(replay_feature, replay_weight, replay_bias):
        replay_devices.append(
            (
                replay_feature.device,
                replay_weight.device,
                replay_bias.device,
            )
        )
        return original_linear(
            replay_feature,
            replay_weight,
            replay_bias,
        )

    monkeypatch.setattr(torch.nn.functional, "linear", capture_linear)
    with pytest.raises(ValueError, match="explicit configured device"):
        runner._validate_latest_feature_head_replay(
            latest_by_sample_id=latest,
            repo_root=tmp_path,
            run_id="run-a",
            checkpoint_state=audited_checkpoint_state,
            device="cpu",
        )
    assert (
        runner._validate_latest_feature_head_replay(
            latest_by_sample_id=latest,
            repo_root=tmp_path,
            run_id="run-a",
            checkpoint_state=audited_checkpoint_state,
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


def test_output_directories_reject_collisions_escape_and_nonempty_roots(
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
    with pytest.raises(ValueError, match="repository"):
        runner._prepare_output_directories(
            repo_root=tmp_path,
            run_dir=tmp_path.parent / "escape-results",
            feature_root=tmp_path / "artifacts",
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


def _model_load_record() -> dict:
    return {
        "class_module": (
            f"_claimforge_npr_verified_" f"{runner.legacy.MODEL_SOURCE_COMMIT[:12]}"
        ),
        "class_name": "ResNet",
        "construction_api": ("verified_source_bytes.resnet50(num_classes=1)"),
        "source_loader": "compile_verified_utf8_source_bytes_no_pyc",
        "checkpoint_load": {
            "api": "torch.load",
            "weights_only": True,
            "map_location": "cpu",
            "strict": True,
            "missing_keys": [],
            "unexpected_keys": [],
        },
        "model_mode": "eval",
        "feature_dimension": runner.legacy.FEATURE_DIMENSION,
        "parameters": runner.legacy.CHECKPOINT["trainable_parameters"],
        "network_access": False,
    }


def test_cpu_preflight_validator_requires_exact_two_forward_evidence():
    _device, runtime = runner.configure_runtime("cpu")
    source = {"commit": runner.legacy.MODEL_SOURCE_COMMIT}
    assets = {"bundle_sha256": runner.EXPECTED_ASSET_BUNDLE_SHA256}
    report = {
        "schema_version": runner.CPU_PREFLIGHT_SCHEMA,
        "status": "passed",
        "source": source,
        "assets": assets,
        "model_load": _model_load_record(),
        "runtime": runtime,
        "golden": _exact_golden_record(),
        "cuda_used": False,
        "cuda_tensor_operations": False,
        "cuda_initialized_before_cpu_model_load": False,
        "cuda_initialized_after_cpu_forwards": False,
        "dataset_manifest_loaded": False,
    }
    runner._validate_preflight_report(
        report,
        source=source,
        assets=assets,
    )
    mutations = (
        ("report", "cuda_tensor_operations", True),
        ("report", "dataset_manifest_loaded", True),
        ("golden", "trim_bottom", 0),
        ("golden", "repeat_feature_file_sha256", "0" * 64),
        ("golden", "repeat_probability", 0.0),
        ("golden", "repeat_model_forward_calls", 0),
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
                assets=assets,
            )


def _run_args(root: Path, *, resume: bool = False) -> argparse.Namespace:
    return argparse.Namespace(
        repo_root=root,
        dataset_manifest=runner.DEFAULT_DATASET_MANIFEST,
        source_root=Path("/source"),
        hf_source_root=Path("/hf-source"),
        checkpoint=Path("/checkpoint"),
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
    source = {
        "commit": runner.legacy.MODEL_SOURCE_COMMIT,
        "hf_space": {"commit": runner.legacy.HF_SPACE_COMMIT},
    }
    assets = {
        "bundle_sha256": runner.EXPECTED_ASSET_BUNDLE_SHA256,
        "license": runner.legacy.LICENSE_RECORD,
        "commercial_clearance_claimed": False,
    }
    model_load = _model_load_record()
    preflight = {
        "schema_version": runner.CPU_PREFLIGHT_SCHEMA,
        "status": "passed",
        "source": source,
        "assets": assets,
        "model_load": model_load,
    }
    monkeypatch.setattr(
        runner,
        "run_cpu_preflight",
        lambda **_kwargs: preflight,
    )
    state = {
        "fc1.weight": torch.zeros(
            (1, runner.legacy.FEATURE_DIMENSION),
            dtype=torch.float32,
        ),
        "fc1.bias": torch.tensor([1.0], dtype=torch.float32),
    }
    monkeypatch.setattr(
        runner,
        "verify_assets",
        lambda **_kwargs: (source, assets, state, object()),
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
    monkeypatch.setattr(
        runner,
        "_local_artifact_policy",
        lambda _root: {
            "visibility": "local_only",
            "gitignored": True,
        },
    )
    load_calls: list[str] = []

    def fake_load(**_kwargs):
        load_calls.append("load")
        return object(), model_load

    monkeypatch.setattr(runner, "_load_model", fake_load)
    feature = np.zeros(
        runner.legacy.FEATURE_DIMENSION,
        dtype=np.float32,
    )
    probability = _float32_sigmoid(1.0)
    monkeypatch.setattr(
        runner,
        "_infer_one",
        lambda **_kwargs: (
            _processed(1.0, probability=probability),
            feature.copy(),
            0,
            2.0,
        ),
    )
    return input_row, contract_dict, load_calls


def test_mocked_single_run_writes_v2_runtime_artifacts_and_resumes(
    monkeypatch,
    tmp_path: Path,
):
    input_row, contract, load_calls = _patch_single_run(
        monkeypatch,
        tmp_path,
    )
    args = _run_args(tmp_path)
    assert runner.run(args) == 0
    assert load_calls == ["load"]
    run_dir = tmp_path / runner.DEFAULT_RESULTS_DIR / "single-run"
    manifest = runner._load_json_strict(
        run_dir / "manifest.json",
        "manifest",
    )
    assert manifest["status"] == "complete"
    assert set(manifest["immutable"]) == runner.IMMUTABLE_CONFIG_KEYS
    assert manifest["immutable"]["dataset_contract"] == contract
    assert manifest["immutable"]["task_scope"] == runner.TASK_SCOPE
    assert manifest["immutable"]["preprocess"] == (runner.PREPROCESS_CONTRACT)
    assert manifest["outputs"]["feature_files"] == 1
    results_path = run_dir / "results.jsonl"
    rows = read_jsonl(results_path)
    assert len(rows) == 1
    assert rows[0]["sample_id"] == input_row["sample_id"]
    assert rows[0]["schema_version"] == runner.RESULT_SCHEMA_VERSION
    assert rows[0]["npr_feature"]["file_bytes"] == 2176
    assert rows[0]["ai_score"] == _float32_sigmoid(1.0)
    summary = runner._load_json_strict(
        run_dir / "summary.json",
        "summary",
    )
    assert summary["summary_kind"] == "runtime_coverage_only"
    assert summary["scientific_metrics"] is None
    assert summary["scientific_metrics_owner"] == ("analyze_npr_balanced.py")
    assert summary["coverage"]["is_complete"] is True

    before = results_path.read_bytes()
    args.resume = True
    load_calls.clear()
    assert runner.run(args) == 0
    assert load_calls == []
    assert results_path.read_bytes() == before
    resumed = runner._load_json_strict(
        run_dir / "manifest.json",
        "resumed manifest",
    )
    assert resumed["execution"]["resume_skips"] == 1
    assert resumed["execution"]["new_successes"] == 0
    assert resumed["execution"]["physical_result_rows"] == 1


def test_post_feature_failure_removes_orphan_and_latest_error_is_incomplete(
    monkeypatch,
    tmp_path: Path,
):
    _patch_single_run(monkeypatch, tmp_path)
    original = runner._build_ok_result

    def fail_after_feature_write(**kwargs):
        original(**kwargs)
        raise RuntimeError("post-feature validation failed")

    monkeypatch.setattr(
        runner,
        "_build_ok_result",
        fail_after_feature_write,
    )
    assert runner.run(_run_args(tmp_path)) == 2
    run_dir = tmp_path / runner.DEFAULT_RESULTS_DIR / "single-run"
    rows = read_jsonl(run_dir / "results.jsonl")
    assert len(rows) == 1
    assert rows[0]["status"] == "error"
    assert rows[0]["valid_for_metrics"] is False
    feature_dir = tmp_path / runner.DEFAULT_ARTIFACTS_DIR / "single-run" / "features"
    assert list(feature_dir.iterdir()) == []
    manifest = runner._load_json_strict(
        run_dir / "manifest.json",
        "manifest",
    )
    assert manifest["status"] == "incomplete"
    assert manifest["outputs"]["feature_files"] == 0
    assert manifest["execution"]["new_errors"] == 1


def test_resume_rejects_orphan_feature_before_manifest_mutation(
    monkeypatch,
    tmp_path: Path,
):
    _patch_single_run(monkeypatch, tmp_path)
    args = _run_args(tmp_path)
    assert runner.run(args) == 0
    run_dir = tmp_path / runner.DEFAULT_RESULTS_DIR / "single-run"
    manifest_path = run_dir / "manifest.json"
    before = manifest_path.read_bytes()
    feature_dir = tmp_path / runner.DEFAULT_ARTIFACTS_DIR / "single-run" / "features"
    (feature_dir / "foreign.npy").write_bytes(b"foreign")
    args.resume = True
    with pytest.raises(ValueError, match="inventory mismatch"):
        runner.run(args)
    assert manifest_path.read_bytes() == before


def test_resume_rejects_immutable_drift_before_manifest_mutation(
    monkeypatch,
    tmp_path: Path,
):
    _patch_single_run(monkeypatch, tmp_path)
    args = _run_args(tmp_path)
    assert runner.run(args) == 0
    run_dir = tmp_path / runner.DEFAULT_RESULTS_DIR / "single-run"
    manifest_path = run_dir / "manifest.json"
    before = manifest_path.read_bytes()
    monkeypatch.setattr(
        runner,
        "adapter_source_contract",
        lambda _root: {
            "adapter.py": {
                "path": "adapter.py",
                "bytes": 2,
                "sha256": "f" * 64,
            }
        },
    )
    args.resume = True
    with pytest.raises(ValueError, match="fingerprint/config drifted"):
        runner.run(args)
    assert manifest_path.read_bytes() == before


def test_resume_rejects_duplicate_successful_physical_attempts(
    monkeypatch,
    tmp_path: Path,
):
    _patch_single_run(monkeypatch, tmp_path)
    args = _run_args(tmp_path)
    assert runner.run(args) == 0
    run_dir = tmp_path / runner.DEFAULT_RESULTS_DIR / "single-run"
    results_path = run_dir / "results.jsonl"
    manifest_path = run_dir / "manifest.json"
    rows = runner._read_jsonl_strict(results_path, "results")
    runner.atomic_write_jsonl(results_path, [rows[0], copy.deepcopy(rows[0])])
    manifest = runner._load_json_strict(manifest_path, "manifest")
    manifest["outputs"]["results_sha256"] = runner.sha256_file(results_path)
    manifest["execution"]["physical_result_rows"] = 2
    manifest["execution"]["latest_result_rows"] = 1
    manifest["execution"]["superseded_attempts"] = 1
    runner.atomic_write_json(manifest_path, manifest)
    before = manifest_path.read_bytes()
    args.resume = True
    with pytest.raises(ValueError, match="duplicate|physical"):
        runner.run(args)
    assert manifest_path.read_bytes() == before


def test_preflight_mode_never_loads_dataset_or_requested_accelerator(
    monkeypatch,
    tmp_path: Path,
    capsys,
):
    source = {"commit": runner.legacy.MODEL_SOURCE_COMMIT}
    assets = {"bundle_sha256": runner.EXPECTED_ASSET_BUNDLE_SHA256}
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
        runner,
        "verify_assets",
        lambda **_kwargs: (source, assets, {}, object()),
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
