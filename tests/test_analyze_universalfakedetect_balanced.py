from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from eval.opensource import (
    analyze_universalfakedetect_balanced as analyzer,
)
from eval.opensource.canonical_release import load_canonical_release


REPO_ROOT = Path(__file__).resolve().parents[1]


class _FakeRunner:
    RUN_MANIFEST_SCHEMA = analyzer.EXPECTED_RUN_MANIFEST_SCHEMA
    RUN_CONFIG_SCHEMA = analyzer.EXPECTED_RUN_CONFIG_SCHEMA
    RUNTIME_SUMMARY_SCHEMA = analyzer.EXPECTED_RUNTIME_SUMMARY_SCHEMA
    CPU_PREFLIGHT_SCHEMA = analyzer.EXPECTED_CPU_PREFLIGHT_SCHEMA
    DEFAULT_SEED = analyzer.EXPECTED_RUNTIME_SEED
    FROZEN_PROFILE = analyzer.CURRENT_PROFILE
    FROZEN_PYTHON_EXECUTABLE = analyzer.EXPECTED_FROZEN_PYTHON_EXECUTABLE
    FROZEN_VENV_PREFIX = analyzer.EXPECTED_FROZEN_VENV_PREFIX
    FROZEN_PYVENV_CONFIG_SHA256 = (
        analyzer.EXPECTED_FROZEN_PYVENV_CONFIG_SHA256
    )
    FROZEN_RUNTIME_VERSIONS = analyzer.EXPECTED_FROZEN_RUNTIME_VERSIONS
    CUBLAS_WORKSPACE_CONFIG = analyzer.EXPECTED_CUBLAS_WORKSPACE_CONFIG
    MINIMUM_CUDA_FREE_BYTES = analyzer.EXPECTED_MINIMUM_CUDA_FREE_BYTES
    CPU_GOLDEN_SAMPLE_ID = analyzer.CPU_GOLDEN_SAMPLE_ID
    CPU_GOLDEN_INPUT_PATH = analyzer.CPU_GOLDEN_INPUT_PATH
    CPU_GOLDEN_IMAGE_SHA256 = analyzer.CPU_GOLDEN_IMAGE_SHA256
    CPU_GOLDEN_DECODED_RGB_SHA256 = (
        analyzer.CPU_GOLDEN_DECODED_RGB_SHA256
    )
    CPU_GOLDEN_CROP_RGB_SHA256 = analyzer.CPU_GOLDEN_CROP_RGB_SHA256
    CPU_GOLDEN_TENSOR_SHA256 = analyzer.CPU_GOLDEN_TENSOR_SHA256
    CPU_GOLDEN_FEATURE_FILE_SHA256 = (
        analyzer.CPU_GOLDEN_FEATURE_FILE_SHA256
    )
    CPU_GOLDEN_FEATURE_ARRAY_SHA256 = (
        analyzer.CPU_GOLDEN_FEATURE_ARRAY_SHA256
    )
    CPU_GOLDEN_RAW_LOGIT = analyzer.CPU_GOLDEN_RAW_LOGIT
    CPU_GOLDEN_PROBABILITY = analyzer.CPU_GOLDEN_PROBABILITY
    ADAPTER_SOURCE_PATHS = analyzer.EXPECTED_ADAPTER_SOURCE_PATHS
    IMMUTABLE_CONFIG_KEYS = analyzer.EXPECTED_IMMUTABLE_CONFIG_KEYS
    PREPROCESS_CONTRACT = analyzer.EXPECTED_PREPROCESS_CONTRACT
    MODEL_CONTRACT = analyzer.EXPECTED_MODEL_CONTRACT
    SOURCE_CHECKPOINT_DRIFT = analyzer.EXPECTED_SOURCE_CHECKPOINT_DRIFT
    TASK_SCOPE = analyzer.EXPECTED_TASK_SCOPE
    ARTIFACT_CONTRACT = analyzer.EXPECTED_ARTIFACT_CONTRACT

    @staticmethod
    def configure_runtime(device_text, *, seed):
        import torch

        assert seed == analyzer.EXPECTED_RUNTIME_SEED
        return (
            torch.device(device_text),
            _runtime(device_text),
        )

    @staticmethod
    def validate_runtime_contract(value, *, label="runtime"):
        del label
        return value

    @staticmethod
    def select_mode_inputs(*_args, **_kwargs):
        raise AssertionError("selection was not mocked")

    @staticmethod
    def _validate_runner_attempt(*_args, **_kwargs):
        return None

    @staticmethod
    def _valid_run_id(value: str) -> str:
        if (
            not isinstance(value, str)
            or not value
            or Path(value).name != value
            or value in (".", "..")
        ):
            raise ValueError("invalid run-id")
        return value


@pytest.fixture(autouse=True)
def _fake_runner(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(analyzer, "_runner", lambda: _FakeRunner)


@pytest.fixture(scope="module")
def canonical_release():
    return load_canonical_release(
        REPO_ROOT,
        Path("outputs/opensource/balanced250_v1/manifest.json"),
        verify_files=False,
    )


def _save_feature(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        np.save(handle, value, allow_pickle=False)


def _classification(
    raw: float = 0.0,
    *,
    probability: float | None = None,
) -> tuple[float, bool, dict, dict, dict]:
    if probability is None:
        probability = analyzer.legacy._float32_sigmoid(raw)
    decision = probability > analyzer.legacy.CLASSIFICATION_THRESHOLD
    classification = {
        "raw_logit": raw,
        "probability": probability,
        "ai_score": probability,
        "score": probability,
        "decision": decision,
        "threshold": analyzer.legacy.CLASSIFICATION_THRESHOLD,
        "threshold_operator": (
            analyzer.legacy.CLASSIFICATION_THRESHOLD_OPERATOR
        ),
        "semantics": "official_sigmoid_probability_higher_is_fake",
    }
    t1 = dict(classification)
    t1.pop("semantics")
    t1["policy"] = "official_UFD_CLIP_linear_probe_probability"
    manual = {
        "raw_logit": raw,
        "probability": probability,
        "ai_score": probability,
        "classification_decision": decision,
        "official_logit_exact_match": True,
        "official_probability_exact_match": True,
        "model_forward_calls": 1,
        "fc_hook_calls": 1,
    }
    return probability, decision, classification, t1, manual


def _runtime(device: str = "cpu") -> dict:
    value = {
        "device": device,
        "python": {
            "implementation": "CPython",
            "version": analyzer.EXPECTED_FROZEN_RUNTIME_VERSIONS["python"],
            "executable": str(analyzer.EXPECTED_FROZEN_PYTHON_EXECUTABLE),
        },
        "venv": {
            "prefix": analyzer.EXPECTED_FROZEN_VENV_PREFIX.as_posix(),
            "base_prefix": "/usr",
            "pyvenv_cfg_path": (
                analyzer.EXPECTED_FROZEN_VENV_PREFIX / "pyvenv.cfg"
            ).as_posix(),
            "pyvenv_cfg_sha256": (
                analyzer.EXPECTED_FROZEN_PYVENV_CONFIG_SHA256
            ),
            "include_system_site_packages": False,
        },
        "platform": "test-platform",
        "packages": {
            "torch": {
                "version": analyzer.EXPECTED_FROZEN_RUNTIME_VERSIONS["torch"],
                "distribution_version": (
                    analyzer.EXPECTED_FROZEN_RUNTIME_VERSIONS[
                        "torch_distribution"
                    ]
                ),
                "cuda_runtime": "12.8",
                "cudnn_version": 91002,
            },
            "torchvision": {
                "version": (
                    analyzer.EXPECTED_FROZEN_RUNTIME_VERSIONS["torchvision"]
                ),
                "distribution_version": (
                    analyzer.EXPECTED_FROZEN_RUNTIME_VERSIONS[
                        "torchvision_distribution"
                    ]
                ),
            },
            "numpy": analyzer.EXPECTED_FROZEN_RUNTIME_VERSIONS["numpy"],
            "Pillow": analyzer.EXPECTED_FROZEN_RUNTIME_VERSIONS["Pillow"],
            "ftfy": analyzer.EXPECTED_FROZEN_RUNTIME_VERSIONS["ftfy"],
            "regex": analyzer.EXPECTED_FROZEN_RUNTIME_VERSIONS["regex"],
            "setuptools": (
                analyzer.EXPECTED_FROZEN_RUNTIME_VERSIONS["setuptools"]
            ),
            "tqdm": analyzer.EXPECTED_FROZEN_RUNTIME_VERSIONS["tqdm"],
        },
        "seed": analyzer.EXPECTED_RUNTIME_SEED,
        "preprocess_profile": analyzer.CURRENT_PROFILE,
        "feature_dtype": "float32",
        "batch_size": 1,
        "autocast": False,
        "deterministic_algorithms_enabled": True,
        "deterministic_algorithms_warn_only": False,
        "cublas_workspace_config": analyzer.EXPECTED_CUBLAS_WORKSPACE_CONFIG,
        "cudnn": {
            "enabled": True,
            "benchmark": False,
            "deterministic": True,
            "allow_tf32": False,
        },
        "matmul_allow_tf32": False,
        "float32_matmul_precision": "highest",
        "minimum_cuda_free_bytes": (
            analyzer.EXPECTED_MINIMUM_CUDA_FREE_BYTES
        ),
    }
    if device.startswith("cuda:"):
        value["cuda"] = {
            "runtime": "12.8",
            "device_index": int(device.split(":")[1]),
            "device_name": "GPU",
            "total_memory_bytes": 80 * 1024**3,
            "capability": [9, 0],
        }
    return value


def _row(
    root: Path,
    *,
    sample_id: str = "sample",
    run_id: str = "run-a",
    fingerprint: str = "a" * 64,
    feature: np.ndarray | None = None,
    raw: float = 0.0,
    probability: float | None = None,
) -> dict:
    value = (
        np.arange(analyzer.FEATURE_DIMENSION, dtype=np.float32)
        if feature is None
        else np.asarray(feature)
    )
    relative = (
        Path("outputs")
        / "opensource"
        / "universalfakedetect"
        / run_id
        / "clip_features"
        / f"{sample_id}.npy"
    )
    path = root / relative
    _save_feature(path, value)
    file_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    array_sha = analyzer._array_sha256(value)
    probability, decision, classification, t1, manual = _classification(
        raw,
        probability=probability,
    )
    return {
        "schema_version": "opensource_result_v2",
        "run_id": run_id,
        "run_manifest_fingerprint": fingerprint,
        "config_fingerprint": fingerprint,
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
        "input_width": 32,
        "input_height": 32,
        "status": "ok",
        "valid_for_metrics": True,
        "completed_at": "2026-07-26T00:00:00+00:00",
        "model": analyzer.legacy.MODEL_NAME,
        "model_slug": analyzer.legacy.MODEL_SLUG,
        "model_arch": analyzer.legacy.MODEL_ARCH,
        "model_source_commit": analyzer.legacy.MODEL_SOURCE_COMMIT,
        "asset_bundle_sha256": "c" * 64,
        "preprocess_profile": analyzer.CURRENT_PROFILE,
        "task_scope": {
            "valid_for_t1": True,
            "valid_for_t2": False,
            "native_dense_output": False,
        },
        "edit_visibility": "not_applicable",
        "edit_visible_gt_fraction": None,
        "edit_visibility_evidence": {
            "basis": "authentic_input_has_no_edit",
        },
        "preprocess": {"geometry": analyzer._current_geometry(32, 32)},
        "preprocess_latency_ms": 1.0,
        "clip_feature": {
            "relative_path": relative.as_posix(),
            "sha256": file_sha,
            "file_bytes": path.stat().st_size,
            "array_sha256": array_sha,
            "dtype": "float32",
            "shape": [analyzer.FEATURE_DIMENSION],
            "nbytes": analyzer.FEATURE_NBYTES,
            "finite": True,
            "semantics": analyzer.FEATURE_SEMANTICS,
        },
        "clip_feature_path": relative.as_posix(),
        "clip_feature_sha256": file_sha,
        "clip_feature_array_sha256": array_sha,
        "clip_feature_shape": [analyzer.FEATURE_DIMENSION],
        "clip_feature_dtype": "float32",
        "clip_feature_nbytes": analyzer.FEATURE_NBYTES,
        "clip_feature_semantics": analyzer.FEATURE_SEMANTICS,
        "artifact_paths": {"clip_feature_npy": relative.as_posix()},
        "raw_logit": raw,
        "probability": probability,
        "ai_score": probability,
        "score": probability,
        "score_semantics": "official_sigmoid_probability_higher_is_fake",
        "classification_decision": decision,
        "classification_threshold": (
            analyzer.legacy.CLASSIFICATION_THRESHOLD
        ),
        "classification_threshold_operator": (
            analyzer.legacy.CLASSIFICATION_THRESHOLD_OPERATOR
        ),
        "classification": classification,
        "t1": t1,
        "manual_replay": manual,
        "latency_ms": 2.0,
        "peak_cuda_memory_bytes": 0,
    }


def _inventory(root: Path, row: dict) -> dict[str, analyzer.FeatureArtifact]:
    feature_dir = (
        root
        / "outputs"
        / "opensource"
        / "universalfakedetect"
        / row["run_id"]
        / "clip_features"
    )
    return analyzer.validate_feature_inventory(
        latest_results=[row],
        repo_root=root,
        feature_dir=feature_dir,
    )


def test_score_sigmoid_aliases_and_strict_threshold(tmp_path: Path):
    row = _row(tmp_path, raw=0.0)
    assert row["probability"] == 0.5
    assert row["classification_decision"] is False
    analyzer._validate_score_payload(row, sample_id="sample")

    tampered = copy.deepcopy(row)
    tampered["classification_threshold_operator"] = ">="
    with pytest.raises(ValueError, match="fixed threshold"):
        analyzer._validate_score_payload(tampered, sample_id="sample")

    tampered = copy.deepcopy(row)
    tampered["manual_replay"]["fc_hook_calls"] = 2
    with pytest.raises(ValueError, match="manual head replay"):
        analyzer._validate_score_payload(tampered, sample_id="sample")


def test_positive_sub_ulp_logit_uses_probability_threshold(tmp_path: Path):
    row = _row(tmp_path, raw=1e-8)
    assert row["raw_logit"] > 0.0
    assert row["probability"] == 0.5
    assert row["classification_decision"] is False
    analyzer._validate_score_payload(row, sample_id="sample")


def test_cuda_sigmoid_value_is_validated_by_recorded_device_replay(
    tmp_path: Path,
):
    raw_logit = 2.6707892417907715
    cuda_probability = 0.9352807402610779
    assert cuda_probability != analyzer.legacy._float32_sigmoid(raw_logit)
    row = _row(
        tmp_path,
        raw=raw_logit,
        probability=cuda_probability,
    )
    analyzer._validate_score_payload(row, sample_id="sample")


@pytest.mark.parametrize("probability", [-0.1, 1.1])
def test_score_probability_must_be_in_unit_interval(
    tmp_path: Path,
    probability: float,
):
    row = _row(tmp_path, probability=probability)
    with pytest.raises(ValueError, match=r"outside \[0,1\]"):
        analyzer._validate_score_payload(row, sample_id="sample")


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda row: row["clip_feature"].update(dtype="float64"), "metadata"),
        (
            lambda row: row.update(clip_feature_shape=[767]),
            "feature alias",
        ),
        (
            lambda row: row["clip_feature"].update(array_sha256="0" * 64),
            "array SHA",
        ),
    ],
)
def test_feature_inventory_is_exact_float32_768(
    tmp_path: Path,
    mutation,
    message: str,
):
    row = _row(tmp_path)
    mutation(row)
    with pytest.raises(ValueError, match=message):
        _inventory(tmp_path, row)


def test_feature_inventory_rejects_extra_file_and_noncanonical_npy(
    tmp_path: Path,
):
    row = _row(tmp_path)
    feature_dir = (
        tmp_path
        / "outputs"
        / "opensource"
        / "universalfakedetect"
        / row["run_id"]
        / "clip_features"
    )
    _save_feature(
        feature_dir / "extra.npy",
        np.zeros(analyzer.FEATURE_DIMENSION, dtype=np.float32),
    )
    with pytest.raises(ValueError, match="inventory mismatch"):
        _inventory(tmp_path, row)

    (feature_dir / "extra.npy").unlink()
    path = tmp_path / row["clip_feature_path"]
    with path.open("wb") as handle:
        np.save(
            handle,
            np.arange(analyzer.FEATURE_DIMENSION, dtype=np.float32),
            allow_pickle=False,
        )
        handle.write(b"trailing")
    row["clip_feature"]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    row["clip_feature"]["file_bytes"] = path.stat().st_size
    row["clip_feature_sha256"] = row["clip_feature"]["sha256"]
    with pytest.raises(ValueError, match="byte|NPY|metadata"):
        _inventory(tmp_path, row)


def test_feature_inventory_rejects_traversal_and_symlink_directory(
    tmp_path: Path,
):
    row = _row(tmp_path)
    row["clip_feature"]["relative_path"] = "../escape.npy"
    row["clip_feature_path"] = "../escape.npy"
    with pytest.raises(ValueError, match="traversing"):
        _inventory(tmp_path, row)

    other_root = tmp_path / "other"
    row = _row(other_root)
    feature_dir = (
        other_root
        / "outputs"
        / "opensource"
        / "universalfakedetect"
        / row["run_id"]
        / "clip_features"
    )
    target = other_root / "feature-target"
    feature_dir.replace(target)
    feature_dir.symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError, match="feature directory.*symlink"):
        _inventory(other_root, row)


def test_visibility_current_head_none_partial_full_and_na(canonical_release):
    by_id = {row["sample_id"]: row for row in canonical_release.inputs}
    cases = {
        "a5f0c416a390a5125e4e2fcc": ("none", 0.0),
        "c663f7fbae3a79c8fa73f7f5": (
            "partial",
            0.7109428768066071,
        ),
        "c839da4f1b1415d5c07f59de": ("full", 1.0),
    }
    for sample_id, (category, fraction) in cases.items():
        result = analyzer._independent_visibility_diagnostic(
            by_id[sample_id], repo_root=REPO_ROOT
        )
        assert result["edit_visibility"] == category
        assert result["edit_visible_gt_fraction"] == fraction
        assert (
            result["edit_visibility_evidence"]["gt"]["profile_id"]
            == analyzer.CURRENT_PROFILE
        )
    for condition in ("real", "fullframe_mouse"):
        row = next(
            item for item in canonical_release.inputs if item["condition"] == condition
        )
        result = analyzer._independent_visibility_diagnostic(
            row, repo_root=REPO_ROOT
        )
        assert result["edit_visibility"] == "not_applicable"
        assert result["edit_visible_gt_fraction"] is None


def test_physical_attempt_visibility_is_recomputed_and_tamper_rejected(
    canonical_release,
):
    input_row = next(
        row
        for row in canonical_release.inputs
        if row["sample_id"] == "c663f7fbae3a79c8fa73f7f5"
    )
    diagnostic = analyzer._independent_visibility_diagnostic(
        input_row, repo_root=REPO_ROOT
    )
    physical = {
        "sample_id": input_row["sample_id"],
        "status": "error",
        "preprocess_profile": analyzer.CURRENT_PROFILE,
        **diagnostic,
    }
    analyzer._validate_physical_attempts(
        physical=[physical],
        selected=[input_row],
        repo_root=REPO_ROOT,
        run_id="run",
        fingerprint="a" * 64,
    )
    physical["edit_visible_gt_fraction"] = 0.0
    with pytest.raises(ValueError, match="crop visibility changed"):
        analyzer._validate_physical_attempts(
            physical=[physical],
            selected=[input_row],
            repo_root=REPO_ROOT,
            run_id="run",
            fingerprint="a" * 64,
        )


def test_physical_attempt_rejects_checkpoint_era_profile(canonical_release):
    input_row = next(
        row for row in canonical_release.inputs if row["condition"] == "real"
    )
    physical = {
        "sample_id": input_row["sample_id"],
        "status": "error",
        "preprocess_profile": analyzer.legacy.CHECKPOINT_ERA_PROFILE,
        **analyzer._independent_visibility_diagnostic(
            input_row, repo_root=REPO_ROOT
        ),
    }
    with pytest.raises(ValueError, match="current-head-only"):
        analyzer._validate_physical_attempts(
            physical=[physical],
            selected=[input_row],
            repo_root=REPO_ROOT,
            run_id="run",
            fingerprint="a" * 64,
        )


def test_current_geometry_is_independent_current_head_only():
    geometry = analyzer._current_geometry(1800, 1350)
    assert geometry["profile_id"] == analyzer.legacy.CURRENT_PROFILE
    assert geometry["resize"]["enabled"] is False
    assert geometry["center_crop"]["size"] == [224, 224]
    assert geometry["effective_native_crop_xyxy"] == [
        788.0,
        563.0,
        1012.0,
        787.0,
    ]
    assert analyzer.legacy.compute_preprocess_geometry(
        1800, 1350, analyzer.CURRENT_PROFILE
    ) == geometry


def test_runtime_pin_is_exact_for_cpu_and_explicit_cuda():
    assert analyzer._validate_runtime_contract(
        _runtime("cpu"), label="runtime"
    )["device"] == "cpu"
    assert analyzer._validate_runtime_contract(
        _runtime("cuda:3"), label="runtime"
    )["cuda"]["device_index"] == 3
    for field, value in (
        ("autocast", True),
        ("preprocess_profile", analyzer.legacy.CHECKPOINT_ERA_PROFILE),
        ("feature_dtype", "float64"),
    ):
        changed = _runtime()
        changed[field] = value
        with pytest.raises(ValueError, match="numerical contract"):
            analyzer._validate_runtime_contract(changed, label="runtime")
    changed = _runtime()
    changed["packages"]["torch"]["version"] = "different"
    with pytest.raises(ValueError, match="frozen versions"):
        analyzer._validate_runtime_contract(changed, label="runtime")
    changed = _runtime()
    changed["python"]["executable"] = "/usr/bin/python3.12"
    with pytest.raises(ValueError, match="dedicated runtime"):
        analyzer._validate_runtime_contract(changed, label="runtime")
    changed = _runtime()
    changed["venv"]["include_system_site_packages"] = True
    with pytest.raises(ValueError, match="clean-environment"):
        analyzer._validate_runtime_contract(changed, label="runtime")


def test_cpu_preflight_is_independently_exact_and_cuda_free(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    image_path = tmp_path / analyzer.CPU_GOLDEN_INPUT_PATH
    image_path.parent.mkdir(parents=True)
    image_path.write_bytes(b"pinned-image-placeholder")
    preprocess = {
        "geometry": analyzer._current_geometry(1800, 1350),
        "decoded_rgb_sha256": analyzer.CPU_GOLDEN_DECODED_RGB_SHA256,
        "crop_rgb_sha256": analyzer.CPU_GOLDEN_CROP_RGB_SHA256,
        "crop_rgb_shape": [224, 224, 3],
        "crop_rgb_dtype": "uint8",
        "tensor_shape": [3, 224, 224],
        "tensor_dtype": "float32",
        "tensor_sha256": analyzer.CPU_GOLDEN_TENSOR_SHA256,
    }
    real_sha = analyzer.sha256_file
    monkeypatch.setattr(
        analyzer,
        "sha256_file",
        lambda path: (
            analyzer.CPU_GOLDEN_IMAGE_SHA256
            if Path(path) == image_path
            else real_sha(Path(path))
        ),
    )
    monkeypatch.setattr(
        analyzer.legacy,
        "preprocess_image",
        lambda path, profile: (
            np.zeros((3, 224, 224), dtype=np.float32),
            preprocess,
        ),
    )
    source = {"root": "/source", "source_files": {}}
    assets = {
        "bundle_sha256": "a" * 64,
        "head": {"path": "/head"},
        "backbone": {"path": "/backbone"},
    }
    model_load = {
        "source": source,
        "assets": assets,
        "class_module": "_claimforge_ufd_76a0e3e.models.clip_models",
        "class_name": "CLIPModel",
        "construction_api": "official models.get_model('CLIP:ViT-L/14')",
        "official_download_patch_calls": [
            {
                "url": analyzer.legacy.BACKBONE_CHECKPOINT["official_url"],
                "requested_cache_root": "/cache",
            }
        ],
        "urlopen_calls": 0,
        "network_blocked": True,
        "clip_torch_load_fallback_blocked": True,
        "head_load": {
            "api": "torch.load",
            "weights_only": True,
            "strict": True,
            "missing_keys": [],
            "unexpected_keys": [],
        },
        "feature_dimension": analyzer.FEATURE_DIMENSION,
        "visual_input_resolution": analyzer.legacy.MODEL_INPUT_SIZE,
        "head_parameters": analyzer.FEATURE_DIMENSION + 1,
    }
    golden = {
        "sample_id": analyzer.CPU_GOLDEN_SAMPLE_ID,
        "input_path": analyzer.CPU_GOLDEN_INPUT_PATH,
        "image_sha256": analyzer.CPU_GOLDEN_IMAGE_SHA256,
        "input_width": 1800,
        "input_height": 1350,
        "preprocess": preprocess,
        "tensor_sha256": analyzer.CPU_GOLDEN_TENSOR_SHA256,
        "feature_file_sha256": _FakeRunner.CPU_GOLDEN_FEATURE_FILE_SHA256,
        "feature_file_bytes": 3200,
        "feature_array_sha256": (
            _FakeRunner.CPU_GOLDEN_FEATURE_ARRAY_SHA256
        ),
        "feature_shape": [analyzer.FEATURE_DIMENSION],
        "feature_dtype": "float32",
        "feature_nbytes": analyzer.FEATURE_NBYTES,
        "raw_logit": analyzer.CPU_GOLDEN_RAW_LOGIT,
        "probability": analyzer.CPU_GOLDEN_PROBABILITY,
        "ai_score": analyzer.CPU_GOLDEN_PROBABILITY,
        "classification_decision": False,
        "full_image_forward": True,
        "model_forward_calls": 1,
        "fc_hook_calls": 1,
        "repeat_feature_file_sha256": (
            _FakeRunner.CPU_GOLDEN_FEATURE_FILE_SHA256
        ),
        "repeat_feature_file_bytes": 3200,
        "repeat_feature_array_sha256": (
            _FakeRunner.CPU_GOLDEN_FEATURE_ARRAY_SHA256
        ),
        "repeat_raw_logit": analyzer.CPU_GOLDEN_RAW_LOGIT,
        "repeat_probability": analyzer.CPU_GOLDEN_PROBABILITY,
        "repeat_ai_score": analyzer.CPU_GOLDEN_PROBABILITY,
        "repeat_classification_decision": False,
        "repeat_full_image_forward": True,
        "repeat_model_forward_calls": 1,
        "repeat_fc_hook_calls": 1,
        "repeat_byte_exact": True,
    }
    report = {
        "schema_version": analyzer.EXPECTED_CPU_PREFLIGHT_SCHEMA,
        "status": "passed",
        "source": source,
        "assets": assets,
        "model_load": model_load,
        "runtime": _runtime("cpu"),
        "golden": golden,
        "cuda_used": False,
        "cuda_tensor_operations": False,
        "cuda_initialized_before_cpu_model_load": False,
        "cuda_initialized_after_cpu_forwards": False,
        "dataset_manifest_loaded": False,
    }
    analyzer._validate_cpu_preflight(
        {
            "performed_before_accelerator_configuration": True,
            "report": report,
        },
        repo_root=tmp_path,
        source=source,
        assets=assets,
    )
    for field, value in (
        ("class_module", "_claimforge_ufd_76a0e3e.models.wrong"),
        ("class_name", "WrongModel"),
    ):
        tampered = copy.deepcopy(report)
        tampered["model_load"][field] = value
        with pytest.raises(ValueError, match="safe model-load evidence"):
            analyzer._validate_cpu_preflight(
                {
                    "performed_before_accelerator_configuration": True,
                    "report": tampered,
                },
                repo_root=tmp_path,
                source=source,
                assets=assets,
            )
    for field in (
        "cuda_used",
        "cuda_tensor_operations",
        "cuda_initialized_before_cpu_model_load",
        "cuda_initialized_after_cpu_forwards",
    ):
        tampered = copy.deepcopy(report)
        tampered[field] = True
        with pytest.raises(ValueError, match="report/provenance"):
            analyzer._validate_cpu_preflight(
                {
                    "performed_before_accelerator_configuration": True,
                    "report": tampered,
                },
                repo_root=tmp_path,
                source=source,
                assets=assets,
            )


def test_analysis_runtime_adds_metric_only_scipy_and_sklearn(
    monkeypatch: pytest.MonkeyPatch,
):
    inference = _runtime("cpu")
    monkeypatch.setattr(
        analyzer,
        "_actual_runtime_contract",
        lambda device: inference if device == "cpu" else None,
    )
    monkeypatch.setattr(
        analyzer.importlib.metadata,
        "version",
        lambda name: analyzer.EXPECTED_ANALYSIS_PACKAGE_VERSIONS[name],
    )
    report = analyzer._analysis_runtime_contract()
    assert report["inference_runtime"] == inference
    assert report["analysis_packages"] == {
        "scipy": "1.17.1",
        "scikit-learn": "1.8.0",
    }


def _selection_rows(per_condition: int, *, formal: bool) -> list[dict]:
    rows = []
    rank = 0
    for condition in analyzer.BALANCED_CONDITIONS:
        count = 275 if formal and condition == "real" else per_condition
        for index in range(count):
            rows.append(
                {
                    "sample_id": f"{condition}-{index}",
                    "condition": condition,
                    "rank": rank,
                }
            )
            rank += 1
    return rows


@pytest.mark.parametrize(
    ("mode", "per_condition", "images"),
    [("smoke", 5, 35), ("formal", 250, 1775)],
)
def test_rebuild_selection_requires_exact_formal_or_smoke_coverage(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mode: str,
    per_condition: int,
    images: int,
):
    manifest_path = tmp_path / "release" / "manifest.json"
    manifest_path.parent.mkdir()
    manifest_path.write_text("{}", encoding="utf-8")
    rows = _selection_rows(per_condition, formal=mode == "formal")
    assert len(rows) == images
    raw_contract = {
        "release": {"manifest_path": "release/manifest.json"},
        "selection": {
            "spec": {
                "per_condition_limit": 5 if mode == "smoke" else None
            }
        },
    }

    class Contract:
        capability = SimpleNamespace(
            as_dict=lambda: {
                "name": "whole_image_t1",
                "conditions": list(analyzer.BALANCED_CONDITIONS),
                "valid_for_t1": True,
                "valid_for_t2": False,
            }
        )

        def as_dict(self):
            return raw_contract

    class Runner(_FakeRunner):
        @staticmethod
        def select_mode_inputs(
            _release,
            *,
            mode: str,
            per_condition_limit,
            sample_id,
        ):
            assert sample_id is None
            assert per_condition_limit == (5 if mode == "smoke" else None)
            return object(), rows

    monkeypatch.setattr(analyzer, "_runner", lambda: Runner)
    monkeypatch.setattr(
        analyzer,
        "load_canonical_release",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        analyzer,
        "build_run_dataset_contract",
        lambda *_args, **_kwargs: Contract(),
    )
    immutable = {
        "mode": mode,
        "dataset_contract": raw_contract,
        "score_spec": analyzer._score_spec().as_dict(),
        "selected_rows_sha256": analyzer._rows_sha256(rows),
        "selected_ids_sha256": analyzer.selected_ids_sha256(
            row["sample_id"] for row in rows
        ),
    }
    _release, selected, _contract = analyzer._rebuild_contract(
        repo_root=tmp_path,
        immutable=immutable,
        expected_mode=mode,
    )
    assert len(selected) == images

    rows.pop()
    immutable["selected_rows_sha256"] = analyzer._rows_sha256(rows)
    immutable["selected_ids_sha256"] = analyzer.selected_ids_sha256(
        row["sample_id"] for row in rows
    )
    with pytest.raises(ValueError, match="selection has"):
        analyzer._rebuild_contract(
            repo_root=tmp_path,
            immutable=immutable,
            expected_mode=mode,
        )


def test_pair_rank_t2_dense_joint_and_localization_are_rejected():
    analyzer._reject_unsupported_claims(
        {
            "valid_for_t2": False,
            "native_dense_output": False,
            "localization_output": None,
            "pixel_center_mapping": "diagnostic",
        },
        "allowed",
    )
    for payload in (
        {"pair_rank": 1},
        {"valid_for_t2": True},
        {"score_map": [0.1]},
        {"pixel_ap": 0.5},
        {"joint_score": 0.5},
        {"predicted_mask_path": "mask.npy"},
    ):
        with pytest.raises(ValueError, match="unsupported"):
            analyzer._reject_unsupported_claims(payload, "payload")


def test_smoke_comparison_is_bit_exact_and_checks_every_extra_field(
    tmp_path: Path,
):
    left_root, right_root = tmp_path / "left", tmp_path / "right"
    left = _row(left_root, run_id="smoke-a", fingerprint="a" * 64)
    right = _row(right_root, run_id="smoke-b", fingerprint="d" * 64)
    right["completed_at"] = "2026-07-26T01:00:00+00:00"
    right["preprocess_latency_ms"] = 20.0
    right["latency_ms"] = 30.0
    right["peak_cuda_memory_bytes"] = 99
    report = analyzer.compare_computational_results(
        reference_rows=[left],
        replay_rows=[right],
        reference_features=_inventory(left_root, left),
        replay_features=_inventory(right_root, right),
    )
    assert report["images_compared"] == 1
    assert report["feature_file_bytes_exact"] is True
    assert report["max_feature_abs_difference"] == 0.0

    right["unreviewed_computation"] = 1
    with pytest.raises(ValueError, match="projection differs"):
        analyzer.compare_computational_results(
            reference_rows=[left],
            replay_rows=[right],
            reference_features=_inventory(left_root, left),
            replay_features=_inventory(right_root, right),
        )


def test_smoke_comparison_rejects_feature_and_coverage_change(tmp_path: Path):
    left_root, right_root = tmp_path / "left", tmp_path / "right"
    left = _row(left_root, run_id="smoke-a")
    changed = np.arange(analyzer.FEATURE_DIMENSION, dtype=np.float32)
    changed[0] += 1
    right = _row(right_root, run_id="smoke-b", feature=changed)
    with pytest.raises(ValueError, match="projection differs|bytes differ"):
        analyzer.compare_computational_results(
            reference_rows=[left],
            replay_rows=[right],
            reference_features=_inventory(left_root, left),
            replay_features=_inventory(right_root, right),
        )
    with pytest.raises(ValueError, match="coverage differs"):
        analyzer.compare_computational_results(
            reference_rows=[left],
            replay_rows=[{**right, "sample_id": "other", "id": "other"}],
            reference_features=_inventory(left_root, left),
            replay_features=_inventory(right_root, right),
        )


def test_independent_linear_head_replays_every_feature(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    import torch

    checkpoint = tmp_path / "head.pth"
    checkpoint.write_bytes(b"head")
    feature = np.zeros(analyzer.FEATURE_DIMENSION, dtype=np.float32)
    row = _row(tmp_path, feature=feature, raw=0.0)
    artifacts = _inventory(tmp_path, row)
    real_sha = analyzer.sha256_file
    monkeypatch.setattr(
        analyzer,
        "sha256_file",
        lambda path: (
            analyzer.legacy.HEAD_CHECKPOINT["sha256"]
            if Path(path) == checkpoint
            else real_sha(Path(path))
        ),
    )
    monkeypatch.setattr(
        torch.serialization,
        "get_unsafe_globals_in_checkpoint",
        lambda _path: [],
    )
    configured = []

    def configure_runtime(device_text, *, seed):
        configured.append((device_text, seed))
        return (
            torch.device("cpu"),
            _runtime("cpu"),
        )

    monkeypatch.setattr(
        _FakeRunner,
        "configure_runtime",
        staticmethod(configure_runtime),
    )

    def load_head(*_args, **_kwargs):
        assert configured
        return {
            "weight": torch.zeros(
                (1, analyzer.FEATURE_DIMENSION), dtype=torch.float32
            ),
            "bias": torch.zeros((1,), dtype=torch.float32),
        }

    monkeypatch.setattr(
        torch,
        "load",
        load_head,
    )
    report = analyzer.replay_linear_head(
        latest_results=[row],
        features=artifacts,
        head_checkpoint=checkpoint,
        device_text="cpu",
        recorded_runtime=_runtime("cpu"),
    )
    assert report["features_replayed"] == 1
    assert report["device"] == "cpu"
    assert report["runtime"] == _runtime("cpu")
    assert report["recorded_runtime_exact_match"] is True
    assert report["max_raw_logit_abs_difference"] == 0.0
    assert report["raw_logit_abs_tolerance"] == 0.0
    assert report["probability_abs_tolerance"] == 0.0

    tampered = copy.deepcopy(row)
    tampered_probability = float(row["probability"]) + 5e-8
    tampered["probability"] = tampered_probability
    tampered["ai_score"] = tampered_probability
    tampered["score"] = tampered_probability
    tampered["classification_decision"] = True
    for alias in ("classification", "t1"):
        tampered[alias]["probability"] = tampered_probability
        tampered[alias]["ai_score"] = tampered_probability
        tampered[alias]["score"] = tampered_probability
        tampered[alias]["decision"] = True
    tampered["manual_replay"]["probability"] = tampered_probability
    tampered["manual_replay"]["ai_score"] = tampered_probability
    tampered["manual_replay"]["classification_decision"] = True
    analyzer._validate_score_payload(tampered, sample_id="sample")
    with pytest.raises(ValueError, match="independent head probability mismatch"):
        analyzer.replay_linear_head(
            latest_results=[tampered],
            features=artifacts,
            head_checkpoint=checkpoint,
            device_text="cpu",
            recorded_runtime=_runtime("cpu"),
        )


def test_linear_head_replay_rejects_cross_device_without_configuring(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    import torch

    def unexpected_configuration(*_args, **_kwargs):
        raise AssertionError("cross-device replay attempted configuration")

    monkeypatch.setattr(
        _FakeRunner,
        "configure_runtime",
        staticmethod(unexpected_configuration),
    )
    with pytest.raises(ValueError, match="device differs from immutable runtime"):
        analyzer.replay_linear_head(
            latest_results=[],
            features={},
            head_checkpoint=tmp_path / "unused.pth",
            device_text="cuda:0",
            recorded_runtime=_runtime("cpu"),
        )

    def drifted_configuration(device_text, *, seed):
        assert device_text == "cpu"
        assert seed == analyzer.EXPECTED_RUNTIME_SEED
        runtime = _runtime("cpu")
        runtime["platform"] = "different-platform"
        return torch.device("cpu"), runtime

    monkeypatch.setattr(
        _FakeRunner,
        "configure_runtime",
        staticmethod(drifted_configuration),
    )

    def unexpected_load(*_args, **_kwargs):
        raise AssertionError("runtime drift reached tensor loading")

    monkeypatch.setattr(torch, "load", unexpected_load)
    with pytest.raises(ValueError, match="current runtime differs"):
        analyzer.replay_linear_head(
            latest_results=[],
            features={},
            head_checkpoint=tmp_path / "unused.pth",
            device_text="cpu",
            recorded_runtime=_runtime("cpu"),
        )


def test_fresh_replay_requires_all_images_and_exact_features(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    monkeypatch.setattr(analyzer, "FORMAL_IMAGES", 2)
    runtime = {
        "device": "cpu",
        "seed": analyzer.EXPECTED_RUNTIME_SEED,
        "preprocess_profile": analyzer.CURRENT_PROFILE,
    }
    monkeypatch.setattr(
        analyzer, "_actual_runtime_contract", lambda _device: runtime
    )
    source = tmp_path / "source"
    source.mkdir()
    head = tmp_path / "head.pth"
    backbone = tmp_path / "backbone.pt"
    head.write_bytes(b"h")
    backbone.write_bytes(b"b")
    assets = {
        "bundle_sha256": "e" * 64,
        "head": {"path": str(head)},
        "backbone": {"path": str(backbone)},
    }
    selected, rows, features = [], [], {}
    for index in range(2):
        sample_id = f"sample-{index}"
        path = tmp_path / f"{sample_id}.jpg"
        path.write_bytes(b"jpeg")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        selected.append(
            {
                "sample_id": sample_id,
                "canonical_path": path.name,
                "canonical_sha256": digest,
                "width": 32,
                "height": 32,
            }
        )
        probability, decision, *_ = _classification(0.0)
        rows.append(
            {
                "sample_id": sample_id,
                "preprocess": {"geometry": analyzer._current_geometry(32, 32)},
                "raw_logit": 0.0,
                "probability": probability,
                "classification_decision": decision,
            }
        )
        array = np.full(
            analyzer.FEATURE_DIMENSION, index, dtype=np.float32
        )
        features[sample_id] = analyzer.FeatureArtifact(
            sample_id=sample_id,
            path=tmp_path / f"{sample_id}.npy",
            file_sha256="f" * 64,
            file_bytes=3200,
            array_sha256=analyzer._array_sha256(array),
            array=array,
        )
    bundle = SimpleNamespace(
        selected=tuple(selected),
        latest_results=tuple(rows),
        features=features,
        immutable={
            "runtime": runtime,
            "source": {"root": str(source)},
            "assets": assets,
        },
        release=SimpleNamespace(repo_root=tmp_path),
    )
    model = SimpleNamespace()
    device = SimpleNamespace(type="cpu")
    monkeypatch.setattr(
        analyzer.legacy,
        "load_model",
        lambda **_kwargs: (
            model,
            device,
            {"source": bundle.immutable["source"], "assets": assets},
        ),
    )
    index = {"value": 0}

    def fake_preprocess(_path, profile):
        assert profile == analyzer.CURRENT_PROFILE
        return np.zeros((3, 224, 224), np.float32), {
            "geometry": analyzer._current_geometry(32, 32)
        }

    def fake_infer(_model, _device, _image):
        current = index["value"]
        index["value"] += 1
        array = np.full(
            analyzer.FEATURE_DIMENSION, current, dtype=np.float32
        )
        probability, decision, *_ = _classification(0.0)
        return (
            {
                "raw_logit": 0.0,
                "probability": probability,
                "ai_score": probability,
                "classification_decision": decision,
                "manual_replay": {},
            },
            array,
            0,
            1.0,
        )

    monkeypatch.setattr(analyzer.legacy, "preprocess_image", fake_preprocess)
    monkeypatch.setattr(analyzer.legacy, "infer_one", fake_infer)
    report = analyzer.replay_model(
        bundle,
        source_root=source,
        head_checkpoint=head,
        backbone_checkpoint=backbone,
        device_text="cpu",
    )
    assert report["images_replayed"] == 2
    assert report["max_feature_abs_difference"] == 0.0

    changed = dict(features)
    changed["sample-1"] = copy.copy(features["sample-1"])
    changed["sample-1"] = analyzer.FeatureArtifact(
        **{
            **changed["sample-1"].__dict__,
            "array": np.zeros(analyzer.FEATURE_DIMENSION, dtype=np.float32),
        }
    )
    bundle.features = changed
    index["value"] = 0
    with pytest.raises(ValueError, match="fresh feature mismatch"):
        analyzer.replay_model(
            bundle,
            source_root=source,
            head_checkpoint=head,
            backbone_checkpoint=backbone,
            device_text="cpu",
        )


def test_output_collision_rejects_run_evidence_and_aliases(tmp_path: Path):
    manifest = tmp_path / "run" / "manifest.json"
    feature_dir = tmp_path / "features"
    with pytest.raises(ValueError, match="collision"):
        analyzer._validate_output_targets(
            {"metrics": manifest},
            protected_files=(manifest,),
            protected_dirs=(feature_dir,),
        )
    with pytest.raises(ValueError, match="distinct"):
        analyzer._validate_output_targets(
            {"metrics": tmp_path / "same.json", "audit": tmp_path / "same.json"},
            protected_files=(),
            protected_dirs=(),
        )
    with pytest.raises(ValueError, match="overwrite"):
        analyzer._validate_output_targets(
            {"audit": feature_dir / "audit.json"},
            protected_files=(),
            protected_dirs=(feature_dir,),
        )


def test_long_smoke_run_ids_use_safe_deterministic_default_output(
    tmp_path: Path,
):
    reference_run_id = "reference_" + "a" * (113 - len("reference_"))
    replay_run_id = "replay_" + "b" * (113 - len("replay_"))
    results_dir = tmp_path / "results"
    output = analyzer._resolve_smoke_comparison_output(
        requested_output=None,
        repo_root=tmp_path,
        results_dir=results_dir,
        reference_run_id=reference_run_id,
        replay_run_id=replay_run_id,
    )
    fingerprint = hashlib.sha256(
        analyzer.stable_json(
            [reference_run_id, replay_run_id]
        ).encode("utf-8")
    ).hexdigest()
    assert output == results_dir / (
        f"{analyzer.SMOKE_COMPARISON_SCHEMA_VERSION}_{fingerprint}.json"
    )
    assert len(output.name.encode("utf-8")) < 200
    assert output == analyzer._resolve_smoke_comparison_output(
        requested_output=None,
        repo_root=tmp_path,
        results_dir=results_dir,
        reference_run_id=reference_run_id,
        replay_run_id=replay_run_id,
    )
    assert output != analyzer._resolve_smoke_comparison_output(
        requested_output=None,
        repo_root=tmp_path,
        results_dir=results_dir,
        reference_run_id=replay_run_id,
        replay_run_id=reference_run_id,
    )
    analyzer._write_json_verified(
        output,
        {"status": "atomic-write-passed"},
        label="long-run-id comparison output",
    )
    assert output.is_file()

    custom = Path("reports/custom-comparison.json")
    assert analyzer._resolve_smoke_comparison_output(
        requested_output=custom,
        repo_root=tmp_path,
        results_dir=results_dir,
        reference_run_id=reference_run_id,
        replay_run_id=replay_run_id,
    ) == (tmp_path / custom).resolve()


def test_json_reader_rejects_duplicates_nonfinite_and_noncanonical(tmp_path: Path):
    path = tmp_path / "rows.jsonl"
    path.write_text('{"a":1,"a":2}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        analyzer._read_jsonl_strict(path, "rows")
    path.write_text('{"a":NaN}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="non-finite"):
        analyzer._read_jsonl_strict(path, "rows")
    path.write_text('{"a": 1}\\n', encoding="utf-8")
    with pytest.raises(ValueError):
        analyzer._read_jsonl_strict(path, "rows")
