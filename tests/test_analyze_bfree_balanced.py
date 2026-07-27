from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from PIL import Image

from eval.opensource import analyze_bfree_balanced as analyzer
from eval.opensource import run_bfree_balanced as runner
from eval.opensource.balanced_run_contract import (
    build_run_dataset_contract,
)
from eval.opensource.canonical_release import load_canonical_release
from eval.opensource.common import sha256_file, stable_json


REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_MANIFEST = (
    REPO_ROOT / "outputs" / "opensource" / "balanced250_v1" / "manifest.json"
)


@pytest.fixture(scope="module")
def release() -> Any:
    return load_canonical_release(
        REPO_ROOT,
        DATASET_MANIFEST,
        verify_files=False,
    )


def _score_fields(
    crop_logits: np.ndarray,
    *,
    raw_logit: float | None = None,
) -> dict[str, Any]:
    crops = np.ascontiguousarray(crop_logits, dtype=np.float32)
    raw = (
        analyzer._cpu_float32_crop_mean_sanity(crops)
        if raw_logit is None
        else float(raw_logit)
    )
    probability = analyzer._sigmoid_float32(raw)
    decision = raw > 0.0
    classification = {
        "raw_logit": raw,
        "ai_score": raw,
        "fake_probability": probability,
        "decision": decision,
        "threshold": 0.0,
        "threshold_operator": ">",
        "semantics": analyzer.legacy.SCORE_SEMANTICS,
    }
    t1 = {
        key: value
        for key, value in classification.items()
        if key != "semantics"
    }
    t1["policy"] = analyzer.legacy.T1_POLICY
    return {
        "raw_logit": raw,
        "ai_score": raw,
        "score": raw,
        "fake_probability": probability,
        "crop_logits": crops.tolist(),
        "score_semantics": analyzer.legacy.SCORE_SEMANTICS,
        "classification_decision": decision,
        "classification_threshold": 0.0,
        "classification_threshold_operator": ">",
        "classification": classification,
        "t1": t1,
        "manual_replay": {
            "crop_logits": crops.tolist(),
            "raw_logit": raw,
            "ai_score": raw,
            "official_crop_logits_exact_match": True,
            "official_mean_exact_match": True,
            "model_forward_calls": 1,
            "classifier_hook_calls": 1,
        },
    }


def _artifact_row(
    repo_root: Path,
    *,
    run_id: str = "run-a",
    sample_id: str = "sample",
    features: np.ndarray | None = None,
    crop_logits: np.ndarray | None = None,
) -> tuple[dict[str, Any], Path]:
    feature_array = np.ascontiguousarray(
        np.arange(5 * 768, dtype=np.float32).reshape(5, 768)
        if features is None
        else features,
        dtype=np.float32,
    )
    crop_array = np.ascontiguousarray(
        np.array([-2.0, -1.0, 0.0, 1.0, 3.0], dtype=np.float32)
        if crop_logits is None
        else crop_logits,
        dtype=np.float32,
    )
    artifact_dir = (
        repo_root
        / analyzer.DEFAULT_ARTIFACTS_DIR
        / run_id
        / "bfree_artifacts"
    )
    artifact_dir.mkdir(parents=True, exist_ok=True)
    path = artifact_dir / f"{sample_id}.npz"
    path.write_bytes(analyzer._npz_bytes(feature_array, crop_array))
    assert path.stat().st_size == analyzer.NPZ_FILE_BYTES
    relative = path.relative_to(repo_root).as_posix()
    file_sha = sha256_file(path)
    feature_sha = analyzer._array_sha256(feature_array)
    crop_sha = analyzer._array_sha256(crop_array)
    metadata = {
        "relative_path": relative,
        "sha256": file_sha,
        "file_bytes": analyzer.NPZ_FILE_BYTES,
        "feature_array_sha256": feature_sha,
        "crop_logits_array_sha256": crop_sha,
        "feature_shape": [5, 768],
        "crop_logits_shape": [5],
        "dtype": "float32",
        "feature_nbytes": analyzer.FEATURE_NBYTES,
        "crop_logits_nbytes": analyzer.CROP_LOGITS_NBYTES,
        "finite": True,
        "feature_semantics": analyzer.legacy.FEATURE_SEMANTICS,
        "crop_logits_semantics": (
            "five_official_crop_raw_logits_in_official_crop_order"
        ),
    }
    row = {
        "sample_id": sample_id,
        "status": "ok",
        **_score_fields(crop_array),
        "bfree_artifact": metadata,
        "bfree_artifact_path": relative,
        "bfree_artifact_sha256": file_sha,
        "feature_array_sha256": feature_sha,
        "crop_logits_array_sha256": crop_sha,
        "feature_shape": [5, 768],
        "feature_dtype": "float32",
        "feature_nbytes": analyzer.FEATURE_NBYTES,
        "feature_semantics": analyzer.legacy.FEATURE_SEMANTICS,
        "crop_logits_shape": [5],
        "crop_logits_dtype": "float32",
        "crop_logits_nbytes": analyzer.CROP_LOGITS_NBYTES,
        "crop_logits_semantics": (
            "five_official_crop_raw_logits_in_official_crop_order"
        ),
        "artifact_paths": {"bfree_npz": relative},
    }
    return row, artifact_dir


def _runtime() -> dict[str, Any]:
    versions = analyzer.EXPECTED_RUNTIME_VERSIONS
    prefix = analyzer.EXPECTED_FROZEN_VENV_PREFIX.as_posix()
    pycache = analyzer.EXPECTED_FROZEN_PYTHONPYCACHEPREFIX.as_posix()
    return {
        "device": "cpu",
        "python": {
            "implementation": "CPython",
            "version": versions["python"],
            "executable": (
                analyzer.EXPECTED_FROZEN_PYTHON_EXECUTABLE.as_posix()
            ),
        },
        "venv": {
            "prefix": prefix,
            "base_prefix": "/usr",
            "pyvenv_cfg_path": f"{prefix}/pyvenv.cfg",
            "pyvenv_cfg_sha256": (
                analyzer.EXPECTED_FROZEN_PYVENV_CONFIG_SHA256
            ),
            "include_system_site_packages": True,
        },
        "platform": "test-platform",
        "packages": {
            "torch": {
                "version": versions["torch"],
                "distribution_version": (
                    analyzer.EXPECTED_TORCH_DISTRIBUTION_VERSION
                ),
                "cuda_runtime": "12.8",
                "cudnn_version": None,
            },
            "torchvision": {
                "version": versions["torchvision"],
                "distribution_version": (
                    analyzer.EXPECTED_TORCHVISION_DISTRIBUTION_VERSION
                ),
            },
            **{
                key: versions[key]
                for key in (
                    "timm",
                    "transformers",
                    "safetensors",
                    "numpy",
                    "Pillow",
                    "PyYAML",
                    "scikit-learn",
                    "scipy",
                    "joblib",
                    "threadpoolctl",
                    "setuptools",
                )
            },
        },
        "seed": analyzer.EXPECTED_RUNTIME_SEED,
        "preprocess_profile": analyzer.legacy.PREPROCESS_PROFILE,
        "inference_dtype": "float32",
        "feature_dtype": "float32",
        "crop_logit_dtype": "float32",
        "batch_size": 1,
        "autocast": False,
        "grad_enabled": False,
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
        "minimum_cuda_free_bytes": runner.MINIMUM_CUDA_FREE_BYTES,
        "bytecode_writes_disabled": True,
        "process_environment": {
            "PYTHONHASHSEED": "0",
            "PYTHONDONTWRITEBYTECODE": "1",
            "NO_ALBUMENTATIONS_UPDATE": "1",
            "PYTHONPYCACHEPREFIX": pycache,
            "python_dont_write_bytecode": True,
            "sys_pycache_prefix": pycache,
            "pycache_prefix_initially_empty": True,
        },
        "legacy_runtime": {
            "device": "cpu",
            "seed": analyzer.EXPECTED_RUNTIME_SEED,
            "dtype": "float32",
            "autocast": False,
            "deterministic_algorithms": True,
            "network_allowed": False,
        },
    }


def _direct_artifact(
    path: Path,
    *,
    sample_id: str,
    features: np.ndarray,
    crop_logits: np.ndarray,
) -> analyzer.BFreeArtifact:
    path.write_bytes(analyzer._npz_bytes(features, crop_logits))
    return analyzer.BFreeArtifact(
        sample_id=sample_id,
        path=path,
        file_sha256=sha256_file(path),
        file_bytes=path.stat().st_size,
        feature_array_sha256=analyzer._array_sha256(features),
        crop_logits_array_sha256=analyzer._array_sha256(crop_logits),
        features=np.ascontiguousarray(features, dtype=np.float32),
        crop_logits=np.ascontiguousarray(crop_logits, dtype=np.float32),
    )


def _smoke_row(
    *,
    sample_id: str,
    run_id: str,
    artifact: analyzer.BFreeArtifact,
) -> dict[str, Any]:
    relative = f"outputs/{run_id}/{sample_id}.npz"
    row = {
        "sample_id": sample_id,
        **_score_fields(artifact.crop_logits),
        "run_id": run_id,
        "run_manifest_fingerprint": "a" * 64,
        "config_fingerprint": "a" * 64,
        "completed_at": "2026-07-27T00:00:00Z",
        "preprocess_latency_ms": 1.0,
        "latency_ms": 2.0,
        "peak_cuda_memory_bytes": 0,
        "valid_for_metrics": True,
        "bfree_artifact": {
            "relative_path": relative,
            "sha256": artifact.file_sha256,
            "file_bytes": artifact.file_bytes,
            "feature_array_sha256": artifact.feature_array_sha256,
            "crop_logits_array_sha256": (
                artifact.crop_logits_array_sha256
            ),
            "feature_shape": [5, 768],
            "crop_logits_shape": [5],
            "dtype": "float32",
            "feature_nbytes": analyzer.FEATURE_NBYTES,
            "crop_logits_nbytes": analyzer.CROP_LOGITS_NBYTES,
            "finite": True,
            "feature_semantics": analyzer.legacy.FEATURE_SEMANTICS,
            "crop_logits_semantics": (
                "five_official_crop_raw_logits_in_official_crop_order"
            ),
        },
    }
    return row


def test_runner_exports_and_analyzer_owned_pins_are_exact() -> None:
    assert analyzer._assert_runner_contract_exports() is runner
    assert analyzer.FORMAL_IMAGES == 1775
    assert analyzer.SMOKE_IMAGES == 35
    assert analyzer.DEFAULT_RUN_ID.endswith("_20260727")
    assert analyzer.DEFAULT_SMOKE_RUN_ID_A == (
        "bfree_dino2reg4_balanced250_v1_smoke5x7_a_r3_20260727"
    )
    assert analyzer.DEFAULT_SMOKE_RUN_ID_B == (
        "bfree_dino2reg4_balanced250_v1_smoke5x7_b_r3_20260727"
    )
    assert (
        "eval/opensource/analyze_bfree_balanced.py"
        in analyzer.EXPECTED_ADAPTER_SOURCE_PATHS
    )
    assert runner.LOCAL_VISIBILITY_CENSUS == analyzer.LOCAL_VISIBILITY_CENSUS
    assert (
        runner.FORMAL_GEOMETRY_CENSUS
        == analyzer.FORMAL_GEOMETRY_CENSUS
    )


def test_runner_export_pin_rejects_bool_for_integer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    changed = copy.deepcopy(runner.FROZEN_PREPROCESS_CONTRACT)
    changed["batch_size"] = True
    monkeypatch.setattr(runner, "FROZEN_PREPROCESS_CONTRACT", changed)
    with pytest.raises(ValueError, match="preprocess contract"):
        analyzer._assert_runner_contract_exports()


def test_same_json_type_prevents_bool_integer_aliasing() -> None:
    assert analyzer._same_json_type_and_value(1, 1)
    assert not analyzer._same_json_type_and_value(True, 1)
    assert not analyzer._same_json_type_and_value(
        {"value": True}, {"value": 1}
    )


def test_json_loader_rejects_duplicate_and_nonfinite_values() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        analyzer._json_loads('{"a":1,"a":2}', "duplicate")
    with pytest.raises(ValueError, match="non-finite"):
        analyzer._json_loads('{"a":Infinity}', "nonfinite")


def test_jsonl_reader_requires_canonical_line_and_final_newline(
    tmp_path: Path,
) -> None:
    path = tmp_path / "rows.jsonl"
    row = {"a": 1, "b": "two"}
    path.write_text(f"{stable_json(row)}\n", encoding="utf-8")
    assert analyzer._read_jsonl_strict(path, "rows") == [row]
    path.write_text('{"b":"two", "a":1}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="canonical"):
        analyzer._read_jsonl_strict(path, "rows")
    path.write_text(stable_json(row), encoding="utf-8")
    with pytest.raises(ValueError, match="final newline"):
        analyzer._read_jsonl_strict(path, "rows")


def test_safe_repo_path_rejects_traversal_and_symlink(
    tmp_path: Path,
) -> None:
    target = tmp_path / "target.bin"
    target.write_bytes(b"x")
    with pytest.raises(ValueError, match="traversing"):
        analyzer._safe_repo_path(
            "../target.bin", repo_root=tmp_path, label="input"
        )
    link = tmp_path / "link.bin"
    link.symlink_to(target)
    with pytest.raises(ValueError, match="symlink"):
        analyzer._safe_repo_path(
            "link.bin", repo_root=tmp_path, label="input"
        )


def test_t2_pair_rank_and_dense_claims_are_rejected() -> None:
    analyzer._reject_unsupported_claims(
        {
            "valid_for_t2": False,
            "native_dense_output": False,
            "localization_output": None,
            "edit_visibility_evidence": {"gt_mask_kind": "exact_diff"},
        }
    )
    for payload in (
        {"pair_rank": 1},
        {"valid_for_t2": True},
        {"localization": {"score": 0.0}},
        {"predicted_mask": None},
        {"t2": {}},
        {"joint_score": 0.0},
    ):
        with pytest.raises(ValueError, match="unsupported|must"):
            analyzer._reject_unsupported_claims(payload)


def test_score_contract_is_fp32_mean_and_strict_greater_than() -> None:
    at_threshold = _score_fields(np.zeros(5, dtype=np.float32))
    analyzer._validate_score_payload(at_threshold, sample_id="zero")
    assert at_threshold["classification_decision"] is False
    above = _score_fields(
        np.full(
            5,
            np.nextafter(np.float32(0.0), np.float32(1.0)),
            dtype=np.float32,
        )
    )
    analyzer._validate_score_payload(above, sample_id="above")
    assert above["raw_logit"] > 0.0
    assert above["classification_decision"] is True


def test_static_cross_device_mean_sanity_handles_both_reduction_cases() -> None:
    torch = pytest.importorskip("torch")
    cpu_matches_cuda = np.asarray(
        [
            -6.9796386,
            -5.8509021,
            -3.1853402,
            -3.0722127,
            -1.5688366,
        ],
        dtype=np.float32,
    )
    cpu_differs_from_cuda = np.asarray(
        [
            -3.1985316,
            -4.557191,
            -5.193743,
            -2.6756914,
            -3.2875714,
        ],
        dtype=np.float32,
    )
    cuda_before = bool(torch.cuda.is_initialized())
    matching_cpu_mean = float(
        torch.mean(
            torch.from_numpy(cpu_matches_cuda), dtype=torch.float32
        ).item()
    )
    matching_numpy_mean = float(
        cpu_matches_cuda.mean(dtype=np.float32)
    )
    assert matching_cpu_mean == -4.1313862800598145
    assert matching_numpy_mean == -4.131385803222656
    assert matching_cpu_mean != matching_numpy_mean
    assert (
        analyzer._cpu_float32_crop_mean_sanity(cpu_matches_cuda)
        == matching_cpu_mean
    )
    matching_row = _score_fields(
        cpu_matches_cuda, raw_logit=-4.1313862800598145
    )
    analyzer._validate_score_payload(
        matching_row, sample_id="cpu-matches-cuda"
    )

    differing_cpu_mean = float(
        torch.mean(
            torch.from_numpy(cpu_differs_from_cuda),
            dtype=torch.float32,
        ).item()
    )
    recorded_cuda_mean = -3.782545566558838
    assert differing_cpu_mean == -3.782545804977417
    assert differing_cpu_mean != recorded_cuda_mean
    assert abs(differing_cpu_mean - recorded_cuda_mean) == (
        2.384185791015625e-07
    )
    differing_row = _score_fields(
        cpu_differs_from_cuda,
        raw_logit=recorded_cuda_mean,
    )
    analyzer._validate_score_payload(
        differing_row, sample_id="cpu-differs-from-cuda"
    )
    assert bool(torch.cuda.is_initialized()) is cuda_before

    excessive = differing_cpu_mean + (
        2.0 * analyzer.CROSS_DEVICE_MEAN_ABS_TOLERANCE
    )
    excessive_row = {
        "raw_logit": excessive,
        "ai_score": excessive,
        "score": excessive,
        "crop_logits": cpu_differs_from_cuda.tolist(),
    }
    with pytest.raises(ValueError, match="cross-device"):
        analyzer._validate_score_payload(
            excessive_row, sample_id="excessive-static-drift"
        )


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("score",), 123.0, "aliases"),
        (("fake_probability",), 0.25, "probability"),
        (("classification_threshold",), False, "policy"),
        (
            ("manual_replay", "official_mean_exact_match"),
            1,
            "manual",
        ),
    ],
)
def test_score_contract_rejects_tampering(
    path: tuple[str, ...],
    value: Any,
    message: str,
) -> None:
    row = _score_fields(np.arange(5, dtype=np.float32))
    target: dict[str, Any] = row
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ValueError, match=message):
        analyzer._validate_score_payload(row, sample_id="sample")


def test_formal_and_smoke_selection_are_independently_rebuilt(
    release: Any,
) -> None:
    formal_spec, formal = analyzer._formal_selection(release)
    smoke_spec, smoke = analyzer._smoke_selection(release)
    assert formal_spec.capability.value == "whole_image_t1"
    assert len(formal) == analyzer.FORMAL_IMAGES
    assert analyzer._rows_sha256(formal) == (
        analyzer.FORMAL_SELECTED_ROWS_SHA256
    )
    assert len(smoke) == analyzer.SMOKE_IMAGES
    assert smoke_spec.per_condition_limit == analyzer.SMOKE_PER_CONDITION
    assert all("pair_rank" not in row for row in (*formal, *smoke))


@pytest.mark.parametrize("mode", ["formal", "smoke"])
def test_rebuild_contract_accepts_real_shared_v2_serialization(
    release: Any,
    mode: str,
) -> None:
    if mode == "formal":
        spec, selected = analyzer._formal_selection(release)
    else:
        spec, selected = analyzer._smoke_selection(release)
    serialized = build_run_dataset_contract(
        release,
        spec,
        selected,
        score_spec=analyzer._score_spec(),
    ).as_dict()
    assert serialized["schema_version"] == (
        analyzer.EXPECTED_RUN_DATASET_CONTRACT_SCHEMA
    )
    assert serialized["release"]["manifest_path"] == (
        "outputs/opensource/balanced250_v1/manifest.json"
    )
    rebuilt_release, rebuilt_selected, rebuilt_contract = (
        analyzer._rebuild_contract(
            repo_root=REPO_ROOT,
            immutable={"dataset_contract": serialized},
            expected_mode=mode,
        )
    )
    assert rebuilt_release.manifest_sha256 == release.manifest_sha256
    assert rebuilt_selected == selected
    assert rebuilt_contract.as_dict() == serialized


def test_rebuild_contract_strictly_rejects_wrong_v2_shape(
    release: Any,
) -> None:
    spec, selected = analyzer._smoke_selection(release)
    serialized = build_run_dataset_contract(
        release,
        spec,
        selected,
        score_spec=analyzer._score_spec(),
    ).as_dict()
    flat = copy.deepcopy(serialized)
    flat["dataset_manifest"] = flat["release"]["manifest_path"]
    del flat["release"]
    with pytest.raises(ValueError, match="v2 key set"):
        analyzer._rebuild_contract(
            repo_root=REPO_ROOT,
            immutable={"dataset_contract": flat},
            expected_mode="smoke",
        )
    for mutation, message in (
        ({"extra": True}, "release binding key set"),
        ({"manifest_path": None}, "safe string"),
        ({"schema_version": "wrong"}, "release identity"),
    ):
        changed = copy.deepcopy(serialized)
        changed["release"].update(mutation)
        with pytest.raises(ValueError, match=message):
            analyzer._rebuild_contract(
                repo_root=REPO_ROOT,
                immutable={"dataset_contract": changed},
                expected_mode="smoke",
            )


def test_full_formal_visibility_and_geometry_census_are_frozen(
    release: Any,
) -> None:
    _spec, selected = analyzer._formal_selection(release)
    census = analyzer._independent_visibility_census(
        selected, repo_root=REPO_ROOT
    )
    assert census["by_condition"] == {
        key: analyzer.LOCAL_VISIBILITY_CENSUS[key]
        for key in ("local_mouse", "local_cat", "local_trash_can")
    }
    assert census["all_local"] == analyzer.LOCAL_VISIBILITY_CENSUS[
        "all_local"
    ]
    assert census["replicate_wrap_by_condition"] == (
        analyzer.FORMAL_GEOMETRY_CENSUS["replicate_wrap_by_condition"]
    )
    assert census["replicate_wrap_total"] == 165
    assert census["distinct_crop_starts_all"] == {
        "1": 165,
        "3": 7,
        "5": 1603,
    }
    assert census["not_applicable_images"] == 1025


def test_local_visibility_uses_exact_mask_not_model_output(
    tmp_path: Path,
) -> None:
    mask = np.zeros((440, 440), dtype=np.uint8)
    mask[0, 0] = 255
    mask[-1, -1] = 255
    path = tmp_path / "mask.png"
    Image.fromarray(mask, mode="L").save(path)
    row = {
        "sample_id": "local",
        "condition": "local_mouse",
        "kind": "forged",
        "width": 440,
        "height": 440,
        "gt_mask_path": "mask.png",
        "gt_mask_sha256": sha256_file(path),
        "gt_positive_pixels": 2,
    }
    diagnostic = analyzer._independent_visibility_diagnostic(
        row, repo_root=tmp_path
    )
    assert diagnostic["edit_visibility"] == "partial"
    assert diagnostic["edit_visible_gt_fraction"] == 0.5
    assert diagnostic["edit_visibility_evidence"]["role"] == (
        "input_condition_stratum_not_model_localization"
    )


def test_npz_artifact_and_inventory_require_exact_canonical_contract(
    tmp_path: Path,
) -> None:
    row, artifact_dir = _artifact_row(tmp_path)
    artifacts = analyzer.validate_artifact_inventory(
        latest_results=[row],
        repo_root=tmp_path,
        artifact_dir=artifact_dir,
        run_id="run-a",
    )
    artifact = artifacts["sample"]
    assert artifact.features.shape == analyzer.FEATURE_SHAPE
    assert artifact.crop_logits.shape == analyzer.CROP_LOGITS_SHAPE
    assert artifact.file_bytes == analyzer.NPZ_FILE_BYTES


def test_npz_artifact_rejects_metadata_and_byte_tampering(
    tmp_path: Path,
) -> None:
    row, artifact_dir = _artifact_row(tmp_path)
    changed = copy.deepcopy(row)
    changed["bfree_artifact"]["finite"] = 1
    with pytest.raises(ValueError, match="metadata"):
        analyzer._load_npz_artifact(
            changed,
            sample_id="sample",
            repo_root=tmp_path,
            run_id="run-a",
        )
    path = artifact_dir / "sample.npz"
    path.write_bytes(path.read_bytes() + b"x")
    row["bfree_artifact"]["sha256"] = sha256_file(path)
    row["bfree_artifact_sha256"] = sha256_file(path)
    with pytest.raises(ValueError, match="byte size"):
        analyzer._load_npz_artifact(
            row,
            sample_id="sample",
            repo_root=tmp_path,
            run_id="run-a",
        )


def test_npz_inventory_rejects_orphan_and_symlink_entries(
    tmp_path: Path,
) -> None:
    row, artifact_dir = _artifact_row(tmp_path)
    (artifact_dir / "orphan.npz").write_bytes(b"orphan")
    with pytest.raises(ValueError, match="coverage"):
        analyzer.validate_artifact_inventory(
            latest_results=[row],
            repo_root=tmp_path,
            artifact_dir=artifact_dir,
            run_id="run-a",
        )
    (artifact_dir / "orphan.npz").unlink()
    link = artifact_dir / "link.npz"
    link.symlink_to(artifact_dir / "sample.npz")
    with pytest.raises(ValueError, match="extra"):
        analyzer.validate_artifact_inventory(
            latest_results=[row],
            repo_root=tmp_path,
            artifact_dir=artifact_dir,
            run_id="run-a",
        )


def test_runtime_contract_pins_environment_and_module_files() -> None:
    runtime = _runtime()
    assert analyzer._validate_runtime_contract(
        runtime, label="runtime"
    ) == runtime
    evidence = analyzer._validate_frozen_module_hashes()
    assert set(evidence) == set(analyzer.EXPECTED_MODULE_HASHES)


@pytest.mark.parametrize(
    ("outer", "inner", "value"),
    [
        ("process_environment", "PYTHONHASHSEED", "20260725"),
        ("process_environment", "NO_ALBUMENTATIONS_UPDATE", "0"),
        ("cudnn", "enabled", 1),
        ("root", "batch_size", True),
    ],
)
def test_runtime_contract_rejects_environment_and_bool_int_drift(
    outer: str,
    inner: str,
    value: Any,
) -> None:
    runtime = _runtime()
    if outer == "root":
        runtime[inner] = value
    else:
        runtime[outer][inner] = value
    with pytest.raises(ValueError):
        analyzer._validate_runtime_contract(runtime, label="runtime")


def test_persisted_linear_head_replay_is_exact_and_cpu_only(
    tmp_path: Path,
) -> None:
    torch = pytest.importorskip("torch")

    class FakeModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = torch.nn.Module()
            self.model.head = torch.nn.Linear(768, 1)

    model = FakeModel().eval()
    with torch.no_grad():
        model.model.head.weight.zero_()
        model.model.head.weight[0, 0] = 2.0
        model.model.head.bias.fill_(-0.5)
    features = np.zeros((5, 768), dtype=np.float32)
    features[:, 0] = np.array([-2, -1, 0, 1, 2], dtype=np.float32)
    with torch.inference_mode():
        crop_logits = (
            model.model.head(torch.from_numpy(features))
            .reshape(5)
            .numpy()
            .astype(np.float32)
        )
    artifact = _direct_artifact(
        tmp_path / "sample.npz",
        sample_id="sample",
        features=features,
        crop_logits=crop_logits,
    )
    row = {"sample_id": "sample", **_score_fields(crop_logits)}
    report = analyzer.replay_persisted_head(
        latest_results=[row],
        artifacts={"sample": artifact},
        model=model,
        device=torch.device("cpu"),
    )
    assert report["linear_head_replays"] == 1
    assert report["maximum_crop_logit_absolute_difference"] == 0.0
    assert report["maximum_raw_logit_absolute_difference"] == 0.0
    assert report["crop_logit_absolute_tolerance"] == 0.0
    assert report["raw_logit_absolute_tolerance"] == 0.0
    one_ulp_raw = float(
        np.nextafter(
            np.float32(row["raw_logit"]),
            np.float32(np.inf),
        )
    )
    assert (
        abs(one_ulp_raw - row["raw_logit"])
        < analyzer.CROSS_DEVICE_MEAN_ABS_TOLERANCE
    )
    one_ulp_row = {
        "sample_id": "sample",
        **_score_fields(crop_logits, raw_logit=one_ulp_raw),
    }
    analyzer._validate_score_payload(
        one_ulp_row, sample_id="static-one-ulp"
    )
    with pytest.raises(ValueError, match="persisted head replay"):
        analyzer.replay_persisted_head(
            latest_results=[one_ulp_row],
            artifacts={"sample": artifact},
            model=model,
            device=torch.device("cpu"),
        )


def test_smoke_comparison_is_byte_exact_and_type_exact(
    tmp_path: Path,
) -> None:
    features = np.arange(5 * 768, dtype=np.float32).reshape(5, 768)
    crops = np.arange(5, dtype=np.float32)
    left = _direct_artifact(
        tmp_path / "left.npz",
        sample_id="sample",
        features=features,
        crop_logits=crops,
    )
    right = _direct_artifact(
        tmp_path / "right.npz",
        sample_id="sample",
        features=features.copy(),
        crop_logits=crops.copy(),
    )
    left_row = _smoke_row(
        sample_id="sample", run_id="run-left", artifact=left
    )
    right_row = _smoke_row(
        sample_id="sample", run_id="run-right", artifact=right
    )
    report = analyzer.compare_computational_results(
        reference_rows=[left_row],
        replay_rows=[right_row],
        reference_artifacts={"sample": left},
        replay_artifacts={"sample": right},
    )
    assert report["images_compared"] == 1
    changed = copy.deepcopy(right_row)
    changed["valid_for_metrics"] = 1
    with pytest.raises(ValueError, match="result differs"):
        analyzer.compare_computational_results(
            reference_rows=[left_row],
            replay_rows=[changed],
            reference_artifacts={"sample": left},
            replay_artifacts={"sample": right},
        )


def test_output_collision_guards_evidence_and_artifact_directories(
    tmp_path: Path,
) -> None:
    evidence = tmp_path / "manifest.json"
    evidence.write_text("{}\n", encoding="utf-8")
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    with pytest.raises(ValueError, match="overwrite evidence"):
        analyzer._validate_output_targets(
            {"audit": evidence},
            protected_files=[evidence],
            protected_dirs=[artifacts],
        )
    with pytest.raises(ValueError, match="overwrite evidence"):
        analyzer._validate_output_targets(
            {"audit": artifacts / "audit.json"},
            protected_files=[evidence],
            protected_dirs=[artifacts],
        )
    with pytest.raises(ValueError, match="collide"):
        analyzer._validate_output_targets(
            {
                "audit": tmp_path / "report.json",
                "metrics": tmp_path / "report.json",
            },
            protected_files=[evidence],
            protected_dirs=[artifacts],
        )


def test_formal_outputs_are_limited_to_current_run_canonical_files(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path
    run_dir = (
        repo_root
        / "results"
        / "opensource"
        / "bfree"
        / "current-run"
    )
    run_dir.mkdir(parents=True)
    metrics = run_dir / "balanced250_metrics.json"
    audit = run_dir / "independent_audit.json"
    authorized = analyzer._validate_formal_output_scope(
        repo_root=repo_root,
        run_dir=run_dir,
        metrics_output_path=metrics,
        audit_output_path=audit,
    )
    assert authorized == {"metrics": metrics, "audit": audit}
    with pytest.raises(ValueError, match="canonical current-run"):
        analyzer._validate_formal_output_scope(
            repo_root=repo_root,
            run_dir=run_dir,
            metrics_output_path=audit,
            audit_output_path=metrics,
        )
    other_run = run_dir.parent / "other-run"
    other_run.mkdir()
    for name, candidate in (
        ("metrics", run_dir / "custom_metrics.json"),
        ("metrics", other_run / "manifest.json"),
        ("audit", repo_root / "external_audit.json"),
    ):
        kwargs = {
            "metrics_output_path": metrics,
            "audit_output_path": audit,
        }
        kwargs[f"{name}_output_path"] = candidate
        with pytest.raises(ValueError, match="canonical current-run"):
            analyzer._validate_formal_output_scope(
                repo_root=repo_root,
                run_dir=run_dir,
                **kwargs,
            )


def test_formal_output_scope_rejects_final_and_parent_symlinks(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "results" / "current-run"
    run_dir.mkdir(parents=True)
    outside = tmp_path / "outside.json"
    outside.write_text("{}\n", encoding="utf-8")
    metrics = run_dir / "balanced250_metrics.json"
    metrics.symlink_to(outside)
    with pytest.raises(ValueError, match="symlink"):
        analyzer._validate_formal_output_scope(
            repo_root=tmp_path,
            run_dir=run_dir,
            metrics_output_path=metrics,
            audit_output_path=None,
        )
    metrics.unlink()
    real_run = tmp_path / "real-run"
    real_run.mkdir()
    linked_run = tmp_path / "linked-run"
    linked_run.symlink_to(real_run, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        analyzer._validate_formal_output_scope(
            repo_root=tmp_path,
            run_dir=linked_run,
            metrics_output_path=(
                linked_run / "balanced250_metrics.json"
            ),
            audit_output_path=None,
        )


def test_smoke_output_scope_allows_only_default_or_reports_tree(
    tmp_path: Path,
) -> None:
    results = tmp_path / "results" / "opensource" / "bfree"
    results.mkdir(parents=True)
    default = analyzer._resolve_smoke_comparison_output(
        requested_output=None,
        results_dir=results,
        reference_run_id="smoke-a",
        replay_run_id="smoke-b",
    )
    assert default.parent == results / "_reports"
    assert default.name.startswith(
        f"{analyzer.SMOKE_COMPARISON_SCHEMA_VERSION}_"
    )
    report = results / "_reports" / "review" / "comparison.json"
    assert analyzer._resolve_smoke_comparison_output(
        requested_output=report,
        results_dir=results,
        reference_run_id="smoke-a",
        replay_run_id="smoke-b",
    ) == report
    for candidate in (
        results / "other-run" / "manifest.json",
        results / "arbitrary.json",
        tmp_path / "external.json",
    ):
        with pytest.raises(ValueError, match="canonical default|_reports"):
            analyzer._resolve_smoke_comparison_output(
                requested_output=candidate,
                results_dir=results,
                reference_run_id="smoke-a",
                replay_run_id="smoke-b",
            )


def test_smoke_output_scope_rejects_symlink_components(
    tmp_path: Path,
) -> None:
    results = tmp_path / "results"
    results.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    reports = results / "_reports"
    reports.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        analyzer._resolve_smoke_comparison_output(
            requested_output=reports / "comparison.json",
            results_dir=results,
            reference_run_id="smoke-a",
            replay_run_id="smoke-b",
        )
    reports.unlink()
    reports.mkdir()
    outside_file = outside / "comparison.json"
    outside_file.write_text("{}\n", encoding="utf-8")
    linked_file = reports / "comparison.json"
    linked_file.symlink_to(outside_file)
    with pytest.raises(ValueError, match="symlink"):
        analyzer._resolve_smoke_comparison_output(
            requested_output=linked_file,
            results_dir=results,
            reference_run_id="smoke-a",
            replay_run_id="smoke-b",
        )


def test_before_after_evidence_hash_detects_tampering(
    tmp_path: Path,
) -> None:
    paths = {}
    for name in ("manifest", "results", "expected", "summary"):
        path = tmp_path / f"{name}.json"
        path.write_text(f"{name}\n", encoding="utf-8")
        paths[name] = path
    bundle = SimpleNamespace(
        manifest_path=paths["manifest"],
        results_path=paths["results"],
        expected_path=paths["expected"],
        summary_path=paths["summary"],
        evidence_snapshot={
            "manifest_sha256": sha256_file(paths["manifest"]),
            "results_sha256": sha256_file(paths["results"]),
            "expected_inputs_sha256": sha256_file(paths["expected"]),
            "summary_sha256": sha256_file(paths["summary"]),
        },
    )
    paths["manifest"].write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="changed after validation"):
        analyzer._verify_bundle_unchanged(bundle, repo_root=tmp_path)


def test_shared_metrics_call_is_frozen_to_1000_and_20260726(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, Any] = {}

    def fake_summary(*args: Any, **kwargs: Any) -> dict[str, Any]:
        observed["args"] = args
        observed["kwargs"] = kwargs
        return {
            "schema_version": analyzer.METRICS_SCHEMA_VERSION,
            "coverage": {"is_complete": True},
        }

    monkeypatch.setattr(
        analyzer, "summarize_balanced250_t1", fake_summary
    )
    contract = SimpleNamespace(as_dict=lambda: {"contract": "exact"})
    bundle = SimpleNamespace(
        release=SimpleNamespace(
            inputs=("input",),
            panel=("panel",),
            source_pairs=("pair",),
        ),
        latest_results=("result",),
        run_id="run-a",
        fingerprint="f" * 64,
        contract=contract,
    )
    result = analyzer.recompute_metrics(bundle)
    assert result["coverage"]["is_complete"] is True
    assert observed["kwargs"]["iterations"] == 1000
    assert observed["kwargs"]["seed"] == 20260726
    with pytest.raises(ValueError, match="iterations=1000"):
        analyzer.recompute_metrics(bundle, iterations=999)


def test_cli_defaults_to_full_fresh_1775_replay() -> None:
    args = analyzer._build_parser().parse_args([])
    assert args.run_id == analyzer.DEFAULT_RUN_ID
    assert args.skip_model_replay is False
    assert args.bootstrap_iterations == 1000
    assert args.bootstrap_seed == 20260726
    incomplete = SimpleNamespace(mode="formal", selected=(object(),))
    with pytest.raises(ValueError, match="full 1,775"):
        analyzer.replay_model(
            incomplete,
            repo_root=REPO_ROOT,
            source_root=analyzer.DEFAULT_SOURCE_ROOT,
            model=object(),
            device=object(),
        )


def test_npz_serialization_is_stable_and_has_exact_member_order() -> None:
    features = np.arange(5 * 768, dtype=np.float32).reshape(5, 768)
    crops = np.arange(5, dtype=np.float32)
    left = analyzer._npz_bytes(features, crops)
    right = analyzer._npz_bytes(features.copy(), crops.copy())
    assert left == right
    assert len(left) == analyzer.NPZ_FILE_BYTES
    assert hashlib.sha256(left).hexdigest() == hashlib.sha256(right).hexdigest()
