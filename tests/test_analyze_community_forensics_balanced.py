from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from PIL import Image

from eval.opensource import analyze_community_forensics_balanced as analyzer
from eval.opensource import run_community_forensics_balanced as runner
from eval.opensource.common import sha256_file, stable_json


def _score_fields(
    *,
    sample_id: str = "sample",
    raw_logit: float = 0.0,
    probability: float = 0.5,
) -> dict[str, Any]:
    decision = bool(
        probability > analyzer.legacy.CLASSIFICATION_THRESHOLD
    )
    classification = {
        "raw_logit": raw_logit,
        "probability": probability,
        "ai_score": probability,
        "score": probability,
        "decision": decision,
        "threshold": analyzer.legacy.CLASSIFICATION_THRESHOLD,
        "threshold_operator": (
            analyzer.legacy.CLASSIFICATION_THRESHOLD_OPERATOR
        ),
        "semantics": analyzer.legacy.SCORE_SEMANTICS,
    }
    t1 = {
        key: value
        for key, value in classification.items()
        if key != "semantics"
    }
    t1["policy"] = analyzer.legacy.T1_POLICY
    return {
        "sample_id": sample_id,
        "raw_logit": raw_logit,
        "probability": probability,
        "ai_score": probability,
        "score": probability,
        "score_semantics": analyzer.legacy.SCORE_SEMANTICS,
        "classification_decision": decision,
        "classification_threshold": (
            analyzer.legacy.CLASSIFICATION_THRESHOLD
        ),
        "classification_threshold_operator": (
            analyzer.legacy.CLASSIFICATION_THRESHOLD_OPERATOR
        ),
        "classification": classification,
        "t1": t1,
        "manual_replay": {
            "raw_logit": raw_logit,
            "probability": probability,
            "ai_score": probability,
            "classification_decision": decision,
            "official_logit_exact_match": True,
            "official_probability_exact_match": True,
            "model_forward_calls": 1,
            "classifier_hook_calls": 1,
        },
    }


def _feature_row(
    repo_root: Path,
    *,
    run_id: str = "run-a",
    sample_id: str = "sample",
    array: np.ndarray | None = None,
) -> tuple[dict[str, Any], Path]:
    value = np.ascontiguousarray(
        np.arange(analyzer.FEATURE_DIMENSION, dtype=np.float32)
        if array is None
        else array
    )
    feature_dir = (
        repo_root
        / "outputs"
        / "opensource"
        / "community_forensics"
        / run_id
        / "commfor_features"
    )
    feature_dir.mkdir(parents=True, exist_ok=True)
    path = feature_dir / f"{sample_id}.npy"
    np.save(path, value, allow_pickle=False)
    relative = path.relative_to(repo_root).as_posix()
    file_sha = sha256_file(path)
    array_sha = analyzer._array_sha256(value)
    row = {
        **_score_fields(sample_id=sample_id),
        "commfor_feature": {
            "relative_path": relative,
            "sha256": file_sha,
            "file_bytes": path.stat().st_size,
            "array_sha256": array_sha,
            "dtype": "float32",
            "shape": [analyzer.FEATURE_DIMENSION],
            "nbytes": analyzer.FEATURE_NBYTES,
            "finite": True,
            "semantics": analyzer.FEATURE_SEMANTICS,
        },
        "commfor_feature_path": relative,
        "commfor_feature_sha256": file_sha,
        "commfor_feature_array_sha256": array_sha,
        "commfor_feature_shape": [analyzer.FEATURE_DIMENSION],
        "commfor_feature_dtype": "float32",
        "commfor_feature_nbytes": analyzer.FEATURE_NBYTES,
        "commfor_feature_semantics": analyzer.FEATURE_SEMANTICS,
        "artifact_paths": {"commfor_feature_npy": relative},
    }
    return row, feature_dir


def _runtime() -> dict[str, Any]:
    prefix = analyzer.EXPECTED_FROZEN_VENV_PREFIX.as_posix()
    pycache = analyzer.EXPECTED_FROZEN_PYTHONPYCACHEPREFIX.as_posix()
    versions = analyzer.EXPECTED_FROZEN_RUNTIME_VERSIONS
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
                "distribution_version": versions["torch_distribution"],
                "cuda_runtime": "12.8",
                "cudnn_version": None,
            },
            "torchvision": {
                "version": versions["torchvision"],
                "distribution_version": (
                    versions["torchvision_distribution"]
                ),
            },
            **{
                key: versions[key]
                for key in (
                    "timm",
                    "safetensors",
                    "numpy",
                    "Pillow",
                    "scikit-learn",
                    "scipy",
                    "joblib",
                    "threadpoolctl",
                    "setuptools",
                )
            },
        },
        "seed": analyzer.EXPECTED_RUNTIME_SEED,
        "preprocess_profile": analyzer.FROZEN_PROFILE,
        "inference_dtype": "float32",
        "feature_dtype": "float32",
        "batch_size": 1,
        "autocast": False,
        "grad_enabled": False,
        "deterministic_algorithms_enabled": True,
        "deterministic_algorithms_warn_only": False,
        "cublas_workspace_config": (
            analyzer.EXPECTED_CUBLAS_WORKSPACE_CONFIG
        ),
        "cudnn": {
            "enabled": False,
            "benchmark": False,
            "deterministic": True,
            "allow_tf32": False,
        },
        "matmul_allow_tf32": False,
        "float32_matmul_precision": "highest",
        "minimum_cuda_free_bytes": (
            analyzer.EXPECTED_MINIMUM_CUDA_FREE_BYTES
        ),
        "bytecode_writes_disabled": True,
        "process_environment": {
            "PYTHONHASHSEED": str(analyzer.EXPECTED_RUNTIME_SEED),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPYCACHEPREFIX": pycache,
            "python_dont_write_bytecode": True,
            "sys_pycache_prefix": pycache,
            "pycache_prefix_initially_empty": True,
        },
    }


def _write_mask(repo_root: Path) -> tuple[str, str]:
    mask = np.zeros((440, 440), dtype=np.uint8)
    mask[0, 0] = 255
    mask[200, 200] = 255
    path = repo_root / "masks" / "partial.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask, mode="L").save(path)
    return path.relative_to(repo_root).as_posix(), sha256_file(path)


def _local_row(
    repo_root: Path,
    *,
    condition: str,
    sample_id: str,
) -> dict[str, Any]:
    relative, digest = _write_mask(repo_root)
    return {
        "sample_id": sample_id,
        "condition": condition,
        "width": 440,
        "height": 440,
        "gt_mask_kind": "exact_diff",
        "gt_mask_path": relative,
        "gt_mask_sha256": digest,
        "gt_positive_pixels": 2,
        "edit_region_xyxy": [0, 0, 440, 440],
    }


def test_runner_contract_exports_match_independent_pins() -> None:
    assert analyzer._assert_runner_contract_exports() is runner
    assert analyzer.EXPECTED_PREPROCESS_CONTRACT == runner.PREPROCESS_CONTRACT
    assert analyzer.EXPECTED_MODEL_CONTRACT == runner.MODEL_CONTRACT
    assert analyzer.EXPECTED_ARTIFACT_CONTRACT == runner.ARTIFACT_CONTRACT
    assert analyzer.EXPECTED_TASK_SCOPE == runner.TASK_SCOPE


def test_json_loader_rejects_duplicate_and_nonfinite_values() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        analyzer._json_loads('{"a":1,"a":2}', "duplicate")
    with pytest.raises(ValueError, match="non-finite"):
        analyzer._json_loads('{"a":NaN}', "nonfinite")


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


def test_safe_repo_path_rejects_traversal_and_symlink(tmp_path: Path) -> None:
    target = tmp_path / "target.bin"
    target.write_bytes(b"x")
    with pytest.raises(ValueError, match="traversing"):
        analyzer._safe_repo_path(
            "../target.bin",
            repo_root=tmp_path,
            label="input",
        )
    link = tmp_path / "link.bin"
    link.symlink_to(target)
    with pytest.raises(ValueError, match="symlink"):
        analyzer._safe_repo_path(
            "link.bin",
            repo_root=tmp_path,
            label="input",
        )


def test_t2_and_localization_claims_are_rejected() -> None:
    analyzer._reject_unsupported_claims(
        {
            "task_scope": {
                "valid_for_t2": False,
                "native_dense_output": False,
                "localization_output": None,
            },
            "edit_visibility_evidence": {
                "pixel_center_mapping": "diagnostic only",
            },
        },
        "payload",
    )
    with pytest.raises(ValueError, match="unsupported"):
        analyzer._reject_unsupported_claims(
            {"task_scope": {"valid_for_t2": True}},
            "payload",
        )
    with pytest.raises(ValueError, match="unsupported"):
        analyzer._reject_unsupported_claims(
            {"localization": {"score": 1.0}},
            "payload",
        )


def test_score_validation_uses_strict_greater_than_threshold() -> None:
    at_threshold = _score_fields(probability=0.5)
    analyzer._validate_score_payload(at_threshold, sample_id="sample")
    assert at_threshold["classification_decision"] is False
    above = _score_fields(probability=np.nextafter(0.5, 1.0))
    analyzer._validate_score_payload(above, sample_id="sample")
    assert above["classification_decision"] is True


def test_score_validation_does_not_recompute_sigmoid_on_host(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    row = _score_fields(raw_logit=123.0, probability=0.25)
    monkeypatch.setattr(
        analyzer.math,
        "exp",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("host sigmoid was called")
        ),
    )
    analyzer._validate_score_payload(row, sample_id="sample")


def test_score_validation_rejects_alias_or_manual_replay_drift() -> None:
    row = _score_fields(probability=0.75)
    row["score"] = 0.5
    with pytest.raises(ValueError, match="aliases"):
        analyzer._validate_score_payload(row, sample_id="sample")
    row = _score_fields(probability=0.75)
    row["manual_replay"]["classifier_hook_calls"] = 2
    with pytest.raises(ValueError, match="manual"):
        analyzer._validate_score_payload(row, sample_id="sample")


def test_geometry_matches_released_resize_and_center_crop() -> None:
    wide = analyzer._current_geometry(1800, 1350)
    assert wide["profile_id"] == analyzer.FROZEN_PROFILE
    assert wide["resize"]["destination_size"] == [586, 440]
    assert wide["center_crop"]["start_xy"] == [101, 28]
    tall = analyzer._current_geometry(100, 400)
    assert tall["resize"]["destination_size"] == [440, 1760]
    assert tall["center_crop"]["start_xy"] == [28, 688]


def test_visibility_is_not_applicable_for_real_and_fullframe(
    tmp_path: Path,
) -> None:
    common = {"width": 440, "height": 440}
    real = analyzer._independent_visibility_diagnostic(
        {**common, "gt_mask_kind": "all_zero"},
        repo_root=tmp_path,
    )
    generated = analyzer._independent_visibility_diagnostic(
        {**common, "gt_mask_kind": "not_applicable"},
        repo_root=tmp_path,
    )
    assert real["edit_visibility"] == "not_applicable"
    assert real["edit_visible_gt_fraction"] is None
    assert generated["edit_visibility"] == "not_applicable"
    assert generated["edit_visible_gt_fraction"] is None


def test_visibility_maps_exact_mask_pixel_centers(tmp_path: Path) -> None:
    row = _local_row(
        tmp_path,
        condition="local_mouse",
        sample_id="local-a",
    )
    diagnostic = analyzer._independent_visibility_diagnostic(
        row,
        repo_root=tmp_path,
    )
    assert diagnostic["edit_visibility"] == "partial"
    assert diagnostic["edit_visible_gt_fraction"] == 0.5
    gt = diagnostic["edit_visibility_evidence"]["gt"]
    assert gt["positive_pixels"] == 2
    assert gt["visible_positive_pixel_centers"] == 1
    assert (
        "not_model_localization"
        not in diagnostic["edit_visibility_evidence"]
    )


def test_visibility_rejects_tampered_mask_hash(tmp_path: Path) -> None:
    row = _local_row(
        tmp_path,
        condition="local_cat",
        sample_id="local-b",
    )
    row["gt_mask_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="hash"):
        analyzer._independent_visibility_diagnostic(
            row,
            repo_root=tmp_path,
        )


def test_selection_visibility_census_is_input_diagnostic(
    tmp_path: Path,
) -> None:
    selected = [
        _local_row(
            tmp_path,
            condition=condition,
            sample_id=f"{condition}-a",
        )
        for condition in (
            "local_mouse",
            "local_cat",
            "local_trash_can",
        )
    ]
    selected.extend(
        [
            {
                "condition": "real",
                "width": 440,
                "height": 440,
                "gt_mask_kind": "all_zero",
            },
            {
                "condition": "fullframe_mouse",
                "width": 440,
                "height": 440,
                "gt_mask_kind": "not_applicable",
            },
        ]
    )
    census = analyzer._independent_selection_visibility_census(
        selected,
        repo_root=tmp_path,
    )
    assert census["role"] == (
        "input_condition_diagnostic_not_model_localization"
    )
    assert census["all_local"] == {
        "full": 0,
        "partial": 3,
        "none": 0,
        "total": 3,
        "mean_edit_visible_gt_fraction": 0.5,
    }
    assert census["not_applicable_images"] == 2


def test_feature_inventory_accepts_only_canonical_float32_npy(
    tmp_path: Path,
) -> None:
    row, feature_dir = _feature_row(tmp_path)
    artifacts = analyzer.validate_feature_inventory(
        latest_results=[row],
        repo_root=tmp_path,
        feature_dir=feature_dir,
    )
    artifact = artifacts["sample"]
    assert artifact.array.shape == (analyzer.FEATURE_DIMENSION,)
    assert artifact.array.dtype == np.float32
    assert artifact.file_bytes == len(analyzer._npy_bytes(artifact.array))


def test_feature_inventory_rejects_extra_file(tmp_path: Path) -> None:
    row, feature_dir = _feature_row(tmp_path)
    (feature_dir / "extra.npy").write_bytes(b"extra")
    with pytest.raises(ValueError, match="inventory mismatch"):
        analyzer.validate_feature_inventory(
            latest_results=[row],
            repo_root=tmp_path,
            feature_dir=feature_dir,
        )


def test_feature_artifact_rejects_noncanonical_trailing_bytes(
    tmp_path: Path,
) -> None:
    row, feature_dir = _feature_row(tmp_path)
    path = feature_dir / "sample.npy"
    path.write_bytes(path.read_bytes() + b"x")
    row["commfor_feature"]["sha256"] = sha256_file(path)
    row["commfor_feature"]["file_bytes"] = path.stat().st_size
    row["commfor_feature_sha256"] = sha256_file(path)
    with pytest.raises(ValueError, match="metadata|non-canonical"):
        analyzer._feature_artifact(
            row=row,
            sample_id="sample",
            repo_root=tmp_path,
            feature_dir=feature_dir,
        )


def test_feature_artifact_rejects_array_hash_alias_drift(
    tmp_path: Path,
) -> None:
    row, feature_dir = _feature_row(tmp_path)
    row["commfor_feature_array_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="alias"):
        analyzer._feature_artifact(
            row=row,
            sample_id="sample",
            repo_root=tmp_path,
            feature_dir=feature_dir,
        )


def test_runtime_contract_accepts_exact_cpu_record() -> None:
    runtime = _runtime()
    assert analyzer._validate_runtime_contract(
        runtime,
        label="runtime",
    ) == runtime


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("cudnn", "enabled"), True),
        (("venv", "include_system_site_packages"), False),
        (("process_environment", "PYTHONHASHSEED"), "101"),
        (("packages", "timm"), "1.0.16"),
    ],
)
def test_runtime_contract_rejects_numerical_or_environment_drift(
    path: tuple[str, str],
    value: Any,
) -> None:
    runtime = _runtime()
    runtime[path[0]][path[1]] = value
    with pytest.raises(ValueError):
        analyzer._validate_runtime_contract(runtime, label="runtime")


def test_execution_accounting_includes_same_device_head_replays() -> None:
    manifest = {
        "execution": {
            "new_successes": 35,
            "resume_skips": 0,
            "new_errors": 0,
            "physical_result_rows": 35,
            "latest_result_rows": 35,
            "superseded_attempts": 0,
            "same_device_feature_head_replays": 35,
        }
    }
    analyzer._validate_execution(
        manifest=manifest,
        selected_images=35,
        physical_rows=35,
        latest_rows=35,
    )
    manifest["execution"]["same_device_feature_head_replays"] = 34
    with pytest.raises(ValueError, match="replays"):
        analyzer._validate_execution(
            manifest=manifest,
            selected_images=35,
            physical_rows=35,
            latest_rows=35,
        )


def test_summary_binds_visibility_and_replay_count() -> None:
    contract = SimpleNamespace(as_dict=lambda: {"contract": "frozen"})
    visibility = {"profile_id": analyzer.FROZEN_PROFILE}
    coverage = {"valid_images": 35, "is_complete": True}
    summary = {
        "schema_version": analyzer.EXPECTED_RUNTIME_SUMMARY_SCHEMA,
        "summary_kind": "runtime_coverage_only",
        "scientific_metrics": None,
        "scientific_metrics_owner": (
            "analyze_community_forensics_balanced.py"
        ),
        "run_id": "run-a",
        "run_manifest_fingerprint": "f" * 64,
        "status": "complete",
        "mode": "smoke",
        "model": analyzer.legacy.MODEL_NAME,
        "model_slug": analyzer.legacy.MODEL_SLUG,
        "preprocess_profile": analyzer.FROZEN_PROFILE,
        "score_spec": analyzer._score_spec().as_dict(),
        "dataset_contract": contract.as_dict(),
        "selection_visibility_census": visibility,
        "same_device_feature_head_replays": 35,
        "coverage": coverage,
        "generated_at": "2026-07-26T00:00:00Z",
    }
    analyzer._validate_summary(
        summary=summary,
        bundle_mode="smoke",
        run_id="run-a",
        fingerprint="f" * 64,
        contract=contract,
        selection_visibility=visibility,
        coverage=coverage,
    )


def test_classifier_head_replay_loads_keys_from_full_safetensors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")
    model_root = tmp_path / "model"
    model_root.mkdir()
    checkpoint = model_root / "model.safetensors"
    checkpoint.write_bytes(b"full-checkpoint")
    checkpoint_record = {
        **analyzer.legacy.CHECKPOINT,
        "filename": checkpoint.name,
        "bytes": checkpoint.stat().st_size,
        "sha256": sha256_file(checkpoint),
    }
    monkeypatch.setattr(analyzer.legacy, "CHECKPOINT", checkpoint_record)
    weight = torch.zeros((1, analyzer.FEATURE_DIMENSION), dtype=torch.float32)
    bias = torch.tensor([0.25], dtype=torch.float32)
    observed: dict[str, Any] = {}

    def checkpoint_schema(path: Path, *, torch_module: Any) -> Any:
        observed["path"] = path
        observed["torch"] = torch_module
        return (
            {"vit.head.weight": weight, "vit.head.bias": bias},
            {
                "items_sha256": analyzer.EXPECTED_MODEL_CONTRACT[
                    "checkpoint_schema_sha256"
                ]
            },
        )

    monkeypatch.setattr(
        analyzer.legacy_audit,
        "_checkpoint_schema",
        checkpoint_schema,
    )
    monkeypatch.setattr(
        analyzer,
        "_configure_exact_recorded_runtime",
        lambda **_kwargs: (torch.device("cpu"), {"device": "cpu"}),
    )
    array = np.zeros((analyzer.FEATURE_DIMENSION,), dtype=np.float32)
    artifact = analyzer.FeatureArtifact(
        sample_id="sample",
        path=tmp_path / "unused.npy",
        file_sha256="b" * 64,
        file_bytes=1664,
        array_sha256=analyzer._array_sha256(array),
        array=array,
    )
    raw = float(bias.item())
    probability = float(torch.sigmoid(bias).item())
    row = _score_fields(raw_logit=raw, probability=probability)
    report = analyzer.replay_linear_head(
        latest_results=[row],
        features={"sample": artifact},
        model_root=model_root,
        device_text="cpu",
        recorded_runtime={"device": "cpu"},
    )
    assert observed == {"path": checkpoint, "torch": torch}
    assert report["features_replayed"] == 1
    assert report["weight_key"] == "vit.head.weight"
    assert report["bias_key"] == "vit.head.bias"
    assert report["max_raw_logit_abs_difference"] == 0.0
    assert report["max_probability_abs_difference"] == 0.0


def test_classifier_head_replay_requires_exact_recorded_score(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")
    model_root = tmp_path / "model"
    model_root.mkdir()
    checkpoint = model_root / "model.safetensors"
    checkpoint.write_bytes(b"full-checkpoint")
    monkeypatch.setattr(
        analyzer.legacy,
        "CHECKPOINT",
        {
            **analyzer.legacy.CHECKPOINT,
            "filename": checkpoint.name,
            "bytes": checkpoint.stat().st_size,
            "sha256": sha256_file(checkpoint),
        },
    )
    monkeypatch.setattr(
        analyzer.legacy_audit,
        "_checkpoint_schema",
        lambda *_args, **_kwargs: (
            {
                "vit.head.weight": torch.zeros(
                    (1, analyzer.FEATURE_DIMENSION),
                    dtype=torch.float32,
                ),
                "vit.head.bias": torch.zeros((1,), dtype=torch.float32),
            },
            {
                "items_sha256": analyzer.EXPECTED_MODEL_CONTRACT[
                    "checkpoint_schema_sha256"
                ]
            },
        ),
    )
    monkeypatch.setattr(
        analyzer,
        "_configure_exact_recorded_runtime",
        lambda **_kwargs: (torch.device("cpu"), {"device": "cpu"}),
    )
    array = np.zeros((analyzer.FEATURE_DIMENSION,), dtype=np.float32)
    artifact = analyzer.FeatureArtifact(
        sample_id="sample",
        path=tmp_path / "unused.npy",
        file_sha256="b" * 64,
        file_bytes=1664,
        array_sha256=analyzer._array_sha256(array),
        array=array,
    )
    row = _score_fields(raw_logit=0.0, probability=0.5000001)
    with pytest.raises(ValueError, match="probability mismatch"):
        analyzer.replay_linear_head(
            latest_results=[row],
            features={"sample": artifact},
            model_root=model_root,
            device_text="cpu",
            recorded_runtime={"device": "cpu"},
        )


def _fresh_replay_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Any, Path, Path, Path]:
    torch = pytest.importorskip("torch")
    source_root = tmp_path / "source"
    model_root = tmp_path / "model"
    processor_root = tmp_path / "processor"
    image_root = tmp_path / "images"
    for directory in (source_root, model_root, processor_root, image_root):
        directory.mkdir()
    checkpoint_path = model_root / "model.safetensors"
    checkpoint_path.write_bytes(b"checkpoint")

    class FakeModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.vit = torch.nn.Module()
            self.vit.head = torch.nn.Linear(
                analyzer.FEATURE_DIMENSION,
                1,
                bias=True,
            )
            with torch.no_grad():
                self.vit.head.weight.zero_()
                self.vit.head.bias.fill_(0.25)

        def forward(self, value: Any) -> Any:
            feature = torch.zeros(
                (value.shape[0], analyzer.FEATURE_DIMENSION),
                dtype=torch.float32,
                device=value.device,
            )
            return self.vit.head(feature)

    source = {"root": source_root.as_posix(), "commit": "source"}
    assets = {
        "checkpoint": {
            "path": checkpoint_path.as_posix(),
            "schema": {"items_sha256": "c" * 64},
        },
        "model_repository": {"root": model_root.as_posix()},
        "processor": {"root": processor_root.as_posix()},
        "bundle_sha256": "d" * 64,
    }
    selected: list[dict[str, Any]] = []
    latest: list[dict[str, Any]] = []
    features: dict[str, analyzer.FeatureArtifact] = {}
    probability = float(
        torch.sigmoid(torch.tensor([0.25], dtype=torch.float32)).item()
    )
    for index in range(2):
        sample_id = f"sample-{index}"
        input_path = image_root / f"{sample_id}.jpg"
        input_path.write_bytes(f"image-{index}".encode())
        selected.append(
            {
                "sample_id": sample_id,
                "canonical_path": input_path.relative_to(tmp_path).as_posix(),
                "canonical_sha256": sha256_file(input_path),
                "width": 440,
                "height": 440,
            }
        )
        audit = {
            "profile": analyzer.FROZEN_PROFILE,
            "geometry": analyzer._current_geometry(440, 440),
        }
        latest.append(
            {
                **_score_fields(
                    sample_id=sample_id,
                    raw_logit=0.25,
                    probability=probability,
                ),
                "preprocess": audit,
            }
        )
        array = np.zeros((analyzer.FEATURE_DIMENSION,), dtype=np.float32)
        features[sample_id] = analyzer.FeatureArtifact(
            sample_id=sample_id,
            path=tmp_path / f"{sample_id}.npy",
            file_sha256="e" * 64,
            file_bytes=1664,
            array_sha256=analyzer._array_sha256(array),
            array=array,
        )
    bundle = SimpleNamespace(
        selected=tuple(selected),
        latest_results=tuple(latest),
        immutable={
            "runtime": {"device": "cpu"},
            "source": source,
            "assets": assets,
        },
        release=SimpleNamespace(repo_root=tmp_path),
        features=features,
    )
    monkeypatch.setattr(analyzer, "FORMAL_IMAGES", 2)
    monkeypatch.setattr(
        analyzer,
        "_configure_exact_recorded_runtime",
        lambda **_kwargs: (torch.device("cpu"), {"device": "cpu"}),
    )
    monkeypatch.setattr(
        analyzer.legacy_audit,
        "_load_runner_pins",
        lambda: object(),
    )
    monkeypatch.setattr(
        analyzer.legacy_audit,
        "_verify_source_tree",
        lambda *_args, **_kwargs: source,
    )
    monkeypatch.setattr(
        analyzer.legacy_audit,
        "_verify_assets",
        lambda **_kwargs: ({"state": torch.tensor(1)}, assets),
    )
    monkeypatch.setattr(
        analyzer.legacy_audit,
        "_construct_model",
        lambda **_kwargs: (FakeModel(), {"strict": True}),
    )

    def preprocess(_path: Path, *, torch_module: Any) -> Any:
        return SimpleNamespace(
            tensor=torch_module.zeros(
                (3, 384, 384),
                dtype=torch_module.float32,
            ),
            audit={
                "profile": analyzer.FROZEN_PROFILE,
                "geometry": analyzer._current_geometry(440, 440),
            },
        )

    monkeypatch.setattr(
        analyzer.legacy_audit,
        "preprocess_image",
        preprocess,
    )
    return bundle, source_root, model_root, processor_root


def test_fresh_model_replay_covers_every_selected_image(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle, source_root, model_root, processor_root = _fresh_replay_fixture(
        tmp_path,
        monkeypatch,
    )
    report = analyzer.replay_model(
        bundle,
        source_root=source_root,
        model_root=model_root,
        processor_root=processor_root,
        device_text="cpu",
    )
    assert report["images_replayed"] == 2
    assert report["full_image_forward_per_input"] is True
    assert report["full_model_replay"] is True
    assert report["classifier_head_only_replay"] is False
    assert report["max_feature_abs_difference"] == 0.0
    assert report["max_raw_logit_abs_difference"] == 0.0


def test_fresh_model_replay_rejects_feature_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle, source_root, model_root, processor_root = _fresh_replay_fixture(
        tmp_path,
        monkeypatch,
    )
    bundle.features["sample-0"].array[0] = 1.0
    with pytest.raises(ValueError, match="fresh feature mismatch"):
        analyzer.replay_model(
            bundle,
            source_root=source_root,
            model_root=model_root,
            processor_root=processor_root,
            device_text="cpu",
        )


def test_output_collision_rejects_evidence_files_and_directories(
    tmp_path: Path,
) -> None:
    evidence = tmp_path / "manifest.json"
    evidence.write_text("{}\n", encoding="utf-8")
    protected_dir = tmp_path / "features"
    protected_dir.mkdir()
    with pytest.raises(ValueError, match="collision"):
        analyzer._validate_output_targets(
            {"audit": evidence},
            protected_files=[evidence],
            protected_dirs=[protected_dir],
        )
    with pytest.raises(ValueError, match="collision"):
        analyzer._validate_output_targets(
            {"audit": protected_dir / "audit.json"},
            protected_files=[evidence],
            protected_dirs=[protected_dir],
        )


def test_bundle_protection_accepts_eval_single_git_blobs(tmp_path: Path) -> None:
    bundle = SimpleNamespace(
        manifest_path=tmp_path / "manifest.json",
        results_path=tmp_path / "results.jsonl",
        expected_path=tmp_path / "expected_inputs.jsonl",
        summary_path=tmp_path / "summary.json",
        immutable={
            "source": {
                "source_files": {
                    "models.py": {
                        "path": (tmp_path / "source/models.py").as_posix(),
                    }
                },
                "eval_single": {
                    "files": {
                        "main.py": {
                            "git_object": "f" * 40 + ":main.py",
                            "bytes": 123,
                            "sha256": "a" * 64,
                        }
                    }
                },
            },
            "assets": {
                "checkpoint": {
                    "path": (tmp_path / "model/model.safetensors").as_posix(),
                },
                "model_repository": {
                    "files": {
                        "config.json": {
                            "path": (tmp_path / "model/config.json").as_posix(),
                        }
                    }
                },
                "processor": {
                    "files": {
                        "preprocessor_config.json": {
                            "path": (
                                tmp_path / "processor/preprocessor_config.json"
                            ).as_posix(),
                        }
                    }
                },
            },
        },
    )
    protected = analyzer._bundle_protected_files(
        bundle,
        repo_root=tmp_path,
    )
    assert bundle.manifest_path in protected
    assert tmp_path / "source/models.py" in protected
    assert all("git_object" not in path.as_posix() for path in protected)


def test_cli_uses_source_model_and_processor_roots() -> None:
    parser = analyzer._build_parser()
    destinations = {action.dest for action in parser._actions}
    assert {"source_root", "model_root", "processor_root"} <= destinations
    assert "head_checkpoint" not in destinations
    assert "backbone_checkpoint" not in destinations


def test_source_contains_no_foreign_detector_architecture_residue() -> None:
    source = Path(analyzer.__file__).read_text(encoding="utf-8")
    forbidden = (
        "CURRENT_" + "PROFILE",
        "HEAD_" + "CHECKPOINT",
        "BACK" + "BONE",
        "Universal" + "Fake",
    )
    assert not any(token in source for token in forbidden)


def test_npy_serialization_is_stable_and_lossless() -> None:
    array = np.arange(analyzer.FEATURE_DIMENSION, dtype=np.float32)
    payload = analyzer._npy_bytes(array)
    assert len(payload) == 1664
    assert hashlib.sha256(payload).hexdigest()
    assert np.array_equal(
        np.load(__import__("io").BytesIO(payload), allow_pickle=False),
        array,
    )


def test_recompute_metrics_rejects_nonfrozen_bootstrap_settings() -> None:
    with pytest.raises(ValueError, match="iterations=1000"):
        analyzer.recompute_metrics(
            SimpleNamespace(),
            iterations=999,
            seed=analyzer.BOOTSTRAP_SEED,
        )
    with pytest.raises(ValueError, match="seed=20260726"):
        analyzer.recompute_metrics(
            SimpleNamespace(),
            iterations=analyzer.BOOTSTRAP_ITERATIONS,
            seed=1,
        )


def test_rows_sha256_binds_order_and_canonical_json() -> None:
    first = [{"sample_id": "a"}, {"sample_id": "b"}]
    second = list(reversed(first))
    assert analyzer._rows_sha256(first) != analyzer._rows_sha256(second)
    expected = hashlib.sha256(
        "".join(f"{stable_json(row)}\n" for row in first).encode()
    ).hexdigest()
    assert analyzer._rows_sha256(first) == expected


def test_intersection_xyxy_handles_full_partial_and_none() -> None:
    assert analyzer._intersection_xyxy([0, 0, 2, 2], [0, 0, 2, 2]) == [
        0.0,
        0.0,
        2.0,
        2.0,
    ]
    assert analyzer._intersection_xyxy([0, 0, 2, 2], [1, 1, 3, 3]) == [
        1.0,
        1.0,
        2.0,
        2.0,
    ]
    assert analyzer._intersection_xyxy([0, 0, 1, 1], [1, 1, 2, 2]) is None


def test_manifest_fingerprint_uses_stable_json() -> None:
    immutable = {"z": 1, "a": {"b": 2}}
    expected = hashlib.sha256(stable_json(immutable).encode()).hexdigest()
    assert len(expected) == 64
    assert json.loads(stable_json(immutable)) == immutable
