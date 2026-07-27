from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pytest
from PIL import Image

from eval.opensource import analyze_trufor_balanced as analyzer
from eval.opensource import run_trufor_balanced as runner
from eval.opensource.canonical_release import load_canonical_release
from eval.opensource.common import repo_relative, sha256_file


REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_MANIFEST = Path("outputs/opensource/balanced250_v1/manifest.json")


def _runtime(device: str = "cpu") -> dict:
    value = {
        "device": device,
        "seed": 42,
        "precision": "float32",
        "batch_size": 1,
        "autocast": False,
        "inference_mode": True,
        "deterministic_algorithms_enabled": False,
        "deterministic_algorithms_warn_only": False,
        "cublas_workspace_config": None,
        "cudnn": {
            "enabled": False,
            "benchmark": False,
            "deterministic": False,
            "allow_tf32": True,
            "source": "official_trufor_ph3_config",
        },
        "matmul_allow_tf32": True,
        "float32_matmul_precision": "high",
        "torch_cuda_runtime": "12.8",
        "cuda_initialized_before_configuration": False,
        "cuda_initialized_after_configuration": device.startswith("cuda:"),
    }
    if device.startswith("cuda:"):
        value["cuda"] = {
            "device_index": int(device.split(":", 1)[1]),
            "device_name": "fixture GPU",
            "total_memory_bytes": 1,
            "capability": [9, 0],
        }
    return value


def _immutable_fixture() -> dict:
    return {
        "schema_version": runner.RUN_CONFIG_SCHEMA,
        "run_id": "fixture",
        "mode": "smoke",
        "adapter_sources": runner.adapter_source_contract(REPO_ROOT),
        "model": {
            "name": runner.MODEL_NAME,
            "slug": runner.MODEL_SLUG,
            "architecture": runner.MODEL_ARCHITECTURE,
            "repository": analyzer.legacy.MODEL_REPO_URL,
            "source_commit": analyzer.legacy.MODEL_SOURCE_COMMIT,
            "checkpoint_id": runner.CHECKPOINT_ID,
            "checkpoint_sha256": analyzer.legacy.CHECKPOINT_SHA256,
            "checkpoint_bytes": runner.CHECKPOINT_BYTES,
            "positive_class_index": 1,
        },
        "preprocess": {
            "profile": runner.PREPROCESS_PROFILE,
            "decode": "PIL_convert_RGB",
            "tensor_layout": "CHW",
            "tensor_dtype": "float32",
            "input_scale_divisor": 256.0,
            "input_resize": None,
            "input_crop": None,
            "network_map_upsample": (
                "bilinear_align_corners_false_to_native_input_size"
            ),
            "batch_size": 1,
            "autocast": False,
        },
        "score_spec": runner.SCORE_SPEC.as_dict(),
        "t2_spec": json.loads(json.dumps(runner.T2_SPEC)),
        "task_scope": json.loads(json.dumps(runner.TASK_SCOPE)),
        "dataset_contract": {},
        "selected_rows_sha256": "0" * 64,
        "selected_ids_sha256": "0" * 64,
        "source": {},
        "assets": {},
        "environment": {},
        "checkpoint_audit": {},
        "model_audit": {},
        "license": json.loads(json.dumps(runner.LICENSE_RECORD)),
        "runtime": _runtime(),
        "cpu_preflight": {},
        "artifact_contract": json.loads(json.dumps(runner.ARTIFACT_CONTRACT)),
        "outputs": {},
    }


def _manifest_fixture() -> dict:
    immutable = _immutable_fixture()
    return {
        "schema_version": runner.RUN_MANIFEST_SCHEMA,
        "run_id": "fixture",
        "status": "complete",
        "started_at": "start",
        "completed_at": "end",
        "fingerprint": analyzer._fingerprint(immutable),
        "immutable": immutable,
        "dataset": {},
        "outputs": {},
        "disk_preflight": {
            "free_bytes_before_inference": 1,
            "estimated_pending_map_bytes_plus_reserve": 1,
            "fixed_reserve_bytes": runner.MIN_DISK_RESERVE_BYTES,
        },
        "execution": {
            "new_successes": 35,
            "resume_skips": 0,
            "new_errors": 0,
            "physical_result_rows": 35,
            "latest_result_rows": 35,
            "superseded_attempts": 0,
        },
    }


def _save_npy(path: Path, array: np.ndarray) -> None:
    runner.legacy._atomic_save_npy(path, array)


def _map_metadata(
    *,
    root: Path,
    path: Path,
    prefix: str,
    array: np.ndarray,
    semantics: str,
) -> dict:
    return {
        f"{prefix}_path": repo_relative(path, root),
        f"{prefix}_sha256": sha256_file(path),
        f"{prefix}_bytes": path.stat().st_size,
        f"{prefix}_array_sha256": analyzer._array_sha256(array),
        f"{prefix}_shape": list(array.shape),
        f"{prefix}_dtype": "float32",
        f"{prefix}_semantics": semantics,
    }


def _artifact_fixture(
    root: Path,
    *,
    sample_id: str = "sample",
    applicable: bool,
) -> tuple[dict, dict, Path]:
    artifact_root = root / "outputs/opensource/trufor/run"
    paths = runner.artifact_paths(artifact_root, sample_id)
    score = np.asarray(
        [[0.25, 0.5, 0.75], [0.0, 1.0, 0.4]],
        dtype=np.float32,
    )
    reliability = np.asarray(
        [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
        dtype=np.float32,
    )
    _save_npy(paths["score"], score)
    _save_npy(paths["reliability"], reliability)
    row = {
        **_map_metadata(
            root=root,
            path=paths["score"],
            prefix="score_map_native",
            array=score,
            semantics=("softmax_localization_logits_channel_1_forged_probability"),
        ),
        **_map_metadata(
            root=root,
            path=paths["reliability"],
            prefix="reliability_map_native",
            array=reliability,
            semantics=("sigmoid_TCP_localization_reliability_not_anomaly"),
        ),
        "mask_threshold": 0.5,
        "mask_threshold_operator": ">=",
        "reliability": {
            "semantics": ("TCP_localization_reliability_not_forged_probability"),
            "used_for_primary_metrics": False,
            "multiplied_into_score_map": False,
            "min": float(np.min(reliability)),
            "mean": float(np.mean(reliability)),
            "median": float(np.median(reliability)),
            "p05": float(np.quantile(reliability, 0.05)),
            "p95": float(np.quantile(reliability, 0.95)),
            "max": float(np.max(reliability)),
        },
    }
    if applicable:
        runner.legacy._atomic_save_mask(paths["mask"], score >= 0.5)
        with Image.open(paths["mask"]) as opened:
            pixels = np.asarray(opened, dtype=np.uint8)
        target = np.zeros(score.shape, dtype=bool)
        row.update(
            {
                "mask_path": repo_relative(paths["mask"], root),
                "mask_sha256": sha256_file(paths["mask"]),
                "mask_bytes": paths["mask"].stat().st_size,
                "mask_array_sha256": analyzer._array_sha256(pixels),
                "mask_shape": list(score.shape),
                "mask_dtype": "uint8",
                "mask_semantics": ("native_probability_map_ge_0_5_encoded_L_0_or_255"),
                "artifact_paths": {
                    "score_map_native": repo_relative(paths["score"], root),
                    "reliability_map_native": repo_relative(paths["reliability"], root),
                    "mask_native": repo_relative(paths["mask"], root),
                },
                "localization": {
                    "native": analyzer._independent_pixel_metrics(
                        score,
                        target,
                        include_ap=False,
                    )
                },
            }
        )
        expected = {
            "sample_id": sample_id,
            "width": 3,
            "height": 2,
            "gt_mask_kind": "all_zero",
            "kind": "real",
            "label": 0,
            "condition": "real",
            "manipulation_scope": "authentic",
            "gt_mask_path": None,
            "gt_mask_sha256": None,
            "gt_positive_pixels": 0,
        }
    else:
        row.update(
            {
                "mask_path": None,
                "mask_sha256": None,
                "mask_bytes": None,
                "mask_array_sha256": None,
                "mask_shape": None,
                "mask_dtype": None,
                "mask_semantics": None,
                "artifact_paths": {
                    "score_map_native": repo_relative(paths["score"], root),
                    "reliability_map_native": repo_relative(paths["reliability"], root),
                    "mask_native": None,
                },
                "localization": None,
            }
        )
        expected = {
            "sample_id": sample_id,
            "width": 3,
            "height": 2,
            "gt_mask_kind": "not_applicable",
            "kind": "forged",
            "label": 1,
            "condition": "fullframe_mouse",
            "manipulation_scope": "conditional_full_frame_edit",
            "gt_mask_path": None,
            "gt_mask_sha256": None,
            "gt_positive_pixels": None,
        }
    return row, expected, artifact_root


def test_import_does_not_initialize_cuda_in_fresh_process():
    code = (
        "import torch; assert not torch.cuda.is_initialized(); "
        "import eval.opensource.analyze_trufor_balanced; "
        "print(torch.cuda.is_initialized())"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stdout.strip() == "False"


def test_frozen_formal_and_smoke_selections_are_exact():
    release = load_canonical_release(
        REPO_ROOT,
        DATASET_MANIFEST,
        verify_files=False,
    )
    _, formal = analyzer._selection_for_mode(
        release,
        mode="formal",
        per_condition_limit=None,
    )
    _, smoke = analyzer._selection_for_mode(
        release,
        mode="smoke",
        per_condition_limit=5,
    )
    assert len(formal) == 1775
    assert len(smoke) == 35
    assert sum(row["gt_mask_kind"] in analyzer._T2_GT_KINDS for row in formal) == 1025
    assert sum(row["gt_mask_kind"] in analyzer._T2_GT_KINDS for row in smoke) == 20


def test_manifest_fingerprint_and_static_contract_are_fail_closed():
    manifest = _manifest_fixture()
    fingerprint, immutable = analyzer._validate_manifest_envelope(
        manifest,
        repo_root=REPO_ROOT,
        run_id="fixture",
        expected_mode="smoke",
    )
    assert fingerprint == analyzer._fingerprint(immutable)
    manifest["fingerprint"] = "0" * 64
    with pytest.raises(ValueError, match="fingerprint"):
        analyzer._validate_manifest_envelope(
            manifest,
            repo_root=REPO_ROOT,
            run_id="fixture",
            expected_mode="smoke",
        )


def test_manifest_rejects_runtime_and_fullframe_contract_drift():
    manifest = _manifest_fixture()
    manifest["immutable"]["runtime"]["cudnn"]["enabled"] = True
    manifest["fingerprint"] = analyzer._fingerprint(manifest["immutable"])
    with pytest.raises(ValueError, match="cudnn"):
        analyzer._validate_manifest_envelope(
            manifest,
            repo_root=REPO_ROOT,
            run_id="fixture",
            expected_mode="smoke",
        )
    manifest = _manifest_fixture()
    manifest["immutable"]["t2_spec"]["not_applicable_conditions"] = []
    manifest["fingerprint"] = analyzer._fingerprint(manifest["immutable"])
    with pytest.raises(ValueError, match="scientific"):
        analyzer._validate_manifest_envelope(
            manifest,
            repo_root=REPO_ROOT,
            run_id="fixture",
            expected_mode="smoke",
        )


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_runtime_contract_accepts_only_frozen_shape(device: str):
    assert (
        analyzer._validate_runtime(
            _runtime(device),
            label="fixture runtime",
        )["device"]
        == device
    )
    changed = _runtime(device)
    changed["deterministic_algorithms_enabled"] = True
    with pytest.raises(ValueError, match="deterministic"):
        analyzer._validate_runtime(changed, label="fixture runtime")


def test_score_payload_is_bound_to_sigmoid_and_threshold():
    logit = 0.25
    score = analyzer._stable_sigmoid(logit)
    row = {
        "raw_detection_logit": logit,
        "raw_outputs": {"binary_forged_logit": logit},
        "class_probabilities": {"real": 1.0 - score, "forged": score},
        "ai_score": score,
        "probability": score,
        "score": score,
        "score_margin": logit,
        "score_semantics": ("sigmoid_binary_logit_probability_of_forged"),
        "calibrated_probability": False,
        "classification_decision": score >= 0.5,
        "classification_threshold": 0.5,
        "classification_threshold_operator": ">=",
    }
    analyzer._validate_score_payload(row, sample_id="sample")
    row["classification_decision"] = not row["classification_decision"]
    with pytest.raises(ValueError, match="score contract"):
        analyzer._validate_score_payload(row, sample_id="sample")


def test_independent_preprocess_is_float32_divide_256(tmp_path: Path):
    path = tmp_path / "input.png"
    Image.fromarray(
        np.asarray([[[0, 128, 255], [255, 0, 64]]], dtype=np.uint8),
        mode="RGB",
    ).save(path)
    tensor, audit = analyzer._independent_preprocess_tensor(path)
    assert tensor.shape == (3, 1, 2)
    assert tensor.dtype == np.float32
    assert tensor[0, 0, 1] == np.float32(255 / 256)
    assert audit["input_resize"] is None
    assert audit["input_crop"] is None


def test_applicable_artifact_validates_raw_maps_png_and_real_fp(
    tmp_path: Path,
):
    row, expected, artifact_root = _artifact_fixture(
        tmp_path,
        applicable=True,
    )
    artifact = analyzer._validate_artifact_row(
        row,
        expected=expected,
        repo_root=tmp_path,
        artifact_root=artifact_root,
    )
    assert artifact.t2_applicable is True
    assert artifact.mask_path is not None


def test_fullframe_keeps_raw_maps_but_has_no_t2_claim(tmp_path: Path):
    row, expected, artifact_root = _artifact_fixture(
        tmp_path,
        applicable=False,
    )
    artifact = analyzer._validate_artifact_row(
        row,
        expected=expected,
        repo_root=tmp_path,
        artifact_root=artifact_root,
    )
    assert artifact.t2_applicable is False
    assert artifact.mask_path is None
    row["localization"] = {"native": {}}
    with pytest.raises(ValueError, match="fabricates T2"):
        analyzer._validate_artifact_row(
            row,
            expected=expected,
            repo_root=tmp_path,
            artifact_root=artifact_root,
        )


def test_artifact_hash_and_float32_contract_are_fail_closed(
    tmp_path: Path,
):
    row, expected, artifact_root = _artifact_fixture(
        tmp_path,
        applicable=False,
    )
    row["score_map_native_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="metadata"):
        analyzer._validate_artifact_row(
            row,
            expected=expected,
            repo_root=tmp_path,
            artifact_root=artifact_root,
        )


def test_nonfinite_raw_map_is_rejected_even_with_resigned_hash(
    tmp_path: Path,
):
    row, expected, artifact_root = _artifact_fixture(
        tmp_path,
        applicable=False,
    )
    path = tmp_path / row["score_map_native_path"]
    changed = np.zeros((2, 3), dtype=np.float32)
    changed[0, 0] = np.nan
    _save_npy(path, changed)
    row["score_map_native_sha256"] = sha256_file(path)
    row["score_map_native_bytes"] = path.stat().st_size
    row["score_map_native_array_sha256"] = analyzer._array_sha256(changed)
    with pytest.raises(ValueError, match="array contract"):
        analyzer._validate_artifact_row(
            row,
            expected=expected,
            repo_root=tmp_path,
            artifact_root=artifact_root,
        )


def test_mask_must_equal_raw_score_map_ge_point_five(tmp_path: Path):
    row, expected, artifact_root = _artifact_fixture(
        tmp_path,
        applicable=True,
    )
    mask_path = tmp_path / row["mask_path"]
    wrong = np.zeros((2, 3), dtype=np.uint8)
    Image.fromarray(wrong, mode="L").save(mask_path, format="PNG")
    row["mask_sha256"] = sha256_file(mask_path)
    row["mask_bytes"] = mask_path.stat().st_size
    row["mask_array_sha256"] = analyzer._array_sha256(wrong)
    with pytest.raises(ValueError, match="score map"):
        analyzer._validate_artifact_row(
            row,
            expected=expected,
            repo_root=tmp_path,
            artifact_root=artifact_root,
        )


def test_artifact_inventory_rejects_extra_top_level_entry(tmp_path: Path):
    artifact_root = tmp_path / "artifacts"
    for name in analyzer.EXPECTED_ARTIFACT_INVENTORY:
        (artifact_root / name).mkdir(parents=True, exist_ok=True)
    (artifact_root / "extra").mkdir()
    with pytest.raises(ValueError, match="artifact-root"):
        analyzer.validate_artifact_inventory(
            latest_results=[],
            selected=[],
            repo_root=tmp_path,
            artifact_root=artifact_root,
        )


def _comparison_fixture(
    tmp_path: Path,
) -> tuple[list[dict], list[dict], dict, dict]:
    left_file = tmp_path / "left.npy"
    right_file = tmp_path / "right.npy"
    np.save(left_file, np.zeros((1,), dtype=np.float32))
    np.save(right_file, np.zeros((1,), dtype=np.float32))
    left_rows = []
    right_rows = []
    left_artifacts = {}
    right_artifacts = {}
    for index in range(35):
        sample_id = f"sample-{index}"
        applicable = index < 20
        base = {
            "sample_id": sample_id,
            "ai_score": 0.25,
            "run_id": "A",
            "run_manifest_fingerprint": "a" * 64,
            "config_fingerprint": "a" * 64,
            "completed_at": "a",
            "latency_ms": 1.0,
            "peak_cuda_memory_bytes": 1,
            "score_map_native_path": "a",
            "reliability_map_native_path": "a",
            "mask_path": "a" if applicable else None,
            "artifact_paths": {},
        }
        other = dict(base)
        other.update(
            {
                "run_id": "B",
                "run_manifest_fingerprint": "b" * 64,
                "config_fingerprint": "b" * 64,
                "completed_at": "b",
                "latency_ms": 2.0,
                "peak_cuda_memory_bytes": 2,
                "score_map_native_path": "b",
                "reliability_map_native_path": "b",
                "mask_path": "b" if applicable else None,
            }
        )
        left_rows.append(base)
        right_rows.append(other)
        left_artifacts[sample_id] = analyzer.DenseArtifacts(
            sample_id,
            left_file,
            "0" * 64,
            "0" * 64,
            left_file,
            "0" * 64,
            "0" * 64,
            left_file if applicable else None,
            "0" * 64 if applicable else None,
            "0" * 64 if applicable else None,
            applicable,
            1,
            1,
        )
        right_artifacts[sample_id] = analyzer.DenseArtifacts(
            sample_id,
            right_file,
            "0" * 64,
            "0" * 64,
            right_file,
            "0" * 64,
            "0" * 64,
            right_file if applicable else None,
            "0" * 64 if applicable else None,
            "0" * 64 if applicable else None,
            applicable,
            1,
            1,
        )
    return left_rows, right_rows, left_artifacts, right_artifacts


def test_smoke_comparison_is_exact_except_frozen_runtime_fields(
    tmp_path: Path,
):
    left, right, left_artifacts, right_artifacts = _comparison_fixture(tmp_path)
    report = analyzer.compare_computational_results(
        left,
        right,
        reference_artifacts=left_artifacts,
        replay_artifacts=right_artifacts,
    )
    assert report["images_compared"] == 35
    assert report["t2_applicable_images_compared"] == 20
    right[7]["ai_score"] = 0.5
    with pytest.raises(ValueError, match="computational row"):
        analyzer.compare_computational_results(
            left,
            right,
            reference_artifacts=left_artifacts,
            replay_artifacts=right_artifacts,
        )


def test_structural_golden_has_no_fabricated_numeric_claim(
    monkeypatch: pytest.MonkeyPatch,
):
    checkpoint = {"schema": "checkpoint"}
    model = {"schema": "model"}
    monkeypatch.setattr(
        runner,
        "_build_cpu_model_audit",
        lambda **_kwargs: (checkpoint, model),
    )
    report = analyzer.independent_structural_golden(
        checkpoint_path=Path("/unused/checkpoint"),
        trufor_root=Path("/unused/source"),
        recorded_checkpoint_audit=checkpoint,
        recorded_model_audit=model,
    )
    assert report["author_published_numerical_golden"] is None
    assert report["model_forwards"] == 0
    assert report["executable_numeric_gates"] == [
        "frozen_smoke_A_B_exact_reproduction",
        "formal_full_selection_exact_fresh_replay",
    ]


def test_standard_roots_and_output_paths_reject_escape(tmp_path: Path):
    root = tmp_path / "repo"
    expected = root / "results/opensource/trufor"
    expected.mkdir(parents=True)
    assert (
        analyzer._safe_standard_root(
            Path("results/opensource/trufor"),
            repo_root=root,
            expected_relative=Path("results/opensource/trufor"),
            label="results",
        )
        == expected
    )
    with pytest.raises(ValueError, match="exactly"):
        analyzer._safe_standard_root(
            tmp_path / "outside",
            repo_root=root,
            expected_relative=Path("results/opensource/trufor"),
            label="results",
        )


def test_parser_defaults_to_full_fresh_replay():
    args = analyzer._build_parser().parse_args([])
    assert args.run_id == runner.DEFAULT_FORMAL_RUN_ID
    assert args.skip_model_replay is False
    assert args.bootstrap_iterations == 1000
    assert args.bootstrap_seed == 20260726
    assert args.device == "cuda:0"


def test_strict_json_rejects_duplicate_keys_and_nonfinite(tmp_path: Path):
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"x":1,"x":2}', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        analyzer._load_json(duplicate, "duplicate")
    nonfinite = tmp_path / "nonfinite.json"
    nonfinite.write_text('{"x":NaN}', encoding="utf-8")
    with pytest.raises(ValueError, match="non-finite"):
        analyzer._load_json(nonfinite, "nonfinite")


def test_verified_json_hash_matches_pretty_atomic_encoding(tmp_path: Path):
    value = {"schema_version": "fixture", "nested": {"value": 1}}
    path = tmp_path / "verified.json"
    analyzer._write_json_verified(path, value, label="fixture report")
    assert sha256_file(path) == analyzer._json_sha256(value)


def test_stable_result_row_accepts_read_only_contract_mapping():
    row = {"sample_id": "fixture", "nested": {"score": 0.25}}
    frozen = MappingProxyType(dict(row))
    assert analyzer._stable_result_row(frozen) == analyzer._stable_result_row(row)
