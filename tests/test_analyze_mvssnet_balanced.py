from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from eval.opensource import analyze_mvssnet_balanced as analyzer
from eval.opensource import run_mvssnet_balanced as runner
from eval.opensource.canonical_release import load_canonical_release
from eval.opensource.common import repo_relative, sha256_file


REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_MANIFEST = Path("outputs/opensource/balanced250_v1/manifest.json")
RUNNER_SHA256 = "210941969074dfb023e162b7b70370629159322cb638b8e08bbe65203100afdd"


def _runtime(device: str = "cuda:0") -> dict:
    value = {
        "device": device,
        "device_type": "cuda",
        "gpu_name": "fixture GPU",
        "gpu_compute_capability": [9, 0],
        "gpu_total_memory_bytes": 1,
        "torch_version": runner.EXPECTED_PACKAGES["torch"],
        "torch_cuda_version": "12.8",
        "precision": "float32",
        "batch_size": 1,
        "autocast": False,
        "apex": False,
        "seed": 42,
        "deterministic_algorithms": True,
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
        "matmul_tf32": False,
        "cudnn_tf32": False,
        "cublas_workspace_config": ":4096:8",
    }
    return {**value, "contract_sha256": analyzer._fingerprint(value)}


def _immutable_fixture() -> dict:
    adapters = runner.adapter_source_contract(REPO_ROOT)
    return {
        "schema_version": runner.RUN_CONFIG_SCHEMA,
        "run_id": "fixture",
        "mode": "smoke",
        "adapter_sources": adapters,
        "adapter_sources_sha256": analyzer._fingerprint(adapters),
        "model": {
            "name": runner.MODEL_NAME,
            "slug": runner.MODEL_SLUG,
            "architecture": runner.MODEL_ARCHITECTURE,
            "repository": analyzer.legacy.MODEL_REPO_URL,
            "source_commit": analyzer.legacy.MODEL_SOURCE_COMMIT,
            "checkpoint_id": runner.CHECKPOINT_ID,
            "checkpoint_sha256": analyzer.legacy.CHECKPOINT_SHA256,
            "checkpoint_bytes": runner.CHECKPOINT_BYTES,
            "training_dataset": "CASIAv2",
            "variant": "original_MVSS-Net_not_MVSS-Net++",
        },
        "preprocess": {
            "profile": runner.PREPROCESS_PROFILE,
            "decode": "opencv_imread_color",
            "channel_order": "BGR",
            "resize": "opencv_INTER_LINEAR_stretch_512x512",
            "scale": "uint8_divide_255",
            "normalization_mean_in_BGR_order": (
                analyzer.legacy.NORMALIZE_MEAN.tolist()
            ),
            "normalization_std_in_BGR_order": (analyzer.legacy.NORMALIZE_STD.tolist()),
            "tensor_layout": "CHW",
            "tensor_dtype": "float32",
            "batch_size": 1,
            "autocast": False,
            "apex": False,
        },
        "inference": {
            "raw_output": "one_channel_segmentation_logits_512",
            "probability_map": "sigmoid_segmentation_logits",
            "auxiliary_edge_output": "discarded_as_official_inference",
            "primary_T1": "continuous_global_max_of_model_512_probability",
            "secondary_T1": ("official_native_saved_uint8_PNG_global_max_divide_255"),
            "native_restore": (
                "probability_times_255_uint8_truncate_before_"
                "opencv_INTER_LINEAR_native_resize"
            ),
            "bayar_state": "constraint_kernel_normalized_in_place_each_forward",
            "resume": ("fresh_checkpoint_and_replay_of_every_successful_prefix_input"),
        },
        "score_spec": runner.SCORE_SPEC.as_dict(),
        "t2_spec": json.loads(json.dumps(runner.T2_SPEC)),
        "task_scope": json.loads(json.dumps(runner.TASK_SCOPE)),
        "dataset_contract": {},
        "selected_rows_sha256": "0" * 64,
        "selected_ids_sha256": "0" * 64,
        "source": {},
        "checkpoint": {},
        "environment": {},
        "checkpoint_audit": {},
        "model_audit": {},
        "mouse_reference": {},
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
            "conservative_pending_bytes_plus_reserve": 1,
            "fixed_reserve_bytes": runner.MIN_DISK_RESERVE_BYTES,
        },
        "execution": {
            "new_successes": 35,
            "resume_skips": 0,
            "new_errors": 0,
            "physical_result_rows": 35,
            "latest_result_rows": 35,
            "superseded_attempts": 0,
            "stateful_prefix_replayed": 0,
        },
    }


def _array_metadata(
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
        f"{prefix}_dtype": str(array.dtype),
        f"{prefix}_semantics": semantics,
    }


def _png_metadata(
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
        f"{prefix}_dtype": "uint8",
        f"{prefix}_mode": "L",
        f"{prefix}_semantics": semantics,
    }


def _artifact_fixture(
    root: Path,
    *,
    applicable: bool,
    sample_id: str = "a" * 24,
) -> tuple[dict, dict, Path]:
    artifact_root = root / "outputs/opensource/mvssnet/run"
    paths = runner.artifact_paths(artifact_root, sample_id)
    logits = np.zeros((512, 512), dtype=np.float32)
    logits[:, 256:] = np.float32(1.0)
    model = analyzer._stable_sigmoid_array(logits)
    native = analyzer._independent_native_postprocess(
        model,
        width=3,
        height=2,
    )
    runner.legacy._atomic_save_npy(paths["raw_logits"], logits)
    runner.legacy._atomic_save_npy(paths["model_score"], model)
    runner.legacy._atomic_save_gray_png(paths["native_score"], native)
    row = {
        **analyzer._score_payload(logits, model, native),
        **_array_metadata(
            root=root,
            path=paths["raw_logits"],
            prefix="raw_logits_model",
            array=logits,
            semantics="official_one_channel_segmentation_logits",
        ),
        **_array_metadata(
            root=root,
            path=paths["model_score"],
            prefix="score_map_model",
            array=model,
            semantics="official_sigmoid_segmentation_probability",
        ),
        **_png_metadata(
            root=root,
            path=paths["native_score"],
            prefix="score_map_native",
            array=native,
            semantics=(
                "official_probability_times_255_uint8_truncate_then_"
                "opencv_INTER_LINEAR_native_resize"
            ),
        ),
        "mask_threshold": 0.5,
        "mask_threshold_operator": ">",
    }
    if applicable:
        mask = np.where(
            native.astype(np.float32) / np.float32(255.0) > 0.5,
            255,
            0,
        ).astype(np.uint8)
        runner.legacy._atomic_save_gray_png(paths["mask"], mask)
        target_native = np.zeros((2, 3), dtype=bool)
        target_model = analyzer._independent_resize_target(
            target_native,
            width=512,
            height=512,
        )
        row.update(
            {
                **_png_metadata(
                    root=root,
                    path=paths["mask"],
                    prefix="mask",
                    array=mask,
                    semantics=("official_native_uint8_divide_255_strict_gt_0_5"),
                ),
                "artifact_paths": {
                    "raw_logits_model_512": repo_relative(
                        paths["raw_logits"],
                        root,
                    ),
                    "score_map_model_512": repo_relative(
                        paths["model_score"],
                        root,
                    ),
                    "score_map_native_official": repo_relative(
                        paths["native_score"],
                        root,
                    ),
                    "mask_native": repo_relative(paths["mask"], root),
                },
                "localization": {
                    "model_512": analyzer._independent_pixel_metrics(
                        model,
                        target_model,
                        include_ap=False,
                    ),
                    "native": analyzer._independent_pixel_metrics(
                        native.astype(np.float32) / np.float32(255.0),
                        target_native,
                        include_ap=False,
                    ),
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
                "mask_mode": None,
                "mask_semantics": None,
                "artifact_paths": {
                    "raw_logits_model_512": repo_relative(
                        paths["raw_logits"],
                        root,
                    ),
                    "score_map_model_512": repo_relative(
                        paths["model_score"],
                        root,
                    ),
                    "score_map_native_official": repo_relative(
                        paths["native_score"],
                        root,
                    ),
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
        "import eval.opensource.analyze_mvssnet_balanced; "
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


def test_runner_remains_frozen():
    assert sha256_file(REPO_ROOT / "eval/opensource/run_mvssnet_balanced.py") == (
        RUNNER_SHA256
    )


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
    manifest["immutable"]["runtime"]["cudnn_deterministic"] = False
    unsigned = dict(manifest["immutable"]["runtime"])
    unsigned.pop("contract_sha256")
    manifest["immutable"]["runtime"]["contract_sha256"] = analyzer._fingerprint(
        unsigned
    )
    manifest["fingerprint"] = analyzer._fingerprint(manifest["immutable"])
    with pytest.raises(ValueError, match="cudnn_deterministic"):
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


def test_runtime_contract_is_hashed_and_requires_explicit_cuda():
    runtime = _runtime()
    assert analyzer._validate_runtime(runtime, label="fixture")["device"] == "cuda:0"
    runtime["gpu_name"] = "changed"
    with pytest.raises(ValueError, match="contract SHA"):
        analyzer._validate_runtime(runtime, label="fixture")
    cpu = _runtime("cpu")
    with pytest.raises(ValueError, match="CUDA"):
        analyzer._validate_runtime(cpu, label="fixture")


def test_score_payload_uses_strict_greater_than_at_exact_half():
    logits = np.zeros((512, 512), dtype=np.float32)
    model = np.full((512, 512), 0.5, dtype=np.float32)
    native = np.full((2, 3), 127, dtype=np.uint8)
    payload = analyzer._score_payload(logits, model, native)
    assert payload["ai_score"] == 0.5
    assert payload["classification_decision"] is False
    assert payload["classification_threshold_operator"] == ">"
    analyzer._validate_score_payload(payload, sample_id="sample")


def test_independent_preprocess_is_bgr_stretch_and_normalize(tmp_path: Path):
    path = tmp_path / "input.png"
    Image.fromarray(
        np.asarray([[[10, 20, 30], [40, 50, 60]]], dtype=np.uint8),
        mode="RGB",
    ).save(path)
    tensor, native_size, audit = analyzer._independent_preprocess_tensor(path)
    assert native_size == (2, 1)
    assert tensor.shape == (3, 512, 512)
    assert tensor.dtype == np.float32
    assert audit["channel_order"] == "BGR"
    assert audit["resize"] == "opencv_inter_linear_stretch"
    expected_blue = (np.float32(30 / 255) - np.float32(0.485)) / np.float32(0.229)
    assert tensor[0, 0, 0] == expected_blue


def test_strict_pixel_threshold_treats_exact_half_as_negative():
    score = np.asarray([[0.5, 0.5001]], dtype=np.float32)
    target = np.asarray([[True, True]])
    metrics = analyzer._independent_pixel_metrics(
        score,
        target,
        include_ap=False,
    )
    assert metrics["threshold_operator"] == ">"
    assert metrics["predicted_positive_pixels"] == 1
    assert metrics["fn"] == 1


def test_applicable_artifacts_replay_every_transform(tmp_path: Path):
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


def test_fullframe_keeps_three_dense_artifacts_but_no_t2_claim(tmp_path: Path):
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


def test_artifact_hash_and_sigmoid_contract_are_fail_closed(tmp_path: Path):
    row, expected, artifact_root = _artifact_fixture(
        tmp_path,
        applicable=False,
    )
    row["raw_logits_model_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="metadata"):
        analyzer._validate_artifact_row(
            row,
            expected=expected,
            repo_root=tmp_path,
            artifact_root=artifact_root,
        )
    row, expected, artifact_root = _artifact_fixture(
        tmp_path / "second",
        applicable=False,
    )
    model_path = tmp_path / "second" / row["score_map_model_path"]
    changed = np.zeros((512, 512), dtype=np.float32)
    runner.legacy._atomic_save_npy(model_path, changed)
    row.update(
        _array_metadata(
            root=tmp_path / "second",
            path=model_path,
            prefix="score_map_model",
            array=changed,
            semantics="official_sigmoid_segmentation_probability",
        )
    )
    native_path = tmp_path / "second" / row["score_map_native_path"]
    changed_native = analyzer._independent_native_postprocess(
        changed,
        width=3,
        height=2,
    )
    runner.legacy._atomic_save_gray_png(native_path, changed_native)
    row.update(
        _png_metadata(
            root=tmp_path / "second",
            path=native_path,
            prefix="score_map_native",
            array=changed_native,
            semantics=(
                "official_probability_times_255_uint8_truncate_then_"
                "opencv_INTER_LINEAR_native_resize"
            ),
        )
    )
    with pytest.raises(ValueError, match="sanity"):
        analyzer._validate_artifact_row(
            row,
            expected=expected,
            repo_root=tmp_path / "second",
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


def test_stateful_history_accepts_recovery_but_rejects_out_of_order():
    selected = [{"sample_id": "first"}, {"sample_id": "second"}]
    attempts = [
        {"sample_id": "first", "status": "error"},
        {"sample_id": "first", "status": "ok"},
        {"sample_id": "second", "status": "ok"},
    ]
    report = analyzer._validate_stateful_history(selected, attempts)
    assert report == {
        "policy": "exact_selected_prefix_with_zero_or_more_errors_before_each_ok",
        "physical_attempts": 3,
        "successful_prefix": 2,
        "errors": 1,
        "recovered_error_to_ok": 1,
    }
    attempts[1]["sample_id"] = "second"
    with pytest.raises(ValueError, match="out of selected order"):
        analyzer._validate_stateful_history(selected, attempts)


def _comparison_fixture(
    tmp_path: Path,
) -> tuple[list[dict], list[dict], dict, dict]:
    left_file = tmp_path / "left.bin"
    right_file = tmp_path / "right.bin"
    left_file.write_bytes(b"exact")
    right_file.write_bytes(b"exact")
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
            "raw_logits_model_path": "a",
            "score_map_model_path": "a",
            "score_map_native_path": "a",
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
                "raw_logits_model_path": "b",
                "score_map_model_path": "b",
                "score_map_native_path": "b",
                "mask_path": "b" if applicable else None,
            }
        )
        left_rows.append(base)
        right_rows.append(other)
        common = {
            "sample_id": sample_id,
            "raw_logits_file_sha256": "0" * 64,
            "raw_logits_array_sha256": "0" * 64,
            "model_score_file_sha256": "0" * 64,
            "model_score_array_sha256": "0" * 64,
            "native_score_file_sha256": "0" * 64,
            "native_score_array_sha256": "0" * 64,
            "mask_file_sha256": "0" * 64 if applicable else None,
            "mask_array_sha256": "0" * 64 if applicable else None,
            "t2_applicable": applicable,
            "width": 1,
            "height": 1,
        }
        left_artifacts[sample_id] = analyzer.DenseArtifacts(
            raw_logits_path=left_file,
            model_score_path=left_file,
            native_score_path=left_file,
            mask_path=left_file if applicable else None,
            **common,
        )
        right_artifacts[sample_id] = analyzer.DenseArtifacts(
            raw_logits_path=right_file,
            model_score_path=right_file,
            native_score_path=right_file,
            mask_path=right_file if applicable else None,
            **common,
        )
    return left_rows, right_rows, left_artifacts, right_artifacts


def test_smoke_comparison_is_exact_except_run_specific_fields(tmp_path: Path):
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


def test_shared_metric_contract_uses_frozen_seed_and_uint8_equivalent_t2(
    monkeypatch: pytest.MonkeyPatch,
):
    selected = [
        {
            "sample_id": str(index),
            "gt_mask_kind": "all_zero" if index < 1025 else "not_applicable",
        }
        for index in range(1775)
    ]
    calls = {}

    def fake_t1(*_args, **kwargs):
        calls["t1"] = kwargs
        return {
            "schema_version": analyzer.T1_METRICS_SCHEMA_VERSION,
            "coverage": {"is_complete": True},
        }

    def fake_t2(*_args, **kwargs):
        calls["t2"] = kwargs
        return {
            "schema_version": analyzer.T2_METRICS_SCHEMA_VERSION,
            "coverage": {
                "is_complete": True,
                "native_maps_evaluated": 1025,
            },
            "excluded_not_applicable": {
                "counts_by_condition": {
                    "fullframe_mouse": 250,
                    "fullframe_cat": 250,
                    "fullframe_trash_can": 250,
                }
            },
        }

    monkeypatch.setattr(analyzer, "summarize_balanced250_t1", fake_t1)
    monkeypatch.setattr(analyzer, "summarize_balanced250_t2", fake_t2)
    bundle = SimpleNamespace(
        mode="formal",
        selected=selected,
        latest_results=[{}] * 1775,
        release=SimpleNamespace(
            inputs=[],
            panel=[],
            source_pairs=[],
            repo_root=REPO_ROOT,
        ),
        run_id=runner.DEFAULT_FORMAL_RUN_ID,
        fingerprint="f" * 64,
        contract=object(),
        artifacts={},
    )
    report = analyzer.recompute_metrics(bundle)
    assert calls["t1"]["iterations"] == 1000
    assert calls["t1"]["seed"] == 20260726
    assert calls["t2"]["threshold_operator"] == ">="
    assert report["official_t2_threshold_operator"] == ">"
    assert report["shared_t2_operator_equivalent_on_uint8_divide_255"] is True


def test_structural_golden_is_cpu_only_and_has_no_fabricated_numeric_claim(
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
        mvssnet_root=Path("/unused/source"),
        recorded_checkpoint_audit=checkpoint,
        recorded_model_audit=model,
    )
    assert report["author_published_numerical_golden"] is None
    assert report["model_forwards"] == 0
    assert report["executable_numeric_gates"] == [
        "frozen_smoke_A_B_exact_reproduction",
        "formal_full_stateful_sequence_exact_fresh_replay",
    ]


def test_parser_defaults_to_full_fresh_replay_and_smoke_comparison():
    args = analyzer._build_parser().parse_args([])
    assert args.run_id == runner.DEFAULT_FORMAL_RUN_ID
    assert args.skip_model_replay is False
    assert args.skip_smoke_comparison is False
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
