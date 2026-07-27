from __future__ import annotations

import hashlib
import importlib.util
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from eval.opensource import analyze_catnet_balanced as analyzer
from eval.opensource import run_catnet_balanced as runner
from eval.opensource.balanced_run_contract import (
    build_result_identity,
    build_run_dataset_contract,
)
from eval.opensource.canonical_release import (
    BALANCED_DATASET_ID,
    BALANCED_SCHEMA,
    CanonicalRelease,
    Capability,
    SelectionSpec,
    load_canonical_release,
)
from eval.opensource.common import sha256_file, stable_json
from eval.opensource.maskclip_metrics import binary_pixel_metrics


REPO_ROOT = Path(__file__).resolve().parents[1]
FORMAL_MANIFEST = Path("outputs/opensource/balanced250_v1/manifest.json")
try:
    import jpegio as _jpegio  # noqa: F401
except ImportError:
    HAS_JPEGIO = False
else:
    HAS_JPEGIO = True


@pytest.fixture(scope="module")
def release() -> CanonicalRelease:
    return load_canonical_release(
        REPO_ROOT,
        FORMAL_MANIFEST,
        verify_files=False,
    )


def test_analyzer_contract_and_frozen_ids_are_t2_only():
    assert analyzer.DEFAULT_FORMAL_RUN_ID == runner.DEFAULT_FORMAL_RUN_ID
    assert analyzer.DEFAULT_SMOKE_RUN_ID_A == runner.DEFAULT_SMOKE_RUN_ID_A
    assert analyzer.DEFAULT_SMOKE_RUN_ID_B == runner.DEFAULT_SMOKE_RUN_ID_B
    assert runner.FORMAL_IMAGES == 1_025
    assert runner.SMOKE_IMAGES == 20
    assert runner.TASK_SCOPE["valid_for_t1"] is False
    assert runner.TASK_SCOPE["valid_for_t2"] is True
    assert runner.TASK_SCOPE["map_statistic_promoted_to_t1"] is False
    assert runner.T2_SPEC["fullframe_output"]["selected"] is False
    assert runner.T2_SPEC["fullframe_output"]["forward_performed"] is False
    assert runner.T2_SPEC["fullframe_output"]["artifact_saved"] is False


def test_analyzer_source_inventory_binds_runner_shared_metrics_and_self():
    contract = analyzer.analyzer_source_contract(REPO_ROOT)
    assert tuple(contract) == analyzer.ANALYZER_SOURCE_PATHS
    for required in (
        "eval/opensource/analyze_catnet_balanced.py",
        "eval/opensource/run_catnet_balanced.py",
        "eval/opensource/run_catnet.py",
        "eval/opensource/balanced250_localization_metrics.py",
    ):
        assert required in contract
    for relative, binding in contract.items():
        path = REPO_ROOT / relative
        assert binding == {
            "path": relative,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }


def test_independent_selection_rebuilds_formal1025_and_smoke20(
    release: CanonicalRelease,
):
    formal_spec, formal = analyzer._selection_for_mode(release, "formal")
    smoke_spec, smoke = analyzer._selection_for_mode(release, "smoke")
    assert formal_spec.capability is Capability.LOCAL_T2_ONLY
    assert smoke_spec.capability is Capability.LOCAL_T2_ONLY
    assert formal_spec.per_condition_limit is None
    assert smoke_spec.per_condition_limit == 5
    assert len(formal) == 1_025
    assert len(smoke) == 20
    assert Counter(row["condition"] for row in formal) == runner.FORMAL_COUNTS
    assert Counter(row["condition"] for row in smoke) == runner.SMOKE_COUNTS
    assert all(row["panel"] is True for row in smoke)
    assert not any(
        str(row["condition"]).startswith("fullframe_")
        for row in (*formal, *smoke)
    )
    formal_contract = build_run_dataset_contract(
        release, formal_spec, formal, score_spec=None
    )
    smoke_contract = build_run_dataset_contract(
        release, smoke_spec, smoke, score_spec=None
    )
    assert formal_contract.score_spec is None
    assert smoke_contract.score_spec is None
    assert formal_contract.capability.valid_for_t1 is False
    assert smoke_contract.capability.valid_for_t2 is True


@pytest.mark.parametrize(
    ("condition", "gt_kind", "semantics"),
    [
        ("real", "all_zero", "all_zero_real_false_positive_area"),
        ("local_mouse", "exact_diff", "exact_diff_local_insertion"),
        ("local_cat", "exact_diff", "exact_diff_local_insertion"),
        ("local_trash_can", "exact_diff", "exact_diff_local_insertion"),
    ],
)
def test_attempt_validation_accepts_only_t2_identity(
    condition: str,
    gt_kind: str,
    semantics: str,
):
    input_row = _minimal_row(condition, gt_kind)
    run_id = "catnet-test"
    fingerprint = "1" * 64
    result = {
        **build_result_identity(
            input_row,
            run_id=run_id,
            run_manifest_fingerprint=fingerprint,
        ),
        **analyzer._expected_result_extras(
            input_row, valid_for_metrics=True
        ),
        "status": "ok",
        "mask_threshold": 0.5,
        "mask_threshold_operator": ">=",
        "latency_ms": 1.0,
        "peak_cuda_memory_bytes": 0,
    }
    analyzer._validate_attempt(
        result,
        input_row,
        run_id=run_id,
        fingerprint=fingerprint,
    )
    assert result["valid_for_t1"] is False
    assert result["task_scope"]["t2_target_semantics"] == semantics
    for forbidden in runner._FORBIDDEN_T1_TOP_LEVEL:
        assert forbidden not in result


def test_attempt_validation_rejects_map_derived_t1_score():
    input_row = _minimal_row("real", "all_zero")
    run_id = "catnet-test"
    fingerprint = "1" * 64
    result = {
        **build_result_identity(
            input_row,
            run_id=run_id,
            run_manifest_fingerprint=fingerprint,
        ),
        **analyzer._expected_result_extras(
            input_row, valid_for_metrics=True
        ),
        "status": "ok",
        "mask_threshold": 0.5,
        "mask_threshold_operator": ">=",
        "latency_ms": 1.0,
        "peak_cuda_memory_bytes": 0,
        "score": 0.9,
    }
    with pytest.raises(ValueError, match="forbidden T1 fields"):
        analyzer._validate_attempt(
            result,
            input_row,
            run_id=run_id,
            fingerprint=fingerprint,
        )


def test_history_allows_error_recovery_but_success_is_terminal():
    selected = [_minimal_row("real", "all_zero")]
    sample_id = selected[0]["sample_id"]
    report = analyzer._validate_history(
        selected,
        [
            {"sample_id": sample_id, "status": "error"},
            {"sample_id": sample_id, "status": "ok"},
        ],
    )
    assert report["recovered_error_to_ok"] == 1
    assert report["success_is_terminal"] is True
    with pytest.raises(ValueError, match="after success"):
        analyzer._validate_history(
            selected,
            [
                {"sample_id": sample_id, "status": "ok"},
                {"sample_id": sample_id, "status": "error"},
            ],
        )


def test_strict_json_rejects_duplicate_keys_and_noncanonical_jsonl(
    tmp_path: Path,
):
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"a":1,"a":2}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        analyzer._load_json(duplicate, "duplicate")
    rows = tmp_path / "rows.jsonl"
    rows.write_text('{"b": 1, "a": 2}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="canonical"):
        analyzer._read_jsonl(rows, "rows")


def test_runtime_validation_is_fingerprinted_and_t1_free():
    stable = {
        "device": "cpu",
        "seed": runner.MODEL_SEED,
        "deterministic_algorithms": True,
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
        "matmul_allow_tf32": False,
        "cudnn_allow_tf32": False,
        "cublas_workspace_config": runner.CUBLAS_WORKSPACE_CONFIG,
        "precision": "float32",
        "batch_size": 1,
        "autocast": False,
        "torch_version": runner.EXPECTED_PACKAGES["torch"],
        "torch_cuda_version": runner.EXPECTED_TORCH_CUDA_VERSION,
    }
    runtime = {**stable, "contract_sha256": analyzer._fingerprint(stable)}
    assert analyzer._validate_runtime(runtime, label="runtime") == runtime
    runtime["seed"] = 7
    with pytest.raises(ValueError, match="fingerprint"):
        analyzer._validate_runtime(runtime, label="runtime")


def test_runtime_validation_rejects_extra_fields_and_cuda_index_drift():
    stable = {
        "device": "cuda:6",
        "seed": runner.MODEL_SEED,
        "deterministic_algorithms": True,
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
        "matmul_allow_tf32": False,
        "cudnn_allow_tf32": False,
        "cublas_workspace_config": runner.CUBLAS_WORKSPACE_CONFIG,
        "precision": "float32",
        "batch_size": 1,
        "autocast": False,
        "torch_version": runner.EXPECTED_PACKAGES["torch"],
        "torch_cuda_version": runner.EXPECTED_TORCH_CUDA_VERSION,
        "cuda": {
            "logical_device_index": 5,
            "device_name": "frozen-test-device",
            "total_memory_bytes": 1,
            "compute_capability": [8, 0],
        },
    }
    runtime = {**stable, "contract_sha256": analyzer._fingerprint(stable)}
    with pytest.raises(ValueError, match="index differs"):
        analyzer._validate_runtime(runtime, label="runtime")

    stable["cuda"]["logical_device_index"] = 6
    stable["unexpected"] = True
    runtime = {**stable, "contract_sha256": analyzer._fingerprint(stable)}
    with pytest.raises(ValueError, match="exact frozen schema"):
        analyzer._validate_runtime(runtime, label="runtime")


@pytest.mark.skipif(not HAS_JPEGIO, reason="jpegio is required")
def test_independent_jpeg_evidence_replays_qtable_dct_and_ceil8(
    tmp_path: Path,
):
    path = tmp_path / "input.jpg"
    pixels = np.arange(9 * 11 * 3, dtype=np.uint8).reshape(9, 11, 3)
    Image.fromarray(pixels, mode="RGB").save(
        path,
        format="JPEG",
        quality=95,
        subsampling=0,
        optimize=False,
    )
    evidence = analyzer._independent_jpeg_evidence(
        path, width=11, height=9
    )
    assert evidence["native_size"] == [11, 9]
    assert evidence["padded_size"] == [16, 16]
    assert evidence["padding"] == {
        "left": 0,
        "top": 0,
        "right": 5,
        "bottom": 7,
    }
    assert evidence["jpeg_sampling_factors"][:3] == [
        [1, 1],
        [1, 1],
        [1, 1],
    ]
    assert len(evidence["qtable_sha256"]) == 64
    assert len(evidence["dct_y_sha256"]) == 64


@pytest.mark.skipif(not HAS_JPEGIO, reason="jpegio is required")
def test_independent_artifact_replay_checks_logits_map_mask_jpeg_and_metrics(
    tmp_path: Path,
):
    repo_root = tmp_path
    image_path = repo_root / "images/aaaaaaaaaaaaaaaaaaaaaaaa.jpg"
    image_path.parent.mkdir(parents=True)
    pixels = np.zeros((8, 8, 3), dtype=np.uint8)
    Image.fromarray(pixels, mode="RGB").save(
        image_path,
        format="JPEG",
        quality=95,
        subsampling=0,
        optimize=False,
    )
    artifact_root = repo_root / "outputs/opensource/catnet/test"
    runner._prepare_artifact_root(artifact_root)
    input_row = _minimal_row("real", "all_zero")
    input_row.update(
        {
            "canonical_path": "images/aaaaaaaaaaaaaaaaaaaaaaaa.jpg",
            "canonical_sha256": sha256_file(image_path),
            "width": 8,
            "height": 8,
            "gt_mask_path": None,
            "gt_mask_sha256": None,
            "gt_positive_pixels": 0,
        }
    )
    sample_id = input_row["sample_id"]
    paths = runner.artifact_paths(artifact_root, sample_id)
    raw = np.zeros((2, 2, 2), dtype=np.float32)
    score = np.full((8, 8), np.float32(0.5), dtype=np.float32)
    mask = np.where(score >= 0.5, 255, 0).astype(np.uint8)
    runner.legacy._atomic_save_npy(paths["raw_logits"], raw)
    runner.legacy._atomic_save_npy(paths["score_map"], score)
    runner.legacy._atomic_save_mask(paths["mask"], score >= 0.5)
    evidence = analyzer._independent_jpeg_evidence(
        image_path, width=8, height=8
    )
    result = {
        "preprocess": evidence,
        "qtable_sha256": evidence["qtable_sha256"],
        "dct_y_sha256": evidence["dct_y_sha256"],
        **runner._artifact_fields(
            repo_root=repo_root,
            paths=paths,
            raw_logits=raw,
            score_map=score,
            mask=mask,
        ),
        "mask_threshold": 0.5,
        "mask_threshold_operator": ">=",
        "localization": {
            "native": binary_pixel_metrics(
                score,
                np.zeros((8, 8), dtype=bool),
                0.5,
                include_ap=False,
            )
        },
    }
    bundle = _minimal_bundle(
        repo_root=repo_root,
        artifact_root=artifact_root,
        selected=(input_row,),
        latest=(result,),
    )
    audited = analyzer._audit_artifact_row(bundle, input_row, result)
    assert audited["checked_files"] == 4
    assert audited["logits_to_native_max_abs_difference"] == 0.0
    assert audited["raw_logits_array_sha256"] == hashlib.sha256(
        raw.tobytes()
    ).hexdigest()
    assert audited["score_map_array_sha256"] == hashlib.sha256(
        score.tobytes()
    ).hexdigest()
    assert audited["mask_array_sha256"] == hashlib.sha256(
        mask.tobytes()
    ).hexdigest()

    Image.fromarray(np.zeros((8, 8), dtype=np.uint8), mode="L").save(
        paths["mask"], format="PNG", optimize=False
    )
    result["mask_sha256"] = sha256_file(paths["mask"])
    result["mask_bytes"] = paths["mask"].stat().st_size
    result["mask_array_sha256"] = hashlib.sha256(
        np.zeros((8, 8), dtype=np.uint8).tobytes()
    ).hexdigest()
    with pytest.raises(ValueError, match="score_map >= 0.5"):
        analyzer._audit_artifact_row(bundle, input_row, result)


def test_independent_numpy_restore_matches_official_torch_postprocess():
    import torch

    logits = torch.tensor(
        [
            [
                [[8.0, -8.0], [-8.0, 8.0]],
                [[-8.0, 8.0], [8.0, -8.0]],
            ]
        ],
        dtype=torch.float32,
    )
    raw, official = runner.legacy.postprocess_logits(
        logits,
        padded_width=8,
        padded_height=8,
        native_width=7,
        native_height=5,
    )
    independent = analyzer._independent_restore_from_logits(
        raw,
        native_width=7,
        native_height=5,
    )
    assert independent.dtype == np.float32
    assert independent.shape == (5, 7)
    np.testing.assert_allclose(
        independent,
        official,
        rtol=0,
        atol=analyzer.STATIC_LOGIT_RESTORE_ABS_TOLERANCE,
    )


def test_exact_inventory_rejects_extra_files(tmp_path: Path):
    artifact_root = tmp_path / "artifacts"
    runner._prepare_artifact_root(artifact_root)
    bundle = _minimal_bundle(
        repo_root=tmp_path,
        artifact_root=artifact_root,
        selected=(),
        latest=(),
    )
    inventory = analyzer._exact_directory_inventory(bundle)
    assert inventory["files"] == 0
    (artifact_root / "masks_native/extra.png").write_bytes(b"x")
    with pytest.raises(ValueError, match="inventory changed"):
        analyzer._exact_directory_inventory(bundle)


def test_smoke_ab_comparison_is_exact_20_and_rejects_map_drift(
    monkeypatch,
):
    monkeypatch.setattr(analyzer, "_verify_bundle_unchanged", lambda bundle: None)
    audited: list[str] = []

    def fake_audit(bundle):
        audited.append(bundle.run_id)
        return {
            "status": "ok",
            "selected_images": runner.SMOKE_IMAGES,
            "independent_logits_to_native_replays": runner.SMOKE_IMAGES,
            "artifact_inventory": {"files": runner.SMOKE_IMAGES * 3},
        }

    monkeypatch.setattr(analyzer, "audit_artifacts", fake_audit)
    selected = tuple(
        {"sample_id": f"{index:024x}", "condition": condition}
        for index, condition in enumerate(
            condition
            for condition in runner.SMOKE_COUNTS
            for _ in range(5)
        )
    )
    left_rows = tuple(
        _computational_row(row["sample_id"], row["condition"])
        for row in selected
    )
    right_rows = tuple(json.loads(stable_json(row)) for row in left_rows)
    base_immutable = {
        "schema_version": runner.RUN_CONFIG_SCHEMA,
        "dataset_contract": {"capability": "local_t2_only"},
        "runtime": {"device": "cuda:0"},
    }
    smoke_a = SimpleNamespace(
        mode="smoke",
        run_id=runner.DEFAULT_SMOKE_RUN_ID_A,
        selected=selected,
        latest_results=left_rows,
        manifest={
            "immutable": {
                **base_immutable,
                "run_id": runner.DEFAULT_SMOKE_RUN_ID_A,
                "outputs": {"results": "a"},
            }
        },
    )
    smoke_b = SimpleNamespace(
        mode="smoke",
        run_id=runner.DEFAULT_SMOKE_RUN_ID_B,
        selected=selected,
        latest_results=right_rows,
        manifest={
            "immutable": {
                **base_immutable,
                "run_id": runner.DEFAULT_SMOKE_RUN_ID_B,
                "outputs": {"results": "b"},
            }
        },
    )
    report = analyzer.compare_smoke_runs(smoke_a, smoke_b)
    assert report["status"] == "pass"
    assert report["selection"]["images"] == 20
    assert report["selection"]["counts_by_condition"] == runner.SMOKE_COUNTS
    assert report["selection"]["fullframe_images"] == 0
    assert report["comparison"]["bit_exact"] is True
    assert report["physical_artifact_audits_passed"] == 2
    assert report["artifact_files_audited"] == 120
    assert report["artifact_files_compared_byte_exact"] == 60
    assert report["t1_fields_compared"] == 0
    assert audited == [
        runner.DEFAULT_SMOKE_RUN_ID_A,
        runner.DEFAULT_SMOKE_RUN_ID_B,
    ]

    right_rows[0]["score_map_array_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="differ"):
        analyzer.compare_smoke_runs(smoke_a, smoke_b)


def test_metrics_delegate_to_shared_t2_reducer_with_formal1025(
    release: CanonicalRelease,
    monkeypatch,
    tmp_path: Path,
):
    spec, selected = analyzer._selection_for_mode(release, "formal")
    contract = build_run_dataset_contract(
        release, spec, selected, score_spec=None
    )
    results = tuple(
        {
            "sample_id": row["sample_id"],
            "condition": row["condition"],
        }
        for row in selected
    )
    for name in ("manifest.json", "results.jsonl", "summary.json"):
        (tmp_path / name).write_text("{}\n", encoding="utf-8")
    bundle = SimpleNamespace(
        mode="formal",
        selected=tuple(selected),
        release=release,
        latest_results=results,
        repo_root=REPO_ROOT,
        run_id=runner.DEFAULT_FORMAL_RUN_ID,
        fingerprint="1" * 64,
        dataset_contract=contract,
        manifest_path=tmp_path / "manifest.json",
        results_path=tmp_path / "results.jsonl",
        summary_path=tmp_path / "summary.json",
    )
    captured: dict = {}

    def fake_summary(inputs, passed_results, **kwargs):
        captured["inputs"] = inputs
        captured["results"] = passed_results
        captured.update(kwargs)
        return {
            "coverage": {
                "selected_results": 1_025,
                "native_maps_evaluated": 1_025,
                "all_zero_real_images": 275,
                "exact_diff_local_images": 750,
                "not_applicable_selected_images": 0,
            },
            "excluded_not_applicable": {
                "selected_images": 0,
                "score_map_loader_calls": 0,
                "counts_by_condition": {
                    "fullframe_mouse": 0,
                    "fullframe_cat": 0,
                    "fullframe_trash_can": 0,
                },
            },
        }

    monkeypatch.setattr(analyzer, "summarize_balanced250_t2", fake_summary)
    monkeypatch.setattr(analyzer, "_verify_bundle_unchanged", lambda bundle: None)
    report = analyzer.recompute_metrics(bundle, iterations=17, seed=23)
    assert report["score_spec"] is None
    assert report["task_scope"]["valid_for_t1"] is False
    assert len(captured["inputs"]) == 1_775
    assert len(captured["results"]) == 1_025
    assert captured["run_dataset_contract"] is contract
    assert captured["threshold"] == 0.5
    assert captured["threshold_operator"] == ">="
    assert captured["iterations"] == 17
    assert captured["seed"] == 23
    assert callable(captured["load_native_score_map"])


def test_static_immutable_contract_rejects_score_spec():
    immutable = {
        "schema_version": runner.RUN_CONFIG_SCHEMA,
        "mode": "formal",
        "model": {
            "name": runner.MODEL_NAME,
            "model_slug": runner.MODEL_SLUG,
            "architecture": runner.MODEL_ARCHITECTURE,
            "repo_url": runner.legacy.MODEL_REPO_URL,
            "source_commit": runner.legacy.MODEL_SOURCE_COMMIT,
            "source_tree": runner.MODEL_TREE,
            "checkpoint_filename": runner.legacy.CHECKPOINT_FILENAME,
            "checkpoint_sha256": runner.legacy.CHECKPOINT_SHA256,
            "checkpoint_bytes": runner.legacy.CHECKPOINT_BYTES,
            "checkpoint_epoch": runner.legacy.CHECKPOINT_EPOCH,
            "checkpoint_state_keys": runner.legacy.CHECKPOINT_STATE_KEYS,
            "checkpoint_strict_load": True,
            "checkpoint_safe_weights_only_load": True,
            "license": runner.LICENSE_RECORD,
        },
        "task_scope": runner.TASK_SCOPE,
        "t2_spec": runner.T2_SPEC,
        "score_spec": None,
        "artifact_contract": runner.ARTIFACT_CONTRACT,
        "resource_expectation": runner.RESOURCE_EXPECTATION,
        "inference": {
            "t1_policy": "unsupported_no_derived_image_score",
            "mask_threshold": 0.5,
            "mask_threshold_operator": ">=",
            "map_restore": (
                "bilinear_logits_to_padded_native_align_corners_false_then_"
                "softmax_channel_1_then_native_crop"
            ),
        },
        "preprocess": {
            "profile": runner.PREPROCESS_PROFILE,
            "input_resize": "none",
            "input_reencode": False,
            "jpeg_reader": "jpegio",
        },
    }
    analyzer._validate_immutable_static(immutable, mode="formal")
    immutable["score_spec"] = {
        "key": "map_mean",
        "direction": "higher_means_fake",
    }
    with pytest.raises(ValueError, match="score spec"):
        analyzer._validate_immutable_static(immutable, mode="formal")


def test_analysis_parser_keeps_frozen_default_ids():
    args = analyzer._build_parser().parse_args(["--phase", "artifact"])
    assert args.formal_run_id == runner.DEFAULT_FORMAL_RUN_ID
    assert args.smoke_run_id_a == runner.DEFAULT_SMOKE_RUN_ID_A
    assert args.smoke_run_id_b == runner.DEFAULT_SMOKE_RUN_ID_B


def _minimal_row(condition: str, gt_kind: str) -> dict:
    real = condition == "real"
    return {
        "schema_version": BALANCED_SCHEMA,
        "dataset_id": BALANCED_DATASET_ID,
        "rank": 0,
        "sample_id": "a" * 24,
        "condition": condition,
        "condition_family": "real" if real else "local_splice",
        "manipulation_scope": "authentic" if real else "local_insertion",
        "normalized_task_id": "task",
        "task_id": "task",
        "kind": "real" if real else "forged",
        "label": 0 if real else 1,
        "domain": "lodging",
        "gt_mask_kind": gt_kind,
        "canonical_path": "images/aaaaaaaaaaaaaaaaaaaaaaaa.jpg",
        "canonical_sha256": "2" * 64,
        "width": 8,
        "height": 8,
        "source_content_cluster": "cluster",
    }


def _minimal_bundle(
    *,
    repo_root: Path,
    artifact_root: Path,
    selected: tuple,
    latest: tuple,
):
    return SimpleNamespace(
        repo_root=repo_root,
        artifact_root=artifact_root,
        selected=selected,
        latest_results=latest,
    )


def _computational_row(sample_id: str, condition: str) -> dict:
    return {
        "sample_id": sample_id,
        "condition": condition,
        "status": "ok",
        "preprocess": {"qtable": "same"},
        "qtable_sha256": "1" * 64,
        "dct_y_sha256": "2" * 64,
        "raw_logits_sha256": "3" * 64,
        "raw_logits_array_sha256": "4" * 64,
        "raw_logits_shape": [2, 2, 2],
        "raw_logits_dtype": "float32",
        "score_map_sha256": "5" * 64,
        "score_map_array_sha256": "6" * 64,
        "score_map_shape": [8, 8],
        "score_map_dtype": "float32",
        "mask_sha256": "7" * 64,
        "mask_array_sha256": "8" * 64,
        "mask_shape": [8, 8],
        "mask_dtype": "uint8",
        "mask_threshold": 0.5,
        "mask_threshold_operator": ">=",
        "localization": {"native": {"f1": 1.0}},
    }
