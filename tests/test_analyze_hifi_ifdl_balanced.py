from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from eval.opensource import analyze_hifi_ifdl_balanced as analyzer
from eval.opensource import run_hifi_ifdl_balanced as runner
from eval.opensource.canonical_release import (
    BALANCED_CONDITIONS,
    BALANCED_DATASET_ID,
    BALANCED_SCHEMA,
    load_canonical_release,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
FORMAL_MANIFEST = Path("outputs/opensource/balanced250_v1/manifest.json")


@pytest.fixture(scope="module")
def release():
    return load_canonical_release(
        REPO_ROOT,
        FORMAL_MANIFEST,
        verify_files=False,
    )


def test_frozen_balanced_contract_and_default_ids():
    assert analyzer.DEFAULT_FORMAL_RUN_ID == (
        "hifi_ifdl_general750001_balanced250_v1_full1775_r2_20260727"
    )
    assert analyzer.DEFAULT_SMOKE_RUN_ID_A == (
        "hifi_ifdl_general750001_balanced250_v1_smoke5x7_a_r2_20260727"
    )
    assert analyzer.DEFAULT_SMOKE_RUN_ID_B == (
        "hifi_ifdl_general750001_balanced250_v1_smoke5x7_b_r2_20260727"
    )
    assert analyzer.FORMAL_IMAGES == 1_775
    assert analyzer.FORMAL_T2_IMAGES == 1_025
    assert analyzer.SMOKE_IMAGES == 35
    assert analyzer.SMOKE_T2_IMAGES == 20
    assert analyzer.BOOTSTRAP_ITERATIONS == 1_000
    assert analyzer.BOOTSTRAP_SEED == 20_260_726
    assert analyzer.CLASSIFICATION_THRESHOLD == 0.5
    assert analyzer.MASK_THRESHOLD == 2.3
    assert analyzer.SIMPLEX_SUM_ABSOLUTE_TOLERANCE == float(
        2 * np.finfo(np.float32).eps
    )
    assert analyzer.STATIC_SOFTMAX_ABSOLUTE_TOLERANCE == float(
        8 * np.finfo(np.float32).eps
    )
    assert analyzer._score_spec().as_dict()["threshold_operator"] == ">"
    sanity = analyzer._expected_inference_contract()["static_cpu_softmax_sanity"]
    assert sanity["classes"] == 14
    assert sanity["simplex_sum_absolute_tolerance"] == float(
        2 * np.finfo(np.float32).eps
    )
    assert sanity["cross_device_absolute_tolerance"] == float(
        8 * np.finfo(np.float32).eps
    )
    assert sanity["recorded_device_smoke_and_fresh_replay_tolerance"] == 0.0
    parser = analyzer._build_parser()
    args = parser.parse_args([])
    assert args.skip_fresh_replay is False
    assert args.run_id == analyzer.DEFAULT_FORMAL_RUN_ID


def test_independent_selection_hashes_exact_formal_and_smoke(release):
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
    assert len(formal) == 1_775
    assert sum(runner._t2_semantics(row)[0] for row in formal) == 1_025
    assert analyzer._rows_sha256(formal) == analyzer.FORMAL_SELECTED_ROWS_SHA256
    assert len(smoke) == 35
    assert sum(runner._t2_semantics(row)[0] for row in smoke) == 20
    assert analyzer._rows_sha256(smoke) == analyzer.SMOKE_SELECTED_ROWS_SHA256
    assert {row["condition"] for row in smoke} == set(BALANCED_CONDITIONS)
    with pytest.raises(ValueError, match="only accepts"):
        analyzer._selection_for_mode(
            release,
            mode="single",
            per_condition_limit=None,
        )


def test_adapter_source_inventory_is_live_and_analyzer_is_not_runner_bound():
    verified = analyzer._verify_adapter_sources(
        runner.adapter_source_contract(REPO_ROOT),
        repo_root=REPO_ROOT,
    )
    assert tuple(verified) == runner.ADAPTER_SOURCE_PATHS
    assert "eval/opensource/run_hifi_ifdl_balanced.py" in verified
    assert "eval/opensource/analyze_hifi_ifdl_balanced.py" not in verified
    changed = {key: dict(value) for key, value in verified.items()}
    first = next(iter(changed))
    changed[first]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="content changed"):
        analyzer._verify_adapter_sources(changed, repo_root=REPO_ROOT)


def test_real_source_and_assets_independently_verify_without_cuda():
    import torch

    assert torch.cuda.is_initialized() is False
    source = analyzer._independent_source_record(analyzer.DEFAULT_HIFI_ROOT)
    assets = analyzer._independent_assets_record(
        hifi_root=analyzer.DEFAULT_HIFI_ROOT,
        hrnet_checkpoint=analyzer.DEFAULT_HRNET_CHECKPOINT,
        nlc_checkpoint=analyzer.DEFAULT_NLC_CHECKPOINT,
    )
    assert source["commit"] == runner.legacy.MODEL_SOURCE_COMMIT
    assert source["tree"] == runner.MODEL_TREE
    assert source["contract_sha256"] == analyzer._fingerprint(
        {key: value for key, value in source.items() if key != "contract_sha256"}
    )
    assert assets["bundle_sha256"] == runner.legacy.CHECKPOINT_BUNDLE_SHA256
    assert set(assets["assets"]) == {
        "initialization_weight",
        "feature_extractor",
        "hierarchical_localizer_classifier",
        "center_radius",
    }
    assert torch.cuda.is_initialized() is False


def test_real_structural_golden_uses_complete_cpu_strict_load():
    if Path(sys.executable) != runner.EXPECTED_PYTHON_EXECUTABLE:
        pytest.skip("real structural golden requires pinned HiFi interpreter")
    if (
        os.environ.get("PYTHONDONTWRITEBYTECODE") != "1"
        or sys.pycache_prefix is None
        or Path(sys.pycache_prefix).resolve()
        != runner.FROZEN_PYTHONPYCACHEPREFIX.resolve()
    ):
        pytest.skip("real structural golden requires frozen bytecode isolation")
    script = """
import json
import torch
from eval.opensource import analyze_hifi_ifdl_balanced as analyzer
from eval.opensource import run_hifi_ifdl_balanced as runner
assert torch.cuda.is_initialized() is False
checkpoint, model = runner._construct_cpu_model_audit(
    hifi_root=analyzer.DEFAULT_HIFI_ROOT,
    hrnet_checkpoint=analyzer.DEFAULT_HRNET_CHECKPOINT,
    nlc_checkpoint=analyzer.DEFAULT_NLC_CHECKPOINT,
)
runner._construct_cpu_model_audit = lambda **_kwargs: (checkpoint, model)
report = analyzer.independent_structural_golden(
    hifi_root=analyzer.DEFAULT_HIFI_ROOT,
    hrnet_checkpoint=analyzer.DEFAULT_HRNET_CHECKPOINT,
    nlc_checkpoint=analyzer.DEFAULT_NLC_CHECKPOINT,
    recorded_checkpoint_audit=checkpoint,
    recorded_model_audit=model,
)
assert report["status"].endswith("_passed")
assert report["model_forwards"] == 0
assert report["cuda_initialized_after"] is False
assert model["parameter_count"] == 6_890_320
assert torch.cuda.is_initialized() is False
print(json.dumps({"status": report["status"], "parameters": model["parameter_count"]}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        env=os.environ.copy(),
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    report = json.loads(completed.stdout.strip().splitlines()[-1])
    assert report == {
        "status": "independent_cpu_structural_golden_passed",
        "parameters": 6_890_320,
    }


def test_independent_preprocess_matches_frozen_legacy(tmp_path: Path):
    if importlib.util.find_spec("imageio") is None:
        pytest.skip("exact preprocessing requires pinned imageio")
    rng = np.random.default_rng(20260726)
    pixels = rng.integers(0, 256, size=(17, 23, 3), dtype=np.uint8)
    path = tmp_path / "fixture.png"
    Image.fromarray(pixels, mode="RGB").save(path)
    actual, size, audit = analyzer._independent_preprocess(path)
    expected, expected_size, expected_audit = runner.legacy.preprocess_image(path)
    assert size == expected_size == (23, 17)
    assert np.array_equal(actual, expected)
    assert audit == expected_audit


def test_pairwise_distance_and_native_restore_match_torch_cpu():
    import torch
    import torch.nn.functional as functional

    rng = np.random.default_rng(42)
    embedding = rng.normal(size=(18, 256, 256)).astype(np.float32)
    center = rng.normal(size=(18,)).astype(np.float32)
    actual = analyzer._pairwise_distance_float32(embedding, center)
    vectors = torch.from_numpy(
        np.ascontiguousarray(embedding.transpose(1, 2, 0).reshape(-1, 18))
    )
    center_tensor = torch.from_numpy(center).reshape(1, 18)
    expected = (
        torch.nn.PairwiseDistance(p=2, eps=1e-6)(
            vectors,
            center_tensor,
        )
        .reshape(256, 256)
        .numpy()
    )
    assert float(np.max(np.abs(actual - expected))) <= (
        analyzer.DISTANCE_ABSOLUTE_TOLERANCE
    )
    restored = analyzer._bilinear_align_corners_false(
        actual,
        width=37,
        height=29,
    )
    torch_restored = functional.interpolate(
        torch.from_numpy(actual)[None, None],
        size=(29, 37),
        mode="bilinear",
        align_corners=False,
    )[0, 0].numpy()
    assert float(np.max(np.abs(restored - torch_restored))) <= (
        analyzer.NATIVE_RESTORE_ABSOLUTE_TOLERANCE
    )


def _score_row() -> dict:
    hierarchy = {
        name: np.zeros(classes, dtype=np.float32)
        for name, classes in runner.HIERARCHY_SPECS
    }
    hierarchy["out3_fine_14class"] = np.asarray(
        [0.0, 1.0, *([0.0] * 12)],
        dtype=np.float32,
    )
    probabilities = runner._stable_softmax(hierarchy["out3_fine_14class"])
    return runner._score_payload(hierarchy, probabilities)


def test_score_projection_is_fine14_and_fail_closed():
    row = _score_row()
    projection = analyzer._score_projection(row, sample_id="fixture")
    assert projection["ai_score"] > 0.5
    assert projection["official_fine_class_index"] == 1
    assert projection["official_binary_decision"] is True
    assert projection["hierarchy_logits"]["out3_fine_14class"].shape == (14,)
    tampered = json.loads(json.dumps(row))
    tampered["classification_hierarchy"]["out3_fine_14class"]["array_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="payload changed"):
        analyzer._score_projection(tampered, sample_id="fixture")


def test_analyzer_accepts_frozen_rank1447_cross_device_roundoff_boundary():
    logits = np.asarray(
        [
            14.364660263061523,
            -6.176663875579834,
            1.4939091205596924,
            0.36177706718444824,
            0.768934428691864,
            -7.829436302185059,
            -3.9948649406433105,
            -11.353510856628418,
            -1.9111732244491577,
            -1.954095482826233,
            -0.5619944334030151,
            -3.9558680057525635,
            -12.539756774902344,
            -2.1511006355285645,
        ],
        dtype=np.float32,
    )
    probabilities = np.asarray(
        [
            0.9999948740005493,
            1.1995375803763864e-09,
            2.572180846982519e-06,
            8.291306130558951e-07,
            1.2458018545657978e-06,
            2.2973360713773872e-10,
            1.0630583524573467e-08,
            6.772334429361315e-12,
            8.54069597266971e-08,
            8.181859811884351e-08,
            3.2918038073148637e-07,
            1.1053339576960752e-08,
            2.0680351962149013e-12,
            6.718838818642325e-08,
        ],
        dtype=np.float32,
    )
    hierarchy = {
        name: np.zeros(classes, dtype=np.float32)
        for name, classes in runner.HIERARCHY_SPECS
    }
    hierarchy["out3_fine_14class"] = logits
    row = runner._score_payload(hierarchy, probabilities)
    projection = analyzer._score_projection(
        row,
        sample_id="5724ab5a9da93056c640a537",
    )
    cpu = analyzer._stable_softmax_float32(logits)
    ulps = np.abs(
        probabilities.view(np.uint32).astype(np.int64)
        - cpu.view(np.uint32).astype(np.int64)
    )
    assert int(ulps.max()) == 7
    assert projection["ai_score"] == row["ai_score"]


@pytest.mark.parametrize(
    "values",
    [
        np.asarray(
            [
                0.0,
                np.nextafter(np.float32(2.3), np.float32(0.0)),
                np.float32(2.3),
                np.nextafter(np.float32(2.3), np.float32(10.0)),
                100.0,
            ],
            dtype=np.float32,
        ),
        np.linspace(0.0, 10.0, 10_001, dtype=np.float32),
    ],
)
def test_shared_t2_transform_preserves_inclusive_2_3_mask(values: np.ndarray):
    raw = values.reshape(1, -1)
    converted = analyzer._distance_to_shared_probability(raw)
    assert converted.dtype == np.float32
    assert np.array_equal(converted >= 0.5, raw >= np.float32(2.3))
    assert np.all(np.diff(converted.reshape(-1)) >= 0.0)
    with pytest.raises(ValueError, match="input changed"):
        analyzer._distance_to_shared_probability(np.asarray([[-1.0]], dtype=np.float32))


def _minimal_row(condition: str) -> dict:
    table = {
        "real": ("real", "real", "authentic", 0, "all_zero", "a" * 24),
        "local_cat": (
            "forged",
            "local_splice",
            "local_insertion",
            1,
            "exact_diff",
            "b" * 24,
        ),
        "fullframe_cat": (
            "forged",
            "full_frame_conditional_edit",
            "conditional_full_frame_edit",
            1,
            "not_applicable",
            "c" * 24,
        ),
    }
    kind, family, scope, label, gt_kind, sample_id = table[condition]
    return {
        "schema_version": BALANCED_SCHEMA,
        "dataset_id": BALANCED_DATASET_ID,
        "rank": 0,
        "sample_id": sample_id,
        "condition": condition,
        "condition_family": family,
        "manipulation_scope": scope,
        "normalized_task_id": "fixture_task",
        "task_id": "fixture_task",
        "kind": kind,
        "label": label,
        "domain": "lodging",
        "source_content_cluster": "fixture_cluster",
        "gt_mask_kind": gt_kind,
        "gt_mask_path": None,
        "gt_mask_sha256": None,
        "gt_positive_pixels": 0 if gt_kind == "all_zero" else None,
        "canonical_path": f"outputs/fixture/{sample_id}.jpg",
        "canonical_sha256": "1" * 64,
        "width": 8,
        "height": 6,
        "panel": True,
        "selection_rank": 0,
    }


def _applicable_artifact_row(
    tmp_path: Path,
) -> tuple[dict, dict, Path, np.ndarray]:
    expected = _minimal_row("real")
    artifact_root = tmp_path / "outputs/opensource/hifi_ifdl/test"
    runner._prepare_artifact_root(artifact_root)
    paths = runner.artifact_paths(artifact_root, expected["sample_id"])
    center = np.linspace(-0.2, 0.2, 18, dtype=np.float32)
    embedding = np.zeros((18, 256, 256), dtype=np.float32)
    model_distance = analyzer._pairwise_distance_float32(embedding, center)
    native_distance = analyzer._bilinear_align_corners_false(
        model_distance,
        width=8,
        height=6,
    )
    runner.legacy._atomic_save_npy(paths["embedding_model_256"], embedding)
    runner.legacy._atomic_save_npy(paths["distance_model_256"], model_distance)
    runner.legacy._atomic_save_npy(paths["distance_native"], native_distance)
    runner.legacy._atomic_save_mask(
        paths["native_mask"],
        native_distance >= np.float32(2.3),
    )
    fields = runner._artifact_fields(
        repo_root=tmp_path,
        paths=paths,
        embedding=embedding,
        distance_model=model_distance,
        distance_native=native_distance,
    )
    row = {
        "sample_id": expected["sample_id"],
        "status": "ok",
        "t2_applicable": True,
        "t2_target_semantics": "all_zero_real_false_positive_area",
        **fields,
        "localization": runner._localization_payload(
            row=expected,
            repo_root=tmp_path,
            model_map=model_distance,
            native_map=native_distance,
        ),
    }
    return expected, row, artifact_root, center


def test_persisted_artifacts_replay_embedding_distance_native_and_mask(
    tmp_path: Path,
):
    expected, row, artifact_root, center = _applicable_artifact_row(tmp_path)
    artifact = analyzer._validate_artifact_row(
        row,
        expected=expected,
        repo_root=tmp_path,
        artifact_root=artifact_root,
        center=center,
    )
    assert artifact is not None
    assert artifact.width == 8
    assert artifact.height == 6
    tampered = dict(row)
    tampered["localization"] = json.loads(json.dumps(row["localization"]))
    tampered["localization"]["native"]["score_max"] += 0.1
    with pytest.raises(ValueError, match="numeric value changed"):
        analyzer._validate_artifact_row(
            tampered,
            expected=expected,
            repo_root=tmp_path,
            artifact_root=artifact_root,
            center=center,
        )


def test_inventory_is_1025_style_applicable_only_and_fullframe_na(
    tmp_path: Path,
):
    expected, real_row, artifact_root, center = _applicable_artifact_row(tmp_path)
    fullframe = _minimal_row("fullframe_cat")
    fullframe_row = {
        "sample_id": fullframe["sample_id"],
        "status": "ok",
        "t2_applicable": False,
        "t2_target_semantics": "not_applicable_fullframe",
        **runner._not_applicable_artifact_fields(),
        "localization": None,
    }
    artifacts = analyzer.validate_artifact_inventory(
        latest_results=(real_row, fullframe_row),
        selected=(expected, fullframe),
        repo_root=tmp_path,
        artifact_root=artifact_root,
        center=center,
    )
    assert set(artifacts) == {expected["sample_id"]}
    assert (
        analyzer._validate_artifact_row(
            fullframe_row,
            expected=fullframe,
            repo_root=tmp_path,
            artifact_root=artifact_root,
            center=center,
        )
        is None
    )
    (artifact_root / "masks_native/extra.png").write_bytes(b"x")
    with pytest.raises(ValueError, match="inventory mismatch"):
        analyzer.validate_artifact_inventory(
            latest_results=(real_row, fullframe_row),
            selected=(expected, fullframe),
            repo_root=tmp_path,
            artifact_root=artifact_root,
            center=center,
        )


def test_result_attempt_independently_replays_preprocess_and_head(tmp_path: Path):
    if importlib.util.find_spec("imageio") is None:
        pytest.skip("exact preprocessing requires pinned imageio")
    expected = _minimal_row("fullframe_cat")
    input_path = tmp_path / expected["canonical_path"]
    input_path.parent.mkdir(parents=True)
    Image.fromarray(np.zeros((6, 8, 3), dtype=np.uint8), mode="RGB").save(input_path)
    expected["canonical_sha256"] = runner.sha256_file(input_path)
    _, _, preprocess = analyzer._independent_preprocess(input_path)
    identity = runner.result_identity(
        expected,
        run_id="fixture-run",
        run_manifest_fingerprint="f" * 64,
        valid_for_metrics=True,
    )
    result = {
        **identity,
        "status": "ok",
        "completed_at": "2026-07-27T00:00:00+00:00",
        "preprocess": preprocess,
        **_score_row(),
        "auxiliary_learned_mask": {
            "shape": [256, 256],
            "dtype": "float32",
            "minimum": 0.25,
            "maximum": 0.25,
            "mean": 0.25,
            "primary_output": False,
            "reason": (
                "the official public localize API ignores this sigmoid mask "
                "and thresholds hypersphere distance instead"
            ),
        },
        **runner._not_applicable_artifact_fields(),
        "mask_threshold": 2.3,
        "mask_threshold_operator": ">=",
        "localization": None,
        "latency_ms": 1.0,
        "peak_cuda_memory_bytes": 0,
    }
    analyzer._validate_attempt(
        result,
        expected=expected,
        repo_root=tmp_path,
        run_id="fixture-run",
        fingerprint="f" * 64,
    )
    result["classification_threshold_operator"] = ">="
    with pytest.raises(ValueError, match="classification_threshold_operator"):
        analyzer._validate_attempt(
            result,
            expected=expected,
            repo_root=tmp_path,
            run_id="fixture-run",
            fingerprint="f" * 64,
        )


def test_history_allows_recovery_but_no_attempt_after_success():
    selected = [_minimal_row("real")]
    sample_id = selected[0]["sample_id"]
    history = analyzer._validate_history(
        selected,
        [
            {"sample_id": sample_id, "status": "error"},
            {"sample_id": sample_id, "status": "ok"},
        ],
    )
    assert history["recovered_error_to_ok"] == 1
    with pytest.raises(ValueError, match="after success"):
        analyzer._validate_history(
            selected,
            [
                {"sample_id": sample_id, "status": "ok"},
                {"sample_id": sample_id, "status": "error"},
            ],
        )


def _cpu_runtime() -> dict:
    body = {
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
        "torch_version": "fixture",
        "torch_cuda_version": None,
    }
    return {**body, "contract_sha256": analyzer._fingerprint(body)}


def test_runtime_and_output_paths_fail_closed(tmp_path: Path):
    assert analyzer._validate_runtime(_cpu_runtime(), label="runtime")["device"] == (
        "cpu"
    )
    changed = _cpu_runtime()
    changed["seed"] += 1
    with pytest.raises(ValueError, match="seed changed"):
        analyzer._validate_runtime(changed, label="runtime")
    result_root = tmp_path / "results/opensource/hifi_ifdl"
    result_root.mkdir(parents=True)
    with pytest.raises(ValueError, match="must be exactly"):
        analyzer._safe_standard_root(
            tmp_path / "other",
            repo_root=tmp_path,
            expected_relative=Path("results/opensource/hifi_ifdl"),
            label="results",
        )


def test_strict_json_rejects_duplicate_keys_and_noncanonical_jsonl(tmp_path: Path):
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"a":1,"a":2}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        analyzer._load_json(duplicate, "duplicate")
    ledger = tmp_path / "ledger.jsonl"
    ledger.write_text('{"b": 2, "a": 1}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="canonical"):
        analyzer._read_jsonl(ledger, "ledger")


def test_metrics_use_shared_reducers_with_frozen_seed(monkeypatch):
    calls: dict[str, dict] = {}

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
                "native_maps_evaluated": analyzer.FORMAL_T2_IMAGES,
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
        selected=tuple({"sample_id": str(index)} for index in range(1_775)),
        artifacts={str(index): object() for index in range(1_025)},
        latest_results=(),
        release=SimpleNamespace(
            inputs=(),
            panel=(),
            source_pairs=(),
            repo_root=REPO_ROOT,
        ),
        run_id="fixture",
        fingerprint="f" * 64,
        contract=object(),
    )
    metrics = analyzer.recompute_metrics(bundle)
    assert metrics["bootstrap_iterations"] == 1_000
    assert metrics["bootstrap_seed"] == 20_260_726
    assert metrics["primary_t1"]["source"] == "fine_14class_head"
    assert metrics["primary_t2"]["threshold"] == 2.3
    assert metrics["shared_t2_compatibility"]["transform"] == "p=d/(d+2.3)"
    assert calls["t1"]["iterations"] == 1_000
    assert calls["t2"]["threshold"] == 0.5
    assert calls["t2"]["threshold_operator"] == ">="
    with pytest.raises(ValueError, match="iterations=1000"):
        analyzer.recompute_metrics(bundle, iterations=10)


def test_smoke_projection_ignores_only_run_specific_paths_and_timing():
    base = {
        "sample_id": "a" * 24,
        "run_id": "a",
        "run_manifest_fingerprint": "1" * 64,
        "completed_at": "first",
        "latency_ms": 1.0,
        "peak_cuda_memory_bytes": 2,
        "artifact_paths": {"embedding_model_256": "a/path"},
        "embedding_artifact": {
            "path": "a/path",
            "sha256": "f" * 64,
            "bytes": 4,
        },
        "distance_model_artifact": None,
        "distance_native_artifact": None,
        "score_map_path": "a/native",
        "mask_path": "a/mask",
        "ai_score": 0.75,
    }
    replay = json.loads(json.dumps(base))
    replay.update(
        {
            "run_id": "b",
            "run_manifest_fingerprint": "2" * 64,
            "completed_at": "second",
            "latency_ms": 9.0,
            "peak_cuda_memory_bytes": 10,
            "artifact_paths": {"embedding_model_256": "b/path"},
            "score_map_path": "b/native",
            "mask_path": "b/mask",
        }
    )
    replay["embedding_artifact"]["path"] = "b/path"
    assert analyzer._smoke_result_projection(base) == (
        analyzer._smoke_result_projection(replay)
    )
    replay["ai_score"] = 0.5
    assert analyzer._smoke_result_projection(base) != (
        analyzer._smoke_result_projection(replay)
    )


def test_verified_json_write_and_frozen_comparison_path(tmp_path: Path):
    path = tmp_path / "report.json"
    value = {"schema_version": "fixture", "status": "passed"}
    digest = analyzer._write_json_verified(path, value)
    assert digest == runner.sha256_file(path)
    assert json.loads(path.read_text()) == value
    expected = analyzer._comparison_output_path(
        repo_root=tmp_path,
        first_run_id=analyzer.DEFAULT_SMOKE_RUN_ID_A,
        second_run_id=analyzer.DEFAULT_SMOKE_RUN_ID_B,
    )
    assert expected.name.endswith("_comparison.json")
    unsafe = tmp_path / "unsafe.json"
    target = tmp_path / "target"
    target.mkdir()
    unsafe.symlink_to(target)
    with pytest.raises(ValueError, match="symlink"):
        analyzer._write_json_verified(unsafe, value)
