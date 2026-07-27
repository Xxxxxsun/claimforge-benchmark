from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from eval.opensource import analyze_psccnet_balanced as analyzer
from eval.opensource import run_psccnet_balanced as runner
from eval.opensource.canonical_release import load_canonical_release
from eval.opensource.common import repo_relative, sha256_file


REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_MANIFEST = Path("outputs/opensource/balanced250_v1/manifest.json")
RUNNER_SHA256 = "0d96b086267d8eb26443455f90d81160e9b31e681609812f10e44eef71219e39"


def _runtime(device: str = "cuda:0") -> dict:
    value: dict = {
        "device": device,
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
        "torch_cuda_version": "12.8",
    }
    if device.startswith("cuda:"):
        value["cuda"] = {
            "logical_device_index": int(device[5:]),
            "device_name": "fixture GPU",
            "total_memory_bytes": 1,
            "compute_capability": [9, 0],
        }
    return {**value, "contract_sha256": analyzer._fingerprint(value)}


def _input_row(root: Path, *, applicable: bool) -> dict:
    sample_id = "a" * 24 if applicable else "b" * 24
    image_path = root / "outputs/fixture/images" / f"{sample_id}.png"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    pixels = np.asarray(
        [
            [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            [[10, 11, 12], [13, 14, 15], [16, 17, 18]],
        ],
        dtype=np.uint8,
    )
    Image.fromarray(pixels, mode="RGB").save(image_path)
    return {
        "sample_id": sample_id,
        "canonical_path": repo_relative(image_path, root),
        "width": 3,
        "height": 2,
        "condition": "real" if applicable else "fullframe_cat",
        "kind": "real" if applicable else "forged",
        "label": 0 if applicable else 1,
        "manipulation_scope": (
            "authentic" if applicable else "conditional_full_frame_edit"
        ),
        "gt_mask_kind": "all_zero" if applicable else "not_applicable",
        "gt_mask_path": None,
        "gt_mask_sha256": None,
        "gt_positive_pixels": 0 if applicable else None,
    }


def _artifact_fixture(
    root: Path,
    *,
    applicable: bool,
) -> tuple[dict, dict, Path, analyzer.DenseArtifacts]:
    input_row = _input_row(root, applicable=applicable)
    sample_id = input_row["sample_id"]
    artifact_root = root / "outputs/opensource/psccnet/fixture"
    runner._prepare_artifact_root(artifact_root)
    paths = runner.artifact_paths(artifact_root, sample_id)
    progressive = [
        np.full(shape, np.float32(0.25), dtype=np.float32)
        for shape in runner.PROGRESSIVE_SHAPES
    ]
    native = analyzer._bilinear_align_corners_true(
        progressive[0],
        width=input_row["width"],
        height=input_row["height"],
    )
    for stage, array in enumerate(progressive, start=1):
        runner.legacy._atomic_save_npy(
            paths[f"progressive_mask{stage}"],
            array,
        )
    runner.legacy._atomic_save_npy(paths["native_probability"], native)
    mask_path = None
    if applicable:
        mask_path = paths["native_mask"]
        runner.legacy._atomic_save_mask(mask_path, native > analyzer.MASK_THRESHOLD)
    _, preprocess = analyzer._independent_preprocess_tensor(
        root / input_row["canonical_path"]
    )
    logits = np.asarray([0.0, 0.0], dtype=np.float32)
    probabilities = np.asarray([0.5, 0.5], dtype=np.float32)
    row = {
        "preprocess": preprocess,
        **runner._score_payload(logits, probabilities),
        **runner._artifact_fields(
            repo_root=root,
            paths=paths,
            model_masks=progressive,
            native_map=native,
            mask_path=mask_path,
        ),
        "mask_threshold": analyzer.MASK_THRESHOLD,
        "mask_threshold_operator": analyzer.THRESHOLD_OPERATOR,
        "localization": (
            runner._localization_payload(
                row=input_row,
                repo_root=root,
                model_map=progressive[0],
                native_map=native,
            )
            if applicable
            else None
        ),
    }
    artifact = analyzer._validate_artifact_row(
        row=row,
        input_row=input_row,
        repo_root=root,
        artifact_root=artifact_root,
    )
    return row, input_row, artifact_root, artifact


def _comparison_fixture(
    root: Path,
) -> tuple[list[dict], list[dict], dict, dict]:
    left_file = root / "left.bin"
    right_file = root / "right.bin"
    left_file.write_bytes(b"exact")
    right_file.write_bytes(b"exact")
    left_rows: list[dict] = []
    right_rows: list[dict] = []
    left_artifacts: dict[str, analyzer.DenseArtifacts] = {}
    right_artifacts: dict[str, analyzer.DenseArtifacts] = {}
    for index in range(analyzer.SMOKE_IMAGES):
        sample_id = f"sample-{index:02d}"
        applicable = index < analyzer.SMOKE_T2_IMAGES
        progressive_left = tuple(left_file for _ in range(4))
        progressive_right = tuple(right_file for _ in range(4))
        base = {
            "sample_id": sample_id,
            "ai_score": 0.25,
            "run_id": "A",
            "run_manifest_fingerprint": "a" * 64,
            "completed_at": "a",
            "latency_ms": 1.0,
            "peak_cuda_memory_bytes": 1,
            "primary_model_score_map_path": "a",
            "score_map_path": "a",
            "mask_path": "a" if applicable else None,
            "artifact_paths": {f"progressive_mask{stage}": "a" for stage in range(1, 5)}
            | {
                "native_probability": "a",
                "native_mask": "a" if applicable else None,
            },
            "progressive_maps": [
                {"stage": stage, "path": "a"} for stage in range(1, 5)
            ],
        }
        other = json.loads(json.dumps(base))
        other.update(
            {
                "run_id": "B",
                "run_manifest_fingerprint": "b" * 64,
                "completed_at": "b",
                "latency_ms": 2.0,
                "peak_cuda_memory_bytes": 2,
                "primary_model_score_map_path": "b",
                "score_map_path": "b",
                "mask_path": "b" if applicable else None,
            }
        )
        for key, value in other["artifact_paths"].items():
            if value is not None:
                other["artifact_paths"][key] = "b"
        for item in other["progressive_maps"]:
            item["path"] = "b"
        left_rows.append(base)
        right_rows.append(other)
        common = {
            "sample_id": sample_id,
            "progressive_file_sha256": ("0" * 64,) * 4,
            "progressive_array_sha256": ("1" * 64,) * 4,
            "native_file_sha256": "2" * 64,
            "native_array_sha256": "3" * 64,
            "mask_file_sha256": "4" * 64 if applicable else None,
            "mask_array_sha256": "5" * 64 if applicable else None,
            "t2_applicable": applicable,
            "width": 1,
            "height": 1,
            "static_softmax_max_abs_diff": 0.0,
            "static_native_max_abs_diff": 0.0,
        }
        left_artifacts[sample_id] = analyzer.DenseArtifacts(
            progressive_paths=progressive_left,
            native_path=left_file,
            mask_path=left_file if applicable else None,
            **common,
        )
        right_artifacts[sample_id] = analyzer.DenseArtifacts(
            progressive_paths=progressive_right,
            native_path=right_file,
            mask_path=right_file if applicable else None,
            **common,
        )
    return left_rows, right_rows, left_artifacts, right_artifacts


def test_import_does_not_initialize_cuda_in_fresh_process():
    code = (
        "import torch; assert not torch.cuda.is_initialized(); "
        "import eval.opensource.analyze_psccnet_balanced; "
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
    assert sha256_file(REPO_ROOT / "eval/opensource/run_psccnet_balanced.py") == (
        RUNNER_SHA256
    )


def test_frozen_formal_and_smoke_selections_are_exact():
    release = load_canonical_release(
        REPO_ROOT,
        DATASET_MANIFEST,
        verify_files=False,
    )
    _, formal = analyzer._selection_for_mode(release, "formal")
    _, smoke = analyzer._selection_for_mode(release, "smoke")
    assert len(formal) == analyzer.FORMAL_IMAGES == 1_775
    assert len(smoke) == analyzer.SMOKE_IMAGES == 35
    assert analyzer._rows_sha256(formal) == analyzer.FORMAL_SELECTED_ROWS_SHA256
    assert analyzer._rows_sha256(smoke) == analyzer.SMOKE_SELECTED_ROWS_SHA256
    assert (
        sum(row["gt_mask_kind"] in analyzer._T2_GT_KINDS for row in formal)
        == analyzer.FORMAL_T2_IMAGES
        == 1_025
    )
    assert (
        sum(row["gt_mask_kind"] in analyzer._T2_GT_KINDS for row in smoke)
        == analyzer.SMOKE_T2_IMAGES
        == 20
    )


def test_runtime_contract_is_hashed_and_device_bound():
    runtime = _runtime()
    assert analyzer._validate_runtime(runtime, label="fixture") == runtime
    runtime["cudnn_deterministic"] = False
    with pytest.raises(ValueError, match="cudnn_deterministic"):
        analyzer._validate_runtime(runtime, label="fixture")
    with pytest.raises(ValueError, match="device"):
        analyzer._configure_recorded_runtime(
            recorded=_runtime(),
            device_text="cuda:1",
        )


def test_independent_preprocess_matches_official_rgb_tensor(tmp_path: Path):
    row = _input_row(tmp_path, applicable=True)
    path = tmp_path / row["canonical_path"]
    independent, audit = analyzer._independent_preprocess_tensor(path)
    official, native_size, official_audit = runner.legacy.preprocess_image(path)
    assert native_size == (3, 2)
    assert np.array_equal(independent, official)
    assert audit == official_audit
    assert independent.dtype == np.float32


def test_float32_softmax_and_strict_half_threshold_are_frozen():
    logits = np.asarray([0.0, 0.0], dtype=np.float32)
    probabilities = np.asarray([0.5, 0.5], dtype=np.float32)
    payload = runner._score_payload(logits, probabilities)
    assert analyzer._validate_score_payload(payload, sample_id="fixture") == 0.0
    assert payload["classification_decision"] == "authentic"
    payload["classification_decision"] = "forged"
    with pytest.raises(ValueError, match="score semantics"):
        analyzer._validate_score_payload(payload, sample_id="fixture")


def test_independent_bilinear_align_corners_true_preserves_corners():
    source = np.asarray([[0.0, 1.0], [0.25, 0.75]], dtype=np.float32)
    restored = analyzer._bilinear_align_corners_true(
        source,
        width=5,
        height=3,
    )
    assert restored.shape == (3, 5)
    assert restored.dtype == np.float32
    assert restored[0, 0] == source[0, 0]
    assert restored[0, -1] == source[0, -1]
    assert restored[-1, 0] == source[-1, 0]
    assert restored[-1, -1] == source[-1, -1]


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


def test_applicable_artifacts_replay_all_four_maps_native_and_mask(tmp_path: Path):
    _, _, _, artifact = _artifact_fixture(tmp_path, applicable=True)
    assert len(artifact.progressive_paths) == 4
    assert artifact.t2_applicable is True
    assert artifact.mask_path is not None
    assert artifact.static_softmax_max_abs_diff == 0.0
    assert artifact.static_native_max_abs_diff == 0.0


def test_fullframe_keeps_dense_maps_but_has_no_t2_claim(tmp_path: Path):
    row, input_row, artifact_root, artifact = _artifact_fixture(
        tmp_path,
        applicable=False,
    )
    assert artifact.t2_applicable is False
    assert artifact.mask_path is None
    row["localization"] = {"native": {}}
    with pytest.raises(ValueError, match="claims T2"):
        analyzer._validate_artifact_row(
            row=row,
            input_row=input_row,
            repo_root=tmp_path,
            artifact_root=artifact_root,
        )


def test_artifact_hash_and_inventory_are_fail_closed(tmp_path: Path):
    row, input_row, artifact_root, artifact = _artifact_fixture(
        tmp_path,
        applicable=True,
    )
    inventory = analyzer.validate_artifact_inventory(
        artifact_root=artifact_root,
        selected=[input_row],
        artifacts={input_row["sample_id"]: artifact},
    )
    assert inventory == {
        "progressive_mask1": 1,
        "progressive_mask2": 1,
        "progressive_mask3": 1,
        "progressive_mask4": 1,
        "score_maps_native": 1,
        "masks_native": 1,
    }
    row["score_map_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="persisted file changed"):
        analyzer._validate_artifact_row(
            row=row,
            input_row=input_row,
            repo_root=tmp_path,
            artifact_root=artifact_root,
        )
    (artifact_root / "masks_native" / "extra.png").write_bytes(b"x")
    with pytest.raises(ValueError, match="inventory changed"):
        analyzer.validate_artifact_inventory(
            artifact_root=artifact_root,
            selected=[input_row],
            artifacts={input_row["sample_id"]: artifact},
        )


def test_attempt_history_allows_recovery_but_success_is_terminal():
    selected = [{"sample_id": "first"}, {"sample_id": "second"}]
    attempts = [
        {"sample_id": "first", "status": "error"},
        {"sample_id": "first", "status": "ok"},
        {"sample_id": "second", "status": "ok"},
    ]
    report = analyzer._validate_attempt_history(selected, attempts)
    assert report["recovered_error_to_ok"] == 1
    attempts.append({"sample_id": "first", "status": "error"})
    with pytest.raises(ValueError, match="after terminal success"):
        analyzer._validate_attempt_history(selected, attempts)


def test_smoke_projection_normalizes_nested_run_specific_paths(tmp_path: Path):
    left, right, _, _ = _comparison_fixture(tmp_path)
    assert analyzer._smoke_projection(left[0]) == analyzer._smoke_projection(right[0])
    left_immutable = {
        "run_id": "A",
        "outputs": {"results_path": "A/results.jsonl", "artifact_root": "A"},
        "scientific": {"fixed": True},
    }
    right_immutable = {
        "run_id": "B",
        "outputs": {"results_path": "B/results.jsonl", "artifact_root": "B"},
        "scientific": {"fixed": True},
    }
    assert analyzer._smoke_immutable_projection(
        left_immutable
    ) == analyzer._smoke_immutable_projection(right_immutable)


def test_smoke_comparison_is_byte_exact_except_run_fields(tmp_path: Path):
    left, right, left_artifacts, right_artifacts = _comparison_fixture(tmp_path)
    report = analyzer.compare_computational_results(
        left,
        right,
        reference_artifacts=left_artifacts,
        replay_artifacts=right_artifacts,
    )
    assert report["images_compared"] == 35
    assert report["artifact_files_compared_exact"] == 195
    right[7]["ai_score"] = 0.5
    with pytest.raises(ValueError, match="computational row"):
        analyzer.compare_computational_results(
            left,
            right,
            reference_artifacts=left_artifacts,
            replay_artifacts=right_artifacts,
        )


def test_recompute_metrics_adapts_only_shared_t2_to_frozen_gte_operator(
    monkeypatch: pytest.MonkeyPatch,
):
    selected = [
        {
            "sample_id": str(index),
            "gt_mask_kind": "all_zero" if index < 1025 else "not_applicable",
        }
        for index in range(1775)
    ]
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
            "coverage": {"is_complete": True, "native_maps_evaluated": 1025},
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
    assert calls["t1"]["iterations"] == analyzer.BOOTSTRAP_ITERATIONS
    assert calls["t1"]["seed"] == analyzer.BOOTSTRAP_SEED
    assert calls["t2"]["threshold"] == 0.5
    assert calls["t2"]["threshold_operator"] == ">="
    assert report["formal_images_t1"] == 1775
    assert report["formal_images_t2"] == 1025
    assert report["mask_threshold_operator"] == ">"
    assert report["official_t2_threshold_operator"] == ">"
    assert report["shared_t2_threshold_operator"] == ">="
    assert report["shared_t2_operator_equivalent_to_official"] is False
    exact_half = np.asarray([0.5], dtype=np.float32)
    assert not bool(np.any(exact_half > analyzer.MASK_THRESHOLD))
    assert bool(np.any(exact_half >= analyzer.MASK_THRESHOLD))


def test_native_map_loader_matches_shared_two_row_callback(tmp_path: Path):
    _, input_row, _, artifact = _artifact_fixture(tmp_path, applicable=True)
    sample_id = input_row["sample_id"]
    bundle = SimpleNamespace(
        selected=[input_row],
        artifacts={sample_id: artifact},
    )
    callback = analyzer._native_map_loader(bundle)
    loaded = callback(input_row, {"sample_id": sample_id})
    expected = np.load(artifact.native_path, allow_pickle=False)
    assert np.array_equal(loaded, expected)

    with pytest.raises(ValueError, match="result identity"):
        callback(input_row, {"sample_id": "different"})


def test_parser_defaults_to_formal_fresh_replay_and_frozen_seed():
    args = analyzer._build_parser().parse_args([])
    assert args.run_id == analyzer.DEFAULT_FORMAL_RUN_ID
    assert args.device == "cuda:0"
    assert args.skip_model_replay is False
    assert args.bootstrap_iterations == 1000
    assert args.bootstrap_seed == 20260726


def test_strict_json_and_verified_output_are_fail_closed(tmp_path: Path):
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"x":1,"x":2}', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        analyzer._load_json(duplicate, "duplicate")
    nonfinite = tmp_path / "nonfinite.json"
    nonfinite.write_text('{"x":NaN}', encoding="utf-8")
    with pytest.raises(ValueError, match="non-finite"):
        analyzer._load_json(nonfinite, "nonfinite")
    output = tmp_path / "verified.json"
    value = {"schema_version": "fixture", "nested": {"value": 1}}
    analyzer._write_json_verified(output, value, label="fixture")
    assert sha256_file(output) == analyzer._json_sha256(value)


def test_license_and_scientific_boundaries_are_explicit():
    assert runner.LICENSE_RECORD["project_license"]["spdx"] == "MIT"
    assert runner.LICENSE_RECORD["project_license"]["commercial_use_permission"] is True
    assert (
        runner.LICENSE_RECORD["limitations"][
            "benchmark_use_does_not_establish_product_clearance"
        ]
        is True
    )
    assert runner.T2_SPEC["not_applicable_conditions"] == [
        "fullframe_mouse",
        "fullframe_cat",
        "fullframe_trash_can",
    ]
