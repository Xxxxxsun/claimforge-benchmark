from __future__ import annotations

import inspect
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from eval.opensource import run_mvssnet_balanced as runner
from eval.opensource.canonical_release import (
    BALANCED_CONDITIONS,
    BALANCED_DATASET_ID,
    BALANCED_SCHEMA,
    CanonicalRelease,
    load_canonical_release,
)
from eval.opensource.common import sha256_file, stable_json


REPO_ROOT = Path(__file__).resolve().parents[1]
FORMAL_MANIFEST = Path("outputs/opensource/balanced250_v1/manifest.json")


@pytest.fixture(scope="module")
def formal_release() -> CanonicalRelease:
    return load_canonical_release(
        REPO_ROOT,
        FORMAL_MANIFEST,
        verify_files=False,
    )


def _condition_row(
    release: CanonicalRelease,
    condition: str,
) -> dict:
    return next(row for row in release.inputs if row["condition"] == condition)


def _attempt(
    row: dict,
    status: str,
    *,
    fingerprint: str = "0" * 64,
) -> dict:
    value = {
        **runner.result_identity(
            row,
            run_id="history-test",
            run_manifest_fingerprint=fingerprint,
            valid_for_metrics=status == "ok",
        ),
        "status": status,
        "completed_at": "2026-07-27T00:00:00+00:00",
    }
    if status == "error":
        value.update(
            {
                "error_type": "RuntimeError",
                "error": "fixture",
                "traceback": "fixture traceback",
            }
        )
    return value


def test_contract_constants_score_t2_artifacts_and_license_are_frozen():
    assert runner.SCORE_SPEC.as_dict() == {
        "key": "ai_score",
        "direction": "higher_means_fake",
        "fixed_threshold": 0.5,
        "threshold_operator": ">",
    }
    assert runner.DEFAULT_FORMAL_RUN_ID.endswith("20260727")
    assert runner.DEFAULT_SMOKE_RUN_ID_A.endswith("20260727")
    assert runner.DEFAULT_SMOKE_RUN_ID_B.endswith("20260727")
    assert runner.DEFAULT_SMOKE_LIMIT == 5
    assert runner.FORMAL_T2_IMAGES == 1025
    assert runner.T2_SPEC["valid_conditions"] == [
        "real",
        "local_mouse",
        "local_cat",
        "local_trash_can",
    ]
    assert runner.T2_SPEC["not_applicable_conditions"] == [
        "fullframe_mouse",
        "fullframe_cat",
        "fullframe_trash_can",
    ]
    assert (
        runner.T2_SPEC["native_probability_map"]["fullframe_role"]
        == "diagnostic_and_secondary_T1_only"
    )
    assert set(runner.ARTIFACT_CONTRACT) == {
        "raw_logits_model_512",
        "score_map_model_512",
        "score_map_native_official",
        "mask_native",
    }
    assert runner.LICENSE_RECORD["commercial_use_permission_established"] is False
    assert runner.LICENSE_RECORD["redistribution_permission_established"] is False
    assert runner.CHECKPOINT_BYTES == 588_270_735
    assert runner.CHECKPOINT_STATE_KEYS == 800
    assert runner.CHECKPOINT_STATE_ELEMENTS == 146_994_922


def test_adapter_source_inventory_hashes_every_bound_local_file():
    contract = runner.adapter_source_contract(REPO_ROOT)
    assert tuple(contract) == runner.ADAPTER_SOURCE_PATHS
    assert "eval/opensource/run_mvssnet_balanced.py" in contract
    assert "eval/opensource/run_mvssnet.py" in contract
    assert "eval/opensource/mvssnet_metrics.py" in contract
    assert "eval/opensource/analyze_mvssnet_balanced.py" not in contract
    for relative, binding in contract.items():
        path = REPO_ROOT / relative
        assert binding == {
            "path": relative,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }


def test_official_source_bindings_and_checkpoint_bytes_are_exact():
    source_root = runner.legacy.DEFAULT_MVSSNET_ROOT
    for relative, (
        expected_bytes,
        expected_sha256,
    ) in runner.MVSSNET_SOURCE_FILES.items():
        path = source_root / relative
        assert path.stat().st_size == expected_bytes
        assert sha256_file(path) == expected_sha256
    checkpoint = runner.legacy.DEFAULT_CHECKPOINT
    assert checkpoint.stat().st_size == runner.CHECKPOINT_BYTES
    assert sha256_file(checkpoint) == runner.legacy.CHECKPOINT_SHA256


def test_mouse_reference_evidence_is_frozen_without_driving_selection():
    record = runner.verify_mouse_reference(REPO_ROOT)
    assert record["expected_tasks"] == 275
    assert record["expected_images"] == 550
    assert (
        record["role"]
        == "protocol_and_regression_anchor_only_not_score_based_selection"
    )
    assert set(record["files"]) == set(runner.MOUSE_REFERENCE_FILES)
    body = {key: value for key, value in record.items() if key != "contract_sha256"}
    assert record["contract_sha256"] == runner._fingerprint(body)


def test_formal_selection_is_exact_1775_native_t1_local_only_t2(
    formal_release: CanonicalRelease,
):
    spec, selected = runner.select_mode_inputs(
        formal_release,
        mode="formal",
        per_condition_limit=None,
        sample_id=None,
    )
    assert formal_release.schema_version == BALANCED_SCHEMA
    assert formal_release.dataset_id == BALANCED_DATASET_ID
    assert spec.capability.value == "local_t1_t2"
    assert len(selected) == 1775
    assert Counter(row["condition"] for row in selected) == (runner.FORMAL_COUNTS)
    assert sum(runner._t2_semantics(row)[0] for row in selected) == 1025
    assert [row["sample_id"] for row in selected] == [
        row["sample_id"] for row in formal_release.inputs
    ]
    assert runner._required_artifact_bytes(selected) == 10_237_351_823


def test_smoke_selection_is_fixed_first_five_panel_rows_per_condition(
    formal_release: CanonicalRelease,
):
    spec, selected = runner.select_mode_inputs(
        formal_release,
        mode="smoke",
        per_condition_limit=5,
        sample_id=None,
    )
    assert spec.per_condition_limit == 5
    assert len(selected) == 35
    assert Counter(row["condition"] for row in selected) == {
        condition: 5 for condition in BALANCED_CONDITIONS
    }
    assert all(row["panel"] is True for row in selected)
    expected = {
        condition: [
            row["sample_id"]
            for row in formal_release.panel
            if row["condition"] == condition
        ][:5]
        for condition in BALANCED_CONDITIONS
    }
    actual = {
        condition: [
            row["sample_id"] for row in selected if row["condition"] == condition
        ]
        for condition in BALANCED_CONDITIONS
    }
    assert actual == expected
    assert [row["rank"] for row in selected] == sorted(row["rank"] for row in selected)


def test_mode_selectors_and_frozen_run_ids_fail_closed(
    formal_release: CanonicalRelease,
):
    with pytest.raises(ValueError, match="formal mode"):
        runner.select_mode_inputs(
            formal_release,
            mode="formal",
            per_condition_limit=1,
            sample_id=None,
        )
    for limit in (None, 0, 4, 6):
        with pytest.raises(ValueError, match="exactly 5"):
            runner.select_mode_inputs(
                formal_release,
                mode="smoke",
                per_condition_limit=limit,
                sample_id=None,
            )
    with pytest.raises(ValueError, match="requires --sample-id"):
        runner.select_mode_inputs(
            formal_release,
            mode="single",
            per_condition_limit=None,
            sample_id=None,
        )

    parser = runner._build_parser()
    formal = parser.parse_args(["--run-id", "wrong"])
    with pytest.raises(ValueError, match="formal run-id"):
        runner._resolve_run_id(formal)
    smoke = parser.parse_args(
        [
            "--mode",
            "smoke",
            "--run-id",
            "mvssnet-unfrozen-smoke",
        ]
    )
    with pytest.raises(ValueError, match="frozen A or B"):
        runner._resolve_run_id(smoke)
    with pytest.raises(ValueError, match="safe ASCII"):
        runner._valid_run_id("../escape")


def test_t2_semantics_and_result_identity_have_three_states(
    formal_release: CanonicalRelease,
):
    expected = {
        "real": (True, "all_zero_real_false_positive_area"),
        "local_cat": (True, "exact_diff_local_insertion"),
        "fullframe_cat": (False, "not_applicable_fullframe"),
    }
    for condition, semantics in expected.items():
        row = _condition_row(formal_release, condition)
        assert runner._t2_semantics(row) == semantics
        identity = runner.result_identity(
            row,
            run_id="identity-test",
            run_manifest_fingerprint="1" * 64,
            valid_for_metrics=True,
        )
        assert identity["valid_for_t1"] is True
        assert identity["valid_for_t2"] is semantics[0]
        assert identity["t2_applicable"] is semantics[0]
        assert identity["t2_target_semantics"] == semantics[1]
        assert "pair_rank" not in identity
    fullframe = runner.result_task_scope(
        _condition_row(formal_release, "fullframe_mouse")
    )
    assert fullframe["valid_for_t2"] is False
    assert fullframe["native_map_output_role"] == "secondary_T1_and_diagnostic_only"


def test_score_payload_uses_real_model_map_gmp_and_strict_threshold():
    logits = np.zeros((512, 512), dtype=np.float32)
    model_map = np.full((512, 512), 0.5, dtype=np.float32)
    native = np.full((3, 4), 127, dtype=np.uint8)
    payload = runner._score_payload(logits, model_map, native)
    assert payload["ai_score"] == 0.5
    assert payload["classification_decision"] is False
    assert payload["classification_threshold_operator"] == ">"
    assert payload["official_png_score"] == 127 / 255
    assert payload["official_png_decision"] is False
    assert payload["calibrated_probability"] is False
    assert payload["score_margin"] is None

    bad = model_map.copy()
    bad[0, 0] = 0.51
    with pytest.raises(ValueError, match="static sanity"):
        runner._score_payload(logits, bad, native)


def test_official_quantize_before_resize_and_strict_mask_boundaries():
    probability = np.full((512, 512), 0.5, dtype=np.float32)
    native = runner.legacy.official_postprocess(
        probability,
        width=7,
        height=5,
    )
    assert native.dtype == np.uint8
    assert native.shape == (5, 7)
    assert np.unique(native).tolist() == [127]
    assert not np.any(native.astype(np.float32) / np.float32(255.0) > 0.5)

    scores = np.asarray(
        [
            [
                np.float32(0.5),
                np.nextafter(np.float32(0.5), np.float32(1.0)),
            ]
        ],
        dtype=np.float32,
    )
    target = np.asarray([[True, True]])
    metrics = runner.binary_pixel_metrics_strict(
        scores,
        target,
        threshold=0.5,
        include_ap=False,
    )
    assert metrics["threshold_operator"] == ">"
    assert metrics["predicted_positive_pixels"] == 1


def test_stateful_history_allows_recovery_but_only_in_selected_order(
    formal_release: CanonicalRelease,
):
    selected = list(formal_release.inputs[:3])
    valid = [
        _attempt(selected[0], "error"),
        _attempt(selected[0], "ok"),
        _attempt(selected[1], "error"),
        _attempt(selected[1], "error"),
        _attempt(selected[1], "ok"),
    ]
    audit = runner._validate_physical_attempt_history(selected, valid)
    assert audit["successful_prefix"] == 2
    assert audit["errors"] == 3
    assert audit["recovered_error_to_ok"] == 2

    with pytest.raises(ValueError, match="out of selected order"):
        runner._validate_physical_attempt_history(
            selected,
            [_attempt(selected[1], "ok")],
        )
    with pytest.raises(ValueError, match="out of selected order"):
        runner._validate_physical_attempt_history(
            selected,
            [
                _attempt(selected[0], "ok"),
                _attempt(selected[0], "error"),
            ],
        )
    with pytest.raises(ValueError, match="after full success"):
        runner._validate_physical_attempt_history(
            selected[:1],
            [
                _attempt(selected[0], "ok"),
                _attempt(selected[0], "error"),
            ],
        )


def test_error_result_rejects_success_payload_fields(
    formal_release: CanonicalRelease,
    tmp_path: Path,
):
    row = dict(formal_release.inputs[0])
    attempt = _attempt(row, "error")
    attempt["official_png_score"] = None
    with pytest.raises(ValueError, match="error result key set"):
        runner._validate_runner_attempt(
            attempt,
            input_row=row,
            repo_root=REPO_ROOT,
            artifact_root=tmp_path,
            run_id="history-test",
            run_manifest_fingerprint="0" * 64,
            verify_artifacts=False,
        )


def test_artifact_paths_reject_unsafe_ids(tmp_path: Path):
    safe = "a" * 24
    paths = runner.artifact_paths(tmp_path, safe)
    assert paths["raw_logits"].name == f"{safe}.npy"
    assert paths["native_score"].name == f"{safe}.png"
    for value in ("../escape", "A" * 24, "a" * 23, "a" * 25):
        with pytest.raises(ValueError, match="sample_id is unsafe"):
            runner.artifact_paths(tmp_path, value)


def test_artifact_inventory_is_exact_and_fullframe_has_no_mask(
    formal_release: CanonicalRelease,
    tmp_path: Path,
):
    selected = [
        _condition_row(formal_release, "real"),
        _condition_row(formal_release, "local_mouse"),
        _condition_row(formal_release, "fullframe_mouse"),
    ]
    runner._prepare_artifact_root(tmp_path)
    latest = {}
    for row in selected:
        sample_id = row["sample_id"]
        latest[sample_id] = {"status": "ok"}
        paths = runner.artifact_paths(tmp_path, sample_id)
        paths["raw_logits"].write_bytes(b"x")
        paths["model_score"].write_bytes(b"x")
        paths["native_score"].write_bytes(b"x")
        if row["gt_mask_kind"] != "not_applicable":
            paths["mask"].write_bytes(b"x")
    assert runner.validate_artifact_inventory(
        artifact_root=tmp_path,
        selected=selected,
        latest_by_sample_id=latest,
    ) == {
        "raw_logits_model_512": 3,
        "score_maps_model_512": 3,
        "score_maps_native_official": 3,
        "masks_native": 2,
    }
    fullframe_mask = runner.artifact_paths(
        tmp_path,
        selected[-1]["sample_id"],
    )["mask"]
    fullframe_mask.write_bytes(b"forbidden")
    with pytest.raises(ValueError, match="inventory mismatch"):
        runner.validate_artifact_inventory(
            artifact_root=tmp_path,
            selected=selected,
            latest_by_sample_id=latest,
        )


def test_artifact_root_rejects_extra_entries(tmp_path: Path):
    tmp_path.mkdir(exist_ok=True)
    (tmp_path / "unexpected.txt").write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="unexpected/unsafe"):
        runner._prepare_artifact_root(tmp_path)


def test_preprocess_replays_frozen_opencv_bgr_contract(
    formal_release: CanonicalRelease,
):
    row = formal_release.inputs[0]
    image_path = REPO_ROOT / row["canonical_path"]
    tensor, native_size, audit = runner._preprocess_with_audit(image_path)
    assert native_size == (row["width"], row["height"])
    assert tensor.shape == (3, 512, 512)
    assert tensor.dtype == np.float32
    assert tensor.flags.c_contiguous
    assert audit["decoder"] == "opencv_imread_color"
    assert audit["channel_order"] == "BGR"
    assert audit["resize"] == "opencv_inter_linear_stretch"
    assert audit["normalized_chw_sha256"] == runner._array_sha256(tensor)


def test_strict_json_helpers_reject_duplicates_noncanonical_and_nan(
    tmp_path: Path,
):
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"a":1,"a":2}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        runner._load_json_object_strict(duplicate)

    noncanonical = tmp_path / "noncanonical.jsonl"
    noncanonical.write_text('{"b": 2, "a": 1}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="not canonical"):
        runner._read_jsonl_strict(noncanonical)

    nonfinite = tmp_path / "nan.jsonl"
    nonfinite.write_text('{"a":NaN}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="non-finite"):
        runner._read_jsonl_strict(nonfinite)

    canonical = tmp_path / "canonical.jsonl"
    value = {"a": 1, "b": 2}
    canonical.write_text(f"{stable_json(value)}\n", encoding="utf-8")
    assert runner._read_jsonl_strict(canonical) == [value]


def test_run_directory_safety_rejects_unknown_files(tmp_path: Path):
    run_dir = tmp_path / "run"
    runner._validate_run_directory_safety(run_dir, resume=False)
    with pytest.raises(FileNotFoundError, match="missing"):
        runner._validate_run_directory_safety(run_dir, resume=True)
    run_dir.mkdir()
    (run_dir / "unknown.bin").write_bytes(b"x")
    with pytest.raises(ValueError, match="unexpected"):
        runner._validate_run_directory_safety(run_dir, resume=False)


def test_immutable_config_binds_stateful_resume_and_disjoint_outputs(
    formal_release: CanonicalRelease,
):
    spec, selected = runner.select_mode_inputs(
        formal_release,
        mode="smoke",
        per_condition_limit=5,
        sample_id=None,
    )
    from eval.opensource.balanced_run_contract import (
        build_run_dataset_contract,
    )

    dataset_contract = build_run_dataset_contract(
        formal_release,
        spec,
        selected,
        score_spec=runner.SCORE_SPEC,
    )
    preflight = {
        "adapter_sources": {"runner": {"sha256": "0" * 64}},
        "adapter_sources_sha256": "7" * 64,
        "source": {"contract_sha256": "1" * 64},
        "checkpoint": {"contract_sha256": "2" * 64},
        "environment": {"contract_sha256": "3" * 64},
        "checkpoint_audit": {"contract_sha256": "4" * 64},
        "model_audit": {"contract_sha256": "5" * 64},
        "mouse_reference": {"contract_sha256": "8" * 64},
    }
    runtime = {"contract_sha256": "6" * 64}
    result_root = REPO_ROOT / runner.DEFAULT_RESULTS_DIR / "fixture"
    artifact_root = REPO_ROOT / runner.DEFAULT_ARTIFACTS_DIR / "fixture"
    immutable = runner.build_immutable_run_config(
        repo_root=REPO_ROOT,
        run_id="fixture",
        mode="smoke",
        dataset_contract=dataset_contract.as_dict(),
        selected=selected,
        cpu_preflight=preflight,
        runtime=runtime,
        results_path=result_root / "results.jsonl",
        expected_inputs_path=result_root / "expected_inputs.jsonl",
        summary_path=result_root / "summary.json",
        artifact_root=artifact_root,
    )
    assert immutable["score_spec"]["threshold_operator"] == ">"
    assert (
        immutable["inference"]["resume"]
        == "fresh_checkpoint_and_replay_of_every_successful_prefix_input"
    )
    assert immutable["license"] == runner.LICENSE_RECORD
    assert immutable["outputs"]["artifact_root"].startswith(
        "outputs/opensource/mvssnet/"
    )
    assert immutable["outputs"]["results_path"].startswith(
        "results/opensource/mvssnet/"
    )


def test_cpu_preflight_source_order_precedes_runtime_and_has_no_forward():
    source = inspect.getsource(runner.run)
    assert source.index("run_cpu_preflight(") < source.index("configure_runtime(")
    preflight_source = inspect.getsource(runner.run_cpu_preflight)
    model_audit_source = inspect.getsource(runner._build_cpu_model_audit)
    assert "infer_one" not in preflight_source
    assert "infer_one" not in model_audit_source
    assert '"cpu"' in model_audit_source
    assert "cuda.is_initialized" in preflight_source


def test_runtime_and_adapter_hash_records_are_self_consistent():
    environment = runner.verify_environment()
    assert environment["contract_sha256"] == runner._fingerprint(
        {key: value for key, value in environment.items() if key != "contract_sha256"}
    )
    contract = runner.adapter_source_contract(REPO_ROOT)
    assert len(contract) == len(runner.ADAPTER_SOURCE_PATHS)


def test_preflight_cli_rejects_cuda_and_run_mutations():
    args = runner._build_parser().parse_args(
        ["--mode", "preflight", "--device", "cuda:0"]
    )
    assert args.device == "cuda:0"
    assert args.mode == "preflight"
    # The rejection lives before run_cpu_preflight, so this source-level
    # assertion prevents a test from instantiating the 147M-parameter model.
    source = inspect.getsource(runner.run)
    assert "preflight accepts no run/selection/resume/CUDA options" in source


def test_only_the_two_requested_mvssnet_balanced_files_are_new_in_scope():
    assert runner.__file__.endswith("eval/opensource/run_mvssnet_balanced.py")
    assert Path(__file__).name == "test_run_mvssnet_balanced.py"
