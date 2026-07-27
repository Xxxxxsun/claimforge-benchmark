from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from eval.opensource import run_catnet_balanced as runner
from eval.opensource.balanced_run_contract import build_run_dataset_contract
from eval.opensource.canonical_release import (
    BALANCED_DATASET_ID,
    BALANCED_SCHEMA,
    CanonicalRelease,
    LedgerView,
    load_canonical_release,
)
from eval.opensource.common import sha256_file


REPO_ROOT = Path(__file__).resolve().parents[1]
FORMAL_MANIFEST = Path("outputs/opensource/balanced250_v1/manifest.json")


@pytest.fixture(scope="module")
def formal_release() -> CanonicalRelease:
    return load_canonical_release(
        REPO_ROOT,
        FORMAL_MANIFEST,
        verify_files=False,
    )


def test_runner_contract_ids_capability_license_and_no_t1_are_frozen():
    assert runner.DEFAULT_FORMAL_RUN_ID == (
        "catnet_v2_ijcv2022_balanced250_v1_full1025_20260727"
    )
    assert runner.DEFAULT_SMOKE_RUN_ID_A == (
        "catnet_v2_ijcv2022_balanced250_v1_smoke5x4_a_20260727"
    )
    assert runner.DEFAULT_SMOKE_RUN_ID_B == (
        "catnet_v2_ijcv2022_balanced250_v1_smoke5x4_b_20260727"
    )
    assert runner.DEFAULT_SMOKE_LIMIT == 5
    assert runner.FORMAL_IMAGES == 1_025
    assert runner.SMOKE_IMAGES == 20
    assert runner.FORMAL_COUNTS == {
        "real": 275,
        "local_mouse": 250,
        "local_cat": 250,
        "local_trash_can": 250,
    }
    assert runner.SMOKE_COUNTS == {
        "real": 5,
        "local_mouse": 5,
        "local_cat": 5,
        "local_trash_can": 5,
    }
    assert runner.FORMAL_SELECTED_IDS_SHA256 == (
        "612e08565e38cb219fe5ea94dc8193580e099455e11fa778822488dbe7071717"
    )
    assert runner.FORMAL_SELECTED_ROWS_SHA256 == (
        "19ff584a5d073dd03cd31eaf0d22b105d079b2dd606ea535fbbcd39fb692b887"
    )
    assert runner.SMOKE_SELECTED_IDS_SHA256 == (
        "3ce822824a5548f12ae0633520a19686048fd175f7add178334ab5c4fe7e78f4"
    )
    assert runner.SMOKE_SELECTED_ROWS_SHA256 == (
        "7ec14339cad5c6e083f6b1fde56a965686d552ca0b9026eea975144ade7d1d6c"
    )
    assert runner.TASK_SCOPE == {
        "primary_task": "T2_native_pixel_localization_only",
        "capability": "local_t2_only",
        "valid_for_t1": False,
        "valid_for_t2": True,
        "native_dense_output": True,
        "separate_image_classification_head": False,
        "map_statistic_promoted_to_t1": False,
        "fullframe_t1": "not_applicable",
        "fullframe_t2": "not_applicable",
    }
    assert runner.T2_SPEC["valid_conditions"] == [
        "real",
        "local_mouse",
        "local_cat",
        "local_trash_can",
    ]
    assert runner.T2_SPEC["not_selected_conditions"] == [
        "fullframe_mouse",
        "fullframe_cat",
        "fullframe_trash_can",
    ]
    assert runner.T2_SPEC["fullframe_output"] == {
        "selected": False,
        "forward_performed": False,
        "artifact_saved": False,
        "t1_derived_from_map": False,
        "t2_scored": False,
    }
    assert (
        runner.LICENSE_RECORD["catnet_project"][
            "project_wide_license_found"
        ]
        is False
    )
    assert (
        runner.LICENSE_RECORD["hrnet_component_notice"]["scope"]
        == "inherited_HRNet_component_only"
    )
    assert runner.legacy.CHECKPOINT_SHA256 == (
        "f82aaafdd1142775231feedcea0bb7027f7370561d9e8d107465454001865989"
    )
    assert runner.legacy.CHECKPOINT_BYTES == 915_503_873
    assert runner.legacy.CHECKPOINT_EPOCH == 196
    assert runner.legacy.CHECKPOINT_STATE_KEYS == 2_926


def test_adapter_source_inventory_hashes_every_bound_file():
    contract = runner.adapter_source_contract(REPO_ROOT)
    assert tuple(contract) == runner.ADAPTER_SOURCE_PATHS
    assert "eval/opensource/run_catnet_balanced.py" in contract
    assert "eval/opensource/run_catnet.py" in contract
    assert "eval/opensource/analyze_catnet_balanced.py" not in contract
    for relative, binding in contract.items():
        path = REPO_ROOT / relative
        assert binding == {
            "path": relative,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }


def test_source_and_asset_bindings_are_exact_without_cuda():
    import torch

    assert torch.cuda.is_initialized() is False
    source = runner.verify_source(runner.legacy.DEFAULT_CATNET_ROOT)
    assets = runner.verify_assets(runner.legacy.DEFAULT_CHECKPOINT)
    assert source["commit"] == runner.legacy.MODEL_SOURCE_COMMIT
    assert source["tree"] == runner.MODEL_TREE
    assert source["origin"] == runner.MODEL_GIT_ORIGIN
    assert source["tracked_and_untracked_clean"] is True
    assert set(source["source_bound_files"]) == set(runner.SOURCE_BOUND_FILES)
    assert assets["checkpoint"]["sha256"] == runner.legacy.CHECKPOINT_SHA256
    assert assets["checkpoint"]["strict_model_load"] is True
    assert assets["checkpoint"]["safe_weights_only_load"] is True
    assert torch.cuda.is_initialized() is False


def test_artifact_output_is_gitignored():
    evidence = runner.verify_artifact_ignore(REPO_ROOT)
    assert evidence["ignored"] is True
    assert evidence["probe"].startswith("outputs/opensource/catnet/")
    assert ".gitignore:" in evidence["git_check_ignore_evidence"]
    assert len(evidence["contract_sha256"]) == 64


def test_formal_selection_is_exact_capability_correct_1025(
    formal_release: CanonicalRelease,
):
    spec, selected = runner.select_mode_inputs(
        formal_release,
        mode="formal",
        per_condition_limit=None,
        sample_id=None,
    )
    assert spec.capability.value == "local_t2_only"
    assert len(selected) == 1_025
    assert Counter(row["condition"] for row in selected) == (
        runner.FORMAL_COUNTS
    )
    assert all(
        row["condition"]
        in ("real", "local_mouse", "local_cat", "local_trash_can")
        for row in selected
    )
    assert not any(
        str(row["condition"]).startswith("fullframe_") for row in selected
    )
    expected = [
        row
        for row in formal_release.inputs
        if row["condition"] in runner.FORMAL_COUNTS
    ]
    assert [row["sample_id"] for row in selected] == [
        row["sample_id"] for row in expected
    ]
    contract = build_run_dataset_contract(
        formal_release,
        spec,
        selected,
        score_spec=None,
    )
    assert contract.capability.name == "local_t2_only"
    assert contract.capability.valid_for_t1 is False
    assert contract.capability.valid_for_t2 is True
    assert contract.score_spec is None
    assert contract.selection.selected_images == 1_025
    assert dict(contract.selection.counts_by_condition) == (
        runner.FORMAL_COUNTS
    )
    assert runner._required_artifact_bytes(selected) == 9_074_344_528


def test_smoke_selection_is_panel_first_5x4_not_fabricated_5x7(
    formal_release: CanonicalRelease,
):
    spec, selected = runner.select_mode_inputs(
        formal_release,
        mode="smoke",
        per_condition_limit=5,
        sample_id=None,
    )
    assert spec.capability.value == "local_t2_only"
    assert spec.per_condition_limit == 5
    assert len(selected) == 20
    assert Counter(row["condition"] for row in selected) == (
        runner.SMOKE_COUNTS
    )
    assert all(row["panel"] is True for row in selected)
    assert not any(
        str(row["condition"]).startswith("fullframe_") for row in selected
    )
    expected = {
        condition: [
            row["sample_id"]
            for row in formal_release.panel
            if row["condition"] == condition
        ][:5]
        for condition in runner.SMOKE_COUNTS
    }
    actual = {
        condition: [
            row["sample_id"]
            for row in selected
            if row["condition"] == condition
        ]
        for condition in runner.SMOKE_COUNTS
    }
    assert actual == expected
    contract = build_run_dataset_contract(
        formal_release, spec, selected, score_spec=None
    )
    assert contract.selection.selected_images == 20
    assert dict(contract.selection.counts_by_condition) == runner.SMOKE_COUNTS
    assert runner._required_artifact_bytes(selected) == 193_038_388


def test_mode_selection_and_frozen_run_ids_fail_closed(
    formal_release: CanonicalRelease,
):
    with pytest.raises(ValueError, match="exactly 5"):
        runner.select_mode_inputs(
            formal_release,
            mode="smoke",
            per_condition_limit=7,
            sample_id=None,
        )
    with pytest.raises(ValueError, match="requires"):
        runner.select_mode_inputs(
            formal_release,
            mode="smoke",
            per_condition_limit=None,
            sample_id=None,
        )
    with pytest.raises(ValueError, match="no selection"):
        runner.select_mode_inputs(
            formal_release,
            mode="formal",
            per_condition_limit=5,
            sample_id=None,
        )
    formal_args = argparse.Namespace(
        mode="formal",
        run_id="wrong",
    )
    with pytest.raises(ValueError, match="frozen"):
        runner._resolve_run_id(formal_args)
    for run_id in (runner.DEFAULT_SMOKE_RUN_ID_A, runner.DEFAULT_SMOKE_RUN_ID_B):
        smoke_args = argparse.Namespace(mode="smoke", run_id=run_id)
        assert runner._resolve_run_id(smoke_args) == run_id
    with pytest.raises(ValueError, match="frozen A or B"):
        runner._resolve_run_id(
            argparse.Namespace(mode="smoke", run_id="catnet-smoke-c")
        )


def test_single_selection_rejects_a_fullframe_input(
    formal_release: CanonicalRelease,
):
    fullframe = next(
        row
        for row in formal_release.inputs
        if str(row["condition"]).startswith("fullframe_")
    )
    with pytest.raises(ValueError, match="outside the requested capability"):
        runner.select_mode_inputs(
            formal_release,
            mode="single",
            per_condition_limit=None,
            sample_id=fullframe["sample_id"],
        )


@pytest.mark.parametrize(
    ("condition", "gt_mask_kind", "semantics"),
    [
        ("real", "all_zero", "all_zero_real_false_positive_area"),
        ("local_mouse", "exact_diff", "exact_diff_local_insertion"),
        ("local_cat", "exact_diff", "exact_diff_local_insertion"),
        ("local_trash_can", "exact_diff", "exact_diff_local_insertion"),
    ],
)
def test_result_identity_is_t2_only_and_has_no_score(
    condition: str,
    gt_mask_kind: str,
    semantics: str,
):
    row = _minimal_row(condition, gt_mask_kind)
    identity = runner.result_identity(
        row,
        run_id="catnet-test",
        run_manifest_fingerprint="1" * 64,
        valid_for_metrics=True,
    )
    assert identity["valid_for_t1"] is False
    assert identity["valid_for_t2"] is True
    assert identity["t2_applicable"] is True
    assert identity["task_scope"] == {
        "valid_for_t1": False,
        "valid_for_t2": True,
        "t2_applicable": True,
        "t2_target_semantics": semantics,
        "map_statistic_promoted_to_t1": False,
    }
    for forbidden in runner._FORBIDDEN_T1_TOP_LEVEL:
        assert forbidden not in identity


def test_result_semantics_reject_fullframe_even_if_called_directly():
    row = _minimal_row("fullframe_cat", "not_applicable")
    row.update(
        {
            "condition_family": "full_frame_conditional_edit",
            "manipulation_scope": "conditional_full_frame_edit",
            "kind": "forged",
            "label": 1,
        }
    )
    with pytest.raises(ValueError, match="T2-only semantics"):
        runner.result_task_scope(row)


def test_attempt_history_is_append_only_and_success_terminal():
    selected = [_minimal_row("real", "all_zero")]
    sample_id = selected[0]["sample_id"]
    assert runner._validate_physical_attempt_history(
        selected,
        [
            {"sample_id": sample_id, "status": "error"},
            {"sample_id": sample_id, "status": "ok"},
        ],
    )["recovered_error_to_ok"] == 1
    with pytest.raises(ValueError, match="after success"):
        runner._validate_physical_attempt_history(
            selected,
            [
                {"sample_id": sample_id, "status": "ok"},
                {"sample_id": sample_id, "status": "error"},
            ],
        )
    with pytest.raises(ValueError, match="after success|multiple successful"):
        runner._validate_physical_attempt_history(
            selected,
            [
                {"sample_id": sample_id, "status": "ok"},
                {"sample_id": sample_id, "status": "ok"},
            ],
        )


def test_strict_json_rejects_duplicate_keys_and_noncanonical_jsonl(
    tmp_path: Path,
):
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"a":1,"a":2}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        runner._load_json_object_strict(duplicate)
    noncanonical = tmp_path / "rows.jsonl"
    noncanonical.write_text('{"b": 1, "a": 2}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="canonical"):
        runner._read_jsonl_strict(noncanonical)


def test_inventory_is_exact_and_rejects_extra_artifacts(tmp_path: Path):
    root = tmp_path / "artifacts"
    runner._prepare_artifact_root(root)
    empty = runner.validate_artifact_inventory(
        artifact_root=root,
        selected=[_minimal_row("real", "all_zero")],
        latest_by_sample_id={},
    )
    assert empty["successful_images"] == 0
    assert empty["files"] == 0
    (root / "score_maps_native" / "extra.npy").write_bytes(b"x")
    with pytest.raises(ValueError, match="inventory mismatch"):
        runner.validate_artifact_inventory(
            artifact_root=root,
            selected=[_minimal_row("real", "all_zero")],
            latest_by_sample_id={},
        )


def test_artifact_fields_bind_file_and_array_hashes(tmp_path: Path):
    root = tmp_path
    artifact_root = root / "outputs/opensource/catnet/test"
    runner._prepare_artifact_root(artifact_root)
    sample_id = "a" * 24
    paths = runner.artifact_paths(artifact_root, sample_id)
    raw = np.zeros((2, 2, 2), dtype=np.float32)
    score = np.asarray([[0.2, 0.7], [0.5, 0.9]], dtype=np.float32)
    mask = np.where(score >= 0.5, 255, 0).astype(np.uint8)
    runner.legacy._atomic_save_npy(paths["raw_logits"], raw)
    runner.legacy._atomic_save_npy(paths["score_map"], score)
    runner.legacy._atomic_save_mask(paths["mask"], score >= 0.5)
    fields = runner._artifact_fields(
        repo_root=root,
        paths=paths,
        raw_logits=raw,
        score_map=score,
        mask=mask,
    )
    assert fields["raw_logits_array_sha256"] == hashlib.sha256(
        raw.tobytes()
    ).hexdigest()
    assert fields["score_map_array_sha256"] == hashlib.sha256(
        score.tobytes()
    ).hexdigest()
    assert fields["mask_array_sha256"] == hashlib.sha256(
        mask.tobytes()
    ).hexdigest()
    assert fields["score_map_semantics"] == (
        "native_probability_of_channel_1_tampered"
    )


def test_path_containment_rejects_symlink_components(tmp_path: Path):
    root = tmp_path / "results"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (root / "escape").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink component"):
        runner._safe_child(root, "escape", "test run")


def test_preflight_cli_rejects_cuda_or_run_options():
    parser = runner._build_parser()
    for extra in (
        ["--device", "cuda:0"],
        ["--run-id", "x"],
        ["--sample-id", "a" * 24],
        ["--per-condition-limit", "5"],
        ["--resume"],
    ):
        args = parser.parse_args(["--mode", "preflight", *extra])
        invalid = (
            args.resume
            or args.run_id is not None
            or args.sample_id is not None
            or args.per_condition_limit is not None
            or (args.device is not None and args.device != "cpu")
        )
        assert invalid is True


def test_immutable_config_has_no_score_spec_and_binds_self_hash(
    formal_release: CanonicalRelease,
    tmp_path: Path,
):
    spec, selected = runner.select_mode_inputs(
        formal_release,
        mode="smoke",
        per_condition_limit=5,
        sample_id=None,
    )
    contract = build_run_dataset_contract(
        formal_release, spec, selected, score_spec=None
    )
    immutable = runner.build_immutable_run_config(
        repo_root=REPO_ROOT,
        run_id=runner.DEFAULT_SMOKE_RUN_ID_A,
        mode="smoke",
        dataset_contract=contract.as_dict(),
        selected=selected,
        cpu_preflight={"contract_sha256": "0" * 64},
        runtime={"device": "cpu", "contract_sha256": "1" * 64},
        results_path=REPO_ROOT / "results/opensource/catnet/x/results.jsonl",
        expected_inputs_path=(
            REPO_ROOT / "results/opensource/catnet/x/expected_inputs.jsonl"
        ),
        summary_path=REPO_ROOT / "results/opensource/catnet/x/summary.json",
        artifact_root=REPO_ROOT / "outputs/opensource/catnet/x",
    )
    assert immutable["score_spec"] is None
    assert immutable["task_scope"]["valid_for_t1"] is False
    assert immutable["selection"]["selected_images"] == 20
    self_record = immutable["adapter_sources"][
        "eval/opensource/run_catnet_balanced.py"
    ]
    assert self_record["sha256"] == sha256_file(
        REPO_ROOT / self_record["path"]
    )
    assert self_record["bytes"] == (
        REPO_ROOT / self_record["path"]
    ).stat().st_size
    del tmp_path


@pytest.mark.skipif(
    Path(sys.executable) != runner.EXPECTED_PYTHON_EXECUTABLE
    or os.environ.get("PYTHONDONTWRITEBYTECODE") != "1"
    or sys.pycache_prefix is None
    or Path(sys.pycache_prefix) != runner.FROZEN_PYTHONPYCACHEPREFIX,
    reason="real CAT-Net CPU preflight requires the pinned isolated venv",
)
def test_real_cpu_preflight_strict_loads_full_model_without_cuda():
    import torch

    assert torch.cuda.is_initialized() is False
    report = runner.run_cpu_preflight(
        repo_root=REPO_ROOT,
        catnet_root=runner.legacy.DEFAULT_CATNET_ROOT,
        checkpoint_path=runner.legacy.DEFAULT_CHECKPOINT,
    )
    assert report["schema_version"] == runner.CPU_PREFLIGHT_SCHEMA
    assert report["cuda_initialized_before"] is False
    assert report["cuda_initialized_after"] is False
    assert report["balanced250_forward_performed"] is False
    assert report["balanced250_score_computed"] is False
    assert report["t1_output_computed"] is False
    assert report["checkpoint_audit"]["state_keys"] == 2_926
    assert report["model_audit"]["strict_load"] is True
    assert report["model_audit"]["parameter_count"] == 114_263_810
    assert report["model_audit"]["forward_performed"] is False
    assert torch.cuda.is_initialized() is False


def _minimal_row(condition: str, gt_mask_kind: str) -> dict:
    real = condition == "real"
    local = condition.startswith("local_")
    fullframe = condition.startswith("fullframe_")
    if not (real or local or fullframe):
        raise ValueError(condition)
    return {
        "schema_version": BALANCED_SCHEMA,
        "dataset_id": BALANCED_DATASET_ID,
        "rank": 0,
        "sample_id": "a" * 24,
        "condition": condition,
        "condition_family": (
            "real"
            if real
            else "local_splice"
            if local
            else "full_frame_conditional_edit"
        ),
        "manipulation_scope": (
            "authentic"
            if real
            else "local_insertion"
            if local
            else "conditional_full_frame_edit"
        ),
        "normalized_task_id": "task",
        "task_id": "task",
        "kind": "real" if real else "forged",
        "label": 0 if real else 1,
        "domain": "lodging",
        "gt_mask_kind": gt_mask_kind,
        "canonical_path": (
            "outputs/opensource/balanced250_v1/images/"
            "aaaaaaaaaaaaaaaaaaaaaaaa.jpg"
        ),
        "canonical_sha256": "2" * 64,
        "width": 8,
        "height": 8,
    }
