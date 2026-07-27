from __future__ import annotations

import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from eval.opensource import run_psccnet_balanced as runner
from eval.opensource.canonical_release import (
    BALANCED_CONDITIONS,
    BALANCED_DATASET_ID,
    BALANCED_SCHEMA,
    CanonicalRelease,
    LedgerView,
    load_canonical_release,
)
from eval.opensource.common import read_jsonl


REPO_ROOT = Path(__file__).resolve().parents[1]
FORMAL_MANIFEST = Path("outputs/opensource/balanced250_v1/manifest.json")


@pytest.fixture(scope="module")
def formal_release() -> CanonicalRelease:
    return load_canonical_release(
        REPO_ROOT,
        FORMAL_MANIFEST,
        verify_files=False,
    )


def test_runner_contract_provenance_capability_and_license_are_frozen():
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
    assert runner.FORMAL_T2_IMAGES == 1_025
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
    assert runner.TASK_SCOPE["separate_image_classification_head"] is True
    assert (
        runner.TASK_SCOPE["native_image_score"]
        == "softmax_class_1_of_independent_detection_head"
    )
    assert runner.LICENSE_RECORD["project_license"]["commercial_use_permission"] is True
    assert runner.LICENSE_RECORD["project_license"]["redistribution_permission"] is True
    assert runner.EXPECTED_MODEL_PARAMETERS == 3_667_942
    assert runner.EXPECTED_MODEL_BUFFERS == 13_490
    assert runner.legacy.CHECKPOINT_BUNDLE_SHA256 == (
        "893626e154e5a3c16322e845a0e8c775029f88a5742de6875818c69f66459560"
    )


def test_adapter_source_inventory_hashes_every_bound_local_file():
    contract = runner.adapter_source_contract(REPO_ROOT)
    assert tuple(contract) == runner.ADAPTER_SOURCE_PATHS
    assert "eval/opensource/run_psccnet_balanced.py" in contract
    assert "eval/opensource/run_psccnet.py" in contract
    assert "eval/opensource/analyze_psccnet_balanced.py" not in contract
    for relative, binding in contract.items():
        path = REPO_ROOT / relative
        assert binding == {
            "path": relative,
            "bytes": path.stat().st_size,
            "sha256": runner.sha256_file(path),
        }


def test_raw_artifact_path_is_gitignored_with_recorded_evidence():
    evidence = runner.verify_artifact_ignore(REPO_ROOT)
    assert evidence["ignored"] is True
    assert evidence["probe"].startswith("outputs/opensource/psccnet/")
    assert ".gitignore:" in evidence["git_check_ignore_evidence"]
    assert len(evidence["contract_sha256"]) == 64


def test_official_source_and_assets_are_exact_and_cpu_only():
    import torch

    before = torch.cuda.is_initialized()
    assert before is False
    source = runner.verify_source(runner.legacy.DEFAULT_PSCCNET_ROOT)
    assets = runner.verify_assets(runner.legacy.DEFAULT_PSCCNET_ROOT)
    assert source["commit"] == runner.legacy.MODEL_SOURCE_COMMIT
    assert source["tracked_and_untracked_clean"] is True
    assert source["origin"] == runner.MODEL_GIT_ORIGIN
    assert set(source["source_bound_files"]) == set(runner.SOURCE_BOUND_FILES)
    assert assets["bundle_sha256"] == (runner.legacy.CHECKPOINT_BUNDLE_SHA256)
    assert set(assets["assets"]) == {
        "initialization_weight",
        "feature_extractor",
        "localization_head",
        "classification_head",
    }
    assert torch.cuda.is_initialized() is False


def test_real_cpu_preflight_strict_loads_all_components_without_cuda():
    if Path(sys.executable) != runner.EXPECTED_PYTHON_EXECUTABLE:
        pytest.skip("real preflight requires the pinned PSCC-Net interpreter")
    import torch

    assert torch.cuda.is_initialized() is False
    report = runner.run_cpu_preflight(
        repo_root=REPO_ROOT,
        psccnet_root=runner.legacy.DEFAULT_PSCCNET_ROOT,
    )
    assert report["schema_version"] == runner.CPU_PREFLIGHT_SCHEMA
    assert report["cuda_initialized_before"] is False
    assert report["cuda_initialized_after"] is False
    assert report["balanced250_forward_performed"] is False
    assert report["balanced250_score_computed"] is False
    assert report["model_audit"]["parameter_count"] == 3_667_942
    assert report["model_audit"]["buffer_elements"] == 13_490
    assert report["model_audit"]["module_count"] == 391
    assert report["model_audit"]["forward_performed"] is False
    assert set(report["checkpoint_audit"]["task_components"]) == {
        "feature_extractor",
        "localization_head",
        "classification_head",
    }
    assert torch.cuda.is_initialized() is False


def test_formal_selection_is_exact_1775_native_t1_t2_contract(
    formal_release: CanonicalRelease,
):
    spec, selected = runner.select_mode_inputs(
        formal_release,
        mode="formal",
        per_condition_limit=None,
        sample_id=None,
    )
    assert spec.capability.value == "local_t1_t2"
    assert len(selected) == 1_775
    assert Counter(row["condition"] for row in selected) == (runner.FORMAL_COUNTS)
    assert sum(runner._t2_semantics(row)[0] for row in selected) == 1_025
    assert runner._required_artifact_bytes(selected) == 15_688_864_972
    assert [row["sample_id"] for row in selected] == [
        row["sample_id"] for row in formal_release.inputs
    ]
    contract = runner.build_run_dataset_contract(
        formal_release,
        spec,
        selected,
        score_spec=runner.SCORE_SPEC,
    )
    assert contract.capability.valid_for_t1 is True
    assert contract.capability.valid_for_t2 is True
    assert contract.selection.selected_images == 1_775


def test_smoke_selection_is_exact_panel_first_five_by_condition(
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
    for limit in (0, 4, 6):
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
            "psccnet-unfrozen-smoke",
        ]
    )
    with pytest.raises(ValueError, match="frozen A or B"):
        runner._resolve_run_id(smoke)
    for unsafe in ("../escape", "/absolute", "space is bad", ".", ".."):
        with pytest.raises(ValueError, match="safe ASCII"):
            runner._valid_run_id(unsafe)


def test_append_only_history_allows_error_recovery_but_success_is_terminal():
    selected = [_minimal_row("real")]
    sample_id = selected[0]["sample_id"]
    history = runner._validate_physical_attempt_history(
        selected,
        [
            {"sample_id": sample_id, "status": "error"},
            {"sample_id": sample_id, "status": "error"},
            {"sample_id": sample_id, "status": "ok"},
        ],
    )
    assert history["recovered_error_to_ok"] == 1
    for statuses in (("ok", "ok"), ("ok", "error")):
        with pytest.raises(ValueError, match="after success"):
            runner._validate_physical_attempt_history(
                selected,
                [{"sample_id": sample_id, "status": status} for status in statuses],
            )


def test_result_identity_has_three_state_t2_semantics(
    formal_release: CanonicalRelease,
):
    expected = {
        "real": (True, "all_zero_real_false_positive_area"),
        "local_cat": (True, "exact_diff_local_insertion"),
        "fullframe_cat": (False, "not_applicable_fullframe"),
    }
    for condition, (applicable, semantics) in expected.items():
        source = next(
            row for row in formal_release.inputs if row["condition"] == condition
        )
        identity = runner.result_identity(
            source,
            run_id="identity-test",
            run_manifest_fingerprint="f" * 64,
            valid_for_metrics=True,
        )
        assert identity["schema_version"] == "opensource_result_v2"
        assert identity["dataset_id"] == BALANCED_DATASET_ID
        assert identity["t2_applicable"] is applicable
        assert identity["t2_target_semantics"] == semantics
        assert identity["task_scope"]["valid_for_t1"] is True
        assert identity["task_scope"]["valid_for_t2"] is applicable
        assert (
            identity["task_scope"]["image_score_source"] == "independent_detection_head"
        )
        assert "pair_rank" not in identity


def test_score_payload_uses_class1_float32_softmax_and_strict_threshold():
    logits = np.asarray([0.0, 0.0], dtype=np.float32)
    probabilities = np.asarray([0.5, 0.5], dtype=np.float32)
    payload = runner._score_payload(logits, probabilities)
    assert payload["ai_score"] == 0.5
    assert payload["classification_decision"] == "authentic"
    assert payload["classification_threshold_operator"] == ">"
    positive = runner._score_payload(
        np.asarray([0.0, 0.001], dtype=np.float32),
        runner._stable_softmax_class1(np.asarray([0.0, 0.001], dtype=np.float32)),
    )
    assert positive["ai_score"] > 0.5
    assert positive["classification_decision"] == "forged"
    with pytest.raises(ValueError, match="disagree"):
        runner._score_payload(
            logits,
            np.asarray([0.4, 0.6], dtype=np.float32),
        )


def _minimal_row(condition: str) -> dict:
    table = {
        "real": (
            "real",
            "real",
            "authentic",
            0,
            "all_zero",
            "a" * 24,
        ),
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
        "normalized_task_id": "lodging_fixture_task",
        "task_id": "lodging_fixture_task",
        "kind": kind,
        "label": label,
        "domain": "lodging",
        "source_content_cluster": "fixture-cluster",
        "gt_mask_kind": gt_kind,
        "gt_mask_path": None,
        "gt_mask_sha256": None,
        "gt_positive_pixels": 0 if gt_kind == "all_zero" else None,
        "canonical_path": f"outputs/fixture/images/{sample_id}.jpg",
        "canonical_sha256": "1" * 64,
        "width": 8,
        "height": 6,
        "panel": True,
        "selection_rank": 0,
    }


def _minimal_release(root: Path, condition: str) -> CanonicalRelease:
    row = _minimal_row(condition)
    if row["gt_mask_kind"] == "exact_diff":
        mask_path = root / "outputs/fixture/masks" / f"{row['sample_id']}.png"
        target = np.zeros((6, 8), dtype=bool)
        target[2, 3] = True
        runner.legacy._atomic_save_mask(mask_path, target)
        row["gt_mask_path"] = mask_path.relative_to(root).as_posix()
        row["gt_mask_sha256"] = runner.sha256_file(mask_path)
        row["gt_positive_pixels"] = 1
    panel_rows = (
        {
            "sample_id": row["sample_id"],
            "condition": row["condition"],
        },
    )
    records = {
        "inputs": {
            "path": "outputs/fixture/inputs.jsonl",
            "sha256": runner._rows_sha256((row,)),
            "rows": 1,
        },
        "panel": {
            "path": "outputs/fixture/panel.jsonl",
            "sha256": runner._rows_sha256(panel_rows),
            "rows": 1,
        },
        "source_pairs": {
            "path": "outputs/fixture/source_pairs.jsonl",
            "sha256": runner._rows_sha256(()),
            "rows": 0,
        },
    }

    def ledger(name: str) -> LedgerView:
        record = records[name]
        return LedgerView(
            name=name,
            path=root / record["path"],
            sha256=record["sha256"],
            rows=record["rows"],
        )

    return CanonicalRelease(
        repo_root=root,
        manifest_path=root / "outputs/fixture/manifest.json",
        manifest_sha256="5" * 64,
        manifest={
            "schema_version": BALANCED_SCHEMA,
            "dataset_id": BALANCED_DATASET_ID,
            "contract_sha256": "6" * 64,
            "ledgers": records,
        },
        schema_version=BALANCED_SCHEMA,
        dataset_id=BALANCED_DATASET_ID,
        release_kind="balanced250",
        contract_sha256="6" * 64,
        inputs_ledger=ledger("inputs"),
        inputs=(row,),
        panel_ledger=ledger("panel"),
        panel=panel_rows,
        source_pairs_ledger=ledger("source_pairs"),
        source_pairs=(),
        legacy_pairs_ledger=None,
        legacy_pairs=(),
    )


def _preprocess_audit() -> dict:
    tensor = np.zeros((3, 6, 8), dtype=np.float32)
    return {
        "decoder": "imageio.v2.imread",
        "channel_order": "RGB",
        "native_size": [8, 6],
        "input_resize": "none",
        "input_crop": None,
        "input_reencode": False,
        "normalization": "uint8_rgb_divide_255",
        "alpha_policy": "not_applicable",
        "tensor_shape": [3, 6, 8],
        "tensor_sha256": hashlib.sha256(tensor.tobytes()).hexdigest(),
    }


def _preflight_fixture() -> dict:
    return {
        "schema_version": runner.CPU_PREFLIGHT_SCHEMA,
        "cuda_initialized_before": False,
        "cuda_initialized_after": False,
        "environment": {
            "python_executable": str(runner.EXPECTED_PYTHON_EXECUTABLE),
        },
        "source": {
            "commit": runner.legacy.MODEL_SOURCE_COMMIT,
            "tracked_and_untracked_clean": True,
        },
        "assets": {
            "bundle_sha256": runner.legacy.CHECKPOINT_BUNDLE_SHA256,
        },
        "adapter_sources": {
            path: {
                "path": path,
                "bytes": index + 1,
                "sha256": f"{index + 1:064x}",
            }
            for index, path in enumerate(runner.ADAPTER_SOURCE_PATHS)
        },
        "artifact_ignore": {"ignored": True},
        "checkpoint_audit": {
            "task_components": {
                role: {"strict": True} for role in runner.legacy.CHECKPOINTS
            },
        },
        "model_audit": {
            "parameter_count": runner.EXPECTED_MODEL_PARAMETERS,
            "forward_performed": False,
        },
        "balanced250_forward_performed": False,
        "balanced250_score_computed": False,
    }


def _patch_cpu_run(
    monkeypatch,
    release: CanonicalRelease,
    events: list[str],
    *,
    inference_error: Exception | None = None,
) -> None:
    import torch

    monkeypatch.setattr(
        runner,
        "load_canonical_release",
        lambda *_args, **_kwargs: release,
    )
    monkeypatch.setattr(runner, "DEFAULT_RESULTS_DIR", Path("results"))
    monkeypatch.setattr(runner, "DEFAULT_ARTIFACTS_DIR", Path("artifacts"))

    def preflight(**_kwargs):
        events.append("cpu_preflight")
        return _preflight_fixture()

    def configure(device_text: str):
        events.append(f"configure:{device_text}")
        device = torch.device("cpu")
        return device, {
            "device": str(device),
            "seed": runner.MODEL_SEED,
        }

    monkeypatch.setattr(runner, "run_cpu_preflight", preflight)
    monkeypatch.setattr(runner, "configure_runtime", configure)
    monkeypatch.setattr(
        runner,
        "_verify_disk_capacity",
        lambda _root, pending: {
            "free_bytes_before_inference": 100_000_000_000,
            "conservative_pending_bytes_plus_reserve": (
                runner._required_artifact_bytes(pending)
            ),
            "fixed_reserve_bytes": (runner.MIN_DISK_RESERVE_BYTES if pending else 0),
        },
    )

    def load_model(**_kwargs):
        events.append("load_model")
        return (SimpleNamespace(),), torch.device("cpu")

    monkeypatch.setattr(runner.legacy, "load_model", load_model)
    tensor = np.zeros((3, 6, 8), dtype=np.float32)
    audit = _preprocess_audit()
    monkeypatch.setattr(
        runner,
        "_preprocess_with_audit",
        lambda _path: (tensor.copy(), (8, 6), dict(audit)),
    )
    masks = [
        np.full(shape, 0.75 if index == 0 else 0.25, dtype=np.float32)
        for index, shape in enumerate(runner.PROGRESSIVE_SHAPES)
    ]
    native = np.full((6, 8), 0.75, dtype=np.float32)
    logits = np.asarray([0.0, 1.0], dtype=np.float32)
    probabilities = runner._stable_softmax_class1(logits)

    def infer(*_args, **_kwargs):
        if inference_error is not None:
            raise inference_error
        return (
            [value.copy() for value in masks],
            native.copy(),
            logits.copy(),
            probabilities.copy(),
            0,
            1.25,
        )

    monkeypatch.setattr(runner.legacy, "infer_one", infer)


def _single_args(
    root: Path,
    condition: str,
    *,
    resume: bool = False,
) -> list[str]:
    sample_id = _minimal_row(condition)["sample_id"]
    run_id = f"psccnet-balanced-{condition}-test"
    values = [
        "--repo-root",
        str(root),
        "--mode",
        "single",
        "--sample-id",
        sample_id,
        "--run-id",
        run_id,
        "--results-dir",
        "results",
        "--artifacts-dir",
        "artifacts",
        "--device",
        "cpu",
    ]
    if resume:
        values.append("--resume")
    return values


@pytest.mark.parametrize(
    ("condition", "expects_mask"),
    [
        ("real", True),
        ("local_cat", True),
        ("fullframe_cat", False),
    ],
)
def test_cpu_single_writes_exact_native_t1_t2_artifacts(
    tmp_path: Path,
    monkeypatch,
    condition: str,
    expects_mask: bool,
):
    release = _minimal_release(tmp_path, condition)
    events: list[str] = []
    _patch_cpu_run(monkeypatch, release, events)
    assert runner.main(_single_args(tmp_path, condition)) == 0
    assert events[:2] == ["cpu_preflight", "configure:cpu"]
    assert events[2:] == ["load_model"]

    run_id = f"psccnet-balanced-{condition}-test"
    run_dir = tmp_path / "results" / run_id
    artifact_root = tmp_path / "artifacts" / run_id
    rows = read_jsonl(run_dir / "results.jsonl")
    assert len(rows) == 1
    row = rows[0]
    assert row["status"] == "ok"
    assert row["ai_score"] > 0.5
    assert row["classification_decision"] == "forged"
    assert len(row["progressive_maps"]) == 4
    assert row["t2_applicable"] is expects_mask
    assert (row["mask_path"] is not None) is expects_mask
    assert (row["localization"] is not None) is expects_mask
    inventory = runner.validate_artifact_inventory(
        artifact_root=artifact_root,
        selected=release.inputs,
        latest_by_sample_id={row["sample_id"]: row},
    )
    assert inventory == {
        "progressive_mask1": 1,
        "progressive_mask2": 1,
        "progressive_mask3": 1,
        "progressive_mask4": 1,
        "score_maps_native": 1,
        "masks_native": int(expects_mask),
    }
    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["scientific_metrics"] is None
    assert summary["coverage"]["valid_images"] == 1
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["status"] == "complete"
    assert manifest["dataset"]["t2_applicable_images"] == int(expects_mask)
    assert (
        manifest["immutable"]["artifact_contract"]["storage"]
        == "local_gitignored_outputs"
    )


def test_complete_resume_validates_every_artifact_and_never_loads_model(
    tmp_path: Path,
    monkeypatch,
):
    release = _minimal_release(tmp_path, "real")
    events: list[str] = []
    _patch_cpu_run(monkeypatch, release, events)
    args = _single_args(tmp_path, "real")
    assert runner.main(args) == 0
    first_rows = (
        tmp_path / "results/psccnet-balanced-real-test/results.jsonl"
    ).read_bytes()
    events.clear()
    assert runner.main(_single_args(tmp_path, "real", resume=True)) == 0
    assert events == ["cpu_preflight", "configure:cpu"]
    assert (
        tmp_path / "results/psccnet-balanced-real-test/results.jsonl"
    ).read_bytes() == first_rows
    manifest = json.loads(
        (tmp_path / "results/psccnet-balanced-real-test/manifest.json").read_text()
    )
    assert manifest["execution"]["new_successes"] == 0
    assert manifest["execution"]["resume_skips"] == 1


def test_resume_fails_closed_on_tampered_map_without_appending(
    tmp_path: Path,
    monkeypatch,
):
    release = _minimal_release(tmp_path, "real")
    events: list[str] = []
    _patch_cpu_run(monkeypatch, release, events)
    assert runner.main(_single_args(tmp_path, "real")) == 0
    run_dir = tmp_path / "results/psccnet-balanced-real-test"
    results_path = run_dir / "results.jsonl"
    before = results_path.read_bytes()
    native_path = (
        tmp_path
        / "artifacts/psccnet-balanced-real-test/score_maps_native"
        / f"{_minimal_row('real')['sample_id']}.npy"
    )
    with native_path.open("r+b") as handle:
        handle.seek(-1, os.SEEK_END)
        last = handle.read(1)
        handle.seek(-1, os.SEEK_END)
        handle.write(bytes([last[0] ^ 1]))
    with pytest.raises(ValueError, match="SHA-256"):
        runner.main(_single_args(tmp_path, "real", resume=True))
    assert results_path.read_bytes() == before


def test_inference_error_is_append_only_invalid_and_cleans_artifacts(
    tmp_path: Path,
    monkeypatch,
):
    release = _minimal_release(tmp_path, "real")
    events: list[str] = []
    _patch_cpu_run(
        monkeypatch,
        release,
        events,
        inference_error=RuntimeError("synthetic forward failure"),
    )
    with pytest.raises(RuntimeError, match="fail-closed"):
        runner.main(_single_args(tmp_path, "real"))
    run_id = "psccnet-balanced-real-test"
    rows = read_jsonl(tmp_path / "results" / run_id / "results.jsonl")
    assert len(rows) == 1
    assert rows[0]["status"] == "error"
    assert rows[0]["valid_for_metrics"] is False
    assert rows[0]["error_type"] == "RuntimeError"
    artifact_root = tmp_path / "artifacts" / run_id
    assert all(not any(path.iterdir()) for path in artifact_root.iterdir())
    manifest = json.loads((tmp_path / "results" / run_id / "manifest.json").read_text())
    assert manifest["status"] == "incomplete"
    assert manifest["execution"]["new_errors"] == 1


def test_inventory_rejects_extra_or_unsafe_files(tmp_path: Path):
    root = tmp_path / "artifacts"
    runner._prepare_artifact_root(root)
    (root / "progressive_mask1" / "extra.npy").write_bytes(b"x")
    with pytest.raises(ValueError, match="inventory mismatch"):
        runner.validate_artifact_inventory(
            artifact_root=root,
            selected=(_minimal_row("real"),),
            latest_by_sample_id={},
        )


def test_path_containment_rejects_symlink_components(tmp_path: Path):
    root = tmp_path / "results"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (root / "escape").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink component"):
        runner._safe_child(root, "escape/run", "test run")


def test_preflight_is_called_before_runtime_configuration(
    tmp_path: Path,
    monkeypatch,
):
    release = _minimal_release(tmp_path, "fullframe_cat")
    events: list[str] = []
    _patch_cpu_run(monkeypatch, release, events)
    runner.main(_single_args(tmp_path, "fullframe_cat"))
    assert events[0] == "cpu_preflight"
    assert events[1] == "configure:cpu"
    assert events.index("cpu_preflight") < events.index("load_model")
