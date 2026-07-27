from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from eval.opensource import run_trufor_balanced as runner
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
FORMAL_MANIFEST = Path(
    "outputs/opensource/balanced250_v1/manifest.json"
)


@pytest.fixture(scope="module")
def formal_release() -> CanonicalRelease:
    return load_canonical_release(
        REPO_ROOT,
        FORMAL_MANIFEST,
        verify_files=False,
    )


def test_runner_contract_constants_provenance_and_license_are_frozen():
    assert runner.SCORE_SPEC.as_dict() == {
        "key": "ai_score",
        "direction": "higher_means_fake",
        "fixed_threshold": 0.5,
        "threshold_operator": ">=",
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
        runner.T2_SPEC["native_reliability_map"][
            "used_for_primary_metrics"
        ]
        is False
    )
    assert (
        runner.T2_SPEC["native_reliability_map"][
            "must_not_be_multiplied_into_probability_map"
        ]
        is True
    )
    assert (
        runner.LICENSE_RECORD["overall"][
            "commercial_use_requires_separate_authorization"
        ]
        is True
    )
    assert (
        runner.LICENSE_RECORD["cmx_component"][
            "does_not_override_overall_trufor_restriction"
        ]
        is True
    )
    assert runner.CHECKPOINT_BYTES == 281_496_429
    assert runner.ARCHIVE_BYTES == 260_878_690
    assert set(runner.TRUFOR_SOURCE_FILES) >= {
        "lib/config/default.py",
        "lib/utils.py",
        "lib/models/DnCNN.py",
        "lib/models/cmx/builder_np_conf.py",
        "lib/models/cmx/encoders/dual_segformer.py",
        "lib/models/cmx/decoders/MLPDecoder.py",
    }


def test_adapter_source_inventory_hashes_every_bound_local_file():
    contract = runner.adapter_source_contract(REPO_ROOT)
    assert tuple(contract) == runner.ADAPTER_SOURCE_PATHS
    assert "eval/opensource/run_trufor_balanced.py" in contract
    assert "eval/opensource/run_trufor.py" in contract
    assert "eval/opensource/analyze_trufor_balanced.py" not in contract
    for relative, binding in contract.items():
        path = REPO_ROOT / relative
        assert binding == {
            "path": relative,
            "bytes": path.stat().st_size,
            "sha256": runner.sha256_file(path),
        }


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
    assert len(selected) == 1775
    assert Counter(row["condition"] for row in selected) == (
        runner.FORMAL_COUNTS
    )
    assert sum(runner._t2_semantics(row)[0] for row in selected) == 1025
    assert runner._required_artifact_bytes(selected) == 24_833_955_064
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
    assert contract.selection.selected_images == 1775


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
            row["sample_id"]
            for row in selected
            if row["condition"] == condition
        ]
        for condition in BALANCED_CONDITIONS
    }
    assert actual == expected
    assert [row["rank"] for row in selected] == sorted(
        row["rank"] for row in selected
    )


def test_mode_selectors_and_frozen_run_ids_are_fail_closed(
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
            "trufor-unfrozen-smoke",
        ]
    )
    with pytest.raises(ValueError, match="frozen A or B"):
        runner._resolve_run_id(smoke)


def test_append_only_history_allows_error_recovery_but_success_is_terminal():
    selected = [_minimal_row("real")]
    sample_id = selected[0]["sample_id"]
    assert runner._validate_physical_attempt_history(
        selected,
        [
            {"sample_id": sample_id, "status": "error"},
            {"sample_id": sample_id, "status": "error"},
            {"sample_id": sample_id, "status": "ok"},
        ],
    )["recovered_error_to_ok"] == 1
    for statuses in (("ok", "ok"), ("ok", "error")):
        with pytest.raises(ValueError, match="after success"):
            runner._validate_physical_attempt_history(
                selected,
                [
                    {"sample_id": sample_id, "status": status}
                    for status in statuses
                ],
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
            row
            for row in formal_release.inputs
            if row["condition"] == condition
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
            identity["task_scope"]["reliability_map_output_role"]
            == "diagnostic_only"
        )
        assert "pair_rank" not in identity


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
        mask_path = (
            root
            / "outputs/fixture/masks"
            / f"{row['sample_id']}.png"
        )
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
    rgb = np.zeros((6, 8, 3), dtype=np.uint8)
    tensor = np.zeros((3, 6, 8), dtype=np.float32)
    return {
        "profile": runner.PREPROCESS_PROFILE,
        "decode": "PIL_convert_RGB",
        "decoded_size": [8, 6],
        "decoded_rgb_shape": [6, 8, 3],
        "decoded_rgb_dtype": "uint8",
        "decoded_rgb_array_sha256": hashlib.sha256(
            rgb.tobytes()
        ).hexdigest(),
        "tensor_shape": [3, 6, 8],
        "tensor_dtype": "float32",
        "tensor_array_sha256": hashlib.sha256(
            tensor.tobytes()
        ).hexdigest(),
        "input_scale": "float32_divide_by_256",
        "input_scale_divisor": 256.0,
        "input_resize": None,
        "input_crop": None,
        "network_map_upsample": (
            "bilinear_align_corners_false_to_native_input_size"
        ),
        "post_network_map_restore": None,
    }


def _preflight_fixture() -> dict:
    return {
        "schema_version": runner.CPU_PREFLIGHT_SCHEMA,
        "status": "passed",
        "environment": {
            "python_executable": str(runner.EXPECTED_PYTHON_EXECUTABLE),
        },
        "source": {
            "commit": runner.legacy.MODEL_SOURCE_COMMIT,
            "tracked_and_untracked_clean": True,
        },
        "assets": {
            "checkpoint": {
                "id": runner.CHECKPOINT_ID,
                "sha256": runner.legacy.CHECKPOINT_SHA256,
            },
        },
        "checkpoint_audit": {
            "state_dict_tensors": runner.CHECKPOINT_STATE_KEYS,
        },
        "model_audit": {
            "strict_state_dict_load": True,
            "model_forwards": 0,
        },
        "license": runner.LICENSE_RECORD,
        "accelerator_model_forwards": 0,
        "balanced250_model_scores_computed": 0,
        "cuda_initialized_before": False,
        "cuda_initialized_after": False,
    }


def _patch_cpu_run(
    monkeypatch,
    release: CanonicalRelease,
    events: list[str],
    *,
    inference_error: Exception | None = None,
) -> None:
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
        return SimpleNamespace(type="cpu"), {
            "device": device_text,
            "seed": runner.MODEL_SEED,
        }

    monkeypatch.setattr(runner, "run_cpu_preflight", preflight)
    monkeypatch.setattr(runner, "configure_runtime", configure)
    monkeypatch.setattr(
        runner,
        "adapter_source_contract",
        lambda _root: {
            path: {
                "path": path,
                "bytes": index + 1,
                "sha256": f"{index + 1:064x}",
            }
            for index, path in enumerate(runner.ADAPTER_SOURCE_PATHS)
        },
    )
    monkeypatch.setattr(
        runner,
        "_verify_disk_capacity",
        lambda _root, pending: {
            "free_bytes_before_inference": 10_000_000_000,
            "estimated_pending_map_bytes_plus_reserve": (
                runner._required_artifact_bytes(pending)
            ),
            "fixed_reserve_bytes": runner.MIN_DISK_RESERVE_BYTES,
        },
    )
    monkeypatch.setattr(
        runner.legacy,
        "load_model",
        lambda **_kwargs: (
            SimpleNamespace(),
            SimpleNamespace(type="cpu"),
        ),
    )
    tensor = np.zeros((3, 6, 8), dtype=np.float32)
    audit = _preprocess_audit()
    monkeypatch.setattr(
        runner,
        "_preprocess_with_audit",
        lambda _path: (tensor.copy(), (8, 6), dict(audit)),
    )
    score_map = np.full((6, 8), 0.75, dtype=np.float32)
    reliability = np.full((6, 8), 0.25, dtype=np.float32)

    def infer(*_args, **_kwargs):
        if inference_error is not None:
            raise inference_error
        return (
            0.75,
            float(np.log(3.0)),
            score_map.copy(),
            reliability.copy(),
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
    run_id = f"trufor-balanced-{condition}-test"
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


def test_cpu_preprocess_matches_official_native_divide_256(tmp_path: Path):
    path = tmp_path / "image.png"
    pixels = np.asarray(
        [
            [[0, 1, 255], [10, 20, 30]],
            [[40, 50, 60], [70, 80, 90]],
        ],
        dtype=np.uint8,
    )
    Image.fromarray(pixels, mode="RGB").save(path)
    tensor, size, audit = runner._preprocess_with_audit(path)
    assert size == (2, 2)
    assert tensor.shape == (3, 2, 2)
    assert tensor.dtype == np.float32
    assert tensor[2, 0, 0] == np.float32(255.0 / 256.0)
    assert audit["input_scale_divisor"] == 256.0
    assert audit["input_resize"] is None
    assert audit["input_crop"] is None
    assert audit["decoded_rgb_array_sha256"] == runner._array_sha256(
        pixels
    )


def test_cpu_real_single_writes_both_raw_maps_mask_and_t2_fp(
    tmp_path: Path,
    monkeypatch,
):
    release = _minimal_release(tmp_path, "real")
    events: list[str] = []
    _patch_cpu_run(monkeypatch, release, events)
    assert runner.main(_single_args(tmp_path, "real")) == 0
    assert events[:2] == ["cpu_preflight", "configure:cpu"]

    run_id = "trufor-balanced-real-test"
    run_dir = tmp_path / "results" / run_id
    artifact_root = tmp_path / "artifacts" / run_id
    manifest = json.loads((run_dir / "manifest.json").read_text())
    summary = json.loads((run_dir / "summary.json").read_text())
    result = read_jsonl(run_dir / "results.jsonl")[0]
    assert manifest["status"] == "complete"
    assert manifest["fingerprint"] == runner._fingerprint(
        manifest["immutable"]
    )
    assert (
        manifest["immutable"]["cpu_preflight"]["report"][
            "cuda_initialized_after"
        ]
        is False
    )
    assert summary["scientific_metrics"] is None
    assert summary["artifact_inventory"] == {
        "score_maps_native": 1,
        "reliability_maps_native": 1,
        "masks_native": 1,
    }
    assert result["schema_version"] == runner.RESULT_SCHEMA_VERSION
    assert result["status"] == "ok"
    assert result["ai_score"] == 0.75
    assert result["classification_decision"] is True
    assert result["t2_applicable"] is True
    assert (
        result["localization"]["native"]["target_positive_pixels"]
        == 0
    )
    assert (
        result["localization"]["native"]["predicted_positive_pixels"]
        == 48
    )
    assert result["localization"]["native"]["pixel_ap"] is None
    assert result["reliability"]["used_for_primary_metrics"] is False
    assert result["reliability"]["multiplied_into_score_map"] is False

    score_map = np.load(
        tmp_path / result["score_map_native_path"],
        allow_pickle=False,
    )
    reliability = np.load(
        tmp_path / result["reliability_map_native_path"],
        allow_pickle=False,
    )
    assert score_map.shape == reliability.shape == (6, 8)
    assert score_map.dtype == reliability.dtype == np.float32
    assert result["score_map_native_array_sha256"] == (
        runner._array_sha256(score_map)
    )
    assert result["reliability_map_native_array_sha256"] == (
        runner._array_sha256(reliability)
    )
    with Image.open(tmp_path / result["mask_path"]) as opened:
        mask = np.asarray(opened, dtype=np.uint8)
    assert np.array_equal(mask == 255, score_map >= 0.5)
    assert artifact_root.is_dir()


def test_cpu_local_exact_diff_writes_native_t2_and_pixel_ap(
    tmp_path: Path,
    monkeypatch,
):
    release = _minimal_release(tmp_path, "local_cat")
    events: list[str] = []
    _patch_cpu_run(monkeypatch, release, events)
    assert runner.main(_single_args(tmp_path, "local_cat")) == 0
    run_dir = tmp_path / "results/trufor-balanced-local_cat-test"
    result = read_jsonl(run_dir / "results.jsonl")[0]
    assert result["t2_applicable"] is True
    assert result["t2_target_semantics"] == "exact_diff_local_insertion"
    assert result["localization"]["native"]["target_positive_pixels"] == 1
    assert result["localization"]["native"]["pixel_ap"] is not None
    assert result["mask_path"] is not None
    assert result["artifact_paths"]["mask_native"] == result["mask_path"]


def test_cpu_fullframe_retains_raw_maps_but_has_no_t2_outputs(
    tmp_path: Path,
    monkeypatch,
):
    release = _minimal_release(tmp_path, "fullframe_cat")
    events: list[str] = []
    _patch_cpu_run(monkeypatch, release, events)
    assert runner.main(_single_args(tmp_path, "fullframe_cat")) == 0
    run_dir = tmp_path / "results/trufor-balanced-fullframe_cat-test"
    result = read_jsonl(run_dir / "results.jsonl")[0]
    summary = json.loads((run_dir / "summary.json").read_text())
    assert result["t2_applicable"] is False
    assert result["t2_target_semantics"] == "not_applicable_fullframe"
    assert result["score_map_native_path"] is not None
    assert result["reliability_map_native_path"] is not None
    for field in (
        "mask_path",
        "mask_sha256",
        "mask_bytes",
        "mask_array_sha256",
        "mask_shape",
        "mask_dtype",
        "mask_semantics",
        "localization",
    ):
        assert result[field] is None
    assert result["artifact_paths"]["mask_native"] is None
    assert summary["artifact_inventory"] == {
        "score_maps_native": 1,
        "reliability_maps_native": 1,
        "masks_native": 0,
    }


def test_complete_resume_replays_preprocess_and_artifacts_without_model(
    tmp_path: Path,
    monkeypatch,
):
    release = _minimal_release(tmp_path, "real")
    events: list[str] = []
    _patch_cpu_run(monkeypatch, release, events)
    assert runner.main(_single_args(tmp_path, "real")) == 0
    monkeypatch.setattr(
        runner.legacy,
        "load_model",
        lambda **_kwargs: pytest.fail("complete resume loaded model"),
    )
    assert runner.main(
        _single_args(tmp_path, "real", resume=True)
    ) == 0
    run_dir = tmp_path / "results/trufor-balanced-real-test"
    assert len(read_jsonl(run_dir / "results.jsonl")) == 1
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["execution"]["resume_skips"] == 1
    assert manifest["execution"]["new_successes"] == 0


def test_resume_rejects_missing_artifact_directory_without_recreating_it(
    tmp_path: Path,
    monkeypatch,
):
    release = _minimal_release(tmp_path, "fullframe_cat")
    events: list[str] = []
    _patch_cpu_run(monkeypatch, release, events)
    assert runner.main(_single_args(tmp_path, "fullframe_cat")) == 0
    run_dir = tmp_path / "results/trufor-balanced-fullframe_cat-test"
    manifest_path = run_dir / "manifest.json"
    manifest_before = manifest_path.read_bytes()
    missing = (
        tmp_path
        / "artifacts/trufor-balanced-fullframe_cat-test/masks_native"
    )
    missing.rmdir()
    with pytest.raises(ValueError, match="root inventory mismatch"):
        runner.main(
            _single_args(tmp_path, "fullframe_cat", resume=True)
        )
    assert not missing.exists()
    assert manifest_path.read_bytes() == manifest_before


def test_resume_fails_closed_and_nonmutating_on_tampered_raw_map(
    tmp_path: Path,
    monkeypatch,
):
    release = _minimal_release(tmp_path, "real")
    events: list[str] = []
    _patch_cpu_run(monkeypatch, release, events)
    assert runner.main(_single_args(tmp_path, "real")) == 0
    run_dir = tmp_path / "results/trufor-balanced-real-test"
    result = read_jsonl(run_dir / "results.jsonl")[0]
    score_path = tmp_path / result["score_map_native_path"]
    manifest_path = run_dir / "manifest.json"
    manifest_before = manifest_path.read_bytes()
    runner.legacy._atomic_save_npy(
        score_path,
        np.zeros((6, 8), dtype=np.float32),
    )
    with pytest.raises(ValueError, match="file metadata changed"):
        runner.main(_single_args(tmp_path, "real", resume=True))
    assert manifest_path.read_bytes() == manifest_before


def test_resume_rejects_unknown_outer_manifest_fields_without_rewrite(
    tmp_path: Path,
    monkeypatch,
):
    release = _minimal_release(tmp_path, "real")
    events: list[str] = []
    _patch_cpu_run(monkeypatch, release, events)
    assert runner.main(_single_args(tmp_path, "real")) == 0
    manifest_path = (
        tmp_path
        / "results/trufor-balanced-real-test/manifest.json"
    )
    manifest = json.loads(manifest_path.read_text())
    manifest["unexpected"] = True
    runner.atomic_write_json(manifest_path, manifest)
    tampered = manifest_path.read_bytes()
    with pytest.raises(ValueError, match="manifest key set changed"):
        runner.main(_single_args(tmp_path, "real", resume=True))
    assert manifest_path.read_bytes() == tampered


@pytest.mark.parametrize("linked_root", ["results", "artifacts"])
def test_run_id_symlink_cannot_escape_output_roots(
    tmp_path: Path,
    monkeypatch,
    linked_root: str,
):
    release = _minimal_release(tmp_path, "real")
    events: list[str] = []
    _patch_cpu_run(monkeypatch, release, events)
    run_id = "trufor-balanced-real-test"
    external = tmp_path / "external"
    external.mkdir()
    parent = tmp_path / linked_root
    parent.mkdir()
    (parent / run_id).symlink_to(external, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink component"):
        runner.main(_single_args(tmp_path, "real"))
    assert not any(external.iterdir())
    assert events == []


def test_running_resume_cannot_append_through_results_symlink(
    tmp_path: Path,
    monkeypatch,
):
    release = _minimal_release(tmp_path, "real")
    events: list[str] = []
    _patch_cpu_run(monkeypatch, release, events)
    run_dir = tmp_path / "results/trufor-balanced-real-test"
    run_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text("{}\n", encoding="utf-8")
    (run_dir / "expected_inputs.jsonl").write_text(
        "{}\n",
        encoding="utf-8",
    )
    external = tmp_path / "external-results.jsonl"
    external.write_text('{"sentinel":true}\n', encoding="utf-8")
    before = external.read_bytes()
    (run_dir / "results.jsonl").symlink_to(external)
    with pytest.raises(ValueError, match="symlink component"):
        runner.main(_single_args(tmp_path, "real", resume=True))
    assert external.read_bytes() == before
    assert events == []


@pytest.mark.parametrize("initial_status", ["ok", "error"])
def test_resume_rejects_unknown_keys_on_all_attempt_types(
    tmp_path: Path,
    monkeypatch,
    initial_status: str,
):
    release = _minimal_release(tmp_path, "real")
    events: list[str] = []
    _patch_cpu_run(
        monkeypatch,
        release,
        events,
        inference_error=(
            RuntimeError("inference failed")
            if initial_status == "error"
            else None
        ),
    )
    expected_exit = 2 if initial_status == "error" else 0
    assert runner.main(_single_args(tmp_path, "real")) == expected_exit
    result_path = (
        tmp_path
        / "results/trufor-balanced-real-test/results.jsonl"
    )
    rows = read_jsonl(result_path)
    rows[0]["unexpected_field"] = "fail closed"
    runner.atomic_write_jsonl(result_path, rows)
    with pytest.raises(ValueError, match="results SHA-256 changed"):
        runner.main(_single_args(tmp_path, "real", resume=True))


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
        inference_error=RuntimeError("inference failed"),
    )
    assert runner.main(_single_args(tmp_path, "real")) == 2
    run_id = "trufor-balanced-real-test"
    run_dir = tmp_path / "results" / run_id
    artifact_root = tmp_path / "artifacts" / run_id
    result = read_jsonl(run_dir / "results.jsonl")[0]
    summary = json.loads((run_dir / "summary.json").read_text())
    assert result["status"] == "error"
    assert result["valid_for_metrics"] is False
    assert "ai_score" not in result
    assert summary["coverage"]["is_complete"] is False
    assert summary["artifact_inventory"] == {
        "score_maps_native": 0,
        "reliability_maps_native": 0,
        "masks_native": 0,
    }
    assert not any(
        path.is_file()
        for path in artifact_root.rglob("*")
    )


def test_inventory_rejects_extra_or_unsafe_files(tmp_path: Path):
    artifact_root = tmp_path / "artifacts"
    runner._prepare_artifact_root(artifact_root)
    (artifact_root / "score_maps_native/extra.npy").write_bytes(b"x")
    with pytest.raises(ValueError, match="inventory mismatch"):
        runner.validate_artifact_inventory(
            artifact_root=artifact_root,
            selected=[_minimal_row("real")],
            latest_by_sample_id={},
        )


def test_complete_resume_requires_no_artifact_disk_reserve():
    assert runner._required_artifact_bytes([]) == 0
    assert runner._required_artifact_bytes([_minimal_row("real")]) > (
        runner.MIN_DISK_RESERVE_BYTES
    )


def test_preflight_verifies_release_and_never_configures_runtime(
    tmp_path: Path,
    monkeypatch,
    capsys,
):
    release = _minimal_release(tmp_path, "real")
    calls: list[tuple[str, object]] = []

    def load_release(*_args, **kwargs):
        calls.append(("release", kwargs.get("verify_files")))
        return release

    monkeypatch.setattr(runner, "load_canonical_release", load_release)
    monkeypatch.setattr(
        runner,
        "run_cpu_preflight",
        lambda **_kwargs: calls.append(("cpu", True))
        or _preflight_fixture(),
    )
    monkeypatch.setattr(
        runner,
        "configure_runtime",
        lambda *_args, **_kwargs: pytest.fail(
            "preflight configured accelerator runtime"
        ),
    )
    assert runner.main(
        [
            "--repo-root",
            str(tmp_path),
            "--mode",
            "preflight",
        ]
    ) == 0
    assert calls == [("release", True), ("cpu", True)]
    output = json.loads(capsys.readouterr().out)
    assert output["dataset"]["verified_images"] == 1
    assert output["cuda_initialized_after"] is False
    with pytest.raises(ValueError, match="preflight accepts no"):
        runner.main(
            [
                "--repo-root",
                str(tmp_path),
                "--mode",
                "preflight",
                "--device",
                "cuda:0",
            ]
        )


def test_cpu_preflight_is_called_before_runtime_configuration(
    tmp_path: Path,
    monkeypatch,
):
    release = _minimal_release(tmp_path, "real")
    events: list[str] = []
    _patch_cpu_run(monkeypatch, release, events)
    assert runner.main(_single_args(tmp_path, "real")) == 0
    assert events[:2] == ["cpu_preflight", "configure:cpu"]
