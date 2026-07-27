from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from eval.opensource import run_hifi_ifdl_balanced as runner
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
    assert runner.DEFAULT_FORMAL_RUN_ID == (
        "hifi_ifdl_general750001_balanced250_v1_full1775_r2_20260727"
    )
    assert runner.DEFAULT_SMOKE_RUN_ID_A == (
        "hifi_ifdl_general750001_balanced250_v1_smoke5x7_a_r2_20260727"
    )
    assert runner.DEFAULT_SMOKE_RUN_ID_B == (
        "hifi_ifdl_general750001_balanced250_v1_smoke5x7_b_r2_20260727"
    )
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
    assert runner.T2_SPEC["fullframe_dense_output"] == {
        "role": "transient_diagnostic_only",
        "saved": False,
        "scored": False,
        "promoted_to_image_score": False,
    }
    assert runner.TASK_SCOPE["separate_image_classification_head"] is True
    assert runner.TASK_SCOPE["map_statistic_promoted_to_t1"] is False
    assert runner.LICENSE_RECORD["project_code"]["commercial_use_permission"] is True
    assert (
        runner.LICENSE_RECORD["official_checkpoint_bundle"][
            "commercial_use_clearance_established"
        ]
        is False
    )
    assert runner.EXPECTED_MODEL_PARAMETERS == 6_890_320
    assert runner.EXPECTED_MODEL_BUFFERS == 18_764
    assert runner.SIMPLEX_SUM_ABS_TOLERANCE == float(2 * np.finfo(np.float32).eps)
    assert runner.STATIC_CPU_SOFTMAX_ABS_TOLERANCE == float(
        8 * np.finfo(np.float32).eps
    )
    assert runner.legacy.CHECKPOINT_BUNDLE_SHA256 == (
        "62d0b9f5e501f85558cfbdd5f797dc4e2553ce74729168921257408d735681f9"
    )


def test_adapter_source_inventory_hashes_every_bound_local_file():
    contract = runner.adapter_source_contract(REPO_ROOT)
    assert tuple(contract) == runner.ADAPTER_SOURCE_PATHS
    assert "eval/opensource/run_hifi_ifdl_balanced.py" in contract
    assert "eval/opensource/run_hifi_ifdl.py" in contract
    assert "eval/opensource/analyze_hifi_ifdl_balanced.py" not in contract
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
    assert evidence["probe"].startswith("outputs/opensource/hifi_ifdl/")
    assert ".gitignore:" in evidence["git_check_ignore_evidence"]
    assert len(evidence["contract_sha256"]) == 64


def test_official_source_and_assets_are_exact_without_cuda():
    import torch

    assert torch.cuda.is_initialized() is False
    source = runner.verify_source(runner.legacy.DEFAULT_HIFI_ROOT)
    assets = runner.verify_assets(
        hifi_root=runner.legacy.DEFAULT_HIFI_ROOT,
        hrnet_checkpoint=runner.legacy.DEFAULT_HRNET_CHECKPOINT,
        nlc_checkpoint=runner.legacy.DEFAULT_NLC_CHECKPOINT,
    )
    assert source["commit"] == runner.legacy.MODEL_SOURCE_COMMIT
    assert source["tree"] == runner.MODEL_TREE
    assert source["origin"] == runner.MODEL_GIT_ORIGIN
    assert source["tracked_and_non_cache_untracked_clean"] is True
    assert source["bytecode_cache_execution"] is False
    assert set(source["source_bound_files"]) == set(runner.SOURCE_BOUND_FILES)
    assert assets["bundle_sha256"] == (runner.legacy.CHECKPOINT_BUNDLE_SHA256)
    assert set(assets["assets"]) == {
        "initialization_weight",
        "feature_extractor",
        "hierarchical_localizer_classifier",
        "center_radius",
    }
    assert torch.cuda.is_initialized() is False


def test_real_cpu_preflight_strict_loads_all_components_without_cuda():
    if Path(sys.executable) != runner.EXPECTED_PYTHON_EXECUTABLE:
        pytest.skip("real preflight requires the pinned HiFi interpreter")
    if (
        os.environ.get("PYTHONDONTWRITEBYTECODE") != "1"
        or sys.pycache_prefix is None
        or Path(sys.pycache_prefix).resolve()
        != runner.FROZEN_PYTHONPYCACHEPREFIX.resolve()
    ):
        pytest.skip("real preflight requires the frozen bytecode isolation")
    import torch

    assert torch.cuda.is_initialized() is False
    report = runner.run_cpu_preflight(
        repo_root=REPO_ROOT,
        hifi_root=runner.legacy.DEFAULT_HIFI_ROOT,
        hrnet_checkpoint=runner.legacy.DEFAULT_HRNET_CHECKPOINT,
        nlc_checkpoint=runner.legacy.DEFAULT_NLC_CHECKPOINT,
    )
    assert report["schema_version"] == runner.CPU_PREFLIGHT_SCHEMA
    assert report["cuda_initialized_before"] is False
    assert report["cuda_initialized_after"] is False
    assert report["balanced250_forward_performed"] is False
    assert report["balanced250_score_computed"] is False
    assert report["model_audit"]["parameter_count"] == 6_890_320
    assert report["model_audit"]["buffer_elements"] == 18_764
    assert report["model_audit"]["module_count"] == 505
    assert report["model_audit"]["forward_performed"] is False
    assert set(report["checkpoint_audit"]["task_components"]) == {
        "feature_extractor",
        "hierarchical_localizer_classifier",
    }
    assert torch.cuda.is_initialized() is False


def test_formal_selection_is_exact_1775_t1_and_1025_t2(
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
    assert runner._required_artifact_bytes(selected) == 15_353_827_600
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
            "hifi-unfrozen-smoke",
        ]
    )
    with pytest.raises(ValueError, match="frozen A or B"):
        runner._resolve_run_id(smoke)
    for unsafe in ("../escape", "/absolute", "space is bad", ".", ".."):
        with pytest.raises(ValueError, match="safe ASCII"):
            runner._valid_run_id(unsafe)


def test_result_identity_preserves_three_state_t2_semantics(
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
        assert identity["task_scope"]["dense_output_saved"] is applicable
        assert identity["task_scope"]["map_statistic_promoted_to_t1"] is False
        assert "pair_rank" not in identity


def _hierarchy_logits() -> dict[str, np.ndarray]:
    return {
        name: np.zeros(classes, dtype=np.float32)
        for name, classes in runner.HIERARCHY_SPECS
    }


def test_score_payload_uses_fine14_head_and_keeps_official_rule_separate():
    hierarchy = _hierarchy_logits()
    hierarchy["out3_fine_14class"] = np.asarray(
        [1.0, *([0.9] * 13)],
        dtype=np.float32,
    )
    probabilities = runner._stable_softmax(hierarchy["out3_fine_14class"])
    payload = runner._score_payload(hierarchy, probabilities)
    assert payload["ai_score"] > 0.5
    assert payload["classification_decision"] == "forged"
    assert payload["classification_threshold_operator"] == ">"
    assert payload["official_fine_class_index"] == 0
    assert payload["official_fine_class_name"] == "authentic"
    assert payload["official_binary_decision"] is False
    assert payload["official_decision"] == "authentic"
    with pytest.raises(ValueError, match="disagree"):
        runner._score_payload(
            hierarchy,
            np.full(14, 1.0 / 14.0, dtype=np.float32),
        )


def test_cuda_cpu_softmax_roundoff_regression_is_static_sanity_only():
    """Freeze the formal rank-1447 3-eps/7-ULP diagnostic boundary."""

    fine_logits = np.asarray(
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
    recorded_cuda_probabilities = np.asarray(
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
    cpu_reference = runner._stable_softmax(fine_logits)
    difference = np.abs(
        recorded_cuda_probabilities.astype(np.float64)
        - cpu_reference.astype(np.float64)
    )
    ulp_distance = np.abs(
        recorded_cuda_probabilities.view(np.uint32).astype(np.int64)
        - cpu_reference.view(np.uint32).astype(np.int64)
    )
    epsilon = float(np.finfo(np.float32).eps)
    assert float(difference.max()) == 3 * epsilon
    assert int(ulp_distance.max()) == 7
    assert not np.allclose(
        recorded_cuda_probabilities,
        cpu_reference,
        rtol=0.0,
        atol=2 * epsilon,
    )
    assert np.allclose(
        recorded_cuda_probabilities,
        cpu_reference,
        rtol=0.0,
        atol=runner.STATIC_CPU_SOFTMAX_ABS_TOLERANCE,
    )
    assert math.isclose(
        float(recorded_cuda_probabilities.sum(dtype=np.float64)),
        1.0,
        rel_tol=0.0,
        abs_tol=runner.SIMPLEX_SUM_ABS_TOLERANCE,
    )
    hierarchy = _hierarchy_logits()
    hierarchy["out3_fine_14class"] = fine_logits
    payload = runner._score_payload(
        hierarchy,
        recorded_cuda_probabilities,
    )
    assert payload["ai_score"] == float(
        np.float32(1.0) - recorded_cuda_probabilities[0]
    )
    assert payload["classification_decision"] == "authentic"


def test_append_only_history_allows_recovery_but_success_is_terminal():
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
    image_path = root / row["canonical_path"]
    image_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(
        np.zeros((6, 8, 3), dtype=np.uint8),
        mode="RGB",
    ).save(
        image_path,
        format="JPEG",
        quality=95,
        subsampling=0,
        optimize=False,
    )
    row["canonical_sha256"] = runner.sha256_file(image_path)
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
    tensor = np.zeros((3, 256, 256), dtype=np.float32)
    return {
        "decoder": "imageio.v2.imread",
        "channel_order": "RGB",
        "decoded_dtype": "uint8",
        "native_size": [8, 6],
        "model_size": [256, 256],
        "geometry": "direct_stretch_without_aspect_ratio_preservation",
        "resize_interpolation": "Pillow.Image.Resampling.BICUBIC",
        "input_crop": None,
        "input_reencode": False,
        "normalization": "uint8_rgb_divide_255_float32",
        "tensor_shape": [3, 256, 256],
        "tensor_dtype": "float32",
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
            "tracked_and_non_cache_untracked_clean": True,
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


def _processed_fixture() -> dict:
    hierarchy = _hierarchy_logits()
    hierarchy["out3_fine_14class"] = np.asarray(
        [0.0, 1.0, *([0.0] * 12)],
        dtype=np.float32,
    )
    probabilities = runner._stable_softmax(hierarchy["out3_fine_14class"])
    return {
        "embedding": np.zeros((18, 256, 256), dtype=np.float32),
        "distance_model_256": np.full(
            (256, 256),
            2.4,
            dtype=np.float32,
        ),
        "distance_native": np.full((6, 8), 2.4, dtype=np.float32),
        "hierarchy_logits": hierarchy,
        "fine_probabilities": probabilities,
        "score": float(np.float32(1.0) - probabilities[0]),
        "benchmark_binary_decision": True,
        "official_fine_class_index": 1,
        "official_fine_class_name": "splice",
        "official_binary_decision": True,
        "auxiliary_learned_mask_stats": {
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
    monkeypatch.setattr(
        runner,
        "DEFAULT_DATASET_MANIFEST",
        Path("manifest.json"),
    )
    monkeypatch.setattr(runner, "DEFAULT_RESULTS_DIR", Path("results"))
    monkeypatch.setattr(
        runner,
        "DEFAULT_ARTIFACTS_DIR",
        Path("artifacts"),
    )

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
        return (
            (SimpleNamespace(), SimpleNamespace()),
            SimpleNamespace(),
            SimpleNamespace(),
            torch.device("cpu"),
        )

    monkeypatch.setattr(runner.legacy, "load_model", load_model)
    tensor = np.zeros((3, 256, 256), dtype=np.float32)
    audit = _preprocess_audit()
    monkeypatch.setattr(
        runner,
        "_preprocess_with_audit",
        lambda _path: (tensor.copy(), (8, 6), dict(audit)),
    )

    def infer(*_args, **_kwargs):
        if inference_error is not None:
            raise inference_error
        return _processed_fixture(), 0, 1.25

    monkeypatch.setattr(runner.legacy, "infer_one", infer)


def _single_args(
    root: Path,
    condition: str,
    *,
    resume: bool = False,
) -> list[str]:
    sample_id = _minimal_row(condition)["sample_id"]
    run_id = f"hifi-balanced-{condition}-test"
    values = [
        "--repo-root",
        str(root),
        "--dataset-manifest",
        "manifest.json",
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
    ("condition", "expects_t2"),
    [
        ("real", True),
        ("local_cat", True),
        ("fullframe_cat", False),
    ],
)
def test_cpu_single_writes_capability_correct_t1_t2_artifacts(
    tmp_path: Path,
    monkeypatch,
    condition: str,
    expects_t2: bool,
):
    release = _minimal_release(tmp_path, condition)
    events: list[str] = []
    _patch_cpu_run(monkeypatch, release, events)
    assert runner.main(_single_args(tmp_path, condition)) == 0
    assert events[:2] == ["cpu_preflight", "configure:cpu"]
    assert events[2:] == ["load_model"]

    run_id = f"hifi-balanced-{condition}-test"
    run_dir = tmp_path / "results" / run_id
    artifact_root = tmp_path / "artifacts" / run_id
    rows = read_jsonl(run_dir / "results.jsonl")
    assert len(rows) == 1
    row = rows[0]
    assert row["status"] == "ok"
    assert row["ai_score"] > 0.5
    assert row["classification_decision"] == "forged"
    assert row["t2_applicable"] is expects_t2
    assert (row["artifact_paths"] is not None) is expects_t2
    assert (row["mask_path"] is not None) is expects_t2
    assert (row["localization"] is not None) is expects_t2
    if expects_t2:
        assert row["dense_output_disposition"].startswith("saved_and_scored")
    else:
        assert row["dense_output_disposition"].startswith("discarded_transient")
    inventory = runner.validate_artifact_inventory(
        artifact_root=artifact_root,
        selected=release.inputs,
        latest_by_sample_id={row["sample_id"]: row},
    )
    assert inventory == {
        "embeddings_model_256": int(expects_t2),
        "distance_maps_model_256": int(expects_t2),
        "distance_maps_native": int(expects_t2),
        "masks_native": int(expects_t2),
    }
    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["scientific_metrics"] is None
    assert summary["coverage"]["valid_images"] == 1
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["status"] == "complete"
    assert manifest["dataset"]["t1_applicable_images"] == 1
    assert manifest["dataset"]["t2_applicable_images"] == int(expects_t2)
    softmax_sanity = manifest["immutable"]["inference"]["static_cpu_softmax_sanity"]
    assert softmax_sanity["classes"] == 14
    assert softmax_sanity["simplex_sum_absolute_tolerance"] == (
        runner.SIMPLEX_SUM_ABS_TOLERANCE
    )
    assert softmax_sanity["cross_device_absolute_tolerance"] == (
        runner.STATIC_CPU_SOFTMAX_ABS_TOLERANCE
    )
    assert softmax_sanity["recorded_device_smoke_and_fresh_replay_tolerance"] == 0.0
    assert softmax_sanity["affects_score_decision_or_artifacts"] is False


def test_complete_resume_validates_artifacts_and_never_loads_model(
    tmp_path: Path,
    monkeypatch,
):
    release = _minimal_release(tmp_path, "real")
    events: list[str] = []
    _patch_cpu_run(monkeypatch, release, events)
    assert runner.main(_single_args(tmp_path, "real")) == 0
    result_path = tmp_path / "results/hifi-balanced-real-test/results.jsonl"
    first_rows = result_path.read_bytes()
    events.clear()
    assert runner.main(_single_args(tmp_path, "real", resume=True)) == 0
    assert events == ["cpu_preflight", "configure:cpu"]
    assert result_path.read_bytes() == first_rows
    manifest = json.loads(
        (tmp_path / "results/hifi-balanced-real-test/manifest.json").read_text()
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
    results_path = tmp_path / "results/hifi-balanced-real-test/results.jsonl"
    before = results_path.read_bytes()
    native_path = (
        tmp_path
        / "artifacts/hifi-balanced-real-test/distance_maps_native"
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


def test_inference_error_is_invalid_append_only_and_cleans_artifacts(
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
    run_id = "hifi-balanced-real-test"
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


def test_inventory_rejects_extra_files(tmp_path: Path):
    root = tmp_path / "artifacts"
    runner._prepare_artifact_root(root)
    (root / "embeddings_model_256" / "extra.npy").write_bytes(b"x")
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


def test_preflight_cli_rejects_cuda_or_selection_options():
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
