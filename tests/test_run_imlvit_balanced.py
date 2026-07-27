from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from eval.opensource import run_imlvit_balanced as module
from eval.opensource.balanced_run_contract import build_run_dataset_contract
from eval.opensource.canonical_release import (
    Capability,
    load_canonical_release,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def release():
    return load_canonical_release(
        REPO_ROOT,
        REPO_ROOT / module.DEFAULT_DATASET_MANIFEST,
        verify_files=False,
    )


def test_method_is_explicitly_t2_only() -> None:
    assert module.TASK_SCOPE == {
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
    assert module.T2_SPEC["not_selected_conditions"] == [
        "fullframe_mouse",
        "fullframe_cat",
        "fullframe_trash_can",
    ]
    assert module.MASK_THRESHOLD_OPERATOR == ">"
    assert module.SHARED_T2_THRESHOLD_OPERATOR == ">="
    assert module.T2_SPEC["native_binary_mask"] == {
        "threshold": 0.5,
        "threshold_operator": ">",
        "role": "official_artifact_and_fresh_replay",
        "encoding": "PNG_L_0_or_255",
    }
    assert module.T2_SPEC["shared_reducer"] == {
        "threshold": 0.5,
        "threshold_operator": ">=",
        "role": "frozen_cross_method_balanced250_metrics",
        "operator_equivalent_to_official": False,
        "non_equivalence_policy": (
            "allow_and_report_native_pixels_and_images_exactly_at_threshold"
        ),
    }


def test_r2_run_ids_and_v3_schemas_are_frozen() -> None:
    assert (
        module.DEFAULT_FORMAL_RUN_ID
        == "imlvit_cat_protocol_balanced250_v1_full1025_r2_20260727"
    )
    assert (
        module.DEFAULT_SMOKE_RUN_ID_A
        == "imlvit_cat_protocol_balanced250_v1_smoke5x4_a_r2_20260727"
    )
    assert (
        module.DEFAULT_SMOKE_RUN_ID_B
        == "imlvit_cat_protocol_balanced250_v1_smoke5x4_b_r2_20260727"
    )
    assert module.RUN_MANIFEST_SCHEMA.endswith("_v3")
    assert module.RUN_CONFIG_SCHEMA.endswith("_v3")
    assert module.RUNTIME_SUMMARY_SCHEMA.endswith("_v3")
    assert module.CPU_PREFLIGHT_SCHEMA.endswith("_v3")
    assert module.FROZEN_PYTHONPYCACHEPREFIX.name == "imlvit-balanced-v3-empty"


def test_formal_selection_is_exact_1025(release) -> None:
    spec, selected = module.select_mode_inputs(
        release,
        mode="formal",
        per_condition_limit=None,
        sample_id=None,
    )
    assert spec.capability is Capability.LOCAL_T2_ONLY
    assert len(selected) == 1025
    assert Counter(row["condition"] for row in selected) == Counter(
        module.FORMAL_COUNTS
    )
    assert not any(str(row["condition"]).startswith("fullframe_") for row in selected)
    assert (
        module._fingerprint([row["sample_id"] for row in selected])
        == "612e08565e38cb219fe5ea94dc8193580e099455e11fa778822488dbe7071717"
    )
    assert (
        module._fingerprint(selected)
        == "19ff584a5d073dd03cd31eaf0d22b105d079b2dd606ea535fbbcd39fb692b887"
    )


def test_smoke_selection_is_panel_first_5x4(release) -> None:
    spec, selected = module.select_mode_inputs(
        release,
        mode="smoke",
        per_condition_limit=5,
        sample_id=None,
    )
    assert spec.capability is Capability.LOCAL_T2_ONLY
    assert spec.per_condition_limit == 5
    assert len(selected) == 20
    assert Counter(row["condition"] for row in selected) == Counter(module.SMOKE_COUNTS)
    assert all(row["panel"] is True for row in selected)
    assert (
        module._fingerprint([row["sample_id"] for row in selected])
        == "3ce822824a5548f12ae0633520a19686048fd175f7add178334ab5c4fe7e78f4"
    )


def test_smoke_rejects_any_limit_other_than_five(release) -> None:
    with pytest.raises(ValueError, match="exactly 5"):
        module.select_mode_inputs(
            release,
            mode="smoke",
            per_condition_limit=4,
            sample_id=None,
        )


def test_single_rejects_fullframe_sample(release) -> None:
    sample_id = next(
        str(row["sample_id"])
        for row in release.inputs
        if str(row["condition"]).startswith("fullframe_")
    )
    with pytest.raises(ValueError, match="outside the requested capability"):
        module.select_mode_inputs(
            release,
            mode="single",
            per_condition_limit=None,
            sample_id=sample_id,
        )


def test_dataset_contract_has_no_t1_score(release) -> None:
    spec, selected = module.select_mode_inputs(
        release,
        mode="formal",
        per_condition_limit=None,
        sample_id=None,
    )
    contract = build_run_dataset_contract(
        release,
        spec,
        selected,
        score_spec=None,
    )
    assert contract.capability.name == "local_t2_only"
    assert contract.capability.valid_for_t1 is False
    assert contract.capability.valid_for_t2 is True
    assert contract.score_spec is None


def test_result_identity_has_no_image_score(release) -> None:
    _, selected = module.select_mode_inputs(
        release,
        mode="formal",
        per_condition_limit=None,
        sample_id=None,
    )
    result = module.result_identity(
        selected[0],
        run_id="unit",
        run_manifest_fingerprint="0" * 64,
        valid_for_metrics=True,
    )
    assert result["valid_for_t1"] is False
    assert result["valid_for_t2"] is True
    assert result["t2_applicable"] is True
    assert result["task_scope"]["map_statistic_promoted_to_t1"] is False
    assert not module._FORBIDDEN_T1_TOP_LEVEL.intersection(result)


def test_error_attempt_rejects_forbidden_t1_field(release, tmp_path) -> None:
    _, selected = module.select_mode_inputs(
        release,
        mode="single",
        per_condition_limit=None,
        sample_id=next(
            row["sample_id"] for row in release.inputs if row["condition"] == "real"
        ),
    )
    input_row = selected[0]
    row = {
        **module.result_identity(
            input_row,
            run_id="unit",
            run_manifest_fingerprint="0" * 64,
            valid_for_metrics=False,
        ),
        "status": "error",
        "completed_at": "unit",
        "error_type": "RuntimeError",
        "error": "unit",
        "traceback": "unit",
        "score": 0.9,
    }
    with pytest.raises(ValueError, match="forbidden T1"):
        module._validate_runner_attempt(
            row,
            input_row=input_row,
            repo_root=tmp_path,
            artifact_root=tmp_path / "artifacts",
            run_id="unit",
            run_manifest_fingerprint="0" * 64,
            verify_artifacts=False,
        )


def test_valid_error_attempt_is_fail_closed(release, tmp_path) -> None:
    input_row = next(row for row in release.inputs if row["condition"] == "real")
    row = {
        **module.result_identity(
            input_row,
            run_id="unit",
            run_manifest_fingerprint="0" * 64,
            valid_for_metrics=False,
        ),
        "status": "error",
        "completed_at": "unit",
        "error_type": "RuntimeError",
        "error": "unit",
        "traceback": "unit",
    }
    module._validate_runner_attempt(
        row,
        input_row=input_row,
        repo_root=tmp_path,
        artifact_root=tmp_path / "artifacts",
        run_id="unit",
        run_manifest_fingerprint="0" * 64,
        verify_artifacts=False,
    )


def test_success_is_terminal_in_append_only_history(release) -> None:
    selected = [next(row for row in release.inputs if row["condition"] == "real")]
    sample_id = selected[0]["sample_id"]
    assert (
        module._validate_physical_attempt_history(
            selected,
            [
                {"sample_id": sample_id, "status": "error"},
                {"sample_id": sample_id, "status": "ok"},
            ],
        )["recovered_error_to_ok"]
        == 1
    )
    with pytest.raises(ValueError, match="after success"):
        module._validate_physical_attempt_history(
            selected,
            [
                {"sample_id": sample_id, "status": "ok"},
                {"sample_id": sample_id, "status": "error"},
            ],
        )


@pytest.mark.parametrize(
    "value",
    ["", ".", "..", "UPPER", "../escape", "slash/value", "white space"],
)
def test_run_id_rejects_unsafe_values(value: str) -> None:
    with pytest.raises(ValueError):
        module._valid_run_id(value)


def test_frozen_run_ids_are_mode_bound() -> None:
    formal = argparse.Namespace(mode="formal", run_id=None)
    assert module._resolve_run_id(formal) == module.DEFAULT_FORMAL_RUN_ID
    with pytest.raises(ValueError, match="frozen"):
        module._resolve_run_id(argparse.Namespace(mode="formal", run_id="different"))
    assert (
        module._resolve_run_id(
            argparse.Namespace(mode="smoke", run_id=module.DEFAULT_SMOKE_RUN_ID_A)
        )
        == module.DEFAULT_SMOKE_RUN_ID_A
    )


def test_preprocess_real_balanced_input(release) -> None:
    pytest.importorskip("albumentations")
    row = next(row for row in release.inputs if row["condition"] == "real")
    tensor, native_size, resized_size, metadata = module._preprocess_with_audit(
        REPO_ROOT / row["canonical_path"]
    )
    assert tensor.shape == (3, 1024, 1024)
    assert tensor.dtype == np.float32
    assert native_size == (row["width"], row["height"])
    assert metadata["resized_content_size"] == list(resized_size)
    assert metadata["tensor_sha256"] == module._array_sha256(tensor)


def test_source_and_checkpoint_byte_bindings() -> None:
    source = module.verify_source(module.legacy.DEFAULT_IMLVIT_ROOT)
    assets = module.verify_assets(module.legacy.DEFAULT_CHECKPOINT)
    assert source["commit"] == module.legacy.MODEL_SOURCE_COMMIT
    assert source["tree"] == module.MODEL_TREE
    assert assets["checkpoint"]["sha256"] == module.legacy.CHECKPOINT["sha256"]
    assert assets["checkpoint"]["strict_model_load"] is True


def test_verified_input_path_binds_bytes_and_hash(release) -> None:
    row = next(row for row in release.inputs if row["condition"] == "real")
    path = module.verified_input_path(row, REPO_ROOT)
    assert path.name == f"{row['sample_id']}.jpg"
    changed = dict(row)
    changed["canonical_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="bytes changed"):
        module.verified_input_path(changed, REPO_ROOT)


def test_verified_input_path_rejects_symlink_component(tmp_path) -> None:
    sample_id = "0" * 24
    target = tmp_path / "target"
    target.mkdir()
    image = target / f"{sample_id}.jpg"
    image.write_bytes(b"unit")
    (tmp_path / "images").symlink_to(target, target_is_directory=True)
    row = {
        "sample_id": sample_id,
        "canonical_path": f"images/{sample_id}.jpg",
        "canonical_bytes": 4,
        "canonical_sha256": module.sha256_file(image),
    }
    with pytest.raises(ValueError, match="symlink"):
        module.verified_input_path(row, tmp_path)


def test_license_scope_does_not_overclaim_checkpoint() -> None:
    assert module.LICENSE_RECORD["project_code"]["spdx"] == "MIT"
    checkpoint = module.LICENSE_RECORD["official_checkpoint"]
    assert checkpoint["project_code_license_extended_to_weights"] is False
    assert checkpoint["commercial_use_clearance_established"] is False


def test_artifact_size_estimate_is_conservative(release) -> None:
    row = next(row for row in release.inputs if row["condition"] == "real")
    required = module._required_artifact_bytes([row])
    raw_probability_bytes = 2 * 1024 * 1024 * 4 + row["width"] * row["height"] * 4
    assert required > raw_probability_bytes


def test_success_artifact_contract_and_strict_mask(release, tmp_path) -> None:
    pytest.importorskip("cv2")
    input_row = min(
        (row for row in release.inputs if row["condition"] == "real"),
        key=lambda row: int(row["width"]) * int(row["height"]),
    )
    width = int(input_row["width"])
    height = int(input_row["height"])
    sample_id = str(input_row["sample_id"])
    artifact_root = tmp_path / "outputs" / "opensource" / "imlvit" / "unit"
    module._prepare_artifact_root(artifact_root)
    paths = module.artifact_paths(artifact_root, sample_id)
    raw = np.zeros((1024, 1024), dtype=np.float32)
    model_score = np.full((1024, 1024), 0.5, dtype=np.float32)
    native_score = np.full((height, width), 0.5, dtype=np.float32)
    module.legacy._atomic_save_npy(paths["raw_logits_model"], raw)
    module.legacy._atomic_save_npy(paths["score_map_model"], model_score)
    module.legacy._atomic_save_npy(paths["score_map_native"], native_score)
    module.legacy._atomic_save_mask(
        paths["mask_native"], native_score > module.MASK_THRESHOLD
    )
    target = np.zeros((height, width), dtype=bool)
    target_model = module.legacy.model_space_target(
        target,
        resized_width=width,
        resized_height=height,
    )
    preprocess = {
        "native_size": [width, height],
        "resized_content_size": [width, height],
        "resize_policy": "none_image_within_1024_limit",
        "tensor_sha256": "0" * 64,
    }
    result = {
        **module.result_identity(
            input_row,
            run_id="unit",
            run_manifest_fingerprint="0" * 64,
            valid_for_metrics=True,
        ),
        "status": "ok",
        "completed_at": "unit",
        "preprocess": preprocess,
        **module._artifact_fields(
            repo_root=tmp_path,
            paths=paths,
            raw_logits_model=raw,
            score_map_model=model_score,
            score_map_native=native_score,
            resized_size=(width, height),
        ),
        "mask_threshold": module.MASK_THRESHOLD,
        "mask_threshold_operator": module.MASK_THRESHOLD_OPERATOR,
        "localization": {
            "model_1024": module.binary_pixel_metrics_strict(
                model_score[:height, :width],
                target_model,
                module.MASK_THRESHOLD,
                include_ap=False,
            ),
            "native": module.binary_pixel_metrics_strict(
                native_score,
                target,
                module.MASK_THRESHOLD,
                include_ap=False,
            ),
        },
        "latency_ms": 1.0,
        "peak_cuda_memory_bytes": 0,
    }
    module._validate_runner_attempt(
        result,
        input_row=input_row,
        repo_root=tmp_path,
        artifact_root=artifact_root,
        run_id="unit",
        run_manifest_fingerprint="0" * 64,
        verify_artifacts=True,
    )
    assert result["localization"]["native"]["predicted_positive_pixels"] == 0

    from PIL import Image

    Image.fromarray(
        np.full((height, width), 255, dtype=np.uint8),
        mode="L",
    ).save(paths["mask_native"])
    result["mask_sha256"] = module.sha256_file(paths["mask_native"])
    with pytest.raises(ValueError, match="mask/probability relation"):
        module._validate_runner_attempt(
            result,
            input_row=input_row,
            repo_root=tmp_path,
            artifact_root=artifact_root,
            run_id="unit",
            run_manifest_fingerprint="0" * 64,
            verify_artifacts=True,
        )


@pytest.mark.skipif(
    Path(sys.executable) != module.EXPECTED_PYTHON_EXECUTABLE,
    reason="requires frozen IML-ViT virtualenv",
)
def test_full_cpu_preflight_never_initializes_cuda() -> None:
    assert os.environ["PYTHONDONTWRITEBYTECODE"] == "1"
    assert os.environ["PYTHONHASHSEED"] == "0"
    report = module.run_cpu_preflight(
        repo_root=REPO_ROOT,
        imlvit_root=module.legacy.DEFAULT_IMLVIT_ROOT,
        checkpoint_path=module.legacy.DEFAULT_CHECKPOINT,
    )
    assert report["cuda_initialized_before"] is False
    assert report["cuda_initialized_after"] is False
    assert report["model_audit"]["strict_load"] is True
    assert report["model_audit"]["forward_performed"] is False
    assert report["balanced250_score_computed"] is False
