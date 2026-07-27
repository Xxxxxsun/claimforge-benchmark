from __future__ import annotations

import copy
import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from eval.opensource import run_spai_balanced as runner
from eval.opensource.canonical_release import load_canonical_release
from eval.opensource.common import sha256_file, stable_json


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def balanced_release(repo_root: Path):
    return load_canonical_release(
        repo_root,
        repo_root / runner.DEFAULT_DATASET_MANIFEST,
        verify_files=False,
    )


@pytest.fixture(scope="session")
def formal_selection(balanced_release):
    return runner.select_mode_inputs(
        balanced_release,
        mode="formal",
        per_condition_limit=None,
        sample_id=None,
    )[1]


def test_frozen_method_and_runtime_contracts() -> None:
    assert runner.RUN_MANIFEST_SCHEMA == "spai_balanced_run_manifest_v2"
    assert runner.RUN_CONFIG_SCHEMA == "spai_balanced_run_config_v2"
    assert runner.CPU_PREFLIGHT_SCHEMA == "spai_balanced_cpu_preflight_v1"
    assert runner.DEFAULT_SEED == 0
    assert runner.SCORE_SPEC.as_dict() == {
        "key": "ai_score",
        "direction": "higher_means_fake",
        "fixed_threshold": 0.5,
        "threshold_operator": ">",
    }
    assert runner.FROZEN_PYVENV_CONFIG_SHA256 == (
        "506c01d6bc866a7500bde63a24b3f0c1fb3013df41051ad9a8bf7c42c85eb091"
    )
    assert runner.FROZEN_RUNTIME_VERSIONS["timm"] == "0.4.12"
    assert runner.FROZEN_RUNTIME_MODULE_FILES["torch"] == (
        "abc68f909360770fb0dd0fc263b43ae65906bd66d1eab99cdcf5c5abf23c0e0d"
    )


def test_model_contract_discloses_rights_and_execution_mismatches() -> None:
    model = runner.MODEL_CONTRACT
    assert model["source_commit"] == (
        "8ff7b3b6779b4fcb43cf313471d9cb1c62d129a4"
    )
    assert model["checkpoint"]["sha256"] == (
        "24159f27d7c8c2cd0cb6c4019189eb89ad0874a0d9d15f8dc9afd39ca9648a55"
    )
    assert model["checkpoint"]["loader"].endswith("weights_only=True)")
    assert model["license"]["commercial_clearance"] == "unresolved"
    assert model["license"]["risk"] == "high"
    reason = model["license"]["reason"]
    assert "S-Lab License 1.0" in reason
    assert "DMimageDetection" in reason
    assert "COCO/LSUN" in reason
    execution = model["released_execution"]
    assert execution["config_minimum_patches"] == 4
    assert execution["checkpoint_embedded_historical_minimum_patches"] == 1
    assert execution[
        "executable_discards_nondivisible_right_bottom_remainder"
    ] is True
    assert execution["project_attention_visualization_is_not_native_T2"] is True


def test_preprocess_and_three_artifact_contracts_are_exact() -> None:
    assert runner.PREPROCESS_CONTRACT["resize"] is False
    assert runner.PREPROCESS_CONTRACT["crop"] is False
    assert runner.PREPROCESS_CONTRACT["minimum_patches"] == 4
    assert set(runner.ARTIFACT_CONTRACT) == {
        "patch_features",
        "feature",
        "attention",
        "replay",
    }
    assert runner.ARTIFACT_CONTRACT["patch_features"]["shape"] == [
        "effective_patch_count",
        1096,
    ]
    assert runner.ARTIFACT_CONTRACT["attention"]["shape"] == [
        12,
        "effective_patch_count",
    ]
    assert runner.ARTIFACT_CONTRACT["attention"]["valid_for_t2"] is False
    assert runner.TASK_SCOPE == {
        "primary_task": "T1_whole_image_AIGC_detection",
        "valid_for_t1": True,
        "valid_for_t2": False,
        "localization_output": None,
        "native_dense_output": False,
    }


def test_adapter_source_inventory_binds_balanced_analyzer(repo_root: Path) -> None:
    assert "eval/opensource/analyze_spai_balanced.py" in (
        runner.ADAPTER_SOURCE_PATHS
    )
    records = runner.adapter_source_contract(repo_root)
    assert tuple(records) == runner.ADAPTER_SOURCE_PATHS
    for relative, record in records.items():
        path = repo_root / relative
        assert record == {
            "path": relative,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }


def test_formal_selection_is_exact(formal_selection) -> None:
    assert len(formal_selection) == 1775
    assert Counter(row["condition"] for row in formal_selection) == (
        runner.FORMAL_COUNTS
    )
    assert runner._rows_sha256(formal_selection) == (
        runner.FORMAL_SELECTED_ROWS_SHA256
    )
    assert runner.selected_ids_sha256(
        str(row["sample_id"]) for row in formal_selection
    ) == runner.FORMAL_SELECTED_IDS_SHA256
    assert all("pair_rank" not in row for row in formal_selection)


def test_smoke_selection_is_five_per_condition(balanced_release) -> None:
    spec, selected = runner.select_mode_inputs(
        balanced_release,
        mode="smoke",
        per_condition_limit=5,
        sample_id=None,
    )
    assert spec.per_condition_limit == 5
    assert len(selected) == 35
    assert Counter(row["condition"] for row in selected) == {
        condition: 5 for condition in runner.BALANCED_CONDITIONS
    }
    assert runner.selected_ids_sha256(
        str(row["sample_id"]) for row in selected
    ) == runner.SMOKE5X7_SELECTED_IDS_SHA256


def test_single_selection_and_mode_rejections(balanced_release) -> None:
    sample_id = str(balanced_release.inputs[43]["sample_id"])
    spec, selected = runner.select_mode_inputs(
        balanced_release,
        mode="single",
        per_condition_limit=None,
        sample_id=sample_id,
    )
    assert spec.sample_id == sample_id
    assert [row["sample_id"] for row in selected] == [sample_id]
    with pytest.raises(ValueError, match="formal mode"):
        runner.select_mode_inputs(
            balanced_release,
            mode="formal",
            per_condition_limit=1,
            sample_id=None,
        )
    with pytest.raises(ValueError, match="requires --sample-id"):
        runner.select_mode_inputs(
            balanced_release,
            mode="single",
            per_condition_limit=None,
            sample_id=None,
        )
    with pytest.raises(ValueError, match=r"\[1, 250\]"):
        runner.select_mode_inputs(
            balanced_release,
            mode="smoke",
            per_condition_limit=0,
            sample_id=None,
        )


def test_formal_visibility_census_is_frozen(
    formal_selection,
    repo_root: Path,
) -> None:
    census = runner.selection_visibility_census(
        formal_selection,
        repo_root=repo_root,
    )
    assert census["by_condition"] == {
        key: runner.LOCAL_VISIBILITY_CENSUS[key]
        for key in ("local_mouse", "local_cat", "local_trash_can")
    }
    assert census["all_local"] == runner.LOCAL_VISIBILITY_CENSUS["all_local"]
    assert census["not_applicable_images"] == 1025
    assert census["role"] == "input_condition_stratum_not_model_localization"


def test_real_and_fullframe_visibility_are_not_localization(
    formal_selection,
    repo_root: Path,
) -> None:
    real = next(row for row in formal_selection if row["condition"] == "real")
    full = next(
        row for row in formal_selection if row["condition"] == "fullframe_cat"
    )
    for row in (real, full):
        value = runner._visibility_diagnostic(row, repo_root=repo_root)
        assert value["edit_visibility"] == "not_applicable"
        assert value["edit_visible_gt_fraction"] is None
        assert value["edit_visibility_evidence"]["geometry"][
            "effective_patch_count"
        ] > 0


@pytest.mark.parametrize(
    "value",
    ["", ".", "..", "../escape", "a/b", "with space", "x" * 161],
)
def test_run_id_safety(value: str) -> None:
    with pytest.raises(ValueError):
        runner._valid_run_id(value)


@pytest.mark.parametrize(
    "payload",
    [
        {"t2": {}},
        {"valid_for_t2": True},
        {"pixel_auroc": 0.9},
        {"attention_map": "fake.npy"},
        {"nested": {"attention_map_path": "fake.npy"}},
        {"nested": {"attention_mask": [1, 0]}},
        {"nested": {"attention_mask_path": "fake.npy"}},
        {"joint_output": {}},
        {"s_joint": 0.1},
        {"nested": {"mask_path": "fake.png"}},
    ],
)
def test_t2_joint_and_attention_map_claims_are_rejected(payload) -> None:
    with pytest.raises(ValueError, match="unsupported SPAI claim"):
        runner._reject_unsupported_claims(payload)


def test_false_and_null_scope_declarations_are_allowed() -> None:
    runner._reject_unsupported_claims(
        {
            "valid_for_t2": False,
            "native_dense_output": False,
            "localization_output": None,
            "t2": None,
        }
    )


def _tiny_model():
    import torch

    class TinySPAI(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.heads = runner.legacy.ATTENTION_HEADS
            self.scale = (
                runner.legacy.ATTENTION_EMBED_DIMENSION // self.heads
            ) ** -0.5
            self.to_kv = torch.nn.Linear(
                runner.legacy.FEATURE_DIMENSION,
                runner.legacy.ATTENTION_EMBED_DIMENSION * 2,
                bias=False,
            )
            self.patch_aggregator = torch.nn.Parameter(
                torch.zeros(self.heads, 1, 128)
            )
            self.attend = torch.nn.Softmax(dim=-1)
            self.dropout = torch.nn.Dropout(0.5)
            self.to_out = torch.nn.Sequential(
                torch.nn.Linear(
                    runner.legacy.ATTENTION_EMBED_DIMENSION,
                    runner.legacy.FEATURE_DIMENSION,
                    bias=False,
                ),
                torch.nn.Dropout(0.5),
            )
            self.norm = torch.nn.LayerNorm(runner.legacy.FEATURE_DIMENSION)
            self.cls_head = torch.nn.Sequential(
                torch.nn.Linear(runner.legacy.FEATURE_DIMENSION, 1)
            )

        def forward(self, images, feature_extraction_batch_size=None):
            del feature_extraction_batch_size
            image = images[0]
            geometry = runner.legacy.compute_patch_geometry(
                int(image.shape[3]),
                int(image.shape[2]),
            )
            patches = int(geometry["effective_patch_count"])
            base = torch.linspace(
                -0.1,
                0.1,
                runner.legacy.FEATURE_DIMENSION,
                dtype=torch.float32,
                device=image.device,
            )
            x = torch.stack(
                [base + index * 0.001 for index in range(patches)]
            ).unsqueeze(0)
            key, value = self.to_kv(x).chunk(2, dim=-1)
            key = key.reshape(1, patches, self.heads, 128).permute(
                0, 2, 1, 3
            )
            value = value.reshape(1, patches, self.heads, 128).permute(
                0, 2, 1, 3
            )
            aggregator = self.patch_aggregator.expand(1, -1, -1, -1)
            attention = self.attend(
                torch.matmul(aggregator, key.transpose(-1, -2)) * self.scale
            )
            attended = torch.matmul(self.dropout(attention), value)
            attended = (
                attended.permute(0, 2, 1, 3)
                .contiguous()
                .reshape(1, 1, runner.legacy.ATTENTION_EMBED_DIMENSION)
            )
            x = self.to_out(attended).squeeze(1)
            x = self.norm(x)
            return self.cls_head(x)

    torch.manual_seed(4)
    return TinySPAI().eval()


def _local_fixture(tmp_path: Path):
    import torch

    sample_id = "0123456789abcdef01234567"
    image_relative = f"images/{sample_id}.jpg"
    image_path = tmp_path / image_relative
    image_path.parent.mkdir(parents=True)
    Image.new("RGB", (224, 224), color=(20, 40, 60)).save(
        image_path,
        quality=95,
    )
    mask_relative = f"masks/{sample_id}.png"
    mask_path = tmp_path / mask_relative
    mask_path.parent.mkdir(parents=True)
    mask = np.zeros((224, 224), dtype=np.uint8)
    mask[10:30, 15:35] = 255
    Image.fromarray(mask, mode="L").save(mask_path)
    canonical = {
        "dataset_id": runner.BALANCED_DATASET_ID,
        "sample_id": sample_id,
        "rank": 1,
        "condition": "local_mouse",
        "condition_family": "local_splice",
        "manipulation_scope": "local_insertion",
        "normalized_task_id": "task-1",
        "task_id": "task-1",
        "kind": "forged",
        "label": 1,
        "domain": "restaurant",
        "gt_mask_kind": "exact_diff",
        "gt_mask_path": mask_relative,
        "gt_mask_sha256": sha256_file(mask_path),
        "gt_positive_pixels": 400,
        "canonical_path": image_relative,
        "canonical_sha256": sha256_file(image_path),
        "width": 224,
        "height": 224,
    }
    image, preprocess = runner.legacy.preprocess_image(image_path)
    model = _tiny_model()
    processed, patch, feature, attention, peak, latency = (
        runner.legacy.infer_one(
            model,
            torch.device("cpu"),
            image,
        )
    )
    run_id = "test-run"
    fingerprint = "a" * 64
    artifact_root = (
        tmp_path / runner.DEFAULT_ARTIFACTS_DIR / run_id
    )
    result = runner._build_ok_result(
        input_row=canonical,
        repo_root=tmp_path,
        run_id=run_id,
        fingerprint=fingerprint,
        artifact_root=artifact_root,
        processed=processed,
        patch=patch,
        feature=feature,
        attention=attention,
        preprocess=preprocess,
        preprocess_latency_ms=1.0,
        latency_ms=latency,
        peak_cuda_memory_bytes=0 if peak is None else peak,
    )
    return canonical, result, model, run_id, fingerprint, artifact_root


def test_ok_result_persists_three_canonical_arrays_and_replays(
    tmp_path: Path,
) -> None:
    import torch

    canonical, result, model, run_id, fingerprint, _root = _local_fixture(
        tmp_path
    )
    arrays = runner._validate_runner_attempt(
        result,
        input_row=canonical,
        repo_root=tmp_path,
        run_id=run_id,
        run_manifest_fingerprint=fingerprint,
    )
    assert arrays is not None
    patch, feature, attention = arrays
    assert patch.shape == (5, 1096)
    assert feature.shape == (1096,)
    assert attention.shape == (12, 5)
    assert np.allclose(attention.sum(axis=1), 1.0)
    assert set(result["artifact_paths"]) == {
        "spai_patch_features_npy",
        "spai_feature_npy",
        "spai_attention_npy",
    }
    assert result["attention_is_diagnostic_not_t2"] is True
    runner._replay_artifacts(
        row=result,
        arrays=arrays,
        model=model,
        device=torch.device("cpu"),
    )


@pytest.mark.parametrize(
    "kind,field",
    [
        ("patch_features", "spai_patch_features"),
        ("feature", "spai_feature"),
        ("attention", "spai_attention"),
    ],
)
def test_each_artifact_tamper_is_rejected(
    tmp_path: Path,
    kind: str,
    field: str,
) -> None:
    canonical, result, _model, run_id, fingerprint, artifact_root = (
        _local_fixture(tmp_path)
    )
    path = artifact_root / kind / f"{canonical['sample_id']}.npy"
    array = np.load(path, allow_pickle=False)
    array.reshape(-1)[0] += np.float32(1.0)
    runner.legacy._atomic_save_npy(path, array)
    with pytest.raises(ValueError, match=rf"{kind} metadata/hash"):
        runner._validate_runner_attempt(
            result,
            input_row=canonical,
            repo_root=tmp_path,
            run_id=run_id,
            run_manifest_fingerprint=fingerprint,
        )
    assert field in result


def test_oversized_artifact_is_rejected_before_numpy_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    canonical, result, _model, run_id, fingerprint, artifact_root = (
        _local_fixture(tmp_path)
    )
    path = artifact_root / "feature" / f"{canonical['sample_id']}.npy"
    path.write_bytes(path.read_bytes() + b"x")
    original_load = runner.np.load

    def guarded_load(candidate, *args, **kwargs):
        if Path(candidate) == path:
            pytest.fail("oversized NPY must be rejected before np.load")
        return original_load(candidate, *args, **kwargs)

    monkeypatch.setattr(
        runner.np,
        "load",
        guarded_load,
    )
    with pytest.raises(ValueError, match="feature metadata/hash"):
        runner._validate_runner_attempt(
            result,
            input_row=canonical,
            repo_root=tmp_path,
            run_id=run_id,
            run_manifest_fingerprint=fingerprint,
        )


def test_manual_replay_metadata_tamper_is_rejected(tmp_path: Path) -> None:
    canonical, result, _model, run_id, fingerprint, _root = _local_fixture(
        tmp_path
    )
    tampered = copy.deepcopy(result)
    tampered["manual_replay"]["norm_hook_calls"] = 2
    with pytest.raises(ValueError, match="score/replay"):
        runner._validate_runner_attempt(
            tampered,
            input_row=canonical,
            repo_root=tmp_path,
            run_id=run_id,
            run_manifest_fingerprint=fingerprint,
        )


def test_score_nested_booleans_cannot_impersonate_numbers(
    tmp_path: Path,
) -> None:
    canonical, result, _model, run_id, fingerprint, _root = _local_fixture(
        tmp_path
    )
    hook_tampered = copy.deepcopy(result)
    hook_tampered["manual_replay"]["norm_hook_calls"] = True
    with pytest.raises(ValueError, match="score/replay"):
        runner._validate_runner_attempt(
            hook_tampered,
            input_row=canonical,
            repo_root=tmp_path,
            run_id=run_id,
            run_manifest_fingerprint=fingerprint,
        )

    scalar_tampered = copy.deepcopy(result)
    scalar_tampered["raw_logit"] = 1.0
    scalar_tampered["classification"]["raw_logit"] = True
    scalar_tampered["t1"]["raw_logit"] = 1.0
    scalar_tampered["manual_replay"]["raw_logit"] = 1.0
    with pytest.raises(ValueError, match="score/replay"):
        runner._validate_runner_attempt(
            scalar_tampered,
            input_row=canonical,
            repo_root=tmp_path,
            run_id=run_id,
            run_manifest_fingerprint=fingerprint,
        )


def test_identity_boolean_cannot_impersonate_integer(
    tmp_path: Path,
) -> None:
    canonical, result, _model, run_id, fingerprint, _root = _local_fixture(
        tmp_path
    )
    assert canonical["rank"] == 1
    tampered = copy.deepcopy(result)
    tampered["rank"] = True
    with pytest.raises(ValueError, match="field rank drifted"):
        runner._validate_runner_attempt(
            tampered,
            input_row=canonical,
            repo_root=tmp_path,
            run_id=run_id,
            run_manifest_fingerprint=fingerprint,
        )


def test_error_result_is_t1_only(tmp_path: Path) -> None:
    canonical, _result, _model, run_id, fingerprint, _root = _local_fixture(
        tmp_path
    )
    try:
        raise RuntimeError("expected")
    except RuntimeError as error:
        row = runner._build_error_result(
            input_row=canonical,
            repo_root=tmp_path,
            run_id=run_id,
            fingerprint=fingerprint,
            error=error,
        )
    assert row["valid_for_metrics"] is False
    assert row["task_scope"]["valid_for_t2"] is False
    assert "artifact_paths" not in row
    assert runner._validate_runner_attempt(
        row,
        input_row=canonical,
        repo_root=tmp_path,
        run_id=run_id,
        run_manifest_fingerprint=fingerprint,
    ) is None


def test_attempt_history_allows_error_retry_but_not_post_success() -> None:
    runner._validate_physical_attempt_history(
        [
            {"sample_id": "a", "status": "error"},
            {"sample_id": "a", "status": "error"},
            {"sample_id": "a", "status": "ok"},
        ]
    )
    with pytest.raises(ValueError, match="after success"):
        runner._validate_physical_attempt_history(
            [
                {"sample_id": "a", "status": "ok"},
                {"sample_id": "a", "status": "error"},
            ]
        )
    with pytest.raises(ValueError, match="duplicate successful"):
        runner._validate_physical_attempt_history(
            [
                {"sample_id": "a", "status": "ok"},
                {"sample_id": "a", "status": "ok"},
            ]
        )


def test_artifact_inventory_requires_all_three_exact_directories(
    tmp_path: Path,
) -> None:
    canonical, result, _model, _run_id, _fingerprint, root = _local_fixture(
        tmp_path
    )
    latest = {canonical["sample_id"]: result}
    assert runner._validate_artifact_inventory(
        artifact_root=root,
        latest_by_sample_id=latest,
    ) == 3
    extra = root / "attention" / "extra.npy"
    np.save(extra, np.zeros(1, dtype=np.float32), allow_pickle=False)
    with pytest.raises(ValueError, match="inventory mismatch"):
        runner._validate_artifact_inventory(
            artifact_root=root,
            latest_by_sample_id=latest,
        )


def test_output_directories_are_disjoint_resume_safe_and_stale_guarded(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "results" / "run"
    artifact_root = tmp_path / "outputs" / "run"
    runner._prepare_output_directories(
        repo_root=tmp_path,
        run_dir=run_dir,
        artifact_root=artifact_root,
        resume=False,
    )
    assert {entry.name for entry in artifact_root.iterdir()} == {
        "patch_features",
        "feature",
        "attention",
    }
    (run_dir / "manifest.json").write_text("{}\n", encoding="utf-8")
    (run_dir / "expected_inputs.jsonl").write_text(
        f"{stable_json({'sample_id': 'a'})}\n",
        encoding="utf-8",
    )
    runner._prepare_output_directories(
        repo_root=tmp_path,
        run_dir=run_dir,
        artifact_root=artifact_root,
        resume=True,
    )
    (run_dir / "balanced250_metrics.json").write_text(
        "{}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="forbidden after analyzer"):
        runner._prepare_output_directories(
            repo_root=tmp_path,
            run_dir=run_dir,
            artifact_root=artifact_root,
            resume=True,
        )
    with pytest.raises(ValueError, match="disjoint"):
        runner._prepare_output_directories(
            repo_root=tmp_path,
            run_dir=tmp_path / "same",
            artifact_root=tmp_path / "same",
            resume=False,
        )


def test_strict_json_rejects_duplicate_keys_constants_and_noncanonical_jsonl(
    tmp_path: Path,
) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"x":1,"x":2}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate JSON key"):
        runner._load_json_strict(duplicate, "duplicate")
    constant = tmp_path / "constant.json"
    constant.write_text('{"x":NaN}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="non-finite JSON constant"):
        runner._load_json_strict(constant, "constant")
    jsonl = tmp_path / "rows.jsonl"
    jsonl.write_text('{"b": 2, "a": 1}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="not canonical JSONL"):
        runner._read_jsonl_strict(jsonl, "rows")


def test_local_artifact_policy_is_gitignored(repo_root: Path) -> None:
    policy = runner._local_artifact_policy(repo_root)
    assert policy["artifact_root"] == "outputs/opensource/spai"
    assert policy["gitignored"] is True
    assert policy["publication"] is False
    assert policy["checkpoint_redistribution"] is False
    assert policy["commercial_clearance_claimed"] is False


def test_cpu_golden_constants_cover_three_arrays() -> None:
    assert runner.CPU_GOLDEN_SAMPLE_ID == "143a80a0a7a34c757d67ff25"
    assert runner.CPU_GOLDEN_RAW_LOGIT == -19.73525619506836
    assert runner.CPU_GOLDEN_PROBABILITY == 2.685883293551683e-09
    assert runner.CPU_GOLDEN_PATCH_ARRAY_SHA256 == (
        "fc8c3ba429aac54a076d58438640cd760cb6ccbee4f150d6cb6fc177cbf831e1"
    )
    assert runner.CPU_GOLDEN_FEATURE_ARRAY_SHA256 == (
        "cea88315feea8612d3f069298ec82f27449bdd16e9d41cd7b2fa6c7b2b72beda"
    )
    assert runner.CPU_GOLDEN_ATTENTION_ARRAY_SHA256 == (
        "85f0aec70d4f4cf80d367e4ab4a22f0395395e8b9329a1201fb821ae1877dad0"
    )
    assert runner.CPU_OFFICIAL_GOLDEN_CASES[0]["raw_logit"] == (
        0.9909074306488037
    )
    assert runner.CPU_OFFICIAL_GOLDEN_CASES[1]["raw_logit"] == (
        1.6814380884170532
    )
    assert (
        runner.CPU_OFFICIAL_GOLDEN_CASES[0]["raw_logit"]
        != runner.legacy.GOLDEN_CASES[0]["raw_logit"]
    )


def test_execution_device_gate_dispatches_without_cross_device_equality(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Device:
        type = "cuda"

        def __str__(self) -> str:
            return "cuda:3"

    sentinel = {"status": "passed", "cases": []}
    monkeypatch.setattr(
        runner.legacy,
        "validate_official_golden",
        lambda **_kwargs: sentinel,
    )
    report = runner.validate_execution_device_golden(
        model=object(),
        device=Device(),
        golden_root=Path("/unused"),
    )
    assert report == {
        "status": "passed",
        "device": "cuda:3",
        "reference_device": "cuda",
        "gate": "released_CUDA_highest_no_TF32_implementation_regression",
        "cross_device_bit_equality_required": False,
        "report": sentinel,
    }


def test_parser_modes_defaults_and_preflight_option_rejection(
    monkeypatch: pytest.MonkeyPatch,
    repo_root: Path,
) -> None:
    parser = runner._build_parser()
    args = parser.parse_args([])
    assert args.mode == "formal"
    assert args.seed == 0
    assert args.device is None
    assert args.resume is False
    with pytest.raises(ValueError, match="accepts no"):
        runner.run(
            parser.parse_args(
                [
                    "--repo-root",
                    str(repo_root),
                    "--mode",
                    "preflight",
                    "--run-id",
                    "bad",
                ]
            )
        )
    monkeypatch.setattr(runner, "run_cpu_preflight", lambda **_kwargs: {})
    with pytest.raises(ValueError, match="accepts no"):
        runner.run(
            parser.parse_args(
                [
                    "--repo-root",
                    str(repo_root),
                    "--mode",
                    "preflight",
                    "--device",
                    "cuda:0",
                ]
            )
        )


def _in_frozen_runtime() -> bool:
    return (
        Path(os.path.abspath(sys.executable))
        == Path(os.path.abspath(runner.FROZEN_PYTHON_EXECUTABLE))
        and os.environ.get("PYTHONHASHSEED") == "0"
        and os.environ.get("PYTHONDONTWRITEBYTECODE") == "1"
        and os.environ.get("PYTHONPYCACHEPREFIX")
        == str(runner.FROZEN_PYTHONPYCACHEPREFIX)
    )


@pytest.mark.skipif(
    not _in_frozen_runtime(),
    reason="real SPAI CPU preflight requires the frozen isolated venv",
)
def test_real_cpu_preflight_is_cuda_free_and_bit_exact(
    repo_root: Path,
) -> None:
    report = runner.run_cpu_preflight(
        repo_root=repo_root,
        source_root=runner.DEFAULT_SOURCE_ROOT,
        checkpoint_path=runner.DEFAULT_CHECKPOINT,
        golden_root=runner.DEFAULT_GOLDEN_ROOT,
    )
    source, assets, state = runner.verify_assets(
        source_root=runner.DEFAULT_SOURCE_ROOT,
        checkpoint_path=runner.DEFAULT_CHECKPOINT,
        golden_root=runner.DEFAULT_GOLDEN_ROOT,
    )
    del state
    runner._validate_preflight_report(
        report,
        source=source,
        assets=assets,
    )
    assert report["status"] == "passed"
    assert report["runtime"]["device"] == "cpu"
    assert report["cuda_used"] is False
    assert report["balanced_golden"]["repeat_byte_exact"] is True
    runtime_mutations = (
        (("seed",), False),
        (("batch_size",), True),
        (("venv", "include_system_site_packages"), 1),
        (("cudnn", "enabled"), 1),
        (("process_environment", "python_dont_write_bytecode"), 1),
    )
    for keys, replacement in runtime_mutations:
        tampered = copy.deepcopy(report["runtime"])
        target = tampered
        for key in keys[:-1]:
            target = target[key]
        target[keys[-1]] = replacement
        with pytest.raises(ValueError, match="changed"):
            runner.validate_runtime_contract(tampered)
