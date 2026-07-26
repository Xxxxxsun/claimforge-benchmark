from __future__ import annotations

import copy
import hashlib
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from eval.opensource import analyze_spai_run as audit
from eval.opensource.common import sha256_file


def _write_rgb(path: Path, array: np.ndarray) -> None:
    Image.fromarray(array.astype(np.uint8), mode="RGB").save(path)


def _write_mask(path: Path, array: np.ndarray) -> None:
    Image.fromarray(array.astype(np.uint8), mode="L").save(path)


def _score_fields(raw_logit: float) -> dict:
    raw = float(np.float32(raw_logit))
    probability = float(
        torch.sigmoid(torch.tensor(raw, dtype=torch.float32)).item()
    )
    decision = probability > 0.5
    return {
        "raw_logit": raw,
        "probability": probability,
        "ai_score": probability,
        "score": probability,
        "score_semantics": audit.SCORE_SEMANTICS,
        "classification_decision": decision,
        "classification_threshold": 0.5,
        "classification_threshold_operator": ">",
        "classification": {
            "raw_logit": raw,
            "probability": probability,
            "ai_score": probability,
            "score": probability,
            "threshold": 0.5,
            "threshold_operator": ">",
            "decision": decision,
            "semantics": audit.SCORE_SEMANTICS,
        },
        "t1": {
            "raw_logit": raw,
            "probability": probability,
            "ai_score": probability,
            "score": probability,
            "threshold": 0.5,
            "threshold_operator": ">",
            "decision": decision,
            "policy": audit.T1_POLICY,
        },
        "manual_replay": {
            "raw_logit": raw,
            "probability": probability,
            "ai_score": probability,
            "classification_decision": decision,
            "model_forward_calls": 1,
            "patch_feature_hook_calls": 1,
            "sca_attention_exact_match": True,
            "sca_feature_exact_match": True,
            "official_logit_exact_match": True,
            "official_probability_exact_match": True,
        },
    }


def _save_array(path: Path, array: np.ndarray) -> tuple[str, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    contiguous = np.ascontiguousarray(array, dtype=np.float32)
    np.save(path, contiguous, allow_pickle=False)
    return sha256_file(path), hashlib.sha256(
        contiguous.tobytes(order="C")
    ).hexdigest()


def _artifact_fields(
    *,
    root: Path,
    run_dir: Path,
    sample_id: str,
    patch: np.ndarray,
    feature: np.ndarray,
    attention: np.ndarray,
) -> dict:
    arrays = {
        "spai_patch_features": (
            patch,
            audit.PATCH_FEATURE_SEMANTICS,
            "spai_patch_features_npy",
        ),
        "spai_feature": (
            feature,
            audit.FEATURE_SEMANTICS,
            "spai_feature_npy",
        ),
        "spai_attention": (
            attention,
            audit.ATTENTION_SEMANTICS,
            "spai_attention_npy",
        ),
    }
    result: dict = {}
    artifact_paths = {}
    for prefix, (array, semantics, path_key) in arrays.items():
        path = run_dir / "features" / f"{sample_id}_{prefix}.npy"
        file_digest, array_digest = _save_array(path, array)
        relative = str(path.relative_to(root))
        result.update(
            {
                f"{prefix}_path": relative,
                f"{prefix}_sha256": file_digest,
                f"{prefix}_array_sha256": array_digest,
                f"{prefix}_shape": list(array.shape),
                f"{prefix}_dtype": "float32",
                f"{prefix}_semantics": semantics,
            }
        )
        artifact_paths[path_key] = relative
    result["feature_array_sha256"] = result["spai_feature_array_sha256"]
    result["artifact_paths"] = artifact_paths
    return result


def _official_golden_fixture(root: Path) -> tuple[dict, list[dict]]:
    assets: list[dict] = []
    cases: list[dict] = []
    for index, frozen_value in enumerate(audit.FROZEN_GOLDEN_CASES):
        frozen = copy.deepcopy(frozen_value)
        path = (root / str(frozen["relative_path"])).resolve()
        assets.append(
            {
                **copy.deepcopy(frozen),
                "path": str(path),
                "bytes": 1000 + index,
            }
        )
        observed = {
            "raw_logit": frozen["raw_logit"],
            "probability": frozen["probability"],
        }
        hashes = {
            "patch_features": f"{index + 1:x}" * 64,
            "feature": f"{index + 3:x}" * 64,
            "attention": f"{index + 5:x}" * 64,
        }
        cases.append(
            {
                **copy.deepcopy(frozen),
                "path": str(path),
                "preprocess": {
                    "profile": audit.PREPROCESS_PROFILE,
                    "native_size": copy.deepcopy(frozen["native_size"]),
                    "decoded_rgb_sha256": frozen["decoded_rgb_sha256"],
                    "tensor_sha256": frozen["tensor_sha256"],
                    "geometry": {
                        "effective_patch_count": frozen["patch_count"],
                    },
                },
                "observed_runs": [
                    copy.deepcopy(observed),
                    copy.deepcopy(observed),
                ],
                "artifact_hashes": [
                    copy.deepcopy(hashes),
                    copy.deepcopy(hashes),
                ],
                "bit_identical_repeats": True,
                "logit_absolute_difference": 0.0,
                "probability_absolute_difference": 0.0,
                "passed": True,
                (
                    "website_derivative_display_matches_released_regression"
                ): False,
            }
        )
    return {
        "status": "passed",
        "source": audit.FROZEN_GOLDEN_SOURCE,
        "runs_per_case": 2,
        "logit_absolute_tolerance": (
            audit.FROZEN_GOLDEN_LOGIT_ABSOLUTE_TOLERANCE
        ),
        "probability_absolute_tolerance": (
            audit.FROZEN_GOLDEN_PROBABILITY_ABSOLUTE_TOLERANCE
        ),
        "official_vs_adapter_full_forward": True,
        "website_display_reference": (
            audit.FROZEN_GOLDEN_WEBSITE_DISPLAY_REFERENCE
        ),
        "cases": cases,
    }, assets


def _set_nested(value, path: tuple, replacement) -> None:
    cursor = value
    for component in path[:-1]:
        cursor = cursor[component]
    cursor[path[-1]] = replacement


def test_grid_geometry_and_native_rgb_preprocess(tmp_path: Path) -> None:
    height, width = 450, 500
    yy, xx = np.mgrid[:height, :width]
    rgb = np.stack(
        (
            (xx * 17 + yy * 3) % 256,
            (xx * 5 + yy * 19) % 256,
            (xx * 13 + yy * 7) % 256,
        ),
        axis=-1,
    ).astype(np.uint8)
    path = tmp_path / "grid.png"
    _write_rgb(path, rgb)

    prepared = audit.preprocess_image(path, torch_module=torch)

    expected = torch.from_numpy(
        np.ascontiguousarray(rgb.transpose(2, 0, 1))
    ).float() * np.float32(1.0 / 255.0)
    assert torch.equal(prepared.tensor, expected)
    assert prepared.geometry["patch_mode"] == "grid"
    assert prepared.geometry["initial_grid"] == {
        "rows": 2,
        "columns": 2,
        "count": 4,
    }
    assert prepared.geometry["grid_covered_xyxy"] == [0, 0, 448, 448]
    assert prepared.geometry["remainder_policy"].startswith(
        "torch_tensor_unfold"
    )
    patches = audit._patchify_tensor(
        prepared.tensor,
        prepared.geometry,
        torch_module=torch,
    )
    assert patches.shape == (4, 3, 224, 224)
    assert torch.equal(patches[0], expected[:, :224, :224])
    assert torch.equal(patches[3], expected[:, 224:448, 224:448])


def test_five_crop_geometry_is_torchvision_exact(tmp_path: Path) -> None:
    height, width = 224, 700
    rgb = np.arange(height * width * 3, dtype=np.uint8).reshape(
        height,
        width,
        3,
    )
    path = tmp_path / "five.png"
    _write_rgb(path, rgb)
    prepared = audit.preprocess_image(path, torch_module=torch)

    assert prepared.geometry["initial_grid"]["count"] == 3
    assert prepared.geometry["patch_mode"] == "five_crop"
    assert prepared.geometry["effective_patch_count"] == 5
    assert prepared.geometry["five_crop_boxes_xyxy"] == [
        [0, 0, 224, 224],
        [476, 0, 700, 224],
        [0, 0, 224, 224],
        [476, 0, 700, 224],
        [238, 0, 462, 224],
    ]
    patches = audit._patchify_tensor(
        prepared.tensor,
        prepared.geometry,
        torch_module=torch,
    )
    from torchvision.transforms.functional import five_crop

    expected = torch.stack(
        five_crop(prepared.tensor, [224, 224]),
        dim=0,
    )
    assert torch.equal(patches, expected)


def test_visibility_uses_union_of_selected_patches(tmp_path: Path) -> None:
    height, width = 500, 500
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[10, 10] = 255
    mask[480, 480] = 255
    path = tmp_path / "mask.png"
    _write_mask(path, mask)
    canonical = {
        "sample_id": "forged",
        "task_id": "task",
        "kind": "forged",
        "label": 1,
        "width": width,
        "height": height,
        "gt_mask_kind": "exact_diff",
        "gt_mask_path": str(path),
        "gt_mask_sha256": sha256_file(path),
        "gt_positive_pixels": 2,
    }
    visibility = audit._visibility_from_exact_gt(
        canonical,
        repo_root=tmp_path,
    )
    assert visibility["category"] == "partial"
    assert visibility["visible_positive_pixels"] == 1
    assert visibility["visible_fraction"] == 0.5
    assert visibility["geometry"]["patch_mode"] == "grid"


def test_five_crop_visibility_does_not_treat_attention_as_localization(
    tmp_path: Path,
) -> None:
    height, width = 224, 700
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[10, 10] = 255
    mask[10, 230] = 255  # lies in the gap between TL and center crops
    path = tmp_path / "five_mask.png"
    _write_mask(path, mask)
    canonical = {
        "sample_id": "forged",
        "task_id": "task",
        "kind": "forged",
        "label": 1,
        "width": width,
        "height": height,
        "gt_mask_kind": "exact_diff",
        "gt_mask_path": str(path),
        "gt_mask_sha256": sha256_file(path),
        "gt_positive_pixels": 2,
    }
    visibility = audit._visibility_from_exact_gt(
        canonical,
        repo_root=tmp_path,
    )
    assert visibility["geometry"]["patch_mode"] == "five_crop"
    assert visibility["visible_fraction"] == 0.5


@pytest.mark.parametrize(
    "payload",
    [
        {"t2": {}},
        {"localization": None},
        {"nested": {"attention_map_path": "invented.png"}},
        {"metrics": [{"s_joint": 0.4}]},
        {"pixel_metrics": {}},
        {"config": {"model_contract": {"valid_for_t2": True}}},
        {"outer": [{"inner": [{"valid_for_t2": True}]}]},
    ],
)
def test_t2_localization_and_joint_fields_are_rejected(payload) -> None:
    with pytest.raises(ValueError, match="unsupported"):
        audit._reject_t2_localization_or_joint(payload, label="payload")
    audit._reject_t2_localization_or_joint(
        {
            "task_scope": {"t2": False},
            "valid_for_t2": False,
            "attention_is_diagnostic_not_T2": True,
            "spai_attention_path": "internal.npy",
        },
        label="diagnostic evidence",
    )


@pytest.mark.parametrize(
    "payload",
    [
        {"valid_for_t2": 1},
        {"config": {"model_contract": {"valid_for_t2": "false"}}},
        {
            "nested": [
                {
                    "attention": {
                        "attention_is_diagnostic_not_T2": False,
                    }
                }
            ]
        },
    ],
)
def test_nested_t2_scope_and_attention_semantics_fail_closed(payload) -> None:
    with pytest.raises(ValueError, match="T2|attention"):
        audit._reject_t2_localization_or_joint(payload, label="payload")


def test_official_golden_contract_is_bound_to_analyzer_constants(
    tmp_path: Path,
) -> None:
    golden, assets = _official_golden_fixture(tmp_path)
    result = audit._validate_official_golden_contract(
        golden_record=golden,
        golden_assets=assets,
    )
    assert result["status"] == "passed"
    assert result["cases_validated"] == len(audit.FROZEN_GOLDEN_CASES)
    assert result["runs_per_case"] == 2
    assert result["website_derivatives_are_diagnostic_mismatches"] is True


@pytest.mark.parametrize(
    ("target", "path", "replacement"),
    [
        ("assets", (0, "relative_path"), "other/224.png"),
        ("assets", (0, "sha256"), "0" * 64),
        ("golden", ("cases", 0, "decoded_rgb_sha256"), "0" * 64),
        ("golden", ("cases", 0, "tensor_sha256"), "0" * 64),
        ("golden", ("cases", 0, "native_size"), [1, 2]),
        ("golden", ("cases", 0, "patch_count"), 21),
        ("golden", ("cases", 0, "raw_logit"), 9.0),
        ("golden", ("cases", 0, "probability"), 0.1),
        (
            "golden",
            ("cases", 0, "preprocess", "decoded_rgb_sha256"),
            "0" * 64,
        ),
        (
            "golden",
            ("cases", 0, "preprocess", "tensor_sha256"),
            "0" * 64,
        ),
        (
            "golden",
            ("cases", 0, "preprocess", "geometry", "effective_patch_count"),
            999,
        ),
        (
            "golden",
            ("cases", 0, "observed_runs", 1, "raw_logit"),
            9.0,
        ),
        (
            "golden",
            ("cases", 0, "artifact_hashes", 1, "attention"),
            "a" * 64,
        ),
        ("golden", ("cases", 0, "bit_identical_repeats"), False),
        ("golden", ("cases", 0, "logit_absolute_difference"), 1.0),
        ("golden", ("cases", 0, "passed"), False),
        (
            "golden",
            (
                "cases",
                0,
                "website_derivative_display_matches_released_regression",
            ),
            True,
        ),
        ("golden", ("status",), "failed"),
        ("golden", ("runs_per_case",), 1),
        ("golden", ("logit_absolute_tolerance",), 1.0),
        (
            "golden",
            ("probability_absolute_tolerance",),
            1.0,
        ),
        (
            "golden",
            ("website_display_reference",),
            "website scores are executable goldens",
        ),
    ],
)
def test_official_golden_contract_rejects_tampering(
    tmp_path: Path,
    target: str,
    path: tuple,
    replacement,
) -> None:
    golden, assets = _official_golden_fixture(tmp_path)
    _set_nested(
        golden if target == "golden" else assets,
        path,
        replacement,
    )
    with pytest.raises(ValueError):
        audit._validate_official_golden_contract(
            golden_record=golden,
            golden_assets=assets,
        )


def test_strict_probability_boundary_and_aliases() -> None:
    row = {"id": "threshold", **_score_fields(0.0)}
    result = audit._audit_score_fields(
        row,
        replay_raw_logit=0.0,
        replay_probability=0.5,
        raw_tolerance=0.0,
        probability_tolerance=0.0,
    )
    assert result["decision"] is False

    tampered = copy.deepcopy(row)
    tampered["classification"]["score"] = 0.500001
    with pytest.raises(ValueError, match="classification.score"):
        audit._audit_score_fields(
            tampered,
            replay_raw_logit=0.0,
            replay_probability=0.5,
            raw_tolerance=0.0,
            probability_tolerance=0.0,
        )

    crossed = {"id": "crossed", **_score_fields(0.0000002)}
    with pytest.raises(ValueError, match="recorded/replay threshold decision"):
        audit._audit_score_fields(
            crossed,
            replay_raw_logit=crossed["raw_logit"],
            replay_probability=0.49999999,
            raw_tolerance=0.0,
            probability_tolerance=1e-6,
        )


def test_physical_history_uses_last_retry() -> None:
    history = audit.summarize_result_history(
        [
            {"id": "a", "status": "error"},
            {"id": "b", "status": "ok"},
            {"id": "a", "status": "ok"},
        ]
    )
    assert history["physical_rows"] == 3
    assert history["unique_ids"] == 2
    assert history["duplicate_rows"] == 1
    assert history["recovered_ids"] == ["a"]
    assert history["latest_status_counts"] == {"ok": 2}


@pytest.mark.parametrize(
    "value",
    ["../escape", "/absolute", "nested/run", r"nested\\run", ".", ".."],
)
def test_run_id_path_traversal_is_rejected(tmp_path: Path, value: str) -> None:
    with pytest.raises(ValueError, match="safe non-empty path component"):
        audit._load_run_files(
            repo_root=tmp_path,
            results_dir=Path("results"),
            run_id=value,
        )


def test_checkpoint_schema_rejects_nonfinite_tensor() -> None:
    state = OrderedDict(
        [
            ("weight", torch.arange(6, dtype=torch.float32).reshape(2, 3)),
            ("counter", torch.tensor(7, dtype=torch.int64)),
        ]
    )
    schema = audit._checkpoint_schema(state, torch_module=torch)
    assert schema["tensor_count"] == 2
    assert schema["state_elements"] == 7

    bad = OrderedDict(state)
    bad["bad"] = torch.tensor(float("nan"))
    with pytest.raises(ValueError, match="not finite"):
        audit._checkpoint_schema(bad, torch_module=torch)


def test_artifact_loader_enforces_hash_shape_dtype_and_run_boundary(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "results" / "run"
    row = {
        "id": "x",
        **_artifact_fields(
            root=tmp_path,
            run_dir=run_dir,
            sample_id="x",
            patch=np.zeros((4, 1096), dtype=np.float32),
            feature=np.zeros(1096, dtype=np.float32),
            attention=np.full((12, 4), 0.25, dtype=np.float32),
        ),
    }
    patch, feature, attention, paths = audit._load_artifacts(
        row,
        patch_count=4,
        repo_root=tmp_path,
        run_dir=run_dir,
    )
    assert patch.shape == (4, 1096)
    assert feature.shape == (1096,)
    assert attention.shape == (12, 4)
    assert all(path.is_file() for path in paths.values())

    wrong_hash = copy.deepcopy(row)
    wrong_hash["spai_attention_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        audit._load_artifacts(
            wrong_hash,
            patch_count=4,
            repo_root=tmp_path,
            run_dir=run_dir,
        )

    outside = tmp_path / "outside.npy"
    file_digest, array_digest = _save_array(
        outside,
        np.zeros(1096, dtype=np.float32),
    )
    outside_row = copy.deepcopy(row)
    outside_row["spai_feature_path"] = str(outside)
    outside_row["spai_feature_sha256"] = file_digest
    outside_row["spai_feature_array_sha256"] = array_digest
    outside_row["feature_array_sha256"] = array_digest
    with pytest.raises(ValueError, match="outside its run directory"):
        audit._load_artifacts(
            outside_row,
            patch_count=4,
            repo_root=tmp_path,
            run_dir=run_dir,
        )


class _TinyMFViT(torch.nn.Module):
    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        mean = patches.mean(dim=(1, 2, 3), keepdim=False)[:, None]
        basis = torch.linspace(
            0.5,
            1.5,
            audit.FEATURE_DIMENSION,
            dtype=patches.dtype,
            device=patches.device,
        )[None, :]
        return mean * basis


class _TinySPAI(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mfvit = _TinyMFViT()
        self.norm = torch.nn.LayerNorm(audit.FEATURE_DIMENSION)
        self.cls_head = torch.nn.Sequential(
            torch.nn.Linear(audit.FEATURE_DIMENSION, 8),
            torch.nn.ReLU(),
            torch.nn.Linear(8, 1),
        )
        torch.manual_seed(13)
        for parameter in self.parameters():
            torch.nn.init.uniform_(parameter, -0.03, 0.03)

    def patches_attention(
        self,
        features: torch.Tensor,
        return_attn: bool = False,
    ):
        batch, patches, _ = features.shape
        attention = torch.full(
            (batch, audit.ATTENTION_HEADS, 1, patches),
            1.0 / patches,
            dtype=features.dtype,
            device=features.device,
        )
        attended = features.mean(dim=1)
        return (attended, attention) if return_attn else attended


def test_forward_and_artifact_replays_cover_all_stages() -> None:
    model = _TinySPAI().eval()
    tensor = torch.linspace(
        0.0,
        1.0,
        3 * 448 * 448,
        dtype=torch.float32,
    ).reshape(3, 448, 448)
    with torch.inference_mode():
        fresh = audit._forward_with_evidence(
            model,
            tensor,
            torch_module=torch,
            feature_extraction_batch=2,
        )
        replay = audit._replay_patch_artifact(
            model,
            fresh.patch_features.clone(),
        )
        feature_logit = model.cls_head(
            fresh.normalized_feature.unsqueeze(0)
        ).reshape(())
    assert fresh.patch_features.shape == (4, 1096)
    assert fresh.normalized_feature.shape == (1096,)
    assert fresh.attention.shape == (12, 4)
    assert torch.equal(replay.attention, fresh.attention)
    assert torch.equal(replay.normalized_feature, fresh.normalized_feature)
    assert torch.equal(replay.raw_logit, fresh.raw_logit)
    assert torch.equal(feature_logit, fresh.raw_logit)


def test_full_artifact_audit_redecodes_and_reruns_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "results" / "run"
    run_dir.mkdir(parents=True)
    height = width = 448
    rgb = np.arange(height * width * 3, dtype=np.uint8).reshape(
        height,
        width,
        3,
    )
    image_path = tmp_path / "forged.png"
    _write_rgb(image_path, rgb)
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[10:20, 30:40] = 255
    mask_path = tmp_path / "mask.png"
    _write_mask(mask_path, mask)
    canonical = {
        "sample_id": "forged",
        "task_id": "task",
        "pair_rank": 0,
        "rank": 0,
        "kind": "forged",
        "label": 1,
        "domain": "test",
        "width": width,
        "height": height,
        "canonical_path": str(image_path),
        "canonical_sha256": sha256_file(image_path),
        "gt_mask_kind": "exact_diff",
        "gt_mask_path": str(mask_path),
        "gt_mask_sha256": sha256_file(mask_path),
        "gt_positive_pixels": 100,
    }
    prepared = audit.preprocess_image(image_path, torch_module=torch)
    model = _TinySPAI().eval()
    with torch.inference_mode():
        evidence = audit._forward_with_evidence(
            model,
            prepared.tensor,
            torch_module=torch,
        )
    patch = evidence.patch_features.numpy().astype(np.float32)
    feature = evidence.normalized_feature.numpy().astype(np.float32)
    attention = evidence.attention.numpy().astype(np.float32)
    raw = float(evidence.raw_logit)
    visibility = audit._visibility_from_exact_gt(
        canonical,
        repo_root=tmp_path,
    )
    row = {
        "id": "forged",
        "sample_id": "forged",
        "task_id": "task",
        "pair_rank": 0,
        "rank": 0,
        "kind": "forged",
        "label": 1,
        "domain": "test",
        "model": "SPAI",
        "model_slug": "spai",
        "preprocess_profile": audit.PREPROCESS_PROFILE,
        "config_fingerprint": "cfg",
        "input_path": str(image_path),
        "input_sha256": sha256_file(image_path),
        "input_width": width,
        "input_height": height,
        "status": "ok",
        "valid_for_metrics": True,
        "valid_for_t2": False,
        "attention_is_diagnostic_not_T2": True,
        "preprocess": prepared.audit,
        "edit_visibility": "full",
        "edit_visible_gt_fraction": 1.0,
        "edit_visibility_evidence": visibility,
        "latency_ms": 1.0,
        "peak_cuda_memory_bytes": None,
        **_score_fields(raw),
        **_artifact_fields(
            root=tmp_path,
            run_dir=run_dir,
            sample_id="forged",
            patch=patch,
            feature=feature,
            attention=attention,
        ),
    }
    files = audit.RunFiles(
        run_dir=run_dir,
        results_path=run_dir / "results.jsonl",
        expected_path=run_dir / "expected_inputs.jsonl",
        summary_path=run_dir / "summary.json",
        manifest_path=run_dir / "manifest.json",
        rows=[row],
        expected=[canonical],
        summary={},
        manifest={
            "config_fingerprint": "cfg",
            "config": {"config_fingerprint": "cfg"},
        },
    )
    monkeypatch.setattr(
        audit,
        "_load_runner_pins",
        lambda: SimpleNamespace(MODEL_NAME="SPAI", MODEL_SLUG="spai"),
    )
    runtime = audit.ReplayRuntime(
        torch=torch,
        device=torch.device("cpu"),
        evidence={"device": "cpu"},
    )
    result = audit.audit_artifacts(
        repo_root=tmp_path,
        source_root=tmp_path,
        all_inputs=[canonical],
        files=files,
        runtime=runtime,
        model=model,
        patch_feature_tolerance=0.0,
        feature_tolerance=0.0,
        attention_tolerance=0.0,
        raw_tolerance=0.0,
        probability_tolerance=0.0,
    )
    assert result["images_audited"] == 1
    assert result["fresh_complete_fft_vit_srs_sca_mlp_forwards"] == 1
    assert result["maximum_absolute_differences"] == {
        "patch_feature": 0.0,
        "feature": 0.0,
        "attention": 0.0,
        "raw_logit": 0.0,
        "probability": 0.0,
        "artifact_sca_feature": 0.0,
        "artifact_sca_attention": 0.0,
        "artifact_sca_logit": 0.0,
        "artifact_mlp_logit": 0.0,
    }

    row["spai_patch_features_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        audit.audit_artifacts(
            repo_root=tmp_path,
            source_root=tmp_path,
            all_inputs=[canonical],
            files=files,
            runtime=runtime,
            model=model,
        )


def test_summary_is_recomputed_and_tamper_rejected() -> None:
    expected = [
        {
            "sample_id": "real",
            "task_id": "task",
            "kind": "real",
            "label": 0,
            "domain": "test",
        },
        {
            "sample_id": "forged",
            "task_id": "task",
            "kind": "forged",
            "label": 1,
            "domain": "test",
        },
    ]
    rows = [
        {
            "id": item["sample_id"],
            "task_id": "task",
            "kind": item["kind"],
            "label": item["label"],
            "domain": "test",
            "status": "ok",
            "ai_score": 0.2 if item["kind"] == "real" else 0.8,
            "edit_visibility": "full",
            "edit_visible_gt_fraction": 1.0,
            "latency_ms": 1.0,
            "peak_cuda_memory_bytes": None,
        }
        for item in expected
    ]
    manifest = {
        "config": {
            "metrics": {
                "bootstrap_samples": 5,
                "bootstrap_seed": 17,
                "fixed_threshold": 0.5,
                "threshold_operator": ">",
            }
        }
    }
    recorded = audit.summarize_spai_results(
        rows,
        expected,
        bootstrap_samples=5,
        seed=17,
    )
    result = audit.recompute_summary(
        result_rows=rows,
        expected_rows=expected,
        manifest=manifest,
        recorded_summary=recorded,
        independent_result_rows=copy.deepcopy(rows),
    )
    assert result["recorded_summary_recomputed"] is True
    assert (
        result["independent_full_model_summary_within_probability_tolerance"]
        is True
    )
    assert (
        result["independent_full_model_summary_float_tolerance"]
        == audit.PROBABILITY_ABSOLUTE_TOLERANCE
    )

    tampered = copy.deepcopy(recorded)
    tampered["coverage"]["valid_images"] += 1
    with pytest.raises(ValueError, match="summary coverage.valid_images"):
        audit.recompute_summary(
            result_rows=rows,
            expected_rows=expected,
            manifest=manifest,
            recorded_summary=tampered,
        )
