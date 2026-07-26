from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from torchvision import transforms

from eval.opensource import analyze_bfree_run as audit
from eval.opensource import run_bfree as runner
from eval.opensource.bfree_metrics import summarize_bfree_results
from eval.opensource.common import sha256_file, stable_json


def _write_rgb(path: Path, pixels: np.ndarray) -> None:
    Image.fromarray(pixels.astype(np.uint8), mode="RGB").save(path)


def _write_mask(path: Path, pixels: np.ndarray) -> None:
    Image.fromarray(pixels.astype(np.uint8), mode="L").save(path)


def _score_fields(crop_logits: np.ndarray) -> dict:
    crops = np.ascontiguousarray(crop_logits, dtype=np.float32)
    raw = float(torch.mean(torch.from_numpy(crops), dtype=torch.float32).item())
    decision = raw > 0.0
    probability = float(
        torch.sigmoid(torch.tensor(raw, dtype=torch.float32)).item()
    )
    return {
        "raw_logit": raw,
        "ai_score": raw,
        "score": raw,
        "crop_logits": crops.tolist(),
        "fake_probability": probability,
        "score_semantics": audit.SCORE_SEMANTICS,
        "classification_decision": decision,
        "classification_threshold": 0.0,
        "classification_threshold_operator": ">",
        "classification": {
            "raw_logit": raw,
            "ai_score": raw,
            "fake_probability": probability,
            "decision": decision,
            "threshold": 0.0,
            "threshold_operator": ">",
            "semantics": audit.SCORE_SEMANTICS,
        },
        "t1": {
            "raw_logit": raw,
            "ai_score": raw,
            "fake_probability": probability,
            "policy": audit.T1_POLICY,
            "decision": decision,
            "threshold": 0.0,
            "threshold_operator": ">",
        },
        "manual_replay": {
            "raw_logit": raw,
            "ai_score": raw,
            "crop_logits": crops.tolist(),
        },
    }


def _save_artifact(
    *,
    repo_root: Path,
    run_dir: Path,
    sample_id: str,
    features: np.ndarray,
    crop_logits: np.ndarray,
) -> dict:
    path = run_dir / "artifacts" / f"{sample_id}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    features_array = np.ascontiguousarray(features, dtype=np.float32)
    logits_array = np.ascontiguousarray(crop_logits, dtype=np.float32)
    np.savez(
        path,
        features=features_array,
        crop_logits=logits_array,
    )
    return {
        "artifact_path": str(path.relative_to(repo_root)),
        "artifact_sha256": sha256_file(path),
        "feature_array_sha256": hashlib.sha256(
            features_array.tobytes(order="C")
        ).hexdigest(),
        "crop_logits_array_sha256": hashlib.sha256(
            logits_array.tobytes(order="C")
        ).hexdigest(),
    }


def test_frozen_constants_and_checkpoint_schema_are_exact() -> None:
    assert audit.FROZEN_SOURCE_COMMIT == (
        "c6a9f898782fb466b29af01f21960b67415afb0e"
    )
    assert audit.FROZEN_ZIP_SHA256 == (
        "8230fd3f0f3a64a6403acb692ce1663718ed16f36a5a4de4a68c0d273781769f"
    )
    assert audit.FROZEN_CHECKPOINT_SHA256 == (
        "5948ca78f4d94e820c250d24cdf155035b4a85960443800bfe6bb7f06bffe947"
    )
    assert audit.FROZEN_CHECKPOINT_TENSORS == 177
    assert audit.FROZEN_CHECKPOINT_ELEMENTS == 86_526_721
    assert audit.FROZEN_CHECKPOINT_SCHEMA_SHA256 == (
        "e4bb9ddd115309740a70235152b7376e2c8299bb90baf243809f2a5e1665f524"
    )
    assert audit.GOLDEN_ABSOLUTE_TOLERANCE == 5e-5
    assert [case["official_raw_logit"] for case in audit.FROZEN_GOLDEN_CASES] == [
        -5.9374785,
        -4.441922,
        4.430519,
        3.8499813,
    ]


def test_frozen_runner_pins_match_independent_constants() -> None:
    pins = audit._load_runner_pins()
    assert pins.MODEL_SOURCE_COMMIT == audit.FROZEN_SOURCE_COMMIT
    assert pins.SOURCE_FILES == audit.FROZEN_SOURCE_FILES
    assert pins.FEATURE_DIMENSION == 768
    assert pins.CROPS == 5
    assert pins.PATCH_SIZE == 14
    assert pins.CROP_SIZE == 504
    assert pins.CANONICAL_RELEASE == audit.FROZEN_CANONICAL_RELEASE
    assert pins.VISIBILITY_CENSUS == audit.FROZEN_VISIBILITY_CENSUS


@pytest.mark.parametrize(
    ("width", "height", "expected_grid", "expected_wrap"),
    [
        (835, 1256, [89, 59], False),
        (1258, 833, [59, 89], False),
        (1024, 1024, [73, 73], False),
        (420, 700, [50, 30], True),
        (700, 420, [30, 50], True),
        (63, 91, [6, 4], True),
    ],
)
def test_geometry_reconstructs_official_token_crops(
    width: int,
    height: int,
    expected_grid: list[int],
    expected_wrap: bool,
) -> None:
    geometry = audit.compute_preprocess_geometry(width, height)
    assert geometry["patch_grid_wh"] == list(reversed(expected_grid))
    assert geometry["replicate_wrap_applied"] is expected_wrap
    assert geometry["crop_order"] == [
        "center",
        "top_left",
        "bottom_left",
        "bottom_right",
        "top_right",
    ]
    assert len(geometry["crop_starts_patch_xy"]) == 5
    if expected_wrap:
        assert geometry["post_wrap_patch_grid_wh"] == [36, 36]
        assert geometry["crop_starts_patch_xy"] == [[0, 0]] * 5
        assert geometry["used_native_rectangles_xyxy"] == [
            [
                0,
                0,
                min(expected_grid[1], 36) * 14,
                min(expected_grid[0], 36) * 14,
            ]
        ] * 5
    else:
        grid_h, grid_w = expected_grid
        assert geometry["crop_starts_patch_xy"] == [
            [(grid_w - 36) // 2, (grid_h - 36) // 2],
            [0, 0],
            [0, grid_h - 36],
            [grid_w - 36, grid_h - 36],
            [grid_w - 36, 0],
        ]


@pytest.mark.parametrize(
    ("width", "height"),
    [(63, 91), (420, 700), (700, 420), (835, 1256), (1024, 1024)],
)
def test_independent_preprocess_is_torchvision_exact(
    tmp_path: Path,
    width: int,
    height: int,
) -> None:
    yy, xx = np.mgrid[:height, :width]
    pixels = np.stack(
        (
            (xx * 17 + yy * 3) % 256,
            (xx * 5 + yy * 19) % 256,
            (xx * 13 + yy * 7) % 256,
        ),
        axis=-1,
    ).astype(np.uint8)
    path = tmp_path / f"rgb_{width}_{height}.png"
    _write_rgb(path, pixels)
    prepared = audit.preprocess_image(path, torch_module=torch)
    runner_array, runner_audit = runner.preprocess_image(path)
    expected = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )(Image.open(path).convert("RGB"))
    assert torch.equal(prepared.tensor, expected)
    assert np.array_equal(prepared.tensor.numpy(), runner_array)
    assert prepared.audit == runner_audit
    assert prepared.tensor.shape == (3, height, width)
    assert prepared.tensor.dtype == torch.float32
    assert prepared.audit["tensor_sha256"] == audit._tensor_sha256(expected)
    assert prepared.audit["geometry"] == audit.compute_preprocess_geometry(
        width, height
    )


def test_official_demo_decode_and_preprocess_hashes_are_frozen() -> None:
    root = audit.DEFAULT_SOURCE_ROOT / "code" / "demo_images"
    for case in audit.FROZEN_GOLDEN_CASES:
        path = root / case["filename"]
        assert sha256_file(path) == case["sha256"]
        prepared = audit.preprocess_image(path, torch_module=torch)
        assert (
            prepared.audit["decoded_rgb_sha256"]
            == case["decoded_rgb_sha256"]
        )
        assert prepared.audit["tensor_sha256"] == case["tensor_sha256"]
        assert (
            prepared.audit["geometry"]["patch_grid_wh"]
            == list(reversed(case["patch_grid_hw"]))
        )


def test_golden_audit_accepts_runner_actual_raw_logit_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    by_tensor_hash = {
        case["tensor_sha256"]: float(case["official_raw_logit"])
        for case in audit.FROZEN_GOLDEN_CASES
    }

    def fake_forward(
        _model: object,
        tensor: torch.Tensor,
        _runtime: audit.ReplayRuntime,
    ) -> audit.ForwardEvidence:
        digest = audit._tensor_sha256(tensor)
        raw = by_tensor_hash[digest]
        features = np.full((5, 768), raw, dtype=np.float32)
        logits = np.full(5, raw, dtype=np.float32)
        return audit.ForwardEvidence(features, logits, raw, raw)

    monkeypatch.setattr(audit, "_forward_with_evidence", fake_forward)
    runtime = audit.ReplayRuntime(
        torch=torch,
        device=torch.device("cpu"),
        evidence={"device": "cpu"},
    )
    recorded = {
        "status": "passed",
        "cases": [
            {
                "filename": case["filename"],
                "actual_raw_logit": case["official_raw_logit"],
            }
            for case in audit.FROZEN_GOLDEN_CASES
        ],
    }
    result = audit._audit_official_golden(
        source_root=audit.DEFAULT_SOURCE_ROOT,
        model=object(),
        runtime=runtime,
        recorded=recorded,
    )
    assert result["status"] == "passed"
    assert len(result["cases"]) == 4

    tampered = copy.deepcopy(recorded)
    tampered["cases"][0]["actual_raw_logit"] += 0.01
    with pytest.raises(ValueError, match="recorded golden"):
        audit._audit_official_golden(
            source_root=audit.DEFAULT_SOURCE_ROOT,
            model=object(),
            runtime=runtime,
            recorded=tampered,
        )


def test_wrap_visibility_uses_source_tokens_not_repeated_coordinates() -> None:
    height, width = 700, 420
    geometry = audit.compute_preprocess_geometry(width, height)
    assert geometry["patch_grid_wh"] == [30, 50]
    assert geometry["replicate_wrap_applied"] is True
    # A short width triggers wrap and the official implementation truncates
    # the otherwise-long height to its first 36 source-token rows.
    assert geometry["used_native_rectangles_xyxy"] == [
        [0, 0, 420, 504]
    ] * 5
    mask = np.zeros((height, width), dtype=bool)
    mask[520:560, 100:180] = True
    invisible = audit._visibility_from_exact_gt(mask, geometry)
    assert invisible["edit_visibility"] == "none"
    assert invisible["edit_visible_gt_fraction"] == 0.0
    mask.fill(False)
    mask[480:540, 100:180] = True
    partial = audit._visibility_from_exact_gt(mask, geometry)
    assert partial["edit_visibility"] == "partial"
    assert partial["edit_visible_gt_fraction"] == 24 / 60


def test_nonwrap_visibility_uses_union_of_five_native_rectangles() -> None:
    geometry = audit.compute_preprocess_geometry(1024, 1024)
    mask = np.zeros((1024, 1024), dtype=bool)
    # Five crops leave some top-centre tokens unseen.
    mask[20:40, 506:516] = True
    result = audit._visibility_from_exact_gt(mask, geometry)
    assert result["edit_visibility"] == "none"
    mask.fill(False)
    mask[20:40, 20:40] = True
    result = audit._visibility_from_exact_gt(mask, geometry)
    assert result["edit_visibility"] == "full"


def test_pair_visibility_reopens_exact_mask(tmp_path: Path) -> None:
    height, width = 700, 420
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[480:540, 100:180] = 255
    mask_path = tmp_path / "mask.png"
    _write_mask(mask_path, mask)
    common = {
        "task_id": "task",
        "domain": "synthetic",
        "width": width,
        "height": height,
    }
    rows = [
        {
            **common,
            "sample_id": "real",
            "kind": "real",
            "label": 0,
            "gt_mask_kind": "all_zero",
            "gt_mask_path": None,
            "gt_mask_sha256": None,
            "gt_positive_pixels": 0,
        },
        {
            **common,
            "sample_id": "forged",
            "kind": "forged",
            "label": 1,
            "gt_mask_kind": "exact_diff",
            "gt_mask_path": str(mask_path),
            "gt_mask_sha256": sha256_file(mask_path),
            "gt_positive_pixels": 60 * 80,
        },
    ]
    result = audit._pair_visibility(rows, repo_root=tmp_path)["task"]
    runner_result = runner.build_pair_visibility(rows, tmp_path)["task"]
    assert result["edit_visibility"] == "partial"
    assert result["edit_visible_gt_fraction"] == 24 / 60
    assert (
        result["edit_visibility_evidence"]
        == runner_result["edit_visibility_evidence"]
    )


def test_score_contract_is_strict_at_zero_and_rejects_alias_tamper() -> None:
    crops = np.asarray([-2, -1, 0, 1, 2], dtype=np.float32)
    row = _score_fields(crops)
    replay = audit._audit_score_fields(
        row,
        replay_crop_logits=crops,
        replay_raw_logit=0.0,
        tolerance=0.0,
    )
    assert replay["decision"] is False

    tampered = copy.deepcopy(row)
    tampered["classification"]["ai_score"] = 0.01
    with pytest.raises(ValueError, match="classification.ai_score"):
        audit._audit_score_fields(
            tampered,
            replay_crop_logits=crops,
            replay_raw_logit=0.0,
            tolerance=0.0,
        )
    wrong_operator = copy.deepcopy(row)
    wrong_operator["classification_threshold_operator"] = ">="
    with pytest.raises(ValueError, match="classification operator"):
        audit._audit_score_fields(
            wrong_operator,
            replay_crop_logits=crops,
            replay_raw_logit=0.0,
            tolerance=0.0,
        )


@pytest.mark.parametrize(
    "payload",
    [
        {"t2": {}},
        {"localization": None},
        {"nested": {"score_map_path": "invented.npy"}},
        {"attention_map": []},
        {"metrics": [{"s_joint": 0.2}]},
        {"pixel_metrics": {}},
    ],
)
def test_t2_localization_and_joint_fields_are_rejected(payload: dict) -> None:
    with pytest.raises(ValueError, match="unsupported"):
        audit._reject_t2_localization_or_joint(payload, label="payload")
    audit._reject_t2_localization_or_joint(
        {"task_scope": {"valid_for_t2": False}, "t2": False},
        label="negative declaration",
    )


@pytest.mark.parametrize(
    "run_id",
    ["..", ".", "../outside", "nested/run", r"nested\\run", "/absolute"],
)
def test_run_loader_rejects_path_traversal(
    tmp_path: Path,
    run_id: str,
) -> None:
    with pytest.raises(ValueError, match="safe non-empty path component"):
        audit._load_run_files(results_dir=tmp_path, run_id=run_id)


def test_npz_loader_separates_file_and_array_hashes(tmp_path: Path) -> None:
    run_dir = tmp_path / "results" / "run"
    features = np.arange(5 * 768, dtype=np.float32).reshape(5, 768)
    logits = np.arange(5, dtype=np.float32)
    row = _save_artifact(
        repo_root=tmp_path,
        run_dir=run_dir,
        sample_id="x",
        features=features,
        crop_logits=logits,
    )
    loaded_features, loaded_logits, path = audit._load_artifact(
        row,
        repo_root=tmp_path,
        run_dir=run_dir,
    )
    assert np.array_equal(loaded_features, features)
    assert np.array_equal(loaded_logits, logits)
    assert sha256_file(path) == row["artifact_sha256"]
    assert row["artifact_sha256"] != row["feature_array_sha256"]

    swapped = dict(row, artifact_sha256=row["feature_array_sha256"])
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        audit._load_artifact(
            swapped,
            repo_root=tmp_path,
            run_dir=run_dir,
        )

    outside = tmp_path / "outside.npz"
    np.savez(outside, features=features, crop_logits=logits)
    outside_row = {
        **row,
        "artifact_path": str(outside),
        "artifact_sha256": sha256_file(outside),
    }
    with pytest.raises(ValueError, match="outside its run directory"):
        audit._load_artifact(
            outside_row,
            repo_root=tmp_path,
            run_dir=run_dir,
        )


def test_npz_loader_rejects_extra_or_object_arrays(tmp_path: Path) -> None:
    extra = tmp_path / "extra.npz"
    np.savez(
        extra,
        features=np.zeros((5, 768), dtype=np.float32),
        crop_logits=np.zeros(5, dtype=np.float32),
        invented=np.zeros(1),
    )
    with pytest.raises(ValueError, match="keys mismatch"):
        audit._safe_npz(
            extra,
            expected_keys={"features", "crop_logits"},
        )
    object_path = tmp_path / "object.npz"
    np.savez(
        object_path,
        features=np.asarray([{"bad": True}], dtype=object),
        crop_logits=np.zeros(5, dtype=np.float32),
    )
    with pytest.raises(ValueError, match="unsafe or malformed|object"):
        audit._safe_npz(
            object_path,
            expected_keys={"features", "crop_logits"},
        )


def test_checkpoint_schema_preserves_state_dict_insertion_order() -> None:
    state = {
        "z": torch.ones((2, 3), dtype=torch.float32),
        "a": torch.zeros((4,), dtype=torch.float32),
    }
    result = audit._checkpoint_schema(state, torch)
    expected = [
        {"k": "z", "shape": [2, 3], "dtype": "torch.float32", "n": 6},
        {"k": "a", "shape": [4], "dtype": "torch.float32", "n": 4},
    ]
    assert result["items"] == expected
    assert result["items_sha256"] == hashlib.sha256(
        stable_json(expected).encode("utf-8")
    ).hexdigest()
    assert result["state_elements"] == 10


class _TinyBody(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.head = torch.nn.Linear(768, 1)
        with torch.no_grad():
            values = torch.linspace(-0.001, 0.001, 768)
            self.head.weight.copy_(values.unsqueeze(0))
            self.head.bias.fill_(0.125)


class _TinyBFree(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = _TinyBody()

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        base = image.mean().to(torch.float32)
        feature = torch.arange(
            5 * 768,
            dtype=torch.float32,
            device=image.device,
        ).reshape(5, 768)
        feature = feature.mul_(1e-4).add_(base)
        crop_logits = self.model.head(feature)
        return crop_logits.mean(dim=0, keepdim=True)


def _artifact_fixture(
    tmp_path: Path,
) -> tuple[
    audit.RunFiles,
    list[dict],
    audit.ReplayRuntime,
    _TinyBFree,
]:
    run_dir = tmp_path / "results" / "run"
    run_dir.mkdir(parents=True)
    height, width = 63, 91
    yy, xx = np.mgrid[:height, :width]
    real_pixels = np.stack(
        (
            (xx * 9 + yy * 4) % 256,
            (xx * 3 + yy * 11) % 256,
            (xx * 7 + yy * 5) % 256,
        ),
        axis=-1,
    ).astype(np.uint8)
    forged_pixels = real_pixels.copy()
    forged_pixels[8:16, 15:25, 1] ^= np.uint8(63)
    real_path = tmp_path / "real.png"
    forged_path = tmp_path / "forged.png"
    _write_rgb(real_path, real_pixels)
    _write_rgb(forged_path, forged_pixels)
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[8:16, 15:25] = 255
    mask_path = tmp_path / "mask.png"
    _write_mask(mask_path, mask)
    common = {
        "task_id": "task",
        "pair_rank": 0,
        "domain": "synthetic",
        "candidate": "mouse",
        "dataset_id": "synthetic",
        "width": width,
        "height": height,
        "edit_region_xyxy": [15, 8, 25, 16],
    }
    real = {
        **common,
        "sample_id": "real",
        "rank": 0,
        "kind": "real",
        "label": 0,
        "canonical_path": str(real_path),
        "canonical_sha256": sha256_file(real_path),
        "gt_mask_kind": "all_zero",
        "gt_mask_path": None,
        "gt_mask_sha256": None,
        "gt_positive_pixels": 0,
    }
    forged = {
        **common,
        "sample_id": "forged",
        "rank": 1,
        "kind": "forged",
        "label": 1,
        "canonical_path": str(forged_path),
        "canonical_sha256": sha256_file(forged_path),
        "gt_mask_kind": "exact_diff",
        "gt_mask_path": str(mask_path),
        "gt_mask_sha256": sha256_file(mask_path),
        "gt_positive_pixels": 80,
    }
    expected = [real, forged]
    visibility = audit._pair_visibility(
        expected,
        repo_root=tmp_path,
    )["task"]
    model = _TinyBFree().eval()
    runtime = audit.ReplayRuntime(
        torch=torch,
        device=torch.device("cpu"),
        evidence={"device": "cpu"},
    )
    rows: list[dict] = []
    for canonical in expected:
        prepared = audit.preprocess_image(
            Path(canonical["canonical_path"]),
            torch_module=torch,
        )
        forward = audit._forward_with_evidence(
            model,
            prepared.tensor,
            runtime,
        )
        row = {
            "id": canonical["sample_id"],
            "sample_id": canonical["sample_id"],
            "task_id": canonical["task_id"],
            "kind": canonical["kind"],
            "label": canonical["label"],
            "domain": canonical["domain"],
            "status": "ok",
            "valid_for_metrics": True,
            "preprocess_profile": audit.FROZEN_PROFILE,
            "preprocess": prepared.audit,
            "edit_visibility": visibility["edit_visibility"],
            "edit_visible_gt_fraction": visibility[
                "edit_visible_gt_fraction"
            ],
            "edit_visibility_evidence": visibility[
                "edit_visibility_evidence"
            ],
            "preprocess_latency_ms": 1.0,
            "latency_ms": 2.0,
            "peak_cuda_memory_bytes": None,
            **_save_artifact(
                repo_root=tmp_path,
                run_dir=run_dir,
                sample_id=canonical["sample_id"],
                features=forward.features,
                crop_logits=forward.crop_logits,
            ),
            **_score_fields(forward.crop_logits),
        }
        rows.append(row)
    paths = {
        name: run_dir / name
        for name in (
            "results.jsonl",
            "expected_inputs.jsonl",
            "summary.json",
            "run_manifest.json",
        )
    }
    for path in paths.values():
        path.write_text("{}\n", encoding="utf-8")
    files = audit.RunFiles(
        run_dir=run_dir,
        results_path=paths["results.jsonl"],
        expected_path=paths["expected_inputs.jsonl"],
        summary_path=paths["summary.json"],
        manifest_path=paths["run_manifest.json"],
        rows=rows,
        expected=expected,
        summary={},
        manifest={},
    )
    return files, expected, runtime, model


def _manifest_identity_fixture(
    tmp_path: Path,
) -> tuple[audit.RunFiles, Path, Path, Path]:
    files, _, _, _ = _artifact_fixture(tmp_path)
    source_root = tmp_path / "source"
    weights_dir = tmp_path / "weights"
    weights_zip = tmp_path / "weights.zip"
    source_files = {
        relative: {
            "path": str((source_root / relative).resolve()),
            "sha256": digest,
        }
        for relative, digest in audit.FROZEN_SOURCE_FILES.items()
    }
    visibility = {
        "pairs": 275,
        "census": dict(audit.FROZEN_VISIBILITY_CENSUS["edit_visibility"]),
        "mean_edit_visible_gt_fraction": audit.FROZEN_VISIBILITY_CENSUS[
            "mean_edit_visible_gt_fraction"
        ],
        "wrapped_pairs": 26,
        "wrapped_visibility_census": dict(
            audit.FROZEN_VISIBILITY_CENSUS["wrap_edit_visibility"]
        ),
        "distinct_crop_starts_census": dict(
            audit.FROZEN_VISIBILITY_CENSUS["distinct_crop_starts"]
        ),
        "domain_census": copy.deepcopy(
            audit.FROZEN_VISIBILITY_CENSUS["by_domain"]
        ),
    }
    golden = {"status": "passed", "cases": []}
    runtime = {"device": "cpu", "cudnn_enabled": True}
    model_audit = {"strict": True}
    config = {
        "model": "B-Free",
        "model_slug": "bfree_dino2reg4",
        "model_arch": audit.FROZEN_TIMM_ARCH,
        "source_commit": audit.FROZEN_SOURCE_COMMIT,
        "source_files": dict(audit.FROZEN_SOURCE_FILES),
        "adapter_contract": {},
        "official_zip_sha256": audit.FROZEN_ZIP_SHA256,
        "config_sha256": audit.FROZEN_CONFIG_SHA256,
        "checkpoint_sha256": audit.FROZEN_CHECKPOINT_SHA256,
        "checkpoint_schema_sha256": (
            audit.FROZEN_CHECKPOINT_SCHEMA_SHA256
        ),
        "preprocess_profile": audit.FROZEN_PROFILE,
        "model_contract": {
            "official_wrapper": True,
            "strict_full_checkpoint_load": True,
            "feature_shape": [5, 768],
            "feature_semantics": audit.FEATURE_SEMANTICS,
            "crop_logits_shape": [5],
            "primary_score": audit.SCORE_SEMANTICS,
            "score_direction": "higher_means_fake",
            "threshold": 0.0,
            "threshold_operator": ">",
            "valid_for_t1": True,
            "valid_for_t2": False,
        },
        "official_golden": golden,
        "runtime_evidence": runtime,
        "model_audit": model_audit,
        "frozen_full_dataset_visibility": visibility,
        "checkpoint_and_protocol_frozen_before_mouse_scores": True,
    }
    fingerprint = audit._fingerprint(config)
    for row in files.rows:
        row["config_fingerprint"] = fingerprint
    files.summary.update(
        {"run_id": "run", "config_fingerprint": fingerprint}
    )
    artifact_dir = files.run_dir / "artifacts"
    manifest = {
        "schema_version": "bfree_detection_run_manifest_v1",
        "run_id": "run",
        "status": "complete",
        "completed_at": "2026-07-25T00:00:00Z",
        "config": config,
        "config_fingerprint": fingerprint,
        "source": {
            "commit": audit.FROZEN_SOURCE_COMMIT,
            "tracked_dirty": False,
            "root": str(source_root.resolve()),
            "files": source_files,
        },
        "assets": {
            "zip": {
                "path": str(weights_zip.resolve()),
                "sha256": audit.FROZEN_ZIP_SHA256,
                "verified_sha256": audit.FROZEN_ZIP_SHA256,
                "md5": audit.FROZEN_ZIP_MD5,
                "verified_md5": audit.FROZEN_ZIP_MD5,
                "bytes": audit.FROZEN_ZIP_BYTES,
                "members": dict(audit.FROZEN_ZIP_MEMBERS),
            },
            "config": {
                "path": str((weights_dir / "config.yaml").resolve()),
                "sha256": audit.FROZEN_CONFIG_SHA256,
                "bytes": audit.FROZEN_CONFIG_BYTES,
                "parsed": dict(audit.FROZEN_CONFIG),
                "parsed_actual": dict(audit.FROZEN_CONFIG),
            },
            "checkpoint": {
                "path": str(
                    (
                        weights_dir / audit.FROZEN_WEIGHTS_FILE
                    ).resolve()
                ),
                "sha256": audit.FROZEN_CHECKPOINT_SHA256,
                "bytes": audit.FROZEN_CHECKPOINT_BYTES,
                "tensor_count": audit.FROZEN_CHECKPOINT_TENSORS,
                "state_elements": audit.FROZEN_CHECKPOINT_ELEMENTS,
                "schema_sha256": audit.FROZEN_CHECKPOINT_SCHEMA_SHA256,
                "top_level_keys": ["model"],
                "state_container": "collections.OrderedDict",
                "dtype": "float32",
                "safe_weights_only_load": True,
                "unsafe_globals": [],
                "schema": {
                    "top_level_keys": ["model"],
                    "state_container": "collections.OrderedDict",
                    "tensor_count": audit.FROZEN_CHECKPOINT_TENSORS,
                    "state_elements": audit.FROZEN_CHECKPOINT_ELEMENTS,
                    "all_dtype": "torch.float32",
                    "all_finite": True,
                    "schema_sha256": (
                        audit.FROZEN_CHECKPOINT_SCHEMA_SHA256
                    ),
                },
            },
        },
        "official_golden": golden,
        "runtime": runtime,
        "model_audit": model_audit,
        "task_scope": {
            "valid_for_t1": True,
            "valid_for_t2": False,
        },
        "dataset": {
            "expected_inputs_path": str(
                files.expected_path.relative_to(tmp_path)
            ),
            "expected_inputs_sha256": sha256_file(files.expected_path),
            "selected_images": len(files.expected),
        },
        "outputs": {
            "results_path": str(files.results_path.relative_to(tmp_path)),
            "results_sha256": sha256_file(files.results_path),
            "summary_path": str(files.summary_path.relative_to(tmp_path)),
            "summary_sha256": sha256_file(files.summary_path),
            "artifact_dir": str(artifact_dir.relative_to(tmp_path)),
            "artifact_files": 2,
        },
    }
    files.manifest.update(manifest)
    return files, source_root, weights_dir, weights_zip


def _resign_manifest_config(files: audit.RunFiles) -> None:
    fingerprint = audit._fingerprint(files.manifest["config"])
    files.manifest["config_fingerprint"] = fingerprint
    files.summary["config_fingerprint"] = fingerprint
    for row in files.rows:
        row["config_fingerprint"] = fingerprint


def test_physical_manifest_identity_binds_all_frozen_layers(
    tmp_path: Path,
) -> None:
    files, source_root, weights_dir, weights_zip = (
        _manifest_identity_fixture(tmp_path)
    )
    result = audit._validate_run_manifest_identity(
        repo_root=tmp_path,
        source_root=source_root,
        weights_dir=weights_dir,
        weights_zip=weights_zip,
        files=files,
    )
    assert result["status"] == "complete"
    assert result["physical_result_rows_bound"] == 2
    assert result["artifact_files"] == 2
    assert result["source_freeze_bound"] is True
    assert result["asset_freeze_bound"] is True
    assert result["config_freeze_bound"] is True


@pytest.mark.parametrize(
    "tamper",
    [
        "schema_version",
        "run_id",
        "status",
        "config_fingerprint",
        "row_config_fingerprint",
        "results_path",
        "results_sha256",
        "summary_sha256",
        "artifact_files",
        "source_commit",
        "source_file_sha256",
        "zip_sha256",
        "checkpoint_schema",
        "config_checkpoint_sha256_resigned",
        "config_golden_resigned",
    ],
)
def test_physical_manifest_identity_rejects_tamper(
    tmp_path: Path,
    tamper: str,
) -> None:
    files, source_root, weights_dir, weights_zip = (
        _manifest_identity_fixture(tmp_path)
    )
    manifest = files.manifest
    if tamper == "schema_version":
        manifest["schema_version"] = "invented"
    elif tamper == "run_id":
        manifest["run_id"] = "other"
    elif tamper == "status":
        manifest["status"] = "running"
    elif tamper == "config_fingerprint":
        manifest["config_fingerprint"] = "0" * 64
    elif tamper == "row_config_fingerprint":
        files.rows[0]["config_fingerprint"] = "0" * 64
    elif tamper == "results_path":
        manifest["outputs"]["results_path"] = "outside.jsonl"
    elif tamper == "results_sha256":
        manifest["outputs"]["results_sha256"] = "0" * 64
    elif tamper == "summary_sha256":
        manifest["outputs"]["summary_sha256"] = "0" * 64
    elif tamper == "artifact_files":
        manifest["outputs"]["artifact_files"] = 1
    elif tamper == "source_commit":
        manifest["source"]["commit"] = "0" * 40
    elif tamper == "source_file_sha256":
        first = next(iter(manifest["source"]["files"].values()))
        first["sha256"] = "0" * 64
    elif tamper == "zip_sha256":
        manifest["assets"]["zip"]["sha256"] = "0" * 64
    elif tamper == "checkpoint_schema":
        manifest["assets"]["checkpoint"]["schema_sha256"] = "0" * 64
    elif tamper == "config_checkpoint_sha256_resigned":
        manifest["config"]["checkpoint_sha256"] = "0" * 64
        _resign_manifest_config(files)
    elif tamper == "config_golden_resigned":
        manifest["config"]["official_golden"] = {"status": "tampered"}
        _resign_manifest_config(files)
    else:  # pragma: no cover - protects the parametrization itself.
        raise AssertionError(tamper)
    with pytest.raises((ValueError, FileNotFoundError)):
        audit._validate_run_manifest_identity(
            repo_root=tmp_path,
            source_root=source_root,
            weights_dir=weights_dir,
            weights_zip=weights_zip,
            files=files,
        )


def test_artifact_replay_runs_full_model_and_independent_head(
    tmp_path: Path,
) -> None:
    files, expected, runtime, model = _artifact_fixture(tmp_path)
    result = audit.audit_artifacts(
        repo_root=tmp_path,
        all_inputs=expected,
        files=files,
        runtime=runtime,
        model=model,
    )
    assert result["selected_images_freshly_reopened"] == 2
    assert result["complete_model_forward_passes"] == 2
    assert result["feature_artifacts_validated"] == 2
    assert result["artifact_head_replays"] == 2
    assert result["maximum_feature_absolute_difference"] == 0.0
    assert result["maximum_crop_logit_absolute_difference"] == 0.0
    assert result["maximum_raw_logit_absolute_difference"] == 0.0
    assert result["strict_decision_replayed"] == "raw_logit > 0"
    physical = result["physically_independent_replay"]
    assert physical["runner_scores_trusted_as_input"] is False
    assert physical["runner_features_trusted_as_input"] is False
    assert physical["all_selected_images_replayed"] is True


def test_artifact_replay_rejects_feature_and_preprocess_tamper(
    tmp_path: Path,
) -> None:
    files, expected, runtime, model = _artifact_fixture(tmp_path)
    feature_tamper = copy.deepcopy(files)
    feature_tamper.rows[0]["feature_array_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="raw-array SHA-256"):
        audit.audit_artifacts(
            repo_root=tmp_path,
            all_inputs=expected,
            files=feature_tamper,
            runtime=runtime,
            model=model,
        )

    preprocess_tamper = copy.deepcopy(files)
    preprocess_tamper.rows[0]["preprocess"]["tensor_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="preprocess.tensor_sha256"):
        audit.audit_artifacts(
            repo_root=tmp_path,
            all_inputs=expected,
            files=preprocess_tamper,
            runtime=runtime,
            model=model,
        )


def test_artifact_replay_rejects_crop_logit_tamper_even_if_mean_is_same(
    tmp_path: Path,
) -> None:
    files, expected, runtime, model = _artifact_fixture(tmp_path)
    row = files.rows[0]
    path = tmp_path / row["artifact_path"]
    with np.load(path, allow_pickle=False) as archive:
        features = archive["features"].copy()
        logits = archive["crop_logits"].copy()
    logits[0] += np.float32(0.25)
    logits[1] -= np.float32(0.25)
    np.savez(path, features=features, crop_logits=logits)
    row["artifact_sha256"] = sha256_file(path)
    row["crop_logits_array_sha256"] = hashlib.sha256(
        logits.tobytes(order="C")
    ).hexdigest()
    with pytest.raises(ValueError, match="crop-logit artifact"):
        audit.audit_artifacts(
            repo_root=tmp_path,
            all_inputs=expected,
            files=files,
            runtime=runtime,
            model=model,
        )


def test_fresh_forward_requires_exact_five_by_768_hook_shape() -> None:
    class BadModel(_TinyBFree):
        def forward(self, image: torch.Tensor) -> torch.Tensor:
            feature = torch.zeros((1, 768))
            return self.model.head(feature)

    runtime = audit.ReplayRuntime(
        torch=torch,
        device=torch.device("cpu"),
        evidence={},
    )
    with pytest.raises(ValueError, match="five by 768"):
        audit._forward_with_evidence(
            BadModel(),
            torch.zeros((3, 20, 20)),
            runtime,
        )


def test_recompute_summary_rejects_recorded_metric_tamper(
    tmp_path: Path,
) -> None:
    files, _, _, _ = _artifact_fixture(tmp_path)
    iterations, seed = 5, 17
    recomputed = summarize_bfree_results(
        files.rows,
        files.expected,
        threshold=0.0,
        bootstrap_samples=iterations,
        seed=seed,
    )
    manifest = {
        "config": {
            "metrics": {
                "bootstrap_samples": iterations,
                "bootstrap_seed": seed,
                "fixed_threshold": 0.0,
                "threshold_operator": ">",
            }
        }
    }
    recorded = {
        **recomputed,
        "run_id": "run",
        "model": "B-Free",
        "generated_at": "2026-07-25T00:00:00Z",
    }
    result = audit.recompute_summary(
        result_rows=files.rows,
        expected_rows=files.expected,
        manifest=manifest,
        recorded_summary=recorded,
        independent_result_rows=copy.deepcopy(files.rows),
    )
    assert result["recorded_summary_exactly_recomputed"] is True
    assert (
        result["independent_full_model_summary_within_float_tolerance"]
        is True
    )

    tampered = copy.deepcopy(recorded)
    tampered["coverage"]["valid_images"] -= 1
    with pytest.raises(ValueError, match="summary.coverage"):
        audit.recompute_summary(
            result_rows=files.rows,
            expected_rows=files.expected,
            manifest=manifest,
            recorded_summary=tampered,
        )


def test_selected_rows_hash_includes_jsonl_newlines() -> None:
    rows = [{"sample_id": "a"}, {"sample_id": "b"}]
    expected = hashlib.sha256(
        "".join(f"{stable_json(row)}\n" for row in rows).encode("utf-8")
    ).hexdigest()
    assert audit._selected_rows_sha256(rows) == expected


def test_error_payload_requires_explicit_null_model_fields() -> None:
    expected = {
        "sample_id": "error",
        "task_id": "task",
        "kind": "real",
        "label": 0,
        "domain": "synthetic",
    }
    row = {
        "id": "error",
        "sample_id": "error",
        "task_id": "task",
        "kind": "real",
        "label": 0,
        "domain": "synthetic",
        "status": "error",
        "valid_for_metrics": False,
        "edit_visibility": "full",
        "edit_visible_gt_fraction": 1.0,
        "raw_logit": None,
        "ai_score": None,
        "score": None,
        "crop_logits": None,
        "classification_decision": None,
        "artifact_path": None,
    }
    audit._validate_result_identity(row, expected, index=0)
    missing = dict(row)
    missing.pop("crop_logits")
    with pytest.raises(ValueError, match="lacks crop_logits"):
        audit._validate_result_identity(missing, expected, index=0)
    non_null = dict(row, raw_logit=0.0)
    with pytest.raises(ValueError, match="raw_logit is not null"):
        audit._validate_result_identity(non_null, expected, index=0)


def test_adapter_contract_recursively_rejects_invented_localization() -> None:
    manifest = {
        "task_scope": {
            "valid_for_t1": True,
            "valid_for_t2": False,
        },
        "config": {
            "adapter_contract": {
                "feature_dimension": 768,
                "crop_count": 5,
                "fixed_threshold": 0.0,
                "threshold_operator": ">",
                "score_semantics": audit.SCORE_SEMANTICS,
                "feature_semantics": audit.FEATURE_SEMANTICS,
            },
        },
    }
    result = audit._validate_adapter_contract(manifest)
    assert result["valid_for_t2"] is False
    tampered = copy.deepcopy(manifest)
    tampered["config"]["adapter_contract"]["nested"] = {
        "attention_map_path": "invented.npy"
    }
    with pytest.raises(ValueError, match="unsupported"):
        audit._validate_adapter_contract(tampered)


def test_manifest_and_summary_json_are_rejected_if_they_claim_t2() -> None:
    payload = {
        "valid_for_t2": False,
        "nested": [{"task_scope": {"t2": False}}],
    }
    audit._reject_t2_localization_or_joint(payload, label="safe")
    tampered = copy.deepcopy(payload)
    tampered["nested"].append({"localization_metrics": {}})
    with pytest.raises(ValueError, match="unsupported"):
        audit._reject_t2_localization_or_joint(tampered, label="tampered")
