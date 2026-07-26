from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image
from torchvision import transforms

from eval.opensource import analyze_community_forensics_run as audit
from eval.opensource import run_community_forensics as runner
from eval.opensource.common import sha256_file, stable_json
from eval.opensource.common import read_jsonl


def _write_rgb(path: Path, pixels: np.ndarray) -> None:
    Image.fromarray(pixels.astype(np.uint8), mode="RGB").save(path)


def _write_mask(path: Path, pixels: np.ndarray) -> None:
    Image.fromarray(pixels.astype(np.uint8), mode="L").save(path)


def _score_fields(raw_logit: float) -> dict:
    raw = float(np.float32(raw_logit))
    probability = float(
        torch.sigmoid(torch.tensor(raw, dtype=torch.float32)).item()
    )
    decision = probability > 0.5
    classification = {
        "raw_logit": raw,
        "probability": probability,
        "ai_score": probability,
        "score": probability,
        "decision": decision,
        "threshold": 0.5,
        "threshold_operator": ">",
        "semantics": audit.SCORE_SEMANTICS,
    }
    t1 = {
        key: value
        for key, value in classification.items()
        if key != "semantics"
    }
    t1["policy"] = audit.T1_POLICY
    return {
        "raw_logit": raw,
        "probability": probability,
        "ai_score": probability,
        "score": probability,
        "score_semantics": audit.SCORE_SEMANTICS,
        "classification_decision": decision,
        "classification_threshold": 0.5,
        "classification_threshold_operator": ">",
        "classification": classification,
        "t1": t1,
        "manual_replay": {
            "raw_logit": raw,
            "probability": probability,
            "ai_score": probability,
            "classification_decision": decision,
            "official_logit_exact_match": True,
            "official_probability_exact_match": True,
            "model_forward_calls": 1,
            "classifier_hook_calls": 1,
        },
    }


def _save_feature(
    *,
    repo_root: Path,
    run_dir: Path,
    sample_id: str,
    feature: np.ndarray,
) -> dict:
    path = run_dir / "features" / f"{sample_id}.npy"
    path.parent.mkdir(parents=True, exist_ok=True)
    contiguous = np.ascontiguousarray(feature, dtype=np.float32)
    np.save(path, contiguous, allow_pickle=False)
    relative = str(path.relative_to(repo_root))
    array_digest = hashlib.sha256(
        contiguous.tobytes(order="C")
    ).hexdigest()
    return {
        "commfor_feature_path": relative,
        "commfor_feature_sha256": sha256_file(path),
        "commfor_feature_array_sha256": array_digest,
        "feature_array_sha256": array_digest,
        "commfor_feature_shape": [384],
        "commfor_feature_dtype": "float32",
        "commfor_feature_semantics": audit.FEATURE_SEMANTICS,
        "artifact_paths": {"commfor_feature_npy": relative},
    }


def test_frozen_runner_pins_match_independent_constants() -> None:
    pins = audit._load_runner_pins()
    assert pins.MODEL_SOURCE_COMMIT == audit.FROZEN_SOURCE_COMMIT
    assert pins.EVAL_SINGLE_COMMIT == audit.FROZEN_EVAL_SINGLE_COMMIT
    assert pins.MODEL_HF_REVISION == audit.FROZEN_MODEL_REVISION
    assert pins.PROCESSOR_HF_REVISION == audit.FROZEN_PROCESSOR_REVISION
    assert pins.SOURCE_FILES == audit.FROZEN_MAIN_SOURCE_FILES
    assert pins.EVAL_SINGLE_FILES == audit.FROZEN_EVAL_SINGLE_FILES
    assert pins.MODEL_FILES == audit.FROZEN_MODEL_FILES
    assert pins.PROCESSOR_FILES == audit.FROZEN_PROCESSOR_FILES
    assert pins.CHECKPOINT["sha256"] == audit.FROZEN_CHECKPOINT_SHA256
    assert pins.CHECKPOINT["bytes"] == 87_262_324
    assert pins.CHECKPOINT["tensor_count"] == 152
    assert pins.CHECKPOINT["state_elements"] == 21_811_969
    assert pins.CHECKPOINT["trainable_parameters"] == 21_811_969
    assert pins.FEATURE_DIMENSION == 384
    assert pins.CANONICAL_RELEASE == audit.FROZEN_CANONICAL_RELEASE
    assert pins.GOLDEN_ABS_TOLERANCE == audit.GOLDEN_ABSOLUTE_TOLERANCE
    assert pins.GOLDEN_ABS_TOLERANCE == 1e-5
    assert tuple(
        (case["filename"], case["sha256"], case["probability"])
        for case in pins.GOLDEN_CASES
    ) == audit.FROZEN_GOLDEN_CASES


def test_frozen_canonical_release_and_pairs_are_cross_checked() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    release_path = (
        repo_root / "outputs/opensource/mouse_canonical_v1/manifest.json"
    )
    release = json.loads(release_path.read_text(encoding="utf-8"))
    for key, expected in audit.FROZEN_CANONICAL_RELEASE.items():
        assert release[key] == expected
    inputs_path = repo_root / release["inputs_path"]
    pairs_path = repo_root / release["pairs_path"]
    assert sha256_file(inputs_path) == audit.FROZEN_CANONICAL_RELEASE[
        "inputs_sha256"
    ]
    assert sha256_file(pairs_path) == audit.FROZEN_CANONICAL_RELEASE[
        "pairs_sha256"
    ]
    inputs = read_jsonl(inputs_path)
    pairs = read_jsonl(pairs_path)
    result = audit._audit_canonical_pairs(pairs, inputs)
    assert result["pairs"] == 275
    assert result["images"] == 550
    assert result["unique_task_ids"] == 275
    assert result["domain_census"] == {
        "lodging": 147,
        "restaurant": 128,
    }
    assert result["pairs_to_inputs_exact"] is True

    tampered = copy.deepcopy(pairs)
    tampered[0]["real"]["canonical_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="canonical_sha256"):
        audit._audit_canonical_pairs(tampered, inputs)


@pytest.mark.parametrize(
    ("height", "width"),
    [
        (31, 47),
        (47, 31),
        (51, 51),
        (29, 113),
    ],
)
def test_independent_preprocess_is_torchvision_and_runner_exact(
    tmp_path: Path,
    height: int,
    width: int,
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
    path = tmp_path / f"rgb_{height}_{width}.png"
    _write_rgb(path, pixels)

    expected_array, expected_audit = runner.preprocess_image(path)
    actual = audit.preprocess_image(path, torch_module=torch)

    assert np.array_equal(actual.tensor.numpy(), expected_array)
    assert actual.audit == expected_audit
    assert (
        actual.audit["geometry"]
        == runner.compute_preprocess_geometry(width, height)
    )
    assert actual.tensor.shape == (3, 384, 384)
    assert actual.tensor.dtype == torch.float32


def test_center_crop_small_image_padding_is_torchvision_exact() -> None:
    pixels = np.arange(5 * 3 * 3, dtype=np.uint8).reshape(3, 5, 3)
    image = Image.fromarray(pixels, mode="RGB")
    expected = np.asarray(transforms.CenterCrop(8)(image), dtype=np.uint8)
    actual = np.asarray(audit._center_crop_rgb(image, 8), dtype=np.uint8)
    geometry = audit._center_crop_geometry(5, 3, 8)

    assert np.array_equal(actual, expected)
    assert geometry["padding_ltrb"] == [1, 2, 2, 3]
    assert geometry["padded_size"] == [8, 8]
    assert np.count_nonzero(actual[:2]) == 0
    assert np.count_nonzero(actual[:, :1]) == 0


def test_score_contract_is_strict_and_rejects_alias_tamper() -> None:
    row = {"id": "boundary", **_score_fields(0.0)}
    replay = audit._audit_score_fields(
        row,
        replay_raw_logit=0.0,
        replay_probability=0.5,
        raw_tolerance=0.0,
        probability_tolerance=0.0,
    )
    assert replay["decision"] is False

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

    wrong_operator = copy.deepcopy(row)
    wrong_operator["classification_threshold_operator"] = ">="
    with pytest.raises(ValueError, match="classification operator"):
        audit._audit_score_fields(
            wrong_operator,
            replay_raw_logit=0.0,
            replay_probability=0.5,
            raw_tolerance=0.0,
            probability_tolerance=0.0,
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
def test_t2_localization_and_joint_fields_are_rejected(payload) -> None:
    with pytest.raises(ValueError, match="unsupported"):
        audit._reject_t2_localization_or_joint(payload, label="payload")
    audit._reject_t2_localization_or_joint(
        {"task_scope": {"t2": False}, "valid_for_t2": False},
        label="capability declaration",
    )


def test_error_payload_requires_explicit_null_scores() -> None:
    row = {
        "id": "error",
        "status": "error",
        "valid_for_metrics": False,
        "completed_at": "2026-07-25T00:00:00Z",
        "raw_logit": None,
        "probability": None,
        "ai_score": None,
        "score": None,
        "classification_decision": None,
        "error_type": "RuntimeError",
        "error": "synthetic",
        "traceback": "traceback",
    }
    audit._validate_result_payload(
        row,
        repo_root=Path("."),
        run_dir=Path("."),
        torch_module=torch,
    )
    missing = dict(row)
    missing.pop("raw_logit")
    with pytest.raises(ValueError, match="lacks raw_logit"):
        audit._validate_result_payload(
            missing,
            repo_root=Path("."),
            run_dir=Path("."),
            torch_module=torch,
        )
    non_null = dict(row, probability=0.0)
    with pytest.raises(ValueError, match="error row error probability"):
        audit._validate_result_payload(
            non_null,
            repo_root=Path("."),
            run_dir=Path("."),
            torch_module=torch,
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


def test_feature_loader_separates_npy_and_raw_array_hashes(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "results" / "run"
    feature = np.arange(384, dtype=np.float32)
    row = {
        "id": "x",
        **_save_feature(
            repo_root=tmp_path,
            run_dir=run_dir,
            sample_id="x",
            feature=feature,
        ),
    }
    loaded, path = audit._load_feature(
        row,
        repo_root=tmp_path,
        run_dir=run_dir,
    )
    assert np.array_equal(loaded, feature)
    assert sha256_file(path) == row["commfor_feature_sha256"]
    assert (
        hashlib.sha256(loaded.tobytes()).hexdigest()
        == row["commfor_feature_array_sha256"]
    )
    assert row["commfor_feature_sha256"] != row[
        "commfor_feature_array_sha256"
    ]

    swapped = {
        **row,
        "commfor_feature_sha256": row["commfor_feature_array_sha256"],
    }
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        audit._load_feature(
            swapped,
            repo_root=tmp_path,
            run_dir=run_dir,
        )

    outside = tmp_path / "outside.npy"
    np.save(outside, feature, allow_pickle=False)
    outside_row = {
        **row,
        "commfor_feature_path": str(outside),
        "commfor_feature_sha256": sha256_file(outside),
    }
    with pytest.raises(ValueError, match="outside its run directory"):
        audit._load_feature(
            outside_row,
            repo_root=tmp_path,
            run_dir=run_dir,
        )


class _TinyVit(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.head = torch.nn.Linear(384, 1)
        with torch.no_grad():
            self.head.weight.fill_(0.002)
            self.head.bias.fill_(0.125)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        pooled = image.mean(dim=(1, 2, 3))
        feature = pooled[:, None].repeat(1, 384)
        return self.head(feature)


class _TinyCommunityModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.vit = _TinyVit()

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.vit(image)


def _artifact_fixture(
    tmp_path: Path,
) -> tuple[
    audit.RunFiles,
    list[dict],
    audit.ReplayRuntime,
    _TinyCommunityModel,
]:
    run_dir = tmp_path / "results" / "run"
    run_dir.mkdir(parents=True)
    height, width = 24, 40
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
    model = _TinyCommunityModel().eval()
    rows: list[dict] = []
    for canonical in expected:
        prepared = audit.preprocess_image(
            Path(canonical["canonical_path"]),
            torch_module=torch,
        )
        captured: list[torch.Tensor] = []
        hook = model.vit.head.register_forward_pre_hook(
            lambda _module, arguments: captured.append(
                arguments[0].detach().clone()
            )
        )
        with torch.inference_mode():
            output = model(prepared.tensor.unsqueeze(0))
        hook.remove()
        feature = np.ascontiguousarray(
            captured[0].squeeze(0).numpy(),
            dtype=np.float32,
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
            **_save_feature(
                repo_root=tmp_path,
                run_dir=run_dir,
                sample_id=canonical["sample_id"],
                feature=feature,
            ),
            **_score_fields(float(output.reshape(()).item())),
        }
        rows.append(row)
    paths = {
        name: run_dir / name
        for name in (
            "results.jsonl",
            "expected_inputs.jsonl",
            "summary.json",
            "manifest.json",
        )
    }
    for path in paths.values():
        path.write_text("{}\n", encoding="utf-8")
    files = audit.RunFiles(
        run_dir=run_dir,
        results_path=paths["results.jsonl"],
        expected_path=paths["expected_inputs.jsonl"],
        summary_path=paths["summary.json"],
        manifest_path=paths["manifest.json"],
        rows=rows,
        expected=expected,
        summary={},
        manifest={},
    )
    runtime = audit.ReplayRuntime(
        torch=torch,
        device=torch.device("cpu"),
        evidence={"device": "cpu"},
    )
    return files, expected, runtime, model


def test_artifact_replay_runs_full_model_feature_and_head(
    tmp_path: Path,
) -> None:
    files, expected, runtime, model = _artifact_fixture(tmp_path)
    result = audit.audit_artifacts(
        repo_root=tmp_path,
        source_root=tmp_path,
        all_inputs=expected,
        files=files,
        runtime=runtime,
        model=model,
    )
    assert result["complete_model_forward_passes"] == 2
    assert result["persisted_features_validated"] == 2
    assert result["maximum_feature_absolute_difference"] == 0.0
    assert result["maximum_raw_logit_absolute_difference"] == 0.0
    assert result["persisted_feature_manual_f_linear_replayed"]
    assert result["strict_decision_replayed"] == "probability > 0.5"
    physical = result["physically_independent_replay"]
    assert physical["runner_scores_trusted_as_input"] is False
    assert physical["fresh_complete_model_forward_per_image"] is True
    assert physical["all_selected_images_replayed"] is True


def test_artifact_replay_rejects_feature_and_preprocess_tamper(
    tmp_path: Path,
) -> None:
    files, expected, runtime, model = _artifact_fixture(tmp_path)
    feature_tamper = copy.deepcopy(files)
    feature_tamper.rows[0]["commfor_feature_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        audit.audit_artifacts(
            repo_root=tmp_path,
            source_root=tmp_path,
            all_inputs=expected,
            files=feature_tamper,
            runtime=runtime,
            model=model,
        )

    preprocess_tamper = copy.deepcopy(files)
    preprocess_tamper.rows[0]["preprocess"]["tensor_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="independent preprocess"):
        audit.audit_artifacts(
            repo_root=tmp_path,
            source_root=tmp_path,
            all_inputs=expected,
            files=preprocess_tamper,
            runtime=runtime,
            model=model,
        )


def test_artifact_replay_preserves_error_to_ok_physical_history(
    tmp_path: Path,
) -> None:
    files, expected, runtime, model = _artifact_fixture(tmp_path)
    historical_error = {
        key: copy.deepcopy(value)
        for key, value in files.rows[0].items()
        if key
        in {
            "id",
            "sample_id",
            "task_id",
            "kind",
            "label",
            "domain",
            "edit_visibility",
            "edit_visible_gt_fraction",
            "edit_visibility_evidence",
        }
    }
    historical_error.update(
        {
            "status": "error",
            "valid_for_metrics": False,
            "raw_logit": None,
            "probability": None,
            "ai_score": None,
            "score": None,
            "classification_decision": None,
        }
    )
    files.rows.insert(0, historical_error)
    result = audit.audit_artifacts(
        repo_root=tmp_path,
        source_root=tmp_path,
        all_inputs=expected,
        files=files,
        runtime=runtime,
        model=model,
    )
    independent = result["_independent_result_rows"]
    assert len(independent) == 3
    assert [row["status"] for row in independent] == [
        "error",
        "ok",
        "ok",
    ]
    original_summary = audit.summarize_community_forensics_results(
        files.rows,
        expected,
        bootstrap_samples=5,
        seed=19,
    )
    independent_summary = audit.summarize_community_forensics_results(
        independent,
        expected,
        bootstrap_samples=5,
        seed=19,
    )
    assert independent_summary == original_summary
    assert original_summary["coverage"]["physical_result_rows"] == 3


def test_recompute_summary_rejects_metric_tamper(tmp_path: Path) -> None:
    files, _, _, _ = _artifact_fixture(tmp_path)
    iterations, seed = 5, 17
    recomputed = audit.summarize_community_forensics_results(
        files.rows,
        files.expected,
        bootstrap_samples=iterations,
        seed=seed,
    )
    golden = {"status": "passed"}
    manifest = {
        "run_id": "run",
        "config_fingerprint": "0" * 64,
        "official_golden": golden,
        "config": {
            "metrics": {
                "bootstrap_samples": iterations,
                "bootstrap_seed": seed,
                "fixed_threshold": 0.5,
                "threshold_operator": ">",
            }
        },
    }
    recorded = {
        **recomputed,
        "run_id": "run",
        "model": "Community Forensics",
        "model_slug": "community_forensics_highres_vit_s16_384",
        "checkpoint_id": "checkpoint",
        "preprocess_profile": audit.FROZEN_PROFILE,
        "config_fingerprint": "0" * 64,
        "official_golden_status": "passed",
        "official_golden_fingerprint": audit._fingerprint(golden),
        "generated_at": "2026-07-25T00:00:00Z",
    }
    result = audit.recompute_summary(
        result_rows=files.rows,
        expected_rows=files.expected,
        manifest=manifest,
        recorded_summary=recorded,
        independent_result_rows=copy.deepcopy(files.rows),
    )
    assert result["independent_complete_model_summary_match"] is True

    tampered = copy.deepcopy(recorded)
    tampered["coverage"]["valid_images"] -= 1
    with pytest.raises(ValueError, match="summary.coverage"):
        audit.recompute_summary(
            result_rows=files.rows,
            expected_rows=files.expected,
            manifest=manifest,
            recorded_summary=tampered,
        )


def _prefix_files(
    *,
    repo_root: Path,
    run_id: str,
    expected: list[dict],
    feature: np.ndarray,
    pair_limit: int | None,
) -> audit.RunFiles:
    run_dir = repo_root / "results" / run_id
    run_dir.mkdir(parents=True)
    sample_id = str(expected[0]["sample_id"])
    row = {
        "id": sample_id,
        "status": "ok",
        "valid_for_metrics": True,
        "preprocess_profile": audit.FROZEN_PROFILE,
        "preprocess": {"same": True},
        "edit_visibility": "full",
        "edit_visible_gt_fraction": 1.0,
        "edit_visibility_evidence": {"same": True},
        **_save_feature(
            repo_root=repo_root,
            run_dir=run_dir,
            sample_id=sample_id,
            feature=feature,
        ),
        **_score_fields(0.25),
    }
    paths = {
        "results": run_dir / "results.jsonl",
        "expected": run_dir / "expected_inputs.jsonl",
        "summary": run_dir / "summary.json",
        "manifest": run_dir / "manifest.json",
    }
    for path in paths.values():
        path.write_text("{}\n", encoding="utf-8")
    config = {
        "same": True,
        "dataset": {
            "same": True,
            "selected_ids": [item["sample_id"] for item in expected],
            "selected_rows_sha256": audit._selected_rows_sha256(expected),
            "pair_limit": pair_limit,
            "sample_id": None,
        },
    }
    manifest = {
        "config": config,
        "source": {"same": True},
        "assets": {"same": True},
        "model_audit": {"same": True},
        "official_golden": {"same": True},
        "runtime": {"same": True},
        "full_dataset_visibility_audit": {"same": True},
    }
    return audit.RunFiles(
        run_dir=run_dir,
        results_path=paths["results"],
        expected_path=paths["expected"],
        summary_path=paths["summary"],
        manifest_path=paths["manifest"],
        rows=[row],
        expected=expected,
        summary={},
        manifest=manifest,
    )


def test_prefix_requires_independent_exact_artifacts(
    tmp_path: Path,
) -> None:
    expected = [{"sample_id": "x"}]
    feature = np.arange(384, dtype=np.float32)
    full = _prefix_files(
        repo_root=tmp_path,
        run_id="full",
        expected=expected,
        feature=feature,
        pair_limit=None,
    )
    prefix = _prefix_files(
        repo_root=tmp_path,
        run_id="prefix",
        expected=expected,
        feature=feature,
        pair_limit=1,
    )
    result = audit.audit_prefix_reproducibility(
        repo_root=tmp_path,
        full=full,
        prefix=prefix,
    )
    assert result["samples_compared"] == 1
    assert result["independent_directories"] is True
    assert result["independent_feature_paths"] is True

    reused = copy.deepcopy(prefix)
    reused.rows[0]["commfor_feature_path"] = full.rows[0][
        "commfor_feature_path"
    ]
    reused.rows[0]["artifact_paths"] = {
        "commfor_feature_npy": full.rows[0]["commfor_feature_path"]
    }
    with pytest.raises(ValueError, match="outside its run directory|reuses"):
        audit.audit_prefix_reproducibility(
            repo_root=tmp_path,
            full=full,
            prefix=reused,
        )


def test_selected_rows_hash_includes_jsonl_newlines() -> None:
    rows = [{"sample_id": "a"}, {"sample_id": "b"}]
    expected = hashlib.sha256(
        "".join(f"{stable_json(row)}\n" for row in rows).encode("utf-8")
    ).hexdigest()
    assert audit._selected_rows_sha256(rows) == expected


def test_runtime_contract_replays_all_recorded_truth_fields_on_cpu() -> None:
    _, evidence = runner.configure_runtime("cpu")
    manifest = {
        "runtime": evidence,
        "config": {
            "runtime_evidence": evidence,
            "runtime_evidence_fingerprint": audit._fingerprint(evidence),
            "runtime_contract": {
                "device": "cpu",
                "seed": 100,
                "dtype": "float32",
                "autocast": False,
                "cudnn_enabled": False,
                "cudnn_benchmark": False,
                "cudnn_deterministic": True,
                "tf32": False,
                "deterministic_algorithms": True,
                "cublas_workspace_config": ":4096:8",
                "network_allowed": False,
            },
        },
    }
    replay = audit._replay_runtime(manifest)
    assert replay.device == torch.device("cpu")
    assert replay.evidence["cudnn_enabled"] is False
    assert replay.evidence["cuda_matmul_allow_tf32"] is False
    assert replay.evidence["cudnn_allow_tf32"] is False
    assert replay.evidence["deterministic_algorithms"] is True
    assert replay.evidence["float32_matmul_precision"] == "highest"


def test_runtime_truth_rejects_a_noop_or_failed_backend_setter() -> None:
    fake = SimpleNamespace(
        backends=SimpleNamespace(
            cudnn=SimpleNamespace(
                enabled=True,
                benchmark=False,
                deterministic=True,
                allow_tf32=False,
            ),
            cuda=SimpleNamespace(
                matmul=SimpleNamespace(allow_tf32=False),
            ),
        ),
        are_deterministic_algorithms_enabled=lambda: True,
        get_float32_matmul_precision=lambda: "highest",
    )
    with pytest.raises(ValueError, match="actual replay runtime truth"):
        audit._validated_runtime_truth(fake)
