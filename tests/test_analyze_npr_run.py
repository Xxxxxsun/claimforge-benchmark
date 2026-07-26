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

from eval.opensource import analyze_npr_run as audit
from eval.opensource import run_npr
from eval.opensource.common import sha256_file


def _write_rgb(path: Path, array: np.ndarray) -> None:
    Image.fromarray(array.astype(np.uint8), mode="RGB").save(path)


def _write_mask(path: Path, array: np.ndarray) -> None:
    Image.fromarray(array.astype(np.uint8), mode="L").save(path)


def _score_fields(raw_logit: float) -> dict:
    raw = float(np.float32(raw_logit))
    probability = float(torch.sigmoid(torch.tensor(raw, dtype=torch.float32)))
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
            "fc_hook_calls": 1,
            "official_logit_exact_match": True,
            "official_probability_exact_match": True,
        },
    }


def _save_feature(path: Path, feature: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.asarray(feature, dtype=np.float32), allow_pickle=False)
    return sha256_file(path)


def test_frozen_runner_pins_match_analyzer() -> None:
    pins = audit._load_runner_pins()
    assert pins.MODEL_SOURCE_COMMIT == audit.FROZEN_SOURCE_COMMIT
    assert pins.CHECKPOINT["sha256"] == audit.FROZEN_CHECKPOINT_SHA256
    assert pins.CHECKPOINT["bytes"] == audit.FROZEN_CHECKPOINT_BYTES
    assert pins.PREPROCESS_PROFILE == audit.FROZEN_PROFILE
    assert pins.FEATURE_DIMENSION == 512
    assert pins.HF_SPACE_COMMIT == audit.FROZEN_HF_SPACE_COMMIT
    assert pins.HF_SOURCE_FILES == audit.FROZEN_HF_SOURCE_FILES


@pytest.mark.skipif(
    not audit.DEFAULT_HF_SOURCE_ROOT.is_dir(),
    reason="pinned NPR Hugging Face source cache is unavailable",
)
def test_cached_hf_space_is_verified_as_corroboration_only() -> None:
    pins = audit._load_runner_pins()
    result = audit._verify_hf_source_tree(
        audit.DEFAULT_HF_SOURCE_ROOT,
        pins=pins,
    )
    assert result["commit"] == audit.FROZEN_HF_SPACE_COMMIT
    assert result["source_files_validated"] == 3
    assert result["calls_model_eval"] is False
    record = audit._expected_hf_source_record(
        audit.DEFAULT_HF_SOURCE_ROOT,
        pins=pins,
    )
    assert record["deployment_mode_defect"]["calls_model_eval"] is False
    assert "corroborating" in record["role"]


@pytest.mark.parametrize(
    ("height", "width", "trim_bottom", "trim_right"),
    [
        (18, 20, 0, 0),
        (19, 20, 1, 0),
        (18, 21, 0, 1),
        (19, 21, 1, 1),
    ],
)
def test_independent_preprocess_is_runner_exact(
    tmp_path: Path,
    height: int,
    width: int,
    trim_bottom: int,
    trim_right: int,
) -> None:
    yy, xx = np.mgrid[:height, :width]
    rgb = np.stack(
        (
            (xx * 17 + yy * 3) % 256,
            (xx * 5 + yy * 19) % 256,
            (xx * 13 + yy * 7) % 256,
        ),
        axis=-1,
    ).astype(np.uint8)
    path = tmp_path / f"odd_{height}_{width}.png"
    _write_rgb(path, rgb)

    expected_tensor, expected_audit = run_npr.preprocess_image(path)
    actual = audit.preprocess_image(path, torch_module=torch)

    assert torch.equal(actual.tensor, expected_tensor)
    assert actual.audit == expected_audit
    assert actual.audit["trim_bottom"] == trim_bottom
    assert actual.audit["trim_right"] == trim_right
    assert actual.tensor.shape == (3, height - trim_bottom, width - trim_right)


def test_score_contract_uses_strict_threshold_and_all_aliases() -> None:
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
    tampered["classification"]["score"] = 0.5000001
    with pytest.raises(ValueError, match="classification.score"):
        audit._audit_score_fields(
            tampered,
            replay_raw_logit=0.0,
            replay_probability=0.5,
            raw_tolerance=0.0,
            probability_tolerance=0.0,
        )

    extra_alias = copy.deepcopy(row)
    extra_alias["classification"]["invented_score"] = 0.5
    with pytest.raises(ValueError, match="keys mismatch"):
        audit._audit_score_fields(
            extra_alias,
            replay_raw_logit=0.0,
            replay_probability=0.5,
            raw_tolerance=0.0,
            probability_tolerance=0.0,
        )


def test_score_contract_records_float32_boundary_decision_disagreement() -> None:
    raw = float(
        torch.nextafter(
            torch.tensor(0.0, dtype=torch.float32),
            torch.tensor(1.0, dtype=torch.float32),
        )
    )
    probability = float(torch.sigmoid(torch.tensor(raw, dtype=torch.float32)))
    assert raw > 0.0
    assert probability == 0.5
    row = {"id": "rounded-boundary", **_score_fields(raw)}
    result = audit._audit_score_fields(
        row,
        replay_raw_logit=raw,
        replay_probability=probability,
        raw_tolerance=0.0,
        probability_tolerance=0.0,
    )
    assert result["decision"] is False
    assert result["raw_logit_decision"] is True
    assert result["decision_equivalent"] is False


@pytest.mark.parametrize(
    "payload",
    [
        {"t2": {}},
        {"localization": None},
        {"nested": {"score_map_path": "invented.npy"}},
        {"metrics": [{"s_joint": 0.4}]},
        {"pixel_metrics": {}},
    ],
)
def test_t2_localization_and_joint_fields_are_rejected(payload) -> None:
    with pytest.raises(ValueError, match="unsupported"):
        audit._reject_t2_localization_or_joint(payload, label="payload")
    # Explicit capability declarations are allowed.
    audit._reject_t2_localization_or_joint(
        {"task_support": {"t2": False}, "valid_for_t2": False},
        label="scope",
    )


def test_feature_loader_checks_run_boundary_hash_shape_and_dtype(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "results" / "run"
    path = run_dir / "features" / "x.npy"
    digest = _save_feature(path, np.arange(512, dtype=np.float32))
    row = {
        "id": "x",
        "npr_feature_path": str(path.relative_to(tmp_path)),
        "npr_feature_sha256": digest,
        "npr_feature_shape": [512],
        "npr_feature_dtype": "float32",
        "npr_feature_semantics": audit.FEATURE_SEMANTICS,
    }
    feature, loaded = audit._load_feature(
        row,
        repo_root=tmp_path,
        run_dir=run_dir,
    )
    assert loaded == path
    assert feature.dtype == np.float32

    wrong_hash = dict(row, npr_feature_sha256="0" * 64)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        audit._load_feature(
            wrong_hash,
            repo_root=tmp_path,
            run_dir=run_dir,
        )

    outside = tmp_path / "outside.npy"
    outside_digest = _save_feature(outside, np.zeros(512, dtype=np.float32))
    outside_row = {
        **row,
        "npr_feature_path": str(outside),
        "npr_feature_sha256": outside_digest,
    }
    with pytest.raises(ValueError, match="outside its run directory"):
        audit._load_feature(
            outside_row,
            repo_root=tmp_path,
            run_dir=run_dir,
        )

    wrong_dtype = run_dir / "features" / "wrong.npy"
    np.save(wrong_dtype, np.zeros(512, dtype=np.float64), allow_pickle=False)
    dtype_row = {
        **row,
        "npr_feature_path": str(wrong_dtype.relative_to(tmp_path)),
        "npr_feature_sha256": sha256_file(wrong_dtype),
    }
    with pytest.raises(ValueError, match="feature dtype"):
        audit._load_feature(
            dtype_row,
            repo_root=tmp_path,
            run_dir=run_dir,
        )


def test_checkpoint_schema_accepts_bn_trackers_and_matches_runner() -> None:
    state = OrderedDict(
        [
            ("conv.weight", torch.arange(6, dtype=torch.float32).reshape(2, 3)),
            ("bn.num_batches_tracked", torch.tensor(7, dtype=torch.int64)),
        ]
    )
    actual = audit._checkpoint_schema(state, torch_module=torch)
    expected = run_npr._checkpoint_schema(state)
    assert actual == expected
    assert actual["entries"] == 2
    assert actual["elements"] == 7

    bad = OrderedDict(state)
    bad["bad"] = torch.tensor(float("nan"))
    with pytest.raises(ValueError, match="not finite"):
        audit._checkpoint_schema(bad, torch_module=torch)


def test_visibility_is_recomputed_after_bottom_right_trim(
    tmp_path: Path,
) -> None:
    height, width = 5, 7
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[1, 1] = 255
    mask[-1, -1] = 255
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
    assert visibility["edit_visibility"] == "partial"
    assert visibility["edit_visible_gt_pixels"] == 1
    assert visibility["edit_visible_gt_fraction"] == 0.5
    assert visibility["effective_native_xyxy"] == [0, 0, 6, 4]
    assert visibility["trim_bottom"] == 1
    assert visibility["trim_right"] == 1


def test_result_history_uses_last_physical_retry() -> None:
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


class _TinyFullNPR(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = torch.nn.Linear(512, 1)
        with torch.no_grad():
            self.fc1.weight.fill_(0.001)
            self.fc1.bias.fill_(0.125)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        pooled = image.mean(dim=(1, 2, 3), keepdim=False)
        feature = pooled[:, None].repeat(1, 512)
        return self.fc1(feature)


def _artifact_fixture(
    tmp_path: Path,
) -> tuple[
    audit.RunFiles,
    list[dict],
    audit.ReplayRuntime,
    _TinyFullNPR,
]:
    run_dir = tmp_path / "results" / "run"
    feature_dir = run_dir / "features"
    run_dir.mkdir(parents=True)
    height, width = 18, 20
    real_array = np.arange(height * width * 3, dtype=np.uint8).reshape(
        height,
        width,
        3,
    )
    forged_array = real_array.copy()
    forged_array[4:8, 5:9, 1] ^= np.uint8(31)
    real_path = tmp_path / "real.png"
    forged_path = tmp_path / "forged.png"
    _write_rgb(real_path, real_array)
    _write_rgb(forged_path, forged_array)
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[4:8, 5:9] = 255
    mask_path = tmp_path / "mask.png"
    _write_mask(mask_path, mask)
    common = {
        "task_id": "task",
        "pair_rank": 0,
        "domain": "test",
        "candidate": "mouse",
        "dataset_id": "dataset",
        "width": width,
        "height": height,
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
        "gt_positive_pixels": 16,
    }
    all_inputs = [real, forged]
    visibility = audit._pair_visibility(all_inputs, repo_root=tmp_path)["task"]
    model = _TinyFullNPR().eval()
    rows: list[dict] = []
    for canonical in all_inputs:
        prepared = audit.preprocess_image(
            Path(canonical["canonical_path"]),
            torch_module=torch,
        )
        captured: list[torch.Tensor] = []
        hook = model.fc1.register_forward_pre_hook(
            lambda _module, arguments: captured.append(arguments[0].detach())
        )
        with torch.inference_mode():
            output = model(prepared.tensor[None, ...])
        hook.remove()
        feature = captured[0].squeeze(0).numpy().astype(np.float32)
        feature_path = feature_dir / f"{canonical['sample_id']}.npy"
        digest = _save_feature(feature_path, feature)
        raw = float(output.reshape(()))
        row = {
            "id": canonical["sample_id"],
            "task_id": canonical["task_id"],
            "kind": canonical["kind"],
            "label": canonical["label"],
            "domain": canonical["domain"],
            "status": "ok",
            "preprocess_profile": audit.FROZEN_PROFILE,
            "preprocess": prepared.audit,
            "edit_visibility": visibility["edit_visibility"],
            "edit_visible_gt_fraction": visibility["edit_visible_gt_fraction"],
            "edit_visibility_evidence": visibility,
            "npr_feature_path": str(feature_path.relative_to(tmp_path)),
            "npr_feature_sha256": digest,
            "npr_feature_shape": [512],
            "npr_feature_dtype": "float32",
            "npr_feature_semantics": audit.FEATURE_SEMANTICS,
            "latency_ms": 1.0,
            "peak_cuda_memory_bytes": None,
            **_score_fields(raw),
        }
        rows.append(row)
    for name in (
        "results.jsonl",
        "expected_inputs.jsonl",
        "summary.json",
        "manifest.json",
    ):
        (run_dir / name).write_text("{}\n", encoding="utf-8")
    files = audit.RunFiles(
        run_dir=run_dir,
        results_path=run_dir / "results.jsonl",
        expected_path=run_dir / "expected_inputs.jsonl",
        summary_path=run_dir / "summary.json",
        manifest_path=run_dir / "manifest.json",
        rows=rows,
        expected=all_inputs,
        summary={},
        manifest={},
    )
    runtime = audit.ReplayRuntime(
        torch=torch,
        device=torch.device("cpu"),
        evidence={"device": "cpu"},
    )
    return files, all_inputs, runtime, model


def test_artifact_audit_redecodes_and_runs_complete_model(
    tmp_path: Path,
) -> None:
    files, all_inputs, runtime, model = _artifact_fixture(tmp_path)
    result = audit.audit_artifacts(
        repo_root=tmp_path,
        source_root=tmp_path,
        all_inputs=all_inputs,
        files=files,
        runtime=runtime,
        model=model,
    )
    assert result["complete_model_forward_passes"] == 2
    assert result["persisted_features_validated"] == 2
    assert result["maximum_feature_absolute_difference"] == 0.0
    assert result["maximum_raw_logit_absolute_difference"] == 0.0
    assert result["strict_decision_replayed"] == "probability > 0.5"
    assert result["raw_logit_decision_equivalence"]["all_images_equal"] is True
    assert result["probability_saturation"]["non_saturated_images"] == 2

    files.rows[0]["npr_feature_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        audit.audit_artifacts(
            repo_root=tmp_path,
            source_root=tmp_path,
            all_inputs=all_inputs,
            files=files,
            runtime=runtime,
            model=model,
        )


def test_artifact_audit_discloses_float32_decision_disagreements(
    tmp_path: Path,
) -> None:
    files, all_inputs, runtime, model = _artifact_fixture(tmp_path)
    raw = float(
        torch.nextafter(
            torch.tensor(0.0, dtype=torch.float32),
            torch.tensor(1.0, dtype=torch.float32),
        )
    )
    with torch.no_grad():
        model.fc1.weight.zero_()
        model.fc1.bias.fill_(raw)
    for row in files.rows:
        row.update(_score_fields(raw))

    result = audit.audit_artifacts(
        repo_root=tmp_path,
        source_root=tmp_path,
        all_inputs=all_inputs,
        files=files,
        runtime=runtime,
        model=model,
    )
    equivalence = result["raw_logit_decision_equivalence"]
    assert equivalence["all_images_equal"] is False
    assert equivalence["disagreement_images"] == 2
    assert equivalence["disagreement_ids"] == ["real", "forged"]


def test_summary_raw_logit_diagnostic_is_recomputed_and_tamper_rejected(
    tmp_path: Path,
) -> None:
    files, _, _, _ = _artifact_fixture(tmp_path)
    iterations = 5
    seed = 17
    main_summary = audit.summarize_npr_results(
        files.rows,
        files.expected,
        bootstrap_samples=iterations,
        seed=seed,
    )
    diagnostic = audit.summarize_npr_raw_logit_diagnostic(
        files.rows,
        files.expected,
        bootstrap_samples=iterations,
        seed=seed,
    )
    recorded = {
        **main_summary,
        "raw_logit_numerical_diagnostic": diagnostic,
        "run_id": "run",
        "model": "NPR",
        "model_slug": "npr_aigcdetect_progan4class",
        "checkpoint_id": "checkpoint",
        "preprocess_profile": audit.FROZEN_PROFILE,
        "config_fingerprint": "0" * 64,
        "generated_at": "2026-07-25T00:00:00Z",
    }
    manifest = {
        "config": {
            "metrics": {
                "bootstrap_samples": iterations,
                "bootstrap_seed": seed,
                "fixed_threshold": 0.5,
                "threshold_operator": ">",
            }
        }
    }
    replayed = audit.recompute_summary(
        result_rows=files.rows,
        expected_rows=files.expected,
        manifest=manifest,
        recorded_summary=recorded,
    )
    assert replayed["raw_logit_numerical_diagnostic"] == diagnostic
    assert replayed["raw_logit_diagnostic_independent_full_model_match"] is False
    assert diagnostic["released_decision_equivalence"]["all_valid_images_equal"]
    assert diagnostic["by_domain"]["test"]["bootstrap_samples"] == iterations
    assert diagnostic["by_edit_visibility"]["full"]["seed"] == seed + 2000

    tampered = copy.deepcopy(recorded)
    tampered["raw_logit_numerical_diagnostic"]["probability_saturation"][
        "exact_zero_images"
    ] += 1
    with pytest.raises(ValueError, match="raw_logit_numerical_diagnostic"):
        audit.recompute_summary(
            result_rows=files.rows,
            expected_rows=files.expected,
            manifest=manifest,
            recorded_summary=tampered,
        )

    extra_domain = copy.deepcopy(recorded)
    extra_domain["raw_logit_numerical_diagnostic"]["by_domain"]["invented"] = (
        copy.deepcopy(diagnostic["by_domain"]["test"])
    )
    with pytest.raises(ValueError, match="keys mismatch"):
        audit.recompute_summary(
            result_rows=files.rows,
            expected_rows=files.expected,
            manifest=manifest,
            recorded_summary=extra_domain,
        )

    extra_visibility = copy.deepcopy(recorded)
    extra_visibility["raw_logit_numerical_diagnostic"]["by_edit_visibility"][
        "partial"
    ] = copy.deepcopy(diagnostic["by_edit_visibility"]["full"])
    with pytest.raises(ValueError, match="keys mismatch"):
        audit.recompute_summary(
            result_rows=files.rows,
            expected_rows=files.expected,
            manifest=manifest,
            recorded_summary=extra_visibility,
        )

    extra_bootstrap = copy.deepcopy(recorded)
    extra_bootstrap["raw_logit_numerical_diagnostic"]["pair_bootstrap"]["invented"] = {
        "estimate": 0.0,
        "ci95_percentile": [0.0, 0.0],
    }
    with pytest.raises(ValueError, match="keys mismatch"):
        audit.recompute_summary(
            result_rows=files.rows,
            expected_rows=files.expected,
            manifest=manifest,
            recorded_summary=extra_bootstrap,
        )

    independent = copy.deepcopy(files.rows)
    independent[1]["raw_logit"] = independent[1]["raw_logit"] + 10.0
    with pytest.raises(ValueError, match="independent_full_model_replay"):
        audit.recompute_summary(
            result_rows=files.rows,
            expected_rows=files.expected,
            manifest=manifest,
            recorded_summary=recorded,
            independent_result_rows=independent,
        )


def _prefix_row(
    *,
    repo_root: Path,
    run_dir: Path,
    sample_id: str,
    feature: np.ndarray,
) -> dict:
    path = run_dir / "features" / f"{sample_id}.npy"
    digest = _save_feature(path, feature)
    return {
        "id": sample_id,
        "status": "ok",
        "preprocess_profile": audit.FROZEN_PROFILE,
        "checkpoint_id": "checkpoint",
        "edit_visibility": "full",
        "edit_visible_gt_fraction": 1.0,
        "edit_visibility_evidence": {"same": True},
        "task_scope": {
            "valid_for_t1": True,
            "valid_for_t2": False,
            "native_dense_output": False,
        },
        "preprocess": {"same": True},
        "npr_feature_path": str(path.relative_to(repo_root)),
        "npr_feature_sha256": digest,
        "npr_feature_shape": [512],
        "npr_feature_dtype": "float32",
        "npr_feature_semantics": audit.FEATURE_SEMANTICS,
        **_score_fields(0.25),
    }


def _prefix_files(
    *,
    root: Path,
    run_id: str,
    expected: list[dict],
    row: dict,
    pair_limit: int | None,
) -> audit.RunFiles:
    run_dir = root / "results" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
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
        "runtime": {"same": True},
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


def test_prefix_audit_requires_independent_identical_artifacts(
    tmp_path: Path,
) -> None:
    expected = [{"sample_id": "x"}]
    feature = np.arange(512, dtype=np.float32)
    full_dir = tmp_path / "results" / "full"
    prefix_dir = tmp_path / "results" / "prefix"
    full_row = _prefix_row(
        repo_root=tmp_path,
        run_dir=full_dir,
        sample_id="x",
        feature=feature,
    )
    prefix_row = _prefix_row(
        repo_root=tmp_path,
        run_dir=prefix_dir,
        sample_id="x",
        feature=feature,
    )
    full = _prefix_files(
        root=tmp_path,
        run_id="full",
        expected=expected,
        row=full_row,
        pair_limit=None,
    )
    prefix = _prefix_files(
        root=tmp_path,
        run_id="prefix",
        expected=expected,
        row=prefix_row,
        pair_limit=1,
    )
    result = audit.audit_prefix_reproducibility(
        repo_root=tmp_path,
        full=full,
        prefix=prefix,
    )
    assert result["samples_compared"] == 1

    reused = copy.deepcopy(prefix)
    reused.rows[0]["npr_feature_path"] = full.rows[0]["npr_feature_path"]
    reused.rows[0]["npr_feature_sha256"] = full.rows[0]["npr_feature_sha256"]
    with pytest.raises(ValueError, match="outside its run directory|reuses"):
        audit.audit_prefix_reproducibility(
            repo_root=tmp_path,
            full=full,
            prefix=reused,
        )


def test_runtime_contract_can_be_replayed_on_cpu() -> None:
    _, evidence = run_npr.configure_runtime("cpu")
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
            },
        },
    }
    replay = audit._replay_runtime(manifest)
    assert replay.device == torch.device("cpu")
    assert replay.evidence["cudnn_enabled"] is False
    assert replay.evidence["deterministic_algorithms_enabled"] is True
    assert replay.evidence["cublas_workspace_config"] == ":4096:8"


def test_adapter_contract_validates_current_runner_metrics_and_common() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    config = {"adapter_contract": run_npr.adapter_contract(repo_root)}
    result = audit._validate_adapter_contract(config, repo_root=repo_root)
    assert result["files_validated"] == 4
    assert result["paths"] == [
        "eval/opensource/common.py",
        "eval/opensource/maskclip_metrics.py",
        "eval/opensource/npr_metrics.py",
        "eval/opensource/run_npr.py",
    ]


def test_selected_rows_hash_includes_jsonl_newlines() -> None:
    rows = [{"sample_id": "a"}, {"sample_id": "b"}]
    expected = hashlib.sha256(
        "".join(f"{run_npr.stable_json(row)}\n" for row in rows).encode()
    ).hexdigest()
    assert audit._selected_rows_sha256(rows) == expected
