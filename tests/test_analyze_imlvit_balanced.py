from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from eval.opensource import analyze_imlvit_balanced as module
from eval.opensource import run_imlvit_balanced as runner
from eval.opensource.common import repo_relative, sha256_file


def test_sigmoid_float32_is_stable_and_bounded() -> None:
    logits = np.asarray([[-100.0, -1.0, 0.0, 1.0, 100.0]], dtype=np.float32)
    probabilities = module._sigmoid_float32(logits)
    assert probabilities.dtype == np.float32
    assert probabilities.flags.c_contiguous
    assert np.isfinite(probabilities).all()
    assert np.all(probabilities >= 0.0)
    assert np.all(probabilities <= 1.0)
    assert probabilities[0, 2] == np.float32(0.5)


def test_bilinear_identity_is_bit_exact() -> None:
    source = np.arange(12, dtype=np.float32).reshape(3, 4)
    restored = module._bilinear_align_corners_false(source, width=4, height=3)
    assert np.array_equal(restored, source)
    assert restored.flags.c_contiguous


def test_bilinear_constant_resize_preserves_constant() -> None:
    source = np.full((2, 3), np.float32(0.375), dtype=np.float32)
    restored = module._bilinear_align_corners_false(source, width=11, height=7)
    assert restored.shape == (7, 11)
    assert np.array_equal(
        restored, np.full((7, 11), np.float32(0.375), dtype=np.float32)
    )


@pytest.mark.parametrize(
    ("width", "height"),
    [(0, 1), (1, 0), (-1, 2)],
)
def test_bilinear_rejects_invalid_output(width: int, height: int) -> None:
    with pytest.raises(ValueError, match="positive"):
        module._bilinear_align_corners_false(
            np.zeros((2, 2), dtype=np.float32),
            width=width,
            height=height,
        )


def test_contract_digest_rejects_tampering() -> None:
    value = {"field": "value"}
    bound = {**value, "contract_sha256": module._fingerprint(value)}
    assert module._verify_contract_digest(bound, "unit") == bound
    with pytest.raises(ValueError, match="digest"):
        module._verify_contract_digest(
            {**bound, "field": "tampered"},
            "unit",
        )


def test_result_projection_removes_only_run_noise() -> None:
    row = {
        "sample_id": "0" * 24,
        "run_id": "a",
        "run_manifest_fingerprint": "0" * 64,
        "completed_at": "time",
        "latency_ms": 1.0,
        "peak_cuda_memory_bytes": 2,
        "raw_logits_model_path": "a",
        "score_map_model_path": "b",
        "score_map_path": "c",
        "mask_path": "d",
        "raw_logits_model_sha256": "1" * 64,
        "localization": {"native": {"f1": 0.25}},
    }
    projection = module._result_projection(row)
    assert "run_id" not in projection
    assert "latency_ms" not in projection
    assert projection["raw_logits_model_sha256"] == "1" * 64
    assert projection["localization"]["native"]["f1"] == 0.25


def _fake_smoke_bundle(tmp_path: Path, run_id: str) -> SimpleNamespace:
    artifact_root = tmp_path / run_id
    selected = tuple(
        {"sample_id": f"{index:024x}"} for index in range(module.SMOKE_IMAGES)
    )
    latest = []
    for row in selected:
        sample_id = row["sample_id"]
        paths = runner.artifact_paths(artifact_root, sample_id)
        for key, path in paths.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"{sample_id}:{key}".encode("ascii"))
        latest.append(
            {
                "sample_id": sample_id,
                "run_id": run_id,
                "run_manifest_fingerprint": (
                    "a" * 64 if run_id.endswith("a") else "b" * 64
                ),
                "completed_at": run_id,
                "latency_ms": 1.0,
                "peak_cuda_memory_bytes": 100,
                "raw_logits_model_path": f"{run_id}/raw.npy",
                "score_map_model_path": f"{run_id}/model.npy",
                "score_map_path": f"{run_id}/native.npy",
                "mask_path": f"{run_id}/mask.png",
                "artifact_sha256": "f" * 64,
                "localization": {"native": {"f1": 0.5}},
            }
        )
    return SimpleNamespace(
        mode="smoke",
        selected=selected,
        latest_results=tuple(latest),
        artifact_root=artifact_root,
        immutable={
            "run_id": run_id,
            "outputs": {"root": run_id},
            "runtime": {"device": "cuda:0"},
            "scientific": {"threshold": 0.5},
        },
    )


def test_smoke_comparison_is_byte_exact(tmp_path) -> None:
    first = _fake_smoke_bundle(tmp_path, "smoke_a")
    second = _fake_smoke_bundle(tmp_path, "smoke_b")
    comparison = module.compare_computational_results(first, second)
    assert comparison["status"] == "passed"
    assert comparison["images_compared"] == 20
    assert comparison["artifact_files_compared_byte_exact"] == 80


def test_smoke_comparison_detects_one_byte_change(tmp_path) -> None:
    first = _fake_smoke_bundle(tmp_path, "smoke_a")
    second = _fake_smoke_bundle(tmp_path, "smoke_b")
    sample_id = second.selected[0]["sample_id"]
    path = runner.artifact_paths(second.artifact_root, sample_id)["score_map_native"]
    path.write_bytes(path.read_bytes() + b"x")
    with pytest.raises(ValueError, match="artifact"):
        module.compare_computational_results(first, second)


def test_write_json_verified_round_trips(tmp_path) -> None:
    path = tmp_path / "report.json"
    digest = module._write_json_verified(path, {"b": 2, "a": 1})
    assert digest == sha256_file(path)
    assert path.read_text(encoding="utf-8").endswith("\n")


def test_safe_run_dir_rejects_symlink(tmp_path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    target = tmp_path / "target"
    target.mkdir()
    (root / "unit").symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        module._safe_run_dir(root, "unit", "unit run")


def test_native_map_loader_allows_exact_half_and_fails_closed_on_identity(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path
    artifact_root = repo_root / "outputs" / "opensource" / "imlvit" / "unit"
    sample_id = "0" * 24
    path = runner.artifact_paths(artifact_root, sample_id)["score_map_native"]
    path.parent.mkdir(parents=True)
    np.save(path, np.asarray([[0.5]], dtype=np.float32), allow_pickle=False)
    input_row = {
        "sample_id": sample_id,
        "width": 1,
        "height": 1,
    }
    result_row = {
        "sample_id": sample_id,
        "score_map_path": repo_relative(path, repo_root),
        "score_map_sha256": sha256_file(path),
    }
    bundle = SimpleNamespace(
        selected=(input_row,),
        repo_root=repo_root,
        artifact_root=artifact_root,
    )
    load_audit = module.NativeMapLoadAudit()
    callback = module._native_map_loader(bundle, load_audit)
    loaded = callback(input_row, result_row)
    assert np.array_equal(loaded, np.asarray([[0.5]], dtype=np.float32))
    assert load_audit.maps_loaded == 1
    assert load_audit.native_pixels_exactly_at_threshold == 1
    assert load_audit.native_images_with_pixels_exactly_at_threshold == 1

    with pytest.raises(ValueError, match="result identity"):
        callback(input_row, {**result_row, "sample_id": "1" * 24})
    with pytest.raises(ValueError):
        callback(input_row, {**result_row, "score_map_path": "wrong.npy"})
    with pytest.raises(ValueError, match="contract"):
        callback(input_row, {**result_row, "score_map_sha256": "f" * 64})


def test_recompute_metrics_separates_official_strict_gt_from_shared_gte(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in ("results.jsonl", "manifest.json", "summary.json"):
        (tmp_path / name).write_text("{}\n", encoding="utf-8")
    selected = tuple(
        {"sample_id": f"{index:024x}"} for index in range(module.FORMAL_IMAGES)
    )
    bundle = SimpleNamespace(
        mode="formal",
        selected=selected,
        release=SimpleNamespace(inputs=selected, manifest_sha256="a" * 64),
        latest_results=({},),
        repo_root=tmp_path,
        run_id=runner.DEFAULT_FORMAL_RUN_ID,
        fingerprint="0" * 64,
        contract=object(),
        results_path=tmp_path / "results.jsonl",
        manifest_path=tmp_path / "manifest.json",
        summary_path=tmp_path / "summary.json",
    )

    def fake_loader(_bundle, load_audit):
        def callback(_input_row, _result_row):
            load_audit.maps_loaded = module.FORMAL_IMAGES
            load_audit.native_pixels_exactly_at_threshold = 2
            load_audit.native_images_with_pixels_exactly_at_threshold = 2
            return np.asarray([[0.5]], dtype=np.float32)

        return callback

    calls = {}

    def fake_summarize(_inputs, _results, **kwargs):
        calls.update(kwargs)
        kwargs["load_native_score_map"]({}, {})
        return {
            "schema_version": module.T2_METRICS_SCHEMA_VERSION,
            "coverage": {
                "is_complete": True,
                "native_maps_evaluated": module.FORMAL_IMAGES,
            },
        }

    monkeypatch.setattr(module, "_native_map_loader", fake_loader)
    monkeypatch.setattr(module, "summarize_balanced250_t2", fake_summarize)
    monkeypatch.setattr(module, "analyzer_source_contract", lambda _root: {})
    report = module.recompute_metrics(bundle)
    assert calls["threshold"] == 0.5
    assert calls["threshold_operator"] == ">="
    assert report["official_t2_threshold_operator"] == ">"
    assert report["shared_t2_threshold_operator"] == ">="
    assert report["native_pixels_exactly_at_threshold"] == 2
    assert report["native_images_with_pixels_exactly_at_threshold"] == 2
    assert report["operator_equivalence_checked"] is True
    assert report["shared_t2_operator_equivalent_to_official"] is False
    assert report["operator_non_equivalence_observed_on_formal_artifacts"] is True


def test_exact_half_is_negative_for_official_mask_and_positive_for_shared() -> None:
    score = np.asarray(
        [0.5, np.nextafter(np.float32(0.5), np.float32(1.0))],
        dtype=np.float32,
    )
    official = score > np.float32(runner.MASK_THRESHOLD)
    shared = score >= np.float32(runner.MASK_THRESHOLD)
    assert official.tolist() == [False, True]
    assert shared.tolist() == [True, True]


def test_frozen_selection_hashes_are_distinct() -> None:
    assert len(module.FORMAL_SELECTED_IDS_SHA256) == 64
    assert len(module.SMOKE_SELECTED_IDS_SHA256) == 64
    assert module.FORMAL_SELECTED_IDS_SHA256 != module.SMOKE_SELECTED_IDS_SHA256


def test_r2_analyzer_schemas_are_v3() -> None:
    assert module.AUDIT_SCHEMA_VERSION.endswith("_v3")
    assert module.METRICS_SCHEMA_VERSION.endswith("_v3")
    assert module.SMOKE_COMPARISON_SCHEMA_VERSION.endswith("_v3")
    assert module.FRESH_REPLAY_SCHEMA_VERSION.endswith("_v3")


def test_analyzer_never_defines_t1_score_contract() -> None:
    assert runner.TASK_SCOPE["valid_for_t1"] is False
    assert runner.TASK_SCOPE["map_statistic_promoted_to_t1"] is False
    assert "score_spec" not in module.recompute_metrics.__annotations__
