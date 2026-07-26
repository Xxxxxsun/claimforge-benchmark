from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from eval.opensource import run_maskclip_balanced as runner
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


def test_runner_contract_constants_and_code_hash_inventory_are_frozen():
    assert runner.SCORE_SPEC.as_dict() == {
        "key": "ai_score",
        "direction": "higher_means_fake",
        "fixed_threshold": 0.5,
        "threshold_operator": ">=",
    }
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
    assert runner.DEFAULT_ARTIFACTS_DIR == Path(
        "outputs/opensource/maskclip"
    )
    assert set(runner.OPENSDI_SOURCE_FILES) == {
        "model/MaskCLIP.py",
        "model/clip_utils.py",
        "model/mae.py",
        "model/prompt_learner.py",
    }
    assert set(runner.ADAPTER_SOURCE_PATHS) == {
        "eval/opensource/run_maskclip_balanced.py",
        "eval/opensource/analyze_maskclip_balanced.py",
        "eval/opensource/run_maskclip.py",
        "eval/opensource/analyze_maskclip_run.py",
        "eval/opensource/maskclip_metrics.py",
        "eval/opensource/balanced250_localization_metrics.py",
        "eval/opensource/canonical_release.py",
        "eval/opensource/balanced_run_contract.py",
        "eval/opensource/balanced250_metrics.py",
        "eval/opensource/common.py",
    }


def test_formal_selection_is_exact_native_t1_t2_cache(
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


def test_smoke_selection_is_panel_first_five_for_every_condition(
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
    for condition in BALANCED_CONDITIONS:
        expected = sorted(
            (
                row
                for row in formal_release.inputs
                if row["condition"] == condition
            ),
            key=lambda row: (
                row.get("panel") is not True,
                row.get("selection_rank")
                if row.get("selection_rank") is not None
                else float("inf"),
                row["rank"],
            ),
        )[:5]
        actual = [
            row for row in selected if row["condition"] == condition
        ]
        assert [row["sample_id"] for row in actual] == [
            row["sample_id"] for row in expected
        ]


def test_mode_selectors_are_fail_closed(formal_release: CanonicalRelease):
    with pytest.raises(ValueError, match="formal mode"):
        runner.select_mode_inputs(
            formal_release,
            mode="formal",
            per_condition_limit=1,
            sample_id=None,
        )
    with pytest.raises(ValueError, match=r"\[1, 250\]"):
        runner.select_mode_inputs(
            formal_release,
            mode="smoke",
            per_condition_limit=0,
            sample_id=None,
        )
    with pytest.raises(ValueError, match="requires --sample-id"):
        runner.select_mode_inputs(
            formal_release,
            mode="single",
            per_condition_limit=None,
            sample_id=None,
        )


def test_result_identity_has_complete_v2_and_three_state_t2_semantics(
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
        assert "pair_rank" not in identity


def test_artifact_root_is_repo_local_and_run_id_bound(tmp_path: Path):
    run_id = "maskclip-test"
    expected = (
        tmp_path
        / runner.DEFAULT_ARTIFACTS_DIR
        / run_id
    ).resolve()
    assert runner.resolve_artifact_root(
        repo_root=tmp_path,
        run_id=run_id,
        artifact_root=None,
    ) == expected
    explicit = tmp_path / "custom" / run_id
    assert runner.resolve_artifact_root(
        repo_root=tmp_path,
        run_id=run_id,
        artifact_root=explicit,
    ) == explicit.resolve()
    with pytest.raises(ValueError, match="exact run-id"):
        runner.resolve_artifact_root(
            repo_root=tmp_path,
            run_id=run_id,
            artifact_root=tmp_path / "custom" / "wrong",
        )
    with pytest.raises(ValueError, match="escapes repository"):
        runner.resolve_artifact_root(
            repo_root=tmp_path,
            run_id=run_id,
            artifact_root=tmp_path.parent / run_id,
        )
    with pytest.raises(ValueError, match="run-id"):
        runner.resolve_artifact_root(
            repo_root=tmp_path,
            run_id="../escape",
            artifact_root=None,
        )
    with pytest.raises(ValueError, match="run-id"):
        runner.resolve_artifact_root(
            repo_root=tmp_path,
            run_id="unsafe run",
            artifact_root=None,
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


def _preprocess_audit(width: int = 8, height: int = 6) -> dict:
    tensor = np.zeros(
        (3, runner.legacy.MODEL_INPUT_SIZE, runner.legacy.MODEL_INPUT_SIZE),
        dtype=np.float32,
    )
    return {
        "profile": runner.PREPROCESS_PROFILE,
        "decoded_size": [width, height],
        "tensor_shape": list(tensor.shape),
        "tensor_dtype": "float32",
        "tensor_sha256": hashlib.sha256(tensor.tobytes()).hexdigest(),
        "input_resize": "opencv_inter_linear_stretch",
        "normalization_mean": runner.legacy.CLIP_MEAN.tolist(),
        "normalization_std": runner.legacy.CLIP_STD.tolist(),
    }


class _Capture:
    def __init__(self, _model):
        self.closed = False

    def close(self):
        self.closed = True


def _patch_cpu_run(
    monkeypatch,
    release: CanonicalRelease,
    *,
    inference_error: Exception | None = None,
) -> None:
    source = {
        "repository": runner.legacy.MODEL_REPO_URL,
        "root": "/source",
        "commit": runner.legacy.MODEL_SOURCE_COMMIT,
        "tracked_dirty": False,
        "core_source_files": {},
    }
    assets = {
        "maskclip": {
            "id": runner.CHECKPOINT_ID,
            "sha256": runner.legacy.CHECKPOINT_SHA256,
        }
    }
    monkeypatch.setattr(
        runner,
        "load_canonical_release",
        lambda *_args, **_kwargs: release,
    )
    monkeypatch.setattr(
        runner,
        "verify_assets",
        lambda **_kwargs: (source, assets),
    )
    monkeypatch.setattr(
        runner,
        "configure_runtime",
        lambda device_text: (
            SimpleNamespace(type="cpu"),
            {
                "device": device_text,
                "seed": runner.MODEL_SEED,
            },
        ),
    )
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
        runner.legacy,
        "load_model",
        lambda **_kwargs: (SimpleNamespace(), SimpleNamespace(type="cpu")),
    )
    monkeypatch.setattr(runner.legacy, "LogitCapture", _Capture)
    tensor = np.zeros(
        (3, runner.legacy.MODEL_INPUT_SIZE, runner.legacy.MODEL_INPUT_SIZE),
        dtype=np.float32,
    )
    audit = _preprocess_audit()
    monkeypatch.setattr(
        runner,
        "_preprocess_with_audit",
        lambda _path: (tensor.copy(), (8, 6), dict(audit)),
    )
    model_map = np.full(
        (runner.legacy.MODEL_INPUT_SIZE, runner.legacy.MODEL_INPUT_SIZE),
        0.75,
        dtype=np.float32,
    )

    def infer(*_args, **_kwargs):
        if inference_error is not None:
            raise inference_error
        return (
            np.asarray([0.0, np.log(3.0)], dtype=np.float32),
            np.asarray([0.25, 0.75], dtype=np.float32),
            0,
            1.25,
            model_map.copy(),
        )

    monkeypatch.setattr(runner.legacy, "infer_one", infer)
    monkeypatch.setattr(
        runner.legacy,
        "restore_score_map",
        lambda model_map, width, height: np.full(
            (height, width),
            float(np.asarray(model_map, dtype=np.float32)[0, 0]),
            dtype=np.float32,
        ),
    )
    monkeypatch.setattr(
        runner.legacy,
        "resize_target",
        lambda target, width, height: np.full(
            (height, width),
            bool(np.asarray(target, dtype=bool).any()),
            dtype=bool,
        ),
    )


def _single_args(
    root: Path,
    condition: str,
    *,
    resume: bool = False,
) -> list[str]:
    sample_id = _minimal_row(condition)["sample_id"]
    run_id = f"maskclip-balanced-{condition}-test"
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
        "--artifact-root",
        str(root / "outputs/artifacts" / run_id),
        "--device",
        "cpu",
    ]
    if resume:
        values.append("--resume")
    return values


def test_cpu_real_single_run_writes_t1_and_applicable_t2_artifacts(
    tmp_path: Path,
    monkeypatch,
):
    release = _minimal_release(tmp_path, "real")
    _patch_cpu_run(monkeypatch, release)
    assert runner.main(_single_args(tmp_path, "real")) == 0

    run_id = "maskclip-balanced-real-test"
    run_dir = tmp_path / "results" / run_id
    artifact_root = tmp_path / "outputs/artifacts" / run_id
    manifest = json.loads((run_dir / "manifest.json").read_text())
    summary = json.loads((run_dir / "summary.json").read_text())
    results = read_jsonl(run_dir / "results.jsonl")
    assert manifest["status"] == "complete"
    assert manifest["fingerprint"] == runner._fingerprint(
        manifest["immutable"]
    )
    assert manifest["immutable"]["outputs"]["artifact_root"] == (
        artifact_root.relative_to(tmp_path).as_posix()
    )
    assert summary["artifact_inventory"] == {
        "model_512": 1,
        "native": 1,
        "masks": 1,
    }
    assert len(results) == 1
    result = results[0]
    assert result["schema_version"] == runner.RESULT_SCHEMA_VERSION
    assert result["status"] == "ok"
    assert result["ai_score"] == 0.75
    assert result["classification_decision"] is True
    assert result["t2_applicable"] is True
    assert result["localization"]["native"]["target_positive_pixels"] == 0
    assert result["localization"]["native"]["predicted_positive_pixels"] == 48
    assert result["localization"]["native"]["pixel_ap"] is None

    model_map = np.load(
        tmp_path / result["score_map_model_path"],
        allow_pickle=False,
    )
    native_map = np.load(
        tmp_path / result["score_map_native_path"],
        allow_pickle=False,
    )
    assert model_map.shape == (512, 512)
    assert model_map.dtype == np.float32
    assert native_map.shape == (6, 8)
    assert native_map.dtype == np.float32
    with runner.Image.open(tmp_path / result["mask_path"]) as opened:
        mask = np.asarray(opened, dtype=np.uint8)
    assert np.array_equal(mask == 255, native_map >= 0.5)


def test_cpu_fullframe_single_saves_diagnostic_model_map_but_no_t2(
    tmp_path: Path,
    monkeypatch,
):
    release = _minimal_release(tmp_path, "fullframe_cat")
    _patch_cpu_run(monkeypatch, release)
    assert runner.main(_single_args(tmp_path, "fullframe_cat")) == 0
    run_id = "maskclip-balanced-fullframe_cat-test"
    run_dir = tmp_path / "results" / run_id
    result = read_jsonl(run_dir / "results.jsonl")[0]
    summary = json.loads((run_dir / "summary.json").read_text())
    assert result["t2_applicable"] is False
    assert result["t2_target_semantics"] == "not_applicable_fullframe"
    assert result["score_map_model_path"] is not None
    for field in (
        "score_map_native_path",
        "score_map_native_sha256",
        "score_map_native_bytes",
        "score_map_native_shape",
        "score_map_native_dtype",
        "score_map_native_semantics",
        "mask_path",
        "mask_sha256",
        "mask_bytes",
        "mask_shape",
        "mask_dtype",
        "mask_semantics",
        "localization",
    ):
        assert result[field] is None
    assert summary["artifact_inventory"] == {
        "model_512": 1,
        "native": 0,
        "masks": 0,
    }


def test_cpu_local_exact_diff_uses_verified_gt_and_saves_native_t2(
    tmp_path: Path,
    monkeypatch,
):
    release = _minimal_release(tmp_path, "local_cat")
    _patch_cpu_run(monkeypatch, release)
    monkeypatch.setattr(
        runner.legacy,
        "resize_target",
        lambda target, width, height: np.zeros(
            (height, width),
            dtype=bool,
        ),
    )
    assert runner.main(_single_args(tmp_path, "local_cat")) == 0
    run_dir = tmp_path / "results/maskclip-balanced-local_cat-test"
    result = read_jsonl(run_dir / "results.jsonl")[0]
    summary = json.loads((run_dir / "summary.json").read_text())
    assert result["t2_applicable"] is True
    assert result["t2_target_semantics"] == "exact_diff_local_insertion"
    assert result["localization"]["native"]["target_positive_pixels"] == 1
    assert result["localization"]["native"]["pixel_ap"] is not None
    assert result["score_map_native_path"] is not None
    assert result["mask_path"] is not None
    assert summary["artifact_inventory"] == {
        "model_512": 1,
        "native": 1,
        "masks": 1,
    }


def test_resume_revalidates_preprocess_and_all_artifacts_without_model(
    tmp_path: Path,
    monkeypatch,
):
    release = _minimal_release(tmp_path, "real")
    _patch_cpu_run(monkeypatch, release)
    assert runner.main(_single_args(tmp_path, "real")) == 0
    monkeypatch.setattr(
        runner.legacy,
        "load_model",
        lambda **_kwargs: pytest.fail("complete resume loaded the model"),
    )
    assert runner.main(_single_args(tmp_path, "real", resume=True)) == 0
    run_dir = tmp_path / "results/maskclip-balanced-real-test"
    assert len(read_jsonl(run_dir / "results.jsonl")) == 1
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["execution"]["resume_skips"] == 1
    assert manifest["execution"]["new_successes"] == 0


def test_resume_fails_closed_on_tampered_probability_map(
    tmp_path: Path,
    monkeypatch,
):
    release = _minimal_release(tmp_path, "real")
    _patch_cpu_run(monkeypatch, release)
    assert runner.main(_single_args(tmp_path, "real")) == 0
    result_path = (
        tmp_path
        / "results/maskclip-balanced-real-test/results.jsonl"
    )
    result = read_jsonl(result_path)[0]
    map_path = tmp_path / result["score_map_model_path"]
    manifest_path = result_path.parent / "manifest.json"
    manifest_before = manifest_path.read_bytes()
    runner.legacy._atomic_save_npy(
        map_path,
        np.zeros((512, 512), dtype=np.float32),
    )
    with pytest.raises(ValueError, match="file metadata changed"):
        runner.main(_single_args(tmp_path, "real", resume=True))
    assert manifest_path.read_bytes() == manifest_before


@pytest.mark.parametrize("initial_status", ["ok", "error"])
def test_resume_rejects_unknown_result_keys(
    tmp_path: Path,
    monkeypatch,
    initial_status: str,
):
    release = _minimal_release(tmp_path, "real")
    _patch_cpu_run(
        monkeypatch,
        release,
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
        / "results/maskclip-balanced-real-test/results.jsonl"
    )
    rows = read_jsonl(result_path)
    rows[0]["unexpected_field"] = "must fail closed"
    runner.atomic_write_jsonl(result_path, rows)
    with pytest.raises(ValueError, match="key set changed"):
        runner.main(_single_args(tmp_path, "real", resume=True))


def test_inference_error_is_append_only_invalid_and_cleans_artifacts(
    tmp_path: Path,
    monkeypatch,
):
    release = _minimal_release(tmp_path, "real")
    _patch_cpu_run(
        monkeypatch,
        release,
        inference_error=RuntimeError("inference failed"),
    )
    assert runner.main(_single_args(tmp_path, "real")) == 2
    run_id = "maskclip-balanced-real-test"
    run_dir = tmp_path / "results" / run_id
    artifact_root = tmp_path / "outputs/artifacts" / run_id
    result = read_jsonl(run_dir / "results.jsonl")[0]
    summary = json.loads((run_dir / "summary.json").read_text())
    assert result["status"] == "error"
    assert result["valid_for_metrics"] is False
    assert "ai_score" not in result
    assert summary["coverage"]["is_complete"] is False
    assert summary["artifact_inventory"] == {
        "model_512": 0,
        "native": 0,
        "masks": 0,
    }
    assert not any(
        path.is_file()
        for path in artifact_root.rglob("*")
    )


def test_inventory_rejects_extra_files(tmp_path: Path):
    artifact_root = tmp_path / "artifacts"
    for name in (
        "score_maps_model_512",
        "score_maps_native",
        "masks_native",
    ):
        (artifact_root / name).mkdir(parents=True, exist_ok=True)
    extra = artifact_root / "score_maps_model_512/extra.npy"
    extra.write_bytes(b"extra")
    with pytest.raises(ValueError, match="inventory mismatch"):
        runner.validate_artifact_inventory(
            run_dir=artifact_root,
            selected=[_minimal_row("real")],
            latest_by_sample_id={},
        )


def test_inventory_rejects_extra_top_level_entry(tmp_path: Path):
    artifact_root = tmp_path / "artifacts"
    runner._prepare_artifact_root(artifact_root)
    (artifact_root / "unexpected").mkdir()
    with pytest.raises(ValueError, match="artifact root inventory mismatch"):
        runner.validate_artifact_inventory(
            run_dir=artifact_root,
            selected=[_minimal_row("real")],
            latest_by_sample_id={},
        )


def test_verify_assets_rejects_indeterminate_git_status(
    tmp_path: Path,
    monkeypatch,
):
    opensdi_root = tmp_path / "OpenSDI"
    opensdi_root.mkdir()

    def git_value(_root, *arguments):
        if arguments == ("rev-parse", "HEAD"):
            return runner.legacy.MODEL_SOURCE_COMMIT
        assert arguments == (
            "status",
            "--short",
            "--untracked-files=no",
        )
        return None

    monkeypatch.setattr(runner.legacy, "_git_value", git_value)
    with pytest.raises(ValueError, match="cannot inspect"):
        runner.verify_assets(
            opensdi_root=opensdi_root,
            checkpoint_path=tmp_path / "checkpoint.pth",
            clip_checkpoint_path=tmp_path / "clip.pt",
        )


def test_cublas_workspace_configuration_is_exact(monkeypatch):
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    assert (
        runner._configure_cublas_workspace()
        == runner.CUBLAS_WORKSPACE_CONFIG
    )
    assert (
        runner.os.environ["CUBLAS_WORKSPACE_CONFIG"]
        == runner.CUBLAS_WORKSPACE_CONFIG
    )
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    with pytest.raises(ValueError, match="must be exactly"):
        runner._configure_cublas_workspace()


def test_normal_run_hashes_adapter_sources_before_runtime_configuration(
    tmp_path: Path,
    monkeypatch,
):
    release = _minimal_release(tmp_path, "real")
    _patch_cpu_run(monkeypatch, release)
    events: list[str] = []

    def adapter_sources(_root):
        events.append("adapter_sources")
        return {
            path: {
                "path": path,
                "bytes": index + 1,
                "sha256": f"{index + 1:064x}",
            }
            for index, path in enumerate(runner.ADAPTER_SOURCE_PATHS)
        }

    def configure(device_text):
        assert events == ["adapter_sources"]
        events.append("configure_runtime")
        return (
            SimpleNamespace(type="cpu"),
            {"device": device_text, "seed": runner.MODEL_SEED},
        )

    monkeypatch.setattr(runner, "adapter_source_contract", adapter_sources)
    monkeypatch.setattr(runner, "configure_runtime", configure)
    assert runner.main(_single_args(tmp_path, "real")) == 0
    assert events == ["adapter_sources", "configure_runtime"]


def test_preflight_is_cpu_only_and_does_not_configure_runtime(
    tmp_path: Path,
    monkeypatch,
    capsys,
):
    release = _minimal_release(tmp_path, "real")
    monkeypatch.setattr(
        runner,
        "load_canonical_release",
        lambda *_args, **_kwargs: release,
    )
    monkeypatch.setattr(
        runner,
        "verify_assets",
        lambda **_kwargs: (
            {"commit": runner.legacy.MODEL_SOURCE_COMMIT},
            {"maskclip": {"sha256": runner.legacy.CHECKPOINT_SHA256}},
        ),
    )
    expected_adapter_sources = {
        path: {
            "path": path,
            "bytes": index + 1,
            "sha256": f"{index + 1:064x}",
        }
        for index, path in enumerate(runner.ADAPTER_SOURCE_PATHS)
    }
    monkeypatch.setattr(
        runner,
        "adapter_source_contract",
        lambda _root: expected_adapter_sources,
    )
    monkeypatch.setattr(
        runner,
        "configure_runtime",
        lambda *_args, **_kwargs: pytest.fail(
            "preflight configured an accelerator runtime"
        ),
    )
    assert runner.main(
        [
            "--repo-root",
            str(tmp_path),
            "--mode",
            "preflight",
            "--device",
            "cpu",
        ]
    ) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "passed"
    assert report["adapter_sources"] == expected_adapter_sources
    assert report["cuda_used"] is False
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
