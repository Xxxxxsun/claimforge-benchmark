from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from eval.opensource import analyze_fsd_balanced as analyzer
from eval.opensource import run_fsd_balanced as real_runner
from eval.opensource.canonical_release import load_canonical_release
from eval.opensource.common import stable_json


REPO_ROOT = Path(__file__).resolve().parents[1]


class _FakeRunner:
    RUN_MANIFEST_SCHEMA = "fsd_balanced_run_manifest_v2"
    RUN_CONFIG_SCHEMA = "fsd_balanced_run_config_v2"
    RUNTIME_SUMMARY_SCHEMA = "fsd_balanced_runtime_summary_v2"
    CPU_PREFLIGHT_SCHEMA = "fsd_balanced_cpu_preflight_v1"
    DEFAULT_RESULTS_DIR = Path("results/opensource/fsd")
    DEFAULT_FORMAL_RUN_ID = analyzer.DEFAULT_RUN_ID
    DEFAULT_SOURCE_ROOT = Path("/source")
    DEFAULT_WEIGHTS_DIR = Path("/weights")
    DEFAULT_SEED = analyzer.EXPECTED_RUNTIME_SEED
    ADAPTER_SOURCE_PATHS = analyzer.EXPECTED_ADAPTER_SOURCE_PATHS
    IMMUTABLE_CONFIG_KEYS = analyzer.EXPECTED_IMMUTABLE_CONFIG_KEYS
    PREPROCESS_CONTRACT = analyzer.EXPECTED_PREPROCESS_CONTRACT
    MODEL_CONTRACT = analyzer.EXPECTED_MODEL_CONTRACT
    TASK_SCOPE = analyzer.EXPECTED_TASK_SCOPE
    ARTIFACT_CONTRACT = analyzer.EXPECTED_ARTIFACT_CONTRACT

    @staticmethod
    def validate_runtime_contract(value, *, label="runtime"):
        del label
        return value

    @staticmethod
    def configure_runtime(device_text, *, seed):
        return SimpleNamespace(type=device_text.split(":", 1)[0]), {
            "device": device_text,
            "seed": seed,
        }

    @staticmethod
    def _validate_runner_attempt(*_args, **_kwargs):
        return None

    @staticmethod
    def _valid_run_id(value: str) -> str:
        if (
            not isinstance(value, str)
            or not value
            or Path(value).name != value
            or value in (".", "..")
        ):
            raise ValueError("run-id is invalid")
        return value


@pytest.fixture(autouse=True)
def _fake_runner(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(analyzer, "_runner", lambda: _FakeRunner)


@pytest.fixture(scope="module")
def canonical_release():
    return load_canonical_release(
        REPO_ROOT,
        real_runner.DEFAULT_DATASET_MANIFEST,
        verify_files=False,
    )


def _save_descriptor(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        np.save(handle, value, allow_pickle=False)


def _scores(ai_score: float = 3.0) -> tuple[float, float, float]:
    z_score = -float(ai_score)
    raw = analyzer.legacy.TRAIN_MEAN + z_score * analyzer.legacy.TRAIN_STD
    z_score = (raw - analyzer.legacy.TRAIN_MEAN) / analyzer.legacy.TRAIN_STD
    return raw, z_score, -z_score


def _row(
    root: Path,
    sample_id: str = "sample",
    *,
    run_id: str = "run-a",
    fingerprint: str = "a" * 64,
    array: np.ndarray | None = None,
    ai_score: float = 3.0,
) -> dict:
    descriptor = (
        np.arange(analyzer.DESCRIPTOR_DIMENSION, dtype=np.float64)
        if array is None
        else np.asarray(array)
    )
    relative = (
        Path("outputs")
        / "opensource"
        / "fsd"
        / run_id
        / "raw_descriptors"
        / f"{sample_id}.npy"
    )
    path = root / relative
    _save_descriptor(path, descriptor)
    file_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    array_sha = analyzer._array_sha256(descriptor)
    raw, z_score, score = _scores(ai_score)
    decision = score > analyzer.legacy.AI_SCORE_THRESHOLD
    released = z_score < analyzer.legacy.RELEASED_Z_THRESHOLD
    classification = {
        "score": score,
        "raw_likelihood": raw,
        "released_z_score": z_score,
        "decision": decision,
        "threshold": analyzer.legacy.AI_SCORE_THRESHOLD,
        "threshold_operator": analyzer.legacy.THRESHOLD_OPERATOR,
        "semantics": "higher_is_more_AI_negative_released_z",
    }
    return {
        "schema_version": "opensource_result_v2",
        "run_id": run_id,
        "run_manifest_fingerprint": fingerprint,
        "config_fingerprint": fingerprint,
        "dataset_id": "dataset",
        "id": sample_id,
        "sample_id": sample_id,
        "rank": 0,
        "condition": "real",
        "condition_family": "real",
        "manipulation_scope": "authentic",
        "normalized_task_id": "normalized",
        "task_id": "task",
        "kind": "real",
        "label": 0,
        "domain": "lodging",
        "gt_mask_kind": "all_zero",
        "input_path": f"inputs/{sample_id}.jpg",
        "input_sha256": "b" * 64,
        "input_width": 32,
        "input_height": 32,
        "status": "ok",
        "valid_for_metrics": True,
        "completed_at": "2026-07-26T00:00:00+00:00",
        "model": analyzer.legacy.MODEL_NAME,
        "model_slug": analyzer.legacy.MODEL_SLUG,
        "task_scope": {
            "valid_for_t1": True,
            "valid_for_t2": False,
            "native_dense_output": False,
        },
        "preprocess": analyzer.legacy.compute_preprocess_geometry(32, 32),
        "descriptor": {
            "relative_path": relative.as_posix(),
            "sha256": file_sha,
            "file_bytes": path.stat().st_size,
            "array_sha256": array_sha,
            "dtype": "float64",
            "shape": [analyzer.DESCRIPTOR_DIMENSION],
            "nbytes": analyzer.DESCRIPTOR_NBYTES,
            "finite": True,
            "semantics": (
                "official_compute_fsd_before_released_transforms"
            ),
        },
        "raw_descriptor_path": relative.as_posix(),
        "raw_descriptor_sha256": file_sha,
        "raw_descriptor_array_sha256": array_sha,
        "raw_descriptor_shape": [analyzer.DESCRIPTOR_DIMENSION],
        "raw_descriptor_dtype": "float64",
        "raw_descriptor_nbytes": analyzer.DESCRIPTOR_NBYTES,
        "raw_descriptor_semantics": (
            "official_compute_fsd_before_released_transforms"
        ),
        "artifact_paths": {"raw_descriptor_npy": relative.as_posix()},
        "raw_likelihood": raw,
        "released_z_score": z_score,
        "ai_score": score,
        "score": score,
        "score_semantics": "negative_released_FSD_z_score",
        "released_is_fake": released,
        "released_threshold": analyzer.legacy.RELEASED_Z_THRESHOLD,
        "released_threshold_operator": (
            analyzer.legacy.RELEASED_THRESHOLD_OPERATOR
        ),
        "classification_decision": decision,
        "classification_threshold": analyzer.legacy.AI_SCORE_THRESHOLD,
        "classification_threshold_operator": analyzer.legacy.THRESHOLD_OPERATOR,
        "classification": classification,
        "t1": {
            "score": score,
            "raw_likelihood": raw,
            "released_z_score": z_score,
            "decision": decision,
            "threshold": analyzer.legacy.AI_SCORE_THRESHOLD,
            "threshold_operator": analyzer.legacy.THRESHOLD_OPERATOR,
            "policy": "released_FSD_whole_image_score_sign_inverted",
        },
        "manual_replay": {
            "raw_likelihood": raw,
            "released_z_score": z_score,
            "ai_score": score,
            "released_is_fake": released,
            "classification_decision": decision,
            "official_raw_exact_match": True,
            "official_z_exact_match": True,
            "compute_fsd_calls": 1,
        },
        "preprocess_latency_ms": 1.0,
        "latency_ms": 2.0,
        "peak_cuda_memory_bytes": 123,
    }


def _inventory(root: Path, row: dict) -> dict[str, analyzer.DescriptorArtifact]:
    return analyzer.validate_descriptor_inventory(
        latest_results=[row],
        repo_root=root,
        descriptor_dir=(
            root
            / "outputs"
            / "opensource"
            / "fsd"
            / row["run_id"]
            / "raw_descriptors"
        ),
    )


def test_strict_json_rejects_duplicate_nonfinite_and_noncanonical_jsonl(
    tmp_path: Path,
):
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"a":1,"a":2}', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate JSON key"):
        analyzer._load_json(duplicate, "duplicate")
    nonfinite = tmp_path / "nonfinite.json"
    nonfinite.write_text('{"a":NaN}', encoding="utf-8")
    with pytest.raises(ValueError, match="non-finite"):
        analyzer._load_json(nonfinite, "nonfinite")
    rows = tmp_path / "rows.jsonl"
    rows.write_text('{"z": 1, "a": 2}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="not canonical"):
        analyzer._read_jsonl_strict(rows, "rows")
    rows.write_text(stable_json({"a": 2}), encoding="utf-8")
    with pytest.raises(ValueError, match="final newline"):
        analyzer._read_jsonl_strict(rows, "rows")


def test_descriptor_inventory_validates_bytes_hash_array_and_exact_files(
    tmp_path: Path,
):
    row = _row(tmp_path)
    artifacts = _inventory(tmp_path, row)
    artifact = artifacts["sample"]
    assert artifact.array.shape == (analyzer.DESCRIPTOR_DIMENSION,)
    assert artifact.array.dtype == np.float64
    assert artifact.array_sha256 == row["descriptor"]["array_sha256"]

    extra = artifact.path.parent / "extra.npy"
    extra.write_bytes(b"extra")
    with pytest.raises(ValueError, match="inventory mismatch"):
        _inventory(tmp_path, row)
    extra.unlink()
    artifact.path.unlink()
    with pytest.raises(ValueError, match="inventory mismatch"):
        _inventory(tmp_path, row)


def test_descriptor_inventory_rejects_tamper_nan_and_keyset(tmp_path: Path):
    row = _row(tmp_path)
    path = tmp_path / row["descriptor"]["relative_path"]
    changed = np.ones(analyzer.DESCRIPTOR_DIMENSION, dtype=np.float64)
    _save_descriptor(path, changed)
    with pytest.raises(ValueError, match="artifact hash mismatch"):
        _inventory(tmp_path, row)

    changed[4] = np.nan
    _save_descriptor(path, changed)
    row["descriptor"]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    row["descriptor"]["file_bytes"] = path.stat().st_size
    row["descriptor"]["array_sha256"] = analyzer._array_sha256(changed)
    row["raw_descriptor_sha256"] = row["descriptor"]["sha256"]
    row["raw_descriptor_array_sha256"] = row["descriptor"]["array_sha256"]
    with pytest.raises(ValueError, match="array is invalid"):
        _inventory(tmp_path, row)

    key_root = tmp_path / "keyset-case"
    row = _row(key_root, sample_id="keyset")
    row["descriptor"]["extra"] = True
    with pytest.raises(ValueError, match="key set"):
        _inventory(key_root, row)
    del row["descriptor"]["extra"]
    del row["descriptor"]["array_sha256"]
    with pytest.raises(ValueError, match="key set"):
        _inventory(key_root, row)


def test_descriptor_inventory_rejects_symlink_and_alias_drift(tmp_path: Path):
    row = _row(tmp_path)
    path = tmp_path / row["descriptor"]["relative_path"]
    target = tmp_path / "target.npy"
    path.replace(target)
    path.symlink_to(target)
    with pytest.raises(ValueError, match="non-regular|symlink"):
        _inventory(tmp_path, row)

    alias_root = tmp_path / "alias-case"
    row = _row(alias_root, sample_id="alias")
    row["raw_descriptor_array_sha256"] = "0" * 64
    with pytest.raises(
        ValueError,
        match="descriptor alias raw_descriptor_array_sha256 differs",
    ):
        _inventory(alias_root, row)
    missing_root = tmp_path / "missing-alias"
    row = _row(missing_root, sample_id="missing")
    del row["raw_descriptor_semantics"]
    with pytest.raises(ValueError, match="raw_descriptor_semantics differs"):
        _inventory(missing_root, row)
    row = _row(
        tmp_path / "missing-artifact-path",
        sample_id="missing-path",
    )
    row["artifact_paths"] = {}
    with pytest.raises(ValueError, match="artifact path alias differs"):
        _inventory(tmp_path / "missing-artifact-path", row)

    directory_root = tmp_path / "directory-link"
    row = _row(directory_root, sample_id="linked-dir")
    descriptor_dir = (
        directory_root
        / "outputs"
        / "opensource"
        / "fsd"
        / row["run_id"]
        / "raw_descriptors"
    )
    target_dir = directory_root / "descriptor-target"
    descriptor_dir.replace(target_dir)
    descriptor_dir.symlink_to(target_dir, target_is_directory=True)
    with pytest.raises(ValueError, match="descriptor directory.*symlink"):
        _inventory(directory_root, row)


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda row: row.__setitem__("ai_score", float("nan")), "not finite"),
        (
            lambda row: row.__setitem__("classification_decision", False),
            "strict decision",
        ),
        (
            lambda row: row.__setitem__(
                "classification_threshold_operator",
                ">=",
            ),
            "fixed threshold",
        ),
        (
            lambda row: row["manual_replay"].__setitem__(
                "compute_fsd_calls",
                2,
            ),
            "manual tail",
        ),
    ],
)
def test_score_calibration_threshold_and_tail_are_fail_closed(
    tmp_path: Path,
    mutator,
    message: str,
):
    row = _row(tmp_path)
    mutator(row)
    with pytest.raises(ValueError, match=message):
        analyzer._validate_score_payload(row, sample_id="sample")


def test_strict_threshold_equality_is_real(tmp_path: Path):
    row = _row(tmp_path, ai_score=2.0)
    assert row["classification_decision"] is False
    assert row["released_is_fake"] is False
    analyzer._validate_score_payload(row, sample_id="sample")


def test_pair_rank_t2_joint_and_dense_claims_are_rejected():
    analyzer._reject_unsupported_claims(
        {
            "task_scope": {
                "valid_for_t2": False,
                "native_dense_output": False,
                "localization_output": None,
                "joint_score": None,
            },
            "pixel_center_mapping": {"formula": "diagnostic_only"},
        },
        "allowed",
    )
    for payload in (
        {"pair_rank": 1},
        {"valid_for_t2": True},
        {"nested": {"score_map": [0.1]}},
        {"t2": {"pixel_ap": 0.5}},
        {"joint_score": 0.5},
        {"predicted_mask_path": "mask.npy"},
        {"localization_metric": 0.5},
        {"heatmap_path": "map.npy"},
    ):
        with pytest.raises(ValueError, match="unsupported"):
            analyzer._reject_unsupported_claims(payload, "payload")


def test_analyzer_recomputes_crop_visibility_and_rejects_tamper(
    canonical_release,
):
    by_id = {
        row["sample_id"]: row for row in canonical_release.inputs
    }
    cases = {
        "5f7535f0b957874982b1b080": ("none", 0.0),
        "a963eb97f270fb854f964463": (
            "partial",
            0.8169155765554433,
        ),
        "a5f0c416a390a5125e4e2fcc": ("full", 1.0),
    }
    selected = []
    physical = []
    for sample_id, (category, fraction) in cases.items():
        input_row = by_id[sample_id]
        diagnostic = analyzer._independent_visibility_diagnostic(
            input_row,
            repo_root=REPO_ROOT,
        )
        assert diagnostic["edit_visibility"] == category
        assert diagnostic["edit_visible_gt_fraction"] == fraction
        selected.append(input_row)
        physical.append(
            {
                "sample_id": sample_id,
                "status": "error",
                **diagnostic,
            }
        )

    real = next(
        row for row in canonical_release.inputs if row["condition"] == "real"
    )
    fullframe = next(
        row
        for row in canonical_release.inputs
        if row["condition"] == "fullframe_mouse"
    )
    for input_row in (real, fullframe):
        diagnostic = analyzer._independent_visibility_diagnostic(
            input_row,
            repo_root=REPO_ROOT,
        )
        assert diagnostic["edit_visibility"] == "not_applicable"
        assert diagnostic["edit_visible_gt_fraction"] is None
        selected.append(input_row)
        physical.append(
            {
                "sample_id": input_row["sample_id"],
                "status": "error",
                **diagnostic,
            }
        )

    analyzer._validate_physical_attempts(
        physical=physical,
        selected=selected,
        repo_root=REPO_ROOT,
        run_id="run",
        fingerprint="a" * 64,
    )
    tampered = copy.deepcopy(physical)
    tampered[1]["edit_visible_gt_fraction"] = 0.0
    with pytest.raises(ValueError, match="crop visibility changed"):
        analyzer._validate_physical_attempts(
            physical=tampered,
            selected=selected,
            repo_root=REPO_ROOT,
            run_id="run",
            fingerprint="a" * 64,
        )


def test_smoke_comparison_ignores_only_runtime_identity_and_checks_bytes(
    tmp_path: Path,
):
    left_root = tmp_path / "left"
    right_root = tmp_path / "right"
    left = _row(left_root, run_id="smoke-a", fingerprint="a" * 64)
    right = _row(right_root, run_id="smoke-b", fingerprint="c" * 64)
    right["completed_at"] = "2026-07-26T01:00:00+00:00"
    right["preprocess_latency_ms"] = 99.0
    right["latency_ms"] = 100.0
    right["peak_cuda_memory_bytes"] = 9999
    report = analyzer.compare_computational_results(
        reference_rows=[left],
        replay_rows=[right],
        reference_descriptors=_inventory(left_root, left),
        replay_descriptors=_inventory(right_root, right),
    )
    assert report["images_compared"] == 1
    assert report["max_ai_score_abs_difference"] == 0.0
    assert report["max_descriptor_abs_difference"] == 0.0
    assert report["descriptor_file_bytes_exact"] is True

    right["unexpected_computation"] = {"value": 1}
    with pytest.raises(ValueError, match="projection differs"):
        analyzer.compare_computational_results(
            reference_rows=[left],
            replay_rows=[right],
            reference_descriptors=_inventory(left_root, left),
            replay_descriptors=_inventory(right_root, right),
        )


def test_smoke_comparison_rejects_descriptor_change_and_coverage(
    tmp_path: Path,
):
    left_root = tmp_path / "left"
    right_root = tmp_path / "right"
    left = _row(left_root, run_id="smoke-a")
    changed = np.arange(analyzer.DESCRIPTOR_DIMENSION, dtype=np.float64)
    changed[0] = 999.0
    right = _row(right_root, run_id="smoke-b", array=changed)
    with pytest.raises(ValueError, match="descriptor .* bytes differ"):
        analyzer.compare_computational_results(
            reference_rows=[left],
            replay_rows=[right],
            reference_descriptors=_inventory(left_root, left),
            replay_descriptors=_inventory(right_root, right),
        )
    with pytest.raises(ValueError, match="coverage differs"):
        analyzer.compare_computational_results(
            reference_rows=[left],
            replay_rows=[{**right, "sample_id": "other", "id": "other"}],
            reference_descriptors=_inventory(left_root, left),
            replay_descriptors=_inventory(right_root, right),
        )


def _immutable_stub(run_id: str) -> dict:
    value = {
        key: {"pinned": key}
        for key in analyzer.EXPECTED_IMMUTABLE_CONFIG_KEYS
    }
    value["run_id"] = run_id
    value["outputs"] = {"run_dir": run_id}
    value["runtime"] = _runtime()
    return value


def _smoke_bundle_stub(tmp_path: Path, run_id: str, immutable: dict):
    selected = tuple(
        {"sample_id": f"sample-{index}"} for index in range(35)
    )
    snapshot = {
        "manifest_sha256": "1" * 64,
        "results_sha256": "2" * 64,
        "expected_inputs_sha256": "3" * 64,
        "summary_sha256": "4" * 64,
        "descriptor_inventory_sha256": "5" * 64,
    }
    return SimpleNamespace(
        run_id=run_id,
        fingerprint="a" * 64,
        immutable=immutable,
        selected=selected,
        latest_results=(),
        descriptors={},
        contract=SimpleNamespace(
            selection=SimpleNamespace(as_dict=lambda: {"images": 35})
        ),
        manifest_path=tmp_path / run_id / "manifest.json",
        results_path=tmp_path / run_id / "results.jsonl",
        expected_path=tmp_path / run_id / "expected_inputs.jsonl",
        summary_path=tmp_path / run_id / "summary.json",
        descriptor_dir=tmp_path / "descriptors" / run_id,
        evidence_snapshot=snapshot,
    )


def test_smoke_comparison_rejects_computational_immutable_drift_and_validates_runtime(
    monkeypatch,
    tmp_path: Path,
):
    reference_immutable = _immutable_stub("smoke-a")
    replay_immutable = copy.deepcopy(reference_immutable)
    replay_immutable["run_id"] = "smoke-b"
    replay_immutable["outputs"] = {"run_dir": "smoke-b"}
    bundles = {
        "smoke-a": _smoke_bundle_stub(
            tmp_path,
            "smoke-a",
            reference_immutable,
        ),
        "smoke-b": _smoke_bundle_stub(
            tmp_path,
            "smoke-b",
            replay_immutable,
        ),
    }
    events = []
    monkeypatch.setattr(
        analyzer,
        "_actual_runtime_contract",
        lambda device: events.append(("runtime", device)) or _runtime(),
    )
    monkeypatch.setattr(
        analyzer,
        "load_smoke_run",
        lambda *, run_id, **_kwargs: bundles[run_id],
    )
    monkeypatch.setattr(
        analyzer,
        "compare_computational_results",
        lambda **_kwargs: events.append(("compare", None))
        or {"images_compared": 35},
    )
    monkeypatch.setattr(
        analyzer,
        "_verify_bundle_unchanged",
        lambda *_args, **_kwargs: None,
    )
    report = analyzer.compare_smoke_runs(
        repo_root=tmp_path,
        results_dir=tmp_path,
        reference_run_id="smoke-a",
        replay_run_id="smoke-b",
        output_path=None,
    )
    assert events[:2] == [("runtime", "cpu"), ("compare", None)]
    assert report["analysis_runtime"]["packages"]["scikit-learn"] == "1.8.0"

    bundles["smoke-b"].immutable["runtime"]["autocast"] = True
    with pytest.raises(
        ValueError,
        match="computational/runtime configurations differ",
    ):
        analyzer.compare_smoke_runs(
            repo_root=tmp_path,
            results_dir=tmp_path,
            reference_run_id="smoke-a",
            replay_run_id="smoke-b",
            output_path=None,
        )


def test_smoke_report_rejects_post_validation_mutation(
    monkeypatch,
    tmp_path: Path,
):
    immutable = _immutable_stub("smoke-a")
    replay_immutable = copy.deepcopy(immutable)
    replay_immutable["run_id"] = "smoke-b"
    replay_immutable["outputs"] = {"run_dir": "smoke-b"}
    reference = _smoke_bundle_stub(tmp_path, "smoke-a", immutable)
    replay = _smoke_bundle_stub(tmp_path, "smoke-b", replay_immutable)
    for bundle in (reference, replay):
        bundle.manifest_path.parent.mkdir(parents=True)
        for key, path in (
            ("manifest_sha256", bundle.manifest_path),
            ("results_sha256", bundle.results_path),
            ("expected_inputs_sha256", bundle.expected_path),
            ("summary_sha256", bundle.summary_path),
        ):
            path.write_bytes(key.encode())
            bundle.evidence_snapshot[key] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
        bundle.descriptor_dir.mkdir(parents=True)
    bundles = {"smoke-a": reference, "smoke-b": replay}
    monkeypatch.setattr(
        analyzer,
        "_actual_runtime_contract",
        lambda _device: _runtime(),
    )
    monkeypatch.setattr(
        analyzer,
        "load_smoke_run",
        lambda *, run_id, **_kwargs: bundles[run_id],
    )

    def mutate_after_compare(**_kwargs):
        reference.results_path.write_bytes(b"mutated")
        return {"images_compared": 35}

    monkeypatch.setattr(
        analyzer,
        "compare_computational_results",
        mutate_after_compare,
    )
    with pytest.raises(ValueError, match="changed after validation"):
        analyzer.compare_smoke_runs(
            repo_root=tmp_path,
            results_dir=tmp_path,
            reference_run_id="smoke-a",
            replay_run_id="smoke-b",
            output_path=None,
        )


def _smoke_rows() -> list[dict]:
    rows = []
    rank = 0
    for condition in analyzer.BALANCED_CONDITIONS:
        for index in range(analyzer.SMOKE_PER_CONDITION):
            rows.append(
                {
                    "sample_id": f"{condition}-{index}",
                    "condition": condition,
                    "rank": rank,
                }
            )
            rank += 1
    return rows


def test_rebuild_smoke_selection_is_exact_35(monkeypatch, tmp_path: Path):
    manifest_path = tmp_path / "release" / "manifest.json"
    manifest_path.parent.mkdir()
    manifest_path.write_text("{}", encoding="utf-8")
    rows = _smoke_rows()
    raw_contract = {
        "release": {"manifest_path": "release/manifest.json"},
        "selection": {
            "spec": {"per_condition_limit": analyzer.SMOKE_PER_CONDITION}
        },
    }

    class Contract:
        selection = SimpleNamespace(as_dict=lambda: raw_contract["selection"])
        capability = SimpleNamespace(
            as_dict=lambda: {
                "name": "whole_image_t1",
                "conditions": list(analyzer.BALANCED_CONDITIONS),
                "valid_for_t1": True,
                "valid_for_t2": False,
            }
        )

        def as_dict(self):
            return raw_contract

    class Runner(_FakeRunner):
        @staticmethod
        def select_mode_inputs(
            _release,
            *,
            mode,
            per_condition_limit,
            sample_id,
        ):
            assert mode == "smoke"
            assert per_condition_limit == 5
            assert sample_id is None
            return object(), rows

    monkeypatch.setattr(analyzer, "_runner", lambda: Runner)
    monkeypatch.setattr(
        analyzer,
        "load_canonical_release",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        analyzer,
        "build_run_dataset_contract",
        lambda *_args, **_kwargs: Contract(),
    )
    immutable = {
        "mode": "smoke",
        "dataset_contract": raw_contract,
        "score_spec": analyzer._score_spec().as_dict(),
        "selected_rows_sha256": analyzer._rows_sha256(rows),
        "selected_ids_sha256": analyzer.selected_ids_sha256(
            row["sample_id"] for row in rows
        ),
    }
    _release, selected, _contract = analyzer._rebuild_contract(
        repo_root=tmp_path,
        immutable=immutable,
        expected_mode="smoke",
    )
    assert len(selected) == 35

    rows.pop()
    with pytest.raises(ValueError, match="35 images"):
        analyzer._rebuild_contract(
            repo_root=tmp_path,
            immutable=immutable,
            expected_mode="smoke",
        )


def test_metrics_only_calls_shared_frozen_summary(monkeypatch):
    calls = {}

    def fake_summary(*args, **kwargs):
        calls["args"] = args
        calls["kwargs"] = kwargs
        return {
            "schema_version": analyzer.METRICS_SCHEMA_VERSION,
            "coverage": {"is_complete": True},
        }

    monkeypatch.setattr(analyzer, "summarize_balanced250_t1", fake_summary)
    release = SimpleNamespace(inputs=(1,), panel=(2,), source_pairs=(3,))
    contract = SimpleNamespace()
    bundle = SimpleNamespace(
        release=release,
        latest_results=({"ai_score": 1.0},),
        run_id="run",
        fingerprint="a" * 64,
        contract=contract,
    )
    result = analyzer.recompute_metrics(bundle)
    assert result["coverage"]["is_complete"] is True
    assert calls["kwargs"]["iterations"] == 1000
    assert calls["kwargs"]["seed"] == 20260726
    assert calls["kwargs"]["run_dataset_contract"] is contract
    with pytest.raises(ValueError, match="iterations=1000"):
        analyzer.recompute_metrics(bundle, iterations=999)


def _analysis_bundle_stub(tmp_path: Path):
    paths = {}
    for name, filename in (
        ("manifest", "manifest.json"),
        ("results", "results.jsonl"),
        ("expected", "expected_inputs.jsonl"),
        ("summary", "summary.json"),
    ):
        path = tmp_path / filename
        path.write_text(f"{name}\n", encoding="utf-8")
        paths[name] = path
    descriptor_dir = tmp_path / "raw_descriptors"
    descriptor_dir.mkdir()
    snapshot = {
        "manifest_sha256": hashlib.sha256(
            paths["manifest"].read_bytes()
        ).hexdigest(),
        "results_sha256": hashlib.sha256(
            paths["results"].read_bytes()
        ).hexdigest(),
        "expected_inputs_sha256": hashlib.sha256(
            paths["expected"].read_bytes()
        ).hexdigest(),
        "summary_sha256": hashlib.sha256(
            paths["summary"].read_bytes()
        ).hexdigest(),
        "descriptor_inventory_sha256": hashlib.sha256(
            stable_json([]).encode()
        ).hexdigest(),
    }
    return SimpleNamespace(
        run_id="formal",
        fingerprint="a" * 64,
        selected=({"sample_id": "sample"},),
        physical_results=({"sample_id": "sample"},),
        latest_results=({"sample_id": "sample"},),
        coverage={"is_complete": True},
        manifest_path=paths["manifest"],
        results_path=paths["results"],
        expected_path=paths["expected"],
        summary_path=paths["summary"],
        descriptor_dir=descriptor_dir,
        descriptors={},
        evidence_snapshot=snapshot,
    )


def test_skip_replay_validates_current_runtime_before_metrics(
    monkeypatch,
    tmp_path: Path,
):
    bundle = _analysis_bundle_stub(tmp_path)
    events = []
    monkeypatch.setattr(
        analyzer,
        "load_formal_run",
        lambda **_kwargs: bundle,
    )
    monkeypatch.setattr(
        analyzer,
        "_actual_runtime_contract",
        lambda device: events.append(("runtime", device)) or _runtime(),
    )

    def fake_metrics(*_args, **_kwargs):
        events.append(("metrics", None))
        return {
            "schema_version": analyzer.METRICS_SCHEMA_VERSION,
            "coverage": {"is_complete": True},
            "bootstrap": {"iterations": 1000},
        }

    monkeypatch.setattr(analyzer, "recompute_metrics", fake_metrics)
    monkeypatch.setattr(
        analyzer,
        "_verify_bundle_unchanged",
        lambda *_args, **_kwargs: events.append(("reverify", None)),
    )
    monkeypatch.setattr(
        analyzer,
        "replay_model",
        lambda *_args, **_kwargs: pytest.fail(
            "skip replay called replay_model"
        ),
    )
    report = analyzer.analyze(
        repo_root=tmp_path,
        results_dir=tmp_path,
        run_id="formal",
        source_root=Path("/source"),
        weights_dir=Path("/weights"),
        device_text="cuda:0",
        metrics_output_path=None,
        audit_output_path=None,
        replay=False,
    )
    assert events == [
        ("runtime", "cpu"),
        ("metrics", None),
        ("reverify", None),
    ]
    assert report["analysis_runtime"]["packages"]["scikit-learn"] == "1.8.0"
    assert report["fresh_model_replay"] is None


def test_analyze_rejects_metrics_mutation_during_replay(
    monkeypatch,
    tmp_path: Path,
):
    bundle = _analysis_bundle_stub(tmp_path)
    metrics_path = tmp_path / "balanced250_metrics.json"
    monkeypatch.setattr(
        analyzer,
        "load_formal_run",
        lambda **_kwargs: bundle,
    )
    monkeypatch.setattr(
        analyzer,
        "_actual_runtime_contract",
        lambda _device: _runtime(),
    )
    monkeypatch.setattr(
        analyzer,
        "recompute_metrics",
        lambda *_args, **_kwargs: {
            "schema_version": analyzer.METRICS_SCHEMA_VERSION,
            "coverage": {"is_complete": True},
            "bootstrap": {"iterations": 1000},
        },
    )

    def mutate_metrics(*_args, **_kwargs):
        metrics_path.write_text("{}\n", encoding="utf-8")
        return {"status": "replayed"}

    monkeypatch.setattr(analyzer, "replay_model", mutate_metrics)
    monkeypatch.setattr(
        analyzer,
        "_verify_bundle_unchanged",
        lambda *_args, **_kwargs: None,
    )
    with pytest.raises(ValueError, match="metrics output changed after write"):
        analyzer.analyze(
            repo_root=tmp_path,
            results_dir=tmp_path,
            run_id="formal",
            source_root=Path("/source"),
            weights_dir=Path("/weights"),
            device_text="cpu",
            metrics_output_path=metrics_path,
            audit_output_path=None,
            replay=True,
        )


def _runtime() -> dict:
    return {
        "device": "cpu",
        "python": {
            "implementation": "CPython",
            "version": "3.12.3",
            "executable": str(
                analyzer.EXPECTED_FROZEN_PYTHON_EXECUTABLE
            ),
        },
        "platform": "test-platform",
        "packages": {
            "torch": {
                "version": "2.10.0+cu128",
                "distribution_version": "2.10.0",
                "cuda_runtime": "12.8",
                "cudnn_version": 91002,
            },
            "numpy": "2.4.3",
            "Pillow": "12.1.1",
            "scipy": "1.17.1",
            "scikit-learn": "1.8.0",
        },
        "seed": analyzer.EXPECTED_RUNTIME_SEED,
        "descriptor_dtype": "float64",
        "batch_size": 1,
        "autocast": False,
        "deterministic_algorithms_enabled": True,
        "deterministic_algorithms_warn_only": False,
        "cublas_workspace_config": (
            analyzer.EXPECTED_CUBLAS_WORKSPACE_CONFIG
        ),
        "cudnn": {
            "enabled": True,
            "benchmark": False,
            "deterministic": True,
            "allow_tf32": False,
        },
        "matmul_allow_tf32": False,
        "float32_matmul_precision": "highest",
        "minimum_cuda_free_bytes": (
            analyzer.EXPECTED_MINIMUM_CUDA_FREE_BYTES
        ),
    }


def _preflight(source: dict, weights: dict) -> dict:
    return {
        "performed_before_accelerator_configuration": True,
        "report": {
            "schema_version": analyzer.EXPECTED_CPU_PREFLIGHT_SCHEMA,
            "status": "passed",
            "source": source,
            "weights": weights,
            "runtime": _runtime(),
            "golden": {
                "sample_id": analyzer.CPU_GOLDEN_SAMPLE_ID,
                "input_path": (
                    "outputs/opensource/balanced250_v1/images/"
                    f"{analyzer.CPU_GOLDEN_SAMPLE_ID}.jpg"
                ),
                "image_sha256": analyzer.CPU_GOLDEN_IMAGE_SHA256,
                "input_width": 1800,
                "input_height": 1350,
                "preprocess": analyzer.legacy.compute_preprocess_geometry(
                    1800,
                    1350,
                ),
                "descriptor_file_sha256": (
                    analyzer.CPU_GOLDEN_DESCRIPTOR_FILE_SHA256
                ),
                "descriptor_file_bytes": 7808,
                "descriptor_array_sha256": (
                    analyzer.CPU_GOLDEN_DESCRIPTOR_ARRAY_SHA256
                ),
                "descriptor_shape": [analyzer.DESCRIPTOR_DIMENSION],
                "descriptor_dtype": "float64",
                "descriptor_nbytes": analyzer.DESCRIPTOR_NBYTES,
                "raw_likelihood": analyzer.CPU_GOLDEN_RAW_LIKELIHOOD,
                "released_z_score": analyzer.CPU_GOLDEN_RELEASED_Z_SCORE,
                "ai_score": analyzer.CPU_GOLDEN_AI_SCORE,
                "classification_decision": False,
                "released_is_fake": False,
                "full_image_forward": True,
                "compute_fsd_calls": 1,
                "repeat_descriptor_file_sha256": (
                    analyzer.CPU_GOLDEN_DESCRIPTOR_FILE_SHA256
                ),
                "repeat_descriptor_file_bytes": 7808,
                "repeat_descriptor_array_sha256": (
                    analyzer.CPU_GOLDEN_DESCRIPTOR_ARRAY_SHA256
                ),
                "repeat_raw_likelihood": (
                    analyzer.CPU_GOLDEN_RAW_LIKELIHOOD
                ),
                "repeat_released_z_score": (
                    analyzer.CPU_GOLDEN_RELEASED_Z_SCORE
                ),
                "repeat_ai_score": analyzer.CPU_GOLDEN_AI_SCORE,
                "repeat_classification_decision": False,
                "repeat_released_is_fake": False,
                "repeat_full_image_forward": True,
                "repeat_compute_fsd_calls": 1,
                "repeat_byte_exact": True,
            },
            "cuda_used": False,
            "cuda_tensor_operations": False,
            "dataset_manifest_loaded": False,
        },
    }


def test_cpu_preflight_pins_exact_full_forward_and_repetition():
    source = {"root": "/source"}
    weights = {"weights_dir": "/weights"}
    value = _preflight(source, weights)
    analyzer._validate_cpu_preflight(
        value,
        source=source,
        weights=weights,
    )
    assert set(value["report"]) == {
        "schema_version",
        "status",
        "source",
        "weights",
        "runtime",
        "golden",
        "cuda_used",
        "cuda_tensor_operations",
        "dataset_manifest_loaded",
    }
    assert len(value["report"]["golden"]) == 30
    for field, replacement in (
        ("image_sha256", "0" * 64),
        ("descriptor_file_sha256", "1" * 64),
        ("descriptor_array_sha256", "2" * 64),
        ("raw_likelihood", analyzer.CPU_GOLDEN_RAW_LIKELIHOOD + 1e-12),
        ("repeat_byte_exact", False),
    ):
        broken = copy.deepcopy(value)
        broken["report"]["golden"][field] = replacement
        with pytest.raises(ValueError, match="CPU golden"):
            analyzer._validate_cpu_preflight(
                broken,
                source=source,
                weights=weights,
            )
    broken = copy.deepcopy(value)
    del broken["report"]["golden"]["repeat_ai_score"]
    with pytest.raises(ValueError, match="CPU golden key set"):
        analyzer._validate_cpu_preflight(
            broken,
            source=source,
            weights=weights,
        )


def test_metrics_runtime_pins_scikit_learn():
    runtime = _runtime()
    assert runtime["packages"]["scikit-learn"] == "1.8.0"
    assert (
        real_runner.FROZEN_RUNTIME_VERSIONS["scikit-learn"]
        == "1.8.0"
    )
    analyzer._validate_runtime_contract(runtime, label="metrics runtime")
    broken = copy.deepcopy(runtime)
    broken["packages"]["scikit-learn"] = "1.8.1"
    with pytest.raises(ValueError, match="frozen versions"):
        analyzer._validate_runtime_contract(
            broken,
            label="metrics runtime",
        )


def test_preprocess_contract_is_exact(monkeypatch):
    expected = copy.deepcopy(_FakeRunner.PREPROCESS_CONTRACT)
    assert analyzer._validate_preprocess_contract(expected) == expected
    broken = copy.deepcopy(expected)
    broken["resize"]["antialias"] = True
    with pytest.raises(ValueError, match="preprocess contract changed"):
        analyzer._validate_preprocess_contract(broken)
    broken = copy.deepcopy(expected)
    broken["extra"] = True
    with pytest.raises(ValueError, match="preprocess contract changed"):
        analyzer._validate_preprocess_contract(broken)


def test_nested_contracts_and_summary_keysets_are_exact():
    assert analyzer._validate_model_contract(
        copy.deepcopy(analyzer.EXPECTED_MODEL_CONTRACT)
    ) == analyzer.EXPECTED_MODEL_CONTRACT
    assert analyzer._validate_task_scope_contract(
        copy.deepcopy(analyzer.EXPECTED_TASK_SCOPE)
    ) == analyzer.EXPECTED_TASK_SCOPE
    assert analyzer._validate_artifact_contract(
        copy.deepcopy(analyzer.EXPECTED_ARTIFACT_CONTRACT)
    ) == analyzer.EXPECTED_ARTIFACT_CONTRACT

    broken_model = copy.deepcopy(analyzer.EXPECTED_MODEL_CONTRACT)
    broken_model["license"]["commercial_use"] = True
    with pytest.raises(ValueError, match="model exact contract"):
        analyzer._validate_model_contract(broken_model)
    broken_task = copy.deepcopy(analyzer.EXPECTED_TASK_SCOPE)
    broken_task["valid_for_t2"] = True
    with pytest.raises(ValueError, match="task_scope changed"):
        analyzer._validate_task_scope_contract(broken_task)
    broken_artifact = copy.deepcopy(analyzer.EXPECTED_ARTIFACT_CONTRACT)
    broken_artifact["descriptor"]["dtype"] = "float32"
    with pytest.raises(ValueError, match="descriptor contract changed"):
        analyzer._validate_artifact_contract(broken_artifact)

    dataset = {"binding": "exact"}
    coverage = {"is_complete": True}
    summary = {
        "schema_version": analyzer.EXPECTED_RUNTIME_SUMMARY_SCHEMA,
        "summary_kind": "runtime_coverage_only",
        "scientific_metrics": None,
        "scientific_metrics_owner": "analyze_fsd_balanced.py",
        "run_id": "run",
        "run_manifest_fingerprint": "a" * 64,
        "status": "complete",
        "mode": "formal",
        "model": analyzer.legacy.MODEL_NAME,
        "model_slug": analyzer.legacy.MODEL_SLUG,
        "score_spec": analyzer._score_spec().as_dict(),
        "dataset_contract": dataset,
        "coverage": coverage,
        "generated_at": "2026-07-26T00:00:00+00:00",
    }
    analyzer._validate_summary(
        summary=summary,
        bundle_mode="formal",
        run_id="run",
        fingerprint="a" * 64,
        contract=SimpleNamespace(as_dict=lambda: dataset),
        coverage=coverage,
    )
    broken_summary = {**summary, "unexpected_metric": 1.0}
    with pytest.raises(ValueError, match="summary key set changed"):
        analyzer._validate_summary(
            summary=broken_summary,
            bundle_mode="formal",
            run_id="run",
            fingerprint="a" * 64,
            contract=SimpleNamespace(as_dict=lambda: dataset),
            coverage=coverage,
        )


def test_adapter_source_keyset_hash_and_symlink_are_fail_closed(
    tmp_path: Path,
):
    records = {}
    for relative in _FakeRunner.ADAPTER_SOURCE_PATHS:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative, encoding="utf-8")
        records[relative] = {
            "path": relative,
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    analyzer._verify_adapter_sources(records, repo_root=tmp_path)
    broken = copy.deepcopy(records)
    broken.pop(next(iter(broken)))
    with pytest.raises(ValueError, match="key set"):
        analyzer._verify_adapter_sources(broken, repo_root=tmp_path)
    broken = copy.deepcopy(records)
    key = next(iter(broken))
    broken[key]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        analyzer._verify_adapter_sources(broken, repo_root=tmp_path)

    key = next(iter(records))
    path = tmp_path / key
    target = tmp_path / "target.py"
    path.replace(target)
    path.symlink_to(target)
    with pytest.raises(ValueError, match="symlink"):
        analyzer._verify_adapter_sources(records, repo_root=tmp_path)


def test_output_collision_gate_protects_all_run_evidence(tmp_path: Path):
    evidence = [
        tmp_path / "manifest.json",
        tmp_path / "results.jsonl",
        tmp_path / "expected_inputs.jsonl",
        tmp_path / "summary.json",
    ]
    descriptors = tmp_path / "raw_descriptors"
    analyzer._validate_output_targets(
        {
            "metrics": tmp_path / "balanced250_metrics.json",
            "audit": tmp_path / "independent_audit.json",
            "comparison": tmp_path / "comparison.json",
        },
        protected_files=evidence,
        protected_dirs=[descriptors],
    )
    with pytest.raises(ValueError, match="report paths must be distinct"):
        analyzer._validate_output_targets(
            {
                "metrics": tmp_path / "same.json",
                "audit": tmp_path / "same.json",
            },
            protected_files=evidence,
            protected_dirs=[descriptors],
        )
    for target in (*evidence, descriptors / "sample.npy"):
        with pytest.raises(ValueError, match="overwrite run evidence"):
            analyzer._validate_output_targets(
                {"metrics": target},
                protected_files=evidence,
                protected_dirs=[descriptors],
            )


def test_evidence_snapshot_rejects_mutation_during_and_after_validation(
    tmp_path: Path,
):
    paths = {}
    for name in ("manifest", "results", "expected", "summary"):
        path = tmp_path / name
        path.write_bytes(name.encode())
        paths[name] = path
    primary = {
        "manifest_sha256": hashlib.sha256(
            paths["manifest"].read_bytes()
        ).hexdigest(),
        "results_sha256": hashlib.sha256(
            paths["results"].read_bytes()
        ).hexdigest(),
        "expected_inputs_sha256": hashlib.sha256(
            paths["expected"].read_bytes()
        ).hexdigest(),
        "summary_sha256": hashlib.sha256(
            paths["summary"].read_bytes()
        ).hexdigest(),
    }
    paths["results"].write_bytes(b"changed-during-validation")
    with pytest.raises(ValueError, match="changed while"):
        analyzer._capture_evidence_snapshot(
            manifest_path=paths["manifest"],
            results_path=paths["results"],
            expected_path=paths["expected"],
            summary_path=paths["summary"],
            descriptors={},
            primary_snapshot=primary,
        )

    paths["results"].write_bytes(b"results")
    snapshot = analyzer._capture_evidence_snapshot(
        manifest_path=paths["manifest"],
        results_path=paths["results"],
        expected_path=paths["expected"],
        summary_path=paths["summary"],
        descriptors={},
        primary_snapshot=primary,
    )
    bundle = SimpleNamespace(
        evidence_snapshot=snapshot,
        manifest_path=paths["manifest"],
        results_path=paths["results"],
        expected_path=paths["expected"],
        summary_path=paths["summary"],
    )
    paths["summary"].write_bytes(b"changed-after-validation")
    with pytest.raises(ValueError, match="changed after validation"):
        analyzer._verify_bundle_unchanged(bundle, repo_root=tmp_path)


def _replay_bundle(tmp_path: Path, count: int = 2):
    selected = []
    rows = []
    for index in range(count):
        sample_id = f"sample-{index}"
        image_relative = Path("inputs") / f"{sample_id}.jpg"
        image_path = tmp_path / image_relative
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image_path.write_bytes(f"image-{index}".encode())
        selected.append(
            {
                "sample_id": sample_id,
                "canonical_path": image_relative.as_posix(),
                "canonical_sha256": hashlib.sha256(
                    image_path.read_bytes()
                ).hexdigest(),
                "width": 32,
                "height": 32,
            }
        )
        row = _row(tmp_path, sample_id=sample_id, run_id="formal")
        rows.append(row)
    descriptors = analyzer.validate_descriptor_inventory(
        latest_results=rows,
        repo_root=tmp_path,
        descriptor_dir=(
            tmp_path
            / "outputs"
            / "opensource"
            / "fsd"
            / "formal"
            / "raw_descriptors"
        ),
    )
    source = {"commit": analyzer.legacy.MODEL_SOURCE_COMMIT}
    weights = {"bundle_sha256": "d" * 64}
    runtime = {"device": "cpu"}
    bundle = SimpleNamespace(
        selected=tuple(selected),
        latest_results=tuple(rows),
        descriptors=descriptors,
        immutable={
            "source": source,
            "weights": weights,
            "runtime": runtime,
        },
        release=SimpleNamespace(repo_root=tmp_path),
    )
    return bundle, source, weights, runtime


def test_fresh_replay_calls_full_image_for_every_input_and_reports_tolerance(
    monkeypatch,
    tmp_path: Path,
):
    bundle, source, weights, runtime = _replay_bundle(tmp_path)
    monkeypatch.setattr(analyzer, "FORMAL_IMAGES", 2)
    monkeypatch.setattr(
        analyzer,
        "_actual_runtime_contract",
        lambda _device: runtime,
    )
    calls = []

    def fake_load_detector(**_kwargs):
        return (
            object(),
            SimpleNamespace(type="cpu"),
            {"source": source, "weights": weights},
        )

    def fake_infer(_detector, _device, path):
        calls.append(path.name)
        index = int(path.stem.split("-")[-1])
        row = bundle.latest_results[index]
        processed = {
            "raw_likelihood": row["raw_likelihood"] + 5e-10,
            "released_z_score": row["released_z_score"] + 5e-13,
            "ai_score": row["ai_score"] + 5e-13,
            "classification_decision": row["classification_decision"],
            "released_is_fake": row["released_is_fake"],
            "manual_replay": row["manual_replay"],
        }
        return processed, bundle.descriptors[path.stem].array.copy(), None, 1.0

    monkeypatch.setattr(analyzer.legacy, "load_detector", fake_load_detector)
    monkeypatch.setattr(analyzer.legacy, "infer_one", fake_infer)
    report = analyzer.replay_model(
        bundle,
        source_root=Path("/source"),
        weights_dir=Path("/weights"),
        device_text="cpu",
    )
    assert calls == ["sample-0.jpg", "sample-1.jpg"]
    assert report["images_replayed"] == 2
    assert report["full_image_forward_per_input"] is True
    assert report["descriptor_tail_only_replay"] is False
    assert report["max_raw_likelihood_abs_difference"] <= 1e-9
    assert report["max_descriptor_abs_difference"] == 0.0


def test_fresh_replay_rejects_tolerance_descriptor_and_incomplete_coverage(
    monkeypatch,
    tmp_path: Path,
):
    bundle, source, weights, runtime = _replay_bundle(tmp_path, count=1)
    monkeypatch.setattr(analyzer, "FORMAL_IMAGES", 1)
    monkeypatch.setattr(
        analyzer,
        "_actual_runtime_contract",
        lambda _device: runtime,
    )
    monkeypatch.setattr(
        analyzer.legacy,
        "load_detector",
        lambda **_kwargs: (
            object(),
            SimpleNamespace(type="cpu"),
            {"source": source, "weights": weights},
        ),
    )
    row = bundle.latest_results[0]

    def bad_raw(*_args):
        processed = {
            "raw_likelihood": row["raw_likelihood"] + 2e-9,
            "released_z_score": row["released_z_score"],
            "ai_score": row["ai_score"],
            "classification_decision": row["classification_decision"],
            "released_is_fake": row["released_is_fake"],
            "manual_replay": row["manual_replay"],
        }
        return processed, bundle.descriptors["sample-0"].array.copy(), None, 1.0

    monkeypatch.setattr(analyzer.legacy, "infer_one", bad_raw)
    with pytest.raises(ValueError, match="raw likelihood replay mismatch"):
        analyzer.replay_model(
            bundle,
            source_root=Path("/source"),
            weights_dir=Path("/weights"),
            device_text="cpu",
        )

    def bad_descriptor(*_args):
        changed = bundle.descriptors["sample-0"].array.copy()
        changed[0] += 1e-15
        processed = {
            "raw_likelihood": row["raw_likelihood"],
            "released_z_score": row["released_z_score"],
            "ai_score": row["ai_score"],
            "classification_decision": row["classification_decision"],
            "released_is_fake": row["released_is_fake"],
            "manual_replay": row["manual_replay"],
        }
        return processed, changed, None, 1.0

    monkeypatch.setattr(analyzer.legacy, "infer_one", bad_descriptor)
    with pytest.raises(ValueError, match="fresh descriptor mismatch"):
        analyzer.replay_model(
            bundle,
            source_root=Path("/source"),
            weights_dir=Path("/weights"),
            device_text="cpu",
        )
    monkeypatch.setattr(analyzer, "FORMAL_IMAGES", 2)
    with pytest.raises(ValueError, match="1,775-image selection"):
        analyzer.replay_model(
            bundle,
            source_root=Path("/source"),
            weights_dir=Path("/weights"),
            device_text="cpu",
        )


def test_cli_defaults_and_smoke_comparison_are_separate(
    monkeypatch,
    tmp_path: Path,
    capsys,
):
    formal = {}

    def fake_analyze(**kwargs):
        formal.update(kwargs)
        return {"status": "artifact_audit_passed"}

    monkeypatch.setattr(analyzer, "analyze", fake_analyze)
    assert analyzer.main(
        [
            "--repo-root",
            str(tmp_path),
            "--results-dir",
            "results",
            "--run-id",
            "formal",
            "--skip-model-replay",
        ]
    ) == 0
    assert formal["metrics_output_path"].name == "balanced250_metrics.json"
    assert formal["audit_output_path"].name == "independent_audit.json"
    assert formal["replay"] is False
    assert json.loads(capsys.readouterr().out)["status"] == (
        "artifact_audit_passed"
    )

    compared = {}

    def fake_compare(**kwargs):
        compared.update(kwargs)
        return {"status": "deterministic_smoke_comparison_passed"}

    monkeypatch.setattr(analyzer, "compare_smoke_runs", fake_compare)
    assert analyzer.main(
        [
            "--repo-root",
            str(tmp_path),
            "--results-dir",
            "results",
            "--run-id",
            "smoke-a",
            "--compare-smoke-run-id",
            "smoke-b",
        ]
    ) == 0
    assert compared["output_path"].name == (
        "smoke-a__vs__smoke-b_comparison.json"
    )
    assert json.loads(capsys.readouterr().out)["status"] == (
        "deterministic_smoke_comparison_passed"
    )


def test_run_path_and_cli_reject_escape_before_dispatch(
    monkeypatch,
    tmp_path: Path,
):
    results = tmp_path / "results"
    results.mkdir()
    with pytest.raises(ValueError, match="run-id"):
        analyzer._resolve_run_dir(results, "../escape")
    outside = tmp_path / "outside"
    outside.mkdir()
    (results / "linked").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="escapes|symlink"):
        analyzer._resolve_run_dir(results, "linked")

    called = False

    def fake_compare(**_kwargs):
        nonlocal called
        called = True
        return {}

    monkeypatch.setattr(analyzer, "compare_smoke_runs", fake_compare)
    with pytest.raises(ValueError, match="run-id"):
        analyzer.main(
            [
                "--repo-root",
                str(tmp_path),
                "--results-dir",
                "results",
                "--run-id",
                "smoke-a",
                "--compare-smoke-run-id",
                "../escape",
            ]
        )
    assert called is False
