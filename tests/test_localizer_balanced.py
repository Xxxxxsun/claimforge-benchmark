from __future__ import annotations

from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from eval.opensource.canonical_release import load_canonical_release
from eval.opensource import localizer_balanced as runner
from eval.opensource.localizer_balanced import (
    FORMAL_COUNTS,
    SMOKE_COUNTS,
    get_spec,
    model_specs,
    select_mode,
)


@pytest.fixture(scope="module")
def balanced_release():
    root = Path(__file__).resolve().parents[1]
    manifest = root / "outputs/opensource/balanced250_v1/manifest.json"
    if not manifest.is_file():
        pytest.skip("materialized Balanced250 release is unavailable")
    return load_canonical_release(root, manifest, verify_files=False)


def test_final_three_model_registry_is_frozen():
    specs = model_specs()
    assert set(specs) == {"mesorch", "relayformer", "dinov3_iml"}
    assert specs["mesorch"].checkpoint_sha256 == (
        "6d8fcd7ce7616d819bec6a9ed461b27187101e67247f8b2d2483fdc1f25f685a"
    )
    assert specs["relayformer"].checkpoint_sha256 == (
        "00a0f145ae4a98e66cad95aa79d2ce470d77821ee4262d6b803b3705c11c2090"
    )
    assert specs["dinov3_iml"].checkpoint_sha256 == (
        "01f23401e048f706ea0e63fb0429ddef80db3197ac0f5707bd584a8b056177fa"
    )
    for spec in specs.values():
        assert spec.native_probability_key in spec.arrays
        assert spec.official_threshold_operator == ">"
        assert "_r2_" in spec.formal_run_id
        assert "_r2_" in spec.smoke_run_id_a
        assert "_r2_" in spec.smoke_run_id_b
        assert spec.formal_run_id.endswith("_20260727")


def test_formal_selection_is_exact_t2_only(balanced_release):
    selection, rows = select_mode(balanced_release, "formal")
    assert selection.capability.value == "local_t2_only"
    assert len(rows) == 1025
    assert Counter(row["condition"] for row in rows) == Counter(FORMAL_COUNTS)
    assert not any(row["condition"].startswith("fullframe_") for row in rows)


def test_smoke_selection_is_exact_panel_priority_5x4(balanced_release):
    selection, rows = select_mode(balanced_release, "smoke")
    assert selection.per_condition_limit == 5
    assert len(rows) == 20
    assert Counter(row["condition"] for row in rows) == Counter(SMOKE_COUNTS)
    assert all(row["panel"] is True for row in rows)


def test_single_selection_rejects_missing_id(balanced_release):
    with pytest.raises(ValueError, match="sample-id"):
        select_mode(balanced_release, "single")


def test_unknown_model_is_rejected():
    with pytest.raises(ValueError, match="unsupported localizer"):
        get_spec("not-a-model")


def test_analyzer_score_loader_uses_input_then_result_order(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    spec = get_spec("mesorch")
    input_row = {"sample_id": "sample", "height": 2, "width": 3}
    result_row = {
        "sample_id": "sample",
        "arrays": {spec.native_probability_key: {"path": "native.npy"}},
    }
    contract = SimpleNamespace(as_dict=lambda: {"contract": "frozen"})
    manifest = {
        "fingerprint": "a" * 64,
        "immutable": {
            "mode": "formal",
            "dataset_contract": contract.as_dict(),
            "outputs": {"artifact_root": "outputs/test"},
        },
    }
    release = SimpleNamespace(inputs=[input_row])

    monkeypatch.setattr(
        runner,
        "_run_bundle",
        lambda *args: (
            tmp_path,
            manifest,
            {"status": "complete"},
            [input_row],
            [result_row],
        ),
    )
    monkeypatch.setattr(
        runner, "load_canonical_release", lambda *args, **kwargs: release
    )
    monkeypatch.setattr(
        runner, "select_mode", lambda *args, **kwargs: (object(), [input_row])
    )
    monkeypatch.setattr(
        runner, "build_run_dataset_contract", lambda *args, **kwargs: contract
    )
    monkeypatch.setattr(
        runner,
        "_artifact_paths",
        lambda *args: (
            {spec.native_probability_key: tmp_path / "native.npy"},
            tmp_path,
        ),
    )
    expected_map = np.zeros((2, 3), dtype=np.float32)

    def validate(record, *, expected_path, expected_shape):
        assert record is result_row["arrays"][spec.native_probability_key]
        assert expected_path == tmp_path / "native.npy"
        assert expected_shape == (2, 3)
        return expected_map

    monkeypatch.setattr(runner, "_validate_artifact_record", validate)

    def summarize(inputs, results, *, load_native_score_map, **kwargs):
        assert inputs == [input_row]
        assert results == [result_row]
        assert load_native_score_map(input_row, result_row) is expected_map
        return {"schema_version": runner.METRICS_SCHEMA, "status": "pass"}

    monkeypatch.setattr(runner, "summarize_balanced250_t2", summarize)
    monkeypatch.setattr(runner, "atomic_write_json", lambda *args: None)

    metrics = runner.analyze_formal(
        spec,
        repo_root=tmp_path,
        dataset_manifest=tmp_path / "manifest.json",
        results_root=tmp_path,
        run_id=spec.formal_run_id,
        iterations=1,
    )
    assert metrics == {
        "schema_version": runner.METRICS_SCHEMA,
        "status": "pass",
        "model": spec.display_name,
        "model_slug": spec.model_slug,
        "checkpoint_sha256": spec.checkpoint_sha256,
        "official_mask_threshold_operator": ">",
        "shared_metric_threshold_operator": ">=",
    }
