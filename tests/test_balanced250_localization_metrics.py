from __future__ import annotations

import copy
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from eval.opensource.balanced250_localization_metrics import (
    DEFAULT_THRESHOLD,
    DEFAULT_THRESHOLD_OPERATOR,
    LOCAL_CONDITIONS,
    NOT_APPLICABLE_CONDITIONS,
    SUMMARY_SCHEMA_VERSION,
    summarize_balanced250_t2,
)
from eval.opensource.balanced_run_contract import (
    CapabilityBinding,
    LedgerBinding,
    RunDatasetContract,
    ScoreSpec,
    SelectionBinding,
    build_result_identity,
    selected_ids_sha256,
)
from eval.opensource.canonical_release import (
    BALANCED_CONDITIONS,
    BALANCED_CONTRACT_SHA256,
    BALANCED_DATASET_ID,
    BALANCED_RELEASE_KIND,
    BALANCED_SCHEMA,
    LOCALIZATION_CONDITIONS,
)
from eval.opensource.common import sha256_file, stable_json


RUN_ID = "maskclip_balanced250_t2_test"
FINGERPRINT = "a" * 64

_SEMANTICS = {
    "real": ("real", "authentic", "real", 0, "all_zero"),
    "local_mouse": (
        "local_splice",
        "local_insertion",
        "forged",
        1,
        "exact_diff",
    ),
    "local_cat": (
        "local_splice",
        "local_insertion",
        "forged",
        1,
        "exact_diff",
    ),
    "local_trash_can": (
        "local_splice",
        "local_insertion",
        "forged",
        1,
        "exact_diff",
    ),
    "fullframe_mouse": (
        "full_frame_conditional_edit",
        "conditional_full_frame_edit",
        "forged",
        1,
        "not_applicable",
    ),
    "fullframe_cat": (
        "full_frame_conditional_edit",
        "conditional_full_frame_edit",
        "forged",
        1,
        "not_applicable",
    ),
    "fullframe_trash_can": (
        "full_frame_conditional_edit",
        "conditional_full_frame_edit",
        "forged",
        1,
        "not_applicable",
    ),
}


def _rows_sha256(rows: list[dict[str, Any]]) -> str:
    payload = "".join(f"{stable_json(row)}\n" for row in rows).encode()
    return hashlib.sha256(payload).hexdigest()


def _make_inputs(repo_root: Path) -> list[dict[str, Any]]:
    mask_dir = repo_root / "outputs/opensource/balanced250_v1/masks"
    mask_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for rank, condition in enumerate(BALANCED_CONDITIONS):
        sample_id = f"{rank + 1:024x}"
        family, scope, kind, label, gt_kind = _SEMANTICS[condition]
        row: dict[str, Any] = {
            "schema_version": BALANCED_SCHEMA,
            "dataset_id": BALANCED_DATASET_ID,
            "sample_id": sample_id,
            "rank": rank,
            "condition": condition,
            "condition_family": family,
            "manipulation_scope": scope,
            "normalized_task_id": f"task_{rank}",
            "task_id": f"task_{rank}",
            "kind": kind,
            "label": label,
            "domain": "lodging",
            "gt_mask_kind": gt_kind,
            "canonical_path": (
                "outputs/opensource/balanced250_v1/images/" f"{sample_id}.jpg"
            ),
            "canonical_sha256": f"{rank + 11:064x}",
            "canonical_bytes": 123,
            "width": 2,
            "height": 2,
            "source_content_cluster": (
                "1" * 64 if condition != "fullframe_trash_can" else "2" * 64
            ),
        }
        if gt_kind == "exact_diff":
            mask_path = mask_dir / f"{sample_id}.png"
            Image.fromarray(
                np.asarray([[255, 0], [0, 0]], dtype=np.uint8),
                mode="L",
            ).save(mask_path)
            row.update(
                {
                    "gt_mask_path": str(mask_path.relative_to(repo_root)),
                    "gt_mask_sha256": sha256_file(mask_path),
                    "gt_positive_pixels": 1,
                }
            )
        elif gt_kind == "all_zero":
            row.update(
                {
                    "gt_mask_path": None,
                    "gt_mask_sha256": None,
                    "gt_positive_pixels": 0,
                }
            )
        else:
            row.update(
                {
                    "gt_mask_path": None,
                    "gt_mask_sha256": None,
                    "gt_positive_pixels": None,
                }
            )
        rows.append(row)
    return rows


def _make_contract(
    inputs: list[dict[str, Any]],
    *,
    capability_name: str = "local_t1_t2",
) -> RunDatasetContract:
    if capability_name == "local_t1_t2":
        conditions = tuple(BALANCED_CONDITIONS)
        valid_for_t1 = True
        score_spec = ScoreSpec(
            key="ai_score",
            direction="higher_means_fake",
            fixed_threshold=0.5,
            threshold_operator=">=",
        )
    elif capability_name == "local_t2_only":
        conditions = tuple(LOCALIZATION_CONDITIONS)
        valid_for_t1 = False
        score_spec = None
    else:
        raise AssertionError(capability_name)
    selected = [row for row in inputs if row["condition"] in conditions]
    counts = Counter(str(row["condition"]) for row in selected)
    return RunDatasetContract(
        release_schema_version=BALANCED_SCHEMA,
        release_kind=BALANCED_RELEASE_KIND,
        dataset_id=BALANCED_DATASET_ID,
        dataset_manifest="outputs/opensource/balanced250_v1/manifest.json",
        dataset_manifest_sha256="b" * 64,
        dataset_contract_sha256=BALANCED_CONTRACT_SHA256,
        inputs_ledger=LedgerBinding(
            name="inputs",
            path="outputs/opensource/balanced250_v1/inputs.jsonl",
            sha256=_rows_sha256(inputs),
            rows=len(inputs),
        ),
        panel_ledger=LedgerBinding(
            name="panel",
            path="outputs/opensource/balanced250_v1/panel.jsonl",
            sha256="c" * 64,
            rows=7,
        ),
        source_pairs_ledger=LedgerBinding(
            name="source_pairs",
            path="outputs/opensource/balanced250_v1/source_pairs.jsonl",
            sha256="d" * 64,
            rows=6,
        ),
        capability=CapabilityBinding(
            name=capability_name,
            conditions=conditions,
            valid_for_t1=valid_for_t1,
            valid_for_t2=True,
        ),
        selection=SelectionBinding(
            capability=capability_name,
            conditions=None,
            per_condition_limit=None,
            sample_id=None,
            pair_limit=None,
            selected_images=len(selected),
            selected_ids_sha256=selected_ids_sha256(
                str(row["sample_id"]) for row in selected
            ),
            counts_by_condition=tuple(counts.items()),
        ),
        score_spec=score_spec,
    )


def _make_results(
    inputs: list[dict[str, Any]],
    contract: RunDatasetContract,
) -> list[dict[str, Any]]:
    allowed = set(contract.capability.conditions)
    rows = []
    for input_row in inputs:
        if input_row["condition"] not in allowed:
            continue
        rows.append(
            {
                **build_result_identity(
                    input_row,
                    run_id=RUN_ID,
                    run_manifest_fingerprint=FINGERPRINT,
                ),
                "status": "ok",
                "valid_for_metrics": True,
                **({"ai_score": 0.1} if contract.score_spec is not None else {}),
            }
        )
    return rows


def _maps(inputs: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    values = {
        "real": [[0.1, 0.5], [0.7, 0.2]],
        "local_mouse": [[0.9, 0.8], [0.2, 0.1]],
        "local_cat": [[0.5, 0.1], [0.2, 0.3]],
        "local_trash_can": [[0.4, 0.3], [0.2, 0.1]],
    }
    return {
        str(row["sample_id"]): np.asarray(
            values[str(row["condition"])],
            dtype=np.float32,
        )
        for row in inputs
        if row["condition"] in values
    }


def _summarize(
    tmp_path: Path,
    *,
    capability_name: str = "local_t1_t2",
    iterations: int = 25,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
    RunDatasetContract,
    dict[str, np.ndarray],
    list[str],
]:
    inputs = _make_inputs(tmp_path)
    contract = _make_contract(inputs, capability_name=capability_name)
    results = _make_results(inputs, contract)
    maps = _maps(inputs)
    calls: list[str] = []

    def loader(
        input_row: dict[str, Any],
        result_row: dict[str, Any],
    ) -> np.ndarray:
        assert input_row["sample_id"] == result_row["sample_id"]
        condition = str(input_row["condition"])
        if condition in NOT_APPLICABLE_CONDITIONS:
            raise AssertionError("full-frame map must not be loaded")
        calls.append(condition)
        return maps[str(input_row["sample_id"])]

    summary = summarize_balanced250_t2(
        inputs,
        results,
        repo_root=tmp_path,
        run_id=RUN_ID,
        run_manifest_fingerprint=FINGERPRINT,
        run_dataset_contract=contract,
        load_native_score_map=loader,
        score_map_name="maskclip_native_probability",
        iterations=iterations,
        seed=123,
    )
    return summary, inputs, results, contract, maps, calls


def test_t1_t2_summary_has_exact_native_metrics_and_exclusions(
    tmp_path: Path,
) -> None:
    summary, _, _, _, _, calls = _summarize(tmp_path)
    assert summary["schema_version"] == SUMMARY_SCHEMA_VERSION
    assert summary["localization_contract"]["threshold"] == DEFAULT_THRESHOLD
    assert (
        summary["localization_contract"]["threshold_operator"]
        == DEFAULT_THRESHOLD_OPERATOR
    )
    assert summary["coverage"] == {
        "release_inputs": 7,
        "selected_results": 7,
        "selected_condition_counts": {
            condition: 1 for condition in BALANCED_CONDITIONS
        },
        "all_results_successful": True,
        "duplicate_result_ids": 0,
        "missing_result_ids": [],
        "unexpected_result_ids": [],
        "native_maps_evaluated": 4,
        "all_zero_real_images": 1,
        "exact_diff_local_images": 3,
        "not_applicable_selected_images": 3,
        "is_complete": True,
        "join_key": "sample_id",
        "task_id_pair_inference": False,
    }
    assert calls == ["real", *LOCAL_CONDITIONS]
    assert summary["excluded_not_applicable"] == {
        "policy": "fullframe_dense_outputs_are_not_scored_for_t2",
        "gt_mask_kind": "not_applicable",
        "conditions": list(NOT_APPLICABLE_CONDITIONS),
        "selected_images": 3,
        "counts_by_condition": {
            condition: 1 for condition in NOT_APPLICABLE_CONDITIONS
        },
        "score_map_loader_calls": 0,
    }

    mouse = summary["local"]["by_condition"]["local_mouse"]["overall"]
    assert mouse["per_image_macro"]["pixel_ap"]["estimate"] == 1.0
    assert mouse["micro_at_threshold"]["confusion"] == {
        "tp": 1,
        "fp": 1,
        "fn": 0,
        "tn": 2,
    }
    assert mouse["micro_at_threshold"]["precision"]["estimate"] == 0.5
    assert mouse["micro_at_threshold"]["recall"]["estimate"] == 1.0
    assert mouse["micro_at_threshold"]["f1"]["estimate"] == pytest.approx(2 / 3)
    assert mouse["micro_at_threshold"]["iou"]["estimate"] == 0.5
    assert mouse["micro_at_threshold"]["mcc"]["estimate"] == pytest.approx(
        1 / np.sqrt(3)
    )


def test_threshold_is_inclusive_and_real_only_reports_false_positive_area(
    tmp_path: Path,
) -> None:
    summary, *_ = _summarize(tmp_path)
    cat = summary["local"]["by_condition"]["local_cat"]["overall"]
    assert cat["micro_at_threshold"]["confusion"] == {
        "tp": 1,
        "fp": 0,
        "fn": 0,
        "tn": 3,
    }
    real = summary["real_false_positive"]["overall"]
    assert real["pixel_ap"] is None
    assert real["false_positive_pixels"] == 2
    assert real["false_positive_pixels_per_image"]["estimate"] == 2.0
    assert real["false_positive_fraction_per_image"]["estimate"] == 0.5
    assert real["false_positive_fraction_micro"]["estimate"] == 0.5
    assert "precision" not in real
    assert "confusion" not in real


def test_undefined_precision_and_mcc_are_json_null(
    tmp_path: Path,
) -> None:
    summary, *_ = _summarize(tmp_path)
    trash = summary["local"]["by_condition"]["local_trash_can"]["overall"]
    assert trash["per_image_macro"]["precision"]["estimate"] is None
    assert trash["per_image_macro"]["precision"]["ci95_percentile"] is None
    assert trash["per_image_macro"]["mcc"]["estimate"] is None
    assert trash["micro_at_threshold"]["precision"]["estimate"] is None
    assert trash["micro_at_threshold"]["mcc"]["estimate"] is None
    json.dumps(summary, allow_nan=False)


def test_bootstrap_is_deterministic_and_shared_across_conditions(
    tmp_path: Path,
) -> None:
    first, *_ = _summarize(tmp_path, iterations=31)
    second, *_ = _summarize(tmp_path, iterations=31)
    assert first == second
    assert first["bootstrap"] == {
        "iterations": 31,
        "seed": 123,
        "unit": (
            "shared_source_content_cluster_poisson_across_real_and_" "local_conditions"
        ),
        "ci": "two_sided_95_percentile",
        "condition_macro_dependency": (
            "aligned_shared_cluster_weights_across_local_conditions"
        ),
    }
    seeds = {
        first["local"]["by_condition"][condition]["overall"]["bootstrap_seed"]
        for condition in LOCAL_CONDITIONS
    }
    assert len(seeds) == 1
    assert (
        first["local"]["all_conditions_macro"]["overall"]["aggregation"]
        == "unweighted_condition_macro"
    )


def test_t2_only_selection_uses_1025_analogue_and_no_fullframe_results(
    tmp_path: Path,
) -> None:
    summary, _, results, _, _, calls = _summarize(
        tmp_path,
        capability_name="local_t2_only",
    )
    assert len(results) == 4
    assert summary["coverage"]["selected_results"] == 4
    assert summary["coverage"]["not_applicable_selected_images"] == 0
    assert summary["excluded_not_applicable"]["selected_images"] == 0
    assert summary["excluded_not_applicable"]["counts_by_condition"] == {
        condition: 0 for condition in NOT_APPLICABLE_CONDITIONS
    }
    assert calls == ["real", *LOCAL_CONDITIONS]


@pytest.mark.parametrize(
    ("threshold", "operator"),
    [
        (0.5000001, ">="),
        (0.5, ">"),
        (True, ">="),
    ],
)
def test_frozen_threshold_drift_fails_closed(
    tmp_path: Path,
    threshold: Any,
    operator: str,
) -> None:
    inputs = _make_inputs(tmp_path)
    contract = _make_contract(inputs)
    results = _make_results(inputs, contract)
    maps = _maps(inputs)
    with pytest.raises(ValueError, match="threshold"):
        summarize_balanced250_t2(
            inputs,
            results,
            repo_root=tmp_path,
            run_id=RUN_ID,
            run_manifest_fingerprint=FINGERPRINT,
            run_dataset_contract=contract,
            load_native_score_map=lambda row, _: maps[str(row["sample_id"])],
            score_map_name="native",
            threshold=threshold,
            threshold_operator=operator,
            iterations=5,
        )


@pytest.mark.parametrize(
    ("replacement", "message"),
    [
        (np.zeros((2, 2), dtype=np.float64), "float32"),
        (np.zeros((1, 4), dtype=np.float32), "shape"),
        (
            np.asarray([[0.0, np.nan], [0.0, 0.0]], dtype=np.float32),
            "non-finite",
        ),
        (
            np.asarray([[0.0, 1.1], [0.0, 0.0]], dtype=np.float32),
            r"outside \[0, 1\]",
        ),
    ],
)
def test_invalid_native_map_fails_closed(
    tmp_path: Path,
    replacement: np.ndarray,
    message: str,
) -> None:
    inputs = _make_inputs(tmp_path)
    contract = _make_contract(inputs)
    results = _make_results(inputs, contract)
    maps = _maps(inputs)
    local_id = str(
        next(row for row in inputs if row["condition"] == "local_mouse")["sample_id"]
    )
    maps[local_id] = replacement
    with pytest.raises(ValueError, match=message):
        summarize_balanced250_t2(
            inputs,
            results,
            repo_root=tmp_path,
            run_id=RUN_ID,
            run_manifest_fingerprint=FINGERPRINT,
            run_dataset_contract=contract,
            load_native_score_map=lambda row, _: maps[str(row["sample_id"])],
            score_map_name="native",
            iterations=5,
        )


def test_result_pair_rank_and_duplicate_fail_closed(tmp_path: Path) -> None:
    inputs = _make_inputs(tmp_path)
    contract = _make_contract(inputs)
    results = _make_results(inputs, contract)
    maps = _maps(inputs)
    drifted = copy.deepcopy(results)
    drifted[0]["pair_rank"] = 0
    with pytest.raises(ValueError, match="pair_rank"):
        summarize_balanced250_t2(
            inputs,
            drifted,
            repo_root=tmp_path,
            run_id=RUN_ID,
            run_manifest_fingerprint=FINGERPRINT,
            run_dataset_contract=contract,
            load_native_score_map=lambda row, _: maps[str(row["sample_id"])],
            score_map_name="native",
            iterations=5,
        )
    with pytest.raises(ValueError, match="duplicate sample_id"):
        summarize_balanced250_t2(
            inputs,
            [*results, copy.deepcopy(results[0])],
            repo_root=tmp_path,
            run_id=RUN_ID,
            run_manifest_fingerprint=FINGERPRINT,
            run_dataset_contract=contract,
            load_native_score_map=lambda row, _: maps[str(row["sample_id"])],
            score_map_name="native",
            iterations=5,
        )


def test_contract_and_gt_drift_fail_closed(tmp_path: Path) -> None:
    inputs = _make_inputs(tmp_path)
    contract = _make_contract(inputs)
    results = _make_results(inputs, contract)
    maps = _maps(inputs)
    drifted_contract = copy.deepcopy(contract)
    object.__setattr__(
        drifted_contract.inputs_ledger,
        "sha256",
        "0" * 64,
    )
    with pytest.raises(ValueError, match="inputs ledger drifted"):
        summarize_balanced250_t2(
            inputs,
            results,
            repo_root=tmp_path,
            run_id=RUN_ID,
            run_manifest_fingerprint=FINGERPRINT,
            run_dataset_contract=drifted_contract,
            load_native_score_map=lambda row, _: maps[str(row["sample_id"])],
            score_map_name="native",
            iterations=5,
        )

    drifted_inputs = copy.deepcopy(inputs)
    local = next(row for row in drifted_inputs if row["condition"] == "local_mouse")
    local["gt_positive_pixels"] = 2
    drifted_contract = _make_contract(drifted_inputs)
    drifted_results = _make_results(drifted_inputs, drifted_contract)
    with pytest.raises(ValueError, match="positive-pixel count changed"):
        summarize_balanced250_t2(
            drifted_inputs,
            drifted_results,
            repo_root=tmp_path,
            run_id=RUN_ID,
            run_manifest_fingerprint=FINGERPRINT,
            run_dataset_contract=drifted_contract,
            load_native_score_map=lambda row, _: maps[str(row["sample_id"])],
            score_map_name="native",
            iterations=5,
        )
