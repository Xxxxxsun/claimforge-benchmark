"""Fail-closed native-resolution T2 metrics for Balanced250.

The localization design is deliberately separate from the two T1 designs:

* authentic ``real`` rows have an explicit all-zero target and contribute
  only false-positive area/fraction diagnostics;
* the three ``local_*`` conditions have verified native-resolution
  exact-difference masks and contribute pixel AP plus frozen-threshold
  localization metrics; and
* the three ``fullframe_*`` conditions have ``gt_mask_kind=not_applicable``
  and are never scored for T2.

The caller supplies a method-specific native score-map loader.  Maps are
loaded and reduced one image at a time, so this module never retains the
large raster artifacts after their per-image sufficient statistics have
been calculated.  Pairing is neither required nor inferred: all joins use
``sample_id`` and bootstrap dependence uses only the explicitly frozen
``source_content_cluster`` field.
"""

from __future__ import annotations

import hashlib
import math
import numbers
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score

from eval.opensource.balanced_run_contract import (
    RESULT_SCHEMA_VERSION,
    RunDatasetContract,
    ResultIdentityV2,
    selected_ids_sha256,
    validate_result_identity,
)
from eval.opensource.canonical_release import (
    BALANCED_CONDITIONS,
    BALANCED_CONTRACT_SHA256,
    BALANCED_DATASET_ID,
    BALANCED_RELEASE_KIND,
    BALANCED_SCHEMA,
    FULLFRAME_CONDITIONS,
    LOCALIZATION_CONDITIONS,
    LOCAL_FORGED_CONDITIONS,
    load_ground_truth,
)
from eval.opensource.common import stable_json


REAL_CONDITION = "real"
LOCAL_CONDITIONS = tuple(
    condition
    for condition in BALANCED_CONDITIONS
    if condition in LOCAL_FORGED_CONDITIONS
)
NOT_APPLICABLE_CONDITIONS = tuple(
    condition for condition in BALANCED_CONDITIONS if condition in FULLFRAME_CONDITIONS
)

SUMMARY_SCHEMA_VERSION = "balanced250_t2_summary_v1"
DEFAULT_THRESHOLD = 0.5
DEFAULT_THRESHOLD_OPERATOR = ">="
DEFAULT_BOOTSTRAP_ITERATIONS = 1000
DEFAULT_BOOTSTRAP_SEED = 20260726

NativeScoreMapLoader = Callable[
    [Mapping[str, Any], Mapping[str, Any]],
    np.ndarray,
]

_LOCAL_CAPABILITY_CONTRACTS = {
    "local_t1_t2": {
        "conditions": tuple(BALANCED_CONDITIONS),
        "valid_for_t1": True,
        "valid_for_t2": True,
        "score_spec_required": True,
    },
    "local_t2_only": {
        "conditions": tuple(LOCALIZATION_CONDITIONS),
        "valid_for_t1": False,
        "valid_for_t2": True,
        "score_spec_required": False,
    },
}

_INPUT_CONDITION_CONTRACT = {
    REAL_CONDITION: {
        "condition_family": "real",
        "manipulation_scope": "authentic",
        "kind": "real",
        "label": 0,
        "gt_mask_kind": "all_zero",
    },
    **{
        condition: {
            "condition_family": "local_splice",
            "manipulation_scope": "local_insertion",
            "kind": "forged",
            "label": 1,
            "gt_mask_kind": "exact_diff",
        }
        for condition in LOCAL_CONDITIONS
    },
    **{
        condition: {
            "condition_family": "full_frame_conditional_edit",
            "manipulation_scope": "conditional_full_frame_edit",
            "kind": "forged",
            "label": 1,
            "gt_mask_kind": "not_applicable",
        }
        for condition in NOT_APPLICABLE_CONDITIONS
    },
}

_MACRO_METRICS = (
    "pixel_ap",
    "precision",
    "recall",
    "f1",
    "iou",
    "mcc",
    "predicted_positive_fraction",
)
_MICRO_METRICS = ("precision", "recall", "f1", "iou", "mcc")


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} is not an object")
    return value


def _nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} is not a non-empty string")
    return value


def _sha256(value: Any, label: str) -> str:
    result = _nonempty_string(value, label)
    if len(result) != 64 or any(
        character not in "0123456789abcdef" for character in result
    ):
        raise ValueError(f"{label} is not a lowercase SHA-256")
    return result


def _positive_integer(value: Any, label: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value,
        (numbers.Integral, np.integer),
    ):
        raise ValueError(f"{label} is not an integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{label} is not positive")
    return result


def _validate_bootstrap_arguments(iterations: Any, seed: Any) -> tuple[int, int]:
    iterations_value = _positive_integer(iterations, "iterations")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(
        seed,
        (numbers.Integral, np.integer),
    ):
        raise ValueError("seed must be an integer")
    return iterations_value, int(seed)


def _rows_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    payload = "".join(f"{stable_json(row)}\n" for row in rows).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _child_seed(seed: int, *parts: str) -> int:
    payload = "\0".join((str(seed), *parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _safe_div(
    numerator: int | float,
    denominator: int | float,
) -> float | None:
    if not denominator:
        return None
    result = float(numerator) / float(denominator)
    if not math.isfinite(result):
        raise ValueError("metric division produced a non-finite value")
    return result


def _mcc(tp: int, fp: int, fn: int, tn: int) -> float | None:
    denominator_squared = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    if denominator_squared == 0:
        return None
    value = (tp * tn - fp * fn) / math.sqrt(float(denominator_squared))
    if not math.isfinite(value):
        raise ValueError("MCC is not finite")
    return float(value)


def _threshold_metrics(
    *,
    tp: int,
    fp: int,
    fn: int,
    tn: int,
) -> dict[str, float | None]:
    return {
        "precision": _safe_div(tp, tp + fp),
        "recall": _safe_div(tp, tp + fn),
        "f1": _safe_div(2 * tp, 2 * tp + fp + fn),
        "iou": _safe_div(tp, tp + fp + fn),
        "mcc": _mcc(tp, fp, fn, tn),
    }


def _percentile_ci(values: Sequence[float]) -> list[float]:
    vector = np.asarray(values, dtype=np.float64)
    if vector.ndim != 1 or not vector.size or not np.isfinite(vector).all():
        raise ValueError("bootstrap values are not a finite non-empty vector")
    return [
        float(np.percentile(vector, 2.5)),
        float(np.percentile(vector, 97.5)),
    ]


def _optional_estimate_ci(
    estimate: float | None,
    replicates: Sequence[float | None],
    *,
    defined_observations: int | None = None,
    total_observations: int | None = None,
) -> dict[str, Any]:
    finite = [
        float(value)
        for value in replicates
        if value is not None and math.isfinite(float(value))
    ]
    result: dict[str, Any] = {
        "estimate": None if estimate is None else float(estimate),
        "ci95_percentile": (
            _percentile_ci(finite)
            if estimate is not None and len(finite) == len(replicates)
            else None
        ),
        "bootstrap_defined_replicates": len(finite),
        "bootstrap_total_replicates": len(replicates),
    }
    if defined_observations is not None or total_observations is not None:
        if defined_observations is None or total_observations is None:
            raise ValueError("defined-observation accounting is incomplete")
        result.update(
            {
                "defined_images": int(defined_observations),
                "undefined_images": int(total_observations - defined_observations),
            }
        )
    return result


def _shared_poisson_cluster_plan(
    *,
    cluster_sets: Sequence[set[str]],
    iterations: int,
    seed: int,
) -> tuple[tuple[str, ...], np.ndarray]:
    """Return aligned Poisson(1) weights shared by every requested slice."""

    if not cluster_sets or any(not values for values in cluster_sets):
        raise ValueError("cluster bootstrap requirements are empty")
    cluster_names = tuple(sorted(set().union(*cluster_sets)))
    index = {cluster: offset for offset, cluster in enumerate(cluster_names)}
    requirements = [
        np.asarray([index[value] for value in sorted(values)], dtype=np.int64)
        for values in cluster_sets
    ]
    rng = np.random.default_rng(seed)
    accepted: list[np.ndarray] = []
    accepted_rows = 0
    while accepted_rows < iterations:
        remaining = iterations - accepted_rows
        candidate = rng.poisson(
            1.0,
            size=(max(32, remaining * 2), len(cluster_names)),
        ).astype(np.int64, copy=False)
        valid = np.ones(candidate.shape[0], dtype=bool)
        for columns in requirements:
            valid &= np.sum(candidate[:, columns], axis=1) > 0
        batch = candidate[valid][:remaining]
        if batch.size:
            accepted.append(batch)
            accepted_rows += batch.shape[0]
    return cluster_names, np.concatenate(accepted, axis=0)[:iterations]


def _validate_input_rows(
    inputs: Sequence[Mapping[str, Any]],
) -> tuple[
    tuple[Mapping[str, Any], ...],
    dict[str, Mapping[str, Any]],
    Counter[str],
    tuple[str, ...],
]:
    if not isinstance(inputs, Sequence) or isinstance(
        inputs,
        (str, bytes, bytearray),
    ):
        raise ValueError("inputs must be a sequence")
    materialized: list[Mapping[str, Any]] = []
    by_id: dict[str, Mapping[str, Any]] = {}
    counts: Counter[str] = Counter()
    domains: set[str] = set()
    ranks: set[int] = set()
    for index, raw in enumerate(inputs):
        row = _require_mapping(raw, f"input row {index}")
        if row.get("schema_version") != BALANCED_SCHEMA:
            raise ValueError(f"input row {index} has wrong schema_version")
        if row.get("dataset_id") != BALANCED_DATASET_ID:
            raise ValueError(f"input row {index} has wrong dataset_id")
        identity = ResultIdentityV2.from_input(
            row,
            run_id="localization-input-validation",
            run_manifest_fingerprint="0" * 64,
        )
        sample_id = identity.sample_id
        if sample_id in by_id:
            raise ValueError(f"inputs contains duplicate sample_id {sample_id}")
        if identity.rank in ranks:
            raise ValueError(f"inputs contains duplicate rank {identity.rank}")
        ranks.add(identity.rank)
        condition = identity.condition
        expected = _INPUT_CONDITION_CONTRACT[condition]
        for field, value in expected.items():
            if row.get(field) != value:
                raise ValueError(f"input {sample_id} condition/{field} mismatch")
        cluster = _nonempty_string(
            row.get("source_content_cluster"),
            f"input {sample_id} source_content_cluster",
        )
        del cluster
        if condition == REAL_CONDITION:
            if not (
                row.get("gt_mask_path") is None
                and row.get("gt_mask_sha256") is None
                and row.get("gt_positive_pixels") == 0
            ):
                raise ValueError(f"real input {sample_id} GT contract changed")
        elif condition in LOCAL_CONDITIONS:
            _nonempty_string(
                row.get("gt_mask_path"),
                f"input {sample_id} gt_mask_path",
            )
            _sha256(
                row.get("gt_mask_sha256"),
                f"input {sample_id} gt_mask_sha256",
            )
            positive = _positive_integer(
                row.get("gt_positive_pixels"),
                f"input {sample_id} gt_positive_pixels",
            )
            if positive >= identity.input_width * identity.input_height:
                raise ValueError(
                    f"input {sample_id} exact-difference mask is not binary-class"
                )
        else:
            if not (
                row.get("gt_mask_path") is None
                and row.get("gt_mask_sha256") is None
                and row.get("gt_positive_pixels") is None
            ):
                raise ValueError(
                    f"not-applicable input {sample_id} GT contract changed"
                )
        by_id[sample_id] = row
        counts[condition] += 1
        domains.add(identity.domain)
        materialized.append(row)
    if not materialized:
        raise ValueError("inputs is empty")
    if set(counts) != set(BALANCED_CONDITIONS):
        raise ValueError("inputs must contain all seven Balanced250 conditions")
    if ranks != set(range(len(materialized))):
        raise ValueError("input ranks are not contiguous from zero")
    return tuple(materialized), by_id, counts, tuple(sorted(domains))


def _validate_contract_and_select(
    *,
    inputs: tuple[Mapping[str, Any], ...],
    input_counts: Mapping[str, int],
    contract: RunDatasetContract,
) -> tuple[tuple[Mapping[str, Any], ...], str]:
    if not isinstance(contract, RunDatasetContract):
        raise ValueError("run_dataset_contract has the wrong type")
    if (
        contract.release_schema_version != BALANCED_SCHEMA
        or contract.release_kind != BALANCED_RELEASE_KIND
        or contract.dataset_id != BALANCED_DATASET_ID
        or contract.dataset_contract_sha256 != BALANCED_CONTRACT_SHA256
    ):
        raise ValueError("run dataset contract is not frozen Balanced250")
    if contract.inputs_ledger.name != "inputs":
        raise ValueError("run dataset contract inputs ledger name changed")
    if contract.inputs_ledger.rows != len(
        inputs
    ) or contract.inputs_ledger.sha256 != _rows_sha256(inputs):
        raise ValueError("run dataset contract inputs ledger drifted")

    capability = contract.capability
    expected_capability = _LOCAL_CAPABILITY_CONTRACTS.get(capability.name)
    if expected_capability is None:
        raise ValueError("run dataset contract is not T2-capable")
    if (
        tuple(capability.conditions) != expected_capability["conditions"]
        or capability.valid_for_t1 is not expected_capability["valid_for_t1"]
        or capability.valid_for_t2 is not expected_capability["valid_for_t2"]
    ):
        raise ValueError("run dataset contract T2 capability drifted")
    if expected_capability["score_spec_required"]:
        if contract.score_spec is None:
            raise ValueError("native T1+T2 contract has no T1 score spec")
    elif contract.score_spec is not None:
        raise ValueError("native T2-only contract must not expose a T1 score")

    selection = contract.selection
    if (
        selection.capability != capability.name
        or selection.conditions is not None
        or selection.per_condition_limit is not None
        or selection.sample_id is not None
        or selection.pair_limit is not None
    ):
        raise ValueError("T2 aggregation requires the unfiltered formal selection")
    allowed = set(capability.conditions)
    selected = tuple(row for row in inputs if row.get("condition") in allowed)
    selected_ids = [str(row["sample_id"]) for row in selected]
    selected_counts = Counter(str(row["condition"]) for row in selected)
    if (
        selection.selected_images != len(selected)
        or selection.selected_ids_sha256 != selected_ids_sha256(selected_ids)
        or dict(selection.counts_by_condition) != dict(selected_counts)
    ):
        raise ValueError("run dataset contract formal selection drifted")
    expected_selected_counts = {
        condition: input_counts[condition] for condition in capability.conditions
    }
    if dict(selected_counts) != expected_selected_counts:
        raise ValueError("formal selection condition coverage drifted")
    contract_digest = hashlib.sha256(
        stable_json(contract.as_dict()).encode("utf-8")
    ).hexdigest()
    return selected, contract_digest


def _validate_results(
    *,
    selected: Sequence[Mapping[str, Any]],
    results: Sequence[Mapping[str, Any]],
    run_id: str,
    run_manifest_fingerprint: str,
) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(results, Sequence) or isinstance(
        results,
        (str, bytes, bytearray),
    ):
        raise ValueError("results must be a sequence")
    expected_by_id = {
        str(row["sample_id"]): ResultIdentityV2.from_input(
            row,
            run_id=run_id,
            run_manifest_fingerprint=run_manifest_fingerprint,
        )
        for row in selected
    }
    result_by_id: dict[str, Mapping[str, Any]] = {}
    for index, raw in enumerate(results):
        row = _require_mapping(raw, f"result row {index}")
        sample_id = _nonempty_string(
            row.get("sample_id"),
            f"result row {index} sample_id",
        )
        if sample_id in result_by_id:
            raise ValueError(f"results contains duplicate sample_id {sample_id}")
        expected = expected_by_id.get(sample_id)
        if expected is None:
            raise ValueError(f"result row {index} has unexpected sample_id")
        validate_result_identity(row, expected, index=index)
        if row.get("schema_version") != RESULT_SCHEMA_VERSION:
            raise ValueError(f"result {sample_id} has wrong schema_version")
        if row.get("status") != "ok":
            raise ValueError(f"result {sample_id} is not status ok")
        if row.get("valid_for_metrics") is not True:
            raise ValueError(f"result {sample_id} is not valid_for_metrics")
        result_by_id[sample_id] = row
    missing = [
        sample_id for sample_id in expected_by_id if sample_id not in result_by_id
    ]
    if missing:
        raise ValueError(f"results is missing sample_id {missing[0]}")
    if len(result_by_id) != len(selected):
        raise ValueError("result coverage does not match formal selection")
    return tuple(result_by_id[str(row["sample_id"])] for row in selected)


def _validate_score_map(
    raw: Any,
    *,
    row: Mapping[str, Any],
) -> np.ndarray:
    sample_id = str(row["sample_id"])
    if not isinstance(raw, np.ndarray):
        raise ValueError(
            f"native score map for {sample_id} is not a NumPy array/memmap"
        )
    score_map = np.asarray(raw)
    expected_shape = (int(row["height"]), int(row["width"]))
    if score_map.ndim != 2 or score_map.shape != expected_shape:
        raise ValueError(
            f"native score map for {sample_id} has shape {score_map.shape}, "
            f"expected {expected_shape}"
        )
    if score_map.dtype != np.float32:
        raise ValueError(f"native score map for {sample_id} is not float32")
    if score_map.size == 0:
        raise ValueError(f"native score map for {sample_id} is empty")
    if not np.isfinite(score_map).all():
        raise ValueError(f"native score map for {sample_id} contains non-finite values")
    minimum = float(np.min(score_map))
    maximum = float(np.max(score_map))
    if minimum < 0.0 or maximum > 1.0:
        raise ValueError(f"native score map for {sample_id} falls outside [0, 1]")
    return score_map


def _local_observation(
    *,
    input_row: Mapping[str, Any],
    result_row: Mapping[str, Any],
    score_map_loader: NativeScoreMapLoader,
    repo_root: Path,
) -> dict[str, Any]:
    sample_id = str(input_row["sample_id"])
    scores = _validate_score_map(
        score_map_loader(input_row, result_row),
        row=input_row,
    )
    target_value = load_ground_truth(input_row, repo_root)
    if target_value is None:
        raise ValueError(f"local input {sample_id} has no T2 ground truth")
    target = np.asarray(target_value, dtype=bool)
    if target.shape != scores.shape:
        raise ValueError(f"native score/GT shape mismatch for {sample_id}")
    positives = int(np.count_nonzero(target))
    pixels = int(target.size)
    if (
        positives != int(input_row["gt_positive_pixels"])
        or positives <= 0
        or positives >= pixels
    ):
        raise ValueError(f"local input {sample_id} GT pixel count drifted")
    prediction = scores >= DEFAULT_THRESHOLD
    tp = int(np.count_nonzero(prediction & target))
    fp = int(np.count_nonzero(prediction & ~target))
    fn = int(np.count_nonzero(~prediction & target))
    tn = int(np.count_nonzero(~prediction & ~target))
    threshold_metrics = _threshold_metrics(tp=tp, fp=fp, fn=fn, tn=tn)
    pixel_ap = float(
        average_precision_score(
            target.reshape(-1),
            scores.reshape(-1),
        )
    )
    if not math.isfinite(pixel_ap):
        raise ValueError(f"pixel AP for {sample_id} is not finite")
    predicted_positive_pixels = tp + fp
    return {
        "sample_id": sample_id,
        "condition": str(input_row["condition"]),
        "domain": str(input_row["domain"]),
        "source_content_cluster": str(input_row["source_content_cluster"]),
        "pixels": pixels,
        "target_positive_pixels": positives,
        "predicted_positive_pixels": predicted_positive_pixels,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "pixel_ap": pixel_ap,
        **threshold_metrics,
        "predicted_positive_fraction": float(predicted_positive_pixels / pixels),
    }


def _real_observation(
    *,
    input_row: Mapping[str, Any],
    result_row: Mapping[str, Any],
    score_map_loader: NativeScoreMapLoader,
) -> dict[str, Any]:
    sample_id = str(input_row["sample_id"])
    if not (
        input_row.get("gt_mask_kind") == "all_zero"
        and input_row.get("gt_positive_pixels") == 0
        and input_row.get("gt_mask_path") is None
        and input_row.get("gt_mask_sha256") is None
    ):
        raise ValueError(f"real input {sample_id} all-zero GT contract drifted")
    scores = _validate_score_map(
        score_map_loader(input_row, result_row),
        row=input_row,
    )
    pixels = int(scores.size)
    false_positive_pixels = int(np.count_nonzero(scores >= DEFAULT_THRESHOLD))
    return {
        "sample_id": sample_id,
        "condition": REAL_CONDITION,
        "domain": str(input_row["domain"]),
        "source_content_cluster": str(input_row["source_content_cluster"]),
        "pixels": pixels,
        "false_positive_pixels": false_positive_pixels,
        "false_positive_fraction": float(false_positive_pixels / pixels),
    }


def _cluster_columns(
    observations: Sequence[Mapping[str, Any]],
    cluster_names: Sequence[str],
) -> np.ndarray:
    index = {cluster: offset for offset, cluster in enumerate(cluster_names)}
    try:
        return np.asarray(
            [
                index[str(observation["source_content_cluster"])]
                for observation in observations
            ],
            dtype=np.int64,
        )
    except KeyError as error:
        raise ValueError(
            "observation references an unplanned source cluster"
        ) from error


def _weighted_optional_mean(
    values: Sequence[float | None],
    weights: np.ndarray,
) -> float | None:
    vector = np.asarray(
        [0.0 if value is None else float(value) for value in values],
        dtype=np.float64,
    )
    defined = np.asarray([value is not None for value in values], dtype=bool)
    selected_weights = weights[defined]
    denominator = int(np.sum(selected_weights))
    if denominator == 0:
        return None
    value = float(np.sum(vector[defined] * selected_weights) / denominator)
    if not math.isfinite(value):
        raise ValueError("weighted macro metric is not finite")
    return value


def _local_point(
    observations: Sequence[Mapping[str, Any]],
    weights: np.ndarray,
) -> dict[str, Any]:
    if weights.shape != (len(observations),):
        raise ValueError("local observation weights have wrong shape")
    if not np.issubdtype(weights.dtype, np.integer) or np.any(weights < 0):
        raise ValueError("local observation weights are invalid")
    if int(np.sum(weights)) == 0:
        raise ValueError("local observation weights select no images")
    totals = {
        name: int(
            sum(
                int(observation[name]) * int(weight)
                for observation, weight in zip(
                    observations,
                    weights,
                    strict=True,
                )
            )
        )
        for name in (
            "pixels",
            "target_positive_pixels",
            "predicted_positive_pixels",
            "tp",
            "fp",
            "fn",
            "tn",
        )
    }
    macro = {
        metric: _weighted_optional_mean(
            [
                (
                    float(observation[metric])
                    if observation.get(metric) is not None
                    else None
                )
                for observation in observations
            ],
            weights,
        )
        for metric in _MACRO_METRICS
    }
    return {
        "images": int(np.sum(weights)),
        **totals,
        "macro": macro,
        "micro": _threshold_metrics(
            tp=totals["tp"],
            fp=totals["fp"],
            fn=totals["fn"],
            tn=totals["tn"],
        ),
    }


def _aggregate_local_slice(
    observations: Sequence[Mapping[str, Any]],
    *,
    shared_cluster_names: Sequence[str],
    shared_cluster_weights: np.ndarray,
    iterations: int,
    seed: int,
) -> tuple[dict[str, Any], dict[str, dict[str, np.ndarray]]]:
    if not observations:
        raise ValueError("local T2 slice is empty")
    columns = _cluster_columns(observations, shared_cluster_names)
    weights = np.asarray(shared_cluster_weights)
    if (
        weights.shape != (iterations, len(shared_cluster_names))
        or not np.issubdtype(weights.dtype, np.integer)
        or np.any(weights < 0)
    ):
        raise ValueError("shared T2 cluster weights are invalid")
    point = _local_point(
        observations,
        np.ones(len(observations), dtype=np.int64),
    )
    macro_replicates: dict[str, list[float | None]] = {
        metric: [] for metric in _MACRO_METRICS
    }
    micro_replicates: dict[str, list[float | None]] = {
        metric: [] for metric in _MICRO_METRICS
    }
    for iteration in range(iterations):
        replicate = _local_point(
            observations,
            weights[iteration, columns],
        )
        for metric in _MACRO_METRICS:
            macro_replicates[metric].append(replicate["macro"][metric])
        for metric in _MICRO_METRICS:
            micro_replicates[metric].append(replicate["micro"][metric])

    macro_public: dict[str, Any] = {}
    for metric in _MACRO_METRICS:
        defined = sum(
            observation.get(metric) is not None for observation in observations
        )
        macro_public[metric] = _optional_estimate_ci(
            point["macro"][metric],
            macro_replicates[metric],
            defined_observations=defined,
            total_observations=len(observations),
        )
    micro_public = {
        metric: _optional_estimate_ci(
            point["micro"][metric],
            micro_replicates[metric],
        )
        for metric in _MICRO_METRICS
    }
    public = {
        "images": len(observations),
        "pixels": point["pixels"],
        "target_positive_pixels": point["target_positive_pixels"],
        "target_positive_fraction": float(
            point["target_positive_pixels"] / point["pixels"]
        ),
        "predicted_positive_pixels": point["predicted_positive_pixels"],
        "predicted_positive_fraction_micro": float(
            point["predicted_positive_pixels"] / point["pixels"]
        ),
        "source_content_clusters": len(
            {str(observation["source_content_cluster"]) for observation in observations}
        ),
        "per_image_macro": macro_public,
        "micro_at_threshold": {
            "threshold": DEFAULT_THRESHOLD,
            "threshold_operator": DEFAULT_THRESHOLD_OPERATOR,
            "confusion": {
                "tp": point["tp"],
                "fp": point["fp"],
                "fn": point["fn"],
                "tn": point["tn"],
            },
            **micro_public,
        },
        "bootstrap_unit": "shared_source_content_cluster_poisson",
        "bootstrap_iterations": iterations,
        "bootstrap_seed": seed,
    }
    private = {
        "macro": {
            metric: np.asarray(
                [np.nan if value is None else float(value) for value in values],
                dtype=np.float64,
            )
            for metric, values in macro_replicates.items()
        },
        "micro": {
            metric: np.asarray(
                [np.nan if value is None else float(value) for value in values],
                dtype=np.float64,
            )
            for metric, values in micro_replicates.items()
        },
    }
    return public, private


def _real_point(
    observations: Sequence[Mapping[str, Any]],
    weights: np.ndarray,
) -> dict[str, Any]:
    if weights.shape != (len(observations),):
        raise ValueError("real observation weights have wrong shape")
    if not np.issubdtype(weights.dtype, np.integer) or np.any(weights < 0):
        raise ValueError("real observation weights are invalid")
    images = int(np.sum(weights))
    if images == 0:
        raise ValueError("real observation weights select no images")
    pixels = int(
        sum(
            int(row["pixels"]) * int(weight)
            for row, weight in zip(observations, weights, strict=True)
        )
    )
    false_positive_pixels = int(
        sum(
            int(row["false_positive_pixels"]) * int(weight)
            for row, weight in zip(observations, weights, strict=True)
        )
    )
    macro_area = float(
        sum(
            int(row["false_positive_pixels"]) * int(weight)
            for row, weight in zip(observations, weights, strict=True)
        )
        / images
    )
    macro_fraction = float(
        sum(
            float(row["false_positive_fraction"]) * int(weight)
            for row, weight in zip(observations, weights, strict=True)
        )
        / images
    )
    return {
        "images": images,
        "pixels": pixels,
        "false_positive_pixels": false_positive_pixels,
        "false_positive_pixels_per_image": macro_area,
        "false_positive_fraction_per_image": macro_fraction,
        "false_positive_fraction_micro": float(false_positive_pixels / pixels),
    }


def _aggregate_real_slice(
    observations: Sequence[Mapping[str, Any]],
    *,
    shared_cluster_names: Sequence[str],
    shared_cluster_weights: np.ndarray,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    if not observations:
        raise ValueError("real false-positive slice is empty")
    columns = _cluster_columns(observations, shared_cluster_names)
    weights = np.asarray(shared_cluster_weights)
    point = _real_point(
        observations,
        np.ones(len(observations), dtype=np.int64),
    )
    replicate_names = (
        "false_positive_pixels_per_image",
        "false_positive_fraction_per_image",
        "false_positive_fraction_micro",
    )
    replicates: dict[str, list[float]] = {name: [] for name in replicate_names}
    for iteration in range(iterations):
        replicate = _real_point(
            observations,
            weights[iteration, columns],
        )
        for name in replicate_names:
            replicates[name].append(float(replicate[name]))
    return {
        "images": len(observations),
        "pixels": point["pixels"],
        "pixel_ap": None,
        "false_positive_pixels": point["false_positive_pixels"],
        "false_positive_pixels_per_image": {
            "estimate": point["false_positive_pixels_per_image"],
            "ci95_percentile": _percentile_ci(
                replicates["false_positive_pixels_per_image"]
            ),
        },
        "false_positive_fraction_per_image": {
            "estimate": point["false_positive_fraction_per_image"],
            "ci95_percentile": _percentile_ci(
                replicates["false_positive_fraction_per_image"]
            ),
        },
        "false_positive_fraction_micro": {
            "estimate": point["false_positive_fraction_micro"],
            "ci95_percentile": _percentile_ci(
                replicates["false_positive_fraction_micro"]
            ),
        },
        "source_content_clusters": len(
            {str(observation["source_content_cluster"]) for observation in observations}
        ),
        "bootstrap_unit": "shared_source_content_cluster_poisson",
        "bootstrap_iterations": iterations,
        "bootstrap_seed": seed,
    }


def _macro_from_slices(
    slices: Mapping[
        str,
        tuple[Mapping[str, Any], Mapping[str, Mapping[str, np.ndarray]]],
    ],
    *,
    conditions: Sequence[str],
) -> dict[str, Any]:
    public: dict[str, Any] = {
        "aggregation": "unweighted_condition_macro",
        "conditions": list(conditions),
        "bootstrap_unit": (
            "condition_macro_with_shared_source_content_cluster_" "poisson_bootstrap"
        ),
        "per_image_macro": {},
        "micro_at_threshold": {
            "threshold": DEFAULT_THRESHOLD,
            "threshold_operator": DEFAULT_THRESHOLD_OPERATOR,
        },
    }
    for family, metrics in (
        ("per_image_macro", _MACRO_METRICS),
        ("micro_at_threshold", _MICRO_METRICS),
    ):
        private_family = "macro" if family == "per_image_macro" else "micro"
        for metric in metrics:
            estimates = [
                slices[condition][0][family][metric]["estimate"]
                for condition in conditions
            ]
            defined_estimates = [
                float(value) for value in estimates if value is not None
            ]
            estimate = (
                float(np.mean(defined_estimates))
                if len(defined_estimates) == len(conditions)
                else None
            )
            matrix = np.stack(
                [
                    slices[condition][1][private_family][metric]
                    for condition in conditions
                ]
            )
            valid_columns = np.isfinite(matrix).all(axis=0)
            replicate_values: list[float | None] = [
                (float(np.mean(matrix[:, index])) if valid_columns[index] else None)
                for index in range(matrix.shape[1])
            ]
            public[family][metric] = _optional_estimate_ci(
                estimate,
                replicate_values,
            )
    return public


def _reject_nonfinite_output(value: Any, label: str = "summary") -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            _reject_nonfinite_output(nested, f"{label}.{key}")
    elif isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        for index, nested in enumerate(value):
            _reject_nonfinite_output(nested, f"{label}[{index}]")
    elif isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        raise ValueError(f"{label} is not finite")


def summarize_balanced250_t2(
    inputs: Sequence[Mapping[str, Any]],
    results: Sequence[Mapping[str, Any]],
    *,
    repo_root: Path,
    run_id: str,
    run_manifest_fingerprint: str,
    run_dataset_contract: RunDatasetContract,
    load_native_score_map: NativeScoreMapLoader,
    score_map_name: str,
    threshold: float = DEFAULT_THRESHOLD,
    threshold_operator: str = DEFAULT_THRESHOLD_OPERATOR,
    iterations: int = DEFAULT_BOOTSTRAP_ITERATIONS,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Validate a complete T2-capable run and summarize native score maps.

    ``inputs`` is the complete 1,775-row Balanced250 release ledger, not a
    Mouse-style pair ledger.  The exact formal result selection is derived
    from ``run_dataset_contract``.  ``load_native_score_map`` is invoked once
    per selected real/local row and never for a full-frame row.  It must
    return a native ``float32`` NumPy array (a read-only memmap is accepted)
    with values in ``[0, 1]``.
    """

    root = Path(repo_root).resolve()
    if not root.is_dir():
        raise ValueError(f"repo_root is not a directory: {root}")
    run_id_value = _nonempty_string(run_id, "run_id")
    fingerprint_value = _sha256(
        run_manifest_fingerprint,
        "run_manifest_fingerprint",
    )
    score_map_name_value = _nonempty_string(
        score_map_name,
        "score_map_name",
    )
    if not callable(load_native_score_map):
        raise ValueError("load_native_score_map is not callable")
    if (
        isinstance(threshold, (bool, np.bool_))
        or not isinstance(
            threshold,
            (numbers.Real, np.integer, np.floating),
        )
        or float(threshold) != DEFAULT_THRESHOLD
    ):
        raise ValueError("Balanced250 T2 threshold is frozen at 0.5")
    if threshold_operator != DEFAULT_THRESHOLD_OPERATOR:
        raise ValueError("Balanced250 T2 threshold operator is frozen at >=")
    iterations_value, seed_value = _validate_bootstrap_arguments(
        iterations,
        seed,
    )

    materialized_inputs, input_by_id, input_counts, domains = _validate_input_rows(
        inputs
    )
    selected, contract_digest = _validate_contract_and_select(
        inputs=materialized_inputs,
        input_counts=input_counts,
        contract=run_dataset_contract,
    )
    validated_results = _validate_results(
        selected=selected,
        results=results,
        run_id=run_id_value,
        run_manifest_fingerprint=fingerprint_value,
    )
    result_by_id = {str(row["sample_id"]): row for row in validated_results}

    local_observations: dict[str, list[dict[str, Any]]] = {
        condition: [] for condition in LOCAL_CONDITIONS
    }
    real_observations: list[dict[str, Any]] = []
    map_loader_calls = 0
    for input_row in selected:
        sample_id = str(input_row["sample_id"])
        condition = str(input_row["condition"])
        result_row = result_by_id[sample_id]
        if condition == REAL_CONDITION:
            real_observations.append(
                _real_observation(
                    input_row=input_row,
                    result_row=result_row,
                    score_map_loader=load_native_score_map,
                )
            )
            map_loader_calls += 1
        elif condition in LOCAL_CONDITIONS:
            local_observations[condition].append(
                _local_observation(
                    input_row=input_row,
                    result_row=result_row,
                    score_map_loader=load_native_score_map,
                    repo_root=root,
                )
            )
            map_loader_calls += 1
        elif condition not in NOT_APPLICABLE_CONDITIONS:
            raise ValueError(f"selected input has unsupported condition {condition}")

    if not real_observations or any(
        not local_observations[condition] for condition in LOCAL_CONDITIONS
    ):
        raise ValueError("T2 formal selection lacks real/local coverage")
    expected_loader_calls = len(real_observations) + sum(
        len(local_observations[condition]) for condition in LOCAL_CONDITIONS
    )
    if map_loader_calls != expected_loader_calls:
        raise ValueError("native score-map loader accounting drifted")

    overall_seed = _child_seed(
        seed_value,
        "t2",
        "shared_clusters",
        "overall",
    )
    overall_cluster_sets = [
        {str(row["source_content_cluster"]) for row in real_observations},
        *[
            {
                str(row["source_content_cluster"])
                for row in local_observations[condition]
            }
            for condition in LOCAL_CONDITIONS
        ],
    ]
    overall_names, overall_weights = _shared_poisson_cluster_plan(
        cluster_sets=overall_cluster_sets,
        iterations=iterations_value,
        seed=overall_seed,
    )

    domain_plans: dict[str, tuple[tuple[str, ...], np.ndarray, int]] = {}
    for domain in domains:
        domain_seed = _child_seed(
            seed_value,
            "t2",
            "shared_clusters",
            domain,
        )
        cluster_sets = [
            {
                str(row["source_content_cluster"])
                for row in real_observations
                if row["domain"] == domain
            },
            *[
                {
                    str(row["source_content_cluster"])
                    for row in local_observations[condition]
                    if row["domain"] == domain
                }
                for condition in LOCAL_CONDITIONS
            ],
        ]
        domain_plans[domain] = (
            *_shared_poisson_cluster_plan(
                cluster_sets=cluster_sets,
                iterations=iterations_value,
                seed=domain_seed,
            ),
            domain_seed,
        )

    condition_public: dict[str, Any] = {}
    condition_private: dict[
        str,
        tuple[Mapping[str, Any], Mapping[str, Mapping[str, np.ndarray]]],
    ] = {}
    domain_private: dict[
        str,
        dict[
            str,
            tuple[
                Mapping[str, Any],
                Mapping[str, Mapping[str, np.ndarray]],
            ],
        ],
    ] = {}
    for condition in LOCAL_CONDITIONS:
        overall_slice = _aggregate_local_slice(
            local_observations[condition],
            shared_cluster_names=overall_names,
            shared_cluster_weights=overall_weights,
            iterations=iterations_value,
            seed=overall_seed,
        )
        condition_private[condition] = overall_slice
        domain_private[condition] = {}
        by_domain: dict[str, Any] = {}
        for domain in domains:
            rows = [
                row for row in local_observations[condition] if row["domain"] == domain
            ]
            sliced = _aggregate_local_slice(
                rows,
                shared_cluster_names=domain_plans[domain][0],
                shared_cluster_weights=domain_plans[domain][1],
                iterations=iterations_value,
                seed=domain_plans[domain][2],
            )
            domain_private[condition][domain] = sliced
            by_domain[domain] = sliced[0]
        condition_public[condition] = {
            "condition_family": "local_splice",
            "manipulation_scope": "local_insertion",
            "overall": overall_slice[0],
            "by_domain": by_domain,
        }

    all_local = [
        row for condition in LOCAL_CONDITIONS for row in local_observations[condition]
    ]
    all_conditions_pooled = _aggregate_local_slice(
        all_local,
        shared_cluster_names=overall_names,
        shared_cluster_weights=overall_weights,
        iterations=iterations_value,
        seed=overall_seed,
    )[0]
    all_conditions_macro = {
        "overall": _macro_from_slices(
            condition_private,
            conditions=LOCAL_CONDITIONS,
        ),
        "by_domain": {
            domain: _macro_from_slices(
                {
                    condition: domain_private[condition][domain]
                    for condition in LOCAL_CONDITIONS
                },
                conditions=LOCAL_CONDITIONS,
            )
            for domain in domains
        },
    }
    real_false_positive = {
        "overall": _aggregate_real_slice(
            real_observations,
            shared_cluster_names=overall_names,
            shared_cluster_weights=overall_weights,
            iterations=iterations_value,
            seed=overall_seed,
        ),
        "by_domain": {
            domain: _aggregate_real_slice(
                [row for row in real_observations if row["domain"] == domain],
                shared_cluster_names=domain_plans[domain][0],
                shared_cluster_weights=domain_plans[domain][1],
                iterations=iterations_value,
                seed=domain_plans[domain][2],
            )
            for domain in domains
        },
    }

    selected_counts = Counter(str(row["condition"]) for row in selected)
    not_applicable_counts = {
        condition: selected_counts.get(condition, 0)
        for condition in NOT_APPLICABLE_CONDITIONS
    }
    not_applicable_selected = sum(not_applicable_counts.values())
    result = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "dataset_schema_version": BALANCED_SCHEMA,
        "dataset_id": BALANCED_DATASET_ID,
        "run_id": run_id_value,
        "run_manifest_fingerprint": fingerprint_value,
        "run_dataset_contract_sha256": contract_digest,
        "localization_contract": {
            "task": "T2_native_localization",
            "score_map_name": score_map_name_value,
            "score_map_space": "native_decoded_pixels",
            "score_map_dtype": "float32",
            "score_map_range": [0.0, 1.0],
            "ground_truth": {
                "real": "all_zero",
                "local": "exact_diff",
                "fullframe": "not_applicable",
            },
            "pixel_ap": ("exact_average_precision_over_continuous_native_score_map"),
            "threshold": DEFAULT_THRESHOLD,
            "threshold_operator": DEFAULT_THRESHOLD_OPERATOR,
            "fullframe_t2": "not_applicable",
        },
        "bootstrap": {
            "iterations": iterations_value,
            "seed": seed_value,
            "unit": (
                "shared_source_content_cluster_poisson_across_real_and_"
                "local_conditions"
            ),
            "ci": "two_sided_95_percentile",
            "condition_macro_dependency": (
                "aligned_shared_cluster_weights_across_local_conditions"
            ),
        },
        "coverage": {
            "release_inputs": len(materialized_inputs),
            "selected_results": len(validated_results),
            "selected_condition_counts": dict(sorted(selected_counts.items())),
            "all_results_successful": True,
            "duplicate_result_ids": 0,
            "missing_result_ids": [],
            "unexpected_result_ids": [],
            "native_maps_evaluated": map_loader_calls,
            "all_zero_real_images": len(real_observations),
            "exact_diff_local_images": len(all_local),
            "not_applicable_selected_images": not_applicable_selected,
            "is_complete": True,
            "join_key": "sample_id",
            "task_id_pair_inference": False,
        },
        "local": {
            "design": "native_exact_difference_localization",
            "conditions": list(LOCAL_CONDITIONS),
            "domains": list(domains),
            "by_condition": condition_public,
            "all_conditions_pooled": all_conditions_pooled,
            "all_conditions_macro": all_conditions_macro,
        },
        "real_false_positive": {
            "design": "all_zero_native_target_false_positive_area_only",
            **real_false_positive,
        },
        "excluded_not_applicable": {
            "policy": "fullframe_dense_outputs_are_not_scored_for_t2",
            "gt_mask_kind": "not_applicable",
            "conditions": list(NOT_APPLICABLE_CONDITIONS),
            "selected_images": not_applicable_selected,
            "counts_by_condition": not_applicable_counts,
            "score_map_loader_calls": 0,
        },
    }
    _reject_nonfinite_output(result)
    return result


__all__ = [
    "DEFAULT_BOOTSTRAP_ITERATIONS",
    "DEFAULT_BOOTSTRAP_SEED",
    "DEFAULT_THRESHOLD",
    "DEFAULT_THRESHOLD_OPERATOR",
    "LOCAL_CONDITIONS",
    "NOT_APPLICABLE_CONDITIONS",
    "NativeScoreMapLoader",
    "REAL_CONDITION",
    "SUMMARY_SCHEMA_VERSION",
    "summarize_balanced250_t2",
]
