"""Fail-closed T1 metrics for the Balanced250 canonical release.

The release has two deliberately different evaluation designs:

* the primary selection-unpaired panel compares one fixed real panel against
  each of six independently selected forged conditions, while confidence
  intervals preserve shared source-content clusters across labels; and
* the secondary source-matched design uses the explicit ``real_sample_id`` and
  ``forged_sample_id`` references recorded in ``source_pairs.jsonl``.

This module never infers a pair from ``task_id``.  In particular, repeated
``task_id`` values across forged conditions are valid and have no effect on
joins or metric grouping.
"""

from __future__ import annotations

import hashlib
import math
import numbers
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from eval.opensource.balanced_run_contract import (
    RESULT_SCHEMA_VERSION,
    RunDatasetContract,
    ScoreSpec,
    selected_ids_sha256,
)
from eval.opensource.common import stable_json


REAL_CONDITION = "real"
LOCAL_CONDITIONS = (
    "local_mouse",
    "local_cat",
    "local_trash_can",
)
FULLFRAME_CONDITIONS = (
    "fullframe_mouse",
    "fullframe_cat",
    "fullframe_trash_can",
)
FORGED_CONDITIONS = LOCAL_CONDITIONS + FULLFRAME_CONDITIONS
PANEL_CONDITIONS = (REAL_CONDITION,) + FORGED_CONDITIONS

TARGET_FPR = 0.05
FPR_QUANTILE_METHOD = "higher"
DEFAULT_SCORE_KEY = "ai_score"
DEFAULT_DIRECTION = "higher_is_forged"
DEFAULT_FIXED_THRESHOLD = 0.5
DEFAULT_BOOTSTRAP_SEED = 20260726
DEFAULT_BOOTSTRAP_ITERATIONS = 1000

_CONDITION_CONTRACT = {
    REAL_CONDITION: {
        "condition_family": "real",
        "manipulation_scope": "authentic",
        "kind": "real",
        "label": 0,
    },
    **{
        condition: {
            "condition_family": "local_splice",
            "manipulation_scope": "local_insertion",
            "kind": "forged",
            "label": 1,
        }
        for condition in LOCAL_CONDITIONS
    },
    **{
        condition: {
            "condition_family": "full_frame_conditional_edit",
            "manipulation_scope": "conditional_full_frame_edit",
            "kind": "forged",
            "label": 1,
        }
        for condition in FULLFRAME_CONDITIONS
    },
}

_REFERENCE_IDENTITY_FIELDS = (
    "condition",
    "condition_family",
    "manipulation_scope",
    "kind",
    "label",
    "domain",
    "dataset_id",
    "schema_version",
    "canonical_path",
    "canonical_sha256",
    "width",
    "height",
    "source_content_cluster",
    "normalized_task_id",
    "task_id",
)

_RESULT_IDENTITY_FIELDS = (
    "condition",
    "condition_family",
    "manipulation_scope",
    "kind",
    "label",
    "domain",
    "dataset_id",
    "normalized_task_id",
    "task_id",
    "gt_mask_kind",
    "rank",
)

_RESULT_INPUT_FIELD_MAP = {
    **{field: field for field in _RESULT_IDENTITY_FIELDS},
    "input_path": "canonical_path",
    "input_sha256": "canonical_sha256",
    "input_width": "width",
    "input_height": "height",
}

_UNPAIRED_BOOTSTRAP_METRICS = (
    "auroc",
    "average_precision",
    "tpr_at_fpr_5_percent",
    "tpr_at_fpr_5_percent_threshold",
    "tpr_at_fpr_5_percent_actual_fpr",
    "accuracy_at_fixed_threshold",
    "balanced_accuracy_at_fixed_threshold",
    "precision_at_fixed_threshold",
    "recall_at_fixed_threshold",
    "f1_at_fixed_threshold",
    "specificity_at_fixed_threshold",
    "fpr_at_fixed_threshold",
)

_DELTA_BOOTSTRAP_METRICS = (
    "mean_score_delta",
    "median_score_delta",
    "strict_matched_ranking_accuracy",
)

_HIGHER_DIRECTIONS = frozenset(
    {
        "higher",
        "higher_is_fake",
        "higher_is_forged",
        "higher_means_fake",
        "higher_means_forged",
    }
)
_LOWER_DIRECTIONS = frozenset(
    {
        "lower",
        "lower_is_fake",
        "lower_is_forged",
        "lower_means_fake",
        "lower_means_forged",
    }
)


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
    if (
        len(result) != 64
        or any(character not in "0123456789abcdef" for character in result)
    ):
        raise ValueError(f"{label} is not a lowercase SHA-256")
    return result


def _finite_real(value: Any, label: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value,
        (numbers.Real, np.integer, np.floating),
    ):
        raise ValueError(f"{label} is not a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} is not finite")
    return result


def _binary_label(value: Any, label: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value,
        (numbers.Integral, np.integer),
    ):
        raise ValueError(f"{label} is not an integer label")
    result = int(value)
    if result not in (0, 1):
        raise ValueError(f"{label} is not a binary label")
    return result


def _validate_bootstrap_arguments(iterations: Any, seed: Any) -> tuple[int, int]:
    if (
        isinstance(iterations, (bool, np.bool_))
        or not isinstance(iterations, (numbers.Integral, np.integer))
        or int(iterations) <= 0
    ):
        raise ValueError("iterations must be a positive integer")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(
        seed,
        (numbers.Integral, np.integer),
    ):
        raise ValueError("seed must be an integer")
    return int(iterations), int(seed)


def _direction_contract(direction: Any) -> tuple[str, float, str]:
    value = _nonempty_string(direction, "score direction")
    if value in _HIGHER_DIRECTIONS:
        return "higher_is_forged", 1.0, ">"
    if value in _LOWER_DIRECTIONS:
        return "lower_is_forged", -1.0, "<"
    raise ValueError(
        "score direction must declare higher_is_forged or lower_is_forged"
    )


def _fixed_threshold_operator(
    value: Any,
    *,
    direction_sign: float,
) -> str:
    if value is None:
        return ">" if direction_sign > 0 else "<"
    operator = _nonempty_string(value, "fixed threshold operator")
    allowed = (">", ">=") if direction_sign > 0 else ("<", "<=")
    if operator not in allowed:
        raise ValueError(
            "fixed threshold operator is inconsistent with score direction"
        )
    return operator


def _child_seed(seed: int, *parts: str) -> int:
    payload = "\0".join((str(seed), *parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _rows_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    payload = "".join(f"{stable_json(row)}\n" for row in rows).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _row_clusters(
    rows: Sequence[Mapping[str, Any]],
    *,
    label: str,
) -> tuple[str, ...]:
    return tuple(
        _nonempty_string(
            row.get("source_content_cluster"),
            f"{label} row {index} source_content_cluster",
        )
        for index, row in enumerate(rows)
    )


def _shared_poisson_cluster_plan(
    *,
    cluster_sets: Sequence[set[str]],
    iterations: int,
    seed: int,
) -> tuple[tuple[str, ...], np.ndarray]:
    """Create aligned Poisson(1) weights for shared source clusters.

    Rejected all-zero slices are redrawn so every requested label/condition
    remains represented in every bootstrap replicate.
    """

    if not cluster_sets or any(not values for values in cluster_sets):
        raise ValueError("cluster bootstrap requirements are empty")
    cluster_names = tuple(sorted(set().union(*cluster_sets)))
    index = {cluster: offset for offset, cluster in enumerate(cluster_names)}
    requirement_indices = [
        np.asarray([index[cluster] for cluster in sorted(values)], dtype=np.int64)
        for values in cluster_sets
    ]
    rng = np.random.default_rng(seed)
    accepted: list[np.ndarray] = []
    while sum(batch.shape[0] for batch in accepted) < iterations:
        remaining = iterations - sum(batch.shape[0] for batch in accepted)
        candidate = rng.poisson(
            1.0,
            size=(max(32, remaining * 2), len(cluster_names)),
        ).astype(np.int64, copy=False)
        valid = np.ones(candidate.shape[0], dtype=bool)
        for columns in requirement_indices:
            valid &= np.sum(candidate[:, columns], axis=1) > 0
        if np.any(valid):
            accepted.append(candidate[valid][:remaining])
    weights = np.concatenate(accepted, axis=0)[:iterations]
    return cluster_names, weights


def _index_by_sample_id(
    rows: Sequence[Mapping[str, Any]],
    *,
    label: str,
) -> dict[str, Mapping[str, Any]]:
    indexed: dict[str, Mapping[str, Any]] = {}
    for index, raw in enumerate(rows):
        row = _require_mapping(raw, f"{label} row {index}")
        sample_id = _nonempty_string(
            row.get("sample_id"),
            f"{label} row {index} sample_id",
        )
        if sample_id in indexed:
            raise ValueError(f"{label} contains duplicate sample_id {sample_id}")
        indexed[sample_id] = row
    if not indexed:
        raise ValueError(f"{label} is empty")
    return indexed


def _uniform_optional_field(
    rows: Sequence[Mapping[str, Any]],
    field: str,
    *,
    label: str,
) -> str | None:
    present = [field in row and row.get(field) is not None for row in rows]
    if not any(present):
        return None
    if not all(present):
        raise ValueError(f"{label} has partially missing {field}")
    values = {
        _nonempty_string(row.get(field), f"{label} {field}") for row in rows
    }
    if len(values) != 1:
        raise ValueError(f"{label} has inconsistent {field}")
    return next(iter(values))


def _validate_condition_identity(
    row: Mapping[str, Any],
    *,
    label: str,
) -> str:
    condition = _nonempty_string(row.get("condition"), f"{label} condition")
    contract = _CONDITION_CONTRACT.get(condition)
    if contract is None:
        raise ValueError(f"{label} has unsupported condition {condition!r}")
    kind = row.get("kind")
    if kind != contract["kind"]:
        raise ValueError(f"{label} condition/kind mismatch")
    if _binary_label(row.get("label"), f"{label} label") != contract["label"]:
        raise ValueError(f"{label} condition/label mismatch")
    for field in ("condition_family", "manipulation_scope"):
        if field in row and row.get(field) != contract[field]:
            raise ValueError(f"{label} condition/{field} mismatch")
    _nonempty_string(row.get("domain"), f"{label} domain")
    return condition


def _validate_optional_release_identity(
    row: Mapping[str, Any],
    *,
    schema_version: str | None,
    dataset_id: str | None,
    label: str,
) -> None:
    for field, expected in (
        ("schema_version", schema_version),
        ("dataset_id", dataset_id),
    ):
        if expected is not None and row.get(field) != expected:
            raise ValueError(f"{label} {field} mismatch")


def _validate_reference_identity(
    reference: Mapping[str, Any],
    canonical: Mapping[str, Any],
    *,
    label: str,
) -> None:
    for field in _REFERENCE_IDENTITY_FIELDS:
        if field in reference and reference.get(field) != canonical.get(field):
            raise ValueError(f"{label} {field} does not match inputs")
    if "input_rank" in reference and "rank" in canonical:
        if reference.get("input_rank") != canonical.get("rank"):
            raise ValueError(f"{label} input_rank does not match inputs rank")


def _validate_inputs(
    inputs: Sequence[Mapping[str, Any]],
) -> tuple[
    dict[str, Mapping[str, Any]],
    str | None,
    str | None,
    Counter[str],
]:
    by_id = _index_by_sample_id(inputs, label="inputs")
    condition_counts: Counter[str] = Counter()
    for sample_id, row in by_id.items():
        condition = _validate_condition_identity(
            row,
            label=f"input {sample_id}",
        )
        condition_counts[condition] += 1
    if set(condition_counts) != set(PANEL_CONDITIONS):
        raise ValueError(
            "inputs must contain real and all six forged conditions"
        )
    schema_version = _uniform_optional_field(
        inputs,
        "schema_version",
        label="inputs",
    )
    dataset_id = _uniform_optional_field(
        inputs,
        "dataset_id",
        label="inputs",
    )
    return by_id, schema_version, dataset_id, condition_counts


def _validate_panel(
    panel: Sequence[Mapping[str, Any]],
    *,
    input_by_id: Mapping[str, Mapping[str, Any]],
    schema_version: str | None,
    dataset_id: str | None,
) -> tuple[
    dict[str, Mapping[str, Any]],
    dict[str, list[Mapping[str, Any]]],
    tuple[str, ...],
    int,
]:
    panel_by_id = _index_by_sample_id(panel, label="panel")
    by_condition: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for sample_id, row in panel_by_id.items():
        canonical = input_by_id.get(sample_id)
        if canonical is None:
            raise ValueError(f"panel references unknown sample_id {sample_id}")
        _validate_optional_release_identity(
            row,
            schema_version=schema_version,
            dataset_id=dataset_id,
            label=f"panel {sample_id}",
        )
        _validate_reference_identity(
            row,
            canonical,
            label=f"panel {sample_id}",
        )
        condition = _validate_condition_identity(
            row,
            label=f"panel {sample_id}",
        )
        by_condition[condition].append(row)
    if set(by_condition) != set(PANEL_CONDITIONS):
        raise ValueError("panel must contain exactly the seven release conditions")
    counts = {condition: len(by_condition[condition]) for condition in PANEL_CONDITIONS}
    panel_size = counts[REAL_CONDITION]
    if panel_size <= 0 or any(value != panel_size for value in counts.values()):
        raise ValueError(
            "panel must have the same non-zero row count in every condition"
        )

    real_domains = {
        _nonempty_string(row.get("domain"), "panel real domain")
        for row in by_condition[REAL_CONDITION]
    }
    if not real_domains:
        raise ValueError("panel real condition has no domains")
    for condition in FORGED_CONDITIONS:
        domains = {
            _nonempty_string(
                row.get("domain"),
                f"panel {condition} domain",
            )
            for row in by_condition[condition]
        }
        if domains != real_domains:
            raise ValueError(
                f"panel {condition} domains do not match the real panel"
            )
    return (
        panel_by_id,
        dict(by_condition),
        tuple(sorted(real_domains)),
        panel_size,
    )


def _validate_source_pairs(
    source_pairs: Sequence[Mapping[str, Any]],
    *,
    input_by_id: Mapping[str, Mapping[str, Any]],
    panel_by_condition: Mapping[str, Sequence[Mapping[str, Any]]],
    schema_version: str | None,
    dataset_id: str | None,
    rows_per_condition: int,
) -> list[dict[str, Any]]:
    if not source_pairs:
        raise ValueError("source_pairs is empty")
    seen_pair_ids: set[str] = set()
    seen_endpoint_pairs: set[tuple[str, str, str]] = set()
    seen_forged_ids: set[str] = set()
    seen_real_by_condition: dict[str, set[str]] = defaultdict(set)
    ranks_by_condition: dict[str, set[int]] = defaultdict(set)
    counts: Counter[str] = Counter()
    validated: list[dict[str, Any]] = []

    for index, raw in enumerate(source_pairs):
        row = _require_mapping(raw, f"source pair row {index}")
        condition = _nonempty_string(
            row.get("condition"),
            f"source pair row {index} condition",
        )
        if condition not in FORGED_CONDITIONS:
            raise ValueError(
                f"source pair row {index} has invalid condition {condition!r}"
            )
        _validate_optional_release_identity(
            row,
            schema_version=schema_version,
            dataset_id=dataset_id,
            label=f"source pair row {index}",
        )
        pair_id_value = row.get("pair_id")
        if pair_id_value is not None:
            pair_id = _nonempty_string(
                pair_id_value,
                f"source pair row {index} pair_id",
            )
            if pair_id in seen_pair_ids:
                raise ValueError(
                    f"source_pairs contains duplicate pair_id {pair_id}"
                )
            seen_pair_ids.add(pair_id)

        real_id = _nonempty_string(
            row.get("real_sample_id"),
            f"source pair row {index} real_sample_id",
        )
        forged_id = _nonempty_string(
            row.get("forged_sample_id"),
            f"source pair row {index} forged_sample_id",
        )
        if real_id == forged_id:
            raise ValueError(f"source pair row {index} reuses one endpoint")
        real = input_by_id.get(real_id)
        forged = input_by_id.get(forged_id)
        if real is None:
            raise ValueError(
                f"source pair row {index} references unknown real_sample_id "
                f"{real_id}"
            )
        if forged is None:
            raise ValueError(
                f"source pair row {index} references unknown forged_sample_id "
                f"{forged_id}"
            )
        if (
            real.get("condition") != REAL_CONDITION
            or real.get("kind") != "real"
            or real.get("label") != 0
        ):
            raise ValueError(f"source pair row {index} real endpoint is not real")
        if (
            forged.get("condition") != condition
            or forged.get("kind") != "forged"
            or forged.get("label") != 1
        ):
            raise ValueError(
                f"source pair row {index} forged endpoint/condition mismatch"
            )
        endpoint_key = (condition, real_id, forged_id)
        if endpoint_key in seen_endpoint_pairs:
            raise ValueError(
                f"source_pairs repeats endpoint pair for {condition}"
            )
        seen_endpoint_pairs.add(endpoint_key)
        if forged_id in seen_forged_ids:
            raise ValueError(
                f"source_pairs repeats forged_sample_id {forged_id}"
            )
        seen_forged_ids.add(forged_id)
        if real_id in seen_real_by_condition[condition]:
            raise ValueError(
                f"source_pairs repeats real_sample_id {real_id} in {condition}"
            )
        seen_real_by_condition[condition].add(real_id)

        real_domain = _nonempty_string(
            real.get("domain"),
            f"source pair row {index} real domain",
        )
        forged_domain = _nonempty_string(
            forged.get("domain"),
            f"source pair row {index} forged domain",
        )
        if real_domain != forged_domain:
            raise ValueError(
                f"source pair row {index} endpoint domains do not match"
            )
        if "domain" in row and row.get("domain") != real_domain:
            raise ValueError(f"source pair row {index} domain mismatch")

        cluster = _nonempty_string(
            row.get("source_content_cluster"),
            f"source pair row {index} source_content_cluster",
        )
        for endpoint_label, endpoint in (("real", real), ("forged", forged)):
            if (
                "source_content_cluster" in endpoint
                and endpoint.get("source_content_cluster") != cluster
            ):
                raise ValueError(
                    f"source pair row {index} {endpoint_label} cluster mismatch"
                )
        if "normalized_task_id" in row:
            normalized = row.get("normalized_task_id")
            for endpoint_label, endpoint in (("real", real), ("forged", forged)):
                if (
                    "normalized_task_id" in endpoint
                    and endpoint.get("normalized_task_id") != normalized
                ):
                    raise ValueError(
                        f"source pair row {index} {endpoint_label} normalized "
                        "task mismatch"
                    )

        if "condition_pair_rank" in row:
            value = row.get("condition_pair_rank")
            if isinstance(value, bool) or not isinstance(
                value,
                (numbers.Integral, np.integer),
            ):
                raise ValueError(
                    f"source pair row {index} has invalid condition_pair_rank"
                )
            rank = int(value)
            if rank < 0 or rank in ranks_by_condition[condition]:
                raise ValueError(
                    f"source_pairs has duplicate/negative rank in {condition}"
                )
            ranks_by_condition[condition].add(rank)

        counts[condition] += 1
        validated.append(
            {
                "condition": condition,
                "real_sample_id": real_id,
                "forged_sample_id": forged_id,
                "source_content_cluster": cluster,
                "domain": real_domain,
            }
        )

    if set(counts) != set(FORGED_CONDITIONS):
        raise ValueError("source_pairs must contain all six forged conditions")
    for condition in FORGED_CONDITIONS:
        if counts[condition] != rows_per_condition:
            raise ValueError(
                f"source_pairs {condition} count does not match panel"
            )
        ranks = ranks_by_condition.get(condition)
        if ranks and ranks != set(range(rows_per_condition)):
            raise ValueError(
                f"source_pairs {condition} ranks are not contiguous"
            )
        panel_forged_ids = {
            str(row["sample_id"]) for row in panel_by_condition[condition]
        }
        paired_forged_ids = {
            row["forged_sample_id"]
            for row in validated
            if row["condition"] == condition
        }
        if paired_forged_ids != panel_forged_ids:
            raise ValueError(
                f"source_pairs {condition} forged coverage does not match panel"
            )
    return validated


def _validate_run_dataset_contract(
    contract: RunDatasetContract,
    *,
    inputs: Sequence[Mapping[str, Any]],
    panel: Sequence[Mapping[str, Any]],
    source_pairs: Sequence[Mapping[str, Any]],
    schema_version: str | None,
    dataset_id: str | None,
    input_counts: Mapping[str, int],
) -> tuple[ScoreSpec, str]:
    if not isinstance(contract, RunDatasetContract):
        raise ValueError("run_dataset_contract has the wrong type")
    if contract.release_schema_version != schema_version:
        raise ValueError("run dataset contract release schema mismatch")
    if contract.dataset_id != dataset_id:
        raise ValueError("run dataset contract dataset_id mismatch")
    for name, binding, rows in (
        ("inputs", contract.inputs_ledger, inputs),
        ("panel", contract.panel_ledger, panel),
        ("source_pairs", contract.source_pairs_ledger, source_pairs),
    ):
        if binding.name != name:
            raise ValueError(f"run dataset contract {name} ledger name mismatch")
        if binding.rows != len(rows):
            raise ValueError(f"run dataset contract {name} row count mismatch")
        if binding.sha256 != _rows_sha256(rows):
            raise ValueError(f"run dataset contract {name} SHA-256 mismatch")

    capability = contract.capability
    if (
        not capability.valid_for_t1
        or set(capability.conditions) != set(PANEL_CONDITIONS)
    ):
        raise ValueError(
            "run dataset contract is not a seven-condition T1 capability"
        )
    selection = contract.selection
    if (
        selection.conditions is not None
        or selection.per_condition_limit is not None
        or selection.sample_id is not None
        or selection.pair_limit is not None
    ):
        raise ValueError("T1 aggregation requires the unfiltered formal selection")
    ordered_ids = [
        _nonempty_string(row.get("sample_id"), "input sample_id")
        for row in inputs
    ]
    if (
        selection.selected_images != len(inputs)
        or selection.selected_ids_sha256 != selected_ids_sha256(ordered_ids)
        or dict(selection.counts_by_condition) != dict(input_counts)
    ):
        raise ValueError("run dataset contract formal selection mismatch")
    if contract.score_spec is None:
        raise ValueError("run dataset contract has no T1 score spec")
    digest = hashlib.sha256(
        stable_json(contract.as_dict()).encode("utf-8")
    ).hexdigest()
    return contract.score_spec, digest


def _validate_results(
    results: Sequence[Mapping[str, Any]],
    *,
    input_by_id: Mapping[str, Mapping[str, Any]],
    score_key: str,
    run_id: str,
    run_manifest_fingerprint: str,
) -> dict[str, float]:
    result_by_id = _index_by_sample_id(results, label="results")
    expected_ids = set(input_by_id)
    actual_ids = set(result_by_id)
    missing = sorted(expected_ids - actual_ids)
    extra = sorted(actual_ids - expected_ids)
    if missing or extra:
        raise ValueError(
            "result coverage mismatch: "
            f"missing={missing[:3]}, extra={extra[:3]}"
        )

    scores: dict[str, float] = {}
    for sample_id, result in result_by_id.items():
        canonical = input_by_id[sample_id]
        if "pair_rank" in result:
            raise ValueError(f"result {sample_id} must not contain pair_rank")
        if result.get("schema_version") != RESULT_SCHEMA_VERSION:
            raise ValueError(f"result {sample_id} has wrong schema_version")
        if result.get("id") != sample_id:
            raise ValueError(f"result {sample_id} id/sample_id mismatch")
        if result.get("run_id") != run_id:
            raise ValueError(f"result {sample_id} run_id mismatch")
        if (
            result.get("run_manifest_fingerprint")
            != run_manifest_fingerprint
        ):
            raise ValueError(
                f"result {sample_id} run_manifest_fingerprint mismatch"
            )
        if result.get("status") != "ok":
            raise ValueError(f"result {sample_id} is not status ok")
        if result.get("valid_for_metrics") is not True:
            raise ValueError(f"result {sample_id} is not valid_for_metrics")
        for result_field, input_field in _RESULT_INPUT_FIELD_MAP.items():
            if input_field not in canonical:
                raise ValueError(
                    f"input {sample_id} lacks required identity field "
                    f"{input_field}"
                )
            if result_field not in result:
                raise ValueError(
                    f"result {sample_id} lacks identity field {result_field}"
                )
            if result.get(result_field) != canonical.get(input_field):
                raise ValueError(
                    f"result {sample_id} {result_field} mismatch"
                )
        if score_key not in result:
            raise ValueError(f"result {sample_id} lacks score_key {score_key}")
        scores[sample_id] = _finite_real(
            result.get(score_key),
            f"result {sample_id} {score_key}",
        )
    return scores


def _safe_div(numerator: int | float, denominator: int | float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def _point_metrics(
    real_scores: np.ndarray,
    forged_scores: np.ndarray,
    *,
    direction_sign: float,
    fixed_threshold: float,
    fixed_threshold_operator: str,
) -> dict[str, Any]:
    real = np.asarray(real_scores, dtype=np.float64)
    forged = np.asarray(forged_scores, dtype=np.float64)
    if real.ndim != 1 or forged.ndim != 1:
        raise ValueError("score arrays must be one-dimensional")
    if not real.size or not forged.size:
        raise ValueError("metrics require non-empty real and forged scores")
    if not np.isfinite(real).all() or not np.isfinite(forged).all():
        raise ValueError("metrics received non-finite scores")

    oriented_real = direction_sign * real
    oriented_forged = direction_sign * forged
    labels = np.concatenate(
        (
            np.zeros(real.size, dtype=np.int64),
            np.ones(forged.size, dtype=np.int64),
        )
    )
    oriented_scores = np.concatenate((oriented_real, oriented_forged))

    if fixed_threshold_operator == ">":
        real_predictions = real > fixed_threshold
        forged_predictions = forged > fixed_threshold
    elif fixed_threshold_operator == ">=":
        real_predictions = real >= fixed_threshold
        forged_predictions = forged >= fixed_threshold
    elif fixed_threshold_operator == "<":
        real_predictions = real < fixed_threshold
        forged_predictions = forged < fixed_threshold
    elif fixed_threshold_operator == "<=":
        real_predictions = real <= fixed_threshold
        forged_predictions = forged <= fixed_threshold
    else:
        raise ValueError("unsupported fixed threshold operator")
    tn = int(np.count_nonzero(~real_predictions))
    fp = int(np.count_nonzero(real_predictions))
    tp = int(np.count_nonzero(forged_predictions))
    fn = int(np.count_nonzero(~forged_predictions))
    recall = _safe_div(tp, tp + fn)
    specificity = _safe_div(tn, tn + fp)
    precision = _safe_div(tp, tp + fp)
    accuracy = _safe_div(tp + tn, real.size + forged.size)
    f1 = _safe_div(2 * tp, 2 * tp + fp + fn)

    fpr_threshold_oriented = float(
        np.quantile(
            oriented_real,
            1.0 - TARGET_FPR,
            method=FPR_QUANTILE_METHOD,
        )
    )
    fpr_threshold_raw = direction_sign * fpr_threshold_oriented
    tpr_at_fpr = float(
        np.mean(oriented_forged > fpr_threshold_oriented)
    )
    actual_fpr = float(np.mean(oriented_real > fpr_threshold_oriented))
    return {
        "auroc": float(roc_auc_score(labels, oriented_scores)),
        "average_precision": float(
            average_precision_score(labels, oriented_scores)
        ),
        "tpr_at_fpr_5_percent": tpr_at_fpr,
        "tpr_at_fpr_5_percent_threshold": fpr_threshold_raw,
        "tpr_at_fpr_5_percent_actual_fpr": actual_fpr,
        "accuracy_at_fixed_threshold": accuracy,
        "balanced_accuracy_at_fixed_threshold": (
            recall + specificity
        )
        / 2.0,
        "precision_at_fixed_threshold": precision,
        "recall_at_fixed_threshold": recall,
        "f1_at_fixed_threshold": f1,
        "specificity_at_fixed_threshold": specificity,
        "fpr_at_fixed_threshold": _safe_div(fp, fp + tn),
        "fixed_threshold_confusion": {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        },
    }


def _percentile_ci(values: Sequence[float] | np.ndarray) -> list[float]:
    vector = np.asarray(values, dtype=np.float64)
    if vector.ndim != 1 or not vector.size or not np.isfinite(vector).all():
        raise ValueError("bootstrap values are not a finite non-empty vector")
    return [
        float(np.percentile(vector, 2.5)),
        float(np.percentile(vector, 97.5)),
    ]


def _unpaired_slice(
    real_scores: np.ndarray,
    forged_scores: np.ndarray,
    *,
    real_clusters: Sequence[str],
    forged_clusters: Sequence[str],
    shared_cluster_names: Sequence[str],
    shared_cluster_weights: np.ndarray,
    score_key: str,
    direction: str,
    direction_sign: float,
    fixed_threshold_operator: str,
    fixed_threshold: float,
    iterations: int,
    seed: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    real = np.asarray(real_scores, dtype=np.float64)
    forged = np.asarray(forged_scores, dtype=np.float64)
    real_cluster_values = tuple(real_clusters)
    forged_cluster_values = tuple(forged_clusters)
    if len(real_cluster_values) != real.size:
        raise ValueError("real score/source-cluster count mismatch")
    if len(forged_cluster_values) != forged.size:
        raise ValueError("forged score/source-cluster count mismatch")
    cluster_index = {
        cluster: index for index, cluster in enumerate(shared_cluster_names)
    }
    try:
        real_columns = np.asarray(
            [cluster_index[cluster] for cluster in real_cluster_values],
            dtype=np.int64,
        )
        forged_columns = np.asarray(
            [cluster_index[cluster] for cluster in forged_cluster_values],
            dtype=np.int64,
        )
    except KeyError as exc:
        raise ValueError("score slice references an unplanned source cluster") from exc
    weights = np.asarray(shared_cluster_weights)
    if (
        weights.shape != (iterations, len(shared_cluster_names))
        or not np.issubdtype(weights.dtype, np.integer)
        or np.any(weights < 0)
    ):
        raise ValueError("shared source-cluster bootstrap weights are invalid")
    point = _point_metrics(
        real,
        forged,
        direction_sign=direction_sign,
        fixed_threshold=fixed_threshold,
        fixed_threshold_operator=fixed_threshold_operator,
    )
    replicates: dict[str, list[float]] = {
        metric: [] for metric in _UNPAIRED_BOOTSTRAP_METRICS
    }
    for iteration in range(iterations):
        sampled_real = np.repeat(
            real,
            weights[iteration, real_columns],
        )
        sampled_forged = np.repeat(
            forged,
            weights[iteration, forged_columns],
        )
        if not sampled_real.size or not sampled_forged.size:
            raise ValueError("cluster bootstrap produced an empty label slice")
        replicate = _point_metrics(
            sampled_real,
            sampled_forged,
            direction_sign=direction_sign,
            fixed_threshold=fixed_threshold,
            fixed_threshold_operator=fixed_threshold_operator,
        )
        for metric in _UNPAIRED_BOOTSTRAP_METRICS:
            replicates[metric].append(float(replicate[metric]))
    arrays = {
        metric: np.asarray(values, dtype=np.float64)
        for metric, values in replicates.items()
    }
    public: dict[str, Any] = {
        "real_images": int(real.size),
        "forged_images": int(forged.size),
        "images": int(real.size + forged.size),
        "score_key": score_key,
        "score_direction": direction,
        "fixed_threshold": fixed_threshold,
        "fixed_threshold_operator": fixed_threshold_operator,
        "fixed_threshold_confusion": point["fixed_threshold_confusion"],
        "real_source_content_clusters": len(set(real_cluster_values)),
        "forged_source_content_clusters": len(set(forged_cluster_values)),
        "union_source_content_clusters": len(
            set(real_cluster_values) | set(forged_cluster_values)
        ),
        "tpr_at_fpr_5_percent_threshold_source": (
            "real_scores_quantile_0_95_method_higher_after_direction_"
            "normalization"
        ),
        "tpr_at_fpr_5_percent_threshold_operator": (
            ">" if direction_sign > 0 else "<"
        ),
        "bootstrap_unit": (
            "shared_source_content_cluster_poisson_across_labels"
        ),
        "point_estimate_pairing": "none",
        "bootstrap_stratified_by_label": False,
        "bootstrap_preserves_cross_label_cluster_dependence": True,
        "bootstrap_iterations": iterations,
        "bootstrap_seed": seed,
    }
    for metric in _UNPAIRED_BOOTSTRAP_METRICS:
        public[metric] = {
            "estimate": float(point[metric]),
            "ci95_percentile": _percentile_ci(arrays[metric]),
        }
    return public, arrays


def _macro_slice(
    condition_slices: Mapping[
        str,
        tuple[Mapping[str, Any], Mapping[str, np.ndarray]],
    ],
    *,
    conditions: Sequence[str],
    score_key: str,
    direction: str,
    fixed_threshold_operator: str,
    fixed_threshold: float,
) -> dict[str, Any]:
    if not conditions:
        raise ValueError("macro slice has no conditions")
    public: dict[str, Any] = {
        "aggregation": "unweighted_condition_macro",
        "conditions": list(conditions),
        "score_key": score_key,
        "score_direction": direction,
        "fixed_threshold": fixed_threshold,
        "fixed_threshold_operator": fixed_threshold_operator,
        "bootstrap_unit": (
            "condition_macro_with_shared_source_content_cluster_"
            "poisson_bootstrap"
        ),
    }
    for metric in _UNPAIRED_BOOTSTRAP_METRICS:
        estimates = [
            float(condition_slices[condition][0][metric]["estimate"])
            for condition in conditions
        ]
        replicate_matrix = np.stack(
            [
                np.asarray(
                    condition_slices[condition][1][metric],
                    dtype=np.float64,
                )
                for condition in conditions
            ]
        )
        macro_replicates = np.mean(replicate_matrix, axis=0)
        public[metric] = {
            "estimate": float(np.mean(estimates)),
            "ci95_percentile": _percentile_ci(macro_replicates),
        }
    return public


def _scores_for_rows(
    rows: Sequence[Mapping[str, Any]],
    scores: Mapping[str, float],
) -> np.ndarray:
    return np.asarray(
        [scores[str(row["sample_id"])] for row in rows],
        dtype=np.float64,
    )


def _summarize_primary(
    *,
    panel_by_condition: Mapping[str, Sequence[Mapping[str, Any]]],
    domains: Sequence[str],
    scores: Mapping[str, float],
    score_key: str,
    direction: str,
    direction_sign: float,
    fixed_threshold_operator: str,
    fixed_threshold: float,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    real_rows = panel_by_condition[REAL_CONDITION]
    real_scores = _scores_for_rows(real_rows, scores)
    overall_seed = _child_seed(seed, "primary", "shared_clusters", "overall")
    overall_cluster_sets = [
        set(_row_clusters(real_rows, label="primary real")),
        *[
            set(
                _row_clusters(
                    panel_by_condition[condition],
                    label=f"primary {condition}",
                )
            )
            for condition in FORGED_CONDITIONS
        ],
    ]
    overall_cluster_names, overall_cluster_weights = (
        _shared_poisson_cluster_plan(
            cluster_sets=overall_cluster_sets,
            iterations=iterations,
            seed=overall_seed,
        )
    )
    domain_real_rows = {
        domain: [
            row for row in real_rows if str(row["domain"]) == domain
        ]
        for domain in domains
    }
    domain_plans = {}
    for domain in domains:
        domain_sets = [
            set(
                _row_clusters(
                    domain_real_rows[domain],
                    label=f"primary real {domain}",
                )
            ),
            *[
                set(
                    _row_clusters(
                        [
                            row
                            for row in panel_by_condition[condition]
                            if str(row["domain"]) == domain
                        ],
                        label=f"primary {condition} {domain}",
                    )
                )
                for condition in FORGED_CONDITIONS
            ],
        ]
        domain_seed = _child_seed(
            seed,
            "primary",
            "shared_clusters",
            domain,
        )
        domain_plans[domain] = (
            *_shared_poisson_cluster_plan(
                cluster_sets=domain_sets,
                iterations=iterations,
                seed=domain_seed,
            ),
            domain_seed,
        )
    public_by_condition: dict[str, Any] = {}
    overall_private: dict[
        str,
        tuple[Mapping[str, Any], Mapping[str, np.ndarray]],
    ] = {}
    domain_private: dict[
        str,
        dict[str, tuple[Mapping[str, Any], Mapping[str, np.ndarray]]],
    ] = {}

    for condition in FORGED_CONDITIONS:
        forged_rows = panel_by_condition[condition]
        overall = _unpaired_slice(
            real_scores,
            _scores_for_rows(forged_rows, scores),
            real_clusters=_row_clusters(
                real_rows,
                label="primary real",
            ),
            forged_clusters=_row_clusters(
                forged_rows,
                label=f"primary {condition}",
            ),
            shared_cluster_names=overall_cluster_names,
            shared_cluster_weights=overall_cluster_weights,
            score_key=score_key,
            direction=direction,
            direction_sign=direction_sign,
            fixed_threshold_operator=fixed_threshold_operator,
            fixed_threshold=fixed_threshold,
            iterations=iterations,
            seed=overall_seed,
        )
        overall_private[condition] = overall
        by_domain_public: dict[str, Any] = {}
        domain_private[condition] = {}
        for domain in domains:
            domain_real = domain_real_rows[domain]
            domain_forged = [
                row for row in forged_rows if str(row["domain"]) == domain
            ]
            sliced = _unpaired_slice(
                _scores_for_rows(domain_real, scores),
                _scores_for_rows(domain_forged, scores),
                real_clusters=_row_clusters(
                    domain_real,
                    label=f"primary real {domain}",
                ),
                forged_clusters=_row_clusters(
                    domain_forged,
                    label=f"primary {condition} {domain}",
                ),
                shared_cluster_names=domain_plans[domain][0],
                shared_cluster_weights=domain_plans[domain][1],
                score_key=score_key,
                direction=direction,
                direction_sign=direction_sign,
                fixed_threshold_operator=fixed_threshold_operator,
                fixed_threshold=fixed_threshold,
                iterations=iterations,
                seed=domain_plans[domain][2],
            )
            by_domain_public[domain] = sliced[0]
            domain_private[condition][domain] = sliced
        public_by_condition[condition] = {
            "condition_family": _CONDITION_CONTRACT[condition][
                "condition_family"
            ],
            "manipulation_scope": _CONDITION_CONTRACT[condition][
                "manipulation_scope"
            ],
            "overall": overall[0],
            "by_domain": by_domain_public,
        }

    def macro_bundle(conditions: Sequence[str]) -> dict[str, Any]:
        return {
            "overall": _macro_slice(
                overall_private,
                conditions=conditions,
                score_key=score_key,
                direction=direction,
                fixed_threshold_operator=fixed_threshold_operator,
                fixed_threshold=fixed_threshold,
            ),
            "by_domain": {
                domain: _macro_slice(
                    {
                        condition: domain_private[condition][domain]
                        for condition in conditions
                    },
                    conditions=conditions,
                    score_key=score_key,
                    direction=direction,
                    fixed_threshold_operator=fixed_threshold_operator,
                    fixed_threshold=fixed_threshold,
                )
                for domain in domains
            },
        }

    return {
        "design": "selection_unpaired_seven_condition_panel",
        "point_estimate_pairing": "none",
        "bootstrap_dependency": (
            "shared_source_content_cluster_across_real_and_forged_conditions"
        ),
        "source_cluster_overlap_with_real": {
            condition: len(overall_cluster_sets[0] & overall_cluster_sets[index])
            for index, condition in enumerate(FORGED_CONDITIONS, start=1)
        },
        "join_key": "sample_id",
        "pair_inference_from_task_id": False,
        "real_condition": REAL_CONDITION,
        "forged_conditions": list(FORGED_CONDITIONS),
        "domains": list(domains),
        "by_condition": public_by_condition,
        "all_conditions_macro": macro_bundle(FORGED_CONDITIONS),
        "family_macro": {
            "local": {
                "conditions": list(LOCAL_CONDITIONS),
                **macro_bundle(LOCAL_CONDITIONS),
            },
            "fullframe": {
                "conditions": list(FULLFRAME_CONDITIONS),
                **macro_bundle(FULLFRAME_CONDITIONS),
            },
        },
    }


def _delta_point(deltas: np.ndarray) -> dict[str, Any]:
    vector = np.asarray(deltas, dtype=np.float64)
    if vector.ndim != 1 or not vector.size or not np.isfinite(vector).all():
        raise ValueError("matched deltas must be a finite non-empty vector")
    wins = int(np.count_nonzero(vector > 0.0))
    losses = int(np.count_nonzero(vector < 0.0))
    ties = int(np.count_nonzero(vector == 0.0))
    return {
        "mean_score_delta": float(np.mean(vector)),
        "median_score_delta": float(np.median(vector)),
        "strict_matched_ranking_accuracy": wins / int(vector.size),
        "standard_deviation": float(np.std(vector)),
        "minimum": float(np.min(vector)),
        "maximum": float(np.max(vector)),
        "wins": wins,
        "losses": losses,
        "ties": ties,
    }


def _cluster_delta_slice(
    observations: Sequence[Mapping[str, Any]],
    *,
    shared_cluster_names: Sequence[str],
    shared_cluster_weights: np.ndarray,
    direction: str,
    iterations: int,
    seed: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    if not observations:
        raise ValueError("secondary matched slice is empty")
    observation_clusters = tuple(
        str(observation["source_content_cluster"])
        for observation in observations
    )
    cluster_names = sorted(set(observation_clusters))
    cluster_index = {
        cluster: index for index, cluster in enumerate(shared_cluster_names)
    }
    try:
        columns = np.asarray(
            [cluster_index[cluster] for cluster in observation_clusters],
            dtype=np.int64,
        )
    except KeyError as exc:
        raise ValueError("secondary slice references an unplanned cluster") from exc
    weights = np.asarray(shared_cluster_weights)
    if (
        weights.shape != (iterations, len(shared_cluster_names))
        or not np.issubdtype(weights.dtype, np.integer)
        or np.any(weights < 0)
    ):
        raise ValueError("shared secondary cluster weights are invalid")
    vector = np.asarray(
        [float(row["score_delta"]) for row in observations],
        dtype=np.float64,
    )
    point = _delta_point(vector)
    replicates: dict[str, list[float]] = {
        metric: [] for metric in _DELTA_BOOTSTRAP_METRICS
    }
    for iteration in range(iterations):
        sampled = np.repeat(vector, weights[iteration, columns])
        if not sampled.size:
            raise ValueError("secondary cluster bootstrap produced an empty slice")
        replicate = _delta_point(sampled)
        for metric in _DELTA_BOOTSTRAP_METRICS:
            replicates[metric].append(float(replicate[metric]))
    arrays = {
        metric: np.asarray(values, dtype=np.float64)
        for metric, values in replicates.items()
    }
    public: dict[str, Any] = {
        "pairs": len(observations),
        "source_content_clusters": len(cluster_names),
        "score_direction": direction,
        "score_delta": (
            "direction_normalized_forged_score_minus_real_score; "
            "positive_means_forged_ranked_more_forged"
        ),
        "wins": point["wins"],
        "losses": point["losses"],
        "ties": point["ties"],
        "score_delta_standard_deviation": point["standard_deviation"],
        "score_delta_minimum": point["minimum"],
        "score_delta_maximum": point["maximum"],
        "bootstrap_unit": "shared_source_content_cluster_poisson",
        "bootstrap_iterations": iterations,
        "bootstrap_seed": seed,
    }
    for metric in _DELTA_BOOTSTRAP_METRICS:
        public[metric] = {
            "estimate": float(point[metric]),
            "ci95_percentile": _percentile_ci(arrays[metric]),
        }
    return public, arrays


def _macro_delta_slice(
    condition_slices: Mapping[
        str,
        tuple[Mapping[str, Any], Mapping[str, np.ndarray]],
    ],
    *,
    conditions: Sequence[str],
    direction: str,
) -> dict[str, Any]:
    public: dict[str, Any] = {
        "aggregation": "unweighted_condition_macro",
        "conditions": list(conditions),
        "score_direction": direction,
        "bootstrap_unit": (
            "condition_macro_with_shared_source_content_cluster_"
            "poisson_bootstrap"
        ),
    }
    for metric in _DELTA_BOOTSTRAP_METRICS:
        estimates = [
            float(condition_slices[condition][0][metric]["estimate"])
            for condition in conditions
        ]
        replicate_matrix = np.stack(
            [
                condition_slices[condition][1][metric]
                for condition in conditions
            ]
        )
        public[metric] = {
            "estimate": float(np.mean(estimates)),
            "ci95_percentile": _percentile_ci(
                np.mean(replicate_matrix, axis=0)
            ),
        }
    return public


def _summarize_secondary(
    *,
    source_pairs: Sequence[Mapping[str, Any]],
    scores: Mapping[str, float],
    direction: str,
    direction_sign: float,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    observations: list[dict[str, Any]] = []
    by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for pair in source_pairs:
        condition = str(pair["condition"])
        delta = direction_sign * (
            scores[str(pair["forged_sample_id"])]
            - scores[str(pair["real_sample_id"])]
        )
        observation = {
            **dict(pair),
            "score_delta": float(delta),
        }
        observations.append(observation)
        by_condition[condition].append(observation)

    shared_seed = _child_seed(seed, "secondary", "shared_clusters")
    shared_cluster_names, shared_cluster_weights = (
        _shared_poisson_cluster_plan(
            cluster_sets=[
                {
                    str(row["source_content_cluster"])
                    for row in by_condition[condition]
                }
                for condition in FORGED_CONDITIONS
            ],
            iterations=iterations,
            seed=shared_seed,
        )
    )
    condition_slices: dict[
        str,
        tuple[Mapping[str, Any], Mapping[str, np.ndarray]],
    ] = {}
    condition_public: dict[str, Any] = {}
    for condition in FORGED_CONDITIONS:
        sliced = _cluster_delta_slice(
            by_condition[condition],
            shared_cluster_names=shared_cluster_names,
            shared_cluster_weights=shared_cluster_weights,
            direction=direction,
            iterations=iterations,
            seed=shared_seed,
        )
        condition_slices[condition] = sliced
        condition_public[condition] = sliced[0]

    all_pairs = _cluster_delta_slice(
        observations,
        shared_cluster_names=shared_cluster_names,
        shared_cluster_weights=shared_cluster_weights,
        direction=direction,
        iterations=iterations,
        seed=shared_seed,
    )[0]
    by_family = {
        "local": _cluster_delta_slice(
            [
                row
                for condition in LOCAL_CONDITIONS
                for row in by_condition[condition]
            ],
            shared_cluster_names=shared_cluster_names,
            shared_cluster_weights=shared_cluster_weights,
            direction=direction,
            iterations=iterations,
            seed=shared_seed,
        )[0],
        "fullframe": _cluster_delta_slice(
            [
                row
                for condition in FULLFRAME_CONDITIONS
                for row in by_condition[condition]
            ],
            shared_cluster_names=shared_cluster_names,
            shared_cluster_weights=shared_cluster_weights,
            direction=direction,
            iterations=iterations,
            seed=shared_seed,
        )[0],
    }
    return {
        "design": "source_matched_six_condition_pairs",
        "bootstrap_dependency": (
            "shared_source_content_cluster_across_conditions"
        ),
        "join_keys": ["real_sample_id", "forged_sample_id"],
        "pair_inference_from_task_id": False,
        "by_condition": condition_public,
        "all_pairs": all_pairs,
        "by_family": by_family,
        "all_conditions_macro": _macro_delta_slice(
            condition_slices,
            conditions=FORGED_CONDITIONS,
            direction=direction,
        ),
        "family_macro": {
            "local": _macro_delta_slice(
                condition_slices,
                conditions=LOCAL_CONDITIONS,
                direction=direction,
            ),
            "fullframe": _macro_delta_slice(
                condition_slices,
                conditions=FULLFRAME_CONDITIONS,
                direction=direction,
            ),
        },
    }


def summarize_balanced250_t1(
    inputs: Sequence[Mapping[str, Any]],
    panel: Sequence[Mapping[str, Any]],
    source_pairs: Sequence[Mapping[str, Any]],
    results: Sequence[Mapping[str, Any]],
    *,
    run_id: str,
    run_manifest_fingerprint: str,
    run_dataset_contract: RunDatasetContract,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    iterations: int = DEFAULT_BOOTSTRAP_ITERATIONS,
) -> dict[str, Any]:
    """Validate complete score coverage and summarize both T1 designs.

    ``results`` must contain exactly one successful, metrics-valid v2 row for
    every row in ``inputs``. Every result must match the expected run ID,
    manifest fingerprint, and full canonical input identity. Primary metrics
    join ``panel`` to those scores by ``sample_id``. Secondary metrics follow
    only the two explicit endpoint IDs in ``source_pairs``; ``task_id`` is
    never consulted for pairing.

    Score key, direction, fixed threshold, and operator are read only from
    the verified run dataset contract. Ranking metrics are direction-
    normalized. TPR at target FPR uses its separately reported strict
    real-score quantile operator.
    """

    run_id_value = _nonempty_string(run_id, "run_id")
    fingerprint_value = _sha256(
        run_manifest_fingerprint,
        "run_manifest_fingerprint",
    )
    iterations_value, seed_value = _validate_bootstrap_arguments(
        iterations,
        seed,
    )

    input_by_id, schema_version, dataset_id, input_counts = _validate_inputs(
        inputs
    )
    (
        panel_by_id,
        panel_by_condition,
        domains,
        panel_rows_per_condition,
    ) = _validate_panel(
        panel,
        input_by_id=input_by_id,
        schema_version=schema_version,
        dataset_id=dataset_id,
    )
    validated_pairs = _validate_source_pairs(
        source_pairs,
        input_by_id=input_by_id,
        panel_by_condition=panel_by_condition,
        schema_version=schema_version,
        dataset_id=dataset_id,
        rows_per_condition=panel_rows_per_condition,
    )
    score_spec, run_dataset_contract_sha256 = _validate_run_dataset_contract(
        run_dataset_contract,
        inputs=inputs,
        panel=panel,
        source_pairs=source_pairs,
        schema_version=schema_version,
        dataset_id=dataset_id,
        input_counts=input_counts,
    )
    score_key_value = score_spec.key
    direction_value, direction_sign, target_fpr_operator = _direction_contract(
        score_spec.direction
    )
    fixed_operator_value = _fixed_threshold_operator(
        score_spec.threshold_operator,
        direction_sign=direction_sign,
    )
    threshold_value = score_spec.fixed_threshold

    referenced_input_ids = set(panel_by_id)
    for pair in validated_pairs:
        referenced_input_ids.add(str(pair["real_sample_id"]))
        referenced_input_ids.add(str(pair["forged_sample_id"]))
    if referenced_input_ids != set(input_by_id):
        missing = sorted(set(input_by_id) - referenced_input_ids)
        extra = sorted(referenced_input_ids - set(input_by_id))
        raise ValueError(
            "input reference coverage mismatch: "
            f"unreferenced={missing[:3]}, unknown={extra[:3]}"
        )

    scores = _validate_results(
        results,
        input_by_id=input_by_id,
        score_key=score_key_value,
        run_id=run_id_value,
        run_manifest_fingerprint=fingerprint_value,
    )
    primary = _summarize_primary(
        panel_by_condition=panel_by_condition,
        domains=domains,
        scores=scores,
        score_key=score_key_value,
        direction=direction_value,
        direction_sign=direction_sign,
        fixed_threshold_operator=fixed_operator_value,
        fixed_threshold=threshold_value,
        iterations=iterations_value,
        seed=seed_value,
    )
    secondary = _summarize_secondary(
        source_pairs=validated_pairs,
        scores=scores,
        direction=direction_value,
        direction_sign=direction_sign,
        iterations=iterations_value,
        seed=seed_value,
    )
    panel_counts = {
        condition: len(panel_by_condition[condition])
        for condition in PANEL_CONDITIONS
    }
    source_pair_counts = dict(
        sorted(Counter(row["condition"] for row in validated_pairs).items())
    )
    return {
        "schema_version": "balanced250_t1_summary_v1",
        "dataset_schema_version": schema_version,
        "dataset_id": dataset_id,
        "run_id": run_id_value,
        "run_manifest_fingerprint": fingerprint_value,
        "run_dataset_contract_sha256": run_dataset_contract_sha256,
        "score_contract": {
            "score_key": score_key_value,
            "direction": direction_value,
            "fixed_threshold": threshold_value,
            "fixed_threshold_operator": fixed_operator_value,
            "tpr_at_fpr_5_percent_threshold_operator": (
                target_fpr_operator
            ),
            "target_fpr": TARGET_FPR,
            "fpr_quantile_method": FPR_QUANTILE_METHOD,
        },
        "bootstrap": {
            "iterations": iterations_value,
            "seed": seed_value,
            "primary_unit": (
                "shared_source_content_cluster_poisson_across_labels_"
                "and_conditions"
            ),
            "secondary_unit": (
                "shared_source_content_cluster_poisson_across_conditions"
            ),
            "ci": "two_sided_95_percentile",
        },
        "coverage": {
            "inputs": len(input_by_id),
            "panel": len(panel_by_id),
            "source_pairs": len(validated_pairs),
            "results": len(scores),
            "input_condition_counts": dict(sorted(input_counts.items())),
            "panel_condition_counts": panel_counts,
            "source_pair_condition_counts": source_pair_counts,
            "panel_rows_per_condition": panel_rows_per_condition,
            "referenced_input_ids": len(referenced_input_ids),
            "missing_result_ids": [],
            "unexpected_result_ids": [],
            "duplicate_result_ids": 0,
            "all_results_successful": True,
            "is_complete": True,
            "join_policy": (
                "sample_id_only_for_primary_and_explicit_source_pair_endpoint_"
                "ids_only_for_secondary"
            ),
            "task_id_pair_inference": False,
        },
        "primary": primary,
        "secondary": secondary,
    }


__all__ = [
    "DEFAULT_BOOTSTRAP_ITERATIONS",
    "DEFAULT_BOOTSTRAP_SEED",
    "DEFAULT_DIRECTION",
    "DEFAULT_FIXED_THRESHOLD",
    "DEFAULT_SCORE_KEY",
    "FORGED_CONDITIONS",
    "FULLFRAME_CONDITIONS",
    "LOCAL_CONDITIONS",
    "PANEL_CONDITIONS",
    "REAL_CONDITION",
    "summarize_balanced250_t1",
]
