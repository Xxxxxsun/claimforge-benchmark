"""Strict run/result contracts for the Balanced250 benchmark.

This module is intentionally independent of model implementations.  It binds
one canonical release and one deterministic input selection to a run, creates
the common identity prefix for v2 result rows, validates append-only attempt
history, and calculates fail-closed coverage.

Mouse-v1 compatibility remains in the existing runners.  In particular,
Balanced250 identities never manufacture a ``pair_rank``: primary membership
comes from ``panel.jsonl`` and secondary pairing comes from
``source_pairs.jsonl``.
"""

from __future__ import annotations

import hashlib
import math
import numbers
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from eval.opensource.common import stable_json

if TYPE_CHECKING:
    from eval.opensource.canonical_release import (
        CanonicalRelease,
        Capability,
        LedgerView,
        SelectionSpec,
    )


RESULT_SCHEMA_VERSION = "opensource_result_v2"
RUN_DATASET_CONTRACT_SCHEMA_VERSION = "opensource_run_dataset_contract_v2"
BALANCED_RELEASE_SCHEMA_VERSION = "claimforge_balanced250_canonical_v1"

CONDITION_ORDER = (
    "real",
    "local_mouse",
    "local_cat",
    "local_trash_can",
    "fullframe_mouse",
    "fullframe_cat",
    "fullframe_trash_can",
)
CONDITIONS = frozenset(CONDITION_ORDER)

SCORE_DIRECTIONS = frozenset(
    {
        "higher_means_fake",
        "lower_means_fake",
    }
)
THRESHOLD_OPERATORS = frozenset({">", ">=", "<", "<="})

_IDENTITY_FIELDS = (
    "dataset_id",
    "sample_id",
    "rank",
    "condition",
    "condition_family",
    "manipulation_scope",
    "normalized_task_id",
    "task_id",
    "kind",
    "label",
    "domain",
    "gt_mask_kind",
    "input_path",
    "input_sha256",
    "input_width",
    "input_height",
)

_CONDITION_SEMANTICS = {
    "real": {
        "condition_family": "real",
        "manipulation_scope": "authentic",
        "kind": "real",
        "label": 0,
        "gt_mask_kind": "all_zero",
    },
    "local_mouse": {
        "condition_family": "local_splice",
        "manipulation_scope": "local_insertion",
        "kind": "forged",
        "label": 1,
        "gt_mask_kind": "exact_diff",
    },
    "local_cat": {
        "condition_family": "local_splice",
        "manipulation_scope": "local_insertion",
        "kind": "forged",
        "label": 1,
        "gt_mask_kind": "exact_diff",
    },
    "local_trash_can": {
        "condition_family": "local_splice",
        "manipulation_scope": "local_insertion",
        "kind": "forged",
        "label": 1,
        "gt_mask_kind": "exact_diff",
    },
    "fullframe_mouse": {
        "condition_family": "full_frame_conditional_edit",
        "manipulation_scope": "conditional_full_frame_edit",
        "kind": "forged",
        "label": 1,
        "gt_mask_kind": "not_applicable",
    },
    "fullframe_cat": {
        "condition_family": "full_frame_conditional_edit",
        "manipulation_scope": "conditional_full_frame_edit",
        "kind": "forged",
        "label": 1,
        "gt_mask_kind": "not_applicable",
    },
    "fullframe_trash_can": {
        "condition_family": "full_frame_conditional_edit",
        "manipulation_scope": "conditional_full_frame_edit",
        "kind": "forged",
        "label": 1,
        "gt_mask_kind": "not_applicable",
    },
}

_RESERVED_SCORE_KEYS = frozenset(
    {
        "schema_version",
        "run_id",
        "run_manifest_fingerprint",
        "id",
        "pair_rank",
        "status",
        "valid_for_metrics",
        "error_type",
        "error_message",
        *_IDENTITY_FIELDS,
    }
)


class ContractError(ValueError):
    """A run, result, or release binding violates the frozen contract."""


class IncompleteCoverageError(ContractError):
    """The latest attempts do not provide successful coverage of every input."""


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractError(f"{label} is not an object")
    return value


def _nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ContractError(f"{label} is not a non-empty string")
    return value


def _nonnegative_integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise ContractError(f"{label} is not a non-negative integer")
    result = int(value)
    if result < 0:
        raise ContractError(f"{label} is not a non-negative integer")
    return result


def _positive_integer(value: Any, label: str) -> int:
    result = _nonnegative_integer(value, label)
    if result == 0:
        raise ContractError(f"{label} is not positive")
    return result


def _binary_label(value: Any, label: str) -> int:
    result = _nonnegative_integer(value, label)
    if result not in (0, 1):
        raise ContractError(f"{label} is not a binary label")
    return result


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ContractError(f"{label} is not a finite real number")
    result = float(value)
    if not math.isfinite(result):
        raise ContractError(f"{label} is not a finite real number")
    return result


def _sha256(value: Any, label: str) -> str:
    result = _nonempty_string(value, label)
    if len(result) != 64 or any(character not in "0123456789abcdef" for character in result):
        raise ContractError(f"{label} is not a lowercase SHA-256")
    return result


def _input_path(value: Any, sample_id: str) -> str:
    result = _nonempty_string(value, "input_path")
    if "\\" in result:
        raise ContractError("input_path must use POSIX separators")
    pure = PurePosixPath(result)
    if (
        pure.is_absolute()
        or pure.as_posix() != result
        or any(part in ("", ".", "..") for part in pure.parts)
    ):
        raise ContractError("input_path is absolute, non-canonical, or traversing")
    if pure.name != f"{sample_id}.jpg":
        raise ContractError("input_path filename does not match sample_id")
    return result


def _identifier(value: Any, label: str) -> str:
    """Return a stable string identifier for strings and string-valued enums."""

    if isinstance(value, Enum):
        if isinstance(value.value, str) and value.value:
            return str(value.value)
        return value.name.lower()
    if isinstance(value, str):
        return str(_nonempty_string(value, label))
    enum_value = getattr(value, "value", None)
    if isinstance(enum_value, str) and enum_value:
        return enum_value
    enum_name = getattr(value, "name", None)
    if isinstance(enum_name, str) and enum_name:
        return enum_name.lower()
    raise ContractError(f"{label} has no stable string identifier")


def _repo_path(path: Any, repo_root: Path, label: str) -> str:
    if not isinstance(path, (str, Path)):
        raise ContractError(f"{label} is not a filesystem path")
    root = repo_root.resolve()
    candidate = Path(path)
    resolved = candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
    try:
        return resolved.relative_to(root).as_posix()
    except ValueError as error:
        raise ContractError(f"{label} escapes the repository root") from error


def _ordered_conditions(values: Iterable[Any], label: str) -> tuple[str, ...]:
    conditions = tuple(_nonempty_string(value, label) for value in values)
    if not conditions:
        raise ContractError(f"{label} is empty")
    if len(conditions) != len(set(conditions)):
        raise ContractError(f"{label} contains duplicates")
    unknown = set(conditions) - CONDITIONS
    if unknown:
        raise ContractError(f"{label} contains unsupported condition {sorted(unknown)[0]!r}")
    present = set(conditions)
    return tuple(condition for condition in CONDITION_ORDER if condition in present)


@dataclass(frozen=True)
class ScoreSpec:
    """The one frozen T1 score exposed by a runner."""

    key: str
    direction: str
    fixed_threshold: float
    threshold_operator: str

    def __post_init__(self) -> None:
        key = _nonempty_string(self.key, "score key")
        if key in _RESERVED_SCORE_KEYS:
            raise ContractError(f"score key {key!r} is reserved")
        direction = _nonempty_string(self.direction, "score direction")
        if direction not in SCORE_DIRECTIONS:
            raise ContractError(f"unsupported score direction {direction!r}")
        operator = _nonempty_string(
            self.threshold_operator,
            "score threshold operator",
        )
        if operator not in THRESHOLD_OPERATORS:
            raise ContractError(f"unsupported threshold operator {operator!r}")
        if direction == "higher_means_fake" and operator not in (">", ">="):
            raise ContractError("higher_means_fake requires > or >= thresholding")
        if direction == "lower_means_fake" and operator not in ("<", "<="):
            raise ContractError("lower_means_fake requires < or <= thresholding")
        threshold = _finite_number(self.fixed_threshold, "fixed threshold")
        object.__setattr__(self, "key", key)
        object.__setattr__(self, "direction", direction)
        object.__setattr__(self, "fixed_threshold", threshold)
        object.__setattr__(self, "threshold_operator", operator)

    def as_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "direction": self.direction,
            "fixed_threshold": self.fixed_threshold,
            "threshold_operator": self.threshold_operator,
        }

    def validate_score(self, value: Any, *, label: str = "score") -> float:
        return _finite_number(value, label)

    def decision(self, value: Any) -> bool:
        score = self.validate_score(value)
        if self.threshold_operator == ">":
            return score > self.fixed_threshold
        if self.threshold_operator == ">=":
            return score >= self.fixed_threshold
        if self.threshold_operator == "<":
            return score < self.fixed_threshold
        return score <= self.fixed_threshold


@dataclass(frozen=True)
class ResultIdentityV2:
    """Common immutable identity carried by every Balanced250 result row."""

    run_id: str
    run_manifest_fingerprint: str
    dataset_id: str
    sample_id: str
    rank: int
    condition: str
    condition_family: str
    manipulation_scope: str
    normalized_task_id: str
    task_id: str
    kind: str
    label: int
    domain: str
    gt_mask_kind: str
    input_path: str
    input_sha256: str
    input_width: int
    input_height: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "run_id", _nonempty_string(self.run_id, "run_id"))
        object.__setattr__(
            self,
            "run_manifest_fingerprint",
            _sha256(
                self.run_manifest_fingerprint,
                "run_manifest_fingerprint",
            ),
        )
        object.__setattr__(
            self,
            "sample_id",
            _nonempty_string(self.sample_id, "sample_id"),
        )
        object.__setattr__(
            self,
            "dataset_id",
            _nonempty_string(self.dataset_id, "dataset_id"),
        )
        object.__setattr__(self, "rank", _nonnegative_integer(self.rank, "rank"))
        condition = _nonempty_string(self.condition, "condition")
        if condition not in CONDITIONS:
            raise ContractError(f"unsupported condition {condition!r}")
        object.__setattr__(self, "condition", condition)
        for field_name in (
            "condition_family",
            "manipulation_scope",
            "normalized_task_id",
            "task_id",
            "kind",
            "domain",
            "gt_mask_kind",
        ):
            object.__setattr__(
                self,
                field_name,
                _nonempty_string(getattr(self, field_name), field_name),
            )
        object.__setattr__(self, "label", _binary_label(self.label, "label"))
        object.__setattr__(
            self,
            "input_path",
            _input_path(self.input_path, self.sample_id),
        )
        object.__setattr__(
            self,
            "input_sha256",
            _sha256(self.input_sha256, "input_sha256"),
        )
        object.__setattr__(
            self,
            "input_width",
            _positive_integer(self.input_width, "input_width"),
        )
        object.__setattr__(
            self,
            "input_height",
            _positive_integer(self.input_height, "input_height"),
        )
        expected = _CONDITION_SEMANTICS[condition]
        for field_name, expected_value in expected.items():
            if getattr(self, field_name) != expected_value:
                raise ContractError(
                    f"{condition} requires {field_name}={expected_value!r}, "
                    f"not {getattr(self, field_name)!r}"
                )

    @classmethod
    def from_input(
        cls,
        row: Mapping[str, Any],
        *,
        run_id: str,
        run_manifest_fingerprint: str,
    ) -> ResultIdentityV2:
        value = _mapping(row, "input row")
        if "pair_rank" in value:
            raise ContractError("Balanced250 input identity must not contain pair_rank")
        return cls(
            run_id=run_id,
            run_manifest_fingerprint=run_manifest_fingerprint,
            dataset_id=value.get("dataset_id"),
            sample_id=value.get("sample_id"),
            rank=value.get("rank"),
            condition=value.get("condition"),
            condition_family=value.get("condition_family"),
            manipulation_scope=value.get("manipulation_scope"),
            normalized_task_id=value.get("normalized_task_id"),
            task_id=value.get("task_id"),
            kind=value.get("kind"),
            label=value.get("label"),
            domain=value.get("domain"),
            gt_mask_kind=value.get("gt_mask_kind"),
            input_path=value.get("canonical_path"),
            input_sha256=value.get("canonical_sha256"),
            input_width=value.get("width"),
            input_height=value.get("height"),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": RESULT_SCHEMA_VERSION,
            "run_id": self.run_id,
            "run_manifest_fingerprint": self.run_manifest_fingerprint,
            "dataset_id": self.dataset_id,
            "id": self.sample_id,
            "sample_id": self.sample_id,
            "rank": self.rank,
            "condition": self.condition,
            "condition_family": self.condition_family,
            "manipulation_scope": self.manipulation_scope,
            "normalized_task_id": self.normalized_task_id,
            "task_id": self.task_id,
            "kind": self.kind,
            "label": self.label,
            "domain": self.domain,
            "gt_mask_kind": self.gt_mask_kind,
            "input_path": self.input_path,
            "input_sha256": self.input_sha256,
            "input_width": self.input_width,
            "input_height": self.input_height,
        }


def build_result_identity(
    input_row: Mapping[str, Any],
    *,
    run_id: str,
    run_manifest_fingerprint: str,
) -> dict[str, Any]:
    """Build the exact common identity prefix for one v2 result."""

    return ResultIdentityV2.from_input(
        input_row,
        run_id=run_id,
        run_manifest_fingerprint=run_manifest_fingerprint,
    ).as_dict()


@dataclass(frozen=True)
class LedgerBinding:
    name: str
    path: str
    sha256: str
    rows: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _nonempty_string(self.name, "ledger name"))
        object.__setattr__(self, "path", _nonempty_string(self.path, "ledger path"))
        object.__setattr__(self, "sha256", _sha256(self.sha256, "ledger SHA-256"))
        object.__setattr__(
            self,
            "rows",
            _nonnegative_integer(self.rows, "ledger row count"),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "sha256": self.sha256,
            "rows": self.rows,
        }


@dataclass(frozen=True)
class CapabilityBinding:
    name: str
    conditions: tuple[str, ...]
    valid_for_t1: bool
    valid_for_t2: bool

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "name",
            _nonempty_string(self.name, "capability name"),
        )
        object.__setattr__(
            self,
            "conditions",
            _ordered_conditions(self.conditions, "capability conditions"),
        )
        if type(self.valid_for_t1) is not bool or type(self.valid_for_t2) is not bool:
            raise ContractError("capability validity flags must be booleans")
        if not self.valid_for_t1 and not self.valid_for_t2:
            raise ContractError("capability must support at least one task")

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "conditions": list(self.conditions),
            "valid_for_t1": self.valid_for_t1,
            "valid_for_t2": self.valid_for_t2,
        }


@dataclass(frozen=True)
class SelectionBinding:
    capability: str
    conditions: tuple[str, ...] | None
    per_condition_limit: int | None
    sample_id: str | None
    pair_limit: int | None
    selected_images: int
    selected_ids_sha256: str
    counts_by_condition: tuple[tuple[str, int], ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "spec": {
                "capability": self.capability,
                "conditions": (
                    list(self.conditions) if self.conditions is not None else None
                ),
                "per_condition_limit": self.per_condition_limit,
                "sample_id": self.sample_id,
                "pair_limit": self.pair_limit,
            },
            "selected_images": self.selected_images,
            "selected_ids_sha256": self.selected_ids_sha256,
            "counts_by_condition": dict(self.counts_by_condition),
        }


@dataclass(frozen=True)
class RunDatasetContract:
    release_schema_version: str
    release_kind: str
    dataset_id: str
    dataset_manifest: str
    dataset_manifest_sha256: str
    dataset_contract_sha256: str
    inputs_ledger: LedgerBinding
    panel_ledger: LedgerBinding
    source_pairs_ledger: LedgerBinding
    capability: CapabilityBinding
    selection: SelectionBinding
    score_spec: ScoreSpec | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": RUN_DATASET_CONTRACT_SCHEMA_VERSION,
            "release": {
                "schema_version": self.release_schema_version,
                "release_kind": self.release_kind,
                "dataset_id": self.dataset_id,
                "manifest_path": self.dataset_manifest,
                "manifest_sha256": self.dataset_manifest_sha256,
                "contract_sha256": self.dataset_contract_sha256,
            },
            "ledgers": {
                "inputs": self.inputs_ledger.as_dict(),
                "panel": self.panel_ledger.as_dict(),
                "source_pairs": self.source_pairs_ledger.as_dict(),
            },
            "capability": self.capability.as_dict(),
            "selection": self.selection.as_dict(),
            "score_spec": (
                self.score_spec.as_dict() if self.score_spec is not None else None
            ),
        }


def selected_ids_sha256(sample_ids: Iterable[str]) -> str:
    """Hash the ordered selected sample IDs with stable JSON encoding."""

    ordered = [
        _nonempty_string(sample_id, f"selected sample_id {index}")
        for index, sample_id in enumerate(sample_ids)
    ]
    if not ordered:
        raise ContractError("selected sample IDs are empty")
    if len(ordered) != len(set(ordered)):
        raise ContractError("selected sample IDs contain duplicates")
    return hashlib.sha256(stable_json(ordered).encode("utf-8")).hexdigest()


def _capability_binding(capability: Any) -> CapabilityBinding:
    try:
        conditions = capability.conditions
        valid_for_t1 = capability.valid_for_t1
        valid_for_t2 = capability.valid_for_t2
    except AttributeError as error:
        raise ContractError("selection capability has an incomplete interface") from error
    return CapabilityBinding(
        name=_identifier(capability, "capability"),
        conditions=tuple(conditions),
        valid_for_t1=valid_for_t1,
        valid_for_t2=valid_for_t2,
    )


def _selection_binding(
    selection_spec: Any,
    capability: CapabilityBinding,
    selected: Sequence[Mapping[str, Any]],
) -> SelectionBinding:
    try:
        selection_capability = selection_spec.capability
        conditions_value = selection_spec.conditions
        per_condition_limit = selection_spec.per_condition_limit
        sample_id = selection_spec.sample_id
        pair_limit = selection_spec.pair_limit
    except AttributeError as error:
        raise ContractError("selection spec has an incomplete interface") from error

    selection_capability_name = _identifier(
        selection_capability,
        "selection capability",
    )
    if selection_capability_name != capability.name:
        raise ContractError("selection capability binding drifted")

    conditions = (
        None
        if conditions_value is None
        else _ordered_conditions(conditions_value, "selected conditions")
    )
    if conditions is not None and not set(conditions).issubset(capability.conditions):
        raise ContractError("selected conditions exceed method capability")

    if per_condition_limit is not None:
        per_condition_limit = _positive_integer(
            per_condition_limit,
            "per_condition_limit",
        )
    if sample_id is not None:
        sample_id = _nonempty_string(sample_id, "selection sample_id")
    if pair_limit is not None:
        raise ContractError("Balanced250 selection must not set pair_limit")
    if sample_id is not None and (
        conditions is not None or per_condition_limit is not None
    ):
        raise ContractError(
            "sample_id is mutually exclusive with condition filters and limits"
        )

    identities = [
        ResultIdentityV2.from_input(
            row,
            run_id="selection-contract",
            run_manifest_fingerprint="0" * 64,
        )
        for row in selected
    ]
    ranks = [identity.rank for identity in identities]
    if ranks != sorted(ranks) or len(ranks) != len(set(ranks)):
        raise ContractError("selected inputs are not in unique rank order")
    sample_ids = [identity.sample_id for identity in identities]
    if sample_id is not None and sample_ids != [sample_id]:
        raise ContractError("selection sample_id does not match selected input")

    allowed_conditions = set(conditions or capability.conditions)
    unsupported = {
        identity.condition
        for identity in identities
        if identity.condition not in allowed_conditions
    }
    if unsupported:
        raise ContractError(
            f"selected input condition is not allowed: {sorted(unsupported)[0]!r}"
        )

    counts = Counter(identity.condition for identity in identities)
    if per_condition_limit is not None and any(
        count > per_condition_limit for count in counts.values()
    ):
        raise ContractError("selected condition count exceeds per_condition_limit")

    ordered_counts = tuple(
        (condition, counts[condition])
        for condition in CONDITION_ORDER
        if counts[condition]
    )
    return SelectionBinding(
        capability=capability.name,
        conditions=conditions,
        per_condition_limit=per_condition_limit,
        sample_id=sample_id,
        pair_limit=None,
        selected_images=len(selected),
        selected_ids_sha256=selected_ids_sha256(sample_ids),
        counts_by_condition=ordered_counts,
    )


def _ledger_binding(
    view: Any,
    *,
    expected_name: str,
    repo_root: Path,
    materialized_rows: Sequence[Mapping[str, Any]],
    manifest_ledgers: Mapping[str, Any],
) -> LedgerBinding:
    if view is None:
        raise ContractError(f"release has no {expected_name} ledger")
    try:
        name = _nonempty_string(view.name, f"{expected_name} ledger name")
        path = _repo_path(view.path, repo_root, f"{expected_name} ledger path")
        digest = _sha256(view.sha256, f"{expected_name} ledger SHA-256")
        rows = _nonnegative_integer(view.rows, f"{expected_name} ledger rows")
    except AttributeError as error:
        raise ContractError(f"{expected_name} ledger has an incomplete interface") from error
    if name != expected_name:
        raise ContractError(
            f"{expected_name} ledger is named {name!r}, not {expected_name!r}"
        )
    if rows != len(materialized_rows):
        raise ContractError(f"{expected_name} ledger materialized row count drifted")
    materialized_digest = hashlib.sha256(
        "".join(
            f"{stable_json(row)}\n" for row in materialized_rows
        ).encode("utf-8")
    ).hexdigest()
    if materialized_digest != digest:
        raise ContractError(
            f"{expected_name} ledger materialized content drifted"
        )

    raw = _mapping(
        manifest_ledgers.get(expected_name),
        f"manifest {expected_name} ledger",
    )
    raw_path = _repo_path(
        raw.get("path"),
        repo_root,
        f"manifest {expected_name} ledger path",
    )
    if raw_path != path:
        raise ContractError(f"{expected_name} ledger path drifted from manifest")
    if raw.get("sha256") != digest:
        raise ContractError(f"{expected_name} ledger SHA-256 drifted from manifest")
    if raw.get("rows") != rows:
        raise ContractError(f"{expected_name} ledger row count drifted from manifest")
    return LedgerBinding(name=name, path=path, sha256=digest, rows=rows)


def _validate_selected_against_release(
    release_rows: Sequence[Mapping[str, Any]],
    selected_rows: Sequence[Mapping[str, Any]],
) -> None:
    release_by_id: dict[str, ResultIdentityV2] = {}
    for row in release_rows:
        identity = ResultIdentityV2.from_input(
            row,
            run_id="release-contract",
            run_manifest_fingerprint="0" * 64,
        )
        if identity.sample_id in release_by_id:
            raise ContractError(
                f"release inputs contain duplicate sample_id {identity.sample_id!r}"
            )
        release_by_id[identity.sample_id] = identity

    for row in selected_rows:
        selected = ResultIdentityV2.from_input(
            row,
            run_id="release-contract",
            run_manifest_fingerprint="0" * 64,
        )
        expected = release_by_id.get(selected.sample_id)
        if expected is None:
            raise ContractError(
                f"selected sample_id {selected.sample_id!r} is not in inputs ledger"
            )
        if selected != expected:
            raise ContractError(
                f"selected identity drifted from inputs ledger for "
                f"{selected.sample_id!r}"
            )


def _validate_exact_selection_materialization(
    release_rows: Sequence[Mapping[str, Any]],
    selected_rows: Sequence[Mapping[str, Any]],
    *,
    capability: CapabilityBinding,
    selection: SelectionBinding,
) -> None:
    """Require the runner's rows to equal the selection spec, not a subset."""

    if selection.sample_id is not None:
        expected = [
            row
            for row in release_rows
            if row.get("sample_id") == selection.sample_id
        ]
    else:
        allowed = set(selection.conditions or capability.conditions)
        expected = [
            row
            for row in release_rows
            if row.get("condition") in allowed
        ]
        if selection.per_condition_limit is not None:
            limited: list[Mapping[str, Any]] = []
            ordered_conditions = selection.conditions or capability.conditions
            for condition in ordered_conditions:
                candidates = [
                    row
                    for row in expected
                    if row.get("condition") == condition
                ]
                candidates.sort(
                    key=lambda row: (
                        row.get("panel") is not True,
                        row.get("selection_rank")
                        if row.get("selection_rank") is not None
                        else math.inf,
                        int(row["rank"]),
                    )
                )
                limited.extend(
                    candidates[: selection.per_condition_limit]
                )
            expected = sorted(limited, key=lambda row: int(row["rank"]))

    expected_ids = [str(row.get("sample_id")) for row in expected]
    selected_ids = [str(row.get("sample_id")) for row in selected_rows]
    if selected_ids != expected_ids:
        raise ContractError(
            "selected rows do not exactly materialize the selection spec"
        )
    for index, (canonical, selected) in enumerate(
        zip(expected, selected_rows, strict=True)
    ):
        if stable_json(selected) != stable_json(canonical):
            raise ContractError(
                "selected row content does not exactly match the canonical "
                f"selection at index {index}"
            )


def build_run_dataset_contract(
    release: CanonicalRelease,
    selection_spec: SelectionSpec,
    selected_rows: Sequence[Mapping[str, Any]],
    *,
    score_spec: ScoreSpec | None,
) -> RunDatasetContract:
    """Bind a validated Balanced250 release and selection to one model run."""

    if not isinstance(selected_rows, Sequence) or isinstance(
        selected_rows,
        (str, bytes),
    ):
        raise ContractError("selected rows are not a sequence")
    if not selected_rows:
        raise ContractError("selected rows are empty")

    try:
        repo_root = Path(release.repo_root)
        manifest_path = release.manifest_path
        manifest_sha256 = release.manifest_sha256
        manifest = _mapping(release.manifest, "release manifest")
        schema_version = release.schema_version
        dataset_id = release.dataset_id
        release_kind = release.release_kind
        contract_sha256 = release.contract_sha256
        release_inputs = release.inputs
        release_panel = release.panel
        release_source_pairs = release.source_pairs
    except AttributeError as error:
        raise ContractError("canonical release has an incomplete interface") from error

    schema_version = _nonempty_string(schema_version, "release schema_version")
    if schema_version != BALANCED_RELEASE_SCHEMA_VERSION:
        raise ContractError(
            f"unsupported Balanced250 release schema {schema_version!r}"
        )
    dataset_id = _nonempty_string(dataset_id, "release dataset_id")
    release_kind_name = _identifier(release_kind, "release kind")
    manifest_sha256 = _sha256(manifest_sha256, "dataset manifest SHA-256")
    contract_sha256 = _sha256(contract_sha256, "dataset contract SHA-256")
    if manifest.get("schema_version") != schema_version:
        raise ContractError("release schema_version drifted from manifest")
    if manifest.get("dataset_id") != dataset_id:
        raise ContractError("release dataset_id drifted from manifest")
    if manifest.get("contract_sha256") != contract_sha256:
        raise ContractError("release contract SHA-256 drifted from manifest")

    manifest_ledgers = _mapping(manifest.get("ledgers"), "manifest ledgers")
    inputs_binding = _ledger_binding(
        release.inputs_ledger,
        expected_name="inputs",
        repo_root=repo_root,
        materialized_rows=release_inputs,
        manifest_ledgers=manifest_ledgers,
    )
    panel_binding = _ledger_binding(
        release.panel_ledger,
        expected_name="panel",
        repo_root=repo_root,
        materialized_rows=release_panel,
        manifest_ledgers=manifest_ledgers,
    )
    pairs_binding = _ledger_binding(
        release.source_pairs_ledger,
        expected_name="source_pairs",
        repo_root=repo_root,
        materialized_rows=release_source_pairs,
        manifest_ledgers=manifest_ledgers,
    )

    capability = _capability_binding(selection_spec.capability)
    if capability.valid_for_t1 and score_spec is None:
        raise ContractError("T1-capable run has no score spec")
    if not capability.valid_for_t1 and score_spec is not None:
        raise ContractError("T2-only run must not declare a T1 score spec")

    _validate_selected_against_release(release_inputs, selected_rows)
    for row in selected_rows:
        if row.get("schema_version") != schema_version:
            raise ContractError("selected row schema_version drifted")
        if row.get("dataset_id") != dataset_id:
            raise ContractError("selected row dataset_id drifted")
    selection = _selection_binding(selection_spec, capability, selected_rows)
    _validate_exact_selection_materialization(
        release_inputs,
        selected_rows,
        capability=capability,
        selection=selection,
    )

    return RunDatasetContract(
        release_schema_version=schema_version,
        release_kind=release_kind_name,
        dataset_id=dataset_id,
        dataset_manifest=_repo_path(
            manifest_path,
            repo_root,
            "dataset manifest path",
        ),
        dataset_manifest_sha256=manifest_sha256,
        dataset_contract_sha256=contract_sha256,
        inputs_ledger=inputs_binding,
        panel_ledger=panel_binding,
        source_pairs_ledger=pairs_binding,
        capability=capability,
        selection=selection,
        score_spec=score_spec,
    )


build_dataset_contract = build_run_dataset_contract


def validate_result_identity(
    result_row: Mapping[str, Any],
    expected: ResultIdentityV2,
    *,
    index: int | None = None,
) -> None:
    """Fail if a physical result attempt drifts from its selected input."""

    row = _mapping(result_row, "result row")
    prefix = f"result row {index}" if index is not None else "result row"
    if "pair_rank" in row:
        raise ContractError(f"{prefix} must not contain pair_rank")
    if row.get("schema_version") != RESULT_SCHEMA_VERSION:
        raise ContractError(f"{prefix} has wrong schema_version")
    if row.get("id") != expected.sample_id:
        raise ContractError(f"{prefix} id does not match sample_id")
    expected_values = expected.as_dict()
    for field_name in (
        "run_id",
        "run_manifest_fingerprint",
        *_IDENTITY_FIELDS,
    ):
        if row.get(field_name) != expected_values[field_name]:
            raise ContractError(f"{prefix} {field_name} identity drifted")


@dataclass(frozen=True)
class LatestAttempts:
    """Validated last-write-wins view of an append-only result JSONL."""

    run_id: str
    run_manifest_fingerprint: str
    expected_ids: tuple[str, ...]
    expected_by_sample_id: Mapping[str, ResultIdentityV2]
    latest_by_sample_id: Mapping[str, Mapping[str, Any]]
    attempts_by_sample_id: Mapping[str, int]
    physical_attempts: int
    superseded_attempts: int

    def pending_sample_ids(self, *, retry_errors: bool = True) -> tuple[str, ...]:
        pending: list[str] = []
        for sample_id in self.expected_ids:
            row = self.latest_by_sample_id.get(sample_id)
            if row is None or (retry_errors and row.get("status") == "error"):
                pending.append(sample_id)
        return tuple(pending)


def index_latest_attempts(
    expected_rows: Sequence[Mapping[str, Any]],
    result_rows: Iterable[Mapping[str, Any]],
    *,
    run_id: str,
    run_manifest_fingerprint: str,
    score_spec: ScoreSpec | None,
) -> LatestAttempts:
    """Validate every physical attempt, then retain the last row per sample."""

    run_id = _nonempty_string(run_id, "run_id")
    fingerprint = _sha256(
        run_manifest_fingerprint,
        "run_manifest_fingerprint",
    )
    if not isinstance(expected_rows, Sequence) or isinstance(
        expected_rows,
        (str, bytes),
    ):
        raise ContractError("expected rows are not a sequence")
    if not expected_rows:
        raise ContractError("expected rows are empty")

    expected: dict[str, ResultIdentityV2] = {}
    expected_ids: list[str] = []
    for row in expected_rows:
        identity = ResultIdentityV2.from_input(
            row,
            run_id=run_id,
            run_manifest_fingerprint=fingerprint,
        )
        if identity.sample_id in expected:
            raise ContractError(
                f"duplicate expected sample_id {identity.sample_id!r}"
            )
        expected[identity.sample_id] = identity
        expected_ids.append(identity.sample_id)

    latest: dict[str, Mapping[str, Any]] = {}
    attempts: Counter[str] = Counter()
    physical_attempts = 0
    for index, raw_row in enumerate(result_rows):
        row = _mapping(raw_row, f"result row {index}")
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or sample_id not in expected:
            raise ContractError(f"result row {index} has unexpected sample_id")
        validate_result_identity(row, expected[sample_id], index=index)
        status = row.get("status")
        if status not in ("ok", "error"):
            raise ContractError(f"result row {index} has invalid status")
        expected_validity = status == "ok"
        if row.get("valid_for_metrics") is not expected_validity:
            raise ContractError(
                f"result row {index} valid_for_metrics/status mismatch"
            )
        if score_spec is not None:
            if status == "ok":
                score_spec.validate_score(
                    row.get(score_spec.key),
                    label=f"result row {index} {score_spec.key}",
                )
            elif (
                score_spec.key in row
                and row.get(score_spec.key) is not None
            ):
                raise ContractError(
                    f"error result row {index} has a non-null "
                    f"{score_spec.key}"
                )
        latest[sample_id] = MappingProxyType(dict(row))
        attempts[sample_id] += 1
        physical_attempts += 1

    attempts_dict = dict(attempts)
    return LatestAttempts(
        run_id=run_id,
        run_manifest_fingerprint=fingerprint,
        expected_ids=tuple(expected_ids),
        expected_by_sample_id=MappingProxyType(expected),
        latest_by_sample_id=MappingProxyType(latest),
        attempts_by_sample_id=MappingProxyType(attempts_dict),
        physical_attempts=physical_attempts,
        superseded_attempts=physical_attempts - len(latest),
    )


@dataclass(frozen=True)
class Coverage:
    expected_images: int
    physical_attempts: int
    result_images: int
    valid_images: int
    error_images: int
    missing_images: int
    superseded_attempts: int
    coverage_fraction: float
    success_fraction: float
    is_complete: bool
    counts_by_condition: tuple[tuple[str, Mapping[str, int]], ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "expected_images": self.expected_images,
            "physical_attempts": self.physical_attempts,
            "result_images": self.result_images,
            "valid_images": self.valid_images,
            "error_images": self.error_images,
            "missing_images": self.missing_images,
            "superseded_attempts": self.superseded_attempts,
            "coverage_fraction": self.coverage_fraction,
            "success_fraction": self.success_fraction,
            "is_complete": self.is_complete,
            "counts_by_condition": {
                condition: dict(counts)
                for condition, counts in self.counts_by_condition
            },
        }

    def require_complete(self) -> None:
        if not self.is_complete:
            raise IncompleteCoverageError(
                "incomplete Balanced250 coverage: "
                f"expected={self.expected_images}, "
                f"valid={self.valid_images}, "
                f"errors={self.error_images}, "
                f"missing={self.missing_images}"
            )


def summarize_coverage(latest: LatestAttempts) -> Coverage:
    """Calculate strict latest-attempt coverage without hiding retry history."""

    expected = len(latest.expected_ids)
    result_images = len(latest.latest_by_sample_id)
    valid = sum(
        row.get("status") == "ok"
        for row in latest.latest_by_sample_id.values()
    )
    errors = sum(
        row.get("status") == "error"
        for row in latest.latest_by_sample_id.values()
    )
    missing = expected - result_images
    if result_images != valid + errors or missing < 0:
        raise ContractError("latest attempt accounting is internally inconsistent")

    condition_counts: dict[str, dict[str, int]] = {
        condition: {
            "expected_images": 0,
            "result_images": 0,
            "valid_images": 0,
            "error_images": 0,
            "missing_images": 0,
        }
        for condition in CONDITION_ORDER
    }
    for sample_id in latest.expected_ids:
        condition = latest.expected_by_sample_id[sample_id].condition
        counts = condition_counts[condition]
        counts["expected_images"] += 1
        row = latest.latest_by_sample_id.get(sample_id)
        if row is None:
            counts["missing_images"] += 1
        else:
            counts["result_images"] += 1
            if row["status"] == "ok":
                counts["valid_images"] += 1
            else:
                counts["error_images"] += 1

    is_complete = (
        valid == expected
        and errors == 0
        and missing == 0
        and result_images == expected
    )
    ordered_counts = tuple(
        (
            condition,
            MappingProxyType(dict(condition_counts[condition])),
        )
        for condition in CONDITION_ORDER
        if condition_counts[condition]["expected_images"]
    )
    return Coverage(
        expected_images=expected,
        physical_attempts=latest.physical_attempts,
        result_images=result_images,
        valid_images=valid,
        error_images=errors,
        missing_images=missing,
        superseded_attempts=latest.superseded_attempts,
        coverage_fraction=result_images / expected,
        success_fraction=valid / expected,
        is_complete=is_complete,
        counts_by_condition=ordered_counts,
    )


def require_complete_coverage(coverage: Coverage) -> None:
    """Raise unless every selected input's latest attempt is successful."""

    coverage.require_complete()


__all__ = [
    "BALANCED_RELEASE_SCHEMA_VERSION",
    "CONDITION_ORDER",
    "ContractError",
    "Coverage",
    "IncompleteCoverageError",
    "LatestAttempts",
    "LedgerBinding",
    "CapabilityBinding",
    "RESULT_SCHEMA_VERSION",
    "RUN_DATASET_CONTRACT_SCHEMA_VERSION",
    "ResultIdentityV2",
    "RunDatasetContract",
    "ScoreSpec",
    "SelectionBinding",
    "build_dataset_contract",
    "build_result_identity",
    "build_run_dataset_contract",
    "index_latest_attempts",
    "require_complete_coverage",
    "selected_ids_sha256",
    "summarize_coverage",
    "validate_result_identity",
]
