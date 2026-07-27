#!/usr/bin/env python3
"""Fail-closed audit and full replay for OmniAID on Balanced250.

The analyzer treats the Balanced v2 manifest, append-only results, runtime
summary, and every per-image NPZ as untrusted evidence.  It independently
rebuilds the canonical selection, validates the exact six-array 9,848-byte
artifact format, reconstructs the released model through the legacy
independent OmniAID audit path, replays the persisted head, softmax and
automatic router, recomputes the shared Balanced250 T1 metrics, and by
default freshly forwards all 1,775 canonical JPEGs through the full model.

OmniAID is a whole-image T1 classifier.  Direct full-canvas resize visibility
is a score-blind input diagnostic and never a predicted mask or T2 result.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import importlib.metadata
import io
import json
import math
import os
import re
import zipfile
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

from eval.opensource import analyze_omniaid_run as legacy_audit
from eval.opensource import run_omniaid as legacy
from eval.opensource.balanced250_metrics import summarize_balanced250_t1
from eval.opensource.balanced_run_contract import (
    RESULT_SCHEMA_VERSION,
    RunDatasetContract,
    ScoreSpec,
    build_result_identity,
    build_run_dataset_contract,
    index_latest_attempts,
    require_complete_coverage,
    selected_ids_sha256,
    summarize_coverage,
)
from eval.opensource.canonical_release import (
    BALANCED_CONDITIONS,
    BALANCED_DATASET_ID,
    BALANCED_SCHEMA,
    CanonicalRelease,
    Capability,
    SelectionSpec,
    load_canonical_release,
)
from eval.opensource.common import (
    atomic_write_json,
    repo_relative,
    sha256_file,
    stable_json,
    utc_now,
)


AUDIT_SCHEMA_VERSION = "omniaid_balanced_replay_audit_v2"
SMOKE_COMPARISON_SCHEMA_VERSION = "omniaid_balanced_smoke_comparison_v2"
METRICS_SCHEMA_VERSION = "balanced250_t1_summary_v1"
EXPECTED_RUN_MANIFEST_SCHEMA = "omniaid_balanced_run_manifest_v2"
EXPECTED_RUN_CONFIG_SCHEMA = "omniaid_balanced_run_config_v2"
EXPECTED_RUNTIME_SUMMARY_SCHEMA = "omniaid_balanced_runtime_summary_v2"
EXPECTED_CPU_PREFLIGHT_SCHEMA = "omniaid_balanced_cpu_preflight_v1"
EXPECTED_DATASET_CONTRACT_SCHEMA = "opensource_run_dataset_contract_v2"

DEFAULT_RESULTS_DIR = Path("results/opensource/omniaid")
DEFAULT_ARTIFACTS_DIR = Path("outputs/opensource/omniaid")
DEFAULT_DATASET_MANIFEST = Path("outputs/opensource/balanced250_v1/manifest.json")
DEFAULT_FORMAL_RUN_ID = "omniaid_dino_v2_mirage_auto_balanced250_v1_full1775_20260727"
DEFAULT_SMOKE_RUN_ID_A = (
    "omniaid_dino_v2_mirage_auto_balanced250_v1_smoke5x7_a_20260727"
)
DEFAULT_SMOKE_RUN_ID_B = (
    "omniaid_dino_v2_mirage_auto_balanced250_v1_smoke5x7_b_20260727"
)
DEFAULT_SOURCE_ROOT = legacy.DEFAULT_SOURCE_ROOT
DEFAULT_SPACE_ROOT = legacy.DEFAULT_SPACE_ROOT
DEFAULT_CHECKPOINT = legacy.DEFAULT_CHECKPOINT
DEFAULT_OMNIAID_CONFIG = legacy.DEFAULT_OMNIAID_CONFIG

FORMAL_IMAGES = 1775
SMOKE_IMAGES = 35
SMOKE_PER_CONDITION = 5
BOOTSTRAP_ITERATIONS = 1000
BOOTSTRAP_SEED = 20260726
EXPECTED_RUNTIME_SEED = legacy.MODEL_SEED

FORMAL_COUNTS = {
    "real": 275,
    "local_mouse": 250,
    "local_cat": 250,
    "local_trash_can": 250,
    "fullframe_mouse": 250,
    "fullframe_cat": 250,
    "fullframe_trash_can": 250,
}
FORMAL_SELECTED_ROWS_SHA256 = (
    "6b5128909eeffdbd88e61af02ca1bd191cb5460f94a23b47c87ebd0172e6d12c"
)
FORMAL_SELECTED_IDS_SHA256 = (
    "e4418d86461f889e4a4423f26aab63243e6f63a435a49624881c34979b812e41"
)
SMOKE5X7_SELECTED_IDS_SHA256 = (
    "b420bc581386a540b742d917d60d007f0e5522b6cca43fa217797944c40667e5"
)

ARTIFACT_FILE_BYTES = 9_848
ARTIFACT_MEMBER_BYTES = {
    "pooler_output.npy": 4_224,
    "class_logits.npy": 136,
    "routing_feature.npy": 4_224,
    "semantic_top_k_indices.npy": 144,
    "semantic_top_k_gates.npy": 136,
    "final_gates.npy": 152,
}
ARTIFACT_SCHEMA = {
    "pooler_output": ((legacy.FEATURE_DIMENSION,), np.float32),
    "class_logits": ((legacy.CLASS_COUNT,), np.float32),
    "routing_feature": ((legacy.FEATURE_DIMENSION,), np.float32),
    "semantic_top_k_indices": ((legacy.SEMANTIC_TOP_K,), np.int64),
    "semantic_top_k_gates": ((legacy.SEMANTIC_TOP_K,), np.float32),
    "final_gates": ((legacy.EXPERT_COUNT,), np.float32),
}

EXPECTED_ADAPTER_SOURCE_PATHS = (
    ".gitignore",
    "eval/__init__.py",
    "eval/opensource/__init__.py",
    "eval/opensource/run_omniaid_balanced.py",
    "eval/opensource/analyze_omniaid_balanced.py",
    "eval/opensource/run_omniaid.py",
    "eval/opensource/analyze_omniaid_run.py",
    "eval/opensource/omniaid_metrics.py",
    "eval/opensource/ufd_metrics.py",
    "eval/opensource/canonical_release.py",
    "eval/opensource/balanced_run_contract.py",
    "eval/opensource/balanced250_metrics.py",
    "eval/opensource/common.py",
)

EXPECTED_IMMUTABLE_KEYS = frozenset(
    {
        "schema_version",
        "run_id",
        "mode",
        "adapter_sources",
        "model",
        "preprocess",
        "score_spec",
        "task_scope",
        "dataset_contract",
        "selected_rows_sha256",
        "selected_ids_sha256",
        "visibility_census",
        "source",
        "assets",
        "runtime",
        "cpu_preflight",
        "execution_model_load",
        "execution_official_golden",
        "artifact_contract",
        "artifact_policy",
        "outputs",
    }
)

EXPECTED_OK_RESULT_FIELDS = frozenset(
    {
        "preprocess",
        "preprocess_latency_ms",
        "artifact_path",
        "artifact_sha256",
        "artifact_bytes",
        "artifact_keys",
        "artifact_paths",
        "artifact_array_sha256",
        "feature_shape",
        "feature_dtype",
        "feature_semantics",
        "feature_array_sha256",
        "class_logits_shape",
        "class_logits_dtype",
        "class_logits_array_sha256",
        "routing_feature_shape",
        "routing_feature_dtype",
        "routing_feature_semantics",
        "semantic_top_k_indices_shape",
        "semantic_top_k_indices_dtype",
        "semantic_top_k_gates_shape",
        "semantic_top_k_gates_dtype",
        "final_gates_shape",
        "final_gates_dtype",
        "class_logits",
        "raw_logit_margin",
        "fake_probability",
        "probability",
        "ai_score",
        "score",
        "routing_mode",
        "semantic_expert_names",
        "artifact_expert_name",
        "semantic_top_k_indices",
        "semantic_top_k_gates",
        "final_expert_gates",
        "semantic_gate_sum",
        "final_gate_sum",
        "classification_decision",
        "classification",
        "t1",
        "manual_replay",
        "latency_ms",
        "peak_cuda_memory_bytes",
    }
)
EXPECTED_ERROR_RESULT_FIELDS = frozenset(
    {
        "class_logits",
        "raw_logit_margin",
        "fake_probability",
        "probability",
        "ai_score",
        "score",
        "classification_decision",
        "latency_ms",
        "peak_cuda_memory_bytes",
        "error_type",
        "error",
        "traceback",
    }
)

FROZEN_RUNTIME_VERSIONS = {
    "python": "3.12.3",
    "torch": "2.8.0.dev20250627+cu128",
    "torch_distribution": "2.8.0.dev20250627+cu128",
    "torchvision": "0.23.0.dev20250627+cu128",
    "torchvision_distribution": "0.23.0.dev20250627+cu128",
    "transformers": "4.57.3",
    "numpy": "2.2.6",
    "Pillow": "12.0.0",
    "huggingface-hub": "0.36.0",
    "tokenizers": "0.22.2",
    "safetensors": "0.5.2",
    "setuptools": "79.0.1",
}
FROZEN_RUNTIME_MODULE_HASHES = {
    "torch": "abc68f909360770fb0dd0fc263b43ae65906bd66d1eab99cdcf5c5abf23c0e0d",
    "torchvision": ("ee2c9f4110cf1203db48c42601607329ac1f19709fa91c152f8d95eb53437a73"),
    "transformers": (
        "7eb0743ed843f24c1d4e8b8daf4b18249cb81403b4571a70c00be0a4c0a67bd4"
    ),
    "numpy": "6ae17b070c0f70a8e3cad89a510a256942e5a1f37ea5feb120cec167ed2a6236",
    "PIL": "43828e12947b4bf5ec8f7d1fbceb2f47de311295f8294b15794c1a54fd5f53cd",
    "huggingface_hub": (
        "017af8861f5bde565c6ce9f231457f63a2579f643588c09560f9ead4560f84e4"
    ),
    "transformers.models.dinov3_vit.configuration_dinov3_vit": (
        "1ac7cb889e2314e8cadf9b0bed43d42cfb05dce6d730f719be4380c8a10a8a46"
    ),
    "transformers.models.dinov3_vit.modeling_dinov3_vit": (
        "79f9cc140c1eca19d992285cecfe57faf0f1c470e6f2b296bf01f7fd94473705"
    ),
    "transformers.models.dinov3_vit.modular_dinov3_vit": (
        "c2e7a9ed2faf064fe111c0f980af5a881144f65563d5ffe41b903bccba07dadb"
    ),
}

EXPECTED_OFFICIAL_GOLDEN_KEYS = frozenset(
    {
        "status",
        "kind",
        "device_family",
        "runtime_abs_tolerance",
        "official_service_abs_tolerance",
        "official_service_observed_at",
        "cases",
        "mouse_model_scores_computed",
    }
)
EXPECTED_OFFICIAL_GOLDEN_CASE_KEYS = frozenset(
    {
        "path",
        "input_sha256",
        "preprocess",
        "logits",
        "fake_probability",
        "final_expert_gates",
        "array_sha256",
        "repeat_all_arrays_exact",
        "observed_official_service_probability",
        "official_service_probability_abs_diff",
        "frozen_runtime_logit_max_abs_diff",
        "frozen_runtime_probability_abs_diff",
        "frozen_runtime_gate_max_abs_diff",
    }
)
OFFICIAL_GOLDEN_KIND = (
    "official_space_examples_plus_observed_official_service_oracle_"
    "not_author_published_numeric_golden"
)
OFFICIAL_INFERENCE_SOURCE = "space/model/omniaid-dino.py automatic-router forward"

_T2_EXACT_KEYS = frozenset(
    {
        "t2",
        "localization",
        "localisation",
        "localization_metrics",
        "localisation_metrics",
        "attention_map",
        "attention_map_path",
        "score_map",
        "score_map_path",
        "predicted_mask",
        "predicted_mask_path",
        "pixel_metrics",
        "pixel_auroc",
        "pixel_ap",
        "iou",
        "miou",
        "dice",
        "pixel_f1",
        "mcc",
        "s_joint",
        "joint_score",
        "joint_metrics",
    }
)
_T2_PREFIXES = (
    "localization_",
    "localisation_",
    "attention_map",
    "score_map",
    "predicted_mask",
    "heatmap",
    "mask_",
)

_SMOKE_IGNORED_RESULT_FIELDS = frozenset(
    {
        "run_id",
        "run_manifest_fingerprint",
        "config_fingerprint",
        "completed_at",
        "preprocess_latency_ms",
        "latency_ms",
        "peak_cuda_memory_bytes",
        "artifact_path",
        "artifact_paths",
    }
)


@dataclass(frozen=True)
class OmniAIDArtifact:
    """One independently validated canonical six-array NPZ."""

    sample_id: str
    path: Path
    file_sha256: str
    file_bytes: int
    array_sha256: Mapping[str, str]
    arrays: Mapping[str, np.ndarray]


@dataclass(frozen=True)
class RunBundle:
    """All validated immutable and append-only evidence for one run."""

    run_id: str
    fingerprint: str
    mode: str
    run_dir: Path
    manifest_path: Path
    results_path: Path
    expected_path: Path
    summary_path: Path
    manifest: Mapping[str, Any]
    immutable: Mapping[str, Any]
    release: CanonicalRelease
    selected: tuple[dict[str, Any], ...]
    contract: RunDatasetContract
    physical_results: tuple[dict[str, Any], ...]
    latest_results: tuple[dict[str, Any], ...]
    coverage: Mapping[str, Any]
    artifact_root: Path
    artifact_dir: Path
    artifacts: Mapping[str, OmniAIDArtifact]
    evidence_snapshot: Mapping[str, str]


def _runner() -> Any:
    return importlib.import_module("eval.opensource.run_omniaid_balanced")


def _fingerprint(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(stable_json(value).encode("utf-8")).hexdigest()


def _rows_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    payload = "".join(f"{stable_json(row)}\n" for row in rows).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _array_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes(order="C")).hexdigest()


def _npz_bytes(arrays: Mapping[str, np.ndarray]) -> bytes:
    if list(arrays) != list(ARTIFACT_SCHEMA):
        raise ValueError("OmniAID NPZ array order changed")
    handle = io.BytesIO()
    np.savez(
        handle,
        **{key: np.ascontiguousarray(value) for key, value in arrays.items()},
    )
    return handle.getvalue()


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value}")


def _require_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} is not an object")
    return dict(value)


def _require_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} is not a non-empty string")
    return value


def _require_sha256(value: Any, label: str) -> str:
    result = _require_string(value, label)
    if len(result) != 64 or any(
        character not in "0123456789abcdef" for character in result
    ):
        raise ValueError(f"{label} is not a lowercase SHA-256")
    return result


def _require_finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(
        value,
        (int, float, np.integer, np.floating),
    ):
        raise ValueError(f"{label} is not a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} is not finite")
    return result


def _require_nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} is not a non-negative integer")
    return value


def _same_json_type_and_value(actual: Any, expected: Any) -> bool:
    if type(actual) is not type(expected):
        return False
    if isinstance(actual, dict):
        return set(actual) == set(expected) and all(
            _same_json_type_and_value(actual[key], expected[key]) for key in actual
        )
    if isinstance(actual, list):
        return len(actual) == len(expected) and all(
            _same_json_type_and_value(left, right)
            for left, right in zip(actual, expected, strict=True)
        )
    return bool(actual == expected)


def _require_regular_file(path: Path, label: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"missing or unsafe {label}: {path}")
    return path


def _load_json(path: Path, label: str) -> dict[str, Any]:
    _require_regular_file(path, label)
    try:
        text = path.read_text(encoding="utf-8")
        value = json.loads(
            text,
            object_pairs_hook=_strict_object,
            parse_constant=_reject_json_constant,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is invalid JSON") from error
    result = _require_mapping(value, label)
    if text != json.dumps(result, ensure_ascii=False, indent=2) + "\n":
        raise ValueError(f"{label} is not canonical JSON")
    return result


def _read_jsonl_strict(path: Path, label: str) -> list[dict[str, Any]]:
    _require_regular_file(path, label)
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                row_label = f"{label}:{line_number}"
                if not line.endswith("\n"):
                    raise ValueError(f"{row_label} lacks final newline")
                if not line.strip():
                    raise ValueError(f"{row_label} is blank")
                value = json.loads(
                    line,
                    object_pairs_hook=_strict_object,
                    parse_constant=_reject_json_constant,
                )
                row = _require_mapping(value, row_label)
                if line != f"{stable_json(row)}\n":
                    raise ValueError(f"{row_label} is not canonical JSONL")
                rows.append(row)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is invalid JSONL") from error
    return rows


def forbidden_t2_claims(
    value: Any,
    path: tuple[str, ...] = (),
) -> set[str]:
    found: set[str] = set()
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key)
            normalized = key.lower()
            rendered = ".".join((*path, key))
            if normalized == "valid_for_t2":
                if child is not False:
                    found.add(rendered)
            elif normalized == "localization_output":
                if child is not None:
                    found.add(rendered)
            elif normalized == "native_dense_output":
                if child is not False:
                    found.add(rendered)
            elif (
                normalized == "pair_rank"
                or normalized in _T2_EXACT_KEYS
                or normalized.startswith(_T2_PREFIXES)
            ):
                found.add(rendered)
            found.update(forbidden_t2_claims(child, (*path, key)))
    elif isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        for index, child in enumerate(value):
            found.update(forbidden_t2_claims(child, (*path, str(index))))
    return found


def _reject_t2(value: Any, label: str) -> None:
    found = forbidden_t2_claims(value)
    if found:
        raise ValueError(f"{label} invents OmniAID T2 output {sorted(found)[0]!r}")


def _valid_run_id(value: Any) -> str:
    if (
        not isinstance(value, str)
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,159}", value) is None
        or Path(value).name != value
        or value in (".", "..")
    ):
        raise ValueError("run-id is not one safe ASCII path component")
    return value


def _lexical_absolute(path: Path, *, base: Path | None = None) -> Path:
    raw = path
    if not raw.is_absolute():
        raw = (base if base is not None else Path.cwd()) / raw
    return Path(os.path.abspath(raw))


def _safe_repo_path(
    value: Any,
    *,
    repo_root: Path,
    label: str,
    require_file: bool = True,
) -> Path:
    raw = Path(_require_string(value, label))
    root = repo_root.resolve()
    absolute = _lexical_absolute(raw, base=root)
    try:
        relative = absolute.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{label} escapes repository root") from error
    current = root
    for part in relative.parts:
        current /= part
        if current.is_symlink():
            raise ValueError(f"{label} contains a symlink")
    if require_file:
        _require_regular_file(absolute, label)
    return absolute


def _safe_standard_root(
    requested: Path,
    *,
    repo_root: Path,
    expected_relative: Path,
    label: str,
) -> Path:
    candidate = _lexical_absolute(requested, base=repo_root)
    expected = _lexical_absolute(expected_relative, base=repo_root)
    if candidate != expected:
        raise ValueError(f"{label} must be the standard path {expected}")
    _safe_repo_path(
        repo_relative(candidate, repo_root),
        repo_root=repo_root,
        label=label,
        require_file=False,
    )
    return candidate


def _resolve_run_dir(results_root: Path, run_id: str) -> Path:
    safe_id = _valid_run_id(run_id)
    run_dir = _lexical_absolute(results_root / safe_id)
    if run_dir.parent != _lexical_absolute(results_root):
        raise ValueError("OmniAID run directory escapes results root")
    return run_dir


def _assert_runner_contract_exports() -> Any:
    """Bind the analyzer to the frozen runner API without trusting its data."""

    runner = _runner()
    expected = {
        "RUN_MANIFEST_SCHEMA": EXPECTED_RUN_MANIFEST_SCHEMA,
        "RUN_CONFIG_SCHEMA": EXPECTED_RUN_CONFIG_SCHEMA,
        "RUNTIME_SUMMARY_SCHEMA": EXPECTED_RUNTIME_SUMMARY_SCHEMA,
        "CPU_PREFLIGHT_SCHEMA": EXPECTED_CPU_PREFLIGHT_SCHEMA,
        "DEFAULT_FORMAL_RUN_ID": DEFAULT_FORMAL_RUN_ID,
        "DEFAULT_SMOKE_RUN_ID_A": DEFAULT_SMOKE_RUN_ID_A,
        "DEFAULT_SMOKE_RUN_ID_B": DEFAULT_SMOKE_RUN_ID_B,
        "FORMAL_SELECTED_ROWS_SHA256": FORMAL_SELECTED_ROWS_SHA256,
        "FORMAL_SELECTED_IDS_SHA256": FORMAL_SELECTED_IDS_SHA256,
        "SMOKE5X7_SELECTED_IDS_SHA256": SMOKE5X7_SELECTED_IDS_SHA256,
        "ARTIFACT_FILE_BYTES": ARTIFACT_FILE_BYTES,
        "FROZEN_RUNTIME_VERSIONS": FROZEN_RUNTIME_VERSIONS,
    }
    for name, value in expected.items():
        if not _same_json_type_and_value(getattr(runner, name, None), value):
            raise ValueError(f"OmniAID runner export {name} drifted")
    if (
        tuple(runner.ADAPTER_SOURCE_PATHS) != EXPECTED_ADAPTER_SOURCE_PATHS
        or frozenset(runner.IMMUTABLE_CONFIG_KEYS) != EXPECTED_IMMUTABLE_KEYS
        or frozenset(runner._OK_RESULT_FIELDS) != EXPECTED_OK_RESULT_FIELDS
        or frozenset(runner._ERROR_RESULT_FIELDS) != EXPECTED_ERROR_RESULT_FIELDS
        or list(runner.legacy.ARTIFACT_SCHEMA) != list(ARTIFACT_SCHEMA)
        or runner.SCORE_SPEC.as_dict()
        != {
            "key": "ai_score",
            "direction": "higher_means_fake",
            "fixed_threshold": 0.5,
            "threshold_operator": ">",
        }
        or runner.TASK_SCOPE
        != {
            "primary_task": "T1_whole_image_AIGC_detection",
            "valid_for_t1": True,
            "valid_for_t2": False,
            "localization_output": None,
            "native_dense_output": False,
        }
        or runner.ARTIFACT_CONTRACT["zip_members"] != ARTIFACT_MEMBER_BYTES
    ):
        raise ValueError("OmniAID runner/analyzer contract drifted")
    _reject_t2(runner.TASK_SCOPE, "runner task scope")
    return runner


def _verify_adapter_sources(
    value: Any,
    *,
    repo_root: Path,
) -> dict[str, dict[str, Any]]:
    recorded = _require_mapping(value, "immutable.adapter_sources")
    if tuple(recorded) != EXPECTED_ADAPTER_SOURCE_PATHS:
        raise ValueError("OmniAID adapter source inventory changed")
    verified: dict[str, dict[str, Any]] = {}
    for relative in EXPECTED_ADAPTER_SOURCE_PATHS:
        row = _require_mapping(
            recorded.get(relative),
            f"adapter source {relative}",
        )
        path = _safe_repo_path(
            relative,
            repo_root=repo_root,
            label=f"adapter source {relative}",
        )
        expected = {
            "path": relative,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        if not _same_json_type_and_value(row, expected):
            raise ValueError(f"OmniAID adapter source changed: {relative}")
        verified[relative] = expected
    return verified


def _formal_selection(
    release: CanonicalRelease,
) -> tuple[SelectionSpec, list[dict[str, Any]]]:
    if (
        release.schema_version != BALANCED_SCHEMA
        or release.dataset_id != BALANCED_DATASET_ID
        or release.release_kind != "balanced250"
    ):
        raise ValueError("OmniAID formal release identity changed")
    spec = SelectionSpec(capability=Capability.WHOLE_IMAGE_T1)
    selected = [
        row
        for row in release.inputs
        if row.get("condition") in Capability.WHOLE_IMAGE_T1.conditions
    ]
    counts = Counter(str(row["condition"]) for row in selected)
    if (
        len(selected) != FORMAL_IMAGES
        or dict(counts) != FORMAL_COUNTS
        or any("pair_rank" in row for row in selected)
        or _rows_sha256(selected) != FORMAL_SELECTED_ROWS_SHA256
        or selected_ids_sha256(str(row["sample_id"]) for row in selected)
        != FORMAL_SELECTED_IDS_SHA256
    ):
        raise ValueError("OmniAID formal selection changed")
    return spec, selected


def _smoke_selection(
    release: CanonicalRelease,
) -> tuple[SelectionSpec, list[dict[str, Any]]]:
    spec = SelectionSpec(
        capability=Capability.WHOLE_IMAGE_T1,
        per_condition_limit=SMOKE_PER_CONDITION,
    )
    inputs = {str(row["sample_id"]): row for row in release.inputs}
    counts: Counter[str] = Counter()
    selected: list[dict[str, Any]] = []
    for panel_row in release.panel:
        condition = str(panel_row["condition"])
        if condition in BALANCED_CONDITIONS and counts[condition] < SMOKE_PER_CONDITION:
            sample_id = str(panel_row["sample_id"])
            row = inputs.get(sample_id)
            if row is None or row.get("panel") is not True:
                raise ValueError("OmniAID smoke panel reference changed")
            selected.append(row)
            counts[condition] += 1
    selected.sort(key=lambda row: int(row["rank"]))
    if (
        len(selected) != SMOKE_IMAGES
        or dict(counts)
        != {condition: SMOKE_PER_CONDITION for condition in BALANCED_CONDITIONS}
        or any("pair_rank" in row for row in selected)
        or selected_ids_sha256(str(row["sample_id"]) for row in selected)
        != SMOKE5X7_SELECTED_IDS_SHA256
    ):
        raise ValueError("OmniAID smoke selection changed")
    return spec, selected


def _visibility_diagnostic(row: Mapping[str, Any]) -> dict[str, Any]:
    width = row.get("width")
    height = row.get("height")
    if (
        isinstance(width, bool)
        or not isinstance(width, int)
        or width <= 0
        or isinstance(height, bool)
        or not isinstance(height, int)
        or height <= 0
    ):
        raise ValueError("OmniAID input dimensions changed")
    gt_kind = row.get("gt_mask_kind")
    if gt_kind == "exact_diff":
        return {
            "edit_visibility": "full",
            "edit_visible_gt_fraction": 1.0,
            "edit_visibility_evidence": {
                "basis": "full_canvas_direct_resize_without_crop",
                "definition": (
                    "every_native_pixel_is_in_the_geometric_model_input_"
                    "domain_after_direct_resize"
                ),
                "native_width": width,
                "native_height": height,
                "model_input_wh": [448, 448],
                "resize_preserves_aspect_ratio": False,
                "crop": None,
                "preprocess_profile": legacy.PREPROCESS_PROFILE,
            },
        }
    if gt_kind == "all_zero":
        basis = "authentic_input_has_all_zero_GT"
    elif gt_kind == "not_applicable":
        basis = "conditional_full_frame_edit_has_no_local_GT"
    else:
        raise ValueError("OmniAID GT kind changed")
    return {
        "edit_visibility": "not_applicable",
        "edit_visible_gt_fraction": None,
        "edit_visibility_evidence": {
            "basis": basis,
            "native_width": width,
            "native_height": height,
            "model_input_wh": [448, 448],
            "crop": None,
            "preprocess_profile": legacy.PREPROCESS_PROFILE,
        },
    }


def _visibility_census(
    selected: Sequence[Mapping[str, Any]],
) -> dict[str, int]:
    counts = Counter(_visibility_diagnostic(row)["edit_visibility"] for row in selected)
    result = {
        key: int(counts[key]) for key in ("full", "not_applicable") if counts[key]
    }
    if len(selected) == FORMAL_IMAGES and result != {
        "full": 750,
        "not_applicable": 1025,
    }:
        raise ValueError("OmniAID formal visibility census changed")
    if sum(result.values()) != len(selected):
        raise ValueError("OmniAID visibility census is incomplete")
    return result


def _expected_identity(
    row: Mapping[str, Any],
    *,
    repo_root: Path,
    run_id: str,
    fingerprint: str,
    valid_for_metrics: bool,
) -> dict[str, Any]:
    path = _safe_repo_path(
        row.get("canonical_path"),
        repo_root=repo_root,
        label=f"{row.get('sample_id')} canonical input",
    )
    result = {
        **build_result_identity(
            row,
            run_id=run_id,
            run_manifest_fingerprint=fingerprint,
        ),
        "valid_for_metrics": valid_for_metrics,
        "input_path": repo_relative(path, repo_root),
        "model": legacy.MODEL_NAME,
        "model_slug": legacy.MODEL_SLUG,
        "checkpoint_id": legacy.CHECKPOINT["id"],
        "preprocess_profile": legacy.PREPROCESS_PROFILE,
        "score_semantics": legacy.SCORE_SEMANTICS,
        "classification_threshold": 0.5,
        "classification_threshold_operator": ">",
        "config_fingerprint": fingerprint,
        **_visibility_diagnostic(row),
        "valid_for_t1": True,
        "valid_for_t2": False,
        "task_scope": {
            "valid_for_t1": True,
            "valid_for_t2": False,
            "localization_output": None,
            "native_dense_output": False,
        },
    }
    if "pair_rank" in result:
        raise ValueError("OmniAID identity invented pair_rank")
    return result


def _validate_gate_arrays(
    indices: np.ndarray,
    gates: np.ndarray,
    final: np.ndarray,
    *,
    label: str,
) -> None:
    legacy_audit._validate_gate_arrays(
        indices,
        gates,
        final,
        label=label,
    )


def _validate_score_payload(
    row: Mapping[str, Any],
    *,
    sample_id: str,
    arrays: Mapping[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    score = _require_finite(row.get("ai_score"), f"{sample_id}.ai_score")
    if not 0.0 <= score <= 1.0:
        raise ValueError(f"{sample_id} score is outside [0,1]")
    for alias in ("fake_probability", "probability", "score"):
        if _require_finite(row.get(alias), f"{sample_id}.{alias}") != score:
            raise ValueError(f"{sample_id} score alias {alias} changed")
    logits_raw = row.get("class_logits")
    if not isinstance(logits_raw, list) or len(logits_raw) != 2:
        raise ValueError(f"{sample_id} class logits changed")
    logits = np.asarray(
        [_require_finite(value, f"{sample_id}.class_logits") for value in logits_raw],
        dtype=np.float32,
    )
    margin = float(np.float32(logits[1] - logits[0]))
    if (
        _require_finite(
            row.get("raw_logit_margin"),
            f"{sample_id}.raw_logit_margin",
        )
        != margin
    ):
        raise ValueError(f"{sample_id} raw logit margin changed")
    decision = score > 0.5
    expected = {
        "score_semantics": legacy.SCORE_SEMANTICS,
        "classification_decision": decision,
        "classification_threshold": 0.5,
        "classification_threshold_operator": ">",
        "routing_mode": "Auto (Router)",
        "semantic_expert_names": [
            "Human",
            "Animal",
            "Object",
            "Scene",
            "Anime",
        ],
        "artifact_expert_name": "Artifact",
        "classification": {
            "decision": decision,
            "threshold": 0.5,
            "operator": ">",
        },
        "t1": {"valid": True, "score": score, "decision": decision},
        "manual_replay": {
            "head_logits_exact": True,
            "softmax_dtype": "float32",
            "fake_class_index": 1,
            "router_scatter_exact": True,
        },
    }
    for key, value in expected.items():
        if not _same_json_type_and_value(row.get(key), value):
            raise ValueError(f"{sample_id} scoring field {key} changed")
    indices_raw = row.get("semantic_top_k_indices")
    gates_raw = row.get("semantic_top_k_gates")
    final_raw = row.get("final_expert_gates")
    if (
        not isinstance(indices_raw, list)
        or len(indices_raw) != 2
        or any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in indices_raw
        )
        or not isinstance(gates_raw, list)
        or len(gates_raw) != 2
        or not isinstance(final_raw, list)
        or len(final_raw) != 6
    ):
        raise ValueError(f"{sample_id} router payload changed")
    indices = np.asarray(indices_raw, dtype=np.int64)
    gates = np.asarray(
        [_require_finite(value, f"{sample_id}.top_gate") for value in gates_raw],
        dtype=np.float32,
    )
    final = np.asarray(
        [_require_finite(value, f"{sample_id}.final_gate") for value in final_raw],
        dtype=np.float32,
    )
    _validate_gate_arrays(indices, gates, final, label=sample_id)
    if row.get("semantic_gate_sum") != float(
        final[:5].sum(dtype=np.float32)
    ) or row.get("final_gate_sum") != float(final.sum(dtype=np.float32)):
        raise ValueError(f"{sample_id} gate sum changed")
    if arrays is not None:
        expected_arrays = {
            "class_logits": [float(value) for value in arrays["class_logits"]],
            "semantic_top_k_indices": [
                int(value) for value in arrays["semantic_top_k_indices"]
            ],
            "semantic_top_k_gates": [
                float(value) for value in arrays["semantic_top_k_gates"]
            ],
            "final_expert_gates": [float(value) for value in arrays["final_gates"]],
        }
        for key, value in expected_arrays.items():
            if row.get(key) != value:
                raise ValueError(f"{sample_id} {key} differs from artifact")
    _reject_t2(row, f"result {sample_id}")
    return {
        "score": score,
        "margin": margin,
        "decision": decision,
        "indices": indices,
        "gates": gates,
        "final": final,
    }


def _load_npz_artifact(
    *,
    row: Mapping[str, Any],
    sample_id: str,
    repo_root: Path,
    artifact_dir: Path,
) -> OmniAIDArtifact:
    recorded_path = _require_string(
        row.get("artifact_path"),
        f"{sample_id}.artifact_path",
    )
    path = _safe_repo_path(
        recorded_path,
        repo_root=repo_root,
        label=f"{sample_id} artifact",
    )
    expected_path = artifact_dir / f"{sample_id}.npz"
    if path != expected_path:
        raise ValueError(f"{sample_id} artifact path is not canonical")
    payload = path.read_bytes()
    if len(payload) != ARTIFACT_FILE_BYTES:
        raise ValueError(f"{sample_id} artifact byte size changed")
    members = list(ARTIFACT_MEMBER_BYTES)
    try:
        with zipfile.ZipFile(io.BytesIO(payload), mode="r") as archive:
            infos = archive.infolist()
            if [
                info.filename for info in infos
            ] != members or archive.testzip() is not None:
                raise ValueError(f"{sample_id} NPZ inventory changed")
            for info in infos:
                pure = PurePosixPath(info.filename)
                if (
                    pure.is_absolute()
                    or len(pure.parts) != 1
                    or any(part in ("", ".", "..") for part in pure.parts)
                    or info.compress_type != zipfile.ZIP_STORED
                    or info.file_size != ARTIFACT_MEMBER_BYTES[info.filename]
                    or info.compress_size != info.file_size
                    or info.date_time != (1980, 1, 1, 0, 0, 0)
                ):
                    raise ValueError(f"{sample_id} NPZ member {info.filename} changed")
    except (OSError, zipfile.BadZipFile) as error:
        raise ValueError(f"{sample_id} artifact is not a safe NPZ") from error
    try:
        with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
            if archive.files != list(ARTIFACT_SCHEMA):
                raise ValueError(f"{sample_id} NPZ array order changed")
            arrays = {
                key: np.ascontiguousarray(archive[key]) for key in ARTIFACT_SCHEMA
            }
    except ValueError:
        raise
    except Exception as error:
        raise ValueError(f"{sample_id} NPZ arrays cannot be loaded") from error
    for key, (shape, dtype) in ARTIFACT_SCHEMA.items():
        array = arrays[key]
        if (
            array.shape != shape
            or array.dtype != dtype
            or not array.flags.c_contiguous
            or (np.issubdtype(dtype, np.floating) and not np.isfinite(array).all())
        ):
            raise ValueError(f"{sample_id} artifact array {key} changed")
    if payload != _npz_bytes(arrays):
        raise ValueError(f"{sample_id} artifact is not canonical NPZ bytes")
    _validate_gate_arrays(
        arrays["semantic_top_k_indices"],
        arrays["semantic_top_k_gates"],
        arrays["final_gates"],
        label=f"{sample_id} artifact",
    )
    file_sha = hashlib.sha256(payload).hexdigest()
    hashes = {key: _array_sha256(value) for key, value in arrays.items()}
    relative = repo_relative(path, repo_root)
    expected_record = {
        "artifact_path": relative,
        "artifact_sha256": file_sha,
        "artifact_bytes": ARTIFACT_FILE_BYTES,
        "artifact_keys": list(ARTIFACT_SCHEMA),
        "artifact_paths": {"omniaid_npz": relative},
        "artifact_array_sha256": hashes,
    }
    for key, value in expected_record.items():
        if not _same_json_type_and_value(row.get(key), value):
            raise ValueError(f"{sample_id} artifact record {key} changed")
    expected_metadata = {
        "feature_shape": [legacy.FEATURE_DIMENSION],
        "feature_dtype": "float32",
        "feature_semantics": legacy.FEATURE_SEMANTICS,
        "feature_array_sha256": hashes["pooler_output"],
        "class_logits_shape": [legacy.CLASS_COUNT],
        "class_logits_dtype": "float32",
        "class_logits_array_sha256": hashes["class_logits"],
        "routing_feature_shape": [legacy.FEATURE_DIMENSION],
        "routing_feature_dtype": "float32",
        "routing_feature_semantics": legacy.ROUTING_FEATURE_SEMANTICS,
        "semantic_top_k_indices_shape": [legacy.SEMANTIC_TOP_K],
        "semantic_top_k_indices_dtype": "int64",
        "semantic_top_k_gates_shape": [legacy.SEMANTIC_TOP_K],
        "semantic_top_k_gates_dtype": "float32",
        "final_gates_shape": [legacy.EXPERT_COUNT],
        "final_gates_dtype": "float32",
    }
    for key, value in expected_metadata.items():
        if not _same_json_type_and_value(row.get(key), value):
            raise ValueError(f"{sample_id} artifact metadata {key} changed")
    _validate_score_payload(row, sample_id=sample_id, arrays=arrays)
    return OmniAIDArtifact(
        sample_id=sample_id,
        path=path,
        file_sha256=file_sha,
        file_bytes=len(payload),
        array_sha256=hashes,
        arrays=arrays,
    )


def validate_artifact_inventory(
    *,
    latest_results: Sequence[Mapping[str, Any]],
    repo_root: Path,
    artifact_root: Path,
    artifact_dir: Path,
) -> dict[str, OmniAIDArtifact]:
    if (
        artifact_root.is_symlink()
        or not artifact_root.is_dir()
        or artifact_dir.is_symlink()
        or not artifact_dir.is_dir()
        or artifact_dir.parent != artifact_root
    ):
        raise ValueError("OmniAID artifact root is missing or unsafe")
    root_entries = list(artifact_root.iterdir())
    if (
        len(root_entries) != 1
        or root_entries[0] != artifact_dir
        or root_entries[0].is_symlink()
    ):
        raise ValueError("OmniAID artifact root has extra entries")
    artifacts: dict[str, OmniAIDArtifact] = {}
    for row in latest_results:
        sample_id = _require_string(
            row.get("sample_id"),
            "artifact result sample_id",
        )
        if row.get("status") != "ok":
            raise ValueError(f"{sample_id} has no valid artifact result")
        if sample_id in artifacts:
            raise ValueError(f"OmniAID artifact sample repeats {sample_id}")
        artifacts[sample_id] = _load_npz_artifact(
            row=row,
            sample_id=sample_id,
            repo_root=repo_root,
            artifact_dir=artifact_dir,
        )
    entries = list(artifact_dir.iterdir())
    if any(entry.is_symlink() or not entry.is_file() for entry in entries):
        raise ValueError("OmniAID artifact directory has an unsafe entry")
    if {entry.resolve() for entry in entries} != {
        artifact.path for artifact in artifacts.values()
    }:
        raise ValueError("OmniAID artifact inventory coverage changed")
    return artifacts


def _artifact_inventory_sha256(
    artifacts: Mapping[str, OmniAIDArtifact],
) -> str:
    value = {
        sample_id: {
            "relative_path": artifact.path.as_posix(),
            "file_sha256": artifact.file_sha256,
            "file_bytes": artifact.file_bytes,
            "array_sha256": dict(artifact.array_sha256),
        }
        for sample_id, artifact in sorted(artifacts.items())
    }
    return _fingerprint(value)


def _validate_runtime_contract(
    value: Any,
    *,
    label: str,
) -> dict[str, Any]:
    runner = _assert_runner_contract_exports()
    runtime = _require_mapping(value, label)
    runner.validate_runtime_contract(runtime, label=label)
    if runtime.get("versions") != FROZEN_RUNTIME_VERSIONS:
        raise ValueError(f"{label} package versions changed")
    modules = _require_mapping(runtime.get("module_files"), f"{label}.modules")
    for name, digest in FROZEN_RUNTIME_MODULE_HASHES.items():
        record = _require_mapping(modules.get(name), f"{label}.{name}")
        path = Path(_require_string(record.get("path"), f"{label}.{name}.path"))
        if (
            path.is_symlink()
            or not path.is_file()
            or record.get("sha256") != digest
            or sha256_file(path) != digest
        ):
            raise ValueError(f"{label} module {name} changed")
    return runtime


def _validate_attempt(
    attempt: Mapping[str, Any],
    *,
    input_row: Mapping[str, Any],
    repo_root: Path,
    run_id: str,
    fingerprint: str,
) -> None:
    status = attempt.get("status")
    if status not in ("ok", "error"):
        raise ValueError("OmniAID attempt status changed")
    expected = _expected_identity(
        input_row,
        repo_root=repo_root,
        run_id=run_id,
        fingerprint=fingerprint,
        valid_for_metrics=status == "ok",
    )
    expected_keys = set(expected) | {"status", "completed_at"}
    expected_keys.update(
        EXPECTED_OK_RESULT_FIELDS if status == "ok" else EXPECTED_ERROR_RESULT_FIELDS
    )
    if set(attempt) != expected_keys:
        raise ValueError(
            "OmniAID attempt key set changed: "
            f"missing={sorted(expected_keys - set(attempt))[:1]}, "
            f"extra={sorted(set(attempt) - expected_keys)[:1]}"
        )
    for key, value in expected.items():
        if not _same_json_type_and_value(attempt.get(key), value):
            raise ValueError(f"OmniAID attempt identity field {key} changed")
    if not isinstance(attempt.get("completed_at"), str) or not attempt["completed_at"]:
        raise ValueError("OmniAID attempt completed_at changed")
    _reject_t2(attempt, "OmniAID attempt")
    if status == "error":
        nullable = (
            "class_logits",
            "raw_logit_margin",
            "fake_probability",
            "probability",
            "ai_score",
            "score",
            "classification_decision",
            "peak_cuda_memory_bytes",
        )
        if (
            any(attempt.get(key) is not None for key in nullable)
            or attempt.get("latency_ms") != 0.0
            or not isinstance(attempt.get("error_type"), str)
            or not attempt["error_type"]
            or not isinstance(attempt.get("error"), str)
            or not isinstance(attempt.get("traceback"), str)
            or not attempt["traceback"]
        ):
            raise ValueError("OmniAID error attempt payload changed")
        return
    sample_id = str(input_row["sample_id"])
    input_path = _safe_repo_path(
        attempt.get("input_path"),
        repo_root=repo_root,
        label=f"{sample_id} input",
    )
    if (
        sha256_file(input_path) != input_row.get("canonical_sha256")
        or input_path.stat().st_size <= 0
    ):
        raise ValueError(f"{sample_id} canonical input changed")
    _tensor, preprocess = legacy_audit._preprocess(input_path)
    if not _same_json_type_and_value(attempt.get("preprocess"), preprocess):
        raise ValueError(f"{sample_id} preprocessing record changed")
    if preprocess.get("native_width") != input_row.get("width") or preprocess.get(
        "native_height"
    ) != input_row.get("height"):
        raise ValueError(f"{sample_id} native dimensions changed")
    for field in ("preprocess_latency_ms", "latency_ms"):
        if _require_finite(attempt.get(field), f"{sample_id}.{field}") < 0.0:
            raise ValueError(f"{sample_id} {field} is negative")
    peak = attempt.get("peak_cuda_memory_bytes")
    if peak is not None:
        _require_nonnegative_int(peak, f"{sample_id}.peak_cuda_memory_bytes")
    _validate_score_payload(attempt, sample_id=sample_id)


def _validate_physical_attempts(
    *,
    selected: Sequence[Mapping[str, Any]],
    physical: Sequence[Mapping[str, Any]],
    repo_root: Path,
    run_id: str,
    fingerprint: str,
) -> Any:
    latest = index_latest_attempts(
        selected,
        physical,
        run_id=run_id,
        run_manifest_fingerprint=fingerprint,
        score_spec=ScoreSpec(
            key="ai_score",
            direction="higher_means_fake",
            fixed_threshold=0.5,
            threshold_operator=">",
        ),
    )
    inputs = {str(row["sample_id"]): row for row in selected}
    successful: set[str] = set()
    for attempt in physical:
        sample_id = _require_string(
            attempt.get("sample_id"),
            "physical attempt sample_id",
        )
        if sample_id in successful:
            raise ValueError(f"OmniAID attempt appears after success for {sample_id}")
        _validate_attempt(
            attempt,
            input_row=inputs[sample_id],
            repo_root=repo_root,
            run_id=run_id,
            fingerprint=fingerprint,
        )
        if attempt.get("status") == "ok":
            successful.add(sample_id)
    return latest


def _rebuild_contract(
    *,
    repo_root: Path,
    immutable: Mapping[str, Any],
    expected_mode: str,
) -> tuple[
    CanonicalRelease,
    list[dict[str, Any]],
    RunDatasetContract,
]:
    recorded_contract = _require_mapping(
        immutable.get("dataset_contract"),
        "immutable.dataset_contract",
    )
    release_record = _require_mapping(
        recorded_contract.get("release"),
        "dataset_contract.release",
    )
    manifest_path = _safe_repo_path(
        release_record.get("manifest_path"),
        repo_root=repo_root,
        label="Balanced250 dataset manifest",
    )
    canonical_manifest = _lexical_absolute(
        DEFAULT_DATASET_MANIFEST,
        base=repo_root,
    )
    if manifest_path != canonical_manifest:
        raise ValueError("OmniAID dataset manifest path is not canonical")
    release = load_canonical_release(
        repo_root,
        manifest_path,
        verify_files=True,
    )
    if expected_mode == "formal":
        spec, selected = _formal_selection(release)
    elif expected_mode == "smoke":
        spec, selected = _smoke_selection(release)
    else:
        raise ValueError("OmniAID analyzer supports formal/smoke only")
    contract = build_run_dataset_contract(
        release,
        spec,
        selected,
        score_spec=ScoreSpec(
            key="ai_score",
            direction="higher_means_fake",
            fixed_threshold=0.5,
            threshold_operator=">",
        ),
    )
    if not _same_json_type_and_value(
        recorded_contract,
        contract.as_dict(),
    ):
        raise ValueError("OmniAID dataset contract changed")
    return release, selected, contract


def _validate_recorded_official_golden(
    value: Any,
    *,
    device_family: str,
    label: str,
) -> dict[str, Any]:
    """Validate the complete legacy official-service golden envelope."""

    report = _require_mapping(value, label)
    cases = report.get("cases")
    if (
        set(report) != EXPECTED_OFFICIAL_GOLDEN_KEYS
        or report.get("status") != "passed"
        or report.get("kind") != OFFICIAL_GOLDEN_KIND
        or report.get("device_family") != device_family
        or report.get("runtime_abs_tolerance") != legacy.GOLDEN_RUNTIME_ABS_TOLERANCE
        or report.get("official_service_abs_tolerance")
        != legacy.GOLDEN_SERVICE_ABS_TOLERANCE
        or report.get("official_service_observed_at") != "2026-07-25"
        or report.get("mouse_model_scores_computed") != 0
        or not isinstance(cases, list)
        or len(cases) != len(legacy.GOLDEN_CASES)
        or device_family not in {"cpu", "cuda"}
    ):
        raise ValueError(f"{label} schema or identity changed")
    for index, (case_value, frozen) in enumerate(
        zip(cases, legacy.GOLDEN_CASES, strict=True)
    ):
        case_label = f"{label}.cases[{index}]"
        case = _require_mapping(case_value, case_label)
        if (
            set(case) != EXPECTED_OFFICIAL_GOLDEN_CASE_KEYS
            or case.get("path") != frozen["path"]
            or case.get("input_sha256") != frozen["sha256"]
            or case.get("repeat_all_arrays_exact") is not True
            or case.get("observed_official_service_probability")
            != frozen["official_service_probability"]
        ):
            raise ValueError(f"{case_label} identity changed")
        preprocess = _require_mapping(
            case.get("preprocess"),
            f"{case_label}.preprocess",
        )
        for key in (
            "decoded_rgb_sha256",
            "resized_rgb_sha256",
            "tensor_sha256",
        ):
            if preprocess.get(key) != frozen[key]:
                raise ValueError(f"{case_label} preprocessing changed")
        logits_raw = case.get("logits")
        gates_raw = case.get("final_expert_gates")
        hashes = _require_mapping(
            case.get("array_sha256"),
            f"{case_label}.array_sha256",
        )
        if (
            not isinstance(logits_raw, list)
            or len(logits_raw) != legacy.CLASS_COUNT
            or not isinstance(gates_raw, list)
            or len(gates_raw) != legacy.EXPERT_COUNT
            or set(hashes) != set(ARTIFACT_SCHEMA)
        ):
            raise ValueError(f"{case_label} scientific payload changed")
        logits = np.asarray(
            [_require_finite(value, f"{case_label}.logits") for value in logits_raw],
            dtype=np.float64,
        )
        gates = np.asarray(
            [
                _require_finite(value, f"{case_label}.final_expert_gates")
                for value in gates_raw
            ],
            dtype=np.float64,
        )
        for key, digest in hashes.items():
            _require_sha256(digest, f"{case_label}.array_sha256.{key}")
        probability = _require_finite(
            case.get("fake_probability"),
            f"{case_label}.fake_probability",
        )
        if not 0.0 <= probability <= 1.0:
            raise ValueError(f"{case_label} probability changed")
        service_difference = abs(
            probability - float(frozen["official_service_probability"])
        )
        if (
            case.get("official_service_probability_abs_diff") != service_difference
            or service_difference > legacy.GOLDEN_SERVICE_ABS_TOLERANCE
        ):
            raise ValueError(f"{case_label} service oracle changed")
        runtime_values = (
            case.get("frozen_runtime_logit_max_abs_diff"),
            case.get("frozen_runtime_probability_abs_diff"),
            case.get("frozen_runtime_gate_max_abs_diff"),
        )
        if device_family == "cpu":
            if runtime_values != (None, None, None):
                raise ValueError(f"{case_label} CPU runtime diffs changed")
        else:
            expected_runtime_values = (
                float(
                    np.max(
                        np.abs(
                            logits
                            - np.asarray(
                                frozen["cuda_logits"],
                                dtype=np.float64,
                            )
                        )
                    )
                ),
                abs(probability - float(frozen["cuda_probability"])),
                float(
                    np.max(
                        np.abs(
                            gates
                            - np.asarray(
                                frozen["cuda_final_gates"],
                                dtype=np.float64,
                            )
                        )
                    )
                ),
            )
            if (
                runtime_values != expected_runtime_values
                or max(expected_runtime_values) > legacy.GOLDEN_RUNTIME_ABS_TOLERANCE
            ):
                raise ValueError(f"{case_label} CUDA runtime oracle changed")
    return report


def _validate_cpu_preflight(
    value: Any,
    *,
    immutable_source: Mapping[str, Any],
    immutable_assets: Mapping[str, Any],
) -> None:
    wrapper = _require_mapping(value, "immutable.cpu_preflight")
    if (
        set(wrapper)
        != {
            "performed_before_dataset_manifest_load",
            "performed_before_accelerator_configuration",
            "report",
        }
        or wrapper.get("performed_before_dataset_manifest_load") is not True
        or (wrapper.get("performed_before_accelerator_configuration") is not True)
    ):
        raise ValueError("OmniAID CPU preflight ordering changed")
    report = _require_mapping(
        wrapper.get("report"),
        "immutable.cpu_preflight.report",
    )
    expected_keys = {
        "schema_version",
        "status",
        "source",
        "assets",
        "model_load",
        "runtime",
        "official_golden",
        "balanced_golden",
        "cuda_used",
        "cuda_tensor_operations",
        "cuda_initialized_before_cpu_model_load",
        "cuda_initialized_after_cpu_forwards",
        "dataset_manifest_loaded",
    }
    if (
        set(report) != expected_keys
        or report.get("schema_version") != EXPECTED_CPU_PREFLIGHT_SCHEMA
        or report.get("status") != "passed"
        or not _same_json_type_and_value(
            report.get("source"),
            immutable_source,
        )
        or not _same_json_type_and_value(
            report.get("assets"),
            immutable_assets,
        )
        or report.get("cuda_used") is not False
        or report.get("cuda_tensor_operations") is not False
        or report.get("cuda_initialized_before_cpu_model_load") is not False
        or report.get("cuda_initialized_after_cpu_forwards") is not False
        or report.get("dataset_manifest_loaded") is not False
    ):
        raise ValueError("OmniAID CPU preflight evidence changed")
    runtime = _validate_runtime_contract(
        report.get("runtime"),
        label="CPU preflight runtime",
    )
    if runtime.get("device") != "cpu":
        raise ValueError("OmniAID CPU preflight used a non-CPU device")
    load = _require_mapping(report.get("model_load"), "CPU model load")
    if (
        load.get("strict_load") is not True
        or load.get("state_entries") != legacy.CHECKPOINT["tensor_count"]
        or load.get("state_elements") != legacy.CHECKPOINT["state_elements"]
        or load.get("svd_modules") != legacy.SVD_MODULE_COUNT
        or load.get("parameter_count") != 507_041_863
        or load.get("base_weights_downloaded") is not False
        or load.get("eval_mode") is not True
    ):
        raise ValueError("OmniAID CPU model load evidence changed")
    official = _validate_recorded_official_golden(
        report.get("official_golden"),
        device_family="cpu",
        label="CPU official golden",
    )
    balanced = _require_mapping(
        report.get("balanced_golden"),
        "CPU Balanced golden",
    )
    artifact = _require_mapping(
        balanced.get("artifact"),
        "CPU Balanced golden artifact",
    )
    runner = _assert_runner_contract_exports()
    if (
        official.get("status") != "passed"
        or official.get("device_family") != "cpu"
        or not isinstance(official.get("cases"), list)
        or len(official["cases"]) != 4
        or balanced.get("sample_id") != runner.CPU_GOLDEN_SAMPLE_ID
        or balanced.get("image_sha256") != runner.CPU_GOLDEN_IMAGE_SHA256
        or balanced.get("probability") != runner.CPU_GOLDEN_PROBABILITY
        or balanced.get("repeat_byte_exact") is not True
        or artifact.get("artifact_bytes") != ARTIFACT_FILE_BYTES
        or artifact.get("artifact_sha256") != runner.CPU_GOLDEN_ARTIFACT_SHA256
        or artifact.get("array_sha256") != runner.CPU_GOLDEN_ARRAY_SHA256
    ):
        raise ValueError("OmniAID CPU golden evidence changed")


def _validate_immutable(
    *,
    immutable: Mapping[str, Any],
    repo_root: Path,
    run_id: str,
    mode: str,
    selected: Sequence[Mapping[str, Any]],
    contract: RunDatasetContract,
    artifact_root: Path,
    artifact_dir: Path,
    run_dir: Path,
) -> dict[str, Any]:
    runner = _assert_runner_contract_exports()
    if (
        set(immutable) != EXPECTED_IMMUTABLE_KEYS
        or immutable.get("schema_version") != EXPECTED_RUN_CONFIG_SCHEMA
        or immutable.get("run_id") != run_id
        or immutable.get("mode") != mode
    ):
        raise ValueError("OmniAID immutable config identity changed")
    _verify_adapter_sources(
        immutable.get("adapter_sources"),
        repo_root=repo_root,
    )
    for key, expected in (
        ("model", runner.MODEL_CONTRACT),
        ("preprocess", runner.PREPROCESS_CONTRACT),
        ("score_spec", runner.SCORE_SPEC.as_dict()),
        ("task_scope", runner.TASK_SCOPE),
        ("artifact_contract", runner.ARTIFACT_CONTRACT),
    ):
        if not _same_json_type_and_value(immutable.get(key), expected):
            raise ValueError(f"OmniAID immutable {key} changed")
    if (
        immutable.get("selected_rows_sha256") != _rows_sha256(selected)
        or immutable.get("selected_ids_sha256")
        != selected_ids_sha256(str(row["sample_id"]) for row in selected)
        or immutable.get("visibility_census") != _visibility_census(selected)
        or immutable.get("dataset_contract") != contract.as_dict()
    ):
        raise ValueError("OmniAID immutable selection evidence changed")
    source = _require_mapping(immutable.get("source"), "immutable.source")
    assets = _require_mapping(immutable.get("assets"), "immutable.assets")
    github_source = _require_mapping(
        source.get("github"),
        "immutable.source.github",
    )
    space_source = _require_mapping(
        source.get("space"),
        "immutable.source.space",
    )
    if (
        set(source) != {"github", "space", "inference_source"}
        or source.get("inference_source") != OFFICIAL_INFERENCE_SOURCE
        or github_source.get("repository") != legacy.MODEL_REPO_URL
        or github_source.get("commit") != legacy.MODEL_SOURCE_COMMIT
        or space_source.get("repository") != legacy.MODEL_SPACE_URL
        or space_source.get("commit") != legacy.MODEL_SPACE_COMMIT
        or set(assets) != {"checkpoint", "omniaid_config", "dinov3_base"}
        or not _same_json_type_and_value(
            assets.get("dinov3_base"),
            legacy.DINO_BASE,
        )
        or assets.get("checkpoint", {}).get("sha256") != legacy.CHECKPOINT["sha256"]
        or assets.get("checkpoint", {}).get("bytes") != legacy.CHECKPOINT["bytes"]
        or assets.get("omniaid_config", {}).get("sha256")
        != legacy.OMNIAID_CONFIG["sha256"]
    ):
        raise ValueError("OmniAID immutable source/assets changed")
    runtime = _validate_runtime_contract(
        immutable.get("runtime"),
        label="immutable.runtime",
    )
    _validate_cpu_preflight(
        immutable.get("cpu_preflight"),
        immutable_source=source,
        immutable_assets=assets,
    )
    execution_load = _require_mapping(
        immutable.get("execution_model_load"),
        "immutable.execution_model_load",
    )
    preflight_load = immutable["cpu_preflight"]["report"]["model_load"]
    if (
        not _same_json_type_and_value(execution_load, preflight_load)
        or execution_load.get("strict_load") is not True
        or execution_load.get("base_weights_downloaded") is not False
    ):
        raise ValueError("OmniAID execution model-load evidence changed")
    golden = _validate_recorded_official_golden(
        immutable.get("execution_official_golden"),
        device_family=str(runtime["device"]).split(":", 1)[0],
        label="immutable.execution_official_golden",
    )
    policy = _require_mapping(
        immutable.get("artifact_policy"),
        "immutable.artifact_policy",
    )
    expected_policy = runner.artifact_policy_contract(
        repo_root=repo_root,
        artifact_root=artifact_root,
    )
    if not _same_json_type_and_value(policy, expected_policy):
        raise ValueError("OmniAID local artifact policy changed")
    expected_outputs = {
        "run_dir": repo_relative(run_dir, repo_root),
        "results_path": repo_relative(run_dir / "results.jsonl", repo_root),
        "expected_inputs_path": repo_relative(
            run_dir / "expected_inputs.jsonl",
            repo_root,
        ),
        "summary_path": repo_relative(run_dir / "summary.json", repo_root),
        "artifact_root": repo_relative(artifact_root, repo_root),
        "artifact_dir": repo_relative(artifact_dir, repo_root),
    }
    if not _same_json_type_and_value(
        immutable.get("outputs"),
        expected_outputs,
    ):
        raise ValueError("OmniAID immutable output paths are non-standard")
    _reject_t2(immutable, "OmniAID immutable config")
    return runtime


def _validate_manifest_summary(
    *,
    manifest: Mapping[str, Any],
    summary: Mapping[str, Any],
    immutable: Mapping[str, Any],
    fingerprint: str,
    run_id: str,
    mode: str,
    contract: RunDatasetContract,
    selected: Sequence[Mapping[str, Any]],
    physical: Sequence[Mapping[str, Any]],
    latest: Any,
    manifest_path: Path,
    results_path: Path,
    expected_path: Path,
    summary_path: Path,
    release: CanonicalRelease,
    artifact_files: int,
    repo_root: Path,
) -> dict[str, Any]:
    if (
        set(manifest)
        != {
            "schema_version",
            "run_id",
            "status",
            "started_at",
            "completed_at",
            "fingerprint",
            "immutable",
            "dataset",
            "outputs",
            "execution",
        }
        or manifest.get("schema_version") != EXPECTED_RUN_MANIFEST_SCHEMA
        or manifest.get("run_id") != run_id
        or manifest.get("status") != "complete"
        or manifest.get("fingerprint") != fingerprint
        or not _same_json_type_and_value(
            manifest.get("immutable"),
            immutable,
        )
        or not isinstance(manifest.get("started_at"), str)
        or not manifest["started_at"]
        or not isinstance(manifest.get("completed_at"), str)
        or not manifest["completed_at"]
    ):
        raise ValueError("OmniAID finalized manifest changed")
    dataset = _require_mapping(manifest.get("dataset"), "manifest.dataset")
    if (
        set(dataset)
        != {
            "contract",
            "manifest_path",
            "manifest_sha256",
            "expected_inputs_path",
            "expected_inputs_sha256",
            "selected_images",
        }
        or dataset.get("contract") != contract.as_dict()
        or dataset.get("manifest_path")
        != repo_relative(release.manifest_path, repo_root)
        or dataset.get("manifest_sha256") != release.manifest_sha256
        or dataset.get("expected_inputs_path")
        != repo_relative(expected_path, repo_root)
        or dataset.get("expected_inputs_sha256") != sha256_file(expected_path)
        or dataset.get("selected_images") != len(selected)
    ):
        raise ValueError("OmniAID manifest dataset evidence changed")
    outputs = _require_mapping(manifest.get("outputs"), "manifest.outputs")
    expected_output_keys = set(immutable["outputs"]) | {
        "expected_inputs_sha256",
        "results_sha256",
        "summary_sha256",
        "artifact_files",
    }
    if (
        set(outputs) != expected_output_keys
        or any(outputs.get(key) != value for key, value in immutable["outputs"].items())
        or outputs.get("expected_inputs_sha256") != sha256_file(expected_path)
        or outputs.get("results_sha256") != sha256_file(results_path)
        or outputs.get("summary_sha256") != sha256_file(summary_path)
        or outputs.get("artifact_files") != artifact_files
    ):
        raise ValueError("OmniAID manifest output evidence changed")
    execution = _require_mapping(
        manifest.get("execution"),
        "manifest.execution",
    )
    if (
        set(execution)
        != {
            "new_successes",
            "resume_skips",
            "new_errors",
            "prior_physical_result_rows",
            "physical_result_rows",
            "latest_result_rows",
            "superseded_attempts",
        }
        or any(
            _require_nonnegative_int(value, f"execution.{key}") < 0
            for key, value in execution.items()
        )
        or execution.get("physical_result_rows") != len(physical)
        or execution.get("latest_result_rows") != len(latest.latest_by_sample_id)
        or execution.get("superseded_attempts") != latest.superseded_attempts
    ):
        raise ValueError("OmniAID execution accounting changed")
    coverage = summarize_coverage(latest).as_dict()
    expected_summary_keys = {
        "schema_version",
        "summary_kind",
        "scientific_metrics",
        "scientific_metrics_owner",
        "run_id",
        "run_manifest_fingerprint",
        "status",
        "mode",
        "model",
        "model_slug",
        "checkpoint_id",
        "score_spec",
        "task_scope",
        "dataset_contract",
        "coverage",
        "artifact_files",
        "generated_at",
    }
    if (
        set(summary) != expected_summary_keys
        or summary.get("schema_version") != EXPECTED_RUNTIME_SUMMARY_SCHEMA
        or summary.get("summary_kind") != "runtime_coverage_only"
        or summary.get("scientific_metrics") is not None
        or summary.get("scientific_metrics_owner") != "analyze_omniaid_balanced.py"
        or summary.get("run_id") != run_id
        or summary.get("run_manifest_fingerprint") != fingerprint
        or summary.get("status") != "complete"
        or summary.get("mode") != mode
        or summary.get("model") != legacy.MODEL_NAME
        or summary.get("model_slug") != legacy.MODEL_SLUG
        or summary.get("checkpoint_id") != legacy.CHECKPOINT["id"]
        or summary.get("score_spec")
        != {
            "key": "ai_score",
            "direction": "higher_means_fake",
            "fixed_threshold": 0.5,
            "threshold_operator": ">",
        }
        or summary.get("task_scope") != immutable["task_scope"]
        or summary.get("dataset_contract") != contract.as_dict()
        or summary.get("coverage") != coverage
        or summary.get("artifact_files") != artifact_files
        or not isinstance(summary.get("generated_at"), str)
        or not summary["generated_at"]
    ):
        raise ValueError("OmniAID runtime summary changed")
    _reject_t2(manifest, "OmniAID manifest")
    _reject_t2(summary, "OmniAID runtime summary")
    return coverage


def _capture_evidence_snapshot(
    *,
    manifest_path: Path,
    results_path: Path,
    expected_path: Path,
    summary_path: Path,
    release: CanonicalRelease,
    artifacts: Mapping[str, OmniAIDArtifact],
) -> dict[str, str]:
    return {
        "manifest_sha256": sha256_file(manifest_path),
        "results_sha256": sha256_file(results_path),
        "expected_inputs_sha256": sha256_file(expected_path),
        "runtime_summary_sha256": sha256_file(summary_path),
        "dataset_manifest_sha256": release.manifest_sha256,
        "artifact_inventory_sha256": _artifact_inventory_sha256(artifacts),
    }


def _load_run(
    *,
    repo_root: Path,
    results_dir: Path,
    artifacts_dir: Path,
    run_id: str,
    mode: str,
) -> RunBundle:
    _assert_runner_contract_exports()
    root = repo_root.resolve()
    results_root = _safe_standard_root(
        results_dir,
        repo_root=root,
        expected_relative=DEFAULT_RESULTS_DIR,
        label="OmniAID results root",
    )
    artifacts_root = _safe_standard_root(
        artifacts_dir,
        repo_root=root,
        expected_relative=DEFAULT_ARTIFACTS_DIR,
        label="OmniAID artifacts root",
    )
    safe_id = _valid_run_id(run_id)
    if mode == "formal":
        if safe_id != DEFAULT_FORMAL_RUN_ID:
            raise ValueError("formal OmniAID run-id is not frozen")
    elif mode == "smoke":
        if safe_id not in {
            DEFAULT_SMOKE_RUN_ID_A,
            DEFAULT_SMOKE_RUN_ID_B,
        }:
            raise ValueError("smoke OmniAID run-id is not frozen A/B")
    else:
        raise ValueError("OmniAID analyzer mode must be formal or smoke")
    run_dir = _resolve_run_dir(results_root, safe_id)
    artifact_root = _resolve_run_dir(artifacts_root, safe_id)
    if run_dir.is_symlink() or not run_dir.is_dir():
        raise ValueError("OmniAID results run directory is missing or unsafe")
    _safe_repo_path(
        (DEFAULT_RESULTS_DIR / safe_id).as_posix(),
        repo_root=root,
        label="OmniAID results run directory",
        require_file=False,
    )
    if artifact_root.is_symlink() or not artifact_root.is_dir():
        raise ValueError("OmniAID artifact run directory is missing or unsafe")
    _safe_repo_path(
        (DEFAULT_ARTIFACTS_DIR / safe_id).as_posix(),
        repo_root=root,
        label="OmniAID artifact run directory",
        require_file=False,
    )
    artifact_dir = artifact_root / "artifacts"
    manifest_path = run_dir / "manifest.json"
    results_path = run_dir / "results.jsonl"
    expected_path = run_dir / "expected_inputs.jsonl"
    summary_path = run_dir / "summary.json"
    for path, label in (
        (manifest_path, "OmniAID manifest"),
        (results_path, "OmniAID results"),
        (expected_path, "OmniAID expected inputs"),
        (summary_path, "OmniAID runtime summary"),
    ):
        _require_regular_file(path, label)
    manifest = _load_json(manifest_path, "OmniAID manifest")
    summary = _load_json(summary_path, "OmniAID runtime summary")
    if (
        manifest.get("schema_version") != EXPECTED_RUN_MANIFEST_SCHEMA
        or manifest.get("run_id") != safe_id
        or manifest.get("status") != "complete"
    ):
        raise ValueError("OmniAID manifest is not a complete v2 run")
    immutable = _require_mapping(
        manifest.get("immutable"),
        "manifest.immutable",
    )
    fingerprint = _require_sha256(
        manifest.get("fingerprint"),
        "manifest fingerprint",
    )
    if (
        _fingerprint(immutable) != fingerprint
        or immutable.get("run_id") != safe_id
        or immutable.get("mode") != mode
    ):
        raise ValueError("OmniAID immutable fingerprint/mode changed")
    release, selected, contract = _rebuild_contract(
        repo_root=root,
        immutable=immutable,
        expected_mode=mode,
    )
    _validate_immutable(
        immutable=immutable,
        repo_root=root,
        run_id=safe_id,
        mode=mode,
        selected=selected,
        contract=contract,
        artifact_root=artifact_root,
        artifact_dir=artifact_dir,
        run_dir=run_dir,
    )
    expected = _read_jsonl_strict(
        expected_path,
        "OmniAID expected inputs",
    )
    if not _same_json_type_and_value(expected, selected):
        raise ValueError("OmniAID expected input snapshot changed")
    physical = _read_jsonl_strict(results_path, "OmniAID results")
    if not physical:
        raise ValueError("OmniAID results are empty")
    if mode == "smoke" and len(physical) != SMOKE_IMAGES:
        raise ValueError("OmniAID smoke requires one attempt per image")
    latest = _validate_physical_attempts(
        selected=selected,
        physical=physical,
        repo_root=root,
        run_id=safe_id,
        fingerprint=fingerprint,
    )
    coverage_record = summarize_coverage(latest)
    require_complete_coverage(coverage_record)
    latest_rows = tuple(
        dict(latest.latest_by_sample_id[str(row["sample_id"])]) for row in selected
    )
    artifacts = validate_artifact_inventory(
        latest_results=latest_rows,
        repo_root=root,
        artifact_root=artifact_root,
        artifact_dir=artifact_dir,
    )
    coverage = _validate_manifest_summary(
        manifest=manifest,
        summary=summary,
        immutable=immutable,
        fingerprint=fingerprint,
        run_id=safe_id,
        mode=mode,
        contract=contract,
        selected=selected,
        physical=physical,
        latest=latest,
        manifest_path=manifest_path,
        results_path=results_path,
        expected_path=expected_path,
        summary_path=summary_path,
        release=release,
        artifact_files=len(artifacts),
        repo_root=root,
    )
    snapshot = _capture_evidence_snapshot(
        manifest_path=manifest_path,
        results_path=results_path,
        expected_path=expected_path,
        summary_path=summary_path,
        release=release,
        artifacts=artifacts,
    )
    return RunBundle(
        run_id=safe_id,
        fingerprint=fingerprint,
        mode=mode,
        run_dir=run_dir,
        manifest_path=manifest_path,
        results_path=results_path,
        expected_path=expected_path,
        summary_path=summary_path,
        manifest=manifest,
        immutable=immutable,
        release=release,
        selected=tuple(selected),
        contract=contract,
        physical_results=tuple(physical),
        latest_results=latest_rows,
        coverage=coverage,
        artifact_root=artifact_root,
        artifact_dir=artifact_dir,
        artifacts=artifacts,
        evidence_snapshot=snapshot,
    )


def load_formal_run(
    *,
    repo_root: Path,
    results_dir: Path,
    artifacts_dir: Path,
    run_id: str,
) -> RunBundle:
    return _load_run(
        repo_root=repo_root,
        results_dir=results_dir,
        artifacts_dir=artifacts_dir,
        run_id=run_id,
        mode="formal",
    )


def load_smoke_run(
    *,
    repo_root: Path,
    results_dir: Path,
    artifacts_dir: Path,
    run_id: str,
) -> RunBundle:
    return _load_run(
        repo_root=repo_root,
        results_dir=results_dir,
        artifacts_dir=artifacts_dir,
        run_id=run_id,
        mode="smoke",
    )


def _verify_bundle_unchanged(
    bundle: RunBundle,
    *,
    repo_root: Path,
) -> None:
    expected = bundle.evidence_snapshot
    for key, path in (
        ("manifest_sha256", bundle.manifest_path),
        ("results_sha256", bundle.results_path),
        ("expected_inputs_sha256", bundle.expected_path),
        ("runtime_summary_sha256", bundle.summary_path),
    ):
        if (
            path.is_symlink()
            or not path.is_file()
            or sha256_file(path) != expected[key]
        ):
            raise ValueError(f"OmniAID evidence changed: {key}")
    artifacts = validate_artifact_inventory(
        latest_results=bundle.latest_results,
        repo_root=repo_root,
        artifact_root=bundle.artifact_root,
        artifact_dir=bundle.artifact_dir,
    )
    if _artifact_inventory_sha256(artifacts) != expected["artifact_inventory_sha256"]:
        raise ValueError("OmniAID artifact evidence changed")
    release, selected, contract = _rebuild_contract(
        repo_root=repo_root,
        immutable=bundle.immutable,
        expected_mode=bundle.mode,
    )
    if (
        release.manifest_sha256 != expected["dataset_manifest_sha256"]
        or stable_json(selected) != stable_json(bundle.selected)
        or stable_json(contract.as_dict()) != stable_json(bundle.contract.as_dict())
    ):
        raise ValueError("OmniAID canonical selection changed")
    _verify_adapter_sources(
        bundle.immutable.get("adapter_sources"),
        repo_root=repo_root,
    )


def recompute_metrics(
    bundle: RunBundle,
    *,
    iterations: int = BOOTSTRAP_ITERATIONS,
    seed: int = BOOTSTRAP_SEED,
    results: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    if iterations != BOOTSTRAP_ITERATIONS or seed != BOOTSTRAP_SEED:
        raise ValueError(
            "OmniAID Balanced250 metrics require iterations=1000 " "and seed=20260726"
        )
    if bundle.mode != "formal" or len(bundle.selected) != FORMAL_IMAGES:
        raise ValueError("Balanced250 metrics require the formal 1775 run")
    rows = bundle.latest_results if results is None else tuple(results)
    metrics = summarize_balanced250_t1(
        bundle.release.inputs,
        bundle.release.panel,
        bundle.release.source_pairs,
        rows,
        run_id=bundle.run_id,
        run_manifest_fingerprint=bundle.fingerprint,
        run_dataset_contract=bundle.contract,
        iterations=iterations,
        seed=seed,
    )
    if (
        metrics.get("schema_version") != METRICS_SCHEMA_VERSION
        or metrics.get("coverage", {}).get("is_complete") is not True
    ):
        raise ValueError("OmniAID Balanced250 T1 metrics are incomplete")
    _reject_t2(metrics, "OmniAID Balanced250 metrics")
    return metrics


def _verify_source_projection(
    *,
    independent: Mapping[str, Any],
    recorded: Mapping[str, Any],
    label: str,
    expected_repository: str,
) -> None:
    if (
        set(independent)
        != {"commit", "tracked_dirty", "tracked_license_files", "files"}
        or set(recorded)
        != {
            "repository",
            "path",
            "commit",
            "tracked_dirty",
            "tracked_license_files",
            "files",
        }
        or recorded.get("repository") != expected_repository
        or independent.get("commit") != recorded.get("commit")
        or independent.get("tracked_dirty") is not False
        or recorded.get("tracked_dirty") is not False
        or independent.get("tracked_license_files")
        != recorded.get("tracked_license_files")
    ):
        raise ValueError(f"OmniAID {label} source evidence changed")
    independent_files = _require_mapping(
        independent.get("files"),
        f"independent {label} files",
    )
    recorded_files = _require_mapping(
        recorded.get("files"),
        f"recorded {label} files",
    )
    if set(independent_files) != set(recorded_files):
        raise ValueError(f"OmniAID {label} source file census changed")
    for relative, record in independent_files.items():
        expected = _require_mapping(
            recorded_files.get(relative),
            f"recorded {label} {relative}",
        )
        if record.get("bytes") != expected.get("bytes") or record.get(
            "sha256"
        ) != expected.get("sha256"):
            raise ValueError(f"OmniAID {label} source file changed: {relative}")


def _verify_source_assets(
    *,
    source_root: Path,
    space_root: Path,
    checkpoint_path: Path,
    config_path: Path,
    recorded_source: Mapping[str, Any],
    recorded_assets: Mapping[str, Any],
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    Mapping[str, Any],
    dict[str, Any],
]:
    if (
        set(recorded_source) != {"github", "space", "inference_source"}
        or recorded_source.get("inference_source") != OFFICIAL_INFERENCE_SOURCE
        or set(recorded_assets) != {"checkpoint", "omniaid_config", "dinov3_base"}
        or not _same_json_type_and_value(
            recorded_assets.get("dinov3_base"),
            legacy.DINO_BASE,
        )
    ):
        raise ValueError("OmniAID source/asset schema changed")
    github_record = _require_mapping(
        recorded_source.get("github"),
        "recorded GitHub source",
    )
    space_record = _require_mapping(
        recorded_source.get("space"),
        "recorded Space source",
    )
    checkpoint_record = _require_mapping(
        recorded_assets.get("checkpoint"),
        "recorded checkpoint",
    )
    config_record = _require_mapping(
        recorded_assets.get("omniaid_config"),
        "recorded OmniAID config",
    )
    if (
        source_root.resolve()
        != Path(_require_string(github_record.get("path"), "GitHub path")).resolve()
        or space_root.resolve()
        != Path(_require_string(space_record.get("path"), "Space path")).resolve()
        or checkpoint_path.resolve()
        != Path(
            _require_string(checkpoint_record.get("path"), "checkpoint path")
        ).resolve()
        or config_path.resolve()
        != Path(_require_string(config_record.get("path"), "config path")).resolve()
    ):
        raise ValueError("OmniAID source/asset paths differ from manifest")
    for path, label in (
        (source_root, "GitHub source root"),
        (space_root, "Space source root"),
    ):
        if path.is_symlink() or not path.is_dir():
            raise FileNotFoundError(f"missing or unsafe OmniAID {label}")
    for path, label in (
        (checkpoint_path, "checkpoint"),
        (config_path, "config"),
    ):
        _require_regular_file(path, f"OmniAID {label}")
    independent_source = legacy_audit._verify_source(
        source_root,
        space_root,
    )
    _verify_source_projection(
        independent=_require_mapping(
            independent_source.get("github"),
            "independent GitHub source",
        ),
        recorded=github_record,
        label="GitHub",
        expected_repository=legacy.MODEL_REPO_URL,
    )
    _verify_source_projection(
        independent=_require_mapping(
            independent_source.get("space"),
            "independent Space source",
        ),
        recorded=space_record,
        label="Space",
        expected_repository=legacy.MODEL_SPACE_URL,
    )
    state, config, evidence = legacy_audit._load_assets(
        checkpoint_path,
        config_path,
    )
    if (
        evidence.get("checkpoint_sha256") != checkpoint_record.get("sha256")
        or evidence.get("checkpoint_bytes") != checkpoint_record.get("bytes")
        or evidence.get("checkpoint_tensor_count")
        != checkpoint_record.get("tensor_count")
        or evidence.get("checkpoint_state_elements")
        != checkpoint_record.get("state_elements")
        or evidence.get("config_sha256") != config_record.get("sha256")
        or evidence.get("config_bytes") != config_record.get("bytes")
    ):
        raise ValueError("OmniAID independent asset evidence changed")
    return independent_source, evidence, state, config


def _configure_exact_recorded_runtime(
    *,
    device_text: str,
    recorded_runtime: Mapping[str, Any],
    label: str,
) -> tuple[Any, dict[str, Any]]:
    if device_text != recorded_runtime.get("device"):
        raise ValueError(f"{label} device differs from recorded runtime")
    runner = _assert_runner_contract_exports()
    device, current = runner.configure_runtime(
        device_text,
        seed=EXPECTED_RUNTIME_SEED,
    )
    validated = _validate_runtime_contract(
        current,
        label=f"{label} current runtime",
    )
    if not _same_json_type_and_value(validated, dict(recorded_runtime)):
        raise ValueError(f"{label} current runtime differs from recorded")
    return device, validated


def _build_independent_model(
    *,
    state: Mapping[str, Any],
    config: Mapping[str, Any],
    device: Any,
    space_root: Path,
) -> tuple[Any, dict[str, Any]]:
    model, evidence = legacy_audit._build_model(
        state,
        config,
        device,
        space_root,
    )
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if (
        evidence.get("strict_load") is not True
        or evidence.get("state_entries") != legacy.CHECKPOINT["tensor_count"]
        or evidence.get("state_elements") != legacy.CHECKPOINT["state_elements"]
        or evidence.get("svd_modules") != legacy.SVD_MODULE_COUNT
        or evidence.get("parameter_count") != 507_041_863
        or evidence.get("base_weights_downloaded") is not False
        or evidence.get("eval_mode") is not True
    ):
        raise ValueError("independent OmniAID model construction changed")
    return model, evidence


def audit_official_golden(
    *,
    model: Any,
    device: Any,
    space_root: Path,
    recorded: Mapping[str, Any],
) -> dict[str, Any]:
    device_family = getattr(device, "type", None)
    if not isinstance(device_family, str):
        raise ValueError("independent OmniAID device family changed")
    validated = _validate_recorded_official_golden(
        recorded,
        device_family=device_family,
        label="recorded OmniAID official golden",
    )
    cases = validated["cases"]
    audited: list[dict[str, Any]] = []
    for index, case_value in enumerate(cases):
        case = _require_mapping(case_value, f"official golden case {index}")
        relative = _require_string(case.get("path"), f"golden {index}.path")
        path = space_root / relative
        _require_regular_file(path, f"official golden input {relative}")
        if sha256_file(path) != case.get("input_sha256"):
            raise ValueError(f"official golden input changed: {relative}")
        image, preprocess = legacy_audit._preprocess(path)
        if not _same_json_type_and_value(case.get("preprocess"), preprocess):
            raise ValueError(f"official golden preprocess changed: {relative}")
        arrays, replay = legacy_audit._fresh_forward(
            model,
            device,
            image,
        )
        hashes = {key: _array_sha256(value) for key, value in arrays.items()}
        if (
            hashes != case.get("array_sha256")
            or [float(value) for value in arrays["class_logits"]] != case.get("logits")
            or [float(value) for value in arrays["final_gates"]]
            != case.get("final_expert_gates")
            or replay.get("score") != case.get("fake_probability")
            or case.get("repeat_all_arrays_exact") is not True
        ):
            raise ValueError(f"official golden output changed: {relative}")
        audited.append(
            {
                "path": relative,
                "input_sha256": case["input_sha256"],
                "array_sha256": hashes,
                "fake_probability": replay["score"],
                "all_six_arrays_exact": True,
            }
        )
    return {
        "cases_audited": len(audited),
        "device_family": getattr(device, "type", None),
        "all_recorded_arrays_exact": True,
        "all_recorded_scores_exact": True,
        "cases": audited,
    }


def replay_persisted_head_softmax_router(
    *,
    latest_results: Sequence[Mapping[str, Any]],
    artifacts: Mapping[str, OmniAIDArtifact],
    model: Any,
    device: Any,
) -> dict[str, Any]:
    import torch
    from torch.nn import functional

    maximum = {
        "class_logits": 0.0,
        "probability": 0.0,
        "raw_logit_margin": 0.0,
        "router_indices": 0.0,
        "router_gates": 0.0,
        "final_gates": 0.0,
    }
    for row in latest_results:
        sample_id = _require_string(
            row.get("sample_id"),
            "persisted replay sample_id",
        )
        artifact = artifacts.get(sample_id)
        if artifact is None:
            raise ValueError(f"missing OmniAID artifact {sample_id}")
        arrays = artifact.arrays
        feature = torch.from_numpy(arrays["pooler_output"]).to(device).unsqueeze(0)
        routing = torch.from_numpy(arrays["routing_feature"]).to(device).unsqueeze(0)
        with torch.inference_mode():
            logits = functional.linear(
                feature,
                model.head.weight,
                model.head.bias,
            )
            probability = torch.softmax(logits, dim=1)[:, 1]
            margin = logits[:, 1] - logits[:, 0]
            routed = model.gating_network(routing)
        replay_logits = np.ascontiguousarray(
            logits[0].detach().cpu().numpy(),
            dtype=np.float32,
        )
        replay_indices = np.ascontiguousarray(
            routed["top_k_indices"][0].detach().cpu().numpy(),
            dtype=np.int64,
        )
        replay_gates = np.ascontiguousarray(
            routed["top_k_gates"][0].detach().cpu().numpy(),
            dtype=np.float32,
        )
        replay_final = np.zeros(legacy.EXPERT_COUNT, dtype=np.float32)
        replay_final[replay_indices] = replay_gates
        replay_final[legacy.ARTIFACT_EXPERT_INDEX] = np.float32(1.0)
        differences = {
            "class_logits": float(
                np.max(
                    np.abs(
                        replay_logits.astype(np.float64)
                        - arrays["class_logits"].astype(np.float64)
                    ),
                    initial=0.0,
                )
            ),
            "probability": abs(
                float(probability[0].item())
                - _require_finite(row.get("ai_score"), f"{sample_id}.score")
            ),
            "raw_logit_margin": abs(
                float(margin[0].item())
                - _require_finite(
                    row.get("raw_logit_margin"),
                    f"{sample_id}.margin",
                )
            ),
            "router_indices": float(
                np.max(
                    np.abs(
                        replay_indices.astype(np.float64)
                        - arrays["semantic_top_k_indices"].astype(np.float64)
                    ),
                    initial=0.0,
                )
            ),
            "router_gates": float(
                np.max(
                    np.abs(
                        replay_gates.astype(np.float64)
                        - arrays["semantic_top_k_gates"].astype(np.float64)
                    ),
                    initial=0.0,
                )
            ),
            "final_gates": float(
                np.max(
                    np.abs(
                        replay_final.astype(np.float64)
                        - arrays["final_gates"].astype(np.float64)
                    ),
                    initial=0.0,
                )
            ),
        }
        for key, difference in differences.items():
            maximum[key] = max(maximum[key], difference)
            if difference != 0.0:
                raise ValueError(f"{sample_id} persisted {key} replay changed")
        _validate_score_payload(
            row,
            sample_id=sample_id,
            arrays=arrays,
        )
    return {
        "artifacts_replayed": len(latest_results),
        "head_replays": len(latest_results),
        "float32_softmax_replays": len(latest_results),
        "automatic_router_replays": len(latest_results),
        "final_gate_scatter_replays": len(latest_results),
        "maximum_absolute_difference": maximum,
        "all_passed": True,
    }


def _canonical_input_path(
    row: Mapping[str, Any],
    *,
    repo_root: Path,
) -> Path:
    sample_id = str(row["sample_id"])
    path = _safe_repo_path(
        row.get("canonical_path"),
        repo_root=repo_root,
        label=f"{sample_id} canonical input",
    )
    if sha256_file(path) != row.get("canonical_sha256"):
        raise ValueError(f"{sample_id} canonical input SHA-256 changed")
    return path


def replay_model(
    bundle: RunBundle,
    *,
    repo_root: Path,
    model: Any,
    device: Any,
) -> tuple[dict[str, Any], tuple[dict[str, Any], ...]]:
    """Freshly replay all 1,775 JPEGs through independent OmniAID."""

    if bundle.mode != "formal" or len(bundle.selected) != FORMAL_IMAGES:
        raise ValueError("fresh OmniAID replay requires the formal 1,775 selection")
    maximum = {key: 0.0 for key in ARTIFACT_SCHEMA}
    maximum_probability = 0.0
    maximum_margin = 0.0
    fresh_rows: list[dict[str, Any]] = []
    for index, (input_row, result_row) in enumerate(
        zip(bundle.selected, bundle.latest_results, strict=True),
        start=1,
    ):
        sample_id = str(input_row["sample_id"])
        if result_row.get("sample_id") != sample_id:
            raise ValueError("OmniAID fresh replay order changed")
        path = _canonical_input_path(input_row, repo_root=repo_root)
        image, preprocess = legacy_audit._preprocess(path)
        if not _same_json_type_and_value(
            result_row.get("preprocess"),
            preprocess,
        ):
            raise ValueError(f"{sample_id} fresh preprocessing changed")
        arrays, replay = legacy_audit._fresh_forward(
            model,
            device,
            image,
        )
        persisted = bundle.artifacts[sample_id].arrays
        for key in ARTIFACT_SCHEMA:
            difference = float(
                np.max(
                    np.abs(
                        arrays[key].astype(np.float64)
                        - persisted[key].astype(np.float64)
                    ),
                    initial=0.0,
                )
            )
            maximum[key] = max(maximum[key], difference)
            if difference != 0.0:
                raise ValueError(f"{sample_id} fresh full-model {key} changed")
        probability_difference = abs(
            float(replay["score"])
            - _require_finite(
                result_row.get("ai_score"),
                f"{sample_id}.ai_score",
            )
        )
        margin_difference = abs(
            float(replay["raw_logit_margin"])
            - _require_finite(
                result_row.get("raw_logit_margin"),
                f"{sample_id}.raw_logit_margin",
            )
        )
        maximum_probability = max(
            maximum_probability,
            probability_difference,
        )
        maximum_margin = max(maximum_margin, margin_difference)
        if probability_difference != 0.0 or margin_difference != 0.0:
            raise ValueError(f"{sample_id} fresh score/margin changed")
        _validate_score_payload(
            result_row,
            sample_id=sample_id,
            arrays=arrays,
        )
        fresh = dict(result_row)
        fresh.update(
            {
                "ai_score": float(replay["score"]),
                "fake_probability": float(replay["score"]),
                "probability": float(replay["score"]),
                "score": float(replay["score"]),
                "raw_logit_margin": float(replay["raw_logit_margin"]),
                "classification_decision": (float(replay["score"]) > 0.5),
            }
        )
        fresh_rows.append(fresh)
        if index % 50 == 0 or index == FORMAL_IMAGES:
            print(
                f"[OmniAID audit {index}/{FORMAL_IMAGES}] "
                "fresh full-model replay exact",
                flush=True,
            )
    return (
        {
            "selected_images_freshly_reopened": FORMAL_IMAGES,
            "selected_images_freshly_preprocessed": FORMAL_IMAGES,
            "complete_model_forward_passes": FORMAL_IMAGES,
            "six_array_sets_compared": FORMAL_IMAGES,
            "scores_compared": FORMAL_IMAGES,
            "margins_compared": FORMAL_IMAGES,
            "maximum_array_absolute_difference": maximum,
            "maximum_probability_absolute_difference": maximum_probability,
            "maximum_raw_logit_margin_absolute_difference": maximum_margin,
            "absolute_tolerance": 0.0,
            "all_passed": True,
            "scope": {
                "valid_for_t1": True,
                "valid_for_t2": False,
                "localization_output": None,
            },
        },
        tuple(fresh_rows),
    )


def _exact_smoke_projection(row: Mapping[str, Any]) -> dict[str, Any]:
    sample_id = _require_string(
        row.get("sample_id"),
        "smoke result sample_id",
    )
    _validate_score_payload(row, sample_id=sample_id)
    missing = _SMOKE_IGNORED_RESULT_FIELDS - set(row)
    if missing:
        raise ValueError(f"{sample_id} lacks runtime field {sorted(missing)[0]}")
    return {
        key: value
        for key, value in row.items()
        if key not in _SMOKE_IGNORED_RESULT_FIELDS
    }


def compare_computational_results(
    *,
    reference_rows: Sequence[Mapping[str, Any]],
    replay_rows: Sequence[Mapping[str, Any]],
    reference_artifacts: Mapping[str, OmniAIDArtifact],
    replay_artifacts: Mapping[str, OmniAIDArtifact],
) -> dict[str, Any]:
    def unique(
        rows: Sequence[Mapping[str, Any]],
        label: str,
    ) -> dict[str, Mapping[str, Any]]:
        result: dict[str, Mapping[str, Any]] = {}
        for index, row in enumerate(rows):
            sample_id = _require_string(
                row.get("sample_id"),
                f"{label}[{index}].sample_id",
            )
            if sample_id in result:
                raise ValueError(f"{label} repeats {sample_id}")
            result[sample_id] = row
        return result

    reference = unique(reference_rows, "reference smoke")
    replay = unique(replay_rows, "replay smoke")
    if (
        len(reference) != SMOKE_IMAGES
        or set(reference) != set(replay)
        or set(reference_artifacts) != set(reference)
        or set(replay_artifacts) != set(replay)
    ):
        raise ValueError("OmniAID smoke computational coverage changed")
    maximum = {key: 0.0 for key in ARTIFACT_SCHEMA}
    maximum_probability = 0.0
    maximum_margin = 0.0
    for sample_id in sorted(reference):
        left_row = reference[sample_id]
        right_row = replay[sample_id]
        if stable_json(_exact_smoke_projection(left_row)) != stable_json(
            _exact_smoke_projection(right_row)
        ):
            raise ValueError(f"OmniAID smoke result differs for {sample_id}")
        left = reference_artifacts[sample_id]
        right = replay_artifacts[sample_id]
        if (
            left.path.read_bytes() != right.path.read_bytes()
            or left.file_sha256 != right.file_sha256
            or left.file_bytes != right.file_bytes
            or dict(left.array_sha256) != dict(right.array_sha256)
        ):
            raise ValueError(f"OmniAID smoke NPZ bytes differ for {sample_id}")
        for key in ARTIFACT_SCHEMA:
            difference = float(
                np.max(
                    np.abs(
                        left.arrays[key].astype(np.float64)
                        - right.arrays[key].astype(np.float64)
                    ),
                    initial=0.0,
                )
            )
            maximum[key] = max(maximum[key], difference)
            if difference != 0.0:
                raise ValueError(f"OmniAID smoke array {key} differs for {sample_id}")
        probability_difference = abs(
            float(left_row["ai_score"]) - float(right_row["ai_score"])
        )
        margin_difference = abs(
            float(left_row["raw_logit_margin"]) - float(right_row["raw_logit_margin"])
        )
        maximum_probability = max(
            maximum_probability,
            probability_difference,
        )
        maximum_margin = max(maximum_margin, margin_difference)
        if probability_difference != 0.0 or margin_difference != 0.0:
            raise ValueError(f"OmniAID smoke scores differ for {sample_id}")
    return {
        "images_compared": SMOKE_IMAGES,
        "ignored_result_fields": sorted(_SMOKE_IGNORED_RESULT_FIELDS),
        "exact_computational_projection": True,
        "npz_file_bytes_exact": True,
        "all_six_arrays_exact": True,
        "maximum_array_absolute_difference": maximum,
        "maximum_probability_absolute_difference": maximum_probability,
        "maximum_raw_logit_margin_absolute_difference": maximum_margin,
    }


def _smoke_immutable_projection(
    immutable: Mapping[str, Any],
) -> dict[str, Any]:
    if set(immutable) != EXPECTED_IMMUTABLE_KEYS:
        raise ValueError("OmniAID smoke immutable key set changed")
    projection = json.loads(stable_json(immutable))
    projection.pop("run_id")
    projection.pop("outputs")
    policy = _require_mapping(
        projection.get("artifact_policy"),
        "smoke artifact policy",
    )
    policy["artifact_root"] = "<run-specific-artifact-root>"
    projection["artifact_policy"] = policy
    return projection


def _analysis_runtime_contract() -> dict[str, Any]:
    import sys

    return {
        "python": {
            "version": ".".join(str(value) for value in sys.version_info[:3]),
            "executable": str(Path(os.path.abspath(sys.executable))),
        },
        "numpy": np.__version__,
        "scipy": importlib.metadata.version("scipy"),
        "scikit-learn": importlib.metadata.version("scikit-learn"),
        "analyzer": "eval/opensource/analyze_omniaid_balanced.py",
        "legacy_scientific_path": ("eval/opensource/analyze_omniaid_run.py"),
    }


def _json_artifact_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _reject_output_symlink_components(
    path: Path,
    *,
    label: str,
) -> None:
    absolute = _lexical_absolute(path)
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        if current.is_symlink():
            raise ValueError(f"{label} output path contains a symlink")
    if absolute.exists() and not absolute.is_file():
        raise ValueError(f"{label} output is not a regular file")


def _write_json_verified(
    path: Path,
    value: Any,
    *,
    label: str,
) -> None:
    _reject_output_symlink_components(path, label=label)
    expected = _json_artifact_sha256(value)
    atomic_write_json(path, value)
    if path.is_symlink() or not path.is_file() or sha256_file(path) != expected:
        raise ValueError(f"{label} changed after atomic write")


def _validate_formal_output_scope(
    *,
    run_dir: Path,
    metrics_output_path: Path | None,
    audit_output_path: Path | None,
) -> dict[str, Path | None]:
    expected = {
        "metrics": _lexical_absolute(run_dir / "balanced250_metrics.json"),
        "audit": _lexical_absolute(run_dir / "independent_audit.json"),
    }
    requested = {
        "metrics": metrics_output_path,
        "audit": audit_output_path,
    }
    result: dict[str, Path | None] = {}
    for name, value in requested.items():
        if value is None:
            result[name] = None
            continue
        candidate = _lexical_absolute(value)
        if candidate != expected[name]:
            raise ValueError(f"OmniAID formal {name} output must be {expected[name]}")
        _reject_output_symlink_components(
            candidate,
            label=f"formal {name}",
        )
        result[name] = candidate
    return result


def _smoke_comparison_default_path(
    *,
    results_dir: Path,
    reference_run_id: str,
    replay_run_id: str,
) -> Path:
    reference = _valid_run_id(reference_run_id)
    replay = _valid_run_id(replay_run_id)
    digest = hashlib.sha256(
        stable_json([reference, replay]).encode("utf-8")
    ).hexdigest()
    return (
        _lexical_absolute(results_dir)
        / "_reports"
        / f"{SMOKE_COMPARISON_SCHEMA_VERSION}_{digest}.json"
    )


def _validate_smoke_output_scope(
    *,
    output_path: Path,
    results_dir: Path,
    reference_run_id: str,
    replay_run_id: str,
) -> Path:
    candidate = _lexical_absolute(output_path)
    expected = _smoke_comparison_default_path(
        results_dir=results_dir,
        reference_run_id=reference_run_id,
        replay_run_id=replay_run_id,
    )
    if candidate != expected:
        raise ValueError(f"OmniAID smoke comparison output must be {expected}")
    _reject_output_symlink_components(
        candidate,
        label="smoke comparison",
    )
    return candidate


def _validate_output_targets(
    outputs: Mapping[str, Path | None],
    *,
    protected_files: Sequence[Path],
    protected_dirs: Sequence[Path],
) -> None:
    resolved: dict[str, Path] = {}
    for name, path in outputs.items():
        if path is None:
            continue
        candidate = _lexical_absolute(path)
        _reject_output_symlink_components(
            candidate,
            label=f"analysis {name}",
        )
        resolved[name] = candidate.resolve()
    if len(set(resolved.values())) != len(resolved):
        raise ValueError("OmniAID analysis output paths collide")
    protected = {path.resolve() for path in protected_files}
    directories = tuple(path.resolve() for path in protected_dirs)
    for name, path in resolved.items():
        if path in protected or any(
            path == directory or directory in path.parents for directory in directories
        ):
            raise ValueError(f"OmniAID analysis output {name} would overwrite evidence")


def _bundle_protected_files(
    bundle: RunBundle,
    *,
    repo_root: Path,
) -> tuple[Path, ...]:
    files = [
        bundle.manifest_path,
        bundle.results_path,
        bundle.expected_path,
        bundle.summary_path,
        *(repo_root / relative for relative in EXPECTED_ADAPTER_SOURCE_PATHS),
    ]
    assets = _require_mapping(
        bundle.immutable.get("assets"),
        "immutable.assets",
    )
    for key in ("checkpoint", "omniaid_config"):
        record = _require_mapping(assets.get(key), f"assets.{key}")
        path = record.get("path")
        if isinstance(path, str) and path:
            files.append(Path(path))
    return tuple(files)


def _verify_source_assets_unchanged(
    *,
    source_root: Path,
    space_root: Path,
    checkpoint_path: Path,
    config_path: Path,
) -> None:
    source = legacy_audit._verify_source(source_root, space_root)
    if (
        source.get("github", {}).get("commit") != legacy.MODEL_SOURCE_COMMIT
        or source.get("space", {}).get("commit") != legacy.MODEL_SPACE_COMMIT
        or sha256_file(checkpoint_path) != legacy.CHECKPOINT["sha256"]
        or checkpoint_path.stat().st_size != legacy.CHECKPOINT["bytes"]
        or sha256_file(config_path) != legacy.OMNIAID_CONFIG["sha256"]
        or config_path.stat().st_size != legacy.OMNIAID_CONFIG["bytes"]
    ):
        raise ValueError("OmniAID source/assets changed during analysis")


def compare_smoke_runs(
    *,
    repo_root: Path,
    results_dir: Path,
    artifacts_dir: Path,
    reference_run_id: str,
    replay_run_id: str,
    source_root: Path,
    space_root: Path,
    checkpoint_path: Path,
    config_path: Path,
    output_path: Path | None = None,
) -> dict[str, Any]:
    if reference_run_id == replay_run_id or {
        reference_run_id,
        replay_run_id,
    } != {DEFAULT_SMOKE_RUN_ID_A, DEFAULT_SMOKE_RUN_ID_B}:
        raise ValueError("smoke comparison requires frozen distinct A/B IDs")
    reference = load_smoke_run(
        repo_root=repo_root,
        results_dir=results_dir,
        artifacts_dir=artifacts_dir,
        run_id=reference_run_id,
    )
    replay = load_smoke_run(
        repo_root=repo_root,
        results_dir=results_dir,
        artifacts_dir=artifacts_dir,
        run_id=replay_run_id,
    )
    if output_path is not None:
        output_path = _validate_smoke_output_scope(
            output_path=output_path,
            results_dir=results_dir,
            reference_run_id=reference_run_id,
            replay_run_id=replay_run_id,
        )
    _validate_output_targets(
        {"comparison": output_path},
        protected_files=(
            *_bundle_protected_files(reference, repo_root=repo_root),
            *_bundle_protected_files(replay, repo_root=repo_root),
        ),
        protected_dirs=(
            reference.artifact_root,
            replay.artifact_root,
            source_root,
            space_root,
            reference.release.manifest_path.parent,
        ),
    )
    if (
        stable_json(_smoke_immutable_projection(reference.immutable))
        != stable_json(_smoke_immutable_projection(replay.immutable))
        or stable_json(reference.selected) != stable_json(replay.selected)
        or len(reference.selected) != SMOKE_IMAGES
        or len(reference.physical_results) != SMOKE_IMAGES
        or len(replay.physical_results) != SMOKE_IMAGES
    ):
        raise ValueError("OmniAID smoke immutable/selection evidence differs")
    recorded_runtime = _validate_runtime_contract(
        reference.immutable.get("runtime"),
        label="smoke recorded runtime",
    )
    if not _same_json_type_and_value(
        replay.immutable.get("runtime"),
        recorded_runtime,
    ):
        raise ValueError("OmniAID smoke recorded runtimes differ")
    recorded_source = _require_mapping(
        reference.immutable.get("source"),
        "smoke recorded source",
    )
    recorded_assets = _require_mapping(
        reference.immutable.get("assets"),
        "smoke recorded assets",
    )
    source_evidence, asset_evidence, state, config = _verify_source_assets(
        source_root=source_root,
        space_root=space_root,
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        recorded_source=recorded_source,
        recorded_assets=recorded_assets,
    )
    device, runtime = _configure_exact_recorded_runtime(
        device_text=str(recorded_runtime["device"]),
        recorded_runtime=recorded_runtime,
        label="smoke comparison",
    )
    model = None
    try:
        model, model_evidence = _build_independent_model(
            state=state,
            config=config,
            device=device,
            space_root=space_root,
        )
        official = audit_official_golden(
            model=model,
            device=device,
            space_root=space_root,
            recorded=_require_mapping(
                reference.immutable.get("execution_official_golden"),
                "smoke official golden",
            ),
        )
        reference_replay = replay_persisted_head_softmax_router(
            latest_results=reference.latest_results,
            artifacts=reference.artifacts,
            model=model,
            device=device,
        )
        replay_replay = replay_persisted_head_softmax_router(
            latest_results=replay.latest_results,
            artifacts=replay.artifacts,
            model=model,
            device=device,
        )
        comparison = compare_computational_results(
            reference_rows=reference.latest_results,
            replay_rows=replay.latest_results,
            reference_artifacts=reference.artifacts,
            replay_artifacts=replay.artifacts,
        )
    finally:
        if model is not None:
            del model
        del state
        gc.collect()
    _verify_bundle_unchanged(reference, repo_root=repo_root)
    _verify_bundle_unchanged(replay, repo_root=repo_root)
    _verify_source_assets_unchanged(
        source_root=source_root,
        space_root=space_root,
        checkpoint_path=checkpoint_path,
        config_path=config_path,
    )
    report = {
        "schema_version": SMOKE_COMPARISON_SCHEMA_VERSION,
        "status": "deterministic_smoke_comparison_passed",
        "compared_at": utc_now(),
        "analysis_runtime": _analysis_runtime_contract(),
        "recorded_runtime_reproduced": runtime,
        "independent_source": source_evidence,
        "independent_assets": asset_evidence,
        "independent_model_load": model_evidence,
        "independent_official_golden": official,
        "reference": {
            "run_id": reference.run_id,
            "run_manifest_fingerprint": reference.fingerprint,
            **dict(reference.evidence_snapshot),
        },
        "replay": {
            "run_id": replay.run_id,
            "run_manifest_fingerprint": replay.fingerprint,
            **dict(replay.evidence_snapshot),
        },
        "selection": reference.contract.selection.as_dict(),
        "reference_persisted_replay": reference_replay,
        "replay_persisted_replay": replay_replay,
        "comparison": comparison,
        "method_boundary": {
            "method": "OmniAID-DINO v2 automatic router",
            "valid_for_t1": True,
            "valid_for_t2": False,
            "localization_output": None,
            "commercial_clearance": False,
        },
        "evidence_reverified_after_comparison": True,
    }
    _reject_t2(report, "OmniAID smoke comparison")
    if output_path is not None:
        _write_json_verified(
            output_path,
            report,
            label="OmniAID smoke comparison",
        )
    return report


def analyze(
    *,
    repo_root: Path,
    results_dir: Path,
    artifacts_dir: Path,
    run_id: str,
    source_root: Path,
    space_root: Path,
    checkpoint_path: Path,
    config_path: Path,
    device_text: str,
    metrics_output_path: Path | None,
    audit_output_path: Path | None,
    replay: bool = True,
    iterations: int = BOOTSTRAP_ITERATIONS,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    bundle = load_formal_run(
        repo_root=repo_root,
        results_dir=results_dir,
        artifacts_dir=artifacts_dir,
        run_id=run_id,
    )
    outputs = _validate_formal_output_scope(
        run_dir=bundle.run_dir,
        metrics_output_path=metrics_output_path,
        audit_output_path=audit_output_path,
    )
    metrics_output_path = outputs["metrics"]
    audit_output_path = outputs["audit"]
    _validate_output_targets(
        {
            "metrics": metrics_output_path,
            "audit": audit_output_path,
        },
        protected_files=_bundle_protected_files(
            bundle,
            repo_root=repo_root,
        ),
        protected_dirs=(
            bundle.artifact_root,
            source_root,
            space_root,
            bundle.release.manifest_path.parent,
        ),
    )
    recorded_source = _require_mapping(
        bundle.immutable.get("source"),
        "formal recorded source",
    )
    recorded_assets = _require_mapping(
        bundle.immutable.get("assets"),
        "formal recorded assets",
    )
    source_evidence, asset_evidence, state, config = _verify_source_assets(
        source_root=source_root,
        space_root=space_root,
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        recorded_source=recorded_source,
        recorded_assets=recorded_assets,
    )
    recorded_runtime = _validate_runtime_contract(
        bundle.immutable.get("runtime"),
        label="formal recorded runtime",
    )
    device, current_runtime = _configure_exact_recorded_runtime(
        device_text=device_text,
        recorded_runtime=recorded_runtime,
        label="formal analysis",
    )
    model = None
    fresh_replay: dict[str, Any] | None = None
    fresh_metrics: dict[str, Any] | None = None
    try:
        model, model_evidence = _build_independent_model(
            state=state,
            config=config,
            device=device,
            space_root=space_root,
        )
        official = audit_official_golden(
            model=model,
            device=device,
            space_root=space_root,
            recorded=_require_mapping(
                bundle.immutable.get("execution_official_golden"),
                "formal official golden",
            ),
        )
        persisted_replay = replay_persisted_head_softmax_router(
            latest_results=bundle.latest_results,
            artifacts=bundle.artifacts,
            model=model,
            device=device,
        )
        metrics = recompute_metrics(
            bundle,
            iterations=iterations,
            seed=seed,
        )
        if replay:
            fresh_replay, fresh_rows = replay_model(
                bundle,
                repo_root=repo_root,
                model=model,
                device=device,
            )
            fresh_metrics = recompute_metrics(
                bundle,
                iterations=iterations,
                seed=seed,
                results=fresh_rows,
            )
            if stable_json(fresh_metrics) != stable_json(metrics):
                raise ValueError("OmniAID fresh full-model metrics changed")
    finally:
        if model is not None:
            del model
        del state
        gc.collect()
    _verify_bundle_unchanged(bundle, repo_root=repo_root)
    _verify_source_assets_unchanged(
        source_root=source_root,
        space_root=space_root,
        checkpoint_path=checkpoint_path,
        config_path=config_path,
    )
    metrics_sha256 = _json_artifact_sha256(metrics)
    audit = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "status": ("replay_audit_passed" if replay else "artifact_audit_passed"),
        "run_id": bundle.run_id,
        "run_manifest_fingerprint": bundle.fingerprint,
        "audited_at": utc_now(),
        "formal_images": len(bundle.selected),
        "physical_result_rows": len(bundle.physical_results),
        "latest_result_rows": len(bundle.latest_results),
        "coverage": dict(bundle.coverage),
        "artifact_files": len(bundle.artifacts),
        "metrics_schema_version": metrics["schema_version"],
        "metrics_bootstrap": metrics["bootstrap"],
        "analysis_runtime": _analysis_runtime_contract(),
        "recorded_runtime_reproduced": current_runtime,
        "independent_source": source_evidence,
        "independent_assets": asset_evidence,
        "independent_model_load": model_evidence,
        "independent_official_golden": official,
        "persisted_artifact_head_softmax_router_replay": (persisted_replay),
        "fresh_model_replay": fresh_replay,
        "fresh_model_metrics_exact": (
            stable_json(fresh_metrics) == stable_json(metrics)
            if fresh_metrics is not None
            else None
        ),
        "method_boundary": {
            "method": "OmniAID-DINO v2 automatic router",
            "architecture": legacy.MODEL_ARCH,
            "preprocess_profile": legacy.PREPROCESS_PROFILE,
            "released_checkpoint_evaluated": True,
            "valid_for_t1": True,
            "valid_for_t2": False,
            "localization_output": None,
            "fullframe_data_kind": (
                "conditional_full_frame_edit_not_pure_text_to_image"
            ),
            "license": legacy.LICENSE_RECORD,
            "commercial_clearance": False,
        },
        "contract_checks": {
            "exact_formal_whole_image_selection_rebuilt": True,
            "all_physical_attempts_validated": True,
            "complete_latest_coverage_required": True,
            "pair_rank_rejected": True,
            "t2_joint_dense_claims_rejected": True,
            "canonical_9848_byte_six_array_npz_validated": True,
            "persisted_head_float32_softmax_router_scatter_replayed": True,
            "shared_balanced250_metrics_only": True,
            "fresh_full_model_replay_default": True,
            "source_assets_and_evidence_reverified_after_replay": True,
        },
        "artifacts": {
            **dict(bundle.evidence_snapshot),
            "metrics_sha256": metrics_sha256,
            "fresh_metrics_sha256": (
                _json_artifact_sha256(fresh_metrics)
                if fresh_metrics is not None
                else None
            ),
        },
    }
    _reject_t2(audit, "OmniAID formal audit")

    # Publication is last: no report is written until all independent replay,
    # metric, and final evidence checks above have passed.
    outputs = _validate_formal_output_scope(
        run_dir=bundle.run_dir,
        metrics_output_path=metrics_output_path,
        audit_output_path=audit_output_path,
    )
    _validate_output_targets(
        outputs,
        protected_files=_bundle_protected_files(
            bundle,
            repo_root=repo_root,
        ),
        protected_dirs=(
            bundle.artifact_root,
            source_root,
            space_root,
            bundle.release.manifest_path.parent,
        ),
    )
    if outputs["metrics"] is not None:
        _write_json_verified(
            outputs["metrics"],
            metrics,
            label="OmniAID Balanced250 metrics",
        )
    if outputs["audit"] is not None:
        _write_json_verified(
            outputs["audit"],
            audit,
            label="OmniAID independent audit",
        )
    if (
        outputs["metrics"] is not None
        and sha256_file(outputs["metrics"]) != metrics_sha256
    ):
        raise ValueError("OmniAID metrics changed after audit write")
    return audit


def _anchored(path: Path, repo_root: Path) -> Path:
    return (
        _lexical_absolute(path)
        if path.is_absolute()
        else _lexical_absolute(path, base=repo_root)
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=DEFAULT_ARTIFACTS_DIR,
    )
    parser.add_argument("--run-id", default=DEFAULT_FORMAL_RUN_ID)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=DEFAULT_SOURCE_ROOT,
    )
    parser.add_argument(
        "--space-root",
        type=Path,
        default=DEFAULT_SPACE_ROOT,
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
    )
    parser.add_argument(
        "--omniaid-config",
        type=Path,
        default=DEFAULT_OMNIAID_CONFIG,
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--skip-model-replay", action="store_true")
    parser.add_argument("--compare-smoke-run-id")
    parser.add_argument(
        "--bootstrap-iterations",
        type=int,
        default=BOOTSTRAP_ITERATIONS,
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=BOOTSTRAP_SEED,
    )
    parser.add_argument("--metrics-output", type=Path)
    parser.add_argument("--audit-output", type=Path)
    parser.add_argument("--comparison-output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    _assert_runner_contract_exports()
    repo_root = args.repo_root.resolve()
    results_dir = _safe_standard_root(
        args.results_dir,
        repo_root=repo_root,
        expected_relative=DEFAULT_RESULTS_DIR,
        label="OmniAID results root",
    )
    artifacts_dir = _safe_standard_root(
        args.artifacts_dir,
        repo_root=repo_root,
        expected_relative=DEFAULT_ARTIFACTS_DIR,
        label="OmniAID artifacts root",
    )
    run_id = _valid_run_id(args.run_id)
    source_root = _anchored(args.source_root, repo_root)
    space_root = _anchored(args.space_root, repo_root)
    checkpoint = _anchored(args.checkpoint, repo_root)
    config = _anchored(args.omniaid_config, repo_root)
    if args.compare_smoke_run_id is not None:
        compare_id = _valid_run_id(args.compare_smoke_run_id)
        if (
            args.metrics_output is not None
            or args.audit_output is not None
            or args.skip_model_replay
            or args.comparison_output is not None
        ):
            raise ValueError("smoke comparison accepts no formal/custom output options")
        output = _smoke_comparison_default_path(
            results_dir=results_dir,
            reference_run_id=run_id,
            replay_run_id=compare_id,
        )
        report = compare_smoke_runs(
            repo_root=repo_root,
            results_dir=results_dir,
            artifacts_dir=artifacts_dir,
            reference_run_id=run_id,
            replay_run_id=compare_id,
            source_root=source_root,
            space_root=space_root,
            checkpoint_path=checkpoint,
            config_path=config,
            output_path=output,
        )
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
        return 0
    if args.comparison_output is not None:
        raise ValueError("--comparison-output is reserved for smoke comparison")
    run_dir = _resolve_run_dir(results_dir, run_id)
    metrics_output = (
        _anchored(args.metrics_output, repo_root)
        if args.metrics_output is not None
        else run_dir / "balanced250_metrics.json"
    )
    audit_output = (
        _anchored(args.audit_output, repo_root)
        if args.audit_output is not None
        else run_dir / "independent_audit.json"
    )
    report = analyze(
        repo_root=repo_root,
        results_dir=results_dir,
        artifacts_dir=artifacts_dir,
        run_id=run_id,
        source_root=source_root,
        space_root=space_root,
        checkpoint_path=checkpoint,
        config_path=config,
        device_text=args.device,
        metrics_output_path=metrics_output,
        audit_output_path=audit_output,
        replay=not args.skip_model_replay,
        iterations=args.bootstrap_iterations,
        seed=args.bootstrap_seed,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
