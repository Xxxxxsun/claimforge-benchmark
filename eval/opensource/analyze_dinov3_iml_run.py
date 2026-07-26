#!/usr/bin/env python3
"""Independently audit and analyze an official DINOv3-IML T2-only run."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Mapping

import numpy as np
from PIL import Image

from eval.opensource.common import (
    atomic_write_json,
    read_jsonl,
    sha256_file,
    stable_json,
    utc_now,
)
from eval.opensource.dinov3_iml_metrics import (
    binary_pixel_metrics_strict,
    summarize_dinov3_iml_pair_slice,
    summarize_dinov3_iml_results,
)


DEFAULT_RUN_ID = (
    "dinov3_iml_cat_vitl_lora_r32_checkpoint48_mouse_canonical_v1_" "full275_20260724"
)
DEFAULT_RESULTS_DIR = Path("results/opensource/dinov3_iml")
DEFAULT_INPUTS = Path("outputs/opensource/mouse_canonical_v1/inputs.jsonl")
DEFAULT_DINOV3_IML_ROOT = Path("/root/.cache/claimforge/third_party/DINOv3-IML")
DEFAULT_DINOV3_ROOT = Path("/root/.cache/claimforge/third_party/dinov3")

MODEL_NAME = "DINOv3-IML"
MODEL_SLUG = "dinov3_iml_cat_vitl_lora_r32_checkpoint48_official"
MODEL_INPUT_SIZE = 512
INTERNAL_LOGIT_SIZE = 32
MASK_THRESHOLD = 0.5
THRESHOLD_OPERATOR = ">"
HISTOGRAM_BINS = 65_536

# These tolerances cover only bounded float32 runtime differences. File
# hashes, dtypes, shapes, geometry metadata, and PNG masks remain exact.
RESIZED_LOGITS_ABSOLUTE_TOLERANCE = 5e-6
MODEL_PROBABILITY_ABSOLUTE_TOLERANCE = 1e-6
NATIVE_RESTORE_ABSOLUTE_TOLERANCE = 3e-6
TRANSFORM_RELATIVE_TOLERANCE = 1e-7

_FORBIDDEN_TOP_LEVEL_RESULT_FIELDS = frozenset(
    {
        "score",
        "decision",
        "detection",
        "classification",
        "classification_threshold",
        "classification_logits",
        "classification_probabilities",
        "class_probabilities",
        "image_score",
        "image_decision",
        "score_source",
        "score_semantics",
    }
)
_FORBIDDEN_TOP_LEVEL_SUMMARY_FIELDS = frozenset(
    {
        "score",
        "decision",
        "detection",
        "classification",
        "classification_threshold",
        "score_by_kind",
        "paired_score_delta",
        "paired_ranking_accuracy",
        "auroc",
        "roc_auc",
        "average_precision",
    }
)
_FORBIDDEN_T1_SEMANTIC_KEYS = frozenset(
    {
        "auroc",
        "roc_auc",
        "average_precision",
        "decision",
        "detection",
        "classification",
        "classification_threshold",
        "classification_logits",
        "classification_probabilities",
        "class_probabilities",
        "image_score",
        "image_decision",
        "score_source",
        "score_semantics",
        "score_by_kind",
        "paired_score_delta",
        "paired_ranking_accuracy",
    }
)
_REPRODUCIBILITY_FIELDS = (
    "raw_logits_model_sha256",
    "raw_logits_model_shape",
    "raw_logits_model_dtype",
    "raw_logits_model_semantics",
    "raw_logits_capture",
    "resized_logits_model_sha256",
    "resized_logits_model_shape",
    "resized_logits_model_dtype",
    "resized_logits_model_semantics",
    "resized_logits_derivation",
    "score_map_model_sha256",
    "score_map_model_shape",
    "score_map_model_dtype",
    "score_map_model_semantics",
    "score_map_sha256",
    "score_map_shape",
    "score_map_dtype",
    "score_map_semantics",
    "score_map_native_source",
    "score_map_native_restore",
    "mask_sha256",
    "mask_shape",
    "mask_dtype",
    "mask_threshold",
    "mask_threshold_operator",
    "localization",
    "preprocess",
)


@dataclass(frozen=True)
class LocalizationPair:
    task_id: str
    domain: str
    real: dict[str, Any]
    forged: dict[str, Any]
    input_row: dict[str, Any]

    @property
    def edit_fraction(self) -> float:
        metrics = self.forged["localization"]["native"]
        pixels = int(metrics["pixels"])
        if pixels <= 0:
            raise ValueError(f"pair {self.task_id} has no native pixels")
        return float(metrics["target_positive_pixels"]) / float(pixels)


def _load_runner_pins() -> SimpleNamespace:
    """Load immutable runner pins without importing the upstream model."""

    from eval.opensource import run_dinov3_iml

    names = (
        "MODEL_REPO_URL",
        "MODEL_SOURCE_COMMIT",
        "SOURCE_FILES",
        "DINOV3_REPO_URL",
        "DINOV3_SOURCE_COMMIT",
        "DINOV3_SOURCE_FILES",
        "CHECKPOINT",
    )
    missing = [name for name in names if not hasattr(run_dinov3_iml, name)]
    if missing:
        raise RuntimeError(
            f"DINOv3-IML runner does not export required audit constants: {missing}"
        )
    return SimpleNamespace(**{name: getattr(run_dinov3_iml, name) for name in names})


def _anchored(path: Path, repo_root: Path) -> Path:
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


def _relative_or_absolute(path: Path, repo_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo_root.resolve()))
    except ValueError:
        return str(path.resolve())


def _require_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} is not a JSON object")
    return value


def _require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise ValueError(f"{label} mismatch: {actual!r} != {expected!r}")


def _require_sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} is not a lowercase SHA-256 digest")
    return value


def _verify_hash(path: Path, expected: Any, label: str) -> None:
    digest = _require_sha256(expected, f"{label} expected hash")
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")
    actual = sha256_file(path)
    if actual != digest:
        raise ValueError(f"{label} SHA-256 mismatch: {actual} != {digest}")


def _git_value(repo: Path, *args: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), *args],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _array_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes(order="C")).hexdigest()


def _selection_contract(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "rank": int(row["rank"]),
            "pair_rank": int(row["pair_rank"]),
            "sample_id": str(row["sample_id"]),
            "task_id": str(row["task_id"]),
            "kind": str(row["kind"]),
            "label": int(row["label"]),
            "canonical_path": str(row["canonical_path"]),
            "canonical_sha256": str(row["canonical_sha256"]),
            "gt_mask_sha256": row.get("gt_mask_sha256"),
        }
        for row in rows
    ]


def _select_manifest_inputs(
    input_rows: list[dict[str, Any]],
    manifest: dict[str, Any],
) -> list[dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(input_rows, start=1):
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError(f"canonical input row {index} has no sample_id")
        if sample_id in by_id:
            raise ValueError(f"canonical inputs contain duplicate ID {sample_id}")
        by_id[sample_id] = row
    ordered = manifest.get("ordered_inputs")
    if not isinstance(ordered, list) or not ordered:
        raise ValueError("run manifest ordered_inputs is empty or invalid")
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(ordered):
        item = _require_mapping(raw, f"ordered input {index}")
        sample_id = item.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError(f"ordered input {index} has no sample_id")
        if sample_id in seen:
            raise ValueError(
                f"run manifest ordered_inputs contains duplicate ID {sample_id}"
            )
        if sample_id not in by_id:
            raise ValueError(f"run manifest selected unknown canonical ID {sample_id}")
        seen.add(sample_id)
        selected.append(by_id[sample_id])
    return selected


def summarize_result_history(
    result_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    histories: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    status_counts: Counter[str] = Counter()
    for line_number, row in enumerate(result_rows, start=1):
        row_id = row.get("id")
        if not isinstance(row_id, str) or not row_id:
            raise ValueError(f"result row {line_number} has no non-empty string id")
        histories[row_id].append((line_number, row))
        status_counts[str(row.get("status"))] += 1

    duplicate_histories: list[dict[str, Any]] = []
    recovered_ids: list[str] = []
    latest_counts: Counter[str] = Counter()
    for row_id, entries in sorted(histories.items()):
        statuses = [str(row.get("status")) for _, row in entries]
        latest_counts[statuses[-1]] += 1
        if len(entries) > 1:
            duplicate_histories.append(
                {
                    "id": row_id,
                    "physical_rows": len(entries),
                    "line_numbers": [line for line, _ in entries],
                    "statuses": statuses,
                }
            )
        if statuses[-1] == "ok" and "error" in statuses[:-1]:
            recovered_ids.append(row_id)
    return {
        "physical_rows": len(result_rows),
        "unique_ids": len(histories),
        "duplicate_rows": len(result_rows) - len(histories),
        "ids_with_multiple_rows": len(duplicate_histories),
        "recovered_error_to_ok": len(recovered_ids),
        "recovered_ids": recovered_ids,
        "historical_status_counts": dict(sorted(status_counts.items())),
        "latest_status_counts": dict(sorted(latest_counts.items())),
        "duplicate_histories": duplicate_histories,
        "latest_policy": "last physical JSONL row for each sample id",
    }


def _latest_by_id(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows, start=1):
        row_id = row.get("id")
        if not isinstance(row_id, str) or not row_id:
            raise ValueError(f"result row {index} has no valid id")
        latest[row_id] = row
    return latest


def _find_t1_semantic_key(value: Any, path: str = "$") -> str | None:
    if isinstance(value, dict):
        for raw_key, child in value.items():
            key = str(raw_key).lower()
            child_path = f"{path}.{raw_key}"
            if (
                key in _FORBIDDEN_T1_SEMANTIC_KEYS
                or key.startswith("classification_")
                or key.endswith("_classification")
                or key.startswith("detection_")
                or key.endswith("_detection")
            ):
                return child_path
            found = _find_t1_semantic_key(child, child_path)
            if found is not None:
                return found
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found = _find_t1_semantic_key(child, f"{path}[{index}]")
            if found is not None:
                return found
    return None


def _reject_t1_contract(
    *,
    manifest: dict[str, Any],
    summary: dict[str, Any],
    result_rows: list[dict[str, Any]],
) -> None:
    inference = _require_mapping(manifest.get("inference"), "manifest inference")
    semantic = _find_t1_semantic_key(inference, "$.inference")
    if semantic is not None:
        raise ValueError(
            f"DINOv3-IML localization-only manifest contains T1 field at {semantic}"
        )
    metrics = _require_mapping(manifest.get("metrics"), "manifest metrics")
    if metrics.get("t1_policy") != "unsupported_no_derived_image_score":
        raise ValueError("manifest does not explicitly exclude derived T1 scores")
    for line_number, row in enumerate(result_rows, start=1):
        present = sorted(_FORBIDDEN_TOP_LEVEL_RESULT_FIELDS.intersection(row))
        if present:
            raise ValueError(
                f"result row {line_number} contains forbidden T1 fields: {present}"
            )
        _require_equal(
            row.get("valid_for_t1"),
            False,
            f"result row {line_number} valid_for_t1",
        )
        semantic = _find_t1_semantic_key(row)
        if semantic is not None:
            raise ValueError(
                f"result row {line_number} contains semantic T1 field at " f"{semantic}"
            )
    present = sorted(_FORBIDDEN_TOP_LEVEL_SUMMARY_FIELDS.intersection(summary))
    if present:
        raise ValueError(f"DINOv3-IML summary contains forbidden T1 fields: {present}")
    semantic = _find_t1_semantic_key(summary)
    if semantic is not None:
        raise ValueError(f"DINOv3-IML summary contains semantic T1 field at {semantic}")
    _require_equal(summary.get("valid_for_t1"), False, "summary valid_for_t1")


def _normalise_source_files(value: Any, label: str) -> dict[str, str]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{label} is empty or invalid")
    result: dict[str, str] = {}
    for index, raw in enumerate(value):
        item = _require_mapping(raw, f"{label} entry {index}")
        path = item.get("path")
        if not isinstance(path, str) or not path:
            raise ValueError(f"{label} entry {index} has no path")
        if path in result:
            raise ValueError(f"{label} repeats path {path}")
        result[path] = _require_sha256(
            item.get("sha256"),
            f"{label} entry {index} SHA-256",
        )
    return result


def _verify_adapter_contract(value: Any, *, repo_root: Path) -> int:
    if not isinstance(value, list) or not value:
        raise ValueError("manifest adapter_contract is empty or invalid")
    paths: set[Path] = set()
    for index, raw in enumerate(value):
        item = _require_mapping(raw, f"adapter contract entry {index}")
        path_value = item.get("path")
        if not isinstance(path_value, str) or not path_value:
            raise ValueError(f"adapter contract entry {index} has no path")
        path = _anchored(Path(path_value), repo_root)
        if path in paths:
            raise ValueError(f"adapter contract repeats path {path}")
        paths.add(path)
        _verify_hash(path, item.get("sha256"), f"adapter contract entry {index}")
    return len(paths)


def validate_provenance(
    *,
    repo_root: Path,
    dinov3_iml_root: Path,
    dinov3_root: Path,
    run_id: str,
    input_path: Path,
    input_rows: list[dict[str, Any]],
    result_rows: list[dict[str, Any]],
    manifest: dict[str, Any],
    summary: dict[str, Any],
) -> dict[str, Any]:
    pins = _load_runner_pins()
    _require_equal(
        manifest.get("schema_version"),
        "opensource_run_manifest_v1",
        "run manifest schema",
    )
    _require_equal(manifest.get("run_id"), run_id, "run manifest ID")
    fingerprint = _require_sha256(
        manifest.get("fingerprint"),
        "run manifest fingerprint",
    )
    immutable = {
        key: value
        for key, value in manifest.items()
        if key not in {"fingerprint", "created_at", "adapter", "environment"}
    }
    computed = hashlib.sha256(stable_json(immutable).encode("utf-8")).hexdigest()
    _require_equal(fingerprint, computed, "run manifest fingerprint")
    runtime_contract = _require_mapping(
        manifest.get("runtime_contract"),
        "manifest runtime contract",
    )
    _require_equal(
        set(runtime_contract),
        {
            "python",
            "packages",
            "optional_imdlbenco_present",
            "accelerator",
            "numerical_flags",
        },
        "runtime contract top-level fields",
    )
    packages = _require_mapping(
        runtime_contract.get("packages"),
        "runtime contract packages",
    )
    required_packages = {
        "torch",
        "peft",
        "transformers",
        "accelerate",
        "huggingface-hub",
        "safetensors",
        "numpy",
        "Pillow",
        "scikit-learn",
    }
    if not required_packages.issubset(packages):
        raise ValueError(
            "runtime contract is missing packages: "
            f"{sorted(required_packages - set(packages))}"
        )
    _require_mapping(runtime_contract.get("python"), "runtime contract python")
    numerical_flags = _require_mapping(
        runtime_contract.get("numerical_flags"),
        "runtime contract numerical flags",
    )
    for key, expected in {
        "cublas_workspace_config": ":4096:8",
        "deterministic_algorithms": True,
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
        "cuda_matmul_allow_tf32": False,
        "cudnn_allow_tf32": False,
        "float32_matmul_precision": "highest",
    }.items():
        _require_equal(
            numerical_flags.get(key),
            expected,
            f"runtime numerical flag {key}",
        )
    if not isinstance(runtime_contract.get("optional_imdlbenco_present"), bool):
        raise ValueError("runtime contract IMDLBenCo presence flag is not boolean")
    _require_mapping(
        runtime_contract.get("accelerator"),
        "runtime contract accelerator",
    )
    _require_equal(
        manifest.get("environment"),
        runtime_contract,
        "manifest environment/runtime contract compatibility copy",
    )

    manifest_input = _require_mapping(manifest.get("input"), "manifest input")
    inputs_digest = sha256_file(input_path)
    _require_equal(
        manifest_input.get("inputs_sha256"),
        inputs_digest,
        "manifest/input JSONL SHA-256",
    )
    inputs_value = manifest_input.get("inputs_manifest")
    if not isinstance(inputs_value, str):
        raise ValueError("manifest has no inputs_manifest path")
    _require_equal(
        _anchored(Path(inputs_value), repo_root),
        input_path.resolve(),
        "manifest/input JSONL path",
    )
    release_value = manifest_input.get("dataset_manifest")
    if not isinstance(release_value, str):
        raise ValueError("manifest has no dataset_manifest path")
    release_path = _anchored(Path(release_value), repo_root)
    _verify_hash(
        release_path,
        manifest_input.get("dataset_manifest_sha256"),
        "canonical dataset manifest",
    )
    release = _require_mapping(
        json.loads(release_path.read_text(encoding="utf-8")),
        "canonical dataset manifest",
    )
    _require_equal(
        release.get("schema_version"),
        "claimforge_mouse_canonical_v1",
        "canonical dataset schema",
    )
    for release_key, manifest_key in (
        ("dataset_id", "dataset_id"),
        ("contract_sha256", "dataset_contract_sha256"),
        ("inputs_sha256", "inputs_sha256"),
    ):
        _require_equal(
            release.get(release_key),
            manifest_input.get(manifest_key),
            f"canonical dataset {release_key}",
        )
    release_inputs = release.get("inputs_path")
    if not isinstance(release_inputs, str):
        raise ValueError("canonical dataset manifest has no inputs_path")
    _require_equal(
        _anchored(Path(release_inputs), repo_root),
        input_path.resolve(),
        "canonical dataset inputs path",
    )

    expected_ids = [str(row["sample_id"]) for row in input_rows]
    if len(expected_ids) != len(set(expected_ids)):
        raise ValueError("selected canonical inputs contain duplicate IDs")
    expected_selection = _selection_contract(input_rows)
    _require_equal(
        manifest.get("ordered_inputs"),
        expected_selection,
        "manifest ordered input selection",
    )
    _require_equal(
        manifest_input.get("selection_sha256"),
        hashlib.sha256(stable_json(expected_selection).encode("utf-8")).hexdigest(),
        "manifest input selection SHA-256",
    )
    _require_equal(
        manifest.get("expected_images"),
        len(input_rows),
        "manifest expected image count",
    )
    expected_kinds_by_task: dict[str, set[str]] = defaultdict(set)
    for row in input_rows:
        expected_kinds_by_task[str(row["task_id"])].add(str(row["kind"]))
    _require_equal(
        manifest.get("expected_complete_pairs"),
        sum(kinds == {"real", "forged"} for kinds in expected_kinds_by_task.values()),
        "manifest expected complete pair count",
    )

    expected_source = {
        str(path): str(digest) for path, digest in dict(pins.SOURCE_FILES).items()
    }
    expected_dinov3_source = {
        str(path): str(digest)
        for path, digest in dict(pins.DINOV3_SOURCE_FILES).items()
    }
    expected_checkpoint = dict(pins.CHECKPOINT)
    model = _require_mapping(manifest.get("model"), "manifest model")
    architecture = _require_mapping(
        model.get("dinov3_architecture_source"),
        "manifest DINOv3 architecture source",
    )
    checkpoint = _require_mapping(model.get("checkpoint"), "manifest checkpoint")
    license_value = _require_mapping(model.get("license"), "manifest license")
    required_model = (
        (model.get("name"), MODEL_NAME, "manifest model name"),
        (model.get("model_slug"), MODEL_SLUG, "manifest model slug"),
        (model.get("repo_url"), pins.MODEL_REPO_URL, "manifest repository URL"),
        (
            Path(str(model.get("source_root"))).resolve(),
            dinov3_iml_root.resolve(),
            "manifest DINOv3-IML source root",
        ),
        (
            model.get("source_commit"),
            pins.MODEL_SOURCE_COMMIT,
            "manifest source commit",
        ),
        (
            architecture.get("repo_url"),
            pins.DINOV3_REPO_URL,
            "manifest DINOv3 repository URL",
        ),
        (
            architecture.get("name"),
            "Meta DINOv3",
            "manifest DINOv3 source name",
        ),
        (
            Path(str(architecture.get("source_root"))).resolve(),
            dinov3_root.resolve(),
            "manifest DINOv3 source root",
        ),
        (
            architecture.get("source_commit"),
            pins.DINOV3_SOURCE_COMMIT,
            "manifest DINOv3 source commit",
        ),
        (
            architecture.get("source_tracked_clean"),
            True,
            "manifest DINOv3 source clean flag",
        ),
        (
            architecture.get("pretrained"),
            False,
            "manifest DINOv3 pretrained flag",
        ),
        (
            architecture.get("separate_backbone_weights_loaded"),
            False,
            "manifest separate backbone weight flag",
        ),
        (
            architecture.get("role"),
            "architecture_only",
            "manifest DINOv3 source role",
        ),
        (model.get("source_tracked_clean"), True, "manifest source clean flag"),
        (
            model.get("variant"),
            "official_CAT_ViT-L16_LoRA-r32_checkpoint-48",
            "manifest model variant",
        ),
        (license_value.get("path"), "LICENSE", "manifest license path"),
        (
            license_value.get("sha256"),
            expected_source["LICENSE"],
            "manifest license SHA-256",
        ),
        (license_value.get("spdx"), "MIT", "manifest license SPDX"),
        (
            license_value.get("scope"),
            "DINOv3-IML_repository_code_only",
            "manifest license scope",
        ),
        (
            license_value.get("checkpoint_license"),
            "not_separately_stated_by_release",
            "manifest checkpoint license statement",
        ),
        (
            model.get("parameter_count"),
            expected_checkpoint["parameters"],
            "manifest parameter count",
        ),
        (
            model.get("buffer_elements"),
            expected_checkpoint["buffers"],
            "manifest buffer count",
        ),
        (
            model.get("trainable_parameter_count"),
            expected_checkpoint["trainable_parameters"],
            "manifest trainable parameter count",
        ),
        (
            model.get("supports_image_level_t1"),
            False,
            "manifest T1 support flag",
        ),
        (model.get("image_score_source"), None, "manifest image score source"),
        (
            model.get("supports_pixel_level_t2"),
            True,
            "manifest T2 support flag",
        ),
        (
            model.get("primary_localization_output"),
            "author_predict_float32_sigmoid_probability",
            "manifest primary localization output",
        ),
    )
    for actual, expected, label in required_model:
        _require_equal(actual, expected, label)
    architecture_license = _require_mapping(
        architecture.get("license"),
        "manifest DINOv3 license",
    )
    for actual, expected, label in (
        (
            architecture_license.get("path"),
            "LICENSE.md",
            "manifest DINOv3 license path",
        ),
        (
            architecture_license.get("sha256"),
            expected_dinov3_source["LICENSE.md"],
            "manifest DINOv3 license SHA-256",
        ),
        (
            architecture_license.get("name"),
            "DINOv3 License Agreement",
            "manifest DINOv3 license name",
        ),
    ):
        _require_equal(actual, expected, label)
    constructor = _require_mapping(
        model.get("constructor"),
        "manifest model constructor",
    )
    for key, expected in {
        "dinov3_model_type": "dinov3_vitl16",
        "image_size": MODEL_INPUT_SIZE,
        "lora_rank": 32,
        "lora_alpha": 64.0,
        "lora_target_modules": ["qkv"],
        "torch_hub_author_calls": 1,
        "weight_downloads_blocked": True,
        "author_from_pretrained_used": False,
        "lora_merged": False,
    }.items():
        _require_equal(
            constructor.get(key),
            expected,
            f"manifest constructor {key}",
        )
    _require_equal(
        _normalise_source_files(model.get("source_files"), "source files"),
        expected_source,
        "manifest source-file pins",
    )
    _require_equal(
        _normalise_source_files(
            architecture.get("source_files"),
            "DINOv3 architecture source files",
        ),
        expected_dinov3_source,
        "manifest DINOv3 source-file pins",
    )
    for key, expected in expected_checkpoint.items():
        _require_equal(checkpoint.get(key), expected, f"manifest checkpoint {key}")
    checkpoint_extras = {
        "strict_load": True,
        "safe_weights_only_load": True,
        "safe_globals": ["argparse.Namespace"],
        "container_selection": "top_level_model_only",
        "schema_fallbacks": False,
        "prefix_rewrites": False,
        "full_state_includes_backbone_lora_and_seg_head": True,
        "separate_backbone_weights_required": False,
    }
    for key, expected in checkpoint_extras.items():
        _require_equal(checkpoint.get(key), expected, f"checkpoint {key}")
    checkpoint_value = checkpoint.get("path")
    if not isinstance(checkpoint_value, str):
        raise ValueError("manifest checkpoint has no path")
    checkpoint_path = Path(checkpoint_value).resolve()
    _verify_hash(
        checkpoint_path,
        expected_checkpoint["sha256"],
        "official DINOv3-IML checkpoint",
    )
    _require_equal(
        checkpoint_path.stat().st_size,
        int(expected_checkpoint["bytes"]),
        "official DINOv3-IML checkpoint byte size",
    )

    if _git_value(dinov3_iml_root, "rev-parse", "HEAD") != pins.MODEL_SOURCE_COMMIT:
        raise ValueError("checked DINOv3-IML source tree is not at the pinned commit")
    if _git_value(
        dinov3_iml_root,
        "status",
        "--short",
        "--untracked-files=no",
    ):
        raise ValueError("checked DINOv3-IML source tree has tracked modifications")
    for relative, digest in expected_source.items():
        _verify_hash(
            dinov3_iml_root / relative,
            digest,
            f"pinned DINOv3-IML source file {relative}",
        )
    if _git_value(dinov3_root, "rev-parse", "HEAD") != pins.DINOV3_SOURCE_COMMIT:
        raise ValueError("checked DINOv3 source tree is not at the pinned commit")
    if _git_value(
        dinov3_root,
        "status",
        "--short",
        "--untracked-files=no",
    ):
        raise ValueError("checked DINOv3 source tree has tracked modifications")
    for relative, digest in expected_dinov3_source.items():
        _verify_hash(
            dinov3_root / relative,
            digest,
            f"pinned DINOv3 architecture source file {relative}",
        )

    inference = _require_mapping(manifest.get("inference"), "manifest inference")
    expected_inference = {
        "precision": "float32",
        "batch_size": 1,
        "deterministic": True,
        "input_source": "canonical_jpeg_original_bytes",
        "decoder": "Pillow.Image.open.convert_RGB",
        "channel_order": "RGB",
        "input_geometry": (
            "direct_stretch_to_512x512_without_aspect_ratio_preservation"
        ),
        "preprocess_protocol": (
            "official_standalone_pillow_rgb_bilinear_stretch_512_imagenet"
        ),
        "resize": "Pillow.Image.resize",
        "resize_interpolation": "Pillow.Image.Resampling.BILINEAR",
        "resize_box": None,
        "resize_reducing_gap": None,
        "input_crop": None,
        "input_reencode": False,
        "normalization": {
            "scale": "float32_divide_255",
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
        "official_model_output": {
            "seg_head_logits_shape": [
                1,
                1,
                INTERNAL_LOGIT_SIZE,
                INTERNAL_LOGIT_SIZE,
            ],
            "logit_resize": ("bilinear_seg_head_logits_32_to_512_align_corners_false"),
            "resized_logits_shape": [
                1,
                1,
                MODEL_INPUT_SIZE,
                MODEL_INPUT_SIZE,
            ],
            "probability": "single_sigmoid_after_logit_resize",
            "captured_by": "one_forward_hook_on_author_model_seg_head",
            "author_predict_calls_per_image": 1,
        },
        "native_compatibility_adapter": {
            "purpose": "CLAIMFORGE cross-method native-resolution comparison",
            "source": "official_model_512_probability_not_logits",
            "operation": (
                "bilinear_official_model_512_probability_to_native_"
                "align_corners_false"
            ),
            "mode": "bilinear",
            "align_corners": False,
            "threshold_after_restore": True,
            "official_model_space_retained_as_auxiliary": True,
        },
        "mask_threshold": MASK_THRESHOLD,
        "mask_threshold_comparison": "strict_greater_than",
        "test_time_augmentation": False,
        "ensemble": False,
        "forward_passes_per_image": 1,
    }
    for key, expected in expected_inference.items():
        _require_equal(inference.get(key), expected, f"manifest inference {key}")
    seed = inference.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("manifest inference seed is not an integer")

    metrics = _require_mapping(manifest.get("metrics"), "manifest metrics")
    expected_metrics = {
        "task": "T2_pixel_localization_only",
        "positive_class": "manipulated_pixel",
        "t1_policy": "unsupported_no_derived_image_score",
        "primary_localization_space": "native",
        "auxiliary_localization_space": "model_512",
        "mask_threshold": MASK_THRESHOLD,
        "threshold_comparison": "strict_greater_than",
        "prediction_inversion": False,
        "native_gt": "exact_canonical_mask",
        "model_space_gt_resize": ("Pillow.Image.Resampling.NEAREST_to_512x512"),
        "forged_pixel_ap_only": True,
        "bootstrap_unit": "task_id_pair",
    }
    for key, expected in expected_metrics.items():
        _require_equal(metrics.get(key), expected, f"manifest metrics {key}")
    bootstrap_samples = metrics.get("bootstrap_samples")
    if (
        isinstance(bootstrap_samples, bool)
        or not isinstance(bootstrap_samples, int)
        or bootstrap_samples <= 0
    ):
        raise ValueError("manifest bootstrap sample count is invalid")

    expected_artifacts = {
        "raw_logits_model_32": {
            "format": "npy",
            "dtype": "float32",
            "shape": [INTERNAL_LOGIT_SIZE, INTERNAL_LOGIT_SIZE],
            "semantics": "official_seg_head_pre_resize_logits",
            "captured_from": "one_forward_hook_on_author_model_seg_head",
        },
        "raw_logits_model_512": {
            "format": "npy",
            "dtype": "float32",
            "shape": [MODEL_INPUT_SIZE, MODEL_INPUT_SIZE],
            "semantics": "official_bilinear_resized_pre_sigmoid_logits",
            "derivation": ("bilinear_seg_head_logits_32_to_512_align_corners_false"),
        },
        "score_maps_model_512": {
            "format": "npy",
            "dtype": "float32",
            "shape": [MODEL_INPUT_SIZE, MODEL_INPUT_SIZE],
            "semantics": "official_author_predict_sigmoid_probability",
        },
        "score_maps_native": {
            "format": "npy",
            "dtype": "float32",
            "shape": "native_HxW",
            "semantics": "model_512_probability_restored_to_native",
            "restore": (
                "bilinear_official_model_512_probability_to_native_"
                "align_corners_false"
            ),
        },
        "masks_native": {
            "format": "lossless_png",
            "dtype": "uint8",
            "values": [0, 255],
            "relation": "score_map_native > 0.5",
        },
    }
    _require_equal(
        manifest.get("artifacts"),
        expected_artifacts,
        "manifest artifact contract",
    )
    _reject_t1_contract(
        manifest=manifest,
        summary=summary,
        result_rows=result_rows,
    )

    artifact_dir_value = manifest.get("artifact_dir")
    if not isinstance(artifact_dir_value, str) or not artifact_dir_value:
        raise ValueError("manifest has no artifact_dir")
    artifact_dir = _anchored(Path(artifact_dir_value), repo_root)
    expected_by_id = {str(row["sample_id"]): row for row in input_rows}
    seen: set[str] = set()
    latest: dict[str, dict[str, Any]] = {}
    for line_number, row in enumerate(result_rows, start=1):
        row_id = row.get("id")
        if not isinstance(row_id, str) or row_id not in expected_by_id:
            raise ValueError(f"unexpected result ID at row {line_number}: {row_id}")
        seen.add(row_id)
        latest[row_id] = row
        source = expected_by_id[row_id]
        expected_values = {
            "schema_version": "opensource_result_v1",
            "run_id": run_id,
            "run_manifest_fingerprint": fingerprint,
            "input_manifest_sha256": inputs_digest,
            "id": row_id,
            "rank": int(source["rank"]),
            "task_id": str(source["task_id"]),
            "pair_rank": int(source["pair_rank"]),
            "domain": str(source["domain"]),
            "kind": str(source["kind"]),
            "label": int(source["label"]),
            "image_path": str(source["canonical_path"]),
            "image_sha256": str(source["canonical_sha256"]),
            "image_size": [int(source["width"]), int(source["height"])],
            "gt_mask_kind": str(source["gt_mask_kind"]),
            "gt_mask_sha256": source.get("gt_mask_sha256"),
            "edit_region_xyxy": [int(value) for value in source["edit_region_xyxy"]],
            "model": MODEL_NAME,
            "model_slug": MODEL_SLUG,
            "model_source_commit": pins.MODEL_SOURCE_COMMIT,
            "dinov3_source_commit": pins.DINOV3_SOURCE_COMMIT,
            "checkpoint_sha256": expected_checkpoint["sha256"],
            "checkpoint_epoch": expected_checkpoint["epoch"],
            "valid_for_t1": False,
            "valid_for_t2": True,
            "t1_policy": "unsupported_no_derived_image_score",
        }
        for key, expected in expected_values.items():
            _require_equal(
                row.get(key),
                expected,
                f"result row {line_number} field {key}",
            )
        status = row.get("status")
        if status not in {"ok", "error"}:
            raise ValueError(f"result row {line_number} has invalid status {status!r}")
        _require_equal(
            row.get("valid_for_metrics"),
            status == "ok",
            f"result row {line_number} valid_for_metrics",
        )
        if status == "ok":
            _require_equal(
                row.get("mask_threshold"),
                MASK_THRESHOLD,
                f"result row {line_number} mask threshold",
            )
            _require_equal(
                row.get("mask_threshold_operator"),
                THRESHOLD_OPERATOR,
                f"result row {line_number} threshold operator",
            )
            _require_equal(
                row.get("mask_dtype"),
                "uint8",
                f"result row {line_number} mask dtype",
            )
            artifact_paths = {
                "raw_logits_model_path": (
                    artifact_dir / "raw_logits_model_32" / f"{row_id}.npy"
                ),
                "resized_logits_model_path": (
                    artifact_dir / "raw_logits_model_512" / f"{row_id}.npy"
                ),
                "score_map_model_path": (
                    artifact_dir / "score_maps_model_512" / f"{row_id}.npy"
                ),
                "score_map_path": (
                    artifact_dir / "score_maps_native" / f"{row_id}.npy"
                ),
                "mask_path": (artifact_dir / "masks_native" / f"{row_id}.png"),
            }
            for path_key, expected_path in artifact_paths.items():
                path_value = row.get(path_key)
                if not isinstance(path_value, str):
                    raise ValueError(f"result row {line_number} has no {path_key}")
                _require_equal(
                    _anchored(Path(path_value), repo_root),
                    expected_path,
                    f"result row {line_number} {path_key}",
                )
            for hash_key in (
                "raw_logits_model_sha256",
                "resized_logits_model_sha256",
                "score_map_model_sha256",
                "score_map_sha256",
                "mask_sha256",
            ):
                _require_sha256(
                    row.get(hash_key),
                    f"result row {line_number} {hash_key}",
                )
    if seen != set(expected_ids):
        missing = sorted(set(expected_ids) - seen)
        raise ValueError(f"result history is missing expected IDs: {missing[:5]}")

    _require_equal(
        summary.get("schema_version"),
        "opensource_summary_v1",
        "summary schema",
    )
    for actual, expected, label in (
        (summary.get("run_id"), run_id, "summary run ID"),
        (summary.get("condition"), manifest.get("condition"), "summary condition"),
        (summary.get("model"), MODEL_NAME, "summary model"),
        (summary.get("model_slug"), MODEL_SLUG, "summary model slug"),
        (
            summary.get("checkpoint_sha256"),
            expected_checkpoint["sha256"],
            "summary checkpoint SHA-256",
        ),
        (
            summary.get("checkpoint_epoch"),
            expected_checkpoint["epoch"],
            "summary checkpoint epoch",
        ),
        (
            summary.get("input_manifest_sha256"),
            inputs_digest,
            "summary input manifest SHA-256",
        ),
        (
            summary.get("run_manifest_fingerprint"),
            fingerprint,
            "summary run manifest fingerprint",
        ),
        (summary.get("valid_for_t1"), False, "summary valid_for_t1"),
        (summary.get("valid_for_t2"), True, "summary valid_for_t2"),
        (
            summary.get("t1_policy"),
            "unsupported_no_derived_image_score",
            "summary T1 policy",
        ),
    ):
        _require_equal(actual, expected, label)
    valid_latest = [row for row in latest.values() if row.get("status") == "ok"]
    coverage = _require_mapping(summary.get("coverage"), "summary coverage")
    expected_coverage = {
        "expected_images": len(input_rows),
        "result_images": len(latest),
        "valid_images": len(valid_latest),
        "error_images": len(latest) - len(valid_latest),
        "missing_images": len(input_rows) - len(latest),
    }
    for key, expected in expected_coverage.items():
        _require_equal(coverage.get(key), expected, f"summary coverage {key}")
    by_task: dict[str, set[str]] = defaultdict(set)
    for row in valid_latest:
        by_task[str(row["task_id"])].add(str(row["kind"]))
    complete_pairs = sum(kinds == {"real", "forged"} for kinds in by_task.values())
    paired = _require_mapping(summary.get("paired_coverage"), "paired coverage")
    for key, expected in {
        "complete_pairs": complete_pairs,
        "paired_images": complete_pairs * 2,
        "unpaired_valid_images": len(valid_latest) - complete_pairs * 2,
    }.items():
        _require_equal(paired.get(key), expected, f"paired coverage {key}")
    expected_scope = {
        "primary_task": "T2_localization",
        "valid_for_t1": False,
        "valid_for_t2": True,
        "primary_localization_space": "native",
        "auxiliary_localization_space": "model_512",
        "localization_semantics": (
            "dinov3_iml_sigmoid_manipulation_probability_float32"
        ),
        "model_space_probability_source": (
            "sigmoid_bilinear_align_corners_false_seg_head_logits_32_to_512"
        ),
        "native_probability_source": (
            "bilinear_align_corners_false_resize_of_model_512_probability"
        ),
        "probability_dtype": "float32",
        "mask_threshold": MASK_THRESHOLD,
        "threshold_operator": THRESHOLD_OPERATOR,
    }
    _require_equal(summary.get("task_scope"), expected_scope, "summary task scope")
    pair_bootstrap = _require_mapping(
        summary.get("pair_bootstrap"),
        "summary pair bootstrap",
    )
    _require_equal(
        pair_bootstrap.get("bootstrap_samples"),
        bootstrap_samples,
        "summary bootstrap samples",
    )
    _require_equal(
        pair_bootstrap.get("seed"),
        seed,
        "summary bootstrap seed",
    )
    recomputed_summary = summarize_dinov3_iml_results(
        result_rows,
        input_rows,
        mask_threshold=MASK_THRESHOLD,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
    )
    for key, expected in recomputed_summary.items():
        _require_equal(
            summary.get(key),
            expected,
            f"recomputed summary field {key}",
        )
    adapter_files = _verify_adapter_contract(
        manifest.get("adapter_contract"),
        repo_root=repo_root,
    )
    return {
        "status": "ok",
        "run_manifest_fingerprint": fingerprint,
        "inputs_sha256": inputs_digest,
        "checkpoint_sha256": expected_checkpoint["sha256"],
        "checkpoint_bytes": expected_checkpoint["bytes"],
        "physical_result_rows_validated": len(result_rows),
        "latest_result_rows_validated": len(latest),
        "expected_unique_result_ids": len(input_rows),
        "full_mouse_unique_550": len(latest) == 550,
        "pinned_source_files_validated": (
            len(expected_source) + len(expected_dinov3_source)
        ),
        "adapter_contract_files_validated": adapter_files,
        "checks": [
            "immutable manifest fingerprint and canonical release hashes",
            (
                "official clean DINOv3-IML and checkpoint-era DINOv3 "
                "architecture commits with pinned source-file hashes"
            ),
            "official checkpoint SHA-256, byte size, and strict-load schema",
            (
                "official preprocessing, output, native adapter, and "
                "T2-only contracts"
            ),
            "every physical result row identity and provenance",
            (
                "latest-row coverage and every aggregate summary metric "
                "independently recomputed"
            ),
            "adapter file hashes",
        ],
    }


def _load_pairs(
    result_rows: list[dict[str, Any]],
    input_rows: list[dict[str, Any]],
) -> list[LocalizationPair]:
    latest = _latest_by_id(result_rows)
    expected_ids = {str(row["sample_id"]) for row in input_rows}
    if set(latest) != expected_ids:
        missing = sorted(expected_ids - set(latest))
        unexpected = sorted(set(latest) - expected_ids)
        raise ValueError(
            f"result/input ID mismatch: missing={missing[:5]} "
            f"unexpected={unexpected[:5]}"
        )
    if any(row.get("status") != "ok" for row in latest.values()):
        raise ValueError("analysis requires every latest result row to be successful")
    inputs_by_id = {str(row["sample_id"]): row for row in input_rows}
    by_task: dict[str, dict[str, dict[str, Any]]] = {}
    for row in latest.values():
        task_id = str(row["task_id"])
        kind = str(row["kind"])
        if kind in by_task.setdefault(task_id, {}):
            raise ValueError(f"duplicate {kind} result within pair {task_id}")
        by_task[task_id][kind] = row
    pairs: list[LocalizationPair] = []
    for task_id, values in by_task.items():
        if set(values) != {"real", "forged"}:
            raise ValueError(f"incomplete pair for {task_id}: {sorted(values)}")
        real, forged = values["real"], values["forged"]
        if real.get("label") != 0 or forged.get("label") != 1:
            raise ValueError(f"invalid pair labels within {task_id}")
        if real.get("domain") != forged.get("domain"):
            raise ValueError(f"domain mismatch within {task_id}")
        if int(real["pair_rank"]) != int(forged["pair_rank"]):
            raise ValueError(f"pair-rank mismatch within {task_id}")
        pairs.append(
            LocalizationPair(
                task_id=task_id,
                domain=str(forged["domain"]),
                real=real,
                forged=forged,
                input_row=inputs_by_id[str(forged["id"])],
            )
        )
    return sorted(pairs, key=lambda pair: int(pair.forged["pair_rank"]))


def _preprocess_evidence(
    image_path: Path,
) -> tuple[
    dict[str, Any],
    np.ndarray,
    tuple[int, int],
]:
    """Independently replay the released forced-square preprocessing."""

    with image_path.open("rb") as handle:
        with Image.open(handle) as opened:
            decoder_format = opened.format
            decoded_image = opened.convert("RGB")
            decoded = np.asarray(decoded_image, dtype=np.uint8)
    if decoded.ndim != 3 or decoded.shape[2] != 3 or decoded.dtype != np.uint8:
        raise ValueError(f"unexpected decoded image array: {decoded.shape}")
    native_height, native_width = decoded.shape[:2]
    if native_width <= 0 or native_height <= 0:
        raise ValueError("native image dimensions must be positive")
    resized = np.asarray(
        Image.fromarray(decoded, mode="RGB").resize(
            (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE),
            resample=Image.Resampling.BILINEAR,
            reducing_gap=None,
        ),
        dtype=np.uint8,
    )
    normalized = resized.astype(np.float32) / np.float32(255.0)
    normalized = (
        normalized - np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
    ) / np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
    tensor = np.ascontiguousarray(normalized.transpose(2, 0, 1))
    evidence = {
        "protocol": ("official_standalone_pillow_rgb_bilinear_stretch_512_imagenet"),
        "reference": "upstream_inference._load_and_preprocess",
        "decoder": "Pillow.Image.open.convert_RGB",
        "decoder_format": decoder_format,
        "channel_order": "RGB",
        "native_size_wh": [native_width, native_height],
        "model_size_wh": [MODEL_INPUT_SIZE, MODEL_INPUT_SIZE],
        "geometry": "direct_stretch_without_aspect_ratio_preservation",
        "resize": "Pillow.Image.resize",
        "resize_interpolation": "Pillow.Image.Resampling.BILINEAR",
        "resize_box": None,
        "resize_reducing_gap": None,
        "input_crop": None,
        "input_reencode": False,
        "normalization": {
            "scale": "float32_divide_255",
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
        "tensor_shape": list(tensor.shape),
        "tensor_dtype": str(tensor.dtype),
        "tensor_sha256": _array_sha256(tensor),
    }
    return (
        evidence,
        tensor,
        (native_width, native_height),
    )


def _sigmoid_float32(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float32)
    result = np.empty_like(values)
    positive = values >= np.float32(0.0)
    result[positive] = np.float32(1.0) / (np.float32(1.0) + np.exp(-values[positive]))
    exponentials = np.exp(values[~positive])
    result[~positive] = exponentials / (np.float32(1.0) + exponentials)
    return np.ascontiguousarray(result, dtype=np.float32)


def _fma_float32(
    multiplier: np.ndarray | np.float32,
    multiplicand: np.ndarray | np.float32,
    addend: np.ndarray | np.float32,
) -> np.ndarray:
    """Emulate one correctly rounded float32 fused multiply-add in NumPy."""

    # Products of float32 operands and their relevant sums are exactly
    # representable at float64 precision here. Casting once at the end therefore
    # reproduces CUDA's fma(a, b, c), instead of rounding a*b before adding c.
    return np.asarray(
        np.asarray(multiplier, dtype=np.float64)
        * np.asarray(multiplicand, dtype=np.float64)
        + np.asarray(addend, dtype=np.float64),
        dtype=np.float32,
    )


def _bilinear_from_coordinates(
    source: np.ndarray,
    *,
    x: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    """Independently replay bilinear interpolation with float32 rounding."""

    source_height, source_width = source.shape
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    x1 = np.minimum(x0 + 1, source_width - 1)
    y1 = np.minimum(y0 + 1, source_height - 1)
    wx = (x - x0.astype(np.float32))[None, :]
    wy = (y - y0.astype(np.float32))[:, None]
    wx0 = np.float32(1.0) - wx
    wy0 = np.float32(1.0) - wy

    # Use one explicit nested float32-FMA order. The audit tolerance covers
    # bounded backend/compiler differences in operation order.
    horizontal = _fma_float32(
        wx0,
        source[:, x0],
        np.multiply(wx, source[:, x1], dtype=np.float32),
    )
    restored = _fma_float32(
        wy0,
        horizontal[y0, :],
        np.multiply(wy, horizontal[y1, :], dtype=np.float32),
    )
    return np.ascontiguousarray(restored, dtype=np.float32)


def _bilinear_align_corners_false(
    score_map: np.ndarray,
    *,
    width: int,
    height: int,
) -> np.ndarray:
    source = np.asarray(score_map, dtype=np.float32)
    if source.ndim != 2 or source.size == 0:
        raise ValueError("source score map must be a non-empty 2D array")
    if width <= 0 or height <= 0:
        raise ValueError("output dimensions must be positive")
    source_height, source_width = source.shape
    if source.shape == (height, width):
        return np.ascontiguousarray(source)
    x = _fma_float32(
        np.float32(np.float32(source_width) / np.float32(width)),
        np.arange(width, dtype=np.float32) + np.float32(0.5),
        np.float32(-0.5),
    )
    y = _fma_float32(
        np.float32(np.float32(source_height) / np.float32(height)),
        np.arange(height, dtype=np.float32) + np.float32(0.5),
        np.float32(-0.5),
    )
    x = np.maximum(x, np.float32(0.0))
    y = np.maximum(y, np.float32(0.0))
    return _bilinear_from_coordinates(source, x=x, y=y)


def _nearest_resize_mask(
    target: np.ndarray,
    *,
    width: int,
    height: int,
) -> np.ndarray:
    source = np.asarray(target, dtype=np.uint8)
    if source.ndim != 2 or source.size == 0:
        raise ValueError("source target must be a non-empty 2D array")
    if width <= 0 or height <= 0:
        raise ValueError("target dimensions must be positive")
    if source.shape == (height, width):
        return np.ascontiguousarray(source > 0)
    resized = np.asarray(
        Image.fromarray(source, mode="L").resize(
            (width, height),
            resample=Image.Resampling.NEAREST,
        ),
        dtype=np.uint8,
    )
    return np.ascontiguousarray(resized > 0)


def _load_float32_map(
    path: Path,
    *,
    expected_shape: tuple[int, int],
    label: str,
    probability: bool,
) -> np.ndarray:
    value = np.load(path, mmap_mode="r", allow_pickle=False)
    if value.shape != expected_shape:
        raise ValueError(f"invalid {label} shape: {value.shape} != {expected_shape}")
    if value.dtype != np.float32:
        raise ValueError(f"invalid {label} dtype: {value.dtype} != float32")
    if not np.isfinite(value).all():
        raise ValueError(f"non-finite {label}")
    if probability and (float(value.min()) < 0.0 or float(value.max()) > 1.0):
        raise ValueError(f"out-of-range {label}")
    return np.asarray(value)


def _compare_metric(recorded: Any, expected: Any, label: str) -> None:
    if expected is None:
        if recorded is not None:
            raise ValueError(f"{label} mismatch: {recorded!r} != None")
        return
    if isinstance(expected, float):
        if (
            isinstance(recorded, bool)
            or not isinstance(recorded, (int, float))
            or not math.isfinite(float(recorded))
            or not math.isclose(
                float(recorded),
                expected,
                rel_tol=1e-6,
                abs_tol=1e-7,
            )
        ):
            raise ValueError(f"{label} mismatch: {recorded!r} != {expected!r}")
        return
    _require_equal(recorded, expected, label)


def _validate_metrics(
    result: dict[str, Any],
    *,
    score_map: np.ndarray,
    target: np.ndarray,
    space: str,
) -> None:
    localization = _require_mapping(
        result.get("localization"),
        f"localization for {result['id']}",
    )
    recorded = _require_mapping(
        localization.get(space),
        f"{space} localization for {result['id']}",
    )
    expected = binary_pixel_metrics_strict(
        score_map,
        target,
        MASK_THRESHOLD,
        include_ap=result.get("kind") == "forged",
    )
    if set(recorded) != set(expected):
        raise ValueError(f"{space} localization field set mismatch for {result['id']}")
    for key, value in expected.items():
        _compare_metric(
            recorded.get(key),
            value,
            f"{space} localization {key} for {result['id']}",
        )


def _load_target(
    *,
    result: dict[str, Any],
    input_row: dict[str, Any],
    repo_root: Path,
    width: int,
    height: int,
    checked_paths: set[Path],
) -> np.ndarray:
    if result["kind"] == "real":
        _require_equal(
            input_row.get("gt_mask_kind"),
            "all_zero",
            f"real GT kind for {result['id']}",
        )
        if input_row.get("gt_mask_path") is not None:
            raise ValueError(f"real input unexpectedly has a GT file: {result['id']}")
        return np.zeros((height, width), dtype=bool)
    path_value = input_row.get("gt_mask_path")
    if not isinstance(path_value, str):
        raise ValueError(f"forged input has no GT mask: {result['id']}")
    path = _anchored(Path(path_value), repo_root)
    _verify_hash(
        path,
        input_row.get("gt_mask_sha256"),
        f"ground-truth mask {result['id']}",
    )
    checked_paths.add(path)
    with Image.open(path) as opened:
        target = np.asarray(opened.convert("L"), dtype=np.uint8) > 0
    if target.shape != (height, width):
        raise ValueError(
            f"ground-truth shape mismatch for {result['id']}: "
            f"{target.shape} != {(height, width)}"
        )
    if not target.any():
        raise ValueError(f"forged ground truth is empty for {result['id']}")
    return target


def _descriptive(values: Iterable[float]) -> dict[str, float | int | None]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return {"count": 0, "mean": None, "median": None, "min": None, "max": None}
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def _best_from_histograms(
    all_hist: np.ndarray,
    positive_hist: np.ndarray,
) -> dict[str, Any]:
    if all_hist.shape != positive_hist.shape or all_hist.ndim != 1:
        raise ValueError("oracle histograms have incompatible shapes")
    bins = int(all_hist.size)
    greater_all = np.zeros(bins, dtype=np.int64)
    greater_positive = np.zeros(bins, dtype=np.int64)
    if bins > 1:
        greater_all[:-1] = np.cumsum(all_hist[:0:-1], dtype=np.int64)[::-1]
        greater_positive[:-1] = np.cumsum(
            positive_hist[:0:-1],
            dtype=np.int64,
        )[::-1]
    tp = greater_positive
    fp = greater_all - tp
    fn = int(np.sum(positive_hist)) - tp
    denominator = 2 * tp + fp + fn
    f1 = np.divide(
        2.0 * tp,
        denominator,
        out=np.zeros_like(tp, dtype=np.float64),
        where=denominator > 0,
    )
    iou_denominator = tp + fp + fn
    iou = np.divide(
        tp,
        iou_denominator,
        out=np.zeros_like(tp, dtype=np.float64),
        where=iou_denominator > 0,
    )
    best = int(np.argmax(f1))
    return {
        "histogram_bins": bins,
        "threshold": best / (bins - 1),
        "comparison": THRESHOLD_OPERATOR,
        "micro_f1": float(f1[best]),
        "micro_iou": float(iou[best]),
        "tp": int(tp[best]),
        "fp": int(fp[best]),
        "fn": int(fn[best]),
    }


def _oracle_histograms(
    score_map: np.ndarray,
    target: np.ndarray,
    *,
    bins: int,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    scores = np.asarray(score_map, dtype=np.float32)
    truth = np.asarray(target, dtype=bool)
    indices = np.minimum(
        np.floor(scores * np.float32(bins - 1)).astype(np.int64),
        bins - 1,
    )
    all_hist = np.bincount(indices.reshape(-1), minlength=bins).astype(np.int64)
    positive_hist = np.bincount(
        indices[truth],
        minlength=bins,
    ).astype(np.int64)
    best = _best_from_histograms(all_hist, positive_hist)
    return (
        {
            "histogram_bins": bins,
            "threshold": best["threshold"],
            "comparison": THRESHOLD_OPERATOR,
            "f1": best["micro_f1"],
            "iou": best["micro_iou"],
            "tp": best["tp"],
            "fp": best["fp"],
            "fn": best["fn"],
        },
        all_hist,
        positive_hist,
    )


def audit_artifacts(
    pairs: list[LocalizationPair],
    *,
    repo_root: Path,
    histogram_bins: int | None,
) -> dict[str, Any]:
    if not pairs:
        raise ValueError("artifact audit requires at least one pair")
    global_all = (
        np.zeros(histogram_bins, dtype=np.int64) if histogram_bins is not None else None
    )
    global_positive = (
        np.zeros(histogram_bins, dtype=np.int64) if histogram_bins is not None else None
    )
    per_image_best: list[dict[str, Any]] = []
    checked_paths: set[Path] = set()
    owners: dict[Path, str] = {}
    box_ious: list[float] = []
    box_coverages: list[float] = []
    prediction_inside: list[float] = []
    box_overlap = 0
    box_hits = 0
    maximum_resized_logits_replay_error = 0.0
    maximum_model_replay_error = 0.0
    maximum_native_replay_error = 0.0

    for pair in pairs:
        forged_target: np.ndarray | None = None
        forged_native: np.ndarray | None = None
        forged_prediction: np.ndarray | None = None
        for result in (pair.real, pair.forged):
            result_id = str(result["id"])
            input_row = (
                pair.input_row
                if result["kind"] == "forged"
                else {"gt_mask_kind": "all_zero", "gt_mask_path": None}
            )
            image_path = _anchored(Path(str(result["image_path"])), repo_root)
            _verify_hash(
                image_path,
                result.get("image_sha256"),
                f"canonical image {result_id}",
            )
            checked_paths.add(image_path)
            evidence, tensor, native_size = _preprocess_evidence(image_path)
            width, height = native_size
            if tensor.shape != (3, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE):
                raise ValueError(f"invalid input tensor shape for {result_id}")
            _require_equal(
                result.get("image_size"),
                [width, height],
                f"decoded image size for {result_id}",
            )
            preprocess = _require_mapping(
                result.get("preprocess"),
                f"preprocess metadata for {result_id}",
            )
            _require_equal(preprocess, evidence, f"preprocess for {result_id}")
            specifications = (
                (
                    "raw_logits_model_path",
                    "raw_logits_model_sha256",
                    "raw model logits",
                ),
                (
                    "resized_logits_model_path",
                    "resized_logits_model_sha256",
                    "512x512 model logits",
                ),
                (
                    "score_map_model_path",
                    "score_map_model_sha256",
                    "model probability",
                ),
                ("score_map_path", "score_map_sha256", "native probability"),
                ("mask_path", "mask_sha256", "native threshold mask"),
            )
            resolved: dict[str, Path] = {}
            for path_key, hash_key, label in specifications:
                path_value = result.get(path_key)
                if not isinstance(path_value, str):
                    raise ValueError(f"{label} for {result_id} has no path")
                path = _anchored(Path(path_value), repo_root)
                previous = owners.get(path)
                if previous is not None and previous != result_id:
                    raise ValueError(
                        f"artifact path {path} is shared by {previous} and "
                        f"{result_id}"
                    )
                owners[path] = result_id
                _verify_hash(path, result.get(hash_key), f"{label} {result_id}")
                checked_paths.add(path)
                resolved[path_key] = path

            logits = _load_float32_map(
                resolved["raw_logits_model_path"],
                expected_shape=(INTERNAL_LOGIT_SIZE, INTERNAL_LOGIT_SIZE),
                label=f"raw 32x32 model logits for {result_id}",
                probability=False,
            )
            resized_logits = _load_float32_map(
                resolved["resized_logits_model_path"],
                expected_shape=(MODEL_INPUT_SIZE, MODEL_INPUT_SIZE),
                label=f"resized 512x512 model logits for {result_id}",
                probability=False,
            )
            model_probability = _load_float32_map(
                resolved["score_map_model_path"],
                expected_shape=(MODEL_INPUT_SIZE, MODEL_INPUT_SIZE),
                label=f"model probability for {result_id}",
                probability=True,
            )
            native_probability = _load_float32_map(
                resolved["score_map_path"],
                expected_shape=(height, width),
                label=f"native probability for {result_id}",
                probability=True,
            )
            metadata = {
                "raw_logits_model_shape": [
                    INTERNAL_LOGIT_SIZE,
                    INTERNAL_LOGIT_SIZE,
                ],
                "raw_logits_model_dtype": "float32",
                "raw_logits_model_semantics": ("official_seg_head_pre_resize_logits"),
                "raw_logits_capture": "one_forward_hook_on_author_model_seg_head",
                "resized_logits_model_shape": [
                    MODEL_INPUT_SIZE,
                    MODEL_INPUT_SIZE,
                ],
                "resized_logits_model_dtype": "float32",
                "resized_logits_model_semantics": (
                    "official_bilinear_resized_pre_sigmoid_logits"
                ),
                "resized_logits_derivation": (
                    "bilinear_seg_head_logits_32_to_512_align_corners_false"
                ),
                "score_map_model_shape": [MODEL_INPUT_SIZE, MODEL_INPUT_SIZE],
                "score_map_model_dtype": "float32",
                "score_map_model_semantics": (
                    "official_author_predict_sigmoid_probability"
                ),
                "score_map_shape": [height, width],
                "score_map_dtype": "float32",
                "score_map_semantics": ("model_512_probability_restored_to_native"),
                "score_map_native_source": "official_model_512_probability",
                "score_map_native_restore": (
                    "bilinear_official_model_512_probability_to_native_"
                    "align_corners_false"
                ),
                "mask_dtype": "uint8",
            }
            for key, expected in metadata.items():
                _require_equal(
                    result.get(key),
                    expected,
                    f"{key} metadata for {result_id}",
                )

            expected_resized_logits = _bilinear_align_corners_false(
                logits,
                width=MODEL_INPUT_SIZE,
                height=MODEL_INPUT_SIZE,
            )
            resized_logits_replay_error = float(
                np.max(np.abs(resized_logits - expected_resized_logits))
            )
            maximum_resized_logits_replay_error = max(
                maximum_resized_logits_replay_error,
                resized_logits_replay_error,
            )
            if not np.allclose(
                resized_logits,
                expected_resized_logits,
                rtol=TRANSFORM_RELATIVE_TOLERANCE,
                atol=RESIZED_LOGITS_ABSOLUTE_TOLERANCE,
            ):
                raise ValueError(
                    "512x512 logits are not the align_corners=False "
                    "bilinear resize of captured 32x32 logits for "
                    f"{result_id}; max_abs={resized_logits_replay_error}"
                )

            expected_model = _sigmoid_float32(expected_resized_logits)
            model_replay_error = float(
                np.max(np.abs(model_probability - expected_model))
            )
            maximum_model_replay_error = max(
                maximum_model_replay_error,
                model_replay_error,
            )
            if not np.allclose(
                model_probability,
                expected_model,
                rtol=TRANSFORM_RELATIVE_TOLERANCE,
                atol=MODEL_PROBABILITY_ABSOLUTE_TOLERANCE,
            ):
                raise ValueError(
                    "model probability is not sigmoid of independently "
                    "resized 512x512 logits for "
                    f"{result_id}; max_abs={model_replay_error}"
                )
            if not np.array_equal(
                model_probability > MASK_THRESHOLD,
                expected_model > MASK_THRESHOLD,
            ):
                raise ValueError(
                    "model probability threshold map disagrees with "
                    f"independent sigmoid(logits) > 0.5 for {result_id}"
                )
            if (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE) == (width, height):
                expected_native = np.ascontiguousarray(expected_model)
            else:
                expected_native = _bilinear_align_corners_false(
                    expected_model,
                    width=width,
                    height=height,
                )
            native_replay_error = float(
                np.max(np.abs(native_probability - expected_native))
            )
            maximum_native_replay_error = max(
                maximum_native_replay_error,
                native_replay_error,
            )
            if not np.allclose(
                native_probability,
                expected_native,
                rtol=TRANSFORM_RELATIVE_TOLERANCE,
                atol=NATIVE_RESTORE_ABSOLUTE_TOLERANCE,
            ):
                raise ValueError(
                    "native probability is not the bilinear "
                    "align_corners=False restore of model probability for "
                    f"{result_id}; max_abs={native_replay_error}"
                )

            with Image.open(resolved["mask_path"]) as opened:
                if opened.mode != "L":
                    raise ValueError(
                        f"native threshold mask is not mode L for {result_id}"
                    )
                mask = np.asarray(opened, dtype=np.uint8)
            if mask.shape != (height, width):
                raise ValueError(f"invalid native mask shape for {result_id}")
            _require_equal(
                result.get("mask_shape"),
                [height, width],
                f"mask shape metadata for {result_id}",
            )
            if not set(np.unique(mask).tolist()).issubset({0, 255}):
                raise ValueError(f"native mask is not binary for {result_id}")
            expected_mask = np.where(
                native_probability > MASK_THRESHOLD,
                np.uint8(255),
                np.uint8(0),
            )
            if not np.array_equal(mask, expected_mask):
                raise ValueError(
                    f"strict >0.5 native threshold mask mismatch for {result_id}"
                )
            replay_mask = np.where(
                expected_native > MASK_THRESHOLD,
                np.uint8(255),
                np.uint8(0),
            )
            if not np.array_equal(mask, replay_mask):
                raise ValueError(
                    "native threshold mask disagrees with independently "
                    f"replayed probability > 0.5 for {result_id}"
                )

            target = _load_target(
                result=result,
                input_row=input_row,
                repo_root=repo_root,
                width=width,
                height=height,
                checked_paths=checked_paths,
            )
            model_target = _nearest_resize_mask(
                target,
                width=MODEL_INPUT_SIZE,
                height=MODEL_INPUT_SIZE,
            )
            _validate_metrics(
                result,
                score_map=model_probability,
                target=model_target,
                space="model_512",
            )
            _validate_metrics(
                result,
                score_map=native_probability,
                target=target,
                space="native",
            )
            if result["kind"] == "forged":
                forged_target = target
                forged_native = native_probability
                forged_prediction = mask > 0

        if forged_target is None or forged_native is None or forged_prediction is None:
            raise ValueError(f"forged artifacts were not loaded for {pair.task_id}")
        if histogram_bins is not None:
            best, all_hist, positive_hist = _oracle_histograms(
                forged_native,
                forged_target,
                bins=histogram_bins,
            )
            per_image_best.append({"task_id": pair.task_id, **best})
            assert global_all is not None
            assert global_positive is not None
            global_all += all_hist
            global_positive += positive_hist

        x1, y1, x2, y2 = (int(value) for value in pair.input_row["edit_region_xyxy"])
        if not (
            0 <= x1 < x2 <= forged_prediction.shape[1]
            and 0 <= y1 < y2 <= forged_prediction.shape[0]
        ):
            raise ValueError(f"invalid edit box for {pair.task_id}")
        box_area = (x2 - x1) * (y2 - y1)
        intersection = int(np.count_nonzero(forged_prediction[y1:y2, x1:x2]))
        predicted_area = int(np.count_nonzero(forged_prediction))
        union = predicted_area + box_area - intersection
        iou = intersection / union if union else 0.0
        box_ious.append(iou)
        box_coverages.append(intersection / box_area)
        prediction_inside.append(
            intersection / predicted_area if predicted_area else 0.0
        )
        box_overlap += int(intersection > 0)
        box_hits += int(iou > 0.3)

    diagnostic: dict[str, Any] | None
    if histogram_bins is None:
        diagnostic = None
    else:
        assert global_all is not None
        assert global_positive is not None
        diagnostic = {
            "status": "posthoc_descriptive_oracle_only",
            "eligible_for_primary_metrics": False,
            "uses_test_set_labels": True,
            "fixed_primary_threshold_unchanged": MASK_THRESHOLD,
            "fixed_primary_threshold_operator": THRESHOLD_OPERATOR,
            "approximation": (
                f"native float32 probabilities quantized into {histogram_bins} "
                "uniform bins over [0,1], with strict greater-than candidates"
            ),
            "per_image_oracle": {
                "images": len(per_image_best),
                "f1": _descriptive(float(row["f1"]) for row in per_image_best),
                "iou": _descriptive(float(row["iou"]) for row in per_image_best),
            },
            "single_global_test_set_oracle": _best_from_histograms(
                global_all,
                global_positive,
            ),
        }
    return {
        "artifact_integrity": {
            "status": "ok",
            "checked_files": len(checked_paths),
            "pairs": len(pairs),
            "result_images": len(pairs) * 2,
            "numeric_tolerances": {
                "resized_logits_absolute": RESIZED_LOGITS_ABSOLUTE_TOLERANCE,
                "model_probability_absolute": (MODEL_PROBABILITY_ABSOLUTE_TOLERANCE),
                "native_restore_absolute": NATIVE_RESTORE_ABSOLUTE_TOLERANCE,
                "relative": TRANSFORM_RELATIVE_TOLERANCE,
                "reason": (
                    "bounded CPU/CUDA float32 bilinear operation-order and "
                    "sigmoid runtime differences"
                ),
            },
            "observed_maximum_absolute_error": {
                "resized_logits_replay": maximum_resized_logits_replay_error,
                "model_probability_replay": maximum_model_replay_error,
                "native_probability_replay": maximum_native_replay_error,
                "threshold_mask_disagreements": 0,
            },
            "checks": [
                "canonical images, GT masks, and all artifact hashes",
                (
                    "independent RGB decode, forced-square Pillow bilinear "
                    "resize, ImageNet normalization, and tensor hash"
                ),
                "captured segmentation-head logits are finite float32 [32,32]",
                (
                    "512x512 logits replay bilinear align_corners=False and "
                    "official model probability replays independent sigmoid"
                ),
                (
                    "native probability restores the 512x512 probability "
                    "directly, never native logits followed by sigmoid"
                ),
                "native PNG bit-exactly equals strict probability > 0.5",
                (
                    "model GT uses nearest resize to 512x512 and model/native "
                    "localization metrics are independently recomputed"
                ),
            ],
        },
        "localization_threshold_diagnostic": diagnostic,
        "box_hit_at_native_mask_threshold_0_5": {
            "status": "posthoc_descriptive_diagnostic_only",
            "eligible_for_primary_metrics": False,
            "uses_test_set_annotations": True,
            "task_scope": "T2_pixel_localization_only",
            "mask_threshold": MASK_THRESHOLD,
            "threshold_operator": THRESHOLD_OPERATOR,
            "box_definition": "canonical edit_region_xyxy half-open rectangle",
            "any_overlap": {
                "hits": box_overlap,
                "images": len(pairs),
                "rate": box_overlap / len(pairs),
            },
            "iou_greater_than_0_3": {
                "hits": box_hits,
                "images": len(pairs),
                "rate": box_hits / len(pairs),
            },
            "box_iou": _descriptive(box_ious),
            "box_pixel_coverage": _descriptive(box_coverages),
            "predicted_pixels_inside_box_fraction": _descriptive(prediction_inside),
        },
    }


def audit_prefix_reproducibility(
    *,
    full_manifest: dict[str, Any],
    full_rows: list[dict[str, Any]],
    prefix_manifest: dict[str, Any],
    prefix_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Prove that a smoke run is a byte/metric-identical prefix of full."""

    full_ordered = full_manifest.get("ordered_inputs")
    prefix_ordered = prefix_manifest.get("ordered_inputs")
    if not isinstance(full_ordered, list) or not isinstance(prefix_ordered, list):
        raise ValueError("reproducibility manifests have no ordered inputs")
    if not prefix_ordered:
        raise ValueError("reproducibility prefix is empty")
    if len(prefix_ordered) > len(full_ordered):
        raise ValueError("reproducibility prefix is longer than full run")
    if full_ordered[: len(prefix_ordered)] != prefix_ordered:
        raise ValueError("reference selection is not an ordered full-run prefix")
    _require_mapping(
        full_manifest.get("runtime_contract"),
        "full-run runtime contract",
    )
    _require_mapping(
        prefix_manifest.get("runtime_contract"),
        "prefix-run runtime contract",
    )
    for key in (
        "runtime_contract",
        "model",
        "inference",
        "metrics",
        "artifacts",
    ):
        full_value = full_manifest.get(key)
        prefix_value = prefix_manifest.get(key)
        if key == "metrics" and isinstance(full_value, dict):
            # Bootstrap sample count is not an inference result and may differ.
            full_value = {
                name: value
                for name, value in full_value.items()
                if name != "bootstrap_samples"
            }
            prefix_value = {
                name: value
                for name, value in _require_mapping(
                    prefix_value,
                    "prefix metrics",
                ).items()
                if name != "bootstrap_samples"
            }
        _require_equal(
            full_value,
            prefix_value,
            f"prefix reproducibility manifest {key}",
        )
    full_latest = _latest_by_id(full_rows)
    prefix_latest = _latest_by_id(prefix_rows)
    prefix_ids = [str(item["sample_id"]) for item in prefix_ordered]
    if set(prefix_latest) != set(prefix_ids):
        raise ValueError("reference latest IDs do not equal its manifest prefix")
    missing = [sample_id for sample_id in prefix_ids if sample_id not in full_latest]
    if missing:
        raise ValueError(f"full run is missing prefix IDs: {missing[:5]}")
    for sample_id in prefix_ids:
        full = full_latest[sample_id]
        prefix = prefix_latest[sample_id]
        if full.get("status") != "ok" or prefix.get("status") != "ok":
            raise ValueError(
                f"prefix reproducibility requires successful row {sample_id}"
            )
        for field in _REPRODUCIBILITY_FIELDS:
            _require_equal(
                full.get(field),
                prefix.get(field),
                f"prefix row {sample_id} field {field}",
            )
    return {
        "status": "ok",
        "policy": "latest physical row per sample id",
        "prefix_images": len(prefix_ids),
        "prefix_pairs": len(prefix_ids) // 2,
        "full_images": len(full_latest),
        "fields_compared": list(_REPRODUCIBILITY_FIELDS),
        "checks": [
            "reference ordered selection is an exact prefix",
            "model, inference, metric, and artifact contracts agree",
            "latest successful artifact hashes, preprocessing, and metrics agree",
        ],
    }


def _quintiles(
    pairs: list[LocalizationPair],
) -> list[tuple[str, list[LocalizationPair]]]:
    if not pairs:
        return []
    ordered = sorted(pairs, key=lambda pair: (pair.edit_fraction, pair.task_id))
    count = min(5, len(ordered))
    chunks = np.array_split(np.asarray(ordered, dtype=object), count)
    return [
        (
            (
                f"q{index}_"
                f"{'smallest' if index == 1 else ''}"
                f"{'largest' if index == count else ''}"
            ).rstrip("_"),
            list(chunk),
        )
        for index, chunk in enumerate(chunks, start=1)
    ]


def _summarize_spaces(
    pairs: list[LocalizationPair],
    *,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    return {
        space: summarize_dinov3_iml_pair_slice(
            pairs,
            iterations=iterations,
            seed=seed,
            localization_space=space,
        )
        for space in ("native", "model_512")
    }


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    if args.bootstrap_iterations <= 0:
        raise ValueError("bootstrap iterations must be positive")
    if not args.skip_threshold_diagnostic and args.histogram_bins < 2:
        raise ValueError("histogram bins must be at least two")
    if (args.prefix_run_id is None) != (args.prefix_results_dir is None):
        raise ValueError(
            "--prefix-run-id and --prefix-results-dir must be supplied together"
        )

    repo_root = args.repo_root.resolve()
    results_dir = _anchored(args.results_dir, repo_root)
    result_path = results_dir / f"{args.run_id}.jsonl"
    manifest_path = results_dir / f"{args.run_id}.run_manifest.json"
    summary_path = results_dir / f"{args.run_id}.summary.json"
    output_path = (
        _anchored(args.output, repo_root)
        if args.output is not None
        else results_dir / f"{args.run_id}.analysis.json"
    )
    input_path = _anchored(args.inputs, repo_root)
    dinov3_iml_root = args.dinov3_iml_root.resolve()
    dinov3_root = args.dinov3_root.resolve()
    for path in (result_path, manifest_path, summary_path, input_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    result_rows = read_jsonl(result_path)
    all_inputs = read_jsonl(input_path)
    manifest = _require_mapping(
        json.loads(manifest_path.read_text(encoding="utf-8")),
        "run manifest",
    )
    summary = _require_mapping(
        json.loads(summary_path.read_text(encoding="utf-8")),
        "run summary",
    )
    input_rows = _select_manifest_inputs(all_inputs, manifest)
    history = summarize_result_history(result_rows)
    provenance = validate_provenance(
        repo_root=repo_root,
        dinov3_iml_root=dinov3_iml_root,
        dinov3_root=dinov3_root,
        run_id=args.run_id,
        input_path=input_path,
        input_rows=input_rows,
        result_rows=result_rows,
        manifest=manifest,
        summary=summary,
    )
    pairs = _load_pairs(result_rows, input_rows)
    overall = _summarize_spaces(
        pairs,
        iterations=args.bootstrap_iterations,
        seed=args.bootstrap_seed,
    )
    domain_slices = {
        domain: _summarize_spaces(
            [pair for pair in pairs if pair.domain == domain],
            iterations=args.bootstrap_iterations,
            seed=args.bootstrap_seed + index,
        )
        for index, domain in enumerate(
            sorted({pair.domain for pair in pairs}),
            start=1,
        )
    }
    quintile_slices = {
        name: _summarize_spaces(
            chunk,
            iterations=args.bootstrap_iterations,
            seed=args.bootstrap_seed + 100 + index,
        )
        for index, (name, chunk) in enumerate(_quintiles(pairs), start=1)
    }
    artifact_audit = audit_artifacts(
        pairs,
        repo_root=repo_root,
        histogram_bins=(
            None if args.skip_threshold_diagnostic else args.histogram_bins
        ),
    )
    reproducibility: dict[str, Any] | None = None
    if args.prefix_run_id is not None:
        assert args.prefix_results_dir is not None
        prefix_dir = _anchored(args.prefix_results_dir, repo_root)
        prefix_result_path = prefix_dir / f"{args.prefix_run_id}.jsonl"
        prefix_manifest_path = prefix_dir / f"{args.prefix_run_id}.run_manifest.json"
        for path in (prefix_result_path, prefix_manifest_path):
            if not path.is_file():
                raise FileNotFoundError(path)
        prefix_manifest = _require_mapping(
            json.loads(prefix_manifest_path.read_text(encoding="utf-8")),
            "prefix run manifest",
        )
        reproducibility = {
            **audit_prefix_reproducibility(
                full_manifest=manifest,
                full_rows=result_rows,
                prefix_manifest=prefix_manifest,
                prefix_rows=read_jsonl(prefix_result_path),
            ),
            "prefix_run_id": args.prefix_run_id,
            "prefix_results_path": _relative_or_absolute(
                prefix_result_path,
                repo_root,
            ),
            "prefix_results_sha256": sha256_file(prefix_result_path),
            "prefix_manifest_path": _relative_or_absolute(
                prefix_manifest_path,
                repo_root,
            ),
            "prefix_manifest_sha256": sha256_file(prefix_manifest_path),
        }
    value = {
        "schema_version": "dinov3_iml_posthoc_analysis_v1",
        "run_id": args.run_id,
        "created_at": utc_now(),
        "task_scope": {
            "primary_task": "T2_localization",
            "valid_for_t1": False,
            "valid_for_t2": True,
            "primary_localization_space": "native",
            "auxiliary_localization_space": "model_512",
            "auxiliary_localization_extent": "full_forced_square_model_canvas",
            "mask_threshold": MASK_THRESHOLD,
            "threshold_operator": THRESHOLD_OPERATOR,
            "t1_policy": "unsupported_no_derived_image_score",
        },
        "sources": {
            "results_path": _relative_or_absolute(result_path, repo_root),
            "results_sha256": sha256_file(result_path),
            "run_manifest_path": _relative_or_absolute(
                manifest_path,
                repo_root,
            ),
            "run_manifest_sha256": sha256_file(manifest_path),
            "summary_path": _relative_or_absolute(summary_path, repo_root),
            "summary_sha256": sha256_file(summary_path),
            "inputs_path": _relative_or_absolute(input_path, repo_root),
            "inputs_sha256": sha256_file(input_path),
        },
        "bootstrap": {
            "unit": "paired task (real and forged resampled together)",
            "iterations": args.bootstrap_iterations,
            "seed": args.bootstrap_seed,
            "interval": "2.5th and 97.5th percentile",
            "spaces": ["native", "model_512"],
            "metrics_scope": "T2 localization only",
        },
        "overall": overall,
        "fixed_threshold_metrics": {
            "status": "primary_frozen_protocol_metrics",
            "mask_threshold": MASK_THRESHOLD,
            "threshold_operator": THRESHOLD_OPERATOR,
            "primary_localization_space": "native",
            "auxiliary_localization_space": "model_512",
            "auxiliary_localization_extent": "full_forced_square_model_canvas",
            "localization_forged": summary["localization_forged"],
            "localization_real": summary["localization_real"],
        },
        "by_domain": {
            "status": "posthoc_stratified_diagnostic_only",
            "eligible_for_primary_metrics": False,
            "uses_test_set_annotations": True,
            "slices": domain_slices,
        },
        "by_edit_fraction_quintile": {
            "status": "posthoc_stratified_diagnostic_only",
            "eligible_for_primary_metrics": False,
            "uses_test_set_annotations": True,
            "slices": quintile_slices,
        },
        "provenance_integrity": provenance,
        "result_history": history,
        "prefix_reproducibility": reproducibility,
        **artifact_audit,
    }
    atomic_write_json(output_path, value)
    print(json.dumps(value, ensure_ascii=False, indent=2))
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--inputs", type=Path, default=DEFAULT_INPUTS)
    parser.add_argument(
        "--dinov3-iml-root",
        type=Path,
        default=DEFAULT_DINOV3_IML_ROOT,
    )
    parser.add_argument(
        "--dinov3-root",
        type=Path,
        default=DEFAULT_DINOV3_ROOT,
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260724)
    parser.add_argument("--histogram-bins", type=int, default=HISTOGRAM_BINS)
    parser.add_argument("--skip-threshold-diagnostic", action="store_true")
    parser.add_argument("--prefix-run-id")
    parser.add_argument("--prefix-results-dir", type=Path)
    return parser.parse_args()


def main() -> None:
    analyze(parse_args())


if __name__ == "__main__":
    main()
