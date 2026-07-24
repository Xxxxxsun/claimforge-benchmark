#!/usr/bin/env python3
"""Independently audit and analyze a paired official PSCC-Net run."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

import numpy as np
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

from eval.opensource.common import (
    atomic_write_json,
    read_jsonl,
    sha256_file,
    stable_json,
    utc_now,
)


DEFAULT_RUN_ID = "psccnet_official_mouse_canonical_v1_full275_20260724"
DEFAULT_RESULTS_DIR = Path("results/opensource/psccnet")
DEFAULT_INPUTS = Path("outputs/opensource/mouse_canonical_v1/inputs.jsonl")

MODEL_NAME = "PSCC-Net"
MODEL_SLUG = "psccnet_tcsvt2022_official"
MODEL_REPO_URL = "https://github.com/proteus1991/PSCC-Net"
MODEL_CROP_SIZE = (256, 256)
PROGRESSIVE_SHAPES = (
    (256, 256),
    (128, 128),
    (64, 64),
    (32, 32),
)
CLASSIFICATION_THRESHOLD = 0.5
MASK_THRESHOLD = 0.5
THRESHOLD_COMPARISON = "strict_greater_than"
THRESHOLD_OPERATOR = ">"
T1_SCORE_SEMANTICS = "softmax_probability_class_1_forged"
T2_MAP_SEMANTICS = "psccnet_native_manipulation_probability"
HISTOGRAM_BINS = 65_536

BOOTSTRAP_METRICS = (
    "auroc",
    "average_precision",
    "tpr_at_fpr_5_percent",
    "accuracy_at_0_5",
    "balanced_accuracy_at_0_5",
    "image_f1_at_0_5",
    "paired_ranking_accuracy",
    "paired_score_delta_mean",
    "pixel_ap_macro",
    "pixel_f1_macro_at_0_5",
    "pixel_iou_macro_at_0_5",
    "pixel_f1_micro_at_0_5",
    "pixel_iou_micro_at_0_5",
    "real_false_positive_area_fraction_macro_at_0_5",
    "real_false_positive_area_fraction_micro_at_0_5",
)


@dataclass(frozen=True)
class Pair:
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
    """Load release pins lazily without importing the model implementation."""

    from eval.opensource import run_psccnet

    required = (
        "MODEL_REPO_URL",
        "MODEL_SOURCE_COMMIT",
        "SOURCE_FILES",
        "INITIALIZATION_WEIGHT",
        "CHECKPOINTS",
        "CHECKPOINT_BUNDLE_SHA256",
    )
    missing = [name for name in required if not hasattr(run_psccnet, name)]
    if missing:
        raise RuntimeError(
            "PSCC-Net runner does not export required audit constants: "
            f"{missing}"
        )
    return SimpleNamespace(
        **{name: getattr(run_psccnet, name) for name in required}
    )


def _anchored(path: Path, repo_root: Path) -> Path:
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


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
    expected_digest = _require_sha256(expected, f"{label} expected hash")
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")
    actual = sha256_file(path)
    if actual != expected_digest:
        raise ValueError(
            f"{label} SHA-256 mismatch: {actual} != {expected_digest}"
        )


def _array_sha256(array: np.ndarray, dtype: np.dtype[Any]) -> str:
    canonical = np.ascontiguousarray(array, dtype=dtype)
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


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
    inputs_by_id: dict[str, dict[str, Any]] = {}
    for line_number, row in enumerate(input_rows, start=1):
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError(
                f"canonical input row {line_number} has no sample_id"
            )
        if sample_id in inputs_by_id:
            raise ValueError(f"canonical inputs contain duplicate ID {sample_id}")
        inputs_by_id[sample_id] = row

    ordered = manifest.get("ordered_inputs")
    if not isinstance(ordered, list) or not ordered:
        raise ValueError("run manifest ordered_inputs is empty or invalid")
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, value in enumerate(ordered):
        item = _require_mapping(value, f"ordered input {index}")
        sample_id = item.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError(f"ordered input {index} has no sample_id")
        if sample_id in seen:
            raise ValueError(
                f"run manifest ordered_inputs contains duplicate ID {sample_id}"
            )
        if sample_id not in inputs_by_id:
            raise ValueError(
                f"run manifest selected unknown canonical ID {sample_id}"
            )
        seen.add(sample_id)
        selected.append(inputs_by_id[sample_id])
    return selected


def summarize_result_history(
    result_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    histories: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    status_counts: Counter[str] = Counter()
    for line_number, row in enumerate(result_rows, start=1):
        row_id = row.get("id")
        if not isinstance(row_id, str) or not row_id:
            raise ValueError(
                f"result row {line_number} has no non-empty string id"
            )
        histories[row_id].append((line_number, row))
        status_counts[str(row.get("status"))] += 1

    duplicate_histories: list[dict[str, Any]] = []
    recovered_ids: list[str] = []
    latest_status_counts: Counter[str] = Counter()
    for row_id, entries in sorted(histories.items()):
        statuses = [str(row.get("status")) for _, row in entries]
        latest_status_counts[statuses[-1]] += 1
        if len(entries) > 1:
            duplicate_histories.append(
                {
                    "id": row_id,
                    "physical_rows": len(entries),
                    "line_numbers": [line_number for line_number, _ in entries],
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
        "latest_status_counts": dict(sorted(latest_status_counts.items())),
        "duplicate_histories": duplicate_histories,
        "latest_policy": "last physical JSONL row for each sample id",
    }


def _verify_adapter_contract(value: Any, *, repo_root: Path) -> int:
    if not isinstance(value, list) or not value:
        raise ValueError("manifest adapter_contract is empty or invalid")
    checked = 0
    for index, raw in enumerate(value):
        item = _require_mapping(raw, f"adapter contract entry {index}")
        path_value = item.get("path")
        if not isinstance(path_value, str):
            raise ValueError(f"adapter contract entry {index} has no path")
        _verify_hash(
            _anchored(Path(path_value), repo_root),
            item.get("sha256"),
            f"adapter contract entry {index}",
        )
        checked += 1
    return checked


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
            raise ValueError(f"{label} contains duplicate path {path}")
        result[path] = _require_sha256(
            item.get("sha256"),
            f"{label} entry {index} SHA-256",
        )
    return result


def _checkpoint_components(value: Any, label: str) -> dict[str, dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{label} is empty or invalid")
    result: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(value):
        item = _require_mapping(raw, f"{label} entry {index}")
        role = item.get("role")
        if not isinstance(role, str) or not role:
            raise ValueError(f"{label} entry {index} has no role")
        if role in result:
            raise ValueError(f"{label} contains duplicate role {role}")
        result[role] = {key: child for key, child in item.items() if key != "role"}
    return result


def validate_provenance(
    *,
    repo_root: Path,
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
    computed_fingerprint = hashlib.sha256(
        stable_json(immutable).encode("utf-8")
    ).hexdigest()
    _require_equal(
        fingerprint,
        computed_fingerprint,
        "run manifest fingerprint",
    )

    manifest_input = _require_mapping(manifest.get("input"), "manifest input")
    actual_inputs_sha256 = sha256_file(input_path)
    _require_equal(
        manifest_input.get("inputs_sha256"),
        actual_inputs_sha256,
        "manifest/input JSONL SHA-256",
    )
    inputs_path_value = manifest_input.get("inputs_manifest")
    if not isinstance(inputs_path_value, str):
        raise ValueError("manifest has no inputs_manifest path")
    _require_equal(
        _anchored(Path(inputs_path_value), repo_root),
        input_path.resolve(),
        "manifest/input JSONL path",
    )

    dataset_manifest_value = manifest_input.get("dataset_manifest")
    if not isinstance(dataset_manifest_value, str):
        raise ValueError("manifest has no dataset_manifest path")
    dataset_manifest_path = _anchored(
        Path(dataset_manifest_value),
        repo_root,
    )
    _verify_hash(
        dataset_manifest_path,
        manifest_input.get("dataset_manifest_sha256"),
        "canonical dataset manifest",
    )
    release = _require_mapping(
        json.loads(dataset_manifest_path.read_text(encoding="utf-8")),
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
    selection_sha256 = hashlib.sha256(
        stable_json(expected_selection).encode("utf-8")
    ).hexdigest()
    _require_equal(
        manifest_input.get("selection_sha256"),
        selection_sha256,
        "manifest input selection SHA-256",
    )
    pair_ranks = {int(row["pair_rank"]) for row in input_rows}
    _require_equal(
        manifest.get("expected_images"),
        len(input_rows),
        "manifest expected image count",
    )
    _require_equal(
        manifest.get("expected_pairs"),
        len(pair_ranks),
        "manifest expected pair count",
    )

    expected_source_files = {
        str(path): str(digest)
        for path, digest in dict(pins.SOURCE_FILES).items()
    }
    expected_initialization = dict(pins.INITIALIZATION_WEIGHT)
    expected_checkpoints = {
        str(role): dict(contract)
        for role, contract in dict(pins.CHECKPOINTS).items()
    }
    model = _require_mapping(manifest.get("model"), "manifest model")
    license_value = _require_mapping(model.get("license"), "manifest license")
    initialization = _require_mapping(
        model.get("initialization_weight"),
        "manifest initialization weight",
    )
    checkpoint = _require_mapping(
        model.get("checkpoint"),
        "manifest checkpoint",
    )
    for actual, expected, label in (
        (model.get("name"), MODEL_NAME, "manifest model name"),
        (model.get("model_slug"), MODEL_SLUG, "manifest model slug"),
        (model.get("repo_url"), pins.MODEL_REPO_URL, "manifest repository URL"),
        (
            model.get("source_commit"),
            pins.MODEL_SOURCE_COMMIT,
            "manifest source commit",
        ),
        (
            model.get("source_tracked_clean"),
            True,
            "manifest source clean flag",
        ),
        (
            model.get("variant"),
            "official_committed_pretrained_checkpoint",
            "manifest model variant",
        ),
        (license_value.get("path"), "LICENSE", "manifest license path"),
        (
            license_value.get("sha256"),
            expected_source_files["LICENSE"],
            "manifest license SHA-256",
        ),
        (license_value.get("spdx"), "MIT", "manifest license SPDX"),
        (
            license_value.get("scope"),
            "project_repository",
            "manifest license scope",
        ),
        (
            checkpoint.get("provider"),
            "official_author_git_repository",
            "manifest checkpoint provider",
        ),
        (
            checkpoint.get("source_commit"),
            pins.MODEL_SOURCE_COMMIT,
            "manifest checkpoint source commit",
        ),
        (
            checkpoint.get("bundle_sha256"),
            pins.CHECKPOINT_BUNDLE_SHA256,
            "manifest checkpoint bundle SHA-256",
        ),
        (
            checkpoint.get("strict_load"),
            True,
            "manifest checkpoint strict-load flag",
        ),
        (
            checkpoint.get("safe_weights_only_load"),
            True,
            "manifest checkpoint safe-load flag",
        ),
        (
            model.get("parameter_count"),
            sum(int(value["parameters"]) for value in expected_checkpoints.values()),
            "manifest parameter count",
        ),
        (
            model.get("buffer_elements"),
            sum(int(value["buffers"]) for value in expected_checkpoints.values()),
            "manifest buffer count",
        ),
        (
            model.get("class_names"),
            ["authentic", "forged"],
            "manifest class names",
        ),
        (
            model.get("positive_class_index"),
            1,
            "manifest positive class index",
        ),
        (
            model.get("supports_image_level_t1"),
            True,
            "manifest T1 support flag",
        ),
        (
            model.get("image_score_source"),
            "native_independent_classification_head",
            "manifest T1 score source",
        ),
        (
            model.get("supports_pixel_level_t2"),
            True,
            "manifest T2 support flag",
        ),
        (
            model.get("primary_localization_output"),
            "progressive_mask1",
            "manifest primary localization output",
        ),
    ):
        _require_equal(actual, expected, label)
    _require_equal(
        _normalise_source_files(model.get("source_files"), "source files"),
        expected_source_files,
        "manifest source-file pins",
    )
    _require_equal(
        initialization,
        expected_initialization,
        "manifest initialization-weight contract",
    )
    _require_equal(
        _checkpoint_components(
            checkpoint.get("components"),
            "checkpoint components",
        ),
        expected_checkpoints,
        "manifest checkpoint components",
    )

    inference = _require_mapping(manifest.get("inference"), "manifest inference")
    expected_inference = {
        "precision": "float32",
        "batch_size": 1,
        "deterministic": True,
        "compatibility_shim": "numpy.int=builtin_int_for_hrnet_constructor",
        "input_source": "canonical_jpeg_original_bytes",
        "decoder": "imageio.v2.imread",
        "channel_order": "RGB",
        "input_resize": "none",
        "input_crop": None,
        "input_reencode": False,
        "normalization": "uint8_rgb_divide_255",
        "feature_extractor": "HRNet-W18-small-v2",
        "internal_crop_size": [256, 256],
        "progressive_output_shapes": [
            [256, 256],
            [128, 128],
            [64, 64],
            [32, 32],
        ],
        "primary_map": "progressive_mask1_sigmoid_probability",
        "primary_map_selection": "fixed_by_official_test_py_index_0",
        "native_restore": (
            "bilinear_probability_align_corners_true_to_input_size"
        ),
        "classification_output": (
            "softmax_two_class_logits_positive_index_1"
        ),
        "classification_threshold": CLASSIFICATION_THRESHOLD,
        "classification_threshold_comparison": THRESHOLD_COMPARISON,
        "mask_threshold": MASK_THRESHOLD,
        "mask_threshold_comparison": THRESHOLD_COMPARISON,
        "test_time_augmentation": False,
        "ensemble": False,
    }
    for key, expected in expected_inference.items():
        _require_equal(
            inference.get(key),
            expected,
            f"manifest inference {key}",
        )
    metrics = _require_mapping(manifest.get("metrics"), "manifest metrics")
    expected_metrics = {
        "task": "T1_image_detection_and_T2_pixel_localization",
        "positive_class": "forged_or_manipulated",
        "classification_threshold": CLASSIFICATION_THRESHOLD,
        "mask_threshold": MASK_THRESHOLD,
        "threshold_comparison": THRESHOLD_COMPARISON,
        "prediction_inversion": False,
        "model_space_gt_resize": "nearest_neighbor_to_256x256",
    }
    for key, expected in expected_metrics.items():
        _require_equal(metrics.get(key), expected, f"manifest metrics {key}")

    expected_by_id = {str(row["sample_id"]): row for row in input_rows}
    seen_ids: set[str] = set()
    latest: dict[str, dict[str, Any]] = {}
    for line_number, row in enumerate(result_rows, start=1):
        row_id = row.get("id")
        if not isinstance(row_id, str) or not row_id:
            raise ValueError(
                f"result row {line_number} has no non-empty string id"
            )
        if row_id not in expected_by_id:
            raise ValueError(f"unexpected result ID at row {line_number}: {row_id}")
        input_row = expected_by_id[row_id]
        seen_ids.add(row_id)
        latest[row_id] = row
        expected_values = {
            "schema_version": "opensource_result_v1",
            "run_id": run_id,
            "run_manifest_fingerprint": fingerprint,
            "input_manifest_sha256": actual_inputs_sha256,
            "id": row_id,
            "rank": int(input_row["rank"]),
            "task_id": str(input_row["task_id"]),
            "pair_rank": int(input_row["pair_rank"]),
            "domain": str(input_row["domain"]),
            "kind": str(input_row["kind"]),
            "label": int(input_row["label"]),
            "image_path": str(input_row["canonical_path"]),
            "image_sha256": str(input_row["canonical_sha256"]),
            "image_size": [
                int(input_row["width"]),
                int(input_row["height"]),
            ],
            "gt_mask_kind": str(input_row["gt_mask_kind"]),
            "gt_mask_sha256": input_row.get("gt_mask_sha256"),
            "edit_region_xyxy": [
                int(value) for value in input_row["edit_region_xyxy"]
            ],
            "model": MODEL_NAME,
            "model_slug": MODEL_SLUG,
            "checkpoint_sha256": pins.CHECKPOINT_BUNDLE_SHA256,
            "valid_for_t1": True,
            "valid_for_t2": True,
        }
        for key, expected in expected_values.items():
            _require_equal(
                row.get(key),
                expected,
                f"result row {line_number} field {key}",
            )
        status = row.get("status")
        if status not in {"ok", "error"}:
            raise ValueError(
                f"result row {line_number} has invalid status {status!r}"
            )
        _require_equal(
            row.get("valid_for_metrics"),
            status == "ok",
            f"result row {line_number} valid_for_metrics",
        )
        if status == "ok":
            for actual, expected, label in (
                (
                    row.get("score_source"),
                    "native_classification_head",
                    "score source",
                ),
                (
                    row.get("score_semantics"),
                    T1_SCORE_SEMANTICS,
                    "score semantics",
                ),
                (
                    row.get("classification_threshold"),
                    CLASSIFICATION_THRESHOLD,
                    "classification threshold",
                ),
                (
                    row.get("classification_threshold_operator"),
                    THRESHOLD_OPERATOR,
                    "classification threshold operator",
                ),
                (
                    row.get("mask_threshold"),
                    MASK_THRESHOLD,
                    "mask threshold",
                ),
                (
                    row.get("mask_threshold_operator"),
                    THRESHOLD_OPERATOR,
                    "mask threshold operator",
                ),
            ):
                _require_equal(
                    actual,
                    expected,
                    f"result row {line_number} {label}",
                )
    if seen_ids != set(expected_ids):
        missing = sorted(set(expected_ids) - seen_ids)
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
            pins.CHECKPOINT_BUNDLE_SHA256,
            "summary checkpoint SHA-256",
        ),
        (
            summary.get("input_manifest_sha256"),
            actual_inputs_sha256,
            "summary input manifest SHA-256",
        ),
        (
            summary.get("run_manifest_fingerprint"),
            fingerprint,
            "summary run manifest fingerprint",
        ),
    ):
        _require_equal(actual, expected, label)
    coverage = _require_mapping(summary.get("coverage"), "summary coverage")
    valid_latest = sum(row.get("status") == "ok" for row in latest.values())
    expected_coverage = {
        "expected_images": len(input_rows),
        "result_images": len(latest),
        "valid_images": valid_latest,
        "error_images": len(latest) - valid_latest,
        "missing_images": len(input_rows) - len(latest),
    }
    for key, expected in expected_coverage.items():
        _require_equal(coverage.get(key), expected, f"summary coverage {key}")
    task_scope = _require_mapping(summary.get("task_scope"), "summary task scope")
    expected_scope = {
        "primary_task": "T1_detection_and_T2_localization",
        "primary_detection_score": "score",
        "primary_detection_semantics": (
            "psccnet_image_level_manipulation_probability"
        ),
        "primary_localization_space": "native",
        "primary_localization_semantics": T2_MAP_SEMANTICS,
        "classification_threshold": CLASSIFICATION_THRESHOLD,
        "mask_threshold": MASK_THRESHOLD,
        "threshold_operator": THRESHOLD_OPERATOR,
    }
    for key, expected in expected_scope.items():
        _require_equal(task_scope.get(key), expected, f"summary task scope {key}")

    adapter_files_checked = _verify_adapter_contract(
        manifest.get("adapter_contract"),
        repo_root=repo_root,
    )
    return {
        "status": "ok",
        "run_manifest_fingerprint": fingerprint,
        "inputs_sha256": actual_inputs_sha256,
        "checkpoint_sha256": pins.CHECKPOINT_BUNDLE_SHA256,
        "physical_result_rows_validated": len(result_rows),
        "latest_result_rows_validated": len(latest),
        "adapter_contract_files_validated": adapter_files_checked,
        "checks": [
            "run manifest schema, ID, and recomputed immutable fingerprint",
            "canonical release hash, input hash, and ordered selection",
            "official source, initialization, and checkpoint bundle pins",
            "fixed positive class, fixed 0.5 thresholds, and no inversion",
            "every physical result row against canonical identity",
            "summary identity, task scope, fingerprint, and latest-row coverage",
            "adapter contract file hashes",
        ],
    }


def _load_pairs(
    result_rows: list[dict[str, Any]],
    input_rows: list[dict[str, Any]],
) -> list[Pair]:
    latest = {
        str(row["id"]): row
        for row in result_rows
        if isinstance(row.get("id"), str)
    }
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

    pairs: list[Pair] = []
    for task_id, values in by_task.items():
        if set(values) != {"real", "forged"}:
            raise ValueError(f"incomplete pair for {task_id}: {sorted(values)}")
        real = values["real"]
        forged = values["forged"]
        if real.get("label") != 0 or forged.get("label") != 1:
            raise ValueError(f"invalid real/forged labels within {task_id}")
        if real.get("domain") != forged.get("domain"):
            raise ValueError(f"domain mismatch within {task_id}")
        if int(real["pair_rank"]) != int(forged["pair_rank"]):
            raise ValueError(f"pair-rank mismatch within {task_id}")
        pairs.append(
            Pair(
                task_id=task_id,
                domain=str(forged["domain"]),
                real=real,
                forged=forged,
                input_row=inputs_by_id[str(forged["id"])],
            )
        )
    return sorted(pairs, key=lambda pair: int(pair.forged["pair_rank"]))


def _softmax_two(logits: Any, *, label: str) -> np.ndarray:
    if not isinstance(logits, list) or len(logits) != 2:
        raise ValueError(f"{label} must contain exactly two logits")
    try:
        values = np.asarray(logits, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} is not numeric") from exc
    if not np.isfinite(values).all():
        raise ValueError(f"{label} contains non-finite values")
    shifted = values - float(np.max(values))
    exponentials = np.exp(shifted)
    return exponentials / float(np.sum(exponentials))


def _recomputed_t1_score(
    result: dict[str, Any],
    *,
    validate_recorded: bool,
) -> float:
    probabilities = _softmax_two(
        result.get("classification_logits"),
        label=f"classification logits for {result.get('id')}",
    )
    recorded_probabilities = result.get("classification_probabilities")
    if validate_recorded:
        if (
            not isinstance(recorded_probabilities, list)
            or len(recorded_probabilities) != 2
        ):
            raise ValueError(
                f"classification probabilities for {result.get('id')} "
                "must contain exactly two values"
            )
        try:
            recorded_array = np.asarray(
                recorded_probabilities,
                dtype=np.float64,
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"classification probabilities for {result.get('id')} "
                "are not numeric"
            ) from exc
        if (
            not np.isfinite(recorded_array).all()
            or not np.allclose(
                recorded_array,
                probabilities,
                rtol=1e-6,
                atol=1e-7,
            )
        ):
            raise ValueError(
                "classification probabilities are not softmax(logits) for "
                f"{result.get('id')}"
            )
    score = float(probabilities[1])
    if validate_recorded:
        recorded_score = result.get("score")
        if not isinstance(recorded_score, (int, float)) or not math.isclose(
            float(recorded_score),
            score,
            rel_tol=1e-6,
            abs_tol=1e-7,
        ):
            raise ValueError(
                f"T1 score mismatch for {result.get('id')}: "
                f"{recorded_score!r} != {score!r}"
            )
        expected_decision = (
            "forged" if score > CLASSIFICATION_THRESHOLD else "authentic"
        )
        _require_equal(
            result.get("decision"),
            expected_decision,
            f"strict T1 decision for {result.get('id')}",
        )
    return score


def _safe_div(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _binary_pixel_metrics_strict(
    score_map: np.ndarray,
    target: np.ndarray,
    *,
    include_ap: bool,
) -> dict[str, Any]:
    scores = np.asarray(score_map, dtype=np.float32)
    truth = np.asarray(target, dtype=bool)
    if scores.ndim != 2 or scores.shape != truth.shape or not scores.size:
        raise ValueError("score/target shape mismatch or empty map")
    if not np.isfinite(scores).all():
        raise ValueError("score map contains non-finite values")
    if float(scores.min()) < 0.0 or float(scores.max()) > 1.0:
        raise ValueError("score map falls outside [0, 1]")
    prediction = scores > MASK_THRESHOLD
    tp = int(np.count_nonzero(prediction & truth))
    fp = int(np.count_nonzero(prediction & ~truth))
    fn = int(np.count_nonzero(~prediction & truth))
    tn = int(np.count_nonzero(~prediction & ~truth))
    denominator = math.sqrt(
        float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    )
    pixel_ap: float | None = None
    if include_ap and truth.any() and (~truth).any():
        pixel_ap = float(
            average_precision_score(truth.reshape(-1), scores.reshape(-1))
        )
    return {
        "threshold": MASK_THRESHOLD,
        "threshold_operator": THRESHOLD_OPERATOR,
        "pixels": int(scores.size),
        "target_positive_pixels": int(np.count_nonzero(truth)),
        "predicted_positive_pixels": int(np.count_nonzero(prediction)),
        "predicted_positive_fraction": float(np.mean(prediction)),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": _safe_div(tp, tp + fp),
        "recall": _safe_div(tp, tp + fn),
        "f1": _safe_div(2 * tp, 2 * tp + fp + fn),
        "iou": _safe_div(tp, tp + fp + fn),
        "mcc": (tp * tn - fp * fn) / denominator if denominator else None,
        "pixel_ap": pixel_ap,
        "score_mean": float(np.mean(scores)),
        "score_max": float(np.max(scores)),
    }


def _compare_metric(recorded: Any, expected: Any, label: str) -> None:
    if expected is None or isinstance(expected, (str, int)):
        _require_equal(recorded, expected, label)
        return
    if not isinstance(recorded, (int, float)) or not math.isclose(
        float(recorded),
        float(expected),
        rel_tol=1e-6,
        abs_tol=1e-7,
    ):
        raise ValueError(f"{label} mismatch: {recorded!r} != {expected!r}")


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
    expected = _binary_pixel_metrics_strict(
        score_map,
        target,
        include_ap=result.get("kind") == "forged",
    )
    for key, value in expected.items():
        _compare_metric(
            recorded.get(key),
            value,
            f"{space} localization {key} for {result['id']}",
        )


def _preprocess_evidence(image_path: Path) -> tuple[dict[str, Any], np.ndarray]:
    # The runner records ImageIO's Pillow-backed decode.  Reopening with
    # Pillow here gives an independent call path while reproducing the exact
    # RGB bytes used by ImageIO for the canonical JPEG inputs.
    with Image.open(image_path) as opened:
        decoded = np.asarray(opened)
    if decoded.ndim == 2:
        decoded = np.repeat(decoded[..., None], 3, axis=2)
        alpha_policy = "grayscale_repeated_to_rgb"
    elif decoded.ndim == 3 and decoded.shape[2] == 4:
        rgba = decoded.astype(np.float32)
        alpha = rgba[..., 3:4] / np.float32(255.0)
        decoded = (
            rgba[..., :3] * alpha
            + np.float32(255.0) * (np.float32(1.0) - alpha)
        ).astype(np.uint8)
        alpha_policy = "official_white_background_rgba_composite"
    elif decoded.ndim == 3 and decoded.shape[2] == 3:
        alpha_policy = "not_applicable"
    else:
        raise ValueError(f"unexpected decoded image shape: {decoded.shape}")
    if decoded.dtype != np.uint8:
        raise ValueError(f"unexpected decoded image dtype: {decoded.dtype}")
    height, width = decoded.shape[:2]
    tensor = np.ascontiguousarray(
        decoded.astype(np.float32).transpose(2, 0, 1) / np.float32(255.0),
        dtype=np.float32,
    )
    return (
        {
            "decoder": "imageio.v2.imread",
            "channel_order": "RGB",
            "native_size": [width, height],
            "input_resize": "none",
            "input_crop": None,
            "input_reencode": False,
            "normalization": "uint8_rgb_divide_255",
            "alpha_policy": alpha_policy,
            "tensor_shape": list(tensor.shape),
            "tensor_sha256": _array_sha256(tensor, np.dtype(np.float32)),
        },
        tensor,
    )


def _resize_target_model_256(target: np.ndarray) -> np.ndarray:
    image = Image.fromarray(
        np.where(np.asarray(target, dtype=bool), 255, 0).astype(np.uint8),
        mode="L",
    )
    resized = image.resize(MODEL_CROP_SIZE, resample=Image.Resampling.NEAREST)
    return np.asarray(resized, dtype=np.uint8) > 0


def _bilinear_align_corners_true(
    score_map: np.ndarray,
    *,
    width: int,
    height: int,
) -> np.ndarray:
    source = np.asarray(score_map, dtype=np.float32)
    if source.ndim != 2:
        raise ValueError("model score map must be two-dimensional")
    if width <= 0 or height <= 0:
        raise ValueError("native dimensions must be positive")
    source_height, source_width = source.shape
    if (height, width) == source.shape:
        return np.ascontiguousarray(source)

    y = (
        np.zeros(height, dtype=np.float32)
        if height == 1
        else np.arange(height, dtype=np.float32)
        * (np.float32(source_height - 1) / np.float32(height - 1))
    )
    x = (
        np.zeros(width, dtype=np.float32)
        if width == 1
        else np.arange(width, dtype=np.float32)
        * (np.float32(source_width - 1) / np.float32(width - 1))
    )
    y0 = np.floor(y).astype(np.int64)
    x0 = np.floor(x).astype(np.int64)
    y1 = np.minimum(y0 + 1, source_height - 1)
    x1 = np.minimum(x0 + 1, source_width - 1)
    wy = (y - y0).astype(np.float32).reshape(-1, 1)
    wx = (x - x0).astype(np.float32).reshape(1, -1)
    top = (
        source[y0[:, None], x0[None, :]] * (np.float32(1.0) - wx)
        + source[y0[:, None], x1[None, :]] * wx
    )
    bottom = (
        source[y1[:, None], x0[None, :]] * (np.float32(1.0) - wx)
        + source[y1[:, None], x1[None, :]] * wx
    )
    restored = top * (np.float32(1.0) - wy) + bottom * wy
    return np.ascontiguousarray(restored, dtype=np.float32)


def _load_float32_map(
    path: Path,
    *,
    expected_shape: tuple[int, int],
    label: str,
) -> np.ndarray:
    value = np.load(path, mmap_mode="r", allow_pickle=False)
    if value.shape != expected_shape:
        raise ValueError(
            f"invalid {label} shape: {value.shape} != {expected_shape}"
        )
    if value.dtype != np.float32:
        raise ValueError(f"invalid {label} dtype: {value.dtype} != float32")
    if not np.isfinite(value).all():
        raise ValueError(f"non-finite {label}")
    if float(value.min()) < 0.0 or float(value.max()) > 1.0:
        raise ValueError(f"out-of-range {label}")
    return np.asarray(value)


def _oracle_histograms(
    score_map: np.ndarray,
    target: np.ndarray,
    *,
    bins: int,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    if isinstance(bins, bool) or not isinstance(bins, int) or bins < 2:
        raise ValueError("oracle histogram bins must be an integer >= 2")
    scores = np.asarray(score_map, dtype=np.float32)
    truth = np.asarray(target, dtype=bool)
    if scores.shape != truth.shape or not truth.any():
        raise ValueError("oracle localization requires a non-empty aligned target")
    indices = np.minimum(
        np.floor(scores * np.float32(bins - 1)).astype(np.int64),
        bins - 1,
    )
    all_hist = np.bincount(indices.reshape(-1), minlength=bins).astype(np.int64)
    positive_hist = np.bincount(
        indices[truth],
        minlength=bins,
    ).astype(np.int64)
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
    f1_denominator = 2 * tp + fp + fn
    iou_denominator = tp + fp + fn
    f1 = np.divide(
        2.0 * tp,
        f1_denominator,
        out=np.zeros_like(tp, dtype=np.float64),
        where=f1_denominator > 0,
    )
    iou = np.divide(
        tp,
        iou_denominator,
        out=np.zeros_like(tp, dtype=np.float64),
        where=iou_denominator > 0,
    )
    best = int(np.argmax(f1))
    return (
        {
            "histogram_bins": bins,
            "threshold": best / (bins - 1),
            "comparison": THRESHOLD_OPERATOR,
            "f1": float(f1[best]),
            "iou": float(iou[best]),
            "tp": int(tp[best]),
            "fp": int(fp[best]),
            "fn": int(fn[best]),
        },
        all_hist,
        positive_hist,
    )


def _best_from_histograms(
    all_hist: np.ndarray,
    positive_hist: np.ndarray,
) -> dict[str, Any]:
    if all_hist.shape != positive_hist.shape or all_hist.ndim != 1:
        raise ValueError("oracle histograms have incompatible shapes")
    bins = int(all_hist.size)
    greater_all = np.zeros(bins, dtype=np.int64)
    greater_positive = np.zeros(bins, dtype=np.int64)
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
    best = int(np.argmax(f1))
    iou_denominator = tp[best] + fp[best] + fn[best]
    return {
        "histogram_bins": bins,
        "threshold": best / (bins - 1),
        "comparison": THRESHOLD_OPERATOR,
        "micro_f1": float(f1[best]),
        "micro_iou": (
            float(tp[best]) / float(iou_denominator)
            if iou_denominator
            else 0.0
        ),
        "tp": int(tp[best]),
        "fp": int(fp[best]),
        "fn": int(fn[best]),
    }


def audit_and_best_threshold(
    pairs: list[Pair],
    *,
    repo_root: Path,
    bins: int = HISTOGRAM_BINS,
) -> dict[str, Any]:
    if not pairs:
        raise ValueError("artifact audit requires at least one pair")
    global_all = np.zeros(bins, dtype=np.int64)
    global_positive = np.zeros(bins, dtype=np.int64)
    per_image_best: list[dict[str, Any]] = []
    box_ious: list[float] = []
    box_hits = 0
    checked_paths: set[Path] = set()

    for pair in pairs:
        forged_target: np.ndarray | None = None
        forged_native: np.ndarray | None = None
        forged_prediction: np.ndarray | None = None
        for result in (pair.real, pair.forged):
            result_id = str(result["id"])
            image_path = _anchored(Path(str(result["image_path"])), repo_root)
            _verify_hash(
                image_path,
                result.get("image_sha256"),
                f"canonical image {result_id}",
            )
            checked_paths.add(image_path)
            width, height = (int(value) for value in result["image_size"])
            evidence, tensor = _preprocess_evidence(image_path)
            if tensor.shape != (3, height, width):
                raise ValueError(f"invalid input tensor shape for {result_id}")
            preprocess = _require_mapping(
                result.get("preprocess"),
                f"preprocess metadata for {result_id}",
            )
            for key, expected in evidence.items():
                _require_equal(
                    preprocess.get(key),
                    expected,
                    f"preprocess {key} for {result_id}",
                )
            _require_equal(
                evidence["native_size"],
                [width, height],
                f"decoded native size for {result_id}",
            )
            _recomputed_t1_score(result, validate_recorded=True)

            progressive = result.get("progressive_maps")
            if not isinstance(progressive, list) or len(progressive) != 4:
                raise ValueError(
                    f"progressive map contract for {result_id} must have 4 stages"
                )
            model_maps: list[np.ndarray] = []
            stage_contracts: list[dict[str, Any]] = []
            for index, (raw, expected_shape) in enumerate(
                zip(progressive, PROGRESSIVE_SHAPES, strict=True),
                start=1,
            ):
                stage = _require_mapping(
                    raw,
                    f"progressive stage {index} for {result_id}",
                )
                _require_equal(
                    stage.get("stage"),
                    index,
                    f"progressive stage number for {result_id}",
                )
                _require_equal(
                    stage.get("shape"),
                    list(expected_shape),
                    f"progressive stage {index} shape metadata for {result_id}",
                )
                _require_equal(
                    stage.get("primary"),
                    index == 1,
                    f"progressive stage {index} primary flag for {result_id}",
                )
                path_value = stage.get("path")
                if not isinstance(path_value, str):
                    raise ValueError(
                        f"progressive stage {index} for {result_id} has no path"
                    )
                stage_path = _anchored(Path(path_value), repo_root)
                _verify_hash(
                    stage_path,
                    stage.get("sha256"),
                    f"progressive stage {index} {result_id}",
                )
                checked_paths.add(stage_path)
                model_maps.append(
                    _load_float32_map(
                        stage_path,
                        expected_shape=expected_shape,
                        label=f"progressive stage {index} for {result_id}",
                    )
                )
                stage_contracts.append(stage)

            primary = stage_contracts[0]
            for result_key, stage_key, label in (
                (
                    "primary_model_score_map_path",
                    "path",
                    "primary model-map path",
                ),
                (
                    "primary_model_score_map_sha256",
                    "sha256",
                    "primary model-map SHA-256",
                ),
                (
                    "primary_model_score_map_shape",
                    "shape",
                    "primary model-map shape",
                ),
            ):
                _require_equal(
                    result.get(result_key),
                    primary.get(stage_key),
                    f"{label} for {result_id}",
                )

            native_path_value = result.get("score_map_path")
            if not isinstance(native_path_value, str):
                raise ValueError(f"native map for {result_id} has no path")
            native_path = _anchored(Path(native_path_value), repo_root)
            _verify_hash(
                native_path,
                result.get("score_map_sha256"),
                f"native float32 score map {result_id}",
            )
            checked_paths.add(native_path)
            _require_equal(
                result.get("score_map_shape"),
                [height, width],
                f"native score-map shape metadata for {result_id}",
            )
            native_map = _load_float32_map(
                native_path,
                expected_shape=(height, width),
                label=f"native score map for {result_id}",
            )
            expected_native = _bilinear_align_corners_true(
                model_maps[0],
                width=width,
                height=height,
            )
            if not np.allclose(
                native_map,
                expected_native,
                # NumPy's independently implemented coordinate arithmetic can
                # differ from PyTorch's float32 kernel by a few ULPs.
                rtol=1e-5,
                atol=2e-6,
            ):
                raise ValueError(
                    "native map is not bilinear align_corners=True restoration "
                    f"of progressive mask1 for {result_id}"
                )

            mask_path_value = result.get("mask_path")
            if not isinstance(mask_path_value, str):
                raise ValueError(f"threshold mask for {result_id} has no path")
            mask_path = _anchored(Path(mask_path_value), repo_root)
            _verify_hash(
                mask_path,
                result.get("mask_sha256"),
                f"native threshold mask {result_id}",
            )
            checked_paths.add(mask_path)
            with Image.open(mask_path) as opened:
                if opened.mode != "L":
                    raise ValueError(
                        f"native threshold mask is not mode L for {result_id}"
                    )
                mask_array = np.asarray(opened, dtype=np.uint8)
            if mask_array.shape != (height, width):
                raise ValueError(f"invalid threshold mask shape for {result_id}")
            _require_equal(
                result.get("mask_shape"),
                [height, width],
                f"threshold mask shape metadata for {result_id}",
            )
            if not set(np.unique(mask_array).tolist()).issubset({0, 255}):
                raise ValueError(f"threshold mask is not binary for {result_id}")
            expected_mask = np.where(
                native_map > MASK_THRESHOLD,
                np.uint8(255),
                np.uint8(0),
            )
            if not np.array_equal(mask_array, expected_mask):
                raise ValueError(
                    f"strict >0.5 threshold mask mismatch for {result_id}"
                )

            if result["kind"] == "forged":
                target_value = pair.input_row.get("gt_mask_path")
                if not isinstance(target_value, str):
                    raise ValueError(
                        f"forged sample has no GT mask: {pair.task_id}"
                    )
                target_path = _anchored(Path(target_value), repo_root)
                _verify_hash(
                    target_path,
                    pair.input_row.get("gt_mask_sha256"),
                    f"ground-truth mask {pair.task_id}",
                )
                checked_paths.add(target_path)
                with Image.open(target_path) as opened:
                    target = np.asarray(
                        opened.convert("L"),
                        dtype=np.uint8,
                    ) > 0
                if target.shape != (height, width):
                    raise ValueError(
                        f"ground-truth shape mismatch for {pair.task_id}"
                    )
                if not target.any():
                    raise ValueError(
                        f"forged ground truth is empty for {pair.task_id}"
                    )
                forged_target = target
                forged_native = native_map
                forged_prediction = mask_array > 0
            else:
                target = np.zeros((height, width), dtype=bool)
            _validate_metrics(
                result,
                score_map=native_map,
                target=target,
                space="native",
            )
            _validate_metrics(
                result,
                score_map=model_maps[0],
                target=_resize_target_model_256(target),
                space="model_256",
            )

        if (
            forged_target is None
            or forged_native is None
            or forged_prediction is None
        ):
            raise ValueError(f"forged artifacts were not loaded for {pair.task_id}")
        best, all_hist, positive_hist = _oracle_histograms(
            forged_native,
            forged_target,
            bins=bins,
        )
        per_image_best.append({"task_id": pair.task_id, **best})
        global_all += all_hist
        global_positive += positive_hist

        x1, y1, x2, y2 = (
            int(value) for value in pair.input_row["edit_region_xyxy"]
        )
        if not (
            0 <= x1 < x2 <= forged_prediction.shape[1]
            and 0 <= y1 < y2 <= forged_prediction.shape[0]
        ):
            raise ValueError(f"invalid edit box for {pair.task_id}")
        box_area = (x2 - x1) * (y2 - y1)
        intersection = int(
            np.count_nonzero(forged_prediction[y1:y2, x1:x2])
        )
        predicted_area = int(np.count_nonzero(forged_prediction))
        union = predicted_area + box_area - intersection
        box_iou = intersection / union if union else 0.0
        box_ious.append(box_iou)
        box_hits += int(box_iou > 0.3)

    global_best = _best_from_histograms(global_all, global_positive)
    best_f1 = [float(row["f1"]) for row in per_image_best]
    best_iou = [float(row["iou"]) for row in per_image_best]
    return {
        "artifact_integrity": {
            "status": "ok",
            "checked_files": len(checked_paths),
            "pairs": len(pairs),
            "result_images": len(pairs) * 2,
            "checks": [
                "canonical images, GT masks, and all artifact file hashes",
                "independent Pillow decode reproduces the recorded RGB tensor hash",
                "all four progressive maps are finite float32 probabilities",
                "native float32 map equals mask1 bilinear align_corners=True restore",
                "classification probability is independently softmaxed from logits",
                "T1 decision and T2 mask both use strict greater-than 0.5",
                "native mask bit-exactly equals native probability map > 0.5",
                "native and model-256 localization metrics are recomputed from GT",
            ],
        },
        "localization_best_threshold": {
            "status": "posthoc_descriptive_only",
            "eligible_for_primary_metrics": False,
            "fixed_threshold_unchanged": MASK_THRESHOLD,
            "test_set_threshold_selection": False,
            "approximation": (
                f"native float32 maps quantized into {bins} uniform bins over "
                "[0,1], with strict greater-than candidates"
            ),
            "per_image_oracle": {
                "images": len(per_image_best),
                "f1_mean": float(np.mean(best_f1)),
                "f1_median": float(np.median(best_f1)),
                "iou_mean": float(np.mean(best_iou)),
                "iou_median": float(np.median(best_iou)),
            },
            "single_global_oracle": global_best,
        },
        "box_hit_at_mask_threshold_0_5": {
            "definition": (
                "IoU(predicted native binary mask, edit_region_xyxy) > 0.3"
            ),
            "mask_threshold": MASK_THRESHOLD,
            "mask_threshold_operator": THRESHOLD_OPERATOR,
            "box_hit_threshold_operator": THRESHOLD_OPERATOR,
            "hits": box_hits,
            "images": len(pairs),
            "rate": box_hits / len(pairs),
            "iou_mean": float(np.mean(box_ious)),
            "iou_median": float(np.median(box_ious)),
            "iou_max": float(np.max(box_ious)),
        },
    }


def _tpr_at_fpr(
    labels: np.ndarray,
    scores: np.ndarray,
    target_fpr: float,
) -> float:
    false_positive_rate, true_positive_rate, _ = roc_curve(
        labels,
        scores,
        drop_intermediate=False,
    )
    eligible = np.flatnonzero(false_positive_rate <= target_fpr)
    return float(np.max(true_positive_rate[eligible]))


def _slice_arrays(pairs: list[Pair]) -> dict[str, np.ndarray]:
    forged_metrics = [
        _require_mapping(
            _require_mapping(
                pair.forged.get("localization"),
                f"forged localization for {pair.task_id}",
            ).get("native"),
            f"forged native localization for {pair.task_id}",
        )
        for pair in pairs
    ]
    real_metrics = [
        _require_mapping(
            _require_mapping(
                pair.real.get("localization"),
                f"real localization for {pair.task_id}",
            ).get("native"),
            f"real native localization for {pair.task_id}",
        )
        for pair in pairs
    ]
    return {
        "real_score": np.asarray(
            [
                _recomputed_t1_score(pair.real, validate_recorded=True)
                for pair in pairs
            ],
            dtype=np.float64,
        ),
        "forged_score": np.asarray(
            [
                _recomputed_t1_score(pair.forged, validate_recorded=True)
                for pair in pairs
            ],
            dtype=np.float64,
        ),
        "pixel_ap": np.asarray(
            [float(row["pixel_ap"]) for row in forged_metrics],
            dtype=np.float64,
        ),
        "pixel_f1": np.asarray(
            [float(row["f1"]) for row in forged_metrics],
            dtype=np.float64,
        ),
        "pixel_iou": np.asarray(
            [float(row["iou"]) for row in forged_metrics],
            dtype=np.float64,
        ),
        "tp": np.asarray(
            [int(row["tp"]) for row in forged_metrics],
            dtype=np.int64,
        ),
        "fp": np.asarray(
            [int(row["fp"]) for row in forged_metrics],
            dtype=np.int64,
        ),
        "fn": np.asarray(
            [int(row["fn"]) for row in forged_metrics],
            dtype=np.int64,
        ),
        "forged_pixels": np.asarray(
            [int(row["pixels"]) for row in forged_metrics],
            dtype=np.int64,
        ),
        "forged_target_positive_pixels": np.asarray(
            [int(row["target_positive_pixels"]) for row in forged_metrics],
            dtype=np.int64,
        ),
        "real_predicted_positive_pixels": np.asarray(
            [int(row["predicted_positive_pixels"]) for row in real_metrics],
            dtype=np.int64,
        ),
        "real_pixels": np.asarray(
            [int(row["pixels"]) for row in real_metrics],
            dtype=np.int64,
        ),
        "real_predicted_positive_fraction": np.asarray(
            [
                float(row["predicted_positive_fraction"])
                for row in real_metrics
            ],
            dtype=np.float64,
        ),
    }


def _detection_point(
    real: np.ndarray,
    forged: np.ndarray,
) -> dict[str, float]:
    labels = np.concatenate(
        [
            np.zeros(real.size, dtype=np.int64),
            np.ones(forged.size, dtype=np.int64),
        ]
    )
    scores = np.concatenate([real, forged])
    if (
        not np.isfinite(scores).all()
        or float(np.min(scores)) < 0.0
        or float(np.max(scores)) > 1.0
    ):
        raise ValueError("T1 scores are non-finite or outside [0, 1]")
    predictions = scores > CLASSIFICATION_THRESHOLD
    positive = labels == 1
    negative = ~positive
    tp = int(np.count_nonzero(predictions & positive))
    fp = int(np.count_nonzero(predictions & negative))
    fn = int(np.count_nonzero(~predictions & positive))
    tn = int(np.count_nonzero(~predictions & negative))
    sensitivity = _safe_div(tp, tp + fn)
    specificity = _safe_div(tn, tn + fp)
    if sensitivity is None or specificity is None:
        raise ValueError("paired T1 slice must contain both classes")
    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "average_precision": float(average_precision_score(labels, scores)),
        "tpr_at_fpr_5_percent": _tpr_at_fpr(labels, scores, 0.05),
        "accuracy_at_0_5": float(np.mean(predictions == positive)),
        "balanced_accuracy_at_0_5": (sensitivity + specificity) / 2.0,
        "image_f1_at_0_5": _safe_div(2 * tp, 2 * tp + fp + fn) or 0.0,
        "paired_ranking_accuracy": float(np.mean(forged > real)),
        "paired_score_delta_mean": float(np.mean(forged - real)),
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "tn": float(tn),
    }


def _point_metrics(arrays: dict[str, np.ndarray]) -> dict[str, float]:
    detection = _detection_point(
        arrays["real_score"],
        arrays["forged_score"],
    )
    tp = int(np.sum(arrays["tp"]))
    fp = int(np.sum(arrays["fp"]))
    fn = int(np.sum(arrays["fn"]))
    real_positive = int(np.sum(arrays["real_predicted_positive_pixels"]))
    real_pixels = int(np.sum(arrays["real_pixels"]))
    f1_denominator = 2 * tp + fp + fn
    iou_denominator = tp + fp + fn
    if f1_denominator <= 0 or iou_denominator <= 0:
        raise ValueError("forged localization slice has no positive target pixels")
    if real_pixels <= 0:
        raise ValueError("real localization slice has no evaluated pixels")
    return {
        **{
            key: detection[key]
            for key in (
                "auroc",
                "average_precision",
                "tpr_at_fpr_5_percent",
                "accuracy_at_0_5",
                "balanced_accuracy_at_0_5",
                "image_f1_at_0_5",
                "paired_ranking_accuracy",
                "paired_score_delta_mean",
            )
        },
        "pixel_ap_macro": float(np.mean(arrays["pixel_ap"])),
        "pixel_f1_macro_at_0_5": float(np.mean(arrays["pixel_f1"])),
        "pixel_iou_macro_at_0_5": float(np.mean(arrays["pixel_iou"])),
        "pixel_f1_micro_at_0_5": 2.0 * tp / f1_denominator,
        "pixel_iou_micro_at_0_5": tp / iou_denominator,
        "real_false_positive_area_fraction_macro_at_0_5": float(
            np.mean(arrays["real_predicted_positive_fraction"])
        ),
        "real_false_positive_area_fraction_micro_at_0_5": (
            real_positive / real_pixels
        ),
        **{
            f"image_{key}_at_0_5": detection[key]
            for key in ("tp", "fp", "fn", "tn")
        },
    }


def _percentile_ci(values: Iterable[float]) -> list[float]:
    array = np.asarray(list(values), dtype=np.float64)
    if not array.size:
        raise ValueError("cannot compute confidence interval from no values")
    return [
        float(np.percentile(array, 2.5)),
        float(np.percentile(array, 97.5)),
    ]


def _sign_test(real: np.ndarray, forged: np.ndarray) -> dict[str, Any]:
    delta = forged - real
    wins = int(np.count_nonzero(delta > 0))
    losses = int(np.count_nonzero(delta < 0))
    ties = int(np.count_nonzero(delta == 0))
    non_ties = wins + losses
    if non_ties:
        lower_tail = min(wins, losses)
        probability = min(
            1.0,
            2.0
            * sum(math.comb(non_ties, k) for k in range(lower_tail + 1))
            / (2**non_ties),
        )
    else:
        probability = 1.0
    return {
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "two_sided_exact_p": probability,
    }


def summarize_psccnet_pair_slice(
    pairs: list[Pair],
    *,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    if not pairs:
        raise ValueError("pair slice is empty")
    if (
        isinstance(iterations, bool)
        or not isinstance(iterations, (int, np.integer))
        or iterations <= 0
    ):
        raise ValueError("bootstrap iterations must be a positive integer")
    arrays = _slice_arrays(pairs)
    point = _point_metrics(arrays)
    rng = np.random.default_rng(seed)
    replicates: dict[str, list[float]] = {
        name: [] for name in BOOTSTRAP_METRICS
    }
    for _ in range(int(iterations)):
        indices = rng.integers(0, len(pairs), size=len(pairs))
        sampled = {name: values[indices] for name, values in arrays.items()}
        values = _point_metrics(sampled)
        for name in BOOTSTRAP_METRICS:
            replicates[name].append(float(values[name]))
    edit_fractions = (
        arrays["forged_target_positive_pixels"] / arrays["forged_pixels"]
    )
    return {
        "pairs": len(pairs),
        "images": len(pairs) * 2,
        "t1_score_semantics": T1_SCORE_SEMANTICS,
        "t1_positive_class_index": 1,
        "t1_prediction_inversion": False,
        "t2_map_semantics": T2_MAP_SEMANTICS,
        "classification_threshold": CLASSIFICATION_THRESHOLD,
        "mask_threshold": MASK_THRESHOLD,
        "threshold_operator": THRESHOLD_OPERATOR,
        "threshold_source": "pre_registered_fixed_0_5_not_test_selected",
        "auc_direction": "class_1_forged_as_recorded_no_flip",
        **{
            name: {
                "estimate": float(point[name]),
                "ci95_percentile": _percentile_ci(replicates[name]),
            }
            for name in BOOTSTRAP_METRICS
        },
        "image_confusion_at_0_5": {
            key: int(point[f"image_{key}_at_0_5"])
            for key in ("tp", "fp", "fn", "tn")
        },
        "paired_sign_test": _sign_test(
            arrays["real_score"],
            arrays["forged_score"],
        ),
        "edit_fraction": {
            "min": float(np.min(edit_fractions)),
            "median": float(np.median(edit_fractions)),
            "mean": float(np.mean(edit_fractions)),
            "max": float(np.max(edit_fractions)),
        },
        "pixel_ap_median": float(np.median(arrays["pixel_ap"])),
    }


_summarize_psccnet_pair_slice_local = summarize_psccnet_pair_slice


def _quintiles(pairs: list[Pair]) -> list[tuple[str, list[Pair]]]:
    ordered = sorted(pairs, key=lambda pair: (pair.edit_fraction, pair.task_id))
    chunk_count = min(5, len(ordered))
    chunks = np.array_split(np.asarray(ordered, dtype=object), chunk_count)
    return [
        (
            (
                f"q{index}_"
                f"{'smallest' if index == 1 else ''}"
                f"{'largest' if index == chunk_count else ''}"
            ).rstrip("_"),
            list(chunk),
        )
        for index, chunk in enumerate(chunks, start=1)
    ]


def _relative_or_absolute(path: Path, repo_root: Path) -> str:
    try:
        return str(path.relative_to(repo_root))
    except ValueError:
        return str(path)


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = args.repo_root.resolve()
    results_dir = _anchored(args.results_dir, repo_root)
    result_path = results_dir / f"{args.run_id}.jsonl"
    run_manifest_path = results_dir / f"{args.run_id}.run_manifest.json"
    summary_path = results_dir / f"{args.run_id}.summary.json"
    output_path = (
        _anchored(args.output, repo_root)
        if args.output is not None
        else results_dir / f"{args.run_id}.analysis.json"
    )
    input_path = _anchored(args.inputs, repo_root)
    for path in (result_path, run_manifest_path, summary_path, input_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    result_rows = read_jsonl(result_path)
    all_input_rows = read_jsonl(input_path)
    manifest = _require_mapping(
        json.loads(run_manifest_path.read_text(encoding="utf-8")),
        "run manifest",
    )
    summary = _require_mapping(
        json.loads(summary_path.read_text(encoding="utf-8")),
        "run summary",
    )
    input_rows = _select_manifest_inputs(all_input_rows, manifest)
    result_history = summarize_result_history(result_rows)
    provenance = validate_provenance(
        repo_root=repo_root,
        run_id=args.run_id,
        input_path=input_path,
        input_rows=input_rows,
        result_rows=result_rows,
        manifest=manifest,
        summary=summary,
    )
    pairs = _load_pairs(result_rows, input_rows)
    audit = audit_and_best_threshold(
        pairs,
        repo_root=repo_root,
        bins=args.oracle_histogram_bins,
    )

    overall = summarize_psccnet_pair_slice(
        pairs,
        iterations=args.bootstrap_iterations,
        seed=args.bootstrap_seed,
    )
    by_domain = {
        domain: summarize_psccnet_pair_slice(
            [pair for pair in pairs if pair.domain == domain],
            iterations=args.bootstrap_iterations,
            seed=args.bootstrap_seed + index,
        )
        for index, domain in enumerate(
            sorted({pair.domain for pair in pairs}),
            start=1,
        )
    }
    by_edit_quintile = {
        name: summarize_psccnet_pair_slice(
            chunk,
            iterations=args.bootstrap_iterations,
            seed=args.bootstrap_seed + 100 + index,
        )
        for index, (name, chunk) in enumerate(_quintiles(pairs), start=1)
    }

    value = {
        "schema_version": "psccnet_posthoc_analysis_v1",
        "run_id": args.run_id,
        "created_at": utc_now(),
        "task_scope": {
            "primary_t1": T1_SCORE_SEMANTICS,
            "primary_t2": T2_MAP_SEMANTICS,
            "positive_class_index": 1,
            "prediction_inversion": False,
            "classification_threshold": CLASSIFICATION_THRESHOLD,
            "mask_threshold": MASK_THRESHOLD,
            "threshold_operator": THRESHOLD_OPERATOR,
            "test_set_threshold_selection": False,
            "auc_flip": False,
        },
        "sources": {
            "results_path": _relative_or_absolute(result_path, repo_root),
            "results_sha256": sha256_file(result_path),
            "run_manifest_path": _relative_or_absolute(
                run_manifest_path,
                repo_root,
            ),
            "run_manifest_sha256": sha256_file(run_manifest_path),
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
        },
        "overall": overall,
        "by_domain": by_domain,
        "by_edit_fraction_quintile": by_edit_quintile,
        "provenance_integrity": provenance,
        "result_history": result_history,
        **audit,
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
    parser.add_argument("--output", type=Path)
    parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260724)
    parser.add_argument(
        "--oracle-histogram-bins",
        type=int,
        default=HISTOGRAM_BINS,
    )
    return parser.parse_args()


def main() -> None:
    analyze(parse_args())


if __name__ == "__main__":
    main()
