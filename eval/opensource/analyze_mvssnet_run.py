#!/usr/bin/env python3
"""Audit and statistically analyze a paired official MVSS-Net CASIAv2 run."""

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
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    roc_auc_score,
    roc_curve,
)

from eval.opensource.common import (
    atomic_write_json,
    read_jsonl,
    sha256_file,
    stable_json,
    utc_now,
)
from eval.opensource.mvssnet_metrics import (
    summarize_mvssnet_pair_slice,
)


DEFAULT_RUN_ID = "mvssnet_casia_mouse_canonical_v1_full275_20260723"
DEFAULT_RESULTS_DIR = Path("results/opensource/mvssnet")
DEFAULT_INPUTS = Path("outputs/opensource/mouse_canonical_v1/inputs.jsonl")

MODEL_NAME = "MVSS-Net (CASIAv2)"
MODEL_SLUG = "mvssnet_casiav2_iccv2021"
MODEL_INPUT_SIZE = 512
CLASSIFICATION_THRESHOLD = 0.5
MASK_THRESHOLD = 0.5
NORMALIZE_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
NORMALIZE_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)

PRIMARY_SCORE_SEMANTICS = (
    "continuous_global_max_of_model_512_sigmoid_probability"
)
OFFICIAL_PNG_SCORE_SEMANTICS = (
    "maximum_of_saved_native_uint8_map_divided_by_255"
)
T2_MAP_SEMANTICS = "official_native_uint8_map_divided_by_255"
THRESHOLD_COMPARISON = "strict_greater_than"
THRESHOLD_OPERATOR = ">"

BOOTSTRAP_METRICS = (
    "auroc",
    "average_precision",
    "tpr_at_fpr_5_percent",
    "accuracy_at_0_5",
    "image_f1_at_0_5",
    "paired_ranking_accuracy",
    "paired_score_delta_mean",
    "official_png_auroc",
    "official_png_average_precision",
    "official_png_accuracy_at_0_5",
    "official_png_image_f1_at_0_5",
    "official_png_paired_ranking_accuracy",
    "official_png_paired_score_delta_mean",
    "pixel_ap_macro",
    "pixel_f1_macro_at_0_5",
    "pixel_iou_macro_at_0_5",
    "pixel_f1_micro_at_0_5",
    "pixel_iou_micro_at_0_5",
    "real_predicted_positive_fraction_macro_at_0_5",
    "real_predicted_positive_fraction_micro_at_0_5",
)


def _cv2():
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "MVSS-Net artifact audit requires an OpenCV build compatible "
            "with the active NumPy runtime"
        ) from exc
    return cv2


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
        return float(metrics["target_positive_pixels"]) / float(
            metrics["pixels"]
        )


def _load_runner_pins() -> SimpleNamespace:
    """Load immutable release constants without importing the model itself."""
    from eval.opensource import run_mvssnet

    required = (
        "MODEL_SOURCE_COMMIT",
        "MODEL_NETWORK_SHA256",
        "MODEL_INFERENCE_SHA256",
        "MODEL_EVALUATE_SHA256",
        "MODEL_TOOLS_SHA256",
        "MODEL_TRANSFORMS_SHA256",
        "CHECKPOINT_SHA256",
        "CHECKPOINT_BYTES",
        "CHECKPOINT_STATE_KEYS",
    )
    missing = [name for name in required if not hasattr(run_mvssnet, name)]
    if missing:
        raise RuntimeError(
            "MVSS-Net runner does not export required audit constants: "
            f"{missing}"
        )
    return SimpleNamespace(
        **{name: getattr(run_mvssnet, name) for name in required}
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
    for index, item in enumerate(ordered):
        contract = _require_mapping(item, f"ordered input {index}")
        sample_id = contract.get("sample_id")
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


def _verify_adapter_contract(
    value: Any,
    *,
    repo_root: Path,
) -> int:
    if not isinstance(value, list) or not value:
        raise ValueError("manifest adapter_contract is empty or invalid")
    checked = 0
    for index, item in enumerate(value):
        contract = _require_mapping(item, f"adapter contract entry {index}")
        path_value = contract.get("path")
        if not isinstance(path_value, str):
            raise ValueError(f"adapter contract entry {index} has no path")
        _verify_hash(
            _anchored(Path(path_value), repo_root),
            contract.get("sha256"),
            f"adapter contract entry {index}",
        )
        checked += 1
    return checked


def _source_hash(
    source_files: dict[str, Any],
    key: str,
    expected: str,
) -> None:
    value = _require_mapping(
        source_files.get(key),
        f"manifest source file {key}",
    )
    _require_equal(
        value.get("sha256"),
        expected,
        f"manifest source file {key} SHA-256",
    )


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
    if not dataset_manifest_path.is_file():
        raise FileNotFoundError(dataset_manifest_path)
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
    _require_equal(
        manifest.get("expected_images"),
        len(input_rows),
        "manifest expected image count",
    )
    _require_equal(
        manifest.get("expected_pairs"),
        len({int(row["pair_rank"]) for row in input_rows}),
        "manifest expected pair count",
    )

    model = _require_mapping(manifest.get("model"), "manifest model")
    implementation = _require_mapping(
        model.get("implementation"),
        "manifest implementation",
    )
    checkpoint = _require_mapping(
        model.get("checkpoint"),
        "manifest checkpoint",
    )
    license_value = _require_mapping(
        model.get("license"),
        "manifest license",
    )
    for actual, expected, label in (
        (model.get("name"), MODEL_NAME, "manifest model name"),
        (model.get("model_slug"), MODEL_SLUG, "manifest model slug"),
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
            model.get("supports_image_level_t1"),
            True,
            "manifest T1 support flag",
        ),
        (
            model.get("supports_pixel_level_t2"),
            True,
            "manifest T2 support flag",
        ),
        (
            model.get("image_level_head"),
            "none_map_global_max_pooling",
            "manifest image-level head",
        ),
        (
            license_value.get("project_wide_status"),
            "no_project_license_found",
            "manifest license status",
        ),
        (
            license_value.get("classification"),
            "source_available_research_release",
            "manifest license classification",
        ),
        (
            checkpoint.get("sha256"),
            pins.CHECKPOINT_SHA256,
            "manifest checkpoint SHA-256",
        ),
        (
            checkpoint.get("bytes"),
            pins.CHECKPOINT_BYTES,
            "manifest checkpoint byte count",
        ),
        (
            checkpoint.get("state_keys"),
            pins.CHECKPOINT_STATE_KEYS,
            "manifest checkpoint state-key count",
        ),
        (
            checkpoint.get("strict_load"),
            True,
            "manifest strict-load flag",
        ),
        (
            checkpoint.get("safe_weights_only_load"),
            True,
            "manifest safe-load flag",
        ),
    ):
        _require_equal(actual, expected, label)
    for actual, expected, label in (
        (
            implementation.get("network_sha256"),
            pins.MODEL_NETWORK_SHA256,
            "manifest network SHA-256",
        ),
        (
            implementation.get("inference_sha256"),
            pins.MODEL_INFERENCE_SHA256,
            "manifest official inference SHA-256",
        ),
        (
            implementation.get("evaluation_sha256"),
            pins.MODEL_EVALUATE_SHA256,
            "manifest official evaluation SHA-256",
        ),
        (
            implementation.get("tools_sha256"),
            pins.MODEL_TOOLS_SHA256,
            "manifest official tools SHA-256",
        ),
        (
            implementation.get("transforms_sha256"),
            pins.MODEL_TRANSFORMS_SHA256,
            "manifest official transforms SHA-256",
        ),
    ):
        _require_equal(actual, expected, label)

    inference = _require_mapping(manifest.get("inference"), "manifest inference")
    for actual, expected, label in (
        (inference.get("precision"), "float32", "manifest precision"),
        (inference.get("batch_size"), 1, "manifest batch size"),
        (
            inference.get("model_input_size"),
            [MODEL_INPUT_SIZE, MODEL_INPUT_SIZE],
            "manifest model input size",
        ),
        (
            inference.get("channel_order"),
            "BGR",
            "manifest input channel order",
        ),
        (
            inference.get("primary_t1_score"),
            PRIMARY_SCORE_SEMANTICS,
            "manifest primary T1 semantics",
        ),
        (
            inference.get("official_evaluate_t1_score"),
            OFFICIAL_PNG_SCORE_SEMANTICS,
            "manifest official-PNG T1 semantics",
        ),
        (
            inference.get("classification_threshold"),
            CLASSIFICATION_THRESHOLD,
            "manifest classification threshold",
        ),
        (
            inference.get("classification_threshold_comparison"),
            THRESHOLD_COMPARISON,
            "manifest classification comparison",
        ),
        (
            inference.get("mask_threshold"),
            MASK_THRESHOLD,
            "manifest mask threshold",
        ),
        (
            inference.get("mask_threshold_comparison"),
            THRESHOLD_COMPARISON,
            "manifest mask comparison",
        ),
    ):
        _require_equal(actual, expected, label)

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
        seen_ids.add(row_id)
        latest[row_id] = row
        input_row = expected_by_id[row_id]
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
            "checkpoint_sha256": pins.CHECKPOINT_SHA256,
            "valid_for_t1": True,
            "valid_for_t2": True,
        }
        for key, expected in expected_values.items():
            _require_equal(
                row.get(key),
                expected,
                f"result row {line_number} field {key}",
            )
        if row.get("status") not in {"ok", "error"}:
            raise ValueError(
                f"result row {line_number} has invalid status "
                f"{row.get('status')!r}"
            )
        if row.get("status") == "ok":
            for actual, expected, label in (
                (
                    row.get("raw_score_semantics"),
                    PRIMARY_SCORE_SEMANTICS,
                    "primary score semantics",
                ),
                (
                    row.get("official_png_score_semantics"),
                    OFFICIAL_PNG_SCORE_SEMANTICS,
                    "official-PNG score semantics",
                ),
                (
                    row.get("classification_threshold"),
                    CLASSIFICATION_THRESHOLD,
                    "classification threshold",
                ),
                (
                    row.get("mask_threshold"),
                    MASK_THRESHOLD,
                    "mask threshold",
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
            pins.CHECKPOINT_SHA256,
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
    for actual, expected, label in (
        (
            task_scope.get("primary_detection_score"),
            "score",
            "summary primary T1 score field",
        ),
        (
            task_scope.get("primary_detection_semantics"),
            "raw_GMP_model_512",
            "summary primary T1 semantics",
        ),
        (
            task_scope.get("secondary_detection_score"),
            "official_png_score",
            "summary secondary T1 score field",
        ),
        (
            task_scope.get("primary_localization_space"),
            "native",
            "summary primary T2 space",
        ),
        (
            task_scope.get("primary_localization_semantics"),
            "official_uint8_div_255",
            "summary primary T2 semantics",
        ),
        (
            task_scope.get("threshold_operator"),
            THRESHOLD_OPERATOR,
            "summary threshold operator",
        ),
    ):
        _require_equal(actual, expected, label)

    adapter_files_checked = _verify_adapter_contract(
        manifest.get("adapter_contract"),
        repo_root=repo_root,
    )
    return {
        "status": "ok",
        "run_manifest_fingerprint": fingerprint,
        "inputs_sha256": actual_inputs_sha256,
        "checkpoint_sha256": pins.CHECKPOINT_SHA256,
        "physical_result_rows_validated": len(result_rows),
        "latest_result_rows_validated": len(latest),
        "adapter_contract_files_validated": adapter_files_checked,
        "checks": [
            "run manifest schema, ID, and recomputed immutable fingerprint",
            "canonical dataset manifest, input hash, and ordered selection",
            "pinned source-file and checkpoint constants",
            "no-project-license, strict-load, and weights-only-load metadata",
            "every physical result row against canonical and model provenance",
            "continuous-GMP primary T1 and official-PNG secondary T1 semantics",
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
        by_task.setdefault(str(row["task_id"]), {})[str(row["kind"])] = row
    pairs: list[Pair] = []
    for task_id, values in by_task.items():
        if set(values) != {"real", "forged"}:
            raise ValueError(f"incomplete pair for {task_id}: {sorted(values)}")
        real = values["real"]
        forged = values["forged"]
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


def _tpr_at_fpr(
    labels: np.ndarray,
    scores: np.ndarray,
    target_fpr: float,
) -> float:
    false_positive_rate, true_positive_rate, _ = roc_curve(labels, scores)
    eligible = np.where(false_positive_rate <= target_fpr)[0]
    return float(np.max(true_positive_rate[eligible]))


def _slice_arrays(pairs: list[Pair]) -> dict[str, np.ndarray]:
    forged_metrics = [
        pair.forged["localization"]["native"] for pair in pairs
    ]
    real_metrics = [pair.real["localization"]["native"] for pair in pairs]
    return {
        "real_score": np.asarray(
            [float(pair.real["score"]) for pair in pairs],
            dtype=np.float64,
        ),
        "forged_score": np.asarray(
            [float(pair.forged["score"]) for pair in pairs],
            dtype=np.float64,
        ),
        "real_official_png_score": np.asarray(
            [float(pair.real["official_png_score"]) for pair in pairs],
            dtype=np.float64,
        ),
        "forged_official_png_score": np.asarray(
            [float(pair.forged["official_png_score"]) for pair in pairs],
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
    predictions = scores > CLASSIFICATION_THRESHOLD
    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "average_precision": float(average_precision_score(labels, scores)),
        "tpr_at_fpr_5_percent": _tpr_at_fpr(labels, scores, 0.05),
        "accuracy_at_0_5": float(np.mean(predictions == labels)),
        "image_f1_at_0_5": float(
            f1_score(labels, predictions, zero_division=0)
        ),
        "paired_ranking_accuracy": float(np.mean(forged > real)),
        "paired_score_delta_mean": float(np.mean(forged - real)),
        "tp": float(np.count_nonzero(predictions & (labels == 1))),
        "fp": float(np.count_nonzero(predictions & (labels == 0))),
        "fn": float(np.count_nonzero(~predictions & (labels == 1))),
        "tn": float(np.count_nonzero(~predictions & (labels == 0))),
    }


def _point_metrics(arrays: dict[str, np.ndarray]) -> dict[str, float]:
    primary = _detection_point(arrays["real_score"], arrays["forged_score"])
    official = _detection_point(
        arrays["real_official_png_score"],
        arrays["forged_official_png_score"],
    )
    tp = int(np.sum(arrays["tp"]))
    fp = int(np.sum(arrays["fp"]))
    fn = int(np.sum(arrays["fn"]))
    real_positive = int(np.sum(arrays["real_predicted_positive_pixels"]))
    real_pixels = int(np.sum(arrays["real_pixels"]))
    f1_denominator = 2 * tp + fp + fn
    iou_denominator = tp + fp + fn
    if f1_denominator <= 0 or iou_denominator <= 0:
        raise ValueError("forged localization slice has no positive pixels")
    if real_pixels <= 0:
        raise ValueError("real localization slice has no evaluated pixels")
    return {
        **{
            key: primary[key]
            for key in (
                "auroc",
                "average_precision",
                "tpr_at_fpr_5_percent",
                "accuracy_at_0_5",
                "image_f1_at_0_5",
                "paired_ranking_accuracy",
                "paired_score_delta_mean",
            )
        },
        **{
            f"official_png_{key}": official[key]
            for key in (
                "auroc",
                "average_precision",
                "accuracy_at_0_5",
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
        "real_predicted_positive_fraction_macro_at_0_5": float(
            np.mean(arrays["real_predicted_positive_fraction"])
        ),
        "real_predicted_positive_fraction_micro_at_0_5": (
            real_positive / real_pixels
        ),
        **{
            f"image_{key}_at_0_5": value
            for key, value in primary.items()
            if key in {"tp", "fp", "fn", "tn"}
        },
        **{
            f"official_png_image_{key}_at_0_5": value
            for key, value in official.items()
            if key in {"tp", "fp", "fn", "tn"}
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


def _summarize_mvssnet_pair_slice_local(
    pairs: list[Pair],
    *,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    if not pairs:
        raise ValueError("pair slice is empty")
    if iterations <= 0:
        raise ValueError("bootstrap iterations must be positive")
    arrays = _slice_arrays(pairs)
    point = _point_metrics(arrays)
    rng = np.random.default_rng(seed)
    replicates: dict[str, list[float]] = {
        name: [] for name in BOOTSTRAP_METRICS
    }
    for _ in range(iterations):
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
        "primary_t1_score_semantics": PRIMARY_SCORE_SEMANTICS,
        "secondary_t1_score_semantics": OFFICIAL_PNG_SCORE_SEMANTICS,
        "primary_t2_map_semantics": T2_MAP_SEMANTICS,
        "threshold": CLASSIFICATION_THRESHOLD,
        "threshold_comparison": THRESHOLD_COMPARISON,
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
        "official_png_image_confusion_at_0_5": {
            key: int(point[f"official_png_image_{key}_at_0_5"])
            for key in ("tp", "fp", "fn", "tn")
        },
        "paired_sign_test": _sign_test(
            arrays["real_score"],
            arrays["forged_score"],
        ),
        "official_png_paired_sign_test": _sign_test(
            arrays["real_official_png_score"],
            arrays["forged_official_png_score"],
        ),
        "edit_fraction": {
            "min": float(np.min(edit_fractions)),
            "median": float(np.median(edit_fractions)),
            "mean": float(np.mean(edit_fractions)),
            "max": float(np.max(edit_fractions)),
        },
        "pixel_ap_median": float(np.median(arrays["pixel_ap"])),
    }


def _preprocess_evidence(
    image_path: Path,
) -> tuple[dict[str, Any], np.ndarray]:
    cv2 = _cv2()
    decoded = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if decoded is None:
        raise ValueError(f"OpenCV could not decode canonical image: {image_path}")
    resized = cv2.resize(
        decoded,
        (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE),
        interpolation=cv2.INTER_LINEAR,
    )
    scaled = resized.astype(np.float32) / np.float32(255.0)
    normalized = (
        scaled - NORMALIZE_MEAN.reshape(1, 1, 3)
    ) / NORMALIZE_STD.reshape(1, 1, 3)
    normalized_chw = np.ascontiguousarray(
        normalized.transpose(2, 0, 1),
        dtype=np.float32,
    )
    return (
        {
            "native_size": [int(decoded.shape[1]), int(decoded.shape[0])],
            "model_size": [MODEL_INPUT_SIZE, MODEL_INPUT_SIZE],
            "decoder": "opencv_imread_color",
            "channel_order": "BGR",
            "resize": "opencv_inter_linear_stretch",
            "normalization": {
                "scale": "uint8_divide_255",
                "mean_in_bgr_order": NORMALIZE_MEAN.tolist(),
                "std_in_bgr_order": NORMALIZE_STD.tolist(),
            },
            "decoded_bgr_dtype": str(decoded.dtype),
            "decoded_bgr_shape": list(decoded.shape),
            "decoded_bgr_sha256": _array_sha256(decoded, np.dtype(np.uint8)),
            "resized_bgr_dtype": str(resized.dtype),
            "resized_bgr_shape": list(resized.shape),
            "resized_bgr_sha256": _array_sha256(
                resized,
                np.dtype(np.uint8),
            ),
            "normalized_chw_dtype": str(normalized_chw.dtype),
            "normalized_chw_shape": list(normalized_chw.shape),
            "normalized_chw_sha256": _array_sha256(
                normalized_chw,
                np.dtype(np.float32),
            ),
        },
        normalized_chw,
    )


def _sigmoid(array: np.ndarray) -> np.ndarray:
    import torch

    tensor = torch.from_numpy(
        np.array(array, dtype=np.float32, order="C", copy=True)
    )
    return torch.sigmoid(tensor).numpy().astype(np.float32, copy=False)


def _official_native_uint8(
    model_score_map: np.ndarray,
    *,
    width: int,
    height: int,
) -> np.ndarray:
    cv2 = _cv2()
    model_uint8 = (
        np.asarray(model_score_map, dtype=np.float32) * np.float32(255.0)
    ).astype(np.uint8)
    return cv2.resize(
        model_uint8,
        (width, height),
        interpolation=cv2.INTER_LINEAR,
    )


def _resize_target_model_512(target: np.ndarray) -> np.ndarray:
    cv2 = _cv2()
    resized = cv2.resize(
        np.asarray(target, dtype=np.uint8),
        (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE),
        interpolation=cv2.INTER_NEAREST,
    )
    return resized > 0


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
    if scores.shape != truth.shape:
        raise ValueError("score/target shape mismatch")
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


def _compare_metric(
    recorded: Any,
    expected: Any,
    label: str,
) -> None:
    if expected is None:
        _require_equal(recorded, None, label)
        return
    if isinstance(expected, str):
        _require_equal(recorded, expected, label)
        return
    if isinstance(expected, int):
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


def _best_uint8_threshold(
    all_hist: np.ndarray,
    positive_hist: np.ndarray,
) -> dict[str, Any]:
    if all_hist.shape != (256,) or positive_hist.shape != (256,):
        raise ValueError("exact MVSS threshold search requires 256-bin histograms")
    cumulative_all = np.cumsum(all_hist, dtype=np.int64)
    cumulative_positive = np.cumsum(positive_hist, dtype=np.int64)
    total_all = int(np.sum(all_hist))
    total_positive = int(np.sum(positive_hist))
    predicted = total_all - cumulative_all
    tp = total_positive - cumulative_positive
    fp = predicted - tp
    fn = total_positive - tp
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
        "threshold_byte": best,
        "threshold": best / 255.0,
        "comparison": THRESHOLD_OPERATOR,
        "f1": float(f1[best]),
        "iou": (
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
) -> dict[str, Any]:
    global_all = np.zeros(256, dtype=np.int64)
    global_positive = np.zeros(256, dtype=np.int64)
    per_image_best: list[dict[str, Any]] = []
    box_ious: list[float] = []
    box_hits = 0
    checked_files = 0

    for pair in pairs:
        forged_target: np.ndarray | None = None
        for result in (pair.real, pair.forged):
            image_path = _anchored(Path(str(result["image_path"])), repo_root)
            logits_path = _anchored(
                Path(str(result["raw_logits_model_path"])),
                repo_root,
            )
            model_map_path = _anchored(
                Path(str(result["score_map_model_path"])),
                repo_root,
            )
            native_map_path = _anchored(
                Path(str(result["score_map_native_path"])),
                repo_root,
            )
            mask_path = _anchored(Path(str(result["mask_path"])), repo_root)
            for path, expected, label in (
                (image_path, result["image_sha256"], "canonical image"),
                (
                    logits_path,
                    result["raw_logits_model_sha256"],
                    "model-space raw logits",
                ),
                (
                    model_map_path,
                    result["score_map_model_sha256"],
                    "model-space sigmoid map",
                ),
                (
                    native_map_path,
                    result["score_map_native_sha256"],
                    "official native uint8 score map",
                ),
                (mask_path, result["mask_sha256"], "native threshold mask"),
            ):
                _verify_hash(path, expected, f"{label} {result['id']}")
                checked_files += 1

            width, height = (int(value) for value in result["image_size"])
            evidence, normalized_input = _preprocess_evidence(image_path)
            if normalized_input.shape != (3, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE):
                raise ValueError(f"invalid normalized input for {result['id']}")
            preprocess = _require_mapping(
                result.get("preprocess"),
                f"preprocess metadata for {result['id']}",
            )
            for key, expected in evidence.items():
                _require_equal(
                    preprocess.get(key),
                    expected,
                    f"preprocess {key} for {result['id']}",
                )
            _require_equal(
                evidence["native_size"],
                [width, height],
                f"decoded native size for {result['id']}",
            )

            logits = np.load(logits_path, mmap_mode="r", allow_pickle=False)
            model_map = np.load(
                model_map_path,
                mmap_mode="r",
                allow_pickle=False,
            )
            expected_model_shape = (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE)
            for name, value, shape_field in (
                ("raw logits", logits, "raw_logits_model_shape"),
                ("model score map", model_map, "score_map_model_shape"),
            ):
                if value.shape != expected_model_shape:
                    raise ValueError(
                        f"invalid {name} shape for {result['id']}: {value.shape}"
                    )
                _require_equal(
                    result.get(shape_field),
                    list(expected_model_shape),
                    f"{name} shape metadata for {result['id']}",
                )
                if value.dtype != np.float32:
                    raise ValueError(
                        f"invalid {name} dtype for {result['id']}: {value.dtype}"
                    )
                if not np.isfinite(value).all():
                    raise ValueError(f"non-finite {name} for {result['id']}")
            if float(model_map.min()) < 0.0 or float(model_map.max()) > 1.0:
                raise ValueError(f"out-of-range model map for {result['id']}")
            expected_model_map = _sigmoid(np.asarray(logits))
            if not np.allclose(
                model_map,
                expected_model_map,
                rtol=1e-6,
                atol=1e-7,
            ):
                raise ValueError(
                    f"model score map is not sigmoid(raw logits) for {result['id']}"
                )

            with Image.open(native_map_path) as opened:
                if opened.mode != "L":
                    raise ValueError(
                        f"native score map is not mode L for {result['id']}"
                    )
                native_uint8 = np.asarray(opened, dtype=np.uint8)
            if native_uint8.shape != (height, width):
                raise ValueError(
                    f"invalid native score-map shape for {result['id']}"
                )
            _require_equal(
                result.get("score_map_native_shape"),
                [height, width],
                f"native score-map shape metadata for {result['id']}",
            )
            expected_native = _official_native_uint8(
                np.asarray(model_map),
                width=width,
                height=height,
            )
            if not np.array_equal(native_uint8, expected_native):
                raise ValueError(
                    "official postprocess mismatch for "
                    f"{result['id']}: expected sigmoid -> uint8 -> native resize"
                )
            native_score = native_uint8.astype(np.float32) / np.float32(255.0)

            with Image.open(mask_path) as opened:
                if opened.mode != "L":
                    raise ValueError(
                        f"threshold mask is not mode L for {result['id']}"
                    )
                mask_uint8 = np.asarray(opened, dtype=np.uint8)
            if mask_uint8.shape != (height, width):
                raise ValueError(f"invalid mask shape for {result['id']}")
            _require_equal(
                result.get("mask_shape"),
                [height, width],
                f"mask shape metadata for {result['id']}",
            )
            if not set(np.unique(mask_uint8).tolist()).issubset({0, 255}):
                raise ValueError(f"mask is not binary for {result['id']}")
            expected_mask = np.where(
                native_score > MASK_THRESHOLD,
                np.uint8(255),
                np.uint8(0),
            )
            if not np.array_equal(mask_uint8, expected_mask):
                raise ValueError(
                    f"strict native threshold mask mismatch for {result['id']}"
                )

            expected_score = float(np.max(model_map))
            expected_official_score = float(np.max(native_uint8)) / 255.0
            for key, expected in (
                ("score", expected_score),
                ("official_png_score", expected_official_score),
            ):
                recorded = result.get(key)
                if not isinstance(recorded, (int, float)) or not math.isclose(
                    float(recorded),
                    expected,
                    rel_tol=1e-7,
                    abs_tol=1e-8,
                ):
                    raise ValueError(
                        f"{key} mismatch for {result['id']}: "
                        f"{recorded!r} != {expected!r}"
                    )
            _require_equal(
                result.get("decision"),
                expected_score > CLASSIFICATION_THRESHOLD,
                f"primary decision for {result['id']}",
            )
            _require_equal(
                result.get("official_png_decision"),
                expected_official_score > CLASSIFICATION_THRESHOLD,
                f"official-PNG decision for {result['id']}",
            )

            if result["kind"] == "forged":
                mask_value = pair.input_row.get("gt_mask_path")
                mask_sha = pair.input_row.get("gt_mask_sha256")
                if not isinstance(mask_value, str):
                    raise ValueError(
                        f"forged sample has no GT mask: {pair.task_id}"
                    )
                target_path = _anchored(Path(mask_value), repo_root)
                _verify_hash(
                    target_path,
                    mask_sha,
                    f"ground-truth mask {pair.task_id}",
                )
                checked_files += 1
                with Image.open(target_path) as opened:
                    target_uint8 = np.asarray(
                        opened.convert("L"),
                        dtype=np.uint8,
                    )
                if target_uint8.shape != (height, width):
                    raise ValueError(
                        f"ground-truth shape mismatch for {pair.task_id}"
                    )
                forged_target = target_uint8 > 0
                target = forged_target
            else:
                target = np.zeros((height, width), dtype=bool)
            _validate_metrics(
                result,
                score_map=native_score,
                target=target,
                space="native",
            )
            _validate_metrics(
                result,
                score_map=np.asarray(model_map),
                target=_resize_target_model_512(target),
                space="model_512",
            )

        if forged_target is None:
            raise ValueError(f"forged target not loaded for {pair.task_id}")
        forged_native_path = _anchored(
            Path(str(pair.forged["score_map_native_path"])),
            repo_root,
        )
        with Image.open(forged_native_path) as opened:
            forged_uint8 = np.asarray(opened.convert("L"), dtype=np.uint8)
        all_hist = np.bincount(
            forged_uint8.reshape(-1),
            minlength=256,
        ).astype(np.int64)
        positive_hist = np.bincount(
            forged_uint8[forged_target],
            minlength=256,
        ).astype(np.int64)
        global_all += all_hist
        global_positive += positive_hist
        per_image_best.append(
            {
                "task_id": pair.task_id,
                **_best_uint8_threshold(all_hist, positive_hist),
            }
        )

        mask_path = _anchored(
            Path(str(pair.forged["mask_path"])),
            repo_root,
        )
        with Image.open(mask_path) as opened:
            prediction = np.asarray(opened.convert("L"), dtype=np.uint8) > 0
        x1, y1, x2, y2 = (
            int(value) for value in pair.input_row["edit_region_xyxy"]
        )
        if not (
            0 <= x1 < x2 <= prediction.shape[1]
            and 0 <= y1 < y2 <= prediction.shape[0]
        ):
            raise ValueError(f"invalid edit box for {pair.task_id}")
        box_area = (x2 - x1) * (y2 - y1)
        intersection = int(np.count_nonzero(prediction[y1:y2, x1:x2]))
        predicted_area = int(np.count_nonzero(prediction))
        union = predicted_area + box_area - intersection
        box_iou = intersection / union if union else 0.0
        box_ious.append(box_iou)
        box_hits += int(box_iou > 0.3)

    global_best = _best_uint8_threshold(global_all, global_positive)
    best_f1 = [float(row["f1"]) for row in per_image_best]
    best_iou = [float(row["iou"]) for row in per_image_best]
    return {
        "artifact_integrity": {
            "status": "ok",
            "checked_files": checked_files,
            "pairs": len(pairs),
            "result_images": len(pairs) * 2,
            "checks": [
                "canonical image and all four per-image artifact hashes",
                "independent OpenCV BGR decode, resize, and normalization hashes",
                "float32 raw logits and sigmoid maps are finite with shape 512x512",
                "model score map equals sigmoid(raw logits)",
                "native PNG bit-exactly equals sigmoid map quantized before OpenCV resize",
                "primary score equals continuous model-space GMP",
                "secondary score equals native uint8 PNG maximum divided by 255",
                "both decisions use strict greater-than 0.5",
                "binary mask bit-exactly equals native PNG score > 0.5",
                "ground-truth hashes and all recorded localization metrics",
            ],
        },
        "localization_best_threshold": {
            "task_scope": T2_MAP_SEMANTICS,
            "search": (
                "exact over all 256 native uint8 score levels using strict "
                "greater-than comparison"
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
            "threshold_comparison": THRESHOLD_COMPARISON,
            "hits": box_hits,
            "images": len(pairs),
            "rate": box_hits / len(pairs),
            "iou_mean": float(np.mean(box_ious)),
            "iou_median": float(np.median(box_ious)),
            "iou_max": float(np.max(box_ious)),
        },
    }


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

    overall = summarize_mvssnet_pair_slice(
        pairs,
        iterations=args.bootstrap_iterations,
        seed=args.bootstrap_seed,
    )
    by_domain = {
        domain: summarize_mvssnet_pair_slice(
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
        name: summarize_mvssnet_pair_slice(
            chunk,
            iterations=args.bootstrap_iterations,
            seed=args.bootstrap_seed + 100 + index,
        )
        for index, (name, chunk) in enumerate(_quintiles(pairs), start=1)
    }
    audit = audit_and_best_threshold(pairs, repo_root=repo_root)

    value = {
        "schema_version": "mvssnet_casia_posthoc_analysis_v1",
        "run_id": args.run_id,
        "created_at": utc_now(),
        "task_scope": {
            "primary_t1": PRIMARY_SCORE_SEMANTICS,
            "secondary_t1": OFFICIAL_PNG_SCORE_SEMANTICS,
            "primary_t2": T2_MAP_SEMANTICS,
            "threshold": CLASSIFICATION_THRESHOLD,
            "threshold_comparison": THRESHOLD_COMPARISON,
        },
        "sources": {
            "results_path": str(result_path.relative_to(repo_root)),
            "results_sha256": sha256_file(result_path),
            "run_manifest_path": str(
                run_manifest_path.relative_to(repo_root)
            ),
            "run_manifest_sha256": sha256_file(run_manifest_path),
            "summary_path": str(summary_path.relative_to(repo_root)),
            "summary_sha256": sha256_file(summary_path),
            "inputs_path": str(input_path.relative_to(repo_root)),
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
    parser.add_argument("--bootstrap-seed", type=int, default=20260723)
    return parser.parse_args()


def main() -> None:
    analyze(parse_args())


if __name__ == "__main__":
    main()
