#!/usr/bin/env python3
"""Audit and statistically analyze a completed paired CAT-Net v2 T2 run."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from eval.opensource.analyze_maskclip_run import (
    HISTOGRAM_BINS,
    histogram_best_metrics,
)
from eval.opensource.catnet_metrics import summarize_catnet_pair_slice
from eval.opensource.common import (
    atomic_write_json,
    read_jsonl,
    sha256_file,
    stable_json,
    utc_now,
)
from eval.opensource.run_catnet import (
    CHECKPOINT_EPOCH,
    CHECKPOINT_SHA256,
    MODEL_CONFIG_SHA256,
    MODEL_LICENSE_SHA256,
    MODEL_NETWORK_SHA256,
    MODEL_SOURCE_COMMIT,
)


DEFAULT_RUN_ID = "catnet_v2_mouse_canonical_v1_full275_20260723"
DEFAULT_RESULTS_DIR = Path("results/opensource/catnet")
DEFAULT_INPUTS = Path("outputs/opensource/mouse_canonical_v1/inputs.jsonl")
MASK_THRESHOLD = 0.5
MODEL_NAME = "CAT-Net v2"
MODEL_SLUG = "catnet_v2_ijcv2022"

_FORBIDDEN_RESULT_T1_FIELDS = frozenset(
    {
        "score",
        "score_margin",
        "decision",
        "classification_threshold",
        "class_probabilities",
        "detection",
    }
)
_FORBIDDEN_SUMMARY_T1_FIELDS = frozenset(
    {
        "detection",
        "score_by_kind",
        "paired_score_delta",
        "paired_ranking_accuracy",
        "classification",
    }
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
        return float(metrics["target_positive_pixels"]) / float(metrics["pixels"])


def _anchored(path: Path, repo_root: Path) -> Path:
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


def _require_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} is not a JSON object")
    return value


def _require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise ValueError(f"{label} mismatch: {actual!r} != {expected!r}")


def _verify_hash(path: Path, expected: Any, label: str) -> None:
    if not isinstance(expected, str):
        raise ValueError(f"{label} has no expected SHA-256")
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"{label} SHA-256 mismatch: {actual} != {expected}")


def _require_sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} is not a lowercase SHA-256 digest")
    return value


def _array_sha256_int32(array: np.ndarray) -> str:
    canonical = np.ascontiguousarray(array, dtype=np.int32)
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


def _jpeg_evidence_hashes(path: Path) -> dict[str, str]:
    try:
        import jpegio
    except ImportError as exc:
        raise RuntimeError(
            "CAT-Net artifact audit requires jpegio to independently verify "
            "the luminance quantization table and DCT coefficients"
        ) from exc

    jpeg = jpegio.read(str(path))
    if not jpeg.coef_arrays or not jpeg.comp_info:
        raise ValueError(f"JPEG has no luminance DCT component: {path}")
    qtable_index = int(jpeg.comp_info[0].quant_tbl_no)
    if qtable_index < 0 or qtable_index >= len(jpeg.quant_tables):
        raise ValueError(f"JPEG has invalid luminance qtable index: {path}")
    coefficients = np.asarray(jpeg.coef_arrays[0], dtype=np.int32)
    qtable = np.asarray(jpeg.quant_tables[qtable_index], dtype=np.int32)
    if qtable.shape != (8, 8):
        raise ValueError(f"JPEG has invalid luminance qtable shape: {path}")
    return {
        "qtable_sha256": _array_sha256_int32(qtable),
        "dct_y_sha256": _array_sha256_int32(coefficients),
    }


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


def _find_key(value: Any, forbidden: str, path: str = "$") -> str | None:
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}"
            if str(key).lower() == forbidden:
                return child_path
            found = _find_key(child, forbidden, child_path)
            if found is not None:
                return found
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found = _find_key(child, forbidden, f"{path}[{index}]")
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
    if "classification_threshold" in inference:
        raise ValueError(
            "CAT-Net v2 localization-only manifest contains T1 field "
            "classification_threshold"
        )
    for line_number, row in enumerate(result_rows, start=1):
        present = sorted(_FORBIDDEN_RESULT_T1_FIELDS.intersection(row))
        if present:
            raise ValueError(
                f"result row {line_number} contains forbidden T1 fields: {present}"
            )
        _require_equal(
            row.get("valid_for_t1"),
            False,
            f"result row {line_number} valid_for_t1",
        )
        for forbidden in ("auroc", "decision", "classification_threshold"):
            forbidden_path = _find_key(row, forbidden)
            if forbidden_path is not None:
                raise ValueError(
                    f"result row {line_number} contains forbidden T1 field "
                    f"{forbidden} at {forbidden_path}"
                )

    present_summary = sorted(_FORBIDDEN_SUMMARY_T1_FIELDS.intersection(summary))
    if present_summary:
        raise ValueError(
            f"CAT-Net v2 summary contains forbidden T1 fields: {present_summary}"
        )
    for forbidden in (
        "auroc",
        "decision",
        "classification_threshold",
        "average_precision",
    ):
        forbidden_path = _find_key(summary, forbidden)
        if forbidden_path is not None:
            raise ValueError(
                f"CAT-Net v2 summary contains forbidden T1 field "
                f"{forbidden} at {forbidden_path}"
            )


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
    manifest_inputs_path_value = manifest_input.get("inputs_manifest")
    if not isinstance(manifest_inputs_path_value, str):
        raise ValueError("manifest has no inputs_manifest path")
    _require_equal(
        _anchored(Path(manifest_inputs_path_value), repo_root),
        input_path.resolve(),
        "manifest/input JSONL path",
    )

    dataset_manifest_value = manifest_input.get("dataset_manifest")
    if not isinstance(dataset_manifest_value, str):
        raise ValueError("manifest has no dataset_manifest path")
    dataset_manifest_path = _anchored(Path(dataset_manifest_value), repo_root)
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
    for key in ("dataset_id", "contract_sha256", "inputs_sha256"):
        manifest_key = (
            "dataset_contract_sha256" if key == "contract_sha256" else key
        )
        _require_equal(
            release.get(key),
            manifest_input.get(manifest_key),
            f"canonical dataset {key}",
        )
    release_inputs_value = release.get("inputs_path")
    if not isinstance(release_inputs_value, str):
        raise ValueError("canonical dataset manifest has no inputs_path")
    _require_equal(
        _anchored(Path(release_inputs_value), repo_root),
        input_path.resolve(),
        "canonical dataset inputs path",
    )

    expected_ids = [str(row["sample_id"]) for row in input_rows]
    if len(expected_ids) != len(set(expected_ids)):
        raise ValueError("canonical inputs contain duplicate sample IDs")
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
    expected_pairs = len({int(row["pair_rank"]) for row in input_rows})
    _require_equal(
        manifest.get("expected_pairs"),
        expected_pairs,
        "manifest expected pair count",
    )

    model = _require_mapping(manifest.get("model"), "manifest model")
    checkpoint = _require_mapping(model.get("checkpoint"), "manifest checkpoint")
    configuration = _require_mapping(
        model.get("configuration"),
        "manifest model configuration",
    )
    license_value = _require_mapping(model.get("license"), "manifest license")
    for actual, expected, label in (
        (model.get("name"), MODEL_NAME, "manifest model name"),
        (model.get("model_slug"), MODEL_SLUG, "manifest model slug"),
        (
            model.get("source_commit"),
            MODEL_SOURCE_COMMIT,
            "manifest model source commit",
        ),
        (
            model.get("source_tracked_clean"),
            True,
            "manifest source clean flag",
        ),
        (
            model.get("positive_class_index"),
            1,
            "manifest positive class index",
        ),
        (
            configuration.get("sha256"),
            MODEL_CONFIG_SHA256,
            "manifest model configuration SHA-256",
        ),
        (
            configuration.get("network_sha256"),
            MODEL_NETWORK_SHA256,
            "manifest network definition SHA-256",
        ),
        (
            license_value.get("sha256"),
            MODEL_LICENSE_SHA256,
            "manifest component license SHA-256",
        ),
        (
            license_value.get("scope"),
            "hrnet_component_only",
            "manifest license scope",
        ),
        (
            license_value.get("project_wide_status"),
            "no_project_wide_license_found",
            "manifest project-wide license status",
        ),
        (
            checkpoint.get("sha256"),
            CHECKPOINT_SHA256,
            "manifest checkpoint SHA-256",
        ),
        (
            checkpoint.get("epoch"),
            CHECKPOINT_EPOCH,
            "manifest checkpoint epoch",
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

    inference = _require_mapping(manifest.get("inference"), "manifest inference")
    for actual, expected, label in (
        (inference.get("precision"), "float32", "manifest inference precision"),
        (inference.get("batch_size"), 1, "manifest inference batch size"),
        (inference.get("input_resize"), "none", "manifest input resize"),
        (
            inference.get("mask_threshold"),
            MASK_THRESHOLD,
            "manifest mask threshold",
        ),
        (
            inference.get("mask_threshold_comparison"),
            "greater_than_or_equal",
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
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "valid_for_t1": False,
        }
        for key, expected in expected_values.items():
            _require_equal(
                row.get(key),
                expected,
                f"result row {line_number} field {key}",
            )
        if row.get("status") not in {"ok", "error"}:
            raise ValueError(
                f"result row {line_number} has invalid status: "
                f"{row.get('status')!r}"
            )
        if row.get("status") == "ok":
            _require_equal(
                row.get("valid_for_t2"),
                True,
                f"result row {line_number} valid_for_t2",
            )
            _require_equal(
                row.get("mask_threshold"),
                MASK_THRESHOLD,
                f"result row {line_number} mask threshold",
            )
            _require_sha256(
                row.get("qtable_sha256"),
                f"result row {line_number} quantization table SHA-256",
            )
            _require_sha256(
                row.get("dct_y_sha256"),
                f"result row {line_number} luminance DCT SHA-256",
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
        (summary.get("model"), MODEL_NAME, "summary model name"),
        (summary.get("model_slug"), MODEL_SLUG, "summary model slug"),
        (
            summary.get("checkpoint_sha256"),
            CHECKPOINT_SHA256,
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

    localization_summary = _require_mapping(
        summary.get("localization"),
        "summary forged localization",
    )
    native_summary = _require_mapping(
        localization_summary.get("native"),
        "summary native localization",
    )
    micro_summary = _require_mapping(
        native_summary.get("micro_at_threshold"),
        "summary native localization threshold metrics",
    )
    _require_equal(
        micro_summary.get("threshold"),
        inference.get("mask_threshold"),
        "summary localization threshold",
    )
    _require_equal(
        summary.get("task_scope"),
        {
            "primary_task": "T2_localization",
            "mask_threshold": MASK_THRESHOLD,
        },
        "summary task scope",
    )
    _reject_t1_contract(
        manifest=manifest,
        summary=summary,
        result_rows=result_rows,
    )
    adapter_files_checked = _verify_adapter_contract(
        manifest.get("adapter_contract"),
        repo_root=repo_root,
    )

    return {
        "status": "ok",
        "run_manifest_fingerprint": fingerprint,
        "inputs_sha256": actual_inputs_sha256,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "physical_result_rows_validated": len(result_rows),
        "latest_result_rows_validated": len(latest),
        "adapter_contract_files_validated": adapter_files_checked,
        "checks": [
            "run manifest schema, ID, and recomputed immutable fingerprint",
            "canonical dataset manifest, input path, input hash, and ordered selection",
            (
                "pinned source, configuration, network, component-license "
                "scope, and checkpoint constants"
            ),
            "strict and weights-only checkpoint loading metadata",
            "every physical result row against canonical input and model provenance",
            "summary identity, fingerprint, T2 threshold, and latest-row coverage",
            "adapter contract file hashes",
            (
                "absence of T1 score, decision, detection, classification "
                "threshold, and AUROC fields"
            ),
        ],
    }


def _load_pairs(
    result_rows: list[dict[str, Any]],
    input_rows: list[dict[str, Any]],
) -> list[LocalizationPair]:
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
    pairs: list[LocalizationPair] = []
    for task_id, values in by_task.items():
        if set(values) != {"real", "forged"}:
            raise ValueError(f"incomplete pair for {task_id}: {sorted(values)}")
        real = values["real"]
        forged = values["forged"]
        input_row = inputs_by_id[str(forged["id"])]
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
                input_row=input_row,
            )
        )
    return sorted(pairs, key=lambda pair: int(pair.forged["pair_rank"]))


def _validate_metric_counts(
    result: dict[str, Any],
    *,
    target: np.ndarray,
    score_map: np.ndarray,
) -> None:
    localization = _require_mapping(
        result.get("localization"),
        f"localization for {result['id']}",
    )
    native = _require_mapping(
        localization.get("native"),
        f"native localization for {result['id']}",
    )
    prediction = score_map >= MASK_THRESHOLD
    truth = np.asarray(target, dtype=bool)
    counts = {
        "pixels": int(score_map.size),
        "target_positive_pixels": int(np.count_nonzero(truth)),
        "predicted_positive_pixels": int(np.count_nonzero(prediction)),
        "tp": int(np.count_nonzero(prediction & truth)),
        "fp": int(np.count_nonzero(prediction & ~truth)),
        "fn": int(np.count_nonzero(~prediction & truth)),
        "tn": int(np.count_nonzero(~prediction & ~truth)),
    }
    for key, expected in counts.items():
        _require_equal(
            native.get(key),
            expected,
            f"native localization {key} for {result['id']}",
        )
    _require_equal(
        native.get("threshold"),
        MASK_THRESHOLD,
        f"native localization threshold for {result['id']}",
    )
    positive_fraction = float(np.mean(prediction))
    score_mean = float(np.mean(score_map))
    score_max = float(np.max(score_map))
    for key, actual in (
        ("predicted_positive_fraction", positive_fraction),
        ("score_mean", score_mean),
        ("score_max", score_max),
    ):
        recorded = native.get(key)
        if not isinstance(recorded, (int, float)) or not math.isclose(
            float(recorded),
            actual,
            rel_tol=1e-6,
            abs_tol=1e-7,
        ):
            raise ValueError(
                f"native localization {key} mismatch for {result['id']}: "
                f"{recorded!r} != {actual!r}"
            )


def audit_and_best_threshold(
    pairs: list[LocalizationPair],
    *,
    repo_root: Path,
    bins: int,
) -> dict[str, Any]:
    per_image_best: list[dict[str, Any]] = []
    global_all = np.zeros(bins, dtype=np.int64)
    global_positive = np.zeros(bins, dtype=np.int64)
    box_ious: list[float] = []
    box_hits = 0
    checked_files = 0

    for pair in pairs:
        forged_target: np.ndarray | None = None
        for result in (pair.real, pair.forged):
            image_path = _anchored(Path(str(result["image_path"])), repo_root)
            logits_path = _anchored(
                Path(str(result["raw_logits_path"])),
                repo_root,
            )
            score_map_path = _anchored(
                Path(str(result["score_map_path"])),
                repo_root,
            )
            mask_path = _anchored(Path(str(result["mask_path"])), repo_root)
            for path, expected, label in (
                (image_path, result["image_sha256"], "canonical image"),
                (
                    logits_path,
                    result["raw_logits_sha256"],
                    "raw localization logits",
                ),
                (
                    score_map_path,
                    result["score_map_sha256"],
                    "native score map",
                ),
                (mask_path, result["mask_sha256"], "threshold mask"),
            ):
                _verify_hash(path, expected, f"{label} {result['id']}")
                checked_files += 1

            width, height = (int(value) for value in result["image_size"])
            with Image.open(image_path) as opened:
                _require_equal(
                    opened.size,
                    (width, height),
                    f"canonical image dimensions for {result['id']}",
                )
            jpeg_evidence = _jpeg_evidence_hashes(image_path)
            for key, expected in jpeg_evidence.items():
                _require_equal(
                    result.get(key),
                    expected,
                    f"{key} for {result['id']}",
                )

            padded_width = ((width + 7) // 8) * 8
            padded_height = ((height + 7) // 8) * 8
            expected_logits_shape = (
                2,
                padded_height // 4,
                padded_width // 4,
            )
            logits = np.load(logits_path, mmap_mode="r", allow_pickle=False)
            if logits.shape != expected_logits_shape:
                raise ValueError(
                    f"invalid raw logits shape for {result['id']}: "
                    f"{logits.shape} != {expected_logits_shape}"
                )
            _require_equal(
                result.get("raw_logits_shape"),
                list(expected_logits_shape),
                f"raw logits shape metadata for {result['id']}",
            )
            if logits.dtype != np.float32:
                raise ValueError(f"invalid raw logits dtype for {result['id']}")
            if not np.isfinite(logits).all():
                raise ValueError(f"non-finite raw logits for {result['id']}")

            score_map = np.load(
                score_map_path,
                mmap_mode="r",
                allow_pickle=False,
            )
            if score_map.shape != (height, width):
                raise ValueError(
                    f"invalid native score map shape for {result['id']}"
                )
            _require_equal(
                result.get("score_map_shape"),
                [height, width],
                f"score map shape metadata for {result['id']}",
            )
            if score_map.dtype != np.float32:
                raise ValueError(f"invalid native score map dtype for {result['id']}")
            if not np.isfinite(score_map).all():
                raise ValueError(f"non-finite native score map for {result['id']}")
            if float(score_map.min()) < 0.0 or float(score_map.max()) > 1.0:
                raise ValueError(
                    f"out-of-range native score map for {result['id']}"
                )

            with Image.open(mask_path) as opened:
                mask_array = np.asarray(opened.convert("L"), dtype=np.uint8)
            if mask_array.shape != (height, width):
                raise ValueError(f"invalid threshold mask shape for {result['id']}")
            _require_equal(
                result.get("mask_shape"),
                [height, width],
                f"mask shape metadata for {result['id']}",
            )
            mask_values = set(np.unique(mask_array).tolist())
            if not mask_values.issubset({0, 255}):
                raise ValueError(
                    f"threshold mask is not bit-exact binary for {result['id']}"
                )
            expected_mask = np.where(
                np.asarray(score_map) >= MASK_THRESHOLD,
                np.uint8(255),
                np.uint8(0),
            )
            if not np.array_equal(mask_array, expected_mask):
                raise ValueError(f"threshold mask mismatch for {result['id']}")

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
                        f"ground-truth mask shape mismatch for {pair.task_id}"
                    )
                forged_target = target_uint8 > 0
                target = forged_target
            else:
                target = np.zeros((height, width), dtype=bool)
            _validate_metric_counts(
                result,
                target=target,
                score_map=np.asarray(score_map),
            )

        if forged_target is None:
            raise ValueError(f"forged target was not loaded for {pair.task_id}")
        forged_score_path = _anchored(
            Path(str(pair.forged["score_map_path"])),
            repo_root,
        )
        forged_score = np.load(
            forged_score_path,
            mmap_mode="r",
            allow_pickle=False,
        )
        best, all_hist, positive_hist = histogram_best_metrics(
            forged_score,
            forged_target,
            bins=bins,
        )
        per_image_best.append({"task_id": pair.task_id, **best})
        global_all += all_hist
        global_positive += positive_hist

        with Image.open(
            _anchored(Path(str(pair.forged["mask_path"])), repo_root)
        ) as opened:
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

    global_tp = np.cumsum(global_positive[::-1], dtype=np.int64)[::-1]
    global_predicted = np.cumsum(global_all[::-1], dtype=np.int64)[::-1]
    global_fp = global_predicted - global_tp
    global_fn = int(np.sum(global_positive)) - global_tp
    global_denominator = 2 * global_tp + global_fp + global_fn
    global_f1 = np.divide(
        2.0 * global_tp,
        global_denominator,
        out=np.zeros_like(global_tp, dtype=np.float64),
        where=global_denominator > 0,
    )
    best_index = int(np.argmax(global_f1))
    best_f1_values = [float(row["f1"]) for row in per_image_best]
    best_iou_values = [float(row["iou"]) for row in per_image_best]
    return {
        "artifact_integrity": {
            "status": "ok",
            "checked_files": checked_files,
            "pairs": len(pairs),
            "result_images": len(pairs) * 2,
            "checks": [
                "the latest-row ID set exactly matches the canonical inputs",
                "every latest result has status=ok and valid_for_t1=false",
                (
                    "canonical image, raw logits, native score map, threshold "
                    "mask, and GT hashes"
                ),
                (
                    "independently recomputed JPEG luminance qtable and DCT "
                    "coefficient hashes"
                ),
                "raw float32 logits are finite with shape [2,H8/4,W8/4]",
                "native float32 score maps are finite and bounded by [0,1]",
                "saved uint8 threshold masks bit-exactly equal native score map >= 0.5",
                "recorded pixel counts agree with score maps, masks, and GT",
            ],
        },
        "localization_best_threshold": {
            "task_scope": "T2_pixel_localization_only",
            "approximation": (
                f"native score maps quantized into {bins} uniform bins over [0,1]"
            ),
            "per_image_oracle": {
                "images": len(per_image_best),
                "f1_mean": float(np.mean(best_f1_values)),
                "f1_median": float(np.median(best_f1_values)),
                "iou_mean": float(np.mean(best_iou_values)),
                "iou_median": float(np.median(best_iou_values)),
            },
            "single_global_oracle": {
                "threshold": best_index / (bins - 1),
                "micro_f1": float(global_f1[best_index]),
                "micro_iou": (
                    float(global_tp[best_index])
                    / float(
                        global_tp[best_index]
                        + global_fp[best_index]
                        + global_fn[best_index]
                    )
                ),
                "tp": int(global_tp[best_index]),
                "fp": int(global_fp[best_index]),
                "fn": int(global_fn[best_index]),
            },
        },
        "box_hit_at_mask_threshold_0_5": {
            "definition": (
                "IoU(predicted native binary mask, edit_region_xyxy) > 0.3"
            ),
            "hits": box_hits,
            "images": len(pairs),
            "rate": box_hits / len(pairs),
            "iou_mean": float(np.mean(box_ious)),
            "iou_median": float(np.median(box_ious)),
            "iou_max": float(np.max(box_ious)),
        },
    }


def _quintiles(
    pairs: list[LocalizationPair],
) -> list[tuple[str, list[LocalizationPair]]]:
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

    overall = summarize_catnet_pair_slice(
        pairs,
        iterations=args.bootstrap_iterations,
        seed=args.bootstrap_seed,
    )
    by_domain = {
        domain: summarize_catnet_pair_slice(
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
        name: summarize_catnet_pair_slice(
            chunk,
            iterations=args.bootstrap_iterations,
            seed=args.bootstrap_seed + 100 + index,
        )
        for index, (name, chunk) in enumerate(_quintiles(pairs), start=1)
    }
    audit = audit_and_best_threshold(
        pairs,
        repo_root=repo_root,
        bins=args.histogram_bins,
    )
    value = {
        "schema_version": "catnet_v2_posthoc_analysis_v1",
        "run_id": args.run_id,
        "created_at": utc_now(),
        "task_scope": {
            "primary_task": "T2_localization",
            "mask_threshold": MASK_THRESHOLD,
        },
        "sources": {
            "results_path": str(result_path.relative_to(repo_root)),
            "results_sha256": sha256_file(result_path),
            "run_manifest_path": str(run_manifest_path.relative_to(repo_root)),
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
            "metrics_scope": "T2 localization only",
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
    parser.add_argument("--histogram-bins", type=int, default=HISTOGRAM_BINS)
    return parser.parse_args()


def main() -> None:
    analyze(parse_args())


if __name__ == "__main__":
    main()
