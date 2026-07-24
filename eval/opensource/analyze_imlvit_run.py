#!/usr/bin/env python3
"""Independently audit and analyze a completed official IML-ViT T2 run."""

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
from typing import Any, Iterable

import numpy as np
from PIL import Image
from sklearn.metrics import average_precision_score

from eval.opensource.common import (
    atomic_write_json,
    read_jsonl,
    sha256_file,
    stable_json,
    utc_now,
)
from eval.opensource.imlvit_metrics import summarize_imlvit_pair_slice


DEFAULT_RUN_ID = "imlvit_cat_protocol_mouse_canonical_v1_full275_20260724"
DEFAULT_RESULTS_DIR = Path("results/opensource/imlvit")
DEFAULT_INPUTS = Path("outputs/opensource/mouse_canonical_v1/inputs.jsonl")
DEFAULT_IMLVIT_ROOT = Path("/root/.cache/claimforge/third_party/IML-ViT")

MODEL_NAME = "IML-ViT"
MODEL_SLUG = "imlvit_cat_protocol_2023_official"
MODEL_INPUT_SIZE = 1024
MASK_THRESHOLD = 0.5
THRESHOLD_OPERATOR = ">"
HISTOGRAM_BINS = 65_536

# These tolerances cover only independently observed CUDA-versus-CPU kernel
# rounding. Artifact hashes, dtypes, shapes, ranges, and masks remain exact.
SIGMOID_ABSOLUTE_TOLERANCE = 2e-7
NATIVE_RESTORE_ABSOLUTE_TOLERANCE = 2e-5
TRANSFORM_RELATIVE_TOLERANCE = 1e-6

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

    from eval.opensource import run_imlvit

    required = (
        "MODEL_REPO_URL",
        "MODEL_SOURCE_COMMIT",
        "SOURCE_FILES",
        "CHECKPOINT",
    )
    missing = [name for name in required if not hasattr(run_imlvit, name)]
    if missing:
        raise RuntimeError(
            "IML-ViT runner does not export required audit constants: "
            f"{missing}"
        )
    return SimpleNamespace(
        **{name: getattr(run_imlvit, name) for name in required}
    )


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
    expected_digest = _require_sha256(expected, f"{label} expected hash")
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")
    actual = sha256_file(path)
    if actual != expected_digest:
        raise ValueError(
            f"{label} SHA-256 mismatch: {actual} != {expected_digest}"
        )


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
    return hashlib.sha256(
        np.ascontiguousarray(array).tobytes(order="C")
    ).hexdigest()


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
    for index, raw in enumerate(ordered):
        item = _require_mapping(raw, f"ordered input {index}")
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
        "latest_status_counts": dict(sorted(latest_status_counts.items())),
        "duplicate_histories": duplicate_histories,
        "latest_policy": "last physical JSONL row for each sample id",
    }


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
            f"IML-ViT localization-only manifest contains T1 field at {semantic}"
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
                f"result row {line_number} contains semantic T1 field at "
                f"{semantic}"
            )

    present_summary = sorted(_FORBIDDEN_TOP_LEVEL_SUMMARY_FIELDS.intersection(summary))
    if present_summary:
        raise ValueError(
            f"IML-ViT summary contains forbidden T1 fields: {present_summary}"
        )
    semantic = _find_t1_semantic_key(summary)
    if semantic is not None:
        raise ValueError(
            f"IML-ViT summary contains semantic T1 field at {semantic}"
        )
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
            raise ValueError(f"{label} contains duplicate path {path}")
        result[path] = _require_sha256(
            item.get("sha256"),
            f"{label} entry {index} SHA-256",
        )
    return result


def _verify_adapter_contract(value: Any, *, repo_root: Path) -> int:
    if not isinstance(value, list) or not value:
        raise ValueError("manifest adapter_contract is empty or invalid")
    seen: set[Path] = set()
    for index, raw in enumerate(value):
        item = _require_mapping(raw, f"adapter contract entry {index}")
        path_value = item.get("path")
        if not isinstance(path_value, str):
            raise ValueError(f"adapter contract entry {index} has no path")
        path = _anchored(Path(path_value), repo_root)
        if path in seen:
            raise ValueError(f"adapter contract repeats path {path}")
        seen.add(path)
        _verify_hash(
            path,
            item.get("sha256"),
            f"adapter contract entry {index}",
        )
    return len(seen)


def validate_provenance(
    *,
    repo_root: Path,
    imlvit_root: Path,
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
    expected_checkpoint = dict(pins.CHECKPOINT)
    model = _require_mapping(manifest.get("model"), "manifest model")
    checkpoint = _require_mapping(
        model.get("checkpoint"),
        "manifest checkpoint",
    )
    license_value = _require_mapping(model.get("license"), "manifest license")
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
            "official_CAT_TruFor_protocol_checkpoint_20231104",
            "manifest model variant",
        ),
        (
            license_value.get("path"),
            "LICENSE",
            "manifest license path",
        ),
        (
            license_value.get("sha256"),
            expected_source_files["LICENSE"],
            "manifest license SHA-256",
        ),
        (license_value.get("spdx"), "MIT", "manifest license SPDX"),
        (
            license_value.get("scope"),
            "project_repository_code_only",
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
            model.get("supports_image_level_t1"),
            False,
            "manifest T1 support flag",
        ),
        (
            model.get("image_score_source"),
            None,
            "manifest image score source",
        ),
        (
            model.get("supports_pixel_level_t2"),
            True,
            "manifest T2 support flag",
        ),
        (
            model.get("primary_localization_output"),
            "sigmoid_of_bilinearly_upsampled_predict_head_logits",
            "manifest primary localization output",
        ),
    ):
        _require_equal(actual, expected, label)
    _require_equal(
        _normalise_source_files(model.get("source_files"), "source files"),
        expected_source_files,
        "manifest source-file pins",
    )
    for key, expected in expected_checkpoint.items():
        _require_equal(
            checkpoint.get(key),
            expected,
            f"manifest checkpoint {key}",
        )
    for key, expected in (
        ("strict_load", True),
        ("safe_weights_only_load", True),
        ("schema_fallbacks", False),
        ("prefix_rewrites", False),
        ("mae_initialization_reloaded", False),
    ):
        _require_equal(
            checkpoint.get(key),
            expected,
            f"manifest checkpoint {key}",
        )
    checkpoint_path_value = checkpoint.get("path")
    if not isinstance(checkpoint_path_value, str):
        raise ValueError("manifest checkpoint has no path")
    checkpoint_path = Path(checkpoint_path_value).resolve()
    _verify_hash(
        checkpoint_path,
        expected_checkpoint["sha256"],
        "official IML-ViT checkpoint",
    )
    _require_equal(
        checkpoint_path.stat().st_size,
        int(expected_checkpoint["bytes"]),
        "official checkpoint byte size",
    )

    if _git_value(imlvit_root, "rev-parse", "HEAD") != pins.MODEL_SOURCE_COMMIT:
        raise ValueError("checked IML-ViT source tree is not at the pinned commit")
    if _git_value(
        imlvit_root,
        "status",
        "--short",
        "--untracked-files=no",
    ):
        raise ValueError("checked IML-ViT source tree has tracked modifications")
    for relative, digest in expected_source_files.items():
        _verify_hash(
            imlvit_root / relative,
            digest,
            f"pinned IML-ViT source file {relative}",
        )

    inference = _require_mapping(manifest.get("inference"), "manifest inference")
    expected_inference = {
        "precision": "float32",
        "batch_size": 1,
        "deterministic": True,
        "input_source": "canonical_jpeg_original_bytes",
        "decoder": "Pillow.Image.open.convert_RGB",
        "channel_order": "RGB",
        "input_reencode": False,
        "normalization": {
            "scale": "uint8_divide_255",
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
        "raw_head_output": "one_channel_logits_at_256x256",
        "model_logit_restore": (
            "bilinear_to_1024x1024_align_corners_false"
        ),
        "model_probability": "single_sigmoid_after_logit_restore",
        "native_restore": (
            "crop_right_bottom_padding_then_bilinear_probability_to_"
            "native_align_corners_false"
        ),
        "mask_threshold": MASK_THRESHOLD,
        "mask_threshold_comparison": "strict_greater_than",
        "test_time_augmentation": False,
        "ensemble": False,
    }
    for key, expected in expected_inference.items():
        _require_equal(
            inference.get(key),
            expected,
            f"manifest inference {key}",
        )
    seed = inference.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("manifest inference seed is not an integer")
    geometry = _require_mapping(
        inference.get("input_geometry"),
        "manifest input geometry",
    )
    expected_geometry = {
        "protocol_reference": {
            "paper": "https://arxiv.org/abs/2307.14863",
            "version": "v4",
            "section": "4.1",
        },
        "paper_protocol": (
            "if max(H,W)>1024, resize longer side to 1024 while "
            "preserving aspect ratio; otherwise keep native size; "
            "top-left place and raw-zero pad right/bottom to 1024"
        ),
        "large_image_resize": (
            "albumentations.LongestMaxSize_max_size_1024_"
            "cv2_INTER_LINEAR_downscale_only_py3round"
        ),
        "small_image_resize": "none",
        "canvas": [MODEL_INPUT_SIZE, MODEL_INPUT_SIZE],
        "placement": "top_left",
        "padding_value_before_normalization": 0,
        "crop": None,
        "reason": (
            "paper section 4.1; uses the intended conditional "
            "LongestMaxSize semantics without copying the hosted "
            "Colab PIL width-height bug, and avoids the README "
            "demo's destructive top-left crop for large images"
        ),
    }
    _require_equal(geometry, expected_geometry, "manifest input geometry")

    metrics = _require_mapping(manifest.get("metrics"), "manifest metrics")
    expected_metrics = {
        "task": "T2_pixel_localization_only",
        "positive_class": "manipulated_pixel",
        "t1_policy": "unsupported_no_derived_image_score",
        "mask_threshold": MASK_THRESHOLD,
        "threshold_comparison": "strict_greater_than",
        "prediction_inversion": False,
        "localization_spaces": ["model_1024", "native"],
        "model_space_policy": (
            "metrics use only the valid resized-content rectangle; "
            "right/bottom padding is excluded"
        ),
        "model_space_gt_resize": "cv2_INTER_NEAREST",
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
        "raw_logits_model_1024": {
            "format": "npy",
            "dtype": "float32",
            "shape": [MODEL_INPUT_SIZE, MODEL_INPUT_SIZE],
        },
        "score_maps_model_1024": {
            "format": "npy",
            "dtype": "float32",
            "shape": [MODEL_INPUT_SIZE, MODEL_INPUT_SIZE],
        },
        "score_maps_native": {
            "format": "npy",
            "dtype": "float32",
            "shape": "native_HxW",
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
            "checkpoint_sha256": expected_checkpoint["sha256"],
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
            raise ValueError(
                f"result row {line_number} has invalid status {status!r}"
            )
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
            expected_checkpoint["sha256"],
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
    paired_coverage = _require_mapping(
        summary.get("paired_coverage"),
        "summary paired coverage",
    )
    expected_paired_coverage = {
        "complete_pairs": complete_pairs,
        "paired_images": complete_pairs * 2,
        "unpaired_valid_images": len(valid_latest) - complete_pairs * 2,
    }
    for key, expected in expected_paired_coverage.items():
        _require_equal(
            paired_coverage.get(key),
            expected,
            f"summary paired coverage {key}",
        )
    task_scope = _require_mapping(summary.get("task_scope"), "summary task scope")
    expected_scope = {
        "primary_task": "T2_localization",
        "valid_for_t1": False,
        "valid_for_t2": True,
        "primary_localization_space": "native",
        "auxiliary_localization_space": "model_1024",
        "localization_semantics": (
            "imlvit_sigmoid_manipulation_probability_float32"
        ),
        "probability_dtype": "float32",
        "mask_threshold": MASK_THRESHOLD,
        "threshold_operator": THRESHOLD_OPERATOR,
    }
    _require_equal(task_scope, expected_scope, "summary task scope")
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

    adapter_files_checked = _verify_adapter_contract(
        manifest.get("adapter_contract"),
        repo_root=repo_root,
    )
    return {
        "status": "ok",
        "run_manifest_fingerprint": fingerprint,
        "inputs_sha256": actual_inputs_sha256,
        "checkpoint_sha256": expected_checkpoint["sha256"],
        "physical_result_rows_validated": len(result_rows),
        "latest_result_rows_validated": len(latest),
        "pinned_source_files_validated": len(expected_source_files),
        "adapter_contract_files_validated": adapter_files_checked,
        "checks": [
            "run manifest schema, ID, and recomputed immutable fingerprint",
            "canonical release hash, input hash, and ordered selection",
            "official source commit, clean tree, source files, and checkpoint pins",
            "checkpoint file hash and byte size",
            "paper geometry, float32 inference, strict >0.5, and T2-only policy",
            "every physical result row against canonical identity",
            "summary identity, task scope, bootstrap seed, and latest-row coverage",
            "adapter contract file hashes",
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
        task_id = str(row["task_id"])
        kind = str(row["kind"])
        if kind in by_task.setdefault(task_id, {}):
            raise ValueError(f"duplicate {kind} result within pair {task_id}")
        by_task[task_id][kind] = row

    pairs: list[LocalizationPair] = []
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
) -> tuple[dict[str, Any], np.ndarray, tuple[int, int], tuple[int, int]]:
    """Independently decode and replay the registered image transformation."""

    import albumentations as albu
    import cv2

    with image_path.open("rb") as handle:
        with Image.open(handle) as opened:
            decoded = np.asarray(opened.convert("RGB"), dtype=np.uint8)
            decoder_format = opened.format
    if decoded.ndim != 3 or decoded.shape[2] != 3:
        raise ValueError(f"unexpected decoded image shape: {decoded.shape}")
    if decoded.dtype != np.uint8:
        raise ValueError(f"unexpected decoded image dtype: {decoded.dtype}")
    native_height, native_width = decoded.shape[:2]

    if max(native_height, native_width) > MODEL_INPUT_SIZE:
        transform = albu.LongestMaxSize(
            max_size=MODEL_INPUT_SIZE,
            interpolation=cv2.INTER_LINEAR,
            always_apply=True,
        )
        resized = np.asarray(transform(image=decoded)["image"], dtype=np.uint8)
        resize_policy = "albumentations_longest_max_size_downscale_only"
    else:
        resized = decoded
        resize_policy = "none_image_within_1024_limit"
    resized_height, resized_width = resized.shape[:2]
    if not (
        0 < resized_width <= MODEL_INPUT_SIZE
        and 0 < resized_height <= MODEL_INPUT_SIZE
    ):
        raise ValueError("independently resized content has invalid dimensions")

    canvas = np.zeros(
        (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE, 3),
        dtype=np.uint8,
    )
    canvas[:resized_height, :resized_width] = resized
    normalize = albu.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        max_pixel_value=255.0,
        always_apply=True,
    )
    normalized = np.asarray(normalize(image=canvas)["image"], dtype=np.float32)
    tensor = np.ascontiguousarray(normalized.transpose(2, 0, 1))
    evidence = {
        "decoder": "Pillow.Image.open.convert_RGB",
        "decoder_format": decoder_format,
        "channel_order": "RGB",
        "native_size": [native_width, native_height],
        "resized_content_size": [resized_width, resized_height],
        "model_canvas_size": [MODEL_INPUT_SIZE, MODEL_INPUT_SIZE],
        "resize_policy": resize_policy,
        "resize_interpolation": "cv2.INTER_LINEAR_via_albumentations",
        "resize_scale_x": resized_width / native_width,
        "resize_scale_y": resized_height / native_height,
        "aspect_ratio_preserved_with_rounding": True,
        "padding": {
            "placement": "top_left",
            "right_pixels": MODEL_INPUT_SIZE - resized_width,
            "bottom_pixels": MODEL_INPUT_SIZE - resized_height,
            "raw_rgb_value": 0,
            "applied_before_normalization": True,
        },
        "input_crop": None,
        "input_reencode": False,
        "normalization": {
            "scale": "uint8_divide_255",
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
        (resized_width, resized_height),
    )


def _sigmoid_float32(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float32)
    result = np.empty_like(values)
    positive = values >= np.float32(0.0)
    result[positive] = np.float32(1.0) / (
        np.float32(1.0) + np.exp(-values[positive])
    )
    exponentials = np.exp(values[~positive])
    result[~positive] = exponentials / (np.float32(1.0) + exponentials)
    return np.ascontiguousarray(result, dtype=np.float32)


def _bilinear_align_corners_false(
    score_map: np.ndarray,
    *,
    width: int,
    height: int,
) -> np.ndarray:
    """Pure NumPy half-pixel bilinear resize matching PyTorch's geometry."""

    source = np.asarray(score_map, dtype=np.float32)
    if source.ndim != 2 or source.size == 0:
        raise ValueError("source score map must be a non-empty 2D array")
    if width <= 0 or height <= 0:
        raise ValueError("output dimensions must be positive")
    source_height, source_width = source.shape
    if (height, width) == source.shape:
        return np.ascontiguousarray(source)

    x = (
        (np.arange(width, dtype=np.float32) + np.float32(0.5))
        * np.float32(source_width / width)
        - np.float32(0.5)
    )
    y = (
        (np.arange(height, dtype=np.float32) + np.float32(0.5))
        * np.float32(source_height / height)
        - np.float32(0.5)
    )
    x = np.maximum(x, np.float32(0.0))
    y = np.maximum(y, np.float32(0.0))
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    x1 = np.minimum(x0 + 1, source_width - 1)
    y1 = np.minimum(y0 + 1, source_height - 1)
    wx = (x - x0.astype(np.float32))[None, :]
    wy = (y - y0.astype(np.float32))[:, None]
    horizontal = (
        source[:, x0] * (np.float32(1.0) - wx)
        + source[:, x1] * wx
    )
    restored = (
        horizontal[y0, :] * (np.float32(1.0) - wy)
        + horizontal[y1, :] * wy
    )
    return np.ascontiguousarray(restored, dtype=np.float32)


def _nearest_resize_mask(
    target: np.ndarray,
    *,
    width: int,
    height: int,
) -> np.ndarray:
    source = np.asarray(target, dtype=bool)
    if source.ndim != 2 or source.size == 0:
        raise ValueError("source target must be a non-empty 2D array")
    if width <= 0 or height <= 0:
        raise ValueError("target dimensions must be positive")
    if source.shape == (height, width):
        return np.ascontiguousarray(source)
    source_height, source_width = source.shape
    y = np.floor(
        np.arange(height, dtype=np.float64) * source_height / height
    ).astype(np.int64)
    x = np.floor(
        np.arange(width, dtype=np.float64) * source_width / width
    ).astype(np.int64)
    return np.ascontiguousarray(source[y[:, None], x[None, :]])


def _load_float32_map(
    path: Path,
    *,
    expected_shape: tuple[int, int],
    label: str,
    probability: bool,
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
    if probability and (
        float(value.min()) < 0.0 or float(value.max()) > 1.0
    ):
        raise ValueError(f"out-of-range {label}")
    return np.asarray(value)


def _safe_div(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _mcc(tp: int, fp: int, fn: int, tn: int) -> float | None:
    denominator = math.sqrt(
        float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    )
    return (tp * tn - fp * fn) / denominator if denominator else None


def _binary_pixel_metrics_strict(
    score_map: np.ndarray,
    target: np.ndarray,
    *,
    include_ap: bool,
) -> dict[str, Any]:
    scores = np.asarray(score_map, dtype=np.float32)
    truth = np.asarray(target, dtype=bool)
    if scores.shape != truth.shape:
        raise ValueError(f"score/target mismatch: {scores.shape} != {truth.shape}")
    if scores.size == 0 or not np.isfinite(scores).all():
        raise ValueError("score map is empty or non-finite")
    if float(scores.min()) < 0.0 or float(scores.max()) > 1.0:
        raise ValueError("score map falls outside [0, 1]")
    prediction = scores > MASK_THRESHOLD
    tp = int(np.count_nonzero(prediction & truth))
    fp = int(np.count_nonzero(prediction & ~truth))
    fn = int(np.count_nonzero(~prediction & truth))
    tn = int(np.count_nonzero(~prediction & ~truth))
    pixel_ap: float | None = None
    if include_ap and truth.any() and (~truth).any():
        pixel_ap = float(
            average_precision_score(
                truth.reshape(-1),
                scores.reshape(-1),
            )
        )
    return {
        "threshold": MASK_THRESHOLD,
        "threshold_operator": THRESHOLD_OPERATOR,
        "probability_dtype": "float32",
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
        "mcc": _mcc(tp, fp, fn, tn),
        "pixel_ap": pixel_ap,
        "score_mean": float(np.mean(scores)),
        "score_max": float(np.max(scores)),
    }


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

    target_value = input_row.get("gt_mask_path")
    if not isinstance(target_value, str):
        raise ValueError(f"forged input has no GT mask: {result['id']}")
    target_path = _anchored(Path(target_value), repo_root)
    _verify_hash(
        target_path,
        input_row.get("gt_mask_sha256"),
        f"ground-truth mask {result['id']}",
    )
    checked_paths.add(target_path)
    with Image.open(target_path) as opened:
        target = np.asarray(opened.convert("L"), dtype=np.uint8) > 0
    if target.shape != (height, width):
        raise ValueError(
            f"ground-truth shape mismatch for {result['id']}: "
            f"{target.shape} != {(height, width)}"
        )
    if not target.any():
        raise ValueError(f"forged ground truth is empty for {result['id']}")
    return target


def _oracle_histograms(
    score_map: np.ndarray,
    target: np.ndarray,
    *,
    bins: int,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    if isinstance(bins, bool) or not isinstance(bins, int) or bins < 2:
        raise ValueError("histogram bins must be an integer >= 2")
    scores = np.asarray(score_map, dtype=np.float32)
    truth = np.asarray(target, dtype=bool)
    if scores.shape != truth.shape or not truth.any():
        raise ValueError("oracle diagnostic requires an aligned non-empty target")
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
        greater_all[:-1] = np.cumsum(
            all_hist[:0:-1],
            dtype=np.int64,
        )[::-1]
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


def _descriptive(values: Iterable[float]) -> dict[str, float | int | None]:
    data = np.asarray(list(values), dtype=np.float64)
    if data.size == 0:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "min": None,
            "max": None,
        }
    return {
        "count": int(data.size),
        "mean": float(np.mean(data)),
        "median": float(np.median(data)),
        "min": float(np.min(data)),
        "max": float(np.max(data)),
    }


def audit_artifacts(
    pairs: list[LocalizationPair],
    *,
    repo_root: Path,
    histogram_bins: int | None,
) -> dict[str, Any]:
    if not pairs:
        raise ValueError("artifact audit requires at least one pair")
    global_all = (
        np.zeros(histogram_bins, dtype=np.int64)
        if histogram_bins is not None
        else None
    )
    global_positive = (
        np.zeros(histogram_bins, dtype=np.int64)
        if histogram_bins is not None
        else None
    )
    per_image_best: list[dict[str, Any]] = []
    checked_paths: set[Path] = set()
    artifact_owners: dict[Path, str] = {}
    box_iou_values: list[float] = []
    box_coverage_values: list[float] = []
    prediction_inside_values: list[float] = []
    box_any_overlap = 0
    box_iou_hits = 0

    for pair in pairs:
        forged_target: np.ndarray | None = None
        forged_native: np.ndarray | None = None
        forged_prediction: np.ndarray | None = None
        for result in (pair.real, pair.forged):
            result_id = str(result["id"])
            # The result identity was already checked against the canonical
            # row. The forged Pair stores the only input fields needed for GT.
            input_row = (
                pair.input_row
                if result["kind"] == "forged"
                else {
                    "gt_mask_kind": "all_zero",
                    "gt_mask_path": None,
                }
            )

            image_path = _anchored(Path(str(result["image_path"])), repo_root)
            _verify_hash(
                image_path,
                result.get("image_sha256"),
                f"canonical image {result_id}",
            )
            checked_paths.add(image_path)
            (
                evidence,
                tensor,
                native_size,
                resized_size,
            ) = _preprocess_evidence(image_path)
            width, height = native_size
            resized_width, resized_height = resized_size
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
            for key, expected in evidence.items():
                _require_equal(
                    preprocess.get(key),
                    expected,
                    f"preprocess {key} for {result_id}",
                )
            _require_equal(
                result.get("model_valid_content_size"),
                [resized_width, resized_height],
                f"valid model content size for {result_id}",
            )

            artifact_specs = (
                (
                    "raw_logits_model_path",
                    "raw_logits_model_sha256",
                    "raw logits",
                ),
                (
                    "score_map_model_path",
                    "score_map_model_sha256",
                    "model score map",
                ),
                ("score_map_path", "score_map_sha256", "native score map"),
                ("mask_path", "mask_sha256", "native threshold mask"),
            )
            resolved: dict[str, Path] = {}
            for path_key, hash_key, label in artifact_specs:
                path_value = result.get(path_key)
                if not isinstance(path_value, str):
                    raise ValueError(f"{label} for {result_id} has no path")
                path = _anchored(Path(path_value), repo_root)
                previous = artifact_owners.get(path)
                if previous is not None and previous != result_id:
                    raise ValueError(
                        f"artifact path {path} is shared by {previous} and "
                        f"{result_id}"
                    )
                artifact_owners[path] = result_id
                _verify_hash(path, result.get(hash_key), f"{label} {result_id}")
                checked_paths.add(path)
                resolved[path_key] = path

            model_shape = (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE)
            raw_logits = _load_float32_map(
                resolved["raw_logits_model_path"],
                expected_shape=model_shape,
                label=f"raw model logits for {result_id}",
                probability=False,
            )
            model_score = _load_float32_map(
                resolved["score_map_model_path"],
                expected_shape=model_shape,
                label=f"model score map for {result_id}",
                probability=True,
            )
            native_score = _load_float32_map(
                resolved["score_map_path"],
                expected_shape=(height, width),
                label=f"native score map for {result_id}",
                probability=True,
            )
            for key, expected in (
                ("raw_logits_model_shape", list(model_shape)),
                ("raw_logits_model_dtype", "float32"),
                ("score_map_model_shape", list(model_shape)),
                ("score_map_model_dtype", "float32"),
                ("score_map_shape", [height, width]),
                ("score_map_dtype", "float32"),
            ):
                _require_equal(
                    result.get(key),
                    expected,
                    f"{key} metadata for {result_id}",
                )

            expected_model_score = _sigmoid_float32(raw_logits)
            if not np.allclose(
                model_score,
                expected_model_score,
                rtol=TRANSFORM_RELATIVE_TOLERANCE,
                atol=SIGMOID_ABSOLUTE_TOLERANCE,
            ):
                maximum = float(np.max(np.abs(model_score - expected_model_score)))
                raise ValueError(
                    "model score map is not sigmoid(raw model logits) for "
                    f"{result_id}; max_abs={maximum}"
                )
            expected_native = _bilinear_align_corners_false(
                model_score[:resized_height, :resized_width],
                width=width,
                height=height,
            )
            if not np.allclose(
                native_score,
                expected_native,
                rtol=TRANSFORM_RELATIVE_TOLERANCE,
                atol=NATIVE_RESTORE_ABSOLUTE_TOLERANCE,
            ):
                maximum = float(np.max(np.abs(native_score - expected_native)))
                raise ValueError(
                    "native score map is not the align_corners=False restore "
                    f"of valid model content for {result_id}; max_abs={maximum}"
                )

            mask_path = resolved["mask_path"]
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
                native_score > MASK_THRESHOLD,
                np.uint8(255),
                np.uint8(0),
            )
            if not np.array_equal(mask_array, expected_mask):
                raise ValueError(
                    f"strict >0.5 threshold mask mismatch for {result_id}"
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
                width=resized_width,
                height=resized_height,
            )
            _validate_metrics(
                result,
                score_map=model_score[:resized_height, :resized_width],
                target=model_target,
                space="model_1024",
            )
            _validate_metrics(
                result,
                score_map=native_score,
                target=target,
                space="native",
            )
            if result["kind"] == "forged":
                forged_target = target
                forged_native = native_score
                forged_prediction = mask_array > 0

        if (
            forged_target is None
            or forged_native is None
            or forged_prediction is None
        ):
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

        x1, y1, x2, y2 = (
            int(value) for value in pair.input_row["edit_region_xyxy"]
        )
        if not (
            0 <= x1 < x2 <= forged_prediction.shape[1]
            and 0 <= y1 < y2 <= forged_prediction.shape[0]
        ):
            raise ValueError(f"invalid edit box for {pair.task_id}")
        box_area = (x2 - x1) * (y2 - y1)
        intersection = int(np.count_nonzero(forged_prediction[y1:y2, x1:x2]))
        predicted_area = int(np.count_nonzero(forged_prediction))
        union = predicted_area + box_area - intersection
        box_iou = intersection / union if union else 0.0
        box_coverage = intersection / box_area
        prediction_inside = (
            intersection / predicted_area if predicted_area else 0.0
        )
        box_iou_values.append(box_iou)
        box_coverage_values.append(box_coverage)
        prediction_inside_values.append(prediction_inside)
        box_any_overlap += int(intersection > 0)
        box_iou_hits += int(box_iou > 0.3)

    diagnostic: dict[str, Any] | None
    if histogram_bins is None:
        diagnostic = None
    else:
        assert global_all is not None
        assert global_positive is not None
        global_best = _best_from_histograms(global_all, global_positive)
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
            "single_global_test_set_oracle": global_best,
        }

    return {
        "artifact_integrity": {
            "status": "ok",
            "checked_files": len(checked_paths),
            "pairs": len(pairs),
            "result_images": len(pairs) * 2,
            "numeric_tolerances": {
                "sigmoid_absolute": SIGMOID_ABSOLUTE_TOLERANCE,
                "native_restore_absolute": NATIVE_RESTORE_ABSOLUTE_TOLERANCE,
                "relative": TRANSFORM_RELATIVE_TOLERANCE,
                "reason": (
                    "bounded float32 CUDA-versus-independent-CPU kernel "
                    "rounding only"
                ),
            },
            "checks": [
                "canonical images, GT masks, and all artifact file hashes",
                "independent RGB decode, conditional resize, padding, normalization, and tensor hash",
                "raw model logits are finite float32 [1024,1024]",
                "model probability is sigmoid(raw logits) within bounded float32 tolerance",
                "model metrics exclude right/bottom padding and use nearest-resized GT",
                "native probability is valid-content bilinear align_corners=False restoration",
                "native uint8 mask bit-exactly equals probability > 0.5",
                "native and valid-content localization metrics are independently recomputed",
            ],
        },
        "localization_threshold_diagnostic": diagnostic,
        "box_hit_at_native_mask_threshold_0_5": {
            "task_scope": "T2_pixel_localization_only",
            "mask_threshold": MASK_THRESHOLD,
            "threshold_operator": THRESHOLD_OPERATOR,
            "box_definition": "canonical edit_region_xyxy half-open rectangle",
            "any_overlap": {
                "hits": box_any_overlap,
                "images": len(pairs),
                "rate": box_any_overlap / len(pairs),
            },
            "iou_greater_than_0_3": {
                "hits": box_iou_hits,
                "images": len(pairs),
                "rate": box_iou_hits / len(pairs),
            },
            "box_iou": _descriptive(box_iou_values),
            "box_pixel_coverage": _descriptive(box_coverage_values),
            "predicted_pixels_inside_box_fraction": _descriptive(
                prediction_inside_values
            ),
        },
    }


def _quintiles(
    pairs: list[LocalizationPair],
) -> list[tuple[str, list[LocalizationPair]]]:
    if not pairs:
        return []
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


def _summarize_spaces(
    pairs: list[LocalizationPair],
    *,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    return {
        space: summarize_imlvit_pair_slice(
            pairs,
            iterations=iterations,
            seed=seed,
            localization_space=space,
        )
        for space in ("native", "model_1024")
    }


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    if args.bootstrap_iterations <= 0:
        raise ValueError("bootstrap iterations must be positive")
    if not args.skip_threshold_diagnostic and args.histogram_bins < 2:
        raise ValueError("histogram bins must be at least two")

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
    imlvit_root = args.imlvit_root.resolve()
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
    history = summarize_result_history(result_rows)
    provenance = validate_provenance(
        repo_root=repo_root,
        imlvit_root=imlvit_root,
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
    by_domain = {
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
    by_edit_quintile = {
        name: _summarize_spaces(
            chunk,
            iterations=args.bootstrap_iterations,
            seed=args.bootstrap_seed + 100 + index,
        )
        for index, (name, chunk) in enumerate(_quintiles(pairs), start=1)
    }
    audit = audit_artifacts(
        pairs,
        repo_root=repo_root,
        histogram_bins=(
            None if args.skip_threshold_diagnostic else args.histogram_bins
        ),
    )
    value = {
        "schema_version": "imlvit_posthoc_analysis_v1",
        "run_id": args.run_id,
        "created_at": utc_now(),
        "task_scope": {
            "primary_task": "T2_localization",
            "valid_for_t1": False,
            "valid_for_t2": True,
            "primary_localization_space": "native",
            "auxiliary_localization_space": "model_1024",
            "auxiliary_evaluation_region": (
                "valid_resized_content_only_excluding_right_bottom_padding"
            ),
            "mask_threshold": MASK_THRESHOLD,
            "threshold_operator": THRESHOLD_OPERATOR,
            "t1_policy": "unsupported_no_derived_image_score",
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
            "spaces": ["native", "model_1024"],
            "model_1024_region": (
                "valid_resized_content_only_excluding_right_bottom_padding"
            ),
            "metrics_scope": "T2 localization only",
        },
        "overall": overall,
        "by_domain": by_domain,
        "by_edit_fraction_quintile": by_edit_quintile,
        "provenance_integrity": provenance,
        "result_history": history,
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
    parser.add_argument("--imlvit-root", type=Path, default=DEFAULT_IMLVIT_ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260724)
    parser.add_argument("--histogram-bins", type=int, default=HISTOGRAM_BINS)
    parser.add_argument("--skip-threshold-diagnostic", action="store_true")
    return parser.parse_args()


def main() -> None:
    analyze(parse_args())


if __name__ == "__main__":
    main()
