#!/usr/bin/env python3
"""Independently audit and analyze a completed official HiFi-IFDL run."""

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
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

from eval.opensource.common import (
    atomic_write_json,
    read_jsonl,
    sha256_file,
    stable_json,
    utc_now,
)
from eval.opensource.hifi_ifdl_metrics import (
    summarize_hifi_ifdl_pair_slice,
)


DEFAULT_RUN_ID = "hifi_ifdl_general_750001_mouse_canonical_v1_full275_20260724"
DEFAULT_RESULTS_DIR = Path("results/opensource/hifi_ifdl")
DEFAULT_INPUTS = Path("outputs/opensource/mouse_canonical_v1/inputs.jsonl")
DEFAULT_HIFI_IFDL_ROOT = Path("/root/.cache/claimforge/third_party/HiFi_IFDL")

MODEL_NAME = "HiFi-Net"
MODEL_SLUG = "hifi_ifdl_general_750001_official"
MODEL_INPUT_SIZE = 256
EMBEDDING_CHANNELS = 18
PAIRWISE_DISTANCE_EPSILON = 1e-6
MASK_THRESHOLD = 2.3
MASK_THRESHOLD_OPERATOR = ">="
CLASSIFICATION_THRESHOLD = 0.5
CLASSIFICATION_THRESHOLD_OPERATOR = ">"
FINE_CLASS_COUNT = 14
CLASSIFICATION_LEVEL_SIZES = (3, 5, 7, 14)
CLASSIFICATION_LEVEL_NAMES = (
    "out0_coarse_3class",
    "out1_5class",
    "out2_7class",
    "out3_fine_14class",
)
FINE_CLASS_NAMES = (
    "authentic",
    "splice",
    "inpainting",
    "copy_move",
    "faceshifter",
    "stgan",
    "star2",
    "hisd",
    "stylegan2",
    "stylegan3",
    "ddpm",
    "ddim",
    "d_latent",
    "glide",
)
HISTOGRAM_BINS = 65_536

# CUDA reductions and interpolation may differ by a few last-place bits from
# this NumPy replay. Artifact hashes, shapes, dtypes, masks, and decisions are
# still checked exactly.
DISTANCE_ABSOLUTE_TOLERANCE = 2e-5
NATIVE_RESTORE_ABSOLUTE_TOLERANCE = 3e-5
PROBABILITY_ABSOLUTE_TOLERANCE = 2e-7


@dataclass(frozen=True)
class Pair:
    task_id: str
    domain: str
    real: dict[str, Any]
    forged: dict[str, Any]
    input_row: dict[str, Any]

    @property
    def edit_fraction(self) -> float:
        metrics = _require_mapping(
            _require_mapping(
                self.forged.get("localization"),
                f"forged localization for {self.task_id}",
            ).get("native"),
            f"forged native localization for {self.task_id}",
        )
        pixels = int(metrics["pixels"])
        if pixels <= 0:
            raise ValueError(f"pair {self.task_id} has no native pixels")
        return float(metrics["target_positive_pixels"]) / float(pixels)


def _load_runner_pins() -> SimpleNamespace:
    """Load immutable runner pins without importing the upstream model."""

    from eval.opensource import run_hifi_ifdl

    required = (
        "MODEL_REPO_URL",
        "MODEL_SOURCE_COMMIT",
        "SOURCE_FILES",
        "INITIALIZATION_WEIGHT",
        "CHECKPOINT_RELEASE",
        "CHECKPOINT_BUNDLE_SHA256",
        "FINE_CLASS_NAMES",
    )
    missing = [name for name in required if not hasattr(run_hifi_ifdl, name)]
    if missing:
        raise RuntimeError(
            "HiFi-IFDL runner does not export required audit constants: "
            f"{missing}"
        )
    values = {name: getattr(run_hifi_ifdl, name) for name in required}
    for alternatives, canonical in (
        (("CHECKPOINTS", "CHECKPOINT"), "CHECKPOINTS"),
        (
            ("CENTER_RADIUS", "CENTER", "CENTER_FILE", "RADIUS_CENTER"),
            "CENTER",
        ),
    ):
        present = [
            name for name in alternatives if hasattr(run_hifi_ifdl, name)
        ]
        if not present:
            raise RuntimeError(
                "HiFi-IFDL runner does not export a required audit pin: "
                f"one of {alternatives}"
            )
        values[canonical] = getattr(run_hifi_ifdl, present[0])
    return SimpleNamespace(**values)


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
        try:
            selected.append(inputs_by_id[sample_id])
        except KeyError as exc:
            raise ValueError(
                f"run manifest selected unknown canonical ID {sample_id}"
            ) from exc
        seen.add(sample_id)
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
    retried = {
        row_id: {
            "physical_rows": len(history),
            "line_numbers": [line_number for line_number, _ in history],
            "statuses": [str(row.get("status")) for _, row in history],
        }
        for row_id, history in sorted(histories.items())
        if len(history) > 1
    }
    return {
        "physical_rows": len(result_rows),
        "unique_ids": len(histories),
        "duplicate_physical_rows": len(result_rows) - len(histories),
        "physical_status_counts": dict(sorted(status_counts.items())),
        "retried_ids": retried,
    }


def _normalise_source_files(value: Any, label: str) -> dict[str, str]:
    if isinstance(value, dict):
        result = {str(path): str(digest) for path, digest in value.items()}
    elif isinstance(value, list):
        result = {}
        for index, raw in enumerate(value):
            item = _require_mapping(raw, f"{label} item {index}")
            path = item.get("path")
            digest = item.get("sha256")
            if not isinstance(path, str) or not path:
                raise ValueError(f"{label} item {index} has no path")
            if path in result:
                raise ValueError(f"{label} contains duplicate path {path}")
            result[path] = _require_sha256(
                digest,
                f"{label} item {index} SHA-256",
            )
    else:
        raise ValueError(f"{label} is not a file-pin mapping or list")
    for path, digest in result.items():
        if not path:
            raise ValueError(f"{label} contains an empty path")
        _require_sha256(digest, f"{label} {path} SHA-256")
    return result


def _normalise_components(value: Any, label: str) -> dict[str, dict[str, Any]]:
    if isinstance(value, dict):
        return {
            str(role): _require_mapping(contract, f"{label} {role}")
            for role, contract in value.items()
        }
    if not isinstance(value, list):
        raise ValueError(f"{label} is not a component list or mapping")
    result: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(value):
        item = _require_mapping(raw, f"{label} item {index}")
        role = item.get("role")
        if not isinstance(role, str) or not role:
            raise ValueError(f"{label} item {index} has no role")
        if role in result:
            raise ValueError(f"{label} contains duplicate role {role}")
        result[role] = {
            key: value
            for key, value in item.items()
            if key != "role"
        }
    return result


def _verify_adapter_contract(value: Any, *, repo_root: Path) -> int:
    if not isinstance(value, list) or not value:
        raise ValueError("manifest adapter contract is empty or invalid")
    seen: set[Path] = set()
    for index, raw in enumerate(value):
        item = _require_mapping(raw, f"adapter contract item {index}")
        path_value = item.get("path")
        if not isinstance(path_value, str) or not path_value:
            raise ValueError(f"adapter contract item {index} has no path")
        path = _anchored(Path(path_value), repo_root)
        if path in seen:
            raise ValueError(f"duplicate adapter contract path: {path}")
        seen.add(path)
        _verify_hash(
            path,
            item.get("sha256"),
            f"adapter contract file {path_value}",
        )
    return len(seen)


def validate_provenance(
    *,
    repo_root: Path,
    hifi_ifdl_root: Path,
    run_id: str,
    input_path: Path,
    input_rows: list[dict[str, Any]],
    result_rows: list[dict[str, Any]],
    manifest: dict[str, Any],
    summary: dict[str, Any],
) -> tuple[dict[str, Any], np.ndarray, float]:
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
    _require_equal(
        manifest_input.get("encoding"),
        release.get("jpeg"),
        "canonical dataset encoding contract",
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

    expected_source_files = _normalise_source_files(
        pins.SOURCE_FILES,
        "runner source pins",
    )
    expected_initialization = dict(pins.INITIALIZATION_WEIGHT)
    expected_checkpoints = _normalise_components(
        pins.CHECKPOINTS,
        "runner checkpoint pins",
    )
    expected_center = dict(pins.CENTER)
    model = _require_mapping(manifest.get("model"), "manifest model")
    license_value = _require_mapping(model.get("license"), "manifest license")
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
            "official_HiFi_IFDL_general_checkpoint_750001",
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
            "project_repository_code_only",
            "manifest license scope",
        ),
        (
            license_value.get("checkpoint_license"),
            "not_separately_stated_by_release",
            "manifest checkpoint license statement",
        ),
        (
            model.get("initialization_weight"),
            expected_initialization,
            "manifest initialization-weight pin",
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
            model.get("fine_class_names"),
            list(pins.FINE_CLASS_NAMES),
            "manifest fine class names",
        ),
        (
            model.get("fine_authentic_class_index"),
            0,
            "manifest authentic class index",
        ),
        (
            model.get("hierarchy_output_class_counts"),
            list(CLASSIFICATION_LEVEL_SIZES),
            "manifest hierarchy dimensions",
        ),
        (
            model.get("supports_image_level_t1"),
            True,
            "manifest T1 support flag",
        ),
        (
            model.get("image_score_source"),
            "native_out3_fine_14class_head",
            "manifest T1 score source",
        ),
        (
            model.get("supports_pixel_level_t2"),
            True,
            "manifest T2 support flag",
        ),
        (
            model.get("primary_localization_output"),
            (
                "euclidean_distance_from_18d_pixel_embedding_to_"
                "released_authentic_center"
            ),
            "manifest primary localization output",
        ),
    ):
        _require_equal(actual, expected, label)
    _require_equal(
        _normalise_source_files(model.get("source_files"), "source files"),
        expected_source_files,
        "manifest source-file pins",
    )
    for key, expected in dict(pins.CHECKPOINT_RELEASE).items():
        _require_equal(
            checkpoint.get(key),
            expected,
            f"manifest checkpoint release {key}",
        )
    for key, expected in (
        ("bundle_sha256", pins.CHECKPOINT_BUNDLE_SHA256),
        ("strict_load", True),
        ("safe_weights_only_load", True),
        ("container_selection", "top_level_model_only"),
        ("prefix_rewrites", False),
        ("schema_fallbacks", False),
    ):
        _require_equal(
            checkpoint.get(key),
            expected,
            f"manifest checkpoint {key}",
        )
    manifest_components = _normalise_components(
        checkpoint.get("components"),
        "manifest checkpoint components",
    )
    _require_equal(
        set(manifest_components),
        set(expected_checkpoints),
        "manifest checkpoint component roles",
    )
    checkpoint_paths: dict[str, Path] = {}
    for role, expected in expected_checkpoints.items():
        actual = manifest_components[role]
        for key, expected_value in expected.items():
            _require_equal(
                actual.get(key),
                expected_value,
                f"manifest checkpoint {role} {key}",
            )
        path_value = actual.get("path")
        if not isinstance(path_value, str) or not path_value:
            raise ValueError(f"manifest checkpoint {role} has no runtime path")
        checkpoint_paths[role] = Path(path_value).resolve()

    manifest_center = _require_mapping(
        checkpoint.get("center_radius"),
        "manifest center/radius",
    )
    for key, expected in expected_center.items():
        _require_equal(
            manifest_center.get(key),
            expected,
            f"manifest center/radius {key}",
        )
    for key, expected in (
        ("loaded_center", True),
        ("loaded_radius_for_provenance_validation", True),
    ):
        _require_equal(
            manifest_center.get(key),
            expected,
            f"manifest center/radius {key}",
        )
    runtime_center = manifest_center.get("runtime_path")
    if not isinstance(runtime_center, str):
        raise ValueError("manifest center/radius has no runtime path")
    _require_equal(
        Path(runtime_center).resolve(),
        (hifi_ifdl_root / str(expected_center["path"])).resolve(),
        "manifest center/radius runtime path",
    )
    center, radius, runtime_audit = _verify_upstream_runtime(
        hifi_ifdl_root=hifi_ifdl_root,
        pins=pins,
        checkpoint_paths=checkpoint_paths,
    )

    inference = _require_mapping(manifest.get("inference"), "manifest inference")
    _require_equal(
        inference.get("compatibility_shims"),
        [
            {
                "shim": "temporary_numpy.int=builtin_int",
                "scope": "HRNet_constructor_only",
                "numerical_effect": "none",
            },
            {
                "shim": "temporary_torch.Tensor.cuda_identity",
                "scope": (
                    "NLCDetection_constructor_two_unused_unregistered_"
                    "split_tensors_only"
                ),
                "numerical_effect": "none_in_forward",
            },
        ],
        "manifest compatibility shims",
    )
    for key, expected in {
        "precision": "float32",
        "batch_size": 1,
        "deterministic": True,
        "input_source": "canonical_jpeg_original_bytes",
        "decoder": "imageio.v2.imread",
        "channel_order": "RGB",
        "input_geometry": (
            "direct_stretch_to_256x256_without_aspect_ratio_preservation"
        ),
        "resize_interpolation": "Pillow.Image.Resampling.BICUBIC",
        "input_crop": None,
        "input_reencode": False,
        "normalization": "uint8_rgb_divide_255_float32",
        "feature_output_shapes": [
            [18, 256, 256],
            [36, 128, 128],
            [72, 64, 64],
            [144, 32, 32],
        ],
        "test_time_augmentation": False,
        "ensemble": False,
        "forward_passes_per_image": 1,
    }.items():
        _require_equal(inference.get(key), expected, f"manifest inference {key}")
    seed = inference.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("manifest inference seed is not an integer")
    _require_equal(
        inference.get("classification"),
        {
            "continuous_score": (
                "1_minus_softmax_fine_14class_probability_index_0"
            ),
            "score_source": "native_out3_fine_14class_head",
            "benchmark_threshold": CLASSIFICATION_THRESHOLD,
            "benchmark_threshold_comparison": "strict_greater_than",
            "official_decision": (
                "argmax_fine_14class_index_not_equal_to_0"
            ),
            "both_decisions_stored_separately": True,
        },
        "manifest classification inference",
    )
    localization_inference = _require_mapping(
        inference.get("localization"),
        "manifest localization inference",
    )
    expected_localization = {
        "embedding_shape": [18, 256, 256],
        "distance": "torch.nn.PairwiseDistance",
        "p": 2.0,
        "eps": PAIRWISE_DISTANCE_EPSILON,
        "center_source": "center/radius_center.pth:center",
        "score_semantics": "raw_unbounded_nonnegative_euclidean_distance",
        "sigmoid_or_probability_normalization": False,
        "public_api_threshold": MASK_THRESHOLD,
        "threshold_comparison": "greater_than_or_equal",
        "internal_loss_threshold_1_85_times_radius": 1.85 * radius,
        "internal_loss_threshold_used": False,
        "auxiliary_learned_sigmoid_mask_used": False,
    }
    for key, expected in expected_localization.items():
        actual = localization_inference.get(key)
        if isinstance(expected, float):
            if not isinstance(actual, (int, float)) or not math.isclose(
                float(actual),
                expected,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError(
                    f"manifest localization inference {key} mismatch"
                )
        else:
            _require_equal(
                actual,
                expected,
                f"manifest localization inference {key}",
            )
    _require_equal(
        inference.get("native_compatibility_adapter"),
        {
            "purpose": "CLAIMFORGE cross-method native-resolution comparison",
            "operation": (
                "bilinear_restore_raw_256_distance_to_native_before_"
                "thresholding"
            ),
            "mode": "bilinear",
            "align_corners": False,
            "threshold_after_restore": True,
            "official_model_space_retained_as_auxiliary": True,
        },
        "manifest native compatibility adapter",
    )

    metrics = _require_mapping(manifest.get("metrics"), "manifest metrics")
    for key, expected in {
        "task": "T1_image_detection_and_T2_pixel_localization",
        "positive_class": "forged_or_manipulated",
        "classification_threshold": CLASSIFICATION_THRESHOLD,
        "classification_threshold_comparison": "strict_greater_than",
        "official_argmax_reported_separately": True,
        "primary_localization_space": "native",
        "auxiliary_localization_space": "model_256",
        "localization_score": "raw_hypersphere_euclidean_distance",
        "mask_threshold": MASK_THRESHOLD,
        "mask_threshold_comparison": "greater_than_or_equal",
        "prediction_inversion": False,
        "native_gt": "exact_canonical_mask",
        "model_space_gt_resize": "Pillow_nearest_neighbor_to_256x256",
        "forged_pixel_ap_only": True,
        "bootstrap_unit": "task_id_pair",
    }.items():
        _require_equal(metrics.get(key), expected, f"manifest metrics {key}")
    bootstrap_samples = metrics.get("bootstrap_samples")
    if (
        isinstance(bootstrap_samples, bool)
        or not isinstance(bootstrap_samples, int)
        or bootstrap_samples <= 0
    ):
        raise ValueError("manifest bootstrap sample count is invalid")
    _require_equal(
        manifest.get("artifacts"),
        {
            "mask_features_model_256": {
                "format": "npy",
                "dtype": "float32",
                "shape": [18, 256, 256],
            },
            "distance_maps_model_256": {
                "format": "npy",
                "dtype": "float32",
                "shape": [256, 256],
                "range": "nonnegative_unbounded",
            },
            "distance_maps_native": {
                "format": "npy",
                "dtype": "float32",
                "shape": "native_HxW",
                "range": "nonnegative_unbounded",
            },
            "masks_native": {
                "format": "lossless_png",
                "dtype": "uint8",
                "values": [0, 255],
                "relation": "distance_map_native >= 2.3",
            },
        },
        "manifest artifact contract",
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
            "checkpoint_released_identifier": "750001",
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
            for key, expected in (
                ("mask_threshold", MASK_THRESHOLD),
                ("mask_threshold_operator", MASK_THRESHOLD_OPERATOR),
                ("classification_threshold", CLASSIFICATION_THRESHOLD),
                (
                    "classification_threshold_operator",
                    CLASSIFICATION_THRESHOLD_OPERATOR,
                ),
            ):
                _require_equal(
                    row.get(key),
                    expected,
                    f"result row {line_number} {key}",
                )
            _validate_t1_record(row)
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
            summary.get("checkpoint_released_identifier"),
            "750001",
            "summary checkpoint identifier",
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
        (summary.get("valid_for_t1"), True, "summary valid_for_t1"),
        (summary.get("valid_for_t2"), True, "summary valid_for_t2"),
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
    for key, expected in {
        "complete_pairs": complete_pairs,
        "paired_images": complete_pairs * 2,
        "unpaired_valid_images": len(valid_latest) - complete_pairs * 2,
    }.items():
        _require_equal(
            paired_coverage.get(key),
            expected,
            f"summary paired coverage {key}",
        )
    _require_equal(
        summary.get("task_scope"),
        {
            "primary_task": "T1_detection_and_T2_localization",
            "valid_for_t1": True,
            "valid_for_t2": True,
            "primary_detection_score": "score",
            "primary_detection_semantics": (
                "one_minus_softmax_probability_of_hifi_fine_class_0_authentic"
            ),
            "benchmark_classification_threshold": CLASSIFICATION_THRESHOLD,
            "benchmark_classification_threshold_operator": (
                CLASSIFICATION_THRESHOLD_OPERATOR
            ),
            "official_binary_decision": (
                "argmax_fine_14_class_index_not_equal_to_0"
            ),
            "primary_localization_space": "native",
            "auxiliary_localization_space": "model_256",
            "localization_semantics": (
                "hifi_hypersphere_euclidean_distance"
            ),
            "mask_threshold": MASK_THRESHOLD,
            "mask_threshold_operator": MASK_THRESHOLD_OPERATOR,
        },
        "summary task scope",
    )
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
    provenance = {
        "status": "ok",
        "run_manifest_fingerprint": fingerprint,
        "inputs_sha256": actual_inputs_sha256,
        "checkpoint_sha256": pins.CHECKPOINT_BUNDLE_SHA256,
        "physical_result_rows_validated": len(result_rows),
        "latest_result_rows_validated": len(latest),
        "pinned_source_files_validated": len(expected_source_files),
        "adapter_contract_files_validated": adapter_files_checked,
        "runtime": runtime_audit,
        "checks": [
            "run manifest schema, ID, and recomputed immutable fingerprint",
            "canonical release hash, input hash, and ordered selection",
            "official clean source tree and every pinned source-file hash",
            "both released checkpoint hashes, schemas, and bundle identity",
            "center/radius hash, tensor schema, radius value, and runtime path",
            "imageio/Pillow preprocessing and raw-distance inference contract",
            "separate official argmax and benchmark >0.5 T1 decisions",
            "primary native and auxiliary model-256 T2 metric spaces",
            "every physical result row against canonical identity",
            "summary identity, task scope, bootstrap seed, and latest-row coverage",
            "adapter contract file hashes",
        ],
    }
    return provenance, center, radius


def _preprocess_evidence(
    image_path: Path,
) -> tuple[dict[str, Any], np.ndarray, tuple[int, int]]:
    """Replay imageio -> Pillow bicubic -> /255 CHW float32 exactly."""

    # ImageIO's v2 Pillow plugin is the registered runner decoder. Reopening
    # through Pillow is an independent call path and reproduces its RGB bytes
    # for the canonical JPEG/PNG inputs without requiring the runner venv.
    with Image.open(image_path) as opened:
        decoded = np.asarray(opened)
    if decoded.dtype != np.uint8:
        raise ValueError(f"unexpected decoded image dtype: {decoded.dtype}")
    if decoded.ndim != 3 or decoded.shape[2] != 3:
        raise ValueError(f"unexpected decoded image shape: {decoded.shape}")
    native_height, native_width = decoded.shape[:2]
    resized_image = Image.fromarray(decoded).resize(
        (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE),
        resample=Image.Resampling.BICUBIC,
    )
    resized = np.asarray(resized_image, dtype=np.uint8)
    tensor = np.ascontiguousarray(
        (
            resized.astype(np.float32)
            / np.float32(255.0)
        ).transpose(2, 0, 1)
    )
    evidence = {
        "decoder": "imageio.v2.imread",
        "channel_order": "RGB",
        "decoded_dtype": "uint8",
        "native_size": [native_width, native_height],
        "model_size": [MODEL_INPUT_SIZE, MODEL_INPUT_SIZE],
        "geometry": "direct_stretch_without_aspect_ratio_preservation",
        "resize_interpolation": "Pillow.Image.Resampling.BICUBIC",
        "input_crop": None,
        "input_reencode": False,
        "normalization": "uint8_rgb_divide_255_float32",
        "tensor_shape": list(tensor.shape),
        "tensor_dtype": str(tensor.dtype),
        "tensor_sha256": _array_sha256(tensor),
    }
    return evidence, tensor, (native_width, native_height)


def _pairwise_distance_float32(
    embedding: np.ndarray,
    center: np.ndarray,
    *,
    eps: float = PAIRWISE_DISTANCE_EPSILON,
) -> np.ndarray:
    """Replay ``torch.nn.PairwiseDistance(p=2, eps=1e-6)`` in NumPy."""

    features = np.asarray(embedding)
    center_array = np.asarray(center)
    if features.dtype != np.float32:
        raise ValueError(f"embedding dtype is {features.dtype}, not float32")
    if center_array.dtype != np.float32:
        raise ValueError(f"center dtype is {center_array.dtype}, not float32")
    if features.shape != (
        EMBEDDING_CHANNELS,
        MODEL_INPUT_SIZE,
        MODEL_INPUT_SIZE,
    ):
        raise ValueError(
            "embedding shape mismatch: "
            f"{features.shape} != "
            f"{(EMBEDDING_CHANNELS, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE)}"
        )
    if center_array.shape not in {
        (EMBEDDING_CHANNELS,),
        (1, EMBEDDING_CHANNELS),
    }:
        raise ValueError(
            f"center shape mismatch: {center_array.shape} != "
            f"{(EMBEDDING_CHANNELS,)}"
        )
    if not np.isfinite(features).all() or not np.isfinite(center_array).all():
        raise ValueError("embedding or center contains non-finite values")
    if not math.isclose(
        float(eps),
        PAIRWISE_DISTANCE_EPSILON,
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        raise ValueError(
            "HiFi-IFDL PairwiseDistance epsilon must remain exactly 1e-6"
        )

    vectors = np.ascontiguousarray(
        features.transpose(1, 2, 0).reshape(-1, EMBEDDING_CHANNELS)
    )
    difference = (
        vectors
        - center_array.reshape(1, EMBEDDING_CHANNELS)
        + np.float32(PAIRWISE_DISTANCE_EPSILON)
    )
    squared = np.multiply(difference, difference, dtype=np.float32)
    distance = np.sqrt(
        np.sum(squared, axis=1, dtype=np.float32),
        dtype=np.float32,
    )
    return np.ascontiguousarray(
        distance.reshape(MODEL_INPUT_SIZE, MODEL_INPUT_SIZE),
        dtype=np.float32,
    )


def _bilinear_align_corners_false(
    score_map: np.ndarray,
    *,
    width: int,
    height: int,
) -> np.ndarray:
    """Pure NumPy half-pixel bilinear resize matching PyTorch geometry."""

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
    # Pillow NEAREST samples the source pixel containing the mapped output
    # pixel center. This is floor((i + 0.5) * input / output), not
    # floor(i * input / output); the distinction matters for non-integer
    # native-to-256 ratios.
    y = np.floor(
        (np.arange(height, dtype=np.float64) + 0.5)
        * source_height
        / height
    ).astype(np.int64)
    x = np.floor(
        (np.arange(width, dtype=np.float64) + 0.5)
        * source_width
        / width
    ).astype(np.int64)
    y = np.minimum(y, source_height - 1)
    x = np.minimum(x, source_width - 1)
    return np.ascontiguousarray(source[y[:, None], x[None, :]])


def _softmax_float64(logits: Any, *, label: str) -> np.ndarray:
    try:
        values = np.asarray(logits, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} is not numeric") from exc
    if values.ndim != 1 or values.size == 0:
        raise ValueError(f"{label} must be a non-empty one-dimensional vector")
    if not np.isfinite(values).all():
        raise ValueError(f"{label} contains non-finite values")
    shifted = values - float(np.max(values))
    exponentials = np.exp(shifted)
    return exponentials / float(np.sum(exponentials))


def _fine_logits_from_result(result: dict[str, Any]) -> list[Any]:
    levels = result.get("classification_hierarchy_logits")
    if not isinstance(levels, dict) or list(levels) != list(
        CLASSIFICATION_LEVEL_NAMES
    ):
        raise ValueError(
            f"classification logits for {result.get('id')} must have exactly "
            f"the four registered out0..out3 levels"
        )
    for index, (name, size) in enumerate(
        zip(
            CLASSIFICATION_LEVEL_NAMES,
            CLASSIFICATION_LEVEL_SIZES,
            strict=True,
        )
    ):
        level = levels[name]
        if not isinstance(level, list) or len(level) != size:
            raise ValueError(
                f"classification {name} logits for {result.get('id')} "
                f"must contain {size} values"
            )
        _softmax_float64(
            level,
            label=f"classification {name} logits for {result.get('id')}",
        )
    fine = levels[CLASSIFICATION_LEVEL_NAMES[-1]]
    return fine


def _recomputed_t1(result: dict[str, Any]) -> dict[str, Any]:
    fine_logits = _fine_logits_from_result(result)
    probabilities = _softmax_float64(
        fine_logits,
        label=f"fine logits for {result.get('id')}",
    )
    # The official rule is argmax of out3 logits. Do not infer it from
    # rounded/stored probabilities.
    predicted_class = int(np.argmax(np.asarray(fine_logits, dtype=np.float64)))
    score = float(1.0 - probabilities[0])
    return {
        "probabilities": probabilities,
        "fine_class_index": predicted_class,
        "score": score,
        "official_binary_decision": predicted_class != 0,
        "benchmark_binary_decision": score > CLASSIFICATION_THRESHOLD,
    }


def _validate_t1_record(result: dict[str, Any]) -> dict[str, Any]:
    replay = _recomputed_t1(result)
    result_id = result.get("id")
    recorded_probabilities = result.get("classification_probabilities")
    if (
        not isinstance(recorded_probabilities, list)
        or len(recorded_probabilities) != FINE_CLASS_COUNT
    ):
        raise ValueError(
            f"classification probabilities for {result_id} must contain "
            f"{FINE_CLASS_COUNT} values"
        )
    try:
        recorded_array = np.asarray(recorded_probabilities, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"classification probabilities for {result_id} are not numeric"
        ) from exc
    if (
        not np.isfinite(recorded_array).all()
        or not np.allclose(
            recorded_array,
            replay["probabilities"],
            rtol=0.0,
            atol=PROBABILITY_ABSOLUTE_TOLERANCE,
        )
    ):
        raise ValueError(
            f"classification probabilities are not softmax(fine logits) "
            f"for {result_id}"
        )
    if not math.isclose(
        float(np.sum(recorded_array)),
        1.0,
        rel_tol=0.0,
        abs_tol=1e-6,
    ):
        raise ValueError(
            f"classification probabilities do not sum to one for {result_id}"
        )

    recorded_score = result.get("score")
    if (
        isinstance(recorded_score, bool)
        or not isinstance(recorded_score, (int, float))
        or not math.isfinite(float(recorded_score))
        or not math.isclose(
            float(recorded_score),
            float(replay["score"]),
            rel_tol=1e-6,
            abs_tol=PROBABILITY_ABSOLUTE_TOLERANCE,
        )
    ):
        raise ValueError(
            f"T1 score mismatch for {result_id}: "
            f"{recorded_score!r} != {replay['score']!r}"
        )
    benchmark_decision = (
        float(recorded_score) > CLASSIFICATION_THRESHOLD
    )
    for key, expected, label in (
        (
            "score_source",
            "native_out3_fine_14class_head",
            "score source",
        ),
        (
            "score_semantics",
            "one_minus_softmax_probability_fine_class_0_authentic",
            "score semantics",
        ),
        (
            "official_fine_class_index",
            replay["fine_class_index"],
            "official fine-class argmax",
        ),
        (
            "official_binary_decision",
            replay["official_binary_decision"],
            "official binary decision",
        ),
        (
            "benchmark_binary_decision",
            benchmark_decision,
            "benchmark binary decision",
        ),
        (
            "classification_threshold",
            CLASSIFICATION_THRESHOLD,
            "classification threshold",
        ),
        (
            "classification_threshold_operator",
            CLASSIFICATION_THRESHOLD_OPERATOR,
            "classification threshold operator",
        ),
        (
            "decision",
            (
                "forged"
                if benchmark_decision
                else "authentic"
            ),
            "benchmark string decision",
        ),
        (
            "official_fine_class_name",
            FINE_CLASS_NAMES[int(replay["fine_class_index"])],
            "official fine-class name",
        ),
        (
            "official_decision",
            (
                "forged"
                if replay["official_binary_decision"]
                else "authentic"
            ),
            "official string decision",
        ),
        (
            "official_decision_rule",
            "argmax_fine_14class_index_not_equal_to_0",
            "official decision rule",
        ),
    ):
        _require_equal(
            result.get(key),
            expected,
            f"{label} for {result_id}",
        )
    _require_equal(
        result.get("pairwise_distance"),
        {"p": 2.0, "eps": PAIRWISE_DISTANCE_EPSILON},
        f"PairwiseDistance contract for {result_id}",
    )
    auxiliary = _require_mapping(
        result.get("auxiliary_learned_mask"),
        f"auxiliary learned mask for {result_id}",
    )
    for key, expected in (
        ("shape", [MODEL_INPUT_SIZE, MODEL_INPUT_SIZE]),
        ("dtype", "float32"),
        ("primary_output", False),
        (
            "reason",
            (
                "the official public localize API ignores this sigmoid mask "
                "and thresholds hypersphere distance instead"
            ),
        ),
    ):
        _require_equal(
            auxiliary.get(key),
            expected,
            f"auxiliary learned mask {key} for {result_id}",
        )
    for key in ("minimum", "maximum", "mean"):
        value = auxiliary.get(key)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or not 0.0 <= float(value) <= 1.0
        ):
            raise ValueError(
                f"auxiliary learned mask {key} is invalid for {result_id}"
            )
    if not (
        float(auxiliary["minimum"])
        <= float(auxiliary["mean"])
        <= float(auxiliary["maximum"])
    ):
        raise ValueError(
            f"auxiliary learned mask summary ordering is invalid for {result_id}"
        )
    return replay


def _safe_div(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _mcc(tp: int, fp: int, fn: int, tn: int) -> float | None:
    denominator = math.sqrt(
        float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    )
    return (tp * tn - fp * fn) / denominator if denominator else None


def _binary_distance_metrics_strict(
    distance_map: np.ndarray,
    target: np.ndarray,
    *,
    include_ap: bool,
) -> dict[str, Any]:
    scores = np.asarray(distance_map, dtype=np.float32)
    truth = np.asarray(target, dtype=bool)
    if scores.shape != truth.shape:
        raise ValueError(
            f"distance/target mismatch: {scores.shape} != {truth.shape}"
        )
    if scores.size == 0 or not np.isfinite(scores).all():
        raise ValueError("distance map is empty or non-finite")
    if float(scores.min()) < 0.0:
        raise ValueError("distance map contains negative values")
    prediction = scores >= np.float32(MASK_THRESHOLD)
    tp = int(np.count_nonzero(prediction & truth))
    fp = int(np.count_nonzero(prediction & ~truth))
    fn = int(np.count_nonzero(~prediction & truth))
    tn = int(np.count_nonzero(~prediction & ~truth))
    pixel_ap: float | None = None
    if include_ap and truth.any():
        pixel_ap = float(
            average_precision_score(
                truth.reshape(-1),
                scores.reshape(-1),
            )
        )
    return {
        "threshold": MASK_THRESHOLD,
        "threshold_operator": MASK_THRESHOLD_OPERATOR,
        "score_semantics": "hifi_hypersphere_euclidean_distance",
        "score_dtype": "float32",
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


def _descriptive(values: Iterable[float]) -> dict[str, float | int | None]:
    data = np.asarray(list(values), dtype=np.float64)
    if not data.size:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "min": None,
            "max": None,
        }
    if not np.isfinite(data).all():
        raise ValueError("descriptive input contains non-finite values")
    return {
        "count": int(data.size),
        "mean": float(np.mean(data)),
        "median": float(np.median(data)),
        "min": float(np.min(data)),
        "max": float(np.max(data)),
    }


def _validate_metrics(
    result: dict[str, Any],
    *,
    distance_map: np.ndarray,
    target: np.ndarray,
    space: str,
) -> None:
    localization = _require_mapping(
        result.get("localization"),
        f"localization for {result.get('id')}",
    )
    recorded = _require_mapping(
        localization.get(space),
        f"{space} localization for {result.get('id')}",
    )
    expected = _binary_distance_metrics_strict(
        distance_map,
        target,
        include_ap=result.get("kind") == "forged",
    )
    for key, value in expected.items():
        _compare_metric(
            recorded.get(key),
            value,
            f"{space} localization {key} for {result.get('id')}",
        )


def _load_center_file(path: Path) -> tuple[np.ndarray, float]:
    """Safely load the pinned official 18-D center and descriptive radius."""

    import torch

    try:
        payload = torch.load(
            path,
            map_location="cpu",
            weights_only=True,
        )
    except TypeError as exc:
        raise RuntimeError(
            "auditing the center requires torch.load(weights_only=True)"
        ) from exc
    if not isinstance(payload, dict) or set(payload) != {"center", "radius"}:
        raise ValueError(
            "official radius_center.pth must contain only center and radius"
        )
    center_value = payload["center"]
    radius_value = payload["radius"]
    if not isinstance(center_value, torch.Tensor):
        raise ValueError("center payload is not a tensor")
    if not isinstance(radius_value, torch.Tensor):
        raise ValueError("radius payload is not a tensor")
    center = np.ascontiguousarray(center_value.detach().cpu().numpy())
    radius_array = np.asarray(radius_value.detach().cpu().numpy())
    if center.dtype != np.float32 or center.shape != (EMBEDDING_CHANNELS,):
        raise ValueError(
            f"center tensor mismatch: dtype={center.dtype} shape={center.shape}"
        )
    if radius_array.dtype != np.float32 or radius_array.shape != ():
        raise ValueError(
            "radius tensor must be a scalar float32 tensor"
        )
    radius = float(radius_array)
    if not np.isfinite(center).all() or not math.isfinite(radius) or radius <= 0:
        raise ValueError("center file contains non-finite or invalid values")
    return center, radius


def _verify_upstream_runtime(
    *,
    hifi_ifdl_root: Path,
    pins: SimpleNamespace,
    checkpoint_paths: dict[str, Path],
) -> tuple[np.ndarray, float, dict[str, Any]]:
    commit = _git_value(hifi_ifdl_root, "rev-parse", "HEAD")
    _require_equal(
        commit,
        pins.MODEL_SOURCE_COMMIT,
        "checked-out HiFi-IFDL source commit",
    )
    tracked_status = _git_value(
        hifi_ifdl_root,
        "status",
        "--short",
        "--untracked-files=no",
    )
    _require_equal(
        tracked_status,
        "",
        "checked-out HiFi-IFDL tracked-source status",
    )

    source_files = _normalise_source_files(
        pins.SOURCE_FILES,
        "runner source pins",
    )
    for relative, digest in source_files.items():
        _verify_hash(
            hifi_ifdl_root / relative,
            digest,
            f"upstream source file {relative}",
        )

    initialization = _require_mapping(
        pins.INITIALIZATION_WEIGHT,
        "runner initialization-weight pin",
    )
    initialization_path_value = initialization.get("path")
    if not isinstance(initialization_path_value, str):
        raise ValueError("runner initialization-weight pin has no path")
    initialization_path = hifi_ifdl_root / initialization_path_value
    _verify_hash(
        initialization_path,
        initialization.get("sha256"),
        "official HRNet initialization weight",
    )
    if initialization_path.stat().st_size != int(initialization["bytes"]):
        raise ValueError("official HRNet initialization weight byte-size mismatch")

    checkpoints = _normalise_components(
        pins.CHECKPOINTS,
        "runner checkpoint pins",
    )
    _require_equal(
        set(checkpoint_paths),
        set(checkpoints),
        "runtime checkpoint roles",
    )
    checkpoint_hashes: dict[str, str] = {}
    for role, contract in checkpoints.items():
        digest = _require_sha256(
            contract.get("sha256"),
            f"checkpoint pin {role} SHA-256",
        )
        path = checkpoint_paths[role]
        _verify_hash(path, digest, f"official checkpoint {role}")
        if "bytes" in contract and path.stat().st_size != int(contract["bytes"]):
            raise ValueError(f"official checkpoint {role} byte-size mismatch")
        checkpoint_hashes[role] = digest

    center_contract = _require_mapping(pins.CENTER, "runner center pin")
    center_path_value = center_contract.get("path")
    if not isinstance(center_path_value, str) or not center_path_value:
        raise ValueError("runner center pin has no path")
    center_digest = _require_sha256(
        center_contract.get("sha256"),
        "runner center pin SHA-256",
    )
    center_path = hifi_ifdl_root / center_path_value
    _verify_hash(center_path, center_digest, "official radius/center file")
    if (
        "bytes" in center_contract
        and center_path.stat().st_size != int(center_contract["bytes"])
    ):
        raise ValueError("official radius/center file byte-size mismatch")
    center, radius = _load_center_file(center_path)
    if "center_shape" in center_contract:
        _require_equal(
            center_contract.get("center_shape"),
            [EMBEDDING_CHANNELS],
            "center pin shape",
        )
    if "radius_value" in center_contract and not math.isclose(
        float(center_contract["radius_value"]),
        radius,
        rel_tol=0.0,
        abs_tol=1e-7,
    ):
        raise ValueError("center pin radius does not match physical center file")

    return (
        center,
        radius,
        {
            "source_commit": commit,
            "source_files_checked": len(source_files),
            "initialization_weight_checked": True,
            "checkpoint_files_checked": len(checkpoints),
            "checkpoint_sha256_by_role": checkpoint_hashes,
            "center_sha256": center_digest,
            "center_radius": radius,
        },
    )


def _load_float32_array(
    path: Path,
    *,
    expected_shape: tuple[int, ...],
    label: str,
) -> np.ndarray:
    try:
        with path.open("rb") as handle:
            value = np.load(handle, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ValueError(f"cannot safely load {label}: {path}") from exc
    if value.dtype != np.float32:
        raise ValueError(f"{label} dtype is {value.dtype}, not float32")
    if value.shape != expected_shape:
        raise ValueError(
            f"{label} shape mismatch: {value.shape} != {expected_shape}"
        )
    if not np.isfinite(value).all():
        raise ValueError(f"{label} contains non-finite values")
    return np.ascontiguousarray(value)


def _load_target(
    *,
    result: dict[str, Any],
    input_row: dict[str, Any],
    repo_root: Path,
    width: int,
    height: int,
    checked_paths: set[Path],
) -> np.ndarray:
    if result.get("kind") == "real":
        _require_equal(
            input_row.get("gt_mask_kind"),
            "all_zero",
            f"real GT kind for {result.get('id')}",
        )
        if input_row.get("gt_mask_path") is not None:
            raise ValueError(
                f"real input unexpectedly has a GT file: {result.get('id')}"
            )
        return np.zeros((height, width), dtype=bool)

    target_value = input_row.get("gt_mask_path")
    if not isinstance(target_value, str):
        raise ValueError(f"forged input has no GT mask: {result.get('id')}")
    target_path = _anchored(Path(target_value), repo_root)
    _verify_hash(
        target_path,
        input_row.get("gt_mask_sha256"),
        f"ground-truth mask {result.get('id')}",
    )
    checked_paths.add(target_path)
    with Image.open(target_path) as opened:
        target = np.asarray(opened.convert("L"), dtype=np.uint8) > 0
    if target.shape != (height, width):
        raise ValueError(
            f"ground-truth shape mismatch for {result.get('id')}: "
            f"{target.shape} != {(height, width)}"
        )
    if not target.any():
        raise ValueError(f"forged ground truth is empty for {result.get('id')}")
    return target


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


def audit_artifacts(
    pairs: list[Pair],
    *,
    repo_root: Path,
    center: np.ndarray,
    radius: float,
) -> dict[str, Any]:
    if not pairs:
        raise ValueError("artifact audit requires at least one pair")
    checked_paths: set[Path] = set()
    artifact_owners: dict[Path, str] = {}
    box_iou_values: list[float] = []
    box_coverage_values: list[float] = []
    prediction_inside_values: list[float] = []
    box_any_overlap = 0
    box_iou_hits = 0

    for pair in pairs:
        forged_prediction: np.ndarray | None = None
        for result in (pair.real, pair.forged):
            result_id = str(result["id"])
            input_row = (
                pair.input_row
                if result.get("kind") == "forged"
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
            for key, expected in evidence.items():
                _require_equal(
                    preprocess.get(key),
                    expected,
                    f"preprocess {key} for {result_id}",
                )

            artifact_specs = (
                (
                    "mask_feature_model_path",
                    "mask_feature_model_sha256",
                    "18-channel model embedding",
                ),
                (
                    "distance_map_model_path",
                    "distance_map_model_sha256",
                    "model-256 distance map",
                ),
                ("score_map_path", "score_map_sha256", "native distance map"),
                ("mask_path", "mask_sha256", "native threshold mask"),
            )
            resolved: dict[str, Path] = {}
            for path_key, hash_key, label in artifact_specs:
                path_value = result.get(path_key)
                if not isinstance(path_value, str) or not path_value:
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

            embedding_shape = (
                EMBEDDING_CHANNELS,
                MODEL_INPUT_SIZE,
                MODEL_INPUT_SIZE,
            )
            model_shape = (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE)
            embedding = _load_float32_array(
                resolved["mask_feature_model_path"],
                expected_shape=embedding_shape,
                label=f"model embedding for {result_id}",
            )
            model_distance = _load_float32_array(
                resolved["distance_map_model_path"],
                expected_shape=model_shape,
                label=f"model-256 distance map for {result_id}",
            )
            native_distance = _load_float32_array(
                resolved["score_map_path"],
                expected_shape=(height, width),
                label=f"native distance map for {result_id}",
            )
            if float(model_distance.min()) < 0.0:
                raise ValueError(
                    f"model-256 distance map is negative for {result_id}"
                )
            if float(native_distance.min()) < 0.0:
                raise ValueError(f"native distance map is negative for {result_id}")
            for key, expected in (
                ("mask_feature_model_shape", list(embedding_shape)),
                ("mask_feature_model_dtype", "float32"),
                ("distance_map_model_shape", list(model_shape)),
                ("distance_map_model_dtype", "float32"),
                ("score_map_shape", [height, width]),
                ("score_map_dtype", "float32"),
                (
                    "score_map_semantics",
                    "raw_hifi_hypersphere_euclidean_distance",
                ),
                (
                    "score_map_native_restore",
                    (
                        "bilinear_align_corners_false_from_256_raw_"
                        "distance_compatibility_adapter"
                    ),
                ),
                ("mask_shape", [height, width]),
                ("mask_threshold", MASK_THRESHOLD),
                ("mask_threshold_operator", MASK_THRESHOLD_OPERATOR),
            ):
                _require_equal(
                    result.get(key),
                    expected,
                    f"{key} metadata for {result_id}",
                )

            replayed_distance = _pairwise_distance_float32(embedding, center)
            if not np.allclose(
                model_distance,
                replayed_distance,
                rtol=0.0,
                atol=DISTANCE_ABSOLUTE_TOLERANCE,
            ):
                maximum = float(
                    np.max(np.abs(model_distance - replayed_distance))
                )
                raise ValueError(
                    "model distance is not PairwiseDistance(embedding, center, "
                    f"p=2, eps=1e-6) for {result_id}; max_abs={maximum}"
                )
            expected_native = _bilinear_align_corners_false(
                model_distance,
                width=width,
                height=height,
            )
            if not np.allclose(
                native_distance,
                expected_native,
                rtol=0.0,
                atol=NATIVE_RESTORE_ABSOLUTE_TOLERANCE,
            ):
                maximum = float(
                    np.max(np.abs(native_distance - expected_native))
                )
                raise ValueError(
                    "native distance is not the bilinear "
                    "align_corners=False restore of raw model distance for "
                    f"{result_id}; max_abs={maximum}"
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
            if not set(np.unique(mask_array).tolist()).issubset({0, 255}):
                raise ValueError(f"threshold mask is not binary for {result_id}")
            expected_mask = np.where(
                native_distance >= np.float32(MASK_THRESHOLD),
                np.uint8(255),
                np.uint8(0),
            )
            if not np.array_equal(mask_array, expected_mask):
                raise ValueError(
                    f"inclusive >=2.3 native threshold mask mismatch for "
                    f"{result_id}"
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
                distance_map=model_distance,
                target=model_target,
                space="model_256",
            )
            _validate_metrics(
                result,
                distance_map=native_distance,
                target=target,
                space="native",
            )
            _validate_t1_record(result)
            if result.get("kind") == "forged":
                forged_prediction = mask_array > 0

        if forged_prediction is None:
            raise ValueError(f"forged artifacts were not loaded for {pair.task_id}")
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

    return {
        "artifact_integrity": {
            "status": "ok",
            "checked_files": len(checked_paths),
            "pairs": len(pairs),
            "result_images": len(pairs) * 2,
            "center_radius": radius,
            "numeric_tolerances": {
                "pairwise_distance_absolute": DISTANCE_ABSOLUTE_TOLERANCE,
                "native_restore_absolute": NATIVE_RESTORE_ABSOLUTE_TOLERANCE,
                "softmax_probability_absolute": (
                    PROBABILITY_ABSOLUTE_TOLERANCE
                ),
                "reason": (
                    "bounded float32 CUDA-versus-independent-CPU kernel "
                    "rounding only"
                ),
            },
            "checks": [
                "canonical images, GT masks, and all artifact file hashes",
                "independent decode reproduces imageio/Pillow bicubic /255 tensor hash",
                "18x256x256 embedding replays PairwiseDistance p=2 eps=1e-6",
                "native float32 distance is bilinear align_corners=False restoration",
                "native uint8 mask bit-exactly equals restored distance >=2.3",
                "all four classification logit levels have registered dimensions",
                "fine probabilities and score independently replay softmax and 1-p(class0)",
                "official argmax and benchmark >0.5 decisions are checked separately",
                "native and model-256 localization metrics are independently recomputed",
            ],
        },
        "box_hit_at_native_mask_threshold_2_3": {
            "task_scope": "T2_pixel_localization",
            "mask_threshold": MASK_THRESHOLD,
            "threshold_operator": MASK_THRESHOLD_OPERATOR,
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


def _quintiles(pairs: list[Pair]) -> list[tuple[str, list[Pair]]]:
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
    pairs: list[Pair],
    *,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    return {
        space: summarize_hifi_ifdl_pair_slice(
            pairs,
            iterations=iterations,
            seed=seed,
            localization_space=space,
        )
        for space in ("native", "model_256")
    }


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    if (
        isinstance(args.bootstrap_iterations, bool)
        or args.bootstrap_iterations <= 0
    ):
        raise ValueError("bootstrap iterations must be positive")
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
    hifi_ifdl_root = args.hifi_ifdl_root.resolve()
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
    provenance, center, radius = validate_provenance(
        repo_root=repo_root,
        hifi_ifdl_root=hifi_ifdl_root,
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
        center=center,
        radius=radius,
    )
    value = {
        "schema_version": "hifi_ifdl_posthoc_analysis_v1",
        "run_id": args.run_id,
        "created_at": utc_now(),
        "task_scope": {
            "primary_task": "T1_detection_and_T2_localization",
            "valid_for_t1": True,
            "valid_for_t2": True,
            "primary_t1": (
                "one_minus_softmax_probability_fine_class_0_authentic"
            ),
            "official_t1_decision": (
                "argmax_fine_14class_index_not_equal_to_0"
            ),
            "benchmark_classification_threshold": CLASSIFICATION_THRESHOLD,
            "benchmark_classification_threshold_operator": (
                CLASSIFICATION_THRESHOLD_OPERATOR
            ),
            "primary_localization_space": "native",
            "auxiliary_localization_space": "model_256",
            "primary_t2": "raw_hifi_hypersphere_euclidean_distance",
            "mask_threshold": MASK_THRESHOLD,
            "mask_threshold_operator": MASK_THRESHOLD_OPERATOR,
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
            "spaces": ["native", "model_256"],
            "metrics_scope": "T1 detection and T2 localization",
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
    parser.add_argument(
        "--hifi-ifdl-root",
        type=Path,
        default=DEFAULT_HIFI_IFDL_ROOT,
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260724)
    return parser.parse_args()


def main() -> None:
    analyze(parse_args())


if __name__ == "__main__":
    main()
