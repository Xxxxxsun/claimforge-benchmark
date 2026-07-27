#!/usr/bin/env python3
"""Independently audit official IML-ViT Balanced250 runs.

Formal analysis is fail closed: it requires exact 1,025-image T2-only
coverage, the frozen A/B smoke gate, complete artifact/provenance replay,
Balanced250 native localization metrics, and by default a fresh full-model
replay on the recorded logical device.  Official artifacts and replay retain
the strict ``> 0.5`` mask, while the frozen cross-method reducer separately
uses ``>= 0.5`` and reports threshold-boundary non-equivalence.  IML-ViT has
no native T1 output, so this analyzer rejects image-level scores and never
evaluates full-frame rows.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import sys
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
from PIL import Image

from eval.opensource import run_imlvit_balanced as runner
from eval.opensource.balanced250_localization_metrics import (
    SUMMARY_SCHEMA_VERSION as T2_METRICS_SCHEMA_VERSION,
)
from eval.opensource.balanced250_localization_metrics import (
    summarize_balanced250_t2,
)
from eval.opensource.balanced_run_contract import (
    RunDatasetContract,
    build_run_dataset_contract,
    index_latest_attempts,
    summarize_coverage,
)
from eval.opensource.canonical_release import (
    Capability,
    CanonicalRelease,
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
from eval.opensource.imlvit_metrics import binary_pixel_metrics_strict


AUDIT_SCHEMA_VERSION = "imlvit_balanced_independent_audit_v3"
METRICS_SCHEMA_VERSION = "imlvit_balanced_metrics_v3"
SMOKE_COMPARISON_SCHEMA_VERSION = "imlvit_balanced_smoke_comparison_v3"
FRESH_REPLAY_SCHEMA_VERSION = "imlvit_balanced_fresh_replay_v3"

DEFAULT_RESULTS_DIR = runner.DEFAULT_RESULTS_DIR
DEFAULT_ARTIFACTS_DIR = runner.DEFAULT_ARTIFACTS_DIR
DEFAULT_FORMAL_RUN_ID = runner.DEFAULT_FORMAL_RUN_ID
DEFAULT_SMOKE_RUN_ID_A = runner.DEFAULT_SMOKE_RUN_ID_A
DEFAULT_SMOKE_RUN_ID_B = runner.DEFAULT_SMOKE_RUN_ID_B
DEFAULT_IMLVIT_ROOT = runner.legacy.DEFAULT_IMLVIT_ROOT
DEFAULT_CHECKPOINT = runner.legacy.DEFAULT_CHECKPOINT

FORMAL_IMAGES = runner.FORMAL_IMAGES
SMOKE_IMAGES = runner.SMOKE_IMAGES
FORMAL_SELECTED_IDS_SHA256 = (
    "612e08565e38cb219fe5ea94dc8193580e099455e11fa778822488dbe7071717"
)
FORMAL_SELECTED_ROWS_SHA256 = (
    "19ff584a5d073dd03cd31eaf0d22b105d079b2dd606ea535fbbcd39fb692b887"
)
SMOKE_SELECTED_IDS_SHA256 = (
    "3ce822824a5548f12ae0633520a19686048fd175f7add178334ab5c4fe7e78f4"
)
SMOKE_SELECTED_ROWS_SHA256 = (
    "7ec14339cad5c6e083f6b1fde56a965686d552ca0b9026eea975144ade7d1d6c"
)

BOOTSTRAP_ITERATIONS = 1_000
BOOTSTRAP_SEED = 20_260_727
SIGMOID_ABSOLUTE_TOLERANCE = 2e-7
NATIVE_RESTORE_ABSOLUTE_TOLERANCE = 2e-5

ANALYZER_SOURCE_PATHS = (
    ".gitignore",
    "eval/__init__.py",
    "eval/opensource/__init__.py",
    "eval/opensource/analyze_imlvit_balanced.py",
    "eval/opensource/run_imlvit_balanced.py",
    "eval/opensource/analyze_imlvit_run.py",
    "eval/opensource/run_imlvit.py",
    "eval/opensource/imlvit_metrics.py",
    "eval/opensource/canonical_release.py",
    "eval/opensource/balanced_run_contract.py",
    "eval/opensource/balanced250_localization_metrics.py",
    "eval/opensource/common.py",
)

_RUN_DEPENDENT_RESULT_FIELDS = frozenset(
    {
        "run_id",
        "run_manifest_fingerprint",
        "completed_at",
        "latency_ms",
        "peak_cuda_memory_bytes",
        "raw_logits_model_path",
        "score_map_model_path",
        "score_map_path",
        "mask_path",
    }
)


@dataclass(frozen=True)
class RunBundle:
    repo_root: Path
    run_id: str
    mode: str
    run_dir: Path
    artifact_root: Path
    manifest_path: Path
    expected_path: Path
    results_path: Path
    summary_path: Path
    manifest: dict[str, Any]
    immutable: dict[str, Any]
    fingerprint: str
    release: CanonicalRelease
    selection_spec: SelectionSpec
    selected: tuple[dict[str, Any], ...]
    contract: RunDatasetContract
    physical_results: tuple[dict[str, Any], ...]
    latest_results: tuple[dict[str, Any], ...]
    summary: dict[str, Any]
    coverage: Mapping[str, Any]
    history: Mapping[str, Any]
    evidence_snapshot: Mapping[str, str]


@dataclass
class NativeMapLoadAudit:
    maps_loaded: int = 0
    native_pixels_exactly_at_threshold: int = 0
    native_images_with_pixels_exactly_at_threshold: int = 0


def _fingerprint(value: Mapping[str, Any] | Sequence[Any]) -> str:
    return hashlib.sha256(stable_json(value).encode("utf-8")).hexdigest()


def _anchored(path: Path, repo_root: Path) -> Path:
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


def _reject_symlink_components(path: Path, label: str) -> None:
    absolute = path.absolute()
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        if current.is_symlink():
            raise ValueError(f"{label} contains a symlink component: {current}")


def _safe_standard_root(
    requested: Path,
    *,
    repo_root: Path,
    expected_relative: Path,
    label: str,
) -> Path:
    expected = (repo_root / expected_relative).resolve()
    unresolved = requested if requested.is_absolute() else repo_root / requested
    _reject_symlink_components(unresolved, label)
    candidate = unresolved.resolve()
    if candidate != expected:
        raise ValueError(f"{label} must be exactly {expected_relative}")
    if not candidate.is_dir() or candidate.is_symlink():
        raise FileNotFoundError(candidate)
    return candidate


def _safe_run_dir(root: Path, run_id: str, label: str) -> Path:
    safe_id = runner._valid_run_id(run_id)
    candidate = root / safe_id
    _reject_symlink_components(candidate, label)
    resolved = candidate.resolve()
    if resolved.parent != root.resolve():
        raise ValueError(f"{label} escapes its standard root")
    if not resolved.is_dir() or resolved.is_symlink():
        raise FileNotFoundError(resolved)
    return resolved


def _safe_exact_file(path: Path, *, parent: Path, label: str) -> Path:
    _reject_symlink_components(path, label)
    resolved = path.resolve()
    if resolved.parent != parent.resolve():
        raise ValueError(f"{label} is outside the frozen run directory")
    if not resolved.is_file() or resolved.is_symlink():
        raise FileNotFoundError(resolved)
    return resolved


def _safe_recorded_repo_path(
    value: Any,
    *,
    repo_root: Path,
    expected: Path,
    label: str,
) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} is not a path")
    pure = PurePosixPath(value)
    if (
        "\\" in value
        or pure.is_absolute()
        or pure.as_posix() != value
        or any(part in ("", ".", "..") for part in pure.parts)
    ):
        raise ValueError(f"{label} is non-canonical or traversing")
    unresolved = repo_root / Path(value)
    _reject_symlink_components(unresolved, label)
    _reject_symlink_components(expected, f"expected {label}")
    candidate = unresolved.resolve()
    if candidate != expected.resolve():
        raise ValueError(f"{label} does not equal the frozen path")
    if not unresolved.is_file() or unresolved.is_symlink() or not candidate.is_file():
        raise FileNotFoundError(candidate)
    return candidate


def _write_json_verified(path: Path, value: Mapping[str, Any]) -> str:
    _reject_symlink_components(path, f"analysis output {path.name}")
    if path.exists() and (path.is_symlink() or not path.is_file()):
        raise ValueError(f"unsafe analysis output: {path}")
    atomic_write_json(path, dict(value))
    loaded = runner._load_json_object_strict(path)
    if stable_json(loaded) != stable_json(dict(value)):
        raise ValueError(f"analysis output round trip changed: {path}")
    return sha256_file(path)


def analyzer_source_contract(
    repo_root: Path,
) -> dict[str, dict[str, Any]]:
    contract: dict[str, dict[str, Any]] = {}
    for relative in ANALYZER_SOURCE_PATHS:
        path = repo_root / relative
        _reject_symlink_components(path, f"analyzer source {relative}")
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(path)
        contract[relative] = {
            "path": relative,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    return contract


def _sigmoid_float32(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float32)
    result = np.empty_like(values)
    positive = values >= np.float32(0.0)
    result[positive] = np.float32(1.0) / (np.float32(1.0) + np.exp(-values[positive]))
    exponentials = np.exp(values[~positive])
    result[~positive] = exponentials / (np.float32(1.0) + exponentials)
    return np.ascontiguousarray(result, dtype=np.float32)


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
    x = (np.arange(width, dtype=np.float32) + np.float32(0.5)) * np.float32(
        source_width / width
    ) - np.float32(0.5)
    y = (np.arange(height, dtype=np.float32) + np.float32(0.5)) * np.float32(
        source_height / height
    ) - np.float32(0.5)
    x = np.maximum(x, np.float32(0.0))
    y = np.maximum(y, np.float32(0.0))
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    x1 = np.minimum(x0 + 1, source_width - 1)
    y1 = np.minimum(y0 + 1, source_height - 1)
    wx = (x - x0.astype(np.float32))[None, :]
    wy = (y - y0.astype(np.float32))[:, None]
    horizontal = source[:, x0] * (np.float32(1.0) - wx) + source[:, x1] * wx
    restored = horizontal[y0, :] * (np.float32(1.0) - wy) + horizontal[y1, :] * wy
    return np.ascontiguousarray(restored, dtype=np.float32)


def _verify_contract_digest(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} is not an object")
    digest = value.get("contract_sha256")
    payload = {key: item for key, item in value.items() if key != "contract_sha256"}
    if digest != _fingerprint(payload):
        raise ValueError(f"{label} contract digest changed")
    return value


def _validate_recorded_provenance(
    immutable: Mapping[str, Any],
    *,
    repo_root: Path,
    imlvit_root: Path,
    checkpoint_path: Path,
) -> dict[str, Any]:
    cpu = immutable.get("cpu_preflight")
    if not isinstance(cpu, Mapping):
        raise ValueError("recorded CPU preflight is not an object")
    _verify_contract_digest(cpu, "CPU preflight")
    environment = runner.verify_environment()
    source = runner.verify_source(imlvit_root)
    assets = runner.verify_assets(checkpoint_path)
    adapter_sources = runner.adapter_source_contract(repo_root)
    artifact_ignore = runner.verify_artifact_ignore(repo_root)
    expected_fields = {
        "schema_version": runner.CPU_PREFLIGHT_SCHEMA,
        "environment": environment,
        "source": source,
        "assets": assets,
        "adapter_sources": adapter_sources,
        "artifact_ignore": artifact_ignore,
        "cuda_initialized_before": False,
        "cuda_initialized_after": False,
        "balanced250_forward_performed": False,
        "balanced250_score_computed": False,
        "t1_output_computed": False,
    }
    for key, expected in expected_fields.items():
        if stable_json(cpu.get(key)) != stable_json(expected):
            raise ValueError(f"recorded CPU preflight {key} changed")
    checkpoint_audit = cpu.get("checkpoint_audit")
    model_audit = cpu.get("model_audit")
    if not isinstance(checkpoint_audit, Mapping) or not isinstance(
        model_audit, Mapping
    ):
        raise ValueError("recorded structural audits are missing")
    if (
        checkpoint_audit.get("state_keys")
        != int(runner.legacy.CHECKPOINT["state_keys"])
        or checkpoint_audit.get("state_elements")
        != int(runner.legacy.CHECKPOINT["state_elements"])
        or checkpoint_audit.get("tensor_bytes")
        != int(runner.legacy.CHECKPOINT["tensor_bytes"])
        or checkpoint_audit.get("dtype_counts")
        != dict(runner.legacy.CHECKPOINT["state_dtypes"])
    ):
        raise ValueError("recorded checkpoint structural audit changed")
    if (
        model_audit.get("strict_load") is not True
        or model_audit.get("missing_keys") != 0
        or model_audit.get("unexpected_keys") != 0
        or model_audit.get("parameter_count")
        != int(runner.legacy.CHECKPOINT["parameters"])
        or model_audit.get("buffer_elements")
        != int(runner.legacy.CHECKPOINT["buffers"])
        or model_audit.get("state_keys") != int(runner.legacy.CHECKPOINT["state_keys"])
        or model_audit.get("device") != "cpu"
        or model_audit.get("eval_mode") is not True
        or model_audit.get("forward_performed") is not False
    ):
        raise ValueError("recorded CPU strict-model audit changed")
    if stable_json(immutable.get("adapter_sources")) != stable_json(adapter_sources):
        raise ValueError("immutable adapter-source binding changed")
    runtime = _verify_contract_digest(immutable.get("runtime"), "runtime")
    return {
        "environment": environment,
        "source": source,
        "assets": assets,
        "adapter_sources": adapter_sources,
        "artifact_ignore": artifact_ignore,
        "checkpoint_audit": dict(checkpoint_audit),
        "model_audit": dict(model_audit),
        "runtime": dict(runtime),
    }


def _selection_for_mode(
    release: CanonicalRelease,
    mode: str,
) -> tuple[SelectionSpec, list[dict[str, Any]]]:
    if mode == "formal":
        spec, selected = runner.select_mode_inputs(
            release,
            mode="formal",
            per_condition_limit=None,
            sample_id=None,
        )
        expected_images = FORMAL_IMAGES
        expected_ids_hash = FORMAL_SELECTED_IDS_SHA256
        expected_rows_hash = FORMAL_SELECTED_ROWS_SHA256
    elif mode == "smoke":
        spec, selected = runner.select_mode_inputs(
            release,
            mode="smoke",
            per_condition_limit=runner.DEFAULT_SMOKE_LIMIT,
            sample_id=None,
        )
        expected_images = SMOKE_IMAGES
        expected_ids_hash = SMOKE_SELECTED_IDS_SHA256
        expected_rows_hash = SMOKE_SELECTED_ROWS_SHA256
    else:
        raise ValueError("analysis supports only frozen formal/smoke runs")
    ids_hash = _fingerprint([str(row["sample_id"]) for row in selected])
    rows_hash = _fingerprint(selected)
    if (
        len(selected) != expected_images
        or ids_hash != expected_ids_hash
        or rows_hash != expected_rows_hash
    ):
        raise ValueError(f"frozen {mode} selection hash changed")
    return spec, selected


def load_run(
    *,
    repo_root: Path,
    results_dir: Path,
    artifacts_dir: Path,
    run_id: str,
    expected_mode: str,
    imlvit_root: Path,
    checkpoint_path: Path,
    verify_artifacts: bool = True,
) -> RunBundle:
    run_dir = _safe_run_dir(results_dir, run_id, "IML-ViT run directory")
    artifact_root = _safe_run_dir(
        artifacts_dir,
        run_id,
        "IML-ViT artifact directory",
    )
    manifest_path = _safe_exact_file(
        run_dir / "manifest.json",
        parent=run_dir,
        label="run manifest",
    )
    expected_path = _safe_exact_file(
        run_dir / "expected_inputs.jsonl",
        parent=run_dir,
        label="expected inputs",
    )
    results_path = _safe_exact_file(
        run_dir / "results.jsonl",
        parent=run_dir,
        label="results",
    )
    summary_path = _safe_exact_file(
        run_dir / "summary.json",
        parent=run_dir,
        label="runtime summary",
    )
    manifest = runner._load_json_object_strict(manifest_path)
    summary = runner._load_json_object_strict(summary_path)
    expected_rows = runner._read_jsonl_strict(expected_path)
    physical_results = runner._read_jsonl_strict(results_path)
    if (
        manifest.get("schema_version") != runner.RUN_MANIFEST_SCHEMA
        or manifest.get("run_id") != run_id
        or manifest.get("status") != "complete"
    ):
        raise ValueError("IML-ViT manifest is not a complete frozen run")
    immutable = manifest.get("immutable")
    if not isinstance(immutable, dict):
        raise ValueError("IML-ViT immutable run config is missing")
    fingerprint = manifest.get("fingerprint")
    if not isinstance(fingerprint, str) or fingerprint != _fingerprint(immutable):
        raise ValueError("IML-ViT run fingerprint changed")
    if (
        immutable.get("schema_version") != runner.RUN_CONFIG_SCHEMA
        or immutable.get("run_id") != run_id
        or immutable.get("mode") != expected_mode
    ):
        raise ValueError("IML-ViT immutable run identity changed")

    dataset_manifest = (repo_root / runner.DEFAULT_DATASET_MANIFEST).resolve()
    dataset = manifest.get("dataset")
    if not isinstance(dataset, Mapping):
        raise ValueError("IML-ViT dataset envelope is missing")
    _safe_recorded_repo_path(
        dataset.get("manifest_path"),
        repo_root=repo_root,
        expected=dataset_manifest,
        label="dataset manifest path",
    )
    release = load_canonical_release(
        repo_root,
        dataset_manifest,
        verify_files=True,
    )
    if dataset.get("manifest_sha256") != release.manifest_sha256:
        raise ValueError("IML-ViT dataset manifest hash changed")
    selection_spec, selected = _selection_for_mode(release, expected_mode)
    if stable_json(expected_rows) != stable_json(selected):
        raise ValueError("IML-ViT expected inputs changed")
    if dataset.get("expected_inputs_sha256") != sha256_file(expected_path):
        raise ValueError("IML-ViT expected-input hash changed")
    contract = build_run_dataset_contract(
        release,
        selection_spec,
        selected,
        score_spec=None,
    )
    if stable_json(dataset.get("contract")) != stable_json(contract.as_dict()):
        raise ValueError("IML-ViT recorded dataset contract changed")
    if stable_json(immutable.get("dataset_contract")) != stable_json(
        contract.as_dict()
    ):
        raise ValueError("IML-ViT immutable dataset contract changed")
    if not (
        dataset.get("selected_images") == len(selected)
        and dataset.get("t1_applicable_images") == 0
        and dataset.get("t2_applicable_images") == len(selected)
        and dataset.get("fullframe_selected_images") == 0
    ):
        raise ValueError("IML-ViT dataset task coverage changed")

    provenance = _validate_recorded_provenance(
        immutable,
        repo_root=repo_root,
        imlvit_root=imlvit_root,
        checkpoint_path=checkpoint_path,
    )
    del provenance
    expected_immutable = runner.build_immutable_run_config(
        repo_root=repo_root,
        run_id=run_id,
        mode=expected_mode,
        dataset_contract=contract.as_dict(),
        selected=selected,
        cpu_preflight=immutable["cpu_preflight"],
        runtime=immutable["runtime"],
        results_path=results_path,
        expected_inputs_path=expected_path,
        summary_path=summary_path,
        artifact_root=artifact_root,
    )
    if stable_json(immutable) != stable_json(expected_immutable):
        raise ValueError("IML-ViT immutable run config cannot be reconstructed")

    outputs = manifest.get("outputs")
    if not isinstance(outputs, Mapping):
        raise ValueError("IML-ViT manifest outputs are missing")
    expected_output_paths = immutable["outputs"]
    for key in (
        "results_path",
        "expected_inputs_path",
        "summary_path",
        "artifact_root",
    ):
        if outputs.get(key) != expected_output_paths[key]:
            raise ValueError(f"IML-ViT output path changed at {key}")
    if outputs.get("results_sha256") != sha256_file(results_path) or outputs.get(
        "summary_sha256"
    ) != sha256_file(summary_path):
        raise ValueError("IML-ViT manifest output hash changed")

    runner._validate_physical_attempt_history(selected, physical_results)
    latest = index_latest_attempts(
        selected,
        physical_results,
        run_id=run_id,
        run_manifest_fingerprint=fingerprint,
        score_spec=None,
    )
    coverage = summarize_coverage(latest)
    coverage.require_complete()
    selected_by_id = {str(row["sample_id"]): row for row in selected}
    for attempt in physical_results:
        input_row = selected_by_id[str(attempt["sample_id"])]
        runner._validate_runner_attempt(
            attempt,
            input_row=input_row,
            repo_root=repo_root,
            artifact_root=artifact_root,
            run_id=run_id,
            run_manifest_fingerprint=fingerprint,
            verify_artifacts=verify_artifacts and attempt.get("status") == "ok",
            recompute_preprocess=verify_artifacts and attempt.get("status") == "ok",
        )
    latest_results = tuple(
        dict(latest.latest_by_sample_id[str(row["sample_id"])]) for row in selected
    )
    inventory = runner.validate_artifact_inventory(
        artifact_root=artifact_root,
        selected=selected,
        latest_by_sample_id=latest.latest_by_sample_id,
    )
    if stable_json(outputs.get("artifact_inventory")) != stable_json(inventory):
        raise ValueError("IML-ViT manifest artifact inventory changed")
    history = runner._validate_physical_attempt_history(selected, physical_results)
    execution = manifest.get("execution")
    if (
        not isinstance(execution, Mapping)
        or execution.get("physical_result_rows") != len(physical_results)
        or execution.get("latest_result_rows") != len(latest.latest_by_sample_id)
        or execution.get("superseded_attempts") != latest.superseded_attempts
        or execution.get("new_errors") != 0
        or not isinstance(execution.get("new_successes"), int)
        or not isinstance(execution.get("resume_skips"), int)
        or execution["new_successes"] < 0
        or execution["resume_skips"] < 0
        or execution["new_successes"] + execution["resume_skips"] != len(selected)
    ):
        raise ValueError("IML-ViT final execution accounting changed")
    if (
        summary.get("schema_version") != runner.RUNTIME_SUMMARY_SCHEMA
        or summary.get("run_id") != run_id
        or summary.get("run_manifest_fingerprint") != fingerprint
        or summary.get("status") != "complete"
        or summary.get("mode") != expected_mode
        or summary.get("score_spec") is not None
        or summary.get("task_scope") != runner.TASK_SCOPE
        or summary.get("t2_spec") != runner.T2_SPEC
        or summary.get("dataset_contract") != contract.as_dict()
        or summary.get("coverage") != coverage.as_dict()
        or summary.get("attempt_history") != history
        or summary.get("artifact_inventory") != inventory
        or summary.get("scientific_metrics") is not None
        or summary.get("scientific_metrics_owner") != "analyze_imlvit_balanced.py"
    ):
        raise ValueError("IML-ViT runtime summary changed")
    if any(
        field in result
        for result in latest_results
        for field in runner._FORBIDDEN_T1_TOP_LEVEL
    ):
        raise ValueError("IML-ViT latest results expose forbidden T1 fields")
    evidence = {
        "manifest_sha256": sha256_file(manifest_path),
        "expected_inputs_sha256": sha256_file(expected_path),
        "results_sha256": sha256_file(results_path),
        "runtime_summary_sha256": sha256_file(summary_path),
        "artifact_inventory_sha256": inventory["inventory_sha256"],
    }
    return RunBundle(
        repo_root=repo_root,
        run_id=run_id,
        mode=expected_mode,
        run_dir=run_dir,
        artifact_root=artifact_root,
        manifest_path=manifest_path,
        expected_path=expected_path,
        results_path=results_path,
        summary_path=summary_path,
        manifest=manifest,
        immutable=immutable,
        fingerprint=fingerprint,
        release=release,
        selection_spec=selection_spec,
        selected=tuple(selected),
        contract=contract,
        physical_results=tuple(physical_results),
        latest_results=latest_results,
        summary=summary,
        coverage=coverage.as_dict(),
        history=history,
        evidence_snapshot=evidence,
    )


def _independent_preprocess(
    image_path: Path,
) -> tuple[dict[str, Any], np.ndarray, tuple[int, int], tuple[int, int]]:
    import albumentations as albu
    import cv2

    with image_path.open("rb") as handle:
        with Image.open(handle) as opened:
            decoded = np.asarray(opened.convert("RGB"), dtype=np.uint8)
            decoder_format = opened.format
    if decoded.ndim != 3 or decoded.shape[2] != 3 or decoded.dtype != np.uint8:
        raise ValueError("independent IML-ViT decode contract changed")
    native_height, native_width = decoded.shape[:2]
    if max(native_height, native_width) > runner.legacy.MODEL_INPUT_SIZE:
        transform = albu.LongestMaxSize(
            max_size=runner.legacy.MODEL_INPUT_SIZE,
            interpolation=cv2.INTER_LINEAR,
            always_apply=True,
        )
        resized = np.asarray(transform(image=decoded)["image"], dtype=np.uint8)
        resize_policy = "albumentations_longest_max_size_downscale_only"
    else:
        resized = decoded
        resize_policy = "none_image_within_1024_limit"
    resized_height, resized_width = resized.shape[:2]
    canvas = np.zeros(
        (
            runner.legacy.MODEL_INPUT_SIZE,
            runner.legacy.MODEL_INPUT_SIZE,
            3,
        ),
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
        "model_canvas_size": [
            runner.legacy.MODEL_INPUT_SIZE,
            runner.legacy.MODEL_INPUT_SIZE,
        ],
        "resize_policy": resize_policy,
        "resize_interpolation": "cv2.INTER_LINEAR_via_albumentations",
        "resize_scale_x": resized_width / native_width,
        "resize_scale_y": resized_height / native_height,
        "aspect_ratio_preserved_with_rounding": True,
        "padding": {
            "placement": "top_left",
            "right_pixels": runner.legacy.MODEL_INPUT_SIZE - resized_width,
            "bottom_pixels": runner.legacy.MODEL_INPUT_SIZE - resized_height,
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
        "tensor_sha256": runner._array_sha256(tensor),
    }
    return (
        evidence,
        tensor,
        (native_width, native_height),
        (resized_width, resized_height),
    )


def audit_artifacts(bundle: RunBundle) -> dict[str, Any]:
    total_files = 0
    total_bytes = 0
    exact_half_pixels = 0
    exact_half_images = 0
    raw_sigmoid_max_abs = 0.0
    native_restore_max_abs = 0.0
    strict_masks = 0
    preprocessing_replays = 0
    localization_replays = 0
    selected_by_id = {str(row["sample_id"]): row for row in bundle.selected}
    for index, result in enumerate(bundle.latest_results, start=1):
        sample_id = str(result["sample_id"])
        input_row = selected_by_id[sample_id]
        width = int(input_row["width"])
        height = int(input_row["height"])
        paths = runner.artifact_paths(bundle.artifact_root, sample_id)
        for path in paths.values():
            if not path.is_file() or path.is_symlink():
                raise FileNotFoundError(path)
            total_files += 1
            total_bytes += path.stat().st_size
        raw = np.load(paths["raw_logits_model"], allow_pickle=False)
        model_score = np.load(paths["score_map_model"], allow_pickle=False)
        native_score = np.load(paths["score_map_native"], allow_pickle=False)
        for label, array, shape, probability in (
            (
                "raw logits",
                raw,
                (
                    runner.legacy.MODEL_INPUT_SIZE,
                    runner.legacy.MODEL_INPUT_SIZE,
                ),
                False,
            ),
            (
                "model score",
                model_score,
                (
                    runner.legacy.MODEL_INPUT_SIZE,
                    runner.legacy.MODEL_INPUT_SIZE,
                ),
                True,
            ),
            ("native score", native_score, (height, width), True),
        ):
            if (
                not isinstance(array, np.ndarray)
                or array.dtype != np.float32
                or array.shape != shape
                or not array.flags.c_contiguous
                or not np.isfinite(array).all()
            ):
                raise ValueError(f"{sample_id} independent {label} audit failed")
            if probability and (float(array.min()) < 0.0 or float(array.max()) > 1.0):
                raise ValueError(f"{sample_id} {label} falls outside [0,1]")
        independent_sigmoid = _sigmoid_float32(raw)
        sigmoid_diff = float(
            np.max(
                np.abs(
                    independent_sigmoid.astype(np.float64)
                    - model_score.astype(np.float64)
                )
            )
        )
        raw_sigmoid_max_abs = max(raw_sigmoid_max_abs, sigmoid_diff)
        if not np.allclose(
            independent_sigmoid,
            model_score,
            rtol=0.0,
            atol=SIGMOID_ABSOLUTE_TOLERANCE,
        ):
            raise ValueError(f"{sample_id} independent sigmoid replay changed")
        resized_width, resized_height = (
            int(value) for value in result["model_valid_content_size"]
        )
        valid_score = model_score[:resized_height, :resized_width]
        independent_native = _bilinear_align_corners_false(
            valid_score,
            width=width,
            height=height,
        )
        restore_diff = float(
            np.max(
                np.abs(
                    independent_native.astype(np.float64)
                    - native_score.astype(np.float64)
                )
            )
        )
        native_restore_max_abs = max(native_restore_max_abs, restore_diff)
        if not np.allclose(
            independent_native,
            native_score,
            rtol=1e-6,
            atol=NATIVE_RESTORE_ABSOLUTE_TOLERANCE,
        ):
            raise ValueError(f"{sample_id} independent native restoration changed")
        image_exact_half_pixels = int(
            np.count_nonzero(native_score == np.float32(runner.MASK_THRESHOLD))
        )
        exact_half_pixels += image_exact_half_pixels
        exact_half_images += int(image_exact_half_pixels > 0)
        with Image.open(paths["mask_native"]) as opened:
            if opened.mode != "L" or opened.size != (width, height):
                raise ValueError(f"{sample_id} native mask contract changed")
            mask = np.asarray(opened, dtype=np.uint8)
        expected_mask = np.where(
            native_score > np.float32(runner.MASK_THRESHOLD),
            np.uint8(255),
            np.uint8(0),
        )
        if not np.array_equal(mask, expected_mask):
            raise ValueError(f"{sample_id} strict threshold mask changed")
        strict_masks += 1

        image_path = runner.verified_input_path(input_row, bundle.repo_root)
        evidence, tensor, native_size, resized_size = _independent_preprocess(
            image_path
        )
        del tensor
        if (
            evidence != result.get("preprocess")
            or native_size != (width, height)
            or resized_size != (resized_width, resized_height)
        ):
            raise ValueError(f"{sample_id} independent preprocessing changed")
        preprocessing_replays += 1
        target = runner.load_ground_truth(input_row, bundle.repo_root)
        if target is None:
            raise ValueError(f"{sample_id} has no T2 target")
        target_native = np.asarray(target, dtype=bool)
        target_model = runner.legacy.model_space_target(
            target_native,
            resized_width=resized_width,
            resized_height=resized_height,
        )
        include_ap = str(input_row["condition"]) != "real"
        localization = {
            "model_1024": binary_pixel_metrics_strict(
                model_score[:resized_height, :resized_width],
                target_model,
                runner.MASK_THRESHOLD,
                include_ap=include_ap,
            ),
            "native": binary_pixel_metrics_strict(
                native_score,
                target_native,
                runner.MASK_THRESHOLD,
                include_ap=include_ap,
            ),
        }
        if stable_json(localization) != stable_json(result.get("localization")):
            raise ValueError(f"{sample_id} independent localization changed")
        localization_replays += 1
        del (
            raw,
            model_score,
            native_score,
            independent_sigmoid,
            independent_native,
            mask,
            expected_mask,
            target_native,
            target_model,
        )
        if index % 100 == 0 or index == len(bundle.latest_results):
            print(
                f"[IML-ViT artifact audit {index}/"
                f"{len(bundle.latest_results)}] {sample_id}",
                flush=True,
            )
    if (
        total_files != len(bundle.latest_results) * 4
        or strict_masks != len(bundle.latest_results)
        or preprocessing_replays != len(bundle.latest_results)
        or localization_replays != len(bundle.latest_results)
    ):
        raise ValueError("IML-ViT independent artifact coverage changed")
    return {
        "status": "passed",
        "images": len(bundle.latest_results),
        "artifact_files": total_files,
        "artifact_bytes": total_bytes,
        "preprocessing_replays": preprocessing_replays,
        "raw_logit_sigmoid_replays": len(bundle.latest_results),
        "raw_logit_sigmoid_max_abs_difference": raw_sigmoid_max_abs,
        "raw_logit_sigmoid_absolute_tolerance": SIGMOID_ABSOLUTE_TOLERANCE,
        "native_restore_replays": len(bundle.latest_results),
        "native_restore_max_abs_difference": native_restore_max_abs,
        "native_restore_absolute_tolerance": NATIVE_RESTORE_ABSOLUTE_TOLERANCE,
        "strict_threshold_masks_replayed": strict_masks,
        "localization_metrics_replayed": localization_replays,
        "native_pixels_exactly_0_5": exact_half_pixels,
        "native_images_with_pixels_exactly_0_5": exact_half_images,
        "official_t2_threshold_operator": runner.MASK_THRESHOLD_OPERATOR,
        "shared_t2_threshold_operator": runner.SHARED_T2_THRESHOLD_OPERATOR,
        "shared_t2_operator_equivalent_to_official": False,
        "operator_non_equivalence_observed_on_formal_artifacts": (
            exact_half_pixels > 0
        ),
        "persisted_artifact_hashes_exact": True,
    }


def _native_map_loader(
    bundle: RunBundle,
    load_audit: NativeMapLoadAudit,
) -> Callable[[Mapping[str, Any], Mapping[str, Any]], np.ndarray]:
    selected_by_id = {str(row["sample_id"]): row for row in bundle.selected}

    def load_native_score_map(
        input_row: Mapping[str, Any],
        result_row: Mapping[str, Any],
    ) -> np.ndarray:
        sample_id = str(input_row.get("sample_id"))
        selected = selected_by_id.get(sample_id)
        if selected is None or stable_json(input_row) != stable_json(selected):
            raise ValueError("native score-map input identity changed")
        if result_row.get("sample_id") != sample_id:
            raise ValueError("native score-map result identity changed")
        paths = runner.artifact_paths(bundle.artifact_root, sample_id)
        expected_path = runner._resolve_exact_artifact(
            result_row.get("score_map_path"),
            repo_root=bundle.repo_root,
            expected=paths["score_map_native"],
            label=f"{sample_id} native map",
        )
        score_map = np.load(
            expected_path,
            mmap_mode="r",
            allow_pickle=False,
        )
        if (
            score_map.dtype != np.float32
            or score_map.shape != (int(input_row["height"]), int(input_row["width"]))
            or not np.isfinite(score_map).all()
            or float(score_map.min()) < 0.0
            or float(score_map.max()) > 1.0
            or sha256_file(expected_path) != result_row.get("score_map_sha256")
        ):
            raise ValueError(f"{sample_id} native map contract changed")
        exact_half_pixels = int(
            np.count_nonzero(score_map == np.float32(runner.MASK_THRESHOLD))
        )
        load_audit.maps_loaded += 1
        load_audit.native_pixels_exactly_at_threshold += exact_half_pixels
        load_audit.native_images_with_pixels_exactly_at_threshold += int(
            exact_half_pixels > 0
        )
        return score_map

    return load_native_score_map


def recompute_metrics(bundle: RunBundle) -> dict[str, Any]:
    if bundle.mode != "formal" or len(bundle.selected) != FORMAL_IMAGES:
        raise ValueError("IML-ViT metrics require the frozen formal run")
    load_audit = NativeMapLoadAudit()
    t2 = summarize_balanced250_t2(
        bundle.release.inputs,
        bundle.latest_results,
        repo_root=bundle.repo_root,
        run_id=bundle.run_id,
        run_manifest_fingerprint=bundle.fingerprint,
        run_dataset_contract=bundle.contract,
        load_native_score_map=_native_map_loader(bundle, load_audit),
        score_map_name="imlvit_native_sigmoid_probability_map",
        threshold=runner.MASK_THRESHOLD,
        threshold_operator=runner.SHARED_T2_THRESHOLD_OPERATOR,
        iterations=BOOTSTRAP_ITERATIONS,
        seed=BOOTSTRAP_SEED,
    )
    if (
        t2.get("schema_version") != T2_METRICS_SCHEMA_VERSION
        or t2.get("coverage", {}).get("is_complete") is not True
        or t2.get("coverage", {}).get("native_maps_evaluated") != FORMAL_IMAGES
        or load_audit.maps_loaded != FORMAL_IMAGES
    ):
        raise ValueError("IML-ViT shared T2 metrics are incomplete")
    return {
        "schema_version": METRICS_SCHEMA_VERSION,
        "method": runner.MODEL_NAME,
        "model_slug": runner.MODEL_SLUG,
        "run_id": bundle.run_id,
        "run_manifest_fingerprint": bundle.fingerprint,
        "task_scope": {
            "t1": "not_applicable_no_native_image_classification_head",
            "t2": "native_pixel_localization",
            "fullframe": "not_selected_T1_and_T2_not_applicable",
            "map_statistic_promoted_to_t1": False,
        },
        "formal_images_t1": 0,
        "formal_images_t2": FORMAL_IMAGES,
        "official_t2_threshold": runner.MASK_THRESHOLD,
        "official_t2_threshold_operator": runner.MASK_THRESHOLD_OPERATOR,
        "shared_t2_threshold": runner.MASK_THRESHOLD,
        "shared_t2_threshold_operator": runner.SHARED_T2_THRESHOLD_OPERATOR,
        "native_pixels_exactly_at_threshold": (
            load_audit.native_pixels_exactly_at_threshold
        ),
        "native_images_with_pixels_exactly_at_threshold": (
            load_audit.native_images_with_pixels_exactly_at_threshold
        ),
        "operator_equivalence_checked": True,
        "shared_t2_operator_equivalent_to_official": False,
        "operator_non_equivalence_observed_on_formal_artifacts": (
            load_audit.native_pixels_exactly_at_threshold > 0
        ),
        "t1": None,
        "t2": t2,
        "sources": {
            "dataset_manifest_sha256": bundle.release.manifest_sha256,
            "results_sha256": sha256_file(bundle.results_path),
            "manifest_sha256": sha256_file(bundle.manifest_path),
            "summary_sha256": sha256_file(bundle.summary_path),
            "analyzer_sources": analyzer_source_contract(bundle.repo_root),
        },
    }


def _result_projection(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in row.items()
        if key not in _RUN_DEPENDENT_RESULT_FIELDS
    }


def _immutable_smoke_projection(value: Mapping[str, Any]) -> dict[str, Any]:
    projected = dict(value)
    projected.pop("run_id", None)
    projected.pop("outputs", None)
    return projected


def _files_equal(first: Path, second: Path) -> bool:
    if first.stat().st_size != second.stat().st_size:
        return False
    with first.open("rb") as left, second.open("rb") as right:
        while True:
            left_chunk = left.read(4 * 1024 * 1024)
            right_chunk = right.read(4 * 1024 * 1024)
            if left_chunk != right_chunk:
                return False
            if not left_chunk:
                return True


def compare_computational_results(
    first: RunBundle,
    second: RunBundle,
) -> dict[str, Any]:
    if (
        first.mode != "smoke"
        or second.mode != "smoke"
        or len(first.selected) != SMOKE_IMAGES
        or len(second.selected) != SMOKE_IMAGES
    ):
        raise ValueError("IML-ViT smoke comparison requires two 20-image runs")
    if stable_json(first.selected) != stable_json(second.selected):
        raise ValueError("IML-ViT smoke selections differ")
    if stable_json(_immutable_smoke_projection(first.immutable)) != stable_json(
        _immutable_smoke_projection(second.immutable)
    ):
        raise ValueError("IML-ViT smoke immutable configurations differ")
    files_compared = 0
    result_rows_compared = 0
    first_by_id = {str(row["sample_id"]): row for row in first.latest_results}
    second_by_id = {str(row["sample_id"]): row for row in second.latest_results}
    if set(first_by_id) != set(second_by_id):
        raise ValueError("IML-ViT smoke result IDs differ")
    for input_row in first.selected:
        sample_id = str(input_row["sample_id"])
        left = first_by_id[sample_id]
        right = second_by_id[sample_id]
        if stable_json(_result_projection(left)) != stable_json(
            _result_projection(right)
        ):
            raise ValueError(f"IML-ViT smoke result changed for {sample_id}")
        left_paths = runner.artifact_paths(first.artifact_root, sample_id)
        right_paths = runner.artifact_paths(second.artifact_root, sample_id)
        if set(left_paths) != set(right_paths):
            raise ValueError("IML-ViT smoke artifact key sets differ")
        for key in left_paths:
            if not _files_equal(left_paths[key], right_paths[key]):
                raise ValueError(
                    f"IML-ViT smoke artifact {key} changed for {sample_id}"
                )
            files_compared += 1
        result_rows_compared += 1
    if files_compared != SMOKE_IMAGES * 4:
        raise ValueError("IML-ViT smoke artifact comparison coverage changed")
    return {
        "status": "passed",
        "computational_outputs_exact": True,
        "selected_inputs_exact": True,
        "images_compared": result_rows_compared,
        "result_projections_compared_exact": result_rows_compared,
        "artifact_files_compared_byte_exact": files_compared,
        "raw_logits_compared_exact": SMOKE_IMAGES,
        "model_probability_maps_compared_exact": SMOKE_IMAGES,
        "native_probability_maps_compared_exact": SMOKE_IMAGES,
        "native_masks_compared_exact": SMOKE_IMAGES,
        "ignored_nondeterministic_fields": sorted(_RUN_DEPENDENT_RESULT_FIELDS),
    }


def _smoke_comparison_output(
    results_dir: Path,
    first_run_id: str,
    second_run_id: str,
) -> Path:
    report_root = results_dir / "_reports"
    _reject_symlink_components(report_root, "IML-ViT report directory")
    if report_root.exists() and (report_root.is_symlink() or not report_root.is_dir()):
        raise ValueError("IML-ViT report directory is unsafe")
    report_root.mkdir(parents=True, exist_ok=True)
    return report_root / f"{first_run_id}_vs_{second_run_id}.smoke_comparison.json"


def compare_smoke_runs(
    *,
    repo_root: Path,
    results_dir: Path,
    artifacts_dir: Path,
    first_run_id: str,
    second_run_id: str,
    imlvit_root: Path,
    checkpoint_path: Path,
    output_path: Path | None = None,
) -> dict[str, Any]:
    if (
        first_run_id != DEFAULT_SMOKE_RUN_ID_A
        or second_run_id != DEFAULT_SMOKE_RUN_ID_B
    ):
        raise ValueError("IML-ViT smoke IDs are frozen at A then B")
    first = load_run(
        repo_root=repo_root,
        results_dir=results_dir,
        artifacts_dir=artifacts_dir,
        run_id=first_run_id,
        expected_mode="smoke",
        imlvit_root=imlvit_root,
        checkpoint_path=checkpoint_path,
    )
    second = load_run(
        repo_root=repo_root,
        results_dir=results_dir,
        artifacts_dir=artifacts_dir,
        run_id=second_run_id,
        expected_mode="smoke",
        imlvit_root=imlvit_root,
        checkpoint_path=checkpoint_path,
    )
    comparison = compare_computational_results(first, second)
    expected_output = _smoke_comparison_output(
        results_dir, first_run_id, second_run_id
    ).resolve()
    requested_output = expected_output if output_path is None else output_path.resolve()
    if requested_output != expected_output:
        raise ValueError("IML-ViT smoke comparison output path is frozen")
    report = {
        "schema_version": SMOKE_COMPARISON_SCHEMA_VERSION,
        "status": "passed",
        "first_run_id": first_run_id,
        "second_run_id": second_run_id,
        "selected_ids_sha256": SMOKE_SELECTED_IDS_SHA256,
        "selected_rows_sha256": SMOKE_SELECTED_ROWS_SHA256,
        "comparison": comparison,
        "evidence": {
            "first": dict(first.evidence_snapshot),
            "second": dict(second.evidence_snapshot),
        },
        "compared_at": utc_now(),
    }
    digest = _write_json_verified(requested_output, report)
    return {
        **report,
        "path": repo_relative(requested_output, repo_root),
        "sha256": digest,
    }


def replay_model(
    bundle: RunBundle,
    *,
    imlvit_root: Path,
    checkpoint_path: Path,
) -> dict[str, Any]:
    if bundle.mode != "formal" or len(bundle.selected) != FORMAL_IMAGES:
        raise ValueError("IML-ViT fresh replay requires formal 1025 coverage")
    fresh_preflight = runner.run_cpu_preflight(
        repo_root=bundle.repo_root,
        imlvit_root=imlvit_root,
        checkpoint_path=checkpoint_path,
    )
    if fresh_preflight != bundle.immutable.get("cpu_preflight"):
        raise ValueError("IML-ViT fresh CPU preflight differs from recorded")
    recorded_runtime = bundle.immutable.get("runtime")
    if not isinstance(recorded_runtime, Mapping):
        raise ValueError("IML-ViT recorded runtime is invalid")
    device_text = recorded_runtime.get("device")
    if not isinstance(device_text, str):
        raise ValueError("IML-ViT recorded device is invalid")
    device, fresh_runtime = runner.configure_runtime(device_text)
    if fresh_runtime != recorded_runtime:
        raise ValueError("IML-ViT fresh runtime differs from recorded runtime")

    import torch

    model = None
    raw_logits_compared = 0
    model_maps_compared = 0
    native_maps_compared = 0
    masks_compared = 0
    preprocess_compared = 0
    localization_compared = 0
    exact_half_pixels = 0
    exact_half_images = 0
    try:
        model, loaded_device = runner.legacy.load_model(
            imlvit_root=imlvit_root,
            checkpoint_path=checkpoint_path,
            device_name=str(device),
        )
        if str(loaded_device) != str(device):
            raise ValueError("IML-ViT fresh model loaded on wrong device")
        by_id = {str(row["sample_id"]): row for row in bundle.latest_results}
        for index, input_row in enumerate(bundle.selected, start=1):
            sample_id = str(input_row["sample_id"])
            persisted = by_id[sample_id]
            image_path = runner.verified_input_path(input_row, bundle.repo_root)
            image, native_size, resized_size, preprocess = (
                runner._preprocess_with_audit(image_path)
            )
            if preprocess != persisted.get("preprocess"):
                raise ValueError(f"{sample_id} fresh preprocessing changed")
            preprocess_compared += 1
            width, height = native_size
            resized_width, resized_height = resized_size
            (
                raw_logits,
                model_score,
                native_score,
                _,
                _,
            ) = runner.legacy.infer_one(
                model,
                loaded_device,
                image,
                native_width=width,
                native_height=height,
                resized_width=resized_width,
                resized_height=resized_height,
            )
            fresh_arrays = {
                "raw_logits_model": np.ascontiguousarray(raw_logits, dtype=np.float32),
                "score_map_model": np.ascontiguousarray(model_score, dtype=np.float32),
                "score_map_native": np.ascontiguousarray(
                    native_score, dtype=np.float32
                ),
            }
            paths = runner.artifact_paths(bundle.artifact_root, sample_id)
            for key, fresh in fresh_arrays.items():
                recorded = np.load(paths[key], allow_pickle=False)
                if not np.array_equal(fresh, recorded):
                    raise ValueError(f"{sample_id} fresh {key} is not bit-exact")
                if key == "raw_logits_model":
                    raw_logits_compared += 1
                elif key == "score_map_model":
                    model_maps_compared += 1
                else:
                    native_maps_compared += 1
            image_exact_half_pixels = int(
                np.count_nonzero(
                    fresh_arrays["score_map_native"]
                    == np.float32(runner.MASK_THRESHOLD)
                )
            )
            exact_half_pixels += image_exact_half_pixels
            exact_half_images += int(image_exact_half_pixels > 0)
            fresh_mask = np.where(
                fresh_arrays["score_map_native"] > runner.MASK_THRESHOLD,
                np.uint8(255),
                np.uint8(0),
            )
            with Image.open(paths["mask_native"]) as opened:
                recorded_mask = np.asarray(opened, dtype=np.uint8)
            if not np.array_equal(fresh_mask, recorded_mask):
                raise ValueError(f"{sample_id} fresh native mask changed")
            masks_compared += 1
            target = runner.load_ground_truth(input_row, bundle.repo_root)
            if target is None:
                raise ValueError(f"{sample_id} fresh T2 target is absent")
            target_native = np.asarray(target, dtype=bool)
            target_model = runner.legacy.model_space_target(
                target_native,
                resized_width=resized_width,
                resized_height=resized_height,
            )
            include_ap = str(input_row["condition"]) != "real"
            fresh_localization = {
                "model_1024": binary_pixel_metrics_strict(
                    fresh_arrays["score_map_model"][:resized_height, :resized_width],
                    target_model,
                    runner.MASK_THRESHOLD,
                    include_ap=include_ap,
                ),
                "native": binary_pixel_metrics_strict(
                    fresh_arrays["score_map_native"],
                    target_native,
                    runner.MASK_THRESHOLD,
                    include_ap=include_ap,
                ),
            }
            if stable_json(fresh_localization) != stable_json(
                persisted.get("localization")
            ):
                raise ValueError(f"{sample_id} fresh localization changed")
            localization_compared += 1
            del (
                image,
                raw_logits,
                model_score,
                native_score,
                fresh_arrays,
                fresh_mask,
                recorded_mask,
                target_native,
                target_model,
            )
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
            if index % 50 == 0 or index == FORMAL_IMAGES:
                print(
                    f"[IML-ViT fresh replay {index}/{FORMAL_IMAGES}] "
                    f"exact {sample_id}",
                    flush=True,
                )
    finally:
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if (
        raw_logits_compared != FORMAL_IMAGES
        or model_maps_compared != FORMAL_IMAGES
        or native_maps_compared != FORMAL_IMAGES
        or masks_compared != FORMAL_IMAGES
        or preprocess_compared != FORMAL_IMAGES
        or localization_compared != FORMAL_IMAGES
    ):
        raise ValueError("IML-ViT fresh replay coverage changed")
    return {
        "schema_version": FRESH_REPLAY_SCHEMA_VERSION,
        "status": "passed",
        "publishable": True,
        "mode": "fresh_full_model_selection_exact_replay",
        "selected_images_replayed": FORMAL_IMAGES,
        "preprocessing_compared_exact": preprocess_compared,
        "raw_logits_compared_bit_exact": raw_logits_compared,
        "model_probability_maps_compared_bit_exact": model_maps_compared,
        "native_probability_maps_compared_bit_exact": native_maps_compared,
        "native_masks_compared_exact": masks_compared,
        "localization_metrics_compared_exact": localization_compared,
        "native_pixels_exactly_0_5": exact_half_pixels,
        "native_images_with_pixels_exactly_0_5": exact_half_images,
        "official_t2_threshold_operator": runner.MASK_THRESHOLD_OPERATOR,
        "shared_t2_reducer_applied_during_fresh_replay": False,
        "recorded_device_tolerance": 0.0,
        "device": device_text,
        "completed_at": utc_now(),
    }


def _verify_bundle_unchanged(bundle: RunBundle) -> None:
    current = {
        "manifest_sha256": sha256_file(bundle.manifest_path),
        "expected_inputs_sha256": sha256_file(bundle.expected_path),
        "results_sha256": sha256_file(bundle.results_path),
        "runtime_summary_sha256": sha256_file(bundle.summary_path),
    }
    for key, expected in bundle.evidence_snapshot.items():
        if key == "artifact_inventory_sha256":
            continue
        if current.get(key) != expected:
            raise ValueError(f"IML-ViT evidence changed during audit: {key}")
    latest = index_latest_attempts(
        bundle.selected,
        bundle.physical_results,
        run_id=bundle.run_id,
        run_manifest_fingerprint=bundle.fingerprint,
        score_spec=None,
    )
    inputs_by_id = {str(row["sample_id"]): row for row in bundle.selected}
    for result in bundle.latest_results:
        runner._validate_runner_attempt(
            result,
            input_row=inputs_by_id[str(result["sample_id"])],
            repo_root=bundle.repo_root,
            artifact_root=bundle.artifact_root,
            run_id=bundle.run_id,
            run_manifest_fingerprint=bundle.fingerprint,
            verify_artifacts=True,
            recompute_preprocess=False,
        )
    inventory = runner.validate_artifact_inventory(
        artifact_root=bundle.artifact_root,
        selected=bundle.selected,
        latest_by_sample_id=latest.latest_by_sample_id,
    )
    if (
        inventory.get("inventory_sha256")
        != bundle.evidence_snapshot["artifact_inventory_sha256"]
    ):
        raise ValueError("IML-ViT artifacts changed during audit")


def _formal_output_paths(
    bundle: RunBundle,
    *,
    metrics_output: Path | None,
    audit_output: Path | None,
) -> tuple[Path, Path]:
    expected_metrics = (bundle.run_dir / "balanced250_metrics.json").resolve()
    expected_audit = (bundle.run_dir / "independent_audit.json").resolve()
    requested_metrics = (
        expected_metrics if metrics_output is None else metrics_output.resolve()
    )
    requested_audit = expected_audit if audit_output is None else audit_output.resolve()
    if requested_metrics != expected_metrics or requested_audit != expected_audit:
        raise ValueError("IML-ViT formal outputs must use frozen run-local paths")
    if requested_metrics == requested_audit:
        raise ValueError("IML-ViT formal output paths overlap")
    protected = {
        bundle.manifest_path.resolve(),
        bundle.results_path.resolve(),
        bundle.expected_path.resolve(),
        bundle.summary_path.resolve(),
        bundle.release.manifest_path.resolve(),
    }
    if requested_metrics in protected or requested_audit in protected:
        raise ValueError("IML-ViT output overlaps protected evidence")
    for path in (requested_metrics, requested_audit):
        _reject_symlink_components(path, f"IML-ViT output {path.name}")
        if path.exists() and (path.is_symlink() or not path.is_file()):
            raise ValueError(f"unsafe IML-ViT output path: {path}")
    return requested_metrics, requested_audit


def analyze(
    *,
    repo_root: Path,
    results_dir: Path,
    artifacts_dir: Path,
    run_id: str,
    imlvit_root: Path,
    checkpoint_path: Path,
    metrics_output: Path | None = None,
    audit_output: Path | None = None,
    fresh_replay: bool = True,
    requested_device: str | None = None,
) -> dict[str, Any]:
    if run_id != DEFAULT_FORMAL_RUN_ID:
        raise ValueError(
            f"IML-ViT formal analysis run-id must be {DEFAULT_FORMAL_RUN_ID}"
        )
    bundle = load_run(
        repo_root=repo_root,
        results_dir=results_dir,
        artifacts_dir=artifacts_dir,
        run_id=run_id,
        expected_mode="formal",
        imlvit_root=imlvit_root,
        checkpoint_path=checkpoint_path,
    )
    recorded_device = bundle.immutable.get("runtime", {}).get("device")
    if requested_device is not None and requested_device != recorded_device:
        raise ValueError(
            f"requested analyzer device {requested_device!r} differs from "
            f"recorded device {recorded_device!r}"
        )
    metrics_path, audit_path = _formal_output_paths(
        bundle,
        metrics_output=metrics_output,
        audit_output=audit_output,
    )
    smoke_gate = compare_smoke_runs(
        repo_root=repo_root,
        results_dir=results_dir,
        artifacts_dir=artifacts_dir,
        first_run_id=DEFAULT_SMOKE_RUN_ID_A,
        second_run_id=DEFAULT_SMOKE_RUN_ID_B,
        imlvit_root=imlvit_root,
        checkpoint_path=checkpoint_path,
    )
    artifact_audit = audit_artifacts(bundle)
    metrics = {
        **recompute_metrics(bundle),
        "generated_at": utc_now(),
    }
    if (
        metrics["native_pixels_exactly_at_threshold"]
        != artifact_audit["native_pixels_exactly_0_5"]
        or metrics["native_images_with_pixels_exactly_at_threshold"]
        != artifact_audit["native_images_with_pixels_exactly_0_5"]
    ):
        raise ValueError("IML-ViT threshold-boundary audit counts changed")
    fresh = (
        replay_model(
            bundle,
            imlvit_root=imlvit_root,
            checkpoint_path=checkpoint_path,
        )
        if fresh_replay
        else {
            "schema_version": FRESH_REPLAY_SCHEMA_VERSION,
            "status": "explicitly_skipped",
            "publishable": False,
            "required_default": True,
            "selected_images_replayed": 0,
        }
    )
    if fresh_replay and (
        fresh["native_pixels_exactly_0_5"]
        != artifact_audit["native_pixels_exactly_0_5"]
        or fresh["native_images_with_pixels_exactly_0_5"]
        != artifact_audit["native_images_with_pixels_exactly_0_5"]
    ):
        raise ValueError("IML-ViT fresh threshold-boundary counts changed")
    _verify_bundle_unchanged(bundle)
    provenance = _validate_recorded_provenance(
        bundle.immutable,
        repo_root=repo_root,
        imlvit_root=imlvit_root,
        checkpoint_path=checkpoint_path,
    )
    metrics_sha = _write_json_verified(metrics_path, metrics)
    analyzer_path = Path(__file__).resolve()
    audit = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "status": "passed" if fresh_replay else "diagnostic_only",
        "publishable": fresh_replay,
        "generated_at": utc_now(),
        "method": runner.MODEL_NAME,
        "model_slug": runner.MODEL_SLUG,
        "run_id": bundle.run_id,
        "run_manifest_fingerprint": bundle.fingerprint,
        "contract_checks": {
            "capability": "local_t2_only",
            "formal_images_t1": 0,
            "formal_images_t2": FORMAL_IMAGES,
            "formal_selected_ids_sha256": FORMAL_SELECTED_IDS_SHA256,
            "formal_selected_rows_sha256": FORMAL_SELECTED_ROWS_SHA256,
            "smoke_images": SMOKE_IMAGES,
            "smoke_selected_ids_sha256": SMOKE_SELECTED_IDS_SHA256,
            "smoke_selected_rows_sha256": SMOKE_SELECTED_ROWS_SHA256,
            "native_t1_head": False,
            "map_statistic_promoted_to_t1": False,
            "fullframe_selected_images": 0,
            "fullframe_t1": "not_applicable",
            "fullframe_t2": "not_applicable",
            "official_t2_threshold": runner.MASK_THRESHOLD,
            "official_t2_threshold_operator": runner.MASK_THRESHOLD_OPERATOR,
            "shared_t2_threshold": runner.MASK_THRESHOLD,
            "shared_t2_threshold_operator": (runner.SHARED_T2_THRESHOLD_OPERATOR),
            "native_pixels_exactly_0_5": artifact_audit["native_pixels_exactly_0_5"],
            "native_images_with_pixels_exactly_0_5": (
                artifact_audit["native_images_with_pixels_exactly_0_5"]
            ),
            "operator_equivalence_checked": True,
            "shared_t2_operator_equivalent_to_official": False,
            "operator_non_equivalence_observed_on_formal_artifacts": (
                artifact_audit["operator_non_equivalence_observed_on_formal_artifacts"]
            ),
            "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
            "bootstrap_seed": BOOTSTRAP_SEED,
        },
        "coverage": dict(bundle.coverage),
        "attempt_history": dict(bundle.history),
        "provenance": provenance,
        "smoke_reproducibility_gate": {
            "status": smoke_gate["status"],
            "path": smoke_gate["path"],
            "sha256": smoke_gate["sha256"],
            "comparison": smoke_gate["comparison"],
        },
        "artifact_audit": artifact_audit,
        "fresh_model_replay": fresh,
        "fresh_model_metrics_exact": fresh_replay,
        "metrics": {
            "path": repo_relative(metrics_path, repo_root),
            "sha256": metrics_sha,
            "schema_version": METRICS_SCHEMA_VERSION,
            "t2_schema_version": T2_METRICS_SCHEMA_VERSION,
            "official_t2_threshold_operator": (
                metrics["official_t2_threshold_operator"]
            ),
            "shared_t2_threshold_operator": metrics["shared_t2_threshold_operator"],
            "shared_t2_operator_equivalent_to_official": metrics[
                "shared_t2_operator_equivalent_to_official"
            ],
            "native_pixels_exactly_at_threshold": metrics[
                "native_pixels_exactly_at_threshold"
            ],
            "native_images_with_pixels_exactly_at_threshold": metrics[
                "native_images_with_pixels_exactly_at_threshold"
            ],
        },
        "evidence_snapshot": dict(bundle.evidence_snapshot),
        "license": runner.LICENSE_RECORD,
        "commercial_use_clearance_established": False,
        "resource_expectation": runner.RESOURCE_EXPECTATION,
        "analyzer": {
            "path": repo_relative(analyzer_path, repo_root),
            "bytes": analyzer_path.stat().st_size,
            "sha256": sha256_file(analyzer_path),
            "python_executable": str(Path(sys.executable)),
            "numpy": np.__version__,
            "source_contract": analyzer_source_contract(repo_root),
        },
    }
    audit_sha = _write_json_verified(audit_path, audit)
    return {
        "status": audit["status"],
        "publishable": audit["publishable"],
        "run_id": bundle.run_id,
        "metrics_path": repo_relative(metrics_path, repo_root),
        "metrics_sha256": metrics_sha,
        "audit_path": repo_relative(audit_path, repo_root),
        "audit_sha256": audit_sha,
        "fresh_replay": fresh["status"],
        "smoke_gate": smoke_gate["status"],
    }


def _absolute_regular_directory(
    path: Path,
    *,
    expected_path: Path,
    label: str,
) -> Path:
    _reject_symlink_components(path, label)
    candidate = path.resolve()
    if candidate != expected_path.resolve():
        raise ValueError(f"{label} must be exactly {expected_path}")
    if not path.is_dir() or path.is_symlink() or not candidate.is_dir():
        raise FileNotFoundError(candidate)
    return candidate


def _absolute_regular_file(
    path: Path,
    *,
    expected_path: Path,
    label: str,
) -> Path:
    _reject_symlink_components(path, label)
    candidate = path.resolve()
    if candidate != expected_path.resolve():
        raise ValueError(f"{label} must be exactly {expected_path}")
    if not path.is_file() or path.is_symlink() or not candidate.is_file():
        raise FileNotFoundError(candidate)
    return candidate


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--artifacts-dir", type=Path, default=DEFAULT_ARTIFACTS_DIR)
    parser.add_argument("--run-id", default=DEFAULT_FORMAL_RUN_ID)
    parser.add_argument("--imlvit-root", type=Path, default=DEFAULT_IMLVIT_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device")
    parser.add_argument("--metrics-output", type=Path)
    parser.add_argument("--audit-output", type=Path)
    parser.add_argument("--skip-fresh-replay", action="store_true")
    parser.add_argument("--compare-smoke", action="store_true")
    parser.add_argument("--smoke-run-id-a", default=DEFAULT_SMOKE_RUN_ID_A)
    parser.add_argument("--smoke-run-id-b", default=DEFAULT_SMOKE_RUN_ID_B)
    parser.add_argument("--comparison-output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    repo_root = (
        Path(args.repo_root).resolve()
        if args.repo_root is not None
        else Path(__file__).resolve().parents[2]
    )
    if not repo_root.is_dir():
        raise FileNotFoundError(repo_root)
    results_dir = _safe_standard_root(
        args.results_dir,
        repo_root=repo_root,
        expected_relative=DEFAULT_RESULTS_DIR,
        label="IML-ViT results root",
    )
    artifacts_dir = _safe_standard_root(
        args.artifacts_dir,
        repo_root=repo_root,
        expected_relative=DEFAULT_ARTIFACTS_DIR,
        label="IML-ViT artifacts root",
    )
    imlvit_requested = (
        args.imlvit_root
        if args.imlvit_root.is_absolute()
        else repo_root / args.imlvit_root
    )
    imlvit_root = _absolute_regular_directory(
        imlvit_requested,
        expected_path=DEFAULT_IMLVIT_ROOT,
        label="IML-ViT source root",
    )
    checkpoint_requested = (
        args.checkpoint
        if args.checkpoint.is_absolute()
        else repo_root / args.checkpoint
    )
    checkpoint_path = _absolute_regular_file(
        checkpoint_requested,
        expected_path=DEFAULT_CHECKPOINT,
        label="IML-ViT checkpoint",
    )
    if args.compare_smoke:
        if (
            args.run_id != DEFAULT_FORMAL_RUN_ID
            or args.metrics_output is not None
            or args.audit_output is not None
            or args.skip_fresh_replay
            or args.device is not None
        ):
            raise ValueError(
                "smoke comparison accepts no formal output/replay/device " "overrides"
            )
        report = compare_smoke_runs(
            repo_root=repo_root,
            results_dir=results_dir,
            artifacts_dir=artifacts_dir,
            first_run_id=runner._valid_run_id(args.smoke_run_id_a),
            second_run_id=runner._valid_run_id(args.smoke_run_id_b),
            imlvit_root=imlvit_root,
            checkpoint_path=checkpoint_path,
            output_path=(
                _anchored(args.comparison_output, repo_root)
                if args.comparison_output is not None
                else None
            ),
        )
    else:
        if (
            args.smoke_run_id_a != DEFAULT_SMOKE_RUN_ID_A
            or args.smoke_run_id_b != DEFAULT_SMOKE_RUN_ID_B
            or args.comparison_output is not None
        ):
            raise ValueError("formal analysis accepts no smoke overrides")
        report = analyze(
            repo_root=repo_root,
            results_dir=results_dir,
            artifacts_dir=artifacts_dir,
            run_id=runner._valid_run_id(args.run_id),
            imlvit_root=imlvit_root,
            checkpoint_path=checkpoint_path,
            metrics_output=(
                _anchored(args.metrics_output, repo_root)
                if args.metrics_output is not None
                else None
            ),
            audit_output=(
                _anchored(args.audit_output, repo_root)
                if args.audit_output is not None
                else None
            ),
            fresh_replay=not args.skip_fresh_replay,
            requested_device=args.device,
        )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
