#!/usr/bin/env python3
"""Shared strict Balanced250 runner and auditor for the final three localizers.

The three models already have audited Mouse-v1 adapters.  This module keeps
those model-specific preprocessing, checkpoint loading, forward, and
postprocessing primitives unchanged while binding them to the common
Balanced250 T2-only selection and result contracts.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
import re
import sys
import traceback
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from eval.opensource.balanced250_localization_metrics import (
    summarize_balanced250_t2,
)
from eval.opensource.balanced_run_contract import (
    RunDatasetContract,
    build_result_identity,
    build_run_dataset_contract,
)
from eval.opensource.canonical_release import (
    BALANCED_DATASET_ID,
    BALANCED_SCHEMA,
    LOCALIZATION_CONDITIONS,
    Capability,
    CanonicalRelease,
    SelectionSpec,
    load_canonical_release,
    select_inputs,
)
from eval.opensource.common import (
    append_jsonl,
    atomic_write_json,
    atomic_write_jsonl,
    read_jsonl,
    repo_relative,
    sha256_file,
    stable_json,
    utc_now,
)


RUN_SCHEMA = "claimforge_localizer_balanced_run_v1"
SUMMARY_SCHEMA = "claimforge_localizer_balanced_summary_v1"
AUDIT_SCHEMA = "claimforge_localizer_balanced_audit_v1"
METRICS_SCHEMA = "balanced250_t2_summary_v1"
DEFAULT_DATASET_MANIFEST = Path("outputs/opensource/balanced250_v1/manifest.json")
DEFAULT_RESULTS_ROOT = Path("results/opensource")
DEFAULT_ARTIFACTS_ROOT = Path("outputs/opensource")
DEFAULT_SMOKE_LIMIT = 5
FORMAL_COUNTS = {
    "real": 275,
    "local_mouse": 250,
    "local_cat": 250,
    "local_trash_can": 250,
}
SMOKE_COUNTS = {condition: 5 for condition in LOCALIZATION_CONDITIONS}
RUN_ID_PATTERN = re.compile(r"[a-z0-9][a-z0-9._-]{0,127}\Z")


@dataclass(frozen=True)
class LocalizerSpec:
    """Frozen model binding reused from one audited Mouse-v1 adapter."""

    key: str
    module_name: str
    results_slug: str
    display_name: str
    model_slug: str
    source_roots: tuple[tuple[str, Path], ...]
    checkpoint: Path
    checkpoint_sha256: str
    source_commits: tuple[tuple[str, str], ...]
    array_directories: tuple[tuple[str, str], ...]
    native_probability_key: str
    official_threshold_operator: str
    formal_run_id: str
    smoke_run_id_a: str
    smoke_run_id_b: str
    bootstrap_seed: int

    @property
    def legacy(self) -> Any:
        return importlib.import_module(self.module_name)

    @property
    def arrays(self) -> dict[str, str]:
        return dict(self.array_directories)

    @property
    def roots(self) -> dict[str, Path]:
        return dict(self.source_roots)


def _legacy_value(module_name: str, name: str) -> Any:
    return getattr(importlib.import_module(module_name), name)


def model_specs() -> dict[str, LocalizerSpec]:
    """Return the exact published model/checkpoint bindings."""

    mesorch_module = "eval.opensource.run_mesorch"
    relay_module = "eval.opensource.run_relayformer"
    dino_module = "eval.opensource.run_dinov3_iml"
    return {
        "mesorch": LocalizerSpec(
            key="mesorch",
            module_name=mesorch_module,
            results_slug="mesorch",
            display_name=_legacy_value(mesorch_module, "MODEL_NAME"),
            model_slug=_legacy_value(mesorch_module, "MODEL_SLUG"),
            source_roots=(
                ("mesorch_root", _legacy_value(mesorch_module, "DEFAULT_MESORCH_ROOT")),
            ),
            checkpoint=_legacy_value(mesorch_module, "DEFAULT_CHECKPOINT"),
            checkpoint_sha256=_legacy_value(mesorch_module, "CHECKPOINT")["sha256"],
            source_commits=(
                ("Mesorch", _legacy_value(mesorch_module, "MODEL_SOURCE_COMMIT")),
            ),
            array_directories=(
                ("internal_logits_model_128", "raw_logits_model_128"),
                ("probability_model_512", "score_maps_model_512"),
                ("probability_native", "score_maps_native"),
            ),
            native_probability_key="probability_native",
            official_threshold_operator=">",
            formal_run_id="mesorch_epoch98_balanced250_v1_full1025_r2_20260727",
            smoke_run_id_a="mesorch_epoch98_balanced250_v1_smoke5x4_a_r2_20260727",
            smoke_run_id_b="mesorch_epoch98_balanced250_v1_smoke5x4_b_r2_20260727",
            bootstrap_seed=2026072701,
        ),
        "relayformer": LocalizerSpec(
            key="relayformer",
            module_name=relay_module,
            results_slug="relayformer",
            display_name=_legacy_value(relay_module, "MODEL_NAME"),
            model_slug=_legacy_value(relay_module, "MODEL_SLUG"),
            source_roots=(
                (
                    "relayformer_root",
                    _legacy_value(relay_module, "DEFAULT_RELAYFORMER_ROOT"),
                ),
            ),
            checkpoint=_legacy_value(relay_module, "DEFAULT_CHECKPOINT"),
            checkpoint_sha256=_legacy_value(relay_module, "CHECKPOINT")["sha256"],
            source_commits=(
                (
                    "RelayFormer",
                    _legacy_value(relay_module, "MODEL_SOURCE_COMMIT"),
                ),
            ),
            array_directories=(
                ("raw_logits_model_1024", "raw_logits_model_1024"),
                ("raw_logits_native", "raw_logits_native"),
                ("probability_model_1024", "score_maps_model_1024"),
                ("probability_valid", "score_maps_valid"),
                ("probability_native", "score_maps_native"),
            ),
            native_probability_key="probability_native",
            official_threshold_operator=">",
            formal_run_id=(
                "relayformer_checkpoint164_balanced250_v1_full1025_r2_20260727"
            ),
            smoke_run_id_a=(
                "relayformer_checkpoint164_balanced250_v1_smoke5x4_a_r2_20260727"
            ),
            smoke_run_id_b=(
                "relayformer_checkpoint164_balanced250_v1_smoke5x4_b_r2_20260727"
            ),
            bootstrap_seed=2026072702,
        ),
        "dinov3_iml": LocalizerSpec(
            key="dinov3_iml",
            module_name=dino_module,
            results_slug="dinov3_iml",
            display_name=_legacy_value(dino_module, "MODEL_NAME"),
            model_slug=_legacy_value(dino_module, "MODEL_SLUG"),
            source_roots=(
                (
                    "dinov3_iml_root",
                    _legacy_value(dino_module, "DEFAULT_DINOV3_IML_ROOT"),
                ),
                ("dinov3_root", _legacy_value(dino_module, "DEFAULT_DINOV3_ROOT")),
            ),
            checkpoint=Path(
                "/root/.cache/claimforge/checkpoints/"
                "dinov3_iml/checkpoint-48.pth"
            ),
            checkpoint_sha256=_legacy_value(dino_module, "CHECKPOINT")["sha256"],
            source_commits=(
                (
                    "DINOv3-IML",
                    _legacy_value(dino_module, "MODEL_SOURCE_COMMIT"),
                ),
                (
                    "DINOv3",
                    _legacy_value(dino_module, "DINOV3_SOURCE_COMMIT"),
                ),
            ),
            array_directories=(
                ("raw_logits_model_32", "raw_logits_model_32"),
                ("raw_logits_model_512", "raw_logits_model_512"),
                ("probability_model_512", "score_maps_model_512"),
                ("probability_native", "score_maps_native"),
            ),
            native_probability_key="probability_native",
            official_threshold_operator=">",
            formal_run_id=(
                "dinov3_iml_checkpoint48_balanced250_v1_full1025_r2_20260727"
            ),
            smoke_run_id_a=(
                "dinov3_iml_checkpoint48_balanced250_v1_smoke5x4_a_r2_20260727"
            ),
            smoke_run_id_b=(
                "dinov3_iml_checkpoint48_balanced250_v1_smoke5x4_b_r2_20260727"
            ),
            bootstrap_seed=2026072703,
        ),
    }


def get_spec(key: str) -> LocalizerSpec:
    try:
        return model_specs()[key]
    except KeyError as error:
        raise ValueError(f"unsupported localizer {key!r}") from error


def _fingerprint(value: Any) -> str:
    return hashlib.sha256(stable_json(value).encode("utf-8")).hexdigest()


def _array_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(array).tobytes(order="C")
    ).hexdigest()


def _package_versions() -> dict[str, str | None]:
    names = (
        "torch",
        "torchvision",
        "timm",
        "IMDLBenCo",
        "albumentations",
        "opencv-python-headless",
        "numpy",
        "Pillow",
        "scikit-learn",
        "rotary-embedding-torch",
        "peft",
        "transformers",
        "accelerate",
    )
    result: dict[str, str | None] = {}
    for name in names:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def _git_value(root: Path, *args: str) -> str | None:
    import subprocess

    process = subprocess.run(
        ["git", "-C", str(root), *args],
        text=True,
        capture_output=True,
        check=False,
    )
    return process.stdout.strip() if process.returncode == 0 else None


def _source_contract(spec: LocalizerSpec) -> dict[str, Any]:
    roots: dict[str, Any] = {}
    for (argument, root), (label, commit) in zip(
        spec.source_roots,
        spec.source_commits,
        strict=True,
    ):
        resolved = root.resolve()
        actual_commit = _git_value(resolved, "rev-parse", "HEAD")
        tracked_status = _git_value(
            resolved, "status", "--short", "--untracked-files=no"
        )
        if actual_commit != commit or tracked_status:
            raise ValueError(f"{label} pinned source checkout changed")
        roots[argument] = {
            "label": label,
            "path": str(resolved),
            "commit": actual_commit,
            "tree": _git_value(resolved, "rev-parse", "HEAD^{tree}"),
            "tracked_clean": True,
        }
    checkpoint = spec.checkpoint.resolve()
    if (
        not checkpoint.is_file()
        or checkpoint.is_symlink()
        or sha256_file(checkpoint) != spec.checkpoint_sha256
    ):
        raise ValueError(f"{spec.display_name} checkpoint binding changed")
    return {
        "repositories": roots,
        "checkpoint": {
            "path": str(checkpoint),
            "bytes": checkpoint.stat().st_size,
            "sha256": spec.checkpoint_sha256,
        },
    }


def _adapter_contract(
    repo_root: Path,
    spec: LocalizerSpec,
) -> dict[str, Any]:
    paths = (
        "eval/opensource/localizer_balanced.py",
        f"{spec.module_name.replace('.', '/')}.py",
        f"eval/opensource/run_{spec.key}_balanced.py",
        "eval/opensource/balanced250_localization_metrics.py",
        "eval/opensource/balanced_run_contract.py",
        "eval/opensource/canonical_release.py",
        "eval/opensource/common.py",
    )
    return {
        path: {
            "bytes": (repo_root / path).stat().st_size,
            "sha256": sha256_file(repo_root / path),
        }
        for path in paths
    }


def _verify_legacy_static_contract(spec: LocalizerSpec) -> None:
    legacy = spec.legacy
    kwargs = {name: path.resolve() for name, path in spec.source_roots}
    kwargs["checkpoint_path"] = spec.checkpoint.resolve()
    legacy._verify_static_contract(**kwargs)


def _load_model(spec: LocalizerSpec, device_name: str) -> tuple[Any, Any]:
    legacy = spec.legacy
    kwargs = {name: path.resolve() for name, path in spec.source_roots}
    kwargs.update(
        {
            "checkpoint_path": spec.checkpoint.resolve(),
            "device_name": device_name,
        }
    )
    return legacy.load_model(**kwargs)


def _preprocess(
    spec: LocalizerSpec,
    path: Path,
) -> tuple[np.ndarray, tuple[int, int], tuple[int, int] | None, dict[str, Any]]:
    value = spec.legacy.preprocess_image(path)
    if spec.key == "relayformer":
        image, native_size, resized_size, metadata = value
    else:
        image, native_size, metadata = value
        resized_size = None
    image = np.ascontiguousarray(image, dtype=np.float32)
    if image.ndim != 3 or image.shape[0] != 3 or not np.isfinite(image).all():
        raise ValueError(f"{spec.display_name} preprocessing contract changed")
    width, height = (int(item) for item in native_size)
    return image, (width, height), resized_size, dict(metadata)


def _infer(
    spec: LocalizerSpec,
    model: Any,
    device: Any,
    image: np.ndarray,
    native_size: tuple[int, int],
    resized_size: tuple[int, int] | None,
) -> tuple[dict[str, np.ndarray], int, float]:
    width, height = native_size
    kwargs: dict[str, Any] = {
        "native_width": width,
        "native_height": height,
    }
    if spec.key == "relayformer":
        if resized_size is None:
            raise ValueError("RelayFormer resized geometry is missing")
        kwargs.update(
            {
                "resized_width": int(resized_size[0]),
                "resized_height": int(resized_size[1]),
            }
        )
    processed, peak, latency = spec.legacy.infer_one(
        model,
        device,
        image,
        **kwargs,
    )
    expected_keys = set(spec.arrays)
    if set(processed) != expected_keys:
        raise ValueError(
            f"{spec.display_name} processed outputs changed: "
            f"{sorted(processed)} != {sorted(expected_keys)}"
        )
    arrays: dict[str, np.ndarray] = {}
    for key, value in processed.items():
        array = np.ascontiguousarray(value, dtype=np.float32)
        if array.ndim != 2 or not array.size or not np.isfinite(array).all():
            raise ValueError(f"{spec.display_name} {key} is not a finite map")
        if key.startswith("probability_") and (
            float(array.min()) < 0.0 or float(array.max()) > 1.0
        ):
            raise ValueError(f"{spec.display_name} {key} falls outside [0, 1]")
        arrays[key] = array
    native = arrays[spec.native_probability_key]
    if native.shape != (height, width):
        raise ValueError(f"{spec.display_name} native output shape changed")
    return arrays, int(peak), float(latency)


def select_mode(
    release: CanonicalRelease,
    mode: str,
    *,
    sample_id: str | None = None,
) -> tuple[SelectionSpec, list[dict[str, Any]]]:
    if mode == "formal":
        selection = SelectionSpec(capability=Capability.LOCAL_T2_ONLY)
    elif mode == "smoke":
        selection = SelectionSpec(
            capability=Capability.LOCAL_T2_ONLY,
            per_condition_limit=DEFAULT_SMOKE_LIMIT,
        )
    elif mode == "single":
        if not sample_id:
            raise ValueError("single mode requires --sample-id")
        selection = SelectionSpec(
            capability=Capability.LOCAL_T2_ONLY,
            sample_id=sample_id,
        )
    else:
        raise ValueError(f"unsupported mode {mode!r}")
    selected = select_inputs(release, selection)
    counts = Counter(str(row["condition"]) for row in selected)
    if mode == "formal" and (
        len(selected) != 1025 or dict(counts) != FORMAL_COUNTS
    ):
        raise ValueError("formal localizer selection drifted")
    if mode == "smoke" and (
        len(selected) != 20
        or dict(counts) != SMOKE_COUNTS
        or any(row.get("panel") is not True for row in selected)
    ):
        raise ValueError("smoke localizer selection drifted")
    if any(str(row["condition"]).startswith("fullframe_") for row in selected):
        raise ValueError("T2-only localizer selected a full-frame condition")
    return selection, selected


def _verified_input_path(row: Mapping[str, Any], repo_root: Path) -> Path:
    value = row.get("canonical_path")
    sample_id = row.get("sample_id")
    if not isinstance(value, str) or not isinstance(sample_id, str):
        raise ValueError("canonical input identity is incomplete")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("canonical input path is unsafe")
    path = (repo_root / relative).resolve()
    try:
        path.relative_to(repo_root)
    except ValueError as error:
        raise ValueError("canonical input escapes repository") from error
    if (
        not path.is_file()
        or path.is_symlink()
        or path.name != f"{sample_id}.jpg"
        or path.stat().st_size != int(row["canonical_bytes"])
        or sha256_file(path) != row["canonical_sha256"]
    ):
        raise ValueError(f"canonical input changed: {sample_id}")
    return path


def _artifact_paths(
    spec: LocalizerSpec,
    artifact_root: Path,
    sample_id: str,
) -> tuple[dict[str, Path], Path]:
    arrays = {
        key: artifact_root / directory / f"{sample_id}.npy"
        for key, directory in spec.array_directories
    }
    return arrays, artifact_root / "masks_native" / f"{sample_id}.png"


def _prepare_artifact_root(spec: LocalizerSpec, root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    expected = {directory for _, directory in spec.array_directories}
    expected.add("masks_native")
    for directory in sorted(expected):
        path = root / directory
        if path.exists() and (path.is_symlink() or not path.is_dir()):
            raise ValueError(f"invalid artifact directory: {path}")
        path.mkdir(exist_ok=True)


def _array_record(path: Path, array: np.ndarray, repo_root: Path) -> dict[str, Any]:
    return {
        "path": repo_relative(path, repo_root),
        "sha256": sha256_file(path),
        "array_sha256": _array_sha256(array),
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "bytes": path.stat().st_size,
    }


def _validate_artifact_record(
    record: Mapping[str, Any],
    *,
    expected_path: Path,
    expected_shape: tuple[int, int] | None = None,
) -> np.ndarray:
    if record.get("path") is None or not expected_path.is_file():
        raise ValueError(f"artifact is missing: {expected_path}")
    if sha256_file(expected_path) != record.get("sha256"):
        raise ValueError(f"artifact file hash changed: {expected_path}")
    array = np.load(expected_path, allow_pickle=False, mmap_mode="r")
    if array.dtype != np.float32 or array.ndim != 2:
        raise ValueError(f"artifact array schema changed: {expected_path}")
    if expected_shape is not None and array.shape != expected_shape:
        raise ValueError(f"artifact array shape changed: {expected_path}")
    if list(array.shape) != record.get("shape"):
        raise ValueError(f"artifact shape metadata changed: {expected_path}")
    if _array_sha256(array) != record.get("array_sha256"):
        raise ValueError(f"artifact array hash changed: {expected_path}")
    return array


def _result_identity(
    spec: LocalizerSpec,
    row: Mapping[str, Any],
    *,
    run_id: str,
    fingerprint: str,
    valid: bool,
) -> dict[str, Any]:
    identity = build_result_identity(
        row,
        run_id=run_id,
        run_manifest_fingerprint=fingerprint,
    )
    condition = str(row["condition"])
    return {
        **identity,
        "model": spec.display_name,
        "model_slug": spec.model_slug,
        "checkpoint_sha256": spec.checkpoint_sha256,
        "valid_for_metrics": valid,
        "valid_for_t1": False,
        "valid_for_t2": True,
        "t2_applicable": True,
        "task_scope": {
            "capability": "local_t2_only",
            "valid_for_t1": False,
            "valid_for_t2": True,
            "native_dense_output": True,
            "map_statistic_promoted_to_t1": False,
            "t2_target_semantics": (
                "all_zero_real_false_positive_area"
                if condition == "real"
                else "exact_diff_local_insertion"
            ),
        },
    }


def _load_latest_results(path: Path) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    rows = read_jsonl(path) if path.is_file() else []
    latest: dict[str, dict[str, Any]] = {}
    for row in rows:
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str):
            raise ValueError("result row has no sample_id")
        if latest.get(sample_id, {}).get("status") == "ok":
            raise ValueError(f"result appended after success: {sample_id}")
        latest[sample_id] = row
    return rows, latest


def _validate_existing_ok(
    spec: LocalizerSpec,
    row: Mapping[str, Any],
    input_row: Mapping[str, Any],
    *,
    repo_root: Path,
    artifact_root: Path,
    run_id: str,
    fingerprint: str,
) -> None:
    expected = _result_identity(
        spec,
        input_row,
        run_id=run_id,
        fingerprint=fingerprint,
        valid=True,
    )
    for key, value in expected.items():
        if stable_json(row.get(key)) != stable_json(value):
            raise ValueError(f"resume identity changed for {input_row['sample_id']}: {key}")
    records = row.get("arrays")
    if not isinstance(records, Mapping) or set(records) != set(spec.arrays):
        raise ValueError("resume array inventory changed")
    array_paths, mask_path = _artifact_paths(
        spec, artifact_root, str(input_row["sample_id"])
    )
    for key, path in array_paths.items():
        record = records.get(key)
        if not isinstance(record, Mapping):
            raise ValueError(f"resume array record changed: {key}")
        expected_shape = (
            (int(input_row["height"]), int(input_row["width"]))
            if key == spec.native_probability_key
            else None
        )
        _validate_artifact_record(
            record,
            expected_path=path,
            expected_shape=expected_shape,
        )
    if (
        not mask_path.is_file()
        or sha256_file(mask_path) != row.get("mask", {}).get("sha256")
    ):
        raise ValueError("resume native mask changed")


def _runtime_contract(device_name: str) -> dict[str, Any]:
    import torch

    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return {
        "python": platform.python_version(),
        "python_executable": str(Path(sys.executable).resolve()),
        "platform": platform.platform(),
        "packages": _package_versions(),
        "device": str(device),
        "cuda": torch.version.cuda,
        "cudnn": (
            torch.backends.cudnn.version()
            if torch.backends.cudnn.is_available()
            else None
        ),
        "gpu": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else None
        ),
    }


def _configure_determinism(spec: LocalizerSpec, seed: int) -> None:
    import torch

    if hasattr(spec.legacy, "configure_determinism"):
        spec.legacy.configure_determinism(seed)
        return
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_experiment(
    spec: LocalizerSpec,
    *,
    repo_root: Path,
    dataset_manifest: Path,
    results_root: Path,
    artifacts_root: Path,
    run_id: str,
    mode: str,
    device_name: str,
    seed: int,
    resume: bool,
    sample_id: str | None = None,
) -> dict[str, Any]:
    """Run or resume one deterministic Balanced250 inference bundle."""

    if RUN_ID_PATTERN.fullmatch(run_id) is None:
        raise ValueError("run_id contains unsafe characters")
    repo_root = repo_root.resolve()
    release = load_canonical_release(
        repo_root,
        dataset_manifest,
        verify_files=True,
    )
    if (
        release.schema_version != BALANCED_SCHEMA
        or release.dataset_id != BALANCED_DATASET_ID
    ):
        raise ValueError("dataset is not frozen Balanced250")
    selection_spec, selected = select_mode(
        release, mode, sample_id=sample_id
    )
    dataset_contract = build_run_dataset_contract(
        release,
        selection_spec,
        selected,
        score_spec=None,
    )
    _verify_legacy_static_contract(spec)
    source_contract = _source_contract(spec)
    runtime = _runtime_contract(device_name)
    adapter = _adapter_contract(repo_root, spec)

    run_dir = (results_root / spec.results_slug / run_id).resolve()
    artifact_root = (artifacts_root / spec.results_slug / run_id).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    _prepare_artifact_root(spec, artifact_root)
    manifest_path = run_dir / "manifest.json"
    expected_inputs_path = run_dir / "expected_inputs.jsonl"
    results_path = run_dir / "results.jsonl"
    summary_path = run_dir / "summary.json"
    immutable = {
        "schema_version": RUN_SCHEMA,
        "run_id": run_id,
        "mode": mode,
        "model": {
            "key": spec.key,
            "name": spec.display_name,
            "slug": spec.model_slug,
            "checkpoint_sha256": spec.checkpoint_sha256,
            "source": source_contract,
        },
        "dataset_contract": dataset_contract.as_dict(),
        "selection": {
            "images": len(selected),
            "counts": dict(
                sorted(Counter(str(row["condition"]) for row in selected).items())
            ),
            "ids_sha256": _fingerprint(
                [str(row["sample_id"]) for row in selected]
            ),
        },
        "task_scope": {
            "capability": "local_t2_only",
            "valid_for_t1": False,
            "valid_for_t2": True,
            "selected_conditions": list(LOCALIZATION_CONDITIONS),
            "fullframe_conditions_forwarded": False,
        },
        "artifacts": {
            "arrays": spec.arrays,
            "native_probability_key": spec.native_probability_key,
            "native_mask_threshold": 0.5,
            "native_mask_threshold_operator": spec.official_threshold_operator,
        },
        "runtime": runtime,
        "adapter_sources": adapter,
        "seed": seed,
        "outputs": {
            "run_dir": repo_relative(run_dir, repo_root),
            "artifact_root": repo_relative(artifact_root, repo_root),
            "expected_inputs": repo_relative(expected_inputs_path, repo_root),
            "results": repo_relative(results_path, repo_root),
            "summary": repo_relative(summary_path, repo_root),
        },
    }
    fingerprint = _fingerprint(immutable)
    started_at = utc_now()
    if manifest_path.exists():
        if not resume:
            raise FileExistsError(f"run exists; pass --resume: {run_dir}")
        prior = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            prior.get("fingerprint") != fingerprint
            or stable_json(prior.get("immutable")) != stable_json(immutable)
        ):
            raise ValueError("resume immutable manifest changed")
        if read_jsonl(expected_inputs_path) != selected:
            raise ValueError("resume expected inputs changed")
        started_at = str(prior["started_at"])
    else:
        if any(run_dir.iterdir()):
            raise ValueError(f"new run directory is not empty: {run_dir}")
        atomic_write_jsonl(expected_inputs_path, selected)
        atomic_write_json(
            manifest_path,
            {
                "schema_version": RUN_SCHEMA,
                "run_id": run_id,
                "status": "running",
                "started_at": started_at,
                "completed_at": None,
                "fingerprint": fingerprint,
                "immutable": immutable,
            },
        )

    physical_rows, latest = _load_latest_results(results_path)
    inputs_by_id = {str(row["sample_id"]): row for row in selected}
    for sample, result in latest.items():
        if sample not in inputs_by_id:
            raise ValueError(f"resume contains unexpected sample: {sample}")
        if result.get("status") == "ok":
            _validate_existing_ok(
                spec,
                result,
                inputs_by_id[sample],
                repo_root=repo_root,
                artifact_root=artifact_root,
                run_id=run_id,
                fingerprint=fingerprint,
            )
    pending = [
        row
        for row in selected
        if latest.get(str(row["sample_id"]), {}).get("status") != "ok"
    ]
    print(
        f"{spec.display_name} {mode}: {len(selected)} selected, "
        f"{len(pending)} pending",
        flush=True,
    )
    _configure_determinism(spec, seed)
    model = None
    device = None
    fatal: BaseException | None = None
    new_successes = 0
    new_errors = 0
    try:
        if pending:
            model, device = _load_model(spec, device_name)
            print(
                f"loaded {spec.display_name} {spec.checkpoint_sha256[:12]} "
                f"on {device}",
                flush=True,
            )
        for index, input_row in enumerate(pending, start=1):
            sample = str(input_row["sample_id"])
            array_paths, mask_path = _artifact_paths(spec, artifact_root, sample)
            try:
                input_path = _verified_input_path(input_row, repo_root)
                image, native_size, resized_size, preprocess = _preprocess(
                    spec, input_path
                )
                if native_size != (
                    int(input_row["width"]),
                    int(input_row["height"]),
                ):
                    raise ValueError("canonical image dimensions changed")
                assert model is not None and device is not None
                arrays, peak, latency = _infer(
                    spec,
                    model,
                    device,
                    image,
                    native_size,
                    resized_size,
                )
                for key, array in arrays.items():
                    spec.legacy._atomic_save_npy(array_paths[key], array)
                native = arrays[spec.native_probability_key]
                spec.legacy._atomic_save_mask(mask_path, native > np.float32(0.5))
                result = {
                    **_result_identity(
                        spec,
                        input_row,
                        run_id=run_id,
                        fingerprint=fingerprint,
                        valid=True,
                    ),
                    "status": "ok",
                    "completed_at": utc_now(),
                    "preprocess": preprocess,
                    "arrays": {
                        key: _array_record(array_paths[key], arrays[key], repo_root)
                        for key in spec.arrays
                    },
                    "native_probability_key": spec.native_probability_key,
                    "mask": {
                        "path": repo_relative(mask_path, repo_root),
                        "sha256": sha256_file(mask_path),
                        "bytes": mask_path.stat().st_size,
                        "shape": list(native.shape),
                        "dtype": "uint8",
                        "threshold": 0.5,
                        "threshold_operator": spec.official_threshold_operator,
                    },
                    "latency_ms": latency,
                    "peak_cuda_memory_bytes": peak,
                }
                _validate_existing_ok(
                    spec,
                    result,
                    input_row,
                    repo_root=repo_root,
                    artifact_root=artifact_root,
                    run_id=run_id,
                    fingerprint=fingerprint,
                )
                append_jsonl(results_path, result)
                new_successes += 1
                print(
                    f"[{index}/{len(pending)}] ok {sample} "
                    f"latency_ms={latency:.3f}",
                    flush=True,
                )
            except BaseException as error:
                for path in (*array_paths.values(), mask_path):
                    path.unlink(missing_ok=True)
                result = {
                    **_result_identity(
                        spec,
                        input_row,
                        run_id=run_id,
                        fingerprint=fingerprint,
                        valid=False,
                    ),
                    "status": "error",
                    "completed_at": utc_now(),
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                }
                append_jsonl(results_path, result)
                new_errors += 1
                fatal = error
                print(f"[{index}/{len(pending)}] error {sample}: {error}", flush=True)
                break
    finally:
        del model
        gc.collect()
        try:
            import torch

            if device is not None and device.type == "cuda":
                torch.cuda.empty_cache()
        except Exception:
            pass

    physical_rows, latest = _load_latest_results(results_path)
    ok_rows = {
        sample: row for sample, row in latest.items() if row.get("status") == "ok"
    }
    counts = Counter(
        str(inputs_by_id[sample]["condition"]) for sample in ok_rows
    )
    errors = sum(row.get("status") == "error" for row in latest.values())
    missing = len(selected) - len(latest)
    complete = len(ok_rows) == len(selected) and errors == 0 and missing == 0
    latencies = [float(row["latency_ms"]) for row in ok_rows.values()]
    inventory_files = len(ok_rows) * (len(spec.arrays) + 1)
    inventory_bytes = sum(
        int(record["bytes"])
        for row in ok_rows.values()
        for record in row["arrays"].values()
    ) + sum(int(row["mask"]["bytes"]) for row in ok_rows.values())
    summary = {
        "schema_version": SUMMARY_SCHEMA,
        "run_id": run_id,
        "run_manifest_fingerprint": fingerprint,
        "status": "complete" if complete else "incomplete",
        "mode": mode,
        "model": spec.display_name,
        "model_slug": spec.model_slug,
        "checkpoint_sha256": spec.checkpoint_sha256,
        "valid_for_t1": False,
        "valid_for_t2": True,
        "scientific_metrics": None,
        "scientific_metrics_owner": "analyze_localizer_balanced.py",
        "dataset_contract": dataset_contract.as_dict(),
        "coverage": {
            "expected_images": len(selected),
            "valid_images": len(ok_rows),
            "error_images": errors,
            "missing_images": missing,
            "counts_by_condition": dict(sorted(counts.items())),
        },
        "attempt_history": {
            "physical_rows": len(physical_rows),
            "latest_rows": len(latest),
            "superseded_rows": len(physical_rows) - len(latest),
            "new_successes": new_successes,
            "new_errors": new_errors,
            "resume_skips": len(selected) - len(pending),
        },
        "latency_ms": {
            "count": len(latencies),
            "mean": float(np.mean(latencies)) if latencies else None,
            "median": float(np.median(latencies)) if latencies else None,
            "min": min(latencies) if latencies else None,
            "max": max(latencies) if latencies else None,
        },
        "artifact_inventory": {
            "files": inventory_files,
            "bytes": inventory_bytes,
            "arrays_per_success": len(spec.arrays),
            "masks_per_success": 1,
        },
        "generated_at": utc_now(),
    }
    atomic_write_json(summary_path, summary)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        {
            "status": summary["status"],
            "completed_at": utc_now(),
            "outputs": {
                "expected_inputs_sha256": sha256_file(expected_inputs_path),
                "results_sha256": (
                    sha256_file(results_path) if results_path.is_file() else None
                ),
                "summary_sha256": sha256_file(summary_path),
                "artifact_inventory": summary["artifact_inventory"],
            },
            "execution": summary["attempt_history"],
        }
    )
    atomic_write_json(manifest_path, manifest)
    print(json.dumps(summary["coverage"], indent=2), flush=True)
    if fatal is not None:
        raise RuntimeError(f"{spec.display_name} inference failed") from fatal
    if not complete:
        raise RuntimeError(f"{spec.display_name} run is incomplete")
    return summary


def _run_bundle(
    spec: LocalizerSpec,
    repo_root: Path,
    results_root: Path,
    run_id: str,
) -> tuple[
    Path,
    dict[str, Any],
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    run_dir = (results_root / spec.results_slug / run_id).resolve()
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    expected = read_jsonl(run_dir / "expected_inputs.jsonl")
    physical, latest = _load_latest_results(run_dir / "results.jsonl")
    del physical
    results = [
        latest[str(row["sample_id"])]
        for row in expected
        if latest.get(str(row["sample_id"]), {}).get("status") == "ok"
    ]
    if (
        manifest.get("status") != "complete"
        or summary.get("status") != "complete"
        or len(results) != len(expected)
    ):
        raise ValueError(f"run bundle is incomplete: {run_dir}")
    return run_dir, manifest, summary, expected, results


def _artifact_root_from_manifest(
    manifest: Mapping[str, Any], repo_root: Path
) -> Path:
    value = manifest["immutable"]["outputs"]["artifact_root"]
    return (repo_root / str(value)).resolve()


def analyze_formal(
    spec: LocalizerSpec,
    *,
    repo_root: Path,
    dataset_manifest: Path,
    results_root: Path,
    run_id: str,
    iterations: int,
) -> dict[str, Any]:
    """Verify a complete formal bundle and calculate shared T2 metrics."""

    run_dir, manifest, _, expected, results = _run_bundle(
        spec, repo_root, results_root, run_id
    )
    if manifest["immutable"]["mode"] != "formal":
        raise ValueError("scientific analysis requires a formal run")
    release = load_canonical_release(repo_root, dataset_manifest, verify_files=True)
    selection_spec, selected = select_mode(release, "formal")
    if expected != selected:
        raise ValueError("formal expected input materialization changed")
    contract = build_run_dataset_contract(
        release, selection_spec, selected, score_spec=None
    )
    recorded_contract = manifest["immutable"]["dataset_contract"]
    if stable_json(recorded_contract) != stable_json(contract.as_dict()):
        raise ValueError("formal recorded dataset contract changed")
    artifact_root = _artifact_root_from_manifest(manifest, repo_root)

    def load_native(
        input_row: Mapping[str, Any],
        result_row: Mapping[str, Any],
    ) -> np.ndarray:
        sample = str(input_row["sample_id"])
        paths, _ = _artifact_paths(spec, artifact_root, sample)
        record = result_row["arrays"][spec.native_probability_key]
        return _validate_artifact_record(
            record,
            expected_path=paths[spec.native_probability_key],
            expected_shape=(int(input_row["height"]), int(input_row["width"])),
        )

    metrics = summarize_balanced250_t2(
        release.inputs,
        results,
        repo_root=repo_root,
        run_id=run_id,
        run_manifest_fingerprint=manifest["fingerprint"],
        run_dataset_contract=contract,
        load_native_score_map=load_native,
        score_map_name=spec.native_probability_key,
        threshold=0.5,
        threshold_operator=">=",
        iterations=iterations,
        seed=spec.bootstrap_seed,
    )
    if metrics.get("schema_version") != METRICS_SCHEMA:
        raise ValueError("shared T2 metric schema changed")
    metrics.update(
        {
            "model": spec.display_name,
            "model_slug": spec.model_slug,
            "checkpoint_sha256": spec.checkpoint_sha256,
            "official_mask_threshold_operator": spec.official_threshold_operator,
            "shared_metric_threshold_operator": ">=",
        }
    )
    atomic_write_json(run_dir / "metrics.json", metrics)
    return metrics


def compare_smokes(
    spec: LocalizerSpec,
    *,
    repo_root: Path,
    results_root: Path,
    run_id_a: str,
    run_id_b: str,
) -> dict[str, Any]:
    """Require two independent smoke runs to be byte-identical."""

    _, manifest_a, _, expected_a, results_a = _run_bundle(
        spec, repo_root, results_root, run_id_a
    )
    _, manifest_b, _, expected_b, results_b = _run_bundle(
        spec, repo_root, results_root, run_id_b
    )
    if expected_a != expected_b or len(expected_a) != 20:
        raise ValueError("smoke selections differ")
    by_a = {str(row["sample_id"]): row for row in results_a}
    by_b = {str(row["sample_id"]): row for row in results_b}
    mismatches: list[dict[str, Any]] = []
    for input_row in expected_a:
        sample = str(input_row["sample_id"])
        for key in spec.arrays:
            left = by_a[sample]["arrays"][key]["array_sha256"]
            right = by_b[sample]["arrays"][key]["array_sha256"]
            if left != right:
                mismatches.append(
                    {"sample_id": sample, "artifact": key, "a": left, "b": right}
                )
        if by_a[sample]["mask"]["sha256"] != by_b[sample]["mask"]["sha256"]:
            mismatches.append({"sample_id": sample, "artifact": "mask"})
    result = {
        "schema_version": AUDIT_SCHEMA,
        "audit_kind": "independent_smoke_exact_reproducibility",
        "model": spec.display_name,
        "run_id_a": run_id_a,
        "run_id_b": run_id_b,
        "fingerprint_a": manifest_a["fingerprint"],
        "fingerprint_b": manifest_b["fingerprint"],
        "selected_images": len(expected_a),
        "arrays_compared": len(expected_a) * len(spec.arrays),
        "masks_compared": len(expected_a),
        "mismatches": mismatches,
        "status": "pass" if not mismatches else "fail",
        "audited_at": utc_now(),
    }
    output = results_root / spec.results_slug / (
        f"{run_id_a}__vs__{run_id_b}_comparison.json"
    )
    atomic_write_json(output, result)
    if mismatches:
        raise RuntimeError(f"{spec.display_name} smoke reproducibility failed")
    return result


def fresh_replay(
    spec: LocalizerSpec,
    *,
    repo_root: Path,
    results_root: Path,
    run_id: str,
    device_name: str,
) -> dict[str, Any]:
    """Freshly load the model and exactly replay every formal output."""

    run_dir, manifest, _, expected, results = _run_bundle(
        spec, repo_root, results_root, run_id
    )
    if manifest["immutable"]["mode"] != "formal" or len(expected) != 1025:
        raise ValueError("fresh replay requires the complete formal run")
    by_id = {str(row["sample_id"]): row for row in results}
    artifact_root = _artifact_root_from_manifest(manifest, repo_root)
    seed = int(manifest["immutable"]["seed"])
    _verify_legacy_static_contract(spec)
    _configure_determinism(spec, seed)
    model, device = _load_model(spec, device_name)
    mismatches: list[dict[str, Any]] = []
    arrays_compared = 0
    masks_compared = 0
    combined = hashlib.sha256()
    try:
        for index, input_row in enumerate(expected, start=1):
            sample = str(input_row["sample_id"])
            image, native_size, resized_size, _ = _preprocess(
                spec, _verified_input_path(input_row, repo_root)
            )
            arrays, _, _ = _infer(
                spec, model, device, image, native_size, resized_size
            )
            paths, mask_path = _artifact_paths(spec, artifact_root, sample)
            recorded = by_id[sample]
            for key in spec.arrays:
                actual_sha = _array_sha256(arrays[key])
                expected_sha = recorded["arrays"][key]["array_sha256"]
                disk = _validate_artifact_record(
                    recorded["arrays"][key],
                    expected_path=paths[key],
                )
                arrays_compared += 1
                combined.update(f"{sample}\0{key}\0{actual_sha}\n".encode())
                if actual_sha != expected_sha or not np.array_equal(arrays[key], disk):
                    mismatches.append(
                        {
                            "sample_id": sample,
                            "artifact": key,
                            "expected": expected_sha,
                            "actual": actual_sha,
                        }
                    )
            replay_mask = np.where(
                arrays[spec.native_probability_key] > np.float32(0.5),
                np.uint8(255),
                np.uint8(0),
            )
            with Image.open(mask_path) as opened:
                disk_mask = np.asarray(opened.convert("L"), dtype=np.uint8)
            masks_compared += 1
            if not np.array_equal(replay_mask, disk_mask):
                mismatches.append({"sample_id": sample, "artifact": "mask"})
            if index % 25 == 0 or index == len(expected):
                print(
                    f"{spec.display_name} fresh replay "
                    f"{index}/{len(expected)} mismatches={len(mismatches)}",
                    flush=True,
                )
    finally:
        del model
        gc.collect()
        try:
            import torch

            if device.type == "cuda":
                torch.cuda.empty_cache()
        except Exception:
            pass
    audit = {
        "schema_version": AUDIT_SCHEMA,
        "audit_kind": "fresh_process_full_exact_replay",
        "model": spec.display_name,
        "run_id": run_id,
        "run_manifest_fingerprint": manifest["fingerprint"],
        "checkpoint_sha256": spec.checkpoint_sha256,
        "selected_images": len(expected),
        "arrays_compared": arrays_compared,
        "masks_compared": masks_compared,
        "replay_digest": combined.hexdigest(),
        "mismatches": mismatches,
        "status": "pass" if not mismatches else "fail",
        "audited_at": utc_now(),
    }
    atomic_write_json(run_dir / "independent_audit.json", audit)
    if mismatches:
        raise RuntimeError(f"{spec.display_name} fresh replay failed")
    return audit


def _default_run_id(spec: LocalizerSpec, mode: str, smoke_label: str) -> str:
    if mode == "formal":
        return spec.formal_run_id
    if mode == "smoke":
        return spec.smoke_run_id_b if smoke_label == "b" else spec.smoke_run_id_a
    raise ValueError("single mode requires an explicit --run-id")


def _parser(spec: LocalizerSpec) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=f"Run and audit {spec.display_name} on Balanced250."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--mode", choices=("smoke", "formal", "single"), required=True)
    run_parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    run_parser.add_argument("--dataset-manifest", type=Path, default=DEFAULT_DATASET_MANIFEST)
    run_parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    run_parser.add_argument("--artifacts-root", type=Path, default=DEFAULT_ARTIFACTS_ROOT)
    run_parser.add_argument("--run-id")
    run_parser.add_argument("--smoke-label", choices=("a", "b"), default="a")
    run_parser.add_argument("--sample-id")
    run_parser.add_argument("--device", default="cuda:0")
    run_parser.add_argument("--seed", type=int, default=42)
    run_parser.add_argument("--resume", action="store_true")

    analyze_parser = subparsers.add_parser("analyze")
    analyze_parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    analyze_parser.add_argument("--dataset-manifest", type=Path, default=DEFAULT_DATASET_MANIFEST)
    analyze_parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    analyze_parser.add_argument("--run-id", default=spec.formal_run_id)
    analyze_parser.add_argument("--bootstrap-iterations", type=int, default=1000)

    compare_parser = subparsers.add_parser("compare-smokes")
    compare_parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    compare_parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    compare_parser.add_argument("--run-id-a", default=spec.smoke_run_id_a)
    compare_parser.add_argument("--run-id-b", default=spec.smoke_run_id_b)

    replay_parser = subparsers.add_parser("fresh-replay")
    replay_parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    replay_parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    replay_parser.add_argument("--run-id", default=spec.formal_run_id)
    replay_parser.add_argument("--device", default="cuda:0")
    return parser


def main(model_key: str, argv: Sequence[str] | None = None) -> int:
    spec = get_spec(model_key)
    args = _parser(spec).parse_args(argv)
    repo_root = args.repo_root.resolve()
    results_root = (
        args.results_root.resolve()
        if args.results_root.is_absolute()
        else (repo_root / args.results_root).resolve()
    )
    if args.command == "run":
        artifacts_root = (
            args.artifacts_root.resolve()
            if args.artifacts_root.is_absolute()
            else (repo_root / args.artifacts_root).resolve()
        )
        dataset_manifest = (
            args.dataset_manifest.resolve()
            if args.dataset_manifest.is_absolute()
            else (repo_root / args.dataset_manifest).resolve()
        )
        run_id = args.run_id or _default_run_id(
            spec, args.mode, args.smoke_label
        )
        run_experiment(
            spec,
            repo_root=repo_root,
            dataset_manifest=dataset_manifest,
            results_root=results_root,
            artifacts_root=artifacts_root,
            run_id=run_id,
            mode=args.mode,
            device_name=args.device,
            seed=args.seed,
            resume=args.resume,
            sample_id=args.sample_id,
        )
        return 0
    if args.command == "analyze":
        dataset_manifest = (
            args.dataset_manifest.resolve()
            if args.dataset_manifest.is_absolute()
            else (repo_root / args.dataset_manifest).resolve()
        )
        metrics = analyze_formal(
            spec,
            repo_root=repo_root,
            dataset_manifest=dataset_manifest,
            results_root=results_root,
            run_id=args.run_id,
            iterations=args.bootstrap_iterations,
        )
        print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)
        return 0
    if args.command == "compare-smokes":
        result = compare_smokes(
            spec,
            repo_root=repo_root,
            results_root=results_root,
            run_id_a=args.run_id_a,
            run_id_b=args.run_id_b,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        return 0
    if args.command == "fresh-replay":
        result = fresh_replay(
            spec,
            repo_root=repo_root,
            results_root=results_root,
            run_id=args.run_id,
            device_name=args.device,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        return 0
    raise AssertionError(args.command)
