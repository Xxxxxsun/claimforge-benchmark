#!/usr/bin/env python3
"""Run frozen Effort inference on the Balanced250 score cache.

This is a versioned v2 orchestration layer.  The Mouse-v1 runner remains
unchanged; model loading, preprocessing, forward inference, official golden
preflight, and NPZ artifact replay are delegated to that frozen adapter.

Effort is a whole-image T1 detector.  It receives every one of the seven
Balanced250 conditions exactly once and never emits or claims a T2 map.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
import time
import traceback
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from eval.opensource import run_effort as legacy
from eval.opensource.balanced_run_contract import (
    RESULT_SCHEMA_VERSION,
    ScoreSpec,
    build_result_identity,
    build_run_dataset_contract,
    index_latest_attempts,
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


RUN_MANIFEST_SCHEMA = "effort_balanced_run_manifest_v2"
RUN_CONFIG_SCHEMA = "effort_balanced_run_config_v2"
RUNTIME_SUMMARY_SCHEMA = "effort_balanced_runtime_summary_v2"

DEFAULT_DATASET_MANIFEST = Path("outputs/opensource/balanced250_v1/manifest.json")
DEFAULT_RESULTS_DIR = legacy.DEFAULT_RESULTS_DIR
DEFAULT_ARTIFACTS_DIR = Path("outputs/opensource/effort")
DEFAULT_FORMAL_RUN_ID = (
    "effort_clip_l14_genimage_sdv14_balanced250_v1_full1775_r2_20260727"
)
DEFAULT_SMOKE_RUN_ID_A = (
    "effort_clip_l14_genimage_sdv14_balanced250_v1_smoke5x7_a_r3_20260727"
)
DEFAULT_SMOKE_RUN_ID_B = (
    "effort_clip_l14_genimage_sdv14_balanced250_v1_smoke5x7_b_r3_20260727"
)
DEFAULT_SMOKE_LIMIT = 5

SCORE_SPEC = ScoreSpec(
    key="ai_score",
    direction="higher_means_fake",
    fixed_threshold=legacy.CLASSIFICATION_THRESHOLD,
    threshold_operator=legacy.CLASSIFICATION_THRESHOLD_OPERATOR,
)

PREPROCESS_CONTRACT = {
    "decode": "cv2.imread_IMREAD_COLOR",
    "color": "cv2_BGR2RGB",
    "resize": "cv2_resize_224x224_INTER_LINEAR_no_aspect_preservation",
    "to_tensor": "uint8_to_float32_divide_255_CHW",
    "mean": list(legacy.CLIP_MEAN),
    "std": list(legacy.CLIP_STD),
    "crop": None,
    "face_alignment": False,
    "known_official_drift": (
        "DeepfakeBench dataset test path uses INTER_CUBIC; the released "
        "natural-image README demo uses INTER_LINEAR and is frozen here"
    ),
}

ARTIFACT_CONTRACT = {
    "format": "NumPy NPZ, allow_pickle=False",
    "keys": ["pooler_output", "class_logits"],
    "pooler_output": {
        "shape": [legacy.FEATURE_DIMENSION],
        "dtype": "float32",
        "semantics": legacy.FEATURE_SEMANTICS,
    },
    "class_logits": {
        "shape": [legacy.CLASS_COUNT],
        "dtype": "float32",
        "semantics": "official_effort_two_class_logits",
    },
    "file_bytes": 4_640,
    "exact_head_and_float32_softmax_replay": True,
    "storage": "local_gitignored_outputs",
}

FORMAL_COUNTS = {
    "real": 275,
    "local_mouse": 250,
    "local_cat": 250,
    "local_trash_can": 250,
    "fullframe_mouse": 250,
    "fullframe_cat": 250,
    "fullframe_trash_can": 250,
}

ADAPTER_SOURCE_PATHS = (
    ".gitignore",
    "eval/__init__.py",
    "eval/opensource/__init__.py",
    "eval/opensource/run_effort_balanced.py",
    "eval/opensource/analyze_effort_balanced.py",
    "eval/opensource/analyze_effort_run.py",
    "eval/opensource/run_effort.py",
    "eval/opensource/canonical_release.py",
    "eval/opensource/balanced_run_contract.py",
    "eval/opensource/balanced250_metrics.py",
    "eval/opensource/common.py",
    "eval/opensource/effort_metrics.py",
    "eval/opensource/ufd_metrics.py",
)


def _anchored(path: Path, repo_root: Path) -> Path:
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


def _fingerprint(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(stable_json(value).encode("utf-8")).hexdigest()


def _rows_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    return hashlib.sha256(
        "".join(f"{stable_json(row)}\n" for row in rows).encode("utf-8")
    ).hexdigest()


def _valid_run_id(value: Any) -> str:
    if (
        not isinstance(value, str)
        or not value
        or Path(value).name != value
        or value in (".", "..")
    ):
        raise ValueError("run-id must be one non-empty path component")
    return value


def adapter_source_contract(repo_root: Path) -> dict[str, dict[str, Any]]:
    """Hash every local source that participates in inference or audit."""

    result: dict[str, dict[str, Any]] = {}
    for relative in ADAPTER_SOURCE_PATHS:
        path = repo_root / relative
        if not path.is_file():
            raise FileNotFoundError(f"missing Effort Balanced adapter source: {path}")
        result[relative] = {
            "path": relative,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    return result


def _formal_selection(
    release: CanonicalRelease,
) -> tuple[SelectionSpec, list[dict[str, Any]]]:
    spec = SelectionSpec(capability=Capability.WHOLE_IMAGE_T1)
    selected = select_inputs(release, spec)
    counts = Counter(str(row["condition"]) for row in selected)
    if (
        release.schema_version != BALANCED_SCHEMA
        or release.dataset_id != BALANCED_DATASET_ID
        or dict(counts) != FORMAL_COUNTS
        or len(selected) != 1775
        or [str(row["sample_id"]) for row in selected]
        != [str(row["sample_id"]) for row in release.inputs]
    ):
        raise ValueError("formal Balanced250 selection drifted")
    return spec, selected


def _smoke_selection(
    release: CanonicalRelease,
    per_condition_limit: int,
) -> tuple[SelectionSpec, list[dict[str, Any]]]:
    if (
        isinstance(per_condition_limit, bool)
        or not isinstance(per_condition_limit, int)
        or per_condition_limit != DEFAULT_SMOKE_LIMIT
    ):
        raise ValueError(
            f"smoke per-condition-limit must be exactly {DEFAULT_SMOKE_LIMIT}"
        )
    spec = SelectionSpec(
        capability=Capability.WHOLE_IMAGE_T1,
        per_condition_limit=per_condition_limit,
    )
    inputs_by_id = {str(row["sample_id"]): row for row in release.inputs}
    counts: Counter[str] = Counter()
    selected: list[dict[str, Any]] = []
    for panel_row in release.panel:
        condition = str(panel_row["condition"])
        if (
            condition in Capability.WHOLE_IMAGE_T1.conditions
            and counts[condition] < per_condition_limit
        ):
            sample_id = str(panel_row["sample_id"])
            source = inputs_by_id.get(sample_id)
            if source is None or source.get("panel") is not True:
                raise ValueError("smoke panel has a dangling/non-panel input")
            selected.append(source)
            counts[condition] += 1
    expected = {condition: per_condition_limit for condition in BALANCED_CONDITIONS}
    if dict(counts) != expected:
        raise ValueError("smoke panel does not cover every condition")
    selected.sort(key=lambda row: int(row["rank"]))
    return spec, selected


def select_mode_inputs(
    release: CanonicalRelease,
    *,
    mode: str,
    per_condition_limit: int | None,
    sample_id: str | None,
) -> tuple[SelectionSpec, list[dict[str, Any]]]:
    """Create the exact selection for formal, panel-smoke, or single mode."""

    if release.release_kind != "balanced250":
        raise ValueError("Effort v2 requires a Balanced250 release")
    if mode == "formal":
        if per_condition_limit is not None or sample_id is not None:
            raise ValueError("formal mode does not accept input selectors")
        return _formal_selection(release)
    if mode == "smoke":
        if sample_id is not None:
            raise ValueError("smoke mode does not accept sample-id")
        return _smoke_selection(
            release,
            (
                DEFAULT_SMOKE_LIMIT
                if per_condition_limit is None
                else per_condition_limit
            ),
        )
    if mode == "single":
        if per_condition_limit is not None:
            raise ValueError("single mode does not accept per-condition-limit")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError("single mode requires --sample-id")
        spec = SelectionSpec(
            capability=Capability.WHOLE_IMAGE_T1,
            sample_id=sample_id,
        )
        return spec, select_inputs(release, spec)
    raise ValueError(f"unsupported inference mode {mode!r}")


def _visibility(row: Mapping[str, Any]) -> dict[str, Any]:
    gt_kind = row.get("gt_mask_kind")
    if gt_kind == "exact_diff":
        return {
            "edit_visibility": "full",
            "edit_visible_gt_fraction": 1.0,
            "edit_visibility_evidence": {
                "basis": "full_canvas_direct_resize_without_crop",
                "preprocess_profile": legacy.PREPROCESS_PROFILE,
            },
        }
    if gt_kind not in ("all_zero", "not_applicable"):
        raise ValueError("unsupported Balanced250 GT kind")
    return {
        "edit_visibility": "not_applicable",
        "edit_visible_gt_fraction": None,
        "edit_visibility_evidence": {
            "basis": (
                "authentic_input"
                if gt_kind == "all_zero"
                else "fullframe_has_no_local_GT"
            ),
            "preprocess_profile": legacy.PREPROCESS_PROFILE,
        },
    }


def result_identity(
    row: Mapping[str, Any],
    *,
    repo_root: Path,
    run_id: str,
    run_manifest_fingerprint: str,
    valid_for_metrics: bool,
) -> dict[str, Any]:
    """Build Effort's runner-specific extension of the shared v2 ID."""

    if type(valid_for_metrics) is not bool:
        raise ValueError("valid_for_metrics must be boolean")
    path = _anchored(Path(str(row["canonical_path"])), repo_root)
    return {
        **build_result_identity(
            row,
            run_id=run_id,
            run_manifest_fingerprint=run_manifest_fingerprint,
        ),
        "valid_for_metrics": valid_for_metrics,
        "dataset_id": str(row["dataset_id"]),
        "input_path": repo_relative(path, repo_root),
        "input_sha256": str(row["canonical_sha256"]),
        "input_width": int(row["width"]),
        "input_height": int(row["height"]),
        "model": legacy.MODEL_NAME,
        "model_slug": legacy.MODEL_SLUG,
        "preprocess_profile": legacy.PREPROCESS_PROFILE,
        "checkpoint_id": str(legacy.CHECKPOINT["id"]),
        "config_fingerprint": run_manifest_fingerprint,
        **_visibility(row),
        "task_scope": {
            "valid_for_t1": True,
            "valid_for_t2": False,
            "native_dense_output": False,
        },
    }


def _validate_runner_attempt(
    attempt: Mapping[str, Any],
    *,
    input_row: Mapping[str, Any],
    repo_root: Path,
    run_id: str,
    run_manifest_fingerprint: str,
) -> None:
    status = attempt.get("status")
    if status not in ("ok", "error"):
        raise ValueError("result attempt has invalid status")
    expected = result_identity(
        input_row,
        repo_root=repo_root,
        run_id=run_id,
        run_manifest_fingerprint=run_manifest_fingerprint,
        valid_for_metrics=status == "ok",
    )
    for key, value in expected.items():
        if attempt.get(key) != value:
            raise ValueError(f"result attempt field {key} drifted")
    unexpected = _forbidden_t2_claims(attempt)
    if unexpected:
        raise ValueError(
            "Effort result contains T2 payload " f"{sorted(unexpected)[0]!r}"
        )


_T2_EXACT_KEYS = frozenset(
    {
        "t2",
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
    "localization",
    "localisation",
    "mask",
    "score_map",
    "predicted_mask",
)


def _forbidden_t2_claims(
    value: Any,
    path: tuple[str, ...] = (),
) -> set[str]:
    """Find dense/localization claims at any nesting depth."""

    found: set[str] = set()
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key)
            normalized = key.lower()
            child_path = (*path, key)
            rendered = ".".join(child_path)
            if normalized == "valid_for_t2":
                if child is not False:
                    found.add(rendered)
            elif normalized == "localization_output":
                if child is not None:
                    found.add(rendered)
            elif normalized in _T2_EXACT_KEYS or normalized.startswith(_T2_PREFIXES):
                found.add(rendered)
            found.update(_forbidden_t2_claims(child, child_path))
    elif isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        for index, child in enumerate(value):
            found.update(_forbidden_t2_claims(child, (*path, str(index))))
    return found


def build_immutable_run_config(
    *,
    repo_root: Path,
    run_id: str,
    mode: str,
    dataset_contract: Mapping[str, Any],
    selected: Sequence[Mapping[str, Any]],
    adapter_sources: Mapping[str, Any],
    source_audit: Mapping[str, Any],
    asset_audit: Mapping[str, Any],
    model_audit: Mapping[str, Any],
    runtime_golden: Mapping[str, Any],
    runtime: Mapping[str, Any],
    cpu_preflight: Mapping[str, Any],
    results_path: Path,
    expected_inputs_path: Path,
    summary_path: Path,
    artifact_dir: Path,
) -> dict[str, Any]:
    return {
        "schema_version": RUN_CONFIG_SCHEMA,
        "run_id": run_id,
        "mode": mode,
        "adapter_sources": dict(adapter_sources),
        "model": {
            "name": legacy.MODEL_NAME,
            "slug": legacy.MODEL_SLUG,
            "architecture": legacy.MODEL_ARCH,
            "repository": legacy.MODEL_REPO_URL,
            "source_commit": legacy.MODEL_SOURCE_COMMIT,
            "checkpoint_id": legacy.CHECKPOINT["id"],
            "checkpoint_sha256": legacy.CHECKPOINT["sha256"],
            "checkpoint_bytes": legacy.CHECKPOINT["bytes"],
        },
        "preprocess": {
            "profile": legacy.PREPROCESS_PROFILE,
            "contract": PREPROCESS_CONTRACT,
            "batch_size": 1,
            "autocast": False,
        },
        "score_spec": SCORE_SPEC.as_dict(),
        "task_scope": {
            "primary_task": "T1_whole_image_AIGC_detection",
            "valid_for_t1": True,
            "valid_for_t2": False,
            "localization_output": None,
        },
        "dataset_contract": dict(dataset_contract),
        "selected_rows_sha256": _rows_sha256(selected),
        "selected_ids_sha256": selected_ids_sha256(
            str(row["sample_id"]) for row in selected
        ),
        "source": dict(source_audit),
        "assets": dict(asset_audit),
        "model_audit": dict(model_audit),
        "runtime_golden": dict(runtime_golden),
        "runtime": dict(runtime),
        "cpu_preflight": {
            "performed_before_accelerator_configuration": True,
            "report": dict(cpu_preflight),
        },
        "license": dict(legacy.LICENSE_RECORD),
        "artifact_contract": ARTIFACT_CONTRACT,
        "outputs": {
            "results_path": repo_relative(results_path, repo_root),
            "expected_inputs_path": repo_relative(
                expected_inputs_path,
                repo_root,
            ),
            "summary_path": repo_relative(summary_path, repo_root),
            "artifact_dir": repo_relative(artifact_dir, repo_root),
        },
    }


def _validate_artifact_inventory(
    artifact_dir: Path,
    latest: Mapping[str, Mapping[str, Any]],
) -> int:
    expected = {
        f"{sample_id}.npz"
        for sample_id, row in latest.items()
        if row.get("status") == "ok"
    }
    entries = list(artifact_dir.iterdir()) if artifact_dir.is_dir() else []
    if any(
        entry.is_symlink() or not entry.is_file() or entry.suffix != ".npz"
        for entry in entries
    ):
        raise ValueError("Effort artifact inventory contains an unsafe entry")
    actual = {path.name for path in entries}
    if actual != expected:
        raise ValueError(
            "Effort artifact inventory mismatch: "
            f"missing={sorted(expected - actual)[:1]}, "
            f"extra={sorted(actual - expected)[:1]}"
        )
    return len(actual)


def run_cpu_preflight(
    *,
    source_root: Path,
    checkpoint_path: Path,
    hf_config_path: Path,
) -> dict[str, Any]:
    """Verify the frozen assets and CPU fixture before selecting CUDA."""

    import torch

    if torch.cuda.is_initialized():
        raise RuntimeError("Effort CPU preflight started after CUDA initialization")
    source = legacy.verify_source(source_root)
    device, runtime = legacy.configure_runtime("cpu")
    assets, state, config_payload = legacy.verify_assets(
        checkpoint_path,
        hf_config_path,
    )
    model, model_audit = legacy._build_model(state, config_payload, device)
    del state
    try:
        golden = legacy.validate_runtime_golden(model, device, source_root)
    finally:
        del model
        gc.collect()
    if torch.cuda.is_initialized():
        raise RuntimeError("Effort CPU preflight initialized CUDA")
    return {
        "schema_version": "effort_balanced_cpu_preflight_v1",
        "status": "passed",
        "source": source,
        "assets": assets,
        "model_audit": model_audit,
        "runtime": runtime,
        "runtime_golden": golden,
        "accelerator_model_forwards": 0,
        "balanced250_model_scores_computed": 0,
        "cuda_initialized_before": False,
        "cuda_initialized_after": False,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path)
    parser.add_argument("--source-root", type=Path, default=legacy.DEFAULT_SOURCE_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=legacy.DEFAULT_CHECKPOINT)
    parser.add_argument("--hf-config", type=Path, default=legacy.DEFAULT_HF_CONFIG)
    parser.add_argument(
        "--dataset-manifest",
        type=Path,
        default=DEFAULT_DATASET_MANIFEST,
    )
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=DEFAULT_ARTIFACTS_DIR,
    )
    parser.add_argument("--run-id")
    parser.add_argument(
        "--mode",
        choices=("formal", "smoke", "single", "preflight"),
        default="formal",
    )
    parser.add_argument("--per-condition-limit", type=int)
    parser.add_argument("--sample-id")
    parser.add_argument(
        "--device",
        help="explicit cpu or cuda:N; inference defaults to cuda:0",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser


def run(args: argparse.Namespace) -> int:
    repo_root = (
        Path(args.repo_root).resolve()
        if args.repo_root is not None
        else Path(__file__).resolve().parents[2]
    )
    source_root = _anchored(args.source_root, repo_root)
    checkpoint_path = _anchored(args.checkpoint, repo_root)
    hf_config_path = _anchored(args.hf_config, repo_root)

    if args.mode == "preflight":
        if (
            args.resume
            or args.sample_id is not None
            or args.per_condition_limit is not None
            or (args.device is not None and args.device != "cpu")
        ):
            raise ValueError("preflight accepts no selection/resume/CUDA options")
        report = run_cpu_preflight(
            source_root=source_root,
            checkpoint_path=checkpoint_path,
            hf_config_path=hf_config_path,
        )
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
        return 0

    run_id = _valid_run_id(args.run_id or DEFAULT_FORMAL_RUN_ID)
    if args.mode != "formal" and args.run_id is None:
        raise ValueError("smoke and single modes require an explicit --run-id")
    device_text = args.device or "cuda:0"
    dataset_manifest_path = _anchored(args.dataset_manifest, repo_root)
    results_root = _anchored(args.results_dir, repo_root)
    expected_results_root = (repo_root / DEFAULT_RESULTS_DIR).resolve()
    if results_root != expected_results_root:
        raise ValueError(f"--results-dir must be exactly {DEFAULT_RESULTS_DIR}")
    artifacts_root = _anchored(args.artifacts_dir, repo_root)
    expected_artifacts_root = (repo_root / DEFAULT_ARTIFACTS_DIR).resolve()
    if artifacts_root != expected_artifacts_root:
        raise ValueError(f"--artifacts-dir must be exactly {DEFAULT_ARTIFACTS_DIR}")
    run_dir = results_root / run_id
    artifact_root = artifacts_root / run_id
    artifact_dir = artifact_root / "artifacts"
    if (
        run_dir == artifact_root
        or run_dir.is_relative_to(artifact_root)
        or artifact_root.is_relative_to(run_dir)
    ):
        raise ValueError("Effort result and artifact directories must be disjoint")
    results_path = run_dir / "results.jsonl"
    expected_path = run_dir / "expected_inputs.jsonl"
    summary_path = run_dir / "summary.json"
    manifest_path = run_dir / "manifest.json"
    if run_dir.exists() and any(run_dir.iterdir()) and not args.resume:
        raise FileExistsError(f"run directory is non-empty; pass --resume: {run_dir}")
    if artifact_root.exists() and any(artifact_root.iterdir()) and not args.resume:
        raise FileExistsError(
            "artifact directory is non-empty; pass --resume: " f"{artifact_root}"
        )

    release = load_canonical_release(
        repo_root,
        dataset_manifest_path,
        verify_files=True,
    )
    selection_spec, selected = select_mode_inputs(
        release,
        mode=args.mode,
        per_condition_limit=args.per_condition_limit,
        sample_id=args.sample_id,
    )
    dataset_contract = build_run_dataset_contract(
        release,
        selection_spec,
        selected,
        score_spec=SCORE_SPEC,
    )

    # This CPU golden gate deliberately precedes any accelerator configuration.
    cpu_preflight = run_cpu_preflight(
        source_root=source_root,
        checkpoint_path=checkpoint_path,
        hf_config_path=hf_config_path,
    )
    source_audit = legacy.verify_source(source_root)
    asset_audit, state, config_payload = legacy.verify_assets(
        checkpoint_path,
        hf_config_path,
    )
    if (
        cpu_preflight.get("source") != source_audit
        or cpu_preflight.get("assets") != asset_audit
    ):
        raise ValueError("CPU preflight asset/source evidence drifted")
    device, runtime = legacy.configure_runtime(device_text)
    model, model_audit = legacy._build_model(state, config_payload, device)
    del state
    runtime_golden = legacy.validate_runtime_golden(
        model,
        device,
        source_root,
    )
    if model_audit != cpu_preflight.get("model_audit") or runtime_golden.get(
        "kind"
    ) != cpu_preflight.get("runtime_golden", {}).get("kind"):
        raise ValueError("Effort model/preflight contract drifted")

    adapter_sources = adapter_source_contract(repo_root)
    immutable = build_immutable_run_config(
        repo_root=repo_root,
        run_id=run_id,
        mode=args.mode,
        dataset_contract=dataset_contract.as_dict(),
        selected=selected,
        adapter_sources=adapter_sources,
        source_audit=source_audit,
        asset_audit=asset_audit,
        model_audit=model_audit,
        runtime_golden=runtime_golden,
        runtime=runtime,
        cpu_preflight=cpu_preflight,
        results_path=results_path,
        expected_inputs_path=expected_path,
        summary_path=summary_path,
        artifact_dir=artifact_dir,
    )
    fingerprint = _fingerprint(immutable)

    if args.resume:
        if not manifest_path.is_file() or not expected_path.is_file():
            raise FileNotFoundError(
                "resume requires manifest.json and expected_inputs.jsonl"
            )
        prior_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            prior_manifest.get("schema_version") != RUN_MANIFEST_SCHEMA
            or prior_manifest.get("run_id") != run_id
            or prior_manifest.get("fingerprint") != fingerprint
            or prior_manifest.get("immutable") != immutable
        ):
            raise ValueError("resume run manifest fingerprint/config drifted")
        if read_jsonl(expected_path) != selected:
            raise ValueError("resume expected input snapshot drifted")
        started_at = prior_manifest.get("started_at")
    else:
        atomic_write_jsonl(expected_path, selected)
        started_at = utc_now()

    manifest: dict[str, Any] = {
        "schema_version": RUN_MANIFEST_SCHEMA,
        "run_id": run_id,
        "status": "running",
        "started_at": started_at,
        "completed_at": None,
        "fingerprint": fingerprint,
        "immutable": immutable,
        "dataset": {
            "contract": dataset_contract.as_dict(),
            "manifest_path": repo_relative(dataset_manifest_path, repo_root),
            "manifest_sha256": release.manifest_sha256,
            "expected_inputs_path": repo_relative(expected_path, repo_root),
            "expected_inputs_sha256": sha256_file(expected_path),
            "selected_images": len(selected),
        },
        "outputs": dict(immutable["outputs"]),
    }
    atomic_write_json(manifest_path, manifest)

    physical_before = read_jsonl(results_path) if results_path.is_file() else []
    latest_before = index_latest_attempts(
        selected,
        physical_before,
        run_id=run_id,
        run_manifest_fingerprint=fingerprint,
        score_spec=SCORE_SPEC,
    )
    inputs_by_id = {str(row["sample_id"]): row for row in selected}
    for attempt in physical_before:
        _validate_runner_attempt(
            attempt,
            input_row=inputs_by_id[str(attempt["sample_id"])],
            repo_root=repo_root,
            run_id=run_id,
            run_manifest_fingerprint=fingerprint,
        )

    new_successes = 0
    resume_skips = 0
    new_errors = 0
    fatal_error: BaseException | None = None
    try:
        for index, row in enumerate(selected, start=1):
            sample_id = str(row["sample_id"])
            prior = latest_before.latest_by_sample_id.get(sample_id)
            expected_ok = result_identity(
                row,
                repo_root=repo_root,
                run_id=run_id,
                run_manifest_fingerprint=fingerprint,
                valid_for_metrics=True,
            )
            if prior is not None and prior.get("status") == "ok":
                legacy._validate_resume_row(
                    prior,
                    expected=expected_ok,
                    repo_root=repo_root,
                    model=model,
                    device=device,
                    run_dir=artifact_root,
                )
                resume_skips += 1
                print(
                    f"[{index}/{len(selected)}] resume {sample_id}",
                    flush=True,
                )
                continue

            input_path = _anchored(
                Path(str(row["canonical_path"])),
                repo_root,
            )
            artifact_path: Path | None = None
            try:
                preprocess_started = time.perf_counter()
                tensor, preprocess = legacy.preprocess_image(input_path)
                preprocess_latency_ms = (
                    time.perf_counter() - preprocess_started
                ) * 1000.0
                if preprocess.get("native_width") != int(
                    row["width"]
                ) or preprocess.get("native_height") != int(row["height"]):
                    raise ValueError("preprocessed image dimensions changed")
                scoring, feature, logits, peak, latency = legacy.infer_one(
                    model,
                    device,
                    tensor,
                )
                artifact_path = legacy._artifact_path(
                    artifact_root,
                    sample_id,
                )
                legacy._atomic_save_artifact(artifact_path, feature, logits)
                persisted_feature, persisted_logits = legacy._load_artifact(
                    artifact_path
                )
                if not np.array_equal(feature, persisted_feature):
                    raise ValueError("Effort persisted feature differs")
                if not np.array_equal(logits, persisted_logits):
                    raise ValueError("Effort persisted logits differ")
                if artifact_path.stat().st_size != ARTIFACT_CONTRACT["file_bytes"]:
                    raise ValueError("Effort persisted artifact size changed")
                relative_artifact = repo_relative(artifact_path, repo_root)
                result = {
                    **expected_ok,
                    "status": "ok",
                    "completed_at": utc_now(),
                    "preprocess": preprocess,
                    "preprocess_latency_ms": preprocess_latency_ms,
                    "artifact_path": relative_artifact,
                    "artifact_sha256": sha256_file(artifact_path),
                    "artifact_bytes": artifact_path.stat().st_size,
                    "artifact_keys": ["pooler_output", "class_logits"],
                    "artifact_paths": {"effort_npz": relative_artifact},
                    "feature_shape": list(feature.shape),
                    "feature_dtype": str(feature.dtype),
                    "feature_semantics": legacy.FEATURE_SEMANTICS,
                    "feature_array_sha256": legacy._array_sha256(feature),
                    "class_logits_shape": list(logits.shape),
                    "class_logits_dtype": str(logits.dtype),
                    "class_logits_array_sha256": legacy._array_sha256(logits),
                    "latency_ms": latency,
                    "peak_cuda_memory_bytes": peak,
                    **scoring,
                }
                append_jsonl(results_path, result)
                new_successes += 1
                print(
                    f"[{index}/{len(selected)}] ok {sample_id} "
                    f"score={result['ai_score']:.9f}",
                    flush=True,
                )
            except Exception as error:
                if artifact_path is not None and artifact_path.is_file():
                    artifact_path.unlink()
                new_errors += 1
                error_result = {
                    **result_identity(
                        row,
                        repo_root=repo_root,
                        run_id=run_id,
                        run_manifest_fingerprint=fingerprint,
                        valid_for_metrics=False,
                    ),
                    "status": "error",
                    "completed_at": utc_now(),
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                }
                append_jsonl(results_path, error_result)
                print(
                    f"[{index}/{len(selected)}] error {sample_id}: {error}",
                    file=sys.stderr,
                    flush=True,
                )
                if args.fail_fast:
                    fatal_error = error
                    break
            finally:
                gc.collect()
                if getattr(device, "type", None) == "cuda":
                    __import__("torch").cuda.empty_cache()
    finally:
        del model
        gc.collect()

    physical_results = read_jsonl(results_path) if results_path.is_file() else []
    latest = index_latest_attempts(
        selected,
        physical_results,
        run_id=run_id,
        run_manifest_fingerprint=fingerprint,
        score_spec=SCORE_SPEC,
    )
    for attempt in physical_results:
        _validate_runner_attempt(
            attempt,
            input_row=inputs_by_id[str(attempt["sample_id"])],
            repo_root=repo_root,
            run_id=run_id,
            run_manifest_fingerprint=fingerprint,
        )
    coverage = summarize_coverage(latest)
    artifact_files = _validate_artifact_inventory(
        artifact_dir,
        latest.latest_by_sample_id,
    )
    summary = {
        "schema_version": RUNTIME_SUMMARY_SCHEMA,
        "summary_kind": "runtime_coverage_only",
        "scientific_metrics": None,
        "scientific_metrics_owner": "analyze_effort_balanced.py",
        "run_id": run_id,
        "run_manifest_fingerprint": fingerprint,
        "status": "complete" if coverage.is_complete else "incomplete",
        "mode": args.mode,
        "model": legacy.MODEL_NAME,
        "model_slug": legacy.MODEL_SLUG,
        "score_spec": SCORE_SPEC.as_dict(),
        "dataset_contract": dataset_contract.as_dict(),
        "coverage": coverage.as_dict(),
        "generated_at": utc_now(),
    }
    atomic_write_json(summary_path, summary)

    manifest["status"] = summary["status"]
    manifest["completed_at"] = utc_now()
    manifest["execution"] = {
        "new_successes": new_successes,
        "resume_skips": resume_skips,
        "new_errors": new_errors,
        "physical_result_rows": len(physical_results),
        "latest_result_rows": len(latest.latest_by_sample_id),
        "superseded_attempts": latest.superseded_attempts,
    }
    manifest["outputs"].update(
        {
            "results_sha256": sha256_file(results_path),
            "summary_sha256": sha256_file(summary_path),
            "artifact_files": artifact_files,
        }
    )
    atomic_write_json(manifest_path, manifest)
    print(
        json.dumps(
            {
                "run_id": run_id,
                "status": manifest["status"],
                "mode": args.mode,
                "coverage": coverage.as_dict(),
                "manifest": str(manifest_path),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    if fatal_error is not None:
        raise RuntimeError("Effort fail-fast inference failed") from fatal_error
    return 0 if coverage.is_complete else 2


def main(argv: list[str] | None = None) -> int:
    return run(_build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
