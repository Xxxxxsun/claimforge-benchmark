"""Independently audit and analyze a paired official NFA-ViT BR-Gen run.

The analyzer is deliberately separate from model inference.  It replays the
native classifier sigmoid, the 128->512 segmentation-logit resize, sigmoid,
512->native probability adapter, strict masks, GT resize, and every recorded
metric from persisted artifacts.  It also validates run provenance, recomputes
paired T1/T2/bootstrap summaries, reports domain/area diagnostics, and can
verify that a smoke run is an exact deterministic prefix of a full run.

Frozen result-schema assumptions
--------------------------------
The canonical row has ``row.score`` and
``classification={raw_logit, probability|score, decision, threshold,
threshold_operator}``; explicit early-run top-level ``classification_*``
aliases accepted by :mod:`nfa_vit_metrics` are also accepted here.  Dense
artifacts may be represented as nested ``artifact_paths``/``artifacts``
metadata or as flat ``<stem>_{path,sha256,shape,dtype}`` fields.  Only the
documented aliases in ``ARTIFACT_ALIASES`` are supported.  No mask mean, max,
or other dense-map statistic is ever accepted as a T1 score.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from eval.opensource.common import (
    atomic_write_json,
    read_jsonl,
    sha256_file,
    stable_json,
    utc_now,
)
from eval.opensource.nfa_vit_metrics import (
    FIXED_CLASSIFICATION_THRESHOLD,
    FIXED_MASK_THRESHOLD,
    LOCALIZATION_SPACES,
    binary_pixel_metrics_strict,
    summarize_nfa_vit_pair_slice,
    summarize_nfa_vit_results,
)


DEFAULT_RUN_ID = "nfa_vit_br_gen_mouse_canonical_v1_full275_20260724"
DEFAULT_RESULTS_DIR = Path("results/opensource/nfa_vit")
DEFAULT_INPUTS = Path("outputs/opensource/mouse_canonical_v1/inputs.jsonl")
DEFAULT_NFA_VIT_ROOT = Path("/root/.cache/claimforge/third_party/BR-Gen")
DEFAULT_IMDLBENCO_ROOT = Path(
    "/root/.cache/claimforge/third_party/IMDLBenCo-4e55633c"
)

MODEL_NAME = "NFA-ViT"
BR_GEN_SOURCE_COMMIT = "4ced0e0966e96b9bd637cb34aa4ab8ab8eade782"
IMDLBENCO_SOURCE_COMMIT = "4e55633c3e68ede63974f72ea9af1a803a7f5ae8"
MODEL_SIZE = 512
RAW_LOGIT_SIZE = 128
THRESHOLD_OPERATOR = ">"

RESIZED_LOGITS_ABSOLUTE_TOLERANCE = 6e-6
MODEL_PROBABILITY_ABSOLUTE_TOLERANCE = 1e-6
NATIVE_PROBABILITY_ABSOLUTE_TOLERANCE = 1e-5
END_TO_END_NATIVE_ABSOLUTE_TOLERANCE = 2e-5
TRANSFORM_RELATIVE_TOLERANCE = 1e-7


ARTIFACT_ALIASES: dict[str, tuple[str, ...]] = {
    "raw_logits_128": (
        "decoder_logits_128",
        "seg_logits_raw_128",
        "raw_seg_logits_128",
        "raw_logits_model",
    ),
    "resized_logits_512": (
        "resized_logits_512",
        "seg_logits_512",
        "seg_logits_resized_512",
        "resized_logits_model",
    ),
    "probability_512": (
        "probability_512",
        "seg_probability_512",
        "score_map_model",
    ),
    "probability_native": (
        "probability_native",
        "seg_probability_native",
        "score_map_native",
        "score_map",
    ),
    "mask_native": (
        "mask_native",
        "mask_native_strict_gt_0_5",
        "mask",
    ),
}


@dataclass(frozen=True)
class Artifact:
    path: Path
    sha256: str
    shape: tuple[int, ...] | None
    dtype: str | None


@dataclass(frozen=True)
class Pair:
    task_id: str
    real: dict[str, Any]
    forged: dict[str, Any]
    expected_real: dict[str, Any]
    expected_forged: dict[str, Any]


@dataclass(frozen=True)
class KernelReplay:
    """Torch runtime/device proven equivalent to the recorded inference run."""

    torch: Any
    device: Any
    requested_device: str
    torch_version: str


def _cv2_module() -> Any:
    """Load the pinned OpenCV runtime only when protocol replay needs it."""

    try:
        import cv2
    except (ImportError, RuntimeError) as exc:
        raise RuntimeError(
            "NFA-ViT artifact audit requires a NumPy-compatible OpenCV "
            "runtime; run the analyzer in the frozen NFA-ViT environment"
        ) from exc
    return cv2


def _anchored(path: Path, repo_root: Path) -> Path:
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


def _relative_or_absolute(path: Path, repo_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo_root.resolve()))
    except ValueError:
        return str(path.resolve())


def _require_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} is not an object")
    return dict(value)


def _require_sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} is not a lowercase SHA-256")
    return value


def _verify_hash(path: Path, expected: Any, label: str) -> None:
    digest = _require_sha256(expected, f"{label} recorded SHA-256")
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")
    actual = sha256_file(path)
    if actual != digest:
        raise ValueError(f"{label} SHA-256 mismatch: {actual} != {digest}")


def _array_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(array).tobytes(order="C")
    ).hexdigest()


def _git_value(repository: Path, *arguments: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repository), *arguments],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _manifest_fingerprint(manifest: Mapping[str, Any]) -> str:
    immutable = {
        key: value
        for key, value in manifest.items()
        if key not in {"fingerprint", "created_at", "adapter", "environment"}
    }
    return hashlib.sha256(stable_json(immutable).encode("utf-8")).hexdigest()


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
    manifest: Mapping[str, Any],
) -> list[dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(input_rows, start=1):
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError(f"canonical input row {index} has no sample_id")
        if sample_id in by_id:
            raise ValueError(f"canonical inputs repeat ID {sample_id}")
        by_id[sample_id] = row
    ordered = manifest.get("ordered_inputs")
    if not isinstance(ordered, list) or not ordered:
        raise ValueError("run manifest ordered_inputs is empty or invalid")
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, item_value in enumerate(ordered):
        item = _require_mapping(item_value, f"ordered input {index}")
        sample_id = item.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError(f"ordered input {index} has no sample_id")
        if sample_id in seen:
            raise ValueError(f"ordered_inputs repeats ID {sample_id}")
        if sample_id not in by_id:
            raise ValueError(f"ordered_inputs selected unknown ID {sample_id}")
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
            raise ValueError(f"result row {line_number} has no valid id")
        histories[row_id].append((line_number, row))
        status_counts[str(row.get("status"))] += 1

    latest_counts: Counter[str] = Counter()
    duplicates: list[dict[str, Any]] = []
    recovered: list[str] = []
    for row_id, entries in sorted(histories.items()):
        statuses = [str(row.get("status")) for _, row in entries]
        latest_counts[statuses[-1]] += 1
        if len(entries) > 1:
            duplicates.append(
                {
                    "id": row_id,
                    "physical_rows": len(entries),
                    "line_numbers": [line for line, _ in entries],
                    "statuses": statuses,
                }
            )
        if statuses[-1] == "ok" and "error" in statuses[:-1]:
            recovered.append(row_id)
    return {
        "physical_rows": len(result_rows),
        "unique_ids": len(histories),
        "duplicate_rows": len(result_rows) - len(histories),
        "ids_with_multiple_rows": len(duplicates),
        "recovered_error_to_ok": len(recovered),
        "recovered_ids": recovered,
        "historical_status_counts": dict(sorted(status_counts.items())),
        "latest_status_counts": dict(sorted(latest_counts.items())),
        "duplicate_histories": duplicates,
        "latest_policy": "last physical JSONL row for each sample id",
    }


def _latest_by_id(
    rows: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows, start=1):
        row_id = row.get("id")
        if not isinstance(row_id, str) or not row_id:
            raise ValueError(f"result row {index} has no valid id")
        latest[row_id] = row
    return latest


def _normalise_source_files(value: Any, label: str) -> list[dict[str, str]]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{label} is empty or invalid")
    result: list[dict[str, str]] = []
    seen: set[str] = set()
    for index, raw in enumerate(value):
        item = _require_mapping(raw, f"{label} entry {index}")
        path = item.get("path")
        if not isinstance(path, str) or not path or Path(path).is_absolute():
            raise ValueError(f"{label} entry {index} has invalid relative path")
        if path in seen:
            raise ValueError(f"{label} repeats path {path}")
        seen.add(path)
        result.append(
            {
                "path": path,
                "sha256": _require_sha256(
                    item.get("sha256"),
                    f"{label} entry {index} SHA-256",
                ),
            }
        )
    return result


def _verify_source_tree(
    source: Mapping[str, Any],
    *,
    root: Path,
    expected_commit: str,
    label: str,
) -> int:
    commit = source.get("source_commit", source.get("commit"))
    if commit != expected_commit:
        raise ValueError(f"{label} source commit mismatch")
    root_recorded = source.get("source_root")
    if isinstance(root_recorded, str):
        if Path(root_recorded).resolve() != root.resolve():
            raise ValueError(f"{label} source root mismatch")
    files = _normalise_source_files(source.get("source_files"), f"{label} files")
    for item in files:
        _verify_hash(
            root / item["path"],
            item["sha256"],
            f"{label} source {item['path']}",
        )
    observed_head = _git_value(root, "rev-parse", "HEAD")
    if observed_head is not None and observed_head != expected_commit:
        raise ValueError(f"{label} working tree HEAD mismatch")
    tracked_changes = _git_value(root, "status", "--short", "--untracked-files=no")
    if tracked_changes not in (None, ""):
        raise ValueError(f"{label} source tree has tracked modifications")
    return len(files)


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
        _verify_hash(path, item.get("sha256"), f"adapter contract {index}")
    return len(paths)


def _find_mapping(
    value: Any,
    *,
    predicate: Any,
) -> dict[str, Any] | None:
    if isinstance(value, Mapping):
        mapping = dict(value)
        if predicate(mapping):
            return mapping
        for child in mapping.values():
            found = _find_mapping(child, predicate=predicate)
            if found is not None:
                return found
    elif isinstance(value, list):
        for child in value:
            found = _find_mapping(child, predicate=predicate)
            if found is not None:
                return found
    return None


def _frozen_runner_checkpoint_sha256(
    explicit_test_contract: str | None,
) -> str:
    """Return the runner's frozen official checkpoint digest.

    Production analysis is deliberately fail-closed while the runner's
    release contract still has ``sha256=None``.  Tests may pass an explicit
    fixture digest through ``explicit_test_contract``; the CLI exposes no
    override, so this cannot silently bless arbitrary production bytes.
    """

    if explicit_test_contract is not None:
        return _require_sha256(
            explicit_test_contract,
            "explicit test checkpoint SHA-256 contract",
        )
    from eval.opensource.run_nfa_vit import CHECKPOINT

    frozen = CHECKPOINT.get("sha256")
    if frozen is None:
        raise ValueError(
            "official checkpoint SHA-256 is not frozen in "
            "eval.opensource.run_nfa_vit.CHECKPOINT; production provenance "
            "audit is fail-closed"
        )
    return _require_sha256(
        frozen,
        "runner-frozen official checkpoint SHA-256",
    )


def _row_provenance_identity(
    row: Mapping[str, Any],
    *,
    row_label: str,
    run_id: str,
    fingerprint: str,
    checkpoint_sha256: str,
) -> None:
    if row.get("run_id") != run_id:
        raise ValueError(f"{row_label} run ID mismatch")
    if row.get("run_manifest_fingerprint") != fingerprint:
        raise ValueError(f"{row_label} fingerprint mismatch")
    recorded_commit = row.get(
        "model_source_commit",
        row.get("source_commit"),
    )
    if recorded_commit != BR_GEN_SOURCE_COMMIT:
        raise ValueError(f"{row_label} BR-Gen commit mismatch")
    if row.get("imdlbenco_source_commit") != IMDLBENCO_SOURCE_COMMIT:
        raise ValueError(f"{row_label} IMDLBenCo commit mismatch")
    if row.get("checkpoint_sha256") != checkpoint_sha256:
        raise ValueError(f"{row_label} checkpoint SHA-256 mismatch")


def validate_provenance(
    *,
    repo_root: Path,
    nfa_vit_root: Path,
    imdlbenco_root: Path,
    run_id: str,
    inputs_path: Path,
    expected_rows: list[dict[str, Any]],
    result_rows: list[dict[str, Any]],
    manifest: dict[str, Any],
    summary: dict[str, Any],
    expected_checkpoint_sha256: str | None = None,
) -> dict[str, Any]:
    if manifest.get("schema_version") != "opensource_run_manifest_v1":
        raise ValueError("run manifest schema mismatch")
    if manifest.get("run_id") != run_id:
        raise ValueError("run manifest ID mismatch")
    fingerprint = _require_sha256(
        manifest.get("fingerprint"),
        "run manifest fingerprint",
    )
    if fingerprint != _manifest_fingerprint(manifest):
        raise ValueError("run manifest fingerprint mismatch")

    manifest_input = _require_mapping(manifest.get("input"), "manifest input")
    if manifest_input.get("inputs_sha256") != sha256_file(inputs_path):
        raise ValueError("manifest/input JSONL SHA-256 mismatch")
    inputs_value = manifest_input.get("inputs_manifest")
    if not isinstance(inputs_value, str):
        raise ValueError("manifest input has no inputs_manifest")
    if _anchored(Path(inputs_value), repo_root) != inputs_path.resolve():
        raise ValueError("manifest/input JSONL path mismatch")

    expected_contract = _selection_contract(expected_rows)
    if manifest.get("ordered_inputs") != expected_contract:
        raise ValueError("manifest ordered_inputs contract mismatch")

    model = _require_mapping(manifest.get("model"), "manifest model")
    if model.get("name") not in (MODEL_NAME, "NFA_ViT", "NFA-ViT / BR-Gen"):
        raise ValueError("manifest model identity mismatch")
    source_files = _verify_source_tree(
        model,
        root=nfa_vit_root,
        expected_commit=BR_GEN_SOURCE_COMMIT,
        label="BR-Gen",
    )
    imdl_source = _find_mapping(
        model,
        predicate=lambda item: item.get("source_commit")
        == IMDLBENCO_SOURCE_COMMIT,
    )
    if imdl_source is None:
        imdl_source = _find_mapping(
            manifest,
            predicate=lambda item: item.get("source_commit")
            == IMDLBENCO_SOURCE_COMMIT,
        )
    if imdl_source is None:
        raise ValueError("manifest has no pinned IMDLBenCo source")
    imdl_files = _verify_source_tree(
        imdl_source,
        root=imdlbenco_root,
        expected_commit=IMDLBENCO_SOURCE_COMMIT,
        label="IMDLBenCo",
    )

    checkpoint = _find_mapping(
        model,
        predicate=lambda item: (
            isinstance(item.get("sha256"), str)
            and (
                "checkpoint" in str(item.get("path", "")).lower()
                or "checkpoint" in str(item.get("original_filename", "")).lower()
            )
        ),
    )
    if checkpoint is None:
        checkpoint = _find_mapping(
            manifest,
            predicate=lambda item: (
                isinstance(item.get("sha256"), str)
                and "checkpoint" in str(item.get("path", "")).lower()
            ),
        )
    if checkpoint is None:
        raise ValueError("manifest has no official checkpoint record")
    checkpoint_sha = _require_sha256(
        checkpoint.get("sha256"),
        "checkpoint SHA-256",
    )
    frozen_checkpoint_sha = _frozen_runner_checkpoint_sha256(
        expected_checkpoint_sha256
    )
    if checkpoint_sha != frozen_checkpoint_sha:
        raise ValueError(
            "manifest checkpoint SHA-256 does not equal the runner-frozen "
            "official checkpoint SHA-256"
        )
    checkpoint_path_value = checkpoint.get("path")
    if not isinstance(checkpoint_path_value, str):
        raise ValueError("checkpoint record has no path")
    checkpoint_path = _anchored(Path(checkpoint_path_value), repo_root)
    _verify_hash(checkpoint_path, checkpoint_sha, "official checkpoint")
    if checkpoint.get("bytes") is not None and checkpoint_path.stat().st_size != int(
        checkpoint["bytes"]
    ):
        raise ValueError("official checkpoint byte size mismatch")

    adapter_files = _verify_adapter_contract(
        manifest.get("adapter_contract"),
        repo_root=repo_root,
    )
    latest = _latest_by_id(result_rows)
    expected_ids = {str(row["sample_id"]) for row in expected_rows}
    if set(latest) != expected_ids:
        raise ValueError("latest result IDs do not equal ordered input IDs")
    for line_number, row in enumerate(result_rows, start=1):
        row_id = row.get("id")
        if row_id not in expected_ids:
            raise ValueError(
                f"physical result row {line_number} has unexpected ID {row_id}"
            )
        _row_provenance_identity(
            row,
            row_label=f"physical result row {line_number} ({row_id})",
            run_id=run_id,
            fingerprint=fingerprint,
            checkpoint_sha256=checkpoint_sha,
        )
    for row_id, row in latest.items():
        if row.get("status") != "ok":
            raise ValueError(f"latest result {row_id} is not status ok")

    if summary.get("run_id") != run_id:
        raise ValueError("summary run ID mismatch")
    if summary.get("run_manifest_fingerprint") != fingerprint:
        raise ValueError("summary fingerprint mismatch")
    if summary.get("checkpoint_sha256") != checkpoint_sha:
        raise ValueError("summary checkpoint SHA-256 mismatch")
    if summary.get("model_source_commit") != BR_GEN_SOURCE_COMMIT:
        raise ValueError("summary BR-Gen commit mismatch")
    if summary.get("imdlbenco_source_commit") != IMDLBENCO_SOURCE_COMMIT:
        raise ValueError("summary IMDLBenCo commit mismatch")
    return {
        "status": "ok",
        "run_manifest_fingerprint": fingerprint,
        "inputs_sha256": sha256_file(inputs_path),
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_bytes": checkpoint_path.stat().st_size,
        "physical_result_rows_validated": len(result_rows),
        "latest_result_rows_validated": len(latest),
        "expected_unique_result_ids": len(expected_ids),
        "pinned_source_files_validated": source_files + imdl_files,
        "adapter_contract_files_validated": adapter_files,
        "checks": [
            "immutable manifest fingerprint and canonical input order",
            "pinned clean BR-Gen and IMDLBenCo source files",
            "runner-frozen official checkpoint SHA-256 and byte size",
            "every physical result identity/provenance and latest-row status",
            "required summary run/fingerprint/checkpoint identity",
            "adapter file hashes",
        ],
    }


def _artifact_value(
    row: Mapping[str, Any],
    aliases: tuple[str, ...],
) -> tuple[Any, str | None]:
    paths = row.get("artifact_paths")
    metadata = row.get("artifacts")
    for alias in aliases:
        candidates = (
            alias,
            f"{alias}_npy",
            f"{alias}_png",
        )
        for container in (paths, metadata):
            if isinstance(container, Mapping):
                for candidate in candidates:
                    if candidate in container:
                        return container[candidate], candidate
        for candidate in candidates:
            key = f"{candidate}_path"
            if key in row:
                return row[key], candidate
        key = f"{alias}_path"
        if key in row:
            return row[key], alias
    return None, None


def _metadata_value(
    row: Mapping[str, Any],
    *,
    alias_key: str | None,
    aliases: tuple[str, ...],
    suffix: str,
) -> Any:
    containers = {
        "sha256": row.get("artifact_sha256"),
        "shape": row.get("artifact_shapes"),
        "dtype": row.get("artifact_dtypes"),
    }
    container = containers.get(suffix)
    keys: list[str] = []
    if alias_key is not None:
        keys.extend((alias_key, alias_key.removesuffix("_npy").removesuffix("_png")))
    keys.extend(aliases)
    if isinstance(container, Mapping):
        for key in keys:
            if key in container:
                return container[key]
    for key in keys:
        flat = f"{key}_{suffix}"
        if flat in row:
            return row[flat]
    return None


def _resolve_artifact(
    row: Mapping[str, Any],
    *,
    canonical_name: str,
    repo_root: Path,
) -> Artifact:
    aliases = ARTIFACT_ALIASES[canonical_name]
    value, alias_key = _artifact_value(row, aliases)
    if value is None:
        raise ValueError(f"result has no {canonical_name} artifact")
    metadata: Mapping[str, Any] = {}
    if isinstance(value, Mapping):
        metadata = value
        path_value = value.get("path")
    else:
        path_value = value
    if not isinstance(path_value, str) or not path_value:
        raise ValueError(f"{canonical_name} artifact has no path")
    sha = metadata.get("sha256")
    if sha is None:
        sha = _metadata_value(
            row,
            alias_key=alias_key,
            aliases=aliases,
            suffix="sha256",
        )
    shape = metadata.get("shape")
    if shape is None:
        shape = _metadata_value(
            row,
            alias_key=alias_key,
            aliases=aliases,
            suffix="shape",
        )
    dtype = metadata.get("dtype")
    if dtype is None:
        dtype = _metadata_value(
            row,
            alias_key=alias_key,
            aliases=aliases,
            suffix="dtype",
        )
    parsed_shape: tuple[int, ...] | None = None
    if shape is not None:
        if (
            not isinstance(shape, list)
            or not shape
            or any(
                isinstance(item, bool) or not isinstance(item, int) or item <= 0
                for item in shape
            )
        ):
            raise ValueError(f"{canonical_name} artifact shape is invalid")
        parsed_shape = tuple(shape)
    return Artifact(
        path=_anchored(Path(path_value), repo_root),
        sha256=_require_sha256(sha, f"{canonical_name} artifact SHA-256"),
        shape=parsed_shape,
        dtype=None if dtype is None else str(dtype),
    )


def _load_float32_artifact(
    artifact: Artifact,
    *,
    expected_shape: tuple[int, int],
    label: str,
) -> np.ndarray:
    _verify_hash(artifact.path, artifact.sha256, label)
    try:
        array = np.load(artifact.path, allow_pickle=False)
    except Exception as exc:
        raise ValueError(f"{label} is not a safe NumPy artifact") from exc
    if array.dtype != np.float32 or array.shape != expected_shape:
        raise ValueError(
            f"{label} schema mismatch: {array.shape}/{array.dtype}, "
            f"expected {expected_shape}/float32"
        )
    if artifact.shape is not None and artifact.shape != expected_shape:
        raise ValueError(f"{label} recorded shape mismatch")
    if artifact.dtype is not None and artifact.dtype != "float32":
        raise ValueError(f"{label} recorded dtype mismatch")
    if not np.isfinite(array).all():
        raise ValueError(f"{label} contains non-finite values")
    return np.ascontiguousarray(array)


def _sigmoid_float32(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float32)
    result = np.empty_like(values)
    positive = values >= np.float32(0.0)
    result[positive] = np.float32(1.0) / (
        np.float32(1.0) + np.exp(-values[positive])
    )
    exponentials = np.exp(values[~positive])
    result[~positive] = exponentials / (
        np.float32(1.0) + exponentials
    )
    return np.ascontiguousarray(result, dtype=np.float32)


def _fma_float32(
    multiplier: np.ndarray | np.float32,
    multiplicand: np.ndarray | np.float32,
    addend: np.ndarray | np.float32,
) -> np.ndarray:
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
    source_height, source_width = source.shape
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    x1 = np.minimum(x0 + 1, source_width - 1)
    y1 = np.minimum(y0 + 1, source_height - 1)
    wx = (x - x0.astype(np.float32))[None, :]
    wy = (y - y0.astype(np.float32))[:, None]
    horizontal = _fma_float32(
        np.float32(1.0) - wx,
        source[:, x0],
        np.multiply(wx, source[:, x1], dtype=np.float32),
    )
    return np.ascontiguousarray(
        _fma_float32(
            np.float32(1.0) - wy,
            horizontal[y0, :],
            np.multiply(wy, horizontal[y1, :], dtype=np.float32),
        ),
        dtype=np.float32,
    )


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
    return _bilinear_from_coordinates(
        source,
        x=np.maximum(x, np.float32(0.0)),
        y=np.maximum(y, np.float32(0.0)),
    )


def _kernel_replay_from_manifest(
    manifest: Mapping[str, Any],
) -> KernelReplay:
    """Resolve a replay device only when it matches immutable run metadata."""

    runtime = _require_mapping(
        manifest.get("runtime_contract"),
        "manifest runtime_contract",
    )
    packages = _require_mapping(
        runtime.get("packages"),
        "manifest runtime packages",
    )
    recorded_torch = packages.get("torch")
    if not isinstance(recorded_torch, str) or not recorded_torch:
        raise ValueError("manifest has no recorded torch runtime version")
    accelerator = _require_mapping(
        runtime.get("accelerator"),
        "manifest runtime accelerator",
    )
    requested = accelerator.get("requested_device")
    if not isinstance(requested, str) or not requested:
        raise ValueError("manifest has no recorded inference device")
    try:
        import torch
    except (ImportError, RuntimeError) as exc:
        raise RuntimeError(
            "kernel-exact protocol decision audit requires the recorded "
            "PyTorch runtime"
        ) from exc
    current_torch = str(torch.__version__)
    if current_torch != recorded_torch:
        raise ValueError(
            "recorded torch runtime cannot be reproduced: "
            f"{recorded_torch!r} != {current_torch!r}"
        )
    try:
        device = torch.device(requested)
    except (TypeError, RuntimeError, ValueError) as exc:
        raise ValueError(f"recorded inference device is invalid: {requested}") from exc
    if device.type == "cpu":
        if device.index is not None:
            raise ValueError("recorded CPU device has an unsupported index")
    elif device.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError(
                "recorded CUDA inference device is unavailable; exact "
                "threshold replay is fail-closed"
            )
        index = torch.cuda.current_device() if device.index is None else device.index
        if index < 0 or index >= torch.cuda.device_count():
            raise ValueError(
                "recorded CUDA inference device index is unavailable; exact "
                "threshold replay is fail-closed"
            )
        device = torch.device("cuda", index)
        recorded_name = accelerator.get("gpu_name")
        recorded_capability = accelerator.get("gpu_capability")
        observed_name = torch.cuda.get_device_name(device)
        observed_capability = list(torch.cuda.get_device_capability(device))
        if (
            recorded_name != observed_name
            or recorded_capability != observed_capability
        ):
            raise ValueError(
                "recorded CUDA hardware cannot be reproduced; exact "
                "threshold replay is fail-closed"
            )
        if accelerator.get("torch_cuda") != torch.version.cuda:
            raise ValueError(
                "recorded CUDA runtime cannot be reproduced; exact "
                "threshold replay is fail-closed"
            )
    else:
        raise ValueError(
            f"unsupported recorded inference device for exact replay: {requested}"
        )
    return KernelReplay(
        torch=torch,
        device=device,
        requested_device=requested,
        torch_version=current_torch,
    )


def _torch_sigmoid(
    values: np.ndarray,
    *,
    replay: KernelReplay,
) -> np.ndarray:
    torch = replay.torch
    source = torch.from_numpy(
        np.ascontiguousarray(values, dtype=np.float32)
    ).to(device=replay.device, dtype=torch.float32)
    with torch.inference_mode():
        result = torch.sigmoid(source)
    return np.ascontiguousarray(result.cpu().numpy(), dtype=np.float32)


def _torch_bilinear_align_corners_false(
    values: np.ndarray,
    *,
    width: int,
    height: int,
    replay: KernelReplay,
) -> np.ndarray:
    torch = replay.torch
    source = torch.from_numpy(
        np.ascontiguousarray(values, dtype=np.float32)
    ).to(device=replay.device, dtype=torch.float32)
    source = source.reshape(1, 1, *source.shape)
    with torch.inference_mode():
        result = torch.nn.functional.interpolate(
            source,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )
    return np.ascontiguousarray(
        result.reshape(height, width).cpu().numpy(),
        dtype=np.float32,
    )


def _torch_end_to_end_native(
    raw: np.ndarray,
    *,
    width: int,
    height: int,
    replay: KernelReplay,
) -> np.ndarray:
    torch = replay.torch
    source = torch.from_numpy(
        np.ascontiguousarray(raw, dtype=np.float32)
    ).to(device=replay.device, dtype=torch.float32)
    source = source.reshape(1, 1, *source.shape)
    with torch.inference_mode():
        resized = torch.nn.functional.interpolate(
            source,
            size=(MODEL_SIZE, MODEL_SIZE),
            mode="bilinear",
            align_corners=False,
        )
        probability = torch.sigmoid(resized)
        native = torch.nn.functional.interpolate(
            probability,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )
    return np.ascontiguousarray(
        native.reshape(height, width).cpu().numpy(),
        dtype=np.float32,
    )


def _require_threshold_decisions_equal(
    stored: np.ndarray | float,
    replayed: np.ndarray | float,
    *,
    threshold: float,
    label: str,
) -> None:
    stored_decision = np.asarray(stored) > threshold
    replayed_decision = np.asarray(replayed) > threshold
    if stored_decision.shape != replayed_decision.shape:
        raise ValueError(f"{label} threshold decision shape mismatch")
    disagreements = int(np.count_nonzero(stored_decision != replayed_decision))
    if disagreements:
        raise ValueError(
            f"{label} threshold decision mismatch at {disagreements} value(s)"
        )


def _model_target(target: np.ndarray) -> np.ndarray:
    source = np.asarray(target, dtype=np.uint8)
    if source.ndim != 2 or source.size == 0:
        raise ValueError("target must be a non-empty 2D array")
    if source.shape == (MODEL_SIZE, MODEL_SIZE):
        return np.ascontiguousarray(source > 0)
    cv2 = _cv2_module()
    resized = cv2.resize(
        source,
        (MODEL_SIZE, MODEL_SIZE),
        interpolation=cv2.INTER_NEAREST,
    )
    return np.ascontiguousarray(resized > 0)


def _official_preprocess(image_path: Path) -> np.ndarray:
    cv2 = _cv2_module()
    with image_path.open("rb") as handle:
        with Image.open(handle) as opened:
            rgb = np.asarray(opened.convert("RGB"), dtype=np.uint8)
    resized = cv2.resize(
        rgb,
        (MODEL_SIZE, MODEL_SIZE),
        interpolation=cv2.INTER_LINEAR,
    )
    # Albumentations 1.3.0 dispatches three-channel input to normalize_cv2:
    # float32 cast, cv2.subtract(mean * 255), then
    # cv2.multiply(reciprocal(std * 255)).  Replaying that operation order is
    # necessary for a bit-exact input-tensor hash.
    values = np.ascontiguousarray(resized.astype("float32"))
    mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
    mean *= np.float32(255.0)
    standard_deviation = np.asarray(
        [0.229, 0.224, 0.225],
        dtype=np.float32,
    )
    standard_deviation *= np.float32(255.0)
    denominator = np.reciprocal(
        standard_deviation,
        dtype=np.float32,
    )
    mean_cv = np.asarray(mean.tolist() + [0.0], dtype=np.float64)
    denominator_cv = np.asarray(
        denominator.tolist() + [1.0],
        dtype=np.float64,
    )
    cv2.subtract(values, mean_cv, values)
    cv2.multiply(values, denominator_cv, values)
    return np.ascontiguousarray(values.transpose(2, 0, 1), dtype=np.float32)


def _preprocess_audit(
    row: Mapping[str, Any],
    *,
    image_path: Path,
    width: int,
    height: int,
) -> None:
    evidence = _require_mapping(row.get("preprocess"), "row preprocess")
    expected_pairs = {
        "channel_order": "RGB",
        "geometry": "direct_stretch_without_aspect_ratio_preservation",
        "tensor_dtype": "float32",
        "tensor_shape": [3, MODEL_SIZE, MODEL_SIZE],
    }
    for key, expected in expected_pairs.items():
        if evidence.get(key) != expected:
            raise ValueError(f"preprocess {key} mismatch")
    decoder = evidence.get("decoder")
    if decoder not in (
        "Pillow.Image.open.convert_RGB",
        "PIL.Image.open.convert_RGB",
    ):
        raise ValueError("preprocess decoder mismatch")
    native_size = evidence.get(
        "native_size_wh",
        evidence.get("native_size"),
    )
    if native_size != [width, height]:
        raise ValueError("preprocess native geometry mismatch")
    model_size = evidence.get(
        "model_size_wh",
        evidence.get("model_size"),
    )
    if model_size != [MODEL_SIZE, MODEL_SIZE]:
        raise ValueError("preprocess model geometry mismatch")
    if evidence.get("input_reencode") not in (None, False):
        raise ValueError("preprocess unexpectedly re-encodes input")
    if evidence.get("exif_transpose") not in (None, False):
        raise ValueError("preprocess unexpectedly applies EXIF transpose")
    if evidence.get("icc_conversion") not in (None, False):
        raise ValueError("preprocess unexpectedly applies ICC conversion")
    interpolation = str(
        evidence.get(
            "resize_interpolation",
            evidence.get("resize"),
        )
    )
    if "INTER_LINEAR" not in interpolation:
        raise ValueError("preprocess interpolation mismatch")
    normalization = _require_mapping(
        evidence.get("normalization"),
        "preprocess normalization",
    )
    if (
        normalization.get("mean") != [0.485, 0.456, 0.406]
        or normalization.get("std") != [0.229, 0.224, 0.225]
        or float(normalization.get("max_pixel_value", 255.0)) != 255.0
    ):
        raise ValueError("preprocess normalization mismatch")
    tensor_hash = _require_sha256(
        evidence.get("tensor_sha256"),
        "preprocess tensor SHA-256",
    )
    if _array_sha256(_official_preprocess(image_path)) != tensor_hash:
        raise ValueError("preprocess tensor SHA-256 mismatch")


def _load_target(
    expected: Mapping[str, Any],
    *,
    repo_root: Path,
    width: int,
    height: int,
) -> np.ndarray:
    kind = expected.get("kind")
    if kind == "real":
        return np.zeros((height, width), dtype=bool)
    path_value = expected.get("gt_mask_path")
    if not isinstance(path_value, str):
        raise ValueError("forged input has no GT mask path")
    path = _anchored(Path(path_value), repo_root)
    _verify_hash(path, expected.get("gt_mask_sha256"), "forged GT mask")
    with Image.open(path) as opened:
        target = np.asarray(opened.convert("L"), dtype=np.uint8)
    if target.shape != (height, width):
        raise ValueError("native GT geometry mismatch")
    return np.ascontiguousarray(target > 0)


def _compare_numeric(
    actual: Any,
    expected: Any,
    *,
    label: str,
) -> None:
    if expected is None:
        if actual is not None:
            raise ValueError(f"{label} must be null")
        return
    try:
        observed = float(actual)
        reference = float(expected)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} is not numeric") from exc
    if not math.isclose(observed, reference, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"{label} mismatch: {observed} != {reference}")


def _validate_recorded_metrics(
    recorded: Any,
    expected: Mapping[str, Any],
    *,
    label: str,
) -> None:
    value = _require_mapping(recorded, label)
    for key, expected_value in expected.items():
        if key in (
            "threshold_operator",
            "probability_dtype",
        ):
            if value.get(key) != expected_value:
                raise ValueError(f"{label} {key} mismatch")
        elif isinstance(expected_value, (int, np.integer)) and key not in (
            "threshold",
        ):
            if value.get(key) != int(expected_value):
                raise ValueError(f"{label} {key} mismatch")
        else:
            _compare_numeric(
                value.get(key),
                expected_value,
                label=f"{label} {key}",
            )


def _classification_logit(row: Mapping[str, Any]) -> float:
    classification = row.get("classification")
    if isinstance(classification, Mapping):
        value = classification.get(
            "raw_logit",
            classification.get("logit"),
        )
    else:
        value = row.get(
            "classification_logit",
            row.get("classification_raw_logit"),
        )
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("row has no finite classification logit") from exc
    if not math.isfinite(result):
        raise ValueError("classification logit is not finite")
    return result


def _classification_audit(
    row: Mapping[str, Any],
    *,
    sample_id: str,
    replay: KernelReplay,
) -> dict[str, float]:
    logit = _classification_logit(row)
    numpy_score = float(
        _sigmoid_float32(np.asarray([logit], dtype=np.float32))[0]
    )
    kernel_score = float(
        _torch_sigmoid(
            np.asarray([logit], dtype=np.float32),
            replay=replay,
        )[0]
    )
    try:
        stored_score = float(row["score"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{sample_id} has no finite classification score") from exc
    if not math.isfinite(stored_score) or not 0.0 <= stored_score <= 1.0:
        raise ValueError(f"{sample_id} classification score is invalid")
    if not math.isclose(
        stored_score,
        numpy_score,
        rel_tol=0.0,
        abs_tol=MODEL_PROBABILITY_ABSOLUTE_TOLERANCE,
    ):
        raise ValueError(f"{sample_id} classification NumPy sigmoid mismatch")
    if not math.isclose(
        stored_score,
        kernel_score,
        rel_tol=0.0,
        abs_tol=MODEL_PROBABILITY_ABSOLUTE_TOLERANCE,
    ):
        raise ValueError(f"{sample_id} classification kernel sigmoid mismatch")
    _require_threshold_decisions_equal(
        stored_score,
        kernel_score,
        threshold=FIXED_CLASSIFICATION_THRESHOLD,
        label=f"{sample_id} classification",
    )
    stored_decision = stored_score > FIXED_CLASSIFICATION_THRESHOLD

    classification = _require_mapping(
        row.get("classification"),
        f"{sample_id} classification",
    )
    nested_logit = classification.get(
        "raw_logit",
        classification.get("logit"),
    )
    _compare_numeric(
        nested_logit,
        logit,
        label=f"{sample_id} nested classification raw logit",
    )
    nested_scores = [
        classification[key]
        for key in ("probability", "score")
        if key in classification
    ]
    if not nested_scores:
        raise ValueError(
            f"{sample_id} nested classification has no probability/score"
        )
    for nested_score in nested_scores:
        _compare_numeric(
            nested_score,
            stored_score,
            label=f"{sample_id} nested classification score",
        )
    nested_decision = classification.get("decision")
    if not isinstance(nested_decision, (bool, np.bool_)):
        raise ValueError(
            f"{sample_id} nested classification decision is not boolean"
        )
    if bool(nested_decision) != stored_decision:
        raise ValueError(
            f"{sample_id} nested classification decision mismatch"
        )
    _compare_numeric(
        classification.get("threshold"),
        FIXED_CLASSIFICATION_THRESHOLD,
        label=f"{sample_id} nested classification threshold",
    )
    if classification.get("threshold_operator") != THRESHOLD_OPERATOR:
        raise ValueError(
            f"{sample_id} nested classification threshold operator mismatch"
        )

    top_level_logits = [
        row[key]
        for key in ("classification_raw_logit", "classification_logit")
        if key in row
    ]
    if not top_level_logits:
        raise ValueError(f"{sample_id} has no top-level classification logit")
    for top_level_logit in top_level_logits:
        _compare_numeric(
            top_level_logit,
            logit,
            label=f"{sample_id} top-level classification logit",
        )
    top_level_scores = [
        row[key]
        for key in ("classification_score", "classification_probability")
        if key in row
    ]
    if not top_level_scores:
        raise ValueError(f"{sample_id} has no top-level classification score")
    for top_level_score in top_level_scores:
        _compare_numeric(
            top_level_score,
            stored_score,
            label=f"{sample_id} top-level classification score",
        )
    top_level_decisions = [
        row[key]
        for key in (
            "classification_decision",
            "classification_decision_strict_gt_0_5",
        )
        if key in row
    ]
    if not top_level_decisions:
        raise ValueError(f"{sample_id} has no top-level classification decision")
    for top_level_decision in top_level_decisions:
        if not isinstance(top_level_decision, (bool, np.bool_)):
            raise ValueError(
                f"{sample_id} top-level classification decision is not boolean"
            )
        if bool(top_level_decision) != stored_decision:
            raise ValueError(
                f"{sample_id} top-level classification decision mismatch"
            )
    if "classification_threshold" in row:
        _compare_numeric(
            row["classification_threshold"],
            FIXED_CLASSIFICATION_THRESHOLD,
            label=f"{sample_id} top-level classification threshold",
        )
    if (
        "classification_threshold_operator" in row
        and row["classification_threshold_operator"] != THRESHOLD_OPERATOR
    ):
        raise ValueError(
            f"{sample_id} top-level classification threshold operator mismatch"
        )
    if "decision" in row:
        expected = "forged" if stored_decision else "authentic"
        decision = row["decision"]
        if isinstance(decision, (bool, np.bool_)):
            valid = bool(decision) == stored_decision
        else:
            valid = decision == expected
        if not valid:
            raise ValueError(f"{sample_id} top-level decision mismatch")
    return {
        "numpy_sigmoid_absolute_error": abs(stored_score - numpy_score),
        "kernel_sigmoid_absolute_error": abs(stored_score - kernel_score),
    }


def audit_artifacts(
    *,
    repo_root: Path,
    expected_rows: list[dict[str, Any]],
    result_rows: list[dict[str, Any]],
    manifest: Mapping[str, Any],
) -> tuple[dict[str, Any], list[Pair]]:
    kernel_replay = _kernel_replay_from_manifest(manifest)
    latest = _latest_by_id(result_rows)
    expected_by_id = {str(row["sample_id"]): row for row in expected_rows}
    if set(latest) != set(expected_by_id):
        raise ValueError("artifact audit ID coverage mismatch")

    maxima = {
        "resized_logits_replay": 0.0,
        "model_probability_replay": 0.0,
        "native_probability_replay": 0.0,
        "numpy_end_to_end_native_replay": 0.0,
        "kernel_end_to_end_native_replay": 0.0,
        "classification_numpy_sigmoid_replay": 0.0,
        "classification_kernel_sigmoid_replay": 0.0,
    }
    mask_disagreements = 0
    checked_files = 0
    by_task: dict[str, dict[str, tuple[dict[str, Any], dict[str, Any]]]] = (
        defaultdict(dict)
    )
    for sample_id, expected in expected_by_id.items():
        row = latest[sample_id]
        if row.get("status") != "ok":
            raise ValueError(f"latest row {sample_id} is not status ok")
        for key in ("task_id", "kind", "label", "domain"):
            if row.get(key) != expected.get(key):
                raise ValueError(f"row {sample_id} {key} mismatch")
        image_value = expected.get("canonical_path")
        if not isinstance(image_value, str):
            raise ValueError(f"input {sample_id} has no canonical path")
        image_path = _anchored(Path(image_value), repo_root)
        _verify_hash(
            image_path,
            expected.get("canonical_sha256"),
            f"canonical image {sample_id}",
        )
        checked_files += 1
        with Image.open(image_path) as opened:
            width, height = opened.size
        if row.get("image_size") not in (
            [width, height],
            {"width": width, "height": height},
        ):
            raise ValueError(f"row {sample_id} image geometry mismatch")
        _preprocess_audit(
            row,
            image_path=image_path,
            width=width,
            height=height,
        )

        target_native = _load_target(
            expected,
            repo_root=repo_root,
            width=width,
            height=height,
        )
        if expected.get("kind") == "forged":
            checked_files += 1
        target_model = _model_target(target_native)

        raw_artifact = _resolve_artifact(
            row,
            canonical_name="raw_logits_128",
            repo_root=repo_root,
        )
        resized_artifact = _resolve_artifact(
            row,
            canonical_name="resized_logits_512",
            repo_root=repo_root,
        )
        probability_artifact = _resolve_artifact(
            row,
            canonical_name="probability_512",
            repo_root=repo_root,
        )
        native_artifact = _resolve_artifact(
            row,
            canonical_name="probability_native",
            repo_root=repo_root,
        )
        mask_artifact = _resolve_artifact(
            row,
            canonical_name="mask_native",
            repo_root=repo_root,
        )
        raw = _load_float32_artifact(
            raw_artifact,
            expected_shape=(RAW_LOGIT_SIZE, RAW_LOGIT_SIZE),
            label=f"{sample_id} raw decoder logits",
        )
        resized = _load_float32_artifact(
            resized_artifact,
            expected_shape=(MODEL_SIZE, MODEL_SIZE),
            label=f"{sample_id} resized decoder logits",
        )
        probability = _load_float32_artifact(
            probability_artifact,
            expected_shape=(MODEL_SIZE, MODEL_SIZE),
            label=f"{sample_id} model probability",
        )
        native = _load_float32_artifact(
            native_artifact,
            expected_shape=(height, width),
            label=f"{sample_id} native probability",
        )
        checked_files += 4

        replay_resized = _bilinear_align_corners_false(
            raw,
            width=MODEL_SIZE,
            height=MODEL_SIZE,
        )
        replay_probability = _sigmoid_float32(resized)
        replay_native = _bilinear_align_corners_false(
            probability,
            width=width,
            height=height,
        )
        replay_probability_from_raw = _sigmoid_float32(replay_resized)
        replay_native_from_raw = _bilinear_align_corners_false(
            replay_probability_from_raw,
            width=width,
            height=height,
        )
        kernel_resized = _torch_bilinear_align_corners_false(
            raw,
            width=MODEL_SIZE,
            height=MODEL_SIZE,
            replay=kernel_replay,
        )
        kernel_probability = _torch_sigmoid(
            resized,
            replay=kernel_replay,
        )
        kernel_native = _torch_bilinear_align_corners_false(
            probability,
            width=width,
            height=height,
            replay=kernel_replay,
        )
        kernel_native_from_raw = _torch_end_to_end_native(
            raw,
            width=width,
            height=height,
            replay=kernel_replay,
        )
        errors = {
            "resized_logits_replay": float(
                np.max(np.abs(resized - replay_resized))
            ),
            "model_probability_replay": float(
                np.max(np.abs(probability - replay_probability))
            ),
            "native_probability_replay": float(
                np.max(np.abs(native - replay_native))
            ),
            "numpy_end_to_end_native_replay": float(
                np.max(np.abs(native - replay_native_from_raw))
            ),
            "kernel_end_to_end_native_replay": float(
                np.max(np.abs(native - kernel_native_from_raw))
            ),
        }
        for key, value in errors.items():
            maxima[key] = max(maxima[key], value)
        if not np.allclose(
            resized,
            replay_resized,
            rtol=TRANSFORM_RELATIVE_TOLERANCE,
            atol=RESIZED_LOGITS_ABSOLUTE_TOLERANCE,
        ):
            raise ValueError(f"{sample_id} resized-logit replay mismatch")
        if not np.allclose(
            probability,
            replay_probability,
            rtol=TRANSFORM_RELATIVE_TOLERANCE,
            atol=MODEL_PROBABILITY_ABSOLUTE_TOLERANCE,
        ):
            raise ValueError(f"{sample_id} model-probability replay mismatch")
        if not np.allclose(
            native,
            replay_native,
            rtol=TRANSFORM_RELATIVE_TOLERANCE,
            atol=NATIVE_PROBABILITY_ABSOLUTE_TOLERANCE,
        ):
            raise ValueError(f"{sample_id} native-probability replay mismatch")
        if not np.allclose(
            native,
            replay_native_from_raw,
            rtol=TRANSFORM_RELATIVE_TOLERANCE,
            atol=END_TO_END_NATIVE_ABSOLUTE_TOLERANCE,
        ):
            raise ValueError(
                f"{sample_id} raw-to-native end-to-end replay mismatch"
            )
        _require_threshold_decisions_equal(
            resized,
            kernel_resized,
            threshold=0.0,
            label=f"{sample_id} raw-to-resized logits",
        )
        _require_threshold_decisions_equal(
            probability,
            kernel_probability,
            threshold=FIXED_MASK_THRESHOLD,
            label=f"{sample_id} resized-logits-to-probability",
        )
        _require_threshold_decisions_equal(
            native,
            kernel_native,
            threshold=FIXED_MASK_THRESHOLD,
            label=f"{sample_id} probability-to-native",
        )
        _require_threshold_decisions_equal(
            native,
            kernel_native_from_raw,
            threshold=FIXED_MASK_THRESHOLD,
            label=f"{sample_id} raw-to-native end-to-end",
        )
        if (
            float(probability.min()) < 0.0
            or float(probability.max()) > 1.0
            or float(native.min()) < 0.0
            or float(native.max()) > 1.0
        ):
            raise ValueError(f"{sample_id} probability falls outside [0,1]")

        _verify_hash(
            mask_artifact.path,
            mask_artifact.sha256,
            f"{sample_id} native mask",
        )
        with Image.open(mask_artifact.path) as opened:
            if opened.format != "PNG":
                raise ValueError(f"{sample_id} mask is not PNG")
            mask = np.asarray(opened)
        if mask.dtype != np.uint8 or mask.shape != (height, width):
            raise ValueError(f"{sample_id} native mask schema mismatch")
        if mask_artifact.shape is not None and mask_artifact.shape != (
            height,
            width,
        ):
            raise ValueError(f"{sample_id} recorded mask shape mismatch")
        if mask_artifact.dtype is not None and mask_artifact.dtype != "uint8":
            raise ValueError(f"{sample_id} recorded mask dtype mismatch")
        if not np.isin(mask, (0, 255)).all():
            raise ValueError(f"{sample_id} native mask is not binary")
        expected_mask = native > FIXED_MASK_THRESHOLD
        disagreements = int(np.count_nonzero((mask > 0) != expected_mask))
        mask_disagreements += disagreements
        if disagreements:
            raise ValueError(f"{sample_id} native threshold mask mismatch")
        checked_files += 1

        classification_errors = _classification_audit(
            row,
            sample_id=sample_id,
            replay=kernel_replay,
        )
        maxima["classification_numpy_sigmoid_replay"] = max(
            maxima["classification_numpy_sigmoid_replay"],
            classification_errors["numpy_sigmoid_absolute_error"],
        )
        maxima["classification_kernel_sigmoid_replay"] = max(
            maxima["classification_kernel_sigmoid_replay"],
            classification_errors["kernel_sigmoid_absolute_error"],
        )

        kind = str(expected["kind"])
        expected_model_metrics = binary_pixel_metrics_strict(
            probability,
            target_model,
            include_ap=kind == "forged",
        )
        expected_native_metrics = binary_pixel_metrics_strict(
            native,
            target_native,
            include_ap=kind == "forged",
        )
        localization = _require_mapping(
            row.get("localization"),
            f"{sample_id} localization",
        )
        _validate_recorded_metrics(
            localization.get("model_512"),
            expected_model_metrics,
            label=f"{sample_id} model_512 metrics",
        )
        _validate_recorded_metrics(
            localization.get("native"),
            expected_native_metrics,
            label=f"{sample_id} native metrics",
        )
        task_id = str(expected["task_id"])
        if kind in by_task[task_id]:
            raise ValueError(f"duplicate {kind} result in task {task_id}")
        by_task[task_id][kind] = (row, expected)

    pairs: list[Pair] = []
    for task_id, values in sorted(by_task.items()):
        if set(values) != {"real", "forged"}:
            raise ValueError(f"task {task_id} is not a complete pair")
        pairs.append(
            Pair(
                task_id=task_id,
                real=values["real"][0],
                forged=values["forged"][0],
                expected_real=values["real"][1],
                expected_forged=values["forged"][1],
            )
        )
    return (
        {
            "status": "ok",
            "checked_files": checked_files,
            "pairs": len(pairs),
            "result_images": len(latest),
            "numeric_tolerances": {
                "resized_logits_absolute": RESIZED_LOGITS_ABSOLUTE_TOLERANCE,
                "model_probability_absolute": (
                    MODEL_PROBABILITY_ABSOLUTE_TOLERANCE
                ),
                "native_restore_absolute": (
                    NATIVE_PROBABILITY_ABSOLUTE_TOLERANCE
                ),
                "raw_to_native_end_to_end_absolute": (
                    END_TO_END_NATIVE_ABSOLUTE_TOLERANCE
                ),
                "relative": TRANSFORM_RELATIVE_TOLERANCE,
            },
            "kernel_exact_decision_replay": {
                "status": "recorded_runtime_and_device_reproduced",
                "requested_device": kernel_replay.requested_device,
                "resolved_device": str(kernel_replay.device),
                "torch_version": kernel_replay.torch_version,
                "threshold_policy": (
                    "stored and kernel-replayed strict decisions must be "
                    "identical outside numeric allclose"
                ),
            },
            "observed_maximum_absolute_error": {
                **maxima,
                "threshold_mask_disagreements": mask_disagreements,
            },
            "checks": [
                "canonical image and GT hashes",
                "official Pillow RGB, OpenCV linear 512 stretch, and ImageNet tensor hash",
                "classifier score independently replays float32 sigmoid(raw logit)",
                "nested/top-level T1 decisions equal kernel-replayed strict score > 0.5",
                "128 logits replay bilinear to 512 and independent sigmoid",
                "each dense transform preserves kernel-replayed threshold decisions outside allclose",
                "native probability restores p512 directly",
                "raw logits replay end-to-end through native probability and strict decision",
                "native PNG bit-exactly equals strict p > 0.5",
                "model/native metrics independently recomputed with nearest GT",
            ],
        },
        pairs,
    )


def _pair_payload(pair: Pair) -> dict[str, dict[str, Any]]:
    return {"real": pair.real, "forged": pair.forged}


def _quintiles(pairs: list[Pair]) -> list[tuple[str, list[Pair]]]:
    ordered = sorted(
        pairs,
        key=lambda pair: (
            pair.forged["localization"]["native"]["target_positive_pixels"]
            / pair.forged["localization"]["native"]["pixels"],
            pair.task_id,
        ),
    )
    if len(ordered) < 5:
        return []
    boundaries = np.linspace(0, len(ordered), 6, dtype=int)
    names = ("q1_smallest", "q2", "q3", "q4", "q5_largest")
    return [
        (names[index], ordered[boundaries[index] : boundaries[index + 1]])
        for index in range(5)
    ]


def _slice_summary(
    pairs: list[Pair],
    *,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    payload = [_pair_payload(pair) for pair in pairs]
    return {
        space: summarize_nfa_vit_pair_slice(
            payload,
            iterations=iterations,
            seed=seed,
            localization_space=space,
        )
        for space in LOCALIZATION_SPACES
    }


def _box_hit(
    pairs: list[Pair],
    *,
    repo_root: Path,
) -> dict[str, Any]:
    hits = 0
    strong = 0
    box_ious: list[float] = []
    coverages: list[float] = []
    for pair in pairs:
        box = pair.expected_forged.get("edit_region_xyxy")
        if not (
            isinstance(box, list)
            and len(box) == 4
            and all(isinstance(value, int) for value in box)
        ):
            raise ValueError(f"task {pair.task_id} has invalid edit box")
        x0, y0, x1, y1 = box
        artifact = _resolve_artifact(
            pair.forged,
            canonical_name="mask_native",
            repo_root=repo_root,
        )
        with Image.open(artifact.path) as opened:
            prediction = np.asarray(opened) > 0
        height, width = prediction.shape
        if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
            raise ValueError(f"task {pair.task_id} edit box is out of bounds")
        box_mask = np.zeros_like(prediction)
        box_mask[y0:y1, x0:x1] = True
        intersection = int(np.count_nonzero(prediction & box_mask))
        union = int(np.count_nonzero(prediction | box_mask))
        box_pixels = (x1 - x0) * (y1 - y0)
        iou = intersection / union if union else 0.0
        coverage = intersection / box_pixels
        hits += int(intersection > 0)
        strong += int(iou > 0.3)
        box_ious.append(iou)
        coverages.append(coverage)
    return {
        "status": "posthoc_descriptive_diagnostic_only",
        "eligible_for_primary_metrics": False,
        "uses_test_set_annotations": True,
        "mask_threshold": FIXED_MASK_THRESHOLD,
        "threshold_operator": THRESHOLD_OPERATOR,
        "any_overlap": {
            "hits": hits,
            "images": len(pairs),
            "rate": hits / len(pairs) if pairs else None,
        },
        "iou_greater_than_0_3": {
            "hits": strong,
            "images": len(pairs),
            "rate": strong / len(pairs) if pairs else None,
        },
        "box_iou": _descriptive(box_ious),
        "box_pixel_coverage": _descriptive(coverages),
    }


def _descriptive(values: Iterable[float]) -> dict[str, Any]:
    array = np.asarray(list(values), dtype=np.float64)
    if not array.size:
        return {"count": 0}
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def _compare_summary(
    recorded: Mapping[str, Any],
    recomputed: Mapping[str, Any],
) -> None:
    keys = (
        "schema_version",
        "task_scope",
        "coverage",
        "paired_coverage",
        "score_by_kind",
        "paired_score_delta",
        "paired_ranking_accuracy",
        "detection",
        "native_t1_saturation_diagnostic",
        "raw_logit_secondary_diagnostic",
        "localization_forged",
        "localization_real",
        "pair_bootstrap",
        "joint_diagnostics",
        "latency_ms",
        "peak_cuda_memory_bytes",
    )

    def compare(actual: Any, expected: Any, path: str) -> None:
        if isinstance(expected, Mapping):
            if not isinstance(actual, Mapping):
                raise ValueError(f"summary {path} is not an object")
            for key, child in expected.items():
                if key not in actual:
                    raise ValueError(f"summary {path}.{key} is missing")
                compare(actual[key], child, f"{path}.{key}")
            return
        if isinstance(expected, list):
            if not isinstance(actual, list) or len(actual) != len(expected):
                raise ValueError(f"summary {path} list mismatch")
            for index, child in enumerate(expected):
                compare(actual[index], child, f"{path}[{index}]")
            return
        if isinstance(expected, float):
            _compare_numeric(actual, expected, label=f"summary {path}")
            return
        if actual != expected:
            raise ValueError(f"summary {path} mismatch")

    for key in keys:
        if key not in recorded:
            raise ValueError(f"recorded summary has no {key}")
        compare(recorded[key], recomputed[key], key)


def _recorded_bootstrap_contract(
    summary: Mapping[str, Any],
) -> tuple[int, int]:
    """Return the inference-run bootstrap contract recorded in its summary.

    Domain and quintile intervals intentionally have a separate post-hoc CLI
    seed.  Using that seed to replay the runner summary would create a false
    mismatch whenever the two seeds differ.
    """

    pair_bootstrap = _require_mapping(
        summary.get("pair_bootstrap"),
        "recorded summary pair_bootstrap",
    )
    samples = pair_bootstrap.get("bootstrap_samples")
    seed = pair_bootstrap.get("seed")
    if (
        isinstance(samples, (bool, np.bool_))
        or not isinstance(samples, (int, np.integer))
        or int(samples) <= 0
    ):
        raise ValueError(
            "recorded summary bootstrap_samples is not a positive integer"
        )
    if isinstance(seed, (bool, np.bool_)) or not isinstance(
        seed,
        (int, np.integer),
    ):
        raise ValueError("recorded summary bootstrap seed is not an integer")
    return int(samples), int(seed)


def audit_prefix_reproducibility(
    *,
    repo_root: Path,
    full_expected: list[dict[str, Any]],
    full_rows: list[dict[str, Any]],
    prefix_run_id: str | None,
    prefix_results_dir: Path,
) -> dict[str, Any] | None:
    if prefix_run_id is None:
        return None
    results_path = prefix_results_dir / f"{prefix_run_id}.jsonl"
    manifest_path = prefix_results_dir / f"{prefix_run_id}.run_manifest.json"
    prefix_rows = read_jsonl(results_path)
    prefix_manifest = _require_mapping(
        json.loads(manifest_path.read_text(encoding="utf-8")),
        "prefix manifest",
    )
    if prefix_manifest.get("run_id") != prefix_run_id:
        raise ValueError("prefix manifest run ID mismatch")
    prefix_fingerprint = _require_sha256(
        prefix_manifest.get("fingerprint"),
        "prefix manifest fingerprint",
    )
    if prefix_fingerprint != _manifest_fingerprint(prefix_manifest):
        raise ValueError("prefix manifest fingerprint mismatch")
    ordered = prefix_manifest.get("ordered_inputs")
    if not isinstance(ordered, list) or not ordered:
        raise ValueError("prefix ordered_inputs is invalid")
    prefix_ids = [str(item["sample_id"]) for item in ordered]
    full_ids = [str(item["sample_id"]) for item in full_expected]
    if full_ids[: len(prefix_ids)] != prefix_ids:
        raise ValueError("reference run is not an exact ordered prefix")
    expected_prefix_contract = _selection_contract(
        full_expected[: len(prefix_ids)]
    )
    if ordered != expected_prefix_contract:
        raise ValueError("prefix ordered_inputs provenance contract mismatch")
    prefix_model = _require_mapping(
        prefix_manifest.get("model"),
        "prefix manifest model",
    )
    prefix_source_commit = prefix_model.get(
        "source_commit",
        prefix_model.get("commit"),
    )
    if prefix_source_commit != BR_GEN_SOURCE_COMMIT:
        raise ValueError("prefix manifest BR-Gen commit mismatch")
    prefix_source_root = prefix_model.get("source_root")
    if not isinstance(prefix_source_root, str) or not prefix_source_root:
        raise ValueError("prefix manifest has no BR-Gen source root")
    _verify_source_tree(
        prefix_model,
        root=Path(prefix_source_root).resolve(),
        expected_commit=BR_GEN_SOURCE_COMMIT,
        label="prefix BR-Gen",
    )
    prefix_imdl_source = _find_mapping(
        prefix_model,
        predicate=lambda item: item.get("source_commit")
        == IMDLBENCO_SOURCE_COMMIT,
    )
    if prefix_imdl_source is None:
        raise ValueError("prefix manifest has no pinned IMDLBenCo source")
    prefix_imdl_root = prefix_imdl_source.get("source_root")
    if not isinstance(prefix_imdl_root, str) or not prefix_imdl_root:
        raise ValueError("prefix manifest has no IMDLBenCo source root")
    _verify_source_tree(
        prefix_imdl_source,
        root=Path(prefix_imdl_root).resolve(),
        expected_commit=IMDLBENCO_SOURCE_COMMIT,
        label="prefix IMDLBenCo",
    )
    prefix_checkpoint = _find_mapping(
        prefix_model,
        predicate=lambda item: (
            isinstance(item.get("sha256"), str)
            and (
                "checkpoint" in str(item.get("path", "")).lower()
                or "checkpoint"
                in str(item.get("original_filename", "")).lower()
            )
        ),
    )
    if prefix_checkpoint is None:
        raise ValueError("prefix manifest has no official checkpoint record")
    prefix_checkpoint_sha = _require_sha256(
        prefix_checkpoint.get("sha256"),
        "prefix checkpoint SHA-256",
    )
    prefix_checkpoint_path_value = prefix_checkpoint.get("path")
    if not isinstance(prefix_checkpoint_path_value, str):
        raise ValueError("prefix checkpoint record has no path")
    prefix_checkpoint_path = _anchored(
        Path(prefix_checkpoint_path_value),
        repo_root,
    )
    _verify_hash(
        prefix_checkpoint_path,
        prefix_checkpoint_sha,
        "prefix official checkpoint",
    )
    if (
        prefix_checkpoint.get("bytes") is not None
        and prefix_checkpoint_path.stat().st_size
        != int(prefix_checkpoint["bytes"])
    ):
        raise ValueError("prefix official checkpoint byte size mismatch")
    expected_prefix_ids = set(prefix_ids)
    for line_number, row in enumerate(prefix_rows, start=1):
        row_id = row.get("id")
        if row_id not in expected_prefix_ids:
            raise ValueError(
                f"physical prefix row {line_number} has unexpected ID {row_id}"
            )
        _row_provenance_identity(
            row,
            row_label=f"physical prefix row {line_number} ({row_id})",
            run_id=prefix_run_id,
            fingerprint=prefix_fingerprint,
            checkpoint_sha256=prefix_checkpoint_sha,
        )
    prefix_latest = _latest_by_id(prefix_rows)
    full_latest = _latest_by_id(full_rows)
    if set(prefix_latest) != expected_prefix_ids:
        raise ValueError("latest prefix result IDs do not equal prefix inputs")
    fields = (
        "score",
        "classification",
        "classification_logit",
        "classification_probability",
        "classification_decision_strict_gt_0_5",
        "localization",
        "preprocess",
        "decoder_logits_128_sha256",
        "seg_logits_raw_128_sha256",
        "resized_logits_512_sha256",
        "seg_logits_512_sha256",
        "probability_512_sha256",
        "seg_probability_512_sha256",
        "probability_native_sha256",
        "seg_probability_native_sha256",
        "mask_native_sha256",
        "mask_sha256",
        "artifact_sha256",
        "artifacts",
        "canonical_artifact_hashes_shapes_dtypes",
    )
    for sample_id in prefix_ids:
        prefix = prefix_latest.get(sample_id)
        full = full_latest.get(sample_id)
        if prefix is None or full is None:
            raise ValueError(f"prefix/full missing result {sample_id}")
        if prefix.get("status") != "ok" or full.get("status") != "ok":
            raise ValueError(f"prefix/full {sample_id} is not status ok")
        full_source_commit = full.get(
            "model_source_commit",
            full.get("source_commit"),
        )
        if full_source_commit != prefix_source_commit:
            raise ValueError(
                f"prefix/full {sample_id} BR-Gen provenance mismatch"
            )
        if full.get("imdlbenco_source_commit") != IMDLBENCO_SOURCE_COMMIT:
            raise ValueError(
                f"prefix/full {sample_id} IMDLBenCo provenance mismatch"
            )
        if full.get("checkpoint_sha256") != prefix_checkpoint_sha:
            raise ValueError(
                f"prefix/full {sample_id} checkpoint provenance mismatch"
            )
        for field in fields:
            if field == "canonical_artifact_hashes_shapes_dtypes":
                for canonical_name in ARTIFACT_ALIASES:
                    prefix_artifact = _resolve_artifact(
                        prefix,
                        canonical_name=canonical_name,
                        repo_root=repo_root,
                    )
                    full_artifact = _resolve_artifact(
                        full,
                        canonical_name=canonical_name,
                        repo_root=repo_root,
                    )
                    if (
                        prefix_artifact.sha256,
                        prefix_artifact.shape,
                        prefix_artifact.dtype,
                    ) != (
                        full_artifact.sha256,
                        full_artifact.shape,
                        full_artifact.dtype,
                    ):
                        raise ValueError(
                            "prefix reproducibility mismatch: "
                            f"{sample_id} {canonical_name}"
                        )
                continue
            if field in prefix or field in full:
                if prefix.get(field) != full.get(field):
                    raise ValueError(
                        f"prefix reproducibility mismatch: {sample_id} {field}"
                    )
    return {
        "status": "ok",
        "policy": (
            "all physical prefix rows provenance-validated; latest physical "
            "row per sample id compared for deterministic outputs"
        ),
        "prefix_images": len(prefix_ids),
        "prefix_pairs": len(prefix_ids) // 2,
        "full_images": len(full_ids),
        "physical_prefix_rows_provenance_validated": len(prefix_rows),
        "prefix_manifest_fingerprint": prefix_fingerprint,
        "prefix_checkpoint_sha256": prefix_checkpoint_sha,
        "prefix_source_commit": prefix_source_commit,
        "fields_compared": list(fields),
        "prefix_run_id": prefix_run_id,
        "prefix_results_path": _relative_or_absolute(results_path, repo_root),
        "prefix_results_sha256": sha256_file(results_path),
        "prefix_manifest_path": _relative_or_absolute(manifest_path, repo_root),
        "prefix_manifest_sha256": sha256_file(manifest_path),
    }


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = Path(args.repo_root).resolve()
    results_dir = _anchored(Path(args.results_dir), repo_root)
    inputs_path = _anchored(Path(args.inputs), repo_root)
    nfa_vit_root = Path(args.nfa_vit_root).resolve()
    imdlbenco_root = Path(args.imdlbenco_root).resolve()
    run_id = str(args.run_id)
    results_path = results_dir / f"{run_id}.jsonl"
    manifest_path = results_dir / f"{run_id}.run_manifest.json"
    summary_path = results_dir / f"{run_id}.summary.json"

    input_rows = read_jsonl(inputs_path)
    result_rows = read_jsonl(results_path)
    manifest = _require_mapping(
        json.loads(manifest_path.read_text(encoding="utf-8")),
        "run manifest",
    )
    summary = _require_mapping(
        json.loads(summary_path.read_text(encoding="utf-8")),
        "run summary",
    )
    expected_rows = _select_manifest_inputs(input_rows, manifest)
    history = summarize_result_history(result_rows)

    provenance = validate_provenance(
        repo_root=repo_root,
        nfa_vit_root=nfa_vit_root,
        imdlbenco_root=imdlbenco_root,
        run_id=run_id,
        inputs_path=inputs_path,
        expected_rows=expected_rows,
        result_rows=result_rows,
        manifest=manifest,
        summary=summary,
        expected_checkpoint_sha256=getattr(
            args,
            "checkpoint_sha256_test_contract",
            None,
        ),
    )
    artifact_integrity, pairs = audit_artifacts(
        repo_root=repo_root,
        expected_rows=expected_rows,
        result_rows=result_rows,
        manifest=manifest,
    )
    recorded_bootstrap_samples, recorded_bootstrap_seed = (
        _recorded_bootstrap_contract(summary)
    )
    recomputed_summary = summarize_nfa_vit_results(
        result_rows,
        expected_rows,
        bootstrap_samples=recorded_bootstrap_samples,
        seed=recorded_bootstrap_seed,
    )
    _compare_summary(summary, recomputed_summary)

    overall = _slice_summary(
        pairs,
        iterations=int(args.bootstrap_iterations),
        seed=int(args.bootstrap_seed),
    )
    by_domain: dict[str, Any] = {}
    domains = sorted({str(pair.forged.get("domain")) for pair in pairs})
    for index, domain in enumerate(domains, start=1):
        selected = [
            pair for pair in pairs if str(pair.forged.get("domain")) == domain
        ]
        by_domain[domain] = _slice_summary(
            selected,
            iterations=int(args.bootstrap_iterations),
            seed=int(args.bootstrap_seed) + index,
        )
    by_quintile = {
        name: _slice_summary(
            selected,
            iterations=int(args.bootstrap_iterations),
            seed=int(args.bootstrap_seed) + 100 + index,
        )
        for index, (name, selected) in enumerate(_quintiles(pairs), start=1)
    }
    prefix = audit_prefix_reproducibility(
        repo_root=repo_root,
        full_expected=expected_rows,
        full_rows=result_rows,
        prefix_run_id=args.prefix_run_id,
        prefix_results_dir=_anchored(
            Path(args.prefix_results_dir),
            repo_root,
        ),
    )
    return {
        "schema_version": "nfa_vit_posthoc_analysis_v1",
        "run_id": run_id,
        "created_at": utc_now(),
        "task_scope": recomputed_summary["task_scope"],
        "overall": overall,
        "fixed_threshold_metrics": {
            "status": "primary_frozen_protocol_metrics",
            "classification_threshold": FIXED_CLASSIFICATION_THRESHOLD,
            "classification_threshold_operator": THRESHOLD_OPERATOR,
            "mask_threshold": FIXED_MASK_THRESHOLD,
            "mask_threshold_operator": THRESHOLD_OPERATOR,
            "detection": recomputed_summary["detection"],
            "native_t1_saturation_diagnostic": recomputed_summary[
                "native_t1_saturation_diagnostic"
            ],
            "raw_logit_secondary_diagnostic": recomputed_summary[
                "raw_logit_secondary_diagnostic"
            ],
            "localization_forged": recomputed_summary[
                "localization_forged"
            ],
            "localization_real": recomputed_summary["localization_real"],
            "joint_diagnostics": recomputed_summary["joint_diagnostics"],
        },
        "by_domain": {
            "status": "posthoc_stratified_diagnostic_only",
            "eligible_for_primary_metrics": False,
            "uses_test_set_annotations": True,
            "slices": by_domain,
        },
        "by_edit_fraction_quintile": {
            "status": "posthoc_stratified_diagnostic_only",
            "eligible_for_primary_metrics": False,
            "uses_test_set_annotations": True,
            "slices": by_quintile,
        },
        "box_hit_at_native_mask_threshold_0_5": _box_hit(
            pairs,
            repo_root=repo_root,
        ),
        "bootstrap": {
            "unit": "paired task (real and forged resampled together)",
            "iterations": int(args.bootstrap_iterations),
            "seed": int(args.bootstrap_seed),
            "interval": "2.5th and 97.5th percentile",
            "spaces": list(LOCALIZATION_SPACES),
            "metrics_scope": "native T1, native T2, and S_joint",
        },
        "prefix_reproducibility": prefix,
        "artifact_integrity": artifact_integrity,
        "provenance_integrity": provenance,
        "result_history": history,
        "sources": {
            "results_path": _relative_or_absolute(results_path, repo_root),
            "results_sha256": sha256_file(results_path),
            "run_manifest_path": _relative_or_absolute(
                manifest_path,
                repo_root,
            ),
            "run_manifest_sha256": sha256_file(manifest_path),
            "summary_path": _relative_or_absolute(summary_path, repo_root),
            "summary_sha256": sha256_file(summary_path),
            "inputs_path": _relative_or_absolute(inputs_path, repo_root),
            "inputs_sha256": sha256_file(inputs_path),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit and analyze one official NFA-ViT paired run",
    )
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--inputs", type=Path, default=DEFAULT_INPUTS)
    parser.add_argument(
        "--nfa-vit-root",
        type=Path,
        default=DEFAULT_NFA_VIT_ROOT,
    )
    parser.add_argument(
        "--imdlbenco-root",
        type=Path,
        default=DEFAULT_IMDLBENCO_ROOT,
    )
    parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260724)
    parser.add_argument("--prefix-run-id")
    parser.add_argument(
        "--prefix-results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    analysis = analyze(args)
    repo_root = Path(args.repo_root).resolve()
    output = (
        _anchored(Path(args.output), repo_root)
        if args.output is not None
        else _anchored(Path(args.results_dir), repo_root)
        / f"{args.run_id}.analysis.json"
    )
    atomic_write_json(output, analysis)
    print(json.dumps(analysis, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
