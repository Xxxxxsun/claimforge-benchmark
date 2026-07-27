#!/usr/bin/env python3
"""Build the fixed 750-forged + 250-real MLLM aggregation ledger."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps

REPO_IMPORT_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_IMPORT_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_IMPORT_ROOT))

from eval.opensource.common import (
    atomic_write_json,
    atomic_write_jsonl,
    read_jsonl,
    repo_relative,
    sha256_file,
    stable_json,
)


SCHEMA_VERSION = "claimforge_mllm_benchmark1000_v1"
ROW_SCHEMA_VERSION = "claimforge_mllm_benchmark_image_v1"
DATASET_ID = "claimforge-mllm-local750-real250-v1"
INFERENCE_DATASET_ID = "claimforge-mllm-inference-full1051-v1"
CAT_EXCLUSION_SEED = (
    "claimforge-mllm-local750-real250-v1::cat-drop::20260727"
)
CONDITION_ORDER = ("mouse", "cat", "trash_can", "real")

TRASH_MANIFEST = Path(
    "spliced_final/claimforge_trash_can_selected_250_20260725/manifest.jsonl"
)
CAT_MANIFEST = Path(
    "spliced_final/claimforge_cat_selected_251_20260725/manifest.jsonl"
)
MOUSE_REVIEW = Path("claimforge_generation_review_labels.json")
TRASH_TASKS = Path("annotations/trash_can_generation_tasks.jsonl")
CAT_TASKS = Path("annotations/cat_generation_tasks.jsonl")
REAL_TEST_EVIDENCE = Path(
    "results/mllm/qwen3_7_plus/"
    "qwen37plus_pilot_good275_c15_v3_20260715T153257_0800.jsonl"
)

DEFAULT_MANIFEST = Path(
    "annotations/claimforge_mllm_benchmark1000_v1.manifest.json"
)
DEFAULT_LEDGER = Path(
    "annotations/claimforge_mllm_benchmark1000_v1.jsonl"
)
DEFAULT_INFERENCE_LIST = Path(
    "annotations/claimforge_mllm_inference_full1051_v1.jsonl"
)


def normalized_task_id(task_id: str) -> str:
    for prefix in ("trash_can_", "cat_"):
        if task_id.startswith(prefix):
            return task_id[len(prefix) :]
    return task_id


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _unique(
    rows: list[dict[str, Any]],
    key,
    label: str,
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        value = str(key(row))
        if not value:
            raise ValueError(f"{label}: empty identity")
        if value in result:
            raise ValueError(f"{label}: duplicate identity {value}")
        result[value] = row
    return result


def _repo_file(root: Path, raw: str | Path) -> Path:
    path = Path(raw)
    resolved = path.resolve() if path.is_absolute() else (root / path).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"path escapes repository: {raw}") from exc
    if not resolved.is_file():
        raise FileNotFoundError(raw)
    return resolved


def _image_record(
    root: Path,
    raw: str | Path,
    cache: dict[Path, dict[str, Any]],
) -> dict[str, Any]:
    path = _repo_file(root, raw)
    cached = cache.get(path)
    if cached is not None:
        return dict(cached)
    with Image.open(path) as opened:
        image = ImageOps.exif_transpose(opened)
        width, height = image.size
    record = {
        "path": repo_relative(path, root),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "width": width,
        "height": height,
    }
    cache[path] = record
    return dict(record)


def _input_binding(root: Path, relative: Path) -> dict[str, Any]:
    path = _repo_file(root, relative)
    return {
        "path": repo_relative(path, root),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def _assert_geometry(
    task_id: str,
    image: dict[str, Any],
    source: dict[str, Any],
) -> None:
    if (
        image["width"] != source["width"]
        or image["height"] != source["height"]
    ):
        raise ValueError(f"{task_id}: forged/source geometry mismatch")


def _row(
    *,
    benchmark_id: str,
    task_id: str,
    scene_id: str,
    label: str,
    candidate: str,
    image: dict[str, Any],
    source: dict[str, Any],
    edit_region_xyxy: Any,
    context_region_xyxy: Any,
    selection: str,
    origin: str,
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metadata = {
        "dataset_id": DATASET_ID,
        "benchmark_id": benchmark_id,
        "scene_id": scene_id,
        "candidate": candidate,
        "selection": selection,
        "origin": origin,
        **(extra_metadata or {}),
    }
    return {
        "schema_version": ROW_SCHEMA_VERSION,
        "id": benchmark_id,
        "benchmark_id": benchmark_id,
        "task_id": task_id,
        "scene_id": scene_id,
        "image_path": image["path"],
        "image_sha256": image["sha256"],
        "image_bytes": image["bytes"],
        "image_size": {
            "width": image["width"],
            "height": image["height"],
        },
        "label": label,
        "candidate": candidate,
        "source_image": source["path"],
        "source_sha256": source["sha256"],
        "edit_region_xyxy": edit_region_xyxy,
        "context_region_xyxy": context_region_xyxy,
        "metadata": metadata,
    }


def _cat_random_rank(task_id: str) -> str:
    value = f"{CAT_EXCLUSION_SEED}\0{task_id}".encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def _real_test_evidence(
    root: Path,
    path: Path,
) -> dict[str, dict[str, Any]]:
    rows = read_jsonl(_repo_file(root, path))
    evidence: dict[str, dict[str, Any]] = {}
    for row in rows:
        if (
            row.get("protocol_key") != "detection"
            or row.get("status") != "ok"
            or row.get("valid_for_metrics") is False
            or not str(row.get("id", "")).endswith("__real")
        ):
            continue
        task_id = str(row.get("task_id", ""))
        if not task_id:
            raise ValueError("real evidence row lacks task_id")
        if task_id in evidence:
            raise ValueError(f"duplicate real evidence for {task_id}")
        evidence[task_id] = row
    if len(evidence) != 275:
        raise ValueError(
            f"expected 275 previously tested real rows, got {len(evidence)}"
        )
    return evidence


def build_benchmark(
    repo_root: Path,
    ledger_path: Path = DEFAULT_LEDGER,
    inference_path: Path = DEFAULT_INFERENCE_LIST,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    root = repo_root.resolve()
    cache: dict[Path, dict[str, Any]] = {}
    trash_manifest_rows = read_jsonl(_repo_file(root, TRASH_MANIFEST))
    cat_manifest_rows = read_jsonl(_repo_file(root, CAT_MANIFEST))
    trash_tasks = _unique(
        read_jsonl(_repo_file(root, TRASH_TASKS)),
        lambda row: row.get("task_id", ""),
        "trash tasks",
    )
    cat_tasks = _unique(
        read_jsonl(_repo_file(root, CAT_TASKS)),
        lambda row: row.get("task_id", ""),
        "cat tasks",
    )
    mouse_payload = _load_json(_repo_file(root, MOUSE_REVIEW))
    mouse_good_rows = [
        row
        for row in mouse_payload.get("records", [])
        if row.get("status") == "good" and row.get("candidates") == "mouse"
    ]
    mouse_good = _unique(
        mouse_good_rows,
        lambda row: normalized_task_id(str(row.get("task_id", ""))),
        "mouse reviewed-good",
    )
    trash_final = _unique(
        trash_manifest_rows,
        lambda row: normalized_task_id(str(row.get("task_id", ""))),
        "trash final250",
    )
    cat_final = _unique(
        cat_manifest_rows,
        lambda row: str(row.get("task_id", "")),
        "cat final251",
    )
    if (
        len(trash_final) != 250
        or len(cat_final) != 251
        or len(mouse_good) != 275
    ):
        raise ValueError(
            "unexpected upstream counts: "
            f"trash={len(trash_final)}, cat={len(cat_final)}, "
            f"mouse={len(mouse_good)}"
        )

    rows_by_condition: dict[str, list[dict[str, Any]]] = {
        key: [] for key in CONDITION_ORDER
    }
    trash_sources: list[tuple[str, dict[str, Any]]] = []
    for scene_id in sorted(trash_final):
        final_row = trash_final[scene_id]
        trash_task_id = str(final_row["task_id"])
        trash_task = trash_tasks.get(trash_task_id)
        mouse_row = mouse_good.get(scene_id)
        if trash_task is None or mouse_row is None:
            raise ValueError(f"{scene_id}: missing trash task or good mouse")
        source = _image_record(
            root, str(trash_task["source_image"]), cache
        )
        mouse_source = _image_record(
            root, str(mouse_row["source_image"]), cache
        )
        if mouse_source["sha256"] != source["sha256"]:
            raise ValueError(
                f"{scene_id}: mouse does not use trash-can original"
            )
        trash_image = _image_record(
            root, str(final_row["image"]), cache
        )
        mouse_image = _image_record(
            root, str(mouse_row["spliced_image"]), cache
        )
        _assert_geometry(trash_task_id, trash_image, source)
        _assert_geometry(str(mouse_row["task_id"]), mouse_image, source)
        trash_sources.append((scene_id, source))
        rows_by_condition["mouse"].append(
            _row(
                benchmark_id=f"local_mouse__{scene_id}",
                task_id=str(mouse_row["task_id"]),
                scene_id=scene_id,
                label="forged",
                candidate="mouse",
                image=mouse_image,
                source=source,
                edit_region_xyxy=mouse_row.get("edit_region_xyxy"),
                context_region_xyxy=mouse_row.get(
                    "context_region_xyxy"
                ),
                selection="source_matches_trash_final250",
                origin="mouse_good275",
            )
        )
        rows_by_condition["trash_can"].append(
            _row(
                benchmark_id=f"local_trash_can__{scene_id}",
                task_id=trash_task_id,
                scene_id=scene_id,
                label="forged",
                candidate="trash_can",
                image=trash_image,
                source=source,
                edit_region_xyxy=trash_task.get("edit_region_xyxy"),
                context_region_xyxy=trash_task.get(
                    "context_region_xyxy"
                ),
                selection=str(final_row.get("selection")),
                origin="final_trash250",
            )
        )

    cat_task_ids = sorted(cat_final)
    excluded_cat_task_id = min(cat_task_ids, key=_cat_random_rank)
    excluded_cat: dict[str, Any] | None = None
    excluded_cat_row: dict[str, Any] | None = None
    for task_id in cat_task_ids:
        final_row = cat_final[task_id]
        task = cat_tasks.get(task_id)
        if task is None:
            raise ValueError(f"{task_id}: missing cat task metadata")
        source = _image_record(root, str(task["source_image"]), cache)
        image = _image_record(root, str(final_row["image"]), cache)
        _assert_geometry(task_id, image, source)
        if task_id == excluded_cat_task_id:
            excluded_cat = {
                "task_id": task_id,
                "scene_id": normalized_task_id(task_id),
                "image": image,
                "source": source,
                "selection": final_row.get("selection"),
                "random_rank_sha256": _cat_random_rank(task_id),
            }
            excluded_cat_row = _row(
                benchmark_id=f"local_cat__{normalized_task_id(task_id)}",
                task_id=task_id,
                scene_id=normalized_task_id(task_id),
                label="forged",
                candidate="cat",
                image=image,
                source=source,
                edit_region_xyxy=task.get("edit_region_xyxy"),
                context_region_xyxy=task.get("context_region_xyxy"),
                selection=str(final_row.get("selection")),
                origin="final_cat251_inference_extra",
            )
            continue
        scene_id = normalized_task_id(task_id)
        rows_by_condition["cat"].append(
            _row(
                benchmark_id=f"local_cat__{scene_id}",
                task_id=task_id,
                scene_id=scene_id,
                label="forged",
                candidate="cat",
                image=image,
                source=source,
                edit_region_xyxy=task.get("edit_region_xyxy"),
                context_region_xyxy=task.get("context_region_xyxy"),
                selection=str(final_row.get("selection")),
                origin="final_cat251_seeded_random250",
                extra_metadata={
                    "cat_exclusion_seed": CAT_EXCLUSION_SEED,
                },
            )
        )
    if (
        excluded_cat is None
        or excluded_cat_row is None
        or len(rows_by_condition["cat"]) != 250
    ):
        raise AssertionError("Cat seeded exclusion did not yield 250 rows")

    evidence = _real_test_evidence(root, REAL_TEST_EVIDENCE)
    real_candidates: dict[str, dict[str, Any]] = {}
    for scene_id, mouse_row in mouse_good.items():
        source = _image_record(
            root, str(mouse_row["source_image"]), cache
        )
        prior = evidence.get(str(mouse_row["task_id"]))
        if prior is None:
            raise ValueError(f"{scene_id}: real image was not tested")
        if prior.get("image_sha256") != source["sha256"]:
            raise ValueError(
                f"{scene_id}: historical real result SHA-256 mismatch"
            )
        real_candidates[scene_id] = {
            "mouse_row": mouse_row,
            "source": source,
            "historical_result": prior,
        }

    selected_real_sha: set[str] = set()
    selected_real_scenes: set[str] = set()

    def add_real(scene_id: str, selection: str) -> bool:
        candidate = real_candidates[scene_id]
        source = candidate["source"]
        if source["sha256"] in selected_real_sha:
            return False
        mouse_row = candidate["mouse_row"]
        rows_by_condition["real"].append(
            _row(
                benchmark_id=f"real__{scene_id}",
                task_id=str(mouse_row["task_id"]),
                scene_id=scene_id,
                label="real",
                candidate="real",
                image=source,
                source=source,
                edit_region_xyxy=None,
                context_region_xyxy=None,
                selection=selection,
                origin="previously_tested_mouse_good275_real",
                extra_metadata={
                    "historical_test_result": repo_relative(
                        _repo_file(root, REAL_TEST_EVIDENCE), root
                    ),
                    "historical_test_run_id": candidate[
                        "historical_result"
                    ].get("run_id"),
                    "historical_test_model_slug": candidate[
                        "historical_result"
                    ].get("model_slug"),
                },
            )
        )
        selected_real_sha.add(source["sha256"])
        selected_real_scenes.add(scene_id)
        return True

    for scene_id, _ in trash_sources:
        add_real(scene_id, "unique_trash_source_priority")
    trash_priority_real_count = len(rows_by_condition["real"])
    for scene_id in sorted(real_candidates):
        if len(rows_by_condition["real"]) == 250:
            break
        if scene_id not in selected_real_scenes:
            add_real(scene_id, "unique_mouse275_supplement")
    if len(rows_by_condition["real"]) != 250:
        raise ValueError("could not select 250 unique previously tested reals")

    for condition, expected in (
        ("mouse", 250),
        ("cat", 250),
        ("trash_can", 250),
        ("real", 250),
    ):
        if len(rows_by_condition[condition]) != expected:
            raise AssertionError(f"{condition}: expected {expected} rows")

    ledger = [
        row
        for condition in CONDITION_ORDER
        for row in rows_by_condition[condition]
    ]
    ids = [str(row["id"]) for row in ledger]
    if len(ledger) != 1000 or len(set(ids)) != 1000:
        raise AssertionError("benchmark ledger must have 1000 unique IDs")
    if len({row["image_sha256"] for row in rows_by_condition["real"]}) != 250:
        raise AssertionError("real250 contains duplicate image content")

    extra_mouse_rows: list[dict[str, Any]] = []
    for scene_id in sorted(set(mouse_good) - set(trash_final)):
        mouse_row = mouse_good[scene_id]
        source = _image_record(
            root, str(mouse_row["source_image"]), cache
        )
        image = _image_record(
            root, str(mouse_row["spliced_image"]), cache
        )
        _assert_geometry(str(mouse_row["task_id"]), image, source)
        extra_mouse_rows.append(
            _row(
                benchmark_id=f"local_mouse__{scene_id}",
                task_id=str(mouse_row["task_id"]),
                scene_id=scene_id,
                label="forged",
                candidate="mouse",
                image=image,
                source=source,
                edit_region_xyxy=mouse_row.get("edit_region_xyxy"),
                context_region_xyxy=mouse_row.get(
                    "context_region_xyxy"
                ),
                selection="mouse_good275_inference_extra",
                origin="mouse_good275",
            )
        )
    extra_real_rows: list[dict[str, Any]] = []
    for scene_id in sorted(set(real_candidates) - selected_real_scenes):
        candidate = real_candidates[scene_id]
        mouse_row = candidate["mouse_row"]
        source = candidate["source"]
        extra_real_rows.append(
            _row(
                benchmark_id=f"real__{scene_id}",
                task_id=str(mouse_row["task_id"]),
                scene_id=scene_id,
                label="real",
                candidate="real",
                image=source,
                source=source,
                edit_region_xyxy=None,
                context_region_xyxy=None,
                selection="mouse_good275_real_inference_extra",
                origin="previously_tested_mouse_good275_real",
                extra_metadata={
                    "historical_test_result": repo_relative(
                        _repo_file(root, REAL_TEST_EVIDENCE), root
                    ),
                    "historical_test_run_id": candidate[
                        "historical_result"
                    ].get("run_id"),
                    "historical_test_model_slug": candidate[
                        "historical_result"
                    ].get("model_slug"),
                },
            )
        )
    if len(extra_mouse_rows) != 25 or len(extra_real_rows) != 25:
        raise AssertionError("full inference extras must be Mouse25 + Real25")

    formal_ids = {row["id"] for row in ledger}

    def inference_row(row: dict[str, Any]) -> dict[str, Any]:
        copied = dict(row)
        copied["metadata"] = {
            **row["metadata"],
            "dataset_id": INFERENCE_DATASET_ID,
            "aggregation_dataset_id": DATASET_ID,
            "in_benchmark1000": row["id"] in formal_ids,
        }
        return copied

    inference_rows = [
        *map(inference_row, rows_by_condition["mouse"]),
        *map(inference_row, extra_mouse_rows),
        *map(inference_row, rows_by_condition["cat"]),
        inference_row(excluded_cat_row),
        *map(inference_row, rows_by_condition["trash_can"]),
        *map(inference_row, rows_by_condition["real"]),
        *map(inference_row, extra_real_rows),
    ]
    inference_ids = [str(row["id"]) for row in inference_rows]
    if len(inference_rows) != 1051 or len(set(inference_ids)) != 1051:
        raise AssertionError("full inference list must have 1051 unique IDs")

    ledger_serialized = "".join(
        f"{stable_json(row)}\n" for row in ledger
    )
    inference_serialized = "".join(
        f"{stable_json(row)}\n" for row in inference_rows
    )
    resolved_ledger = (
        ledger_path if ledger_path.is_absolute() else root / ledger_path
    )
    resolved_inference = (
        inference_path
        if inference_path.is_absolute()
        else root / inference_path
    )
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "dataset_id": DATASET_ID,
        "release_ready": True,
        "counts": {
            "total": 1000,
            "forged": 750,
            "real": 250,
            "mouse": 250,
            "cat": 250,
            "trash_can": 250,
            "cat_population": 251,
            "cat_excluded": 1,
            "mouse_population": 275,
            "mouse_excluded": 25,
            "real_unique_sha256": 250,
            "real_from_unique_trash_sources": (
                trash_priority_real_count
            ),
            "real_supplemental_unique": (
                250 - trash_priority_real_count
            ),
        },
        "policy": {
            "inference_may_process_superset": True,
            "aggregation_must_filter_to_ledger": True,
            "mouse_selection": (
                "same scene/original as every final Trash-can-250 row"
            ),
            "cat_selection": (
                "seeded SHA-256 pseudorandom exclusion of one final "
                "Cat-251 row; no model outputs used"
            ),
            "trash_can_selection": "all final Trash-can-250 rows",
            "real_selection": (
                "250 unique SHA-256 images previously evaluated as real; "
                "prioritize unique Trash-can sources, then supplement from "
                "Mouse-good275"
            ),
        },
        "cat_sampling": {
            "algorithm": (
                "exclude min SHA256(seed + NUL + sorted task_id)"
            ),
            "seed": CAT_EXCLUSION_SEED,
            "excluded": excluded_cat,
        },
        "real_sampling": {
            "historical_test_evidence": _input_binding(
                root, REAL_TEST_EVIDENCE
            ),
            "unique_content_required": True,
            "trash_source_priority_count": trash_priority_real_count,
            "supplemental_count": 250 - trash_priority_real_count,
        },
        "built_from": [
            _input_binding(root, path)
            for path in (
                TRASH_MANIFEST,
                CAT_MANIFEST,
                MOUSE_REVIEW,
                TRASH_TASKS,
                CAT_TASKS,
                REAL_TEST_EVIDENCE,
            )
        ],
        "formal_ledger": {
            "path": repo_relative(resolved_ledger, root),
            "rows": len(ledger),
            "sha256": hashlib.sha256(
                ledger_serialized.encode("utf-8")
            ).hexdigest(),
        },
        "inference_superset": {
            "dataset_id": INFERENCE_DATASET_ID,
            "path": repo_relative(resolved_inference, root),
            "rows": len(inference_rows),
            "sha256": hashlib.sha256(
                inference_serialized.encode("utf-8")
            ).hexdigest(),
            "counts": {
                "mouse": 275,
                "cat": 251,
                "trash_can": 250,
                "real": 275,
                "total": 1051,
                "in_benchmark1000": 1000,
                "aggregation_extras": 51,
            },
        },
    }
    fingerprint = stable_json(manifest)
    manifest["content_sha256"] = hashlib.sha256(
        fingerprint.encode("utf-8")
    ).hexdigest()
    return manifest, ledger, inference_rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the fixed MLLM benchmark1000 aggregation ledger"
    )
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument(
        "--output-inference-list",
        type=Path,
        default=DEFAULT_INFERENCE_LIST,
    )
    args = parser.parse_args()
    root = args.repo_root.resolve()
    manifest, ledger, inference_rows = build_benchmark(
        root,
        args.output_ledger,
        args.output_inference_list,
    )
    manifest_path = (
        args.output_manifest
        if args.output_manifest.is_absolute()
        else root / args.output_manifest
    )
    ledger_path = (
        args.output_ledger
        if args.output_ledger.is_absolute()
        else root / args.output_ledger
    )
    inference_path = (
        args.output_inference_list
        if args.output_inference_list.is_absolute()
        else root / args.output_inference_list
    )
    atomic_write_jsonl(ledger_path, ledger)
    atomic_write_jsonl(inference_path, inference_rows)
    if sha256_file(ledger_path) != manifest["formal_ledger"]["sha256"]:
        raise AssertionError("written ledger SHA-256 differs from manifest")
    if (
        sha256_file(inference_path)
        != manifest["inference_superset"]["sha256"]
    ):
        raise AssertionError(
            "written inference-list SHA-256 differs from manifest"
        )
    atomic_write_json(manifest_path, manifest)
    print(
        json.dumps(
            {
                "dataset_id": DATASET_ID,
                "counts": manifest["counts"],
                "excluded_cat_task_id": manifest["cat_sampling"][
                    "excluded"
                ]["task_id"],
                "manifest": repo_relative(manifest_path, root),
                "ledger": repo_relative(ledger_path, root),
                "inference_list": repo_relative(inference_path, root),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
