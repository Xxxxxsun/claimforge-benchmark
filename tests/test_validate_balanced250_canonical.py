import hashlib
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import PIL
from PIL import Image, ImageChops, ImageOps, features

from eval.opensource.common import sha256_file, stable_json
from eval.opensource.validate_balanced250_canonical import (
    DATASET_ID,
    SCHEMA_VERSION,
    ValidationError,
    ValidationSpec,
    validate_release,
)


CONDITIONS = (
    "real",
    "local_mouse",
    "local_cat",
    "local_trash_can",
    "fullframe_mouse",
    "fullframe_cat",
    "fullframe_trash_can",
)
FORGED = CONDITIONS[1:]
LOCAL = CONDITIONS[1:4]


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(f"{stable_json(row)}\n" for row in rows),
        encoding="utf-8",
    )


def _norm(task_id: str) -> str:
    for prefix in ("trash_can_", "cat_"):
        if task_id.startswith(prefix):
            return task_id[len(prefix) :]
    return task_id


def _selection_key(condition: str, task_id: str) -> str:
    return hashlib.sha256(
        f"{DATASET_ID}\0{condition}\0{task_id}".encode()
    ).hexdigest()


def _sample_id(condition: str, task_id: str) -> str:
    return hashlib.sha256(
        f"{DATASET_ID}\0{condition}\0{task_id}\0sample".encode()
    ).hexdigest()[:24]


def _pair_id(condition: str, task_id: str) -> str:
    return hashlib.sha256(
        f"{DATASET_ID}\0{condition}\0{task_id}\0source-pair".encode()
    ).hexdigest()[:24]


def _rows_hash(rows: list[dict]) -> str:
    return hashlib.sha256(
        "".join(f"{stable_json(row)}\n" for row in rows).encode()
    ).hexdigest()


def _id_hash(values: list[str]) -> str:
    return hashlib.sha256("".join(f"{value}\n" for value in values).encode()).hexdigest()


class TinyRelease:
    spec = ValidationSpec(
        real_cache=2,
        forged_cache_per_condition=1,
        panel_per_condition=1,
        local_cat_eligible=1,
        local_trash_can_eligible=1,
        fullframe_cat_eligible=1,
        fullframe_trash_can_eligible=1,
    )

    def __init__(self, root: Path):
        self.root = root
        self.release = root / "release"
        (self.release / "images").mkdir(parents=True)
        (self.release / "masks").mkdir()
        self.tasks = (
            "lodging_001_slot_001",
            "restaurant_002_slot_001",
        )
        self.source_paths: dict[str, Path] = {}
        self.old_real: dict[str, dict] = {}
        self.old_forged: dict[str, dict] = {}
        self.old_pairs: dict[str, dict] = {}
        self.source_files: dict[str, Path] = {}
        self.eligibility_records: dict[str, list[dict]] = {
            condition: [] for condition in CONDITIONS
        }
        self.eligible_rows: dict[str, list[dict]] = {}
        self.selected_rows: dict[str, list[dict]] = {}
        self.inputs: list[dict] = []
        self.panel: list[dict] = []
        self.pairs: list[dict] = []
        self.manifest: dict = {}
        self._build()

    def rel(self, path: Path) -> str:
        return path.relative_to(self.root).as_posix()

    def image_record(self, path: Path) -> dict:
        with Image.open(path) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
        return {
            "path": self.rel(path),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
            "decoded_width": image.width,
            "decoded_height": image.height,
        }

    def file_record(self, path: Path) -> dict:
        return {
            "path": self.rel(path),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }

    def save_png(
        self,
        path: Path,
        color: tuple[int, int, int],
        *,
        changed: tuple[int, int, tuple[int, int, int]] | None = None,
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        image = Image.new("RGB", (8, 6), color)
        if changed is not None:
            image.putpixel(changed[:2], changed[2])
        image.save(path, format="PNG")

    def save_canonical(self, raw: Path, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(raw) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
        image.save(
            destination,
            format="JPEG",
            quality=95,
            subsampling=0,
            optimize=False,
        )

    def exact_mask(self, source: Path, forged: Path, destination: Path) -> dict:
        with Image.open(source) as opened:
            real = ImageOps.exif_transpose(opened).convert("RGB")
        with Image.open(forged) as opened:
            fake = ImageOps.exif_transpose(opened).convert("RGB")
        channels = ImageChops.difference(real, fake).split()
        maximum = ImageChops.lighter(
            channels[0],
            ImageChops.lighter(channels[1], channels[2]),
        )
        mask = maximum.point(lambda value: 255 if value else 0, mode="L")
        destination.parent.mkdir(parents=True, exist_ok=True)
        mask.save(destination, format="PNG", optimize=False)
        positive = sum(mask.histogram()[1:])
        context = [0, 0, 5, 5]
        inside = sum(mask.crop(tuple(context)).histogram()[1:])
        return {
            "path": self.rel(destination),
            "sha256": sha256_file(destination),
            "positive": positive,
            "bbox": list(mask.getbbox()),
            "outside": positive - inside,
            "fraction": positive / 48,
            "context": context,
        }

    def canonical_fields(
        self,
        raw: Path,
        canonical: Path,
        origin: str,
    ) -> dict:
        return {
            "canonical_path": self.rel(canonical),
            "canonical_sha256": sha256_file(canonical),
            "canonical_bytes": canonical.stat().st_size,
            "canonical_origin": origin,
            "width": 8,
            "height": 6,
            "raw_path": self.rel(raw),
            "raw_sha256": sha256_file(raw),
            "raw_bytes": raw.stat().st_size,
        }

    def eligibility_hash(self, condition: str) -> str:
        ordered = sorted(
            self.eligibility_records[condition],
            key=lambda row: row["normalized_task_id"],
        )
        return _rows_hash(ordered)

    def ranked(self, rows: list[dict], condition: str) -> list[dict]:
        return sorted(
            rows,
            key=lambda row: (
                _selection_key(condition, _norm(row["task_id"])),
                _norm(row["task_id"]),
            ),
        )

    def _build_mouse(self) -> None:
        mouse_inputs: list[dict] = []
        mouse_pairs: list[dict] = []
        colors = ((20, 30, 40), (80, 90, 100))
        for index, (task_id, color) in enumerate(zip(self.tasks, colors)):
            source = self.root / f"upstream/raw/{task_id}.png"
            forged = self.root / f"upstream/mouse/{task_id}.png"
            self.save_png(source, color)
            self.save_png(
                forged,
                color,
                changed=(1 + index, 1, (220, 10, 10)),
            )
            self.source_paths[task_id] = source
            real_id = f"old-real-{index}"
            fake_id = f"old-fake-{index}"
            real_jpeg = self.root / f"upstream/mouse-canonical/{real_id}.jpg"
            fake_jpeg = self.root / f"upstream/mouse-canonical/{fake_id}.jpg"
            self.save_canonical(source, real_jpeg)
            self.save_canonical(forged, fake_jpeg)
            mask_info = self.exact_mask(
                source,
                forged,
                self.root / f"upstream/mouse-masks/{fake_id}.png",
            )
            common = {
                "task_id": task_id,
                "width": 8,
                "height": 6,
                "edit_region_xyxy": [1, 1, 3, 3],
                "context_region_xyxy": mask_info["context"],
            }
            real = {
                **common,
                "sample_id": real_id,
                "kind": "real",
                "label": 0,
                "raw_path": self.rel(source),
                "raw_sha256": sha256_file(source),
                "canonical_path": self.rel(real_jpeg),
                "canonical_sha256": sha256_file(real_jpeg),
                "canonical_bytes": real_jpeg.stat().st_size,
            }
            fake = {
                **common,
                "sample_id": fake_id,
                "kind": "forged",
                "label": 1,
                "raw_path": self.rel(forged),
                "raw_sha256": sha256_file(forged),
                "canonical_path": self.rel(fake_jpeg),
                "canonical_sha256": sha256_file(fake_jpeg),
                "canonical_bytes": fake_jpeg.stat().st_size,
                "gt_mask_path": mask_info["path"],
                "gt_mask_sha256": mask_info["sha256"],
                "gt_positive_pixels": mask_info["positive"],
            }
            variant_keys = (
                "canonical_bytes",
                "canonical_path",
                "canonical_sha256",
                "kind",
                "label",
                "raw_path",
                "raw_sha256",
                "sample_id",
            )
            pair = {
                "task_id": task_id,
                "real": {key: real[key] for key in variant_keys},
                "forged": {key: fake[key] for key in variant_keys},
                "gt_mask_sha256": mask_info["sha256"],
                "gt_positive_pixels": mask_info["positive"],
                "gt_bbox_xyxy": mask_info["bbox"],
                "gt_pixels_outside_context": mask_info["outside"],
                "gt_fraction": mask_info["fraction"],
            }
            self.old_real[task_id] = real
            self.old_forged[task_id] = fake
            self.old_pairs[task_id] = pair
            mouse_inputs.extend((real, fake))
            mouse_pairs.append(pair)
            self.eligibility_records["real"].append(
                {
                    "condition": "real",
                    "task_id": task_id,
                    "normalized_task_id": task_id,
                    "source": self.image_record(source),
                    "canonical": self.image_record(real_jpeg),
                }
            )
            self.eligibility_records["local_mouse"].append(
                {
                    "condition": "local_mouse",
                    "task_id": task_id,
                    "normalized_task_id": task_id,
                    "source": self.image_record(source),
                    "candidate": self.image_record(forged),
                    "canonical": self.image_record(fake_jpeg),
                    "mask": self.file_record(
                        self.root / mask_info["path"]
                    ),
                }
            )
        inputs_path = self.root / "upstream/mouse-inputs.jsonl"
        pairs_path = self.root / "upstream/mouse-pairs.jsonl"
        _write_jsonl(inputs_path, mouse_inputs)
        _write_jsonl(pairs_path, mouse_pairs)
        mouse_manifest = {
            "schema_version": "claimforge_mouse_canonical_v1",
            "inputs_path": self.rel(inputs_path),
            "inputs_sha256": sha256_file(inputs_path),
            "pairs_path": self.rel(pairs_path),
            "pairs_sha256": sha256_file(pairs_path),
        }
        manifest_path = self.root / "upstream/mouse-manifest.json"
        _write_json(manifest_path, mouse_manifest)
        self.source_files.update(
            {
                "mouse_release_manifest": manifest_path,
                "mouse_inputs": inputs_path,
                "mouse_pairs": pairs_path,
            }
        )

    def _task_row(self, task_id: str, candidate: str) -> dict:
        source = self.source_paths[_norm(task_id)]
        return {
            "task_id": task_id,
            "source_image": self.rel(source),
            "image_size": {"width": 8, "height": 6},
            "edit_region_xyxy": [1, 1, 3, 3],
            "context_region_xyxy": [0, 0, 5, 5],
            "candidates": candidate,
        }

    def _build_local_and_whole_sources(self) -> None:
        condition_task = {
            "local_cat": f"cat_{self.tasks[0]}",
            "local_trash_can": f"trash_can_{self.tasks[1]}",
        }
        for condition, task_id in condition_task.items():
            prefix = "cat" if condition == "local_cat" else "trash"
            source = self.source_paths[_norm(task_id)]
            forged = self.root / f"upstream/{prefix}-local/{task_id}.png"
            with Image.open(source) as opened:
                color = opened.convert("RGB").getpixel((0, 0))
            self.save_png(forged, color, changed=(2, 2, (5, 230, 15)))
            task = self._task_row(
                task_id,
                "cat" if prefix == "cat" else "trash can",
            )
            provenance = f"upstream/missing-provenance/{task_id}.png"
            delivered = {
                "task_id": task_id,
                "image": self.rel(forged),
                "source_image": provenance,
                "selection": "selected",
                "image_size": [8, 6],
                "bytes": forged.stat().st_size,
            }
            selection = {
                "task_id": task_id,
                "selection": "selected",
                "selected_spliced_full": provenance,
            }
            selection_path = self.root / f"upstream/{prefix}-selection.json"
            materialized_path = (
                self.root / f"upstream/{prefix}-materialized.jsonl"
            )
            _write_json(selection_path, {"selections": [selection]})
            _write_jsonl(materialized_path, [delivered])
            self.source_files[f"{prefix}_selection"] = selection_path
            self.source_files[f"{prefix}_materialized"] = materialized_path
            self.eligibility_records[condition].append(
                {
                    "condition": condition,
                    "task_id": task_id,
                    "normalized_task_id": _norm(task_id),
                    "selection": "selected",
                    "selected_candidate_path": provenance,
                    "source": self.image_record(source),
                    "candidate": self.image_record(forged),
                }
            )

        whole_task_rows: dict[str, list[dict]] = {}
        whole_run_rows: dict[str, list[dict]] = {}
        condition_tasks = {
            "fullframe_mouse": list(self.tasks),
            "fullframe_cat": [f"cat_{self.tasks[0]}"],
            "fullframe_trash_can": [f"trash_can_{self.tasks[1]}"],
        }
        metadata = {
            "fullframe_mouse": ("mouse", "mouse"),
            "fullframe_cat": ("cat", "cat"),
            "fullframe_trash_can": ("trash can", "trash-can"),
        }
        for condition, task_ids in condition_tasks.items():
            task_rows: list[dict] = []
            run_rows: list[dict] = []
            candidate, object_kind = metadata[condition]
            for offset, task_id in enumerate(task_ids):
                source = self.source_paths[_norm(task_id)]
                task = self._task_row(task_id, candidate)
                output = self.root / f"upstream/{condition}/{task_id}.png"
                with Image.open(source) as opened:
                    color = opened.convert("RGB").getpixel((0, 0))
                self.save_png(
                    output,
                    color,
                    changed=(4, 3, (10 + offset, 20, 240)),
                )
                run = {
                    "task_id": task_id,
                    "status": "ok",
                    "input_source_image": self.rel(source),
                    "input_source_sha256": sha256_file(source),
                    "output_image": self.rel(output),
                    "original_size": [8, 6],
                    "input_mode": "full-image-orange-box",
                    "orange_box_xyxy": task["edit_region_xyxy"],
                    "candidate": candidate,
                    "object_kind": object_kind,
                    "model": "tiny-model",
                    "service_model": "tiny-service",
                    "seed": offset,
                    "steps": 5,
                    "guidance_scale": 1.0,
                    "bot_task": f"bot-{offset}",
                }
                task_rows.append(task)
                run_rows.append(run)
                self.eligibility_records[condition].append(
                    {
                        "condition": condition,
                        "task_id": task_id,
                        "normalized_task_id": _norm(task_id),
                        "source": self.image_record(source),
                        "candidate": self.image_record(output),
                        "latest_run_row_index": len(run_rows) - 1,
                        "conditioning_box_xyxy": task["edit_region_xyxy"],
                        "input_mode": "full-image-orange-box",
                        "object_kind": object_kind,
                    }
                )
            task_path = self.root / f"upstream/{condition}-tasks.jsonl"
            run_path = self.root / f"upstream/{condition}-run.jsonl"
            _write_jsonl(task_path, task_rows)
            _write_jsonl(run_path, run_rows)
            self.source_files[f"{condition}_tasks"] = task_path
            self.source_files[f"{condition}_run"] = run_path
            whole_task_rows[condition] = task_rows
            whole_run_rows[condition] = run_rows

        qc_path = self.root / "upstream/trash-qc.json"
        _write_json(
            qc_path,
            {"failures": [], "summary": {"total": 1, "usable": 1, "failed": 0}},
        )
        self.source_files["trash_whole_qc"] = qc_path
        self.whole_tasks = {
            condition: {row["task_id"]: row for row in rows}
            for condition, rows in whole_task_rows.items()
        }
        self.whole_runs = {
            condition: {row["task_id"]: row for row in rows}
            for condition, rows in whole_run_rows.items()
        }
        self.local_tasks = {
            "local_cat": self.whole_tasks["fullframe_cat"],
            "local_trash_can": self.whole_tasks["fullframe_trash_can"],
        }
        self.local_delivered = {
            "local_cat": json.loads(
                self.source_files["cat_materialized"].read_text().splitlines()[0]
            ),
            "local_trash_can": json.loads(
                self.source_files["trash_materialized"].read_text().splitlines()[0]
            ),
        }

    def _make_common_input(
        self,
        condition: str,
        task_id: str,
        canonical: dict,
        eligibility_rank: int,
        selection_rank: int | None,
        panel: bool,
    ) -> dict:
        normalized = _norm(task_id)
        family = (
            "real"
            if condition == "real"
            else (
                "local_splice"
                if condition in LOCAL
                else "full_frame_conditional_edit"
            )
        )
        scope = (
            "authentic"
            if condition == "real"
            else (
                "local_insertion"
                if condition in LOCAL
                else "conditional_full_frame_edit"
            )
        )
        candidate = {
            "local_mouse": "mouse",
            "local_cat": "cat",
            "local_trash_can": "trash_can",
            "fullframe_mouse": "mouse",
            "fullframe_cat": "cat",
            "fullframe_trash_can": "trash_can",
        }.get(condition)
        return {
            "schema_version": SCHEMA_VERSION,
            "dataset_id": DATASET_ID,
            "sample_id": _sample_id(condition, normalized),
            "condition": condition,
            "condition_family": family,
            "kind": "real" if condition == "real" else "forged",
            "label": 0 if condition == "real" else 1,
            "manipulation_scope": scope,
            "candidate": candidate,
            "task_id": task_id,
            "normalized_task_id": normalized,
            "domain": normalized.split("_", 1)[0],
            "selection_key": _selection_key(condition, normalized),
            "eligibility_rank": eligibility_rank,
            "selection_rank": selection_rank,
            "panel": panel,
            "eligible_set_sha256": self.eligibility_hash(condition),
            **canonical,
        }

    def _build_inputs(self) -> None:
        self.eligible_rows = {
            "real": list(self.old_real.values()),
            "local_mouse": list(self.old_forged.values()),
            "local_cat": [self.local_delivered["local_cat"]],
            "local_trash_can": [self.local_delivered["local_trash_can"]],
            **{
                condition: list(rows.values())
                for condition, rows in self.whole_tasks.items()
            },
        }
        for condition, rows in self.eligible_rows.items():
            ranked = self.ranked(rows, condition)
            self.eligible_rows[condition] = ranked
            self.selected_rows[condition] = ranked[:1]

        selected_real = {
            _norm(row["task_id"]): index
            for index, row in enumerate(self.selected_rows["real"])
        }
        built: list[dict] = []
        for eligibility_rank, row in enumerate(self.eligible_rows["real"]):
            task_id = row["task_id"]
            source = self.source_paths[task_id]
            sample_id = _sample_id("real", task_id)
            canonical_path = self.release / "images" / f"{sample_id}.jpg"
            self.save_canonical(source, canonical_path)
            input_row = self._make_common_input(
                "real",
                task_id,
                self.canonical_fields(
                    source,
                    canonical_path,
                    "balanced250_v1_reencode",
                ),
                eligibility_rank,
                selected_real.get(task_id),
                task_id in selected_real,
            )
            input_row.update(
                {
                    "matched_source_task_id": task_id,
                    "matched_source_raw_path": self.rel(source),
                    "matched_source_raw_sha256": sha256_file(source),
                    "gt_mask_kind": "all_zero",
                    "gt_mask_path": None,
                    "gt_mask_sha256": None,
                    "gt_positive_pixels": 0,
                    "support_semantics": "authentic_all_zero",
                    "edit_region_xyxy": row["edit_region_xyxy"],
                    "context_region_xyxy": row["context_region_xyxy"],
                    "source_release_sample_id": row["sample_id"],
                }
            )
            built.append(input_row)

        for condition in FORGED:
            source_row = self.selected_rows[condition][0]
            task_id = source_row["task_id"]
            normalized = _norm(task_id)
            source = self.source_paths[normalized]
            if condition == "local_mouse":
                old = self.old_forged[task_id]
                sample_id = _sample_id(condition, normalized)
                raw_path = self.root / old["raw_path"]
                canonical_path = self.release / "images" / f"{sample_id}.jpg"
                self.save_canonical(raw_path, canonical_path)
                mask_info = self.exact_mask(
                    source,
                    raw_path,
                    self.release / "masks" / f"{sample_id}.png",
                )
                canonical = self.canonical_fields(
                    raw_path,
                    canonical_path,
                    "balanced250_v1_reencode",
                )
                pair = self.old_pairs[task_id]
                row = self._make_common_input(
                    condition,
                    task_id,
                    canonical,
                    self.eligible_rows[condition].index(source_row),
                    0,
                    True,
                )
                row.update(
                    {
                        "matched_source_task_id": normalized,
                        "matched_source_raw_path": self.rel(source),
                        "matched_source_raw_sha256": sha256_file(source),
                        "gt_mask_kind": "exact_diff",
                        "gt_mask_path": mask_info["path"],
                        "gt_mask_sha256": mask_info["sha256"],
                        "gt_positive_pixels": mask_info["positive"],
                        "gt_bbox_xyxy": mask_info["bbox"],
                        "gt_pixels_outside_context": mask_info["outside"],
                        "gt_fraction": mask_info["fraction"],
                        "support_semantics": (
                            "decoded_source_vs_local_forged_exact_diff"
                        ),
                        "edit_region_xyxy": old["edit_region_xyxy"],
                        "context_region_xyxy": old["context_region_xyxy"],
                        "source_release_sample_id": self.old_real[task_id][
                            "sample_id"
                        ],
                        "forged_source_release_sample_id": old["sample_id"],
                        "local_selection_method": "mouse_human_review_good",
                    }
                )
            elif condition in {"local_cat", "local_trash_can"}:
                delivered = self.local_delivered[condition]
                raw = self.root / delivered["image"]
                sample_id = _sample_id(condition, normalized)
                canonical_path = self.release / "images" / f"{sample_id}.jpg"
                self.save_canonical(raw, canonical_path)
                mask_info = self.exact_mask(
                    source,
                    raw,
                    self.release / "masks" / f"{sample_id}.png",
                )
                row = self._make_common_input(
                    condition,
                    task_id,
                    self.canonical_fields(
                        raw,
                        canonical_path,
                        "balanced250_v1_reencode",
                    ),
                    0,
                    0,
                    True,
                )
                prefix = "cat" if condition == "local_cat" else "trash"
                row.update(
                    {
                        "matched_source_task_id": normalized,
                        "matched_source_raw_path": self.rel(source),
                        "matched_source_raw_sha256": sha256_file(source),
                        "gt_mask_kind": "exact_diff",
                        "gt_mask_path": mask_info["path"],
                        "gt_mask_sha256": mask_info["sha256"],
                        "gt_positive_pixels": mask_info["positive"],
                        "gt_bbox_xyxy": mask_info["bbox"],
                        "gt_pixels_outside_context": mask_info["outside"],
                        "gt_fraction": mask_info["fraction"],
                        "support_semantics": (
                            "decoded_source_vs_local_forged_exact_diff"
                        ),
                        "edit_region_xyxy": [1, 1, 3, 3],
                        "context_region_xyxy": mask_info["context"],
                        "local_selection_method": "selected",
                        "local_materialized_manifest_path": self.rel(
                            self.source_files[f"{prefix}_materialized"]
                        ),
                        "local_materialized_candidate_path": delivered[
                            "source_image"
                        ],
                    }
                )
            else:
                run = self.whole_runs[condition][task_id]
                raw = self.root / run["output_image"]
                sample_id = _sample_id(condition, normalized)
                canonical_path = self.release / "images" / f"{sample_id}.jpg"
                self.save_canonical(raw, canonical_path)
                row = self._make_common_input(
                    condition,
                    task_id,
                    self.canonical_fields(
                        raw,
                        canonical_path,
                        "balanced250_v1_reencode",
                    ),
                    self.eligible_rows[condition].index(source_row),
                    0,
                    True,
                )
                qc = "usable" if condition == "fullframe_trash_can" else "not_reviewed"
                row.update(
                    {
                        "matched_source_task_id": normalized,
                        "matched_source_raw_path": self.rel(source),
                        "matched_source_raw_sha256": sha256_file(source),
                        "gt_mask_kind": "not_applicable",
                        "gt_mask_path": None,
                        "gt_mask_sha256": None,
                        "gt_positive_pixels": None,
                        "support_semantics": (
                            "full_frame_conditional_edit_no_localization_target"
                        ),
                        "conditioning_box_xyxy": [1, 1, 3, 3],
                        "context_region_xyxy": [0, 0, 5, 5],
                        "generation_manifest_path": self.rel(
                            self.source_files[f"{condition}_run"]
                        ),
                        "generation_manifest_latest_row_index": list(
                            self.whole_runs[condition]
                        ).index(task_id),
                        "generation_model": run["model"],
                        "generation_service_model": run["service_model"],
                        "generation_seed": run["seed"],
                        "generation_steps": run["steps"],
                        "generation_guidance_scale": run["guidance_scale"],
                        "generation_bot_task": run["bot_task"],
                        "fullframe_semantic_qc_status": qc,
                        "fullframe_semantic_qc_categories": [],
                        "fullframe_semantic_qc_reason": None,
                    }
                )
            built.append(row)

        condition_index = {condition: index for index, condition in enumerate(CONDITIONS)}
        cluster_counts = {
            condition: Counter(
                row["matched_source_raw_sha256"]
                for row in built
                if row["condition"] == condition
            )
            for condition in CONDITIONS
        }
        for row in built:
            cluster = row["matched_source_raw_sha256"]
            size = cluster_counts[row["condition"]][cluster]
            row["source_content_cluster"] = cluster
            row["source_content_cluster_size_within_condition"] = size
            row["source_content_is_duplicated_within_condition"] = size > 1
        built.sort(
            key=lambda row: (
                condition_index[row["condition"]],
                row["eligibility_rank"]
                if row["condition"] == "real"
                else row["selection_rank"],
                row["normalized_task_id"],
            )
        )
        for rank, row in enumerate(built):
            row["rank"] = rank
        self.inputs = built

    def _content_clusters(self, rows: list[dict]) -> dict:
        grouped: dict[str, list[str]] = {}
        for row in rows:
            grouped.setdefault(row["matched_source_raw_sha256"], []).append(
                row["normalized_task_id"]
            )
        duplicate = [
            {
                "source_sha256": digest,
                "normalized_task_ids": sorted(ids),
            }
            for digest, ids in grouped.items()
            if len(ids) > 1
        ]
        duplicate.sort(key=lambda row: row["source_sha256"])
        return {
            "rows": len(rows),
            "unique_source_sha256": len(grouped),
            "duplicate_cluster_count": len(duplicate),
            "duplicate_row_count": sum(
                len(row["normalized_task_ids"]) - 1 for row in duplicate
            ),
            "duplicate_clusters": duplicate,
        }

    def _build_release_ledgers(self) -> None:
        by_sample = {row["sample_id"]: row for row in self.inputs}
        for condition in CONDITIONS:
            rows = [
                row
                for row in self.inputs
                if row["condition"] == condition and row["panel"]
            ]
            rows.sort(key=lambda row: row["selection_rank"])
            row = rows[0]
            self.panel.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "dataset_id": DATASET_ID,
                    "panel_rank": len(self.panel),
                    "condition": condition,
                    "condition_rank": 0,
                    "sample_id": row["sample_id"],
                    "input_rank": row["rank"],
                    "task_id": row["task_id"],
                    "normalized_task_id": row["normalized_task_id"],
                    "label": row["label"],
                    "domain": row["domain"],
                    "kind": row["kind"],
                    "condition_family": row["condition_family"],
                    "manipulation_scope": row["manipulation_scope"],
                    "selection_key": row["selection_key"],
                    "eligible_set_sha256": row["eligible_set_sha256"],
                    "canonical_path": row["canonical_path"],
                    "canonical_sha256": row["canonical_sha256"],
                    "canonical_bytes": row["canonical_bytes"],
                    "width": row["width"],
                    "height": row["height"],
                    "source_content_cluster": row["source_content_cluster"],
                    "source_content_cluster_size_within_condition": row[
                        "source_content_cluster_size_within_condition"
                    ],
                    "gt_mask_kind": row["gt_mask_kind"],
                    "gt_mask_path": row["gt_mask_path"],
                    "gt_mask_sha256": row["gt_mask_sha256"],
                    "gt_positive_pixels": row["gt_positive_pixels"],
                }
            )
        for condition in FORGED:
            forged = next(
                row for row in self.inputs if row["condition"] == condition
            )
            real = next(
                row
                for row in self.inputs
                if row["condition"] == "real"
                and row["normalized_task_id"] == forged["normalized_task_id"]
            )
            ref_fields = (
                "canonical_path",
                "canonical_sha256",
                "canonical_bytes",
                "width",
                "height",
            )
            self.pairs.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "dataset_id": DATASET_ID,
                    "rank": len(self.pairs),
                    "pair_id": _pair_id(
                        condition,
                        forged["normalized_task_id"],
                    ),
                    "condition": condition,
                    "pair_rank": len(self.pairs),
                    "condition_pair_rank": 0,
                    "normalized_task_id": forged["normalized_task_id"],
                    "domain": forged["domain"],
                    "selection_key": forged["selection_key"],
                    "eligible_set_sha256": forged["eligible_set_sha256"],
                    "real_sample_id": real["sample_id"],
                    "forged_sample_id": forged["sample_id"],
                    "real": {field: real[field] for field in ref_fields},
                    "forged": {field: forged[field] for field in ref_fields},
                    "source_raw_path": real["raw_path"],
                    "source_raw_sha256": real["raw_sha256"],
                    "source_content_cluster": real["raw_sha256"],
                    "source_content_cluster_size_within_condition": forged[
                        "source_content_cluster_size_within_condition"
                    ],
                    "comparison_design": "source_matched_secondary",
                }
            )
        _write_jsonl(self.release / "inputs.jsonl", self.inputs)
        _write_jsonl(self.release / "panel.jsonl", self.panel)
        _write_jsonl(self.release / "source_pairs.jsonl", self.pairs)

    def _build_manifest(self) -> None:
        source_contracts: dict[str, dict] = {}
        jsonl_names = {
            "mouse_inputs",
            "mouse_pairs",
            "cat_materialized",
            "trash_materialized",
            "fullframe_mouse_tasks",
            "fullframe_mouse_run",
            "fullframe_cat_tasks",
            "fullframe_cat_run",
            "fullframe_trash_can_tasks",
            "fullframe_trash_can_run",
        }
        for name, path in self.source_files.items():
            contract = self.file_record(path)
            if name in jsonl_names:
                contract["rows"] = len(
                    [line for line in path.read_text().splitlines() if line]
                )
            source_contracts[name] = contract
        condition_summaries: dict[str, dict] = {}
        for condition in CONDITIONS:
            eligible = self.eligible_rows[condition]
            selected = self.selected_rows[condition]
            inputs = [
                row for row in self.inputs if row["condition"] == condition
            ]
            selected_ids = [_norm(row["task_id"]) for row in selected]
            summary = {
                "eligible_rows": len(eligible),
                "expected_eligible_rows": len(eligible),
                "eligible_set_sha256": self.eligibility_hash(condition),
                "cache_rows": len(inputs),
                "panel_rows": 1,
                "eligible_normalized_task_ids_sha256": _id_hash(
                    sorted(_norm(row["task_id"]) for row in eligible)
                ),
                "selected_normalized_task_ids_sha256": _id_hash(selected_ids),
                "selection_key_sha256": _id_hash(
                    [
                        _selection_key(condition, task_id)
                        for task_id in selected_ids
                    ]
                ),
                "domains": dict(
                    sorted(Counter(row["domain"] for row in inputs).items())
                ),
                "source_content": self._content_clusters(inputs),
            }
            if condition in LOCAL:
                summary.update(
                    {
                        "gt_positive_pixels": sum(
                            row["gt_positive_pixels"] for row in inputs
                        ),
                        "gt_pixels_outside_context": sum(
                            row["gt_pixels_outside_context"] for row in inputs
                        ),
                        "rows_with_gt_outside_context": sum(
                            row["gt_pixels_outside_context"] > 0 for row in inputs
                        ),
                    }
                )
            if condition == "fullframe_trash_can":
                summary["semantic_qc"] = {"usable": 1}
            condition_summaries[condition] = summary
        ledgers = {}
        for name, filename in (
            ("inputs", "inputs.jsonl"),
            ("panel", "panel.jsonl"),
            ("source_pairs", "source_pairs.jsonl"),
        ):
            path = self.release / filename
            ledgers[name] = {
                "path": self.rel(path),
                "rows": len(path.read_text().splitlines()),
                "sha256": sha256_file(path),
            }
        deterministic = {
            "schema_version": SCHEMA_VERSION,
            "dataset_id": DATASET_ID,
            "design": {
                "primary": "independent_seven_condition_panel",
                "secondary": "source_matched_six_condition_pairs",
                "panel_conditions": list(CONDITIONS),
                "panel_rows_per_condition": 1,
                "real_cache_rows": 2,
                "forged_cache_rows_per_condition": 1,
                "self_contained_canonical_inputs": True,
                "release_canonical_images": 8,
                "release_local_masks": 3,
            },
            "selection": {
                "score_blind": True,
                "key": (
                    "sha256(dataset_id + NUL + condition + NUL + "
                    "normalized_task_id)"
                ),
                "collision_policy": "reject duplicate selection keys",
                "real_policy": (
                    "rank all eligible real tasks by key; retain the first task "
                    "per raw_sha256 until 250 content-unique panel rows are selected"
                ),
                "forged_policy": "first 250 eligible unique normalized task IDs",
                "semantic_qc_used_for_selection": False,
            },
            "canonicalization": {
                "decode": "Pillow ImageOps.exif_transpose then RGB",
                "format": "JPEG",
                "quality": 95,
                "subsampling": 0,
                "optimize": False,
                "metadata": "stripped",
                "resize": False,
                "all_inputs_reencoded_from_frozen_raw": True,
                "encoder": {
                    "pillow": PIL.__version__,
                    "libjpeg": features.version_codec("jpg"),
                },
            },
            "localization": {
                "local_conditions": sorted(LOCAL),
                "mask_space": "decoded_pre_canonicalization_rgb",
                "mask_rule": "max_abs_rgb_difference_gt_0",
                "context_box_is_not_ground_truth": True,
                "fullframe_gt_mask_kind": "not_applicable",
            },
            "ledgers": ledgers,
            "source_contracts": source_contracts,
            "conditions": condition_summaries,
            "fullframe_semantics": {
                "label": "conditional_full_frame_edit",
                "fully_synthetic": False,
                "trash_primary_qc_summary": {
                    "total": 1,
                    "usable": 1,
                    "failed": 0,
                },
            },
        }
        self.manifest = {
            **deterministic,
            "contract_sha256": hashlib.sha256(
                stable_json(deterministic).encode()
            ).hexdigest(),
            "created_at": "2026-07-26T00:00:00+00:00",
            "repo_root": str(self.root),
            "output_dir": self.rel(self.release),
            "inputs_rows": len(self.inputs),
            "panel_rows": len(self.panel),
            "source_pair_rows": len(self.pairs),
            "new_canonical_images": self.spec.inputs,
            "new_local_masks": self.spec.local_masks,
            "status": "complete",
        }
        _write_json(self.release / "manifest.json", self.manifest)

    def _build(self) -> None:
        self._build_mouse()
        self._build_local_and_whole_sources()
        self._build_inputs()
        self._build_release_ledgers()
        self._build_manifest()

    def reseal_ledger(self, name: str, rows: list[dict]) -> None:
        filename = {
            "inputs": "inputs.jsonl",
            "panel": "panel.jsonl",
            "source_pairs": "source_pairs.jsonl",
        }[name]
        path = self.release / filename
        _write_jsonl(path, rows)
        self.manifest["ledgers"][name]["sha256"] = sha256_file(path)
        self.manifest["ledgers"][name]["rows"] = len(rows)
        self.reseal_manifest()

    def reseal_manifest(self) -> None:
        nondeterministic = {
            "contract_sha256",
            "created_at",
            "repo_root",
            "output_dir",
            "inputs_rows",
            "panel_rows",
            "source_pair_rows",
            "new_canonical_images",
            "new_local_masks",
            "status",
        }
        deterministic = {
            key: value
            for key, value in self.manifest.items()
            if key not in nondeterministic
        }
        self.manifest["contract_sha256"] = hashlib.sha256(
            stable_json(deterministic).encode()
        ).hexdigest()
        _write_json(self.release / "manifest.json", self.manifest)


class ValidateBalanced250CanonicalTest(unittest.TestCase):
    def test_valid_tiny_release(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = TinyRelease(Path(temporary))

            summary = validate_release(
                fixture.release,
                repo_root=fixture.root,
                spec=fixture.spec,
            )

            self.assertEqual(summary["status"], "valid")
            self.assertEqual(summary["counts"]["inputs"], 8)
            self.assertEqual(summary["counts"]["panel"], 7)
            self.assertEqual(summary["counts"]["source_pairs"], 6)
            self.assertEqual(summary["counts"]["new_canonical_images"], 8)
            self.assertEqual(summary["counts"]["new_local_masks"], 3)

    def test_rejects_resealed_selection_key_tamper(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = TinyRelease(Path(temporary))
            inputs = [
                json.loads(line)
                for line in (fixture.release / "inputs.jsonl").read_text().splitlines()
            ]
            inputs[2]["selection_key"] = "0" * 64
            fixture.reseal_ledger("inputs", inputs)

            with self.assertRaisesRegex(ValidationError, "selection_key mismatch"):
                validate_release(
                    fixture.release,
                    repo_root=fixture.root,
                    spec=fixture.spec,
                )

    def test_rejects_path_traversal_even_when_ledger_is_resealed(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = TinyRelease(Path(temporary))
            inputs = [
                json.loads(line)
                for line in (fixture.release / "inputs.jsonl").read_text().splitlines()
            ]
            inputs[0]["raw_path"] = "../escape.png"
            fixture.reseal_ledger("inputs", inputs)

            with self.assertRaisesRegex(ValidationError, "travers"):
                validate_release(
                    fixture.release,
                    repo_root=fixture.root,
                    spec=fixture.spec,
                )

    def test_rejects_source_ledger_hash_drift(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = TinyRelease(Path(temporary))
            with fixture.source_files["cat_selection"].open(
                "a",
                encoding="utf-8",
            ) as handle:
                handle.write(" ")

            with self.assertRaisesRegex(ValidationError, "SHA-256 mismatch"):
                validate_release(
                    fixture.release,
                    repo_root=fixture.root,
                    spec=fixture.spec,
                )

    def test_rejects_missing_or_extra_release_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = TinyRelease(Path(temporary))
            (fixture.release / "unexpected.bin").write_bytes(b"x")

            with self.assertRaisesRegex(ValidationError, "inventory mismatch"):
                validate_release(
                    fixture.release,
                    repo_root=fixture.root,
                    spec=fixture.spec,
                )

        with tempfile.TemporaryDirectory() as temporary:
            fixture = TinyRelease(Path(temporary))
            next((fixture.release / "images").glob("*.jpg")).unlink()

            with self.assertRaisesRegex(ValidationError, "missing file"):
                validate_release(
                    fixture.release,
                    repo_root=fixture.root,
                    spec=fixture.spec,
                )

    def test_rejects_mask_or_wholeframe_semantic_drift(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = TinyRelease(Path(temporary))
            inputs = [
                json.loads(line)
                for line in (fixture.release / "inputs.jsonl").read_text().splitlines()
            ]
            full = next(
                row for row in inputs if row["condition"] == "fullframe_cat"
            )
            full["gt_mask_kind"] = "exact_diff"
            fixture.reseal_ledger("inputs", inputs)

            with self.assertRaisesRegex(ValidationError, "full-frame GT"):
                validate_release(
                    fixture.release,
                    repo_root=fixture.root,
                    spec=fixture.spec,
                )

    def test_rejects_jpeg_comment_even_when_hash_is_resealed(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = TinyRelease(Path(temporary))
            inputs = [
                json.loads(line)
                for line in (fixture.release / "inputs.jsonl").read_text().splitlines()
            ]
            row = inputs[0]
            canonical = fixture.root / row["canonical_path"]
            with Image.open(canonical) as opened:
                image = opened.convert("RGB")
            image.save(
                canonical,
                format="JPEG",
                quality=95,
                subsampling=0,
                optimize=False,
                comment=b"forbidden",
            )
            row["canonical_sha256"] = sha256_file(canonical)
            row["canonical_bytes"] = canonical.stat().st_size
            fixture.reseal_ledger("inputs", inputs)

            with self.assertRaisesRegex(ValidationError, "forbidden JPEG metadata"):
                validate_release(
                    fixture.release,
                    repo_root=fixture.root,
                    spec=fixture.spec,
                )


if __name__ == "__main__":
    unittest.main()
