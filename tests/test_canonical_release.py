from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from eval.opensource.canonical_release import (
    BALANCED_CONDITIONS,
    BALANCED_CONTRACT_SHA256,
    BALANCED_SCHEMA,
    BALANCED_RELEASE_KIND,
    LEGACY_MOUSE_RELEASE_KIND,
    LOCALIZATION_CONDITIONS,
    MOUSE_DATASET_ID,
    MOUSE_SCHEMA,
    CanonicalReleaseError,
    Capability,
    SelectionSpec,
    load_canonical_release,
    load_ground_truth,
    select_inputs,
    _validate_canonical_path,
)
from eval.opensource.common import sha256_file, stable_json


REPO_ROOT = Path(__file__).resolve().parents[1]
BALANCED_MANIFEST = Path(
    "outputs/opensource/balanced250_v1/manifest.json"
)
MOUSE_MANIFEST = Path(
    "outputs/opensource/mouse_canonical_v1/manifest.json"
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(f"{stable_json(row)}\n" for row in rows),
        encoding="utf-8",
    )


def _sample_id(task_id: str, kind: str) -> str:
    value = f"{MOUSE_DATASET_ID}\0{task_id}\0{kind}".encode()
    return hashlib.sha256(value).hexdigest()[:24]


def _write_mouse_fixture(
    root: Path,
    *,
    mutate_inputs=None,
    mutate_manifest=None,
) -> Path:
    release_dir = root / "outputs" / "mouse"
    inputs: list[dict] = []
    pairs: list[dict] = []
    for pair_rank in range(275):
        task_id = f"lodging_{pair_rank:03d}_slot_001"
        variants: dict[str, dict] = {}
        mask_id = _sample_id(task_id, "mask")
        mask_path = f"outputs/mouse/masks/{mask_id}.png"
        mask_sha = hashlib.sha256(f"mask-{pair_rank}".encode()).hexdigest()
        for kind in ("real", "forged"):
            sample_id = _sample_id(task_id, kind)
            canonical_sha = hashlib.sha256(sample_id.encode()).hexdigest()
            row = {
                "schema_version": MOUSE_SCHEMA,
                "dataset_id": MOUSE_DATASET_ID,
                "rank": len(inputs),
                "pair_rank": pair_rank,
                "sample_id": sample_id,
                "task_id": task_id,
                "domain": "lodging",
                "candidate": "mouse",
                "kind": kind,
                "label": 0 if kind == "real" else 1,
                "raw_path": f"raw/{task_id}-{kind}.png",
                "raw_sha256": hashlib.sha256(
                    f"raw-{task_id}-{kind}".encode()
                ).hexdigest(),
                "canonical_path": f"outputs/mouse/images/{sample_id}.jpg",
                "canonical_sha256": canonical_sha,
                "canonical_bytes": 1,
                "width": 4,
                "height": 3,
                "edit_region_xyxy": [1, 1, 2, 2],
                "context_region_xyxy": [0, 0, 3, 3],
                "gt_mask_kind": "all_zero" if kind == "real" else "exact_diff",
                "gt_mask_path": None if kind == "real" else mask_path,
                "gt_mask_sha256": None if kind == "real" else mask_sha,
                "gt_positive_pixels": 0 if kind == "real" else 1,
            }
            inputs.append(row)
            variants[kind] = {
                key: row[key]
                for key in (
                    "kind",
                    "label",
                    "sample_id",
                    "raw_path",
                    "raw_sha256",
                    "canonical_path",
                    "canonical_sha256",
                    "canonical_bytes",
                )
            }
        pairs.append(
            {
                "schema_version": MOUSE_SCHEMA,
                "dataset_id": MOUSE_DATASET_ID,
                "pair_rank": pair_rank,
                "task_id": task_id,
                "domain": "lodging",
                "candidate": "mouse",
                "width": 4,
                "height": 3,
                "edit_region_xyxy": [1, 1, 2, 2],
                "context_region_xyxy": [0, 0, 3, 3],
                "gt_mask_path": mask_path,
                "gt_mask_sha256": mask_sha,
                "gt_positive_pixels": 1,
                "gt_fraction": 1 / 12,
                "gt_bbox_xyxy": [1, 1, 2, 2],
                "gt_pixels_outside_context": 0,
                "real": variants["real"],
                "forged": variants["forged"],
            }
        )
    if mutate_inputs is not None:
        mutate_inputs(inputs, pairs)
    inputs_path = release_dir / "inputs.jsonl"
    pairs_path = release_dir / "pairs.jsonl"
    _write_jsonl(inputs_path, inputs)
    _write_jsonl(pairs_path, pairs)
    deterministic = {
        "schema_version": MOUSE_SCHEMA,
        "dataset_id": MOUSE_DATASET_ID,
        "source_review_sha256": "a" * 64,
        "source_order_manifest_sha256": "b" * 64,
        "jpeg": {
            "quality": 95,
            "subsampling": 0,
            "optimize": False,
            "metadata": "stripped",
            "encoder": {"pillow": "test", "libjpeg": "test"},
        },
        "gt_mask": {
            "space": "decoded_pre_canonicalization_rgb",
            "rule": "max_abs_rgb_difference_gt_threshold",
            "threshold": 0,
        },
        "inputs_sha256": sha256_file(inputs_path),
        "pairs_sha256": sha256_file(pairs_path),
    }
    manifest = {
        **deterministic,
        "contract_sha256": hashlib.sha256(
            stable_json(deterministic).encode()
        ).hexdigest(),
        "created_at": "2026-07-26T00:00:00+00:00",
        "repo_root": str(root.resolve()),
        "source_review": "source/review.json",
        "source_order_manifest": "source/order.json",
        "inputs_path": "outputs/mouse/inputs.jsonl",
        "pairs_path": "outputs/mouse/pairs.jsonl",
        "pairs": 275,
        "images": 550,
        "domains": {"lodging": 275},
        "gt": {
            "positive_pixels": 275,
            "image_pixels": 3300,
            "mean_fraction": 1 / 12,
            "pixels_outside_context": 0,
        },
    }
    if mutate_manifest is not None:
        mutate_manifest(manifest)
    manifest_path = release_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest_path.relative_to(root)


def test_capabilities_freeze_task_and_condition_scope() -> None:
    assert Capability.WHOLE_IMAGE_T1.conditions == BALANCED_CONDITIONS
    assert Capability.WHOLE_IMAGE_T1.valid_for_t1 is True
    assert Capability.WHOLE_IMAGE_T1.valid_for_t2 is False
    assert Capability.LOCAL_T1_T2.conditions == BALANCED_CONDITIONS
    assert Capability.LOCAL_T1_T2.valid_for_t1 is True
    assert Capability.LOCAL_T1_T2.valid_for_t2 is True
    assert Capability.LOCAL_T2_ONLY.conditions == LOCALIZATION_CONDITIONS
    assert Capability.LOCAL_T2_ONLY.valid_for_t1 is False
    assert Capability.LOCAL_T2_ONLY.valid_for_t2 is True


def test_loads_frozen_balanced_release_and_all_ledgers() -> None:
    release = load_canonical_release(
        REPO_ROOT,
        BALANCED_MANIFEST,
        verify_files=False,
    )
    assert release.release_kind == BALANCED_RELEASE_KIND
    assert release.contract_sha256 == BALANCED_CONTRACT_SHA256
    assert release.manifest_sha256 == sha256_file(REPO_ROOT / BALANCED_MANIFEST)
    assert release.inputs_ledger.rows == len(release.inputs) == 1775
    assert release.panel_ledger is not None
    assert release.panel_ledger.rows == len(release.panel) == 1750
    assert release.source_pairs_ledger is not None
    assert release.source_pairs_ledger.rows == len(release.source_pairs) == 1500
    assert release.legacy_pairs_ledger is None
    assert release.legacy_pairs == ()


def test_balanced_capability_selection_is_1775_or_1025() -> None:
    release = load_canonical_release(
        REPO_ROOT,
        BALANCED_MANIFEST,
        verify_files=False,
    )
    whole = select_inputs(
        release,
        SelectionSpec(capability=Capability.WHOLE_IMAGE_T1),
    )
    joint = select_inputs(
        release,
        SelectionSpec(capability=Capability.LOCAL_T1_T2),
    )
    t2_only = select_inputs(
        release,
        SelectionSpec(capability=Capability.LOCAL_T2_ONLY),
    )
    assert len(whole) == len(joint) == 1775
    assert len(t2_only) == 1025
    assert {row["condition"] for row in t2_only} == set(
        LOCALIZATION_CONDITIONS
    )
    assert all(row["gt_mask_kind"] != "not_applicable" for row in t2_only)


def test_balanced_condition_limit_and_single_sample_are_explicit() -> None:
    release = load_canonical_release(
        REPO_ROOT,
        BALANCED_MANIFEST,
        verify_files=False,
    )
    smoke = select_inputs(
        release,
        SelectionSpec(
            capability=Capability.LOCAL_T1_T2,
            conditions=("real", "local_cat"),
            per_condition_limit=2,
        ),
    )
    assert [row["condition"] for row in smoke] == [
        "real",
        "real",
        "local_cat",
        "local_cat",
    ]
    assert all(row["panel"] is True for row in smoke)
    assert [
        row["selection_rank"]
        for row in smoke
        if row["condition"] == "real"
    ] == [0, 1]

    non_panel_real = next(
        row
        for row in release.inputs
        if row["condition"] == "real" and row["panel"] is False
    )
    reordered = replace(
        release,
        inputs=(
            non_panel_real,
            *(
                row
                for row in release.inputs
                if row["sample_id"] != non_panel_real["sample_id"]
            ),
        ),
    )
    reordered_smoke = select_inputs(
        reordered,
        SelectionSpec(
            capability=Capability.WHOLE_IMAGE_T1,
            conditions=("real",),
            per_condition_limit=2,
        ),
    )
    assert all(row["panel"] is True for row in reordered_smoke)
    assert [row["selection_rank"] for row in reordered_smoke] == [0, 1]
    one = select_inputs(
        release,
        SelectionSpec(
            capability=Capability.WHOLE_IMAGE_T1,
            sample_id=release.inputs[-1]["sample_id"],
        ),
    )
    assert one == [release.inputs[-1]]
    with pytest.raises(CanonicalReleaseError, match="pair_limit is unsupported"):
        select_inputs(
            release,
            SelectionSpec(
                capability=Capability.WHOLE_IMAGE_T1,
                pair_limit=1,
            ),
        )
    with pytest.raises(CanonicalReleaseError, match="outside"):
        select_inputs(
            release,
            SelectionSpec(
                capability=Capability.LOCAL_T2_ONLY,
                conditions=("fullframe_mouse",),
            ),
        )


def test_loads_legacy_mouse_and_supplements_condition_scope() -> None:
    release = load_canonical_release(
        REPO_ROOT,
        MOUSE_MANIFEST,
        verify_files=False,
    )
    assert release.release_kind == LEGACY_MOUSE_RELEASE_KIND
    assert len(release.inputs) == 550
    assert release.legacy_pairs_ledger is not None
    assert len(release.legacy_pairs) == 275
    assert release.panel == release.source_pairs == ()
    assert release.inputs[0]["condition"] == "real"
    assert release.inputs[0]["manipulation_scope"] == "authentic"
    assert release.inputs[1]["condition"] == "local_mouse"
    assert release.inputs[1]["manipulation_scope"] == "local_insertion"
    selected = select_inputs(
        release,
        SelectionSpec(
            capability=Capability.LOCAL_T1_T2,
            pair_limit=2,
        ),
    )
    assert len(selected) == 4
    assert {row["pair_rank"] for row in selected} == {0, 1}
    with pytest.raises(CanonicalReleaseError, match="unsupported for legacy"):
        select_inputs(
            release,
            SelectionSpec(
                capability=Capability.LOCAL_T1_T2,
                conditions=("local_mouse",),
            ),
        )


def test_mouse_loader_rejects_rank_and_path_drift(tmp_path: Path) -> None:
    def bad_rank(inputs, _pairs) -> None:
        inputs[10]["rank"] = 99

    manifest = _write_mouse_fixture(tmp_path, mutate_inputs=bad_rank)
    with pytest.raises(CanonicalReleaseError, match="rank is not contiguous"):
        load_canonical_release(tmp_path, manifest, verify_files=False)

    other_root = tmp_path / "other"

    def bad_path(inputs, _pairs) -> None:
        inputs[0]["canonical_path"] = "../escape.jpg"

    manifest = _write_mouse_fixture(other_root, mutate_inputs=bad_path)
    with pytest.raises(CanonicalReleaseError, match="traversing"):
        load_canonical_release(other_root, manifest, verify_files=False)


def test_mouse_loader_rejects_sample_and_gt_state_drift(tmp_path: Path) -> None:
    def bad_sample(inputs, _pairs) -> None:
        inputs[0]["sample_id"] = "f" * 24

    manifest = _write_mouse_fixture(tmp_path / "sample", mutate_inputs=bad_sample)
    with pytest.raises(CanonicalReleaseError, match="sample_id|sample ID"):
        load_canonical_release(
            tmp_path / "sample",
            manifest,
            verify_files=False,
        )

    def bad_gt(inputs, pairs) -> None:
        forged = inputs[1]
        forged["gt_mask_kind"] = "not_applicable"
        forged["gt_mask_path"] = None
        forged["gt_mask_sha256"] = None
        forged["gt_positive_pixels"] = None
        for key in ("gt_mask_path", "gt_mask_sha256", "gt_positive_pixels"):
            pairs[0][key] = forged[key]

    manifest = _write_mouse_fixture(tmp_path / "gt", mutate_inputs=bad_gt)
    with pytest.raises(CanonicalReleaseError, match="not-applicable"):
        load_canonical_release(tmp_path / "gt", manifest, verify_files=False)


def test_mouse_loader_rejects_ledger_hash_drift_even_without_file_checks(
    tmp_path: Path,
) -> None:
    manifest = _write_mouse_fixture(tmp_path)
    inputs_path = tmp_path / "outputs" / "mouse" / "inputs.jsonl"
    inputs_path.write_text(
        inputs_path.read_text(encoding="utf-8") + "{}\n",
        encoding="utf-8",
    )
    with pytest.raises(CanonicalReleaseError, match="SHA-256 mismatch"):
        load_canonical_release(tmp_path, manifest, verify_files=False)


def test_load_ground_truth_preserves_three_states(tmp_path: Path) -> None:
    mask_path = tmp_path / "masks" / "exact.png"
    mask_path.parent.mkdir(parents=True)
    pixels = np.asarray([[0, 255], [255, 0]], dtype=np.uint8)
    Image.fromarray(pixels, mode="L").save(mask_path, format="PNG")
    base = {
        "sample_id": "a" * 24,
        "width": 2,
        "height": 2,
    }
    exact = {
        **base,
        "kind": "forged",
        "label": 1,
        "condition": "local_mouse",
        "manipulation_scope": "local_insertion",
        "gt_mask_kind": "exact_diff",
        "gt_mask_path": "masks/exact.png",
        "gt_mask_sha256": sha256_file(mask_path),
        "gt_positive_pixels": 2,
        "gt_fraction": 0.5,
        "gt_bbox_xyxy": [0, 0, 2, 2],
    }
    loaded = load_ground_truth(exact, tmp_path)
    assert loaded is not None
    assert loaded.dtype == np.bool_
    assert np.array_equal(loaded, pixels > 0)
    all_zero = {
        **base,
        "kind": "real",
        "label": 0,
        "condition": "real",
        "manipulation_scope": "authentic",
        "gt_mask_kind": "all_zero",
        "gt_mask_path": None,
        "gt_mask_sha256": None,
        "gt_positive_pixels": 0,
    }
    assert not load_ground_truth(all_zero, tmp_path).any()
    not_applicable = {
        **base,
        "kind": "forged",
        "label": 1,
        "condition": "fullframe_mouse",
        "manipulation_scope": "conditional_full_frame_edit",
        "gt_mask_kind": "not_applicable",
        "gt_mask_path": None,
        "gt_mask_sha256": None,
        "gt_positive_pixels": None,
    }
    assert load_ground_truth(not_applicable, tmp_path) is None
    changed = copy.deepcopy(exact)
    changed["gt_mask_sha256"] = "0" * 64
    with pytest.raises(CanonicalReleaseError, match="SHA-256 mismatch"):
        load_ground_truth(changed, tmp_path)


def test_canonical_jpeg_contract_rejects_non_q95_subsampling_and_metadata(
    tmp_path: Path,
) -> None:
    sample_id = "a" * 24
    release_dir = tmp_path / "release"
    image_path = release_dir / "images" / f"{sample_id}.jpg"
    image_path.parent.mkdir(parents=True)
    pixels = np.zeros((8, 8, 3), dtype=np.uint8)

    def write_image(**kwargs) -> dict:
        Image.fromarray(pixels, mode="RGB").save(
            image_path,
            format="JPEG",
            optimize=False,
            **kwargs,
        )
        return {
            "schema_version": BALANCED_SCHEMA,
            "sample_id": sample_id,
            "canonical_path": f"release/images/{sample_id}.jpg",
            "canonical_sha256": sha256_file(image_path),
            "canonical_bytes": image_path.stat().st_size,
            "width": 8,
            "height": 8,
        }

    clean = write_image(quality=95, subsampling=0)
    _validate_canonical_path(
        clean,
        repo_root=tmp_path.resolve(),
        release_dir=release_dir.resolve(),
        label="fixture",
        verify_file=True,
    )

    q90 = write_image(quality=90, subsampling=0)
    with pytest.raises(CanonicalReleaseError, match="quality 95"):
        _validate_canonical_path(
            q90,
            repo_root=tmp_path.resolve(),
            release_dir=release_dir.resolve(),
            label="fixture",
            verify_file=True,
        )

    subsampled = write_image(quality=95, subsampling=2)
    with pytest.raises(CanonicalReleaseError, match="4:4:4"):
        _validate_canonical_path(
            subsampled,
            repo_root=tmp_path.resolve(),
            release_dir=release_dir.resolve(),
            label="fixture",
            verify_file=True,
        )

    with_icc = write_image(
        quality=95,
        subsampling=0,
        icc_profile=b"fixture-icc",
    )
    with pytest.raises(CanonicalReleaseError, match="retained metadata"):
        _validate_canonical_path(
            with_icc,
            repo_root=tmp_path.resolve(),
            release_dir=release_dir.resolve(),
            label="fixture",
            verify_file=True,
        )

    truncated = write_image(quality=95, subsampling=0)
    image_path.write_bytes(image_path.read_bytes()[:-20])
    truncated["canonical_sha256"] = sha256_file(image_path)
    truncated["canonical_bytes"] = image_path.stat().st_size
    with pytest.raises(CanonicalReleaseError, match="cannot decode"):
        _validate_canonical_path(
            truncated,
            repo_root=tmp_path.resolve(),
            release_dir=release_dir.resolve(),
            label="fixture",
            verify_file=True,
        )
