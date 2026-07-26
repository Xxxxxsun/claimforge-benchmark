from __future__ import annotations

import copy
import hashlib
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image
from torchvision import transforms

from eval.opensource import analyze_universalfakedetect_run as audit
from eval.opensource.common import sha256_file


CURRENT_PROFILE = "current_head_native_center_crop224"
LEGACY_PROFILE = "checkpoint_era_resize256_center_crop224"


def _pins() -> SimpleNamespace:
    return SimpleNamespace(
        MODEL_NAME="UniversalFakeDetect",
        MODEL_SLUG="universalfakedetect_clip_vit_l14_ours_lc",
        MODEL_REPO_URL="https://example.invalid/ufd",
        MODEL_SOURCE_COMMIT="a" * 40,
        MODEL_ARCH="CLIP:ViT-L/14",
        SOURCE_FILES={"validate.py": "0" * 64},
        HEAD_CHECKPOINT={
            "path": "fc_weights.pth",
            "bytes": 1,
            "sha256": "1" * 64,
        },
        BACKBONE_CHECKPOINT={
            "path": "ViT-L-14.pt",
            "bytes": 1,
            "sha256": "2" * 64,
        },
        PREPROCESS_PROFILES={
            CURRENT_PROFILE: {
                "id": CURRENT_PROFILE,
                "resize": {"enabled": False},
            },
            LEGACY_PROFILE: {
                "id": LEGACY_PROFILE,
                "resize": {"enabled": True, "short_side": 256},
            },
        },
        SOURCE_CHECKPOINT_DRIFT={"current_head_removed_resize": True},
        FEATURE_DIMENSION=768,
        CLASSIFICATION_THRESHOLD=0.5,
    )


def _write_rgb(path: Path, array: np.ndarray) -> None:
    Image.fromarray(array.astype(np.uint8), mode="RGB").save(path)


def _save_feature(path: Path, feature: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.asarray(feature, dtype=np.float32), allow_pickle=False)
    return sha256_file(path)


def _sigmoid(value: float) -> float:
    return float(torch.sigmoid(torch.tensor(value, dtype=torch.float32)).item())


def _score_row(raw: float) -> dict:
    probability = _sigmoid(raw)
    decision = probability > 0.5
    return {
        "id": "sample",
        "raw_logit": raw,
        "probability": probability,
        "ai_score": probability,
        "score": probability,
        "score_semantics": "official_sigmoid_probability_higher_is_fake",
        "classification_decision": decision,
        "classification_threshold": 0.5,
        "classification_threshold_operator": ">",
        "classification": {
            "raw_logit": raw,
            "probability": probability,
            "ai_score": probability,
            "score": probability,
            "threshold": 0.5,
            "threshold_operator": ">",
            "decision": decision,
            "semantics": "official_sigmoid_probability_higher_is_fake",
        },
        "t1": {
            "raw_logit": raw,
            "probability": probability,
            "ai_score": probability,
            "score": probability,
            "threshold": 0.5,
            "threshold_operator": ">",
            "decision": decision,
            "policy": "official_UFD_CLIP_linear_probe_probability",
        },
        "manual_replay": {
            "raw_logit": raw,
            "probability": probability,
            "ai_score": probability,
            "classification_decision": decision,
            "model_forward_calls": 1,
            "fc_hook_calls": 1,
            "official_logit_exact_match": True,
            "official_probability_exact_match": True,
        },
    }


def test_profile_kind_rejects_unknown() -> None:
    pins = _pins()
    assert audit._profile_kind(CURRENT_PROFILE, pins) == "current_head_native"
    assert (
        audit._profile_kind(LEGACY_PROFILE, pins)
        == "checkpoint_era_resize256"
    )
    with pytest.raises(ValueError, match="unknown preprocess profile"):
        audit._profile_kind("invented", pins)


def test_resize256_geometry_uses_int_truncation_and_python_round() -> None:
    geometry = audit._preprocess_geometry(
        1800,
        1350,
        profile_kind="checkpoint_era_resize256",
    )
    assert geometry["resize"]["destination_size"] == [341, 256]
    assert geometry["center_crop"]["start_xy"] == [58, 16]
    assert geometry["center_crop"]["end_xy"] == [282, 240]
    # An odd excess exercises Python's ties-to-even behavior.
    assert audit._center_crop_start(341) == 58
    assert audit._center_crop_start(343) == 60


@pytest.mark.parametrize(
    ("profile_kind", "reference"),
    [
        (
            "current_head_native",
            transforms.Compose(
                [
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(audit.CLIP_MEAN, audit.CLIP_STD),
                ]
            ),
        ),
        (
            "checkpoint_era_resize256",
            transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(audit.CLIP_MEAN, audit.CLIP_STD),
                ]
            ),
        ),
    ],
)
def test_independent_preprocess_is_torchvision_exact(
    tmp_path: Path,
    profile_kind: str,
    reference,
) -> None:
    yy, xx = np.mgrid[:301, :407]
    array = np.stack(
        (
            xx % 256,
            yy % 256,
            (xx * 3 + yy * 5) % 256,
        ),
        axis=-1,
    ).astype(np.uint8)
    path = tmp_path / "odd.png"
    _write_rgb(path, array)
    actual = audit.preprocess_image(
        path,
        profile_kind=profile_kind,
        torch_module=torch,
    )
    expected = reference(Image.open(path).convert("RGB"))
    assert torch.equal(actual.tensor, expected)
    assert actual.crop_rgb.shape == (224, 224, 3)
    assert (
        actual.tensor_sha256
        == hashlib.sha256(expected.numpy().tobytes()).hexdigest()
    )


def test_center_crop_padding_matches_torchvision(tmp_path: Path) -> None:
    array = np.arange(101 * 151 * 3, dtype=np.uint8).reshape(101, 151, 3)
    path = tmp_path / "small.png"
    _write_rgb(path, array)
    actual = audit.preprocess_image(
        path,
        profile_kind="current_head_native",
        torch_module=torch,
    )
    expected = transforms.Compose(
        [
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(audit.CLIP_MEAN, audit.CLIP_STD),
        ]
    )(Image.open(path).convert("RGB"))
    assert torch.equal(actual.tensor, expected)
    assert actual.geometry["center_crop"]["padding_ltrb"] == [36, 61, 37, 62]


def test_score_contract_is_strict_and_rejects_alias_tamper() -> None:
    row = _score_row(0.0)
    replay = audit._audit_score_fields(
        row,
        replay_raw_logit=0.0,
        replay_probability=0.5,
        raw_tolerance=0.0,
        probability_tolerance=0.0,
    )
    assert replay["decision"] is False
    bad = copy.deepcopy(row)
    bad["classification_decision"] = True
    with pytest.raises(ValueError, match="classification_decision"):
        audit._audit_score_fields(
            bad,
            replay_raw_logit=0.0,
            replay_probability=0.5,
            raw_tolerance=0.0,
            probability_tolerance=0.0,
        )


def test_score_contract_allows_frozen_cross_device_sigmoid_tolerance() -> None:
    raw = -7.389519214630127
    stored_gpu_probability = 0.0006170368869788945
    replay_cpu_probability = 0.0006170368287712336
    row = _score_row(raw)
    row["probability"] = stored_gpu_probability
    row["ai_score"] = stored_gpu_probability
    row["score"] = stored_gpu_probability
    for key in ("classification", "t1"):
        row[key]["probability"] = stored_gpu_probability
        row[key]["ai_score"] = stored_gpu_probability
        row[key]["score"] = stored_gpu_probability
    row["manual_replay"]["probability"] = stored_gpu_probability
    row["manual_replay"]["ai_score"] = stored_gpu_probability
    replay = audit._audit_score_fields(
        row,
        replay_raw_logit=raw,
        replay_probability=replay_cpu_probability,
    )
    assert replay["decision"] is False


def test_score_contract_rejects_probability_decision_opposite_logit() -> None:
    row = _score_row(-1e-8)
    probability = 0.50000001
    row["probability"] = probability
    row["ai_score"] = probability
    row["score"] = probability
    row["classification_decision"] = True
    row["classification"].update(
        probability=probability,
        ai_score=probability,
        score=probability,
        decision=True,
    )
    row["t1"].update(
        probability=probability,
        ai_score=probability,
        score=probability,
        decision=True,
    )
    row["manual_replay"].update(
        probability=probability,
        ai_score=probability,
        classification_decision=True,
    )
    with pytest.raises(ValueError, match="probability/logit decision"):
        audit._audit_score_fields(
            row,
            replay_raw_logit=-1e-8,
            replay_probability=probability,
        )


@pytest.mark.parametrize(
    "payload",
    [
        {"localization": None},
        {"nested": {"mask_path": "invented.png"}},
        {"task": {"s_joint": 0.7}},
        [{"pixel_metrics": {}}],
    ],
)
def test_t2_localization_and_joint_are_rejected(payload) -> None:
    with pytest.raises(ValueError, match="unsupported"):
        audit._reject_t2_localization_or_joint(payload, label="payload")


def test_physical_history_uses_last_row() -> None:
    history = audit.summarize_result_history(
        [
            {"id": "a", "status": "error"},
            {"id": "b", "status": "ok"},
            {"id": "a", "status": "ok"},
        ]
    )
    assert history["physical_rows"] == 3
    assert history["unique_ids"] == 2
    assert history["duplicate_rows"] == 1
    assert history["recovered_ids"] == ["a"]
    assert history["latest_status_counts"] == {"ok": 2}


def test_feature_loader_rejects_tamper_and_wrong_dtype(tmp_path: Path) -> None:
    run_id = "run_a"
    path = tmp_path / run_id / "clip_features" / "x.npy"
    digest = _save_feature(path, np.zeros(768, dtype=np.float32))
    row = {
        "id": "x",
        "clip_feature_path": str(path),
        "clip_feature_sha256": digest,
        "clip_feature_shape": [768],
        "clip_feature_dtype": "float32",
    }
    feature, loaded = audit._load_feature(
        row,
        repo_root=tmp_path,
        run_id=run_id,
    )
    assert loaded == path
    assert feature.dtype == np.float32
    row["clip_feature_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        audit._load_feature(row, repo_root=tmp_path, run_id=run_id)

    wrong_path = tmp_path / run_id / "clip_features" / "wrong.npy"
    np.save(
        wrong_path,
        np.zeros(768, dtype=np.float64),
        allow_pickle=False,
    )
    wrong_digest = sha256_file(wrong_path)
    wrong = {
        **row,
        "clip_feature_path": str(wrong_path),
        "clip_feature_sha256": wrong_digest,
    }
    with pytest.raises(ValueError, match="feature dtype"):
        audit._load_feature(wrong, repo_root=tmp_path, run_id=run_id)


def test_visibility_profiles_have_frozen_mouse_census() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    rows = [
        json.loads(line)
        for line in (
            repo_root
            / "outputs/opensource/mouse_canonical_v1/inputs.jsonl"
        ).read_text(encoding="utf-8").splitlines()
    ]
    forged = [row for row in rows if row["kind"] == "forged"]
    current = Counter(
        audit._visibility_from_exact_gt(
            row,
            repo_root=repo_root,
            profile_kind="current_head_native",
        )["category"]
        for row in forged
    )
    legacy = Counter(
        audit._visibility_from_exact_gt(
            row,
            repo_root=repo_root,
            profile_kind="checkpoint_era_resize256",
        )["category"]
        for row in forged
    )
    assert dict(current) == audit.CURRENT_HEAD_VISIBILITY_CENSUS
    assert dict(legacy) == audit.CHECKPOINT_ERA_VISIBILITY_CENSUS


def _fingerprinted_manifest(run_id: str, selection: list[dict]) -> dict:
    manifest = {
        "schema_version": "opensource_run_manifest_v1",
        "run_id": run_id,
        "preprocess_profile": CURRENT_PROFILE,
        "source_checkpoint_drift": {"removed_resize": True},
        "runtime_contract": {"same": True},
        "environment": {"same": True},
        "protocol": {"same": True},
        "dataset": {"same": True},
        "adapter_contract": [{"same": True}],
        "model": {"same": True},
        "selection": {"rows": selection},
    }
    manifest["fingerprint"] = audit._manifest_fingerprint(manifest)
    return manifest


def test_prefix_requires_independent_artifacts_and_exact_values(
    tmp_path: Path,
) -> None:
    full_run = "full_run"
    prefix_run = "prefix_run"
    ordered = [{"sample_id": "x"}]
    full_manifest = _fingerprinted_manifest(full_run, ordered)
    prefix_manifest = _fingerprinted_manifest(prefix_run, ordered)
    feature = np.arange(768, dtype=np.float32)
    full_path = tmp_path / full_run / "clip_features" / "x.npy"
    prefix_path = tmp_path / prefix_run / "clip_features" / "x.npy"
    full_digest = _save_feature(full_path, feature)
    prefix_digest = _save_feature(prefix_path, feature)
    assert full_digest == prefix_digest
    common = {
        "id": "x",
        "status": "ok",
        "clip_feature_sha256": full_digest,
        "clip_feature_shape": [768],
        "clip_feature_dtype": "float32",
    }
    full_row = {
        **common,
        "run_id": full_run,
        "run_manifest_fingerprint": full_manifest["fingerprint"],
        "clip_feature_path": str(full_path),
    }
    prefix_row = {
        **common,
        "run_id": prefix_run,
        "run_manifest_fingerprint": prefix_manifest["fingerprint"],
        "clip_feature_path": str(prefix_path),
    }
    result = audit.audit_prefix_reproducibility(
        repo_root=tmp_path,
        full_run_id=full_run,
        full_manifest=full_manifest,
        full_rows=[full_row],
        prefix_run_id=prefix_run,
        prefix_manifest=prefix_manifest,
        prefix_rows=[prefix_row],
    )
    assert result["samples_compared"] == 1

    copied = dict(prefix_row)
    copied["clip_feature_path"] = str(full_path)
    with pytest.raises(ValueError, match="outside its run directory|reuses"):
        audit.audit_prefix_reproducibility(
            repo_root=tmp_path,
            full_run_id=full_run,
            full_manifest=full_manifest,
            full_rows=[full_row],
            prefix_run_id=prefix_run,
            prefix_manifest=prefix_manifest,
            prefix_rows=[copied],
        )


class _ConstantEncoder:
    def encode_image(self, tensor: torch.Tensor) -> torch.Tensor:
        value = tensor.mean(dim=(1, 2, 3), keepdim=False)
        return value[:, None].repeat(1, 768)


def _artifact_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    canonical_crop_difference: bool = False,
) -> tuple[dict, list[dict], list[dict], dict[str, dict], audit.ReplayRuntime]:
    pins = _pins()
    head_asset = tmp_path / "fc_weights.pth"
    backbone_asset = tmp_path / "ViT-L-14.pt"
    head_asset.write_bytes(b"h")
    backbone_asset.write_bytes(b"b")
    pins.HEAD_CHECKPOINT = {
        "path": head_asset.name,
        "bytes": 1,
        "sha256": sha256_file(head_asset),
    }
    pins.BACKBONE_CHECKPOINT = {
        "path": backbone_asset.name,
        "bytes": 1,
        "sha256": sha256_file(backbone_asset),
    }
    monkeypatch.setattr(audit, "_load_runner_pins", lambda: pins)

    image = np.zeros((300, 300, 3), dtype=np.uint8)
    image[100:200, 100:200] = 127
    real_path = tmp_path / "real.png"
    forged_path = tmp_path / "forged.png"
    _write_rgb(real_path, image)
    forged_image = image.copy()
    if canonical_crop_difference:
        forged_image[150, 150, 0] = np.uint8(
            (int(forged_image[150, 150, 0]) + 1) % 256
        )
    _write_rgb(forged_path, forged_image)
    mask = np.zeros((300, 300), dtype=np.uint8)
    mask[0:2, 0:2] = 255
    mask_path = tmp_path / "mask.png"
    Image.fromarray(mask, mode="L").save(mask_path)
    run_id = "artifact_run"
    common_input = {
        "task_id": "task",
        "pair_rank": 0,
        "domain": "test",
        "width": 300,
        "height": 300,
        "edit_region_xyxy": [0, 0, 2, 2],
    }
    real = {
        **common_input,
        "sample_id": "real",
        "rank": 0,
        "kind": "real",
        "label": 0,
        "canonical_path": str(real_path),
        "canonical_sha256": sha256_file(real_path),
        "gt_mask_kind": "all_zero",
        "gt_mask_path": None,
        "gt_mask_sha256": None,
        "gt_positive_pixels": 0,
    }
    forged = {
        **common_input,
        "sample_id": "forged",
        "rank": 1,
        "kind": "forged",
        "label": 1,
        "canonical_path": str(forged_path),
        "canonical_sha256": sha256_file(forged_path),
        "gt_mask_kind": "exact_diff",
        "gt_mask_path": str(mask_path),
        "gt_mask_sha256": sha256_file(mask_path),
        "gt_positive_pixels": 4,
    }
    head = torch.nn.Linear(768, 1)
    with torch.no_grad():
        head.weight.zero_()
        head.bias.fill_(0.25)
    encoder = _ConstantEncoder()
    latest: dict[str, dict] = {}
    for canonical in (real, forged):
        prepared = audit.preprocess_image(
            Path(canonical["canonical_path"]),
            profile_kind="current_head_native",
            torch_module=torch,
        )
        with torch.inference_mode():
            feature = encoder.encode_image(prepared.tensor[None, ...]).reshape(768)
            raw = float(head(feature[None, ...]).item())
        feature_path = (
            tmp_path / run_id / "clip_features" / f"{canonical['sample_id']}.npy"
        )
        digest = _save_feature(feature_path, feature.numpy())
        visibility = audit._visibility_from_exact_gt(
            forged,
            repo_root=tmp_path,
            profile_kind="current_head_native",
        )
        row = {
            **_score_row(raw),
            "id": canonical["sample_id"],
            "status": "ok",
            "preprocess_profile": CURRENT_PROFILE,
            "clip_feature_path": str(feature_path),
            "clip_feature_sha256": digest,
            "clip_feature_shape": [768],
            "clip_feature_dtype": "float32",
            "clip_feature_semantics": "raw_not_l2_normalized",
            "preprocess": {
                "geometry": prepared.geometry,
                "decoded_rgb_sha256": prepared.decoded_rgb_sha256,
                "crop_rgb_sha256": prepared.crop_rgb_sha256,
                "crop_rgb_shape": [224, 224, 3],
                "crop_rgb_dtype": "uint8",
                "tensor_shape": [3, 224, 224],
                "tensor_dtype": "float32",
                "tensor_sha256": prepared.tensor_sha256,
            },
            "edit_visibility": visibility["category"],
            "edit_visible_gt_fraction": visibility["visible_fraction"],
            "edit_visibility_evidence": {
                "gt": {
                    key: visibility[key]
                    for key in (
                        "basis",
                        "category",
                        "visible_fraction",
                        "positive_pixels",
                        "visible_positive_pixel_centers",
                        "forged_sample_id",
                        "profile_id",
                        "formula",
                    )
                },
                "edit_box": visibility["edit_box"],
            },
        }
        latest[canonical["sample_id"]] = row
    manifest = {
        "run_id": run_id,
        "preprocess_profile": CURRENT_PROFILE,
        "model": {
            "head_checkpoint": {
                "path": str(head_asset),
                "bytes": 1,
                "sha256": sha256_file(head_asset),
            },
            "backbone_checkpoint": {
                "path": str(backbone_asset),
                "bytes": 1,
                "sha256": sha256_file(backbone_asset),
            },
        },
    }
    runtime = audit.ReplayRuntime(
        torch=torch,
        device=torch.device("cpu"),
        recorded_device="cpu",
        evidence={"injected": True},
    )
    return manifest, [real, forged], [real, forged], latest, runtime, encoder, head


def test_artifact_replay_enforces_canonical_crop_equal_downstream_exactness(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        manifest,
        all_inputs,
        selected,
        latest,
        runtime,
        encoder,
        head,
    ) = _artifact_fixture(tmp_path, monkeypatch)
    result = audit.audit_artifacts(
        repo_root=tmp_path,
        source_root=tmp_path,
        manifest=manifest,
        all_input_rows=all_inputs,
        selected=selected,
        latest=latest,
        runtime=runtime,
        encoder=encoder,
        head=head,
    )
    assert result["images_fully_reencoded"] == 2
    assert result["maximum_feature_absolute_difference"] == 0.0
    equivalence = result["canonical_crop_pair_equivalence"]
    assert equivalence["crop_equal_pairs"] == 1
    assert equivalence["crop_different_pairs"] == 0
    assert equivalence["by_edit_visibility"]["none"] == {
        "pairs": 1,
        "crop_equal_pairs": 1,
        "crop_different_pairs": 0,
        "crop_equal_fraction": 1.0,
    }
    assert equivalence["crop_equal_downstream_exact_pairs_validated"] == 1


def test_precanonical_visibility_none_does_not_require_canonical_crop_equality(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        manifest,
        all_inputs,
        selected,
        latest,
        runtime,
        encoder,
        head,
    ) = _artifact_fixture(
        tmp_path,
        monkeypatch,
        canonical_crop_difference=True,
    )
    result = audit.audit_artifacts(
        repo_root=tmp_path,
        source_root=tmp_path,
        manifest=manifest,
        all_input_rows=all_inputs,
        selected=selected,
        latest=latest,
        runtime=runtime,
        encoder=encoder,
        head=head,
    )
    equivalence = result["canonical_crop_pair_equivalence"]
    assert result["edit_visibility_tasks"] == {"none": 1}
    assert equivalence["crop_equal_pairs"] == 0
    assert equivalence["crop_different_pairs"] == 1
    assert equivalence["by_edit_visibility"]["none"] == {
        "pairs": 1,
        "crop_equal_pairs": 0,
        "crop_different_pairs": 1,
        "crop_equal_fraction": 0.0,
    }
    assert equivalence["crop_equal_downstream_exact_pairs_validated"] == 0
    detail = equivalence["crop_different_pair_details"][0]
    assert detail["task_id"] == "task"
    assert detail["edit_visibility"] == "none"
    assert detail["differing_pixels"] == 1


def test_artifact_replay_rejects_persisted_feature_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        manifest,
        all_inputs,
        selected,
        latest,
        runtime,
        encoder,
        head,
    ) = _artifact_fixture(tmp_path, monkeypatch)
    row = latest["forged"]
    path = Path(row["clip_feature_path"])
    tampered = np.load(path, allow_pickle=False)
    tampered[0] += np.float32(1.0)
    row["clip_feature_sha256"] = _save_feature(path, tampered)
    with pytest.raises(ValueError, match="persisted feature differs"):
        audit.audit_artifacts(
            repo_root=tmp_path,
            source_root=tmp_path,
            manifest=manifest,
            all_input_rows=all_inputs,
            selected=selected,
            latest=latest,
            runtime=runtime,
            encoder=encoder,
            head=head,
        )
