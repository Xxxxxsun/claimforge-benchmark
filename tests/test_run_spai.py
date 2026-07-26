from __future__ import annotations

from pathlib import Path

import copy
import numpy as np
import pytest
from PIL import Image

from eval.opensource.common import sha256_file
from eval.opensource import run_spai as runner


def test_frozen_release_constants() -> None:
    assert runner.MODEL_SOURCE_COMMIT == (
        "8ff7b3b6779b4fcb43cf313471d9cb1c62d129a4"
    )
    assert runner.CHECKPOINT["bytes"] == 934_865_338
    assert runner.CHECKPOINT["sha256"] == (
        "24159f27d7c8c2cd0cb6c4019189eb89ad0874a0d9d15f8dc9afd39ca9648a55"
    )
    assert runner.CHECKPOINT["tensor_count"] == 324
    assert runner.CHECKPOINT["state_elements"] == 139_945_243
    assert runner.FEATURE_DIMENSION == 1096
    assert runner.ATTENTION_HEADS == 12
    assert runner.CLASSIFICATION_THRESHOLD_OPERATOR == ">"
    assert runner.PRIMARY_DEVICE == "cuda:0"


def test_cli_rejects_non_golden_device_before_loading_assets() -> None:
    with pytest.raises(ValueError, match="frozen on cuda:0"):
        runner.main(["--device", "cpu", "--preflight-only"])


def test_patch_geometry_matches_unfold_and_five_crop() -> None:
    grid = runner.compute_patch_geometry(1800, 1350)
    assert grid["patch_mode"] == "grid"
    assert grid["initial_grid"] == {"rows": 6, "columns": 8, "count": 48}
    assert grid["grid_covered_xyxy"] == [0, 0, 1792, 1344]
    assert grid["effective_patch_count"] == 48

    five = runner.compute_patch_geometry(683, 1024)
    assert five["initial_grid"]["count"] == 12
    assert five["patch_mode"] == "grid"

    fallback = runner.compute_patch_geometry(449, 231)
    assert fallback["initial_grid"]["count"] == 2
    assert fallback["patch_mode"] == "five_crop"
    assert fallback["effective_patch_count"] == 5
    assert len(fallback["five_crop_boxes_xyxy"]) == 5


def test_sub_224_input_uses_center_reflect101_padding(
    tmp_path: Path,
) -> None:
    pixels = (
        np.arange(13 * 17 * 3, dtype=np.uint32).reshape(13, 17, 3) * 31
    ).astype(np.uint8)
    path = tmp_path / "tiny.png"
    Image.fromarray(pixels, mode="RGB").save(path)
    tensor, audit = runner.preprocess_image(path)
    padding = audit["geometry"]["pad_if_needed"]
    assert tensor.shape == (3, 224, 224)
    assert tensor.dtype == np.float32
    assert padding == {
        "enabled": True,
        "minimum_size": [224, 224],
        "left": 103,
        "right": 104,
        "top": 105,
        "bottom": 106,
        "position": "center",
        "border_mode": "cv2.BORDER_REFLECT_101",
        "border_mode_value": 4,
    }


def test_preprocess_matches_official_albumentations_byte_for_byte() -> None:
    evidence = runner.validate_official_preprocess_equivalence()
    assert evidence["status"] == "passed"
    assert len(evidence["cases"]) == 2
    assert all(case["exact_match"] for case in evidence["cases"])
    assert evidence["cases"][0]["pad_if_needed"]["enabled"] is True


def test_full_visibility_census_matches_frozen_contract() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    _, _, rows = runner.load_release(
        repo_root,
        repo_root / runner.DEFAULT_DATASET_MANIFEST,
    )
    visibility = runner.build_pair_visibility(rows, repo_root)
    audit = runner.validate_frozen_visibility_census(visibility)
    assert audit["census"] == {"full": 243, "partial": 14, "none": 18}
    assert audit["patch_modes"] == {"grid": 262, "five_crop": 13}
    assert audit["mean_edit_visible_gt_fraction"] == pytest.approx(
        0.9096355444251016,
        abs=1e-15,
    )
    assert audit["by_domain"]["lodging"] == {
        "full": 132,
        "partial": 3,
        "none": 12,
    }


def _tiny_model():
    import torch

    class TinySPAI(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.heads = runner.ATTENTION_HEADS
            self.scale = (
                runner.ATTENTION_EMBED_DIMENSION // self.heads
            ) ** -0.5
            self.to_kv = torch.nn.Linear(
                runner.FEATURE_DIMENSION,
                runner.ATTENTION_EMBED_DIMENSION * 2,
                bias=False,
            )
            self.patch_aggregator = torch.nn.Parameter(
                torch.zeros(self.heads, 1, 128)
            )
            self.attend = torch.nn.Softmax(dim=-1)
            self.dropout = torch.nn.Dropout(0.5)
            self.to_out = torch.nn.Sequential(
                torch.nn.Linear(
                    runner.ATTENTION_EMBED_DIMENSION,
                    runner.FEATURE_DIMENSION,
                    bias=False,
                ),
                torch.nn.Dropout(0.5),
            )
            self.norm = torch.nn.LayerNorm(runner.FEATURE_DIMENSION)
            self.cls_head = torch.nn.Sequential(
                torch.nn.Linear(runner.FEATURE_DIMENSION, 1)
            )

        def forward(self, images, feature_extraction_batch_size=None):
            del feature_extraction_batch_size
            image = images[0]
            geometry = runner.compute_patch_geometry(
                int(image.shape[3]),
                int(image.shape[2]),
            )
            patches = int(geometry["effective_patch_count"])
            base = torch.linspace(
                -0.1,
                0.1,
                runner.FEATURE_DIMENSION,
                dtype=torch.float32,
                device=image.device,
            )
            x = torch.stack(
                [base + index * 0.001 for index in range(patches)]
            ).unsqueeze(0)
            key, value = self.to_kv(x).chunk(2, dim=-1)
            key = key.reshape(1, patches, self.heads, 128).permute(
                0, 2, 1, 3
            )
            value = value.reshape(1, patches, self.heads, 128).permute(
                0, 2, 1, 3
            )
            aggregator = self.patch_aggregator.expand(1, -1, -1, -1)
            attention = self.attend(
                torch.matmul(aggregator, key.transpose(-1, -2)) * self.scale
            )
            attended = torch.matmul(self.dropout(attention), value)
            attended = (
                attended.permute(0, 2, 1, 3)
                .contiguous()
                .reshape(1, 1, runner.ATTENTION_EMBED_DIMENSION)
            )
            x = self.to_out(attended).squeeze(1)
            x = self.norm(x)
            return self.cls_head(x)

    torch.manual_seed(4)
    return TinySPAI().eval()


def test_infer_captures_and_replays_all_three_artifacts() -> None:
    import torch

    model = _tiny_model()
    image = np.zeros((3, 224, 224), dtype=np.float32)
    scoring, patch, feature, attention, peak, latency = runner.infer_one(
        model,
        torch.device("cpu"),
        image,
    )
    assert patch.shape == (5, 1096)
    assert feature.shape == (1096,)
    assert attention.shape == (12, 5)
    assert np.allclose(attention.sum(axis=1), 1.0)
    assert scoring["manual_replay"]["official_attention_exact_match"] is True
    assert scoring["manual_replay"]["official_feature_exact_match"] is True
    assert scoring["manual_replay"]["complete_mlp_replay"] is True
    assert peak is None
    assert latency >= 0.0


def test_zero_logit_uses_strict_threshold() -> None:
    import torch

    model = _tiny_model()
    with torch.no_grad():
        model.cls_head[0].weight.zero_()
        model.cls_head[0].bias.zero_()
    patch = torch.zeros((3, runner.FEATURE_DIMENSION), dtype=torch.float32)
    scoring, _ = runner.replay_sca_norm_head(
        model=model,
        patch_features=patch,
    )
    assert scoring["raw_logit"] == 0.0
    assert scoring["probability"] == 0.5
    assert scoring["classification_decision"] is False
    assert scoring["t1"]["policy"] == runner.T1_POLICY


def test_three_artifacts_are_atomic_hashed_and_t2_free(
    tmp_path: Path,
) -> None:
    patch = np.zeros((5, 1096), dtype=np.float32)
    feature = np.zeros(1096, dtype=np.float32)
    attention = np.full((12, 5), 0.2, dtype=np.float32)
    fields = runner._persist_artifacts(
        artifact_dir=tmp_path / "artifacts",
        sample_id="safe-id",
        patch=patch,
        feature=feature,
        attention=attention,
        repo_root=tmp_path,
    )
    assert set(fields["artifact_paths"]) == {
        "spai_patch_features_npy",
        "spai_feature_npy",
        "spai_attention_npy",
    }
    for key, relative in fields["artifact_paths"].items():
        del key
        path = tmp_path / relative
        assert path.is_file()
        prefix = (
            "spai_patch_features"
            if "patch_features" in relative
            else "spai_attention"
            if "attention" in relative
            else "spai_feature"
        )
        assert sha256_file(path) == fields[f"{prefix}_sha256"]
    assert fields["spai_attention_semantics"].endswith("not_localization")
    assert not {"t2", "mask_path", "attention_map_path"}.intersection(fields)


def _resume_fixture(tmp_path: Path):
    import torch

    image_path = tmp_path / "input.png"
    Image.new("RGB", (224, 224), color=(20, 40, 60)).save(image_path)
    canonical = {
        "sample_id": "safe-id",
        "task_id": "task-1",
        "pair_rank": 0,
        "rank": 0,
        "kind": "forged",
        "label": 1,
        "domain": "lodging",
        "canonical_path": str(image_path),
        "canonical_sha256": sha256_file(image_path),
        "width": 224,
        "height": 224,
    }
    visibility = {
        "edit_visibility": "full",
        "edit_visible_gt_fraction": 1.0,
        "edit_visibility_evidence": {"test": True},
    }
    fingerprint = "a" * 64
    identity = runner._result_identity(
        canonical,
        repo_root=tmp_path,
        visibility=visibility,
        config_fingerprint=fingerprint,
    )
    image, preprocess = runner.preprocess_image(image_path)
    model = _tiny_model()
    scoring, patch, feature, attention, _, _ = runner.infer_one(
        model,
        torch.device("cpu"),
        image,
    )
    artifacts = runner._persist_artifacts(
        artifact_dir=tmp_path / "artifacts",
        sample_id="safe-id",
        patch=patch,
        feature=feature,
        attention=attention,
        repo_root=tmp_path,
    )
    row = {
        **identity,
        "status": "ok",
        "valid_for_metrics": True,
        "preprocess": preprocess,
        "preprocess_latency_ms": 1.0,
        "latency_ms": 2.0,
        "peak_cuda_memory_bytes": None,
        **artifacts,
        **scoring,
    }
    return row, identity, model, fingerprint


def test_resume_replays_all_artifacts_and_rejects_tamper(
    tmp_path: Path,
) -> None:
    import torch

    row, identity, model, fingerprint = _resume_fixture(tmp_path)
    runner._validate_resume_row(
        row,
        expected=identity,
        repo_root=tmp_path,
        run_dir=tmp_path,
        config_fingerprint=fingerprint,
        model=model,
        device=torch.device("cpu"),
    )
    tampered = copy.deepcopy(row)
    tampered["manual_replay"]["norm_hook_calls"] = 2
    with pytest.raises(ValueError, match="scalar/call"):
        runner._validate_resume_row(
            tampered,
            expected=identity,
            repo_root=tmp_path,
            run_dir=tmp_path,
            config_fingerprint=fingerprint,
            model=model,
            device=torch.device("cpu"),
        )


@pytest.mark.parametrize("value", ["../escape", "a/b", ".", "", "with space"])
def test_unsafe_run_component_is_rejected(value: str) -> None:
    with pytest.raises(ValueError):
        runner._safe_component(value, label="run-id")
