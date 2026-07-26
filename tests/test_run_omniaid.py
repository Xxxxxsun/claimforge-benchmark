from __future__ import annotations

import argparse
import hashlib
import json
import types
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from eval.opensource import run_omniaid as runner


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_frozen_release_contract_is_omniaid_dino_v2() -> None:
    assert runner.MODEL_SLUG == "omniaid_dino_v2_mirage_auto_router"
    assert runner.MODEL_INPUT_SIZE == 448
    assert runner.PREPROCESS_PROFILE.endswith("auto_router_448_v1")
    assert runner.CHECKPOINT["bytes"] == 3_238_483_725
    assert runner.CHECKPOINT["tensor_count"] == 2_852
    assert runner.CHECKPOINT["state_elements"] == 808_835_239
    assert runner.CHECKPOINT["unsafe_globals_allowlisted"] == [
        "argparse.Namespace"
    ]
    assert runner.DINO_BASE["architecture_contract"]["patch_size"] == 16
    assert runner.EXPERT_COUNT == 6
    assert runner.SEMANTIC_EXPERT_COUNT == 5
    assert runner.SEMANTIC_TOP_K == 2
    assert runner.ARTIFACT_EXPERT_INDEX == 5
    assert runner.CLASSIFICATION_THRESHOLD_OPERATOR == ">"
    assert runner.LICENSE_RECORD["commercial_use_cleared"] is False


def test_pinned_github_and_space_sources_verify() -> None:
    verified = runner.verify_source(
        runner.DEFAULT_SOURCE_ROOT,
        runner.DEFAULT_SPACE_ROOT,
    )
    assert verified["github"]["commit"] == runner.MODEL_SOURCE_COMMIT
    assert verified["space"]["commit"] == runner.MODEL_SPACE_COMMIT
    assert (
        verified["space"]["files"]["model/omniaid-dino.py"]["sha256"]
        == runner.SPACE_SOURCE_FILES["model/omniaid-dino.py"]
    )
    assert verified["github"]["tracked_license_files"] == []
    assert verified["space"]["tracked_license_files"] == []


def test_preprocess_matches_official_space_transform_and_no_exif_transpose(
    tmp_path: Path,
) -> None:
    import torch
    from torchvision import transforms

    pixels = np.zeros((17, 29, 3), dtype=np.uint8)
    pixels[:, :, 0] = np.arange(29, dtype=np.uint8)
    pixels[:, :, 1] = np.arange(17, dtype=np.uint8)[:, None]
    pixels[:, :, 2] = 193
    path = tmp_path / "input.png"
    Image.fromarray(pixels, mode="RGB").save(path)

    actual, audit = runner.preprocess_image(path)
    with Image.open(path) as opened:
        rgb = opened.convert("RGB")
        expected = transforms.Compose(
            [
                transforms.Resize([448, 448]),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=list(runner.IMAGENET_MEAN),
                    std=list(runner.IMAGENET_STD),
                ),
            ]
        )(rgb)
    assert np.array_equal(actual, expected.numpy())
    assert actual.shape == (3, 448, 448)
    assert actual.dtype == np.float32
    assert audit["decode"] == "PIL.Image.open_then_convert_RGB"
    assert audit["exif_transpose"] is False
    assert audit["resize"]["preserve_aspect_ratio"] is False
    assert audit["resize"]["crop"] is None
    assert audit["tensor_sha256"] == runner._array_sha256(actual)
    assert expected.dtype == torch.float32


def test_official_example_preprocessing_hashes_are_frozen() -> None:
    for frozen in runner.GOLDEN_CASES:
        path = runner.DEFAULT_SPACE_ROOT / frozen["path"]
        tensor, audit = runner.preprocess_image(path)
        assert _sha(path) == frozen["sha256"]
        assert audit["decoded_rgb_sha256"] == frozen["decoded_rgb_sha256"]
        assert audit["resized_rgb_sha256"] == frozen["resized_rgb_sha256"]
        assert audit["tensor_sha256"] == frozen["tensor_sha256"]
        assert runner._array_sha256(tensor) == frozen["tensor_sha256"]


class _FeatureExtractor(torch.nn.Module):
    """Hookable tiny module with the official pooler-output interface."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, image):
        pooled = torch.full(
            (image.shape[0], runner.FEATURE_DIMENSION),
            0.25,
            dtype=torch.float32,
            device=image.device,
        )
        return types.SimpleNamespace(pooler_output=pooled)


class _GatingNetwork(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, feature):
        return {
            "top_k_indices": torch.tensor(
                [[1, 3]], dtype=torch.int64, device=feature.device
            ),
            "top_k_gates": torch.tensor(
                [[0.25, 0.75]],
                dtype=torch.float32,
                device=feature.device,
            ),
        }


class _TinyOfficialModel:
    def __init__(self, *, bad_artifact_gate: bool = False) -> None:
        import torch

        self.feature_extractor = _FeatureExtractor()
        self.gating_network = _GatingNetwork()
        self.head = torch.nn.Linear(runner.FEATURE_DIMENSION, 2)
        with torch.no_grad():
            self.head.weight.zero_()
            self.head.bias.copy_(torch.tensor([-0.5, 0.5]))
        self.bad_artifact_gate = bad_artifact_gate

    def __call__(self, image, manual_weights=None):
        import torch

        assert manual_weights is None
        routing = self.feature_extractor(image).pooler_output
        gates = self.gating_network(routing)
        pooled = routing * 2.0
        logits = self.head(pooled)
        final = torch.zeros(
            (image.shape[0], runner.EXPERT_COUNT),
            dtype=torch.float32,
            device=image.device,
        )
        final.scatter_(
            1,
            gates["top_k_indices"],
            gates["top_k_gates"],
        )
        final[:, runner.ARTIFACT_EXPERT_INDEX] = (
            0.5 if self.bad_artifact_gate else 1.0
        )
        return {
            "prob": torch.softmax(logits, dim=1)[:, 1],
            "final_gates": final,
        }


def test_infer_one_captures_and_replays_all_official_arrays() -> None:
    import torch

    scoring, arrays, peak, latency = runner.infer_one(
        _TinyOfficialModel(),
        torch.device("cpu"),
        np.zeros((3, 448, 448), dtype=np.float32),
    )
    assert peak is None
    assert latency >= 0.0
    assert scoring["ai_score"] == pytest.approx(0.7310586)
    assert scoring["classification_decision"] is True
    assert scoring["semantic_top_k_indices"] == [1, 3]
    assert scoring["semantic_top_k_gates"] == [0.25, 0.75]
    assert scoring["final_expert_gates"] == [0.0, 0.25, 0.0, 0.75, 0.0, 1.0]
    assert scoring["manual_replay"]["head_logits_exact"] is True
    assert set(arrays) == set(runner.ARTIFACT_SCHEMA)
    assert arrays["pooler_output"].shape == (1024,)
    assert arrays["routing_feature"].shape == (1024,)
    assert arrays["semantic_top_k_indices"].dtype == np.int64
    assert arrays["final_gates"].sum() == 2.0


def test_infer_one_rejects_broken_always_on_artifact_expert() -> None:
    import torch

    with pytest.raises(ValueError, match="gate invariant"):
        runner.infer_one(
            _TinyOfficialModel(bad_artifact_gate=True),
            torch.device("cpu"),
            np.zeros((3, 448, 448), dtype=np.float32),
        )


def test_artifact_roundtrip_is_pickle_free_and_schema_strict(
    tmp_path: Path,
) -> None:
    arrays = {
        key: np.zeros(shape, dtype=dtype)
        for key, (shape, dtype) in runner.ARTIFACT_SCHEMA.items()
    }
    arrays["semantic_top_k_indices"][:] = [0, 2]
    arrays["semantic_top_k_gates"][:] = [0.4, 0.6]
    arrays["final_gates"][:] = [0.4, 0, 0.6, 0, 0, 1]
    path = tmp_path / "artifact.npz"
    runner._atomic_save_artifact(path, arrays)
    replay = runner._load_artifact(path)
    assert all(np.array_equal(arrays[key], replay[key]) for key in arrays)

    with np.load(path, allow_pickle=False) as payload:
        assert set(payload.files) == set(runner.ARTIFACT_SCHEMA)

    bad = dict(arrays)
    bad.pop("routing_feature")
    with pytest.raises(ValueError, match="unexpected keys"):
        runner._atomic_save_artifact(tmp_path / "bad.npz", bad)


def test_artifact_path_and_t2_guards() -> None:
    root = Path("/tmp/omniaid-test-run")
    assert runner._artifact_path(root, "safe-id").name == "safe-id.npz"
    with pytest.raises(ValueError, match="safe path component"):
        runner._artifact_path(root, "../escape")
    with pytest.raises(ValueError, match="invents OmniAID T2"):
        runner._reject_t2_payload({"nested": {"attention_map": "invented"}})


def test_verify_assets_requires_only_namespace_allowlist(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch

    state = {
        "head.weight": torch.zeros(2, 1024),
        "head.bias": torch.zeros(2),
        "gating_network.network.0.weight": torch.zeros(256, 1024),
        "gating_network.network.2.weight": torch.zeros(5, 256),
    }
    checkpoint = tmp_path / "checkpoint.pth"
    torch.save(
        {
            "model": state,
            "optimizer": {},
            "epoch": 0,
            "scaler": None,
            "args": argparse.Namespace(
                img_size=448,
                model="OmniAID_DINO",
                is_hybrid=True,
                training_mode="stage2_router_training",
                data_path="frozen",
            ),
        },
        checkpoint,
    )
    config_payload = {
        "DINOV3_path": runner.DINO_BASE["model_id"],
        "num_experts": 6,
        "rank_per_expert": 1,
        "moe_router_hidden_dim": 256,
        "moe_top_k": 2,
        "gradient_checkpointing_enable": False,
    }
    config = tmp_path / "config.json"
    config.write_text(json.dumps(config_payload), encoding="utf-8")
    schema = runner._state_schema_sha256(state)
    key_sha = hashlib.sha256("\n".join(state).encode()).hexdigest()
    monkeypatch.setattr(
        runner,
        "CHECKPOINT",
        {
            "bytes": checkpoint.stat().st_size,
            "sha256": _sha(checkpoint),
            "unsafe_globals_allowlisted": ["argparse.Namespace"],
            "top_level_keys": [
                "model",
                "optimizer",
                "epoch",
                "scaler",
                "args",
            ],
            "epoch": 0,
            "tensor_count": len(state),
            "state_elements": sum(value.numel() for value in state.values()),
            "ordered_key_sha256": key_sha,
            "schema_sha256": schema,
        },
    )
    monkeypatch.setattr(
        runner,
        "OMNIAID_CONFIG",
        {
            "bytes": config.stat().st_size,
            "sha256": _sha(config),
        },
    )
    assets, observed, observed_config = runner.verify_assets(
        checkpoint,
        config,
    )
    assert list(observed) == list(state)
    assert observed_config == config_payload
    assert assets["checkpoint"]["serialization_safety"]["allowlist"] == [
        "argparse.Namespace"
    ]
    assert assets["checkpoint"]["serialization_safety"][
        "arbitrary_code_execution_enabled"
    ] is False


def test_pair_selection_preserves_both_members_and_sample_id_mode() -> None:
    rows = [
        {
            "sample_id": f"{task}-{kind}",
            "task_id": task,
            "pair_rank": rank,
            "kind": kind,
        }
        for rank, task in enumerate(("a", "b"))
        for kind in ("real", "forged")
    ]
    selected = runner.select_inputs(rows, pair_limit=1)
    assert {row["sample_id"] for row in selected} == {"a-real", "a-forged"}
    one = runner.select_inputs(rows, pair_limit=None, sample_id="b-forged")
    assert [row["sample_id"] for row in one] == ["b-forged"]
    with pytest.raises(ValueError, match="mutually exclusive"):
        runner.select_inputs(rows, pair_limit=1, sample_id="a-real")
