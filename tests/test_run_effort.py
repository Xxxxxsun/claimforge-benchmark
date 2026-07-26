from __future__ import annotations

import copy
import gc
import json
import os
from collections import OrderedDict
from pathlib import Path
from unittest import mock

import cv2
import numpy as np
import pytest
import torch
import transformers
from PIL import Image
from torch import nn
from torchvision import transforms

from eval.opensource import run_effort as runner
from eval.opensource.common import sha256_file


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_frozen_effort_contract_constants() -> None:
    assert runner.MODEL_SOURCE_COMMIT == (
        "96f5dea2b534d400cfd7003f053c7e93c8e16461"
    )
    assert runner.MODEL_SLUG == "effort_clip_l14_genimage_sdv14"
    assert (
        runner.PREPROCESS_PROFILE
        == "official_deepfakebench_demo_natural_image_linear224_v1"
    )
    assert runner.MODEL_INPUT_SIZE == 224
    assert runner.FEATURE_DIMENSION == 1024
    assert runner.CLASS_COUNT == 2
    assert runner.SVD_MODULE_COUNT == 96
    assert runner.SVD_FROZEN_RANK == 1023
    assert runner.SVD_RESIDUAL_RANK == 1
    assert runner.CLASSIFICATION_THRESHOLD == 0.5
    assert runner.CLASSIFICATION_THRESHOLD_OPERATOR == ">"
    assert runner.MODEL_SEED == 20260724
    assert runner.CHECKPOINT["tensor_count"] == 681
    assert runner.CHECKPOINT["state_elements"] == 303_378_530
    assert runner.CHECKPOINT["bytes"] == 1_213_769_519
    assert runner.CHECKPOINT["sha256"] == (
        "7c32ceb4e66d303050e8fc5dc7543fa347693fb4ee6b5df4d6eaf9f6a92fb813"
    )
    assert runner.CHECKPOINT["ordered_key_sha256"] == (
        "1782f72f07007cebae76a0f315845f1c60456d9223d47c8ce2f35a8f43816da7"
    )
    assert runner.CHECKPOINT["schema_sha256"] == (
        "bb1d4ba1c015ab4354b42e11af101e29b19a1ab71704b0302bac465c6d3f1489"
    )
    assert runner.HF_CONFIG["revision"] == (
        "32bd64288804d66eefd0ccbe215aa642df71cc41"
    )
    assert runner.HF_CONFIG["sha256"] == (
        "8a09b467700c58138c29d53c605b34ebc69beaadd13274a8a2af8ad2c2f4032a"
    )
    assert runner.LICENSE_RECORD["commercial_use_cleared"] is False
    assert runner.LICENSE_RECORD["tracked_license_file_present"] is False


def test_real_pinned_source_contract_passes() -> None:
    assert runner.DEFAULT_SOURCE_ROOT.is_dir()
    source = runner.verify_source(runner.DEFAULT_SOURCE_ROOT)

    assert source["commit"] == runner.MODEL_SOURCE_COMMIT
    assert source["tracked_dirty"] is False
    assert source["tracked_license_files"] == []
    assert set(source["files"]) == set(runner.SOURCE_FILES)
    for relative, expected_sha256 in runner.SOURCE_FILES.items():
        assert source["files"][relative]["sha256"] == expected_sha256


def test_real_checkpoint_and_clip_config_have_strict_681_schema() -> None:
    assert runner.DEFAULT_CHECKPOINT.is_file()
    assert runner.DEFAULT_HF_CONFIG.is_file()

    assets, state, config = runner.verify_assets(
        runner.DEFAULT_CHECKPOINT,
        runner.DEFAULT_HF_CONFIG,
    )
    try:
        assert assets["checkpoint"]["schema_verified"] is True
        assert assets["checkpoint"]["serialization_safety"] == {
            "weights_only": True,
            "unsafe_globals": [],
        }
        assert len(state) == 681
        assert sum(value.numel() for value in state.values()) == 303_378_530
        assert runner._state_schema_sha256(
            OrderedDict(
                (f"module.{key}", value) for key, value in state.items()
            )
        ) == runner.CHECKPOINT["schema_sha256"]
        assert tuple(state["head.weight"].shape) == (2, 1024)
        assert tuple(state["head.bias"].shape) == (2,)
        assert state["head.weight"].dtype == torch.float32

        weight_main_keys = [
            key
            for key in state
            if key.endswith(".self_attn.k_proj.weight_main")
            or key.endswith(".self_attn.v_proj.weight_main")
            or key.endswith(".self_attn.q_proj.weight_main")
            or key.endswith(".self_attn.out_proj.weight_main")
        ]
        residual_s_keys = [
            key
            for key in state
            if key.endswith(".self_attn.k_proj.S_residual")
            or key.endswith(".self_attn.v_proj.S_residual")
            or key.endswith(".self_attn.q_proj.S_residual")
            or key.endswith(".self_attn.out_proj.S_residual")
        ]
        assert len(weight_main_keys) == 96
        assert len(residual_s_keys) == 96
        assert all(tuple(state[key].shape) == (1024, 1024) for key in weight_main_keys)
        assert all(tuple(state[key].shape) == (1,) for key in residual_s_keys)

        vision = config["vision_config"]
        assert vision["hidden_size"] == 1024
        assert vision["num_hidden_layers"] == 24
        assert vision["num_attention_heads"] == 16
        assert vision["image_size"] == 224
        assert vision["patch_size"] == 14
    finally:
        del state
        gc.collect()


class _FakeClipVisionConfig:
    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs


class _FakeAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # Registration order deliberately matches the pinned HF CLIP graph.
        self.k_proj = nn.Linear(1, 1)
        self.v_proj = nn.Linear(1, 1)
        self.q_proj = nn.Linear(1, 1)
        self.out_proj = nn.Linear(1, 1)


class _FakeLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = _FakeAttention()


class _FakeEmbeddings(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer(
            "position_ids",
            torch.arange(257, dtype=torch.long).expand(1, -1),
            persistent=False,
        )


class _FakeBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embeddings = _FakeEmbeddings()
        self.encoder = nn.Module()
        self.encoder.layers = nn.ModuleList(
            [_FakeLayer() for _ in range(24)]
        )


class _FakeClipVisionModel:
    def __init__(self, _config: object) -> None:
        self.vision_model = _FakeBackbone()


def _lightweight_svd_state() -> OrderedDict[str, torch.Tensor]:
    """Create correct logical shapes backed by one shared scalar storage."""

    scalar = torch.tensor(0.0, dtype=torch.float32)

    def expanded(*shape: int) -> torch.Tensor:
        return scalar.expand(*shape)

    state: OrderedDict[str, torch.Tensor] = OrderedDict()
    for layer in range(24):
        for projection in ("k_proj", "v_proj", "q_proj", "out_proj"):
            base = (
                f"backbone.encoder.layers.{layer}.self_attn.{projection}"
            )
            state[f"{base}.weight_main"] = expanded(1024, 1024)
            state[f"{base}.bias"] = expanded(1024)
            state[f"{base}.S_residual"] = expanded(1)
            state[f"{base}.U_residual"] = expanded(1024, 1)
            state[f"{base}.V_residual"] = expanded(1, 1024)
    state["head.weight"] = expanded(2, 1024)
    state["head.bias"] = expanded(2)
    return state


def test_shape_only_model_build_has_exact_96_rank_one_svd_modules(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        transformers,
        "CLIPVisionConfig",
        _FakeClipVisionConfig,
    )
    monkeypatch.setattr(
        transformers,
        "CLIPVisionModel",
        _FakeClipVisionModel,
    )
    model, audit = runner._build_model(
        _lightweight_svd_state(),
        {"vision_config": {}},
        torch.device("cpu"),
    )

    assert audit["strict_load"] is True
    assert audit["missing_keys"] == []
    assert audit["unexpected_keys"] == []
    assert audit["svd_modules"] == 96
    assert audit["frozen_rank"] == 1023
    assert audit["residual_rank"] == 1
    assert audit["svd_module_names"][0] == (
        "backbone.encoder.layers.0.self_attn.k_proj"
    )
    assert audit["svd_module_names"][-1] == (
        "backbone.encoder.layers.23.self_attn.out_proj"
    )
    assert audit["head_weight_shape"] == [2, 1024]
    assert audit["head_bias_shape"] == [2]
    assert audit["position_ids_shape"] == [1, 257]
    assert not any(parameter.is_meta for parameter in model.parameters())
    assert not any(buffer.is_meta for buffer in model.buffers())

    modules = [
        module
        for module in model.modules()
        if type(module).__name__ == "SVDResidualLinear"
    ]
    assert len(modules) == 96
    assert all(module.r == 1023 for module in modules)
    assert all(tuple(module.weight_main.shape) == (1024, 1024) for module in modules)
    assert all(tuple(module.U_residual.shape) == (1024, 1) for module in modules)
    assert all(tuple(module.S_residual.shape) == (1,) for module in modules)
    assert all(tuple(module.V_residual.shape) == (1, 1024) for module in modules)

    output = modules[0](torch.zeros((1, 1024), dtype=torch.float32))
    assert output.shape == (1, 1024)
    assert torch.count_nonzero(output).item() == 0


def test_preprocess_matches_official_opencv_and_torchvision_exactly() -> None:
    for frozen in runner.GOLDEN_CASES:
        path = runner.DEFAULT_SOURCE_ROOT / frozen["path"]
        actual, audit = runner.preprocess_image(path)

        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        assert bgr is not None
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(
            rgb,
            (224, 224),
            interpolation=cv2.INTER_LINEAR,
        )
        expected = transforms.Normalize(
            runner.CLIP_MEAN,
            runner.CLIP_STD,
        )(transforms.ToTensor()(Image.fromarray(resized))).numpy()

        np.testing.assert_array_equal(actual, expected)
        assert actual.dtype == np.float32
        assert actual.flags.c_contiguous
        assert audit["decode"] == "cv2.imread_IMREAD_COLOR"
        assert audit["resize"]["interpolation"] == "cv2_INTER_LINEAR"
        assert audit["resize"]["preserve_aspect_ratio"] is False
        assert audit["resize"]["crop"] is None
        assert audit["resize"]["face_alignment"] is False
        assert audit["decoded_rgb_sha256"] == frozen["decoded_rgb_sha256"]
        assert audit["resized_rgb_sha256"] == frozen["resized_rgb_sha256"]
        assert audit["tensor_sha256"] == frozen["tensor_sha256"]
        assert audit["tensor_sha256"] == runner._array_sha256(actual)


def test_inter_linear_profile_cannot_silently_change_to_cubic(
    tmp_path: Path,
) -> None:
    y, x = np.indices((13, 17))
    rgb = np.stack(
        [
            ((x + y) % 2) * 255,
            (x * 37 + y * 19) % 256,
            (x * 11 + y * 53) % 256,
        ],
        axis=2,
    ).astype(np.uint8)
    path = tmp_path / "high_frequency.png"
    Image.fromarray(rgb, mode="RGB").save(path)

    _, audit = runner.preprocess_image(path)
    linear = cv2.resize(
        rgb,
        (224, 224),
        interpolation=cv2.INTER_LINEAR,
    )
    cubic = cv2.resize(
        rgb,
        (224, 224),
        interpolation=cv2.INTER_CUBIC,
    )

    assert not np.array_equal(linear, cubic)
    assert audit["resized_rgb_sha256"] == runner._array_sha256(linear)
    assert audit["resized_rgb_sha256"] != runner._array_sha256(cubic)


def test_atomic_artifact_round_trip_has_exact_safe_contract(
    tmp_path: Path,
) -> None:
    path = tmp_path / "artifact.npz"
    feature = np.linspace(-1.0, 1.0, 1024, dtype=np.float32)
    logits = np.asarray([-0.25, 0.75], dtype=np.float32)
    runner._atomic_save_artifact(path, feature, logits)

    with np.load(path, allow_pickle=False) as payload:
        assert set(payload.files) == {"pooler_output", "class_logits"}
    loaded_feature, loaded_logits = runner._load_artifact(path)
    np.testing.assert_array_equal(loaded_feature, feature)
    np.testing.assert_array_equal(loaded_logits, logits)
    assert loaded_feature.dtype == np.float32
    assert loaded_logits.dtype == np.float32
    assert loaded_feature.flags.c_contiguous
    assert loaded_logits.flags.c_contiguous
    assert len(sha256_file(path)) == 64


@pytest.mark.parametrize(
    "variant",
    ["extra", "wrong_shape", "wrong_dtype", "nonfinite", "object"],
)
def test_artifact_loader_rejects_malformed_npz(
    tmp_path: Path,
    variant: str,
) -> None:
    path = tmp_path / f"{variant}.npz"
    feature: np.ndarray = np.zeros(1024, dtype=np.float32)
    logits: np.ndarray = np.zeros(2, dtype=np.float32)
    extras: dict[str, np.ndarray] = {}
    if variant == "extra":
        extras["unexpected"] = np.zeros(1, dtype=np.float32)
    elif variant == "wrong_shape":
        feature = np.zeros(1023, dtype=np.float32)
    elif variant == "wrong_dtype":
        logits = np.zeros(2, dtype=np.float64)
    elif variant == "nonfinite":
        feature[0] = np.nan
    elif variant == "object":
        feature = np.asarray([object()] * 1024, dtype=object)
    np.savez(
        path,
        pooler_output=feature,
        class_logits=logits,
        **extras,
    )

    with pytest.raises(ValueError):
        runner._load_artifact(path)


class _ToyEffort(nn.Module):
    def __init__(self, *, drift: float = 0.0) -> None:
        super().__init__()
        self.head = nn.Linear(1024, 2, dtype=torch.float32)
        self.forward_calls = 0
        self.drift = drift
        with torch.no_grad():
            self.head.weight.zero_()
            self.head.bias.zero_()

    def forward(
        self,
        image: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.forward_calls += 1
        feature = torch.zeros(
            (image.shape[0], 1024),
            dtype=torch.float32,
            device=image.device,
        )
        logits = self.head(feature)
        if self.drift:
            logits = logits + torch.tensor(
                [0.0, self.drift],
                dtype=torch.float32,
                device=image.device,
            )
        return logits, feature


def test_infer_replays_head_softmax_and_strict_half_tie() -> None:
    model = _ToyEffort()
    scoring, feature, logits, peak, latency = runner.infer_one(
        model,
        torch.device("cpu"),
        np.zeros((3, 224, 224), dtype=np.float32),
    )

    assert model.forward_calls == 1
    assert feature.shape == (1024,)
    assert feature.dtype == np.float32
    assert logits.tolist() == [0.0, 0.0]
    assert scoring["raw_logit_margin"] == 0.0
    assert scoring["fake_probability"] == 0.5
    assert scoring["ai_score"] == scoring["score"] == scoring["probability"]
    assert scoring["classification_decision"] is False
    assert scoring["classification"]["decision"] is False
    assert scoring["t1"]["decision"] is False
    assert scoring["manual_replay"]["head_logits_exact"] is True
    assert scoring["manual_replay"]["softmax_dtype"] == "float32"
    assert peak is None
    assert latency >= 0.0


def test_float32_softmax_and_manual_head_replay_fail_closed() -> None:
    probability = runner._float32_probability(
        torch.tensor([[0.0, np.log(3.0)]], dtype=torch.float32)
    )
    assert probability.dtype == torch.float32
    assert float(probability.item()) == pytest.approx(0.75, abs=1e-7)

    with pytest.raises(ValueError, match="not float32"):
        runner._float32_probability(
            torch.zeros((1, 2), dtype=torch.float64)
        )
    with pytest.raises(ValueError, match="head replay"):
        runner.infer_one(
            _ToyEffort(drift=0.125),
            torch.device("cpu"),
            np.zeros((3, 224, 224), dtype=np.float32),
        )


def test_canonical_selection_and_full_visibility_without_model_scores() -> None:
    release, _, rows = runner.load_release(
        REPO_ROOT,
        (REPO_ROOT / runner.DEFAULT_DATASET_MANIFEST).resolve(),
    )
    assert release["pairs"] == 275
    assert release["images"] == 550

    selected = runner.select_inputs(rows, pair_limit=1)
    assert len(selected) == 2
    assert {row["kind"] for row in selected} == {"real", "forged"}
    assert len({row["task_id"] for row in selected}) == 1
    runner.validate_selected_inputs(selected, REPO_ROOT)

    single = runner.select_inputs(
        rows,
        pair_limit=None,
        sample_id=str(selected[1]["sample_id"]),
    )
    assert single == [selected[1]]

    visibility = runner.build_pair_visibility(rows, REPO_ROOT)
    assert len(visibility) == 275
    assert {
        item["edit_visibility"] for item in visibility.values()
    } == {"full"}
    assert {
        item["edit_visible_gt_fraction"] for item in visibility.values()
    } == {1.0}
    assert all(
        item["edit_visibility_evidence"]["crop"] is None
        for item in visibility.values()
    )


def test_selection_arguments_and_incomplete_pairs_are_rejected() -> None:
    rows = [
        {
            "sample_id": "a-real",
            "task_id": "a",
            "pair_rank": 0,
            "kind": "real",
        },
        {
            "sample_id": "a-forged",
            "task_id": "a",
            "pair_rank": 0,
            "kind": "forged",
        },
    ]
    with pytest.raises(ValueError, match="mutually exclusive"):
        runner.select_inputs(rows, pair_limit=1, sample_id="a-real")
    with pytest.raises(ValueError, match="positive"):
        runner.select_inputs(rows, pair_limit=0)
    with pytest.raises(ValueError, match="incomplete"):
        runner.select_inputs(rows[:1], pair_limit=None)


def test_nested_t2_and_joint_payloads_are_rejected() -> None:
    runner._reject_t2_payload(
        {
            "valid_for_t1": True,
            "safe": [{"classification": {"decision": False}}],
        }
    )
    for key in ("pixel_ap", "attention_map", "s_joint", "joint_metrics"):
        with pytest.raises(ValueError, match=r"invents Effort T2 fields"):
            runner._reject_t2_payload(
                {"outer": [{"deeper": {key: "invented"}}]}
            )


def test_safe_components_and_artifact_paths_reject_traversal(
    tmp_path: Path,
) -> None:
    for value in ("", ".", "..", "../escape", "a/b", "a\\b", "/absolute"):
        with pytest.raises(ValueError):
            runner._safe_component(value, label="run-id")
        with pytest.raises(ValueError):
            runner._artifact_path(tmp_path, value)

    path = runner._artifact_path(tmp_path, "safe-id")
    assert path == (tmp_path / "artifacts" / "safe-id.npz").resolve()
    assert path.parent == (tmp_path / "artifacts").resolve()


def test_runtime_configuration_is_frozen_with_mocked_side_effects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    set_threads = mock.Mock()
    manual_seed = mock.Mock()
    deterministic = mock.Mock()
    precision = mock.Mock()
    monkeypatch.setattr(torch, "set_num_threads", set_threads)
    monkeypatch.setattr(torch, "manual_seed", manual_seed)
    monkeypatch.setattr(torch, "use_deterministic_algorithms", deterministic)
    monkeypatch.setattr(torch, "set_float32_matmul_precision", precision)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    device, audit = runner.configure_runtime("cpu")

    assert device == torch.device("cpu")
    set_threads.assert_called_once_with(16)
    manual_seed.assert_called_once_with(20260724)
    deterministic.assert_called_once_with(True)
    precision.assert_called_once_with("highest")
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    assert audit["deterministic_algorithms"] is True
    assert audit["cudnn_enabled"] is True
    assert audit["cudnn_benchmark"] is False
    assert audit["cudnn_deterministic"] is True
    assert audit["allow_tf32_matmul"] is False
    assert audit["allow_tf32_cudnn"] is False
    assert audit["autocast"] is False
    assert audit["batch_size"] == 1
    assert audit["seed"] == 20260724
    assert audit["cuda_available"] is False

    with pytest.raises(RuntimeError, match="unavailable"):
        runner.configure_runtime("cuda:0")


def test_runtime_version_drift_is_rejected_before_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runner, "TORCH_VERSION", "0.0.0-drift")
    with pytest.raises(ValueError, match="runtime torch changed"):
        runner.configure_runtime("cpu")


def _resume_fixture(
    tmp_path: Path,
) -> tuple[
    dict[str, object],
    dict[str, object],
    Path,
    _ToyEffort,
    torch.device,
]:
    image_path = tmp_path / "resume.png"
    rgb = np.arange(19 * 23 * 3, dtype=np.uint8).reshape(19, 23, 3)
    Image.fromarray(rgb, mode="RGB").save(image_path)
    _, preprocess = runner.preprocess_image(image_path)

    model = _ToyEffort()
    feature = np.zeros(1024, dtype=np.float32)
    logits = np.zeros(2, dtype=np.float32)
    run_dir = tmp_path / "run"
    artifact_path = runner._artifact_path(run_dir, "sample")
    runner._atomic_save_artifact(artifact_path, feature, logits)

    expected: dict[str, object] = {
        "id": "sample",
        "input_path": str(image_path.resolve()),
        "input_sha256": sha256_file(image_path),
    }
    row: dict[str, object] = {
        **expected,
        "status": "ok",
        "valid_for_metrics": True,
        "ai_score": 0.5,
        "score": 0.5,
        "probability": 0.5,
        "fake_probability": 0.5,
        "classification_decision": False,
        "preprocess": preprocess,
        "artifact_path": str(artifact_path),
        "artifact_sha256": sha256_file(artifact_path),
        "feature_array_sha256": runner._array_sha256(feature),
        "class_logits_array_sha256": runner._array_sha256(logits),
        "class_logits": [0.0, 0.0],
    }
    return expected, row, run_dir, model, torch.device("cpu")


def test_resume_row_replays_preprocess_artifact_head_and_softmax(
    tmp_path: Path,
) -> None:
    expected, row, run_dir, model, device = _resume_fixture(tmp_path)
    runner._validate_resume_row(
        row,
        expected=expected,
        repo_root=REPO_ROOT,
        run_dir=run_dir,
        model=model,
        device=device,
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda row: row.__setitem__("score", 0.5001), "aliases changed"),
        (
            lambda row: row.__setitem__("classification_decision", True),
            "strict decision changed",
        ),
        (
            lambda row: row["preprocess"].__setitem__(
                "tensor_sha256",
                "0" * 64,
            ),
            "preprocessing does not replay",
        ),
        (
            lambda row: row.__setitem__("class_logits", [0.0, 1.0]),
            "artifact content changed",
        ),
        (
            lambda row: row.__setitem__(
                "nested",
                {"audit": [{"pixel_ap": 1.0}]},
            ),
            "invents Effort T2 fields",
        ),
    ],
)
def test_resume_rejects_row_tamper(
    tmp_path: Path,
    mutation: object,
    message: str,
) -> None:
    expected, row, run_dir, model, device = _resume_fixture(tmp_path)
    tampered = copy.deepcopy(row)
    mutation(tampered)

    with pytest.raises(ValueError, match=message):
        runner._validate_resume_row(
            tampered,
            expected=expected,
            repo_root=REPO_ROOT,
            run_dir=run_dir,
            model=model,
            device=device,
        )


def test_resume_rejects_artifact_file_tamper(tmp_path: Path) -> None:
    expected, row, run_dir, model, device = _resume_fixture(tmp_path)
    artifact_path = Path(str(row["artifact_path"]))
    runner._atomic_save_artifact(
        artifact_path,
        np.ones(1024, dtype=np.float32),
        np.zeros(2, dtype=np.float32),
    )

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        runner._validate_resume_row(
            row,
            expected=expected,
            repo_root=REPO_ROOT,
            run_dir=run_dir,
            model=model,
            device=device,
        )


def test_parser_exposes_non_scoring_preflight_mode() -> None:
    args = runner._build_parser().parse_args(
        ["--preflight-only", "--device", "cpu"]
    )
    assert args.preflight_only is True
    assert args.device == "cpu"
    assert args.bootstrap_seed == 20260724
