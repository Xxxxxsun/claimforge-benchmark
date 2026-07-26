from __future__ import annotations

import hashlib
import json
import math
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from eval.opensource import run_bfree as runner
from eval.opensource.common import sha256_file


def test_frozen_release_constants() -> None:
    assert runner.MODEL_SOURCE_COMMIT == (
        "c6a9f898782fb466b29af01f21960b67415afb0e"
    )
    assert runner.OFFICIAL_ZIP["bytes"] == 321_653_488
    assert runner.OFFICIAL_ZIP["sha256"] == (
        "8230fd3f0f3a64a6403acb692ce1663718ed16f36a5a4de4a68c0d273781769f"
    )
    assert runner.CHECKPOINT["bytes"] == 346_171_370
    assert runner.CHECKPOINT["sha256"] == (
        "5948ca78f4d94e820c250d24cdf155035b4a85960443800bfe6bb7f06bffe947"
    )
    assert runner.CHECKPOINT["tensor_count"] == 177
    assert runner.CHECKPOINT["state_elements"] == 86_526_721
    assert runner.CHECKPOINT["schema_sha256"] == (
        "e4bb9ddd115309740a70235152b7376e2c8299bb90baf243809f2a5e1665f524"
    )
    assert runner.CONFIG["bytes"] == 153
    assert runner.CONFIG["sha256"] == (
        "1f0cb4988933de06a4c2427b1b5b015baa18cea7bc5223a9f54ca5e077ec8d40"
    )
    assert runner.GOLDEN_ABS_TOLERANCE == 5e-5
    assert runner.GOLDEN_RUNTIME_REGRESSION_ABS_TOLERANCE == 1e-6
    assert runner.PATCH_SIZE == runner.PATCH_STRIDE == 14
    assert runner.CROPS == runner.CROP_COUNT == 5
    assert runner.FROZEN_VISIBILITY_CENSUS["edit_visibility"] == {
        "full": 173,
        "partial": 36,
        "none": 66,
    }
    assert [case["published_raw_logit"] for case in runner.GOLDEN_CASES] == [
        -5.9374785,
        -4.441922,
        4.430519,
        3.8499813,
    ]


@pytest.mark.parametrize(
    ("width", "height", "grid", "starts", "unique", "wrapped"),
    [
        (
            835,
            1256,
            [59, 89],
            [[11, 26], [0, 0], [0, 53], [23, 53], [23, 0]],
            5,
            False,
        ),
        (
            1258,
            833,
            [89, 59],
            [[26, 11], [0, 0], [0, 23], [53, 23], [53, 0]],
            5,
            False,
        ),
        (
            1024,
            1024,
            [73, 73],
            [[18, 18], [0, 0], [0, 37], [37, 37], [37, 0]],
            5,
            False,
        ),
        (500, 900, [35, 64], [[0, 0]] * 5, 1, True),
        (504, 700, [36, 50], [[0, 7], [0, 0], [0, 14], [0, 14], [0, 0]], 3, False),
    ],
)
def test_geometry_exactly_matches_official_wrapper(
    width: int,
    height: int,
    grid: list[int],
    starts: list[list[int]],
    unique: int,
    wrapped: bool,
) -> None:
    geometry = runner.compute_preprocess_geometry(width, height)
    assert geometry["patch_grid_wh"] == grid
    assert geometry["crop_starts_patch_xy"] == starts
    assert geometry["distinct_crop_starts"] == unique
    assert geometry["replicate_wrap_applied"] is wrapped
    assert geometry["resize"]["enabled"] is False


def test_wrapping_truncates_long_dimension_to_first_36_patches() -> None:
    geometry = runner.compute_preprocess_geometry(1400, 500)
    assert geometry["replicate_wrap_applied"] is True
    assert geometry["post_wrap_patch_grid_wh"] == [36, 36]
    assert geometry["used_native_rectangles_xyxy"] == [[0, 0, 504, 490]] * 5


def test_tiny_below_one_patch_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least one 14"):
        runner.compute_preprocess_geometry(13, 100)


def test_preprocess_demo_images_match_frozen_tensor_contract() -> None:
    source = runner.DEFAULT_SOURCE_ROOT
    if not source.is_dir():
        pytest.skip("official B-Free source is not installed")
    for case in runner.GOLDEN_CASES:
        array, audit = runner.preprocess_image(
            source / "code" / "demo_images" / case["filename"]
        )
        assert audit["decoded_rgb_sha256"] == case["decoded_rgb_sha256"]
        assert audit["tensor_sha256"] == case["tensor_sha256"]
        assert audit["geometry"]["patch_grid_wh"] == case["patch_grid_wh"]
        assert (
            audit["geometry"]["crop_starts_patch_xy"]
            == case["crop_starts_patch_xy"]
        )
        assert array.dtype == np.float32
        assert array.flags.c_contiguous
        assert np.isfinite(array).all()


def test_replay_head_preserves_five_crop_raw_logit_mean() -> None:
    torch = pytest.importorskip("torch")
    head = torch.nn.Linear(768, 1, dtype=torch.float32)
    features = torch.arange(5 * 768, dtype=torch.float32).reshape(5, 768)
    official_crop_logits = head(features)
    official_mean = official_crop_logits.mean(dim=0)
    scoring = runner.replay_head(
        official_mean,
        features,
        head,
        official_crop_logits=official_crop_logits,
    )
    expected = official_crop_logits.detach().reshape(5).tolist()
    assert scoring["crop_logits"] == expected
    assert scoring["raw_logit"] == float(official_mean.item())
    assert scoring["ai_score"] == scoring["raw_logit"]
    assert scoring["classification_decision"] is (
        scoring["raw_logit"] > 0.0
    )
    assert 0.0 <= scoring["fake_probability"] <= 1.0
    assert scoring["manual_replay"]["official_mean_exact_match"] is True


def test_replay_head_rejects_wrong_feature_shape_and_nonfinite() -> None:
    torch = pytest.importorskip("torch")
    head = torch.nn.Linear(768, 1)
    with pytest.raises(ValueError, match=r"\[5,768\]"):
        runner.replay_head(torch.zeros(1), torch.zeros(1, 768), head)
    features = torch.zeros(5, 768)
    features[0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        runner.replay_head(torch.zeros(1), features, head)


def test_atomic_npz_artifact_roundtrip_has_exact_keys_and_hashes() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "artifact.npz"
        features = np.arange(5 * 768, dtype=np.float32).reshape(5, 768)
        logits = np.linspace(-1, 1, 5, dtype=np.float32)
        runner._atomic_save_artifact(path, features, logits)
        loaded_features, loaded_logits = runner._load_artifact(path)
        assert np.array_equal(loaded_features, features)
        assert np.array_equal(loaded_logits, logits)
        assert runner._array_sha256(loaded_features) == hashlib.sha256(
            features.tobytes()
        ).hexdigest()
        assert len(sha256_file(path)) == 64


def test_artifact_loader_rejects_extra_keys_and_pickle() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "bad.npz"
        np.savez(
            path,
            features=np.zeros((5, 768), dtype=np.float32),
            crop_logits=np.zeros(5, dtype=np.float32),
            extra=np.zeros(1, dtype=np.float32),
        )
        with pytest.raises(ValueError, match="keys"):
            runner._load_artifact(path)


def test_safe_component_and_output_path_escape_are_rejected() -> None:
    for value in ("", ".", "..", "../x", "a/b", "a\\b", "/abs"):
        with pytest.raises(ValueError):
            runner._safe_component(value, label="run-id")
    with tempfile.TemporaryDirectory() as temporary:
        run_dir = Path(temporary) / "run"
        run_dir.mkdir()
        with pytest.raises(ValueError, match="safe non-empty"):
            runner._artifact_path(run_dir, "../bad")


def test_source_and_installed_weight_contracts() -> None:
    if not (
        runner.DEFAULT_SOURCE_ROOT.is_dir()
        and runner.DEFAULT_WEIGHTS_DIR.is_dir()
        and runner.DEFAULT_WEIGHTS_ZIP.is_file()
    ):
        pytest.skip("official B-Free assets are not installed")
    source, assets, state = runner.verify_assets(
        source_root=runner.DEFAULT_SOURCE_ROOT,
        weights_dir=runner.DEFAULT_WEIGHTS_DIR,
        weights_zip=runner.DEFAULT_WEIGHTS_ZIP,
    )
    assert source["commit"] == runner.MODEL_SOURCE_COMMIT
    assert assets["checkpoint"]["schema"]["tensor_count"] == 177
    assert assets["checkpoint"]["schema"]["state_elements"] == 86_526_721
    assert list(state) == assets["checkpoint"]["schema"]["keys"]


def test_frozen_visibility_census_replays_without_model_scores() -> None:
    manifest = runner.DEFAULT_DATASET_MANIFEST
    if not manifest.is_file():
        pytest.skip("canonical Mouse input manifest is not installed")
    repo = Path(__file__).resolve().parents[1]
    _, _, rows = runner.load_release(repo, manifest.resolve())
    visibility = runner.build_pair_visibility(rows, repo)
    audit = runner.validate_frozen_visibility_census(visibility)
    assert audit["census"] == {"full": 173, "partial": 36, "none": 66}
    assert audit["wrapped_pairs"] == 26
    assert audit["distinct_crop_starts_census"] == {1: 26, 3: 1, 5: 248}
    assert audit["mean_edit_visible_gt_fraction"] == pytest.approx(
        0.6891766376903072
    )


def test_result_identity_is_t1_only() -> None:
    identity = runner._result_identity(
        {
            "sample_id": "abc",
            "task_id": "task",
            "pair_rank": 0,
            "rank": 0,
            "kind": "real",
            "label": 0,
            "domain": "lodging",
            "canonical_path": "image.jpg",
            "canonical_sha256": "a" * 64,
            "width": 1000,
            "height": 800,
        },
        repo_root=Path("/repo"),
        visibility={
            "edit_visibility": "full",
            "edit_visible_gt_fraction": 1.0,
            "edit_visibility_evidence": {},
        },
        config_fingerprint="f" * 64,
    )
    assert identity["valid_for_t1"] is True
    assert identity["valid_for_t2"] is False
    runner._reject_t2_payload(identity)
    bad = dict(identity)
    bad["localization"] = {"mask": "invented"}
    with pytest.raises(ValueError, match="T2"):
        runner._reject_t2_payload(bad)


def test_parser_supports_preflight_without_mouse_scoring() -> None:
    args = runner._build_parser().parse_args(["--preflight-only", "--device", "cpu"])
    assert args.preflight_only is True
    assert args.device == "cpu"


def test_runtime_freeze_keeps_official_cudnn_enabled() -> None:
    device, runtime = runner.configure_runtime("cpu")
    assert device.type == "cpu"
    assert runtime["cudnn_enabled"] is True
    assert runtime["cudnn_benchmark"] is False
    assert runtime["cudnn_deterministic"] is True
    assert runtime["cuda_matmul_allow_tf32"] is False
    assert runtime["cudnn_allow_tf32"] is False
    assert runtime["deterministic_algorithms"] is True
    assert runtime["float32_matmul_precision"] == "highest"
