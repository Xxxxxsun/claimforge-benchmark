from __future__ import annotations

import copy
import hashlib
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from eval.opensource import analyze_maskclip_balanced as analyzer
from eval.opensource.common import stable_json


def _install_fake_cv2(
    monkeypatch: pytest.MonkeyPatch,
) -> SimpleNamespace:
    calls: list[tuple[tuple[int, int], int]] = []
    module = SimpleNamespace(INTER_NEAREST=0, INTER_LINEAR=1)

    def resize(
        value: np.ndarray,
        size: tuple[int, int],
        *,
        interpolation: int,
    ) -> np.ndarray:
        source = np.asarray(value)
        width, height = size
        calls.append((size, interpolation))
        if interpolation == module.INTER_NEAREST:
            ys = np.minimum(
                np.floor(np.arange(height) * source.shape[0] / height).astype(int),
                source.shape[0] - 1,
            )
            xs = np.minimum(
                np.floor(np.arange(width) * source.shape[1] / width).astype(int),
                source.shape[1] - 1,
            )
            return source[np.ix_(ys, xs)]
        if interpolation != module.INTER_LINEAR:
            raise AssertionError("unexpected interpolation")
        source_float = source.astype(np.float64)
        source_height, source_width = source.shape[:2]
        y = (np.arange(height, dtype=np.float64) + 0.5) * source_height / height - 0.5
        x = (np.arange(width, dtype=np.float64) + 0.5) * source_width / width - 0.5
        y0 = np.floor(y).astype(int)
        x0 = np.floor(x).astype(int)
        y1 = np.clip(y0 + 1, 0, source_height - 1)
        x1 = np.clip(x0 + 1, 0, source_width - 1)
        y0 = np.clip(y0, 0, source_height - 1)
        x0 = np.clip(x0, 0, source_width - 1)
        wy = np.clip(y - y0, 0.0, 1.0)
        wx = np.clip(x - x0, 0.0, 1.0)
        top = (
            source_float[y0[:, None], x0[None, :]] * (1.0 - wx[None, :])
            + source_float[y0[:, None], x1[None, :]] * wx[None, :]
        )
        bottom = (
            source_float[y1[:, None], x0[None, :]] * (1.0 - wx[None, :])
            + source_float[y1[:, None], x1[None, :]] * wx[None, :]
        )
        return (top * (1.0 - wy[:, None]) + bottom * wy[:, None]).astype(source.dtype)

    module.resize = resize
    module.calls = calls
    monkeypatch.setitem(sys.modules, "cv2", module)
    return module


def _save_npy(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        np.save(handle, np.ascontiguousarray(value), allow_pickle=False)


def _score_fields(score: float = 0.75) -> dict:
    forged_logit = float(math.log(score / (1.0 - score)))
    real_logit = 0.0
    return {
        "class_logits": {
            "real": real_logit,
            "forged": forged_logit,
        },
        "class_probabilities": {
            "real": 1.0 - score,
            "forged": score,
        },
        "ai_score": score,
        "probability": score,
        "score": score,
        "score_margin": float(np.float32(forged_logit) - np.float32(real_logit)),
        "score_semantics": "softmax_probability_of_class_1_forged",
        "calibrated_probability": False,
        "classification_decision": score >= 0.5,
        "classification_threshold": 0.5,
        "classification_threshold_operator": ">=",
    }


def _runtime(device: str = "cpu") -> dict:
    value = {
        "device": device,
        "python": "3.12.0",
        "platform": "fixture",
        "torch": "2.7.0",
        "torchvision": "0.22.0",
        "numpy": "2.2.6",
        "opencv": "4.11.0",
        "pillow": "11.0.0",
        "imdlbenco": None,
        "timm": "1.0.0",
        "seed": 42,
        "precision": "float32",
        "batch_size": 1,
        "autocast": False,
        "deterministic_algorithms_enabled": True,
        "deterministic_algorithms_warn_only": False,
        "cublas_workspace_config": ":4096:8",
        "cudnn": {
            "benchmark": False,
            "deterministic": True,
            "allow_tf32": False,
        },
        "matmul_allow_tf32": False,
    }
    if device.startswith("cuda:"):
        value["cuda"] = {
            "runtime": "12.8",
            "device_index": int(device.removeprefix("cuda:")),
            "device_name": "fixture GPU",
            "total_memory_bytes": 1024,
            "capability": [8, 0],
        }
    return value


def _expected_row(
    sample_id: str,
    *,
    width: int,
    height: int,
    gt_kind: str,
) -> dict:
    if gt_kind == "all_zero":
        return {
            "sample_id": sample_id,
            "width": width,
            "height": height,
            "kind": "real",
            "label": 0,
            "condition": "real",
            "manipulation_scope": "authentic",
            "gt_mask_kind": "all_zero",
            "gt_mask_path": None,
            "gt_mask_sha256": None,
            "gt_positive_pixels": 0,
        }
    if gt_kind == "not_applicable":
        return {
            "sample_id": sample_id,
            "width": width,
            "height": height,
            "kind": "forged",
            "label": 1,
            "condition": "fullframe_cat",
            "manipulation_scope": "conditional_full_frame_edit",
            "gt_mask_kind": "not_applicable",
            "gt_mask_path": None,
            "gt_mask_sha256": None,
            "gt_positive_pixels": None,
        }
    raise AssertionError(f"unsupported fixture GT kind: {gt_kind}")


def _artifact_row(
    repo_root: Path,
    artifact_root: Path,
    expected: dict,
    model_map: np.ndarray,
) -> tuple[dict, analyzer.DenseArtifacts]:
    sample_id = expected["sample_id"]
    applicable = expected["gt_mask_kind"] in {"all_zero", "exact_diff"}
    model_path = artifact_root / "score_maps_model_512" / f"{sample_id}.npy"
    native_path = artifact_root / "score_maps_native" / f"{sample_id}.npy"
    mask_path = artifact_root / "masks_native" / f"{sample_id}.png"
    for directory in (
        model_path.parent,
        native_path.parent,
        mask_path.parent,
    ):
        directory.mkdir(parents=True, exist_ok=True)
    _save_npy(model_path, model_map)
    row = {
        "sample_id": sample_id,
        **_score_fields(),
        "task_scope": {
            "valid_for_t1": True,
            "valid_for_t2": applicable,
            "native_dense_output": True,
            "model_512_output_role": (
                "t2_and_diagnostic" if applicable else "diagnostic_only"
            ),
        },
        "t2_applicable": applicable,
        "t2_target_semantics": (
            "all_zero_real_false_positive_area"
            if applicable
            else "not_applicable_fullframe"
        ),
        "mask_threshold": 0.5,
        "mask_threshold_operator": ">=",
        "score_map_model_path": model_path.relative_to(repo_root).as_posix(),
        "score_map_model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "score_map_model_bytes": model_path.stat().st_size,
        "score_map_model_shape": list(model_map.shape),
        "score_map_model_dtype": "float32",
        "score_map_model_semantics": ("released_sigmoid_forged_probability"),
    }
    if not applicable:
        row.update(
            {
                field: None
                for field in (
                    *analyzer._nullable_artifact_fields("score_map_native"),
                    *analyzer._nullable_artifact_fields("mask"),
                )
            }
        )
        row["localization"] = None
        artifact = analyzer.DenseArtifacts(
            sample_id=sample_id,
            model_path=model_path,
            model_sha256=row["score_map_model_sha256"],
            model_bytes=model_path.stat().st_size,
            native_path=None,
            native_sha256=None,
            native_bytes=None,
            mask_path=None,
            mask_sha256=None,
            mask_bytes=None,
            t2_applicable=False,
            width=expected["width"],
            height=expected["height"],
        )
        return row, artifact

    native_map = analyzer._independent_restore_native(
        model_map,
        width=expected["width"],
        height=expected["height"],
    )
    _save_npy(native_path, native_map)
    mask = np.where(native_map >= 0.5, 255, 0).astype(np.uint8)
    Image.fromarray(mask, mode="L").save(mask_path, format="PNG")
    target = np.zeros(native_map.shape, dtype=bool)
    localization = {
        "model_512": analyzer._independent_pixel_metrics(
            model_map,
            np.zeros(model_map.shape, dtype=bool),
            include_ap=False,
        ),
        "native": analyzer._independent_pixel_metrics(
            native_map,
            target,
            include_ap=False,
        ),
    }
    row.update(
        {
            "score_map_native_path": native_path.relative_to(repo_root).as_posix(),
            "score_map_native_sha256": hashlib.sha256(
                native_path.read_bytes()
            ).hexdigest(),
            "score_map_native_bytes": native_path.stat().st_size,
            "score_map_native_shape": list(native_map.shape),
            "score_map_native_dtype": "float32",
            "score_map_native_semantics": (
                "model_512_probability_map_restored_opencv_inter_linear"
            ),
            "mask_path": mask_path.relative_to(repo_root).as_posix(),
            "mask_sha256": hashlib.sha256(mask_path.read_bytes()).hexdigest(),
            "mask_bytes": mask_path.stat().st_size,
            "mask_shape": list(mask.shape),
            "mask_dtype": "uint8",
            "mask_semantics": ("native_probability_map_ge_0_5_encoded_L_0_or_255"),
            "localization": localization,
        }
    )
    artifact = analyzer.DenseArtifacts(
        sample_id=sample_id,
        model_path=model_path,
        model_sha256=row["score_map_model_sha256"],
        model_bytes=model_path.stat().st_size,
        native_path=native_path,
        native_sha256=row["score_map_native_sha256"],
        native_bytes=native_path.stat().st_size,
        mask_path=mask_path,
        mask_sha256=row["mask_sha256"],
        mask_bytes=mask_path.stat().st_size,
        t2_applicable=True,
        width=expected["width"],
        height=expected["height"],
    )
    return row, artifact


def test_strict_jsonl_rejects_noncanonical_and_missing_newline(
    tmp_path: Path,
):
    path = tmp_path / "rows.jsonl"
    path.write_text('{"z": 1, "a": 2}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="not canonical"):
        analyzer._read_jsonl_strict(path, "rows")
    path.write_text(stable_json({"a": 2, "z": 1}), encoding="utf-8")
    with pytest.raises(ValueError, match="final newline"):
        analyzer._read_jsonl_strict(path, "rows")


def test_restore_is_inter_linear_and_target_resize_is_inter_nearest(
    monkeypatch: pytest.MonkeyPatch,
):
    cv2 = _install_fake_cv2(monkeypatch)
    monkeypatch.setattr(analyzer, "MODEL_SIZE", 4)
    source = np.arange(16, dtype=np.float32).reshape(4, 4) / 15.0
    actual = analyzer._independent_restore_native(
        source,
        width=7,
        height=5,
    )
    expected = cv2.resize(
        source,
        (7, 5),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.float32)
    assert np.array_equal(actual, expected)

    target = np.asarray([[False, True], [True, False]], dtype=bool)
    resized = analyzer._independent_resize_target(target)
    nearest = cv2.resize(
        target.astype(np.uint8),
        (4, 4),
        interpolation=cv2.INTER_NEAREST,
    )
    assert np.array_equal(resized, nearest > 0)
    assert cv2.calls == [
        ((7, 5), cv2.INTER_LINEAR),
        ((7, 5), cv2.INTER_LINEAR),
        ((4, 4), cv2.INTER_NEAREST),
        ((4, 4), cv2.INTER_NEAREST),
    ]


def test_inventory_validates_exact_applicable_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    _install_fake_cv2(monkeypatch)
    monkeypatch.setattr(analyzer, "MODEL_SIZE", 4)
    artifact_root = tmp_path / "outputs" / "run"
    expected = _expected_row(
        "real-sample",
        width=7,
        height=5,
        gt_kind="all_zero",
    )
    model_map = np.linspace(0.0, 1.0, 16, dtype=np.float32).reshape(4, 4)
    row, _ = _artifact_row(
        tmp_path,
        artifact_root,
        expected,
        model_map,
    )
    artifacts = analyzer.validate_artifact_inventory(
        selected=[expected],
        latest_results=[row],
        repo_root=tmp_path,
        artifact_root=artifact_root,
    )
    assert artifacts["real-sample"].t2_applicable is True
    assert artifacts["real-sample"].native_path is not None
    assert artifacts["real-sample"].mask_path is not None


def test_inventory_rejects_native_map_not_exactly_restored(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    _install_fake_cv2(monkeypatch)
    monkeypatch.setattr(analyzer, "MODEL_SIZE", 4)
    artifact_root = tmp_path / "outputs" / "run"
    expected = _expected_row(
        "real-sample",
        width=7,
        height=5,
        gt_kind="all_zero",
    )
    model_map = np.linspace(0.0, 1.0, 16, dtype=np.float32).reshape(4, 4)
    row, _ = _artifact_row(
        tmp_path,
        artifact_root,
        expected,
        model_map,
    )
    native_path = tmp_path / row["score_map_native_path"]
    native = np.load(native_path, allow_pickle=False)
    native[0, 0] = np.nextafter(native[0, 0], np.float32(1.0))
    _save_npy(native_path, native)
    row["score_map_native_sha256"] = hashlib.sha256(
        native_path.read_bytes()
    ).hexdigest()
    row["score_map_native_bytes"] = native_path.stat().st_size
    with pytest.raises(ValueError, match="not exact INTER_LINEAR"):
        analyzer.validate_artifact_inventory(
            selected=[expected],
            latest_results=[row],
            repo_root=tmp_path,
            artifact_root=artifact_root,
        )


def test_inventory_rejects_mask_not_equal_native_threshold(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    _install_fake_cv2(monkeypatch)
    monkeypatch.setattr(analyzer, "MODEL_SIZE", 4)
    artifact_root = tmp_path / "outputs" / "run"
    expected = _expected_row(
        "real-sample",
        width=7,
        height=5,
        gt_kind="all_zero",
    )
    model_map = np.linspace(0.0, 1.0, 16, dtype=np.float32).reshape(4, 4)
    row, _ = _artifact_row(
        tmp_path,
        artifact_root,
        expected,
        model_map,
    )
    mask_path = tmp_path / row["mask_path"]
    with Image.open(mask_path) as opened:
        mask = np.asarray(opened, dtype=np.uint8).copy()
    mask[0, 0] = 255 if mask[0, 0] == 0 else 0
    Image.fromarray(mask, mode="L").save(mask_path, format="PNG")
    row["mask_sha256"] = hashlib.sha256(mask_path.read_bytes()).hexdigest()
    row["mask_bytes"] = mask_path.stat().st_size
    with pytest.raises(ValueError, match=r"native score map >= 0.5"):
        analyzer.validate_artifact_inventory(
            selected=[expected],
            latest_results=[row],
            repo_root=tmp_path,
            artifact_root=artifact_root,
        )


def test_fullframe_keeps_only_diagnostic_model_map(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(analyzer, "MODEL_SIZE", 4)
    artifact_root = tmp_path / "outputs" / "run"
    expected = _expected_row(
        "fullframe-sample",
        width=7,
        height=5,
        gt_kind="not_applicable",
    )
    model_map = np.full((4, 4), 0.25, dtype=np.float32)
    row, _ = _artifact_row(
        tmp_path,
        artifact_root,
        expected,
        model_map,
    )
    artifacts = analyzer.validate_artifact_inventory(
        selected=[expected],
        latest_results=[row],
        repo_root=tmp_path,
        artifact_root=artifact_root,
    )
    artifact = artifacts["fullframe-sample"]
    assert artifact.t2_applicable is False
    assert artifact.native_path is None
    assert artifact.mask_path is None


def test_inventory_rejects_extra_artifact_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(analyzer, "MODEL_SIZE", 4)
    artifact_root = tmp_path / "outputs" / "run"
    expected = _expected_row(
        "fullframe-sample",
        width=7,
        height=5,
        gt_kind="not_applicable",
    )
    row, _ = _artifact_row(
        tmp_path,
        artifact_root,
        expected,
        np.full((4, 4), 0.25, dtype=np.float32),
    )
    _save_npy(
        artifact_root / "score_maps_native" / "unexpected.npy",
        np.zeros((5, 7), dtype=np.float32),
    )
    with pytest.raises(ValueError, match="native-map directory"):
        analyzer.validate_artifact_inventory(
            selected=[expected],
            latest_results=[row],
            repo_root=tmp_path,
            artifact_root=artifact_root,
        )


def test_inventory_rejects_extra_artifact_root_entry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(analyzer, "MODEL_SIZE", 4)
    artifact_root = tmp_path / "outputs" / "run"
    expected = _expected_row(
        "fullframe-sample",
        width=7,
        height=5,
        gt_kind="not_applicable",
    )
    row, _ = _artifact_row(
        tmp_path,
        artifact_root,
        expected,
        np.full((4, 4), 0.25, dtype=np.float32),
    )
    (artifact_root / "unexpected.txt").write_text("unexpected")
    with pytest.raises(ValueError, match="top-level inventory"):
        analyzer.validate_artifact_inventory(
            selected=[expected],
            latest_results=[row],
            repo_root=tmp_path,
            artifact_root=artifact_root,
        )


def test_score_payload_rejects_nonfinite_softmax_and_decision_drift():
    row = _score_fields()
    broken = copy.deepcopy(row)
    broken["ai_score"] = float("nan")
    with pytest.raises(ValueError, match="not finite"):
        analyzer._validate_score_payload(broken, sample_id="sample")

    broken = copy.deepcopy(row)
    broken["class_probabilities"]["forged"] = 0.70
    broken["ai_score"] = broken["probability"] = broken["score"] = 0.70
    with pytest.raises(ValueError, match="softmax"):
        analyzer._validate_score_payload(broken, sample_id="sample")

    broken = copy.deepcopy(row)
    broken["classification_decision"] = False
    with pytest.raises(ValueError, match="decision"):
        analyzer._validate_score_payload(broken, sample_id="sample")

    broken = copy.deepcopy(row)
    broken["score_semantics"] = "some_probability"
    with pytest.raises(ValueError, match="semantics"):
        analyzer._validate_score_payload(broken, sample_id="sample")


def test_runtime_evidence_is_exact_and_cuda_device_bound():
    assert (
        analyzer._validate_runtime_evidence(
            _runtime("cpu"),
            label="runtime",
        )["device"]
        == "cpu"
    )
    assert (
        analyzer._validate_runtime_evidence(
            _runtime("cuda:2"),
            label="runtime",
        )[
            "cuda"
        ]["device_index"]
        == 2
    )

    broken = _runtime("cpu")
    broken["deterministic_algorithms_enabled"] = False
    with pytest.raises(ValueError, match="deterministic_algorithms"):
        analyzer._validate_runtime_evidence(broken, label="runtime")

    broken = _runtime("cuda:2")
    broken["cuda"]["device_index"] = 1
    with pytest.raises(ValueError, match="disagrees"):
        analyzer._validate_runtime_evidence(broken, label="runtime")

    broken = _runtime("cpu")
    broken["extra"] = True
    with pytest.raises(ValueError, match="key set"):
        analyzer._validate_runtime_evidence(broken, label="runtime")


def test_smoke_comparison_ignores_identity_timing_paths_but_not_bytes(
    tmp_path: Path,
):
    left_path = tmp_path / "left.npy"
    right_path = tmp_path / "right.npy"
    payload = np.arange(8, dtype=np.float32)
    _save_npy(left_path, payload)
    _save_npy(right_path, payload)
    shared_sha = hashlib.sha256(left_path.read_bytes()).hexdigest()
    left_row = {
        "sample_id": "sample",
        "status": "ok",
        "valid_for_metrics": True,
        "run_id": "run-a",
        "run_manifest_fingerprint": "a" * 64,
        "config_fingerprint": "a" * 64,
        "completed_at": "first",
        "latency_ms": 1.0,
        "peak_cuda_memory_bytes": 10,
        "score_map_model_path": "left.npy",
        "score_map_model_sha256": shared_sha,
        "score_map_model_bytes": left_path.stat().st_size,
        **_score_fields(),
    }
    right_row = {
        **left_row,
        "run_id": "run-b",
        "run_manifest_fingerprint": "b" * 64,
        "config_fingerprint": "b" * 64,
        "completed_at": "second",
        "latency_ms": 2.0,
        "peak_cuda_memory_bytes": 20,
        "score_map_model_path": "right.npy",
    }
    left_artifact = analyzer.DenseArtifacts(
        sample_id="sample",
        model_path=left_path,
        model_sha256=shared_sha,
        model_bytes=left_path.stat().st_size,
        native_path=None,
        native_sha256=None,
        native_bytes=None,
        mask_path=None,
        mask_sha256=None,
        mask_bytes=None,
        t2_applicable=False,
        width=2,
        height=4,
    )
    right_artifact = analyzer.DenseArtifacts(
        **{
            **left_artifact.__dict__,
            "model_path": right_path,
        }
    )
    report = analyzer.compare_computational_results(
        reference_rows=[left_row],
        replay_rows=[right_row],
        reference_artifacts={"sample": left_artifact},
        replay_artifacts={"sample": right_artifact},
    )
    assert report["exact_computational_projection"] is True
    assert report["model_map_file_bytes_exact"] is True

    right_with_unknown_timing = {
        **right_row,
        "preprocess_latency_ms": 99.0,
    }
    with pytest.raises(ValueError, match="computational row differs"):
        analyzer.compare_computational_results(
            reference_rows=[left_row],
            replay_rows=[right_with_unknown_timing],
            reference_artifacts={"sample": left_artifact},
            replay_artifacts={"sample": right_artifact},
        )

    _save_npy(right_path, payload + np.float32(1.0))
    with pytest.raises(ValueError, match="model map bytes differ"):
        analyzer.compare_computational_results(
            reference_rows=[left_row],
            replay_rows=[right_row],
            reference_artifacts={"sample": left_artifact},
            replay_artifacts={"sample": right_artifact},
        )


def test_native_map_loader_rejects_fullframe_callback(tmp_path: Path):
    input_row = {"sample_id": "sample"}
    artifact_path = tmp_path / "model.npy"
    _save_npy(artifact_path, np.zeros((2, 2), dtype=np.float32))
    artifact = analyzer.DenseArtifacts(
        sample_id="sample",
        model_path=artifact_path,
        model_sha256="a" * 64,
        model_bytes=artifact_path.stat().st_size,
        native_path=None,
        native_sha256=None,
        native_bytes=None,
        mask_path=None,
        mask_sha256=None,
        mask_bytes=None,
        t2_applicable=False,
        width=2,
        height=2,
    )
    bundle = type(
        "Bundle",
        (),
        {
            "selected": (input_row,),
            "artifacts": {"sample": artifact},
        },
    )()
    callback = analyzer._native_map_loader(bundle)
    with pytest.raises(ValueError, match="non-applicable"):
        callback(input_row, {"sample_id": "sample"})


def test_persisted_custom_artifact_root_is_repo_local_and_run_bound(
    tmp_path: Path,
):
    custom = (tmp_path / "custom" / "run").resolve()
    assert (
        analyzer._expected_artifact_root(
            repo_root=tmp_path,
            run_id="run",
            persisted_artifact_root=custom,
        )
        == custom
    )
    with pytest.raises(ValueError, match="exact run-id"):
        analyzer._expected_artifact_root(
            repo_root=tmp_path,
            run_id="run",
            persisted_artifact_root=tmp_path / "custom" / "other",
        )
    with pytest.raises(ValueError, match="escapes repository"):
        analyzer._expected_artifact_root(
            repo_root=tmp_path,
            run_id="run",
            persisted_artifact_root=tmp_path.parent / "run",
        )


def test_cli_defaults_to_full_replay_outputs_and_explicit_device(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    captured: dict = {}

    def fake_analyze(**kwargs):
        captured.update(kwargs)
        return {"status": "ok"}

    monkeypatch.setattr(analyzer, "analyze", fake_analyze)
    assert (
        analyzer.main(
            [
                "--repo-root",
                str(tmp_path),
                "--results-dir",
                "results",
                "--run-id",
                "run",
                "--device",
                "cpu",
            ]
        )
        == 0
    )
    assert captured["replay"] is True
    assert captured["device_text"] == "cpu"
    assert captured["metrics_output_path"] == (
        tmp_path / "results" / "run" / "balanced250_metrics.json"
    )
    assert captured["audit_output_path"] == (
        tmp_path / "results" / "run" / "independent_audit.json"
    )

    with pytest.raises(ValueError, match="explicit cpu or cuda"):
        analyzer.main(
            [
                "--repo-root",
                str(tmp_path),
                "--results-dir",
                "results",
                "--run-id",
                "run",
                "--device",
                "cuda",
            ]
        )
