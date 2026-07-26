#!/usr/bin/env python3
"""Replay-audit a complete CNNDetection inference run.

The analyzer verifies source/checkpoint/adapter provenance, physical JSONL
history, immutable input and feature artifacts, strict score aliases, summary
recomputation, and every latest successful image by replaying the pinned model.
It rejects any T2, localization, mask, or joint-score claim because the
official detector is image-level only.  Statistical summaries are recomputed
with the benchmark's shared metric implementation; this is an inference and
artifact replay audit, not a claim of fully independent statistical code.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from eval.opensource.cnndetection_metrics import (
    summarize_cnndetection_raw_logits,
    summarize_cnndetection_results,
)
from eval.opensource.common import (
    atomic_write_json,
    read_jsonl,
    repo_relative,
    sha256_file,
    stable_json,
    utc_now,
)
from eval.opensource import run_cnndetection as runner


DEFAULT_RUN_ID = runner.DEFAULT_RUN_ID
DEFAULT_RESULTS_DIR = runner.DEFAULT_RESULTS_DIR
DEFAULT_SOURCE_ROOT = runner.DEFAULT_SOURCE_ROOT
DEFAULT_CHECKPOINT = runner.DEFAULT_CHECKPOINT

RAW_LOGIT_ABS_TOLERANCE = 1e-4
SCORE_ABS_TOLERANCE = 1e-7
FEATURE_ABS_TOLERANCE = 1e-4

_FORBIDDEN_LOCALIZATION_KEYS = frozenset(
    {
        "t2",
        "localization",
        "localisation",
        "localization_metrics",
        "localisation_metrics",
        "score_map",
        "score_map_path",
        "predicted_mask",
        "predicted_mask_path",
        "mask_path",
        "pixel_metrics",
        "pixel_auroc",
        "pixel_ap",
        "iou",
        "miou",
        "dice",
        "pixel_f1",
        "s_joint",
        "joint_score",
        "joint_metrics",
    }
)


def _anchored(path: Path, repo_root: Path) -> Path:
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


def _require_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} is not a JSON object")
    return value


def _require_finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} is not a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} is not finite")
    return result


def _require_probability(value: Any, label: str) -> float:
    result = _require_finite(value, label)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{label} falls outside [0, 1]")
    return result


def _require_sha256(value: Any, label: str) -> str:
    if not runner._valid_sha256(value):
        raise ValueError(f"{label} is not a lowercase SHA-256")
    return str(value)


def _array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(value).tobytes(order="C")
    ).hexdigest()


def _reject_localization_claims(value: Any, label: str) -> None:
    if isinstance(value, Mapping):
        forbidden = sorted(set(value) & _FORBIDDEN_LOCALIZATION_KEYS)
        if forbidden:
            raise ValueError(
                f"{label} contains unsupported localization key "
                f"{forbidden[0]!r}"
            )
        for key, nested in value.items():
            _reject_localization_claims(nested, f"{label}.{key}")
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            _reject_localization_claims(nested, f"{label}[{index}]")


def _independent_geometry(
    width: int,
    height: int,
    profile_id: str,
) -> dict[str, Any]:
    if profile_id == runner.PRIMARY_PROFILE:
        return {
            "decoded_size": [width, height],
            "resize": {"enabled": False},
            "center_crop": {"enabled": False},
            "output_size": [width, height],
            "effective_native_xyxy": [0, 0, width, height],
        }
    if profile_id != runner.PAPER_CROP_PROFILE:
        raise ValueError(f"unsupported CNNDetection profile {profile_id}")
    size = runner.CROP_SIZE
    left_pad = (size - width) // 2 if size > width else 0
    top_pad = (size - height) // 2 if size > height else 0
    right_pad = (size - width + 1) // 2 if size > width else 0
    bottom_pad = (size - height + 1) // 2 if size > height else 0
    padded_width = width + left_pad + right_pad
    padded_height = height + top_pad + bottom_pad
    crop_left = int(round((padded_width - size) / 2.0))
    crop_top = int(round((padded_height - size) / 2.0))
    native_left = crop_left - left_pad
    native_top = crop_top - top_pad
    native_right = native_left + size
    native_bottom = native_top + size
    return {
        "decoded_size": [width, height],
        "resize": {"enabled": False},
        "center_crop": {
            "enabled": True,
            "size": [size, size],
            "padding_ltrb": [
                left_pad,
                top_pad,
                right_pad,
                bottom_pad,
            ],
            "start_xy_in_padded_image": [crop_left, crop_top],
            "native_crop_xyxy": [
                native_left,
                native_top,
                native_right,
                native_bottom,
            ],
        },
        "output_size": [size, size],
        "effective_native_xyxy": [
            max(0, native_left),
            max(0, native_top),
            min(width, native_right),
            min(height, native_bottom),
        ],
    }


def independent_preprocess_image(
    path: Path,
    profile_id: str,
) -> tuple[Any, dict[str, Any]]:
    """Reimplement official preprocessing without calling runner code."""

    import torch
    from torchvision.transforms import functional as vision_functional

    with Image.open(path) as opened:
        rgb = opened.convert("RGB")
        width, height = rgb.size
        decoded = np.ascontiguousarray(np.asarray(rgb, dtype=np.uint8))
        transformed = (
            vision_functional.center_crop(
                rgb,
                [runner.CROP_SIZE, runner.CROP_SIZE],
            )
            if profile_id == runner.PAPER_CROP_PROFILE
            else rgb
        )
        if profile_id not in runner.PREPROCESS_PROFILES:
            raise ValueError(f"unsupported CNNDetection profile {profile_id}")
        output = np.ascontiguousarray(
            np.asarray(transformed, dtype=np.uint8)
        )
        tensor = vision_functional.to_tensor(transformed)
    tensor = vision_functional.normalize(
        tensor,
        runner.IMAGE_MEAN,
        runner.IMAGE_STD,
    ).contiguous()
    if tensor.dtype != torch.float32:
        raise ValueError("independent preprocessing did not produce float32")
    audit = {
        "profile": profile_id,
        "steps": list(runner.PREPROCESS_PROFILES[profile_id]["steps"]),
        "decoded_size": [width, height],
        "decoded_rgb_shape": list(decoded.shape),
        "decoded_rgb_dtype": str(decoded.dtype),
        "decoded_rgb_sha256": _array_sha256(decoded),
        "output_rgb_shape": list(output.shape),
        "output_rgb_dtype": str(output.dtype),
        "output_rgb_sha256": _array_sha256(output),
        "tensor_shape": list(tensor.shape),
        "tensor_dtype": str(tensor.numpy().dtype),
        "tensor_sha256": _array_sha256(tensor.numpy()),
        "normalization": {
            "mean": list(runner.IMAGE_MEAN),
            "std": list(runner.IMAGE_STD),
        },
        "geometry": _independent_geometry(width, height, profile_id),
    }
    return tensor, audit


def _independent_infer(
    model: Any,
    tensor: Any,
    device: Any,
) -> tuple[float, float, bool, np.ndarray]:
    import torch

    captured: list[Any] = []

    def capture(_module: Any, arguments: tuple[Any, ...]) -> None:
        captured.append(arguments[0].detach())

    hook = model.fc.register_forward_pre_hook(capture)
    try:
        with torch.inference_mode():
            output = model(
                tensor.unsqueeze(0).to(
                    device=device,
                    dtype=torch.float32,
                    non_blocking=False,
                )
            )
    finally:
        hook.remove()
    if list(output.shape) != [1, 1] or len(captured) != 1:
        raise ValueError("independent CNNDetection replay shape changed")
    logit = float(output.reshape(()).item())
    score = float(torch.sigmoid(output).reshape(()).item())
    feature = np.ascontiguousarray(
        captured[0].squeeze(0).detach().cpu().numpy(),
        dtype=np.float32,
    )
    if (
        not math.isfinite(logit)
        or not 0.0 <= score <= 1.0
        or feature.shape != (runner.FEATURE_DIMENSION,)
        or not np.isfinite(feature).all()
    ):
        raise ValueError("independent CNNDetection replay is invalid")
    return logit, score, score > runner.CLASSIFICATION_THRESHOLD, feature


def _compare_score_fields(
    row: Mapping[str, Any],
    *,
    replay_logit: float,
    replay_score: float,
    replay_decision: bool,
) -> None:
    raw_logit = _require_finite(row.get("raw_logit"), "result raw_logit")
    score = _require_probability(row.get("ai_score"), "result ai_score")
    if not math.isclose(
        raw_logit,
        replay_logit,
        rel_tol=0.0,
        abs_tol=RAW_LOGIT_ABS_TOLERANCE,
    ):
        raise ValueError("CNNDetection replay raw logit mismatch")
    if not math.isclose(
        score,
        replay_score,
        rel_tol=0.0,
        abs_tol=SCORE_ABS_TOLERANCE,
    ):
        raise ValueError("CNNDetection replay score mismatch")
    for key in ("probability", "score"):
        if _require_probability(row.get(key), f"result {key}") != score:
            raise ValueError(f"CNNDetection result {key} alias mismatch")
    if row.get("score_semantics") != (
        "official_float32_sigmoid_uncalibrated_fake_score"
    ):
        raise ValueError("CNNDetection score semantics changed")
    if row.get("calibrated_probability") is not False:
        raise ValueError("CNNDetection score incorrectly marked calibrated")
    if row.get("classification_threshold") != 0.5:
        raise ValueError("CNNDetection released threshold changed")
    if row.get("classification_threshold_operator") != ">":
        raise ValueError("CNNDetection threshold operator changed")
    decision = score > 0.5
    if row.get("classification_decision") is not decision:
        raise ValueError("CNNDetection persisted decision is inconsistent")
    if replay_decision is not decision:
        raise ValueError("CNNDetection replay decision mismatch")
    for nested_key in ("classification", "t1"):
        nested = _require_mapping(
            row.get(nested_key),
            f"result {nested_key}",
        )
        for key, expected in (
            ("raw_logit", raw_logit),
            ("probability", score),
            ("ai_score", score),
            ("score", score),
            ("threshold", 0.5),
            ("threshold_operator", ">"),
            ("decision", decision),
        ):
            if nested.get(key) != expected:
                raise ValueError(
                    f"CNNDetection {nested_key}.{key} alias mismatch"
                )
        if nested.get("semantics") != row["score_semantics"]:
            raise ValueError(
                f"CNNDetection {nested_key}.semantics alias mismatch"
            )
    if row["t1"].get("policy") != (
        "official_CNNDetection_float32_sigmoid_strict_gt_0_5"
    ):
        raise ValueError("CNNDetection T1 policy changed")
    replay = _require_mapping(
        row.get("manual_replay"),
        "result manual_replay",
    )
    for key, expected in (
        ("raw_logit", raw_logit),
        ("probability", score),
        ("ai_score", score),
        ("classification_decision", decision),
        ("model_forward_calls", 1),
        ("fc_hook_calls", 1),
        ("official_logit_exact_match", True),
        ("official_score_exact_match", True),
    ):
        if replay.get(key) != expected:
            raise ValueError(
                f"CNNDetection manual_replay.{key} mismatch"
            )


def _latest_by_id(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows):
        row_id = row.get("id")
        if not isinstance(row_id, str) or not row_id:
            raise ValueError(f"physical result row {index} has invalid id")
        latest[row_id] = row
    return latest


def _verify_provenance(
    *,
    manifest: dict[str, Any],
    repo_root: Path,
    source_root: Path,
    checkpoint_path: Path,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    Any,
    Any,
    dict[str, Any],
]:
    if manifest.get("schema_version") != "cnndetection_run_manifest_v1":
        raise ValueError("unsupported CNNDetection run manifest schema")
    config = _require_mapping(manifest.get("config"), "manifest config")
    if manifest.get("config_fingerprint") != hashlib.sha256(
        stable_json(config).encode("utf-8")
    ).hexdigest():
        raise ValueError("CNNDetection config fingerprint mismatch")
    current_adapter = runner.adapter_contract(repo_root)
    if config.get("adapter") != current_adapter:
        raise ValueError("CNNDetection adapter source changed since run")
    expected_constants = {
        "model": runner.MODEL_NAME,
        "model_slug": runner.MODEL_SLUG,
        "source_commit": runner.MODEL_SOURCE_COMMIT,
        "checkpoint_id": runner.CHECKPOINT["id"],
        "checkpoint_sha256": runner.CHECKPOINT["sha256"],
        "checkpoint_bytes": runner.CHECKPOINT["bytes"],
    }
    for key, expected in expected_constants.items():
        if config.get(key) != expected:
            raise ValueError(f"CNNDetection manifest config {key} changed")
    profile_id = config.get("preprocess_profile")
    if profile_id not in runner.PREPROCESS_PROFILES:
        raise ValueError("CNNDetection manifest profile is unsupported")
    if config.get("preprocess_profile_contract") != (
        runner.PREPROCESS_PROFILES[profile_id]
    ):
        raise ValueError("CNNDetection preprocess profile contract changed")
    if config.get("no_test_time_blur_or_jpeg") is not True:
        raise ValueError("CNNDetection manifest permits test augmentation")
    source, asset, state, module = runner.verify_assets(
        source_root=source_root,
        checkpoint_path=checkpoint_path,
    )
    if manifest.get("source") != source or manifest.get("asset") != asset:
        raise ValueError("CNNDetection manifest asset provenance changed")
    _reject_localization_claims(manifest, "manifest")
    return source, asset, state, module, config


def _compare_summary(
    *,
    stored: dict[str, Any],
    recomputed: dict[str, Any],
) -> None:
    ignored = {
        "run_id",
        "model",
        "model_slug",
        "checkpoint_id",
        "preprocess_profile",
        "config_fingerprint",
        "generated_at",
    }
    stored_core = {
        key: value for key, value in stored.items() if key not in ignored
    }
    if stored_core != recomputed:
        raise ValueError("CNNDetection stored summary does not recompute")


def _require_complete_replay_target(
    *,
    manifest: Mapping[str, Any],
    summary: Mapping[str, Any],
    expected_images: int,
) -> None:
    """Reject incomplete, missing, or currently errored runs."""

    if manifest.get("status") != "complete":
        raise ValueError(
            "CNNDetection replay audit requires manifest status complete"
        )
    coverage = _require_mapping(
        summary.get("coverage"),
        "recomputed summary coverage",
    )
    required = {
        "expected_images": expected_images,
        "result_images": expected_images,
        "valid_images": expected_images,
        "error_images": 0,
        "missing_images": 0,
        "coverage_fraction": 1.0,
        "valid_fraction": 1.0,
        "is_complete": True,
    }
    for key, expected in required.items():
        if coverage.get(key) != expected:
            raise ValueError(
                "CNNDetection replay audit requires complete successful "
                f"coverage; {key}={coverage.get(key)!r}, expected "
                f"{expected!r}"
            )


def analyze(
    *,
    repo_root: Path,
    results_dir: Path,
    run_id: str,
    source_root: Path,
    checkpoint_path: Path,
    device_text: str,
    output_path: Path | None = None,
) -> dict[str, Any]:
    run_dir = results_dir / run_id
    manifest_path = run_dir / "manifest.json"
    results_path = run_dir / "results.jsonl"
    expected_path = run_dir / "expected_inputs.jsonl"
    summary_path = run_dir / "summary.json"
    for path in (manifest_path, results_path, expected_path, summary_path):
        if not path.is_file():
            raise FileNotFoundError(f"missing CNNDetection run artifact: {path}")
    manifest = _require_mapping(
        json.loads(manifest_path.read_text(encoding="utf-8")),
        "run manifest",
    )
    if manifest.get("run_id") != run_id:
        raise ValueError("CNNDetection manifest run ID mismatch")
    if manifest.get("repo_root") != str(repo_root.resolve()):
        raise ValueError("CNNDetection manifest repository root mismatch")
    if manifest.get("status") != "complete":
        raise ValueError(
            "CNNDetection replay audit requires manifest status complete"
        )
    summary = _require_mapping(
        json.loads(summary_path.read_text(encoding="utf-8")),
        "run summary",
    )
    rows = read_jsonl(results_path)
    expected = read_jsonl(expected_path)
    outputs = _require_mapping(manifest.get("outputs"), "manifest outputs")
    if outputs.get("results_sha256") != sha256_file(results_path):
        raise ValueError("CNNDetection results artifact hash mismatch")
    if outputs.get("summary_sha256") != sha256_file(summary_path):
        raise ValueError("CNNDetection summary artifact hash mismatch")
    expected_output_paths = {
        "results_path": repo_relative(results_path, repo_root),
        "summary_path": repo_relative(summary_path, repo_root),
        "feature_dir": repo_relative(run_dir / "features", repo_root),
    }
    for key, expected_value in expected_output_paths.items():
        if outputs.get(key) != expected_value:
            raise ValueError(f"CNNDetection manifest output {key} changed")
    feature_file_count = sum(
        1 for path in (run_dir / "features").glob("*.npy") if path.is_file()
    )
    if outputs.get("feature_files") != feature_file_count:
        raise ValueError("CNNDetection manifest feature-file count changed")
    dataset = _require_mapping(manifest.get("dataset"), "manifest dataset")
    if dataset.get("expected_inputs_sha256") != sha256_file(expected_path):
        raise ValueError("CNNDetection expected-input hash mismatch")

    source, asset, state, module, config = _verify_provenance(
        manifest=manifest,
        repo_root=repo_root,
        source_root=source_root,
        checkpoint_path=checkpoint_path,
    )
    canonical_manifest_value = dataset.get("manifest_path")
    canonical_manifest_digest = _require_sha256(
        dataset.get("manifest_sha256"),
        "canonical dataset manifest SHA-256",
    )
    if not isinstance(canonical_manifest_value, str):
        raise ValueError("manifest has no canonical dataset manifest path")
    canonical_manifest_path = _anchored(
        Path(canonical_manifest_value),
        repo_root,
    )
    if sha256_file(canonical_manifest_path) != canonical_manifest_digest:
        raise ValueError("canonical dataset manifest hash mismatch")
    release, canonical_inputs_path, all_inputs = runner.load_release(
        repo_root,
        canonical_manifest_path,
    )
    if dataset.get("inputs_path") != repo_relative(
        canonical_inputs_path,
        repo_root,
    ):
        raise ValueError("canonical inputs path changed")
    if dataset.get("inputs_sha256") != sha256_file(canonical_inputs_path):
        raise ValueError("canonical inputs hash changed")
    config_dataset = _require_mapping(
        config.get("dataset"),
        "manifest config dataset",
    )
    if config_dataset.get("inputs_sha256") != release["inputs_sha256"]:
        raise ValueError("config canonical inputs hash changed")
    selected = runner.select_inputs(
        all_inputs,
        config_dataset.get("pair_limit"),
        config_dataset.get("sample_id"),
    )
    if selected != expected:
        raise ValueError("expected-input snapshot is not canonical selection")
    if config_dataset.get("selected_ids") != [
        str(row["sample_id"]) for row in selected
    ]:
        raise ValueError("config selected ID order changed")
    selected_rows_sha256 = hashlib.sha256(
        "".join(f"{stable_json(row)}\n" for row in selected).encode("utf-8")
    ).hexdigest()
    if config_dataset.get("selected_rows_sha256") != selected_rows_sha256:
        raise ValueError("config selected-row fingerprint changed")
    runner.validate_selected_inputs(selected, repo_root)
    visibility = runner.build_pair_visibility(
        all_inputs,
        repo_root,
        str(config["preprocess_profile"]),
    )
    bootstrap_samples = int(config["metrics"]["bootstrap_samples"])
    bootstrap_seed = int(config["metrics"]["bootstrap_seed"])
    recomputed = summarize_cnndetection_results(
        rows,
        expected,
        threshold=runner.CLASSIFICATION_THRESHOLD,
        bootstrap_samples=bootstrap_samples,
        seed=bootstrap_seed,
    )
    recomputed["raw_logit_diagnostic"] = (
        summarize_cnndetection_raw_logits(rows, expected)
    )
    expected_summary_metadata = {
        "run_id": run_id,
        "model": runner.MODEL_NAME,
        "model_slug": runner.MODEL_SLUG,
        "checkpoint_id": runner.CHECKPOINT["id"],
        "preprocess_profile": config["preprocess_profile"],
        "config_fingerprint": manifest["config_fingerprint"],
    }
    for key, expected_value in expected_summary_metadata.items():
        if summary.get(key) != expected_value:
            raise ValueError(f"CNNDetection summary {key} changed")
    _compare_summary(stored=summary, recomputed=recomputed)
    _require_complete_replay_target(
        manifest=manifest,
        summary=recomputed,
        expected_images=len(expected),
    )

    latest = _latest_by_id(rows)
    expected_by_id = {str(row["sample_id"]): row for row in expected}
    if set(latest) != set(expected_by_id):
        raise ValueError(
            "CNNDetection replay audit requires exactly all expected IDs"
        )
    device, replay_runtime = runner.configure_runtime(device_text)
    model = runner.load_model(module=module, state=state, device=device)
    replayed = 0
    feature_replay_max_abs = 0.0
    try:
        for sample_id, expected_row in expected_by_id.items():
            row = latest.get(sample_id)
            if row is None or row.get("status") != "ok":
                continue
            _reject_localization_claims(row, f"result {sample_id}")
            expected_visibility = visibility[str(expected_row["task_id"])]
            expected_input_path = _anchored(
                Path(str(expected_row["canonical_path"])),
                repo_root,
            )
            for key, value in (
                ("schema_version", "cnndetection_result_v1"),
                ("id", sample_id),
                ("sample_id", sample_id),
                ("rank", int(expected_row["rank"])),
                ("pair_rank", int(expected_row["pair_rank"])),
                ("task_id", expected_row["task_id"]),
                ("kind", expected_row["kind"]),
                ("label", expected_row["label"]),
                ("domain", expected_row["domain"]),
                (
                    "candidate",
                    str(expected_row.get("candidate", "mouse")),
                ),
                ("dataset_id", str(expected_row.get("dataset_id"))),
                (
                    "input_path",
                    repo_relative(expected_input_path, repo_root),
                ),
                ("input_sha256", expected_row["canonical_sha256"]),
                ("input_width", int(expected_row["width"])),
                ("input_height", int(expected_row["height"])),
                ("preprocess_profile", config["preprocess_profile"]),
                ("checkpoint_id", runner.CHECKPOINT["id"]),
                ("config_fingerprint", manifest["config_fingerprint"]),
                (
                    "edit_visibility",
                    expected_visibility["edit_visibility"],
                ),
                (
                    "edit_visible_gt_fraction",
                    expected_visibility["edit_visible_gt_fraction"],
                ),
                (
                    "edit_visibility_evidence",
                    expected_visibility,
                ),
                (
                    "task_scope",
                    {
                        "valid_for_t1": True,
                        "valid_for_t2": False,
                        "native_dense_output": False,
                    },
                ),
            ):
                if row.get(key) != value:
                    raise ValueError(
                        f"CNNDetection result {sample_id} {key} mismatch"
                    )
            input_path = expected_input_path
            if sha256_file(input_path) != expected_row["canonical_sha256"]:
                raise ValueError(
                    f"CNNDetection result {sample_id} input hash changed"
                )
            tensor, preprocess = independent_preprocess_image(
                input_path,
                str(config["preprocess_profile"]),
            )
            if row.get("preprocess") != preprocess:
                raise ValueError(
                    f"CNNDetection result {sample_id} preprocessing mismatch"
                )
            logit, score, decision, replay_feature = _independent_infer(
                model,
                tensor,
                device,
            )
            _compare_score_fields(
                row,
                replay_logit=logit,
                replay_score=score,
                replay_decision=decision,
            )
            feature_value = row.get("cnndetection_feature_path")
            feature_digest = _require_sha256(
                row.get("cnndetection_feature_sha256"),
                f"result {sample_id} feature SHA-256",
            )
            if not isinstance(feature_value, str):
                raise ValueError(
                    f"result {sample_id} feature path is invalid"
                )
            feature_path = _anchored(Path(feature_value), repo_root)
            if sha256_file(feature_path) != feature_digest:
                raise ValueError(
                    f"CNNDetection result {sample_id} feature hash changed"
                )
            feature = np.load(feature_path, allow_pickle=False)
            if feature.shape != (runner.FEATURE_DIMENSION,):
                raise ValueError(
                    f"CNNDetection result {sample_id} feature shape changed"
                )
            if feature.dtype != np.float32 or not np.isfinite(feature).all():
                raise ValueError(
                    f"CNNDetection result {sample_id} feature is invalid"
                )
            if row.get("cnndetection_feature_shape") != [
                runner.FEATURE_DIMENSION
            ]:
                raise ValueError(
                    f"CNNDetection result {sample_id} feature metadata shape "
                    "changed"
                )
            if row.get("cnndetection_feature_dtype") != "float32":
                raise ValueError(
                    f"CNNDetection result {sample_id} feature metadata dtype "
                    "changed"
                )
            if row.get("cnndetection_feature_semantics") != (
                "official_fc_input_after_adaptive_global_average_pool"
            ):
                raise ValueError(
                    f"CNNDetection result {sample_id} feature semantics changed"
                )
            difference = float(
                np.max(np.abs(feature.astype(np.float64) - replay_feature))
            )
            feature_replay_max_abs = max(
                feature_replay_max_abs,
                difference,
            )
            if difference > FEATURE_ABS_TOLERANCE:
                raise ValueError(
                    f"CNNDetection result {sample_id} feature replay mismatch"
                )
            import torch

            with torch.inference_mode():
                saved_feature = torch.from_numpy(feature).to(
                    device=device,
                    dtype=torch.float32,
                )
                saved_logit = float(
                    torch.nn.functional.linear(
                        saved_feature.unsqueeze(0),
                        model.fc.weight,
                        model.fc.bias,
                    ).reshape(()).item()
                )
            if not math.isclose(
                saved_logit,
                float(row["raw_logit"]),
                rel_tol=0.0,
                abs_tol=RAW_LOGIT_ABS_TOLERANCE,
            ):
                raise ValueError(
                    f"CNNDetection result {sample_id} saved feature/logit "
                    "mismatch"
                )
            replayed += 1
    finally:
        del model
        del state
        gc.collect()
        if device.type == "cuda":
            __import__("torch").cuda.empty_cache()
    if replayed != len(expected):
        raise ValueError(
            "CNNDetection replay audit did not replay every expected image: "
            f"{replayed} != {len(expected)}"
        )

    report = {
        "schema_version": "cnndetection_replay_audit_v1",
        "status": "replay_audit_passed",
        "run_id": run_id,
        "audited_at": utc_now(),
        "source_commit": source["commit"],
        "checkpoint_sha256": asset["sha256"],
        "preprocess_profile": config["preprocess_profile"],
        "physical_result_rows": len(rows),
        "latest_result_rows": len(latest),
        "expected_images": len(expected),
        "successful_images_replayed": replayed,
        "feature_replay_max_abs_difference": feature_replay_max_abs,
        "raw_logit_abs_tolerance": RAW_LOGIT_ABS_TOLERANCE,
        "score_abs_tolerance": SCORE_ABS_TOLERANCE,
        "feature_abs_tolerance": FEATURE_ABS_TOLERANCE,
        "replay_runtime": replay_runtime,
        "audit_scope": {
            "source_checkpoint_artifact_replay": True,
            "every_expected_image_inference_replayed": True,
            "summary_recomputed_with_shared_metrics": True,
            "fully_independent_statistical_implementation": False,
        },
        "summary_recomputed_with_shared_metrics": True,
        "localization_claims_rejected": True,
        "manifest_sha256": sha256_file(manifest_path),
        "results_sha256": sha256_file(results_path),
        "summary_sha256": sha256_file(summary_path),
        "expected_inputs_sha256": sha256_file(expected_path),
    }
    if output_path is not None:
        atomic_write_json(output_path, report)
    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    repo_root = args.repo_root.resolve()
    results_dir = _anchored(args.results_dir, repo_root)
    source_root = _anchored(args.source_root, repo_root)
    checkpoint_path = _anchored(args.checkpoint, repo_root)
    output = (
        _anchored(args.output, repo_root)
        if args.output is not None
        else results_dir / args.run_id / "independent_audit.json"
    )
    report = analyze(
        repo_root=repo_root,
        results_dir=results_dir,
        run_id=args.run_id,
        source_root=source_root,
        checkpoint_path=checkpoint_path,
        device_text=args.device,
        output_path=output,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
