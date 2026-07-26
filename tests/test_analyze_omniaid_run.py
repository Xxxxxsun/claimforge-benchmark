from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn

from eval.opensource import analyze_omniaid_run as analyzer
from eval.opensource.common import (
    atomic_write_json,
    atomic_write_jsonl,
    read_jsonl,
    sha256_file,
)
from eval.opensource.omniaid_metrics import summarize_omniaid_results


class _TinyOmniAID(nn.Module):
    """Small deterministic substitute for independent full-model forwards."""

    def __init__(self) -> None:
        super().__init__()
        self.feature_extractor = _TinyFeatureExtractor()
        self.gating_network = _TinyRouter()
        self.head = nn.Linear(1024, 2, dtype=torch.float32)
        with torch.no_grad():
            self.head.weight.zero_()
            self.head.bias.zero_()
            self.head.weight[0, 0] = -1.0
            self.head.weight[1, 0] = 1.0

    def forward(
        self,
        image: torch.Tensor,
        manual_weights: object = None,
    ) -> dict[str, torch.Tensor]:
        assert manual_weights is None
        routing = self.feature_extractor(image).pooler_output
        routed = self.gating_network(routing)
        feature = routing
        logits = self.head(feature)
        final_gates = torch.zeros(
            (image.shape[0], analyzer.EXPERT_COUNT),
            dtype=torch.float32,
            device=image.device,
        )
        final_gates.scatter_(
            1,
            routed["top_k_indices"],
            routed["top_k_gates"],
        )
        final_gates[:, analyzer.ARTIFACT_EXPERT_INDEX] = 1.0
        return {
            "prob": torch.softmax(logits, dim=1)[:, 1],
            "final_gates": final_gates,
        }


class _TinyFeatureExtractor(nn.Module):
    def forward(self, image: torch.Tensor) -> types.SimpleNamespace:
        mean = image.mean(dim=(1, 2, 3))
        feature = torch.zeros(
            (image.shape[0], 1024),
            dtype=torch.float32,
            device=image.device,
        )
        feature[:, 0] = mean
        return types.SimpleNamespace(pooler_output=feature)


class _TinyRouter(nn.Module):
    def forward(self, feature: torch.Tensor) -> dict[str, torch.Tensor]:
        batch = feature.shape[0]
        return {
            "top_k_indices": torch.tensor(
                [[0, 2]],
                dtype=torch.int64,
                device=feature.device,
            ).expand(batch, -1),
            "top_k_gates": torch.tensor(
                [[0.6997385025024414, 0.3002614378929138]],
                dtype=torch.float32,
                device=feature.device,
            ).expand(batch, -1),
        }


@dataclass
class _TinyRun:
    repo_root: Path
    run_dir: Path
    output_path: Path
    expected_path: Path
    results_path: Path
    summary_path: Path
    manifest_path: Path
    expected: list[dict]
    rows: list[dict]
    model: _TinyOmniAID


def _write_artifact(
    path: Path,
    feature: np.ndarray,
    logits: np.ndarray,
    **extras: np.ndarray,
) -> None:
    arrays = {
        "pooler_output": np.ascontiguousarray(feature, dtype=feature.dtype),
        "class_logits": np.ascontiguousarray(logits, dtype=logits.dtype),
        "routing_feature": np.ascontiguousarray(
            extras.pop("routing_feature", feature.copy())
        ),
        "semantic_top_k_indices": np.ascontiguousarray(
            extras.pop(
                "semantic_top_k_indices",
                np.asarray([0, 2], dtype=np.int64),
            )
        ),
        "semantic_top_k_gates": np.ascontiguousarray(
            extras.pop(
                "semantic_top_k_gates",
                np.asarray([0.25, 0.75], dtype=np.float32),
            )
        ),
        "final_gates": np.ascontiguousarray(
            extras.pop(
                "final_gates",
                np.asarray(
                    [0.25, 0.0, 0.75, 0.0, 0.0, 1.0],
                    dtype=np.float32,
                ),
            )
        ),
        **extras,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)


def _build_tiny_run(tmp_path: Path) -> _TinyRun:
    repo_root = tmp_path / "repo"
    run_dir = repo_root / "results" / "tiny-omniaid"
    image_dir = repo_root / "inputs"
    artifact_dir = run_dir / "artifacts"
    image_dir.mkdir(parents=True)
    artifact_dir.mkdir(parents=True)

    model = _TinyOmniAID().eval()
    expected: list[dict] = []
    rows: list[dict] = []
    source = {
        "github": {"commit": analyzer.FROZEN_SOURCE_COMMIT},
        "space": {"commit": analyzer.FROZEN_SPACE_COMMIT},
    }
    assets = {
        "checkpoint": {
            "sha256": analyzer.FROZEN_CHECKPOINT_SHA256,
        }
    }
    model_audit = {
        "strict_load": True,
        "state_entries": analyzer.FROZEN_CHECKPOINT_TENSORS,
        "svd_modules": analyzer.SVD_MODULES,
    }
    runtime = {"device": "cpu"}
    runtime_golden = {
        "status": "passed",
        "kind": "synthetic_non_mouse_test_fixture",
        "mouse_model_scores_computed": 0,
    }
    config = {
        "model": "OmniAID",
        "model_slug": analyzer.FROZEN_MODEL_SLUG,
        "source_commit": analyzer.FROZEN_SOURCE_COMMIT,
        "checkpoint": {
            "id": analyzer.FROZEN_CHECKPOINT_ID,
            "bytes": analyzer.FROZEN_CHECKPOINT_BYTES,
            "sha256": analyzer.FROZEN_CHECKPOINT_SHA256,
            "tensor_count": analyzer.FROZEN_CHECKPOINT_TENSORS,
            "state_elements": analyzer.FROZEN_CHECKPOINT_ELEMENTS,
            "ordered_key_sha256": analyzer.FROZEN_ORDERED_KEY_SHA256,
            "schema_sha256": analyzer.FROZEN_SCHEMA_SHA256,
            "unsafe_globals_allowlisted": ["argparse.Namespace"],
        },
        "omniaid_config": {
            "bytes": analyzer.FROZEN_CONFIG_BYTES,
            "sha256": analyzer.FROZEN_CONFIG_SHA256,
        },
        "dinov3_base": {
            "model_id": analyzer.DINO_MODEL_ID,
            "revision": analyzer.DINO_REVISION,
            "architecture_contract": analyzer.DINO_ARCHITECTURE,
        },
        "preprocess_profile": analyzer.FROZEN_PREPROCESS_PROFILE,
        "preprocess_contract": {
            "decode": "PIL.Image.open_convert_RGB_no_EXIF_transpose",
            "resize": (
                "torchvision_Resize_list_448x448_PIL_BILINEAR_" "no_aspect_preservation"
            ),
            "to_tensor": "torchvision_ToTensor_float32_divide_255",
            "mean": list(analyzer.IMAGENET_MEAN),
            "std": list(analyzer.IMAGENET_STD),
            "face_alignment": False,
            "router_mode": "Auto (Router)",
            "manual_weights": None,
        },
        "score_semantics": analyzer.FROZEN_SCORE_SEMANTICS,
        "classification_threshold": analyzer.THRESHOLD,
        "classification_threshold_operator": ">",
        "source": source,
        "assets": assets,
        "model_audit": model_audit,
        "runtime_golden": runtime_golden,
        "runtime": runtime,
        "adapter": {},
        "release": analyzer.CANONICAL_RELEASE,
        "selected_sample_ids": ["tiny-real", "tiny-forged"],
        "selected_tasks": ["tiny-task"],
        "checkpoint_and_protocol_frozen_before_mouse_scores": True,
        "bootstrap_samples": 5,
        "bootstrap_seed": 20260724,
    }
    config_fingerprint = analyzer._fingerprint(config)

    specifications = (
        ("tiny-real", "real", 0, 0),
        ("tiny-forged", "forged", 1, 255),
    )
    for rank, (sample_id, kind, label, level) in enumerate(specifications):
        rgb = np.full((9, 13, 3), level, dtype=np.uint8)
        image_path = image_dir / f"{sample_id}.png"
        Image.fromarray(rgb, mode="RGB").save(image_path)
        canonical_path = str(image_path.relative_to(repo_root))
        expected_row = {
            "sample_id": sample_id,
            "task_id": "tiny-task",
            "pair_rank": 0,
            "rank": rank,
            "kind": kind,
            "label": label,
            "domain": "synthetic",
            "canonical_path": canonical_path,
            "canonical_sha256": sha256_file(image_path),
            "width": 13,
            "height": 9,
            "gt_positive_pixels": 0 if kind == "real" else 1,
        }
        expected.append(expected_row)

        image, preprocess = analyzer._preprocess(image_path)
        arrays, replay = analyzer._fresh_forward(
            model,
            torch.device("cpu"),
            image,
        )
        feature = arrays["pooler_output"]
        logits = arrays["class_logits"]
        artifact_path = artifact_dir / f"{sample_id}.npz"
        _write_artifact(
            artifact_path,
            feature,
            logits,
            routing_feature=arrays["routing_feature"],
            semantic_top_k_indices=arrays["semantic_top_k_indices"],
            semantic_top_k_gates=arrays["semantic_top_k_gates"],
            final_gates=arrays["final_gates"],
        )
        score = float(replay["score"])
        visibility = {
            "definition": (
                "whole_canvas_direct_resize_to_448_all_native_pixels_"
                "within_geometric_input_domain"
            ),
            "native_width": 13,
            "native_height": 9,
            "gt_positive_pixels": 1,
            "model_input_wh": [analyzer.INPUT_SIZE, analyzer.INPUT_SIZE],
            "resize": ("torchvision_PIL_bilinear_direct_aspect_ratio_distortion"),
            "crop": None,
        }
        array_hashes = {
            key: analyzer._array_sha256(value) for key, value in arrays.items()
        }
        relative_artifact = str(artifact_path.relative_to(repo_root))
        row = {
            "id": sample_id,
            "sample_id": sample_id,
            "task_id": "tiny-task",
            "pair_rank": 0,
            "rank": rank,
            "kind": kind,
            "label": label,
            "domain": "synthetic",
            "input_path": canonical_path,
            "input_sha256": sha256_file(image_path),
            "input_width": 13,
            "input_height": 9,
            "model": "OmniAID",
            "model_slug": analyzer.FROZEN_MODEL_SLUG,
            "checkpoint_id": analyzer.FROZEN_CHECKPOINT_ID,
            "preprocess_profile": analyzer.FROZEN_PREPROCESS_PROFILE,
            "score_semantics": analyzer.FROZEN_SCORE_SEMANTICS,
            "classification_threshold": analyzer.THRESHOLD,
            "classification_threshold_operator": ">",
            "config_fingerprint": config_fingerprint,
            "edit_visibility": "full",
            "edit_visible_gt_fraction": 1.0,
            "edit_visibility_evidence": visibility,
            "valid_for_t1": True,
            "valid_for_t2": False,
            "status": "ok",
            "valid_for_metrics": True,
            "preprocess": preprocess,
            "artifact_path": relative_artifact,
            "artifact_sha256": sha256_file(artifact_path),
            "artifact_keys": list(analyzer.ARTIFACT_SCHEMA),
            "artifact_paths": {"omniaid_npz": relative_artifact},
            "artifact_array_sha256": array_hashes,
            "feature_shape": [1024],
            "feature_dtype": "float32",
            "feature_semantics": analyzer.FEATURE_SEMANTICS,
            "feature_array_sha256": array_hashes["pooler_output"],
            "class_logits_shape": [2],
            "class_logits_dtype": "float32",
            "class_logits_array_sha256": array_hashes["class_logits"],
            "routing_feature_shape": [1024],
            "routing_feature_dtype": "float32",
            "routing_feature_semantics": analyzer.ROUTING_FEATURE_SEMANTICS,
            "semantic_top_k_indices_shape": [2],
            "semantic_top_k_gates_shape": [2],
            "final_gates_shape": [6],
            "class_logits": [float(value) for value in logits.tolist()],
            "raw_logit_margin": replay["raw_logit_margin"],
            "fake_probability": score,
            "probability": score,
            "ai_score": score,
            "score": score,
            "routing_mode": "Auto (Router)",
            "semantic_expert_names": [
                "Human",
                "Animal",
                "Object",
                "Scene",
                "Anime",
            ],
            "artifact_expert_name": "Artifact",
            "semantic_top_k_indices": [
                int(value) for value in arrays["semantic_top_k_indices"].tolist()
            ],
            "semantic_top_k_gates": [
                float(value) for value in arrays["semantic_top_k_gates"].tolist()
            ],
            "final_expert_gates": [
                float(value) for value in arrays["final_gates"].tolist()
            ],
            "semantic_gate_sum": float(
                arrays["final_gates"][
                    : analyzer.SEMANTIC_EXPERT_COUNT
                ].sum()
            ),
            "final_gate_sum": float(arrays["final_gates"].sum()),
            "classification_decision": score > 0.5,
            "classification": {
                "decision": score > 0.5,
                "threshold": 0.5,
                "operator": ">",
            },
            "t1": {
                "valid": True,
                "score": score,
                "decision": score > 0.5,
            },
            "manual_replay": {
                "head_logits_exact": True,
                "softmax_dtype": "float32",
                "fake_class_index": 1,
                "router_scatter_exact": True,
            },
            "latency_ms": 1.0,
            "peak_cuda_memory_bytes": None,
        }
        rows.append(row)

    expected_path = run_dir / "expected_inputs.jsonl"
    results_path = run_dir / "results.jsonl"
    summary_path = run_dir / "summary.json"
    manifest_path = run_dir / "run_manifest.json"
    output_path = run_dir / "independent_audit.json"
    atomic_write_jsonl(expected_path, expected)
    atomic_write_jsonl(results_path, rows)

    summary = summarize_omniaid_results(
        rows,
        expected,
        threshold=0.5,
        bootstrap_samples=5,
        seed=20260724,
    )
    summary.update(
        {
            "run_id": run_dir.name,
            "model": "OmniAID",
            "model_slug": analyzer.FROZEN_MODEL_SLUG,
            "checkpoint_id": analyzer.FROZEN_CHECKPOINT_ID,
            "checkpoint_sha256": analyzer.FROZEN_CHECKPOINT_SHA256,
            "preprocess_profile": analyzer.FROZEN_PREPROCESS_PROFILE,
            "config_fingerprint": config_fingerprint,
            "runtime_golden_status": runtime_golden["status"],
            "runtime_golden_fingerprint": analyzer._fingerprint(runtime_golden),
            "generated_at": "2026-07-25T00:00:00Z",
        }
    )
    atomic_write_json(summary_path, summary)

    manifest = {
        "schema_version": "omniaid_detection_run_manifest_v1",
        "run_id": run_dir.name,
        "status": "complete",
        "config": config,
        "config_fingerprint": config_fingerprint,
        "source": source,
        "assets": assets,
        "model_audit": model_audit,
        "runtime": runtime,
        "runtime_golden": runtime_golden,
        "dataset": {
            "selected_images": 2,
            "selected_tasks": 1,
            "inputs_sha256": analyzer.CANONICAL_RELEASE["inputs_sha256"],
            "expected_inputs_path": str(expected_path.relative_to(repo_root)),
            "expected_inputs_sha256": sha256_file(expected_path),
        },
        "visibility_census": {"full": 1},
        "outputs": {
            "results_path": str(results_path.relative_to(repo_root)),
            "summary_path": str(summary_path.relative_to(repo_root)),
            "artifact_dir": str(artifact_dir.relative_to(repo_root)),
            "results_sha256": sha256_file(results_path),
            "summary_sha256": sha256_file(summary_path),
            "artifact_files": 2,
        },
        "task_scope": {
            "valid_for_t1": True,
            "valid_for_t2": False,
        },
    }
    atomic_write_json(manifest_path, manifest)
    return _TinyRun(
        repo_root=repo_root,
        run_dir=run_dir,
        output_path=output_path,
        expected_path=expected_path,
        results_path=results_path,
        summary_path=summary_path,
        manifest_path=manifest_path,
        expected=expected,
        rows=rows,
        model=model,
    )


def _install_independent_mocks(
    case: _TinyRun,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        analyzer,
        "_verify_source",
        lambda _source, _space: {
            "github": {
                "commit": analyzer.FROZEN_SOURCE_COMMIT,
                "tracked_dirty": False,
                "tracked_license_files": [],
                "files": {},
            },
            "space": {
                "commit": analyzer.FROZEN_SPACE_COMMIT,
                "tracked_dirty": False,
                "tracked_license_files": [],
                "files": {},
            },
            "inference_source": "synthetic",
        },
    )
    monkeypatch.setattr(
        analyzer,
        "_verify_adapter_contract",
        lambda _config, _root: {"synthetic": True},
    )
    monkeypatch.setattr(
        analyzer,
        "_load_assets",
        lambda _checkpoint, _config: (
            {},
            {},
            {
                "checkpoint_sha256": analyzer.FROZEN_CHECKPOINT_SHA256,
                "checkpoint_tensor_count": analyzer.FROZEN_CHECKPOINT_TENSORS,
                "weights_only": True,
                "unsafe_globals": ["argparse.Namespace"],
            },
        ),
    )
    monkeypatch.setattr(
        analyzer,
        "_configure_runtime",
        lambda device_text: (
            torch.device(device_text),
            {"device": str(torch.device(device_text))},
        ),
    )
    monkeypatch.setattr(
        analyzer,
        "_build_model",
        lambda _state, _config, _device, _space: (
            case.model,
            {
                "strict_load": True,
                "missing_keys": [],
                "unexpected_keys": [],
                "svd_modules": analyzer.SVD_MODULES,
                "state_entries": analyzer.FROZEN_CHECKPOINT_TENSORS,
            },
        ),
    )


def _audit(
    case: _TinyRun,
    monkeypatch: pytest.MonkeyPatch,
) -> dict:
    _install_independent_mocks(case, monkeypatch)
    return analyzer._audit_run(
        repo_root=case.repo_root,
        run_dir=case.run_dir,
        source_root=case.repo_root / "source",
        space_root=case.repo_root / "space",
        checkpoint_path=case.repo_root / "checkpoint.pth",
        config_path=case.repo_root / "config.json",
        device_text="cpu",
        output_path=case.output_path,
    )


def _manifest(case: _TinyRun) -> dict:
    return json.loads(case.manifest_path.read_text(encoding="utf-8"))


def _summary(case: _TinyRun) -> dict:
    return json.loads(case.summary_path.read_text(encoding="utf-8"))


def _rewrite_results_and_sync_hash(
    case: _TinyRun,
    rows: list[dict],
) -> None:
    atomic_write_jsonl(case.results_path, rows)
    manifest = _manifest(case)
    manifest["outputs"]["results_sha256"] = sha256_file(case.results_path)
    atomic_write_json(case.manifest_path, manifest)


def _rewrite_summary_and_sync_hash(
    case: _TinyRun,
    summary: dict,
) -> None:
    atomic_write_json(case.summary_path, summary)
    manifest = _manifest(case)
    manifest["outputs"]["summary_sha256"] = sha256_file(case.summary_path)
    atomic_write_json(case.manifest_path, manifest)


def test_dino_v2_contract_is_frozen_without_runner_import() -> None:
    source = Path(analyzer.__file__).read_text(encoding="utf-8")
    imported_modules = {
        alias.name
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported_modules.update(
        node.module
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.ImportFrom) and node.module is not None
    )
    assert "eval.opensource.run_omniaid" not in imported_modules
    assert analyzer.FROZEN_SOURCE_COMMIT == ("40749406fbcd8893c11a160edf4a72a2d4dc7056")
    assert analyzer.FROZEN_SPACE_COMMIT == ("cf99ed518af8b7256854d01994d6e41165553bb3")
    assert analyzer.FROZEN_CHECKPOINT_BYTES == 3_238_483_725
    assert analyzer.FROZEN_CHECKPOINT_SHA256 == (
        "8135cf83a7acbd3d88e457062f7ad693b1f2e27ffc8d5ae7ec73fcb5de806ea9"
    )
    assert analyzer.FROZEN_CHECKPOINT_TENSORS == 2_852
    assert analyzer.FROZEN_CHECKPOINT_ELEMENTS == 808_835_239
    assert analyzer.FROZEN_SCHEMA_SHA256 == (
        "1b5a03a08369fa7dc5034b1b9aa8a4757295386afd3c91f093ef41b6e2c9b67d"
    )
    assert analyzer.DEFAULT_RUN_ID == (
        "omniaid_dino_v2_mirage_auto_" "mouse_canonical_v1_full275_20260725"
    )
    assert list(analyzer.ARTIFACT_SCHEMA) == [
        "pooler_output",
        "class_logits",
        "routing_feature",
        "semantic_top_k_indices",
        "semantic_top_k_gates",
        "final_gates",
    ]


def test_preprocess_is_pil_rgb_direct_bilinear_imagenet(
    tmp_path: Path,
) -> None:
    rgb = np.arange(7 * 11 * 3, dtype=np.uint8).reshape(7, 11, 3)
    path = tmp_path / "input.png"
    Image.fromarray(rgb, mode="RGB").save(path)
    tensor, record = analyzer._preprocess(path)
    assert tensor.shape == (3, 448, 448)
    assert tensor.dtype == np.float32
    assert record["decode"] == "PIL.Image.open_then_convert_RGB"
    assert record["exif_transpose"] is False
    assert record["resize"] == {
        "output_wh": [448, 448],
        "implementation": "torchvision.transforms.Resize_list_hw",
        "interpolation": "PIL_BILINEAR_default",
        "antialias": True,
        "preserve_aspect_ratio": False,
        "crop": None,
        "face_alignment": False,
    }
    assert record["normalization_mean"] == list(analyzer.IMAGENET_MEAN)
    assert record["normalization_std"] == list(analyzer.IMAGENET_STD)
    assert record["tensor_sha256"] == analyzer._array_sha256(tensor)


def test_synthetic_happy_path_audits_every_layer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _build_tiny_run(tmp_path)
    audit = _audit(case, monkeypatch)

    assert audit["schema_version"] == "omniaid_independent_audit_v1"
    assert audit["status"] == "ok"
    assert audit["run_id"] == "tiny-omniaid"
    assert audit["coverage"] == {
        "expected_images": 2,
        "physical_result_rows": 2,
        "latest_images": 2,
        "fresh_full_model_forwards": 2,
        "artifact_replays": 2,
        "complete_pairs": 1,
    }
    assert audit["replay"]["max_abs_array_diff"] == {
        key: 0.0 for key in analyzer.ARTIFACT_SCHEMA
    }
    assert audit["replay"]["max_abs_head_replay_diff"] == 0.0
    assert audit["replay"]["max_abs_router_index_replay_diff"] == 0.0
    assert audit["replay"]["max_abs_router_gate_replay_diff"] == 0.0
    assert audit["replay"]["max_abs_probability_diff"] == 0.0
    assert audit["replay"]["max_abs_margin_diff"] == 0.0
    assert audit["replay"]["all_six_artifact_arrays_exact"] is True
    assert audit["replay"]["all_router_gates_exact"] is True
    assert audit["replay"]["fresh_forward_summary_exact_recompute"] is True
    assert audit["hashes"]["manifest_sha256"] == sha256_file(case.manifest_path)
    assert audit["hashes"]["results_sha256"] == sha256_file(case.results_path)
    assert audit["hashes"]["summary_sha256"] == sha256_file(case.summary_path)
    assert audit["hashes"]["expected_inputs_sha256"] == sha256_file(case.expected_path)
    assert set(audit["hashes"]["result_fingerprints"]) == {
        "tiny-real",
        "tiny-forged",
    }
    assert json.loads(case.output_path.read_text(encoding="utf-8")) == audit
    analyzer._reject_t2(audit, path="audit")


@pytest.mark.parametrize(
    ("target", "message"),
    [
        ("results_sha256", "OmniAID results SHA-256 mismatch"),
        ("summary_sha256", "OmniAID summary SHA-256 mismatch"),
    ],
)
def test_manifest_output_hash_tamper_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target: str,
    message: str,
) -> None:
    case = _build_tiny_run(tmp_path)
    manifest = _manifest(case)
    manifest["outputs"][target] = "0" * 64
    atomic_write_json(case.manifest_path, manifest)

    with pytest.raises(ValueError, match=message):
        _audit(case, monkeypatch)


def test_expected_inputs_hash_and_count_tamper_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _build_tiny_run(tmp_path)
    manifest = _manifest(case)
    manifest["dataset"]["expected_inputs_sha256"] = "0" * 64
    atomic_write_json(case.manifest_path, manifest)
    with pytest.raises(ValueError, match="expected inputs SHA-256 mismatch"):
        _audit(case, monkeypatch)

    case = _build_tiny_run(tmp_path / "count")
    manifest = _manifest(case)
    manifest["dataset"]["selected_images"] = 3
    atomic_write_json(case.manifest_path, manifest)
    with pytest.raises(ValueError, match="expected image count mismatch"):
        _audit(case, monkeypatch)


def test_config_fingerprint_and_pinned_fields_are_independently_checked(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _build_tiny_run(tmp_path)
    manifest = _manifest(case)
    manifest["config"]["bootstrap_seed"] = 1
    atomic_write_json(case.manifest_path, manifest)
    with pytest.raises(ValueError, match="config fingerprint mismatch"):
        _audit(case, monkeypatch)

    case = _build_tiny_run(tmp_path / "pin")
    manifest = _manifest(case)
    manifest["config"]["source_commit"] = "0" * 40
    manifest["config_fingerprint"] = analyzer._fingerprint(manifest["config"])
    atomic_write_json(case.manifest_path, manifest)
    with pytest.raises(ValueError, match="source pin mismatch"):
        _audit(case, monkeypatch)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("schema_version", "wrong", "manifest schema mismatch"),
        ("status", "incomplete", "manifest is not complete"),
        ("run_id", "other", "manifest run_id mismatch"),
    ],
)
def test_manifest_identity_tamper_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: str,
    message: str,
) -> None:
    case = _build_tiny_run(tmp_path)
    manifest = _manifest(case)
    manifest[field] = value
    atomic_write_json(case.manifest_path, manifest)
    with pytest.raises(ValueError, match=message):
        _audit(case, monkeypatch)


def test_canonical_release_accepts_full_metadata_and_rejects_identity_drift():
    full_record = {
        **analyzer.CANONICAL_RELEASE,
        "created_at": "2026-07-23T14:03:24.374009+00:00",
        "domains": {"lodging": 147, "restaurant": 128},
        "jpeg": {"quality": 95},
    }
    assert analyzer._verify_canonical_release(full_record) == full_record

    changed = {
        **full_record,
        "inputs_sha256": "0" * 64,
    }
    with pytest.raises(
        ValueError,
        match="canonical release inputs_sha256 mismatch",
    ):
        analyzer._verify_canonical_release(changed)


def test_source_and_checkpoint_drift_propagate_as_hard_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _build_tiny_run(tmp_path)
    _install_independent_mocks(case, monkeypatch)

    def reject_source(_source: Path, _space: Path) -> dict:
        raise ValueError("independent OmniAID source commit mismatch")

    monkeypatch.setattr(
        analyzer,
        "_verify_source",
        reject_source,
    )
    with pytest.raises(ValueError, match="source commit mismatch"):
        analyzer._audit_run(
            repo_root=case.repo_root,
            run_dir=case.run_dir,
            source_root=case.repo_root / "source",
            space_root=case.repo_root / "space",
            checkpoint_path=case.repo_root / "checkpoint.pth",
            config_path=case.repo_root / "config.json",
            device_text="cpu",
            output_path=case.output_path,
        )

    case = _build_tiny_run(tmp_path / "checkpoint")
    _install_independent_mocks(case, monkeypatch)

    def reject_checkpoint(
        _checkpoint: Path,
        _config: Path,
    ) -> tuple[dict, dict, dict]:
        raise ValueError("independent OmniAID checkpoint byte size mismatch")

    monkeypatch.setattr(
        analyzer,
        "_load_assets",
        reject_checkpoint,
    )
    with pytest.raises(ValueError, match="checkpoint byte size mismatch"):
        analyzer._audit_run(
            repo_root=case.repo_root,
            run_dir=case.run_dir,
            source_root=case.repo_root / "source",
            space_root=case.repo_root / "space",
            checkpoint_path=case.repo_root / "checkpoint.pth",
            config_path=case.repo_root / "config.json",
            device_text="cpu",
            output_path=case.output_path,
        )


def test_private_source_and_asset_guards_reject_drift_before_loading(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(analyzer, "_git_value", lambda *_args: "0" * 40)
    with pytest.raises(ValueError, match="source commit mismatch"):
        analyzer._verify_source(tmp_path, tmp_path)

    checkpoint = tmp_path / "tiny.pth"
    config = tmp_path / "config.json"
    checkpoint.write_bytes(b"not-the-pinned-checkpoint")
    config.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="checkpoint byte size mismatch"):
        analyzer._load_assets(checkpoint, config)


def test_checkpoint_loader_allows_only_argparse_namespace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "checkpoint.pth"
    checkpoint.write_bytes(b"synthetic-omniaid-checkpoint")
    config_path = tmp_path / "config.json"
    config_payload = {
        "DINOV3_path": "facebook/dinov3-vitl16-pretrain-lvd1689m",
        "num_experts": 6,
        "rank_per_expert": 1,
        "moe_router_hidden_dim": 256,
        "moe_top_k": 2,
        "gradient_checkpointing_enable": False,
    }
    config_path.write_text(json.dumps(config_payload), encoding="utf-8")
    state = {
        "head.weight": torch.zeros((2, 1024), dtype=torch.float32),
        "head.bias": torch.zeros(2, dtype=torch.float32),
        "gating_network.network.0.weight": torch.zeros(
            (256, 1024),
            dtype=torch.float32,
        ),
        "gating_network.network.2.weight": torch.zeros(
            (5, 256),
            dtype=torch.float32,
        ),
    }
    payload = {
        "model": state,
        "optimizer": {},
        "epoch": 0,
        "scaler": {},
        "args": argparse.Namespace(model="omniaid-dino-v2"),
    }
    monkeypatch.setattr(
        analyzer,
        "FROZEN_CHECKPOINT_BYTES",
        checkpoint.stat().st_size,
    )
    monkeypatch.setattr(
        analyzer,
        "FROZEN_CHECKPOINT_SHA256",
        sha256_file(checkpoint),
    )
    monkeypatch.setattr(analyzer, "FROZEN_CHECKPOINT_TENSORS", len(state))
    monkeypatch.setattr(
        analyzer,
        "FROZEN_CHECKPOINT_ELEMENTS",
        sum(value.numel() for value in state.values()),
    )
    monkeypatch.setattr(
        analyzer,
        "FROZEN_ORDERED_KEY_SHA256",
        hashlib.sha256("\n".join(state).encode("utf-8")).hexdigest(),
    )
    monkeypatch.setattr(
        analyzer,
        "FROZEN_SCHEMA_SHA256",
        analyzer._schema_sha256(state),
    )
    monkeypatch.setattr(
        analyzer,
        "FROZEN_CONFIG_BYTES",
        config_path.stat().st_size,
    )
    monkeypatch.setattr(
        analyzer,
        "FROZEN_CONFIG_SHA256",
        sha256_file(config_path),
    )
    monkeypatch.setattr(
        torch.serialization,
        "get_unsafe_globals_in_checkpoint",
        lambda _path: ["argparse.Namespace"],
    )
    observed: dict[str, object] = {}

    def fake_load(
        path: Path,
        *,
        map_location: str,
        weights_only: bool,
        mmap: bool,
    ) -> dict:
        observed.update(
            {
                "path": path,
                "map_location": map_location,
                "weights_only": weights_only,
                "mmap": mmap,
            }
        )
        return payload

    monkeypatch.setattr(torch, "load", fake_load)
    loaded, loaded_config, evidence = analyzer._load_assets(
        checkpoint,
        config_path,
    )
    assert loaded is state
    assert loaded_config == config_payload
    assert evidence["unsafe_globals"] == ["argparse.Namespace"]
    assert evidence["safe_globals_allowlist"] == ["argparse.Namespace"]
    assert evidence["arbitrary_code_execution_enabled"] is False
    assert observed == {
        "path": checkpoint,
        "map_location": "cpu",
        "weights_only": True,
        "mmap": True,
    }

    monkeypatch.setattr(
        torch.serialization,
        "get_unsafe_globals_in_checkpoint",
        lambda _path: ["malicious.CustomType"],
    )
    with pytest.raises(ValueError, match="unsafe-global census mismatch"):
        analyzer._load_assets(checkpoint, config_path)


def test_adapter_contract_binds_runner_metrics_and_shared_helpers(
    tmp_path: Path,
) -> None:
    relatives = [
        "eval/opensource/run_omniaid.py",
        "eval/opensource/omniaid_metrics.py",
        "eval/opensource/ufd_metrics.py",
        "eval/opensource/common.py",
    ]
    adapter: dict[str, dict[str, object]] = {}
    for index, relative in enumerate(relatives):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"fixture-{index}\n", encoding="utf-8")
        adapter[relative] = {
            "path": str(path.resolve()),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    evidence = analyzer._verify_adapter_contract(
        {"adapter": adapter},
        tmp_path,
    )
    assert set(evidence) == set(relatives)

    changed = tmp_path / "eval/opensource/omniaid_metrics.py"
    changed.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="adapter .* SHA-256 mismatch"):
        analyzer._verify_adapter_contract({"adapter": adapter}, tmp_path)


def test_official_space_model_is_built_on_meta_with_strict_assign_and_two_ropes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeDinoConfig:
        def __init__(self, **values: object) -> None:
            self.values = values

    class FakeDinoModel(nn.Module):
        def __init__(self, _config: object) -> None:
            super().__init__()
            self.rope_embeddings = nn.Module()

        @classmethod
        def from_pretrained(cls, *_args: object, **_kwargs: object) -> object:
            raise AssertionError("from_pretrained must be shape-only patched")

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.DINOv3ViTConfig = FakeDinoConfig
    fake_transformers.DINOv3ViTModel = FakeDinoModel
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    source_dir = tmp_path / "space" / "model"
    source_dir.mkdir(parents=True)
    source_path = source_dir / "omniaid-dino.py"
    source_path.write_text(
        """
from collections import OrderedDict
import types
import torch
from torch import nn
from transformers import DINOv3ViTModel

LAST_LOAD = None

class SVDMoeLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight_main = torch.empty((1024, 1024))
        self.U_experts = [torch.empty((1024, 1)) for _ in range(6)]
        self.S_experts = [torch.empty((1,)) for _ in range(6)]
        self.V_experts = [torch.empty((1, 1024)) for _ in range(6)]

class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = SVDMoeLinear()
        self.k_proj = SVDMoeLinear()
        self.v_proj = SVDMoeLinear()
        self.o_proj = SVDMoeLinear()

class Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = Attention()

class OmniAID_DINO(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.feature_extractor = DINOv3ViTModel.from_pretrained(
            config.DINOV3_path
        )
        self.rope_embeddings = nn.Module()
        self.layer = nn.ModuleList([Layer() for _ in range(24)])
        self.head = nn.Linear(1024, 2)

    def state_dict(self, *args, **kwargs):
        return OrderedDict(
            [
                ("head.weight", self.head.weight),
                ("head.bias", self.head.bias),
            ]
        )

    def load_state_dict(self, state, strict=True, assign=False):
        global LAST_LOAD
        LAST_LOAD = {"strict": strict, "assign": assign}
        self.head = nn.Linear(1024, 2)
        self.head.weight = nn.Parameter(state["head.weight"])
        self.head.bias = nn.Parameter(state["head.bias"])
        return types.SimpleNamespace(missing_keys=[], unexpected_keys=[])
""".lstrip(),
        encoding="utf-8",
    )
    state = {
        "head.weight": torch.zeros((2, 1024), dtype=torch.float32),
        "head.bias": torch.zeros(2, dtype=torch.float32),
    }
    config = {
        "DINOV3_path": "facebook/dinov3-vitl16-pretrain-lvd1689m",
    }
    model, evidence = analyzer._build_model(
        state,
        config,
        torch.device("cpu"),
        tmp_path / "space",
    )
    official = __import__(
        "claimforge_independent_omniaid_dino",
        fromlist=["LAST_LOAD"],
    )
    assert official.LAST_LOAD == {"strict": True, "assign": True}
    assert evidence["constructor"] == (
        "official_space_inference_class_shape_only_dinov3_base_on_meta"
    )
    assert evidence["svd_modules"] == 96
    assert evidence["nonpersistent_rope_buffers_materialized"] == [
        "feature_extractor.rope_embeddings.inv_freq",
        "rope_embeddings.inv_freq",
    ]
    assert evidence["rope_inv_freq_shape"] == [16]
    assert evidence["base_weights_downloaded"] is False
    assert not any(parameter.is_meta for parameter in model.parameters())
    torch.testing.assert_close(
        model.feature_extractor.rope_embeddings.inv_freq,
        model.rope_embeddings.inv_freq,
        rtol=0.0,
        atol=0.0,
    )
    expected_inv_freq = 1.0 / 100.0 ** torch.arange(
        0,
        1,
        4 / 64,
        dtype=torch.float32,
    )
    torch.testing.assert_close(
        model.rope_embeddings.inv_freq,
        expected_inv_freq,
        rtol=0.0,
        atol=0.0,
    )


def test_safe_npz_loader_accepts_only_exact_finite_float32_contract(
    tmp_path: Path,
) -> None:
    path = tmp_path / "valid.npz"
    feature = np.arange(1024, dtype=np.float32)
    logits = np.asarray([-1.0, 1.0], dtype=np.float32)
    _write_artifact(path, feature, logits)
    loaded = analyzer._load_artifact(path)
    assert set(loaded) == set(analyzer.ARTIFACT_SCHEMA)
    np.testing.assert_array_equal(loaded["pooler_output"], feature)
    np.testing.assert_array_equal(loaded["class_logits"], logits)


@pytest.mark.parametrize(
    "variant",
    ["extra", "shape", "dtype", "nonfinite", "object"],
)
def test_safe_npz_loader_rejects_malformed_artifacts(
    tmp_path: Path,
    variant: str,
) -> None:
    path = tmp_path / f"{variant}.npz"
    feature: np.ndarray = np.zeros(1024, dtype=np.float32)
    logits: np.ndarray = np.zeros(2, dtype=np.float32)
    extras: dict[str, np.ndarray] = {}
    if variant == "extra":
        extras["unexpected"] = np.zeros(1, dtype=np.float32)
    elif variant == "shape":
        feature = np.zeros(1023, dtype=np.float32)
    elif variant == "dtype":
        logits = np.zeros(2, dtype=np.float64)
    elif variant == "nonfinite":
        logits[0] = np.inf
    elif variant == "object":
        feature = np.asarray([object()] * 1024, dtype=object)
    _write_artifact(path, feature, logits, **extras)
    with pytest.raises(ValueError):
        analyzer._load_artifact(path)


@pytest.mark.parametrize(
    ("indices", "gates", "final_gates"),
    [
        (
            np.asarray([0, 0], dtype=np.int64),
            np.asarray([0.25, 0.75], dtype=np.float32),
            np.asarray([0.25, 0, 0.75, 0, 0, 1], dtype=np.float32),
        ),
        (
            np.asarray([0, 5], dtype=np.int64),
            np.asarray([0.25, 0.75], dtype=np.float32),
            np.asarray([0.25, 0, 0.75, 0, 0, 1], dtype=np.float32),
        ),
        (
            np.asarray([0, 2], dtype=np.int64),
            np.asarray([0.25, 0.5], dtype=np.float32),
            np.asarray([0.25, 0, 0.75, 0, 0, 1], dtype=np.float32),
        ),
        (
            np.asarray([0, 2], dtype=np.int64),
            np.asarray([0.25, 0.75], dtype=np.float32),
            np.asarray([0.25, 0, 0.75, 0, 0, 0.5], dtype=np.float32),
        ),
        (
            np.asarray([0, 2], dtype=np.int64),
            np.asarray([0.25, 0.75], dtype=np.float32),
            np.asarray([0.75, 0, 0.25, 0, 0, 1], dtype=np.float32),
        ),
    ],
)
def test_safe_npz_loader_rejects_invalid_router_gates(
    tmp_path: Path,
    indices: np.ndarray,
    gates: np.ndarray,
    final_gates: np.ndarray,
) -> None:
    path = tmp_path / "invalid-gates.npz"
    _write_artifact(
        path,
        np.zeros(1024, dtype=np.float32),
        np.zeros(2, dtype=np.float32),
        semantic_top_k_indices=indices,
        semantic_top_k_gates=gates,
        final_gates=final_gates,
    )
    with pytest.raises(ValueError, match="gate"):
        analyzer._load_artifact(path)


def test_artifact_path_traversal_and_duplicate_reference_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _build_tiny_run(tmp_path)
    rows = copy.deepcopy(case.rows)
    outside = case.repo_root / "outside.npz"
    original = case.run_dir / "artifacts" / "tiny-real.npz"
    outside.write_bytes(original.read_bytes())
    rows[0]["artifact_path"] = str(outside.relative_to(case.repo_root))
    rows[0]["artifact_sha256"] = sha256_file(outside)
    _rewrite_results_and_sync_hash(case, rows)
    with pytest.raises(ValueError, match="artifact escapes run"):
        _audit(case, monkeypatch)

    case = _build_tiny_run(tmp_path / "duplicate")
    rows = copy.deepcopy(case.rows)
    for key in (
        "artifact_path",
        "artifact_sha256",
        "feature_array_sha256",
        "class_logits_array_sha256",
        "class_logits",
    ):
        rows[1][key] = copy.deepcopy(rows[0][key])
    _rewrite_results_and_sync_hash(case, rows)
    with pytest.raises(ValueError, match="artifact name mismatch"):
        _audit(case, monkeypatch)


def test_extra_artifact_file_and_manifest_count_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _build_tiny_run(tmp_path)
    _write_artifact(
        case.run_dir / "artifacts" / "extra.npz",
        np.zeros(1024, dtype=np.float32),
        np.zeros(2, dtype=np.float32),
    )
    with pytest.raises(ValueError, match="missing/extra files"):
        _audit(case, monkeypatch)

    case = _build_tiny_run(tmp_path / "count")
    manifest = _manifest(case)
    manifest["outputs"]["artifact_files"] = 3
    atomic_write_json(case.manifest_path, manifest)
    with pytest.raises(ValueError, match="artifact count mismatch"):
        _audit(case, monkeypatch)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("task_id", "other-task"),
        ("model", "Other"),
        ("checkpoint_id", "wrong"),
        ("preprocess_profile", "cubic"),
        ("valid_for_t2", True),
    ],
)
def test_result_identity_tamper_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
) -> None:
    case = _build_tiny_run(tmp_path)
    rows = copy.deepcopy(case.rows)
    rows[0][field] = value
    _rewrite_results_and_sync_hash(case, rows)
    with pytest.raises(ValueError, match=f"identity {field} mismatch"):
        _audit(case, monkeypatch)


def test_preprocess_tamper_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _build_tiny_run(tmp_path)
    rows = copy.deepcopy(case.rows)
    rows[0]["preprocess"]["tensor_sha256"] = "0" * 64
    _rewrite_results_and_sync_hash(case, rows)
    with pytest.raises(ValueError, match="preprocess mismatch"):
        _audit(case, monkeypatch)


def test_feature_and_logit_artifact_tamper_fail_fresh_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _build_tiny_run(tmp_path)
    rows = copy.deepcopy(case.rows)
    artifact = case.run_dir / "artifacts" / "tiny-real.npz"
    arrays = analyzer._load_artifact(artifact)
    feature = arrays["pooler_output"]
    logits = arrays["class_logits"]
    feature[1] = 1.0
    _write_artifact(
        artifact,
        feature,
        logits,
        **{
            key: value
            for key, value in arrays.items()
            if key not in {"pooler_output", "class_logits"}
        },
    )
    changed = analyzer._load_artifact(artifact)
    rows[0]["artifact_sha256"] = sha256_file(artifact)
    rows[0]["feature_array_sha256"] = analyzer._array_sha256(feature)
    rows[0]["artifact_array_sha256"] = {
        key: analyzer._array_sha256(value) for key, value in changed.items()
    }
    _rewrite_results_and_sync_hash(case, rows)
    with pytest.raises(ValueError, match="fresh pooler_output replay differs"):
        _audit(case, monkeypatch)

    case = _build_tiny_run(tmp_path / "logit")
    rows = copy.deepcopy(case.rows)
    artifact = case.run_dir / "artifacts" / "tiny-real.npz"
    arrays = analyzer._load_artifact(artifact)
    feature = arrays["pooler_output"]
    logits = arrays["class_logits"]
    logits[0] += np.float32(0.25)
    _write_artifact(
        artifact,
        feature,
        logits,
        **{
            key: value
            for key, value in arrays.items()
            if key not in {"pooler_output", "class_logits"}
        },
    )
    changed = analyzer._load_artifact(artifact)
    rows[0]["artifact_sha256"] = sha256_file(artifact)
    rows[0]["class_logits_array_sha256"] = analyzer._array_sha256(logits)
    rows[0]["artifact_array_sha256"] = {
        key: analyzer._array_sha256(value) for key, value in changed.items()
    }
    rows[0]["class_logits"] = [float(value) for value in logits.tolist()]
    _rewrite_results_and_sync_hash(case, rows)
    with pytest.raises(ValueError, match="fresh class_logits replay differs"):
        _audit(case, monkeypatch)


def test_routing_feature_and_valid_gate_tamper_fail_fresh_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _build_tiny_run(tmp_path)
    rows = copy.deepcopy(case.rows)
    artifact = case.run_dir / "artifacts" / "tiny-real.npz"
    arrays = analyzer._load_artifact(artifact)
    arrays["routing_feature"][1] = np.float32(1.0)
    _write_artifact(
        artifact,
        arrays["pooler_output"],
        arrays["class_logits"],
        **{
            key: value
            for key, value in arrays.items()
            if key not in {"pooler_output", "class_logits"}
        },
    )
    changed = analyzer._load_artifact(artifact)
    rows[0]["artifact_sha256"] = sha256_file(artifact)
    rows[0]["artifact_array_sha256"] = {
        key: analyzer._array_sha256(value) for key, value in changed.items()
    }
    _rewrite_results_and_sync_hash(case, rows)
    with pytest.raises(ValueError, match="fresh routing_feature replay differs"):
        _audit(case, monkeypatch)

    case = _build_tiny_run(tmp_path / "gates")
    rows = copy.deepcopy(case.rows)
    artifact = case.run_dir / "artifacts" / "tiny-real.npz"
    arrays = analyzer._load_artifact(artifact)
    arrays["semantic_top_k_indices"] = np.asarray(
        [2, 0],
        dtype=np.int64,
    )
    arrays["semantic_top_k_gates"] = np.asarray(
        [0.3002614378929138, 0.6997385025024414],
        dtype=np.float32,
    )
    _write_artifact(
        artifact,
        arrays["pooler_output"],
        arrays["class_logits"],
        **{
            key: value
            for key, value in arrays.items()
            if key not in {"pooler_output", "class_logits"}
        },
    )
    changed = analyzer._load_artifact(artifact)
    rows[0]["artifact_sha256"] = sha256_file(artifact)
    rows[0]["artifact_array_sha256"] = {
        key: analyzer._array_sha256(value) for key, value in changed.items()
    }
    rows[0]["semantic_top_k_indices"] = [2, 0]
    rows[0]["semantic_top_k_gates"] = [0.75, 0.25]
    _rewrite_results_and_sync_hash(case, rows)
    with pytest.raises(
        ValueError,
        match="fresh semantic_top_k_indices replay differs",
    ):
        _audit(case, monkeypatch)


def test_feature_hash_and_embedded_logits_tamper_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _build_tiny_run(tmp_path)
    rows = copy.deepcopy(case.rows)
    rows[0]["feature_array_sha256"] = "0" * 64
    _rewrite_results_and_sync_hash(case, rows)
    with pytest.raises(ValueError, match="feature hash mismatch"):
        _audit(case, monkeypatch)

    case = _build_tiny_run(tmp_path / "embedded")
    rows = copy.deepcopy(case.rows)
    rows[0]["class_logits"][0] += 0.5
    _rewrite_results_and_sync_hash(case, rows)
    with pytest.raises(ValueError, match="embedded logits mismatch"):
        _audit(case, monkeypatch)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("ai_score", 0.25, "probability replay differs"),
        ("fake_probability", 0.25, "score alias fake_probability drifted"),
        ("raw_logit_margin", 123.0, "margin replay differs"),
        ("classification_decision", None, "decision mismatch"),
    ],
)
def test_probability_margin_alias_and_decision_tamper_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
    message: str,
) -> None:
    case = _build_tiny_run(tmp_path)
    rows = copy.deepcopy(case.rows)
    rows[0][field] = value
    _rewrite_results_and_sync_hash(case, rows)
    with pytest.raises(ValueError, match=message):
        _audit(case, monkeypatch)


def test_embedded_router_gate_and_nested_score_tamper_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _build_tiny_run(tmp_path)
    rows = copy.deepcopy(case.rows)
    rows[0]["final_expert_gates"][0] = 0.5
    _rewrite_results_and_sync_hash(case, rows)
    with pytest.raises(ValueError, match="embedded final_expert_gates mismatch"):
        _audit(case, monkeypatch)

    case = _build_tiny_run(tmp_path / "nested")
    rows = copy.deepcopy(case.rows)
    rows[0]["classification"]["decision"] = not rows[0]["classification_decision"]
    _rewrite_results_and_sync_hash(case, rows)
    with pytest.raises(ValueError, match="scoring field classification mismatch"):
        _audit(case, monkeypatch)


def test_summary_tamper_with_updated_hash_still_fails_recompute(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _build_tiny_run(tmp_path)
    summary = _summary(case)
    summary["detection"]["accuracy_at_0_5"] = 0.123
    _rewrite_summary_and_sync_hash(case, summary)
    with pytest.raises(ValueError, match="summary does not exactly recompute"):
        _audit(case, monkeypatch)


@pytest.mark.parametrize("container", ["manifest", "summary", "result"])
def test_nested_t2_tamper_is_rejected_everywhere(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    container: str,
) -> None:
    case = _build_tiny_run(tmp_path)
    invented = {"outer": [{"inner": {"pixel_ap": 0.9}}]}
    if container == "manifest":
        manifest = _manifest(case)
        manifest["invented"] = invented
        atomic_write_json(case.manifest_path, manifest)
    elif container == "summary":
        summary = _summary(case)
        summary["invented"] = invented
        _rewrite_summary_and_sync_hash(case, summary)
    else:
        rows = copy.deepcopy(case.rows)
        rows[0]["invented"] = invented
        _rewrite_results_and_sync_hash(case, rows)

    with pytest.raises(ValueError, match="invents OmniAID T2 fields"):
        _audit(case, monkeypatch)


def test_physical_duplicate_expected_id_and_extra_result_id_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _build_tiny_run(tmp_path)
    expected = copy.deepcopy(case.expected)
    expected[1]["sample_id"] = expected[0]["sample_id"]
    atomic_write_jsonl(case.expected_path, expected)
    manifest = _manifest(case)
    manifest["dataset"]["expected_inputs_sha256"] = sha256_file(case.expected_path)
    atomic_write_json(case.manifest_path, manifest)
    with pytest.raises(ValueError, match="expected IDs are duplicated"):
        _audit(case, monkeypatch)

    case = _build_tiny_run(tmp_path / "extra")
    rows = copy.deepcopy(case.rows)
    extra = copy.deepcopy(rows[0])
    extra["id"] = "unexpected"
    rows.append(extra)
    _rewrite_results_and_sync_hash(case, rows)
    with pytest.raises(ValueError, match="latest result coverage mismatch"):
        _audit(case, monkeypatch)


def test_main_rejects_run_and_output_path_traversal(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="run-id escapes"):
        analyzer.main(
            [
                "--results-dir",
                str(tmp_path),
                "--run-id",
                "../escape",
                "--device",
                "cpu",
            ]
        )


def test_main_allows_external_json_audit_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    results_dir = tmp_path / "results"
    run_dir = results_dir / "safe-run"
    run_dir.mkdir(parents=True)
    output = tmp_path / "external-audit.json"

    def fake_audit(**kwargs: object) -> dict:
        assert kwargs["run_dir"] == run_dir.resolve()
        assert kwargs["output_path"] == output.resolve()
        atomic_write_json(output, {"status": "ok"})
        return {
            "status": "ok",
            "run_id": "safe-run",
            "coverage": {},
            "replay": {},
        }

    monkeypatch.setattr(analyzer, "_audit_run", fake_audit)
    assert (
        analyzer.main(
            [
                "--results-dir",
                str(results_dir),
                "--run-id",
                "safe-run",
                "--device",
                "cpu",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    assert json.loads(output.read_text(encoding="utf-8")) == {"status": "ok"}

    run_dir = tmp_path / "safe-run"
    run_dir.mkdir()
    with pytest.raises(ValueError, match="cannot overwrite run evidence"):
        analyzer.main(
            [
                "--results-dir",
                str(tmp_path),
                "--run-id",
                "safe-run",
                "--device",
                "cpu",
                "--output",
                str(run_dir / "summary.json"),
            ]
        )
