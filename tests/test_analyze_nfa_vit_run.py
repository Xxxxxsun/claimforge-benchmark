from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from PIL import Image

if int(np.__version__.split(".", maxsplit=1)[0]) < 2:
    import cv2
else:
    cv2 = None

from eval.opensource.analyze_nfa_vit_run import (
    BR_GEN_SOURCE_COMMIT,
    IMDLBENCO_SOURCE_COMMIT,
    _array_sha256,
    _bilinear_align_corners_false,
    _manifest_fingerprint,
    _official_preprocess,
    _sigmoid_float32,
    analyze,
    audit_artifacts,
    audit_prefix_reproducibility,
    summarize_result_history,
)
from eval.opensource.common import (
    atomic_write_json,
    atomic_write_jsonl,
    sha256_file,
)
from eval.opensource.nfa_vit_metrics import (
    binary_pixel_metrics_strict,
    summarize_nfa_vit_results,
)


def _logit(probability: float) -> float:
    return math.log(probability / (1.0 - probability))


class NFAViTAnalysisFixture:
    def __init__(self, root: Path, *, pairs: int = 3) -> None:
        self.root = root
        self.repo = root / "repo"
        self.results_dir = self.repo / "results" / "opensource" / "nfa_vit"
        self.inputs_path = self.repo / "outputs" / "inputs.jsonl"
        self.nfa_root = root / "BR-Gen"
        self.imdl_root = root / "IMDLBenCo"
        self.run_id = "nfa_fixture_full"
        for path in (
            self.repo,
            self.results_dir,
            self.inputs_path.parent,
            self.nfa_root,
            self.imdl_root,
        ):
            path.mkdir(parents=True, exist_ok=True)

        self.checkpoint = self.repo / "weights" / "checkpoint-9999.pth"
        self.checkpoint.parent.mkdir(parents=True)
        self.checkpoint.write_bytes(b"official nfa fixture checkpoint")
        self.adapter_paths = [
            self.repo / "adapter" / "nfa_vit_metrics.py",
            self.repo / "adapter" / "analyze_nfa_vit_run.py",
        ]
        for index, path in enumerate(self.adapter_paths):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"adapter fixture {index}\n", encoding="utf-8")

        (self.nfa_root / "README.md").write_text(
            "BR-Gen fixture\n",
            encoding="utf-8",
        )
        (self.nfa_root / "models.py").write_text(
            "NFA fixture\n",
            encoding="utf-8",
        )
        (self.imdl_root / "README.md").write_text(
            "IMDLBenCo fixture\n",
            encoding="utf-8",
        )
        self.input_rows: list[dict[str, object]] = []
        self.result_rows: list[dict[str, object]] = []
        for pair_rank in range(pairs):
            task_id = f"task-{pair_rank:02d}"
            for offset, kind in enumerate(("real", "forged")):
                self._add_image(
                    rank=pair_rank * 2 + offset,
                    pair_rank=pair_rank,
                    task_id=task_id,
                    kind=kind,
                )
        atomic_write_jsonl(self.inputs_path, self.input_rows)
        self.manifest = self._manifest()
        fingerprint = self.manifest["fingerprint"]
        for row in self.result_rows:
            row["run_manifest_fingerprint"] = fingerprint
        atomic_write_jsonl(self.results_path, self.result_rows)
        self.summary = summarize_nfa_vit_results(
            self.result_rows,
            self.input_rows,
            bootstrap_samples=12,
            seed=31,
        )
        self.summary.update(
            {
                "run_id": self.run_id,
                "run_manifest_fingerprint": fingerprint,
                "checkpoint_sha256": sha256_file(self.checkpoint),
                "model_source_commit": BR_GEN_SOURCE_COMMIT,
                "imdlbenco_source_commit": IMDLBENCO_SOURCE_COMMIT,
            }
        )
        atomic_write_json(self.manifest_path, self.manifest)
        atomic_write_json(self.summary_path, self.summary)

    @property
    def results_path(self) -> Path:
        return self.results_dir / f"{self.run_id}.jsonl"

    @property
    def manifest_path(self) -> Path:
        return self.results_dir / f"{self.run_id}.run_manifest.json"

    @property
    def summary_path(self) -> Path:
        return self.results_dir / f"{self.run_id}.summary.json"

    def _relative(self, path: Path) -> str:
        return str(path.relative_to(self.repo))

    def _save_npy(
        self,
        path: Path,
        array: np.ndarray,
    ) -> dict[str, object]:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, np.asarray(array, dtype=np.float32), allow_pickle=False)
        return {
            "path": self._relative(path),
            "sha256": sha256_file(path),
            "shape": list(array.shape),
            "dtype": "float32",
        }

    def _add_image(
        self,
        *,
        rank: int,
        pair_rank: int,
        task_id: str,
        kind: str,
    ) -> None:
        sample_id = f"{task_id}-{kind}"
        width = 12
        height = 8
        image = np.zeros((height, width, 3), dtype=np.uint8)
        image[..., 0] = 40 + pair_rank * 10
        image[..., 1] = np.arange(width, dtype=np.uint8)[None, :] * 8
        image[..., 2] = np.arange(height, dtype=np.uint8)[:, None] * 12
        if kind == "forged":
            image[2:5, 4:8, :] += 35
        image_path = self.repo / "images" / f"{sample_id}.jpg"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(image, mode="RGB").save(
            image_path,
            format="JPEG",
            quality=95,
        )

        gt_path: Path | None = None
        gt_sha: str | None = None
        target = np.zeros((height, width), dtype=np.uint8)
        if kind == "forged":
            target[2:5, 4:8] = 255
            gt_path = self.repo / "masks" / f"{sample_id}.png"
            gt_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(target, mode="L").save(gt_path)
            gt_sha = sha256_file(gt_path)

        self.input_rows.append(
            {
                "rank": rank,
                "pair_rank": pair_rank,
                "sample_id": sample_id,
                "task_id": task_id,
                "kind": kind,
                "label": int(kind == "forged"),
                "domain": "lodging" if pair_rank % 2 else "restaurant",
                "canonical_path": self._relative(image_path),
                "canonical_sha256": sha256_file(image_path),
                "gt_mask_path": (
                    None if gt_path is None else self._relative(gt_path)
                ),
                "gt_mask_sha256": gt_sha,
                "edit_region_xyxy": [4, 2, 8, 5],
            }
        )

        raw = np.full((128, 128), -4.0, dtype=np.float32)
        raw += np.linspace(
            -0.1,
            0.1,
            128,
            dtype=np.float32,
        )[None, :]
        if kind == "forged":
            raw[32:80, 42:86] = np.float32(4.0)
        resized = _bilinear_align_corners_false(raw, width=512, height=512)
        probability = _sigmoid_float32(resized)
        native = _bilinear_align_corners_false(
            probability,
            width=width,
            height=height,
        )
        artifact_root = self.repo / "artifacts" / sample_id
        artifacts = {
            "decoder_logits_128_npy": self._save_npy(
                artifact_root / "raw.npy",
                raw,
            ),
            "resized_logits_512_npy": self._save_npy(
                artifact_root / "resized.npy",
                resized,
            ),
            "probability_512_npy": self._save_npy(
                artifact_root / "probability.npy",
                probability,
            ),
            "probability_native_npy": self._save_npy(
                artifact_root / "native.npy",
                native,
            ),
        }
        mask = np.asarray(native > 0.5, dtype=np.uint8) * 255
        mask_path = artifact_root / "mask.png"
        Image.fromarray(mask, mode="L").save(mask_path)
        artifacts["mask_native_png"] = {
            "path": self._relative(mask_path),
            "sha256": sha256_file(mask_path),
            "shape": [height, width],
            "dtype": "uint8",
        }

        target_bool = target > 0
        target_model = cv2.resize(
            target,
            (512, 512),
            interpolation=cv2.INTER_NEAREST,
        ) > 0
        model_metrics = binary_pixel_metrics_strict(
            probability,
            target_model,
            include_ap=kind == "forged",
        )
        native_metrics = binary_pixel_metrics_strict(
            native,
            target_bool,
            include_ap=kind == "forged",
        )
        score = 0.1 if kind == "real" else 0.9
        tensor = _official_preprocess(image_path)
        self.result_rows.append(
            {
                "schema_version": "opensource_result_v1",
                "run_id": self.run_id,
                "id": sample_id,
                "task_id": task_id,
                "rank": rank,
                "pair_rank": pair_rank,
                "kind": kind,
                "label": int(kind == "forged"),
                "domain": "lodging" if pair_rank % 2 else "restaurant",
                "status": "ok",
                "valid_for_t1": True,
                "valid_for_t2": True,
                "model": "NFA-ViT",
                "model_source_commit": BR_GEN_SOURCE_COMMIT,
                "imdlbenco_source_commit": IMDLBENCO_SOURCE_COMMIT,
                "checkpoint_sha256": sha256_file(self.checkpoint),
                "image_size": [width, height],
                "score": score,
                "classification_raw_logit": _logit(score),
                "classification_score": score,
                "classification_decision": score > 0.5,
                "classification_threshold": 0.5,
                "classification_threshold_operator": ">",
                "classification": {
                    "raw_logit": _logit(score),
                    "probability": score,
                    "score": score,
                    "decision": score > 0.5,
                    "threshold": 0.5,
                    "threshold_operator": ">",
                },
                "decision": "forged" if score > 0.5 else "authentic",
                "preprocess": {
                    "decoder": "Pillow.Image.open.convert_RGB",
                    "channel_order": "RGB",
                    "geometry": (
                        "direct_stretch_without_aspect_ratio_preservation"
                    ),
                    "resize_interpolation": "cv2.INTER_LINEAR",
                    "input_reencode": False,
                    "normalization": {
                        "mean": [0.485, 0.456, 0.406],
                        "std": [0.229, 0.224, 0.225],
                        "max_pixel_value": 255.0,
                    },
                    "native_size_wh": [width, height],
                    "model_size_wh": [512, 512],
                    "tensor_shape": [3, 512, 512],
                    "tensor_dtype": "float32",
                    "tensor_sha256": _array_sha256(tensor),
                },
                "artifact_paths": artifacts,
                "localization": {
                    "model_512": model_metrics,
                    "native": native_metrics,
                },
                "latency_ms": 12.0 + rank,
                "peak_cuda_memory_bytes": 4096,
            }
        )

    def _source_record(
        self,
        root: Path,
        commit: str,
        files: tuple[str, ...],
    ) -> dict[str, object]:
        return {
            "source_root": str(root.resolve()),
            "source_commit": commit,
            "source_files": [
                {
                    "path": path,
                    "sha256": sha256_file(root / path),
                }
                for path in files
            ],
        }

    def _manifest(self) -> dict[str, object]:
        immutable: dict[str, object] = {
            "schema_version": "opensource_run_manifest_v1",
            "run_id": self.run_id,
            "input": {
                "inputs_manifest": self._relative(self.inputs_path),
                "inputs_sha256": sha256_file(self.inputs_path),
            },
            "ordered_inputs": [
                {
                    "rank": int(row["rank"]),
                    "pair_rank": int(row["pair_rank"]),
                    "sample_id": row["sample_id"],
                    "task_id": row["task_id"],
                    "kind": row["kind"],
                    "label": int(row["label"]),
                    "canonical_path": row["canonical_path"],
                    "canonical_sha256": row["canonical_sha256"],
                    "gt_mask_sha256": row["gt_mask_sha256"],
                }
                for row in self.input_rows
            ],
            "model": {
                "name": "NFA-ViT",
                **self._source_record(
                    self.nfa_root,
                    BR_GEN_SOURCE_COMMIT,
                    ("README.md", "models.py"),
                ),
                "imdlbenco_source": self._source_record(
                    self.imdl_root,
                    IMDLBENCO_SOURCE_COMMIT,
                    ("README.md",),
                ),
                "checkpoint": {
                    "path": self._relative(self.checkpoint),
                    "original_filename": "checkpoint-9999.pth",
                    "sha256": sha256_file(self.checkpoint),
                    "bytes": self.checkpoint.stat().st_size,
                },
            },
            "runtime_contract": {
                "packages": {
                    "torch": str(torch.__version__),
                },
                "accelerator": {
                    "requested_device": "cpu",
                    "torch_cuda": torch.version.cuda,
                    "gpu_name": None,
                    "gpu_capability": None,
                },
            },
            "adapter_contract": [
                {
                    "path": self._relative(path),
                    "sha256": sha256_file(path),
                }
                for path in self.adapter_paths
            ],
        }
        return {
            **immutable,
            "fingerprint": _manifest_fingerprint(immutable),
        }

    def args(self) -> argparse.Namespace:
        return argparse.Namespace(
            repo_root=self.repo,
            run_id=self.run_id,
            results_dir=self.results_dir,
            inputs=self.inputs_path,
            nfa_vit_root=self.nfa_root,
            imdlbenco_root=self.imdl_root,
            bootstrap_iterations=12,
            bootstrap_seed=31,
            prefix_run_id=None,
            prefix_results_dir=self.results_dir,
            output=None,
            checkpoint_sha256_test_contract=sha256_file(self.checkpoint),
        )


@unittest.skipUnless(
    cv2 is not None,
    "NFA-ViT analyzer tests require a NumPy-compatible OpenCV runtime",
)
class AnalyzeNFAViTRunTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.fixture = NFAViTAnalysisFixture(Path(self.temporary.name))

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _replace_npy(
        self,
        row: dict[str, object],
        artifact_key: str,
        array: np.ndarray,
    ) -> None:
        record = row["artifact_paths"][artifact_key]
        path = self.fixture.repo / record["path"]
        np.save(path, np.asarray(array, dtype=np.float32), allow_pickle=False)
        record["sha256"] = sha256_file(path)

    def test_full_analysis_replays_provenance_artifacts_and_metrics(self) -> None:
        result = analyze(self.fixture.args())
        self.assertEqual(result["schema_version"], "nfa_vit_posthoc_analysis_v1")
        self.assertEqual(result["provenance_integrity"]["status"], "ok")
        self.assertEqual(result["artifact_integrity"]["status"], "ok")
        self.assertEqual(result["artifact_integrity"]["pairs"], 3)
        self.assertEqual(result["artifact_integrity"]["result_images"], 6)
        self.assertEqual(result["artifact_integrity"]["checked_files"], 39)
        self.assertEqual(
            result["artifact_integrity"][
                "observed_maximum_absolute_error"
            ]["threshold_mask_disagreements"],
            0,
        )
        self.assertEqual(
            result["fixed_threshold_metrics"]["detection"]["auroc"],
            1.0,
        )
        self.assertEqual(result["overall"]["native"]["pairs"], 3)
        self.assertIn("lodging", result["by_domain"]["slices"])
        self.assertIn("restaurant", result["by_domain"]["slices"])
        self.assertEqual(
            result["box_hit_at_native_mask_threshold_0_5"]["any_overlap"][
                "images"
            ],
            3,
        )
        self.assertIsNone(result["prefix_reproducibility"])

    def test_posthoc_bootstrap_seed_is_independent_of_recorded_summary(self) -> None:
        args = self.fixture.args()
        args.bootstrap_iterations = 9
        args.bootstrap_seed = 991
        result = analyze(args)
        self.assertEqual(result["overall"]["native"]["bootstrap_samples"], 9)
        self.assertEqual(result["overall"]["native"]["seed"], 991)
        self.assertEqual(
            result["fixed_threshold_metrics"]["detection"]["auroc"],
            1.0,
        )

    def test_artifact_audit_rejects_probability_tamper(self) -> None:
        row = self.fixture.result_rows[1]
        record = row["artifact_paths"]["probability_512_npy"]
        path = self.fixture.repo / record["path"]
        probability = np.load(path, allow_pickle=False)
        probability[0, 0] = np.float32(0.75)
        np.save(path, probability, allow_pickle=False)
        record["sha256"] = sha256_file(path)
        with self.assertRaisesRegex(ValueError, "model-probability replay"):
            audit_artifacts(
                repo_root=self.fixture.repo,
                expected_rows=self.fixture.input_rows,
                result_rows=self.fixture.result_rows,
                manifest=self.fixture.manifest,
            )

    def test_artifact_audit_rejects_non_strict_mask(self) -> None:
        row = self.fixture.result_rows[0]
        native_record = row["artifact_paths"]["probability_native_npy"]
        native_path = self.fixture.repo / native_record["path"]
        native = np.load(native_path, allow_pickle=False)
        native[0, 0] = np.float32(0.5)
        np.save(native_path, native, allow_pickle=False)
        native_record["sha256"] = sha256_file(native_path)

        mask_record = row["artifact_paths"]["mask_native_png"]
        mask_path = self.fixture.repo / mask_record["path"]
        with Image.open(mask_path) as opened:
            mask = np.asarray(opened).copy()
        mask[0, 0] = 255
        Image.fromarray(mask, mode="L").save(mask_path)
        mask_record["sha256"] = sha256_file(mask_path)
        with self.assertRaisesRegex(ValueError, "native-probability replay"):
            audit_artifacts(
                repo_root=self.fixture.repo,
                expected_rows=self.fixture.input_rows,
                result_rows=self.fixture.result_rows,
                manifest=self.fixture.manifest,
            )

    def test_artifact_audit_rejects_classifier_sigmoid_mismatch(self) -> None:
        self.fixture.result_rows[0]["score"] = 0.2
        with self.assertRaisesRegex(ValueError, "classification .*sigmoid"):
            audit_artifacts(
                repo_root=self.fixture.repo,
                expected_rows=self.fixture.input_rows,
                result_rows=self.fixture.result_rows,
                manifest=self.fixture.manifest,
            )

    def test_t1_nextafter_score_cannot_flip_replayed_decision(self) -> None:
        row = self.fixture.result_rows[0]
        score = float(
            np.nextafter(
                np.float32(0.5),
                np.float32(1.0),
            )
        )
        row["score"] = score
        row["classification_raw_logit"] = 0.0
        row["classification_score"] = score
        row["classification_decision"] = True
        row["decision"] = "forged"
        row["classification"].update(
            {
                "raw_logit": 0.0,
                "probability": score,
                "score": score,
                "decision": True,
            }
        )
        with self.assertRaisesRegex(
            ValueError,
            "classification threshold decision mismatch",
        ):
            audit_artifacts(
                repo_root=self.fixture.repo,
                expected_rows=self.fixture.input_rows,
                result_rows=self.fixture.result_rows,
                manifest=self.fixture.manifest,
            )

    def test_t1_rejects_nested_and_top_level_decision_drift(self) -> None:
        for field in ("nested", "top-level"):
            with self.subTest(field=field):
                rows = copy.deepcopy(self.fixture.result_rows)
                if field == "nested":
                    rows[0]["classification"]["decision"] = True
                    message = "nested classification decision"
                else:
                    rows[0]["classification_decision"] = True
                    message = "top-level classification decision"
                with self.assertRaisesRegex(ValueError, message):
                    audit_artifacts(
                        repo_root=self.fixture.repo,
                        expected_rows=self.fixture.input_rows,
                        result_rows=rows,
                        manifest=self.fixture.manifest,
                    )

    def test_t2_nextafter_probability_decision_is_rejected(self) -> None:
        row = self.fixture.result_rows[0]
        raw = np.zeros((128, 128), dtype=np.float32)
        resized = np.zeros((512, 512), dtype=np.float32)
        probability = np.full((512, 512), 0.5, dtype=np.float32)
        probability[0, 0] = np.nextafter(
            np.float32(0.5),
            np.float32(1.0),
        )
        native = _bilinear_align_corners_false(
            probability,
            width=12,
            height=8,
        )
        self._replace_npy(row, "decoder_logits_128_npy", raw)
        self._replace_npy(row, "resized_logits_512_npy", resized)
        self._replace_npy(row, "probability_512_npy", probability)
        self._replace_npy(row, "probability_native_npy", native)
        with self.assertRaisesRegex(
            ValueError,
            "resized-logits-to-probability threshold decision mismatch",
        ):
            audit_artifacts(
                repo_root=self.fixture.repo,
                expected_rows=self.fixture.input_rows,
                result_rows=self.fixture.result_rows,
                manifest=self.fixture.manifest,
            )

    def test_t2_cumulative_allclose_drift_cannot_flip_native_decision(
        self,
    ) -> None:
        row = self.fixture.result_rows[0]
        raw = np.full((128, 128), -0.01, dtype=np.float32)
        raw[:, 16:] = np.float32(0.01)
        resized = _bilinear_align_corners_false(
            raw,
            width=512,
            height=512,
        )
        replay_probability = _sigmoid_float32(resized)
        probability = replay_probability + np.float32(5e-7)
        native = _bilinear_align_corners_false(
            probability,
            width=12,
            height=8,
        )
        # At native column 1 the unperturbed symmetric sigmoid replay is
        # exactly 0.5, while the individually allclose intermediate drift
        # accumulates to a strict-positive decision.
        self.assertEqual(
            _bilinear_align_corners_false(
                replay_probability,
                width=12,
                height=8,
            )[0, 1],
            np.float32(0.5),
        )
        self.assertGreater(native[0, 1], np.float32(0.5))
        self._replace_npy(row, "decoder_logits_128_npy", raw)
        self._replace_npy(row, "resized_logits_512_npy", resized)
        self._replace_npy(row, "probability_512_npy", probability)
        self._replace_npy(row, "probability_native_npy", native)
        with self.assertRaisesRegex(
            ValueError,
            "raw-to-native end-to-end threshold decision mismatch",
        ):
            audit_artifacts(
                repo_root=self.fixture.repo,
                expected_rows=self.fixture.input_rows,
                result_rows=self.fixture.result_rows,
                manifest=self.fixture.manifest,
            )

    def test_artifact_audit_fails_closed_for_unavailable_recorded_device(
        self,
    ) -> None:
        manifest = copy.deepcopy(self.fixture.manifest)
        manifest["runtime_contract"]["accelerator"][
            "requested_device"
        ] = "cuda:99999"
        with self.assertRaisesRegex(
            ValueError,
            "CUDA inference device",
        ):
            audit_artifacts(
                repo_root=self.fixture.repo,
                expected_rows=self.fixture.input_rows,
                result_rows=self.fixture.result_rows,
                manifest=manifest,
            )

    def test_provenance_rejects_modified_source(self) -> None:
        (self.fixture.nfa_root / "models.py").write_text(
            "modified\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            analyze(self.fixture.args())

    def test_provenance_requires_summary_identity_fields(self) -> None:
        for field, message in (
            ("run_id", "summary run ID"),
            ("run_manifest_fingerprint", "summary fingerprint"),
            ("checkpoint_sha256", "summary checkpoint"),
        ):
            with self.subTest(field=field):
                summary = copy.deepcopy(self.fixture.summary)
                del summary[field]
                atomic_write_json(self.fixture.summary_path, summary)
                with self.assertRaisesRegex(ValueError, message):
                    analyze(self.fixture.args())
                atomic_write_json(
                    self.fixture.summary_path,
                    self.fixture.summary,
                )

    def test_provenance_validates_every_physical_result_row(self) -> None:
        stale = copy.deepcopy(self.fixture.result_rows[0])
        stale["status"] = "error"
        stale["run_manifest_fingerprint"] = "0" * 64
        atomic_write_jsonl(
            self.fixture.results_path,
            [stale, *self.fixture.result_rows],
        )
        with self.assertRaisesRegex(ValueError, "physical result row 1"):
            analyze(self.fixture.args())

    def test_production_checkpoint_contract_is_fail_closed_until_frozen(
        self,
    ) -> None:
        args = self.fixture.args()
        delattr(args, "checkpoint_sha256_test_contract")
        with mock.patch.dict(
            "eval.opensource.run_nfa_vit.CHECKPOINT",
            {"sha256": None},
        ):
            with self.assertRaisesRegex(
                ValueError,
                "official checkpoint SHA-256 is not frozen",
            ):
                analyze(args)

    def test_result_history_preserves_resume_rows(self) -> None:
        stale = copy.deepcopy(self.fixture.result_rows[0])
        stale["status"] = "error"
        rows = [stale, *self.fixture.result_rows]
        history = summarize_result_history(rows)
        self.assertEqual(history["physical_rows"], 7)
        self.assertEqual(history["unique_ids"], 6)
        self.assertEqual(history["duplicate_rows"], 1)
        self.assertEqual(history["recovered_error_to_ok"], 1)

    def test_prefix_reproducibility_accepts_exact_first_pair(self) -> None:
        prefix_id = "nfa_fixture_prefix"
        prefix_rows = copy.deepcopy(self.fixture.result_rows[:2])
        prefix_manifest = copy.deepcopy(self.fixture.manifest)
        prefix_manifest["run_id"] = prefix_id
        prefix_manifest["ordered_inputs"] = prefix_manifest["ordered_inputs"][:2]
        prefix_manifest["fingerprint"] = _manifest_fingerprint(prefix_manifest)
        for row in prefix_rows:
            row["run_id"] = prefix_id
            row["run_manifest_fingerprint"] = prefix_manifest["fingerprint"]
        prefix_results = self.fixture.results_dir / f"{prefix_id}.jsonl"
        prefix_manifest_path = (
            self.fixture.results_dir / f"{prefix_id}.run_manifest.json"
        )
        atomic_write_jsonl(prefix_results, prefix_rows)
        atomic_write_json(prefix_manifest_path, prefix_manifest)
        result = audit_prefix_reproducibility(
            repo_root=self.fixture.repo,
            full_expected=self.fixture.input_rows,
            full_rows=self.fixture.result_rows,
            prefix_run_id=prefix_id,
            prefix_results_dir=self.fixture.results_dir,
        )
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["prefix_images"], 2)
        self.assertEqual(
            result["physical_prefix_rows_provenance_validated"],
            2,
        )

    def test_prefix_reproducibility_rejects_copied_full_rows(self) -> None:
        prefix_id = "nfa_fixture_copied_rows"
        prefix_rows = copy.deepcopy(self.fixture.result_rows[:2])
        prefix_manifest = copy.deepcopy(self.fixture.manifest)
        prefix_manifest["run_id"] = prefix_id
        prefix_manifest["ordered_inputs"] = prefix_manifest["ordered_inputs"][:2]
        prefix_manifest["fingerprint"] = _manifest_fingerprint(prefix_manifest)
        atomic_write_jsonl(
            self.fixture.results_dir / f"{prefix_id}.jsonl",
            prefix_rows,
        )
        atomic_write_json(
            self.fixture.results_dir / f"{prefix_id}.run_manifest.json",
            prefix_manifest,
        )
        with self.assertRaisesRegex(ValueError, "physical prefix row 1"):
            audit_prefix_reproducibility(
                repo_root=self.fixture.repo,
                full_expected=self.fixture.input_rows,
                full_rows=self.fixture.result_rows,
                prefix_run_id=prefix_id,
                prefix_results_dir=self.fixture.results_dir,
            )

    def test_prefix_reproducibility_rejects_score_drift(self) -> None:
        prefix_id = "nfa_fixture_bad_prefix"
        prefix_rows = copy.deepcopy(self.fixture.result_rows[:2])
        prefix_rows[1]["score"] = 0.8
        prefix_manifest = copy.deepcopy(self.fixture.manifest)
        prefix_manifest["run_id"] = prefix_id
        prefix_manifest["ordered_inputs"] = prefix_manifest["ordered_inputs"][:2]
        prefix_manifest["fingerprint"] = _manifest_fingerprint(prefix_manifest)
        for row in prefix_rows:
            row["run_id"] = prefix_id
            row["run_manifest_fingerprint"] = prefix_manifest["fingerprint"]
        atomic_write_jsonl(
            self.fixture.results_dir / f"{prefix_id}.jsonl",
            prefix_rows,
        )
        atomic_write_json(
            self.fixture.results_dir / f"{prefix_id}.run_manifest.json",
            prefix_manifest,
        )
        with self.assertRaisesRegex(ValueError, "score"):
            audit_prefix_reproducibility(
                repo_root=self.fixture.repo,
                full_expected=self.fixture.input_rows,
                full_rows=self.fixture.result_rows,
                prefix_run_id=prefix_id,
                prefix_results_dir=self.fixture.results_dir,
            )


if __name__ == "__main__":
    unittest.main()
