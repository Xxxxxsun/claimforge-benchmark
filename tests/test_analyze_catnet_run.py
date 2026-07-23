import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from eval.opensource.analyze_catnet_run import (
    _jpeg_evidence_hashes,
    _load_pairs,
    _select_manifest_inputs,
    _selection_contract,
    audit_and_best_threshold,
    summarize_result_history,
    validate_provenance,
)
from eval.opensource.catnet_metrics import binary_pixel_metrics
from eval.opensource.common import (
    atomic_write_json,
    atomic_write_jsonl,
    sha256_file,
    stable_json,
)
from eval.opensource.run_catnet import (
    CHECKPOINT_EPOCH,
    CHECKPOINT_SHA256,
    MODEL_CONFIG_SHA256,
    MODEL_LICENSE_SHA256,
    MODEL_NETWORK_SHA256,
    MODEL_SOURCE_COMMIT,
)

JPEGIO_AVAILABLE = importlib.util.find_spec("jpegio") is not None


class AnalyzeCATNetRunTest(unittest.TestCase):
    def _save_npy(self, path: Path, value: np.ndarray) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            np.save(handle, value, allow_pickle=False)

    def _save_image(self, path: Path, value: np.ndarray, mode: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(value, mode=mode).save(path)

    def _save_jpeg(self, path: Path, value: np.ndarray) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(value, mode="RGB").save(
            path,
            format="JPEG",
            quality=95,
            subsampling=0,
            optimize=False,
        )

    def _fixture(self, root: Path):
        run_id = "catnet_test"
        image_dir = root / "images"
        real_image_path = image_dir / "real.jpg"
        forged_image_path = image_dir / "forged.jpg"
        gt_path = image_dir / "forged_gt.png"
        self._save_jpeg(
            real_image_path,
            np.zeros((2, 2, 3), dtype=np.uint8),
        )
        self._save_jpeg(
            forged_image_path,
            np.full((2, 2, 3), 64, dtype=np.uint8),
        )
        target_uint8 = np.asarray([[255, 0], [0, 0]], dtype=np.uint8)
        self._save_image(gt_path, target_uint8, "L")

        input_rows = [
            {
                "schema_version": "claimforge_mouse_canonical_v1",
                "dataset_id": "test-dataset",
                "rank": index,
                "pair_rank": 0,
                "sample_id": kind,
                "task_id": "task",
                "domain": "lodging",
                "kind": kind,
                "label": index,
                "canonical_path": f"images/{kind}.jpg",
                "canonical_sha256": sha256_file(
                    real_image_path if kind == "real" else forged_image_path
                ),
                "gt_mask_path": (
                    None if kind == "real" else "images/forged_gt.png"
                ),
                "gt_mask_kind": "all_zero" if kind == "real" else "exact_diff",
                "gt_mask_sha256": (
                    None if kind == "real" else sha256_file(gt_path)
                ),
                "edit_region_xyxy": [0, 0, 1, 1],
                "width": 2,
                "height": 2,
            }
            for index, kind in enumerate(("real", "forged"))
        ]
        input_path = root / "inputs.jsonl"
        atomic_write_jsonl(input_path, input_rows)
        inputs_sha256 = sha256_file(input_path)
        release = {
            "schema_version": "claimforge_mouse_canonical_v1",
            "dataset_id": "test-dataset",
            "contract_sha256": "contract",
            "inputs_path": "inputs.jsonl",
            "inputs_sha256": inputs_sha256,
        }
        atomic_write_json(root / "manifest.json", release)

        scores = {
            "real": np.asarray([[0.1, 0.1], [0.1, 0.1]], dtype=np.float32),
            "forged": np.asarray([[0.9, 0.1], [0.1, 0.1]], dtype=np.float32),
        }
        artifacts: dict[str, dict[str, Path]] = {}
        for kind in ("real", "forged"):
            logits_path = root / "artifacts" / "raw_logits" / f"{kind}.npy"
            score_path = root / "artifacts" / "score_maps" / f"{kind}.npy"
            mask_path = root / "artifacts" / "masks" / f"{kind}.png"
            self._save_npy(
                logits_path,
                np.zeros((2, 2, 2), dtype=np.float32),
            )
            self._save_npy(score_path, scores[kind])
            self._save_image(
                mask_path,
                np.where(scores[kind] >= 0.5, 255, 0).astype(np.uint8),
                "L",
            )
            artifacts[kind] = {
                "logits": logits_path,
                "score": score_path,
                "mask": mask_path,
            }

        identity_rows = {
            row["sample_id"]: {
                "schema_version": "opensource_result_v1",
                "run_id": run_id,
                "input_manifest_sha256": inputs_sha256,
                "id": row["sample_id"],
                "rank": row["rank"],
                "task_id": row["task_id"],
                "pair_rank": row["pair_rank"],
                "domain": row["domain"],
                "kind": row["kind"],
                "label": row["label"],
                "image_path": row["canonical_path"],
                "image_sha256": row["canonical_sha256"],
                "image_size": [row["width"], row["height"]],
                "gt_mask_kind": row["gt_mask_kind"],
                "gt_mask_sha256": row["gt_mask_sha256"],
                "edit_region_xyxy": row["edit_region_xyxy"],
                "model": "CAT-Net v2",
                "model_slug": "catnet_v2_ijcv2022",
                "checkpoint_sha256": CHECKPOINT_SHA256,
                "valid_for_t1": False,
            }
            for row in input_rows
        }
        ok_rows: dict[str, dict] = {}
        for kind in ("real", "forged"):
            artifact = artifacts[kind]
            image_path = (
                real_image_path if kind == "real" else forged_image_path
            )
            jpeg_evidence = (
                _jpeg_evidence_hashes(image_path)
                if JPEGIO_AVAILABLE
                else {
                    "qtable_sha256": "a" * 64,
                    "dct_y_sha256": "b" * 64,
                }
            )
            target = (
                target_uint8 > 0
                if kind == "forged"
                else np.zeros((2, 2), dtype=bool)
            )
            ok_rows[kind] = {
                **identity_rows[kind],
                "status": "ok",
                "valid_for_t2": True,
                **jpeg_evidence,
                "raw_logits_path": str(artifact["logits"].relative_to(root)),
                "raw_logits_sha256": sha256_file(artifact["logits"]),
                "raw_logits_shape": [2, 2, 2],
                "score_map_path": str(artifact["score"].relative_to(root)),
                "score_map_sha256": sha256_file(artifact["score"]),
                "score_map_shape": [2, 2],
                "mask_path": str(artifact["mask"].relative_to(root)),
                "mask_sha256": sha256_file(artifact["mask"]),
                "mask_shape": [2, 2],
                "mask_threshold": 0.5,
                "localization": {
                    "native": binary_pixel_metrics(
                        scores[kind],
                        target,
                        0.5,
                        include_ap=kind == "forged",
                    )
                },
            }
        result_rows = [
            ok_rows["real"],
            {
                **identity_rows["forged"],
                "status": "error",
                "valid_for_t2": False,
                "error_type": "RuntimeError",
            },
            ok_rows["forged"],
        ]

        ordered_inputs = _selection_contract(input_rows)
        immutable = {
            "schema_version": "opensource_run_manifest_v1",
            "run_id": run_id,
            "condition": "test",
            "input": {
                "dataset_id": "test-dataset",
                "dataset_manifest": "manifest.json",
                "dataset_contract_sha256": "contract",
                "inputs_manifest": "inputs.jsonl",
                "inputs_sha256": inputs_sha256,
                "selection_sha256": hashlib.sha256(
                    stable_json(ordered_inputs).encode("utf-8")
                ).hexdigest(),
            },
            "ordered_inputs": ordered_inputs,
            "model": {
                "name": "CAT-Net v2",
                "model_slug": "catnet_v2_ijcv2022",
                "source_commit": MODEL_SOURCE_COMMIT,
                "source_tracked_clean": True,
                "positive_class_index": 1,
                "license": {
                    "sha256": MODEL_LICENSE_SHA256,
                    "scope": "hrnet_component_only",
                    "project_wide_status": "no_project_wide_license_found",
                },
                "configuration": {
                    "sha256": MODEL_CONFIG_SHA256,
                    "network_sha256": MODEL_NETWORK_SHA256,
                },
                "checkpoint": {
                    "sha256": CHECKPOINT_SHA256,
                    "epoch": CHECKPOINT_EPOCH,
                    "strict_load": True,
                    "safe_weights_only_load": True,
                },
            },
            "inference": {
                "precision": "float32",
                "batch_size": 1,
                "input_resize": "none",
                "mask_threshold": 0.5,
                "mask_threshold_comparison": "greater_than_or_equal",
            },
            "expected_pairs": 1,
            "expected_images": 2,
            "adapter_contract": [
                {
                    "path": "inputs.jsonl",
                    "sha256": inputs_sha256,
                }
            ],
        }
        fingerprint = hashlib.sha256(
            stable_json(immutable).encode("utf-8")
        ).hexdigest()
        manifest = {
            **immutable,
            "fingerprint": fingerprint,
            "created_at": "test",
            "adapter": {},
            "environment": {},
        }
        for row in result_rows:
            row["run_manifest_fingerprint"] = fingerprint
        summary = {
            "schema_version": "opensource_summary_v1",
            "run_id": run_id,
            "condition": "test",
            "model": "CAT-Net v2",
            "model_slug": "catnet_v2_ijcv2022",
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "input_manifest_sha256": inputs_sha256,
            "run_manifest_fingerprint": fingerprint,
            "task_scope": {
                "primary_task": "T2_localization",
                "mask_threshold": 0.5,
            },
            "coverage": {
                "expected_images": 2,
                "result_images": 2,
                "valid_images": 2,
                "error_images": 0,
                "missing_images": 0,
            },
            "localization": {
                "native": {"micro_at_threshold": {"threshold": 0.5}}
            },
            "real_localization": {
                "native": {"micro_at_threshold": {"threshold": 0.5}}
            },
        }
        return {
            "run_id": run_id,
            "input_path": input_path,
            "input_rows": input_rows,
            "result_rows": result_rows,
            "manifest": manifest,
            "summary": summary,
            "artifacts": artifacts,
        }

    def _validate(self, root: Path, fixture: dict):
        return validate_provenance(
            repo_root=root,
            run_id=fixture["run_id"],
            input_path=fixture["input_path"],
            input_rows=fixture["input_rows"],
            result_rows=fixture["result_rows"],
            manifest=fixture["manifest"],
            summary=fixture["summary"],
        )

    def test_error_then_ok_history_is_allowed_and_reported(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = self._fixture(root)
            provenance = self._validate(root, fixture)
            history = summarize_result_history(fixture["result_rows"])

        self.assertEqual(provenance["physical_result_rows_validated"], 3)
        self.assertEqual(provenance["latest_result_rows_validated"], 2)
        self.assertEqual(history["duplicate_rows"], 1)
        self.assertEqual(history["recovered_error_to_ok"], 1)
        self.assertEqual(
            history["duplicate_histories"][0]["statuses"],
            ["error", "ok"],
        )

    def test_manifest_fingerprint_is_recomputed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = self._fixture(root)
            fixture["manifest"]["expected_images"] = 999
            with self.assertRaisesRegex(ValueError, "run manifest fingerprint"):
                self._validate(root, fixture)

    def test_manifest_selection_filters_full_inputs_in_declared_order(self):
        full_rows = [
            {"sample_id": "unselected"},
            {"sample_id": "forged"},
            {"sample_id": "real"},
        ]
        manifest = {
            "ordered_inputs": [
                {"sample_id": "real"},
                {"sample_id": "forged"},
            ]
        }

        selected = _select_manifest_inputs(full_rows, manifest)

        self.assertEqual(
            [row["sample_id"] for row in selected],
            ["real", "forged"],
        )

    def test_project_wide_license_must_not_be_claimed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = self._fixture(root)
            manifest = fixture["manifest"]
            manifest["model"]["license"]["project_wide_status"] = "licensed"
            immutable = {
                key: value
                for key, value in manifest.items()
                if key not in {"fingerprint", "created_at", "adapter", "environment"}
            }
            manifest["fingerprint"] = hashlib.sha256(
                stable_json(immutable).encode("utf-8")
            ).hexdigest()
            fixture["summary"]["run_manifest_fingerprint"] = manifest[
                "fingerprint"
            ]
            with self.assertRaisesRegex(
                ValueError,
                "project-wide license status",
            ):
                self._validate(root, fixture)

    def test_t1_score_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = self._fixture(root)
            fixture["result_rows"][0]["score"] = 0.9
            with self.assertRaisesRegex(ValueError, "forbidden T1 fields"):
                self._validate(root, fixture)

    @unittest.skipUnless(JPEGIO_AVAILABLE, "jpegio is required for CAT audit")
    def test_artifact_audit_checks_logits_score_mask_and_gt(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = self._fixture(root)
            pairs = _load_pairs(
                fixture["result_rows"],
                fixture["input_rows"],
            )
            audit = audit_and_best_threshold(pairs, repo_root=root, bins=256)

        self.assertEqual(audit["artifact_integrity"]["status"], "ok")
        self.assertEqual(audit["artifact_integrity"]["checked_files"], 9)
        self.assertEqual(
            audit["localization_best_threshold"]["per_image_oracle"]["f1_mean"],
            1.0,
        )

    @unittest.skipUnless(JPEGIO_AVAILABLE, "jpegio is required for CAT audit")
    def test_threshold_mask_must_bit_exactly_match_score(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = self._fixture(root)
            forged = fixture["result_rows"][-1]
            mask_path = fixture["artifacts"]["forged"]["mask"]
            self._save_image(
                mask_path,
                np.zeros((2, 2), dtype=np.uint8),
                "L",
            )
            forged["mask_sha256"] = sha256_file(mask_path)
            pairs = _load_pairs(
                fixture["result_rows"],
                fixture["input_rows"],
            )
            with self.assertRaisesRegex(ValueError, "threshold mask mismatch"):
                audit_and_best_threshold(pairs, repo_root=root, bins=256)

    @unittest.skipUnless(JPEGIO_AVAILABLE, "jpegio is required for CAT audit")
    def test_raw_logits_shape_is_derived_from_padded_input(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = self._fixture(root)
            forged = fixture["result_rows"][-1]
            logits_path = fixture["artifacts"]["forged"]["logits"]
            self._save_npy(
                logits_path,
                np.zeros((2, 1, 1), dtype=np.float32),
            )
            forged["raw_logits_sha256"] = sha256_file(logits_path)
            forged["raw_logits_shape"] = [2, 1, 1]
            pairs = _load_pairs(
                fixture["result_rows"],
                fixture["input_rows"],
            )
            with self.assertRaisesRegex(ValueError, "invalid raw logits shape"):
                audit_and_best_threshold(pairs, repo_root=root, bins=256)


if __name__ == "__main__":
    unittest.main()
