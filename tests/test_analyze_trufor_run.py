import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from eval.opensource.analyze_trufor_run import (
    _selection_contract,
    summarize_result_history,
    validate_provenance,
)
from eval.opensource.common import atomic_write_json, atomic_write_jsonl, sha256_file
from eval.opensource.run_trufor import (
    CHECKPOINT_EPOCH,
    CHECKPOINT_SHA256,
    MODEL_CONFIG_SHA256,
    MODEL_LICENSE_SHA256,
    MODEL_SOURCE_COMMIT,
)


class AnalyzeTruForRunTest(unittest.TestCase):
    def _fixture(self, root: Path):
        run_id = "trufor_test"
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
                "canonical_sha256": f"{index + 1:064x}",
                "gt_mask_sha256": None if kind == "real" else f"{3:064x}",
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

        identity_rows = {
            row["sample_id"]: {
                "schema_version": "opensource_result_v1",
                "run_id": run_id,
                "input_manifest_sha256": inputs_sha256,
                "id": row["sample_id"],
                "task_id": row["task_id"],
                "pair_rank": row["pair_rank"],
                "domain": row["domain"],
                "kind": row["kind"],
                "label": row["label"],
                "image_path": row["canonical_path"],
                "image_sha256": row["canonical_sha256"],
                "image_size": [row["width"], row["height"]],
                "model": "TruFor",
                "model_slug": "trufor_cvpr2023",
                "checkpoint_sha256": CHECKPOINT_SHA256,
            }
            for row in input_rows
        }
        result_rows = [
            {**identity_rows["real"], "status": "ok"},
            {
                **identity_rows["forged"],
                "status": "error",
                "error_type": "RuntimeError",
            },
            {**identity_rows["forged"], "status": "ok"},
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
                    json.dumps(
                        ordered_inputs,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("utf-8")
                ).hexdigest(),
            },
            "ordered_inputs": ordered_inputs,
            "model": {
                "name": "TruFor",
                "model_slug": "trufor_cvpr2023",
                "source_commit": MODEL_SOURCE_COMMIT,
                "source_tracked_clean": True,
                "license": {"sha256": MODEL_LICENSE_SHA256},
                "configuration": {"sha256": MODEL_CONFIG_SHA256},
                "checkpoint": {
                    "sha256": CHECKPOINT_SHA256,
                    "epoch": CHECKPOINT_EPOCH,
                    "strict_load": True,
                    "safe_weights_only_load": True,
                },
            },
            "inference": {
                "classification_threshold": 0.5,
                "mask_threshold": 0.5,
            },
            "expected_pairs": 1,
            "expected_images": 2,
        }
        fingerprint = hashlib.sha256(
            json.dumps(
                immutable,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        manifest = {
            **immutable,
            "fingerprint": fingerprint,
            "created_at": "test",
            "adapter": {},
            "environment": {},
        }
        summary = {
            "schema_version": "opensource_summary_v1",
            "run_id": run_id,
            "condition": "test",
            "model": "TruFor",
            "model_slug": "trufor_cvpr2023",
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "input_manifest_sha256": inputs_sha256,
            "run_manifest_fingerprint": fingerprint,
            "coverage": {
                "expected_images": 2,
                "result_images": 2,
                "valid_images": 2,
                "error_images": 0,
                "missing_images": 0,
            },
            "detection": {"threshold": 0.5},
            "localization_forged": {
                "native": {"micro_at_threshold": {"threshold": 0.5}}
            },
        }
        return run_id, input_path, input_rows, result_rows, manifest, summary

    def test_error_then_ok_history_is_allowed_and_reported(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = self._fixture(root)
            run_id, input_path, inputs, results, manifest, summary = fixture
            provenance = validate_provenance(
                repo_root=root,
                run_id=run_id,
                input_path=input_path,
                input_rows=inputs,
                result_rows=results,
                manifest=manifest,
                summary=summary,
            )
            history = summarize_result_history(results)

        self.assertEqual(provenance["physical_result_rows_validated"], 3)
        self.assertEqual(provenance["latest_result_rows_validated"], 2)
        self.assertEqual(history["duplicate_rows"], 1)
        self.assertEqual(history["recovered_error_to_ok"], 1)
        self.assertEqual(history["duplicate_histories"][0]["statuses"], ["error", "ok"])

    def test_every_historical_row_checkpoint_is_validated(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = self._fixture(root)
            run_id, input_path, inputs, results, manifest, summary = fixture
            results[1]["checkpoint_sha256"] = "wrong"
            with self.assertRaisesRegex(
                ValueError,
                "result row 2 field checkpoint_sha256",
            ):
                validate_provenance(
                    repo_root=root,
                    run_id=run_id,
                    input_path=input_path,
                    input_rows=inputs,
                    result_rows=results,
                    manifest=manifest,
                    summary=summary,
                )

    def test_manifest_fingerprint_is_recomputed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = self._fixture(root)
            run_id, input_path, inputs, results, manifest, summary = fixture
            manifest["expected_images"] = 999
            with self.assertRaisesRegex(ValueError, "run manifest fingerprint"):
                validate_provenance(
                    repo_root=root,
                    run_id=run_id,
                    input_path=input_path,
                    input_rows=inputs,
                    result_rows=results,
                    manifest=manifest,
                    summary=summary,
                )


if __name__ == "__main__":
    unittest.main()
