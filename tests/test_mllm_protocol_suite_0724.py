import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

from eval.mllm.prompts import PROTOCOL_SUITE_VERSION, PROTOCOL_VERSIONS
from eval.mllm.results import (
    completed_aggregate_keys,
    completed_raw_keys,
    successful_raw,
)
from eval.mllm.run_mllm import _existing_or_new_manifest, _protocol_manifest


class MLLMProtocolSuite0724Test(unittest.TestCase):
    def test_both_protocols_share_one_suite_without_losing_leaf_versions(self):
        manifest = _protocol_manifest(["detection", "localization"])

        self.assertEqual(manifest["version"], PROTOCOL_SUITE_VERSION)
        self.assertEqual(manifest["suite_version"], PROTOCOL_SUITE_VERSION)
        self.assertEqual(manifest["keys"], ["detection", "localization"])
        self.assertEqual(manifest["versions"], PROTOCOL_VERSIONS)
        self.assertEqual(manifest["replicates_required"], 3)

    def test_single_protocol_keeps_its_existing_version(self):
        manifest = _protocol_manifest(["detection"])

        self.assertEqual(manifest["version"], PROTOCOL_VERSIONS["detection"])
        self.assertNotIn("suite_version", manifest)
        self.assertEqual(
            manifest["versions"],
            {"detection": PROTOCOL_VERSIONS["detection"]},
        )

    def test_old_single_protocol_manifest_without_version_map_can_resume(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "run.run_manifest.json"
            fresh = {
                "run_id": "existing_detection",
                "condition": "condition",
                "model": {"id": "model", "concurrency": 15},
                "protocol": _protocol_manifest(["detection"]),
                "input": {"manifest_sha256": "input"},
                "image": {"transport": "base64"},
            }
            existing = deepcopy(fresh)
            del existing["protocol"]["versions"]
            path.write_text(json.dumps(existing), encoding="utf-8")

            resumed = _existing_or_new_manifest(fresh, path)

            self.assertEqual(resumed, existing)

    def test_resume_filters_each_protocol_by_its_own_version(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw_path = root / "run.raw.jsonl"
            aggregate_path = root / "run.jsonl"
            parsed = {
                "reasoning": "test",
                "decision": "not_edited",
                "p_ai_edited": 10,
                "evidence": [],
            }
            raw_rows = [
                {
                    "id": "image",
                    "protocol_key": "detection",
                    "protocol_version": PROTOCOL_VERSIONS["detection"],
                    "replicate_index": 1,
                    "status": "ok",
                    "parsed": parsed,
                },
                {
                    "id": "image",
                    "protocol_key": "localization",
                    "protocol_version": PROTOCOL_VERSIONS["localization"],
                    "replicate_index": 1,
                    "status": "ok",
                    "parsed": parsed,
                },
                {
                    "id": "stale",
                    "protocol_key": "detection",
                    "protocol_version": PROTOCOL_VERSIONS["localization"],
                    "replicate_index": 1,
                    "status": "ok",
                    "parsed": parsed,
                },
            ]
            raw_path.write_text(
                "".join(json.dumps(row) + "\n" for row in raw_rows),
                encoding="utf-8",
            )
            aggregate_rows = [
                {
                    "id": "image",
                    "protocol_key": protocol,
                    "protocol_version": PROTOCOL_VERSIONS[protocol],
                    "status": "ok",
                }
                for protocol in ("detection", "localization")
            ]
            aggregate_rows.append({
                "id": "stale",
                "protocol_key": "localization",
                "protocol_version": PROTOCOL_VERSIONS["detection"],
                "status": "ok",
            })
            aggregate_path.write_text(
                "".join(json.dumps(row) + "\n" for row in aggregate_rows),
                encoding="utf-8",
            )

            selector = dict(PROTOCOL_VERSIONS)
            self.assertEqual(
                completed_raw_keys(raw_path, selector),
                {
                    ("image", "detection", 1),
                    ("image", "localization", 1),
                },
            )
            self.assertEqual(
                set(successful_raw(raw_path, selector)),
                {
                    ("image", "detection", 1),
                    ("image", "localization", 1),
                },
            )
            self.assertEqual(
                completed_aggregate_keys(aggregate_path, selector),
                {
                    ("image", "detection"),
                    ("image", "localization"),
                },
            )


if __name__ == "__main__":
    unittest.main()
