import json
import copy
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from eval.mllm.schema import SchemaError
from eval.mllm.results import completed_raw_keys, successful_raw
from eval.mllm.zoom_agent import (
    AGENT_PROTOCOL_VERSION,
    create_zoom_crop,
    parse_agent_action,
    run_agent_episode,
    summarize_agent_run,
)


def _zoom(box):
    return json.dumps({
        "action": "zoom_in",
        "reasoning": "inspect a suspicious edge",
        "bbox_px": box,
    })


def _final():
    return json.dumps({
        "action": "final",
        "reasoning": "The enlarged edge has inconsistent texture and lighting.",
        "decision": "edited",
        "p_ai_edited": 83,
        "evidence": ["inconsistent edge"],
        "regions": [{
            "bbox_px": [50, 20, 75, 40],
            "confidence": 88,
            "evidence": "edge halo",
        }],
    })


class _FakeClient:
    def __init__(self, replies):
        self.replies = list(replies)
        self.histories = []

    def image_part(self, image):
        return {"type": "image_url", "image_url": {"url": image}}

    def image_data_url(self, path):
        self.asserted_crop_path = path
        return "data:image/png;base64,CROP"

    def call_messages(self, system_prompt, messages):
        self.histories.append((system_prompt, copy.deepcopy(messages)))
        return self.replies.pop(0), 7


class ZoomAgentSchemaTest(unittest.TestCase):
    def test_parses_zoom_and_final_in_original_coordinates(self):
        zoom = parse_agent_action(
            _zoom([20, 20, 80, 80]),
            (200, 100),
            0,
            2,
        )
        self.assertEqual(zoom["action"], "zoom_in")
        self.assertEqual(zoom["bbox_px"], [20, 20, 80, 80])

        final = parse_agent_action(_final(), (200, 100), 2, 2)
        self.assertEqual(final["detection"]["decision"], "edited")
        self.assertEqual(
            final["localization"]["regions"][0]["bbox_px"],
            [50.0, 20.0, 75.0, 40.0],
        )

    def test_rejects_out_of_range_pixel_box(self):
        with self.assertRaisesRegex(SchemaError, "200x100"):
            parse_agent_action(
                _zoom([180, 20, 220, 80]),
                (200, 100),
                0,
                2,
            )

    def test_rejects_zoom_after_call_limit(self):
        with self.assertRaisesRegex(SchemaError, "limit"):
            parse_agent_action(_zoom([0, 0, 50, 50]), (100, 100), 2, 2)

    def test_executes_only_first_object_from_imagined_trajectory(self):
        response = "\n".join([
            _zoom([0, 0, 50, 50]),
            _zoom([50, 50, 100, 100]),
            _final(),
        ])
        action = parse_agent_action(response, (100, 100), 0, 2)
        self.assertEqual(action["action"], "zoom_in")
        self.assertEqual(action["bbox_px"], [0, 0, 50, 50])

    def test_not_edited_cannot_return_regions(self):
        value = json.loads(_final())
        value["decision"] = "not_edited"
        with self.assertRaisesRegex(SchemaError, "empty regions"):
            parse_agent_action(json.dumps(value), (100, 100), 0, 2)


class ZoomCropTest(unittest.TestCase):
    def test_crop_uses_original_coordinates_and_enlarges_losslessly(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = root / "original.png"
            output = root / "crop.png"
            Image.new("RGB", (100, 50), (10, 20, 30)).save(original)

            result = create_zoom_crop(
                original,
                [10, 10, 60, 40],
                output,
                long_side=200,
            )

            self.assertEqual(result["bbox_px"], [10, 10, 60, 40])
            self.assertEqual(result["crop_input_size"], [50, 30])
            self.assertEqual(result["crop_output_size"], [200, 120])
            with Image.open(output) as crop:
                self.assertEqual(crop.size, (200, 120))

    def test_crop_rejects_fractional_pixel_coordinates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = root / "original.png"
            Image.new("RGB", (101, 51), "white").save(original)
            with self.assertRaisesRegex(SchemaError, "integer"):
                create_zoom_crop(
                    original,
                    [1.5, 1, 99, 49],
                    root / "crop.png",
                    long_side=100,
                )


class ZoomAgentLoopTest(unittest.TestCase):
    def test_two_zoom_calls_then_final(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = root / "original.png"
            Image.new("RGB", (120, 80), (40, 50, 60)).save(original)
            client = _FakeClient([
                _zoom([0, 0, 60, 40]),
                _zoom([60, 16, 108, 64]),
                _final(),
            ])

            result = run_agent_episode(
                client,
                original,
                "data:image/png;base64,ORIGINAL",
                "sample",
                1,
                root / "crops",
                {
                    "maxRetriesPerReplicate": 0,
                    "baseBackoffSeconds": [0],
                },
                max_zoom_calls=2,
                zoom_long_side=160,
            )

            self.assertEqual(result["status"], "ok")
            self.assertEqual(len(result["tool_calls"]), 2)
            self.assertEqual(len(result["turns"]), 3)
            self.assertEqual(result["latency_ms"], 21)
            self.assertTrue(Path(result["tool_calls"][0]["crop_path"]).is_file())
            self.assertEqual(len(client.histories[0][1]), 1)
            self.assertEqual(len(client.histories[1][1]), 3)
            self.assertEqual(len(client.histories[2][1]), 5)
            self.assertIn(
                "original bbox_px",
                client.histories[1][1][-1]["content"][0]["text"],
            )

    def test_five_zoom_calls_then_final(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = root / "original.png"
            Image.new("RGB", (100, 100), "white").save(original)
            client = _FakeClient([
                _zoom([0, 0, 50, 50]),
                _zoom([10, 10, 60, 60]),
                _zoom([20, 20, 70, 70]),
                _zoom([30, 30, 80, 80]),
                _zoom([40, 40, 90, 90]),
                _final(),
            ])

            result = run_agent_episode(
                client,
                original,
                "data:image/png;base64,ORIGINAL",
                "sample",
                1,
                root / "crops",
                {
                    "maxRetriesPerReplicate": 0,
                    "baseBackoffSeconds": [0],
                },
                max_zoom_calls=5,
                zoom_long_side=100,
            )

            self.assertEqual(result["status"], "ok")
            self.assertEqual(len(result["tool_calls"]), 5)
            self.assertEqual(len(result["turns"]), 6)

    def test_third_zoom_is_rejected_and_repaired_to_final(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = root / "original.png"
            Image.new("RGB", (80, 80), "white").save(original)
            client = _FakeClient([
                _zoom([0, 0, 40, 40]),
                _zoom([40, 40, 80, 80]),
                _zoom([10, 10, 70, 70]),
                _final(),
            ])

            result = run_agent_episode(
                client,
                original,
                "data:image/png;base64,ORIGINAL",
                "sample",
                1,
                root / "crops",
                {
                    "maxRetriesPerReplicate": 1,
                    "baseBackoffSeconds": [0],
                },
                max_zoom_calls=2,
                zoom_long_side=100,
            )

            self.assertEqual(result["status"], "ok")
            self.assertEqual(len(result["tool_calls"]), 2)
            self.assertEqual(len(result["turns"][-1]["attempts"]), 2)
            repair = client.histories[-1][1][-1]["content"]
            self.assertIn("must return action=final", repair)

    def test_single_image_turns_rebuilds_context_with_latest_crop(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = root / "original.png"
            Image.new("RGB", (120, 80), (40, 50, 60)).save(original)
            client = _FakeClient([
                _zoom([0, 0, 60, 40]),
                _final(),
            ])

            result = run_agent_episode(
                client,
                original,
                "data:image/png;base64,ORIGINAL",
                "sample",
                1,
                root / "crops",
                {
                    "maxRetriesPerReplicate": 0,
                    "baseBackoffSeconds": [0],
                },
                max_zoom_calls=2,
                zoom_long_side=160,
                single_image_turns=True,
            )

            self.assertEqual(result["status"], "ok")
            self.assertEqual(len(client.histories[1][1]), 1)
            content = client.histories[1][1][0]["content"]
            image_parts = [
                part for part in content if part.get("type") == "image_url"
            ]
            self.assertEqual(len(image_parts), 1)
            self.assertIn("Previously executed action", content[0]["text"])
            self.assertIn("only the newest crop", content[0]["text"])


class ZoomAgentMetricsTest(unittest.TestCase):
    def test_resume_handles_unicode_line_separator_inside_json_string(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.raw.jsonl"
            row = {
                "id": "a",
                "protocol_key": "agent_zoom",
                "replicate_index": 1,
                "protocol_version": AGENT_PROTOCOL_VERSION,
                "status": "ok",
                "parsed": {
                    "detection": {
                        "reasoning": "first\u2028second",
                        "decision": "not_edited",
                    },
                },
            }
            path.write_text(
                json.dumps(row, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )

            self.assertEqual(
                completed_raw_keys(path, AGENT_PROTOCOL_VERSION),
                {("a", "agent_zoom", 1)},
            )
            parsed = successful_raw(path, AGENT_PROTOCOL_VERSION)
            self.assertEqual(
                parsed[("a", "agent_zoom", 1)]["detection"]["reasoning"],
                "first\u2028second",
            )

    def test_latest_episode_rows_drive_tool_use_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw_path = root / "run.raw.jsonl"
            rows = [
                {
                    "run_id": "run",
                    "id": "a",
                    "replicate_index": 1,
                    "protocol_version": AGENT_PROTOCOL_VERSION,
                    "status": "error",
                    "turns": [],
                    "tool_calls": [],
                },
                {
                    "run_id": "run",
                    "id": "a",
                    "replicate_index": 1,
                    "protocol_version": AGENT_PROTOCOL_VERSION,
                    "status": "ok",
                    "latency_ms": 10,
                    "turns": [{"attempts": [{"status": "ok"}]}],
                    "tool_calls": [{"bbox_px": [0, 0, 10, 10]}],
                    "parsed": {"detection": {"decision": "edited"}},
                },
                {
                    "run_id": "run",
                    "id": "a",
                    "replicate_index": 2,
                    "protocol_version": AGENT_PROTOCOL_VERSION,
                    "status": "ok",
                    "latency_ms": 20,
                    "turns": [{"attempts": [{"status": "ok"}]}],
                    "tool_calls": [],
                    "parsed": {"detection": {"decision": "not_edited"}},
                },
            ]
            raw_path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )

            summary = summarize_agent_run(
                raw_path,
                root / "run.agent_metrics.json",
                expected_images=1,
                run_id="run",
                model_slug="model",
                max_zoom_calls=5,
            )

            self.assertEqual(summary["recorded_latest_episodes"], 2)
            self.assertEqual(summary["successful_episodes"], 2)
            self.assertEqual(summary["zoom_calls_total"], 1)
            self.assertEqual(summary["episodes_with_any_zoom_rate"], 0.5)
            self.assertEqual(
                summary["zoom_call_count_histogram"],
                {"0": 1, "1": 1, "2": 0, "3": 0, "4": 0, "5": 0},
            )
            self.assertEqual(summary["mean_latency_ms_per_successful_episode"], 15)
            self.assertTrue((root / "run.agent_metrics.csv").is_file())


if __name__ == "__main__":
    unittest.main()
