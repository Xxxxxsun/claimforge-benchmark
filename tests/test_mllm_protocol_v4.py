import json
import unittest
from pathlib import Path
from unittest import mock

from eval.mllm.inputs import ImageItem
from eval.mllm.masks import boxes_to_1000, boxes_to_pixels
from eval.mllm.metrics import _iou
from eval.mllm.prompts import (
    PROTOCOL_SUITE_VERSION,
    PROTOCOL_VERSION,
    PROTOCOL_VERSIONS,
)
from eval.mllm.run_mllm import _one_replicate, _protocol_prompt
from eval.mllm.schema import SchemaError, aggregate, parse


class PixelCoordinateSchemaTest(unittest.TestCase):
    def test_protocol_uses_original_pixel_coordinates(self):
        self.assertEqual(PROTOCOL_VERSION, "mllm_protocol_v4_reasoning_pixel_coordinates")
        self.assertEqual(PROTOCOL_SUITE_VERSION, "mllm_protocol_suite_20260724")
        self.assertEqual(PROTOCOL_VERSIONS["detection"], "mllm_protocol_v3_reasoning_image_coordinates")
        self.assertEqual(PROTOCOL_VERSIONS["localization"], "mllm_protocol_v4_reasoning_pixel_coordinates")
        prompt = _protocol_prompt("localization", (1024, 683))
        self.assertIn("bbox_px", prompt)
        self.assertIn("1024 pixels wide by 683 pixels high", prompt)
        self.assertIn("Do not normalize coordinates", prompt)

    def test_parses_pixel_box_and_rejects_old_or_out_of_range_box(self):
        valid = json.dumps({
            "reasoning": "The cat has inconsistent lighting.",
            "decision": "localized_edit",
            "p_ai_edited": 80,
            "regions": [{
                "bbox_px": [456, 313, 579, 379],
                "confidence": 75,
                "evidence": "Suspicious cat",
            }],
        })
        parsed = parse("localization", valid, (1024, 683))
        self.assertEqual(parsed["regions"][0]["bbox_px"], [456.0, 313.0, 579.0, 379.0])

        old_schema = valid.replace("bbox_px", "bbox_1000")
        with self.assertRaises(SchemaError):
            parse("localization", old_schema, (1024, 683))

        out_of_range = valid.replace("579, 379", "1100, 379")
        with self.assertRaises(SchemaError):
            parse("localization", out_of_range, (1024, 683))

    def test_aggregation_and_derived_normalized_coordinates(self):
        replies = [
            {
                "reasoning": f"replicate {index}",
                "decision": "localized_edit",
                "p_ai_edited": 70 + index,
                "regions": [{
                    "bbox_px": box,
                    "confidence": 80,
                    "evidence": "cat",
                }],
            }
            for index, box in enumerate((
                [456.0, 313.0, 579.0, 379.0],
                [458.0, 312.0, 580.0, 380.0],
                [454.0, 314.0, 578.0, 378.0],
            ))
        ]
        result = aggregate("localization", replies)
        self.assertEqual(result["regions"][0]["bbox_px"], [456.0, 313.0, 579.0, 379.0])
        self.assertEqual(boxes_to_pixels(result["regions"], 1024, 683), [[456, 313, 579, 379]])
        normalized = boxes_to_1000(result["regions"], 1024, 683)[0]["bbox_1000"]
        self.assertAlmostEqual(normalized[0], 445.3125)
        self.assertAlmostEqual(normalized[1], 458.2723279648609)

    def test_schema_repair_keeps_image_dimensions(self):
        class Client:
            def __init__(self):
                self.prompts = []

            def image_url(self, path, external_url):
                return "data:image/png;base64,AA=="

            def call(self, system_prompt, user_prompt, image):
                self.prompts.append(user_prompt)
                if len(self.prompts) == 1:
                    return json.dumps({
                        "reasoning": "invalid old coordinate field",
                        "decision": "localized_edit",
                        "p_ai_edited": 80,
                        "regions": [{"bbox_1000": [10, 10, 20, 20], "confidence": 80}],
                    }), 1
                return json.dumps({
                    "reasoning": "valid repaired pixel box",
                    "decision": "localized_edit",
                    "p_ai_edited": 80,
                    "regions": [{"bbox_px": [10, 10, 20, 20], "confidence": 80}],
                }), 1

        client = Client()
        item = ImageItem("image", Path("unused.png"), None)
        with mock.patch("eval.mllm.run_mllm.time.sleep"):
            result = _one_replicate(
                client,
                item,
                "localization",
                1,
                {"maxRetriesPerReplicate": 1, "baseBackoffSeconds": [0]},
                False,
                (1024, 683),
            )
        self.assertEqual(result["status"], "ok")
        self.assertEqual(len(client.prompts), 2)
        self.assertIn("1024 pixels wide by 683 pixels high", client.prompts[1])
        self.assertIn("bbox_px", client.prompts[1])


class LocalizationIoUTest(unittest.TestCase):
    def test_iou_threshold_examples(self):
        gt = [0.0, 0.0, 100.0, 100.0]
        self.assertEqual(_iou(gt, gt), 1.0)
        self.assertEqual(_iou([100.0, 0.0, 200.0, 100.0], gt), 0.0)
        self.assertAlmostEqual(_iou([0.0, 0.0, 50.0, 100.0], gt), 0.5)
        self.assertAlmostEqual(_iou([0.0, 0.0, 25.0, 100.0], gt), 0.25)
        self.assertAlmostEqual(_iou([0.0, 0.0, 10.0, 100.0], gt), 0.1)


if __name__ == "__main__":
    unittest.main()
