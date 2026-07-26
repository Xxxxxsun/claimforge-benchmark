import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from eval.mllm.metrics import evaluate_review_export


PIL_AVAILABLE = importlib.util.find_spec("PIL") is not None


@unittest.skipUnless(PIL_AVAILABLE, "Pillow is required for exact-diff tests")
class MLLMExactDiffPixelMetricsTest(unittest.TestCase):
    def _evaluate(self, predicted_box: list[int]) -> tuple[dict, list[dict]]:
        from PIL import Image

        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)

        source = Image.new("RGB", (8, 8), (0, 0, 0))
        forged = source.copy()
        for y in range(2, 4):
            for x in range(2, 4):
                forged.putpixel((x, y), (255, 255, 255))
        source.save(root / "source.png")
        forged.save(root / "forged.png")

        review = {
            "records": [
                {
                    "task_id": "restaurant_000_slot_001",
                    "status": "good",
                    "source_image": "source.png",
                    "spliced_image": "forged.png",
                    "edit_region_xyxy": [2, 2, 4, 4],
                }
            ]
        }
        review_path = root / "review.json"
        review_path.write_text(json.dumps(review), encoding="utf-8")

        rows = [
            {
                "id": "restaurant_000_slot_001__forged",
                "protocol_key": "localization",
                "protocol_version": "test_protocol",
                "status": "ok",
                "valid_for_metrics": True,
                "decision": "localized_edit",
                "score": 0.8,
                "p_ai_edited": 80,
                "image_size": [8, 8],
                "regions_px": [predicted_box],
                "mask_path": "predicted-forged.png",
                "result": {},
            },
            {
                "id": "restaurant_000_slot_001__real",
                "protocol_key": "localization",
                "protocol_version": "test_protocol",
                "status": "ok",
                "valid_for_metrics": True,
                "decision": "no_localized_edit",
                "score": 0.1,
                "p_ai_edited": 10,
                "image_size": [8, 8],
                "regions_px": [],
                "mask_path": "predicted-real.png",
                "result": {},
            },
        ]
        results_path = root / "results.jsonl"
        results_path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows),
            encoding="utf-8",
        )
        output_dir = root / "metrics"
        summary = evaluate_review_export(
            results_path,
            review_path,
            output_dir,
            protocol_version="test_protocol",
            repo_root=root,
        )
        per_image = [
            json.loads(line)
            for line in (output_dir / "localization_per_image.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        return summary, per_image

    def test_exact_bbox_is_perfect_primary_pixel_result(self):
        summary, rows = self._evaluate([2, 2, 4, 4])
        localization = summary["localization"]
        self.assertEqual(
            localization["primary_t2_metric"],
            "forged_macro_pixel_iou_exact_diff",
        )
        self.assertEqual(localization["primary_t2_value"], 1.0)
        self.assertEqual(localization["forged_pixel_macro_f1"], 1.0)
        self.assertEqual(localization["forged_pixel_micro_iou"], 1.0)
        self.assertEqual(localization["auxiliary_box_hit_successes"], 1)
        self.assertEqual(
            localization["real_predicted_positive_fraction_micro"],
            0.0,
        )

        forged = next(row for row in rows if row["gt_label"] == "edited")
        self.assertEqual(forged["gt_mask_kind"], "exact_diff")
        self.assertEqual(forged["gt_positive_pixels"], 4)
        self.assertEqual(forged["pixel_metrics"]["tp"], 4)
        self.assertEqual(forged["pixel_iou"], 1.0)

    def test_larger_bbox_is_penalized_by_exact_pixel_mask(self):
        summary, rows = self._evaluate([1, 1, 5, 5])
        localization = summary["localization"]
        self.assertEqual(localization["primary_t2_value"], 0.25)
        self.assertAlmostEqual(localization["forged_pixel_macro_f1"], 0.4)
        self.assertEqual(localization["auxiliary_box_hit_successes"], 0)

        forged = next(row for row in rows if row["gt_label"] == "edited")
        self.assertEqual(forged["pixel_metrics"]["tp"], 4)
        self.assertEqual(forged["pixel_metrics"]["fp"], 12)
        self.assertEqual(forged["pixel_metrics"]["fn"], 0)
        self.assertEqual(forged["pixel_precision"], 0.25)
        self.assertEqual(forged["pixel_recall"], 1.0)
        self.assertEqual(forged["union_mask_edit_box_iou"], 0.25)


if __name__ == "__main__":
    unittest.main()
