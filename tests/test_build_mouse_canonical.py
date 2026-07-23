import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw

from eval.opensource.build_mouse_canonical import build_dataset
from eval.opensource.common import read_jsonl, sha256_file


class BuildMouseCanonicalTest(unittest.TestCase):
    def _fixture(self, root: Path, forged_size: tuple[int, int] = (12, 8)):
        (root / "raw").mkdir()
        source = Image.new("RGB", (12, 8), (80, 90, 100))
        source.save(root / "raw/source.jpg", quality=95)
        with Image.open(root / "raw/source.jpg") as opened:
            forged = opened.convert("RGB")
        if forged_size != forged.size:
            forged = forged.resize(forged_size)
        draw = ImageDraw.Draw(forged)
        draw.rectangle((3, 2, 5, 4), fill=(220, 20, 10))
        forged.save(root / "raw/forged.png")

        review = {
            "records": [
                {
                    "task_id": "restaurant_001_slot_001",
                    "status": "good",
                    "candidates": "mouse",
                    "source_image": "raw/source.jpg",
                    "spliced_image": "raw/forged.png",
                    "image_size": [12, 8],
                    "edit_region_xyxy": [3, 2, 6, 5],
                    "context_region_xyxy": [2, 1, 7, 6],
                }
            ]
        }
        (root / "review.json").write_text(json.dumps(review), encoding="utf-8")
        ordering = {
            "ordered_inputs": [{"task_id": "restaurant_001_slot_001"}]
        }
        (root / "order.json").write_text(json.dumps(ordering), encoding="utf-8")

    def test_builds_paired_identically_encoded_inputs_and_exact_mask(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._fixture(root)
            manifest = build_dataset(
                repo_root=root,
                review_path=root / "review.json",
                order_manifest_path=root / "order.json",
                output_dir=root / "out",
                expected_pairs=1,
            )

            self.assertEqual(manifest["pairs"], 1)
            self.assertEqual(manifest["images"], 2)
            rows = read_jsonl(root / "out/inputs.jsonl")
            self.assertEqual([row["kind"] for row in rows], ["real", "forged"])
            self.assertEqual([row["label"] for row in rows], [0, 1])
            self.assertEqual(rows[0]["gt_mask_kind"], "all_zero")
            self.assertIsNone(rows[0]["gt_mask_path"])
            self.assertEqual(rows[1]["gt_positive_pixels"], 9)

            for row in rows:
                image_path = root / row["canonical_path"]
                with Image.open(image_path) as image:
                    self.assertEqual(image.format, "JPEG")
                    self.assertEqual(image.mode, "RGB")
                    self.assertEqual(image.size, (12, 8))
                    self.assertFalse(image.getexif())
                self.assertEqual(sha256_file(image_path), row["canonical_sha256"])
                self.assertNotIn(row["kind"], image_path.name)

            mask_path = root / rows[1]["gt_mask_path"]
            with Image.open(mask_path) as mask:
                self.assertEqual(mask.mode, "L")
                self.assertEqual(mask.histogram()[255], 9)

    def test_rejects_mismatched_pair_dimensions(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._fixture(root, forged_size=(10, 8))
            with self.assertRaisesRegex(ValueError, "pair size mismatch"):
                build_dataset(
                    repo_root=root,
                    review_path=root / "review.json",
                    order_manifest_path=root / "order.json",
                    output_dir=root / "out",
                    expected_pairs=1,
                )

    def test_v1_rejects_noncanonical_quality(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._fixture(root)
            with self.assertRaisesRegex(ValueError, "quality=95"):
                build_dataset(
                    repo_root=root,
                    review_path=root / "review.json",
                    order_manifest_path=root / "order.json",
                    output_dir=root / "out",
                    quality=90,
                    expected_pairs=1,
                )


if __name__ == "__main__":
    unittest.main()
