import tempfile
import unittest
from pathlib import Path

import numpy as np

from eval.segmentation.run_fal_sam3 import (
    PilotItem,
    _decode_coco_counts,
    decode_rle,
    domain_from_task_id,
    hybrid_mask,
    mask_candidates,
    queue_app_id,
    request_spec,
    semantic_quality,
    select_pilot_items,
)


def encode_compressed_coco_counts(counts):
    encoded = []
    for index, count in enumerate(counts):
        value = count - counts[index - 2] if index > 2 else count
        more = True
        while more:
            code = value & 0x1F
            value >>= 5
            more = value != (-1 if code & 0x10 else 0)
            if more:
                code |= 0x20
            encoded.append(chr(code + 48))
    return "".join(encoded)


class FalRLEDecoderTest(unittest.TestCase):
    def test_queue_status_uses_app_root_not_endpoint_subpath(self):
        self.assertEqual(
            queue_app_id("fal-ai/sam-3/image-rle"),
            "fal-ai/sam-3",
        )

    def test_decodes_uncompressed_coco_in_column_major_order(self):
        mask = _decode_coco_counts([1, 2, 3], (2, 3))
        np.testing.assert_array_equal(
            mask,
            np.array([[False, True, False], [True, False, False]]),
        )

    def test_decodes_one_based_start_length_text(self):
        mask, encoding, encoded_shape = decode_rle("2 2 6 1", (2, 3))
        self.assertEqual(encoding, "start_length_text")
        self.assertEqual(encoded_shape, (2, 3))
        np.testing.assert_array_equal(
            mask,
            np.array([[False, True, True], [False, False, True]]),
        )

    def test_decodes_compressed_coco_object(self):
        counts = [1, 2, 1, 3, 1]
        encoded = encode_compressed_coco_counts(counts)
        mask, encoding, encoded_shape = decode_rle(
            {"size": [2, 4], "counts": encoded}, (2, 4)
        )
        self.assertEqual(encoding, "coco_compressed")
        self.assertEqual(encoded_shape, (2, 4))
        np.testing.assert_array_equal(mask, _decode_coco_counts(counts, (2, 4)))

    def test_text_only_request_omits_box_prompt(self):
        item = PilotItem(
            task_id="cat_restaurant_001",
            domain="restaurant",
            source_path=Path("source.png"),
            source_relative="source.png",
            generated_path=Path("generated.png"),
            generated_relative="generated.png",
            context_box=(0, 0, 4, 4),
            edit_box=(1, 1, 3, 3),
            crop_size=(4, 4),
            input_sha256="digest",
            input_bytes=1,
            edit_area_fraction=0.25,
            threshold_disagreement=0.5,
        )
        self.assertEqual(request_spec(item, "cat", 3, False)["box_prompts"], [])

    def test_candidate_ranking_prefers_residual_supported_mask(self):
        residual = np.zeros((4, 4), dtype=bool)
        residual.reshape(-1)[8:12] = True
        candidates = mask_candidates(
            {"rle": ["1 4", "9 4"], "scores": [0.95, 0.60]},
            (4, 4),
            (0, 0, 4, 4),
            residual,
        )
        self.assertEqual(candidates[0]["index"], 1)
        self.assertTrue(semantic_quality(candidates[0])["pass"])
        self.assertFalse(semantic_quality(candidates[1])["pass"])


class HybridMaskTest(unittest.TestCase):
    def test_adds_nearby_residual_but_not_distant_component(self):
        semantic = np.zeros((12, 12), dtype=bool)
        semantic[5:7, 5:7] = True
        residual = np.zeros_like(semantic)
        residual[7, 5:8] = True
        residual[0, 0:3] = True
        combined, support = hybrid_mask(semantic, residual, support_radius=2)
        self.assertTrue(combined[7, 6])
        self.assertTrue(support[7, 6])
        self.assertFalse(combined[0, 1])
        self.assertTrue(np.all(combined[semantic]))


class PilotSelectionTest(unittest.TestCase):
    def test_extracts_domain_from_multiword_candidate_task_id(self):
        self.assertEqual(
            domain_from_task_id("trash_can_restaurant_001_slot_001"),
            "restaurant",
        )

    def test_balances_domains_and_area_buckets(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            candidates = []
            for domain in ("lodging", "restaurant"):
                for index in range(9):
                    path = base / f"{domain}-{index}.png"
                    path.touch()
                    candidates.append(
                        PilotItem(
                            task_id=f"cat_{domain}_{index:03d}",
                            domain=domain,
                            source_path=path,
                            source_relative=path.name,
                            generated_path=path,
                            generated_relative=path.name,
                            context_box=(0, 0, 10, 10),
                            edit_box=(1, 1, 5, 5),
                            crop_size=(10, 10),
                            input_sha256=str(index),
                            input_bytes=0,
                            edit_area_fraction=(index + 1) / 100,
                            threshold_disagreement=index / 10,
                        )
                    )
            selected = select_pilot_items(candidates, 10)
        counts = {domain: sum(item.domain == domain for item in selected) for domain in ("lodging", "restaurant")}
        self.assertEqual(counts, {"lodging": 5, "restaurant": 5})
        for domain in counts:
            reasons = [item.selection_reason for item in selected if item.domain == domain]
            self.assertTrue(any(":small:" in reason for reason in reasons))
            self.assertTrue(any(":medium:" in reason for reason in reasons))
            self.assertTrue(any(":large:" in reason for reason in reasons))


if __name__ == "__main__":
    unittest.main()
