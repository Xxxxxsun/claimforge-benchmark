import tempfile
import unittest
from pathlib import Path

import numpy as np

from eval.segmentation.run_fal_sam3 import (
    HybridConfig,
    PilotItem,
    _decode_coco_counts,
    decode_rle,
    domain_from_task_id,
    hybrid_mask,
    inner_alpha_mask,
    mask_candidates,
    queue_app_id,
    request_spec,
    semantic_hysteresis_mask,
    semantic_quality,
    select_pilot_items,
)
from eval.segmentation.materialize_hysteresis_sam3 import (
    HysteresisSam3Config,
    hysteresis_sam3_mask,
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
    def test_inner_alpha_never_leaks_outside_semantic_mask(self):
        semantic = np.zeros((15, 15), dtype=bool)
        semantic[4:11, 4:11] = True
        alpha = np.asarray(inner_alpha_mask(semantic, feather=2.0))
        self.assertTrue(np.all(alpha[~semantic] == 0))
        self.assertGreater(int(alpha[semantic].max()), 0)
        self.assertLess(int(alpha[4, 7]), 255)

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

    def test_semantic_hysteresis_recovers_distant_connected_shadow(self):
        semantic = np.zeros((60, 80), dtype=bool)
        semantic[25:40, 40:50] = True
        difference = np.zeros(semantic.shape, dtype=np.float32)
        difference[semantic] = 100
        difference[34:38, 10:40] = 55
        config = HybridConfig(
            mode="semantic_hysteresis",
            support_radius=2,
            edge_radius=2,
            hysteresis_reach_ratio=0.5,
            hysteresis_auto_expand_ratio=0.75,
            max_hybrid_growth=10.0,
            max_added_fraction=0.5,
        )
        combined, edge, distance, stats = semantic_hysteresis_mask(
            semantic,
            difference,
            (20, 15, 60, 50),
            config,
            shadow_darkening=difference,
        )
        self.assertTrue(combined[36, 12])
        self.assertTrue(distance[36, 12])
        self.assertTrue(np.all(combined[semantic]))
        self.assertFalse(stats["guard_fallback"])
        self.assertGreater(edge.sum(), 0)

    def test_semantic_hysteresis_guard_rejects_runaway_growth(self):
        semantic = np.zeros((40, 40), dtype=bool)
        semantic[19:22, 19:22] = True
        difference = np.full(semantic.shape, 100, dtype=np.float32)
        config = HybridConfig(
            mode="semantic_hysteresis",
            support_radius=1,
            edge_radius=1,
            hysteresis_reach_ratio=1.0,
            hysteresis_auto_expand_ratio=1.0,
            max_hybrid_growth=2.0,
            max_added_fraction=0.05,
        )
        combined, _, distance, stats = semantic_hysteresis_mask(
            semantic,
            difference,
            (5, 5, 35, 35),
            config,
            shadow_darkening=difference,
        )
        self.assertTrue(stats["guard_fallback"])
        self.assertEqual(int(distance.sum()), 0)
        self.assertLess(int(combined.sum()), 100)
        self.assertTrue(np.all(combined[semantic]))

    def test_semantic_hysteresis_rejects_bright_far_field_reconstruction(self):
        semantic = np.zeros((60, 80), dtype=bool)
        semantic[25:40, 40:50] = True
        difference = np.zeros(semantic.shape, dtype=np.float32)
        difference[semantic] = 100
        difference[34:38, 10:40] = 55
        shadow_darkening = difference.copy()
        shadow_darkening[34:38, 10:40] = -20
        config = HybridConfig(
            mode="semantic_hysteresis",
            support_radius=2,
            edge_radius=2,
            hysteresis_reach_ratio=0.5,
            hysteresis_auto_expand_ratio=0.75,
            max_hybrid_growth=10.0,
            max_added_fraction=0.5,
        )
        combined, _, distance, stats = semantic_hysteresis_mask(
            semantic,
            difference,
            (20, 15, 60, 50),
            config,
            shadow_darkening=shadow_darkening,
        )
        self.assertFalse(combined[36, 12])
        self.assertFalse(distance[36, 12])
        self.assertTrue(stats["propagation"]["shadow_direction_filter_applied"])

    def test_semantic_hysteresis_rejects_dark_far_field_above_object(self):
        semantic = np.zeros((60, 80), dtype=bool)
        semantic[25:40, 40:50] = True
        difference = np.zeros(semantic.shape, dtype=np.float32)
        difference[semantic] = 100
        difference[15:19, 10:43] = 55
        combined, _, distance, stats = semantic_hysteresis_mask(
            semantic,
            difference,
            (20, 10, 60, 50),
            HybridConfig(
                mode="semantic_hysteresis",
                support_radius=2,
                edge_radius=2,
                hysteresis_reach_ratio=0.5,
                hysteresis_auto_expand_ratio=0.75,
                max_hybrid_growth=10.0,
                max_added_fraction=0.5,
            ),
            shadow_darkening=difference,
        )
        self.assertFalse(combined[16, 12])
        self.assertFalse(distance[16, 12])
        self.assertGreater(stats["propagation"]["shadow_min_y"], 25)

    def test_semantic_shadow_uses_edge_ring_only_as_internal_bridge(self):
        semantic = np.zeros((60, 80), dtype=bool)
        semantic[20:35, 40:50] = True
        difference = np.zeros(semantic.shape, dtype=np.float32)
        difference[semantic] = 100
        difference[18:38, 38:52] = 55
        difference[29:33, 10:40] = 55
        shadow_darkening = np.full(semantic.shape, -20, dtype=np.float32)
        shadow_darkening[29:33, 10:40] = 55
        config = HybridConfig(
            mode="semantic_shadow",
            support_radius=2,
            edge_radius=2,
            shadow_output_grow=1,
            hysteresis_reach_ratio=0.5,
            hysteresis_auto_expand_ratio=0.75,
            max_hybrid_growth=10.0,
            max_added_fraction=0.5,
        )
        combined, edge, distance, stats = semantic_hysteresis_mask(
            semantic,
            difference,
            (20, 10, 60, 45),
            config,
            shadow_darkening=shadow_darkening,
        )
        self.assertGreater(int(edge.sum()), 0)
        self.assertFalse(stats["edge_support_emitted"])
        self.assertFalse(combined[19, 45])
        self.assertTrue(combined[31, 12])
        self.assertTrue(distance[31, 12])
        self.assertTrue(np.all(combined[semantic]))


class HysteresisSam3MaskTest(unittest.TestCase):
    def setUp(self):
        self.config = HysteresisSam3Config(
            low_threshold=10,
            high_threshold=30,
            far_threshold=25,
            distance_power=1,
            reach_scale=0.5,
            min_reach_pixels=4,
            max_reach_pixels=8,
            close_iterations=0,
            grow_iterations=0,
            min_component_pixels=1,
            semantic_feather=0,
            residual_feather=0,
            core_erosion=0,
            unbounded_growth=False,
            component_size_cap=True,
        )

    def test_keeps_sam_subject_and_grows_attached_weak_residual(self):
        semantic = np.zeros((20, 20), dtype=bool)
        semantic[8:12, 8:12] = True
        difference = np.zeros((20, 20), dtype=np.float32)
        difference[9:11, 12:16] = 25

        combined, support, alpha, stats = hysteresis_sam3_mask(
            semantic,
            difference,
            self.config,
        )

        self.assertTrue(np.all(combined[semantic]))
        self.assertTrue(support[9, 14])
        self.assertEqual(alpha[9, 14], 255)
        self.assertGreater(stats["growth_over_semantic"], 0)

    def test_rejects_distant_residual_outside_adaptive_reach(self):
        semantic = np.zeros((24, 24), dtype=bool)
        semantic[10:14, 10:14] = True
        difference = np.zeros((24, 24), dtype=np.float32)
        difference[0:2, 0:2] = 255

        combined, support, _, stats = hysteresis_sam3_mask(
            semantic,
            difference,
            self.config,
        )

        self.assertFalse(support[0, 0])
        self.assertFalse(combined[0, 0])
        self.assertEqual(stats["residual_support_pixels"], 0)

    def test_distance_ramp_rejects_weak_far_pixel_but_keeps_near_pixel(self):
        semantic = np.zeros((20, 20), dtype=bool)
        semantic[8:12, 8:12] = True
        difference = np.zeros((20, 20), dtype=np.float32)
        difference[9, 12] = 15
        difference[9, 16] = 15

        combined, support, _, _ = hysteresis_sam3_mask(
            semantic,
            difference,
            self.config,
        )

        self.assertTrue(support[9, 12])
        self.assertFalse(support[9, 16])
        self.assertTrue(combined[9, 12])

    def test_unbounded_mode_keeps_connected_residual_beyond_reach(self):
        semantic = np.zeros((20, 30), dtype=bool)
        semantic[8:12, 8:12] = True
        difference = np.zeros((20, 30), dtype=np.float32)
        difference[9:11, 12:27] = 50
        open_config = HysteresisSam3Config(
            **{
                **self.config.__dict__,
                "unbounded_growth": True,
                "component_size_cap": False,
            }
        )

        combined, support, _, stats = hysteresis_sam3_mask(
            semantic,
            difference,
            open_config,
        )

        self.assertTrue(support[9, 25])
        self.assertTrue(combined[9, 25])
        self.assertTrue(stats["unbounded_growth"])
        self.assertIsNone(stats["component_size_cap_pixels"])

    def test_auto_expands_connected_dark_shadow_that_touches_reach(self):
        semantic = np.zeros((24, 32), dtype=bool)
        semantic[8:12, 8:12] = True
        difference = np.zeros((24, 32), dtype=np.float32)
        difference[10:12, 12:21] = 35
        darkening = np.zeros_like(difference)
        darkening[10:12, 12:21] = 20
        config = HysteresisSam3Config(
            **{
                **self.config.__dict__,
                "reach_scale": 0.5,
                "min_reach_pixels": 4,
                "max_reach_pixels": 4,
                "auto_expand_scale": 2.0,
                "auto_expand_max_reach_pixels": 10,
                "auto_expand_max_growth_over_semantic": 2.0,
                "far_direction_start_ratio": 0.25,
                "far_shadow_channel_min": 3,
            }
        )

        combined, support, _, stats = hysteresis_sam3_mask(
            semantic,
            difference,
            config,
            shadow_channel_darkening=darkening,
        )

        self.assertTrue(support[10, 19])
        self.assertTrue(combined[10, 19])
        self.assertTrue(stats["auto_expand_attempted"])
        self.assertTrue(stats["auto_expand_applied"])
        self.assertEqual(stats["initial_reach_pixels"], 4)
        self.assertEqual(stats["expanded_reach_pixels"], 8)

    def test_far_direction_filter_rejects_bright_background_bridge(self):
        semantic = np.zeros((24, 32), dtype=bool)
        semantic[8:12, 8:12] = True
        difference = np.zeros((24, 32), dtype=np.float32)
        difference[10:12, 12:21] = 35
        brightening = np.zeros_like(difference)
        brightening[10:12, 12:21] = -20
        config = HysteresisSam3Config(
            **{
                **self.config.__dict__,
                "reach_scale": 2.0,
                "min_reach_pixels": 8,
                "max_reach_pixels": 8,
                "auto_expand_scale": 0,
                "far_direction_start_ratio": 0.25,
                "far_shadow_channel_min": 3,
            }
        )

        _, support, _, stats = hysteresis_sam3_mask(
            semantic,
            difference,
            config,
            shadow_channel_darkening=brightening,
        )

        self.assertFalse(support[10, 18])
        self.assertTrue(stats["shadow_direction_filter_applied"])

    def test_postconnect_filter_removes_near_brightening_and_feather(self):
        semantic = np.zeros((20, 20), dtype=bool)
        semantic[8:12, 8:12] = True
        difference = np.zeros((20, 20), dtype=np.float32)
        difference[9:11, 12:16] = 35
        darkening = np.zeros_like(difference)
        darkening[9, 13] = -1
        config = HysteresisSam3Config(
            **{
                **self.config.__dict__,
                "semantic_feather": 1,
                "residual_feather": 1,
                "far_direction_start_ratio": 1,
            }
        )

        _, support, alpha, stats = hysteresis_sam3_mask(
            semantic,
            difference,
            config,
            shadow_channel_darkening=darkening,
        )

        self.assertFalse(support[9, 13])
        self.assertEqual(alpha[9, 13], 0)
        self.assertTrue(stats["postconnect_nonbright_filter_applied"])
        self.assertEqual(stats["postconnect_bright_pixels_removed"], 1)
        self.assertGreater(stats["residual_alpha_bright_pixels_zeroed"], 0)
        self.assertGreater(stats["final_alpha_bright_pixels_zeroed"], 0)


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
