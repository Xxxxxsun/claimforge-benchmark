import unittest

import numpy as np

from eval.opensource.run_maskclip import restore_score_map, select_inputs


class RunMaskCLIPTest(unittest.TestCase):
    def test_select_inputs_keeps_complete_fixed_pairs(self):
        rows = [
            {
                "rank": pair * 2 + offset,
                "pair_rank": pair,
                "kind": kind,
            }
            for pair in range(3)
            for offset, kind in enumerate(("real", "forged"))
        ]
        selected = select_inputs(rows, pair_limit=2)
        self.assertEqual(len(selected), 4)
        self.assertEqual(
            [(row["pair_rank"], row["kind"]) for row in selected],
            [(0, "real"), (0, "forged"), (1, "real"), (1, "forged")],
        )

    def test_restore_score_map_preserves_range_and_native_shape(self):
        score_map = np.linspace(0, 1, 512 * 512, dtype=np.float32).reshape(
            512,
            512,
        )
        restored = restore_score_map(score_map, width=13, height=7)
        self.assertEqual(restored.shape, (7, 13))
        self.assertEqual(restored.dtype, np.float32)
        self.assertGreaterEqual(float(restored.min()), 0.0)
        self.assertLessEqual(float(restored.max()), 1.0)


if __name__ == "__main__":
    unittest.main()
