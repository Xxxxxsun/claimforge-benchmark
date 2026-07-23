import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from eval.opensource.run_trufor import (
    postprocess_outputs,
    preprocess_image,
    select_inputs,
)


class RunTruForTest(unittest.TestCase):
    def test_preprocess_matches_official_divide_by_256_contract(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "input.png"
            pixels = np.asarray(
                [[[0, 128, 255], [64, 32, 16]]],
                dtype=np.uint8,
            )
            Image.fromarray(pixels, mode="RGB").save(path)
            tensor, size = preprocess_image(path)
        self.assertEqual(size, (2, 1))
        self.assertEqual(tensor.shape, (3, 1, 2))
        self.assertEqual(tensor.dtype, np.float32)
        self.assertEqual(float(tensor[0, 0, 0]), 0.0)
        self.assertEqual(float(tensor[1, 0, 0]), 0.5)
        self.assertEqual(float(tensor[2, 0, 0]), 255 / 256)

    def test_postprocess_uses_forged_channel_and_separate_reliability(self):
        pred = torch.tensor(
            [[[[0.0, 2.0]], [[2.0, 0.0]]]],
            dtype=torch.float32,
        )
        confidence = torch.zeros((1, 1, 1, 2), dtype=torch.float32)
        detection = torch.tensor([[2.0]], dtype=torch.float32)
        score, logit, score_map, reliability = postprocess_outputs(
            pred,
            confidence,
            detection,
        )
        self.assertAlmostEqual(score, float(torch.sigmoid(torch.tensor(2.0))))
        self.assertEqual(logit, 2.0)
        self.assertGreater(float(score_map[0, 0]), 0.8)
        self.assertLess(float(score_map[0, 1]), 0.2)
        np.testing.assert_array_equal(
            reliability,
            np.full((1, 2), 0.5, dtype=np.float32),
        )

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
        self.assertEqual(
            [(row["pair_rank"], row["kind"]) for row in selected],
            [(0, "real"), (0, "forged"), (1, "real"), (1, "forged")],
        )


if __name__ == "__main__":
    unittest.main()
