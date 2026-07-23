import argparse
import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

from eval.opensource.common import sha256_file
from eval.opensource.run_catnet import (
    CHECKPOINT_BYTES,
    CHECKPOINT_EPOCH,
    CHECKPOINT_SHA256,
    CHECKPOINT_STATE_KEYS,
    DEFAULT_CATNET_ROOT,
    DEFAULT_CHECKPOINT,
    DCT_BINS,
    MASK_THRESHOLD,
    _write_or_validate_run_manifest,
    build_run_manifest,
    dct_volume_from_coefficients,
    postprocess_logits,
    preprocess_jpeg,
    select_inputs,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
CANONICAL_MANIFEST = (
    REPO_ROOT / "outputs/opensource/mouse_canonical_v1/manifest.json"
)
CANONICAL_INPUTS = (
    REPO_ROOT / "outputs/opensource/mouse_canonical_v1/inputs.jsonl"
)
# A real canonical image whose width and height both require ceil-8 padding.
CANONICAL_REAL = (
    REPO_ROOT
    / "outputs/opensource/mouse_canonical_v1/images"
    / "cdf069c6dc6b3200817f940d.jpg"
)
CANONICAL_REAL_SHA256 = (
    "5981124fb5f4811577ad0d2d9fde8edcf4f61c2c9b85a144282a9ebc643fa1b0"
)
CANONICAL_REAL_QTABLE_SHA256 = (
    "1218b4eee2d82f6468dae979e0828d5129cd791e1c5cfde2f9c3cd6866a0465f"
)
CANONICAL_REAL_DCT_Y_SHA256 = (
    "ee0eece6d5afa9e7d593fdfe619a67ff604f36bd89cdf98753a3782f2258875b"
)
HAS_JPEGIO = importlib.util.find_spec("jpegio") is not None


def _int32_sha256(array: np.ndarray) -> str:
    canonical = np.ascontiguousarray(array, dtype=np.int32)
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


def _canonical_rows() -> list[dict]:
    return [
        json.loads(line)
        for line in CANONICAL_INPUTS.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


class RunCatNetTest(unittest.TestCase):
    def test_dct_volume_uses_absolute_21_bin_contract(self):
        coefficients = np.asarray(
            [[-21, -20, -19, -1, 0, 1, 19, 20, 21]],
            dtype=np.int32,
        )
        original = coefficients.copy()

        volume = dct_volume_from_coefficients(coefficients)

        self.assertEqual(volume.shape, (DCT_BINS, 1, 9))
        self.assertEqual(volume.dtype, np.float32)
        np.testing.assert_array_equal(coefficients, original)
        np.testing.assert_array_equal(
            np.argmax(volume, axis=0),
            np.asarray([[20, 20, 19, 1, 0, 1, 19, 20, 20]]),
        )
        np.testing.assert_array_equal(
            volume.sum(axis=0),
            np.ones((1, 9), dtype=np.float32),
        )
        with self.assertRaisesRegex(ValueError, "two-dimensional"):
            dct_volume_from_coefficients(np.zeros((1, 1, 1), dtype=np.int32))

    @unittest.skipUnless(
        HAS_JPEGIO and CANONICAL_REAL.is_file(),
        "canonical JPEG fixture and jpegio are required",
    )
    def test_preprocess_reads_original_jpeg_dct_and_pads_only_bottom_right(self):
        import jpegio

        self.assertEqual(sha256_file(CANONICAL_REAL), CANONICAL_REAL_SHA256)
        jpeg = jpegio.read(str(CANONICAL_REAL))
        qtable_index = int(jpeg.comp_info[0].quant_tbl_no)
        direct_qtable = np.asarray(
            jpeg.quant_tables[qtable_index],
            dtype=np.int32,
        )
        direct_coefficients = np.asarray(
            jpeg.coef_arrays[0],
            dtype=np.int32,
        )
        self.assertEqual(
            _int32_sha256(direct_qtable),
            CANONICAL_REAL_QTABLE_SHA256,
        )
        self.assertEqual(
            _int32_sha256(direct_coefficients),
            CANONICAL_REAL_DCT_Y_SHA256,
        )

        with mock.patch.object(
            jpegio,
            "read",
            wraps=jpegio.read,
        ) as read_jpeg:
            image, qtable, metadata = preprocess_jpeg(CANONICAL_REAL)

        read_jpeg.assert_called_once_with(str(CANONICAL_REAL))
        self.assertEqual(sha256_file(CANONICAL_REAL), CANONICAL_REAL_SHA256)
        self.assertEqual(metadata["native_size"], [813, 625])
        self.assertEqual(metadata["padded_size"], [816, 632])
        self.assertEqual(
            metadata["padding"],
            {"left": 0, "top": 0, "right": 3, "bottom": 7},
        )
        self.assertEqual(
            metadata["jpeg_sampling_factors"],
            [[1, 1], [1, 1], [1, 1]],
        )
        self.assertEqual(
            metadata["qtable_sha256"],
            CANONICAL_REAL_QTABLE_SHA256,
        )
        self.assertEqual(
            metadata["dct_y_sha256"],
            CANONICAL_REAL_DCT_Y_SHA256,
        )
        self.assertEqual(image.shape, (3 + DCT_BINS, 632, 816))
        self.assertEqual(image.dtype, np.float32)
        self.assertEqual(qtable.shape, (1, 8, 8))
        self.assertEqual(qtable.dtype, np.float32)
        np.testing.assert_array_equal(qtable[0], direct_qtable)

        with Image.open(CANONICAL_REAL) as opened:
            decoded_rgb = np.asarray(opened.convert("RGB"), dtype=np.uint8)
        expected_rgb = (
            decoded_rgb.astype(np.float32).transpose(2, 0, 1) - 127.5
        ) / 127.5
        np.testing.assert_array_equal(image[:3, :625, :813], expected_rgb)
        np.testing.assert_array_equal(
            image[:3, 625:, :],
            np.zeros((3, 7, 816), dtype=np.float32),
        )
        np.testing.assert_array_equal(
            image[:3, :, 813:],
            np.zeros((3, 632, 3), dtype=np.float32),
        )

        expected_bins = np.minimum(
            np.abs(direct_coefficients.astype(np.int64)),
            DCT_BINS - 1,
        )
        np.testing.assert_array_equal(
            np.argmax(image[3:], axis=0),
            expected_bins,
        )
        np.testing.assert_array_equal(
            image[3:].sum(axis=0),
            np.ones((632, 816), dtype=np.float32),
        )

    def test_postprocess_resizes_logits_before_softmax_then_crops(self):
        logits = torch.tensor(
            [
                [
                    [[8.0, -8.0], [-8.0, 8.0]],
                    [[-8.0, 8.0], [8.0, -8.0]],
                ]
            ],
            dtype=torch.float32,
        )

        raw_logits, native_map = postprocess_logits(
            logits,
            padded_width=8,
            padded_height=8,
            native_width=7,
            native_height=5,
        )

        expected = torch.softmax(
            F.interpolate(
                logits,
                size=(8, 8),
                mode="bilinear",
                align_corners=False,
            ),
            dim=1,
        )[0, 1, :5, :7].numpy()
        wrong_softmax_first = F.interpolate(
            torch.softmax(logits, dim=1),
            size=(8, 8),
            mode="bilinear",
            align_corners=False,
        )[0, 1, :5, :7].numpy()
        np.testing.assert_array_equal(raw_logits, logits[0].numpy())
        self.assertEqual(native_map.shape, (5, 7))
        self.assertEqual(native_map.dtype, np.float32)
        np.testing.assert_allclose(native_map, expected, rtol=0, atol=1e-7)
        self.assertGreater(
            float(np.max(np.abs(native_map - wrong_softmax_first))),
            0.3,
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
        with self.assertRaisesRegex(ValueError, "positive"):
            select_inputs(rows, pair_limit=0)
        with self.assertRaisesRegex(ValueError, "incomplete pairs"):
            select_inputs(rows[:-1], pair_limit=None)

    def test_existing_manifest_rejects_fingerprint_conflict(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "run_manifest.json"
            first = {"run_id": "catnet-test", "fingerprint": "a" * 64}
            _write_or_validate_run_manifest(path, first)
            _write_or_validate_run_manifest(
                path,
                {**first, "created_at": "ignored-for-resume"},
            )
            with self.assertRaisesRegex(ValueError, "incompatible"):
                _write_or_validate_run_manifest(
                    path,
                    {**first, "fingerprint": "b" * 64},
                )

    @unittest.skipUnless(
        CANONICAL_MANIFEST.is_file() and CANONICAL_INPUTS.is_file(),
        "canonical release fixture is required",
    )
    def test_manifest_declares_t2_only_license_and_safe_load_contract(self):
        release = json.loads(
            CANONICAL_MANIFEST.read_text(encoding="utf-8")
        )
        selected = _canonical_rows()[:2]
        args = argparse.Namespace(
            run_id="catnet_unit_test",
            condition="mouse_canonical_v1",
            seed=42,
            device="cpu",
            mask_threshold=MASK_THRESHOLD,
        )
        with tempfile.TemporaryDirectory() as temporary:
            manifest = build_run_manifest(
                args=args,
                repo_root=REPO_ROOT,
                dataset_manifest_path=CANONICAL_MANIFEST,
                release=release,
                inputs_path=CANONICAL_INPUTS,
                selected=selected,
                catnet_root=DEFAULT_CATNET_ROOT,
                checkpoint_path=DEFAULT_CHECKPOINT,
                artifact_dir=Path(temporary),
            )

        model = manifest["model"]
        inference = manifest["inference"]
        metrics = manifest["metrics"]
        self.assertFalse(model["supports_image_level_t1"])
        self.assertTrue(model["supports_pixel_level_t2"])
        self.assertEqual(
            inference["t1_policy"],
            "unsupported_no_derived_image_score",
        )
        self.assertEqual(metrics["task"], "T2_pixel_localization_only")
        for section in (inference, metrics):
            for forbidden in (
                "classification",
                "classification_threshold",
                "decision",
                "image_score",
                "score",
                "t1_threshold",
            ):
                self.assertNotIn(forbidden, section)

        self.assertEqual(
            model["license"],
            {
                "path": "LICENSE of HRNet",
                "sha256": (
                    "f1f33c3bec144f048d1cbff4dcae8d47"
                    "a28faf263930ce779c61a7f4913bf055"
                ),
                "scope": "hrnet_component_only",
                "project_wide_status": "no_project_wide_license_found",
                "classification": "source_available_research_release",
            },
        )
        checkpoint = model["checkpoint"]
        self.assertTrue(checkpoint["strict_load"])
        self.assertTrue(checkpoint["safe_weights_only_load"])
        self.assertIn("weights_only=True", checkpoint["safe_load"])
        self.assertIn("allowlisted", checkpoint["safe_load"])

    @unittest.skipUnless(
        DEFAULT_CHECKPOINT.is_file(),
        "official CAT-Net v2 checkpoint is not installed",
    )
    def test_official_checkpoint_safe_loads_with_expected_schema(self):
        self.assertEqual(DEFAULT_CHECKPOINT.stat().st_size, CHECKPOINT_BYTES)
        self.assertEqual(sha256_file(DEFAULT_CHECKPOINT), CHECKPOINT_SHA256)
        safe_globals = [
            np.core.multiarray.scalar,
            np.dtype,
            type(np.dtype(np.float64)),
        ]
        with torch.serialization.safe_globals(safe_globals):
            checkpoint = torch.load(
                DEFAULT_CHECKPOINT,
                map_location="cpu",
                weights_only=True,
            )

        self.assertIsInstance(checkpoint, dict)
        self.assertEqual(
            set(checkpoint),
            {"best_p_mIoU", "epoch", "optimizer", "state_dict"},
        )
        self.assertEqual(checkpoint["epoch"], CHECKPOINT_EPOCH)
        state = checkpoint["state_dict"]
        self.assertIsInstance(state, dict)
        self.assertEqual(len(state), CHECKPOINT_STATE_KEYS)
        self.assertEqual(tuple(state["conv1.weight"].shape), (64, 3, 3, 3))
        self.assertEqual(
            tuple(state["last_layer.3.weight"].shape),
            (2, 360, 1, 1),
        )
        self.assertTrue(all(isinstance(key, str) for key in state))
        self.assertTrue(all(torch.is_tensor(value) for value in state.values()))


if __name__ == "__main__":
    unittest.main()
