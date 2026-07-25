from pathlib import Path
import sys

from PIL import Image, ImageDraw
import pytest


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import validate_hunyuan_full_image_orange_box as validator


def test_orange_fraction_detects_guide_ring():
    image = Image.new("RGB", (100, 80), "white")
    box = [20, 10, 70, 60]
    ImageDraw.Draw(image).rectangle(box, outline=validator.ORANGE, width=4)
    assert validator.orange_fraction(image, box, 4) == 1.0


def test_qc_crop_contains_target_and_fits_tile():
    image = Image.new("RGB", (1000, 800), "black")
    ImageDraw.Draw(image).rectangle([400, 300, 450, 350], fill="white")
    crop = validator.qc_crop(
        image,
        [400, 300, 450, 350],
        tile_image_size=(284, 204),
    )
    assert crop.width <= 284
    assert crop.height <= 204
    assert max(channel[1] for channel in crop.getextrema()) == 255


def test_select_tasks_resolves_mixed_index_and_id_strictly():
    rows = [{"task_id": "a"}, {"task_id": "b"}, {"task_id": "c"}]
    assert validator.select_tasks(rows, "0,c") == [rows[0], rows[2]]
    with pytest.raises(ValueError, match="did not resolve"):
        validator.select_tasks(rows, "missing")
    with pytest.raises(ValueError, match="same task twice"):
        validator.select_tasks(rows, "b,1")
