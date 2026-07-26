from __future__ import annotations

import numpy as np
from PIL import Image

from eval.our_defense.build_final_canonical import (
    _context_to_full,
    _observed_context_mask,
)


def test_context_to_full_places_mask_at_reviewed_coordinates() -> None:
    context = Image.new("L", (3, 2), 255)
    full = _context_to_full("task", context, [2, 1, 5, 3], (7, 5))
    array = np.asarray(full)
    assert int(np.count_nonzero(array)) == 6
    assert np.all(array[1:3, 2:5] == 255)
    assert np.all(array[:1] == 0)


def test_observed_context_mask_excludes_changes_outside_context() -> None:
    source = Image.new("RGB", (6, 4), (0, 0, 0))
    forged = source.copy()
    forged.putpixel((0, 0), (255, 255, 255))
    forged.putpixel((3, 2), (1, 0, 0))
    mask = _observed_context_mask(source, forged, [2, 1, 5, 4])
    array = np.asarray(mask)
    assert mask.size == (3, 3)
    assert int(np.count_nonzero(array)) == 1
    assert array[1, 1] == 255
