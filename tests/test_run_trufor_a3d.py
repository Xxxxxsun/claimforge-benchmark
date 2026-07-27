from __future__ import annotations

import numpy as np
from PIL import Image

from eval.our_defense.aggregate_a3d_results import (
    _fixed_operating_point,
    _roc_operating_point,
)
from eval.our_defense.run_trufor_a3d import (
    _calibration_threshold,
    _fused_map,
    _jpeg_recompress,
    _logit_mean_score,
    _rank_proposals,
)
from eval.our_defense.run_a3d_generated_full import (
    _canonicalize,
    _score_summary,
)
from eval.our_defense.run_trufor_adaptive_scan import _grid_boxes


def test_grid_boxes_cover_right_and_bottom_edges() -> None:
    boxes = _grid_boxes(1800, 1200, side=512, stride=384)
    assert len(boxes) == 15
    assert boxes[0] == (0, 0, 512, 512)
    assert boxes[-1] == (1288, 688, 1800, 1200)


def test_rank_proposals_uses_evidence_not_ground_truth() -> None:
    score_map = np.zeros((8, 16), dtype=np.float32)
    reliability = np.ones_like(score_map)
    score_map[:, 8:] = 0.8
    boxes = [(0, 0, 8, 8), (8, 0, 16, 8)]
    proposals, selected = _rank_proposals(
        score_map=score_map,
        reliability=reliability,
        boxes=boxes,
        budget=1,
    )
    assert len(proposals) == 2
    assert selected[0]["grid_index"] == 1


def test_fused_map_uses_highest_crop_scores() -> None:
    crops = [
        {"grid_index": 0, "box_xyxy": [0, 0, 2, 2], "score": 0.2},
        {"grid_index": 1, "box_xyxy": [2, 0, 4, 2], "score": 0.9},
    ]
    maps = [
        np.full((2, 2), 0.2, dtype=np.float32),
        np.full((2, 2), 0.9, dtype=np.float32),
    ]
    fused, selected = _fused_map((2, 4), crops, maps, count=1)
    assert selected == [1]
    np.testing.assert_array_equal(fused[:, :2], 0.0)
    np.testing.assert_array_equal(fused[:, 2:], 0.9)


def test_jpeg_recompress_preserves_geometry() -> None:
    image = Image.new("RGB", (17, 13), (120, 30, 200))
    recompressed = _jpeg_recompress(image, 90)
    assert recompressed.mode == "RGB"
    assert recompressed.size == image.size
    assert _jpeg_recompress(image, None) is image


def test_calibration_threshold_is_above_dev_real_quantile() -> None:
    rows = [
        {"kind": "real", "split": "dev", "a3d_score": value}
        for value in (0.1, 0.2, 0.3, 0.4)
    ]
    threshold = _calibration_threshold(rows, "a3d_score", alpha=0.05)
    assert threshold is not None
    assert threshold > 0.4


def test_logit_mean_score_is_symmetric_and_idempotent() -> None:
    assert _logit_mean_score(0.2, 0.8) == _logit_mean_score(0.8, 0.2)
    assert np.isclose(_logit_mean_score(0.73, 0.73), 0.73)
    assert np.isclose(_logit_mean_score(0.2, 0.8), 0.5)


def test_generated_full_canonicalization_preserves_geometry() -> None:
    image = Image.new("RGB", (19, 11), (30, 140, 220))
    canonical = _canonicalize(image, quality=95, subsampling=0)
    assert canonical.mode == "RGB"
    assert canonical.size == image.size


def test_generated_full_summary_uses_fixed_threshold() -> None:
    rows = [
        {
            "full_score": 0.1,
            "a3d_score": 0.2,
            "a3d_fused_score": 0.3,
            "total_latency_ms": 10,
        },
        {
            "full_score": 0.9,
            "a3d_score": 0.8,
            "a3d_fused_score": 0.7,
            "total_latency_ms": 20,
        },
    ]
    summary = _score_summary(rows, threshold=0.65)
    assert summary["images"] == 2
    assert summary["fused"]["positive_images"] == 1
    assert summary["fused"]["positive_rate"] == 0.5


def test_aggregate_operating_points_use_fused_scores() -> None:
    pairs = [
        {
            "real": {"full_score": real, "a3d_score": real},
            "forged": {"full_score": forged, "a3d_score": forged},
        }
        for real, forged in ((0.1, 0.9), (0.2, 0.8), (0.3, 0.7))
    ]
    roc_point = _roc_operating_point(
        pairs,
        "a3d_fused_score",
        max_fpr=0.01,
    )
    assert roc_point["actual_fpr"] == 0.0
    assert roc_point["tpr"] == 1.0

    fixed = _fixed_operating_point(pairs, threshold=0.65)
    assert fixed["fpr"] == 0.0
    assert fixed["tpr"] == 1.0
