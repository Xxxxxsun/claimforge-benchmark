import argparse
from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import build_full_image_orange_box_tasks as builder


def test_select_good_mouse_uses_reviewed_visual_snapshot_on_id_collision():
    base = [
        {
            "task_id": "lodging_001_slot_001",
            "image_id": "lodging_001",
            "slot_id": "slot_001",
            "source_image": "source_pool/new/lodging_001.jpg",
            "image_size": {"width": 1800, "height": 1350},
            "edit_region_xyxy": [1345, 1118, 1368, 1138],
            "candidates": "mouse",
        }
    ]
    reviewed = {
        "records": [
            {
                "task_id": "lodging_001_slot_001",
                "source_image": "images/lodging_001.jpg",
                "image_size": [500, 750],
                "edit_region_xyxy": [145, 224, 178, 251],
                "context_region_xyxy": [123, 210, 204, 272],
                "status": "good",
                "candidates": "mouse",
            }
        ]
    }

    selected = builder.select_good_mouse(base, reviewed)

    assert selected == [
        {
            **reviewed["records"][0],
            "image_id": "lodging_001",
            "slot_id": "slot_001",
            "insert_box": {
                "x": 145,
                "y": 224,
                "width": 33,
                "height": 27,
            },
        }
    ]


def test_build_full_control_task_counts_without_writing():
    args = argparse.Namespace(
        mouse_base=builder.DEFAULT_MOUSE_BASE,
        mouse_review=builder.DEFAULT_MOUSE_REVIEW,
        cat_base=builder.DEFAULT_CAT_BASE,
        cat_replacements=builder.DEFAULT_CAT_REPLACEMENTS,
        trash_base=builder.DEFAULT_TRASH_BASE,
        trash_replacements=builder.DEFAULT_TRASH_REPLACEMENTS,
        mouse_output=builder.DEFAULT_MOUSE_OUTPUT,
        cat_output=builder.DEFAULT_CAT_OUTPUT,
        trash_output=builder.DEFAULT_TRASH_OUTPUT,
        dry_run=True,
    )
    result = builder.build(args)
    assert result["mouse"]["tasks"] == 275
    assert result["cat"] == {
        "tasks": 272,
        "replacements": 28,
        "output": builder.DEFAULT_CAT_OUTPUT.as_posix(),
    }
    assert result["trash_can"] == {
        "tasks": 260,
        "replacements": 55,
        "output": builder.DEFAULT_TRASH_OUTPUT.as_posix(),
    }
    assert result["total_tasks"] == 807
