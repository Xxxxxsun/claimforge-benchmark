from pathlib import Path
import sys

from PIL import Image
import pytest


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import run_hunyuan_full_image_orange_box as full_image


def test_model_size_keeps_whole_aspect_and_service_bounds():
    assert full_image.model_size(1800, 1200) == (1216, 832)
    assert full_image.model_size(375, 500) == (768, 1024)
    width, height = full_image.model_size(1920, 757)
    assert (width, height) in full_image.hunyuan_buckets()
    assert width % 16 == 0
    assert height % 16 == 0
    assert min(width, height) >= 512
    assert max(width, height) <= 2048


def test_scale_box_and_orange_guide():
    scaled = full_image.scale_box(
        [100, 50, 300, 200],
        (1000, 500),
        (1024, 512),
    )
    assert scaled == [102, 51, 307, 205]
    source = Image.new("RGB", (1024, 512), "white")
    guided, width = full_image.draw_orange_guide(source, scaled)
    assert width == 4
    assert guided.getpixel((scaled[0], scaled[1])) == full_image.DEFAULT_ORANGE
    assert guided.getpixel((200, 100)) == (255, 255, 255)


def test_restore_guide_ring_changes_only_the_outline():
    source = Image.new("RGB", (100, 80), (12, 34, 56))
    output = Image.new("RGB", (100, 80), (90, 80, 70))
    restored = full_image.restore_guide_ring(
        output,
        source,
        [20, 10, 70, 60],
        3,
    )
    assert restored.getpixel((20, 10)) == (12, 34, 56)
    assert restored.getpixel((30, 30)) == (90, 80, 70)
    assert restored.getpixel((0, 0)) == (90, 80, 70)


def test_prompts_use_only_orange_guide_as_location_signal():
    for kind in ("mouse", "cat", "trash-can"):
        prompt = full_image.make_prompt(kind, "task_001", "variant")
        assert "orange rectangle" in prompt
        assert "top-left" not in prompt
        assert "%" not in prompt
        assert "Remove the entire orange guide" in prompt


def test_cat_pose_varies_deterministically():
    first = full_image.make_prompt("cat", "task_001", "variant-a")
    assert first == full_image.make_prompt("cat", "task_001", "variant-a")
    variants = {
        full_image.make_prompt("cat", "task_001", f"variant-{index}")
        for index in range(20)
    }
    assert len(variants) > 1


def test_select_tasks_accepts_indices_and_ids():
    rows = [{"task_id": "a"}, {"task_id": "b"}, {"task_id": "c"}]
    assert full_image.select_tasks(rows, "0,c") == [rows[0], rows[2]]


def test_select_tasks_rejects_unknown_and_duplicate_tokens():
    rows = [{"task_id": "a"}, {"task_id": "b"}]
    with pytest.raises(ValueError, match="did not resolve"):
        full_image.select_tasks(rows, "missing")
    with pytest.raises(ValueError, match="same task twice"):
        full_image.select_tasks(rows, "a,0")
