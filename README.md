# CLAIMFORGE generation input handoff

This directory contains source images and local-edit annotations for the lodging/restaurant demo batch.

If you are the remote generation agent, read `AGENT_BRIEF.md` first.

Counts:
- source images: 22
- slots: 22

Coordinate system:
- all boxes are in original source-image pixel coordinates.
- `insert_box` / `edit_region_xyxy` is the orange target area where the object should be inserted.
- `crop_box` / `context_region_xyxy` is the blue context crop to send to the image-editing model.
- `edit_region_in_context_xyxy` is the insert region after translating into the context crop coordinate frame.

Important files:
- `images/`: original source images.
- `annotations/generation_tasks.jsonl`: easiest file for a generation agent. One JSON object per slot.
- `annotations/rekey_method_slots_payload.json`: same structure as the browser labeler export.
- `annotations/annotation_rows.jsonl`: REKEY-style rows: edit_region, context_region, add.objects.
- `crops/context/`: blue-box crops to send to the generation model.
- `crops/insert/`: orange-box crops for quick visual checking.
- `masks/`: full-resolution binary masks for insert regions.
- `overlays/`: source images with blue context boxes and orange insert boxes drawn.
- `generated_crops/`: generated context crops returned by image-editing models.
- `spliced_full/`: full source images with generated context crops pasted back.

Expected remote flow:
1. For each row in `annotations/generation_tasks.jsonl`, run the image-edit model on `context_crop`.
2. Ask it to add `candidates` inside `edit_region_in_context_xyxy`.
3. Save generated crops with the same `task_id`.
4. Send generated crops back for spliced-back composition using `context_region_xyxy`.
