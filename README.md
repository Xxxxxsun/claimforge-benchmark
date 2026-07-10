# CLAIMFORGE generation input handoff

This directory contains source images and local-edit annotations for the CLAIMFORGE restaurant/lodging benchmark expansion.

If you are the remote generation agent, read `AGENT_BRIEF.md` first.

## Expanded source pool and labeler

The current candidate source pool for the benchmark expansion is:

- `source_pool/openimages_v7_600/restaurant/`: 300 restaurant candidates.
- `source_pool/openimages_v7_600/lodging/`: 300 lodging/hotel/B&B candidates.
- `source_pool/openimages_v7_600/manifest.json`: combined 600-image manifest.
- `source_pool/openimages_v7_600/SUMMARY.md`: source-pool stats and screening notes.
- `source_pool/openimages_v7_600/contact_sheets/pages/`: 50-image pages for manual review.

The browser labeler is `tools/claimforge-labeler.html`. Serve this repository over
HTTP and open:

```text
http://localhost:8000/tools/claimforge-labeler.html
```

The labeler loads `source_pool/openimages_v7_600/manifest.json` by default.
Use the orange box for `insert_box` and the blue box for `crop_box`.

Current generation handoff:
- source pool: 600 images total, from `source_pool/openimages_v7_600/`.
- active batch: restaurant first part.
- complete restaurant tasks: 297.
- held out for later labeling: `restaurant_209` and `restaurant_212` are partial, `restaurant_229` is empty.
- lodging images are present in the source pool but are not part of this generation handoff yet.

Coordinate system:
- all boxes are in original source-image pixel coordinates.
- `insert_box` / `edit_region_xyxy` is the orange target area where the object should be inserted.
- `crop_box` / `context_region_xyxy` is the blue context crop to send to the image-editing model.
- `edit_region_in_context_xyxy` is the insert region after translating into the context crop coordinate frame.

Important files:
- `source_pool/openimages_v7_600/`: original source images and source metadata.
- `annotations/generation_tasks.jsonl`: easiest file for a generation agent. One JSON object per slot.
- `annotations/rekey_method_slots_payload.json`: same structure as the browser labeler export.
- `annotations/annotation_rows.jsonl`: REKEY-style rows: edit_region, context_region, add.objects.
- `source_manifest.json`: manifest subset for the 297 complete generation tasks.
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
