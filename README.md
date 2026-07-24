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

## HunyuanImage-3.0-Instruct-Distil with vLLM-Omni

The local integration uses vLLM-Omni's OpenAI-compatible image edit endpoint.
Defaults assume an 8x80GB host, the model at
`/root/models/HunyuanImage-3-Instruct-Distil`, and the service on
`127.0.0.1:8001`.

Create/update the Python 3.12 environment. The setup script clones vLLM-Omni
v0.24.1 when needed and idempotently applies the included Distil/MeanFlow
compatibility patch:

```bash
scripts/setup_hunyuan_vllm.sh
```

Download or resume the pinned public snapshot through `hf-mirror` with all
proxy variables removed:

```bash
scripts/download_hunyuan_model.sh
```

Start the 4-GPU AR + 4-GPU DiT service:

```bash
scripts/start_hunyuan_vllm.sh
```

Run one task as an end-to-end smoke test:

```bash
python run_hunyuan_generation.py --only 0 --steps 8 \
  --model-name hunyuan_image3_distil_smoke
```

Export and run all completed cat slots, then compose them back into the full
source images:

```bash
python scripts/export_cat_generation_tasks.py
python run_hunyuan_generation.py \
  --tasks annotations/cat_generation_tasks.jsonl \
  --prompt-kind cat \
  --model-name hunyuan_image3_distil_cat_272_native_style_v2_20260722 \
  --steps 8 --timeout 1800 --resume
python compose_spliced_full.py \
  --tasks annotations/cat_generation_tasks.jsonl \
  --model-name hunyuan_image3_distil_cat_272_native_style_v2_20260722
```

The cat prompt matches the source crop's style and image quality, including
blur and compression artifacts, and deterministically varies natural poses and
off-camera orientations by task ID. The versioned output directory preserves
the earlier cat run and its downstream review artifacts.

Export and run all completed trash-can slots, then compose them back into the
full source images:

```bash
python scripts/export_trash_can_generation_tasks.py
python run_hunyuan_generation.py \
  --tasks annotations/trash_can_generation_tasks_natural_112.jsonl \
  --prompt-kind trash-can \
  --model-name hunyuan_image3_distil_trash_can_complete_natural_v4 \
  --steps 8 --timeout 1800 \
  --seed-salt complete-natural-v4 --resume
python compose_spliced_full.py \
  --tasks annotations/trash_can_generation_tasks_natural_112.jsonl \
  --model-name hunyuan_image3_distil_trash_can_complete_natural_v4
```

The trash-can prompt asks for a small, complete, unobstructed bin safely inset
from every crop edge, in an unobtrusive and physically sensible floor/ground
location that does not block a doorway, walkway, seat, or work area. It also
matches source sharpness or blur, detail, noise, compression, lighting,
perspective, and depth of field instead of forcing a uniformly photorealistic
insert.

The original 260 generic insert slots were also reviewed for trash-can
suitability. The strict 112-task manifest above removes targets on beds, sofas,
counters, bathroom fixtures, people, and furniture-crowded or border-constrained
regions where a complete freestanding bin cannot also look natural. See
`docs/TRASH_CAN_SOURCE_SUITABILITY_2026-07-23.md` for the full audit and excluded
IDs.

The reviewed 2026-07-23 deliverable is
`generated_crops/hunyuan_image3_distil_trash_can_complete_natural_usable85_20260723/`.
It contains only the 85 outputs that passed strict visual QA (restaurant 29,
lodging 56), in their original task order, including 12 successful targeted
repairs. The 27 rejected outputs are not part of the deliverable; their tasks
are ready for regeneration in
`annotations/trash_can_generation_tasks_regenerate_27_20260724.jsonl`.
Per-task status and reasons are in
`annotations/trash_can_complete_natural_review_20260723.jsonl`, and the generation
and QA summary is in
`docs/TRASH_CAN_COMPLETE_NATURAL_QA_2026-07-23.md`. The initial and raw repair
directories were removed after consolidation; selected revision metadata remains
in the final manifest and review record.

The generation client posts multipart requests to `/v1/images/edits`, keeps
the input and output dimensions equal, disables environment proxy discovery,
and decodes the returned `b64_json` image. Use `--api-style legacy` only for
the retired Tencent vLLM fork's custom chat-completions endpoint.

## Hosted SAM 3 semantic splice masks

The fal SAM 3 runner segments the generated object, adds only nearby
source/generated residual pixels for shadow and fine edges, and pastes the
hybrid result back inside the context crop. It is append-only and resumes saved
queue request IDs without duplicate submissions.

For generated cat crops, use the cheaper SAM 3 RLE endpoint with a text-only
prompt:

```bash
export FAL_KEY='<fal-key>'
python -m eval.segmentation.run_fal_sam3 \
  --prompt-mode text_only --endpoints sam3 --tasks 10 \
  --output-dir results/segmentation/my_sam3_cat_pilot
unset FAL_KEY
```

The 2026-07-21 pilot, prompt/endpoint comparison, quality gate, and saved
artifacts are documented in
`docs/SAM3_CAT_SPLICE_PILOT_RESULTS_2026-07-21.md`.
