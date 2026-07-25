# Full-image orange-box control (2026-07-24)

## Purpose

This is the full-context control for the normal CLAIMFORGE
`context crop -> generate -> splice` workflow. It uses the same real source
images and human annotation boxes, but sends the complete source frame to
HunyuanImage-3 and supplies only one visible orange rectangle as the spatial
cue. It is a real-image conditional editing control, not the separate
whole-image synthetic contrast set.

The primary result is one fixed-seed, single-shot output per task. Conditional
retries and prompt/guide-width pilots are diagnostics only and must not replace
primary outputs: doing so would turn the control into an unequal-budget
best-of-N experiment.

## Frozen task sets

The task lists are built and validated by
`scripts/build_full_image_orange_box_tasks.py`.

| Class | Tasks | Frozen task list | Provenance |
|---|---:|---|---|
| mouse | 275 | `annotations/full_image_orange_box_mouse_good275_latest_20260724.jsonl` | Exact visual fields from `claimforge_generation_review_labels.json` records with `status=good` and `candidates=mouse` |
| cat | 272 | `annotations/full_image_orange_box_cat_latest272_20260724.jsonl` | 244 base rows plus 28 newest relabel rows replacing old rows by `task_id` |
| trash can | 260 | `annotations/full_image_orange_box_trash_can_latest260_20260724.jsonl` | 205 reviewed base rows plus 55 newest surface-aware relabel rows replacing old rows by `task_id` |

All 807 frozen tasks have unique task IDs and `(image_id, slot_id)` pairs.
Every source exists, each declared image size matches the file, and every edit
box lies within image bounds.

### Mouse review-snapshot correction

Seven lodging task IDs had been reused after the lodging source pool changed:
`lodging_001`, `007`, `009`, `013`, `015`, `017`, and `020` (all
`slot_001`). Joining the review to the current base only by `task_id` therefore
attached old `good` decisions to seven different images. The frozen control now
uses the immutable review snapshot's source, size, edit box, and context box,
which preserves the actual human-reviewed good-275 set. The seven superseded
full-image outputs and their pre-correction manifest/validation are retained in
`generated_full_images/hunyuan_image3_distil_full_input_orange_box_mouse_wrong_reused_id_backup_20260724/`.

Future review joins should include at least task ID, source SHA-256, and
edit/context coordinates.

## Input and output protocol

For each task:

1. Load the complete source image; do not crop it.
2. Select the nearest learned Hunyuan 1024-base aspect bucket and resize the
   complete frame to that exact bucket without center-cropping.
3. Scale `edit_region_xyxy` into model coordinates and draw one orange
   `(234, 122, 24)` rectangle. The text prompt contains no coordinate,
   percentage, top-left, or crop-position cue.
4. Call the local `vllm_hunyuan_image3` Omni edits endpoint.
5. Require the service output size to equal the requested learned bucket, then
   resize the complete returned frame to the exact original source dimensions.
6. Save the raw complete-frame RGB PNG. Do not splice source pixels back, and
   do not restore the guide ring from the source.

Because the whole service output is retained, global reconstruction changes and
orange-guide removal are both part of the control result.

## Fixed primary parameters

- HunyuanImage-3 Instruct Distil served as `vllm_hunyuan_image3`
- API mode: Omni image edits
- `bot_task=think_recaption` and `sys_type=en_unified`
- 8 diffusion steps
- guidance scale 5.0
- four concurrent requests
- deterministic per-task seed:
  `SHA-256(task_id + NUL + seed_salt)`, mapped to `[1, 9_000_000]`
- class-specific prompts require one complete object, exact orange-box
  placement, original-image style/sharpness/blur/noise/compression, and no
  unrelated edit
- cat pose and off-camera orientation vary deterministically by task
- trash-can support under the box is authoritative, including beds, tables,
  desks, counters, shelves, stools, and floors

One cat task, `cat_restaurant_134_slot_001`, deterministically triggered a
vLLM-Omni `q_len + ar_kv_len != seq_len` assertion under
`think_recaption`. Its complete image, box, prompt, seed, bucket, steps, and
guidance were kept fixed; only `bot_task=recaption` was used for that one
compatibility fallback. The manifest records the exception.

## Primary run directories

- Mouse:
  `generated_full_images/hunyuan_image3_distil_full_input_orange_box_mouse_good275_g5_v1_20260724/`
- Cat:
  `generated_full_images/hunyuan_image3_distil_full_input_orange_box_cat_latest272_g5_v1_20260724/`
- Trash can:
  `generated_full_images/hunyuan_image3_distil_full_input_orange_box_trash_can_latest260_g5_v1_20260724/`

Each run contains one `<task_id>.png`, an append-only `manifest.jsonl`, and a
generated `validation.json`. Manifest rows record the source path and SHA-256,
original/model/service sizes, source/model boxes, guide properties, exact
prompt, seed and salt, service settings, status, and elapsed time.

## Validation and interpretation

`scripts/validate_hunyuan_full_image_orange_box.py` checks:

- exact task/manifest/latest-success set agreement
- candidate class and object-kind agreement
- source path and SHA-256
- annotation box and original-size agreement
- model/service bucket-size agreement
- RGB mode and exact source-size output
- one PNG and one unique output hash per task
- orange-like pixels around the former guide as manual-review candidates

The disposable QC contact sheets draw the requested box in cyan after
generation. Cyan is never written into a model output.

Structural validity does not mean the requested object is correctly localized.
The full-image control frequently generates a recognizable object on a nearby
more convenient table, chair, bed, or floor; sometimes only a head or tail
touches the target, and sometimes the object is omitted. These are experimental
outcomes, not reasons to cherry-pick a retry. Orange-residual detection is also
heuristic and can be triggered by naturally orange furniture or lighting, so
every flag requires visual review.

## Completion and structural validation

| Class | Tasks | Latest successes | PNGs | Unique output hashes | Orange candidates |
|---|---:|---:|---:|---:|---:|
| mouse | 275 | 275 | 275 | 275 | 3 |
| cat | 272 | 272 | 272 | 272 | 1 |
| trash can | 260 | 260 | 260 | 260 | 0 |
| **Total** | **807** | **807** | **807** | **807** | **4** |

All four orange candidates were manually inspected and were caused by naturally
orange furniture or lighting, not a retained guide rectangle. Model input and
service output bucket distributions match exactly for all 807 latest results.

The mouse manifest is append-only and includes the seven pre-correction local
format failures plus the seven successful review-snapshot corrections. The cat
manifest includes two recorded `think_recaption` failures for
`cat_restaurant_134_slot_001` followed by its successful `recaption` fallback.
The latest row for every task is successful; historical failures remain for
auditability.

### Trash-can single-shot semantic QC

All 260 primary trash-can outputs were manually reviewed from the 11 cyan-box
contact sheets, with suspicious tiles opened at full resolution. A pass
requires exactly one recognizable and complete bin, including its base; no
frame-edge crop or substantial occlusion; placement at or immediately beside
the annotation on the same support surface; and a natural scale.

- Usable: **218 / 260 (83.85%)**
- Failed: **42 / 260 (16.15%)**

The main failure modes are displacement to a different support surface,
occluded or cropped bases, omission, multiple bins, wrong-object generation,
and conspicuous scale. The full decision record and per-task reasons are in
`annotations/full_image_orange_box_trash_can_single_shot_manual_qc_20260724.json`.
No failed primary output was replaced by a retry.
