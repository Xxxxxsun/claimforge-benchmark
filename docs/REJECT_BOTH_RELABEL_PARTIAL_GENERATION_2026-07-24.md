# Reject-Both Partial Relabel Generation (2026-07-24)

This run generated three HunyuanImage-3.0 candidates for every currently
complete Cat and trash-can relabel slot. It did not guess coordinates for
unfinished annotations.

## Ready inputs

- Cat: 28 complete slots.
  - 11 images still have no slot.
  - `lodging_170_slot_001` has a context box but no insertion box and was
    intentionally skipped.
- Trash can: 55 complete slots.
  - 6 images still have no slot.

The exported partial task and context sets are:

- `annotations/cat_generation_tasks_relabel_reject_both_partial28_20260724.jsonl`
- `crops/context_cat_relabel_reject_both_partial28_20260724/`
- `annotations/trash_can_generation_tasks_relabel_reject_both_partial55_20260724.jsonl`
- `crops/context_trash_can_relabel_reject_both_partial55_20260724/`

## Surface-aware trash-can placement

The per-task placement plan is:

`annotations/trash_can_relabel_reject_both_partial55_placement_plan_20260724.jsonl`

Its 55 marked supports comprise:

- 10 beds or mattresses;
- 20 tables, counters, shelves, or ledges;
- 4 other explicit supports: one closed toilet lid, one sofa seat, and two
  stool seats;
- 21 floors or ground areas.

The positioned generation tasks are:

`annotations/trash_can_generation_tasks_relabel_reject_both_partial55_positioned_20260724.jsonl`

Every prompt treats the human-marked support as authoritative. A bin marked on
a bed remains on the bed; a bin marked on a table remains on that table. The
prompt also requires a complete rim, body, and base, visible contact with the
support, source-matched blur or sharpness, lighting, noise, and compression.

## Outputs

Cat:

- `generated_crops/hunyuan_image3_distil_cat_relabel_reject_both_partial28_variant1_20260724/`
- `generated_crops/hunyuan_image3_distil_cat_relabel_reject_both_partial28_variant2_20260724/`
- `generated_crops/hunyuan_image3_distil_cat_relabel_reject_both_partial28_variant3_20260724/`

Trash can:

- `generated_crops/hunyuan_image3_distil_trash_can_relabel_reject_both_partial55_variant1_20260724/`
- `generated_crops/hunyuan_image3_distil_trash_can_relabel_reject_both_partial55_variant2_20260724/`
- `generated_crops/hunyuan_image3_distil_trash_can_relabel_reject_both_partial55_variant3_20260724/`

Each directory contains one PNG per task plus `manifest.jsonl`. Separate output
directories and deterministic seed salts prevent one variant from overwriting
another. Cat pose and off-camera orientation also vary with the variant salt.

## Visual-QC repairs

The first pass repeatedly omitted the requested bin for:

- `trash_can_lodging_013_slot_001`;
- `trash_can_lodging_050_slot_001`;
- `trash_can_restaurant_064_slot_001`.

All nine corresponding final candidates were replaced with visually checked
results. The floor and tabletop cases used a tighter edit-box context and were
feathered back into the original context crop. Every pixel outside the padded
marked region remained identical to the source context. The bed case retained
the full context and used a stronger support-specific retry. Final manifest
rows record these repairs in `qc_retry`.

## Validation

- Cat: `28 × 3 = 84` successful PNGs.
- Trash can: `55 × 3 = 165` successful PNGs.
- Total: `249` successful PNGs.
- Every final manifest has exactly one successful row per task.
- Every PNG is decodable RGB and matches both its manifest size and source
  context-crop size.
- Each task has three distinct seeds and three distinct SHA-256 image hashes.
- Trash-can prompts contain no residual instruction that forbids bed or
  furniture-top placement.
