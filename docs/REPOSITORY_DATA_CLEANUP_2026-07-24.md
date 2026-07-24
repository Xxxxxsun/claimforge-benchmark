# Repository data cleanup (2026-07-24)

This cleanup removes artifacts that have a confirmed replacement and no
remaining current-tree consumer. It deliberately preserves inputs required by
the current cat and trash-can workflows.

## Removed from the current `main` tree

| Directory | Files | Approximate checkout size | Reason |
|---|---:|---:|---|
| `generated_crops/hunyuan_image3_distil_trash_can_260_native_style_v1_20260722/` | 261 | 62 MB | Superseded by the audited complete-natural 112-task run and its 85 usable outputs |
| `spliced_full/hunyuan_image3_cat/` | 11 | 22 MB | Early 10-image cat pilot with no remaining current reference |
| `spliced_full/hunyuan_image3_distil_cat_272_fullblue_t40/` | 273 | 518 MB | Old threshold-40 splice experiment with no remaining current reference |

Total removed from the current checkout is approximately 602 MB across 545
tracked files.

## Final dataset consolidation

The same cleanup pass also:

- removed the isolated 10-image stain pilot bundle (generated outputs, dedicated
  context crops, and task manifest), which had no external consumer;
- consolidated the trash-can generation into
  `generated_crops/hunyuan_image3_distil_trash_can_complete_natural_usable85_20260723/`,
  containing only 85 usable PNGs and one 85-row manifest;
- removed the raw 112-image initial trash-can directory, both repair
  directories, and the 27 rejected images after preserving review provenance;
- created
  `annotations/trash_can_generation_tasks_regenerate_27_20260724.jsonl` so the
  rejected tasks can be regenerated directly.

This consolidation removes 206 obsolete tracked files (approximately 43.3 MiB)
from the current checkout, while the 85 retained images are moved without
content changes.

## Retired generic cat lineage

The generic cat pilot and first 272-image generation are superseded by the
native-style v2 generation. The current tree therefore removes:

- `generated_crops/hunyuan_image3_cat/`;
- `generated_crops/hunyuan_image3_distil_cat_272/`;
- the three corresponding `spliced_full` directories (plain, threshold-30, and
  Hysteresis-Distance);
- the 2026-07-21 SAM3 pilot, fallback, and remaining-task result directories;
- the pilot-specific result document.

The SAM3 runner and Hysteresis-Distance tutorial now default to the retained
native-style v2 lineage.

## Intentionally retained

- `generated_crops/hunyuan_image3_distil_cat_272_native_style_v2_20260722/`
  remains the direct generated-crop input for Hysteresis-SAM3 v2.
- `crops/context_trash_can/` remains shared input for the complete-natural
  trash-can generation and its 85 usable outputs.
- `spliced_full/hunyuan_image3_distil_cat_272_native_style_v2_20260722_hysteresis_distance/`
  remains the current native-style v2 cat baseline.
- `spliced_full/hunyuan_image3/` remains the mouse benchmark image set.

## Git-history boundary

The deletion removes these paths from the tip of `main`, so future ordinary
checkouts do not materialize them. Their blobs remain reachable through older
commits. Actually reducing full-history repository storage would require a
separate coordinated history rewrite and force push; that is intentionally
outside this cleanup.
