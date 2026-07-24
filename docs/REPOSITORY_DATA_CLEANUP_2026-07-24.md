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

## Intentionally retained

- `generated_crops/hunyuan_image3_distil_cat_272_native_style_v2_20260722/`
  remains the direct generated-crop input for Hysteresis-SAM3 v2.
- `crops/context_trash_can/` remains shared input for the complete-natural
  trash-can generation and its 85 usable outputs.
- `spliced_full/hunyuan_image3_distil_cat_272_fullblue_t30/` remains referenced
  by the SAM3 runner default and historical run manifests.
- `spliced_full/hunyuan_image3_distil_cat_272_hysteresis_distance/` remains
  documented as a reproducible baseline in `CAT_SPLICE_BACK_TUTORIAL.md`.
- `spliced_full/hunyuan_image3_distil_cat_272_native_style_v2_20260722_hysteresis_distance/`
  remains the current native-style v2 cat baseline.
- `spliced_full/hunyuan_image3/` remains the mouse benchmark image set.

## Git-history boundary

The deletion removes these paths from the tip of `main`, so future ordinary
checkouts do not materialize them. Their blobs remain reachable through older
commits. Actually reducing full-history repository storage would require a
separate coordinated history rewrite and force push; that is intentionally
outside this cleanup.
