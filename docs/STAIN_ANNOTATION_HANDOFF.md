# Stain Annotation Handoff

This document describes the second CLAIMFORGE annotation pass for medium-size stain edits.

## Purpose

The first generated/reviewed class was `mouse`. From the 594 mouse generations, 275 were marked `good` in:

```text
claimforge_generation_review_labels.json
```

Those 275 good sample IDs are reused only as a quality filter. The stain pass must annotate the original source images, not the mouse-spliced images.

## Open The Labeler

From the repository root:

```bash
python3 -m http.server 8000 --directory .
```

Open:

```text
http://localhost:8000/tools/stain-labeler.html
```

This redirects to `tools/claimforge-labeler.html` with stain-specific query parameters.

## Input Manifest

The stain source subset is:

```text
source_pool/good_mouse_source_stain_275/manifest.json
```

It contains 275 original source images:

- 128 restaurant images
- 147 lodging images

Every `path` in this manifest points to the original source image, for example:

```text
../../source_pool/openimages_v7_600/restaurant/restaurant_000.jpg
```

The manifest also keeps provenance fields from the mouse review, including:

- `selection_spliced_image`
- `selection_generated_crop`
- `previous_edit_region_xyxy`
- `previous_context_region_xyxy`

These are for traceability only. Do not use them as the stain boxes.

## Annotation Target

Default candidate:

```text
mold stain
```

Use the same two-box convention:

- orange `insert_box`: where the stain should appear
- blue `crop_box`: the context crop sent to the image-editing model

The stain should be placed on plausible surfaces:

- wall, ceiling, floor
- table, countertop
- bedsheet, curtain, sofa, carpet
- bathroom corner, tile, sink area

Avoid labeling a source if the underlying image is unsuitable for a stain edit.

## Browser Storage

The stain labeler stores progress in browser `localStorage` under:

```text
claimforge-good-mouse-source-stain-275-labels-v1
```

This is intentionally separate from the previous mouse annotation storage.

## Export

Click `Download` in the labeler after annotation. The expected filename is:

```text
claimforge-good-mouse-source-stain-275-slots.json
```

Expected payload shape:

```text
task: claimforge_stain_slots
coordinate_space: original_image_pixels
```

Each image entry contains one or more slots with:

- `insert_box`
- `crop_box`
- `candidates`

## Current Local State

The source manifest and labeler entrypoint are in the repo. The actual exported annotation JSON is only present if the browser user clicked `Download` and added the JSON file to the repo.

If the exported JSON is missing, open the labeler, click `Download`, and place the downloaded JSON at the repo root or under `annotations/` before running generation.

## Suggested Next Processing Step

Convert the exported slots JSON into the same task format used by the mouse pipeline:

- `insert_box` -> `edit_region_xyxy`
- `crop_box` -> `context_region_xyxy`
- `candidates` -> `add` / prompt target

Suggested output names:

```text
annotations/stain_generation_tasks.jsonl
generated_crops/<model>_stain/
spliced_full/<model>_stain/
```

Keep this pass separate from the existing mouse outputs so detector experiments can compare object size/type ladders cleanly.
