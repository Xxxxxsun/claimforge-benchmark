# Cat Annotation Handoff

We are pivoting the second CLAIMFORGE edit type from stain to cat.

Reason: the stain pilot was visually unstable. Hunyuan often produced weak shadows, surface wear, or patch-like artifacts rather than a clearly controllable stain. A cat is a cleaner second category because it remains object insertion like `mouse`, but is much larger and easier to judge.

## Open The Labeler

Run the local save server from the repository root:

```bash
python3 scripts/claimforge_label_server.py --port 8000
```

Open:

```text
http://localhost:8000/tools/cat-labeler.html
```

This redirects to `tools/claimforge-labeler.html` with cat-specific parameters.

## Input Images

The cat pass reuses the same 275 original source images selected from good mouse generations:

```text
source_pool/good_mouse_source_stain_275/manifest.json
```

Despite the directory name containing `stain`, the manifest paths point to the original source images, not mouse-spliced or stain-spliced images.

Counts:

- 128 restaurant images
- 147 lodging images

## Annotation Target

Default candidate:

```text
cat
```

Use the same two-box convention:

- orange `insert_box`: where the cat should appear
- blue `crop_box`: the context crop sent to the image-editing model

A cat needs more space than the mouse pass. Prefer plausible locations such as:

- restaurant floor, under/near tables, chairs, counters
- hotel room floor, bed, sofa, hallway, lobby
- avoid tiny surfaces where a cat cannot physically fit

## Storage And Autosave

Browser localStorage key:

```text
claimforge-good-mouse-source-cat-275-labels-v1
```

Autosaved repo JSON:

```text
annotations/claimforge-good-mouse-source-cat-275-slots.json
```

Manual download filename:

```text
claimforge-good-mouse-source-cat-275-slots.json
```

Expected task name:

```text
claimforge_cat_slots
```

Keep cat outputs separate from mouse and stain pilot outputs:

```text
annotations/cat_generation_tasks.jsonl
generated_crops/<model>_cat/
spliced_full/<model>_cat/
```
