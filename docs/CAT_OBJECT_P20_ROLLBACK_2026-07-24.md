# Cat object-p20 rollback (2026-07-24)

## Decision

Use the legacy single-threshold object mask for the native-style v2 cat
composites:

```text
blend=object
object_search=padded
object_pad=20
object_threshold=30
feather=2
```

This replaces the active `hysteresis-distance`/SAM3 experiments with a simple
fixed-pixel search limit. Those experimental outputs remain in their original
directories for comparison and are not deleted.

## Output

```text
spliced_full/hunyuan_image3_distil_cat_272_native_style_v2_20260722/
```

The directory contains 272 PNG files and a 272-row `manifest.jsonl`.

Validation:

- 272/272 output files present;
- 272 unique task IDs;
- every manifest row records `paste_mode=object_only`;
- every row records `object_search=padded`, `object_pad=20`, and
  `object_threshold=30`;
- 0 images modify pixels outside `context_region_xyxy`.

The 20-pixel value limits candidate-mask search around the edit box. The
two-pixel feather creates a small alpha transition beyond the binary search
boundary; observed changed pixels extend at most 25 pixels from the edit box.

## Reproduce

```bash
python compose_spliced_full.py \
  --tasks annotations/cat_generation_tasks.jsonl \
  --model-name hunyuan_image3_distil_cat_272_native_style_v2_20260722 \
  --generated-manifest generated_crops/hunyuan_image3_distil_cat_272_native_style_v2_20260722/manifest.jsonl \
  --out-dir spliced_full/hunyuan_image3_distil_cat_272_native_style_v2_20260722 \
  --blend object \
  --object-search padded \
  --object-pad 20 \
  --object-thr 30 \
  --feather 2
```

## Review

Start the repository server, then open:

```text
http://localhost:8000/tools/cat-object-p20-review.html
```

The review labels use a new storage key and save path, so they cannot overwrite
the prior hysteresis-distance or SAM3 labels.
