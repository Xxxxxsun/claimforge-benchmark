# SAM 3 semantic-core plus shadow candidate (2026-07-23)

## Status

**Rejected for automatic production use; not promoted into `spliced_full/`.**

Manual review found the complementary failure to the 2026-07-22 halo issue:
when the area guard rejects broad background changes, subtle contact shadows
such as the `013` examples disappear completely. Preserving a narrow six-pixel
contact band reintroduces a U-shaped silhouette ring rather than isolating a
true shadow. Difference heuristics therefore trade false background support
against missed shadows and are not a reliable automatic shadow segmenter.

The 2026-07-22 distance-aware experiment was rejected because its low-threshold
edge support formed a visible ring of regenerated background around many cats.
This follow-up keeps the useful shadow search but removes that ring from the
actual composite.

## Change

The new `semantic_shadow` mode separates connectivity evidence from emitted
pixels:

```text
internal search seed = SAM3 semantic mask + low-threshold edge bridge
final object alpha   = SAM3 semantic mask, feathered inward only
final effect alpha   = lower, darkened, connected shadow support
```

The edge bridge may still look ring-shaped when displayed as a diagnostic mask.
It is never unioned into the final hybrid mask. It only lets hysteresis cross a
small antialiased gap between the SAM silhouette and a cast shadow.

Additional safeguards:

- semantic alpha is zero everywhere outside the SAM mask;
- shadow pixels must be below the top 40% of the SAM silhouette;
- shadow pixels must darken the exact generation input by at least five luma
  levels;
- shadow alpha has a separate three-pixel feather;
- excessive support still falls back to the pure SAM semantic mask.

## Full 272-image result

| Check | Result |
|---|---:|
| Materialized outputs | 272/272 |
| New API requests | 0 |
| Missing files | 0 |
| Dimension mismatches | 0 |
| Outside-context changes | 0 |
| Tasks emitting any upper edge-ring pixel | 0 |
| Tasks emitting edge-only support pixels | 0 |
| Accepted shadow support | 251 |
| Pure-SAM safety fallbacks | 21 |
| Mean semantic-mask fraction | 0.1007 |
| Mean shadow-support fraction | 0.0292 |
| Mean final hybrid-mask fraction | 0.1299 |

For `cat_lodging_064_slot_001`, the baseline local mask covers 6.18% of the
long-shadow diagnostic ROI; `semantic_shadow` covers 93.27%.

The existing placement warning for `cat_restaurant_262_slot_001` remains: its
generated cat moved outside the original edit box. This is not an empty-mask or
materialization failure.

## Reproduction

```bash
python -m eval.segmentation.run_fal_sam3 \
  --tasks 272 \
  --endpoints sam3 \
  --prompt-mode text_only \
  --hybrid-mode semantic_shadow \
  --api-results-dir results/segmentation/fal_sam3_cat_textonly_full272_20260721_v1 \
  --output-dir results/segmentation/fal_sam3_cat_semantic_shadow_full272_20260723_v1 \
  --materialize-only
```

Final candidate artifacts:

- `results/segmentation/fal_sam3_cat_semantic_shadow_full272_20260723_v1/spliced_hybrid/sam3/`
- `results/segmentation/fal_sam3_cat_semantic_shadow_full272_20260723_v1/masks/sam3/`
- `results/segmentation/fal_sam3_cat_semantic_shadow_full272_20260723_v1/splice_results.jsonl`
- `results/segmentation/fal_sam3_cat_semantic_shadow_full272_20260723_v1/summary.json`
- `results/segmentation/fal_sam3_cat_semantic_shadow_full272_20260723_v1/contact_sheet.jpg`

Review locally at:

```text
http://127.0.0.1:8765/tools/sam3-cat-review.html?resultDir=results/segmentation/fal_sam3_cat_semantic_shadow_full272_20260723_v1
```

The page otherwise defaults to the accepted 2026-07-21 baseline. With the
experimental result directory selected, it labels the ring-shaped diagnostic
as “内部连通桥（不贴回）” and shows the actually emitted shadow mask separately.

## Remaining limitation

Dark, regenerated support surfaces can still resemble a cast shadow. The area
and direction guards reduce this failure mode but cannot prove shadow semantics
without a shadow-specific model or manual ground truth. This candidate should
therefore be visually reviewed before promotion.
