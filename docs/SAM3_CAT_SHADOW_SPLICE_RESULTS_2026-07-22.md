# SAM 3 distance-aware cat splice results (2026-07-22)

> **Status: rejected for production.** Manual review found visible rings of
> altered background around many cats. Connected, darkened differences are not
> reliable evidence that a pixel is fur or cast shadow. The v3 files are kept
> only for diagnosis; the Review page and active recommendation have been
> rolled back to the 2026-07-21 local-support baseline.
>
> A safer follow-up that excludes the edge ring from the final alpha is
> documented in `SAM3_CAT_SEMANTIC_SHADOW_RESULTS_2026-07-23.md`.

## Outcome

The long cast shadow in `cat_lodging_064_slot_001` was clipped because the old
hybrid mask retained source/generated residuals only inside a 12-pixel dilation
of the SAM 3 cat mask. SAM 3 itself should remain the identity-safe cat core;
cast shadows are appearance effects and should be recovered from the exact
generation-input difference rather than requested as part of the semantic cat
mask.

The replacement pipeline processed all 272 generated cats successfully in a
new result directory. It reused the saved SAM 3 RLE responses, made no network
requests, and did not modify `generated_crops/` or `spliced_full/`.

For a deterministic shadow ROI to the left of the SAM mask in task 064, the old
local hybrid covered 6.18% of changed pixels and the final mask covers 93.45%.
The final result preserves the complete visible shadow without using a box-wide
fallback.

## Final method

For each generated context crop:

1. Use the text-only SAM 3 mask as the semantic cat core.
2. Compute the difference against the exact `input_context_crop` recorded by
   the generation manifest, not a newly cropped or re-encoded source image.
3. Recover fur and antialiased object boundaries with max-channel difference
   greater than 8 inside a six-pixel dilation of the SAM mask.
4. Recover longer connected effects with a distance-aware threshold that rises
   from 20 near the cat to 40 at the search limit.
5. Outside the 12-pixel near-object zone, accept only pixels that are at least
   five luminance levels darker than the generation input and that originate
   below the top 40% of the SAM silhouette. This is the cast-shadow direction
   guard.
6. Close three pixels, propagate only from the trusted semantic/edge seed, grow
   two pixels, and optionally expand the search reach from 0.5 to 0.75 of the
   edit-box scale when the first reach boundary is hit.
7. Reject all far support if the candidate exceeds 3.5 times the semantic-mask
   area or adds more than 8% of the context crop. A rejected candidate safely
   falls back to the semantic core plus near-edge support.
8. Feather the semantic/edge core by one pixel and the accepted shadow support
   by three pixels before compositing.

The directional and area guards matter. Without them, connected differences in
pillows, counters, taps, bottles, or wall texture can be mistaken for part of
the object effect.

## Full-batch checks

| Check | Result |
|---|---:|
| Saved SAM 3 responses reused | 272 |
| Newly billed API requests | 0 |
| Materialized hybrid outputs | 272/272 |
| Materialization errors | 0 |
| Missing output files | 0 |
| Dimension mismatches | 0 |
| Outside-context pixel changes | 0 |
| Distance support accepted | 223 |
| Safe near-edge fallbacks | 49 |
| Mean semantic-mask fraction | 0.1007 |
| Mean final hybrid-mask fraction | 0.1460 |
| Mean distance-support fraction | 0.0156 |

One existing quality warning remains for `cat_restaurant_262_slot_001`: the
generated cat moved completely outside the original edit box. The text-only SAM
mask is nevertheless cat-shaped, has 94.65% residual support, and the rendered
splice is visually correct. This is a generation-placement warning, not a
missing or empty segmentation result.

The endpoint's historical listed cost for 272 requests was USD 1.36, but this
materialization incurred USD 0 because it read the existing append-only API
JSONL.

## Reproduction

```bash
python -m eval.segmentation.run_fal_sam3 \
  --tasks 272 \
  --endpoints sam3 \
  --prompt-mode text_only \
  --hybrid-mode semantic_hysteresis \
  --api-results-dir results/segmentation/fal_sam3_cat_textonly_full272_20260721_v1 \
  --output-dir results/segmentation/fal_sam3_cat_semantic_hysteresis_full272_20260722_v3 \
  --materialize-only
```

The important defaults are recorded in `run_manifest.json` and `summary.json`.
They can also be overridden with the `--edge-*`, `--far-*`, `--hysteresis-*`,
and `--max-*` command-line arguments.

## Artifacts and review

Final outputs:

- `results/segmentation/fal_sam3_cat_semantic_hysteresis_full272_20260722_v3/spliced_hybrid/sam3/`
- `results/segmentation/fal_sam3_cat_semantic_hysteresis_full272_20260722_v3/masks/sam3/`
- `results/segmentation/fal_sam3_cat_semantic_hysteresis_full272_20260722_v3/splice_results.jsonl`
- `results/segmentation/fal_sam3_cat_semantic_hysteresis_full272_20260722_v3/summary.json`
- `results/segmentation/fal_sam3_cat_semantic_hysteresis_full272_20260722_v3/contact_sheet.jpg`

Start the repository-aware local server:

```bash
python3 scripts/claimforge_label_server.py --port 8765
```

Then open:

```text
http://127.0.0.1:8765/tools/sam3-cat-review.html
```

The viewer defaults to the accepted 2026-07-21 baseline. To inspect this
rejected v3 run, append:

```text
?resultDir=results/segmentation/fal_sam3_cat_semantic_hysteresis_full272_20260722_v3
```

That view displays semantic, near-edge, and distance masks separately and
prioritizes the 49 safe-fallback cases plus the single edit-box placement
warning for review.

## Limitation

There is no human-drawn shadow ground truth. Cast-shadow recovery is therefore
validated with invariants, targeted visual inspection, and conservative
guardrails rather than pixel IoU against an annotation. A support surface can
still be visually inseparable from a contact shadow; such cases should remain
reviewable rather than being described as semantic cat pixels.
