# Mouse pipeline figure assets: lodging_009_slot_001

This package uses the frozen CLAIMFORGE v1 local-splice sample
`mouse/lodging_009_slot_001`.

Core files:

- `01_benchmark_full.png`: frozen benchmark image.
- `02_benchmark_full_context_box.png`: full image with the 78×53 context
  region outlined.
- `03_source_context.png`: context reconstructed from the frozen source image
  and manifest coordinates.
- `05_generated_context.png`: original generated context crop.
- `06_sam3_mask.png`: SAM3 binary mouse mask.
- `07_sam3_cutout_transparent.png`: transparent SAM3 subject cutout.
- `08_sam3_cutout_white.png`: white-background SAM3 subject cutout.
- `09_sam3_overlay.png`: SAM3 mask over the generated context.
- `10_diff_absolute_rgb.png`: raw absolute per-channel RGB difference.
- `11_diff_absolute_max_channel.png`: exact max-channel difference intensity.
- `12_diff_absolute_visualized.png`: contrast-enhanced difference for
  presentation only.
- `13_diff_t30_mechanical_mask.png`: legacy mechanical difference mask at
  threshold 30.
- `14_diff_t30_cutout_transparent.png`: mechanical-difference cutout.
- `15_local_diff_support_near_sam3.png`: local difference pixels added around
  SAM3.
- `16_sam3_plus_diff_hybrid_mask.png`: combined SAM3-plus-local-difference
  mask.
- `17_sam3_plus_diff_cutout_transparent.png`: combined transparent cutout.
- `18_sam3_semantic_context_composite.png` and
  `19_sam3_plus_diff_context_composite.png`: context composites.
- `20_sam3_mask_full_canvas.png` and `21_diff_t30_mask_full_canvas.png`:
  full-image-size mask canvases.
- `22_source_context_8x.png`, `23_generated_context_8x.png`,
  `24_sam3_mask_8x_nearest.png`, `25_diff_t30_mask_8x_nearest.png`, and
  `26_sam3_cutout_white_8x.png`: figure-ready enlarged versions.
- `00_contact_sheet.png`: quick overview.

## Provenance note

The frozen manifest specifies context `[17, 271, 95, 324]`, giving 78×53
pixels, and the original generated crop is also 78×53. The shared path
`crops/context/lodging_009_slot_001_context.jpg` is currently 170×120 because
it was overwritten by a later task. It is therefore intentionally excluded.
`04_reconstructed_manifest_context.png` is reconstructed from the frozen
source image and coordinates.

SAM3 was run with text prompt `mouse` using
`fal-ai/sam-3/image-rle`. The selected mask passed the repository quality gate
with provider score 0.6474.
