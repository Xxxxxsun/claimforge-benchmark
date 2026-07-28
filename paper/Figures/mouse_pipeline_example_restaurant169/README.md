# Mouse pipeline figure assets

This package uses the frozen CLAIMFORGE v1 local-splice sample
`mouse/restaurant_169_slot_001`.

- Full benchmark image: `01_benchmark_full.png`
- Full image with the frozen context region outlined:
  `02_benchmark_full_context_box.png`
- Source-image context crop used by the legacy splice difference:
  `03_source_context.png`
- Exact JPEG context supplied to the image generator:
  `04_exact_generator_input_context.jpg` and a PNG conversion
- Generated context crop: `05_generated_context.png`
- SAM3 binary mask: `06_sam3_mask.png`
- SAM3 subject cutouts: `07_sam3_cutout_transparent.png` and
  `08_sam3_cutout_white.png`
- SAM3 overlay: `09_sam3_overlay.png`
- Raw absolute RGB difference: `10_diff_absolute_rgb.png`
- Exact max-channel difference intensity:
  `11_diff_absolute_max_channel.png`
- Contrast-enhanced difference for presentation only:
  `12_diff_absolute_visualized.png`
- Legacy mechanical difference mask at threshold 30:
  `13_diff_t30_mechanical_mask.png`
- Legacy mechanical-difference cutout:
  `14_diff_t30_cutout_transparent.png`
- Difference support added immediately around SAM3:
  `15_local_diff_support_near_sam3.png`
- Combined SAM3-plus-local-difference mask and cutout:
  `16_sam3_plus_diff_hybrid_mask.png` and
  `17_sam3_plus_diff_cutout_transparent.png`
- Context composites: `18_sam3_semantic_context_composite.png` and
  `19_sam3_plus_diff_context_composite.png`
- Full-image-size mask canvases: `20_sam3_mask_full_canvas.png` and
  `21_diff_t30_mask_full_canvas.png`
- Quick overview: `00_contact_sheet.png`

The mechanical difference is computed between the decoded real source-image
context and the generated context, matching the legacy local-splice benchmark
path. The exact generator-input JPEG is provided separately because JPEG
re-encoding makes it slightly different from a direct crop of the source
image.

SAM3 was run with the text prompt `mouse` using
`fal-ai/sam-3/image-rle`. The selected mask passed the repository quality gate
with provider score 0.9475.
