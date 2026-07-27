# Mouse A3D durable results

This bundle makes the previously ignored mouse A3D artifacts Git-auditable.

## Final paired runs

- `clean/`: 275 pairs / 550 images, all successful.
- `jpeg90/`: the same 275 pairs recompressed in memory at JPEG quality 90,
  all successful.

Each directory contains the original per-image JSONL plus a summary regenerated
with the current three detection branches:

- `full_score`: frozen full-image TruFor;
- `a3d_score`: maximum of four selected crop scores (local-only ablation);
- `a3d_fused_score`: equal-weight mean of full/local logits (current primary).

Older physical JSONL rows predate the explicit fused field. The runner and
analyzer reconstruct the fused score deterministically from the two stored
scores. No inference or fitting is repeated.

The analysis files use 2,000 pair-level bootstrap replicates. The fixed
threshold `0.6353510120379108` uses only 80 clean mouse dev-real images.

## Q1 diagnostics

- `diagnostics/deployable_q1/`: the original four-crop Q1 pilot.
- `diagnostics/adaptive_scan_q1/`: exhaustive and 4/8-crop proposal-budget
  comparisons.
- `diagnostics/oracle_zoom_q1/`: ground-truth-centered 256/512 scale
  opportunity diagnostic. This is not deployable.

The full interpretation and every metric table are in
`docs/A3D_ADAPTIVE_DEFENSE_FULL_REPORT_2026-07-27.md`.
