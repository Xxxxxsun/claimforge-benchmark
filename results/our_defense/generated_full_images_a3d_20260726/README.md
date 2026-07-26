# Generated-full A3D results

## Scope

This bundle contains frozen A3D detection results for every image present in
the three `generated_full_images/*` generation manifests:

| Category | Manifest images | Completed | Failed |
|---|---:|---:|---:|
| Mouse | 275 | 275 | 0 |
| Cat | 272 | 272 | 0 |
| Trash-can | 260 | 260 | 0 |
| **All** | **807** | **807** | **0** |

The manifest-derived IDs exactly match the 807 image files on disk. Duplicate
manifest retry records are reduced to their latest successful output.

## Protocol

- Frozen TruFor checkpoint:
  `ac1d90e329a72e0d66e8665e123a19e94bfae3209c3ef8a4f9ca3b91578c7844`
- Input canonicalization: JPEG quality 95, 4:4:4 subsampling.
- Proposal grid: 512x512 windows at stride 384.
- Proposal budget: four crops.
- Local score: maximum of the four crop probabilities.
- Primary score:
  `sigmoid((logit(full_score) + logit(local_score)) / 2)`.
- Fixed threshold: `0.6353510120379108`, transferred unchanged from the
  80-image mouse dev-real reference set.

No label, mask, object box, or generation prompt is used during inference.

## Fixed-threshold detection

| Category | Images | Full detected | Local detected | Fused detected |
|---|---:|---:|---:|---:|
| Mouse | 275 | 22 (8.00%) | 66 (24.00%) | 38 (13.82%) |
| Cat | 272 | 19 (6.99%) | 68 (25.00%) | 41 (15.07%) |
| Trash-can | 260 | 13 (5.00%) | 63 (24.23%) | 35 (13.46%) |
| **All** | **807** | **54 (6.69%)** | **197 (24.41%)** | **114 (14.13%)** |

These are detected fractions over generated images, not AUROC/AP: this folder
contains only generated candidates and supplies neither a negative population
nor pixel-level ground truth.

## Score distributions

| Category | Full mean / median | Local mean / median | Fused mean / median |
|---|---:|---:|---:|
| Mouse | 0.2730 / 0.2167 | 0.4236 / 0.3407 | 0.3509 / 0.2865 |
| Cat | 0.2656 / 0.2085 | 0.4342 / 0.3838 | 0.3501 / 0.2854 |
| Trash-can | 0.2419 / 0.2039 | 0.4025 / 0.3129 | 0.3259 / 0.2592 |
| **All** | **0.2605 / 0.2096** | **0.4204 / 0.3470** | **0.3426 / 0.2796** |

Median end-to-end latency is 508.9 ms per image.

## Interpretation

This threat condition is materially different from the final spliced
benchmark. Full-image generation tends to harmonize the whole image, while the
final benchmark deliberately preserves a real background and composites a
generated object. The current TruFor-based A3D is much more sensitive to the
local forensic inconsistency created by the latter.

The local branch alone detects about 24.4% of generated-full images, but the
equal-logit fusion reduces this to 14.1% because the full-image branch is
usually low. Therefore the existing fusion remains appropriate for the
small-splice benchmark, but it is not a strong universal detector for
fully-generated or globally regenerated images.

## Files

- `generated_full_all_807_a3d_q95_v1.jsonl`: one prediction and complete
  proposal diagnostics per generated image.
- `generated_full_all_807_a3d_q95_v1.summary.json`: aggregate and per-category
  score distributions, fixed-threshold counts, provenance, and latency.

The reusable runner is `eval/our_defense/run_a3d_generated_full.py`.
