# Resemble full-image mouse-250 results (2026-07-25)

## Scope

- Benchmark: `benchmark/claimforge_v1_250x3x2`
- Cell: `method=full_image`, `category=mouse`
- Frozen target: 250 forged images
- Domains: 128 restaurant and 122 lodging
- Input encoding: metadata-free RGB JPEG, quality 95, 4:4:4 subsampling
- API mode: synchronous Resemble Detect with `visualize=true`
- Real controls: not included in this run

The run manifest records all ordered task IDs, image paths, hashes, encoding
settings, and the benchmark manifest selector before inference.

## Completion and verdicts

| Measure | Value |
|---|---:|
| Expected | 250 |
| Valid | 250 |
| Errors | 0 |
| Heatmaps saved | 250 |
| Visualizations saved | 250 |
| `Fake` | 248 |
| `Likely fake` | 1 |
| `Real` | 1 |
| Combined positive (`Fake` + `Likely fake`) | 249 / 250 (99.6%) |

Provider score:

- mean: 0.988448
- median: 0.996084
- minimum: 0.010900
- maximum: 0.998714

IFL score:

- mean: 0.983205
- median: 0.993404
- minimum: 0.015743
- maximum: 0.998479

All 500 returned image artifacts match the SHA-256 values stored in the result
rows.

## Matched local-splice comparison

The earlier complete Resemble run contains the same 250 task IDs under the
local-splice route:

| Measure | Full image | Local splice |
|---|---:|---:|
| Valid matched tasks | 250 | 250 |
| Combined positive | 249 (99.6%) | 34 (13.6%) |
| `Fake` | 248 | 26 |
| `Likely fake` | 1 | 8 |
| Mean provider score | 0.988448 | 0.225887 |
| Median provider score | 0.996084 | 0.097917 |

The full-image provider score is greater for 245/250 matched tasks. The mean
paired `full_image - local_splice` score difference is 0.762560, and the median
difference is 0.897431. There are 215 tasks that are negative under the
local-splice combined rule but positive after full-image generation.

The mean paired IFL-score increase is 0.854885, with a median increase of
0.976201.

This is a strong route contrast: Resemble detects near-global reconstruction
very reliably but misses most local splice-back edits on the same source
tasks. It does not establish complete classifier performance because this run
is forged-only and has no matched real controls.

## Artifacts

- Results:
  `results/commercial/resemble/claimforge_v1_full_image_mouse250_canonical_jpeg_q95_20260725.jsonl`
- Frozen run manifest:
  `results/commercial/resemble/claimforge_v1_full_image_mouse250_canonical_jpeg_q95_20260725.run_manifest.json`
- Derived summary:
  `results/commercial/resemble/claimforge_v1_full_image_mouse250_canonical_jpeg_q95_20260725.summary.json`
- Heatmaps:
  `results/commercial/resemble/claimforge_v1_full_image_mouse250_canonical_jpeg_q95_20260725/heatmaps/`
- Visualizations:
  `results/commercial/resemble/claimforge_v1_full_image_mouse250_canonical_jpeg_q95_20260725/visualizations/`
