# Hive full-image mouse-250 run status (updated 2026-07-27)

## Scope

- Benchmark: `benchmark/claimforge_v1_250x3x2`
- Cell: `method=full_image`, `category=mouse`
- Frozen target: 250 forged images
- Input encoding: metadata-free RGB JPEG, quality 95, 4:4:4 subsampling
- Hive endpoint: V3 AI-generated and deepfake content detection
- Vendor threshold: 0.9
- Real controls: not included in this run

The runner reads the frozen benchmark manifest directly. Its run manifest
records all 250 ordered inputs and hashes before the first API call.

## Current completion

| Measure | Value |
|---|---:|
| Expected | 250 |
| Valid | 250 |
| Pending | 0 |
| Detected at 0.9 | 107 / 250 |
| Detection rate | 42.8% |
| Mean AI score | 0.604841 |
| Median AI score | 0.777383 |
| Minimum AI score | 0.000010 |
| Maximum AI score | 0.999995 |

The first credential batch produced 100 valid requests. A newly supplied
credential produced exactly 100 additional valid requests before returning
`429 Too Many Requests`. On the final resume, that new organization returned
HTTP 405 because it had been paused; a previously supplied active credential
then completed the remaining 50 images. Historical error rows remain in the
append-only JSONL, while the latest row for every input is valid.

The complete cell contains 128 restaurant and 122 lodging images.

## Matched local-splice comparison

The earlier complete Hive run contains the same 250 task IDs under the
local-splice route:

| Measure | Full image | Local splice |
|---|---:|---:|
| Valid matched tasks | 250 | 250 |
| Detected at 0.9 | 107 | 0 |
| Mean AI score | 0.604841 | 0.000778 |
| Median AI score | 0.777383 | 0.000166 |

The full-image score is greater for 248/250 matched tasks. The mean paired
`full_image - local_splice` score difference is 0.604063, and the median
difference is 0.775313.

This contrast is consistent with the benchmark hypothesis that retaining most
real pixels makes local splice-back edits harder for whole-image AI detectors.
The forged-only cell is complete, but it still lacks matched real controls.

## Artifacts

- Results:
  `results/commercial/hive/claimforge_v1_full_image_mouse250_canonical_jpeg_q95_20260725.jsonl`
- Frozen run manifest:
  `results/commercial/hive/claimforge_v1_full_image_mouse250_canonical_jpeg_q95_20260725.run_manifest.json`
- Derived summary:
  `results/commercial/hive/claimforge_v1_full_image_mouse250_canonical_jpeg_q95_20260725.summary.json`

## Reproduce

The same command can be used to reproduce or resume the run:

```bash
HIVE_API_KEY=... .venv/bin/python -m eval.commercial.run_hive \
  --benchmark-manifest benchmark/claimforge_v1_250x3x2/manifest.jsonl \
  --benchmark-category mouse \
  --benchmark-method full_image \
  --tasks 250 \
  --include forged \
  --output results/commercial/hive/claimforge_v1_full_image_mouse250_canonical_jpeg_q95_20260725.jsonl \
  --run-id hive_claimforge_v1_full_image_mouse250_20260725 \
  --minimum-interval 0.5
```

The runner now skips all 250 IDs because every latest row has `status=ok`.
