# Resemble Detect: CLAIMFORGE Balanced250

- Run completed: `2026-07-27T20:21:59.489604+00:00`
- Decision rule: provider label is `Fake` or `Likely fake`.
- Upload policy: metadata-free RGB JPEG Q95 with 4:4:4 subsampling.
- Resemble heatmaps are retained as provider-rendered RGB/JPEG artifacts; they are not interpreted as calibrated pixel scores.

## Coverage

- New panel: 750/750 valid, 0 final errors.
- Raw API attempts: 751; 1 provider-processing failure was superseded by a successful retry.
- Reused mouse calls: 500/500 valid.
- Evaluated primary cells: 1250/1750 images.
- Deferred before submission: 500 full-frame cat/trash-can images.
- Saved provider artifacts in the new run: 750 heatmaps and 750 visualizations.

## Image-Level Results

Real false positives: 3/250 (1.2%); specificity 98.8%.

| Condition | TP / 250 | TPR | Specificity | Balanced acc. | 95% bootstrap CI |
|---|---:|---:|---:|---:|---:|
| `local_mouse` | 34 / 250 | 13.6% | 98.8% | 56.2% | 54.0%-58.6% |
| `local_cat` | 179 / 250 | 71.6% | 98.8% | 85.2% | 82.2%-88.0% |
| `local_trash_can` | 143 / 250 | 57.2% | 98.8% | 78.0% | 74.8%-81.0% |
| `fullframe_mouse` | 249 / 250 | 99.6% | 98.8% | 99.2% | 98.4%-99.8% |
| `fullframe_cat` | deferred | deferred | deferred | deferred | deferred |
| `fullframe_trash_can` | deferred | deferred | deferred | deferred | deferred |

## Labels

| Condition | Label counts |
|---|---|
| `local_mouse` | Fake: 26, Likely fake: 8, Likely real: 34, Neutral/Uncertain: 5, Real: 177 |
| `local_cat` | Fake: 162, Likely fake: 17, Likely real: 5, Neutral/Uncertain: 11, Real: 55 |
| `local_trash_can` | Fake: 126, Likely fake: 17, Likely real: 5, Neutral/Uncertain: 10, Real: 92 |
| `fullframe_mouse` | Fake: 248, Likely fake: 1, Real: 1 |
| `fullframe_cat` | deferred |
| `fullframe_trash_can` | deferred |

## Reuse Audit

- Local mouse: 250/250 frozen PNGs are byte-identical.
- Full-frame mouse: 250/250 frozen PNGs are byte-identical.
- Full-frame cat and trash-can were deferred before submission.

## Artifacts

- New raw results: `results/commercial/resemble/claimforge_balanced250_real_local750_canonical_jpeg_q95_20260727.jsonl`
- New run manifest: `results/commercial/resemble/claimforge_balanced250_real_local750_canonical_jpeg_q95_20260727.run_manifest.json`
- Formal summary: `results/commercial/resemble/claimforge_balanced250_main_table_20260727.summary.json`
- Reused local mouse results: `results/commercial/resemble/good275_mouse_forged_canonical_jpeg_q95_20260720.jsonl`
- Reused full-frame mouse results: `results/commercial/resemble/claimforge_v1_full_image_mouse250_canonical_jpeg_q95_20260725.jsonl`
