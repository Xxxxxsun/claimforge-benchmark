# Hive V3: CLAIMFORGE Balanced250

- Run completed: `2026-07-28T09:05:09.702635+00:00`
- Decision rule: `ai_generated >= 0.9`.
- Upload policy: metadata-free RGB JPEG Q95 with 4:4:4 subsampling.

## Coverage

- New panel: 750/750 valid, 0 final errors.
- Raw API attempts: 814; 64 superseded attempt rows.
- Canonical upload hashes: 750/750 match.
- Reused mouse calls: 500/500 valid.
- Evaluated primary cells: 1250/1750 images.
- Deferred before submission: 500 full-frame cat/trash-can images.

## Image-Level Results

Real false positives: 0/250 (0.0%); specificity 100.0%.

| Condition | TP / 250 | TPR | Specificity | Balanced acc. | 95% bootstrap CI |
|---|---:|---:|---:|---:|---:|
| `local_mouse` | 0 / 250 | 0.0% | 100.0% | 50.0% | 50.0%-50.0% |
| `local_cat` | 1 / 250 | 0.4% | 100.0% | 50.2% | 50.0%-50.6% |
| `local_trash_can` | 0 / 250 | 0.0% | 100.0% | 50.0% | 50.0%-50.0% |
| `fullframe_mouse` | 107 / 250 | 42.8% | 100.0% | 71.4% | 68.4%-74.4% |
| `fullframe_cat` | deferred | deferred | deferred | deferred | deferred |
| `fullframe_trash_can` | deferred | deferred | deferred | deferred | deferred |

## Reuse Audit

- Local mouse: 250/250 frozen PNGs are byte-identical.
- Full-frame mouse: 250/250 frozen PNGs are byte-identical.
- Full-frame cat and trash-can were deferred before submission.

## Artifacts

- New raw results: `results/commercial/hive/claimforge_balanced250_real_local750_canonical_jpeg_q95_20260727.jsonl`
- New run manifest: `results/commercial/hive/claimforge_balanced250_real_local750_canonical_jpeg_q95_20260727.run_manifest.json`
- Formal summary: `results/commercial/hive/claimforge_balanced250_main_table_20260727.summary.json`
- Reused local mouse results: `results/commercial/hive/good275_mouse_forged_canonical_jpeg_q95_20260720.jsonl`
- Reused full-frame mouse results: `results/commercial/hive/claimforge_v1_full_image_mouse250_canonical_jpeg_q95_20260725.jsonl`
