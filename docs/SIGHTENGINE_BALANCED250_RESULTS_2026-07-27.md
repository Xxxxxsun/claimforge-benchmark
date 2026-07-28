# Sightengine genai: CLAIMFORGE Balanced250

- Run completed: `2026-07-27T21:30:08.150519+00:00`
- Decision rule: `type.ai_generated >= 0.5`.
- Upload policy: exact benchmark raw JPEG/PNG files without resizing or re-encoding.

## Coverage

- New submissions: 822/822 valid, 0 final errors.
- Raw API attempts: 823; 1 failed attempt was superseded by a successful retry.
- New operations consumed: 4110.
- Reused results: 428/428 valid (178 local mouse plus 250 full-frame mouse).
- Evaluated primary cells: 1250/1750 images.
- Deferred before submission: 500 full-frame cat/trash-can images.

## Image-Level Results

Real false positives: 0/250 (0.0%); specificity 100.0%.

| Condition | TP / 250 | TPR | Specificity | Balanced acc. | 95% bootstrap CI |
|---|---:|---:|---:|---:|---:|
| `local_mouse` | 0 / 250 | 0.0% | 100.0% | 50.0% | 50.0%-50.0% |
| `local_cat` | 1 / 250 | 0.4% | 100.0% | 50.2% | 50.0%-50.6% |
| `local_trash_can` | 2 / 250 | 0.8% | 100.0% | 50.4% | 50.0%-51.0% |
| `fullframe_mouse` | 237 / 250 | 94.8% | 100.0% | 97.4% | 96.0%-98.6% |
| `fullframe_cat` | deferred | deferred | deferred | deferred | deferred |
| `fullframe_trash_can` | deferred | deferred | deferred | deferred | deferred |

## Reuse Audit

- Local mouse: 178 prior + 72 new; 250/250 raw hashes match.
- Full-frame mouse: 250/250 raw hashes match.
- Full-frame cat and trash-can were deferred before submission.

## Artifacts

- New raw results: `results/commercial/sightengine/claimforge_balanced250_core822_original_files_20260727.jsonl`
- New run manifest: `results/commercial/sightengine/claimforge_balanced250_core822_original_files_20260727.run_manifest.json`
- Formal summary: `results/commercial/sightengine/claimforge_balanced250_main_table_20260727.summary.json`
- Prior local-mouse results: `results/commercial/sightengine/pilot_good275_mouse_forged_original_png_20260720.jsonl`
- Full-frame mouse results: `results/commercial/sightengine/claimforge_v1_full_image_mouse250_original_png_20260725.jsonl`
