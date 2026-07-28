# Copyleaks Ultra: CLAIMFORGE Balanced250

- Run completed: `2026-07-27T19:33:57.830622+00:00`
- Model: `ai-image-1-ultra`, `sandbox=false`.
- Decision rule: vendor `isAiDetected=true`.
- Upload policy: metadata-free RGB PNG, resized only when required by the vendor's image-size contract.
- Primary comparison: the same independent real250 panel versus each 250-image forged condition.

## Coverage

- New calls: 750/750 valid, 0 final errors.
- Reused mouse calls: 500/500 valid.
- Evaluated primary cells: 1250/1750 images.
- Deferred before submission: 500 full-frame cat/trash-can images.
- New-call credits reported by Copyleaks: 750.

## Image-Level Results

Real false positives: 0/250 (0.0%); specificity 100.0%.

| Condition | TP / 250 | TPR | Specificity | Balanced acc. | 95% bootstrap CI |
|---|---:|---:|---:|---:|---:|
| `local_mouse` | 99 / 250 | 39.6% | 100.0% | 69.8% | 66.8%-72.8% |
| `local_cat` | 208 / 250 | 83.2% | 100.0% | 91.6% | 89.2%-93.8% |
| `local_trash_can` | 225 / 250 | 90.0% | 100.0% | 95.0% | 93.0%-96.8% |
| `fullframe_mouse` | 220 / 250 | 88.0% | 100.0% | 94.0% | 92.0%-96.0% |
| `fullframe_cat` | deferred | deferred | deferred | deferred | deferred |
| `fullframe_trash_can` | deferred | deferred | deferred | deferred | deferred |

## Native Localization

Copyleaks returns a native RLE mask. Local-splice rows are compared against the frozen exact-difference mask; full-frame rows have no localization target.

| Condition | Evaluated | Positive masks | Any GT overlap | Mean IoU, all | Mean IoU, detected |
|---|---:|---:|---:|---:|---:|
| `local_mouse` | 250 | 99 | 99 | 32.7% | 82.5% |
| `local_cat` | 250 | 208 | 208 | 59.9% | 72.0% |
| `local_trash_can` | 250 | 225 | 225 | 32.3% | 35.9% |

## Reuse Audit

- Local mouse: 250/250 frozen PNGs are byte-identical.
- Full-frame mouse: 250/250 frozen PNGs are byte-identical.
- Full-frame cat and trash-can were deferred at the user's request before any Copyleaks request was submitted.
- Copyleaks uses its own canonical PNG upload policy, so frozen Q95 JPEG hashes are not treated as expected upload hashes.

## Artifacts

- Missing-cell raw results: `results/commercial/copyleaks/claimforge_balanced250_missing1250_canonical_png_20260727.jsonl`
- Missing-cell run manifest: `results/commercial/copyleaks/claimforge_balanced250_missing1250_canonical_png_20260727.run_manifest.json`
- Formal summary: `results/commercial/copyleaks/claimforge_balanced250_main_table_20260727.summary.json`
- Reused local mouse results: `results/commercial/copyleaks/good275_mouse_forged_canonical_png_20260720.jsonl`
- Reused full-frame mouse results: `results/commercial/copyleaks/claimforge_v1_full_image_mouse250_canonical_png_20260725.jsonl`
