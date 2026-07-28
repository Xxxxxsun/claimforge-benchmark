# AI or Not: CLAIMFORGE Balanced250

- Run completed: `2026-07-27T18:34:14.621727+00:00`
- Endpoint/report: `/v2/image/sync`, `only=ai_generated`
- Decision rule: vendor `ai_detected=true`.
- Primary comparison: the same independent real250 panel versus each 250-image forged condition.

## Coverage

- New calls: 1250/1250 valid, 0 final errors.
- Reused mouse calls: 500/500 valid.
- Complete evaluation: 1750/1750 valid.
- Estimated new-call cost: USD 25.00.

## Raw Results

Real false positives: 3/250 (1.2%); specificity 98.8%.

| Condition | TP / 250 | TPR | Specificity | Balanced acc. | 95% bootstrap CI |
|---|---:|---:|---:|---:|---:|
| `local_mouse` | 4 / 250 | 1.6% | 98.8% | 50.2% | 49.2%-51.2% |
| `local_cat` | 2 / 250 | 0.8% | 98.8% | 49.8% | 49.0%-50.6% |
| `local_trash_can` | 3 / 250 | 1.2% | 98.8% | 50.0% | 49.0%-51.0% |
| `fullframe_mouse` | 230 / 250 | 92.0% | 98.8% | 95.4% | 93.6%-97.0% |
| `fullframe_cat` | 235 / 250 | 94.0% | 98.8% | 96.4% | 94.6%-98.0% |
| `fullframe_trash_can` | 231 / 250 | 92.4% | 98.8% | 95.6% | 93.8%-97.2% |

## Main Finding

AI or Not is almost perfectly specific on the shared authentic panel and detects the full-frame controls strongly, but its three local-splice sensitivities are near zero. Consequently, local balanced accuracy is approximately chance despite excellent full-frame balanced accuracy.

## Reuse Audit

- Local mouse: 250/250 frozen PNGs are byte-identical.
- Full-frame mouse: 250/250 frozen PNGs are byte-identical.
- Of 1702 uploads with a frozen canonical JPEG reference, 1685 are byte-identical and 17 were re-encoded by the current Pillow/libjpeg under the same Q95 policy.
- The raw new-results ledger has 1251 rows because one rejected old-key preflight was retained; the later success for that ID is authoritative.

## Artifacts

- Missing-cell raw results: `results/commercial/aiornot/claimforge_balanced250_missing1250_canonical_jpeg_q95_20260727.jsonl`
- Missing-cell run manifest: `results/commercial/aiornot/claimforge_balanced250_missing1250_canonical_jpeg_q95_20260727.run_manifest.json`
- Formal summary: `results/commercial/aiornot/claimforge_balanced250_main_table_20260727.summary.json`
- Reused local mouse results: `results/commercial/aiornot/good275_mouse_forged_canonical_jpeg_q95_20260720.jsonl`
- Reused full-frame mouse results: `results/commercial/aiornot/claimforge_v1_full_image_mouse250_canonical_jpeg_q95_20260725.jsonl`
