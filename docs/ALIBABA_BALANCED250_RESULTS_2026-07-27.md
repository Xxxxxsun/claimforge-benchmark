# Alibaba Cloud Ultra: CLAIMFORGE Balanced250

- Run completed: `2026-07-27T17:30:20.609115+00:00`
- Service: `aigcDetector_ultra` (`cn-beijing`)
- Decision rule: positive when any of `risk_aigc`, `risk_fake`, or `risk_edit` is returned.
- Primary comparison: the same independent real250 panel versus each 250-image forged condition.

## Coverage

- New calls: 1250/1250 valid, 0 errors.
- Reused mouse calls: 500/500 valid.
- Complete primary panel: 1750/1750 valid.
- Estimated new-call cost: CNY 25.00.

## Raw Results

Real false positives: 28/250 (11.2%); specificity 88.8%.

| Condition | TP / 250 | TPR | Specificity | Balanced acc. | 95% bootstrap CI |
|---|---:|---:|---:|---:|---:|
| `local_mouse` | 29 / 250 | 11.6% | 88.8% | 50.2% | 47.4%-53.0% |
| `local_cat` | 50 / 250 | 20.0% | 88.8% | 54.4% | 51.4%-57.6% |
| `local_trash_can` | 27 / 250 | 10.8% | 88.8% | 49.8% | 47.2%-52.6% |
| `fullframe_mouse` | 230 / 250 | 92.0% | 88.8% | 90.4% | 87.8%-93.0% |
| `fullframe_cat` | 244 / 250 | 97.6% | 88.8% | 93.2% | 91.0%-95.4% |
| `fullframe_trash_can` | 234 / 250 | 93.6% | 88.8% | 91.2% | 88.6%-93.6% |

## Main Finding

Alibaba detects the full-frame controls strongly but is near chance on the local-splice conditions once false positives on the shared real panel are included. The result supports reporting local and full-frame generation separately rather than pooling them.

## Reuse Audit

- Local mouse: 250/250 frozen PNGs are byte-identical.
- Full-frame mouse: 250/250 frozen PNGs are byte-identical.
- Across the 1702 uploads with a frozen canonical JPEG reference, 1685 are byte-identical to the frozen Q95 JPEG and 17 differ after re-encoding with the current Pillow/libjpeg under the same Q95 policy. Another 48 reused mouse inputs are outside the later canonical selection; their frozen benchmark PNG hashes were validated instead.

## Artifacts

- Missing-cell raw results: `results/commercial/alibaba/claimforge_balanced250_missing1250_canonical_jpeg_q95_20260727.jsonl`
- Missing-cell run manifest: `results/commercial/alibaba/claimforge_balanced250_missing1250_canonical_jpeg_q95_20260727.run_manifest.json`
- Formal summary: `results/commercial/alibaba/claimforge_balanced250_main_table_20260727.summary.json`
- Reused local mouse results: `results/commercial/alibaba/good275_mouse_forged_canonical_jpeg_q95_20260720.jsonl`
- Reused full-frame mouse results: `results/commercial/alibaba/claimforge_v1_full_image_mouse250_canonical_jpeg_q95_20260725.jsonl`
