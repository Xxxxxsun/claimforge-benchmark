# Claude Mouse oversized-image recovery

## Scope

The historical Claude Opus 4.8 Mouse-good275 run had six forged PNGs for
which both detection v3 and localization v3 had zero valid replies because
the original gateway rejected base64 image payloads larger than 5 MB.

Five of these images are in the fixed Benchmark1000 Mouse-250 subset. The
sixth, `restaurant_119_slot_001`, remains part of the Full1051 extension and
the historical Mouse-good275 collection.

| Image ID | PNG bytes | Base64 bytes | Benchmark1000 |
|---|---:|---:|---|
| `lodging_154_slot_001__forged` | 4,506,866 | 6,009,156 | yes |
| `lodging_162_slot_001__forged` | 4,224,656 | 5,632,876 | yes |
| `restaurant_051_slot_001__forged` | 4,627,477 | 6,169,972 | yes |
| `restaurant_052_slot_001__forged` | 4,343,263 | 5,791,020 | yes |
| `restaurant_119_slot_001__forged` | 4,032,941 | 5,377,256 | no |
| `restaurant_253_slot_001__forged` | 4,081,457 | 5,441,944 | yes |

## Recovery run

- Model: `claude-opus-4-8`
- Transport: Anthropic-native `/v1/messages`, base64 image source
- Protocols: detection v3 and localization v3 (`bbox_1000`)
- Replicates: three per image/protocol unit
- Raw replies: 36/36 valid
- Aggregate units: 12/12 valid
- Size, HTTP, and schema failures: 0

All six detection aggregates returned `not_edited`. All six localization
aggregates returned `no_localized_edit` with empty regions.

## Merged coverage

The supplement replaces only the 12 incomplete size-failure rows. The merged
Mouse-good275 collection has exactly 1,100 unique `(id, protocol_key)` rows:

| Scope | Detection | Localization |
|---|---:|---:|
| Forged Mouse-good275 | 275/275 | 274/275 |
| All forged + paired real rows | 550/550 | 549/550 |

The only remaining incomplete Mouse unit is localization for
`lodging_233_slot_001__forged`. It has two valid replicates and one persistent
`bbox_1000` schema failure; it is unrelated to request size.

## Result artifacts

- `results/mllm/claude_opus_4_8/mouse6_oversize_anthropic_native_v3_20260727.raw.jsonl`
- `results/mllm/claude_opus_4_8/mouse6_oversize_anthropic_native_v3_20260727.jsonl`
- `results/mllm/claude_opus_4_8/mouse6_oversize_anthropic_native_v3_20260727.run_manifest.json`
- `results/mllm/claude_opus_4_8/mouse_good275_total550_v3_20260727.jsonl`
