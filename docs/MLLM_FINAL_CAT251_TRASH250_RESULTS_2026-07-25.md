# MLLM final cat251 + trash-can250 results

## Scope

- Final forged-image set: 501 images.
- Cat: 251 images.
- Trash can: 250 images.
- Protocol suite: detection v3 + localization v4 (`mllm_protocol_suite_20260724`).
- Replicates: three valid independent replies are required for each image/protocol unit.
- Models: Qwen 3.7 Plus, GPT-5.6 Luna, and Claude Opus 4.8.

The previous 431-image run covered 232 cat and 199 trash-can images. All 431 overlapping final images were verified byte-for-byte identical, so their results were reused. The remaining 70 images (19 cat and 51 trash-can) were evaluated separately and merged with the previous results.

All inputs in this report are forged images. Therefore, the rates below are positive-decision rates on forged inputs, not full classification accuracy, precision, or specificity.

## Completeness

| Model | Total units | Valid units | Incomplete units |
|---|---:|---:|---:|
| Qwen 3.7 Plus | 1,002 | 1,001 | 1 |
| GPT-5.6 Luna | 1,002 | 999 | 3 |
| Claude Opus 4.8 | 1,002 | 1,002 | 0 |

Claude initially had 18 incomplete units from nine images whose base64 payloads
exceeded the original provider's 5 MB limit. On 2026-07-27, those nine images
were recovered through an Anthropic-native supplement run. All 54 independent
replicates completed successfully, producing nine valid detection units and nine
valid localization units. The supplement results replace only the corresponding
incomplete rows in the combined collection.

## Detection: `edited` decisions

| Model | Cat | Trash can |
|---|---:|---:|
| Qwen 3.7 Plus | 15 / 251 (5.98%) | 5 / 250 (2.00%) |
| GPT-5.6 Luna | 39 / 249 (15.66%) | 4 / 250 (1.60%) |
| Claude Opus 4.8 | 173 / 251 (68.92%) | 14 / 250 (5.60%) |

For Claude, 172 of the 173 detected cat images explicitly mention a cat, kitten, feline, or tabby in all three independent reasoning records. The remaining detected cat image (`cat_restaurant_088_slot_001`) was flagged because of suspicious graffiti rather than the cat.

## Localization: `localized_edit` decisions

| Model | Cat | Trash can |
|---|---:|---:|
| Qwen 3.7 Plus | 16 / 250 (6.40%) | 2 / 250 (0.80%) |
| GPT-5.6 Luna | 30 / 250 (12.00%) | 4 / 250 (1.60%) |
| Claude Opus 4.8 | 187 / 251 (74.50%) | 16 / 250 (6.40%) |

This table measures whether the model returned `localized_edit`. It does not yet measure whether the predicted box overlaps the ground-truth mask or reaches an IoU threshold.

## Combined result files

- `results/mllm/qwen3_7_plus/final_cat251_trash250_total501_suite0724_20260725.jsonl`
- `results/mllm/gpt/final_cat251_trash250_total501_suite0724_20260725.jsonl`
- `results/mllm/claude_opus_4_8/final_cat251_trash250_total501_suite0724_20260725.jsonl`

Claude oversize-image supplement artifacts:

- `results/mllm/claude_opus_4_8/final501_oversize9_anthropic_native_suite0724_20260727.raw.jsonl`
- `results/mllm/claude_opus_4_8/final501_oversize9_anthropic_native_suite0724_20260727.jsonl`
- `results/mllm/claude_opus_4_8/final501_oversize9_anthropic_native_suite0724_20260727.run_manifest.json`

Each combined row retains its original `run_id` and adds:

- `collection_id`: the final 501-image aggregate collection.
- `source_run_id`: the source run from which the row was taken.
- `final_image_path`: the canonical path in the final cat251/trash-can250 set.
