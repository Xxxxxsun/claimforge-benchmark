# MLLM full-image orange-box detection ablation

## Scope

- Full-image control: 807 generated complete-frame outputs.
- Mouse: 275 images.
- Cat: 272 images.
- Trash can: 260 images.
- Protocol: detection v3 (`mllm_protocol_v3_reasoning_image_coordinates`).
- Models: Qwen 3.7 Plus, GPT-5.6 Luna, and Claude Opus 4.8.
- Aggregation: three independent valid records per image, with majority voting for `edited` versus `not_edited`.

The generator received the complete real source frame with an orange placement
box, and the complete returned frame was retained without splicing source
pixels back. Thus the whole output frame has passed through the generator.
These are fixed-seed, single-shot outputs from the frozen task lists described
in `FULL_IMAGE_ORANGE_BOX_CONTROL_2026-07-24.md`.

All 807 inputs are positive examples for this ablation. The numbers below are
therefore `edited` decision rates (positive recall on this set), not full
classification accuracy, precision, specificity, or AUROC.

## Detection results

| Model | Mouse | Cat | Trash can | Overall |
|---|---:|---:|---:|---:|
| Qwen 3.7 Plus | 137 / 275 (49.82%) | 37 / 272 (13.60%) | 11 / 260 (4.23%) | 185 / 807 (22.92%) |
| GPT-5.6 Luna | 175 / 275 (63.64%) | 75 / 272 (27.57%) | 17 / 260 (6.54%) | 267 / 807 (33.09%) |
| Claude Opus 4.8 | 253 / 275 (92.00%) | 216 / 272 (79.41%) | 79 / 260 (30.38%) | 548 / 807 (67.91%) |

The binary rate uses only the majority-voted `decision`. `p_ai_edited` is
aggregated separately as the median of the three replies and is not thresholded
again when computing this table.

## Completeness and fallback handling

All three final aggregate files contain exactly 807 unique, metrics-valid rows.

GPT repeatedly returned an empty response for three replicates after all
configured retries:

- `cat_lodging_173_slot_001`, replicate 3.
- `cat_lodging_180_slot_001`, replicates 1 and 3.

Per the user-directed failure policy, these three records were explicitly
imputed as `not_edited` with `p_ai_edited=0`. Both the raw rows and final
aggregate rows retain structured `persistent_empty_response_default` fallback
metadata. They are not represented as ordinary model responses.

The first GPT and Claude attempt used the shared default provider quota, which
was exhausted during the run. The unfinished replicates were resumed with the
verified `llm_application_intent` quota while preserving the original run ID,
inputs, prompts, model IDs, and concurrency.

## Interpretation notes

- The 42 trash-can outputs that failed the separate single-shot semantic QA
  were retained. Removing or replacing them would change the frozen
  single-shot experiment into a selected best-case subset.
- Localization is not evaluated here because complete-frame generation does
  not provide a meaningful local manipulation ground-truth region.
- Claude has the highest detection rate in every class. Mouse is the easiest
  class for Qwen, GPT, and Claude; trash can is the hardest.
- Compared with locally spliced images, complete-frame outputs generally
  expose more global reconstruction evidence. The gap is especially large for
  Claude on trash cans, but generation method, object class, and semantic
  success must be considered when interpreting the difference.

## Result files

- `results/mllm/qwen3_7_plus/fullai_orangebox_all807_detectionv3_20260725.jsonl`
- `results/mllm/gpt/fullai_orangebox_all807_detectionv3_20260725.jsonl`
- `results/mllm/claude_opus_4_8/fullai_orangebox_all807_detectionv3_20260725.jsonl`

Each model directory also contains the corresponding `.raw.jsonl`,
`.run_manifest.json`, and `.log` files.
