# Claude Zoom Agent oversized-image recovery

## Scope

The Claude Opus 4.8 Full1051 Zoom Agent run had 15 forged PNGs for which
the original gateway rejected the base64 image payload before the agent
could complete an episode. Both the detection and localization aggregates
were therefore incomplete for these images.

| Candidate | Images |
|---|---:|
| Cat | 4 |
| Mouse | 6 |
| Trash can | 5 |
| Total | 15 |

The affected image IDs were:

```text
local_cat__lodging_154_slot_001
local_cat__lodging_162_slot_001
local_cat__restaurant_051_slot_001
local_cat__restaurant_052_slot_001
local_mouse__lodging_154_slot_001
local_mouse__lodging_162_slot_001
local_mouse__restaurant_051_slot_001
local_mouse__restaurant_052_slot_001
local_mouse__restaurant_119_slot_001
local_mouse__restaurant_253_slot_001
local_trash_can__lodging_154_slot_001
local_trash_can__lodging_162_slot_001
local_trash_can__restaurant_051_slot_001
local_trash_can__restaurant_052_slot_001
local_trash_can__restaurant_253_slot_001
```

## Recovery run

- Model: `claude-opus-4-8`
- Transport: Anthropic-native `/v1/messages`, base64 image source
- Agent protocol: `mllm_zoom_agent_v2_bboxpx_20260728`
- Maximum zoom calls: 5 per episode
- Replicates: 3 per image
- Raw episodes: 45/45 successful
- Aggregate units: 30/30 valid
- Size, HTTP, and schema failures: 0
- Zoom calls: 180 total, 4.0 mean per episode

The 15 detection aggregates contain 3 `edited` and 12 `not_edited`
decisions. The corresponding localization aggregates contain 3
`localized_edit` and 12 `no_localized_edit` decisions.

## Merged coverage

The supplement replaces only the 30 incomplete
`(id, protocol_key)` units. The existing 2,072 valid aggregate rows are
preserved byte-for-byte. The historical aggregate also contained 20 stale
duplicate rows from recovery waves, so the merge removed all 50 invalid
rows and inserted the 30 valid supplement rows.

The merged Full1051 aggregate now contains exactly 2,102 unique units:

| Scope | Valid | Expected |
|---|---:|---:|
| Detection | 1,051 | 1,051 |
| Localization | 1,051 | 1,051 |
| Total | 2,102 | 2,102 |

Every merged row is `status=ok`, `valid_for_metrics=true`, and has three
successful replicates.

## Result artifacts

- `results/mllm/claude_opus_4_8/agent_zoom/claudeopus48_zoom_oversize15_anthropic_native_bboxpx_v2_20260728.raw.jsonl`
- `results/mllm/claude_opus_4_8/agent_zoom/claudeopus48_zoom_oversize15_anthropic_native_bboxpx_v2_20260728.jsonl`
- `results/mllm/claude_opus_4_8/agent_zoom/claudeopus48_zoom_oversize15_anthropic_native_bboxpx_v2_20260728.run_manifest.json`
- `results/mllm/claude_opus_4_8/agent_zoom/claudeopus48_zoom_oversize15_anthropic_native_bboxpx_v2_20260728.agent_metrics.json`
- `results/mllm/claude_opus_4_8/agent_zoom/claudeopus48_zoom_oversize15_anthropic_native_bboxpx_v2_20260728.agent_metrics.csv`
- `results/mllm/claude_opus_4_8/agent_zoom/claudeopus48_zoom_full1051_c15_z5_bboxpx_v2_20260728.jsonl`

The temporary Anthropic transport adapter used for this recovery is not
part of the repository changes.
