# Complete, natural trash-can generation QA (2026-07-23)

## Goal

Generate trash cans that satisfy all of the following at the same time:

- the complete rim/lid, both side contours, and full base are visible;
- the silhouette is unobstructed and separated from every image edge;
- placement is physically sensible and unobtrusive, not on furniture or in the
  middle of a doorway, walkway, seating area, or work area;
- scale, perspective, contact shadow, lighting, color, depth of field,
  sharpness/blur, noise, and compression match the source;
- a soft, blurry, compressed, or otherwise non-photorealistic source receives a
  correspondingly soft and stylistically consistent bin rather than a uniformly
  sharp “realistic” insert.

## Input filtering

The 260 generic trash-can slots were reviewed before generation. The strict
manifest retains 112 source crops with enough plausible support geometry:

- tasks: `annotations/trash_can_generation_tasks_natural_112.jsonl`
- source audit: `docs/TRASH_CAN_SOURCE_SUITABILITY_2026-07-23.md`

The audit removes 91 directly unsuitable targets, 47 furniture-crowded
high-risk targets, and 10 borderline targets.

## Generation and review rounds

All requests completed successfully:

| Round | Directory | Requests | Selected |
|---|---|---:|---:|
| Initial | `hunyuan_image3_distil_trash_can_112_complete_natural_v2_20260723` | 112/112 | 100 |
| Repair 1 | `hunyuan_image3_distil_trash_can_repair1_complete_natural_v3_20260723` | 32/32 | 11 |
| Repair 2 | `hunyuan_image3_distil_trash_can_repair2_complete_natural_v4_20260723` | 10/10 | 1 |

The final reviewed directory is:

`generated_crops/hunyuan_image3_distil_trash_can_112_complete_natural_reviewed_20260723/`

It retains all 112 task IDs in the original task order and selects only repairs
that are clearly better under the complete-and-natural criteria. Raw initial and
repair outputs remain available for provenance.

## Final usability

- **85/112 usable**
  - restaurant: **29/41**
  - lodging: **56/71**
- **27/112 excluded**
- usable provenance:
  - 73 initial outputs
  - 11 repair-1 outputs
  - 1 repair-2 output

The 12 selected repairs are:

```text
trash_can_restaurant_002_slot_001
trash_can_lodging_033_slot_001
trash_can_lodging_042_slot_001
trash_can_lodging_050_slot_001
trash_can_lodging_093_slot_001
trash_can_lodging_116_slot_001
trash_can_lodging_135_slot_001
trash_can_lodging_150_slot_001
trash_can_lodging_232_slot_001
trash_can_lodging_259_slot_001
trash_can_lodging_276_slot_001
trash_can_lodging_286_slot_001
```

Per-task status, selected revision, final path, and the visual reason are stored
in:

`annotations/trash_can_complete_natural_review_20260723.jsonl`

## Exclusions

The 27 rejected tasks are deliberately retained in the generated artifact but
marked `exclude`; they must not be treated as usable examples.

No recognizable complete bin, or the only bin remains cropped/occluded:

```text
trash_can_restaurant_023_slot_001
trash_can_restaurant_095_slot_001
trash_can_restaurant_121_slot_001
trash_can_restaurant_138_slot_001
trash_can_restaurant_260_slot_001
trash_can_restaurant_285_slot_001
trash_can_lodging_031_slot_001
trash_can_lodging_078_slot_001
trash_can_lodging_155_slot_001
trash_can_lodging_157_slot_001
trash_can_lodging_162_slot_001
trash_can_lodging_246_slot_001
trash_can_lodging_291_slot_001
```

Unnatural support or intrusive placement:

```text
trash_can_restaurant_107_slot_001
trash_can_restaurant_171_slot_001
trash_can_restaurant_237_slot_001
trash_can_lodging_017_slot_001
trash_can_lodging_057_slot_001
trash_can_lodging_161_slot_001
trash_can_lodging_194_slot_001
```

The source already contains one or more bin-like objects:

```text
trash_can_restaurant_173_slot_001
trash_can_restaurant_216_slot_001
trash_can_restaurant_253_slot_001
trash_can_lodging_043_slot_001
trash_can_lodging_156_slot_001
trash_can_lodging_178_slot_001
trash_can_lodging_233_slot_001
```

## Validation

Independent visual reviewers compared source, initial, and repair variants.
Automated final validation completed with zero errors and zero warnings:

- tasks, review, manifest, and PNG sets all contain 112 unique IDs in identical
  order;
- review contains exactly 85 `usable` and 27 `exclude` rows, matching each final
  manifest `review_status`;
- all 112 outputs are decodable RGB PNGs with the expected dimensions;
- all 112 output hashes are unique;
- selected provenance is 100 initial, 11 repair 1, and 1 repair 2;
- every final image hash matches the recorded selected source revision;
- every prompt, seed, seed salt, size, and status matches the selected source
  manifest, and all deterministic seeds recompute correctly.
