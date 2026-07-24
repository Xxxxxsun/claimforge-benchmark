# Full 260 trash-can generation (2026-07-24)

## Outcome

The 148 tasks omitted by the earlier 112-task source-suitability selection were
all sent through HunyuanImage generation. None is removed from the new
delivery.

The primary outputs are:

- remaining 148:
  `generated_crops/hunyuan_image3_distil_trash_can_remaining_148_reviewed_20260724/`
- complete 260:
  `generated_crops/hunyuan_image3_distil_trash_can_260_complete_reviewed_20260724/`

Each directory contains exactly one PNG per task plus a same-order
`manifest.jsonl`. The 260-task directory follows the exact order of
`annotations/trash_can_generation_tasks.jsonl`.

## Retention and QA policy

Visual QA is a label, not a deletion rule:

- `usable`: passed strict manual QA for a complete, naturally placed trash can
  whose blur, sharpness, lighting, perspective, noise, and compression match
  the source;
- `needs_review`: a generated candidate is retained, but no retry passed that
  strict threshold.

There are no `exclude` rows in the new 148-task or complete 260-task delivery.
The older 27 `exclude` labels from the reviewed 112 set are converted to
`needs_review` in the combined delivery, without changing or removing their
images.

| Cohort | Images retained | Strict `usable` | `needs_review` |
|---|---:|---:|---:|
| Previously reviewed 112 | 112 | 85 | 27 |
| Newly generated remaining 148 | 148 | 94 | 54 |
| Complete delivery | 260 | 179 | 81 |

The new 148 strict passes comprise 46 restaurant and 48 lodging tasks. The
remaining retained candidates comprise 31 restaurant and 23 lodging tasks.

## Generation rounds

The first complement run generated all 148 tasks successfully. Difficult cases
then received wider source context, per-image placement prompts, and additional
seeds.

| Round | Requested | Successful outputs | Purpose |
|---|---:|---:|---|
| v5 | 148 | 148 | full omitted-task complement |
| v6 | 89 | 89 | expanded-context retry |
| v7 | 32 | 32 | positioned lodging retry |
| v8 pilot | 4 | 4 | `think_recaption` pilot |
| v8a | 49 | 49 | positioned `think_recaption` retry |
| v8b | 17 | 17 | remaining positioned retry |
| Total | 339 | 339 | raw generation attempts |

All round manifests report `status=ok`; every output is a decodable RGB PNG
with the requested dimensions. The v8 rounds use `bot_task=think_recaption`,
matching the Distil checkpoint's own `generation_config.json`. The client now
uses that checkpoint-compatible mode by default.

## Reproducibility files

- exact 148-task complement:
  `annotations/trash_can_generation_tasks_remaining_148.jsonl`
- reviewed 148 mixed-context tasks:
  `annotations/trash_can_generation_tasks_remaining_148_reviewed_mixed_context_20260724.jsonl`
- reviewed 148 QA:
  `annotations/trash_can_remaining_148_review_20260724.jsonl`
- complete 260 mixed-context tasks:
  `annotations/trash_can_generation_tasks_260_reviewed_mixed_context_20260724.jsonl`
- complete 260 QA:
  `annotations/trash_can_full_260_review_20260724.jsonl`
- deterministic delivery builder:
  `scripts/build_trash_can_full_delivery.py`

The mixed-context manifests are required because successful repair candidates
may use an expanded context crop. Their context and edit coordinates match the
selected output dimensions.

## Validation

Independent validation confirmed:

- the prior 112 and remaining 148 ID sets are disjoint and their union is the
  original 260 tasks;
- task, review, manifest, and PNG sets contain 148/148 and 260/260 unique IDs in
  identical order;
- all 148 and all 260 PNGs decode as RGB and match their selected task
  dimensions;
- all 148 hashes are unique and all 260 hashes are unique;
- every copied file hash matches its selected source candidate;
- no task has an `exclude` status in either new delivery.
