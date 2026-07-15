# MLLM protocol v1

The frozen prompts are implemented in `eval/mllm/prompts.py`. The authoritative
review document is `docs/MLLM_DETECTION_PROTOCOL_PLAN_2026-07-11.md`.

- `mllm_detection_v1`: image-level `edited` / `not_edited` / `inconclusive`
  and a 0--100 probability.
- `mllm_localization_v1`: independent image-level likelihood plus up to three
  normalized `[x1, y1, x2, y2]` boxes on a 1000 by 1000 canvas.

Any wording change requires a new protocol version and a separate result run.
