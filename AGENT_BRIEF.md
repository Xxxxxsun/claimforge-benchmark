# CLAIMFORGE Agent Brief

## What This Repo Is

This repository is a small generation handoff for the CLAIMFORGE pilot benchmark.

The research question is whether pixel-preserving local AI edits can evade image manipulation / AI-generated-image detectors in realistic consumer claim settings. For this pilot, the domain is lodging / restaurant complaint-style imagery: bathrooms, hotel rooms, restaurant interiors, and kitchens.

Your job as the remote generation agent is not to evaluate detectors. Your job is to use the deployed image-editing model to produce realistic localized edits from the provided crops and coordinates.

## Key Idea

Each task starts from a real source image. A human marked two boxes:

- `insert_box`: the small target area where the synthetic object should appear.
- `crop_box`: a larger context region around the target area. This is the crop that should be sent to the image-editing model.

The desired output is an edited crop where only the requested object is added inside the insert region, while the rest of the crop remains as close as possible to the input crop. The main benchmark pipeline will later splice the edited crop back into the original source image.

## Files To Use

Use this file as the task list:

```text
annotations/generation_tasks.jsonl
```

Each line is one JSON task. Important fields:

- `task_id`: stable ID for naming outputs.
- `source_image`: full source image path, relative to repo root.
- `context_crop`: crop image path to send to the edit model.
- `insert_crop`: tiny crop of the insert area, only for inspection.
- `insert_mask`: full-resolution source-image mask for the insert box.
- `candidates`: object to insert, for example `mouse` or `cockroach`.
- `insert_box`: target box in original source-image pixels.
- `crop_box`: context crop box in original source-image pixels.
- `edit_region_in_context_xyxy`: target box translated into context-crop coordinates.

The easiest path is:

1. Read each JSON object from `annotations/generation_tasks.jsonl`.
2. Load `context_crop`.
3. Ask the image-edit model to add `candidates` inside `edit_region_in_context_xyxy`.
4. Save the edited crop using the same `task_id`.

## Output Contract

Please create outputs under:

```text
generated_crops/<model_name>/
```

Use one image per input task:

```text
generated_crops/<model_name>/<task_id>.png
```

Also write a manifest:

```text
generated_crops/<model_name>/manifest.jsonl
```

Each manifest row should include:

```json
{
  "task_id": "lodging_000_slot_001",
  "input_context_crop": "crops/context/lodging_000_slot_001_context.jpg",
  "output_crop": "generated_crops/<model_name>/lodging_000_slot_001.png",
  "model": "<model_name>",
  "prompt": "<actual prompt used>",
  "status": "ok"
}
```

If a task fails, still write a row with `status: "failed"` and an `error` string.

## Editing Requirements

Keep the edit simple and localized:

- Add the requested object only inside the insert region.
- Preserve image geometry, lighting, background, and camera perspective.
- Do not change the whole room, table, sink, bed, or kitchen.
- Do not crop, resize, rotate, or otherwise change the context-crop dimensions.
- Return an edited crop with exactly the same width and height as the input `context_crop`.

The object does not need to be dramatic. The pilot only needs plausible local edits for detector testing.

## Prompt Template

A reasonable default prompt is:

```text
Add a small realistic <candidates> inside the marked target area. Keep the rest of the image unchanged. Preserve lighting, shadows, texture, camera perspective, and JPEG-like realism. Do not alter unrelated objects or the background.
```

If your model accepts a mask, use the insert region translated to crop coordinates (`edit_region_in_context_xyxy`) to build a crop-sized mask. If it does not accept a mask, include the coordinates in the prompt or use your model's preferred regional editing interface.

## Coordinate Notes

All boxes in `insert_box`, `crop_box`, `edit_region_xyxy`, and `context_region_xyxy` are in original source-image pixel coordinates.

For crop-local editing, use:

```text
edit_region_in_context_xyxy = [x1, y1, x2, y2]
```

This is already translated into the coordinate frame of `context_crop`.

## Quality Check

Before pushing results back:

- Verify there are 22 output crops.
- Verify each output crop has the same dimensions as its input context crop.
- Spot-check `overlays/` to understand where the blue context box and orange insert box were placed.
- Prefer subtle plausible edits over visually obvious artifacts.

## Git Workflow

After generation:

```bash
git add generated_crops/<model_name>/
git commit -m "Add <model_name> generated crops"
git push
```

The local benchmark side can then `git pull` and run the spliced-back composition / detector tests.
