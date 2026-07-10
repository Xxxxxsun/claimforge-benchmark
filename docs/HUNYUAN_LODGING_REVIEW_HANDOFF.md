# ClaimForge Generation Review Handoff

This repo contains a small local review tool for manually checking the generated restaurant and lodging splice results.

## What To Open

From the repository root:

```bash
python3 -m http.server 8000 --directory .
```

Then open:

```text
http://localhost:8000/tools/hunyuan-lodging-review.html
```

Use HTTP, not `file://`. The page uses canvas to crop the displayed image; opening the HTML directly from disk can break canvas operations.

## What The Tool Reviews

The page reads and merges these splice manifests:

```text
new_test/spliced_full/manifest.jsonl
spliced_full/hunyuan_image3/manifest.jsonl
```

It filters to `restaurant_*` and `lodging_*` rows with `status: "ok"`.

At the time this document was written, that is 594 review items:

- 297 restaurant items from `new_test/spliced_full/manifest.jsonl`
- 297 lodging items from `spliced_full/hunyuan_image3/manifest.jsonl`

Important: do not use `annotations/generation_tasks.jsonl` as the coordinate source for this review page. Some generated splice outputs come from manifests whose image sizes and coordinates differ from the later annotation file. The splice manifests are the coordinate sources that match the PNGs currently being reviewed.

## What Is Shown

Main panel:

- The full spliced image from the active manifest row's `spliced_full` path.
- A small "在这里" arrow pointing to the upper corner of the original context window, not the object center. This avoids covering the edited area.

Right panel:

- `生成 crop`: the model-generated context crop from the active manifest row's `generated_crop` path.
- `拼回 crop（蓝框区域）`: the actual context-region crop cut from the final full spliced image using `context_region_xyxy`.
- Notes and current count statistics.

## Labels

Use three labels:

- `好`: the generated/spliced result is acceptable for the benchmark.
- `不好`: the generation or splice is visibly bad.
- `图本质不行`: the source image or selected context is inherently unsuitable, independent of model quality.

Keyboard shortcuts:

- `1`: 好
- `2`: 不好
- `3`: 图本质不行
- `Left / Right`: previous / next image
- `E`: export JSON

## Storage And Export

Labels are saved in browser `localStorage` under:

```text
claimforge-generation-review-v1
```

The page also imports existing labels from the older lodging-only key `claimforge-hunyuan-lodging-review-v1`, if present, so already-started lodging labels are not lost.

Click `导出 JSON` to download:

```text
claimforge_generation_review_labels.json
```

The exported JSON contains one record per task with:

- `task_id`
- `review_manifest`
- review `status`
- optional `note`
- `spliced_image`
- `generated_crop`
- `source_image`
- `image_size`
- `edit_region_xyxy`
- `context_region_xyxy`
- `candidates`

## Quick Validation

Basic local checks:

```bash
node --check <(awk '/<script>/{flag=1;next}/<\/script>/{flag=0}flag' tools/hunyuan-lodging-review.html)
curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:8000/tools/hunyuan-lodging-review.html
```

Expected HTTP status:

```text
200
```

Count check:

```bash
node - <<'NODE'
const fs = require('fs');
const paths = ['new_test/spliced_full/manifest.jsonl', 'spliced_full/hunyuan_image3/manifest.jsonl'];
let rows = [];
for (const path of paths) rows = rows.concat(fs.readFileSync(path, 'utf8').trim().split(/\n+/).map(JSON.parse));
const usable = rows.filter((row) => /^(restaurant|lodging)_/.test(row.task_id) && row.status === 'ok');
console.log(usable.length);
NODE
```

Expected count:

```text
594
```
