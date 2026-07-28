# CLAIMFORGE v1 250x3x2

This directory is the frozen edited-image benchmark slice with:

- 3 object categories: mouse, cat, and trash can;
- 2 generation routes: local crop generation plus splice-back, and full-image
  orange-box conditional generation;
- 250 matched tasks per category and route.

The result contains 1500 edited PNGs arranged as:

```text
local_splice/<category>/*.png
full_image/<category>/*.png
```

`pairs.jsonl` contains the 750 matched task pairs.
`manifest.jsonl` contains one row per edited image. Each method/category
directory also has its own 250-row `manifest.jsonl`.

## Selection policy

For each category, the eligible local-splice quality-approved task set is
filtered through the frozen full-image task-list order. The first 250 task IDs
are retained, and both generation routes use exactly those same task IDs and
real source images. This is a deterministic prefix, not a random sample.

Mouse eligibility is `status=good` in
`claimforge_generation_review_labels.json`. Cat and trash-can eligibility comes
from the manually selected final splice manifests. The full-image side always
uses the fixed single-shot primary output; retries are not substituted.

## Full-image semantic QC

The selected trash-can full-image slice retains the existing manual QC labels:
212/250 usable and 38/250 failed. These
failures remain in the benchmark because the full-image control is a fixed
single-shot generation condition. Use `full_image_manual_qc` in the manifests
for stratified analysis; do not silently drop failures from headline metrics.

Cat and mouse full-image outputs have structural validation but no equivalent
complete manual semantic-QC file in this repository snapshot.

## Rebuild

From the repository root:

```bash
python3 scripts/build_final_benchmark.py --force
```
