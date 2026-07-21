# SAM 3 cat-mask splice pilot results (2026-07-21)

## Decision

Use fal's older `fal-ai/sam-3/image-rle` endpoint with a **text-only `cat`
prompt** as the primary hosted segmentation path. Do not send the text and edit
box prompts together in the same first request.

For this repository's generated cat context crops, the recommended pipeline is:

```text
generated context crop
  -> SAM 3 text-only cat mask
  -> local edit-box/difference candidate ranking and quality gate
  -> add only nearby source/generated residual components
  -> 1 px feather
  -> paste the resulting context crop into the source image
```

The complete 10-image text-only SAM 3 run had 10/10 successful API responses,
10/10 passing local quality gates, and 0/10 violations of the invariant that
pixels outside the context box remain byte-identical to the source. Manual
inspection of the saved contact sheet found a cat-shaped semantic mask in all
10 cases.

SAM 3.1 did not show a useful advantage in this pilot. After quality-gated
fallback selection, SAM 3 and SAM 3.1 had mean semantic-mask IoU 0.985 and mean
hybrid-mask IoU 0.991 over the same 10 images. At the prices displayed by fal
on 2026-07-21, SAM 3 costs USD 0.005/request and SAM 3.1 costs USD
0.01/request. The cheaper endpoint is therefore the better default for the
current cat splice pipeline.

This is a workflow decision, not a claim that SAM 3 is universally better than
SAM 3.1. The pilot is small and has no manually drawn pixel ground truth.

## What was run

The deterministic pilot selected 10 generated cat crops from
`hunyuan_image3_distil_cat_272`, balanced across five restaurant and five
lodging images. Within each domain it covered small, medium, and large edit
regions and prioritized cases where the existing threshold-30 and threshold-40
mechanical masks disagreed.

| Stage | Prompt mode | Endpoint(s) | Requests | Estimated cost | Purpose |
|---|---|---:|---:|---:|---|
| Primary A/B | `cat` text + edit box | SAM 3 and SAM 3.1 | 20 | USD 0.15 | Endpoint and prompt-behavior comparison |
| Targeted fallback | `cat` text only | SAM 3 and SAM 3.1 | 4 | USD 0.03 | Recover the two primary quality failures |
| Text-only completion | `cat` text only | SAM 3 | 8 | USD 0.04 | Complete a non-duplicated 10-image text-only run |
| **Total research calls** | | | **32** | **USD 0.22** | |

Every request used a base64 data URI and the platform headers
`X-Fal-Store-IO: 0` and
`X-Fal-Object-Lifecycle-Preference: {"expiration_duration_seconds":3600}`.
The adapter stores request IDs and RLE results locally, but never stores the
credential or input data URI.

## Quantitative results

### Prompt behavior

| Condition | Endpoint | API success | Quality-gate pass | Mean semantic area | Mean hybrid area | Mean hybrid growth | Outside-context failures |
|---|---|---:|---:|---:|---:|---:|---:|
| Text + box | SAM 3 | 10/10 | 8/10 | 0.079 | 0.092 | 0.353 | 0 |
| Text + box | SAM 3.1 | 10/10 | 8/10 | 0.079 | 0.094 | 0.310 | 0 |
| **Text only** | **SAM 3** | **10/10** | **10/10** | **0.095** | **0.106** | **0.114** | **0** |

The text-plus-box failures were the same two tasks for both endpoints:

- `cat_restaurant_291_slot_001`
- `cat_lodging_246_slot_001`

In one failure the returned mask was sparse and fragmented; in the other it
segmented a rectangular wall feature instead of the cat. Both text-only retries
returned full cat silhouettes and passed the quality gate on SAM 3 and SAM 3.1.

When both prompt types returned separate candidates, the provider's
highest-confidence candidate was sometimes the pre-existing table/chair object
inside the box, while the lower-confidence text candidate was the inserted cat.
This is why provider score alone is not a safe selector.

### Endpoint comparison after fallback

The quality-selected A/B output contains 16 primary text-plus-box results and
four text-only fallback results, covering both endpoints for all 10 tasks.

| Metric | Value |
|---|---:|
| Selected results | 20/20 |
| Passing quality gates | 20/20 |
| Unresolved results | 0 |
| Mean SAM 3 vs SAM 3.1 semantic-mask IoU | 0.985 |
| Mean SAM 3 vs SAM 3.1 hybrid-mask IoU | 0.991 |
| SAM 3 mean hybrid growth over semantic mask | 0.148 |
| SAM 3.1 mean hybrid growth over semantic mask | 0.151 |
| Outside-context invariance failures | 0 |

The complete SAM 3 text-only output and the quality-selected mixed-prompt
output have mean semantic-mask IoU 0.926 and hybrid-mask IoU 0.949. Both look
reasonable in the contact sheets, but text-only is operationally simpler,
passed all 10 gates without a second API call, and costs only USD 0.05 per 10
images.

At USD 0.005/request, a single-pass text-only run would cost approximately USD
1.36 for the current 272 cat images, or USD 2.97 for 594 images. Prices should
be rechecked before a large run.

## Important implementation findings

### 1. fal queue routes differ for endpoint subpaths

Submission uses the complete endpoint path:

```text
POST https://queue.fal.run/fal-ai/sam-3/image-rle
```

Status and result retrieval use the app root, without `/image-rle`:

```text
GET https://queue.fal.run/fal-ai/sam-3/requests/{request_id}/status
GET https://queue.fal.run/fal-ai/sam-3/requests/{request_id}
```

The first preflight exposed this distinction as HTTP 405. Its already-saved
request ID was resumed successfully after the route was corrected; no duplicate
request was submitted.

### 2. fal's simple start/length RLE is row-major

The observed RLE strings look like:

```text
8733 2 8946 14 9161 10 ...
```

They are one-based `(start, length)` pairs over a row-major flattened mask.
They are not COCO's alternating column-major counts. The adapter supports this
observed format as well as compressed and uncompressed COCO RLE, and unit tests
cover each decoder.

### 3. Candidate selection needs local evidence

The implemented rank score is:

```text
0.30 * provider confidence
+ 0.45 * fraction of semantic-mask pixels supported by source/generated diff
+ 0.10 * edit-box coverage
+ 0.05 * edit-box precision
+ 0.10 * plausible-area indicator
```

The edit box remains local evidence used after inference. It should not be sent
with the text prompt on the primary call for this single-cat crop setting.

### 4. The quality gate must be explicit

A selected semantic mask passes only when it:

- has a plausible nonzero area;
- intersects the expected edit box; and
- has at least 50% of its pixels supported by the source/generated crop
  difference.

Failed masks are marked and require a separately logged retry or review. The
pipeline never silently substitutes the whole edit box.

### 5. SAM is the semantic core; difference recovers local effects

The current hybrid post-processing uses:

- max-channel source/generated difference threshold: 20;
- residual connected components of at least three pixels;
- component must intersect a six-pixel dilation of the semantic mask;
- retained residual is clipped to a 12-pixel dilation of the semantic mask;
- final alpha feather: one pixel.

This adds nearby fur edges and contact shadow while excluding disconnected
background/color-shift components. In the full text-only SAM 3 run, the hybrid
mask was on average 11.4% larger than the semantic mask.

## Recommended production policy

1. Run `fal-ai/sam-3/image-rle` on the generated context crop with only
   `prompt: "cat"`.
2. Ask for up to three masks, scores, and boxes.
3. Decode RLE locally and rank candidates using residual support plus the known
   edit region.
4. Reject candidates that fail the explicit quality gate.
5. Form the hybrid mask from the semantic core plus only nearby residual
   components.
6. Composite inside the context crop and assert that every pixel outside the
   context box equals the source.
7. If the text-only call fails, make a separately logged point/box retry. Do not
   combine prompts by default and do not use an unmarked full-box fallback.

For the full 272-image cat batch, run sequentially or with conservative
concurrency, keep the append-only API JSONL, and resume saved request IDs after
interruptions. Review a contact sheet before promoting the output into the
benchmark.

## Reproduction

The runner reads the credential only from `FAL_KEY`:

```bash
export FAL_KEY='<fal-key>'
python -m eval.segmentation.run_fal_sam3 \
  --prompt-mode text_only \
  --endpoints sam3 \
  --tasks 10 \
  --output-dir results/segmentation/my_sam3_cat_pilot
unset FAL_KEY
```

Dry-run selection and cost estimation do not require a credential:

```bash
python -m eval.segmentation.run_fal_sam3 \
  --prompt-mode text_only --endpoints sam3 --tasks 10 --dry-run
```

Rerunning the same command resumes submitted request IDs and skips completed
rows. `--materialize-only` rebuilds masks and composites without network calls.

## Artifacts

Primary A/B and final selections:

- `results/segmentation/fal_sam3_cat_pilot10_20260721/run_manifest.json`
- `results/segmentation/fal_sam3_cat_pilot10_20260721/api_results.jsonl`
- `results/segmentation/fal_sam3_cat_pilot10_20260721/splice_results.jsonl`
- `results/segmentation/fal_sam3_cat_pilot10_20260721/summary.json`
- `results/segmentation/fal_sam3_cat_pilot10_20260721/selected_splice_results.jsonl`
- `results/segmentation/fal_sam3_cat_pilot10_20260721/selected_summary.json`
- `results/segmentation/fal_sam3_cat_pilot10_20260721/selected_contact_sheet.jpg`

Complete text-only SAM 3 view assembled without duplicate calls:

- `results/segmentation/fal_sam3_cat_pilot10_20260721/text_only_sam3_full10_results.jsonl`
- `results/segmentation/fal_sam3_cat_pilot10_20260721/text_only_sam3_full10_summary.json`
- `results/segmentation/fal_sam3_cat_pilot10_20260721/text_only_sam3_full10_contact_sheet.jpg`

Text-only shards:

- `results/segmentation/fal_sam3_cat_textonly_fallback2_20260721/`
- `results/segmentation/fal_sam3_cat_textonly_remaining8_20260721/`

Code and tests:

- `eval/segmentation/run_fal_sam3.py`
- `tests/test_run_fal_sam3.py`

## Limitations

- Only 10 generated cat crops were visually reviewed.
- No human-drawn pixel masks exist, so IoU against a true object boundary is
  unavailable. IoU values here compare methods/endpoints, not ground truth.
- The residual-based quality gate assumes the inserted object creates a strong
  source/generated difference. It may need tuning for very subtle stains,
  translucent edits, or large generative reconstruction drift.
- The observed row-major start/length RLE behavior is provider-specific and
  should remain covered by a preflight when the endpoint changes.
- Text-only behavior should be revalidated for other candidates such as mouse,
  cockroach, and stain.

Official API references:

- https://fal.ai/models/fal-ai/sam-3/image-rle/api
- https://fal.ai/models/fal-ai/sam-3-1/image-rle/api
- https://fal.ai/docs/documentation/model-apis/inference/queue
- https://fal.ai/docs/documentation/model-apis/media-expiration
