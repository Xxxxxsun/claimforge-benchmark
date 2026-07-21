# Copyleaks Mouse Paired Pilot Results (2026-07-20)

## 1. Scope

This pilot validates the production Copyleaks AI Image Detection Ultra endpoint on two fixed CLAIMFORGE mouse pairs:

- `lodging_088_slot_001`
- `restaurant_089_slot_001`

Each pair contains the real source image and its reviewed pixel-preserving spliced image. All four requests used `sandbox=false` and model `ai-image-1-ultra`; 4/4 returned valid HTTP 200 responses. The integration smoke test consumed one additional credit on a duplicate forged image and is not included in the pilot statistics.

Official API documentation:

- https://docs.copyleaks.com/guides/ai-detector/ai-image-detection/
- https://docs.copyleaks.com/reference/actions/ai-image-detector/check
- https://docs.copyleaks.com/using-the-apis/authentication/

## 2. Input and localization protocol

- Both real and forged inputs are decoded, EXIF-transposed, converted to RGB, stripped of metadata, and encoded as PNG.
- Images below the API minimum are resized so both dimensions are at least 512 pixels; the two pilot pairs required no resize.
- Copyleaks returns a binary pixel mask as zero-based row-major RLE (`starts`, `lengths`). The raw RLE is preserved in each JSONL row.
- For each forged image, the canonical real and forged pixels are differenced channel-wise. A pixel belongs to the GT difference mask when any RGB channel has absolute difference greater than zero.
- The vendor RLE mask is compared directly with this exact SP difference mask. The annotated insert and context boxes are also retained as secondary overlap checks.

This exact-difference GT is valid here because CLAIMFORGE's SP construction preserves source pixels outside the pasted edit rather than regenerating the full image.

## 3. Results

| Task | Real verdict / AI fraction | Forged verdict / AI fraction | Pred. pixels | GT pixels | Precision | Recall | IoU |
|---|---:|---:|---:|---:|---:|---:|---:|
| lodging_088 | negative / 0 | positive / 0.0011510 | 2,797 | 2,697 | 0.9503 | 0.9855 | 0.9372 |
| restaurant_089 | negative / 0 | positive / 0.0009955 | 2,419 | 2,840 | 0.9831 | 0.8373 | 0.8254 |
| **Mean** | **0/2 positive** | **2/2 positive / 0.0010733** | - | - | **0.9667** | **0.9114** | **0.8813** |

Important interpretation:

- `summary.ai` is the fraction of pixels marked AI, not an ordinary whole-image confidence score. It numerically matches the decoded RLE area in both samples.
- Copyleaks returned `isAiDetected=true` even though only about 0.10% of each image was marked AI.
- All predicted pixels fell inside the annotated context box for both forged images.
- The pilot is only two pairs and cannot establish a population detection rate. It does establish that the production endpoint can detect and accurately localize at least some CLAIMFORGE SP edits, unlike the previously tested commercial heatmap output.

## 4. Credits and next action

Each successful image costs one credit. The account's five starter credits were exhausted by one integration probe plus the four pilot images; the post-run balance is zero.

The account was subsequently refilled with 100 credits and used for the next 100 previously untested forged images, producing the historical 102-image prefix in `COPYLEAKS_MOUSE_FORGED_102_RESULTS_2026-07-20.md`. A later refill completed all 275 forged images on 2026-07-21; current aggregate results are in `COPYLEAKS_MOUSE_FULL_RESULTS_2026-07-21.md`.

## 5. Reproducibility artifacts

- Runner: `eval/commercial/run_copyleaks.py`
- Pair 1: `results/commercial/copyleaks/preflight_good_mouse_pair1_canonical_png_20260720.jsonl`
- Pair 2: `results/commercial/copyleaks/preflight_good_mouse_pair2_canonical_png_20260720.jsonl`
- Each result has a matching `.run_manifest.json` and `.summary.json`.

Commercial results are versioned under `results/commercial/`. Credentials are read from `COPYLEAKS_EMAIL` and `COPYLEAKS_API_KEY`; neither credential nor the temporary access token is written to results or repository files.
