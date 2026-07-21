# Copyleaks Mouse Forged-Only Results: First 102 Images (2026-07-20)

> Historical prefix snapshot. The 275-image run was completed on 2026-07-21; current aggregate results are in `COPYLEAKS_MOUSE_FULL_RESULTS_2026-07-21.md`.

## 1. Scope

This run evaluates the first 102 unique reviewed mouse forgeries in the fixed commercial-API ordering. It contains 54 lodging and 48 restaurant images. The two images from the earlier paired pilot were reused without another API call; the next 100 previously untested forged images were submitted with `sandbox=false` to Copyleaks model `ai-image-1-ultra`.

The current artifact is a resumable prefix of the planned 275-image forged-only run, not a completed full-dataset result.

## 2. Image-level detection

- Valid results: **102/102**.
- Copyleaks positive verdicts: **38/102 (37.25%)**.
- Misses: **64/102 (62.75%)**.
- Lodging: **21/54 (38.89%)** detected.
- Restaurant: **17/48 (35.42%)** detected.
- `summary.ai` over detected images: mean 0.002124, median 0.001143, range 0.000437-0.009651. These values are AI-pixel fractions, not calibrated image-level confidence scores.

The 100-image expansion initially produced 99 valid responses and one transient Cloudflare HTTP 520. The resumable runner retried only that failed ID; the retry succeeded, leaving no invalid samples.

## 3. Localization

Copyleaks returns a native binary RLE mask. Each mask was decoded and compared with the exact channel-wise pixel difference between the canonical source and forged image.

### Conditional on a positive Copyleaks verdict (38 images)

- Mean precision: **0.9804**.
- Mean recall: **0.8433**.
- Mean IoU: **0.8266**; median 0.8538; range 0.1615-0.9372.
- All 38 predicted masks placed 100% of their positive pixels inside the annotated context box.

### Including detection misses as empty masks (all 102 images)

- Mean recall: **0.3142**.
- Mean IoU: **0.3080**.
- Median IoU: **0**.

The distinction is essential: Copyleaks localizes very accurately when it fires, but it returns an empty mask for most current samples. The benchmark must therefore report unconditional T2 metrics alongside conditional localization quality.

Nine inputs were resized to satisfy the API's 512x512 minimum. Real and forged canonicalization uses the same RGB PNG and Lanczos policy, and localization is evaluated in the submitted image coordinate system.

## 4. Credits and continuation

- The refill contained 100 credits and produced 100 additional valid forged-image results.
- Current balance after the run: **0 credits**, confirmed through the balance endpoint.
- Completed unique forged images: **102/275**.
- Remaining unique forged images: **173**, requiring **173 additional credits** at the observed one-credit-per-image rate.

Resume command after adding credits:

```bash
COPYLEAKS_EMAIL='...' COPYLEAKS_API_KEY='...' PYTHONPATH=. .venv/bin/python \
  eval/commercial/run_copyleaks.py \
  --tasks 275 \
  --include forged \
  --output results/commercial/copyleaks/good275_mouse_forged_canonical_png_20260720.jsonl \
  --run-id copyleaks_good275_mouse_forged_20260720
```

The runner currently reports `already_valid=102` and `pending=173`; successful rows are not resubmitted.

## 5. Artifacts

- Runner: `eval/commercial/run_copyleaks.py`
- Resumable full-run JSONL: `results/commercial/copyleaks/good275_mouse_forged_canonical_png_20260720.jsonl`
- Run manifest: same basename with `.run_manifest.json`
- Partial summary: same basename with `.summary.json`

Commercial result artifacts are versioned under `results/commercial/`. Credentials and temporary bearer tokens are not persisted.
