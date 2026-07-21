# Copyleaks Mouse Forged-Only Full Results (2026-07-21)

## 1. Scope and protocol

This run completes the fixed set of 275 reviewed `good` mouse forgeries. The first two forged rows were reused from the paired production preflight, the next 100 were completed on 2026-07-20, and the remaining 173 were submitted on 2026-07-21. No successful ID was charged twice in the resumable full-run artifact.

- Provider model: Copyleaks `ai-image-1-ultra`, `sandbox=false`.
- Input: metadata-free RGB PNG; images below the API minimum were enlarged with the same Lanczos policy for source and forged inputs.
- Resized inputs: 26/275; unchanged-size inputs: 249/275.
- Provider output: image verdict, AI-pixel fraction, and native zero-based row-major RLE mask.
- Localization GT: exact channel-wise difference between the submitted canonical source and forged images, with threshold 0.

## 2. Coverage and image-level detection

| Metric | Result |
|---|---:|
| Expected forged images | 275 |
| Valid results | 275/275 |
| API errors | 0 |
| Actual credits recorded | 275 |
| Positive verdicts | 111/275 (40.36%) |
| Negative verdicts / empty masks | 164/275 (59.64%) |

By domain:

| Domain | Detected | Total | Detection rate |
|---|---:|---:|---:|
| lodging | 66 | 147 | 44.90% |
| restaurant | 45 | 128 | 35.16% |

The returned `summary.ai` value is the predicted AI-pixel fraction, not a calibrated image-level confidence score. Across all 275 images its mean is 0.0008104, median 0, P95 0.0039175, and maximum 0.0096509. Conditional on the 111 positive verdicts, its mean is 0.0020079, median 0.0013435, and range 0.0000909–0.0096509.

## 3. Native-mask localization

The RLE mask is compared directly with the exact pixel-difference GT; there is no fitted threshold or post-hoc mask selection.

### Conditional on a positive mask (111 images)

| Metric | Mean | Median | Min | Max | P95 |
|---|---:|---:|---:|---:|---:|
| Precision | 0.9739 | 0.9852 | 0.7914 | 1.0000 | 1.0000 |
| Recall | 0.8389 | 0.8741 | 0.1615 | 0.9996 | 0.9699 |
| IoU | 0.8165 | 0.8557 | 0.1615 | 0.9544 | 0.9353 |

All 111 positive masks overlap the exact-difference GT. For 110/111 masks, 100% of predicted pixels lie inside the annotated context box; the remaining mask has 96.30% inside.

### Including detection misses as empty masks (all 275 images)

| Metric | Mean | Median | Max | P95 |
|---|---:|---:|---:|---:|
| Recall | 0.3386 | 0 | 0.9996 | 0.9614 |
| IoU | 0.3296 | 0 | 0.9544 | 0.9226 |

The conditional and unconditional results answer different questions. Copyleaks localizes the inserted region accurately when it returns a positive mask, but 164 empty-mask misses remain image-level and localization failures. The paper must report both views together.

## 4. Integrity and interpretation limits

- The final resumable artifact contains 275 successful, unique forged IDs and no current error rows.
- Every authoritative row records HTTP 200, one actual credit, the raw RLE, canonical upload hash, scan ID, and localization measurements.
- The forged-only run cannot estimate false-positive rate, AUROC, AP, or paired accuracy. Only the earlier two-pair preflight contains real controls.
- The 40.36% detection rate is a provider-verdict rate for this fixed local-edit condition, not a general estimate of detector sensitivity.
- Credentials, account email, bearer tokens, and authorization headers are not persisted.

## 5. Reproducible artifacts

- Runner: `eval/commercial/run_copyleaks.py`
- JSONL: `results/commercial/copyleaks/good275_mouse_forged_canonical_png_20260720.jsonl`
- Run manifest: same basename with `.run_manifest.json`
- Summary: same basename with `.summary.json`

SHA-256:

```text
JSONL        1667820ab0248f55e53dc69c2862f6985ae844ab34b95bc691ecbcf5a0e72182
run manifest 8fedfbec560390c24ec51442d69eb4782f3dcb4179f598380976b63724767c09
summary      94d4311b8654ae95211f10516335fe89489528de2889086b19cd452ecd797d6c
```

The earlier `COPYLEAKS_MOUSE_FORGED_102_RESULTS_2026-07-20.md` remains as the historical prefix report; this document supersedes it for current forged-only aggregate numbers.
