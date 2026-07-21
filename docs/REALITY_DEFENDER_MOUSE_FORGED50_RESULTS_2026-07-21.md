# Reality Defender good-mouse forged-50 results (2026-07-21)

## 1. Scope and protocol

This authenticated coverage pilot evaluates the first 50 reviewed `good` mouse forgeries in the fixed commercial-API ordering.

- API: Reality Defender production presigned-upload and polling workflow.
- Input: metadata-free RGB JPEG quality 95 with 4:4:4 subsampling.
- Selection: forged-only; 27 lodging and 23 restaurant images.
- Release version returned by the provider: `2.3.17`.
- Runner: `eval/commercial/run_reality_defender.py`.

The runner persists each upload `requestId` before polling. The JSONL therefore contains two rows per image—one submitted row and one terminal row—so an interruption can resume polling without uploading or charging the image again.

## 2. Coverage and verdicts

| Metric | Result |
|---|---:|
| Expected scans | 50 |
| Terminal results | 50/50 |
| Applicable results | 50/50 (100%) |
| API errors | 0 |
| `AUTHENTIC` | 50/50 |
| `FAKE` | 0/50 |
| `SUSPICIOUS` | 0/50 |
| Overall `NOT_APPLICABLE` / `UNABLE_TO_EVALUATE` | 0/50 |

The original concern that non-face restaurant/lodging scenes might be globally `NOT_APPLICABLE` did not materialize in this 50-image slice. Reality Defender returned an applicable overall image verdict for every input, but classified every local mouse edit as `AUTHENTIC`.

## 3. Scores

The runner normalizes the provider's 0–100 `resultsSummary.metadata.finalScore` to 0–1.

| Normalized score | Images |
|---:|---:|
| 0.01 | 47 |
| 0.02 | 1 |
| 0.03 | 2 |

Mean score is 0.011, median and P95 are 0.01, and the range is 0.01–0.03. Lodging has 27/27 `AUTHENTIC` results with mean score 0.01148; restaurant has 23/23 with mean 0.01043.

The provider response also includes model-level rows. Audio/video/text models correctly report `NOT_APPLICABLE`, and some image submodels can remain `ANALYZING` in a response whose overall `resultsSummary.status` is already terminal. Coverage and verdict counts in this report use the provider's overall terminal summary, not unrelated modality-level statuses.

## 4. Interpretation limits

- This is a forged-only prefix, not a paired classification experiment; it cannot estimate false-positive rate, AUROC, AP, or accuracy.
- The result demonstrates usable API coverage but complete threshold-level evasion on these 50 local edits.
- Reality Defender exposes whole-image ensemble scores and statuses, not a general edit-region mask or heatmap.
- The pilot submitted 50 scans. If the account is on the advertised free tier without paid or promotional scans, this consumes its 50-scan monthly allocation.
- A full paired evaluation would require real controls and additional quota; it should not silently reuse these forged-only results as a balanced test set.

## 5. Reproducible artifacts

- JSONL: `results/commercial/reality_defender/good_mouse_forged50_canonical_jpeg_q95_20260721.jsonl`
- Run manifest: same basename with `.run_manifest.json`
- Summary: same basename with `.summary.json`

SHA-256:

```text
JSONL        e7648aa3668286a7ac296787b72b6ae1dc25f13f906283bbc167cae69bc7229c
run manifest 30712d4b12c88451bc00667116e0c82db4b7a4b1a6dd42289ad73df5fc8551b6
summary      075f1522b4bea9c192887bc02ad24997593d745fa8878c741cf4d9fe51701ca6
```

No API key, authorization header, or presigned upload URL is stored in these artifacts.
