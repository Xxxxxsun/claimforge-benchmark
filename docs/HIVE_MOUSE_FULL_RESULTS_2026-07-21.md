# Hive V3 good-mouse forged-only full results (2026-07-21)

## 1. Scope and protocol

This run completes all 275 reviewed `good` mouse forgeries with Hive V3 model `hive/ai-generated-and-deepfake-content-detection`.

- Endpoint: `POST /api/v3/hive/ai-generated-and-deepfake-content-detection`.
- Input: every forged image is decoded to RGB and re-encoded as metadata-free JPEG quality 95 with 4:4:4 subsampling.
- Upload filename: neutral `image.jpg`.
- Provider threshold: `ai_generated >= 0.9`.
- Runner: `eval/commercial/run_hive.py`, with append-only JSONL and success-ID resume.

The run crossed multiple self-serve quota windows and API keys. Quota failures were retained as historical JSONL rows, while the latest row for each image ID is authoritative.

## 2. Full forged-only result

| Metric | Result |
|---|---:|
| Expected images | 275 |
| Unique latest results | 275 |
| Valid HTTP 200 results | 275/275 |
| Current errors | 0 |
| Detected at 0.9 | 0/275 (0%) |
| Mean AI score | 0.0007397 |
| Median AI score | 0.0001733 |
| P95 AI score | 0.0027711 |
| Minimum / maximum | 0.0000008 / 0.0337280 |

The maximum-scoring input is `restaurant_004_slot_002__forged` at 0.0337280, still more than an order of magnitude below the vendor threshold. The next four highest scores are also below 0.01.

By domain:

| Domain | N | Mean | Median | Min | Max | Detected |
|---|---:|---:|---:|---:|---:|---:|
| lodging | 147 | 0.0004401 | 0.0001284 | 0.0000008 | 0.0058085 | 0 |
| restaurant | 128 | 0.0010837 | 0.0002555 | 0.0000011 | 0.0337280 | 0 |

## 3. Interpretation limits

- This is a forged-only stress test. It establishes a 0/275 vendor-threshold hit rate for the local mouse-edit condition, not accuracy, AUROC, AP, specificity, or false-positive rate.
- Only the earlier five-pair pilot contains real controls; a full paired real run is still required for classification metrics.
- Hive provides whole-image scores and generator attribution, not a general edit-region mask, box, or heatmap.
- The extremely low scores should be reported as raw API outputs rather than calibrated probabilities.
- The raw JSONL has 278 physical rows: 275 authoritative successes plus three superseded historical HTTP 429 rows. The generated summary correctly resolves rows by image ID.

## 4. Reproducible artifacts

- JSONL: `results/commercial/hive/good275_mouse_forged_canonical_jpeg_q95_20260720.jsonl`
- Run manifest: same basename with `.run_manifest.json`
- Summary: same basename with `.summary.json`

SHA-256:

```text
JSONL        974d33038b2c27fc0216c7ae396af8036bc86ee18d29559f58f26b4f3ca116a7
run manifest 60551cef1a77996d775c6030b19b16a464a95acb834552d67717f76906e32ac2
summary      a34b10b6b95527db718670e40ae0ecb6e251c783445d7545453246ec2308feff
```

No API key, key ID, authorization header, or account credential is stored in these artifacts.
