# Commercial API Result Index (2026-07-20)

This directory contains the raw JSONL responses, run manifests, generated summaries, and provider-returned artifacts for CLAIMFORGE commercial detector experiments.

## Current execution status

The forged-only target is the fixed set of 275 reviewed mouse edits. A standard paired pilot is five tasks (five real sources plus five corresponding forgeries). `Complete` below refers only to the named run, not to every planned benchmark condition.

| Service | Paired pilot | Forged-only progress | Current result | Status / next action |
|---|---:|---:|---|---|
| Sightengine `genai` | Not run in canonical paired form | 99/275 valid | 0/99 detected at threshold 0.5 | **Partial.** Original-PNG engineering pilot consumed the daily operation allowance; 176 forged remain, and the canonical paired run is still required. |
| Hive V3 | 10/10 valid (5 pairs) | 88/275 valid | 0/88 detected at vendor threshold 0.9 | **Partial.** Request 89 returned HTTP 429 under the self-serve daily quota; 187 valid forged results remain. |
| Resemble Detect | 10/10 valid (5 pairs) | 274/275 valid | 30 `Fake` + 11 `Likely fake` = 41/274 positive under the combined reporting rule | **Almost complete.** One image returned HTTP 402 because the wallet was one cent short; rerun that ID after adding balance. |
| Alibaba Cloud Ultra | 10/10 valid (5 pairs), plus one-image preflight | 275/275 valid | 30 `risk_edit` + 1 separate `risk_fake` = 31/275 any-risk | **Forged-only complete.** No errors; paired pilot also complete. |
| AI or Not | 10/10 valid (5 pairs) | 275/275 valid | 4/275 detected (1.45%) | **Forged-only complete.** No errors; paired pilot also complete. |
| Copyleaks Ultra | 4/4 valid (2 pairs) | 102/275 valid | 38/102 detected (37.25%); detected-only mask IoU 0.8266, all-image IoU 0.3080 | **Partial.** Balance is zero; 173 additional credits are required to finish forged-only. The next three pilot forgeries are included in the 102, but their corresponding real controls remain untested. |
| Illuminarty | No current valid run | 0/275 | Service unavailable during the 2026-07-20 recheck | **Unavailable.** Do not treat the historical endpoint as an executable baseline. |
| Reality Defender | Not run | 0/275 | No authenticated result | **Not started.** Requires credentials and a non-face coverage pilot before inclusion. |

Sensity, Winston AI, Tencent Cloud `IMAGE_AIGC`, and Google AI Content Detection have no local result artifacts in this snapshot. They remain access/trial candidates rather than executed baselines.

## Directory contents

- `sightengine/`: 99-image original-PNG forged pilot.
- `hive/`: five-pair canonical pilot and quota-stopped forged run.
- `resemble/`: five-pair canonical pilot, 274-image forged run, and all returned heatmap/visualization JPEG artifacts.
- `alibaba/`: one-image preflight, five-pair canonical pilot, and complete 275-image forged run.
- `aiornot/`: five-pair canonical pilot and complete 275-image forged run.
- `copyleaks/`: two paired preflights, the 100-image expansion, and a resumable 275-image file seeded with the first 102 unique forged results. Copyleaks RLE masks are stored directly in JSONL rows.

Every run uses three primary files:

- `*.jsonl`: one append-only result row per attempted image; the latest row for an ID is authoritative after a retry.
- `*.run_manifest.json`: ordered inputs, hashes, endpoint/model, encoding policy, and adapter provenance.
- `*.summary.json`: derived coverage, verdict, score, error, and localization statistics.

## Interpretation constraints

- Detection rates across services are not automatically comparable: Sightengine used original PNGs, some services currently cover only a prefix, and only small paired controls are complete.
- Forged-only rates do not measure false-positive rate, AUROC, AP, or paired accuracy. Those require the canonical real/forged run.
- Copyleaks localization must be reported both over all forged images (empty masks count as misses) and conditional on a positive mask.
- Resemble heatmaps and visualizations are provider-rendered JPEG artifacts, not calibrated single-channel score maps.
- `nonLabel`, `NOT_APPLICABLE`, HTTP errors, and quota failures must remain explicit rather than being silently dropped.

Detailed interpretation is maintained in `docs/COMMERCIAL_API_STATUS_2026-07-20.md` and the service-specific result reports under `docs/`.

## Security

No API key, account email, access token, authorization header, signed URL, or provider secret is stored in this directory. Runners read credentials from environment variables. Credentials pasted into chat or a terminal should still be rotated after the experiment.
