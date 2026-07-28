# Commercial API Result Index (updated 2026-07-28)

This directory contains the raw JSONL responses, run manifests, generated summaries, and provider-returned artifacts for CLAIMFORGE commercial detector experiments.

## Current execution status

The forged-only target is the fixed set of 275 reviewed mouse edits. A standard paired pilot is five tasks (five real sources plus five corresponding forgeries). `Complete` below refers only to the named run, not to every planned benchmark condition.

| Service | Paired pilot | Forged-only progress | Current result | Status / next action |
|---|---:|---:|---|---|
| Sightengine `genai` | Not run in canonical paired form | 199/275 valid | 0/199 detected at threshold 0.5 | **Partial.** The 2026-07-21 resume added 100 valid results using 500 operations; 76 forged remain, and the canonical paired run is still required. |
| Hive V3 | 10/10 valid (5 pairs) | 275/275 valid | 0/275 detected at vendor threshold 0.9 | **Forged-only complete.** All scores and generator-attribution outputs are saved; full paired real controls remain untested. |
| Resemble Detect | 10/10 valid (5 pairs) | 275/275 valid | 30 `Fake` + 11 `Likely fake` = 41/275 positive under the combined reporting rule | **Forged-only complete.** All 275 classifications, IFL heatmaps, and visualizations are saved. |
| Alibaba Cloud Ultra | 10/10 valid (5 pairs), plus one-image preflight | 275/275 valid | 30 `risk_edit` + 1 separate `risk_fake` = 31/275 any-risk | **Forged-only complete.** No errors; paired pilot also complete. |
| AI or Not | 10/10 valid (5 pairs) | 275/275 valid | 4/275 detected (1.45%) | **Forged-only complete.** No errors; paired pilot also complete. |
| Copyleaks Ultra | 4/4 valid (2 pairs) | 275/275 valid | 111/275 detected (40.36%); detected-only mask IoU 0.8165, all-image IoU 0.3296 | **Forged-only complete.** All native RLE masks and exact-difference localization metrics are saved; paired real controls remain limited to two tasks. |
| Illuminarty | No current valid run | 0/275 | Service unavailable during the 2026-07-20 recheck | **Unavailable.** Do not treat the historical endpoint as an executable baseline. |
| Reality Defender | Not run in paired form | 50/275 valid prefix | 50/50 `AUTHENTIC`; 100% applicable; normalized score 0.01–0.03 | **Coverage pilot complete.** Non-face coverage is sufficient in this slice, but all 50 local edits evade the overall verdict; real controls and additional quota are still required. |

Sensity, Winston AI, Tencent Cloud `IMAGE_AIGC`, and Google AI Content Detection have no local result artifacts in this snapshot. They remain access/trial candidates rather than executed baselines.

## Frozen 250x3x2 benchmark execution

The newer frozen benchmark is
`benchmark/claimforge_v1_250x3x2`, with 250 matched tasks per object class and
generation route.

Current `full_image/mouse` execution status:

| Service | Valid/250 | Positive on valid | Status |
|---|---:|---:|---|
| Hive V3 | 250 | 107 (threshold 0.9) | Complete |
| Resemble Detect | 250 | 249 `Fake` or `Likely fake` | Complete |
| Alibaba Cloud Ultra | 250 | 230 any-risk | Complete |
| AI or Not | 250 | 230 | Complete |
| Copyleaks Ultra | 250 | 220 non-empty AI masks | Complete |
| Sightengine `genai` | 250 | 237 (threshold 0.5) | Complete |
| Reality Defender | 0 | N/A | Input dry-run passed; API key unavailable |
| Illuminarty | 0 | N/A | One-image preflight returned HTTP 502 five times |

The machine-readable status is
`full_image_mouse250_status_20260725.json`; detailed interpretation and the
matched local-splice comparison are in
`docs/COMMERCIAL_FULL_IMAGE_MOUSE250_RESULTS_2026-07-25.md`.

Current commercial main-table coverage:

| Service | Valid / planned | Deferred by protocol |
|---|---:|---|
| Alibaba Cloud Ultra | 1,750 / 1,750 | None |
| AI or Not | 1,750 / 1,750 | None |
| Hive V3 | 1,250 / 1,750 | Full-frame cat and trash-can |
| Resemble Detect | 1,250 / 1,750 | Full-frame cat and trash-can |
| Copyleaks Ultra | 1,250 / 1,750 | Full-frame cat and trash-can |
| Sightengine `genai` | 1,250 / 1,750 | Full-frame cat and trash-can |

Alibaba Cloud Ultra now also has a complete commercial main-table evaluation:
1,250/1,250 newly submitted real/cat/trash images completed without API errors,
and the two byte-identical frozen mouse250 runs were reused. The resulting
1,750-image evaluation uses one shared independent real250 panel. Under the
predeclared any-risk rule (`risk_aigc OR risk_fake OR risk_edit`), real-panel
specificity is 88.8%. Balanced accuracy is 50.2%, 54.4%, and 49.8% for local
mouse/cat/trash-can, versus 90.4%, 93.2%, and 91.2% for their full-frame
controls. Raw results, provenance, bootstrap intervals, and the reuse audit are
in:

- `alibaba/claimforge_balanced250_missing1250_canonical_jpeg_q95_20260727.jsonl`
- `alibaba/claimforge_balanced250_main_table_20260727.summary.json`
- `docs/ALIBABA_BALANCED250_RESULTS_2026-07-27.md`

AI or Not is complete under the same commercial main-table protocol. The run
added 1,250/1,250 valid real/cat/trash results with no final API errors and
reused the two byte-identical frozen mouse250 runs. Vendor `ai_detected` gives
98.8% specificity on real250. Balanced accuracy is 50.2%, 49.8%, and 50.0%
for local mouse/cat/trash-can, versus 95.4%, 96.4%, and 95.6% for their
full-frame controls. The rejected old-key preflight remains as one superseded
JSONL row; the subsequent successful row for that ID is authoritative.

- `aiornot/claimforge_balanced250_missing1250_canonical_jpeg_q95_20260727.jsonl`
- `aiornot/claimforge_balanced250_main_table_20260727.summary.json`
- `docs/AIORNOT_BALANCED250_RESULTS_2026-07-27.md`

Hive is complete for the 1,250-image core main-table scope. The new independent
real/local-cat/local-trash panel is 750/750 valid with no final API errors, and
the byte-identical local/full-frame mouse250 runs are reused. At Hive's 0.9
threshold, real-panel specificity is 100.0%. Balanced accuracy is 50.0%, 50.2%,
and 50.0% for local mouse/cat/trash-can, versus 71.4% for full-frame mouse.
Full-frame cat and trash-can were deferred before submission. The append-only
new-run log has 814 API-attempt rows; 64 earlier attempts were superseded by
success. All 750 latest uploads match the frozen canonical JPEG hashes.

- `hive/claimforge_balanced250_real_local750_canonical_jpeg_q95_20260727.jsonl`
- `hive/claimforge_balanced250_main_table_20260727.summary.json`
- `docs/HIVE_BALANCED250_RESULTS_2026-07-27.md`

Resemble Detect is complete on the same `full_image/mouse` cell: 250/250 valid,
with 250 heatmaps and 250 visualizations saved. Labels are 248 `Fake`, one
`Likely fake`, and one `Real`, so the existing combined-positive reporting rule
gives 249/250 (99.6%). On the same 250 task IDs, the local-splice run gives
34/250 combined positives. No real controls were included in this full-image
run, so these forged-only rates are not classification accuracy or AUROC.

Artifacts:

- `hive/claimforge_v1_full_image_mouse250_canonical_jpeg_q95_20260725.jsonl`
- the matching `.run_manifest.json` and `.summary.json`
- `docs/HIVE_FULL_IMAGE_MOUSE250_RESULTS_2026-07-25.md`
- `resemble/claimforge_v1_full_image_mouse250_canonical_jpeg_q95_20260725.jsonl`
- the matching `.run_manifest.json`, `.summary.json`, and artifact directory
- `docs/RESEMBLE_FULL_IMAGE_MOUSE250_RESULTS_2026-07-25.md`

## Directory contents

- `sightengine/`: resumable 199-image local-splice pilot and complete 250-image
  full-image mouse run.
- `hive/`: five-pair canonical pilot, complete 275-image local-splice forged
  run, complete 250-image full-image mouse run, and complete 1,250-image core
  main-table evaluation.
- `resemble/`: five-pair canonical pilot, complete 275-image local-splice run,
  complete 250-image full-image mouse run, and all returned
  heatmap/visualization JPEG artifacts.
- `alibaba/`: one-image preflight, five-pair canonical pilot, complete
  275-image local-splice run, and complete 250-image full-image run.
- `aiornot/`: five-pair canonical pilot, complete 275-image local-splice run,
  and complete 250-image full-image run.
- `copyleaks/`: paired preflights, complete 275-image local-splice run, and
  complete 250-image full-image run. Copyleaks RLE masks are stored directly
  in JSONL rows.
- `illuminarty/`: failed full-image preflight artifacts retained for service
  availability provenance.
- `reality_defender/`: authenticated 50-image forged-only canonical coverage pilot produced by `eval/commercial/run_reality_defender.py`.

Every run uses three primary files:

- `*.jsonl`: one append-only result row per attempted image; the latest row for an ID is authoritative after a retry.
- `*.run_manifest.json`: ordered inputs, hashes, endpoint/model, encoding policy, and adapter provenance.
- `*.summary.json`: derived coverage, verdict, score, error, and localization statistics.

## Interpretation constraints

- Detection rates across services are not automatically comparable because
  decision rules, returned outputs, and upload encodings differ.
- Forged-only rates do not measure false-positive rate, AUROC, AP, or paired accuracy. Those require the canonical real/forged run.
- Copyleaks localization must be reported both over all forged images (empty masks count as misses) and conditional on a positive mask.
- Resemble heatmaps and visualizations are provider-rendered JPEG artifacts, not calibrated single-channel score maps.
- `nonLabel`, `NOT_APPLICABLE`, HTTP errors, and quota failures must remain explicit rather than being silently dropped.

Detailed interpretation is maintained in `docs/COMMERCIAL_API_STATUS_2026-07-20.md` and the service-specific result reports under `docs/`.

## Security

No API key, account email, access token, authorization header, signed URL, or provider secret is stored in this directory. Runners read credentials from environment variables. Credentials pasted into chat or a terminal should still be rotated after the experiment.
