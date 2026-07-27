# Balanced250 open-source expansion plan

**Frozen:** 2026-07-26

**Updated:** 2026-07-27

**Status:** 8/19 methods complete (7/9 whole-image, 1/10 local); next Effort

**Scope:** the 19 executable open-source methods already completed on the
Mouse benchmark: ten local-forensics methods and nine whole-image AIGC
detectors.

## 1. Why this expansion is required

At the time this plan was frozen, all existing formal full runs used
`outputs/opensource/mouse_canonical_v1/manifest.json`: 275 matched real/mouse
pairs and 550 canonical JPEG inputs. There were no formal detector results on
the final Cat set, final Trash-can set, or any of the three full-frame
conditional-edit sets. The expansion is now underway on the frozen
Balanced250 release; completed methods are marked below.

The expansion must not reuse the Mouse-specific schema name for non-Mouse
conditions. It must also preserve the distinction between:

- a local insertion, which has an exact-difference localization target;
- a Hunyuan full-frame conditional edit, which has no valid local T2 target;
- a fully synthetic image generated independently of a real source, which is
  not part of the currently available data.

## 2. Frozen seven-condition panel

The main panel contains 1,750 independently selected images:

| Condition | Panel rows | Eligible rows before selection |
|---|---:|---:|
| `real` | 250 | 275 |
| `local_mouse` | 250 | 275 |
| `local_cat` | 250 | 251 |
| `local_trash_can` | 250 | 250 |
| `fullframe_mouse` | 250 | 275 |
| `fullframe_cat` | 250 | 272 |
| `fullframe_trash_can` | 250 | 260 |

The six forged conditions are selected independently. They are not required
to share task IDs. Selection is score-blind and deterministic:

```text
sha256(dataset_id + NUL + condition + NUL + normalized_task_id)
```

The first 250 eligible rows in hash order are selected. For the `real` panel,
one task per raw-content SHA-256 is preferred before hash ranking so that 250
content-unique real rows can be frozen when the eligible pool permits it.

The score cache retains all 275 real rows, including the 25 real rows outside
the displayed panel, plus the six selected forged sets. It therefore contains
1,775 unique model inputs. This small superset permits a strict source-matched
secondary analysis without repeating model inference.

Selection may not use detector scores or full-frame semantic-QC decisions.
The existing Trash-can full-frame QC label is carried as a stratum. The
primary label is “the primary single-shot image passed through the AIGC
full-frame conditional-edit process,” not “a natural Trash-can was
successfully inserted.”

## 3. Canonical release

The release lives at:

```text
outputs/opensource/balanced250_v1/
├── manifest.json
├── inputs.jsonl
├── panel.jsonl
├── source_pairs.jsonl
├── images/
└── masks/
```

Every model input is decoded with EXIF transpose, converted to RGB, and
encoded as JPEG with quality 95, 4:4:4 (`subsampling=0`), `optimize=False`,
and no metadata. Native decoded geometry is retained. The release is
self-contained: all 1,775 canonical JPEGs are re-encoded from their frozen
raw inputs, including the Mouse cache, and all 750 local exact-difference
masks live inside this release rather than depending on the earlier
Mouse-canonical directory.

`inputs.jsonl` is the 1,775-row score cache. `panel.jsonl` references the
1,750 rows in the main independent panel. `source_pairs.jsonl` records the
1,500 genuine source links for the six forged conditions.

Local Mouse, Cat, and Trash-can rows receive exact decoded-space difference
masks. Diff pixels outside an annotation context are retained and counted;
the context box is not treated as ground truth. Full-frame rows use
`gt_mask_kind=not_applicable` and record the orange placement box only as
conditioning metadata.

The release validator must fail closed on missing or extra files, duplicate
identities, path traversal, malformed rows, hash or byte mismatches,
non-RGB/non-JPEG canonical inputs, invalid masks, wrong condition semantics,
source-link inconsistencies, selection drift, and ledger-hash drift.

The completed release passed a full independent source-to-output replay:

| Frozen artifact | Rows/files | SHA-256 |
|---|---:|---|
| deterministic contract | — | `671d1739bebf4370d26b4629ca26b56cc546a817d469ba505cc39bda8b33102c` |
| `manifest.json` | 1 | `b2bbf3eb7a835f9c729cdffe29a40247225125779fe21551270fefe95d667c7f` |
| `inputs.jsonl` | 1,775 | `6b5128909eeffdbd88e61af02ca1bd191cb5460f94a23b47c87ebd0172e6d12c` |
| `panel.jsonl` | 1,750 | `e01d7985b41cee5262a3f8b6d71420986feae96771b11c46fda98c3e72a0d424` |
| `source_pairs.jsonl` | 1,500 | `391fdcf06eecff4cf1843ddb3688acacf52a293725c501660b7a361173b09b30` |
| canonical JPEG inventory | 1,775 | individually frozen in `inputs.jsonl` |
| local-mask inventory | 750 | individually frozen in `inputs.jsonl` |

The exact release inventory is 2,529 regular files in two directories. The
validator independently re-decodes every raw input, reproduces every JPEG
byte-for-byte, recomputes every local mask, and rejects JPEG COM, EXIF, ICC,
or other non-JFIF metadata.

## 4. Evaluation design

The primary T1 comparisons are unpaired:

- each 250-image forged condition versus the independent `real250` panel;
- AUROC, AP, TPR at 5% FPR, released fixed-threshold metrics, coverage, and
  confidence intervals;
- separate lodging and restaurant results;
- macro averages across the three local conditions and across the three
  full-frame conditions.

No paired ranking, paired delta, sign test, or paired bootstrap may be
reported for the independent primary panel.

The secondary analysis uses `source_pairs.jsonl` and the cached real275 pool.
It reports genuine source-matched deltas and task/source-cluster bootstrap
intervals. Repeated source content is clustered by raw source SHA-256 rather
than counted as independent evidence.

T2 applies only to local insertions. It uses the full exact-difference mask
and reports pixel AP, precision, recall, F1, IoU, MCC, and real-image
false-positive area. Full-frame conditional edits have T2=`N/A`; dense maps
may be saved as diagnostics but are not scored against the orange box or an
invented all-image mask.

## 5. Method matrix

### Local-forensics methods

Native T1+T2:

1. OpenSDI / MaskCLIP — completed and pushed; see
   [formal report](MASKCLIP_BALANCED250_FULL_RESULTS_2026-07-26.md)
2. TruFor
3. MVSS-Net
4. PSCC-Net
5. HiFi-IFDL

Native T2 only:

6. CAT-Net v2
7. IML-ViT
8. Mesorch
9. RelayFormer
10. DINOv3-IML

All ten are evaluated on the three local conditions. The five native-T1
methods also enter the full-frame T1 table. T2-only methods must not promote a
map statistic to an image score.

### Whole-image AIGC methods

1. FSD
2. UniversalFakeDetect
3. NPR
4. Community Forensics
5. SPAI
6. B-Free
7. Effort
8. OmniAID
9. CNNDetection

Completed whole-image methods:

1. FSD — [formal report](FSD_BALANCED250_FULL_RESULTS_2026-07-26.md)
2. UniversalFakeDetect —
   [formal report](UNIVERSALFAKEDETECT_BALANCED250_FULL_RESULTS_2026-07-26.md)
3. NPR — [formal report](NPR_BALANCED250_FULL_RESULTS_2026-07-26.md)
4. Community Forensics —
   [formal report](COMMUNITY_FORENSICS_BALANCED250_FULL_RESULTS_2026-07-26.md)
5. SPAI — [formal report](SPAI_BALANCED250_FULL_RESULTS_2026-07-26.md)
6. CNNDetection —
   [formal report](CNNDETECTION_BALANCED250_FULL_RESULTS_2026-07-26.md)
7. B-Free —
   [formal report](BFREE_BALANCED250_FULL_RESULTS_2026-07-27.md)

Pending whole-image methods, in frozen order: Effort, OmniAID.

All nine are evaluated on all seven panel conditions for T1. LTD remains
blocked on exact official-weight access and is not counted as executable.
NFA-ViT remains the analogous blocked local method.

### Expansion progress ledger

- completed: 8/19;
- whole-image: 7/9 complete, 2 pending;
- local-forensics: 1/10 complete, 9 pending;
- total remaining: 11;
- next queued method: Effort.

## 6. Execution and audit order

The shared dataset loader and analysis code are generalized before any
Cat/Trash detector score is produced. Existing official model contracts,
preprocessing profiles, checkpoint hashes, score directions, and thresholds
remain frozen.

The first two canaries are:

1. CNNDetection, to validate generic T1 and all seven condition groups.
2. MaskCLIP, to validate combined native T1/T2 and local-mask restoration.

For every method:

```text
CPU preflight
-> five-image-per-condition A/B deterministic smoke
-> capability-correct full run
-> complete artifact/statistical analysis
-> fresh model replay audit
-> result report
-> commit and push
```

The nine whole-image AIGC detectors and five native-T1 local methods each run
the full 1,775-input cache. The five T2-only methods run the 1,025 applicable
inputs (`real275 + local_mouse250 + local_cat250 + local_trash_can250`);
full-frame rows are T1=`N/A`, T2=`N/A`, not synthetic image scores derived
from map statistics.

After the canaries, the remaining whole-image methods run one at a time,
followed by the remaining local-forensics methods. A method is not marked
complete or pushed until coverage, hashes, metrics, replay, documentation,
and the clean-worktree check all pass.

The exhaustive capability-correct score cache requires 29,975 formal model
forwards. A full fresh replay requires the same number again, for 59,950
forwards before smoke tests.

After B-Free, the remaining executable queue requires 15,775 formal forwards
and 15,775 fresh-replay forwards, or 31,550 total before smoke tests. The
original 29,975 / 59,950 figures above remain the frozen all-method totals.

## 7. GPU occupancy

The Hunyuan keepalive remains active during CPU-only preparation, analysis,
documentation, pushes, and after the complete queue finishes. Before each
CUDA benchmark it is paused and allowed to drain all in-flight requests. It
is resumed immediately after the CUDA window closes, including after a
failure or interrupted run.
