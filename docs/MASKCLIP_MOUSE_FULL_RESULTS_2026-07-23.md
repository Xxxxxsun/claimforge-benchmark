# OpenSDI MaskCLIP on the canonical mouse set (2026-07-23)

## 1. Status and why this method was first

The first open-source baseline is complete: 275 paired tasks, or 550 images,
finished with 550 valid results and no errors.

MaskCLIP was selected first because it is the most directly matched released
baseline in the survey: it was designed for diffusion-inpainted local edits and
provides both an image-level real/forged score (T1) and a pixel-level forged
probability map (T2). This makes it more informative for CLAIMFORGE than starting
with a whole-image AIGC classifier.

The run pins:

- OpenSDI commit `02c93d4891303637cb5d6852d3de63a099d69843`.
- Hugging Face repository revision
  `765f09adbce63ae201dfa451256fbbc419919450`.
- `MaskCLIP_sd15_20241109_08_53_19.pth`, epoch 13, SHA-256
  `481c8bd16077f942efec2901f93c1bc7008f6992402a1ab69fda2652408ca90f`.
- OpenAI CLIP ViT-L/14 SHA-256
  `b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836`.
- MAE initialization SHA-256
  `aec5f0b68e5f3193a00b07bc65a37440db549c15b36b8bea242606cc40c4bc5d`.

The older checkpoint named in the OpenSDI README belongs to an earlier
SideCLIP-shaped architecture and does not strictly match current `main`. The
chosen checkpoint strictly loaded all 778 state-dict keys into the pinned source.
Class direction was verified as index 0 = real and index 1 = forged. The adapter
uses `softmax(logits)[1]` for T1 and the released sigmoid mask for T2.

## 2. Evaluation contract

The authoritative input set is the 275 records with `status=good` and
`candidates=mouse` in `claimforge_generation_review_labels.json`:

| Domain | Pairs |
|---|---:|
| lodging | 147 |
| restaurant | 128 |
| total | 275 |

Each source and forged image was independently decoded to RGB and materialized
as a metadata-free JPEG at quality 95, 4:4:4 subsampling. Both sides therefore
receive the same canonicalization, and inference reads those materialized files
without another encoding pass.

The forged GT is the exact nonzero RGB difference between the decoded source and
decoded forged PNG *before* canonical JPEG encoding. The generation anchor box
is not used as the object mask: the true edit extends beyond that box in 234 of
275 cases. Every GT difference lies inside the recorded context crop.

The edits are unusually small:

| Edit fraction of the full image | Value |
|---|---:|
| minimum | 0.02486% |
| median | 0.11264% |
| mean | 0.16846% |
| maximum | 1.29272% |

Inference uses the official 512×512 stretch-resize convention, CLIP
normalization, float32, batch size 1, and seed 42. Score maps are preserved both
at 512×512 and after linear restoration to the original image size. Pixel AP is
computed from the continuous native-resolution score map; fixed-threshold
metrics use the released 0.5 threshold.

Confidence intervals use 1,000 percentile-bootstrap replicates with the paired
task as the resampling unit, so each real image and its forged counterpart remain
together.

## 3. Image-level detection (T1)

| Metric | Estimate | Pair-bootstrap 95% CI |
|---|---:|---:|
| AUROC | 0.5073 | [0.5042, 0.5117] |
| Average precision | 0.5118 | [0.5099, 0.5226] |
| TPR at FPR ≤ 5% | 0.0509 | [0.0436, 0.0691] |
| Accuracy at 0.5 | 0.5000 | [0.5000, 0.5000] |
| Image F1 at 0.5 | 0.0484 | [0.0143, 0.0803] |
| Paired ranking accuracy | 0.6473 | [0.5926, 0.7055] |
| Mean paired score change | +0.004019 | [+0.002048, +0.006212] |

The 0.5-threshold confusion matrix is `TP=7, FP=7, FN=268, TN=268`.
In other words, exactly the same seven source scenes cross 0.5 on both the real
and forged versions. The threshold therefore detects scene/content properties,
not the inserted mouse.

The forged score is nevertheless larger than its matched real score for 178 of
275 pairs (64.7%; exact two-sided sign-test `p=1.19e-6`). This is a measurable
paired response, but its mean magnitude is only 0.004 and it does not produce
useful absolute real-vs-forged separation. AUROC and AP remain essentially at
the balanced random baseline.

## 4. Pixel localization (T2)

| Native-resolution metric | Estimate | Pair-bootstrap 95% CI |
|---|---:|---:|
| Macro pixel AP | 0.04740 | [0.03135, 0.06552] |
| Median per-image pixel AP | 0.00463 | — |
| Macro pixel F1 at 0.5 | 0.00564 | [0.00078, 0.01219] |
| Macro pixel IoU at 0.5 | 0.00367 | [0.00042, 0.00812] |
| Micro pixel F1 at 0.5 | 0.00622 | [0.00123, 0.01255] |
| Micro pixel IoU at 0.5 | 0.00312 | [0.00062, 0.00631] |
| Box-hit rate at 0.5 | 1/275 (0.36%) | — |

At the released 0.5 threshold, 259/275 forged images have an empty predicted
mask. A box hit means IoU greater than 0.3 between the predicted binary mask and
the recorded edit anchor.

Threshold optimization does not rescue the method:

- A single oracle threshold selected on all evaluation pixels is 0.27326. It
  reaches only micro F1 0.03523 and micro IoU 0.01793.
- Allowing a different oracle threshold for every image gives mean/median F1
  0.06747/0.01371. This is an intentionally optimistic upper envelope, not a
  deployable result.

Continuous maps occasionally rank the correct pixels well but are badly
calibrated. For example, `lodging_104_slot_001` has pixel AP 0.9147 while its
0.5-threshold mask is empty. This is why the report retains both threshold-free
pixel AP and deployable fixed-threshold metrics.

## 5. Diagnostic slices

### Domain

| Domain | Pairs | AUROC | Paired rank acc. | Macro pixel AP | Median pixel AP | Micro IoU@0.5 |
|---|---:|---:|---:|---:|---:|---:|
| lodging | 147 | 0.5088 | 0.6599 | 0.07023 | 0.00613 | 0.00623 |
| restaurant | 128 | 0.5071 | 0.6328 | 0.02118 | 0.00288 | 0.00076 |

Neither domain has meaningful image-level separation. Localization is less poor
on lodging, but both values remain far below a usable mask.

### Edit-size quintiles

| Quintile | Pairs | Median edit fraction | AUROC | Paired rank acc. | Macro pixel AP |
|---|---:|---:|---:|---:|---:|
| Q1, smallest | 55 | 0.05347% | 0.5018 | 0.5636 | 0.01415 |
| Q2 | 55 | 0.07780% | 0.5068 | 0.6727 | 0.00675 |
| Q3 | 55 | 0.11264% | 0.5031 | 0.6364 | 0.01369 |
| Q4 | 55 | 0.16293% | 0.5051 | 0.6000 | 0.03541 |
| Q5, largest | 55 | 0.37441% | 0.5339 | 0.7636 | 0.16701 |

Only the largest-edit quintile shows a clear improvement, and even there AUROC
is 0.5339. This supports the expected size-sensitivity story: MaskCLIP has some
local ranking signal when the manipulated region is larger, but CLAIMFORGE's
sub-percent edits mostly fall below its reliable operating regime.

## 6. Interpretation

This is a valid negative baseline, not a failed pipeline:

- All 550 images completed and produced finite, nonconstant class scores and
  continuous maps.
- The official source and all three weight files were pinned and hash-verified.
- The checkpoint loaded with `strict=True`; unsafe unrestricted pickle loading
  was not used.
- The first fixed five-pair smoke run completed before the full run.
- The 10 images shared by the independent smoke and full runs match exactly in
  class score, logit margin, both score-map hashes, binary-mask hash, and all
  localization metrics (60/60 field comparisons).
- A post-hoc audit re-hashed 2,475 files and checked score-map dimensions,
  float32 dtype, finite values, `[0,1]` range, GT hashes, and exact equality
  between every saved binary mask and `native_score_map >= 0.5`.

The result says that even a released detector trained specifically for diffusion
local edits does not reliably detect or localize these tiny mouse insertions
zero-shot. It reacts slightly in paired comparisons and has occasional
pixel-ranking successes, but neither signal translates into an operational
decision threshold.

This result should not be generalized to larger edits, other object classes,
other editors, laundering conditions, or a fine-tuned MaskCLIP model. It also
should not be compared directly with forged-only commercial runs using AUROC or
FPR until their 275 real controls are complete.

## 7. Reproducible artifacts

| Artifact | Path | SHA-256 |
|---|---|---|
| canonical manifest | `outputs/opensource/mouse_canonical_v1/manifest.json` | `beb3c30e436db682bbadef794404838f33a4812f18f22819dd6ab1ef3de6f0b1` |
| full results | `results/opensource/maskclip/maskclip_mouse_canonical_v1_full275_20260723.jsonl` | `ac6e631215cdb9d0d93cf022eeb697f69588274da4b331db240f7aee78c020d4` |
| run manifest | same basename, `.run_manifest.json` | `04d1b70ba115f0fb34c555dc4e845633026b1334b3a0b74b725aba3a8c28e7bb` |
| runner summary | same basename, `.summary.json` | `461c0d68349484bdecaf6a7fbed4f7d7bb2f23c5721b467c6da930b385282d64` |
| post-hoc analysis | same basename, `.analysis.json` | `91000d24f747078cc11efaa34f02e0b1a59c0890a46e031b8a663fba7001bcdd` |

The 3.9 GB run-specific map artifacts are under
`outputs/opensource/maskclip/maskclip_mouse_canonical_v1_full275_20260723/`.
The append-only JSONL contains one authoritative row per input.
The manifest's deterministic dataset-contract hash is
`c419e24d6f9d69822ca575e00e30f2c769ba7a28a2fcea1f6634466caf540757`.

Environment: Python 3.12.3, PyTorch
`2.8.0.dev20250627+cu128`, CUDA 12.8, OpenCV 4.10.0,
IMDLBenCo 0.1.45, and one NVIDIA L20Z. Median measured forward latency is
34.48 ms/image at batch size 1; this excludes image decode, metric calculation,
hashing, and artifact writes. Treat latency as operational metadata rather than
a clean performance benchmark: a separate Hunyuan service was restarted by an
active generation job during the full run and shared GPU 0 for part of it. This
does not change the deterministic model outputs or evaluation metrics.

Commands:

```bash
python -m eval.opensource.build_mouse_canonical

CUDA_VISIBLE_DEVICES=0 \
  /root/.cache/claimforge/venvs/opensdi-02c93d/bin/python \
  -m eval.opensource.run_maskclip \
  --run-id maskclip_mouse_canonical_v1_full275_20260723 \
  --condition mouse_canonical_v1_full275 \
  --device cuda:0 --fail-fast

/root/.cache/claimforge/venvs/opensdi-02c93d/bin/python \
  -m eval.opensource.analyze_maskclip_run \
  --run-id maskclip_mouse_canonical_v1_full275_20260723 \
  --bootstrap-iterations 1000 --bootstrap-seed 20260723
```

The next method in the open-source run order should be TruFor: it is the standard
general-purpose image-manipulation reference, provides both image and pixel
outputs, and gives the cleanest contrast with this diffusion-edit-specific
baseline.
