# IML-ViT CAT-protocol checkpoint on the canonical mouse set (2026-07-24)

## 1. Status and headline

The sixth publicly reproducible local-manipulation baseline is complete.
The official IML-ViT CAT/TruFor-protocol checkpoint ran successfully on all
275 matched tasks, or 550 canonical JPEG images, with no errors.

IML-ViT detects a real subset of CLAIMFORGE's small local diffusion insertions,
but it is not reliable enough as a zero-shot localizer:

- native macro pixel AP is **0.15513**, with paired-task bootstrap 95% CI
  **[0.12553, 0.18738]**;
- the per-image pixel-AP median is only **0.02189**, showing that a minority of
  strong cases raises the mean;
- native macro pixel F1 at the official strict `> 0.5` threshold is
  **0.14325**, CI **[0.11611, 0.17258]**;
- native micro pixel F1 is **0.06995**, CI **[0.04476, 0.11081]**;
- 131/275 forged images have exactly zero fixed-threshold F1;
- 23/275 masks reach edit-box IoU greater than 0.3; and
- real images have **0.8023%** mean false-positive area, with a long tail up
  to **24.3091%**.

This places IML-ViT clearly above MaskCLIP, MVSS-Net, and PSCC-Net for T2, but
well below CAT-Net v2 and TruFor. It is the current middle tier rather than a
second state-of-the-art result.

IML-ViT has no native image-level classification head. T1 is therefore
**N/A**. No heatmap mean, maximum, or other aggregation is promoted into an
unofficial image score.

The independent analyzer verified 3,025 unique files and reproduced the RGB
preprocessing, model-logit sigmoid, valid-content crop, native interpolation,
strict threshold mask, GT mapping, and both localization spaces. These are
complete audited results, not partial coverage or a score-direction error.

## 2. Pinned method and official assets

The run uses the authors'
[IML-ViT repository](https://github.com/SunnyHaze/IML-ViT) at commit
`07dd2be0f4ea27a5c97c9fa5ffbe236733833eac`. The code repository is MIT
licensed. The released checkpoint has no separately stated license, so the
manifest deliberately does not claim that the weight file itself is MIT.

The selected author release is:

| Field | Frozen value |
|---|---|
| README identity | CAT-Net protocol checkpoint; TruFor follows the same protocol |
| Original filename | `iml-vit_checkpoint_trufor_20231104.pth` |
| Google Drive file ID | `1jlXw97GkyBbY4u5-e_liuhahKSQWCAFu` |
| Release announcement commit | `5ad22146b1223eac841fa3e0e28c1c4e8948cc95` |
| Bytes | 367,195,954 |
| SHA-256 | `9fa9ae88cafeb6eab28c2afd5bef74679416cf0a790b2370fa6a6fb4c122c58c` |

The filename contains `trufor`, but this run is IML-ViT, not the TruFor model.
The official README describes this file as the CAT-Net-protocol checkpoint and
notes that TruFor follows that evaluation protocol.

Safe inspection returns a raw 212-key `collections.OrderedDict`, not a training
wrapper with a top-level `model` key. It contains 91,778,242 state elements:
91,777,729 parameters and 513 buffer elements. All 212 entries are tensors;
211 are float32 and one is int64. The default BN-head architecture loads every
key with `strict=True`, with no prefix rewrite, schema fallback, or secondary
MAE initialization.

The checkpoint does not contain an auditable training manifest, exact epoch,
or precise sample list. The report therefore identifies the released protocol
and exact bytes without inventing a more specific training history.

## 3. Exact inference contract

### Input geometry

The primary run follows section 4.1 of the
[IML-ViT paper](https://arxiv.org/abs/2307.14863), not the incomplete
large-image path in the repository demo:

1. Decode the canonical JPEG bytes with Pillow and convert to RGB.
2. If and only if `max(H,W) > 1024`, shrink the longer side to 1024 while
   preserving aspect ratio. The frozen implementation uses
   Albumentations 1.3.0 `LongestMaxSize`, Python-3 rounding, and
   `cv2.INTER_LINEAR`.
3. Do not enlarge images already within the limit.
4. Place the resized image at the top left of a 1024×1024 raw-RGB canvas.
5. Pad only the right and bottom with raw RGB value zero.
6. Apply ImageNet mean `[0.485, 0.456, 0.406]` and standard deviation
   `[0.229, 0.224, 0.225]` after padding.
7. Apply no crop, re-encoding, TTA, or ensemble.

This distinction is material. 202/275 pairs contain images exceeding 1024.
The repository's literal `PadIfNeeded → Normalize → top-left Crop` path would
completely remove the exact-difference GT region in 83 forged images and
partially crop it in four more. Stretching every image to a square would cover
the image but contradict the paper. The frozen fit-and-pad path preserves the
complete scene and aspect ratio.

The hosted Colab confirms that large-image resizing is conditional, but its
Pillow helper swaps width and height. The adapter implements the paper's
stated geometry rather than copying that bug.

### Model and output

The numerical path is:

```text
3×1024×1024 normalized RGB
→ ViT-B/16 window/global-attention encoder
→ five-level simple feature pyramid
→ fused one-channel prediction head
→ 256×256 logits
→ bilinear logit interpolation to 1024×1024, align_corners=False
→ one sigmoid
→ manipulated-pixel probability
```

The adapter calls the encoder, pyramid, and head directly because the official
`forward()` unnecessarily requires GT tensors, calculates training losses, and
prints debug shapes during inference. On a canonical forged image, this direct
path was compared with official `forward()` in `eval()` mode and produced a
bit-identical 1024 probability map.

The complete 1024 float32 logit and probability maps are saved. For native
evaluation:

1. crop the right/bottom padding from the probability map;
2. bilinearly restore the valid continuous map to the original dimensions
   with `align_corners=False`;
3. save that native float32 probability; and
4. create the lossless PNG mask with strict `probability > 0.5`.

The native restoration is a declared CLAIMFORGE compatibility adapter. The
probability is restored before thresholding.

The auxiliary `model_1024` metrics use only the valid resized-content
rectangle. Padding pixels are excluded, and native GT is mapped to that
rectangle with nearest-neighbor interpolation.

## 4. Why IML-ViT can be strong

IML-ViT combines a large ViT-B/16 encoder with windowed and selected global
attention, a multi-scale feature pyramid, and a decoder that fuses five
resolutions. This gives it two useful properties:

- transformer context can compare distant regions and model image-wide
  inconsistency; and
- the feature pyramid recovers spatial detail that a plain low-resolution ViT
  classification token would discard.

That design explains the strong tail of the CLAIMFORGE result: the best
per-image AP is 0.9891 and 38/275 forged images reach AP at least 0.5.
The method can sharply rank the edited pixels when a generated insertion
creates traces aligned with its learned representation.

The same design does not guarantee tiny-edit robustness. The median edit is
only 0.1126% of the native image. Large inputs are downsampled, and a very
small object may occupy only a few patch tokens. A well-blended diffusion
insertion also need not reproduce the boundary, resampling, or source-mismatch
statistics in the released training distribution. These are plausible
mechanisms consistent with the observed size dependence; they are not causal
claims from a controlled ablation.

## 5. Preflight and run-order checks

The final-code one-pair preflight completed with:

- 2/2 valid images and no errors;
- peak allocated CUDA memory 2,921,058,304 bytes;
- forged native pixel AP 0.80017 and F1 0.64678; and
- matched-real false-positive area 1.7303%.

The five-pair smoke then completed 10/10 images:

- forged native macro AP 0.36292;
- forged native macro F1 0.27577; and
- real native false-positive area macro 0.50864%.

The smoke was resumed with zero pending samples and remained exactly ten JSONL
rows. All ten shared smoke/full inputs match bit-for-bit on raw logits, model
probabilities, native probabilities, masks, preprocessing metadata, and
recorded localization metrics.

The full run was also resumed after completion: it reported 550 selected and
zero pending, while the JSONL remained exactly 550 rows.

## 6. Primary T2 result

All metrics below are native-resolution results. Pixel AP is calculated only
on forged images. F1 and IoU use the fixed official strict threshold.
Confidence intervals use 1,000 task-paired bootstrap resamples.

| Native metric | Estimate | Pair-bootstrap 95% CI |
|---|---:|---:|
| Macro pixel AP | **0.15513** | [0.12553, 0.18738] |
| Median per-image pixel AP | 0.02189 | — |
| Macro pixel F1 at 0.5 | **0.14325** | [0.11611, 0.17258] |
| Macro pixel IoU at 0.5 | 0.09843 | [0.07834, 0.12022] |
| Micro pixel F1 at 0.5 | 0.06995 | [0.04476, 0.11081] |
| Micro pixel IoU at 0.5 | 0.03624 | [0.02289, 0.05865] |
| Real FP area, macro | 0.80226% | [0.58104%, 1.06218%] |
| Real FP area, micro | 0.90031% | [0.59726%, 1.28728%] |

The forged-image native confusion totals are:

| TP | FP | FN | TN |
|---:|---:|---:|---:|
| 132,319 | 3,054,980 | 463,584 | 438,602,121 |

This gives micro precision `0.04151` and recall `0.22205`. The model predicts
3,187,299 positive pixels against 595,903 GT-positive pixels; only about 4.2%
of its predicted positives are correct.

The per-image result is highly heterogeneous:

- 131/275 forged images have F1 exactly zero;
- all 275 forged images have a nonempty prediction, so most zero-F1 cases are
  predictions in the wrong location rather than empty outputs;
- 144/275 masks hit at least one exact-difference GT pixel;
- 38/275 images have AP at least 0.5; and
- 34/275 have fixed-threshold F1 at least 0.5.

The edit-box diagnostic is:

| Box diagnostic | Hits | Rate |
|---|---:|---:|
| Any predicted pixel inside edit box | 146/275 | 53.09% |
| Mask/edit-box IoU greater than 0.3 | 23/275 | 8.36% |

Mean edit-box IoU is `0.06820`, but its median is only `0.00111`.

## 7. Model-space and calibration diagnostics

The valid-content model-space result is nearly identical to native macro
behavior:

| Metric | Model valid content | Native |
|---|---:|---:|
| Macro pixel AP | 0.15412 | 0.15513 |
| Macro F1 | 0.14329 | 0.14325 |
| Macro IoU | 0.09845 | 0.09843 |

This rules out native probability restoration as the main failure mechanism.
Micro values differ because downsampling changes pixel weighting, not because
the macro conclusion changes.

Ground-truth-dependent threshold diagnostics are explicitly non-primary:

- the best approximate single threshold on the complete test set is
  `0.81805`, yielding micro F1 `0.08408`;
- a separately optimized threshold for every forged image yields mean F1
  `0.18563` and median F1 `0.05794`.

The global oracle improves only modestly over the registered micro F1 of
`0.06995`. Threshold calibration contributes to the error but does not explain
the low median AP or the frequent wrong-location response. Neither diagnostic
replaces the fixed `> 0.5` result.

## 8. Pristine-image and paired-background behavior

All 275 real images contain at least one positive pixel at the fixed threshold:

- mean false-positive area: **0.8023%**;
- median: **0.3232%**;
- 95th percentile: **2.1974%**;
- maximum: **24.3091%**;
- 43/275 real images exceed 1% false-positive area;
- 7/275 exceed 5%; and
- 3/275 exceed 10%.

The five largest real outliers contribute 38.0% of all real false-positive
pixels. This is a meaningful tail, even though the mean is below 1%.

Shared scene structure dominates much of the response:

- forged mean positive area is 0.6257%, lower than the real mean of 0.8023%;
- paired real/forged positive-area correlation is `0.9139`;
- paired binary-mask IoU has median `0.8839`; and
- 207/275 paired masks have IoU above 0.5.

These are localization-map diagnostics, not an image-level detection score.
They show that much of the fixed-threshold mask persists on the shared
background, while a subset of edits still causes a useful local response.

## 9. Diagnostic slices

### Domain

| Domain | Pairs | Macro AP | Median AP | Macro F1 | Micro IoU | Real FP area |
|---|---:|---:|---:|---:|---:|---:|
| lodging | 147 | 0.19357 | 0.04605 | 0.17577 | 0.03530 | 0.97320% |
| restaurant | 128 | 0.11097 | 0.01222 | 0.10590 | 0.03764 | 0.60595% |

Lodging has better macro ranking and per-image F1, but its larger false-positive
tail keeps micro IoU similar to restaurant. This is a descriptive domain
difference, not a causal claim.

### Edit-size quintiles

| Quintile | Pairs | Median edit fraction | Macro AP | Macro F1 | Micro IoU | Zero-F1 |
|---|---:|---:|---:|---:|---:|---:|
| Q1, smallest | 55 | 0.05347% | 0.05317 | 0.05347 | 0.00722 | 40 |
| Q2 | 55 | 0.07780% | 0.06796 | 0.06799 | 0.01543 | 30 |
| Q3 | 55 | 0.11264% | 0.12170 | 0.11305 | 0.01719 | 24 |
| Q4 | 55 | 0.16293% | 0.18008 | 0.16458 | 0.04302 | 27 |
| Q5, largest | 55 | 0.37441% | 0.35272 | 0.31716 | 0.18353 | 10 |

The size effect is substantial. Q5 macro AP is about 6.6 times Q1, and the
zero-F1 rate drops from 72.7% to 18.2%. Spearman correlations are `0.486`
between edit fraction and AP and `0.382` between edit fraction and fixed F1.
AP naturally benefits from higher positive prevalence, but AP divided by edit
fraction still has a small positive correlation (`rho=0.155`, `p=0.010`).
The fixed-threshold trend confirms that prevalence alone does not explain the
entire effect.

## 10. Comparison with completed local baselines

All six methods use the same 550-image canonical manifest and exact-difference
native GT. The table compares T2 only; T1 availability differs by architecture.

| Method | Native T1 | Macro AP | Macro F1 | Macro IoU | Micro IoU | Box IoU>0.3 |
|---|---|---:|---:|---:|---:|---:|
| CAT-Net v2 | no | **0.61234** | 0.47481 | 0.35743 | 0.18138 | 65 |
| TruFor | yes | 0.57989 | **0.50010** | **0.37410** | **0.19800** | **83** |
| **IML-ViT** | **no** | **0.15513** | **0.14325** | **0.09843** | **0.03624** | **23** |
| MaskCLIP | yes | 0.04740 | 0.00564 | 0.00367 | 0.00312 | 1 |
| MVSS-Net | map GMP | 0.03415 | 0.01962 | 0.01248 | 0.00369 | 4 |
| PSCC-Net | independent head | 0.01528 | 0.00007 | 0.00004 | 0.00115 | 0 |

IML-ViT is unambiguously above the three weak localizers and below the two
leaders on point estimates. Its macro-AP CI does not overlap CAT-Net or
TruFor's reported levels, although a formal pairwise method-difference
bootstrap has not been run.

The result also clarifies why method count alone is insufficient: modern
transformer architecture and high input resolution improve over older weak
baselines, but the checkpoint still misses nearly half the edits at its fixed
operating point. CAT-Net and TruFor remain the robust local-method anchors.

## 11. Validation and audit

The final result is backed by:

- safe `torch.load(..., weights_only=True)` and strict checkpoint loading;
- eight pinned official source-file hashes and a clean source commit;
- 18 runner/metrics tests and nine independent-analyzer tests;
- final-code preflight followed by five-pair smoke and 275-pair full;
- 550 unique successful full rows, zero errors, and zero duplicate histories;
- a zero-pending full-run resume;
- bit-identical smoke/full artifacts for the shared ten-image prefix;
- independent validation of 550 canonical JPEGs, 275 forged GT masks, and
  2,200 generated artifacts; and
- independent reproduction of preprocessing tensor hashes, sigmoid maps,
  native interpolation, strict masks, and both recorded metric spaces.

The full repository suite collected 144 tests: 140 passed and four
environment-gated tests were skipped. The 2,200 full-run model artifacts total
8,153,834,700 bytes.

Median measured model-forward latency is 40.11 ms on an NVIDIA L20Z, with
peak allocated CUDA memory of 2,921,058,304 bytes. Timing excludes decoding,
artifact serialization, and metric calculation.

## 12. Reproducible artifacts

Primary files:

- runner: `eval/opensource/run_imlvit.py`
- metric implementation: `eval/opensource/imlvit_metrics.py`
- independent analyzer: `eval/opensource/analyze_imlvit_run.py`
- full JSONL:
  `results/opensource/imlvit/imlvit_cat_protocol_mouse_canonical_v1_full275_20260724.jsonl`
- immutable run manifest:
  `results/opensource/imlvit/imlvit_cat_protocol_mouse_canonical_v1_full275_20260724.run_manifest.json`
- run summary:
  `results/opensource/imlvit/imlvit_cat_protocol_mouse_canonical_v1_full275_20260724.summary.json`
- audited paired analysis:
  `results/opensource/imlvit/imlvit_cat_protocol_mouse_canonical_v1_full275_20260724.analysis.json`

Key SHA-256 digests:

| Artifact | SHA-256 |
|---|---|
| full JSONL | `2afb2265c2e1368158e87f9338a5d9caae9aa1bf07e8d7da16f9b3820225203f` |
| run manifest | `a4de00d18897ff75e425bdf5815cb29f52f69956e9110ca00e30cfd85ba4f28c` |
| summary | `3b0b7ca455752201851bfba485049f58ce831aa2adaa13c197844d53f26265fc` |
| analysis | `700bf6ec350b0c9669e41e0e40f36f15d6f949e5a7b1b837921f840d18944513` |

Commands:

```bash
PYTHONPATH=/root/claimforge-benchmark \
/root/.cache/claimforge/venvs/imlvit-07dd2be/bin/python \
  -m eval.opensource.run_imlvit \
  --repo-root /root/claimforge-benchmark \
  --run-id imlvit_cat_protocol_mouse_canonical_v1_full275_20260724 \
  --device cuda:0 --bootstrap-samples 1000 --fail-fast

PYTHONPATH=/root/claimforge-benchmark \
/root/.cache/claimforge/venvs/imlvit-07dd2be/bin/python \
  -m eval.opensource.analyze_imlvit_run \
  --repo-root /root/claimforge-benchmark \
  --run-id imlvit_cat_protocol_mouse_canonical_v1_full275_20260724 \
  --bootstrap-iterations 1000 \
  --bootstrap-seed 20260724 \
  --histogram-bins 65536
```

## 13. Conclusion and next method

IML-ViT is a useful positive-but-incomplete transfer result. It localizes some
small generated objects very well and is materially stronger than the three
weakest completed baselines. Its low median AP, 47.6% zero-F1 rate, background-
correlated masks, and strong edit-size dependence prevent treating it as a
reliable general solution.

The next frozen local-manipulation baseline is **HiFi-IFDL**, using its
general-forgery checkpoint covering GAN/diffusion content. Unlike IML-ViT,
HiFi-IFDL provides native T1 and T2 outputs, so both benchmark tasks should be
evaluated without deriving either score from the other.
