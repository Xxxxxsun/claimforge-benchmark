# PSCC-Net on the canonical mouse set (2026-07-24)

## 1. Status and headline

The fifth publicly reproducible research baseline is complete. The official
PSCC-Net checkpoint ran successfully on all 275 matched tasks, or 550
canonical images, with no errors and without resizing the inputs.

PSCC-Net does not transfer into an operational detector for CLAIMFORGE's tiny
local diffusion insertions:

- T1 image AUROC is **0.50408**, with paired-task bootstrap 95% CI
  **[0.50134, 0.50786]**;
- T1 average precision is **0.50731**, against a 0.5 balanced-set baseline;
- the official strict `> 0.5` decision gives 12 TP, 12 FP, 263 FN, and 263 TN,
  so accuracy and balanced accuracy are both **0.50000**;
- native macro pixel AP is **0.01528**, with a per-image median of
  **0.00184**;
- native macro and micro pixel F1 at the official threshold are
  **0.000073** and **0.002297**; and
- **0/275** predicted masks overlap the recorded edit box at IoU greater than
  0.3.

There is a very small paired directional response: the forged score is larger
than its matched real score in 156/275 pairs. That shift is statistically
detectable by an exact sign test (`p=0.02975`) but practically negligible. The
mean forged-minus-real probability is only `0.0003317`, its bootstrap interval
includes zero, and no pair changes its fixed-threshold T1 decision after the
edit.

The independent analyzer validated 4,125 unique files and reproduced the
complete RGB decode, float preprocessing, four progressive probability maps,
native interpolation, classification softmax, decisions, masks, and T1/T2
metrics. This is therefore a complete zero-shot failure result, not an
incomplete run or a silent score-direction error.

The conclusion is checkpoint- and threat-model-specific. It does not establish
that the PSCC-Net architecture fails on its original manipulation benchmarks
or that every retrained PSCC variant fails on generative edits.

## 2. Pinned method and official assets

The run uses the authors'
[PSCC-Net repository](https://github.com/proteus1991/PSCC-Net) at commit
`53e5ff77d8dc5feddda060cd085f9b765761f816`. The repository is MIT licensed.
The method is *Progressive Spatio-Channel Correlation Network for Image
Manipulation Detection and Localization* (IEEE TCSVT 2022).

The repository contains one released synthetic-pretrained model bundle. All
three task checkpoints are required:

| Component | Bytes | SHA-256 |
|---|---:|---|
| HRNet feature extractor | 8,305,545 | `d3b21edc4930187a6801cc818bd7b999fb5d8078d8f2e2193572e91ea5160096` |
| NLC localization head | 2,900,709 | `11ea3461253cf059b299ad4b6b89008485f94a2d2b2da83ec28c2282a095b00b` |
| DetectionHead classifier | 3,739,969 | `a17581e8a3489a360257a266ca9b2db1b7c9b43337fbcc4aeb8d751f593f66f5` |

The registered bundle SHA-256 is
`893626e154e5a3c16322e845a0e8c775029f88a5742de6875818c69f66459560`.
The three state dictionaries contain the exact official `module.` keys and
load with `strict=True`, with zero missing or unexpected entries.

`models/hrnet_w18_small_v2.pth` is only the constructor-time ImageNet
initialization, not a substitute for the task checkpoint. Its 16,012,341-byte
file is also pinned at SHA-256
`06924c741ea8c076a569d5e164aa628910a72020800e4a4945e8b40b241ce5cb`;
the full official HRNet task state overwrites it before inference.

The complete model has 3,667,942 parameters. The run does not substitute an
IMDL-BenCo retraining or select a checkpoint after observing CLAIMFORGE.

## 3. Exact inference contract

The adapter follows the original test path:

1. Decode with `imageio.v2.imread` in RGB order.
2. For RGBA inputs, composite onto white using the official float32 formula.
3. Convert uint8 RGB to contiguous float32 CHW and divide by 255.
4. Apply no ImageNet mean/std normalization.
5. Apply no input resize, crop, letterbox, re-encoding, or test-time
   augmentation. Each canonical image is evaluated at its original size with
   batch size one.
6. Run the HRNet-W18-small-v2 feature extractor.
7. Run the NLC head, which returns four already-sigmoided probability maps at
   256×256, 128×128, 64×64, and 32×32.
8. Fix the primary T2 output to `mask1`, the first and finest progressive map,
   exactly as in the official test code. No second sigmoid is applied.
9. Restore `mask1` to the input dimensions using bilinear interpolation with
   `align_corners=True`.
10. Save the native continuous map as float32 NPY. The PNG is only the strict
    `probability > 0.5` binary mask and is not used to calculate AP.
11. Run the independent two-class DetectionHead. The native T1 score is
    `softmax(logits)[1]`, where class 1 means forged.
12. Use strict `score > 0.5` and `mask > 0.5` for the released operating
    point.

The original source uses the removed NumPy `np.int` alias during HRNet
construction. The adapter installs an in-memory compatibility alias to built-in
`int`; it does not patch the pinned third-party checkout or alter numerical
inference.

The primary metrics do not reproduce the paper's historical practice of
turning a per-image pixel AUC below 0.5 into `1-AUC`. CLAIMFORGE has a defined
positive direction—high values mean manipulated—so a post-hoc GT-dependent
flip would be invalid. The paper also reports some threshold-selected
localization metrics; this evaluation keeps the released 0.5 threshold fixed.

## 4. Why the method can be strong on its original task

PSCC-Net is not a simple pixel classifier. It combines:

- a high-resolution backbone that retains local spatial detail;
- progressive non-local correlation modules that compare spatial positions
  and feature channels at four scales;
- coarse-to-fine mask refinement, where one stage guides the next; and
- a separate image classification head rather than a max or mean derived from
  the final heatmap.

This design is well matched to copy-move, classical splice, and removal traces:
duplicated regions create long-range spatial correlations, pasted regions can
break feature consistency, and progressive multi-scale reasoning can refine
their boundaries. The released sample set reflects those manipulation types,
and the pinned adapter classifies all six authentic and all six manipulated
official samples correctly.

CLAIMFORGE is a different shift. Its median edit covers only 0.1126% of the
image; most pixels and nearly all global scene structure are identical within
each pair. A well-blended diffusion-generated mouse need not create the
copy-move correspondences or classical splice/removal boundary statistics in
the checkpoint's training distribution. The observed near-identity of paired
outputs is consistent with the shared background dominating both heads. This
is an evidence-supported explanation, not a causal architectural ablation.

## 5. Sanity checks before the full run

The official 12-image sample folder was evaluated first:

- authentic: 6/6 predicted authentic;
- copy-move, removal, and splice: 6/6 predicted forged;
- authentic forged-class probabilities range from `0.00144` to `0.00391`;
- manipulated probabilities range from `0.99971` to `1.00000`; and
- all primary mask values are finite and within `[0,1]`.

One canonical pair and then five fixed canonical pairs completed before the
full run. The five-pair smoke already showed the transfer problem: AUROC
`0.48`, paired ranking `0.40`, and zero positive localization pixels on all
five forged images.

The 10 shared smoke/full images match bit-for-bit on the classification score,
logits, primary 256 map, native float map, and binary mask. This rules out
run-order drift over the shared prefix.

## 6. T1 image-level result

| Image-level metric | Estimate | Pair-bootstrap 95% CI |
|---|---:|---:|
| AUROC | **0.50408** | [0.50134, 0.50786] |
| Average precision | 0.50731 | [0.50652, 0.51464] |
| TPR at FPR at most 5% | 0.04727 | [0.04000, 0.05818] |
| Accuracy at 0.5 | 0.50000 | [0.50000, 0.50000] |
| Balanced accuracy at 0.5 | 0.50000 | [0.50000, 0.50000] |
| Positive-class F1 at 0.5 | 0.08027 | [0.04181, 0.12141] |
| Paired forged-greater-than-real rate | 0.56727 | [0.51273, 0.62909] |
| Mean paired score delta | 0.0003317 | [-0.0000072, 0.0006989] |

The fixed-threshold confusion matrix is symmetric:

| TP | FP | FN | TN |
|---:|---:|---:|---:|
| 12 | 12 | 263 | 263 |

The same 12 matched pairs are positive before and after editing. The other 263
pairs are negative before and after editing. Consequently, the edit changes
**0/275** deployed decisions even though 156 forged probabilities are
numerically larger than their matched real probabilities.

The real and forged probabilities have Pearson correlation `0.999888`. The
paired delta median is only `0.0000214`. These facts explain why the paired
sign test can detect a tiny direction while AUROC, accuracy, and deployable
decisions remain ineffective.

The AUROC bootstrap interval being narrowly above 0.5 should not be described
as meaningful detection. The point estimate differs from chance by 0.004,
TPR at the operational low-FPR region is below 5%, and the paired mean-effect
interval includes zero.

## 7. T2 localization result

| Native-resolution metric | Estimate | Pair-bootstrap 95% CI |
|---|---:|---:|
| Macro pixel AP | **0.015278** | [0.008008, 0.024046] |
| Median per-image pixel AP | 0.001841 | — |
| Macro pixel F1 at 0.5 | 0.000073 | [0.000000, 0.000184] |
| Macro pixel IoU at 0.5 | 0.000036 | [0.000000, 0.000092] |
| Micro pixel F1 at 0.5 | 0.002297 | [0.000000, 0.005860] |
| Micro pixel IoU at 0.5 | 0.001150 | [0.000000, 0.002939] |
| Edit-box hits at 0.5 | 0/275 | — |

At the released threshold:

- 21/275 forged images contain any predicted positive pixel;
- only 3/275 masks hit even one GT-positive pixel;
- 272/275 per-image F1 values are exactly zero;
- micro precision is `0.00133` and micro recall is `0.00829`; and
- the best per-image fixed-threshold F1 is only `0.01074`.

The result is not caused by native resizing. The stored 256-space diagnostic
has macro AP `0.01459` and macro F1 `0.000074`, effectively the same conclusion
as native space.

It is also not only a calibration failure. Ground-truth-dependent diagnostics,
which are not eligible for primary reporting, give:

- approximate best single global threshold `0.16236`, with micro F1
  `0.003881`;
- a separately optimized threshold for every forged image, with mean F1
  `0.02506` and median F1 `0.00551`.

Even the per-image oracle remains very weak. No test-selected threshold
replaces the registered 0.5 result.

## 8. Pristine-image behavior and shared-background dominance

The mean positive area is almost identical before and after editing:

| Statistic | Real | Forged |
|---|---:|---:|
| Mean positive area at 0.5 | 0.8265% | 0.8272% |
| Images with a nonempty mask | 22/275 | 21/275 |
| Maximum positive area | 96.45% | 96.43% |

The paired real/forged positive-area correlation is `0.999919`. Most images
produce an empty mask, while a few shared scene outliers trigger very large
regions in both members of a pair:

- 253/275 real masks are empty;
- the largest real mask covers 96.45% of the image;
- the single largest real outlier contributes about 56.4% of all real
  false-positive pixels; and
- the five largest contribute about 95.9%.

Mean false-positive area therefore hides a long-tailed failure mode. A low
average here cannot be read as clean localization because it is paired with
near-zero recall and rare catastrophic shared-background responses.

## 9. Diagnostic slices

### Domain

| Domain | Pairs | T1 AUROC | Paired rank | Pixel AP | Micro IoU | Real FP area |
|---|---:|---:|---:|---:|---:|---:|
| lodging | 147 | 0.50363 | 0.55782 | 0.01495 | 0.000660 | 1.3062% |
| restaurant | 128 | 0.50519 | 0.57812 | 0.01566 | 0.002763 | 0.2757% |

Neither domain has useful T1 or T2 performance. Lodging has a higher mean real
positive area, but that descriptive difference is driven by sparse outliers
and has a wide bootstrap interval. It is not evidence of a stable domain
effect.

### Edit-size quintiles

| Quintile | Pairs | Median edit fraction | Pixel AP | Macro F1@0.5 |
|---|---:|---:|---:|---:|
| Q1, smallest | 55 | 0.05347% | 0.00450 | 0.000000 |
| Q2 | 55 | 0.07780% | 0.00249 | 0.000031 |
| Q3 | 55 | 0.11264% | 0.00372 | 0.000137 |
| Q4 | 55 | 0.16293% | 0.01138 | 0.000195 |
| Q5, largest | 55 | 0.37441% | 0.05430 | 0.000000 |

Raw pixel AP correlates with edit fraction (`Spearman rho=0.618`). This is not
enough to claim that PSCC-Net becomes reliably more sensitive to larger edits:
the random AP baseline itself equals positive-pixel prevalence. After dividing
AP by edit fraction, the correlation is `rho=0.011` (`p=0.856`), while fixed
F1 also has essentially no area relationship (`rho=-0.011`, `p=0.854`).

The safe conclusion is that raw AP rises with mask prevalence, but neither
normalized ranking gain nor fixed-threshold performance shows a stable
edit-size trend.

## 10. Comparison with completed research baselines

All five full runs use the identical 550-image canonical input manifest and
the exact-difference native GT. T1 availability is not uniform: CAT-Net has no
native image head, MVSS uses its paper-defined map global maximum, and
PSCC-Net has an independent classification head.

| Method | Native T1 | T1 AUROC | Paired rank | Pixel AP | Macro F1 | Micro F1 | Real FP area |
|---|---|---:|---:|---:|---:|---:|---:|
| TruFor | yes | **0.81790** | **0.90909** | 0.57989 | **0.50010** | **0.33056** | 1.7764% |
| CAT-Net v2 | no | — | — | **0.61234** | 0.47481 | 0.30706 | 0.4359% |
| MaskCLIP | yes | 0.50728 | 0.64727 | 0.04740 | 0.00564 | 0.00622 | 0.0595% |
| MVSS-Net | map GMP | 0.50576 | 0.42182 | 0.03415 | 0.01962 | 0.00735 | 3.2887% |
| PSCC-Net | yes, independent head | 0.50408 | 0.56727 | 0.01528 | 0.00007 | 0.00230 | 0.8265% |

TruFor remains the only completed method with strong T1 separation. CAT-Net
and TruFor remain far ahead for T2: CAT-Net has the higher continuous pixel AP,
while TruFor has the higher fixed-threshold F1. No direct paired method-
difference bootstrap has yet established that one is significantly better
than the other.

MaskCLIP, MVSS-Net, and PSCC-Net are all globally near chance for T1 and weak
for T2, but they fail differently. MVSS fires broadly on both classes;
MaskCLIP and PSCC are usually empty, with PSCC additionally showing rare
large shared-background masks. A lower real FP area cannot be treated as
better localization when recall is nearly zero.

This is a cross-domain robustness table over off-the-shelf official
checkpoints, not a controlled architecture leaderboard with matched training
data.

## 11. Validation and audit

The final result is backed by the following checks:

- The official three-checkpoint bundle loaded safely and strictly.
- The official sample regression produced 12/12 correct T1 labels.
- Preflight and five-pair smoke runs completed before the full run.
- All 550 full-run rows are unique, successful, and tied to the immutable
  run-manifest fingerprint; there are zero duplicate or recovered histories.
- The independent analyzer validated 4,125 files: 550 canonical JPEGs, 275
  forged GT masks, and 3,300 generated artifacts.
- Every file hash, array dtype, shape, probability range, RGB tensor hash,
  softmax score, strict decision, native interpolation, mask, and recorded
  localization metric was independently reproduced.
- The 31 PSCC-specific runner, metric, and analyzer tests pass.
- The complete repository suite passes with 113 tests and 4 environment-gated
  skips in the existing NumPy-1/OpenCV-compatible MVSS test environment.

The full output contains 3,300 model artifacts totaling 3,730,881,672 bytes.
Median model-forward latency is 149.1 ms on an NVIDIA L20Z, with peak allocated
CUDA memory of 8,435,241,984 bytes. Timing excludes image decoding, artifact
serialization, and metric calculation.

## 12. Reproducible artifacts

Primary files:

- runner: `eval/opensource/run_psccnet.py`
- metric implementation: `eval/opensource/psccnet_metrics.py`
- independent analyzer: `eval/opensource/analyze_psccnet_run.py`
- full results:
  `results/opensource/psccnet/psccnet_official_mouse_canonical_v1_full275_20260724.jsonl`
- immutable run manifest:
  `results/opensource/psccnet/psccnet_official_mouse_canonical_v1_full275_20260724.run_manifest.json`
- run summary:
  `results/opensource/psccnet/psccnet_official_mouse_canonical_v1_full275_20260724.summary.json`
- audited paired analysis:
  `results/opensource/psccnet/psccnet_official_mouse_canonical_v1_full275_20260724.analysis.json`

Key SHA-256 digests:

| Artifact | SHA-256 |
|---|---|
| full JSONL | `8952da5de8a4a7cd58da39420d5219fb887a8ecdc7a3e25593beb65a385f0e7e` |
| run manifest | `bd58d7c974a3cb85486551afb52db6db208993ca29d4dc264b4a317eecdc2f26` |
| summary | `c1e2efb6ef30c349217fe40623edf2f65500e0beeff37c0cf0e1dc440781d692` |
| analysis | `6ba2c33b95ae429f4270383b317677aa7406f4e2dc4ae2fb382f8ca719bd32ae` |

Commands:

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=/root/claimforge-benchmark \
/root/.cache/claimforge/venvs/psccnet/bin/python \
  -m eval.opensource.run_psccnet \
  --repo-root /root/claimforge-benchmark \
  --run-id psccnet_official_mouse_canonical_v1_full275_20260724 \
  --device cuda:0 --fail-fast

PYTHONPATH=/root/claimforge-benchmark \
/root/.cache/claimforge/venvs/psccnet/bin/python \
  eval/opensource/analyze_psccnet_run.py \
  --repo-root /root/claimforge-benchmark \
  --run-id psccnet_official_mouse_canonical_v1_full275_20260724 \
  --bootstrap-iterations 1000 \
  --bootstrap-seed 20260724
```

## 13. Recommended next method

The next frozen local-manipulation baseline is **IML-ViT**, using the
pre-registered CAT-protocol checkpoint. It provides a modern transformer
localizer and tests whether CAT-Net's strong result persists in a different
architecture trained under a closely related manipulation-localization
protocol. IML-ViT has no native independent image classifier, so its primary
result will be T2 and T1 will remain N/A rather than being synthesized from a
heatmap.
