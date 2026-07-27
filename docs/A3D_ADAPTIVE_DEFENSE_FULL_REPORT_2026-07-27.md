# A3D adaptive defense: method, implementation, and complete results

Date: 2026-07-27

## Executive summary

A3D (Adaptive Anomaly-Aware Defense) is a category-agnostic, test-time
coarse-to-fine wrapper around a frozen TruFor forensic detector. It was designed
for CLAIMFORGE's difficult regime: a generated object may occupy less than
0.1% of an otherwise real image, so whole-image evidence is diluted.

The frozen detector makes one full-image pass, ranks a deterministic
native-resolution crop grid from that pass, evaluates only four crops, and
combines global and strongest-local evidence by equal-weight logit averaging:

```text
s_fused = sigmoid((logit(s_full) + logit(max_i s_crop_i)) / 2)
```

The current fixed decision threshold is `0.6353510120379108`. It was calibrated
from 80 known-real mouse development images and is transferred unchanged to
mouse test, cat, trash-can, and generated-full images.

The main findings are:

- On mouse 275 pairs, fused A3D raises AUROC from `0.8180` to `0.8796` and
  TPR@5%FPR from `0.4436` to `0.6655`.
- On cat 251 pairs, it raises AUROC from `0.9693` to `0.9808` and TPR@5%FPR
  from `0.9004` to `0.9522`.
- On trash-can 250 pairs, it raises AUROC from `0.9291` to `0.9318` and
  TPR@5%FPR from `0.7280` to `0.7880`. The AUROC gain is not statistically
  conclusive, but the fixed fusion avoids the large regression of local-only.
- At the fixed mouse-reference threshold, cat and trash-can reach respectively
  `3.98% / 94.82%` and `4.80% / 78.80%` FPR/TPR.
- JPEG quality-90 recompression is a major weakness: mouse fused AUROC falls
  to `0.6101`.
- On the 807 images in the three final generated-full manifests, fused A3D
  detects only `14.13%`; local-only detects `24.41%`. A3D is therefore useful
  for locally composited forensic inconsistency, not a universal fully
  generated image detector.

No CLAIMFORGE image was used to train or fine-tune the frozen TruFor weights.
Inference also uses no category label, target box, mask, prompt, or test label.
However, equal-logit fusion was finalized after aggregate cat/trash results
were inspected. Cat/trash fused results are consequently post-selection
cross-object validation, not a pristine held-out estimate. The rule is now
fully specified and must be evaluated unchanged on a future category for a
strict blind claim.

## 1. Evaluated data and completeness

### 1.1 Paired spliced benchmark

| Set | Pairs | Images | Dev pairs | Hash-test pairs | Completed |
|---|---:|---:|---:|---:|---:|
| Mouse | 275 | 550 | 80 | 195 | 550/550 |
| Cat | 251 | 502 | 97 | 154 | 502/502 |
| Trash-can | 250 | 500 | 87 | 163 | 500/500 |
| Cat + trash-can | 501 | 1,002 | 184 | 317 | 1,002/1,002 |
| All three categories | 776 | 1,552 | 264 | 512 | 1,552/1,552 |

Every durable JSONL above has one latest successful row per expected image,
no missing ID, no failed latest row, and no superseded physical row. Cat and
trash-can canonical task IDs exactly match their final 251- and 250-pair
selection manifests.

The all-three aggregate mixes mouse's existing native canonical input condition
with the explicitly JPEG-Q95 canonicalized cat/trash condition. It is included
as a descriptive roll-up, not as the primary controlled comparison.

### 1.2 Generated-full data

The generated-full evaluation contains the three final generation manifests:

| Category | Final-manifest images | Completed |
|---|---:|---:|
| Mouse | 275 | 275/275 |
| Cat | 272 | 272/272 |
| Trash-can | 260 | 260/260 |
| **Total** | **807** | **807/807** |

After the latest repository pull, `generated_full_images/` contains 893
recursive image files in seven directories. The other 86 files are mouse QC
retry, pilot, or wrong-ID backup intermediates:

| Non-final directory type | Images |
|---|---:|
| QC retry 1 | 54 |
| QC retry 2 | 21 |
| QC retry 3 pilot | 4 |
| Wrong reused-ID backup | 7 |
| **Total excluded intermediates** | **86** |

The reported generated-full result is therefore exactly the 807-image final
manifest set, not every recursive file currently present. This distinction is
intentional and now recorded in the aggregate result JSON.

## 2. Threat model and design objective

The primary CLAIMFORGE threat is a generated object spliced back into an
otherwise real background. The edited region can be extremely small:

- mouse median edit fraction: `0.11264%`;
- smallest mouse quintile median: `0.05347%`;
- cat median: `0.75637%`;
- trash-can median: `1.26413%`.

A whole-image forensic network must compress evidence from millions of pixels
into one score. Small manipulated regions are therefore underweighted. A3D
spends additional compute only where the frozen model's full-resolution map
already contains anomalous evidence.

The design constraints are:

1. no target-category classifier or object detector;
2. no mask, box, prompt, or label at inference;
3. no test-time weight update;
4. fixed crop geometry and compute budget;
5. deterministic proposal and tie-breaking rules;
6. one separately calibrated decision threshold.

## 3. Frozen base detector and training-data audit

The evaluated checkpoint is:

```text
/root/.cache/claimforge/checkpoints/trufor/weights/trufor.pth.tar
SHA-256 ac1d90e329a72e0d66e8665e123a19e94bfae3209c3ef8a4f9ca3b91578c7844
```

The pinned TruFor source is:

```text
/root/.cache/claimforge/third_party/TruFor
commit ae54475df6f41a491d7615100feb19263dec13f7
```

The official training README and phase-2/phase-3 configs list:

```text
IMD, FantasticReality, CASIA 2.0 revised, tampCOCO, compRAISE
```

Both published configs contain:

```yaml
DATASET:
  TRAIN: [IMD, FR, CA, COCO, RAISE]
  VALID: [IMD, FR, CA, COCO, RAISE]
```

The following claims are supported:

- The released TruFor checkpoint is frozen.
- This project performs no TruFor training or fine-tuning.
- Exact CLAIMFORGE mouse, cat, and trash-can benchmark images are not listed
  among TruFor's training datasets and were created after the released model.
- Cat/trash images were never used to update weights.

The following stronger claim is not supported:

- We cannot claim that TruFor has never seen the semantic concepts “cat” or
  “trash can.” Its ImageNet-preprocessed backbone and COCO-derived training data
  are broad and may contain these concepts.

Accordingly, the correct description is “target-benchmark-data-free frozen
detector,” not “semantically category-unseen pretrained model.”

TruFor's upstream license is informational and nonprofit only. Its terms must
be checked before use beyond research evaluation.

## 4. Detection algorithm

### 4.1 Full-image pass

For an RGB image `x`, frozen TruFor returns:

- a scalar manipulation probability `s_full`;
- a dense manipulation map `M`;
- a dense reliability map `R`.

### 4.2 Deterministic proposal grid

The native image is covered by `512 x 512` windows at stride `384`. Axis starts
always include the right and bottom boundary, so the entire image is covered.
Images smaller than 512 pixels on an axis use the available extent.

For crop window `B`, the raw proposal score is the mean of the top `0.1%` of
`M[B]`. Reliability weighting is:

```text
p_raw(B) = mean(top_0.1%(M[B]))
p(B)     = p_raw(B) * mean(0.25 + 0.75 * R[B])
```

All grid crops are scored from the already-computed maps; no crop inference is
needed at this stage. Proposals are ordered by descending `p(B)`, then by
ascending deterministic grid index. The first four are selected.

### 4.3 Local inference and fusion

Frozen TruFor is rerun on each of the four selected native-resolution crops.
The local branch is:

```text
s_local = max(s_crop_1, ..., s_crop_4)
```

The primary A3D image score is:

```text
logit(p) = log(clip(p, 1e-6, 1 - 1e-6) /
               (1 - clip(p, 1e-6, 1 - 1e-6)))

s_fused = sigmoid(0.5 * (logit(s_full) + logit(s_local)))
```

Equal logit fusion is symmetric and idempotent. Compared with a probability
mean, it combines evidence in the detector's natural odds space. Compared with
local-only, it prevents a single locally anomalous real crop from completely
overriding globally consistent evidence.

### 4.4 Localization output

The stable primary localization output is the full-image TruFor map. Three
local diagnostics are retained:

- `a3d_top1`: highest-scoring crop map;
- `a3d_top2`: pixelwise maximum of the two highest-scoring crop maps;
- `a3d_all4`: pixelwise maximum of all four crop maps.

Crop maps are placed back at their native coordinates with zeros elsewhere.
Overlaps use pixelwise maximum. Crop-map ranking uses descending crop
classification score and ascending grid index as the tie-breaker.

Local top-1 strongly improves mouse localization, but degrades trash-can and
some larger edits. It is therefore an informative diagnostic, not the universal
primary localization rule.

### 4.5 Pseudocode

```text
function A3D(image):
    s_full, M, R = TruFor(image)
    boxes = grid(image.width, image.height, side=512, stride=384)
    proposals = [(proposal_score(M, R, box), grid_index, box)
                 for box in boxes]
    selected = sort(proposals, by=(-score, grid_index))[:4]

    crop_outputs = [TruFor(image.crop(box)) for box in selected]
    s_local = max(output.score for output in crop_outputs)
    s_fused = sigmoid((logit(s_full) + logit(s_local)) / 2)

    decision = (s_fused >= 0.6353510120379108)
    return s_full, s_local, s_fused, decision, M, diagnostics
```

## 5. Calibration and blindness protocol

### 5.1 Current fixed threshold

Task IDs are deterministically hashed into mouse development and test splits.
The 80 development real-image fused scores are the only values used for the
deployed threshold:

```text
q = empirical 95th percentile with method="higher"
threshold = next representable float above q
          = 0.6353510120379108
```

Using a value strictly above the empirical quantile gives a development
empirical FPR of `3.75%`. The threshold is not changed per category or image.

### 5.2 What is and is not blind

| Property | Status | Explanation |
|---|---|---|
| No CLAIMFORGE weight training/fine-tuning | Yes | Frozen released TruFor checkpoint |
| Per-image inference label-blind | Yes | No class label, GT, box, mask, or prompt used |
| Threshold target-label-free for cat/trash | Yes | Uses only 80 mouse dev-real scores |
| Fusion strictly held out from cat/trash | No | Candidate fusion rules were compared after aggregate cat/trash inspection |
| Cat/trash result usable as a measured result | Yes | Scores and metrics are valid for these files |
| Cat/trash result usable as pristine unseen-category evidence | No | Fusion post-selection introduces evaluation feedback |

This is the precise meaning of the earlier statement that “the fusion rule was
selected after looking at the new results.” The model did not train on cat or
trash images, and labels never enter inference. Nevertheless, aggregate
cat/trash labels influenced which already-computed full/local score combination
was reported as primary.

### 5.3 Recommended future strict-blind protocol

For the next object category:

1. freeze checkpoint, JPEG preprocessing, grid, proposal budget, equal-logit
   fusion, localization rule, and fixed threshold before receiving labels;
2. optionally receive 80-100 known-real reference images only;
3. use reference-real scores only for a target-FPR quantile;
4. anonymize and shuffle every evaluation image;
5. write and hash `predictions.jsonl` before opening the label key;
6. report AUROC, AP, TPR@1%FPR, TPR@5%FPR, and the fixed operating point once;
7. treat any subsequent method change as a new version requiring another
   unseen category.

An arbitrary unlabeled real/fake mixture is not sufficient for trustworthy
5% real-image FPR calibration without an explicit contamination model.

## 6. Image-level detection results

All accuracy, balanced accuracy, and F1 values in this section use score
threshold `0.5`; they are included for completeness. Ranking metrics and the
separately calibrated fixed operating point are more meaningful.

### 6.1 Frozen full-image TruFor

| Set | AUROC | AP | Acc. | Bal. acc. | F1 | TPR@1% | TPR@5% | Pair rank |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Mouse | 0.8180 | 0.8394 | 0.7109 | 0.7109 | 0.6294 | 0.2000 | 0.4436 | 0.9055 |
| Cat | 0.9693 | 0.9760 | 0.9183 | 0.9183 | 0.9172 | 0.7809 | 0.9004 | 0.9880 |
| Trash-can | 0.9291 | 0.9337 | 0.8500 | 0.8500 | 0.8366 | 0.3680 | 0.7280 | 0.9800 |
| Cat + trash-can | 0.9493 | 0.9568 | 0.8842 | 0.8842 | 0.8784 | 0.6028 | 0.8184 | 0.9840 |
| All three, mixed preprocessing | 0.9033 | 0.9203 | 0.8228 | 0.8228 | 0.8012 | 0.4240 | 0.6817 | 0.9562 |

### 6.2 Local-only A3D ablation

| Set | AUROC | AP | Acc. | Bal. acc. | F1 | TPR@1% | TPR@5% | Pair rank |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Mouse | 0.8798 | 0.9044 | 0.8055 | 0.8055 | 0.8065 | 0.4655 | 0.6727 | 0.8436 |
| Cat | 0.9791 | 0.9803 | 0.8924 | 0.8924 | 0.9004 | 0.7968 | 0.9402 | 0.9880 |
| Trash-can | 0.9051 | 0.9127 | 0.8180 | 0.8180 | 0.8205 | 0.4000 | 0.7040 | 0.9200 |
| Cat + trash-can | 0.9427 | 0.9499 | 0.8553 | 0.8553 | 0.8618 | 0.5988 | 0.8244 | 0.9541 |
| All three, mixed preprocessing | 0.9207 | 0.9345 | 0.8376 | 0.8376 | 0.8427 | 0.5515 | 0.7784 | 0.9149 |

Local-only is strongest on tiny mouse edits but regresses on trash-can and has
lower paired ranking accuracy. This instability motivated fusion.

### 6.3 Current primary equal-logit fusion

| Set | AUROC | AP | Acc. | Bal. acc. | F1 | TPR@1% | TPR@5% | Pair rank |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Mouse | 0.8796 | 0.9019 | 0.8364 | 0.8364 | 0.8249 | 0.3418 | 0.6655 | 0.9091 |
| Cat | 0.9808 | 0.9845 | 0.9343 | 0.9343 | 0.9354 | 0.8207 | 0.9522 | 0.9920 |
| Trash-can | 0.9318 | 0.9381 | 0.8760 | 0.8760 | 0.8714 | 0.4280 | 0.7880 | 0.9640 |
| Cat + trash-can | 0.9567 | 0.9634 | 0.9052 | 0.9052 | 0.9043 | 0.6248 | 0.8743 | 0.9780 |
| All three, mixed preprocessing | 0.9298 | 0.9434 | 0.8808 | 0.8808 | 0.8772 | 0.5245 | 0.8144 | 0.9536 |

Fused A3D is slightly below local-only on mouse AUROC and TPR@5%FPR, but it
improves mouse accuracy, F1, and pair ranking while avoiding the trash-can
local-only regression. It is the most stable single category-agnostic rule.

### 6.4 Deterministic hash-test split

| Set | Pairs | Fused AUROC | Fused AP | Fused Acc. | Fused F1 | TPR@1% | TPR@5% |
|---|---:|---:|---:|---:|---:|---:|---:|
| Mouse | 195 | 0.8737 | 0.8974 | 0.8385 | 0.8264 | 0.2359 | 0.6667 |
| Cat | 154 | 0.9744 | 0.9787 | 0.9221 | 0.9236 | 0.6753 | 0.9156 |
| Trash-can | 163 | 0.9211 | 0.9270 | 0.8742 | 0.8690 | 0.2945 | 0.7178 |
| Cat + trash-can | 317 | 0.9470 | 0.9542 | 0.8975 | 0.8963 | 0.4795 | 0.8139 |
| All three, mixed preprocessing | 512 | 0.9188 | 0.9336 | 0.8750 | 0.8707 | 0.3867 | 0.7480 |

### 6.5 Fixed mouse-reference threshold

| Evaluation | FPR | TPR |
|---|---:|---:|
| Mouse, all | 5.09% | 71.64% |
| Mouse, hash-test | 5.64% | 72.31% |
| Cat, all | 3.98% | 94.82% |
| Cat, hash-test | 5.19% | 93.51% |
| Trash-can, all | 4.80% | 78.80% |
| Trash-can, hash-test | 5.52% | 77.91% |
| Cat + trash-can, all | 4.39% | 86.83% |
| Cat + trash-can, hash-test | 5.36% | 85.49% |
| All three, all | 4.64% | 81.44% |
| All three, hash-test | 5.47% | 80.47% |

The transferred threshold remains close to its 5% target FPR across objects.

## 7. Localization and proposal results

Localization metrics are macro averages over forged images at pixel threshold
`0.5`. Pixel AP is threshold-free. `A3D all4` does not store pixel AP because
that map was retained only as a binary diagnostic.

### 7.1 Full-image map: stable primary localization

| Set | Pixel AP | F1 | IoU | MCC |
|---|---:|---:|---:|---:|
| Mouse | 0.5798 | 0.5001 | 0.3741 | 0.5305 |
| Cat | 0.9133 | 0.8353 | 0.7390 | 0.8412 |
| Trash-can | 0.8125 | 0.7250 | 0.6008 | 0.7388 |
| Cat + trash-can | 0.8630 | 0.7802 | 0.6701 | 0.7902 |
| All three, mixed preprocessing | 0.7626 | 0.6809 | 0.5652 | 0.6983 |

### 7.2 Local-map diagnostics

| Set | Strategy | Pixel AP | F1 | IoU | MCC |
|---|---|---:|---:|---:|---:|
| Mouse | Top-1 | 0.7016 | 0.5599 | 0.4271 | 0.5936 |
| Mouse | Top-2 | 0.6316 | 0.5126 | 0.3862 | 0.5431 |
| Mouse | All-4 | N/A | 0.3821 | 0.2726 | 0.4150 |
| Cat | Top-1 | 0.8810 | 0.8393 | 0.7533 | 0.8489 |
| Cat | Top-2 | 0.9176 | 0.8578 | 0.7739 | 0.8636 |
| Cat | All-4 | N/A | 0.8228 | 0.7286 | 0.8300 |
| Trash-can | Top-1 | 0.7454 | 0.6568 | 0.5422 | 0.6856 |
| Trash-can | Top-2 | 0.8041 | 0.7072 | 0.5906 | 0.7257 |
| Trash-can | All-4 | N/A | 0.7015 | 0.5803 | 0.7142 |
| Cat + trash-can | Top-1 | 0.8133 | 0.7482 | 0.6480 | 0.7676 |
| Cat + trash-can | Top-2 | 0.8609 | 0.7827 | 0.6824 | 0.7949 |
| Cat + trash-can | All-4 | N/A | 0.7623 | 0.6546 | 0.7724 |

The mouse gain proves that correct zooming can recover tiny edits. The
trash-can regression proves that a zero-filled crop map can discard useful
global evidence on larger edits. The stable output therefore remains full-map.

### 7.3 Proposal coverage

| Set | Budget | Mean target recall | Any-hit rate | Full-cover rate |
|---|---:|---:|---:|---:|
| Mouse | 1 | 0.8914 | 0.9018 | 0.8473 |
| Mouse | 2 | 0.9289 | 0.9345 | 0.9164 |
| Mouse | 4 | 0.9598 | 0.9600 | 0.9527 |
| Cat | 1 | 0.9115 | 0.9920 | 0.6972 |
| Cat | 2 | 0.9606 | 0.9960 | 0.8048 |
| Cat | 4 | 0.9861 | 1.0000 | 0.8765 |
| Trash-can | 1 | 0.8305 | 0.9720 | 0.5800 |
| Trash-can | 2 | 0.9150 | 0.9800 | 0.7120 |
| Trash-can | 4 | 0.9605 | 0.9960 | 0.7520 |

Proposal recall measures whether the chosen crop covers target pixels and is
computed only after inference. It does not affect proposal ranking.

## 8. Edit-size analysis

Mouse gives the cleanest controlled view because its 275 pairs span extremely
small edits.

| Quintile | Median edit % | Full AUROC | Local AUROC | Fused AUROC | Full F1 | Top-1 F1 | Four-crop hit |
|---|---:|---:|---:|---:|---:|---:|---:|
| Q1 | 0.05347 | 0.7226 | 0.8600 | 0.8387 | 0.3468 | 0.4860 | 0.9091 |
| Q2 | 0.07780 | 0.8155 | 0.8762 | 0.8889 | 0.4774 | 0.5502 | 0.9091 |
| Q3 | 0.11264 | 0.8575 | 0.8855 | 0.8926 | 0.5196 | 0.5770 | 0.9818 |
| Q4 | 0.16293 | 0.8489 | 0.8504 | 0.8648 | 0.5306 | 0.5680 | 1.0000 |
| Q5 | 0.37441 | 0.9412 | 0.9383 | 0.9438 | 0.6260 | 0.6181 | 1.0000 |

The local branch is most valuable in Q1-Q3. Q5 is already near saturation and
does not need zooming. Fused A3D preserves full-image performance in Q5.

## 9. Statistical uncertainty

Paired bootstrap uses 2,000 pair-level resamples. Detection intervals below are
fused-minus-full:

| Set / scope | AUROC 95% CI | AP 95% CI | TPR@5% 95% CI |
|---|---:|---:|---:|
| Mouse, all | [0.0448, 0.0788] | [0.0462, 0.0784] | [0.1345, 0.3236] |
| Mouse, Q1 | [0.0674, 0.1636] | [0.0750, 0.1667] | [0.0182, 0.4182] |
| Mouse, Q2-Q5 | [0.0238, 0.0606] | [0.0286, 0.0636] | [0.1227, 0.3136] |
| Cat, all | [0.0028, 0.0211] | [0.0023, 0.0149] | [0.0159, 0.0876] |
| Trash-can, all | [-0.0102, 0.0156] | [-0.0088, 0.0166] | [-0.0361, 0.1280] |
| Cat + trash-can, all | [-0.0000, 0.0156] | [0.0005, 0.0130] | [0.0100, 0.0978] |
| Cat + trash-can, Q1 | [0.0053, 0.0524] | [0.0080, 0.0414] | [0.0198, 0.1782] |

Mouse and cat provide clear evidence of improvement. Trash-can alone is
compatible with either a small gain or a small loss. The combined cat/trash AP
and TPR@5% gains exclude zero, while its AUROC lower bound is effectively zero.

For local top-1 localization minus full-map localization:

| Set | Pixel AP 95% CI | F1 95% CI |
|---|---:|---:|
| Mouse | [0.1003, 0.1449] | [0.0392, 0.0835] |
| Cat | [-0.0583, -0.0064] | [-0.0192, 0.0278] |
| Trash-can | [-0.1001, -0.0354] | [-0.0972, -0.0399] |
| Cat + trash-can | [-0.0701, -0.0284] | [-0.0502, -0.0129] |

This is why local maps are not promoted as the general localization output.

## 10. JPEG-90 stress test

Mouse images were recompressed in memory at JPEG quality 90, 4:2:0
subsampling, before both the full and crop passes. Geometry is unchanged.

| Method | AUROC | AP | Acc. | F1 | TPR@1% | TPR@5% | Pair rank |
|---|---:|---:|---:|---:|---:|---:|---:|
| Full | 0.5662 | 0.5897 | 0.5436 | 0.2507 | 0.0364 | 0.1164 | 0.6545 |
| Local-only | 0.6166 | 0.6478 | 0.5909 | 0.4780 | 0.0873 | 0.1745 | 0.4255 |
| Fused | 0.6101 | 0.6388 | 0.5782 | 0.3763 | 0.0400 | 0.1782 | 0.6400 |

At the unchanged clean-reference threshold, fused JPEG-90 FPR/TPR is
`4.73% / 16.00%` on all mouse pairs and `5.13% / 14.87%` on hash-test.

| Localization | Pixel AP | F1 | IoU | MCC |
|---|---:|---:|---:|---:|
| Full map | 0.2297 | 0.2036 | 0.1426 | 0.2237 |
| Top-1 crop map | 0.2567 | 0.2055 | 0.1527 | 0.2264 |
| Top-2 crop map | 0.2434 | 0.2051 | 0.1489 | 0.2231 |

Fused-minus-full AUROC remains positive with 95% CI `[0.0280, 0.0606]`, but
absolute discrimination is weak. Proposal all-four any-hit falls from `96.00%`
to `77.09%`. JPEG both suppresses full-map proposal evidence and makes selected
crop scores less separable.

## 11. Exploratory diagnostics that led to A3D

These diagnostics use mouse Q1 (55 pairs) and are not deployable headline
results.

### 11.1 Ground-truth-centered oracle zoom

Oracle crops use the target location and therefore measure opportunity, not a
valid defense.

| Scale | AUROC | AP | TPR@5% | Pixel AP | Pixel F1 |
|---|---:|---:|---:|---:|---:|
| Full image | 0.7226 | 0.7295 | 0.2182 | 0.4242 | 0.3468 |
| Oracle square 256 | 0.9144 | 0.9357 | 0.7091 | 0.7042 | 0.5442 |
| Oracle square 512 | 0.9088 | 0.9250 | 0.6727 | 0.6802 | 0.5547 |

This established that tiny-edit dilution, rather than complete lack of
forensic signal, was a major failure source.

### 11.2 Exhaustive scan and proposal-budget diagnostic

| Strategy / score | Crop budget | AUROC | AP | TPR@5% |
|---|---:|---:|---:|---:|
| Full | 0 | 0.7226 | 0.7295 | 0.2182 |
| Exhaustive scan, max crop | All grid crops | 0.8666 | 0.8878 | 0.6182 |
| Map-ranked adaptive, max crop | 4 | 0.8600 | 0.8902 | 0.6000 |
| Map-ranked adaptive, max crop | 8 | 0.8440 | 0.8791 | 0.6000 |

Four map-ranked crops recover nearly all of the exhaustive Q1 AUROC at much
lower cost. Eight crops add false-positive opportunities without improving
TPR@5%.

The four-crop full mouse run has median end-to-end latency about `509 ms` per
image. Exhaustive Q1 scanning has median latency about `1.279 s`.

## 12. Generated-full results

This set contains generated positives only and has no matched real negative
population or pixel masks. AUROC, AP, FPR, and localization metrics are
therefore undefined. The valid outputs are score distributions and the
fraction above the already-fixed threshold.

| Category | Images | Full detected | Local detected | Fused detected |
|---|---:|---:|---:|---:|
| Mouse | 275 | 22 (8.00%) | 66 (24.00%) | 38 (13.82%) |
| Cat | 272 | 19 (6.99%) | 68 (25.00%) | 41 (15.07%) |
| Trash-can | 260 | 13 (5.00%) | 63 (24.23%) | 35 (13.46%) |
| **All** | **807** | **54 (6.69%)** | **197 (24.41%)** | **114 (14.13%)** |

| Category | Full mean / median | Local mean / median | Fused mean / median |
|---|---:|---:|---:|
| Mouse | 0.2730 / 0.2167 | 0.4236 / 0.3407 | 0.3509 / 0.2865 |
| Cat | 0.2656 / 0.2085 | 0.4342 / 0.3838 | 0.3501 / 0.2854 |
| Trash-can | 0.2419 / 0.2039 | 0.4025 / 0.3129 | 0.3259 / 0.2592 |
| **All** | **0.2605 / 0.2096** | **0.4204 / 0.3470** | **0.3426 / 0.2796** |

Fully generated images are globally harmonized and often lack the local
splice-boundary inconsistency A3D targets. The low full score also suppresses
equal-logit fusion. A complementary global synthetic-image detector is needed
for a universal defense.

## 13. Implementation details

### 13.1 Source map

| File | Responsibility |
|---|---|
| `eval/our_defense/run_trufor_a3d.py` | Deployable paired full/crop inference, fused score, maps, resume, summary |
| `eval/our_defense/run_a3d_generated_full.py` | Manifest-only generated-full inference |
| `eval/our_defense/analyze_trufor_a3d.py` | Pairing, slices, bootstrap, hard cases, transferred threshold |
| `eval/our_defense/build_final_canonical.py` | Cat/trash provenance reconstruction, JPEG-Q95 canonicalization, masks |
| `eval/our_defense/run_trufor_adaptive_scan.py` | Exhaustive and proposal-budget diagnostic |
| `eval/our_defense/run_trufor_adaptive_zoom.py` | Ground-truth-centered oracle scale diagnostic |
| `eval/our_defense/aggregate_a3d_results.py` | Unified metrics, TPR@1%, coverage, checksums, generated inventory |

### 13.2 Canonicalization

Cat and trash real/forged variants are both decoded and re-encoded with:

```text
JPEG quality 95
4:4:4 subsampling
optimize false
metadata stripped
```

The builder writes deterministic `inputs.jsonl`, `pairs.jsonl`, and
`manifest.json`, plus contract and file hashes. Masks are reconstructed in full
image coordinates from the reviewed upstream provenance. It rejects duplicate
task IDs, invalid boxes, missing sources, empty masks, geometry mismatches, and
positive pixels outside the reviewed context.

The generated-full runner uses the same JPEG-Q95/4:4:4 transform. Mouse clean
uses its existing canonical bundle; JPEG-90 is an explicit in-memory stress
transform.

### 13.3 JSONL result behavior

The paired result ID is:

```text
{task_id}|{real_or_forged}
```

Important row fields include:

```text
id, run_id, status, task_id, pair_rank, domain, kind, label, split,
gt_fraction, image_path, input_transform, crop_side, crop_stride,
grid_crops, proposal_budget, full_score, a3d_score, a3d_fused_score,
proposals, selected_crops, localization, latency, checkpoint_sha256
```

Older mouse rows predate the explicit `a3d_fused_score` field. Current analysis
reconstructs it exactly from stored `full_score` and `a3d_score`; no model
rerun or new fitting is needed.

Runners are append-only and resumable:

1. read all physical rows;
2. retain the latest row for each string ID;
3. infer only missing expected IDs;
4. verify every expected ID exists;
5. construct summaries in expected manifest order;
6. atomically replace the summary JSON.

Generated-full manifest retries are reduced to the latest successful output
path. The runner verifies exact equality between final manifest images and
files directly present in each final manifest directory, rejects cross-manifest
duplicates, and sorts by category, task ID, and path.

### 13.4 Tests and invariants

The focused tests cover:

- grid coverage of right and bottom image boundaries;
- proposal ranking independent of ground truth;
- deterministic crop-map selection;
- JPEG geometry preservation;
- quantile threshold strictly above the selected real score;
- logit fusion symmetry and idempotence;
- generated-full canonicalization geometry;
- fixed-threshold generated-full summaries;
- reviewed-context mask placement and exclusion.

## 14. Durable artifacts

### 14.1 Mouse clean, JPEG-90, and diagnostics

```text
results/our_defense/mouse_a3d_20260725/
  clean/
  jpeg90/
  diagnostics/deployable_q1/
  diagnostics/adaptive_scan_q1/
  diagnostics/oracle_zoom_q1/
```

The clean and JPEG-90 directories contain per-image JSONL, current fused-aware
summary JSON, 2,000-replicate analysis JSON, and Markdown. The diagnostic
directories contain original per-image JSONL and summaries.

### 14.2 Cat and trash-can

```text
results/our_defense/cat_trash_a3d_20260725/
```

This contains 251-pair cat and 250-pair trash JSONLs, summaries, individual
and combined bootstrap analyses, and immutable canonical manifest snapshots.

### 14.3 Generated-full

```text
results/our_defense/generated_full_images_a3d_20260726/
```

This contains all 807 final-manifest predictions and their aggregate summary.

### 14.4 Unified aggregate

```text
results/our_defense/a3d_aggregate_20260727/
  aggregate_metrics.json
  artifact_checksums.sha256
```

`aggregate_metrics.json` is the machine-readable source for every table in
this report. It includes TPR@1% details, fixed-threshold points, localization,
proposal metrics, generated-full inventory, diagnostic summaries, source paths,
and SHA-256 checksums.

## 15. Reproduction

Rebuild the machine-readable aggregate without model inference:

```bash
/root/.cache/claimforge/venvs/trufor-ae54475/bin/python \
  -m eval.our_defense.aggregate_a3d_results
```

Run a paired A3D evaluation:

```bash
CUDA_VISIBLE_DEVICES=0 \
  /root/.cache/claimforge/venvs/trufor-ae54475/bin/python \
  -m eval.our_defense.run_trufor_a3d \
  --pairs outputs/opensource/mouse_canonical_v1/pairs.jsonl \
  --run-id claim_a3d_deployable_all_20260725 \
  --scope all \
  --output-dir results/our_defense/mouse_a3d_20260725/clean \
  --device cuda:0
```

Add `--jpeg-quality 90` for the JPEG stress test.

Analyze a paired result:

```bash
/root/.cache/claimforge/venvs/trufor-ae54475/bin/python \
  -m eval.our_defense.analyze_trufor_a3d \
  results/our_defense/mouse_a3d_20260725/clean/claim_a3d_deployable_all_20260725.jsonl \
  --output-json results/our_defense/mouse_a3d_20260725/clean/claim_a3d_deployable_all_20260725.analysis.json \
  --output-markdown results/our_defense/mouse_a3d_20260725/clean/claim_a3d_deployable_all_20260725.analysis.md \
  --bootstrap-replicates 2000 \
  --calibration-results results/our_defense/mouse_a3d_20260725/clean/claim_a3d_deployable_all_20260725.jsonl
```

## 16. Limitations and next steps

1. **JPEG robustness is poor.** Proposal and crop classification share the same
   TruFor evidence source, so compression causes correlated failure.
2. **Fully generated images remain mostly undetected.** Add a complementary
   global synthetic-image detector and calibrate a threat-aware combination.
3. **Fusion selection is post-hoc for cat/trash.** Freeze A3D v1 and evaluate
   one new object category without changes.
4. **One threshold assumes comparable score distributions.** Future work may
   use known-real reference normalization, but must not use forged labels.
5. **Four crop passes add compute.** Proposal reuse makes it cheaper than an
   exhaustive scan, but latency is still about half a second per image here.
6. **Local maps are not universally better.** Keep the global map primary or
   learn a localization selector on separate development data.
7. **The base detector is not trained for current post-processing.** A future
   proposal/crop model should use matched JPEG, resize, blur, screenshot, and
   recapture augmentation.

## 17. Final conclusion

A3D is a stable adaptive defense for CLAIMFORGE's local splice regime. Its
benefit is strongest when the generated object is tiny, and equal-logit fusion
generalizes more safely than local-only across mouse, cat, and trash-can.

The method should be claimed as:

> A frozen, target-benchmark-data-free, label-blind test-time defense with a
> mouse-real calibrated threshold; cat/trash fused results are post-selection
> cross-object validation, and a future unchanged-category evaluation is
> required for a strict blind generalization claim.

It should not be claimed as a universal generated-image detector or as
JPEG-robust in its current form.
