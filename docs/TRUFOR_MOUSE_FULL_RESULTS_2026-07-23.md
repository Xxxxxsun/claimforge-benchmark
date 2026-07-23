# TruFor on the canonical mouse set (2026-07-23)

## 1. Status and headline

The second publicly released research baseline is complete: all 275 paired
tasks, or 550 images, finished successfully with no errors.

TruFor is the first strong source-available research result on this benchmark.
On exactly the same canonical inputs and ground truth used for MaskCLIP, it
reaches:

- image-level AUROC 0.8179 and average precision 0.8393;
- 90.91% paired ranking accuracy;
- native-resolution macro pixel AP 0.5799 and macro IoU 0.3741 at the released
  0.5 map threshold; and
- 83/275 edit-box hits, compared with 1/275 for MaskCLIP.

This is a large improvement over MaskCLIP, but it is not a solved task. At the
fixed image threshold of 0.5, TruFor still misses 140/275 forged images. Its
smallest-edit quintile is materially weaker than its largest-edit quintile, and
the localization map contains some positive pixels on 273/275 pristine images.

## 2. Pinned method and inference contract

The run pins the official
[GRIP-UNINA TruFor repository](https://github.com/grip-unina/TruFor) at commit
`ae54475df6f41a491d7615100feb19263dec13f7`.

The exact released assets are:

- experiment configuration `trufor_ph3`, SHA-256
  `a87108eb0df40d9bab6a303eb91419564b7c106d5105bbd5d8ecaec1567b5b8b`;
- official weights archive, published MD5
  `7bee48f3476c75616c3c5721ab256ff8` and local SHA-256
  `953f1f7eda0dd2c5ece322ae9c185ba1079c1265aa5fdf319ef5a20604d206d8`;
- inner checkpoint `trufor.pth.tar`, epoch 81, 952 state-dict keys, SHA-256
  `ac1d90e329a72e0d66e8665e123a19e94bfae3209c3ef8a4f9ca3b91578c7844`;
  and
- official license file, SHA-256
  `07201e07e3d2c1ac55480037a87734fcccacbb0cd0e25a31e3b89ac7ffadf8b4`.

The checkpoint strictly loaded every key. Loading used PyTorch's
`weights_only=True` path with the minimum required NumPy allowlist; unrestricted
pickle loading was not used.

The adapter follows the released implementation exactly:

- decode with PIL and convert to RGB;
- HWC to CHW float32, divided by **256.0**, not 255;
- no input resize or crop;
- batch size 1 and float32 inference;
- `sigmoid(det_logit)` as the T1 probability of forged;
- channel 1 of `softmax(pred_logits)` as the T2 forged-pixel probability; and
- `sigmoid(conf)` retained as TruFor's localization-reliability/TCP map, not
  multiplied into the forged map and not treated as a second anomaly score.

TruFor already restores the map and reliability output to the input H×W inside
the network. The adapter therefore saves those native arrays without a second
interpolation.

The official repository's license permits free reproduction, modification, and
use only for informational and nonprofit purposes, requires preservation of
notices and attribution, and expressly prohibits unauthorized industrial or
profit-oriented use. The model, code, or a derived service must not be presented
as a commercially deployable component without separate authorization. A
research context alone does not establish license compliance: each use must
independently satisfy both the informational and nonprofit restrictions or
obtain authorization from the rightsholder.

## 3. CLAIMFORGE evaluation contract

The authoritative set is the 275 `status=good`, `candidates=mouse` records in
`claimforge_generation_review_labels.json`:

| Domain | Pairs |
|---|---:|
| lodging | 147 |
| restaurant | 128 |
| total | 275 |

Real and forged images are both independently decoded to RGB and materialized
as metadata-free JPEG at quality 95 with 4:4:4 subsampling. Inference reads
those materialized files, and each input SHA-256 is verified before use.

Forged GT is the exact nonzero RGB difference between decoded source and forged
PNG before canonical JPEG encoding. The generation anchor box is not substituted
for the true object mask. The edit fractions are very small:

| Edit fraction of full image | Value |
|---|---:|
| minimum | 0.02486% |
| median | 0.11264% |
| mean | 0.16846% |
| maximum | 1.29272% |

Primary T2 metrics use the direct forged-probability channel, the full exact-diff
GT, and a fixed threshold of 0.5. Pixel AP uses the continuous native map.
Confidence intervals use 1,000 percentile-bootstrap replicates with paired task
as the resampling unit, preserving each real/forged pair.

## 4. Image-level detection (T1)

| Metric | Estimate | Pair-bootstrap 95% CI |
|---|---:|---:|
| AUROC | 0.8179 | [0.7902, 0.8447] |
| Average precision | 0.8393 | [0.8125, 0.8658] |
| TPR at FPR ≤ 5% | 0.4436 | [0.3709, 0.5309] |
| Accuracy at 0.5 | 0.7109 | [0.6818, 0.7400] |
| Image F1 at 0.5 | 0.6294 | [0.5764, 0.6771] |
| Paired ranking accuracy | 0.9091 | [0.8727, 0.9418] |
| Mean paired score change | +0.34625 | [+0.31165, +0.38230] |

At threshold 0.5, the confusion matrix is
`TP=135, FP=19, FN=140, TN=256`: TPR is 49.09% and FPR is 6.91%. The operating
point that keeps empirical FPR at or below 5% uses threshold 0.58574 and reaches
44.36% TPR.

The paired signal is much stronger than the fixed-threshold recall: the forged
score exceeds its matched real score in 250/275 pairs. The exact two-sided sign
test gives `p=7.37e-48`. This means the model responds consistently to the local
edit even when the absolute detection head remains below 0.5.

## 5. Pixel localization (T2)

| Native-resolution metric | Estimate | Pair-bootstrap 95% CI |
|---|---:|---:|
| Macro pixel AP | 0.57989 | [0.54696, 0.61162] |
| Median per-image pixel AP | 0.66459 | — |
| Macro pixel F1 at 0.5 | 0.50010 | [0.46763, 0.53183] |
| Macro pixel IoU at 0.5 | 0.37410 | [0.34668, 0.40061] |
| Micro pixel F1 at 0.5 | 0.33056 | [0.27481, 0.39665] |
| Micro pixel IoU at 0.5 | 0.19800 | [0.15929, 0.24739] |
| Box-hit rate at 0.5 | 83/275 (30.18%) | — |

A box hit is IoU greater than 0.3 between the predicted native binary mask and
the recorded edit anchor. Only 1/275 forged images has an empty 0.5-threshold
mask.

The fixed map threshold is not micro-optimal for this set:

- a single oracle threshold selected over all forged evaluation pixels is
  approximately 0.92943 and reaches micro F1 0.46908 and micro IoU 0.30640;
- allowing a different oracle threshold for each forged image gives mean/median
  F1 0.59884/0.67665.

These are deliberately optimistic post-hoc diagnostics, not deployable scores.
The fixed 0.5 results remain the primary numbers.

Pristine-image localization is a relevant caveat. At threshold 0.5, 273/275
real images contain at least one positive pixel. The mean, median, and maximum
positive fractions are 1.776%, 0.715%, and 44.853%, respectively. Therefore the
localization head should not be interpreted independently of the image-level
decision, especially on pristine inputs.

## 6. Diagnostic slices

### Domain

| Domain | Pairs | AUROC | Paired rank acc. | Macro pixel AP | Median pixel AP | Micro IoU@0.5 |
|---|---:|---:|---:|---:|---:|---:|
| lodging | 147 | 0.8416 | 0.9252 | 0.61732 | 0.70789 | 0.25285 |
| restaurant | 128 | 0.7913 | 0.8906 | 0.53691 | 0.60448 | 0.16327 |

Lodging is consistently easier, but restaurant remains well above chance at
T1 and retains meaningful localization signal.

### Edit-size quintiles

| Quintile | Pairs | Median edit fraction | AUROC | Paired rank acc. | Macro pixel AP | Micro IoU@0.5 |
|---|---:|---:|---:|---:|---:|---:|
| Q1, smallest | 55 | 0.05347% | 0.7223 | 0.9455 | 0.42400 | 0.07698 |
| Q2 | 55 | 0.07780% | 0.8159 | 0.8909 | 0.56528 | 0.11970 |
| Q3 | 55 | 0.11264% | 0.8575 | 0.9091 | 0.58145 | 0.28548 |
| Q4 | 55 | 0.16293% | 0.8479 | 0.8545 | 0.58939 | 0.21531 |
| Q5, largest | 55 | 0.37441% | 0.9412 | 0.9455 | 0.73933 | 0.36336 |

Performance increases substantially with edit size. The relationship is not
perfectly monotonic for every fixed-threshold metric, but the smallest versus
largest contrast is clear: AUROC rises from 0.7223 to 0.9412 and macro pixel AP
from 0.4240 to 0.7393.

## 7. Direct comparison with MaskCLIP

Both methods were run on the same 275 paired canonical JPEGs, with the same
exact-diff masks and paired bootstrap design.

| Metric | MaskCLIP | TruFor |
|---|---:|---:|
| T1 AUROC | 0.5073 | **0.8179** |
| T1 average precision | 0.5118 | **0.8393** |
| TPR at FPR ≤ 5% | 0.0509 | **0.4436** |
| Paired ranking accuracy | 0.6473 | **0.9091** |
| Macro pixel AP | 0.04740 | **0.57989** |
| Macro pixel IoU@0.5 | 0.00367 | **0.37410** |
| Micro pixel IoU@0.5 | 0.00312 | **0.19800** |
| Edit-box hits@0.5 | 1/275 | **83/275** |

MaskCLIP is effectively at random for absolute T1 separation and almost always
empty at its released T2 threshold. TruFor supplies useful absolute and paired
T1 signal and materially overlaps the true mouse edits. This makes TruFor the
current publicly released research reference result for CLAIMFORGE.

The comparison should not be interpreted as a general ranking of the two papers.
It is specific to these sub-percent mouse insertions, this canonicalization, the
released checkpoints, and zero-shot inference.

## 8. Protocol caveat relative to the TruFor paper

The official TruFor metric code ignores a band around forged-region boundaries
and reports the better F1 of the map and its inverse. CLAIMFORGE does neither:
it fixes the released forged-channel direction, scores every exact-diff pixel,
and never chooses an inverse after looking at GT.

This stricter contract is intentional because all publicly released methods need the
same semantic direction and the same GT. Consequently, the localization F1 and
IoU in this report are cross-method CLAIMFORGE metrics and must not be compared
directly with the official TruFor paper's F1.

## 9. Validation and artifact audit

The result is backed by the following checks:

- On the official `tampered2.png` sample, the adapter and official test path
  produced exactly identical input tensors, image score, forged map, and
  reliability map.
- The released checkpoint loaded strictly with 952/952 state keys at epoch 81.
- A fixed five-pair smoke run completed before the full run.
- The 10 overlapping smoke/full images are bit-exact in score, logit margin,
  score-map hash, reliability-map hash, binary-mask hash, class probabilities,
  decisions, reliability statistics, and localization metrics.
- The post-hoc analyzer re-hashed 2,475 files and verified every canonical
  image, native score map, native reliability map, binary mask, and forged GT.
- It checked float32 dtype, native dimensions, finite values, `[0,1]` range,
  and exact equality of every saved mask with `score_map >= 0.5`.
- It independently recomputed and matched the run-manifest fingerprint,
  canonical input hash and order, pinned source/config/license/checkpoint
  metadata, all 550 result identities, and runner-summary coverage.
- The append-only JSONL contains 550 physical rows and 550 unique latest rows,
  all successful, with no duplicate or recovered history.

## 10. Reproducible artifacts

| Artifact | Path | SHA-256 |
|---|---|---|
| canonical manifest | `outputs/opensource/mouse_canonical_v1/manifest.json` | `beb3c30e436db682bbadef794404838f33a4812f18f22819dd6ab1ef3de6f0b1` |
| full results | `results/opensource/trufor/trufor_mouse_canonical_v1_full275_20260723.jsonl` | `1c393d212c2f6cb236f82fdc397b8067e8887b8b73f0368b9e98f7edb8dd4afe` |
| run manifest | same basename, `.run_manifest.json` | `beb695245bc6cf4384c6340d3e7bfbd32500e250e90159ca8e027f4b3d779efc` |
| runner summary | same basename, `.summary.json` | `324afb697119281fd49a166a56551028bc81ea938f3e4c5b024720bd6bff876c` |
| post-hoc analysis | same basename, `.analysis.json` | `476c40b4b449c5ce187834304b049001f28196e9db1f8c708ce07d12e5dbea5d` |

The native score and reliability arrays plus threshold masks occupy 6.6 GB
under
`outputs/opensource/trufor/trufor_mouse_canonical_v1_full275_20260723/`.
The deterministic run-manifest fingerprint is
`5933764a88028863f7fceb688a5694e24eb3e016da1cd1b68b8ad3c9f48931ea`.

Environment: Python 3.12.3, PyTorch `2.8.0.dev20250627+cu128`, CUDA 12.8,
timm 0.5.4, NumPy 1.26.4, Pillow 11.1.0, scikit-learn 1.5.2, and one
NVIDIA L20Z. Median measured model-forward latency is 226.53 ms/image at batch
size 1; the mean is 173.30 ms/image because source resolutions vary
substantially. Peak allocated CUDA memory is 7.41 GB. Decode, metrics, hashing,
and artifact writes are excluded from latency. Treat latency as operational
metadata rather than a clean performance benchmark: a separate Hunyuan vLLM
service began restarting during the final seconds of this run. This does not
change the bit-exact model outputs or evaluation metrics.

Commands:

```bash
CUDA_VISIBLE_DEVICES=0 \
  /root/.cache/claimforge/venvs/trufor-ae54475/bin/python \
  -m eval.opensource.run_trufor \
  --run-id trufor_mouse_canonical_v1_full275_20260723 \
  --condition mouse_canonical_v1_full275 \
  --device cuda:0 --fail-fast

/root/.cache/claimforge/venvs/trufor-ae54475/bin/python \
  -m eval.opensource.analyze_trufor_run \
  --run-id trufor_mouse_canonical_v1_full275_20260723 \
  --bootstrap-iterations 1000 --bootstrap-seed 20260723
```

## 11. Decision and next method

TruFor should remain in the paper's core source-available research-method table
as the current strong T1+T2 reference. The result is positive enough to be
scientifically useful and still leaves clear headroom on image-level recall,
the smallest edits, restaurant scenes, and pristine-image map false positives.

The next run should start the IMDL-BenCo model-zoo block with **CAT-Net v2**.
Its JPEG/DCT stream is methodologically distinct from TruFor and particularly
relevant to CLAIMFORGE's splice-back-into-JPEG construction. CAT-Net should be
reported as T2-only unless the exact released checkpoint/harness exposes an
official image-level head; an improvised aggregation of its map must not be
silently labeled as T1.
