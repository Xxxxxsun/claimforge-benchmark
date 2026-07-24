# HiFi-IFDL general checkpoint 750001 on the canonical mouse set (2026-07-24)

## 1. Status and headline

The seventh publicly reproducible local-manipulation baseline is complete.
The official HiFi-Net general detection-and-localization checkpoint ran on all
275 matched tasks, or 550 canonical JPEG images, with no errors.

HiFi-Net fails on this small local AIGC insertion condition in both of its
native tasks:

- both the benchmark `score > 0.5` rule and the official fine-class argmax
  classify **0/275 forged images** as forged;
- T1 AUROC is **0.49921**, with task-paired bootstrap 95% CI
  **[0.49714, 0.50074]**;
- native macro pixel AP is **0.003613**, CI
  **[0.002985, 0.004297]**;
- native macro pixel F1 at the public `distance >= 2.3` threshold is
  **0.000055**, CI **[0, 0.000156]**;
- only 4/275 forged images produce any positive mask pixels, and the matched
  real image is also positive for each of those four tasks;
- only 2/275 forged masks overlap any exact-difference GT pixel or edit-box
  pixel; and
- no mask reaches edit-box IoU greater than 0.3.

The two nonzero-overlap tasks share the same pristine source image. They
contribute every native true-positive pixel and approximately 99.2% of both
forged predicted-positive pixels and pristine false-positive pixels. The
nonzero micro IoU is therefore a repeated scene outlier, not stable mouse
localization.

The independent analyzer verified 3,025 files and replayed the complete
preprocessing, embedding-to-center distance, native interpolation, masks,
four classification heads, both image decisions, and both localization
spaces. This is a complete audited result rather than a checkpoint-loading,
score-direction, or output-selection error.

## 2. Pinned method and official release

The run uses the authors'
[HiFi-IFDL repository](https://github.com/CHELSEA234/HiFi_IFDL) at commit
`0ca70d651087bb09959dec583947031c47d30209`. The repository code is MIT
licensed. The externally hosted checkpoints do not have a separately stated
license, so this report does not extend the code license to the weight files.

The [CVPR 2023 paper](https://openaccess.thecvf.com/content/CVPR2023/html/Guo_Hierarchical_Fine-Grained_Image_Forgery_Detection_and_Localization_CVPR_2023_paper.html)
introduces HiFi-Net as a joint image forgery detection and localization model.
The repository publishes two different use cases:

1. localization-only weights for conventional manipulation datasets; and
2. general HiFi-IFDL detection-and-localization weights for forged content
   including GAN- and diffusion-generated images.

The author explicitly distinguishes these releases in
[issue 19](https://github.com/CHELSEA234/HiFi_IFDL/issues/19). CLAIMFORGE
pre-registered the second release because it provides native T1 and T2 and
because the threat model contains diffusion-generated content. No
localization-only checkpoint was selected after observing the test results.

The selected official bundle is:

| Component | Released path | Bytes | SHA-256 |
|---|---|---:|---|
| HRNet feature extractor | `HRNet/750001.pth` | 81,112,652 | `be21278afb4e657bdafdf581d8d8bc6bc09f3b4507b10502ce98f1ae7ef1c5c1` |
| NLC localizer/classifier | `NLCDetection/750001.pth` | 57,487,769 | `7615fcb054e7cbd0b25d647d72655a690424232668abbc911551648e84b5f8fc` |
| Authentic center/radius | `center/radius_center.pth` | 543 | `e41e09256e65bcff9ba43e72f08701bf4d3904ccdb749f2d32a008af92c2483b` |

The two model files plus the released center/radius have registered bundle
SHA-256
`62d0b9f5e501f85558cfbdd5f797dc4e2553ce74729168921257408d735681f9`.
Both load with `weights_only=True` and `strict=True`, without key rewriting or
schema fallback. Together they contain 6,890,320 parameters and 18,764 buffer
elements.

`750001` is retained as the author's released identifier, not described as an
epoch. Both optimizer states record step 750001, while the paper states
400,000 iterations and its supplement states 13 epochs. Those descriptions do
not establish that the release identifier is an epoch or that all three
numbers refer to the same training schedule.

The released authentic center is an 18-element float32 vector. Its recorded
radius is `1.2404824495315552`. The public API uses threshold `2.3`; the
internal training expression `1.85 × radius = 2.2948925` is recorded for
provenance but is not substituted for the public threshold.

## 3. Exact inference contract

### Input

The adapter reproduces the public `HiFi_Net.py` path:

1. decode the canonical JPEG bytes with `imageio.v2.imread`;
2. require an RGB uint8 image;
3. stretch directly to 256×256 with Pillow bicubic interpolation;
4. convert to contiguous CHW float32 and divide by 255; and
5. apply no mean/std normalization, crop, padding, re-encoding, TTA, or
   ensemble.

Aspect ratio is not preserved. This is an important property of the official
API, not an arbitrary CLAIMFORGE resize choice.

### Shared model output

One forward pass produces four HRNet feature resolutions:

```text
18×256×256
36×128×128
72×64×64
144×32×32
```

The NLC module returns:

- an 18×256×256 pixel embedding;
- an auxiliary learned sigmoid mask; and
- classification logits with 3, 5, 7, and 14 entries.

The adapter stores all four logit levels and the auxiliary output, but does
not silently use the auxiliary sigmoid mask as the official localization
score.

### Native T1

The finest 14 classes are authentic, three conventional image-editing
categories, four CNN face-editing categories, and six GAN/diffusion
generation categories. The continuous forged score is:

```text
score = 1 - softmax(fine_14_class_logits)[authentic_index_0]
```

Two decisions are preserved separately:

- CLAIMFORGE's registered score rule: strict `score > 0.5`;
- the public model rule: `argmax(fine_14_class_logits) != 0`.

Only the continuous fine-head score is used for AUROC and AP. No heatmap
aggregation is promoted to T1.

### Native T2

The public localization output is the per-pixel Euclidean distance between
the 18-dimensional embedding and the released authentic center:

```text
PairwiseDistance(p=2, eps=1e-6)
```

This is a raw nonnegative, unbounded distance, not a probability. The public
mask is `distance >= 2.3`. The adapter does not apply sigmoid, clipping,
normalization, score inversion, or test-set threshold selection.

The primary CLAIMFORGE native-space adapter:

1. bilinearly restores the continuous 256×256 distance to the canonical input
   dimensions with `align_corners=False`; and
2. applies the inclusive `>= 2.3` threshold after restoration.

The exact 256×256 output and a Pillow-nearest 256×256 GT are retained as an
auxiliary `model_256` result. Agreement between the two spaces tests whether
native restoration explains the result.

## 4. Why HiFi-Net can be strong, and why this condition is different

HiFi-Net combines several useful ideas:

- HRNet maintains high-resolution features while exchanging information
  across four scales;
- multi-level fine-grained supervision teaches a hierarchy from broad forgery
  families down to specific generation or editing methods;
- the image classifier and pixel localizer share a representation; and
- the localization objective organizes authentic pixel embeddings around a
  learned center, making distance from that center an interpretable anomaly
  score.

This is why HiFi-Net is an appealing joint T1+T2 baseline and why a model with
only about 6.9 million parameters can be effective on its intended
distribution.

The design does not guarantee transfer to CLAIMFORGE. The median native edit
occupies only 0.1126% of the image, and the official 256×256 stretch can reduce
it to very few pixels. Training on fully synthesized diffusion categories
also does not imply robustness to a tiny, well-blended diffusion insertion in
an otherwise real scene.

This limitation is anticipated by the authors rather than invented from this
benchmark. The
[paper supplement](https://openaccess.thecvf.com/content/CVPR2023/supplemental/Guo_Hierarchical_Fine-Grained_Image_CVPR_2023_supplemental.pdf)
states that a well-trained model can generalize poorly to images partially
manipulated by diffusion methods. In
[issue 42](https://github.com/CHELSEA234/HiFi_IFDL/issues/42), the repository
owner also acknowledges poor localization from the released pretrained
weights and advises users to focus on detection. CLAIMFORGE finds that the
released detection head fails here as well.

## 5. Preflight and run-order checks

The final-code one-pair preflight completed 2/2 images:

- real score `0.00582772`;
- forged score `0.00534183`;
- both T1 decisions were authentic;
- both masks were empty; and
- forged native pixel AP was `0.00093894`.

The initial independent audit caught a one-pixel discrepancy in its own
model-space GT replay: the runner used the registered Pillow nearest-neighbor
resize, while the analyzer used legacy top-left floor coordinates. The
analyzer was corrected to Pillow's half-pixel-center coordinates and a
1350×1800→256 regression test was added. No runner output or metric protocol
was changed.

The five-pair smoke then completed 10/10 images:

- T1 AUROC `0.5`, with 0/5 forged images positive;
- forged native macro pixel AP `0.00228831`;
- forged native macro F1 `0`; and
- pristine native false-positive area `0`.

The smoke resumed with ten selected and zero pending. Its ten shared rows are
identical to the full-run prefix for 22 deterministic fields, including input
hashes, preprocessing evidence, all logits and probabilities, embeddings,
distance maps, masks, and both localization spaces.

The 275-pair full run completed 550/550 images with no errors. Re-running the
same command reported 550 selected and zero pending; the JSONL remained 550
rows with 550 unique result IDs.

## 6. Native T1 result

Confidence intervals use 1,000 task-paired bootstrap resamples with seed
`20260724`.

| Metric | Estimate | Pair-bootstrap 95% CI |
|---|---:|---:|
| AUROC | **0.499213** | [0.497137, 0.500741] |
| Average precision | 0.503139 | [0.501377, 0.509568] |
| TPR at 5% FPR | 0.050909 | [0.036364, 0.054545] |
| Accuracy at `score > 0.5` | 0.500000 | [0.500000, 0.500000] |
| Image F1 at `score > 0.5` | **0** | [0, 0] |
| Official argmax accuracy | 0.500000 | [0.500000, 0.500000] |
| Official argmax image F1 | **0** | [0, 0] |
| Paired ranking accuracy | **0.356364** | [0.298182, 0.414545] |
| Mean paired score delta, forged minus real | -0.00008019 | [-0.00034431, 0.00013050] |

Both binary rules give the same confusion matrix:

| TP | FP | FN | TN |
|---:|---:|---:|---:|
| 0 | 0 | 275 | 275 |

Every fine-head argmax is authentic, and every continuous score is below
0.5. The maximum forged score is `0.458809`; the maximum real score is
`0.447246`.

The paired sign result is 98 wins, 134 losses, and 43 exact ties
(`p=0.02137`, two-sided exact test). Among non-tied pairs, adding the mouse
more often decreases the forged score than increases it. Real and forged
scores have correlation `0.99833`, while the median absolute paired change is
only `7.15e-7`. The fine head is therefore dominated by the shared scene
rather than the local edit.

## 7. Primary T2 result

| Metric | Native primary | Model 256 |
|---|---:|---:|
| Macro pixel AP | **0.00361302** | 0.00377677 |
| AP 95% CI | [0.00298482, 0.00429726] | [0.00311892, 0.00448175] |
| Median per-image pixel AP | 0.00175989 | 0.00187171 |
| Macro F1 at `>= 2.3` | **0.00005506** | 0.00005486 |
| Macro F1 95% CI | [0, 0.00015646] | [0, 0.00015605] |
| Macro IoU at `>= 2.3` | 0.00002769 | 0.00002759 |
| Micro F1 at `>= 2.3` | 0.00446081 | 0.00614707 |
| Micro IoU at `>= 2.3` | 0.00223539 | 0.00308301 |
| Real FP area, macro | 0.728855% | 0.729004% |

The native forged-image confusion totals are:

| TP | FP | FN | TN |
|---:|---:|---:|---:|
| 3,246 | 856,192 | 592,657 | 440,800,909 |

This gives micro precision `0.003777` and recall `0.005447`. HiFi predicts
859,438 positive pixels against 595,903 GT-positive pixels, but almost all
predicted pixels come from two near-full-image outliers.

The fixed-threshold image behavior is:

- 271/275 forged masks are empty;
- 271/275 real masks are empty;
- the same four task IDs are nonempty on both sides of the pair;
- only two forged images have any true-positive pixel;
- only two masks touch the registered edit box; and
- 0/275 masks have edit-box IoU greater than 0.3.

Mean edit-box IoU is `0.00001527`, its median is zero, and its maximum is only
`0.00288463`.

Model-space and native results are equally poor. Native interpolation is
therefore not the cause of failure.

## 8. Repeated-source outlier and pristine behavior

The canonical manifest contains 275 task rows but 270 unique pristine JPEG
hashes: five pristine images each appear in two task pairs. This is part of
the frozen benchmark and affects every method identically, but it matters for
interpreting HiFi's sparse outliers.

The only two tasks with any native true-positive or edit-box overlap are
`lodging_078_slot_001` and `lodging_009_slot_001`. They share the same
pristine image SHA-256. On both pristine task rows, HiFi marks approximately
99.984% of the image positive; their forged counterparts are also almost
entirely positive.

Together these two repeated-scene tasks contribute:

- 3,246/3,246 native true-positive pixels;
- 99.19% of forged predicted-positive pixels;
- 99.20% of pristine false-positive pixels; and
- both edit-box overlaps.

Consequently, native micro F1 `0.00446`, micro IoU `0.00224`, and pristine
micro false-positive area `0.19436%` should not be read as broad low-level
skill. Native macro false-positive area is `0.72886%` because each task is a
registered evaluation unit.

The frozen confidence intervals resample task pairs, matching all prior
baseline reports. They do not cluster by pristine SHA, so the two appearances
of this source can make uncertainty slightly optimistic. A future
source-cluster bootstrap should be added consistently across every baseline;
it is not substituted post hoc for HiFi alone.

## 9. Domain and edit-size diagnostics

### Domain

| Domain | Pairs | T1 AUROC | Native pixel AP | Macro F1 | Real FP area |
|---|---:|---:|---:|---:|---:|
| lodging | 147 | 0.49764 | 0.0032703 | 0.0001030 | 1.36033% |
| restaurant | 128 | 0.50003 | 0.0040066 | 0 | 0.003645% |

Lodging's nonzero fixed-threshold result is driven by the duplicated
near-full-image outlier and is accompanied by much larger pristine false
positive area. It is not evidence that HiFi generalizes better to lodging.
Both domains are at chance for T1 and extremely weak for continuous T2.

### Edit-size quintiles

| Quintile | Pairs | Median edit fraction | T1 AUROC | Native pixel AP | Macro F1 |
|---|---:|---:|---:|---:|---:|
| Q1, smallest | 55 | 0.05347% | 0.50198 | 0.0012023 | 0 |
| Q2 | 55 | 0.07780% | 0.49785 | 0.0020788 | 0 |
| Q3 | 55 | 0.11264% | 0.50215 | 0.0024073 | 0.00004365 |
| Q4 | 55 | 0.16293% | 0.49636 | 0.0032372 | 0 |
| Q5, largest | 55 | 0.37441% | 0.49455 | 0.0091395 | 0.00023167 |

Q5 pixel AP is 7.60 times Q1, but mean positive-pixel prevalence increases by
8.57 times. Across individual images, edit fraction and raw AP have Spearman
`rho=0.691`, while edit fraction and `AP / edit_fraction` have
`rho=-0.0004` (`p=0.994`). The apparent AP size trend is therefore primarily
the changing random-AP baseline. T1 does not improve with edit size, and
fixed-threshold T2 remains effectively zero.

## 10. Comparison with completed local baselines

All seven methods use the same canonical manifest and exact-difference native
GT. Continuous native pixel AP is the cleanest cross-method T2 comparison.
Fixed-threshold metrics describe each author's published operating point;
HiFi uses raw distance `>= 2.3`, while most probability-map methods use a
threshold around 0.5.

| Method | Native T1 | T1 AUROC | Paired rank | Native pixel AP | Macro F1 | Macro IoU | Micro IoU | Real FP area |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| CAT-Net v2 | no | — | — | **0.61234** | 0.47481 | 0.35743 | 0.18138 | 0.4359% |
| TruFor | yes | **0.81790** | **0.90909** | 0.57989 | **0.50010** | **0.37410** | **0.19800** | 1.7764% |
| IML-ViT | no | — | — | 0.15513 | 0.14325 | 0.09843 | 0.03624 | 0.8023% |
| MaskCLIP | yes | 0.50728 | 0.64727 | 0.04740 | 0.00564 | 0.00367 | 0.00312 | 0.0596% |
| MVSS-Net | map GMP | 0.50576 | 0.42182 | 0.03415 | 0.01962 | 0.01248 | 0.00369 | 3.2887% |
| PSCC-Net | independent head | 0.50408 | 0.56727 | 0.01528 | 0.0000726 | 0.0000365 | 0.00115 | 0.8265% |
| **HiFi-Net** | **independent head** | **0.49921** | **0.35636** | **0.003613** | **0.0000551** | **0.0000277** | **0.002235** | **0.7289%** |

HiFi has the lowest macro pixel AP, macro F1, and macro IoU of all seven
completed local methods. Its micro IoU is slightly above PSCC-Net only because
micro weighting magnifies the repeated near-full-image anomaly.

For native T1, TruFor remains the only clearly effective completed method.
HiFi's independent head is not merely below threshold: its AUROC is at chance
and its paired ranking is worse than the other reported T1 baselines.

This is an off-the-shelf cross-domain robustness comparison, not a controlled
architecture ranking with matched training data.

## 11. Validation and audit

The final result is backed by:

- safe and strict loading of both official model checkpoints and the released
  center/radius;
- a clean pinned source commit and 11 pinned official source-file hashes;
- 50/50 HiFi runner, metrics, and independent-analyzer tests;
- one-pair preflight, five-pair smoke, and 275-pair full execution;
- 550 unique successful full rows, zero errors, and zero retry histories;
- zero-pending resume checks for smoke and full;
- identical smoke/full artifacts over the shared ten-image prefix;
- independent validation of 550 canonical JPEGs, 275 forged GT masks, and
  2,200 generated model artifacts; and
- independent replay of preprocessing, PairwiseDistance, native bilinear
  restoration, inclusive masks, all four classification outputs, both T1
  decisions, and both T2 metric spaces.

The compatible whole-repository test environment collected 201 tests:
197 passed and four environment-gated tests were skipped. The system Python
with NumPy 2 is not the authoritative cross-baseline environment because
older MVSS/OpenCV and CAT checkpoint serialization require the established
NumPy-1-compatible environment.

The 2,200 generated full-run artifacts total 6,278,551,521 bytes. Median model
forward latency is 10.44 ms on an NVIDIA L20Z, with peak allocated CUDA memory
341,163,008 bytes. Timing excludes image decoding, artifact serialization,
and metric calculation.

## 12. Reproducible artifacts

Primary files:

- runner: `eval/opensource/run_hifi_ifdl.py`
- metric implementation: `eval/opensource/hifi_ifdl_metrics.py`
- independent analyzer: `eval/opensource/analyze_hifi_ifdl_run.py`
- full JSONL:
  `results/opensource/hifi_ifdl/hifi_ifdl_general750001_mouse_canonical_v1_full275_20260724.jsonl`
- immutable run manifest:
  `results/opensource/hifi_ifdl/hifi_ifdl_general750001_mouse_canonical_v1_full275_20260724.run_manifest.json`
- run summary:
  `results/opensource/hifi_ifdl/hifi_ifdl_general750001_mouse_canonical_v1_full275_20260724.summary.json`
- audited paired analysis:
  `results/opensource/hifi_ifdl/hifi_ifdl_general750001_mouse_canonical_v1_full275_20260724.analysis.json`

Key SHA-256 digests:

| Artifact | SHA-256 |
|---|---|
| full JSONL | `3f7a084dfdeadd03c7dfcc0466a07244b0fdf136e70bf8e462a92ffb0e8c5756` |
| run manifest | `e5bbfc1bb98597fd6dcf1fa02efc20aec58f14026b0923e62aef7615fda341a4` |
| summary | `f27a170a832b102e9e69ad86e1cc239292d063c8807475c7227f46b64da50526` |
| analysis | `73d45ddbc28fa9eb4b61babb69879ddbc4ca3f2b867a7a7106255b26d3504c08` |

Commands:

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=/root/claimforge-benchmark \
/root/.cache/claimforge/venvs/hifi-ifdl-0ca70d6/bin/python \
  -m eval.opensource.run_hifi_ifdl \
  --repo-root /root/claimforge-benchmark \
  --run-id hifi_ifdl_general750001_mouse_canonical_v1_full275_20260724 \
  --device cuda:0 --bootstrap-samples 1000 --fail-fast

PYTHONPATH=/root/claimforge-benchmark \
/root/.cache/claimforge/venvs/hifi-ifdl-0ca70d6/bin/python \
  -m eval.opensource.analyze_hifi_ifdl_run \
  --repo-root /root/claimforge-benchmark \
  --run-id hifi_ifdl_general750001_mouse_canonical_v1_full275_20260724 \
  --bootstrap-iterations 1000 \
  --bootstrap-seed 20260724
```

## 13. Conclusion and next method

HiFi-Net's hierarchical design is compelling, but its official general
checkpoint does not transfer to CLAIMFORGE's tiny local diffusion insertions.
The classifier remains tied to the shared background, and the public
hypersphere threshold is almost always empty except for a few matched
real/forged scene anomalies.

The next frozen local-manipulation baseline is **Mesorch**, using the
pre-registered official `mesorch-98.pth` checkpoint. Mesorch provides a native
pixel map but no independent image-classification head, so its primary result
will be T2 and T1 will remain N/A rather than being synthesized from the map.
