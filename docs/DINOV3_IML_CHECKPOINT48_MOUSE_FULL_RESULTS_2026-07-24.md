# DINOv3-IML checkpoint 48 on the canonical mouse set (2026-07-24)

## 1. Status and headline

The tenth publicly reproducible local-manipulation baseline is complete.
DINOv3-IML's official CAT ViT-L/16 LoRA-r32 checkpoint at epoch 48 ran on
all 275 matched tasks, or 550 canonical JPEG images, with no errors.

DINOv3-IML is third of the ten completed local methods by native macro pixel
AP, behind CAT-Net v2 and TruFor and ahead of RelayFormer. It shows a strong
success mode on larger edits but transfers poorly to the typical tiny
CLAIMFORGE insertion:

- native macro pixel AP is **0.296765**, with task-paired bootstrap 95% CI
  **[0.256238, 0.341924]**;
- median per-image pixel AP is only **0.040193**;
- native macro pixel F1 at the frozen strict `probability > 0.5` rule is
  **0.177574**, CI **[0.150055, 0.206864]**;
- native micro pixel F1 is **0.130077**, CI
  **[0.088126, 0.191006]**;
- 151/275 forged images have exact-GT F1 equal to zero, while 124/275 hit at
  least one exact manipulated pixel;
- 85/275 forged images reach pixel AP at least 0.5, and 45/275 reach F1 at
  least 0.5;
- 125/275 masks overlap the registered edit box at all, but only 23/275
  satisfy the planned box-hit rule of edit-box IoU greater than 0.3; and
- mean pristine false-positive area is **0.6514%**, CI
  **[0.3504%, 1.0783%]**, although one severe outlier reaches 45.04%.

Edit area is the clearest diagnostic. Native macro AP rises monotonically
from **0.070490** in the smallest edit-area quintile to **0.706413** in the
largest. Macro F1 rises from **0.019332** to **0.485866** over the same
range.

The independent analyzer validated 3,575 files, reconstructed the official
preprocessing, independently replayed the complete
`32 logits -> 512 logits -> sigmoid probability -> native probability`
chain, regenerated all 550 strict-threshold masks, and recomputed every
metric. The largest replay errors are bounded at `4.7683716e-6` for logits,
`5.9604645e-7` for model probabilities, and `2.0265579e-6` for native
probabilities. There are zero threshold-mask pixel disagreements.

This checkpoint has no image-classification head. DINOv3-IML returns only a
dense manipulation map, so T1 is **N/A**. Map maximum, mean, or mask
nonemptiness is not promoted to an unofficial image detector.

## 2. Pinned paper, code, architecture, and checkpoint

The method is described in the authors'
[DINOv3-IML technical report](https://arxiv.org/html/2604.16083),
“DINOv3 Beats Specialized Detectors: A Simple Foundation Model Baseline for
Image Forensics,” arXiv v1 dated 2026-04-17. It is labeled here as a 2026
non-peer-reviewed preprint.

The run uses the
[official DINOv3-IML repository](https://github.com/Irennnne/DINOv3-IML/tree/ba45b0a203c698b36fe2b0e658bb49ebbb1163cc)
at commit `ba45b0a203c698b36fe2b0e658bb49ebbb1163cc`. The repository does
not identify the exact source revision used to train the released file, so
this is an operational reproduction pin, not a claimed training commit.

The checkpoint needs Meta's DINOv3 architecture implementation. The adapter
pins the
[Meta DINOv3 repository](https://github.com/facebookresearch/dinov3/tree/31703e4cbf1ccb7c4a72daa1350405f86754b6d1)
at `31703e4cbf1ccb7c4a72daa1350405f86754b6d1`, the latest observed
commit preceding the checkpoint's recorded modification time. This is also
an architecture-reproduction pin, not a claim about the authors' training
revision.

The selected artifact was registered before looking at full CLAIMFORGE
performance:

| Field | Value |
|---|---|
| Released filename | `checkpoint-48.pth` |
| Provider | Author-linked [Google Drive folder](https://drive.google.com/drive/folders/125leLub_M-lICa1ILTOL-FCz4ZY6eutj) |
| Google Drive file ID | `1xqZDqhSQUl_1vs3SD4EfjHHmeu2pwLh9` |
| Recorded modification time | 2026-04-07 14:28:38 UTC |
| Bytes | 1,321,705,819 |
| SHA-256 | `01f23401e048f706ea0e63fb0429ddef80db3197ac0f5707bd584a8b056177fa` |
| Recorded epoch | 48 |
| State tensors | 432 |
| State elements | 312,275,987 |
| Tensor payload bytes | 1,249,103,956 |
| Parameters / buffers | 312,200,705 / 75,282 |
| Trainable parameters | 9,046,529 |

The top-level checkpoint keys are exactly `model`, `optimizer`, `epoch`,
`scaler`, and `args`. The state has 416 `backbone.*` keys, including 48 LoRA
keys, and 16 `seg_head.*` keys. It contains the complete DINOv3 backbone,
LoRA adapters, batch-normalization buffers, and segmentation head; a
separate gated Meta backbone checkpoint is not required.

The adapter safely loads the full container with `weights_only=True` and an
explicit `argparse.Namespace` allowlist. It validates the optimizer, AMP
scaler, saved arguments, state shapes, parameter partition, dtype
distribution, byte size, and SHA-256. It then constructs the pinned Meta
ViT-L/16 architecture with `pretrained=False`, blocks weight downloads, and
strict-loads the complete state without missing or unexpected keys.

The authors selected one checkpoint per configuration by the highest mean
pixel F1 across four external test sets. The CAT LoRA-r32 configuration
reports the strongest average among the released ViT-L variants in the
paper: 0.847 versus 0.837 for LoRA-r64 and 0.826 for full fine-tuning.
CLAIMFORGE did not participate in this selection, but the external-test-set
epoch selection is disclosed rather than treated as a validation-only
choice.

### License boundary

The DINOv3-IML repository code is MIT licensed. That does not by itself make
the complete checkpoint an unrestricted MIT artifact: the release does not
state a separate checkpoint license, and the file embeds the full Meta
DINOv3 backbone. Redistribution or downstream use must therefore separately
review and comply with the
[Meta DINOv3 License Agreement](https://ai.meta.com/resources/models-and-libraries/dinov3-license/),
including its notice, attribution, and redistribution requirements. This
report describes reproducibility, not legal advice.

## 3. What the method does and why it is strong

DINOv3-IML tests a deliberately simple claim: a large self-supervised
foundation vision backbone can be a better forensic feature extractor than a
highly specialized manipulation architecture.

1. The 512x512 RGB input is divided into 16x16 patches.
2. A 24-block DINOv3 ViT-L backbone produces a 32x32 grid of
   1,024-dimensional dense features.
3. The pretrained backbone remains frozen except for LoRA matrices injected
   into every attention block's QKV projection.
4. LoRA rank 32 with alpha 64 adapts attention using about 9.05 million
   trainable parameters in total when combined with the dense head, while
   retaining the roughly 312-million-parameter foundation representation.
5. A lightweight convolutional head maps
   `1024 -> 512 -> 256 -> 1`, using batch normalization and ReLU between
   convolutions.
6. The 32x32 logit map is bilinearly enlarged to 512x512, and a sigmoid
   yields the manipulation probability.

The approach is plausible because DINOv3's self-supervised dense
representations already encode strong local structure and long-range visual
context. LoRA can redirect attention toward forensic inconsistencies without
fully overwriting the foundation representation, while the small dense head
keeps the learning problem simple. The CAT checkpoint was trained with
CASIAv2, FantasticReality, IMD2020, and TampCOCO and uses an edge-weighted
loss in addition to pixel BCE.

This combination is very strong on the paper's traditional CAT protocol.
It does not guarantee transfer to CLAIMFORGE. The native mouse edit occupies
a median **0.1126%** of pixels. At the 32x32 output-grid scale, that is only
about **1.15 grid cells of equivalent area**. The smallest quintile spans
roughly 0.25 to 0.69 cells of equivalent area. Bilinear upsampling cannot
recreate fine evidence that was not retained in the coarse feature grid.
This resolution bottleneck is consistent with the observed size curve, but
it is not established as the only cause: training-domain, texture, scene,
position, and compositing differences remain potential confounders.

## 4. Exact inference and output contract

### Official model input

For every canonical JPEG, the frozen adapter:

1. decodes the exact bytes with Pillow and converts to RGB;
2. directly stretches the image to 512x512 with Pillow bilinear resize,
   without preserving aspect ratio;
3. converts to float32 and divides by 255;
4. applies ImageNet mean `(0.485, 0.456, 0.406)` and standard deviation
   `(0.229, 0.224, 0.225)`; and
5. performs one deterministic float32 `model.predict` call.

There is no crop, padding, re-encoding, test-time augmentation, ensemble, or
second forward pass. The first canonical input changes from 1800x1350 to
512x512 and has frozen preprocessing-tensor SHA-256
`0eaabc875a4abe662f520ae854609027b4f1c9fc54aa2df8a0bb3da4b56cd20a`.

### Continuous output and standalone-script quantization

The official model method returns a continuous float32 sigmoid probability,
but the repository's standalone `predict()` wrapper converts it to an
8-bit Pillow image before `_save_mask()` reconstructs a float map and applies
the threshold. That display round-trip makes the CLI binary boundary
effectively `floor(255p) > 127`, or approximately `p >= 0.501961`.

CLAIMFORGE captures the continuous tensor directly from the official
`model.predict()` method before presentation quantization and applies the
pre-registered strict `p > 0.5` rule. This is a disclosed continuous-output
adapter, not a claim of bit-exact equality with the standalone CLI's saved
mask.

A single forward hook on `seg_head` also retains the 32x32 pre-resize logits.
For every image, the run stores:

- captured 32x32 float32 logits;
- official bilinear-resized 512x512 float32 logits;
- official 512x512 continuous sigmoid probability;
- the 512 probability bilinearly restored to native resolution with
  `align_corners=False`; and
- a lossless native PNG equal to strict native probability `> 0.5`.

The native adapter restores **probability**, not logits followed by another
sigmoid. Primary T2 metrics use the exact native difference mask. Auxiliary
512-space metrics resize GT with nearest-neighbor interpolation. Prediction
inversion, permutation F1, threshold tuning, and image-score derivation are
not used.

## 5. Preflight, smoke, and resume checks

The final-code one-forged-image preflight completed without error:

- native pixel AP `0.00274630`;
- native F1 `0`;
- predicted-positive fraction `0.0009877%`; and
- cold forward latency `328.757 ms`.

The five-pair smoke completed 10/10 images:

- native macro AP `0.21472039`;
- native macro F1 `0.12015154`;
- native micro F1 `0.02888863`;
- mean real false-positive area `0.413311%`;
- 2/5 forged masks overlapped the edit box at all; and
- 0/5 reached edit-box IoU greater than 0.3.

The smoke resumed with ten selected and zero pending. Its JSONL remained ten
rows and retained SHA-256
`c961f578dc25620f45a5bc3b2ee3649d45bf380057e43494f35cd9ffef6cfbe4`.
The ten shared rows are also identical to the full-run prefix for all 27
registered artifact, preprocessing, threshold, geometry, and metric fields.

## 6. Primary native T2 result

Confidence intervals below come from the final independent analyzer using
1,000 task-paired bootstrap resamples with seed `20260724`.

| Metric | Estimate | Pair-bootstrap 95% CI |
|---|---:|---:|
| Macro pixel AP | **0.296765** | [0.256238, 0.341924] |
| Median per-image pixel AP | **0.040193** | — |
| Macro precision at `> 0.5` | 0.373324 | — |
| Macro recall at `> 0.5` | 0.145952 | — |
| Macro F1 at `> 0.5` | **0.177574** | [0.150055, 0.206864] |
| Macro IoU at `> 0.5` | **0.123332** | [0.103144, 0.145529] |
| Micro precision at `> 0.5` | 0.090717 | — |
| Micro recall at `> 0.5` | 0.229769 | — |
| Micro F1 at `> 0.5` | **0.130077** | [0.088126, 0.191006] |
| Micro IoU at `> 0.5` | **0.069563** | [0.046094, 0.105587] |
| Real FP area, macro | **0.651390%** | [0.350430%, 1.078325%] |
| Real FP area, micro | **0.462997%** | [0.315410%, 0.641550%] |

The forged native confusion counts at the fixed threshold are:

| TP | FP | FN | TN |
|---:|---:|---:|---:|
| 136,920 | 1,372,396 | 458,983 | 440,284,705 |

The predicted-positive fraction on forged images is `0.341279%`.

The distribution is strongly skewed:

- AP ranges from `0.000238` to `0.993448`;
- 123/275 images have AP at least 0.1;
- 85/275 have AP at least 0.5, and 31/275 have AP at least 0.9;
- 41/275 forged masks are empty, and only 124/275 hit an exact-GT pixel;
- 105/275 reach F1 at least 0.1, 74/275 reach 0.3, and 45/275 reach 0.5;
- 53/275 reach exact-GT IoU greater than 0.3; and
- median F1 and median IoU are both zero.

The mean AP therefore describes a substantial success subset, not the
typical task.

## 7. Domain and edit-size behavior

### Domain

| Domain | Pairs | Macro AP | Median AP | Macro F1 | Micro F1 | Macro IoU | Real FP area |
|---|---:|---:|---:|---:|---:|---:|---:|
| Lodging | 147 | **0.378451** | 0.145199 | 0.217916 | 0.190081 | 0.148570 | 0.6021% |
| Restaurant | 128 | **0.202953** | 0.013574 | 0.131243 | 0.098027 | 0.094350 | 0.7080% |

Lodging macro AP is about 1.86 times the restaurant result. This is a
post-hoc test-set slice: it establishes association in this dataset, not a
causal claim that the architecture inherently understands lodging scenes
better.

### Native edit-area quintile

| Quintile | Median edit area | Edit-area range | Macro AP | Median AP | Macro F1 | Micro F1 |
|---|---:|---:|---:|---:|---:|---:|
| Q1, smallest | 0.0535% | 0.0249–0.0672% | 0.070490 | 0.003292 | 0.019332 | 0.009755 |
| Q2 | 0.0778% | 0.0673–0.0938% | 0.178127 | 0.015217 | 0.099805 | 0.074076 |
| Q3 | 0.1126% | 0.0953–0.1295% | 0.224182 | 0.030671 | 0.129684 | 0.052135 |
| Q4 | 0.1629% | 0.1310–0.2178% | 0.304611 | 0.087799 | 0.153181 | 0.047327 |
| Q5, largest | 0.3744% | 0.2196–1.2927% | **0.706413** | **0.847032** | **0.485866** | **0.487350** |

Macro AP increases monotonically by a factor of about 10 from Q1 to Q5.
Macro F1 increases by about 25 times. Q5's result demonstrates that the
checkpoint is not generally broken on this domain; the core transfer failure
is concentrated in the smallest local insertions.

The fixed-threshold micro curve is not monotonic from Q2 through Q4 because
micro aggregation weights native pixels and is sensitive to a few large
false-positive regions. Q5 paired pristine images also have a larger mean
false-positive area of 1.669%. Area groups can therefore carry scene and
texture confounders in addition to edit size.

## 8. Model-space versus native-space result

| Metric | Forced-square model 512 | Native adapter |
|---|---:|---:|
| Macro AP | 0.296430 | 0.296765 |
| Macro F1 | 0.178508 | 0.177574 |
| Macro IoU | 0.124085 | 0.123332 |
| Micro F1 | 0.171811 | 0.130077 |
| Micro IoU | 0.093979 | 0.069563 |
| Real FP area, macro | 0.652403% | 0.651390% |

Macro estimates are nearly unchanged, so the 512-to-native probability
restore does not explain the typical failures. The micro difference should
not be read as a pure interpolation penalty: 512-space micro aggregation
assigns every image the same number of model pixels, whereas native
aggregation weights each image by its original dimensions and uses exact
native GT geometry.

## 9. Fixed threshold, pristine behavior, and edit-box overlap

The strict 0.5 threshold remains the only primary threshold. A descriptive
test-label oracle was computed only to diagnose calibration:

- the best single 65,536-bin global threshold is `0.470497`;
- it changes native micro F1 only from `0.130077` to `0.131518`; and
- a separate label-oracle threshold for every image gives mean F1
  `0.307286`, but median F1 only `0.107729`.

These values use test labels and are ineligible for the main table. The
single global oracle improves micro F1 by only `0.001441`, so one threshold
change does not repair the localization errors. The per-image oracle is not
deployable; it only shows that calibration and score distributions vary
substantially across images.

At `> 0.5`, 209/275 pristine images contain at least one positive pixel. This
is not an image-level false-positive rate because the model has no T1 head.
Their native positive-area distribution is:

- mean `0.651390%`;
- median `0.048065%`;
- 95th percentile `1.952214%`; and
- maximum `45.043945%`.

Twenty-nine pristine images exceed 1% positive area, and four exceed 5%.
The large maximum explains why the macro mean is much higher than the
median.

For the 275 forged images:

- 125 masks overlap the registered half-open edit box at all;
- 23 satisfy the planned box-hit rule, edit-box IoU greater than 0.3;
- mean edit-box IoU is `0.082426`;
- median edit-box IoU is `0`; and
- maximum edit-box IoU is `0.717473`.

The main box-hit value is therefore **23/275 = 8.36%**. The more permissive
125/275 any-overlap statistic is reported as a separate diagnostic and must
not be described as image detection.

## 10. Why the paper result and CLAIMFORGE result differ

The authors report 0.847 mean pixel F1 for this CAT LoRA-r32 configuration
across standard external forensic datasets. The same official checkpoint
gets 0.178 native macro F1 under the frozen CLAIMFORGE protocol. This is
strong evidence of a threat-distribution transfer gap, but the two numbers
are not a controlled one-metric subtraction:

- the paper averages multiple datasets, while CLAIMFORGE reports per-image
  native macro metrics on 275 paired tasks;
- source images, manipulation types, edit sizes, GT semantics, resolutions,
  and output adapters differ;
- the paper's checkpoint epoch was selected using external-test-set mean F1;
- CLAIMFORGE also audits paired pristine false-positive area; and
- the paper is currently a technical report rather than a peer-reviewed
  result.

The defensible statement is: **the authors report 0.847 average F1 under
their CAT protocol; the pre-registered zero-shot CLAIMFORGE protocol obtains
0.178 native macro F1, with performance strongly dependent on edit area.**

## 11. Position among completed local methods

All rows below use the same 275 paired tasks, canonical JPEGs, native
exact-difference GT, and each checkpoint's registered public operating rule.
Native continuous AP is the cleanest common ranking metric.

| Method | Native macro AP | Macro F1 | Macro IoU | Micro F1 | Real FP area |
|---|---:|---:|---:|---:|---:|
| CAT-Net v2 | 0.612336 | 0.474814 | 0.357430 | 0.307062 | 0.4359% |
| TruFor | 0.579892 | 0.500100 | 0.374100 | 0.330555 | 1.7764% |
| **DINOv3-IML** | **0.296765** | **0.177574** | **0.123332** | **0.130077** | **0.6514%** |
| RelayFormer | 0.227755 | 0.187240 | 0.132916 | 0.095885 | 0.6147% |
| IML-ViT | 0.155126 | 0.143250 | 0.098432 | 0.069951 | 0.8023% |
| Mesorch | 0.092200 | 0.045363 | 0.031371 | 0.036127 | 0.3578% |
| MaskCLIP | 0.047401 | 0.005641 | 0.003666 | 0.006219 | 0.0596% |
| MVSS-Net | 0.034147 | 0.019619 | 0.012484 | 0.007352 | 3.2887% |
| PSCC-Net | 0.015278 | 0.000073 | 0.000036 | 0.002297 | 0.8265% |
| HiFi-IFDL | 0.003613 | 0.000055 | 0.000028 | 0.004461 | 0.7289% |

DINOv3-IML is 0.069010 above RelayFormer by macro AP, but its fixed macro
F1 is 0.009666 lower. Its micro F1 is higher. Continuous ranking quality and
one fixed operating point therefore tell different parts of the result.

This is a checkpoint-level robustness comparison, not a controlled
architecture ablation: training sets, model resolutions, preprocessing,
output construction, and public thresholds differ.

## 12. Independent audit and tests

The final analyzer reports:

| Audit item | Result |
|---|---:|
| Physical/latest result rows | 550 / 550 |
| Unique expected result IDs | 550 |
| Pinned official source files | 17 |
| Pinned adapter-contract files | 4 |
| Files checked end to end | 3,575 |
| 32-to-512 logit replay max absolute error | `4.7683716e-6` |
| Model-probability replay max absolute error | `5.9604645e-7` |
| Native-probability replay max absolute error | `2.0265579e-6` |
| Threshold-mask disagreements | **0 pixels** |
| Smoke-prefix reproducibility | 10/10 images, exact |
| Duplicate result rows | **0** |
| Provenance status | `ok` |
| Artifact status | `ok` |

The analyzer uses separate, bounded tolerances for CPU/CUDA float32
operation-order differences: `5e-6` for resized logits, `1e-6` for model
probabilities, and `3e-6` for native restoration. File hashes, dtypes,
shapes, geometry, thresholds, and PNG masks remain exact. Full-set scanning
showed zero threshold changes. This treatment follows
[PyTorch's numerical-accuracy guidance](https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html),
which does not promise bitwise-identical CPU and GPU floating-point results.
Tests accept the observed rounding envelope but reject larger coordinated
tampering and even a one-ULP change that flips a strict-threshold pixel.

DINOv3-IML runner, metric, and analyzer tests: **44 passed**. The complete
repository suite in the compatible baseline environment reports
**320 passed, 4 skipped**. The three warnings are upstream SciPy and timm
deprecations.

## 13. Runtime and artifacts

The recorded environment uses CPython 3.12.3, PyTorch
`2.8.0.dev20250627+cu128`, PEFT 0.18.1, Transformers 4.53.2,
Accelerate 1.9.0, NumPy 2.2.6, Pillow 11.1.0, and scikit-learn 1.5.2 on an
NVIDIA L20Z. CUDA math is deterministic, TF32 and non-math SDPA kernels are
disabled, and `CUBLAS_WORKSPACE_CONFIG=:4096:8`.

Recorded forward-only latency over 550 images:

| Statistic | Milliseconds |
|---|---:|
| Median | 47.107 |
| P95 | 66.488 |
| Mean including first-image warm-up | 46.723 |
| Maximum first image | 406.719 |

The timing encloses `model.predict` plus CUDA synchronization. It excludes
decode, preprocessing, artifact writes, and analysis, so it is not an
end-to-end service latency.

Peak allocated CUDA memory is 1,474,092,544 bytes, approximately 1.37 GiB.
The complete full-run artifact directory is 4,695,129,799 bytes,
approximately 4.37 GiB. It retains four float32 arrays and one lossless mask
for every image.

Primary machine-readable files:

- `results/opensource/dinov3_iml/dinov3_iml_cat_vitl_lora_r32_checkpoint48_mouse_canonical_v1_full275_20260724.jsonl`
- `results/opensource/dinov3_iml/dinov3_iml_cat_vitl_lora_r32_checkpoint48_mouse_canonical_v1_full275_20260724.run_manifest.json`
- `results/opensource/dinov3_iml/dinov3_iml_cat_vitl_lora_r32_checkpoint48_mouse_canonical_v1_full275_20260724.summary.json`
- `results/opensource/dinov3_iml/dinov3_iml_cat_vitl_lora_r32_checkpoint48_mouse_canonical_v1_full275_20260724.analysis.json`
- `outputs/opensource/dinov3_iml/dinov3_iml_cat_vitl_lora_r32_checkpoint48_mouse_canonical_v1_full275_20260724/`

Implementation:

- `eval/opensource/run_dinov3_iml.py`
- `eval/opensource/dinov3_iml_metrics.py`
- `eval/opensource/analyze_dinov3_iml_run.py`
- `tests/test_run_dinov3_iml.py`
- `tests/test_dinov3_iml_metrics.py`
- `tests/test_analyze_dinov3_iml_run.py`

## 14. Reproduction

Full inference or artifact-validating resume:

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=/root/claimforge-benchmark \
/root/.cache/claimforge/venvs/dinov3iml-ba45b0a/bin/python \
  -m eval.opensource.run_dinov3_iml \
  --repo-root /root/claimforge-benchmark \
  --run-id \
    dinov3_iml_cat_vitl_lora_r32_checkpoint48_mouse_canonical_v1_full275_20260724 \
  --device cuda:0 \
  --bootstrap-samples 1000 \
  --fail-fast
```

Independent analysis:

```bash
PYTHONPATH=/root/claimforge-benchmark \
python -m eval.opensource.analyze_dinov3_iml_run \
  --repo-root /root/claimforge-benchmark \
  --run-id \
    dinov3_iml_cat_vitl_lora_r32_checkpoint48_mouse_canonical_v1_full275_20260724 \
  --results-dir results/opensource/dinov3_iml \
  --inputs outputs/opensource/mouse_canonical_v1/inputs.jsonl \
  --dinov3-iml-root /root/.cache/claimforge/third_party/DINOv3-IML \
  --dinov3-root /root/.cache/claimforge/third_party/dinov3 \
  --bootstrap-iterations 1000 \
  --bootstrap-seed 20260724 \
  --prefix-run-id \
    dinov3_iml_cat_vitl_lora_r32_checkpoint48_mouse_canonical_v1_smoke5_20260724 \
  --prefix-results-dir results/opensource/dinov3_iml
```

## 15. Bottom line and next method

DINOv3-IML is a meaningful open-source result and ranks third of ten
completed local methods by native macro AP. Its foundation-model
representation produces excellent masks on a nontrivial subset, especially
in the largest edit quintile. It does not solve CLAIMFORGE's central tiny
insertion regime: median AP is `0.040193`, median F1 is zero, 151/275 forged
images receive no exact-GT pixel at the registered threshold, and the formal
box-hit rate is only 8.36%.

For the CLAIMFORGE main table, report DINOv3-IML as **T2 only, third of ten by
native macro AP, strongly edit-size dependent, with a disclosed continuous
output adapter and a checkpoint-license boundary inherited from the embedded
Meta DINOv3 backbone**.

The next and final frozen local method is **NFA-ViT / BR-Gen**, using its
official BR-Gen checkpoint with native T1 and T2 outputs.
