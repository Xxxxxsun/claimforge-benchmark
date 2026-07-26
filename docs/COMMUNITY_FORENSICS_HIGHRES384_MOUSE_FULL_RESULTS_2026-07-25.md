# Community Forensics High-res 384 on the canonical Mouse set (2026-07-25)

## 1. Status and headline

Community Forensics has completed the frozen **local-splice** condition on all
275 matched Mouse tasks, or 550 canonical JPEG images. The run contains
550/550 valid images, 275/275 complete real/forged pairs, zero errors, zero
missing images, and exactly one physical successful result row per image.
The independent analyzer subsequently reopened and preprocessed all 550
images, executed 550 fresh complete ViT forwards, and reproduced every stored
384-dimensional feature, raw logit, probability, and decision with maximum
absolute error `0.0`.

The released whole-image probability is effectively a random-ranking score on
this condition:

- AUROC is **0.502340**, with a 1,000-task-pair bootstrap 95% percentile CI
  of **[0.500674, 0.504873]**;
- average precision is **0.504511**, CI **[0.502691, 0.511090]**;
- TPR at a real-only 5% FPR operating point is **4.7273%**, CI
  **[3.6364%, 5.8182%]**;
- the released strict rule `probability > 0.5` detects **1/275 forged
  images** and falsely flags **1/275 real images**; and
- the resulting confusion matrix is **TP=1, FP=1, FN=274, TN=274**.

Those two positive images are not independent successes and failures. They
are the real and forged members of the **same** partially visible pair,
`lodging_297_slot_001`: the real score is `0.6781423092` and the forged score
is `0.7297614217`. Thus the detector identifies no pair for which the forged
image is positive while its matched real control remains negative.

Community Forensics is a whole-image binary classifier. It emits one logit
and has no native dense manipulation map. Consequently, this run is valid for
**T1 whole-image detection only**; **T2 localization and the joint T1/T2
score are N/A**. Classifier features or attention must not be relabeled as
localization output.

This finishes Community Forensics only for the local-splice condition. It
does **not** complete the method's intended fully synthetic-image evaluation:
the same-domain fully synthetic contrast set has not yet been built or run.

## 2. Pinned paper, source, model, processor, and license

The method is described in the CVPR 2025 paper
[“Community Forensics: Using Thousands of Generators to Train Fake Image
Detectors”](https://openaccess.thecvf.com/content/CVPR2025/html/Park_Community_Forensics_Using_Thousands_of_Generators_to_Train_Fake_Image_CVPR_2025_paper.html).
The protocol uses the authors'
[official repository](https://github.com/JeongsooP/Community-Forensics) and
the Andrew Owens Research Lab's official Hugging Face releases.

All executable assets were frozen before any Mouse model score was inspected:

| Asset | Frozen revision | Size / structure | SHA-256 or role |
|---|---|---:|---|
| [Official GitHub source](https://github.com/JeongsooP/Community-Forensics/tree/ee5b71d43db0f3779e1edd64ee927b13f2dd6ad4) | `ee5b71d43db0f3779e1edd64ee927b13f2dd6ad4` | main branch snapshot | Training/evaluation implementation |
| [Official `eval_single` source](https://github.com/JeongsooP/Community-Forensics/tree/5e52ed690bdbd609f9bb1705c4c80d11872a05bd) | `5e52ed690bdbd609f9bb1705c4c80d11872a05bd` | separate official branch | RGB, resize, crop, sigmoid, and strict-threshold single-image semantics |
| [High-res model](https://huggingface.co/OwensLab/commfor-model-384/tree/6076002bf0d9dd37537f965ee2f06f826c333b61) | `6076002bf0d9dd37537f965ee2f06f826c333b61` | 87,262,324-byte `model.safetensors`; 152 FP32 tensors; 21,811,969 state elements | `b89f36275f3bf5e2b040eee36597a8f19db051bff9a473a9cf7b2466284fb387` |
| [Official processor](https://huggingface.co/OwensLab/commfor-data-preprocessor/tree/3540a3f0d688f8bf492a8aed48613b891f88047e) | `3540a3f0d688f8bf492a8aed48613b891f88047e` | six pinned repository files | Exact test transform |

The checkpoint is loaded with `safetensors.torch.load_file`; no pickle is
executed. Its complete state strictly covers the constructed network, with no
missing or unexpected keys. The frozen source/model/processor bundle digest
is `810a7592a82f09cbf638985e9c59eed9ebd2c3ff28ebab97f348bfd3c69b7fb3`.

The GitHub code carries an
[MIT license](https://github.com/JeongsooP/Community-Forensics/blob/ee5b71d43db0f3779e1edd64ee927b13f2dd6ad4/LICENSE),
and both selected Hugging Face cards declare MIT metadata. This establishes
MIT terms for the pinned code, model release, and processor release. It is
not a separate legal audit of every source image or generator represented in
the Community Forensics training dataset.

The public 224-pixel checkpoint was excluded from the primary protocol before
Mouse scoring. The paper identifies the 384-pixel “High res.” variant as its
best-performing detector and uses it in the subsequent experiments, so Mouse
performance was not used to select between released resolutions.

## 3. What the method does and why it is strong in its intended setting

Community Forensics makes a deliberately simple architectural claim: a
standard image classifier can generalize to unseen generators if its training
data contains enough generator diversity.

The paper's
[dataset description](https://arxiv.org/html/2411.04125#S3) reports 2.7
million generated images sampled from **4,803 distinct generators**. The
count has two related but different meanings:

- **4,803** is total dataset coverage: 4,763 systematically collected latent
  diffusion models, 19 manually selected models, 11 commercial models, and
  10 additional held-out evaluation models.
- The paper holds out 21 evaluation generators—the 11 commercial models and
  10 additional models—so the training split contains **4,782 generators**:
  4,763 systematic plus 19 manual.

The training experiment pairs the 2.7 million generated images with 2.7
million real images, for 5.4 million binary-classification examples. The
paper's central ablation shows that performance continues to improve as more
generators are added even when many share a latent-diffusion family. The
interpretation is that thousands of implementations expose the classifier to
variation in architecture, fine-tuning, content, resizing, compression, and
sampling pipelines, discouraging dependence on a single generator's narrow
fingerprint.

For the released High-res model, the operational computation is:

```text
Pillow RGB
-> aspect-preserving bilinear Resize(short edge = 440)
-> CenterCrop(384 x 384)
-> float32 tensor in [0, 1]
-> ImageNet mean/std normalization
-> end-to-end fine-tuned ViT-S/16
-> 384-dimensional classifier input
-> one linear logit
-> float32 sigmoid probability
-> generated iff probability > 0.5
```

The backbone is not frozen during the paper's training. This matters: the
authors report that end-to-end adaptation is better than a frozen feature
extractor, allowing the representation itself to learn broadly shared
forensic evidence.

### Released-code versus paper naming

The paper describes a “plain CLIP-ViT-S” pretrained with a CLIP objective and
mentions LAION-2B, ImageNet-21K, and ImageNet-1K. The released executable code,
however, explicitly constructs the timm identifier
`vit_small_patch16_384.augreg_in21k_ft_in1k`, then replaces its head and loads
the complete Community Forensics checkpoint. That identifier is what this
benchmark records and executes.

The complete checkpoint overwrites every constructed parameter, so the
operational released detector is unambiguous. The relationship between the
paper's “CLIP-ViT-S” prose and the released timm AugReg identifier is not
fully documented; this report therefore does not claim a stricter
pretraining identity than the released code establishes.

The paper reports mean mAP `0.994` and mean accuracy `0.923` for its High-res
model across its benchmark table. Those figures are context only. They were
not reproduced here, use fully generated images from different datasets, and
are not directly comparable to CLAIMFORGE's small local insertions.

## 4. Frozen Mouse protocol

The primary protocol was fixed before the full-set run:

| Component | Frozen value |
|---|---|
| Dataset | `claimforge-mouse-good275-canonical-jpeg-q95-v1` |
| Coverage | 275 matched tasks; 550 canonical JPEG images |
| Model | `OwensLab/commfor-model-384`, High-res ViT-S/16 |
| Architecture | timm 1.0.15 `vit_small_patch16_384.augreg_in21k_ft_in1k` |
| Construction | `pretrained=False`, one-output linear head, then strict full-state load |
| Decode | `PIL.Image.open(...).convert("RGB")`; no EXIF transpose or ICC conversion |
| Resize | Short edge 440, aspect ratio preserved, Pillow bilinear with antialiasing |
| Crop | torchvision `CenterCrop(384)` |
| Tensor | RGB uint8 divided by 255 into float32 |
| Normalize | mean `[0.485, 0.456, 0.406]`; std `[0.229, 0.224, 0.225]` |
| Inference | Batch 1, `eval()`, float32, no AMP, no TF32 |
| Primary score | float32 `sigmoid(raw_logit)`, higher means generated |
| Released decision | strict `probability > 0.5` |
| 5% FPR point | real-only 95th percentile, `method="higher"`, strict `>` |
| Bootstrap | 1,000 complete-task-pair resamples, seed `20260724` |
| T2 | N/A; the release has no native dense output |

The official repository's wrapper asks timm for `pretrained=True` before
loading the complete Community Forensics state. The benchmark constructs the
same network with `pretrained=False` to avoid a redundant, mutable base-model
download that would immediately be overwritten. Strict state loading proves
full coverage, and inference is network-disabled.

Before Mouse inference, this path was gated against the five DALL-E 2 images
and probabilities embedded in the official Hugging Face notebook. All five
passed the frozen absolute tolerance of `1e-5`; the maximum probability
difference was `4.530e-6`.

The official probability remains the sole primary score. Unlike NPR's
float32 underflow case, every Community Forensics probability in this run is
finite and strictly inside `(0, 1)`, from `1.878e-7` to `0.729761`. No
post-hoc raw-logit score variant is introduced. The audit finds zero exact
zeros, zero exact ones, and 465 unique probabilities among 550 images.

## 5. Center-crop visibility and exact crop equality

The official resize-and-center-crop transform does not necessarily retain a
small peripheral insertion. Visibility was computed before scores using the
positive pixels of each forged image's exact-difference GT, mapped through
the frozen resize/crop geometry:

| Domain | Full | Partial | None | Total |
|---|---:|---:|---:|---:|
| Lodging | 96 | 14 | 37 | 147 |
| Restaurant | 66 | 18 | 44 | 128 |
| **All** | **162** | **32** | **81** | **275** |

The mean retained GT fraction is `0.646589`; the median is `1.0`.
`edit_visibility` is an input-condition stratum copied to both members of a
matched pair. It is not a prediction by Community Forensics.

Independent hash comparison within the stored run gives a particularly clear
crop audit:

| Visibility | Pairs | Exactly equal RGB crops | Equal tensors | Equal features | Equal probabilities |
|---|---:|---:|---:|---:|---:|
| Full | 162 | 0 | 0 | 0 | 0 |
| Partial | 32 | 0 | 0 | 0 | 0 |
| None | 81 | 80 | 80 | 80 | 80 |

Thus 80 of the 81 `none` pairs are literally the same classifier input and
must tie. The sole exception is `lodging_191_slot_001`: no positive
exact-difference pixel center falls inside the crop, but the continuous edit
box touches the crop boundary. Bilinear support leaks across that boundary,
leaving exactly seven changed pixels along the crop's left edge, each with
absolute uint8 difference one. Its paired probability change is only
`1.535e-10`.

Cropping is therefore a major measured part of the failure, but not the whole
explanation. Even the 162 `full` pairs produce AUROC only `0.505144`, and the
released threshold detects none of their forged images.

## 6. Primary whole-image result

Brackets below are 95% percentile intervals from 1,000 resamples of complete
`task_id` pairs.

| Metric | Estimate | 95% CI |
|---|---:|---:|
| AUROC | 0.502340 | [0.500674, 0.504873] |
| Average precision | 0.504511 | [0.502691, 0.511090] |
| TPR @ target FPR 5% | 0.047273 | [0.036364, 0.058182] |
| Real-only 5% FPR threshold | 0.023390 | [0.010563, 0.063245] |
| Actual FPR at that threshold | 0.047273 | [0.036364, 0.047273] |
| Accuracy @ 0.5 | 0.500000 | [0.500000, 0.500000] |
| Balanced accuracy @ 0.5 | 0.500000 | [0.500000, 0.500000] |
| Precision @ 0.5 | 0.500000 | [0.000000, 0.500000] |
| Recall @ 0.5 | 0.003636 | [0.000000, 0.010909] |
| F1 @ 0.5 | 0.007220 | [0.000000, 0.021352] |
| Specificity @ 0.5 | 0.996364 | [0.989091, 1.000000] |
| Strict paired ranking | 0.436364 | [0.378182, 0.494545] |
| Mean forged-real probability delta | 0.000244 | [-0.000107, 0.000733] |

| Operating point | TP | FP | FN | TN |
|---|---:|---:|---:|---:|
| Released `probability > 0.5` | 1 | 1 | 274 | 274 |

The real-only 5% FPR rule flags 13 real and 13 forged images, so its empirical
TPR and FPR are both 13/275. The narrow paired-bootstrap interval around an
AUROC slightly above 0.5 does not make the effect operationally useful:
ranking quality differs from chance by only 0.00234, while released-threshold
recall is 0.36%.

Official probability distributions are:

| Kind | Min | Mean | Median | P05 | P95 | Max |
|---|---:|---:|---:|---:|---:|---:|
| Forged | `1.919e-7` | 0.007610 | `6.789e-5` | `1.079e-6` | 0.021952 | 0.729761 |
| Real | `1.878e-7` | 0.007367 | `5.762e-5` | `1.042e-6` | 0.022672 | 0.678142 |

## 7. Domain and crop-visibility strata

### Domain strata

| Domain | Pairs | AUROC [95% CI] | AP [95% CI] | TPR@5% FPR [95% CI] | TP / FP | Paired W / L / T |
|---|---:|---:|---:|---:|---:|---:|
| Lodging | 147 | 0.502314 [0.499234, 0.506711] | 0.507320 [0.503917, 0.519295] | 0.054422 [0.034014, 0.068027] | 1 / 1 | 71 / 40 / 36 |
| Restaurant | 128 | 0.502258 [0.499023, 0.506744] | 0.503637 [0.502619, 0.515634] | 0.046875 [0.031250, 0.062500] | 0 / 0 | 49 / 35 / 44 |

| Domain | Strict paired ranking [95% CI] | Mean paired delta [95% CI] | Exact sign-test p |
|---|---:|---:|---:|
| Lodging | 0.482993 [0.401361, 0.564626] | 0.000430 [-0.000168, 0.001306] | 0.004196 |
| Restaurant | 0.382812 [0.296875, 0.468750] | 0.000029 [-0.000116, 0.000194] | 0.155656 |

Both domains remain effectively random as standalone whole-image
classifiers. Lodging's sign-test result comes from excluding its 36 ties and
does not imply useful global ranking or thresholded detection.

### Visibility strata

| Visibility | Pairs | AUROC [95% CI] | AP [95% CI] | TPR@5% FPR [95% CI] | TP / FP | Paired W / L / T |
|---|---:|---:|---:|---:|---:|---:|
| Full | 162 | 0.505144 [0.502018, 0.510213] | 0.509830 [0.508281, 0.521828] | 0.049383 [0.043056, 0.074074] | 0 / 0 | 103 / 59 / 0 |
| None | 81 | 0.500152 [0.500000, 0.501143] | 0.500041 [0.500000, 0.500394] | 0.049383 [0.024691, 0.049383] | 0 / 0 | 1 / 0 / 80 |
| Partial | 32 | 0.499023 [0.479492, 0.518555] | 0.521697 [0.505662, 0.565154] | 0.031250 [0.000000, 0.093750] | 1 / 1 | 16 / 16 / 0 |

| Visibility | Strict paired ranking [95% CI] | Mean paired delta [95% CI] | Exact sign-test p |
|---|---:|---:|---:|
| Full | 0.635802 [0.561728, 0.710031] | 0.000144 [-0.000147, 0.000490] | 0.000681 |
| None | 0.012346 [0.000000, 0.037037] | `1.895e-12` [0, `5.684e-12`] | 1.000000 |
| Partial | 0.500000 [0.343750, 0.656250] | 0.001367 [-0.000526, 0.004767] | 1.000000 |

The full-visibility sign test is evidence of a weak directional response:
forged scores rise more often than they fall when the background scene is
held fixed. It is not evidence of a useful image-only detector. Full-stratum
AUROC remains `0.505144`, the mean-delta CI includes zero, and every
full-visibility image remains below the released threshold.

## 8. Why paired sensitivity and global random ranking coexist

Across all 275 pairs, the forged probability is greater than the matched real
probability 120 times, smaller 75 times, and exactly equal 80 times. The
two-sided exact sign test over the 195 non-ties gives `p=0.001561`.

This does not contradict the global AUROC of `0.502340`:

1. A paired comparison subtracts nearly the same scene, camera, JPEG, and
   crop content, exposing a very small edit-associated shift.
2. AUROC compares each forged image against every real image. Across unrelated
   scenes, content and acquisition variation dominate the classifier score.
3. The mean paired shift is only `+0.000244`, with a CI crossing zero, while
   real-image scores have standard deviation `0.04576`—about 188 times the
   mean shift—and the combined score range spans roughly seven orders of
   magnitude.
4. A deployed one-image detector does not receive the matched real
   counterfactual, so it cannot exploit pair subtraction.

The sign test and mean delta also answer different questions. The sign test
ignores 80 ties and weights every non-zero direction equally; the mean is
sensitive to a few larger positive and negative changes. Both are retained,
but neither supersedes the official probability AUROC, AP, or released
threshold.

The narrow conclusion is that the model notices some fully visible local
changes in matched comparisons, but that signal is swamped by scene-level
variation and is not calibrated for small local insertions.

## 9. Determinism and runtime

Two separate final-code CUDA smokes, each covering the same five complete
pairs, used identical model, preprocessing, runtime, and 1,000-bootstrap
configuration fingerprints. After excluding expected run IDs, timestamps,
and artifact paths, all ten rows match exactly for:

- decoded, resized, cropped, and normalized-tensor hashes;
- persisted 384-dimensional feature contents and hashes;
- raw logits and float32 probabilities;
- strict decisions; and
- summary detection and paired metrics.

The raw `results.jsonl` files are not byte-identical because provenance fields
include different run IDs and completion times. The inference-relevant
contents are exact.

The independent audit also compared CUDA smoke A's ten ordered images against
the corresponding full-run prefix stored in a separate directory with
separate feature files. Decoded/resized/crop/tensor hashes, 384-dimensional
features, logits, probabilities, aliases, and decisions are byte-for-byte
identical for all ten images.

The full run used one NVIDIA L20Z, batch size one, deterministic algorithms,
float32, no autocast, no TF32, cuDNN disabled, and no network access.

| Runtime quantity | Value |
|---|---:|
| Model-forward latency, mean | 8.606 ms/image |
| Model-forward latency, median | 6.090 ms/image |
| Model-forward latency, P95 | 15.956 ms/image |
| Model-forward latency, max | 214.094 ms/image |
| Decode/resize/crop/normalize/hash preparation, mean | 103.501 ms/image |
| Peak allocated CUDA memory | 133,535,232 bytes (127.349 MiB) |
| Full runner wall time | 111.995 s |
| End-to-end runner throughput | 4.911 images/s |

The wall time additionally includes feature persistence and readback,
per-row JSONL writes, hashing, and final metric/bootstrap computation. It
should not be inferred by simply adding the two per-image timing fields.

Runtime versions include Python 3.12.3, PyTorch
`2.8.0.dev20250627+cu128`, torchvision
`0.23.0.dev20250627+cu128`, timm `1.0.15`, Pillow `11.1.0`,
safetensors `0.5.2`, NumPy `2.2.6`, and scikit-learn `1.5.2`.

## 10. Independent audit

Audit status is **`audited`**. The analyzer does not trust stored runner
scores or stored features as model inputs. For every canonical path, it
performs a fresh Pillow decode, uses an independent implementation of the
frozen resize/crop/normalization contract, executes the complete pinned ViT,
captures the fresh 384-dimensional classifier input, and only then loads the
persisted feature for comparison. It independently applies
`torch.nn.functional.linear` and float32 sigmoid to the persisted feature as
a second classifier-head replay.

The audit verifies:

- the pinned GitHub source and `eval_single` commits, source-file hashes,
  MIT records, HF model and processor revisions, checkpoint bytes, 152-tensor
  schema, and complete strict model-state coverage;
- adapter, runtime, dataset, expected-input, manifest, results, summary, and
  feature-artifact contracts;
- 550 physical result rows, 550 unique sample IDs, no duplicates, no
  recovered errors, and 275 exact complete pairs;
- 550 fresh RGB decodes, resized images, center crops, normalized tensors,
  complete model forwards, and captured classifier features;
- all 550 persisted 384-dimensional features, plus manual head replay of
  every logit, float32 probability, alias, and strict decision;
- independent recomputation of coverage, score distributions, threshold
  metrics, paired rankings and sign tests, domain/visibility strata, and all
  1,000-pair bootstrap intervals; and
- explicit rejection of localization and joint outputs because the release
  has no native T2 prediction.

Maximum absolute replay differences are:

| Replayed quantity | Maximum absolute difference |
|---|---:|
| 384-dimensional feature | 0.0 |
| Raw logit | 0.0 |
| Float32 probability | 0.0 |

The independently recomputed complete summary matches the stored summary.
The analyzer additionally validates ten smoke-A prefix images in independent
run directories and feature paths; copied full-run artifacts are rejected.

## 11. Scope, limitations, and conclusion

The supported local-splice claim is:

> The pinned Community Forensics High-res 384 release does not transfer to
> CLAIMFORGE Mouse's small local insertions as a standalone whole-image
> detector. Its official probability has AUROC 0.5023, its real-only 5% FPR
> point has 4.73% TPR, and its released threshold identifies only one forged
> image while falsely flagging that image's matched real control.

Important limits are:

- this is a local-splice stress test, not the fully generated-image task on
  which Community Forensics was trained;
- 81/275 edits have no exact-difference GT pixel center inside the official
  crop, although the full-visibility stratum also remains ineffective;
- the result covers the frozen canonical JPEG-Q95 Mouse construction, two
  domains, and the released 384-pixel checkpoint, not every possible
  compression, crop, checkpoint, or generator;
- no Mouse labels were used to tune the model or select an oracle threshold;
- no localization claim is possible because the model has no native T2
  output; and
- independent replay establishes execution fidelity, not external validity
  beyond this frozen dataset and protocol.

The next frozen local-splice method is **SPAI**, the planned any-resolution
spectral/OOD baseline. Separately, the benchmark still needs a same-domain
fully synthetic lodging/restaurant contrast. Until that contrast exists,
Community Forensics is **local-splice complete but contrast incomplete**.

## 12. Reproducibility artifacts

Primary run:

```text
results/opensource/community_forensics/
  community_forensics_highres_vit_s16_384_mouse_canonical_v1_full275_20260725/
```

Important artifacts are:

- `manifest.json` — frozen source, assets, model construction, preprocessing,
  runtime, dataset, metric, license, and output contracts;
- `expected_inputs.jsonl` — exact ordered 550-image input ledger;
- `results.jsonl` — one physical successful row per canonical image;
- `features/*.npy` — 550 persisted 384-dimensional classifier-input
  features;
- `summary.json` — complete official-probability metrics, paired bootstrap,
  domain strata, and visibility strata; and
- `analysis.json` — physically independent full-model replay, prefix
  reproducibility, metric recomputation, and provenance audit.

The reusable runner and metric entry points are:

```text
eval/opensource/run_community_forensics.py
eval/opensource/community_forensics_metrics.py
eval/opensource/analyze_community_forensics_run.py
tests/test_run_community_forensics.py
tests/test_community_forensics_metrics.py
tests/test_analyze_community_forensics_run.py
```

The completed run can be regenerated from the repository root with the
pinned local assets:

```bash
python -m eval.opensource.run_community_forensics \
  --run-id community_forensics_highres_vit_s16_384_mouse_canonical_v1_full275_20260725 \
  --device cuda:0 \
  --bootstrap-samples 1000 \
  --bootstrap-seed 20260724 \
  --fail-fast
```

The physically independent audit, including the ten-image smoke-A prefix
comparison, is reproduced with:

```bash
python -m eval.opensource.analyze_community_forensics_run \
  --run-id community_forensics_highres_vit_s16_384_mouse_canonical_v1_full275_20260725 \
  --prefix-run-id community_forensics_highres_vit_s16_384_mouse_pair5_cuda_smoke_a_20260725
```

For new canonical data, use a new immutable run ID and dataset manifest.
Successful rows are resumable, while changes to the asset, source, adapter,
runtime, selection, or metric contracts are rejected rather than silently
mixed.
