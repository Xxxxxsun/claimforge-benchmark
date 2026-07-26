# UniversalFakeDetect Ours LC on the canonical mouse set (2026-07-24)

## 1. Status and headline

UniversalFakeDetect (UFD) **Ours LC** has completed both frozen
preprocessing conditions on all 275 matched mouse tasks, or 550 canonical
JPEG images. Both runs have 550/550 valid images, 275/275 complete pairs,
zero errors, zero missing rows, and independently audited artifacts.

The result designated for comparison is the **current official repository
HEAD** contract:

```text
RGB -> native-resolution CenterCrop(224) -> ToTensor -> CLIP normalize
```

Its result is effectively chance:

- AUROC **0.499650**, task-paired bootstrap 95% CI
  **[0.493190, 0.505666]**;
- average precision **0.497293**, CI **[0.489990, 0.503872]**;
- TPR at a real-only 5% FPR operating point **4.0000%**, CI
  **[2.9091%, 4.7273%]**;
- accuracy at the released strict rule `ai_score > 0.5` **49.8182%**,
  CI **[49.4545%, 50.0000%]**;
- confusion at that rule: **TP=10, FP=11, FN=265, TN=264**;
- forged scores exceed matched-real scores in only **17/275** pairs,
  with 12 losses and 246 ties; and
- mean paired score change is **-0.003805**, CI
  **[-0.010888, 0.000440]**.

The large tie count is real, not a rounding artifact. Under current HEAD,
only 28/275 edits have any pre-canonical GT pixel center inside the 224 crop.
Of the 247 `none` pairs, 246 independently decoded canonical RGB crops are
exactly equal and therefore produce exactly equal tensors, features, logits,
scores, and decisions.

The checkpoint-era `Resize(256) -> CenterCrop(224)` result is reported only
as a **sensitivity analysis**, never as a second primary result. It expands
the nominally visible set to 195/275 pairs, but image AUROC remains
**0.503260** and the released threshold predicts every image as real.

UFD is a whole-image classifier. It emits no native dense manipulation map,
so **T2 and the joint score are N/A**. This completes only the local-splice
half of the whole-image experiment. A same-domain fully synthetic control
still has to be built and run before judging UFD on its intended task.

## 2. Pinned method, source, and paper context

The method is from the
[CVPR 2023 paper](https://openaccess.thecvf.com/content/CVPR2023/html/Ojha_Towards_Universal_Fake_Image_Detectors_That_Generalize_Across_Generative_Models_CVPR_2023_paper.html),
“Towards Universal Fake Image Detectors that Generalize Across Generative
Models.” The authors also provide an
[HTML paper with appendix](https://arxiv.org/html/2302.10174) and an
[official project page](https://utkarshojha.github.io/universal-fake-detection/).

This run pins the authors'
[official repository](https://github.com/WisconsinAIVision/UniversalFakeDetect)
at commit
[`76a0e3e60a8a06458707a625d269ba815a2e5919`](https://github.com/WisconsinAIVision/UniversalFakeDetect/tree/76a0e3e60a8a06458707a625d269ba815a2e5919).
The evaluated release is exactly the repository's `Ours (L/14 + LC)`:
OpenAI CLIP ViT-L/14 plus the bundled `fc_weights.pth`. The paper's nearest
neighbor variant is not represented by a released feature bank/checkpoint
and was not substituted here.

The paper reports **93.38 total mAP** and **81.38% average accuracy** for
Ours LC across its 19 generator settings. Those are paper context only:
this run neither uses the paper test datasets nor claims to reproduce those
numbers. All paper-generated test images are 256x256, while Mouse contains
variable-resolution camera images with small local diffusion insertions.

## 3. What the method does and why the idea is strong

The paper argues that an end-to-end fake classifier can latch onto the
fingerprint of the training generator. Unseen fakes lacking that fingerprint
then fall into a broad “real” sink class. UFD instead keeps a representation
learned for an unrelated, broad image-language task fixed:

1. OpenAI CLIP ViT-L/14 encodes the 224x224 image into one 768-dimensional
   global image feature.
2. The CLIP backbone remains frozen.
3. Only a one-output linear probe is trained on ProGAN/real examples.
4. The released raw logit is converted with float32 sigmoid.
5. `ai_score = sigmoid(logit)` and fake means strictly `ai_score > 0.5`.

The head contains only 768 weights plus one bias, or **769 parameters**.
The feature is not L2-normalized before the head.

This is powerful in the paper setting because CLIP was exposed to broad
internet-scale image-text data, its representation was not optimized to
memorize a particular fake-vs-real shortcut, and ViT-L/14's 14-pixel patches
retain relatively fine visual structure. The same design also explains this
Mouse result: one global embedding is dominated by the untouched photograph,
and the repository evaluator may crop the small insertion out completely.

## 4. Frozen evaluation protocol

Both conditions use the same 275 real/forged pairs, hashes, model assets,
float32 inference, batch size one, seed `20260724`, and 1,000 task-pair
bootstrap resamples. Higher scores mean more likely fake.

The released operating point is the strict comparison `ai_score > 0.5`.
The additional 5% FPR threshold is computed only from real scores using the
95th percentile (`method="higher"`); it is not an oracle best threshold.

The main metrics are T1 AUROC, AP, TPR@5% FPR, released-threshold confusion
metrics, strict paired ranking, paired score delta, and an exact two-sided
sign test over non-ties. Bootstrap CIs resample complete `task_id` pairs, so
the matched dependence is preserved.

The two immutable geometry profiles are:

| Status | Profile | Exact transform |
|---|---|---|
| **Primary** | `current_head_native_center_crop224` | Pillow RGB -> native `CenterCrop(224)` -> tensor -> CLIP normalize |
| Sensitivity only | `checkpoint_era_resize256_center_crop224` | Pillow RGB -> bilinear short-side `Resize(256)` -> `CenterCrop(224)` -> tensor -> CLIP normalize |

No standard `clip.load()` bicubic transform, test-time augmentation, oracle
threshold, or full-image resizing is silently inserted into the primary run.

## 5. Complete overall metrics

Values in brackets are task-paired bootstrap 95% percentile CIs.

| Metric | Current HEAD primary | Checkpoint-era sensitivity |
|---|---:|---:|
| AUROC | 0.499650 [0.493190, 0.505666] | 0.503260 [0.496806, 0.508946] |
| Average precision | 0.497293 [0.489990, 0.503872] | 0.510188 [0.505536, 0.521318] |
| TPR @ target FPR 5% | 0.040000 [0.029091, 0.047273] | 0.047273 [0.036364, 0.069091] |
| Real-only threshold | 0.299962 [0.176428, 0.594233] | 0.018843 [0.009159, 0.028800] |
| Actual FPR | 0.047273 [0.036364, 0.047273] | 0.047273 [0.040000, 0.047273] |
| Accuracy @ 0.5 | 0.498182 [0.494545, 0.500000] | 0.500000 [0.500000, 0.500000] |
| Balanced accuracy @ 0.5 | 0.498182 [0.494545, 0.500000] | 0.500000 [0.500000, 0.500000] |
| Precision @ 0.5 | 0.476190 [0.416667, 0.500000] | 0.000000 [0.000000, 0.000000] |
| Recall @ 0.5 | 0.036364 [0.018182, 0.058182] | 0.000000 [0.000000, 0.000000] |
| F1 @ 0.5 | 0.067568 [0.034840, 0.103905] | 0.000000 [0.000000, 0.000000] |
| Specificity @ 0.5 | 0.960000 [0.938091, 0.978182] | 1.000000 [1.000000, 1.000000] |
| Strict paired ranking | 0.061818 [0.036364, 0.090909] | 0.443636 [0.385455, 0.501818] |
| Mean forged-real score delta | -0.003805 [-0.010888, 0.000440] | 0.000498 [0.000045, 0.001212] |

| Condition | TP | FP | FN | TN |
|---|---:|---:|---:|---:|
| Current HEAD primary | 10 | 11 | 265 | 264 |
| Checkpoint-era sensitivity | 0 | 0 | 275 | 275 |

The checkpoint-era paired shift is statistically detectable among non-ties,
but it is tiny and does not yield image-level separation or a usable released
operating point.

## 6. Complete paired results and score distributions

| Condition | Wins / losses / ties | Non-ties | Exact sign-test p | Strict accuracy |
|---|---:|---:|---:|---:|
| Current HEAD primary | 17 / 12 / 246 | 29 | 0.458258 | 0.061818 |
| Checkpoint-era sensitivity | 122 / 74 / 79 | 196 | 0.000746 | 0.443636 |

| Condition | Min delta | Mean | Median | P05 | P95 | Max |
|---|---:|---:|---:|---:|---:|---:|
| Current HEAD primary | -0.676777 | -0.003805 | 0 | 0 | 0.000023 | 0.046603 |
| Checkpoint-era sensitivity | -0.010344 | 0.000498 | 0 | -0.000168 | 0.001398 | 0.072965 |

| Condition / kind | Min | Mean | Median | P05 | P95 | Max |
|---|---:|---:|---:|---:|---:|---:|
| Current / forged | 0.000002 | 0.046798 | 0.003276 | 0.000032 | 0.231152 | 0.895451 |
| Current / real | 0.000002 | 0.050603 | 0.002910 | 0.000032 | 0.274621 | 0.895451 |
| Checkpoint-era / forged | 0.000000055 | 0.004183 | 0.000283 | 0.000004 | 0.018369 | 0.278589 |
| Checkpoint-era / real | 0.000000055 | 0.003685 | 0.000242 | 0.000004 | 0.018522 | 0.274917 |

## 7. Domain breakdown

Each row contains AUROC, AP, TPR@5% FPR, released-threshold metrics,
confusion, and matched-pair statistics. Brackets are 95% CIs.

| Condition / domain | Pairs | AUROC | AP | TPR@5% FPR | Acc / bal. acc. @0.5 |
|---|---:|---:|---:|---:|---:|
| Current / lodging | 147 | 0.497663 [0.485509, 0.509188] | 0.494241 [0.480941, 0.506240] | 0.040816 [0.020408, 0.047619] | 0.496599 [0.489796, 0.500000] |
| Current / restaurant | 128 | 0.501465 [0.496307, 0.506379] | 0.500864 [0.498709, 0.503583] | 0.046875 [0.023438, 0.046875] | 0.500000 [0.500000, 0.500000] |
| Checkpoint / lodging | 147 | 0.502684 [0.490720, 0.514029] | 0.512654 [0.506597, 0.527628] | 0.061224 [0.034014, 0.081633] | 0.500000 [0.500000, 0.500000] |
| Checkpoint / restaurant | 128 | 0.505219 [0.501860, 0.510163] | 0.515010 [0.505960, 0.531370] | 0.046875 [0.023438, 0.070312] | 0.500000 [0.500000, 0.500000] |

| Condition / domain | Precision / recall / F1 / specificity @0.5 | TP/FP/FN/TN |
|---|---|---:|
| Current / lodging | 0.466667 / 0.047619 / 0.086420 / 0.945578 | 7/8/140/139 |
| Current / restaurant | 0.500000 / 0.023438 / 0.044776 / 0.976562 | 3/3/125/125 |
| Checkpoint / lodging | 0 / 0 / 0 / 1 | 0/0/147/147 |
| Checkpoint / restaurant | 0 / 0 / 0 / 1 | 0/0/128/128 |

| Condition / domain | Pair wins/losses/ties; p | Strict paired ranking (95% CI) | Mean delta (95% CI) |
|---|---|---:|---:|
| Current / lodging | 11/10/126; 1.000000 | 0.074830 [0.034014, 0.115646] | -0.007134 [-0.019762, 0.000900] |
| Current / restaurant | 6/2/120; 0.289062 | 0.046875 [0.015625, 0.085938] | 0.000017 [-0.000026, 0.000062] |
| Checkpoint / lodging | 69/42/36; 0.013234 | 0.469388 [0.387755, 0.557823] | 0.000411 [0.000004, 0.000924] |
| Checkpoint / restaurant | 53/32/43; 0.029456 | 0.414062 [0.328125, 0.492188] | 0.000598 [-0.000089, 0.001828] |

## 8. Visibility breakdown and crop-equality census

`edit_visibility` is a **pre-canonical location condition**. Mouse exact-diff
GT is computed before the real and forged images are independently
canonicalized as JPEG quality 95. It therefore cannot by itself prove that
the final canonical crops, tensors, or scores are equal.

| Condition / visibility | Pairs | AUROC (95% CI) | AP (95% CI) | TPR@5% FPR | Acc @0.5 | TP/FP/FN/TN |
|---|---:|---:|---:|---:|---:|---:|
| Current / full | 14 | 0.500000 [0.382526, 0.622577] | 0.568636 [0.489255, 0.707622] | 0.071429 | 0.500000 | 0/0/14/14 |
| Current / none | 247 | 0.500008 [0.500000, 0.500074] | 0.500010 [0.500000, 0.500097] | 0.048583 | 0.500000 | 10/10/237/237 |
| Current / partial | 14 | 0.474490 [0.336480, 0.602041] | 0.478635 [0.432419, 0.610279] | 0.000000 | 0.464286 | 0/1/14/13 |
| Checkpoint / full | 162 | 0.503734 [0.492719, 0.514327] | 0.516647 [0.508776, 0.533203] | 0.055556 | 0.500000 | 0/0/162/162 |
| Checkpoint / none | 80 | 0.499844 [0.498594, 0.500000] | 0.499917 [0.499440, 0.500000] | 0.037500 | 0.500000 | 0/0/80/80 |
| Checkpoint / partial | 33 | 0.508724 [0.487603, 0.539027] | 0.536744 [0.521995, 0.590089] | 0.030303 | 0.500000 | 0/0/33/33 |

| Condition / visibility | Wins/losses/ties; p | Strict paired ranking (95% CI) | Mean delta (95% CI) |
|---|---|---:|---:|
| Current / full | 10/4/0; 0.179565 | 0.714286 [0.428571, 0.928571] | 0.001620 [-0.001388, 0.006078] |
| Current / none | 1/0/246; 1.000000 | 0.004049 [0, 0.012146] | 0.000000070 [0, 0.000000209] |
| Current / partial | 6/8/0; 0.790527 | 0.428571 [0.142857, 0.714286] | -0.076367 [-0.192936, 0.006829] |
| Checkpoint / full | 100/62/0; 0.003519 | 0.617284 [0.537037, 0.691358] | 0.000772 [0.000051, 0.001861] |
| Checkpoint / none | 0/1/79; 1.000000 | 0 [0, 0] | -0.000000001 [-0.000000004, 0] |
| Checkpoint / partial | 22/11/0; 0.080143 | 0.666667 [0.484848, 0.818182] | 0.000359 [-0.000632, 0.001433] |

The independent census decodes and preprocesses each canonical JPEG
separately, then applies exact `np.array_equal` to RGB uint8 crops:

| Condition / visibility | Pairs | Crop equal | Crop different |
|---|---:|---:|---:|
| **Current total** | **275** | **246** | **29** |
| Current / full | 14 | 0 | 14 |
| Current / none | 247 | 246 | 1 |
| Current / partial | 14 | 0 | 14 |
| **Checkpoint total** | **275** | **79** | **196** |
| Checkpoint / full | 162 | 0 | 162 |
| Checkpoint / none | 80 | 79 | 1 |
| Checkpoint / partial | 33 | 0 | 33 |

The sole current-HEAD `none` exception is
`lodging_205_slot_001`: independent JPEG-q95 canonicalization diffuses a
one-level RGB difference into the crop (7 pixels, 21 channel values,
maximum absolute difference 1), producing a score delta of
`+1.7176847904920578e-05`. This is why the correct statement is
**246/247 current `none` pairs are crop-equal**, not 247/247.

For completeness, the checkpoint-era `none` exception is
`lodging_191_slot_001` (6 pixels, 8 channel values, maximum difference 1;
score delta `-1.1714291758835316e-07`). Every crop-equal pair was verified
exact through normalized tensor, persisted and independently re-encoded CLIP
feature, logit, sigmoid score, and decision.

The pre-canonical visible-fraction summaries are:

| Condition | Min | Mean | Median | P05 | P95 | Max |
|---|---:|---:|---:|---:|---:|---:|
| Current HEAD | 0 | 0.072016 | 0 | 0 | 0.979471 | 1 |
| Checkpoint-era | 0 | 0.648889 | 1 | 0 | 1 | 1 |

## 9. Source, backbone, weights, and preprocessing drift

| Asset | Bytes | SHA-256 |
|---|---:|---|
| Bundled `pretrained_weights/fc_weights.pth` | 4,083 | `477100745713bcc957beb2b40859536859b6483fd6301b3b9293151b194c7847` |
| Official OpenAI `ViT-L-14.pt` | 932,768,134 | `b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836` |
| Registered two-asset bundle | — | `b57c7a8865336b82fe716dc7871006883417cf60f7b36ca8a9a5f925da009121` |
| Canonical inputs JSONL | — | `e4cb3d6a78fa68f06341457e2234c630a455a9b6b9789e59abf45c15b292060a` |

The CLIP file comes from OpenAI's
[official ViT-L/14 URL](https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt).
The loader used the official UFD `models.get_model('CLIP:ViT-L/14')`
construction, patched its downloader to the exact local asset, blocked
network access, and observed zero `urlopen` calls. The head was preflighted
with no unsafe globals and loaded using `weights_only=True`, strict keys,
with no missing or unexpected parameters.

Key pinned source hashes are:

| Source file | SHA-256 |
|---|---|
| `validate.py` | `9ab4021cc6f85002a8b8cd0fc28baa9b4861b59bffcfbd98e10d02b08c42b2d6` |
| `models/clip_models.py` | `57ce5898bf0bc7ff52b5922a0aefae8bc34a9237e4d59e4a1615ff5d8c6ff7a6` |
| `models/clip/clip.py` | `c3f1c09abe0a0d9c429e0d47f2b8d4a4ec09dbd11eec3d7dd9846babd717e43b` |
| `models/clip/model.py` | `c071d011e92226f1ca0a6f7c5098d8d7d08eadc7d6db125a81b52fc234b1ec59` |

There is a genuine official-source ambiguity. The linear head was introduced
in commit
[`763391e`](https://github.com/WisconsinAIVision/UniversalFakeDetect/commit/763391eff3284f6950ffb323599c1a7a819f2ecd)
while `validate.py` still applied `Resize(256)` before `CenterCrop(224)`.
Commit
[`3bf7228`](https://github.com/WisconsinAIVision/UniversalFakeDetect/commit/3bf72282088e47be7e784e104e577790a55d4e48)
later removed that resize; the release never states which transform should
remain attached to the old head. The two paths coincide geometrically on the
paper's 256x256 images but differ sharply on Mouse. Therefore current HEAD is
the reproducible primary, and checkpoint-era is separately named sensitivity.

## 10. Independent replay, runtime, and tests

Both full analyses have status `audited`. For each condition, the analyzer:

- re-decoded and fully re-encoded all 550 images;
- validated all 550 persisted 768-dimensional float32 features;
- independently replayed the linear head, float32 sigmoid, and strict rule;
- obtained maximum feature, raw-logit, and probability differences of
  exactly `0.0`;
- found 550 physical rows, 550 unique IDs, no duplicates, and no recovered
  error histories; and
- exactly matched an independently executed 10-image smoke prefix.

Runtime was CPython 3.12.3, PyTorch
`2.8.0.dev20250627+cu128`, CUDA 12.8, cuDNN 91002, and NVIDIA L20Z on
`cuda:4`. Deterministic cuDNN was enabled; cuDNN benchmark, matmul TF32, and
cuDNN TF32 were disabled. Silent CPU fallback was rejected.

| Condition | Latency ms min / mean / median / P95 / max | Peak CUDA memory |
|---|---|---:|
| Current HEAD | 9.596 / 11.208 / 9.698 / 21.003 / 197.917 | 1,806,185,472 bytes |
| Checkpoint-era | 9.639 / 11.492 / 9.765 / 20.856 / 207.395 | 1,806,185,472 bytes |

At the frozen implementation checkpoint, the focused suite reported
**55 passed** across:

- `tests/test_ufd_metrics.py`
- `tests/test_run_universalfakedetect.py`
- `tests/test_analyze_universalfakedetect_run.py`

## 11. Artifacts

Primary current-HEAD artifacts:

- `results/opensource/universalfakedetect/ufd_clip_vitl14_current_head_mouse_canonical_v1_full275_20260724.jsonl`
- `results/opensource/universalfakedetect/ufd_clip_vitl14_current_head_mouse_canonical_v1_full275_20260724.summary.json`
- `results/opensource/universalfakedetect/ufd_clip_vitl14_current_head_mouse_canonical_v1_full275_20260724.run_manifest.json`
- `results/opensource/universalfakedetect/ufd_clip_vitl14_current_head_mouse_canonical_v1_full275_20260724.analysis.json`
- `outputs/opensource/universalfakedetect/ufd_clip_vitl14_current_head_mouse_canonical_v1_full275_20260724/clip_features/`

Checkpoint-era sensitivity artifacts use the same suffixes under:

```text
results/opensource/universalfakedetect/
  ufd_clip_vitl14_checkpoint_era_mouse_canonical_v1_full275_20260724.*
outputs/opensource/universalfakedetect/
  ufd_clip_vitl14_checkpoint_era_mouse_canonical_v1_full275_20260724/
```

The current and checkpoint run-manifest fingerprints are respectively:

```text
88d101117d02fd2466da5bac2fb5c35a3444cc2c6fcc2d240350928a768e84c6
42f1c96bc501fbb89da0b64187c51f128b8be1ea69cd6a60af5719e46363766c
```

## 12. Scope, license, and final interpretation

The pinned UFD repository's
[current license is MIT](https://github.com/WisconsinAIVision/UniversalFakeDetect/blob/76a0e3e60a8a06458707a625d269ba815a2e5919/LICENSE),
which permits commercial use of the covered software. OpenAI CLIP code also
has an
[MIT license](https://github.com/openai/CLIP/blob/main/LICENSE).
However:

- `fc_weights.pth` has no separate explicit weight terms or model card;
- the OpenAI CLIP
  [model card](https://github.com/openai/CLIP/blob/main/model-card.md)
  describes deployed use, commercial or otherwise, as out of scope pending
  careful task-specific study;
- UFD vendors OpenAI CLIP source, so redistribution should preserve the
  applicable OpenAI MIT notice; and
- training/evaluation data do not have one unified commercial license.

Accordingly, code being MIT does **not** establish end-to-end commercial
clearance for the checkpoint, backbone, data, or deployment. The project
should record this method as **commercial clearance not established; legal
and safety review required**.

The defensible scientific conclusion is narrow:

> Under the current official UFD HEAD preprocessing, Ours LC does not
> separate CLAIMFORGE's small local mouse insertions from their matched real
> images. Most edits are outside the native center crop, and 246 matched
> canonical crops are exactly identical. The checkpoint-era resize exposes
> many more edits but still gives chance-level image ranking and zero forged
> recall at the released threshold.

This does not show that UFD is generally weak at detecting fully synthetic
images. The paper primarily studies images generated from scratch, not small
local insertions. The next required whole-image experiment is therefore the
frozen same-domain fully synthetic control; until it is complete, Track B's
two-condition interpretation remains unfinished.
