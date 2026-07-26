# NPR AIGC ProGAN-4class on the canonical mouse set (2026-07-25)

## 1. Status and headline

NPR has completed the frozen local-splice condition on all 275 matched mouse
tasks, or 550 canonical JPEG images. The run contains 550/550 valid images,
275/275 complete pairs, zero errors, zero missing rows, and no duplicate
physical result rows. An independent analyzer subsequently decoded and
preprocessed every image again, executed 550 complete model forward passes,
and reproduced every stored 512-dimensional feature, raw logit, float32
sigmoid probability, and decision with maximum absolute error `0.0`.

Under the released operational score and threshold, NPR fails completely on
this condition:

- the strict rule `sigmoid(logit) > 0.5` detects **0/275 forged images** and
  falsely flags **0/275 real images**;
- AUROC is **0.500198**, with task-paired bootstrap 95% CI
  **[0.499649, 0.500873]**;
- average precision is **0.502661**, CI **[0.502412, 0.508930]**;
- TPR at a real-only 5% FPR operating point is **4.7273%**, CI
  **[4.0000%, 5.8182%]**; and
- the confusion matrix at 0.5 is **TP=0, FP=0, FN=275, TN=275**.

The pre-registered raw-logit numerical diagnostic does not rescue
image-level detection. Its AUROC is **0.500906**, AP is **0.503328**, and all
550 logits remain negative, so the equivalent zero-logit decision also
predicts every image as real.

There is nevertheless a small matched-pair signal before sigmoid saturation:
the forged logit is higher than the corresponding real logit in **167/275**
pairs (60.73%), with an exact two-sided sign-test `p=0.000449`. This is not a
contradiction. The within-scene shift is only `+0.0455` logit on average,
whereas logits vary by roughly 179 points across different scenes. A detector
that receives one image at a time cannot use the unavailable matched-real
reference, so the global AUROC remains essentially 0.5 and the released
operating point remains unusable.

NPR is a whole-image binary classifier and emits no native dense map.
Therefore **T2 localization and the joint score are N/A**. This completes
only the local-splice half of NPR's evaluation. The same-domain fully
synthetic control is still required before drawing conclusions about its
intended whole-image AIGC task.

## 2. Pinned paper, source, checkpoint, and license

The method is described in the
[CVPR 2024 paper](https://openaccess.thecvf.com/content/CVPR2024/html/Tan_Rethinking_the_Up-Sampling_Operations_in_CNN-based_Generative_Network_for_Generalizable_CVPR_2024_paper.html),
“Rethinking the Up-Sampling Operations in CNN-based Generative Network for
Generalizable Deepfake Detection.” The paper's
[HTML version](https://arxiv.org/html/2312.10461) explains the neighboring
pixel relationship and evaluates generalization across 28 generators.

The run pins the authors'
[official repository](https://github.com/chuangchuangtan/NPR-DeepfakeDetection)
at commit
[`781ced3f7ca2cdc69ec9dd4ef27e8d0b3c07752a`](https://github.com/chuangchuangtan/NPR-DeepfakeDetection/tree/781ced3f7ca2cdc69ec9dd4ef27e8d0b3c07752a).
The main checkpoint was frozen before viewing Mouse scores:

| Asset | Bytes | SHA-256 | Role |
|---|---:|---|---|
| `model_epoch_last_3090.pth` | 5,842,385 | `b67a91555ce786a6d0463ff0cb2b0b874d1c3f971b0e3febd2ae5618a80f7e8a` | Official AIGCDetectBenchmark ProGAN-4class checkpoint |

It is a flat 146-entry `OrderedDict` containing 1,447,897 state elements.
The model has 1,437,761 trainable parameters. It is loaded with
`weights_only=True`, `map_location="cpu"`, an empty unsafe-global list, a
strict state-dict match, and a frozen schema hash.

The repository contains two other plausible assets, neither selected using
Mouse performance:

| Excluded asset | SHA-256 | Reason |
|---|---|---|
| `NPR.pth` | `3939297e9399e0b992f87211610769d87d899de50d56da0204d6cbda2d483a53` | Older nested model/optimizer/step snapshot with `module.` keys; incompatible with current `test.py` and not the AIGCDetectBenchmark link |
| GenImage SDv1.4 checkpoint | `9bc961e7d643581aa0ea879cbd322dcc2e543877568a43d2f6cdb92906379015` | Separately trained GenImage/SDv1.4 model |

No `LICENSE`, `COPYING`, or `NOTICE` file exists at the pinned GitHub
repository root. The code and checkpoint are publicly downloadable, but an
OSI license and commercial-use clearance are **not established**. The
Hugging Face Space's Apache-2.0 metadata applies to that Space repository; it
does not relicense the upstream GitHub checkpoint.

## 3. What NPR does and why the idea is strong

Many GAN and diffusion decoders repeatedly enlarge low-resolution feature
maps. Nearest, bilinear, and learned upsampling create local dependence
between pixels that originated from the same low-resolution cell. NPR tries
to expose this generator-side structural trace instead of recognizing image
semantics or one generator's visual style.

For the released implementation, the frozen computation is:

```text
Pillow RGB
-> ToTensor
-> ImageNet channel normalization
-> remove final bottom row/right column only when a dimension is odd
-> x_half = nearest_downsample(x, scale=0.5)
-> x_reconstructed = nearest_upsample(x_half, scale=2)
-> NPR = x - x_reconstructed
-> NPR * (2/3)
-> truncated ResNet-50 stem + layer1 + layer2
-> adaptive global average pool
-> 512-dimensional feature
-> one linear logit
-> float32 sigmoid
```

Within each effective 2x2 neighborhood, the down-up reconstruction keeps a
representative value and the subtraction exposes the relative differences of
the other pixels. Because these relationships are local and arise from a
common generator operation, the paper argues that they transfer better than
generator-specific spectral fingerprints. The paper reports strong
cross-GAN and cross-diffusion performance, while the official repository's
AIGCDetectBenchmark table reports 91.7% average accuracy for NPR under its
ProGAN-4class checkpoint/profile. Those numbers are context only and are not
reproduced here.

The same design explains both the small paired signal and the operational
failure on CLAIMFORGE. NPR does respond slightly to the inserted diffusion
pixels when the untouched scene is held fixed. But the final global average
pool mixes that small response with the much larger unchanged photograph,
and scene-dependent camera/JPEG texture dominates comparisons between
unrelated images. The training distribution also contrasts fully synthetic
ProGAN images against LSUN real images, whereas CLAIMFORGE changes only one
small object inside a real camera image.

## 4. Frozen Mouse protocol

The protocol was frozen before full-set scores were inspected:

| Component | Frozen value |
|---|---|
| Dataset | `claimforge-mouse-good275-canonical-jpeg-q95-v1` |
| Coverage | 275 matched tasks, 550 images |
| Source | GitHub commit `781ced3f7ca2cdc69ec9dd4ef27e8d0b3c07752a` |
| Checkpoint | `model_epoch_last_3090.pth`, SHA-256 `b67a9155...80f7e8a` |
| Preprocessing | RGB, native resolution, no resize, no crop, batch size 1 |
| Odd dimensions | Drop only the final bottom row and/or right column before NPR |
| Model mode | `eval()`, following official GitHub `test.py` |
| Primary score | float32 `sigmoid(raw_logit)` |
| Released decision | strict `probability > 0.5` |
| FPR operating point | real-only 95th percentile, `method="higher"`, strict `>` |
| Bootstrap | 1,000 complete-task-pair resamples, seed `20260724` |
| T2 | N/A; no native dense output |

All 275 exact-difference GT edits remain fully visible after the at-most-one
row/column trim. Across the 550 images, 416 need no trim, 110 lose one bottom
row, 20 lose one right column, and 4 lose both. The trim is therefore a
parity requirement for the 2x down-up residual, not a crop that removes the
inserted object.

The official GitHub evaluator calls `model.eval()`. The authors'
[Hugging Face demo](https://huggingface.co/spaces/tancc/Generalizable_Deepfake_Detection-NPR-CVPR2024)
corroborates the checkpoint and native-size preprocessing, but its pinned
`app.py` omits `model.eval()`. BatchNorm consequently remains in train mode
with batch-one statistics, mutates running buffers, and can reverse a
prediction relative to evaluation mode. The Space is recorded as a
deployment defect and provenance clue, not used as an executable reference
or an extra sensitivity chosen after seeing Mouse results.

## 5. Primary released-probability result

Brackets are 1,000-resample task-paired bootstrap 95% percentile confidence
intervals.

| Metric | Estimate | 95% CI |
|---|---:|---:|
| AUROC | 0.500198 | [0.499649, 0.500873] |
| Average precision | 0.502661 | [0.502412, 0.508930] |
| TPR @ target FPR 5% | 0.047273 | [0.040000, 0.058182] |
| Real-only 5% FPR threshold | `5.696e-19` | [`2.589e-23`, `2.760e-15`] |
| Actual FPR | 0.047273 | [0.036364, 0.047273] |
| Accuracy @ 0.5 | 0.500000 | [0.500000, 0.500000] |
| Balanced accuracy @ 0.5 | 0.500000 | [0.500000, 0.500000] |
| Precision @ 0.5 | 0.000000 | [0.000000, 0.000000] |
| Recall @ 0.5 | 0.000000 | [0.000000, 0.000000] |
| F1 @ 0.5 | 0.000000 | [0.000000, 0.000000] |
| Specificity @ 0.5 | 1.000000 | [1.000000, 1.000000] |
| Strict paired ranking | 0.218182 | [0.170909, 0.269091] |
| Mean forged-real probability delta | `-2.267e-6` | [`-9.067e-6`, `6.923e-10`] |

| Threshold | TP | FP | FN | TN |
|---|---:|---:|---:|---:|
| Released `probability > 0.5` | 0 | 0 | 275 | 275 |

The official probability is exactly zero for 322/550 images. There are only
227 distinct probability values, and 161 matched pairs have equal
probabilities despite different finite logits. Hence the probability paired
result contains 60 wins, 54 losses, and 161 ties; the exact sign-test over
114 non-ties gives `p=0.639769`.

| Kind | Min | Mean | Median | P95 | Max |
|---|---:|---:|---:|---:|---:|
| Forged probability | 0 | `3.5292e-5` | 0 | `1.4505e-19` | 0.009703 |
| Real probability | 0 | `3.7558e-5` | 0 | `1.8698e-19` | 0.010327 |

## 6. Pre-registered raw-logit numerical diagnostic

The CUDA smoke, performed before the 275-pair run, exposed finite logits near
`-170` whose float32 sigmoid rounded to exact zero. The protocol therefore
pre-registered an always-reported raw-logit view. It preserves finite ranking
information but never replaces the official probability, threshold, or
decision.

| Diagnostic metric | Estimate | 95% CI |
|---|---:|---:|
| Raw-logit AUROC | 0.500906 | [0.500364, 0.502202] |
| Raw-logit AP | 0.503328 | [0.503551, 0.510191] |
| Raw-logit TPR @ target FPR 5% | 0.047273 | [0.040000, 0.058182] |
| Raw real-only 5% FPR threshold | -42.009357 | [-52.008110, -33.523674] |
| Strict paired raw ranking | 0.607273 | [0.549091, 0.665455] |
| Mean forged-real raw delta | 0.045507 | [0.014189, 0.079763] |

These are percentile-bootstrap intervals rather than centered or basic
intervals. The slightly biased raw-AP bootstrap distribution therefore yields
a lower endpoint marginally above the full-sample point estimate; the stored
summary and independent recomputation agree exactly on both values.

The raw paired result is 167 wins, 108 losses, and zero ties. Its exact
two-sided sign-test is `p=0.000448729`. The average logit is `-100.3385` for
forged images and `-100.3840` for real images, while the overall ranges are:

| Kind | Min | Mean | Median | P05 | P95 | Max |
|---|---:|---:|---:|---:|---:|---:|
| Forged raw logit | -183.5835 | -100.3385 | -97.5710 | -165.0530 | -44.3134 | -4.6255 |
| Real raw logit | -183.7282 | -100.3840 | -97.6924 | -165.1072 | -44.2561 | -4.5626 |

The useful interpretation is narrow: NPR notices a weak local change when
the same scene supplies its own control, especially in restaurant images.
It does **not** separate an arbitrary forged image from arbitrary real
images, and every sample remains far below the released fake threshold.
Reporting the 60.7% paired ranking without the 0.5009 global AUROC and 0/275
operational recall would therefore be misleading.

## 7. Domain breakdown

| Domain | Pairs | Official AUROC | Official AP | Raw AUROC | Raw AP |
|---|---:|---:|---:|---:|---:|
| Lodging | 147 | 0.500069 | 0.507109 | 0.500578 | 0.508065 |
| Restaurant | 128 | 0.500641 | 0.504001 | 0.502380 | 0.505471 |

| Domain | Official wins/losses/ties | Raw wins/losses/ties | Raw paired accuracy | Raw sign-test p | Mean raw delta |
|---|---:|---:|---:|---:|---:|
| Lodging | 29 / 26 / 92 | 83 / 64 / 0 | 0.564626 | 0.137378 | 0.028332 |
| Restaurant | 31 / 28 / 69 | 84 / 44 / 0 | 0.656250 | 0.000516 | 0.065232 |

Both domains remain chance-level as standalone image classifiers. The larger
restaurant paired shift is a diagnostic of within-scene sensitivity, not a
deployable domain-specific detector.

## 8. Determinism, replay audit, and runtime

Two independent final-code CUDA pair smokes produced exactly equal normalized
tensor hashes, NPR residual hashes, 512-dimensional feature hashes, raw
logits, float32 probabilities, decisions, and runtime/config fingerprints.
A separately generated 1-pair prefix using the full run's 1,000-bootstrap
configuration also matched the first two ordered full-run images exactly.

The final independent audit validates:

- the pinned GitHub commit and hashes of seven official source files;
- the pinned Hugging Face Space commit and hashes of three source files;
- checkpoint bytes, safe loading, 146-key schema, and bundle hash;
- all canonical inputs, GT geometry, adapter files, configuration, runtime,
  manifests, output hashes, and the 550-row physical history;
- independent RGB decoding, normalization, even-dimension trim, NPR residual,
  full network, 512-dimensional feature, logit, sigmoid, aliases, decisions,
  summary, strata, and bootstrap metrics; and
- explicit rejection of T2/localization output for this method.

Audit status is `audited`. Maximum replay differences for features, raw
logits, and probabilities are all `0.0`. The NPR-specific test suite contains
58 passing tests; the only emitted warning is PyTorch's deprecation notice
for the legacy TF32 inspection API.

On one NVIDIA L20Z with batch size one, model-forward latency averages
**19.734 ms/image** (median 16.268 ms, P95 46.512 ms). Peak allocated CUDA
memory averages **490 MB** and reaches **744 MB**. Separately timed image
decoding, normalization, residual construction, geometry/GT evidence, and
hash preparation average **245.816 ms/image**; the mean of per-image
preprocessing plus forward time is therefore **265.550 ms**. Feature
persistence, remaining file I/O, bootstrap computation, and the independent
second replay are still outside that combined number.

## 9. Scope and conclusion

The valid claim is:

> The official NPR AIGCDetectBenchmark ProGAN-4class model does not transfer
> to CLAIMFORGE's small local mouse insertions as a standalone whole-image
> detector: AUROC is 0.5002 and its released threshold detects none of 275
> forgeries. A weak within-pair raw-logit shift survives, but it is unusable
> without a matched real reference and does not improve global ranking.

The result does not show that NPR fails on fully synthetic images, does not
evaluate localization, and does not compare alternative weights selected by
Mouse performance. The required next contrast is a same-domain set of fully
synthetic lodging and restaurant scenes.

## 10. Reproducibility artifacts

Primary run:

```text
results/opensource/npr/
  npr_aigcdetect_progan4class_mouse_canonical_v1_full275_20260725/
```

Important files:

- `manifest.json` — frozen source, checkpoint, preprocessing, runtime,
  dataset, metric, license, and output contracts;
- `expected_inputs.jsonl` — exact selected 550-image input ledger;
- `results.jsonl` — one physical successful row per image;
- `features/*.npy` — 550 persisted 512-dimensional features;
- `summary.json` — official probability metrics plus mandatory raw-logit
  diagnostic; and
- `analysis.json` — independent full-model replay and provenance audit.

The reusable entry points are:

```text
eval/opensource/run_npr.py
eval/opensource/npr_metrics.py
eval/opensource/analyze_npr_run.py
```

New canonical data can be evaluated with a new immutable run ID using the
same runner. Successful rows are resumable, configuration drift is rejected,
and the analyzer can replay the new run independently.
