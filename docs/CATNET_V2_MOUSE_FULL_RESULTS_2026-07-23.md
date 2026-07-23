# CAT-Net v2 on the canonical mouse set (2026-07-23)

## 1. Status and headline

The third publicly released research baseline is complete. All 275 paired tasks,
or 550 images, finished successfully with no errors.

CAT-Net v2 is a strong localization-only result on CLAIMFORGE:

- native-resolution macro pixel AP is **0.61234**;
- macro pixel F1/IoU at the fixed 0.5 threshold are **0.47481/0.35743**;
- micro pixel F1/IoU at 0.5 are **0.30706/0.18138**; and
- the fixed mask overlaps the recorded edit box at IoU greater than 0.3 in
  **65/275** forged images.

CAT-Net has no image-level integrity head. This run therefore reports T2 pixel
localization only. It does not turn the maximum, mean, or area of the
localization map into an improvised T1 score.

The main result is nuanced. CAT-Net has the highest continuous-map macro AP of
the three completed methods, slightly above TruFor's 0.57989, but TruFor remains
better at the released 0.5 operating point and provides a useful native T1 head.
CAT-Net's strength is pixel ranking; its main weakness is threshold calibration
and occasional large false-positive regions.

## 2. Pinned method and checkpoint

The run uses the original author's
[CAT-Net repository](https://github.com/mjkwon2021/CAT-Net) at commit
`b50d391ffc423d3631fd7947714468788c791805`, not the separately retrained
IMDL-BenCo checkpoint.

The exact released asset is:

- official `CAT_full_v2.pth.tar` from the author's
  [Google Drive folder](https://drive.google.com/drive/folders/1hBEfnFtGG6q_srBHVEmbF3fTq0IhP8jq);
- 915,503,873 bytes;
- SHA-256
  `f82aaafdd1142775231feedcea0bb7027f7370561d9e8d107465454001865989`;
- epoch 196 with 2,926 state-dict entries; and
- 114,263,810 model parameters.

All checkpoint keys loaded with `strict=True`, with zero missing and zero
unexpected keys. Loading used `torch.load(weights_only=True)` and a minimal
three-type NumPy scalar/dtype allowlist. Unrestricted pickle loading was never
used. The final checkpoint completely covers the network, so the separate
HRNet and DCT-stream initialization weights are not required for inference.

## 3. Exact JPEG/DCT inference contract

CAT-Net's defining signal is the combination of a spatial RGB stream and a
JPEG-artifact stream. The adapter follows the original validation protocol:

1. It opens the materialized canonical JPEG directly. It does not resize, crop,
   tile, or re-encode it.
2. PIL supplies RGB uint8 pixels, normalized as
   `(RGB - 127.5) / 127.5`.
3. `jpegio` reads the luminance-channel quantized DCT coefficients and the
   original luminance quantization table from the same JPEG bytes.
4. The DCT coefficients become a 21-channel volume: coefficient 0, absolute
   values 1 through 19, and absolute value at least 20.
5. Width and height are padded only on the right and bottom to multiples of
   eight. RGB padding is 127.5 before normalization and DCT padding is zero.
6. The network receives `[1,24,H8,W8]`: RGB 3 channels plus DCT volume 21
   channels. The quantization table is `[1,1,8,8]`.
7. The raw output is two-channel logits at one-quarter resolution. Following
   the official validation path, the adapter first bilinearly upsamples both
   logits to padded native size, then applies softmax, selects channel 1
   (`tampered`), and crops away padding.

The order in the last step matters: `interpolate(logits) -> softmax` is not
equivalent to `softmax(logits) -> interpolate`.

The installed IMDL-BenCo 0.1.45 CAT wrapper was intentionally not used for this
run. Its generic path resizes the RGB tensor and generates a temporary JPEG
before reading DCT, which changes the very coefficients and quantization table
that CAT-Net is meant to inspect. The original checkpoint and original-JPEG
protocol are the scientifically cleaner baseline. A BenCo-retrained CAT model,
if run later, must be named and reported as a separate variant.

## 4. Why this method is effective

The [CAT-Net v2 paper](https://link.springer.com/article/10.1007/s11263-022-01617-5)
combines complementary evidence:

- an HRNet-style RGB stream preserves fine boundaries and multi-scale spatial
  discontinuities;
- a DCT stream directly observes quantized JPEG coefficient patterns;
- the JPEG quantization table conditions those DCT features on compression
  quality; and
- multi-resolution fusion combines small boundary evidence with larger
  manipulated regions.

That inductive bias is unusually well matched to CLAIMFORGE: a generated mouse
is composited into a real photograph and the result is materialized as JPEG.
Unlike a whole-image generator classifier, CAT-Net looks for a locally
inconsistent forensic process. This explains why it can strongly rank the mouse
pixels even though the altered area is typically around one tenth of one
percent of the image.

The same design also explains its failure cases. Natural JPEG structure,
texture, and recompression artifacts can trigger the DCT stream on pristine
regions; uniform Q95 re-encoding can suppress some earlier compression
differences; and a well-blended generative insertion need not create a sharp
classical splice boundary.

## 5. CLAIMFORGE evaluation contract

The authoritative set contains the 275 `status=good`, `candidates=mouse`
records:

| Domain | Pairs |
|---|---:|
| lodging | 147 |
| restaurant | 128 |
| total | 275 |

Real and forged images are both materialized as metadata-free JPEG at quality
95 with 4:4:4 sampling. The forged GT is the exact nonzero RGB difference
between the decoded source and generated forged PNG before canonical JPEG
encoding; the generation anchor is not substituted for the true mask.

The edits are extremely small:

| Edit fraction of the full image | Value |
|---|---:|
| minimum | 0.02486% |
| median | 0.11264% |
| mean | 0.16846% |
| maximum | 1.29272% |

Pixel AP uses the continuous native probability map. Primary thresholded
metrics use direct class-1 semantics and the fixed threshold 0.5, with no
prediction inversion. Confidence intervals use 1,000 percentile-bootstrap
replicates with the paired task as the resampling unit.

## 6. Primary T2 result

| Native-resolution metric | Estimate | Pair-bootstrap 95% CI |
|---|---:|---:|
| Macro pixel AP | **0.61234** | [0.58458, 0.64259] |
| Median per-image pixel AP | 0.65894 | — |
| Macro pixel F1 at 0.5 | 0.47481 | [0.43978, 0.50765] |
| Macro pixel IoU at 0.5 | 0.35743 | [0.32671, 0.38694] |
| Micro pixel F1 at 0.5 | 0.30706 | [0.25523, 0.36611] |
| Micro pixel IoU at 0.5 | 0.18138 | [0.14628, 0.22407] |
| Edit-box hits at 0.5 | 65/275 (23.64%) | — |

At threshold 0.5, 10/275 forged images have an empty predicted mask. On the
remaining images the behavior ranges from near-perfect localization to large
scene-level false positives. That heavy-tailed error pattern explains the gap
between macro F1 0.4748 and micro F1 0.3071.

The released threshold is not micro-optimal for this benchmark:

- one post-hoc global threshold, approximately 0.86918, reaches micro F1
  0.49433 and micro IoU 0.32831;
- allowing a different oracle threshold for every forged image gives
  mean/median F1 0.63781/0.69426.

These are deliberately optimistic diagnostics, not deployable primary results.
The fixed 0.5 metrics remain authoritative.

## 7. Pristine-image behavior

At threshold 0.5, 201/275 real images contain at least one positive pixel and
74/275 have an empty mask.

| Real-image statistic | Value |
|---|---:|
| mean predicted-positive fraction | 0.43587% |
| median predicted-positive fraction | 0.04506% |
| maximum predicted-positive fraction | 10.94462% |
| micro predicted-positive fraction | 0.40773% |

The median false-positive area is small, but a few large errors dominate the
pixel totals. Since CAT-Net has no image-level head to gate these masks, users
cannot assume that a nonempty localization map is itself a calibrated forged
verdict.

## 8. Diagnostic slices

### Domain

| Domain | Pairs | Macro pixel AP | Macro F1@0.5 | Micro IoU@0.5 | Real positive fraction, macro |
|---|---:|---:|---:|---:|---:|
| lodging | 147 | 0.61173 | 0.44065 | 0.13237 | 0.64364% |
| restaurant | 128 | 0.61303 | 0.51404 | 0.28681 | 0.19725% |

Continuous ranking is essentially identical across domains. Fixed-threshold
performance is better on restaurant scenes because lodging has more and larger
false-positive regions.

### Edit-size quintiles

| Quintile | Pairs | Median edit fraction | Macro pixel AP | Macro F1@0.5 | Micro IoU@0.5 |
|---|---:|---:|---:|---:|---:|
| Q1, smallest | 55 | 0.05347% | 0.54550 | 0.41763 | 0.10594 |
| Q2 | 55 | 0.07780% | 0.55683 | 0.38624 | 0.09557 |
| Q3 | 55 | 0.11264% | 0.58055 | 0.44043 | 0.14606 |
| Q4 | 55 | 0.16293% | 0.61595 | 0.48515 | 0.18638 |
| Q5, largest | 55 | 0.37441% | 0.76286 | 0.64462 | 0.40737 |

Pixel AP increases monotonically with edit size. Even the smallest quintile has
substantial continuous ranking signal, but the largest edits are far easier at
the fixed threshold.

## 9. Direct comparison with completed research methods

All three methods use the same 275 paired canonical inputs, native exact-diff
GT, fixed map threshold 0.5, and paired bootstrap design.

| Metric | MaskCLIP | TruFor | CAT-Net v2 |
|---|---:|---:|---:|
| Native T1 available | yes | yes | **no** |
| Macro pixel AP | 0.04740 | 0.57989 | **0.61234** |
| Median pixel AP | 0.00463 | **0.66459** | 0.65894 |
| Macro pixel F1@0.5 | 0.00564 | **0.50010** | 0.47481 |
| Macro pixel IoU@0.5 | 0.00367 | **0.37410** | 0.35743 |
| Micro pixel IoU@0.5 | 0.00312 | **0.19800** | 0.18138 |
| Edit-box hits@0.5 | 1/275 | **83/275** | 65/275 |
| Mean real positive fraction@0.5 | — | 1.776% | **0.436%** |

CAT-Net and TruFor are close, but their strengths differ:

- CAT-Net has higher mean continuous pixel AP and substantially less
  real-image positive area.
- TruFor is slightly better at the fixed localization threshold, produces more
  edit-box hits, and supplies a native image-level score and reliability map.
- Their macro AP confidence intervals overlap, so this run does not establish a
  clean statistical winner.
- MaskCLIP remains far behind both classical forensic localizers on this
  zero-shot mouse benchmark.

CAT-Net is therefore a core T2 baseline, while TruFor remains the most complete
T1+T2 research reference.

## 10. Validation and audit

The result is backed by the following checks:

- The official checkpoint safely loaded all 2,926 keys at epoch 196 with
  `strict=True`.
- A fixed five-pair smoke run completed and passed audit before the full run.
- The 10 overlapping smoke/full images are bit-exact in raw logits, native
  probability maps, threshold masks, DCT hashes, quantization-table hashes,
  preprocessing metadata, and all localization metrics: 100/100 compared
  fields match.
- The full analyzer re-hashed 2,475 files: every canonical JPEG, raw logits
  array, native score map, binary mask, and forged GT.
- It independently re-read every JPEG with `jpegio` and reproduced the
  luminance DCT and quantization-table hashes.
- It checked finite float32 logits with shape `[2,H8/4,W8/4]`, finite float32
  native maps bounded by `[0,1]`, and bit-exact equality between every saved
  mask and `score_map >= 0.5`.
- It recomputed the immutable manifest fingerprint, validated all 550 result
  identities and hashes, and rejected any T1 score, decision, detection,
  classification-threshold, or AUROC field.
- The append-only JSONL has 550 physical rows and 550 unique latest rows, all
  successful, with no duplicate or recovered history.
- The full repository test suite passes: 62/62 tests.

## 11. License boundary

The CAT-Net repository exposes source and weights publicly, but it does not
provide a project-wide license for the author's CAT-Net code or checkpoint.
The bundled `LICENSE of HRNet` covers the inherited HRNet component; it must not
be presented as an MIT license for the entire CAT-Net release.

This report therefore calls CAT-Net a publicly released or source-available
academic method, not an OSI-licensed open-source package. Redistribution or
commercial reuse of the CAT-Net-specific code or weights requires separate
rights review.

## 12. Reproducible artifacts

| Artifact | Path | SHA-256 |
|---|---|---|
| canonical manifest | `outputs/opensource/mouse_canonical_v1/manifest.json` | `beb3c30e436db682bbadef794404838f33a4812f18f22819dd6ab1ef3de6f0b1` |
| full results | `results/opensource/catnet/catnet_v2_mouse_canonical_v1_full275_20260723.jsonl` | `f683685b5ffe2169348483a627ec80ae03c231da2023bc7738a4cec2ae25cb7e` |
| run manifest | same basename, `.run_manifest.json` | `55ef32ad585a766a40f196e650cf33bdfeed82644e6b3a200747dff9bfb0ebd5` |
| runner summary | same basename, `.summary.json` | `ac21e7692b43ca14ae3832c6d75745e756a4945c70169e6de839ee26736b1ae0` |
| post-hoc analysis | same basename, `.analysis.json` | `b907842f4ac97d8a487c6114ce9feb363a4e9d476cf881c6c46d089c25a4ffac` |

The 1,650 raw-logit, native-map, and mask artifacts occupy 3.98 GB under
`outputs/opensource/catnet/catnet_v2_mouse_canonical_v1_full275_20260723/`.
The immutable run-manifest fingerprint is
`df5bb1fba62f741c5da29ec72939eb5ded1e52aa327d9ca7159ca2dcb5ea2fe5`.

Environment: Python 3.12.3, PyTorch `2.8.0.dev20250627+cu128`, CUDA 12.8,
NumPy 1.26.4, Pillow 11.1.0, `jpegio` 0.2.8, `yacs` 0.1.8, and one NVIDIA
L20Z. Median measured model-forward latency is 59.03 ms/image at batch size 1;
maximum peak allocated CUDA memory is 2.32 GB. Decode, DCT construction,
metrics, hashing, and artifact writes are excluded from latency.

Commands:

```bash
/root/.cache/claimforge/venvs/catnet-b50d391/bin/python \
  -m eval.opensource.run_catnet \
  --run-id catnet_v2_mouse_canonical_v1_full275_20260723 \
  --device cuda:0 --fail-fast

/root/.cache/claimforge/venvs/catnet-b50d391/bin/python \
  -m eval.opensource.analyze_catnet_run \
  --run-id catnet_v2_mouse_canonical_v1_full275_20260723 \
  --bootstrap-iterations 1000 --bootstrap-seed 20260723
```

## 13. Decision and next method

CAT-Net v2 should remain in the paper's core source-available research table as
a strong JPEG-aware T2 baseline. Its high continuous-map AP demonstrates that
these sub-percent edits are not forensically invisible, but the absence of a
native T1 head and the gap between macro and micro fixed-threshold performance
leave substantial operational headroom.

The next model-zoo run should be **MVSS-Net**. It supplies image and pixel
outputs and uses edge/noise-sensitive multi-view supervision, giving a clean
contrast with CAT-Net's DCT stream. Its original repository has no project-wide
license and dated dependencies, so the exact official-versus-IMDL-BenCo
checkpoint and preprocessing contract must be resolved before execution.
