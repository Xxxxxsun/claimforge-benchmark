# RelayFormer checkpoint 164 on the canonical mouse set (2026-07-24)

## 1. Status and headline

The ninth publicly reproducible local-manipulation baseline is complete.
RelayFormer's official image-only paper checkpoint at epoch 164 ran on all
275 matched tasks, or 550 canonical JPEG images, with no errors.

RelayFormer is the third-strongest of the nine completed local methods by
native macro pixel AP, behind CAT-Net v2 and TruFor and ahead of IML-ViT.
It nevertheless remains highly uneven on CLAIMFORGE's small local insertions:

- native macro pixel AP is **0.227755**, with task-paired bootstrap 95% CI
  **[0.188467, 0.268776]**;
- median per-image pixel AP is only **0.025562**;
- native macro pixel F1 at the official strict `probability > 0.5` rule is
  **0.187240**, CI **[0.155931, 0.219522]**;
- native micro pixel F1 is **0.095885**, CI
  **[0.073259, 0.127001]**;
- 147/275 forged images have exact-GT F1 equal to zero, while 128/275 hit at
  least one exact manipulated pixel;
- 62/275 forged images reach pixel AP at least 0.5, and 53/275 reach F1 at
  least 0.5;
- 129/275 masks overlap the registered edit box at all, but only 34/275
  reach edit-box IoU greater than 0.3; and
- the mean pristine false-positive area is **0.6147%**, CI
  **[0.5260%, 0.7245%]**.

The strongest explanatory variable is edit size. Native macro AP rises
monotonically from **0.052953** in the smallest edit-area quintile to
**0.483606** in the largest. Macro F1 rises from **0.041523** to
**0.383660** over the same range.

The independent analyzer validated 3,575 files, reproduced the frozen
paper-v3 preprocessing, independently replayed the model and native
probabilities from captured logits, regenerated all 550 strict-threshold
masks, and recomputed every metric. The largest probability replay error is
`1.7881393e-7`, and there are zero threshold-mask pixel disagreements.

This checkpoint has no image-classification head. The upstream forward
returns only a localization probability tensor, so T1 is **N/A**. A map
maximum, mean, or nonempty-mask indicator is not promoted to an unofficial
image detector.

## 2. Pinned method and official release

The run uses the authors'
[official RelayFormer repository](https://github.com/WenOOI/RelayFormer) at
commit
[`3fc863c7691d93fb5b11ca8e12e3a214d771e384`](https://github.com/WenOOI/RelayFormer/tree/3fc863c7691d93fb5b11ca8e12e3a214d771e384).
The method is described in the
[ICLR 2026 paper, arXiv v3](https://arxiv.org/abs/2508.09459), revised
2026-06-10.

The repository code is MIT licensed. The official Hugging Face checkpoint
repository is marked Apache-2.0. The selected artifact was registered before
examining full-set performance:

| Field | Value |
|---|---|
| Released filename | `checkpoint-164.pth` |
| Provider | Author-linked [Hugging Face repository `Wenn11/RelayFormer`](https://huggingface.co/Wenn11/RelayFormer) |
| Pinned HF revision | `9ef11f4ac16ac50e2684d4af522e442cb290e2c1` |
| Bytes | 1,102,625,388 |
| SHA-256 | `00a0f145ae4a98e66cad95aa79d2ce470d77821ee4262d6b803b3705c11c2090` |
| Recorded epoch | 164 |
| State tensors | 410 float32 tensors |
| State elements | 91,909,179 |
| Tensor payload bytes | 367,636,716 |

The checkpoint has the exact top-level keys `model`, `optimizer`, `epoch`,
`scaler`, and `args`. Its saved arguments identify the `RelayFormer`
image-only, padding, 1024-pixel configuration and contain two 768-dimensional
relay tokens per patch.

The adapter safely deserializes the complete file with
`weights_only=True` and an explicit `argparse.Namespace` allowlist. It
validates the container, arguments, tensor count, element count, dtype
distribution, epoch, byte size, and relay-token shape before construction.
The official constructor normally calls `torch.load` internally; the adapter
replays that one call from the already validated in-memory payload, preventing
both unsafe deserialization and timm's hidden pretrained-weight download.
It then loads the complete model state with `strict=True` and calls the
official `merge_lora()`.

The constructor prints `missing=0, unexpected=67` during its official
ViT-only initialization because it passes the complete RelayFormer state to
the backbone's partial loader. This is not the final checkpoint load. The
subsequent full-model strict load succeeds without missing or unexpected
keys, and all registered parameter and buffer counts match.

The registered buffer count, 786,435 elements, describes the strict-load
model before `merge_lora()`. The official merge implementation materializes
large derived local/global weight buffers for faster inference; those
ephemeral derived buffers are not checkpoint tensors and are not included in
the checkpoint-state count.

The repository also links image-and-video and application-oriented weights.
They were not selected: the former is hosted without a stable published
checksum, while the latter uses a different `tokens_per_patch=2` setting.
This run uses the official image-only paper checkpoint with
`tokens_per_patch=3`.

## 3. What RelayFormer does

RelayFormer is designed to preserve fine local forensic evidence while still
allowing distant image regions to exchange context:

1. a 1024 canvas is divided into four overlapping 528×528 sub-images;
2. each sub-image is encoded locally by a ViT-Base backbone;
3. a small set of Global-Local Relay (GLR) tokens is appended to each
   sub-image and absorbs local evidence;
4. relay tokens from all sub-images attend globally, using 4D rotary
   positional encoding for temporal position, relay-token identity, vertical
   position, and horizontal position;
5. the enriched relay tokens are returned to their local sub-images at the
   next block, iteratively carrying global context without full
   all-pixel attention;
6. overlapping features are reassembled; and
7. a lightweight query decoder uses cross-attention, self-attention, and
   learned gates to combine eight query masks into the final logit map.

The architecture is strong because it avoids the usual choice between
strictly local evidence and expensive full-image attention. The 16-pixel
sub-image overlap preserves boundary continuity, while relay tokens can
compare object and scene consistency across quadrants. The paper's ablations
report that two relay tokens outperform zero, one, or three; removing 4D RoPE
also lowers average benchmark performance.

This design does not guarantee transfer to CLAIMFORGE. The image checkpoint
was trained on CASIAv2-style conventional manipulations, whereas the canonical
mouse edit occupies a median **0.1126%** of native pixels and is produced by a
well-blended local diffusion insertion. RelayFormer shows a genuine
high-quality success mode, but most edits still provide too little or the
wrong kind of learned forensic evidence.

## 4. Exact inference contract and release discrepancy

### Frozen paper-v3 input protocol

The paper's v3 Appendix explicitly states that images above 1024 are scaled
so the long edge becomes 1024 while preserving aspect ratio, then zero-padded
to 1024×1024. The pre-registered CLAIMFORGE protocol follows that statement:

1. decode the exact canonical JPEG bytes with Pillow and convert to RGB;
2. if the native long edge exceeds 1024, downscale only that image with
   `Pillow.Image.thumbnail`, bilinear interpolation, and
   `reducing_gap=None`;
3. preserve aspect ratio using Pillow's deterministic thumbnail rounding;
4. place the resized or unchanged image at the top-left of a 1024×1024
   raw-RGB canvas and zero-pad only its right and bottom;
5. divide the resulting uint8 canvas by 255 in float32 and apply ImageNet
   normalization; and
6. pass the valid resized height and width as `origin_shape`, with
   `clip_len=1`.

There is no crop, re-encoding, test-time augmentation, ensemble, or second
forward pass. The first canonical real input changes from 1800×1350 to
1024×768 and has frozen preprocessing-tensor SHA-256
`7a32d4419d732be17259e5249c91d4281a4821baa895b9f6209e28c26e7fd7e4`.

Normalization is an explicit NumPy float32 implementation of the same
`(pixel / 255 - mean) / std` formula used by the release. It is not claimed
to be bit-exact with Albumentations 1.3.0 `A.Normalize`, whose internal
operation ordering can differ by a few float32 ULPs. This implementation and
its tensor hashes were frozen before smoke and full metrics; changing it
after observing results would constitute a new protocol, not a correction to
this run.

The model uses the paper configuration: 1024 input, a 2×2 grid, 528-pixel
sub-images with 16-pixel overlap, 33-pixel feature sub-images with one-pixel
overlap, and three tokens per patch.

### Why this is not release-`infer.py` bit-exact preprocessing

The current released `datasets/inference_dataset.py` does not implement the
Appendix rule for large images. It applies `PadIfNeeded` and then a top-left
`Crop(0, 0, 1024, 1024)`. Images larger than the canvas are therefore
truncated rather than downscaled.

That difference is material on this benchmark:

- 404/550 canonical images have a long edge above 1024;
- under the released top-left crop, 81/275 forged edit boxes would be fully
  outside the model input; and
- another 9/275 edit boxes would be partially cut.

Using that release path would measure edit position relative to an accidental
crop as much as model quality. The main result therefore uses the latest
paper's explicit preprocessing contract and labels it as a **paper-v3
compatibility protocol**, not as bit-exact execution of the released
`infer.py`.

There is also tension within the paper: its headline and raw-4K ablation
emphasize processing without interpolation, while the v3 Appendix caps large
images at a 1024-pixel long edge. CLAIMFORGE follows the more explicit
Appendix rule for the released 1024 checkpoint and records every resize.
Consequently, 404 inputs in this run are interpolated; the report does not
claim raw-resolution evaluation.

### Output and native adapter

One official forward produces a 1024×1024 float32 manipulation probability.
The adapter temporarily wraps the instance's official
`assemble_and_decode` method to capture its pre-sigmoid 1024×1024 logits
without changing upstream source or adding a second forward pass.

The auxiliary model-space result:

- crops away right/bottom padding;
- scores only the valid resized content;
- resizes the exact native GT to that content with nearest-neighbor
  interpolation; and
- uses the official continuous probability for AP and strict
  `probability > 0.5` for binary metrics.

For the primary native result, the adapter independently:

1. crops padding from both logits and probability;
2. if the input was downscaled, restores each continuous map to canonical
   dimensions with PyTorch bilinear interpolation and
   `align_corners=False`; and
3. applies the same strict threshold only after probability restoration.

Native GT is the exact canonical pixel-difference mask. Prediction inversion
and permutation F1 are not used. Both the full model logits/probability and
native logits/probability are retained, together with a lossless native
threshold mask.

## 5. Preflight, smoke, and resume checks

The final-code one-pair preflight completed 2/2 images:

- forged native pixel AP `0.00314480`;
- forged native F1 `0`;
- real false-positive area `1.579918%`; and
- no runtime or artifact errors.

The five-pair smoke completed 10/10 images:

- native macro AP `0.18723784`;
- native macro F1 `0.15458333`;
- native micro F1 `0.03775249`;
- mean real false-positive area `0.444675%`;
- 1/5 forged masks overlapped its edit box; and
- 1/5 reached edit-box IoU greater than 0.3.

The smoke resumed with ten selected and zero pending. Its JSONL retained
SHA-256
`8de54aeca12d6bb4cce87f081a9cd97b58bbed4d6064d181f3bc788177e06b38`.
Its ten shared rows are also identical to the full-run prefix for all 26
registered artifact, preprocessing, geometry, threshold, and metric fields.

The 275-pair full run completed 550/550 images with no errors. Re-running the
same command reported 550 selected and zero pending. The JSONL remained 550
rows and retained SHA-256
`83b9f03e6729c4d82b5d1f9f447c73f9a20d20fcc8482d70ba9806c01ede963f`.

## 6. Primary native T2 result

Confidence intervals below come from the final independent analyzer using
1,000 task-paired bootstrap resamples with seed `20260724`.

| Metric | Estimate | Pair-bootstrap 95% CI |
|---|---:|---:|
| Macro pixel AP | **0.227755** | [0.188467, 0.268776] |
| Median per-image pixel AP | **0.025562** | — |
| Macro precision at `> 0.5` | 0.222333 | — |
| Macro recall at `> 0.5` | 0.225672 | — |
| Macro F1 at `> 0.5` | **0.187240** | [0.155931, 0.219522] |
| Macro IoU at `> 0.5` | **0.132916** | [0.108850, 0.157461] |
| Micro F1 at `> 0.5` | **0.095885** | [0.073259, 0.127001] |
| Micro IoU at `> 0.5` | **0.050357** | [0.038022, 0.067807] |
| Real FP area, macro | **0.614692%** | [0.525964%, 0.724452%] |
| Real FP area, micro | 0.617130% | [0.511267%, 0.739713%] |

The forged native confusion counts at the fixed threshold are:

| TP | FP | FN | TN |
|---:|---:|---:|---:|
| 143,634 | 2,256,418 | 452,269 | 439,400,683 |

This corresponds to micro precision `0.059846`, recall `0.241036`, and a
predicted-positive fraction of `0.542688%` on forged images.

The result distribution is strongly skewed:

- AP ranges from `0.000183` to `0.993251`;
- 104/275 images have AP at least 0.1;
- 62/275 have AP at least 0.5, and 24/275 have AP at least 0.9;
- all 275 forged masks are nonempty, but only 128 hit an exact-GT pixel;
- 105/275 reach F1 at least 0.1, 78/275 reach 0.3, and 53/275 reach 0.5;
- 55/275 reach exact-GT IoU greater than 0.3; and
- median F1 and median IoU are both zero.

RelayFormer therefore has a substantial success subset, but its mean is not
representative of the typical task.

## 7. Domain and edit-size behavior

### Domain

| Domain | Pairs | Macro AP | Median AP | Macro F1 | Micro F1 | Real FP area |
|---|---:|---:|---:|---:|---:|---:|
| Lodging | 147 | **0.303770** | 0.094133 | 0.249667 | 0.113261 | 0.6428% |
| Restaurant | 128 | **0.140456** | 0.010835 | 0.115546 | 0.074325 | 0.5824% |

Lodging macro AP and F1 are each about 2.16 times the restaurant result.
Pristine false-positive area is similar, so the gap is not explained simply
by one domain receiving much larger predicted masks.

### Native edit-area quintile

| Quintile | Native edit range | Macro AP | Median AP | Macro F1 | Micro F1 |
|---|---:|---:|---:|---:|---:|
| Q1, smallest | 0.0249–0.0672% | 0.052953 | 0.003623 | 0.041523 | 0.020171 |
| Q2 | 0.0673–0.0938% | 0.095604 | 0.010659 | 0.085820 | 0.041592 |
| Q3 | 0.0953–0.1295% | 0.199107 | 0.021642 | 0.174292 | 0.092548 |
| Q4 | 0.1310–0.2178% | 0.307504 | 0.094875 | 0.250904 | 0.091843 |
| Q5, largest | 0.2196–1.2927% | **0.483606** | **0.495590** | **0.383660** | **0.254077** |

The monotonic macro curve is unusually clear. Q5 versus Q1 is a 9.13-fold AP
increase and a 9.24-fold F1 increase; median AP grows by about 137 times.
RelayFormer is not wholly blind to very small edits, but usable localization
becomes much more common as the manipulated evidence grows.

## 8. Model-valid-content versus native space

| Metric | Valid model content | Native adapter |
|---|---:|---:|
| Macro AP | 0.227615 | 0.227755 |
| Macro F1 | 0.187170 | 0.187240 |
| Macro IoU | 0.132853 | 0.132916 |
| Micro F1 | 0.123214 | 0.095885 |
| Micro IoU | 0.065651 | 0.050357 |
| Real FP area, macro | 0.615160% | 0.614692% |

Macro estimates are effectively unchanged, so native restoration does not
explain the model's typical failures. The micro difference reflects both
geometry and weighting: valid model content gives each resized pixel its
model-space weight, whereas native micro aggregation uses each canonical
image's native dimensions and exact pixel mask.

The auxiliary result is deliberately not described as the complete official
1024 canvas: padded pixels are excluded from metrics, since they do not
correspond to image content.

## 9. Fixed threshold and post-hoc diagnostic

The strict official 0.5 threshold remains the only primary threshold. A
descriptive test-label oracle was computed solely to test whether
miscalibration explains the result:

- the best single 65,536-bin global threshold is `0.631876`;
- its native micro F1 is `0.101467`, versus the primary `0.095885`; and
- a separate oracle threshold for every image gives mean F1 `0.246971` but
  median F1 only `0.069745`.

These values use test-set labels and are ineligible for the main table. The
single global oracle improves micro F1 by only `0.005582`, so changing one
threshold does not resolve the main localization errors. The per-image
oracle confirms useful ranking signal in a subset, but even its median
remains low.

## 10. Pristine behavior and edit-box overlap

At `> 0.5`, 271/275 pristine images have a nonempty native mask. Their
false-positive area has:

- mean `0.614692%`;
- median `0.345197%`;
- 95th percentile `2.058683%`; and
- maximum `6.849609%`.

Forty-two pristine images exceed 1% positive area, and two exceed 5%.
Nonempty-mask rate cannot be interpreted as T1 detection: nearly every
pristine image is also nonempty, and the checkpoint has no image-level head.

For the 275 forged images:

- 129 masks overlap the half-open registered edit box at all;
- 34 have edit-box IoU greater than 0.3;
- mean edit-box IoU is `0.094509`;
- median edit-box IoU is `0`; and
- maximum edit-box IoU is `0.655050`.

The 129 box overlaps versus 128 exact-GT pixel hits are not contradictory.
The registered rectangle includes pixels that may be unchanged in the exact
difference mask.

## 11. Position among completed local methods

All rows below use the same 275 paired tasks, canonical JPEGs, native
exact-difference GT, and each method's registered public threshold. Native
continuous AP is the cleanest common ranking metric.

| Method | Native macro AP | Macro F1 | Macro IoU | Micro F1 | Real FP area |
|---|---:|---:|---:|---:|---:|
| CAT-Net v2 | 0.612336 | 0.474814 | 0.357430 | 0.307062 | 0.4359% |
| TruFor | 0.579892 | 0.500100 | 0.374100 | 0.330555 | 1.7764% |
| **RelayFormer** | **0.227755** | **0.187240** | **0.132916** | **0.095885** | **0.6147%** |
| IML-ViT | 0.155126 | 0.143250 | 0.098432 | 0.069951 | 0.8023% |
| Mesorch | 0.092200 | 0.045363 | 0.031371 | 0.036127 | 0.3578% |
| MaskCLIP | 0.047401 | 0.005641 | 0.003666 | 0.006219 | 0.0596% |
| MVSS-Net | 0.034147 | 0.019619 | 0.012484 | 0.007352 | 3.2887% |
| PSCC-Net | 0.015278 | 0.000073 | 0.000036 | 0.002297 | 0.8265% |
| HiFi-IFDL | 0.003613 | 0.000055 | 0.000028 | 0.004461 | 0.7289% |

RelayFormer ranks third by native macro AP. It is `0.072629` above IML-ViT
and `0.352138` below TruFor. This is a cross-checkpoint robustness
comparison, not a controlled architecture ablation: training data, model
resolution, output construction, and fixed operating thresholds differ.

## 12. Independent audit and tests

The final analyzer reports:

| Audit item | Result |
|---|---:|
| Physical/latest result rows | 550 / 550 |
| Unique expected result IDs | 550 |
| Pinned official source files | 10 |
| Pinned adapter-contract files | 3 |
| Files checked end to end | 3,575 |
| Model-probability replay max absolute error | `1.1920929e-7` |
| Native-logit replay max absolute error | **0** |
| Native-probability replay max absolute error | `1.7881393e-7` |
| Threshold-mask disagreements | **0 pixels** |
| Smoke-prefix reproducibility | 10/10 images, exact |
| Duplicate result rows | **0** |
| Provenance status | `ok` |
| Artifact status | `ok` |

The analyzer independently verifies the canonical image and GT hashes,
Pillow decode, conditional thumbnail geometry and rounding, padding,
normalization tensor hash, finite 1024 logits, sigmoid probability,
right/bottom padding removal, native bilinear restoration, lossless mask, and
both localization spaces. A regression test also rejects coordinated
probability/mask/metric/hash tampering across the strict 0.5 boundary.

RelayFormer runner, metric, and analyzer tests: **38 passed**. The complete
repository suite reports **276 passed, 4 skipped**. The only warnings are
upstream SciPy namespace deprecations and a timm import deprecation.

## 13. Runtime and artifacts

The recorded environment uses CPython 3.12.3, PyTorch
`2.8.0.dev20250627+cu128`, torchvision
`0.23.0.dev20250627+cu128`, timm 1.0.15, IMDLBenCo 0.1.45,
rotary-embedding-torch 0.9.1, Albumentations 1.3.0, NumPy 1.26.4,
scikit-learn 1.5.2, Pillow 11.1.0, and OpenCV headless 4.11.0 on an NVIDIA
L20Z.

The official README recommends Python 3.9. The run uses Python 3.12 because
that is the available pinned accelerator environment. Every critical package
path and version is recorded, and preprocessing and outputs are independently
replayed.

Recorded forward latency over 550 images:

| Statistic | Milliseconds |
|---|---:|
| Median | 35.496 |
| P95 | 36.318 |
| Mean including first-image warm-up | 35.710 |
| Maximum first-image warm-up | 287.671 |

Peak allocated CUDA memory is 1,299,416,576 bytes, approximately 1.21 GiB.
The complete full-run artifact directory is 11,691,427,323 bytes,
approximately 10.89 GiB. It retains five model artifacts for every image:
1024 logits, native logits, 1024 probability, native probability, and the
native lossless threshold mask.

Primary machine-readable files:

- `results/opensource/relayformer/relayformer_checkpoint164_paper_v3_mouse_canonical_v1_full275_20260724.jsonl`
- `results/opensource/relayformer/relayformer_checkpoint164_paper_v3_mouse_canonical_v1_full275_20260724.run_manifest.json`
- `results/opensource/relayformer/relayformer_checkpoint164_paper_v3_mouse_canonical_v1_full275_20260724.summary.json`
- `results/opensource/relayformer/relayformer_checkpoint164_paper_v3_mouse_canonical_v1_full275_20260724.analysis.json`
- `outputs/opensource/relayformer/relayformer_checkpoint164_paper_v3_mouse_canonical_v1_full275_20260724/`

Implementation:

- `eval/opensource/run_relayformer.py`
- `eval/opensource/relayformer_metrics.py`
- `eval/opensource/analyze_relayformer_run.py`
- `tests/test_run_relayformer.py`
- `tests/test_relayformer_metrics.py`
- `tests/test_analyze_relayformer_run.py`

## 14. Reproduction

Full inference or artifact-validating resume:

```bash
PYTHONDONTWRITEBYTECODE=1 \
/root/.cache/claimforge/venvs/relayformer-3fc863c/bin/python \
  -m eval.opensource.run_relayformer \
  --repo-root /root/claimforge-benchmark \
  --run-id relayformer_checkpoint164_paper_v3_mouse_canonical_v1_full275_20260724 \
  --seed 42 \
  --device cuda:0 \
  --fail-fast
```

Independent analysis:

```bash
PYTHONDONTWRITEBYTECODE=1 \
/root/.cache/claimforge/venvs/relayformer-3fc863c/bin/python \
  -m eval.opensource.analyze_relayformer_run \
  --repo-root /root/claimforge-benchmark \
  --run-id relayformer_checkpoint164_paper_v3_mouse_canonical_v1_full275_20260724 \
  --results-dir results/opensource/relayformer \
  --output \
    results/opensource/relayformer/relayformer_checkpoint164_paper_v3_mouse_canonical_v1_full275_20260724.analysis.json \
  --bootstrap-iterations 1000 \
  --bootstrap-seed 20260724 \
  --prefix-run-id relayformer_checkpoint164_paper_v3_mouse_canonical_v1_smoke5_20260724 \
  --prefix-results-dir results/opensource/relayformer
```

## 15. Bottom line and next method

RelayFormer is a meaningful improvement over most completed localizers. Its
relay-based local/global design produces high-quality masks on a sizeable
minority of examples and places third of nine by native macro AP. However, a
typical tiny insertion is still poorly ranked: median AP is `0.025562`,
median F1 is zero, and 147/275 forged images receive no exact-GT pixel at the
official threshold.

For the CLAIMFORGE main table, RelayFormer should be reported as **T2 only,
third of nine completed local methods by native macro AP, with strong
edit-size and domain dependence and a clearly disclosed paper-versus-release
preprocessing discrepancy**.

The next frozen local method is **DINOv3-IML**, using its official CAT
ViT-L LoRA-r32 checkpoint and labeled as a non-peer-reviewed 2026 preprint.
