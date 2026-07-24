# Mesorch epoch 98 on the canonical mouse set (2026-07-24)

## 1. Status and headline

The eighth publicly reproducible local-manipulation baseline is complete.
The official unpruned `MesorchFull` checkpoint at epoch 98 ran on all 275
matched tasks, or 550 canonical JPEG images, with no errors.

Mesorch is materially better than the weaker completed localizers on this
condition, but its success is concentrated in a small tail of larger edits:

- native macro pixel AP is **0.092200**, with task-paired bootstrap 95% CI
  **[0.067674, 0.122684]**;
- median per-image pixel AP is only **0.004910**;
- native macro pixel F1 at the official strict `probability > 0.5` rule is
  **0.045363**, CI **[0.029625, 0.065018]**;
- native micro pixel F1 is **0.036127**, CI
  **[0.017653, 0.063989]**;
- 136/275 forged images have any positive prediction, but only 34/275 have
  any true-positive pixel;
- 23/275 forged images reach pixel AP at least 0.5, while 11/275 reach F1 at
  least 0.5;
- only 34/275 masks overlap the registered edit box at all, and 10/275 reach
  edit-box IoU greater than 0.3; and
- the mean pristine false-positive area is **0.3578%**, CI
  **[0.2188%, 0.5545%]**.

Edit size explains much of the long tail. The smallest two native edit-size
quintiles have zero F1 at threshold 0.5. The largest quintile reaches macro
AP **0.311238** and macro F1 **0.173439**, compared with AP **0.010104** and
F1 **0** in the smallest quintile.

The independent analyzer validated 3,025 files, reproduced the official
preprocessing, reconstructed the 512×512 probability from captured 128×128
logits, restored every map to native resolution, regenerated every threshold
mask, and recomputed every metric. Native probability replay is bit-exact for
all 550 images. This is therefore an audited failure on most tiny insertions,
not a checkpoint-loading, score-direction, or interpolation error.

Mesorch exposes no image-classification head. Its official
`pred_label` is `None`, so T1 is **N/A**. No map mean, maximum, or area is
promoted to an unofficial image-level detector.

## 2. Pinned method and official release

The run uses the authors'
[official Mesorch repository](https://github.com/scu-zjz/Mesorch) at commit
[`ea82b0274b92244115d09b81663c88f57c7b78ee`](https://github.com/scu-zjz/Mesorch/tree/ea82b0274b92244115d09b81663c88f57c7b78ee).
The method is described in the
[AAAI 2025 paper](https://arxiv.org/abs/2412.13753).

The repository code is MIT licensed. The externally hosted checkpoint does
not have a separately stated license, so this report does not extend the code
license to the weight file.

The selected official artifact is:

| Field | Value |
|---|---|
| Released filename | `mesorch-98.pth` |
| Provider | Author-linked Google Drive |
| Drive file ID | `1PJxKteinMyaAYokKy0JhuzBnBc6bGsau` |
| Last modified | 2024-12-19 08:31:40 UTC |
| Bytes | 1,023,886,070 |
| SHA-256 | `6d8fcd7ce7616d819bec6a9ed461b27187101e67247f8b2d2483fdc1f25f685a` |
| Recorded epoch | 98 |
| Model tensors | 804 float32 tensors |
| Parameters | 85,753,944 |
| Buffer elements | 0 |

The checkpoint is a mapping with exact top-level keys `model`, `optimizer`,
`epoch`, `scaler`, and `args`. It loads with `weights_only=True`, an explicit
`argparse.Namespace` allowlist, and `strict=True`. The adapter selects only
the top-level `model` state and performs no prefix rewrite, schema fallback,
or hidden pretrained-weight download.

The current upstream inference entry point has a reproducibility defect.
Commit `998fe8d` renamed the registered class from `Mesorch` to
`MesorchFull`, but `test.py` still imports `Mesorch` and
`test_mesorch_f1.sh` still requests `--model Mesorch`. CLAIMFORGE leaves the
official checkout clean and directly instantiates the now-canonical
`MesorchFull` class. The loaded state schema and output path remain official.

## 3. What Mesorch does

Mesorch is designed around a “mesoscopic” view of manipulation evidence:
small forensic artifacts alone may miss semantic inconsistency, while global
semantics alone may miss fine local traces. Its unpruned model combines both:

1. a DCT module separates high- and low-frequency representations;
2. RGB plus high-frequency input enters a ConvNeXt local branch;
3. RGB plus low-frequency input enters a SegFormer global branch;
4. each branch emits four feature scales and a decoder converts all eight
   features into candidate manipulation logits;
5. a nine-channel RGB/high/low-frequency gating network predicts eight
   pixel-wise softmax weights; and
6. the weighted logits are summed, resized, and passed through sigmoid.

This architecture can be strong because the branches are complementary.
ConvNeXt preserves local texture and boundary evidence; SegFormer supplies
long-range object and scene structure; DCT features expose frequency
irregularities; and the gate can choose which branch and scale to trust at
each pixel. The paper's ablations support the combination rather than any
single branch alone.

The same design does not guarantee transfer to CLAIMFORGE. The official model
was trained under Protocol-CAT at 512×512 on conventional manipulation data.
The canonical mouse edit occupies a median **0.1126%** of native pixels.
Directly stretching the full image to 512×512 leaves very little support for
many insertions, while a well-blended diffusion object may not reproduce the
local artifacts or semantic violations learned from Protocol-CAT.

The observed size curve is consistent with this explanation: Mesorch can be
very good on a minority of sufficiently visible edits, but usually does not
cross its fixed decision threshold for the smallest edits.

## 4. Exact inference contract

### Input

The adapter reproduces the official evaluation transform:

1. decode the canonical JPEG bytes with Pillow and convert to RGB;
2. stretch directly to 512×512 with
   `albumentations.Resize(512, 512)` and OpenCV linear interpolation;
3. scale uint8 channels to `[0, 1]`;
4. normalize with ImageNet mean `(0.485, 0.456, 0.406)` and standard
   deviation `(0.229, 0.224, 0.225)`; and
5. apply the official no-op crop and `ToTensorV2`.

Aspect ratio is not preserved. There is no padding, re-encoding, test-time
augmentation, or ensemble. The frozen preprocessing tensor SHA-256 for the
first canonical input is
`9bd5b56c520796c8a75bc8016d4a373e4151e4006a4d231d57144a132089739d`.

The official `forward` requires a mask only because it computes
`BCEWithLogitsLoss`. In inference, the adapter supplies a 512×512 all-zero
dummy mask. The mask does not feed either feature branch, the gate, or
`pred_mask`.

### Official model-space output

One forward pass produces a 128×128 fused logit map. The model then applies:

```text
bilinear resize to 512×512, align_corners=True
sigmoid
strict probability > 0.5 mask
```

The adapter captures the 128×128 tensor using one pre-hook on the official
resize module and retains both the raw logits and the official 512×512
float32 probability. It does not run a second forward pass.

The model-space GT is the binary native GT resized to 512×512 with OpenCV
nearest-neighbor interpolation. Pixel AP uses the continuous manipulation
probability. F1, IoU, precision, recall, and MCC use the official strict
`> 0.5` rule; prediction inversion and permutation F1 are not used for the
primary result.

### Native CLAIMFORGE adapter

The primary cross-method result is native-space:

1. bilinearly restore the continuous official 512×512 probability to the
   canonical input dimensions with `align_corners=False`;
2. compare it with the exact native canonical difference mask; and
3. apply the same strict `probability > 0.5` rule after restoration.

This restoration is a benchmark compatibility adapter, not an upstream model
head. The official 512×512 result is retained as an auxiliary result so the
effect of restoration is visible.

## 5. Preflight, smoke, and resume checks

The final-code one-pair preflight completed 2/2 images:

- forged native pixel AP `0.00977028`;
- forged native F1 `0`;
- real false-positive area `0.116049%`; and
- no runtime or artifact errors.

The five-pair smoke completed 10/10 images:

- native macro AP `0.0777471`;
- native macro F1 `0.0376399`;
- native micro F1 `0.00393508`;
- mean real false-positive area `0.712032%`;
- 1/5 forged masks overlapped its edit box; and
- 0/5 reached edit-box IoU greater than 0.3.

The smoke resumed with ten selected and zero pending. Its ten shared rows are
identical to the full-run prefix for 20 deterministic fields, including raw
logits, both probability maps, masks, preprocessing evidence, and metrics.

The 275-pair full run completed 550/550 images with no errors. Re-running the
same command reported 550 selected and zero pending. The JSONL remained 550
rows and retained SHA-256
`cfdd025541871116044f60f584ef7abe4c3b194128e1977af080b0531da47a91`.

## 6. Primary native T2 result

Confidence intervals below come from the final independent analyzer using
1,000 task-paired bootstrap resamples with seed `20260724`.

| Metric | Estimate | Pair-bootstrap 95% CI |
|---|---:|---:|
| Macro pixel AP | **0.092200** | [0.067674, 0.122684] |
| Median per-image pixel AP | **0.004910** | — |
| Macro precision at `> 0.5` | 0.165323 | — |
| Macro recall at `> 0.5` | 0.037333 | — |
| Macro F1 at `> 0.5` | **0.045363** | [0.029625, 0.065018] |
| Macro IoU at `> 0.5` | **0.031371** | [0.019578, 0.045923] |
| Micro F1 at `> 0.5` | **0.036127** | [0.017653, 0.063989] |
| Micro IoU at `> 0.5` | **0.018396** | [0.008905, 0.033052] |
| Real FP area, macro | **0.357825%** | [0.218763%, 0.554481%] |
| Real FP area, micro | 0.347091% | [0.222352%, 0.485132%] |

The forged native confusion counts at the fixed threshold are:

| TP | FP | FN | TN |
|---:|---:|---:|---:|
| 36,629 | 1,395,243 | 559,274 | 440,261,858 |

This corresponds to micro precision `0.025581`, recall `0.061468`, and a
predicted-positive fraction of `0.323768%` on forged images.

The result distribution is highly skewed:

- AP ranges from `0.000246` to `0.977912`;
- 42/275 images have AP at least 0.1;
- 23/275 have AP at least 0.5;
- 136/275 have a nonempty fixed-threshold mask;
- 34/275 have any true-positive pixel;
- 12/275 reach exact-GT IoU greater than 0.3; and
- median F1 and median IoU are both zero.

The mean is therefore not representative of a typical task. Mesorch contains
a real high-quality success mode, but it is activated on a minority of this
dataset.

## 7. Domain and edit-size behavior

### Domain

| Domain | Pairs | Macro AP | Median AP | Macro F1 | Micro F1 | Real FP area |
|---|---:|---:|---:|---:|---:|---:|
| Lodging | 147 | **0.136414** | 0.010016 | 0.066991 | 0.037990 | 0.3476% |
| Restaurant | 128 | **0.041423** | 0.003042 | 0.020524 | 0.033851 | 0.3696% |

Lodging AP is about 3.3 times restaurant AP. Real false-positive area is
similar across the two domains, so the AP gap is not explained simply by one
domain receiving much larger masks.

### Native edit-area quintile

| Quintile | Native edit range | Macro AP | Median AP | Macro F1 | Micro F1 |
|---|---:|---:|---:|---:|---:|
| Q1, smallest | 0.0249–0.0672% | 0.010104 | 0.001564 | **0** | **0** |
| Q2 | 0.0673–0.0938% | 0.018968 | 0.003201 | **0** | **0** |
| Q3 | 0.0953–0.1295% | 0.035716 | 0.003813 | 0.014971 | 0.009718 |
| Q4 | 0.1310–0.2178% | 0.084976 | 0.008635 | 0.038404 | 0.020245 |
| Q5, largest | 0.2196–1.2927% | **0.311238** | **0.066599** | **0.173439** | **0.130225** |

The monotonic increase is the clearest diagnostic in the run. Continuous AP
begins rising before the fixed mask becomes useful, while the first two
quintiles never produce a true-positive threshold mask. Mesorch is sensitive
to the amount of manipulated evidence available after the 512×512 stretch.

## 8. Model-space versus native-space

| Metric | Official 512×512 | Native adapter |
|---|---:|---:|
| Macro AP | 0.092488 | 0.092200 |
| Macro F1 | 0.045703 | 0.045363 |
| Macro IoU | 0.031632 | 0.031371 |
| Micro F1 | 0.071053 | 0.036127 |
| Micro IoU | 0.036835 | 0.018396 |
| Real FP area, macro | 0.358651% | 0.357825% |

Macro estimates agree closely, so native restoration does not explain the
low typical performance. The larger micro-F1 difference reflects both
interpolation/mask geometry and weighting: every model-space image has the
same 512×512 pixel weight, whereas native micro aggregation weights the
canonical dimensions and exact native masks.

## 9. Fixed threshold and post-hoc diagnostic

The official fixed threshold remains the only primary threshold. A
descriptive test-set oracle was computed only to test whether a poor choice
of 0.5 explains the result:

- the best single histogram-approximated global threshold is `0.781903`;
- its native micro F1 is `0.036782`, only slightly above the primary
  `0.036127`; and
- a separate oracle threshold for every image gives mean F1 `0.108887` but
  median F1 only `0.013239`.

These numbers use test labels and are ineligible for the main table. They show
that a single recalibrated threshold would not solve the localization
failure. Per-image tuning can expose more of the continuous ranking signal,
but that is not a deployable zero-shot protocol.

## 10. Pristine behavior and edit-box overlap

At `> 0.5`, 129/275 pristine images have a nonempty mask. Their native
false-positive area has:

- mean `0.357825%`;
- median `0`;
- 95th percentile `2.025593%`; and
- maximum `18.433594%`.

The low mean false-positive area should not be read in isolation as strong
specificity: the forged recall is also only `3.7333%` on average, so the model
is generally sparse.

For the 275 forged images:

- 34 masks overlap the half-open registered edit box at all;
- 10 have edit-box IoU greater than 0.3;
- mean edit-box IoU is `0.024604`;
- median edit-box IoU is `0`; and
- maximum edit-box IoU is `0.746141`.

## 11. Position among completed local methods

All rows below use the same 275 paired tasks, canonical JPEGs, native
exact-difference GT, and each method's registered public threshold. Native
continuous AP is the cleanest common ranking metric.

| Method | Native macro AP | Macro F1 | Macro IoU | Micro F1 | Real FP area |
|---|---:|---:|---:|---:|---:|
| CAT-Net v2 | 0.612336 | 0.474814 | 0.357430 | 0.307062 | 0.4359% |
| TruFor | 0.579892 | 0.500100 | 0.374100 | 0.330555 | 1.7764% |
| IML-ViT | 0.155126 | 0.143250 | 0.098432 | 0.069951 | 0.8023% |
| **Mesorch** | **0.092200** | **0.045363** | **0.031371** | **0.036127** | **0.3578%** |
| MaskCLIP | 0.047401 | 0.005641 | 0.003666 | 0.006219 | 0.0596% |
| MVSS-Net | 0.034147 | 0.019619 | 0.012484 | 0.007352 | 3.2887% |
| PSCC-Net | 0.015278 | 0.000073 | 0.000036 | 0.002297 | 0.8265% |
| HiFi-Net | 0.003613 | 0.000055 | 0.000028 | 0.004461 | 0.7289% |

Mesorch ranks fourth by native macro AP, between IML-ViT and MaskCLIP. Its AP
is `0.044799` above MaskCLIP but `0.062925` below IML-ViT. This is a
cross-checkpoint robustness comparison, not a controlled architecture
ablation: training data, native resolution, output construction, and fixed
thresholds differ across methods.

## 12. Independent audit and tests

The final analyzer reports:

| Audit item | Result |
|---|---:|
| Physical/latest result rows | 550 / 550 |
| Unique expected result IDs | 550 |
| Pinned official source files | 8 |
| Pinned adapter-contract files | 3 |
| Files checked end to end | 3,025 |
| Model-probability replay max absolute error | `1.1920929e-7` |
| Native-probability replay max absolute error | **0** |
| Threshold-mask disagreements | **0 pixels** |
| Smoke-prefix reproducibility | 10/10 images, exact |
| Provenance status | `ok` |
| Artifact status | `ok` |

During the first full audit, a conventional NumPy bilinear implementation
differed from CUDA by at most `3.7551e-6` on three high-resolution images,
with zero threshold-mask differences. The analyzer was improved instead of
loosening the tolerance: it now independently reproduces PyTorch CUDA's
float32 fused-multiply-add coordinate and weighted-sum order. All 550 native
maps then became bit-exact. CUDA golden regression tests cover both
`align_corners` modes and a realistic 512→1800×1200 restoration.

Test results:

- Mesorch runner/metrics/analyzer tests: **41 passed**;
- complete repository suite: **238 passed, 4 skipped**; and
- warnings are limited to upstream SciPy namespace deprecations and a timm
  import deprecation.

## 13. Runtime and artifacts

The recorded environment uses CPython 3.12.3, PyTorch
`2.8.0.dev20250627+cu128`, timm 1.0.15, IMDLBenCo 0.1.45,
Albumentations 1.3.0, NumPy 1.26.4, scikit-learn 1.5.2, Pillow 11.1.0, and
OpenCV headless 4.11.0 on an NVIDIA L20Z.

The official README recommends Python 3.10. The run uses Python 3.12 because
that is the available pinned accelerator environment; the manifest records
every critical package path and version, and preprocessing, logits,
probabilities, and masks are independently replayed.

Recorded forward latency over 550 images:

| Statistic | Milliseconds |
|---|---:|
| Median | 18.3275 |
| P95 | 19.4246 |
| Mean including first-load warm-up | 22.8631 |
| Maximum first-image warm-up | 2436.315 |

Peak allocated CUDA memory is approximately 815.63 MB. The complete artifact
directory is approximately 3.9 GB and retains four artifacts for every image:
128×128 logits, official 512×512 probability, native probability, and native
lossless threshold mask.

Primary machine-readable files:

- `results/opensource/mesorch/mesorch_epoch98_mouse_canonical_v1_full275_20260724.jsonl`
- `results/opensource/mesorch/mesorch_epoch98_mouse_canonical_v1_full275_20260724.run_manifest.json`
- `results/opensource/mesorch/mesorch_epoch98_mouse_canonical_v1_full275_20260724.summary.json`
- `results/opensource/mesorch/mesorch_epoch98_mouse_canonical_v1_full275_20260724.analysis.json`
- `outputs/opensource/mesorch/mesorch_epoch98_mouse_canonical_v1_full275_20260724/`

Implementation:

- `eval/opensource/run_mesorch.py`
- `eval/opensource/mesorch_metrics.py`
- `eval/opensource/analyze_mesorch_run.py`
- `tests/test_run_mesorch.py`
- `tests/test_mesorch_metrics.py`
- `tests/test_analyze_mesorch_run.py`

## 14. Reproduction

Full inference or artifact-validating resume:

```bash
PYTHONDONTWRITEBYTECODE=1 \
/root/.cache/claimforge/venvs/mesorch-ea82b02/bin/python \
  -m eval.opensource.run_mesorch \
  --repo-root /root/claimforge-benchmark \
  --run-id mesorch_epoch98_mouse_canonical_v1_full275_20260724 \
  --seed 20260724 \
  --device cuda:0 \
  --fail-fast
```

Independent analysis:

```bash
PYTHONDONTWRITEBYTECODE=1 \
/root/.cache/claimforge/venvs/mesorch-ea82b02/bin/python \
  -m eval.opensource.analyze_mesorch_run \
  --repo-root /root/claimforge-benchmark \
  --run-id mesorch_epoch98_mouse_canonical_v1_full275_20260724 \
  --results-dir results/opensource/mesorch \
  --output \
    results/opensource/mesorch/mesorch_epoch98_mouse_canonical_v1_full275_20260724.analysis.json \
  --bootstrap-iterations 1000 \
  --bootstrap-seed 20260724 \
  --prefix-run-id mesorch_epoch98_mouse_canonical_v1_smoke5_20260724 \
  --prefix-results-dir results/opensource/mesorch
```

## 15. Bottom line

Mesorch is not uniformly weak. Its hybrid frequency/local/global design
produces some of the best individual masks seen outside CAT-Net and TruFor,
and its largest edit quintile is clearly informative. But on the canonical
tiny-mouse condition, a typical image remains essentially undetected:
median AP is `0.004910`, median F1 is zero, and only 34/275 forged images
receive any true-positive pixel at the official threshold.

For the CLAIMFORGE main table, Mesorch should be reported as **T2 only,
fourth of eight completed local methods by native macro AP, with strong
edit-size dependence and a narrow high-quality success tail**.
