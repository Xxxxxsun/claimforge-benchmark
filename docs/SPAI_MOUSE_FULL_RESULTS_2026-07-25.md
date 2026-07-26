# SPAI on the canonical Mouse set (2026-07-25)

## 1. Status and headline

SPAI has completed the frozen **local-splice** condition on all 275 matched
Mouse tasks, or 550 canonical JPEG images. The run contains 550/550 valid
images, 275/275 complete real/forged pairs, zero errors, zero missing images,
and exactly one physical successful result row per image.

The released whole-image fake probability does not discriminate these small
local insertions:

- AUROC is **0.497931**, with a 1,000-task-pair bootstrap 95% percentile CI
  of **[0.495543, 0.499836]**;
- average precision is **0.500215**, CI **[0.499264, 0.506572]**;
- TPR at a real-only 5% FPR operating point is **4.7273%**, CI
  **[3.2727%, 6.1818%]**;
- the released strict rule `probability > 0.5` detects **46/275 forged
  images** while falsely flagging **48/275 real images**; and
- the resulting confusion matrix is **TP=46, FP=48, FN=229, TN=227**.

The matched controls make the threshold result still clearer. Every one of
the 46 detected forged images belongs to a pair whose real image is also
positive. There are **zero** pairs in which the forged image is positive and
its matched real control is negative. Two pairs have the reverse outcome:
real positive and forged negative. The other 227 pairs are negative for both
images.

Forged scores exceed their matched real scores in 111 pairs, fall below them
in 145, and tie in 19. The mean forged-minus-real probability difference is
`-0.003603`, CI `[-0.006439, -0.001193]`. This is a small anti-directional
shift, not useful detection.

SPAI is a whole-image binary classifier. Its spectral-context attention
weights indicate which classifier patches contributed to the image-level
decision; they are not manipulation probabilities or a native dense
prediction. Consequently, this run is valid for **T1 whole-image detection
only**. **T2 localization and the joint T1/T2 score are N/A**, and no
attention visualization is counted as localization.

This completes SPAI only for CLAIMFORGE's local-splice condition. It does
**not** reproduce the paper's aggregate evaluation and does **not** complete
the method's intended fully synthetic-image condition. The same-domain fully
synthetic lodging/restaurant contrast is still pending.

## 2. Pinned paper, source, checkpoint, runtime, and license

SPAI is described in the CVPR 2025 paper
[“Any-Resolution AI-Generated Image Detection by Spectral
Learning”](https://openaccess.thecvf.com/content/CVPR2025/html/Karageorgiou_Any-Resolution_AI-Generated_Image_Detection_by_Spectral_Learning_CVPR_2025_paper.html).
The authors also provide an
[official project page](https://mever-team.github.io/spai/) and the
[official repository](https://github.com/mever-team/spai).

All executable assets and the Mouse protocol were frozen before any Mouse
model score was inspected:

| Asset | Frozen revision / identity | Size / structure | SHA-256 or role |
|---|---|---:|---|
| [Official GitHub source](https://github.com/mever-team/spai/tree/8ff7b3b6779b4fcb43cf313471d9cb1c62d129a4) | `8ff7b3b6779b4fcb43cf313471d9cb1c62d129a4` | 20 hashed executable/provenance files | Current released inference implementation |
| [Official checkpoint](https://drive.google.com/file/d/1vvXmZqs6TVJdj8iF1oJ4L_fcgdQrp_YI/view?usp=sharing) | Google Drive file `1vvXmZqs6TVJdj8iF1oJ4L_fcgdQrp_YI`, `spai.pth` | 934,865,338 bytes; 324 state tensors; 139,945,243 state elements | `24159f27d7c8c2cd0cb6c4019189eb89ad0874a0d9d15f8dc9afd39ca9648a55` |
| Checkpoint tensor schema | 323 FP32 tensors and one int64 frequency mask | Strictly covers the constructed network | `ffe751246ec65936d5583a1db62bf617697484e6185f1bfad7c678f1dad36ef8` |
| Runtime | Python 3.12.3, PyTorch `2.8.0.dev20250627+cu128`, torchvision `0.23.0.dev20250627+cu128`, timm `0.4.12` | NVIDIA L20Z / CUDA 12.8 | Actual module paths and hashes recorded in the manifest |

The checkpoint is a PyTorch container rather than a pickle-free format. It
was loaded using `torch.load(map_location="cpu", weights_only=True)` with
only `yacs.config.CfgNode` added to the safe-global allowlist. Unrestricted
pickle execution was never used. The adapter verifies the complete
per-tensor schema and hashes, then strictly loads the state with no missing or
unexpected keys and checks that loaded tensors exactly equal the audited
payload.

The current README links one trained checkpoint, so no alternative checkpoint
or epoch was selected using Mouse performance. Network access is disabled
during model construction and inference.

The pinned repository carries an
[Apache-2.0 license](https://github.com/mever-team/spai/blob/8ff7b3b6779b4fcb43cf313471d9cb1c62d129a4/LICENSE),
and its README explicitly states that the project's source code and model
weights are released under Apache 2.0. That records the official release
terms for these assets; it is not a legal audit of third-party code or any
training/evaluation images, which retain their own terms.

## 3. What SPAI does and why it is strong in its intended setting

The paper starts from a useful generalization hypothesis: generator-specific
artifacts change rapidly, whereas the spectral distribution of real images
is a more stable reference. It therefore models real-image spectral behavior
through self-supervised masked spectral learning and treats generated images
as out-of-distribution samples.

For each image patch, the released implementation:

1. computes a two-dimensional FFT and separates the spectrum with a circular
   frequency mask into a filtered component and its residual;
2. transforms the original, filtered, and residual images with a ViT-B/16;
3. projects intermediate representations from all 12 transformer blocks;
4. computes cosine similarities for original/filtered,
   original/residual, and filtered/residual representations, retaining the
   token-wise mean and standard deviation for each layer;
5. combines those spectral-reconstruction-similarity statistics with an
   original-image spectral context representation, producing one
   1,096-dimensional vector per 224-pixel image patch; and
6. applies 12-head spectral-context attention across all image patches,
   followed by LayerNorm and the complete three-linear-layer MLP
   classification head.

This design has two attractive properties for fully generated images:

- it asks whether an image follows a learned model of real spectral
  structure, instead of requiring one fixed fingerprint from every future
  generator; and
- spectral-context attention aggregates a variable number of patch features,
  so the classifier can use native-resolution evidence rather than reducing
  every image to one small fixed crop.

The CVPR paper reports a 5.5 percentage-point absolute AUC improvement over
the prior state of the art across 13 recent generative approaches, together
with robustness tests for common online perturbations. Those paper figures
are context only. The present run uses the current released source and sole
checkpoint on CLAIMFORGE's locally edited JPEGs; it is neither a
reproduction of the paper's 13-generator aggregate nor a test of the paper's
full training procedure.

## 4. Frozen released inference protocol

| Component | Frozen value |
|---|---|
| Dataset | `claimforge-mouse-good275-canonical-jpeg-q95-v1` |
| Coverage | 275 matched tasks; 550 canonical JPEG images |
| Model | Official `PatchBasedMFViT`, ViT-B/16 frequency-restoration path |
| Decode | `PIL.Image.open(...).convert("RGB")`; no EXIF transpose or ICC conversion |
| Small-image pad | Center `PadIfNeeded(224,224)`, OpenCV `BORDER_REFLECT_101` |
| Resize / crop | None before patch extraction |
| Tensor conversion | Albumentations 1.4.14-equivalent uint8-to-float32 lookup/multiplication into `[0,1]`; mean 0, std 1 |
| Patch extraction | Non-overlapping 224×224 patches, stride 224 |
| Few-patch fallback | If the regular grid has fewer than 4 patches, torchvision five-crop at 224 |
| Patch feature chunk | At most 400 patches per encoder call |
| Patch aggregation | 12-head spectral-context attention |
| Classifier feature | 1,096 dimensions after SCA and LayerNorm |
| Inference | Batch 1 image, `eval()`, float32, no autocast, no TF32 |
| Primary score | Float32 `sigmoid(raw_logit)`, higher means fake |
| Released decision | Strict `probability > 0.5` |
| 5% FPR point | Real-only 95th percentile, `method="higher"`, strict `>` |
| Bootstrap | 1,000 complete-task-pair resamples, seed `20260724` |
| T2 | N/A; classifier attention is diagnostic only |

The adapter was checked against the official Albumentations transform on a
padded `17×13` synthetic case and an unpadded `449×231` case. The resulting
float32 tensors match exactly. None of the 550 Mouse images requires the
small-image pad; 524 images use the regular patch grid and 26 images, or 13
matched pairs, use five-crop.

The runtime is deterministic: seed 0, deterministic algorithms enabled,
cuDNN deterministic mode, cuDNN benchmarking disabled,
`CUBLAS_WORKSPACE_CONFIG=:4096:8`, matmul precision `highest`, and CUDA and
cuDNN TF32 disabled. This explicitly overrides the host image's initial
`high`/TF32 settings.

### Current inference config versus the checkpoint's historical config

The checkpoint embeds a historical training configuration with
`MODEL.PATCH_VIT.MINIMUM_PATCHES=1`. The current checked-out
`configs/spai.yaml` and the README's official evaluation command specify a
minimum of **4** patches. The benchmark follows the current released
inference path and does not restore the stale embedded value.

This choice was frozen before Mouse scoring. It is reported because it
changes the fallback geometry for images whose regular grid contains one to
three patches; it must not be treated as a post-hoc ablation or an
alternative selected using this result.

## 5. Non-Mouse executable golden and the website-display boundary

Before Mouse inference, the current release was exercised on two original
files from the authors' official 3.72 GB evaluation bundle. For each case,
the official transform/model forward and the benchmark adapter forward were
run twice in the frozen runtime. Their patch features, final features,
attention arrays, logits, and probabilities were bit-identical across
repeats.

| Official-bundle original | Native size | Patches | Current released raw logit | Current released probability | Project-page display |
|---|---:|---:|---:|---:|---:|
| Midjourney v6.1 `224.png` | 1232×928 | 20 | 0.9909347296 | 0.7292724848 | 0.748 |
| Stable Diffusion 3 `000001046_4.webp` | 1024×1024 | 16 | 1.6814128160 | 0.8430914879 | 0.87 |

The current-release probabilities do not fall inside the rounding intervals
implied by the project page's displayed values. This remains true with the
original evaluation files rather than the webpage's compressed display
derivatives. The page does not publish a checkpoint hash or full-precision
reference value.

Accordingly, the website numbers are retained as stale or approximate release
evidence, not used to tune preprocessing, substitute an unpinned checkpoint,
or block the Mouse run. The two deterministic full-precision values above
are implementation-regression references for the sole current source and
checkpoint. Passing this gate establishes adapter equivalence to the current
released executable; it does not claim exact reproduction of every
historical webpage display.

## 6. Patch coverage and edit visibility

“Any resolution” does not mean that every native-resolution pixel reaches
the classifier. For the regular grid, `torch.Tensor.unfold` takes complete
224×224 patches from the top-left and discards non-divisible right and bottom
remainders. Its covered rectangle is:

```text
[0, 0, floor(width / 224) * 224, floor(height / 224) * 224]
```

For the five-crop fallback, visibility is measured against the union of the
five official crop boxes. Before any score was inspected, each forged
image's exact-difference positive pixels were intersected with the relevant
union of classifier receptive fields:

| Domain | Full | Partial | None | Total |
|---|---:|---:|---:|---:|
| Lodging | 132 | 3 | 12 | 147 |
| Restaurant | 111 | 11 | 6 | 128 |
| **All** | **243** | **14** | **18** | **275** |

Mean retained GT fraction is `0.9096355`; median is `1.0`. All 13 five-crop
pairs are fully visible. `edit_visibility` is an input-condition stratum
copied to both images in a matched pair. It is not a SPAI prediction.

Stored artifact equality confirms the geometry:

| Visibility | Pairs | Equal per-patch features | Equal SCA features | Equal attention arrays | Equal probabilities |
|---|---:|---:|---:|---:|---:|
| Full | 243 | 0 | 0 | 0 | 0 |
| Partial | 14 | 0 | 0 | 0 | 1 |
| None | 18 | 18 | 18 | 18 | 18 |

All 18 `none` pairs give SPAI exactly the same effective patch tensors and
must tie. Edge dropping is therefore a measured source of unavoidable
failure for those pairs. It does not explain the overall result: even among
the 243 fully visible edits, AUROC is only `0.497858` and the mean paired
score shift is negative.

## 7. Primary whole-image result

Brackets below are 95% percentile intervals from 1,000 resamples of complete
`task_id` pairs.

| Metric | Estimate | 95% CI |
|---|---:|---:|
| AUROC | 0.497931 | [0.495543, 0.499836] |
| Average precision | 0.500215 | [0.499264, 0.506572] |
| TPR @ target FPR 5% | 0.047273 | [0.032727, 0.061818] |
| Real-only 5% FPR threshold | 0.935157 | [0.838069, 0.992166] |
| Actual FPR at that threshold | 0.047273 | [0.036364, 0.047273] |
| Accuracy @ 0.5 | 0.496364 | [0.490909, 0.500000] |
| Balanced accuracy @ 0.5 | 0.496364 | [0.490909, 0.500000] |
| Precision @ 0.5 | 0.489362 | [0.470588, 0.500000] |
| Recall @ 0.5 | 0.167273 | [0.123636, 0.210909] |
| F1 @ 0.5 | 0.249322 | [0.196532, 0.296675] |
| Specificity @ 0.5 | 0.825455 | [0.781727, 0.869091] |
| Strict paired ranking | 0.403636 | [0.341818, 0.461818] |
| Mean forged-real probability delta | -0.003603 | [-0.006439, -0.001193] |

| Operating point | TP | FP | FN | TN |
|---|---:|---:|---:|---:|
| Released `probability > 0.5` | 46 | 48 | 229 | 227 |

The real-only 5% FPR rule flags 13 real and 13 forged images. Thus its
empirical TPR and FPR are both 13/275, and these are the same 13 matched
pairs: this operating point also produces zero forged-positive,
real-negative pairs.

Probability distributions are extremely skewed:

| Kind | Min | Mean | Median | P05 | P95 | Max |
|---|---:|---:|---:|---:|---:|---:|
| Forged | 0.0 | 0.172315 | 0.000110 | `3.630e-22` | 0.920665 | 0.9999987 |
| Real | 0.0 | 0.175919 | 0.000120 | `1.826e-22` | 0.921173 | 0.9999990 |

There are two exact float32 probability zeros and 526 unique probabilities
among 550 images. The official probability remains the sole primary score;
raw logits are not introduced post hoc to remove the two underflow ties.

Within matched pairs, real and forged probabilities have Pearson correlation
`0.9973003` and Spearman correlation `0.9988810`. Scene-level score variation
therefore dominates the edit-associated difference. The 19 exact score ties
are the 18 `none` pairs plus one partially visible pair.

## 8. Domain and visibility strata

### Domain strata

| Domain | Pairs | AUROC [95% CI] | AP [95% CI] | TPR@5% FPR [95% CI] | TP / FP | Paired W / L / T |
|---|---:|---:|---:|---:|---:|---:|
| Lodging | 147 | 0.494956 [0.490745, 0.498219] | 0.500030 [0.498297, 0.511396] | 0.047619 [0.034014, 0.068027] | 39 / 41 | 56 / 79 / 12 |
| Restaurant | 128 | 0.499725 [0.495727, 0.502901] | 0.503605 [0.501195, 0.517303] | 0.054688 [0.031250, 0.070312] | 7 / 7 | 55 / 66 / 7 |

| Domain | Strict paired ranking [95% CI] | Mean paired delta [95% CI] | Exact sign-test p |
|---|---:|---:|---:|
| Lodging | 0.380952 [0.299320, 0.455782] | -0.005548 [-0.010343, -0.001521] | 0.057894 |
| Restaurant | 0.429688 [0.343750, 0.515625] | -0.001370 [-0.004167, 0.000397] | 0.363363 |

Neither domain is a useful standalone detector in this condition. Lodging
shows the larger negative paired shift, but both whole-image AUROCs remain
near 0.5.

### Visibility strata

| Visibility | Pairs | AUROC [95% CI] | AP [95% CI] | TPR@5% FPR [95% CI] | TP / FP | Paired W / L / T |
|---|---:|---:|---:|---:|---:|---:|
| Full | 243 | 0.497858 [0.495029, 0.499890] | 0.500707 [0.499609, 0.507976] | 0.057613 [0.041152, 0.065844] | 41 / 43 | 106 / 137 / 0 |
| Partial | 14 | 0.492347 [0.431122, 0.540816] | 0.520662 [0.503699, 0.617273] | 0.000000 [0.000000, 0.214286] | 1 / 1 | 5 / 8 / 1 |
| None | 18 | 0.500000 [0.500000, 0.500000] | 0.500000 [0.500000, 0.500000] | 0.000000 [0.000000, 0.000000] | 4 / 4 | 0 / 0 / 18 |

| Visibility | Strict paired ranking [95% CI] | Mean paired delta [95% CI] | Exact sign-test p |
|---|---:|---:|---:|
| Full | 0.436214 [0.378498, 0.497942] | -0.004060 [-0.007151, -0.001254] | 0.054068 |
| Partial | 0.357143 [0.142857, 0.642857] | -0.000302 [-0.002704, 0.002022] | 0.581055 |
| None | 0.000000 [0.000000, 0.000000] | 0.000000 [0.000000, 0.000000] | 1.000000 |

The all-pair exact sign test excludes 19 ties and gives `p=0.038952` for 111
wins versus 145 losses. This is evidence of a small shift in the wrong
direction, not evidence of successful detection. A deployed detector also
does not receive the matched real counterfactual, so paired subtraction
cannot replace the image-only AUROC, AP, or operating-point results.

## 9. Determinism and runtime

Two separate final-code CUDA smokes each processed the same five complete
pairs. Their ten rows match exactly for decoded/tensor hashes, patch-feature
arrays, spectral-context attention arrays, 1,096-dimensional final features,
raw logits, float32 probabilities, strict decisions, and summary metrics.
The two smoke configuration fingerprints are identical; only expected run
identities, timestamps, paths, and timing measurements differ.

The two non-Mouse golden cases were also bit-identical across their two
repeated complete forwards.

The full run used one NVIDIA L20Z, batch size one, deterministic float32, no
autocast, no TF32, and no network access.

| Runtime quantity | Value |
|---|---:|
| Model-forward latency, mean | 125.715 ms/image |
| Model-forward latency, median | 153.131 ms/image |
| Model-forward latency, P95 | 190.373 ms/image |
| Model-forward latency, max | 626.108 ms/image |
| Peak allocated CUDA memory | 4,506,983,424 bytes (4.198 GiB) |
| Full runner wall time | 175.690 s |
| End-to-end runner throughput | 3.131 images/s |

Wall time also includes decoding, preprocessing and hashing, three NumPy
artifact writes and readbacks per image, JSONL persistence, and final metric
and bootstrap computation. The repository recommends Python 3.11 and
PyTorch/CUDA 12.4 installation in general; this report records the exact
Python 3.12 / CUDA 12.8 execution environment instead of claiming an
identical training environment.

## 10. Independent audit

Audit status is **`ok`**. The independent analyzer validated the pinned clean
source, checkpoint bytes and tensor schema, safe loader, strict full-state
construction, runtime, dataset ledger, output files, and rejection of T2
fields.

For every one of the 550 selected images, it independently:

- reopened the canonical image with Pillow and reconstructed native
  preprocessing and grid/five-crop patch selection;
- executed a fresh complete FFT, ViT, spectral-reconstruction-similarity,
  spectral-context-attention, LayerNorm, and MLP forward;
- compared fresh per-patch features, SCA attention, final features, logit,
  probability, and strict decision with the persisted run;
- replayed SCA/LayerNorm/MLP from persisted per-patch features and replayed
  the complete MLP from the persisted final feature; and
- recomputed coverage, score distributions, threshold metrics, paired
  results, domain/visibility strata, and all 1,000 pair-bootstrap intervals.

Maximum absolute replay differences are:

| Replayed quantity | Maximum absolute difference | Audit tolerance |
|---|---:|---:|
| Per-patch 1,096-dimensional feature | 0.0 | `1e-5` |
| Final 1,096-dimensional feature | 0.0 | `1e-5` |
| Spectral-context attention | 0.0 | `1e-6` |
| Raw logit | 0.0 | `1e-5` |
| Float32 probability | `5.960464477539063e-08` | `1e-7` |
| Artifact SCA replay feature | 0.0 | — |
| Artifact SCA replay attention | 0.0 | — |
| Artifact SCA/MLP and feature/MLP logits | 0.0 | — |

The complete independently recomputed summary matches the stored summary
within the frozen probability tolerance. Exact feature and logit agreement
plus the one-float32-ULP-scale probability bound establishes faithful
execution of the frozen release. It does not establish generalization beyond
this protocol.

## 11. Scope, limitations, and conclusion

The supported local-splice claim is:

> The pinned current SPAI release does not transfer to CLAIMFORGE Mouse's
> small local insertions as a standalone whole-image detector. Its released
> probability has AUROC 0.4979, the real-only 5% FPR point has 4.73% TPR,
> and the released threshold finds no pair in which the forged image is
> positive while the matched real control remains negative.

Important limits are:

- this is a local-splice stress test, whereas SPAI is principally motivated
  and reported as an AI-generated whole-image detector;
- the present result is not the CVPR paper's 13-generator aggregate and does
  not reproduce its training;
- 18/275 edits lie entirely outside the released patch receptive fields,
  although the 243 fully visible pairs also remain ineffective;
- the current released inference config uses minimum four patches, while the
  checkpoint embeds a historical value of one; only the current behavior was
  run, as frozen before Mouse scores;
- the result covers the canonical JPEG-Q95 construction, two domains, one
  pinned source revision, and the sole current official checkpoint, not every
  compression, edit size, generator, release, or deployment transform;
- no Mouse labels were used to tune the model, choose a checkpoint, repair
  the project-page display mismatch, or select an oracle threshold;
- SCA attention is classifier diagnostic evidence, not T2 localization; and
- independent replay proves execution fidelity, not external validity.

The next local-splice method is **B-Free**, whose bias-controlled training and
local-inpainting construction are especially relevant to the Mouse threat
model. SPAI brings the preferred local-splice suite to 5/10 and the minimum
mechanism-complete suite to 5/7. However, all whole-image methods still need
the same-domain fully synthetic contrast; until that condition exists, SPAI
is **local-splice complete but contrast incomplete**.

## 12. Reproducibility artifacts

Primary run:

```text
results/opensource/spai/
  spai_any_resolution_spectral_mouse_canonical_v1_full275_20260725/
```

Important artifacts are:

- `manifest.json` — frozen source, checkpoint, model construction,
  preprocessing, golden, runtime, dataset, metric, license, and output
  contracts;
- `expected_inputs.jsonl` — exact ordered 550-image input ledger;
- `results.jsonl` — one physical successful T1 row per canonical image;
- `artifacts/*.patch_features.npy` — 550 variable-length arrays of
  per-patch 1,096-dimensional spectral features;
- `artifacts/*.feature.npy` — 550 post-SCA/LayerNorm classifier inputs;
- `artifacts/*.attention.npy` — 550 classifier-diagnostic SCA arrays, never
  T2 masks;
- `summary.json` — complete probability metrics, paired bootstrap, domain
  strata, and visibility strata; and
- `independent_audit.json` — independent 550-image full-model and artifact
  replay plus metric recomputation.

The reusable implementation entry points are:

```text
eval/opensource/run_spai.py
eval/opensource/spai_metrics.py
eval/opensource/analyze_spai_run.py
tests/test_run_spai.py
tests/test_spai_metrics.py
tests/test_analyze_spai_run.py
```

The frozen pipeline can be rerun from the repository root with the pinned
local source, checkpoint, official-bundle originals, and a new immutable run
ID:

```bash
python -m eval.opensource.run_spai \
  --run-id <new-immutable-run-id> \
  --device cuda:0 \
  --bootstrap-samples 1000 \
  --bootstrap-seed 20260724 \
  --fail-fast
```

The physically independent audit is reproduced with:

```bash
python -m eval.opensource.analyze_spai_run \
  --run-id spai_any_resolution_spectral_mouse_canonical_v1_full275_20260725
```

For new canonical data, use a new immutable run ID and dataset manifest.
Successful rows are resumable, while changes to source, checkpoint, adapter,
runtime, input selection, or metric contracts are rejected instead of being
silently mixed.
