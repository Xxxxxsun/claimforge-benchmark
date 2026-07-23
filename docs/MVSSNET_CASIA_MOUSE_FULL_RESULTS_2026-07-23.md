# MVSS-Net (CASIAv2) on the canonical mouse set (2026-07-23)

## 1. Status and headline

The fourth publicly released research baseline is complete. All 275 paired
tasks, or 550 images, finished successfully with no errors.

The official CASIAv2-trained MVSS-Net does not transfer successfully to the
CLAIMFORGE mouse task:

- primary image-level AUROC is **0.50576**, with paired-task bootstrap 95% CI
  **[0.49466, 0.51684]**;
- native-resolution macro pixel AP is **0.03415**;
- macro pixel F1/IoU at the released 0.5 threshold are
  **0.01962/0.01248**;
- micro pixel F1/IoU are **0.00735/0.00369**; and
- only **4/275** forged masks overlap the recorded edit box at IoU greater
  than 0.3.

This is a genuine zero-shot failure, not an incomplete run or a silent
preprocessing mismatch. The independent analyzer validated 3,025
canonical-image, ground-truth, and artifact files and reproduced the complete
BGR decode, normalization, raw-logit, quantization, native-resize, score, mask,
and metric chain.

The conclusion must remain checkpoint- and task-specific: it applies to the
official original MVSS-Net CASIAv2 checkpoint on tiny diffusion-generated
splice-back edits. It is not a claim that every MVSS-Net variant fails on every
manipulation dataset.

## 2. Pinned method and checkpoint

The run uses the original authors'
[MVSS-Net repository](https://github.com/dong03/MVSS-Net) at commit
`cc2aed77a823723015f95e4a6a3e344f3ddb7ccc` and the original
[ICCV 2021 method](https://openaccess.thecvf.com/content/ICCV2021/html/Chen_Image_Manipulation_Detection_by_Multi-View_Multi-Scale_Supervision_ICCV_2021_paper.html).

The exact released checkpoint is:

- `mvssnet_casia.pt`, which is the checkpoint selected by the authors'
  `do_pred.sh`;
- official Google Drive file ID
  `1MHoe91a24GiBMG2JYoghPRDd4Ro6RIVq`;
- 588,270,735 bytes;
- SHA-256
  `080bc6c3aae59f748b547dbf090786fe9d31a6e50749daaa40871e298d6a7e50`;
- a raw `OrderedDict` containing 800 tensor entries and 146,994,922 state
  elements; and
- 146,880,335 model parameters plus 114,587 buffer elements.

The checkpoint loaded with `torch.load(weights_only=True)` and
`load_state_dict(..., strict=True)`. There were zero missing, unexpected, or
shape-mismatched entries. Unrestricted pickle loading was never used.

The official constructor tries to download ImageNet ResNet-50 weights twice
even though the final checkpoint covers the complete state. The adapter
suppresses those unpinned downloads during construction and then performs the
complete strict load. This changes no final parameter or buffer.

## 3. Why the original CASIAv2 model was selected

The repository releases original MVSS-Net checkpoints trained on CASIAv2 and
DEFACTO-84k. The authors' own inference script defaults to the CASIAv2 model,
and the released table shows that version doing better on three of four listed
external test sets. It is therefore the least arbitrary primary zero-shot
choice.

MVSS-Net++ was not silently substituted. Although the same Drive exposes
files named as Plus checkpoints, the public repository does not contain the
Plus-specific ConvGeM image-classification implementation described by the
extended paper. Passing a Plus file through the original inference path would
not reproduce the complete advertised method. A future Plus run must be named
and justified as a separate variant.

The IMDL-BenCo wrapper was also not used for this primary run. Its PIL/RGB
input differs from the original repository's OpenCV/BGR checkpoint protocol,
and its checkpoint is a separately retrained benchmark asset. It can be
evaluated later as a distinct BenCo variant.

## 4. Exact inference contract

The adapter reproduces the original source's unusual but load-bearing input
and output order:

1. Open the canonical JPEG with `cv2.imread(..., IMREAD_COLOR)`, retaining
   BGR channel order.
2. Stretch the image to exactly 512 by 512 with OpenCV `INTER_LINEAR`.
3. Divide by 255, convert HWC to CHW, and apply ImageNet mean
   `[0.485, 0.456, 0.406]` and standard deviation
   `[0.229, 0.224, 0.225]` in the existing BGR order.
4. Run the original `resnet50`, `sobel=True`, `constrain=True`, one-class
   network in float32.
5. Discard the auxiliary edge-supervision output, as the official inference
   code does.
6. Apply sigmoid to the one-channel 512-by-512 segmentation logits.
7. Multiply the probability map by 255 and cast directly to uint8, reproducing
   the truncation used by Torchvision 0.6.1 `ToPILImage`.
8. Resize that **uint8** map to the original image dimensions with OpenCV
   `INTER_LINEAR`, then save it as grayscale PNG.
9. Evaluate native T2 on `saved_uint8 / 255`, with the official strict
   comparison `score > 0.5`.

The order in steps 7 and 8 is intentional. Quantizing before native resizing is
not equivalent to resizing a float map and quantizing afterward.

The original 2021 script wrapped inference with NVIDIA Apex O1. The current
environment does not provide a compatible NVIDIA `apex.amp` implementation,
and a generic PyTorch autocast run would not be an exact substitute. The
audited primary run therefore uses deterministic float32 and records this
deviation explicitly. It preserves the model, preprocessing, output semantics,
and thresholds, but should not be described as bit-exact to the authors'
CUDA 10.1/Apex environment.

The official Bayar constrained convolution also normalizes its kernel in place
on every forward. This creates tiny within-process floating-point drift even
though the operation is mathematically idempotent. CLAIMFORGE fixes input
order, starts from the same checkpoint, and replays a completed prefix before
any partial resume. Independently started smoke and full runs match bit for bit
over their common ordered prefix.

## 5. What MVSS-Net is designed to detect

MVSS-Net combines several forensic views:

- an RGB ResNet stream;
- a constrained Bayar residual stream intended to suppress semantic content
  and expose local noise inconsistencies;
- a Sobel-guided, multi-scale edge branch intended to learn manipulation
  boundaries; and
- position/channel attention that fuses the views into a pixel mask.

Training applies edge-, pixel-, and image-level supervision. The method does
not contain a separate image-classification head. Its paper defines the
image-level probability using global maximum pooling over the segmentation
probability map.

That design explains both the original motivation and the present failure
mode. CLAIMFORGE edits occupy a median 0.1126% of the original image and are
stretched into a 512-by-512 square. Global max pooling needs only one of
262,144 pixels to react to an ordinary scene edge, compression pattern, lamp,
table, or high-frequency texture in order to produce a high image score.
Meanwhile, a well-blended diffusion insertion need not reproduce the
copy-move, classical splice, or inpainting boundaries represented in CASIAv2.
These are evidence-consistent explanations, not causal ablations.

## 6. Two official image-score paths

There are two closely related score paths in the authors' code:

1. `inference_single()` returns the continuous maximum of the 512-by-512
   sigmoid map. This matches the paper's GMP definition and is the primary T1
   score in CLAIMFORGE.
2. `evaluate.py` reopens the saved native uint8 PNG and takes its maximum
   divided by 255. This is retained as the secondary
   `official_png_score`.

They lead to the same scientific conclusion:

| T1 metric | Continuous 512 GMP | Saved-native-PNG GMP |
|---|---:|---:|
| AUROC | **0.50576** | 0.50531 |
| Average precision | 0.50681 | 0.50259 |
| Accuracy at 0.5 | 0.50000 | 0.50182 |
| Balanced accuracy at 0.5 | 0.50000 | 0.50182 |
| TPR at FPR at most 5% | 0.04727 | 0.03636 |

Only 1/550 image decisions differs between the two score paths. Native PNG
quantization is therefore not the explanation for the poor T1 result.

## 7. Primary T1 result

| Image-level metric | Estimate | Pair-bootstrap 95% CI |
|---|---:|---:|
| AUROC | **0.50576** | [0.49466, 0.51684] |
| Average precision | 0.50681 | [0.50023, 0.52089] |
| TPR at FPR at most 5% | 0.04727 | [0.03273, 0.06182] |
| Accuracy at 0.5 | 0.50000 | [0.50000, 0.50000] |
| Balanced accuracy at 0.5 | 0.50000 | [0.50000, 0.50000] |
| Standard positive-class F1 at 0.5 | 0.65582 | [0.64879, 0.66091] |
| Paired forged-greater-than-real rate | 0.42182 | [0.36364, 0.47636] |

The F1 value is actively misleading if read alone. At threshold 0.5 the
confusion counts are:

| TP | FP | FN | TN |
|---:|---:|---:|---:|
| 262 | 262 | 13 | 13 |

The model predicts "manipulated" for 95.3% of both classes. On a balanced
dataset, a trivial all-positive classifier already obtains standard
positive-class F1 of 2/3. MVSS-Net's 0.6558 is slightly worse than that trivial
reference, while balanced accuracy correctly remains 0.5. The authors'
unconventional sensitivity/specificity harmonic metric, derived from these
same counts, is approximately 0.0901.

Within matched pairs, the forged score is larger in 116 cases and smaller in
159, with no continuous-score ties. The two-sided exact sign-test p-value is
0.0112. This does not contradict AUROC 0.5058: the sign test considers only
each forged image against its own real counterpart, while AUROC compares all
forged scores with all real scores. The paired delta median is slightly
negative, while a few larger positive outliers make the mean delta positive
0.00563; its bootstrap CI `[-0.00013, 0.01197]` includes zero.

## 8. Primary T2 result

| Native-resolution metric | Estimate | Pair-bootstrap 95% CI |
|---|---:|---:|
| Macro pixel AP | **0.03415** | [0.02213, 0.04775] |
| Median per-image pixel AP | 0.00313 | — |
| Macro pixel F1 at 0.5 | 0.01962 | [0.01071, 0.02940] |
| Macro pixel IoU at 0.5 | 0.01248 | [0.00625, 0.01934] |
| Micro pixel F1 at 0.5 | 0.00735 | [0.00469, 0.01057] |
| Micro pixel IoU at 0.5 | 0.00369 | [0.00235, 0.00531] |
| Edit-box hits at 0.5 | 4/275 (1.45%) | — |

The median forged-image F1 is exactly zero. There are isolated successes—the
best forged image reaches AP 0.8833 and F1 0.7412—but these are rare and do not
represent stable zero-shot behavior.

Calibration is not the main problem:

- the exact best global uint8 threshold is `166/255 = 0.65098`, using strict
  `>`; it raises micro F1 only from 0.00735 to 0.00760;
- a separate oracle threshold for every forged image gives mean F1 0.05181
  and median F1 0.00947.

These oracle values use ground truth and are not deployable results. More
importantly, even the oracle upper bound is low.

The unquantized 512 model-space diagnostic is similarly weak: macro AP
0.03508 and macro F1 0.02010. Native resizing and uint8 serialization therefore
do not account for the failure.

## 9. Pristine-image behavior

At threshold 0.5, 261/275 pristine images have a nonempty mask. The pristine
and forged predicted areas are almost indistinguishable:

| Statistic | Pristine | Forged |
|---|---:|---:|
| Mean positive area | **3.2887%** | 3.1166% |
| Median positive area | 0.4960% | 0.4598% |
| Maximum positive area | 39.8725% | 39.9006% |
| Empty masks | 14/275 | 13/275 |

The pristine macro mean has 95% CI `[2.5924%, 3.9754%]`; its micro positive
fraction is 3.7562%. A nonempty MVSS-Net mask is therefore not useful evidence
that an image is manipulated.

For context, mean pristine positive area is 0.0596% for MaskCLIP, 0.4359% for
CAT-Net v2, and 1.7764% for TruFor. MVSS-Net fires much more broadly than all
three on this set.

## 10. Diagnostic slices

### Domain

| Domain | Pairs | T1 AUROC | Macro pixel AP | Macro F1@0.5 | Real positive area |
|---|---:|---:|---:|---:|---:|
| lodging | 147 | 0.51094 | 0.05090 | 0.03178 | 4.2418% |
| restaurant | 128 | 0.49542 | 0.01490 | 0.00565 | 2.1941% |

Lodging has higher localization point estimates but also roughly twice the
pristine positive area. These are descriptive slices; no direct between-domain
contrast was pre-registered, so they should not be called statistically
significant differences.

### Edit-size quintiles

| Quintile | Pairs | Median edit fraction | Macro pixel AP | Macro F1@0.5 |
|---|---:|---:|---:|---:|
| Q1, smallest | 55 | 0.05347% | 0.00657 | 0.00033 |
| Q2 | 55 | 0.07780% | 0.01913 | 0.01534 |
| Q3 | 55 | 0.11264% | 0.01933 | 0.01423 |
| Q4 | 55 | 0.16293% | 0.04264 | 0.02417 |
| Q5, largest | 55 | 0.37441% | 0.08307 | 0.04403 |

Localization improves descriptively as the edit grows, but even the largest
quintile remains weak. T1 AUROC stays approximately 0.499 to 0.511 across the
five slices. This association is not a causal size ablation.

## 11. Direct comparison with completed research methods

All methods below use the same 275 paired canonical inputs and native exact
difference ground truth. MVSS-Net uses the original method's strict `> 0.5`;
the other methods retain their own released class semantics at threshold 0.5.

| Metric | MaskCLIP | TruFor | CAT-Net v2 | MVSS-Net |
|---|---:|---:|---:|---:|
| Native T1 available | yes | yes | no | yes, map GMP |
| T1 AUROC | 0.50728 | **0.81790** | — | 0.50576 |
| Macro pixel AP | 0.04740 | 0.57989 | **0.61234** | 0.03415 |
| Macro pixel F1@0.5 | 0.00564 | **0.50010** | 0.47481 | 0.01962 |
| Macro pixel IoU@0.5 | 0.00367 | **0.37410** | 0.35743 | 0.01248 |
| Micro pixel IoU@0.5 | 0.00312 | **0.19800** | 0.18138 | 0.00369 |
| Edit-box hits@0.5 | 1/275 | **83/275** | 65/275 | 4/275 |
| Mean pristine positive area | **0.0596%** | 1.7764% | 0.4359% | 3.2887% |

Aligned paired-task bootstrap differences support the main comparison:

- MVSS minus MaskCLIP macro AP is -0.01325, CI
  `[-0.02931, 0.00276]`; there is no clear AP difference.
- MVSS minus MaskCLIP macro F1 is +0.01398, CI
  `[0.00480, 0.02463]`, but pristine positive area also increases by
  3.229 percentage points, CI `[2.578, 4.000]`. The F1 improvement comes with
  much broader firing and should not be called overall superiority.
- MVSS minus TruFor macro AP is -0.54574, CI
  `[-0.57626, -0.51381]`.
- MVSS minus CAT-Net macro AP is -0.57819, CI
  `[-0.60862, -0.54693]`.

TruFor remains the strongest complete T1+T2 reference. CAT-Net remains the
strongest continuous T2 result. MVSS-Net and MaskCLIP are both near chance for
T1 and far below the two forensic localizers for T2.

## 12. Validation and audit

The final result is backed by the following checks:

- The official checkpoint safely loaded all 800 tensors with `strict=True`.
- A fixed five-pair smoke run completed and passed audit before the full run.
- The 10 overlapping smoke/full images are bit-exact across raw logits,
  512 score maps, native uint8 maps, binary masks, both T1 paths,
  preprocessing metadata, and both localization spaces: 140/140 comparisons
  match.
- The full analyzer re-hashed 3,025 files: all 550 canonical JPEGs, 2,200
  per-image artifacts, and 275 forged ground-truth masks.
- It independently reproduced OpenCV BGR decode, 512 resize, BGR-order
  normalization, sigmoid maps, uint8 truncation, native uint8 interpolation,
  both GMP scores, strict decisions, masks, and all T2 counts and metrics.
- All 550 physical result rows are unique, successful, and tied to the
  immutable run-manifest fingerprint.
- The repository test suite reports 82 passed and 4 skipped; the MVSS-Net
  runner, metric, and analyzer tests all pass.
- `git diff --check` passes.

The full artifact directory contains 2,200 files and 1,210,518,448 bytes. The
median model-forward latency after warm-up is 9.41 ms on an NVIDIA L20Z, and
peak allocated CUDA memory is 907,737,088 bytes. These timing values exclude
JPEG decoding, artifact serialization, and metric computation.

## 13. License status

The official repository contains no `LICENSE`, `COPYING`, or `NOTICE` file,
GitHub reports no detected license, and the weight folder gives no separate
terms. MVSS-Net should therefore be described as a **publicly released,
source-available research method**, not as MIT, OSI open source, or explicitly
cleared for commercial use or redistribution.

The present use is an internal research evaluation with attribution. Any
commercial integration or redistribution of the source or weights requires a
separate rights review or author permission.

## 14. Reproducible artifacts

Primary files:

- runner:
  `eval/opensource/run_mvssnet.py`
- metric implementation:
  `eval/opensource/mvssnet_metrics.py`
- independent analyzer:
  `eval/opensource/analyze_mvssnet_run.py`
- full append-only results:
  `results/opensource/mvssnet/mvssnet_casia_mouse_canonical_v1_full275_20260723.jsonl`
- immutable run manifest:
  `results/opensource/mvssnet/mvssnet_casia_mouse_canonical_v1_full275_20260723.run_manifest.json`
- run summary:
  `results/opensource/mvssnet/mvssnet_casia_mouse_canonical_v1_full275_20260723.summary.json`
- audited paired analysis:
  `results/opensource/mvssnet/mvssnet_casia_mouse_canonical_v1_full275_20260723.analysis.json`

Key SHA-256 digests:

| Artifact | SHA-256 |
|---|---|
| full JSONL | `57872dc4030f5b93fc5bdf6eee6c1d2203ffde846275f95478de4a0f8a07831a` |
| run manifest | `5176adfef8b950bd42cc8be89dc89d996dcdd5471ec8c3f734b063fd8515b7cb` |
| summary | `9f813029bcfbf6623d4f0bcbb46f1d8fb9e0e06829eade4db419a83a106d8d95` |
| analysis | `c2db65790e895042a713a2c34e4491af05002cd46727a5ed71a39f203117abe1` |

Commands:

```bash
PYTHONPATH=/root/claimforge-benchmark \
/root/.cache/claimforge/venvs/mvssnet-cc2aed7/bin/python \
  eval/opensource/run_mvssnet.py \
  --run-id mvssnet_casia_mouse_canonical_v1_full275_20260723 \
  --device cuda:0 --fail-fast

PYTHONPATH=/root/claimforge-benchmark \
/root/.cache/claimforge/venvs/mvssnet-cc2aed7/bin/python \
  eval/opensource/analyze_mvssnet_run.py \
  --run-id mvssnet_casia_mouse_canonical_v1_full275_20260723 \
  --bootstrap-iterations 1000
```

## 15. Recommended next method

The next baseline should be **PSCC-Net** using its official MIT-licensed source
and committed checkpoint. It supplies both image-level and pixel-level outputs
through a progressive spatio-channel correlation design, providing a useful
test of whether the failure is specific to MVSS-Net's edge/noise/GMP design
rather than a general limitation of pre-diffusion manipulation localizers.
