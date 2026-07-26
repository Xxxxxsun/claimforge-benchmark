# CLAIMFORGE Open-Source Baseline Execution Plan

**Frozen:** 2026-07-24
**Status updated:** 2026-07-25
**Scope:** zero-shot, public-research baselines with author-released code and
weights (or a genuinely training-free official implementation).

This document is the execution source of truth for the two open-source
forensics tracks.  It separates:

- **Local manipulation detection/localization:** methods designed to detect or
  localize a manipulated region.  Native image-level heads are T1; native
  pixel maps are T2.
- **Whole-image AIGC detection:** methods designed primarily to classify a
  fully synthetic image.  These are T1-only controls when evaluated on
  CLAIMFORGE's small local insertions.

“Open-source” below is shorthand for publicly reproducible research code.  It
does not imply that every repository uses an OSI-approved or commercial-use
license.

## 1. Frozen benchmark protocol

- Dataset: `mouse_canonical_v1`, 275 fixed real/forged pairs, 550 canonical
  JPEG inputs.
- Input order and identity come only from
  `outputs/opensource/mouse_canonical_v1/manifest.json`.
- Main runs use the authors' off-the-shelf weights with no CLAIMFORGE
  fine-tuning.
- A checkpoint and its provenance must be registered before looking at full
  mouse results.  No post-hoc checkpoint selection is allowed.
- T1 uses only an author-defined image score or classification head.  A
  heatmap mean/max is not silently promoted to a native image score.
- T2 uses only a native dense output mapped back to input pixels.  Grad-CAM,
  attention, bounding boxes, and other adapters are diagnostic results, not
  native T2.
- Every runner must record source commit, source-file hashes, checkpoint hash,
  preprocessing, score direction, thresholds, package versions, input hashes,
  runtime, and per-image errors.
- Required stages are: one-image preflight, five-pair smoke, then all 275
  pairs.  Full runs are resumable and may not silently omit failures.
- Primary metrics are:
  - T1: AUROC, average precision, TPR at 5% FPR, fixed-threshold results when
    the author defines a threshold, paired ranking accuracy, paired score
    delta, and bootstrap confidence intervals.
  - T2: native-space and model-space pixel AP, precision, recall, F1, IoU,
    MCC, false-positive area on real images, micro/macro summaries, and
    bootstrap confidence intervals.

## 2. Track A — local manipulation detection/localization

### Completed full runs (10)

1. [x] OpenSDI / MaskCLIP
2. [x] TruFor
3. [x] CAT-Net v2
4. [x] MVSS-Net (official CASIAv2 checkpoint)
5. [x] [PSCC-Net](PSCCNET_MOUSE_FULL_RESULTS_2026-07-24.md) (official
   synthetic-pretrained checkpoint)
6. [x] [IML-ViT](IMLVIT_CAT_PROTOCOL_MOUSE_FULL_RESULTS_2026-07-24.md)
   (official CAT/TruFor-protocol checkpoint; native T2 only)
7. [x] [HiFi-IFDL](HIFI_IFDL_GENERAL750001_MOUSE_FULL_RESULTS_2026-07-24.md)
   (official general checkpoint `750001`; native T1 + T2)
8. [x] [Mesorch](MESORCH_EPOCH98_MOUSE_FULL_RESULTS_2026-07-24.md)
   (official unpruned checkpoint `mesorch-98.pth`; native T2 only)
9. [x] [RelayFormer](RELAYFORMER_CHECKPOINT164_MOUSE_FULL_RESULTS_2026-07-24.md)
   (official image-only paper checkpoint `checkpoint-164.pth`; native T2
   only)
10. [x] [DINOv3-IML](DINOV3_IML_CHECKPOINT48_MOUSE_FULL_RESULTS_2026-07-24.md)
    (official CAT ViT-L/16 LoRA-r32 `checkpoint-48.pth`; native T2 only;
    non-peer-reviewed 2026 preprint)

### Deferred local method (1)

1. [ ] **NFA-ViT / BR-Gen** — deferred on 2026-07-24 because the official
   checkpoint could not be obtained without an authenticated Baidu Netdisk
   workflow; native T1 + T2;
   training manipulations closely match localized generative editing.
   Runner, metrics, analyzer, and protocol tests are ready as of 2026-07-24.
   The 275-pair run can be resumed if the authors' `checkpoint-9999.pth`
   becomes directly available.

The local main table currently contains 10 completed methods and one explicitly
deferred method. NFA-ViT is not counted as a completed result.

### Appendix-only candidates

AdaIFL, SparseViT, SAFIRE, FOCAL, and ForensicsSAM.  SAFIRE is not in the
frozen main queue because its repository does not state a project license.
RITA remains excluded until its advertised inference components are complete.

## 3. Track B — whole-image AIGC detection

All methods in this track output an image-level score.  Their CLAIMFORGE result
answers whether a whole-image synthetic detector transfers to a small local
diffusion insertion; it does not replace a localizer.

### Frozen main-table queue (10)

1. [x] [**FSD / Forensic Self-Descriptions**](FSD_V1_2_0_MOUSE_FULL_RESULTS_2026-07-24.md)
   — local-splice condition completed 2026-07-24 with the official v1.2
   inference release: 550/550 valid images, AUROC `0.500350`, AP `0.502708`,
   and no native T2. The public inference release differs materially from the
   CVPR paper implementation, so this is labeled as a release result rather
   than a strict paper reproduction. Its required fully synthetic control
   remains pending.
2. [x] [**UniversalFakeDetect**](UNIVERSALFAKEDETECT_OURS_LC_MOUSE_FULL_RESULTS_2026-07-24.md)
   — local-splice condition completed 2026-07-24 with the official released
   `Ours LC` head and OpenAI CLIP ViT-L/14. The primary current-HEAD
   preprocessing result has 550/550 valid images, AUROC `0.499650`, AP
   `0.497293`, and no native T2. A separately frozen checkpoint-era
   `Resize(256) -> CenterCrop(224)` sensitivity run also completed 550/550
   images (AUROC `0.503260`, AP `0.510188`); it is not mixed with the primary
   result. The same-domain fully synthetic control remains pending.
3. [x] [**NPR**](NPR_AIGC_PROGAN4CLASS_MOUSE_FULL_RESULTS_2026-07-25.md)
   — neighboring-pixel and upsampling artifacts. The local-splice condition
   completed and passed independent full-model audit on 2026-07-25:
   550/550 valid images, 275/275 complete pairs, zero errors, official
   probability AUROC `0.500198`, AP `0.502661`, and 0/275 forged detections
   at the released strict `>0.5` rule. NPR has no native T2, and its
   same-domain fully synthetic control remains pending. The protocol was
   frozen before Mouse inference to the official repository's
   AIGCDetectBenchmark recipe: source commit
   `781ced3f7ca2cdc69ec9dd4ef27e8d0b3c07752a`, the explicitly linked
   ProGAN-4class `model_epoch_last_3090.pth` checkpoint (SHA-256
   `b67a91555ce786a6d0463ff0cb2b0b874d1c3f971b0e3febd2ae5618a80f7e8a`),
   Pillow RGB decoding, native resolution with no resize/crop, batch size 1,
   and removal of only the final row/column when a dimension is odd before
   the neighboring-pixel residual. The released decision is the strict rule
   `sigmoid(logit) > 0.5`; NPR has no native T2 output. The repository also
   contains `NPR.pth`, but its checkpoint-dict layout is incompatible with
   current `test.py` and it is not the checkpoint linked by the repository's
   AIGCDetectBenchmark section, so Mouse scores will not be used to choose
   between the two assets. A CUDA pair smoke performed before the full run
   found finite logits near `-170` whose official float32 sigmoid underflowed
   to exact zero. Therefore two score views are pre-registered and must both
   be reported for all 275 pairs: the released sigmoid probability remains
   the primary operational score and sole source of the `>0.5` decision,
   while raw-logit AUROC/AP, real-only 5% FPR, paired ranking/delta, and the
   same pair bootstrap are an always-on numerical diagnostic. The diagnostic
   is not a replacement score and results cannot be selected between the two.
   The official HF Space corroborates this checkpoint and native-size
   preprocessing, but omits `model.eval()` and therefore leaves BatchNorm in
   train mode; it is recorded as a deployment defect, not treated as the
   executable reference or added as a sensitivity. The main model mode is the
   official GitHub `test.py` evaluation mode.
4. [x] **Community Forensics** — broad generator-diversity supervision.
   Protocol frozen before any Mouse model score on 2026-07-25 to the paper's
   best-performing 384x384 High-res ViT-S/16 and the official repository's
   default 384 evaluation example. Source is pinned to main commit
   `ee5b71d43db0f3779e1edd64ee927b13f2dd6ad4`; single-image execution
   semantics are additionally pinned to the official `eval_single` commit
   `5e52ed690bdbd609f9bb1705c4c80d11872a05bd`. The selected public,
   ungated HF asset is `OwensLab/commfor-model-384` revision
   `6076002bf0d9dd37537f965ee2f06f826c333b61`,
   `model.safetensors`, 87,262,324 bytes, SHA-256
   `b89f36275f3bf5e2b040eee36597a8f19db051bff9a473a9cf7b2466284fb387`,
   with 152 float32 tensors and 21,811,969 parameters. The HF processor is
   pinned to revision `3540a3f0d688f8bf492a8aed48613b891f88047e`.
   The exact test transform is Pillow RGB, bilinear aspect-preserving
   short-edge `Resize(440)`, `CenterCrop(384)`, tensor in `[0,1]`, ImageNet
   normalization, and float32. Inference is batch one, eval mode, no AMP;
   the released score is float32 `sigmoid(logit)` and the official
   single-image branch uses strict `probability > 0.5` for generated. The
   method has no native T2.

   The full safetensors state covers every model parameter. To avoid the
   official wrapper's redundant mutable download of a base model whose values
   are immediately overwritten, the adapter constructs the identical timm
   1.0.15 `vit_small_patch16_384.augreg_in21k_ft_in1k` architecture with
   `pretrained=False`, replaces its head with one output, and strict-loads all
   pinned tensors. This no-download load path matches the five four-decimal
   DALL-E 2 probabilities displayed by the official HF notebook and passes
   independently frozen full-precision references at `1e-5` absolute
   tolerance. The released 224 checkpoint is excluded from the primary
   run before scoring because the paper explicitly uses High res. as its
   best-performing model for subsequent experiments; it will not be selected
   or added based on Mouse results.

   Exact pre-score crop visibility is 162/275 `full`, 32 `partial`, and
   81 `none` pairs, with mean visible GT fraction `0.646589`. Visibility is
   a counterfactual input-condition stratum copied from each forged image to
   its matched real image, not a localization prediction. Attention or
   feature maps must not be reported as T2.

   The local-splice condition completed on 2026-07-25 with 550/550 valid
   images and zero errors. Released-probability AUROC is `0.502340`
   (1,000-pair-bootstrap 95% CI `[0.500674, 0.504873]`), AP is `0.504511`
   (`[0.502691, 0.511090]`), and the strict released threshold produces
   TP/FP/FN/TN=`1/1/274/274`. The only positive forged image belongs to the
   same pair as the only positive real image. A physically independent audit
   redecoded and reprocessed all 550 images, executed 550 fresh complete ViT
   forwards, and reproduced every 384-dimensional feature, logit,
   probability, and decision with maximum absolute error `0.0`. See
   [`COMMUNITY_FORENSICS_HIGHRES384_MOUSE_FULL_RESULTS_2026-07-25.md`](COMMUNITY_FORENSICS_HIGHRES384_MOUSE_FULL_RESULTS_2026-07-25.md).
5. [x] [**SPAI**](SPAI_MOUSE_FULL_RESULTS_2026-07-25.md) — any-resolution
   spectral OOD detection.
   Protocol frozen before any Mouse model score on 2026-07-25 to the sole
   official CVPR 2025 release. Source is pinned to
   `mever-team/spai` main commit
   `8ff7b3b6779b4fcb43cf313471d9cb1c62d129a4`; the sole checkpoint linked
   by the official README is Google Drive file
   `1vvXmZqs6TVJdj8iF1oJ4L_fcgdQrp_YI`, `spai.pth`, 934,865,338 bytes,
   SHA-256
   `24159f27d7c8c2cd0cb6c4019189eb89ad0874a0d9d15f8dc9afd39ca9648a55`.
   Its 324 model tensors contain 139,945,243 elements and strictly match the
   current official architecture. The frozen release path is Pillow RGB,
   native resolution, `[0,1]` float32, no resize/crop or test augmentation,
   eval mode, no AMP, and batch one. SPAI divides each image into
   non-overlapping 224x224 patches at stride 224, uses the current released
   `MODEL.PATCH_VIT.MINIMUM_PATCHES=4` fallback to torchvision five-crop,
   processes patch features in chunks of at most 400, aggregates them with
   spectral-context attention, and returns float32 `sigmoid(logit)`.
   Class 0 is real and class 1 is fake; the released torchmetrics 1.4
   operating rule is strict `probability > 0.5`.

   The checkpoint embeds an older training config with
   `MINIMUM_PATCHES=1`, whereas both the current inference config and the
   README's evaluation command explicitly freeze it to 4. The main run uses
   the current released inference behavior; the embedded stale value will
   not be chosen or restored based on Mouse results. For inputs with at
   least four grid patches, PyTorch `unfold` discards the non-divisible
   right and bottom remainders. The exact pre-score Mouse visibility census
   is therefore 243/275 `full`, 14 `partial`, and 18 `none`, with mean
   visible GT fraction `0.9096355444251016`: 262 pairs use the regular grid
   and 13 use five-crop, with all 13 five-crop edits fully visible. By
   domain this is lodging 132/3/12 and restaurant 111/11/6 for
   full/partial/none. This is an input-condition stratum copied to both
   images of each matched pair, not a localization output. SPAI's optional
   spectral-context attention rendering is a classifier-importance
   visualization rather than an edit probability mask, so T2 and the joint
   T1/T2 gate remain N/A.

   A non-Mouse release audit was also completed before the main run. Two
   source images from the official 3.72 GB evaluation bundle were decoded by
   the official transform and passed through the pinned current checkpoint.
   The execution contract explicitly overrides this NGC image's nonstandard
   `high`/TF32 defaults with PyTorch's standard strict-float32 setting:
   matmul precision `highest`, CUDA matmul TF32 disabled, and cuDNN TF32
   disabled. Midjourney v6.1 sample `224.png` produced logit
   `0.9909347295761108` and probability `0.7292724847793579`; Stable
   Diffusion 3 sample `000001046_4.webp` produced logit
   `1.6814128160476685` and probability `0.8430914878845215`. These
   current-release
   values do not fall inside the rounding intervals implied by the project
   page's displayed `0.748` and `0.87`, even when using the original
   evaluation files rather than the page's re-encoded JPEGs. The project
   page does not state a checkpoint hash or full-precision reference, so its
   hand-displayed scores are recorded as stale/approximate release evidence,
   not used to tune preprocessing, substitute a checkpoint, or gate Mouse
   inference. The deterministic full-precision values from the sole current
   checkpoint are frozen as implementation-regression references.

   The local-splice condition completed and passed independent full-model
   audit on 2026-07-25: 550/550 valid images, 275/275 complete pairs, and
   zero errors. Released-probability AUROC is `0.497931` (1,000-pair-bootstrap
   95% CI `[0.495543, 0.499836]`) and AP is `0.500215`
   (`[0.499264, 0.506572]`). At the released strict `probability > 0.5`
   rule, TP/FP/FN/TN=`46/48/229/227`. Forged images rank above their matched
   real controls in only 111/275 pairs, versus 145 losses and 19 ties; the
   mean forged-minus-real score delta is `-0.003603` (95% CI
   `[-0.006439, -0.001193]`). All 18 `none`-visibility pairs tie exactly
   because their edits lie wholly outside SPAI's consumed pixels. A physical
   audit redecoded and reprocessed all 550 images, ran 550 fresh complete
   FFT/ViT/SRS/SCA/MLP forwards, and exactly reproduced every patch feature,
   aggregate feature, attention diagnostic, and raw logit; probabilities
   differ by at most one float32 ULP (`5.960464477539063e-08`). T2 remains
   N/A. See
   [`SPAI_MOUSE_FULL_RESULTS_2026-07-25.md`](SPAI_MOUSE_FULL_RESULTS_2026-07-25.md).
6. [x] [**B-Free**](BFREE_DINO2REG4_MOUSE_FULL_RESULTS_2026-07-25.md)
   — bias-controlled training with local inpainting and
   restored real backgrounds; closest whole-image detector to the CLAIMFORGE
   threat model.

   Protocol frozen before any Mouse model score on 2026-07-25 to the sole
   official CVPR 2025 release. Source is pinned to
   `grip-unina/B-Free` main commit
   `c6a9f898782fb466b29af01f21960b67415afb0e`. The only checkpoint linked
   by the official README is `BFREE_dino2reg4.zip`, 321,653,488 bytes,
   MD5 `f3f53fa647848b16cf81c913f148a198`, SHA-256
   `8230fd3f0f3a64a6403acb692ce1663718ed16f36a5a4de4a68c0d273781769f`.
   It contains a 346,171,370-byte `model_epoch_best.pth` with SHA-256
   `5948ca78f4d94e820c250d24cdf155035b4a85960443800bfe6bb7f06bffe947`
   and a 153-byte config with SHA-256
   `1f0cb4988933de06a4c2427b1b5b015baa18cea7bc5223a9f54ca5e077ec8d40`.
   The restricted `weights_only=True` load has no unsafe globals and exposes
   only a 177-tensor all-float32 model state with 86,526,721 elements. It
   strictly covers the released end-to-end-finetuned DINOv2 ViT-B/14 with
   four registers and one raw-logit head.

   The frozen inference path is Pillow RGB, torchvision `ToTensor`,
   ImageNet/ResNet normalization, native resolution, batch one, eval mode,
   float32, and no resize. A kernel-14/stride-14 patch embedding first drops
   right/bottom remainders shorter than 14 pixels. If both patch-grid
   dimensions are at least 36, the release selects five 36x36-token
   windows—center, top-left, bottom-left, bottom-right, and top-right—and
   averages their five **raw logits**. If either dimension is below 36,
   `replicate_wrap` periodically repeats that token dimension but also
   truncates the other dimension to its first 36 tokens; the subsequent five
   windows are identical. This release behavior is more specific than the
   paper's prose description of post-embedding padding and multiple crops
   and is therefore the executable reference. The official primary score is
   the averaged raw logit, higher means fake, and the released operating rule
   is strict `logit > 0`. Crop logits or token features are classifier
   diagnostics, not T2 localization; T2 and the joint gate are N/A.

   Exact pre-score Mouse visibility was frozen against the union of native
   pixels consumed by these five token windows. It is 173/275 `full`,
   36 `partial`, and 66 `none`, with mean visible GT fraction
   `0.6891766376903072`. By domain, lodging is 95/18/34 and restaurant is
   78/18/32 for full/partial/none. Twenty-six pairs enter the wrap path
   (17 full, two partial, seven none); their five classifier crops are
   identical. Visibility is an input-condition stratum copied to both
   members of each matched pair, never a model localization output.

   Four source-controlled non-Mouse demo images and raw logits are frozen as
   the release golden: `-5.9374785`, `-4.441922`, `4.430519`, and
   `3.8499813`. The published values are finite decimal references rather
   than a bit-exact runtime contract. The current official-code CPU preflight
   is repeat-identical and differs by at most `2.5853210448900654e-05`;
   the CUDA acceptance tolerance was therefore pre-registered as absolute
   `5e-5` before that preflight or any Mouse scoring. The strict deterministic
   CUDA run then produced `-5.937470436096191`, `-4.441922187805176`,
   `4.430531978607178`, and `3.8499915599823`, each bit-identically on two
   complete forwards. Its maximum difference from the official CSV is
   `1.297860717741628e-05`, so the release gate passes without adjusting the
   protocol.

   B-Free is unusually relevant because its paper explicitly includes
   `origBG` training variants that restore pristine real background pixels
   around an inpainted generated region, described as effectively a local
   image edit. It is not an exact Mouse match: training uses COCO and Stable
   Diffusion 2.1, mixes self-conditioned whole-image regeneration with local
   variants, and applies additional global augmentations. The released
   checkpoint config also does not encode the claimed `inpainted++` recipe,
   and training code is not public, so this run is labeled as official
   release inference rather than a from-scratch paper reproduction.
   Finally, the GRIP license is source-available for informational and
   nonprofit use only and explicitly prohibits industrial/profit-oriented
   use; it must not be described as permissive or commercially cleared.

   The local-splice condition completed and passed independent full-model
   audit on 2026-07-25. Run
   `bfree_dino2reg4_mouse_canonical_v1_full275_20260725` has 550/550 valid
   images, 275/275 complete pairs, and zero errors. Raw-logit AUROC is
   `0.512529` (1,000-pair-bootstrap 95% CI
   `[0.507815, 0.518612]`) and AP is `0.513062`
   (`[0.509910, 0.521203]`). At the released strict `raw_logit > 0`
   operating rule, TP/FP/FN/TN=`2/1/273/274`, so forged recall is only
   `0.007273`. Forged images rank strictly above their matched real controls
   in 143/275 pairs, versus 67 losses and 65 ties; paired ranking accuracy is
   `0.520000` (95% CI `[0.458182, 0.578182]`) and mean
   forged-minus-real delta is `0.061657` (`[0.039677, 0.084790]`).
   Visibility exposes a weak local signal despite the unusable released
   operating point: paired ranking accuracy is `0.693642` for the 173
   `full` pairs and `0.638889` for the 36 `partial` pairs, while 65 of 66
   `none` pairs tie exactly. The physical audit redecoded and reprocessed all
   550 images, ran 550 fresh complete forwards, validated every persisted
   `[5,768]` feature and `[5]` crop-logit artifact, and reproduced features,
   crop logits, raw logits, decisions, and the summary with maximum absolute
   difference `0.0`. T2 remains N/A.
7. [x] [**Effort**](EFFORT_CLIP_L14_GENIMAGE_SDV14_MOUSE_FULL_RESULTS_2026-07-25.md)
   — orthogonal-subspace generalization baseline.

   The local-splice condition completed and passed independent full-model
   audit on 2026-07-25. The primary was frozen before Mouse scoring to the
   official GenImage SDv1.4 CLIP-L/14 release at commit
   `96f5dea2b534d400cfd7003f053c7e93c8e16461`, checkpoint SHA-256
   `7c32ceb4e66d303050e8fc5dc7543fa347693fb4ee6b5df4d6eaf9f6a92fb813`,
   and the official natural-image demo's OpenCV RGB conversion plus direct
   224x224 `INTER_LINEAR` resize. The exact 681-tensor graph strictly loads
   all 303,378,530 FP32 elements and contains 96 attention projections with
   rank-1023 frozen main weights plus rank-1 residuals.

   Run `effort_clip_l14_genimage_sdv14_mouse_canonical_v1_full275_20260725`
   has 550/550 valid images, 275/275 complete pairs, and zero errors. AUROC is
   `0.500456` (95% pair-bootstrap CI `[0.498379,0.502850]`), AP is
   `0.506262`, and TPR@5%FPR is `0.054545`. At the released strict
   `softmax(class 1) > 0.5` rule, TP/FP/FN/TN=`23/23/252/252`; all 275
   paired threshold decisions are unchanged by the local insertion. Paired
   ranking is `0.530909` (CI `[0.469091,0.589091]`), with 146 wins,
   129 losses, no ties, and exact sign-test `p=0.334638`. A physical audit
   redecoded and reprocessed all 550 inputs, ran 550 fresh full-model
   forwards, replayed every persisted feature and class logit, and reproduced
   all arrays, scores, decisions, and summary fields with maximum absolute
   difference `0.0`. T2 remains N/A.
8. [x] [**OmniAID**](OMNIAID_DINO_V2_MIRAGE_MOUSE_FULL_RESULTS_2026-07-25.md)
   — 2026 semantic/artifact mixture-of-experts baseline.

   The local-splice condition completed and passed independent full-model
   audit on 2026-07-25. The primary was frozen before Mouse scoring to the
   currently recommended/default official DINO v2 Mirage auto-router release:
   GitHub commit `40749406fbcd8893c11a160edf4a72a2d4dc7056`, Space commit
   `cf99ed518af8b7256854d01994d6e41165553bb3`, and the
   3,238,483,725-byte checkpoint with SHA-256
   `8135cf83a7acbd3d88e457062f7ad693b1f2e27ffc8d5ae7ec73fcb5de806ea9`.
   The graph contains two DINOv3 ViT-L/16 instances, 96 SVD-MoE attention
   projections, a top-2-of-5 semantic router, and the always-enabled Artifact
   expert. The frozen input path is RGB, direct bilinear+antialias 448x448
   resize, ImageNet normalization, float32, batch one; class-1 softmax with
   strict `>0.5` is the released T1 rule and there is no native T2 output.

   Run `omniaid_dino_v2_mirage_auto_mouse_canonical_v1_full275_20260725`
   has 550/550 valid images, 275/275 complete pairs, and zero errors. AUROC is
   `0.499636` (95% pair-bootstrap CI `[0.496661,0.502401]`), AP is `0.505429`,
   and TPR@5%FPR is `0.058182`. At the released threshold,
   TP/FP/FN/TN=`8/8/267/267`; all 275 paired decisions are unchanged. Paired
   ranking is `0.501818` (CI `[0.447273,0.560000]`), with 138 wins, 137
   losses, no ties, and mean forged-minus-real delta `0.000575010`
   (CI `[-0.000497496,0.001597893]`). Semantic top-2 selections are unchanged
   in 274/275 pairs. The independent audit rebuilt the official graph,
   redecoded all inputs, ran 550 fresh full-model forwards, replayed all six
   persisted artifact arrays plus router/head/softmax/decision logic, and
   reproduced every array and both stored/fresh summaries with maximum
   absolute difference `0.0`. T2 remains N/A.
9. [ ] [**LTD — official-weight blocked**](LTD_DRCT_SDV14_OFFICIAL_WEIGHT_BLOCKER_2026-07-25.md)
   — 2026 latent-transition consistency baseline.

   The code and protocol were frozen on 2026-07-25 to the official CVPR 2026
   repository commit `27a8a7e6acd97c1b50b584f85dcca47c1584614b` and the
   README's DRCT-2M primary `DRCT_sdv1.4.pth`. The Google Drive link returns
   a 1,789-byte owner-permission error page rather than checkpoint bytes.
   The official Baidu share code is valid and exposes the filename, exact
   size `1,862,028,941` bytes, and `fs_id`, but anonymous download returns no
   `dlink` because login state and a signed request are required. No author
   SHA-256 or license text is published.

   No third-party mirror was substituted, no CUDA inference was started, and
   no Mouse score was fabricated. The official RGB → Resize(256) →
   CenterCrop(224) → CLIP normalization path, strict sigmoid score direction,
   crop visibility audit, and released hard-Gumbel selector RNG contract are
   pre-registered in the blocker report. The method remains incomplete until
   the exact official bytes pass the CPU-only size/hash/safe-load/schema gate.
10. [x] [**CNNDetection**](CNNDETECTION_BLUR_JPG_PROB0_1_NATIVE_MOUSE_FULL_RESULTS_2026-07-25.md)
    — historical frequency/artifact anchor.

    The local-splice primary, preregistered paper-era crop sensitivity, and
    full-model replay audits completed on 2026-07-25. Before Mouse scoring, the
    primary was frozen to official commit
    `ea0b5622365e3a9cd31d1b54b6b5971131a839ab`, the official
    Blur+JPEG(0.1) checkpoint (282,442,597 bytes; SHA-256
    `a73295ac66f9cb74d558ce3ade46f75e2f2997ed05eeed0f4b774623372058ea`),
    and the post-release recommended RGB, native-resolution, no-resize,
    no-crop, batch-one path. Blur/JPEG are training augmentations only.
    The official one-logit float32 output is passed through an uncalibrated
    sigmoid and classified with strict `>0.5`; there is no native T2 output.

    Native run
    `cnndetection_blur_jpg_prob0_1_native_mouse_canonical_v1_full275_20260725`
    has 550/550 valid images, 275/275 complete pairs, and zero errors. AUROC is
    `0.498896` (95% pair-bootstrap CI `[0.497401,0.500046]`), AP is `0.502089`,
    and TPR@5%FPR is `0.047273`. At the released threshold,
    TP/FP/FN/TN=`0/0/275/275`; every paired decision is `0→0`. Paired ranking
    is `0.454545` (CI `[0.396364,0.512727]`), with 125 wins, 150 losses, no
    ties, exact sign-test `p=0.147691`, and mean score delta
    `-7.381589e-7` (CI `[-2.581092e-6,5.426804e-7]`). The replay audit ran
    550 fresh forwards and reproduced every persisted 2,048-dimensional
    feature with maximum absolute difference `0.0`. It reuses the shared
    metrics implementation and therefore is not represented as a second fully
    independent statistical implementation.

    The separately identified `CenterCrop(224)` sensitivity also completed
    550/550 plus audit: AUROC `0.499702`, AP `0.499548`, and
    TP/FP/FN/TN=`2/2/273/273`. Its visibility census is only 14 full, 14
    partial, and 247 none; 246 pairs tie exactly. It records paper-era
    preprocessing ambiguity and never replaces the native primary.

The first seven form the minimum publishable mechanism-complete suite. The
last three complete the preferred main table. Nine of ten local-splice runs
are complete; all candidates with retrievable exact official weights have
finished. The sole remaining item is official-weight-blocked LTD. All seven
minimum-suite methods are complete.
Zero of ten methods is
contrast-complete until the same-domain fully synthetic
condition is available.

### Appendix-only candidates

AEROBLADE, DRCT, AlignedForensics, AIDE, FerretNet, and GAPL.  These are kept
outside the main table to avoid repeating closely related mechanisms while
preserving an expansion path.

## 4. Required whole-image contrast condition

The AIGC track is incomplete even after running the ten models unless the
same detectors are also evaluated on a same-domain fully synthetic control:

- approximately 150 whole-image synthetic lodging/restaurant scenes;
- matched real controls processed through the same canonical encoding;
- the same T1 metrics and score direction used for local-splice inputs.

This condition distinguishes “the detector is broken” from “the detector
works on full synthesis but cannot see a very small synthetic fraction.”

## 5. Execution order

Run one method at a time:

1. PSCC-Net — completed 2026-07-24
2. IML-ViT — completed 2026-07-24
3. HiFi-IFDL — completed 2026-07-24
4. Mesorch — completed 2026-07-24
5. RelayFormer — completed 2026-07-24
6. DINOv3-IML — completed 2026-07-24
7. NFA-ViT / BR-Gen — deferred; official checkpoint unavailable
8. FSD — local-splice condition completed 2026-07-24; fully synthetic
   control pending
9. UniversalFakeDetect — local-splice current-HEAD primary and
   checkpoint-era preprocessing sensitivity completed 2026-07-24; fully
   synthetic control pending
10. NPR — local-splice condition completed and independently audited
    2026-07-25; fully synthetic control pending
11. Community Forensics — local-splice condition completed and independently
    audited 2026-07-25; fully synthetic control pending
12. SPAI — local-splice condition completed and independently audited
    2026-07-25; fully synthetic control pending
13. B-Free — local-splice condition completed and independently audited
    2026-07-25; fully synthetic control pending
14. Effort — local-splice condition completed and independently audited
    2026-07-25; fully synthetic control pending
15. OmniAID — local-splice condition completed and independently audited
    2026-07-25; fully synthetic control pending
16. LTD — official DRCT-2M weight access blocked and protocol documented
    2026-07-25; no Mouse score emitted
17. CNNDetection — native local-splice primary, paper-era crop sensitivity,
    and both replay audits completed 2026-07-25; fully synthetic control
    pending

All retrievable official-weight AIGC methods are now complete on the
local-splice condition. FSD, UniversalFakeDetect, NPR, Community Forensics,
SPAI, B-Free, Effort, OmniAID, and CNNDetection have audited results, but all
nine required same-domain fully synthetic controls remain open. LTD remains
blocked on exact official checkpoint access, and NFA-ViT
remains deferred and its checkpoint SHA-256 gate remains closed.
A method is complete only when its runner, tests, source/checkpoint provenance,
smoke output, 275-pair output, machine-readable summary, and audited result
report are all present.
