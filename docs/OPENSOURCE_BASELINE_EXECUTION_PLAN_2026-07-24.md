# CLAIMFORGE Open-Source Baseline Execution Plan

**Frozen:** 2026-07-24
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

### Completed full runs (6)

1. [x] OpenSDI / MaskCLIP
2. [x] TruFor
3. [x] CAT-Net v2
4. [x] MVSS-Net (official CASIAv2 checkpoint)
5. [x] [PSCC-Net](PSCCNET_MOUSE_FULL_RESULTS_2026-07-24.md) (official
   synthetic-pretrained checkpoint)
6. [x] [IML-ViT](IMLVIT_CAT_PROTOCOL_MOUSE_FULL_RESULTS_2026-07-24.md)
   (official CAT/TruFor-protocol checkpoint; native T2 only)

### Frozen main-table queue (5)

1. [ ] **HiFi-IFDL** — general-forgery checkpoint covering
   GAN/diffusion content; native T1 + T2.
2. [ ] **Mesorch** — pre-register `mesorch-98.pth`; native T2.
3. [ ] **RelayFormer** — official paper image-only checkpoint; native T2.
4. [ ] **DINOv3-IML** — CAT ViT-L LoRA-r32 checkpoint; native T2; label as a
   non-peer-reviewed 2026 preprint.
5. [ ] **NFA-ViT / BR-Gen** — official BR-Gen checkpoint; native T1 + T2;
   training manipulations closely match localized generative editing.

After this queue, the local main table contains 11 methods: 6 completed and 5
new runs.

### Appendix-only candidates

AdaIFL, SparseViT, SAFIRE, FOCAL, and ForensicsSAM.  SAFIRE is not in the
frozen main queue because its repository does not state a project license.
RITA remains excluded until its advertised inference components are complete.

## 3. Track B — whole-image AIGC detection

All methods in this track output an image-level score.  Their CLAIMFORGE result
answers whether a whole-image synthetic detector transfers to a small local
diffusion insertion; it does not replace a localizer.

### Frozen main-table queue (10)

1. [ ] **FSD / Forensic Self-Descriptions** — real-only residual/GMM detector.
2. [ ] **UniversalFakeDetect** — canonical CLIP linear-probe baseline.
3. [ ] **NPR** — neighboring-pixel and upsampling artifacts.
4. [ ] **Community Forensics** — broad generator-diversity supervision.
5. [ ] **SPAI** — any-resolution spectral OOD detection.
6. [ ] **B-Free** — bias-controlled training with local inpainting and
   restored real backgrounds; closest whole-image detector to the CLAIMFORGE
   threat model.
7. [ ] **Effort** — orthogonal-subspace generalization baseline.
8. [ ] **OmniAID** — 2026 semantic/artifact mixture-of-experts baseline.
9. [ ] **LTD** — 2026 latent-transition consistency baseline.
10. [ ] **CNNDetection** — historical frequency/artifact anchor.

The first seven form the minimum publishable mechanism-complete suite.  The
last three complete the preferred main table.

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
3. HiFi-IFDL
4. Mesorch
5. RelayFormer
6. DINOv3-IML
7. NFA-ViT / BR-Gen
8. FSD
9. UniversalFakeDetect
10. NPR
11. Community Forensics
12. SPAI
13. B-Free
14. Effort
15. OmniAID
16. LTD
17. CNNDetection

The immediate active item is **HiFi-IFDL**.  A method is complete only when its
runner, tests, source/checkpoint provenance, smoke output, 275-pair output,
machine-readable summary, and audited result report are all present.
