# Paper Plan

**Working title**: Mostly Real, Locally Manipulated: CLAIMFORGE, a Paired Mouse-Core Stress Test for Localized AI Edits

**Broader title (use only after the multi-object integrity audit and evaluation are complete)**: Mostly Real, Locally Manipulated: CLAIMFORGE, a Paired Stress Test for Localized AI-Edited Consumer Evidence

**One-sentence contribution**: CLAIMFORGE formalizes tiny local AI object insertions in consumer-evidence photos and shows that, on a 275-pair canonical mouse core, none of eight reproduced local-forensics methods evaluated under their registered native capabilities simultaneously reaches the study-specific base-condition references of image AUROC at least 0.90 and macro pixel F1 at least 0.50.

**Venue**: AAAI-27, AI for Social Impact special track

**Type**: Empirical benchmark / paired stress test

**Date**: 2026-07-25

**Page budget**: 7 pages of main content, with pages 8--9 reserved for references

**Structure**: 6 numbered sections plus abstract

## Evidence Scope

The paper must keep three evidence tiers separate:

1. **Primary evidence**: the canonical mouse core (275 real/forged pairs, 550 matched JPEGs) and eight complete open-source local-forensics runs.
2. **Diagnostic evidence**: one legacy four-run MLLM export and heterogeneous commercial-service runs. The MLLM export is format-confounded (source-JPEG reals versus PNG forgeries), and the commercial protocols are mostly forged-only; neither can support the primary performance claim.
3. **Expansion/future evidence**: 251 selected cat and 250 selected trash-can variants. These assets are not in the detector results and have an unresolved cross-decoder pixel-drift audit; they cannot support the headline.

The old draft's “600 candidates / 297 slots / 22 pilot generations / 0.232%” snapshot is obsolete. The current accounting is:

- 600 screened source candidates;
- 594 annotated mouse slots; six screened candidates do not appear in the annotation manifests, and their exclusion reasons require release-manifest reconciliation;
- 594 mouse generation attempts manually reviewed;
- 275 accepted mouse forgeries, 288 rejected forgeries, and 31 rejected sources;
- 275 canonical mouse pairs (147 lodging, 128 restaurant);
- median exact changed-pixel fraction 0.11264%, meaning a median 99.88736% of pixels are unchanged before matched canonical encoding;
- 251 cat and 250 trash-can expansion variants, excluded from the current evaluation.

## Claims--Evidence Matrix

| ID | Claim | Evidence | Status | Paper location |
|---|---|---|---|---|
| C1 | Consumer-evidence photos create a distinct mostly-real local-edit threat that requires both detection and localization. | Threat definition; paired T1/T2 protocol; restaurant and lodging examples. | Supported as a problem formulation. | Abstract, §1, §2 |
| C2 | The canonical mouse core controls container/EXIF shortcuts and supplies pre-encoding exact RGB-difference localization targets. | Both sides are freshly encoded JPEG Q95 with 4:4:4 subsampling and inherited EXIF stripped; GT is computed before canonical encoding; all changed pixels are constrained to the reviewed context. | Supported for mouse-275 only. | §2 |
| C3 | The eight reproduced methods exhibit an uneven capability tradeoff rather than universal random failure. | TruFor AUROC 0.8179 and macro F1 point estimate 0.5001 (provisional paired-task CI crosses 0.50); CAT-Net pixel AP 0.6123 but no registered T1; other methods are lower. | Supported. | §4 |
| C4 | No reproduced method covers both required capabilities at the study-specific base-condition references. | Among methods evaluated with registered native outputs, none simultaneously reaches AUROC ≥ 0.90 and macro pixel F1 ≥ 0.50 on mouse-275. The full proposed solved rule additionally requires JPEG-75 robustness and remains unevaluated. | Supported only on the base canonical mouse condition. | Abstract, §3, §4, §6 |
| C5 | Smaller edits are harder for the strongest joint model. | TruFor AUROC rises from 0.7223 in the smallest edit quintile to 0.9412 in the largest. | Supported for TruFor; do not generalize into a universal mechanism. | §5 |
| C6 | Commercial diagnostics illustrate the difference between conditional localization and end-to-end coverage. | Copyleaks has 0.8165 positive-only IoU but 0.3296 all-image IoU; other vendors have heterogeneous forged-only coverage. The legacy MLLM export is retained only as a format-leakage warning and supports no performance claim. | Diagnostic only; protocols differ. | §5 |
| C7 | CLAIMFORGE is robust across objects, editors, and laundering. | No completed common evaluation over cat/trash, a second editor, or laundering. | Not supported. | Limitations / future work only |

## Page Budget

| Component | Target pages | Purpose |
|---|---:|---|
| Title + abstract | 0.25 | Threat, exact scope, strongest evidence |
| §1 Introduction and Evaluation Gap | 1.05 | Stakes, gap, RQs, contributions, hero figure |
| §2 CLAIMFORGE Mouse-Core Construction | 1.25 | Data funnel, QA, canonicalization, task definition |
| §3 Capability-Aware Evaluation Protocol | 0.85 | Models, native-output policy, metrics, thresholds, statistics |
| §4 Main Results on the Canonical Mouse Core | 1.55 | Main table and joint-capability conclusion |
| §5 Stress Patterns and Diagnostic Comparisons | 0.80 | Edit size, pristine false positives, diagnostic ecosystems |
| §6 Limitations, Ethics, and Conclusion | 0.55 | Scope boundary, dual use, narrow conclusion |
| **Planned total** | **6.30** | Includes planned floats and leaves 0.70 page for citation/layout drift |

If the draft exceeds seven main-content pages, move the detailed data funnel and qualitative examples to the supplement first; preserve the hero figure, main results table, and edit-size figure.

## Structure

### §0 Abstract

- Open with the counterintuitive fact: the median forged pair changes only 0.11264% of pixels.
- State the practical setting: restaurants and lodging claims.
- Define the evaluated artifact exactly: 275 QA-approved source/forged pairs under matched JPEG canonicalization.
- Name the primary evaluation: eight fully reproduced local-forensics methods.
- Give the nuanced result:
  - TruFor is the strongest registered joint model (AUROC 0.8179; TPR@5%FPR 0.4436; macro pixel F1 point estimate 0.5001).
  - CAT-Net has the best pixel AP (0.6123) but no registered image-level head.
  - No method simultaneously reaches the two study-specific base-condition references.
- Close with “meaningful but incomplete forensic signal,” not “all detectors are random” or a deployment-readiness claim.
- Do not mention the 501 unevaluated expansion variants.
- Target length: 160--190 words.

### §1 Introduction and Evaluation Gap

**Opening**: A consumer-evidence photo can be materially false even when nearly every pixel is from a real camera photograph.

**Stakes**: Explain why refunds, platform penalties, reputation, and insurance decisions make this a trust-and-safety problem. Add verified non-CS literature on consumer fraud, digital evidence, and platform governance before submission.

**Threat**: Contrast tiny semantic insertions with whole-image generation. The inserted object, not the global image, changes the claim.

**Research questions**:

- RQ1: Can current methods detect these tiny edits at low false-positive rates?
- RQ2: Can they localize the changed evidence accurately enough for review?
- RQ3: How does the strongest joint model vary with edit size and descriptive domain slices?

**Evaluation gap**: Synthesize whole-image detection, manipulation localization, inpainting benchmarks, and shortcut studies. End with the missing conjunction: consumer evidence + mostly-real paired construction + native detection/localization reporting.

**Contributions**:

1. A scoped threat formulation and paired T1/T2 task.
2. A reproducible mouse-core construction and canonicalization pipeline with manual QA and exact-difference masks.
3. A capability-aware protocol that preserves N/A rather than manufacturing unsupported heads.
4. An eight-method result showing a detection/localization tradeoff and a strong edit-size effect without claiming universal random failure or satisfaction of the full JPEG-75 solved rule.

**Hero figure**: source → target crop → generated object → composite → exact-difference mask, with a small solved-quadrant inset. The inset and caption must explicitly say “canonical mouse core.”

### §2 CLAIMFORGE Mouse-Core Construction

#### §2.1 Threat and Tasks

- Formalize source image \(x\), forged image \(\tilde{x}\), image label \(y\), and pre-encoding exact RGB-difference mask \(M\).
- T1 is image-level manipulation detection.
- T2 is localization in the decoded pre-canonicalization RGB coordinate space.
- Explain why both are required for adjudication.

#### §2.2 Source Screening, Generation, and Human QA

- Start from the 600-image screened restaurant/lodging pool.
- Explain context boxes, edit boxes, crop generation, difference-derived object-only paste-back, final-composite review, and relabel/repair.
- Record the provenance caveat: lodging records identify HunyuanImage-3, but the 128 restaurant records retain only the legacy `new_test` editor tag; reconcile before claiming a single editor.
- Report the funnel: 600 screened candidates → 594 annotated/reviewed mouse attempts → 275 accepted forgeries.
- State that six screened candidates do not appear in the annotation manifests and require final release-accounting reconciliation.
- Keep lower-level SAM/difference/hysteresis implementation details in the supplement.

#### §2.3 Canonical Mouse Core

- Build 275 matched real/forged pairs (550 images).
- Independently decode and freshly encode both sides as JPEG Q95, 4:4:4, with inherited EXIF stripped.
- Compute GT before canonical encoding from exact RGB differences.
- Report domain counts and edit-fraction distribution.
- State that the current evaluated core contains 270 unique pristine hashes across 275 task pairs; five sources repeat.

#### §2.4 Scope Boundary

- The 251 cat and 250 trash-can selections are expansion assets only.
- Do not call them an evaluated benchmark release.
- State in Limitations that their canonical cross-decoder integrity audit must be resolved before inclusion.

### §3 Capability-Aware Evaluation Protocol

#### §3.1 Reproduced Methods and Native-Capability Rule

- Primary methods: MaskCLIP, TruFor, CAT-Net v2, MVSS-Net, PSCC-Net, IML-ViT, HiFi-IFDL, and Mesorch.
- Preserve each method's official preprocessing, checkpoint, output, and public threshold.
- Report a T1 or T2 result only when the registered method exposes that capability.
- Mark missing capabilities N/A; do not turn heatmap mean/max into an improvised image detector.

#### §3.2 Metrics

- T1: AUROC, AP, TPR at FPR ≤ 5%, fixed-threshold results, paired rank/delta.
- T2: pixel AP, macro and micro F1/IoU/MCC, box hit, and pristine-image false-positive area.
- Prefer a two-dimensional capability quadrant over a scalar joint score.
- Define 0.90 AUROC and 0.50 macro F1 as study-specific base-condition references, not a universal standard.
- State that the broader proposed solved rule also requires preserving both thresholds after JPEG-75 laundering; that composite rule remains unevaluated.
- Put the compact point estimates in Table 2 and route detection AP/fixed decisions/paired deltas plus micro F1/IoU/MCC/box hits to Table S3.

#### §3.3 Statistics and Diagnostic Protocols

- Report the existing 1,000-replicate task-pair bootstrap CIs as provisional.
- Explicitly note that repeated pristine sources can make those intervals optimistic.
- Add source-hash-cluster bootstrap before final submission; if it is unavailable, do not present the provisional intervals as final uncertainty.
- Keep the format-confounded legacy MLLM export and commercial forged-only runs outside the primary table.

### §4 Main Results on the Canonical Mouse Core

#### §4.1 Image-Level Detection

- TruFor provides the strongest signal: AUROC 0.8179, AP 0.8393, and TPR@5%FPR 0.4436.
- MaskCLIP, MVSS-Net, PSCC-Net, and HiFi-IFDL remain near 0.5 AUROC.
- Methods without registered T1 remain N/A.

#### §4.2 Pixel-Level Localization

- CAT-Net leads continuous ranking with pixel AP 0.6123.
- TruFor leads fixed-threshold macro F1 at a point estimate of 0.5001 and macro IoU at 0.3741; its provisional paired-task F1 interval (0.4676--0.5318) crosses the 0.50 reference.
- Report real-image positive area to prevent a forged-only interpretation of masks.

#### §4.3 Joint Capability

- Plot methods with both registered capabilities in AUROC × macro-F1 space.
- State the narrow headline exactly:

> Among eight reproduced methods evaluated under their registered native capabilities on the canonical mouse core, none simultaneously reaches the study-specific base-condition references of image AUROC 0.90 and macro pixel F1 0.50.

- Explain that CAT-Net is a strong localizer without T1, not evidence that a joint method nearly solves the task.
- Add the repeated-source uncertainty caveat in this section, not only in Limitations.

### §5 Stress Patterns and Diagnostic Comparisons

#### §5.1 Edit Size and Descriptive Domain Slices

- Make the edit-size effect the main analysis: TruFor AUROC 0.7223 (smallest quintile) versus 0.9412 (largest).
- Treat restaurant/lodging differences as descriptive, not causal.
- Pair the curve with 3--4 qualitative successes/failures.

#### §5.2 Pristine False Positives

- Explain that a useful localizer must not merely produce plausible-looking maps on every image.
- Use TruFor's pristine-map behavior to motivate joint review of image and mask outputs.

#### §5.3 Diagnostic Ecosystems

- MLLMs: the legacy export is format-confounded (JPEG reals versus PNG forgeries), and box localization uses a different target. Treat it only as a protocol warning; rerun matched canonical JPEG pairs before reporting model performance.
- Commercial services: report coverage and vendor-threshold verdicts, never infer paired AUROC/FPR from forged-only runs.
- Copyleaks illustrates conditional versus end-to-end localization: 0.8165 IoU when positive, 0.3296 when misses count as empty masks.

#### §5.4 Operational Takeaways

- The results motivate evaluation protocols that jointly report low-FPR detection and spatial evidence.
- Coverage, native output semantics, and empty-mask handling must be reported.
- Do not claim that a model “detects local AI edits” from a whole-image score alone.

### §6 Limitations, Ethics, and Conclusion

#### §6.1 Limitations

- One recorded generation workflow and one fully evaluated object class; exact editor provenance for 128 restaurant pairs still requires reconciliation.
- No completed laundering, held-out-editor, whole-image contrast, real-object paste-back, real-with-object, or human study.
- Five repeated pristine hashes; source-cluster bootstrap is pending.
- Cat/trash expansion is excluded pending canonicalization and cross-decoder integrity resolution.
- The legacy MLLM export is format-confounded and cannot support a performance claim.
- Commercial protocols are incomplete and heterogeneous.

#### §6.2 Ethics and Release

- The dataset is dual-use because it contains plausible fraudulent evidence.
- Release terms should be gated/research-only, but the paper must describe this as a plan until the policy is actually implemented.
- Document source licenses and avoid implying deployment readiness.
- Frame the benefit as evaluating trust-and-safety defenses, not enabling operational fraud.

#### §6.3 Conclusion

- Restate only the mouse-core result.
- Emphasize that current methods contain useful signal but do not cover both required capabilities at the declared operating target.
- End with the concrete next evaluation: source-cluster uncertainty, multi-object canonicalization, and laundering.

## Figure and Table Plan

| ID | Type | Description | Data source | Priority |
|---|---|---|---|---|
| Fig. 1 | Hero composite | Full source, target crop, generated mouse, composite, exact-diff mask, and a compact native-capability quadrant for the five methods with native T1+T2 | `docs/figures/fig1_threat_model.jpg`; exact example task ID and source/generation paths still `[VERIFY]`; `results/opensource/{trufor,maskclip,mvssnet,psccnet,hifi_ifdl}/*full275*.summary.json` | High |
| Table 1 | Data funnel | 600 screened → 594 reviewed attempts → 275 evaluated pairs; list 251/250 expansion assets as excluded from results | Source manifest, review labels, selection JSONs | High |
| Table 2 | Main results | Registered T1, AUROC, TPR@5%FPR, pixel AP, macro F1, pristine FP area; preserve N/A | `results/opensource/*/*full275*.summary.json` | High |
| Fig. 2 | Stress analysis | TruFor metrics across exact edit-size quintiles plus qualitative success/failure examples | `results/opensource/trufor/trufor_mouse_canonical_v1_full275_20260723.analysis.json`; exact qualitative task IDs `[VERIFY]` | High |
| Table S1 | Diagnostics | Legacy format-confounded MLLM export labeled as invalid for performance claims; commercial forged-only coverage/verdict rates in a separate block | Exact MLLM export and commercial summary paths `[VERIFY]` | Supplement |
| Table S2 | Reproducibility | Checkpoints, commits, preprocessing, thresholds, licenses, and coverage | Method reports and run manifests | Supplement |
| Table S3 | Full metrics and uncertainty | Detection AP/fixed decisions/paired deltas; micro F1/IoU/MCC; box hits; provisional pair bootstrap and final source-cluster intervals | `results/opensource/*/*full275*.{summary,analysis}.json` | Supplement |

### Hero Figure Caption Draft

> CLAIMFORGE evaluates a mostly-real threat: a claim-relevant object is generated in a local context and composited into a real consumer-evidence photo, while exact pre-canonical RGB differences define the localization target. The evaluated canonical mouse core changes a median 0.11264% of image pixels. The capability inset reports only methods with registered native outputs on this 275-pair condition; no evaluated joint method enters the study-specific base-condition region of AUROC ≥ 0.90 and macro pixel F1 ≥ 0.50.

## Citation Plan

- **§1**: GenImage, Inpainting Exchange, Fake-or-JPEG, plus non-CS work on consumer fraud, platform adjudication, and digital evidence; concrete non-CS candidates remain `[VERIFY]`.
- **§1 gap**: TruFor, HiFi-IFDL, IMDL-BenCo, GIM, OpenSDI, COCO-Inpaint, Mesorch, SAFIRE.
- **§2**: dataset-source/license references and the HunyuanImage-3 model card/paper, all still `[VERIFY]`.
- **§3**: original papers for all eight reproduced methods; current draft directly cites TruFor, OpenSDI/MaskCLIP, HiFi-IFDL, Mesorch, and IMDL-BenCo, while CAT-Net, MVSS-Net, PSCC-Net, and IML-ViT originals remain `[VERIFY]`.
- **§5 diagnostics**: verified MLLM/local-edit and commercial-detector benchmark papers `[VERIFY]`; do not generate BibTeX from memory.
- Existing entries should be upgraded from arXiv to published versions where available.

## Independent Reviewer Feedback

Independent GPT-5.4 review scores:

| Dimension | Score |
|---|---:|
| Logical flow | 8/10 |
| Claim--evidence alignment | 7/10 |
| Missing experiments/analysis | 6/10 |
| Positioning | 8/10 |
| Seven-page feasibility | 6/10 |
| Front matter / hero figure | 8/10 |

The reviewer recommended **REVISE**, with four minimum changes that are incorporated here:

1. Scope the title, abstract, contributions, and conclusion to the canonical mouse core.
2. Separate primary evidence from MLLM/commercial diagnostics and unfinished expansion assets.
3. Surface the repeated-source/bootstrap caveat in Results.
4. Compress the paper to six sections and rename the analysis section “Stress Patterns and Diagnostic Comparisons.”

## Integrity Blockers Before Broader Claims

- Rebuild or independently canonicalize cat/trash real-forged pairs and resolve the observed cross-decoder context-exterior RGB drift.
- Re-run an exact-difference/context-boundary audit on every expansion pair.
- Ensure all selected-variant provenance is committed and reproducible from a clean clone.
- Replace the legacy `new_test` editor tag with auditable provenance for all 128 restaurant mouse pairs.
- Use source-hash groups for splits and uncertainty estimates.

## Next Steps

- [ ] Generate the hero figure and edit-size figure from audited artifacts.
- [ ] Run source-cluster bootstrap for the main research-model results before treating confidence intervals as final.
- [ ] Reconcile why six screened candidates do not appear in the 594-slot annotation manifests.
- [ ] Reconcile the exact editor provenance of the 128 restaurant pairs currently tagged `new_test`.
- [ ] Rerun MLLMs on matched canonical JPEG pairs before making any MLLM performance claim.
- [ ] Verify and update all bibliography metadata.
- [ ] Decide whether to keep the evidence-safe mouse-core title or expand only after cat/trash evaluation.
- [ ] Complete the Reproducibility Checklist.
- [ ] Flatten `\input` files into one submission `.tex` only at final packaging time.
