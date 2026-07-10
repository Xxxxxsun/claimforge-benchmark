# CLAIMFORGE — Benchmark & Evasion Study: Design Document

*Working design grounded in a verified literature sweep (2023–2026). Citations are arXiv/venue IDs. "SOTA/first" claims are hedged — the field moves fast; re-check at submission.*

Scope locked with author: **(a)** baselines cover BOTH families — image-manipulation localization (IML) **and** whole-image AIGC detection; **(b)** the paper is **benchmark + systematic evasion study**; **(c)** evaluated methods must do BOTH **image-level detection** and **pixel/region localization**.

---

## 1. Positioning & novelty (what makes this publishable)

CLAIMFORGE = first benchmark for **localized AI object-insertion forgeries in consumer-claim / complaint imagery** (lodging & restaurant: rooms, bathrooms, kitchens, dining), where a *real* photo gets a small synthetic object (pest / contamination / damage) inpainted into a marked region as a **pixel-preserving local edit** (only object pixels change; rest byte-identical), used as fraudulent claim evidence and crafted to evade detectors.

Two novelty pillars the literature confirms are genuinely open:

1. **Domain is unaddressed.** None of the comparable benchmarks (GIM, OpenSDI, GenImage, INP-X) use consumer-claim lodging/restaurant photos. This is real novelty — but it also means there is **no in-domain scale precedent**, which shapes the scale decision (§5).
2. **The pixel-preserving design structurally avoids the field's #1 dataset shortcut.** "Fake or JPEG?" (arXiv:2403.17608) shows most AIGC datasets store fakes as lossless PNG and reals as JPEG, so detectors learn *format*, not *forgery*, and collapse under recompression. Because CLAIMFORGE keeps the whole frame byte-identical to a real JPEG source except the object, real and fake share **identical post-processing** → the shortcut is neutralized by construction. **Foreground this as a methodological strength.**

The evasion thesis is strongly pre-supported: **INP-X / "Inpainting Exchange"** (arXiv:2602.00192) shows AIGC detectors over-rely on *global* VAE encode/decode artifacts and **collapse from ~94% → ~55%** on local inpainted edits once global traces are laundered out (Corvi2023 94.2→55.4; Hive 91.4→54.8; Sightengine 92.6→55.0; fine-tuned models ~98→~60). That is *exactly* CLAIMFORGE's threat model — cite it as the central motivation.

---

## 2. Task definition (require BOTH; this is the standard)

Requiring detection **and** localization is the established standard in the three closest works — **IMDL-BenCo** (NeurIPS'24, arXiv:2406.10580), **OpenSDI** (CVPR'25), **GIM** (AAAI'25, arXiv:2406.16531) — all score image-level Acc/F1/AUC **plus** pixel-level F1/IoU.

- **T1 Detection:** real vs forged, image-level.
- **T2 Localization:** per-pixel forged-region mask (you already ship `masks/`).
- Methods that natively do only one task still compete on the other via a documented adapter (§3), with the adapter disclosed.

---

## 3. Baseline suite (both families) — adopt an existing harness

**Do not re-implement.** Adopt **IMDL-BenCo** (github.com/scu-zjz/IMDLBenCo) as the IML harness: it already ships **8 SOTA IML models** with unified protocols, **15 GPU metrics**, and **3 robustness evals**, fixing the "inconsistent protocols → unfair comparison" pitfall the field calls out.

### 3a. IML / forensics family (output: pixel mask + usually image score)
| Method | Venue | Output | Notes for CLAIMFORGE |
|---|---|---|---|
| ManTra-Net | CVPR'19 | mask | in IMDL-BenCo |
| MVSS-Net | ICCV'21 | mask + score | in IMDL-BenCo |
| CAT-Net v2 | IJCV'22 | mask | JPEG-aware; strong on splicing; in IMDL-BenCo |
| PSCC-Net | TCSVT'22 | mask + score | in IMDL-BenCo; collapses cross-gen (4.4% F1 on GIM-DDNM) |
| ObjectFormer | CVPR'22 | mask + score | reproduced in IMDL-BenCo |
| IML-ViT | AAAI'24 | mask | in IMDL-BenCo |
| **TruFor** | CVPR'23 | **mask + integrity score + reliability map** | **best dual-output baseline**; but ~0.166 IoU/0.212 F1 on CocoGlide, ~5.7% F1 on GIM-DDNM → ideal stress-test |
| **Mesorch** | AAAI'25 | mask (localization-only) | current SOTA pixel-F1 (avg 0.6771; best on 3/4 std sets — CAT-Net 0.9150 wins Columbia); evaluated **only on traditional manip** → exposes CLAIMFORGE's gap; needs image-level head for T1 |
| **SAFIRE** | AAAI'25 | mask (SAM point-prompt) | open-source; localization-only; needs aggregation for T1 |
| HiFi-IFDL | CVPR'23 | mask + score | hierarchical; covers CNN-synth + editing (13 methods); 96.8% det-AUC / 95.3% loc-AUC |

### 3b. Whole-image AIGC family (output: image score only → needs mask adapter for T2)
CNNDetection (CVPR'20), DIRE (ICCV'23), UniversalFakeDetect/CLIP-probe, NPR (CVPR'24), FatFormer (CVPR'24), C2P-CLIP (arXiv:2408.09647), AIDE, DRCT.
- **Adapter for T2:** Grad-CAM / patch-score aggregation → coarse mask, **clearly marked as a non-native localizer**. The fact that whole-image AIGC detectors *cannot natively localize* is itself a result.

### 3c. Bridging models / "advanced" baselines (closest prior art — the crux)
- **GIMFormer** (from GIM) and **MaskCLIP** (from OpenSDI, CLIP+MAE) — purpose-built detect+localize on diffusion inpainting; the strongest honest baselines. Showing *these* fail on CLAIMFORGE (cross-domain + laundered) is what makes the "unsolved" claim strong.
- PAL, IID-Net — inpainting-specific localizers.

### 3d. MLLM / VLM detectors (close the "just ask GPT-4o" gap)
- Zero-shot: GPT-4o, Gemini-2.x, Qwen-VL prompted as forgery detectors.
- Explainable IFDL: **FakeShield, SIDA, ForgeryGPT** (detect + localize + rationale).
- Expected failure: subtle, plausible small objects → unreliable, high FPR/miss, no reliable pixel mask.

### 3e. Active provenance / watermark (orthogonal — include to dismiss)
- C2PA, SynthID, Stable-Signature. **Not applicable** to the threat: real source has no provenance and the attacker uses non-watermarking editors → motivates passive detection. Report as a "why detection still matters" point, not a beatable baseline.

### 3f. Human / expert baseline
- AMT crowd + domain experts (hotel ops / claims adjusters). If humans fail too, the deception is real and automated detection is necessary.

---

## 4. Comparable benchmarks (scale & design) — and the verdict on 400

| Benchmark | Venue | Scale | Build | Tasks / metrics | Cross-gen |
|---|---|---|---|---|---|
| **GIM** | AAAI'25 | **1.14M** manip+origin (~320K masked core) | SAM mask → SD/GLIDE inpaint (insert), DDNM (remove) on ImageNet/VOC — **same threat model** | det (Cls.acc) + loc (pixel-AUC/F1) | train SD → test GLIDE/DDNM/VOC; **F1 collapses 39%→19%** |
| **OpenSDI** | CVPR'25 | **300K** (OpenSDID) | pixel-preserving inpaint (only masked region), masks via VLM+SAM+Florence-2; 4 VLM instructions | det (F1/Acc) + loc (F1/IoU) | train SD1.5 → test SD2.1/SDXL/SD3/Flux.1 (20K each) |
| **GenImage** | NeurIPS'23 | **~2.68M** (1.33M real + 1.35M fake, 8 gens) | whole-image AIGC | det only | + **degraded task** (↓res, JPEG Q65/30, blur σ3/5) |
| CocoGlide | (TruFor line) | 512 | COCO val + GLIDE inpaint | loc | diffusion-inpaint test set |
| AutoSplice | 2023 | 2,273 real + 3,621 fake | DALL·E2 inpaint; **JPEG QF 75/90/100** | det + loc | built-in JPEG robustness |
| COCO-Inpaint | 2025 | — | multi inpainter | det + loc | inpaint-specific |

**Verdict on ~400 (200 hotel + 200 lodging):** ~2,500× below GenImage's core, 800–2,850× below GIM/OpenSDI. **As a standalone train/test benchmark it will be rejected as too small / single-domain / single-generator.** Two viable paths:

- **Path A (recommended): position the curated set as an evaluation-only "stress-test / hidden eval suite"**, with training drawn from GIM/OpenSDI/synthetic. Then a few hundred *carefully matched* in-domain pairs is defensible — but verify per-editor/per-object F1/IoU are statistically stable (report bootstrap CIs).
- **Path B: scale generation** to 10⁴–10⁵ via the cross product in §5 and present a full train/val/test benchmark.
- Strongest paper = **B for the dataset + A's curated subset as the headline hidden test + leaderboard.**

---

## 5. Scale-up plan (the matrix that turns "a few hundred" into a benchmark)

| Axis | Target | Why |
|---|---|---|
| Domains | 5–6: room / bathroom / restaurant / kitchen / rental(Airbnb) / vehicle-or-retail | cross-domain split; claim-relevant |
| Real source images | 1,000–2,000 licensed/owned w/ provenance | base for matched pairs |
| **Editing models** | **≥5**: HunyuanImage-3 (yours) + SDXL-inpaint + Flux.1-Fill + Qwen-Image-Edit + 1 commercial (GPT-Image/Firefly/Nano-banana) | **cross-generator is the core generalization & evasion axis** (GIM/OpenSDI) |
| Object categories | pests (mouse, cockroach, ant, bedbug, fly), contamination (hair, mold, stain, insect-in-food), damage (crack, water-stain, burn) | claim taxonomy; per-object breakdown |
| Perturbations / laundering | JPEG QF sweep, resize, blur/noise, WeChat/WhatsApp/iMessage round-trip, screenshot, double-compression | GenImage degraded task + INP-X/Fake-or-JPEG |
| **Controls (critical)** | (i) real-only; (ii) **real-with-real-object** (genuine pest/defect photos); (iii) **real-object paste-back** (same splice pipeline, real cutout not AI) | prevents "object-present = fake" shortcut; (iii) decouples *AI-ness* from *manipulation-ness* |

Cross product → easily 10⁴–10⁵ forged + matched reals; sample the hidden test from it.

---

## 6. Metrics & protocols (consensus across IMDL-BenCo / GIM / OpenSDI / GenImage)

- **Image-level (T1):** AUC **and** AP **and** fixed-0.5 Accuracy. *Always report AUC alongside fixed-threshold acc* — calibration work (arXiv:2602.01973) shows fixed-0.5 understates separability; don't overstate "detectors fail" on Acc alone.
- **Pixel-level (T2), scored on forged images only:** **F1 at fixed 0.5 AND optimal threshold**, IoU, MCC, pixel-AP.
- **Combined:** report a detect-then-localize composite (e.g., mean rank, or F1_img × F1_pix). ⚠️ **Open problem:** no standardized cross-family combined metric exists for ranking localization-only IML vs classification-only AIGC — you'll need to define and justify one (see §11).
- **Generalization protocols:** **cross-generator** (train-on-one-editor, test-on-unseen — GIM/OpenSDI) and **cross-domain** (hotel↔restaurant). These are where the story lives.
- **Robustness axes:** JPEG recompression, resize, blur/noise, social-media laundering — report degradation curves.

---

## 7. Best practices & pitfalls (reviewer-killers to pre-empt)

1. **Matched real/fake post-processing** — re-encode the WHOLE frame uniformly after paste-back (don't leave the edited region at a different quantization). Your byte-identical design already neutralizes the JPEG/PNG shortcut — **say so explicitly and audit it** (train a detector on format only; show it can't separate).
2. **Splice-seam confound** — your paste-back seam is itself a forensic signal IML methods can catch. The **real-object paste-back control (§5-iii)** disentangles "detected AI texture" from "detected splice." Mandatory for a clean claim.
3. **Single-generator overfit** — multi-editor + report cross-generator (collapse is the expected, publishable result).
4. **Region-prior leakage** — randomize insert size/location so models can't learn "where the box is."
5. **Shortcut audit** — explicit bias probes (format, size, JPEG-grid, saturation) as an ablation table.
6. **Licensing/ethics** — owned/CC/consented source photos + datasheet; gated release; explicit "research-only, not for real claims" terms. (Open question §11: GIM/OpenSDI/GenImage don't detail consumer-claim-photo provenance — you must define yours.)

---

## 8. Evasion / attack track (the second half of the paper)

- **Primary evasion result:** replicate INP-X logic in-domain — show detector AUC collapses on pixel-preserving local edits vs whole-image generation.
- **Evasion rate** = fraction of forgeries scored "real" at a fixed low FPR (e.g., 1%/5%).
- **Laundering pipeline:** re-JPEG, resize, social round-trip, re-inpaint with a 2nd model.
- **Adaptive attacker (upper bound):** per-image, pick the editor + post-processing that minimizes the strongest detector's score.
- Report which axis most degrades the **strongest 2025 localizers** (MaskCLIP, GIMFormer, Mesorch) on local insertions.

---

## 9. Splits & leaderboard

- Public **train/val** + **hidden test** with public leaderboard.
- Hold out **≥1 generator** and **≥1 domain** entirely for zero-shot generalization.
- Two leaderboards (Detection, Localization) + combined ranking.

---

## 10. Ablations & analyses a top-venue reviewer will expect

Per-editor / per-object / per-domain breakdowns · object-size vs detectability curve · JPEG-QF & resize robustness curves · cross-dataset zero-shot (existing IML/AIGC weights → CLAIMFORGE) · shortcut/bias audit table · **human perceptual study** (can people/experts spot the fakes — establishes the deception is real) · the real-object-paste-back control result.

---

## 11. Decisions you need to make (open questions from the sweep)

1. **400 = hidden eval suite (Path A) or scale to 10⁴–10⁵ (Path B)?** Recommendation: do B, headline a curated A subset.
2. **Combined cross-family metric** — not standardized; we must define/justify one (proposal: report both leaderboards separately + a mean-reciprocal-rank composite).
3. **Ethics/licensing of consumer-claim photos** — provenance/consent framework (no platform scraping of review photos); define before release.
4. **Calibration** — commit to reporting AUC + fixed-threshold acc (don't overstate failure).

---

## 13. Making the "no existing method solves CLAIMFORGE" claim bulletproof

To claim universality of failure, do three things:

1. **Cover every paradigm** with ≥1 strong open-source representative (the six families in §3a–3f). A reviewer must not be able to say "you didn't try category X."
2. **Define a concrete "solved" bar** up front, e.g.: a method *solves* CLAIMFORGE if it reaches **image-AUC ≥ 0.90 AND pixel-F1 ≥ 0.50 on the cross-generator + laundered hidden test.** Show all paradigms fall well below it.
3. **Report three escalating regimes** (each fairer to the defender) so the claim survives "but if you train on it…":
   - (i) **zero-shot** — off-the-shelf weights.
   - (ii) **fine-tuned on CLAIMFORGE-train, tested in-distribution** — establishes the upper bound when the defender has your data.
   - (iii) **fine-tuned, tested on held-out editor + held-out domain + laundering** — the **headline "still unsolved" number.**
   If even a method fine-tuned on your data fails to generalize in regime (iii), the claim holds.

**Don't over-claim:** phrase as "no existing method, across classical forensics, AIGC detection, diffusion-inpainting localization, MLLM detectors, and provenance, reaches [the bar] under realistic cross-generator/laundered conditions" — not a literal "no method." Pair AUC with fixed-threshold accuracy (calibration caveat, arXiv:2602.01973) so the failure isn't an artifact of a bad threshold.

**Per-family failure mechanism (state the *why*, not just the number):**
- Classical IML — pixel-preserving + uniform re-encode removes double-JPEG/noise-boundary/splice cues; local generative content lacks classic tamper statistics (GIM: TruFor → 5.7% F1 cross-gen).
- Whole-image AIGC — 99% real pixels → no global artifact (INP-X 94→55%); and no native localization.
- Diffusion-inpaint localizers — overfit to training editors/domains → break under unseen editor + lodging domain + laundering.
- MLLM — small plausible objects → unstable judgments, no reliable mask.
- Provenance — no signal exists to verify.

## 14. Proposed solutions / directions (propose, prototype, leave headroom)

A benchmark that also proposes solutions is stronger — **but** propose directions that *meaningfully improve yet do not fully close the gap*, keeping the benchmark open while showing tractability. Ranked by fit to the pixel-preserving threat model:

1. **Intra-image self-consistency on the camera fingerprint (PRNU / Noiseprint++ inconsistency) — headline proposal, uniquely enabled here.** The whole frame except the patch is the genuine photo → an *in-image reference* for the sensor-noise model; the inserted region is a statistical outlier, localized as a fingerprint anomaly. Caveat: heavy laundering attenuates sensor noise → report a robustness curve.
2. **Dense diffusion-reconstruction localization (DIRE → pixel-level).** Per-patch reconstruction error under a diffusion model; locally over-reconstructable regions are suspect → yields a mask.
3. **Generalization via diverse-editor + laundering-augmented training** of a strong localizer (TruFor / Mesorch / MaskCLIP). Data-centric; show it *narrows but does not close* the cross-generator gap.
4. **Cross-paradigm ensemble + localization head** — fuse noise + frequency + CLIP-semantic + reconstruction cues (no single cue suffices).
5. **MLLM forensic-reasoning agent** — VLM reasons about physical plausibility (lighting/shadow/scale) and calls forensic tools; explainable rationale suited to claims adjudication. Future direction.
6. **Capture-time provenance (C2PA / secure camera)** — orthogonal long-term fix; acknowledge the ceiling of passive detection.

**Recommendation:** actually build **#1 + #3 (optionally #4)** in the paper; present #5–#6 as discussion. The honest result — "our strongest proposed method lifts held-out pixel-F1 from X to Y but remains below the solved bar" — is exactly what a benchmark+attack paper wants.

## 12. Key references (verified)
- IMDL-BenCo, NeurIPS'24 — arXiv:2406.10580 · github.com/scu-zjz/IMDLBenCo
- GIM, AAAI'25 — arXiv:2406.16531
- OpenSDI, CVPR'25 — arXiv:2503.19653 · github.com/iamwangyabin/OpenSDI
- GenImage, NeurIPS'23 — arXiv:2306.08571
- TruFor, CVPR'23 — arXiv:2212.10957 · github.com/grip-unina/TruFor
- Mesorch, AAAI'25 — arXiv:2412.13753
- SAFIRE, AAAI'25 — arXiv:2412.08197 · github.com/mjkwon2021/SAFIRE
- HiFi-IFDL, CVPR'23 — arXiv:2303.17111 · github.com/CHELSEA234/HiFi_IFDL
- INP-X "Inpainting Exchange," 2026 — arXiv:2602.00192
- "Fake or JPEG?," 2024 — arXiv:2403.17608
- AIGC-detector failure benchmark, 2026 — arXiv:2602.07814 · calibration caveat arXiv:2602.01973
- AIGC baselines: CNNDetection CVPR'20 (arXiv:2104.02984 line), DIRE ICCV'23 (arXiv:2303.09295), C2P-CLIP (arXiv:2408.09647)
- Inpaint datasets: CocoGlide (TruFor), AutoSplice, COCO-Inpaint (arXiv:2504.18361)

*Data corrections carried forward: TruFor Coverage pixel-F1 = 0.4573 (not 0.5573); CAT-Net (0.9150) beats Mesorch on Columbia → Mesorch is best on 3/4 std sets + on average, not uniformly.*
