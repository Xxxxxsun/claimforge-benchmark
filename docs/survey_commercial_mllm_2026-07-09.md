# Commercial & MLLM Baselines for AI-Inpainting Detection — Research Report (as of 2026-07-09)

Method: 5 parallel search tracks with primary-source fetches (vendor docs/pricing pages fetched live, live API probes for Illuminarty, arXiv API verification of every cited paper), followed by an independent verification pass. Confidence flagged where a claim rests on a single or secondary source.

> **Commercial API availability update — 2026-07-21.** The 2026-07-09 observations below remain a historical snapshot, but Illuminarty is no longer an executable baseline. Hive V3, Alibaba Cloud Ultra, AI or Not, Resemble Detect, and Copyleaks Ultra have since completed authenticated runs on all 275 mouse forgeries. Hive detected 0/275 at its 0.9 threshold; Copyleaks detected 111/275, with 164 empty-mask misses remaining explicit. Full endpoints, results, costs, and execution rules are recorded in `COMMERCIAL_API_STATUS_2026-07-20.md`.

## 1. Sightengine GenAI detection (integrated)

- **Output: whole-image score only, no localization.** The `genai` model returns `type.ai_generated` (0–1) plus optional per-generator scores (`type.ai_generators`: dalle, firefly, flux, gan, gpt, midjourney, imagen, nano-banana class, qwen, seedream, stable_diffusion, etc.). No regions, masks, or heatmaps anywhere in the documented schema. Docs: https://sightengine.com/docs/ai-generated-image-detection
- **AI-edit/inpainting coverage: claimed, but score-level only.** The FAQ defines the score as confidence the image was "produced **or edited** by a generative AI model," and the changelog (https://sightengine.com/docs/changelog) records a **May 2026 "major upgrade": "support for AI edit detection (image-to-image edits)"** — an upgrade to the existing genai model, not a separate inpainting model, and still no localization. Detection is purely pixel-based (C2PA/EXIF/watermarks ignored). A separate `deepfake` model covers face swaps only (single `type.deepfake` score).
- **Pricing:** Free 2,000 ops/month (500/day); Starter $29/mo = 10k ops; Pro $99/mo = 40k ops; overage $0.002/op. https://sightengine.com/pricing
- **Independent evidence:** ranked #1 commercial detector on fully-generated images in the ARIA study (98.3% acc; Li et al. 2024, per https://sightengine.com/best-ai-image-detectors-benchmark); but see Section 6 — on inpainted images with global artifacts removed it collapses to ~55% acc / 0.12 recall (INP-X, tested ~Jan 2026, i.e., pre-May-2026 upgrade — worth noting in the paper).

## 2. Illuminarty (historical; unavailable as of 2026-07-20)

- **Status changed after the original survey.** The site, webapp backend, and classify API responded to probes on 2026-07-09. On 2026-07-20, the official webapp displayed `Service currently not available: Cannot connect to server`, and no valid inference could be obtained. This supports an operational `unavailable` label, not a claim about the company's staff or cause of failure. https://app.illuminarty.ai/
- **The historical localization path is not a stable public API.** The public API never documented a localization endpoint; the webapp used an internal response field rendered as coarse points. Because the service is now unavailable, this interface cannot support a reproducible benchmark and must not receive further budget or adapter work.
- **Prior literature remains citable.** Ha et al. CCS'24 (arXiv:2402.03214) reported 72.65% accuracy with a 67.4% false-positive rate on human art; arXiv:2411.13553 reported 0.80 accuracy dropping to 0.57 under Gaussian noise. These are historical vendor results, not evidence of current availability.

## 3. Other commercial detectors (status verified July 2026)

| Service | 2026-07-20 access | Output granularity | CLAIMFORGE role / caveat |
|---|---|---|---|
| **Copyleaks Ultra** | Self-serve `POST /v1/ai-image-detector/{scanId}/check`; 1 credit/image in authenticated runs | Image verdict, AI-pixel fraction, and native binary RLE mask | 275/275 mouse forgeries complete: 111/275 detected; positive-only mean IoU 0.8165 versus exact-difference GT, but all-image mean IoU is 0.3296 after empty-mask misses |
| **Hive AI** | Self-serve V3; $6/1k; default 100 requests/day (https://thehive.ai/pricing) | Whole-image AI/Human + generator attribution (including Hunyuan and inpainting sources) + deepfake score | 275/275 forged run complete with 0/275 above the 0.9 vendor threshold; highest-priority literature-comparable T1 baseline, no localization |
| **Resemble Detect** | Self-serve Flex; `POST /api/v2/detect`; listed at $0.04/second for images (https://www.resemble.ai/pricing) | Whole-image fake/real score plus optional `ifl.heatmap` visualization | Closest active localization candidate; heatmap semantics and static-image billing require paired preflight |
| **Alibaba Cloud Ultra** | China (Beijing) `aigcDetector_ultra` validated; CNY 200/10k | Thresholded whole-image `risk_aigc`, `risk_fake`, and explicit `risk_edit` labels | 275/275 valid on mouse forgeries; 30 `risk_edit` plus 1 `risk_fake`, for 31/275 any-risk detections |
| **AI or Not** | Self-serve `POST /v2/image/sync`; authenticated 275/275 forged run complete | Whole-image AI/Human verdict and continuous confidence; no general edit localization | Only 4/275 mouse forgeries detected (1.45%); 5-pair pilot scores were nearly unchanged by editing |
| **Reality Defender** | Self-serve API; free 50 image/audio scans/month | Ensemble authenticity/deepfake status and scores; no public general-edit mask | Authenticated forged-50 pilot: 50/50 applicable but 50/50 `AUTHENTIC`, normalized scores 0.01–0.03 |
| **Sensity** | API/auth online; developer account is contact-sales | Whole-image AI-generated/deepfake analysis; some vendor visualizations | Apply for trial in parallel; no public price and continuous-score behavior needs confirmation |
| **Winston AI** | Self-serve developer API; 2,000 starter credits | Basic whole-image AI/Human score; advanced forensic report | Backup only: image API requires a public URL and forensic visualization is not yet validated as an edit map |
| **Google AI Content Detection** | Private Preview; application required | Detects generated or modified images; no public localization schema | Watchlist, not an immediately executable baseline |

**Cross-cutting finding:** most commercial APIs still expose only whole-image scores. Copyleaks is the first verified exception in this study: across all 275 forgeries it detected 111, whose native RLE masks achieved mean IoU 0.8165 against exact SP differences. However, counting 164 empty-mask misses reduces all-image mean IoU to 0.3296, so localization quality must never be reported without detection coverage. Resemble's optional image heatmap did not yield a usable local-edit map. Alibaba's edit-specific label is operational but thresholded and image-level: 30/275 mouse forgeries triggered `risk_edit`, while non-hits returned `nonLabel` without a continuous score.

## 4. Frontier MLLMs as zero-shot forgery detectors

**Landscape (July 9, 2026):** OpenAI **GPT-5.5 / GPT-5.5 Pro** is the GA API flagship (`gpt-5.5-2026-04-23`, $5/$30 per Mtok; https://openai.com/index/introducing-gpt-5-5/), with the GPT-5.6 family (Sol/Terra/Luna) launching publicly this week (https://openai.com/index/previewing-gpt-5-6-sol/). Google's GA flagship is **Gemini 3.1 Pro** (Feb 2026; https://deepmind.google/models/model-cards/gemini-3-1-pro/), with Gemini 3.5 Pro announced at I/O but not yet GA. Anthropic's current lineup is **Claude Opus 4.8** (May 2026; https://www.anthropic.com/news/claude-opus-4-8) plus the Fable 5 tier (June 2026). Open-weight VL flagship: **Qwen3.5-397B-A17B** (natively multimodal, Feb 2026; https://qwen.ai/blog?id=qwen3.5), superseding Qwen3-VL/Qwen2.5-VL.

**Citable evidence (all arXiv-verified):**
- **FakeBench** (arXiv:2404.13306): GPT-4V best at ~78% detection on fully-generated fakes; strong "call it real" bias; CoT didn't help.
- **LOKI** (arXiv:2410.09732, ICLR'25 Spotlight): best LMM ~64% on real/synthetic judgment vs ~50% chance and ~76–86% humans.
- **Forensics-Bench** (arXiv:2503.15024, CVPR'25): best of 25 LVLMs ~67% over 112 forgery types; proprietary models underperform open LLaVA variants; LVLMs fail hardest at **localization**.
- **FragFake — "Can VLMs Detect and Localize Fine-Grained AI-Edited Images?"** (arXiv:2505.15644): the most on-point citation — abstract states "pretrained VLMs, including GPT4o, **perform poorly**" at edited-image classification and edited-region localization, while fine-tuned Qwen2.5-VL succeeds; also reports Hive flagging only 55/100 partially-edited images.
- **SHIELD** (NeurIPS 2025, https://openreview.net/forum?id=hEZPVTDXCy): first systematic zero-shot benchmark of 24 VLMs on AI-edited real photos; direct prompting + greedy decoding beats CoT (numbers unverified — OpenReview PDF unfetched).
- **"LLMs Are Not Yet Ready for Deepfake Image Detection"** (arXiv:2506.10474): zero-shot GPT-4o/Claude/Gemini/Grok "not yet dependable as standalone detection systems"; Gemini's real-image accuracy driven by a bias toward predicting "real"; vintage aesthetics act as false authenticity cues.
- **2026:** arXiv:2511.13442 scores vanilla **GPT-5** at 1.26/5 forensic explanation accuracy (vs 4.94/5 readability) on editing-forgery data; arXiv:2603.12930 finds VLM priors "hardly benefit" forgery detection/localization due to semantic-plausibility bias. No published Gemini-3 or Claude-Fable forgery numbers yet — your paper would be among the first.

**Synthesis:** frontier MLLMs are modestly above chance on fully-generated images (~64–78%) but degrade toward chance on local edits, with a systematic bias toward "real" — motivating the 2025–26 wave of fine-tuned specialist frameworks (FakeShield arXiv:2410.02761, SIDA arXiv:2412.04292, LEGION arXiv:2503.15264, So-Fake arXiv:2505.18660).

## 5. Provenance / watermarking (one paragraph to cite as orthogonal)

C2PA Content Credentials (spec now at v2.3/2.4, with a conformance program) has broad industry backing — Adobe, Microsoft, Google, and OpenAI on the steering committee; native capture support in Leica, Sony, Nikon, Canon, and Pixel 10 cameras; display on TikTok and LinkedIn — and Google reports SynthID has watermarked **100B+ images/videos**, with verification built into the Gemini app (50M uses), rolling into Search and Chrome, and a Cloud detection API (May 2026: https://blog.google/innovation-and-ai/products/identifying-ai-generated-media-online/); OpenAI ships C2PA in DALL-E 3/gpt-image/Sora outputs, adopted SynthID in May 2026, and runs a verify portal (https://openai.com/index/advancing-content-provenance/), while Meta's C2PA/IPTC-driven "AI info" labels persist on Facebook/Instagram. However, provenance is opt-in on the generator side and metadata is trivially stripped by re-encoding, screenshots, and most social-platform pipelines (acknowledged by OpenAI itself) — so it is orthogonal to, and no substitute for, pixel-based detection of adversarial local edits.

## 6. Prior academic evidence: commercial detectors on local edits

- **The key citation — INP-X** (arXiv:2602.00192, Jan 2026, "AI-Generated Image Detectors Overrely on Global Artifacts: Evidence from Inpainting Exchange"; verified verbatim): 90K-image benchmark; tested 11 academic detectors + **Hive and Sightengine**. When original pixels are restored outside the edit mask (isolating truly local content), commercial detectors "exhibit a dramatic drop in accuracy (e.g., from 91% to 55%), frequently approaching chance level" — Sightengine 0.926 → 0.550 acc, recall 0.12. Direct prior evidence that commercial detectors key on global VAE artifacts, not local edits. Code: https://github.com/emirhanbilgic/INP-X
- **VendorBench-100** (arXiv:2607.06254, published 2026-07-07; verified): first unified cross-paradigm comparison — commercial APIs (Reality Defender, Hive, Sightengine, TruthScan, Neural Defend) vs zero-shot VLMs vs open detectors, on 100 adversarial images incl. an **"AI photo edits"** family; commercial APIs win on median but show a consistent AUC-vs-MCC (threshold) divergence. Tiny corpus — cite as concurrent work.
- **Deepfake-Eval-2024** (arXiv:2503.02857): 22 commercial models (Hive, Reality Defender, AI or Not, Sensity, …; results anonymized) on in-the-wild fakes; commercial > open-source but below forensic analysts; largest error increases on selective/non-facial manipulations.
- Supporting: Ha et al. (arXiv:2402.03214, Hive/Optic/Illuminarty on generated art), arXiv:2411.13553 (Hive/Illuminarty fragile to perturbations), RAID (arXiv:2506.03988, adversarial transfer to Hive/Sightengine), TGIF/TGIF2 (arXiv:2407.11566, 2603.28613 — academic localizers fail on regenerative inpainting). **Gap your paper fills:** current commercial APIs have not been publicly evaluated as a paired set on tiny, non-facial, evidence-photo object insertions; their advertised edit scores and visualizations also lack quantitative localization validation in this setting.

## 7. Recommended baseline set

**Commercial active roster:**
1. **Copyleaks Ultra** — completed 275-image forged-only T1+T2 baseline; retain both positive-only and all-image localization metrics, and expand paired real controls separately.
2. **Sightengine genai** — retained core T1 baseline; the 199-image 2026-07-20/21 result is forged-only original-PNG pilot evidence, while the canonical paired run remains required.
3. **Hive AI** — 275-image forged-only run complete with 0/275 detected at the 0.9 threshold; strongest literature comparability, while a full paired real run remains required.
4. **Resemble Detect** — retain T1, but do not use its current visualization as a GT-compatible T2 mask.
5. **Alibaba Cloud Ultra** — validated edit-specific T1 baseline via China (Beijing) `risk_edit`; 275/275 forged run complete, with 30 `risk_edit` and one additional `risk_fake` detection.
6. **AI or Not** — validated low-cost whole-image baseline; 275/275 forged run complete with 4/275 positive verdicts and no API errors.
7. **Reality Defender** — forged-50 coverage pilot is complete and 100% applicable, but all 50 edits were labeled `AUTHENTIC`; retain as a T1 prefix result and add real controls only with more quota.

**Retired:** Illuminarty is unavailable as of 2026-07-20 and no longer counts toward the active commercial baseline total.

**MLLMs (3):**
1. **GPT-5.5** (or GPT-5.6 Sol once GA this week) — OpenAI flagship; extends the GPT-4V/4o/5 lineage used in FakeBench, LOKI, and arXiv:2511.13442, giving longitudinal comparability.
2. **Gemini 3.1 Pro** — Google's GA flagship; prior Gemini versions show a documented "call it real" bias (arXiv:2506.10474) worth re-testing on local edits.
3. **Qwen3.5-397B-A17B** (open-weight) — successor to Qwen2.5-VL, the model that fine-tunes successfully on FragFake; an open-weight baseline enables reproducibility and a zero-shot-vs-fine-tuned contrast. (Claude Opus 4.8 is a reasonable optional fourth if budget allows.)
