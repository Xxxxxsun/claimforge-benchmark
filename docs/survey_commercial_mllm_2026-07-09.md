# Commercial & MLLM Baselines for AI-Inpainting Detection — Research Report (as of 2026-07-09)

Method: 5 parallel search tracks with primary-source fetches (vendor docs/pricing pages fetched live, live API probes for Illuminarty, arXiv API verification of every cited paper), followed by an independent verification pass. Confidence flagged where a claim rests on a single or secondary source.

## 1. Sightengine GenAI detection (integrated)

- **Output: whole-image score only, no localization.** The `genai` model returns `type.ai_generated` (0–1) plus optional per-generator scores (`type.ai_generators`: dalle, firefly, flux, gan, gpt, midjourney, imagen, nano-banana class, qwen, seedream, stable_diffusion, etc.). No regions, masks, or heatmaps anywhere in the documented schema. Docs: https://sightengine.com/docs/ai-generated-image-detection
- **AI-edit/inpainting coverage: claimed, but score-level only.** The FAQ defines the score as confidence the image was "produced **or edited** by a generative AI model," and the changelog (https://sightengine.com/docs/changelog) records a **May 2026 "major upgrade": "support for AI edit detection (image-to-image edits)"** — an upgrade to the existing genai model, not a separate inpainting model, and still no localization. Detection is purely pixel-based (C2PA/EXIF/watermarks ignored). A separate `deepfake` model covers face swaps only (single `type.deepfake` score).
- **Pricing:** Free 2,000 ops/month (500/day); Starter $29/mo = 10k ops; Pro $99/mo = 40k ops; overage $0.002/op. https://sightengine.com/pricing
- **Independent evidence:** ranked #1 commercial detector on fully-generated images in the ARIA study (98.3% acc; Li et al. 2024, per https://sightengine.com/best-ai-image-detectors-benchmark); but see Section 6 — on inpainted images with global artifacts removed it collapses to ~55% acc / 0.12 recall (INP-X, tested ~Jan 2026, i.e., pre-May-2026 upgrade — worth noting in the paper).

## 2. Illuminarty (integrated)

- **Alive and operational** (site, webapp v1.4 backend, and API all responded to live probes on 2026-07-09). `POST https://api.illuminarty.ai/v1/image/classify` exists and enforces auth (`X-API-Key` header, multipart `file`); returns an AI probability. https://illuminarty.ai/en/
- **Localization: real feature, but not a documented public-API endpoint.** Probes confirm there is **no** `/v1/image/localize` (404, along with ~10 variants). Localization ("we denote the specific regions… that increased the likelihood of AI generation", a paid Basic-plan feature, $11/mo live price) is served through the webapp's internal endpoint as `details.localization` — an array of [x, y] points rendered client-side as highlighted circular regions, not a server heatmap image. Whether the paid v1 classify response includes this field is **unverified** (no public schema doc); if your integration receives it, that is worth stating explicitly in the paper since it is undocumented. Plans: Free (classification), Basic $10–11/mo (localization + API, 10k req/day), Pro $30–33/mo.
- **Reputation:** a common academic commercial baseline, consistently the weakest one: Ha et al. CCS'24 (arXiv:2402.03214) report 72.65% acc with a **67.4% false-positive rate on human art**; arXiv:2411.13553 reports 0.80 acc dropping to 0.57 under Gaussian noise; Gold Penguin (June 2026) found it missed 3/7 AI images. Its localization makes it uniquely relevant to a local-edit benchmark despite weak classification.

## 3. Other commercial detectors (status verified July 2026)

| Service | Alive? | API | Output granularity | Targets |
|---|---|---|---|---|
| **Hive AI** (thehive.ai) | Yes | Yes, self-serve, $6/1k images (https://docs.thehive.ai/docs/ai-image-and-video-detection) | Whole-image only ("determines whether or not the input is **entirely AI-generated**") + ~100-class generator attribution (incl. `sdxlinpaint`, `stablediffusioninpaint`) + per-face deepfake score | Fully-generated + deepfake faces |
| **AI or Not** (Optic, aiornot.com) | Yes | Yes (`api.aiornot.com/v2/image/sync`, https://docs.aiornot.com) | Whole-image verdict + generator attribution; face bboxes (`rois[]`) for deepfakes only — no edit-region localization | Generated + "tampered by generative-AI"; deepfakes |
| **Reality Defender** | Yes (Gartner "Market Shaper" 2026) | Yes — public API + SDKs since Aug 2025 (https://docs.realitydefender.com); Free 50 scans/mo, Business $399/mo | Whole-file `overallStatus` + ensemble score + per-model scores; "highlights where altered" is marketing/UI, not in API schema | Deepfake/impersonation across image/audio/video |
| **Winston AI** (gowinston.ai) | Yes | Yes (`POST /v2/image-detection`, credit-based) | Whole-image human/AI score; web-only forensic visualizations (ELA, noise, edge heatmap) not confirmed in API | Fully-generated images |
| **Decopy AI** | Yes | **No API found** (web tool + extension only) | Whole-image | Fully-generated |
| **IsItAI** | Yes | Yes (API in all plans, incl. free 5/mo) | Whole-image + generator guess | Fully-generated; mixed Trustpilot reviews |
| **TrueMedia.org** | **Shut down Jan 14, 2025** (https://www.truemedia.org/post/shutting-down-truemedia); code open-sourced (github.com/truemediaorg); relaunched 2026 as a Georgetown University academic project in closed beta | No commercial API | — | Political deepfakes |
| **Deepware** (deepware.ai) | Site alive, activity stale (flag) | Contact-gated API | Per-video score | **Video-only** deepfake faces — irrelevant for images |

**Cross-cutting finding:** none of the eight documents pixel/region-level localization of AI-edited or inpainted areas; the ecosystem is built around whole-image "entirely AI-generated" verdicts, which is precisely the assumption your benchmark's local-insertion edits violate.

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
- Supporting: Ha et al. (arXiv:2402.03214, Hive/Optic/Illuminarty on generated art), arXiv:2411.13553 (Hive/Illuminarty fragile to perturbations), RAID (arXiv:2506.03988, adversarial transfer to Hive/Sightengine), TGIF/TGIF2 (arXiv:2407.11566, 2603.28613 — academic localizers fail on regenerative inpainting). **Gap your paper fills:** no existing work evaluates Illuminarty or AI-or-Not on inpainted images, and none does so in a realistic fraud scenario (evidence-photo object insertion).

## 7. Recommended baseline set

**Commercial (4):**
1. **Sightengine genai** (integrated) — top-ranked commercial detector on fully-generated images, now advertising AI-edit detection (May 2026), yet shown to collapse on isolated local edits (INP-X) — the perfect "strong global detector" baseline.
2. **Illuminarty** (integrated) — the only commercial service with region-level localization output, directly comparable to your localization task; known-weak classifier makes it a useful lower anchor.
3. **Hive AI** (add) — the field's de-facto commercial SOTA baseline (Ha et al., ImageDetectBench, RAID, INP-X, FragFake all use it), self-serve API at $6/1k; pairing it with Sightengine reproduces INP-X's exact commercial pair on your data.
4. **AI or Not** (optional add) — cheap self-serve API with generator attribution and explicit "tampered by generative-AI" claim; appears in Deepfake-Eval-2024's vendor pool but has never been tested on inpainting. (Reality Defender is the alternative, but its API is deepfake/impersonation-oriented and pricing is less benchmark-friendly.)

**MLLMs (3):**
1. **GPT-5.5** (or GPT-5.6 Sol once GA this week) — OpenAI flagship; extends the GPT-4V/4o/5 lineage used in FakeBench, LOKI, and arXiv:2511.13442, giving longitudinal comparability.
2. **Gemini 3.1 Pro** — Google's GA flagship; prior Gemini versions show a documented "call it real" bias (arXiv:2506.10474) worth re-testing on local edits.
3. **Qwen3.5-397B-A17B** (open-weight) — successor to Qwen2.5-VL, the model that fine-tunes successfully on FragFake; an open-weight baseline enables reproducibility and a zero-shot-vs-fine-tuned contrast. (Claude Opus 4.8 is a reasonable optional fourth if budget allows.)
