# CLAIMFORGE Zero-Shot Detector Baseline Survey — Availability Verified as of 2026-07-09

**Method:** five parallel research passes (one per family) fetching primary sources (GitHub READMEs via raw.githubusercontent.com, GitHub API metadata, HuggingFace API, arXiv, docs sites), plus an independent second-pass verification of every load-bearing "weights available" claim for the recommended suite. Facts below marked "verified" were read directly from the repo/HF source. GDrive = Google Drive.

**Task fit reminder:** CLAIMFORGE forgeries are *local diffusion edits spliced back into an otherwise byte-identical real JPEG*. Family A/C localizers are the scientifically matched tools (T1+T2); Family B whole-image AIGC detectors are trained on *fully synthetic* images and are expected to struggle on a 2% inserted object — they serve as an informative control group for T1.

---

## Family A — Image Manipulation Localization (IML)

| Method | Venue / arXiv | GitHub | License | Pretrained weights | Output | Inference ease |
|---|---|---|---|---|---|---|
| **TruFor** | CVPR'23 / [2212.10957](https://arxiv.org/abs/2212.10957) | [grip-unina/TruFor](https://github.com/grip-unina/TruFor) | Custom, **non-commercial research only** | **YES** — auto-downloaded from [grip.unina.it](https://www.grip.unina.it/download/prog/TruFor/TruFor_weights.zip) (MD5 published) | mask + confidence map + image score | **Excellent**: Docker inference-only pipeline, single file/dir/glob input, CPU fallback, `visualize.py`. Maintained (2025 pushes) |
| **MVSS-Net** | ICCV'21 / [2104.06832](https://arxiv.org/abs/2104.06832) | [dong03/MVSS-Net](https://github.com/dong03/MVSS-Net) | **None** (all rights reserved) | **YES** — GDrive + Baidu (CASIAv2 & DEFACTO ckpts) | mask + image score | Painful standalone: Python 3.6, CUDA 10.1, NVIDIA apex. **Use via IMDL-BenCo instead** |
| **CAT-Net v2** | WACV'21/IJCV'22 / [2108.12947](https://arxiv.org/abs/2108.12947) | [mjkwon2021/CAT-Net](https://github.com/mjkwon2021/CAT-Net) | None (HRNet license only) | **YES** — GDrive + Baidu (`CAT_full_v1/v2.pth.tar`) | pixel heatmap (no score head) | Dated: Py3.6/PyTorch 1.1; inference = drop images in `input/`, edit `tools/infer.py`. DCT/JPEG stream is well-matched to splice-back JPEG forensics. Also in BenCo |
| **PSCC-Net** | TCSVT'22 / [2103.10596](https://arxiv.org/abs/2103.10596) | [proteus1991/PSCC-Net](https://github.com/proteus1991/PSCC-Net) | MIT | **YES — weights committed in the repo** (verified via git tree) | mask + image score | Easy: `python test.py` on a `sample/` folder; plain PyTorch/HRNet. Stale since 2023. Also in BenCo |
| **IML-ViT** | arXiv [2307.14863](https://arxiv.org/abs/2307.14863) (no verified venue; often cited as AAAI'24) | [SunnyHaze/IML-ViT](https://github.com/SunnyHaze/IML-ViT) | MIT | **YES** — GDrive + Baidu (CASIAv2 + CAT-Net-protocol ckpts) | pixel mask only | Very good: Colab demo + `Demo.ipynb`; plain requirements. Also in BenCo |
| **Mesorch** | AAAI'25 / [2412.13753](https://arxiv.org/abs/2412.13753) | [scu-zjz/Mesorch](https://github.com/scu-zjz/Mesorch) | MIT | **YES** — GDrive + Baidu (`mesorch-98.pth`, `mesorch_p-118.pth`) | pixel mask | Good: modern `pip install imdlbenco` stack (Py3.10, no mmcv); test scripts are dataset-shaped, not single-image. Maintained (June 2026) |
| **SAFIRE** | AAAI'25 / [2412.08197](https://arxiv.org/abs/2412.08197) | [mjkwon2021/SAFIRE](https://github.com/mjkwon2021/SAFIRE) | **None** | **YES** — GDrive only (`safire.pth`) | pixel heatmap + multi-source partitioning | Very good for arbitrary images: drop into `ForensicsEval/inputs`, `python infer_binary.py`. SAM-based (heavier GPU). Not in BenCo |
| **HiFi-Net (HiFi-IFDL)** | CVPR'23 / [2303.17111](https://arxiv.org/abs/2303.17111) | [CHELSEA234/HiFi_IFDL](https://github.com/CHELSEA234/HiFi_IFDL) | MIT | **YES** — GDrive, but **two separate task-specific weight sets** (easy to mix up) | image score + mask (+ forgery attribution) | Good API (`HiFi.detect(img)` / `HiFi.localize(img)`); aging Py3.7 env. Community reproducibility grumbles unverified |
| **IMDL-BenCo** (harness) | NeurIPS'24 D&B Spotlight / [2406.10580](https://arxiv.org/abs/2406.10580) | [scu-zjz/IMDLBenCo](https://github.com/scu-zjz/IMDLBenCo) | CC-BY-4.0 | **YES** — paper-metric checkpoints on **Baidu AND [GDrive](https://drive.google.com/drive/folders/1DCqc016-N4YvoMKKA87bFtrCdPVIDxAp)** (verified in [docs](https://scu-zjz.github.io/IMDLBenCo-doc/guide/quickstart/2_load_ckpt.html)) | mask + image metrics per model | pip `imdlbenco` v0.1.45 (June 2026), actively maintained. `benco init model_zoo` scaffolds test scripts; arbitrary-image inference exists (**Case Four** `test_save_images.py`) but is dataset-folder-shaped, **no single-image CLI** |

**IMDL-BenCo model zoo (verified from `model_zoo/__init__.py`):** IML-ViT, CAT-Net, MantraNet, MVSS-Net, ObjectFormer, PSCC-Net, SPAN, TruFor, Mesorch, SparseViT, MSCDI-Net. Checkpoints are stripped `model` state-dicts (e.g., `iml_vit_casiav2.pth`), i.e., mostly CASIAv2/CAT-protocol-trained — exactly what you want for zero-shot cross-domain testing. Caveats: per-model coverage of the Drive folder not enumerable from this sandbox; README warns to use ≥ v0.1.28 (image-level-metric bug before that) and that its "CAT protocol" differs from original CAT-Net settings.

---

## Family B — Whole-Image AIGC Detectors (all image-level score only, no mask)

| Method | Venue / arXiv | GitHub | Weights | Ease / notes |
|---|---|---|---|---|
| **CNNDetection** | CVPR'20 / [1912.11035](https://arxiv.org/abs/1912.11035) | [peterwang512/CNNDetection](https://github.com/peterwang512/CNNDetection) | **YES** — Dropbox via `weights/download_weights.sh` (verified script) | **Trivial**: `python demo.py -f img.png -m weights/blur_jpg_prob0.5.pth`. Historical anchor |
| **DIRE** | ICCV'23 / [2303.09295](https://arxiv.org/abs/2303.09295) | [ZhendongWang6/DIRE](https://github.com/ZhendongWang6/DIRE) | **YES, but** Baidu (pwd `dire`) + [USTC RecDrive](https://rec.ustc.edu.cn/share/ec980150-4615-11ee-be0a-eb822f25e070) mirror only — no GDrive/HF | `demo.py` exists, but inference requires **full ADM/DDIM reconstruction per image** (heavy: guided-diffusion weights + inversion). Flag: expensive |
| **UniversalFakeDetect** | CVPR'23 / [2302.10174](https://arxiv.org/abs/2302.10174) | [WisconsinAIVision/UniversalFakeDetect](https://github.com/WisconsinAIVision/UniversalFakeDetect) | **YES** — linear-probe `pretrained_weights/fc_weights.pth` shipped with repo; CLIP ViT-L/14 auto-fetched | Easy: `validate.py --arch=CLIP:ViT-L/14 --ckpt=...`. Canonical CLIP-probe baseline |
| **NPR** | CVPR'24 / [2312.10461](https://arxiv.org/abs/2312.10461) | [chuangchuangtan/NPR-DeepfakeDetection](https://github.com/chuangchuangtan/NPR-DeepfakeDetection) | **YES — checkpoint committed in repo** (`model_epoch_last_3090.pth`) + [HF Space demo](https://huggingface.co/spaces/tancc/Generalizable_Deepfake_Detection-NPR-CVPR2024) | Easy; low-level up-sampling-artifact detector (complements CLIP-semantic ones) |
| **FatFormer** | CVPR'24 / [2312.16649](https://arxiv.org/abs/2312.16649) | [Michel-liu/FatFormer](https://github.com/Michel-liu/FatFormer) | **YES** — Baidu + OneDrive + GDrive (verified all three links) | Apache-2.0. Needs OpenAI CLIP ViT-L/14 ckpt; eval-script-shaped, no single-image demo |
| **C2P-CLIP** | AAAI'25 / [2408.09647](https://arxiv.org/abs/2408.09647) | [chuangchuangtan/C2P-CLIP-DeepfakeDetection](https://github.com/chuangchuangtan/C2P-CLIP-DeepfakeDetection) | **YES** — GDrive + HF CLIP ViT-L/14 backbone | Easy, lightweight CLIP-based |
| **AIDE** | ICLR'25 / [2406.19435](https://arxiv.org/abs/2406.19435) | [shilinyan99/AIDE](https://github.com/shilinyan99/AIDE) | **YES** — GDrive checkpoints (released 2024-06) | Moderate; note its Chameleon dataset is email-gated / academic-only (dataset, not weights) |
| **DRCT** | ICML'24 Spotlight (PMLR) | [beibuwandeluori/DRCT](https://github.com/beibuwandeluori/DRCT) | **YES, but ModelScope-only** ([pretrained.zip](https://modelscope.cn/datasets/BokingChen/DRCT-2M/files)) | Moderate; Chinese hosting friction, eval-script-shaped |

---

## Family C — Diffusion-Inpainting-Specific Detectors/Localizers

| Method | Venue / arXiv | GitHub | Weights | Output | Notes |
|---|---|---|---|---|---|
| **OpenSDI / MaskCLIP** | CVPR'25 / [2503.19653](https://arxiv.org/abs/2503.19653) | [iamwangyabin/OpenSDI](https://github.com/iamwangyabin/OpenSDI) | **YES** — [HF `nebula/MaskCLIP-weights`](https://huggingface.co/nebula/MaskCLIP-weights), public, verified via API (updated 2026-06). **All ckpts SD1.5-trained** (paper's cross-generator protocol). Bonus: also ships OpenSDI-trained TruFor/CAT-Net/PSCC/IML-ViT/DeCLIP/ObjectFormer ckpts | **mask + image score** | The **only released detector actually trained on diffusion local edits**. No code license; `test.sh` is a torchrun/IMDLBenCo dataset evaluator — arbitrary-image use needs a small dataset-JSON adapter. Moderate effort, high value |
| **GIM / GIMFormer** | AAAI'25 / [2406.16531](https://arxiv.org/abs/2406.16531) | [chenyirui/GIM](https://github.com/chenyirui/GIM) | **NO — placeholder repo** (README + figure only; no code, dataset, or weights ~2 years on) | — | **Exclude** |
| **PAL4Inpaint** | ECCV'22 Oral / [2208.03357](https://arxiv.org/abs/2208.03357) | [owenzlz/PAL4Inpaint](https://github.com/owenzlz/PAL4Inpaint) | **YES** — GDrive torchscript via `download_checkpoints.sh` | pixel mask | Adobe Research License (non-commercial, no redistribution). Easy single-image CLI, deps = torch only. Caveat: localizes *perceptual artifacts*, not forensic traces — high-quality inpaints are out of scope by design |
| **PAL4VST** | ICCV'23 | [owenzlz/PAL4VST](https://github.com/owenzlz/PAL4VST) | **PARTIAL** — only "unified" + shadow-removal ckpts (GDrive, torchscript + pytorch) | pixel mask | Easy `test_torchscript.py`; same perceptual-artifact caveat; no license file |
| **IID-Net** | TCSVT'22 | [HighwayWu/InpaintingForensics](https://github.com/HighwayWu/InpaintingForensics) | **YES — `weights/IID_weights.pth` in repo** (verified) | pixel mask | torch 1.6; README says test env *requires 2 GPUs* (DataParallel quirk). Trained on GAN/classical inpainting — pre-diffusion, expect weak transfer. Unmaintained |
| **COCO-Inpaint** | arXiv'25 / [2504.18361](https://arxiv.org/abs/2504.18361) | **none found** | **NO** — paper-only; no repo, no dataset, no weights (~15 months after v1) | — | **Exclude** |
| **TGIF / TGIF2** | WIFS'24 / [2407.11566](https://arxiv.org/abs/2407.11566); JIS'26 | [IDLabMedia/tgif-dataset](https://github.com/IDLabMedia/tgif-dataset) | **Dataset only** (CC BY-SA 4.0; 271k text-guided inpaints incl. FLUX.1-Fill, with *spliced* variants like yours) | — | No detector weights, but the closest public *dataset* analog to CLAIMFORGE's construction — useful for the paper's related-work/comparison |
| **DiffForensics** | CVPR'24 | none found | **NO** | — | Exclude |

---

## Family D — MLLM-Based Explainable Detectors

| Method | Venue / arXiv | GitHub | Weights | Output | Local runnability |
|---|---|---|---|---|---|
| **FakeShield** | ICLR'25 / [2410.02761](https://arxiv.org/abs/2410.02761) | [zhipeixu/FakeShield](https://github.com/zhipeixu/FakeShield) | **YES** — [HF `zhipeixu/fakeshield-v1-22b`](https://huggingface.co/zhipeixu/fakeshield-v1-22b) (DTE-FDM + MFLM + DTG bundled; README: "have open-sourced all code & pre-trained model weights"), + SAM ViT-H | verdict + **pixel mask** + text explanation | Apache-2.0, Docker images, `scripts/cli_demo.sh`. LLaVA/LISA-stack; ~40 GB of weights, needs a big GPU (A100-class recommended). Repo active into 2026 |
| **SIDA** | CVPR'25 / [2412.04292](https://arxiv.org/abs/2412.04292) | [hzlsaber/SIDA](https://github.com/hzlsaber/SIDA) | **YES** — HF [`saberzl/SIDA-7B`](https://huggingface.co/saberzl/SIDA-7B), [`SIDA-13B`](https://huggingface.co/saberzl/SIDA-13B), + `-description` variants (verified in README) | verdict + **pixel mask** + explanation | LISA-based (SAM ViT-H + LLaVA); 7B variant fits a 24–48 GB GPU. Training/eval scripts present; less turnkey than FakeShield |
| **ForgeryGPT** | arXiv [2410.10238](https://arxiv.org/abs/2410.10238) | none (the [woody-panda/ForgeryGPT](https://github.com/woody-panda/ForgeryGPT) repo is a *different* face-forgery paper — name collision) | **NO — nothing released** | — | **Exclude** |
| **FakeVLM** | NeurIPS'25 / [2503.14905](https://arxiv.org/abs/2503.14905) | [opendatalab/FakeVLM](https://github.com/opendatalab/FakeVLM) | **YES** — [HF `lingcco/fakeVLM`](https://huggingface.co/lingcco/fakeVLM) (LLaVA-1.5-7B) | verdict + explanation (**no mask**) | Moderate (LLaVA stack, 7B) |
| **LEGION** | ICCV'25 Highlight / [2503.15264](https://arxiv.org/abs/2503.15264) | [opendatalab/LEGION](https://github.com/opendatalab/LEGION) | **PARTIAL** — authors *lost the final weights*; only [intermediate ckpts](https://huggingface.co/khr0516/legion_LE) survive | artifact mask + explanation | Heavy (GLaMM, mmcv 1.4.7, SAM ViT-H). Flag the lost-weights caveat if you use it |
| **Veritas** | ICLR'26 Oral / [2508.21048](https://arxiv.org/abs/2508.21048) | [EricTan7/Veritas](https://github.com/EricTan7/Veritas) | **YES** — ModelScope only ([EricTanh/Veritas](https://www.modelscope.cn/models/EricTanh/Veritas)) | verdict + reasoning trace | Apache-2.0, vLLM single-image script; ModelScope hosting friction |

---

## Family E — Notable New 2025–2026 Methods (verified weights)

| Method | Venue / arXiv | GitHub | Weights | Output | Why include |
|---|---|---|---|---|---|
| **Effort** | ICML'25 Oral / [2411.15633](https://arxiv.org/abs/2411.15633) | [YZY-stack/Effort-AIGI-Detection](https://github.com/YZY-stack/Effort-AIGI-Detection) | **YES** — GDrive (GenImage-, Chameleon-, FF++-trained CLIP-L ckpts) | image score | #1 on ForensicHub's cross-domain FIDL leaderboard; **one-line single-image demo** (`training/demo.py --image ...`). CC BY-NC 4.0 |
| **Community Forensics** | CVPR'25 / [2411.04125](https://arxiv.org/abs/2411.04125) | [JeongsooP/Community-Forensics](https://github.com/JeongsooP/Community-Forensics) | **YES** — HF [`OwensLab/commfor-model-384`](https://huggingface.co/OwensLab/commfor-model-384)/`-224` (MIT model card) | image score | Trained on 2.7M images from **4,803 generators** — broadest generator coverage available; tiny ViT-S, HF-native notebook |
| **RelayFormer** | ICLR'26 / [2508.09459](https://arxiv.org/abs/2508.09459) | [WenOOI/RelayFormer](https://github.com/WenOOI/RelayFormer) | **YES** — [HF `Wenn11/RelayFormer`](https://huggingface.co/Wenn11/RelayFormer/tree/main) + a GDrive "application-oriented" ckpt trained on broader data | pixel mask (image+video), native-resolution | MIT; `infer.py` folder inference; claims better masks than TruFor/Mesorch |
| **ForensicsSAM** | arXiv'25 (preprint) / [2508.07402](https://arxiv.org/abs/2508.07402) | [siriusPRX/ForensicsSAM](https://github.com/siriusPRX/ForensicsSAM) | **YES** — GDrive bundle | image score + mask | Apache-2.0, single `inference.py`; SAM ViT-H; adversarially robust IFDL. Not peer-reviewed yet |
| **RITA** | CVPR'26 / [2509.20006](https://arxiv.org/abs/2509.20006) | [scu-zjz/RITA](https://github.com/scu-zjz/RITA) | **YES, Baidu-only** (friction outside China) | pixel mask | Freshest IML from the BenCo lab (June 2026); include only if you can pull from Baidu |
| **ForensicHub** (harness) | NeurIPS'25 / [2505.11003](https://arxiv.org/abs/2505.11003) | [scu-zjz/ForensicHub](https://github.com/scu-zjz/ForensicHub) | **YES** — IFF-protocol ckpts on Baidu **+ OneDrive**; AIGC/Document ckpts on Baidu **+ GDrive** | per-model | Successor/superset of IMDL-BenCo: 4 domains, 42 models, cross-domain protocol, and a **single-image inference script** (added 2025-07). CC-BY-4.0, actively maintained |

**Verified NOT-available (do not plan around):** GIM/GIMFormer (empty repo), COCO-Inpaint (paper-only), ForgeryGPT (nothing), DiffForensics (nothing), AIGI-Holmes (ICCV'25 — weights ToDo unchecked), Omni-IML (ICLR'26 — "expected in a month", not out), So-Fake-R1 (datasets only, empty models link), UNITE (CVPR'25 — website only), X-Edit (no repo).

---

## Recommended Zero-Shot Baseline Suite (coverage × ease, in run order)

**Tier 1 — core suite (~12 models, all weights verified):**

1. **TruFor** — T1+T2. The standard IML reference; Docker one-command inference; its own paper evaluates on CocoGlide (diffusion local edits), so it's the fairest strong baseline for CLAIMFORGE. (Non-commercial license — fine for a paper.)
2. **MaskCLIP (OpenSDI)** — T1+T2. The only public detector *trained on* diffusion-inpainted local edits; HF weights. Budget half a day to wrap your images in the IMDLBenCo dataset-JSON format — that same adapter then feeds #3.
3. **IMDL-BenCo model zoo** — T2 (+image metrics): one harness gives you **CAT-Net v2, MVSS-Net, PSCC-Net, IML-ViT, Mesorch** (and TruFor re-impl) with paper checkpoints from Google Drive. Best coverage-per-engineering-hour in the whole survey; avoids the Python-3.6 dependency hell of the original MVSS/CAT-Net repos. CAT-Net's JPEG-DCT stream is especially relevant to your splice-back-into-original-JPEG construction.
4. **SAFIRE** — T2, AAAI'25, drop-folder inference, SAM-based; methodologically distinct (point-prompted source partitioning).
5. **RelayFormer** — T2, ICLR'26, HF weights + "application-oriented" checkpoint, folder `infer.py`; your newest localization baseline.
6. **HiFi-Net** — T1+T2, CVPR'23, two-line Python API (mind the two weight sets).
7. **Effort** — T1, ICML'25 Oral, single-command demo; current best-generalizing image-level detector.
8. **UniversalFakeDetect** — T1, canonical CLIP-probe; weights in-repo, minutes to run.
9. **NPR** — T1, low-level artifact counterpart to #8; checkpoint in-repo.
10. **Community Forensics** — T1, HF-native/MIT, broadest generator coverage.
11. **CNNDetection** — T1, trivial `demo.py`; the historical anchor reviewers expect.
12. **FakeShield** — MLLM T1+T2+explanations, Apache-2.0, HF weights, Docker + CLI demo (needs a large GPU).

**Tier 2 — add if budget allows:** SIDA (second MLLM, CVPR'25, mask output), Mesorch standalone (if you want its "-P" variant beyond BenCo), FatFormer / C2P-CLIP / AIDE (extra T1 points, all GDrive), ForensicsSAM (adversarial-robustness angle), PSCC-Net standalone (weights in repo), IID-Net (pre-diffusion inpainting detector — a useful "era gap" datapoint), PAL4VST (perceptual-artifact mask, conceptually different signal).

**Deliberately deprioritized:** DIRE (weights Baidu/RecDrive-only *and* per-image diffusion inversion makes benchmarking expensive), DRCT + Veritas (ModelScope-only hosting), RITA (Baidu-only), LEGION (only intermediate, non-paper weights survive), and everything in the "verified NOT-available" list above.

**Practical notes for the paper:** (1) Every Tier-1 method has non-Baidu weight hosting — no China-cloud friction. (2) License review before any redistribution of outputs: TruFor and PAL are non-commercial; MVSS-Net, CAT-Net, SAFIRE, OpenSDI code have *no* license; MIT/Apache: PSCC-Net, IML-ViT, Mesorch, HiFi-Net, FakeShield, FatFormer, RelayFormer, Community Forensics. (3) Expect Family B detectors near chance on T1 — that mismatch (whole-image detectors vs. localized edits in real photos) is itself a headline result; Families A/C/D provide the competitive T2 numbers. (4) If you'd rather run one harness than twelve repos, ForensicHub (NeurIPS'25) now wraps IMDL + AIGC + deepfake models with checkpoints and a single-image inference script, and is the most future-proof integration point.
