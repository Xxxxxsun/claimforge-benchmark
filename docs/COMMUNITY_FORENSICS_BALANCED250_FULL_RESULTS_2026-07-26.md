# Community Forensics High-res 384 在 Balanced250 上的正式结果

日期：2026-07-26（UTC）

> **文档状态：正式完成，独立全链路 replay 审计通过**
>
> 本文件报告冻结的 1,775-image formal run、两个独立 35-image CUDA
> smoke、共享 Balanced250 指标和独立 replay。所有主结果均来自下列机器
> 产物；旧 Mouse run 和论文表格没有用于填写本报告数字。

正式 run：
`community_forensics_highres_vit_s16_384_balanced250_v1_full1775_20260726`

核心机器证据：

- [run manifest](../results/opensource/community_forensics/community_forensics_highres_vit_s16_384_balanced250_v1_full1775_20260726/manifest.json)：
  `797517bea69b95f0ef9ddb0491a1c2872f3a9b54264a27795ac606c4b6620a17`
- [逐图结果](../results/opensource/community_forensics/community_forensics_highres_vit_s16_384_balanced250_v1_full1775_20260726/results.jsonl)：
  `93dafe47ec1b99047d187693ee211b49c31cd82e4a1f88bd06f79d91ebbc61da`
- [coverage summary](../results/opensource/community_forensics/community_forensics_highres_vit_s16_384_balanced250_v1_full1775_20260726/summary.json)：
  `9c2a0cd8c110268c4e1c9708a2eca87a5a3aabe948dab560ffb1d8a1939e3954`
- [Balanced250 metrics](../results/opensource/community_forensics/community_forensics_highres_vit_s16_384_balanced250_v1_full1775_20260726/balanced250_metrics.json)：
  `8785d9dc6da8927635d48f9108d78203ff38ec11cfdc178a1bb1a2201d7300ba`
- [independent replay audit](../results/opensource/community_forensics/community_forensics_highres_vit_s16_384_balanced250_v1_full1775_20260726/independent_audit.json)：
  `1fc3acda5a45a593d8f54a42668190e625a667dd158951ff3b789e24db9e4794`
- [双 smoke comparison](../results/opensource/community_forensics/community_forensics_balanced_smoke_comparison_v2_3006f900c4e5a0023c5738d80ff8bc1c938016439af38672c289b251b5d93446.json)：
  `b166fee75c9d9c11ef95dfd02a28dda0130c3c04280565011f219c2ed08f3d47`

最终审计状态：`replay_audit_passed`。

## 1. 结论摘要

Community Forensics 的官方公开 High-res 384 checkpoint 已按冻结的
Balanced250 whole-image T1 协议完成：

- 1,775-input score cache 的 valid、error、missing 和 superseded coverage；
- 三种 local insertion 各自及 condition-macro 的 AUROC、AP、
  TPR@5%FPR 和官方固定阈值表现；
- 三种 full-frame conditional edit 各自及 condition-macro 的同类指标；
- 六种 forged condition 等权 macro，但明确说明它混合了 local 与
  full-frame 两种不同 manipulation family；
- 共享 real250 panel 上的固定阈值 false-positive rate，以及每个 forged
  condition 的 recall；
- source-matched secondary 的 paired score delta、strict ranking、
  wins/losses/ties 和 cluster-aware interval；
- resize-and-center-crop 后三种 local exact-difference GT 的
  `full`、`partial`、`none` visibility；
- 两个相同 35-image smoke 是否达到 inference-relevant bit-exact；
- persisted feature-to-head replay 与 1,775-image fresh full-model replay
  是否完整通过；
- 正式回归测试总数与最终审计状态。

结果摘要：

- formal coverage 为 1,775/1,775，零 error、missing、superseded；
- local 三条件 macro AUROC 为 **0.505683**，AP 为 **0.507287**，
  与随机排序接近；官方 `p_fake > 0.5` 的 local recall 只有 **0.4%**；
- full-frame 三条件 macro AUROC 为 **0.829051**，AP 为 **0.834244**，
  TPR@5%FPR 为 **40.8%**，但官方 0.5 阈值 recall 仍只有 **9.87%**；
- source-matched local strict ranking 为 **52.4%**，mean delta
  `+0.001259`；full-frame 为 **93.2%**，mean delta `+0.117845`；
- center crop 后 local visibility pooled 为 full/partial/none
  `322/321/107`；这解释了部分输入不可见，但不把 T1 输出变成 T2；
- 两个 CUDA smoke 的 35/35 computational projection、feature、logit
  和 probability exact；正式 feature-head 与 fresh full-model replay
  也均为 1,775/1,775，所有最大数值差异为 0。

最准确的能力表述必须保持以下边界：

1. Community Forensics 是一张图输出一个 logit/probability 的
   **T1 whole-image AIGC detector**。
2. 它没有 native dense map、mask、bbox 或 patch-level localization
   output；**T2 localization 和 joint score 均为 N/A**。
3. 对 local insertion 的结果只说明整图分类器是否对局部改动产生可用的
   image-level 排序或阈值响应，不能改写为定位能力。
4. `fullframe_mouse`、`fullframe_cat` 和 `fullframe_trash_can` 是由真实
   源图条件编辑得到的全图结果，不是脱离真实源图独立采样的纯 T2I。
5. 本 run 不能单独回答 Community Forensics 对所有纯整图生成模型的
   泛化能力。

## 2. 官方 primary sources、release 与 checkpoint 选择

### 2.1 论文、项目页与官方源代码

方法来自 Park 和 Owens 的 CVPR 2025 论文
[Community Forensics: Using Thousands of Generators to Train Fake Image
Detectors](https://openaccess.thecvf.com/content/CVPR2025/html/Park_Community_Forensics_Using_Thousands_of_Generators_to_Train_Fake_Image_CVPR_2025_paper.html)。
同时冻结作者
[项目页](https://jespark.net/projects/2024/community_forensics/)、
[arXiv 2411.04125v2](https://arxiv.org/abs/2411.04125) 和
[官方 GitHub repository](https://github.com/JeongsooP/Community-Forensics)。

| 官方组件 | 冻结 revision | 作用 |
|---|---|---|
| [GitHub main](https://github.com/JeongsooP/Community-Forensics/tree/ee5b71d43db0f3779e1edd64ee927b13f2dd6ad4) | `ee5b71d43db0f3779e1edd64ee927b13f2dd6ad4` | 完整训练、评估、model 与 processor 实现 |
| [GitHub `eval_single`](https://github.com/JeongsooP/Community-Forensics/tree/5e52ed690bdbd609f9bb1705c4c80d11872a05bd) | `5e52ed690bdbd609f9bb1705c4c80d11872a05bd` | 单图 RGB、resize、crop、sigmoid 与 strict-threshold 语义 |
| [High-res 384 model](https://huggingface.co/OwensLab/commfor-model-384/tree/6076002bf0d9dd37537f965ee2f06f826c333b61) | `6076002bf0d9dd37537f965ee2f06f826c333b61` | 本 benchmark 的 primary checkpoint |
| [224 model](https://huggingface.co/OwensLab/commfor-model-224/tree/26afc31e6b40c312c3fd42c05a758be62446215b) | `26afc31e6b40c312c3fd42c05a758be62446215b` | 公开的次要分辨率 variant；本次排除 |
| [Official processor](https://huggingface.co/OwensLab/commfor-data-preprocessor/tree/3540a3f0d688f8bf492a8aed48613b891f88047e) | `3540a3f0d688f8bf492a8aed48613b891f88047e` | test transform 的独立发布证据 |

`eval_single` 是与 main 分开的官方 branch。其 2026-01-06 commit 把早期
single-image wrapper 的 normalization 修正为 ImageNet mean/std。Main 的
完整 pipeline 和官方 processor 已使用同一组 ImageNet normalization，
因此本次 profile 不保留旧 single-image branch 的 CLIP normalization。

### 2.2 为什么 primary 只选择 384

官方只发布两个直接对应的分辨率 checkpoint：224 和 384。论文把 384
称为 `High res.`，将它作为最佳 detector，并在后续主要实验中使用。
因此本 benchmark 在查看 Balanced250 Community Forensics 分数之前，
score-blind 地冻结 384 为唯一 primary variant。

224 checkpoint 不参加 model selection、ensemble 或 post-hoc fallback。
若未来单独运行 224，必须使用独立 method/profile/run ID，并明确标记为
resolution ablation；不得把两个概率平均后称为官方 ensemble。官方发布中
没有五模型 ensemble。

### 2.3 HF safetensors 与 Dropbox checkpoint 的 bit-exact 身份

本次使用不执行 pickle 的 HF `model.safetensors`：

| Variant | HF weight bytes | State tensors | State elements | Weight SHA-256 |
|---|---:|---:|---:|---|
| 384 High-res | 87,262,324 | 152 | 21,811,969 | `b89f36275f3bf5e2b040eee36597a8f19db051bff9a473a9cf7b2466284fb387` |
| 224 | 86,678,644 | 152 | 21,666,049 | `a6cc439d5a6d2dfadd60c77d27a2838ad55b34e601ecd30f46ad97266d6ac4e0` |

官方 GitHub 同时链接一个
[Dropbox `model_weights.tar`](https://www.dropbox.com/scl/fi/e8titz35ci9a2ij1oq5mu/model_weights.tar?rlkey=tmyz3tjqf7b4dg071kypsgoal&st=09ud9hdj&dl=0)：

| Dropbox asset | Bytes | SHA-256 |
|---|---:|---|
| `model_weights.tar` | 522,147,840 | `2d938f210561ed1185c1536b8f0a7d0c10acf0caf422b6440b383afd55e8dd3b` |
| `model_v11_ViT_384_base_ckpt.pt` | 261,947,930 | `d134f6f2c6185c9146ed54e7ea1c43e9f67ea42d98e9bdc88791d4163095dcde` |
| `model_v11_ViT_224_base_ckpt.pt` | 260,193,218 | `c6f398c3a2569b888b9516017a1949cca1b1410baf4fcfa10c05361d8ac820a9` |

研究阶段已离线解包两个 Dropbox checkpoint，并将其 `model` state dict 与
对应 HF safetensors 的 152 个 key 逐项比较。两种分辨率的 key 集合完全
相同，所有 tensor 均满足 `torch.equal`；HF safetensors 是 Dropbox
checkpoint 模型部分的 bit-exact 安全镜像，而不是另一个训练 variant。
Dropbox 文件额外携带 optimizer、scheduler、scaler、epoch 和 iteration
训练状态，本次推理不需要也不加载这些 pickle 内容。

正式 run 冻结的 model/processor asset bundle SHA-256 为
`810a7592a82f09cbf638985e9c59eed9ebd2c3ff28ebab97f348bfd3c69b7fb3`。
Source 由 manifest 的 `immutable.source` 单独绑定。Adapter 也没有另造
一个未记录的 combined bundle hash；其逐文件 SHA-256 inventory 保存在
`immutable.adapter_sources`，完整 immutable run-config fingerprint 为
`85e6c75524334d68f4240df5605e7470f148e49e34d4971e3c2d7d6e901490de`。

## 3. 方法原理与训练多样性

Community Forensics 的核心主张不是设计复杂的新取证 head，而是让标准图像
分类器在足够多的生成器上端到端学习，从而减少对单一 GAN、diffusion
实现、采样器或编码管线的狭窄指纹依赖。

论文数据总计覆盖 4,803 个生成器和约 2.7M 张生成图。这个总数包含训练和
held-out evaluation：

- 4,763 个系统化收集的 Hugging Face latent-diffusion models；
- 19 个手工选择、覆盖更多架构的训练 models；
- 11 个 commercial evaluation models；
- 10 个额外 held-out evaluation models。

因此实际 fake training split 包含 4,782 个 generators：

| Training architecture family | Generator count |
|---|---:|
| Latent diffusion | 4,766 |
| GAN | 12 |
| Pixel diffusion | 3 |
| Other | 1 |
| **Total** | **4,782** |

约 2.7M 张 fake 与约 2.7M 张 real 组成约 5.4M 个平衡训练样本。Real 数据
来自 LAION、ImageNet、COCO、FFHQ、CelebA、MetFaces、AFHQ-v2、
Forchheim、IMD2020、Landscapes HQ 和 VISION。论文训练还使用可模拟现实
后处理组合的 RandomStateAugmentation，包括 JPEG、不同插值 resize/crop、
水平/垂直翻转、旋转、平移、shear、padding 和 cutout。

论文训练设置为端到端更新 backbone，而不是只训练线性 head；optimizer 为
AdamW，学习率 `2e-5`、weight decay `1e-2`、batch size 512、mixed
precision、20% warmup 后 cosine schedule，主模型训练约 52K iterations。
Dropbox checkpoint 的训练 metadata 记录 iteration 52,085，与该 full
training setting 一致。

这种 generator diversity、端到端 adaptation 和后处理增强解释了该方法在
未见生成器的整图 benchmark 上可能很强。但它仍是中心 crop 的整图
classifier：

- 小区域信号可能被 crop 完全排除；
- 即使保留，也可能在全局 CLS representation 中被真实背景稀释；
- 训练目标不要求输出被改动位置；
- 强整图 AIGC generalization 不自动等价于局部植入 detection。

## 4. Released artifact 与论文 CLIP prose 的冲突

论文正文和表注把 backbone 描述为 `plain CLIP-ViT-S`，并提到 CLIP
objective、LAION-2B、ImageNet-21K 和 ImageNet-1K pretraining。

但官方 released code 明确构造：

```text
vit_small_patch16_384.augreg_in21k_ft_in1k
```

对应的
[timm 官方模型卡](https://huggingface.co/timm/vit_small_patch16_384.augreg_in21k_ft_in1k)
把它描述为 JAX AugReg ViT，ImageNet-21K pretrain 后在 ImageNet-1K
fine-tune，并标记 Apache-2.0；它不是 OpenCLIP/LAION-2B checkpoint。

这不是可以通过猜测消除的措辞差异。本 benchmark 的处理原则是：

1. 执行 contract 以官方 released code 的 timm identifier 和完整
   Community Forensics checkpoint 为准；
2. 不把 backbone 替换成任意 CLIP/OpenCLIP ViT-S；
3. checkpoint strict load 必须覆盖全部参数；
4. 报告保留 paper-prose/released-artifact discrepancy；
5. 不对未被 executable artifact 建立的 CLIP pretraining identity
   作更强 provenance 声明。

完整 fine-tuned state 覆盖网络全部 152 个 tensors，因此本次 operational
detector 是明确的；存在歧义的是论文对其基础预训练身份的文字描述。

## 5. Frozen executable contract

正式实现入口：

- `eval/opensource/run_community_forensics_balanced.py`
- `eval/opensource/analyze_community_forensics_balanced.py`

正式文件 SHA-256：

| File | SHA-256 |
|---|---|
| runner | `bac691967fa8dea2eed206a4a65e88343c777c42a8d7567ed37d8b65ec16a022` |
| analyzer | `7baec8ce438951aea1c0b530b51ddbac2779a3e8546589e1cfededf52af0487a` |
| runner tests | `6ab7916731176c0ab84ae10b68f55ef01c16ecf7afd2c50e0d2325828e99805a` |
| analyzer tests | `a39bc44ef73428601920b6f87d6383e05a9c21695b81a7c98a0929227dbb5a63` |

Runner、analyzer 与共享依赖的逐文件 inventory 被 formal manifest 绑定；
两项 test file hash 是提交时的独立文件证据，不属于推理 manifest。

### 5.1 模型构造与安全加载

```text
timm 1.0.15
-> create vit_small_patch16_384.augreg_in21k_ft_in1k
   with pretrained=False
-> replace head with Linear(384, 1)
-> safetensors.torch.load_file(pinned local model.safetensors)
-> strict state-dict load:
   152/152 keys, no missing, no unexpected
-> model.eval()
```

官方 wrapper 会先请求 `pretrained=True`，随后用完整 Community Forensics
state 覆盖每个参数。本 benchmark 使用 `pretrained=False` 构造完全相同的
结构，避免下载一个立即被覆盖、且可能随远端变化的 base checkpoint。Strict
full-state coverage 是该等价性的机器门禁。

推理阶段不得：

- 执行 Dropbox pickle；
- 使用 mutable `main` 下载；
- 依赖 Hugging Face remote custom code；
- 回退到未哈希 base model；
- 在 checkpoint load 后留下 missing/unexpected key；
- 访问网络。

### 5.2 解码、resize、crop 与 tensor

每张 canonical JPEG 的唯一正式输入路径为：

```text
Pillow.Image.open(canonical_jpeg).convert("RGB")
-> no EXIF transpose
-> no ICC conversion
-> aspect-preserving bilinear Resize(short edge = 440)
-> CenterCrop(384 x 384)
-> RGB uint8 / 255 -> float32 tensor
-> ImageNet normalization:
   mean = [0.485, 0.456, 0.406]
   std  = [0.229, 0.224, 0.225]
```

官方 `eval_single` requirements 只给出 PyTorch/torchvision 最低版本
`torch>=2.2.1`、`torchvision>=0.17.1`，并不构成精确运行 pin。实际 formal
run 使用下列冻结环境：

| Package | Frozen target |
|---|---|
| Python | `3.12.3` |
| PyTorch | `2.8.0.dev20250627+cu128` |
| torchvision | `0.23.0.dev20250627+cu128` |
| timm | `1.0.15` |
| Pillow | `11.1.0` |
| safetensors | `0.5.2` |
| CUDA / GPU | CUDA runtime `12.8`; logical `cuda:0`; `NVIDIA L20Z` |

Runner 必须记录实际 import/distribution versions、venv path、Python
executable、determinism flags 和 source inventory。Manifest 没有记录
driver 或 GPU UUID，因此本报告不使用事后 `nvidia-smi` 值补写这两项。
版本与 runtime gate 均通过；venv 是
`/root/.cache/claimforge/venvs/community-forensics-balanced-nightly20250627`
（`include-system-site-packages=true`），`pyvenv.cfg` SHA-256 为
`7a40b0582b3525537e9e005348ceec3a23259899af45afc367014c7acbdf91f4`。

### 5.3 前向、feature、score 与 released decision

```text
normalized [1, 3, 384, 384] float32 tensor
-> ViT-S/16 patch embedding and transformer blocks
-> 384-dimensional classifier input feature
-> Linear(384, 1) raw_logit
-> float32 sigmoid
-> ai_score = p_fake
-> generated iff p_fake > 0.5
```

Label 语义固定为 `real=0`、`fake/generated=1`，score 越高越 synthetic。
恰好等于 `0.5` 不算 generated。Official probability 是唯一 primary
score；raw logit 可以保存为数值诊断和 replay 证据，但不得在看见结果后
替代 probability 成为主指标。

每张成功图必须保存或绑定：

- input row identity 和 canonical JPEG SHA-256；
- decoded RGB、resized RGB、cropped RGB 和 normalized tensor digest；
- 384-D classifier-input feature 及其 array/file digest；
- raw logit、float32 probability、`ai_score` aliases 和 strict decision；
- model/profile/runtime/source/asset fingerprints；
- attempt number、completion status 和 deterministic ordering evidence。

### 5.4 数值确定性和 replay 门禁

正式 runner 和 analyzer 应执行：

1. forward-hook 捕获官方 head 输入；
2. 同一记录设备上从 persisted 384-D feature 手工执行
   `F.linear(feature, weight, bias)`；
3. 要求 raw logit exact match；
4. 同设备执行 float32 sigmoid 并要求 probability exact match；
5. A/B smoke 比较所有 inference-relevant fields 和 feature arrays；
6. formal run 完成后重新构造模型，从全部 1,775 张 canonical JPEG 执行
   fresh full-model replay；
7. feature、logit、probability 与 decision 全部通过后才允许最终报告。

不把 CUDA logit 搬到 CPU 后重新 sigmoid 的跨设备 bit equality 当作主要
门禁；可信性来自更强的同设备 feature-to-head replay 和相同 runtime 的
fresh full-model replay。

双 smoke 状态：`deterministic_smoke_comparison_passed`，35/35
inference-relevant computational projection exact。

Fresh replay 状态：`fresh_full_image_to_feature_replay_passed`，
1,775/1,775 完整图像前向通过。

## 6. Frozen Balanced250 设计

数据 release 为
`claimforge-balanced250-independent-panel-jpeg-q95-v1`：

| Frozen ledger | Rows | SHA-256 |
|---|---:|---|
| release `manifest.json` | 1 | `b2bbf3eb7a835f9c729cdffe29a40247225125779fe21551270fefe95d667c7f` |
| deterministic contract | — | `671d1739bebf4370d26b4629ca26b56cc546a817d469ba505cc39bda8b33102c` |
| `inputs.jsonl` | 1,775 | `6b5128909eeffdbd88f61af02ca1bd191cb5460f94a23b47c87ebd0172e6d12c` |
| `panel.jsonl` | 1,750 | `e01d7985b41cee5262a3f8b6d71420986feae96771b11c46fda98c3e72a0d424` |
| `source_pairs.jsonl` | 1,500 | `391fdcf06eecff4cf1843ddb3688acacf52a293725c501660b7a361173b09b30` |

Score cache 包含：

| Condition | Cache rows | Primary-panel rows | T1 | T2 |
|---|---:|---:|---|---|
| `real` | 275 | 250 | applicable | N/A |
| `local_mouse` | 250 | 250 | applicable | model output N/A |
| `local_cat` | 250 | 250 | applicable | model output N/A |
| `local_trash_can` | 250 | 250 | applicable | model output N/A |
| `fullframe_mouse` | 250 | 250 | applicable | N/A |
| `fullframe_cat` | 250 | 250 | applicable | N/A |
| `fullframe_trash_can` | 250 | 250 | applicable | N/A |
| **Total** | **1,775** | **1,750** |  |  |

### 6.1 Primary selection-unpaired comparisons

每个 forged250 condition 与同一个、独立冻结的 real250 panel 比较：

```text
real250 vs local_mouse250
real250 vs local_cat250
real250 vs local_trash_can250
real250 vs fullframe_mouse250
real250 vs fullframe_cat250
real250 vs fullframe_trash_can250
```

Primary 报告每个 comparison 的：

- AUROC 和 AP；
- TPR@5%FPR；
- released strict `p_fake > 0.5` confusion metrics；
- condition、family 和 all-six equal-condition macros；
- lodging、restaurant domain diagnostics；
- score distribution 和 numerical saturation diagnostics；
- 95% percentile confidence intervals。

Primary panel 是 selection-unpaired；不得在本节从 `task_id` 猜配对，也不得
把 paired ranking 当作 primary image-only performance。

### 6.2 Source-matched secondary

Secondary 只使用 `source_pairs.jsonl` 中 1,500 个明确 real/forged
endpoints。额外 25 张 non-panel real 只为补齐 source pairs，不进入
real250 primary panel。

Secondary 应报告：

- forged minus source-real mean score delta；
- strict matched ranking；
- wins、losses、ties；
- condition 和 family 汇总；
- source-content-cluster-aware confidence intervals。

共享 `balanced250_t1_summary_v1` schema 不输出 secondary domain/visibility
strata 或 exact sign-test p。本报告不在看到结果后追加另一套未冻结的
inferential analysis；primary domain diagnostics、输入 visibility census
和 paired condition/family 结果分别报告。

### 6.3 Bootstrap 与 operating point

置信区间使用跨 label/condition 共享 source-content-cluster 权重的
Poisson bootstrap：

```text
resamples: 1000
root seed: 20260726
interval: two-sided 95% percentile
```

TPR@5%FPR threshold 使用相应 real score 的 95th percentile、
`method="higher"` 和 strict `>`。这个 threshold 只是 real-only
诊断 operating point，不替代官方固定 `p_fake > 0.5`。

### 6.4 Full-frame 条件语义

三组 `fullframe_*` 是真实源图上的 Hunyuan conditional edit：

- 它们经过整图生成/编辑过程，适合测试 whole-image detector 的 T1 响应；
- 它们仍可能保留源图内容、结构或低层统计；
- Trash-can condition 表示 primary single-shot pipeline 已执行，不保证
  目标物体另经语义 QC 后一定成功出现；
- orange placement box 只是 conditioning metadata；
- 不得把 orange box、全图 mask 或 dense diagnostic 当作 T2 GT。

## 7. Local center-crop visibility

Community Forensics 只看 resize 后的中央 `384 x 384`。Local edit 位于图像
边缘时，exact-difference pixels 可能全部或部分落在 crop 外。因此每个
local row 必须在 score-blind analyzer 中计算输入可见性：

```text
canonical exact-difference GT positive pixels
-> apply the frozen aspect-preserving short-edge-440 resize geometry
-> apply the frozen 384 x 384 center-crop geometry
-> retained fraction == 1: full
-> retained fraction == 0: none
-> otherwise: partial
```

Visibility 还应结合 real/forged transformed RGB、tensor、feature 和
probability equality 检查。Bilinear interpolation 的 support 可能让连续
edit box 刚好在 crop 外、但 crop 边界仍出现极小变化；因此几何 visibility
与 exact transformed-input equality 都应保留，不能互相替代。

这些数据是输入条件诊断，不是模型输出：

- `local_mouse`、`local_cat`、`local_trash_can`：计算
  `full/partial/none`；
- `real`：`not_applicable`；
- 三组 `fullframe_*`：`not_applicable`；
- 不计算 pixel AP、IoU、MCC 或 predicted-mask metrics。

正式 visibility inventory：

| Local condition | Full | Partial | None | Total | Mean retained GT fraction |
|---|---:|---:|---:|---:|---:|
| `local_mouse` | 149 | 30 | 71 | 250 | 0.6519108174 |
| `local_cat` | 126 | 106 | 18 | 250 | 0.7371812661 |
| `local_trash_can` | 47 | 185 | 18 | 250 | 0.6185209864 |
| **Pooled** | **322** | **321** | **107** | **750** | **0.6692043566** |

## 8. Coverage

| Condition | Expected | Result | Valid | Error | Missing |
|---|---:|---:|---:|---:|---:|
| `real` | 275 | 275 | 275 | 0 | 0 |
| `local_mouse` | 250 | 250 | 250 | 0 | 0 |
| `local_cat` | 250 | 250 | 250 | 0 | 0 |
| `local_trash_can` | 250 | 250 | 250 | 0 | 0 |
| `fullframe_mouse` | 250 | 250 | 250 | 0 | 0 |
| `fullframe_cat` | 250 | 250 | 250 | 0 | 0 |
| `fullframe_trash_can` | 250 | 250 | 250 | 0 | 0 |
| **Total** | **1,775** | **1,775** | **1,775** | **0** | **0** |

```text
physical result rows:              1775
latest result rows:                1775
superseded attempts:               0
coverage fraction:                 1.0
success fraction:                  1.0
persisted 384-D feature files:     1775
formal selected-ID SHA-256:        e4418d86461f889e4a4423f26aab63243e6f63a435a49624881c34979b812e41
```

## 9. Primary whole-image T1 results

下表每行使用同一个 real250 panel。区间为 1,000-resample
shared-cluster bootstrap 的双侧 95% percentile CI；confusion 使用官方
strict `p_fake > 0.5`。

| Forged condition | AUROC [95% CI] | AP [95% CI] | TPR@5%FPR [95% CI] | Accuracy@0.5 | Recall@0.5 | TP / FP / FN / TN |
|---|---:|---:|---:|---:|---:|---:|
| `local_mouse` | 0.494152 [0.482062, 0.506850] | 0.498869 [0.486827, 0.514231] | 0.044000 [0.032371, 0.057921] | 0.500000 | 0.004000 | 1 / 1 / 249 / 249 |
| `local_cat` | 0.520832 [0.505179, 0.537969] | 0.515315 [0.502514, 0.535159] | 0.056000 [0.040804, 0.082306] | 0.500000 | 0.004000 | 1 / 1 / 249 / 249 |
| `local_trash_can` | 0.502064 [0.487573, 0.517549] | 0.507677 [0.492640, 0.526894] | 0.052000 [0.038626, 0.075704] | 0.500000 | 0.004000 | 1 / 1 / 249 / 249 |
| `fullframe_mouse` | 0.821584 [0.793693, 0.850596] | 0.826191 [0.794251, 0.857022] | 0.372000 [0.285689, 0.493788] | 0.548000 | 0.100000 | 25 / 1 / 225 / 249 |
| `fullframe_cat` | 0.842912 [0.817628, 0.871844] | 0.846590 [0.816321, 0.877829] | 0.444000 [0.333333, 0.554218] | 0.546000 | 0.096000 | 24 / 1 / 226 / 249 |
| `fullframe_trash_can` | 0.822656 [0.794032, 0.852775] | 0.829951 [0.799578, 0.861870] | 0.408000 [0.289878, 0.531390] | 0.548000 | 0.100000 | 25 / 1 / 225 / 249 |

Condition-equal macro：

| Family | AUROC [95% CI] | AP [95% CI] | TPR@5%FPR [95% CI] | Accuracy@0.5 | Recall@0.5 |
|---|---:|---:|---:|---:|---:|
| local 三条件 | 0.505683 [0.494482, 0.517739] | 0.507287 [0.498841, 0.522736] | 0.050667 [0.042227, 0.067360] | 0.500000 | 0.004000 |
| full-frame 三条件 | 0.829051 [0.804174, 0.856227] | 0.834244 [0.805494, 0.864408] | 0.408000 [0.309271, 0.518471] | 0.547333 | 0.098667 |
| 全六条件 | 0.667367 [0.652879, 0.683555] | 0.670765 [0.656445, 0.688579] | 0.229333 [0.181231, 0.287990] | 0.523667 | 0.051333 |

Real-only 5% FPR threshold、actual FPR 和 CI：
threshold `0.0233896393 [0.0084937643, 0.0632447824]`；actual FPR
`0.048000 [0.0382919826, 0.0497925311]`。

固定 `p_fake > 0.5` 在七个 cache conditions 上的 positive counts：
`real 1/275`、`local_mouse 1/250`、`local_cat 1/250`、
`local_trash_can 1/250`、`fullframe_mouse 25/250`、
`fullframe_cat 24/250`、`fullframe_trash_can 25/250`。

这组结果的关键分裂不是“方法完全无效”，而是它对 manipulation family
高度敏感：local 排序接近随机，full-frame 排序明显有效。另一方面，released
0.5 阈值在两类 forged 上都很保守，因此 ranking 能力不能直接等同于默认
operating point 的召回率。

### 9.1 Score distributions 与 numerical diagnostics

| Condition/family | Min | Mean | Median | P05 | P95 | Max | Unique | Exact 0 / 1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `real` | 1.87763e-7 | 0.007366864 | 5.76192e-5 | 1.04209e-6 | 0.022671964 | 0.678142309 | 270 | 0 / 0 |
| local pooled | 1.91948e-7 | 0.009065865 | 6.80005e-5 | 1.27846e-6 | 0.023681943 | 0.847478449 | 726 | 0 / 0 |
| full-frame pooled | 8.39186e-7 | 0.125426093 | 0.010336512 | 1.84030e-5 | 0.812056202 | 0.994952440 | 750 | 0 / 0 |

`p_fake` 是模型 sigmoid 输出，不是经过 Balanced250 或真实部署流量重新
calibrate 的概率。AUROC/AP、real-only 5% FPR 点和 released 0.5
operating point 回答不同问题。

本表是对已哈希 `results.jsonl` 的描述性聚合；P05/P95 使用 NumPy 2.2.6
默认 linear quantile。共享 metrics schema 不重复保存这些 distribution
字段。所有 1,775 个概率有限且严格位于 `(0, 1)`，因此没有 NPR 式 sigmoid
saturation 问题，也没有理由在结果后改用 raw logit 作为 primary score。

### 9.2 Domain diagnostics

| Family | Domain | AUROC [95% CI] | AP [95% CI] | TPR@5%FPR [95% CI] |
|---|---|---:|---:|---:|
| local | lodging | 0.512141 [0.498486, 0.529004] | 0.525120 [0.516006, 0.547156] | 0.043480 [0.032087, 0.066157] |
| local | restaurant | 0.503504 [0.485757, 0.522146] | 0.496554 [0.480036, 0.526268] | 0.056578 [0.036364, 0.079691] |
| full-frame | lodging | 0.826928 [0.790224, 0.865102] | 0.827209 [0.788181, 0.870777] | 0.327376 [0.226752, 0.500904] |
| full-frame | restaurant | 0.838620 [0.798371, 0.878496] | 0.856332 [0.823297, 0.892943] | 0.515626 [0.344751, 0.617463] |
| all-six | lodging | 0.669534 [0.649324, 0.692255] | 0.676164 [0.658508, 0.703081] | 0.185428 [0.135203, 0.277666] |
| all-six | restaurant | 0.671062 [0.647789, 0.695267] | 0.676443 [0.657740, 0.702662] | 0.286102 [0.195941, 0.335034] |

这些是分层诊断，不自动构成 domain 差异的 simultaneous hypothesis test。

### 9.3 Local visibility diagnostics

正式 artifact 冻结的是第 7 节的 score-blind geometric visibility
inventory；共享 metrics schema 没有定义 visibility-stratified
AUROC/AP/bootstrap estimand，因此本报告不补写一张无机器字段对应的推断表。

作为逐图结果的可复算一致性诊断：322 个 `full` pair 和 321 个 `partial`
pair 中，real/forged crop、tensor、feature、probability 均没有 exact-equal
pair；107 个 `none` pair 中有 105 个在这四级全部 exact-equal，另 2 个因
bilinear resize support 触及 crop 边缘而仍发生变化。750 个 pair 的 resized
RGB 都不完全相等。该描述由已哈希 `results.jsonl` 与 `source_pairs.jsonl`
直接聚合，不是模型 localization output，也不是独立的 bootstrap 指标。

## 10. Source-matched secondary results

| Family | Pairs | Mean forged-real delta [95% CI] | Strict matched ranking [95% CI] | Wins / losses / ties |
|---|---:|---:|---:|---:|
| local pooled | 750 | 0.001259 [0.000157, 0.002692] | 0.524000 [0.485944, 0.562983] | 393 / 252 / 105 |
| full-frame pooled | 750 | 0.117845 [0.091708, 0.144924] | 0.932000 [0.906208, 0.956232] | 699 / 51 / 0 |

Per-condition paired results：

| Condition | Pairs | Mean delta [95% CI] | Strict ranking [95% CI] | Wins / losses / ties |
|---|---:|---:|---:|---:|
| `local_mouse` | 250 | 0.000264 [-0.000130, 0.000760] | 0.444000 [0.379440, 0.502043] | 111 / 69 / 70 |
| `local_cat` | 250 | 0.001019 [-0.000455, 0.002834] | 0.616000 [0.555522, 0.682659] | 154 / 78 / 18 |
| `local_trash_can` | 250 | 0.002494 [0.000346, 0.005453] | 0.512000 [0.456514, 0.572094] | 128 / 105 / 17 |
| `fullframe_mouse` | 250 | 0.117023 [0.087840, 0.147213] | 0.936000 [0.901564, 0.964916] | 234 / 16 / 0 |
| `fullframe_cat` | 250 | 0.118197 [0.088634, 0.147582] | 0.944000 [0.914588, 0.970957] | 236 / 14 / 0 |
| `fullframe_trash_can` | 250 | 0.118313 [0.088399, 0.149307] | 0.916000 [0.882102, 0.949229] | 229 / 21 / 0 |

Paired sensitivity 与 image-only deployment ranking 回答不同问题。即使
matched forged score 常高于 matched real，也不表示一个没有 real
counterfactual 的单图 detector 已有同等能力；正式解释必须同时检查全局
AUROC/AP、delta magnitude、ties 和 released-threshold behavior。

这里再次显示 local/full-frame 分裂：local pooled CI 包含接近随机的
strict ranking，且绝对 score delta 很小；full-frame paired ranking 和
delta 都很强。Metrics artifact 没有 sign-test 字段，本报告不追加一个
report-only p-value。

## 11. Determinism、runtime 与审计

### 11.1 CPU preflight

CPU preflight sample、expected hashes、logit/probability 和最终状态：

```text
status:                    passed
sample_id:                 2c80d38ac19c2d3b76950996
canonical JPEG SHA-256:    12607f3cdada1480038f3d506146cdc1fa0c1c50034afda5e3a5f175433e716b
decoded RGB SHA-256:       5a4747a6e3a8313f8c9ec3dde2504bb53184666276d7e54dc5fab53ca0e7194b
resized RGB SHA-256:       dd85a9c31b4e7248da5857f6928e64cd8955ffa8e984d19600620e5c42321fb7
crop RGB SHA-256:          eb7c1b6c7c527f8bdbacaf6bd3957cea3577bd0e2029cd69b253042aa4e1328a
tensor SHA-256:            9540fe65ec48c8a1e6ecafb0ede0c96888c076eda1ba046117f3b2830f4a881e
feature array SHA-256:     dfd24b8af514df33ce6e8d8fd45464d56fa1ef5ffafab54384f0e34f0cb03e5d
raw logit:                 -8.351208686828613
probability:               0.00023605521710123867
two CPU forwards:          exact
official five-image max Δ: 4.2596481897305694e-11
```

### 11.2 双 CUDA smoke

两个 smoke 必须分别写入不可变目录，覆盖每个 condition 的相同五张图，
合计 35 images：

| Field | Smoke A | Smoke B |
|---|---|---|
| run ID | `community_forensics_highres_vit_s16_384_balanced250_v1_smoke5x7_a_20260726` | `community_forensics_highres_vit_s16_384_balanced250_v1_smoke5x7_b_20260726` |
| selected-ID SHA-256 | `b420bc581386a540b742d917d60d007f0e5522b6cca43fa217797944c40667e5` | `b420bc581386a540b742d917d60d007f0e5522b6cca43fa217797944c40667e5` |
| results SHA-256 | `d3c25cd2c5eda7ac4c7c9ffc4d1d5ad480886994fe3525ec4c99316b2b478652` | `5ad5475cb41404efe4d3bdaf19dac7365f8ac6ef4420834931e454c7b7fba6c9` |
| feature inventory SHA-256 | `f2cd83bd1f8c00ed1e87db9471595fcdd480601c4d615b9bea735002d875f811` | `1b7f4e6a8aba6c1798d67edb600cbf8aa15f3d935d23193917e2d1dc5ddacecb` |
| run-manifest fingerprint | `aa6a720c226113cda8ec1eb56f7a3a9b339c873c0ff526a4ca8aefc5a40d84fe` | `90e2e6ada2c9089a0073c14df0222ee0084b16d9236e8217487782e9f7a10302` |

Comparison artifact SHA-256：
`b166fee75c9d9c11ef95dfd02a28dda0130c3c04280565011f219c2ed08f3d47`

Inference-relevant exact comparison：35/35 computational projection exact；
feature file bytes 与 arrays exact；max feature/raw-logit/probability/
`ai_score` difference 均为 `0.0`；两边独立 classifier-head replay
也分别为 35/35、max difference `0.0`。

Raw JSONL 可因 run ID、timestamp 和 artifact path 不同而不 byte-exact；
比较必须排除这些预期 provenance 差异，但不能排除 input/tensor/feature/
logit/probability/decision/runtime contract 或 computational configuration。

### 11.3 Formal runtime

| Runtime quantity | Value |
|---|---:|
| GPU model / logical device | NVIDIA L20Z / `cuda:0`; UUID 未被 immutable manifest 记录 |
| Forward latency mean / median / P95-higher / max | 8.218242 / 6.239127 / 14.940735 / 283.494767 ms |
| Preprocess latency mean / median / P95-higher / max | 102.961827 / 116.012037 / 137.666688 / 151.343631 ms |
| Peak allocated CUDA memory | 133,535,232 bytes |
| Recorded model-load network attempts | disabled；`urlopen/create_connection/connect = 0/0/0` |

Latency 与 peak 是对 1,775 条 formal `results.jsonl` 的确定性描述聚合；
P95 使用 `method="higher"`。Network 行只陈述 formal/fresh model-load
instrumentation 的范围，不扩大成未记录的全进程流量声明。

### 11.4 Independent replay

| Audit gate | Expected | Actual |
|---|---:|---:|
| persisted feature-to-head replay | 1,775 | 1,775 passed；max raw/probability Δ = 0 |
| fresh full-model image replay | 1,775 | 1,775 passed |
| feature mismatches | 0 | 0；`numpy.array_equal`，max Δ = 0 |
| raw-logit mismatches | 0 | 0；max Δ = 0 |
| probability mismatches | 0 | 0；max Δ = 0 |
| decision mismatches | 0 | 0；fail-closed replay passed |

“0 mismatches” 是由 fail-closed replay 成功和相应 max difference 为 0
推出；audit schema 没有另设冗余 mismatch counter。

Independent audit artifact SHA-256：
`1fc3acda5a45a593d8f54a42668190e625a667dd158951ff3b789e24db9e4794`

最终审计状态：`replay_audit_passed`。

## 12. License 与商用边界

### 12.1 代码、released weights 与 base model

| Component | 官方 metadata / license evidence |
|---|---|
| Community Forensics GitHub code | MIT |
| `OwensLab/commfor-model-384` | HF model card declares MIT |
| `OwensLab/commfor-model-224` | HF model card declares MIT |
| `OwensLab/commfor-data-preprocessor` | HF card declares MIT |
| timm executable base identifier | Official timm card declares Apache-2.0 |

这些 evidence 支持在本 benchmark 中把代码和公开 checkpoint 记录为具有
permissive release metadata。运行使用 HF safetensors，不执行或重新分发
Dropbox training-state pickle。

### 12.2 训练数据许可不能由模型卡自动解决

官方 dataset releases 的边界并不相同：

| Dataset release | Frozen revision | Card metadata / text |
|---|---|---|
| `OwensLab/CommunityForensics` | `f0b06f79d20389706bb76c626391fa870e0bf604` | metadata 为 CC-BY-4.0，但正文同时写 `for research purposes only` |
| `OwensLab/CommunityForensics-Small` | `6c539a534c07917307c381f5af4053c6091b5278` | CC-BY-NC-SA-4.0；明确 non-commercial research |
| `OwensLab/CommunityForensics-Eval` | `7d4a74a88d2cac93b513c0853bf92c260eaceea0` | CC-BY-NC-SA-4.0；明确 non-commercial research/education |

Full dataset 的标准 CC-BY-4.0 metadata 与正文额外的
`research purposes only` 表述存在许可语义张力。更重要的是，训练图来自
数千个生成器和多个 real datasets；官方也要求按 generator metadata
检查各自许可，并指出多数 generator 使用 CreativeML OpenRAIL-M。

本 benchmark 只在 CLAIMFORGE 自有冻结输入上执行 MIT-metadata checkpoint，
不下载、打包或再发布 Community Forensics training/evaluation images。
这降低了直接重新分发 NC dataset 的风险，但不会自动建立完整训练数据
权利链，也不是对 checkpoint 衍生权利的法律保证。

因此最终 commercial 记录应为：

```text
code_release_metadata: MIT
weight_release_metadata: MIT
base_model_metadata: Apache-2.0
training_data_commercial_provenance: unresolved
overall_commercial_clearance: not established
```

正式 manifest/audit 中的 release metadata 为 GitHub code、selected
model、processor 均声明 MIT；独立审计同时把 commercial clearance 明确
记录为 `training_data_commercial_lineage_not_established`。因此可以说
“公开代码与权重 metadata permissive”，不能说“完整商用权利链已确认”。

本节是技术 provenance 记录，不构成法律意见。任何商业部署或再分发决定
仍需独立法律审查。

## 13. 完成门禁

完成门禁按实际机器 schema 收口如下：

- [x] runner/analyzer 与 focused tests 冻结；
- [x] official source evidence、model/processor asset bundle、adapter
  逐文件 inventory 与 immutable config fingerprint 冻结；
- [x] CPU preflight 通过；
- [x] 两个 35-image CUDA smoke inference-relevant bit-exact；
- [x] 1,775/1,775 formal coverage，零 missing/error/superseded；
- [x] 每张 persisted 384-D feature 完整且哈希匹配；
- [x] shared-schema primary condition/macro/domain 与 paired
  condition/family secondary 完整；
- [x] score-blind visibility census 完整，且未虚构 schema 外的 visibility
  bootstrap 指标；
- [x] 1,775/1,775 same-device feature-to-head replay 通过；
- [x] 1,775/1,775 fresh full-model replay 通过；
- [x] result JSON/JSONL schemas、aliases、score direction 和 threshold 校验；
- [x] result/report 内 artifact links 和 SHA-256 由机器产物填写；
- [x] 扩展回归测试通过；
- [x] 最终 audit 状态为 `replay_audit_passed`；
- [x] 无残留 draft 占位符；
- [x] 提交候选范围只包含 Community Forensics 本方法的正式文件。

正式测试结果：冻结环境中 **199 passed, 4 non-failing warnings,
105.83 seconds**。Warnings 来自 PyTorch deprecated API 提示和 system-site
APEX docstring escape，不是测试失败或结果差异。

最终完整性结论：正式运行、双 smoke、独立全链路重放和扩展回归均通过，
Community Forensics Balanced250 T1 结果完整可复核。
