# OmniAID-DINO v2（Mirage / Auto Router）在 Mouse full275 上的正式结果

日期：2026-07-25（UTC）

正式 run：
`omniaid_dino_v2_mirage_auto_mouse_canonical_v1_full275_20260725`

核心证据：
[run manifest](../results/opensource/omniaid/omniaid_dino_v2_mirage_auto_mouse_canonical_v1_full275_20260725/run_manifest.json)、
[逐图结果](../results/opensource/omniaid/omniaid_dino_v2_mirage_auto_mouse_canonical_v1_full275_20260725/results.jsonl)、
[正式汇总](../results/opensource/omniaid/omniaid_dino_v2_mirage_auto_mouse_canonical_v1_full275_20260725/summary.json)。

## 1. 结论摘要

OmniAID-DINO v2 的官方发布和本地实现都通过了严格门禁，但在 Mouse
小面积局部植入上未发现可靠、可用的检测能力。

- 覆盖为 550/550 张图、275/275 个 real/forged 配对，错误、缺失和未配对
  图像均为 0。
- 图像级 AUROC 为 **0.499636**，95% 配对 bootstrap CI
  **[0.496661, 0.502401]**；AP 为 **0.505429**，TPR@5%FPR 为
  **0.058182**。排序与随机水平一致。
- 官方分数为 float32 softmax class-1 fake score；在严格 `score > 0.5`
  下，`TP/FP/FN/TN = 8/8/267/267`，forged recall 只有
  **0.029091**。
- 275 对图的阈值判定完全不变：267 对 `0 → 0`，8 对 `1 → 1`，
  没有 `0 → 1` 或 `1 → 0`。
- forged 分数严格高于 matched real 的 pair 为 138，对应 137 个 loss、
  0 个 tie；paired ranking accuracy 为 **0.501818**，95% CI
  **[0.447273, 0.560000]**，精确双侧 sign-test `p = 1.0`。
- `forged - real` 平均分差仅 **0.000575010**，95% CI
  **[-0.000497496, 0.001597893]**；中位数只有
  `7.34e-7`。
- 同一任务的 real/forged 分数 Pearson 相关为 **0.997639**，
  Spearman 为 **0.997833**。这与输出主要受配对共享底图主导、Mouse
  植入只造成微小扰动的解释一致，但相关性本身不证明因果。
- 语义路由也几乎不随植入改变：274/275 对的 top-2 semantic expert
  集合完全相同，只有 1 对发生变化；固定 Artifact expert 的 gate 在全部
  550 张图中都严格为 1。

因此，正式结论是：**OmniAID 的 semantic/artifact 解耦和现代 Mirage
训练没有转化为对这组小面积局部植入的可靠敏感性。模型能在作者的整图
AIGC 任务上很强，与它在 Mouse 条件上接近随机并不矛盾。**

本方法只输出整图 fake score 和分类路由权重，属于 T1 whole-image AIGC
detection。它不输出编辑概率图、bbox 或 mask，因此 T2 和 joint
localization gate 必须记 N/A。D4 同域完全合成整图对照仍待建立；当前
Mouse forged 不能替代该对照。

## 2. 方法身份与主版本冻结

OmniAID 对应论文
[OmniAID: Decoupling Semantics and Artifacts for Universal AI-Generated Image Detection in the Wild](https://arxiv.org/abs/2511.08423)。
官方仓库将其标记为 ICML 2026，并在 2026-02-03 增加 DINOv3 backbone。

本次固定：

```text
GitHub: https://github.com/yunncheng/OmniAID
commit: 40749406fbcd8893c11a160edf4a72a2d4dc7056

official Space: https://huggingface.co/spaces/Yunncheng/OmniAID-Demo
commit: cf99ed518af8b7256854d01994d6e41165553bb3

official model repository: https://huggingface.co/Yunncheng/OmniAID
revision: 279cae7398ac6636f46fc4668f755f11210b36bf
```

主 checkpoint 是当前 README 标注为 Recommended、官方 Space 默认选择的
`OmniAID-DINO v2`：

```text
filename: ckpt/checkpoint_omniaid_dino_v2.pth
bytes:    3,238,483,725
SHA-256:  8135cf83a7acbd3d88e457062f7ad693b1f2e27ffc8d5ae7ec73fcb5de806ea9
```

这个选择在查看任何 Mouse 模型分数前冻结。没有在 DINO v2、CLIP v2、
v1 或 GenImage checkpoint 之间根据 Mouse 结果选优，也没有 ensemble。
CLIP v2 只保留为后续吞吐/许可敏感性候选。

需要特别区分：

- 原始论文发布于 2025-11；
- DINOv3 v2 是 2026 年加入的官方推荐 release；
- 因此本 run 是**当前官方 release inference**，不是论文表格的严格复现；
- checkpoint 内嵌训练参数列出 Mirage-Train 和 DDA-COCO 两个路径，
  不能把它简写成只在单一 GenImage 或 Mouse 数据上训练。

## 3. 原理：为什么它在整图 AIGC 上有潜力

普通整图 AIGC detector 容易把两类线索混在同一个表征中：

1. **semantic flaw**：人物、动物、物体、场景、动漫等内容域中的语义异常；
2. **artifact**：与具体内容无关、由生成器或合成管线产生的底层痕迹。

OmniAID 用 hybrid mixture-of-experts 显式拆开这两部分：

- 5 个可路由 semantic expert：Human、Animal、Object、Scene、Anime；
- 1 个 universal Artifact expert；
- router 每图从 5 个 semantic expert 中选 top-2，权重 softmax 后和为 1；
- Artifact expert 不参与竞争，始终额外以权重 1 开启；
- 所以每张图的 6 维 `final_gates` 总和严格为 2。

官方 DINO v2 推理图包含两份 DINOv3 ViT-L/16：

1. `feature_extractor` 输出 1024 维 CLS feature，供轻量 router
   `1024 → 256 → 5` 决定 semantic experts；
2. MoE DINO 主干执行最终分类。

DINOv3 配置为 hidden size 1024、MLP 4096、24 层、16 heads、patch size
16、4 register tokens。输入 448×448 时产生 28×28 个 patch token，加
CLS 和 4 registers，共 789 tokens。

MoE 主干的每一层 q/k/v/o 四个 `1024×1024` attention linear 都被替换：

```text
24 layers × 4 projections = 96 SVD-MoE linear modules
main rank = 1023
rank per expert = 1
experts = 6
```

每次前向使用冻结的 rank-1023 主权重，再叠加 router 选中的两个 rank-1
semantic expert 和固定的 rank-1 artifact expert。该设计的潜在优势是：

- semantic expert 可以学习“生成了什么”相关的异常；
- Artifact expert 可以学习“如何生成”相关的通用痕迹；
- top-2 routing 避免所有语义模式在一个单体分类器中互相干扰；
- rank-1 expert 限制适配自由度，有助于保留 DINOv3 的通用表征。

但这种结构仍是**整图分类器**。Mouse 编辑平均只占原图约 0.1685%，
局部痕迹会在全局 CLS 聚合中被大面积真实背景稀释；正式结果显示 router
和分类头都几乎不因植入而改变。

## 4. Checkpoint 安全与严格加载

checkpoint 是训练期 PyTorch archive，顶层键固定为：

```text
model, optimizer, epoch, scaler, args
epoch = 0
```

现代 PyTorch 检查到的 unsafe global census 只有：

```text
argparse.Namespace
```

runner 只在 `weights_only=True` 下显式 allowlist 这一个类型，使用
`mmap=True`，没有启用任意 pickle 代码执行。`model` state 的冻结合同为：

```text
tensors:              2,852
FP32 elements:        808,835,239
ordered-key SHA-256:  d894e539c44bfd3b036413db0e5c91d7de75552df5af2153ca4a83bc40e7d788
schema SHA-256:       1b5a03a08369fa7dc5034b1b9aa8a4757295386afd3c91f093ef41b6e2c9b67d
missing keys:         []
unexpected keys:      []
strict load:          true
```

官方 DINOv3 base
`facebook/dinov3-vitl16-pretrain-lvd1689m` 是 gated model，固定 revision
为 `ea8dc2863c51be0a264bab82070e3e8836b02d51`。本 benchmark 没有绕过
gate 或使用第三方镜像。OmniAID checkpoint 已包含
`feature_extractor` 和 MoE DINO 的完整权重，因此 adapter：

1. 按官方 DINOv3 config 在 meta device 构造 shape-only base；
2. 调用固定 Space 中的官方 `OmniAID_DINO` inference class；
3. 要求 2,852/2,852 key 顺序和 shape 完全一致；
4. 用 `assign=True` 严格装载完整 checkpoint；
5. 只按 Transformers 官方公式重新物化两份非持久化 RoPE
   `inv_freq[16]` buffer。

这不是近似模型，也没有用随机 base 权重参加推理。checkpoint 覆盖完整
forward state；meta 构造只避免 gated base 的重复下载和构造期 SVD。

## 5. License 边界

OmniAID GitHub README badge、HF model metadata 和 Space metadata 都写
MIT，但固定 GitHub/Space checkout 中没有跟踪的 `LICENSE`、`COPYING`
或 `NOTICE` 文本。更重要的是，其 backbone 来自 Meta DINOv3，官方模型卡
标记为自定义 `dinov3-license` 且需要接受 gate 条件。

因此 manifest 保守记录：

```text
tracked_license_file_present: false
code_license_text_verified:    false
dinov3_base_license:           custom_dinov3_license
commercial_use_cleared:        false
benchmark_role:                research_evaluation_only
```

本次仅把这些资产用于研究评测记录；这不是许可意见，也不构成授权。不能
从 MIT badge 推导出整套代码、衍生 checkpoint 和 backbone 已经获得商用
许可；产品化和再分发前必须逐层做法律审核。

## 6. 官方推理协议

本 run 严格复现当前官方 Space 的 DINO v2 / Auto Router 路径：

```text
PIL.Image.open
→ convert("RGB")
→ torchvision.transforms.Resize([448, 448])
→ ToTensor()
→ ImageNet normalize
→ float32, batch=1, eval, no AMP
→ automatic router
→ softmax(class logits, dim=1)[class 1]
```

归一化：

```text
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
```

重要细节：

- `Resize([448,448])` 是指定高宽、`antialias=true` 的直接 bilinear
  拉伸，不保持宽高比；
- 不做 center crop、face alignment 或 EXIF transpose；
- 448 能被 patch size 16 整除；
- 使用 `manual_weights=None`，没有手工调整专家权重；
- 正式 score 是 float32 softmax class 1；
- decision 为严格 `score > 0.5`，恰好 0.5 判 real；
- score 是官方输出，但未在 Mouse 域校准，不能解释成可信后验概率。

所有 275 个 forged GT 区域在几何上都完整位于被直接 resize 的整幅画布
中，所以 `edit_visibility=full`、`edit_visible_gt_fraction=1.0`。这只说明
没有像 crop 方法那样丢掉编辑，不说明局部信号足够大。

## 7. 官方样例与线上服务对齐

正式 Mouse 运行前，使用固定 Space 的 4 张示例图做门禁。官方没有发布
论文级 full-precision golden；这里冻结的是 2026-07-25 观察到的官方
Space service 输出，并把它明确标成 service oracle，而不是作者论文数字。

| fixture | 本地 CUDA fake score | 官方 Space score | 绝对差 |
|---|---:|---:|---:|
| `real_0.jpg` | 0.239961311 | 0.239963502 | 0.000002190 |
| `real_1.jpg` | 0.078057170 | 0.078057334 | 0.000000164 |
| `fake_0.jpg` | 0.857224584 | 0.857226014 | 0.000001431 |
| `fake_1.jpg` | 0.609500527 | 0.609498858 | 0.000001669 |

最大差约 `2.2e-6`，远小于预注册的 `5e-5` service 容差。每个 fixture
连续两次完整 forward 的以下数组全部 bit-exact：

- `pooler_output[1024]`；
- `class_logits[2]`；
- `routing_feature[1024]`；
- `semantic_top_k_indices[2]`；
- `semantic_top_k_gates[2]`；
- `final_gates[6]`。

本地 CUDA logits 和 gates 还通过 `1e-5` 的固定 runtime-regression 门禁。
4 个样例在整个门禁阶段计算的 Mouse model score 数为 0。

## 8. 双 smoke 与正式覆盖

正式运行前执行了两个彼此独立的 5-pair/10-image CUDA smoke：

- `omniaid_dino_v2_mirage_auto_mouse_pair5_cuda_smoke_a_20260725`；
- `omniaid_dino_v2_mirage_auto_mouse_pair5_cuda_smoke_b_20260725`。

A/B 逐 sample 比较确认：

- 10/10 的 score、logits、margin、decision 和 preprocess record 完全相同；
- 10/10 的六类 NPZ 数组全部逐元素 bit-exact；
- feature、routing feature、logits 和 gates 的最大绝对差均为 0；
- 全部非运行耗时类 summary 指标完全一致。

正式 full run 从 2026-07-25 21:51:08 UTC 到 21:52:58 UTC：

| 项目 | 数量 |
|---|---:|
| expected images | 550 |
| physical result rows | 550 |
| valid images | 550 |
| error / missing images | 0 / 0 |
| expected / complete pairs | 275 / 275 |
| paired valid images | 550 |
| unpaired valid images | 0 |

config fingerprint：

```text
248c9e9cfc9185ab953d6f570c5335f1574b126c2b1134e8d1aa6bce876942e2
```

每个 sample 保存一份无 pickle 的 NPZ，固定包含上述六类数组；550 个
artifact 均为 9,848 bytes，总计 5,416,400 bytes。

## 9. 正式检测结果

### 9.1 图像级排序

| 指标 | 点估计 | 95% 配对 bootstrap CI |
|---|---:|---:|
| AUROC | 0.499636 | [0.496661, 0.502401] |
| Average Precision | 0.505429 | [0.504109, 0.512723] |
| TPR @ 5% FPR | 0.058182 | [0.040000, 0.072727] |
| real-only threshold | 0.391419 | [0.327483, 0.540386] |
| 实际 FPR | 0.047273 | [0.036364, 0.047273] |

数据正负严格平衡，所以随机排序的 AUROC 和 AP 基线约为 0.5。当前 AUROC
CI 覆盖 0.5，AP 也只有随机量级，5% FPR 只能召回约 5.8% forged。

### 9.2 发布阈值 `score > 0.5`

| 指标 | 数值 | 95% CI（如有） |
|---|---:|---:|
| TP / FP / FN / TN | 8 / 8 / 267 / 267 | — |
| accuracy | 0.500000 | [0.500000, 0.500000] |
| balanced accuracy | 0.500000 | [0.500000, 0.500000] |
| precision | 0.500000 | [0.500000, 0.500000] |
| recall | 0.029091 | [0.010909, 0.050909] |
| specificity | 0.970909 | [0.949091, 0.989091] |
| F1 | 0.054983 | [0.021352, 0.092409] |

pair 判定迁移：

| real decision → forged decision | pairs |
|---|---:|
| `0 → 0` | 267 |
| `0 → 1` | 0 |
| `1 → 0` | 0 |
| `1 → 1` | 8 |

所以 50% accuracy 不是成功检测一半 forged，而是每个 pair 都给相同判定，
在一真一假的平衡设计中自然得到 50%。

### 9.3 分数与配对敏感性

| kind | n | mean | median | p05 | p95 | min | max |
|---|---:|---:|---:|---:|---:|---:|---:|
| real | 275 | 0.119884 | 0.075670 | 0.015339 | 0.384541 | 0.004691 | 0.680171 |
| forged | 275 | 0.120459 | 0.076250 | 0.015516 | 0.401739 | 0.004689 | 0.708250 |

定义 `delta = forged_score - real_score`：

| 指标 | 结果 |
|---|---:|
| mean delta | 0.000575010 |
| mean delta 95% CI | [-0.000497496, 0.001597893] |
| median delta | 0.000000734 |
| p05 / p95 | -0.007787673 / 0.015400214 |
| min / max | -0.048454076 / 0.047701001 |
| wins / losses / ties | 138 / 137 / 0 |
| paired ranking accuracy | 0.501818 |
| paired ranking 95% CI | [0.447273, 0.560000] |
| exact two-sided sign-test p | 1.0 |
| real-forged Pearson r | 0.997639 |
| real-forged Spearman rho | 0.997833 |

无论按平均幅度、方向、排序还是发布阈值，都未发现可靠、可用的局部植入
检测信号；这不等同于证明内部表征中的局部信息严格为零。

### 9.4 分 domain

| domain | pairs | AUROC [95% CI] | AP [95% CI] | TPR@5%FPR | TP/FP/FN/TN | paired rank | mean delta |
|---|---:|---:|---:|---:|---:|---:|---:|
| lodging | 147 | 0.500856 [0.496645,0.505485] | 0.508740 [0.506475,0.521527] | 0.047619 | 8/8/139/139 | 0.503401 | 0.000771294 |
| restaurant | 128 | 0.498657 [0.492795,0.503906] | 0.512240 [0.508990,0.530364] | 0.062500 | 0/0/128/128 | 0.500000 | 0.000349589 |

lodging 胜/负为 74/73，restaurant 为 64/64；两个 domain 的 sign-test
都为 `p=1.0`。不存在一个 domain 可以支持部署结论。

## 10. Router 与 feature 诊断

这部分是从正式结果和六数组 artifact 做的只读诊断，不是额外选模依据。

semantic expert 被选择的次数：

| expert | real | forged |
|---|---:|---:|
| Human (0) | 33 | 32 |
| Animal (1) | 10 | 11 |
| Object (2) | 237 | 237 |
| Scene (3) | 268 | 268 |
| Anime (4) | 2 | 2 |

230/275 对都选择 `(Object, Scene)`。274/275 对的 semantic top-2 集合在
real 与 forged 间完全不变；唯一变化是在 `(Human,Object)` 和
`(Animal,Object)` 之间切换一次。semantic gate 的 paired L1 差：

```text
mean   0.0155923
median 0.0039201
p95    0.0667368
max    0.3595898
```

paired cosine similarity：

| 表征 | mean | median | p05 | min |
|---|---:|---:|---:|---:|
| routing feature | 0.998332 | 0.999852 | 0.995081 | 0.928769 |
| final MoE pooler feature | 0.998017 | 0.999760 | 0.988167 | 0.930751 |

这与 score 高相关和判定零翻转一致：Mouse 植入没有稳定改变语义路由，
组合模型输出也没有发生阈值翻转。现有 artifact 只保存 gate 和组合后的
pooler/logits，没有保存每个 expert 的独立输出，因此不能把失败单独归因给
Artifact expert。也不能把 gate 当作定位图；它只有 6 个整图专家权重。

## 11. 校准与面积探索

以下不是冻结主指标，只是从 550 条正式 score 只读计算：

| 诊断 | 数值 |
|---|---:|
| binary NLL | 1.355970 |
| Brier score | 0.409821 |
| ECE-15 | 0.387818 |

在 50/50 平衡数据上，恒定 0.5 的 NLL 约 0.693、Brier 为 0.25；当前值
更差，说明 softmax score 在 Mouse 域没有可用概率校准。不能在同一 275 对
上拟合校准器再报告测试性能。

编辑面积与 pair delta 的探索：

| 相关 | 系数 | p |
|---|---:|---:|
| Pearson | 0.105993 | 0.079321 |
| Spearman | 0.029147 | 0.630334 |

两者均不显著，且面积与内容、domain、位置共同变化；不能声称编辑面积增加
会让 OmniAID 更容易检测。

## 12. 性能与运行环境

```text
Python       3.12.3
PyTorch      2.8.0.dev20250627+cu128
torchvision  0.23.0.dev20250627+cu128
transformers 4.57.3
NumPy        2.2.6
Pillow       12.0.0
device       cuda:0 / NVIDIA L20Z
batch size   1
dtype        float32
autocast     false
TF32         disabled
deterministic algorithms true
```

单图 timed inference/replay segment（包含 full forward、head/softmax
重放、margin 与 CUDA synchronize；不含 Pillow/resize、JSON 和 NPZ 写盘）：

| min | mean | median | p95 | max |
|---:|---:|---:|---:|---:|
| 82.951 ms | 84.669 ms | 83.782 ms | 87.650 ms | 105.225 ms |

每行 peak CUDA allocated memory 为 `3,372,986,880 bytes`
（约 3.14 GiB）。该值只对应当前 batch=1、float32 和指定软件栈。

## 13. 独立审计

独立审计器不信任 runner 的 JSON、NPZ、score 或 summary，并且不 import
`run_omniaid.py`。它独立验证 source/checkpoint/config、重新构造官方
Space graph、重新解码和预处理全部 550 张图，并执行了 550 次 fresh
full-model forward。正式结果为：

```text
expected/latest/successful images  550 / 550 / 550
complete pairs                     275
fresh full-model forwards          550
artifact replays                   550
max abs pooler/logits/router diff  0 / 0 / 0
max abs top-k/final-gates diff     0 / 0 / 0
max abs head/router/prob/margin    0 / 0 / 0 / 0
preprocess records exact           true
all decisions exact                true
stored summary exact recompute     true
fresh-forward summary exact        true
```

开发审计器时，真实 run 门禁还暴露并修复了两个测试夹具没有覆盖的问题：
完整 release manifest 是冻结身份字段的超集，以及 float32 semantic gate
之和不保证逐位等于数学常数 1。两处均先增加回归测试，再重新从头审计；
最终报告的所有数值来自修复后的成功重放。

正式审计产物：
[independent_audit.json](../results/opensource/omniaid/omniaid_dino_v2_mirage_auto_mouse_canonical_v1_full275_20260725/independent_audit.json)。

## 14. 文件哈希

### 14.1 正式 run

| 文件 | SHA-256 |
|---|---|
| `expected_inputs.jsonl` | `e4cb3d6a78fa68f06341457e2234c630a455a9b6b9789e59abf45c15b292060a` |
| `results.jsonl` | `0f78f94fd338be10c9b6ed4748610fe6276fc65267aeac40909653c7d0a92ec1` |
| `summary.json` | `d45180b2a25f1f467893a18860a27e2d141856375606bd2edf7958c9aad3e051` |
| `run_manifest.json` | `f5ba2b405cb4d09b189cf0e303e1cfd33590c5b51a6a229c78d4482c99dfbb37` |
| `independent_audit.json` | `9ece4d819b744eeb8845ca54bfacf766debefef9bdfd1f545e37072dc66aa7d5` |

### 14.2 双 smoke

| run/file | SHA-256 |
|---|---|
| smoke A `results.jsonl` | `ecd6b40f1ca3cc4e1d96ad0f1769f86c989ef140fc7e4a1ac4dea51057ac0eee` |
| smoke A `summary.json` | `6bee70eef9505374c4d9ed44ad0952129e2265387d92c0cd3d1be730d760d1a1` |
| smoke A `run_manifest.json` | `1bd72e30b136ef55d57f526a4452771cd03b88109f3a55824c3a68ac1699797a` |
| smoke B `results.jsonl` | `2a46f1fb025d3b33649789b195439a76579022170cf3b858565ea6817c10d434` |
| smoke B `summary.json` | `08c7d790f07bd688faa5c17d5943c055802589f101392ab91ef8acd0c1df6c08` |
| smoke B `run_manifest.json` | `a4f036c3d1a46b5017dcceec5581556e7e62a4ab922cf41f1894d09f469fb27b` |

## 15. 安全重跑

实现入口：
[run_omniaid.py](../eval/opensource/run_omniaid.py)、
[analyze_omniaid_run.py](../eval/opensource/analyze_omniaid_run.py)、
[omniaid_metrics.py](../eval/opensource/omniaid_metrics.py)。

已有正式 run ID 不应再交给 runner，即使使用 `--resume` 也会更新 summary
和 manifest 的时间/执行元数据。若要物理验证正式结果而不修改其目录，可把
新审计写入临时文件：

```bash
omniaid_python=/root/.cache/claimforge/venvs/omniaid/bin/python
omniaid_audit_tmp="$(mktemp /tmp/omniaid-audit.XXXXXX.json)"

"$omniaid_python" -m eval.opensource.analyze_omniaid_run \
  --run-id omniaid_dino_v2_mirage_auto_mouse_canonical_v1_full275_20260725 \
  --device cuda:0 \
  --output "$omniaid_audit_tmp"
```

这会重新执行 550 次 forward，但只写临时审计文件；正式四个核心文件和
本报告记录的 `independent_audit.json` 均不变。

真正独立复跑必须使用不存在的新 ID：

```bash
omniaid_python=/root/.cache/claimforge/venvs/omniaid/bin/python
omniaid_new_run_id=omniaid_dino_v2_mirage_auto_mouse_canonical_v1_full275_rerun_20260725_01

test ! -e "results/opensource/omniaid/$omniaid_new_run_id"

"$omniaid_python" -m eval.opensource.run_omniaid \
  --run-id "$omniaid_new_run_id" \
  --device cuda:0

"$omniaid_python" -m eval.opensource.analyze_omniaid_run \
  --run-id "$omniaid_new_run_id" \
  --device cuda:0
```

## 16. 正确论文表述

本结果支持：

> 当前官方推荐的 OmniAID-DINO v2 在 4 个官方 Space 示例上与线上服务
> 对齐，但在 CLAIMFORGE Mouse 小面积局部植入条件上，图像级排序、配对
> 方向、发布阈值和 semantic routing 都接近不变。

本结果不支持：

- “OmniAID 对所有 AIGC 都失效”；
- “OmniAID 的整图 Mirage 结果被推翻”；
- “router gate 可以定位 Mouse”；
- “当前 softmax score 是校准概率”；
- “MIT badge 自动覆盖 DINOv3 的自定义许可”。

下一项必要实验是使用同一 detector 评估 matched real 与同域完全合成整图。
如果它在 fully synthetic contrast 上恢复高性能，而在 Mouse 上仍接近随机，
才完整支持“检测器没有坏，而是局部生成比例导致逃逸”的论文主故事。
