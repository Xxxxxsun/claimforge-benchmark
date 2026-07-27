# IML-ViT（CAT/TruFor protocol checkpoint）在 Balanced250 上的正式结果

日期：2026-07-27（UTC）

> **文档状态：正式完成；r2 双 smoke、1,025-image formal、独立
> artifact audit 与 fresh full-model replay 全部通过**
>
> IML-ViT 是原生像素定位方法，没有独立图像级 classification head。
> 本报告因此只发布 T2；没有把 heatmap 的最大值、均值、面积或
> nonempty 状态改造成 T1。三类 full-frame AIGC 图没有进入本 run，
> 不能用本结果回答整图 AIGC 检测问题。

正式 run：
`imlvit_cat_protocol_balanced250_v1_full1025_r2_20260727`

Formal immutable run-config fingerprint：
`5d7e54f183f11379337eb6681026786da522f57642088b1a5ed1b6c63b2928b1`

核心机器证据：

- [run manifest](../results/opensource/imlvit/imlvit_cat_protocol_balanced250_v1_full1025_r2_20260727/manifest.json)：
  `439a4bc258c99b13464ab3488d9c4ee1f66abda061cddc5d99484dd60593c3f5`
- [expected inputs](../results/opensource/imlvit/imlvit_cat_protocol_balanced250_v1_full1025_r2_20260727/expected_inputs.jsonl)：
  `a952b6adcbd5af20ce635a5929c7045350b9f8e6729bae8d9cbf1978050f2cca`
- [逐图结果](../results/opensource/imlvit/imlvit_cat_protocol_balanced250_v1_full1025_r2_20260727/results.jsonl)：
  `84eadf35603d2faa175d56183ec2211b6a64732bbf1d5ee0b03fe50a3e17d1d2`
- [coverage summary](../results/opensource/imlvit/imlvit_cat_protocol_balanced250_v1_full1025_r2_20260727/summary.json)：
  `fb6d58984bb3eef1e615487ef063bea13e6d2727fe966922c1c28fbcb7eadc45`
- [Balanced250 T2 metrics](../results/opensource/imlvit/imlvit_cat_protocol_balanced250_v1_full1025_r2_20260727/balanced250_metrics.json)：
  `ff90b7d273cac6728f241ac6c2a9872e8d5982ebd30e07aaf8061f2dc43f8e39`
- [independent audit](../results/opensource/imlvit/imlvit_cat_protocol_balanced250_v1_full1025_r2_20260727/independent_audit.json)：
  `692e4f45c845d4817f0a1239a52dc0017bfc7f866803925da9d8c4aacd64b2e8`
- [双 smoke comparison](../results/opensource/imlvit/_reports/imlvit_cat_protocol_balanced250_v1_smoke5x4_a_r2_20260727_vs_imlvit_cat_protocol_balanced250_v1_smoke5x4_b_r2_20260727.smoke_comparison.json)：
  `8118866ee1352476bc753838aa2f76798bf7ea1db278cdea8d1131f937c12c70`

## 1. 结论摘要

IML-ViT 在 Balanced250 上有明确但不均匀的 local-forgery localization
能力。它对 Cat 植入很强，对 tiny Mouse 有弱到中等信号，而对大面积
Trash-can 的连续排序尚可、固定阈值 coverage 很低。

- formal coverage 为 **1,025/1,025**，零 error、missing、duplicate
  和 superseded attempt；
- 三类 local 等权的 per-image pixel AP 为
  **0.404412 [0.380006, 0.428168]**；
- 三类 local 等权的 per-image F1/IoU 为
  **0.256760 / 0.207278**；
- Local Mouse/Cat/Trash-can 的 per-image pixel AP 分别为
  **0.159124 / 0.679551 / 0.374561**；
- Local Mouse/Cat/Trash-can 的 per-image F1 分别为
  **0.147143 / 0.549470 / 0.073668**；
- 750 张 local 的 pooled-pixel precision/recall/F1/IoU 为
  **0.461325 / 0.042533 / 0.077886 / 0.040521**；
- 275 张 real 的 mean per-image false-positive fraction 为
  **0.008045**（约 0.804%），micro fraction 为 **0.009023**；
- 每张 real 的 official strict mask 都非空；把 nonempty map 当成
  forged image verdict 会得到 275/275 image-level false positives；
- 双 smoke 的 20 张 computational projections 和 80 个 artifacts
  全部 byte-exact；
- formal audit 完整检查 1,025 张图和 4,100 个 artifacts；
- fresh replay 重新 preprocess、strict-load、forward 和 postprocess
  **1,025/1,025**，raw logits、model/native maps、official masks 和
  逐图 strict localization metrics 全部 exact。

需要特别说明阈值协议。官方 IML-ViT artifact、逐图 ledger 与 fresh
replay 使用 strict `score > 0.5`；跨方法共享 Balanced250 T2 reducer
冻结为 `score >= 0.5`。Formal maps 中恰好有 **2 个 exact-0.5 pixels，
分布在 2 张图**，所以两种 operator 确实不等价。r2 没有隐瞒或删除这些
像素，而是分别保留两个合法协议并明确记录 `equivalent=false`。

## 2. 官方来源、checkpoint 与许可证边界

方法对应
[Image Manipulation Localization Using Multi-scale Feature Fusion and Vision Transformer](https://arxiv.org/abs/2307.14863)，
正式执行作者
[IML-ViT repository](https://github.com/SunnyHaze/IML-ViT)
commit `07dd2be0f4ea27a5c97c9fa5ffbe236733833eac`，tree
`cfae1470a71de9f146df3c13e994bea41d70624a`。

正式权重是作者 Google Drive 发布的 CAT/TruFor-protocol checkpoint：

| 项 | 冻结值 |
|---|---|
| original filename | `iml-vit_checkpoint_trufor_20231104.pth` |
| provider | official author Google Drive |
| Drive file ID | `1jlXw97GkyBbY4u5-e_liuhahKSQWCAFu` |
| release folder ID | `1Ztyiy2cKJVmyusYMUlwuyPecBefTJCPT` |
| announcement commit | `5ad22146b1223eac841fa3e0e28c1c4e8948cc95` |
| bytes | 367,195,954 |
| SHA-256 | `9fa9ae88cafeb6eab28c2afd5bef74679416cf0a790b2370fa6a6fb4c122c58c` |
| container | raw `collections.OrderedDict` state dict |
| state keys / elements | 212 / 91,778,242 |
| model parameters / buffers | 91,777,729 / 513 |
| state dtypes | 211 float32、1 int64 |
| load | `torch.load(..., weights_only=True)` |
| strict load | true；missing/unexpected 为 0/0 |

文件名含 `trufor`，但执行模型仍是 IML-ViT。作者 README 将其标为
CAT-Net protocol checkpoint，并说明 TruFor 使用相同 protocol；本报告
不把它误写为 TruFor model。

许可证必须把代码和权重分开：

- project repository 的 `LICENSE` 是 MIT，SHA-256
  `7bce5d24d372c0abbf618951988ee2dc072e60027c55615f0229bdab0dad73c3`；
- MIT 记录只明确覆盖 repository code，代码 commercial use 与
  redistribution permission 为 true；
- official checkpoint release 没有发现单独 license/terms；
- 没有证据表明 project code license 自动扩展到 weight bytes；
- 因此 checkpoint 的 commercial-use clearance **未建立**。

公开下载、完成 benchmark 或保存结果不等于获得商业产品权利。训练数据
权利、第三方依赖和具体法域仍需单独审查；本报告不是法律意见。

## 3. 方法原理与能力边界

IML-ViT 把高容量 ViT encoder 与 simple feature pyramid 和
one-channel localization decoder 结合：

1. ViT-B/16 将图像分成 patch tokens；
2. window attention 保留可控计算量，选定 block 的 global attention
   提供跨远距离区域比较；
3. simple feature pyramid 从 transformer 表征恢复多个空间尺度；
4. decoder 融合五级特征，输出一张 dense manipulation-logit map；
5. 连续 map 直接用于 pixel ranking，固定阈值 mask 用于 F1/IoU。

这种设计比单一低分辨率 classification token 更适合局部植入：它既能比较
全图不同区域的统计一致性，也能通过多尺度特征恢复边界和小区域信息。
Balanced250 的 Cat 结果证明 checkpoint 能把部分生成式对象与真实背景
明显区分。

限制同样清楚：

- 16×16 patch 和大图 downscale 会压缩 tiny object evidence；
- 训练 distribution 中常见的 splice、copy-move、压缩或边界信号不一定
  覆盖平滑 diffusion insertion；
- 大对象内部 evidence 可能只在局部区域超过 0.5；
- 背景纹理也会产生稳定 positive response；
- localization head 不是独立 image-integrity classifier。

这些是与架构和观测一致的解释，不是 causal ablation。不能据此断言某一
attention block 是成功或失败的唯一原因，也不能外推到重新训练的
IML-ViT variant。

## 4. Frozen executable contract

正式入口：

- [Balanced250 runner](../eval/opensource/run_imlvit_balanced.py)
- [independent analyzer](../eval/opensource/analyze_imlvit_balanced.py)
- [legacy official-contract adapter](../eval/opensource/run_imlvit.py)
- [legacy strict IML-ViT metrics](../eval/opensource/imlvit_metrics.py)
- [shared Balanced250 T2 reducer](../eval/opensource/balanced250_localization_metrics.py)

冻结文件 SHA-256：

| File | SHA-256 |
|---|---|
| Balanced250 runner | `41fbbeca51a1945c8ad5f3acf07479284a0b1d2e8b21eaf2d26f49b7a1b49c5c` |
| independent analyzer | `46a981e5e8b53e8fd5185e5aa9a6af51fb0b868ba904b835cdf08182296d8ab7` |
| runner tests | `cddff6128de02b6c2c080c1921760ff3471c41c22e3d85a4e468dcd95ca9c7a8` |
| analyzer tests | `51f6ac394902967e6473772c2f92ffede7d8dfb65237805f4c178c89e5728049` |
| legacy official adapter | `98d0fdcabf63b29111fb778fdffc8734563e258ca62efef81eb982a3208be60b` |
| legacy strict metrics | `7c37d4afc721028a7ca2eac75ab68eddcc39a45afc4398ea09dcbd2b517141f5` |
| shared Balanced250 T2 reducer | `83ac07257078fc41276742fa4b9f2eb936ac51c8ff93bf1253b8c45f2b704b2a` |
| canonical release loader | `6f4261dd2bc5335722aae253d851c363f663aa89cf42913647931d9cc60ac892` |
| run-contract helper | `95d9e648840ce648fd75d441d440e33ca2b9ede509aa548f040c4ac1bc4356b6` |

### 4.1 Preprocess

| 组件 | 冻结行为 |
|---|---|
| input | canonical JPEG original bytes |
| decoder | Pillow `Image.open(...).convert("RGB")` |
| large image | 仅当 `max(H,W)>1024` 时保持比例缩到 longest side 1024 |
| interpolation | Albumentations 1.3.0 `LongestMaxSize` / `cv2.INTER_LINEAR` |
| small image | 不放大 |
| canvas | 1024×1024，image 放在左上 |
| padding | 仅右/下 raw RGB zero，发生在 normalization 前 |
| normalization | ImageNet mean/std |
| crop / re-encode | none / false |

该路径遵循 paper section 4.1 的 conditional fit-and-pad 设计。它没有使用
会裁掉大图右侧或底部内容的 literal demo crop，也没有把所有图强制拉伸
成正方形。

### 4.2 Forward 与 native restore

数值路径为：

```text
3×1024×1024 normalized RGB
  -> ViT-B/16 window/global-attention encoder
  -> five-level simple feature pyramid
  -> one-channel prediction head
  -> 256×256 float32 logits
  -> bilinear logits to 1024×1024, align_corners=False
  -> sigmoid exactly once
  -> crop right/bottom padding
  -> bilinear probability to native geometry, align_corners=False
```

Formal 保存四类 artifacts：

- 1024-space raw logits FP32 NPY；
- 1024-space sigmoid probability FP32 NPY；
- native-resolution probability FP32 NPY；
- native official strict-threshold PNG mask。

Native probability 在 threshold 前恢复；padding 不参与 native metrics。
Model-space diagnostic 也只使用 valid resized-content rectangle。

### 4.3 T1、full-frame 与 GT

冻结 capability：

```text
primary task:                  T2 native localization only
independent image head:        false
valid_for_t1:                  false
map statistic promoted T1:     false
full-frame T1/T2:              N/A / N/A
```

Local GT 是 decoded source/forged RGB exact difference；conditioning box
不是 GT。Real GT 是 all-zero，只报告 false-positive area。三类
full-frame 共 750 张没有 forward、artifact 或 T2 score-map-loader call。

## 5. r1 fail-closed 与 r2 双阈值协议

最初的 r1 runner 已完成 smoke 和 formal，但 analyzer 在完整
1,025-image artifact audit 后发现 native maps 存在 exact `0.5`，按当时
“`>` 与 `>=` 必须等价”的错误 gate 主动失败：

```text
ValueError: IML-ViT native maps contain exact 0.5 values;
official strict > and shared >= T2 reducers are not equivalent
```

r1 没有产生 publishable metrics/audit，也没有执行完成态 fresh replay。
旧目录被保留为 fail-closed 诊断证据，没有覆盖或冒充正式结果。

r2 采用新 run IDs 和 v3 schemas，重新运行 A/B smoke、formal 和 fresh。
它没有改变 model、checkpoint、preprocess、score map 或 shared reducer：

```text
official artifact / ledger / fresh mask: score >  0.5
shared cross-method Balanced250 metrics: score >= 0.5
operator equivalence assumed:            false
exact-threshold policy:                   allow and report
```

四个独立 evidence location 给出完全相同的 boundary count：

| Evidence | Exact-0.5 pixels | Images |
|---|---:|---:|
| metrics top-level | 2 | 2 |
| audit contract checks | 2 | 2 |
| independent artifact audit | 2 | 2 |
| fresh full-model replay | 2 | 2 |

两个 pixels 分别位于：

| Sample | Condition | Native coordinate `(y,x)` | GT |
|---|---|---:|---|
| `44a4f73fe43587289dab3873` | local Cat | `(794,940)` | negative |
| `fd3b74aa78bb1d08a51f6365` | local Trash-can | `(426,748)` | negative |

因此 shared `>=` 相比 official `>` 恰好多计 2 个 FP：Cat 一个、
Trash-can 一个，没有增加 TP。Pixel AP 完全不受 fixed-threshold
operator 影响；F1/IoU/confusion 有极小但真实的差异。报告的正式
跨方法 T2 tables 使用 shared `>=`，而 official PNG/fresh assertions
继续使用 strict `>`。

## 6. Balanced250 coverage 与统计设计

Canonical release 有 1,775 个唯一 score-cache inputs；IML-ViT 只选择
合法 T2 subset：

| Condition | Selected | T2 target | T1 |
|---|---:|---|---|
| real | 275 | all-zero FP area | N/A |
| local Mouse | 250 | exact difference | N/A |
| local Cat | 250 | exact difference | N/A |
| local Trash-can | 250 | exact difference | N/A |
| full-frame Mouse | 0 | N/A | N/A |
| full-frame Cat | 0 | N/A | N/A |
| full-frame Trash-can | 0 | N/A | N/A |
| **Total** | **1,025** | **1,025 applicable** | **0** |

Formal selected-ID SHA-256：
`612e08565e38cb219fe5ea94dc8193580e099455e11fa778822488dbe7071717`。

Formal accounting：

```text
expected / physical / latest rows: 1025 / 1025 / 1025
valid rows:                        1025
error / missing / unexpected:        0 / 0 / 0
duplicate IDs / superseded:           0 / 0
coverage / success fraction:        1.0 / 1.0
```

Confidence intervals 使用 1,000 次
`shared_source_content_cluster_poisson` bootstrap，root seed
`20260727`。Bootstrap unit 是明确冻结的 source-content cluster，不把
同源重复注册内容当成完全独立观察。

## 7. Native T2 结果

方括号为 1,000 次 cluster bootstrap 的 95% percentile interval。
`Target fraction` 是 condition 内 pooled pixel prevalence；AP/F1/IoU
主体是 per-image macro，不能把 prevalence 当成完全相同权重的 AP
random baseline。

### 7.1 条件级 per-image macro

| Condition | Target fraction | Pixel AP [95% CI] | Precision | Recall | F1 | IoU |
|---|---:|---:|---:|---:|---:|---:|
| local Mouse | 0.001350 | 0.159124 [0.126569, 0.191834] | 0.187436 | 0.172082 | 0.147143 | 0.101329 |
| local Cat | 0.063678 | **0.679551** [0.639468, 0.722637] | **0.762402** | **0.502396** | **0.549470** | **0.466896** |
| local Trash-can | 0.219667 | 0.374561 [0.339903, 0.406209] | 0.495674 | 0.061344 | 0.073668 | 0.053608 |
| **condition macro** | — | **0.404412** [0.380006, 0.428168] | **0.481837** | **0.245274** | **0.256760** | **0.207278** |

Official strict-ledger diagnostics：

| Condition | Nonempty masks | Any GT overlap | Median AP | Median F1 | Zero-F1 |
|---|---:|---:|---:|---:|---:|
| local Mouse | 250/250 | 133/250 | 0.022604 | 0.003965 | 117 |
| local Cat | 250/250 | 225/250 | 0.848083 | 0.764492 | 25 |
| local Trash-can | 250/250 | 189/250 | 0.424983 | 0.010185 | 61 |

“Any overlap”只要求一个 TP pixel，不能解释为高质量定位率。正式质量结论
仍来自 AP、F1、IoU 和 confusion counts。

Cat 是最稳定的 condition：median AP 为 0.848，per-image macro F1
为 0.549。Mouse 的平均 AP 仍高于极小的 0.135% pooled prevalence，
说明存在真实 ranking signal，但 117/250 张 fixed-threshold F1 为 0。
Trash-can AP 为 0.375，却只有 0.061 macro recall；模型往往给大目标内部
少量区域较高分，无法覆盖整个 object。

### 7.2 Pooled-pixel micro

| Condition | Predicted fraction | Micro precision | Micro recall | Micro F1 | Micro IoU |
|---|---:|---:|---:|---:|---:|
| local Mouse | 0.007503 | 0.040051 | **0.222623** | 0.067888 | 0.035137 |
| local Cat | 0.008575 | **0.789144** | 0.106262 | **0.187302** | **0.103328** |
| local Trash-can | 0.010288 | 0.494573 | 0.023162 | 0.044252 | 0.022627 |
| **All local pixels** | **0.008794** | **0.461325** | **0.042533** | **0.077886** | **0.040521** |

750 张 local 共包含 1,206,543,236 pixels：

```text
target positive pixels:     115,086,458
predicted positive pixels:   10,610,749
TP:                           4,895,002
FP:                           5,715,747
FN:                         110,191,456
TN:                       1,085,741,031
```

Macro 和 micro 回答不同问题。Per-image macro 给每张图相同权重；pooled
micro 让高分辨率和大目标贡献更多 pixels。这里 `macro IoU=0.207` 不能
写成“找出了 20.7% 的全部篡改像素”；真正的 pooled recall 只有 4.25%。

### 7.3 Domain slice

| Domain | Condition-macro AP | Precision | Recall | F1 | IoU | Predicted fraction |
|---|---:|---:|---:|---:|---:|---:|
| lodging | 0.431537 | 0.533051 | 0.270403 | 0.287488 | 0.232016 | 0.009111 |
| restaurant | 0.371318 | 0.419233 | 0.214166 | 0.218832 | 0.176751 | 0.006982 |

Lodging 在三类 local 的等权 macro 上更强，同时预测面积也更大。这里没有
预注册 direct domain-difference test，只作描述，不能把差异写成因果结论。

## 8. Real-image false-positive area

Real GT 全零，所以不发布 real pixel AP、precision 或 IoU；正式指标是
false-positive area：

| Slice | Images | FP pixels | Mean per-image FP fraction [95% CI] | Micro FP fraction |
|---|---:|---:|---:|---:|
| lodging | 147 | 2,580,962 | 0.009759 [0.005861, 0.014450] | 0.011564 |
| restaurant | 128 | 1,409,296 | 0.006076 [0.004917, 0.007351] | 0.006433 |
| **overall** | **275** | **3,990,258** | **0.008045 [0.005860, 0.010522]** | **0.009023** |

Official strict-ledger diagnostics：

```text
nonempty real masks:           275 / 275
median per-image FP fraction:  0.00325350
maximum per-image FP fraction: 0.243106
```

平均 FP area 低于 1%，但所有 real images 都至少有一个 positive pixel，
且最大值达到 24.31%。这再次证明“mask 是否非空”不是合法 T1 score。

## 9. Determinism、artifact audit 与 fresh replay

### 9.1 双 smoke

两个 frozen r2 smoke：

```text
imlvit_cat_protocol_balanced250_v1_smoke5x4_a_r2_20260727
imlvit_cat_protocol_balanced250_v1_smoke5x4_b_r2_20260727
```

每个包含 real/local Mouse/local Cat/local Trash-can 各 5 张：

```text
images compared:                       20
result projections compared exact:     20
raw logits compared exact:             20
model probability maps exact:          20
native probability maps exact:         20
official native masks exact:            20
artifact files compared byte-exact:    80
mismatches:                              0
```

Run ID、timestamp、latency 和 artifact path 是允许不同的 nondeterministic
fields；模型 computational projection 和 physical artifact bytes 必须
完全相同。

### 9.2 Formal artifact inventory

Formal gitignored artifacts：

| Artifact | Files | Bytes |
|---|---:|---:|
| raw logits model-1024 FP32 NPY | 1,025 | 4,299,292,800 |
| score maps model-1024 FP32 NPY | 1,025 | 4,299,292,800 |
| score maps native FP32 NPY | 1,025 | 6,595,316,160 |
| official strict native masks PNG | 1,025 | 3,389,686 |
| **Total** | **4,100** | **15,197,291,446** |

Independent audit 对每张图验证：

- canonical JPEG path、bytes、SHA-256 和 decoded dimensions；
- model/native artifact path containment、file hash、shape、dtype、
  contiguous、finite 和 probability range；
- raw-logit float32 sigmoid replay，最大 abs difference
  `1.1920928955078125e-7`，低于 `2e-7` tolerance；
- pure NumPy half-pixel native bilinear replay，最大 abs difference
  `1.1920928955078125e-7`，低于 `2e-5` tolerance；
- official PNG 精确等于 native probability strict `>0.5`；
- 1,025 次 independent preprocess 和 localization replay；
- exact-threshold pixels/images 为 2/2；
- persisted artifact inventory 与 hashes 在分析期间保持不变。

### 9.3 Fresh full-model replay

Fresh replay 先在 CPU 重做完整 preflight 和 strict load，确认记录的
source、checkpoint、environment 与 model structure 没变；随后在 formal
记录的 logical `cuda:0` 上重新创建模型并执行 1,025 次 forward：

```text
status / publishable:                 passed / true
images replayed:                      1025
preprocess compared exact:            1025
raw logits bit-exact:                 1025
model probability maps bit-exact:     1025
native probability maps bit-exact:    1025
official strict masks exact:          1025
strict localization metrics exact:    1025
exact-0.5 pixels / images:             2 / 2
recorded-device tolerance:             0.0
shared reducer used inside fresh:      false
```

Fresh replay 没有用 shared `>=` 替代 official mask；shared metrics 只在
独立 reducer 阶段基于相同 bit-exact native maps 计算。

## 10. Runtime、tests 与格式状态

冻结 runtime：

| 项 | 值 |
|---|---|
| grouped physical GPU | 7 |
| process-visible device | `cuda:0` |
| GPU | NVIDIA L20Z，compute capability 9.0 |
| Python | 3.12.3 |
| PyTorch / torchvision | `2.8.0.dev20250627+cu128` / `0.23.0.dev20250627+cu128` |
| CUDA runtime | 12.8 |
| NumPy / Pillow / Albumentations | 1.26.4 / 11.1.0 / 1.3.0 |
| precision / batch / autocast | FP32 / 1 / false |
| deterministic algorithms | true |
| cuDNN deterministic / benchmark | true / false |
| matmul TF32 / cuDNN TF32 | false / false |
| seed | 42 |

Formal wall time 为 **1,134.938 s**。逐图 model-forward latency
median/mean 为 **65.096 / 75.936 ms**，最大 formal peak allocated CUDA
memory 为 **2,921,058,304 bytes**。Decode、preprocess、artifact 写盘、
hash、GT 和 bootstrap 不包含在 forward latency 内。

Formal 完成到 final audit `generated_at` 相隔约 **2,446.033 s**；这段包括
完整 artifact audit、shared T2 reducer、1,000-bootstrap、fresh 1,025
forward、最终全量 hash 和 provenance gate，不应解释为纯 model latency。
本机当时还有 Hunyuan keepalive workload 共存，因此这些 wall/latency
数字是本次 operational record，不是隔离条件下的 throughput benchmark。

CPU-only tests：

```text
targeted Balanced runner + analyzer: 45 / 45 passed
extended related suite:             119 / 119 passed
```

Extended suite 包含 legacy runner/analyzer、IML metrics、canonical release、
run contract、shared localization reducer 和两份 Balanced tests。六条 warning
来自 dependency deprecation/SyntaxWarning，不是 test failure。

`black --check` 对 runner、analyzer 和两份 tests 为 clean；四文件逐个
`git diff --no-index --check` 没有 whitespace error。Report agent 没有
执行 CUDA、修改 runner/analyzer/tests、commit 或 push。

## 11. 可复现命令

必须使用新的空 pycache path；旧 v2 path 和旧 r1 evidence 不复用：

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=0
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX=/root/.cache/claimforge/pycache/imlvit-balanced-v3-empty
export PYTHONPATH=/root/claimforge-benchmark
```

Smoke A/B：

```bash
/root/.cache/claimforge/venvs/imlvit-07dd2be/bin/python \
  -m eval.opensource.run_imlvit_balanced \
  --mode smoke \
  --repo-root /root/claimforge-benchmark \
  --run-id imlvit_cat_protocol_balanced250_v1_smoke5x4_a_r2_20260727 \
  --per-condition-limit 5 \
  --device cuda:0

/root/.cache/claimforge/venvs/imlvit-07dd2be/bin/python \
  -m eval.opensource.run_imlvit_balanced \
  --mode smoke \
  --repo-root /root/claimforge-benchmark \
  --run-id imlvit_cat_protocol_balanced250_v1_smoke5x4_b_r2_20260727 \
  --per-condition-limit 5 \
  --device cuda:0
```

Smoke comparison：

```bash
/root/.cache/claimforge/venvs/imlvit-07dd2be/bin/python \
  -m eval.opensource.analyze_imlvit_balanced \
  --repo-root /root/claimforge-benchmark \
  --compare-smoke \
  --smoke-run-id-a \
    imlvit_cat_protocol_balanced250_v1_smoke5x4_a_r2_20260727 \
  --smoke-run-id-b \
    imlvit_cat_protocol_balanced250_v1_smoke5x4_b_r2_20260727
```

Formal：

```bash
/root/.cache/claimforge/venvs/imlvit-07dd2be/bin/python \
  -m eval.opensource.run_imlvit_balanced \
  --mode formal \
  --repo-root /root/claimforge-benchmark \
  --run-id imlvit_cat_protocol_balanced250_v1_full1025_r2_20260727 \
  --device cuda:0
```

Independent analysis + fresh replay：

```bash
/root/.cache/claimforge/venvs/imlvit-07dd2be/bin/python \
  -m eval.opensource.analyze_imlvit_balanced \
  --repo-root /root/claimforge-benchmark \
  --run-id imlvit_cat_protocol_balanced250_v1_full1025_r2_20260727 \
  --device cuda:0
```

Grouped supervisor 将 physical GPU 7 映射为进程内 `cuda:0`。复现时应通过
`CUDA_VISIBLE_DEVICES` 选择物理卡，不要把 physical ordinal 直接写入
冻结的 logical `--device`。

## 12. 最终判定

IML-ViT 应作为一个有效但明显 condition-dependent 的 **T2-only
transformer localization baseline**：

- Cat local insertion 上有很强 signal；
- Mouse 上能检测一部分 tiny edits，但 fixed-threshold miss 很多；
- Trash-can 的 ranking 尚可，完整区域 coverage 很弱；
- real mean FP area 低于 1%，但所有 real masks 都非空且存在长尾；
- 它没有合法 T1，不能检测或评估三类 full-frame AIGC；
- code 是 MIT，但 checkpoint 商用许可没有单独建立。

与同一 Balanced250 panel 上已经完成的
[CAT-Net v2](CATNET_BALANCED250_FULL_RESULTS_2026-07-27.md) 相比，
IML-ViT 的 condition-macro AP/F1 为 `0.404/0.257`，低于 CAT-Net 的
`0.680/0.510`；但它不依赖 CAT-Net 的显式 JPEG-DCT stream，仍提供了
有价值的 transformer-based architecture diversity。

论文表格应把 IML-ViT 放在 localization/T2 区域，T1 与 full-frame 明确
标为 `N/A`。任何 map-to-image aggregation、threshold calibration、
BenCo-retrained checkpoint 或其他权重都应视为新的方法 variant，并使用
新的 run IDs 重做 smoke、formal、artifact audit 与 fresh replay。
