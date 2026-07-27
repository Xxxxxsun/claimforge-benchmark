# CAT-Net v2（IJCV 2022 / CAT_full_v2）在 Balanced250 上的正式结果

日期：2026-07-27（UTC）

> **文档状态：正式完成，双 smoke、独立 artifact audit 与
> 1,025-image fresh full-model replay 均通过**
>
> CAT-Net v2 是原生像素定位方法，没有独立图像级 integrity head。
> 本报告因此严格只发布 T2；没有把 probability map 的最大值、均值、
> 面积或“是否非空”改造成 T1。CAT smoke 的合法规模是
> **20 张（real/local Mouse/local Cat/local Trash-can 各 5）**，
> 不是包含 full-frame 条件的 35 张。

正式 run：
`catnet_v2_ijcv2022_balanced250_v1_full1025_20260727`

Formal immutable run-config fingerprint：
`3651ea38ba02ceb08cc1d71646f02bceb034497a446a354eaf727f0986375702`

核心机器证据：

- [run manifest](../results/opensource/catnet/catnet_v2_ijcv2022_balanced250_v1_full1025_20260727/manifest.json)：
  `54b5dd7245166933a38aed77d747e4d4531d2ac20dd107f0056c15979ae13197`
- [expected inputs](../results/opensource/catnet/catnet_v2_ijcv2022_balanced250_v1_full1025_20260727/expected_inputs.jsonl)：
  `a952b6adcbd5af20ce635a5929c7045350b9f8e6729bae8d9cbf1978050f2cca`
- [逐图结果](../results/opensource/catnet/catnet_v2_ijcv2022_balanced250_v1_full1025_20260727/results.jsonl)：
  `089bdf8689cd8d7302c75013af36148b26d2503d10f7e5b8507ecc2a1a80bda6`
- [coverage summary](../results/opensource/catnet/catnet_v2_ijcv2022_balanced250_v1_full1025_20260727/summary.json)：
  `c0d88990e902d9c47f40057129eaf79c90a007923f1e963801f4a3071432cd01`
- [Balanced250 T2 metrics](../results/opensource/catnet/catnet_v2_ijcv2022_balanced250_v1_full1025_20260727/metrics.json)：
  `d11b944e16fcd1f3bb4d45e7811c21020c4507c0b4def226e080be4d151b061b`
- [independent artifact audit](../results/opensource/catnet/catnet_v2_ijcv2022_balanced250_v1_full1025_20260727/artifact_audit.json)：
  `e890458a16ba705a3f6b1e3c54881a5979defe47ac5d3321f5b825bf54b7b944`
- [fresh full-model replay](../results/opensource/catnet/catnet_v2_ijcv2022_balanced250_v1_full1025_20260727/fresh_replay.json)：
  `be90ce9f8007b292d88b20ab97ce17de69eefc6ab29b026eb4f8a23abf21e17e`
- [双 smoke comparison](../results/opensource/catnet/_reports/catnet_balanced_smoke_ab.json)：
  `594e97b2122ebe446b5e03555ad7b7d6e9a9d2d1e03710b2fcac498c96d1e6ba`

## 1. 结论摘要

CAT-Net v2 在 Balanced250 的三类局部植入上表现出很强的连续像素排序，
但固定阈值的表现随目标大小明显变化；它也不能回答整图 AIGC 检测问题。

- formal coverage 为 **1,025/1,025**，零 error、missing、duplicate
  和 superseded attempt；
- 三类 local 等权的 per-image pixel AP 为
  **0.680097 [0.659289, 0.700396]**；
- 三类 local 等权的 per-image F1/IoU@0.5 为
  **0.509752 / 0.418596**；
- 750 张 local 的直接 pooled-pixel precision/recall/F1/IoU 为
  **0.710162 / 0.078740 / 0.141761 / 0.076288**；
- Local Mouse/Cat/Trash-can 的 per-image AP 分别为
  **0.609194 / 0.827721 / 0.603376**；
- 750 张 local 中 737 张的固定阈值 mask 非空，731 张与 GT 至少有
  一个 pixel 重叠；“任意重叠”不是高质量定位率，正式质量仍由 AP、
  F1、IoU 与 confusion counts 表示；
- 275 张 real 的 mean per-image false-positive fraction 为
  **0.004359**（约 0.436%），micro fraction 为 **0.004078**；
- real 中 201/275 张存在至少一个 positive pixel，说明不能把
  “mask 非空”升级成可靠的图像级 forged verdict；
- 两次独立 20-image smoke 的 computational projection 与三类
  artifacts 全部 bit-exact，且与 formal 的两个 overlap slice
  分别 exact；
- fresh model replay 完成 **1,025/1,025**，raw logits、native
  probability maps、binary masks 和逐图 localization metrics 全部
  exact。

结果支持以下更精确的判断：

1. CAT-Net 对三类局部生成式植入都有真实的 pixel-ranking signal；
   尤其 Cat 的 per-image AP/IoU 达到 0.828/0.610。
2. 较大的目标没有自动带来更高 fixed-threshold coverage。Trash-can
   占用的 GT pixels 很多，但 micro recall 只有 5.66%；模型常只标出
   其中较小的一部分。
3. Mouse 的 GT 极小，固定阈值容易被少量场景 false positives 淹没：
   micro recall 很高（64.61%），micro precision 只有 19.53%。
4. Macro 与 micro 回答不同问题。Per-image macro 对每张图等权；
   pooled-pixel micro 会让大目标和高分辨率图贡献更多像素，不能把
   `macro IoU=0.419` 写成“找出了 41.9% 的全部篡改像素”。
5. 本 run 没有执行三类 full-frame inputs，也没有 T1 score。它不能被
   用来声称 CAT-Net 能或不能检测整图 AIGC。

## 2. 官方来源、checkpoint 与许可证边界

方法对应
[Learning JPEG Compression Artifacts for Image Manipulation Detection and Localization](https://link.springer.com/article/10.1007/s11263-022-01617-5)，
正式执行作者
[CAT-Net repository](https://github.com/mjkwon2021/CAT-Net)
commit `b50d391ffc423d3631fd7947714468788c791805`，tree
`ab0302cf11760ba1d07e0b419f20e9add0681112`。

正式权重是作者 Google Drive 发布的 `CAT_full_v2.pth.tar`：

| 项 | 冻结值 |
|---|---|
| provider | official author Google Drive |
| Drive file ID | `1tyOKVdx6UMys2OcNpUj9r6scxNIpcoLE` |
| bytes | 915,503,873 |
| SHA-256 | `f82aaafdd1142775231feedcea0bb7027f7370561d9e8d107465454001865989` |
| recorded epoch | 196 |
| state entries | 2,926 |
| state elements | 114,403,825 |
| model parameters / buffers | 114,263,810 / 140,015 |
| modules | 1,698 |
| load | CPU `weights_only=True` + minimal NumPy safe globals |
| strict load | true；missing/unexpected 为 0/0 |

CPU preflight 在任何 CUDA 初始化和 Balanced250 forward 之前完成
checkpoint schema、source tree、关键源码、环境和 strict model load
验证。正式 inference 与 fresh replay 都重新 strict-load 完整 checkpoint；
没有依赖单独的 RGB/DCT initialization weights 填补缺失参数。

许可证必须 fail-closed 地拆开：

- CAT-Net repository 没有发现 project-wide license；
- 仓库中的 `LICENSE of HRNet`（SHA-256
  `f1f33c3bec144f048d1cbff4dcae8d47a28faf263930ce779c61a7f4913bf055`）
  只覆盖 inherited HRNet component，不能表示 CAT-Net 全项目许可；
- official checkpoint 没有发现单独 license/terms；
- 因此代码和权重的 commercial-use / redistribution clearance 均未建立。

本报告将它称为 **source-available academic release**，而不是
project-wide OSI-licensed package。公开下载和完成 benchmark 不建立商业
产品权利；训练数据、第三方依赖和具体法域仍需单独审查，这也不是法律意见。

## 3. 方法原理与为什么适合局部植入

CAT-Net 的核心不是普通 RGB segmentation，而是显式融合两个取证视角：

1. RGB HRNet stream 保持多尺度高分辨率特征，寻找边界、纹理和语义空间
   中的局部不一致；
2. JPEG/DCT stream 直接读取同一 JPEG 的 luminance quantized DCT
   coefficients，并把绝对系数量化为 21-channel volume；
3. 原始 luminance quantization table 作为额外输入，使网络能区分不同
   JPEG quality 下的系数分布；
4. 多分辨率 fusion 把局部 compression trace 与 RGB evidence 合并，
   输出两类 dense logits。

这种 inductive bias 与 Balanced250 local threat model 很匹配：对象被插入
真实场景后，最终图像被统一 materialize 为 JPEG；局部区域仍可能保留与
周围不同的生成、融合或压缩痕迹。CAT-Net 不需要让一个占全图很小比例的
对象主导 global embedding，而是直接对每个区域进行 forensic ranking。

同时，结果也展示其限制：

- 统一 Q95 re-encoding 可能抹平早期压缩差异；
- 自然纹理、既有 JPEG 历史和高频结构会触发 DCT stream；
- 平滑生成式融合不一定留下经典 splice boundary；
- 一个目标内部的 evidence 并不均匀，所以大目标可能只有少量区域超过
  0.5，造成高 per-image ranking AP、低 pooled-pixel recall。

这些解释与架构和观测结果一致，但不是 causal ablation。不能据此断言
JPEG stream 是所有成功或失败的唯一原因，也不能外推到重新训练的 CAT
variant、原始未压缩图或不同 JPEG pipeline。

## 4. Frozen executable contract

正式入口：

- [Balanced250 runner](../eval/opensource/run_catnet_balanced.py)
- [independent analyzer](../eval/opensource/analyze_catnet_balanced.py)
- [legacy official-contract adapter](../eval/opensource/run_catnet.py)
- [legacy CAT metrics](../eval/opensource/catnet_metrics.py)
- [shared Balanced250 T2 reducer](../eval/opensource/balanced250_localization_metrics.py)

冻结文件 SHA-256：

| File | SHA-256 |
|---|---|
| Balanced250 runner | `26a28b89c079e72985801df6172f1896d9e27944b07140b557b5a3c20e2fe54d` |
| independent analyzer | `e9643b56f00de11d9ba9fb5f65d485be58580cacb258c82e07e2f1a8ebce9687` |
| runner tests | `b09df282502c4f6971ecd17c4b99557448070293160a7cf9a34dbdf03de6df1d` |
| analyzer tests | `51ce917ff82d589425c444b717554852ae4f4f5bf5d68c99d31fae21a5a79116` |
| legacy official adapter | `ca454e8ec92a86e9fab5d79f614d4777f2254a853498d6ad325afee8e0d9ad60` |
| legacy CAT metrics | `32d22f9999464e1889cbd7d1a5dce7b18a127bd4414fb26598154aae96d8d972` |
| shared Balanced250 T2 reducer | `83ac07257078fc41276742fa4b9f2eb936ac51c8ff93bf1253b8c45f2b704b2a` |
| canonical release loader | `6f4261dd2bc5335722aae253d851c363f663aa89cf42913647931d9cc60ac892` |
| run-contract helper | `95d9e648840ce648fd75d441d440e33ca2b9ede509aa548f040c4ac1bc4356b6` |

### 4.1 JPEG/DCT preprocess

| 组件 | 冻结行为 |
|---|---|
| input | canonical JPEG 原始 bytes |
| RGB decode | Pillow RGB uint8 |
| RGB normalization | `(uint8 - 127.5) / 127.5` |
| resize / crop / re-encode | none / none / false |
| JPEG parser | `jpegio` |
| DCT component | luminance Y quantized coefficients |
| DCT volume | 21 bins：0、abs 1–19、abs ≥20 |
| qtable | 原始 luminance 8×8 quantization table |
| geometry | right/bottom pad 到 ceil-8；RGB pad 127.5、DCT pad 0 |
| model input | 3 RGB + 21 DCT channels，batch 1，FP32 |

这条路径没有使用会 resize RGB 或生成临时 JPEG 的 generic wrapper。对
CAT-Net 而言，重编码会改变其要检测的 DCT coefficients 和 qtable，因此
必须直接读取 canonical JPEG。

### 4.2 Forward 与 native restore

原始输出是 padded geometry 上的 two-channel quarter-resolution logits。
Frozen postprocess 是：

```text
quarter logits
  -> bilinear resize 到 padded native geometry, align_corners=False
  -> float32 softmax over two channels
  -> channel 1 ("tampered")
  -> crop right/bottom padding
```

顺序不能交换：`interpolate(logits) -> softmax` 不等价于
`softmax(logits) -> interpolate`。

正式 T2：

```text
continuous score map = native channel-1 float32 probability
binary mask          = score_map >= 0.5
local GT             = exact decoded source/forged RGB difference
real GT              = all-zero, only false-positive area is meaningful
```

没有 TTA、ensemble、autocast、threshold fitting 或 Balanced250
fine-tuning。

### 4.3 T1 与 full-frame 边界

冻结 capability 是：

```text
primary task:                 T2 native localization only
independent image head:       false
valid_for_t1:                 false
map statistic promoted T1:    false
full-frame T1/T2:             N/A / N/A
```

Full-frame Mouse/Cat/Trash-can 共 750 张没有被 runner 选择，没有 forward、
artifact 或 score-map-loader call。`metrics.json` 明确记录三个 full-frame
condition 的 selected count 都为 0，loader calls 为 0。

## 5. Balanced250 数据设计与 coverage

Canonical release 仍包含 1,775 个唯一 score-cache inputs；CAT-Net 只选
合法 T2 subset：

| Condition | Rows | T2 target | T1 |
|---|---:|---|---|
| real | 275 | all-zero FP-area | N/A |
| local Mouse | 250 | exact-difference | N/A |
| local Cat | 250 | exact-difference | N/A |
| local Trash-can | 250 | exact-difference | N/A |
| full-frame Mouse | 0 selected | N/A | N/A |
| full-frame Cat | 0 selected | N/A | N/A |
| full-frame Trash-can | 0 selected | N/A | N/A |
| **Total** | **1,025** | **1,025 applicable** | **0** |

Formal selected-ID SHA-256：
`612e08565e38cb219fe5ea94dc8193580e099455e11fa778822488dbe7071717`。

Formal accounting：

```text
expected images:       1025
physical result rows:  1025
latest result rows:    1025
valid rows:            1025
error / missing:       0 / 0
superseded attempts:   0
```

Shared statistics使用 1,000 次 shared-source-content-cluster Poisson
bootstrap，root seed `20260726`。它按 source-content cluster 处理同源
依赖，不把重复注册的同一内容当成完全独立观察。

## 6. Native T2 结果

方括号为 1,000 次 cluster bootstrap 的 95% percentile interval。
`Target fraction` 是 condition 内 pooled pixel prevalence；AP/F1/IoU
列是 per-image macro，不能把 target fraction 当成其完全相同权重的
random baseline。

### 6.1 条件级 per-image macro

| Condition | Target fraction | Pixel AP [95% CI] | Precision@0.5 | Recall@0.5 | F1@0.5 | IoU@0.5 |
|---|---:|---:|---:|---:|---:|---:|
| local Mouse | 0.001350 | 0.609194 [0.577300, 0.640366] | 0.545172 | 0.580350 | 0.467775 | 0.351662 |
| local Cat | 0.063678 | **0.827721** [0.800419, 0.853968] | 0.826939 | 0.726323 | **0.698955** | **0.609690** |
| local Trash-can | 0.219667 | 0.603376 [0.571326, 0.632179] | **0.861351** | 0.334914 | 0.362526 | 0.294434 |
| **condition macro** | — | **0.680097** [0.659289, 0.700396] | **0.744487** | **0.547196** | **0.509752** | **0.418596** |

Precision 的 condition rows 按有定义的 nonempty masks 计算：Mouse
240/250、Cat 250/250、Trash-can 247/250。F1/IoU 则对全部 250 张计入
空 mask 的 0。

直接读取逐图 ledger 的补充诊断为：

| Condition | Nonempty masks | Any GT overlap | Median AP | Median F1 | Median IoU |
|---|---:|---:|---:|---:|---:|
| local Mouse | 240/250 | 235/250 | 0.657148 | 0.519046 | 0.350490 |
| local Cat | 250/250 | 250/250 | 0.921127 | 0.850511 | 0.739905 |
| local Trash-can | 247/250 | 246/250 | 0.533430 | 0.136855 | 0.073457 |
| **All local** | **737/750** | **731/750** | **0.748000** | **0.627182** | **0.456858** |

这些 median/overlap 是 ledger diagnostics，不替代 shared-bootstrap
primary table。特别是一个 pixel 的 overlap 也会记为 hit，不能把
731/750 写成 97.5% 高质量定位准确率。

### 6.2 Pooled-pixel micro

| Condition | Predicted fraction | Micro precision | Micro recall | Micro F1 | Micro IoU |
|---|---:|---:|---:|---:|---:|
| local Mouse | 0.004466 | 0.195299 | **0.646138** | 0.299939 | 0.176429 |
| local Cat | 0.011915 | 0.768315 | 0.143756 | 0.242196 | 0.137783 |
| local Trash-can | 0.015289 | **0.813921** | 0.056648 | 0.105923 | 0.055924 |
| **All local pixels** | **0.010576** | **0.710162** | **0.078740** | **0.141761** | **0.076288** |

750 张 local 共包含 1,206,543,236 pixels：

```text
target positive pixels:   115,086,458
predicted positive pixels:  12,760,286
TP:                          9,061,870
FP:                          3,698,416
FN:                        106,024,588
TN:                      1,087,758,362
```

这解释了 headline 的 macro/micro gap。Cat 和 Trash-can 的大面积 GT
没有被完整覆盖；许多图在局部区域内排序很好，但 `>=0.5` 只保留较小的
高置信区域。Mouse 的绝对目标很小，预测面积反而约为 GT pooled area 的
3.31 倍，因此 recall 高、precision 低。

### 6.3 Domain slice

| Domain | Condition-macro AP | F1@0.5 | IoU@0.5 | Predicted fraction |
|---|---:|---:|---:|---:|
| lodging | 0.667830 | 0.496211 | 0.405062 | 0.012544 |
| restaurant | 0.695006 | 0.525915 | 0.434813 | 0.009509 |

两个 domain 都保留较强 continuous ranking；restaurant 的 fixed-threshold
质量略高。这里没有预注册 direct domain-difference test，只作描述。

## 7. Real-image false-positive area

Real GT 全零，所以 pixel AP、precision、IoU 都不适合作为真实图 headline；
正式报告 false-positive area：

| Slice | Images | FP pixels | Mean per-image FP fraction [95% CI] | Micro FP fraction |
|---|---:|---:|---:|---:|
| lodging | 147 | 1,401,750 | 0.006436 [0.003924, 0.009236] | 0.006280 |
| restaurant | 128 | 401,583 | 0.001973 [0.001276, 0.002832] | 0.001833 |
| **overall** | **275** | **1,803,333** | **0.004359 [0.003038, 0.005999]** | **0.004078** |

逐图 diagnostic：

```text
nonempty real masks:             201 / 275
median per-image FP fraction:    0.000447737
maximum per-image FP fraction:   0.109450
```

因此 CAT-Net 的真实图平均 FP area 较小，但并非“真实图全部空 mask”。
它没有独立 T1 head 来 gate 这些区域，任何把 nonempty mask 当成 forged
decision 的做法都会在本 panel 上产生 201/275 的 image-level false
positives；这正是本 adapter 禁止 map-to-T1 promotion 的原因。

## 8. Determinism、artifact 与 fresh replay

### 8.1 双 smoke

两个 frozen smoke：

```text
catnet_v2_ijcv2022_balanced250_v1_smoke5x4_a_20260727
catnet_v2_ijcv2022_balanced250_v1_smoke5x4_b_20260727
```

每个都包含 real/local Mouse/local Cat/local Trash-can 各 5 张，共 20 张：

```text
images compared:                         20
computational fields per image:          20
mismatches:                               0
physical artifact audits passed:          2
artifact files audited:                 120
artifact files compared byte-exact:      60
formal overlap per smoke:                20
formal overlap exact:                  true
T1 fields compared:                       0
```

A/B 的 run IDs 和时间戳不同，所以整个 JSONL 文件 hash 不要求相同；
比较的是冻结 computational projection 与三个 physical artifacts。

### 8.2 Formal artifact audit

Formal raw artifacts 位于 gitignored
`outputs/opensource/catnet/catnet_v2_ijcv2022_balanced250_v1_full1025_20260727/`：

| Artifact | Files | Bytes |
|---|---:|---:|
| quarter-resolution two-channel logits FP32 NPY | 1,025 | 826,033,728 |
| native channel-1 score maps FP32 NPY | 1,025 | 6,595,316,160 |
| native threshold masks PNG | 1,025 | 2,643,672 |
| **Total** | **3,075** | **7,423,993,560** |

Audit 检查 4,850 个文件引用：1,025 canonical JPEG、3,075 model
artifacts 和 750 exact-difference GT masks。它还：

- 独立用 `jpegio` 重读每张 JPEG，核对 4:4:4 sampling、Y-DCT 和 qtable；
- 核对 raw logits 的 ceil-8-derived shape、FP32 和 finite；
- 核对 native maps 的 shape、FP32、finite 与 `[0,1]`；
- 用 pure NumPy half-pixel bilinear + float32 softmax 独立重建 1,025 张
  native maps；
- 得到最大 abs difference
  `1.7881393432617188e-7`，低于冻结的
  `2.384185791015625e-7`（2×float32 epsilon）；
- 逐张证明 PNG 等于 `score_map >= 0.5`；
- 重新加载 exact-difference GT 并重算逐图 T2 metrics；
- 证明 artifact inventory all-and-only，无 full-frame 或 T1 输出。

### 8.3 Fresh full-model replay

Fresh replay 先在 CPU 新建并 strict-load 一次完整模型，确认 CUDA 仍未
初始化；随后在 formal 记录的 logical `cuda:0` 上再次新建模型并执行
1,025 次 forward：

```text
status:                                  pass
images replayed:                         1025
fresh model instance:                    true
raw logits bit-exact:                    true
native score maps bit-exact:             true
native masks bit-exact:                  true
JPEG qtable / DCT evidence exact:        true
localization metrics exact:              true
T1 outputs compared:                        0
map statistic promoted to T1:            false
maximum replay peak CUDA bytes:    5,940,603,904
```

Fresh replay 使用与 formal 相同的 deterministic flags 和 GPU identity；
它不是读取已有 score map 后重算指标，而是重新 preprocess、load model、
forward、postprocess，再与保存的 raw logits/maps/masks 逐项比较。

## 9. Runtime 与执行重试说明

冻结 runtime：

| 项 | 值 |
|---|---|
| grouped physical GPU | 6 |
| process-visible device | `cuda:0` |
| GPU | NVIDIA L20Z，compute capability 9.0 |
| Python | 3.12.3 |
| PyTorch / torchvision | `2.8.0.dev20250627+cu128` / `0.23.0.dev20250627+cu128` |
| CUDA runtime | 12.8 |
| NumPy / Pillow / jpegio | 1.26.4 / 11.1.0 / 0.2.8 |
| precision / batch / autocast | FP32 / 1 / false |
| deterministic algorithms | true |
| cuDNN deterministic / benchmark | true / false |
| matmul TF32 / cuDNN TF32 | false / false |
| seed | 42 |

Formal grouped job wall time 为 1,134.377 s。逐图记录的 model-forward
latency median/mean 为 140.850/170.473 ms；最大值包含初次运行开销。
最大 formal peak allocated CUDA memory 与 fresh replay 都为
5,940,603,904 bytes。JPEG decode、DCT volume construction、artifact
写盘、hash、GT 与 bootstrap 不包含在 forward latency 内。

第一次 analyzer invocation 在 38.235 s 时被 bytecode-isolation
environment gate fail-closed：

```text
ValueError: analyzer bytecode-isolation environment changed
```

它在 formal artifact loop、metrics 和 fresh replay 之前中止，没有产生
可发布完成证据。第二次 invocation 在冻结
`PYTHONDONTWRITEBYTECODE=1`、`PYTHONPYCACHEPREFIX`、`PYTHONHASHSEED`
与 `CUBLAS_WORKSPACE_CONFIG` 下执行成功，return code 0，wall time
1,557.178 s。两次之间没有改 model、checkpoint、dataset、threshold、
runner/analyzer source 或 formal run；正式报告只引用 rc0 输出。

## 10. Tests 与格式状态

在 frozen CAT-Net venv、CPU-only（`CUDA_VISIBLE_DEVICES=''`）下执行：

```text
tests/test_run_catnet_balanced.py
tests/test_analyze_catnet_balanced.py
```

结果为 **41/41 passed**。覆盖 selection、T2-only identity、license/source/
checkpoint binding、CPU strict preflight、append-only recovery、path
containment、artifact inventory、independent JPEG/DCT replay、NumPy restore、
smoke exact comparison、shared T2 reducer delegation 与 no-T1 guards。

`black --check` 对 runner、analyzer 和两份测试返回
`would reformat`。本次没有自动格式化，因为这会改变 formal manifest 与
fresh audit 已绑定的 source SHA-256，使现有科学证据失配。Black 因此不是
此次发布门；这是明确记录的 style debt，不被写成“格式检查通过”，也不
影响 41/41 functional tests 和机器证据的有效性。若以后格式化，必须使用
新的 run IDs 重建相应 evidence。

## 11. 可复现命令

关键环境：

```bash
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX=/root/.cache/claimforge/pycache/catnet-balanced-v2-empty
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
```

Smoke A/B：

```bash
/root/.cache/claimforge/venvs/catnet-b50d391/bin/python \
  -m eval.opensource.run_catnet_balanced \
  --mode smoke --per-condition-limit 5 \
  --run-id catnet_v2_ijcv2022_balanced250_v1_smoke5x4_a_20260727 \
  --device cuda:0

/root/.cache/claimforge/venvs/catnet-b50d391/bin/python \
  -m eval.opensource.run_catnet_balanced \
  --mode smoke --per-condition-limit 5 \
  --run-id catnet_v2_ijcv2022_balanced250_v1_smoke5x4_b_20260727 \
  --device cuda:0
```

Formal：

```bash
/root/.cache/claimforge/venvs/catnet-b50d391/bin/python \
  -m eval.opensource.run_catnet_balanced \
  --mode formal \
  --run-id catnet_v2_ijcv2022_balanced250_v1_full1025_20260727 \
  --device cuda:0
```

Independent analysis：

```bash
/root/.cache/claimforge/venvs/catnet-b50d391/bin/python \
  -m eval.opensource.analyze_catnet_balanced \
  --phase all \
  --repo-root /root/claimforge-benchmark \
  --bootstrap-iterations 1000 \
  --bootstrap-seed 20260726
```

在 grouped supervisor 下，`CUDA_VISIBLE_DEVICES` 将 physical GPU 6 映射为
进程内 `cuda:0`；复现时不得把 physical ordinal 直接写入冻结命令。

## 12. 最终判定

CAT-Net v2 应作为 Balanced250 的核心 **JPEG-aware T2 baseline**：

- 三类 local 的 continuous localization 都明显有用；
- Cat 表现尤其强；
- Mouse 受 tiny-target precision 限制；
- Trash-can 受 large-target coverage 限制；
- real false-positive area 总体较低但并非没有长尾；
- 它没有合法 T1，不能与 whole-image AIGC detector 混成同一能力结论。

因此，论文表格应把 CAT-Net 放在 localization/T2 区域，T1 与 full-frame
列明确记为 `N/A`。任何后续 map-to-image aggregation、threshold
calibration、BenCo-retrained CAT 或其他 checkpoint 都应视为新的方法
variant，并使用新 run IDs 重新执行完整 smoke、formal、artifact audit
与 fresh replay。
