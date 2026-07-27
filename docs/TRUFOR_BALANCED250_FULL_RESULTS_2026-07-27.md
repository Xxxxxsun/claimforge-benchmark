# TruFor（CVPR 2023 / Phase 3）在 Balanced250 上的正式结果

日期：2026-07-27（UTC）

> **文档状态：正式完成，独立全链路 fresh replay 审计通过**
>
> 本报告只使用冻结的 1,775-image formal run、两个独立 35-image CUDA
> smoke、共享 Balanced250 T1/T2 指标、逐文件 artifact audit 和 fresh
> full-model replay。旧 Mouse run、论文表格和作者展示分数没有被拿来填写
> 本报告结果。

正式 run：
`trufor_cvpr2023_balanced250_v1_full1775_20260727`

Formal immutable run-config fingerprint：
`eb24f168fbfdad870064c487cd71553095c742cc4bfc565a7d700dc2ddf18997`

核心机器证据：

- [run manifest](../results/opensource/trufor/trufor_cvpr2023_balanced250_v1_full1775_20260727/manifest.json)：
  `99d620a8ff39b3d33f2b713cc32c9b0eb22e37cfdac7c123a21c26188d951801`
- [expected inputs](../results/opensource/trufor/trufor_cvpr2023_balanced250_v1_full1775_20260727/expected_inputs.jsonl)：
  `6b5128909eeffdbd88e61af02ca1bd191cb5460f94a23b47c87ebd0172e6d12c`
- [逐图结果](../results/opensource/trufor/trufor_cvpr2023_balanced250_v1_full1775_20260727/results.jsonl)：
  `42b34721c087613d5a2e44f180957e66fba85443fe0f99608f9788ff5b35ef28`
- [coverage summary](../results/opensource/trufor/trufor_cvpr2023_balanced250_v1_full1775_20260727/summary.json)：
  `5c67402ace06c96a87fd3ff93b4ca7859d08cb130cbcebeb1fd1cb071880f461`
- [Balanced250 T1/T2 metrics](../results/opensource/trufor/trufor_cvpr2023_balanced250_v1_full1775_20260727/balanced250_metrics.json)：
  `5a37b7fcdc74100197ab5a3f0e6180bdd609f94e2ecf9e29659de558e6e479a4`
- [independent replay audit](../results/opensource/trufor/trufor_cvpr2023_balanced250_v1_full1775_20260727/independent_audit.json)：
  `e1c335fdb20831d34be1b0d3fdfe45d9787e911775e4705b4f176db95fb95873`
- [双 smoke comparison](../results/opensource/trufor/_reports/trufor_cvpr2023_balanced250_v1_smoke5x7_a_20260727__vs__trufor_cvpr2023_balanced250_v1_smoke5x7_b_20260727_comparison.json)：
  `e5eb0e7c677d728692263b9f6adb7b98e31877995e11e76a80839c49b7502f1e`

最终审计状态：`replay_audit_passed`。

## 1. 结论摘要

TruFor 是目前 Balanced250 上第二个完成正式扩展的 local-forensics
方法，也是第一个在三种 local insertion 上同时给出强 T1 与有用原生 T2
结果的方法。它的能力边界非常清楚：**局部植入强，Hunyuan 全图条件编辑
弱，真实图上的 dense false-positive area 不能忽略。**

- formal coverage 为 1,775/1,775，零 error、missing 和 superseded；
- local 三条件等权 macro AUROC 为 **0.903579**，AP 为
  **0.914374**，TPR@5%FPR 为 **68.40%**；
- 发布阈值 `sigmoid(detection_logit) >= 0.5` 的 local macro recall
  为 **72.40%**，accuracy 为 **82.60%**；
- full-frame 三条件 macro AUROC 只有 **0.623285**，AP 为
  **0.596862**，TPR@5%FPR 为 **7.20%**；
- full-frame 发布阈值 recall 只有 **11.47%**，accuracy 为
  **52.13%**；
- source-matched local pooled strict ranking 为 **96.00%**，mean
  score delta 为 `+0.534769`；
- source-matched full-frame pooled strict ranking 为 **64.27%**，
  mean delta 为 `+0.062534`；
- 750 张 local 的 pooled per-image pixel AP 为 **0.676670**，
  per-image IoU@0.5 为 **0.415657**；
- 同一批 local 像素的 pooled micro IoU 只有 **0.077767**，micro
  recall 为 **0.079313**，说明大面积 GT 中仍有大量漏检像素；
- 275 张 real 的 mean per-image false-positive fraction 为
  **0.017764**（约 1.776%），micro fraction 为 **0.016038**；
- 两个 smoke 的 T1、raw score maps、raw reliability maps 和适用
  masks 全部 byte-exact；
- fresh full-model replay 为 1,775/1,775，T1 score/logit、两类
  float32 native map 的最大差异全部为 `0.0`，1,025 个 mask 全部
  exact 重建。

当前证据支持以下结论：

1. TruFor 对这三类 AIGC 局部植入具有真实且跨对象类别可重复的信号；
   local Cat 最强，Mouse 相对最难，但 Mouse AUROC 仍为 `0.817504`。
2. 这种优势不自动转移到整图 AIGC 检测。三组 full-frame conditional
   edit 只有有限排序信号，low-FPR recall 和发布阈值 recall 都很弱。
3. 原生 forged-probability map 的 per-image 排序和重叠指标很有用，
   但 pooled micro recall 以及 pristine false-positive area 暴露了明确
   的部署风险；不能只引用 per-image AP。
4. TCP reliability map 是模型的置信度输出，不是第二张 anomaly map。
   CLAIMFORGE 不把它乘进 T2 score map，也不把它独立评分。
5. 三组 full-frame 数据是从真实源图出发的 Hunyuan 条件全图编辑，不是
   脱离真实源图独立采样的纯 T2I。本 run 不能代表所有整图生成器。

## 2. 官方来源、checkpoint 与许可证

TruFor 对应 CVPR 2023 论文
[TruFor: Leveraging All-Round Clues for Trustworthy Image Forgery Detection and Localization](https://openaccess.thecvf.com/content/CVPR2023/html/Guillaro_TruFor_Leveraging_All-Round_Clues_for_Trustworthy_Image_Forgery_Detection_and_CVPR_2023_paper.html)，
正式执行冻结作者的
[官方 GitHub repository](https://github.com/grip-unina/TruFor)。

| 资产 | 冻结身份 | 大小 / 结构 | SHA-256 |
|---|---|---:|---|
| official source | commit `ae54475df6f41a491d7615100feb19263dec13f7` | tracked + untracked clean；17 个关键 source/asset 文件绑定 | 见 manifest |
| `TruFor_weights.zip` | 作者官方下载；published MD5 `7bee48f3476c75616c3c5721ab256ff8` | 260,878,690 bytes；唯一 checkpoint member | `953f1f7eda0dd2c5ece322ae9c185ba1079c1265aa5fdf319ef5a20604d206d8` |
| `trufor.pth.tar` | official phase-3 epoch 81 checkpoint | 281,496,429 bytes；952 tensors；68,705,510 elements | `ac1d90e329a72e0d66e8665e123a19e94bfae3209c3ef8a4f9ca3b91578c7844` |
| checkpoint tensor schema | ordered key/shape/dtype/count | 927 FP32 + 25 int64 tensors | `8a9ebd68344360a8117337c0d2e65b7b8ef82a0040503b1a7ac3cdf7a784f9cd` |
| `trufor_ph3.yaml` | official inference/training config | 1,124 bytes | `a87108eb0df40d9bab6a303eb91419564b7c106d5105bbd5d8ecaec1567b5b8b` |
| overall `LICENSE.txt` | TruFor custom license | 1,837 bytes | `07201e07e3d2c1ac55480037a87734fcccacbb0cd0e25a31e3b89ac7ffadf8b4` |
| `LICENSE_CMX.txt` | CMX component MIT text | 1,086 bytes | `687ff4d0ea13200541df7359799a08c2db626094ecacbb8a3fe63ffad177f2a1` |

Checkpoint 通过 `torch.load(..., map_location="cpu", weights_only=True)`
加载。静态扫描发现的 NumPy scalar/dtype globals 使用最小 allowlist；
没有启用 unrestricted pickle。952/952 个 state entries 严格加载，
missing 和 unexpected keys 均为空。模型审计记录：

```text
parameters:            68,697,421
trainable parameters:   1,578,242
buffers:                    8,089
modules:                      872
constructor external weights used: false
```

许可证边界是明确的，而不是“不确定”：整体 `LICENSE.txt` 只允许
informational 和 nonprofit use，未经授权禁止 industrial / profit-oriented
use，并要求保留 notice 和 attribution。CMX 子组件的 MIT 文本不覆盖
TruFor 整体限制。Manifest 与独立 audit 因此冻结：

```text
commercial_use_cleared:                       false
commercial_use_requires_separate_authorization: true
cmx_mit_does_not_override_overall_restriction: true
```

所以本结果只证明研究评测可以严格执行和复核，**不代表可以把 TruFor
代码、权重或衍生服务直接用于商业产品**。

## 3. 方法原理与训练边界

TruFor 的核心不是单看 RGB 内容，而是把高层视觉内容和低层成像痕迹一起
建模：

1. 17-layer DnCNN 风格的 Noiseprint++ extractor 从输入中提取
   camera/internal/external processing fingerprint；
2. RGB 与 Noiseprint++ 两个模态进入 CMX / MiT-B2 cross-modal
   Transformer backbone；
3. MLP decoder 输出二通道 pixel localization logits；
4. 第二个 decoder 输出单通道 TCP confidence logit；
5. detection head 对 confidence 以及 confidence-weighted localization
   logit margin 分别做 min/max/mean/mean-square pooling，拼成 8 维
   feature，再经 `Linear(8,128) -> ReLU -> Dropout -> Linear(128,1)`
   产生整图 detection logit。

这个设计之所以适合局部取证，是因为它把伪造看成相对于同一图像内部正常
处理规律的局部异常，而不只依赖“这张图的语义像不像 AI”。RGB 分支提供
上下文，Noiseprint++ 分支寻找相机和处理管线不连续，cross-modal fusion
让两类线索互相约束；confidence head 则让 image-level pooling 更重视
模型认为可靠的位置。

要特别区分两个事实：

- 官方 T1 detection head **内部**使用 confidence-aware pooling；
- CLAIMFORGE 的 T2 指标直接使用 forged-probability map，不做任何
  reliability 后处理，也不把 reliability 当作独立 anomaly score。

Phase-3 config 的训练集合记录为 IMD、FR、CA、COCO 和 RAISE。该阶段冻结
Noiseprint++、CMX backbone 和 localization head，只训练 confidence 与
detection 部分。本评测使用作者发布的 epoch-81 phase-3 checkpoint
做 pinned official-release inference，没有在 Balanced250 上训练、
调阈值或拟合校准器。

官方 release 没有发布与该 checkpoint 绑定的 frozen numerical output
fixture。我们没有把任意自选图片伪装成作者 golden；CPU strict-load
structural golden、双 smoke exact reproduction 和 1,775-image fresh
replay 共同构成可执行门禁。

## 4. Frozen executable contract

正式实现入口：

- [runner](../eval/opensource/run_trufor_balanced.py)
- [independent analyzer](../eval/opensource/analyze_trufor_balanced.py)
- [shared T1 metrics](../eval/opensource/balanced250_metrics.py)
- [shared T2 metrics](../eval/opensource/balanced250_localization_metrics.py)

冻结文件 SHA-256：

| File | SHA-256 |
|---|---|
| runner | `d9520b89665de6cd789cdc9bda6201bf1ac2cb2fad5c56668cac7f62d04d8303` |
| analyzer | `66d11687e616e03cb270af33b6ca37b9559395e17c567560412dd3a9bcbd2e92` |
| runner tests | `d3b8d3467b7d765aac06dff7f3fd64cb8b19dbf229ce02fee0f1e6c017762e50` |
| analyzer tests | `5ccbb4d2425f99ea4a23341f2b096f3cf6a78858bdcf0bf6558b22d5b1f98f98` |
| legacy TruFor metrics | `81000c7eed7fd276b33fb5e7f5bc13d2a1ebc82d2ceb757222a82b835d4288c7` |
| shared Balanced250 T1 metrics | `f3932099bb63b766f063a66684e1d45f6e12601337d73859591d79297dbbed1c` |
| shared Balanced250 T2 metrics | `83ac07257078fc41276742fa4b9f2eb936ac51c8ff93bf1253b8c45f2b704b2a` |

Formal manifest 绑定 11 个本地 adapter/source 文件；最终 audit 再次把
11/11 当前 bytes 和 SHA-256 与 manifest 精确核对。Analyzer 自身 SHA
同时绑定在 smoke comparison 和 formal audit 中。

### 4.1 Preprocess

| 组件 | 冻结行为 |
|---|---|
| decode | `Pillow.Image.open(...).convert("RGB")` |
| EXIF / ICC | 不做额外 EXIF transpose；不做 ICC conversion |
| geometry | 保留 native decoded H×W；不 resize、不 crop |
| tensor | contiguous HWC uint8 → CHW FP32 |
| scale | 除以 **256.0**，不是 255 |
| internal RGB branch | official model 内部 ImageNet normalization |
| Noiseprint++ branch | 接收除以 256 的 RGB tensor |
| output restore | network 内部 bilinear upsample 到输入 H×W，`align_corners=False` |
| batch/autocast | batch 1；autocast disabled |

`/256`、`/255` 和常见 torchvision normalization 不是可互换的实现细节。
本报告所有结果只属于
`official_trufor_native_rgb_float32_divide_256`。

### 4.2 T1、T2 与 artifacts

T1：

- official detection head 输出单个 FP32 logit；
- `ai_score = sigmoid(detection_logit)`；
- 越高越 fake；
- released decision 为 `ai_score >= 0.5`；
- 该 score 未声明为 calibrated probability。

T2：

- raw forged map 为二通道 localization logits 的
  `softmax(...)[class 1]`；
- raw TCP reliability 为 `sigmoid(confidence_logit)`；
- 二值 mask 为 `forged_probability >= 0.5`；
- local GT 为完整 exact-difference mask；
- real GT 全零，只报告 false-positive area；
- full-frame raw maps 仅作诊断，不创建 mask、不计算 T2。

每张成功输入保存两个 native-resolution FP32 `.npy`。T2 适用的
1,025 张另存一个 `L` mode、只含 0/255 的 PNG。Result row 分别绑定
file SHA、array SHA、bytes、shape、dtype 和语义。

### 4.3 Runtime

| 项 | 值 |
|---|---|
| physical GPU identity | NVIDIA L20Z |
| manifest-recorded logical device | `cuda:0` |
| Python | CPython 3.12.3 |
| PyTorch / torchvision | `2.8.0.dev20250627+cu128` / `0.23.0.dev20250627+cu128` |
| timm / NumPy / Pillow | 0.5.4 / 1.26.4 / 11.1.0 |
| precision / batch | FP32 / 1 |
| autocast / inference mode | false / true |
| deterministic algorithms | disabled，沿用 official runtime |
| cuDNN | disabled，沿用 `trufor_ph3.yaml` |
| matmul TF32 / precision | true / `high` |
| seed | 42 |

物理卡序号由 grouped supervisor 选择；manifest 只冻结进程内 logical
`cuda:0`、设备 identity 与完整 runtime shape。这里证明同一冻结 runtime
下 exact reproduction，不把它外推成任意硬件上的 byte identity。

## 5. Frozen Balanced250 设计与任务边界

共享协议包含：

- 1,775 个唯一 score-cache inputs；
- 1,750-row primary panel：real 与六个 forged condition 各 250；
- 1,500 个显式 source-matched pairs；
- primary：每个 forged condition 对同一 independently selected
  `real250` panel；
- secondary：只使用 `source_pairs.jsonl` 的显式端点；
- 不从 `task_id` 推断配对；
- 1,000 次 shared-source-content-cluster Poisson bootstrap；
- bootstrap root seed `20260726`；
- TPR@5%FPR 使用 real score 的 0.95 `higher` quantile 和严格 `>`。

TruFor 的 capability contract 是：

```text
T1: real275 + local750 + fullframe750 = 1775
T2: real275 + local750                 = 1025
fullframe T2: N/A
```

Full-frame 虽然也保存 raw score/reliability maps，但这些图没有合法局部
GT。Analyzer 在 T2 阶段对 750 张 full-frame 的 map loader 调用次数严格
为 0；橙色 conditioning box 没有被当成 GT，也没有构造全图 mask。

六条件 mixed macro 只作为导航摘要。Local 与 full-frame 是两个不同任务，
必须分别报告。

## 6. Coverage 与 artifact inventory

| Condition | T1 expected / valid | T2 applicable | Error | Missing |
|---|---:|---:|---:|---:|
| real score cache | 275 / 275 | 275（all-zero FP only） | 0 | 0 |
| local mouse | 250 / 250 | 250 | 0 | 0 |
| local cat | 250 / 250 | 250 | 0 | 0 |
| local trash can | 250 / 250 | 250 | 0 | 0 |
| full-frame mouse | 250 / 250 | N/A | 0 | 0 |
| full-frame cat | 250 / 250 | N/A | 0 | 0 |
| full-frame trash can | 250 / 250 | N/A | 0 | 0 |
| **Total** | **1,775 / 1,775** | **1,025** | **0** | **0** |

Runner accounting：

```text
physical result rows: 1775
latest result rows:   1775
superseded attempts:     0
```

Local outputs 保存在 gitignored
`outputs/opensource/trufor/trufor_cvpr2023_balanced250_v1_full1775_20260727/`：

| Artifact | Files | Total bytes | Role |
|---|---:|---:|---|
| forged-probability FP32 NPY | 1,775 | 11,416,977,532 | T2 for real/local；diagnostic for full-frame |
| TCP reliability FP32 NPY | 1,775 | 11,416,977,532 | diagnostic / official T1 internal confidence |
| threshold PNG mask | 1,025 | 3,108,491 | real/local only |
| **Total** | **4,575** | **22,837,063,555** | gitignored local evidence |

Independent analyzer 对 4,575 个文件逐一重开，验证 canonical path、file
SHA、array SHA、native shape、dtype、finite range，并确认所有 1,025
PNG 都逐像素等于 `score_map >= 0.5`。

## 7. Primary T1 结果

方括号为 1,000 次 shared-cluster bootstrap 的 95% percentile interval。

### 7.1 Family macro

| Scope | AUROC [95% CI] | AP [95% CI] | TPR@5%FPR [95% CI] | Accuracy @ 0.5 [95% CI] | Recall @ 0.5 [95% CI] |
|---|---:|---:|---:|---:|---:|
| local macro | 0.903579 [0.886721, 0.919421] | 0.914374 [0.895996, 0.931245] | 0.684000 [0.627701, 0.745405] | 0.826000 [0.803505, 0.846327] | 0.724000 [0.688140, 0.759392] |
| full-frame macro | 0.623285 [0.582280, 0.666727] | 0.596862 [0.558858, 0.648401] | 0.072000 [0.039996, 0.133026] | 0.521333 [0.498261, 0.543889] | 0.114667 [0.083013, 0.148365] |
| all-six mixed macro | 0.763432 [0.738401, 0.789262] | 0.755618 [0.730288, 0.788367] | 0.378000 [0.338177, 0.435070] | 0.673667 [0.654075, 0.691460] | 0.419333 [0.392728, 0.443645] |

`all-six` 数值处在两个任务之间，不能用它掩盖 local/full-frame 分裂。

### 7.2 条件级 ranking 与 low-FPR

| Condition | AUROC [95% CI] | AP [95% CI] | TPR@5%FPR [95% CI] |
|---|---:|---:|---:|
| local mouse | 0.817504 [0.786727, 0.848397] | 0.837843 [0.805726, 0.868530] | 0.436000 [0.357672, 0.547539] |
| local cat | 0.967136 [0.951754, 0.979247] | 0.973827 [0.962694, 0.983040] | 0.888000 [0.847971, 0.931583] |
| local trash can | 0.926096 [0.905394, 0.946710] | 0.931453 [0.910615, 0.951657] | 0.728000 [0.636346, 0.799213] |
| full-frame mouse | 0.635424 [0.591293, 0.680348] | 0.613403 [0.566993, 0.667402] | 0.072000 [0.044026, 0.145232] |
| full-frame cat | 0.622528 [0.578036, 0.670547] | 0.600475 [0.556490, 0.655073] | 0.084000 [0.039052, 0.157463] |
| full-frame trash can | 0.611904 [0.567163, 0.655287] | 0.576707 [0.536633, 0.630571] | 0.060000 [0.023728, 0.113889] |

六个 condition 共用同一个 independent real250，point-estimate 5% FPR
threshold 都是 `0.5935231447219849`，实际 FPR 为 `0.048`。

### 7.3 Released threshold `score >= 0.5`

| Condition | Accuracy | Precision | Recall | F1 | TP / FP / FN / TN |
|---|---:|---:|---:|---:|---:|
| local mouse | 0.714 | 0.874126 | 0.500 | 0.636132 | 125 / 18 / 125 / 232 |
| local cat | 0.916 | 0.926230 | 0.904 | 0.914980 | 226 / 18 / 24 / 232 |
| local trash can | 0.848 | 0.914286 | 0.768 | 0.834783 | 192 / 18 / 58 / 232 |
| full-frame mouse | 0.524 | 0.625000 | 0.120 | 0.201342 | 30 / 18 / 220 / 232 |
| full-frame cat | 0.528 | 0.640000 | 0.128 | 0.213333 | 32 / 18 / 218 / 232 |
| full-frame trash can | 0.512 | 0.571429 | 0.096 | 0.164384 | 24 / 18 / 226 / 232 |

表中的六组 FP 都是同一个 real250 panel 的 18 个 false positives，不是
108 张不同 real。Local 三类合计命中 543/750；full-frame 只命中
86/750。三条件的 common specificity 为 `0.928`。

### 7.4 Domain

| Family | Domain | AUROC | AP | TPR@5%FPR | Accuracy @ 0.5 | Recall @ 0.5 |
|---|---|---:|---:|---:|---:|---:|
| local | lodging | 0.912738 | 0.926127 | 0.703382 | 0.830317 | 0.763595 |
| local | restaurant | 0.893262 | 0.900394 | 0.687065 | 0.820966 | 0.675160 |
| full-frame | lodging | 0.592660 | 0.590957 | 0.068942 | 0.505102 | 0.123102 |
| full-frame | restaurant | 0.658542 | 0.611825 | 0.110411 | 0.540007 | 0.104761 |

Local 信号在两个 domain 都强。Full-frame restaurant 的 AUROC 较高，
但固定阈值 recall 没有同步提高；这里没有做 domain-difference
simultaneous test，因此只作描述。

## 8. Source-matched secondary

Secondary 只使用 frozen `source_pairs.jsonl` 的显式端点，回答“同一源图
经过编辑后分数是否上移”。它不替代真实部署时的独立单图 primary。

| Scope | Pairs | Clusters | Mean delta [95% CI] | Median delta | Strict ranking [95% CI] | W / L / T |
|---|---:|---:|---:|---:|---:|---:|
| local mouse | 250 | 247 | +0.351723 [+0.315193, +0.390796] | +0.276066 | 0.912000 [0.872078, 0.945099] | 228 / 22 / 0 |
| local cat | 250 | 247 | +0.700897 [+0.668945, +0.733307] | +0.815450 | 0.988000 [0.973684, 1.000000] | 247 / 3 / 0 |
| local trash can | 250 | 246 | +0.551687 [+0.513642, +0.589013] | +0.629327 | 0.980000 [0.961035, 0.995902] | 245 / 5 / 0 |
| full-frame mouse | 250 | 247 | +0.071309 [+0.039360, +0.100498] | +0.072624 | 0.652000 [0.594485, 0.706112] | 163 / 87 / 0 |
| full-frame cat | 250 | 248 | +0.063937 [+0.032143, +0.095730] | +0.057040 | 0.648000 [0.586866, 0.710623] | 162 / 88 / 0 |
| full-frame trash can | 250 | 246 | +0.052356 [+0.024551, +0.079380] | +0.056293 | 0.628000 [0.567565, 0.686526] | 157 / 93 / 0 |
| **local pooled** | **750** | **270** | **+0.534769 [+0.509430, +0.561293]** | **+0.625060** | **0.960000 [0.943287, 0.973278]** | **720 / 30 / 0** |
| **full-frame pooled** | **750** | **269** | **+0.062534 [+0.033731, +0.088463]** | **+0.062563** | **0.642667 [0.590088, 0.692898]** | **482 / 268 / 0** |

Local primary 与 matched secondary 方向一致，说明不是只靠 panel
composition 得到的强结果。Full-frame matched ranking 比 0.5 高，但幅度
远弱于 local；它也不能被写成 64.27% 的单图 deployment accuracy。

## 9. Native T2 localization

T2 的连续分数是 forged-probability map；固定 operating point 是
`map >= 0.5`。以下 per-image 指标先逐图计算，再对图像平均。

### 9.1 Per-image metrics

| Scope | GT positive fraction | Pixel AP [95% CI] | Precision@0.5 | Recall@0.5 | F1@0.5 | IoU@0.5 | MCC@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| local mouse | 0.001350 | 0.582006 [0.545147, 0.613913] | 0.621025 | 0.533863 | 0.503870 | 0.377607 | 0.533961 |
| local cat | 0.063678 | 0.817299 [0.791571, 0.841107] | 0.904713 | 0.649871 | 0.687664 | 0.593433 | 0.706870 |
| local trash can | 0.219667 | 0.630706 [0.605099, 0.655641] | 0.897936 | 0.307299 | 0.353837 | 0.275932 | 0.395681 |
| **750-image pooled** | **0.095385** | **0.676670 [0.657483, 0.695415]** | **0.808021** | **0.497011** | **0.515124** | **0.415657** | **0.545720** |

Pooled precision/MCC 分别在 748/750 张有定义；两张空预测 mask 的
precision/MCC 为 undefined，而 recall/F1/IoU 和 continuous AP 仍按共享
协议处理。

### 9.2 Pooled-pixel micro metrics

| Scope | Predicted positive fraction | Micro precision | Micro recall | Micro F1 | Micro IoU | Micro MCC |
|---|---:|---:|---:|---:|---:|---:|
| local mouse | 0.003147 | 0.241259 | 0.562532 | 0.337690 | 0.203145 | 0.367161 |
| local cat | 0.009852 | 0.832692 | 0.128827 | 0.223133 | 0.125577 | 0.314147 |
| local trash can | 0.015313 | 0.891982 | 0.062178 | 0.116253 | 0.061714 | 0.202500 |
| **all local pixels** | **0.009461** | **0.799631** | **0.079313** | **0.144312** | **0.077767** | **0.234306** |

750 张 local 一共包含 1,206,543,236 个像素：

```text
TP   9,127,856
FP   2,287,229
FN 105,958,602
TN 1,089,169,549
```

Per-image 与 micro 的差距不是矛盾。Mouse GT 极小，Cat/Trash-can 中又有
少数非常大的 exact-diff 区域；per-image macro 给每张图相同权重，micro
则由大 mask 的像素数主导。Trash-can 的 per-image AP 为 0.631，但
micro recall 只有 6.22%，明确说明模型常只点亮大 GT 的一部分。

### 9.3 Real-image false-positive area

275 张 real 的 GT 全零，因此 pixel AP 为 `null`。固定阈值下：

- false-positive pixels：7,092,711 / 442,253,004；
- mean per-image FP fraction：
  `0.017764 [0.013766, 0.022687]`；
- micro FP fraction：
  `0.016038 [0.012523, 0.020168]`；
- 273/275 张 real 至少有一个 positive pixel；
- median per-image FP fraction 为 `0.007145`；
- maximum 为 `0.448532`。

因此 TruFor map 不能脱离 T1 score 和上下文被理解成“点亮即伪造”。特别是
maximum real FP area 接近 44.85%，这是部署与可视化时必须展示的 caveat。

### 9.4 与论文官方 metric 的边界

TruFor 官方 metric 会忽略伪造边界附近像素，并在 map 与 inverse map
之间取更高 F1。CLAIMFORGE 两者都不做：

- 固定 class-1 forged direction；
- 对完整 exact-difference GT 的每个像素评分；
- 不看 GT 后翻转 map；
- real 只报告 all-zero false-positive area；
- full-frame T2 严格 N/A。

所以这里的 T2 F1/IoU 是跨方法共享 CLAIMFORGE 指标，不能直接与论文表格
中的官方 F1 数字横向比较。

## 10. Determinism 与独立审计

### 10.1 CPU preflight 与 structural golden

CPU preflight 在 accelerator configuration 和任何 Balanced250 model
score 之前完成：

- `cuda_initialized_before=false`；
- `cuda_initialized_after=false`；
- accelerator model forwards：0；
- Balanced250 model scores：0；
- official source、archive、checkpoint、config、license 全部 exact；
- 952/952 strict state load；
- model parameters/modules/trainability 与冻结值一致。

独立 analyzer 再做一次 CPU strict-load structural golden，状态为
`independent_cpu_structural_golden_passed`。作者没有发布 numeric fixture，
audit 明确记录 `author_published_numerical_golden=null`，没有虚构 golden。

### 10.2 A/B smoke

最终 smoke：

- A：
  `trufor_cvpr2023_balanced250_v1_smoke5x7_a_20260727`
  （fingerprint
  `a233694cfcb074437e6cf27caaa98acc6b3166e9f770884b4fd71a19338ec618`）；
- B：
  `trufor_cvpr2023_balanced250_v1_smoke5x7_b_20260727`
  （fingerprint
  `daff5a5678c71d1130c1574de0fbaa88c4f3c9d6749ae21bea0df6d501eed332`）。

每次覆盖七个 condition 各 5 张，共 35 张，其中 20 张 T2 适用。
Comparison 状态为 `exact_reproduction_passed`：

- 35/35 computational result projection exact；
- T1 scores 和 detection logits exact；
- 35 对 raw score-map NPY file bytes exact；
- 35 对 raw reliability-map NPY file bytes exact；
- 20 对 threshold PNG file bytes exact；
- 15 张 full-frame 在两个 run 中都没有 mask；
- recorded runtime exact。

### 10.3 Formal artifact audit 与 fresh replay

Formal artifact audit：

- 1,775 raw forged-probability maps verified；
- 1,775 raw reliability maps verified；
- 1,025 applicable threshold PNGs verified；
- 750 full-frame masks absent；
- 全部 path、file hash、array hash、shape、dtype、range exact；
- reliability 未乘入 T2 map；
- full-frame localization metrics absent。

Fresh full-model replay：

```text
selected images reopened:          1775
selected images preprocessed:      1775
model forwards:                    1775
T1 scores/logits compared exact:   1775 / 1775
raw score maps compared exact:     1775 / 1775
raw reliability maps exact:        1775 / 1775
applicable masks rederived exact:  1025 / 1025
fullframe masks not created:        750 / 750
maximum T1 score abs diff:          0.0
maximum detection-logit abs diff:   0.0
maximum score-map abs diff:         0.0
maximum reliability-map abs diff:   0.0
```

Fresh metric equivalence 不是靠放宽 tolerance 得到：全部 1,775 个 T1
metric rows 和 1,025 张 T2 score maps 已 exact，因此同一 deterministic
reducer 的输入字节完全相同。Audit 记录
`fresh_model_metrics_exact=true`。

## 11. Runtime、测试与同协议对照

Formal manifest interval 为
`2026-07-27T07:01:46.003362Z` 至
`2026-07-27T07:24:31.349897Z`，约 1,365.347 秒（22 分 45 秒）。

| Field | Mean | Median | P95 (`higher`) | Max |
|---|---:|---:|---:|---:|
| model-forward latency | 180.426 ms | 229.483 ms | 277.147 ms | 721.644 ms |
| peak CUDA allocation | — | — | — | 7,413,214,208 bytes（6.904 GiB） |

Latency 随 native image size 明显变化，而且不包含完整 artifact hashing、
bootstrap 和 fresh audit；它是运行元数据，不是干净的吞吐 benchmark。

最终定向 CPU-only 回归：

```text
46 passed in 8.10s
```

覆盖 selection、resume、append-only error recovery、CPU preflight ordering、
safe roots、strict JSON、source/license/checkpoint binding、native preprocess、
三态 T2 semantics、raw-map/PNG fail-closed validation、full-frame T2 exclusion、
exact smoke comparison 和 default full fresh replay。

与第一个 local canary MaskCLIP 的同协议点估计对照：

| Metric | MaskCLIP | TruFor |
|---|---:|---:|
| local T1 macro AUROC | 0.488323 | **0.903579** |
| full-frame T1 macro AUROC | **0.709403** | 0.623285 |
| local pooled per-image pixel AP | 0.203323 | **0.676670** |
| local pooled per-image IoU@0.5 | 0.012472 | **0.415657** |
| local pooled micro IoU@0.5 | 0.001596 | **0.077767** |
| real mean per-image FP fraction | **0.000593** | 0.017764 |

这不是两篇论文的普遍排名，而是当前 checkpoint、Balanced250 数据与共享
协议下的结果。TruFor 显著强化 local T1/T2，但 MaskCLIP 在当前
full-frame T1 上更强，且其 real-map FP area 更低。

## 12. 解释、限制与复现入口

本 run 的正确表述是：

```text
native T1+T2 local-forensics baseline complete
local insertion T1 strong across all three object classes
native local T2 useful, with a large per-image versus micro-recall gap
full-frame conditional-edit T1 weak at low-FPR and released thresholds
full-frame T2 not applicable
real dense false-positive area non-trivial
commercial clearance false
```

不能写成：

- “TruFor 能可靠检测所有 AIGC”；
- “full-frame 条件编辑就是纯文生图”；
- “pixel AP 0.677 代表所有 forged pixels 都已定位”；
- “reliability map 是第二张 forged heatmap”；
- “full-frame raw map 已完成 T2”；
- “CMX 的 MIT license 让整个 TruFor 可以商用”。

Formal ID 和其结果目录是 finalized immutable evidence，不能覆盖。下面命令
只用于在干净、没有同名 finalized run 的隔离工作区复现完整流程：

```bash
export PYTHONDONTWRITEBYTECODE=1
export PYTHONHASHSEED=0
export CUDA_VISIBLE_DEVICES=6

/root/.cache/claimforge/venvs/trufor-ae54475/bin/python \
  -m eval.opensource.run_trufor_balanced \
  --repo-root . \
  --mode formal \
  --run-id trufor_cvpr2023_balanced250_v1_full1775_20260727 \
  --device cuda:0 \
  --fail-fast

/root/.cache/claimforge/venvs/trufor-ae54475/bin/python \
  -m eval.opensource.analyze_trufor_balanced \
  --repo-root . \
  --run-id trufor_cvpr2023_balanced250_v1_full1775_20260727 \
  --device cuda:0
```

GPU occupancy 由项目的 grouped benchmark supervisor 与 Hunyuan
keepalive 协议协调：CPU-only preparation、文档和 push 阶段保持
keepalive；CUDA group 前暂停并排空在途任务，按显式 GPU pin 运行，
整个 group 结束或异常时恢复一次。
