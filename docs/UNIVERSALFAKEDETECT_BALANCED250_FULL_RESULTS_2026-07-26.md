# UniversalFakeDetect released Ours (L/14 + LC) 在 Balanced250 上的正式结果

日期：2026-07-26（UTC）

正式 run：
`universalfakedetect_clip_vit_l14_ours_lc_current_head_native_center_crop224_balanced250_v1_full1775_r2_20260726`

核心机器证据：
[run manifest](../results/opensource/universalfakedetect/universalfakedetect_clip_vit_l14_ours_lc_current_head_native_center_crop224_balanced250_v1_full1775_r2_20260726/manifest.json)、
[逐图结果](../results/opensource/universalfakedetect/universalfakedetect_clip_vit_l14_ours_lc_current_head_native_center_crop224_balanced250_v1_full1775_r2_20260726/results.jsonl)、
[coverage summary](../results/opensource/universalfakedetect/universalfakedetect_clip_vit_l14_ours_lc_current_head_native_center_crop224_balanced250_v1_full1775_r2_20260726/summary.json)、
[Balanced250 metrics](../results/opensource/universalfakedetect/universalfakedetect_clip_vit_l14_ours_lc_current_head_native_center_crop224_balanced250_v1_full1775_r2_20260726/balanced250_metrics.json)、
[fresh replay audit](../results/opensource/universalfakedetect/universalfakedetect_clip_vit_l14_ours_lc_current_head_native_center_crop224_balanced250_v1_full1775_r2_20260726/independent_audit.json)、
[双 smoke comparison](../results/opensource/universalfakedetect/universalfakedetect_balanced_smoke_comparison_v2_34881d5c7c5466418f2cff944d15eb7bc14bb76f9764d03dda11b3afa8e4d3c5.json)。

## 1. 结论摘要

UniversalFakeDetect 的官方公开 `Ours (L/14 + LC)` 线性头已按冻结的
Balanced250 whole-image T1 协议完成正式运行：

- score cache 覆盖 1,775/1,775 张图，全部成功；error、missing 和
  superseded attempt 均为 0；
- 三种 local insertion 的 primary condition-macro AUROC 为
  **0.515899 [0.502395, 0.529261]**，AP 为
  **0.513165 [0.499001, 0.530189]**。虽然区间略高于 0.5，但效应很小，
  不能视为可用的局部植入检测能力；
- 三种 full-frame conditional edit 的 macro AUROC 为
  **0.703248 [0.673844, 0.735360]**，AP 为
  **0.717311 [0.684905, 0.753477]**，存在中等强度的可重复排序信号；
- 六条件等权 macro AUROC 为
  **0.609573 [0.591042, 0.628323]**。它混合了两种行为明显不同的
  manipulation family，不对应一个自然部署流量；
- 官方判定严格使用 `probability > 0.5`。在 primary real250 panel 上有
  10 个 false positive，FPR 为 4.0%；但 local 三条件 macro recall
  只有 **3.33%**，full-frame recall 也只有 **15.20%**。这个固定阈值
  很保守，不能只看 specificity 就认为已经可部署；
- source-matched secondary 呈现同样的分裂：750 个 local pair 的
  pooled mean forged-real score delta 为
  **+0.001099 [-0.004434, +0.006531]**，strict matched ranking
  accuracy 为 **0.216000 [0.186809, 0.251117]**；750 个 full-frame pair
  则分别为 **+0.137255 [0.108939, 0.166046]** 和
  **0.817333 [0.781554, 0.854693]**；
- 当前 source HEAD 的 validation transform 不做 resize，只取 native
  image 中央 `224 x 224`。750 张 local 图中有 463 张的 exact-difference
  GT 在 crop 内完全不可见，仅 22 张完整可见；这使本次 local 结果同时反映
  模型和当前预处理配置的限制；
- 两个 35-image CUDA smoke 的 raw logit、probability、AI score 和
  768 维 CLIP feature 均 bit-exact；
- fresh model replay 对全部 1,775 张 canonical JPEG 重新执行完整模型，
  feature、raw logit、probability 和 AI score 的最大差异全部为 0.0，
  最终 audit 状态为 `replay_audit_passed`。

最准确的读法是：这个冻结的 UniversalFakeDetect release/profile 对当前
三组 full-frame conditional edit 有中等排序能力，但对小面积局部植入的
whole-image 检测接近随机，而且当前 native CenterCrop(224) 会直接看不到
多数植入区域。结果不能外推为“能检测所有整图生成 AIGC”，因为本数据中的
full-frame 样本仍由真实源图条件编辑而来；也不能外推为定位能力。

本次运行捕获官方 `CLIPModel.fc` 的输入并保存 768 维 feature，使用公开
线性头的 sigmoid probability 作为 `ai_score`。模型只产生单个整图分数，
没有 dense map、mask 或 bbox。本 run 只计入 **T1 whole-image AIGC
detection**；**T2 localization 和 joint score 均为 N/A**。

## 2. 方法、release 与许可边界

### 2.1 原理与本次实际前向

UniversalFakeDetect 的核心不是寻找某个生成器特有的像素指纹，而是利用
大规模视觉-语言预训练得到的 CLIP 表征，把真实图与生成图投到更有迁移性的
语义/视觉特征空间，再只学习一个很小的线性二分类头。官方 README 说明主模型
使用 Wang et al. 2020 发布的 20 类 ProGAN real/fake training set，
训练时通过 `--fix_backbone` 冻结 CLIP，只优化线性层。

本次路径为：

```text
canonical JPEG
-> Pillow.Image.open(...).convert("RGB")
-> current source HEAD: native CenterCrop(224), no resize
-> float32 ToTensor + OpenAI CLIP normalization
-> frozen OpenAI CLIP ViT-L/14 image encoder
-> 768-dimensional float32 image feature
-> released Ours (L/14 + LC) Linear(768, 1)
-> float32 sigmoid
-> ai_score = probability
-> strict probability > 0.5
```

CLIP 的优势在于其表征不是只为某一个 GAN 的训练标签建立，因此对未见过的
生成架构可能有更好的迁移性。这个原理解释了它为什么在许多跨生成器 benchmark
中有效，但不保证对任意新域、局部编辑或新预处理都有效。本次结果正好显示：
全图编辑会显著改变中央 crop 的 CLIP 表征，而小区域改动很容易被 crop 排除
或被整图表征稀释。

`ai_score` 是公开线性分类器的 sigmoid 输出，不是经过本域 calibration 的
真实世界概率。AUROC/AP 与固定 `0.5` operating point 回答的是不同问题。

### 2.2 冻结 source 与资产

| 字段 | 冻结值 |
|---|---|
| official repository | `WisconsinAIVision/UniversalFakeDetect` |
| source commit | `76a0e3e60a8a06458707a625d269ba815a2e5919` |
| released head | `Ours (L/14 + LC)` |
| head parameters | 769 |
| head SHA-256 | `477100745713bcc957beb2b40859536859b6483fd6301b3b9293151b194c7847` |
| backbone | OpenAI CLIP ViT-L/14 |
| backbone bytes | 932,768,134 |
| backbone SHA-256 | `b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836` |
| asset bundle SHA-256 | `b57c7a8865336b82fe716dc7871006883417cf60f7b36ca8a9a5f925da009121` |
| repository code license | MIT |
| overall commercial clearance | not established |

Head checkpoint 通过
`torch.serialization.get_unsafe_globals_in_checkpoint` 后使用
`torch.load(..., weights_only=True)`，并以 strict state-dict 方式加载；
OpenAI CLIP backbone 使用官方 bundled loader 的 TorchScript 路径。运行时
把下载解析固定到本地已哈希 checkpoint，阻止 `urlopen` 和不受控
`torch.load` fallback，实际网络调用为 0。

仓库代码是 MIT，但未发现对公开线性头权重另行授予明确条款；OpenAI CLIP
代码虽为 MIT，其 model card 也不等于为所有部署用途提供商用保证。因此本
benchmark 记录为 `commercial_clearance = not_established`，不构成法律意见。

### 2.3 Released head 与 validation transform 的历史歧义

公开 head 首次出现于 commit
`763391eff3284f6950ffb323599c1a7a819f2ecd`。该时期的 `validate.py`
包含 `Resize(256) -> CenterCrop(224)`；之后 commit
`3bf72282088e47be7e784e104e577790a55d4e48` 删除了 resize。当前冻结 source
commit 只做 native `CenterCrop(224)`。

仓库没有把权重文件与应使用的 transform 版本建立机器可验证绑定。本次选择
并只执行 **current source HEAD profile**：

```text
profile_id: current_head_native_center_crop224
resize:     disabled
crop:       native center 224 x 224
```

没有执行 checkpoint-era `Resize(256)` profile。因此这里是
“公开 head + 当前 source validation transform”的结果，不能称为
checkpoint-era preprocessing replication。这个歧义是解释本次 local
结果时必须保留的限制。

## 3. Frozen T1 协议

实现入口：
[Balanced250 runner](../eval/opensource/run_universalfakedetect_balanced.py)
和
[Balanced250 analyzer](../eval/opensource/analyze_universalfakedetect_balanced.py)。
旧版 Mouse runner/analyzer 没有被改写。

### 3.1 图像、数值与判定

```text
Pillow RGB decode
-> no EXIF transpose
-> no ICC conversion
-> no resize
-> CenterCrop(224); dimensions below 224 use zero padding
-> uint8 / 255 to float32
-> normalize mean [0.48145466, 0.4578275, 0.40821073]
             std  [0.26862954, 0.26130258, 0.27577711]
-> batch size 1, no autocast, TF32 disabled
-> deterministic algorithms, CUBLAS_WORKSPACE_CONFIG=:4096:8
-> one official full-image forward
```

官方 output 与 forward-hook 捕获 feature 的手工
`F.linear(feature, weight, bias)` 必须在同一设备上逐位相等；sigmoid 也必须
逐位相等。每个成功结果保存精确 feature，并在 resume 前和 finalization 前
用记录的相同 CUDA runtime 以 logit/probability 绝对容差 0.0 重放分类头。

判定只依据已产生的 probability，严格使用 `> 0.5`；等于 0.5 不算 fake。
不同设备的 float32 sigmoid kernel 不要求逐位相同，因此不使用“把 CUDA
logit 拿到 CPU 再 sigmoid”的跨设备关系作为证据门槛。最终可信性来自更强的
同设备 feature-to-head exact replay 与完整模型 fresh replay。

### 3.2 数据与统计设计

数据集为
`claimforge-balanced250-independent-panel-jpeg-q95-v1`：

| Ledger | Rows | SHA-256 |
|---|---:|---|
| `inputs.jsonl` | 1,775 | `6b5128909eeffdbd88e61af02ca1bd191cb5460f94a23b47c87ebd0172e6d12c` |
| `panel.jsonl` | 1,750 | `e01d7985b41cee5262a3f8b6d71420986feae96771b11c46fda98c3e72a0d424` |
| `source_pairs.jsonl` | 1,500 | `391fdcf06eecff4cf1843ddb3688acacf52a293725c501660b7a361173b09b30` |

Release manifest SHA-256 为
`b2bbf3eb7a835f9c729cdffe29a40247225125779fe21551270fefe95d667c7f`，
deterministic contract SHA-256 为
`671d1739bebf4370d26b4629ca26b56cc546a817d469ba505cc39bda8b33102c`。

Primary point estimate 是 selection-unpaired：

```text
real250 vs local_mouse250
real250 vs local_cat250
real250 vs local_trash_can250
real250 vs fullframe_mouse250
real250 vs fullframe_cat250
real250 vs fullframe_trash_can250
```

每组共享同一个独立 real250 panel。Secondary 只使用
`source_pairs.jsonl` 的 1,500 个显式 real/forged endpoint，不从
`task_id` 猜配对。额外 25 张 non-panel real 只用于补齐这些 source pair。

置信区间使用跨 label/condition 共享 source-content-cluster 权重的 Poisson
bootstrap，1,000 次，root seed `20260726`。TPR@5%FPR 的 threshold 使用
real score 95th percentile、`method="higher"` 和 strict `>`；它只是报告
诊断，不替代 release 固定阈值。

### 3.3 Full-frame 与 T2 边界

`fullframe_mouse`、`fullframe_cat` 和 `fullframe_trash_can` 是以真实图为
条件执行 Hunyuan full-frame edit 的结果，不是脱离真实源图的纯 T2I。
Trash-can condition 只表示样本通过 single-shot 生成流程，不保证目标物体
经过额外语义 QC 后必然成功出现。

UniversalFakeDetect 的冻结路径只有整图分数。后文 crop visibility 来自
canonical exact-difference GT，是**输入可见性诊断**，不是模型预测、
localization 指标或 T2 输出。Real 与 full-frame 的 T2 均为 N/A。

## 4. Coverage

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

其他门禁：

```text
physical result rows: 1775
latest result rows:   1775
superseded attempts:  0
coverage fraction:    1.0
success fraction:     1.0
feature files:        1775
```

正式 selection ID SHA-256 为
`e4418d86461f889e4a4423f26aab63243e6f63a435a49624881c34979b812e41`。

## 5. Primary：whole-image T1

下表每行都是同一个 real250 panel 对一个 forged250 condition。区间为双侧
95% percentile CI。Confusion 使用 strict `probability > 0.5`。

| Forged condition | AUROC [95% CI] | AP [95% CI] | TPR@5%FPR [95% CI] | Accuracy@0.5 | Recall@0.5 | TP / FP / FN / TN |
|---|---:|---:|---:|---:|---:|---:|
| `local_mouse` | 0.503840 [0.487900, 0.518919] | 0.501596 [0.479927, 0.524326] | 0.040000 [0.024896, 0.054184] | 0.498000 | 0.036000 | 9 / 10 / 241 / 240 |
| `local_cat` | 0.538696 [0.515041, 0.562458] | 0.530252 [0.506562, 0.559432] | 0.056000 [0.029915, 0.082401] | 0.498000 | 0.036000 | 9 / 10 / 241 / 240 |
| `local_trash_can` | 0.505160 [0.489741, 0.520179] | 0.507647 [0.489793, 0.526279] | 0.044000 [0.025269, 0.059041] | 0.494000 | 0.028000 | 7 / 10 / 243 / 240 |
| `fullframe_mouse` | 0.703152 [0.673387, 0.736976] | 0.715229 [0.684425, 0.752476] | 0.200000 [0.115547, 0.300414] | 0.552000 | 0.144000 | 36 / 10 / 214 / 240 |
| `fullframe_cat` | 0.687520 [0.654243, 0.721760] | 0.700865 [0.658957, 0.741449] | 0.224000 [0.095756, 0.305226] | 0.548000 | 0.136000 | 34 / 10 / 216 / 240 |
| `fullframe_trash_can` | 0.719072 [0.685376, 0.752870] | 0.735839 [0.700998, 0.772963] | 0.252000 [0.139142, 0.350191] | 0.568000 | 0.176000 | 44 / 10 / 206 / 240 |

条件等权 macro：

| Family | AUROC [95% CI] | AP [95% CI] | TPR@5%FPR [95% CI] | Accuracy@0.5 | Recall@0.5 |
|---|---:|---:|---:|---:|---:|
| local 三条件 | 0.515899 [0.502395, 0.529261] | 0.513165 [0.499001, 0.530189] | 0.046667 [0.030953, 0.058423] | 0.496667 | 0.033333 |
| full-frame 三条件 | 0.703248 [0.673844, 0.735360] | 0.717311 [0.684905, 0.753477] | 0.225333 [0.123017, 0.312028] | 0.556000 | 0.152000 |
| 全六条件 | 0.609573 [0.591042, 0.628323] | 0.615238 [0.594579, 0.637911] | 0.136000 [0.077946, 0.182055] | 0.526333 | 0.092667 |

六个 comparison 共享同一批 real score，所以每行的 false positive 都是
10/250，而不是六批独立错误。正式 1,775 张 cache 的 strict-threshold
positive 诊断为：

| Condition | Positive / total |
|---|---:|
| real | 11 / 275 |
| local mouse / cat / trash can | 9 / 250；9 / 250；7 / 250 |
| full-frame mouse / cat / trash can | 36 / 250；34 / 250；44 / 250 |

### 5.1 Domain 诊断

| Family | Domain | AUROC [95% CI] | AP [95% CI] | TPR@5%FPR [95% CI] |
|---|---|---:|---:|---:|
| local | lodging | 0.509974 [0.489338, 0.529925] | 0.518869 [0.500300, 0.542563] | 0.033835 [0.019220, 0.052421] |
| local | restaurant | 0.516416 [0.497518, 0.536141] | 0.503888 [0.483331, 0.530697] | 0.041616 [0.026823, 0.066972] |
| full-frame | lodging | 0.644273 [0.603443, 0.688186] | 0.655260 [0.611846, 0.710377] | 0.102610 [0.064318, 0.214839] |
| full-frame | restaurant | 0.762270 [0.720338, 0.806188] | 0.788109 [0.748362, 0.832411] | 0.387595 [0.288455, 0.496694] |
| all-six | lodging | 0.577123 [0.549310, 0.604729] | 0.587064 [0.560703, 0.622176] | 0.068222 [0.046962, 0.127914] |
| all-six | restaurant | 0.639343 [0.615097, 0.663920] | 0.645998 [0.622641, 0.674284] | 0.214605 [0.163580, 0.273640] |

Full-frame restaurant 的点估计明显高于 lodging，但这里没有执行专门的
simultaneous hypothesis test。该表是分域诊断，不证明 domain 因果效应。

## 6. Secondary：显式 source-matched 分析

`score delta = forged probability - matched real probability`，正数表示 forged
被排得更假。Strict matched ranking 把 tie 计为不胜。

| Condition | Pairs | Win / loss / tie | Mean delta [95% CI] | Median delta [95% CI] | Strict ranking [95% CI] |
|---|---:|---:|---:|---:|---:|
| `local_mouse` | 250 | 17 / 12 / 221 | -0.004186 [-0.011433, 0.000588] | 0.000000 [0.000000, 0.000000] | 0.068000 [0.039465, 0.102131] |
| `local_cat` | 250 | 66 / 35 / 149 | +0.010741 [0.001658, 0.020391] | 0.000000 [0.000000, 0.000000] | 0.264000 [0.213948, 0.320166] |
| `local_trash_can` | 250 | 79 / 79 / 92 | -0.003259 [-0.012719, 0.003644] | 0.000000 [0.000000, 0.000000] | 0.316000 [0.259252, 0.373519] |
| `fullframe_mouse` | 250 | 210 / 40 / 0 | +0.130698 [0.100308, 0.162062] | +0.026049 [0.016835, 0.037697] | 0.840000 [0.795079, 0.886869] |
| `fullframe_cat` | 250 | 189 / 61 / 0 | +0.120709 [0.090549, 0.151056] | +0.019308 [0.009029, 0.032881] | 0.756000 [0.702343, 0.804551] |
| `fullframe_trash_can` | 250 | 214 / 36 / 0 | +0.160358 [0.128793, 0.195065] | +0.031548 [0.015383, 0.057412] | 0.856000 [0.811794, 0.900901] |

Family pooled：

| Family | Pairs | Mean delta [95% CI] | Median delta [95% CI] | Strict ranking [95% CI] |
|---|---:|---:|---:|---:|
| local pooled | 750 | +0.001099 [-0.004434, 0.006531] | 0.000000 [0.000000, 0.000000] | 0.216000 [0.186809, 0.251117] |
| full-frame pooled | 750 | +0.137255 [0.108939, 0.166046] | +0.024261 [0.014386, 0.037575] | 0.817333 [0.781554, 0.854693] |
| all pairs pooled | 1,500 | +0.069177 [0.054558, 0.084295] | +0.000017 [0.000000, 0.000098] | 0.516667 [0.493697, 0.540252] |

Local 的 462 个 tie 不是普通的“模型犹豫”：当 forged 与 matched real 在
所选中央 crop 内具有完全相同的像素时，冻结 CLIP feature 和分数就完全相同。
这与下一节的 463 个 `visibility=none` 几乎一一对应。这个现象证明当前 profile
没有接收到多数 edit，而不是证明模型定位了 edit。

## 7. Center-crop visibility 诊断

当前 profile 直接在 native image 中央取 224×224。Local exact-difference
GT 在 crop 中的可见情况为：

| Local condition | Full | Partial | None | Mean visible GT fraction |
|---|---:|---:|---:|---:|
| mouse | 14 | 14 | 222 | 0.079217 |
| cat | 6 | 95 | 149 | 0.104637 |
| trash can | 2 | 156 | 92 | 0.056629 |
| **Pooled** | **22** | **265** | **463** | — |

Matched-pair 交叉检查为：

```text
local_mouse:      none/tie 221, none/different 1
local_cat:        none/tie 149
local_trash_can:  none/tie 92
partial/full:     all scores differ from their matched real
```

`none` 的定义是没有 exact-difference GT 正像素中心落入 effective native
crop。Mouse 中唯一一个 `none/different` 可能来自 JPEG、边界或像素中心定义
的低位影响，不能把 `none` 解释为整个前向在数学上必然相同。反过来，
`partial/full` 也不意味着整图分类器一定能检测。这个表不是 localization
accuracy。

## 8. 可重复性与运行审计

### 8.1 CPU golden preflight

在任何 accelerator 配置之前，runner 对固定 canonical 图
`5f7535f0b957874982b1b080` 执行两次完整 CPU 前向：

| 字段 | 固定值 |
|---|---|
| image SHA-256 | `f90c849192fd53e2e9560192d91b5b37a6162f80c14c862e24d37482784b8078` |
| crop RGB SHA-256 | `105cbdcc566d3a48b85c5b34198c81dfd2a69a6ec79fe1005af1f7c36dc31dbe` |
| tensor SHA-256 | `bf9ce4ebbfd24f886d8bf70845b85d194ad942439275934e36b23284c22ca0cb` |
| raw logit | `-1.3850042819976807` |
| probability / AI score | `0.20020648837089539` |
| decision | false |
| feature array SHA-256 | `6e320b71683f9a9a294d497c2a895ff437854e543ed2f4407ff4128b9296c29d` |
| feature `.npy` SHA-256 | `325da96b2f9d3e2d060bb8e19b49835b29e0190182d40070b2f58b3e0bcfb06d` |

两次 feature 文件 byte-exact，CUDA 在 preflight 前后都未初始化。

### 8.2 A/B deterministic smoke

两个 smoke 各取七条件前 5 张，共 35 张。Selection SHA-256 为
`b420bc581386a540b742d917d60d007f0e5522b6cca43fa217797944c40667e5`。
Comparison 状态为 `deterministic_smoke_comparison_passed`：

```text
images compared:                    35
exact computational projection:     true
feature file bytes exact:           true
feature arrays exact:               true
max raw-logit difference:           0.0
max probability difference:         0.0
max AI-score difference:            0.0
max feature difference:             0.0
```

两个 smoke 的 independent linear-head replay 也各自以
raw/probability tolerance 0.0 通过。

### 8.3 正式 fresh model replay

独立 analyzer 从 canonical ledger 重建完整 selection，验证全部 physical
row、feature path/bytes/file SHA/array SHA/shape/dtype/finite、source、
assets、runtime、preprocess 和 metrics contract。随后：

```text
independent feature-to-head replay:  passed
features replayed:                   1775
recorded runtime exact match:        true
max raw-logit difference:            0.0
max probability difference:          0.0

fresh full model replay:             passed
images replayed:                     1775
full-image forward per input:        true
feature-tail-only replay:            false
feature comparison:                  numpy.array_equal
max feature difference:              0.0
max raw-logit difference:            0.0
max probability difference:          0.0
max AI-score difference:             0.0
final audit status:                   replay_audit_passed
```

正式 runtime 为 NVIDIA L20Z、CUDA 12.8、PyTorch 2.8.0+cu128、
torchvision 0.23.0+cu128、NumPy 2.2.6、Pillow 11.1.0、SciPy 1.17.1
和 scikit-learn 1.8.0。正式 manifest 的
`started_at -> completed_at` window 为约 653.4 秒；它从 CPU preflight
之后开始，不是完整 CLI wall time。逐图官方模型 forward 的平均 latency
约 12.08 ms，中位数 10.61 ms，最大 peak CUDA allocation 为
1,806,185,472 bytes；这些 latency 不含 decode、hash、落盘、逐图 GC 和
benchmark orchestration。

### 8.4 跨设备 sigmoid 数值门禁修正

正式 run 前的一个 incomplete exploratory run 在样本
`de258d452de14508bbfe9b33` 被 host-CPU sigmoid sanity check 误拒绝：

```text
raw logit:             2.6707892417907715
official CUDA sigmoid: 0.9352807402610779
CPU sigmoid replay:    0.9352808594703674
difference:            1.1920928955078125e-7
bit distance:          2 float32 ULP
```

官方 CUDA probability 离 float64 sigmoid 的 float32 正确舍入只差 1 ULP。
PyTorch 不保证 CPU/CUDA sigmoid kernel 在固定十进制 tolerance 内一致，
所以这个跨设备检查不是合法的 exact invariant。最终 adapter 删除了该门禁，
增加 probability `[0,1]` 约束，并保留更强的：

- 同设备 official output 与手工 `F.linear + sigmoid` 逐位相等；
- resume/finalization 的同记录设备 feature-to-head 零容差重放；
- analyzer 的 1,775 张全模型 fresh replay。

回归测试覆盖这个精确 CUDA 数值。Incomplete exploratory output 没有导入、
改写或混入最终 r2 run；最终 run 使用新的 adapter SHA/fingerprint 从零执行。

## 9. 哈希、产物、测试与命令

| 文件 | SHA-256 |
|---|---|
| manifest | `a3f70959e815b2220a611350d01cfb3019276cdb800c2bf4bccc81d113366e9e` |
| expected inputs | `6b5128909eeffdbd88e61af02ca1bd191cb5460f94a23b47c87ebd0172e6d12c` |
| results | `580e05eb596ee0090416e732d88f2217d03df99b83def2170065fa3b9062a665` |
| summary | `ce50c144f9127ad4873658b2e138f314023750e0469159fffef7331895ffa2cc` |
| metrics | `1c5802aea3adc7671d576de0950c25eeff9c4e3e0a5917d3a0f7547452091c5d` |
| independent audit | `e809a9f16b796ca659d4561fb96d25955a06a3e734244ba5b9d48b368cad0383` |
| feature inventory | `9efedea4b9a4cf140db5383f015fa766a5a73c755461e34597f661da3ae49dd1` |
| smoke comparison | `6f1e8e3c8bc8ecc9a98ab1e62139a6da37b323b13ff341498fd15bdf1cde5b6d` |

冻结实现：

| 文件 | SHA-256 |
|---|---|
| `run_universalfakedetect_balanced.py` | `4e12c296fc0daea4e904b8bcbad5f0fcd0c2d2966d559290a6322ac858e1e32c` |
| `analyze_universalfakedetect_balanced.py` | `2ba89579ef714414ba437e3828414f0eea866eba8f9f21d3625ba6e9a5a8e4f2` |
| `test_run_universalfakedetect_balanced.py` | `e54d78adb42ed099e2497a47d2719fd83a50897558c5bec854133bffc7a1b23a` |
| `test_analyze_universalfakedetect_balanced.py` | `da01c8d219920257f10aca828af39233c5a0cd4a20737234796cc3a74325a3ad` |

1,775 个 float32 feature 位于 gitignored `outputs/`，不会随 git clone
自动取得。每个 results row 都绑定其相对路径、bytes、文件 SHA、array SHA、
shape、dtype 和语义；fresh replay audit 依赖本机保留这些 feature。

核心重跑命令如下；`CUDA_VISIBLE_DEVICES=4` 是本机物理卡映射，其他机器应
按实际设备调整：

```bash
export CUDA_VISIBLE_DEVICES=4
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONPYCACHEPREFIX=/tmp/claimforge-ufd-replay-pyc
UFD_PY=/root/.cache/claimforge/venvs/ufd-balanced-torch2.8.0/bin/python

$UFD_PY -m eval.opensource.run_universalfakedetect_balanced \
  --mode formal \
  --run-id universalfakedetect_clip_vit_l14_ours_lc_current_head_native_center_crop224_balanced250_v1_full1775_r2_20260726 \
  --device cuda:0 \
  --fail-fast

$UFD_PY -m eval.opensource.analyze_universalfakedetect_balanced \
  --run-id universalfakedetect_clip_vit_l14_ours_lc_current_head_native_center_crop224_balanced250_v1_full1775_r2_20260726 \
  --device cuda:0
```

实际运行额外使用 Hunyuan keepalive handoff wrapper，只负责在 CUDA 阶段
暂停/恢复后台生成，不改变模型、输入、runtime 或数值路径。

## 10. 支持与不支持的结论

推荐论文表述：

> The released UniversalFakeDetect Ours (CLIP ViT-L/14 + LC) head,
> evaluated with the repository's current native CenterCrop(224) transform
> on the independent Balanced250 panel, was near chance on local insertions
> (condition-macro AUROC 0.516) but transferred better to full-frame
> conditional edits (AUROC 0.703). Source-matched comparisons showed the
> same split (strict ranking 0.216 vs. 0.817). The current crop excluded all
> exact-difference pixels for 463/750 local samples.

不要写成：

- “复现了论文训练流程或论文中的跨生成器 benchmark 数字”；
- “UniversalFakeDetect 能可靠检测本数据中的局部植入”；
- “full-frame conditional edit 等于纯文生图或完全合成图”；
- “crop visibility 是模型的 localization 输出”；
- “sigmoid score 是本域校准概率”；
- “checkpoint-era Resize(256) profile 已经评估”；
- “固定 0.5 阈值在本域已经适合部署”；
- “MIT code license 自动建立了全部权重和模型的商用授权”。
