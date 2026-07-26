# OpenSDI / MaskCLIP 在 Balanced250 上的正式结果

日期：2026-07-26
正式运行：
`maskclip_sd15_balanced250_v1_full1775_20260726`

## 1. 结论

MaskCLIP 已按冻结的 Balanced250 `LOCAL_T1_T2` 合同完成正式运行：

- 1,775/1,775 张成功，0 error、0 missing、0 superseded；
- T1 覆盖 275 张 real、三类 local 各 250 张、三类 full-frame 各
  250 张；
- T2 仅适用于 275 张 real 和 750 张 local，共 1,025 张；
- 750 张 full-frame 只有诊断用的 512×512 模型热图，T2 严格为
  `N/A`；
- A/B smoke、逐文件 artifact audit 和 1,775 张 fresh model replay
  全部通过。

核心结果是：MaskCLIP 对本数据中的局部植入几乎没有可靠的整图判别能力，
但对 full-frame 条件编辑有中等强度的排序能力。局部 T1 的三条件宏平均
AUROC 为 `0.488323`，full-frame T1 为 `0.709403`。定位热图有一些连续
排序信号，但官方固定阈值 `>= 0.5` 极其保守：750 张 local 的 pooled
per-image pixel AP 为 `0.203323`，而 pooled micro IoU 仅
`0.001596`。

可复核产物：

- [正式 run manifest](../results/opensource/maskclip/maskclip_sd15_balanced250_v1_full1775_20260726/manifest.json)
- [逐图结果](../results/opensource/maskclip/maskclip_sd15_balanced250_v1_full1775_20260726/results.jsonl)
- [coverage summary](../results/opensource/maskclip/maskclip_sd15_balanced250_v1_full1775_20260726/summary.json)
- [Balanced250 metrics](../results/opensource/maskclip/maskclip_sd15_balanced250_v1_full1775_20260726/balanced250_metrics.json)
- [artifact + fresh replay audit](../results/opensource/maskclip/maskclip_sd15_balanced250_v1_full1775_20260726/independent_audit.json)
- [A/B smoke comparison](../results/opensource/maskclip/maskclip_sd15_balanced250_v1_smoke5x7_a_20260726__vs__maskclip_sd15_balanced250_v1_smoke5x7_b_20260726_comparison.json)

## 2. 方法与冻结合同

本次使用 OpenSDI 的原生 MaskCLIP `ViTL` 模型：

- OpenSDI commit：
  `02c93d4891303637cb5d6852d3de63a099d69843`；
- MaskCLIP checkpoint：
  `MaskCLIP_sd15_20241109_08_53_19.pth`，
  SHA-256
  `481c8bd16077f942efec2901f93c1bc7008f6992402a1ab69fda2652408ca90f`；
- MAE 初始化权重 SHA-256：
  `aec5f0b68e5f3193a00b07bc65a37440db549c15b36b8bea242606cc40c4bc5d`；
- OpenAI CLIP ViT-L/14 权重 SHA-256：
  `b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836`。

预处理沿用官方路径：OpenCV `INTER_LINEAR` 拉伸到 512×512，随后使用
CLIP mean/std 做 float32 标准化。T1 分数为
`softmax(class_logits)[forged]`，方向是越高越假，冻结阈值为
`ai_score >= 0.5`。

T2 使用模型原生 512×512 forged probability map。对 T2 适用图像，
它通过 OpenCV `INTER_LINEAR` 精确恢复到原图尺寸；二值 mask 使用
`native_probability >= 0.5`。local GT 是 canonical real 与 local
图像的完整逐像素 exact-difference mask；real GT 全零，因此 real 不报告
无定义的 pixel AP，只报告假阳性面积。

实现：

- [Balanced250 runner](../eval/opensource/run_maskclip_balanced.py)
- [Balanced250 analyzer](../eval/opensource/analyze_maskclip_balanced.py)
- [共享 T2 指标](../eval/opensource/balanced250_localization_metrics.py)

## 3. 数据范围和统计设计

正式输入条件如下：

| 条件 | 张数 | T1 | T2 |
|---|---:|---:|---:|
| real | 275 | 是 | 假阳性面积 |
| local mouse | 250 | 是 | 是 |
| local cat | 250 | 是 | 是 |
| local trash can | 250 | 是 | 是 |
| full-frame mouse | 250 | 是 | N/A |
| full-frame cat | 250 | 是 | N/A |
| full-frame trash can | 250 | 是 | N/A |
| 合计 | 1,775 | 1,775 | 1,025 |

这里的 full-frame 样本是同域的**整图条件编辑**，不是可以不加区分地写成
“纯文生图”或“完全合成图”的集合。它们适合评估整图 T1 信号，但没有
局部植入 GT，不能把橙框、全图 mask 或模型热图强行当成 T2 真值。

T1 primary point estimate 使用每个 forged 条件 250 张与独立 panel 中
250 张 real，按条件做 selection-unpaired 比较。secondary 分析只使用
明确记录的 250 对 source-matched endpoints。置信区间使用共享
source-content cluster 的 Poisson bootstrap，1,000 次，根 seed
`20260726`；重复源内容不会被当成独立证据。

## 4. T1 整图判别结果

下表每个条件均为 250 real 对 250 forged。括号为 95% percentile CI。
`TP/FP/FN/TN` 使用冻结阈值 `>= 0.5`。

| forged 条件 | AUROC | AP | TPR@5% FPR | Accuracy@0.5 | Recall@0.5 | TP/FP/FN/TN |
|---|---:|---:|---:|---:|---:|---:|
| local mouse | 0.509112 [0.494575, 0.524070] | 0.509821 [0.492315, 0.534789] | 0.048 [0.035017, 0.077492] | 0.500 [0.488592, 0.511958] | 0.024 [0.008264, 0.043483] | 6/6/244/244 |
| local cat | 0.510256 [0.495043, 0.526091] | 0.515822 [0.499522, 0.540557] | 0.060 [0.035019, 0.096787] | 0.500 [0.487076, 0.513338] | 0.024 [0.008096, 0.044645] | 6/6/244/244 |
| local trash can | 0.445600 [0.428834, 0.460771] | 0.465937 [0.451870, 0.488224] | 0.040 [0.021728, 0.060840] | 0.502 [0.488056, 0.517085] | 0.028 [0.008583, 0.049623] | 7/6/243/244 |
| full-frame mouse | 0.701152 [0.670457, 0.730007] | 0.692020 [0.655769, 0.732905] | 0.208 [0.125000, 0.271573] | 0.542 [0.519189, 0.564557] | 0.108 [0.071153, 0.146122] | 27/6/223/244 |
| full-frame cat | 0.740096 [0.713619, 0.768116] | 0.733924 [0.698469, 0.771716] | 0.248 [0.173570, 0.323660] | 0.572 [0.547004, 0.596567] | 0.168 [0.123505, 0.213992] | 42/6/208/244 |
| full-frame trash can | 0.686960 [0.659846, 0.713420] | 0.670461 [0.635892, 0.707871] | 0.172 [0.105838, 0.235583] | 0.544 [0.521434, 0.567994] | 0.112 [0.076025, 0.152686] | 28/6/222/244 |

条件宏平均：

| family | AUROC | AP | TPR@5% FPR | Accuracy@0.5 | Recall@0.5 |
|---|---:|---:|---:|---:|---:|
| local 三条件 | 0.488323 [0.476783, 0.500079] | 0.497193 [0.487198, 0.518175] | 0.049333 [0.036498, 0.070273] | 0.500667 [0.489667, 0.510782] | 0.025333 [0.011094, 0.042501] |
| full-frame 三条件 | 0.709403 [0.683446, 0.735654] | 0.698802 [0.666457, 0.736292] | 0.209333 [0.140860, 0.267331] | 0.552667 [0.531754, 0.572109] | 0.129333 [0.095756, 0.163091] |
| 全六条件 | 0.598863 [0.583638, 0.614227] | 0.597997 [0.579844, 0.623630] | 0.129333 [0.091769, 0.164743] | 0.526667 [0.512708, 0.539443] | 0.077333 [0.058104, 0.098334] |

固定阈值在 250 张 primary real 上只产生 6 个 false positives，说明它很
保守；代价是 local 三类合计只命中 19/750，full-frame 三类也只命中
97/750。不能用约 97.6% 的 specificity 掩盖极低的 forged recall。

source-matched secondary 结果进一步显示这种差异：

| 条件 | forged score 严格高于 source 的比例 | mean score delta |
|---|---:|---:|
| local mouse | 0.640 [0.579536, 0.700409] | +0.004244 [0.001991, 0.006574] |
| local cat | 0.624 [0.559992, 0.684631] | +0.006557 [0.000560, 0.012739] |
| local trash can | 0.252 [0.199186, 0.309239] | -0.013335 [-0.022383, -0.005348] |
| full-frame mouse | 0.844 [0.798394, 0.885328] | +0.109624 [0.081826, 0.137072] |
| full-frame cat | 0.876 [0.832696, 0.915325] | +0.147233 [0.114490, 0.181626] |
| full-frame trash can | 0.808 [0.754462, 0.858930] | +0.088202 [0.063343, 0.115792] |

因此，MaskCLIP 对整图条件编辑确实存在可重复的相对排序信号；局部信号则
弱、类别依赖明显，trash-can local 甚至主要朝错误方向移动。

## 5. T2 局部定位结果

下表的 AP、Recall 和 IoU 是每张图先计算、再取平均；阈值指标使用
`native_probability >= 0.5`。

| local 条件 | GT 正像素比例 | per-image pixel AP | per-image Recall@0.5 | per-image IoU@0.5 | micro IoU@0.5 |
|---|---:|---:|---:|---:|---:|
| mouse | 0.001350 | 0.048467 [0.031449, 0.068028] | 0.006314 [0.000295, 0.014961] | 0.004058 [0.000291, 0.009098] | 0.003423 [0.000530, 0.007203] |
| cat | 0.063678 | 0.267391 [0.234356, 0.298777] | 0.041007 [0.020794, 0.065862] | 0.029718 [0.014910, 0.046512] | 0.005355 [0.002077, 0.010409] |
| trash can | 0.219667 | 0.294111 [0.264226, 0.323059] | 0.003643 [0.000127, 0.010371] | 0.003641 [0.000127, 0.010367] | 0.000499 [0.000117, 0.001035] |

三类 pooled（750 张）：

| 指标 | estimate [95% CI] |
|---|---:|
| per-image pixel AP | 0.203323 [0.186794, 0.220121] |
| per-image precision@0.5 | 0.537062 [0.402718, 0.679939] |
| per-image recall@0.5 | 0.016988 [0.009467, 0.025319] |
| per-image F1@0.5 | 0.015990 [0.009175, 0.023586] |
| per-image IoU@0.5 | 0.012472 [0.006937, 0.018770] |
| micro recall@0.5 | 0.001600 [0.000745, 0.002773] |
| micro F1@0.5 | 0.003186 [0.001485, 0.005514] |
| micro IoU@0.5 | 0.001596 [0.000743, 0.002765] |

pooled GT 正像素比例为 `0.095385`，预测正像素比例只有
`0.0003857`。这解释了为什么 precision 看起来不低，而 recall/IoU
非常低：模型的默认阈值几乎不点亮像素。连续 AP 与固定阈值结果回答的是
不同问题，不能把 `0.203323` 的 AP 写成“已成功定位”。

275 张 real 的 GT 全零：

- pixel AP：`null`，因为全零 target 下无定义；
- 假阳性像素：109,527 / 442,253,004；
- mean per-image false-positive fraction：
  `0.0005933 [0.0000529, 0.0013721]`，即约
  `0.05933% [0.00529%, 0.13721%]`；
- micro false-positive fraction：
  `0.0002477 [0.0000349, 0.0006029]`。

## 6. 可重复性和独立审计

### A/B deterministic smoke

两个 smoke 各取七个条件的前 5 张，共 35 张；其中 20 张 T2 适用。
comparison 状态为 `deterministic_smoke_comparison_passed`：

- logits、class probabilities、AI score、512 map 的最大差均为 0；
- 35 份 512 map 文件 byte-exact；
- 20 份 native map 和 20 份 native mask 文件 byte-exact。

### 正式 artifact audit

正式目录约 7.9GB，不进入 git；结果账本保存其路径、bytes、SHA-256、
shape、dtype 和数值范围。独立 analyzer 重新验证：

- 1,775 份 512×512 float32 model maps；
- 1,025 份原尺寸 float32 native maps；
- 1,025 份原尺寸 PNG binary masks；
- native map 精确等于独立 `INTER_LINEAR` 恢复结果；
- mask 精确等于 `native >= 0.5`；
- 750 个 full-frame row 的 native/mask/localization 字段全部为 null。

### Fresh model replay

fresh replay 在 NVIDIA L20Z、float32、batch size 1、禁用 TF32、
deterministic algorithms 和
`CUBLAS_WORKSPACE_CONFIG=:4096:8` 下重算全部 1,775 张。结果：

- images replayed：1,775；
- T2-applicable images replayed：1,025；
- max logit difference：0；
- max probability difference：0；
- max AI-score difference：0；
- max 512-map difference：0；
- max applicable native-map difference：0；
- derived native threshold masks：逐像素 exact。

最终 audit 状态为 `replay_audit_passed`。

## 7. 哈希和测试

| 文件 | SHA-256 |
|---|---|
| manifest | `68144bd8376210567417f2960ac3039c7607a89c2dfa6fb3037072cba05370f4` |
| expected inputs | `6b5128909eeffdbd88e61af02ca1bd191cb5460f94a23b47c87ebd0172e6d12c` |
| results | `78c36935eb3e7a2ac25a340eabf36e401fdd99c6b25c6338821fca08ee50c04f` |
| summary | `076a8325527513d480e474751aec945d8fe15d9be2532751b0d045d46fa24edc` |
| metrics | `d67ef86f24f75ecbb8fc2eaac4db0d7d67061cc8b044bcac665242202b8349c0` |
| independent audit | `ded67fd742b012bd249b1b920f3cc4a1f2f1fb60343cf46f98b4ba487a7b8a60` |
| smoke comparison | `a559c7d937185fa45e984c4dea73f3dde0525902a627485c4290cf6b6c100dba` |

冻结实现哈希：

- runner：
  `3c95cd49410a7cef26a09673930e778ca83614524dd4881791be697079616a27`；
- analyzer：
  `ab97f98bd933674a730f1e9f4fd776c9a8682a55649056282a5b51632ef4d79b`；
- T2 metrics：
  `83ac07257078fc41276742fa4b9f2eb936ac51c8ff93bf1253b8c45f2b704b2a`。

相关 CPU 回归为 124/124 passed；`py_compile`、全部 JSON/JSONL parse 和
`git diff --check` 均通过。

## 8. 许可边界

本次 checkout 中没有找到明确覆盖 OpenSDI code 的 LICENSE 文件。README
里关于 OpenSDI dataset 的 CC BY-SA 4.0 / academic-use 声明，不能自动
延伸为对代码或模型权重的商用、再分发许可。因此，这个结果证明技术上可以
审计和复现，不等于已经取得 code/weights 的商用授权；任何商用集成或权重
再分发都需要另行向权利人确认。

## 9. 论文可用表述

推荐写成：

> OpenSDI MaskCLIP, evaluated with its released SD1.5 checkpoint and native
> image/dense outputs on the independent Balanced250 panel, was near chance
> on local insertions (condition-macro AUROC 0.488) but transferred better to
> full-frame conditional edits (AUROC 0.709). Its local heatmaps retained
> limited continuous ranking signal (pooled per-image pixel AP 0.203), while
> the released 0.5 operating point produced very low localization recall
> (pooled per-image recall 0.017; micro IoU 0.0016).

不要写成：

- “MaskCLIP 能可靠检测所有 AIGC”；
- “full-frame 条件编辑就是纯文生图”；
- “pixel AP 0.203 代表默认阈值下定位成功”；
- “full-frame 已完成 T2”；
- “OpenSDI dataset 的许可自动授权了代码和权重商用”。
