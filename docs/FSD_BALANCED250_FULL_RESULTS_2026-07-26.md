# FSD official v1.2 inference release 在 Balanced250 上的正式结果

日期：2026-07-26（UTC）

正式 run：
`fsd_v1_2_0_official_balanced250_v1_full1775_20260726`

核心机器证据：
[run manifest](../results/opensource/fsd/fsd_v1_2_0_official_balanced250_v1_full1775_20260726/manifest.json)、
[逐图结果](../results/opensource/fsd/fsd_v1_2_0_official_balanced250_v1_full1775_20260726/results.jsonl)、
[coverage summary](../results/opensource/fsd/fsd_v1_2_0_official_balanced250_v1_full1775_20260726/summary.json)、
[Balanced250 metrics](../results/opensource/fsd/fsd_v1_2_0_official_balanced250_v1_full1775_20260726/balanced250_metrics.json)、
[fresh replay audit](../results/opensource/fsd/fsd_v1_2_0_official_balanced250_v1_full1775_20260726/independent_audit.json)、
[双 smoke comparison](../results/opensource/fsd/fsd_v1_2_0_official_balanced250_v1_smoke5x7_a_20260726__vs__fsd_v1_2_0_official_balanced250_v1_smoke5x7_b_20260726_comparison.json)。

## 1. 结论摘要

FSD 的官方 v1.2 推理 API 已按冻结的 Balanced250 whole-image T1 协议
完成正式运行：

- score cache 覆盖 1,775/1,775 张图，全部成功；error、missing 和
  superseded attempt 均为 0；
- 三种 local insertion 的 primary condition-macro AUROC 为
  **0.489491 [0.477632, 0.500448]**，AP 为
  **0.497740 [0.484460, 0.517118]**，即没有显示出可靠的整图排序能力；
- 三种 full-frame conditional edit 的 primary macro AUROC 为
  **0.700075 [0.668743, 0.732202]**，AP 为
  **0.703137 [0.668915, 0.743228]**，存在中等强度的可重复排序信号；
- 六条件等权 macro AUROC 为
  **0.594783 [0.575546, 0.613234]**。它只是混合 local 与 full-frame
  的诊断汇总，不对应某个自然部署流量；
- release 固定判定是 `ai_score > 2.0`。在 primary 的 250 张 real
  panel 上有 66 个 false positive，FPR 为 26.4%；local 三条件
  macro recall 为 25.87%，full-frame 为 60.67%。这个发布阈值在本域
  并没有把真实图误报率控制在可直接部署的水平；
- source-matched secondary 同样呈现两种不同的行为：750 个 local pair
  的 pooled mean forged-real score delta 为
  **-0.025204 [-0.039198, -0.009881]**，strict matched ranking
  accuracy 为 **0.318667 [0.282813, 0.356488]**；750 个 full-frame
  pair 则分别为 **+1.436108 [1.194598, 1.732068]** 和
  **0.872000 [0.834749, 0.905249]**；
- 两个 35-image CUDA smoke 的 raw likelihood、released z-score、
  AI score 和 960 维 descriptor 均 bit-exact；
- fresh model replay 对全部 1,775 张 canonical JPEG 重新执行完整模型，
  四类最大绝对差全部为 0.0，最终 audit 状态为
  `replay_audit_passed`。

最准确的读法是：FSD v1.2 release 对当前三组小面积局部植入没有可用的
whole-image detection 信号，但能更好地区分当前三组 Hunyuan
full-frame conditional edit。这个结论不能外推为“能检测所有整图生成
AIGC”，因为本数据中的 full-frame 样本仍由真实源图条件编辑而来。

本次冻结的 adapter 从 FSD `detector.score()` detection path 的内部
`compute_fsd` 前向捕获并保存整图 descriptor，同时记录 likelihood、
released z-score 和判定；这条路径不产生 dense map、mask 或 bbox。本 run
只计入 **T1 whole-image AIGC detection**；**T2 localization 和 joint
score 均为 N/A**。后文的 source-matched ranking 是对单图分数做的
secondary post-hoc 分析，不是模型原生的 pair 输出。

## 2. 方法、release 与许可边界

### 2.1 本次实际运行的方法

FSD（Forensic Self-Descriptions）的核心思想不是学习一个固定的 RGB
二分类特征，而是先描述每张图自身的局部残差关系，再判断这种
self-description 在真实图统计模型下有多异常。本次冻结的 v1.2 release
路径可以概括为：

```text
canonical JPEG
-> Pillow grayscale decode
-> 8-channel 15x15 forensic residual extractor
-> three image scales
-> per-image float64 constrained least-squares self-description
-> 960-dimensional descriptor
-> 20 residual MLP transforms
-> five-component tied-covariance GMM likelihood
-> released z-score
-> ai_score = -released_z_score
-> strict ai_score > 2.0
```

这种设计试图利用生成或编辑过程破坏自然图像内部统计一致性的现象，而且不依赖
某个具体语义类别。一种与本次结果相符、但未由本 benchmark 做因果验证的
解释是：全图变化更容易改变这种统计，而小面积改动的整图 descriptor 会被
大量未修改内容稀释；模型自身的 resize/center crop 也可能只看到部分甚至
完全看不到植入区域。本次 local 结果表明，这个限制不能靠方法原理上的
吸引力来忽略。

### 2.2 source、权重与 paper/release 差异

| 字段 | 冻结值 |
|---|---|
| official repository | `ductai199x/Forensic-Self-Descriptions-CVPR25` |
| source commit | `50f2eae06efdac2e5a33f407ca9a27a2295133ac` |
| release tag | `v1.2.0` |
| release-tag commit | `5b317a00251988b5ec5a47317f4d82e5bdfd009d` |
| weights bundle SHA-256 | `3c1959f0092fdbe681e41c96f12c1b6d3762e46f21b37b08d4a7e617d1acdfce` |
| repository license | CC-BY-NC-SA-4.0 |
| commercial use established | false |
| share-alike obligation | true |

固定 source commit 比 `v1.2.0` tag 多一个 commit，变动文件为
`fsd/__init__.py`、`fsd/attribution.py` 和 `fsd/weights.py`；
本 adapter 静态检查确认 `score()` 使用的 detection-math 文件没有改变。

实际权重如下：

| 文件 | Bytes | SHA-256 |
|---|---:|---|
| `config.json` | 634 | `7cc34433045adb998762e00de7de25c50f9c1e10dbac1c18899c6c63c4cfafe4` |
| `fre.pt` | 9,861 | `d95b9c50837dbf7b660bbefa20cdaa5db5e59601a9d6544573c10e78e04906bb` |
| `gmm.pt` | 14,786,229 | `0f9fa030a3d5816266d0329fd0fb614b65e322d4bda6d083613c713bfe9bc829` |
| `fsd_transforms.pt` | 42,177,409 | `1e87d792b413101e58d9de71551182a1fab8b879ca6f6ba9780b6adcb9a5a699` |

所有 `.pt` 都通过 `torch.load(..., weights_only=True)` 和 unsafe-global
preflight；没有使用自动下载。

本报告刻意称它为 **FSD official v1.2 inference release**，不称为
CVPR 2025 paper-protocol replication。公开 release 与论文描述存在实质
差异：

- 论文描述的 per-image optimizer 是带 plateau scheduling、最多 10,000
  iteration 的 AdamW；release 使用 float64 KKT constrained
  least-squares；
- 论文给出的 forensic residual neighborhood 是 `M=11 x 11`；release
  实际加载的 FRE 权重是 `8 x 1 x 15 x 15`；
- 论文描述的 scoring chain 是 descriptor 直接进入 GMM；release 在 GMM
  前加入 20 个 `960 -> 128 -> 128 -> 960` residual MLP transform，
  共 5,267,200 个参数；
- release 的 strict operating point 来自 v1.2 `config.json`，
  不能拿论文较早版本的 normalized threshold 替换；
- 仓库没有足够的训练/evaluation 程序让本 benchmark 独立验证
  “release 权重只用真实图训练”，也没有复现论文报告的约 0.960 AUC。

因此，这里的结果只证明冻结推理 release 在本 panel 上的行为。

CC BY-NC-SA 4.0 含 NonCommercial 与 ShareAlike 条款，而且不是 OSI
open-source license。未发现对权重另行授予商用许可的条款；本 benchmark
不建立商用授权，也不构成法律意见。

## 3. Frozen T1 协议

实现入口：
[Balanced250 runner](../eval/opensource/run_fsd_balanced.py) 和
[Balanced250 analyzer](../eval/opensource/analyze_fsd_balanced.py)。
旧版 Mouse runner/analyzer 没有被改写。

### 3.1 图像与模型前向

```text
Pillow.Image.open(path).convert("L")
-> uint8 grayscale, range [0, 255]
-> no EXIF transpose, no ICC conversion
-> 15x15 FRE, trim seven pixels on each side
-> bilinear resize short side to 1024
-> center crop at most 1024x1024
-> 1024 / 512 / 256 three scales
-> float64 KKT constrained least-squares, lambda=1e-5
-> 960-D float64 descriptor
-> official v1.2 transform/GMM scoring
```

Resize 使用 `torch.nn.functional.interpolate(..., mode="bilinear",
align_corners=False, antialias=False)`，尺寸使用 Python `round`。正式
运行是 batch size 1、无 autocast、禁用 TF32、deterministic algorithms，
并设置 `CUBLAS_WORKSPACE_CONFIG=:4096:8`。

`ai_score=-released_z_score`，方向是越高越像 release 所定义的 fake。
它不是目标域校准概率。固定判定严格使用 `ai_score > 2.0`，等于 2.0
不算 positive。

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

每组使用相同的独立 real250 panel。Secondary 只使用
`source_pairs.jsonl` 明确记录的 1,500 个 real/forged endpoint，不从
`task_id` 猜测配对。额外 25 张非 panel real 只用于补齐这些显式 pair。

置信区间是共享 source-content-cluster 的 Poisson bootstrap，1,000 次，
root seed `20260726`。同一 source cluster 的权重会跨 label 和 condition
复用，不把重复源内容错误地当成独立证据。

TPR@5%FPR 的阈值由各 comparison 的 real score 95th percentile
（`method="higher"`）得到，并使用 strict `>`。它只是报告用的 real-only
诊断 operating point，不替代 release 固定阈值。

### 3.3 Full-frame 与 T2 边界

`fullframe_mouse`、`fullframe_cat` 和 `fullframe_trash_can` 是基于真实源图
执行 Hunyuan full-frame conditional-edit 流程的结果，不是脱离真实源图
独立采样的纯文生图。尤其是 trash-can label 只表示样本通过了 single-shot
生成流程，不保证目标物体经过语义 QC 后一定成功植入。

本次冻结并执行的 `detector.score()` detection path 没有 native dense
output；官方包中的其他 attribution functionality 不在本 run 的冻结协议
内，不能事后补作 T2 结果。报告中的 center-crop visibility 是用 canonical
real/local exact-difference GT 独立计算的**输入可见性诊断**，不是模型
预测、定位能力或 T2 指标。Real 与 full-frame 的 T2 均为 N/A。

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
descriptor files:     1775
```

正式 selection ID SHA-256 为
`e4418d86461f889e4a4423f26aab63243e6f63a435a49624881c34979b812e41`。

## 5. Primary：whole-image T1

下表每行都是同一个 real250 panel 对一个 forged250 condition。区间为双侧
95% percentile CI。Confusion 使用 release 的 strict `ai_score > 2.0`。

| Forged condition | AUROC [95% CI] | AP [95% CI] | TPR@5%FPR [95% CI] | Accuracy@2 | Recall@2 | TP / FP / FN / TN |
|---|---:|---:|---:|---:|---:|---:|
| `local_mouse` | 0.490568 [0.476665, 0.503419] | 0.501937 [0.485157, 0.522637] | 0.048000 [0.034628, 0.062055] | 0.502000 | 0.268000 | 67 / 66 / 183 / 184 |
| `local_cat` | 0.480728 [0.463929, 0.496255] | 0.490861 [0.471538, 0.514358] | 0.032000 [0.019001, 0.058334] | 0.492000 | 0.248000 | 62 / 66 / 188 / 184 |
| `local_trash_can` | 0.497176 [0.482449, 0.510767] | 0.500423 [0.484392, 0.522208] | 0.044000 [0.028101, 0.059831] | 0.498000 | 0.260000 | 65 / 66 / 185 / 184 |
| `fullframe_mouse` | 0.701616 [0.667216, 0.734337] | 0.707455 [0.673736, 0.747169] | 0.196000 [0.140562, 0.263415] | 0.674000 | 0.612000 | 153 / 66 / 97 / 184 |
| `fullframe_cat` | 0.689392 [0.655031, 0.723148] | 0.696155 [0.658839, 0.736532] | 0.184000 [0.121763, 0.264230] | 0.668000 | 0.600000 | 150 / 66 / 100 / 184 |
| `fullframe_trash_can` | 0.709216 [0.675846, 0.744677] | 0.705800 [0.668087, 0.749901] | 0.172000 [0.113946, 0.282451] | 0.672000 | 0.608000 | 152 / 66 / 98 / 184 |

条件等权 macro：

| Family | AUROC [95% CI] | AP [95% CI] | TPR@5%FPR [95% CI] | Accuracy@2 | Recall@2 |
|---|---:|---:|---:|---:|---:|
| local 三条件 | 0.489491 [0.477632, 0.500448] | 0.497740 [0.484460, 0.517118] | 0.041333 [0.030018, 0.057278] | 0.497333 | 0.258667 |
| full-frame 三条件 | 0.700075 [0.668743, 0.732202] | 0.703137 [0.668915, 0.743228] | 0.184000 [0.130404, 0.263278] | 0.671333 | 0.606667 |
| 全六条件 | 0.594783 [0.575546, 0.613234] | 0.600439 [0.581241, 0.622066] | 0.112667 [0.084995, 0.156544] | 0.584333 | 0.432667 |

六个 comparison 共享同一批 real score，因此它们的 fixed-threshold false
positive 都是 66/250，而不是六批独立 false positive。正式 1,775 张 cache
中的 strict-threshold positive 诊断计数为：

| Condition | Positive / total |
|---|---:|
| real | 72 / 275 |
| local mouse / cat / trash can | 67 / 250；62 / 250；65 / 250 |
| full-frame mouse / cat / trash can | 153 / 250；150 / 250；152 / 250 |

`ai_score` 不是概率，所以不能把 “2.0” 解读为 200% 或直接拿这些 decision
count 表示校准质量。连续 AUROC/AP 与固定 operating point 回答的是不同
问题。

### 5.1 Domain 诊断

| Family | Domain | AUROC [95% CI] | AP [95% CI] | TPR@5%FPR [95% CI] |
|---|---|---:|---:|---:|
| local | lodging | 0.478010 [0.460795, 0.492956] | 0.495965 [0.484935, 0.514971] | 0.041064 [0.021168, 0.049512] |
| local | restaurant | 0.498000 [0.480195, 0.515334] | 0.501419 [0.477695, 0.537559] | 0.050545 [0.030765, 0.083078] |
| full-frame | lodging | 0.632526 [0.596812, 0.674858] | 0.653076 [0.619351, 0.698926] | 0.186888 [0.094692, 0.245118] |
| full-frame | restaurant | 0.780543 [0.732100, 0.829234] | 0.785133 [0.720711, 0.850033] | 0.338638 [0.115131, 0.562245] |
| all-six | lodging | 0.555268 [0.531902, 0.579564] | 0.574520 [0.555668, 0.600898] | 0.113976 [0.063229, 0.142454] |
| all-six | restaurant | 0.639271 [0.611823, 0.666321] | 0.643276 [0.609866, 0.685231] | 0.194591 [0.074787, 0.313361] |

Full-frame restaurant 的点估计明显高于 lodging，但这里没有为 domain
差异执行专门的 simultaneous hypothesis test。这个表只能作为分域诊断，
不能写成已经证明某种 domain 因果效应。

## 6. Secondary：显式 source-matched 分析

`score delta = forged ai_score - matched real ai_score`，正数表示 forged
被排得更假。Strict matched ranking 把 tie 计为不胜。

| Condition | Pairs | Win / loss / tie | Mean delta [95% CI] | Median delta [95% CI] | Strict ranking [95% CI] |
|---|---:|---:|---:|---:|---:|
| `local_mouse` | 250 | 70 / 140 / 40 | +0.001095 [-0.006976, 0.012303] | -0.001350 [-0.002837, 0.000000] | 0.280000 [0.223919, 0.335938] |
| `local_cat` | 250 | 63 / 182 / 5 | -0.055789 [-0.089942, -0.017348] | -0.021872 [-0.031198, -0.013406] | 0.252000 [0.195636, 0.305972] |
| `local_trash_can` | 250 | 106 / 134 / 10 | -0.020918 [-0.040143, -0.004716] | -0.001363 [-0.005377, 0.000000] | 0.424000 [0.361096, 0.485415] |
| `fullframe_mouse` | 250 | 219 / 31 / 0 | +1.526378 [1.261241, 1.837888] | +1.124254 [0.890841, 1.437148] | 0.876000 [0.834007, 0.916675] |
| `fullframe_cat` | 250 | 214 / 36 / 0 | +1.388333 [1.110313, 1.693741] | +1.084163 [0.852336, 1.393600] | 0.856000 [0.811202, 0.895165] |
| `fullframe_trash_can` | 250 | 221 / 29 / 0 | +1.393611 [1.144892, 1.680686] | +1.068906 [0.827587, 1.319497] | 0.884000 [0.842073, 0.922450] |

Family pooled 结果：

| Family | Pairs | Mean delta [95% CI] | Median delta [95% CI] | Strict ranking [95% CI] |
|---|---:|---:|---:|---:|
| local pooled | 750 | -0.025204 [-0.039198, -0.009881] | -0.004331 [-0.006334, -0.002466] | 0.318667 [0.282813, 0.356488] |
| full-frame pooled | 750 | +1.436108 [1.194598, 1.732068] | +1.093549 [0.890824, 1.334044] | 0.872000 [0.834749, 0.905249] |
| all pairs pooled | 1,500 | +0.705452 [0.581548, 0.848298] | +0.017250 [0.010729, 0.033632] | 0.595333 [0.570219, 0.621108] |

这里的 pooled 不是 condition-macro。由于三个 family condition 都各有 250
pair，local/full-frame 的 mean delta 与 strict ranking 点估计恰好等于
各自等权 macro；median 的 pooled 与 condition-macro 不相同。完整 JSON
同时保留两种聚合口径。

Local pair 中，植入后的分数整体反而略低于 source real，尤其是 cat；
这与 primary 的近随机 AUROC 相互印证。Full-frame 三条件则在 85.6% 至
88.4% 的 matched pair 中把编辑图排得更假，说明其连续相对排序信号不是
由独立 panel 选择偶然造成的。

## 7. Center-crop visibility 诊断

FSD 的有效 native crop 由 FRE border、resize 和 center crop 共同决定。
下面是 local exact-difference GT 在这个 crop 中的可见情况：

| Local condition | Full | Partial | None | Mean visible GT fraction |
|---|---:|---:|---:|---:|
| mouse | 176 | 31 | 43 | 0.757830 |
| cat | 146 | 98 | 6 | 0.831431 |
| trash can | 64 | 176 | 10 | 0.740667 |
| **Pooled** | **386** | **305** | **59** | — |

`none` 表示没有任何 exact-difference GT 正像素的中心落入 effective
native crop，不代表前向在数值上一定完全不受 edit 影响：FRE 支撑域、
边界和插值仍可能引入极小差异。
反过来，`full` 也不意味着整图 descriptor 足以检测小区域。这个表不能
作为定位准确率。

## 8. 可重复性与运行审计

### 8.1 CPU golden preflight

在任何 accelerator 配置之前，runner 对固定 canonical 图
`5f7535f0b957874982b1b080` 执行两次完整 CPU 前向：

| 字段 | 固定值 |
|---|---|
| image SHA-256 | `f90c849192fd53e2e9560192d91b5b37a6162f80c14c862e24d37482784b8078` |
| raw likelihood | `-289.2140144870369` |
| released z-score | `-0.34977523419069584` |
| AI score | `0.34977523419069584` |
| decision | false |
| descriptor array SHA-256 | `96ee62ffc9e5efd54070f1dd182f3d474305d7508218aad84c2ad9d4690478e1` |
| descriptor `.npy` SHA-256 | `233a1645b1d93d6c97e540c7e7c2f022d948ee861eb316d4e71db2a032fca842` |

两次 descriptor 文件 byte-exact，且各自只调用一次完整 FSD computation。
CPU 与 CUDA 允许有平台相关的低位差；确定性主张来自相同 CUDA 环境的 A/B
smoke 和正式 fresh replay。

### 8.2 A/B deterministic smoke

两个 smoke 各取七条件前 5 张，共 35 张。Selection SHA-256 为
`b420bc581386a540b742d917d60d007f0e5522b6cca43fa217797944c40667e5`。
Comparison 状态为 `deterministic_smoke_comparison_passed`：

```text
images compared:                       35
exact computational projection:        true
descriptor file bytes exact:           true
descriptor arrays exact:               true
max raw likelihood difference:         0.0
max released z-score difference:       0.0
max AI-score difference:               0.0
max descriptor difference:             0.0
```

### 8.3 正式 fresh model replay

独立 analyzer 先重新验证 manifest、所有 physical/latest row、1,775 份
descriptor 的路径/bytes/SHA/shape/dtype/finite、canonical JPEG、source、
weights、runtime 和 metrics contract，再使用同一个冻结的
`balanced250_metrics.py` 重新计算统计。它不是第二套独立统计实现。随后
analyzer 重新加载模型，对全部 1,775 张图逐张执行完整前向；结束后再次
验证所有输入和运行证据没有变化。

```text
status:                                 fresh_full_image_replay_passed
images replayed:                        1775
full image forward per input:           true
descriptor-tail-only replay:            false
descriptor comparison:                  numpy.array_equal
max raw likelihood difference:          0.0
max released z-score difference:        0.0
max AI-score difference:                0.0
max descriptor difference:              0.0
final audit status:                      replay_audit_passed
```

正式运行和 replay 都使用 NVIDIA L20Z、CUDA 12.8、PyTorch 2.10.0+cu128、
NumPy 2.4.3、Pillow 12.1.1、SciPy 1.17.1 和 scikit-learn 1.8.0。
正式 manifest 的 `started_at -> completed_at` window 为 429.151 秒；
`started_at` 位于 CPU preflight 之后，所以这不是完整 CLI end-to-end
wall time。逐图官方 `detector.score(path)` 的平均 latency 约 145 ms，
最大 peak CUDA allocation 为 11,274,094,592 bytes。这里的
`latency_ms` 不包含整个 benchmark orchestration，不能直接当成批量服务
吞吐。

本地 Python、venv 和启动器属于本审计的可信计算基。正式命令与 replay
使用空的独立 `PYTHONPYCACHEPREFIX` 并禁用 bytecode 写入，以确保本地
evaluation modules 从冻结 `.py` source 执行；这不等于防御恶意本地操作
系统或解释器。

## 9. 哈希、产物与测试

| 文件 | SHA-256 |
|---|---|
| manifest | `236899b44f842891f8bece1badaa7d09928d3b9a84a5e7b227ee212bc2dae136` |
| expected inputs | `6b5128909eeffdbd88e61af02ca1bd191cb5460f94a23b47c87ebd0172e6d12c` |
| results | `192ac3bd9c8d6fb444223e70f2b2db8d042d1869c6d14b8ab153d4720db1ea41` |
| summary | `35cbc77438a8df1d6e2531b11cb5c1de7ee3c5a28817fc6a5784031b974f9ed0` |
| metrics | `9e03cd7694c30937307490b76b1a0675a9ca9826d6d70f41f0c3dd33a80c79da` |
| independent audit | `4e91704a4e4b1debea2511bf1fc032b97a23b5d63ad587d0f7b553df582b66db` |
| smoke comparison | `f21e2e6c827586c0f20a41781ab7f42698c0673dee5f750c7ca6ff6a59faf503` |

冻结实现：

| 文件 | SHA-256 |
|---|---|
| `run_fsd_balanced.py` | `81aab041396527dd920bd35853bde5ba4b5c3b3b67f0613401d7be7a8f1fb76e` |
| `analyze_fsd_balanced.py` | `6244e83bc577d27a068f3e94e8d6540500d1361390797e75706c0de78ce2a7a1` |
| `test_run_fsd_balanced.py` | `ee191b1d42a17393347363d698ea225cf73f6305e3c3857e0819a03bd8d8b854` |
| `test_analyze_fsd_balanced.py` | `3338deea77cb54616040de9eaddeaa93c8819e39f937940ad924e5fcf9aa946a` |

FSD adapter inventory 还冻结了两个 package initializer、legacy FSD
runner/analyzer、canonical release、Balanced run contract、共享
Balanced250 metrics 和 common helper；完整逐文件哈希在 manifest。

相关 CPU 回归为 129/129 passed；`py_compile`、两个 CLI `--help`、本次
FSD 产物的全部 JSON/JSONL parse 和 `git diff --check` 均通过。

1,775 个 float64 descriptor 位于 gitignored `outputs/`，不会随 git clone
自动取得。每个 `results.jsonl` row 都冻结其相对路径、bytes、文件 SHA、
array SHA、shape 和 dtype；fresh replay audit 依赖本机保留的这些
descriptor artifact。

## 10. 支持与不支持的结论

推荐论文表述：

> The official FSD v1.2 inference release, evaluated on the independent
> Balanced250 panel, was near chance on local insertions
> (condition-macro AUROC 0.489) but transferred better to full-frame
> conditional edits (AUROC 0.700). Explicit source-matched comparisons
> showed the same split: local edits reduced the score on average
> (mean delta -0.025; strict ranking 0.319), whereas full-frame edits
> increased it substantially (mean delta 1.436; strict ranking 0.872).

不要写成：

- “复现了 FSD CVPR 论文的训练或论文 AUC”；
- “FSD 能可靠检测局部植入”；
- “full-frame conditional edit 就是纯文生图或完全合成图”；
- “crop visibility 是 FSD 的定位结果”；
- “`ai_score` 是校准概率”；
- “固定阈值在本域可直接商用部署”；
- “CC-BY-NC-SA-4.0 已提供商用授权”。
