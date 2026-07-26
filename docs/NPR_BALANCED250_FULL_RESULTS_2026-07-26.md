# NPR AIGCDetectBenchmark ProGAN-4class 在 Balanced250 上的正式结果

日期：2026-07-26（UTC）

正式 run：
`npr_aigcdetect_progan4class_author_documented_native_even_trim_balanced250_v1_full1775_20260726`

核心机器证据：
[run manifest](../results/opensource/npr/npr_aigcdetect_progan4class_author_documented_native_even_trim_balanced250_v1_full1775_20260726/manifest.json)、
[逐图结果](../results/opensource/npr/npr_aigcdetect_progan4class_author_documented_native_even_trim_balanced250_v1_full1775_20260726/results.jsonl)、
[coverage summary](../results/opensource/npr/npr_aigcdetect_progan4class_author_documented_native_even_trim_balanced250_v1_full1775_20260726/summary.json)、
[官方概率 Balanced250 metrics](../results/opensource/npr/npr_aigcdetect_progan4class_author_documented_native_even_trim_balanced250_v1_full1775_20260726/balanced250_metrics.json)、
[raw-logit diagnostic](../results/opensource/npr/npr_aigcdetect_progan4class_author_documented_native_even_trim_balanced250_v1_full1775_20260726/npr_raw_logit_diagnostic.json)、
[independent audit](../results/opensource/npr/npr_aigcdetect_progan4class_author_documented_native_even_trim_balanced250_v1_full1775_20260726/independent_audit.json)、
[r2 双 smoke comparison](../results/opensource/npr/npr_smoke_comparison_v2_ff240d16627de52a292d.json)。

> **最终审计状态：** `replay_audit_passed`。独立 analyzer 已在记录的
> CUDA runtime 上完成 1,775/1,775 个 persisted feature 的 exact
> feature-to-head/sigmoid replay，并从 canonical JPEG 对 1,775/1,775
> 张图重新执行 fresh full-model forward。Feature 使用
> `numpy.array_equal` 比较，feature、raw logit 和 probability 的最大绝对
> 差均为 `0.0`。最终 audit SHA-256 为
> `4a54c41c83091c694582ef26ee7a5264101407c858df5540a54cb94d2e0a568f`。

## 1. 结论摘要

NPR 的公开 AIGCDetectBenchmark ProGAN-4class checkpoint 已按冻结的
Balanced250 whole-image T1 协议完成正式推理：

- score cache 覆盖 1,775/1,775 张图，全部成功；error、missing 和
  superseded attempt 均为 0；
- 官方主分数是模型 float32 logit 的 float32 sigmoid，固定判定严格使用
  `probability > 0.5`。本次所有 1,775 个 raw logit 都小于 0，最大值仅
  `-4.562627`，所以模型把 **0/1,775** 张图判为 fake；
- 三种 local insertion 的官方 probability condition-macro AUROC 为
  **0.481371 [0.469650, 0.493430]**，AP 为
  **0.488772 [0.480151, 0.502709]**。它没有显示出可用的局部植入
  whole-image 检测能力；
- 三种 full-frame conditional edit 的官方 macro AUROC 为
  **0.327165 [0.298079, 0.358372]**，AP 为
  **0.468377 [0.455728, 0.482902]**。这不是“较弱但方向正确”，而是生成
  条件编辑图整体被排得比真实图更 real；
- 六条件等权官方 macro AUROC 为
  **0.404268 [0.387278, 0.422935]**，固定阈值下六组 recall 均为 0；
- 官方 probability 在 **1,296/1,775（73.01%）** 图像上饱和为精确 0，
  全集只有 478 个不同的 probability。预注册 raw-logit diagnostic 保留
  了这些饱和样本之间的有限排序信息，但结果更差：local macro AUROC
  **0.477813 [0.463990, 0.491264]**，full-frame macro AUROC
  **0.202101 [0.175426, 0.229436]**，六条件 macro AUROC
  **0.339957 [0.323602, 0.356547]**；
- source-matched secondary 也不能把它解释成一个可部署 detector。
  官方 probability 的 local pooled strict ranking 为
  **0.197333 [0.163950, 0.231443]**；raw logit 的 local pooled strict
  ranking 虽为 **0.526667 [0.488256, 0.563467]**，但 pooled mean delta
  是显著负向的 **-1.233376 [-1.702644, -0.822723]**，三种对象的行为
  彼此不一致。full-frame raw strict ranking 仅
  **0.050667 [0.028961, 0.075109]**；
- parity trim 只会移除奇数尺寸图像最后一行和/或最后一列。750 张 local
  图中，exact-difference GT 有 700 张完全保留、50 张仅边缘部分被 trim、
  0 张完全不可见。因此 local 失败不能归因于像 CenterCrop 那样完全错过
  大量植入区域；
- 两个 35-image r2 CUDA smoke 的预处理计算投影、512 维 feature 文件、
  feature array、raw logit 和 probability 均 bit-exact，最大差全部为
  0.0；
- 独立 analyzer 对全部 1,775 张图完成 feature-to-head replay 和 fresh
  full-model replay；feature、raw logit 和 probability 的最大绝对差均为
  0.0，最终状态为 `replay_audit_passed`；
- 最终扩展回归为 **227 passed in 80.27s**。

最准确的读法是：这个 NPR release/profile 不仅没有迁移到 CLAIMFORGE 的
三种小面积局部植入，其对当前三种 full-frame conditional edit 还产生了
强烈的反向 whole-image 排序。sigmoid 饱和会把大量有限但很负的 logits
压成相同的 0，从而把部分反向排序拉回接近 0.5；raw logit 揭示的
full-frame 反向效应更强。

这个结果不能外推为“NPR 对所有纯整图生成 AIGC 都失败”。Balanced250 的
`fullframe_*` 是从真实源图出发的 Hunyuan full-frame conditional edit，
不是脱离真实源图独立采样的纯文生图。NPR 只输出一个整图分数，没有 native
dense map、mask 或 bbox；本 run 只计入 **T1 whole-image AIGC
detection**，**T2 localization 和 joint score 均为 N/A**。

## 2. 方法、source、checkpoint 与许可边界

### 2.1 NPR 的原理与本次实际前向

NPR 的出发点是：GAN 或 diffusion decoder 的上采样会在相邻像素之间留下
结构性依赖。与直接学习内容语义或某个生成器的视觉风格相比，这类
neighboring-pixel relationship 理论上更可能跨生成器迁移。

本次冻结的实际计算为：

```text
canonical JPEG
-> Pillow.Image.open(...).convert("RGB")
-> float32 ToTensor
-> ImageNet channel normalization
-> 若高度为奇数，删除最后一行
-> 若宽度为奇数，删除最后一列
-> x_half = nearest_downsample(x, scale_factor=0.5)
-> x_reconstructed = nearest_upsample(x_half, scale_factor=2)
-> NPR residual = (x - x_reconstructed) * (2/3)
-> truncated ResNet-50 stem + layer1 + layer2
-> adaptive global average pool
-> 512-dimensional float32 feature
-> fc1: Linear(512, 1)
-> float32 sigmoid
-> ai_score = probability
-> strict probability > 0.5
```

这个设计为何在论文的跨生成器 benchmark 中可能强、在这里却失败并不矛盾。
NPR 最终仍对整张 native-resolution 图做 global average pooling。小面积
植入信号会与大量未修改的相机/JPEG 内容混合，而 ProGAN-4class checkpoint
的训练对比与 CLAIMFORGE 的局部编辑、真实源图条件编辑都存在显著分布差异。
对 full-frame conditional edit，本次模型不是“没反应”，而是沿训练所得
方向的反方向强烈变化。

`ai_score` 是 released classifier 的 sigmoid 输出，不是经过本域校准的
真实世界概率。AUROC/AP、real-only 5% FPR 诊断点和固定 0.5 threshold
回答的是不同问题。

### 2.2 冻结 provenance 与资产

方法来自 CVPR 2024 论文
[Rethinking the Up-Sampling Operations in CNN-based Generative Network for Generalizable Deepfake Detection](https://openaccess.thecvf.com/content/CVPR2024/html/Tan_Rethinking_the_Up-Sampling_Operations_in_CNN-based_Generative_Network_for_Generalizable_CVPR_2024_paper.html)。

| 字段 | 冻结值 |
|---|---|
| official repository | `chuangchuangtan/NPR-DeepfakeDetection` |
| source commit | `781ced3f7ca2cdc69ec9dd4ef27e8d0b3c07752a` |
| author HF Space commit | `522a9f1020f7454d486f28a0d5c148ec37919b32` |
| checkpoint | `model_epoch_last_3090.pth` |
| checkpoint introduced commit | `68338a07847e891534f3d0b0a0e25bb137b684f7` |
| checkpoint bytes | 5,842,385 |
| checkpoint SHA-256 | `b67a91555ce786a6d0463ff0cb2b0b874d1c3f971b0e3febd2ae5618a80f7e8a` |
| checkpoint entries / elements | 146 / 1,447,897 |
| trainable parameters | 1,437,761 |
| checkpoint schema SHA-256 | `e60d79370c937aede4ff54ff57663207b6282f566c28caf56f6afd924af530d6` |
| asset bundle SHA-256 | `9c19c48e4a3a42f4628b89445e2a39fe564802efbfb8c93854aeae55dfa81b66` |
| inference mode | `eval()` |
| feature dimension | 512 |

Checkpoint 通过 unsafe-global preflight，使用
`torch.load(..., weights_only=True, map_location="cpu")` 和 strict
state-dict match 加载。运行中不联网，也不重新分发 checkpoint。

没有使用的两个候选资产为：

| Excluded asset | SHA-256 | 原因 |
|---|---|---|
| `NPR.pth` | `3939297e9399e0b992f87211610769d87d899de50d56da0204d6cbda2d483a53` | paper-era nested model/optimizer/step snapshot，带 `module.` keys，不是当前 AIGCDetectBenchmark/HF checkpoint |
| `NPR_GenImage_sdv4.pth` | `9bc961e7d643581aa0ea879cbd322dcc2e543877568a43d2f6cdb92906379015` | 单独训练的 GenImage SDv1.4 checkpoint |

### 2.3 Odd-dimension compatibility completion

这里不能把 profile 简称为“pinned GitHub live preprocessing 原样复现”：

- pinned GitHub `networks/resnet.py` 的 live forward **没有执行** odd-size
  trim；相关代码被注释；
- official README 明确要求 AIGCDetectBenchmark 使用者加入同一个
  final-row/final-column trim，否则奇数尺寸会使 0.5× downsample 和 2×
  upsample 后的 tensor 无法与原 tensor 相减；
- pinned author HF Space 也执行这个 trim，进一步佐证作者预期；
- 但 HF Space 的 `app.py` 没有调用 `model.eval()`，会让 BatchNorm 留在
  batch-one train mode、更新 running buffers，并产生输入顺序相关行为。
  因此它只用来佐证 checkpoint 与 native-size/trim 预处理，不作为可执行
  reference。

本报告将该 profile 精确称为
`author_documented_aigcdetect_native_even_trim_completion`，即
**author-documented、deployment-corroborated compatibility completion**。
它不是 pinned GitHub live forward as-is，也不声称复现论文训练/测试协议。

Balanced250 中 odd width 为 81 张、odd height 为 376 张、两者同时为奇数
为 13 张，union 为 **444/1,775**。不做 completion 时，这些输入无法完成
原始 residual subtraction。

### 2.4 License 与商用边界

Pinned GitHub repository 根目录没有 `LICENSE`、`COPYING` 或 `NOTICE`：

| 组件 | 记录 |
|---|---|
| GitHub code | license not stated；OSI open-source license 未建立 |
| GitHub checkpoint | license not stated；commercial clearance 未建立 |
| HF Space metadata | Apache-2.0，但只覆盖 Space repository |
| 整体商用授权 | not established |

HF Space 的 Apache-2.0 metadata 不会自动重新许可 upstream GitHub code 或
checkpoint。NPR 是公开可取得的方法和权重，但不能据此写成“已确认可商用的
OSI 开源模型”。本报告不是法律意见。

## 3. Frozen Balanced250 T1 协议

实现入口：
[Balanced250 runner](../eval/opensource/run_npr_balanced.py) 和
[Balanced250 analyzer](../eval/opensource/analyze_npr_balanced.py)。
旧版 Mouse runner、analyzer 和结果没有被改写。

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
正式 selection ID SHA-256 为
`e4418d86461f889e4a4423f26aab63243e6f63a435a49624881c34979b812e41`。

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
`source_pairs.jsonl` 显式记录的 1,500 个 real/forged endpoint，不从
`task_id` 猜测配对。额外 25 张非 panel real 只用于补齐这些 source pairs。

置信区间使用共享 source-content-cluster Poisson bootstrap，1,000 次，
root seed `20260726`。同一 source cluster 的权重跨 label 和 condition
复用，区间为双侧 95% percentile CI。

TPR@5%FPR 的阈值来自各 comparison 的 real score 95th percentile，
`method="higher"`，判定严格使用 `>`。它只是报告用 real-only 诊断点，
不替代 released 0.5 threshold。

`fullframe_mouse`、`fullframe_cat` 和 `fullframe_trash_can` 是真实源图上的
Hunyuan full-frame conditional edit，不等于纯文生图或完全独立的
whole-image generation；trash-can label 也不保证生成物经过额外语义 QC。

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
physical result rows:                         1775
latest result rows:                           1775
superseded attempts:                          0
coverage fraction:                            1.0
success fraction:                             1.0
persisted 512-D feature files:                1775
runner same-device feature/head/sigmoid replay: 1775
```

最后一项是 runner 在正式运行/收尾路径上的同记录设备精确 replay；独立
analyzer 随后也完成了 feature-to-head 和 fresh full-model replay，见
第 9.4 节。

## 5. Primary：官方 float32 sigmoid probability

每行使用同一个 real250 panel 对一个 forged250 condition。括号是
1,000-resample shared-cluster bootstrap 95% percentile CI。Confusion
使用 released strict `probability > 0.5`。

| Forged condition | AUROC [95% CI] | AP [95% CI] | TPR@5%FPR [95% CI] | Accuracy@0.5 | Recall@0.5 | TP / FP / FN / TN |
|---|---:|---:|---:|---:|---:|---:|
| `local_mouse` | 0.502664 [0.490264, 0.514609] | 0.501164 [0.491218, 0.515914] | 0.048000 [0.034184, 0.057780] | 0.500000 | 0.000000 | 0 / 0 / 250 / 250 |
| `local_cat` | 0.482872 [0.466000, 0.499460] | 0.492742 [0.478688, 0.510822] | 0.044000 [0.025424, 0.059579] | 0.500000 | 0.000000 | 0 / 0 / 250 / 250 |
| `local_trash_can` | 0.458576 [0.440214, 0.476946] | 0.472411 [0.457082, 0.489976] | 0.020000 [0.008438, 0.040816] | 0.500000 | 0.000000 | 0 / 0 / 250 / 250 |
| `fullframe_mouse` | 0.327424 [0.298080, 0.360098] | 0.468914 [0.454668, 0.485659] | 0.000000 [0.000000, 0.000000] | 0.500000 | 0.000000 | 0 / 0 / 250 / 250 |
| `fullframe_cat` | 0.335464 [0.306092, 0.367321] | 0.463843 [0.449563, 0.479942] | 0.000000 [0.000000, 0.008299] | 0.500000 | 0.000000 | 0 / 0 / 250 / 250 |
| `fullframe_trash_can` | 0.318608 [0.290139, 0.349770] | 0.472375 [0.457106, 0.489600] | 0.000000 [0.000000, 0.000000] | 0.500000 | 0.000000 | 0 / 0 / 250 / 250 |

条件等权 macro：

| Family | AUROC [95% CI] | AP [95% CI] | TPR@5%FPR [95% CI] | Accuracy@0.5 | Recall@0.5 |
|---|---:|---:|---:|---:|---:|
| local 三条件 | 0.481371 [0.469650, 0.493430] | 0.488772 [0.480151, 0.502709] | 0.037333 [0.028244, 0.048486] | 0.500000 | 0.000000 |
| full-frame 三条件 | 0.327165 [0.298079, 0.358372] | 0.468377 [0.455728, 0.482902] | 0.000000 [0.000000, 0.002813] | 0.500000 | 0.000000 |
| 全六条件 | 0.404268 [0.387278, 0.422935] | 0.478575 [0.469614, 0.491271] | 0.018667 [0.014122, 0.024409] | 0.500000 | 0.000000 |

正式 cache 中七个 condition 的 strict-threshold positive 都是 0，包括
0/275 real、三组 0/250 local 和三组 0/250 full-frame。六组 comparison
共享 real250 panel，因此 FP=0 是同一批 real score 的重复使用，不是六批
独立的零误报实验。

Real-only 5% FPR probability threshold 为
`9.027106e-18 [5.423008e-23, 1.121791e-13]`，actual FPR 为
`0.048000 [0.038134, 0.049793]`。这个极小 threshold 反映分数严重靠近
0，不说明模型经过了良好校准。

### 5.1 官方概率的 domain 诊断

| Family | Domain | AUROC [95% CI] | AP [95% CI] | TPR@5%FPR [95% CI] |
|---|---|---:|---:|---:|
| local | lodging | 0.479352 [0.462167, 0.495995] | 0.509670 [0.499344, 0.528394] | 0.041082 [0.031751, 0.063134] |
| local | restaurant | 0.486318 [0.469141, 0.501750] | 0.472753 [0.461507, 0.494177] | 0.026760 [0.015994, 0.049767] |
| full-frame | lodging | 0.333577 [0.291607, 0.377205] | 0.486762 [0.472477, 0.505907] | 0.000000 [0.000000, 0.007879] |
| full-frame | restaurant | 0.321296 [0.279824, 0.359779] | 0.450349 [0.433922, 0.473044] | 0.000000 [0.000000, 0.005559] |
| all-six | lodging | 0.406465 [0.380188, 0.433395] | 0.498216 [0.487441, 0.515290] | 0.020541 [0.016339, 0.033548] |
| all-six | restaurant | 0.403807 [0.378333, 0.427391] | 0.461551 [0.450303, 0.481645] | 0.013380 [0.007997, 0.025645] |

两个 domain 的 full-frame AUROC 都显著低于 0.5；没有证据表明反向效应只由
单一 domain 驱动。这里没有执行 domain 差异的 simultaneous hypothesis
test，因此不能把点估计差异解释为因果 domain effect。

## 6. Pre-registered raw-logit numerical diagnostic

Raw logit 使用与官方结果完全相同的 1,775 个 sample ID、panel、
source pairs、bootstrap dependency、1,000 iterations 和 seed；两个
metrics contract 的唯一差异是 score spec。它的作用只是恢复 sigmoid
数值饱和前的排序，**绝不替代官方 probability、0.5 threshold 或正式
decision**。

Raw diagnostic 的 fixed threshold 是 `raw_logit > 0`。由于本次所有 logits
均为负，它与官方 0.5 decision 在 1,775/1,775 张图上相同。

| Forged condition | Raw AUROC [95% CI] | Raw AP [95% CI] | Raw TPR@5%FPR [95% CI] | Recall@0 |
|---|---:|---:|---:|---:|
| `local_mouse` | 0.499360 [0.485089, 0.512977] | 0.501483 [0.490230, 0.517321] | 0.048000 [0.034184, 0.057780] | 0.000000 |
| `local_cat` | 0.480016 [0.462217, 0.498479] | 0.487967 [0.473388, 0.508841] | 0.044000 [0.025424, 0.059579] | 0.000000 |
| `local_trash_can` | 0.454064 [0.435756, 0.470185] | 0.461893 [0.445550, 0.481767] | 0.020000 [0.008438, 0.040816] | 0.000000 |
| `fullframe_mouse` | 0.199360 [0.171079, 0.226147] | 0.344565 [0.332301, 0.361263] | 0.000000 [0.000000, 0.000000] | 0.000000 |
| `fullframe_cat` | 0.219424 [0.190703, 0.247951] | 0.350628 [0.337842, 0.369197] | 0.000000 [0.000000, 0.008299] | 0.000000 |
| `fullframe_trash_can` | 0.187520 [0.160731, 0.215512] | 0.340519 [0.328668, 0.355329] | 0.000000 [0.000000, 0.000000] | 0.000000 |

Raw condition-macro：

| Family | Raw AUROC [95% CI] | Raw AP [95% CI] | Raw TPR@5%FPR [95% CI] |
|---|---:|---:|---:|
| local 三条件 | 0.477813 [0.463990, 0.491264] | 0.483781 [0.474000, 0.499446] | 0.037333 [0.028244, 0.048486] |
| full-frame 三条件 | 0.202101 [0.175426, 0.229436] | 0.345237 [0.334772, 0.360673] | 0.000000 [0.000000, 0.002813] |
| 全六条件 | 0.339957 [0.323602, 0.356547] | 0.414509 [0.406331, 0.428059] | 0.018667 [0.014122, 0.024409] |

Raw real-only 5% FPR threshold 为
`-39.246300 [-51.268806, -29.818680]`，actual FPR 同样是
`0.048000 [0.038134, 0.049793]`。顶部 tail 没有发生 sigmoid-to-zero
collapse，所以该 operating point 的 TPR 与 probability 表相同；这不代表
全局 AUROC/AP 也相同。

### 6.1 Probability saturation

| Condition | Exact-zero probability | Total | Zero rate | Raw-logit min | Raw-logit max |
|---|---:|---:|---:|---:|---:|
| `real` | 161 | 275 | 58.55% | -183.728180 | -4.562627 |
| `local_mouse` | 141 | 250 | 56.40% | -183.583481 | -4.625526 |
| `local_cat` | 151 | 250 | 60.40% | -178.757599 | -4.770916 |
| `local_trash_can` | 160 | 250 | 64.00% | -178.514740 | -19.915743 |
| `fullframe_mouse` | 228 | 250 | 91.20% | -235.381882 | -54.375301 |
| `fullframe_cat` | 223 | 250 | 89.20% | -238.436188 | -46.322002 |
| `fullframe_trash_can` | 232 | 250 | 92.80% | -237.998611 | -52.030109 |
| **Total** | **1,296** | **1,775** | **73.01%** | **-238.436188** | **-4.562627** |

全体 probability minimum 为 0，maximum 为 `0.0103268521`，没有 exact 1
或 exact 0.5，只有 478 个 unique values。Raw-positive/probability-not-
above-half 和 raw-nonpositive/probability-above-half 的 boundary
disagreement 都是 0。

Sigmoid 的 monotonicity 只在未饱和的有限数值上保持严格排序。大量精确 0
会制造 ties，并把非常差的 raw full-frame AUROC 往 0.5 方向拉。因此
official AUROC 0.327 和 raw AUROC 0.202 不是互相冲突，也不能挑其中较高的
一个当作“更公平”的主结果。

### 6.2 Raw-logit domain 诊断

| Family | Domain | Raw AUROC [95% CI] | Raw AP [95% CI] | Raw TPR@5%FPR [95% CI] |
|---|---|---:|---:|---:|
| local | lodging | 0.480191 [0.460995, 0.497855] | 0.504483 [0.493945, 0.525683] | 0.041082 [0.031751, 0.063134] |
| local | restaurant | 0.480468 [0.461223, 0.498066] | 0.470049 [0.458062, 0.493798] | 0.026760 [0.015994, 0.049767] |
| full-frame | lodging | 0.180437 [0.140322, 0.214590] | 0.348102 [0.334039, 0.370138] | 0.000000 [0.000000, 0.007879] |
| full-frame | restaurant | 0.216780 [0.178632, 0.255429] | 0.343254 [0.330300, 0.366421] | 0.000000 [0.000000, 0.005559] |
| all-six | lodging | 0.330314 [0.303604, 0.353352] | 0.426293 [0.415557, 0.446748] | 0.020541 [0.016339, 0.033548] |
| all-six | restaurant | 0.348624 [0.325867, 0.371158] | 0.406652 [0.396996, 0.427206] | 0.013380 [0.007997, 0.025645] |

## 7. Secondary：显式 source-matched 分析

`score delta = forged score - matched real score`，正数表示 forged 被排得
更 fake。Strict matched ranking 把 tie 计为不胜。它是对单图 score 的
secondary post-hoc 比较，不是 NPR 原生的 pair inference，也不能在没有
matched-real reference 的部署中直接使用。

### 7.1 官方 probability pairs

| Condition | Pairs | Win / loss / tie | Mean delta [95% CI] | Median delta [95% CI] | Strict ranking [95% CI] |
|---|---:|---:|---:|---:|---:|
| `local_mouse` | 250 | 57 / 52 / 141 | -2.493e-6 [-7.958e-6, +1.076e-9] | 0 [0, 0] | 0.228000 [0.177684, 0.278990] |
| `local_cat` | 250 | 70 / 34 / 146 | -7.700e-6 [-2.437e-5, +5.691e-9] | 0 [0, 0] | 0.280000 [0.224000, 0.336039] |
| `local_trash_can` | 250 | 21 / 83 / 146 | -4.131e-5 [-1.296e-4, -2.453e-12] | 0 [0, 0] | 0.084000 [0.051580, 0.121345] |
| `fullframe_mouse` | 250 | 10 / 98 / 142 | -4.131e-5 [-1.308e-4, -5.550e-13] | 0 [0, 0] | 0.040000 [0.018116, 0.066674] |
| `fullframe_cat` | 250 | 14 / 94 / 142 | -4.131e-5 [-1.296e-4, -4.995e-13] | 0 [0, 0] | 0.056000 [0.029409, 0.085113] |
| `fullframe_trash_can` | 250 | 6 / 98 / 146 | -4.131e-5 [-1.286e-4, -2.921e-12] | 0 [0, 0] | 0.024000 [0.007843, 0.044645] |

Family pooled：

| Family | Pairs | Win / loss / tie | Mean delta [95% CI] | Median delta [95% CI] | Strict ranking [95% CI] |
|---|---:|---:|---:|---:|---:|
| local pooled | 750 | 148 / 169 / 433 | -1.717e-5 [-5.388e-5, -6.487e-13] | 0 [0, 0] | 0.197333 [0.163950, 0.231443] |
| full-frame pooled | 750 | 30 / 290 / 430 | -4.131e-5 [-1.289e-4, -2.059e-12] | 0 [0, 0] | 0.040000 [0.019732, 0.062504] |
| all pairs pooled | 1,500 | 178 / 459 / 863 | -2.924e-5 [-9.145e-5, -1.358e-12] | 0 [0, 0] | 0.118667 [0.097406, 0.141537] |

863/1,500 ties 主要来自 float32 sigmoid saturation，不能被解释成模型在
语义上认为 pair 完全相同。

### 7.2 Raw-logit pairs

| Condition | Pairs | Win / loss / tie | Mean raw delta [95% CI] | Median raw delta [95% CI] | Strict ranking [95% CI] |
|---|---:|---:|---:|---:|---:|
| `local_mouse` | 250 | 148 / 102 / 0 | +0.048130 [+0.013168, +0.083889] | +0.026028 [+0.008507, +0.055504] | 0.592000 [0.531734, 0.650000] |
| `local_cat` | 250 | 180 / 70 / 0 | -0.322118 [-0.978367, +0.267715] | +0.379734 [+0.261873, +0.483429] | 0.720000 [0.665207, 0.774202] |
| `local_trash_can` | 250 | 67 / 183 / 0 | -3.426141 [-4.440115, -2.560441] | -1.094437 [-1.637253, -0.863499] | 0.268000 [0.213946, 0.324582] |
| `fullframe_mouse` | 250 | 11 / 239 / 0 | -42.643759 [-46.736232, -39.270010] | -39.526089 [-42.767097, -35.319193] | 0.044000 [0.021457, 0.070797] |
| `fullframe_cat` | 250 | 16 / 234 / 0 | -37.233667 [-40.505727, -33.951842] | -34.102760 [-37.814255, -30.933448] | 0.064000 [0.036034, 0.094424] |
| `fullframe_trash_can` | 250 | 11 / 239 / 0 | -42.866082 [-46.250402, -39.658034] | -38.171795 [-41.938274, -35.551897] | 0.044000 [0.020074, 0.070428] |

Family pooled：

| Family | Pairs | Win / loss / tie | Mean raw delta [95% CI] | Median raw delta [95% CI] | Strict ranking [95% CI] |
|---|---:|---:|---:|---:|---:|
| local pooled | 750 | 395 / 355 / 0 | -1.233376 [-1.702644, -0.822723] | +0.022755 [-0.014482, +0.059731] | 0.526667 [0.488256, 0.563467] |
| full-frame pooled | 750 | 38 / 712 / 0 | -40.914503 [-44.190921, -37.693787] | -37.450861 [-41.103095, -33.918793] | 0.050667 [0.028961, 0.075109] |
| all pairs pooled | 1,500 | 433 / 1,067 / 0 | -21.073940 [-22.874917, -19.387475] | -8.684723 [-11.372208, -5.956421] | 0.288667 [0.264375, 0.313255] |

Local raw pair 结果说明对象类别之间存在强异质性：mouse 有小幅正向变化，
cat 多数 pair 正向但少数大负值使 mean 为负，trash-can 则明显负向。把三组
只汇总为 52.7% strict ranking 会掩盖这种方向不一致，而且其 CI 跨 0.5。
更重要的是，selection-unpaired raw local macro AUROC 只有 0.478；没有
matched reference 的单图 detector 不能利用这种 pair-specific 差值。

## 8. Native-even trim visibility 与 T2 边界

Local exact-difference GT 在最终一行/一列 parity trim 后的输入可见情况：

| Local condition | Full | Partial | None |
|---|---:|---:|---:|
| mouse | 250 | 0 | 0 |
| cat | 239 | 11 | 0 |
| trash can | 211 | 39 | 0 |
| **Pooled** | **700** | **50** | **0** |

`full` 表示 exact-difference GT 全部保留；`partial` 只表示部分 GT 像素落在
被删除的最后一行/列；`none` 为 0。这个诊断支持“模型实际看到了 local edit
所在区域”的输入级结论，但不能证明模型利用了它。

Visibility 是从 canonical real/local exact-difference GT 独立计算的输入
几何证据，不是 NPR 输出。Real 的 GT 是 all-zero，full-frame 没有同义的
局部 mask，二者 visibility 均为 not applicable。NPR 没有 native dense
output，因此不能用这个表补作 T2、pixel AUROC、IoU、mask 或 joint score。

## 9. 可重复性、runtime 与最终审计

### 9.1 CPU dual-odd golden preflight

在任何 accelerator 配置之前，runner 对同时具有奇数宽高的固定 canonical
图 `7aeae0f17050bf766257b47d` 执行两次完整 CPU forward：

| 字段 | 固定值 |
|---|---|
| input size | 1285×1137 |
| effective size | 1284×1136 |
| image SHA-256 | `21bfef64a1863cda43e122846c6cde1c40d97adcc33f813409f8204732e5093b` |
| decoded RGB SHA-256 | `51e0da2279f209abe1b8349f0d215df0b2a989291c6ab02026b8b8b40e76a3e3` |
| normalized tensor SHA-256 | `1ff28c34fdfdf89c8a684d99b71fbe78bf3e582ab1c5b0c9192b4ff18fd04640` |
| NPR residual SHA-256 | `84613a075de69e102fa62b71d99e7cca5f638774ae8b667928f91dcc6e8e9715` |
| feature array SHA-256 | `521a7bfbd00dbfee27d21271649c0d24b100842bd64b23f44dc422e9acddbbd4` |
| feature `.npy` SHA-256 | `6c3fc67b81c69bac75159dcfffd56d127393546a93fefca5d664c34c355bb8a4` |
| raw logit | `-84.44386291503906` |
| probability / AI score | `2.120783389925294e-37` |
| two-forward result | byte-exact |

Preflight 期间 CUDA 未初始化。这个 golden 同时验证 Pillow decode、
normalization、双 odd trim、NPR residual、完整网络、feature capture、
linear head 和 float32 sigmoid。

### 9.2 r2 A/B deterministic smoke

两个 smoke 各取七条件前 5 张，共 35 张。Selection SHA-256 为
`b420bc581386a540b742d917d60d007f0e5522b6cca43fa217797944c40667e5`。
Comparison 状态为 `deterministic_smoke_comparison_passed`：

```text
images compared:                    35
exact computational projection:     true
immutable runtime/config exact:     true
feature file bytes exact:           true
feature arrays exact:               true
max raw-logit difference:           0.0
max probability difference:         0.0
max feature difference:             0.0
```

两次 smoke 的 feature inventory SHA-256 都是
`2fa3195b206d060f65ae8065f86fbdc5dd32bb023f8f32ce267c3d87f8fd8fee`。

更早的 r1 smoke 在 analyzer 对 visibility claim 执行 fail-closed 检查后
暴露出 source/evidence contract 需要修正，因此作废且不提交。修正后的
source 使用新的 immutable r2 run IDs 从零执行；本报告只把 r2 comparison
视为正式 smoke 证据。

### 9.3 正式 runtime

正式环境为 NVIDIA L20Z、CUDA 12.8、PyTorch 2.8.0+cu128、
torchvision 0.23.0+cu128、NumPy 2.2.6、Pillow 11.1.0、SciPy 1.17.1
和 scikit-learn 1.8.0。Python 3.12.3 来自隔离环境
`/root/.cache/claimforge/venvs/npr-balanced-torch2.8.0`，不包含 system
site packages。

运行固定 seed 100、batch size 1、float32、无 autocast、无 grad、关闭
TF32、启用 deterministic algorithms，并设置
`CUBLAS_WORKSPACE_CONFIG=:4096:8`、`PYTHONHASHSEED=100`、
`PYTHONDONTWRITEBYTECODE=1` 和初始为空的隔离
`PYTHONPYCACHEPREFIX`。Pinned external source 从已验证 UTF-8 bytes
直接 `compile`，不执行第三方目录中的 pyc。

正式 manifest 的 `started_at -> completed_at` window 约 1,497.8 秒。
逐图 model-forward latency 平均 19.236 ms、中位数 16.166 ms、empirical
P95（`method="higher"`）38.579 ms；最大 peak CUDA allocation 为
744,489,984 bytes。单独记录的 decode、hash、normalization、residual 和
几何证据 preprocessing 平均 243.248 ms/image。这些逐图计时不等于完整
CLI wall time。

### 9.4 独立 fresh full-model replay

最终 analyzer audit 验证了：

- 1,775 个 physical/latest rows 和完整 coverage；
- 每个 feature 的 path、bytes、file SHA、array SHA、shape、dtype 与
  finite；
- pinned source/HF/checkpoint/runtime/preprocess/dataset contracts；
- 官方 probability metrics 与 raw-logit diagnostic 使用相同 selection、
  panel、pairs 和 bootstrap；
- local visibility、T1-only method boundary 和显式 T2 rejection。

随后 analyzer 在与 manifest 完全一致的记录 runtime 上加载 pinned source
和 checkpoint，先对全部 persisted feature 执行独立 fc1/sigmoid replay，
再从 canonical JPEG 重新 decode、preprocess 并执行完整 NPR 网络。最终机器
证据为：

```text
independent feature-to-head replay:    passed
persisted features replayed:           1775
feature-to-fc1 comparison:             exact_float32_scalar
sigmoid comparison:                    exact_float32_scalar
recorded runtime exact match:          true

fresh full-model replay:               passed
fresh full-model images replayed:      1775
full-image forward per input:          true
feature-tail-only replay:              false
fresh feature comparison:              numpy.array_equal
max fresh feature difference:          0.0
max fresh raw-logit difference:        0.0
max fresh probability difference:      0.0
cross-device sigmoid gate used:        false
final audit status:                     replay_audit_passed
```

Fresh replay 的 `ai_score` 是官方 probability 的强制同值 alias；analyzer
同时验证结果 row 的 score aliases 和 decision contract。独立 replay
刻意不使用 host-CPU sigmoid 作为跨设备 exact gate，而是在记录的
`cuda:0` 上执行 exact float32 scalar comparison。

## 10. 哈希、产物与重跑

当前已稳定的正式产物：

| 文件 | SHA-256 |
|---|---|
| manifest | `8525485f4ddfa8cbe7abb892d018b30f2bd50d85988214c14314cee2ff9c1198` |
| expected inputs | `6b5128909eeffdbd88e61af02ca1bd191cb5460f94a23b47c87ebd0172e6d12c` |
| results | `d104183618a6b056a80bf3c97fe969a934234bd2c9b4418d280e0dc0d6a94289` |
| summary | `af18ae075f9f80f31a1a3b36a86b66c2a8682b1d0ed0493dfc422b35e67c1142` |
| official metrics | `c76f4f93a344d84b7c45a83f1fda407154f59678f9c79271c7296546028ad7a9` |
| raw-logit diagnostic | `ba42a220c8440ec8235f7a27cc2e127ccfe06314cbc9b6119f7bd7f8b9816859` |
| feature inventory | `6b2961e6e7340caee9cf9761ad4c606228b1e80c8626ce0db40e85c56d27a10c` |
| smoke comparison | `67d81251cebe0dac7635e35a805cf7123afb2b153ae903db0c286451a2dcf169` |
| independent audit | `4a54c41c83091c694582ef26ee7a5264101407c858df5540a54cb94d2e0a568f` |

冻结实现：

| 文件 | SHA-256 |
|---|---|
| `run_npr_balanced.py` | `334890bc764378d470d85cf7ff85fa9740e56c2a550902a0ae0e292b89579106` |
| `analyze_npr_balanced.py` | `326ff0c25652af8a3b69674b6fa56ddab3c41dfcbdaaafa94869c4e2bf407e89` |
| `test_run_npr_balanced.py` | `38fd5fdc6a5165fd2f544a0492a344a860e97c4a02b1a4239faa87db95818d75` |
| `test_analyze_npr_balanced.py` | `6521dbbf9c4ebd04bd418889fe59d07e0ba403f46667e2274496b911c674478b` |

最终 NPR 扩展回归结果为：

```text
227 passed in 80.27s
```

1,775 个 float32 feature 位于 gitignored
`outputs/opensource/npr/.../features/`，不会随 git clone 自动取得。每个
result row 都绑定 feature 的相对路径、bytes、文件 SHA、array SHA、shape、
dtype 和语义；fresh replay audit 依赖本机保留这些 artifacts。

核心重跑命令如下；`CUDA_VISIBLE_DEVICES=4` 是本机物理卡映射，其他机器应
按实际设备调整：

```bash
export CUDA_VISIBLE_DEVICES=4
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=100
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX=/root/.cache/claimforge/pycache/npr-balanced-torch2.8.0
NPR_PY=/root/.cache/claimforge/venvs/npr-balanced-torch2.8.0/bin/python

$NPR_PY -m eval.opensource.run_npr_balanced \
  --mode formal \
  --run-id npr_aigcdetect_progan4class_author_documented_native_even_trim_balanced250_v1_full1775_20260726 \
  --device cuda:0 \
  --fail-fast

$NPR_PY -m eval.opensource.analyze_npr_balanced \
  --run-id npr_aigcdetect_progan4class_author_documented_native_even_trim_balanced250_v1_full1775_20260726 \
  --device cuda:0
```

实际运行额外使用 Hunyuan keepalive handoff wrapper，只负责在 NPR CUDA
窗口暂停、drain 和恢复后台生成，不改变模型、输入、runtime 或数值路径。
新数据应使用新的 immutable run ID；成功 rows 可 resume，配置漂移会被拒绝。

## 11. 支持与不支持的结论

推荐论文表述：

> The released NPR AIGCDetectBenchmark ProGAN-4class checkpoint, evaluated
> with the author-documented native even-dimension compatibility completion
> on the independent Balanced250 panel, did not transfer to local insertions
> (official probability condition-macro AUROC 0.481) and inversely ranked the
> full-frame conditional edits (AUROC 0.327). Float32 sigmoid saturation
> affected 1,296/1,775 images; the pre-registered raw-logit diagnostic made
> the full-frame inversion stronger (AUROC 0.202), not better. All 1,775
> images remained below the released 0.5 threshold. An independent fresh
> full-model replay reproduced all 1,775 stored features, logits and
> probabilities exactly.

不要写成：

- “复现了 NPR 论文训练流程或论文报告的跨生成器数字”；
- “原 GitHub live forward 原样支持 Balanced250 奇数尺寸”；
- “raw logit 是比官方 probability 更好的替代主分数”；
- “NPR 在 Balanced250 上有 52.7% 的 local detection accuracy”；
- “NPR 能检测或定位局部植入”；
- “full-frame conditional edit 等于纯文生图或完全合成图”；
- “input visibility 是模型 localization 输出”；
- “sigmoid score 是本域校准概率”；
- “公开下载或 HF Apache-2.0 metadata 已建立 upstream checkpoint 商用授权”。
