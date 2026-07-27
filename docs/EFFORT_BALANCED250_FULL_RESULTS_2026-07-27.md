# Effort CLIP-L/14（GenImage SDv1.4）在 Balanced250 上的正式结果

日期：2026-07-27（UTC）

> **文档状态：正式完成，独立全链路 fresh replay 审计通过**
>
> 本报告只使用冻结的 1,775-image formal run、两个独立 35-image CUDA
> smoke、共享 Balanced250 T1 指标、persisted-artifact head replay 和
> fresh full-model replay。旧 Mouse run、论文表格和项目页展示分数没有被
> 拿来填写本报告结果。

正式 run：
`effort_clip_l14_genimage_sdv14_balanced250_v1_full1775_r2_20260727`

Formal immutable run-config fingerprint：
`b6b92ee73242030e6a3c15fc30a42bdca91d132ae42fee561f4955e4781dae3e`

核心机器证据：

- [run manifest](../results/opensource/effort/effort_clip_l14_genimage_sdv14_balanced250_v1_full1775_r2_20260727/manifest.json)：
  `548403a412ae19413caefc33de794427a71cf7daa7a38d6abb0118f2d903e39d`
- [expected inputs](../results/opensource/effort/effort_clip_l14_genimage_sdv14_balanced250_v1_full1775_r2_20260727/expected_inputs.jsonl)：
  `6b5128909eeffdbd88e61af02ca1bd191cb5460f94a23b47c87ebd0172e6d12c`
- [逐图结果](../results/opensource/effort/effort_clip_l14_genimage_sdv14_balanced250_v1_full1775_r2_20260727/results.jsonl)：
  `bdebcc5c89b1a7cc699c7e35f0fb6070fe20217651d9cfe6cd0d135aaaa2d50b`
- [coverage summary](../results/opensource/effort/effort_clip_l14_genimage_sdv14_balanced250_v1_full1775_r2_20260727/summary.json)：
  `e35b6b2e26984a6003fc869c63a31c3b33b5024bc715468eefefbfe421f2b2cc`
- [Balanced250 metrics](../results/opensource/effort/effort_clip_l14_genimage_sdv14_balanced250_v1_full1775_r2_20260727/balanced250_metrics.json)：
  `5f53a8f6a36187b93c15e8e418e533f127f82002f4718bffc99a4a28d9ee3075`
- [independent replay audit](../results/opensource/effort/effort_clip_l14_genimage_sdv14_balanced250_v1_full1775_r2_20260727/independent_audit.json)：
  `d00b6fd6d4f78ee77724238c521a1317f3a9a9ba0a68b6596ebc8aa1a1e3d05a`
- [双 smoke comparison](../results/opensource/effort/_reports/effort_clip_l14_genimage_sdv14_balanced250_v1_smoke5x7_a_r3_20260727__vs__effort_clip_l14_genimage_sdv14_balanced250_v1_smoke5x7_b_r3_20260727_comparison.json)：
  `d3d7b218b5c262c4a39f169a7b382bae36042b74f931a2b43e98d80920e43f47`

最终审计状态：`full_fresh_replay_audit_passed`。

## 1. 结论摘要

Effort 的正式执行完整、确定、可复核，但在这组跨形态数据上呈现明显的
local/full-frame 分裂，而且 full-frame 也没有达到强部署基线：

- formal coverage 为 1,775/1,775，零 error、missing 和 superseded；
- local 三条件等权 macro AUROC 为 **0.500955**，AP 为
  **0.503710**，TPR@5%FPR 为 **4.80%**；
- 发布规则 `class-1 float32 softmax > 0.5` 在 local 上的 macro recall
  只有 **7.60%**，macro accuracy 为 **49.80%**；
- full-frame 三条件等权 macro AUROC 为 **0.663179**，AP 为
  **0.665116**，TPR@5%FPR 为 **13.60%**；
- 发布阈值在 full-frame 上的 macro recall 为 **22.93%**，accuracy 为
  **57.47%**；
- source-matched local pooled strict ranking 为 **44.93%**，mean
  probability delta 为 `-0.00304650`；
- source-matched full-frame pooled strict ranking 为 **76.67%**，mean
  probability delta 为 `+0.14508695`；
- 两个 CUDA smoke 的 35/35 computational projection、feature arrays、
  class-logit arrays、score 和 decision 全部 exact；
- formal persisted head replay 和 fresh full-model replay 都是
  1,775/1,775，所有实际最大差异为 `0.0`。

当前证据支持以下结论：

1. Effort 对三组小面积 local insertion 的单图检测与随机排序一致，
   发布阈值也没有可用召回率。
2. 它对 Hunyuan full-frame conditional edits 有中等方向性响应，但
   AUROC、低 FPR 召回和发布阈值召回仍明显不足，不能据此宣称部署级
   整图 AIGC 检测。
3. Full-frame matched ranking 强于 unpaired primary，说明对同一源图做
   条件编辑后分数经常上移；部署时没有 matched real counterfactual，
   因而 secondary 不能替代单图 primary。
4. `pooler_output[1024]` 和两个 class logits 是分类器内部 T1 evidence，
   **不是** heatmap、mask、bbox 或 native dense output；Effort 不进入
   T2 localization。
5. 三组 full-frame 数据是从真实源图出发的 Hunyuan 条件全图编辑，不是
   脱离真实源图独立采样的纯 T2I。本 run 仍不能单独代表所有纯整图生成器。

## 2. 官方来源、checkpoint 与许可证

Effort 对应 ICML 2025 论文
[Orthogonal Subspace Decomposition for Generalizable AI-Generated Image Detection](https://proceedings.mlr.press/v267/yan25b.html)，
正式执行冻结作者的
[官方 GitHub repository](https://github.com/YZY-stack/Effort-AIGI-Detection)。

| 资产 | 冻结身份 | 大小 / 结构 | SHA-256 |
|---|---|---:|---|
| official source | commit `96f5dea2b534d400cfd7003f053c7e93c8e16461` | tracked clean；7 个关键文件逐一绑定 | 见 manifest |
| `effort_clip_L14_trainOn_sdv14.pth` | official GenImage SDv1.4 checkpoint | 1,213,769,519 bytes；681 FP32 tensors；303,378,530 elements | `7c32ceb4e66d303050e8fc5dc7543fa347693fb4ee6b5df4d6eaf9f6a92fb813` |
| checkpoint schema | key/order/shape/dtype/count | strict complete | `bb1d4ba1c015ab4354b42e11af101e29b19a1ab71704b0302bac465c6d3f1489` |
| CLIP-L/14 config | HF revision `32bd64288804d66eefd0ccbe215aa642df71cc41` | 4,519 bytes | `8a09b467700c58138c29d53c605b34ebc69beaadd13274a8a2af8ad2c2f4032a` |

Checkpoint 只通过
`torch.load(map_location="cpu", weights_only=True)` 加载，unsafe globals
为空。模型与 checkpoint 严格完整匹配，无 missing 或 unexpected keys；
模型构建、golden 和推理均不依赖运行时联网。

许可证边界不能省略：固定 commit 的 README 顶部显示
`CC BY-NC 4.0` badge，但仓库跟踪文件中不存在 `LICENSE`、`COPYING`
或 `NOTICE`，也没有单独核实到 checkpoint 许可文本。Manifest 因此冻结：

```text
tracked_license_file_present:       false
code_license_text_verified:         false
checkpoint_license_text_verified:   false
commercial_use_cleared:             false
benchmark_role:                     research_evaluation_only
```

所以这里的“公开方法”表示源码和权重可以公开取得并用于本研究评测，
**不表示已核验完整许可文本，也不表示已获得商业使用授权**。

## 3. 方法原理与训练边界

Effort 的核心是给预训练视觉基础模型的微调增加正交子空间约束。论文认为，
普通二分类微调容易用有限训练生成器的单调 fake pattern 覆盖预训练表示，
造成跨生成器泛化下降。Effort 对 attention projection 权重做 SVD：

1. 保留并冻结主奇异子空间，尽量保存 CLIP 的预训练知识；
2. 只让剩余正交子空间适配 AIGC 痕迹；
3. 推理时把冻结主权重和学习到的 residual 重新相加。

本 checkpoint 使用 CLIP ViT-L/14 vision encoder：

- 输入为 `224×224`，patch size 为 14；
- hidden size 为 1024，共 24 个 Transformer block、16 个 attention
  head；
- 每个 block 的 `q/k/v/out` 四个 `1024×1024` attention linear
  都被替换，共 `24 × 4 = 96` 个 SVD residual module；
- 每个 module 冻结 rank 1023 的主权重，只保留 rank 1 residual；
- 实际前向权重严格为：

```text
W_effective = W_main + U_residual @ diag(S_residual) @ V_residual
```

- CLIP `pooler_output[1024]` 进入 `Linear(1024, 2)` 分类头；
- 主分数是 class 1 的 float32 softmax probability。

这个归纳偏置的优势是用很受限的适配自由度学习 fake pattern，同时尽量
保留 CLIP 的通用表征。但它不保证对任意 manipulation scope 都敏感。
Local insertion 只改变一小块区域，整图又被直接拉伸到 `224×224`；
本结果说明其整图训练目标并没有转化为稳定的局部植入检测。Full-frame
虽然让更多像素经过生成管线，但测试生成器 Hunyuan 与 checkpoint 的
GenImage SDv1.4 训练域、内容域和后处理仍有明显差异。

本评测是 **pinned official release inference**，不是从头重训 Effort，
也不是论文全部跨生成器表格的完整复现。官方另一个 Chameleon SDv1.4
checkpoint 是独立泛化轴，不能与本报告混写。

## 4. Frozen executable contract

正式实现入口：

- [runner](../eval/opensource/run_effort_balanced.py)
- [independent analyzer](../eval/opensource/analyze_effort_balanced.py)

冻结文件 SHA-256：

| File | SHA-256 |
|---|---|
| runner | `19a3eb6b872d33f7272d5a118dff230986cb8b0bb955f2392b55899fc53d81b1` |
| analyzer | `3be8a9760b6ae8a6fb03dea1da977a8bf5df1814d1accc651f1ba302641ee7ca` |
| runner tests | `f63bfbb4aebe59ff4eb5a0fbe20256a6b28c149c3694bcaf08d4b0c093cf63b8` |
| analyzer tests | `1199560d10793788d2b920c295dd83ada08ae0b9a4cd2f558962bf4cfbf7eb73` |

Formal manifest 共绑定 13 个本地 adapter/source 文件；最终交付前
13/13 的 bytes 与 SHA-256 再次和当前工作树精确匹配。

### 4.1 Preprocess

官方发布中存在 natural-image demo 的 `INTER_LINEAR` 路径与
DeepfakeBench dataset/test loader 的 `INTER_CUBIC` 路径。本 run 冻结
README 指向的 natural-image demo 行为：

| 组件 | 冻结行为 |
|---|---|
| decode | `cv2.imread(..., IMREAD_COLOR)` |
| color | OpenCV BGR → RGB |
| resize | 直接拉伸至 `224×224`，`cv2.INTER_LINEAR` |
| geometry | 不保持宽高比、不 crop、无人脸对齐 |
| tensor | uint8 除以 255，FP32 CHW |
| normalization | CLIP mean `[0.48145466, 0.45782750, 0.40821073]`；std `[0.26862954, 0.26130258, 0.27577711]` |
| batch/autocast | batch 1；autocast disabled |

`INTER_LINEAR` 和 `INTER_CUBIC` 不是可互换的拼写。本报告所有分数只属于
`official_deepfakebench_demo_natural_image_linear224_v1`；若要评估
CUBIC，必须用新协议和新 run ID 单独执行。

Direct full-canvas resize 表示三组 local edit 在几何上都进入模型输入，
不表示缩放后仍保留足够局部信息，也不表示模型产生了 localization。

### 4.2 Score 与 artifact

- 模型产生 `pooler_output[1024]` FP32 feature；
- 官方 linear head 产生 `[2]` FP32 class logits；
- primary `ai_score` 是同设备 float32 softmax 的 class 1 probability；
- 分数越高越 fake，发布 decision 为严格 `ai_score > 0.5`；
- 每图保存一个 NPZ，keys 为 `pooler_output`、`class_logits`；
- 每个 NPZ 固定 4,640 bytes，1,775 个 formal artifact 共
  8,236,000 bytes，保存在 gitignored local outputs；
- artifact file、feature array 和 logit array 均有 SHA-256，并逐图绑定
  result row。

Analyzer 载入 JSON/NPZ 时先用 CPU softmax 做无设备模型的静态 sanity。
这个检查的 absolute tolerance 是：

```text
2 * np.finfo(np.float32).eps = 2.384185791015625e-7
```

它只处理同一 persisted logits 在 PyTorch CPU/CUDA softmax kernel 间的
极小末位差异，**不是**正式 replay 容差。两个 smoke 的同设备 head replay
以及 formal persisted-head/fresh-model replay 对 feature、logit、
probability、margin 的 tolerance 全部仍为 `0.0`。

### 4.3 Runtime

| 项 | 值 |
|---|---|
| physical GPU identity | NVIDIA L20Z |
| manifest-recorded logical device | `cuda:0` |
| Python | CPython 3.12.3 |
| PyTorch / torchvision | `2.8.0.dev20250627+cu128` / `0.23.0.dev20250627+cu128` |
| transformers / NumPy / OpenCV | 4.53.2 / 1.26.4 / 4.10.0 |
| cuBLAS workspace | `:4096:8` |
| deterministic algorithms | enabled |
| cuDNN benchmark / deterministic | false / true |
| TF32 | matmul=false；cuDNN=false |
| float32 matmul precision | `highest` |
| CPU threads / seed | 16 / 20260724 |

物理卡序号由调度器通过 `CUDA_VISIBLE_DEVICES` 选择；manifest 只冻结进程
内 logical `cuda:0` 和实际设备 identity，不把宿主机物理序号当成跨机器
科学身份。

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
- bootstrap seed `20260726`；
- TPR@5%FPR 使用 real score 的 0.95 `higher` quantile 和严格 `>`。

六条件总 macro 把 local 与 full-frame 两类不同任务混合，只作为导航摘要。
Local macro 与 full-frame macro 是主要 family-level 结果。

Effort 没有 native dense output，因此：

- T1 whole-image detection：有效；
- T2 local manipulation localization：N/A；
- joint T1/T2：N/A；
- full-frame T2：N/A。

## 6. Coverage、artifact 与 local visibility

### 6.1 Formal coverage

| Condition | Expected | Result | Valid | Error | Missing |
|---|---:|---:|---:|---:|---:|
| real score cache | 275 | 275 | 275 | 0 | 0 |
| local mouse | 250 | 250 | 250 | 0 | 0 |
| local cat | 250 | 250 | 250 | 0 | 0 |
| local trash can | 250 | 250 | 250 | 0 | 0 |
| full-frame mouse | 250 | 250 | 250 | 0 | 0 |
| full-frame cat | 250 | 250 | 250 | 0 | 0 |
| full-frame trash can | 250 | 250 | 250 | 0 | 0 |
| **Total** | **1,775** | **1,775** | **1,775** | **0** | **0** |

Runner inventory：

- physical result rows：1,775；
- latest result rows：1,775；
- superseded attempts：0；
- NPZ files：1,775；
- 每个 NPZ：4,640 bytes；
- persisted same-device head replay：1,775/1,775。

### 6.2 Frozen local visibility

| Condition | Full | Partial | None | Mean visible GT |
|---|---:|---:|---:|---:|
| local mouse | 250 | 0 | 0 | 1.0 |
| local cat | 250 | 0 | 0 | 1.0 |
| local trash can | 250 | 0 | 0 | 1.0 |
| **Local pooled** | **750** | **0** | **0** | **1.0** |

另外 1,025 张 real/full-frame 输入的 local edit visibility 为
`not_applicable`。这里的 `full` 只表示 direct full-canvas resize 没有
几何 crop 掉 GT 区域；它不是模型定位结果。Local primary 仍接近随机，
说明小区域在 `224×224` 表征中被稀释，而不是简单的 crop blind spot。

## 7. Primary T1 结果

### 7.1 Family macro

方括号为 1,000 次 shared-cluster bootstrap 的 95% percentile interval。

| Scope | AUROC [95% CI] | AP [95% CI] | TPR@5%FPR [95% CI] | Accuracy @ 0.5 [95% CI] | Recall @ 0.5 [95% CI] |
|---|---:|---:|---:|---:|---:|
| local macro | 0.500955 [0.488750, 0.514239] | 0.503710 [0.493119, 0.521789] | 0.048000 [0.036336, 0.060352] | 0.498000 [0.487167, 0.508229] | 0.076000 [0.047380, 0.108188] |
| full-frame macro | 0.663179 [0.636796, 0.690252] | 0.665116 [0.634067, 0.700356] | 0.136000 [0.094805, 0.235576] | 0.574667 [0.553432, 0.597957] | 0.229333 [0.187590, 0.272439] |
| all-six mixed macro | 0.582067 [0.566947, 0.598456] | 0.584413 [0.567307, 0.608073] | 0.092000 [0.067766, 0.144177] | 0.536333 [0.522098, 0.550172] | 0.152667 [0.122472, 0.185091] |

`all-six` 混合值位于两类任务之间，不能用它掩盖 local/full-frame 分裂。

### 7.2 条件级 ranking 与 5% FPR

| Condition | AUROC [95% CI] | AP [95% CI] | TPR@5%FPR [95% CI] |
|---|---:|---:|---:|
| local mouse | 0.501984 [0.488293, 0.516105] | 0.507701 [0.493670, 0.529746] | 0.048000 [0.032920, 0.066413] |
| local cat | 0.505456 [0.488835, 0.523607] | 0.503805 [0.486763, 0.527472] | 0.056000 [0.031987, 0.071444] |
| local trash can | 0.495424 [0.479576, 0.511868] | 0.499623 [0.485028, 0.519910] | 0.040000 [0.029535, 0.059339] |
| full-frame mouse | 0.673056 [0.644470, 0.702278] | 0.675123 [0.644787, 0.710787] | 0.152000 [0.098813, 0.255978] |
| full-frame cat | 0.649232 [0.619066, 0.682027] | 0.650819 [0.615477, 0.690168] | 0.120000 [0.074373, 0.218627] |
| full-frame trash can | 0.667248 [0.639463, 0.696163] | 0.669406 [0.634001, 0.705333] | 0.136000 [0.096522, 0.238300] |

所有 overall 条件共享同一个 independent real250，所以 real-only 5% FPR
threshold 都是 `0.7845233678817749`，实际 point-estimate FPR 为
`0.048`。Threshold CI 为 `[0.46368688344955444,
0.9312365055084229]`，actual-FPR CI 为 `[0.037914, 0.049793]`。

### 7.3 Released threshold `probability > 0.5`

| Condition | Accuracy | Precision | Recall | F1 | TP / FP / FN / TN |
|---|---:|---:|---:|---:|---:|
| local mouse | 0.496 | 0.473684 | 0.072 | 0.125000 | 18 / 20 / 232 / 230 |
| local cat | 0.500 | 0.500000 | 0.080 | 0.137931 | 20 / 20 / 230 / 230 |
| local trash can | 0.498 | 0.487179 | 0.076 | 0.131488 | 19 / 20 / 231 / 230 |
| full-frame mouse | 0.586 | 0.759036 | 0.252 | 0.378378 | 63 / 20 / 187 / 230 |
| full-frame cat | 0.562 | 0.718310 | 0.204 | 0.317757 | 51 / 20 / 199 / 230 |
| full-frame trash can | 0.576 | 0.743590 | 0.232 | 0.353659 | 58 / 20 / 192 / 230 |

表中的六组 FP 都对应重复使用的同一个 real250 panel 中 20 个
`probability > 0.5` real，不是 120 张不同 real。Local 三条件合计检出
57/750，full-frame 三条件检出 172/750。由于每个条件正负样本各 250，
balanced accuracy 与表中 accuracy 相同；六条件 specificity 均为
`0.92`，FPR 均为 `0.08`。

### 7.4 Domain

| Family | Domain | AUROC | AP | TPR@5%FPR | Accuracy @ 0.5 | Recall @ 0.5 |
|---|---|---:|---:|---:|---:|---:|
| local | lodging | 0.496082 | 0.516105 | 0.048328 | 0.485759 | 0.033835 |
| local | restaurant | 0.517135 | 0.501609 | 0.047595 | 0.512272 | 0.127983 |
| full-frame | lodging | 0.669736 | 0.690607 | 0.213507 | 0.568698 | 0.188843 |
| full-frame | restaurant | 0.667076 | 0.657932 | 0.128689 | 0.581560 | 0.277355 |
| all-six mixed | lodging | 0.582909 | 0.603356 | 0.130917 | 0.527228 | 0.111339 |
| all-six mixed | restaurant | 0.592105 | 0.579770 | 0.088142 | 0.546916 | 0.202669 |

Local 在两个 domain 都接近随机。Full-frame 的 AUROC 在两个 domain
相近，但 low-FPR 和发布阈值 operating points 不同；这里没有做
simultaneous domain-difference test，因此只作描述。

### 7.5 Probability distribution

| Group | N | Mean | Median | P05 (`linear`) | P95 (`linear`) | `> 0.5` |
|---|---:|---:|---:|---:|---:|---:|
| real score cache | 275 | 0.110997 | 0.009962 | 0.000141 | 0.776973 | 23 |
| local mouse | 250 | 0.106805 | 0.008096 | 0.000111 | 0.717178 | 18 |
| local cat | 250 | 0.105735 | 0.010502 | 0.000113 | 0.802922 | 20 |
| local trash can | 250 | 0.102096 | 0.009589 | 0.000092 | 0.716970 | 19 |
| full-frame mouse | 250 | 0.266101 | 0.058251 | 0.000471 | 0.980338 | 63 |
| full-frame cat | 250 | 0.237321 | 0.051212 | 0.000426 | 0.966345 | 51 |
| full-frame trash can | 250 | 0.253945 | 0.065886 | 0.000568 | 0.982703 | 58 |

Local 与 real 的 absolute probability distributions 几乎完全重叠。
Full-frame mean 和 upper tail 明显上移，但 median 仍远低于 0.5，
这解释了中等 AUROC 与较低发布阈值 recall 同时出现。

## 8. Source-matched secondary

Secondary 只使用 frozen `source_pairs.jsonl` 的真实端点。它测量同一源图
经过编辑后 probability 如何变化，不替代单图 primary。

| Scope | Pairs | Clusters | Mean delta [95% CI] | Median delta | Strict ranking [95% CI] | W / L / T |
|---|---:|---:|---:|---:|---:|---:|
| local mouse | 250 | 247 | 0.001628 [0.000259, 0.003325] | 0.000004 | 0.528000 [0.468860, 0.594404] | 132 / 118 / 0 |
| local cat | 250 | 247 | -0.006272 [-0.012216, -0.000575] | -0.000043 | 0.432000 [0.372947, 0.488281] | 108 / 142 / 0 |
| local trash can | 250 | 246 | -0.004495 [-0.009210, 0.000514] | -0.000099 | 0.388000 [0.325744, 0.452385] | 97 / 153 / 0 |
| full-frame mouse | 250 | 247 | 0.159015 [0.124636, 0.198712] | 0.020903 | 0.768000 [0.715342, 0.821874] | 192 / 58 / 0 |
| full-frame cat | 250 | 248 | 0.126398 [0.093565, 0.160730] | 0.017515 | 0.732000 [0.676224, 0.785447] | 183 / 67 / 0 |
| full-frame trash can | 250 | 246 | 0.149848 [0.115847, 0.185972] | 0.018659 | 0.800000 [0.750000, 0.850748] | 200 / 50 / 0 |
| **local pooled** | **750** | **270** | **-0.003047 [-0.005920, -0.000201]** | **-0.000023** | **0.449333 [0.412910, 0.485315]** | **337 / 413 / 0** |
| **full-frame pooled** | **750** | **269** | **0.145087 [0.114613, 0.176560]** | **0.020354** | **0.766667 [0.724092, 0.810182]** | **575 / 175 / 0** |
| all-pairs mixed | 1,500 | 270 | 0.071020 [0.055662, 0.086864] | 0.000438 | 0.608000 [0.578509, 0.637827] | 912 / 588 / 0 |

Local Mouse 的 mean delta 虽为很小的正值，但 CI for strict ranking
覆盖 0.5；Cat 和 Trash-can 方向反而为负。Local pooled 结果与
primary 的随机表现一致。Full-frame 三条件的 matched delta 均为正，
但 ranking 仍只有约 0.73–0.80，不能把 matched counterfactual 结果
宣传为单图部署准确率。

## 9. Determinism 与独立审计

### 9.1 CPU preflight 与 repository fixtures

CPU preflight 在加载 Balanced250 模型分数和配置 CUDA 之前完成：

- `cuda_initialized_before=false`；
- `cuda_initialized_after=false`；
- accelerator model forwards：0；
- Balanced250 model scores：0；
- source/checkpoint/config 与 681/681 strict model schema 全部通过；
- 96 个 SVD module、rank、position IDs 和参数数目全部重验。

两张官方仓库静态图片作为
`repository_fixture_runtime_regression_not_author_published_golden`，
不是作者发布的 benchmark golden：

| Fixture | CPU fake p | CUDA fake p | CPU/CUDA max logit diff |
|---|---:|---:|---:|
| `figs/effort_pipeline.png` | 0.172120079 | 0.172122210 | 0.000007868 |
| `figs/deepfake_tab1.png` | 0.312006861 | 0.312009096 | 0.000007287 |

CPU 与 CUDA 各自的连续 forward feature/logit 都 exact；frozen runtime
tolerance 为 `1e-6`，CPU/CUDA cross-device acceptance 为 `5e-5`。
Formal fresh audit 再次重跑两张 CUDA fixture，recorded logits、
probabilities 和 feature SHA 全部 exact。

### 9.2 CUDA smoke

最终 canonical smoke：

- `effort_clip_l14_genimage_sdv14_balanced250_v1_smoke5x7_a_r3_20260727`
  （fingerprint
  `c826b7789a8f0ecdacfcb1e37b6cb62e7d3b1aeeb75df8b149351aec42c5438a`）；
- `effort_clip_l14_genimage_sdv14_balanced250_v1_smoke5x7_b_r3_20260727`
  （fingerprint
  `d74a30dc32ab35c1f9cf3ecb728511ec46e3dd504d28a3df1cb37a2ae60c6371`）。

每次覆盖七个 condition 各 5 张，共 35 张。Comparison 结果：

- 35/35 computational projection exact；
- feature arrays 与 class-logit arrays SHA exact；
- score 与 decision exact；
- maximum feature difference：`0.0`；
- maximum class-logit difference：`0.0`；
- A/B persisted head replay 都为 35/35；
- head logit、probability、margin maximum difference 均为 `0.0`，
  tolerance 均为 `0.0`。

NPZ container SHA 没有拿来做 A/B 相等条件，因为 ZIP metadata timestamp
可以不同；内部 feature/logit arrays、shape、dtype、bytes 和各自 run 内
的 file SHA 仍被严格验证。

### 9.3 Formal replay

Persisted artifact head replay：

- artifacts/head/softmax/strict decisions：1,775/1,775；
- head-logit maximum difference：`0.0`，tolerance `0.0`；
- probability maximum difference：`0.0`，tolerance `0.0`；
- margin maximum difference：`0.0`，tolerance `0.0`；
- recorded runtime exact match：true。

Fresh full-model replay：

- freshly reopened and preprocessed：1,775；
- complete model forwards：1,775；
- maximum feature difference：`0.0`，tolerance `0.0`；
- maximum class-logit difference：`0.0`，tolerance `0.0`；
- maximum fresh-head difference：`0.0`，tolerance `0.0`；
- maximum saved-feature-head difference：`0.0`，tolerance `0.0`；
- maximum probability difference：`0.0`，tolerance `0.0`；
- maximum margin difference：`0.0`，tolerance `0.0`；
- preprocess records 与 decisions 全部 exact。

这证明当前结果由冻结官方模型、资产和 LINEAR preprocessing 可精确
重现，但不把执行正确性误写成所有外部分布都有效。

## 10. Runtime 与测试

Formal manifest interval 为
`2026-07-27T06:47:05.772348Z` 至
`2026-07-27T06:50:36.088458Z`，约 **210.316 s**。这段包含正式
1,775-image loop、artifact inventory 和收尾；CPU preflight、CUDA/model
setup 与 runtime golden 发生在 manifest `started_at` 之前。

| Field | Mean | Median | P95 (`higher`) | Max |
|---|---:|---:|---:|---:|
| preprocess latency | 19.903 ms | 23.885 ms | 30.477 ms | 38.541 ms |
| model latency | 27.071 ms | 25.287 ms | 30.315 ms | 164.055 ms |
| peak CUDA allocation | 1209.613 MiB | 1209.613 MiB | 1209.613 MiB | 1209.613 MiB |

最终定向 CPU-only 回归：

```text
22 passed
```

覆盖 runner selection/resume/error/preflight、strict JSON、T2 fail-closed、
NPZ inventory、CPU/CUDA softmax sanity boundary、same-device head replay、
cross-device rejection、fresh model replay 和 canonical output scope。
Black 对 runner/analyzer 及两份测试文件检查通过；Python bytecode compile
通过。最终额外执行了 CPU-only 1,775-result/1,775-artifact load 与共享
metrics recompute，`torch.cuda.is_initialized()` 前后均为 false。

## 11. 解释与限制

本 run 最重要的不是一个混合总分，而是这些边界：

- **局部植入：** 三条件 AUROC 都约 0.5，matched pooled ranking 低于
  0.5；不能把 Effort 当作局部植入 detector；
- **整图条件编辑：** 存在中等 signal，但 low-FPR 和发布阈值 recall
  仍低；它不是强部署结果；
- **定位：** pooled feature 和 class logits 不是 dense prediction，
  T2 必须保持 N/A；
- **数据定义：** full-frame 是 conditional edit，不是 independent
  fully synthetic T2I；
- **协议选择：** 当前结果只属于 natural-image demo 的 LINEAR resize；
- **域迁移：** 训练域 GenImage SDv1.4，测试域 lodging/restaurant/Hunyuan，
  结果不能外推到任意生成器、内容域、压缩或后处理；
- **模型轴：** Chameleon checkpoint 未在本 run 测试；
- **许可证：** 研究评测完成不产生商业授权。

因此推荐在总 benchmark 中把 Effort 记录为：

```text
whole-image T1 complete
local-image T1 evaluated but chance-level
full-frame conditional-edit T1 moderate but weak at deployment thresholds
T2 not applicable
commercial clearance false
```

## 12. 复现入口

本机执行时的调度环境如下；已发布 formal ID 及目录是 finalized immutable
evidence，不能覆盖或用 `--resume` 改写。复现必须创建新 run ID：

```bash
export PYTHONDONTWRITEBYTECODE=1
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CUDA_VISIBLE_DEVICES=4
export EFFORT_REPRO_RUN_ID="effort_clip_l14_genimage_sdv14_balanced250_v1_repro_$(date -u +%Y%m%dT%H%M%SZ)"
```

正式 runner：

```bash
/root/.cache/claimforge/venvs/effort/bin/python \
  -m eval.opensource.run_effort_balanced \
  --repo-root . \
  --mode formal \
  --run-id "$EFFORT_REPRO_RUN_ID" \
  --device cuda:0 \
  --fail-fast
```

独立 analyzer：

```bash
/root/.cache/claimforge/venvs/effort/bin/python \
  -m eval.opensource.analyze_effort_balanced \
  --repo-root . \
  --run-id "$EFFORT_REPRO_RUN_ID" \
  --device cuda:0
```

GPU occupancy 由项目的 grouped benchmark supervisor 与 Hunyuan
keepalive 协议协调：CPU preparation、文档、push 和全部队列结束后保持
keepalive；CUDA group 前暂停并排空在途任务，把兼容方法固定到显式 logical
device，整个 group 结束或异常时只恢复一次 keepalive。
