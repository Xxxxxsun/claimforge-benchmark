# Effort CLIP-L/14（GenImage SDv1.4）在 Mouse full275 上的正式结果

日期：2026-07-25（UTC）

正式 run：
`effort_clip_l14_genimage_sdv14_mouse_canonical_v1_full275_20260725`

核心证据：
[run manifest](../results/opensource/effort/effort_clip_l14_genimage_sdv14_mouse_canonical_v1_full275_20260725/run_manifest.json)、
[逐图结果](../results/opensource/effort/effort_clip_l14_genimage_sdv14_mouse_canonical_v1_full275_20260725/results.jsonl)、
[正式汇总](../results/opensource/effort/effort_clip_l14_genimage_sdv14_mouse_canonical_v1_full275_20260725/summary.json)、
[独立审计](../results/opensource/effort/effort_clip_l14_genimage_sdv14_mouse_canonical_v1_full275_20260725/independent_audit.json)。

## 1. 结论摘要

这次正式运行完整、可复核，但检测效果不具备部署价值。

- 覆盖为 550/550 张图、275/275 个 real/forged 配对，错误、缺失和未配对图像均为 0。
- 图像级 AUROC 为 **0.500456**，95% 配对 bootstrap CI 为
  **[0.498379, 0.502850]**；AP 为 **0.506262**，TPR@5%FPR 为
  **0.054545**。整体表现与随机排序基本一致。
- 在发布阈值 `fake_probability > 0.5` 下，
  `TP/FP/FN/TN = 23/23/252/252`，伪造召回率只有
  **0.083636**。
- 275 对图的阈值判定全部没有变化：252 对为 `real 0 → forged 0`，
  23 对为 `real 1 → forged 1`，没有任何 `0 → 1` 或 `1 → 0`。
- 同一任务内 real 与 forged 分数的 Pearson 相关系数为
  **0.998603**。模型几乎完全跟随底图内容，而不是 Mouse 局部植入。
- forged-real 平均分差虽为正：
  **0.001815588**，95% CI **[0.000480220, 0.003486500]**；
  但胜/负/平为 `146/129/0`，严格配对排序准确率只有
  **0.530909**，其 CI **[0.469091, 0.589091]**，精确双侧符号检验
  `p = 0.334638`。这说明存在一个很小的平均幅度偏移，但没有稳定的
  逐对方向优势。
- 额外校准诊断为 NLL **2.419153**、Brier **0.453701**、
  15-bin ECE **0.440090**。在这个正负严格平衡的数据集上，模型同时
  表现出严重的 real 偏置和概率失校准。

因此，本结果应表述为：**Effort 在这组“小面积局部植入”的 Mouse
压力测试上接近随机，且发布阈值严重偏向 real；微弱的平均分差不能转化为
可靠排序、阈值翻转或可部署检测能力。**

本 run 只产生整图 AIGC 分数，属于 **T1 whole-image detection**；
它不输出定位图，不能计入 T2 局部定位结果。另一个仍然缺失、而且更贴近
Effort 原始任务设定的实验，是 **真实图 vs 完全由生成模型合成的整图**
对照。当前 Mouse forged 图只是在真实底图上做局部植入，不能替代该对照。

## 2. 方法与发布身份

Effort 对应论文
[Orthogonal Subspace Decomposition for Generalizable AI-Generated Image Detection](https://proceedings.mlr.press/v267/yan25b.html)，
发表于 ICML 2025（PMLR 267:70268–70288，ICML Oral）。arXiv 编号为
[2411.15633](https://arxiv.org/abs/2411.15633)。

本次固定的是官方仓库
[YZY-stack/Effort-AIGI-Detection](https://github.com/YZY-stack/Effort-AIGI-Detection)
的 commit：

```text
96f5dea2b534d400cfd7003f053c7e93c8e16461
```

该 checkout 在运行和独立审计时均为 tracked clean。官方同时发布了两个
自然图像检测 checkpoint：

- CLIP-L/14 + Effort，训练于 GenImage SDv1.4；
- CLIP-L/14 + Effort，训练于 Chameleon SDv1.4。

本报告只对应第一个，即：

```text
checkpoint id: official_effort_genimage_sdv14_clip_l14
filename:      effort_clip_L14_trainOn_sdv14.pth
Google Drive:  1UXf1hC9FC1yV93uKwXSkdtepsgpIAU9d
SHA-256:       7c32ceb4e66d303050e8fc5dc7543fa347693fb4ee6b5df4d6eaf9f6a92fb813
bytes:         1,213,769,519
```

Chameleon checkpoint 的结果不能与本结果混写；它仍是一个独立的
checkpoint 泛化轴。

### 2.1 原理及其理论优势

论文观察到，普通二分类微调容易过拟合训练集中有限、单调的 fake
pattern，使特征空间变得受限、低秩，并损害跨生成器泛化。Effort 的核心
做法是对预训练视觉基础模型中的权重做 SVD，将其分成两个正交子空间：

1. 保留并冻结主奇异子空间，以保存 CLIP 的预训练知识；
2. 只让剩余子空间适应伪造模式；
3. 推理时把冻结主权重和学习到的残差重新相加。

本 checkpoint 使用 CLIP ViT-L/14 vision encoder：

- 输入为 `224×224`，patch size 为 14；
- hidden size 为 1024，共 24 个 Transformer block、16 个 attention
  head；
- 每个 block 的 `q/k/v/out` 四个 `1024×1024` attention linear
  都被替换，共 **24 × 4 = 96** 个 SVD residual module；
- 每个 module 冻结 rank 1023 的主权重，只保留 rank 1 residual，
  其形状为 `U[1024,1]`、`S[1]`、`V[1,1024]`；
- 实际前向权重严格为：

```text
W_effective = W_main + U_residual @ diag(S_residual) @ V_residual
```

- CLIP `pooler_output[1024]` 送入 `Linear(1024, 2)` 分类头；
- 正式分数是 float32 softmax 的 class 1 概率，分数越高越像 fake。

这类设计在论文任务上强的原因，是它用很受限的适配自由度学习 fake
pattern，同时尽量不破坏大模型已有的通用视觉表征。然而，该归纳偏置并不
保证对任何伪造形态都有效。本次 Mouse 的局部编辑平均只占原图约
0.1685%，再被直接缩放到 224×224；正式结果显示，论文中的整图 AIGC
泛化优势没有转化为对该局部植入的可靠敏感性。

### 2.2 严格 checkpoint 加载

官方 demo 使用 `load_state_dict(..., strict=False)`，该行在官方源码中还
标有 `FIXME`。本 benchmark 没有沿用这个宽松条件，而是按 checkpoint
实际保存的前向图构造模型，并要求严格加载：

- state tensor：681/681；
- state element：303,378,530；
- missing keys：`[]`；
- unexpected keys：`[]`；
- strict load：`true`；
- 96 个 SVD module 名称和形状全部核对；
- 非持久化 `position_ids[1,257]` 明确重新物化；
- 参数总数：303,378,530。

这里的 “shape-only exact checkpoint forward graph” 只是在构造阶段避免
为了得到形状而重新执行 96 次昂贵 SVD；正式推理仍逐层执行与官方一致的
`W_main + rank-1 residual` 前向公式，并没有用近似模型替代。

## 3. License 边界

固定 commit 的 README 顶部显示 `CC BY-NC 4.0` badge，但仓库跟踪文件中
不存在 `LICENSE`、`COPYING` 或 `NOTICE`，也没有单独核实到 checkpoint
许可文本。因此 manifest 明确记录：

```text
tracked_license_file_present: false
code_license_text_verified:    false
checkpoint_license_text_verified: false
commercial_use_cleared:        false
benchmark_role:                research_evaluation_only
```

所以本次可以作为研究评测使用，但**不能因为 README badge 就认定代码或
权重已经获得商用授权**。若后续产品化，需要向权利人取得清晰、可归档的
代码和 checkpoint 授权。

## 4. 推理协议与 LINEAR/CUBIC 歧义

官方发布中存在两条不同的预处理路径，二者不能静默混用。

| 路径 | 解码与颜色 | resize | crop/对齐 | 本 run 是否采用 |
|---|---|---|---|---|
| README 指向的 natural-image demo | `cv2.imread` BGR → `cv2.COLOR_BGR2RGB` | 直接拉伸至 `224×224`，显式 `cv2.INTER_LINEAR` | 无 landmark 时不裁剪、不做人脸对齐 | 是 |
| DeepfakeBench dataset/test loader | BGR → RGB | 直接拉伸至 `224×224`，显式 `cv2.INTER_CUBIC` | 使用数据集既定输入 | 否 |

两条路径随后都进入 CLIP normalization：

```text
mean = [0.48145466, 0.45782750, 0.40821073]
std  = [0.26862954, 0.26130258, 0.27577711]
```

本 run 冻结的 profile 为：

```text
official_deepfakebench_demo_natural_image_linear224_v1
```

也就是 `cv2.imread(IMREAD_COLOR) → BGR2RGB → 224×224 INTER_LINEAR →
uint8/255 float32 CHW → CLIP normalization`，不保持原宽高比，不裁剪，
也不做人脸对齐。这与官方 natural-image demo 的可执行路径一致。

`INTER_LINEAR` 与 `INTER_CUBIC` 不是同义写法。前者做双线性插值，后者
使用更宽的三次插值邻域；同一原图会得到不同的 uint8 像素和归一化 tensor。
对本次 550 张正式输入做了不写盘的只读复核：

- 550/550 张图的 LINEAR 与 CUBIC `224×224×3` 数组均不完全相同；
- 每图 RGB channel 元素绝对差的均值：总体均值 **0.865198**，
  中位数 **0.742862**，范围 **[0.204115, 2.702075]**；
- 每图不同 channel 元素比例：总体均值 **44.855982%**，
  中位数 **44.276812%**；
- 所有图中的最大单 channel 绝对差为 **55**。

这段量化只说明预处理像素确实不同，**没有生成 CUBIC 模型分数**。本报告
中的所有正式指标只属于 LINEAR 协议；如需判断指标对插值方式是否稳健，
必须把 CUBIC 作为新协议、使用新 run ID 单独运行，不能把它并入当前结果。

## 5. 数据范围与评价定义

正式输入来自 Mouse canonical v1：

```text
dataset_id: claimforge-mouse-good275-canonical-jpeg-q95-v1
pairs:      275
images:     550
lodging:    147 pairs
restaurant: 128 pairs
inputs SHA-256:
e4cb3d6a78fa68f06341457e2234c630a455a9b6b9789e59abf45c15b292060a
```

每个 task 包含一张 real 和一张只做 Mouse 局部植入的 forged 图。所有图按
同一个 canonical JPEG Q95 合同保存，降低由文件格式不一致带来的捷径。

所有 275 个 forged GT 区域在几何上都完整位于送入 resize 的整幅画布内，
所以 `edit_visible_gt_fraction = 1.0`。这只表示没有被 crop 截掉，不表示
编辑在 224×224 输入中足够大。原生分辨率上的 GT 面积比例为：

| 统计量 | GT positive pixels / 原图 pixels |
|---|---:|
| min | 0.000248611 |
| p05 | 0.000433499 |
| median | 0.001126389 |
| mean | 0.001684580 |
| p95 | 0.005225509 |
| max | 0.012927246 |

正式任务合同：

- `primary_task = T1_whole_image_AIGC_detection`；
- `valid_for_t1 = true`；
- `valid_for_t2 = false`；
- 主分数为 `ai_score == fake_probability`；
- 发布阈值为 `score > 0.5`；
- 5% FPR 点使用 real scores 的 0.95 quantile、NumPy `method="higher"`，
  同样采用严格 `>`；
- bootstrap 单位为完整 `task_id` 配对，而不是把 550 张图当作相互独立；
- bootstrap samples 为 1,000，seed 为 20260724。

## 6. 运行前回归与 smoke 验证

### 6.1 非 Mouse CPU/CUDA 回归

在计算任何 Mouse 模型分数前，runner 先用官方仓库内两张静态图片做
runtime regression。必须强调：这是本 benchmark 建立的
`repository_fixture_runtime_regression_not_author_published_golden`，
不是作者论文发布的 golden output。

| fixture | CPU logits | CPU fake p | CUDA logits | CUDA fake p | CPU/CUDA 最大 logit 差 |
|---|---|---:|---|---:|---:|
| `figs/effort_pipeline.png` | `[0.229814023, -1.340861678]` | 0.172120079 | `[0.229807034, -1.340853810]` | 0.172122210 | 0.000007868 |
| `figs/deepfake_tab1.png` | `[-0.161961675, -0.952715337]` | 0.312006861 | `[-0.161968961, -0.952712119]` | 0.312009096 | 0.000007287 |

本次 CUDA 结果与冻结 CUDA 值的最大差为 0，连续两次 forward 的 feature
和 logits 最大差也为 0。运行时容差为 `1e-6`，CPU/CUDA 交叉容差为
`5e-5`；两例均通过。runtime golden fingerprint 为：

```text
cd91d107627401e2eee449da4e70f6802f2a81305194e0c03f2fb4df50004e8e
```

### 6.2 两次 5-pair CUDA smoke

正式运行前完成了两个彼此独立的 5-pair/10-image smoke：

- [smoke A results](../results/opensource/effort/effort_clip_l14_genimage_sdv14_mouse_pair5_cuda_smoke_a_20260725/results.jsonl)
  与
  [smoke A audit](../results/opensource/effort/effort_clip_l14_genimage_sdv14_mouse_pair5_cuda_smoke_a_20260725/independent_audit.json)；
- [smoke B results](../results/opensource/effort/effort_clip_l14_genimage_sdv14_mouse_pair5_cuda_smoke_b_20260725/results.jsonl)。

两次 smoke 的 config fingerprint 均为：

```text
b57a657d08144eda9ffbc3f8d952cb7db49e5702525c48efc92efd6b2f180aba
```

逐 sample 比较结果：

- A vs B：10/10 的 score、两个 logits、margin、decision、完整 preprocess
  记录、feature/logit array hash 全部精确相等，最大绝对差均为 0；
- A vs 正式 full run：同样 10/10 精确相等；
- B vs 正式 full run：同样 10/10 精确相等；
- 三组对应 NPZ 中的 `pooler_output` 和 `class_logits` 均逐元素 bit-exact，
  最大绝对差为 0。

两个 `results.jsonl` 的整文件 SHA-256 不相同，是因为其中包含不同的
`completed_at`、latency 和 run-specific artifact path；不能把“逐样本数值
精确一致”误写成“整个 JSONL 文件 byte-identical”。

smoke A 的独立审计重新执行了 10 次完整模型 forward 和 10 次 artifact
replay，feature/logit/head replay/probability/margin 的最大绝对差均为 0，
并确认 preprocess、decision 和 summary 精确一致。

## 7. 正式运行完整性

正式 run 从 2026-07-25 21:18:41 UTC 运行至 21:19:49 UTC，状态为
`complete`，config fingerprint 为：

```text
4d0a9d95c37e7330c289da96d6177a9f65e5d64f79ea79af53fd8cc59a87404d
```

| 项目 | 数量 |
|---|---:|
| expected images | 550 |
| physical result rows | 550 |
| result images | 550 |
| valid images | 550 |
| error images | 0 |
| missing images | 0 |
| expected / complete pairs | 275 / 275 |
| paired valid images | 550 |
| unpaired valid images | 0 |

独立审计不是只重算 summary，而是：

- 对 550 张图重新执行 550 次完整模型 forward；
- 对 550 个 NPZ artifact 全量 replay；
- 检查 275 个完整配对；
- feature 最大绝对差：0；
- class logits 最大绝对差：0；
- 分类头手工 replay 最大绝对差：0；
- probability 最大绝对差：0；
- logit margin 最大绝对差：0；
- 550 个 decision 全部精确一致；
- 550 份 preprocess record 全部精确一致；
- summary 全字段重算精确一致。

另行逐文件计算了全部 550 个 NPZ 的 SHA-256，并与各自结果行中的
`artifact_sha256` 比较：550/550 匹配，0 个 mismatch。每个 artifact 为
4,640 bytes，总计 2,552,000 bytes。每个 sample 的具体 artifact hash 可在
[逐图结果](../results/opensource/effort/effort_clip_l14_genimage_sdv14_mouse_canonical_v1_full275_20260725/results.jsonl)
和
[独立审计](../results/opensource/effort/effort_clip_l14_genimage_sdv14_mouse_canonical_v1_full275_20260725/independent_audit.json)
中查到。

## 8. 正式检测结果

### 8.1 图像级排序

| 指标 | 点估计 | 95% 配对 bootstrap CI |
|---|---:|---:|
| AUROC | 0.500456 | [0.498379, 0.502850] |
| Average Precision | 0.506262 | [0.505766, 0.513688] |
| TPR @ 5% FPR | 0.054545 | [0.040000, 0.069091] |
| 对应 real-only threshold | 0.784523 | [0.511267, 0.931237] |
| 实际 FPR | 0.047273 | [0.040000, 0.047273] |

数据严格正负平衡，因此随机排序的 AUROC 基线为 0.5，AP 基线约为 0.5。
当前 AUROC 和 AP 都没有显示有实用意义的区分能力。5% FPR 工作点只能
召回约 5.45% forged。

### 8.2 发布阈值 `score > 0.5`

| 指标 | 数值 | 95% 配对 bootstrap CI（如有） |
|---|---:|---:|
| TP | 23 | — |
| FP | 23 | — |
| FN | 252 | — |
| TN | 252 | — |
| accuracy | 0.500000 | [0.500000, 0.500000] |
| balanced accuracy | 0.500000 | [0.500000, 0.500000] |
| precision | 0.500000 | [0.500000, 0.500000] |
| recall | 0.083636 | [0.054455, 0.116364] |
| specificity | 0.916364 | [0.883636, 0.945545] |
| F1 | 0.143302 | [0.098212, 0.188791] |

275 个 pair 的判定迁移矩阵为：

| real decision → forged decision | pairs |
|---|---:|
| `0 → 0` | 252 |
| `0 → 1` | 0 |
| `1 → 0` | 0 |
| `1 → 1` | 23 |

所以 0.5 accuracy 不是“检测对了一半 forged”的表现，而是：每对图都给出
相同判定，在一真一假的平衡设计中自然得到 50% 正确率。

### 8.3 分数分布

| kind | n | mean | median | p05 | p95 | min | max |
|---|---:|---:|---:|---:|---:|---:|---:|
| real | 275 | 0.110997 | 0.009962 | 0.000141 | 0.776973 | 0.00000758 | 0.989493 |
| forged | 275 | 0.112813 | 0.009828 | 0.000131 | 0.799213 | 0.00000893 | 0.990453 |

两类分数的均值和分位数高度重叠；中位数都约为 0.01。模型把绝大多数
real 和 forged 都赋成非常低的 fake 概率，这就是发布阈值下严重
real-biased 的直接来源。

### 8.4 配对敏感性

定义：

```text
delta = forged_ai_score - real_ai_score
```

| 指标 | 结果 |
|---|---:|
| mean delta | 0.001815588 |
| mean delta 95% CI | [0.000480220, 0.003486500] |
| median delta | 0.000004990 |
| p05 / p95 | -0.006321457 / 0.013504738 |
| min / max | -0.039987952 / 0.118479669 |
| wins / losses / ties | 146 / 129 / 0 |
| strict paired ranking accuracy | 0.530909 |
| paired ranking 95% CI | [0.469091, 0.589091] |
| exact two-sided sign-test p | 0.334638 |
| real-forged Pearson r | 0.998603 |
| real-forged Spearman rho | 0.998825 |

均值 delta 的 CI 不含 0，与符号检验不显著并不矛盾：前者对少数较大正
幅度变化敏感，后者只看每对变化方向。当前证据支持“存在极小平均正偏移”，
不支持“多数 pair 能稳定被正确排序”。尤其是所有阈值判定均不改变，
这个小偏移没有形成可操作的检测收益。

### 8.5 分 domain

| domain | pairs | AUROC [95% CI] | AP [95% CI] | TPR@5%FPR | TP/FP/FN/TN | paired rank [95% CI] | mean delta [95% CI] |
|---|---:|---:|---:|---:|---:|---:|---:|
| lodging | 147 | 0.500902 [0.497013, 0.504975] | 0.507765 [0.506105, 0.523671] | 0.068027 | 5/5/142/142 | 0.510204 [0.435374, 0.585034] | 0.002443302 [0.000365377, 0.005156058] |
| restaurant | 128 | 0.501343 [0.497435, 0.506351] | 0.511272 [0.509987, 0.525059] | 0.046875 | 18/18/110/110 | 0.554688 [0.460938, 0.640625] | 0.001094699 [-0.000568610, 0.003152227] |

补充：

- lodging 的 5% FPR threshold 为 0.357941，实际 FPR 为 0.047619；
- restaurant 的 5% FPR threshold 为 0.931237，实际 FPR 为 0.046875；
- lodging 的 pair 胜/负/平为 `75/72/0`，符号检验 `p=0.869050`；
- restaurant 为 `71/57/0`，符号检验 `p=0.250440`。

两个 domain 的 AUROC 均紧贴 0.5，且 paired ranking CI 都覆盖 0.5。
restaurant 在发布阈值下召回更高，主要反映基础内容分数分布和阈值位置
不同，不能解释成对局部植入更敏感。

## 9. 额外校准诊断

以下三项不是 `summary.json` 的冻结主指标，而是从 550 条正式
`results.jsonl` 逐行只读重算的补充诊断：

| 指标 | 定义 | 结果 |
|---|---|---:|
| binary NLL | `-mean[y log p + (1-y) log(1-p)]` | 2.419153 |
| Brier score | `mean[(p-y)^2]` | 0.453701 |
| ECE-15 | fake probability 的 15 个等宽 bin | 0.440090 |

在本数据 50/50 平衡的前提下，恒定输出 0.5 的参照 NLL 为约 0.6931、
Brier 为 0.25。这里 NLL 和 Brier 反而明显更差，ECE 也很高。它们与
0.110–0.113 的两类平均 fake probability 一致，说明该 checkpoint 在当前
域上不仅无法排序，也不能把 softmax 值直接当作可信概率。

不能在同一 275 对上重新拟合校准器再报告“测试效果”；如后续需要校准，
必须另设互不重叠的 calibration split，并在独立 test split 上评估。

## 10. 编辑面积关联：仅探索，不作因果解释

以每个 forged 图的 `GT positive pixels / native image pixels` 为编辑面积，
对面积与 pair delta 做只读探索：

| 关联 | Pearson | p | Spearman | p |
|---|---:|---:|---:|---:|
| area fraction vs delta | 0.161933 | 0.007126 | -0.016907 | 0.780162 |
| area fraction vs `abs(delta)` | 0.193079 | 0.001293 | -0.042120 | 0.486674 |
| area fraction vs forged score | -0.038072 | 0.529544 | -0.199257 | 0.000892 |

线性 Pearson 与秩相关 Spearman 的方向和显著性并不一致，说明结果很可能
受少数大面积/大 delta 点、非线性关系和混杂因素影响。此外，编辑面积没有
被随机分配，它与 domain、底图内容、植入位置和局部纹理共同变化；这里
也没有做多重检验校正。

因此只能把它记录为探索性关联，**不能声称“面积增大导致 Effort 检测
变好”**。本次最稳定的事实仍是：所有 pair 的 0.5 阈值判定均未改变，
且整体 paired ranking CI 覆盖随机水平。

## 11. 性能与运行环境

正式运行环境：

```text
Python       3.12.3
PyTorch      2.8.0.dev20250627+cu128
torchvision  0.23.0.dev20250627+cu128
transformers 4.53.2
NumPy        1.26.4
OpenCV       4.10.0
device       cuda:0 / NVIDIA L20Z
batch size   1
dtype        float32
autocast     false
CPU threads  16
deterministic algorithms true
TF32         disabled
```

`latency_ms` 是 runner 包围模型 forward 的单图耗时，不含 OpenCV
decode/resize、结果序列化和 artifact 写盘：

| 统计量 | forward latency (ms) |
|---|---:|
| min | 24.371594 |
| p05 | 24.616719 |
| median | 24.931427 |
| mean | 25.498452 |
| p95 | 28.847700 |
| max | 35.073265 |

550 行记录的 peak CUDA allocated memory 均为
`1,268,370,432 bytes`（约 1.181 GiB）；CPU 未测量行数为 0。该数值是
当前 batch=1、float32、指定 CUDA 软件栈下的进程内指标，不能直接外推为
其他 GPU、batch size 或并发服务的显存需求。

## 12. 可复现资产与文件哈希

### 12.1 checkpoint 与 CLIP config

| 资产 | bytes / tensors | SHA-256 |
|---|---:|---|
| GenImage SDv1.4 checkpoint | 1,213,769,519 bytes | `7c32ceb4e66d303050e8fc5dc7543fa347693fb4ee6b5df4d6eaf9f6a92fb813` |
| checkpoint schema | 681 tensors | `bb1d4ba1c015ab4354b42e11af101e29b19a1ab71704b0302bac465c6d3f1489` |
| checkpoint ordered key list | 681 keys | `1782f72f07007cebae76a0f315845f1c60456d9223d47c8ce2f35a8f43816da7` |
| pinned CLIP config | 4,519 bytes | `8a09b467700c58138c29d53c605b34ebc69beaadd13274a8a2af8ad2c2f4032a` |

checkpoint 使用 `weights_only=true` 加载，unsafe globals 为 `[]`。

### 12.2 固定官方源码文件

| 文件 | bytes | SHA-256 |
|---|---:|---|
| `README.md` | 7,972 | `f5c0f66ed8566c65818162722c9935485721ad401bc8494d2742b32b75fd5721` |
| `install.sh` | 1,000 | `b37e791f514b28f09f64ad84b689d5a9deaff2c3abb688350aae7cbb0711c7fd` |
| `DeepfakeBench/training/demo.py` | 8,899 | `009db8f76d3983e22d0e241ef602b11908f652f84d0fd5f5857f1973fdd12f9c` |
| `DeepfakeBench/training/detectors/effort_detector.py` | 14,154 | `366b1cde008f537e4b9c8c8e4c65ee20b430c4bca1ccee1b1b86c20a9831fac9` |
| `DeepfakeBench/training/config/detector/effort.yaml` | 2,808 | `1fd1398cf245b3a5c13cb130d7c6e209057ae6b56b561eca5f3c032283c5527b` |
| `figs/effort_pipeline.png` | 148,026 | `f84fad60b6152b915874cfbd58ee7e21646fd4a36a642683dbb425e6f6bc879b` |
| `figs/deepfake_tab1.png` | 222,119 | `f8494b571f9d663639193344fa8e0e18f1d41f42089f01f06133dc881ab39fc7` |

### 12.3 canonical 数据合同

| 文件或合同 | SHA-256 |
|---|---|
| [manifest.json](../outputs/opensource/mouse_canonical_v1/manifest.json) | `beb3c30e436db682bbadef794404838f33a4812f18f22819dd6ab1ef3de6f0b1` |
| [inputs.jsonl](../outputs/opensource/mouse_canonical_v1/inputs.jsonl) | `e4cb3d6a78fa68f06341457e2234c630a455a9b6b9789e59abf45c15b292060a` |
| [pairs.jsonl](../outputs/opensource/mouse_canonical_v1/pairs.jsonl) | `bb6328be7cc7d4ae74b1e5b0b132f7fb6133c6fe73f294ebb46aebeda4f8f4b8` |
| release contract fingerprint | `c419e24d6f9d69822ca575e00e30f2c769ba7a28a2fcea1f6634466caf540757` |

### 12.4 正式 run 顶层文件

| 文件 | SHA-256 |
|---|---|
| [expected_inputs.jsonl](../results/opensource/effort/effort_clip_l14_genimage_sdv14_mouse_canonical_v1_full275_20260725/expected_inputs.jsonl) | `e4cb3d6a78fa68f06341457e2234c630a455a9b6b9789e59abf45c15b292060a` |
| [results.jsonl](../results/opensource/effort/effort_clip_l14_genimage_sdv14_mouse_canonical_v1_full275_20260725/results.jsonl) | `5f70c6d0df22d011e2d81ef83ef038e7492b490a4df12f487e92f3af5e647ad2` |
| [summary.json](../results/opensource/effort/effort_clip_l14_genimage_sdv14_mouse_canonical_v1_full275_20260725/summary.json) | `bb4befcbc46b308792daec7ca15359bfcc95f38354b07a95310f1c0e23f19767` |
| [run_manifest.json](../results/opensource/effort/effort_clip_l14_genimage_sdv14_mouse_canonical_v1_full275_20260725/run_manifest.json) | `2e11cb3884742ec6303ec7c13f8770b3acb9756a2c4e239187517bf1a5586eaf` |
| [independent_audit.json](../results/opensource/effort/effort_clip_l14_genimage_sdv14_mouse_canonical_v1_full275_20260725/independent_audit.json) | `53401a7d1c9fae6dffca837d1654252ab822f25b19b005375a5a3bdc5f56d98c` |

### 12.5 smoke 顶层文件

| run/file | SHA-256 |
|---|---|
| smoke A `expected_inputs.jsonl` | `d18681c46babc0f0e4e2ab1811b8cf8e6bba38fcdf3c8276e5b840fb8990efef` |
| smoke A `results.jsonl` | `c92223df45e564a010f4f21e2f90740fac2cebb2d28a416428e0ba42385bf93c` |
| smoke A `summary.json` | `1d6cbf8833dd3276a83df2dbd67bf4487dc641ee150f47de42de141ea2807c2d` |
| smoke A `run_manifest.json` | `384fee2b4d12838f83819b265d2ff7bb29bcad906f9b1cf79d72a63c10fb68ab` |
| smoke A `independent_audit.json` | `fb8e9b752437f870309b700fb29c0ab1c58418f4faccd8b768f00159f969f106` |
| smoke B `expected_inputs.jsonl` | `d18681c46babc0f0e4e2ab1811b8cf8e6bba38fcdf3c8276e5b840fb8990efef` |
| smoke B `results.jsonl` | `6cd2e2aaba97c419c7f50f3cc7aadc13eaf4a15952d1897e1bf70106a24459af` |
| smoke B `summary.json` | `a46247f18cc224f5ecddcdf9770d26e9a917c8afa6c36830f4e5304d5aef1c4d` |
| smoke B `run_manifest.json` | `fbdf4ba943b3abf241f0d2a5e8124c639e1548e2210301c09cb43b20d993f25e` |

## 13. 安全重跑命令

实现入口：
[run_effort.py](../eval/opensource/run_effort.py)、
[analyze_effort_run.py](../eval/opensource/analyze_effort_run.py)、
[effort_metrics.py](../eval/opensource/effort_metrics.py)。

已有正式 run ID 已经存在。若只是安全恢复或验证已有任务，必须使用
`--resume`，不能把同一 ID 当成新 run：

```bash
effort_python=/root/.cache/claimforge/venvs/effort/bin/python
test -x "$effort_python" || {
  printf '%s\n' "Effort Python 不可执行: $effort_python" >&2
  exit 1
}

"$effort_python" eval/opensource/run_effort.py \
  --run-id effort_clip_l14_genimage_sdv14_mouse_canonical_v1_full275_20260725 \
  --device cuda:0 \
  --resume

"$effort_python" eval/opensource/analyze_effort_run.py \
  --run-id effort_clip_l14_genimage_sdv14_mouse_canonical_v1_full275_20260725 \
  --device cuda:0
```

第二条命令会重新生成该目录内的 `independent_audit.json`，从而改变其
`audited_at` 和文件 SHA-256；如要保留本报告冻结的审计文件，不要在原
run 目录执行第二条。

若要做真正独立的复跑，必须使用一个从未存在的新 ID：

```bash
effort_python=/root/.cache/claimforge/venvs/effort/bin/python
effort_new_run_id=effort_clip_l14_genimage_sdv14_mouse_canonical_v1_full275_rerun_20260725_01
test -x "$effort_python" || {
  printf '%s\n' "Effort Python 不可执行: $effort_python" >&2
  exit 1
}
test ! -e "results/opensource/effort/$effort_new_run_id" || {
  printf '%s\n' "run ID 已存在，请换一个新 ID: $effort_new_run_id" >&2
  exit 1
}

"$effort_python" eval/opensource/run_effort.py \
  --run-id "$effort_new_run_id" \
  --device cuda:0

"$effort_python" eval/opensource/analyze_effort_run.py \
  --run-id "$effort_new_run_id" \
  --device cuda:0
```

这些命令默认使用当前冻结的官方 source、GenImage SDv1.4 checkpoint、
CLIP config、Mouse canonical manifest 和 LINEAR224 profile。若更换
checkpoint、插值路径、数据 release 或软件栈，结果必须使用新的 run ID，
并视为新协议，不能追加到本正式 run。

## 14. 最终判定与后续缺口

本次结果回答了一个明确问题：**官方 Effort GenImage SDv1.4 checkpoint
能否把同一真实底图上的 Mouse 局部植入稳定地识别为更 fake？**

答案是否定的：

- 整体 AUROC 约为 0.5005；
- 低 FPR 下召回约为 5.45%；
- 发布阈值下 forged recall 约为 8.36%；
- 275 对图没有一个 threshold decision 因植入而改变；
- real/forged 分数相关达到 0.9986；
- tiny positive mean delta 没有稳定的逐对方向证据；
- 概率明显 real-biased 且严重失校准。

所以它不能作为当前 Mouse 局部植入场景的可部署 detector，也不能因为论文
在 AIGC benchmark 上很强，就把论文结论直接外推到这里。

仍需明确补齐的结果有：

1. **完全合成整图 T1 对照**：真实图 vs 全图生成图，并匹配分辨率、JPEG
   编码和 domain，避免格式捷径。这才直接检验 Effort 的原始 AIGC 任务。
2. **T2 局部定位方法**：Effort 只给整图概率，没有 mask 或 box；当前
   结果不能替代局部定位评测。
3. 可选的 checkpoint 鲁棒性轴：官方 Chameleon SDv1.4 checkpoint。
4. 可选的预处理敏感性轴：单独的新 run 比较 LINEAR224 与 CUBIC224；
   当前正式 run 只报告 LINEAR，不能事后混合。

在第 1 项完成前，对 Effort 更公平、也更完整的总评应当是：
**“局部 Mouse 植入压力测试失败；面向完全合成整图的目标场景尚未测试。”**
