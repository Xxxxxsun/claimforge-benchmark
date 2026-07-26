# B-Free DINO2reg4 在 canonical Mouse 集上的全量结果（2026-07-25）

## 1. 状态与结论先行

B-Free 已按预先冻结的官方 release inference 协议完成 CLAIMFORGE
**local-splice** 条件的全量评测。正式 run
`bfree_dino2reg4_mouse_canonical_v1_full275_20260725` 包含 275 个 matched
task、550 张 canonical JPEG；550/550 图像有效，275/275 配对完整，零错误、
零缺失、零重复物理结果行。随后进行的独立审计重新打开并预处理全部 550
张图像，执行 550 次新的完整模型前向，并以最大绝对误差 `0.0` 复现所有
`[5,768]` feature、五个 crop logits、平均 raw logit 和严格决策。

结论是：**B-Free 对被模型实际看到的局部植入存在可测的 matched-pair
方向性反应，但作为一张图像单独输入的检测器仍接近随机排序，而且官方
阈值在 Mouse 上几乎完全失效。**

- 官方 raw-logit 主分数的 AUROC 为 **0.512529**，1,000 次 task-pair
  bootstrap 的 95% percentile CI 为 **[0.507815, 0.518612]**；
- average precision 为 **0.513062**，CI **[0.509910, 0.521203]**；
- real-only 5% FPR 工作点的 TPR 为 **5.4545%**，CI
  **[3.6364%, 6.9091%]**；
- 官方严格规则 `mean raw logit > 0` 检出 **2/275 forged**，同时误报
  **1/275 real**，混淆矩阵为 **TP=2, FP=1, FN=273, TN=274**；
- 这三个正例来自两个 lodging task，其中一个 task 的 real 和 forged
  同时为正；因此官方阈值实际只产生 **1 个 forged-positive /
  matched-real-negative 配对**；
- forged 分数严格高于 matched real 的配对为 143 个，低于为 67 个，
  exact tie 为 65 个；平均 forged-minus-real raw-logit delta 为
  `+0.061657`，CI `[0.039677, 0.084790]`；
- 在 173 个 `full`-visibility 配对中，120 个向正确方向移动、53 个向
  相反方向移动、零 tie，strict paired ranking 为 **69.36%**；但该
  stratum 的 AUROC 仍仅为 **0.516723**，官方阈值 recall 仅
  **1.16%**。

因此不能把本结果简化成“模型完全没有检测到变化”，也不能把 69.36%
的 full-visible 配对排序宣传成可部署检测性能。前者忽略了真实但较弱的
局部敏感性，后者依赖实际部署时并不存在的 matched real
counterfactual。对任意单张 forged 与任意单张 real 的区分能力、固定阈值
recall 和 calibration 才是 T1 检测的主要结论。

B-Free 是 whole-image binary classifier。五个 crop logits 和五个
768 维 classifier features 只是 T1 诊断产物，不是 manipulation
probability map。因此本 run 仅对 **T1 whole-image AIGC detection**
有效；**T2 localization 与 joint T1/T2 score 均为 N/A**。

本结果只完成 B-Free 的 local-splice 条件。它不是论文全量 benchmark
复现，也没有完成同域 fully synthetic lodging/restaurant 对照；在该对照
完成前，B-Free 应标记为 **local-splice complete、contrast
incomplete**。

## 2. 固定论文、源码、checkpoint 与许可证

B-Free 来自 CVPR 2025 论文
[“A Bias-Free Training Paradigm for More General AI-generated Image
Detection”](https://openaccess.thecvf.com/content/CVPR2025/html/Guillaro_A_Bias-Free_Training_Paradigm_for_More_General_AI-generated_Image_Detection_CVPR_2025_paper.html)；
作者同时发布了
[项目页](https://grip-unina.github.io/B-Free/)、
[论文 HTML](https://arxiv.org/html/2412.17671v2) 和
[官方 GitHub repository](https://github.com/grip-unina/B-Free)。

源码、权重和执行协议均在查看任何 Mouse 模型分数之前冻结：

| 资产 | 固定身份 | 大小 / 结构 | SHA-256 / 作用 |
|---|---|---:|---|
| [官方源码](https://github.com/grip-unina/B-Free/tree/c6a9f898782fb466b29af01f21960b67415afb0e) | commit `c6a9f898782fb466b29af01f21960b67415afb0e` | clean tracked worktree；18 个关键文件逐一 hash | 唯一执行源码 |
| [官方权重包](https://www.grip.unina.it/download/prog/B-Free/weights/BFREE_dino2reg4.zip) | `BFREE_dino2reg4.zip` | 321,653,488 bytes；MD5 `f3f53fa647848b16cf81c913f148a198` | `8230fd3f0f3a64a6403acb692ce1663718ed16f36a5a4de4a68c0d273781769f` |
| Release config | `config.yaml` | 153 bytes | `1f0cb4988933de06a4c2427b1b5b015baa18cea7bc5223a9f54ca5e077ec8d40` |
| Official checkpoint | `model_epoch_best.pth` | 346,171,370 bytes；177 个 FP32 tensor；86,526,721 state elements | `5948ca78f4d94e820c250d24cdf155035b4a85960443800bfe6bb7f06bffe947` |
| Checkpoint schema | 完整 key/shape/dtype/count 清单 | top-level key 仅 `model`；全部 finite FP32 | `e4bb9ddd115309740a70235152b7376e2c8299bb90baf243809f2a5e1665f524` |

checkpoint 使用 `torch.load(map_location="cpu", weights_only=True)` 加载；
unsafe globals 清单为空。构建后的网络与 checkpoint 严格完整匹配，无
missing 或 unexpected keys；网络构建和推理期间禁用外网。

许可证边界必须单独强调。该 repository 的
[GRIP-UNINA 自定义许可证](https://github.com/grip-unina/B-Free/blob/c6a9f898782fb466b29af01f21960b67415afb0e/LICENSE.txt)
只允许 informational 和 nonprofit purposes，并明确禁止未经授权的
industrial 或 profit-oriented use。许可证文件 SHA-256 为
`cd00edf99fbfdbb173831bb0a4d5bfc40423c6e5041f62d7afdda220c4be8b27`。
所以这里的“官方开源方法”表示源码和权重公开可下载、可进行本研究评测，
**不表示 OSI 宽松开源，也不表示商用获准**。本报告不是对其第三方依赖、
训练图像或数据许可的独立法律审计。

## 3. 方法原理，以及为什么它与 Mouse 特别相关

B-Free 的主要贡献不是一个复杂的新 classifier head，而是尽量消除训练
数据偏差的构造方法。很多 real/fake detector 可能学到内容、分辨率、
文件格式或数据来源差异，而不是生成模型留下的痕迹。B-Free 从同一张
COCO real image 出发，通过 Stable Diffusion 2.1 的 conditioning
机制生成语义对齐的 fake，使 real/fake 差异更集中在生成过程本身。

官方训练构造包括：

1. 用空 mask 进行 self-conditioned regeneration，使生成图保持原图
   内容；
2. 用 object mask 替换同类物体，或用矩形 mask 替换不同类别物体；
3. 除默认会重生成整图背景的 inpainting 外，再构造恢复原始 pristine
   real background 的 `origBG` 版本；
4. `inpainted++` 训练策略进一步加入 blur、JPEG compression、scaling、
   cut-out、noise 和 jitter；
5. 用该数据对 DINOv2 预训练、带四个 register token 的 ViT-B/14
   端到端微调，而不是只训练一个冻结 feature 上的 linear probe。

论文和项目页记录 51,517 张 COCO real 与 309,102 张 SD 2.1 generated
训练图像。`origBG` 将原始背景恢复到局部生成区域周围，论文将其描述为
实际上形成 local image edit；这使 B-Free 比只在全图生成图像上训练的
whole-image detector 更接近 Mouse 的 threat model。

但它仍不是 Mouse 的精确训练分布：

- 训练生成器是 SD 2.1，而 Mouse 当前 forged 来自 Hunyuan；
- 公开训练构造先对 COCO 做 512×512 处理，并混合 whole-image
  regeneration 与多种局部变体；
- Mouse 是真实 lodging/restaurant 场景中的小目标植入，并统一输出为
  canonical JPEG-Q95；
- 全局 augmentation、生成区域比例和内容域都不同。

此外，release config 只记录网络结构、normalization 和 checkpoint
文件名，并不记录 `inpainted++` 的完整训练 recipe；repository 也未发布
可从头复现该 checkpoint 的完整训练代码。因此这里准确的声明是
**pinned official release inference**，不是 from-scratch paper
reproduction。

## 4. 预先冻结的官方推理与指标协议

| 组件 | 冻结值 |
|---|---|
| Dataset | `claimforge-mouse-good275-canonical-jpeg-q95-v1` |
| Coverage | 275 matched tasks；550 canonical JPEG images |
| Model | `BFREE_dino2reg4`，end-to-end fine-tuned DINOv2 ViT-B/14，4 registers，CLS-token global pool，768→1 linear head |
| Decode | `PIL.Image.open(...).convert("RGB")`；不做 EXIF transpose 或 ICC conversion |
| Resize | 无；保持原生分辨率 |
| Tensor | torchvision `ToTensor`，uint8 除以 255 得 FP32 |
| Normalize | ImageNet/ResNet mean `[0.485,0.456,0.406]`、std `[0.229,0.224,0.225]` |
| Patch projection | kernel 14、stride 14；丢弃不足 14 像素的右边和下边 remainder |
| Normal five-crop | token grid 两维都至少 36 时，按 center、top-left、bottom-left、bottom-right、top-right 取五个 36×36-token，即 504×504-pixel crop |
| Small-grid fallback | 任一 token-grid 维度小于 36 时执行 release 的 periodic `replicate_wrap`，同时把另一维截为最前 36 tokens；随后五个 crop input 相同 |
| Artifact | 每张图保存 `[5,768]` pre-head features 与 `[5]` crop logits |
| Primary score | 五个 crop **raw logits 的 FP32 mean**，无界 finite，越高越 fake |
| Released decision | strict `mean raw logit > 0` |
| Diagnostic probability | `sigmoid(mean raw logit)`；不替换官方 raw-logit 主分数 |
| 5% FPR point | real-only 95th percentile，NumPy `method="higher"`，strict `>` |
| Bootstrap | 1,000 次 complete `task_id` pair resampling，seed `20260724` |
| Calibration | 官方 `code/utils/dmetrics.py` 定义的 balanced NLL 与 15-bin balanced ECE |
| Runtime | batch 1、`eval()`、FP32、无 autocast、无 TF32、deterministic algorithms |
| Scope | 仅 T1；crop features/logits 不是 T2 |

“原生分辨率”和“五 crop”不等于所有像素都被模型消费。正常路径只消费
五个 504×504 receptive-field 矩形的 union；patch projection 还会舍弃
右/下不足 14 像素的余数。小网格路径也不是简单地在短边 padding 后保留
长边全部位置：release 的 `replicate_wrap` 会把最终 grid 固定为
36×36，使五次 classifier forward 接收到相同 token window。本评测以
可执行 release 行为为准，而不是用论文较宽泛的“padding / multiple
crops”描述自行改写。

每张图的保存 feature 都通过同一个官方 linear head 手工重放，五个
重放 crop logits 与 hook 捕获值一致，且其 FP32 mean 与官方 wrapper
输出一致。它们用于证明 T1 执行忠实性，不被重解释成局部热图。

## 5. Mouse 打分前的协议冻结与官方 golden

四张 repository 自带 demo image 及官方 `results.csv` raw logit 构成
非 Mouse executable golden。CPU 预检先建立当前 runtime reference；
由于官方 CSV 只给有限小数位，CPU 对官方值的最大绝对差为
`2.5853210448900654e-05`，所以在 CUDA 预检和任何 Mouse score 之前将
官方 acceptance tolerance 冻结为 `5e-5`，当前 runtime regression
tolerance 冻结为 `1e-6`。

| Demo | Label | 官方 CSV raw logit | 冻结 CPU | 正式 CUDA | CUDA 对官方绝对差 |
|---|---:|---:|---:|---:|---:|
| `img0000.png` | 0 | -5.9374785 | -5.93747091293335 | -5.937470436096191 | `8.063903808697148e-06` |
| `img0001.png` | 0 | -4.441922 | -4.441921710968018 | -4.441922187805176 | `1.8780517585526013e-07` |
| `img0002.png` | 1 | 4.430519 | 4.430544853210449 | 4.430531978607178 | `1.297860717741628e-05` |
| `img0003.png` | 1 | 3.8499813 | 3.8499996662139893 | 3.8499915599823 | `1.0259982299754e-05` |

四个 CUDA case 都在两个完整 forward 中 bit-identical，最大官方差
`1.297860717741628e-05 < 5e-5`。正式 full run 与独立审计再次通过相同
golden；manifest 明确记录 golden 阶段
`mouse_model_scores_computed: 0`。因此 checkpoint、预处理、score、
threshold、visibility 和容差均不是查看 Mouse 表现后调出的。

## 6. CUDA smoke 的独立复现

正式 275-pair run 前，先对相同的前 5 个完整 task、10 张图像建立两个
不同 run directory：

- `bfree_dino2reg4_mouse_pair5_cuda_smoke_a_20260725`
- `bfree_dino2reg4_mouse_pair5_cuda_smoke_b_20260725`

两次 smoke 的配置 fingerprint 都是
`1318cf5666ed7e9dc2c68319b583e85c41f08165de9a14823b50104bd8722454`。
逐 sample ID 比较后，10/10 的 input SHA、完整 preprocessing record、
raw logits、五个 crop logits、`[5,768]` feature content hash、
crop-logit content hash、diagnostic probability 和严格 decision 全部
精确相同。Smoke A 的独立审计又完成 10 次 fresh full-model forward，
上述 feature/logit/score 最大绝对差全部为 `0.0`。

在最终 contract 字符串加固后又运行
`bfree_dino2reg4_mouse_pair5_cuda_smoke_c_hardened_20260725`。C 与 A/B
对共同 10 张图像的全部推理相关字段仍逐项完全一致；C 的独立审计同样
完成 10 次 fresh forward，最大差全部为 `0.0`。正式 full run 中对应的
前 10 张图像又与 C 的 input、preprocess、feature hash、crop logits、
raw logit、probability 和 decision 精确一致。

三个 smoke 的 5-pair AUROC 都是 `0.52`、AP 都是
`0.5311111111111111`，但这些小样本指标只用于验证执行和持久化，不用于
选择 checkpoint、修改协议或决定是否继续 full run。

## 7. 五 crop 覆盖率、wrap 与 edit visibility

在任何 Mouse score 之前，forged image 的冻结 exact-difference GT
positive pixels 已与五个官方 crop receptive-field 矩形的 union 相交：

| Domain | Full | Partial | None | Total |
|---|---:|---:|---:|---:|
| Lodging | 95 | 18 | 34 | 147 |
| Restaurant | 78 | 18 | 32 | 128 |
| **All** | **173** | **36** | **66** | **275** |

平均 visible GT fraction 为 `0.6891766376903072`，median 为 `1.0`。
`edit_visibility` 是复制给 matched real/forged 两行的输入条件 stratum，
不是 B-Free prediction。

共有 26 个配对进入 `replicate_wrap` 路径，其中 17 full、2 partial、
7 none；这些图像的五个 crop starts 只有一个 distinct 位置。全数据的
distinct crop-start census 为：248 对有 5 个位置，1 对有 3 个位置，
26 对有 1 个位置。

正式结果中的 matched artifact equality 进一步验证可见性影响：

| Visibility | Pairs | 相同 `[5,768]` feature hash | 相同五 crop-logit hash | 相同 mean raw logit | 相同 strict decision |
|---|---:|---:|---:|---:|---:|
| Full | 173 | 0 | 0 | 0 | 172 |
| Partial | 36 | 0 | 0 | 0 | 36 |
| None | 66 | 65 | 65 | 65 | 66 |

65/66 个 `none` pair 的全部 classifier features 和 logits 完全相同，
因此必然 tie。唯一非 tie 是 `lodging_247_slot_001`：real 为
`-5.307627201080322`，forged 为 `-5.3081278800964355`，delta
`-0.0005006790161132812`；它仍远低于官方阈值，且两者 decision 相同。
这里不把 `none` 标签扩大解释成“所有 decoded RGB byte 必须相同”；
该标签的严格定义仅是冻结 GT positive pixels 对官方 receptive-field
union 的可见比例为零。

coverage loss 是失败的一部分，但不是全部原因：173 个 fully visible
pair 的全部 feature 都发生变化，而且 120 个分数向 fake 方向上移；
然而该 stratum 的 standalone AUROC 仍只有 `0.516723`，官方阈值只检出
2 个 forged。

## 8. Full-275 主要 T1 指标

以下方括号均为 1,000 次 complete-task-pair resampling 的 95%
percentile interval。

| Metric | Estimate | 95% CI |
|---|---:|---:|
| AUROC | 0.512529 | [0.507815, 0.518612] |
| Average precision | 0.513062 | [0.509910, 0.521203] |
| TPR @ target FPR 5% | 0.054545 | [0.036364, 0.069091] |
| Real-only 5% FPR threshold | -1.723888 | [-2.202178, -0.789752] |
| Actual FPR at that threshold | 0.047273 | [0.036364, 0.047273] |
| Accuracy @ raw logit 0 | 0.501818 | [0.500000, 0.505455] |
| Balanced accuracy @ raw logit 0 | 0.501818 | [0.500000, 0.505455] |
| Precision @ raw logit 0 | 0.666667 | [0.000000, 1.000000] |
| Recall @ raw logit 0 | 0.007273 | [0.000000, 0.018182] |
| F1 @ raw logit 0 | 0.014388 | [0.000000, 0.035339] |
| Specificity @ raw logit 0 | 0.996364 | [0.989091, 1.000000] |
| Strict paired ranking | 0.520000 | [0.458182, 0.578182] |
| Mean forged-real raw-logit delta | 0.061657 | [0.039677, 0.084790] |

| Operating point | TP | FP | FN | TN |
|---|---:|---:|---:|---:|
| Released `mean raw logit > 0` | 2 | 1 | 273 | 274 |
| Real-only threshold `-1.7238876819610596` | 15 | 13 | 260 | 262 |

在官方零阈值处，273 对双方均为负、1 对双方均为正、1 对 forged-only
为正、0 对 real-only 为正。具体两个正 forged task 是：

| Task | Real raw logit | Forged raw logit | Pair state |
|---|---:|---:|---|
| `lodging_268_slot_001` | 1.9245823621749878 | 2.0162904262542725 | both positive |
| `lodging_156_slot_001` | -0.6945870518684387 | 0.8855333328247070 | forged-only positive |

在 real-only 5% FPR 工作点，13 对双方均为正，另有 2 对 forged-only
为正，没有 real-only 正例；因此 15/275 TPR 只比 13/275 FPR 多两个
forged。

官方 raw-logit 分布为：

| Kind | Min | Mean | Median | P05 | P95 | Max |
|---|---:|---:|---:|---:|---:|---:|
| Forged | -7.794010 | -4.595822 | -4.911594 | -6.673516 | -1.557746 | 2.016290 |
| Real | -7.819443 | -4.657479 | -4.946311 | -6.742280 | -1.732715 | 1.924582 |

两类分数都集中在官方零阈值下方。按官方 `dmetrics` 定义复算的 balanced
NLL 为 `2.3398149041022123`，15-bin balanced ECE 为
`0.4696395616325198`，说明论文强调的 calibration 优势没有直接迁移到
当前 local-splice distribution。这里不进行 Mouse-label threshold
recalibration，也不以 diagnostic sigmoid probability 替换官方 raw
logit。

## 9. Domain 与 visibility strata

### Domain strata

| Domain | Pairs | AUROC [95% CI] | AP [95% CI] | TPR@5% FPR [95% CI] | TP / FP | Paired W / L / T |
|---|---:|---:|---:|---:|---:|---:|
| Lodging | 147 | 0.512772 [0.506593, 0.521567] | 0.514098 [0.508323, 0.526627] | 0.047619 [0.020408, 0.068027] | 2 / 1 | 81 / 33 / 33 |
| Restaurant | 128 | 0.514099 [0.504790, 0.525513] | 0.518277 [0.512744, 0.533764] | 0.054688 [0.031250, 0.078125] | 0 / 0 | 62 / 34 / 32 |

| Domain | Strict paired ranking [95% CI] | Mean paired delta [95% CI] | Exact sign-test p |
|---|---:|---:|---:|
| Lodging | 0.551020 [0.469388, 0.632653] | 0.064474 [0.033072, 0.103184] | `8.0043218008e-06` |
| Restaurant | 0.484375 [0.406250, 0.570312] | 0.058421 [0.029054, 0.091396] | `0.00557299776216` |

两个 domain 都有小幅高于 0.5 的 global AUROC，也都有正的 mean paired
delta；但 restaurant 的 128 个 forged 全部低于官方阈值，lodging 也只
检出两个。Domain split 不支持一个可用的 domain-specific deployment
claim。

### Visibility strata

| Visibility | Pairs | AUROC [95% CI] | AP [95% CI] | TPR@5% FPR [95% CI] | TP / FP | Paired W / L / T |
|---|---:|---:|---:|---:|---:|---:|
| Full | 173 | 0.516723 [0.511041, 0.525817] | 0.518705 [0.515413, 0.532951] | 0.057803 [0.028757, 0.075145] | 2 / 1 | 120 / 53 / 0 |
| Partial | 36 | 0.516975 [0.493056, 0.552488] | 0.527646 [0.516598, 0.587295] | 0.083333 [0.000000, 0.138889] | 0 / 0 | 23 / 13 / 0 |
| None | 66 | 0.499885 [0.498967, 0.500000] | 0.500000 [0.500000, 0.500000] | 0.045455 [0.000000, 0.045455] | 0 / 0 | 0 / 1 / 65 |

| Visibility | Strict paired ranking [95% CI] | Mean paired delta [95% CI] | Exact sign-test p |
|---|---:|---:|---:|
| Full | 0.693642 [0.624277, 0.763006] | 0.088164 [0.056598, 0.124613] | `3.78123037222e-07` |
| Partial | 0.638889 [0.472222, 0.805556] | 0.047325 [-0.013568, 0.130864] | `0.132498163963` |
| None | 0.000000 [0.000000, 0.000000] | `-7.586e-06` [`-2.276e-05`, 0] | `1.0` |

full stratum 给出了最清楚的“模型确实看到了什么”：所有 173 对都改变
feature，69.36% 的 forged 分数上升，平均上升 `0.088164`。但这些变化
多数仍发生在远低于零阈值的区间，并没有形成足够的跨场景 separation。
partial stratum 只有 36 对，interval 更宽；none stratum 基本是由完全
相同的 classifier evidence 形成的 tie。

## 10. 为什么“matched sensitivity”与“单图检测失败”能够同时成立

本结果不是逻辑矛盾，而是 paired benchmark 揭示了两个不同尺度的变化。

1. **局部 edit 确实推动了一部分分数。** 全部配对中 143 wins、67
   losses、65 ties；排除 ties 的 exact two-sided sign test 为
   `p=1.6853916320559544e-07`。full-visible 中则是 120 wins 对 53
   losses。
2. **scene identity 对绝对分数的影响更大。** 从正式 `results.jsonl`
   直接复算，matched real/forged raw logits 的 Pearson correlation 为
   `0.9915428031`，Spearman correlation 为 `0.9885678329`。real score
   的 population standard deviation 为 `1.5212821646`，约为全体平均
   paired delta `0.0616566195` 的 24.7 倍；real score 的范围跨
   `9.7440251112` logits。
3. **AUROC 使用所有跨场景 real/fake 比较。** 它不会自动减去 matched
   scene baseline。局部 edit 引起的正向 shift 在 unrelated scene 的
   content、capture 和 JPEG variation 中只留下 AUROC `0.512529` 的
   小效应。
4. **官方 calibration 在此分布上失配。** real 与 forged 平均 score
   分别为 `-4.657479` 和 `-4.595822`，都远低于零；balanced ECE 达
   `0.469640`。所以轻微的 paired shift 很少跨过 released threshold。
5. **五 crop coverage 会直接抹掉一部分 edit。** 66 个 none pair 中
   65 个 feature/logit exact tie；但即使删除这一结构性不可见部分，
   173 个 full pair 的 AUROC 也只有 `0.516723`。

因此最严谨的表述是：B-Free 在看见局部 Hunyuan 植入时比前述许多
whole-image baseline 表现出更明确的 matched response，但该 response
还不足以成为无需 matched real 的可靠 image-level detector。AUROC 的
paired-bootstrap CI 位于 0.5 以上，说明不应称其“严格随机”；然而
`+0.0125` 的 AUROC excess、5.45% 的 TPR@5% FPR、0.73% 的官方阈值
recall 和极差 calibration 同样不支持实用成功的结论。

## 11. Determinism、runtime 与独立审计

正式 run 使用一张 NVIDIA L20Z，batch size 1；Python 3.12.3、PyTorch
`2.8.0.dev20250627+cu128`、torchvision
`0.23.0.dev20250627+cu128`、timm `1.0.12`、CUDA 12.8。执行时：

- FP32、无 autocast；
- deterministic algorithms 开启；
- cuDNN 开启，benchmark 关闭，deterministic 开启；
- CUDA matmul 与 cuDNN TF32 均关闭；
- `CUBLAS_WORKSPACE_CONFIG=:4096:8`；
- model `eval()`，所有 parameter `requires_grad=False`；
- 外网禁用，模型构建阶段无网络请求。

| Runtime quantity | Value |
|---|---:|
| Model-forward latency mean | 45.264 ms/image |
| Model-forward latency median | 41.425 ms/image |
| Model-forward latency P95 | 57.898 ms/image |
| Model-forward latency max | 521.298 ms/image |
| Peak allocated CUDA memory mean | 662,416,645 bytes |
| Peak allocated CUDA memory max | 672,772,608 bytes |
| Full runner wall time | 142.674 s |
| End-to-end runner throughput | 3.855 images/s |

wall time 还包括四图 golden、Pillow decode、normalization、geometry 与
visibility evidence、hash、550 个 NPZ 写入和读回、逐行 JSONL
持久化，以及最终 metric/bootstrap；它不等同于简单累加 per-image
forward latency。

正式独立审计状态为 **`ok`**，且不把 stored runner score 或 stored
feature 当作模型输入。审计验证：

- pinned clean source、18 个关键源码/许可/demo hash；
- ZIP、config、checkpoint bytes、177-tensor schema、safe load 和
  strict full-state construction；
- canonical dataset identity、550 行 expected-input ledger、selection
  order、visibility census 和完整 physical result history；
- runtime、离线状态、T1-only adapter contract、manifest 对 outputs 的
  cryptographic binding；
- 四个官方 golden case 均在 tolerance 内，且两个 forward
  bit-identical；
- 550 次 fresh Pillow reopen、fresh preprocess 和 fresh complete
  model forward；
- 550 个 `[5,768]` feature artifact、550 个 `[5]` crop-logit
  artifact，以及从 persisted feature 对 linear head 的 550 次 replay；
- coverage、score distribution、confusion、paired ranking、sign test、
  domain/visibility strata 和全部 1,000-resample bootstrap interval。

| Replayed quantity | 最大绝对差 |
|---|---:|
| Fresh `[5,768]` features vs artifact | 0.0 |
| Fresh five crop logits vs artifact | 0.0 |
| Fresh mean raw logit vs result | 0.0 |
| Artifact linear-head replay mean logit vs result | 0.0 |
| 独立 full-model score vs result | 0.0 |

审计还确认 recorded summary 被精确重算，独立 full-model summary 在
`1e-6` tolerance 内，最大独立 score 差为 `0.0`。这证明 frozen
release 在该机器上的执行和记录忠实性；它不证明模型对当前范围之外的
数据具有外部有效性。

## 12. 范围、局限、最终判断与下一步

本 run 支持的最窄结论是：

> 固定的 B-Free DINO2reg4 官方 release 对 CLAIMFORGE Mouse 的小型
> 局部植入表现出可测的 matched-pair 正向反应，尤其是在 edit 完全进入
> 五 crop receptive fields 时；但它作为 standalone whole-image
> detector 只有 0.5125 AUROC，real-only 5% FPR 下只有 5.45% TPR，
> 官方零阈值仅检出 2/275 forged。因此它没有在该 local-splice 条件上
> 达到可用检测性能。

重要局限包括：

- 这是 canonical JPEG-Q95、两个场景 domain、一个 pinned source commit
  和唯一官方 checkpoint 的 local-splice stress test；
- 它不复现论文的 27-generator aggregate，也不反驳 B-Free 在 fully
  generated image 上的论文结果；
- 训练使用 COCO/SD 2.1 和 mixed whole/local construction，测试使用
  lodging/restaurant/Hunyuan local insertion，存在明确 domain、
  generator 和 edit-scale shift；
- 66/275 个 edit 在冻结 GT 定义下不进入五 crop union；尽管
  full-visible stratum 同样没有形成可用 standalone detection；
- 没有用 Mouse label 选择 checkpoint、修改 crop、调 threshold 或建立
  oracle calibration；
- diagnostic sigmoid、crop logits 和 features 都没有替换官方主分数，
  也没有被冒充为 T2 mask；
- 独立 replay 证明执行正确，不等于证明所有压缩、生成器、图像类别、
  edit size 或未来 release 上都得到相同结论；
- GRIP-UNINA license 不允许未授权商业/盈利使用，因此方法不能作为
  commercially cleared baseline 交付。

B-Free 完成后，计划中的 whole-image local-splice 主表为 **6/10
完成、尚余 4 个**；最小机制完备集为 **6/7 完成**。下一个方法是
**Effort**，随后是 OmniAID、LTD、CNNDetection。与此同时，同域 fully
synthetic contrast 仍未建立，因此目前仍是 **0/10 方法达到
local-splice + fully synthetic 双条件完成标准**。

## 13. 可复现产物与命令

### 正式 run

正式目录：

[`results/opensource/bfree/bfree_dino2reg4_mouse_canonical_v1_full275_20260725/`](../results/opensource/bfree/bfree_dino2reg4_mouse_canonical_v1_full275_20260725/)

关键产物：

- [`run_manifest.json`](../results/opensource/bfree/bfree_dino2reg4_mouse_canonical_v1_full275_20260725/run_manifest.json)
  — source、资产、checkpoint schema、预处理、golden、runtime、
  dataset、visibility、metric、license 和 output contract；
- [`expected_inputs.jsonl`](../results/opensource/bfree/bfree_dino2reg4_mouse_canonical_v1_full275_20260725/expected_inputs.jsonl)
  — 精确有序的 550-image input ledger；
- [`results.jsonl`](../results/opensource/bfree/bfree_dino2reg4_mouse_canonical_v1_full275_20260725/results.jsonl)
  — 每张图唯一物理 T1 结果行；
- [`artifacts/`](../results/opensource/bfree/bfree_dino2reg4_mouse_canonical_v1_full275_20260725/artifacts/)
  — 550 个 NPZ，每个含 `[5,768]` features 与 `[5]` crop logits；
- [`summary.json`](../results/opensource/bfree/bfree_dino2reg4_mouse_canonical_v1_full275_20260725/summary.json)
  — 主指标、official calibration、paired bootstrap、domain 与
  visibility strata；
- [`independent_audit.json`](../results/opensource/bfree/bfree_dino2reg4_mouse_canonical_v1_full275_20260725/independent_audit.json)
  — 550-image fresh full-model / artifact / metric replay。

正式文件 binding：

| File | SHA-256 |
|---|---|
| `expected_inputs.jsonl` | `e4cb3d6a78fa68f06341457e2234c630a455a9b6b9789e59abf45c15b292060a` |
| `results.jsonl` | `2bf1de91ffb0d2dd436909898f832e49e3badf69c9a7c927860c0dda08237382` |
| `summary.json` | `505db2bba41a259a208e9cf6dc34fa93d30bb8d7ac9e850aaa5ea87034feec44` |
| `run_manifest.json` | `85d24a99dc4be0befe29d275ab6f44b2c00796ad971838a718245ba3037c9a6d` |

### Smoke 产物

- [Smoke A](../results/opensource/bfree/bfree_dino2reg4_mouse_pair5_cuda_smoke_a_20260725/)
- [Smoke B](../results/opensource/bfree/bfree_dino2reg4_mouse_pair5_cuda_smoke_b_20260725/)
- [Hardened smoke C](../results/opensource/bfree/bfree_dino2reg4_mouse_pair5_cuda_smoke_c_hardened_20260725/)

### 可复用代码

- [`eval/opensource/run_bfree.py`](../eval/opensource/run_bfree.py)
- [`eval/opensource/bfree_metrics.py`](../eval/opensource/bfree_metrics.py)
- [`eval/opensource/analyze_bfree_run.py`](../eval/opensource/analyze_bfree_run.py)
- [`tests/test_run_bfree.py`](../tests/test_run_bfree.py)
- [`tests/test_bfree_metrics.py`](../tests/test_bfree_metrics.py)
- [`tests/test_analyze_bfree_run.py`](../tests/test_analyze_bfree_run.py)

先执行不计算 Mouse score 的官方 golden / provenance preflight：

```bash
PYTHONPATH=. /root/.cache/claimforge/venvs/bfree/bin/python \
  -m eval.opensource.run_bfree \
  --preflight-only \
  --device cpu
```

已有正式 run ID 的目录不能无条件覆盖。下面的 `--resume` 会严格验证
config、资产和 input ledger，并跳过已经成功的 550 行；它用于安全恢复，
**不会重新推理成功行**：

```bash
CUDA_VISIBLE_DEVICES=4 \
PYTHONPATH=. /root/.cache/claimforge/venvs/bfree/bin/python \
  -m eval.opensource.run_bfree \
  --run-id bfree_dino2reg4_mouse_canonical_v1_full275_20260725 \
  --device cuda:0 \
  --bootstrap-samples 1000 \
  --bootstrap-seed 20260724 \
  --fail-fast \
  --resume
```

若要真正重新执行模型，必须使用新的 immutable run ID：

```bash
CUDA_VISIBLE_DEVICES=4 \
PYTHONPATH=. /root/.cache/claimforge/venvs/bfree/bin/python \
  -m eval.opensource.run_bfree \
  --run-id bfree_dino2reg4_mouse_canonical_v1_full275_rerun_<YYYYMMDD> \
  --device cuda:0 \
  --bootstrap-samples 1000 \
  --bootstrap-seed 20260724 \
  --fail-fast
```

对现有正式 run 重新做独立审计时，可把新审计写入临时文件，避免覆盖
冻结的 `independent_audit.json`：

```bash
CUDA_VISIBLE_DEVICES=4 \
PYTHONPATH=. /root/.cache/claimforge/venvs/bfree/bin/python \
  -m eval.opensource.analyze_bfree_run \
  --run-id bfree_dino2reg4_mouse_canonical_v1_full275_20260725 \
  --device cuda:0 \
  --output /tmp/bfree_independent_audit_rerun.json
```

新 canonical 数据到达时，通过 `--dataset-manifest` 指向新 manifest，
并使用新的 immutable run ID。runner 会拒绝 source、checkpoint、
adapter、runtime、selection 或 metric contract 漂移，已经成功的行可
安全 resume；analyzer 可以随后对新 run 进行同样的全部图像 fresh
replay。
