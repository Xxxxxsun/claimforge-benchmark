# CNNDetection Blur+JPEG(0.1) native 在 Mouse full275 上的正式结果

日期：2026-07-25（UTC）

正式 run：
`cnndetection_blur_jpg_prob0_1_native_mouse_canonical_v1_full275_20260725`

核心证据：
[run manifest](../results/opensource/cnndetection/cnndetection_blur_jpg_prob0_1_native_mouse_canonical_v1_full275_20260725/manifest.json)、
[逐图结果](../results/opensource/cnndetection/cnndetection_blur_jpg_prob0_1_native_mouse_canonical_v1_full275_20260725/results.jsonl)、
[正式汇总](../results/opensource/cnndetection/cnndetection_blur_jpg_prob0_1_native_mouse_canonical_v1_full275_20260725/summary.json)、
[独立重放审计](../results/opensource/cnndetection/cnndetection_blur_jpg_prob0_1_native_mouse_canonical_v1_full275_20260725/independent_audit.json)。

## 1. 结论摘要

CNNDetection 的官方 Blur+JPEG(0.1) checkpoint 已按预先冻结的 native
协议跑完全部 Mouse canonical v1：

- 覆盖为 550/550 张图、275/275 个 real/forged 完整配对；
- 有效图 550，错误、缺失、未配对图和重复物理结果行均为 0；
- 图像级 AUROC 为 **0.498896**，任务配对 bootstrap 95% CI 为
  **[0.497401, 0.500046]**；
- average precision 为 **0.502089**，95% CI 为
  **[0.501066, 0.509603]**；
- TPR@5%FPR 为 **0.047273**，95% CI 为
  **[0.036364, 0.058182]**；
- 发布规则 `float32 sigmoid(raw_logit) > 0.5` 的混淆矩阵为
  `TP/FP/FN/TN = 0/0/275/275`，即 **0/275 forged 被检出**；
- 275 个配对的阈值判定全部为 `real 0 → forged 0`，没有任何翻转；
- forged 分数严格高于其配对 real 的任务只有 125/275，严格配对排序
  准确率为 **0.454545**，精确双侧符号检验 `p = 0.147691`；
- forged-real 平均分差为 **-7.381589e-7**，95% CI
  **[-2.581092e-6, 5.426804e-7]**。

因此，这个 frozen primary 在当前 Mouse 条件下没有可用的整图检测能力：
全局排序与随机基本一致，发布阈值把所有图都判为 real，配对方向也没有显著
优势。这个结论只针对“真实照片中局部植入 Mouse”的压力测试，不能外推为
CNNDetection 对其原始任务——整幅 CNN 生成图像——无效。

预登记的论文时期 CenterCrop224 sensitivity 也已完成，AUROC 为
**0.499702**。但该 crop 将 247/275 个编辑完全排除、只完整保留 14 个；
它用于解释预处理歧义，不能替换 native primary 或作为更优结果选择。

CNNDetection 只输出一个整图 logit，没有原生 dense map、mask 或 bbox。
本 run 只对 **T1 whole-image AIGC detection** 有效；**T2 localization
和 joint score 均为 N/A**。

## 2. 方法、官方来源与原理

方法来自 CVPR 2020 论文
[CNN-Generated Images Are Surprisingly Easy to Spot... for Now](https://openaccess.thecvf.com/content_CVPR_2020/html/Wang_CNN-Generated_Images_Are_Surprisingly_Easy_to_Spot..._for_Now_CVPR_2020_paper.html)。
本次固定作者的
[官方仓库](https://github.com/PeterWang512/CNNDetection)
commit：

```text
ea0b5622365e3a9cd31d1b54b6b5971131a839ab
```

用于核对论文时期核心推理实现的稳定 commit 为：

```text
f692c138482137c92280c01a45ae190379f16790
```

manifest 记录的结论是：当前固定 commit 与论文时期 commit 的核心推理代码
字节一致；正式运行时官方 checkout 的 tracked 文件为 clean。

### 2.1 原理

CNNDetection 的基本思路不是先指定某一种肉眼可见的伪影，而是让一个
ResNet-50 二分类器从大量 ProGAN fake 与 LSUN real 图像中学习可跨类别
复用的生成痕迹。原始训练覆盖 20 个物体类别，希望模型学到生成过程的共同
统计特征，而不是记住单一语义类别。

Blur+JPEG 训练增强是其泛化设计的重要部分。训练时，以给定概率对样本施加
随机模糊和 JPEG 压缩，使分类器不能只依赖特别脆弱、经过常见图像处理就会
消失的高频捷径。本 checkpoint 名称中的 `0.1` 指训练时 blur/JPEG 增强的
概率配置；**正式测试没有再给输入做 blur 或 JPEG 增强**。

本 run 的实际前向为：

```text
Pillow decode -> RGB
-> native resolution, no resize, no crop
-> float32 ToTensor
-> ImageNet mean/std normalization
-> official vendored ResNet-50
-> adaptive global average pooling
-> 2048-dimensional pooled feature
-> Linear(2048, 1)
-> one float32 raw logit
-> float32 sigmoid
-> strict score > 0.5
```

这种设计在“整幅图由 CNN generator 生成”的原始场景中有合理优势：生成
痕迹可遍布全图，global average pooling 可以汇总大量局部证据；训练增强
又提高了这些证据对模糊和压缩的鲁棒性。

Mouse 的 forged 图则保留绝大多数真实照片，只替换一个很小的局部区域。
在这种条件下，局部信号可能被整图平均池化稀释，且 Mouse 使用的生成分布、
真实底图和 JPEG 合同也不同于原始 ProGAN/LSUN 训练设置。正式结果与这些
机制相容，但当前单次 benchmark 不能单独识别究竟是哪一种因素导致失败，
因此不作更强的因果断言。

## 3. 为什么 primary 冻结为 0.1，而不是 0.5

官方同时发布 Blur+JPEG(0.1) 和 Blur+JPEG(0.5) 两个 checkpoint。README
quick-start 示例目前使用 0.5，但论文把 0.1 描述为性能与增强之间的良好
平衡，并在跨生成器结果、StyleGAN2、ranking 和 calibration 实验中采用
该模型；manifest 记录的论文跨生成器 mean AP 为 92.6。

因此，在查看 Mouse 模型分数之前，本 benchmark 将
**Blur+JPEG(0.1) 冻结为 primary**。这不是在 0.1 与 0.5 的 Mouse 结果中
择优：

```text
checkpoint_selection_frozen_before_mouse_scores: true
primary:  Blur+JPEG(0.1)
excluded primary variant: Blur+JPEG(0.5)
exclusion reason: README quick-start default only;
                  not selected through post-hoc Mouse performance
```

本报告所有 primary 数字只来自 0.1。没有把 0.5 跑出的数字混入、替换或
辅助选择 primary。若将来评测 0.5，它必须使用不同 run ID，作为明确的
checkpoint sensitivity 单独报告。

## 4. Checkpoint 门禁与加载

正式 checkpoint 来自官方 `weights/download_weights.sh` 指向的 Dropbox：

| 字段 | 固定值 |
|---|---|
| ID | `CNNDetection-BlurJPEG0.1@official-dropbox` |
| 文件 | `blur_jpg_prob0.1.pth` |
| 字节数 | 282,442,597 |
| SHA-256 | `a73295ac66f9cb74d558ce3ade46f75e2f2997ed05eeed0f4b774623372058ea` |
| 官方是否发布 digest | 否 |
| 本地格式 | legacy `torch.save` nested training checkpoint |
| 外层 keys | `model`, `optimizer`, `total_steps` |
| model state entries | 320 |
| model state elements | 23,563,254 |
| trainable parameters | 23,510,081 |
| optimizer state entries | 161 |
| total steps | 270,048 |
| state payload SHA-256 | `8c62f887d5b97a0337f0ed598ac80cb9d86929613d3bc5c08fb0331b470c8931` |

模型图按官方 vendored ResNet-50 构建，分类头为 `Linear(2048, 1)`；
state dict 使用 `strict=True` 加载。checkpoint 通过
`torch.load(..., weights_only=True, map_location="cpu")` 成功读取，
没有使用 unrestricted pickle。PyTorch 的静态 unsafe-global scanner
不支持该 legacy stream，所以不能把“scanner 不支持”误写成“静态扫描
为空”；安全证据是 `weights_only=True` 实际加载成功，加上严格 schema、
形状、元素数与 payload hash 门禁。

由于官方没有发布 checkpoint digest，本报告中的 SHA-256 是本 benchmark
对实际下载字节的冻结标识，不应表述为作者发布的校验值。

## 5. License 与商用边界

固定仓库包含 `LICENSE.txt`，内容为
**CC BY-NC-SA 4.0（Attribution-NonCommercial-ShareAlike）**。该许可包含
NonCommercial 条款，不是 OSI open-source license。

未发现 checkpoint 的独立授权文本，因此 manifest 对权重保守沿用仓库的
非商用边界：

```text
repository_license: CC-BY-NC-SA-4.0
commercial_use_permitted: false
checkpoint_separate_terms_found: false
checkpoint_commercial_clearance_established: false
overall_commercial_clearance: not_established
```

本次仅作研究 benchmark 记录，不构成许可意见或授权。不能据此认定代码或
权重可用于商业部署；需要商用时，应另行取得权利人的明确授权。

## 6. Frozen primary 推理协议

正式协议在 full275 分数产生前冻结：

| 组件 | 正式值 |
|---|---|
| 数据集 | `claimforge-mouse-good275-canonical-jpeg-q95-v1` |
| 任务 / 图像 | 275 个 matched tasks / 550 张图 |
| checkpoint | 官方 Blur+JPEG(0.1)，SHA-256 `a73295ac...058ea` |
| profile | `official_recommended_native_rgb_no_resize_no_crop` |
| 解码 | `Pillow.Image.open(...).convert("RGB")` |
| EXIF | 不做 EXIF transpose |
| 几何 | 原生宽高，不 resize，不 crop |
| tensor | float32，`ToTensor` |
| normalization mean | `[0.485, 0.456, 0.406]` |
| normalization std | `[0.229, 0.224, 0.225]` |
| batch size | 1 |
| test-time blur/JPEG | 无 |
| model mode | `eval()` |
| precision | float32，无 autocast |
| primary score | `torch.sigmoid(raw_logit)` 的 float32 输出 |
| score 语义 | uncalibrated fake score，越大越像 fake |
| 发布判定 | strict `score > 0.5` |
| 5% FPR 阈值 | real score 的 95th percentile，`method="higher"`，strict `>` |
| bootstrap | 1,000 次完整 task-pair 重采样，seed `20260724` |
| T2 | N/A；模型无原生定位输出 |

README 明确把单图输出称为 uncalibrated prediction。虽然 sigmoid 值位于
`[0,1]`，本报告只称其为“未校准 fake score”或“sigmoid score”，不将它
解释为目标域中的校准概率。

官方 2020-06 更新推荐在 batch size 1 下评测 uncropped images，并报告其在
多数官方类别上优于 224 crop；官方 `demo.py` 也直接处理输入原始尺寸。
因此 native no-resize/no-crop 是本次 primary。论文时期的
native `CenterCrop(224)` 已在 full275 前登记为 sensitivity，但不参与
primary 选择，见第 12 节。

## 7. 数据范围、coverage 与可见性

输入账本固定为：

```text
dataset_id: claimforge-mouse-good275-canonical-jpeg-q95-v1
tasks:      275
images:     550
lodging:    147 pairs / 294 images
restaurant: 128 pairs / 256 images
inputs SHA-256:
e4cb3d6a78fa68f06341457e2234c630a455a9b6b9789e59abf45c15b292060a
```

Coverage 审计如下：

| 字段 | 数值 |
|---|---:|
| expected images | 550 |
| physical result rows | 550 |
| result images | 550 |
| valid images | 550 |
| error images | 0 |
| missing images | 0 |
| expected complete pairs | 275 |
| complete valid pairs | 275 |
| paired valid images | 550 |
| unpaired valid images | 0 |
| coverage / valid fraction | 1.0 / 1.0 |

primary 不裁剪，所以所有 275 个 forged GT 区域均完整位于模型输入画布中：
`edit_visible_gt_fraction` 的 min、P05、median、mean、P95、max 全为
`1.0`，visibility census 为 `full: 275`。

这里的 `edit_visibility = full` 仅表示**局部编辑区域没有被预处理裁掉**；
它不表示 forged 图是“完全生成的整图”。每张 forged 仍然是现实底图上的
Mouse 局部植入。

## 8. Primary 图像级结果

下列置信区间均为 1,000 次完整 task-pair bootstrap 的 95% percentile
区间。

| Metric | Estimate | 95% CI |
|---|---:|---:|
| AUROC | 0.498896 | [0.497401, 0.500046] |
| Average precision | 0.502089 | [0.501066, 0.509603] |
| TPR @ target FPR 5% | 0.047273 | [0.036364, 0.058182] |
| Real-only 5% FPR score threshold | `1.445300e-5` | [`4.172020e-6`, `5.428998e-5`] |
| Actual FPR | 0.047273 | [0.036364, 0.047273] |
| Accuracy @ 0.5 | 0.500000 | [0.500000, 0.500000] |
| Balanced accuracy @ 0.5 | 0.500000 | [0.500000, 0.500000] |
| Precision @ 0.5 | 0.000000 | [0.000000, 0.000000] |
| Recall @ 0.5 | 0.000000 | [0.000000, 0.000000] |
| F1 @ 0.5 | 0.000000 | [0.000000, 0.000000] |
| Specificity @ 0.5 | 1.000000 | [1.000000, 1.000000] |

发布阈值的混淆矩阵：

| Rule | TP | FP | FN | TN |
|---|---:|---:|---:|---:|
| strict `sigmoid score > 0.5` | 0 | 0 | 275 | 275 |

所有 550 个 sigmoid score 都低于 0.001，更远低于发布阈值 0.5：

| Kind | Min | Mean | Median | P05 | P95 | Max |
|---|---:|---:|---:|---:|---:|---:|
| Real | `1.761151e-15` | `9.813736e-6` | `8.329596e-9` | `1.026233e-12` | `1.404023e-5` | `9.634749e-4` |
| Forged | `1.794222e-15` | `9.075577e-6` | `7.973332e-9` | `1.055329e-12` | `1.318702e-5` | `7.414150e-4` |

这不是“阈值刚好偏高”造成的轻微误差：即使完全忽略固定阈值、只看连续分数
的全局排序，AUROC 仍为 0.498896。TPR@5%FPR 也只有 4.73%。因此不能仅靠
把 0.5 调低就声称得到一个可泛化的检测器；在 Mouse 目标域上拟合新阈值还
会违反本 benchmark 禁止 target-domain calibration 的协议。

## 9. 配对结果

每个 task 的 real 与 forged 只相差 Mouse 局部植入，因此配对统计可以减少
跨场景差异。不过，这种分析只用于诊断局部变化敏感性；实际单图检测时没有
对应的未编辑 real 作为参照。

| Paired metric | Estimate | 95% CI |
|---|---:|---:|
| Strict ranking `forged score > real score` | 0.454545 | [0.396364, 0.512727] |
| Mean forged-real score delta | `-7.381589e-7` | [`-2.581092e-6`, `5.426804e-7`] |

配对明细：

```text
wins / losses / ties: 125 / 150 / 0
exact two-sided sign-test p: 0.14769147782971262
score delta min:    -2.220598e-4
score delta median: -1.615843e-13
score delta mean:   -7.381589e-7
score delta P05:    -2.668686e-7
score delta P95:     3.361443e-8
score delta max:     6.869674e-5
```

阈值判定转移为：

| Real decision → forged decision | Pairs |
|---|---:|
| `0 → 0` | 275 |
| `0 → 1` | 0 |
| `1 → 0` | 0 |
| `1 → 1` | 0 |

同一 task 内 real 与 forged 的分数高度相关：

| Pair correlation | Value |
|---|---:|
| Pearson, sigmoid score | 0.987167 |
| Spearman, sigmoid score | 0.999742 |
| Pearson, raw logit | 0.999777 |
| Spearman, raw logit | 0.999742 |

这些相关性和 275/275 不翻转说明，局部植入前后的输出几乎保持原场景排序。
它们与“底图内容贡献远大于小面积改动”相容，但相关性本身不证明具体因果
机制。

## 10. 分域结果

### 10.1 图像级指标

| Domain | Pairs | AUROC [95% CI] | AP [95% CI] |
|---|---:|---:|---:|
| Lodging | 147 | 0.497941 [0.494562, 0.500116] | 0.503846 [0.501596, 0.516905] |
| Restaurant | 128 | 0.499817 [0.496582, 0.502563] | 0.503836 [0.502469, 0.518903] |

| Domain | TPR@5%FPR [95% CI] | Score threshold [95% CI] | Actual FPR [95% CI] |
|---|---:|---:|---:|
| Lodging | 0.054422 [0.027211, 0.061224] | `8.802715e-5` [`1.386333e-5`, `2.535591e-4`] | 0.047619 [0.027211, 0.047619] |
| Restaurant | 0.046875 [0.023438, 0.070312] | `2.368734e-6` [`9.973672e-7`, `4.172020e-6`] | 0.046875 [0.031055, 0.046875] |

两个 domain 在发布阈值下都把全部样本判为 real：

| Domain | TP | FP | FN | TN |
|---|---:|---:|---:|---:|
| Lodging | 0 | 0 | 147 | 147 |
| Restaurant | 0 | 0 | 128 | 128 |

### 10.2 分域配对指标

| Domain | Wins / losses / ties | Strict rank [95% CI] | Sign-test p |
|---|---:|---:|---:|
| Lodging | 64 / 83 / 0 | 0.435374 [0.353741, 0.517007] | 0.137378 |
| Restaurant | 61 / 67 / 0 | 0.476562 [0.390625, 0.562500] | 0.658701 |

| Domain | Mean score delta [95% CI] | Median | Min | Max |
|---|---:|---:|---:|---:|
| Lodging | `-1.377648e-6` [`-4.606028e-6`, `1.021803e-6`] | `-7.761694e-12` | `-2.220598e-4` | `6.869674e-5` |
| Restaurant | `-3.745763e-9` [`-1.466325e-8`, `7.982137e-9`] | `-7.531166e-15` | `-3.776332e-7` | `3.597966e-7` |

lodging 与 restaurant 的独立图像 AUROC 都接近 0.5，配对区间也都覆盖
0.5；本结果没有支持某一个 domain 存在稳定可用的检测能力。

## 11. Raw-logit 数值诊断

raw logit 是 sigmoid 前的官方单标量输出。该诊断始终与发布 score 并列
报告，但**不替换**正式 sigmoid score、0.5 阈值或判定。

本 run 没有 sigmoid underflow 到精确 0：550 个 score 全部为有限正数。
raw logit 与 float32 sigmoid 各有 545 个不同值。由于 sigmoid 在当前
数值范围内保持单调，raw-logit 的全局 AUROC、AP、TPR@5%FPR 和严格配对
排序与 sigmoid score 相同：

| Raw-logit metric | Estimate |
|---|---:|
| AUROC | 0.498896 |
| Average precision | 0.502089 |
| TPR @ target FPR 5% | 0.047273 |
| Real-only raw threshold | -11.144594 |
| Actual FPR | 0.047273 |
| Strict paired ranking | 0.454545 |
| Accuracy at strict `logit > 0` | 0.500000 |
| Balanced accuracy at strict `logit > 0` | 0.500000 |

`logit > 0` 与 `sigmoid(logit) > 0.5` 等价，所以 raw threshold 0 下仍为
`TP/FP/FN/TN = 0/0/275/275`。

Raw-logit 分布：

| Kind | Min | Mean | Median | P05 | P95 | Max |
|---|---:|---:|---:|---:|---:|---:|
| Real | -33.972809 | -18.935806 | -18.603451 | -27.605369 | -11.173753 | -6.944000 |
| Forged | -33.954205 | -18.952442 | -18.647163 | -27.579251 | -11.236622 | -7.206208 |

Raw-logit 配对差：

```text
count:  275
min:   -0.630311966
P05:   -0.203690624
median:-0.007725716
mean:  -0.016635701
P95:    0.114140129
max:    0.493688583
wins / losses / ties: 125 / 150 / 0
```

分域 raw-logit 诊断：

| Domain | Raw AUROC | Raw AP | TPR@5%FPR | Raw threshold | Mean paired delta |
|---|---:|---:|---:|---:|---:|
| Lodging | 0.497941 | 0.503846 | 0.054422 | -9.337777 | -0.026077 |
| Restaurant | 0.499817 | 0.503836 | 0.046875 | -12.953153 | -0.005793 |

raw view 没有揭示被 sigmoid 数值饱和掩盖的额外排序能力；它确认失败发生在
logit 排序和绝对偏置层面，而不只是 score 表示层面。

## 12. CenterCrop224 sensitivity

论文时期的 `paper_native_center_crop224_no_resize` profile 已在查看 full275
结果前登记为 sensitivity：

```text
Pillow RGB
-> no resize
-> torchvision center_crop(224)
-> ToTensor
-> ImageNet normalization
-> the same Blur+JPEG(0.1) checkpoint
```

它使用同一个 Blur+JPEG(0.1) checkpoint 和同一份 550-image input ledger，
只改变 crop profile。它**不会替换第 6 节的 native primary，也没有用于
二选一择优汇报**。

Sensitivity run：

```text
cnndetection_blur_jpg_prob0_1_crop224_mouse_canonical_v1_
full275_sensitivity_20260725
```

机器证据：
[crop manifest](../results/opensource/cnndetection/cnndetection_blur_jpg_prob0_1_crop224_mouse_canonical_v1_full275_sensitivity_20260725/manifest.json)、
[crop results](../results/opensource/cnndetection/cnndetection_blur_jpg_prob0_1_crop224_mouse_canonical_v1_full275_sensitivity_20260725/results.jsonl)、
[crop summary](../results/opensource/cnndetection/cnndetection_blur_jpg_prob0_1_crop224_mouse_canonical_v1_full275_sensitivity_20260725/summary.json)、
[crop replay audit](../results/opensource/cnndetection/cnndetection_blur_jpg_prob0_1_crop224_mouse_canonical_v1_full275_sensitivity_20260725/independent_audit.json)。

### 12.1 Coverage 与 aggregate 指标

Coverage 仍为 550/550 有效图、275/275 完整配对，error、missing 和
unpaired 均为 0。主要结果：

| Metric | Crop224 estimate | 95% paired-bootstrap CI |
|---|---:|---:|
| AUROC | 0.499702 | [0.497732, 0.501719] |
| Average precision | 0.499548 | [0.497783, 0.501985] |
| TPR @ target FPR 5% | 0.043636 | [0.032727, 0.054545] |
| Real-only 5% FPR threshold | `8.856411e-4` | [`2.558512e-4`, `2.906959e-3`] |
| Actual FPR | 0.047273 | [0.036364, 0.047273] |
| Accuracy @ 0.5 | 0.500000 | [0.500000, 0.500000] |
| Balanced accuracy @ 0.5 | 0.500000 | [0.500000, 0.500000] |
| Recall @ 0.5 | 0.007273 | [0.000000, 0.018182] |
| Specificity @ 0.5 | 0.992727 | [0.981818, 1.000000] |

发布阈值下，`TP/FP/FN/TN = 2/2/273/273`。这 4 个 positive decision
来自 2 个配对的 `1 → 1`；其余 273 对均为 `0 → 0`，仍然没有任何
real/forged 判定翻转。检出 2 张 forged 的同时也误报 2 张 real，不能视为
相对 native primary 的实质改进。

配对统计为：

```text
wins / losses / ties: 15 / 14 / 246
strict paired ranking over all 275 pairs: 0.054545
95% CI: [0.029091, 0.083636]
non-ties: 29
exact two-sided sign-test p: 1.0
mean forged-real score delta: -5.984756e-6
95% CI: [-2.013176e-5, 4.358079e-6]
median delta: 0.0
```

这里的 `0.054545` 不能解释成“模型以 5.45% 的准确率稳定反向排序”：
指标定义是 `forged > real` 的 strict wins 除以全部 275 对，246 个相等
score 也计入分母但不计为 win。只看 29 个 non-ties 是 15 win、14 loss，
符号检验 `p=1.0`，不存在方向优势。

### 12.2 Crop 后的编辑可见性

CenterCrop224 与 native primary 的关键差异不是简单的输入尺寸变化，而是
它会直接移除大多数 Mouse 编辑：

| Visibility | Pairs | Fraction of 275 |
|---|---:|---:|
| None | 247 | 89.82% |
| Partial | 14 | 5.09% |
| Full | 14 | 5.09% |

275 个 GT 的 visible fraction 统计：

```text
min:    0.0
P05:    0.0
median: 0.0
mean:   0.072015558
P95:    0.979471154
max:    1.0
```

按 visibility 分层：

| Visibility | Pairs | AUROC | AP | Wins / losses / ties | Strict rank | Mean score delta | Sign p |
|---|---:|---:|---:|---:|---:|---:|---:|
| Full | 14 | 0.489796 | 0.502962 | 7 / 7 / 0 | 0.500000 | `-1.362771e-7` | 1.0 |
| Partial | 14 | 0.494898 | 0.517496 | 7 / 7 / 0 | 0.500000 | `-1.174214e-4` | 1.0 |
| None | 247 | 0.500008 | 0.500005 | 1 / 0 / 246 | 0.004049 | `6.182147e-18` | 1.0 |

full-visible 与 partial-visible 两个很小的 14-pair strata 都是 7 win、
7 loss；它们不支持稳定的局部变化方向。none stratum 中 246/247 对产生
完全相等的 score，直接解释了 aggregate 的大量 ties。剩余 1 对虽在 GT
几何上归为 none，但 crop 后 RGB hash 不同，score 差仅
`1.526990e-15`；不能据此声称模型看到了已被 crop 排除的 Mouse 区域。

### 12.3 Native primary 与 Crop224 的正确关系

| Protocol | AUROC | AP | TPR@5%FPR | TP / FP | Visible full / partial / none |
|---|---:|---:|---:|---:|---:|
| Native primary | 0.498896 | 0.502089 | 0.047273 | 0 / 0 | 275 / 0 / 0 |
| Crop224 sensitivity | 0.499702 | 0.499548 | 0.043636 | 2 / 2 | 14 / 14 / 247 |

两条协议的 AUROC 都约为 0.5。Crop224 不是更好的替代方案，而且由于
89.82% 的编辑完全不在 crop 内，它还是一个对局部植入覆盖很差的协议。
它的价值是记录并排除论文时期预处理歧义，而不是为 primary 寻找更好数字。

Crop224 的模型推理段平均为 6.373 ms/image，中位数 4.098 ms，P95
14.477 ms；peak allocated CUDA memory 最大为 138,881,024 bytes。它比
native 更快、更省显存与固定的 224×224 输入相符，但 runtime 差异不能
补偿编辑覆盖丢失。

Crop audit 的 schema/status 为
`cnndetection_replay_audit_v1 / replay_audit_passed`；550/550 张图全部
重新前向，2,048 维 feature replay 最大绝对差 `0.0`，summary 使用共享
metrics 复算通过。与 primary 审计相同，它不是完全独立的第二套统计实现：
`fully_independent_statistical_implementation = false`。

## 13. 双 smoke、确定性与正式重放审计

full275 前执行了两个独立的 final-code CUDA pair5 smoke：

```text
cnndetection_blur_jpg_prob0_1_native_mouse_pair5_cuda_smoke_a_20260725
cnndetection_blur_jpg_prob0_1_native_mouse_pair5_cuda_smoke_b_20260725
```

两个 smoke 都为 10/10 有效图、5/5 完整配对、0 error。A 与 B 的以下内容
逐图完全相同：

- 输入 ID、输入 SHA-256；
- decoded RGB hash 与 normalized tensor hash；
- 2,048 维 feature 文件字节与 feature SHA-256；
- raw logit，最大绝对差 `0.0`；
- float32 sigmoid score，最大绝对差 `0.0`；
- classification decision；
- config fingerprint。

smoke A/B 与 full run 前 10 个有序样本的 RGB、normalized tensor、
feature、raw logit、score 和 decision 也完全相同；smoke 与 full 的
config fingerprint 按设计不同，因为 pair limit/bootstrap 合同不同。
原始 `results.jsonl` 和 `summary.json` 文件 hash 不应相同，因为其中包含
真实执行时间和 latency。

正式审计文件记录：

```text
schema_version: cnndetection_replay_audit_v1
status: replay_audit_passed
expected_images: 550
physical_result_rows: 550
latest_result_rows: 550
successful_images_replayed: 550
feature_replay_max_abs_difference: 0.0
summary_recomputed_with_shared_metrics: true
localization_claims_rejected: true
```

审计从固定 source/checkpoint 重新解码、预处理并前向 550 张图，重放全部
2,048 维 feature；feature 最大绝对差为 `0.0`。它也检查逐图 logit/score
在冻结容差内，并使用共享的、已测试 metrics 实现复算 summary。

边界需要明确：审计的
`fully_independent_statistical_implementation = false`。它是独立的完整
模型重放与 artifact 一致性审计，但统计复算复用了正式 metrics 模块，不能
声称为第二套完全独立开发的统计实现。

## 14. Runtime 与显存

正式运行环境：

```text
device:      NVIDIA L20Z, cuda:0
CUDA:        12.8
Python:      3.12.3
PyTorch:     2.8.0.dev20250627+cu128
torchvision: 0.23.0.dev20250627+cu128
Pillow:      11.1.0
NumPy:       2.2.6
dtype:       float32
autocast:    false
batch size:  1
deterministic algorithms: true
cuDNN benchmark: false
cuDNN deterministic: true
TF32: false
```

模型推理段 latency：

| Statistic | ms / image |
|---|---:|
| Min | 4.192 |
| Mean | 24.911 |
| Median | 23.096 |
| P05 | 5.103 |
| P95 | 40.667 |
| Max | 721.075 |

该段包含把单图 tensor 送到目标 device、模型前向与 CUDA 同步。最大值包含
首张图的 warm-up/cold-start 影响，因此稳定吞吐更应参考 median 与 P95，
而不是只看 max。

单独计时的 decode、RGB 转换、native tensor 构造和 normalization：

| Statistic | ms / image |
|---|---:|
| Min | 5.857 |
| Mean | 122.554 |
| Median | 127.152 |
| P05 | 34.797 |
| P95 | 166.681 |
| Max | 844.705 |

逐图 preprocessing 加模型段的和平均为 **147.465 ms/image**，中位数
**150.365 ms/image**，P95 **197.607 ms/image**。这仍不包含全部 GT
evidence、hash、feature 落盘、summary/bootstrap 与启动开销。

manifest 的 wall-clock 为
`22:29:13.881128` 至 `22:31:19.151348`，即 **125.270 秒**。

每图 reset 后记录的 peak allocated CUDA memory：

| Statistic | Bytes | MiB |
|---|---:|---:|
| Min | 164,329,472 | 156.717 |
| Mean | 482,471,404 | 460.121 |
| Median | 596,706,304 | 569.063 |
| P95 | 663,191,552 | 632.469 |
| Max | 682,621,440 | 650.999 |

该值包含常驻模型 allocation，不是 checkpoint 文件大小，也不是设备总显存。

## 15. Hash 与机器可复核产物

### 15.1 正式 run

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| `expected_inputs.jsonl` | 491,138 | `e4cb3d6a78fa68f06341457e2234c630a455a9b6b9789e59abf45c15b292060a` |
| `results.jsonl` | 2,174,788 | `8c980673e30fd61d3565c681debb5c754297ee7b8fe503a51ea5707c9ff773dd` |
| `summary.json` | 20,290 | `ef5fea1e9becd9a3bdcb4e9bf2afcc68e7d077647ba5a12027b8230a89d5f902` |
| `manifest.json` | 36,427 | `1949250c5d2c1c2e19d1938740d889fb685e445d26ec141cbd6644c7a584853d` |
| `independent_audit.json` | 2,282 | `b61e19ec7f437c5ebb9bb06bd93e4a02679bc339d2aa13cf88147885702f471f` |
| checkpoint | 282,442,597 | `a73295ac66f9cb74d558ce3ade46f75e2f2997ed05eeed0f4b774623372058ea` |

其他固定标识：

```text
full config fingerprint:
91e519ac2f4ccddff4f2ed013ca24bca283a2e43f4515ab79335f5ba0205ea5a

checkpoint state payload SHA-256:
8c62f887d5b97a0337f0ed598ac80cb9d86929613d3bc5c08fb0331b470c8931

feature files: 550
feature shape: [2048], dtype float32
total feature bytes: 4,576,000
sorted sha256sum-ledger digest:
d1a61af861b4367c77126b528538a41c651ae700e0e1ed533c0f95194da992fc

results-order concatenated .npy-file SHA-256:
608edbd6b1e7f59e37d1ddf49c30221cc37f3411be81f021f80681ae4dce8c47

results-order concatenated raw float32 payload SHA-256:
f6c6c418b9ff73360ae0cf770bdbc4fa691862fa6d04a5e3f3e09395a5b1a5b2
```

### 15.2 Crop224 sensitivity

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| `expected_inputs.jsonl` | 491,138 | `e4cb3d6a78fa68f06341457e2234c630a455a9b6b9789e59abf45c15b292060a` |
| `results.jsonl` | 2,248,613 | `39c5e46217aad84c128f521d3b9a0894e1e56e573c0ec4cec22f0713c0d0fce4` |
| `summary.json` | 27,925 | `a0a1404d52ae90ab609f86011241732ad85058e21b024d03fa109a2e89e34e6b` |
| `manifest.json` | 36,550 | `6ea13047b5c819f19858f1d0d0b214591ad4a381da52927b05b3cc2ec480e26c` |
| `independent_audit.json` | 2,283 | `0ce484e5268a3fc8b6bd8e000f00e2071d649a87aa33ee57b6f0570304764d28` |

Crop224 config fingerprint：

```text
773171e34b1ce1306a069fd69b0ed49557e8a89ec173750fe461adf47136e788
```

### 15.3 双 smoke

| Run | Artifact | SHA-256 |
|---|---|---|
| smoke A | `expected_inputs.jsonl` | `d18681c46babc0f0e4e2ab1811b8cf8e6bba38fcdf3c8276e5b840fb8990efef` |
| smoke A | `results.jsonl` | `023ccde0c32459f369afecbbfa28a29075b5ad307bd6578e02ae12138101e03f` |
| smoke A | `summary.json` | `eedf72b39e223079e98fa5f76a8f1014c350f4d9dc63079ce9ead37b0e46f028` |
| smoke A | `manifest.json` | `daa47ee905dff44763cdc8f656b2357656600bc0d763abc46788dfe82942668b` |
| smoke B | `expected_inputs.jsonl` | `d18681c46babc0f0e4e2ab1811b8cf8e6bba38fcdf3c8276e5b840fb8990efef` |
| smoke B | `results.jsonl` | `d2543d14cb7218933be02a67db9734fb52e5bbeda925c3e13b8cf899410d424b` |
| smoke B | `summary.json` | `ca73a279aa4c7a57916142a1bc0268bf6d55b76eb8536ab54b4747f72b893ad5` |
| smoke B | `manifest.json` | `196b2a02c7b0b6bbe8916b68a1a2a8ef2bfdf8c138f4243f9c4e66aac71ee664` |

两个 smoke 的共同 config fingerprint 为：

```text
d47933bbebfce8cc75c11f77a11363ba17b45ccf5c63860bd44bc394aa43e014
```

## 16. 有效结论与尚未回答的问题

本次可以支持的结论是：

> 官方 CNNDetection Blur+JPEG(0.1) 在 native RGB、no-resize、
> no-crop 的冻结协议下，无法把 CLAIMFORGE Mouse 小面积局部植入作为
> 独立整图 AIGC 检测任务解决。AUROC 为 0.4989，发布阈值检出
> 0/275 forged，配对排序也没有显著优势。

本结果不支持以下更强表述：

- 不能说 CNNDetection 对完全由 GAN/CNN 生成的整图无效；
- 不能说所有 CNNDetection checkpoint 都无效；本报告只测 0.1 primary；
- 不能把未测试的 0.5 当成已比较或已淘汰；
- 不能把 sigmoid score 称为目标域校准概率；
- 不能把 `edit_visibility=full` 误解为 fully synthetic；
- 不能把该整图 classifier 计入 T2 localization；
- 不能把高度配对相关性单独解释成已证明的失败因果。

最重要的后续对照仍是同领域的 real vs fully synthetic
lodging/restaurant 整图数据。它能区分“方法本身失效”和“方法仍能检测
整图生成，但不适合小面积局部植入”这两个不同命题。

## 17. 复现实务

正式目录：

```text
results/opensource/cnndetection/
  cnndetection_blur_jpg_prob0_1_native_mouse_canonical_v1_full275_20260725/
```

复用入口：

```text
eval/opensource/run_cnndetection.py
eval/opensource/cnndetection_metrics.py
eval/opensource/analyze_cnndetection_run.py
```

对新 canonical 数据，可以使用新的 immutable run ID 直接复用同一套
checkpoint、profile、逐图 feature 保存、summary 和 replay audit 流程。
runner 支持对成功行安全 resume，并拒绝配置漂移；不同 checkpoint 或
preprocess sensitivity 必须使用不同 run ID，不能覆盖本次正式结果。
