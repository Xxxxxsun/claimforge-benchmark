# CNNDetection Blur+JPEG(0.1) 在 Balanced250 上的正式结果

日期：2026-07-26（UTC）

正式 run：
`cnndetection_blur_jpg_prob0_1_native_balanced250_v1_full1775_20260726`

核心机器证据：
[run manifest](../results/opensource/cnndetection/cnndetection_blur_jpg_prob0_1_native_balanced250_v1_full1775_20260726/manifest.json)、
[逐图结果](../results/opensource/cnndetection/cnndetection_blur_jpg_prob0_1_native_balanced250_v1_full1775_20260726/results.jsonl)、
[coverage summary](../results/opensource/cnndetection/cnndetection_blur_jpg_prob0_1_native_balanced250_v1_full1775_20260726/summary.json)、
[Balanced250 metrics](../results/opensource/cnndetection/cnndetection_blur_jpg_prob0_1_native_balanced250_v1_full1775_20260726/balanced250_metrics.json)、
[fresh replay audit](../results/opensource/cnndetection/cnndetection_blur_jpg_prob0_1_native_balanced250_v1_full1775_20260726/independent_audit.json)、
[双 smoke comparison](../results/opensource/cnndetection/cnndetection_blur_jpg_prob0_1_native_balanced250_v1_smoke_a_20260726__vs__cnndetection_blur_jpg_prob0_1_native_balanced250_v1_smoke_b_20260726_comparison.json)。

## 1. 结论摘要

CNNDetection 的官方 Blur+JPEG(0.1) checkpoint 已按冻结的 native
whole-image 协议完成 Balanced250：

- score cache 覆盖 **1,775/1,775** 张图，全部成功；error、missing、
  unexpected、duplicate 和 superseded attempt 均为 0；
- primary panel 为 `real250` 分别对六组独立选择的 forged250，共 1,750
  个 panel row；额外 25 张 real 只用于补全 secondary source matching；
- 三种局部植入的 primary macro AUROC 为
  **0.494549 [0.483491, 0.505233]**，AP 为
  **0.494860 [0.486443, 0.508025]**；
- 三种 `fullframe_*` 条件的 primary macro AUROC 为
  **0.774736 [0.747902, 0.806377]**，AP 为
  **0.806773 [0.781342, 0.834324]**；
- 六条件 macro AUROC 为 **0.634643 [0.618418, 0.652453]**；这个值混合了
  机制和难度明显不同的 local 与 full-frame 条件，只是等权诊断汇总，不应
  当作某个自然部署流量上的单一性能；
- 尽管 `fullframe_*` 的连续分数具有明显排序能力，所有 1,775 张图的
  sigmoid score 都没有超过冻结的 strict `0.5` 阈值。六个 primary
  comparison 均为 `TP/FP/FN/TN = 0/0/250/250`；
- secondary 显式 source-matched 结果同样区分出两种行为：local 750 对的
  mean forged-real score delta 为 **-1.778764e-6**，strict matched
  ranking 为 **0.452**；full-frame 750 对分别为
  **+2.954999e-3** 和 **0.888**；
- 正式 run 前的两个 35-image CUDA smoke 在 computational projection、
  raw logit、score 和 2,048 维 feature 上最大差均为 **0.0**；
- fresh model replay 对全部 1,775 张图重新前向，raw logit、score 和
  feature 最大绝对差也全部为 **0.0**。

最准确的读法是：这个 checkpoint 对三组小面积局部植入没有可用的整图排序
能力；对当前三组 Hunyuan full-frame conditional-edit 条件有中等偏强的
相对排序能力，但官方固定阈值在本域完全失配，不能直接输出 positive
decision。`fullframe_*` 是本 benchmark 的处理条件，不代表这些图是脱离
真实源图独立生成的 fully synthetic image。

CNNDetection 只有整图 logit，没有原生 dense map、mask 或 bbox。本 run
只计入 **T1 whole-image AIGC detection**；**T2 localization 与 joint
score 均为 N/A**。

## 2. 方法、模型与许可

CNNDetection 来自 CVPR 2020 的 *CNN-Generated Images Are Surprisingly
Easy to Spot... for Now*。方法和来源的完整核验见
[既有 CNNDetection 正式报告](CNNDETECTION_BLUR_JPG_PROB0_1_NATIVE_MOUSE_FULL_RESULTS_2026-07-25.md#2-方法官方来源与原理)。
本次复用同一份冻结来源：

| 字段 | 固定值 |
|---|---|
| official repository commit | `ea0b5622365e3a9cd31d1b54b6b5971131a839ab` |
| paper-era comparison commit | `f692c138482137c92280c01a45ae190379f16790` |
| architecture | official vendored ResNet-50，one-logit head |
| checkpoint | `CNNDetection-BlurJPEG0.1@official-dropbox` |
| checkpoint bytes | 282,442,597 |
| checkpoint SHA-256 | `a73295ac66f9cb74d558ce3ade46f75e2f2997ed05eeed0f4b774623372058ea` |
| state payload SHA-256 | `8c62f887d5b97a0337f0ed598ac80cb9d86929613d3bc5c08fb0331b470c8931` |
| trainable parameters | 23,510,081 |
| checkpoint load | `torch.load(..., weights_only=True)` 成功；`strict=True` |

它用 ProGAN fake 与 LSUN real 训练 ResNet-50 二分类器，希望学习跨语义类别
共享的 CNN 生成痕迹。Blur+JPEG 训练增强降低了对特别脆弱高频捷径的依赖。
本次测试没有再对图像施加 blur 或 JPEG augmentation。

正式 checkpoint 的 `0.1` 选择在查看本 benchmark 分数前已经冻结，不是
在多个 checkpoint 的 Balanced250 结果里择优。官方没有发布 checkpoint
digest；上表 SHA-256 是本 benchmark 对实际下载字节的冻结标识，不是作者
发布的校验值。

固定仓库的 `LICENSE.txt` 为 **CC BY-NC-SA 4.0**，含
NonCommercial 条款，且不是 OSI open-source license。未发现 checkpoint
独立许可，因此不能据此建立商用许可。完整边界见
[既有许可证核验](CNNDETECTION_BLUR_JPG_PROB0_1_NATIVE_MOUSE_FULL_RESULTS_2026-07-25.md#5-license-与商用边界)：

```text
repository license: CC-BY-NC-SA-4.0
commercial use permitted by that license: false
checkpoint separate terms found: false
commercial clearance established: false
```

本次只是研究 benchmark，不构成法律意见或商业授权。

## 3. Frozen T1 协议

本 run 遵循
[Balanced250 generic evaluation contract](BALANCED250_GENERIC_EVALUATION_CONTRACT_2026-07-26.md)。
方法适配器与分析器分别为
[run_cnndetection_balanced.py](../eval/opensource/run_cnndetection_balanced.py)
和
[analyze_cnndetection_balanced.py](../eval/opensource/analyze_cnndetection_balanced.py)。

### 3.1 模型前向

```text
canonical JPEG
-> Pillow.Image.open(...).convert("RGB")
-> no EXIF transpose in the model adapter
-> native resolution, no resize, no crop
-> float32 ToTensor
-> ImageNet mean/std normalization
-> batch size 1, eval(), no autocast
-> official ResNet-50
-> adaptive global average pooling
-> 2,048-dimensional float32 feature
-> Linear(2048, 1)
-> one float32 raw logit
-> float32 sigmoid = ai_score
-> strict ai_score > 0.5
```

Canonical release 构建阶段已经完成 EXIF orientation、RGB 转换和 JPEG
Q95/4:4:4 规范化；adapter 不再重复 EXIF transpose。sigmoid 输出只称为
未校准的 fake score，**不是目标域上的校准概率**。

### 3.2 数据与 primary/secondary 分离

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

Primary 是 selection-unpaired point estimate：

```text
real250 vs local_mouse250
real250 vs local_cat250
real250 vs local_trash_can250
real250 vs fullframe_mouse250
real250 vs fullframe_cat250
real250 vs fullframe_trash_can250
```

它不会从 `task_id` 推断配对，也不会把 secondary matched delta 混入
primary。六组 forged 独立选择。置信区间使用 1,000 次共享
source-content-cluster Poisson bootstrap，seed `20260726`；同一 cluster
权重跨 real/forged label 和六个 condition 复用，以保留实际 source
重叠。六组 forged 与 real 的 source-cluster overlap 依次为
232、228、231、230、229、228。

Secondary 只接受 `source_pairs.jsonl` 明确记录的
`real_sample_id`/`forged_sample_id`，共 1,500 对。它使用跨 condition
共享的 source-content-cluster Poisson weights；额外 25 张非 panel real
只用于补齐这些显式 source links。

TPR@5%FPR 的阈值取各 comparison real score 的 95th percentile，
`method="higher"`，判定同样使用 strict `>`。它是报告用的 real-only
operating point，不替代方法发布的固定阈值 `0.5`。

### 3.3 Full-frame 条件的语义边界

`fullframe_mouse`、`fullframe_cat` 和 `fullframe_trash_can` 表示图像通过了
Hunyuan full-frame conditional-edit 流程。它们仍有真实源图和条件输入；
当前数据集不包含“脱离真实源图独立生成”的 fully synthetic 对照。尤其
Trash-can 的 primary label 表示 single-shot 图像通过了该流程，不保证目标
物体在语义 QC 中成功植入。

因此本报告可以评价 CNNDetection 对这三组 benchmark 条件的区分能力，但
不能把它写成对所有纯文生图、所有 fully synthetic 图像或所有生成模型的
性能。

## 4. Coverage

正式 score cache 的 condition 计数与审计如下：

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

其他 coverage 门禁：

```text
physical result rows: 1775
latest result rows:   1775
superseded attempts:  0
unexpected IDs:       0
duplicate result IDs: 0
coverage fraction:    1.0
success fraction:     1.0
feature files:        1775
```

## 5. Primary：六个 condition

下表均使用同一个独立 `real250` panel。区间是共享 source-cluster Poisson
bootstrap 的双侧 95% percentile CI。

| Forged condition | AUROC [95% CI] | AP [95% CI] | TPR@5%FPR [95% CI] | TP/FP/FN/TN @ strict 0.5 |
|---|---:|---:|---:|---:|
| `local_mouse` | 0.493280 [0.480426, 0.506652] | 0.497766 [0.487481, 0.512653] | 0.048000 [0.036734, 0.059289] | 0 / 0 / 250 / 250 |
| `local_cat` | 0.502672 [0.487706, 0.518056] | 0.499759 [0.489848, 0.513501] | 0.044000 [0.029533, 0.056185] | 0 / 0 / 250 / 250 |
| `local_trash_can` | 0.487696 [0.473766, 0.501560] | 0.487056 [0.471744, 0.506518] | 0.040000 [0.026119, 0.049386] | 0 / 0 / 250 / 250 |
| `fullframe_mouse` | 0.770960 [0.743109, 0.801857] | 0.800185 [0.774452, 0.828692] | 0.408000 [0.276395, 0.481692] | 0 / 0 / 250 / 250 |
| `fullframe_cat` | 0.779248 [0.749301, 0.811972] | 0.809709 [0.783547, 0.837893] | 0.396000 [0.308934, 0.498073] | 0 / 0 / 250 / 250 |
| `fullframe_trash_can` | 0.774000 [0.744582, 0.808049] | 0.810426 [0.782401, 0.841346] | 0.428000 [0.312408, 0.509885] | 0 / 0 / 250 / 250 |

三种 local AUROC 都在 0.5 附近，且各自 95% CI 均包含 0.5。三种
full-frame AUROC 则稳定在 0.77–0.78，AP 在 0.80–0.81，说明连续分数
可以对当前 full-frame 条件做相对排序。

但是两种能力不能混为一谈。正式 run 中 score 最大值只有
`0.2640593052`，仍低于固定阈值 0.5；因此：

```text
all 1775 classification_decision: false
all 1500 forged @ fixed 0.5:       false negative
all 275 real @ fixed 0.5:          true negative
```

full-frame 的好 AUROC 不等于冻结发布阈值可以直接部署。也不能在看完目标域
结果后调低阈值，再把调参后的值冒充本次 frozen primary。

## 6. Primary macro

Macro 是 condition-level point estimate 的等权平均。bootstrap replicate
沿用同一组跨 condition 的 cluster weights。

| Macro group | AUROC [95% CI] | AP [95% CI] | TPR@5%FPR [95% CI] |
|---|---:|---:|---:|
| Local 3-condition macro | 0.494549 [0.483491, 0.505233] | 0.494860 [0.486443, 0.508025] | 0.044000 [0.032725, 0.051625] |
| Full-frame 3-condition macro | 0.774736 [0.747902, 0.806377] | 0.806773 [0.781342, 0.834324] | 0.410667 [0.300785, 0.491127] |
| All 6-condition macro | 0.634643 [0.618418, 0.652453] | 0.650817 [0.637072, 0.667968] | 0.227333 [0.172951, 0.269963] |

三个 macro 的 fixed-threshold accuracy 都为 0.5、balanced accuracy 都为
0.5；precision、recall 和 F1 都为 0，specificity 为 1。原因不是 macro
算法，而是每个 250-real/250-forged comparison 都把全部图判为 real。

## 7. Domain 结果

完整 condition × domain point estimates、bootstrap CI、样本数与混淆矩阵
保存在
[metrics artifact](../results/opensource/cnndetection/cnndetection_blur_jpg_prob0_1_native_balanced250_v1_full1775_20260726/balanced250_metrics.json)。
所有 domain cell 在 fixed `0.5` 下仍为 `TP=0, FP=0`。等权
condition-macro 如下：

| Macro group / domain | AUROC [95% CI] | AP [95% CI] | TPR@5%FPR [95% CI] |
|---|---:|---:|---:|
| Local / lodging | 0.488361 [0.472125, 0.501324] | 0.503683 [0.492900, 0.524740] | 0.038649 [0.020288, 0.057361] |
| Local / restaurant | 0.490405 [0.471623, 0.507805] | 0.481225 [0.469500, 0.503385] | 0.035636 [0.021285, 0.059042] |
| Full-frame / lodging | 0.731153 [0.688794, 0.769604] | 0.778718 [0.742636, 0.814991] | 0.294917 [0.227412, 0.431397] |
| Full-frame / restaurant | 0.823630 [0.783788, 0.864275] | 0.853737 [0.820710, 0.886819] | 0.493106 [0.405477, 0.585930] |
| All / lodging | 0.609757 [0.585272, 0.631412] | 0.641201 [0.621587, 0.665257] | 0.166783 [0.126774, 0.236646] |
| All / restaurant | 0.657017 [0.632869, 0.681175] | 0.667481 [0.651218, 0.690511] | 0.264371 [0.219465, 0.312607] |

观察值上，full-frame restaurant 高于 full-frame lodging；本报告没有计算
两者差值的专门 simultaneous hypothesis test，因此不把这个点估计差写成
已证明的 domain 因果效应。

## 8. Secondary source-matched 结果

delta 定义为 direction-normalized
`forged ai_score - matched real ai_score`；正数表示 forged 被排得更像
fake。这里的配对只来自 1,500 条显式 source link，不从 `task_id` 猜测。
区间使用跨 condition 共享的 source-content-cluster Poisson bootstrap。

| Condition | Pairs | Wins/losses/ties | Mean delta [95% CI] | Median delta | Strict matched rank [95% CI] |
|---|---:|---:|---:|---:|---:|
| `local_mouse` | 250 | 114 / 136 / 0 | `-8.062926e-7` [`-2.943824e-6`, `5.756049e-7`] | `-1.146206e-13` | 0.456 [0.396074, 0.515396] |
| `local_cat` | 250 | 117 / 133 / 0 | `-4.066644e-6` [`-9.455539e-6`, `-7.375127e-7`] | `-2.135602e-13` | 0.468 [0.409822, 0.528743] |
| `local_trash_can` | 250 | 108 / 142 / 0 | `-4.633539e-7` [`-1.972560e-6`, `1.152585e-6`] | `-5.858380e-12` | 0.432 [0.368392, 0.492494] |
| `fullframe_mouse` | 250 | 219 / 31 / 0 | `3.462086e-3` [`1.294737e-3`, `6.309473e-3`] | `2.967613e-6` | 0.876 [0.831265, 0.915967] |
| `fullframe_cat` | 250 | 225 / 25 / 0 | `3.505398e-3` [`1.599806e-3`, `6.228363e-3`] | `3.720556e-6` | 0.900 [0.858957, 0.935487] |
| `fullframe_trash_can` | 250 | 222 / 28 / 0 | `1.897512e-3` [`9.752530e-4`, `3.131820e-3`] | `3.304583e-6` | 0.888 [0.848739, 0.925809] |

Pooled family 统计：

| Group | Pairs | Mean delta [95% CI] | Strict matched rank [95% CI] |
|---|---:|---:|---:|
| Local | 750 | `-1.778764e-6` [`-4.208546e-6`, `-2.525091e-7`] | 0.452 [0.414552, 0.488951] |
| Full-frame | 750 | `2.954999e-3` [`1.380590e-3`, `5.001818e-3`] | 0.888 [0.849794, 0.923182] |
| All | 1,500 | `1.476610e-3` [`6.903213e-4`, `2.513582e-3`] | 0.670 [0.641929, 0.696132] |

等权 condition-macro 的对应结果为：

| Macro group | Mean delta [95% CI] | Strict matched rank [95% CI] |
|---|---:|---:|
| Local | `-1.778764e-6` [`-4.190587e-6`, `-2.535271e-7`] | 0.452 [0.414301, 0.488813] |
| Full-frame | `2.954999e-3` [`1.377934e-3`, `4.997003e-3`] | 0.888 [0.849801, 0.923095] |
| All | `1.476610e-3` [`6.882802e-4`, `2.497929e-3`] | 0.670 [0.642606, 0.696210] |

Primary 与 secondary 回答不同问题：primary 衡量独立选择 panel 上的单图
区分；secondary 利用真实 source link 诊断同一源内容经过处理前后的 score
变化。不能用 secondary paired ranking 替代实际部署时没有 matched real
可用的 primary T1。

## 9. 双 smoke 与 fresh replay

正式 run 前执行：

```text
cnndetection_blur_jpg_prob0_1_native_balanced250_v1_smoke_a_20260726
cnndetection_blur_jpg_prob0_1_native_balanced250_v1_smoke_b_20260726
```

每个 smoke 从七个 condition 各取 5 张 panel 图，共 35 张；两个 run 都是
35/35 成功。比较器只忽略按设计必然不同的 run identity，以及执行时间、
latency、feature path 等 volatile 字段。模型计算内容的结果为：

| Smoke A/B check | Result |
|---|---:|
| images compared | 35 |
| exact computational projection | true |
| max raw-logit absolute difference | 0.0 |
| max `ai_score` absolute difference | 0.0 |
| max feature absolute difference | 0.0 |
| feature shape/dtype verified | true |
| feature file SHA-256/bytes verified | true |

正式
[fresh replay audit](../results/opensource/cnndetection/cnndetection_blur_jpg_prob0_1_native_balanced250_v1_full1775_20260726/independent_audit.json)
重新创建模型、加载相同 source/checkpoint，并重新前向全部 1,775 张图：

| Fresh replay check | Result |
|---|---:|
| images replayed | 1,775 |
| max raw-logit absolute difference | 0.0 |
| max `ai_score` absolute difference | 0.0 |
| max 2,048-D feature absolute difference | 0.0 |
| saved feature → fresh FC logit verified | true |
| saved feature → fresh sigmoid verified | true |
| all feature hash/shape/dtype/finiteness checks | passed |
| primary and secondary metrics recomputed | passed |

审计重新计算统计时复用了冻结的
[balanced250_metrics.py](../eval/opensource/balanced250_metrics.py)。它是完整
fresh model replay 与 artifact/contract audit，但不应描述成第二套完全
独立开发的统计实现。

## 10. Runtime 与显存

正式运行环境：

```text
GPU:         NVIDIA L20Z, cuda:0
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
TF32:        false
```

正式 run 从 `12:14:39.446863` 到 `12:21:12.779116`，wall-clock 为
**393.332 秒（6 分 33.332 秒）**。

逐图计时：

| Segment | Min ms | Mean ms | Median ms | P05 ms | P95 ms | Max ms |
|---|---:|---:|---:|---:|---:|---:|
| model/device segment | 3.892 | 21.426 | 22.153 | 4.850 | 39.386 | 392.850 |
| decode/preprocess | 7.996 | 122.656 | 130.405 | 34.979 | 167.671 | 192.312 |
| 两段逐图之和 | 12.682 | 144.081 | 148.794 | 49.792 | 199.270 | 561.469 |

两段之和不包含全部 evidence hash、feature 落盘、JSON 写入、metrics、
bootstrap、进程启动和审计时间，因此不能用它直接替代 wall-clock。

每张图 reset 后记录的 peak allocated CUDA memory：

| Statistic | Bytes | MiB |
|---|---:|---:|
| Min | 164,329,472 | 156.717 |
| Mean | 482,469,475 | 460.119 |
| Median | 589,117,952 | 561.827 |
| P05 | 188,097,024 | 179.383 |
| P95 | 663,191,552 | 632.469 |
| Max | 682,621,440 | 650.999 |

该值包含常驻模型 allocation；不是 checkpoint 文件大小，也不是整卡占用或
设备总显存。

## 11. Artifact hashes

正式 config fingerprint：

```text
6803d619bf760c9d997f314fa31e4113810bc21eb6c38901f25e2f26d3c9b0f0
```

正式 run：

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| `expected_inputs.jsonl` | 4,015,069 | `6b5128909eeffdbd88e61af02ca1bd191cb5460f94a23b47c87ebd0172e6d12c` |
| `results.jsonl` | 7,491,542 | `528801e7f66ceb3745aeaaa0bb5c79019d2d4681b4217679619b367f7ed9ea4c` |
| `summary.json` | 4,520 | `7e04a75f1dab2beaf1568d31b98ae5c8f853cec8c7ae7afa73fb61d00946e758` |
| `manifest.json` | 24,315 | `c1038871a8faa2a91175104363529b252e58aaf7035a65948ed64b8ea4cede1d` |
| `balanced250_metrics.json` | 107,140 | `e858c3af33c5b7d10338a463a2bfc91660e686ff588e491b45874b714f4656db` |
| `independent_audit.json` | 4,727 | `c561ebec498a1e748f3e48f54c2c4709e52fff378be20148258d5d71e1553678` |
| checkpoint | 282,442,597 | `a73295ac66f9cb74d558ce3ade46f75e2f2997ed05eeed0f4b774623372058ea` |

Feature inventory：

```text
files:       1775
shape/file:  [2048]
dtype:       float32
bytes/file:  8320
total bytes: 14768000
semantics:   official FC input after adaptive global average pooling
```

每个 feature 的路径、字节数和 SHA-256 都冻结在对应
`results.jsonl` row 中，并由 fresh replay 逐个验证。

Smoke comparison 文件为 2,521 bytes，SHA-256：

```text
66a40cc60c6cd87c3118436e24e4240945e8a9b9ad55574f50a5f297345d12a6
```

Smoke A/B 的主要 hash：

| Run | Manifest SHA-256 | Results SHA-256 | Expected-inputs SHA-256 |
|---|---|---|---|
| A | `2e36c73d215e0fb19a04ea10cdb17c9b8f7de234cfe8afb437725a5378a231ab` | `ef43fbcc52b4dda3eb4417adbfa1f93d22fb1cfc2f96a88a59bf2a24c65544ad` | `21e556dd791960afde6cc900d9aba79da61a1a935c6da7b0e0928d3f6b26afa0` |
| B | `9ac106d199c2edbb2d49dd2364174758d4b6d532ca089a94c65714e5fdd8012a` | `f303d3ea93bce06bf3cd36f98ecc0d0a9b3f0bfe449d2cd816f1e33cb26168f0` | `21e556dd791960afde6cc900d9aba79da61a1a935c6da7b0e0928d3f6b26afa0` |

原始 smoke artifact 的整个文件 hash 按设计不同，因为 run identity、
timestamp、latency 和路径不同；确定性结论来自显式忽略这些字段后的 exact
computational comparison，而不是要求两个目录 byte-identical。

## 12. 能支持与不能支持的结论

本次证据支持：

1. 官方 CNNDetection Blur+JPEG(0.1) 在冻结 native 协议下，对 Mouse、Cat、
   Trash-can 三种局部植入的整图 T1 排序约为随机，local macro AUROC
   0.4945。
2. 同一个模型对当前三种 Hunyuan full-frame conditional-edit condition
   有稳定的相对排序信号，full-frame macro AUROC 0.7747、AP 0.8068。
3. 这个相对排序信号没有转化为固定 `score > 0.5` 的 positive decision；
   1,500 张 forged 全部为 fixed-threshold false negative。
4. 双 smoke 与 1,775-image fresh replay 都精确复现，结果不是一次随机
   CUDA 漂移造成。

本次证据不支持：

- 不能把 sigmoid score 称为校准概率；
- 不能事后在 Balanced250 上选阈值，再把结果写成冻结的 zero-shot
  operating point；
- 不能把 `fullframe_*` 称为 fully synthetic，或外推到所有文生图模型；
- 不能把 full-frame restaurant 与 lodging 的点估计差直接写成已证明的
  domain 因果差；
- 不能把 all-condition macro 当作固定自然流量配比的部署指标；
- 不能把 source-matched secondary 当作不需要 matched real 的单图检测；
- 不能把 CNNDetection 计入 T2 localization；它没有可评分的原生定位输出；
- 不能据此声称其他 CNNDetection checkpoint、其他 preprocessing 或
  target-domain calibration 已经被测试或淘汰。
