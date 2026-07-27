# HiFi-IFDL（CVPR 2023 / general 750001）在 Balanced250 上的正式结果

日期：2026-07-27（UTC）

> **文档状态：正式完成，双 smoke 与独立全链路 fresh replay 审计通过**
>
> 本报告只使用最终冻结的 r2 证据：1,775-image formal run、两个独立
> 35-image CUDA smoke、共享 Balanced250 T1/T2 指标、逐文件 artifact
> audit 和 1,775-image fresh full-model replay。第一次 r1 formal 因
> fail-closed 数值 sanity gate 中止，既未补写结果，也未用于本报告指标。

正式 run：
`hifi_ifdl_general750001_balanced250_v1_full1775_r2_20260727`

Formal immutable run-config fingerprint：
`d7d56453962dae4a56d972cfe9e87ee6168d06de95cf319e85ea08bdfd997cfe`

核心机器证据：

- [run manifest](../results/opensource/hifi_ifdl/hifi_ifdl_general750001_balanced250_v1_full1775_r2_20260727/manifest.json)：
  `88797682a603795ae025fdb403e36e7f1001c38597062565f141cf7a91428f80`
- [expected inputs](../results/opensource/hifi_ifdl/hifi_ifdl_general750001_balanced250_v1_full1775_r2_20260727/expected_inputs.jsonl)：
  `6b5128909eeffdbd88e61af02ca1bd191cb5460f94a23b47c87ebd0172e6d12c`
- [逐图结果](../results/opensource/hifi_ifdl/hifi_ifdl_general750001_balanced250_v1_full1775_r2_20260727/results.jsonl)：
  `8ecd091c83b76f70633cb71680739c3f0ea2f2952d9765f759eceb5b72b100c6`
- [coverage summary](../results/opensource/hifi_ifdl/hifi_ifdl_general750001_balanced250_v1_full1775_r2_20260727/summary.json)：
  `60aaac7802bcf0524df5bd98484b5217b01ab4d47c041d0fb3bc2f7f96208bf1`
- [Balanced250 T1/T2 metrics](../results/opensource/hifi_ifdl/hifi_ifdl_general750001_balanced250_v1_full1775_r2_20260727/balanced250_metrics.json)：
  `b7eadbe6cb4a1ec7fcb65bd6f0571edcddd7023838751a6c11bae6360ac1b5fd`
- [independent replay audit](../results/opensource/hifi_ifdl/hifi_ifdl_general750001_balanced250_v1_full1775_r2_20260727/independent_audit.json)：
  `117b1feaad01f6e4b9bfe812e79eca6a1710b1a35a41684fb742597ade740e80`
- [双 smoke comparison](../results/opensource/hifi_ifdl/_reports/hifi_ifdl_general750001_balanced250_v1_smoke5x7_a_r2_20260727__vs__hifi_ifdl_general750001_balanced250_v1_smoke5x7_b_r2_20260727_comparison.json)：
  `9bcec1894b6e8f43c544124ae6cea5efd204fb126e4c7935d867662d0f2492c8`

最终机器门禁为 `status=passed`、`publishable=true`。

## 1. 结论摘要

HiFi-IFDL 是 Balanced250 上第五个完成正式扩展的 local-forensics
方法。它的结果与模型训练目标高度一致，但能力边界也非常鲜明：
**三类局部植入检测和定位都失败；三类 Hunyuan 全图条件编辑却有强而一致
的 T1 排序信号。**

- formal coverage 为 1,775/1,775，零 error、missing 和 superseded；
- local 三条件等权 macro AUROC 为 **0.484539**，AP 为
  **0.487231**，TPR@5%FPR 为 **4.00%**；
- local 的发布分数阈值 `score > 0.5` 对 750 张 forged 全部报
  authentic，recall 为 **0**；
- full-frame 三条件 macro AUROC 为 **0.824704**，AP 为
  **0.825834**，TPR@5%FPR 为 **38.40%**；
- 但 `score > 0.5` 的 full-frame macro recall 只有 **10.67%**，
  表明排序有用而发布阈值没有在本数据上校准；
- source-matched local pooled strict ranking 为 **34.93%**，mean
  score delta 为 `-0.000434`；
- source-matched full-frame pooled strict ranking 为 **88.80%**，
  mean score delta 为 `+0.118196`；
- 750 张 local 的 pooled per-image pixel AP 为 **0.098547**，但
  per-image IoU@2.3 只有 **0.0000846**；
- pooled-pixel micro recall 只有 **0.01715%**，micro IoU 为
  **0.000168**；
- 740/750 张 local 的发布阈值 mask 为空，只有 6/750 张与任意 GT
  pixel 重叠；
- 275 张 real 的 mean per-image false-positive fraction 为
  **0.007289**（约 0.729%），micro fraction 为 **0.001944**；
- 两个 smoke 的四级 classification logits、fine probabilities、
  T1 scores、embeddings、两级 distance maps 和适用 masks 全部
  exact；
- fresh replay 完成 1,775/1,775 forwards；全部 classification
  heads、T1 scores 和 1,025 组适用 dense artifacts byte-exact，
  最大差异全部为 `0.0`。

当前证据支持以下结论：

1. general 750001 checkpoint 能稳定区分当前三类 full-frame Hunyuan
   conditional edits 与真实图；三条件 AUROC 都在 `0.818–0.831`，
   source-matched ranking 都在 `0.872–0.896`。
2. 这种 whole-image 信号没有转移到小范围局部插入。局部 T1 不仅接近
   chance，matched score 还更常下降。
3. T2 的 Cat/Trash-can AP 看起来高于 Mouse，主要原因之一是 GT positive
   prevalence 分别高达 `6.37%` 和 `21.97%`；fixed-threshold mask 几乎
   全空，不能把 AP 写成可靠定位能力。
4. `AUROC=0.8247` 不等于“检测出 82.47%”。在冻结的 `0.5` 阈值上，
   full-frame 只检出 80/750；实际部署必须独立校准 operating point。
5. 三组 full-frame 数据是从真实源图出发的 Hunyuan 条件全图编辑，不是
   脱离真实源图独立采样的纯 T2I。本结果不能外推到所有生成器、压缩链或
   无条件生成分布。

## 2. 官方来源、checkpoint 与许可证

本方法对应 CVPR 2023 论文
[Hierarchical Fine-Grained Image Forgery Detection and Localization](https://openaccess.thecvf.com/content/CVPR2023/html/Guo_Hierarchical_Fine-Grained_Image_Forgery_Detection_and_Localization_CVPR_2023_paper.html)，
正式执行冻结作者的
[HiFi-IFDL repository](https://github.com/CHELSEA234/HiFi_IFDL)。

| 资产 | 冻结身份 | 大小 / 结构 | SHA-256 |
|---|---|---:|---|
| official source | commit `0ca70d651087bb09959dec583947031c47d30209`；tree `f5dbb144329a048c2c03f102d441c3ef37a89a3f` | tracked + non-cache untracked clean；12 个关键 source/license 文件绑定 | 见 manifest |
| HRNet feature extractor | official Google Drive `HRNet/750001.pth` | 81,112,652 bytes；699 state tensors | `be21278afb4e657bdafdf581d8d8bc6bc09f3b4507b10502ce98f1ae7ef1c5c1` |
| hierarchical localizer/classifier | official Google Drive `NLCDetection/750001.pth` | 57,487,769 bytes；66 state tensors | `7615fcb054e7cbd0b25d647d72655a690424232668abbc911551648e84b5f8fc` |
| authentic center/radius | repository `center/radius_center.pth` | 543 bytes；18-element FP32 center + scalar radius | `e41e09256e65bcff9ba43e72f08701bf4d3904ccdb749f2d32a008af92c2483b` |
| registered task bundle | 两个 task checkpoints + center/radius | 765 state tensors | `62d0b9f5e501f85558cfbdd5f797dc4e2553ce74729168921257408d735681f9` |
| repository `LICENSE` | MIT text | 1,065 bytes | `b01d7140e1f323024b0db35e0db18ba7cd3fd3380abbec57aec55e6141864e2f` |

两份 task checkpoint 均通过
`torch.load(..., map_location="cpu", weights_only=True)` 加载，随后
strict load：

```text
feature extractor state:          699 / 699 exact
localizer/classifier state:        66 / 66 exact
missing keys:                               0
unexpected keys:                            0
state-prefix rewrite:                   false
parameters:                         6,890,320
trainable parameters:               6,890,131
buffers:                                18,764
modules:                                   505
```

Repository 还含一个 HRNet initialization weight，但 released general
task checkpoint 已完整覆盖 HRNet parameters 和 buffers；正式模型没有
先加载该 initialization weight。

`750001` 只作为作者发布的 identifier 保留，不被本报告称为 epoch。两份
optimizer state 记录 step 750001，但论文、补充材料与 release wording
不足以证明它等于 epoch 编号。

许可证必须拆成两层解释：

- repository code 的 MIT 文本允许 commercial use 和 redistribution；
- official Google Drive checkpoint bundle 没有找到单独 license/terms，
  repository 的代码许可证也没有明确延伸到权重。

因此 audit 冻结：

```text
project code commercial permission:        true
checkpoint commercial clearance:          false
overall product commercial clearance:      false
```

这不是法律意见；本 benchmark 结果不建立 checkpoint、训练数据或依赖的
商业产品许可。

## 3. 方法原理与训练边界

HiFi-IFDL 把整图分类与像素级定位放进同一个分层细粒度模型：

1. HRNet-W18-small-v2 始终保留高分辨率分支，并在四个尺度交换信息；
2. 一次 forward 产生
   `18×256×256`、`36×128×128`、`72×64×64` 和
   `144×32×32` 四级 feature；
3. NLCDetection head 输出 18-channel pixel embedding、一个 auxiliary
   learned sigmoid mask，以及 3、5、7、14 类的层级 classification
   logits；
4. 最细的 14 类包含 authentic、传统 splice/inpainting/copy-move、
   face-editing，以及 GAN/diffusion generation categories；
5. T1 用 fine head 的 `1 - P(authentic class 0)`；
6. T2 把每个像素的 18-D embedding 与 released authentic center 做
   `PairwiseDistance(p=2, eps=1e-6)`，距离越大越异常。

这种设计的优势是：分类 head 学习 broad-to-fine manipulation family，
而 hypersphere distance 给出可解释的一类 authentic-pixel 异常度。HRNet
不必先把全部空间细节压到很低分辨率，理论上也更适合 localization。

Balanced250 揭示了两个不同的 domain-transfer 结果：

- 全图 conditional edit 会改变大范围生成痕迹，fine head 训练过的
  GAN/diffusion family signal 仍能形成稳定排序；
- 局部插入只占图像一小部分，256×256 非等比 stretch 后更容易被真实场景
  主体淹没，fine head 和 hypersphere threshold 都没有稳定响应。

这是与模型设计和结果一致的解释，不是 causal ablation。不能据此断言
resize 是唯一原因，也不能把当前 checkpoint 的失败外推到重新训练过的
HiFi variant。

正式选择的是作者区分出的 general detection-and-localization release，
不是 localization-only weights。这个选择在观察 Balanced250 指标之前
冻结，因为 general release 同时提供原生 T1、T2，并包含生成式伪造类别。
CLAIMFORGE 没有在 Balanced250 上训练、微调、拟合 calibration 或搜索
T2 threshold。

## 4. r1 fail-closed 与 r2 数值修正

第一次 formal ID（无 `_r2`）没有完成：

```text
successful rows: 1447
error rows:         1
missing rows:     327
failing rank:    1447
failing sample:  5724ab5a9da93056c640a537
error: HiFi-IFDL fine logits/probabilities disagree
```

它在 `fullframe_cat` 上触发的不是 model、input 或 checkpoint drift，而是
一项静态交叉设备 sanity：runner 把 CUDA 上记录的 float32
`torch.softmax` 与 CPU NumPy float32 exp/sum/div reference 比较。失败样本
最大差异为 `3.5762786865234375e-7`，即 3 个 float32 epsilon、7 ULP；
GPU 与 torch CPU 的最大差异也只有 2 epsilon。36-image diagnostic 中
没有任何样本超过 3 epsilon。

r2 只修改这项静态 CPU-vs-recorded-CUDA sanity 的理论容差：

```text
float32 eps:                              1.1920928955078125e-7
simplex-sum tolerance:                    2 eps（不变）
14-way cross-device softmax sanity:       8 eps
roundoff basis:                           u=eps/2; 14u≈7eps;
                                          向上取二进制整数界 8eps
same-device A/B smoke tolerance:          0
same-device fresh replay tolerance:       0
```

`8 eps` 不参与 score、decision、ranking、artifact 或 metric 计算，只用于
静态跨实现一致性检查。模型 inference、softmax 输出、T1 公式、T2
postprocess 和阈值均未放宽。最终 r2 的 A/B smoke 和 1,775-image fresh
replay仍要求 `np.array_equal`，实际最大差异全部为 `0.0`。

r1 的 incomplete formal 和两份旧 smoke 已移入 recoverable local trash；
没有重命名成正式结果、没有复用为 r2 evidence，也没有发布任何 incomplete
指标。这次处理保留了 fail-closed 原则：先中止、诊断并冻结有理论依据的
最小修正，再从空的新 run IDs 重跑全部证据。

## 5. Frozen executable contract

正式实现入口：

- [runner](../eval/opensource/run_hifi_ifdl_balanced.py)
- [independent analyzer](../eval/opensource/analyze_hifi_ifdl_balanced.py)
- [legacy official-contract adapter](../eval/opensource/run_hifi_ifdl.py)
- [legacy HiFi metrics](../eval/opensource/hifi_ifdl_metrics.py)
- [shared T1 metrics](../eval/opensource/balanced250_metrics.py)
- [shared T2 metrics](../eval/opensource/balanced250_localization_metrics.py)

冻结文件 SHA-256：

| File | SHA-256 |
|---|---|
| Balanced250 runner | `2e66c50f38ca38980abbf00ad293a5b0947b86fbe11c743258746a987312337b` |
| Balanced250 analyzer | `7243dc98d4b27ba21928c4c9ed3cf3d5830e3c6591338f7fe3b3e11cd548a7fb` |
| runner tests | `6ba6025f0e0305ddb72e9df0fc2dfa5c16a93057c516eb5d7f5bdfad202e8a6d` |
| analyzer tests | `cb7655742e8da28330bdc17d6bda7d281c5da9886aafdae873938f46d48f6a10` |
| legacy official-contract adapter | `1eca5d96cb9fb13f82a5d885a48bb78266d07d2ef3ed214239504ab61dea6544` |
| legacy HiFi metrics | `27a1352600c2fb7c97d52143de27643a25cc30bef8bfb279748a76fa515ca32b` |
| shared Balanced250 T1 metrics | `f3932099bb63b766f063a66684e1d45f6e12601337d73859591d79297dbbed1c` |
| shared Balanced250 T2 metrics | `83ac07257078fc41276742fa4b9f2eb936ac51c8ff93bf1253b8c45f2b704b2a` |

Formal manifest 绑定 9 个本地 adapter/source 文件；independent audit
再绑定 analyzer 本身，并在执行前后核对 source、asset、environment 和
formal evidence bytes 未变化。

### 5.1 Preprocess

| 组件 | 冻结行为 |
|---|---|
| decode | `imageio.v2.imread` |
| channel / dtype | RGB uint8 |
| geometry | 直接非等比 stretch 到 256×256 |
| resize | Pillow bicubic |
| scale | contiguous CHW FP32，除以 255 |
| normalization | 无 mean/std normalization |
| crop / re-encode | 无 / 无 |
| batch / TTA / ensemble | 1 / false / false |

非等比 stretch 是官方 `HiFi_Net.py` contract，而不是 CLAIMFORGE 后加的
resize。它不能被保持长宽比的 letterbox 或常见 ImageNet normalization
静默替换。

### 5.2 T1

```text
ai_score = float32(1) - softmax(fine_14_logits)[authentic_index_0]
direction = higher means forged
shared fixed decision = ai_score > 0.5
official decision = argmax(fine_14_logits) != 0
```

共享 T1 使用连续 `ai_score`。Official argmax 单独保存，不替代 shared
score rule。由于 13 个 non-authentic class 的总概率可能超过 0.5，而
authentic 仍是单一最大 class，两种 binary rule 不必逐图相同。

### 5.3 T2

原始输出：

```text
d = EuclideanDistance(18-D pixel embedding, released authentic center)
model space = 256×256 FP32
native restore = bilinear, align_corners=False
official mask = native d >= 2.3
```

Raw distance 非负、无上界，也不是 probability。共享 T2 reducer 接口要求
`[0,1]` continuous map，因此 analyzer 只在 metric adapter 中使用严格
单调变换：

```text
p = d / (d + 2.3)
d >= 2.3  <=>  p >= 0.5
```

该变换保持 pixel ranking 和 AP，并逐图确认 threshold mask bit-exact。
保存、哈希和 replay 的 primary artifacts 始终是 raw distance，没有把
`p` 写成“模型概率”。

T2 的合法范围：

- real275：全零 GT，只计算 false-positive area；
- local mouse/cat/trash-can 各 250：完整 exact-difference GT；
- full-frame 750：T2 `N/A`，conditioning box 不是 GT；
- auxiliary learned sigmoid mask 只记录诊断，不冒充官方 localization
  output。

### 5.4 Runtime

| 项 | 值 |
|---|---|
| grouped physical GPU | 4 |
| process-visible device | `cuda:0` |
| GPU identity | NVIDIA L20Z，compute capability 9.0 |
| Python | CPython 3.12.3 |
| PyTorch / torchvision | `2.8.0.dev20250627+cu128` / `0.23.0.dev20250627+cu128` |
| NumPy / Pillow / imageio | 2.2.6 / 11.1.0 / 2.37.0 |
| precision / batch | FP32 / 1 |
| autocast | false |
| deterministic algorithms | true |
| cuDNN deterministic / benchmark | true / false |
| matmul TF32 / cuDNN TF32 | false / false |
| `CUBLAS_WORKSPACE_CONFIG` | `:4096:8` |
| seed | 42 |

Manifest 冻结 logical device 与 GPU identity；物理序号由 grouped
supervisor 做进程隔离映射。

## 6. Frozen Balanced250 设计与任务边界

共享协议包含：

- 1,775 个唯一 score-cache inputs；
- 1,750-row primary panel：real 与六个 forged condition 各 250；
- 1,500 个显式 source-matched pairs；
- primary：每个 forged condition 对同一 independently selected
  `real250` panel；
- secondary：只使用 `source_pairs.jsonl` 的显式端点；
- 不从 `task_id` 猜测配对；
- 1,000 次 shared-source-content-cluster Poisson bootstrap；
- bootstrap root seed `20260726`；
- TPR@5%FPR 使用 real score 的 0.95 `higher` quantile和严格 `>`。

HiFi-IFDL capability contract：

```text
T1: real275 + local750 + fullframe750 = 1775
T2: real275 + local750                 = 1025
fullframe T2: N/A
```

Analyzer 在 T2 阶段对 750 张 full-frame 的 score-map loader 调用次数为
0。六条件 mixed macro 只作导航摘要；local 与 full-frame 必须分开解释。

## 7. Coverage 与 artifact inventory

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

Raw local evidence 保存在 gitignored
`outputs/opensource/hifi_ifdl/hifi_ifdl_general750001_balanced250_v1_full1775_r2_20260727/`：

| Artifact | Files | Total bytes | Role |
|---|---:|---:|---|
| 18×256×256 embedding FP32 NPY | 1,025 | 4,836,688,000 | raw model embedding / replay |
| 256×256 distance FP32 NPY | 1,025 | 268,828,800 | official model-space distance |
| native distance FP32 NPY | 1,025 | 6,595,316,160 | primary T2 continuous map |
| native threshold PNG mask | 1,025 | 1,699,592 | `distance >= 2.3` |
| **Total** | **4,100** | **11,702,532,552** | gitignored local evidence |

Full-frame forward 仍检查 embedding/distance 的 shape 与 finite，但不持久化
这些 transient dense outputs，也不创建 750 张没有合法 GT 的 mask。

## 8. Primary T1 结果

方括号为 1,000 次 shared-cluster bootstrap 的 95% percentile interval。

### 8.1 Family macro

| Scope | AUROC [95% CI] | AP [95% CI] | TPR@5%FPR [95% CI] | Accuracy @ 0.5 | Recall @ 0.5 |
|---|---:|---:|---:|---:|---:|
| local macro | 0.484539 [0.471708, 0.498340] | 0.487231 [0.475331, 0.504501] | 0.040000 [0.025762, 0.053763] | 0.500000 | 0 |
| full-frame macro | **0.824704** [0.795672, 0.854150] | **0.825834** [0.793165, 0.858040] | **0.384000** [0.275953, 0.490452] | 0.553333 | 0.106667 |
| all-six mixed macro | 0.654621 [0.636747, 0.672014] | 0.656533 [0.637264, 0.677817] | 0.212000 [0.153259, 0.267308] | 0.526667 | 0.053333 |

Local AUROC interval 的上端仍低于 `0.5`；这批局部插入没有可用 T1
signal。Full-frame 的 ranking interval 明确高于 chance，但 fixed
threshold accuracy 只有 `0.5533`，因为 sensitivity 太低。

三组 point-estimate 5% FPR threshold 都来自同一个 independent real250：

```text
threshold: 0.0072435736656188965
actual FPR: 0.048
```

这个阈值远低于 released `0.5`，进一步说明 score 可以用于排序，但不能
把未经校准的原始阈值直接当成当前部署 operating point。

### 8.2 条件级 ranking 与 low-FPR

| Condition | AUROC [95% CI] | AP [95% CI] | TPR@5%FPR [95% CI] |
|---|---:|---:|---:|
| local mouse | 0.514368 [0.499535, 0.529682] | 0.514826 [0.504694, 0.530932] | 0.056 [0.039996, 0.066667] |
| local cat | 0.482832 [0.462562, 0.500578] | 0.482637 [0.462117, 0.507503] | 0.036 [0.008727, 0.056823] |
| local trash can | 0.456416 [0.438473, 0.475906] | 0.464232 [0.446936, 0.487454] | 0.028 [0.011809, 0.046512] |
| full-frame mouse | **0.830600** [0.801403, 0.861174] | 0.832354 [0.800154, 0.863553] | **0.416** [0.279591, 0.525971] |
| full-frame cat | **0.825544** [0.796541, 0.855757] | 0.829957 [0.795296, 0.861470] | **0.396** [0.280885, 0.510149] |
| full-frame trash can | **0.817968** [0.786214, 0.849087] | 0.815191 [0.779802, 0.851508] | **0.340** [0.254732, 0.449847] |

三个 full-frame object categories 的点估计和 interval 高度一致，没有
依赖单一类别。Local Mouse 刚高于 chance，而 Cat/Trash-can 低于 chance；
不应把 Mouse 的单条件下界接近 0.5 写成整体局部检测能力。

### 8.3 Released score threshold 与 official argmax

共享 `score > 0.5`：

| Condition | TP / FP / FN / TN | Accuracy | Recall |
|---|---:|---:|---:|
| local mouse | 0 / 0 / 250 / 250 | 0.500 | 0 |
| local cat | 0 / 0 / 250 / 250 | 0.500 | 0 |
| local trash can | 0 / 0 / 250 / 250 | 0.500 | 0 |
| full-frame mouse | 28 / 0 / 222 / 250 | 0.556 | 0.112 |
| full-frame cat | 30 / 0 / 220 / 250 | 0.560 | 0.120 |
| full-frame trash can | 22 / 0 / 228 / 250 | 0.544 | 0.088 |

同一个 real250 panel 在六组比较中均为 0 FP；不是 1,500 张不同 real。
完整 real275 也全部低于 0.5。

Official fine-head argmax 对 full-frame Mouse/Cat/Trash-can 分别检出
26、28、20 张，共 74/750；shared score threshold 检出 80/750。两种
规则对 local750 和 real275 都是零 positive。

### 8.4 Domain

| Family | Domain | AUROC | AP | TPR@5%FPR | Accuracy @ 0.5 | Recall @ 0.5 |
|---|---|---:|---:|---:|---:|---:|
| local | lodging | 0.473753 | 0.497158 | 0.026553 | 0.486993 | 0 |
| local | restaurant | 0.497810 | 0.481497 | 0.032686 | 0.515158 | 0 |
| full-frame | lodging | 0.802488 | 0.794523 | 0.285471 | 0.513781 | 0.044168 |
| full-frame | restaurant | 0.859946 | 0.864575 | 0.483822 | 0.598492 | 0.180214 |

Restaurant full-frame 明显强于 lodging，但两个 domain 的 AUROC 都高于
0.8。这里没有预注册直接 domain-difference simultaneous test，只作描述。

## 9. Source-matched secondary

Secondary 只使用 frozen `source_pairs.jsonl` 的显式端点，回答“同一真实
源图经过编辑后 score 是否上移”。它不替代 independent single-image
primary。

| Scope | Pairs | Mean delta [95% CI] | Median delta | Strict ranking [95% CI] | W / L / T |
|---|---:|---:|---:|---:|---:|
| local mouse | 250 | -0.000088 [-0.000354, +0.000143] | 0 | 0.372 [0.312494, 0.429226] | 93 / 120 / 37 |
| local cat | 250 | -0.000219 [-0.000534, +0.000012] | -0.00000024 | 0.380 [0.323396, 0.436519] | 95 / 133 / 22 |
| local trash can | 250 | -0.000995 [-0.002325, -0.000024] | -0.00000155 | 0.296 [0.237898, 0.347644] | 74 / 168 / 8 |
| full-frame mouse | 250 | +0.117665 [+0.087745, +0.151198] | +0.002113 | 0.896 [0.851707, 0.935130] | 224 / 26 / 0 |
| full-frame cat | 250 | +0.129842 [+0.098083, +0.166411] | +0.002010 | 0.896 [0.856058, 0.932787] | 224 / 26 / 0 |
| full-frame trash can | 250 | +0.107082 [+0.077634, +0.139752] | +0.001124 | 0.872 [0.827451, 0.911456] | 218 / 32 / 0 |
| **local pooled** | **750** | **-0.000434 [-0.000987, -0.000030]** | **-0.00000024** | **0.349333 [0.311971, 0.387058]** | **262 / 421 / 67** |
| **full-frame pooled** | **750** | **+0.118196 [+0.091363, +0.147936]** | **+0.001433** | **0.888000 [0.851359, 0.922704]** | **666 / 84 / 0** |

Matched 与 independent primary 给出同一方向：local edit 没有让模型 score
稳定上升，full-frame edit 则在三类对象上都有一致上移。Full-frame mean
delta 远大于 median，说明少数高-confidence 样本拉高均值；但 88.8% 的
strict wins 表明结果并非只由这些尾部样本造成。

## 10. Native T2 localization

以下 AP 使用与 raw distance 严格同序的 `d/(d+2.3)` compatibility map；
fixed-threshold 指标逐像素等价于 official raw `distance >= 2.3`。

### 10.1 Per-image metrics

| Scope | GT positive fraction | Pixel AP [95% CI] | Recall@2.3 | F1@2.3 | IoU@2.3 | Nonempty / overlap |
|---|---:|---:|---:|---:|---:|---:|
| local mouse | 0.001350 | 0.003688 [0.003029, 0.004465] | 0.008000 | 0.0000606 | 0.0000305 | 3 / 2 |
| local cat | 0.063678 | 0.069327 [0.054115, 0.085311] | 0.008000 | 0.0001439 | 0.0000726 | 4 / 2 |
| local trash can | 0.219667 | 0.222624 [0.197632, 0.247063] | 0.004112 | 0.0002954 | 0.0001509 | 3 / 2 |
| **750-image pooled** | **0.095385** | **0.098547 [0.088454, 0.108183]** | **0.006704** | **0.0001666** | **0.0000846** | **10 / 6** |

Per-image precision 只在 10 张 nonempty masks 上有定义；把其 defined-only
均值与 740 张空 mask 混在一个 headline 会误导，因此表中不展示。

Cat 与 Trash-can 的 AP 分别只比对应 positive prevalence 高约
`0.00565` 和 `0.00296`。AP 是 ranking metric，且 random baseline 会随
positive prevalence 上升；`0.2226` 不能写成“定位了 22.26% 的垃圾桶
区域”。

### 10.2 Pooled-pixel micro metrics

| Scope | Predicted positive fraction | Micro precision | Micro recall | Micro F1 | Micro IoU |
|---|---:|---:|---:|---:|---:|
| local mouse | 0.002130 | 0.003806 | 0.006006 | 0.004659 | 0.002335 |
| local cat | 0.002149 | 0.008990 | 0.000303 | 0.000586 | 0.000293 |
| local trash can | 0.001105 | 0.019540 | 0.0000983 | 0.000196 | 0.0000978 |
| **all local pixels** | **0.001792** | **0.009130** | **0.0001715** | **0.0003367** | **0.0001684** |

750 张 local 共包含 1,206,543,236 个像素：

```text
TP        19,736
FP     2,141,818
FN   115,066,722
TN 1,089,314,960
```

模型在发布阈值下只覆盖约 `0.01715%` 的 GT pixels。Per-image AP
`0.098547` 与这个 coverage 不是同一个量，不能互相替代。

### 10.3 Real-image false-positive area 与重复源异常

275 张 real 的 GT 全零，因此 pixel AP 为 `null`：

- false-positive pixels：859,606 / 442,253,004；
- mean per-image FP fraction：
  `0.007289 [0, 0.024840]`；
- micro FP fraction：
  `0.001944 [0, 0.006561]`；
- 4/275 张 real mask 非空；
- maximum per-image FP fraction 为 `0.999859`。

其中 `lodging_078_slot_001` 和 `lodging_009_slot_001` 是同一 pristine
JPEG 的两个注册 evaluation rows；两者各有 426,340 个 FP pixels，
positive fraction 都为 `99.9859%`。它们共同贡献
852,680/859,606，即约 **99.19%** 的 real FP pixels。

所以“只有 4 张 real nonempty”与“mean FP area 0.729%”并不矛盾：绝大
多数图完全不响应，但同一重复场景产生了两次近乎整图的灾难性 false
positive。Shared bootstrap 按 source-content cluster 保持这种依赖，
没有把两行当成完全独立内容。

### 10.4 Full-frame T2 边界

Full-frame 只有合法 T1 label，没有局部 forged-pixel GT。Analyzer 对
750 张 full-frame：

- 检查 forward dense output 的 shape 与 finite；
- 不保存 embedding/distance artifacts；
- 不创建 binary mask；
- 不加载 score map；
- 不计算 T2。

因此本报告的强 full-frame 结论只属于 14-class image head，不能写成
HiFi 同时完成了 full-frame “定位”。

## 11. Determinism、独立审计与运行成本

### 11.1 CPU preflight 与 structural golden

CPU preflight 在 accelerator configuration 和任何 Balanced250 forward
之前完成：

- `cuda_initialized_before=false`；
- `cuda_initialized_after=false`；
- model forwards：0；
- Balanced250 scores：0；
- official source、assets、environment 与 adapter sources exact；
- 765/765 task checkpoint state tensors strict load；
- parameters、buffers、modules 与冻结值一致。

Independent analyzer 又完成一次 CPU structural golden。官方 release
没有提供与该 checkpoint 绑定的 frozen numerical output fixture；audit
明确记录 `author_published_numerical_golden=null`，没有把自选图片包装成
作者 golden。可执行 numerical gates 是双 smoke 与 full fresh replay。

### 11.2 A/B smoke

最终 smoke：

- A：
  `hifi_ifdl_general750001_balanced250_v1_smoke5x7_a_r2_20260727`
  （fingerprint
  `bbd8bdd9ac0e052df99f7b8d565643f751612d61d6f26bca47028e19aab66cff`）；
- B：
  `hifi_ifdl_general750001_balanced250_v1_smoke5x7_b_r2_20260727`
  （fingerprint
  `c0f7dc6a9cc8a90b42f5a4ca48e7afaaf38ff824773eb4b8a17003755a297ce7`）。

每次覆盖七个 condition 各 5 张，共 35 张，其中 20 张 T2 适用。
Comparison 状态 `passed`：

```text
images compared:                         35
T2 artifact sets compared:               20
classification hierarchy exact:        true
fine probabilities exact:              true
T1 scores exact:                        true
embeddings exact:                       true
model distance maps exact:             true
native distance maps exact:            true
native masks exact:                    true
maximum embedding/map difference:       0.0
```

### 11.3 Formal artifact audit 与 fresh replay

Independent artifact audit 重开并验证：

- 1,025 embeddings；
- 1,025 model-space distances；
- 1,025 native distances；
- 1,025 threshold masks；
- path、file SHA、array SHA、bytes、shape、dtype 和 finite exact；
- embedding-to-center distance 与 native bilinear restore independently
  rederived；
- 750 张 full-frame 没有伪造 T2 artifacts。

Fresh full-model replay：

```text
fresh model instances:                    1
ordered model forwards:                1775
T1 heads replayed:                     1775
T2 dense artifact sets replayed:       1025
fullframe transient dense checks:       750
classification logits exact:           true
fine probabilities exact:              true
T1 scores exact:                       true
embeddings exact:                      true
model/native distances exact:          true
comparison tolerance:                   0.0
maximum difference, every payload:      0.0
```

Audit 文件的 `status=passed`、`publishable=true`；grouped job 自身
return code 为 0。同组另一个独立 PSCC job 的失败使 supervisor group
总体状态为 failed，但不改变 HiFi job、HiFi evidence 或 HiFi audit
门禁。

### 11.4 Runtime 与测试

Formal manifest interval：

```text
start: 2026-07-27T09:12:10.480655Z
end:   2026-07-27T09:22:52.786288Z
wall:  642.306 s（10 分 42.3 秒）
```

| Field | Mean | Median | P95 (`higher`) | Max |
|---|---:|---:|---:|---:|
| synchronized model-forward latency | 40.423 ms | 28.560 ms | 92.052 ms | 2,063.594 ms |
| peak CUDA allocation | — | — | — | 584,130,048 bytes（557.07 MiB） |

Latency 只覆盖 synchronized model forward，不含 decode、postprocess、
native artifact serialization/hash、bootstrap 和 fresh audit。

最终 Balanced250 runner/analyzer 在 pinned HiFi interpreter 中隔离运行：

```text
41 passed in 28.76s
```

CPU-preflight 测试要求进入进程时 CUDA 从未初始化，所以不能在前序测试已
触发 CUDA 的同一 pytest 进程中把该 fresh-process 前置条件当作可交换的
测试顺序。

### 11.5 同协议 local-method 对照

| Metric | MaskCLIP | TruFor | MVSS-Net | PSCC-Net | HiFi-IFDL |
|---|---:|---:|---:|---:|---:|
| local T1 macro AUROC | 0.488323 | **0.903579** | 0.508371 | 0.525445 | 0.484539 |
| full-frame T1 macro AUROC | 0.709403 | 0.623285 | 0.488320 | 0.588851 | **0.824704** |
| local pooled per-image pixel AP | 0.203323 | **0.676670** | 0.144810 | 0.178419 | 0.098547 |
| local pooled per-image IoU@shared threshold | 0.012472 | **0.415657** | 0.023075 | 0.005300 | 0.0000846 |
| local pooled micro IoU@shared threshold | 0.001596 | **0.077767** | 0.030353 | 0.006453 | 0.000168 |
| real mean per-image FP fraction | **0.000593** | 0.017764 | 0.032915 | 0.008259 | 0.007289 |

表中 threshold 两行统一使用共享 T2 reducer 的 operating point。PSCC-Net
的共享 `>= 0.5` 与其原生 strict `> 0.5` 不等价，因此对应两格不能重述为
PSCC 原生发布 mask 的结果。

HiFi-IFDL 是当前五个完成 Balanced250 的 local-capable checkpoints 中
full-frame T1 最强的一个，但 local T1/T2 最弱之一。这不是论文方法的
普遍排名；checkpoint、训练分布、native score 和 published threshold
都不同。

## 12. 解释、限制与复现入口

本 run 的正确表述：

```text
native hierarchical-head T1 + hypersphere-distance T2 baseline complete
local insertion T1 has no useful signal
full-frame conditional-edit T1 has strong ranking but an uncalibrated threshold
native local T2 is effectively unusable at the public threshold
full-frame T2 is not applicable
code is MIT; checkpoint commercial clearance is not established
```

不能写成：

- “HiFi 检测出 82.47% 的整图 AIGC”；
- “0.2226 pixel AP 表示定位了 22.26% 的 Trash-can pixels”；
- “0.5 threshold 已经能实用部署”；
- “full-frame forward 有 dense output，所以完成了 T2”；
- “GitHub code 是 MIT，所以 Google Drive 权重也已确认可商用”；
- “general 750001 的局部失败证明所有 HiFi 训练方案都失败”；
- “Hunyuan conditional edits 代表所有纯 T2I 或所有生成器”。

Formal ID 与结果目录是 finalized immutable evidence，不能覆盖。下面命令
只用于在干净、没有同名 finalized run 的隔离工作区复现：

```bash
export PYTHONDONTWRITEBYTECODE=1
export PYTHONHASHSEED=0
export PYTHONPYCACHEPREFIX=/root/.cache/claimforge/pycache/hifi-ifdl-balanced-v2-empty
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CUDA_VISIBLE_DEVICES=4

/root/.cache/claimforge/venvs/hifi-ifdl-0ca70d6/bin/python \
  -m eval.opensource.run_hifi_ifdl_balanced \
  --repo-root . \
  --mode formal \
  --run-id hifi_ifdl_general750001_balanced250_v1_full1775_r2_20260727 \
  --device cuda:0

/root/.cache/claimforge/venvs/hifi-ifdl-0ca70d6/bin/python \
  -m eval.opensource.analyze_hifi_ifdl_balanced \
  --repo-root . \
  --run-id hifi_ifdl_general750001_balanced250_v1_full1775_r2_20260727
```

GPU occupancy 由 grouped benchmark supervisor 与 Hunyuan keepalive
协议协调：CUDA group 前暂停并排空在途生成任务，benchmark 结束或异常时
恢复 keepalive；不应在 Hunyuan 正在生成时直接启动上述 CUDA 命令。
