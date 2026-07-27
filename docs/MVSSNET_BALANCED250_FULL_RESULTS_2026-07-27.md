# MVSS-Net（ICCV 2021 / CASIAv2）在 Balanced250 上的正式结果

日期：2026-07-27（UTC）

> **文档状态：正式完成，双 smoke 与独立全链路 stateful fresh replay
> 审计通过**
>
> 本报告只使用冻结的 1,775-image formal run、两个独立 35-image CUDA
> smoke、共享 Balanced250 T1/T2 指标、逐文件 artifact audit 和 fresh
> full-model replay。旧 Mouse run、论文表格和作者展示分数没有被拿来填写
> 本报告结果。

正式 run：
`mvssnet_casiav2_iccv2021_balanced250_v1_full1775_20260727`

Formal immutable run-config fingerprint：
`8a28c8fff5f18bda8d00bcbb40dc0f0b93784c67be40d8a1362a029801356cb6`

核心机器证据：

- [run manifest](../results/opensource/mvssnet/mvssnet_casiav2_iccv2021_balanced250_v1_full1775_20260727/manifest.json)：
  `86a77afdc647c8b251e34b10140742f779de8f4758cc29940e3b08eb37a4e5d9`
- [expected inputs](../results/opensource/mvssnet/mvssnet_casiav2_iccv2021_balanced250_v1_full1775_20260727/expected_inputs.jsonl)：
  `6b5128909eeffdbd88e61af02ca1bd191cb5460f94a23b47c87ebd0172e6d12c`
- [逐图结果](../results/opensource/mvssnet/mvssnet_casiav2_iccv2021_balanced250_v1_full1775_20260727/results.jsonl)：
  `e5d169b9ffaa3912358eb0e2c7e024369f2f246322ecf758384d839183df2740`
- [coverage summary](../results/opensource/mvssnet/mvssnet_casiav2_iccv2021_balanced250_v1_full1775_20260727/summary.json)：
  `c7aaea6bfcea03ca15f7d2dbb882b880b305547e3ba1dde702a59e532f8bfd72`
- [Balanced250 T1/T2 metrics](../results/opensource/mvssnet/mvssnet_casiav2_iccv2021_balanced250_v1_full1775_20260727/balanced250_metrics.json)：
  `57068b9f37b1895b8486f7ced92648372237b566db395fb652d6629b2662f321`
- [independent replay audit](../results/opensource/mvssnet/mvssnet_casiav2_iccv2021_balanced250_v1_full1775_20260727/independent_audit.json)：
  `3814e80de08ec130da15fae552767bbfb3a0889db96cefbf70cf61b20c7bc5ae`
- [双 smoke comparison](../results/opensource/mvssnet/_reports/mvssnet_casiav2_iccv2021_balanced250_v1_smoke5x7_a_20260727__vs__mvssnet_casiav2_iccv2021_balanced250_v1_smoke5x7_b_20260727_comparison.json)：
  `90ba89b69d8f482c3ed2a92ee66d480deffad74816cd48a7b96d1352530516bd`

最终审计状态：`replay_and_smoke_audit_passed`。

## 1. 结论摘要

MVSS-Net 是 Balanced250 上第三个完成正式扩展的 local-forensics
方法。它提供原生 dense map，并按论文定义用 map 的 global maximum
作为 T1 score；但当前官方 CASIAv2 checkpoint 在三类局部植入和三类
Hunyuan 全图条件编辑上都没有可用的零样本 T1 排序能力，T2 也很弱。

- formal coverage 为 1,775/1,775，零 error、missing 和 superseded；
- local 三条件等权 macro AUROC 为 **0.508371**，AP 为
  **0.508596**，TPR@5%FPR 为 **5.33%**；
- full-frame 三条件 macro AUROC 为 **0.488320**，AP 为
  **0.502496**，TPR@5%FPR 为 **5.60%**；
- 发布阈值 `score > 0.5` 把 independent real250 中的 240 张判为
  manipulated，specificity 只有 **4.00%**；
- 因此 local macro recall `96.53%` 和 full-frame macro recall
  `94.00%` 不是强检测证据；相应 accuracy 仅为 `50.27%` 和
  `49.00%`；
- source-matched local pooled strict ranking 为 **46.67%**，mean
  score delta 只有 `+0.007572`；
- source-matched full-frame pooled strict ranking 为 **49.47%**，
  mean delta 为 `-0.005670`；
- 750 张 local 的 pooled per-image pixel AP 为 **0.144810**，
  per-image IoU@0.5 为 **0.023075**；
- 同一批 local 像素的 pooled micro recall 为 **4.00%**，micro
  IoU 为 **0.030353**；
- 275 张 real 的 mean per-image false-positive fraction 为
  **0.032915**（约 3.29%），micro fraction 为 **0.037593**；
- 两个 smoke 的 T1、raw logits、model-space score maps、
  native PNG maps 和适用 masks 全部 byte-exact；
- fresh full stateful replay 按冻结顺序完成 1,775/1,775 forwards，
  raw logits、两级 score map、T1 payload 和 1,025 个适用 mask
  全部 exact，最大 recorded-output 差异均为 `0.0`。

当前证据支持以下结论：

1. 原始 MVSS-Net CASIAv2 checkpoint 没有从经典图像篡改稳定迁移到
   Balanced250 的三类 diffusion-generated local insertion。
2. 这个 checkpoint 也不能作为当前三类 full-frame conditional edit
   的有效 AIGC detector；local 与 full-frame AUROC 都在 chance 附近。
3. 高 fixed-threshold recall 来自模型几乎把两类都判成 positive，而不是
   有效区分。单独引用 recall 或正类 F1 会产生误导。
4. T2 对较大的 Trash-can exact-difference 区域有一些连续排序信号，
   但 pooled recall、IoU 和 real false-positive area 仍不支持可靠部署。
5. 三组 full-frame 数据是从真实源图出发的 Hunyuan 条件全图编辑，不是
   脱离真实源图独立采样的纯 T2I；本结果不能代表所有整图生成器。

## 2. 官方来源、checkpoint 与许可证

本方法对应 ICCV 2021 论文
[Image Manipulation Detection by Multi-View Multi-Scale Supervision](https://openaccess.thecvf.com/content/ICCV2021/html/Chen_Image_Manipulation_Detection_by_Multi-View_Multi-Scale_Supervision_ICCV_2021_paper.html)，
正式执行冻结作者的
[MVSS-Net repository](https://github.com/dong03/MVSS-Net)。

| 资产 | 冻结身份 | 大小 / 结构 | SHA-256 |
|---|---|---:|---|
| official source | commit `cc2aed77a823723015f95e4a6a3e344f3ddb7ccc` | tracked + untracked clean；10 个关键文件绑定 | 见 manifest |
| `mvssnet_casia.pt` | 作者 `do_pred.sh` 选择的 CASIAv2 checkpoint；official Google Drive file ID `1MHoe91a24GiBMG2JYoghPRDd4Ro6RIVq` | 588,270,735 bytes | `080bc6c3aae59f748b547dbf090786fe9d31a6e50749daaa40871e298d6a7e50` |
| checkpoint state | raw `collections.OrderedDict` | 800 tensors；146,994,922 elements | ordered keys `6f44de38d505a59fb9b0c2e548bd75ef822d00594c5e60d40ff629a371e3dcbf` |
| checkpoint tensor schema | ordered key/shape/dtype/count | 675 FP32 + 125 int64 tensors | `5755c949b4cc66709f8a2e6f6c6d9c19abba01b6e1e483c0b6647e2862560bbf` |

Checkpoint 通过 `torch.load(..., map_location="cpu", weights_only=True)`
加载。800/800 state entries 严格加载，missing 和 unexpected keys 均为空；
没有使用 unrestricted pickle。模型审计记录：

```text
parameters:            146,880,335
trainable parameters:  146,811,215
buffers:                    114,587
modules:                        400
constructor ImageNet downloads used: false
complete checkpoint strict load:     true
```

官方 constructor 会尝试下载未冻结的 ImageNet ResNet-50 权重，即使最终
checkpoint 覆盖完整 state。Adapter 在 constructor 阶段屏蔽这些外部
下载，再执行完整 strict load；最终参数和 buffer 没有因此改变。

本次主结果明确使用原始 MVSS-Net，不是 MVSS-Net++。官方公开仓库没有包含
extended paper 所述 Plus-specific ConvGeM image-classification 完整实现；
把名称包含 Plus 的权重直接塞入原始 inference path 不能被称为完整
MVSS-Net++。IMDL-BenCo 的 PIL/RGB wrapper 和重训练权重也属于另一个
variant，没有在本 run 中静默替换官方 OpenCV/BGR contract。

许可证状态不是“已开源可商用”：冻结 repository 没有 `LICENSE`、
`COPYING` 或 `NOTICE`，checkpoint 也没有单独授权条款。Audit 分类为：

```text
source_available_research_release_no_grant_found
commercial_use_cleared: false
redistribution_cleared: false
```

所以本结果证明该公开研究 release 可以被严格执行和复核，**不代表源代码、
权重或衍生服务已获商业使用或再分发许可**。商业集成需要单独权利审查或
作者授权。

## 3. 方法原理与训练边界

MVSS-Net 把 manipulation localization 建模成多视角、多尺度的边界与噪声
不一致检测：

1. RGB ResNet stream 学习图像内容与篡改区域上下文；
2. constrained Bayar residual stream 抑制部分语义内容，寻找局部成像与
   处理噪声不连续；
3. Sobel-guided multi-scale edge branch 强化 manipulation boundary；
4. position/channel attention 与 DAHead 融合多视角、多尺度信息；
5. one-channel segmentation head 输出 512×512 logits；训练同时使用
   edge、pixel 和 image-level supervision。

这个 release 没有独立 image-classification head。论文的 image-level
score 是 segmentation probability map 的 global maximum pooling
（GMP）。因此只要 262,144 个 model-space pixels 中有一个普通场景边缘、
压缩痕迹或高频纹理强烈响应，整图 T1 score 就会很高。本 run 的
independent real250 有 240 张在 `0.5` 以上，与这个能力边界一致。

Checkpoint 的训练数据冻结为 CASIAv2。CLAIMFORGE 没有在 Balanced250
上训练、微调、校准或搜索阈值。经典 splice/copy-move 边界与
diffusion-generated blend 之间存在明显 domain gap；同时 512×512
非等比 stretch 会改变小编辑的几何尺度。这些是与结果一致的解释，不是
因果 ablation，不能写成已证明的唯一失败原因。

另一个执行边界是 Bayar constrained convolution：官方 forward 会原地
重新归一化 kernel，造成 state 随 forward 顺序发生极小变化。Runner
因此冻结输入顺序；resume 会从 checkpoint 建立 fresh model，并重放全部
成功 prefix。独立 audit 也必须按同一顺序重放 1,775 次，不能逐图重建
model 后声称等价。

## 4. Frozen executable contract

正式实现入口：

- [runner](../eval/opensource/run_mvssnet_balanced.py)
- [independent analyzer](../eval/opensource/analyze_mvssnet_balanced.py)
- [legacy official-contract adapter](../eval/opensource/run_mvssnet.py)
- [legacy MVSS-Net metrics](../eval/opensource/mvssnet_metrics.py)
- [shared T1 metrics](../eval/opensource/balanced250_metrics.py)
- [shared T2 metrics](../eval/opensource/balanced250_localization_metrics.py)

冻结文件 SHA-256：

| File | SHA-256 |
|---|---|
| runner | `210941969074dfb023e162b7b70370629159322cb638b8e08bbe65203100afdd` |
| analyzer | `5af8ef726f1f75951cd8adfc49e29fce08e7683b92d5dae887c4076f2889b695` |
| runner tests | `6f3d16a72636982d7a109cb6d0c7e8208e3f2a564f3763e784db2923e52e5021` |
| analyzer tests | `129c6def9a08cb3fa61774bb549ddcb7aaf81246268f1bbe13ddf9afc5a4cfbe` |
| legacy official-contract adapter | `669aa91ef1d331885f4cfd257534e4b5d0f1c1e1d835f79606cd65535469e46c` |
| legacy MVSS-Net metrics | `c046e2acd2ef3ceb09aee3d99b27baeed03515ff87260127240476a0a1919e6e` |
| shared Balanced250 T1 metrics | `f3932099bb63b766f063a66684e1d45f6e12601337d73859591d79297dbbed1c` |
| shared Balanced250 T2 metrics | `83ac07257078fc41276742fa4b9f2eb936ac51c8ff93bf1253b8c45f2b704b2a` |

Formal manifest 绑定 10 个本地 adapter/source 文件，集合 fingerprint 为
`8b02ef547a0cc06f03ba17904676f36e182b183437dd8ad08ba72b1eaa643b0f`。
最终 analyzer 再次把当前 source、checkpoint、environment 和 manifest
证据精确核对。

### 4.1 Preprocess

| 组件 | 冻结行为 |
|---|---|
| decode | `cv2.imread(..., IMREAD_COLOR)` |
| channel order | 保留 OpenCV **BGR**，不转 RGB |
| geometry | OpenCV `INTER_LINEAR` 非等比 stretch 到 512×512 |
| scale | uint8 除以 255 |
| normalization | ImageNet mean `[0.485, 0.456, 0.406]`、std `[0.229, 0.224, 0.225]`，按现有 **BGR 顺序**应用 |
| tensor | contiguous HWC → CHW FP32 |
| precision / batch | FP32 / 1；autocast false；Apex false |

BGR 顺序、stretch 和 normalization 顺序都是 checkpoint contract 的一
部分，不能用常见 PIL/RGB pipeline 替换。官方 2021 environment 在
inference 外包了一层 NVIDIA Apex O1；当前没有兼容的 `apex.amp`，
generic PyTorch autocast 也不是精确替代，因此本 run 明确使用
deterministic FP32。它复现模型、preprocess、postprocess 和阈值语义，
但不声称与作者 CUDA 10.1/Apex environment bit-exact。

### 4.2 T1、T2 与 artifacts

Model forward 返回 one-channel 512×512 segmentation logits；auxiliary
edge output 与官方 inference 一样被丢弃。保存与评分链为：

1. logits 以 FP32 NPY 保存；
2. `sigmoid(logits)` 得到 model-space FP32 probability map 并保存；
3. T1 primary score 为该 512×512 probability map 的连续 global max；
4. probability 先乘 255 并截断成 uint8，再以 OpenCV
   `INTER_LINEAR` resize 回 native decoded H×W；
5. native uint8 map 以 `L` PNG 保存，并以 `uint8 / 255` 作为 T2
   continuous score map；
6. real/local 使用 strict `native_score > 0.5` 生成 0/255 mask；
   full-frame 不生成 mask。

T1：

- `ai_score = max(sigmoid(segmentation_logits_512))`；
- 越高越 manipulated；
- released decision 为 strict `ai_score > 0.5`；
- score 没有被声明为 calibrated probability；
- saved-native-PNG global max 作为 secondary diagnostic 保留，但共享
  primary T1 表只使用连续 model-space GMP。

T2：

- local GT 是完整 exact-difference mask；
- real GT 全零，只报告 false-positive area；
- full-frame T2 严格 `N/A`，conditioning box 不是 GT；
- 共享 T2 reducer 写作 `score >= 0.5`，而官方 mask 写作
  `score > 0.5`；由于 score 是 `uint8 / 255`，不可能精确等于
  `0.5`，两个 operator 在本 contract 下逐像素等价；
- full-frame native maps 只作 dense diagnostic，不进入 T2 loader。

Quantize-before-resize 是 load-bearing contract。先 resize float map 再
量化不等价，也不能用这种替代路径解释或改写本结果。

### 4.3 Runtime

| 项 | 值 |
|---|---|
| physical GPU identity | NVIDIA L20Z |
| manifest-recorded logical device | `cuda:0` |
| Python | CPython 3.12.3 |
| PyTorch / torchvision | `2.8.0.dev20250627+cu128` / `0.23.0.dev20250627+cu128` |
| NumPy / Pillow / imported OpenCV | 1.26.4 / 11.1.0 / 4.10.0 |
| precision / batch | FP32 / 1 |
| autocast / Apex | false / false |
| deterministic algorithms | true |
| cuDNN deterministic / benchmark | true / false |
| matmul TF32 / cuDNN TF32 | false / false |
| `CUBLAS_WORKSPACE_CONFIG` | `:4096:8` |
| seed | 42 |

物理卡序号由 grouped supervisor 选择；manifest 冻结进程内 logical
`cuda:0`、设备 identity 和完整 numerical runtime。这里证明同一冻结
runtime 的 exact reproduction，不外推成任意硬件都 byte-identical。

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
- bootstrap root seed `20260726`；
- TPR@5%FPR 使用 real score 的 0.95 `higher` quantile 和严格 `>`。

MVSS-Net 的 capability contract 是：

```text
T1: real275 + local750 + fullframe750 = 1775
T2: real275 + local750                 = 1025
fullframe T2: N/A
```

Analyzer 在 T2 阶段对 750 张 full-frame 的 map loader 调用次数严格为
0。六条件 mixed macro 只作导航摘要；local 与 full-frame 必须分别解释。

## 6. Coverage 与 artifact inventory

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

Local outputs 保存在 gitignored
`outputs/opensource/mvssnet/mvssnet_casiav2_iccv2021_balanced250_v1_full1775_20260727/`：

| Artifact | Files | Total bytes | Role |
|---|---:|---:|---|
| raw logits FP32 NPY, 512×512 | 1,775 | 1,861,449,600 | replay / sigmoid audit |
| model probability FP32 NPY, 512×512 | 1,775 | 1,861,449,600 | primary T1 / model-space diagnostic |
| official native uint8 PNG map | 1,775 | 167,967,850 | T2 for real/local；diagnostic for full-frame |
| native threshold PNG mask | 1,025 | 3,423,713 | real/local only |
| **Total** | **6,350** | **3,894,290,763** | gitignored local evidence |

Independent analyzer 对 6,350 个文件逐一重开，验证 canonical path、file
SHA、array SHA、shape、dtype、finite range、sigmoid chain、
quantize-before-resize postprocess，并确认 1,025 个 mask 逐像素等于
official strict threshold；750 个 full-frame mask 全部不存在。

## 7. Primary T1 结果

方括号为 1,000 次 shared-cluster bootstrap 的 95% percentile interval。

### 7.1 Family macro

| Scope | AUROC [95% CI] | AP [95% CI] | TPR@5%FPR [95% CI] | Accuracy @ 0.5 | Recall @ 0.5 |
|---|---:|---:|---:|---:|---:|
| local macro | 0.508371 [0.492083, 0.523248] | 0.508596 [0.497326, 0.525934] | 0.053333 [0.038241, 0.062384] | 0.502667 | 0.965333 |
| full-frame macro | 0.488320 [0.461881, 0.515577] | 0.502496 [0.482241, 0.533813] | 0.056000 [0.031475, 0.092164] | 0.490000 | 0.940000 |
| all-six mixed macro | 0.498345 [0.480470, 0.515888] | 0.505546 [0.491872, 0.524846] | 0.054667 [0.037398, 0.072941] | 0.496333 | 0.952667 |

Local 与 full-frame 的 AUROC interval 都覆盖 `0.5`。接近 95% 的 recall
不能与 chance-level ranking 分开解读；它由极低的 fixed threshold
specificity 驱动。

### 7.2 条件级 ranking 与 low-FPR

| Condition | AUROC [95% CI] | AP [95% CI] | TPR@5%FPR [95% CI] |
|---|---:|---:|---:|
| local mouse | 0.499336 [0.480309, 0.517151] | 0.504168 [0.490124, 0.522928] | 0.052000 [0.032517, 0.062997] |
| local cat | 0.501608 [0.479053, 0.522243] | 0.501510 [0.480648, 0.527041] | 0.048000 [0.026920, 0.066419] |
| local trash can | 0.524168 [0.503129, 0.544669] | 0.520109 [0.505331, 0.542098] | 0.060000 [0.037654, 0.077242] |
| full-frame mouse | 0.506080 [0.473073, 0.540239] | 0.504537 [0.479007, 0.540795] | 0.048000 [0.020916, 0.085834] |
| full-frame cat | 0.493920 [0.463296, 0.526058] | 0.517972 [0.488384, 0.557233] | 0.080000 [0.043155, 0.114431] |
| full-frame trash can | 0.464960 [0.430492, 0.498798] | 0.484980 [0.460434, 0.516654] | 0.040000 [0.015254, 0.089497] |

六个 condition 共用同一个 independent real250。Point-estimate 5% FPR
threshold 都是 `0.9953274726867676`，实际 FPR 为 `0.048`。这说明 GMP
real score 本身已高度饱和；部署在 low-FPR operating point 时必须把
threshold 推到接近 1。

### 7.3 Released threshold `score > 0.5`

| Condition | Accuracy | Precision | Recall | F1 | TP / FP / FN / TN |
|---|---:|---:|---:|---:|---:|
| local mouse | 0.498 | 0.498956 | 0.956 | 0.655693 | 239 / 240 / 11 / 10 |
| local cat | 0.506 | 0.503106 | 0.972 | 0.663029 | 243 / 240 / 7 / 10 |
| local trash can | 0.504 | 0.502075 | 0.968 | 0.661202 | 242 / 240 / 8 / 10 |
| full-frame mouse | 0.488 | 0.493671 | 0.936 | 0.646409 | 234 / 240 / 16 / 10 |
| full-frame cat | 0.488 | 0.493671 | 0.936 | 0.646409 | 234 / 240 / 16 / 10 |
| full-frame trash can | 0.494 | 0.496855 | 0.948 | 0.651994 | 237 / 240 / 13 / 10 |

六组 FP 都是同一个 real250 panel 的 240 张，不是 1,440 张不同 real。
Common specificity 只有 `0.04`，FPR 为 `0.96`。在 balanced binary
dataset 上，几乎全报 positive 的 classifier 本来就能取得接近 `2/3`
的正类 F1；所以这里约 `0.65` 的 F1 不是有效检测证据。

### 7.4 Domain

| Family | Domain | AUROC | AP | TPR@5%FPR | Accuracy @ 0.5 | Recall @ 0.5 |
|---|---|---:|---:|---:|---:|---:|
| local | lodging | 0.522994 | 0.529619 | 0.043445 | 0.513021 | 0.956573 |
| local | restaurant | 0.484824 | 0.482454 | 0.038692 | 0.490627 | 0.976243 |
| full-frame | lodging | 0.480729 | 0.502037 | 0.037053 | 0.497399 | 0.933434 |
| full-frame | restaurant | 0.496426 | 0.517090 | 0.066906 | 0.481339 | 0.947396 |

这些 domain slices 都没有改变总体结论。这里没有预注册直接
domain-difference simultaneous test，只作描述。

## 8. Source-matched secondary

Secondary 只使用 frozen `source_pairs.jsonl` 的显式端点，回答“同一源图
经过编辑后 GMP score 是否上移”。它不替代 independent single-image
primary。

| Scope | Pairs | Clusters | Mean delta [95% CI] | Median delta | Strict ranking [95% CI] | W / L / T |
|---|---:|---:|---:|---:|---:|---:|
| local mouse | 250 | 247 | +0.004967 [-0.001500, +0.012272] | -0.000013 | 0.416000 [0.351643, 0.472991] | 104 / 141 / 5 |
| local cat | 250 | 247 | +0.006171 [-0.002794, +0.016847] | -0.000008 | 0.468000 [0.408372, 0.528963] | 117 / 132 / 1 |
| local trash can | 250 | 246 | +0.011579 [+0.004305, +0.019983] | +0.000003 | 0.516000 [0.457359, 0.577874] | 129 / 120 / 1 |
| full-frame mouse | 250 | 247 | +0.006679 [-0.013266, +0.024810] | +0.004874 | 0.544000 [0.483450, 0.606317] | 136 / 114 / 0 |
| full-frame cat | 250 | 248 | -0.004415 [-0.021132, +0.010824] | -0.000215 | 0.496000 [0.433062, 0.553291] | 124 / 126 / 0 |
| full-frame trash can | 250 | 246 | -0.019274 [-0.037747, -0.003477] | -0.012775 | 0.444000 [0.385099, 0.505848] | 111 / 139 / 0 |
| **local pooled** | **750** | **270** | **+0.007572 [+0.002235, +0.013665]** | **-0.000006** | **0.466667 [0.428919, 0.501488]** | **350 / 393 / 7** |
| **full-frame pooled** | **750** | **269** | **-0.005670 [-0.020858, +0.007304]** | **-0.000247** | **0.494667 [0.446708, 0.540594]** | **371 / 379 / 0** |

Local pooled mean delta 略大于零，但 median 为负，strict wins 少于
losses。这来自少数较大正 delta 拉高均值，不能表述成 46.67% 的部署
accuracy。Matched 与 primary 都没有显示稳定可用的 local/full-frame
T1 signal。

## 9. Native T2 localization

T2 continuous score 是 official native uint8 map 除以 255。以下
per-image 指标先逐图计算，再对图像平均。

### 9.1 Per-image metrics

| Scope | GT positive fraction | Pixel AP [95% CI] | Precision@0.5 | Recall@0.5 | F1@0.5 | IoU@0.5 | MCC@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| local mouse | 0.001350 | 0.032432 [0.020342, 0.047163] | 0.031937 | 0.067880 | 0.019660 | 0.012611 | 0.023995 |
| local cat | 0.063678 | 0.136606 [0.112974, 0.162593] | 0.152966 | 0.069460 | 0.042796 | 0.027081 | 0.041436 |
| local trash can | 0.219667 | 0.265391 [0.235993, 0.290618] | 0.312404 | 0.057941 | 0.050671 | 0.029532 | 0.031050 |
| **750-image pooled** | **0.095385** | **0.144810 [0.133023, 0.158376]** | **0.166324** | **0.065094** | **0.037709** | **0.023075** | **0.032194** |

Pooled precision 和 MCC 在 723/750 张有定义；27 张空 prediction mask
对应的 precision/MCC 为 undefined，而 recall、F1、IoU 和 continuous
AP 仍按共享协议处理。

Trash-can 的 pixel AP 比 Mouse 高，但 GT positive fraction 也从
`0.001350` 增至 `0.219667`。这不是 object category 的纯因果比较；
mask size、domain 和生成质量都同时变化。

### 9.2 Pooled-pixel micro metrics

| Scope | Predicted positive fraction | Micro precision | Micro recall | Micro F1 | Micro IoU | Micro MCC |
|---|---:|---:|---:|---:|---:|---:|
| local mouse | 0.036077 | 0.003554 | 0.094977 | 0.006851 | 0.003437 | 0.011612 |
| local cat | 0.032748 | 0.074244 | 0.038181 | 0.050429 | 0.025867 | 0.007962 |
| local trash can | 0.034052 | 0.259655 | 0.040251 | 0.069697 | 0.036107 | 0.018134 |
| **all local pixels** | **0.034291** | **0.111402** | **0.040049** | **0.058917** | **0.030353** | **0.010275** |

750 张 local 一共包含 1,206,543,236 个像素：

```text
TP     4,609,054
FP    36,764,136
FN   110,477,404
TN 1,054,692,642
```

模型只覆盖约 4.00% 的 GT pixels，同时产生约 3,676 万 FP pixels。
Per-image AP `0.144810` 不能被解释成定位了 14.48% 的伪造像素；AP 是
ranking summary，不是 coverage。

### 9.3 Real-image false-positive area

275 张 real 的 GT 全零，因此 pixel AP 为 `null`。固定阈值下：

- false-positive pixels：16,625,700 / 442,253,004；
- mean per-image FP fraction：
  `0.032915 [0.026277, 0.040349]`；
- micro FP fraction：
  `0.037593 [0.029036, 0.046706]`；
- 261/275 张 real 至少有一个 positive pixel；
- median per-image FP fraction 为 `0.004962`；
- maximum 为 `0.399132`。

因此 nonempty MVSS-Net mask 不能独立作为“图像被篡改”的证据。少量
real images 上的大面积响应也解释了 mean、median 和 micro fraction
之间的差异。

### 9.4 与论文指标及 full-frame 的边界

CLAIMFORGE 使用跨方法共享 exact-difference GT，不做看过 GT 后的 map
翻转、不调 oracle threshold，也不把 annotation/conditioning box 当成
mask。Full-frame 没有合法局部 GT，所以 T2 为 N/A；保存 dense map
不等于完成 full-frame localization evaluation。

这里的 T2 数值属于当前 frozen postprocess 与共享指标，不能直接拿来填
论文 benchmark 表格或与使用不同 boundary tolerance、GT 和
postprocess 的数字横向比较。

## 10. Determinism 与独立审计

### 10.1 CPU preflight 与 structural golden

CPU preflight 在 accelerator configuration 和任何 Balanced250 model
score 之前完成：

- `cuda_initialized_before=false`；
- `cuda_initialized_after=false`；
- model forwards：0；
- Balanced250 model scores：0；
- official source、checkpoint、environment 与 adapter sources exact；
- 800/800 strict state load；
- parameters、buffers、modules 与冻结值一致。

独立 analyzer 再做一次 CPU strict-load structural golden，状态为
`independent_cpu_structural_golden_passed`。作者没有发布与该 checkpoint
绑定的 frozen numerical output fixture；audit 明确记录
`author_published_numerical_golden=null`，没有把任意自选图伪装成作者
golden。

### 10.2 A/B smoke

最终 smoke：

- A：
  `mvssnet_casiav2_iccv2021_balanced250_v1_smoke5x7_a_20260727`
  （fingerprint
  `8010eb96ba65f73d10f8bea40190f39b52f002fc147db54ac20252177c864bda`）；
- B：
  `mvssnet_casiav2_iccv2021_balanced250_v1_smoke5x7_b_20260727`
  （fingerprint
  `507e203514fe95ad7082523b5315106a1a861fe1b809355808095f287f7ff0b2`）。

每次覆盖七个 condition 各 5 张，共 35 张，其中 20 张 T2 适用。
Comparison 状态为 `exact_reproduction_passed`：

- 35/35 computational result projection exact；
- T1 GMP scores exact；
- 35 对 raw-logit NPY file bytes exact；
- 35 对 model-space probability NPY file bytes exact；
- 35 对 native score-map PNG file bytes exact；
- 20 对 applicable threshold PNG file bytes exact；
- 15 张 full-frame 在两个 run 中都没有 mask；
- strict localization summaries、stateful order 和 recorded runtime exact。

### 10.3 Formal artifact audit 与 fresh stateful replay

Formal artifact audit：

- 1,775 raw logits verified；
- 1,775 model-space FP32 probability maps verified；
- 1,775 official native uint8 maps verified；
- 1,025 applicable threshold PNGs verified；
- 750 full-frame masks absent；
- 全部 path、file hash、array hash、shape、dtype、range exact；
- official quantize-before-resize postprocess exact；
- full-frame localization metrics absent。

Fresh full-model replay：

```text
fresh model instances:                 1
selected images reopened:           1775
selected images preprocessed:       1775
ordered model forwards:             1775
Bayar prefix forwards replayed:     1775
raw logits compared exact:          1775 / 1775
model score maps compared exact:    1775 / 1775
native score maps rederived exact:  1775 / 1775
T1 payloads compared exact:         1775 / 1775
applicable masks rederived exact:   1025 / 1025
fullframe masks not created:         750 / 750
maximum raw-logit abs diff:              0.0
maximum model-map abs diff:              0.0
maximum native-map abs diff:             0.0
```

GPU output 与 fresh replay 完全相同。额外的静态 CPU sigmoid sanity check
最大差异为 `8.940696716308594e-08`；它是跨 device implementation 的
有限精度检查，不是 recorded GPU replay 的放宽 tolerance。所有 T1
payload 和 1,025 张 T2 native maps 已 exact，audit 因而记录
`fresh_model_metrics_exact=true`。

## 11. Runtime、测试与同协议对照

Formal manifest interval 为
`2026-07-27T08:24:06.476506Z` 至
`2026-07-27T08:36:27.749004Z`，约 741.272 秒（12 分 21.3 秒）。

| Field | Mean | Median | P95 (`higher`) | Max |
|---|---:|---:|---:|---:|
| model-forward latency | 27.232 ms | 21.912 ms | 46.458 ms | 575.202 ms |
| peak CUDA allocation | — | — | — | 757,986,816 bytes（约 0.706 GiB） |

Latency 不包含完整 JPEG decode、artifact serialization/hash、bootstrap
和 fresh audit；它是运行元数据，不是端到端吞吐 benchmark。

最终 pinned MVSS-Net venv CPU-only 回归：

```text
43 passed in 6.83s
```

覆盖 selection、resume 与 ordered Bayar prefix replay、CPU preflight、
safe roots、strict JSON、source/checkpoint/license binding、BGR preprocess、
quantize-before-resize、T1/T2 三态 semantics、artifact fail-closed audit、
full-frame T2 exclusion、exact smoke comparison 和默认 full fresh replay。

与前两个 local 方法的同协议点估计对照：

| Metric | MaskCLIP | TruFor | MVSS-Net |
|---|---:|---:|---:|
| local T1 macro AUROC | 0.488323 | **0.903579** | 0.508371 |
| full-frame T1 macro AUROC | **0.709403** | 0.623285 | 0.488320 |
| local pooled per-image pixel AP | 0.203323 | **0.676670** | 0.144810 |
| local pooled per-image IoU@0.5 | 0.012472 | **0.415657** | 0.023075 |
| local pooled micro IoU@0.5 | 0.001596 | **0.077767** | 0.030353 |
| real mean per-image FP fraction | **0.000593** | 0.017764 | 0.032915 |

这不是论文方法的普遍排名，只是当前 checkpoint、Balanced250 数据和共享
协议下的结果。MVSS-Net 的 threshold IoU 略高于 MaskCLIP，但 continuous
AP 更低，real FP area 又高约 55 倍；不能把单个 IoU point estimate
写成总体更强。

## 12. 解释、限制与复现入口

本 run 的正确表述是：

```text
native map-GMP T1 + native T2 local-forensics baseline complete
local insertion T1 near chance
full-frame conditional-edit T1 near chance
native local T2 weak, with low recall and broad pristine firing
full-frame T2 not applicable
commercial and redistribution clearance false
```

不能写成：

- “95% recall 说明 MVSS-Net 能检测大多数 AIGC”；
- “正类 F1 约 0.65 说明模型有效”；
- “pixel AP 0.145 表示定位了 14.5% 的 forged pixels”；
- “full-frame dense map 已完成 T2”；
- “公开 GitHub repository 等于 OSI license 或商用许可”；
- “原始 CASIAv2 checkpoint 的失败证明所有 MVSS-Net 变体都失败”。

Formal ID 和结果目录是 finalized immutable evidence，不能覆盖。下面命令
只用于在干净、没有同名 finalized run 的隔离工作区复现完整流程：

```bash
export PYTHONDONTWRITEBYTECODE=1
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CUDA_VISIBLE_DEVICES=6

/root/.cache/claimforge/venvs/mvssnet-cc2aed7/bin/python \
  -m eval.opensource.run_mvssnet_balanced \
  --repo-root . \
  --mode formal \
  --run-id mvssnet_casiav2_iccv2021_balanced250_v1_full1775_20260727 \
  --device cuda:0

/root/.cache/claimforge/venvs/mvssnet-cc2aed7/bin/python \
  -m eval.opensource.analyze_mvssnet_balanced \
  --repo-root . \
  --run-id mvssnet_casiav2_iccv2021_balanced250_v1_full1775_20260727 \
  --device cuda:0
```

GPU occupancy 由项目的 grouped benchmark supervisor 与 Hunyuan
keepalive 协议协调：CPU-only preparation、文档和 push 阶段保持
keepalive；CUDA group 前暂停并排空在途任务，按显式 GPU pin 运行，
整个 group 结束或异常时恢复一次。
