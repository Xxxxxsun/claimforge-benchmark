# B-Free DINO2reg4 在 Balanced250 上的正式结果

日期：2026-07-27（UTC）

> **文档状态：正式完成，独立全链路 fresh replay 审计通过**
>
> 本报告只使用冻结的 1,775-image formal run、两个独立 35-image CUDA
> smoke、共享 Balanced250 T1 指标、persisted-artifact replay 和 fresh
> full-model replay。旧 Mouse run、论文表格和项目页展示分数没有被拿来
> 填写本报告结果。

正式 run：
`bfree_dino2reg4_balanced250_v1_full1775_20260727`

Formal immutable run-config fingerprint：
`080dfaaed8b453a06b6532faabe61bdd4c03b6eabb38da932871b00e62b24fa8`

核心机器证据：

- [run manifest](../results/opensource/bfree/bfree_dino2reg4_balanced250_v1_full1775_20260727/manifest.json)：
  `2c3b7f7b3ce6e18b07f52a9c594ab4e97a891d231f2cd141816941e936148ed8`
- [expected inputs](../results/opensource/bfree/bfree_dino2reg4_balanced250_v1_full1775_20260727/expected_inputs.jsonl)：
  `6b5128909eeffdbd88e61af02ca1bd191cb5460f94a23b47c87ebd0172e6d12c`
- [逐图结果](../results/opensource/bfree/bfree_dino2reg4_balanced250_v1_full1775_20260727/results.jsonl)：
  `b6103a6547773eeba5b2c5946aed18382f822914140b15237944a73620c889d7`
- [coverage summary](../results/opensource/bfree/bfree_dino2reg4_balanced250_v1_full1775_20260727/summary.json)：
  `0a531f8d97019b446ad98de7e6533ad8b054f7a62bfa79292100657feb49405c`
- [Balanced250 metrics](../results/opensource/bfree/bfree_dino2reg4_balanced250_v1_full1775_20260727/balanced250_metrics.json)：
  `263bacc825ad5613a5bb3693a0c0fc24da2a98dbd2dc171e16fb2671e0b2bb78`
- [independent replay audit](../results/opensource/bfree/bfree_dino2reg4_balanced250_v1_full1775_20260727/independent_audit.json)：
  `5b5e690db66f85f163a73558f7356f9ecfe157f13a6b64f3bbedfdf10e3d1001`
- [双 smoke comparison](../results/opensource/bfree/_reports/bfree_balanced_smoke_comparison_v2_489badda0d1a6d1db693e40ccda436ebc720c284c1ef90ee66e731919a8e7893.json)：
  `59c36da68954c0d1fb856ee63e63e76c991adc2b8cd76323d6090ae1c8aed55d`

最终审计状态：`replay_audit_passed`。

## 1. 结论摘要

B-Free 的官方 `BFREE_dino2reg4` checkpoint 已按冻结的 Balanced250
whole-image T1 协议完成运行。结果呈现非常清楚的 family split：

- formal coverage 为 1,775/1,775，零 error、missing 和 superseded；
- local 三条件等权 macro AUROC 为 **0.533925**，AP 为 **0.536795**，
  TPR@5%FPR 为 **6.80%**；
- 官方严格规则 `mean raw logit > 0` 在 local 上的 macro recall 只有
  **2.00%**，macro accuracy 为 **50.80%**；
- full-frame 三条件等权 macro AUROC 为 **0.924448**，AP 为
  **0.950152**，TPR@5%FPR 为 **84.00%**；
- 官方阈值在 full-frame 上的 macro recall 为 **80.93%**，accuracy 为
  **90.27%**；
- source-matched local pooled strict ranking 为 **58.67%**，mean raw
  logit delta 为 `+0.224364`；
- source-matched full-frame pooled strict ranking 为 **95.60%**，mean
  raw logit delta 为 `+8.820814`；
- 两个 CUDA smoke 的 35/35 computational projection、NPZ 文件字节、
  feature arrays 和 crop-logit arrays 全部 exact；
- formal persisted head replay 和 fresh full-model replay 都是
  1,775/1,775，所有实际最大差异为 `0.0`。

当前证据支持以下结论：

1. B-Free 是本组 full-frame conditional edits 上很强的
   **T1 whole-image AIGC detector**。
2. 它对本组 local insertions 的单图检测接近随机，released threshold
   几乎完全不检出；不能把 full-frame 能力外推成局部植入检测能力。
3. 它对局部编辑并非完全无反应：source-matched 分数总体存在小幅正移，
   其中 Trash-can 最明显。但实际部署没有 matched real
   counterfactual，这不等于可用的单图检测性能。
4. `[5,768]` features 和五个 crop logits 是分类器内部 T1 诊断，
   **不是** pixel heatmap、mask、bbox、native dense output 或 T2
   localization。
5. 三组 full-frame 数据是从真实源图出发的 Hunyuan 条件全图编辑，不是
   脱离真实源图独立采样的纯 T2I；本 run 不能单独代表所有纯整图生成器。

## 2. 官方来源、checkpoint 与许可证

B-Free 来自 CVPR 2025 论文
[A Bias-Free Training Paradigm for More General AI-generated Image Detection](https://openaccess.thecvf.com/content/CVPR2025/html/Guillaro_A_Bias-Free_Training_Paradigm_for_More_General_AI-generated_Image_Detection_CVPR_2025_paper.html)，
正式执行冻结作者的
[官方 GitHub repository](https://github.com/grip-unina/B-Free)。

| 资产 | 冻结身份 | 大小 / 结构 | SHA-256 |
|---|---|---:|---|
| official source | commit `c6a9f898782fb466b29af01f21960b67415afb0e` | tracked clean | 18 个关键文件逐一绑定 |
| `BFREE_dino2reg4.zip` | 官方下载包 | 321,653,488 bytes | `8230fd3f0f3a64a6403acb692ce1663718ed16f36a5a4de4a68c0d273781769f` |
| `config.yaml` | release config | 153 bytes | `1f0cb4988933de06a4c2427b1b5b015baa18cea7bc5223a9f54ca5e077ec8d40` |
| `model_epoch_best.pth` | official checkpoint | 346,171,370 bytes；177 FP32 tensors；86,526,721 elements | `5948ca78f4d94e820c250d24cdf155035b4a85960443800bfe6bb7f06bffe947` |
| checkpoint schema | 完整 key/shape/dtype/count | strict complete | `e4bb9ddd115309740a70235152b7376e2c8299bb90baf243809f2a5e1665f524` |
| asset bundle | ZIP/config/checkpoint contract | — | `58859ff170ba42edd9c13bfcbc0094513de227d7001e5a261f7c37dd69db8349` |

Checkpoint 只通过
`torch.load(map_location="cpu", weights_only=True)` 加载，unsafe globals
为空。模型与 checkpoint 严格完整匹配，无 missing 或 unexpected keys；
模型构建、golden 和推理均禁用外网。

许可证边界不能省略：GRIP-UNINA 自定义许可证只允许 informational 和
nonprofit purposes，并禁止未经授权的 industrial 或 profit-oriented
use。许可证文件 SHA-256 为
`cd00edf99fbfdbb173831bb0a4d5bfc40423c6e5041f62d7afdda220c4be8b27`。
因此这里的“公开方法”表示源码和权重可以公开取得并用于本研究评测，
**不表示 OSI 宽松开源，也不表示已获得商业使用许可**。

## 3. 方法原理与训练边界

B-Free 的重点不是复杂的新 classifier head，而是减少 real/fake 训练集
偏差。普通 detector 很容易学到内容、图像来源、分辨率、压缩或文件格式，
而不是真正的生成痕迹。B-Free 从同一 COCO real image 构造语义匹配的
Stable Diffusion 2.1 fake，使标签差异更集中在生成过程本身。

公开训练构造包括：

1. self-conditioned whole-image regeneration；
2. object-mask 同类替换和矩形-mask 异类替换；
3. 将局部区域外恢复为 pristine real background 的 `origBG`；
4. 加入 blur、JPEG、scaling、cut-out、noise 和 jitter 的
   `inpainted++`；
5. 端到端微调带四个 register token 的 DINOv2 ViT-B/14。

论文与项目页披露 51,517 张 COCO real 和 309,102 张 SD 2.1 generated
训练图。这个构造解释了它为什么能在生成器变化后仍抓住一些共性：
语义匹配降低了“猫、床、房间内容本身”等捷径，DINOv2 backbone 又提供
较强的跨内容表征，训练被迫更多依靠生成区域的纹理、频率和一致性痕迹。

本次 family split 也说明了限制：

- full-frame 输入的大部分 crop 都经历 Hunyuan 管线，生成痕迹容易在五个
  crop 中形成一致证据；
- local insertion 只改变一小部分区域，五个 504×504 crop 可能完全看不到
  或只部分看到它；
- 即使 crop 看见了局部，五个 raw logits 的平均仍会被真实背景稀释；
- 训练生成器是 SD 2.1，而本数据来自 Hunyuan；
- COCO 与 lodging/restaurant、训练处理与 canonical JPEG-Q95 均有域差异。

Release 没有提供可从头重建该 checkpoint 的完整训练代码。因此本评测是
**pinned official release inference**，不是从头训练复现，也不是论文
27-generator benchmark 的完整复现。

## 4. Frozen executable contract

正式实现入口：

- [runner](../eval/opensource/run_bfree_balanced.py)
- [independent analyzer](../eval/opensource/analyze_bfree_balanced.py)

冻结文件 SHA-256：

| File | SHA-256 |
|---|---|
| runner | `53ccb6d1ea5bf652a1cd319bde843c1fe378ec1c6ccdc06c6408816b793d4a9d` |
| analyzer | `e45880300a9dfb19ca6d3b1533797f06e2c6fd25da2dcdbe3a4e61ff60f85b1f` |
| runner tests | `6e385e744402265c0722d9b9eea166de27b6b3a00b4b6562cb2167297c9933fb` |
| analyzer tests | `5adb24fb84dbf2266723d7626a05a5cea29349fb6e2240d8df46185846f5df45` |

### 4.1 Preprocess 与五 crop

| 组件 | 冻结行为 |
|---|---|
| decode | `Pillow.Image.open(...).convert("RGB")`；无 EXIF transpose、无 ICC conversion |
| resize | 无；保持 native decoded size |
| tensor | torchvision `ToTensor`，uint8 除以 255 得 FP32 |
| normalization | ImageNet/ResNet mean `[0.485,0.456,0.406]`、std `[0.229,0.224,0.225]` |
| patch projection | kernel 14、stride 14；丢弃不足 14 像素的右/下 remainder |
| normal path | center、top-left、bottom-left、bottom-right、top-right 五个 36×36-token crop |
| crop size | 504×504 pixels |
| small-grid path | 任一 grid 维度小于 36 时执行 release 的 periodic `replicate_wrap`，再截到 36×36 |
| model | DINOv2 ViT-B/14、4 register tokens、CLS global pool、768→1 linear head |
| runtime | batch 1、FP32、`eval()`、无 autocast、TF32 disabled、deterministic algorithms |

“原生分辨率”和“五 crop”不表示所有像素都被模型消费。正常路径只消费五个
504×504 receptive-field rectangles 的 union；小图 wrap 后五个 crop
甚至可能完全相同。本报告的 local visibility 只描述输入是否进入这些
receptive fields，不把它解释成 localization。

### 4.2 Score 与 artifact

- 每图得到 `[5,768]` FP32 pre-head features；
- 同一个官方 linear head 产生 `[5]` FP32 crop logits；
- primary `ai_score` 是五个 crop raw logits 的 **FP32 mean**；
- 分数越高越 fake，released decision 为严格 `raw logit > 0`；
- `sigmoid(raw logit)` 只保存为诊断 probability，不替代主分数；
- 每图保存一个 canonical NPZ，固定为 15,904 bytes；
- feature/crop-logit 数组和 NPZ 字节均有 SHA-256。

这些 artifact 用于证明 T1 执行忠实性，不进入 T2 指标。

### 4.3 Runtime

| 项 | 值 |
|---|---|
| physical GPU | NVIDIA L20Z |
| manifest-recorded logical device | `cuda:0` |
| CUDA | 12.8 |
| Python | CPython 3.12.3 |
| PyTorch | `2.8.0.dev20250627+cu128` |
| torchvision | `0.23.0.dev20250627+cu128` |
| timm | `1.0.12` |
| NumPy | `2.2.6` |
| cuBLAS workspace | `:4096:8` |
| deterministic algorithms | enabled，warn-only=false |

## 5. Frozen Balanced250 设计与任务边界

共享协议包含：

- 1,775 个唯一 score-cache inputs；
- 1,750-row primary panel：real 与六个 forged condition 各 250；
- 1,500 个显式 source-matched pairs；
- primary：每个 forged condition 对独立 `real250`；
- secondary：只使用 `source_pairs.jsonl` 的显式端点；
- 不从 `task_id` 推断配对；
- 1,000 次 shared-source-content-cluster Poisson bootstrap；
- bootstrap seed `20260726`；
- TPR@5%FPR 使用 real score 的 0.95 `higher` quantile 和严格 `>`。

六条件总 macro 把 local 与 full-frame 两类不同任务混合，只作为导航摘要。
local macro 与 full-frame macro 是主要 family-level 结果。

B-Free 没有 native dense output，因此：

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
- same-device artifact head replays：1,775；
- NPZ files：1,775；
- 每个 NPZ：15,904 bytes；
- artifact inventory SHA-256：
  `5a20f54d4fc693be3c53ca5a9c4bea820c2f23f0cd1b2730644bba45280f1f64`。

### 6.2 Frozen local visibility

| Condition | Full | Partial | None | Mean visible GT | Wrap | Distinct starts `1/3/5` |
|---|---:|---:|---:|---:|---:|---:|
| local mouse | 159 | 32 | 59 | 0.692834 | 23 | 23 / 1 / 226 |
| local cat | 111 | 118 | 21 | 0.758272 | 24 | 24 / 1 / 225 |
| local trash can | 47 | 195 | 8 | 0.709694 | 21 | 21 / 1 / 228 |
| **Local pooled** | **317** | **345** | **88** | **0.720267** | **68** | **68 / 3 / 679** |

全部 1,775 张图中有 165 张触发 `replicate_wrap`；distinct crop starts
总 census 为 `1:165, 3:7, 5:1603`。

以下是从已哈希结果复算的描述性 exact-equality 诊断，不是新增 bootstrap
推断：

| Visibility | Pairs | Equal features/logits/raw | W / L / T | Same decision |
|---|---:|---:|---:|---:|
| full | 317 | 0 | 204 / 113 / 0 | 313 |
| partial | 345 | 0 | 236 / 109 / 0 | 337 |
| none | 88 | 87 | 0 / 1 / 87 | 88 |

87/88 个 `none` pair 的五个 crop classifier evidence 完全相同，说明
crop coverage 确实会造成结构性盲区。但 full/partial pair 即使全部改变
feature，overall local primary AUROC 与 fixed-threshold recall 仍很弱，
所以 coverage 不是 local 失败的唯一原因。

## 7. Primary T1 结果

### 7.1 Family macro

方括号为 1,000 次 shared-cluster bootstrap 的 95% percentile interval。

| Scope | AUROC [95% CI] | AP [95% CI] | TPR@5%FPR [95% CI] | Accuracy @ 0 [95% CI] | Recall @ 0 [95% CI] |
|---|---:|---:|---:|---:|---:|
| local macro | 0.533925 [0.520867, 0.549662] | 0.536795 [0.524674, 0.556145] | 0.068000 [0.040416, 0.098443] | 0.508000 [0.497094, 0.518681] | 0.020000 [0.007746, 0.032769] |
| full-frame macro | 0.924448 [0.901878, 0.945413] | 0.950152 [0.936566, 0.964287] | 0.840000 [0.798189, 0.880989] | 0.902667 [0.879669, 0.925746] | 0.809333 [0.765096, 0.853440] |
| all-six mixed macro | 0.729187 [0.714048, 0.743734] | 0.743473 [0.734593, 0.755629] | 0.454000 [0.428877, 0.480493] | 0.705333 [0.692131, 0.718180] | 0.414667 [0.392899, 0.436865] |

`all-six` 混合值位于两类任务之间，不能用它掩盖 local/full-frame
分裂。

### 7.2 条件级 ranking 与 5% FPR

| Condition | AUROC [95% CI] | AP [95% CI] | TPR@5%FPR [95% CI] |
|---|---:|---:|---:|
| local mouse | 0.505104 [0.490605, 0.521418] | 0.506654 [0.491775, 0.525476] | 0.048000 [0.033188, 0.072290] |
| local cat | 0.520336 [0.500742, 0.541129] | 0.530074 [0.510604, 0.554879] | 0.076000 [0.034467, 0.104355] |
| local trash can | 0.576336 [0.553955, 0.601778] | 0.573658 [0.553566, 0.602941] | 0.080000 [0.042634, 0.135135] |
| full-frame mouse | 0.927088 [0.902630, 0.949943] | 0.950238 [0.934072, 0.964917] | 0.828000 [0.776778, 0.871703] |
| full-frame cat | 0.914240 [0.887571, 0.939580] | 0.944050 [0.927099, 0.960226] | 0.836000 [0.787769, 0.881857] |
| full-frame trash can | 0.932016 [0.907215, 0.954717] | 0.956168 [0.940605, 0.970396] | 0.856000 [0.815118, 0.902657] |

所有 overall 条件共享同一个 independent real250，所以 real-only 5% FPR
threshold 都是 `-1.5235459804534912`，实际 point-estimate FPR 为
`0.048`。Threshold CI 为 `[-2.238131284713745,
-0.7606217265129089]`，actual-FPR CI 为 `[0.037815, 0.049793]`。

### 7.3 Released threshold `raw logit > 0`

| Condition | Accuracy | Precision | Recall | F1 | TP / FP / FN / TN |
|---|---:|---:|---:|---:|---:|
| local mouse | 0.502 | 0.666667 | 0.008 | 0.015810 | 2 / 1 / 248 / 249 |
| local cat | 0.512 | 0.875000 | 0.028 | 0.054264 | 7 / 1 / 243 / 249 |
| local trash can | 0.510 | 0.857143 | 0.024 | 0.046693 | 6 / 1 / 244 / 249 |
| full-frame mouse | 0.896 | 0.995000 | 0.796 | 0.884444 | 199 / 1 / 51 / 249 |
| full-frame cat | 0.904 | 0.995098 | 0.812 | 0.894273 | 203 / 1 / 47 / 249 |
| full-frame trash can | 0.908 | 0.995146 | 0.820 | 0.899123 | 205 / 1 / 45 / 249 |

表中的六个 FP 都对应各条件重复使用的同一个 real250 panel 中的一个
positive real，不是六张不同 real。Local 三条件合计只检出 15/750，
full-frame 三条件检出 607/750。由于每个条件正负样本各 250，
balanced accuracy 与表中 accuracy 相同；六条件 specificity 均为
`0.996`，FPR 均为 `0.004`。

### 7.4 Domain

| Family | Domain | AUROC | AP | TPR@5%FPR | Accuracy @ 0 | Recall @ 0 |
|---|---|---:|---:|---:|---:|---:|
| local | lodging | 0.527102 | 0.544135 | 0.060371 | 0.495668 | 0.024156 |
| local | restaurant | 0.544289 | 0.534366 | 0.077386 | 0.522367 | 0.014856 |
| full-frame | lodging | 0.884801 | 0.925147 | 0.781386 | 0.866278 | 0.744534 |
| full-frame | restaurant | 0.970539 | 0.979396 | 0.912971 | 0.944328 | 0.886442 |
| all-six mixed | lodging | 0.705951 | 0.734641 | 0.420879 | 0.680973 | 0.384345 |
| all-six mixed | restaurant | 0.757414 | 0.756881 | 0.495179 | 0.733347 | 0.450649 |

Full-frame 在两个 domain 都强，restaurant 的各项 point estimate 高于
lodging；这里没有做 simultaneous domain-difference test，因此只作描述。
Local 两个 domain 都接近随机，domain split 没有改变主要结论。

### 7.5 Raw-logit distribution

| Group | N | Mean | Median | P05 (`linear`) | P95 (`linear`) | `> 0` |
|---|---:|---:|---:|---:|---:|---:|
| real score cache | 275 | -4.657479 | -4.946311 | -6.742280 | -1.732715 | 1 |
| local mouse | 250 | -4.610520 | -4.917341 | -6.700994 | -1.648508 | 2 |
| local cat | 250 | -4.473757 | -4.891350 | -6.583154 | -1.267233 | 7 |
| local trash can | 250 | -4.199908 | -4.566382 | -6.478787 | -0.931828 | 6 |
| full-frame mouse | 250 | 4.066033 | 5.941423 | -5.082816 | 7.778944 | 199 |
| full-frame cat | 250 | 4.091840 | 5.873166 | -5.359413 | 7.833680 | 203 |
| full-frame trash can | 250 | 4.396860 | 6.125371 | -5.137605 | 7.878434 | 205 |

Local 与 real 的 absolute score distributions 大幅重叠。以 275-image
real score-cache mean 为描述性基线，三个 full-frame condition 的 mean
分别高 `8.723512`、`8.749319` 和 `9.054339` logits。这直接解释了
family-level 指标差异。

## 8. Source-matched secondary

Secondary 只使用 frozen `source_pairs.jsonl` 的真实端点。它测量同一源图
经过编辑后分数如何变化，不替代单图 primary。

| Scope | Pairs | Clusters | Mean delta [95% CI] | Median delta | Strict ranking [95% CI] | W / L / T |
|---|---:|---:|---:|---:|---:|---:|
| local mouse | 250 | 247 | 0.066794 [0.042129, 0.094181] | 0.005056 | 0.540000 [0.471997, 0.603156] | 135 / 57 / 58 |
| local cat | 250 | 247 | 0.149636 [0.060629, 0.249407] | 0.001303 | 0.504000 [0.443128, 0.562506] | 126 / 103 / 21 |
| local trash can | 250 | 246 | 0.456661 [0.361442, 0.560775] | 0.161362 | 0.716000 [0.659225, 0.766682] | 179 / 63 / 8 |
| full-frame mouse | 250 | 247 | 8.684928 [8.156191, 9.208957] | 10.288414 | 0.956000 [0.927333, 0.978075] | 239 / 11 / 0 |
| full-frame cat | 250 | 248 | 8.749484 [8.171127, 9.282281] | 10.548153 | 0.940000 [0.909072, 0.967230] | 235 / 15 / 0 |
| full-frame trash can | 250 | 246 | 9.028032 [8.509049, 9.516300] | 10.326423 | 0.972000 [0.951217, 0.991668] | 243 / 7 / 0 |
| **local pooled** | **750** | **270** | **0.224364 [0.175357, 0.276204]** | **0.020211** | **0.586667 [0.546918, 0.624689]** | **440 / 223 / 87** |
| **full-frame pooled** | **750** | **269** | **8.820814 [8.315210, 9.298574]** | **10.362077** | **0.956000 [0.935251, 0.974060]** | **717 / 33 / 0** |
| all-pairs mixed | 1,500 | 270 | 4.522589 [4.276161, 4.763455] | 0.983859 | 0.771333 [0.751320, 0.792920] | 1,157 / 256 / 87 |

Local pooled mean 为正，但 median 只有 `0.020211`，远小于 scene-to-scene
绝对分数变化。Cat 的 mean 为正而 strict ranking 接近 0.5，也说明少量
大 delta 会影响 mean；不能把 matched mean 宣传为部署级 detection。

## 9. Determinism 与独立审计

### 9.1 Golden

- CPU preflight 在加载 Balanced250 manifest 和配置 CUDA 前完成；
- 四张官方 demo 全部落在 benchmark 预先冻结的 `5e-5` acceptance
  tolerance 内；
- CPU preflight 对 published CSV 的最大 absolute difference 为
  `2.5853210448900654e-05`；
- formal CUDA 与 fresh audit 对 published CSV 的最大 absolute
  difference 为 `1.297860717741628e-05`；
- 四个 CUDA golden 都在两次 forward 中 bit-identical；
- 固定 Balanced250 golden
  `2c80d38ac19c2d3b76950996` 在 CPU 两次执行中 artifact byte-exact。

### 9.2 CUDA smoke

最终 canonical smoke：

- `bfree_dino2reg4_balanced250_v1_smoke5x7_a_r3_20260727`
- `bfree_dino2reg4_balanced250_v1_smoke5x7_b_r3_20260727`

每次覆盖七个 condition 各 5 张，共 35 张。Comparison 结果：

- computational projection exact；
- NPZ file bytes exact；
- feature arrays exact；
- crop-logit arrays exact；
- raw-logit、feature、crop-logit maximum difference 均为 `0.0`；
- A/B persisted head replay 都为 35/35，maximum difference `0.0`；
- comparison 后再次重验全部 evidence。

### 9.3 Formal replay

Persisted artifact replay：

- artifacts/head/strict decisions：1,775/1,775；
- crop-logit maximum difference：`0.0`，tolerance `0.0`；
- raw-logit maximum difference：`0.0`，tolerance `0.0`；
- CPU 对 CUDA mean 的静态 sanity tolerance 为 `2e-6`，不冒充
  same-device exact replay。

Fresh full-model replay：

- freshly reopened：1,775；
- freshly preprocessed：1,775；
- complete model forwards：1,775；
- features/crop logits/raw logits compared：各 1,775；
- maximum feature difference：`0.0`，tolerance `0.0`；
- maximum crop-logit difference：`0.0`，tolerance `1e-6`；
- maximum raw-logit difference：`0.0`，tolerance `1e-6`；
- `all_passed=true`。

这证明当前结果由冻结官方模型与 preprocessing 可复现，但不把执行
正确性误写成所有外部分布都有效。

## 10. Runtime 与测试

Manifest-enveloped formal run 从 `2026-07-27T04:30:37Z` 到
`04:46:56Z`，总 wall time `978.891 s`，覆盖 1,775 张正式推理、
artifact inventory 和 same-device final head replay。更早的 CPU
preflight、dataset/visibility 验证、CUDA/model setup 和 official golden
发生在 manifest `started_at` 之前，不计入这段 wall time。

| Field | Mean | Median | P95 (`higher`) | Max |
|---|---:|---:|---:|---:|
| preprocess latency | 122.417 ms | 128.309 ms | 160.012 ms | 178.750 ms |
| model latency | 46.675 ms | 41.445 ms | 65.685 ms | 598.893 ms |
| peak CUDA allocation | 631.73 MiB | 636.85 MiB | 640.85 MiB | 641.61 MiB |

最终联合回归覆盖新 Balanced adapter、旧 Mouse B-Free、共享
canonical/validator/contract/T1/T2 metrics：

```text
300 passed, 4 warnings
```

四个 warning 均来自第三方 Apex/PyTorch 的弃用或 docstring escape
warning，不是测试失败。专用 `PYTHONPYCACHEPREFIX` 在运行前后保持空。

## 11. 解释与限制

本 run 最重要的不是一个混合总分，而是以下边界：

- **局部植入：** B-Free 能对一些可见 edit 产生方向性响应，特别是
  Trash-can，但 absolute separation 与 released calibration 都不够；
- **整图条件编辑：** 大多数 crop 都携带一致生成痕迹，排序与官方阈值
  都很强；
- **定位：** 五 crop 不是 dense prediction，T2 必须保持 N/A；
- **数据定义：** full-frame 是 conditional edit，不是 independent
  fully synthetic T2I；
- **域迁移：** 训练域 COCO/SD2.1，测试域 lodging/restaurant/Hunyuan，
  结果不能外推到任意生成器、后处理或内容域；
- **许可证：** 研究评测完成不产生商业授权。

因此推荐在总 benchmark 中把 B-Free 记录为：

```text
whole-image T1 complete
local-image T1 evaluated but weak
T2 not applicable
commercial clearance false
```

## 12. 复现入口

本机执行时的调度环境如下；formal manifest 固定的是 logical `cuda:0` 与
NVIDIA L20Z identity，不把物理序号 `4` 作为跨机器身份：

```bash
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX=/root/.cache/claimforge/pycache/bfree-balanced-v2-empty
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export NO_ALBUMENTATIONS_UPDATE=1
export CUDA_VISIBLE_DEVICES=4
export BFREE_REPRO_RUN_ID="bfree_dino2reg4_balanced250_v1_full1775_repro_$(date -u +%Y%m%dT%H%M%SZ)"
```

已发布的 formal ID 及其目录是 finalized immutable evidence，不能直接
覆盖，也不能在已存在 metrics/audit 后用 `--resume` 重跑。上面的
`BFREE_REPRO_RUN_ID` 因而创建一个全新目录。

正式 runner：

```bash
/root/.cache/claimforge/venvs/bfree/bin/python \
  -m eval.opensource.run_bfree_balanced \
  --repo-root . \
  --mode formal \
  --run-id "$BFREE_REPRO_RUN_ID" \
  --device cuda:0 \
  --fail-fast
```

独立 analyzer：

```bash
/root/.cache/claimforge/venvs/bfree/bin/python \
  -m eval.opensource.analyze_bfree_balanced \
  --repo-root . \
  --run-id "$BFREE_REPRO_RUN_ID" \
  --device cuda:0
```

GPU occupancy 由项目的 Hunyuan keepalive 协议协调：CPU preparation、
analysis、文档、push 和全部队列结束后保持 keepalive；benchmark 真正
进入 CUDA 前暂停并排空在途任务，结束或异常时立即恢复。
