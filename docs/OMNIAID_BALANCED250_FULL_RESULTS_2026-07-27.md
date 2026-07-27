# OmniAID-DINO v2（Mirage / Auto Router）在 Balanced250 上的正式结果

日期：2026-07-27（UTC）

> **文档状态：正式完成，独立全链路 fresh replay 审计通过**
>
> 本报告只使用冻结的 1,775-image formal run、两个独立 35-image CUDA
> smoke、共享 Balanced250 T1 指标、persisted-artifact replay 和 fresh
> full-model replay。旧 Mouse run、论文表格和项目页展示分数没有被拿来
> 填写本报告结果。

正式 run：
`omniaid_dino_v2_mirage_auto_balanced250_v1_full1775_20260727`

Formal immutable run-config fingerprint：
`44becae754c6234c5810f5ef413c694609d58102540c41a21ba11d727dc7999c`

核心机器证据：

- [run manifest](../results/opensource/omniaid/omniaid_dino_v2_mirage_auto_balanced250_v1_full1775_20260727/manifest.json)：
  `656da5aa5c0670294bdc6ebf8246b233ccb7029e18019654c78c1066265a9e62`
- [expected inputs](../results/opensource/omniaid/omniaid_dino_v2_mirage_auto_balanced250_v1_full1775_20260727/expected_inputs.jsonl)：
  `6b5128909eeffdbd88e61af02ca1bd191cb5460f94a23b47c87ebd0172e6d12c`
- [逐图结果](../results/opensource/omniaid/omniaid_dino_v2_mirage_auto_balanced250_v1_full1775_20260727/results.jsonl)：
  `36cdf38e6958f99ea1f3250625c22aab9b9530716f7399c368d7e9c7e3501921`
- [coverage summary](../results/opensource/omniaid/omniaid_dino_v2_mirage_auto_balanced250_v1_full1775_20260727/summary.json)：
  `788ccc13a5fbf5e612a7f008c0cd3607a69c8a2d365fcdb26d8aa024dc4992d9`
- [Balanced250 metrics](../results/opensource/omniaid/omniaid_dino_v2_mirage_auto_balanced250_v1_full1775_20260727/balanced250_metrics.json)：
  `31e86ebafca6d5d8fbd1b6b5563b5c69e825b210cd19c8f49c50e1ed5f9a92df`
- [independent replay audit](../results/opensource/omniaid/omniaid_dino_v2_mirage_auto_balanced250_v1_full1775_20260727/independent_audit.json)：
  `87e7d373841f7e1ee9b9985746bfc4aa117ec33af886a910b595a0b28ba502e2`
- [双 smoke comparison](../results/opensource/omniaid/_reports/omniaid_balanced_smoke_comparison_v2_fbc70e8a27fded70fcc931c1b2310a271c90e4c8e5edf908e6682b21c743de90.json)：
  `028873fd236e2b9ffa6d9c13765a928c33ef3e5248761c79b1d56b08ff757ea8`

最终审计状态：`replay_audit_passed`。

## 1. 结论摘要

OmniAID-DINO v2 的正式执行完整、确定、可复核。它在三组 local
insertion 上接近随机，在三组 Hunyuan full-frame conditional edit 上有
中等排序信号，但发布阈值 operating point 仍很弱：

- formal coverage 为 1,775/1,775，零 error、missing 和 superseded；
- local 三条件等权 macro AUROC 为 **0.509205**，AP 为
  **0.514932**，TPR@5%FPR 为 **6.67%**；
- 官方规则 `class-1 float32 softmax > 0.5` 在 local 上的 macro recall
  只有 **3.07%**，macro accuracy 为 **49.93%**；
- full-frame 三条件等权 macro AUROC 为 **0.621979**，AP 为
  **0.610304**，TPR@5%FPR 为 **10.53%**；
- 发布阈值在 full-frame 上的 macro recall 只有 **5.87%**，accuracy 为
  **51.33%**；
- source-matched local pooled strict ranking 为 **49.60%**，mean
  probability delta 为 `+0.00386335`；
- source-matched full-frame pooled strict ranking 为 **72.53%**，mean
  probability delta 为 `+0.05354705`；
- 两个 CUDA smoke 的 35/35 computational projection、NPZ bytes、
  六个数组、score 和 decision 全部 exact；
- formal persisted replay 和 fresh full-model replay 都是
  1,775/1,775，六数组、probability 和 raw-logit margin 的实际最大差异
  全部为 `0.0`。

当前证据支持以下结论：

1. OmniAID 对三组小面积 local insertion 的独立单图检测与随机排序接近，
   发布阈值也没有可用召回率。
2. 它对 full-frame conditional edits 有方向正确但有限的 signal，
   strongest condition 是 full-frame Cat（AUROC `0.662864`）；整体
   low-FPR recall 与发布阈值 recall 仍明显不足。
3. Full-frame source-matched ranking 比独立 primary 强，说明同一源图经过
   全图生成管线后分数经常上移；真实部署没有 matched real
   counterfactual，因此 secondary 不能替代单图 primary。
4. 两个 1,024 维 feature、class logits 和 router gates 都是分类器内部
   T1 evidence，**不是** heatmap、mask、bbox 或 native dense output；
   OmniAID 严格不进入 T2 localization。
5. 三组 full-frame 数据是从真实源图出发的 Hunyuan 条件全图编辑，不是
   脱离真实源图独立采样的纯 T2I。本 run 不能单独代表所有纯整图生成器。

## 2. 官方来源、checkpoint 与许可证

OmniAID 对应论文
[OmniAID: Decoupling Semantics and Artifacts for Universal AI-Generated Image Detection in the Wild](https://arxiv.org/abs/2511.08423)。
正式执行冻结作者的
[官方 GitHub repository](https://github.com/yunncheng/OmniAID)、
[官方 Space](https://huggingface.co/spaces/Yunncheng/OmniAID-Demo) 和
[官方 model repository](https://huggingface.co/Yunncheng/OmniAID)。

| 资产 | 冻结身份 | 大小 / 结构 | SHA-256 |
|---|---|---:|---|
| GitHub source | commit `40749406fbcd8893c11a160edf4a72a2d4dc7056` | tracked clean；6 个关键文件绑定 | 见 manifest |
| Space source | commit `cf99ed518af8b7256854d01994d6e41165553bb3` | tracked clean；9 个关键文件绑定 | 见 manifest |
| `checkpoint_omniaid_dino_v2.pth` | HF revision `279cae7398ac6636f46fc4668f755f11210b36bf`；Mirage-Train + DDA-COCO | 3,238,483,725 bytes；2,852 FP32 tensors；808,835,239 state elements | `8135cf83a7acbd3d88e457062f7ad693b1f2e27ffc8d5ae7ec73fcb5de806ea9` |
| checkpoint schema | ordered key/shape/dtype/count | strict complete | `1b5a03a08369fa7dc5034b1b9aa8a4757295386afd3c91f093ef41b6e2c9b67d` |
| `config_omniaid_dino_v2.json` | official DINO v2 config | 696 bytes | `d97ded19543ca9459a86eddd4c0f08a8476dcd013a50f3bf81c4649f67536719` |
| DINOv3 ViT-L/16 base | revision `ea8dc2863c51be0a264bab82070e3e8836b02d51` | complete forward state embedded in OmniAID checkpoint | weight identity `dcb2e45127cccbf1601e5f42fef165eea275c8e5213197e8dcf3f48822718179` |

Checkpoint 是训练期 PyTorch archive，顶层包含 `model`、`optimizer`、
`epoch`、`scaler` 和 `args`。加载时使用 `weights_only=True` 和
`mmap=True`，只显式 allowlist 检出的 `argparse.Namespace`；未启用任意
pickle 代码执行。2,852 个 state entries 严格加载，无 missing 或
unexpected keys。Gated DINOv3 base 没有被绕过下载；checkpoint 已包含两份
DINOv3 forward 所需的完整权重，meta-device base 只用于构造官方 shape。

许可证边界不能省略：GitHub README badge、HF model card 和 Space
metadata 显示 MIT，但固定 GitHub/Space tree 中没有跟踪的 `LICENSE`、
`COPYING` 或 `NOTICE` 文本；内嵌 DINOv3-derived weights 又受 Meta
自定义 DINOv3 license 约束。因此 manifest 冻结：

```text
tracked_license_file_present: false
code_license_text_verified:    false
dinov3_base_license:           custom_dinov3_license
commercial_use_cleared:        false
benchmark_role:                research_evaluation_only
```

所以这里的“公开方法”表示源码和权重可以公开取得并用于本研究评测，
**不表示整套依赖和衍生权重已获得商业使用许可**。产品化或再分发前仍需
逐层法律审核。

## 3. 方法原理与训练边界

OmniAID 的目标是把整图 AIGC 检测中的两类线索解耦：

1. 与人物、动物、物体、场景、动漫等内容有关的 semantic flaws；
2. 与具体内容相对独立、由生成器或合成管线产生的 artifacts。

当前官方 DINO v2 / Auto Router 路径使用 hybrid mixture-of-experts：

- 5 个可路由 semantic expert：Human、Animal、Object、Scene、Anime；
- 1 个 universal Artifact expert；
- 独立 DINOv3 ViT-L/16 `feature_extractor` 产生 1,024 维 routing
  feature；
- `1024 → 256 → 5` router 从五个 semantic expert 中选择 top-2，
  两个 semantic gate 经 softmax 后和为 1；
- Artifact expert 不参与竞争，始终以 gate 1 开启；
- 因此最终六维 gate 只在三个 expert 上非零，总和为 2。

最终 MoE backbone 是另一份 DINOv3 ViT-L/16。它有 24 层、16 heads、
hidden size 1,024、MLP size 4,096、patch size 16 和 4 个 register
tokens。输入 `448×448` 时包含 `28×28` patch tokens，再加 CLS 和
register tokens。

每层 attention 的 q/k/v/o 四个 `1024×1024` linear 都替换为 SVD-MoE：

```text
24 layers × 4 projections = 96 SVD-MoE modules
main rank = 1023
rank per expert = 1
semantic experts used per image = top 2 of 5
artifact expert = index 5, always on
```

一次前向在 rank-1023 主权重上叠加两个被路由的 rank-1 semantic
residual 和固定的 rank-1 Artifact residual，最后由
`Linear(1024, 2)` 分类头输出 logits。

这种设计的潜在优势是让 semantic expert 学习“生成了什么”，让 Artifact
expert 学习“如何生成”，top-2 routing 减少不同内容域互相干扰，低秩
residual 则限制适配自由度。但它仍是全局 CLS 分类器。Local insertion
只改变很小区域，信号会被大量真实背景稀释；本结果显示 semantic/artifact
解耦没有自动转化成可用的局部植入检测。

Checkpoint 训练参数绑定 Mirage-Train 和 DDA-COCO。DINO v2 是当前官方
推荐 release，不是原始论文表格的逐项复现；本评测是
**pinned official release inference**，没有从头训练、在 Balanced250
上调参、选择 checkpoint 或拟合校准器。

## 4. Frozen executable contract

正式实现入口：

- [runner](../eval/opensource/run_omniaid_balanced.py)
- [independent analyzer](../eval/opensource/analyze_omniaid_balanced.py)

冻结文件 SHA-256：

| File | SHA-256 |
|---|---|
| runner | `b6b7e02e2e8f5dc8ccc8e90e41dfc7de78f2e0a26ed9e7941143a23df843b016` |
| analyzer | `457784137bccf47f1eb89db441c8f8c8100c49e663e55dab20eed474c63db709` |
| runner tests | `59b764de1c2f39fa0144eb3394937d05adc428f60b941d32bb5a4cc682939b4d` |
| analyzer tests | `11a048314cdce3876bd4fa05d3f365e461f413b985b3f3d8be1dacc357e8e833` |

Formal manifest 共绑定 13 个本地 adapter/source 文件；最终文档核验时
13/13 的 bytes 与 SHA-256 和当前工作树精确匹配。

### 4.1 Preprocess

本 run 严格固定当前官方 Space 的 DINO v2 / Auto Router 路径：

| 组件 | 冻结行为 |
|---|---|
| decode | `Pillow.Image.open(...).convert("RGB")` |
| EXIF / ICC | 不做 EXIF transpose；不做 ICC conversion |
| resize | 直接拉伸至 `448×448`；`torchvision.Resize([448,448])` 默认 PIL bilinear；antialias=true |
| geometry | 不保持宽高比、不 crop、无人脸对齐 |
| tensor | torchvision `ToTensor`；uint8 除以 255 得 FP32 CHW |
| normalization | ImageNet mean `[0.485,0.456,0.406]`；std `[0.229,0.224,0.225]` |
| routing | `Auto (Router)`；`manual_weights=None` |
| batch/autocast | batch 1；autocast disabled |

三组 local edit 都随整张画布进入 `448×448` 输入，所以 visibility census
为 750 个 `full`；另外 1,025 个 real/full-frame 输入为
`not_applicable`。这里的 `full` 只表示没有被 crop 掉，不是模型定位结果，
也不表示局部痕迹在全局 CLS 中足够强。

### 4.2 Score 与 artifact

- 正式 `ai_score` 是同设备 float32 softmax 的 class-1 probability；
- 分数越高越 fake，发布 decision 为严格 `ai_score > 0.5`；
- 每图保存一个 canonical `ZIP_STORED` NPZ，固定为 9,848 bytes；
- 1,775 个 formal NPZ 共 17,480,200 bytes，保存在 gitignored local
  outputs；
- NPZ file bytes 和每个数组都有 SHA-256，并逐图绑定 result row。

六个冻结数组为：

| Array | Shape | Dtype | 语义 |
|---|---:|---|---|
| `pooler_output` | `[1024]` | FP32 | 最终 MoE CLS feature、分类头之前 |
| `class_logits` | `[2]` | FP32 | real/fake logits |
| `routing_feature` | `[1024]` | FP32 | 独立 feature extractor 的 router input |
| `semantic_top_k_indices` | `[2]` | int64 | 被选中的两个 semantic expert |
| `semantic_top_k_gates` | `[2]` | FP32 | top-2 semantic weights |
| `final_gates` | `[6]` | FP32 | 五个 semantic slots 加固定 Artifact slot |

这些数组可以重放 head、float32 softmax、automatic router 和 final-gate
scatter，但都不是空间 dense prediction，因此不能进入 T2。

### 4.3 Runtime

| 项 | 值 |
|---|---|
| physical GPU identity | NVIDIA L20Z |
| manifest-recorded logical device | `cuda:0` |
| CUDA | 12.8 |
| Python | CPython 3.12.3 |
| PyTorch / torchvision | `2.8.0.dev20250627+cu128` / `0.23.0.dev20250627+cu128` |
| transformers / NumPy / Pillow | 4.57.3 / 2.2.6 / 12.0.0 |
| cuBLAS workspace | `:4096:8` |
| deterministic algorithms | enabled，warn-only=false |
| cuDNN benchmark / deterministic | false / true |
| TF32 | matmul=false；cuDNN=false |
| float32 matmul precision | `highest` |
| CPU threads / seed | 16 / 20260725 |

物理卡序号由 grouped supervisor 通过 `CUDA_VISIBLE_DEVICES` 选择；manifest
只冻结进程内 logical `cuda:0` 和设备 identity，不把宿主机物理序号当成
跨机器科学身份。

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

OmniAID 没有 native dense output，因此：

- T1 whole-image detection：有效；
- T2 local manipulation localization：N/A；
- joint T1/T2：N/A；
- full-frame T2：N/A。

## 6. Coverage 与 input visibility

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
- 每个 NPZ：9,848 bytes；
- artifact inventory SHA-256：
  `9312f4f679603cb8b72578bd7edd92eceb4d70a48d12d22a8ce2914f50db8655`。

三组 local condition 各 250 张的 GT 都在 direct full-canvas resize 中
完整可见，`edit_visible_gt_fraction=1.0`。正式 local primary 仍接近随机，
说明问题不是 crop blind spot，而是局部证据经过全局 resize、backbone 和
CLS aggregation 后没有形成稳定的单图判别信号。

## 7. Primary T1 结果

### 7.1 Family macro

方括号为 1,000 次 shared-cluster bootstrap 的 95% percentile interval。

| Scope | AUROC [95% CI] | AP [95% CI] | TPR@5%FPR [95% CI] | Accuracy @ 0.5 [95% CI] | Recall @ 0.5 [95% CI] |
|---|---:|---:|---:|---:|---:|
| local macro | 0.509205 [0.497192, 0.521272] | 0.514932 [0.505757, 0.532538] | 0.066667 [0.044735, 0.079946] | 0.499333 [0.488865, 0.509131] | 0.030667 [0.012839, 0.050859] |
| full-frame macro | 0.621979 [0.598396, 0.647861] | 0.610304 [0.586985, 0.640005] | 0.105333 [0.058122, 0.142543] | 0.513333 [0.499380, 0.527368] | 0.058667 [0.035241, 0.083976] |
| all-six mixed macro | 0.565592 [0.550794, 0.581423] | 0.562618 [0.549821, 0.583384] | 0.086000 [0.051533, 0.108547] | 0.506333 [0.495447, 0.516866] | 0.044667 [0.025490, 0.063970] |

`all-six` 混合值位于两类任务之间，不能用它掩盖 local/full-frame
分裂。

### 7.2 条件级 ranking 与 5% FPR

| Condition | AUROC [95% CI] | AP [95% CI] | TPR@5%FPR [95% CI] |
|---|---:|---:|---:|
| local mouse | 0.501408 [0.486433, 0.516021] | 0.508614 [0.495236, 0.528293] | 0.068000 [0.042634, 0.083341] |
| local cat | 0.522880 [0.505887, 0.541352] | 0.526880 [0.508713, 0.550406] | 0.084000 [0.037789, 0.103313] |
| local trash can | 0.503328 [0.488541, 0.518825] | 0.509301 [0.497356, 0.529063] | 0.048000 [0.037190, 0.084632] |
| full-frame mouse | 0.607536 [0.581610, 0.634887] | 0.588998 [0.560242, 0.622713] | 0.096000 [0.038023, 0.135341] |
| full-frame cat | 0.662864 [0.633918, 0.693525] | 0.655793 [0.624848, 0.690233] | 0.136000 [0.086134, 0.187520] |
| full-frame trash can | 0.595536 [0.568906, 0.624599] | 0.586121 [0.561795, 0.618168] | 0.084000 [0.043926, 0.127662] |

所有 overall 条件共享同一个 independent real250，所以 real-only 5% FPR
threshold 都是 `0.38159313797950745`，实际 point-estimate FPR 为
`0.048`。Threshold CI 为 `[0.2965194880962372,
0.5403862595558167]`，actual-FPR CI 为 `[0.038454, 0.049793]`。

### 7.3 Released threshold `probability > 0.5`

| Condition | Accuracy | Precision | Recall | F1 | TP / FP / FN / TN |
|---|---:|---:|---:|---:|---:|
| local mouse | 0.500 | 0.500000 | 0.032 | 0.060150 | 8 / 8 / 242 / 242 |
| local cat | 0.498 | 0.466667 | 0.028 | 0.052830 | 7 / 8 / 243 / 242 |
| local trash can | 0.500 | 0.500000 | 0.032 | 0.060150 | 8 / 8 / 242 / 242 |
| full-frame mouse | 0.506 | 0.578947 | 0.044 | 0.081784 | 11 / 8 / 239 / 242 |
| full-frame cat | 0.526 | 0.724138 | 0.084 | 0.150538 | 21 / 8 / 229 / 242 |
| full-frame trash can | 0.508 | 0.600000 | 0.048 | 0.088889 | 12 / 8 / 238 / 242 |

表中的六组 FP 都对应重复使用的同一个 real250 panel 中 8 个
`probability > 0.5` real，不是 48 张不同 real。Local 三条件只检出
23/750，full-frame 三条件也只检出 44/750。六条件 specificity 均为
`0.968`。因此 full-frame 的中等 AUROC 不能被描述为发布阈值下的可用
detector。

### 7.4 Domain

| Family | Domain | AUROC | AP | TPR@5%FPR | Accuracy @ 0.5 | Recall @ 0.5 |
|---|---|---:|---:|---:|---:|---:|
| local | lodging | 0.508492 | 0.531329 | 0.043445 | 0.484506 | 0.053107 |
| local | restaurant | 0.501915 | 0.495777 | 0.053547 | 0.516595 | 0.002950 |
| full-frame | lodging | 0.590920 | 0.594600 | 0.056961 | 0.497607 | 0.071507 |
| full-frame | restaurant | 0.662034 | 0.662586 | 0.145361 | 0.531362 | 0.043202 |
| all-six mixed | lodging | 0.549706 | 0.562964 | 0.050203 | 0.491056 | 0.062307 |
| all-six mixed | restaurant | 0.581974 | 0.579182 | 0.099454 | 0.523978 | 0.023076 |

Local 在两个 domain 都接近随机。Full-frame restaurant 的 ranking 和
low-FPR point estimate 高于 lodging，但发布阈值 recall 仍低；这里没有做
simultaneous domain-difference test，因此只作描述。

## 8. Source-matched secondary

Secondary 只使用 frozen `source_pairs.jsonl` 的真实端点。它测量同一源图
经过编辑后 probability 如何变化，不替代单图 primary。

| Scope | Pairs | Clusters | Mean delta [95% CI] | Median delta | Strict ranking [95% CI] | W / L / T |
|---|---:|---:|---:|---:|---:|---:|
| local mouse | 250 | 247 | 0.000574 [-0.000482, 0.001636] | -0.000022 | 0.484000 [0.419841, 0.544019] | 121 / 129 / 0 |
| local cat | 250 | 247 | 0.010123 [0.005761, 0.015529] | 0.000525 | 0.540000 [0.479085, 0.598414] | 135 / 115 / 0 |
| local trash can | 250 | 246 | 0.000893 [-0.001809, 0.003585] | -0.000300 | 0.464000 [0.399161, 0.528574] | 116 / 134 / 0 |
| full-frame mouse | 250 | 247 | 0.041940 [0.030977, 0.053254] | 0.018152 | 0.712000 [0.658259, 0.771666] | 178 / 72 / 0 |
| full-frame cat | 250 | 248 | 0.079690 [0.064772, 0.094007] | 0.036375 | 0.756000 [0.701942, 0.809175] | 189 / 61 / 0 |
| full-frame trash can | 250 | 246 | 0.039011 [0.028351, 0.050059] | 0.017374 | 0.708000 [0.647535, 0.768348] | 177 / 73 / 0 |
| **local pooled** | **750** | **270** | **0.003863 [0.002054, 0.005937]** | **-0.000009** | **0.496000 [0.451005, 0.535665]** | **372 / 378 / 0** |
| **full-frame pooled** | **750** | **269** | **0.053547 [0.042904, 0.064671]** | **0.022826** | **0.725333 [0.681146, 0.773189]** | **544 / 206 / 0** |
| all-pairs mixed | 1,500 | 270 | 0.028705 [0.023133, 0.034373] | 0.002078 | 0.610667 [0.576275, 0.648055] | 916 / 584 / 0 |

Local pooled mean delta 的 CI 虽为正，但 median 几乎为零，strict ranking
为 49.6%，而独立 primary 也接近随机；均值主要受到 local Cat 的少量较大
正移影响，不能据此宣称 local detection 成功。Full-frame 三条件的 matched
delta 和 ranking 都明确为正，但 70.8%–75.6% 的 counterfactual ranking
仍不能替代部署中的 unpaired single-image performance。

## 9. Determinism 与独立审计

### 9.1 CPU preflight 与官方 fixtures

CPU preflight 在加载 Balanced250 manifest 和配置 accelerator 之前完成：

- source、Space、checkpoint、config 与 2,852-entry schema 全部通过；
- 96 个 SVD-MoE module、rank、experts、head 和 RoPE buffer 全部重验；
- `cuda_used=false`，CUDA tensor operations=false；
- CPU model load 前、两次 CPU golden forward 后
  `torch.cuda.is_initialized()` 均为 false；
- 一张 frozen Balanced250 CPU fixture 连续两次的 canonical NPZ bytes、
  六数组、probability、decision 和 router scatter 全部 exact；
- CPU preflight 没有加载 Balanced250 dataset manifest，也没有计算正式
  dataset scores。

四张官方 Space 示例不是作者发布的论文级 numeric golden；它们是固定
runtime regression 加 2026-07-25 观察到的官方 service oracle：

| Fixture | Local CUDA fake p | Service abs diff | 连续六数组 |
|---|---:|---:|---|
| `examples/real_0.jpg` | 0.239961311 | 0.000002190 | exact |
| `examples/real_1.jpg` | 0.078057170 | 0.000000164 | exact |
| `examples/fake_0.jpg` | 0.857224584 | 0.000001431 | exact |
| `examples/fake_1.jpg` | 0.609500527 | 0.000001669 | exact |

全部 service 差异小于预注册 `5e-5`；本地 frozen-runtime logits、
probabilities 和 gates 的连续 forward 最大差异都是 `0.0`。Formal fresh
audit 又独立重跑这四张 CUDA fixture，六数组和分数全部 exact。

### 9.2 CUDA smoke

最终 canonical smoke：

- `omniaid_dino_v2_mirage_auto_balanced250_v1_smoke5x7_a_20260727`
  （fingerprint
  `2b1481457a4c8516bd3a3d939bd46a4892a4d0ec7495a88cd3384cb8f6dd11f5`）；
- `omniaid_dino_v2_mirage_auto_balanced250_v1_smoke5x7_b_20260727`
  （fingerprint
  `e190f11a22a2bc5d391b5f7a74ea059ffdf4595a33b9d1e8508c24c556df45e0`）。

每次覆盖七个 condition 各 5 张，共 35 张。Comparison 结果：

- 35/35 computational projection exact；
- NPZ file bytes exact；
- 六个 artifact arrays 的 SHA 与逐元素内容 exact；
- 六数组 maximum absolute difference 全部为 `0.0`；
- probability、raw-logit margin 和 decision exact；
- A/B persisted head、softmax、router 和 scatter replay 都为 35/35；
- replay 的 logit、probability、margin、indices 和 gates 最大差异全部为
  `0.0`。

### 9.3 Formal replay

Persisted artifact replay：

- artifact/head/float32-softmax/automatic-router/final-gate-scatter：
  1,775/1,775；
- class logits、probability、margin、router indices、router gates 和
  final gates 的 maximum difference 全部为 `0.0`；
- recorded runtime exact match：true。

Fresh full-model replay：

- freshly reopened and preprocessed：1,775；
- complete model forwards：1,775；
- six-array sets compared：1,775；
- `pooler_output`、`class_logits`、`routing_feature`、
  `semantic_top_k_indices`、`semantic_top_k_gates`、`final_gates` 的
  maximum difference 全部为 `0.0`；
- probability maximum difference：`0.0`；
- raw-logit margin maximum difference：`0.0`；
- absolute tolerance：`0.0`；
- fresh metrics 与 persisted metrics JSON exact。

这证明当前结果由冻结官方模型、资产和 Auto Router preprocessing 可精确
重现，但不把执行正确性误写成所有外部分布都有效。

### 9.4 被拒绝的启动不属于科学结果

并行队列第一次启动 OmniAID 时，调度环境只传入了
`PYTHONHASHSEED=0` 和 `PYTHONDONTWRITEBYTECODE=1`，缺少冻结要求的
`NO_ALBUMENTATIONS_UPDATE=1` 与绝对、空的 `PYTHONPYCACHEPREFIX`。
Runner 在 CPU preflight 的 startup-isolation gate 立即 fail closed：

```text
OmniAID startup isolation requires PYTHONHASHSEED=0,
PYTHONDONTWRITEBYTECODE=1, NO_ALBUMENTATIONS_UPDATE=1,
and an absolute empty PYTHONPYCACHEPREFIX=...
```

该尝试发生在 dataset load、checkpoint model load 和任何 Balanced250
forward 之前，没有生成图像分数、artifact 或 run evidence。随后 comparator
和 analyzer 因 run directory 不存在而失败，只是同一个启动错误的连锁结果。
因此它们是**环境启动失败，不是模型失败，也不是科学结果**；本报告只使用
补齐冻结环境后的 A/B smoke、formal 和 fresh replay。

## 10. Runtime 与测试

Formal manifest interval 为
`2026-07-27T07:14:35.784926Z` 至
`2026-07-27T07:22:36.368344Z`，约 **480.583 s**。

| Field | Mean | Median | P95 (`higher`) | Max |
|---|---:|---:|---:|---:|
| preprocess latency | 28.700 ms | 33.629 ms | 43.474 ms | 66.280 ms |
| model latency | 84.816 ms | 84.130 ms | 87.409 ms | 133.955 ms |
| peak CUDA allocation | 3,372,986,880 bytes | same | same | same |

最终定向 CPU-only 回归包含真实 3.2GB checkpoint preflight：

```text
36 passed, 1 warning in 49.04s
```

测试覆盖 frozen selection、startup isolation、safe checkpoint load、
T2 fail-closed、canonical 9,848-byte six-array NPZ、resume/error history、
35-image smoke exact comparison、persisted router/head replay、共享
Balanced250 metrics、formal 1,775 fresh replay、output scope 和 source
mutation rejection。Heavy CPU preflight 测试在前后都断言 CUDA 未初始化。

## 11. 解释与限制

本 run 最重要的不是 mixed macro，而是以下边界：

- **局部植入：** 三条件 AUROC 为 0.501–0.523，pooled matched ranking
  为 0.496；OmniAID 不能当作局部植入 detector；
- **整图条件编辑：** 有中等 ranking signal，尤其 full-frame Cat，但
  AUROC、low-FPR recall 和发布阈值 recall 都不足以支持强部署结论；
- **路由证据：** semantic/artifact gates 是整图 expert 权重，不是空间
  localization；
- **定位：** 六个 persisted arrays 都不是 dense prediction，T2 必须
  保持 N/A；
- **数据定义：** full-frame 是 conditional edit，不是 independent
  fully synthetic T2I；
- **域迁移：** 训练域 Mirage/DDA-COCO，测试域
  lodging/restaurant/Hunyuan，结果不能外推到任意生成器、内容域、压缩或
  后处理；
- **版本轴：** 当前只测试官方推荐 DINO v2 / Auto Router，没有根据本
  benchmark 在 CLIP v2、v1 或手工 expert weights 间选优；
- **校准：** class-1 softmax 是官方 score，但未在 Balanced250 域校准，
  不能解释成可信后验概率；
- **许可证：** 研究评测完成不产生商业授权。

因此推荐在总 benchmark 中把 OmniAID 记录为：

```text
whole-image T1 complete
local-image T1 evaluated but near chance
full-frame conditional-edit T1 moderate ranking but weak operating points
T2 not applicable
commercial clearance false
```

## 12. 复现入口

已发布 formal ID 及目录是 finalized immutable evidence，不能覆盖或用
`--resume` 改写。复现必须先准备一个绝对且为空的 pycache prefix，并创建
新 run ID：

```bash
export PYTHONDONTWRITEBYTECODE=1
export PYTHONHASHSEED=0
export NO_ALBUMENTATIONS_UPDATE=1
export PYTHONPYCACHEPREFIX=/root/.cache/claimforge/pycache/omniaid-balanced-v2-empty
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CUDA_VISIBLE_DEVICES=5
export OMNIAID_REPRO_RUN_ID="omniaid_dino_v2_mirage_auto_balanced250_v1_repro_$(date -u +%Y%m%dT%H%M%SZ)"
```

正式 runner：

```bash
/root/.cache/claimforge/venvs/omniaid/bin/python \
  -m eval.opensource.run_omniaid_balanced \
  --repo-root . \
  --mode formal \
  --run-id "$OMNIAID_REPRO_RUN_ID" \
  --device cuda:0 \
  --fail-fast
```

独立 analyzer：

```bash
/root/.cache/claimforge/venvs/omniaid/bin/python \
  -m eval.opensource.analyze_omniaid_balanced \
  --repo-root . \
  --run-id "$OMNIAID_REPRO_RUN_ID" \
  --device cuda:0
```

GPU occupancy 由项目的 grouped benchmark supervisor 与 Hunyuan
keepalive 协议协调：CPU preparation、文档、push 和全部队列结束后保持
keepalive；CUDA group 前暂停并排空在途任务，把兼容方法固定到显式 logical
device，整个 group 结束或异常时只恢复一次 keepalive。
