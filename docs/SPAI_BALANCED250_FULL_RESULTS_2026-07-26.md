# SPAI Any-Resolution Spectral 在 Balanced250 上的正式结果

日期：2026-07-26（UTC）

完成审计：2026-07-27（UTC）

> **文档状态：正式完成，独立全链路 fresh replay 审计通过**
>
> 本文件报告冻结的 1,775-image formal run、两个独立 35-image CUDA
> smoke、共享 Balanced250 T1 指标、persisted-artifact replay 和 fresh
> full-model replay。所有数值均来自下列机器产物；旧 Mouse run、论文表格
> 和项目网站展示分数没有用于填写本报告结果。

正式 run：
`spai_any_resolution_spectral_balanced250_v1_full1775_20260726`

核心机器证据：

- [run manifest](../results/opensource/spai/spai_any_resolution_spectral_balanced250_v1_full1775_20260726/manifest.json)：
  `0fb8134c0fabe43616821fd39c5c7d5d740d44d6bd70b9b44daa3e11cd002e63`
- [逐图结果](../results/opensource/spai/spai_any_resolution_spectral_balanced250_v1_full1775_20260726/results.jsonl)：
  `96eef0c5907ead3ff6e91554339c721f7d73629fecaf110906cb9526e09bd049`
- [coverage summary](../results/opensource/spai/spai_any_resolution_spectral_balanced250_v1_full1775_20260726/summary.json)：
  `d0af8b0550ab7ba8b07058e88bd0f6bc9b2298a6ff2ea33cae2f582fe9cc85d4`
- [Balanced250 metrics](../results/opensource/spai/spai_any_resolution_spectral_balanced250_v1_full1775_20260726/balanced250_metrics.json)：
  `15d7748ade1c8c712d1664e7411b7ec40224696951b68d7de38e0d71493908b0`
- [independent replay audit](../results/opensource/spai/spai_any_resolution_spectral_balanced250_v1_full1775_20260726/independent_audit.json)：
  `57da75de9fd6eef81463d9398adf1b24628b4020408f50ae2ebfa11ef25d3902`
- [双 smoke comparison](../results/opensource/spai/_reports/spai_balanced_smoke_comparison_v2_4593c8fc741b05274ac041b13f40dd0f5d9fef38ea2356fc04b595a0771c494e.json)：
  `cbe5772ece7327442864794d44f583c9303bbcebeb860f14dd0cc0187babd362`

最终审计状态：`fresh_full_replay_audit_passed`。

## 1. 结论摘要

SPAI 的官方公开 checkpoint 已按冻结的 Balanced250 whole-image T1
协议完整运行。最重要的结果不是“SPAI 总体有多强”，而是它在两种
manipulation family 上表现完全不同：

- formal coverage 为 1,775/1,775，零 error、missing 和 superseded；
- local 三条件等权 macro AUROC 为 **0.512581**，AP 为 **0.517997**，
  TPR@5%FPR 为 **6.53%**，接近随机排序；
- 官方 `p_fake > 0.5` 在 local 上的 macro recall 为 **19.07%**，
  accuracy 为 **50.53%**；
- full-frame 三条件等权 macro AUROC 为 **0.915432**，AP 为
  **0.937046**，TPR@5%FPR 为 **78.53%**；
- 官方阈值在 full-frame 上的 macro recall 为 **83.87%**，accuracy 为
  **82.93%**；
- source-matched local pooled strict ranking 为 **57.2%**，mean score
  delta 为 `+0.013677`；其中 Mouse 单项反而是 **41.2%** 和
  `-0.003715`；
- source-matched full-frame pooled strict ranking 为 **92.0%**，mean
  score delta 为 `+0.666923`；
- local 输入经 SPAI patch geometry 后，exact-difference GT 的
  full/partial/none visibility 为 `472/255/23`；
- 两个 CUDA smoke 的 35/35 computational projection 和三种产物均
  byte/array exact；正式 persisted replay 和 fresh full-model replay
  也都是 1,775/1,775，所有最大数值差异为 0。

因此，当前证据支持：

1. SPAI 是一个对本组 full-frame conditional edits 很强的
   **T1 whole-image AIGC detector**。
2. 它对本组 local insertions 的 image-level 泛化很弱；不能把
   full-frame 结果外推成局部植入检测能力。
3. 保存的 `[12, P]` spectral-context attention 只是分类器内部诊断，
   **不是** pixel heatmap、mask、bbox、native dense output 或 T2
   localization。
4. `fullframe_mouse`、`fullframe_cat` 和 `fullframe_trash_can` 是从真实
   源图出发的全图条件编辑，不是脱离真实源图独立采样的纯 T2I。
5. 本 run 因而不能单独回答 SPAI 对所有纯整图生成模型的泛化能力。

## 2. 官方来源、checkpoint 与训练披露

方法来自 Karageorgiou 等人的 CVPR 2025 论文
[Any-Resolution AI-Generated Image Detection by Spectral Learning](https://openaccess.thecvf.com/content/CVPR2025/html/Karageorgiou_Any-Resolution_AI-Generated_Image_Detection_by_Spectral_Learning_CVPR_2025_paper.html)，
正式执行冻结作者的
[官方 GitHub repository](https://github.com/mever-team/spai)：

| 组件 | 冻结值 |
|---|---|
| official source commit | `8ff7b3b6779b4fcb43cf313471d9cb1c62d129a4` |
| source worktree | tracked clean |
| checkpoint ID | `official-google-drive-1vvXmZqs6TVJdj8iF1oJ4L_fcgdQrp_YI` |
| checkpoint bytes | 934,865,338 |
| checkpoint SHA-256 | `24159f27d7c8c2cd0cb6c4019189eb89ad0874a0d9d15f8dc9afd39ca9648a55` |
| state tensors | 324 |
| state elements | 139,945,243 |
| state schema SHA-256 | `ffe751246ec65936d5583a1db62bf617697484e6185f1bfad7c678f1dad36ef8` |

Checkpoint 只通过
`torch.load(map_location="cpu", weights_only=True)` 加载；唯一显式安全
allowlist 是 `yacs.config.CfgNode`，没有执行任意 checkpoint pickle。
Benchmark 不重新分发 checkpoint。

论文披露的训练集为约 180K LDM-generated fake 和约 180K 来自
COCO/LSUN 的 real images。MFM ViT-B/16 backbone 在 SPAI 训练中冻结。
这些是论文级训练披露，不等价于对所有训练图像权利或商业用途的重新授权。

## 3. 方法原理

冻结的 released executable 可概括为：

```text
native RGB image in [0, 1]
-> non-overlapping 224 x 224 image patches
-> FFT frequency restoration
-> frozen MFM ViT-B/16
-> 12-layer spectral representation sequence (SRS)
-> one 1096-D representation per image patch
-> 12-head spectral context attention (SCA)
-> LayerNorm
-> complete three-linear-layer MLP
-> one raw logit
-> sigmoid probability
```

它强的原因在于不只依赖肉眼可见的语义错误。FFT restoration 和
multi-layer spectral representation 让模型关注生成管线在频率统计、
纹理与跨 patch 一致性中留下的系统性偏差；SCA 再根据整张图的多个 patch
自适应聚合证据。这很适合“整张图都经历同一生成/编辑管线”的输入，也与本次
full-frame 高分一致。

但该结构不会自动解决局部植入：

- 一个小局部只影响少数 patch，其证据可被大量真实背景稀释；
- 非整除边缘会被 patch grid 丢弃；
- 小图 fallback 是 five-crop，不是滑窗 dense localization；
- 训练目标只监督一张图的真假标签，不要求指出被修改位置；
- SCA attention 表示分类聚合权重，而不是有定位语义的监督 mask。

这也解释了本次 local macro 接近随机、full-frame macro 很强的分裂结果。

## 4. Frozen executable contract

正式实现入口：

- `eval/opensource/run_spai_balanced.py`
- `eval/opensource/analyze_spai_balanced.py`

冻结文件 SHA-256：

| File | SHA-256 |
|---|---|
| runner | `92627a9f6019ef25f7ff69cedf12ee18b5ea4c5b2761b7c2cf8857fb4a522bff` |
| analyzer | `04f9f4866dca9aaf209b81cdef987b4383bd08b75d6b4f4f819aad452f7fe69b` |
| runner tests | `9ccc2cf65ff3d23e02c4ac7c28e86a93c794c4b183ad59a5c1896eaf546f161c` |
| analyzer tests | `a8c97842dc95a9cfee57a956adcfa966643782e476eb18ca8924f1bcd49d33c6` |

Formal immutable run-config fingerprint 为
`f44c5f7b9e43c639b1b1c96765c84579ac5496725888d89b3f39b5562301837f`。
Manifest 另外逐文件绑定 official source、adapter、共享 Balanced250
loader/metrics 以及 runtime inventory。

### 4.1 Preprocess 与 patch geometry

正式 profile 为 `official_pillow_rgb_native_float32_0_1`：

- `Pillow.Image.open(...).convert("RGB")`；
- 不做 EXIF transpose 或 ICC conversion；
- 不 resize、不 crop、不做 test-time augmentation；
- uint8 乘 float32 `1/255`，不使用 ImageNet mean/std；
- 小于 `224 x 224` 时用 `cv2.BORDER_REFLECT_101` 居中补齐；
- 以 stride 224 提取不重叠 `224 x 224` patches；
- 初始 patch 少于 4 时使用 torchvision five-crop fallback；
- 右侧和下侧不能整除 224 的 remainder 被 `Tensor.unfold` 丢弃；
- batch size 1、float32、无 autocast。

这里有两个必须公开的 released-artifact discrepancy：

1. 当前 official `configs/spai.yaml` 的 `minimum_patches=4`，checkpoint
   embedded historical config 是 1；本 benchmark 执行当前 released
   inference config，不把历史值偷偷恢复。
2. 论文使用“all spectral information”类表述，但当前 executable 对
   非整除右/下边缘确实会丢弃 remainder；本报告按实际代码描述。

项目网站展示的是压缩 derivative 图片及其短小数分数。它们与原始
evaluation-bundle 文件在当前 released executable 下的回归值不一致，
因此只作为弱展示参考，不是 executable golden gate。

### 4.2 Score、decision 与保存产物

- `raw_logit`：完整 MLP 的单标量输出；
- `ai_score = sigmoid(raw_logit)`，方向为越高越 fake；
- released fixed decision 为严格 `ai_score > 0.5`；
- 每图保存 canonical finite float32 NPY：
  - `patch_features`：`[P, 1096]`；
  - `feature`：LayerNorm 后 `[1096]`；
  - `attention`：`[12, P]` 分类诊断权重。

这三类 NPY 共 5,325 个，位于 gitignored output 目录。提交的 JSON
evidence 保存其相对路径、shape、dtype、字节数和哈希；attention 明确不
进入 T2 评分。

### 4.3 Deterministic runtime

正式执行环境：

| 项 | 值 |
|---|---|
| device | NVIDIA L20Z / CUDA 12.8 |
| Python | CPython 3.12.3 |
| PyTorch | `2.8.0.dev20250627+cu128` |
| torchvision | `0.23.0.dev20250627+cu128` |
| timm | `0.4.12` |
| NumPy | `1.26.4` |
| inference dtype | float32 |
| deterministic algorithms | enabled, warn-only=false |
| cuBLAS workspace | `:4096:8` |
| TF32 | disabled |

CPU preflight 在加载 Balanced250 manifest/selection 和配置 CUDA 之前
完成：source/checkpoint/runtime、official originals、一个固定的
Balanced250 golden JPEG、两次 CPU forward 和手工
SCA/LayerNorm/complete-MLP replay 均通过；preflight 前后 `torch.cuda`
均未初始化。

## 5. Frozen Balanced250 设计

共享协议包含：

- 1,775 个唯一 score-cache inputs；
- 1,750-row independent panel：real 和六种 forged condition 各 250；
- 1,500 个显式 source-matched pairs；
- primary 是每个 forged condition 对独立 real250 的 unpaired comparison；
- secondary 只使用 `source_pairs.jsonl` 的显式端点；
- primary/secondary 都不从 `task_id` 推断配对；
- 1,000 次 shared-source-content-cluster Poisson bootstrap；
- bootstrap seed `20260726`；
- fixed threshold `ai_score > 0.5`；
- TPR@5%FPR 使用 real score 的 0.95 `higher` quantile 和严格 `>`。

六条件总 macro 混合 local 与 full-frame 两类不同任务，只作为导航摘要；
local macro 和 full-frame macro 才是主要 family-level 结果。

## 6. Coverage、artifact 与 local 输入可见性

### 6.1 Formal coverage

| Condition | Expected | Valid | Error | Missing |
|---|---:|---:|---:|---:|
| real score cache | 275 | 275 | 0 | 0 |
| local mouse | 250 | 250 | 0 | 0 |
| local cat | 250 | 250 | 0 | 0 |
| local trash can | 250 | 250 | 0 | 0 |
| full-frame mouse | 250 | 250 | 0 | 0 |
| full-frame cat | 250 | 250 | 0 | 0 |
| full-frame trash can | 250 | 250 | 0 | 0 |
| **Total** | **1,775** | **1,775** | **0** | **0** |

Runner inventory：

- physical result rows：1,775；
- latest result rows：1,775；
- superseded attempts：0；
- same-device final artifact replays：1,775；
- artifact files：5,325。

### 6.2 Local edit visibility

这是 exact-difference GT 与 SPAI **实际输入 patch geometry** 的交集诊断，
不是模型 localization 输出。

| Local condition | Full | Partial | None | Mean visible GT fraction | Grid / five-crop |
|---|---:|---:|---:|---:|---:|
| mouse | 221 | 12 | 17 | 0.907486 | 238 / 12 |
| cat | 176 | 71 | 3 | 0.925579 | 237 / 13 |
| trash can | 75 | 172 | 3 | 0.884742 | 238 / 12 |
| **Pooled** | **472** | **255** | **23** | **0.905936** | **713 / 37** |

23/750 local edits 对 SPAI 输入完全不可见，255/750 只部分可见。这会压低
local 上限，但不足以把 0.5126 macro AUROC 全部解释为 crop loss：大多数
local edits 仍完全或部分进入了模型，整图聚合和训练目标不匹配同样重要。

## 7. Primary whole-image T1 结果

每个 condition 都与同一个独立 real250 panel 比较。表中 fixed accuracy /
recall 使用 released `ai_score > 0.5`。

| Forged condition | AUROC | AP | TPR@5%FPR | Fixed accuracy | Forged recall | TP/FP/FN/TN |
|---|---:|---:|---:|---:|---:|---:|
| local mouse | 0.499992 | 0.501769 | 0.056 | 0.494 | 0.168 | 42/45/208/205 |
| local cat | 0.514472 | 0.528121 | 0.080 | 0.510 | 0.200 | 50/45/200/205 |
| local trash can | 0.523280 | 0.524103 | 0.060 | 0.512 | 0.204 | 51/45/199/205 |
| full-frame mouse | 0.902104 | 0.926853 | 0.744 | 0.820 | 0.820 | 205/45/45/205 |
| full-frame cat | 0.934176 | 0.951396 | 0.840 | 0.842 | 0.864 | 216/45/34/205 |
| full-frame trash can | 0.910016 | 0.932888 | 0.772 | 0.826 | 0.832 | 208/45/42/205 |

对应的 shared-cluster bootstrap 95% CI：

| Forged condition | AUROC 95% CI | AP 95% CI | TPR@5%FPR 95% CI |
|---|---|---|---|
| local mouse | [0.486508, 0.512871] | [0.484813, 0.523105] | [0.031743, 0.068598] |
| local cat | [0.499572, 0.532146] | [0.505334, 0.558555] | [0.046691, 0.110298] |
| local trash can | [0.510266, 0.538322] | [0.503014, 0.550418] | [0.033895, 0.082686] |
| full-frame mouse | [0.875187, 0.927374] | [0.907668, 0.945695] | [0.675838, 0.805058] |
| full-frame cat | [0.913353, 0.954934] | [0.935642, 0.966601] | [0.765672, 0.882822] |
| full-frame trash can | [0.884591, 0.935115] | [0.914921, 0.950965] | [0.694748, 0.828702] |

### 7.1 Family macro 与 95% CI

| Macro | AUROC [95% CI] | AP [95% CI] | TPR@5%FPR [95% CI] | Fixed accuracy | Recall |
|---|---|---|---|---:|---:|
| local | 0.512581 [0.501794, 0.523764] | 0.517997 [0.501767, 0.539699] | 0.065333 [0.041887, 0.081038] | 0.505333 | 0.190667 |
| full-frame | 0.915432 [0.892596, 0.937646] | 0.937046 [0.920031, 0.953471] | 0.785333 [0.718911, 0.834040] | 0.829333 | 0.838667 |
| all six, mixed-family | 0.714007 [0.700735, 0.728259] | 0.727522 [0.715443, 0.742446] | 0.425333 [0.385309, 0.450612] | 0.667333 | 0.514667 |

共享 real250 在 fixed threshold 下为 FP=45、TN=205，即 FPR=18%。这说明
released 0.5 threshold 在当前真实域并未校准到 5% FPR；TPR@5%FPR 是另一
个由 real scores 重新确定 operating point 的指标，二者不能混用。

### 7.2 Domain slice

六条件等权 macro 的 domain slice：

| Domain | AUROC [95% CI] | AP [95% CI] | TPR@5%FPR [95% CI] | Fixed accuracy [95% CI] |
|---|---|---|---|---|
| lodging | 0.688384 [0.664671, 0.712603] | 0.717201 [0.697938, 0.743064] | 0.380678 [0.313299, 0.425973] | 0.625407 [0.595635, 0.654640] |
| restaurant | 0.732898 [0.717729, 0.747083] | 0.734967 [0.722662, 0.752310] | 0.450667 [0.425579, 0.495370] | 0.715450 [0.693317, 0.734739] |

Domain slice 同样混合 local 与 full-frame，不应替代 family-level 结果。

## 8. Source-matched secondary

Delta 定义为方向标准化后的
`forged_ai_score - matched_real_ai_score`；正值表示 forged endpoint 被
排得更像 fake。Secondary 不是 primary 的替代品。

| Condition | Pairs | Mean delta [95% CI] | Strict ranking [95% CI] | W/L/T |
|---|---:|---|---|---:|
| local mouse | 250 | -0.003715 [-0.006977, -0.000945] | 0.412 [0.349991, 0.469654] | 103/129/18 |
| local cat | 250 | +0.023228 [0.004478, 0.043559] | 0.616 [0.556764, 0.675125] | 154/92/4 |
| local trash can | 250 | +0.021518 [0.008261, 0.036207] | 0.688 [0.627447, 0.746988] | 172/74/4 |
| full-frame mouse | 250 | +0.639461 [0.578901, 0.702302] | 0.904 [0.863432, 0.943201] | 226/24/0 |
| full-frame cat | 250 | +0.704524 [0.642928, 0.757280] | 0.936 [0.903462, 0.966169] | 234/16/0 |
| full-frame trash can | 250 | +0.656784 [0.597298, 0.710272] | 0.920 [0.880642, 0.951966] | 230/20/0 |

真正 pooled family 结果：

| Family | Pairs | Mean delta [95% CI] | Median delta | Strict ranking [95% CI] | W/L/T |
|---|---:|---|---:|---|---:|
| local | 750 | +0.013677 [0.005611, 0.022349] | ~0 | 0.572 [0.532223, 0.608704] | 429/295/26 |
| full-frame | 750 | +0.666923 [0.609605, 0.721078] | 0.979213 | 0.920 [0.887529, 0.949229] | 690/60/0 |
| all pairs | 1,500 | +0.340300 [0.310317, 0.368158] | 0.009306 | 0.746 [0.719408, 0.769787] | 1119/355/26 |

Local pooled mean delta 的 CI 虽然高于 0，但效应很小，且 Mouse 单项方向为
负；不能把统计上可分辨的微小 paired shift 表述为可靠 local detector。

## 9. Determinism 与独立 replay

### 9.1 双 CUDA smoke

两个独立 run：

- `spai_any_resolution_spectral_balanced250_v1_smoke5x7_a_20260726`
- `spai_any_resolution_spectral_balanced250_v1_smoke5x7_b_20260726`

每个 run 都包含七条件各 5 张，共 35 张。Comparison 结果：

- status：`deterministic_spai_smoke_comparison_passed`；
- images compared：35；
- exact computational projection：true；
- independent artifact paths：true；
- patch-feature、feature、attention 文件字节和数组：exact；
- raw logit、probability、patch features、feature、attention 最大差异：0；
- 两边 persisted replay：各 35/35。

### 9.2 Formal persisted-artifact replay

独立 analyzer 从 5,325 个保存产物重新执行：

1. `patch_features -> SCA -> LayerNorm -> complete MLP`；
2. `feature -> complete MLP`。

两条路径均重放 1,775/1,775。最大差异：

| Comparison | Max absolute difference |
|---|---:|
| SCA feature | 0 |
| SCA attention | 0 |
| SCA raw logit | 0 |
| feature-MLP raw logit | 0 |
| SCA probability | 0 |
| feature-MLP probability | 0 |

### 9.3 Fresh full-model replay

Analyzer 重新读取 1,775 张 canonical image，重新执行完整
FFT/ViT/SRS/SCA/LayerNorm/MLP，而不是复用 runner 中间特征：

| Comparison | Images | Max difference | Tolerance |
|---|---:|---:|---:|
| patch features | 1,775 | 0 | 1e-5 |
| feature | 1,775 | 0 | 1e-5 |
| attention | 1,775 | 0 | 1e-6 |
| raw logit | 1,775 | 0 | 1e-5 |
| probability | 1,775 | 0 | 1e-7 |

审计完成后又重新验证 manifest、results、expected inputs、summary、
artifact inventory 和 canonical dataset evidence 在审计期间没有变化，
之后才发布 metrics 和 independent audit。

## 10. 商用与许可边界

SPAI repository 的 LICENSE 和 README 把官方 code/weights 标为
Apache-2.0；但这不足以把整个权利链直接判为商用安全：

- SPAI 明确基于 MFM code/backbone，而其 upstream S-Lab License 1.0
  在没有 contributor permission 时限制为非商用；
- linked DMimageDetection training archive 标为 nonprofit-only；
- COCO/LSUN 图像权利不构成 blanket commercial grant；
- 官方没有提供能消除这些依赖冲突的完整 checkpoint model card 或训练
  provenance manifest。

因此本 benchmark 的结论是：

```text
commercial_clearance = unresolved
risk = high
```

研究复现通过不等于商业部署许可。计划商用者需要对 MFM/DM archive、
训练数据和 checkpoint 派生关系做独立法律与供应链核查。

## 11. 测试与复现入口

冻结代码在正式运行前通过：

- final focused audit：112 passed，1 deselected；
- legacy + Balanced250 CPU-only suite：185 passed，1 deselected；
- real CPU preflight test：1 passed；
- 专用 bytecode cache 在测试和正式运行后保持为空。

以下复现命令必须从 repository root 运行，并为每次完整重跑选择新的、
不可变的 run ID。命令假定目标 benchmark GPU 已经 drain；在当前机器上
必须先通过 narrow wrapper 暂停 Hunyuan keepalive，并在命令退出后的
`finally` 中立即恢复，不能在 keepalive 正生成时直接运行裸命令。

正式复现入口：

```bash
export SPAI_REPRO_RUN_ID="spai_any_resolution_spectral_balanced250_v1_full1775_repro_$(date -u +%Y%m%d_%H%M%S)"

CUDA_VISIBLE_DEVICES=4 \
PYTHONHASHSEED=0 \
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPYCACHEPREFIX=/root/.cache/claimforge/pycache/spai-balanced-v2-empty \
CUBLAS_WORKSPACE_CONFIG=:4096:8 \
NO_ALBUMENTATIONS_UPDATE=1 \
/root/.cache/claimforge/venvs/spai/bin/python \
  -m eval.opensource.run_spai_balanced \
  --repo-root . \
  --mode formal \
  --run-id "$SPAI_REPRO_RUN_ID" \
  --device cuda:0 \
  --fail-fast
```

独立审计：

```bash
CUDA_VISIBLE_DEVICES=4 \
PYTHONHASHSEED=0 \
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPYCACHEPREFIX=/root/.cache/claimforge/pycache/spai-balanced-v2-empty \
CUBLAS_WORKSPACE_CONFIG=:4096:8 \
NO_ALBUMENTATIONS_UPDATE=1 \
/root/.cache/claimforge/venvs/spai/bin/python \
  -m eval.opensource.analyze_spai_balanced \
  --repo-root . \
  --run-id "$SPAI_REPRO_RUN_ID" \
  --device cuda:0
```

Strict resume 只适用于尚未完成、且 analyzer outputs
`balanced250_metrics.json`/`independent_audit.json` 尚不存在的 run。已经
finalized 的 run 一律使用新 run ID 重跑，不能原地覆盖正式证据。若进程被
硬杀并留下 partial JSONL/orphan artifact，应先人工归档损坏 run，再使用
新的 run ID，不能把不完整证据自动“修复”成正式结果。
