# MLLM 四模型结果汇总（Doubao bbox_1000 修正版）

**日期：** 2026-07-27

## 1. 评测范围

当前主表包含四个已有正式结果的模型：

- Qwen 3.7 Plus
- GPT-5.6 Luna
- Claude Opus 4.8
- Doubao Seed 2.1 Pro

数据分为三组：

| 数据组 | Mouse | Cat | Trash can | 总数 | 用途 |
|---|---:|---:|---:|---:|---|
| 局部拼接（local splice） | 275 | 251 | 250 | 776 | T1 + T2 主评测 |
| 全图生成（full-image orange-box） | 275 | 272 | 260 | 807 | T1 生成方式消融 |
| 去重 real 原图 | — | — | — | 270 | T1 共同负样本 |

三个旧模型的 mouse paired 集虽然包含 275 个 real 行，但按
`image_sha256` 去重后是 270 张；这 270 个哈希与 Doubao 的
`real_source_union270` 完全一致。因此 T1 汇总使用同一组 270 张
real 原图。旧结果中同一哈希出现多次时，按冻结输入顺序保留第一条
有效聚合记录；五组重复图在三个旧模型中均没有离散 decision 分歧。

T1 统一使用 detection v3。当前 T2 汇总使用各模型已经完成的最终结果：

- Qwen、GPT、Claude：mouse 为历史 localization v3，cat/trash-can 为
  localization v4 像素坐标协议；
- Doubao：localization v5 `bbox_1000` 协议，解析后立即转换为原图像素，
  再进行坐标校验、三次聚合和 bbox-to-mask。

因此本文件是“当前所有可用结果”的汇总。严格的同协议模型比较仍应在
Qwen、GPT、Claude 的 mouse localization 也统一重跑后进行。

Gemini 3.1 Pro Preview 仅完成能力与网关稳定性 pilot，没有完整主实验，
不进入本表。GLM-5.2 不具备本次调用所需的稳定图像理解能力，已由
Doubao 替代。

各模型在局部拼接主评测中的有效覆盖如下。Claude 的 cat/trash-can
和 mouse 超限图片均已通过 Anthropic-native 补测；mouse detection
已完整，localization 仅剩 `lodging_233_slot_001__forged` 的一个历史
schema 失败单元，与图片大小无关。

| 模型 | T1 Detection 有效数 | T2 Localization 有效数 |
|---|---:|---:|
| Qwen 3.7 Plus | 776/776 | 775/776 |
| GPT-5.6 Luna | 776/776 | 775/776 |
| Claude Opus 4.8 | 776/776 | 775/776 |
| Doubao Seed 2.1 Pro | 776/776 | 776/776 |

GPT 的两条 Cat detection 无效单元和一条 Cat localization 无效单元已于
2026-07-28 补齐并替换。失败原因是模型返回空正文/无效 JSON，并非图片
大小；补跑保持模型、prompt 和协议不变，仅对持续空响应的 detection
调用将输出上限从 600 提高到 2,000 tokens。GPT 当前唯一未补齐项是
历史 Mouse localization 的 1 个单元。

## 2. Doubao bbox_1000 修复口径

Doubao 的旧输出经常把 `[0, 1000]` 归一化坐标写入 `bbox_px` 字段。
新口径单独要求 Doubao 输出 `bbox_1000`，并在解析后立即映射到原图
像素坐标；后续校验、IoU 聚类、多数投票和 mask 栅格化全部在像素空间
执行。

本次离线修复覆盖 2,328 次 replicate：

| 修复类型 | Replicate 数 |
|---|---:|
| 将旧 `bbox_px` 解释为 `bbox_1000` | 2,232 |
| 确实遵循旧像素协议，保留为像素坐标 | 3 |
| 旧越界尝试后最终改为阴性，保留最终阴性 | 93 |
| 合计 | 2,328 |

后 93 次涉及 57 张图。按照当前评测口径，坐标格式理解、schema
遵循和重试稳定性都属于模型能力，因此这些最终
`no_localized_edit` 结果照常进入三次聚合，而不是重新请求模型。

最终 776/776 张图片均有三次记录，所有聚合坐标均位于原图范围内。

## 3. T1：局部拼接图片检出率

所有局部拼接输入均为 forged，因此下表是 `edited` 检出率（TPR），
不是完整分类准确率。

| 模型 | Mouse | Cat | Trash can | 总体 |
|---|---:|---:|---:|---:|
| Qwen 3.7 Plus | 30/275 (10.91%) | 15/251 (5.98%) | 5/250 (2.00%) | 50/776 (6.44%) |
| GPT-5.6 Luna | 46/275 (16.73%) | 40/251 (15.94%) | 4/250 (1.60%) | 90/776 (11.60%) |
| Claude Opus 4.8 | 129/275 (46.91%) | 173/251 (68.92%) | 14/250 (5.60%) | 316/776 (40.72%) |
| Doubao Seed 2.1 Pro | 84/275 (30.55%) | 61/251 (24.30%) | 24/250 (9.60%) | 169/776 (21.78%) |

## 4. T1：共同 real270 与完整分类指标

### 4.1 Real 原图误报

| 模型 | `edited` 误报 | FPR |
|---|---:|---:|
| Qwen 3.7 Plus | 7/270 | 2.59% |
| GPT-5.6 Luna | 7/270 | 2.59% |
| Claude Opus 4.8 | 17/270 | 6.30% |
| Doubao Seed 2.1 Pro | 20/270 | 7.41% |

### 4.2 局部 forged + 共同 real270

离散指标直接使用三次多数投票后的 `decision`；AUROC 和 AP 使用三次
`p_ai_edited` 中位数除以 100。由于该集合 forged 占约 74%，AP 的随机
基准不再是 50%，因此同时报告 AUROC 和 balanced accuracy。

| 模型 | 有效 forged / real | TP / TN / FP / FN | Accuracy | Precision | Recall | F1 | Balanced Acc. | AUROC | AP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen 3.7 Plus | 776 / 270 | 50 / 263 / 7 / 726 | 29.92% | 87.72% | 6.44% | 12.00% | 51.93% | 52.46% | 75.24% |
| GPT-5.6 Luna | 776 / 270 | 90 / 263 / 7 / 686 | 33.75% | 92.78% | 11.60% | 20.62% | 54.50% | 64.16% | 82.48% |
| Claude Opus 4.8 | 776 / 270 | 316 / 253 / 17 / 460 | 54.40% | 94.89% | 40.72% | 56.99% | 67.21% | 69.76% | 87.09% |
| Doubao Seed 2.1 Pro | 776 / 270 | 169 / 250 / 20 / 607 | 40.06% | 89.42% | 21.78% | 35.03% | 57.19% | 61.47% | 80.70% |

## 5. T1：全图生成消融

全图生成的完整画面都经过生成模型，所有 807 张均为正样本。下表是
`edited` 检出率。

| 模型 | Mouse | Cat | Trash can | 总体 |
|---|---:|---:|---:|---:|
| Qwen 3.7 Plus | 137/275 (49.82%) | 37/272 (13.60%) | 11/260 (4.23%) | 185/807 (22.92%) |
| GPT-5.6 Luna | 175/275 (63.64%) | 75/272 (27.57%) | 17/260 (6.54%) | 267/807 (33.09%) |
| Claude Opus 4.8 | 253/275 (92.00%) | 216/272 (79.41%) | 79/260 (30.38%) | 548/807 (67.91%) |
| Doubao Seed 2.1 Pro | 206/275 (74.91%) | 120/272 (44.12%) | 109/260 (41.92%) | 435/807 (53.90%) |

从局部拼接切换到全图生成后，总检出率分别提高：

| 模型 | 局部拼接 | 全图生成 | 差值 |
|---|---:|---:|---:|
| Qwen 3.7 Plus | 6.44% | 22.92% | +16.48 pp |
| GPT-5.6 Luna | 11.60% | 33.09% | +21.49 pp |
| Claude Opus 4.8 | 40.72% | 67.91% | +27.19 pp |
| Doubao Seed 2.1 Pro | 21.78% | 53.90% | +32.12 pp |

这说明四个模型都更容易发现整帧重生成留下的全局信号，而不是仅占
图像小区域的局部拼接。

## 6. T2：模型是否返回定位结果

下表仅统计聚合 decision 是否为 `localized_edit`，不代表坐标命中
GT。

| 模型 | Mouse | Cat | Trash can | 总体 |
|---|---:|---:|---:|---:|
| Qwen 3.7 Plus | 28/275 (10.18%) | 16/250 (6.40%) | 2/250 (0.80%) | 46/775 (5.94%) |
| GPT-5.6 Luna | 44/274 (16.06%) | 31/251 (12.35%) | 4/250 (1.60%) | 79/775 (10.19%) |
| Claude Opus 4.8 | 137/274 (50.00%) | 187/251 (74.50%) | 16/250 (6.40%) | 340/775 (43.87%) |
| Doubao Seed 2.1 Pro | 40/275 (14.55%) | 25/251 (9.96%) | 8/250 (3.20%) | 73/776 (9.41%) |

## 7. T2：bbox-to-mask 主指标

聚合 bbox 的并集被栅格化为二值预测 mask。GT 是 decoded real source
与最终局部拼接 PNG 的逐像素 RGB 非零差分。该 mask 是 MLLM
bbox-to-mask adapter，不是模型原生像素分割。

### 7.1 Macro pixel IoU（主指标）

| 模型 | Mouse | Cat | Trash can | 总体 |
|---|---:|---:|---:|---:|
| Qwen 3.7 Plus | 4.74% | 0.83% | 0.10% | **1.98%** |
| GPT-5.6 Luna | 8.52% | 1.53% | 0.12% | **3.55%** |
| Claude Opus 4.8 | 14.71% | 30.75% | 1.32% | **15.59%** |
| Doubao Seed 2.1 Pro | 8.92% | 3.52% | 0.37% | **4.42%** |

### 7.2 Macro pixel F1

| 模型 | Mouse | Cat | Trash can | 总体 |
|---|---:|---:|---:|---:|
| Qwen 3.7 Plus | 5.85% | 1.11% | 0.15% | 2.48% |
| GPT-5.6 Luna | 10.25% | 2.09% | 0.18% | 4.36% |
| Claude Opus 4.8 | 21.07% | 40.74% | 1.54% | 21.14% |
| Doubao Seed 2.1 Pro | 10.61% | 4.45% | 0.43% | 5.34% |

## 8. T2：辅助坐标指标

辅助指标使用矩形 `edit_region_xyxy`：

- Overlap：任一聚合框与 GT 框有正面积交集；
- IoU@0.1/0.25/0.5：最佳预测框与 GT 框的 IoU 达到相应阈值。

| 模型 | 有效 forged | Overlap | IoU@0.1 | IoU@0.25 | IoU@0.5 |
|---|---:|---:|---:|---:|---:|
| Qwen 3.7 Plus | 775 | 29 (3.74%) | 27 (3.48%) | 25 (3.23%) | 9 (1.16%) |
| GPT-5.6 Luna | 775 | 49 (6.32%) | 47 (6.06%) | 44 (5.68%) | 22 (2.84%) |
| Claude Opus 4.8 | 775 | 295 (38.06%) | 287 (37.03%) | 251 (32.39%) | 114 (14.71%) |
| Doubao Seed 2.1 Pro | 776 | 59 (7.60%) | 57 (7.35%) | 54 (6.96%) | 21 (2.71%) |

修正后 Doubao 的 Overlap、IoU@0.1 和 IoU@0.25 均高于 GPT 和 Qwen；
IoU@0.5 与 GPT 接近。旧结果中 Doubao 坐标被错误当作像素坐标时，
总体只有 6 个 overlap、1 个 IoU@0.1 命中，不能继续用于主表。

## 9. 结论

1. **Claude 当前总体最强。** 它在局部拼接 T1、全图生成 T1 和 T2
   macro pixel IoU 上均领先，尤其擅长 cat；但 trash-can 定位仍很弱。
2. **Doubao 修正后是当前第二强的总体 T2。** 总体 macro pixel IoU
   为 4.42%，高于 GPT 的 3.55% 和 Qwen 的 1.98%，但仍远低于 Claude
   的 15.59%。
3. **Doubao 的 T1 位于 Claude 与 GPT/Qwen 之间。** 局部拼接检出率
   21.78%，全图生成检出率 53.90%，共同 real270 上的 FPR 为 7.41%。
4. **Trash can 是所有模型的共同难点。** 即使全图生成时 T1 有所提升，
   局部拼接 T2 macro pixel IoU 仍全部低于 1.5%。
5. **全图生成远比局部拼接容易检出。** 四模型总体检出率提升
   16.48–32.12 个百分点，支持“整帧生成信号更明显、局部编辑更隐蔽”
   的消融结论。

## 10. 主要结果文件

### Qwen 3.7 Plus

- `results/mllm/qwen3_7_plus/qwen37plus_pilot_good275_c15_v3_20260715T153257_0800.jsonl`
- `results/mllm/qwen3_7_plus/final_cat251_trash250_total501_suite0724_20260725.jsonl`
- `results/mllm/qwen3_7_plus/fullai_orangebox_all807_detectionv3_20260725.jsonl`

### GPT-5.6 Luna

- `results/mllm/gpt/gpt56luna_pilot_good275_c15_v3_20260715T153257_0800.jsonl`
- `results/mllm/gpt/final_cat251_trash250_total501_suite0724_20260725.jsonl`
- `results/mllm/gpt/fullai_orangebox_all807_detectionv3_20260725.jsonl`
- `results/mllm/gpt/gpt_cat_detection_recovery_intent_tokens2000_20260728.jsonl`
- `results/mllm/gpt/gpt_cat_localization_recovery_combined_20260728.jsonl`

### Claude Opus 4.8

- `results/mllm/claude_opus_4_8/mouse_good275_total550_v3_20260727.jsonl`
- `results/mllm/claude_opus_4_8/mouse6_oversize_anthropic_native_v3_20260727.jsonl`
- `results/mllm/claude_opus_4_8/final_cat251_trash250_total501_suite0724_20260725.jsonl`
- `results/mllm/claude_opus_4_8/final501_oversize9_anthropic_native_suite0724_20260727.jsonl`
- `results/mllm/claude_opus_4_8/fullai_orangebox_all807_detectionv3_20260725.jsonl`

### Doubao Seed 2.1 Pro

- `results/mllm/doubao_seed_2_1_pro_260628/doubao_main_local776_suite0724_20260726.jsonl`
- `results/mllm/doubao_seed_2_1_pro_260628/doubao_main_local776_localization_bbox1000_v5_20260727.jsonl`
- `results/mllm/doubao_seed_2_1_pro_260628/doubao_real_source_union270_detectionv3_20260726.jsonl`
- `results/mllm/doubao_seed_2_1_pro_260628/doubao_fullai_orangebox_all807_detectionv3_20260726.jsonl`
