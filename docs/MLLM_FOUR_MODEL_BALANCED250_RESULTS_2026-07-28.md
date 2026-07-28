# MLLM 四模型 Balanced250 结果（严格 Localization）

**日期：** 2026-07-28

## 1. 统计口径

本文件只使用定稿的 `local 750 + real 250`：

- Local Mouse / Cat / Trash can 各 250；
- Real 250；
- 与 Commercial API Balanced250 的 `panel=true` 选择完全一致；
- Detection 在全部 1,000 张上计算；
- Localization 在全部 750 张 forged 上计算；
- Localization 缺失、无效、空 bbox 或没有任何图内有效 bbox，一律记
  为 miss。

具体规范见：

```text
docs/MLLM_BALANCED250_LOCAL1000_STANDARD_2026-07-28.md
```

## 2. Detection（750 forged + 250 real）

| 模型 | Accuracy | Precision | Recall/TPR | Specificity | F1 | Balanced Acc. | AUROC | AP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen 3.7 Plus | 29.10% | 87.27% | 6.40% | 97.20% | 11.93% | 51.80% | 52.34% | 75.97% |
| GPT-5.6 Luna | 32.90% | 92.47% | 11.47% | 97.20% | 20.40% | 54.33% | 65.17% | 83.38% |
| Claude Opus 4.8 | 54.00% | 95.31% | 40.67% | 94.00% | 57.01% | 67.33% | 70.12% | 87.64% |
| Doubao Seed 2.1 Pro | 39.00% | 88.89% | 21.33% | 92.00% | 34.41% | 56.67% | 61.14% | 81.09% |

各 forged 条件的 `edited` 检出率：

| 模型 | Mouse | Cat | Trash can | Local 总体 |
|---|---:|---:|---:|---:|
| Qwen 3.7 Plus | 11.20% | 6.00% | 2.00% | 6.40% |
| GPT-5.6 Luna | 16.80% | 16.00% | 1.60% | 11.47% |
| Claude Opus 4.8 | 47.20% | 69.20% | 5.60% | 40.67% |
| Doubao Seed 2.1 Pro | 30.00% | 24.40% | 9.60% | 21.33% |

Real 250 的误报率分别为 Qwen 2.80%、GPT 2.80%、Claude 6.00%、
Doubao 8.00%。

## 3. Localization 严格覆盖

`result_valid` 只表示存在三次有效聚合；指标分母始终是 750。`严格
miss` 表示没有任何可计分的图内 bbox，包含模型判断
`no_localized_edit` 的情况。

| 模型 | Result valid | 有有效 bbox | 严格 miss | Overlap | IoU@0.1 | IoU@0.25 | IoU@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen 3.7 Plus | 749/750 | 43/750 | 707/750 | 3.60% | 3.33% | 3.07% | 1.20% |
| GPT-5.6 Luna | 749/750 | 70/750 | 680/750 | 6.13% | 5.87% | 5.47% | 2.67% |
| Claude Opus 4.8 | 750/750 | 328/750 | 422/750 | 40.27% | 39.20% | 34.53% | 16.00% |
| Doubao Seed 2.1 Pro | 750/750 | 67/750 | 683/750 | 7.60% | 7.47% | 7.07% | 2.80% |

两条历史缺失结果不再被排除：

- GPT：1 个 Mouse localization 无有效记录；
- Qwen：1 个 Cat localization 无有效记录。

它们都作为严格 miss 进入 750 分母。

## 4. Localization bbox-to-mask

预测 bbox 并集栅格化为二值 mask；GT 是 decoded source 与局部拼接
PNG 的逐像素非零 RGB 差分。

| 模型 | Macro pixel IoU | Macro pixel F1 | Micro pixel IoU |
|---|---:|---:|---:|
| Qwen 3.7 Plus | 1.85% | 2.34% | 0.19% |
| GPT-5.6 Luna | 3.37% | 4.17% | 0.28% |
| Claude Opus 4.8 | 15.74% | 21.33% | 2.93% |
| Doubao Seed 2.1 Pro | 4.29% | 5.19% | 0.51% |

按候选类别的严格 Macro pixel IoU：

| 模型 | Mouse | Cat | Trash can |
|---|---:|---:|---:|
| Qwen 3.7 Plus | 4.63% | 0.83% | 0.10% |
| GPT-5.6 Luna | 8.46% | 1.54% | 0.12% |
| Claude Opus 4.8 | 15.04% | 30.88% | 1.32% |
| Doubao Seed 2.1 Pro | 8.95% | 3.54% | 0.37% |

## 5. 结果解释

1. Claude 在 Detection 和 Localization 上均显著领先，尤其是 Cat；
   Trash can 仍是所有模型最难的局部编辑。
2. Doubao 的坐标修复后，严格 Overlap 和 IoU 阈值指标整体高于 GPT 与
   Qwen，但有效 bbox 只有 67/750。
3. 严格口径会同时惩罚“没有发现编辑”和“发现编辑但没有给出合法
   bbox”。这正是端到端 Localization 能力，而不是只在成功返回 bbox 的
   子集上衡量坐标精度。
4. Macro pixel IoU 远高于 Micro pixel IoU，是因为局部对象面积很小，
   少量大框的像素误报会显著拉低全数据集 micro 指标。

## 6. 机器可读结果

```text
results/mllm/balanced250_local1000_v2/summary.json
results/mllm/balanced250_local1000_v2/summary.csv
results/mllm/balanced250_local1000_v2/<model>/
```

每个模型目录均包含 detection/localization 的逐图 JSONL 和 JSON/CSV
汇总。逐图 Localization 文件显式记录
`strict_localization_miss`、`result_valid_for_metrics`、
`invalid_or_out_of_bounds_box_count` 以及所有 bbox/pixel 指标。
