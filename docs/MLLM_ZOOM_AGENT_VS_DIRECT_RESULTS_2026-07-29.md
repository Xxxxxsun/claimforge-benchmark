# MLLM Zoom Agent 与直接 MLLM 对比结果

日期：2026-07-29

## 1. 结论

Zoom Agent 并未给所有模型带来一致提升：

- **Qwen 3.7 Plus 改善明显**。Detection accuracy 提高 15.1 个百分点，
  localization overlap 提高 19.2 个百分点。
- **GPT-5.6 Luna 没有获得有效提升**。Detection 和 localization 的固定阈值
  指标均略有下降，但 AUROC 提高，说明连续分数的排序能力有所改善，而最终
  决策变得更保守。
- **Claude Opus 4.8 整体下降**。Mouse 类别有所改善，但 Cat 类别出现严重
  退化，抵消了 Mouse 的收益。

因此，当前 Zoom Agent 更适合作为 **Qwen 3.7 Plus 的增强方案**，不宜在未做
模型专属调优的情况下替换所有模型的直接 MLLM 协议。

## 2. 对比口径

为了避免 Full1051 与定稿 benchmark 的样本差异干扰比较，本次将 Zoom Agent
结果按 image SHA-256 映射到当前正式的 Balanced250 MLLM 口径：

- 250 张 Mouse 局部编辑图；
- 250 张 Cat 局部编辑图；
- 250 张 Trash can 局部编辑图；
- 250 张 Real 图；
- Detection 总计 1,000 张；
- Localization 总计 750 张 forged 图。

三个 Zoom Agent 结果在该口径上均完整匹配：

| 模型 | Detection | Localization |
|---|---:|---:|
| GPT-5.6 Luna | 1,000 / 1,000 | 750 / 750 |
| Qwen 3.7 Plus | 1,000 / 1,000 | 750 / 750 |
| Claude Opus 4.8 | 1,000 / 1,000 | 750 / 750 |

Localization 使用与直接 MLLM 相同的严格规则：缺失结果、无效结果、空 bbox，
或所有 bbox 均越界，都作为 localization miss 纳入 750 张 forged 图的分母。
GT mask 使用 source image 与 forged image 的 exact-diff mask。

Doubao Seed 2.1 Pro 260628 尚无对应 Zoom Agent 全量结果，因此未纳入本次对比。

## 3. Detection

### 3.1 总体结果

| 模型 | Direct Accuracy | Zoom Accuracy | 变化 | Direct Balanced Acc. | Zoom Balanced Acc. | 变化 |
|---|---:|---:|---:|---:|---:|---:|
| GPT-5.6 Luna | 32.90% | 29.50% | -3.40 pp | 54.33% | 52.47% | -1.87 pp |
| Qwen 3.7 Plus | 29.10% | **44.20%** | **+15.10 pp** | 51.80% | **61.33%** | **+9.53 pp** |
| Claude Opus 4.8 | **54.00%** | 45.20% | -8.80 pp | **67.33%** | 62.67% | -4.67 pp |

### 3.2 Recall、specificity 与排序指标

| 模型 | Forged Recall：Direct → Zoom | Real Specificity：Direct → Zoom | AUROC：Direct → Zoom | AP：Direct → Zoom |
|---|---:|---:|---:|---:|
| GPT-5.6 Luna | 11.47% → 6.53% | 97.20% → 98.40% | 65.17% → 70.45% | 83.38% → 85.56% |
| Qwen 3.7 Plus | **6.40% → 27.07%** | 97.20% → 95.60% | **52.34% → 62.29%** | 75.97% → 81.00% |
| Claude Opus 4.8 | 40.67% → 27.73% | 94.00% → 97.60% | 70.12% → 71.86% | 87.64% → 86.87% |

GPT 和 Claude 的 AUROC 提高，但固定阈值下 forged recall 下降、real specificity
提高。这表明 Zoom Agent 使这两个模型更倾向于输出 `not_edited`。其连续分数
排序略有改善，但现有决策阈值或最终投票策略没有把排序收益转化为检测收益。

### 3.3 各类别 edited rate

Forged 类别中的 edited rate 等于该类别的检出率；Real 的 edited rate 是
false-positive rate。

| 模型 | Mouse：Direct → Zoom | Cat：Direct → Zoom | Trash can：Direct → Zoom | Real FPR：Direct → Zoom |
|---|---:|---:|---:|---:|
| GPT-5.6 Luna | 16.80% → 10.80% | 16.00% → 7.60% | 1.60% → 1.20% | 2.80% → 1.60% |
| Qwen 3.7 Plus | **11.20% → 35.60%** | **6.00% → 37.20%** | **2.00% → 8.40%** | 2.80% → 4.40% |
| Claude Opus 4.8 | **47.20% → 58.00%** | 69.20% → 20.80% | 5.60% → 4.40% | 6.00% → 2.40% |

Claude 的总体下降主要由 Cat 导致：Cat 检出率下降 48.4 个百分点。Claude
对 Mouse 则提高了 10.8 个百分点。

### 3.4 配对变化

| 模型 | Direct 与 Zoom 都正确 | 仅 Zoom 正确 | 仅 Direct 正确 | 两者都错误 |
|---|---:|---:|---:|---:|
| GPT-5.6 Luna | 277 | 18 | 52 | 653 |
| Qwen 3.7 Plus | 278 | **164** | 13 | 545 |
| Claude Opus 4.8 | 372 | 80 | **168** | 380 |

Qwen 的 164 个“仅 Zoom 正确”样本远多于 13 个“仅 Direct 正确”样本，说明
其改善并非由少量样本波动造成。GPT 和 Claude 的净变化方向相反。

## 4. Localization

### 4.1 总体结果

| 模型 | Overlap：Direct → Zoom | IoU@0.1：Direct → Zoom | IoU@0.25：Direct → Zoom | IoU@0.5：Direct → Zoom |
|---|---:|---:|---:|---:|
| GPT-5.6 Luna | 6.13% → 5.73% | 5.87% → 5.20% | 5.47% → 4.13% | 2.67% → 1.33% |
| Qwen 3.7 Plus | **3.60% → 22.80%** | **3.33% → 21.07%** | **3.07% → 14.53%** | **1.20% → 3.20%** |
| Claude Opus 4.8 | 40.27% → 24.80% | 39.20% → 23.33% | 34.53% → 16.00% | 16.00% → 5.20% |

### 4.2 Pixel-level 指标

| 模型 | Macro pixel F1：Direct → Zoom | Macro pixel IoU：Direct → Zoom |
|---|---:|---:|
| GPT-5.6 Luna | 4.17% → 2.30% | 3.37% → 1.55% |
| Qwen 3.7 Plus | **2.34% → 9.05%** | **1.85% → 6.01%** |
| Claude Opus 4.8 | 21.33% → 9.26% | 15.74% → 6.00% |

### 4.3 各类别 overlap

| 模型 | Mouse：Direct → Zoom | Cat：Direct → Zoom | Trash can：Direct → Zoom |
|---|---:|---:|---:|
| GPT-5.6 Luna | 13.20% → 9.60% | 4.80% → 7.20% | 0.40% → 0.40% |
| Qwen 3.7 Plus | **8.00% → 30.40%** | **2.80% → 34.40%** | **0.00% → 3.60%** |
| Claude Opus 4.8 | **44.40% → 54.80%** | 73.20% → 18.80% | 3.20% → 0.80% |

Zoom Agent 对 Qwen 的 Mouse、Cat 和 Trash can 都有提升，其中 Cat 提升最大。
Claude 的 Mouse overlap 提高 10.4 个百分点，但 Cat overlap 下降 54.4 个
百分点，是 Claude 总体 localization 退化的主要来源。

### 4.4 配对变化

以“至少一个预测 bbox 与 GT edit region 重合”为成功：

| 模型 | 两者都成功 | 仅 Zoom 成功 | 仅 Direct 成功 | 两者都失败 |
|---|---:|---:|---:|---:|
| GPT-5.6 Luna | 21 | 22 | 25 | 682 |
| Qwen 3.7 Plus | 24 | **147** | 3 | 576 |
| Claude Opus 4.8 | 137 | 49 | **165** | 399 |

GPT 的 22 对 25 接近持平。Qwen 有 147 张图通过 Zoom 从失败转为成功，而仅
3 张图从成功转为失败。Claude 则有 165 张图从成功变为失败。

## 5. 解释与后续建议

1. **Qwen 的主要瓶颈确实包含视觉搜索与局部观察能力。**
   允许模型自主放大区域后，Qwen 的检测和定位均显著改善，说明 Zoom Agent
   能有效补充其直接全图判断能力。

2. **GPT 的问题不是单纯分辨率不足。**
   Zoom 后 AUROC 提高，但固定阈值 recall 和 localization 没有改善。可进一步
   尝试重新校准 detection threshold，或将 Direct 与 Zoom 的连续分数融合，
   而不是直接采用 Zoom 的多数决策。

3. **Claude 需要按候选类别分析 Agent 行为。**
   Claude 对 Mouse 改善，但对 Cat 严重退化。应检查 Cat episode 的 zoom
   区域选择、工具调用后的上下文保留，以及 Agent prompt 是否让模型把猫本身
   视为正常场景内容，从而忽略合成边缘。

4. **Trash can 对所有 MLLM 仍然困难。**
   即使加入 Zoom，Trash can 的 detection 与 localization 仍显著低于 Mouse
   和 Cat。反光、规则边缘和场景中原本存在的容器类物体可能削弱了视觉异常
   信号。

5. **暂不采用统一替换策略。**
   当前建议保留 Direct MLLM 作为统一基线，并将 Qwen Zoom Agent 作为增强
   实验；GPT 可尝试分数融合，Claude 则应先解决 Cat 退化后再评估。

## 6. 结果来源

直接 MLLM 的正式聚合结果：

- `results/mllm/balanced250_local1000_v2/`

Zoom Agent 的 Full1051 结果：

- `results/mllm/gpt/agent_zoom/gpt56luna_zoom_full1051_c15_z5_bboxpx_v2_20260728.jsonl`
- `results/mllm/qwen3_7_plus/agent_zoom/qwen37plus_zoom_full1051_c15_z5_bboxpx_v2_20260728.jsonl`
- `results/mllm/claude_opus_4_8/agent_zoom/claudeopus48_zoom_full1051_c15_z5_bboxpx_v2_20260728.jsonl`

Claude 超限图片补跑说明：

- `docs/MLLM_CLAUDE_ZOOM_OVERSIZE_RECOVERY_2026-07-28.md`

正式样本口径：

- `annotations/claimforge_mllm_benchmark1000_v2.jsonl`
- `docs/MLLM_BALANCED250_LOCAL1000_STANDARD_2026-07-28.md`
