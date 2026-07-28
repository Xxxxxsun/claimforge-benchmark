# MLLM Balanced250 Local1000 定稿口径（2026-07-28）

## 1. 权威统计集合

MLLM 主结果与 Commercial API 使用同一份 Balanced250 独立抽样面板。
只选择其 `panel=true` 的四个条件：

| 条件 | 数量 | 标签 | 用途 |
|---|---:|---|---|
| `local_mouse` | 250 | forged | Detection + Localization |
| `local_cat` | 250 | forged | Detection + Localization |
| `local_trash_can` | 250 | forged | Detection + Localization |
| `real` | 250 | real | Detection 负类 |
| 合计 | 1,000 | 750 forged + 250 real | 主统计 |

正式 dataset ID：

```text
claimforge-mllm-balanced250-local750-real250-v2
```

正式清单：

```text
annotations/claimforge_mllm_benchmark1000_v2.jsonl
annotations/claimforge_mllm_benchmark1000_v2.manifest.json
```

选择来源：

```text
results/opensource/community_forensics/
community_forensics_highres_vit_s16_384_balanced250_v1_full1775_20260726/
expected_inputs.jsonl
```

选择规则是该文件中 `panel=true` 且 `condition` 为 `real`、
`local_mouse`、`local_cat`、`local_trash_can` 的行。MLLM 不再使用旧的
“Mouse 跟随 Trash-can、Cat 随机排除一张、Real 优先 Trash 来源”
口径。

清单按 `image_sha256` 与历史推理结果连接。因此历史 run 可以继续保留
多跑的 Mouse、Cat 或 Real，正式统计时只过滤出这 1,000 张。

## 2. Detection 统计

Detection 在 1,000 张上计算：

- forged 750 为正类；
- real 250 为负类；
- 离散指标使用三次有效调用多数票后的 `decision`；
- AUROC/AP 使用聚合后的 `p_ai_edited / 100`；
- 输出 Accuracy、Precision、Recall/TPR、Specificity/TNR、FPR、F1、
  Balanced Accuracy、AUROC 和 AP。

## 3. Localization 严格口径

Localization 主分母固定为全部 750 张 forged 图片。以下情况均直接记为
localization miss，不再从分母中排除：

1. 没有结果行；
2. API/解析/schema 失败，`valid_for_metrics != true`；
3. 三次聚合结果没有 bbox；
4. 所有 bbox 均为空、顺序非法或超出原图坐标范围。

因此：

```text
forged_scored = forged_expected = 750
```

`result_valid` 与 `result_coverage` 只用于审计，不改变指标分母。有效返回
`no_localized_edit` 但没有 bbox，在 forged 样本上同样是 miss。

Localization 报告：

- 任一预测框与 `edit_region_xyxy` 有正面积交集（Overlap）；
- 最佳预测框的 IoU@0.1、IoU@0.25、IoU@0.5；
- 所有预测框均位于 GT 框内；
- bbox 并集栅格化后，相对 source/spliced PNG exact RGB diff mask 的
  macro/micro pixel Precision、Recall、F1、IoU、MCC。

Real 250 不进入 forged localization 命中率分母。Real 的负类性能由
Detection 统计；Real localization 只可作为额外诊断，不能替代
Detection specificity。

## 4. 重建与聚合

重建固定输入清单：

```bash
conda run -n utils python scripts/build_mllm_benchmark1000.py
```

聚合四个模型现有结果：

```bash
conda run -n utils python scripts/aggregate_mllm_balanced250.py
```

聚合输出：

```text
results/mllm/balanced250_local1000_v2/
├── summary.json
├── summary.csv
├── <model>/detection_per_image.jsonl
├── <model>/detection_metrics.json
├── <model>/detection_metrics.csv
├── <model>/localization_per_image.jsonl
├── <model>/localization_metrics.json
└── <model>/localization_metrics.csv
```

通用 `eval/mllm/metrics.py` 也采用相同的严格 forged denominator 逻辑，
因此后续单 run 统计和四模型汇总不会再出现缺失 bbox 被排除的差异。

## 5. 推理入口

普通 MLLM：

```bash
conda run -n utils python -m eval.mllm.run_mllm \
  --config config/mllm_eval.example.json \
  --source benchmark1000 \
  --protocol both \
  --condition balanced250_local1000_v2 \
  --run-id <unique-run-id>
```

Zoom agent：

```bash
conda run -n utils python -m eval.mllm.run_zoom_agent \
  --config config/mllm_eval.example.json \
  --source benchmark1000 \
  --condition balanced250_local1000_zoom5_v2 \
  --max-zoom-calls 5 \
  --run-id <unique-run-id>
```

`--source benchmark1000` 会在发起请求前校验 dataset ID、manifest/ledger
哈希绑定、1,000 个唯一 ID、750/250 标签分布以及四个条件各 250 张。

## 6. 图片物化策略

统计清单直接引用仓库内现有 `raw_path`，不需要复制一套定稿图片。
`benchmark/claimforge_v1_250x3x2/local_splice` 与
`benchmark/claimforge_v1_250x3x2/full_image` 的 1,500 张物化副本已删除；
顶层 manifest、pairs、summary 和 README 保留作历史选择审计。
