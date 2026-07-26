# CLAIMFORGE MLLM v3：good-275 结果与指标说明

**日期：** 2026-07-15
**协议版本：** `mllm_protocol_v3_reasoning_image_coordinates`
**评测子集：** `claimforge_generation_review_labels.json` 中 `status=good` 的 275 个任务；每个任务包含 1 张 forged 图和 1 张 real/source 图，共 **550 张**（275 edited、275 not_edited）。

## 1. 运行与有效性规则

每张图片在 **detection** 和 **localization** 两个协议下各独立调用 3 次。只有一个「图片 × 协议」的 3 次响应均通过 JSON schema 校验，才标记为 `valid_for_metrics=true` 并纳入主指标。`coverage = valid_images / 550` 单列报告；无效/缺失样本不作为预测错误计入分母。

聚合方式如下：

- **Detection：** `decision`（`edited` / `not_edited`）按 3 次投票多数决定；`p_ai_edited` 取 3 次的中位数。
- **Localization：** 当至少 2 次判为 `localized_edit` 时，聚合 decision 为正类。来自不同 replicate 的候选框以 IoU ≥ 0.10 聚类；只有获得至少 2 个 replicate 支持的簇才保留，框坐标取逐坐标中位数。
- v3 在 localization 请求中动态附加该图片的真实宽高，要求 `bbox_1000` 是相对完整图像、范围 `[0,1000]`、且 `x1<x2, y1<y2` 的归一化坐标。模型没有获得 GT 标签或 GT 框。

## 2. Detection 指标如何计算

GT 标签来自 review export：forged 图为 `edited`，source 图为 `not_edited`。准确率、混淆矩阵以及阈值型指标**直接比较聚合后的离散 `decision` 与 GT 标签**；不会将 `p_ai_edited / 100` 以 0.5 阈值重新转为类别。

- TP：GT=`edited` 且 decision=`edited`；TN：GT=`not_edited` 且 decision=`not_edited`。
- FP：GT=`not_edited` 却判为 `edited`；FN：GT=`edited` 却判为 `not_edited`。
- Accuracy=`(TP+TN)/(TP+TN+FP+FN)`；Precision=`TP/(TP+FP)`；Recall/TPR=`TP/(TP+FN)`；Specificity/TNR=`TN/(TN+FP)`；F1=`2TP/(2TP+FP+FN)`；Balanced Accuracy=`(TPR+TNR)/2`。
- **AUROC 与 AP**是例外：二者使用连续 `score = p_ai_edited / 100` 排序计算，而非离散 decision。

## 3. Detection 结果

百分比均在该模型的有效图片上计算；Coverage 单独体现调用/解析完成度。

| 模型 | 有效 / 550（Coverage） | TP / TN / FP / FN | Accuracy | Precision | Recall | Specificity | F1 | Balanced Acc. | AUROC | AP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen 3.7 Plus | 550 / 550 (100.00%) | 30 / 268 / 7 / 245 | 54.18% | 81.08% | 10.91% | 97.45% | 19.23% | 54.18% | 54.83% | 53.64% |
| GPT-5.6 Luna | 550 / 550 (100.00%) | 46 / 268 / 7 / 229 | 57.09% | 86.79% | 16.73% | 97.45% | 28.05% | 57.09% | 64.65% | 64.05% |
| Claude Opus 4.8 | 544 / 550 (98.91%) | 129 / 258 / 17 / 140 | 71.14% | 88.36% | 47.96% | 93.82% | 62.17% | 70.89% | 74.86% | 77.63% |

## 4. Localization 指标如何计算

定位先将聚合后的所有 bbox 取并集，栅格化为原图大小的二值预测
mask。该 mask 是 MLLM bbox-to-mask adapter，不是模型原生像素分割。

- **主 T2 GT：** decoded source RGB 与 lossless spliced PNG 在 canonical
  JPEG 编码前的逐像素非零差分 mask；不是矩形
  `edit_region_xyxy`。
- **主结果：** forged 图片的 macro pixel IoU；同时报告 macro pixel
  F1 和 micro pixel F1/IoU。
- **Pixel AP：** MLLM 没有连续逐像素分数，二值 bbox union mask
  不计算 pixel AP。
- **辅助 box-hit：** bbox union mask 与 `edit_region_xyxy` mask 的
  IoU 严格大于 0.3。
- **辅助 Overlap：** 任一预测框与 `edit_region_xyxy` 有正面积交集；
  这不是像素 IoU。
- **Real no-edit accuracy：** 对 real/source 图，模型必须返回 `no_localized_edit` 且没有任何聚合预测框，才成功。
- forged 与 real 分开计算分母和 coverage；真实图使用全零像素 GT，
  并额外记录预测正像素比例。

## 5. Localization 结果

像素指标和辅助 box 指标的分母均为有效 forged 图；`Real no-edit`
的分母为有效 real 图。

| 模型 | 有效 / 550 | Forged 有效 | Macro pixel F1 | **Macro pixel IoU（主）** | Micro pixel F1 | Micro pixel IoU |
|---|---:|---:|---:|---:|---:|---:|
| Qwen 3.7 Plus | 550 (100.00%) | 275 | 5.85% | **4.74%** | 15.99% | 8.69% |
| GPT-5.6 Luna | 549 (99.82%) | 274 | 10.25% | **8.52%** | 25.18% | 14.40% |
| Claude Opus 4.8 | 543 (98.73%) | 268 | 21.54% | **15.04%** | 35.73% | 21.75% |

辅助诊断：

| 模型 | Box-hit@0.3 | 任意框 Overlap | Real no-edit |
|---|---:|---:|---:|
| Qwen 3.7 Plus | 18 / 275 (6.55%) | 22 / 275 (8.00%) | 273 / 275 (99.27%) |
| GPT-5.6 Luna | 30 / 274 (10.95%) | 36 / 274 (13.14%) | 272 / 275 (98.91%) |
| Claude Opus 4.8 | 66 / 268 (24.63%) | 119 / 268 (44.40%) | 261 / 275 (94.91%) |

这些数值是对已存储 v3 聚合框的事后重算。v3 后来发现存在像素坐标与
`bbox_1000` 解释歧义，尤其影响 Claude 的部分样本，因此最终主表应以
v4 像素坐标协议的新运行结果为准；v3 表保留为历史诊断。

## 6. 覆盖率与未完成项说明

- **Qwen：** detection 与 localization 均为 100% coverage。
- **GPT：** localization 缺 1 个有效单元：`restaurant_285_slot_001__forged` 的第 3 个 replicate 持续返回空内容，无法通过 JSON 解析；detection 已 100% coverage。
- **Claude：** detection 缺 6 个、localization 缺 7 个有效单元。其中 6 张 forged PNG 因网关的请求体大小限制被明确排除（影响两种协议）；另有 `lodging_233_slot_001__forged` 的 localization replicate 持续产生越界/退化 `bbox_1000`。这些样本未计入相应有效分母。

## 7. 结果文件

每个模型的完整逐图记录与 CSV/JSON metrics 位于：

- `results/mllm/<model-slug>/<run-id>.jsonl`：聚合结果（含有效性字段）。
- `results/mllm/<model-slug>/<run-id>.raw.jsonl`：所有 replicate 与失败尝试。
- `results/mllm/<model-slug>/metrics/<run-id>/detection_per_image.jsonl` 与 `localization_per_image.jsonl`：逐图 GT 对齐后的判定记录。
- 同目录下的 `detection_metrics.{csv,json}` 与 `localization_metrics.{csv,json}`：本文件引用的汇总数值。
