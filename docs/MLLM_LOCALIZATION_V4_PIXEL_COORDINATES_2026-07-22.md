# MLLM Localization v4：像素坐标与 IoU 评测

## 1. 变更目的

v3 同时向模型提供原图宽高并要求返回归一化 `bbox_1000`。Claude 的部分回复实际给出了像素坐标，但 runner 又把它按 0–1000 坐标换算为像素，造成纵坐标系统偏移。v4 只让模型输出一种坐标，消除这一歧义。

协议版本：`mllm_protocol_v4_reasoning_pixel_coordinates`。

Detection prompt 没有变化，继续使用 `mllm_protocol_v3_reasoning_image_coordinates`。Runner 按协议分别选择版本，并要求 detection 与 localization 分开运行，从而允许 detection v3 断点续跑，同时为 localization 新建 v4 run；任何结果文件都不会混合两个协议版本的 replicate。

## 2. 模型输出协议

localization 请求会附加原图尺寸，并要求模型只输出原图像素坐标：

```json
{
  "reasoning": "...",
  "decision": "localized_edit",
  "p_ai_edited": 80,
  "regions": [{
    "bbox_px": [456, 313, 579, 379],
    "confidence": 75,
    "evidence": "..."
  }]
}
```

对宽 `W`、高 `H` 的图片，动态约束为：

- 原点是完整原图左上角 `(0, 0)`；
- `0 <= x1 < x2 <= W`；
- `0 <= y1 < y2 <= H`；
- 不允许模型返回 `bbox_1000`，也不允许先做 0–1000 归一化。

schema repair 重试复用完整动态 prompt，因此不会再丢失图片宽高和像素坐标约束。

## 3. 聚合和结果存储

三次独立调用均在 `bbox_px` 空间中按 IoU 聚类。至少两个 replicate 支持同一簇时，对四个坐标分别取中位数，形成最终像素框。

`bbox_px` 是唯一权威坐标。聚合完成后，代码派生用于跨分辨率查看的 `bbox_1000`：

```text
x_1000 = x_px / W * 1000
y_1000 = y_px / H * 1000
```

结果同时保存：

- `regions_px`：最终像素框；
- `regions_1000`：由代码派生的归一化框；
- `result.regions`：带置信度和 replicate support 的权威 `bbox_px` 聚合记录。

不得根据 GT 决定如何解释模型坐标；v3 历史结果保持原样，v4 使用新的 run ID 重跑。

## 4. Localization 指标

对每张有效 forged 图片，GT 为 review export 的实际插入框 `edit_region_xyxy`。存在多个预测框时先计算每个预测框与 GT 的 IoU，再取最大值：

```text
best_iou = max(area(pred_i ∩ GT) / area(pred_i ∪ GT))
```

没有预测框时 `best_iou = 0`。

主指标拆成：

| 指标 | 成功条件 | 解释 |
|---|---|---|
| Overlap | 任一框与 GT 有正面积交集 | 是否大致找到目标；接近 `IoU > 0` |
| IoU@0.1 | `best_iou >= 0.10` | 粗定位，排除仅擦到边缘的框 |
| IoU@0.25 | `best_iou >= 0.25` | 中等精度定位 |
| IoU@0.5 | `best_iou >= 0.50` | 较严格的目标框定位 |

`All boxes inside GT` 继续保留为诊断项，但不作为主要 localization 指标，因为任意边界超出紧致 GT 即会失败。

逐图结果新增：

- `best_box_iou`；
- `box_iou_at_0_1`；
- `box_iou_at_0_25`；
- `box_iou_at_0_5`。

汇总 CSV/JSON 新增每个阈值的 successes 和 accuracy。

## 5. Claude v4 坐标 probe

选择 v3 中四张“reasoning 找到猫，但坐标发生错位”的 forged 图片，每图三次独立调用，共 12 次：

- 12/12 请求成功；
- 所有原始 region 均使用 `bbox_px`；
- 没有 schema repair；
- 3 张形成两票以上聚合框，3/3 均与 GT 重合；
- 另 1 张只有一次判断为 `localized_edit`，按现有多数规则没有聚合框，属于模型判断波动而非坐标转换错误。

形成聚合框的三张图片 best IoU 分别约为 `0.4620`、`0.4622`、`0.5745`。该 probe 验证了像素框、聚合、派生归一化框和 GT 评测链路。
