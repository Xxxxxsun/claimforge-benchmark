# Native-style v2 猫图可用数量（2026-07-23）

## 结论

当前两套人工审核都覆盖同一批 272 个猫任务，且没有未标注项：

| 拼回方案 | good | bad | source_bad | total |
|---|---:|---:|---:|---:|
| hysteresis-distance | 192 | 68 | 12 | 272 |
| SAM3 hybrid | 133 | 137 | 2 | 272 |

如果要求所有图片来自同一种拼回方案，当前最多可直接使用
**192 张 hysteresis-distance 结果**。

如果允许逐任务从两套结果中择优：

- 两套都为 `good`：101 张；
- 仅 hysteresis-distance 为 `good`：91 张；
- 仅 SAM3 为 `good`：32 张；
- 原始 `good` 并集：224 张；
- 两套都不是 `good`：48 张。

原始并集中有 7 张在另一套审核中被标为 `source_bad`。在二次人工仲裁前，
建议将任一 `source_bad` 视为一票否决，因此当前**保守混合可用数为 217 张**
（restaurant 97，lodging 120）。目前尚未把这 217 张整理成统一输出目录或
selection manifest。

## `source_bad` 冲突项

以下任务在一套审核中为 `good`，但在另一套审核中为 `source_bad`：

```text
cat_lodging_118_slot_001
cat_restaurant_002_slot_001
cat_restaurant_115_slot_001
cat_restaurant_188_slot_001
cat_restaurant_256_slot_001
cat_restaurant_276_slot_001
cat_restaurant_285_slot_001
```

这些任务需要重新查看源图/context 后再决定是否加入最终集合。

## 统计来源与口径

- hysteresis-distance 标签：
  `annotations/claimforge_cat_native_style_v2_hysteresis_distance_review_labels.json`
- SAM3 标签：
  `annotations/claimforge_cat_native_style_v2_sam3_review_labels.json`
- 接受字段：`records[].status == "good"`
- `bad` 表示当前拼回结果不可用；`source_bad` 表示源图/context 本质不适合；
  `unlabeled` 表示尚未审核。

生成目录中的 272 条 `status=ok` 只代表生成请求和文件写入成功，不能替代人工
`good` 判定。本文数字统计的是最终拼回结果的人工审核标签，不是 crop-only
生成成功率。
