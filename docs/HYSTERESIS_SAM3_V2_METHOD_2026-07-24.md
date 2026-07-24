# Hysteresis-SAM3 v2 拼回算法

更新时间：2026-07-24

状态：**当前人工审核候选；尚未替代 `hysteresis-distance` 生产基线。**

对应实现：

- [`eval/segmentation/materialize_hysteresis_sam3.py`](../eval/segmentation/materialize_hysteresis_sam3.py)
- [`tools/hysteresis-sam3-cat-review.html`](../tools/hysteresis-sam3-cat-review.html)

## 1. 目标

Hysteresis-SAM3 v2 结合两类互补证据：

1. SAM3 提供猫主体的高置信语义轮廓；
2. 源 context crop 与生成 crop 的像素差分补回 SAM3 容易遗漏的毛发、接触阴影和投影。

单独使用 SAM3 容易截断阴影；单独使用 `hysteresis-distance` 又可能把重建背景或亮部连入 mask。v2 因此将 SAM3 作为可信主体锚点，只允许与主体连通、范围受限且不会让主体外像素变亮的残差进入最终拼接。

算法只在 `context_region_xyxy` 内工作。完成后再将处理过的 context 放回完整源图，context 外像素必须与源图逐字节一致。

## 2. 总体流程

```text
源 context + 生成 crop + SAM3 主体 mask
                    |
                    v
       计算 source/generated RGB 差分
                    |
                    v
  从 SAM3 边界向外寻找距离自适应 residual
                    |
                    v
   连通传播 + component gate + 闭运算/填洞
                    |
                    v
 删除任一 RGB 通道高于原图的主体外 residual
                    |
                    v
 SAM3 主体 alpha 与 residual alpha 分别羽化
                    |
                    v
 再次清除主体外会造成亮化的最终 alpha 像素
                    |
                    v
       生成 crop 与源 context 合成并贴回整图
```

## 3. 输入与坐标

每条任务使用：

- `source_image`：完整源图；
- `generated_crop`：模型生成的 context crop；
- `context_region_xyxy`：context 在完整源图中的位置；
- `sam3_semantic_mask`：与生成 crop 同尺寸的 SAM3 猫主体 mask；
- `edit_region_xyxy`：仅作为任务元数据保留，不限制本算法的残差方向。

所有 mask、距离和 RGB 约束都在 context crop 坐标空间内计算。v2 不假设阴影一定位于猫下方，可以向任意方向搜索。

## 4. 残差候选

### 4.1 像素差分

源 context 记为 \(S\)，生成 crop 记为 \(G\)。逐像素差分为：

```text
D(x, y) = max(|S_R - G_R|, |S_G - G_G|, |S_B - G_B|)
```

这里的源 context 是生成时对应的原始输入区域，不是拼接成品。

### 4.2 自适应搜索距离

SAM3 主体面积为 \(A\)，以 `sqrt(A)` 估计主体尺度。初始搜索距离为：

```text
initial_reach = clip(round(sqrt(A) * 0.20), 12, 64)
```

如果残差触及初始 reach 边界，则尝试扩张到：

```text
expanded_reach = clip(round(sqrt(A) * 0.35), 12, 64)
```

扩张后的残差面积不得超过 SAM3 主体面积的 70%，否则保留初始结果。reach 只表示距 SAM3 主体的欧氏距离，不施加“必须在猫下方”等方向先验。

### 4.3 距离阈值

距离 SAM3 主体为 \(d\)，当前 reach 为 \(R\)。support 阈值为：

```text
T(d) = 12 + (28 - 12) * clip(d / R, 0, 1)^1.5
```

原始候选需要满足：

```text
D(x, y) > T(d)
```

离主体越远，进入 support 所需的差分越大。

### 4.4 远距离阴影方向约束

attachment band 为 `clip(round(reach * 0.20), 3, 8)`。超过下面距离的像素属于远区：

```text
max(attachment_band, reach * 0.35)
```

远区原始候选还必须满足生成像素的三个 RGB 通道均比源像素暗超过 5；对于整数 RGB，即至少暗 6：

```text
min(S - G) > 5
```

该条件只用于远区原始候选。最终结果还有更严格的全残差非增亮约束，见第 6 节。

## 5. 连通与形态学筛选

原始 support 依次执行：

1. 两轮 `3×3 binary_closing`，连接小断点；
2. 在紧邻 SAM3 的 attachment band 内允许较弱种子；
3. reach 前半段中 `D >= 40` 的像素可作为强种子；
4. 仅保留能从上述种子传播到的八连通 component；
5. component 至少包含 3 个像素；
6. component 面积不得超过 `max(2 × SAM3 面积, context 面积 × 15%)`；
7. 对保留区域填洞；
8. 当前不执行额外膨胀（`grow_iterations = 0`）；
9. 从 residual 中移除 SAM3 主体区域。

这一阶段解决“阴影与主体之间存在小断点”的问题，但闭运算和填洞也可能连接到原始候选之外的像素，因此之后必须重新执行颜色约束。

## 6. 连通后的 RGB 非增亮硬约束

这是 2026-07-24 v2 的最终修正。

完成连通和形态学处理后，对每个 residual 像素重新比较源图与生成图：

```text
保留条件：
G_R <= S_R
and G_G <= S_G
and G_B <= S_B
```

只要生成像素任一通道高于对应源像素，该像素就从 residual mask 删除。相等允许保留。

该约束只作用于 SAM3 主体外的补充残差；SAM3 识别到的猫主体始终保留。它的目的不是证明某处在语义上属于阴影，而是保证额外贴回的“阴影候选”不会让原图变亮。

## 7. 最终 alpha 与拼接

语义主体和 residual 分别羽化：

```text
semantic_alpha = GaussianBlur(SAM3 mask, 0.5 px)
residual_alpha = GaussianBlur(residual mask, 1.0 px)
```

为避免羽化重新引入亮部：

1. `residual_alpha` 在任一 RGB 通道变亮的像素上清零；
2. 两个 alpha 取逐像素最大值；
3. 在 SAM3 主体外，再次将任一 RGB 通道变亮位置的最终 alpha 清零。

最终：

```text
alpha = max(semantic_alpha, residual_alpha)
context = Image.composite(generated, source_context, alpha)
```

处理后的 context 贴回完整源图。程序同时检查 context 外像素是否保持完全不变。

页面中的几类诊断图含义如下：

- `v2 二值范围 mask`：SAM3 主体与最终 residual 的并集，仅用于查看覆盖范围；
- `v2 残差支持`：经过连通及非增亮硬约束后的 residual；
- `v2 最终使用 alpha`：真正传入 `Image.composite` 的灰度 alpha；
- `v2 拼接成品`：最终完整图片。

## 8. 当前参数

| 参数 | 值 | 作用 |
|---|---:|---|
| `low_threshold` | 12 | 主体附近差分下限 |
| `far_threshold` | 28 | reach 外缘差分下限 |
| `high_threshold` | 40 | 强种子阈值 |
| `distance_power` | 1.5 | 距离阈值曲线 |
| `reach_scale` | 0.20 | 初始 reach |
| `auto_expand_scale` | 0.35 | 触边后的扩张 reach |
| `min_reach_pixels` | 12 | 最小 reach |
| `max_reach_pixels` | 64 | 最大 reach |
| `auto_expand_max_growth_over_semantic` | 0.70 | 扩张面积安全门 |
| `far_direction_start_ratio` | 0.35 | 远区起点 |
| `far_shadow_channel_min` | 5 | 远区每通道最小变暗量 |
| `close_iterations` | 2 | 小断点连接 |
| `min_component_pixels` | 3 | 最小 component |
| `semantic_feather` | 0.5 px | 主体边缘羽化 |
| `residual_feather` | 1.0 px | 阴影/残差羽化 |
| `grow_iterations` | 0 | 不额外膨胀 |

## 9. 272 张猫图验证

结果目录：

```text
results/segmentation/hysteresis_sam3_v2_cat_native_style_v2_full272_20260724/
```

| 检查项 | 结果 |
|---|---:|
| 成功生成 | 272/272 |
| context 外像素变化 | 0 |
| 非增亮过滤已启用 | 272/272 |
| 自动扩张尝试 | 192 |
| 自动扩张采用 | 177 |
| 自动扩张拒绝 | 15 |
| 删除亮 residual 像素 | 433,526 |
| 清零 residual 羽化亮像素 | 564,911 |
| 清零最终主体外 alpha 亮像素 | 112,605 |
| 最终 residual 中任一通道亮化违规 | 0 |
| 最终主体外 alpha 中亮化违规 | 0 |
| 平均 residual/SAM3 面积比 | 0.3065 |
| 中位 residual/SAM3 面积比 | 0.2828 |

`cat_lodging_027_slot_001` 是本次修正的代表案例：

- SAM3 主体：17,510 像素；
- 最终 residual：4,946 像素；
- 连通后删除亮 residual：2,724 像素；
- 最终清除主体外羽化亮像素：542 像素；
- residual 与主体外最终 alpha 的亮化违规均为 0。

## 10. 复现

无需再次调用 SAM3 API。使用已经保存的 SAM3 结果离线生成：

```bash
PYTHONPATH=. conda run -n utils python \
  eval/segmentation/materialize_hysteresis_sam3.py \
  --sam3-results-dir \
  results/segmentation/fal_sam3_cat_native_style_v2_full272_20260723 \
  --output-dir \
  results/segmentation/hysteresis_sam3_v2_cat_native_style_v2_full272_20260724
```

运行测试：

```bash
PYTHONPATH=. conda run -n utils python -m unittest \
  tests.test_run_fal_sam3
```

本地审核：

```text
http://localhost:8000/tools/hysteresis-sam3-cat-review.html
```

页面默认以全图拖动方式比较 Hysteresis-SAM3 v1 与 v2。

## 11. 局限与审核重点

- RGB 非增亮是光度约束，不是阴影语义分类器；与原图等亮或更暗的背景重建误差仍可能被保留。
- 彩色反射、半透明物体或局部曝光变化不一定符合“三通道均不增亮”的假设，可能被过度删除。
- SAM3 主体是可信核心；如果 SAM3 主体本身包含背景，算法不会用非增亮约束删除该部分。
- 仍有 126 张结果的 residual 触及当前 reach 边界，27 张触及 context 边界，需要重点人工检查阴影是否仍被截断或是否连接到背景。
- 当前结果仍是审核候选。完成 good/bad 标注和与 v1、SAM3 semantic-shadow 的定量比较前，不应直接替换 benchmark 的正式图片。
