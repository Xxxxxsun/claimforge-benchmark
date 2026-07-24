# ClaimForge 猫图拼回逻辑教程

更新时间：2026-07-24

对应实现：[`compose_spliced_full.py`](../compose_spliced_full.py)

## 1. 先说结论

当前推荐的猫图拼回模式是：

```text
--blend object --object-search padded --object-pad 20 --object-thr 30
```

这是 2026-07-24 回退后的简单 baseline。它的核心思路是：

1. 对比原始 context crop 和模型生成的 crop；
2. 只在橙框向外固定扩展 20 像素的区域内保留差分大于 30 的像素；
3. 保留与橙框相交的最大连通域；
4. 轻微膨胀并羽化边界；
5. 只把 mask 内的生成像素混回原始 context，再放回整张源图。

固定 20 像素的是差分搜索范围。`feather=2` 会让最终 alpha 在边界外再产生少量平滑过渡，但不会引入新的远距离连通区域。此前的
`hysteresis-distance` 和 SAM3 方案保留用于对照，不再作为当前默认方案。

## 2. 三个坐标空间

每个任务同时涉及三层图像空间：

```text
整张 source image
└── 蓝框：context_region_xyxy
    └── 橙框：edit_region_in_context_xyxy
```

- `source_image`：原始完整图片；
- `context_region_xyxy`：蓝框在整图中的坐标，也是送给生成模型的 crop；
- `edit_region_xyxy`：橙框在整图中的坐标；
- `edit_region_in_context_xyxy`：橙框换算到蓝色 context 内部后的坐标。

例如任务中的：

```json
{
  "context_region_xyxy": [443, 400, 588, 494],
  "edit_region_xyxy": [487, 417, 570, 465],
  "edit_region_in_context_xyxy": [44, 17, 127, 65]
}
```

满足：

```text
487 - 443 = 44
417 - 400 = 17
570 - 443 = 127
465 - 400 = 65
```

所有 mask 运算都发生在蓝色 context crop 的坐标空间中。最终只把处理后的 context 放回 `(443, 400)`；蓝框以外的源图像素不会参与生成或混合。

## 3. 输入数据

脚本读取两份 JSONL。

### 3.1 任务文件

猫任务默认来自：

```text
annotations/cat_generation_tasks.jsonl
```

它提供源图、蓝框、橙框和生成前的 context crop。

关键字段：

```text
task_id
source_image
context_crop
context_region_xyxy
edit_region_xyxy
edit_region_in_context_xyxy
candidates
```

### 3.2 生成 manifest

当前 272 张 native-style v2 猫 crop 来自：

```text
generated_crops/hunyuan_image3_distil_cat_272_native_style_v2_20260722/manifest.jsonl
```

关键字段：

```text
task_id
input_context_crop
output_crop
paste_back
status
```

脚本只处理 `status == "ok"` 的行。

`paste_back` 决定了最上层分支：

- `paste_back: false`：需要从生成 crop 中提取 object mask，再与源图混合；`--blend` 参数在这里生效；
- 其他值或 `true`：整张生成 context 直接贴回，`--blend` 不生效，manifest 中记录 `paste_mode: full_context_crop`。

当前猫数据使用的是 `paste_back: false`。

## 4. 四种 blend 模式

入口参数定义在 [`compose_spliced_full.py#L414`](../compose_spliced_full.py#L414)。

| 模式 | mask 逻辑 | 适用场景 |
|---|---|---|
| `box` | 直接使用羽化后的橙框矩形 | 最简单 baseline；容易产生矩形色差 |
| `object` | 单阈值差分 + 固定像素 padding + 最大锚定连通域 | **当前 baseline** |
| `hysteresis` | 高阈值种子 + 固定低阈值 support 区域生长 | 历史实验；背景噪声可能形成桥接 |
| `hysteresis-distance` | hysteresis + 距离阈值 + 自动扩大 reach | 历史实验；审核后回退 |

当前 `object` baseline 实现在 [`object_mask()`](../compose_spliced_full.py#L63)。后续章节保留
`hysteresis_object_mask()` 的实现说明，便于复现实验结果。

## 5. 历史 `hysteresis-distance` 算法

下面按代码执行顺序拆解。

### 5.1 使用正确的 reference crop

mask 不是拿生成 crop 和一次新的源图裁剪直接比较，而是优先使用生成 manifest 中的：

```text
input_context_crop
```

如果它不存在，才回退到 task 中的 `context_crop`。代码位于 [`compose_spliced_full.py#L523-L530`](../compose_spliced_full.py#L523)。

这样做很重要：送入模型的 context 常常是 JPEG。重新从源图解码、裁剪得到的像素可能与实际输入有细小差别；使用同一输入文件能减少由编码差异产生的假 residual。

### 5.2 计算逐像素差分

reference 和 generated 都转换为 RGB，然后计算：

```text
D(x, y) = max(|R_ref - R_gen|,
              |G_ref - G_gen|,
              |B_ref - B_gen|)
```

代码是：

```python
diff = np.abs(ref - generated).max(2)
```

因此阈值范围是 RGB 的 `0..255`，它不是感知色差，也不是模型置信度。

### 5.3 建立橙框 anchor

`anchor` 是橙框对应的布尔 mask：

```python
anchor = box_mask(diff.shape, box)
```

只有橙框内的高置信变化像素可以成为初始种子，但后续 mask 可以长到橙框外。

### 5.4 建立 reach 区域

默认：

```text
--hysteresis-reach-ratio 0.5
```

先取橙框宽、高中的较大值：

```text
L = max(edit_box_width, edit_box_height)
```

再计算：

```text
reach_pad = round(L × reach_ratio)
```

reach 是橙框向四周扩展 `reach_pad` 后，与 context 边界相交得到的矩形。它限制 support 和区域生长，防止低阈值变化一路连到蓝框远处。

`reach_ratio = 0` 是特殊值，表示 support 可以搜索整个 context。

注意：reach 限制的是 support 和传播范围。最终的 `grow` 膨胀没有再次与 reach 相交，所以最终 mask 还能向外多出 `grow_iterations` 个像素。

### 5.5 构造随距离变化的阈值图

普通 `hysteresis` 在整个 reach 中使用固定 `low_thr`。

`hysteresis-distance` 对每个像素使用不同阈值：

```text
f(x, y) = clip(distance_to_anchor / reach_pad, 0, 1)

T(x, y) = low_thr + (far_thr - low_thr) × f(x, y)^power
```

默认参数：

```text
low_thr = 20
far_thr = 40
power   = 1
```

也就是说：

- 橙框内及紧邻区域的阈值接近 20，允许低对比的毛发、腿和阴影进入 support；
- 越接近 reach 外缘，阈值越接近 40；
- 远离橙框的轻微色调漂移很难进入 support。

`power` 的影响：

- `power = 1`：线性上升；
- `power > 1`：中间区域阈值更低，较宽松，接近外缘时才快速上升；
- `0 < power < 1`：离开橙框后更早升高阈值，较严格。

### 5.6 构造 support，并先做 `binary_closing`

原始 support：

```python
raw_support = (diff > support_threshold) & reach
```

随后执行：

```python
support = ndimage.binary_closing(
    support,
    structure=np.ones((3, 3), dtype=bool),
    iterations=close_iterations,
) & reach
```

所以 `binary_closing` 是当前代码中已经存在并实际使用的方案，而且发生在区域生长之前。

closing 是“先膨胀、再腐蚀”，主要用于连接以下断裂：

- 低对比毛发之间的小缝；
- 腿与身体间的窄断点；
- 被 residual 阈值切开的细尾巴；
- 小面积阴影断层。

代价是：`close_iterations` 太大时，附近的桌面纹理、椅腿或背景噪声也可能被连接成一条桥。

### 5.7 在橙框中选择 seeds

默认高阈值：

```text
--hysteresis-high 40
```

种子定义为：

```python
seeds = (diff > high_thr) & anchor
```

高阈值只负责确认“这确实是模型强烈修改过的位置”，并不直接决定最终边缘。

如果橙框内完全没有高阈值像素，代码会退回：

```python
seeds = support & anchor
```

并记录：

```text
used_low_seed_fallback: true
```

如果仍然没有种子，则返回羽化橙框，并记录 `used_box_fallback: true`。

### 5.8 8 邻域区域生长

区域生长使用：

```python
propagated = ndimage.binary_propagation(
    seeds,
    structure=np.ones((3, 3), dtype=bool),
    mask=support,
)
```

它可以从橙框种子向 8 个方向扩张，但只能走在 support 内。

与旧 `object` 模式不同，当前代码不会只保留一个最大连通域。传播后会保留每一个同时满足以下条件的连通分量：

1. 与 seed 相交；
2. 面积至少为 `hysteresis_min_component`，默认 6 像素。

这能避免“身体是最大分量，但腿、尾巴或另一片接触阴影被全部扔掉”的问题。

### 5.9 填洞、膨胀和羽化

有效分量合并后依次执行：

```text
binary_fill_holes
binary_dilation(grow_iterations)
GaussianBlur(feather)
```

默认：

```text
grow_iterations = 2
feather = 2
```

- `fill_holes` 恢复被差分阈值挖空的身体内部；
- `grow` 给毛发边缘和接触阴影留少量余量；
- `feather` 把硬二值边缘变成 alpha 过渡。

最终 mask 中：

- `255` 选择生成 crop；
- `0` 选择源图 crop；
- 中间值做 alpha 混合。

### 5.10 人工 reach 边界与真实 context 边界

代码把两种“碰边”分开记录：

#### `touches_reach_boundary`

mask 碰到了人为设置的 reach 边界，但还没有碰到蓝色 context 的真实边缘。

这通常说明猫可能只是被搜索窗口截断，而不是生成图本身缺失。

#### `touches_context_edge`

mask 在蓝色 context 最外侧约 2 像素带内仍包含 `diff > high_thr` 的高置信变化。

这通常说明生成的猫本身可能已经贴到或超出蓝框边缘。mask 拼回只能选择已有像素，不能恢复生成 crop 之外不存在的身体；代码因此设置：

```text
needs_regeneration: true
```

### 5.11 自动扩大 reach

默认参数：

```text
initial reach ratio = 0.50
auto-expand ratio  = 0.75
max added fraction = 0.05
```

当且仅当：

```text
touches_reach_boundary == true
touches_context_edge == false
```

代码会用 `reach_ratio = 0.75` 递归重算一次 mask。只重试一次，不会无限扩张。

扩张结果要同时满足：

1. 没有退回 box mask；
2. 没有碰到真实 context 边缘；
3. 新增 mask 像素不超过整个 context 面积的 5%。

满足时记录：

```text
auto_expand_attempted: true
auto_expand_applied: true
effective_reach_ratio: 0.75
```

否则保留原 mask，并在 `auto_expand_rejected_reason` 中写明原因。

当前接受条件没有要求扩张后的 mask 必须完全离开新的 reach 边界。因此审核时仍应同时查看 `touches_reach_boundary`，不能只看 `auto_expand_applied`。

## 6. 最终是怎样拼回整图的

当 `paste_back: false` 时，代码执行：

```python
crop_to_paste = Image.composite(
    generated_crop,
    original_crop,
    mask,
)
```

这里要区分：

- `reference_crop` 用于计算 residual；
- `original_crop = source.crop(context_region)` 用作最终背景。

也就是说，mask 外最终保留的是整张源图中解码出来的原始像素，而不是生成模型重建后的背景。

最后：

```python
spliced = source.copy()
spliced.paste(crop_to_paste, (context_x1, context_y1))
```

输出保存为 PNG，所以在 RGB 解码值层面，蓝框外应与源图完全一致。整张输出的宽高也与源图一致。

## 7. 默认参数与调参方向

| 参数 | 默认值 | 增大后的主要效果 | 主要风险 |
|---|---:|---|---|
| `--hysteresis-low` | 20 | 实际上增大它会让 support 更严格、更小 | 猫的低对比身体或阴影断裂 |
| `--hysteresis-high` | 40 | 种子更可靠、更少 | 橙框内可能没有 seed，触发 fallback |
| `--hysteresis-far-thr` | 40 | 远处更严格 | 远离橙框的尾巴或腿丢失 |
| `--hysteresis-distance-power` | 1.0 | `>1` 让中间距离更宽松 | 背景桥接增加 |
| `--hysteresis-close` | 3 | 连接更大的断裂 | 把背景纹理粘到猫上 |
| `--hysteresis-grow` | 2 | mask 外扩，保留边缘和阴影 | 光晕、背景泄漏 |
| `--hysteresis-min-component` | 6 | 过滤更多小块 | 小尾巴、小爪子被删 |
| `--hysteresis-reach-ratio` | 0.5 | 初始搜索区域更大 | 更容易吸入远处噪声 |
| `--hysteresis-auto-expand-ratio` | 0.75 | 碰人工边界时重试更大范围 | 扩张结果更容易过大 |
| `--hysteresis-auto-expand-max-growth` | 0.05 | 允许自动扩张增加更多像素 | 大片背景可能被接受 |
| `--feather` | 2.0 | 边缘更柔和 | 猫边缘发虚、出现半透明背景 |

注意 `low` 的名字容易误导：**降低** `low` 才会放宽 support；提高它会让 mask 更小。

## 8. 常见问题的诊断顺序

### 8.1 猫被截成一半

按以下顺序看 manifest：

1. `touches_context_edge == true`
   - 蓝框可能真的裁掉了生成主体；
   - 拼回 mask 无法创造缺失像素；
   - 应扩大 context crop 后重新生成，或者把该图标记为 bad。
2. `touches_reach_boundary == true` 且没有碰 context edge
   - 人工 reach 太小；
   - 看 `auto_expand_attempted` 和 `auto_expand_applied`；
   - 必要时增大 reach ratio。
3. 两个边界都没碰，但身体内部断裂
   - `low` 或 `far` 可能太高；
   - `close` 可能太小；
   - 也可能 reference 和 generated 没有正确配准。
4. `used_box_fallback == true`
   - 没有可靠 seed 或有效分量；
   - 当前返回的是羽化矩形，不是实际检测到的猫 mask。

### 8.2 mask 吃进大块背景

优先尝试：

1. 提高 `far_thr`；
2. 降低 `reach_ratio`；
3. 使用小于 1 的 `distance_power`，让阈值更早升高；
4. 减少 `close_iterations`；
5. 减少 `grow_iterations` 或 `feather`。

### 8.3 猫完整，但腿、尾巴或阴影缺失

优先尝试：

1. 降低 `low_thr`；
2. 若缺失发生在远离橙框的位置，降低 `far_thr`；
3. 适度增大 `close_iterations`；
4. 增大 reach，或启用自动扩张；
5. 检查缺失部分在 generated crop 中是否本来就不存在。

### 8.4 有矩形接缝或整体色块

检查：

- 是否使用了 `box`；
- 是否触发了 `used_box_fallback`；
- 生成 manifest 是否为 `paste_back: true`，导致整张 context 直接贴回；
- support 是否通过低阈值桥接成大面积背景。

## 9. 推荐运行命令

### 9.1 先输出 Hysteresis trial 到新目录

不要一开始覆盖已有审核结果。先运行到新目录：

```bash
python compose_spliced_full.py \
  --tasks annotations/cat_generation_tasks.jsonl \
  --model-name hunyuan_image3_distil_cat_272_native_style_v2_20260722 \
  --generated-manifest generated_crops/hunyuan_image3_distil_cat_272_native_style_v2_20260722/manifest.jsonl \
  --out-dir spliced_full/hunyuan_image3_distil_cat_272_native_style_v2_20260722_hysteresis_distance_trial \
  --blend hysteresis-distance \
  --hysteresis-low 20 \
  --hysteresis-high 40 \
  --hysteresis-close 3 \
  --hysteresis-grow 2 \
  --hysteresis-reach-ratio 0.5 \
  --hysteresis-far-thr 40 \
  --hysteresis-distance-power 1.0 \
  --hysteresis-auto-expand-ratio 0.75 \
  --hysteresis-auto-expand-max-growth 0.05 \
  --feather 2
```

### 9.2 固定 20 像素 object baseline

以下命令复现 272 张 native-style v2 猫图的固定 padding baseline：

```bash
python compose_spliced_full.py \
  --tasks annotations/cat_generation_tasks.jsonl \
  --model-name hunyuan_image3_distil_cat_272_native_style_v2_20260722 \
  --generated-manifest generated_crops/hunyuan_image3_distil_cat_272_native_style_v2_20260722/manifest.jsonl \
  --out-dir spliced_full/hunyuan_image3_distil_cat_272_native_style_v2_20260722 \
  --blend object \
  --object-search padded \
  --object-pad 20 \
  --object-thr 30 \
  --feather 2
```

### 9.3 只测试历史 hysteresis 方法

`--only` 接受逗号分隔的 `task_id` 或生成 manifest 中的零基索引：

```bash
python compose_spliced_full.py \
  --tasks annotations/cat_generation_tasks.jsonl \
  --model-name hunyuan_image3_distil_cat_272_native_style_v2_20260722 \
  --generated-manifest generated_crops/hunyuan_image3_distil_cat_272_native_style_v2_20260722/manifest.jsonl \
  --out-dir spliced_full/cat_hysteresis_debug \
  --blend hysteresis-distance \
  --only cat_restaurant_166_slot_001,cat_lodging_151_slot_001
```

不加 `--update-existing-manifest` 时，输出 manifest 只包含本次选中的任务。

### 9.4 覆盖历史 hysteresis 结果中的单张图

确认参数后，才对已有目录使用：

```bash
python compose_spliced_full.py \
  --tasks annotations/cat_generation_tasks.jsonl \
  --model-name hunyuan_image3_distil_cat_272_native_style_v2_20260722 \
  --generated-manifest generated_crops/hunyuan_image3_distil_cat_272_native_style_v2_20260722/manifest.jsonl \
  --out-dir spliced_full/hunyuan_image3_distil_cat_272_native_style_v2_20260722_hysteresis_distance \
  --blend hysteresis-distance \
  --only cat_restaurant_166_slot_001 \
  --update-existing-manifest
```

这个模式会：

1. 覆盖该 task 对应的 PNG；
2. 在现有 manifest 中替换同 `task_id` 的行；
3. 保留其他 task 的行；
4. 把当前命令参数写进被替换的行。

因此，单张修复后 manifest 可能包含多组参数。分析时要逐行读取参数，不能假设整个目录完全同构。

## 10. manifest 中最重要的诊断字段

拼回 manifest 位于：

```text
spliced_full/<out-dir>/manifest.jsonl
```

顶层字段：

```text
paste_mode
mask_reference
hysteresis_low_threshold
hysteresis_high_threshold
hysteresis_close_iterations
hysteresis_grow_iterations
hysteresis_reach_ratio
hysteresis_far_threshold
hysteresis_distance_power
hysteresis_auto_expand_ratio
hysteresis_auto_expand_max_growth
hysteresis_stats
```

`hysteresis_stats` 中重点看：

| 字段 | 含义 |
|---|---|
| `seed_pixels` | 橙框内 seed 数量 |
| `raw_support_pixels` | closing 前 support 面积 |
| `support_pixels` | closing 后 support 面积 |
| `mask_pixels` | 填洞、膨胀后的二值 mask 面积；box fallback 时不要把它当作实际矩形 alpha 面积 |
| `used_low_seed_fallback` | 高阈值无 seed，是否改用低阈值 seed |
| `used_box_fallback` | 是否退回羽化橙框 |
| `reach_pad` | 实际扩张像素数 |
| `requested_reach_ratio` | 命令指定的初始 ratio |
| `effective_reach_ratio` | 最终使用的 ratio |
| `touches_reach_boundary` | 是否碰人工搜索边界 |
| `touches_context_edge` | 是否碰蓝色 context 真边缘 |
| `needs_regeneration` | 是否更像生成 crop 本身被截断 |
| `auto_expand_attempted` | 是否尝试自动扩大 |
| `auto_expand_applied` | 扩大结果是否被采用 |
| `auto_expand_added_fraction` | 扩大后新增 mask 占 context 面积比例 |
| `auto_expand_rejected_reason` | 扩张拒绝原因 |

快速筛选风险任务：

```bash
python3 - <<'PY'
import json
from pathlib import Path

path = Path(
    "spliced_full/"
    "hunyuan_image3_distil_cat_272_native_style_v2_20260722_hysteresis_distance/"
    "manifest.jsonl"
)

for line in path.read_text().splitlines():
    row = json.loads(line)
    stats = row.get("hysteresis_stats") or {}
    if (
        stats.get("used_box_fallback")
        or stats.get("touches_reach_boundary")
        or stats.get("touches_context_edge")
        or stats.get("needs_regeneration")
        or stats.get("auto_expand_rejected_reason")
    ):
        print(row["task_id"], stats)
PY
```

## 11. 单独导出一张 mask 调试

主脚本目前不保存 mask PNG。需要肉眼检查时，可以直接调用同一个函数：

```bash
python3 - <<'PY'
import json
from pathlib import Path

from PIL import Image

from compose_spliced_full import hysteresis_object_mask

task_id = "cat_restaurant_166_slot_001"
tasks_path = Path("annotations/cat_generation_tasks.jsonl")
generated_path = Path(
    "generated_crops/"
    "hunyuan_image3_distil_cat_272_native_style_v2_20260722/manifest.jsonl"
)

tasks = {
    row["task_id"]: row
    for row in map(json.loads, tasks_path.read_text().splitlines())
}
generated = {
    row["task_id"]: row
    for row in map(json.loads, generated_path.read_text().splitlines())
}

task = tasks[task_id]
row = generated[task_id]
reference = Image.open(row["input_context_crop"]).convert("RGB")
edited = Image.open(row["output_crop"]).convert("RGB")

mask, stats = hysteresis_object_mask(
    reference,
    edited,
    task["edit_region_in_context_xyxy"],
    low_thr=20,
    high_thr=40,
    feather=2,
    close_iterations=3,
    grow_iterations=2,
    min_component_px=6,
    reach_ratio=0.5,
    far_thr=40,
    distance_power=1.0,
    auto_expand_ratio=0.75,
    auto_expand_max_growth=0.05,
)

Path("tmp").mkdir(exist_ok=True)
mask.save(f"tmp/{task_id}_mask.png")
print(stats)
PY
```

运行环境需要安装 `numpy`、`Pillow` 和 `scipy`。

## 12. 拼回后的必要验证

至少检查四件事：

1. 输出图尺寸与 source 完全相同；
2. 蓝框外 RGB 像素与 source 完全相同；
3. manifest 行数和 task_id 唯一数符合预期；
4. `needs_regeneration`、box fallback 和自动扩张拒绝项已进入人工审核。

验证蓝框外像素：

```bash
python3 - <<'PY'
import json
from pathlib import Path

import numpy as np
from PIL import Image

manifest = Path(
    "spliced_full/"
    "hunyuan_image3_distil_cat_272_native_style_v2_20260722_hysteresis_distance/"
    "manifest.jsonl"
)

for line in manifest.read_text().splitlines():
    row = json.loads(line)
    source = np.asarray(Image.open(row["source_image"]).convert("RGB"))
    output = np.asarray(Image.open(row["spliced_full"]).convert("RGB"))
    assert source.shape == output.shape

    x1, y1, x2, y2 = row["context_region_xyxy"]
    outside = np.ones(source.shape[:2], dtype=bool)
    outside[y1:y2, x1:x2] = False
    assert np.array_equal(source[outside], output[outside]), row["task_id"]

print("all context-exterior pixels are identical")
PY
```

## 13. 当前 272 张结果的版本状态

当前保留的 native-style v2 基线目录是：

```text
spliced_full/hunyuan_image3_distil_cat_272_native_style_v2_20260722_hysteresis_distance/
```

它与当前
`generated_crops/hunyuan_image3_distil_cat_272_native_style_v2_20260722/`
生成集配套。旧的通用 `hunyuan_image3_distil_cat_272` lineage 及其拼接、
SAM3 结果已从当前仓库版本移除。

分析现有结果时仍应逐行读取 manifest 参数。如果需要严格同参数的研究
对照，应使用新的 `out-dir` 对全部 272 张统一重拼，避免覆盖已审核基线。

## 14. 当前方法的能力边界

这仍然是 residual 驱动的 mask，不是语义分割：

- 它不知道哪些像素在语义上属于猫；
- 与 seed 连通的强背景重建变化仍可能被吸入；
- Gaussian feather 只是边缘平滑，不是真实 alpha matting；
- 生成 crop 已经裁掉的身体无法由拼回逻辑恢复；
- box fallback 会重新引入矩形区域风险。

如果差分 mask 的上限不能满足要求，下一步是 SAM 3 语义 mask 与 residual 的混合方案。相关调研见 [`VISION_BANANA_SAM3_SPLICEBACK_RESEARCH_2026-07-21.md`](VISION_BANANA_SAM3_SPLICEBACK_RESEARCH_2026-07-21.md)。

## 15. 一句话调参决策树

```text
猫碰蓝框真边缘？
├─ 是：扩大 context 并重新生成；mask 无法补不存在的像素
└─ 否
   ├─ mask 碰人工 reach 边界：自动扩张或增大 reach
   ├─ 猫内部断裂：降低 low/far，或小幅增加 close
   ├─ 背景泄漏：提高 far，减小 reach/close/grow
   ├─ 边缘光晕：减小 grow/feather
   └─ box fallback：检查 reference 对齐、seed 阈值和生成质量
```
