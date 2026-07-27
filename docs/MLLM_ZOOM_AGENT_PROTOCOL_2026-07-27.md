# MLLM Zoom-Agent 检测协议（2026-07-27）

## 1. 目标

本协议在普通单轮 MLLM 检测之外增加一个主动视觉检查条件：

1. 模型先查看完整原图；
2. 模型可以请求 `zoom_in`，从原始分辨率图像裁剪一个区域；
3. runner 将放大的局部图作为新的 observation 回传给同一模型；
4. 每个独立 episode 最多执行五次 `zoom_in`；
5. 模型结合整图上下文和局部证据给出 detection 与 localization 最终结果。

协议版本：

| 层级 | 版本 |
| --- | --- |
| Agent suite | `mllm_zoom_agent_v1_20260727` |
| Detection | `mllm_zoom_agent_detection_v1_20260727` |
| Localization | `mllm_zoom_agent_localization_v1_bbox1000_20260727` |

该条件用于回答：在模型能够主动检查局部高分辨率细节时，局部伪造检出率和定位质量是否改善。

## 2. 参考实现与选择

[Zoom-Refine](https://arxiv.org/abs/2506.01663)采用“整图初步判断和区域定位 → 从原始高分辨率图裁剪 → 将局部图交回同一 MLLM 做 self-refinement”的训练外流程。本实现沿用这三个关键点，但允许模型最多进行五次自适应 zoom，而不是固定一次。

[OpenAI Agents SDK 的 function-tools 文档](https://openai.github.io/openai-agents-python/tools/)说明本地 function tool 可以使用结构化参数，并可将图片作为 tool output 返回给模型。当前实验的 GPT、Claude、Qwen、Doubao 通过同一 OpenAI-compatible 网关调用，不同后端对原生 `tool_calls` 和图片型 tool result 的支持并不一致。因此，本协议使用等价的应用层 JSON action：

- 模型输出结构化 `zoom_in` action；
- Python runner 校验参数并执行本地裁剪；
- crop 以新的多模态 user observation 回传；
- 完整消息历史仍包含原始整图、模型 action 和所有 crop。

这种实现不依赖某一家 SDK 的 agent runtime，并且每一轮 action、裁剪框和图像文件均可审计。

## 3. Zoom 工具

### 3.1 调用格式

```json
{
  "action": "zoom_in",
  "reasoning": "需要检查该区域边缘和阴影是否连续",
  "bbox_1000": [x1, y1, x2, y2]
}
```

规则：

- 坐标始终相对**原始完整图像**，不是相对上一张 crop；
- 左上角为 `(0, 0)`，右下角为 `(1000, 1000)`；
- 必须满足 `0 <= x1 < x2 <= 1000`、`0 <= y1 < y2 <= 1000`；
- 正式全量实验每个 episode 最多执行 5 次；
- 第五次工具结果返回后，下一轮必须给出 `final`；
- 请求第六次 zoom 会触发 schema repair，不会执行第六次裁剪。

### 3.2 裁剪与放大

工具从本地原始图像执行以下操作：

1. 应用 EXIF orientation；
2. 将 `bbox_1000` 映射回原图像素；
3. 用向外取整的像素边界裁剪，确保请求区域不丢失；
4. 保持宽高比，用 Lanczos 将 crop 的长边至少放大到 `1536 px`；
5. 以无损 PNG 保存并用 base64 回传。

如果 crop 的长边原本大于 1536 px，则不缩小。该行为避免 zoom 工具因二次 JPEG 压缩引入新的伪造痕迹。

可配置参数：

```text
--max-zoom-calls 5
--zoom-long-side 1536
```

为保证协议可比性，正式全量实验固定 `--max-zoom-calls 5`。早期 pilot 使用 2 次上限；二者必须通过 run manifest 和 run ID 分开统计。可以用 `--max-zoom-calls 0` 做同一 prompt 家族下的无工具消融。

## 4. Agent loop

一次正式 episode 最多包含六个推理 turn：

```text
Turn 1: 原始整图
  ├─ final → 结束
  └─ zoom_in #1
       ↓
Turn 2: 完整历史 + crop #1
  ├─ final → 结束
  └─ zoom_in #2
       ↓
Turn 3–5: 按需继续 zoom_in #3–#5
       ↓
Turn 6: crop #5 后必须 final
```

每次 API 返回都先进行严格 JSON/schema 校验。格式错误、越界框或超出工具次数会在同一 turn 内重试；网络类重试也不会消耗 zoom 配额。每个 episode 的有效条件是最终得到一个合法 `final`。

## 5. 最终输出

```json
{
  "action": "final",
  "reasoning": "结合整图与局部放大图的详细判断",
  "decision": "edited",
  "p_ai_edited": 85,
  "evidence": ["局部边缘存在不连续光晕"],
  "regions": [{
    "bbox_1000": [520, 210, 830, 760],
    "confidence": 88,
    "evidence": "目标周围的纹理和阴影不连续"
  }]
}
```

- `decision` 只使用 `edited` / `not_edited`；
- `regions` 最多 3 个，并始终使用原图 `bbox_1000`；
- `not_edited` 必须配合空 `regions`；
- 如果模型认为整图有编辑，但不能可靠定位，可返回 `edited` 和空 `regions`；
- runner 在解析后立刻将 `bbox_1000` 转换为原图像素坐标，再进入三次聚合。

## 6. 三次独立检测与聚合

每张图运行 3 个相互独立的 agent episode。一个 episode 内最多有 6 次模型推理和 5 次工具执行。

Detection：

- 三次 `edited` / `not_edited` 多数票；
- `p_ai_edited` 取中位数。

Localization：

- 每次 episode 的 final regions 转为原图像素；
- 跨 replicate 以 `IoU >= 0.10` 聚类；
- 至少得到两个不同 replicate 支持的 cluster 才进入聚合 regions；
- 决策由三次 episode 中是否给出有效区域进行多数票；
- 继续生成 bbox union 二值 mask，因此可直接复用现有 T2 metrics。

Detection 与 localization 来自同一组三次 agent episode，不会为两个 protocol 重复调用模型。

## 7. 结果目录

```text
results/mllm/<model>/agent_zoom/
  <run_id>.raw.jsonl
  <run_id>.jsonl
  <run_id>.run_manifest.json
  <run_id>.agent_metrics.json
  <run_id>.agent_metrics.csv
  crops/<run_id>/<image_id>/replicate_<n>_zoom_<n>.png
  masks/<run_id>/<image_id>.png
  metrics/<run_id>/
```

`raw.jsonl` 每行对应一个完整 episode，记录：

- 所有 inference turns；
- 每个 turn 的原始响应、解析 action、重试与延迟；
- 每次工具调用的 `bbox_1000`、`bbox_px`；
- crop 输入/输出尺寸与文件路径；
- 最终 detection/localization 解析结果。

聚合文件 `<run_id>.jsonl` 每张图产生两行，分别是 detection 和 localization，可直接交给现有 `eval.mllm.metrics`。

`agent_metrics` 不使用 GT，汇总 episode coverage、0–5 次 zoom 分布、工具使用率、用满预算比例、推理 turn 数、延迟、最终判断分布和各类重试状态。它用于区分“工具可用但模型没有调用”和“模型调用工具后仍然判断失败”。

所有文件均不向模型暴露或在 prompt 中包含 GT、review 状态、任务类别或文件名。

## 8. 运行示例

对 review export 中的 good forged 图和 paired real 图运行一个模型：

```bash
conda run -n utils python -m eval.mllm.run_zoom_agent \
  --config config/mllm_eval.local.json \
  --source review-export \
  --review-export annotations/<review_export>.json \
  --review-status good \
  --include-source-pairs \
  --condition zoom_agent_good \
  --run-id <unique_run_id> \
  --model-slug qwen3_7_plus \
  --concurrency 15 \
  --max-zoom-calls 5 \
  --zoom-long-side 1536 \
  --retry-until-complete \
  --write-metrics
```

对模块化图片名单运行：

```bash
conda run -n utils python -m eval.mllm.run_zoom_agent \
  --config config/mllm_eval.local.json \
  --source list \
  --list config/<input_list>.jsonl \
  --condition zoom_agent_list \
  --run-id <unique_run_id> \
  --model-slug gpt \
  --concurrency 15
```

名单仍使用现有格式：

```json
{"id":"sample_001","image_path":"spliced_full/.../sample_001.png"}
```

Zoom 必须从原图像素裁剪，因此输入必须包含可访问的本地 `image_path`。首轮整图仍可通过 `--image-transport url` 发送；工具生成的 crop 固定用 base64 回传。

## 9. 断点续跑

runner 按 `(image_id, replicate_index, mllm_zoom_agent_v1_20260727)` 识别成功 episode，并按 detection/localization 各自的 leaf protocol version 识别已完成聚合。重复执行相同命令会：

- 跳过已有的成功 episode；
- 只补缺少的 replicate；
- 三个有效 episode 齐备后才写有效聚合；
- 不把普通单轮协议或其他 zoom-agent 版本的结果混入本次统计。
