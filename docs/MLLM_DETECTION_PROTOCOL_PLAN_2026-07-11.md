# CLAIMFORGE 家族 D：MLLM 零样本检测协议与实现计划

**状态：方案已按用户确认落地为本地框架；尚未发起任何真实模型调用或批量实验。**
**日期：2026-07-11**
**目标：** 在 CLAIMFORGE 的规范化图像上，以可复现的零样本协议评测配置化的多模态 LLM。每张图输出两类结果：整图 AI 局部编辑检测（T1）与候选编辑区域定位（T2）。

本方案落实 [`CLAIMFORGE_experiment_plan_2026-07.md`](CLAIMFORGE_experiment_plan_2026-07.md) §3 家族 D、§4 与 §6 的要求，并保持模型、输入来源、API 网关和结果路径均可配置。

---

## 1. 文献复核：两篇论文实际采用了什么

### 1.1 FragFake（arXiv:2505.15644）

论文：[Can VLMs Detect and Localize Fine-Grained AI-Edited Images](https://arxiv.org/abs/2505.15644)（FragFake）。

- **任务与数据：** 作者构造了局部 object addition / replacement 的编辑图数据集；原文的 v2 报告约 20,222 张编辑图，来自 MagicBrush、GoT、UltraEdit 与 Gemini-IG。
- **零样本评测：** 比较 GPT-4o(-mini)、GLM-4V、Gemini 2.5 与若干开源 VLM；温度为 0.1。输出先给解释，再给一个对象级最终结论。分类通过关键词匹配计分；定位不是坐标或 mask，而是人工判断模型说出的物体/区域是否与 GT 语义对应（Region Precision / Object Precision）。
- **关键结果：** GPT-4o 是预训练模型中最强的一档，但在 Gemini-IG 测试子集上的 object precision 仍仅约 45–46%；许多预训练模型接近随机。该论文的强结果来自用 FragFake 数据进行 LoRA 微调的 Qwen2.5-VL，而不是换一个零样本 prompt。
- **对 CLAIMFORGE 的启示：** 不能把模型一句“鼠标看起来奇怪”当作原生定位能力；需显式要求可解析坐标，并披露 bbox→mask 是评测适配器。

**论文公开的检测 prompt（Appendix B.2，短摘录与完整结构）：** 该 prompt 将模型设为视觉分析助手，要求检查光照、阴影、纹理、边缘、透视和逻辑构图等细微不一致，并要求“先解释、后结论”。最终结论只能二选一：

```text
The <thing> in the image has been modified.
Nothing has been modified in this image.
```

`<thing>` 应替换为被判断修改的简短对象描述。原论文没有要求数值置信度、JSON、坐标框或像素掩码；其定位评估的是对象/区域文本描述与 GT 的语义对应，而不是 bbox/mask。

### 1.2 LLMs Are Not Yet Ready for Deepfake Image Detection（arXiv:2506.10474）

论文：[LLMs Are Not Yet Ready for Deepfake Image Detection](https://arxiv.org/abs/2506.10474)。

- **任务与模型：** 对 GPT-4o、Claude Sonnet 4、Gemini 2.5 Flash、Grok 3 做零样本单图评测，覆盖 faceswap、reenactment 与整图 GAN/diffusion 图；模型经各自网页接口调用。
- **统一格式：** 论文使用匿名 persona 后附“`Analyse the following image. Follow your analysis framework.`”，并固定输出 `Decision`（Real / Fake / Inconclusive）、`Confidence`、`Reasoning`、`Uncertainty`。每张图由各模型独立判断；作者还在子集重复查询以观察不一致性。
- **关键结果：** GPT-4o 相对最好，但没有模型能跨伪造类型稳定可靠；Gemini 对真实类的偏好会漏掉很多假图，ChatGPT 对风格化/棚拍真图会误报，复古/胶片风格可诱发错误真实性判断。
- **对 CLAIMFORGE 的启示：** 需把 `reasoning`、置信度和原始输出完整记录；不得因回答内容重试或挑选结果。真实照片的风格、清晰度、物体语义本身不能被当作伪造证据。

**论文公开的 prompt 格式（§2.2）：** 论文的完整 persona 未公开（双盲时以链接匿名），但公开了逐图任务句及严格回答结构：

```text
Analyse the following image. Follow your analysis framework.

Decision: [Real / Fake / Inconclusive]
Confidence: [e.g., 82% confident it’s real]
Reasoning: [Detailed paragraph with evidence and rationale]
Uncertainty: [If confidence < 80%, describe limitations]
```

这篇论文没有要求坐标框；其可重复性检查是在子集上重复提问，记录预测标签与置信度的变异。

### 1.3 与本协议的差异（有意为之）

两篇论文都没有提供适合 CLAIMFORGE 像素指标的、强制机器可读坐标协议。本文档因此保留它们的**零样本、固定提示词、显式置信度、记录不确定性**原则，但新增独立的定位调用，并将 0–1000 标准化 bbox 明确转换为 mask。该转换只能称为 *MLLM bbox-to-mask adapter*，不能称为模型原生像素分割。

---

## 2. 评测对象、盲化与运行范围

### 2.1 模型

框架不硬编码厂商；每个模型由单独配置条目给出。**已确认的 pilot 模型为 GPT 与 Gemini**（具体 API model ID、base URL、鉴权和网关 extra body 待用户提供）。

每个模型均使用同一 `mllm_protocol_v1`。不得将模型名、图像文件名、`task_id`、域名、生成器、物体类别、GT 标签、review 的好/坏状态传入 prompt。

### 2.2 评测图像

主实验目标是 D2 canonical v1 中的每个伪图及其同源真图。若 canonical v1 尚未生成，输入模块可暂从 review export 选样：

- `claimforge_generation_review_labels.json`：按 `status == "good"` 选用合格生成/拼回图；`spliced_image` 作为 forged 图。
- `--include-source-pairs`：为每一条 forged 图加入同记录的 `source_image` 作为 real 图，并按路径去重。
- 注意：review `status` 只是生成质量筛选条件，不是传给模型的标签，也不是模型输出的监督信号。

**已确认的 pilot：** 读取当前 review export 中 `status == "good"` 的 275 条 forged 记录，并各自加入同记录的 `source_image`；预期得到 275 forged + 275 real = **550 张**图。runner 必须在启动时报告去重后的实际数量及输入清单 SHA-256；若 source path 缺失或与 forged 图相同，记录并停止该 pair，而非静默替换。

使用其他图像名单时，输入 JSONL 的最小行格式为：

```json
{"id":"example_001","image_path":"/absolute/or/repo-relative/path.jpg","label":"real|forged","mask_path":"optional.png","metadata":{"domain":"restaurant"}}
```

`label` 与 `mask_path` 仅供运行后 metrics join；runner 不会把它们放入请求。

### 2.3 图像传输与公平性

- 支持两种 OpenAI 兼容图文输入，并在 run manifest 中冻结其模式：

  ```json
  [{"role":"user","content":[
    {"type":"text","text":"<protocol prompt>"},
    {"type":"image_url","image_url":{"url":"https://public-or-presigned.example/image.jpg"}}
  ]}]
  ```

  或将 `url` 替换为 `data:<mime>;base64,...`。`url` 模式必须是模型供应商可访问的 HTTPS 外链/预签名 URL，不能传 `file://` 或 `localhost`；本地 review 图可使用 base64 模式。
- 每条记录保存 `image_transport`（`url` / `base64`）、URL host（不保存签名 query）或发送图像 SHA-256、MIME、宽高和实际发送字节数。
- 不裁剪、不锐化、不做局部放大、不读取 EXIF；两类图完全走同一预处理。
- 仅当 API 尺寸限制拒绝图像时，按配置把长边等比例缩至 2048 px，并在每条结果中写入变换。real 与 forged 必须使用同一规则。
- 请求中使用视觉输入的高细节模式（若网关支持）；不传生成任务 prompt、mask、候选物体或任何人类标注。

---

## 3. 两种输出协议（每图两次独立调用）

两个调用都发送同一张图、相同模型参数，但不共享前一问回答。这样 T1 的分数不会因为模型被迫“找一个框”而被污染，T2 也不会被第一问的答案锚定。

### 协议 A：`mllm_detection_v1`（T1，整图检测）

**用途：** 主 T1 分数。`p_ai_edited / 100` 进入 AUC、AP、Acc@0.5 和 TPR@5%FPR；默认阈值必须只在 real 校准集上确定。

**system prompt：**

```text
You are an image-forensics assistant. Assess only visible pixel evidence in the
provided image. Do not use filenames, metadata, expected object categories,
or dataset knowledge. An unusual subject, artistic style, compression, blur,
or low image quality alone is not evidence of AI editing.
```

**user prompt（图像与以下文本同一条 user message）：**

```text
Inspect this single image for a localized AI-generated or AI-assisted edit.
Look for visual evidence such as inconsistent illumination, shadows, texture,
edges, perspective, occlusion, geometry, or semantic integration. Do not
assume that a particular object is suspicious merely because it is present.

Return exactly one JSON object and no Markdown:
{
  "reasoning": "<detailed explanation>",
  "decision": "edited" | "not_edited",
  "p_ai_edited": <integer 0-100>,
  "evidence": [<at most 3 short visible-evidence statements>]
}

The probability must express your own likelihood that this image contains a
localized AI-generated or AI-assisted alteration.
```

要求模型输出 `reasoning` 作为可审计解释；`evidence` 是可公开保存的简短观察。

### 协议 B：`mllm_localization_v1`（T1 辅助 + T2，候选区域定位）

**用途：** 产生 T2 的候选框。此调用的 `p_ai_edited` 只作一致性/敏感性分析，主 T1 使用协议 A 的分数。

**system prompt：** 与协议 A 完全相同。

**user prompt：**

```text
Independently inspect this single image for localized visual evidence of an
AI-generated or AI-assisted edit. If one or more regions are suspicious, mark
only the smallest region(s) needed to cover the suspected alteration. Use a
1000 by 1000 coordinate canvas: [x1, y1, x2, y2], where 0 <= x1 < x2 <= 1000
and 0 <= y1 < y2 <= 1000. Coordinates are normalized to the full image, not a
crop. Do not invent a region when there is no visual evidence.

Return exactly one JSON object and no Markdown:
{
  "reasoning": "<detailed explanation>",
  "decision": "localized_edit" | "no_localized_edit",
  "p_ai_edited": <integer 0-100>,
  "regions": [
    {
      "bbox_1000": [<x1>, <y1>, <x2>, <y2>],
      "confidence": <integer 0-100>,
      "evidence": "<short visible-evidence statement>"
    }
  ]
}

Use an empty regions array when no specific region can be supported by visible
evidence. Return no more than 3 regions, ordered by confidence.
```

**bbox adapter：** 将标准化框映射到原始图像尺寸，裁剪到边界后取所有框的并集，栅格化为二值 mask。缺框、非法框或空框记为 `no_valid_region`，T2 预测为空 mask。结果表与论文必须标注：这是语言模型框输出的后处理，**不是原生像素 mask**。

---

## 4. 调用、重试与可复现实验规则

### 4.1 固定推理参数与三次成功采样

```text
temperature = 0
top_p = 1               # 仅当网关支持时发送
max_tokens = 600
enable_thinking = false    # 仅用于 Qwen；GPT/Gemini 不发送该字段
```

每个 `(model, image_id, protocol_id, condition)` 需要 **3 次独立、可解析的成功回答**。协议 A 与协议 B 各自独立采样三次，即每张图、每个模型共 6 个成功视觉请求；不把协议 A 的回答传给协议 B，也不在任一协议内传入前一次回答。

- **协议 A 聚合：** `decision` 用三票多数；`p_ai_edited` 用三次概率的中位数；`reasoning`/`evidence` 保存为三条 replicate 的原样数组，不由 runner 改写。
- **协议 B 聚合：** 以 `decision == localized_edit` 且至少一个合法 bbox 记为定位正票。至少两正票才构造预测区域；将三次 bbox 按同类区域的 IoU >= 0.10 聚类，仅保留至少两次支持的 cluster，并对每个 cluster 的坐标取中位数。无两票支持的区域输出空 mask。`p_ai_edited` 同样取中位数。
- 协议 A 为 `edited` / `not_edited` 二分类；三次成功采样天然有多数票。若后续协议再次引入第三类，必须同步更新 schema、聚合规则与 metrics 过滤规则。
- 不得因“判真”“低置信度”“空框”重试、剔除或挑选最好结果。

### 4.2 仅限操作性重试

```text
max_retries_per_replicate = 5  # 初始请求 + 最多 5 次操作性重试 = 最多 6 次尝试
retryable = timeout, connection error, HTTP 429, HTTP 5xx, invalid JSON/schema
backoff = 2s, 4s, 8s, 16s, 32s + uniform jitter(0, 1s)
```

- JSON/schema 无效时，下一次重试发送同一图与同一协议，追加一句：`Your previous response was not valid for the required JSON schema. Output only one valid JSON object.`
- 拒绝、安全拦截、HTTP 4xx（429 除外）与五次操作性重试耗尽均记为 `error`，保留原始响应/错误；绝不猜一个结果补齐。
- 429 读取 `Retry-After` 时优先服从该值；支持每模型并发数配置（默认 1），避免并发引入速率差异。
- 单个 replicate 在 6 次尝试内无法成功时，不能用 1 或 2 个成功回答凑多数。该 `(image, protocol)` 记为 `incomplete_replicates`，不进入对应主指标，且在 coverage 中报告。
- 三次 successful replicate 本身就是稳定性审计；附录额外报告 vote agreement、概率标准差及 bbox cluster 一致性，不再另跑一套重复子样本。

---

## 5. 配置与 OpenAI 兼容调用方式

实现复用 `/Users/muyouzi/workplace/safety-query/query-tag/risk-classifier/server.py` 的约定：JSON 配置、`${ENV_VAR}` 展开、`Authorization: Bearer`、`provider.apiBase + "/chat/completions"`、`provider.extraBody` 原样透传、`requests.post(..., timeout=...)`。不在仓库提交密钥或含密钥的 config。

建议单独放置本地 config，例如 `config/mllm_eval.local.json`（应被 `.gitignore` 忽略）。若所有模型走同一个网关，推荐把 provider 写成共享配置；当前 loader 也兼容“第一个模型带 provider，其余模型继承”的写法：

```json
{
  "api": {"timeout": 120},
  "provider": {
    "apiKey": "${QWEN_API_KEY}",
    "apiBase": "https://llm-chat-api.alibaba-inc.com/openai",
    "extraBody": {
      "app": "${QWEN_APP}",
      "quota_id": "${QWEN_QUOTA_ID}",
      "user_id": "${QWEN_USER_ID}",
      "access_key": "${QWEN_ACCESS_KEY}"
    }
  },
  "models": [
    {
      "id": "qwen3.7-plus", "slug": "qwen3_7_plus", "temperature": 0,
      "maxTokens": 600, "concurrency": 1,
      "extraBody": {"enable_thinking": false}
    },
    {
      "id": "<gpt-vision-model-id>", "slug": "gpt", "temperature": 0,
      "maxTokens": 600, "concurrency": 1
    },
    {
      "id": "<gemini-vision-model-id>", "slug": "gemini", "temperature": 0,
      "maxTokens": 600, "concurrency": 1,
      "requestFormat": "gemini_httpstream",
      "extraBody": {"tag": "web_chat_client", "category": "问答"},
      "geminiParams": {
        "use_gemini_httpstream_api": "1",
        "includeThoughts": "false",
        "thinkingBudget": 0,
        "responseMimeType": "application/json"
      }
    }
  ],
  "retry": {"maxRetriesPerReplicate": 5, "baseBackoffSeconds": [2, 4, 8, 16, 32]},
  "image": {"transport": "base64|url", "maxLongSide": 2048, "detail": "high"}
}
```

请求采用 OpenAI 风格的多模态 message：文字 prompt 与外链 URL 或 `data:<mime>;base64,...` 图像 URL 放在同一 user message 的 `content` 数组中。将该编码封装为唯一的 `OpenAICompatibleVisionClient`；若用户网关要求不同 image part schema，只替换该 client，而不改输入、prompt、解析、重试或结果模块。

---

## 6. 已实现框架

```text
eval/
  mllm/
    run_mllm.py                 # CLI：选择模型、输入、协议、condition、resume
    config.py                   # 与 risk-classifier 相同的 config/env 展开
    client.py                   # OpenAI-compatible multimodal POST + 限速 + retry
    inputs.py                   # review-export 与 generic JSONL list 两种 source adapter
    prompts.py                  # 本文档的 v1 prompt 常量及版本号
    schema.py                   # 严格 JSON 解析、bbox 校验、1000->pixel/mask adapter
    results.py                  # 三次 replicate、聚合 JSONL、断点续跑、原子写入
  prompts/
    mllm_protocol_v1.md         # 与本文档完全一致的冻结 prompt 文本
results/
  mllm/<model_slug>/<run_id>.jsonl
  mllm/<model_slug>/<run_id>.raw.jsonl
  mllm/<model_slug>/<run_id>.run_manifest.json
  mllm/<model_slug>/metrics/<run_id>/
    detection_per_image.jsonl       # detection 逐图预测、GT 与正误
    detection_metrics.{json,csv}    # detection 汇总表
    localization_per_image.jsonl    # localization 逐图框、GT 与两种命中判定
    localization_metrics.{json,csv} # localization 汇总表
```

另提供无密钥模板：`config/mllm_eval.example.json`。本地真实配置请写入 `config/mllm_eval.local.json`；该路径已加入 `.gitignore`。

CLI 预期形态：

```bash
python -m eval.mllm.run_mllm \
  --config /secure/path/mllm_eval.local.json \
  --source review-export \
  --review-export claimforge_generation_review_labels.json \
  --review-status good --include-source-pairs \
  --protocol both --replicates 3 --condition pilot_good275 \
  --run-id qwen37plus_pilot_good275_v2_20260714 \
  --model-slug qwen3_7_plus --concurrency 10 --resume --write-metrics

python -m eval.mllm.run_mllm \
  --config /secure/path/mllm_eval.local.json \
  --source list --list /path/images.jsonl \
  --protocol detection --condition custom_list --resume

# 对已经完成的一个模型结果单独补跑评估（不调用模型）
python -m eval.mllm.metrics \
  --results results/mllm/qwen3_7_plus/qwen37plus_pilot_good275_v2_20260714.jsonl \
  --review-export claimforge_generation_review_labels.json \
  --review-status good \
  --protocol-version mllm_protocol_v2_reasoning \
  --output-dir results/mllm/qwen3_7_plus/metrics/qwen37plus_pilot_good275_v2_20260714
```

`--run-id` 是每次运行的必填、全局可读标识，只能含字母、数字、`.`、`_`、`-`。它决定聚合 JSONL、raw JSONL、run manifest 和 metrics 目录名，也写入每条 raw/聚合/逐图评测记录。`--resume` 仅在**同一 run_id 的文件内**恢复：runner 默认按当前 `protocol_version` 内的 `(model, run_id, id, protocol_key, replicate_index)` 跳过已有 `status == "ok"` 的 raw replicate，并按当前 `protocol_version` 内的 `(id, protocol_key)` 跳过已有成功聚合记录。仅在同一单元的三次 replicate 全部成功后才生成 `status == "ok", valid_for_metrics=true` 的聚合记录；若不足三次成功，写入 `incomplete_replicates, valid_for_metrics=false` 以便 coverage 统计。

---

## 7. 结果 JSONL schema 与评测对接

`results/mllm/<model>/<condition>.jsonl` 保存每图每协议的**聚合记录**；同目录的 `<condition>.raw.jsonl` 保存三次 replicate 的原始请求结果。metrics 通过 `task_id`/`id` 回连 manifest，而不是把 GT 送给模型：

```json
{
  "schema_version": "mllm_result_v1",
  "run_id": "2026-07-11T120000Z_gpt_pilot_good275",
  "id": "restaurant_000_slot_001__forged",
  "task_id": "restaurant_000_slot_001",
  "image_path": "new_test/spliced_full/restaurant_000_slot_001.png",
  "image_sha256": "...",
  "image_size": [864, 576],
  "condition": "canonical_q95",
  "model": "<gpt-or-gemini-model-id>",
  "protocol_id": "mllm_localization_v1",
  "request_params": {"temperature": 0, "max_tokens": 600},
  "status": "ok",
  "valid_for_metrics": true,
  "replicate_count": 3,
  "successful_replicates": 3,
  "aggregation": {"decision": "majority", "probability": "median", "regions": "two-vote-iou-clusters"},
  "decision": "localized_edit",
  "p_ai_edited": 71,
  "score": 0.71,
  "regions_1000": [{"bbox_1000": [331, 620, 412, 704], "confidence": 68, "evidence": "..."}],
  "regions_px": [[286, 357, 356, 405]],
  "mask_path": "results/mllm/qwen3_7_plus/masks/<id>.png",
  "replicate_refs": ["<raw-jsonl-line-id-1>", "<raw-jsonl-line-id-2>", "<raw-jsonl-line-id-3>"],
  "parse_warnings": []
}
```

- 协议 A 的 `score = p_ai_edited / 100` 是 MLLM 的官方 T1 分数。
- 协议 B 的 bbox union mask 用于 forged 图的 pixel Precision、Recall、F1、IoU、MCC 和辅助 box-hit；空框为全零 mask。主 T2 是与 decoded source RGB / lossless spliced PNG 的 canonical exact-diff GT 比较得到的 macro pixel IoU，`edit_region_xyxy` 只用于辅助 box 指标。MLLM 没有连续像素分数，因此 pixel AP 记为不适用。
- 对 refusal/error/`incomplete_replicates` 单列 coverage 与 error reason；主指标预先规定为：错误不替换、不重标，按三次均成功的样本计算并同时报告覆盖率。若覆盖率低于 99%，主表脚注须列出成功数/总数。
- raw JSONL 保存 `raw_response`、prompt 版本、请求参数、输入 hash、重试历史与每次 latency；不得只保存最终分数。
- `--write-metrics` 在每个模型运行结束后自动回连本地 review export，输出两套完全分离的表。该 GT join 只发生在所有模型调用完成后，不会发送给模型。
- 每次运行额外写入 `<run_id>.run_manifest.json`。其中包含不含密钥的模型 ID、protocol/prompt version、输入清单 hash、输入筛选条件、推理参数、重试、图像传输与输出路径；`detection_metrics.csv` 和 `localization_metrics.csv` 也重复这些运行配置列，便于跨 run 汇总。
- `--concurrency N` 覆盖所选模型本次 run 的请求并发数（默认使用 config 中该模型的 `concurrency`，当前为 1）。每个 replicate 仍独立请求、独立重试；raw JSONL 由主线程按完成顺序追加，故文件行序不代表图像顺序。并发值会写入 run manifest 和两份 metrics CSV。
- 需要完整 coverage 时传入 `--retry-until-complete`。初始批次结束后，runner 仅将最终失败的 replicate 放入 recovery wave；每一 wave 仍先执行单个 replicate 的最多 5 次操作性重试。全部 replicate 成功之前不会写聚合 JSONL 或调用 `--write-metrics`。`--recovery-backoff-seconds` 控制 wave 间初始等待（默认 10 秒，逐 wave 线性增长且最多 60 秒）。
- **detection 表：** 对 550 张配对图均按 `edited` / `not_edited` 统计 TP、TN、FP、FN、Accuracy、Precision、Recall/TPR、Specificity/TNR、FPR、F1、Balanced Accuracy、AUROC、AP 以及覆盖率。仅 `valid_for_metrics=true` 的三次成功聚合记录进入主指标。
- **localization 表：** 主 T2 对 forged 图使用 source/spliced 的实际像素变化 mask，报告 macro/micro pixel Precision、Recall、F1、IoU 和 MCC，其中 macro pixel IoU 为主结果。`edit_region_xyxy` 仅用于辅助 box-hit、best-box IoU 阈值和 overlap；真图使用全零像素 GT，单独报告预测正像素比例以及 `no_localized_edit` 且无预测框的拒绝正确率。

---

## 8. 执行顺序与验收门槛

1. **接口 smoke test：** 每模型 10 real + 10 forged，两个协议、每协议三次成功采样，目标共 120 个成功调用；检查外链 URL 与 base64 两种输入、JSON 成功率、置信度范围、bbox 合法率、majority aggregation 与结果可恢复。
2. **冻结配置与 prompt：** 将模型 ID、网关 extra body、prompt version、timeout、并发、输入清单 hash 写入 run manifest；之后不改 prompt 重跑同一主表。
3. **pilot 主运行：** 当前 275 forged + 275 paired real，逐模型运行协议 A + B；每个单元三次成功采样，先产出 raw 与聚合 JSONL，再运行统一 `metrics.py`。
4. **完整 E1：** D2 canonical v1 就绪后，用同一冻结协议跑完整 paired 集。
5. **E2 laundering：** 按既定分层子样本，每条件只复用冻结协议与同一输入规则；不得为某个条件调 prompt。

验收：JSON/schema 成功率 >= 99%，每条 `ok` 结果可重放到相同 image hash、所有 bbox 可转换、`score` 在 [0,1]、结果目录可由 metrics 无人工清洗读取。

---

## 9. 已确认与尚缺配置

**已确认：** GPT + Gemini；`status=good` 的 275 forged 与 275 paired real pilot；每图两种协议、每协议 3 个成功 replicate 多数/中位数聚合；每个 replicate 最多 5 次操作性重试；结果存入 `results/mllm/<model>/<condition>.jsonl`。

**正式发起模型调用前仍需你提供：** 两个模型各自的实际 API model ID、`apiBase`、密钥环境变量名/鉴权方式、是否需要网关 `extraBody`，以及 pilot 优先使用 `base64` 还是供应商可访问的 HTTPS 图像 URL。当前框架已经可做 dry-run/smoke test；不会在缺少本地 config 的情况下调用任何 LLM。
