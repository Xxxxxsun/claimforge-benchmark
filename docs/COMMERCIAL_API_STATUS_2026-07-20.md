# 商业图片检测 API 可用性与替代计划（2026-07-20）

本文记录 CLAIMFORGE 商业图片检测服务的当前可用性、接入方式和实验优先级。它更新并取代 `survey_commercial_mllm_2026-07-09.md` 中关于 Illuminarty 仍可运行以及商业基线排序的结论。

## 1. 当前结论

- **Illuminarty 停止使用。** 官方 Webapp 当前显示 `Service currently not available: Cannot connect to server`；已有 key 也无法完成有效推理。无法仅凭这些现象判断团队状况，但从实验执行角度应将其标为 unavailable，不再购买额度或开发新 adapter。官方状态：https://app.illuminarty.ai/
- **Sightengine 保留。** 2026-07-20 已在 99 张 good-mouse forged PNG 上获得 99/99 个有效响应；详细结果见 `SIGHTENGINE_MOUSE_PILOT_RESULTS_2026-07-20.md`。该批仍是 forged-only pilot，不是 canonical paired 主表结果。
- **新增首选：Hive 与 Resemble Detect。** Hive 提供最强的论文可比性；Resemble 是当前最接近 Illuminarty 定位能力的替代项。
- **新增专项候选：Alibaba Cloud AI-Generated Image Detection Ultra。** 官方明确返回 AI local-edit 风险标签，但正式批跑前必须先确认账号区域和实际 service code。
- **新增低成本补充：AI or Not。** 自助、便宜、可直接上传本地图片，适合快速增加一个 whole-image 商业黑盒基线。
- **Reality Defender 仅做 coverage pilot。** 它是成熟商业服务，但无脸或脸过小的图片可能返回 `NOT_APPLICABLE`，与 restaurant/lodging 数据存在明显适配风险。

## 2. “当前可用”的验证边界

本次状态检查只使用官方文档、官方价格页和无凭据连通性探测，不创建账号、不消耗推理额度：

| Service | 2026-07-20 无凭据探测 | 说明 |
|---|---|---|
| AI or Not | health HTTP 200，`is_live=true`；image API HTTP 401 | 健康检查与鉴权路由在线 |
| Hive V3 | HTTP 401，`Authorization header missing` | 官方 V3 路由在线并拒绝无凭据请求 |
| Resemble Detect | HTTP 401 | 正式 endpoint 在线并执行鉴权 |
| Reality Defender | HTTP 401 | 正式 endpoint 在线并执行鉴权 |
| Sensity | HTTP 401 | 正式 endpoint 在线并执行鉴权 |
| Alibaba Cloud | 官方 endpoint 可达；无签名根路径返回 HTTP 404 | 服务域名在线；RAM 签名、服务开通状态和 region 仍须 authenticated smoke test 验证 |

HTTP 401 在无 key 探测中是预期结果，只能确认 DNS/TLS/路由/鉴权层当前存在，**不能替代带有效凭据的推理 smoke test**。每家服务在进入批量运行前仍须完成少量 paired preflight；正式纳入还必须保存至少一个 authenticated HTTP 200 原始响应。

## 3. 推荐服务

### 3.1 Hive AI — 论文商业基线首选

- V3 endpoint：`POST https://api.thehive.ai/api/v3/hive/ai-generated-and-deepfake-content-detection`
- 鉴权：`Authorization: Bearer <V3_SECRET_KEY>`
- 输入：单张 URL、base64 或 multipart 本地文件；支持 PNG/JPEG/WebP/GIF。
- 自助额度：默认 100 image requests/day，更高吞吐需要联系 Sales。
- 输出：整图 `ai_generated` / `not_ai_generated` 分数、生成器归因、C2PA 信息以及 deepfake 结果。生成器类别包括 `hunyuan`、`sdxlinpaint` 和 `stablediffusioninpaint`。
- 官方推荐 AI-image 阈值：0.9。实验中同时保留原始连续分数，最终 paired 集报告 AUROC/AP 与厂商阈值结果。
- 定位：没有通用 edit-region bbox、mask 或 heatmap；deepfake 分支的人脸 bbox 与 mouse T2 无关，不能当作编辑区域定位。
- 价格：USD 6 / 1,000 image requests；275 张约 USD 1.65，550 张 paired 集约 USD 3.30。默认日限额下分别至少需要 3 天和 6 天。

官方资料：

- https://docs.thehive.ai/docs/ai-generated-and-deepfake-content-detection-playground
- https://docs.thehive.ai/docs/ai-image-and-video-detection
- https://docs.thehive.ai/docs/getting-started
- https://docs.thehive.ai/reference/authentication
- https://thehive.ai/pricing

**实验角色：** 与 Sightengine 组成核心 whole-image 商业检测对，并与 INP-X 等已有工作直接比较。

### 3.2 Resemble Detect — Illuminarty 定位能力的首选替代

- Endpoint：`POST https://app.resemble.ai/api/v2/detect`
- 鉴权：`Authorization: Bearer <API_TOKEN>`
- 接入：自助 Flex pay-as-you-go；支持直接 multipart 文件、公开 URL 或 secure-upload token。
- 同步方式：请求头 `Prefer: wait`；也支持异步 UUID、轮询和 callback。
- 关键参数：`visualize=true` 请求可视化；`zero_retention_mode=true` 请求分析后删除媒体。
- 输出：still image 的 `image_metrics.label`、`image_metrics.score`、分析树，以及可选 `image_metrics.ifl.score` / `image_metrics.ifl.heatmap`。
- 定位：官方 schema 确实暴露图片 heatmap，是目前最值得与 GT object mask 做 overlap、pixel-AP 和 box-hit 试验的商业输出；必须先确认 heatmap 的空间含义和分辨率，不能在验证前称为原生编辑 mask。
- 价格：官网列 image detection 为 USD 0.04 / second。静态图片的最小计费单位没有写清，正式跑 275 张前要以少量调用后的账单为准。

官方资料：

- https://docs.resemble.ai/detect/create
- https://docs.resemble.ai/detect/get
- https://www.resemble.ai/products/detect
- https://www.resemble.ai/pricing

**实验角色：** 商业分类 + 候选定位输出；优先用 5–10 对图确认 heatmap 是否真正响应局部 mouse 编辑。

### 3.3 Alibaba Cloud Ultra — local-edit 专项候选

- API operation：`ImageModeration`
- 服务名：最新国际文档写作 `aigcDetector_ultra_global`。
- 鉴权：Alibaba Cloud RAM AccessKey 签名；使用专用 RAM user 和厂商要求的 `AliyunYundunGreenWebFullAccess` policy，绝不使用 root AccessKey。
- 输入：公开 URL、OSS 对象或 SDK 本地文件上传；本地文件在服务端短暂保存 30 分钟。
- 输出标签：`risk_aigc`、`risk_fake`、`risk_edit`，每项 confidence 为 0–100。
- 定位：`risk_edit` 是整图风险分数，不返回像素 mask。
- 价格：最新国际文档列 USD 0.60 / 1,000 calls；275 张约 USD 0.165，550 张约 USD 0.33。
- 阻断性注意：同一官方页面在 Ultra 的 service code、区域支持和计费分类上存在不一致。必须先用 1 张做 preflight，确认实际 service、region、返回标签和账单，再批跑。

官方资料：

- https://www.alibabacloud.com/help/en/content-moderation/latest/image-audit-enhanced-edition-detects-aigc-infringement
- https://www.alibabacloud.com/help/en/content-moderation/latest/billing-description

**实验角色：** 唯一明确把“AI local editing”写入官方检测目标的候选；即使 `risk_edit` 失败，也能形成直接可写的 vendor-claim stress test。

### 3.4 AI or Not — 最容易增加的低成本黑盒基线

- Endpoint：`POST https://api.aiornot.com/v2/image/sync`
- 鉴权：`Authorization: Bearer <API_KEY>`
- 输入：multipart 字段 `image`；保守按 10 MB 上限，支持 PNG/JPEG/WebP/HEIC/HEIF/TIFF。
- 调用参数：使用 `only=ai_generated`。默认请求还会运行 deepfake、NSFW 和 quality，其中 AI-generated 与 deepfake 分别计费。
- 输出：`ai_generated.ai` / `ai_generated.human` 下的 `is_detected` 与 `confidence`、生成器归因和图片元数据。解析器必须容忍未来新增生成器字段。
- 定位：`ai_generated` 没有 ROI；只有 face-deepfake 分支返回 bbox，不能当作通用 local-edit 定位。
- 价格：免费页列 20 次 image checks；其后 USD 0.02 / image。275 张约 USD 5.50，550 张约 USD 11。

官方资料：

- https://docs.aiornot.com/api-reference/reports-by-modality/image
- https://docs.aiornot.com/setup
- https://www.aiornot.com/pricing
- https://www.aiornot.com/register

**实验角色：** 先用免费 20 次运行 10 对 paired smoke；coverage 和 score 正常后再决定是否全量。

### 3.5 Reality Defender — 知名服务但低优先级

- 鉴权：`X-API-KEY`。
- 流程：请求 AWS presigned upload URL，上传本地文件，再按 `request_id` 轮询结果。
- 输出：ensemble authenticity/deepfake 状态、归一化分数和模型级结果；公开 schema 没有通用编辑区域 mask。
- 额度：Free 为每月 50 次 image/audio scans；适合先跑 5–10 对。
- 风险：官方结果文档列出图片在“无脸或脸太小”时可能返回 `NOT_APPLICABLE`。CLAIMFORGE 多数图片不是人脸任务，因此必须先量化有效 coverage，不能直接纳入主表分母。

官方资料：

- https://docs.realitydefender.com/api-reference/quickstart
- https://docs.realitydefender.com/api-reference/endpoint/get_media_detail
- https://www.realitydefender.com/product/realapi

## 4. 次级候选与明确排除

- **Sensity：** API 与鉴权层在线，支持 AI-generated image analysis 和部分 heatmap，但创建 developer account 必须联系销售；无公开单价。可并行申请 7-day trial，不进入当前 P0 自助执行清单。https://docs.sensity.ai/
- **Winston AI：** API 与文档在线，开发者账号提供 2,000 starter credits；basic image detection 为 300 credits/image，advanced 为 500 credits/image。图片端点只接受公开 URL，且 advanced forensic visualization 是否等价于编辑区域仍需验证，列为备选。https://docs.gowinston.ai/api-reference/introduction
- **GetReal Security：** 有企业 API 宣传但没有公开 endpoint/schema/价格；只作为 contact-sales 采购候选。https://www.getrealsecurity.com/
- **Google AI Content Detection：** 可检测生成或修改图片，但截至 2026-07-20 仍为 Private Preview，需要申请访问，不属于可立即运行的基线。https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/ai-content-detection
- **Optic：** 不再作为独立图片检测 API。当前 AI or Not 是独立服务；旧 `optic.xyz` 已转向 Bioptic 的药物发现业务。
- **Deepware：** 正式 API 面向视频，不适用于当前静态图片集。
- **Illuminarty：** unavailable；不再申请额度、不再依赖网页内部 localization endpoint。

## 5. 实际执行顺序

### Phase A：paired preflight

默认每家先跑相同的 5 对 `real + forged`，且 real/forged 使用同一 canonical 编码；Alibaba 先用 1 张确认 service code，Reality Defender 先用 5–10 张测 coverage。检查：

1. HTTP/API 成功率与有效 coverage。
2. 原始连续分数、厂商 verdict、版本字段和计费单位。
3. 是否回显文件名、任务 ID 或凭据相关字段。
4. Resemble heatmap 是否能下载、是否与输入尺寸对齐、是否对 mouse 区域有响应。
5. Alibaba 是否实际返回 `risk_edit`，以及 Ultra service code 与区域是否可用。
6. Reality Defender 在无脸图片上的 `NOT_APPLICABLE` 比例。

### Phase B：核心商业主表

1. Sightengine `genai`
2. Hive AI-generated image + deepfake classifier
3. Resemble Detect

若 Phase A 正常，再增加 Alibaba Ultra 与 AI or Not。Reality Defender 只有在有效 coverage 足够时进入主表，否则作为 coverage/failure appendix。

### Phase C：统一报告规则

- 保存原始连续 score，不只保存离散 verdict。
- 同时报厂商推荐阈值和统一 benchmark 指标；不能把 vendor confidence 默认解释为校准概率。
- paired real/forged 必须使用完全相同的编码与元数据处理，避免 PNG-vs-JPEG 捷径。
- 无效/`NOT_APPLICABLE` 单独报告 coverage，不静默丢弃，也不直接当预测错误混入主指标。
- 商业 heatmap 必须保留原文件、尺寸、颜色空间和 score-to-mask adapter；任何阈值化规则在看 GT 前固定。
- 每条结果记录 endpoint/model、请求时间、延迟、原始响应、输入 SHA-256、计费字段和 adapter SHA-256。

## 6. 凭据与 runner 约定

建议 runner 只读取以下环境变量，不接受命令行明文 secret，也不把 secret 写入 manifest、日志或结果：

| Service | 环境变量 |
|---|---|
| Hive | `HIVE_API_KEY` |
| Resemble | `RESEMBLE_API_TOKEN` |
| Alibaba Cloud | `ALIBABA_CLOUD_ACCESS_KEY_ID`, `ALIBABA_CLOUD_ACCESS_KEY_SECRET` |
| AI or Not | `AIORNOT_API_KEY` |
| Reality Defender | `REALITY_DEFENDER_API_KEY` |
| Sensity | `SENSITY_USERNAME`, `SENSITY_PASSWORD`（只用于换取短期 Bearer token） |

凭据不得提交到 Git。若凭据曾在聊天、终端历史或日志中明文出现，应在实验结束后轮换。
