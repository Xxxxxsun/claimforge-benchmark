# 商业图片检测 API 可用性与替代计划（2026-07-20）

本文记录 CLAIMFORGE 商业图片检测服务的当前可用性、接入方式和实验优先级。它更新并取代 `survey_commercial_mllm_2026-07-09.md` 中关于 Illuminarty 仍可运行以及商业基线排序的结论。所有已保存结果及 completed/partial/not-run 状态的统一入口见 `results/commercial/README.md`。

## 1. 当前结论

- **Illuminarty 停止使用。** 官方 Webapp 当前显示 `Service currently not available: Cannot connect to server`；已有 key 也无法完成有效推理。无法仅凭这些现象判断团队状况，但从实验执行角度应将其标为 unavailable，不再购买额度或开发新 adapter。官方状态：https://app.illuminarty.ai/
- **Sightengine 保留。** 2026-07-20 已在 99 张 good-mouse forged PNG 上获得 99/99 个有效响应；详细结果见 `SIGHTENGINE_MOUSE_PILOT_RESULTS_2026-07-20.md`。该批仍是 forged-only pilot，不是 canonical paired 主表结果。
- **Hive 已完成 authenticated paired preflight。** 2026-07-20 在 5 对 good-mouse `real + forged` 上获得 10/10 个有效响应；厂商阈值 0.9 下 real 与 forged 均为 0/5 检出。Hive 提供较强的论文可比性，现保留为核心 whole-image 商业基线。
- **Resemble Detect 已完成 authenticated paired preflight。** 5 对 mouse 输入获得 10/10 个有效分类和 10/10 个 IFL heatmap artifact，但仅 1/5 forged 被标为 `Likely fake`，且当前返回的可视化没有形成可用的 mouse 局部定位信号。
- **Alibaba Cloud Ultra 已完成全量 forged-only 运行。** 国内版北京地域的 `aigcDetector_ultra` 已通过鉴权和本地临时上传验证；275/275 请求有效，30 张命中 `risk_edit`，另 1 张命中 `risk_fake`，任一风险检出率为 31/275（11.27%）。
- **AI or Not 已完成全量 forged-only 运行。** 275/275 请求有效，仅 4/275（1.45%）被厂商判为 AI；271/275（98.55%）未检出。5 对 paired pilot 中 real 与 forged 均为 0/5 检出，且成对分数几乎不变。
- **Copyleaks Ultra 已完成前 102 张 forged。** 102/102 有效，38/102（37.25%）被判为 AI。原生 RLE 在 38 个命中样本上平均 precision 0.9804、recall 0.8433、IoU 0.8266；把 64 个空 mask 漏检计入后，全体 mean IoU 为 0.3080。当前是 275 张计划的可断点前缀，尚余 173 张。
- **Reality Defender 仅做 coverage pilot。** 它是成熟商业服务，但无脸或脸过小的图片可能返回 `NOT_APPLICABLE`，与 restaurant/lodging 数据存在明显适配风险。

## 2. “当前可用”的验证边界

本次服务盘点主要使用官方文档、官方价格页和无凭据连通性探测；Copyleaks、Hive、Resemble、Alibaba、AI or Not 与 Sightengine 另已完成带有效凭据的推理验证：

| Service | 2026-07-20 无凭据探测 | 说明 |
|---|---|---|
| Copyleaks Ultra | authenticated 102/102 unique mouse forged 请求有效 | production `ai-image-1-ultra`、multipart PNG、离散 verdict、AI-pixel fraction 和原生 RLE mask 均已验证；38/102 正判 |
| AI or Not | authenticated 275/275 forged 请求有效 | `only=ai_generated` 本地上传、连续分数和厂商 verdict 均已验证 |
| Hive V3 | authenticated HTTP 200，10/10 mouse paired 请求有效 | API、凭据和 multipart 图片推理均已验证 |
| Resemble Detect | authenticated HTTP 200，10/10 mouse paired 请求有效 | 分类、IFL score 和 heatmap artifact 均已验证 |
| Reality Defender | HTTP 401 | 正式 endpoint 在线并执行鉴权 |
| Sensity | HTTP 401 | 正式 endpoint 在线并执行鉴权 |
| Alibaba Cloud | authenticated 275/275 forged 请求有效 | 国内版 `cn-beijing` / `aigcDetector_ultra`、RAM 签名、本地临时上传及局部编辑标签均已验证 |

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

**2026-07-20 mouse paired preflight：**

- 样本：按 Sightengine pilot 的固定顺序选取前 5 个审核为 `good` 的 mouse task，共 5 张真实原图和 5 张对应拼回图。
- 输入控制：两类图片均先解码为 RGB，再编码为无元数据 JPEG quality 95、4:4:4 subsampling，并统一以 `image.jpg` 上传。
- API coverage：10/10 HTTP 200，10/10 返回有效 `ai_generated` 分数。
- 厂商阈值 0.9：real 0/5 检出，forged 0/5 检出。
- forged 分数：范围 0.0000123–0.0026269，均值 0.0007090，中位数 0.0001605。
- real 分数：范围 0.0000116–0.0026285，均值 0.0007127，中位数 0.0001621。
- paired `forged - real`：均值 -0.00000371；这 5 对中编辑前后的输出几乎不变。`hunyuan`、`sdxlinpaint` 与 `stablediffusioninpaint` 子类分数同样接近零。
- 解释边界：该结果只证明当前 5 对局部 mouse 编辑构成了明确失败案例，不足以替代 275 对主实验或估计稳定的总体检测率。
- Runner：`eval/commercial/run_hive.py`。
- 结果：`results/commercial/hive/pilot_good_mouse_pairs5_canonical_jpeg_q95_20260720.jsonl`、对应 `.run_manifest.json` 和 `.summary.json`。

**275 张 mouse forged-only 批跑状态：** 2026-07-20 已完成 88/275 个有效请求，0/88 达到 0.9 阈值；最高分 0.0095834，均值 0.0006725。第 89 个待测输入触发 HTTP 429，与 V3 self-serve 100 requests/day 限额一致；剩余 187 张保留为断点任务。当前结果位于 `results/commercial/hive/good275_mouse_forged_canonical_jpeg_q95_20260720.jsonl`，再次运行同一命令会跳过这 88 个成功 ID。

### 3.2 Resemble Detect — 已验证但定位能力较弱

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

**2026-07-20 mouse paired preflight：**

- 样本与输入控制：使用 Hive pilot 相同的 5 对 `real + forged`，两类图片统一转为无元数据 JPEG quality 95、4:4:4 subsampling。
- API coverage：10/10 HTTP 200；10/10 返回 `image_metrics.score`、`ifl.score`、heatmap 和 visualization artifact。
- real 标签：`Real` 3 张、`Likely real` 1 张、`Neutral/Uncertain` 1 张；整图 score 均值 0.2049。
- forged 标签：`Real` 3 张、`Likely real` 1 张、`Likely fake` 1 张；整图 score 均值 0.2403。
- paired 整图 score `forged - real`：均值 0.0354，中位数 0.0173；最大增量 0.1055 来自唯一的 `Likely fake` 样本。
- paired IFL score `forged - real`：均值 0.00989，中位数 0.00084。该小样本中整体变化很弱。
- 定位检查：4/5 forged 的 heatmap 与 visualization 在解码后像素完全相同，没有可见热区；唯一 `Likely fake` 样本的 visualization 是全图红色覆盖，GT mouse 框内的覆盖变化没有高于框外。`ifl.heatmap` 是厂商渲染后的 RGB/JPEG artifact，不是带有明确数值语义的单通道 score map，因此不能直接当作原生编辑 mask 计算 pixel-AP。
- 隐私限制：当前账号套餐不支持 `zero_retention_mode`；带该参数的请求返回 HTTP 400。正式扩大样本前需要接受默认存储策略或联系 Resemble 开通该能力。
- Runner：`eval/commercial/run_resemble.py`。
- 结果：`results/commercial/resemble/pilot_good_mouse_pairs5_canonical_jpeg_q95_20260720.jsonl`、对应 `.run_manifest.json`、`.summary.json` 及同名 artifact 目录。

**275 张 mouse forged-only 批跑状态：** 已获得 274/275 个有效结果和 274/274 个 heatmap artifact；`Fake` 30、`Likely fake` 11，合并为厂商正判时是 41/274（14.96%）。最后一张因 wallet 余额比单次费用少 1 cent 而返回 HTTP 402，补充余额后可断点补跑。详细记录见 `RESEMBLE_MOUSE_FULL_RESULTS_2026-07-20.md`。

### 3.3 Alibaba Cloud Ultra — 已验证的 local-edit 专项基线

- API operation：`ImageModeration`
- 服务名：阿里云中国站 `aigcDetector_ultra`。
- 地域与 endpoint：`cn-beijing` / `green-cip.cn-beijing.aliyuncs.com`。
- 鉴权：Alibaba Cloud RAM AccessKey 签名；使用专用 RAM user 和厂商要求的 `AliyunYundunGreenWebFullAccess` policy，绝不使用 root AccessKey。
- 输入：公开 URL、OSS 对象或 SDK 本地文件上传；本地文件在服务端短暂保存 30 分钟。
- 输出标签：`risk_aigc`、`risk_fake`、`risk_edit`，命中标签的 confidence 为 0–100；未触发账号默认阈值时只返回无 confidence 的 `nonLabel`，因此当前输出不能直接用于 AUROC/AP。
- 定位：`risk_edit` 是整图风险分数，不返回像素 mask。
- 国内价格：人民币 200 元/万次；275 张约 5.50 元，550 张 paired 集约 11 元。

官方资料：

- https://help.aliyun.com/zh/document_detail/2672918.html
- https://help.aliyun.com/zh/document_detail/467828.html
- https://help.aliyun.com/zh/document_detail/477720.html

**实验角色：** 唯一明确把“AI local editing”写入官方检测目标的候选；即使 `risk_edit` 失败，也能形成直接可写的 vendor-claim stress test。

**2026-07-20 mouse 运行结果：**

- 单张 preflight：1/1 有效，返回 `nonLabel`，证明 RAM 权限、北京 Ultra service 和临时上传链路可用。
- paired pilot：与 Hive/Resemble 相同的前 5 对 canonical `real + forged` 共 10/10 有效；real 与 forged 均 0/5 命中任一风险标签。
- forged-only 全量：275/275 有效、0 错误；`risk_edit` 30/275（10.91%），另有 1/275 仅命中 `risk_fake`，任一风险共 31/275（11.27%）；`risk_aigc` 0/275。
- `risk_edit` 命中 confidence：均值 86.38，中位数 86.31，范围 70.58–99.47。风险等级为 high 13、medium 8、low 10、none 244。
- domain：lodging 的 `risk_edit` 为 11/147（7.48%）；restaurant 为 19/128（14.84%）。该切片差异仅作描述，尚未控制编辑面积等混杂因素。
- Runner：`eval/commercial/run_alibaba.py`。
- 结果：`results/commercial/alibaba/good275_mouse_forged_canonical_jpeg_q95_20260720.jsonl`、对应 `.run_manifest.json` 和 `.summary.json`。详细记录见 `ALIBABA_MOUSE_FULL_RESULTS_2026-07-20.md`。

### 3.4 AI or Not — 已验证的低检出率 whole-image 基线

- Endpoint：`POST https://api.aiornot.com/v2/image/sync`
- 鉴权：`Authorization: Bearer <API_KEY>`
- 输入：multipart 字段 `image`；保守按 10 MB 上限，支持 PNG/JPEG/WebP/HEIC/HEIF/TIFF。
- 调用参数：使用 `only=ai_generated`。默认请求还会运行 deepfake、NSFW 和 quality，其中 AI-generated 与 deepfake 分别计费。
- 输出：`ai_generated.ai` / `ai_generated.human` 下的 `is_detected` 与 `confidence`、生成器归因和图片元数据。解析器必须容忍未来新增生成器字段。
- 定位：`ai_generated` 没有 ROI；只有 face-deepfake 分支返回 bbox，不能当作通用 local-edit 定位。
- 价格：当前免费页列 20 次 image checks 和 API key；Pro 为 USD 5/月并列出 500 image checks。实际调用只启用 `ai_generated`，避免 deepfake 等报告的独立计费。

官方资料：

- https://docs.aiornot.com/api-reference/reports-by-modality/image
- https://docs.aiornot.com/setup
- https://www.aiornot.com/pricing
- https://www.aiornot.com/register

**实验角色：** 低成本 whole-image 连续分数基线；没有通用 edit localization。

**2026-07-20 mouse 运行结果：**

- paired pilot：与其他商业 API 相同的前 5 对 canonical `real + forged`，10/10 有效；real 与 forged 均 0/5 被判为 AI。
- pilot real 分数均值 0.04957，forged 均值 0.04949；paired `forged - real` 均值 -0.0000818。5 对分数几乎完全相同。
- forged-only 全量：275/275 有效、0 错误；4/275（1.45%）`ai_detected=true`，271/275（98.55%）未检出。
- forged `ai_confidence`：均值 0.03982，中位数 0.01462，P95 0.13041，范围 0.00173–0.90252。4 个正判 confidence 为 0.59555、0.83564、0.89352、0.90252；最高负判为 0.32095。
- domain：4 个正判全部来自 lodging，即 lodging 4/147（2.72%）、restaurant 0/128。该结果只作描述，不足以推断稳定领域差异。
- Runner：`eval/commercial/run_aiornot.py`。
- 结果：`results/commercial/aiornot/good275_mouse_forged_canonical_jpeg_q95_20260720.jsonl`、对应 `.run_manifest.json` 和 `.summary.json`。详细记录见 `AIORNOT_MOUSE_FULL_RESULTS_2026-07-20.md`。

### 3.5 Copyleaks AI Image Detection Ultra — 已验证的原生定位基线

- Endpoint：`POST https://api.copyleaks.com/v1/ai-image-detector/{scanId}/check`。
- 鉴权：先用账户邮箱和永久 API key 调用 `POST https://id.copyleaks.com/v3/account/login/api`，再使用 48 小时 Bearer token。
- 模型：`ai-image-1-ultra`；production 请求必须显式使用 `sandbox=false`，否则只返回不计费的 mock result。
- 输入：multipart PNG/JPEG 等；最小 512x512，最大 6000x4500 / 27 MP，文件小于 32 MB。
- 输出：`isAiDetected`、AI/Human 像素比例，以及由 `starts` / `lengths` 构成的像素级 RLE mask。
- 计费：当前实测 1 credit/image；可用 `GET /v3/scans/credits` 查询余额。公开文档未给出足以稳定换算的单张美元价格，因此预算以 credits 报告。

官方资料：

- https://docs.copyleaks.com/guides/ai-detector/ai-image-detection/
- https://docs.copyleaks.com/reference/actions/ai-image-detector/check
- https://docs.copyleaks.com/using-the-apis/authentication/
- https://docs.copyleaks.com/reference/actions/admin/check-credits/

**2026-07-20 mouse 结果：** 固定顺序前两对共 4/4 有效；real 0/2 正判，forged 2/2 正判。随后优先扩跑 100 张未测 forged，最终得到前 102 张 unique forged 的完整结果：38/102（37.25%）正判，lodging 21/54、restaurant 17/48。命中的 38 张上，RLE 对 canonical real-vs-forged 精确像素差分 mask 的平均 precision 0.9804、recall 0.8433、IoU 0.8266，且预测像素全部位于 context box 内；若把 64 个漏检的空 mask 纳入，mean IoU 为 0.3080、median 为 0。该结果说明“命中后定位很准”但不能掩盖 62.75% image-level miss rate。

- Runner：`eval/commercial/run_copyleaks.py`。
- 详细记录：`COPYLEAKS_MOUSE_PILOT_RESULTS_2026-07-20.md`、`COPYLEAKS_MOUSE_FORGED_102_RESULTS_2026-07-20.md`。
- 当前 forged-only 进度：102/275，断点文件已验证还剩 173 张；按实测 1 credit/image 还需 173 credits。

### 3.6 Reality Defender — 知名服务但低优先级

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

默认每家先跑相同的 5 对 `real + forged`，且 real/forged 使用同一 canonical 编码；Alibaba 先用 1 张确认 service code，Copyleaks 当前完成前 2 对和 forged-only 前 102 张，Reality Defender 先用 5–10 张测 coverage。检查：

1. HTTP/API 成功率与有效 coverage。
2. 原始连续分数、厂商 verdict、版本字段和计费单位。
3. 是否回显文件名、任务 ID 或凭据相关字段。
4. Resemble heatmap 是否能下载、是否与输入尺寸对齐、是否对 mouse 区域有响应。
5. Alibaba 是否实际返回 `risk_edit`，以及 Ultra service code 与区域是否可用。
6. Copyleaks RLE 与精确像素差分 GT 的 IoU/precision/recall，以及 `summary.ai` 是否与 RLE 面积一致。
7. Reality Defender 在无脸图片上的 `NOT_APPLICABLE` 比例。

### Phase B：核心商业主表

1. Sightengine `genai`
2. Hive AI-generated image + deepfake classifier
3. Copyleaks `ai-image-1-ultra`（T1 + 原生 T2）
4. Resemble Detect（T1；当前 heatmap 不作为 T2）

Alibaba Ultra 与 AI or Not 已完成 mouse forged-only 全量，作为已验证补充行保留。Copyleaks 在补充 credits 后优先扩跑；Reality Defender 只有在有效 coverage 足够时进入主表，否则作为 coverage/failure appendix。

### Phase C：统一报告规则

- 保存原始连续 score，不只保存离散 verdict。
- 同时报厂商推荐阈值和统一 benchmark 指标；不能把 vendor confidence 默认解释为校准概率。
- paired real/forged 必须使用完全相同的编码与元数据处理，避免 PNG-vs-JPEG 捷径。
- 无效/`NOT_APPLICABLE` 单独报告 coverage，不静默丢弃，也不直接当预测错误混入主指标。
- 商业 heatmap 必须保留原文件、尺寸、颜色空间和 score-to-mask adapter；任何阈值化规则在看 GT 前固定。
- Copyleaks 原生二值 RLE 不做后验阈值搜索；按官方零基 row-major 示例解码，并同时报告与 exact-diff mask 和标注框的重合。
- 每条结果记录 endpoint/model、请求时间、延迟、原始响应、输入 SHA-256、计费字段和 adapter SHA-256。

## 6. 凭据与 runner 约定

建议 runner 只读取以下环境变量，不接受命令行明文 secret，也不把 secret 写入 manifest、日志或结果：

| Service | 环境变量 |
|---|---|
| Hive | `HIVE_API_KEY` |
| Resemble | `RESEMBLE_API_TOKEN` |
| Alibaba Cloud | `ALIBABA_CLOUD_ACCESS_KEY_ID`, `ALIBABA_CLOUD_ACCESS_KEY_SECRET` |
| AI or Not | `AIORNOT_API_KEY` |
| Copyleaks | `COPYLEAKS_EMAIL`, `COPYLEAKS_API_KEY`（只用于换取短期 Bearer token） |
| Reality Defender | `REALITY_DEFENDER_API_KEY` |
| Sensity | `SENSITY_USERNAME`, `SENSITY_PASSWORD`（只用于换取短期 Bearer token） |

凭据不得提交到 Git。若凭据曾在聊天、终端历史或日志中明文出现，应在实验结束后轮换。
