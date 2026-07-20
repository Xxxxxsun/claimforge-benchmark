# CLAIMFORGE 实验设计与论文执行计划（AAAI-27 AISI track）

*定稿日期 2026-07-09。配套调研：`survey_opensource_detectors_2026-07-09.md`（开源方法，权重可用性已逐一验证）、`survey_commercial_mllm_2026-07-09.md`（商用 API + MLLM + 文献证据）。早期设计讨论见 `CLAIMFORGE_paper_design.md`，本文件取代其中的实验部分。*

*商用 API 状态更新（2026-07-20）：Illuminarty 当前不可用；active roster 包含 Sightengine、Hive、Copyleaks 与 Resemble，并对 Alibaba Ultra、AI or Not、Reality Defender 做分级执行。Copyleaks 已完成 forged-only 固定顺序前 102 张，38/102 正判，并验证了命中样本上的原生 RLE 定位。完整核验见 `COMMERCIAL_API_STATUS_2026-07-20.md`。原始时间线与 7/9 定稿内容保留作为计划快照。*

---

## 0. 一页总览

**论文一句话**：真实照片里局部插入一个 AI 生成的小物体（老鼠、蟑螂、污渍），拼回原图作为虚假消费索赔证据——我们构建首个该场景的检测基准 CLAIMFORGE，并系统评测四类现有检测手段（取证定位模型、整图 AIGC 检测器、商用 API、多模态大模型），证明在这种"像素保持局部编辑"威胁下没有任何现成方法同时做到可靠检测与定位。

**硬性时间**（AAAI-27，AISI 与主 track 同时间线，均 AoE）：
- **摘要注册：7 月 21 日**（还剩 12 天）
- **全文提交：7 月 28 日**（还剩 19 天）
- 补充材料/代码：7 月 31 日
- 页数：正文 7 页 + 参考文献至 9 页；双盲；Reproducibility Checklist 必交

**"解决"判据（写在论文里，跑实验前定死）**：某方法"解决"CLAIMFORGE 当且仅当在规范测试集上 image-level AUC ≥ 0.90 **且** pixel-level F1@0.5 ≥ 0.50，并且在 JPEG-75 再压缩后仍保持。预期结论：全部四个范式都远低于该线。

**实验三大块（保持简单，不绕）**：
- **E1 主实验**：~20 个检测器 × 两个任务（T1 检测 / T2 定位），零样本，一张主表。
- **E2 鲁棒性**：laundering（JPEG 质量、缩放、社交媒体模拟）下的退化曲线。
- **E3 诊断与对照**：整图生成对照集（同域整图 AI 图，检测器表现好 → 反衬局部编辑逃逸）、真实物体贴回对照、格式/捷径审计、物体面积-可检测性曲线、按域拆分。

可选加分项（P2，时间允许再做）：小规模人类实验、单模型 fine-tune 上限、第三个编辑器。

---

## 1. AISI track 契合点（写作时对着打分表）

AISI 评审按六项打分（问题重要性 / 跨学科文献 / 对 AI 社区的意义 / 严谨性 / 可复用性 / 社会影响落地），**不以纯技术新颖性为门槛**。对应写法：

1. **问题重要性**：照片是消费纠纷、平台仲裁、保险理赔的事实证据。局部 AI 编辑让"伪造证据"的门槛降为一句 prompt；受害方是小商家（餐厅被假蟑螂照勒索差评/退款）、房东/平台（假损坏索赔）、保险公司与诚实消费者（欺诈成本转嫁）。引 1-2 条业界/媒体已报道的 AI 图片退款欺诈案例 + 保险/平台欺诈的非 CS 文献（见 §8 写作任务）。
2. **跨学科文献**：保险欺诈检测（criminology/risk journals）、平台信任与虚假评论（hospitality/IS 期刊）、证据法里的图像证据可采性。related work 里单列一小段。
3. **对 AI 社区的意义**：新问题形式化（pixel-preserving 局部插入 + 索赔场景）+ 公开数据集 + 跨范式系统评测 = AISI 认可的 "data gathering + scientific insight" 型贡献。
4. **严谨性**：对照组、捷径审计、bootstrap 置信区间、AUC/AP 与固定阈值并报。
5. **可复用性**：公开数据集（HF gated）+ 全部评测代码 + 每个检测器的 adapter + 结果 JSONL。
6. **落地**：讨论段写清楚 trust & safety 团队如何用（当前商用工具在此威胁下的失效模式、应当要求的能力：局部定位 + 对小编辑敏感），以及 provenance（C2PA/SynthID）为何管不住这个威胁（攻击起点是真图、编辑器不打水印）。

---

## 2. 数据集定稿（D 系列任务）

### 现状（已验证）
- 594 个标注槽位：restaurant 297 + lodging 297；每槽 insert_box + context_box + 全分辨率 box mask。
- 插入区域极小：面积中位数 0.232% / 均值 0.363%（这是卖点数字，进摘要）。
- restaurant 已用 HunyuanImage-3 全量生成并 object-only 回贴（`new_test/spliced_full/`，297 张）；lodging 只有 22 张 pilot。
- **已核验 pixel-preserving 性质成立**：拼接图在 context box 之外与源图解码后逐像素一致（抽样 maxdiff=0），box 内也只有物体像素改变。
- **当前唯一物体类别是 mouse** —— 单类别有"检测到老鼠=假"的捷径风险，靠 D3+D4 缓解。
- **格式隐患**：伪图存 PNG、真图是 JPEG —— 若不统一，任何检测器都能靠格式作弊，这正是 "Fake or JPEG?" (arXiv:2403.17608) 批评的坑。D2 必须先做。

### D1（P0）补全 lodging 生成
用与 new_test 相同的 HunyuanImage-3 配置跑完 lodging 297 槽位（prompt 同模板），`compose_spliced_full.py --blend object` 回贴。产出后基准规模 = **594 forged + 594 real（同源配对）**。

### D2（P0）规范化发布格式（canonicalization）——最先做
- 统一管线：`real = decode(source.jpg) → encode JPEG q=95`；`fake = decode(source.jpg) + paste(object) → encode JPEG q=95`。两类经过**完全相同**的最终编码，EXIF 全部剥离，文件名随机化（不含 real/fake 线索）。
- GT 定位掩码两套都存：`object mask`（实际被替换像素，用 pre-encode diff>0 计算）与 `box mask`（insert_box 矩形）。主指标用 object mask，box 级命中率作次要指标（物体极小，pixel F1 很苛刻，box hit 更贴近"审核员能不能被指到位置"）。
- 输出 `benchmark_release/v1/{images,masks,manifest.jsonl}`；manifest 含 task_id、label、domain、editor、object、mask 路径、insert 面积占比。
- **旧商业 API 结果不可进入主表**：旧跑法是 PNG 伪图 vs JPEG 真图（格式混淆），论文数字必须来自规范化后的 paired v1 集。Sightengine 仍需在 canonical 数据上重跑；`SIGHTENGINE_MOUSE_PILOT_RESULTS_2026-07-20.md` 的 99/275 forged-only original-PNG pilot 只能作为失效现象与工程验证，不能计算主表所需的 AUC、FPR 或 paired accuracy。
- **Illuminarty 不再重跑**：7/9 曾验证可用，但 7/20 官方 Webapp 已显示 service unavailable，当前 API 不能作为可执行基线。保留历史 adapter 和失败记录用于 availability/provenance 说明，不再购买额度或依赖网页内部 localization 接口。

### D3（P1）第二编辑器 + 物体多样化
- 编辑器 2：**FLUX.1-Fill-dev**（开源、掩码原生 inpainting、diffusers 直接跑，与 Hunyuan 架构谱系不同）。备选/替换：Qwen-Image-Edit（本地或 DashScope API）。产出再 +594 forged。
- 借这一轮把物体多样化：按 task_id 确定性分配小分类表——restaurant: {mouse, cockroach, fly, hair, stain/mold}；lodging: {mouse, cockroach, bedbug, stain/mold, water-damage/crack}。这样跨全基准"物体在场"不再与标签完全相关，同时保留 per-object 分析维度。Hunyuan 全 mouse 集保留（editor×object 不完全交叉没关系，论文里说明即可）。
- 编辑器 3（P2，可选）：一个商用编辑器（gpt-image-1 或 Gemini 系图像编辑），补"商用编辑器生成的伪图"象限，各 100-200 张即可。

### D4（P1）对照组（三件套，防审稿人一击）
1. **整图生成对照集（contrast set）**：同域整图 AI 图 ~150 张（用 Hunyuan/FLUX 对同批场景做整图生成或高强度 img2img）。作用：证明"这些检测器不是不行，而是只对整图伪造行"——第一张结果图就是这个反差，故事一图讲完。
2. **真实物体贴回（real-object paste-back）**：~80-100 张，用真实老鼠/蟑螂/污渍抠图（Open Images 分割标注或 CC 图手工抠）走同一 object-only 贴回管线。作用：区分"检测到 AI 纹理"vs"检测到贴图痕迹"。
3. **真实含物照片（real-with-object）**：~80-100 张真实拍到害虫/污渍的照片（Open Images/Wikimedia 检索），标签为 real。作用：量化"物体在场捷径"——若某检测器在这组上 FPR 爆炸，说明它学的是语义不是取证。

### D5（P0）官方划分
按源图 60/20/20 切 train/val/test（同源图的所有变体不跨划分）。零样本主表报**全集**成绩（附 95% bootstrap CI）；fine-tune 实验（E4）只动 train/val，在 test 上对比。划分文件进 release。

---

## 3. 基线套件（定稿名单）

原则：每个范式 ≥3 个有代表性、**权重已验证可下载**的方法；总数 ~20 行主表。全部零样本（off-the-shelf 权重）。

### 家族 A：取证定位 IML（T1+T2，核心竞争者）— 8 个
经 IMDL-BenCo（pip `imdlbenco`，GDrive 权重）统一跑：**CAT-Net v2、MVSS-Net、PSCC-Net、IML-ViT、Mesorch**；原repo 跑：**TruFor**（Docker 一条命令，标准参照）、**SAFIRE**（AAAI'25，SAM-based，drop-folder 推理）、**RelayFormer**（ICLR'26，HF 权重，最新定位基线）。
（HiFi-Net 备选；RITA/GIMFormer/COCO-Inpaint 确认不可用或 Baidu-only，不排。）

### 家族 B：整图 AIGC 检测（仅 T1，预期接近随机——本身就是结果）— 5 个
**CNNDetection**（历史锚点）、**UniversalFakeDetect**（CLIP probe）、**NPR**（低层伪影）、**Effort**（ICML'25 Oral，当前泛化最强）、**Community Forensics**（CVPR'25，4803 个生成器训练，HF/MIT）。全部权重在 repo/HF/GDrive，推理都很轻。

### 家族 C：inpainting 专用（T1+T2）— 1 个
**MaskCLIP（OpenSDI, CVPR'25）**：唯一公开的、真正在扩散局部编辑上训练过的检测器（HF 权重）。它若也失败，"unsolved"主张最有力。需半天写 IMDLBenCo dataset-JSON adapter（与家族 A 复用）。

### 家族 D：MLLM — 4 个
- 零样本 prompt：**GPT-5.5**、**Gemini 3.1 Pro**（两家旗舰；已有文献证明上一代在局部编辑上接近随机且偏"真"，FragFake arXiv:2505.15644、arXiv:2506.10474）。
- 开源专用：**FakeShield**（ICLR'25，Apache-2.0，HF 22B 权重，verdict+mask+解释，需 40GB 级 GPU）。
- 开源通用（可选第 4 个）：Qwen 系旗舰 VL 零样本。
- 统一 prompt 协议：固定两问("这张照片是否被 AI 编辑过？给 0-100 置信度" / "若有，给出编辑区域 bounding box")，温度 0，每图 1 次；box→mask 作为**非原生定位 adapter** 明确标注。

### 家族 E：商用 API — 核心 4 个 + 分级 preflight
- **Sightengine genai（核心，T1）**：成熟 pixel-only 整图基线，2026-05 刚升级 AI-edit 检测。当前已有 99 张 forged-only pilot，但 canonical paired v1 仍待跑。未来主结果使用 `results/commercial/sightengine/` 下独立 run ID，不能与 pilot 混合。
- **Hive AI-generated image + deepfake classifier（核心，T1）**：文献可比性最强的新增商业基线；自助 V3、$6/1k、默认 100 requests/day，返回整图 AI/Human 分数和生成器归因，无定位。与 Sightengine 组成 INP-X 同款商用对。
- **Resemble Detect（核心候选，T1；T2 需 preflight）**：返回整图 fake/real 分数；官方 schema 在 `visualize=true` 时可返回 image heatmap。先用 5–10 对验证 heatmap 是否稳定、是否与输入空间对齐、是否真正响应局部编辑；验证失败则 T2 记 N/A，不能把可视化直接当 GT-compatible mask。
- **Copyleaks `ai-image-1-ultra`（核心，T1+T2）**：生产端点已完成前 102 张 unique mouse forged，38/102（37.25%）正判。命中的 38 张上，原生 RLE mask 对 SP 精确像素差分 GT 的平均 precision 0.9804、recall 0.8433、IoU 0.8266；计入空 mask 漏检后的全体 mean IoU 0.3080。断点还剩 173/275，不能把 conditional localization 质量写成整体定位性能。
- **Alibaba Cloud `aigcDetector_ultra`（已验证专项基线，T1）**：国内版北京地域已完成 275 张 mouse forged-only 运行，275/275 有效；30 张命中 `risk_edit`，另 1 张命中 `risk_fake`。该 API 只返回越过厂商阈值的图片级标签，`nonLabel` 无连续分数，因此当前不能直接算 AUROC/AP，也不提供定位 mask。
- **AI or Not（已验证，T1）**：使用 `only=ai_generated` 的 275 张 mouse forged-only 运行已完成，275/275 有效，仅 4/275（1.45%）正判；5 对 paired pilot 中 real 与 forged 均 0/5 正判且分数几乎不变。保留连续 confidence，完整 paired 集完成后可算 AUROC/AP。
- **Reality Defender（coverage pilot）**：每月免费 50 次，但无脸或脸太小时可能 `NOT_APPLICABLE`。先量化 restaurant/lodging coverage；coverage 不足则只进附录。
- **Illuminarty（retired/unavailable）**：7/20 起移出 active baseline，不再分配预算。7/9 的可用性记录保留为历史状态变化，不能写成未经证实的团队状况判断。
- 主表附两行 trivial 基线：随机分数；location-prior（永远预测所有 insert_box 的先验热区）——给定位指标一个诚实下限。

### 明确不跑、只引用的
- 主动溯源（C2PA / SynthID / 平台标签）：讨论段一段话说清为何结构性不适用（真图起点无凭证、开源编辑器不打水印、再编码即剥离）。
- GIMFormer、COCO-Inpaint、ForgeryGPT 等：权重未发布（调研已验证），related work 提及即可。

---

## 4. 任务、指标与协议

**T1 检测（image-level）**：AUC、AP、Acc@0.5、**TPR@5%FPR**（= 1 − 逃逸率；阈值在 real 上定）。商用 API 用返回分数算 AUC/AP，另报厂商默认阈值下的 Acc。MLLM 用置信度当分数。
**T2 定位（pixel-level，只在 forged 上算）**：pixel F1（固定 0.5 + best-threshold 两个都报）、IoU、MCC、pixel-AP（对 object mask）；次要指标 **box-hit rate**（预测 mask 与 insert_box IoU>0.3 记命中）。
**联合**：S_joint = F1_img × F1_pix（防止只解一半排名靠前；两个榜单也分开给）。
**统计**：所有数字带 95% bootstrap CI（按图重采样 1000 次）；主表按家族分组、组内按 S_joint 排序。
**适配器披露**：非原生输出（MLLM box→mask、Resemble visualization/heatmap→score map、整图 AIGC 检测器无 mask 记 N/A）在表注中逐一说明。Copyleaks RLE 是原生二值像素 mask，按官方零基 row-major 示例直接解码，不做阈值搜索；Resemble 只有在有效 API 响应确实返回空间 heatmap 后才评 T2，其余整图 API 不强行造 mask。

---

## 5. 实验块

### E1 主实验（P0）：零样本跨范式评测
规范化 v1 全集（594 real + 594×k forged，k=编辑器数）上跑 §3 全部方法 → **Table 2（主表）**：家族 × {AUC, AP, Acc, TPR@5%FPR | pixel-F1, IoU, box-hit | S_joint}。
预期故事线：家族 B 与商用整图 API 接近随机（INP-X 的域内复现）；家族 A/C 定位显著高于随机但远低于其在 CASIA/OpenSDI 上的自报成绩（跨域+小物体崩塌）;MLLM 偏"真"、定位不可用；没有方法过"解决线"。

商业 API 在批跑前默认使用相同的 5 对 canonical `real + forged` 做 preflight，先验证分数方向、有效 coverage、计费、响应 schema 和数据保留行为；Alibaba 已完成单张 preflight、5 对 pilot 和 275 张 forged-only 全量；Copyleaks 已完成前 2 对和 forged-only 前 102 张，断点剩余 173 张；Reality Defender 仍先用 5–10 张测 coverage。无效或 `NOT_APPLICABLE` 单独报告 coverage，不静默丢弃或直接混入主指标分母。

### E2 鲁棒性 / laundering（P1）
条件（作用于规范化图，两类同变换）：JPEG q ∈ {95(基准), 85, 75, 65, 50}；缩放 {0.75×, 0.5×}；"社交媒体模拟"（长边 1280 + q72）。共 8 个条件。
执行：开源方法全上（脚本批量）；商用 API 与 MLLM 用分层子样本（每条件 150 forged + 150 real）控成本。
产出：**Fig 3 退化曲线**（top-5 方法 + 2 商用，AUC 与 pixel-F1 随条件变化）。预期：本就不高的性能进一步塌向随机 → "解决线"在任何条件下都无人达到。

### E3 诊断与对照（P1）
1. **整图 vs 局部对比（headline 图,Fig 2）**：同一批检测器在 D4-1 整图对照集 vs CLAIMFORGE 局部编辑上的 AUC 柱状对比——"检测器没坏，是威胁模型变了"。
2. **真实物体贴回**：各检测器把多少真贴图判为伪造（若 IML 方法在这组也全报警 → 它们检测的是贴痕不是 AI 纹理；表格一列说清）。
3. **真实含物照片**：FPR 列（物体在场捷径审计）。
4. **格式/元数据探针**：仅用 {文件大小、量化表、分辨率} 训一个浅层分类器，验证 AUC≈0.5（构造无捷径的证明）。
5. **位置先验探针**：insert_box 空间分布热图 + location-prior 基线成绩（说明定位不是靠"猜下半张图"就能解）。
6. **切片分析**：按 domain（restaurant/lodging）、editor、object、**插入面积五分位**（Fig 4：面积-可检测性曲线）拆分 top 方法成绩。

### E4 上限与人类（P2，时间富余才做；否则一段话留作 future work）
- **fine-tune 上限**：MaskCLIP（或 Mesorch）在 train split 上微调 → test in-distribution vs held-out editor vs laundered 三档。证明"喂了数据也难泛化"（或给出防御方向）。
- **人类实验**：25+ 名被试 × 60 图（30 real/30 fake 平衡，含定位点击），报 Acc/AUC/定位命中；建立"人也看不出来 → 欺骗真实存在"。需走最简 IRB/豁免流程，来不及就砍。

---

## 6. 工程与产物

```
eval/
  build_release.py        # D2: 规范化 + manifest + GT masks + 划分
  transforms.py           # E2: laundering 条件
  adapters/               # 每方法一个: run_<name>.py → results/<name>/<condition>.jsonl
  metrics.py              # 统一读 results/ → 全部指标 + bootstrap CI → CSV/LaTeX 表
  prompts/mllm_protocol.md
results/
  <detector>/<condition>.jsonl   # {task_id, score, mask_path?, latency, raw}
```
- 统一结果 schema 是关键：所有 20 个方法只在 adapter 层有差异，metrics 一份代码。
- 环境：IMDL-BenCo 一个 venv 吃掉 5 个模型；TruFor 用 Docker；SAFIRE/RelayFormer/FakeShield 各自 venv。GPU：24GB 可跑除 FakeShield 外全部；FakeShield 需 40GB+（或砍成 SIDA-7B）。
- API 成本估算：Sightengine 当前免费日额度已用于 pilot（99 mouse=495 ops，另有 5-op smoke）；canonical paired/E2 需后续额度或付费计划。Hive 为 $6/1k（275/550 张约 $1.65/$3.30）；AI or Not 为 $0.02/张（约 $5.50/$11）；Alibaba 国内 `aigcDetector_ultra` 为 200 元/万次（275/550 张约 5.50/11 元）；Copyleaks 实测 1 credit/图但当前公开资料不足以稳定换算美元，账户 5 starter credits 已耗尽；Resemble 按秒计费；Reality Defender 前 50 次/月免费。GPT-5.5 + Gemini 3.1 Pro 约 3-4k 图 × 2 问 ≈ $100-200。总计仍以 MLLM 成本为主。
- 数据发布：HF gated dataset（研究用途条款）+ GitHub 代码；datasheet + 来源许可清单（Open Images CC BY 2.0 / Wikimedia Commons，可再分发）。

---

## 7. 时间线（今天 7/9 → 全文 7/28）

| 日期 | 任务 | 优先级 |
|---|---|---|
| 7/9–7/11 | D2 规范化管线 + D5 划分 + metrics.py + D1 lodging 生成启动 | P0 |
| 7/11–7/14 | E1：IMDL-BenCo 五模型 + TruFor + AIGC 五连 + SAFIRE/RelayFormer/MaskCLIP adapter | P0 |
| 7/13–7/15 | 商用 API 重跑（规范化集）+ MLLM 协议跑完；D3 FLUX.1-Fill 生成 + 物体多样化 | P0/P1 |
| 7/15–7/17 | E2 laundering 全跑；D4 三个对照组构建 + E3 审计与切片 | P1 |
| 7/17–7/21 | 主表/图定稿；写 Results + 重写 Intro/Abstract；**7/21 交摘要** | P0 |
| 7/21–7/26 | E4（fine-tune / 人类实验，二选一或全砍）；FakeShield 补跑；写 Discussion/AISI 落地段 | P2 |
| 7/26–7/28 | 全文打磨 + Reproducibility Checklist + 双盲检查；**7/28 交全文** | P0 |
| 7/29–7/31 | 补充材料：代码打包、full 结果表、datasheet | P0 |

砍单原则：进度落后先砍 E4 → 编辑器 3 → Reality Defender → AI or Not → Alibaba（若 region/service code 阻塞）→ Resemble T2（若无有效 heatmap）→ FakeShield（改 SIDA-7B 或砍）→ E2 商用子样本减半。Hive 因文献可比性优先保留。**E1 + D2/D4 对照 + E2 开源曲线是不可砍的最小完整论文。**

---

## 8. 写作任务映射（对现有 sections/）

- `0_abstract`/`1_introduction`：改为"已完成评测"口径；卖点数字：594 对同源真伪 × k 编辑器、插入面积中位 0.232%、~20 个检测器四范式、无一过解决线。
- `2_related_work`：加 FragFake (2505.15644)、TGIF (2407.11566)、VendorBench-100 (2607.06254, concurrent)、SHIELD、Deepfake-Eval-2024 (2503.02857)；加一小段非 CS 文献（保险欺诈、平台虚假评论、图像证据法）。
- `3_benchmark`：加入 D2 规范化、D4 对照组、D5 划分的正式描述 + Table 1（与 GIM/OpenSDI/TGIF/CocoGlide 的定位差异表）。
- `4_evaluation_protocol`：基本可留，补 TPR@5%FPR、box-hit、adapter 披露表。
- `5_preliminary_status` → **改为 Results**（Table 2 主表 + Fig 2 整图vs局部 + Fig 3 laundering + Fig 4 面积曲线 + 对照/审计表）。
- `6_discussion`：AISI 落地段（trust&safety 采购建议、provenance 局限、对平台/监管的含义）+ limitations（单一来源域、编辑器数量、无对抗攻击者）+ 伦理声明（gated release、research-only、双重用途讨论）。

**图表清单**：Fig 1 威胁模型+管线示例（已有素材）；Table 1 基准对比；Table 2 主表（E1）；Fig 2 整图vs局部反差；Fig 3 laundering 曲线；Fig 4 面积-可检测性；Table 3 对照与捷径审计；（可选）Table 4 fine-tune / 人类实验。

---

## 9. 风险与预案

| 风险 | 预案 |
|---|---|
| 时间不够 | §7 砍单序列；最小完整论文已定义 |
| Sightengine 5 月升级后表现很好 | 也是重要结果（"商用已开始响应此威胁，但定位仍缺失"）；且有 laundering 曲线兜底 |
| Illuminarty 当前服务不可用 | 移出 active baseline，不再分配预算；保留 7/9→7/20 状态变化和失败 adapter 作为 reproducibility/availability 说明，不推断团队状况 |
| Resemble heatmap 不稳定或不对应编辑区域 | paired preflight 先验证尺寸、空间对齐和响应语义；失败则只报 T1、T2=N/A |
| Copyleaks conditional localization 很高但 62.75% forged 漏检 | 主表同时报告 image-level coverage、全体 T2 和 positive-only T2；补 173 credits 完成 forged-only 后再下总体结论 |
| Alibaba Ultra service/region 文档不一致 | 先用 1 张 preflight 确认 service code、region、`risk_edit` 和账单；未通过则不批跑 |
| Reality Defender 对无脸图返回 N/A | 先报告 5–10 对 coverage；coverage 不足则移到附录，不进入主表有效分母 |
| 单物体类别捷径质疑 | D3 多样化 + D4-3 real-with-object FPR + 正文明说 mouse 集为受控子集 |
| IML 方法其实靠贴痕拿高分 | D4-2 真贴回对照直接回答；无论结果如何都是有内容的一列 |
| MaskCLIP adapter 费时 | 与 IMDL-BenCo dataset-JSON 复用；预算半天，超时先跑其余 |
| FakeShield 显存不够 | 换 SIDA-7B（24GB 可跑）或砍，MLLM 家族仍有 GPT/Gemini/Qwen |
| 审稿质疑规模（594×k） | 定位为 evaluation/stress-test 基准（先例：CocoGlide 512、AutoSplice 3.6k）；bootstrap CI 证明结论稳定；AISI rubric 不卡规模卡严谨 |

---

## 10. 关键参考（均已验证存在）

INP-X arXiv:2602.00192（商用/学术检测器在局部编辑上 91→55%，中心动机）· Fake-or-JPEG arXiv:2403.17608（格式捷径）· FragFake arXiv:2505.15644（VLM 检测/定位局部编辑差）· OpenSDI CVPR'25 arXiv:2503.19653 · IMDL-BenCo arXiv:2406.10580 · TruFor arXiv:2212.10957 · Mesorch arXiv:2412.13753 · SAFIRE arXiv:2412.08197 · RelayFormer arXiv:2508.09459 · Effort arXiv:2411.15633 · Community Forensics arXiv:2411.04125 · FakeShield arXiv:2410.02761 · SIDA arXiv:2412.04292 · TGIF arXiv:2407.11566 · VendorBench-100 arXiv:2607.06254（concurrent）· Deepfake-Eval-2024 arXiv:2503.02857 · LOKI arXiv:2410.09732 · Forensics-Bench arXiv:2503.15024 · GenImage arXiv:2306.08571 · 校准告诫 arXiv:2602.01973。
