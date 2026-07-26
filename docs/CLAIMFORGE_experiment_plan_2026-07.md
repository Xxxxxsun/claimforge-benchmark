# CLAIMFORGE 实验设计与论文执行计划（AAAI-27 AISI track）

*定稿日期 2026-07-09。配套调研：`survey_opensource_detectors_2026-07-09.md`（开源方法，权重可用性已逐一验证）、`survey_commercial_mllm_2026-07-09.md`（商用 API + MLLM + 文献证据）。早期设计讨论见 `CLAIMFORGE_paper_design.md`，本文件取代其中的实验部分。*

*商用 API 状态更新（2026-07-21）：Illuminarty 当前不可用；active roster 包含 Sightengine、Hive、Copyleaks 与 Resemble，并对 Alibaba Ultra、AI or Not、Reality Defender 做分级执行。Hive 与 Copyleaks 均已完成 forged-only 全量 275 张：Hive 在厂商阈值下 0/275 正判；Copyleaks 111/275 正判，原生 RLE 的全体 mean IoU 为 0.3296。完整核验见 `COMMERCIAL_API_STATUS_2026-07-20.md`。原始时间线与 7/9 定稿内容保留作为计划快照。*

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
- **旧商业 API 结果不可进入主表**：旧跑法是 PNG 伪图 vs JPEG 真图（格式混淆），论文数字必须来自规范化后的 paired v1 集。Sightengine 仍需在 canonical 数据上重跑；`SIGHTENGINE_MOUSE_PILOT_RESULTS_2026-07-20.md` 的 199/275 forged-only original-PNG pilot 只能作为失效现象与工程验证，不能计算主表所需的 AUC、FPR 或 paired accuracy。
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

### 家族 A/C：局部取证检测与定位（T1+T2，核心竞争者）— 11 个
已完成全量：**CAT-Net v2、MVSS-Net、TruFor、MaskCLIP、PSCC-Net、IML-ViT、
HiFi-IFDL、Mesorch、RelayFormer、DINOv3-IML**。
暂缓：**NFA-ViT/BR-Gen**（runner、指标与独立审计已就绪，但官方
`checkpoint-9999.pth` 只有需登录的百度网盘入口；不计作完成）。其中只有作者原生
提供独立 image score/head 的方法进入
原生 T1；map-only 方法的 T1 记 N/A，不把 map mean/max 冒充分类头。

### 家族 B：整图 AIGC 检测（仅 T1，预期接近随机——本身就是结果）— 10 个
已完成 9 个 local-splice 全量。**FSD official v1.2 inference release**
为 550/550 有效，AUROC `0.500350`、AP `0.502708`、无原生 T2（详见
[`FSD_V1_2_0_MOUSE_FULL_RESULTS_2026-07-24.md`](FSD_V1_2_0_MOUSE_FULL_RESULTS_2026-07-24.md)）；
**UniversalFakeDetect official Ours LC** 当前官方 HEAD 主流程为
550/550 有效，AUROC `0.499650`、AP
`0.497293`、无原生 T2，另有独立 checkpoint-era preprocessing
sensitivity（AUROC `0.503260`、AP `0.510188`；详见
[`UNIVERSALFAKEDETECT_OURS_LC_MOUSE_FULL_RESULTS_2026-07-24.md`](UNIVERSALFAKEDETECT_OURS_LC_MOUSE_FULL_RESULTS_2026-07-24.md)）。
**NPR official AIGCDetectBenchmark ProGAN-4class** 已于 2026-07-25
完成并通过独立全模型审计：550/550 有效，官方 probability AUROC
`0.500198`、AP `0.502661`，严格 `sigmoid(logit) > 0.5` 下
TP/FP/FN/TN=`0/0/275/275`，无原生 T2（详见
[`NPR_AIGC_PROGAN4CLASS_MOUSE_FULL_RESULTS_2026-07-25.md`](NPR_AIGC_PROGAN4CLASS_MOUSE_FULL_RESULTS_2026-07-25.md)）。
其主协议在查看结果前冻结为官方仓库 AIGCDetectBenchmark 配置：commit
`781ced3f7ca2cdc69ec9dd4ef27e8d0b3c07752a`、官方链接的 ProGAN-4class
`model_epoch_last_3090.pth`（SHA-256
`b67a91555ce786a6d0463ff0cb2b0b874d1c3f971b0e3febd2ae5618a80f7e8a`）、
原生分辨率、不 resize/crop、batch 1，奇数边仅删最后一行/列，并以严格
`sigmoid(logit) > 0.5` 判 fake；无原生 T2。仓库另一份 `NPR.pth`
只记录为发布歧义，不根据 Mouse 分数事后选权重。全量前 CUDA pair smoke
发现约 `-170` 的有限 logit 会在 float32 sigmoid 中精确下溢成 0，因此预注册
双轨且全量无条件同时报告：官方 sigmoid 概率仍是主 operational score 和
`>0.5` 判定唯一依据；raw-logit AUROC/AP、real-only 5% FPR、paired
ranking/delta 与同一 pair bootstrap 只作数值稳定诊断，不能在两者间择优。
官方 HF Space 只用于佐证 checkpoint 与原生尺寸预处理；其代码漏掉
`model.eval()`，会令 BatchNorm 保持 train mode，因此记为部署缺陷，不作为
可执行参考或额外 sensitivity。主跑遵循官方 GitHub `test.py` 的 eval mode。
**Community Forensics** 已于 2026-07-25 完成 Mouse local-splice 主条件并
通过独立全模型审计。其主协议在查看任何 Mouse 模型分数前冻结为论文后续
实验采用的最佳 384×384
High-res ViT-S/16：官方 main commit
`ee5b71d43db0f3779e1edd64ee927b13f2dd6ad4`，单图执行语义佐证 commit
`5e52ed690bdbd609f9bb1705c4c80d11872a05bd`，HF 模型 revision
`6076002bf0d9dd37537f965ee2f06f826c333b61` 的
`model.safetensors`（87,262,324 bytes；SHA-256
`b89f36275f3bf5e2b040eee36597a8f19db051bff9a473a9cf7b2466284fb387`）
及 HF processor revision `3540a3f0d688f8bf492a8aed48613b891f88047e`。
主预处理为 Pillow RGB → bilinear 短边 `Resize(440)` → `CenterCrop(384)`
→ `[0,1]` tensor → ImageNet normalize → float32，batch 1、eval mode、
无 AMP；主分数为 float32 `sigmoid(logit)`，严格 `>0.5` 判 fake，
无原生 T2。完整 safetensors 覆盖全部 21,811,969 个参数，因此 adapter
以 `pretrained=False` 构造同一 timm 1.0.15 架构后 strict-load，避免官方
class 在随即被完整覆盖前再下载一个可变 ImageNet base；官方 notebook 五张
DALL·E 2 golden probability 已在 `1e-5` 绝对误差门禁内复现，并匹配其
四位小数显示。224 版本不根据 Mouse 分数追加或
替换；论文明确将 384 版本作为 best-performing model。裁剪前置可见性也已
冻结：162/275 full、32 partial、81 none，不能把被裁掉的 edit 当模型漏检，
且不能把分类 attention/feature 冒充定位输出。
全量为 550/550 有效、0 error；官方 probability AUROC `0.502340`（pair
bootstrap 95% CI `[0.500674, 0.504873]`）、AP `0.504511`
（`[0.502691, 0.511090]`），严格 `probability > 0.5` 下
TP/FP/FN/TN=`1/1/274/274`。唯一被判 fake 的 forged 与其 matched real
同时被判 fake。独立审计重新执行 550 次完整 ViT 前向并以最大绝对误差
`0.0` 重现全部 384 维 feature、logit、probability 与 decision（详见
[`COMMUNITY_FORENSICS_HIGHRES384_MOUSE_FULL_RESULTS_2026-07-25.md`](COMMUNITY_FORENSICS_HIGHRES384_MOUSE_FULL_RESULTS_2026-07-25.md)）。
**SPAI** 已于 2026-07-25 完成 Mouse local-splice 主条件并通过独立全模型
审计。其主协议在查看任何 Mouse 模型分数前冻结为
CVPR 2025 官方唯一 release：`mever-team/spai` commit
`8ff7b3b6779b4fcb43cf313471d9cb1c62d129a4`，Google Drive 官方唯一
`spai.pth`（934,865,338 bytes；SHA-256
`24159f27d7c8c2cd0cb6c4019189eb89ad0874a0d9d15f8dc9afd39ca9648a55`；
324 tensors / 139,945,243 elements）。主路径为 Pillow RGB、原生分辨率、
`[0,1]` float32、无 resize/crop、batch 1、eval、无 AMP；224×224
非重叠 patch、stride 224，当前官方 config/README 的
`MINIMUM_PATCHES=4`（少于 4 块时 five-crop），patch feature chunk 400，
SCA 聚合后输出 float32 `sigmoid(logit)`，严格 `>0.5` 判 fake。
checkpoint 内嵌的旧训练 config 仍是 `MINIMUM_PATCHES=1`，但不覆盖当前
release inference config，也不能根据 Mouse 分数事后改选。由于常规
`unfold` 会丢弃不能整除 224 的右/下余边，前置 exact-GT 可见性已冻结为
243/275 full、14 partial、18 none，平均可见 GT 比例
`0.9096355444251016`；262 对走网格、13 对走 five-crop（后者全部 full）。
该分层只是输入条件，不是模型定位。官方可选 SCA attention 是分类决策
重要性可视化而非编辑概率 mask，因此 T2 与 joint gate 记 N/A。
Mouse 主跑前还用官方 3.72 GB evaluation bundle 的两个原始样本完成了
非 Mouse release audit。当前 NGC 环境启动时把 matmul precision/TF32
设为 `high/true/true`，主协议显式覆盖为 PyTorch 标准严格 float32：
matmul precision `highest`、CUDA matmul TF32 false、cuDNN TF32 false。
在这个冻结精度下，MJ v6.1 `224.png` 为
logit `0.9909347295761108` / probability `0.7292724847793579`，SD3
`000001046_4.webp` 为 `1.6814128160476685` / `0.8430914878845215`。
当前 checkpoint/code 的重复前向 bit-identical，但这两个概率不落在项目页
手写 `0.748`、`0.87` 的舍入区间内；项目页没有提供 checkpoint hash 或
full-precision reference。因此网页值只记作 stale/approximate release
evidence，不据此调预处理、换权重或阻止 Mouse 主跑；上述当前唯一 release
的 full-precision 值冻结为 implementation regression reference。
全量为 550/550 有效、275/275 complete pairs、0 error；官方 probability
AUROC `0.497931`（pair-bootstrap 95% CI `[0.495543, 0.499836]`），
AP `0.500215`（`[0.499264, 0.506572]`），严格
`probability > 0.5` 下 TP/FP/FN/TN=`46/48/229/227`。forged 分数仅在
111/275 对中严格高于 matched real，另有 145 losses、19 ties；平均
forged-minus-real delta 为 `-0.003603`（95% CI
`[-0.006439, -0.001193]`）。18 个 none-visibility pair 因 edit 全在模型
未消费的余边像素而逐对 exact tie。独立审计重新执行 550 次完整
FFT/ViT/SRS/SCA/MLP 前向，所有 patch feature、聚合 feature、attention
diagnostic 与 raw logit 最大绝对误差均为 `0.0`，probability 最大误差仅
一个 float32 ULP（`5.960464477539063e-08`）；T2 仍为 N/A（详见
[`SPAI_MOUSE_FULL_RESULTS_2026-07-25.md`](SPAI_MOUSE_FULL_RESULTS_2026-07-25.md)）。
**B-Free** 已于 2026-07-25 完成 Mouse local-splice 主条件并通过独立全模型
审计。其主协议在查看任何 Mouse 模型分数前冻结为
CVPR 2025 官方唯一 release：`grip-unina/B-Free` commit
`c6a9f898782fb466b29af01f21960b67415afb0e`；官方唯一
`BFREE_dino2reg4.zip` 为 321,653,488 bytes、MD5
`f3f53fa647848b16cf81c913f148a198`、SHA-256
`8230fd3f0f3a64a6403acb692ce1663718ed16f36a5a4de4a68c0d273781769f`，
其中 checkpoint SHA-256 为
`5948ca78f4d94e820c250d24cdf155035b4a85960443800bfe6bb7f06bffe947`。
模型是端到端微调的 DINOv2 ViT-B/14 + 4 registers + 单 logit head。
主路径为 Pillow RGB → ToTensor → ImageNet normalize，保持原生分辨率；
14×14 patch embedding 后取 center/TL/BL/BR/TR 五个 504×504 等价 token
crop，并平均 **raw logits**，严格 `logit > 0` 判 fake。任一 token-grid
维不足 36 时，release 实际执行 periodic wrap，并把另一维也截为最前 36
tokens，之后五份相同；这比论文的“padding/multiple crops”描述更具体，
因此以 release 为准。前置 exact-GT 可见性已冻结为 173/275 full、36
partial、66 none，平均可见 GT 比例 `0.6891766376903072`；26 对进入 wrap
路径。crop logit/feature 不是定位输出，T2/joint 均 N/A。
B-Free 的 `origBG` 训练版本会把生成区域与原始真实背景组合，论文明确称其
“effectively a local image edit”，因此与 Mouse 高度相关；但 COCO/SD2.1、
全图重生成样本、额外全局增强和当前 Hunyuan 场景仍有明显分布差异。
官方 config 不记录 inpainted++ recipe 且未发布训练代码，所以只声称官方
release inference。其 GRIP 自定义许可仅允许 informational/nonprofit use，
不是可商用的宽松开源许可。
正式 run `bfree_dino2reg4_mouse_canonical_v1_full275_20260725` 为
550/550 有效、275/275 complete pairs、0 error。官方 raw-logit AUROC
`0.512529`（1,000-pair-bootstrap 95% CI
`[0.507815, 0.518612]`），AP `0.513062`
（`[0.509910, 0.521203]`）；严格 `raw_logit > 0` 下
TP/FP/FN/TN=`2/1/273/274`，forged recall 仅 `0.007273`。
forged 分数在 143/275 对中严格高于 matched real，另有 67 losses、65
ties；paired ranking accuracy `0.520000`（95% CI
`[0.458182, 0.578182]`），平均 forged-minus-real delta `0.061657`
（`[0.039677, 0.084790]`）。可见性分层揭示弱但真实的局部响应：
173 个 `full` pair 的 paired ranking accuracy 为 `0.693642`，36 个
`partial` 为 `0.638889`，而 66 个 `none` 中 65 个精确打平。独立审计
重新解码并完整前向全部 550 张图，逐一验证 550 个 `[5,768]` feature 和
`[5]` crop-logit 产物，并以最大绝对误差 `0.0` 重放 feature、crop logit、
raw logit、decision 与汇总（详见
[`BFREE_DINO2REG4_MOUSE_FULL_RESULTS_2026-07-25.md`](BFREE_DINO2REG4_MOUSE_FULL_RESULTS_2026-07-25.md)）。
**Effort** 已于 2026-07-25 完成 Mouse local-splice 主条件并通过独立
全模型审计。主协议在查看 Mouse 模型分数前冻结为 ICML 2025 官方
GenImage SDv1.4 checkpoint：源码 commit
`96f5dea2b534d400cfd7003f053c7e93c8e16461`，checkpoint
1,213,769,519 bytes、SHA-256
`7c32ceb4e66d303050e8fc5dc7543fa347693fb4ee6b5df4d6eaf9f6a92fb813`，
以及官方 natural-image demo 的 OpenCV BGR→RGB、直接
224×224 `INTER_LINEAR`、CLIP normalization 路径。模型为
CLIP ViT-L/14；24 层的 q/k/v/out 共 96 个 attention linear 都使用
rank-1023 冻结主子空间加 rank-1 可训练残差，随后以两类 float32
softmax class-1 概率、严格 `>0.5` 判 fake。checkpoint 的 681 个
FP32 tensor / 303,378,530 elements 已严格 681/681 加载；无原生定位
输出，T2/joint 均 N/A。正式 run 为 550/550 有效、275/275 complete
pairs、0 error；AUROC `0.500456`（95% pair-bootstrap CI
`[0.498379,0.502850]`）、AP `0.506262`，TPR@5%FPR
`0.054545`。发布阈值下 TP/FP/FN/TN=`23/23/252/252`，全部 275 对的
real/forged 判定均相同；paired ranking 为 `0.530909`
（CI `[0.469091,0.589091]`），胜/负/平 `146/129/0`，sign-test
`p=0.334638`。独立审计 fresh-forward 全部 550 张图并以最大绝对误差
`0.0` 重现 feature、logit、概率、判定和 summary（详见
[`EFFORT_CLIP_L14_GENIMAGE_SDV14_MOUSE_FULL_RESULTS_2026-07-25.md`](EFFORT_CLIP_L14_GENIMAGE_SDV14_MOUSE_FULL_RESULTS_2026-07-25.md)）。
**OmniAID-DINO v2** 已于 2026-07-25 完成 Mouse local-splice 主条件并
通过独立全模型审计。主协议在查看 Mouse 分数前冻结为当前官方推荐且 Space
默认的 DINO v2 / Mirage / Auto Router：GitHub commit
`40749406fbcd8893c11a160edf4a72a2d4dc7056`、Space commit
`cf99ed518af8b7256854d01994d6e41165553bb3`、checkpoint
3,238,483,725 bytes / SHA-256
`8135cf83a7acbd3d88e457062f7ad693b1f2e27ffc8d5ae7ec73fcb5de806ea9`。
模型含两个 DINOv3 ViT-L/16 图、96 个 SVD-MoE attention projection、
top-2/5 semantic router 和固定 Artifact expert；主路径为 RGB、直接
bilinear+antialias 448×448 拉伸、ImageNet normalize、float32、batch 1，
以 softmax class-1 严格 `>0.5` 判 fake，无原生 T2。
正式 run 为 550/550 有效、275/275 complete pairs、0 error；AUROC
`0.499636`（95% pair-bootstrap CI `[0.496661,0.502401]`）、AP
`0.505429`、TPR@5%FPR `0.058182`。发布阈值下
TP/FP/FN/TN=`8/8/267/267`，全部 275 对的判定均未翻转；paired ranking
`0.501818`（CI `[0.447273,0.560000]`），胜/负/平 `138/137/0`，
平均 forged-minus-real delta `0.000575010`
（CI `[-0.000497496,0.001597893]`）。274/275 对的 semantic top-2
集合不变。独立审计重新构图并 fresh-forward 全部 550 张图，逐项重放六类
artifact、router/head/softmax、判定和汇总，所有最大绝对差均为 `0.0`
（详见
[`OMNIAID_DINO_V2_MIRAGE_MOUSE_FULL_RESULTS_2026-07-25.md`](OMNIAID_DINO_V2_MIRAGE_MOUSE_FULL_RESULTS_2026-07-25.md)）。
**CNNDetection Blur+JPEG(0.1)** 已于 2026-07-25 完成 Mouse
local-splice 主条件、预注册的 paper-era crop sensitivity 和全模型 replay
审计。主协议在查看 Mouse 分数前冻结为官方 commit
`ea0b5622365e3a9cd31d1b54b6b5971131a839ab`、官方 Dropbox 的
Blur+JPEG(0.1) checkpoint（282,442,597 bytes；SHA-256
`a73295ac66f9cb74d558ce3ade46f75e2f2997ed05eeed0f4b774623372058ea`）
及官方 2020-06 推荐的 RGB、原生分辨率、no resize/no crop、batch 1、
ImageNet normalize 路径。单 float32 logit 经未校准 sigmoid 后严格
`>0.5` 判 fake，无原生 T2；Blur/JPEG 只属于训练增强，测试时不施加。
正式 native run 为 550/550 有效、275/275 complete pairs、0 error；
AUROC `0.498896`（95% pair-bootstrap CI
`[0.497401,0.500046]`）、AP `0.502089`、TPR@5%FPR `0.047273`，
发布阈值下 TP/FP/FN/TN=`0/0/275/275`。paired ranking 为
`0.454545`（CI `[0.396364,0.512727]`），胜/负/平 `125/150/0`，
sign-test `p=0.147691`，平均 forged-minus-real score delta
`-7.381589e-7`（CI `[-2.581092e-6,5.426804e-7]`）。全模型 replay
重新前向 550 张图并以最大绝对误差 `0.0` 核验全部 2,048 维 feature；
统计复算使用共享 metrics，因此明确不声称第二套完全独立统计实现。
预注册的 native CenterCrop(224) sensitivity 同样 550/550 完成和审计，
AUROC `0.499702`、AP `0.499548`；它只保留 14 个 full、14 个 partial
编辑，247 个 none，产生 246 个 exact score ties，不能替换 native
primary（详见
[`CNNDETECTION_BLUR_JPG_PROB0_1_NATIVE_MOUSE_FULL_RESULTS_2026-07-25.md`](CNNDETECTION_BLUR_JPG_PROB0_1_NATIVE_MOUSE_FULL_RESULTS_2026-07-25.md)）。
至此所有可取得官方权重的候选都已执行；仅 **LTD** 的官方权重访问受阻，
没有用第三方权重或伪造 Mouse 分数。前 7 个构成的最小机制完备集已全部
完成；当前完成 9/10 个 local-splice 条件、尚余 1 个 official-weight
blocker。D4 同域整图生成
对照集尚未建立，因此 0/10 达到“local-splice + fully synthetic contrast”
双条件完成标准。

完整 checkpoint、顺序、完成判据和 appendix 候选冻结在
[`OPENSOURCE_BASELINE_EXECUTION_PLAN_2026-07-24.md`](OPENSOURCE_BASELINE_EXECUTION_PLAN_2026-07-24.md)。

### 家族 D：MLLM — 4 个
- 零样本 prompt：**GPT-5.5**、**Gemini 3.1 Pro**（两家旗舰；已有文献证明上一代在局部编辑上接近随机且偏"真"，FragFake arXiv:2505.15644、arXiv:2506.10474）。
- 开源专用：**FakeShield**（ICLR'25，Apache-2.0，HF 22B 权重，verdict+mask+解释，需 40GB 级 GPU）。
- 开源通用（可选第 4 个）：Qwen 系旗舰 VL 零样本。
- 统一 prompt 协议：固定两问("这张照片是否被 AI 编辑过？给 0-100 置信度" / "若有，给出编辑区域 bounding box")，温度 0，每图 1 次；box→mask 作为**非原生定位 adapter** 明确标注。

### 家族 E：商用 API — 核心 4 个 + 分级 preflight
- **Sightengine genai（核心，T1）**：成熟 pixel-only 整图基线，2026-05 刚升级 AI-edit 检测。当前已有 199 张 forged-only original-PNG pilot，但 canonical paired v1 仍待跑。未来主结果使用 `results/commercial/sightengine/` 下独立 run ID，不能与 pilot 混合。
- **Hive AI-generated image + deepfake classifier（核心，T1）**：文献可比性最强的新增商业基线；forged-only 已完成 275/275，厂商阈值 0.9 下 0/275 检出，最高分 0.033728。自助 V3、$6/1k、默认 100 requests/day，返回整图 AI/Human 分数和生成器归因，无定位。与 Sightengine 组成 INP-X 同款商用对；完整 paired real 仍待跑。
- **Resemble Detect（核心候选，T1；T2 需 preflight）**：返回整图 fake/real 分数；官方 schema 在 `visualize=true` 时可返回 image heatmap。先用 5–10 对验证 heatmap 是否稳定、是否与输入空间对齐、是否真正响应局部编辑；验证失败则 T2 记 N/A，不能把可视化直接当 GT-compatible mask。
- **Copyleaks `ai-image-1-ultra`（核心，T1+T2）**：生产端点已完成 275/275 unique mouse forged，111/275（40.36%）正判。命中的 111 张上，原生 RLE mask 对 SP 精确像素差分 GT 的平均 precision 0.9739、recall 0.8389、IoU 0.8165；计入 164 个空 mask 漏检后的全体 mean IoU 0.3296。不能把 conditional localization 质量写成整体定位性能。
- **Alibaba Cloud `aigcDetector_ultra`（已验证专项基线，T1）**：国内版北京地域已完成 275 张 mouse forged-only 运行，275/275 有效；30 张命中 `risk_edit`，另 1 张命中 `risk_fake`。该 API 只返回越过厂商阈值的图片级标签，`nonLabel` 无连续分数，因此当前不能直接算 AUROC/AP，也不提供定位 mask。
- **AI or Not（已验证，T1）**：使用 `only=ai_generated` 的 275 张 mouse forged-only 运行已完成，275/275 有效，仅 4/275（1.45%）正判；5 对 paired pilot 中 real 与 forged 均 0/5 正判且分数几乎不变。保留连续 confidence，完整 paired 集完成后可算 AUROC/AP。
- **Reality Defender（forged-50 coverage pilot 已完成）**：50/50 restaurant/lodging 输入均 overall applicable，但全部判为 `AUTHENTIC`，归一化分数仅 0.01–0.03。可作为 T1 前缀失败结果；额外 real controls 需要新的月度或付费额度。
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

商业 API 在批跑前默认使用相同的 5 对 canonical `real + forged` 做 preflight，先验证分数方向、有效 coverage、计费、响应 schema 和数据保留行为；Alibaba 已完成单张 preflight、5 对 pilot 和 275 张 forged-only 全量；Copyleaks 已完成前 2 对和 forged-only 全量 275 张；Reality Defender 已完成 forged-only 前 50 张，coverage 为 100%。无效或 `NOT_APPLICABLE` 单独报告 coverage，不静默丢弃或直接混入主指标分母。

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
- API 成本估算：Sightengine 两个日批次已用于 pilot（199 mouse=995 ops，另有首日 5-op smoke），尚余 76 张 original-PNG forged；canonical paired/E2 需后续额度或付费计划。Hive 为 $6/1k（275/550 张约 $1.65/$3.30）；AI or Not 为 $0.02/张（约 $5.50/$11）；Alibaba 国内 `aigcDetector_ultra` 为 200 元/万次（275/550 张约 5.50/11 元）；Copyleaks forged-only 全量实测 275 credits（1 credit/图），但当前公开资料不足以稳定换算美元；Resemble 按秒计费；Reality Defender 本次 forged coverage pilot 提交了 50 次 scan，若账户按公开 free tier 且无额外额度计，则会占满月度 50 次。GPT-5.5 + Gemini 3.1 Pro 约 3-4k 图 × 2 问 ≈ $100-200。总计仍以 MLLM 成本为主。
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
| Copyleaks conditional localization 很高但 59.64% forged 漏检 | 主表同时报告 image-level coverage、全体 T2 和 positive-only T2；全量 forged-only 已完成，paired real controls 另行扩充 |
| Alibaba Ultra service/region 文档不一致 | 先用 1 张 preflight 确认 service code、region、`risk_edit` 和账单；未通过则不批跑 |
| Reality Defender 对无脸图返回 N/A | 实测 forged-50 overall coverage 为 100%，但 50/50 均判 `AUTHENTIC`；报告 coverage 与漏检，并在无 real controls 时限制为前缀结果 |
| 单物体类别捷径质疑 | D3 多样化 + D4-3 real-with-object FPR + 正文明说 mouse 集为受控子集 |
| IML 方法其实靠贴痕拿高分 | D4-2 真贴回对照直接回答；无论结果如何都是有内容的一列 |
| MaskCLIP adapter 费时 | 与 IMDL-BenCo dataset-JSON 复用；预算半天，超时先跑其余 |
| FakeShield 显存不够 | 换 SIDA-7B（24GB 可跑）或砍，MLLM 家族仍有 GPT/Gemini/Qwen |
| 审稿质疑规模（594×k） | 定位为 evaluation/stress-test 基准（先例：CocoGlide 512、AutoSplice 3.6k）；bootstrap CI 证明结论稳定；AISI rubric 不卡规模卡严谨 |

---

## 10. 关键参考（均已验证存在）

INP-X arXiv:2602.00192（商用/学术检测器在局部编辑上 91→55%，中心动机）· Fake-or-JPEG arXiv:2403.17608（格式捷径）· FragFake arXiv:2505.15644（VLM 检测/定位局部编辑差）· OpenSDI CVPR'25 arXiv:2503.19653 · IMDL-BenCo arXiv:2406.10580 · TruFor arXiv:2212.10957 · Mesorch arXiv:2412.13753 · SAFIRE arXiv:2412.08197 · RelayFormer arXiv:2508.09459 · Effort arXiv:2411.15633 · Community Forensics arXiv:2411.04125 · FakeShield arXiv:2410.02761 · SIDA arXiv:2412.04292 · TGIF arXiv:2407.11566 · VendorBench-100 arXiv:2607.06254（concurrent）· Deepfake-Eval-2024 arXiv:2503.02857 · LOKI arXiv:2410.09732 · Forensics-Bench arXiv:2503.15024 · GenImage arXiv:2306.08571 · 校准告诫 arXiv:2602.01973。
