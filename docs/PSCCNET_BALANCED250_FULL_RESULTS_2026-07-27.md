# PSCC-Net（TCSVT 2022 / official synthetic-pretrained）在 Balanced250 上的正式结果

日期：2026-07-27（UTC）

> **文档状态：完成（final replay audit passed）**
>
> 本报告只使用冻结的 1,775-image formal run、两个独立 35-image CUDA
> smoke、共享 Balanced250 T1/T2 指标、逐文件 artifact audit 和 fresh
> full-model replay。旧 Mouse run、论文表格和作者展示分数没有被拿来填写
> Balanced250 结果。

正式 run：
`psccnet_tcsvt2022_official_balanced250_v1_full1775_20260727`

Formal immutable run-config fingerprint：
`b8aa95b8852366e8ab73b2ff47a4f27a7db1985addd0c48fd9dd5e8c0b06edb4`

核心机器证据：

- [run manifest](../results/opensource/psccnet/psccnet_tcsvt2022_official_balanced250_v1_full1775_20260727/manifest.json)：
  `cc05a2988eb8af287289ad507fe0da2dd560b010ac3676d98bfa3a65b93b6ec3`
- [expected inputs](../results/opensource/psccnet/psccnet_tcsvt2022_official_balanced250_v1_full1775_20260727/expected_inputs.jsonl)：
  `6b5128909eeffdbd88e61af02ca1bd191cb5460f94a23b47c87ebd0172e6d12c`
- [逐图结果](../results/opensource/psccnet/psccnet_tcsvt2022_official_balanced250_v1_full1775_20260727/results.jsonl)：
  `471723b5e0650acfe29dcf6c634e24aa70c7aaa8f5f6e8d3e2aa0beffef464f9`
- [coverage summary](../results/opensource/psccnet/psccnet_tcsvt2022_official_balanced250_v1_full1775_20260727/summary.json)：
  `27467418d11935c4e0763047d84b3193c3d8912d4ee7e1fa8933097a0409228d`
- [Balanced250 T1/T2 metrics](../results/opensource/psccnet/psccnet_tcsvt2022_official_balanced250_v1_full1775_20260727/balanced250_metrics.json)：
  `8d2a698dc450fb55a6d00c7d99aee26334465057594bd8c29fc860215488dee7`
- [independent replay audit](../results/opensource/psccnet/psccnet_tcsvt2022_official_balanced250_v1_full1775_20260727/independent_audit.json)：
  `c14c974f3510e4d15cec81bfadbadbe617705b904dab78dc63d0106f860acd99`
- [双 smoke comparison](../results/opensource/psccnet/_reports/psccnet_tcsvt2022_official_balanced250_v1_smoke5x7_a_20260727__vs__psccnet_tcsvt2022_official_balanced250_v1_smoke5x7_b_20260727_comparison.json)：
  `de89593bc841a48a897dbe11ed85cf7b2e9823b824233567e2901716d788472c`

## 1. 结论摘要

PSCC-Net 是 Balanced250 上第四个完成正式扩展的 local-forensics
checkpoint。它同时提供独立图像分类 head 和原生 dense localization
map，因此 1,775 张都进入 T1，real275 与三类 local 共 1,025 张进入
T2；三类 full-frame 的 T2 严格为 `N/A`。

- formal coverage 为 1,775/1,775，零 error、missing 和 superseded；
- 三类 local 等权 macro T1 AUROC 为 **0.525445**，AP 为
  **0.520271**，TPR@5%FPR 为 **5.33%**；
- 三类 full-frame 等权 macro T1 AUROC 为 **0.588851**，AP 为
  **0.657848**，TPR@5%FPR 为 **25.33%**；
- `score > 0.5` 在 independent real250 上的 specificity 为
  **95.60%**，但 local macro recall 只有 **4.80%**；
- full-frame macro recall 为 **22.13%**，仍不足以构成可靠的整图
  AIGC detector；
- source-matched local strict ranking 为 **58.27%**，mean score
  delta 为 `+0.003008`；
- source-matched full-frame strict ranking 为 **61.47%**，mean
  score delta 为 `+0.172618`；
- 三类 local 的等权 per-image pixel AP 为 **0.178419**；
- 对应 per-image F1/IoU@0.5 只有 **0.007708 / 0.005300**；
- local pooled micro recall 为 **0.70%**，micro IoU 为
  **0.004149**；
- 275 张 real 的 mean per-image false-positive fraction 为
  **0.008259**（约 0.83%），micro fraction 为 **0.008344**；
- 两个 smoke 的 T1、logits、四级 progressive maps、native maps
  和适用 masks 全部 exact。

这批证据支持一个比 Mouse-only 结果更细的结论：

1. 官方 PSCC checkpoint 对三类局部 diffusion insertion 的 T1 仍接近
   chance，低 FPR 区域没有可用召回。
2. Cat/Trash-can 的连续 pixel AP 明显高于 Mouse，但固定阈值召回仍极低；
   特别是 Trash-can 本身占图面积大，AP 的随机基线也更高，不能只看 AP
   数字就声称定位成功。
3. 三类 full-frame conditional edit 出现一致但有限的图像级信号，
   AUROC 约 0.59、TPR@5%FPR 约 25%；它比 local 条件强，但仍达不到
   可部署检测水平。
4. 当前 full-frame 数据是从真实源图进行 Hunyuan 条件全图编辑，不是
   脱离真实图独立采样的纯 text-to-image；本结果不能外推为对所有整图
   AIGC 生成器的结论。
5. 以上都是 checkpoint、训练分布和 threat model 特定的零样本结果，
   不能写成“PSCC 架构在原始 copy-move/splice/removal 任务上无效”。

## 2. 官方来源、checkpoint 与许可证

方法对应
[Progressive Spatio-Channel Correlation Network for Image Manipulation Detection and Localization](https://arxiv.org/abs/2103.10596)，
正式执行作者
[PSCC-Net repository](https://github.com/proteus1991/PSCC-Net)
commit `53e5ff77d8dc5feddda060cd085f9b765761f816`。

仓库内提交了完整 synthetic-pretrained bundle：

| Component | Bytes | SHA-256 |
|---|---:|---|
| HRNet task checkpoint | 8,305,545 | `d3b21edc4930187a6801cc818bd7b999fb5d8078d8f2e2193572e91ea5160096` |
| NLC localization checkpoint | 2,900,709 | `11ea3461253cf059b299ad4b6b89008485f94a2d2b2da83ec28c2282a095b00b` |
| DetectionHead checkpoint | 3,739,969 | `a17581e8a3489a360257a266ca9b2db1b7c9b43337fbcc4aeb8d751f593f66f5` |
| constructor HRNet initialization | 16,012,341 | `06924c741ea8c076a569d5e164aa628910a72020800e4a4945e8b40b241ce5cb` |

三任务权重 bundle SHA-256 为
`893626e154e5a3c16322e845a0e8c775029f88a5742de6875818c69f66459560`。
四个权重文件都用 CPU `weights_only=True` 读取并完成 tensor-schema
审计。CPU structural gate 抑制 constructor-only initialization load，
再将三个 task checkpoint strict-load；正式与 fresh runtime 则保留官方
constructor path，先加载已逐字节绑定的 initialization weight，再由完整
HRNet task checkpoint strict 覆盖最终状态。三个最终执行组件合计
3,667,942 parameters、13,490 buffer elements、391 modules，
missing/unexpected keys 均为空。

代码仓库和四个权重文件都由作者 Git 仓库直接提供，关键 source 和 asset
逐文件绑定。仓库包含 MIT license；audit 将代码和仓库内 checkpoint
分类为允许使用和再分发。这个技术分类不审计训练数据权利、第三方依赖义务
或具体产品法域，也不构成法律意见。

## 3. 方法原理与 domain gap

PSCC-Net 不只是对每个像素独立分类。它组合：

1. HRNet-W18-small-v2，在较高空间分辨率上保留局部线索；
2. progressive non-local correlation modules，同时比较空间位置与通道
   的长程相关性；
3. 从粗到细的四级 localization outputs，前一阶段引导后续细化；
4. 与 segmentation map 分离的 DetectionHead，直接输出两类整图 logits。

这种设计适合 copy-move、经典 splice 和 removal：重复区域会产生长程
空间对应，粘贴边界会破坏局部统计，progressive correlation 可以跨尺度
聚合这些痕迹。官方 checkpoint 的训练 manipulation 冻结为 authentic、
splicing、copy-move 和 RFR-Net object-removal inpainting。

Balanced250 local insertion 的生成机制不同：扩散模型在真实场景内合成并
融合对象，未必保留复制来源，也未必产生传统剪贴边界。Mouse 的目标面积
尤其小；Cat 和 Trash-can 面积更大，所以连续 pixel ranking 可以上升，
但固定阈值下依然很少触发。这是与结果一致的 domain-gap 解释，不是因果
ablation。

Full-frame 条件编辑改变更大范围的图像统计，因此 independent DetectionHead
出现比 local 更强的响应也符合模型机制。不过 AUROC 约 0.59 仍说明真实与
编辑分布大量重叠。

## 4. Frozen executable contract

正式入口：

- [runner](../eval/opensource/run_psccnet_balanced.py)
- [independent analyzer](../eval/opensource/analyze_psccnet_balanced.py)
- [legacy official-contract adapter](../eval/opensource/run_psccnet.py)
- [legacy PSCC metrics](../eval/opensource/psccnet_metrics.py)
- [shared T1 metrics](../eval/opensource/balanced250_metrics.py)
- [shared T2 metrics](../eval/opensource/balanced250_localization_metrics.py)

冻结文件 SHA-256：

| File | SHA-256 |
|---|---|
| runner | `0d96b086267d8eb26443455f90d81160e9b31e681609812f10e44eef71219e39` |
| analyzer | `e8f01144759f3dc06f4dd1b644ff2acd18fc5522e02627d0cfe777d1cd414a6c` |
| runner tests | `85de5287ee81190a098485c14688dac6b8311b960ebb4f825c188add6d17e25a` |
| analyzer tests | `d074537d76c2eaa2bd929917a46b3d4273eaf4e6704f758c39b231babd2ffe6e` |
| legacy official-contract adapter | `cc3660f972cebd426111efe5c7d32beed8739c50994accafb4657e78ebeb5bff` |
| legacy PSCC metrics | `ccd7925c0254d9944116e6cfe9b30b012909d1df8ab81711adc5357040004f5a` |
| shared Balanced250 T1 metrics | `f3932099bb63b766f063a66684e1d45f6e12601337d73859591d79297dbbed1c` |
| shared Balanced250 T2 metrics | `83ac07257078fc41276742fa4b9f2eb936ac51c8ff93bf1253b8c45f2b704b2a` |

### 4.1 Preprocess 与 forward

| Component | Frozen behavior |
|---|---|
| decode | `imageio.v2.imread` |
| channels | RGB；RGBA 按官方 float32 white-background formula composite |
| geometry | 保持 native input；无 resize/crop/letterbox |
| scale | contiguous CHW FP32，uint8 除以 255 |
| normalization | 无 ImageNet mean/std |
| batch / precision | batch 1 / FP32；autocast false |
| TTA / ensemble | false / false |

Forward 依次运行 HRNet、NLCDetection 和 DetectionHead。NLC 直接返回四个
已经 sigmoid 的 probability maps，尺寸为 256²、128²、64²、32²；
adapter 不施加第二次 sigmoid。

### 4.2 T1 与 T2

T1 使用独立 DetectionHead：

```text
ai_score = softmax(float32_two_class_logits)[1]
decision = ai_score > 0.5
```

它不是 localization map 的 max/mean，也没有被称为校准概率。

T2 主输出固定为官方第一层 `mask1`（256×256）。它以 bilinear、
`align_corners=True` 恢复到 native decoded geometry，并保存 float32
NPY。适用条件的正式 PNG mask 使用 strict `native_probability > 0.5`；
full-frame 只保存 diagnostic probability map，不生成 T2 mask。

共享 Balanced250 T2 reducer 的冻结 operator 是 `>= 0.5`。PSCC 原生
probability 为任意 float32，理论上可以精确等于 0.5，因此两者不能被宣称
为普遍等价。正式 artifact audit 与 fresh replay 使用官方 strict `>`；
共享跨方法表保留冻结的 `>=`，metrics 明确记录
`shared_t2_operator_equivalent_to_official=false`，不静默改协议。

## 5. 数据覆盖与能力边界

正式 cache 为：

| Condition | T1 | T2 | Rows |
|---|---|---|---:|
| real | valid | all-zero FP-area only | 275 |
| local Mouse | valid | exact-difference mask | 250 |
| local Cat | valid | exact-difference mask | 250 |
| local Trash-can | valid | exact-difference mask | 250 |
| full-frame Mouse | valid | N/A | 250 |
| full-frame Cat | valid | N/A | 250 |
| full-frame Trash-can | valid | N/A | 250 |

Primary T1 使用独立 real250 panel 与每个 forged250 做非配对比较。Secondary
source-matched analysis 使用额外缓存的 25 张 real 和明确 source links；
不会从 task ID 猜 pair。T2 只使用 real275 与三类 local750。橙色
conditioning box 从未被当作 full-frame GT。

## 6. T1 图像级结果

### 6.1 Independent primary panel

| Condition | AUROC | AP | TPR@5%FPR | Accuracy@0.5 | Recall@0.5 |
|---|---:|---:|---:|---:|---:|
| local Mouse | 0.507808 | 0.510925 | 0.052 | 0.502 | 0.048 |
| local Cat | 0.537248 | 0.532171 | 0.056 | 0.504 | 0.052 |
| local Trash-can | 0.531280 | 0.517718 | 0.052 | 0.500 | 0.044 |
| **local macro** | **0.525445** | **0.520271** | **0.0533** | **0.502** | **0.0480** |
| full-frame Mouse | 0.593200 | 0.661656 | 0.260 | 0.592 | 0.228 |
| full-frame Cat | 0.577680 | 0.650020 | 0.224 | 0.580 | 0.204 |
| full-frame Trash-can | 0.595672 | 0.661867 | 0.276 | 0.594 | 0.232 |
| **full-frame macro** | **0.588851** | **0.657848** | **0.2533** | **0.5887** | **0.2213** |

所有行复用同一 independent real250；其 fixed-threshold specificity 为
0.956。Local AUROC CI 为 `[0.511458, 0.539379]`，统计上虽略高于
0.5，但实际增益小，TPR@5%FPR 仍只有 5.33%。Full-frame AUROC CI 为
`[0.540143, 0.634985]`，说明存在信号，但不是高准确检测器。

### 6.2 Source-matched secondary

下表统一使用等权 condition-macro aggregation；`Pairs` 只表示明确
source-matched endpoints 的数量，不把 nonlinear median 偷换成 pooled
median。

| Family | Pairs | Strict forged > source | Mean score delta | Median delta |
|---|---:|---:|---:|---:|
| local | 750 | 0.582667 | +0.003008 | +0.000174 |
| full-frame | 750 | 0.614667 | +0.172618 | +0.020947 |
| all | 1,500 | 0.598667 | +0.087813 | +0.010561 |

Local mean delta 的 95% cluster-bootstrap CI 为
`[0.001485, 0.004823]`；full-frame 为 `[0.135085, 0.206945]`。
配对结果能揭示微小方向性，但不替代 independent primary AUROC，也不能
把 58% paired ranking 写成高质量分类。

## 7. T2 定位结果

`Target pixel fraction` 是 condition 内 pooled pixel prevalence；其余列是
per-image macro，不能与 pooled micro 指标混读。所有 `@0.5` 列遵循共享
reducer 的 `>= 0.5`，不是官方 mask artifact 的 strict `> 0.5`。

| Local condition | Target pixel fraction | Pixel AP | Precision@0.5 | Recall@0.5 | F1@0.5 | IoU@0.5 |
|---|---:|---:|---:|---:|---:|---:|
| Mouse | 0.001350 | 0.016313 | 0.000477 | 0.011613 | 0.000080 | 0.000040 |
| Cat | 0.063678 | 0.224083 | 0.421449 | 0.020285 | 0.015102 | 0.010806 |
| Trash-can | 0.219667 | 0.294861 | 0.381721 | 0.007813 | 0.007943 | 0.005053 |
| **condition macro** | — | **0.178419** | **0.267882** | **0.013237** | **0.007708** | **0.005300** |

三类 pooled micro 指标为：

| Metric | Estimate |
|---|---:|
| precision | 0.104659 |
| recall | 0.006998 |
| F1 | 0.008249 |
| IoU | 0.004149 |
| MCC | 0.000275 |

Cat 与 Trash-can 的连续 AP 不能脱离 prevalence 解读：两类 target pixel
fraction 分别约 6.37% 和 21.97%，而 Mouse 只有 0.135%。即使
Trash-can AP 达 0.295，fixed-threshold recall 仍只有 0.77%、IoU
只有 0.51%。这更像有部分连续排序信息但操作点失效，而不是成功定位。

275 张 real 的 false-positive area：

| Statistic | Estimate |
|---|---:|
| mean per-image FP fraction | 0.008259 |
| micro FP fraction | 0.008344 |
| mean FP pixels/image | 13,418.56 |

Lodging mean FP fraction 为 1.305%，restaurant 为 0.276%；这是描述性
slice，不能在没有预注册域间检验时写成稳定域效应。

## 8. Determinism 与独立审计

两个 smoke 各包含 35 张：real 和六类 forged 条件各 5 张。A/B 选择、
环境、checkpoint 和运行 contract 相同但 run directory 独立。Comparison
由最终冻结 analyzer 在隐藏 CUDA 的 CPU-only 模式重新生成，其内嵌
analyzer SHA-256 与本报告的 `e8f01144759f...` 一致，并验证：

- 35/35 terminal rows exact；
- T1 score、decision、raw logits exact；
- 四级 progressive float32 NPY exact；
- 35 个 native float32 maps exact；
- 20 个 T2-applicable masks exact；
- 共 195 个生成 artifact files exact。

Formal artifact audit 验证：

- 1,775/1,775 rows 与冻结 selection 一致；
- 7,100 个 progressive NPY；
- 1,775 个 native float32 probability maps；
- 1,025 个 strict-threshold PNG masks；
- 750 个 full-frame masks 正确缺席；
- path、file hash、array hash、shape、dtype 和 finite/range 全部一致；
- independent softmax 与 native `align_corners=True` interpolation
  复算在注册容差内；
- shared T1/T2 metrics 只从独立验证后的 result rows 与 native maps 重算。

最终 fresh replay 已从 checkpoint 新建完整模型，并按冻结顺序重跑
1,775 次 forward。Audit 写出 `status=replay_audit_passed`：

- 1,775 张重新打开、重新预处理并完成 forward；
- 1,775 组 logits、probabilities、T1 scores 与 decisions exact；
- 7,100 个 progressive maps 与 1,775 个 native maps exact；
- 1,025 个适用 masks 由 strict `> 0.5` 重新导出并 exact；
- 750 个 full-frame masks 仍正确缺席；
- logits、probabilities、progressive maps 和 native maps 的 fresh replay
  maximum absolute difference 均为 `0.0`。

`fresh_model_metrics_exact=true` 来自 metric-bearing rows 与 1,025 个
适用 native maps 的穷尽 exact 证明；bootstrap reducer 没有对完全相同的
输入再跑一次。当前 `psccnet_balanced_replay_audit_v2` schema 不含独立
`publishable` 布尔字段，因此本报告不伪写 `publishable=true`；该 schema
的发布门禁由 `status=replay_audit_passed` 和上述 exact counters 共同
表达。

## 9. 科学边界

本报告允许的表述：

- “official synthetic-pretrained PSCC checkpoint 在 Balanced250 local
  insertion 上 T1 接近 chance，T2 fixed-threshold recall 很低”；
- “对三类 Hunyuan source-conditioned full-frame edits 有有限 T1
  ranking signal，但远未达到可靠检测”；
- “Cat/Trash-can 连续 pixel AP 高于 Mouse，但受目标面积和固定阈值失效
  共同影响”。

本报告不支持：

- 把 full-frame conditioning box 当 localization GT；
- 把当前 full-frame 条件编辑称为独立纯 T2I 数据；
- 用 T2 map 的 max/mean 替代原生 DetectionHead T1；
- 用 test-set 搜索阈值替换官方 0.5；
- 将某个条件较高的 AP 直接解释为可部署 mask；
- 将单个 checkpoint 的 domain gap 扩大为整个架构无效。

## 10. 测试与复现

PSCC Balanced runner/analyzer 的 39 个针对性测试全部通过，Black check
clean。测试覆盖 selection、CPU strict-load、manifest fail-closed、
artifact inventory、native restore、strict threshold、shared reducer
callback、smoke exact comparison、metrics 和 fresh replay contracts。

正式命令：

```bash
CUDA_VISIBLE_DEVICES=5 \
PYTHONHASHSEED=0 \
PYTHONDONTWRITEBYTECODE=1 \
CUBLAS_WORKSPACE_CONFIG=:4096:8 \
/root/.cache/claimforge/venvs/psccnet/bin/python \
  -m eval.opensource.run_psccnet_balanced \
  --mode formal \
  --repo-root /root/claimforge-benchmark \
  --run-id psccnet_tcsvt2022_official_balanced250_v1_full1775_20260727 \
  --device cuda:0

CUDA_VISIBLE_DEVICES=5 \
PYTHONHASHSEED=0 \
PYTHONDONTWRITEBYTECODE=1 \
CUBLAS_WORKSPACE_CONFIG=:4096:8 \
/root/.cache/claimforge/venvs/psccnet/bin/python \
  -m eval.opensource.analyze_psccnet_balanced \
  --run-id psccnet_tcsvt2022_official_balanced250_v1_full1775_20260727 \
  --device cuda:0
```

Raw float maps 在 `outputs/` 下并受 `.gitignore` 保护；Git 提交只纳入
runner、analyzer、tests、small JSON evidence 和报告，不提交数 GB 的
逐图 NPY/PNG。

## 11. 下一步

PSCC 完成并单独 push 后，按冻结顺序发布已经完成 GPU/audit 的
HiFi-IFDL；随后是 T2-only 的 CAT-Net v2、IML-ViT、Mesorch、
RelayFormer 和 DINOv3-IML。T2-only 方法不会被人为提升成整图 AIGC
detector。
