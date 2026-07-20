# Alibaba Cloud Ultra good-mouse 全量运行记录（2026-07-20）

## 1. 运行范围

- 输入集合：`claimforge_generation_review_labels.json` 中 275 个 `status=good`、`candidates=mouse` 的 forged 拼回图。
- 顺序：沿用 Sightengine pilot 的固定 `ordered_inputs` 顺序。
- 输入协议：原图解码为 RGB，再编码为无元数据 JPEG quality 95、4:4:4 subsampling。
- API：国内版 `ImageModeration`，地域 `cn-beijing`，endpoint `green-cip.cn-beijing.aliyuncs.com`，service `aigcDetector_ultra`。
- 上传：使用内容安全 SDK 签发的短期凭据上传到临时 OSS；不依赖项目自己的 OSS Bucket。
- Runner：`eval/commercial/run_alibaba.py`，逐条 JSONL 落盘并按成功 ID 断点续跑。

## 2. 覆盖与标签

| 项目 | 数量 | 比例 |
|---|---:|---:|
| 目标 forged 图片 | 275 | 100% |
| 有效结果 | 275 | 100% |
| API 错误 | 0 | 0% |
| `risk_edit` | 30 | 10.91% |
| `risk_fake` | 1 | 0.36% |
| `risk_aigc` | 0 | 0% |
| 任一风险标签 | 31 | 11.27% |
| `nonLabel` | 244 | 88.73% |

30 个 `risk_edit` confidence 的均值为 86.38，中位数为 86.31，范围为 70.58–99.47。唯一 `risk_fake` confidence 为 82.39。返回风险等级为 high 13、medium 8、low 10、none 244。

按 domain 描述，lodging 有 11/147（7.48%）命中 `risk_edit`，restaurant 有 19/128（14.84%）命中；另一个 restaurant 样本仅命中 `risk_fake`。该差异可能受编辑面积、画面纹理和场景组成影响，不能仅凭这次 forged-only 运行解释为领域效应。

## 3. Paired preflight

在全量运行前，使用与 Hive/Resemble 相同的前 5 个 task 运行 5 对 canonical `real + forged`：

- 10/10 请求有效。
- 5 张 real 全部返回 `nonLabel`。
- 5 张 forged 全部返回 `nonLabel`。
- 这 5 对的厂商默认阈值下没有真阳性或假阳性。

## 4. 解释边界

- 275 张主批次是 forged-only；不能由 11.27% 计算 FPR、AUROC 或 AP。5 对 pilot 只能提供很弱的真实图 sanity check。
- Ultra 只返回越过账号当前风险阈值的标签与 confidence；未命中时返回无 confidence 的 `nonLabel`。缺失的风险分数不能当作 0，也不能据此构造连续 ROC 曲线。
- `risk_edit` 是图片级标签，不包含 bbox、mask 或 heatmap，不能用于 T2 定位指标。
- 31/275 是厂商默认规则下的告警率；若论文称为检出率，必须明确这是 forged-only vendor-threshold detection rate，而不是完整 benchmark TPR。
- 国内版公开计价为人民币 200 元/万次，因此该 275 张主批次预计费用约 5.50 元。

## 5. 产物与恢复

- JSONL：`results/commercial/alibaba/good275_mouse_forged_canonical_jpeg_q95_20260720.jsonl`
- Run manifest：同名 `.run_manifest.json`
- Summary：同名 `.summary.json`
- 代码：`eval/commercial/run_alibaba.py`

恢复命令：

```bash
.venv/bin/python -m eval.commercial.run_alibaba \
  --tasks 275 \
  --include forged \
  --output results/commercial/alibaba/good275_mouse_forged_canonical_jpeg_q95_20260720.jsonl \
  --run-id alibaba_ultra_mouse_forged275_20260720
```

依赖为 `alibabacloud_green20220302==3.2.4` 与 `oss2`。凭据只从 `ALIBABA_CLOUD_ACCESS_KEY_ID` 和 `ALIBABA_CLOUD_ACCESS_KEY_SECRET` 环境变量读取；runner、manifest 和结果均不保存凭据。`results/commercial/**` 现已作为可复现实验产物显式纳入 Git。
