# Sightengine genai：good-275 mouse forged pilot 结果

**日期：** 2026-07-20

**运行 ID：** `sightengine_pilot_good275_mouse_forged_20260720`

**条件：** `pilot_good275_mouse_forged_original_png`

**结论状态：** pilot / 非主表结果

## 1. 结论摘要

Sightengine `genai` 在本次送测的 99 张 mouse 局部编辑图上均返回了有效结果，但在报告阈值 `ai_generated >= 0.5` 下没有检出任何一张：**检出 0/99，逃逸 99/99（100%）**。本文将 `ai_generated` 称为 API score，不假设它是经过校准的概率。

| 项目 | 结果 |
|---|---:|
| good mouse forged 总池 | 275 |
| 本次送测 | 99（总池的 36.00%） |
| 有效响应 | 99/99（100%） |
| HTTP / API 错误 | 0 |
| 分数分布 | `0.001 × 98`，`0.05 × 1` |
| 平均分 / 中位数 | 0.001495 / 0.001 |
| `>= 0.5` 检出 | 0/99（0%） |
| `< 0.5` 逃逸 | 99/99（100%） |
| 批量运行消耗 | 495 operations（每图 5） |

这支持“局部生成内容可以逃逸整图 AIGC 检测”的失效现象，但当前结果只覆盖 forged 正类，不能据此计算 accuracy、AUROC、AP、specificity 或 false-positive rate，也不能替代最终的配对主实验。

## 2. 输入与抽样

输入来自 `claimforge_generation_review_labels.json`，筛选规则为：

- `status = good`
- `candidates = mouse`
- 仅使用 `spliced_image`（`kind = forged`，`label = edited`）
- 保留仓库中的原始 PNG 字节，不重新编码；上传文件名统一为中性的 `image.png`

完整输入池为 275 张，其中 restaurant 128 张、lodging 147 张。runner 先在 `domain × orientation` 分层内按任务 ID 的稳定 SHA-256 排序，再按各层占比交错排列；本次受当日额度限制，运行该确定性顺序的前 99 张：

| Domain | Landscape | Portrait | Square | 合计 |
|---|---:|---:|---:|---:|
| lodging | 41 | 11 | 1 | 53 |
| restaurant | 41 | 4 | 1 | 46 |
| **合计** | **82** | **15** | **2** | **99** |

完整 275 张有序输入清单及每张图的 SHA-256 位于 run manifest。该输入清单的摘要为：

```text
a44669ac27c7b8abac6a4adb59db2786d7b527969814b293508e72d5d164bae7
```

## 3. API 与运行协议

- Endpoint：`POST https://api.sightengine.com/1.0/check.json`
- Model：`genai`
- 输出字段：`type.ai_generated`
- 调用方式：逐张、串行、禁用重定向；请求间隔至少 1.05 秒
- 重试：仅对网络异常及 408/409/425/5xx 做有限重试；认证、额度或 operation 单价异常立即停止
- 凭据：只从 `SIGHTENGINE_API_USER` 与 `SIGHTENGINE_API_SECRET` 环境变量读取，不进入命令行、manifest 或结果文件
- 可恢复性：每张响应立即 append + `fsync` 到 JSONL；以图片 ID 检查已有有效结果并支持续跑

在仓库根目录、已设置上述环境变量后，本批调用方式为：

```bash
python -m eval.commercial.run_sightengine \
  --max-successes 99 --operation-budget 495
```

每个成功响应都报告 5 operations。本批结果文件记录 99 × 5 = 495 operations；同一天此前用于确认凭据和 endpoint 的单张非 mouse smoke test 消耗 5 operations、未纳入本结果，因此当天合计使用 500 operations。

## 4. 结果明细

### 4.1 总体分数

| 指标 | 值 |
|---|---:|
| Min | 0.001 |
| P05 / P25 / Median / P75 / P95 | 0.001 |
| Mean | 0.0014949495 |
| Max | 0.05 |
| `< 0.01` | 98/99 |
| `< 0.05` | 98/99 |
| `< 0.10` | 99/99 |
| `>= 0.5` | 0/99 |

唯一不是 0.001 的样本是 `lodging_043_slot_001__forged`，分数为 0.05，仍低于 0.5 报告阈值。所有响应的 `type.ai_generators` 均缺失，因此没有可报告的生成器归因分数。

### 4.2 分 domain

| Domain | N | Mean | Median | Min | Max | `>= 0.5` | 逃逸率 |
|---|---:|---:|---:|---:|---:|---:|---:|
| lodging | 53 | 0.0019245283 | 0.001 | 0.001 | 0.05 | 0 | 100% |
| restaurant | 46 | 0.001 | 0.001 | 0.001 | 0.001 | 0 | 100% |

### 4.3 请求完整性

- 99/99 记录均为 HTTP 200、API `status=success`、本地 `status=ok`。
- 99/99 均在第一次请求成功，没有重试记录。
- 请求延迟 min / median / mean / max 为 408 / 1587 / 1502.263 / 4356 ms。
- 每条记录包含任务 ID、相对图片路径、图片 SHA-256、图片尺寸、字节数、延迟、operation 数、经脱敏的原始响应及完成时间。
- 结果、manifest 和 summary 中不保存 API user、secret、header 或环境变量值。

## 5. 解释边界与后续主实验

本次 pilot 可以直接说明：在这 99 张已人工审核为 good 的 mouse 局部编辑图上，Sightengine 给出的整图 AI score 极低，固定 0.5 报告阈值完全没有检出正类。

以下结论当前不能推出：

- **不能报告分类整体性能。** 没有同时送测配对 real/source 图，因此没有 TN/FP，也不能计算 accuracy、balanced accuracy、AUROC、AP 或 FPR。
- **不能视为完整 good-275 结果。** 当前只覆盖确定性抽样的 99/275；剩余 176 张尚未调用。
- **不能与旧的 PNG-forged / JPEG-real 结果合并。** 当前仍是 original-PNG forged-only pilot；最终配对实验必须按统一的图像规范处理 real 与 forged，消除文件格式混淆。
- **没有定位指标。** Sightengine `genai` 只返回整图分数，没有原生 mask、box 或 heatmap。
- **0.5 是本报告固定的判定阈值。** 完整 paired score 出齐后还应同时报告 threshold-free 指标与阈值敏感性，避免只凭单一阈值下结论。
- **极低分数按接口原值报告。** 98 个结果都恰为 0.001，这可能反映接口下限或量化；不应把它解释为已校准的 0.1% 概率。

## 6. 可复现文件

- `eval/commercial/run_sightengine.py`：Sightengine 批量 runner。
- `eval/commercial/run_illuminarty.py`：当前 runner 复用的通用输入、JSONL、hash、脱敏与统计工具。
- `results/commercial/sightengine/pilot_good275_mouse_forged_original_png_20260720.jsonl`：99 张逐图完整响应。
- `results/commercial/sightengine/pilot_good275_mouse_forged_original_png_20260720.run_manifest.json`：完整 275 张确定性顺序、输入 hash 与运行元数据。
- `results/commercial/sightengine/pilot_good275_mouse_forged_original_png_20260720.summary.json`：机器可读汇总。

manifest 中记录的 adapter SHA-256 为：

```text
5a45ec2a7351fd65029328b458a6bf22619f5391a8a805a574ce44f06c6e9bfd
```

它与本次提交的 `eval/commercial/run_sightengine.py` 一致。

三份结果文件的 SHA-256 为：

```text
JSONL        37a680a5f35d826e410eb6d60e1d38bee53bfd092b7c9b1d99b493de42a62836
run manifest 20dced6b722c9fbaf8f08408e91c4b0d23b4d348712f91263fd5051e0a3536ee
summary      19c9f5ff8e563e932f23c262ee69081457e6b2af8d7b0ecfa4d9d3c72cde2294
```
