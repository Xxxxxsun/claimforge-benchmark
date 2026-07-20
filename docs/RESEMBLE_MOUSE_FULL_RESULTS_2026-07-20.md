# Resemble Detect good-mouse 全量运行记录（2026-07-20）

## 1. 运行范围

- 输入集合：`claimforge_generation_review_labels.json` 中 275 个 `status=good`、`candidates=mouse` 的 forged 拼回图。
- 顺序：沿用 Sightengine pilot 的固定 `ordered_inputs` 顺序。
- 输入协议：解码为 RGB，再编码为无元数据 JPEG quality 95、4:4:4 subsampling，统一上传文件名 `image.jpg`。
- API：`POST https://app.resemble.ai/api/v2/detect`，请求 `Prefer: wait` 和 `visualize=true`。
- Runner：`eval/commercial/run_resemble.py`，4 workers，逐条 JSONL 落盘并按成功 ID 断点续跑。

## 2. 当前覆盖

| 项目 | 数量 |
|---|---:|
| 目标 forged 图片 | 275 |
| 有效结果 | 274 |
| API 错误 | 1 |
| 已保存 IFL heatmap | 274 |
| 已保存 visualization | 274 |

唯一未完成项为 `lodging_220_slot_001__forged`。API 返回 HTTP 402 `insufficient_balance`：单张预计 3.5 cents，运行结束时 wallet 余额 2.5 cents，差 1 cent。补充余额后再次运行同一命令只会重试这一项。

## 3. 厂商标签

| Resemble label | 数量 | 占 274 个有效结果 |
|---|---:|---:|
| `Fake` | 30 | 10.95% |
| `Likely fake` | 11 | 4.01% |
| `Neutral/Uncertain` | 5 | 1.82% |
| `Likely real` | 37 | 13.50% |
| `Real` | 191 | 69.71% |

若把 `Fake + Likely fake` 都视为厂商正判，当前 forged 检出为 41/274（14.96%），未正判为 233/274（85.04%）。若只把 `Fake` 视为正判，则为 30/274（10.95%）。正式论文必须同时保留原始标签和连续 score，不能只选择更有利的二值化规则。

整图 `image_metrics.score`：均值 0.23496，中位数 0.09792，P95 0.93636，范围 0.00301–0.99780。IFL score：均值 0.13563，中位数 0.01308，P95 0.78558，范围 0.00669–0.99138。

## 4. 解释边界

- 这是 forged-only 全量运行；只有先前 5 对 pilot 含真实原图，因此 14.96% 不是完整 benchmark 的 TPR，也无法由此计算 FPR、AUROC 或 AP。
- 5 对 pilot 中，4/5 forged 没有可见局部热区；唯一 `Likely fake` 样本是全图覆盖，mouse GT 框内变化没有高于框外。全量 heatmap 已保存，但在明确其 RGB 渲染语义前不能直接作为单通道 score mask 计算 pixel-AP。
- 当前账号不支持 `zero_retention_mode`，因此该批使用 Resemble 默认媒体存储策略。
- 最后一张补跑成功前，所有统计均以 274 个有效结果为分母。

## 5. 产物与恢复

- JSONL：`results/commercial/resemble/good275_mouse_forged_canonical_jpeg_q95_20260720.jsonl`
- Run manifest：同名 `.run_manifest.json`
- Summary：同名 `.summary.json`
- Heatmap/visualization：同名无后缀目录，共约 187 MB。

恢复命令：

```bash
.venv/bin/python -m eval.commercial.run_resemble \
  --tasks 275 \
  --include forged \
  --workers 4 \
  --output results/commercial/resemble/good275_mouse_forged_canonical_jpeg_q95_20260720.jsonl \
  --run-id resemble_good275_mouse_forged_20260720
```

凭据只通过 `RESEMBLE_API_TOKEN` 环境变量提供。按当前仓库共享要求，JSONL、manifest、summary 以及完整 heatmap/visualization artifact 均纳入 `results/commercial/**`；这些 JPEG 单文件均低于 1 MB，Git 会对字节相同的 artifact blob 去重。
