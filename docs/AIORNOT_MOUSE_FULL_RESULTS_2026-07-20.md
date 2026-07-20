# AI or Not good-mouse 全量运行记录（2026-07-20）

## 1. 运行范围

- 输入集合：`claimforge_generation_review_labels.json` 中 275 个 `status=good`、`candidates=mouse` 的 forged 拼回图。
- 顺序：沿用 Sightengine pilot 的固定 `ordered_inputs` 顺序。
- 输入协议：解码为 RGB，再编码为无元数据 JPEG quality 95、4:4:4 subsampling，统一上传文件名 `image.jpg`。
- API：`POST https://api.aiornot.com/v2/image/sync`，查询参数 `only=ai_generated`。
- Runner：`eval/commercial/run_aiornot.py`，顺序请求、逐条 JSONL 落盘并按成功 ID 断点续跑。

## 2. 全量结果

| 项目 | 数量 | 比例 |
|---|---:|---:|
| 目标 forged 图片 | 275 | 100% |
| 有效结果 | 275 | 100% |
| API 错误 | 0 | 0% |
| `ai_detected=true` | 4 | 1.45% |
| `ai_detected=false` | 271 | 98.55% |

Forged `ai_confidence` 的均值为 0.03982，中位数为 0.01462，P95 为 0.13041，范围为 0.00173–0.90252。4 个厂商正判 confidence 分别为 0.59555、0.83564、0.89352 和 0.90252；所有负判中最高 confidence 为 0.32095。

4 个正判全部来自 lodging：lodging 为 4/147（2.72%），restaurant 为 0/128。样本构成和编辑面积等变量尚未控制，不能据此声称模型在 restaurant 上必然更弱。

## 3. Paired preflight

使用与 Hive、Resemble 和 Alibaba 相同的前 5 个 task 运行 5 对 canonical `real + forged`：

- 10/10 请求有效。
- real 0/5、forged 0/5 被判为 AI。
- real 分数均值 0.04957，forged 分数均值 0.04949。
- paired `forged - real` 均值 -0.0000818，中位数 -0.0000542，范围 -0.0003474–0.0000874。

在该 pilot 中，局部 mouse 编辑几乎没有改变 API 输出。

## 4. 解释边界

- 275 张主批次是 forged-only。1.45% 是厂商 verdict 下的 forged detection rate，不能单独计算 FPR、AUROC、AP 或完整 benchmark TPR。
- API 返回连续 `ai_confidence`，因此完成完整 real/forged paired 集后可以计算阈值无关指标；当前 5 张 real 只够做链路和方向 sanity check。
- API 目标是整图 AI-generated detection，并未明确提供通用局部编辑检测或定位；结果没有 bbox、mask 或 heatmap。
- 请求显式使用 `only=ai_generated`。默认请求还会运行 deepfake、NSFW 和 quality，其中部分报告独立计费，不能把默认多报告调用与本次成本混为一谈。

## 5. 产物与恢复

- JSONL：`results/commercial/aiornot/good275_mouse_forged_canonical_jpeg_q95_20260720.jsonl`
- Run manifest：同名 `.run_manifest.json`
- Summary：同名 `.summary.json`
- 代码：`eval/commercial/run_aiornot.py`

恢复命令：

```bash
.venv/bin/python -m eval.commercial.run_aiornot \
  --tasks 275 \
  --include forged \
  --output results/commercial/aiornot/good275_mouse_forged_canonical_jpeg_q95_20260720.jsonl \
  --run-id aiornot_mouse_forged275_20260720
```

凭据只通过 `AIORNOT_API_KEY` 环境变量提供。runner、manifest 和结果均不保存 API key；`results/commercial/**` 现已作为可复现实验产物显式纳入 Git。
