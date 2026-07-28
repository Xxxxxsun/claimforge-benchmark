# Full-image mouse-250 商业 API 结果

**日期：** 2026-07-25

**最近更新：** 2026-07-27

**条件：** `claimforge_v1_full_image_mouse250`

## 1. 输入与协议

本轮固定使用 Hunyuan full-image 生成 manifest 中前 250 个成功样本，
包含 restaurant 128 张、lodging 122 张。每个服务的 run manifest 都包含
250 个唯一 task，且 task 集合完全一致。

- Hive、Resemble、Alibaba 和 AI or Not 使用无元数据 canonical JPEG Q95。
- Copyleaks 使用满足其尺寸限制的 canonical RGB PNG。
- Sightengine 和 Illuminarty 使用生成结果原始 PNG。
- 本轮只测 forged/full-image 正类，不包含 real controls，因此检出率不是
  accuracy、AUROC、AP 或 false-positive rate。

## 2. 执行状态

| Service | 有效/目标 | 正判 | 状态 |
|---|---:|---:|---|
| Hive | 250/250 | 107/250（42.80%，阈值 0.9） | Complete，0 个待测 |
| Resemble | 250/250 | `Fake` 248 + `Likely fake` 1 = 249/250 | Complete；250 heatmaps + 250 visualizations |
| Alibaba Ultra | 250/250 | `risk_edit` 230；any-risk 230/250 | Complete，0 错误 |
| AI or Not | 250/250 | 230/250（92.00%） | Complete，0 错误 |
| Copyleaks Ultra | 250/250 | 220/250（88.00%） | Complete，0 错误，消耗 250 credits |
| Sightengine | 250/250 | 237/250（94.80%，阈值 0.5） | Complete，0 个待测 |
| Reality Defender | 0/250 | N/A | 输入 dry-run 通过；当前环境缺 API key |
| Illuminarty | 0/250 | N/A | 单图 preflight 连续 5 次 HTTP 502 |

Alibaba 的 230 个 any-risk 均包含 `risk_edit`；其中 4 张也包含
`risk_fake`，`risk_aigc` 为 0。Copyleaks 的 220 个正判均带原生 RLE
mask，30 张为空 mask；平均预测 AI 像素占比为 0.01233。

Sightengine 早期 runner 未将 HTTP 400 `usage_limit` 识别为 quota，因而在
append-only JSONL 中保存了历史失败记录；runner 现已在首个 quota error 后
立即停止。2026-07-27 使用另一组有额度的账号完成剩余 33 张，最新记录覆盖
250 个 task 且全部有效。历史错误不进入有效样本指标。

## 3. 与局部拼接的同任务对照

下表只比较两种方法都已有有效结果的相同 task。每个 API 内部沿用同一上传
编码与判定规则，因此可以直接观察 pixel-preserving local splice 与
full-image generation 的差异。

| Service | Matched N | Local-splice 正判 | Full-image 正判 | 仅 full-image 正判 |
|---|---:|---:|---:|---:|
| Hive | 250 | 0 | 107 | 107 |
| Resemble | 250 | 34 | 249 | 215 |
| Alibaba Ultra | 250 | 29 | 230 | 202 |
| AI or Not | 250 | 4 | 230 | 226 |
| Copyleaks Ultra | 250 | 99 | 220 | 127 |
| Sightengine | 178 | 0 | 168 | 168 |

这个对照明确支持当前 benchmark 的核心猜想：同一 source/task 下，整图生成
很容易触发商业 AIGC 检测器，而保留绝大多数原始像素的局部拼接显著降低检出。
这不是对 detector 通用质量的结论；它只说明检测结果对编辑机制高度敏感，而
这种敏感性正是 CLAIMFORGE 要测量的对象。

## 4. 剩余工作

尚未完成的服务：

- Reality Defender：提供新 API key/额度后运行；此前 local-splice free-plan
  50-image pilot 已用完月度免费切片。
- Illuminarty：等待 endpoint 恢复；当前不应批量重试 502。

完整机器可读状态位于
`results/commercial/full_image_mouse250_status_20260725.json`。逐图结果均为
append-only JSONL，并配有输入 hash、run manifest 和 summary。新生成结果及
manifest 已核对：有效 ID 均属于固定 250-task 集合，未发现凭据字符串。

Hive、Resemble 及本轮其他已执行服务的逐图结果、run manifest 和 summary
均已同步到主工作树。
