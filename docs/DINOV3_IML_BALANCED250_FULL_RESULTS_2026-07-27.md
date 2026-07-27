# DINOv3-IML checkpoint 48 在 Balanced250 上的正式结果

日期：2026-07-27（UTC）

> **状态：正式完成。r2 双 smoke、1,025-image formal、统一 T2 分析与
> fresh full-model replay 全部通过。**
>
> DINOv3-IML 输出 dense manipulation map，没有独立 image-level head。
> 本报告严格只发布 T2；不把 map statistic 提升为 T1，full-frame
> 条件为 T1=`N/A`、T2=`N/A`。

正式 run：
`dinov3_iml_checkpoint48_balanced250_v1_full1025_r2_20260727`

Formal immutable fingerprint：
`e2e14a951c1d1298c2f2a7488af86a7ffb04dd2449ae90cfa1a2cfecf23e7ec9`

## 1. 结论

DINOv3-IML 是最后三种 T2-only 方法中 Balanced250 macro pixel AP 和
固定阈值 macro F1 最高者；Cat 最强，Mouse 和 Trash-can 仍有明显的
resolution/threshold coverage 限制。

| 指标 | 结果 |
|---|---:|
| formal coverage | 1,025 / 1,025 |
| 三条件 macro per-image pixel AP | 0.505394 [0.478313, 0.531152] |
| 三条件 macro per-image F1 / IoU | 0.301665 / 0.243559 |
| pooled-pixel precision / recall | 0.625025 / 0.042527 |
| pooled-pixel F1 / IoU | 0.079635 / 0.041469 |
| real mean per-image false-positive fraction | 0.006514 |
| real micro false-positive fraction | 0.004630 |

分条件结果：

| 条件 | Images | Per-image pixel AP | Per-image F1 |
|---|---:|---:|---:|
| Local Mouse | 250 | 0.305788 | 0.183692 |
| Local Cat | 250 | 0.722740 | 0.549485 |
| Local Trash-can | 250 | 0.487654 | 0.171818 |

Cat 的 AP 0.723、F1 0.549 显示 checkpoint 对该类植入有很强迁移信号。
但全部 local pooled recall 仍只有 4.253%；高 precision 说明超过阈值的
区域通常较可信，不代表模型覆盖了大部分 GT pixels。

## 2. 冻结方法与 checkpoint

执行作者的
[DINOv3-IML repository](https://github.com/Irennnne/DINOv3-IML)
commit `ba45b0a203c698b36fe2b0e658bb49ebbb1163cc`，tree
`2a3c3451770e9eb4217f9640d1f735b054ccd873`；DINOv3 architecture
checkout 固定为 Meta repository commit
`31703e4cbf1ccb7c4a72daa1350405f86754b6d1`，tree
`17a6bd45271dcb2cdc32bc77c6aebae557372981`。

| 项 | 冻结值 |
|---|---|
| checkpoint | `checkpoint-48.pth` |
| bytes | 1,321,705,819 |
| SHA-256 | `01f23401e048f706ea0e63fb0429ddef80db3197ac0f5707bd584a8b056177fa` |
| configuration | CAT ViT-L/16, LoRA rank 32 |
| recorded epoch | 48 |
| load | full backbone + LoRA + segmentation head, strict |

DINOv3-IML repository code 是 MIT，但 checkpoint 内含完整 Meta DINOv3
backbone，且 release 没有单独 checkpoint license；下游使用和再分发必须
同时审查 Meta DINOv3 License Agreement。本报告不是法律意见。

## 3. 执行与统计协议

ViT-L/16 backbone 产生 32x32 dense features，LoRA-r32 调整各 attention
block 的 QKV，轻量 segmentation head 输出 manipulation logits。冻结路径：

```text
canonical JPEG
  -> Pillow RGB
  -> direct bilinear stretch 512x512
  -> ImageNet normalization
  -> 32x32 logits
  -> bilinear logits to 512x512
  -> sigmoid
  -> bilinear probability to native geometry
```

没有 crop、padding、re-encode、TTA、ensemble 或 Balanced250
fine-tuning。保存 32-space logits、512-space logits、512 probability、
native probability 和 official PNG mask。

官方 standalone script 会经过 8-bit display round-trip；CLAIMFORGE
按照预注册 continuous-output adapter 直接保存 `model.predict()` 的
float32 sigmoid tensor，避免展示量化改变连续 AP。Official artifact
使用 strict `score > 0.5`，共享 T2 metrics 使用 `score >= 0.5`。

正式选择是 real 275 + Mouse/Cat/Trash-can 各 250，共 1,025。Local GT
是 native exact difference；real 是 all-zero；750 个 full-frame rows
没有 forward 或 score-map-loader call。统计使用 1,000 次共享
source-content-cluster Poisson bootstrap。

## 4. 可复现性与机器证据

r2 A/B smoke 各 20 张，80 个数组与 20 个掩码全部 exact，0 mismatch。
Fresh replay 重新加载 checkpoint 并重放 1,025/1,025，比较 4,100 个
数组和 1,025 个掩码，0 mismatch。Replay digest：
`507b03a58c0b82702ac3aea8ce6a7ce6eea20cfcbbe7b7c03d465681307875f7`。

- [manifest](../results/opensource/dinov3_iml/dinov3_iml_checkpoint48_balanced250_v1_full1025_r2_20260727/manifest.json) —
  `18fd642f7543284242377a9366b724667bf4da1f39331d444f00722370dd918f`
- [expected inputs](../results/opensource/dinov3_iml/dinov3_iml_checkpoint48_balanced250_v1_full1025_r2_20260727/expected_inputs.jsonl) —
  `a952b6adcbd5af20ce635a5929c7045350b9f8e6729bae8d9cbf1978050f2cca`
- [per-image results](../results/opensource/dinov3_iml/dinov3_iml_checkpoint48_balanced250_v1_full1025_r2_20260727/results.jsonl) —
  `d7d0a3cd76b1fc90b830d6a83a1834f76ee3abd32d41fce008737ee3bf998f50`
- [coverage summary](../results/opensource/dinov3_iml/dinov3_iml_checkpoint48_balanced250_v1_full1025_r2_20260727/summary.json) —
  `7c8793fa4af3f5365a8694bac5c8943f802e8f9d0d5c502271c52e9cbe36dc1e`
- [metrics](../results/opensource/dinov3_iml/dinov3_iml_checkpoint48_balanced250_v1_full1025_r2_20260727/metrics.json) —
  `81972c77303d5464903364325aaada3f2e7338124ba8b3c7067188f1f3fd9312`
- [fresh replay audit](../results/opensource/dinov3_iml/dinov3_iml_checkpoint48_balanced250_v1_full1025_r2_20260727/independent_audit.json) —
  `e1ab71a3fd0948da496144d5b9130664e9c1e41273bc0f6f7dd5d7cc781b2f50`
- [A/B smoke comparison](../results/opensource/dinov3_iml/dinov3_iml_checkpoint48_balanced250_v1_smoke5x4_a_r2_20260727__vs__dinov3_iml_checkpoint48_balanced250_v1_smoke5x4_b_r2_20260727_comparison.json) —
  `638f3d27e1723fd827aa2c7f7b82b69a57336eaf133f34ca40b21a474f256320`

Dataset deterministic contract：
`671d1739bebf4370d26b4629ca26b56cc546a817d469ba505cc39bda8b33102c`。
本次重建 manifest SHA 是
`5685071fb752cded8ddf8841b8fb80547c9f5d046d180f3ad7dd9faa728cc15e`；
三个核心 ledger SHA 与冻结 release 完全一致。

## 5. r1 fail-closed 记录

首轮推理完成后，共享 analyzer callback 参数顺序错误导致读取结果时
fail closed。修复后从 A/B smoke 到 formal、metrics、full replay 全部
重跑为 r2。两轮 inference arrays、masks 与统计数值一致；r2 的 adapter
source hash `7596aef4a4b153205c915ed7aef67de6349cfa5c2372540e2e0d39898dadd310`
与提交代码一致，所以只发布 r2。
