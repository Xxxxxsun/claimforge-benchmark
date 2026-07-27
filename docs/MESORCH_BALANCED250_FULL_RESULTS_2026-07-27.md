# Mesorch epoch 98 在 Balanced250 上的正式结果

日期：2026-07-27（UTC）

> **状态：正式完成。r2 双 smoke、1,025-image formal、统一 T2 分析与
> fresh full-model replay 全部通过。**
>
> Mesorch 没有独立图像分类 head。本报告只发布原生像素定位 T2，
> 不把 heatmap 的最大值、均值、面积或 nonempty 状态改造成 T1；
> 三类 full-frame 条件均为 T1=`N/A`、T2=`N/A`。

正式 run：
`mesorch_epoch98_balanced250_v1_full1025_r2_20260727`

Formal immutable fingerprint：
`0ed32e5918e55875e2e27c77670dc400c600d86e9d91c197e5ce68be9d22c17a`

## 1. 结论

Mesorch 对 Cat 和 Trash-can 植入有连续像素排序信号，但固定 0.5
阈值的 recall 很低；tiny Mouse 的迁移尤其弱。

| 指标 | 结果 |
|---|---:|
| formal coverage | 1,025 / 1,025 |
| 三条件 macro per-image pixel AP | 0.296096 [0.272303, 0.321255] |
| 三条件 macro per-image F1 / IoU | 0.098080 / 0.076472 |
| pooled-pixel precision / recall | 0.380869 / 0.015650 |
| pooled-pixel F1 / IoU | 0.030065 / 0.015262 |
| real mean per-image false-positive fraction | 0.003578 |
| real micro false-positive fraction | 0.003471 |

分条件结果：

| 条件 | Images | Per-image pixel AP | Per-image F1 |
|---|---:|---:|---:|
| Local Mouse | 250 | 0.092243 | 0.042740 |
| Local Cat | 250 | 0.425032 | 0.202865 |
| Local Trash-can | 250 | 0.371012 | 0.048634 |

AP 与固定阈值结果回答不同问题：Cat/Trash-can 的排序能力不等于
0.5 阈值覆盖了同样比例的篡改区域。这里 pooled recall 只有 1.565%，
因此不能只用 AP 描述部署时的固定阈值表现。

## 2. 冻结方法与 checkpoint

执行作者的
[Mesorch repository](https://github.com/scu-zjz/Mesorch)
commit `ea82b0274b92244115d09b81663c88f57c7b78ee`，tree
`18e2d3325a77da00d2cd13d7c4a2f39ba2c74ea2`。

| 项 | 冻结值 |
|---|---|
| checkpoint | `mesorch-98.pth` |
| bytes | 1,023,886,070 |
| SHA-256 | `6d8fcd7ce7616d819bec6a9ed461b27187101e67247f8b2d2483fdc1f25f685a` |
| recorded epoch | 98 |
| load | safe container validation + strict model-state load |

Repository code 是 MIT；外部 checkpoint 没有单独声明 license，
所以这里不把代码许可证自动扩展到权重文件。

## 3. 执行与统计协议

Mesorch 将 RGB/频率信息送入 ConvNeXt local branch、SegFormer global
branch 和像素级 gating decoder。冻结输入与输出路径为：

```text
canonical JPEG
  -> Pillow RGB
  -> direct Albumentations stretch 512x512
  -> ImageNet normalization
  -> 128x128 fused logits
  -> bilinear logits to 512x512, align_corners=True
  -> sigmoid
  -> bilinear probability to native geometry, align_corners=False
```

没有 crop、re-encode、TTA、ensemble 或 Balanced250 fine-tuning。
正式产物保存 128-space logits、512-space probability、
native probability 和 official native PNG mask。

官方 artifact 使用 strict `score > 0.5`；跨方法 Balanced250 T2
reducer 冻结为 `score >= 0.5`。连续 native map 用于 exact pixel AP；
local GT 是 decoded source/forged RGB exact difference，real GT 是
all-zero，只报告 false-positive area。

选择严格为：

| 条件 | 数量 |
|---|---:|
| real | 275 |
| local_mouse | 250 |
| local_cat | 250 |
| local_trash_can | 250 |

三类 full-frame 共 750 张没有进入 forward，也没有被赋予伪造的 T2
target。Bootstrap 使用共享 source-content-cluster Poisson weights，
1,000 iterations，报告双侧 95% percentile interval。

## 4. 可复现性与机器证据

r2 两个独立 smoke 各包含 real/Mouse/Cat/Trash-can 各 5 张。两次运行的
60 个数组和 20 个掩码全部 exact，0 mismatch。

Fresh replay 从 checkpoint 重新构造模型，重新 preprocess、forward 和
postprocess 全部 1,025 张；比较 3,075 个数组和 1,025 个掩码，0
mismatch。Replay digest：
`3fb13c86f8c27ee1282a4e7009a91e7e617ec6bf19e98bb617e22543431c08a9`。

- [manifest](../results/opensource/mesorch/mesorch_epoch98_balanced250_v1_full1025_r2_20260727/manifest.json) —
  `93e731c7b4cd7e347dc462cadba40de2dfce8c89e6dc2c2504376fd30061784b`
- [expected inputs](../results/opensource/mesorch/mesorch_epoch98_balanced250_v1_full1025_r2_20260727/expected_inputs.jsonl) —
  `a952b6adcbd5af20ce635a5929c7045350b9f8e6729bae8d9cbf1978050f2cca`
- [per-image results](../results/opensource/mesorch/mesorch_epoch98_balanced250_v1_full1025_r2_20260727/results.jsonl) —
  `69e87682ab06de70eea6960ae958b5662fda578596666e3fe8963bbb3da9cdcb`
- [coverage summary](../results/opensource/mesorch/mesorch_epoch98_balanced250_v1_full1025_r2_20260727/summary.json) —
  `a7b04a2db68cae7112b3331877c9062cc66171df1cd1905dee98fc2e2650489f`
- [metrics](../results/opensource/mesorch/mesorch_epoch98_balanced250_v1_full1025_r2_20260727/metrics.json) —
  `799a64f85d39a17bfe2fba44867c0d6c01f5eaae1d9e92291c4cca81aa0333dd`
- [fresh replay audit](../results/opensource/mesorch/mesorch_epoch98_balanced250_v1_full1025_r2_20260727/independent_audit.json) —
  `4523991e545c15595ad248814773fba4c2b4c3dda6991347068ddbacc6e184dd`
- [A/B smoke comparison](../results/opensource/mesorch/mesorch_epoch98_balanced250_v1_smoke5x4_a_r2_20260727__vs__mesorch_epoch98_balanced250_v1_smoke5x4_b_r2_20260727_comparison.json) —
  `a379b9f014164affede96d392bc335cb20437626e2897b3bcc366ccd1b4643c8`

Dataset deterministic contract 是
`671d1739bebf4370d26b4629ca26b56cc546a817d469ba505cc39bda8b33102c`；
本次重建 manifest SHA 是
`5685071fb752cded8ddf8841b8fb80547c9f5d046d180f3ad7dd9faa728cc15e`，
三个冻结 ledger SHA 与原 release 完全一致。

## 5. r1 fail-closed 记录

首轮推理完整，但统一 analyzer 的 callback 参数顺序写反，导致分析在读取
第一张结果时 fail closed；没有用手工指标替代。修复后重新执行 A/B
smoke、formal、analysis 和 full replay，形成当前 r2。r1 与 r2 的模型
输出及正式指标一致；只有 r2 的 source fingerprint 与提交代码一致，
因此 r1 不作为发布证据。
