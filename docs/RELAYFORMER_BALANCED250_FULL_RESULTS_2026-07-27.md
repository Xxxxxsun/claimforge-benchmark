# RelayFormer checkpoint 164 在 Balanced250 上的正式结果

日期：2026-07-27（UTC）

> **状态：正式完成。r2 双 smoke、1,025-image formal、统一 T2 分析与
> fresh full-model replay 全部通过。**
>
> RelayFormer checkpoint 164 只有 dense localization output，没有独立
> image-integrity head。本报告只发布 T2，不从 map 构造 T1；
> full-frame 条件为 T1=`N/A`、T2=`N/A`。

正式 run：
`relayformer_checkpoint164_balanced250_v1_full1025_r2_20260727`

Formal immutable fingerprint：
`2e84c6a4b5a890215668fc1fa8774cd9dbbe919306a1f4cb2645df2149b6220c`

## 1. 结论

RelayFormer 在三个 local 条件上均有像素排序信号，对 Cat 最强；固定
0.5 阈值的 pooled recall 仍很低，不能把排序能力等同于完整覆盖。

| 指标 | 结果 |
|---|---:|
| formal coverage | 1,025 / 1,025 |
| 三条件 macro per-image pixel AP | 0.389330 [0.360467, 0.419203] |
| 三条件 macro per-image F1 / IoU | 0.242205 / 0.194977 |
| pooled-pixel precision / recall | 0.392266 / 0.026167 |
| pooled-pixel F1 / IoU | 0.049061 / 0.025148 |
| real mean per-image false-positive fraction | 0.006147 |
| real micro false-positive fraction | 0.006171 |

分条件结果：

| 条件 | Images | Per-image pixel AP | Per-image F1 |
|---|---:|---:|---:|
| Local Mouse | 250 | 0.227121 | 0.188084 |
| Local Cat | 250 | 0.561163 | 0.441462 |
| Local Trash-can | 250 | 0.379707 | 0.097068 |

Cat 的 per-image AP/F1 明显高于 Mouse 和 Trash-can。Trash-can 的连续
排序尚可，但固定阈值覆盖很低；Mouse 目标极小，少量背景 response 就会
显著影响 precision。这些是 checkpoint 在当前 frozen protocol 下的
观测，不是架构因果消融。

## 2. 冻结方法与 checkpoint

执行作者的
[RelayFormer repository](https://github.com/WenOOI/RelayFormer)
commit `3fc863c7691d93fb5b11ca8e12e3a214d771e384`，tree
`1622bda3a45e11497a4bd50bf8e5d372aa8b434a`。

| 项 | 冻结值 |
|---|---|
| checkpoint | `checkpoint-164.pth` |
| bytes | 1,102,625,388 |
| SHA-256 | `00a0f145ae4a98e66cad95aa79d2ce470d77821ee4262d6b803b3705c11c2090` |
| recorded epoch | 164 |
| configuration | official image-only paper-v3, 1024 canvas |
| final load | complete model state strict load |

构造器打印的 `missing=0, unexpected=67` 来自官方 ViT-only partial
initialization；随后完整 RelayFormer state 的 strict load 成功，
不是接受了 67 个未加载的最终 checkpoint tensors。

Repository code 是 MIT，作者 checkpoint repository 标记 Apache-2.0；
权重与第三方依赖的具体下游权利仍需按各自条款审查。

## 3. 执行与统计协议

RelayFormer 用 overlapping local ViT windows、GLR relay tokens 和
query decoder 交换局部与全局证据。执行 paper-v3 compatibility
preprocess：

```text
canonical JPEG
  -> Pillow RGB
  -> preserve aspect ratio, long edge <= 1024
  -> top-left placement on 1024x1024 raw-RGB zero-padded canvas
  -> ImageNet normalization
  -> 1024-space logits
  -> crop valid content
  -> native probability restore
```

没有 crop 掉原图内容、re-encode、TTA、ensemble 或 Balanced250
fine-tuning。正式保存 1024 logits、native logits、1024 probability、
valid-content probability、native probability 和 official PNG mask。

当前 upstream `infer.py` 对大图会 top-left crop；本 run 遵循论文 v3
附录明确的 long-edge resize + pad，因此命名为 paper-v3 protocol，
不声称与 release script 的 accidental crop bit-exact。

官方 artifact 使用 strict `score > 0.5`；共享 Balanced250 T2 reducer
使用冻结的 `score >= 0.5`。Local GT 是 native decoded-space exact
difference；real GT 是 all-zero。选择为 real 275 加三类 local 各 250，
共 1,025；full-frame 不进入 forward。Bootstrap 为 1,000 次共享
source-content-cluster Poisson resampling。

## 4. 可复现性与机器证据

r2 A/B smoke 各 20 张，100 个数组与 20 个掩码全部 exact，0 mismatch。
Fresh replay 重新加载 checkpoint 并重放全部 1,025 张，比较 5,125 个
数组与 1,025 个掩码，0 mismatch。Replay digest：
`c5590012822b995db1abbadb2a6885260e7617a88c09a18b357f2a69f1558d07`。

- [manifest](../results/opensource/relayformer/relayformer_checkpoint164_balanced250_v1_full1025_r2_20260727/manifest.json) —
  `b67f694d9f15f3459e8303d87fd88cc1c91293b7ce419ec90e4e384be4cfe4e7`
- [expected inputs](../results/opensource/relayformer/relayformer_checkpoint164_balanced250_v1_full1025_r2_20260727/expected_inputs.jsonl) —
  `a952b6adcbd5af20ce635a5929c7045350b9f8e6729bae8d9cbf1978050f2cca`
- [per-image results](../results/opensource/relayformer/relayformer_checkpoint164_balanced250_v1_full1025_r2_20260727/results.jsonl) —
  `ea5eb7e0e445cebf136e420bbcd9652f65e1de6663adfe18578ef4020b8023ad`
- [coverage summary](../results/opensource/relayformer/relayformer_checkpoint164_balanced250_v1_full1025_r2_20260727/summary.json) —
  `7d4add7841e4afd8db01763df445f70054ff847cd74daf7ed9080daad8e58edd`
- [metrics](../results/opensource/relayformer/relayformer_checkpoint164_balanced250_v1_full1025_r2_20260727/metrics.json) —
  `f3849b4a056dffe56423f3e40a8536ff479aca1d606cde5f6062593678cca147`
- [fresh replay audit](../results/opensource/relayformer/relayformer_checkpoint164_balanced250_v1_full1025_r2_20260727/independent_audit.json) —
  `ddcf1569b0a91e34bcf72c2e8138efa24e9084017712dd1e309c47c3fd35bf80`
- [A/B smoke comparison](../results/opensource/relayformer/relayformer_checkpoint164_balanced250_v1_smoke5x4_a_r2_20260727__vs__relayformer_checkpoint164_balanced250_v1_smoke5x4_b_r2_20260727_comparison.json) —
  `8bf85c560158a138313cc98efedd2548ababa0859bdcca8030d5f828307f5d92`

Dataset deterministic contract：
`671d1739bebf4370d26b4629ca26b56cc546a817d469ba505cc39bda8b33102c`。
本次重建 manifest SHA 是
`5685071fb752cded8ddf8841b8fb80547c9f5d046d180f3ad7dd9faa728cc15e`；
inputs、panel、source-pairs 三个 ledger SHA 均与冻结 release 一致。

## 5. r1 fail-closed 记录

首轮完整推理之后，共享 analyzer 因 callback 参数顺序错误在第一张结果
fail closed。修复后重新执行 smoke、formal、analysis 和 full replay，
形成 source fingerprint 与提交代码一致的 r2。r1/r2 inference arrays、
masks 和统计值一致；只发布 r2。
