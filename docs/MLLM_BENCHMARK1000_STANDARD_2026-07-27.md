# MLLM Benchmark1000 聚合标准（2026-07-27）

## 定稿口径

后续 MLLM 主结果统一使用固定的 1,000 张图片：

| 条件 | 数量 | 选择方式 |
|---|---:|---|
| Local Mouse | 250 | 与最终 Trash-can-250 使用相同源图单元 |
| Local Cat | 250 | 从最终 Cat-251 中按固定随机种子排除 1 张 |
| Local Trash can | 250 | 最终 Trash-can-250 全部保留 |
| Real | 250 | 此前测过且文件 SHA-256 不重复的真实图 |
| 合计 | 1,000 | 750 forged + 250 real |

正式 dataset ID：

```text
claimforge-mllm-local750-real250-v1
```

推理时仍然允许跑上游全集：Mouse-275、Cat-251、Trash-can-250 和
Real-275，共 1,051 张。正式聚合时只按固定 ledger 过滤，因此不用
为节省额外 51 张推理而改动已有断点续跑逻辑。

固定的推理全集清单为：

```text
annotations/claimforge_mllm_inference_full1051_v1.jsonl
```

每行通过 `metadata.in_benchmark1000` 标识是否属于正式 1,000 张。
该字段只用于聚合过滤，不会出现在发送给模型的 prompt 中。

## Cat-250

Cat 只来自：

```text
spliced_final/claimforge_cat_selected_251_20260725
```

不使用任何模型输出选择样本。为了让“随机排除一张”可复现，构建器对
每个排序后的 Cat task ID 计算：

```text
SHA256(seed + NUL + task_id)
```

并排除哈希值最小的一项。固定 seed 为：

```text
claimforge-mllm-local750-real250-v1::cat-drop::20260727
```

本版本排除：

```text
cat_restaurant_295_slot_001
```

因此 Cat 抽样不依赖 GPT、Claude、Qwen 或 Doubao 的检测结果。

## Mouse-250 与 Trash-can-250

最终 Trash-can-250 定义 Mouse 的源图筛选范围。每个 Trash-can
scene ID 必须在人工审核 `status=good` 的 Mouse-275 中找到对应
Mouse，并验证两者原始源文件 SHA-256 相同。

这一步得到：

- Mouse：250；
- Trash can：250；
- 从 Mouse-275 排除：25。

这里不要求 Cat 与 Mouse/Trash-can 使用相同源图；Cat 是独立随机
Cat-250 面板。

## Real-250

Real 候选来自 Mouse-good275 的 source image，并用以下历史 run
证明它们此前已作为 real 输入完成检测：

```text
results/mllm/qwen3_7_plus/
qwen37plus_pilot_good275_c15_v3_20260715T153257_0800.jsonl
```

选择顺序：

1. 优先加入 Trash-can-250 对应的不同源图内容；
2. Trash-can-250 含 4 组重复源内容，因此先得到 246 个不同
   SHA-256；
3. 再从 Mouse-good275 的剩余已测 real 中补 4 个不同 SHA-256；
4. 最终 Real-250 的内容 SHA-256 全部唯一。

Real 的文件路径、SHA-256 和历史 run 证据都记录在 ledger/manifest
中。

## 唯一聚合清单

正式文件：

```text
annotations/claimforge_mllm_benchmark1000_v1.jsonl
annotations/claimforge_mllm_benchmark1000_v1.manifest.json
```

JSONL 每行至少记录：

- `benchmark_id` / `task_id` / `scene_id`；
- `image_path` / `image_sha256` / 图片尺寸；
- `label` 与 `candidate`；
- 原图路径与 SHA-256；
- forged 的 GT edit/context box；
- 选择来源和 dataset ID。

Manifest 绑定所有上游文件的哈希、Cat 随机 seed、被排除 Cat、Real
历史测评证据、四类计数，以及正式 ledger 的 SHA-256。修改 JSONL
后如果不重新构建 manifest，runner 会拒绝加载。

重建命令：

```bash
conda run -n utils python scripts/build_mllm_benchmark1000.py
```

## 推理与聚合使用方式

普通 MLLM：

```bash
conda run -n utils python -m eval.mllm.run_mllm \
  --config config/mllm_eval.example.json \
  --source benchmark1000 \
  --protocol both \
  --condition benchmark1000 \
  --run-id <unique-run-id>
```

Zoom agent（正式默认最多五次 zoom）：

```bash
conda run -n utils python -m eval.mllm.run_zoom_agent \
  --config config/mllm_eval.example.json \
  --source benchmark1000 \
  --condition benchmark1000_zoom5 \
  --run-id <unique-run-id>
```

当前 Full1051 zoom run 使用：

```bash
conda run -n utils python -m eval.mllm.run_zoom_agent \
  --config config/mllm_eval.example.json \
  --source list \
  --list annotations/claimforge_mllm_inference_full1051_v1.jsonl \
  --condition zoom_agent_full1051_v1 \
  --replicates 3 \
  --concurrency 15 \
  --max-zoom-calls 5 \
  --run-id <model-specific-run-id>
```

该运行覆盖全部 1,051 张，但主表仍只聚合其中
`metadata.in_benchmark1000=true` 的 1,000 张。

`--source benchmark1000` 会在任何 API 请求前验证：

- dataset ID；
- manifest/ledger SHA-256 绑定；
- 总数 1,000；
- forged/real 为 750/250；
- Mouse/Cat/Trash-can/Real 各 250；
- 1,000 个 benchmark ID 唯一；
- 每行 dataset ID 一致。

已有全量结果不需要重跑。聚合程序可按 ledger 中的
`image_sha256`，或按 `task_id + candidate` 映射，只统计这 1,000
张；多跑的 Cat 1 张和 Mouse 25 张保持在原始结果文件中，但不进入
正式表。

## Zoom 历史数据

此前 zoom run 的结果、crop 和 mask 已清空。协议代码保留，新的
zoom 全量实验应绑定本 Benchmark1000，不与旧 pilot 聚合。
