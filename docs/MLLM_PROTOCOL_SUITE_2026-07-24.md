# MLLM 2026-07-24 协议套件

## 1. 冻结版本

2026-07-24 版本把两种既有协议合并为一次可执行的 suite，但不修改任何
prompt 或输出 schema：

| Suite/协议 | 冻结版本 |
|---|---|
| Combined suite | `mllm_protocol_suite_20260724` |
| Detection | `mllm_protocol_v3_reasoning_image_coordinates` |
| Localization | `mllm_protocol_v4_reasoning_pixel_coordinates` |

调用 `--protocol both` 时，每张图片分别建立 detection 和 localization
单元，每个单元仍要求三次有效独立调用。模型按照配置文件中的顺序串行
运行；单个模型内部由 `--concurrency` 控制请求并发。

## 2. 结果与 run manifest

同一模型的一次 combined run 使用一组文件：

```text
results/mllm/<model>/<run_id>.raw.jsonl
results/mllm/<model>/<run_id>.jsonl
results/mllm/<model>/<run_id>.run_manifest.json
```

raw 和聚合 JSONL 中的 detection/localization 记录通过 `protocol_key`
区分。每条记录保存：

- `protocol_suite_version=mllm_protocol_suite_20260724`；
- 与该记录对应的 leaf `protocol_version`；
- `run_id`、输入 manifest hash 和 config fingerprint。

run manifest 的协议部分为：

```json
{
  "version": "mllm_protocol_suite_20260724",
  "suite_version": "mllm_protocol_suite_20260724",
  "keys": ["detection", "localization"],
  "versions": {
    "detection": "mllm_protocol_v3_reasoning_image_coordinates",
    "localization": "mllm_protocol_v4_reasoning_pixel_coordinates"
  },
  "replicates_required": 3
}
```

## 3. 断点续跑

combined run 不使用 suite version 直接筛 replicate，而是按协议分别
匹配 leaf version：

```text
(id, protocol_key, expected_leaf_protocol_version, replicate_index)
```

因此 detection v3 与 localization v4 可以写在同一 run 文件中，同时
排除同一文件里可能存在的旧协议行。已有单协议 run 保持兼容，仍可用
原 run ID 续跑。

run manifest 冻结输入图片 hash。后续新增图片时不要修改已启动 run 的
输入清单；应建立带新 run ID 的 supplement 清单，最终评测时再合并主
run 与 supplement run。这样可以准确追踪每张图片实际使用的配置。

## 4. 运行示例

```bash
python -m eval.mllm.run_mllm \
  --config config/mllm_eval.example.json \
  --source list \
  --list config/images.local.jsonl \
  --protocol both \
  --condition selected_partial \
  --run-id selected_partial_suite0724_20260725 \
  --model-slug qwen3_7_plus \
  --concurrency 15 \
  --retry-until-complete
```

不传 `--write-metrics` 时，只写 raw、三次有效结果的聚合记录和 run
manifest，不生成 metrics。需要评测时可在完整数据补齐后统一执行。
