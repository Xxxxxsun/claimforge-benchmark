# Vision Banana / SAM 3 用于生成裁块拼回的调研

**日期：** 2026-07-21

**范围：** `compose_spliced_full.py` 中 `paste_back: false` 的局部生成裁块拼回；重点是 cat/mouse 等有明确语义的新增对象。

**结论：** 不等待或复刻 Vision Banana。第一版应采用 **SAM 3 语义 mask + 原图/生成图 residual 扩张 + 窄边缘羽化** 的混合方案，并保留现有差分方案作为对照基线和显式回退路径。

## 1. 决策摘要

| 方案 | 当前可用性 | 对本仓库的适配度 | 建议 |
|---|---:|---:|---|
| 现有固定阈值差分 | 已实现 | 中；对背景偏色、细边缘和阴影不稳 | 保留为 baseline |
| Vision Banana | 无公开代码、权重或 API | 理念相关，但不能部署 | 不进入关键路径 |
| 普通生成模型直接“画 mask” | 可试验 | 输出随机、颜色和空间对齐无保证 | 仅作为研究 ablation |
| SAM 3 语义 mask | 官方代码和权重可用，权重需授权 | 高；支持文本和 box prompt | 作为主体 mask |
| SAM 3 + residual 混合 | 需要少量工程实现 | 最高；同时覆盖主体与接触阴影 | 推荐方案 |

这里要区分两个目标：

1. Vision Banana 证明“生成模型经过专门对齐后可以输出可解码的视觉结果”；
2. 本仓库实际需要的是稳定、可复现的新增对象 alpha/mask，而不是生成一张看起来像分割图的 RGB 图片。

两者相关，但不是同一个问题。Vision Banana 在通用分割 benchmark 上的结果也不能直接证明它更适合识别“生成 crop 相比原 crop 新增了哪些像素”。本任务拥有一项通用分割 benchmark 没有的强监督信号：**同尺寸、配准好的原始 crop 与生成 crop 的逐像素 residual**。应充分利用这个信号，而不是完全交给语义模型。

## 2. 当前拼回机制和失效原因

当前实现位于 [`compose_spliced_full.py`](../compose_spliced_full.py)。当生成 manifest 中 `paste_back` 为 `false` 时，`object_mask()` 执行：

1. 计算原 crop 和生成 crop 的最大通道绝对差；
2. 使用固定阈值 `d > object_thr`，默认阈值为 30；
3. 对二值图执行 opening；
4. 只保留与 edit box 相交的最大连通分量；
5. closing、fill holes、膨胀和 Gaussian feather；
6. 如果有效像素少于 6，退回整块 feathered edit box。

这套方法的优点是简单、完全离线、对明显且连通的插入对象有效，也能保证 context box 外的源图像素保持不变。但它有几类结构性问题：

- **固定阈值不适应场景。** 暗色猫落在暗色地面上时，真实主体可能低于阈值；生成模型造成的全局色调漂移又可能高于阈值。
- **opening 会损失细结构。** 毛发、耳尖、尾巴、腿和胡须容易被腐蚀。
- **单一最大连通分量假设过强。** 分离的腿、尾部、软阴影或反射可能被丢弃。
- **edit box 只能作为锚点，不能代表真实对象边界。** 即使使用 `--object-search context`，不与 edit box 连通或相交的合理变化仍可能被忽略。
- **形态学 closing/fill/dilation 不能恢复真实 alpha。** 它可能填入猫腿之间的背景，也可能把邻近背景一起粘上。
- **整 box 回退风险最高。** subtle edit 会退回矩形区域，从而重新引入生成 crop 的背景偏色和矩形接缝。
- **缺少 mask 审计产物。** 当前 manifest 记录阈值和模式，但没有保存最终 object mask、模型分数、候选数和失败原因，后续难以复现和审核。

因此，当前问题并不是再寻找一个更复杂的全局阈值，而是增加一个与像素差互补的**语义先验**。

## 3. Vision Banana 到底做了什么

[Vision Banana 项目页](https://vision-banana.github.io/)和[技术报告](https://arxiv.org/abs/2604.20329)描述的方法是：

- 以闭源的 Nano Banana Pro 为基础；
- 混入少量视觉任务数据进行 instruction tuning；
- 把语义分割、实例分割、指代表达分割、深度和法线等任务统一表示为 RGB 图片生成；
- 对分割任务，通过 prompt 指定类别和颜色，再按与目标颜色的距离聚类生成像素，解码出 mask；
- 对实例分割，每次只请求一个类别，让模型为不同实例分配不同颜色，再做颜色聚类。

关键点是：**这不是只靠 prompt 就能稳定复现的方法。** 技术报告明确把稳定、可测量的 RGB 输出归因于对 Nano Banana Pro 的任务指令微调。仅向普通图片生成模型输入“把猫画成纯黄色、背景画成黑色”，可能得到漂亮的可视化，但无法保证：

- 输出颜色严格可解码；
- mask 与输入逐像素配准；
- 多次调用结果确定；
- 细边缘不被生成模型重新绘制；
- 不凭空补全、删除或移动对象。

截至 2026-07-21，官方项目页只链接技术报告和展示结果，没有发布 Vision Banana 的代码、checkpoint、模型卡或 API。因此它目前不能成为可复现 benchmark 的生产依赖。

### 3.1 能否复刻其思路

理论上可以在开源图片生成/编辑模型上复刻“视觉任务输出也是 RGB 图片”的训练范式，但这会变成一个独立研究项目：

1. 构建图片、文本指令、精确 RGB mask 三元组；
2. 对生成模型做 LoRA 或全量 instruction tuning；
3. 约束颜色、尺寸和几何配准；
4. 设计颜色聚类解码器和拒绝策略；
5. 在本仓库场景上重新评价边界、漏分、幻觉和随机性。

它比部署专用分割模型复杂得多，而且本仓库只需要 object mask，不需要一个统一处理深度、法线和生成任务的通用模型。现阶段投入产出比不合适。

可以把“普通生成模型直接生成 mask 图”保留为非关键路径 ablation，用来验证 Vision Banana 的训练对齐是否真的必要；不能把它用于生成最终 benchmark 样本。

## 4. 为什么选择 SAM 3

[Meta 官方 SAM 3 仓库](https://github.com/facebookresearch/sam3)提供推理代码、训练代码和示例。SAM 3：

- 支持短文本、点、box、mask 和图像 exemplar；
- 可以返回所有匹配开放词汇概念的实例 mask、box 和 score；
- 支持图片批量推理；
- 模型为 848M 参数；
- 当前仓库也发布了 SAM 3.1 checkpoint；3.1 的主要改进偏向多对象视频追踪，本项目的单图 mask 不依赖其 Object Multiplex 特性。

[Hugging Face 模型页](https://huggingface.co/facebook/sam3)还提供 `transformers` 的 `Sam3Model` / `Sam3Processor` 接口，支持文本、单 box、正负 box、组合 prompt 和批处理。权重是 gated 的：需要接受条款、共享联系信息并登录下载。代码和权重使用 [SAM License](https://github.com/facebookresearch/sam3/blob/main/LICENSE)，不是 MIT/Apache；论文中使用时还需按许可证要求确认致谢和再分发规则。

SAM 3 对 cat/mouse 这类有清晰名词概念的对象很合适。但它不会天然把“猫的接触阴影”“猫引起的反射”都视为猫的一部分，因此不能完全替代 residual。

对于 stain 这类不规则、材质化、边界模糊的编辑，语义 mask 的收益预计较小，应继续以 residual 为主，把 SAM 3 仅作为弱先验或不使用。

## 5. 推荐的混合 mask 算法

### 5.1 输入

每个 task 已经提供所需信息：

- 原图：`source_image`；
- 生成 crop：generated manifest 的 `output_crop`；
- context box：`context_region_xyxy`；
- 空间锚点：`edit_region_in_context_xyxy`；
- 语义词：`candidates`，cat 任务当前为 `cat`。

SAM 3 应在**生成 crop**上运行，而不是在最终大图上运行。crop 更小、目标占比更高，而且坐标可以直接用于现有 composite。

### 5.2 候选 mask 选择

文本 prompt 会返回零个、一个或多个实例。不能简单选最高置信度，因为场景里可能本来就有相同类别。对每个候选 mask `M_i`，使用以下信号联合排序：

- 与 edit box 的交叠或中心距离；
- mask 内原图/生成图 residual 的强度和覆盖率；
- mask 面积是否落在合理范围；
- SAM 3 confidence；
- 是否存在明显高 residual 但与候选主体完全无关的区域。

核心原则是：**语义判断“这是不是猫”，paired residual 判断“这是不是这次新加的猫”。**

如果没有候选通过最低门槛，应输出 `needs_review` 或显式的 `diff_fallback`，不能静默退回整个 edit box。

### 5.3 主体与附属变化融合

定义：

- `M_sem`：选中的 SAM 3 主体 mask；
- `D`：原 crop 与生成 crop 的颜色 residual，可从 max-RGB diff 升级为局部背景归一化的 Lab/亮度色度差；
- `M_res`：自适应阈值得到的可信变化像素；
- `M_near`：与 `M_sem` 或其小范围膨胀相接触的 residual 分量。

推荐组合：

```text
主体 core       = M_sem
附属变化 support = M_res 中与 dilate(M_sem) 接触、且靠近 edit box 的分量
最终硬 mask      = core ∪ support
最终 alpha       = 对最终硬 mask 做 1–2 px 的窄边缘软化
```

这样可以：

- 用语义 mask 保住低对比毛发、腿和内部区域；
- 用 residual 补入接触阴影、反射和 SAM 未覆盖的生成变化；
- 用 paired residual 排除原场景中已有的同类对象；
- 避免把整个生成 crop 的背景色漂移贴回去。

不建议默认使用大范围 Gaussian feather 或 Poisson blending。CLAIMFORGE 需要明确、可审核的局部改动范围；过度融合会增加不必要的修改像素，并模糊最终 ground-truth mask 的定义。

## 6. 仓库接口设计

建议把分割和拼回拆成两个可缓存阶段，而不是在 composer 内部即时加载大模型。

### 阶段 A：批量生成语义 mask

新增脚本建议命名为：

```text
scripts/generate_sam3_object_masks.py
```

输入：

```text
annotations/cat_generation_tasks.jsonl
generated_crops/<generation_model>/manifest.jsonl
```

输出放到新的命名空间，避免与现有矩形 insert mask 混淆：

```text
generated_object_masks/<generation_model>__sam3/<task_id>.png
generated_object_masks/<generation_model>__sam3/manifest.jsonl
```

mask manifest 至少记录：

```json
{
  "task_id": "cat_restaurant_000_slot_001",
  "generated_crop": "generated_crops/...png",
  "object_mask": "generated_object_masks/...png",
  "prompt": "cat",
  "model": "facebook/sam3",
  "model_revision": "<pinned revision>",
  "candidate_count": 1,
  "selected_score": 0.91,
  "selection_features": {
    "edit_box_overlap": 0.72,
    "changed_pixel_fraction": 0.84
  },
  "status": "ok"
}
```

所有模型 revision、threshold 和后处理参数都必须落盘；mask PNG 也要保留，不能只保存最终 composite。

### 阶段 B：混合并拼回

扩展 `compose_spliced_full.py`，建议增加：

```text
--blend hybrid
--semantic-manifest generated_object_masks/.../manifest.jsonl
--semantic-dilate <pixels>
--residual-threshold <value or adaptive mode>
--failure-policy review|diff
```

输出 manifest 进一步记录：

- semantic mask 路径；
- residual mask 路径或其参数；
- 最终 alpha mask 路径；
- 语义面积、support 面积和最终面积；
- 使用了 `hybrid`、`diff_fallback` 还是 `needs_review`；
- context box 外像素一致性校验结果。

## 7. SAM 3 部署复杂度

### 7.1 结论

在当前 8×80GB Hunyuan 服务器上，SAM 3 模型跑通约为 **3/10** 难度，做成可审计的混合拼回流水线约为 **6/10**。

它不需要 vLLM、tensor parallel、NCCL 服务或 8 卡常驻。单卡直接 PyTorch 离线批推理即可。80GB 显存对 848M 参数模型有充足余量；实际峰值仍应在第一批 crop 上记录，而不是写死估计值。

官方当前安装前提是：

- Python 3.12+；
- PyTorch 2.7+；
- CUDA 12.6+；
- Hugging Face checkpoint 访问权限；
- `flash-attn-3` 和 `cc_torch` 只是可选加速项，不是基础推理必需。

### 7.2 必须隔离环境

当前 Hunyuan 环境由 [`scripts/setup_hunyuan_vllm.sh`](../scripts/setup_hunyuan_vllm.sh)固定到 vLLM 0.24.0 / vLLM-Omni 0.24.1，并应用本地 Distil compatibility patch；[`scripts/start_hunyuan_vllm.sh`](../scripts/start_hunyuan_vllm.sh)按 8-GPU 配置启动服务。

SAM 3 不应安装进这个 venv。推荐单独使用：

```text
/root/sam3/.venv
/root/models/sam3
```

或一个独立容器。最稳妥的调度方式是：

1. Hunyuan 完成 crop 生成；
2. 停止或释放生成服务占用的 GPU；
3. SAM 3 单卡批量生成 mask；
4. CPU 执行 residual 融合与 composite。

基础部署按官方仓库执行即可：

```bash
conda create -n sam3 python=3.12
conda activate sam3
pip install torch==2.10.0 torchvision --index-url https://download.pytorch.org/whl/cu128
git clone https://github.com/facebookresearch/sam3.git /root/sam3-src
pip install -e /root/sam3-src
hf auth login
```

正式流水线要把 SAM 3 仓库 commit、checkpoint revision 和依赖锁定；上面的版本只是 2026-07-21 官方 README 给出的起点，不能长期依赖浮动的 `main`。

若只做最小 smoke test，也可以直接使用 Hugging Face Transformers 接口。正式实现更倾向官方仓库并 pin commit，方便与官方 notebook 行为保持一致。

## 8. 最小验证计划

不建议直接重跑 272 张。先固定 20–30 个困难样本，覆盖：

- 暗色主体与暗色背景；
- 浅色主体与纹理复杂背景；
- 尾巴、耳尖或腿超出 edit box；
- 明显接触阴影；
- crop 内原本已有相似动物或物体；
- 生成 crop 有整体亮度/色偏；
- 当前算法触发整 box fallback 的样本。

在同一组样本上比较：

1. 当前 `object` diff；
2. SAM 3 semantic-only；
3. SAM 3 + residual hybrid。

至少保存和检查：

- 原 crop、生成 crop；
- raw residual；
- SAM 候选 mask；
- 选中主体 mask；
- residual support；
- 最终 alpha；
- 拼回全图和边界放大图。

建议评价指标：

- 小规模人工 object-mask IoU / boundary F-score；
- 背景泄漏率；
- 主体缺失率；
- 阴影保留率；
- context box 外 exact pixel equality；
- mask 面积异常率和失败/回退率；
- 盲评的接缝与真实性通过率。

只有 hybrid 在困难集上明确优于现有 diff，才扩展到 272 张并更新正式 composite。试验使用的样本集合、参数和模型 revision 应在运行前冻结，避免看结果后逐样本调阈值。

## 9. 风险和停止条件

- **权重访问：** SAM 3 checkpoint 需要 Hugging Face 授权；申请未通过前只能完成适配代码，不能完成真实推理验证。
- **许可证：** SAM License 允许的使用和分发方式需要项目负责人确认，尤其是模型材料再分发；不要把 checkpoint 提交进本仓库。
- **语义误选：** crop 内已有同类对象时，必须由 residual 和 edit-box anchor 共同选实例。
- **小图上采样：** 当前 crop 约百余像素，SAM 3 会预处理到模型分辨率；边界可能被插值平滑，最终必须回到原 crop 尺寸评价。
- **阴影不属于对象：** semantic-only 结果不应直接作为最终 alpha。
- **不规则 stain：** 若 pilot 证明 SAM 3 对材质型编辑无增益，应停止在 stain 上投入，继续改进 paired residual。
- **benchmark 污染：** 不允许人工逐图修 mask 后又把它描述为自动流水线结果；人工修订必须单独标注和报告。

## 10. 推荐执行顺序

1. 获取 SAM 3 权重权限并在独立环境跑通一个 cat crop；
2. 实现只读生成 `generated_object_masks/...` 的批量脚本；
3. 冻结 20–30 张困难 pilot 和参数；
4. 实现 hybrid composer，同时保留 current-diff baseline；
5. 生成可视化 contact sheet 和量化报告；
6. 通过 pilot 后再决定是否重做 272 张正式拼回图。

预计工程量：权重可访问后，单图 smoke test 约 1 小时；批量 mask 和 manifest 接入约半天到一天；困难样本上的混合策略、审计输出和质量验证约再需一到两天。这里最大的未知数是质量调优，不是模型部署。

## 参考资料

- Vision Banana project: <https://vision-banana.github.io/>
- Gabeur et al., *Image Generators are Generalist Vision Learners*: <https://arxiv.org/abs/2604.20329>
- Meta SAM 3 official repository: <https://github.com/facebookresearch/sam3>
- SAM 3.1 release notes: <https://github.com/facebookresearch/sam3/blob/main/RELEASE_SAM3p1.md>
- SAM 3 Hugging Face model card and gated checkpoints: <https://huggingface.co/facebook/sam3>
- Hugging Face Transformers SAM 3 documentation: <https://huggingface.co/docs/transformers/model_doc/sam3>
- SAM License: <https://github.com/facebookresearch/sam3/blob/main/LICENSE>
