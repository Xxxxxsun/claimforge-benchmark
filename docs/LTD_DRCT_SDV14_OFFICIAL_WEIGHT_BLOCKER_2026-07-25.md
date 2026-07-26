# LTD / `DRCT_sdv1.4.pth` 官方权重获取阻塞报告

> 复核日期：2026-07-25（UTC）
> 方法：Layer Consistency Matters / Latent Transition Discrepancy（LTD）
> 代码基线：官方仓库 [`yywencs/LTD`](https://github.com/yywencs/LTD)，固定提交 [`27a8a7e6acd97c1b50b584f85dcca47c1584614b`](https://github.com/yywencs/LTD/commit/27a8a7e6acd97c1b50b584f85dcca47c1584614b)
> 目标权重：`DRCT_sdv1.4.pth`（README 中的 **DRCT-2M** checkpoint）
> 当前状态：**BLOCKED — 两个官方入口都未能返回 checkpoint 字节**
> 本轮执行状态：**没有启动 CUDA、没有生成 Mouse 分数、没有使用第三方镜像**

## 1. 结论

截至 2026-07-25，本次只从官方 README 发布的两个入口重试：

1. Google Drive 分享页可看到文件名，但直接下载端点返回 1,789-byte HTML 错误页。错误页明确表示文件 owner 没有授予下载权限，当前只有 owner/editor 可以下载。
2. 百度网盘提取码有效，匿名会话可进入分享页，并可读取 `DRCT_sdv1.4.pth`、文件大小 `1,862,028,941` bytes 和 `fs_id`；但实际下载要求登录态以及页面生成的下载签名。匿名调用 `/api/sharedownload` 没有得到 `dlink`，只得到 `errno=2` 或 `errno=113`。

所以本轮没有获得 checkpoint，不能合法、诚实地声称 LTD 已在 Mouse 上运行。当前正确状态是 **官方权重访问阻塞**，而不是模型失败、环境失败或显存失败。

要解除阻塞，需要以下任意一种官方路径：

- 作者把 Google Drive 文件改成允许 viewer 下载；
- 用户自行登录百度网盘，从 README 的官方分享链接下载该文件，再把文件放入本项目约定的权重目录；不要把账号、Cookie 或登录 token 交给自动化脚本；
- 作者重新发布可下载的官方链接，并最好同时发布 checkpoint 的 SHA-256 与权重许可证。

拿到文件后，第一步必须是 **CPU-only 隔离校验并立即汇报**；本报告对应的本轮任务不授权直接启动 CUDA。

## 2. 官方来源与版本身份

### 2.1 论文

- 官方 CVF 页面：[Layer Consistency Matters: Elegant Latent Transition Discrepancy for Generalizable Synthetic Image Detection](https://openaccess.thecvf.com/content/CVPR2026/html/Yang_Layer_Consistency_Matters_Elegant_Latent_Transition_Discrepancy_for_Generalizable_Synthetic_CVPR_2026_paper.html)
- 官方 CVF PDF：[CVPR 2026 paper](https://openaccess.thecvf.com/content/CVPR2026/papers/Yang_Layer_Consistency_Matters_Elegant_Latent_Transition_Discrepancy_for_Generalizable_Synthetic_CVPR_2026_paper.pdf)
- 作者：Yawen Yang、Feng Li、Shuqi Kong、Yunfeng Diao、Xinjian Gao、Zenglin Shi、Meng Wang
- CVPR 2026，pp. 38111–38121
- 本次取到的官方 PDF：`6,830,206` bytes
- PDF SHA-256：`e16094a549015528709c84175a82140fbcbfdbd213f57622ede69772eb5c8477`

### 2.2 代码

- 官方仓库：<https://github.com/yywencs/LTD>
- 固定提交：`27a8a7e6acd97c1b50b584f85dcca47c1584614b`
- 提交标题：`feat:first init`
- 提交时间：2026-05-01 19:52:02 +08:00
- 本次 checkout 后工作树干净；该提交共有 35 个 tracked files。

后续复现实验不得跟随仓库浮动的默认分支，必须继续固定上面的完整 commit。关键固定文件：

- [README.md](https://github.com/yywencs/LTD/blob/27a8a7e6acd97c1b50b584f85dcca47c1584614b/README.md)
- [test.sh](https://github.com/yywencs/LTD/blob/27a8a7e6acd97c1b50b584f85dcca47c1584614b/test.sh)
- [validate.py](https://github.com/yywencs/LTD/blob/27a8a7e6acd97c1b50b584f85dcca47c1584614b/validate.py)
- [models/clip_models.py](https://github.com/yywencs/LTD/blob/27a8a7e6acd97c1b50b584f85dcca47c1584614b/models/clip_models.py)
- [dataset_paths.py](https://github.com/yywencs/LTD/blob/27a8a7e6acd97c1b50b584f85dcca47c1584614b/dataset_paths.py)
- [networks/base_model.py](https://github.com/yywencs/LTD/blob/27a8a7e6acd97c1b50b584f85dcca47c1584614b/networks/base_model.py)

## 3. 目标 checkpoint 身份

官方 README 的 pretrained-model 表给出两个不同训练来源的权重。当前目标严格限定为第二行 **DRCT-2M**，不能拿第一行 UFD/GenImage 权重替代。

| 字段 | 固定值 |
|---|---|
| README 条目 | `DRCT-2M` |
| 文件名 | `DRCT_sdv1.4.pth` |
| Google Drive 官方链接 | <https://drive.google.com/file/d/1203ng6Kj9f2LK5UnMZFKBrTX-HpkwqB_/view?usp=sharing> |
| Google file id | `1203ng6Kj9f2LK5UnMZFKBrTX-HpkwqB_` |
| 百度网盘官方链接 | <https://pan.baidu.com/s/1DUr_iVKKrRePpah34gJbzg?pwd=6tkx> |
| 百度提取码 | `6tkx` |
| 百度页面报告大小 | `1,862,028,941` bytes（约 1.862 GB / 1.734 GiB） |
| 百度 `fs_id` | `1107940297858445` |
| 作者发布的 checkpoint SHA-256 | **未提供** |
| 百度页面 `md5` 字段 | 空字符串 |

Google 分享页标题和百度解锁后的分享页都指向同一个文件名。由于作者没有公布 checkpoint digest，当前不能仅凭相同文件名认证第三方副本；这也是本次不访问第三方镜像的一个独立原因。

本机约定缓存路径和本轮临时目录中均未发现精确文件名 `DRCT_sdv1.4.pth`。

## 4. 官方入口重试结果

### 4.1 Google Drive

| 尝试 | 返回 | 判定 |
|---|---|---|
| README 的 `/file/d/.../view` 分享页 | HTTP 200，HTML 登录/预览页 | 不是 checkpoint |
| `drive.google.com/uc?export=download&id=...`，跟随跳转并请求前 1 MiB | 303 跳转到 `drive.usercontent.google.com`，最终 HTTP 200、`text/html; charset=utf-8`、1,789 bytes | 不是 checkpoint |
| `drive.usercontent.google.com/download?...&confirm=t` | HTTP 200、HTML、1,789 bytes | 不是 checkpoint |

最终 HTML 的页面标题是 `Google Drive - Can't download file`，核心提示为：

> Sorry, the owner hasn't given you permission to download this file.

因此这里的 HTTP 200 只表示错误页成功返回，不能当作下载成功。两次 direct-download body 的 SHA-256 分别为：

- `6532eb8cb8133f32c18039f1e0ea9fb626d123ea9b07c016a0fde89acf2c117e`
- `9fe96855f82d4acf90146738114b5be31a6c53640bbbe5ffc14d7fbca2e91a02`

两个 body 内容含动态 nonce，所以 digest 不同，但都只有 1,789 bytes，且都包含同一 owner-permission 错误。

### 4.2 百度网盘

| 尝试 | 返回 | 判定 |
|---|---|---|
| 打开 README 官方链接和公开提取码 | HTTP 200；跳转到 `/share/init?surl=...&pwd=6tkx`；15,058-byte HTML | 分享入口可用 |
| 向官方 `/share/verify` 提交 `pwd=6tkx` | HTTP 200；JSON `errno=0`；建立匿名 `BDCLND` 会话 | 提取码有效 |
| 携带匿名会话重新打开分享页 | HTTP 200；76,061-byte HTML；显示文件名、大小和 `fs_id` | 元数据可读，但仍为 `loginstate=0` |
| 调用官方 `/api/sharedownload`，没有可用签名 | HTTP 200；JSON `errno=2`，`show_msg=请求失败` | 没有 `dlink` |
| 加页面时间戳、空 `sign` 再调用 | HTTP 200；JSON `errno=113`，`show_msg=验证码签名错误` | 没有 `dlink` |

解锁后的页面状态同时显示：

- `loginstate=0`
- `bdstoken=""`
- `public=0`
- checkpoint `md5=""`

为确认阻塞不是端点猜错，本次还检查了该页面实际加载的百度官方静态脚本。脚本中：

- 分享下载端点明确为 `/api/sharedownload`；
- 请求需要页面上下文中的 `sign` 和 `timestamp`；
- 普通下载分支检查 `currentProduct === "share"` 且没有 `loginstate` 时，会调用百度登录 UI 并终止当前下载。

本次取到的官方脚本 URL：

<https://nd-static.bdstatic.com/m-static/function-widget-1/pkg/download-all_a27c9a2.js>

该脚本为 `107,313` bytes，SHA-256 为：

`e0c3cb902eebbd4e9b1e8cae809f671fbf7ec8d7ef43774822ffb6e75a047f0e`

这说明公开提取码只解锁了分享信息；它没有把匿名会话变成可取得文件字节的登录下载会话。

百度证据 body 的 SHA-256：

| body | bytes | SHA-256 |
|---|---:|---|
| 初始分享页 | 15,058 | `6ff714d09dec09185f8ce6467081a1f477557c935fe895dcee58fe80a36a5829` |
| `/share/verify` JSON | 118 | `5115d8fbb59cb66723b2af595016782031f13791135a9e889e3a73658111ffd7` |
| 解锁后的分享页 | 76,061 | `79b39b13cd87432e8abcfb715bc58b9c4f03205672b204d3e3c75fc21b0970e5` |
| `/api/sharedownload`：无签名 | 81 | `beae6414cfbec8b293f6af32d8e00d46edfdeedc5d7b187d633a3326db01c28a` |
| `/api/sharedownload`：空签名 + 时间戳 | 126 | `41ee955018f1d7dfac8dce2952867b265817895af608375bf14b288cbb862bcc` |

`BDCLND`、`randsk` 等会话值没有写进本报告：它们是临时会话凭据，不是模型文件，也不应作为可共享的复现材料。

## 5. 可复核命令

以下命令只访问论文、GitHub、Google Drive 和百度网盘的官方地址。输出目录应使用新的临时目录；命令不会启动 Python 模型或 CUDA。

### 5.1 固定官方代码与论文

```bash
retry_dir="$(mktemp -d)"

git clone --filter=blob:none https://github.com/yywencs/LTD.git \
  "$retry_dir/LTD"
git -C "$retry_dir/LTD" checkout \
  27a8a7e6acd97c1b50b584f85dcca47c1584614b
git -C "$retry_dir/LTD" rev-parse HEAD
git -C "$retry_dir/LTD" status --short

curl -fL \
  'https://openaccess.thecvf.com/content/CVPR2026/papers/Yang_Layer_Consistency_Matters_Elegant_Latent_Transition_Discrepancy_for_Generalizable_Synthetic_CVPR_2026_paper.pdf' \
  -o "$retry_dir/ltd_cvpr2026.pdf"
sha256sum "$retry_dir/ltd_cvpr2026.pdf"
```

### 5.2 Google Drive 两种官方 direct-download 形式

```bash
file_id='1203ng6Kj9f2LK5UnMZFKBrTX-HpkwqB_'

curl -sS -L --range 0-1048575 \
  -D "$retry_dir/drive_uc.headers" \
  -o "$retry_dir/drive_uc.body" \
  "https://drive.google.com/uc?export=download&id=${file_id}"

curl -sS -L --range 0-1048575 \
  -D "$retry_dir/drive_usercontent.headers" \
  -o "$retry_dir/drive_usercontent.body" \
  "https://drive.usercontent.google.com/download?id=${file_id}&export=download&confirm=t"

wc -c "$retry_dir"/drive_*.body
file "$retry_dir"/drive_*.body
sha256sum "$retry_dir"/drive_*.body
rg -n 'owner|permission|download' "$retry_dir"/drive_*.body
```

预期阻塞证据是 body 类型为 HTML、大小 1,789 bytes，并出现 owner-permission 提示；不是 `.pth` 字节。

### 5.3 百度网盘官方分享页与提取码验证

```bash
baidu_cookie_jar="$retry_dir/baidu.cookies"

curl -sS -L \
  -c "$baidu_cookie_jar" \
  -b "$baidu_cookie_jar" \
  -D "$retry_dir/baidu_share.headers" \
  -o "$retry_dir/baidu_share.body" \
  'https://pan.baidu.com/s/1DUr_iVKKrRePpah34gJbzg?pwd=6tkx'

curl -sS \
  -c "$baidu_cookie_jar" \
  -b "$baidu_cookie_jar" \
  -D "$retry_dir/baidu_verify.headers" \
  -o "$retry_dir/baidu_verify.body" \
  -H 'Content-Type: application/x-www-form-urlencoded; charset=UTF-8' \
  --data 'pwd=6tkx&vcode=&vcode_str=' \
  'https://pan.baidu.com/share/verify?shareid=62752323144&uk=1065112316&channel=chunlei&web=1&app_id=250528&bdstoken=&clienttype=0'

curl -sS -L \
  -c "$baidu_cookie_jar" \
  -b "$baidu_cookie_jar" \
  -o "$retry_dir/baidu_after_verify.body" \
  'https://pan.baidu.com/s/1DUr_iVKKrRePpah34gJbzg?pwd=6tkx'

wc -c "$retry_dir"/baidu_{share,verify,after_verify}.body
rg -o 'DRCT_sdv1\.4\.pth|1862028941|1107940297858445|loginstate[^,}]+' \
  "$retry_dir/baidu_after_verify.body"
```

`/api/sharedownload` 的复核使用解锁页公开给匿名会话的 `share_uk=1065112316`、`shareid=62752323144`、`fs_id=1107940297858445`，以及当前匿名 `BDCLND`。实际两次请求等价于：

```text
POST https://pan.baidu.com/api/sharedownload
POST https://pan.baidu.com/api/sharedownload?sign=&timestamp=1785016055

encrypt=0
product=share
uk=1065112316
primaryid=62752323144
fid_list=[1107940297858445]
extra={"sekey":"<URL-decoded BDCLND from this anonymous verify session>"}
```

第一条返回 `errno=2`，第二条返回 `errno=113`。不应把临时 Cookie 粘贴到报告、issue 或聊天中，也不应尝试绕过百度的登录/签名机制。

## 6. LTD 的原理与发布实现

论文的核心观察是：在冻结的 CLIP ViT 中，真实图像的相邻中层表征转换相对稳定，而合成图像更容易在层间出现突变。LTD 不只读最终层，而是同时建模“当前层特征”和“相邻层变化量”。

固定提交中的推理路径如下：

1. 加载 OpenAI CLIP `ViT-L/14`，并调用 `requires_grad_(False)` 冻结 CLIP backbone。
2. hook visual Transformer 的 block index `11..19`，共取得 9 个中层 CLS 特征，每个宽度 1,024。
3. `select_k=5` 时，9 层形成 5 个候选连续窗口；`LayerSelector` 为每个候选窗口学习一个 logit。
4. 对选中的 5 个原始层特征建立 origin branch。
5. 计算 4 个相邻差分

   \[
   d^{(k)} = f^{(k+1)} - f^{(k)}
   \]

   并建立 delta branch。
6. 两个 branch 分别加自己的 learned positional embedding 和 learned CLS token，但复用同一个单层 Transformer encoder：

   - `d_model=1024`
   - `nhead=8`
   - `dim_feedforward=4096`
   - `dropout=0.3`
   - `activation=gelu`

7. 取两个 branch 输出的 CLS，各 1,024 维；拼成 2,048 维，经 `LayerNorm(2048)` 与 `Linear(2048, 1)` 输出一个 logit。

这种设计强的地方不在于额外训练一个巨大视觉 backbone，而在于利用已训练 CLIP 的中层表征，并显式比较层间转换规律。论文报告的 DRCT-2M 训练配置使用 236k 张 SD v1.4 合成图像、80 个 MSCOCO 类别；backbone 冻结，学习的主要是层选择器、两条分支的 token/position、共享 encoder 和分类头。

但需要严格限定能力范围：

- LTD 输出一个整图标量，是 **T1 whole-image synthetic-image detection**。
- 它不输出 mask、box 或 pixel map，不能作为 **T2 局部篡改定位** 方法。
- Mouse 的局部植入图仍可作为“整图是否含 AI 合成内容”的 T1 正样本测试，但 LTD 无法告诉我们植入区域在哪里。

## 7. 输入预处理与分数语义

### 7.1 官方实现

测试数据读取：

```text
PIL.Image.open(path).convert("RGB")
→ optional Resize((256, 256))
→ CenterCrop(224)
→ ToTensor()
→ Normalize(
    mean=[0.48145466, 0.4578275, 0.40821073],
    std =[0.26862954, 0.26130258, 0.27577711]
  )
```

这里有一个必须显式冻结的发布差异：

- 论文写的是所有输入先 resize 到 `256×256`，再 center crop 到 `224×224`。
- `dataset_paths.py` 给 DRCT 条目设置 `is_resize=True`。
- 但发布版 `validate.py` 把 `dataset_paths` 硬编码成 UFD，而 UFD/GenImage 条目设置 `is_resize=False`。
- `clip.load()` 返回的 CLIP preprocess 在 `CLIPModel` 中明确没有用于这里的 Dataset 推理。

因此，使用 **DRCT-2M / `DRCT_sdv1.4.pth`** 测 Mouse 时，主协议冻结为论文及 DRCT 分支：

```text
RGB
→ torchvision 0.16.1 Resize((256, 256))，PIL 默认 bilinear
→ CenterCrop(224)
→ ToTensor
→ 官方 CLIP mean/std
```

不得在看到结果后改成 CLIP 自带 shortest-side resize、letterbox、原尺寸直裁或 UFD 的 no-resize 路径。也不启用 JPEG、blur、resize robustness 参数。

`256→224` 的 center crop 会从每侧去掉 16 pixels，即每个边缘去掉输入宽/高的 6.25%，保留 87.5% 的宽和高。Mouse 跑分前必须利用既有 mask/box 元数据报告植入区域在裁剪后：

- 完全可见；
- 部分可见；
- 完全被裁掉。

该可见性只用于分层分析，不能作为丢弃困难样本的理由。

### 7.2 分数

- checkpoint 载入路径：`torch.load(ckpt, map_location="cpu")["model"]`
- `load_state_dict` 为默认 strict 行为
- 模型输出：单个 logit
- 对外分数：`score_fake = sigmoid(logit)`
- 标签：real/original = `0`，fake/edited = `1`
- 方向：分数越高，模型越认为图像为 fake
- 官方固定阈值：严格使用 `score_fake > 0.5`

发布版还会在测试标签上寻找 best threshold。这个数使用了测试集真值，不可作为 ClaimForge 主结果。主结果只允许预先固定的 0.5 阈值和无阈值排名指标；test-set oracle threshold 如保留，只能清楚标成附录诊断。

## 8. 必须冻结的随机层选择问题

固定提交存在一个不应被掩盖的推理随机性：

1. `LayerSelector.forward()` 使用 `F.gumbel_softmax(..., tau=1.0, hard=True)`。
2. `_update_tau()` 调用被注释，所以 temperature 不会按训练/测试状态切换。
3. `CLIPModel` 虽收到 `training=False`，构建 selector 时仍硬编码 `training=True`。
4. `model.eval()` 不会关闭 Gumbel sampling。

因此同一图像的 score 不是天然的纯函数：它会受 RNG 状态、样本顺序和 batch 划分影响。未经声明地把 Gumbel 改成 argmax 会偏离发布模型；未经固定地保留随机采样又会让结果不可审计。

### 8.1 主结果：`LTD-DRCT-released-seeded`

Mouse 主结果冻结为发布代码忠实模式：

- 模型数学不改，保留 hard Gumbel-Softmax；
- `select_k=5`；
- batch size `256`，与官方 `test.sh` 相同；
- `shuffle=False`；
- 进程启动后、推理前只调用一次官方 seed 逻辑，seed 固定为 `0`；
- 输入顺序固定为：全部 275 张 original/real，按 canonical `pair_rank` 升序；随后全部 275 张 edited/fake，按相同 `pair_rank` 升序；
- 不允许单张失败后插入式 retry，因为这会移动后续 RNG stream；
- 中断时优先从头重跑；若实现 checkpoint-resume，必须同时保存并恢复 Python、NumPy、CPU Torch 和 CUDA RNG state；
- 从全新进程、seed=0 完整执行两次；逐图 score 最大绝对差要求 `≤1e-6`。不满足时状态记为 nondeterministic，不能把两次结果平均后假装成唯一官方分数；
- 记录每个样本实际抽中的 5-layer 窗口，但只做无副作用日志，不改变 forward。

这个协议能让“整条固定顺序的官方 evaluator stream”可重复，但仍应在结果报告中披露其 order/batch dependence。

### 8.2 单独诊断：`LTD-DRCT-argmax-adapter`

可以预注册一个确定性敏感性诊断：用 learned selector logits 的 argmax 选择唯一窗口，不采 Gumbel noise。它必须：

- 使用独立方法名 `LTD-DRCT-argmax-adapter`；
- 保留同一 checkpoint、输入预处理、encoder 和 head；
- 不替代或混入 released-seeded 主结果；
- 与主结果并列表明这是 benchmark adapter，而非作者发布 evaluator 的原样输出。

在 checkpoint 到手前，这两种模式都不能产生分数；当前只完成协议冻结。

## 9. Mouse 预冻结评测合同

拿到并通过 checkpoint 校验后，最小适配器只允许做两件事：

1. 把既有 canonical Mouse manifest 提供给官方模型；
2. 保存逐图 logit、`score_fake`、标签、pair id、pair rank、抽中的 layer window 与错误状态。

不允许改网络、重训、微调、校准、对测试图做新增增强，或把 mask/box 输入模型。

### 9.1 样本与覆盖

- 使用本仓库已经冻结的 canonical Mouse 275 pairs / 550 images；
- 每对包含同源 original/real 与 edited/fake；
- 文件内容 hash、manifest 版本和 pair 顺序必须写入 run manifest；
- 报告成功数、失败数、失败原因和 coverage；
- 任何 decode/OOM/NaN 失败都保留在审计表，不能静默丢弃；
- 单独按植入面积、裁剪后可见性、来源域等既定字段分层。

### 9.2 主指标

- image-level AUROC；
- image-level AP，fake 为正类；
- threshold=0.5 的 accuracy、real accuracy、fake accuracy；
- TPR@5% FPR；
- paired delta：`score_fake(edited) - score_fake(original)`；
- paired win/tie/loss；
- exact sign test；
- 以 pair 为重采样单元的 bootstrap 95% CI。

不得在 Mouse 测试标签上拟合阈值、反转方向或选择“表现最好”的 selector 模式。

### 9.3 官方基础模型依赖

固定代码会从 OpenAI 官方地址取得 CLIP ViT-L/14：

<https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt>

URL 中声明的预期 SHA-256 为：

`b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836`

这只是 CLIP backbone digest，**不是** `DRCT_sdv1.4.pth` digest，不能用它替代 LTD checkpoint 校验。

## 10. 权重到手后的 CPU-only 接收门

正式跑分前按以下顺序处理：

1. 把下载文件放进无机密、默认断网的隔离目录；PyTorch `.pth` 可能包含 pickle，不应在有凭据的通用环境中直接反序列化。
2. 记录来源 URL、下载时间、精确 byte size 和 SHA-256。
3. 预期大小先与百度官方页面的 `1,862,028,941` bytes 比较；若不一致，停止。
4. 检查文件类型，拒绝 HTML、JSON、网盘错误页和稀疏占位文件。
5. 在 CPU-only 隔离环境使用发布依赖版本读取：

   - Python 3.9.23
   - torch 2.1.1
   - torchvision 0.16.1
   - numpy 1.26.4
   - OpenCV 4.8.0.74
   - Pillow 11.3.0
   - transformers 4.30.2

6. 确认顶层是 mapping，且存在 `model`；官方保存格式还包含 `optimizer` 和 `total_steps`。
7. 构建 `CLIP:ViT-L/14, num_classes=1, select_k=5`，在 CPU 上 strict load；missing/unexpected keys 必须都为空。
8. 输出参数 key、shape、dtype 汇总与 selector logits；不输出张量内容。
9. 只有上述接收门全部通过，才把状态从 `weight_blocked` 改成 `ready_for_authorized_gpu_run`。

作者没有发布 SHA-256，所以首次成功取得的官方文件 digest 应进入内部 artifact registry，但不能虚构成“作者声明的 hash”。

## 11. 许可证与商用边界

固定提交的 35 个 tracked files 中没有 `LICENSE`、`COPYING` 或 `NOTICE`；README 也没有代码许可证、checkpoint 许可证或商用授权文字。

因此：

- “官方公开了源码”不等于已经获得 OSI 开源许可证；
- CVF 论文页面的论文版权声明不构成代码或模型权重许可证；
- README 对 UniversalFakeDetect 的 acknowledgement 不会自动把其他项目的许可证传递给 LTD；
- 官方网盘链接可访问或需要登录，都不等于授予再分发、衍生或商业使用权；
- 当前应记录 `commercial_use_cleared=false`；
- 在作者补充许可证或法务确认前，只能把它视为待许可澄清的研究评测候选，不得把代码/权重重新发布进本仓库、公共对象存储或镜像。

本次坚持官方入口不仅是实验 provenance 要求，也是因为没有作者 digest 时，第三方副本无法可靠认证，且其再分发权限未知。

## 12. 为什么不能“先假跑一个结果”

以下做法都会产生错误结论：

- 使用随机初始化：没有学到 selector、encoder、position/CLS token 和 classification head。
- 只用 CLIP backbone：CLIP 的相似度不是 LTD 的 fake score。
- 用 README 的 UFD/GenImage checkpoint 替代：训练来源不同，不能标成 `DRCT_sdv1.4`。
- 自己按论文重训：那是复现模型，不是作者发布 checkpoint，必须用不同方法名报告。
- 从论文表格抄指标：论文表格不是 Mouse 275-pair 数据上的实测结果。
- 把 1,789-byte Google HTML 改名为 `.pth`：这只是权限错误页。
- 使用未经认证的第三方镜像：既违反本轮官方来源约束，也无法在缺少作者 hash 时证明权重相同。
- 猜测或插值 Mouse 分数：模型的 learned selector 与分类头参数未知，分数无法从论文描述推导。

checkpoint 保存的是完整 `model.state_dict()`；即使 CLIP backbone 冻结，它仍会被写入 state dict，这也解释了约 1.86 GB 的文件规模。没有该文件，就没有足够信息计算作者模型的 Mouse logit。

## 13. 本轮核心证据摘要

| 对象 | SHA-256 |
|---|---|
| 官方 CVPR PDF | `e16094a549015528709c84175a82140fbcbfdbd213f57622ede69772eb5c8477` |
| pinned `README.md` | `5103f550f045160f3def15a3b7945741b0b2703484fcff73372d9ecf25136f66` |
| pinned `test.sh` | `a162fb231feb20d8b68a3691668fe9aa3996e978f66d56ce74743f23ed01574d` |
| pinned `validate.py` | `512d03f187e9135412d0cc3c080918bed91920d0b3a360f017d5b4c82d317b76` |
| pinned `models/clip_models.py` | `8e505a7e670a85efb670ca8e8df454f7c5e8d0a4e4548b1063dafe958ee67589` |
| pinned `dataset_paths.py` | `3c3b4449b4705d3f69123a3ac757e02f1ee2e9cbba68ebf93db230fad3b80861` |
| pinned `networks/base_model.py` | `8b73d8d8d77c4a19df8b3e40628618723095d50b1c68f3460e1e2eeb04cd76d1` |
| pinned `requirements.txt` | `b0315c6142ed677c4ea73b76da82c78073ab1d508bf2ca481571b6ebbd943d5b` |

最终审计状态：

```text
method=LTD
code_commit=27a8a7e6acd97c1b50b584f85dcca47c1584614b
checkpoint=DRCT_sdv1.4.pth
checkpoint_source_policy=official_only
google_drive=blocked_by_owner_download_permission
baidu_share=metadata_visible_but_download_requires_login_and_signature
checkpoint_bytes_acquired=false
checkpoint_sha256=unknown
cuda_started=false
mouse_scores_emitted=false
commercial_use_cleared=false
next_state=await_official_weight_access
```
