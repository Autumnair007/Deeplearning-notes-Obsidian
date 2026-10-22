---
type: code-note
tags:
  - cv
  - semantic-segmentation
  - open-vocabulary-segmentation
  - clip
  - dinov3
  - neural-ode
model: DINOde
status: in-progress
---

# DINOde：从冻结特征到连续对齐的源码学习

论文：[DINOde: Continuous Vision-Text Alignment for Open-Vocabulary Semantic Segmentation](https://arxiv.org/abs/2607.21371)；官方仓库：[yoon307/DINOde](https://github.com/yoon307/DINOde)。

配套阅读：[[DINOde_Translation]] 保存完整译文，[[DINOde_paper_notes]] 解释方法动机与论文公式。本页聚焦已经实际阅读的官方源码；代码链接固定到所读版本，避免以后分支更新后找不到对应实现。

核心文件：[model/dinode.py](https://github.com/yoon307/DINOde/blob/0a2c5182c44107fdd8c786b2192fcc3c5e5bebd1/model/dinode.py)、[model/components.py](https://github.com/yoon307/DINOde/blob/0a2c5182c44107fdd8c786b2192fcc3c5e5bebd1/model/components.py)。[eval.py](https://github.com/yoon307/DINOde/blob/0a2c5182c44107fdd8c786b2192fcc3c5e5bebd1/eval.py) 只用于确认类别文本怎样进入模型、分数怎样恢复到图像空间。

可复用算子：[[Cross_Modal_Alignment]]、[[Pooling_and_Region_Aggregation]]、[[Contrastive_Regularization]]。本文展示的“源码节选”保留实际计算，允许省略注释或不相关分支；标注“教学改写”的代码才是便于迁移的简化版本。数字 shape 示例用于理解，不代表另一套实验配置。

## 1. 先把整条 Method 走通

DINOde 的代码可以先理解成：**保留 DINOv3 已经学好的空间特征，训练一个小网络把 CLIP 文本搬进这个视觉空间。** 搬运不是调用一次 MLP 就结束，而是反复询问同一个速度网络：“现在位于这个特征点、处于这个时间，应往哪个方向走一点？”训练时再用图像的局部汇总和全局 CLS 两种信息，约束移动后的文本。

这里有两条流。语义文本流 STF（Semantic Text Flow）处理文本；全局上下文流 GCF（Global Context Flow）处理 CLS。两者使用同一种速度网络结构，但拥有独立参数。GCF 没有把 CLS 送进 STF 充当条件，STF 的输入只有文本当前状态与时间。因此同一类别的文本锚点可以被所有测试图像复用。

```text
训练：
图像 → 冻结 DINOv3 ┬→ patch 特征 → 逐位置归一化 → min-max k 池化 → p
                  └→ CLS → 两层投影 MLP → GCF → loss 前归一化 → c
该图的多条 caption → 冻结 CLIP → 先归一化各条、再平均
                   → 文本线性投影 → STF → t

图像向量 [p, c] 与文本向量 [t, stop_gradient(t)]
                   → 批内 B×B 相似度 → 双向对比交叉熵

推理：
类别名称 → 提示模板 → CLIP → 模板平均 → STF → 类别锚点
测试图像 → DINOv3 patch 特征 → 与每个锚点点积
                   → [B,类别数,H_patch,W_patch] → 插值 → 像素分类
```

训练用 caption 配对提供图像级监督；基础分割推理没有 GCF、池化或图像级拼接。推理直接使用未被 GCF 改动的密集 patch 特征。不能把训练时的 `[B,2048]` 联合向量误当成每个 patch 的分类特征。

### 从哪几个函数开始读

| 论文中的步骤 | 源码入口 | 主要输入 → 输出 |
|---|---|---|
| 冻结视觉特征提取 | `components.py::DinoV3HFBackbone.forward_grid` | 图像 → patch 网格、原始 CLS |
| 文本编码与聚合 | `components.py::CLIPTextEncoder.encode` | 多条文本 → CLIP 向量或平均向量 |
| 时间条件速度 | `dinode.py::TimeConditionedTextFlowNet.forward` | 当前状态、时间 → 同维速度 |
| STF | `TextCondHead.apply_text_flow` | `[K,768] → [K,1024]` |
| GCF | `TextCondHead.apply_cls_flow` | `[B,1024] → [B,1024]` |
| 局部特征汇总 | `min_max_k_pooling` | `[B,D,H_p,W_p] → [B,D]` |
| 文本拼接与停止梯度 | `FlowTrainer._encode_texts_captions_batch` | `[B,768] → [B,2D]`，这里假设所有样本都有 caption |
| 联合监督 | `FlowTrainer.train_step`、`clip_style_contrastive_loss` | 两组 `[B,2D] → [B,B] → loss` |
| 密集类别得分 | `TextCondHead.forward` | patch 网格、类别锚点 → 类别得分网格 |

以下用 `B` 表示图像数，`K` 表示送入某次文本流的文本数，`M` 表示推理类别数，`D=1024` 表示 DINO 特征维，`E=768` 表示 CLIP 文本维，`N=H_pW_p` 表示 patch 数。源码部分位置用 `C` 表示特征通道，不要与类别数混淆。

## 2. 冻结特征怎样变成两个可用接口

### 2.1 DINO 输出的 token 不能直接全部 reshape 成图像

直觉是：CLS 是“整图摘要”，register 是辅助 token，只有后面的 patch token 才有二维位置。先区分它们，再恢复网格。

源码节选：[components.py::forward_grid](https://github.com/yoon307/DINOde/blob/0a2c5182c44107fdd8c786b2192fcc3c5e5bebd1/model/components.py#L434-L483)。

```python
@torch.no_grad()
def forward_grid(self, pixel_values):
    # 省略移动设备等代码
    output = self.model(pixel_values=pixel_values)
    last_hidden_state = output.last_hidden_state
    B, N, C = last_hidden_state.shape
    num_register_tokens = getattr(self.model.config, "num_register_tokens", 0)
    patch_size = getattr(self.model.config, "patch_size", 16)
    H_in, W_in = pixel_values.shape[-2:]
    H_p, W_p = H_in // patch_size, W_in // patch_size
    cls_token = last_hidden_state[:, 0, :]
    patch_tokens = last_hidden_state[:, 1 + num_register_tokens:, :]
    patch_features = patch_tokens.transpose(1, 2).reshape(B, C, H_p, W_p)
    return patch_features, cls_token
```

分三步看：

1. 原始序列是 `[B,1+R+H_pW_p,D]`，`R` 是 register 数量。`[:,0,:]` 取每张图的第一个 token，得到 `[B,D]`；第二个切片跳过 CLS 和全部 register。
2. patch 原本是 `[B,N,D]`。`transpose(1,2)` 只交换位置轴和通道轴，得到 `[B,D,N]`，不会把不同 patch 混合。
3. `reshape(B,D,H_p,W_p)` 用原有的栅格顺序拆开位置轴。这里从输入尺寸计算高宽，允许长方形网格，不需要假定 `sqrt(N)` 一定是边长。`reshape` 也比直接对转置结果调用 `view` 更适合可能不连续的布局。

例如两张 `224×224` 图像、patch 边长为 16、假设模型有 4 个 register，序列是 `[2,201,1024]`。跳过前 5 个 token 后是 `[2,196,1024]`，恢复为 `[2,1024,14,14]`；CLS 另外保留为 `[2,1024]`。

**值得迁移的写法：** 包装视觉骨干时同时返回“空间网格”和“全局向量”，让后续分支各取所需。换一个 ViT 时，需要确认特殊 token 的排列、patch 大小和 token 的空间顺序；不能沿用“只丢第一个 token”的假设。

### 2.2 冻结、eval 和不记录梯度分别做什么

这里的冻结具有明确的代码边界。CLIP 构造时调用 `eval()`，并对参数执行 `requires_grad_(False)`；`encode` 还带有 `@torch.no_grad()`。DINO 包装器把模型设为 `eval()`，`forward_grid` 带有 `@torch.no_grad()`，而 `FlowTrainer` 的优化参数只收集投影和 flow 分支。因此在这条实际训练路径中，两个大编码器不会被对齐 loss 更新。

`eval()` 改变 Dropout、BatchNorm 等模块的训练行为，本身不关闭自动求导；`no_grad()` 控制本次计算是否记录图；参数是否交给优化器决定哪些参数被更新。这三件事不能只用一句“调用了 eval，所以冻结了”代替。

尤其注意梯度边界的位置：CLIP 编码放在 `no_grad()` 中，**之后的 `text_proj(T_clip)` 在边界之外**。固定输入特征不需要梯度，小投影器的参数仍然可以正常学习。固定特征可以预先缓存，缓存取代的是编码器计算，不是可训练的 STF 结果。

### 2.3 多个模板或 caption 怎样聚合

源码节选：[components.py::CLIPTextEncoder.encode](https://github.com/yoon307/DINOde/blob/0a2c5182c44107fdd8c786b2192fcc3c5e5bebd1/model/components.py#L332-L347)。

```python
tokens = self.tokenizer(prompts).to(self.device)
text_embeddings = self.model.encode_text(tokens)
text_embeddings = l2norm(text_embeddings)
if aggregate == "mean":
    return text_embeddings.mean(0, keepdim=True)
else:
    return text_embeddings
```

假设某类别有 `P=3` 条提示，CLIP 输出 `[3,768]`。`l2norm` 默认沿最后一维归一化，即每条文本分别除以自己的范数；`mean(0,keepdim=True)` 沿提示条数求平均，输出 `[1,768]`，768 个通道都保留。

`keepdim=True` 让结果继续保留“有一行向量”的二维接口；调用者再 `squeeze(0)` 得到 `[768]`，多个类别用 `torch.stack(...,dim=0)` 组成 `[M,768]`。训练时，同一张图的多条 caption 也先在 CLIP 空间平均，然后才送入 STF；不是让每条 caption 分别经过流后再平均。

平均单位向量通常不是单位向量。例如 `(1,0)` 与 `(0,1)` 的均值 `(0.5,0.5)` 范数是约 `0.707`。源码的 `encode(...,"mean")` 没有再次归一化这个均值；后面的 STF 初始投影输出才会归一化。阅读缓存时，要明确缓存的是哪一个阶段的向量。

**迁移思路：** 对多个描述先平均，可以把可变数量文本变成每张图一行固定维度特征，便于批量训练。但它压掉了描述间的差异，不能等同于多正样本对比学习；而且非线性映射通常满足 `flow(mean(T)) != mean(flow(T))`，聚合的位置属于模型设计。

## 3. STF 与 GCF 的起点：先对齐维度，再学习运动

大白话说，CLIP 文本和 DINO 特征不在同一个坐标系里。先准备一个能进入 DINO 维度的起点，随后再由 flow 修正它的位置。CLS 本来已经是 1024 维，但它承担的全局语义与 patch 不完全相同，所以也有可学习的初始投影。

源码节选：[dinode.py::TextCondHead.__init__](https://github.com/yoon307/DINOde/blob/0a2c5182c44107fdd8c786b2192fcc3c5e5bebd1/model/dinode.py#L295-L305)。

```python
self.text_flow_init = nn.Linear(768, in_channels, bias=False)
self.cls_flow_init = nn.Sequential(
    nn.Linear(in_channels, in_channels),
    nn.GELU(),
    nn.Linear(in_channels, in_channels),
)
```

STF 的 `nn.Linear(768,1024,bias=False)` 对最后一维做线性变换，权重在 PyTorch 中存为 `[1024,768]`。所以 `[K,768]` 相当于乘上权重转置 `[768,1024]`，得到 `[K,1024]`；它不会在不同文本之间交换信息。GCF 则是 `1024→1024→1024` 的两层 MLP，中间有 GELU，两层线性层使用默认偏置。

分别进入流之前执行：

```python
# apply_text_flow 中的真实计算
Z = self.text_flow_init(T_clip)
Z = l2norm(Z)

# apply_cls_flow 中的真实计算
Z = l2norm(self.cls_flow_init(cls_token_raw))
```

`l2norm` 的实现是 `x / (x.norm(p=2,dim=-1,keepdim=True) + eps)`。输入 `[K,D]` 的范数是 `[K,1]`，广播为每一行的 D 个通道使用同一个分母。它约束整行向量的长度，不是把每个通道的数值缩放到 `[0,1]`。

例如文本 `[3,768]` 与 CLS `[2,1024]` 分别得到 `[3,1024]` 与 `[2,1024]`，可以复用同一种速度网络类。**相同的类与 shape 不意味着共享权重**：`text_flow_net`、`cls_flow_net` 是分别实例化的两个网络。

> [!note] 论文符号与源码粒度
> 论文把 GCF 起点写成 $W_{cls}$ 投影；这里实际是两层非线性 MLP。理解方法时可以把它看成一个投影头，自己写实现时则不能机械抄成单个线性层。STF 的初始投影才是这里展示的单个无偏置线性层。

## 4. 时间正弦编码：把一个数变成可调制网络的向量

同一个位置在轨迹早期和晚期可能需要不同速度。因此网络不仅看当前 `Z`，还需要知道“走到了哪一时刻”。正弦编码先把标量时间展开成一组不同变化速率的分量，再用小 MLP 学习如何使用它们。

源码节选：[dinode.py::sinusoidal_time_embedding](https://github.com/yoon307/DINOde/blob/0a2c5182c44107fdd8c786b2192fcc3c5e5bebd1/model/dinode.py#L38-L55)。

```python
t = t.view(-1)
half_dim = dim // 2
freqs = torch.exp(
    -math.log(10000.0)
    * torch.arange(half_dim, device=t.device, dtype=t.dtype)
    / half_dim
)
args = t[:, None] * freqs[None, :]
embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
```

逐步拆开广播：

1. `t.view(-1)` 把时间整理成 `[K]`。速度网络会先把 Python 标量转为 `[K]`：`torch.full((K,),t,device=z.device,dtype=z.dtype)`；零维 tensor 则用 `unsqueeze(0).expand(K)` 扩展。所有样本可以共享同一个时间，也可分别给定时间。
2. 当 `dim=256` 时，`half_dim=128`，`arange(128)` 产生 0 到 127。指数计算得到 128 个频率，首个为 1，之后按几何比例减小。这里没有额外乘 `2π`，也没有把时间乘到 1000；不能自动套用其他扩散模型的频率约定。
3. `t[:,None]` 是 `[K,1]`，`freqs[None,:]` 是 `[1,128]`。逐元素相乘时，两侧长度为 1 的轴广播，得到 `[K,128]`。每一行表示一个时间与全部频率的乘积。
4. `sin` 与 `cos` 各自产生 `[K,128]`，沿最后一维拼接后成为 `[K,256]`。接下来的 `time_mlp` 是 `256→256→256`，中间使用 GELU，学习供 FiLM 使用的时间表示。

一个可以手算的例子：令时间维度为 4，则 `half_dim=2`，频率是 `[1,0.01]`。时间 `t=0` 编码为 `[0,0,1,1]`；`t=0.5` 编码约为 `[0.4794,0.0050,0.8776,0.99999]`。不同频率让同一时间产生不同尺度的变化；代码在 `t=0` 也有非零的余弦分量。

**迁移思路：** 这种编码适用于迭代细化、连续深度或时间条件网络。需要一起检查时间取值范围、频率范围以及投影层的输入尺寸。此实现返回 `2*(dim//2)` 维，所以复用时应使用偶数 `time_dim`，或自行补齐奇数维；不要只把构造参数改成奇数而保持后续 `Linear(time_dim,...)` 不变。

## 5. FiLM 残差速度网络：时间怎样真正影响每个通道

直觉上，MLP 根据当前位置提出一个内部变换，时间再告诉它哪些通道该增强、减弱或平移。FiLM（Feature-wise Linear Modulation，逐特征线性调制）就是对每个通道使用一个缩放和偏移，而不是在 token 之间做 Attention。

源码节选：[dinode.py::TimeConditionedTextFlowNet](https://github.com/yoon307/DINOde/blob/0a2c5182c44107fdd8c786b2192fcc3c5e5bebd1/model/dinode.py#L129-L202)。

```python
# 构造时：每个残差块各有一套参数
self.film_layers = nn.ModuleList([
    nn.Linear(time_dim, dim * 2) for _ in range(depth)
])
self.blocks = nn.ModuleList([
    nn.Sequential(
        nn.Linear(dim, hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, dim)
    ) for _ in range(depth)
])
self.layer_norms = nn.ModuleList([
    nn.LayerNorm(dim) for _ in range(depth)
])
self.tail = nn.Linear(dim, dim, bias=False)
nn.init.zeros_(self.tail.weight)

# forward 中：省略上面的标量时间处理
t_sin = sinusoidal_time_embedding(t, self.time_dim)
t_emb = self.time_mlp(t_sin)
h = z
for block, film, layer_norm in zip(
    self.blocks, self.film_layers, self.layer_norms
):
    film_params = film(t_emb)
    scale, shift = film_params.chunk(2, dim=-1)
    h_block = block(h)
    h_film = (1 + scale) * h_block + shift
    h = h + h_film
    h = layer_norm(h)
v = self.tail(h)
```

以 `K=3,D=1024,time_dim=256,hidden_dim=512` 为例，一块内的数据是：

```text
t_emb [3,256] → Linear → [3,2048] → chunk → scale [3,1024], shift [3,1024]
h [3,1024]   → Linear → [3,512]  → GELU → Linear → h_block [3,1024]
(1+scale)*h_block+shift → [3,1024]
加回 h → LayerNorm → [3,1024]
全部残差块结束 → tail → v [3,1024]
```

`chunk(2,dim=-1)` 将最后一维一分为二，不是产生两个不同 batch。FiLM 乘法是对应通道相乘，没有对维度求和，也没有矩阵乘法：若某通道 `h_block=2,scale=0.1,shift=-0.3`，调制结果就是 `1.1×2-0.3=1.9`。`1+scale` 让 `scale=0` 对应保留原始 MLP 输出的幅度，而不是把它清零；这里只是参数化方式，不表示 FiLM 层本身已经零初始化。

`h+h_film` 是**速度网络内部的残差**，它帮助组合特征。外部的 `Z=Z+dt*V` 才是轨迹状态更新，两个残差不能混为一谈。内部 `LayerNorm(dim)` 对每行最后 D 个通道做均值方差归一化，并带可学习仿射参数；它不保证向量 L2 范数为 1，不能替代球面归一化。

`nn.ModuleList` 把多个块注册为模型子模块，参数才能被 `parameters()`、设备移动和保存权重等机制发现。不同块的权重各自独立；**同一个完整速度网络会被每一个 Euler 步重复调用**。所以网络深度是单次速度预测经过多少残差块，积分步数是整套速度预测重复多少次，两者不是同一个超参数。

最后一层权重全零意味着初始化时 `v=0`，初始轨迹暂时不移动，模型从已经投影好的起点开始学习。第一次反传时，若有有效 loss 梯度，`tail` 可以得到梯度；其前面的内部网络收到的梯度还要经过全零 `tail`，因此一开始受到抑制，待尾层不再为零后逐步参与。这是一种让残差分支从小扰动开始的常见办法。

**迁移思路：** 如果需要“同一个小网络在不同迭代阶段做不同事情”，可以复用时间编码 + FiLM。替换任务时修改状态维 `D`、隐藏维和条件来源即可；条件也可以是噪声水平或其他连续量。若状态改为 `[B,N,D]`，应明确条件是逐图还是逐 token：逐图的 `[B,D]` scale/shift 要先变为 `[B,1,D]` 才能沿 token 维广播，不能直接照搬这里的二维输入。

## 6. VTP 与 Euler：如何把速度变成轨迹

### 6.1 一行切向投影到底减掉了什么

如果特征主要靠方向表达语义，我们希望更新沿球面侧向移动，不要只把向量变长。速度切向投影 VTP（Velocity Tangent Projection）尝试减去速度沿当前位置径向的分量。

源码节选：[dinode.py::apply_text_flow](https://github.com/yoon307/DINOde/blob/0a2c5182c44107fdd8c786b2192fcc3c5e5bebd1/model/dinode.py#L427-L443)。

```python
for step in range(self.text_flow_steps):
    t_val = step * self.text_flow_dt
    V = self.text_flow_net(Z, t_val)
    V = V - (V * Z).sum(dim=-1, keepdim=True) * Z
    Z = Z + self.text_flow_dt * V
Z = l2norm(Z)
return Z
```

输入 `V,Z` 都是 `[K,D]`。`V*Z` 逐元素相乘后仍为 `[K,D]`；沿 `dim=-1` 求和，得到每一行的一次点积 `[K,1]`。保留的长度为 1 的轴广播到 D 个通道，乘 `Z` 后成为 `[K,D]` 的径向分量。最终从原始速度中减去它。

$$
v_{\perp}=v-(v^\top z)z,\qquad z^\top v_{\perp}=0\ \text{当}\ \|z\|_2=1.
$$

**公式解释：** 对每个样本单独处理一个 D 维位置 $z$ 和 D 维速度 $v$。点积沿特征维消去 D 得到标量，再将标量乘回 $z$，形成需要去掉的径向向量。输出仍为 D 维；批量形式是 `[K,D]`。右边的正交关系依赖当前位置为单位向量。

例如 `z=(1,0),v=(2,3)`，点积是 2，投影后是 `(0,3)`。沿 x 轴“把特征拉长”的部分被减掉，只留下沿 y 轴改变方向的部分。这里不存在一个显式构造的 `[D,D]` 投影矩阵，利用点积加广播即可完成，空间开销更小。

### 6.2 外层循环与普通堆叠 MLP 有什么不同

Euler 更新 `Z+dt*V` 用当前点速度近似一小段运动。比如 `dt=0.1`，从 `(1,0)` 出发、投影速度为 `(0,3)`，一步到 `(1,0.3)`。下一步速度网络重新接收这个新位置与新时间，不是把上一步的速度直接重复加上。

当 `steps=10,dt=0.1` 时，速度在 `0,0.1,...,0.9` 这些左端点求值，更新后得到总时长约为 1 的状态。所有步复用 `self.text_flow_net` 的参数，自动求导可以沿展开的这些普通 PyTorch 运算反传；这段实现没有调用专用 ODE 求解器或 adjoint 反传。

### 6.3 最需要看缩进的地方：归一化不在每一步里面

上面 `Z=l2norm(Z)` 的缩进与 `for` 相同，因此 STF 是“起点归一化 → 多次 Euler → 终点归一化”。源码附近的注释提到 Euler 后 retraction，但真实执行位置要看缩进，不能凭注释写成每步重新投回单位球面。

对于初始单位向量与真正切向速度，一步 Euler 后有：

$$
\|z+\Delta t\,v_{\perp}\|_2^2=1+\Delta t^2\|v_{\perp}\|_2^2.
$$

**公式解释：** 这是单个样本更新后的平方范数；特征维在范数求和中消失，输出为标量。由于起点与切向速度正交，交叉项为零，但二阶项仍让长度增大。上面的 `(1,0.3)` 长度约为 `1.044`，并不严格位于单位圆。

下一步若继续直接用未归一化的 `Z`，`V-(V·Z)Z` 也不再是严格的正交投影；一般有 `Z·V_new=(Z·V_old)*(1-||Z||²)`。因此这份实现采用了球面几何启发的更新，但不能据此声称离散中间状态每一步都严格在单位球面上。

GCF 的差别更明显：[apply_cls_flow](https://github.com/yoon307/DINOde/blob/0a2c5182c44107fdd8c786b2192fcc3c5e5bebd1/model/dinode.py#L512-L528) 中，起点归一化，循环内的 `# Z = l2norm(Z)` 被注释，flow 分支返回前也没有最终归一化。随后 `train_step` 在拼接前执行 `F.normalize(cls_token,dim=1)`。也就是说，GCF **用于 loss 的输出**归一化了，但函数内部轨迹没有逐步保持单位长度。

| 位置 | STF | GCF 的 flow 分支 |
|---|---|---|
| 初始投影之后 | L2 归一化 | L2 归一化 |
| 每个 Euler 步之后 | 不归一化 | 归一化语句被注释 |
| flow 函数返回之前 | L2 归一化 | 不额外归一化 |
| loss 使用前 | 调用方也会归一化文本 | `train_step` 明确归一化 CLS |

**迁移时怎么选：** 若自己的设计要求每个离散状态都在球面上，可以每步归一化，或使用球面指数映射；这会改变作者当前代码的数值轨迹，应当当作设计变体。下面是明确做每步回投的教学改写，不能用它代替上面的源码事实。

```python
# 教学改写：每步回投的球面残差流，不是 DINOde 原始循环
def spherical_flow(z, velocity, steps):
    z = F.normalize(z, dim=-1)
    dt = 1.0 / steps  # 假定 steps > 0，总时间固定为 1
    for i in range(steps):
        v = velocity(z, i * dt)
        v = v - (v * z).sum(-1, keepdim=True) * z
        z = F.normalize(z + dt * v, dim=-1)
    return z
```

还有一个与轨迹理解直接相关的入口：[apply_text_flow_with_steps](https://github.com/yoon307/DINOde/blob/0a2c5182c44107fdd8c786b2192fcc3c5e5bebd1/model/dinode.py#L489-L496) 使用 `dt=self.text_flow_dt`，而不是 `dt=1.0/num_steps`。因此它改变 `num_steps` 时也改变累计推进时间：若固定 `dt=0.1`，3 步得到约 `t=0.3` 的状态，10 步才到约 `t=1`。这适合查看同一轨迹的不同阶段，不能解释成“始终到同一终点，只改变求解精度”。

## 7. min-max k pooling：每个通道各自找强弱响应

训练需要把整张 patch 网格压成一个图像级向量，与 caption 对齐。这里并不是挑出 k 个“最像目标”的完整 patch，而是**在每个特征通道内部，分别取空间上最大的 k 个值和最小的 k 个值，再平均这 2k 个数**。

先看输入。`process_visual_features` 返回原始 `A` 与 `Z_map=l2norm_hw(A)`；`l2norm_hw` 沿通道轴 `dim=1` 归一化每个空间位置，得到 `[B,D,H_p,W_p]`。后续池化使用的是这个 `Z_map`，所以是在各 patch 先做向量长度归一化之后，再比较各通道在空间上的响应。

源码节选：[dinode.py::min_max_k_pooling](https://github.com/yoon307/DINOde/blob/0a2c5182c44107fdd8c786b2192fcc3c5e5bebd1/model/dinode.py#L580-L595)。

```python
B, D, H, W = x.shape
x_flat = x.view(B, D, -1)
max_k, _ = torch.topk(x_flat, k, dim=2)
min_k, _ = torch.topk(x_flat, k, dim=2, largest=False)
pooled = torch.cat([max_k, min_k], dim=2)
return pooled.mean(dim=2)
```

逐步读这几行：

1. `view(B,D,-1)` 合并 H、W，仅剩位置轴，`[B,D,H_p,W_p] → [B,D,N]`。源码这里使用 `view`；迁移到可能非连续的输入时，可改用 `flatten(2)` 或合适的 `reshape`。
2. `topk(...,dim=2)` 在 N 个位置中挑最大值，输出值和位置索引，各为 `[B,D,k]`；这里只保留值，索引用 `_` 丢弃。
3. `largest=False` 改为找最小值，仍是 `[B,D,k]`。每个通道独立排序，所以一个通道选第 1 个 patch，另一个通道可能选第 9 个 patch。
4. 沿选择数轴拼成 `[B,D,2k]`，再沿同轴求均值，才得到 `[B,D]`。最后还会执行 `F.normalize(...,dim=1)`，使池化后的整个 D 维向量成为用于相似度的单位方向。

例如输入 `[2,1024,14,14]`、`k=20`，先变成 `[2,1024,196]`，最大和最小值各为 `[2,1024,20]`，拼成 `[2,1024,40]`，最终输出 `[2,1024]`。被消去的是空间位置，1024 个特征通道都保留。

用一个小例子看出它为何不等于全局平均。假设某个通道的 6 个位置是 `[-4,-1,2,3,5,9]`，取 `k=2`：最大两个为 `[9,5]`，最小两个为 `[-4,-1]`，结果为 `(9+5-4-1)/4=2.25`；普通平均则约为 `2.33`。当 `k=1` 时它是 `(max+min)/2`；当 `k=N` 时两组都覆盖全部位置，平均结果又等于普通均值。 这里的数值只演示池化算术，不代表已经逐 patch 归一化的 DINO 特征取值范围。

负值只是该特征通道在某些位置的响应，不应直接叫作“背景概率”。这个向量也不一定对应任何一个真实 patch，它是多个通道各自汇总的结果。取最大和最小值能保留两端响应，但也可能受到异常值影响；它没有显式语义标签或区域掩码。

**迁移思路：** 这种池化适合“已有密集特征，但监督只有整图文本”的汇总接口。至少检查 `1≤k≤N`；当 `2k>N` 时最大、最小集合会重叠，重叠位置在均值中出现两次。若任务需要一个真实区域或共同 patch 集合，应另外定义空间打分并用索引收集完整向量，而不是把逐通道 `topk` 当作区域选择。通用归纳见 [[Pooling_and_Region_Aggregation]]。

## 8. 拼接、detach 与双向 NCE：谁在教谁

### 8.1 先把两种图像摘要拼起来

局部池化向量 `p` 表示 patch 的汇总；GCF 输出 `c` 表示修正后的全局上下文。它们都各自归一化，再拼成一个联合图像向量。文本也要有相同的长度，但它的第二份副本停止梯度。

源码节选：[dinode.py::train_step](https://github.com/yoon307/DINOde/blob/0a2c5182c44107fdd8c786b2192fcc3c5e5bebd1/model/dinode.py#L858-L879) 与 [_encode_texts_captions_batch](https://github.com/yoon307/DINOde/blob/0a2c5182c44107fdd8c786b2192fcc3c5e5bebd1/model/dinode.py#L779-L795)。

```python
# 图像侧，B 张图
_z_img_pooled = F.normalize(
    spatial_pooling(Z_map, method="min_max_k", k=self.head.topk), dim=1
)
_cls_token = F.normalize(cls_token, dim=1)
z_img_with_cls = torch.cat([_z_img_pooled, _cls_token], dim=1)

# 文本侧，下面处理单张图的 caption 平均向量经过 STF 后的结果
feature = text_features_valid[valid_idx]
feature = l2norm(feature.unsqueeze(0)).squeeze(0)
feature_2d = torch.cat([feature, feature.detach()], dim=0)
```

图像两组 `[B,D]` 沿 `dim=1` 拼成 `[B,2D]`；单条文本 `[D]` 沿 `dim=0` 拼成 `[2D]`，最后批量 stack 为 `[B,2D]`。这两处 concat 的轴不同，是因为一处带 batch 轴、另一处已经索引掉 batch 轴。

`detach()` 不改变数值，不等同于把第二份文本置零。前向时两份文本完全相同，反向时第二份不再把梯度送回 STF。忽略用于数值稳定的 epsilon，单个联合匹配分数可展开为：

$$
S_{ij}=\frac{p_i^\top t_j+c_i^\top\operatorname{sg}(t_j)}{\tau}.
$$

**公式解释：** $i$ 索引图像，$j$ 索引文本；$p_i,c_i,t_j$ 都是 D 维单位向量，$\operatorname{sg}$ 表示前向数值不变、反向停止梯度。两个点积各自消去特征维，得到标量，相加并除以温度后成为相似度矩阵 `[B,B]` 的一个元素。这是 `[p_i,c_i]` 与 `[t_j,sg(t_j)]` 点积的代数展开，不是另加了一项独立损失。

特别注意：源码对两半分别归一化，没有再把 `[B,2D]` 整体归一化。因此拼接向量的范数约为 `sqrt(2)`，点积是两项余弦的**和**，不是两项余弦的平均。若迁移时对完整拼接向量再做归一化，相同温度下的 logit 会缩小约一半，会改变对比学习的锐利程度。

梯度关系可以这样读：

- `p` 来源于冻结 DINO 与无参数池化，它作为固定视觉参考，第一项推动 STF 文本向 patch 表征对齐。
- `c` 来源于可训练 GCF；第二项中的文本已经 detach，所以这一项直接推动 GCF 向当前文本方向靠近，不通过第二份文本反向推动 STF。
- 两项共用一个 `S`，一起进入 softmax。因此它们通过同一个竞争概率相互影响；不能把“第二份文本停止梯度”解释成“GCF 与 STF 完全独立训练”。

这是一种可迁移的“联合打分、非对称梯度”写法。适合需要一个分支充当另一分支当前参考的情况，但停止哪一份梯度必须和监督方向对应。相关抽象见 [[Contrastive_Regularization]]。

### 8.2 `B×B` 相似度为什么是图像级对比，而不是分割分类

源码节选：[dinode.py::clip_style_contrastive_loss](https://github.com/yoon307/DINOde/blob/0a2c5182c44107fdd8c786b2192fcc3c5e5bebd1/model/dinode.py#L630-L644)。

```python
logits_ii = (z_img @ z_txt.t()) * temperature
logits_tt = logits_ii.t()
target = torch.arange(z_img.size(0), device=z_img.device)
loss_i2t = F.cross_entropy(logits_ii, target)
loss_t2i = F.cross_entropy(logits_tt, target)
```

调用处传入的 `temperature` 实际是 `1.0/self.head.tau`，即逆温度这个乘法尺度；不要看到名字就理解为再乘上小数 `tau`。调用方最终用 `0.5*(loss_i2t+loss_t2i)` 得到对称损失。

若 `B=3,D=1024`，`z_img` 为 `[3,2048]`，`z_txt.t()` 为 `[2048,3]`，矩阵乘法消去 2048 维，输出 `[3,3]`。第 i 行代表“第 i 张图与当前 batch 三条聚合文本分别有多匹配”；第 j 列代表“第 j 条文本与当前 batch 三张图分别有多匹配”。

`target=[0,1,2]` 表示正样本在对角线。第一次交叉熵逐行让图像找配对文本，第二次对转置矩阵逐行让文本找配对图像。`F.cross_entropy` 接收原始 logits，内部完成 log-softmax，不需要预先手动 softmax。

这里 `B` 是 batch 中图文对的数量，不是开放词汇类别数，也不是 patch 数。因此整图 caption 监督能够训练对齐器，但 loss 本身没有逐像素真值去指定“这个 patch 应是狗”。密集定位主要依靠冻结 DINO 的空间特征，以及文本对齐能否迁移到这些局部特征上。

**迁移时的前提：** 两侧第 i 行必须是同一个图文对；缺失文本时应同步筛选双方并重新建立对角标签。以上教学数据流按全部 caption 有效的常见情况说明，不把源码中可选空文本占位逻辑当成通用的变长批处理模板。若同一 batch 中有多张语义相同的图片，单正样本交叉熵仍会把非对角项当作竞争对象；需要多正样本目标时，应显式改正样本矩阵。

### 8.3 这里的 flow 不是靠默认的速度回归目标训练

源码保留了 Rectified Flow、终点项和往返一致性等变量，但公开主配置 [dinode_coco_stuff.json](https://github.com/yoon307/DINOde/blob/0a2c5182c44107fdd8c786b2192fcc3c5e5bebd1/configs/dinode_coco_stuff.json#L35-L37) 明确使用 `w_nce=1,w_rf=0,w_text_consistency=0`。因此在这套设置中，有效优化目标是上面的图文 NCE；速度网络通过 Euler 轨迹最终输出的对比损失学习。

`FlowTrainer` 构造函数本身的 `w_rf` 默认值与该配置不同，不能只看函数签名判断主实验训练了什么。相关辅助分支即使在前向中被计算，乘零后也不贡献默认目标；本页不把这些代码写成论文主方法额外必需的监督。

## 9. 从类别锚点到像素预测：最后只有一次通道收缩

直觉是：为每个候选类别准备一根已经搬到 DINO 空间的“语义方向”，让每个 patch 看自己最接近哪根方向。类别集合可以变，特征维 D 不变。

[eval.py 的类别文本入口](https://github.com/yoon307/DINOde/blob/0a2c5182c44107fdd8c786b2192fcc3c5e5bebd1/eval.py#L845-L855) 先用 `build_prompts(n,dataset_type)` 为类别生成提示，CLIP 平均编码后 stack，再传给 `trainer._encode_texts(...,cached_clip_embeddings=...)`。后者调用 `head.text_proj`；这个可调用包装器进一步转到 `head.apply_text_flow`。因此 `TextCondHead.forward` 接收的 `text_features` 已经是 `[M,1024]` 的对齐文本，不能再送入原始 `[M,768]` CLIP 特征。

源码节选：[dinode.py::TextCondHead.forward](https://github.com/yoon307/DINOde/blob/0a2c5182c44107fdd8c786b2192fcc3c5e5bebd1/model/dinode.py#L559-L564)。

```python
A_processed, Z_map = self.process_visual_features(A)
logits = torch.einsum('bdhw,kd->bkhw', Z_map, text_features) / self.tau
return logits, A_processed, Z_map
```

`einsum` 的 `d` 同时出现在两个输入，却没有出现在输出，所以沿通道求和。`b,h,w` 来自 patch 网格，`k` 来自类别文本，它们全部保留。等价的逐元素解释是：

$$
L_{b,m,h,w}=\frac1\tau\sum_{d=1}^{D}\widehat A_{b,d,h,w}\,T_{m,d}.
$$

**公式解释：** $\widehat A$ 为每个 patch 沿 D 维归一化后的视觉网格 `[B,D,H_p,W_p]`，$T$ 为对齐并归一化的类别锚点 `[M,D]`。求和消去特征轴 D，输出 `[B,M,H_p,W_p]`；一个元素表示某张图、某个 patch 与某个类别的余弦匹配分数除以温度。这里没有对空间位置求和，每个 patch 的位置都保留。

例如 `[2,1024,14,14]` 与 `[20,1024]` 得到 `[2,20,14,14]`。`logits[0,3,5,6]` 表示第一张图、第 4 个候选类别、网格第 6 行第 7 列的分数。等价的教学矩阵写法更容易看出维度关系：

```python
# 教学改写：与上面的 einsum 对照，不是作者 forward 的逐字复制
patches = Z_map.flatten(2).transpose(1, 2)     # [B,N,D]
scores = patches @ text_features.T            # [B,N,M]
logits = scores.transpose(1, 2).reshape(B, M, H_p, W_p) / tau
```

第一步展平空间、把通道放到最后；第二步 `[B,N,D]@[D,M]` 对每个 patch 计算 M 个类别分数；最后把类别轴移到前面，再按原网格顺序恢复空间。不能在 `[B,N,M]` 上直接 reshape 成 `[B,M,H_p,W_p]` 而跳过换轴，否则类别与空间的数值排列会错位。

标准整图路径接下来执行：[eval.py](https://github.com/yoon307/DINOde/blob/0a2c5182c44107fdd8c786b2192fcc3c5e5bebd1/eval.py#L941-L953)。

```python
logits, _, _ = self.head(A, text_features)
pred_masks = F.interpolate(
    logits, size=masks.shape[-2:], mode='bilinear', align_corners=False
)
# 此处省略源码针对多背景提示的通道合并分支
pred_probabilities = F.softmax(pred_masks, dim=1)
pred_labels_raw = torch.argmax(pred_masks, dim=1)
```

从 `[2,20,14,14]` 插值到例如 `[2,20,224,224]`，只是把每个类别的连续得分图放大；通道数不变，没有新增可学习解码器。随后 `argmax(dim=1)` 在 20 个类别中选索引，类别轴消去，输出 `[2,224,224]`。在这里对同一组 logits 做 softmax 不改变 argmax，但 softmax 概率可供后续阈值判断使用。

要先插值连续分数，再取 argmax；类别索引是离散整数，不能先分类后做双线性插值，否则会混出不存在的“中间类别”。若需要缩放已经生成的标签图，应使用最近邻插值。`align_corners=False` 定义插值的坐标对齐方式，不负责恢复真实物体边界。基础概念可回看 [[downsampling_and_upsampling(下采样与上采样)]]。

公开评估还包含背景提示合并、阈值、滑窗与 PAMR（Pixel-Adaptive Mask Refinement，像素自适应掩码细化）等路径。它们位于基础分数生成之后或外围，不改变这里“固定 DINO patch 与 STF 类别锚点做点积”的核心接口；本页不继续展开完整评估流程。

**迁移思路：** 如果训练后文本对齐器与类别集合固定，可以一次生成 `[M,D]` 类别锚点，再让多张图共享。更换类别只重算文本侧。若需要更精细边界，应在空间特征或分数图上设计细化，不能指望增加 STF 文本积分步数自动补回 patch 网格丢失的像素细节。

## 10. 回头看，最值得自己写会的几段代码

| 实现思路 | 再读入口 | 将来适合用在哪里 |
|---|---|---|
| 固定大编码器，在边界之外训练小映射 | `CLIPTextEncoder.encode`、`forward_grid`、`_encode_texts_captions_batch` | 复用预训练特征、缓存特征后做轻量对齐 |
| 不同模态先整理到统一状态维度 | `text_flow_init`、`cls_flow_init` | 跨模态融合前的接口设计；维度对齐后仍需语义对齐 |
| 时间编码 + FiLM + 残差速度预测 | `sinusoidal_time_embedding`、`TimeConditionedTextFlowNet` | 连续条件模型、迭代细化、共享参数多步更新 |
| 点积和广播实现切向更新 | `apply_text_flow`、`apply_cls_flow` | 方向型表示的迭代优化；明确离散归一化位置 |
| 逐通道 top/bottom-k 汇总 | `min_max_k_pooling` | 只有图像级监督时汇总密集视觉证据 |
| 联合分数与非对称梯度分开设计 | caption 拼接处、`clip_style_contrastive_loss` | 既利用多个分支，又限制某条监督的梯度方向 |
| 特征通道收缩与空间维恢复 | `TextCondHead.forward`、`eval.py` 的插值处 | 开放词汇分类、patch CAM、密集相似度解码 |

这份代码的关键连接是：冻结视觉表征提供稳定的空间坐标，STF 学会把文本放进去，池化和 GCF 把图像级监督送到对齐模块，而推理把学好的文本方向直接应用于每个 patch。自己借鉴时，最应先画清楚的是“哪一维被聚合、哪一端允许学习、在哪一步归一化、最终类别分数从哪次点积产生”。这些关系清楚后，ODE、FiLM、拼接和 `einsum` 就能成为可独立修改的实现选择。
