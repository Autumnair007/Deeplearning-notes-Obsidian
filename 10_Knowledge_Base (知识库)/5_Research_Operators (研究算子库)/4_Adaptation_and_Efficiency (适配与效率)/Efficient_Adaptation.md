---
type: operator-note
aliases:
  - Efficient Adaptation
  - Parameter-Efficient Adaptation
  - 高效适配
tags:
  - research-operator
  - adapter
  - parameter-efficient
  - frozen-backbone
  - weakly-supervised-segmentation
  - open-vocabulary-segmentation
status: in-progress
---

# Efficient Adaptation（高效适配）

## 1. 本页定位

本页整理经典视觉模型、弱监督语义分割（Weakly Supervised Semantic Segmentation，**WSSS**）和开放词汇分割（Open-Vocabulary Segmentation，**OVS**）中，怎样在尽量保留预训练骨干的前提下，用少量可学习参数完成任务适配。范围包括线性 probe、瓶颈适配器（adapter）、低秩适配（Low-Rank Adaptation，**LoRA**）、可学习投影、门控、缓存初始化适配器、轻量解码器和部分解冻。

这不是“参数越少越好”的排行榜，也不是完整训练工程。重点是：真正错位发生在文本、通道、空间关系还是解码端；可训练参数插在哪里；冻结骨干是否仍参与前向和反向图；训练时借用的教师是否在部署时移除。完整方法回到 [[WeCLIP_paper_notes]]、[[WeCLIP+_paper_notes]]、[[ExCEL_paper_notes]]、[[DiCLIP_paper_notes]]、[[Talk2DINO_paper_notes]] 和 [[VDA_paper_notes]]。

> [!abstract] 一句话直觉
> 预训练模型已经会“看”和“理解词”，下游任务通常只需要教它怎样读取密集空间、怎样跨特征空间翻译，或怎样修正少量系统偏差。高效适配就是把学习自由度放在这些真正错位的位置。

## 2. 这个算子解决什么问题

### 2.1 大白话解释

全量微调会给每个骨干参数都分配梯度和优化器状态，成本高，也可能把对比语言—图像预训练（Contrastive Language-Image Pre-training，**CLIP**）的开放词汇语义或 DINOv2 的通用空间结构改坏。完全冻结又可能不够：图像分类预训练只需要全局表示，而分割要求 patch 级类别、边界和多尺度恢复；CLIP 文本和 DINOv2 patch 还处于不同坐标空间。

高效适配常放在以下位置：

```text
输入/文本提示
→ 可学习 prompt 或文本映射
→ 冻结视觉/文本骨干
→ block 内 Adapter / LoRA / Norm 微调
→ 多层特征投影与轻量解码器
→ 原型/缓存读出适配器
→ 分割输出和损失
```

它解决的是“已有知识怎样低成本重组”。若冻结骨干完全没有目标域所需的视觉证据，例如医学影像中从未学到的纹理或极细边界，小模块不能凭空创造信息；此时需要逐层解冻、域内预训练或更合适的基础模型。

### 2.2 专业表述

给定冻结骨干 $f_{\theta_0}$ 和小型可训练模块 $g_\phi$，优化目标可写成：

$$
\min_{\phi}\ \mathcal L\bigl(g_\phi(f_{\theta_0}(x)),y\bigr),
\qquad
\nabla_{\theta_0}\mathcal L=0.
$$

**公式解释：** 输入 $x$ 先经过参数固定为 $\theta_0$ 的骨干得到特征，再由参数 $\phi$ 的适配模块输出预测并与监督 $y$ 计算标量损失。损失内部会归约 batch、类别和空间维；第二个条件明确骨干参数梯度为零，只有 $\phi$ 更新。它并不表示骨干没有前向计算，也不表示中间激活一定不用保存——这取决于适配器插入位置和是否需要对输入特征反传。

### 2.3 容易混淆但不同的“效率”

- **参数效率**：可训练参数少、优化器状态小。
- **训练显存效率**：反向保存的激活少、冻结分支可用 `no_grad`。
- **训练时间效率**：每步前向/反向快，教师调用次数少。
- **推理效率**：最终部署模型、浮点运算次数（Floating-Point Operations，**FLOPs**）、延迟和缓存占用小。

冻结一个十亿参数教师会降低其梯度内存，却不会自动消除教师前向；LoRA 减少可训练参数，也不会让原始大矩阵 $W$ 的乘法消失。论文声称“efficient”时必须说明是哪一种效率。

> [!note] 我的理解｜先找错位，再选适配器
> 语义类名不适配，应先改 prompt 或文本投影；视觉空间和文本空间不同，应学习跨空间映射；边界差，应改关系或解码；只有当骨干高层表征本身不适合目标域时，才值得把 Adapter/LoRA 塞进多个 block。

## 3. 统一输入输出张量

### 3.1 冻结骨干特征

视觉 Transformer 常输出：

$$
X\in\mathbb R^{B\times N\times D}.
$$

**公式解释：** $B$ 是 batch 大小，$N$ 是视觉 token 数，$D$ 是特征维；$X[b,n,:]$ 是第 $b$ 张图第 $n$ 个 token 的 $D$ 维表示。该式只声明适配器输入，没有矩阵运算或维度消去。若序列含类别标记（class token，**CLS token**），则 $N=1+H'W'$；密集解码前通常要明确是否去掉第 0 个 token。

若骨干输出 `[B,D,H',W']`，逐位置通道适配可直接用 $1\times1$ 卷积；它等价于对每个空间位置独立应用线性层，不混合 $H',W'$。

### 3.2 瓶颈 Adapter

令 $W_{down}\in\mathbb R^{d\times D}$、$W_{up}\in\mathbb R^{D\times d}$，则对 batch-first token 可写：

$$
\operatorname{Adapter}(X)=
X+s\,\sigma(XW_{down}^{T}+b_{down})W_{up}^{T}+s\,b_{up}.
$$

**公式解释：** `X=[B,N,D]` 先与 `W_down^T=[D,d]` 相乘，特征维 $D$ 被消去，得到 `[B,N,d]` 瓶颈表示；`b_down=[d]` 在 $B,N$ 上广播，非线性 $\sigma$ 不改变 shape。随后与 `W_up^T=[d,D]` 相乘，瓶颈维 $d$ 被消去，恢复 `[B,N,D]`，加上广播偏置和同 shape 残差 $X$，最终输出仍为 `[B,N,D]`。$s$ 是残差分支尺度，控制适配初期对预训练表示的扰动。

无偏置时两层参数量约为：

$$
P_{adapter}=Dd+dD=2Dd.
$$

**公式解释：** 下投影矩阵有 $dD$ 个标量，上投影矩阵有 $Dd$ 个标量，相加得到 $2Dd$；这里的 $D,d$ 是维度大小，不是张量轴上的求和，所以输出是参数数量标量。若包含偏置，还需增加 $d+D$。例如 $D=512,d=64$ 时，无偏置参数为 $65{,}536$。

### 3.3 LoRA

对冻结线性权重 $W\in\mathbb R^{D_{out}\times D_{in}}$，LoRA 学习低秩增量：

$$
W'=W+\frac{\alpha}{r}BA,
\qquad
A\in\mathbb R^{r\times D_{in}},
\quad
B\in\mathbb R^{D_{out}\times r}.
$$

**公式解释：** `B=[D_out,r]` 与 `A=[r,D_in]` 相乘，低秩维 $r$ 被消去，得到与 $W$ 相同的 `[D_out,D_in]` 增量；标量 $\alpha/r$ 调整更新尺度，再与冻结 $W$ 逐元素相加，输出 $W'$ shape 不变。可训练参数为 $r(D_{in}+D_{out})$，但前向仍要计算原始 $Wx$，因此参数减少不等于骨干 FLOPs 按同一比例减少。

LoRA 可放在注意力的 query/key/value、输出投影或多层感知机（Multi-Layer Perceptron，**MLP**）中。分割中修改 query/key 会直接改变 token 关系，应额外检查边界、过平滑和小物体召回。

### 3.4 文本到视觉空间映射

[[Talk2DINO_paper_notes]] 使用非线性映射把 CLIP 文本向量送入 DINOv2 空间，可抽象为：

$$
\psi(t)=W_b^T\tanh(W_a^Tt+b_a)+b_b,
\qquad
t\in\mathbb R^{D_t},\quad \psi(t)\in\mathbb R^{D_v}.
$$

**公式解释：** 文本向量 $t$ 先与第一层权重相乘，输入特征维 $D_t$ 被消去，得到隐藏维 $D_h$；`tanh` 逐元素作用，shape 不变。第二层再沿 $D_h$ 相乘求和并消去隐藏维，输出 $D_v$ 维向量 $\psi(t)$，其每个元素表示映射到 DINOv2 视觉坐标系后的一个语义分量。骨干可保持冻结，只有 $W_a,W_b$ 和偏置训练。

### 3.5 动态门控适配

多个固定/可学习分支可用门控融合：

$$
Y=X+G(X)\odot\Delta X,
\qquad
G(X)\in[0,1]^{B\times N\times1}.
$$

**公式解释：** $X,\Delta X$ 都是 `[B,N,D]`，门控 `G=[B,N,1]` 在特征维 $D$ 广播，与增量逐元素相乘，不进行求和或消去维度；再与原特征相加，输出 `Y=[B,N,D]`。$Y[b,n,d]$ 表示原通道值加上该 token 门控后的适配增量。若门控是 `[B,N,D]`，每个通道可独立控制，但参数和过拟合风险更高。

## 4. 代表论文逐篇说明

| 论文 | 冻结起点 | 原方法存在的问题 | 具体可训练部分与数据流 | 与本算子的关系 |
|---|---|---|---|---|
| [[WeCLIP_paper_notes]] | 冻结 CLIP 图像/文本编码器及其多层注意力 | CLIP 为全局图文对齐训练，直接类别激活图（Class Activation Map，**CAM**）静态且局部空间解释不足；全量微调又成本高 | 轻量 Transformer 解码器读取冻结 CLIP 多层 patch 特征产生分割；冻结 CLIP CAM 细化模块（Refinement Module，**RFM**）结合冻结注意力与解码器动态亲和力，在线修正初始 CAM | 典型“冻结骨干 + 可训练密集解码器”；适配发生在读取和关系细化，不改 CLIP 主体 |
| [[WeCLIP+_paper_notes]] | 冻结 CLIP 与冻结 DINO | 纯 CLIP 局部语义不足，多层解码器仍需较多学习参数 | 引入 DINO 最后一层特征补充局部结构，与 CLIP 共享解码器；增强 RFM（RFM+）用两种冻结特征和动态解码关系更新伪标签 | 展示增加冻结特征源可以减少解码器学习负担，但双骨干前向仍有训练/推理成本 |
| [[ExCEL_paper_notes]] | 冻结 CLIP 主体与静态视觉校准分支 | CLIP patch 关系过平滑，静态 CAM 错误无法适应目标数据 | 可学习视觉校准（Learnable Visual Calibration，**LVC**）adapter 修正动态 patch 特征/关系；静态伪标签把像素对同异类关系转成监督，分割头学习动态 CAM | 适配重点是视觉关系而非类别头；需要区分静态教师目标与可学习动态分支 |
| [[DiCLIP_paper_notes]] | CLIP 和扩散模型保持固定 | 静态视觉缓存与目标域存在偏差，严格键值检索不能随训练信号调整；冻结多层特征还需解码 | 用前景/背景缓存初始化两层键值适配器，使真实 patch 经过可学习类别映射；SegFormer 风格多层融合头和 Transformer 解码器产生分割；优化器参数组只纳入适配器和解码头 | “知识初始化 + 参数高效适配”的代表；初始化时像检索，训练后不再是严格最近邻缓存 |
| [[Talk2DINO_paper_notes]] | 冻结 DINOv2 和 CLIP 编码器 | DINOv2 patch 有空间结构但不能直接读取 CLIP 文本 | 只学习小型非线性 warping，把 CLIP caption/类别嵌入映射到 DINOv2 空间；训练时由 DINO 注意力头选择视觉候选并做图文对比 | 适配跨模态接口而非视觉骨干；参数少且保留 DINO 空间能力 |
| [[VDA_paper_notes]] | 以 CLIP 通用表示为起点 | 静态文本原型不能表达实例属性，分割解码器语义与类别锚点不一致 | 视觉属性建模与解耦（Visual Attribute Modeling and Disentanglement，**VAMD**）学习属性/类别原型，动态视觉描述组装模块按实例生成锚点；解码器语义增强对适配器嵌入施加原型约束 | 适配自由度放在属性原型和解码器语义，而不是粗暴全量改写 CLIP |
| [[SSR_paper_notes]] | CLIP 图文表示与初始 CAM | 视觉—文本模态差距使原型错位，空间传播会把语义错误扩散 | 跨模态原型对齐（Cross-Modal Prototype Alignment，**CMPA**）学习投影并以对比约束校正视觉/文本原型，再结合超像素空间约束 | 表明低成本投影也需可靠伪标签和结构约束；只对齐全局向量不足以保证边界 |

## 5. 常见实现形式归纳

| 实现形式 | 改动位置 | 是否随类别数增长 | 优点 | 局限 | 代表论文 |
|---|---|---:|---|---|---|
| Linear probe/固定类分割头 | 骨干输出后 | 是 | 最简单，可检验冻结特征可分性 | 容易封闭到训练类 | [[WeCLIP_paper_notes]] 的最低成本对照 |
| 瓶颈 Adapter | block 内或特征后 | 通常否 | 参数少、残差保留原表示 | 插太多层仍有激活成本 | [[ExCEL_paper_notes]] |
| LoRA | 注意力/MLP 权重 | 否 | 可合并权重，训练参数少 | 不天然减少主干 FLOPs | 经典高效微调基线 |
| 文本/跨空间投影 | 文本编码后 | 通常否 | 保留视觉空间，只翻译语义接口 | 映射容量过小会欠拟合 | [[Talk2DINO_paper_notes]]、[[SSR_paper_notes]] |
| 缓存初始化 Adapter | patch 读出端 | 输出层可能随类别数增长 | 兼具外部知识与学习能力 | 训练后失去严格键值解释 | [[DiCLIP_paper_notes]] |
| 轻量多层解码器 | 冻结骨干之后 | 最终类别头会 | 专门恢复密集空间与边界 | 骨干缺失信息时补不回来 | [[WeCLIP_paper_notes]]、[[WeCLIP+_paper_notes]] |
| 部分解冻 | 后若干 block/Norm | 否 | 比纯 Adapter 自由度大 | 遗忘与优化器内存上升 | 冻结不足时的渐进方案 |

这些形式可组合，例如“冻结 DINOv2 + 文本映射 + 轻量分割后处理”，或“冻结 CLIP + 多层解码器 + 关系 adapter”。组合时应分别统计每部分参数和计算，不能只报告最小模块而忽略重型冻结教师与解码器。

## 6. 各种实现怎样工作

### 6.1 Head-only 与轻量解码器

**直觉**：先不动骨干，检查现有特征经过一个简单读取器能做到什么程度。

**数据流**：冻结 token → 去 CLS/恢复网格 → $1\times1$ 通道投影 → 多尺度融合/上采样 → 类别 logit。

**适用场景**：骨干已包含目标语义，主要缺少像素级读取和空间恢复。

**局限**：固定 $C$ 类卷积头会把输出接口封闭；OVS 更适合输出类别无关区域嵌入，再与动态文本词表比较。

### 6.2 Block 内 Adapter/LoRA

**直觉**：允许每层做小幅任务修正，但保留预训练主路径。

**数据流**：原注意力/MLP → 小残差增量 → 后续 block；或在 Q/K/V 权重上叠加低秩矩阵。

**适用场景**：冻结特征可用但系统性偏离目标域，仅改最终头不足。

**局限**：参数虽少，若插入早期层，后续所有激活都需为适配器梯度保留；密集高分辨率训练的显存节省可能远小于参数比例。

### 6.3 跨空间投影

**直觉**：不改两位专家，只训练一个翻译器让文本进入视觉坐标系。

**数据流**：冻结文本向量 `[C,D_t]` → MLP/线性映射 → `[C,D_v]` → 与冻结 patch `[B,N,D_v]` 点积 → `[B,N,C]`。

**适用场景**：DINOv2/SAM 等视觉特征空间强，但缺少开放文本接口。

**局限**：训练配对若只包含整图 caption，映射可能依赖场景；需要区域选择、注意力池化或其他机制把监督落到局部。

### 6.4 知识初始化的缓存 Adapter

**直觉**：先把外部视觉原型写进线性层，让模型从一个有意义的检索器出发，再允许它用目标任务监督调整。

**数据流**：缓存键 $K$ 初始化第一层权重 → patch 查询 → 非线性/Dropout → 缓存值 $V$ 初始化第二层 → 类别响应 → 伪标签/分割损失更新。

**适用场景**：有合成图像、检索库或原型先验，但静态最近邻与目标域有差距。

**局限**：加入偏置、非线性和训练后，权重不再等同于原缓存；必须与“冻结静态缓存读出”和“随机初始化同结构”分别消融。

## 7. 冻结、梯度与模式状态

以下三件事必须分别检查：

1. `requires_grad=False`：参数不积累梯度；
2. `torch.no_grad()`/`detach()`：是否构建从输出回到输入的计算图；
3. `model.eval()`：Dropout、BatchNorm 等模块采用推理行为。

骨干参数不在优化器中，也能阻止它被更新，但仍可能计算无用梯度并占显存。反过来，`eval()` 不会自动冻结参数。若适配器位于骨干输出之后且不需要对骨干输入求梯度，可在冻结骨干前向外使用 `no_grad`；若 Adapter 插在骨干内部，前层激活仍可能需要参与后续适配器梯度计算。

## 8. 官方仓库静态分析：DiCLIP 的缓存适配器

- 官方仓库：[zwyang6/DiCLIP](https://github.com/zwyang6/DiCLIP)
- 阅读 commit：[`1c3f6ff7d4fde2afff32d527d78b28d119583602`](https://github.com/zwyang6/DiCLIP/tree/1c3f6ff7d4fde2afff32d527d78b28d119583602)
- 适配器定义：[`model/model_diclip.py::KV_Adapter`](https://github.com/zwyang6/DiCLIP/blob/1c3f6ff7d4fde2afff32d527d78b28d119583602/model/model_diclip.py#L18-L42)
- 初始化与参数组：[`model_diclip.py#L54-L83`](https://github.com/zwyang6/DiCLIP/blob/1c3f6ff7d4fde2afff32d527d78b28d119583602/model/model_diclip.py#L54-L83)
- 前向读出：[`model_diclip.py#L148-L185`](https://github.com/zwyang6/DiCLIP/blob/1c3f6ff7d4fde2afff32d527d78b28d119583602/model/model_diclip.py#L148-L185)

### 8.1 调用链

```text
离线前景/背景视觉缓存 (cache_key, cache_value)
→ 初始化 KV_Adapter.fc1 / fc2 权重
输入图像
→ 冻结 CLIP 生成 image_features [B,1+N,512]
→ 去掉 CLS: [B,N,512]
→ fc1: [B,N,adapter_size]
→ ReLU + Dropout
→ fc2: [B,N,C]
→ Dropout
→ permute: [B,C,N]
→ reshape: [B,C,H',W']
→ 动态 CAM 监督/融合
```

### 8.2 缓存怎样初始化两层线性层

核心代码为：

```python
cache_key, cache_value = kv_cache
idx = cache_key.shape[0]
self.fc1 = nn.Linear(in_features, hidden_features)
k_prompt = trunc_normal_(torch.zeros(hidden_features, in_features))
k_prompt[:idx] = cache_key
self.fc1.weight = nn.Parameter(k_prompt.clone())

self.fc2 = nn.Linear(hidden_features, out_features)
v_prompt = trunc_normal_(torch.zeros(hidden_features, out_features))
v_prompt[:idx] = cache_value
self.fc2.weight = nn.Parameter(v_prompt.t().clone())
```

若 `in_features=512`、`hidden_features=312`、缓存条目数为 $U=idx$、输出类别数为 $C$：

- `fc1.weight=[312,512]`；前 $U$ 行写入 `cache_key=[U,512]`，剩余隐单元保持截断正态随机初始化；
- `fc2.weight=[C,312]`；代码先构造 `[312,C]`，前 $U$ 行写入 `cache_value=[U,C]`，转置后赋给线性层；
- 因此缓存条目对应前 $U$ 个隐单元，额外隐单元提供可学习容量；要求 $U\le312$，否则切片赋值会失败；
- `fc1` 和 `fc2` 的 bias 没有由缓存初始化，仍保留 `nn.Linear` 默认初始化。

### 8.3 前向 shape 与梯度

```python
x = self.fc1(x)
x = F.relu(x)
x = self.drop(x)
x = self.fc2(x)
x = self.drop(x)
```

输入 `x=[B,N,512]` 经第一层得到 `[B,N,312]`，ReLU/Dropout 不改 shape；第二层把隐维 312 消去并输出 `[B,N,C]`。所有权重和偏置都是 `nn.Parameter`，未 detach，因此损失会同时更新缓存初始化部分和随机扩展部分。

主模型中：

```python
dynamic_maps = self.dynamic_adapter(image_features[:, 1:, :])
dynamic_maps_pred = dynamic_maps.permute(0, 2, 1).reshape(
    b, self.num_classes, f_h, f_w
)
```

`[:,1:,:]` 去掉 CLS，要求剩余 $N=f_hf_w$；`permute` 把类别维移到通道位置，再按 patch 顺序恢复 `[B,C,H',W']`。这里没有插值，空间尺寸取自融合特征 `fts`；如果 CLIP token 数与 `f_hf_w` 不一致，reshape 会直接失败。

### 8.4 哪些参数真正训练

[`get_param_groups`](https://github.com/zwyang6/DiCLIP/blob/1c3f6ff7d4fde2afff32d527d78b28d119583602/model/model_diclip.py#L73-L83) 只把以下参数放入返回组：

- `dynamic_adapter.parameters()`；
- Transformer `decoder.parameters()`；
- 多层特征融合头 `decoder_fts_fuse.parameters()`。

CLIP `encoder` 和扩散提取器不在这些优化器参数组中。代码还调用 `self.encoder.eval()` 固定其模式；但“未进入优化器”和“`requires_grad=False`”仍是不同概念，应结合训练入口确认是否有无用梯度图。`diff_maps.detach()` 在最终返回时显式切断静态扩散/缓存 CAM 对后续损失的梯度，而 `dynamic_maps_pred` 保留适配器梯度。

### 8.5 论文叙事与代码真实实现的差异

- 静态缓存读出可写成相似度乘值矩阵；动态实现却包含 bias、ReLU 和两次 Dropout，训练后不是严格余弦/最近邻检索。
- 构造参数提供 `act_layer=nn.GELU` 和 `drop=0.`，实例也创建了 `self.act`，但 `forward` 实际硬编码 `F.relu`，没有使用 `self.act`；Dropout 也固定为 `0.1`，没有使用传入的 `drop`。
- 只有前 $U$ 个隐单元具备缓存语义，其余单元随机初始化；把整个 312 维隐层都称为“缓存条目”并不准确。
- 适配器输出通道数固定为缓存值的 `out`，主模型又按 `self.num_classes` reshape；两者必须一致，否则代码接口不成立。

## 9. 选型指南

| 当前问题 | 优先考虑 | 不值得或需先检查 |
|---|---|---|
| 冻结 token 已线性可分 | linear probe/轻量解码头 | 不先给所有 block 加 Adapter |
| 文本和视觉处于不同空间 | 小型线性/非线性投影 | 不用全量微调视觉骨干代替坐标映射 |
| 静态缓存有用但域偏差明显 | 缓存初始化 Adapter + 静态分支消融 | 若随机初始化同结构一样好，缓存先验价值有限 |
| patch 关系过平滑 | Q/K 或关系校准 Adapter | 先监控边界与小物体，不只看分类损失 |
| 冻结骨干缺少密集恢复 | 多层轻量解码器 | 单层 $1\times1$ 头无法恢复不存在的空间信息 |
| 全冻结明显欠拟合 | 逐步解冻最后 block/Norm | 不直接跳到全量微调 |
| OVS 未见类下降 | 共享类无关 Adapter、保留文本接口 | 不使用只输出 seen 类的固定头作为唯一接口 |
| 参数少但训练仍很慢 | 检查冻结教师次数、激活和数据流 | 参数量不能代表端到端时间 |

建议按自由度递增做基线：head-only → 跨空间投影/单个 Adapter → 多层 Adapter 或 LoRA → 解冻最后若干层 → 全量微调。每一步都报告可训练参数、总参数、训练峰值显存、每步时间、部署路径和 seen/unseen 性能，才能判断“高效”是否真实。

## 10. 论文与源码索引

- [[WeCLIP_paper_notes]]：冻结 CLIP 多层特征怎样由轻量解码器读取，RFM 怎样在线修正静态目标。
- [[WeCLIP+_paper_notes]]：双冻结骨干、共享解码器与参数/计算之间的取舍。
- [[ExCEL_paper_notes]]：可学习视觉校准器如何修正 patch 关系。
- [[Talk2DINO_paper_notes]]：只学习文本 warping、保留 DINOv2 视觉空间。
- [[VDA_paper_notes]]：属性原型与解码器语义增强的可学习接口。
- [DiCLIP `KV_Adapter`](https://github.com/zwyang6/DiCLIP/blob/1c3f6ff7d4fde2afff32d527d78b28d119583602/model/model_diclip.py#L18-L42)：缓存键值初始化、真实激活函数和 Dropout。
- [DiCLIP 参数组](https://github.com/zwyang6/DiCLIP/blob/1c3f6ff7d4fde2afff32d527d78b28d119583602/model/model_diclip.py#L73-L83)：确认优化器应接收哪些模块。
- [DiCLIP 动态 CAM 前向](https://github.com/zwyang6/DiCLIP/blob/1c3f6ff7d4fde2afff32d527d78b28d119583602/model/model_diclip.py#L148-L185)：从 `[B,N,512]` 到 `[B,C,H',W']` 及 detach 边界。

## 11. 当前整理结论

高效适配的核心不是选择 Adapter 还是 LoRA，而是定位最小必要自由度。阅读新论文时应追问：冻结的具体参数和模式是什么、适配器输入输出 shape 如何对应密集空间、优化器实际包含谁、教师是否参与部署、参数效率是否换来了训练或推理效率。修改模型时先验证冻结特征是否已含所需信息，再从读取头、跨空间映射和单点关系校准开始，只有这些都不足时才逐步深入骨干。
