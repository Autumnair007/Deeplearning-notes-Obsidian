---
type: operator-note
aliases:
  - Multi-Level Fusion
  - 多层级融合
tags:
  - research-operator
  - fusion
  - multi-scale
  - semantic-segmentation
  - weakly-supervised
  - open-vocabulary
status: in-progress
---

# Multi-Level Fusion（多层级融合）

> [!abstract] 本页定位
> 本页整理**如何把不同深度、尺度、骨干或预测分支的互补信息，变成一个可供分割头使用的统一张量**。证据来自经典语义分割模型，以及弱监督语义分割（Weakly-Supervised Semantic Segmentation, WSSS）和开放词汇语义分割（Open-Vocabulary Segmentation, OVS）论文。单篇论文笔记保存完整方法，本页负责比较可复用的融合接口、张量变化、选型依据与源码入口。

> [!tip] 基础机制入口
> 编码器—解码器、上采样与跳跃连接先看 [[fcn_notes]]、[[u-net_notes]] 和 [[downsampling_and_upsampling(下采样与上采样)]]；特征金字塔看 [[upernet_notes]]；Transformer多层重组看 [[dpt_notes]] 与 [[segformer_notes]]。跨模态得分怎样形成见 [[Cross_Modal_Alignment]]。

## 1. 这个算子解决什么问题？

大白话说，模型的不同位置“看到的东西不一样”：

- 浅层分辨率高，保存边缘、纹理和小物体，但不容易判断类别；
- 深层感受野大、语义强，却把空间压得很粗；
- 对比语言—图像预训练（Contrastive Language-Image Pre-training, CLIP）分支懂类别；无标签自蒸馏（self-DIstillation with NO labels, DINO）、分割一切模型（Segment Anything Model, SAM）或扩散分支更懂结构；
- 一个分支可能给出完整但含噪的响应，另一个只激活可靠的判别区域。

多层级融合要回答的不是“要不要把它们放在一起”，而是四个更具体的问题：

1. **融合谁**：同一骨干的不同层、同一层的不同感受野、不同模型，还是已经得到的类别分数？
2. **在哪里融合**：分类前的特征空间、分类后的logit/类别激活图（Class Activation Map, CAM）空间，还是迭代解码器的query空间？
3. **怎样对齐**：空间网格、通道数、类别顺序、数值尺度分别怎样统一？
4. **怎样分配信任**：固定相加、拼接后学习、全局权重，还是逐位置动态门控？

专业地说，多层级融合是一个映射：

$$
\Phi:\{F_l\}_{l=1}^{L}\longrightarrow F_{out}
\quad\text{或}\quad
\Phi:\{Z_m\}_{m=1}^{M}\longrightarrow Z_{out},
$$

**公式解释：** $\{F_l\}_{l=1}^{L}$ 表示 $L$ 个层级特征，$\{Z_m\}_{m=1}^{M}$ 表示 $M$ 个分支类别响应，$\Phi$ 是对齐与组合操作。第一条把层级来源索引 $l$ 汇总成一个 $F_{out}$，第二条把分支索引 $m$ 汇总成 $Z_{out}$；具体被消去的维度取决于后续是求和还是拼接。前者发生在分类头之前，后者发生在各分支已输出同一类别集合之后。

> [!note] 我的理解｜“多尺度”至少有三种含义
> 输入图像缩放、多层特征金字塔、同一特征上的多感受野分支都常被叫作多尺度。它们的数据流和成本不同。阅读论文时必须追问：尺度变化发生在输入、骨干层级，还是ASPP/PPM这类上下文模块中？

### 1.1 哪些相似现象不由本算子负责？

- 类别本身识别错误：先检查 [[Cross_Modal_Alignment]]、文本提示或分类头，而不是盲目加浅层特征。
- 正确种子无法覆盖完整物体：可能需要 [[Spatial_Propagation]]。
- 关系矩阵跨越物体边界：应检查 [[Attention_and_Affinity_Refinement]]。
- 外部视觉原型覆盖不足：属于 [[Retrieval_and_Memory]] 的记忆质量问题。
- 单一路径只是分辨率太低：简单上采样可能已足够，不一定需要额外融合模块。

## 2. 统一输入输出张量

### 2.1 情形A：CNN或层次Transformer的空间金字塔

设第 $l$ 层特征为：

$$
F_l\in\mathbb{R}^{B\times D_l\times H_l\times W_l}.
$$

**公式解释：** $B$ 是批量大小，$D_l$ 是第 $l$ 层通道数，$H_l,W_l$ 是空间尺寸；$F_l[b,d,h,w]$ 是该层一个具体特征值。这里只声明输入 shape，没有求和或维度消去。不同层通常既不同通道，也不同分辨率，因此不能直接相加或拼接。

第一步，用 $1\times1$ 卷积统一通道：

$$
\bar F_l=\phi_l(F_l)
\in\mathbb{R}^{B\times D\times H_l\times W_l}.
$$

**公式解释：** $\phi_l$ 通常是 $1\times1$ 卷积，在每个 $(h,w)$ 独立把 $D_l$ 个输入通道线性组合成 $D$ 个输出通道。通道维从 $D_l$ 变为 $D$，空间和 batch 维保留；$\bar F_l[b,d,h,w]$ 是第 $l$ 层在共同通道坐标中的第 $d$ 个响应。它不会恢复已丢失的边界。

第二步，统一空间网格：

$$
\tilde F_l=\operatorname{Resize}(\bar F_l;H_*,W_*)
\in\mathbb{R}^{B\times D\times H_*\times W_*}.
$$

**公式解释：** `Resize` 只把空间维 $H_l,W_l$ 改成公共网格 $H_*,W_*$，batch 与通道维 $B,D$ 不变，也没有求和消去维度。$\tilde F_l[b,d,h,w]$ 表示第 $l$ 层投影特征在公共坐标 $(h,w)$ 的值。放大常用双线性插值，缩小常用步幅卷积或池化。

具体例子：输入为 `[2,3,320,320]`，骨干输出：

```text
F1: [2,  64, 80, 80]
F2: [2, 128, 40, 40]
F3: [2, 320, 20, 20]
F4: [2, 512, 10, 10]
```

若统一到 $D=256$、$H_*=W_*=80$：

```text
各层 1×1 projection → [2,256,H_l,W_l]
各层 interpolate      → [2,256,80,80]
concat(dim=1)         → [2,1024,80,80]
1×1 fusion            → [2,256,80,80]
```

这里拼接只增加通道维，批量与空间维不变；最后的 $1\times1$ 卷积把四层信息从1024通道压回256通道。

### 2.2 情形B：ViT同一patch网格上的多层特征

标准视觉Transformer（Vision Transformer, ViT）的各块通常保持相同token数：

$$
X_l\in\mathbb{R}^{B\times(1+N)\times D_l},
\qquad N=H'W'.
$$

**公式解释：** $X_l$ 是 Transformer 第 $l$ 层 token 序列，长度 $1+N$ 由一个 CLS token 和 $N=H'W'$ 个 patch token 组成，每个 token 有 $D_l$ 维。这里只声明 shape；恢复二维前需去掉 CLS，并把 $N$ 拆回 $H'\times W'$，没有做数值求和。

去掉类别标记（class token, CLS token）并恢复二维网格：

```text
[B,1+N,D_l]
→ 去掉CLS
→ [B,N,D_l]
→ permute(0,2,1)
→ [B,D_l,N]
→ reshape(B,D_l,H',W')
→ [B,D_l,H',W']
```

如果所有层使用相同patch网格，就不需要插值，只需统一通道。不要机械套用特征金字塔逻辑：ViT的“多层”可以只有语义深度变化，没有空间尺寸变化。

### 2.3 情形C：分数级多分支融合

设第 $m$ 个分支已经输出：

$$
Z^{(m)}\in\mathbb{R}^{B\times C\times H\times W}.
$$

**公式解释：** $m$ 是分支索引，$Z^{(m)}[b,c,h,w]$ 是第 $m$ 个分支对像素 $(h,w)$ 属于类别 $c$ 的分数。各分支 shape 相同只说明接口对齐，尚未发生融合；相加前仍要确认类别顺序、背景定义和数值尺度一致。

相加前至少必须确认：

- $C$ 个通道的类别顺序完全一致；
- 是否都含背景通道，背景含义是否一致；
- 都是logit、概率还是归一化CAM；
- 温度、数值范围和空间分辨率是否可比。

形状相同不等于语义可加。例如CLIP余弦相似度、扩散CAM和分割头logit的数值尺度不同，直接用同一个系数相加不能把系数解释成概率。

## 3. 代表模型与论文

| 论文/模型 | 任务与起点 | 原方法存在的问题 | 具体做法 | 与多层融合的关系 |
|---|---|---|---|---|
| [[fcn_notes]] | 经典全监督分割；深层分类特征 | 32倍上采样只给出粗轮廓 | 将深层类别分数图逐步上采样，与pool4、pool3经 $1\times1$ 卷积得到的同通道分数图逐元素相加 | 奠定“深层语义 + 浅层细节”的跳跃相加基线 |
| [[deeplabv3+_notes]] | 全监督分割；空洞卷积骨干 | 单一感受野不适应大小不同的目标，深层输出边界粗 | 空洞空间金字塔池化（Atrous Spatial Pyramid Pooling, ASPP）并行提取多感受野特征并拼接；再将高层特征上采样，与压缩后的低层特征拼接细化 | 同时包含同层多感受野融合和浅深层融合 |
| [[upernet_notes]] | 通用场景解析；多级骨干 | 单一层级难同时处理局部目标与全局场景 | 金字塔池化模块（Pyramid Pooling Module, PPM）补全顶层全局上下文，特征金字塔网络（Feature Pyramid Network, FPN）自顶向下逐级相加 | 展示“顶层上下文 + 多层金字塔”的级联融合 |
| [[segformer_notes]] | 轻量Transformer分割 | 重型解码器成本高 | 四级特征分别经MLP投影到同一维度，上采样到共同网格，通道拼接后线性融合 | 说明简单投影、插值和拼接也能成为强基线 |
| [[dpt_notes]] | Transformer密集预测 | 最终token缺少多尺度空间细节 | 从多个Transformer层读出token，经reassemble恢复空间，再用逐级融合块整合 | 同网格token可以被重新组织成层级式解码路径 |
| [[mask2former_notes]] | 通用图像分割 | 一次性融合所有像素特征成本高且缺少对象级选择 | 像素解码器生成多尺度特征；Transformer解码器连续层轮流读取不同分辨率，并用上一层掩码限制交叉注意力 | 多尺度信息以“逐层交互”而非一次concat进入query |
| [[WeCLIP_paper_notes]] | WSSS；冻结CLIP | 最后一层CLIP特征难直接产生精细分割 | 每个Transformer块的特征经独立多层感知机（Multi-Layer Perceptron, MLP），通道拼接和卷积得到融合特征，再由轻量解码器预测 | 把经典多层解码器迁移到冻结视觉—语言骨干 |
| [[ComCD_paper_notes]] | WSSS；CLIP与扩散模型 | CLIP语义强但不完整，扩散分支空间完整但类别特异性弱 | 分别生成两路CAM，用像素预测熵衡量局部不确定性并动态决定更可信分支 | 属于位置级分数融合，而非骨干层级拼接 |
| [[DiCLIP_paper_notes]] | WSSS；CLIP、扩散关系与视觉缓存 | 单一补丁—文本CAM覆盖不足 | 基础文本CAM、静态缓存CAM、动态适配器CAM用固定系数组合；最终分割头同时读取CLIP 12层特征 | 同时使用分数级融合和同网格多层特征融合 |
| [[Trident_paper_notes]] | 高分辨率OVS；CLIP、DINO、SAM | 滑窗先预测后拼接破坏跨窗口上下文 | 先拼接子图特征，再让DINO提供局部对象关系、SAM提供全局聚合、CLIP提供开放类别语义 | 属于多骨干职责分工，不等价于简单加权求和 |

> [!note] 我的理解｜融合对象决定了融合规则
> FCN融合的是同一类别空间中的分数图；SegFormer和WeCLIP融合的是分类前特征；Mask2Former让query分阶段读取多尺度特征；ComCD融合的是不同模型产生的CAM。把这些都简称“feature fusion”会掩盖最重要的接口差异。

## 4. 常见实现形式归纳

| 实现形式 | 输入单元 | 是否训练 | 优点 | 局限 | 代表工作 |
|---|---|---:|---|---|---|
| 跳跃相加 | 同通道、同网格特征或分数 | 投影层可训练 | 参数少，梯度路径直接 | 相加后难追踪各分支贡献 | [[fcn_notes]]、[[upernet_notes]] |
| 拼接后投影 | 多层特征 | 是 | 保留来源信息，学习跨层通道组合 | 中间通道与显存随层数增长 | [[segformer_notes]]、[[WeCLIP_paper_notes]] |
| 自顶向下金字塔 | 相邻尺度特征 | 是 | 逐步把高层语义送到高分辨率层 | 多次插值可能模糊边界 | [[upernet_notes]] |
| 并行多感受野 | 同一特征图的多分支 | 是 | 不改变骨干层级即可获得多尺度上下文 | 大扩张率可能出现栅格效应 | [[deeplabv3+_notes]] |
| 多尺度交叉注意力 | query与多个尺度的Key/Value | 是 | 按内容选择尺度和位置 | 成本依赖 $N_qN_k$，实现复杂 | [[mask2former_notes]] |
| 固定分数加权 | 对齐后的多路CAM/logit | 否 | 最简单，便于先验证互补性 | 所有样本、位置共享同一信任度 | [[DiCLIP_paper_notes]] |
| 动态门控 | 多路特征或分数 + 门控特征 | 是 | 可按图像、通道或位置选择来源 | 容易饱和或学到数据偏差 | [[ComCD_paper_notes]] |

这些形式可以组合：DeepLabV3+先做ASPP并行拼接，再做低层跳跃拼接；UPerNet先用PPM增强顶层，再用FPN逐级融合；DiCLIP先构造多路CAM，再让最终分割头读取12层CLIP特征。组合时应逐级消融，不能把所有提升统称为“多尺度有效”。

## 5. 各种实现怎样工作？

### 5.1 跳跃相加：先对齐，再逐元素合并

直觉：把深层“这是什么”的判断上采样，与浅层“边界在哪里”的信息在同一坐标上相加。

$$
P_l=\phi_l(F_l),\qquad
P_{l-1}=\phi_{l-1}(F_{l-1}),
$$

**公式解释：** 两个 $\phi$ 分别把深层 $F_l$ 和浅层 $F_{l-1}$ 的通道投影到同一 $D$ 维。此步只改变通道坐标，不消去空间维；$P_l$ 分辨率仍较低，$P_{l-1}$ 分辨率较高，所以还不能直接相加。

$$
Y_{l-1}=P_{l-1}+\operatorname{Upsample}(P_l).
$$

**公式解释：** `Upsample(P_l)` 把深层空间尺寸恢复到 $H_{l-1},W_{l-1}$；此时它与 $P_{l-1}$ 都是 `[B,D,H_{l-1},W_{l-1}]`。逐元素相加不消去任何维度，$Y_{l-1}[b,d,h,w]$ 是浅层局部信息与深层语义在同一位置、同一通道的和。

若两支都是类别分数，$D=C$；若两支还是中间特征，$D$ 是共享嵌入维度。FCN-16s更接近前者，FPN更接近后者。

**适用**：层间含义相近、希望低成本融合。**局限**：若浅层噪声很强，它会与深层语义同权进入结果；相加后也无法让后续层知道某个值来自哪一支。

对应：[[fcn_notes]]、[[upernet_notes]]。

### 5.2 拼接后投影：保留来源，再学习压缩

$$
F_{cat}=\operatorname{Concat}(\tilde F_1,\ldots,\tilde F_L;\text{channel})
\in\mathbb{R}^{B\times LD\times H_*\times W_*},
$$

**公式解释：** 每个 $\tilde F_l$ 都是 `[B,D,H_*,W_*]`，沿通道维并排拼接后，层索引 $L$ 被展开进通道，得到 $LD$ 个通道；空间、batch 不变。Concat 不做数值混合，也没有求和消去来源，只保留每层的全部通道。

$$
F_{out}=\operatorname{Conv}_{1\times1}(F_{cat})
\in\mathbb{R}^{B\times D_o\times H_*\times W_*}.
$$

**公式解释：** $1\times1$ 卷积在每个位置把 $LD$ 个拼接通道线性组合为 $D_o$ 个通道，通道输入维被卷积求和并替换为输出维；$H_*,W_*,B$ 保留。$F_{out}[b,d_o,h,w]$ 是所有层级通道在该位置的可学习融合。若 4 层各 256 通道，输入 1024 通道，输出 256 通道。

**适用**：不知道哪些层最重要，希望让数据学习组合。**局限**：拼接前中间激活占用大；层数增加时参数线性增长。对应：[[segformer_notes]]、[[WeCLIP_paper_notes]]。

### 5.3 FPN式自顶向下融合：逐级传播语义

特征金字塔网络（Feature Pyramid Network, FPN）的典型递推为：

$$
P_L=\phi_L(F_L),
$$

**公式解释：** 最深层 $F_L$ 先经侧向投影 $\phi_L$ 统一为公共通道数 $D$，得到金字塔顶层 $P_L$。该步不融合其他层，也不消去空间维，只建立递推起点。

$$
P_l=\phi_l(F_l)+\operatorname{Upsample}(P_{l+1}),
\qquad l=L-1,\ldots,1.
$$

**公式解释：** 对每个 $l$，$\phi_l(F_l)$ 与上采样后的 $P_{l+1}$ 都对齐为 `[B,D,H_l,W_l]`，逐元素相加不消去维度，输出同 shape 的 $P_l$。层级索引通过递推逐步处理而不是在一次张量运算中求和；$P_l[b,d,h,w]$ 同时含当前层细节和所有更深层语义。

上采样通常是2倍双线性或最近邻插值。最近邻不产生新混合值、边缘更硬；双线性更平滑；转置卷积可学习但有额外参数。经典语义分割常在融合后再用 $3\times3$ 卷积消除上采样造成的混叠。

**适用**：骨干天然输出2倍尺度递减的金字塔。**局限**：尺寸不是整倍数时必须明确 `size=` 而不是只用 `scale_factor`，否则奇数尺寸容易错一格。对应：[[upernet_notes]]。

### 5.4 ASPP/PPM：同一层上的上下文融合

空洞空间金字塔池化（Atrous Spatial Pyramid Pooling, ASPP）对同一输入 $F$ 使用不同扩张率：

$$
Y=\phi\bigl(\operatorname{Concat}[
\operatorname{Conv}_{1\times1}(F),
\operatorname{AtrousConv}_{r_1}(F),
\operatorname{AtrousConv}_{r_2}(F),
\operatorname{Pool}(F)]\bigr).
$$

**公式解释：** 同一输入 $F=[B,D,H,W]$ 被送入普通 $1\times1$ 卷积、不同扩张率卷积和池化分支；各分支最终恢复到相同 $H,W$，再沿通道拼接。Concat 把分支索引并入通道而不求和，$\phi$ 再把总通道投影到目标通道数；输出 $Y[b,d,h,w]$ 是不同感受野在该位置的可学习组合。扩张率只改变采样位置，不改变输出空间尺寸。

**适用**：问题是目标尺度差异而非浅深层语义差异。**局限**：它不能替代高分辨率浅层边界；大扩张率在小特征图上可能大量采到padding。对应：[[deeplabv3+_notes]]、[[upernet_notes]]。

### 5.5 多尺度query交互：让对象查询轮流读不同网格

令对象查询为 $Q\in\mathbb{R}^{B\times R\times D}$，第 $l$ 个尺度展平后的图像特征为 $K_l,V_l\in\mathbb{R}^{B\times N_l\times D}$：

$$
Q'=\operatorname{softmax}\left(\frac{QK_l^T}{\sqrt D}+G_l\right)V_l.
$$

**公式解释：** $Q=[B,R,D]$ 是 $R$ 个对象 query，$K_l,V_l=[B,N_l,D]$ 是第 $l$ 个尺度的空间 token。`Q @ K_l^T` 消去特征维 $D$，得到 `[B,R,N_l]`；加同 shape 掩码 logit $G_l$ 后，softmax 沿位置维 $N_l$ 归一化。再乘 $V_l$ 时消去 $N_l$，输出 $Q'=[B,R,D]$；$Q'[b,r,:]$ 是 query $r$ 从该尺度允许位置读取后的新特征。

Mask2Former不是先把所有尺度拼成一个巨大张量，而是让连续解码层循环读取不同尺度。这种融合保留query状态，适合对象级分割；但若只需要像素分类，一个轻量FPN或MLP解码头通常更简单。

对应：[[mask2former_notes]]。

### 5.6 固定加权与动态门控

固定加权：

$$
Z=\sum_{m=1}^{M}\alpha_m\tilde Z^{(m)}.
$$

**公式解释：** 每个 $\tilde Z^{(m)}$ 都已对齐为 `[B,C,H,W]`，$\alpha_m$ 是第 $m$ 个分支的标量权重。沿 $M$ 加权求和并消去分支索引，输出 $Z$ 仍为 `[B,C,H,W]`；$Z[b,c,h,w]$ 是所有分支对同一像素、同一类别的固定比例投票。

动态门控可写成：

$$
G=\operatorname{softmax}(g(F))
\in\mathbb{R}^{B\times M\times H\times W},
$$

**公式解释：** 门控网络 $g$ 从上下文 $F$ 预测每个像素的 $M$ 个分支 logit；softmax 沿分支维 $M$ 归一化，shape 仍为 `[B,M,H,W]`。$G[b,m,h,w]$ 表示位置 $(h,w)$ 对分支 $m$ 的信任比例，并满足对 $m$ 求和为 1；它不是类别概率。

$$
Z_{b,c,h,w}=\sum_m G_{b,m,h,w}Z^{(m)}_{b,c,h,w}.
$$

**公式解释：** 对固定 $b,c,h,w$，每个分支类别分数乘以同位置门控权重，再沿分支维 $m$ 求和并消去它。输出仍是 `[B,C,H,W]`；$Z[b,c,h,w]$ 是位置自适应的融合分数。门控在类别维广播，因此同一像素所有类别共用一组分支权重。

ComCD使用像素级不确定性决定更相信CLIP还是扩散响应，可作为位置门控的实例。若门控只输入某一分支自身分数，它可能学会“分数越大越相信自己”的捷径，最好同时输入两支特征或不确定性证据。

对应：[[ComCD_paper_notes]]、[[DiCLIP_paper_notes]]。

## 6. 官方仓库静态分析：DiCLIP的12层CLIP特征融合

### 6.1 仓库与固定版本

- 官方仓库：[zwyang6/DiCLIP](https://github.com/zwyang6/DiCLIP)
- 阅读commit：[`1c3f6ff7d4fde2afff32d527d78b28d119583602`](https://github.com/zwyang6/DiCLIP/tree/1c3f6ff7d4fde2afff32d527d78b28d119583602)
- 主调用：[model/model_diclip.py#L148-L166](https://github.com/zwyang6/DiCLIP/blob/1c3f6ff7d4fde2afff32d527d78b28d119583602/model/model_diclip.py#L148-L166)
- 融合头：[model/segformer_head.py#L47-L77](https://github.com/zwyang6/DiCLIP/blob/1c3f6ff7d4fde2afff32d527d78b28d119583602/model/segformer_head.py#L47-L77)
- 参数分组：[model/model_diclip.py#L73-L83](https://github.com/zwyang6/DiCLIP/blob/1c3f6ff7d4fde2afff32d527d78b28d119583602/model/model_diclip.py#L73-L83)

### 6.2 调用链

```text
img [B,3,H,W]
→ generate_clip_fts
→ 12层 all_feats（含CLS）
→ 去掉CLS、permute、reshape
→ [12,B,512,H/16,W/16]
→ SegFormerHead：每层独立 MLP 512→256→256
→ 12层沿通道 concat
→ [B,3072,H/16,W/16]
→ 1×1 Conv 3072→256 + Dropout2d
→ DecoderTransformer
→ segmentation logits
```

主模型真实代码：

```python
all_img_tokens = all_feats[:, :, 1:, ...]
all_img_tokens = all_img_tokens.permute(0, 1, 3, 2)
all_img_tokens = all_img_tokens.reshape(
    12, b, all_img_tokens.size(-2), h // 16, w // 16
)
fts = self.decoder_fts_fuse(all_img_tokens)
seg, seg_attn_weight_list = self.decoder(fts)
```

逐步解释：

1. `all_feats[:, :, 1:, ...]` 去掉每层第0个CLS token，因为分割需要与patch网格一一对应的token。
2. 原token布局被转成 `[layer,B,D,N]`，随后将 $N=(H/16)(W/16)$ 恢复成二维网格。
3. 12层CLIP token共享同一patch网格，所以这里没有插值；融合的“level”是Transformer深度，不是不同空间尺度。
4. `h // 16, w // 16` 假设输入尺寸可被16整除，且token数量与这两个整数的乘积完全一致；否则 `reshape` 会失败。

融合头核心：

```python
_x = self.linears_modulelist[ind](x.float()) \
    .permute(0, 2, 1) \
    .reshape(n, -1, x.shape[2], x.shape[3])
x_list.append(_x)
x_list = torch.cat(x_list, dim=1)
x = self.linear_fuse(x_list)
x = self.dropout(x)
```

每个 `MLP` 先执行 `flatten(2).transpose(1,2)`：

```text
[B,512,H',W']
→ flatten(2)
→ [B,512,N]
→ transpose(1,2)
→ [B,N,512]
→ Linear 512→256→256
→ [B,N,256]
→ permute + reshape
→ [B,256,H',W']
```

12个 `[B,256,H',W']` 在通道维拼接为 `[B,3072,H',W']`，`linear_fuse` 是 $1\times1$ 卷积，把每个位置的3072维多层描述压到256维。

### 6.3 哪些参数训练，哪些路径冻结？

`get_param_groups()` 把 `decoder_fts_fuse` 与后续 `decoder` 参数加入分割头参数组；CLIP编码器没有加入优化器参数组，并在构造和前向中设为 `eval()`。因此：

- 12层CLIP特征是冻结输入；
- 每层独立MLP、`linear_fuse`、Dropout后的解码器会训练；
- 本条融合路径没有显式 `detach()`，但冻结编码器不更新；
- 主函数最终返回的伪CAM `diff_maps.detach()` 与分割特征融合是两条不同监督/预测路径。

### 6.4 论文表述与代码实现的差异/细节

- 代码类名沿用 `SegFormerHead`，但它没有处理SegFormer经典四级不同分辨率特征；这里处理的是12个同分辨率CLIP层。
- 构造参数 `num_classes` 被保存，却没有在该融合头的 `forward` 中用于分类；类别预测由后续 `DecoderTransformer` 完成。
- `MLP.__init__` 接受 `act_layer` 的思路没有出现在这个文件中，实际激活固定写为 `F.relu`。
- 所有层一视同仁地拼接，没有显式层权重或门控；层选择完全交给独立MLP和最终 $1\times1$ 卷积隐式学习。
- `x.float()` 强制融合输入转成32位浮点（32-bit floating point, FP32），会提高数值稳定性，也增加相对半精度的激活显存。

> [!note] 我的理解｜这是一种“同网格、深度维融合”
> 它借用了SegFormer“每层投影 + 拼接 + 线性融合”的外形，但真正省掉的是空间对齐，而不是多层选择。若要改成只取后4层，必须同步修改 `index`、传入层数和 `linear_fuse` 输入通道，不能只切片一个张量。

## 7. 选型指南

| 当前问题 | 优先考虑 | 不建议先做 |
|---|---|---|
| 深层预测类别对，但轮廓粗 | 浅层跳跃连接或FPN | 直接换跨模态提示 |
| 同一目标大小差异明显 | ASPP/PPM或输入尺度消融 | 无差别拼接全部骨干层 |
| 冻结ViT各层同网格，想低成本利用中间表征 | 独立投影 + concat + $1\times1$ 融合 | 为制造“多尺度”强行插值 |
| 两路CAM互补且已同类别空间 | 先做归一化后的固定加权基线 | 一开始就上复杂门控 |
| 两路可靠性随位置明显变化 | 位置级门控，并可视化门控图 | 只学习每分支一个全局标量 |
| 高分辨率对象级分割，需要query反复细化 | 多尺度交叉注意力 | 一次拼接全部高分辨率token |
| OVS一支支持任意文本、一支只支持训练类 | 先在共同候选词表重算/映射 | 用零填充假装类别兼容 |
| 小物体在融合后消失 | 提高高分辨率路径权重、保留残差 | 继续增加传播步数 |

## 8. 诊断与正确消融

1. 保存每个输入分支、完成空间/通道对齐后的分支，以及最终融合输出。
2. 分别比较：单层、最后一层、等权相加、固定权重、concat投影、动态门控。
3. 将“通道投影”“空间插值”“融合规则”拆开消融，避免把仅由更高分辨率带来的收益归因于门控。
4. 对门控权重画直方图与空间热图，检查是否长期饱和到单一分支。
5. 同时报总体mIoU、小/中/大目标和边界指标；多尺度模块应在对应症状上产生可解释变化。
6. 报告推理时实际依赖的骨干数、浮点运算量（Floating-Point Operations, FLOPs）、峰值激活和延迟，而不只报告可训练参数。
7. 检查插值参数：分割logit常用双线性，离散标签只能用最近邻；`align_corners` 应在训练和推理中一致。

## 9. 论文与源码索引

### 经典模型入口

- [[fcn_notes]]：类别分数图跳跃相加的最小基线。
- [[deeplabv3+_notes]]：ASPP并行感受野与低层解码融合。
- [[upernet_notes]]：PPM + FPN的两级金字塔。
- [[segformer_notes]]：多级投影、插值、拼接的轻量头。
- [[dpt_notes]]：Transformer中间层如何恢复为密集特征。
- [[mask2former_notes]]：query怎样跨解码层轮流读取多尺度特征。

### WSSS与OVS入口

- [[WeCLIP_paper_notes]]：冻结CLIP多层特征的解码融合。
- [[ComCD_paper_notes]]：CLIP与扩散CAM的像素级不确定性融合。
- [[DiCLIP_paper_notes]]：12层CLIP融合头与多路CAM固定加权。
- [[Trident_paper_notes]]：CLIP、DINO、SAM的职责式多骨干融合。

### DiCLIP源码入口

- [主模型特征重排与调用](https://github.com/zwyang6/DiCLIP/blob/1c3f6ff7d4fde2afff32d527d78b28d119583602/model/model_diclip.py#L148-L166)：从12层token到融合头。
- [每层MLP与通道拼接](https://github.com/zwyang6/DiCLIP/blob/1c3f6ff7d4fde2afff32d527d78b28d119583602/model/segformer_head.py#L12-L27)：`[B,D,H,W] ↔ [B,N,D]`。
- [SegFormerHead融合](https://github.com/zwyang6/DiCLIP/blob/1c3f6ff7d4fde2afff32d527d78b28d119583602/model/segformer_head.py#L47-L77)：12路拼接与 $1\times1$ 压缩。
- [参数分组](https://github.com/zwyang6/DiCLIP/blob/1c3f6ff7d4fde2afff32d527d78b28d119583602/model/model_diclip.py#L73-L83)：确认融合头参与训练。

## 10. 当前整理结论

多层级融合的核心选择不是“相加还是拼接”，而是：

$$
\boxed{
\text{先辨认融合对象}
\rightarrow
\text{统一空间与语义接口}
\rightarrow
\text{选择固定或动态组合}
\rightarrow
\text{验证每个来源是否真的互补}
}.
$$

**公式解释：** 这不是张量计算式，而是四步检查顺序：先判断融合的是特征、关系、query 还是 logit；再对齐空间、通道和类别；随后选择固定求和、拼接或动态门控；最后通过单分支和 oracle 对照验证来源是否真的互补。

阅读新论文时，先追问融合发生在特征、关系、query还是类别分数上；修改模型时，先检查空间网格、通道、类别顺序、背景定义和数值尺度。经典分割模型提供结构基线，WSSS/OVS论文只是把输入来源换成冻结CLIP、扩散关系、DINO/SAM结构或多路CAM；它们仍然可以放回“对齐接口—融合规则—输出头”这条共同数据流中理解。
