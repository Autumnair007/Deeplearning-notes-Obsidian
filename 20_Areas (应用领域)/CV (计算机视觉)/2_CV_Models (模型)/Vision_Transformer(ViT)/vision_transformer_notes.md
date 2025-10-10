---
type: concept-note
tags:
  - cv
  - image-classification
  - transformer
  - vit
  - self-attention
  - full-supervision
  - activation-function
status: done
model: Vision Transformer
year: 2020
---
学习资料：[Vision Transformer详解-CSDN博客](https://blog.csdn.net/qq_37541097/article/details/118242600)

[ViT（Vision Transformer）解析 - 知乎](https://zhuanlan.zhihu.com/p/445122996)

论文原文：[[2010.11929\] An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)
***

![](../../../../99_Assets%20(资源文件)/images/image-20250424105541787.png)

图1：模型概览。我们将图像分割为固定尺寸的图块，对每个图块进行线性嵌入处理，添加位置编码后，将生成的向量序列输入标准Transformer编码器。为执行分类任务，我们采用常规方法——在序列中添加一个可学习的"分类标记"。本示意图中的Transformer编码器结构参考了Vaswani等人（2017）的设计。

------

## Vision Transformer (ViT) 模型详解

### 核心思想 (Elaboration)

传统上，卷积神经网络（CNN）是图像识别的主流方法，它利用卷积核来捕捉图像的局部空间层次结构特征。而 Transformer 最初是为处理序列数据（如文本）设计的，其核心机制——自注意力（Self-Attention）——擅长捕捉序列中元素之间的长距离依赖关系。

ViT 的核心创新在于，它找到了一种方法**将 2D 的图像“序列化”**，使其能够被 Transformer 理解。它将图像分割成一系列固定大小的小块（Patches），并将这些块视为 Transformer 输入序列中的“词元”（Tokens）。通过这种类比，ViT 成功地将强大的 Transformer 架构引入了视觉领域，证明了在足够大的数据集上进行预训练后，纯粹基于 Transformer 的模型也能在图像识别任务上达到甚至超越顶尖 CNN 的性能，而无需依赖卷积操作固有的归纳偏置（Inductive Bias，如局部性和平移不变性）。ViT 的成功表明，**大规模数据驱动的自注意力机制本身就足以学习到图像中的有效表示**。

### 模型架构详解

#### 1. 图像分块 (Image Patching)

*   **目的**: 这是将 2D 图像数据适配给 1D 序列处理的 Transformer 的关键第一步。Transformer 需要一个序列作为输入，而图像本质上是网格结构。
*   **过程**:
    *   **输入**: 接收一张标准彩色图像 $x$，其维度通常表示为 $H \times W \times C$（图像高度 H，宽度 W，颜色通道数 C，例如 RGB 图像 C=3）。
    *   **分割**: 将图像 $x$ 分割成一个网格，网格中的每个单元是一个正方形的小图像块 (Patch)。每个块的大小是预先设定的 $P \times P$ 像素。例如，如果 $P=16$，则每个块是 16x16 像素。
    *   **块的数量**: 分割后得到的总块数 $N$ 由图像尺寸和块大小决定：$N = (H/P) \times (W/P)$。这个 $N$ 就定义了输入序列的“长度”（不包括后面添加的特殊 Token）。对于 224x224 图像和 P=16，我们得到 $14 \times 14 = 196$ 个图像块。
    *   **展平 (Flattening)**: 每个 $P \times P \times C$ 的 3D 图像块被**展平成一个一维向量**。展平的方式通常是按顺序读取像素值，例如，先读完第一个通道的所有像素，再读第二个通道，以此类推，或者交叉读取。展平后，每个块变成一个长度为 $P^2 \times C$ 的向量。
    *   **输出**: 最终得到一个由 $N$ 个展平后的块向量组成的序列 $x_p$。
*   **数学表示**: $x \in R^{H \times W \times C} \rightarrow x_p \in R^{N \times (P^2 \times C)}$
    *   $x$: 原始图像张量。
    *   $x_p$: 图像块序列。这是一个包含 $N$ 个向量的集合，每个向量的维度是 $P^2 \times C$。可以看作是一个形状为 $(N, P^2 C)$ 的矩阵。

#### 2. 块嵌入 (Patch Embedding) / 线性投射 (Linear Projection)

*   **目的**: 将高维度的、原始的像素块向量（维度 $P^2 \times C$ 可能很大）映射到一个**固定大小的、更低维度的、可学习的**嵌入空间（维度 $D$）。这个 $D$ 维度贯穿整个 Transformer 模型。这与 NLP 中将单词 ID 映射为词嵌入向量类似，目的是让模型学习每个块的更有意义的表示。简单来说，就是将原始图像块（无论其内部像素值如何）转换成一个**固定维度**的向量，这个维度就是我们为 Transformer 模型选择的 `embed_dim`。
*   **过程**:
    *   对 $N$ 个展平后的块向量中的**每一个**，都应用一个**相同的、可学习的**线性变换（即一个全连接层）。这个线性变换由一个权重矩阵 $E$（维度为 $(P^2 \times C) \times D$）和一个可选的偏置项（通常省略或包含在后续的层归一化中）定义。
    *   这个操作将每个 $P^2 \times C$ 维的块向量 $x_{p_i}$ 转换为一个 $D$ 维的嵌入向量 $z_{patch_i}$。
*   **数学表示**: 对序列中的每个块 $i=1...N$ 进行 $z_{patch_i} = x_{p_i} \times E$。整个序列的嵌入可以表示为 $z_{patch} = x_p \times E$，其中：
    *   $x_p$: 形状为 $(N, P^2 C)$ 的块序列矩阵。
    *   $E$: 形状为 $(P^2 C, D)$ 的可学习嵌入（投射）矩阵。
    *   $z_{patch}$: 形状为 $(N, D)$ 的块嵌入序列矩阵。

- **使用卷积实现 (`nn.Conv2d`)**：在代码实现中，可以通过 `nn.Conv2d` 来完成这个任务的。
  *   `in_channels`: 对应原始图像的通道数 (例如 3 for RGB)。
  *   `out_channels`: **这个参数就是你设置的 `embed_dim`**。它决定了卷积层输出的特征图数量。
  *   `kernel_size=patch_size`, `stride=patch_size`: 这确保了卷积核每次恰好覆盖一个不重叠的图像块，并将这个块的信息映射到 `out_channels`（也就是 `embed_dim`）个值上。
- **通过卷积核数量控制 `embed_dim`**: nn.Conv2d` 的 `out_channels` 参数（输出通道数）直接决定了每个 patch 被映射成的向量维度。**有多少个输出通道（卷积核），每个 patch 就会被转换成多少维的向量**。

所以，正是通过在 `nn.Conv2d` 中设置 `out_channels=embed_dim`，我们实现了将任意内容的图像块转换成我们想要的、统一的 `embed_dim` 维度向量的目标。

#### 3. [CLS] Token 嵌入

**目的**: 受到 BERT 模型在 NLP 中成功实践的启发，ViT 引入了一个特殊的可学习嵌入向量，称为 [CLS] (Classification) Token。这个 Token 不对应图像中的任何特定块，而是作为一个“全局信息聚合器”。在经过 Transformer 编码器的多层处理后，最终只使用这个 [CLS] Token 对应的输出向量来进行整个图像的分类。这样做的好处是提供了一个单一的、集成的表示来代表整个图像，简化了下游分类任务的接口。

**过程与原理**:

*   **创建可学习向量**: 创建一个**可学习的** $D$ 维向量 $x_{class}$（与块嵌入具有相同的维度 $D$）。
*   **拼接**: 将这个 $x_{class}$ 向量**拼接 (Concatenate)** 到 $N$ 个块嵌入序列 $z_{patch}$ 的**最前面**。

**详细解释**:

1.  **可学习参数**:
    *   [CLS] token 本身并不对应输入图像的任何特定区域（不像其他的 token 对应图像块/patch）。
    *   它不是一个“空”的占位符。它会被初始化为一个**独立的、可学习的嵌入向量 (learnable embedding vector)**。
    *   这个嵌入向量的维度通常与 patch 嵌入的维度 (embed_dim) 相同，这样它才能和其他 patch 嵌入拼接在一起形成序列。

2.  **初始化**:
    *   在模型训练开始之前，这个 [CLS] token 的嵌入向量会像网络中的其他权重（比如线性层的权重、偏置）一样被**随机初始化**（通常是根据某种分布，如正态分布或均匀分布，取较小的值）。
    *   所以，它一开始确实没有包含关于当前输入图像的“具体内容”，但它有一个**初始的数值表示**。

3.  **学习过程**:
    *   这个初始化的 [CLS] token 嵌入向量会**参与整个训练过程**。
    *   它会和 patch 嵌入一起加上位置编码。
    *   然后，整个序列（[CLS] token + patch tokens）会通过 Transformer Encoder 层。
    *   在 Transformer 的自注意力 (Self-Attention) 机制中，[CLS] token 会与其他所有 token（包括它自己和其他 patch token）进行交互，**聚合来自整个图像的信息**。
    *   通过反向传播，这个 [CLS] token 的嵌入向量以及模型的所有其他参数都会被**不断更新和学习**，目标是让**最终**通过 Transformer Encoder 后的 [CLS] token 的状态能够最好地用于下游任务（通常是图像分类）。

**总结**:

[CLS] token 的嵌入是一个**专门为它创建的可学习参数**。它以随机值开始，**没有预设的“具体内容”**，但也不是“空的”。它的**最终意义和内容是在训练过程中通过与图像的其他部分交互而学习到的**，目的是在 Transformer 处理后，它的最终状态能代表整个图像的全局信息，用于最终的分类决策。

你可以把它想象成一个特殊的“代表”，一开始它什么都不知道（随机初始化），但在“会议”（Transformer层）中，它听取了所有其他“成员”（patch tokens）的发言，并结合自己的理解，最终形成了一个总结报告（最终的 [CLS] 输出），这个报告用来做最后的决定（分类）。这个“代表”本身的能力（它的嵌入向量）也是在一次次开会（训练）中不断提升的。

#### 4. 位置嵌入 (Positional Embedding)

*   **目的**: 标准的 Transformer 架构是**排列不变 (Permutation Invariant)** 的，即打乱输入序列的顺序不会改变输出结果（除了对应位置的输出会跟着移动）。然而，对于图像来说，**图像块的相对空间位置**包含了重要的结构信息（例如，“眼睛”通常在“鼻子”上方）。为了让 Transformer 模型能够利用这种空间信息，必须显式地将位置信息注入到输入中。
*   **过程**:
    *   创建一个**可学习的**位置嵌入矩阵 $E_{pos}$。由于现在序列包含了 $N$ 个块嵌入和 1 个 [CLS] Token，总共有 $N+1$ 个元素，每个元素的维度是 $D$，所以 $E_{pos}$ 的维度是 $(N+1) \times D$。
    *   $E_{pos}$ 的每一行对应序列中的一个位置（第 0 行对应 [CLS] Token，第 1 到 N 行对应 N 个图像块）。这些行向量在训练开始时被初始化（例如，从标准正态分布采样），并在训练过程中被学习。
    *   将位置嵌入 $E_{pos}$ **直接加到 (Element-wise Addition)** 拼接了 [CLS] Token 的块嵌入序列上。也就是说，序列中的第 $i$ 个嵌入向量会加上 $E_{pos}$ 的第 $i$ 行向量。
*   **数学表示**: $z_0 = [x_{class}; z_{patch}] + E_{pos}$
    *   $z_0$: 这是最终输入给 Transformer 编码器的序列，形状为 $(N+1, D)$。
    *   $[x_{class}; z_{patch}]$: 表示将 [CLS] Token 向量和 $N$ 个块嵌入向量按顺序拼接起来，形成一个 $(N+1, D)$ 的矩阵。
    *   $E_{pos}$: 形状为 $(N+1, D)$ 的可学习位置嵌入矩阵。
    *   `+`: 表示逐元素相加。
*   在标准的 Vision Transformer (ViT) 代码实现中，这个 `self.pos_embed` **通常是可训练的参数 (learnable parameter)**。解释如下：
    1.  **可训练性 (Trainable):**
        *   `self.pos_embed` 通常被定义为一个 `torch.nn.Parameter`。这意味着它和模型中的其他权重（比如卷积核、全连接层的权重）一样，会在训练过程中通过反向传播和优化器（如 Adam、SGD）进行学习和更新。
        *   模型会自己学习到最适合当前任务和数据集的位置表示方式。虽然也有使用固定（如正弦/余弦）位置编码的变种，但可学习的位置嵌入在 ViT 中更常见且效果通常不错。

    2.  **形状匹配 (Shape Matching):**
        *   **输入 `x` 的形状:** `[batch_size, num_patches + 1, embed_dim]`
            *   `batch_size`: 一批处理多少张图片。
            *   `num_patches + 1`: 图片被切分成 `num_patches` 个块 (patch)，再加上一个额外的 [CLS] token 用于分类，所以序列长度是 `num_patches + 1`。
            *   `embed_dim`: 每个 patch (以及 [CLS] token) 经过 Embedding 层后转换成的向量维度。
        *   **`self.pos_embed` 的形状:** `[1, num_patches + 1, embed_dim]`
            *   `1`: 这个维度是 1，因为这组学习到的位置编码对于**同一批次中的所有图片都是一样的**。我们只需要学习一套位置编码。
            *   `num_patches + 1`: 必须和输入序列的长度完全一致，保证每个 token (patch embedding 或 [CLS] token embedding) 都有一个对应的位置编码向量。
            *   `embed_dim`: 必须和 token embedding 的维度一致，这样才能进行**元素级别的相加 (element-wise addition)**。
        *   **相加操作:** `x = x + self.pos_embed`
            *   PyTorch 的广播机制 (Broadcasting) 在这里起作用。它会自动将 `self.pos_embed` 的第一个维度（大小为 1）扩展到 `batch_size`，使其形状变为 `[batch_size, num_patches + 1, embed_dim]`，然后才能和 `x` 对应元素相加。
            *   相加的目的是将学习到的位置信息注入到每个 token embedding 中，这样后续的 Transformer Encoder 层就能同时利用 token 的内容信息和位置信息。


#### 5. Transformer 编码器 (Transformer Encoder)

![](../../../../99_Assets%20(资源文件)/images/image-20250424112225451.png)

*   **目的**: 这是 ViT 的核心计算引擎。它的任务是处理包含内容（来自块嵌入）和位置信息（来自位置嵌入）的输入序列 $z_0$，通过多层自注意力机制**捕捉图像块之间以及块与 [CLS] Token 之间的复杂依赖关系（包括长距离依赖）**，并生成一个信息更丰富的表示序列 $z_L$。
*   **过程**:
    *   编码器由 $L$ 个完全相同的 **Transformer Block** 堆叠而成。序列 $z_0$ 首先输入第一个 Block，其输出 $z_1$ 输入第二个 Block，依此类推，直到最后一个 Block 输出 $z_L$。
    *   **每个 Transformer Block 内部结构**:
        *   **层归一化 (Layer Normalization, LN)**: 在进入主要计算层之前，对输入进行归一化。LN 对每个样本（在这里是序列中的每个 Token 嵌入）的所有特征维度进行归一化，使其均值为 0，方差为 1（然后可能会进行仿射变换）。这有助于稳定训练动态，加速收敛，并减少对初始化值的敏感性。
        *   **多头自注意力 (Multi-Head Self-Attention, MHSA)**:
            *   **自注意力 (Self-Attention)**: 允许序列中的每个 Token（包括 [CLS] Token 和所有块 Token）“关注”序列中的所有其他 Token（包括自身）。对于每个 Token，模型计算它与所有 Token 的“相关性”得分（通过 Query 和 Key 的点积），然后使用这些得分（经过 Softmax 归一化）对所有 Token 的 Value 表示进行加权求和，得到该 Token 的新表示。这使得每个 Token 的输出表示都融入了来自整个序列的上下文信息。
            *   **Q, K, V**: 输入序列 $z_{l-1}$（经过 LN）通过**三个不同的、可学习的线性变换**分别生成 Query (Q), Key (K), 和 Value (V) 矩阵。
            *   **缩放点积注意力**: $Attention(Q, K, V) = softmax(\frac{QK^T}{\sqrt{d_k}}) V$。除以 $\sqrt{d_k}$（$d_k$ 是 Key 向量的维度）是为了防止点积结果过大导致 Softmax 函数梯度消失。
            *   **多头 (Multi-Head)**: 与其执行一次 $D$ 维的自注意力，不如将 $D$ 维空间分割成 $h$ 个“头”（Head），每个头的维度是 $d_h = D/h$。在每个头内部独立地进行 Q, K, V 的线性变换和缩放点积注意力计算。然后将 $h$ 个头的输出结果拼接 (Concatenate) 起来，再通过一个最终的线性变换层，将维度恢复到 $D$。这样做的好处是允许模型在不同的表示子空间中并行地关注来自不同位置 不同方面的信息，增强了模型的表达能力。
        *   **残差连接 (Residual Connection)**: MHSA 层的输入（即 LN 之前的 $z_{l-1}$）被**直接加到** MHSA 层的输出上。即 $Output_{MHSA} = MHSA(LN(z_{l-1})) + z_{l-1}$。
        *   **层归一化 (Layer Normalization, LN)**: 对经过残差连接的结果再次进行层归一化。
        *   **多层感知机 (MLP) / 前馈网络 (Feed Forward Network, FFN)**:
            *   这是一个**独立应用于序列中每个 Token 位置**的前馈神经网络。在 ViT 中，它通常由**两个线性（全连接）层**组成。
            *   第一个线性层将维度从 $D$ 扩展到一个更大的中间维度（例如 $4D$），然后应用一个**非线性激活函数**，ViT 中常用的是 **GELU (Gaussian Error Linear Unit)**。
            *   （可选，但常见）在 GELU 之后通常会接一个 **Dropout** 层用于正则化。
            *   第二个线性层将维度从中间维度压缩回 $D$。
            *   （可选，但常见）在第二个线性层之后也可能接一个 **Dropout** 层。
            *   **结构**: Linear($D \rightarrow 4D$) -> GELU -> Dropout -> Linear($4D \rightarrow D$) -> Dropout
        *   **残差连接 (Residual Connection)**: MLP 层的输入（即第二次 LN 之前的 $z'_l = Output_{MHSA}$）被**直接加到** MLP 层的输出上。即 $Output_{MLP} = MLP(LN(z'_l)) + z'_l$。
    *   **残差连接的重要性**: 允许梯度在反向传播时更容易地流过深层网络，有效缓解了梯度消失问题，使得训练非常深的 Transformer 模型成为可能。
    *   **数学表示 (第 $l$ 层，从 $z_{l-1}$ 到 $z_l$)**:
        1.  $z'_{l} = MHSA(LN(z_{l-1})) + z_{l-1}$  （第一个子层：MHSA + 残差）
        2.  $z_l = MLP(LN(z'_{l})) + z'_{l}$     （第二个子层：MLP + 残差）

#### 6. 分类头 (Classification Head)

*   **目的**: 利用 Transformer 编码器学习到的丰富表示来进行最终的图像分类。
*   **过程**:
    *   **提取 [CLS] Token 表示**: 从 Transformer 编码器的最后一层输出 $z_L$（形状为 $(N+1, D)$）中，**只取出第一个向量**，即对应于 [CLS] Token 的输出向量 $z_L^0$（形状为 $(1, D)$）。这个向量被认为是整个图像的聚合表示。
    *   **（可选）层归一化**: 在 ViT 的原始实现中，通常会在将 $z_L^0$ 送入最终分类器之前再应用一次**层归一化 (LN)**。
    *   **最终线性层**: 将（可能经过 LN 的）$z_L^0$ 输入到一个简单的**线性层（全连接层）**。这个线性层的输出维度等于**目标类别的数量**（例如，ImageNet 有 1000 类）。这一层的输出是每个类别的 **logits**（未归一化的对数概率）。
    *   **获得概率**: 如果需要得到概率分布，可以将 logits 通过 Softmax 函数进行转换。
*   **数学表示**: $y = Linear(LN(z_L^0))$
    *   $z_L^0$: 最后一层编码器输出的 [CLS] Token 向量。
    *   $LN$: （可选但常用的）层归一化。
    *   $Linear$: 输出维度为类别数的最终线性分类层。
    *   $y$: 模型的最终输出（logits）。

通过以上六个步骤，ViT 将一张图像转换成一系列嵌入向量，利用 Transformer 强大的序列处理能力捕捉全局依赖关系，并最终通过 [CLS] Token 的输出进行分类。整个模型（除了初始的图像分块操作）是端到端可训练的。

### 关键特性与讨论

*   **缺乏归纳偏置 (Inductive Bias)**: 与 CNN 不同，ViT 在设计上几乎没有内置的图像特有的归纳偏置（如局部性、平移不变性）。CNN 的卷积核天然地关注局部区域，并且权重共享使得它能检测到图像中任何位置的相同模式。ViT 则必须从数据中从头学习这些空间关系。
*   **数据需求**: 正是因为缺乏归纳偏置，ViT 通常需要**非常大**的数据集进行预训练（例如 ImageNet-21k 或 JFT-300M）才能学习到强大的视觉表示，并超越同等规模的 CNN。在较小的数据集上，其性能可能不如 CNN。
*   **全局感受野**: MHSA 使得模型在早期层就能拥有全局感受野，每个块都能与其他所有块进行信息交互，这与 CNN 逐层扩大感受野的方式不同。这使得 ViT 非常擅长捕捉图像的全局上下文信息。
*   **可扩展性**: Transformer 架构已被证明在模型规模和数据量方面具有良好的可扩展性。增加模型深度 ($L$)、宽度 ($D$) 或头的数量，通常能带来性能提升（需要配合足够的数据）。
*   **激活函数的选择**: ViT 沿用了 NLP Transformer 的成功实践，在 MLP 层中使用了 **GELU** 激活函数，这被认为对其优异性能有所贡献。

### 总结

![](../../../../99_Assets%20(资源文件)/images/721d65bffe3543a7e87b970a4a15eb54.png)

ViT 通过将图像巧妙地转换为序列数据，成功地将强大的 Transformer 模型引入了计算机视觉领域。它摒弃了 CNN 固有的归纳偏置，依赖大规模数据学习图像的内在结构和关系，并利用 GELU 等先进组件，展现了优异的性能和扩展性，是近年来计算机视觉领域最重要的突破之一。理解其分块、嵌入、位置编码和 Transformer 核心机制（包括 MHSA 和使用 GELU 的 MLP）是掌握 ViT 的关键。

### VIT的理解

这正是因为ViT（Vision Transformer）做了一件天才般的事情：**它把图像问题强行转换成了一个语言问题！**

1.  **图像 -> 句子**: ViT先把一张大图片切成很多个小图块 (Patches)。这就好比把一篇长文章拆成一个个词语。`[patch1, patch2, patch3, ...]` 这个序列，就等价于NLP里的 `[word1, word2, word3, ...]`。
2.  **图块 -> 词嵌入**: 接着，ViT把每个小图块拉平成一个一维向量，再通过一个全连接层，把它变成一个固定维度的向量（比如768维）。这**完全等价于**NLP里的词嵌入过程！
3.  **应用Transformer**: 之后，ViT就可以用和处理语言一模一样的Transformer结构来处理这些“图像词嵌入”了。

所以，在ViT的内部，当数据进入Transformer模块后，它的形态就是一个序列的向量，**不再有CNN里“通道”和“特征图”的概念**。因此，ViT里使用的是**层归一化 (Layer Normalization)**，而不是批归一化或实例归一化，这和BERT、GPT等语言模型的选择是完全一致的！
