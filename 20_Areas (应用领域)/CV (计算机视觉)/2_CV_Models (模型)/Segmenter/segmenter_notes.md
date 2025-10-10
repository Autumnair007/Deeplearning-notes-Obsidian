---
type: concept-note
tags:
  - cv
  - semantic-segmentation
  - transformer
  - segmenter
  - vit
  - encoder-decoder
  - full-supervision
status: done
model: Segmenter
year: 2021
---
参考资料：[[2105.05633v3\] Segmenter: Transformer for Semantic Segmentation](https://arxiv.org/abs/2105.05633v3)

------

## Segmenter 模型深度解析

Segmenter 是一个在语义分割领域具有里程碑意义的模型。它首次证明，一个纯粹基于 Transformer 的架构，无需复杂的解码器设计或依赖卷积网络（CNN）的固有特性（如局部归纳偏置），也能够在语义分割任务上取得顶尖性能。其核心思想是将像素级的分割问题巧妙地重构为一个序列到序列（sequence-to-sequence）的预测任务。

### **核心思想：分割即序列预测**

传统的语义分割模型，如 FCN 和 U-Net，普遍采用编码器-解码器结构。编码器（通常是 CNN）负责提取层次化的图像特征，而解码器则逐步将这些特征上采样，以恢复空间分辨率并进行像素级别的分类。这种方法的优势在于 CNN 强大的局部特征提取能力和通过跳跃连接保留高分辨率细节的能力。
Segmenter 则另辟蹊径，完全摒弃了卷积操作，并借鉴了自然语言处理（NLP）领域中 Transformer 模型的巨大成功。它将图像视为一个“序列”进行处理：首先将图像分解为一系列的小图像块（Patches），然后将这个图像块序列作为 Transformer 的输入，最终输出每个图像块对应的类别标签，再通过简单的上采样映射回像素级的分割图。这种方法的核心优势在于 Transformer 能够捕捉图像中任意两个块之间的长距离依赖关系，从而建立真正的全局上下文。

### **模型整体架构**

Segmenter 模型由两个主要部分构成：
1.  **编码器（Encoder）**: 一个标准的 Vision Transformer (ViT)，负责将输入的图像块序列编码成富含全局上下文信息的深度特征表示。
2.  **解码器（Decoder）**: 一个被称为“Mask Transformer”的模块，它接收编码器的输出，并将其解码为最终的分割掩码（Segmentation Mask）。

![](../../../../99_Assets%20(资源文件)/images/239a2c815ec24a7c5ade67da91633ae1.png)

<center>Segmenter 模型结构示意图</center>

### **1. 编码器：Vision Transformer (ViT)**

编码器的职责是从输入图像中提取出强大的特征表示，其设计完全遵循标准的 Vision Transformer (ViT) 架构。它的核心任务是将二维的像素网格转换为一系列包含丰富上下文信息的编码向量。

#### **1.1. 图像分块与嵌入 (Patching & Embedding)**

为了让本质上处理序列数据的 Transformer 能够处理 2D 图像，首先需要将图像转换成一个 1D 的向量序列。这个过程包含三个关键步骤：

*   **分块 (Patching)**: 将一张分辨率为 $H \times W$、拥有 $C$ 个通道的输入图像 $x \in \mathbb{R}^{H \times W \times C}$，分割成 $N$ 个不重叠的小图像块（Patches）。每个块的分辨率为 $P \times P$。因此，序列的长度（即图像块的数量）为 $N = \frac{H \times W}{P^2}$。
*   **展平与线性投射 (Flatten & Linear Projection)**: 每个图像块在物理上被“展平”（Flatten）成一个一维向量，其维度为 $P \times P \times C$。然后，通过一个可学习的线性投射矩阵 $E \in \mathbb{R}^{D \times (P^2 C)}$，将这个高维向量映射到一个 $D$ 维的嵌入空间（$D$ 是 Transformer 内部的工作维度）。这个过程的输出被称为**块嵌入 (Patch Embeddings)**，记为 $x^0 = [Ex_1, ..., Ex_N] \in \mathbb{R}^{N \times D}$。
*   **位置嵌入 (Positional Embedding)**: Transformer 的自注意力机制是置换不变的（Permutation-invariant），它本身无法感知序列中元素的顺序。为了让模型理解每个图像块的原始空间位置，必须显式地加入位置信息。Segmenter 采用可学习的**位置嵌入** $pos \in \mathbb{R}^{N \times D}$，它是一个与块嵌入维度完全相同的参数矩阵，在训练开始前随机初始化，并在训练过程中与其他模型参数一同被优化。最终的输入序列 $z^0$ 是块嵌入和位置嵌入**逐元素相加**的结果。
$$
z^0 = x^0 + pos = [Ex_1+pos_1, ..., Ex_N+pos_N]
$$
这里的 $x_{patch}^i$ 是第 $i$ 个展平后的图像块，$E$ 是可学习的线性投射矩阵。经过此步骤，一张 2D 图像就成功转换为一个包含空间位置信息的向量序列 $z^0 \in \mathbb{R}^{N \times D}$，这便是 Transformer 编码器的最终输入。

#### **1.2. Transformer 编码器层**

该向量序列 $z^0$ 随后被送入一个由 $L$ 层堆叠而成的标准 Transformer 编码器中。每一层都由两个核心子模块交替构成，并严格遵循 `Pre-Norm` 结构（即在每个核心模块作用前进行层归一化），这有助于稳定训练过程和加速收敛。
对于第 $i$ 层（$i \in \{1, ..., L\}$），其内部计算流程如下：

1.  对第 $i-1$ 层的输出 $z^{i-1}$ 先进行层归一化（Layer Normalization），然后送入多头自注意力模块，其输出与原始输入 $z^{i-1}$ 进行残差连接，得到中间结果 $a^{i-1}$。
$$
a^{i-1} = \text{MSA}(\text{LN}(z^{i-1})) + z^{i-1}
$$
2.  对中间结果 $a^{i-1}$ 同样先进行层归一化，然后送入一个 MLP 块，其输出与输入 $a^{i-1}$ 进行残差连接，得到第 $i$ 层的最终输出 $z^i$。
$$
z^i = \text{MLP}(\text{LN}(a^{i-1})) + a^{i-1}
$$
经过 $L$ 层的计算后，编码器最终输出一个深度融合了全局上下文信息的块嵌入序列 $z_L \in \mathbb{R}^{N \times D}$。

*   **多头自注意力 (Multi-Head Self-Attention, MSA)**: 这是 Transformer 的灵魂。它允许序列中的每一个块嵌入都能关注到序列中所有其他的块嵌入，并根据相关性动态地加权聚合信息来更新自身的表示。
    
    *   **Q, K, V 的生成**：对于输入序列 $Z_{in} \in \mathbb{R}^{N \times D}$ (即 $\text{LN}(z^{i-1})$)，MSA模块首先通过三个独立的可学习线性变换（即三个全连接层，权重分别为 $W_Q, W_K, W_V$）将其投影到三个不同的空间，得到查询（Query）、键（Key）和值（Value）矩阵：$Q = Z_{in}W_Q$, $K = Z_{in}W_K$, $V = Z_{in}W_V$。这三个矩阵的维度通常是 $\mathbb{R}^{N \times d}$，其中 $d$ 是每个注意力头的维度。
    *   **注意力计算**：其核心是缩放点积注意力（Scaled Dot-Product Attention）。$Q$ 和 $K$ 的矩阵乘法 $QK^T$ 计算了序列中每个token的“查询”与所有token的“键”之间的原始相似度分数。$\sqrt{d}$ 是一个缩放因子，用于稳定梯度，防止点积结果过大导致 softmax 函数进入饱和区。Softmax 函数将分数归一化为注意力权重，这些权重表示每个查询应该从各个值中提取多少信息。最后，将权重矩阵与 $V$ 相乘，得到加权后的特征。
    
    $$
    \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right)V
    $$
    
    *   **“多头”机制**：该过程会并行地进行多次（例如 $h$ 次，即 $h$ 个头）。每个头都有自己独立的 $W_Q, W_K, W_V$ 权重矩阵，可以从不同的表示子空间学习信息。最后，所有 $h$ 个头的输出结果被拼接（concatenate）起来，并通过另一次线性变换投影回原始的 $D$ 维空间。这使得模型能够高效地捕捉图像范围内的多种依赖关系。
    
    *   **详细的数学解释**：我们来拆解这个过程。假设输入序列包含 $N$ 个 token，每个 token 的 embedding 是一个行向量 $z_i \in \mathbb{R}^{1 \times D}$。对于单个注意力头，我们有权重矩阵 $W_Q, W_K, W_V \in \mathbb{R}^{D \times d}$。
        
        1.  **生成 Q, K, V 向量**：对序列中的每一个 token $z_i$（$i=1, \dots, N$），我们都生成其对应的查询、键、值向量：$q_i = z_i W_Q$, $k_i = z_i W_K$, $v_i = z_i W_V$。每个向量都是 $1 \times d$ 维的。
        2.  **计算注意力分数**：为了更新第 $i$ 个 token 的表示，我们用它的查询向量 $q_i$ 与**所有** token 的键向量 $k_j$（$j=1, \dots, N$）进行点积运算，得到一个标量“注意力分数” $e_{ij} = q_i \cdot k_j^T$。这个分数衡量了第 $i$ 个 token 对第 $j$ 个 token 的“关注程度”。
        3.  **形成分数矩阵**：将所有这些分数汇集起来，就形成了一个注意力分数矩阵 $E \in \mathbb{R}^{N \times N}$，其中 $E_{ij} = q_i k_j^T$。这在宏观上等价于 $E = QK^T$。
        4.  **缩放与归一化**：将分数矩阵 $E$ 的每个元素都除以缩放因子 $\sqrt{d}$，然后对矩阵的每一行独立地应用 Softmax 函数。对于第 $i$ 行，我们得到注意力权重向量 $\alpha_i = [\alpha_{i1}, \dots, \alpha_{iN}] = \text{softmax}([e_{i1}/\sqrt{d}, \dots, e_{iN}/\sqrt{d}])$。其中 $\sum_{j=1}^{N} \alpha_{ij} = 1$。
        5.  **加权求和**：最后，第 $i$ 个 token 的新表示 $z'_i \in \mathbb{R}^{1 \times d}$ 是所有 token 的值向量 $v_j$ 的加权和，权重就是刚算出的 $\alpha_{ij}$。
        
        $$
        z'_i = \sum_{j=1}^{N} \alpha_{ij} v_j
        $$
        
        这在宏观上对应于最终的矩阵运算：$Z' = \text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right)V$，其中 $Z'$ 的第 $i$ 行就是 $z'_i$。
        
    *   **一个具体的计算示例**：假设我们有一个极简的序列，包含2个 token（$N=2$）。输入 embedding 维度为4（$D=4$），单个注意力头的维度为3（$d=3$）。
        
        *   **输入**：$Z_{in} = \begin{pmatrix} 1 & 0 & 1 & 0 \\ 0 & 1 & 0 & 1 \end{pmatrix}$
        *   **权重矩阵** (随机设定)：
            $W_Q = \begin{pmatrix} 1 & 0 & 1 \\ 0 & 1 & 0 \\ 1 & 0 & 0 \\ 0 & 1 & 1 \end{pmatrix}$, $W_K = \begin{pmatrix} 0 & 1 & 1 \\ 1 & 0 & 1 \\ 1 & 1 & 0 \\ 0 & 0 & 1 \end{pmatrix}$, $W_V = \begin{pmatrix} 0 & 2 & 0 \\ 1 & 0 & 3 \\ 1 & 1 & 0 \\ 0 & 0 & 1 \end{pmatrix}$
        *   **1. 计算 Q, K, V**：
            $Q = Z_{in}W_Q = \begin{pmatrix} 2 & 0 & 1 \\ 0 & 2 & 1 \end{pmatrix}$
            $K = Z_{in}W_K = \begin{pmatrix} 1 & 2 & 1 \\ 1 & 0 & 2 \end{pmatrix}$
            $V = Z_{in}W_V = \begin{pmatrix} 1 & 3 & 0 \\ 1 & 0 & 4 \end{pmatrix}$
        *   **2. 计算注意力分数** ($E=QK^T$)：
            $QK^T = \begin{pmatrix} 2 & 0 & 1 \\ 0 & 2 & 1 \end{pmatrix} \begin{pmatrix} 1 & 1 \\ 2 & 0 \\ 1 & 2 \end{pmatrix} = \begin{pmatrix} 3 & 4 \\ 5 & 2 \end{pmatrix}$
        *   **3. 缩放与归一化** (缩放因子 $\sqrt{d}=\sqrt{3} \approx 1.732$)：
            $\frac{QK^T}{\sqrt{d}} \approx \begin{pmatrix} 1.732 & 2.309 \\ 2.887 & 1.155 \end{pmatrix}$
            
            对每一行应用 Softmax:
            $\text{softmax}([1.732, 2.309]) = [0.359, 0.641]$
            $\text{softmax}([2.887, 1.155]) = [0.850, 0.150]$
            
            注意力权重矩阵 $A = \begin{pmatrix} 0.359 & 0.641 \\ 0.850 & 0.150 \end{pmatrix}$
        *   **4. 加权求和** ($Z'=AV$)：
            $Z' = \begin{pmatrix} 0.359 & 0.641 \\ 0.850 & 0.150 \end{pmatrix} \begin{pmatrix} 1 & 3 & 0 \\ 1 & 0 & 4 \end{pmatrix} = \begin{pmatrix} 0.359 \cdot 1 + 0.641 \cdot 1 & 0.359 \cdot 3 + 0.641 \cdot 0 & 0.359 \cdot 0 + 0.641 \cdot 4 \\ 0.850 \cdot 1 + 0.150 \cdot 1 & 0.850 \cdot 3 + 0.150 \cdot 0 & 0.850 \cdot 0 + 0.150 \cdot 4 \end{pmatrix}$
            $Z' = \begin{pmatrix} 1.000 & 1.077 & 2.564 \\ 1.000 & 2.550 & 0.600 \end{pmatrix}$
        *   **结果**：$Z'$ 就是这个单头注意力模块的输出。第一个 token 的新表示 $[1.000, 1.077, 2.564]$ 是由 $35.9\%$ 的第一个 token 信息和 $64.1\%$ 的第二个 token 信息聚合而成。这清晰地展示了信息是如何在序列的不同部分之间动态流动和融合的。
    
*   **前馈网络 (Feed-Forward Network, FFN)**: 这是一个简单的多层感知机（MLP），通常由两个线性层和一个 GELU 激活函数组成。它对自注意力层的输出进行非线性的逐点变换，进一步增强模型的表达能力。

#### **编码器数据流与张量变化详解**

让我们以一个具体的例子来追踪数据在编码器中的流动过程：
1.  **输入图像**: 假设我们有一张输入图片 $x$，其尺寸为 $224 \times 224 \times 3$。
    *   $x \in \mathbb{R}^{224 \times 224 \times 3}$ ($H=224, W=224, C=3$)
2.  **分块**: 使用大小为 $16 \times 16$ 的 Patch。
    *   $P=16$。
    *   Patch 数量 $N = (224 \times 224) / (16 \times 16) = 14 \times 14 = 196$。
    *   图像被重塑为 $x_{patches} \in \mathbb{R}^{196 \times (16 \times 16 \times 3)} = \mathbb{R}^{196 \times 768}$。
3.  **线性投射**: 将每个 Patch 向量投射到 Transformer 的工作维度 $D$ (例如 $D=768$)。
    *   投射矩阵 $E \in \mathbb{R}^{768 \times 768}$。
    *   块嵌入 $x^0 = x_{patches} E^T \in \mathbb{R}^{196 \times 768}$。（这里假设 $D=P^2C$）
4.  **位置嵌入**: 添加可学习的位置信息。
    *   位置嵌入矩阵 $pos \in \mathbb{R}^{196 \times 768}$。
    *   最终输入序列 $z^0 = x^0 + pos \in \mathbb{R}^{196 \times 768}$。
5.  **Transformer 层内部**: 以第 $i$ 层为例，输入为 $z^{i-1} \in \mathbb{R}^{196 \times 768}$。
    *   **Pre-Norm**: $LN(z^{i-1})$ 输出维度不变，仍为 $\mathbb{R}^{196 \times 768}$。
    *   **MSA (假设 12 个头, $h=12$)**:
        *   每个头的维度 $d = D/h = 768 / 12 = 64$。
        *   为每个头生成 $Q_j, K_j, V_j \in \mathbb{R}^{196 \times 64}$。
        *   $Q_j K_j^T \in \mathbb{R}^{196 \times 196}$ (注意力分数矩阵)。
        *   $\text{softmax}(\dots)V_j \in \mathbb{R}^{196 \times 64}$ (单个头的输出)。
        *   12个头的输出拼接后为 $\in \mathbb{R}^{196 \times (12 \times 64)} = \mathbb{R}^{196 \times 768}$。
        *   经过最后的线性层，MSA 输出维度为 $\mathbb{R}^{196 \times 768}$。
    *   **残差连接**: $a^{i-1} = \text{MSA}(\dots) + z^{i-1}$，维度仍为 $\mathbb{R}^{196 \times 768}$。
    *   **MLP**: 经过两个线性层和激活函数，维度变化通常是 $768 \to 3072 \to 768$。输出维度仍为 $\mathbb{R}^{196 \times 768}$。
    *   **第二个残差连接**: $z^i = \text{MLP}(\dots) + a^{i-1}$，维度为 $\mathbb{R}^{196 \times 768}$。
6.  **编码器最终输出**: 经过 $L$ 层（例如 $L=12$）后，最终输出 $z_L$ 的维度依然是 $\mathbb{R}^{196 \times 768}$。这个张量包含了 196 个图像块的深度上下文编码，将作为解码器的输入。

### **2. 解码器：Mask Transformer**

解码器的任务是将编码器输出的 $N$ 个 $D$ 维块特征 $z_L$ 转换为 $K$ 个语义类别的分割掩码（$K$ 是类别总数）。这是 Segmenter 的关键创新所在，其设计极为简洁高效。论文提出了两种解码器方案：一个简单的线性解码器作为基线，以及核心的 **Mask Transformer** 解码器。

#### **2.1. 类别嵌入 (Class Embeddings)**

Mask Transformer 解码器引入了一组可学习的参数化向量，称为**类别嵌入**（Class Embeddings），记为 $cls \in \mathbb{R}^{K \times D}$。每一个语义类别（如“人”、“车”、“天空”）都对应一个 $D$ 维的向量。这些向量在模型初始化时被随机赋值，并在训练过程中与模型的其他参数一同被学习和优化。最终，它们会编码每个类别的抽象语义概念，可以被视为每个类别的“语义原型”或“概念探针”。

#### **2.2. 从块特征到分割掩码：Mask Transformer 的核心机制**

与简单的线性解码器（仅用一个线性层将 patch 特征映射到类别 logits）不同，Mask Transformer 设计了一个更强大的机制，让类别信息和图像特征进行深度交互。其核心计算过程如下：
1.  **特征交互**: 这是 Mask Transformer 最关键的一步。它并非直接计算相似度，而是先将编码器输出的 patch 编码序列 $z_L \in \mathbb{R}^{N \times D}$ 和可学习的类别嵌入 $cls \in \mathbb{R}^{K \times D}$ 在序列维度上拼接（concatenate）起来，形成一个更长的混合序列 $[z_L; cls] \in \mathbb{R}^{(N+K) \times D}$。这个混合序列被送入一个由 $M$ 层组成的、结构与编码器层类似的 Transformer 模块（即解码器本身也是一个 Transformer）。在这个模块中，通过多头自注意力，patch 特征之间、类别嵌入之间、以及最重要的——**patch 特征与类别嵌入之间**都能进行信息交互。这个过程可以被解读为：让每个类别嵌入去“查询”所有的图像块特征，并根据自注意力机制，将最相关的图像块信息聚合起来，从而生成一个被当前图像内容“上下文感知”或“动态细化”过的新类别嵌入。同时，图像块的表示也会被类别信息所影响。
2.  **输出分离与归一化**: 经过 $M$ 层 Transformer 解码器处理后，输出的混合序列被重新分离成两部分：经过优化的 patch 编码 $z'_M \in \mathbb{R}^{N \times D}$ 和经过优化的类别嵌入 $c \in \mathbb{R}^{K \times D}$。根据论文，输出的 patch 编码 $z'_M$ 会进行 L2 归一化。
3.  **生成掩码 (Mask Generation)**: 接下来，通过计算 L2 归一化后的 patch 编码 $z'_M$ 和优化后的类别嵌入 $c$ 之间的矩阵乘法（点积），来为所有类别一次性生成所有 patch 的类别得分。
$$
Masks(z'_M, c) = z'_M c^T
$$
这里 $z'_M \in \mathbb{R}^{N \times D}$，$c^T \in \mathbb{R}^{D \times K}$。运算结果 $Masks \in \mathbb{R}^{N \times K}$ 是一个得分矩阵。矩阵中的元素 $(i, j)$ 表示第 $i$ 个图像块属于第 $j$ 个类别的“得分”或“对齐程度”。这个过程可以直观地理解为：用被图像内容动态调整过的“语义探针”$c$ 去探测同样被优化的图像区域特征 $z'_M$，看看哪个区域的特征与之最匹配。这也被称为一种**动态滤波器**机制，其中类别嵌入 $c$ 充当了随输入变化的类别特定滤波器。
4.  **重塑 (Reshape)**: 得到的 $N \times K$ 的得分矩阵被重塑成一个三维的特征图，其空间尺寸为 $H/P \times W/P \times K$。这一步操作恢复了图像块在原始图像中的空间布局，生成了一个低分辨率的类别激活图。
5.  **上采样 (Upsampling)**: 这个低分辨率的 $H/P \times W/P \times K$ 特征图通过简单的双线性插值（Bilinear Interpolation）被上采样到原始图像的分辨率 $H \times W \times K$，从而得到全分辨率的分割得分图。
6.  **最终输出**: 最后，在每个像素位置上，沿着类别维度（$K$个通道）应用 `softmax` 函数，即可得到该像素属于每个类别的最终概率分布。论文中还提到，在 softmax 之后可以再接一个层归一化。这样处理后，可以确保每个像素的所有类别概率之和为 1（“软排他”），形成最终的分割图。训练时的损失函数通常是标准的像素级交叉熵损失。

#### **解码器数据流与张量变化详解**

我们延续编码器部分的例子来追踪数据在 Mask Transformer 解码器中的流动：
1.  **输入特征**:
    *   编码器输出的 patch 编码：$z_L \in \mathbb{R}^{196 \times 768}$ ($N=196, D=768$)
    *   可学习的类别嵌入：假设有 $K=21$ 个类别（如 Pascal VOC 数据集），则 $cls \in \mathbb{R}^{21 \times 768}$。
2.  **特征拼接**:
    *   将两者在序列维度上拼接，形成解码器输入：$[z_L; cls] \in \mathbb{R}^{(196+21) \times 768} = \mathbb{R}^{217 \times 768}$。
3.  **Transformer 解码器处理**:
    *   这个 $217 \times 768$ 的张量流经 $M$ 个 Transformer 层。在每一层内部，自注意力机制在 $217 \times 217$ 的维度上计算，允许所有 patch 和所有类别嵌入之间进行信息交换。
    *   经过 $M$ 层后，输出的张量维度仍然是 $\mathbb{R}^{217 \times 768}$。
4.  **特征分离**:
    *   将输出张量按原先的拼接方式分离：
        *   优化的 patch 编码 $z'_{M\_raw} \in \mathbb{R}^{196 \times 768}$。
        *   优化的类别嵌入 $c \in \mathbb{R}^{21 \times 768}$。
5.  **L2 归一化**:
    *   对 patch 编码进行 L2 归一化：$z'_M = \text{L2Norm}(z'_{M\_raw}) \in \mathbb{R}^{196 \times 768}$。
6.  **生成掩码**:
    *   计算矩阵乘法：$Masks = z'_M \cdot c^T$。
    *   $z'_M \in \mathbb{R}^{196 \times 768}$，$c^T \in \mathbb{R}^{768 \times 21}$。
    *   输出的 patch 级得分矩阵 $Masks \in \mathbb{R}^{196 \times 21}$。
7.  **重塑**:
    *   将得分矩阵重塑为空间特征图：$s_{mask} \in \mathbb{R}^{14 \times 14 \times 21}$ (因为 $H/P = 224/16 = 14$, $W/P = 224/16 = 14$)。
8.  **上采样**:
    *   通过双线性插值，将特征图上采样到原始图像分辨率：$s \in \mathbb{R}^{224 \times 224 \times 21}$。
9.  **最终分类**:
    *   对最后一个维度（类别维度）应用 softmax，得到最终的像素级概率图 $\in \mathbb{R}^{224 \times 224 \times 21}$。每个像素点的 $21$ 个通道值之和为 $1$。

### **总结与评价**

总而言之，Segmenter 模型的核心步骤可以概括为：
1.  **分块与序列化**: 将 2D 图像转换为 1D 的图像块序列。
2.  **ViT 编码**: 使用一个标准的 Transformer 编码器提取序列中每个块的深度全局上下文特征 $z_L$。
3.  **类别查询**: 引入可学习的类别嵌入 $c$，作为每个语义类别的可学习原型。
4.  **解码与生成掩码**: 通过计算 $z_L$ 和 $c$ 之间的点积相似度，为每个类别生成一个低分辨率的分割图，然后通过简单的双线性插值恢复到原始分辨率，最终得到像素级的分割结果。

Segmenter 的成功之处在于它用一个极为简洁、统一的 Transformer 框架优雅地解决了语义分割问题。它证明了强大的全局上下文建模能力对于密集预测任务至关重要，并且这种能力可以不依赖于 CNN 的卷积结构。Segmenter 不仅在当时的各大分割基准上取得了 SOTA 性能，更重要的是它为后续一系列基于 Transformer 的视觉模型（尤其在分割、检测等领域，如 SegFormer, MaskFormer, Mask2Former 等）铺平了道路，是计算机视觉范式变革中的一个关键奠基石。
