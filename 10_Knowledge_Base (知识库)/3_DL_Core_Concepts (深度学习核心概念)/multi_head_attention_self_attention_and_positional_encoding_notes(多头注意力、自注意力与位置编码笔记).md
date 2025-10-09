---
type: concept-note
tags:
  - nlp
  - cv
  - attention
  - transformer
  - self-attention
  - multi-head-attention
  - positional-encoding
status: done
topic: Transformer核心机制
multi_head_core: 并行计算H个子空间，然后拼接
pe_sinusoidal_advantage: 引入相对位置信息，无需训练
pe_concept: 通过相加注入位置信息，弥补自注意力的无序性
---
学习资料：[10.6. 自注意力和位置编码 — 动手学深度学习 2.0.0 documentation](https://zh-v2.d2l.ai/chapter_attention-mechanisms/self-attention-and-positional-encoding.html)

[10.5. 多头注意力 — 动手学深度学习 2.0.0 documentation](https://zh-v2.d2l.ai/chapter_attention-mechanisms/multihead-attention.html)

[详解自注意力机制中的位置编码（第一部分） - 知乎](https://zhuanlan.zhihu.com/p/352233973)（写得很好）

[详解自注意力机制中的位置编码（第二部分） - 知乎](https://zhuanlan.zhihu.com/p/354963727)

[(2 封私信 / 40 条消息) 如何理解Transformer论文中的positional encoding，和三角函数有什么关系？ - 知乎](https://www.zhihu.com/question/347678607)

[详解Transformer中Self-Attention以及Multi-Head Attention_transformer multi head-CSDN博客](https://blog.csdn.net/qq_37541097/article/details/117691873)

[(7 封私信 / 30 条消息) 位置编码背后的理论解释——傅里叶特征 (Fourier Feature）与核回归 - 知乎](https://zhuanlan.zhihu.com/p/10748639711)

------

## 自注意力 (Self-Attention)

自注意力的核心思想是计算序列中每个元素（比如单词）与其他所有元素（包括自身）的关联程度，然后根据这个关联程度（权重）来加权融合所有元素的信息，生成该元素的新表示。

1.  **输入**: 假设我们有一个输入序列，包含 $n$ 个元素（例如，$n$ 个单词的嵌入向量），每个元素的维度是 $d_{model}$。我们将整个序列表示为一个矩阵 $X$ (维度 $n \times d_{model}$)。

2.  **生成 Q, K, V**: 为了计算注意力，我们需要为每个输入元素生成三个向量：查询（Query, Q）、键（Key, K）和值（Value, V）。这是通过将输入矩阵 $X$ 分别乘以三个可学习的权重矩阵 $W_Q$, $W_K$, $W_V$ (维度均为 $d_{model} \times d_k$，通常 $d_k = d_{model} / h$，其中 $h$ 是注意力头的数量) 来实现的：
    $$
    Q = X \times W_Q
    $$
    $$
    K = X \times W_K
    $$
    $$
    V = X \times W_V
    $$
    ($Q$ 维度 $n \times d_k$, $K$ 维度 $n \times d_k$, $V$ 维度 $n \times d_v$，通常 $d_v = d_k$)

3.  **计算注意力分数**: 我们计算每个 Query 向量与所有 Key 向量的点积，来衡量它们之间的相似度或相关性。
    $$
    Scores = Q \times K^T
    $$
    这里的 $K^T$ 是 K 矩阵的转置 (维度 $d_k \times n$) 。$Scores$ 矩阵中的第 $i$ 行第 $j$ 列的元素表示第 $i$ 个元素的 Query 与第 $j$ 个元素的 Key 之间的原始注意力分数。

4.  **缩放 (Scaling)**: 为了防止点积结果过大导致 Softmax 函数的梯度变得非常小，需要对分数进行缩放。通常除以 $d_k$ 的平方根 ($\sqrt{d_k}$)。
    $$
    Scaled\_Scores = Scores / \sqrt{d_k}
    $$

5.  **Softmax 归一化**: 对缩放后的分数按行应用 Softmax 函数，将分数转换为概率分布，即得到每个元素对其他元素的注意力权重。每行的权重加起来等于 1。
    $$
    Attention\_Weights = softmax(Scaled\_Scores)
    $$
    ($Attention\_Weights$ 矩阵维度 $n \times n$)
    $Attention\_Weights$ 矩阵的第 $i$ 行表示第 $i$ 个元素对序列中所有元素的注意力权重分布。

6.  **加权求和**: 将得到的注意力权重矩阵 $Attention\_Weights$ 乘以 Value 矩阵 $V$，得到自注意力层的最终输出。输出矩阵的第 $i$ 行是序列中所有元素的 Value 向量根据第 $i$ 个元素的注意力权重进行的加权和。
    $$
    Output = Attention\_Weights \times V
    $$
    (输出矩阵维度 $n \times d_v$)

**总结公式**:
将以上步骤合并，单头自注意力的计算可以表示为：
$$
Attention(Q, K, V) = softmax( \frac{Q \times K^T}{\sqrt{d_k}} ) \times V
$$

### 自注意力计算实例

我们用一个简单的例子手动计算一次自注意力。

**场景设定:**
假设输入序列是 "thinking machines"，我们已经获得了这两个词的词嵌入向量。

*   $x_1$ (thinking): $[1, 0, 1, 0]$
*   $x_2$ (machines): $[0, 2, 0, 2]$

输入矩阵 $X$ (维度 $2 \times 4$):
$$
X = \begin{bmatrix} 1 & 0 & 1 & 0 \\ 0 & 2 & 0 & 2 \end{bmatrix}
$$
为了简化，我们设定输出的 $Q, K, V$ 向量维度 $d_k = d_v = 3$。我们需要随机初始化三个权重矩阵 $W_Q, W_K, W_V$ (维度 $d_{model} \times d_k = 4 \times 3$)。

$$
W_Q = \begin{bmatrix} 1 & 0 & 1 \\ 1 & 1 & 0 \\ 0 & 1 & 1 \\ 1 & 0 & 0 \end{bmatrix} \quad W_K = \begin{bmatrix} 0 & 1 & 0 \\ 1 & 1 & 0 \\ 1 & 0 & 1 \\ 0 & 1 & 1 \end{bmatrix} \quad W_V = \begin{bmatrix} 0 & 2 & 0 \\ 1 & 0 & 3 \\ 1 & 1 & 0 \\ 0 & 0 & 1 \end{bmatrix}
$$

**1. 生成 Q, K, V 矩阵**
$$
Q = X W_Q = \begin{bmatrix} 1 & 1 & 2 \\ 4 & 2 & 0 \end{bmatrix}
$$
$$
K = X W_K = \begin{bmatrix} 1 & 1 & 1 \\ 2 & 4 & 2 \end{bmatrix}
$$
$$
V = X W_V = \begin{bmatrix} 1 & 3 & 0 \\ 2 & 0 & 8 \end{bmatrix}
$$

**2. 计算注意力分数**
$$
Scores = QK^T = \begin{bmatrix} 1 & 1 & 2 \\ 4 & 2 & 0 \end{bmatrix} \begin{bmatrix} 1 & 2 \\ 1 & 4 \\ 1 & 2 \end{bmatrix} = \begin{bmatrix} 4 & 10 \\ 6 & 16 \end{bmatrix}
$$

**3. 缩放**
$d_k = 3$, $\sqrt{d_k} \approx 1.732$
$$
Scaled\_Scores = \frac{Scores}{\sqrt{d_k}} = \begin{bmatrix} 4/1.732 & 10/1.732 \\ 6/1.732 & 16/1.732 \end{bmatrix} = \begin{bmatrix} 2.31 & 5.77 \\ 3.46 & 9.24 \end{bmatrix}
$$

**4. Softmax 归一化**
对 $Scaled\_Scores$ 的每一行独立应用 Softmax。

*   **第一行:**
    *   $\exp(2.31) \approx 10.07$, $\exp(5.77) \approx 320.5$
    *   $softmax([2.31, 5.77]) = [\frac{10.07}{10.07+320.5}, \frac{320.5}{10.07+320.5}] \approx [0.03, 0.97]$
*   **第二行:**
    *   $\exp(3.46) \approx 31.8$, $\exp(9.24) \approx 10300$
    *   $softmax([3.46, 9.24]) = [\frac{31.8}{31.8+10300}, \frac{10300}{31.8+10300}] \approx [0.003, 0.997]$

$$
Attention\_Weights = \begin{bmatrix} 0.03 & 0.97 \\ 0.003 & 0.997 \end{bmatrix}
$$
**解读:** 对于 "thinking" 这个词，它 97% 的注意力放在了 "machines" 上。对于 "machines"，它 99.7% 的注意力也放在了 "machines" 上（包括自己）。

**5. 加权求和得到输出**
$$
Output = Attention\_Weights \times V = \begin{bmatrix} 0.03 & 0.97 \\ 0.003 & 0.997 \end{bmatrix} \begin{bmatrix} 1 & 3 & 0 \\ 2 & 0 & 8 \end{bmatrix}
$$
$$
Output = \begin{bmatrix} (0.03*1+0.97*2) & (0.03*3+0.97*0) & (0.03*0+0.97*8) \\ (0.003*1+0.997*2) & (0.003*3+0.997*0) & (0.003*0+0.997*8) \end{bmatrix}
$$
$$
Output = \begin{bmatrix} 1.97 & 0.09 & 7.76 \\ 1.997 & 0.009 & 7.976 \end{bmatrix}
$$
这就是自注意力层最终为 "thinking" 和 "machines" 两个词生成的新表示。

## 多头注意力 (Multi-Head Attention)

在标准的自注意力机制中，我们计算一组查询（Q）、键（K）和值（V）向量，并通过计算 Q 和 K 之间的点积来获得注意力权重，然后用这些权重来加权 V。虽然这能让模型关注序列中的相关部分，但它可能只关注到一种类型的相关性或信息子空间。

在Transformer中，进入多头自注意力模块的向量维度，就是最开始通过 patch_embed (卷积嵌入)、添加 cls_token、再叠加上 pos_embed (位置编码) 之后得到的那个维度。这个维度在整个 Transformer 编码器堆栈中通常是**保持不变**的。贯穿 Transformer 编码器/解码器堆栈的嵌入维度 (embed_dim) 必须保持一致。

**核心思想**:
多头注意力的核心思想是：**与其只进行一次注意力计算，不如并行地进行多次独立的注意力计算（称为“头”），每个头关注输入信息的不同表示子空间（representation subspace），然后将这些头的结果合并起来。**

这就像让多个专家（头）从不同的角度同时分析同一个问题（输入序列），每个专家关注不同的特征或关系，最后综合所有专家的意见得到最终结论。

**工作流程**:

1.  **输入**: 与单头自注意力一样，输入是序列的嵌入表示 $X$（通常已经加上了位置编码），维度为 $n \times d_{model}$，其中 $n$ 是序列长度，$d_{model}$ 是嵌入维度。
2.  **线性投影**:
    *   首先，为每个头 $i$（假设总共有 $h$ 个头）定义**独立的**、可学习的权重矩阵：$W_Q^i, W_K^i, W_V^i$。
    *   将输入 $X$ 分别乘以这些权重矩阵，为每个头 $i$ 生成对应的 $Q_i, K_i, V_i$：
        $$
        Q_i = X \times W_Q^i
        $$
        $$
        K_i = X \times W_K^i
        $$
        $$
        V_i = X \times W_V^i
        $$
    *   **维度划分**: 通常，每个头的 $Q, K, V$ 向量的维度会被设定为 $d_k = d_v = d_{model} / h$。这意味着每个头处理的是原始 $d_{model}$ 维度的一个子空间。例如，如果 $d_{model}=512$ 且 $h=8$，则每个头的 $Q_i, K_i, V_i$ 维度将是 $n \times 64$ ($512/8 = 64$)。
3.  **并行计算注意力**:
    *   对每个头 $i$，独立地执行标准的缩放点积注意力计算：
        $$
        head_i = Attention(Q_i, K_i, V_i) = softmax\left( \frac{Q_i K_i^T}{\sqrt{d_k}} \right) V_i
        $$
    *   每个 $head_i$ 的输出维度是 $n \times d_v$。由于 $d_v = d_{model} / h$，这里是 $n \times (d_{model} / h)$。
4.  **拼接 (Concatenation)**:
    *   **拼接操作就是将这 $h$ 个 $n \times d_v$ 的矩阵沿着它们的最后一个维度（即特征维度 $d_v$）组合起来，形成一个更大的矩阵。**
    *   将所有 $h$ 个头的输出 $head_1, head_2, ..., head_h$ 在最后一个维度（特征维度）上拼接起来：
        $$
        Concat = Concat(head_1, head_2, ..., head_h)
        $$
    *   拼接后的矩阵维度是 $n \times (h \times d_v)$。因为 $h \times d_v = d_{model}$，所以拼接后的维度恢复到 $n \times d_{model}$。

5.  **最终线性投影**:
    *   将拼接后的结果乘以**另一个**可学习的权重矩阵 $W_O$（维度 $d_{model} \times d_{model}$），进行最终的线性变换，得到多头注意力的最终输出：
        $$
        MultiHead(Q, K, V) = Concat \times W_O
        $$
    *   输出维度仍然是 $n \times d_{model}$，与输入 $X$ 的维度一致，方便在 Transformer 模型中堆叠多层。

**总结公式**:
$$
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W_O
$$
$$
\text{where} \quad head_i = Attention(XW_Q^i, XW_K^i, XW_V^i)
$$

**优势**:

1.  **扩展模型关注不同位置信息的能力**: 单个注意力头可能只关注到一种模式或关系。多头允许模型同时关注来自不同表示子空间的信息。例如，一个头可能关注语法依赖，另一个头可能关注语义相似性，还有一个头可能关注位置邻近性。
2.  **提高表示能力**: 通过将 $d_{model}$ 维空间划分为 $h$ 个子空间，并在每个子空间独立计算注意力，模型可以捕捉到更丰富、更多样的特征组合。
3.  **稳定训练**: 有研究表明，多头机制有时能让注意力机制的训练过程更稳定。

### 为何不同的头能学习不同模式？

虽然所有头接收相同的初始输入 $x$，但它们之所以能学习到不同的注意力模式，关键在于以下几点：

1.  **独立的、随机初始化的权重矩阵**:
    *   这是**最重要**的原因。如上所述，每个头 $i$ 都有**自己独立**的一组线性投影权重矩阵 $W_i^Q, W_i^K, W_i^V$ (以及最终的 $W^O$)。
    *   这些权重矩阵在训练开始时是**随机初始化**的，并且**彼此不同**。
    *   因此，即使输入 $x$ 相同，由于初始权重不同，每个头计算出的 $Q_i, K_i, V_i$ 就是**不同**的。

2.  **不同的梯度路径**:
    *   因为每个头的初始 $Q_i, K_i, V_i$ 不同，它们计算出的注意力得分和加权后的 $Attention_i$ 也不同。
    *   在反向传播计算梯度时，损失函数对每个头内部的独特权重矩阵 ($W_i^Q, W_i^K, W_i^V$) 的梯度也会是**不同**的。
    *   优化器（如 Adam）会根据这些不同的梯度来更新每个头的权重。

3.  **不同的学习“方向”**:
    *   由于初始状态不同，梯度不同，每个头在训练过程中会探索不同的参数空间区域。
    *   它们会各自找到能够降低整体损失的不同方式，从而学习到关注输入序列中不同方面的信息或不同类型的依赖关系。

4.  **Dropout 等正则化技术 (若使用)**:
    *   正则化技术（如 Dropout）会进一步增加随机性，使得不同的头更难学习到完全一样的模式。

**核心机制:** 正是**为每个头使用独立且随机初始化的权重矩阵**打破了对称性，使得即使输入相同，不同的头也能在训练过程中分化，学习捕捉输入数据中不同子空间的特征和关系，使得模型能够同时从多个角度理解输入，获得更强大的表示能力。

#### **强制子空间学习 (理论动机)**:

将原始的高维空间投影到多个**不同的低维子空间**，被认为是**强制**每个头去关注输入表示的不同方面或子特征。每个头只能看到原始信息的一个“低维快照”，因此它们更有可能学习到互补的信息。

#### **计算成本**:

与单头注意力相比，如果保持总计算量相似（通过设置 $d_k = d_{model} / h$），多头注意力的计算成本并不会显著增加，因为大部分计算可以并行执行。主要的额外开销在于参数数量的增加（需要 $h$ 组 $W_Q, W_K, W_V$ 和一个 $W_O$）。

总之，多头注意力是 Transformer 成功的关键因素之一，它通过并行地在不同的表示子空间中计算注意力，显著增强了模型捕捉复杂依赖关系的能力。

### 多头注意力计算实例

我们继续使用上一个例子的输入，但这次使用2个注意力头 ($h=2$)。

**场景设定:**
*   输入矩阵 $X$ (维度 $2 \times 4$):
    $$
    X = \begin{bmatrix} 1 & 0 & 1 & 0 \\ 0 & 2 & 0 & 2 \end{bmatrix}
    $$
*   模型总维度 $d_{model} = 4$, 头数 $h=2$。
*   每个头的维度 $d_k = d_v = d_{model} / h = 4 / 2 = 2$。

现在我们需要为每个头定义独立的权重矩阵 (维度 $d_{model} \times d_k = 4 \times 2$)。

**Head 1 的权重矩阵:**
$$
W_Q^1 = \begin{bmatrix} 1 & 0 \\ 1 & 1 \\ 0 & 1 \\ 1 & 0 \end{bmatrix} \quad W_K^1 = \begin{bmatrix} 0 & 1 \\ 1 & 1 \\ 1 & 0 \\ 0 & 1 \end{bmatrix} \quad W_V^1 = \begin{bmatrix} 0 & 2 \\ 1 & 0 \\ 1 & 1 \\ 0 & 0 \end{bmatrix}
$$

**Head 2 的权重矩阵:**
$$
W_Q^2 = \begin{bmatrix} 0 & 1 \\ 1 & 0 \\ 1 & 1 \\ 0 & 0 \end{bmatrix} \quad W_K^2 = \begin{bmatrix} 1 & 1 \\ 0 & 1 \\ 0 & 0 \\ 1 & 0 \end{bmatrix} \quad W_V^2 = \begin{bmatrix} 1 & 1 \\ 0 & 3 \\ 1 & 0 \\ 2 & 1 \end{bmatrix}
$$

**1. 为每个头计算 Q, K, V**

*   **Head 1:**
    *   $Q_1 = X W_Q^1 = \begin{bmatrix} 1 & 1 \\ 4 & 2 \end{bmatrix}$
    *   $K_1 = X W_K^1 = \begin{bmatrix} 1 & 1 \\ 2 & 4 \end{bmatrix}$
    *   $V_1 = X W_V^1 = \begin{bmatrix} 1 & 3 \\ 2 & 0 \end{bmatrix}$
*   **Head 2:**
    *   $Q_2 = X W_Q^2 = \begin{bmatrix} 1 & 2 \\ 2 & 0 \end{bmatrix}$
    *   $K_2 = X W_K^2 = \begin{bmatrix} 1 & 1 \\ 2 & 2 \end{bmatrix}$
    *   $V_2 = X W_V^2 = \begin{bmatrix} 2 & 1 \\ 4 & 8 \end{bmatrix}$

**2. 并行计算每个头的注意力输出**

*   **计算 Head 1 输出 ($head_1$):**
    *   $Scores_1 = Q_1 K_1^T = \begin{bmatrix} 1 & 1 \\ 4 & 2 \end{bmatrix} \begin{bmatrix} 1 & 2 \\ 1 & 4 \end{bmatrix} = \begin{bmatrix} 2 & 6 \\ 6 & 16 \end{bmatrix}$
    *   $Scaled\_Scores_1 = Scores_1 / \sqrt{d_k=2} = \begin{bmatrix} 1.41 & 4.24 \\ 4.24 & 11.31 \end{bmatrix}$
    *   $Weights_1 = softmax(Scaled\_Scores_1) \approx \begin{bmatrix} 0.05 & 0.95 \\ 0.001 & 0.999 \end{bmatrix}$
    *   $head_1 = Weights_1 \times V_1 = \begin{bmatrix} 0.05 & 0.95 \\ 0.001 & 0.999 \end{bmatrix} \begin{bmatrix} 1 & 3 \\ 2 & 0 \end{bmatrix} = \begin{bmatrix} 1.95 & 0.15 \\ 1.999 & 0.003 \end{bmatrix}$

*   **计算 Head 2 输出 ($head_2$):**
    *   $Scores_2 = Q_2 K_2^T = \begin{bmatrix} 1 & 2 \\ 2 & 0 \end{bmatrix} \begin{bmatrix} 1 & 2 \\ 1 & 2 \end{bmatrix} = \begin{bmatrix} 3 & 6 \\ 2 & 4 \end{bmatrix}$
    *   $Scaled\_Scores_2 = Scores_2 / \sqrt{d_k=2} = \begin{bmatrix} 2.12 & 4.24 \\ 1.41 & 2.82 \end{bmatrix}$
    *   $Weights_2 = softmax(Scaled\_Scores_2) \approx \begin{bmatrix} 0.11 & 0.89 \\ 0.19 & 0.81 \end{bmatrix}$
    *   $head_2 = Weights_2 \times V_2 = \begin{bmatrix} 0.11 & 0.89 \\ 0.19 & 0.81 \end{bmatrix} \begin{bmatrix} 2 & 1 \\ 4 & 8 \end{bmatrix} = \begin{bmatrix} 3.78 & 7.23 \\ 3.62 & 6.67 \end{bmatrix}$

**3. 拼接 (Concatenation)**
将 $head_1$ 和 $head_2$ (维度都是 $2 \times 2$) 沿特征维度拼接起来。
$$
Concat = \begin{bmatrix} 1.95 & 0.15 & 3.78 & 7.23 \\ 1.999 & 0.003 & 3.62 & 6.67 \end{bmatrix}
$$
拼接后矩阵维度恢复到 $n \times d_{model} = 2 \times 4$。

**4. 最终线性投影**
将拼接后的矩阵乘以最终的输出权重矩阵 $W_O$ (维度 $d_{model} \times d_{model} = 4 \times 4$)。
假设 $W_O$ (随机初始化):
$$
W_O = \begin{bmatrix} 1 & 0 & 1 & 0 \\ 0 & 2 & 0 & 1 \\ 1 & 0 & 0 & 1 \\ 0 & 1 & 1 & 0 \end{bmatrix}
$$
$$
Output = Concat \times W_O = \begin{bmatrix} 1.95 & 0.15 & 3.78 & 7.23 \\ 1.999 & 0.003 & 3.62 & 6.67 \end{bmatrix} \begin{bmatrix} 1 & 0 & 1 & 0 \\ 0 & 2 & 0 & 1 \\ 1 & 0 & 0 & 1 \\ 0 & 1 & 1 & 0 \end{bmatrix}
$$
$$
Output \approx \begin{bmatrix} 5.73 & 7.53 & 9.18 & 3.93 \\ 5.62 & 6.676 & 8.619 & 3.626 \end{bmatrix}
$$
这个 $2 \times 4$ 的矩阵就是多头注意力层最终的输出，它融合了来自两个不同注意力头的信息。

------

## 位置编码 (Positional Encoding)

正如我们之前讨论的，Transformer 模型的核心机制——自注意力（Self-Attention）——在处理输入序列时是“无序”的。它并行地计算序列中所有元素对之间的关系，但这种计算本身并不包含元素在序列中的位置信息。例如，对于句子“狗 追 猫”和“猫 追 狗”，自注意力机制如果不结合位置信息，可能会产生非常相似甚至相同的表示，因为它只关注词与词之间的关系，而忽略了它们的排列顺序。

为了解决这个问题，Transformer 引入了位置编码，其核心目标是为模型注入关于元素在序列中位置的信息。

### 为什么需要位置编码？

*   **弥补结构缺陷**: Transformer 放弃了 RNN 的循环结构和 CNN 的局部卷积窗口，这两种结构天然地包含了序列顺序或局部位置信息。自注意力机制的并行计算特性使其失去了这种内置的顺序感。
*   **区分相同词语**: 同一个词在句子的不同位置可能扮演不同的语法角色或具有不同的重要性。位置编码使得模型能够区分这种情况。例如，“苹果 公司 发布了 新 苹果 手机”中的两个“苹果”。
*   **捕捉语序依赖**: 语言理解严重依赖于词语的顺序。主语、谓语、宾语的位置关系决定了句子的基本含义。位置编码是让模型理解这种结构关系的基础。

### 正弦/余弦位置编码 (Sinusoidal Positional Encoding)

这是原始 Transformer 论文中提出的经典方法。

**公式**:
$$
PE(pos, 2i) = sin(pos / 10000^{2i / d_{model}})
$$
$$
PE(pos, 2i+1) = cos(pos / 10000^{2i / d_{model}})
$$

**关键特性与优势**:

1.  **唯一性**: 对于每个位置 $pos$，这个公式都能生成一个独特的 $d_{model}$ 维向量。
2.  **确定性**: 它是固定的、非学习的。这意味着它不需要通过数据训练得到，节省了参数，并且对于任何给定长度的序列，其编码是预先确定的。
3.  **相对位置信息**: 这是该方法最巧妙的设计之一。可以证明，对于任何固定的偏移量 $k$，$PE(pos+k)$ 可以表示为 $PE(pos)$ 的线性变换。这意味着模型可以通过学习一个线性变换来关注相对位置，例如，“当前词后面第 3 个词”。具体来说，存在一个与 $pos$ 无关的变换矩阵 $M_k$，使得 $PE(pos+k) \approx M_k \times PE(pos)$。这种线性关系使得模型更容易学习到位置间的相对关系。
4.  **平滑性**: 相邻位置的位置编码是相似的，随着距离增加，编码差异变大，这符合直觉。
5.  **外推能力**: 理论上，由于正弦和余弦函数的周期性，这种编码方式可以推广到比训练时遇到的序列更长的序列，尽管在实践中，对于非常长的序列，其效果可能会下降。

**工作原理的直观理解**:
可以想象，位置编码向量的每个维度都在以不同的频率振荡（由 $10000^{2i / d_{model}}$ 控制）。低维度的 $i$ 对应低频（长波长），高维度的 $i$ 对应高频（短波长）。一个位置的编码就是这些不同频率波在该位置上的值的组合。由于频率组合的多样性，每个位置都能得到一个独特的“振荡模式”签名。而相对位置关系则体现在这些波的相位差上。

关于这一部分的详细解释可以看：[详解自注意力机制中的位置编码（第一部分） - 知乎](https://zhuanlan.zhihu.com/p/352233973)

### 傅里叶位置编码 (Fourier Positional Encoding)

在深入探讨其他类型的位置编码之前，非常有必要从一个更深刻的视角来理解经典的正弦/余弦编码——**傅里叶分析**的视角。实际上，正弦/余弦位置编码**就是傅里叶特征的一种形式**，而“傅里叶位置编码”这个术语通常指代将这种思想进行泛化和应用的编码方式，尤其是在处理高维数据（如图像）时。

#### **1. 核心思想：将位置看作频率域的信号**

傅里叶分析告诉我们，任何复杂的信号都可以被分解为一系列不同频率和振幅的正弦波和余弦波的叠加。傅里ye位置编码正是借鉴了这一思想，它不把位置看作一个离散的整数（如第1、2、3个位置），而是将其视为一个连续信号，并用其在频率域的特征来表示。

原始的正弦/余弦编码公式：
$$
PE(pos, 2i) = sin(\omega_i \cdot pos)
$$

$$
PE(pos, 2i+1) = cos(\omega_i \cdot pos)
$$

其中，角频率 $\omega_i = 1 / 10000^{2i / d_{model}}$。

这个公式的本质是：
**用一组预设好的、不同频率($\omega_i$)的三角函数基，去对位置 $pos$ 这个值进行“采样”，从而得到一个能够唯一标识该位置的向量。**

这个编码向量的每一个维度，都代表了位置 $pos$ 在某一个特定频率上的分量信息。低频分量（$i$ 较小，$\omega_i$ 较小）捕捉位置的宏观、粗略信息；高频分量（$i$ 较大，$\omega_i$ 较大）则捕捉位置的微观、精细信息。

#### **2. 傅里叶特征在图像中的应用 (2D Positional Encoding)**

当我们将这个思想从一维序列扩展到二维图像时，傅里叶位置编码的优势就更加明显了。对于一个尺寸为 $H \times W$ 的图像，一个像素点的位置由一个二维坐标 $(x, y)$ 决定，其中 $0 \le x < W, 0 \le y < H$。

我们可以为 $x$ 轴和 $y$ 轴分别计算傅里叶特征，然后将它们拼接起来，形成二维的位置编码。

假设我们为每个坐标轴分配 $d_{model}/2$ 的维度。

1. **计算 $x$ 坐标的编码 $PE_x$**:
   $$
   PE(x, 2i) = sin(x / 10000^{2i / (d_{model}/2)})
   $$

   $$
   PE(x, 2i+1) = cos(x / 10000^{2i / (d_{model}/2)})
   $$

   这将得到一个 $d_{model}/2$ 维的向量。

2. **计算 $y$ 坐标的编码 $PE_y$**:
   $$
   PE(y, 2j) = sin(y / 10000^{2j / (d_{model}/2)})
   $$

   $$
   PE(y, 2j+1) = cos(y / 10000^{2j / (d_{model}/2)})
   $$

   这将得到另一个 $d_{model}/2$ 维的向量。

3. **合并编码**:
   最简单的方式是将两个向量拼接（concatenate）起来：
   $$
   PE(x, y) = \text{concat}(PE_x, PE_y)
   $$
   这样，对于图像中的每一个像素点 $(x, y)$，我们都得到了一个独特的 $d_{model}$ 维位置编码向量。这个向量同时包含了该点在水平和垂直方向上的位置信息。

这种方法在很多视觉 Transformer (ViT) 的变体中被使用，因为它能够自然地将一维序列的成功经验扩展到二维空间，并且保持了正弦/余弦编码的所有优良特性（如相对位置关系、外推性等）。

#### **3. 优势总结**

*   **维度扩展性强**：傅里叶特征的思想可以轻松地从一维推广到二维（图像）、三维（视频、医疗影像）甚至更高维度的空间，只需为每个维度独立计算傅里叶特征然后拼接即可。
*   **分辨率无关性**：由于位置被视为连续坐标进行编码，理论上这种方法对于不同分辨率的输入具有更好的泛化能力。例如，一个在 $256 \times 256$ 图像上训练的模型，其位置编码机制可以平滑地应用于 $512 \times 512$ 的图像。
*   **概念统一**：将经典的位置编码理解为傅里叶特征，为我们提供了一个更强大、更通用的理论框架，有助于启发更多新型的位置表示方法。例如，我们可以不使用固定的频率，而是让模型去学习最优的频率基（即学习 $\omega_i$），或者引入随机的傅里叶特征来增加模型的鲁棒性。

### 其他类型的位置编码

虽然正弦/余弦编码很常用，但并非唯一选择。

1.  **学习的位置编码 (Learned Positional Encoding)**:
    *   **代表作**: BERT, GPT 系列
    *   另一种常见的方法是直接为每个可能的位置创建一个可学习的嵌入向量（类似于词嵌入）。
    *   模型在训练过程中学习这些位置嵌入。
    *   **优点**: 可能更灵活，能更好地适应特定任务的数据分布。BERT 和 GPT 系列模型通常采用这种方式。
    *   **缺点**:
        *   需要额外的参数。
        *   通常需要设定一个最大序列长度，对于超过该长度的序列处理能力受限（尽管也有一些技术试图缓解这个问题）。
        *   可能不如正弦/余弦编码那样具有良好的外推性。
2.  **相对位置编码 (Relative Positional Encoding)**:
    *   **代表作**: Transformer-XL, T5, DeBERTa
    *   这类方法不直接编码绝对位置，而是试图在自注意力计算中直接注入元素之间的相对距离信息。
    *   例如，在计算第 $i$ 个元素对第 $j$ 个元素的注意力分数时，会额外考虑它们之间的距离 $j-i$。这个相对距离本身会被编码成一个向量，并融入到注意力分数的计算中（例如，加到 Key 向量上，或者直接加到 $Q \times K^T$ 的结果上）。
    *   **优点**: 可能更直观地捕捉局部依赖关系，并且在某些任务上表现更好。Transformer-XL, T5, DeBERTa 等模型采用了不同形式的相对位置编码。
    *   **缺点**: 实现相对复杂一些。

3.  **旋转位置编码 (Rotary Positional Embedding - RoPE)**:
    *   **代表作**: Llama, PaLM
    *   由 LLaMA 等模型推广使用。它不是将位置信息加到嵌入上，而是在计算 Q 和 K 向量后，根据它们的位置对它们进行“旋转”操作。
    *   它巧妙地将相对位置信息融入了注意力计算中，同时保持了一定的外推能力。

### 如何结合

最常见的方式（用于绝对位置编码，如正弦/余弦或学习的编码）是将位置编码向量与对应的词嵌入向量**相加**：
$$
X_{final} = Word\_Embedding(X) + Positional\_Encoding(pos)
$$
这个结合了词义信息和位置信息的 $X_{final}$ 才是输入到 Transformer 后续层（如自注意力层、前馈网络层）的最终表示。

### 总结

位置编码是 Transformer 模型不可或缺的一部分，它为模型提供了处理序列顺序的关键能力。正弦/余弦编码是一种经典且有效的非学习方法，利用不同频率的周期函数为每个位置生成独特且具有良好相对位置属性的编码。同时，学习的位置编码和各种相对位置编码方案也提供了不同的选择，各有优劣，并在许多现代 Transformer 架构中得到了应用。选择哪种位置编码方式取决于具体的模型设计和任务需求。



