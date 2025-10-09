#  Transformer笔记（DeepSeek生成）

学习资料：[Transformer通俗笔记：从Word2Vec、Seq2Seq逐步理解到GPT、BERT-CSDN博客](https://blog.csdn.net/v_JULY_v/article/details/127411638?ops_request_misc=%7B%22request%5Fid%22%3A%229d7c8f6c3ec83074f33e1e1ebe062d64%22%2C%22scm%22%3A%2220140713.130102334.pc%5Fall.%22%7D&request_id=9d7c8f6c3ec83074f33e1e1ebe062d64&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~first_rank_ecpm_v1~hot_rank-1-127411638-null-null.142^v102^pc_search_result_base3&utm_term=seq2seq模型&spm=1018.2226.3001.4187)

[10.7. Transformer — 动手学深度学习 2.0.0 documentation](https://zh-v2.d2l.ai/chapter_attention-mechanisms/transformer.html)

[【超详细】【原理篇&实战篇】一文读懂Transformer-CSDN博客](https://blog.csdn.net/weixin_42475060/article/details/121101749?ops_request_misc=%7B%22request%5Fid%22%3A%22272188ca87198310ab1b006e1bb0d1ce%22%2C%22scm%22%3A%2220140713.130102334..%22%7D&request_id=272188ca87198310ab1b006e1bb0d1ce&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~top_positive~default-1-121101749-null-null.142^v102^pc_search_result_base3&utm_term=transformer&spm=1018.2226.3001.4187)

其他笔记： [多头注意力、自注意力与位置编码笔记](../../Natural_Language_Processing_Theory(自然语言处理理论)/multi_head_attention_self_attention_and_positional_encoding_notes(多头注意力、自注意力与位置编码笔记).md)

------

### 1. 整体架构 (Overall Architecture)

Transformer 模型遵循经典的 **Encoder-Decoder** 架构。

*   **Encoder (编码器)**：负责接收输入序列（例如，源语言句子），并将其转换成一系列连续的表示（Contextual Embeddings）。它由 N 个相同的层堆叠而成（论文中 N=6）。
*   **Decoder (解码器)**：接收编码器的输出以及目标序列（在训练时是目标语言句子，在推理时是已生成的部分），并生成下一个词的概率分布。它也由 N 个相同的层堆叠而成（论文中 N=6）。

![1](transformer_notes.assets/1.png)

### 2. 输入处理 (Input Processing)

#### a. 输入嵌入 (Input Embedding)

与大多数 NLP 模型类似，Transformer 首先将输入序列中的每个词（Token）转换成固定维度的向量。这通常通过一个可学习的嵌入矩阵（Embedding Matrix）实现。假设词汇表大小为 $V$，嵌入维度为$ d_{model}$，那么嵌入矩阵就是一个 $V \times d_{model}$ 的矩阵。

#### b. 位置编码 (Positional Encoding)

由于 Transformer 没有 RNN 的循环结构或 CNN 的卷积操作，它本身无法感知序列中词语的位置信息。为了解决这个问题，模型引入了 **位置编码 (Positional Encoding)**。这些编码向量被加到对应的词嵌入向量上。

Transformer 使用 **正弦和余弦函数** 来生成位置编码，其公式如下：

对于位置 pos 和维度 i（其中 i 从 0 到 d_model-1）：

*   **偶数维度 (2i)**: $PE(pos, 2i) = \sin(pos / 10000^{2i / d_{model}})$
*   **奇数维度 (2i+1)**: $PE(pos, 2i+1) = \cos(pos / 10000^{2i / d_{model}})$

其中：
*   pos 是词在序列中的位置（从 0 开始）。
*   i 是嵌入向量中的维度索引。
*   $d_{model}$ 是嵌入向量的维度（论文中为 512）。

这种设计有几个优点：
1.  每个位置都有独特的编码。
2.  能够表示相对位置信息，因为对于固定的偏移 k，PE(pos+k) 可以表示为 PE(pos) 的线性函数。
3.  可以扩展到比训练时遇到的序列更长的序列。

**最终输入 = 词嵌入 (Word Embedding) + 位置编码 (Positional Encoding)**

详情请看： [多头注意力、自注意力与位置编码笔记](../Basic Concepts-NLP/多头注意力、自注意力与位置编码笔记.md)

### 3. 编码器 (Encoder)

每个编码器层包含两个主要的子层：

#### a. 多头自注意力机制 (Multi-Head Self-Attention)

这是 Transformer 的核心创新之一。**自注意力 (Self-Attention)** 允许模型在处理一个词时，关注输入序列中的所有其他词，并根据相关性计算该词的表示。

**i. 缩放点积注意力 (Scaled Dot-Product Attention)**

这是自注意力的基础。对于输入序列中的每个词，我们创建三个向量：查询 (Query, Q)、键 (Key, K) 和值 (Value, V)。这些向量是通过将词的嵌入（加上位置编码）乘以三个不同的可学习权重矩阵 ($W^Q, W^K, W^V$) 得到的。

*   $Q = X W^Q$
*   $K = X W^K$
*   $V = X W^V$
    (其中 X 是输入嵌入矩阵)

注意力得分是通过计算查询向量 Q 与所有键向量 K 的点积得到的。为了防止点积结果过大导致梯度消失，需要将其除以一个 **缩放因子** $\sqrt{d_k}$，其中 $d_k$ 是键向量的维度（通常是 $d_{model} / h$，h 是头的数量）。然后，通过 **Softmax** 函数将得分转换为概率（权重），表示每个词对当前词的重要性。最后，将这些权重乘以对应的值向量 V 并求和，得到该词的注意力输出。

**公式:**
$Attention(Q, K, V) = \text{softmax}\left( \frac{Q K^T}{\sqrt{d_k}} \right) V$

**ii. 多头注意力 (Multi-Head Attention)**

为了让模型能够关注来自不同表示子空间的信息，Transformer 使用了 **多头注意力**。它不是只计算一次注意力，而是将 Q, K, V 通过不同的、可学习的线性投影（权重矩阵 $W_i^Q, W_i^K, W_i^V$）投影 h 次（h 是头的数量，论文中 h=8）。对每个投影后的 Q, K, V 并行地执行缩放点积注意力计算，得到 h 个输出。

$head_i = Attention(Q W_i^Q, K W_i^K, V W_i^V)$

然后，将这 h 个输出拼接 (Concatenate) 起来，并通过另一个可学习的线性投影（权重矩阵 $W^O$）得到最终的多头注意力输出。

**公式:**
$MultiHead(Q, K, V) = \text{Concat}(head_1, ..., head_h) W^O$

多头机制允许模型在不同位置共同关注来自不同表示子空间的信息。

#### b. 位置前馈网络 (Position-wise Feed-Forward Network, FFN)

这是编码器层的第二个子层。它是一个简单的、全连接的前馈网络，独立地应用于每个位置（即序列中的每个词）。它包含两个线性变换和一个 ReLU 激活函数。

**公式:**
$FFN(x) = \max(0, x W_1 + b_1) W_2 + b_2$

其中 x 是前一个子层（多头注意力）的输出，$W_1, b_1, W_2, b_2$ 是可学习的参数。这个网络的输入和输出维度都是 $d_{model}$，中间层的维度 $d_{ff}$ 通常更大（论文中为 2048）。

#### c. 残差连接与层归一化 (Add & Norm)

在每个子层（多头注意力和 FFN）的周围，都使用了 **残差连接 (Residual Connection)**，然后进行 **层归一化 (Layer Normalization)**。

*   **残差连接**: 将子层的输入 x 直接加到子层的输出 Sublayer(x) 上，即 $x + \text{Sublayer}(x)$。这有助于缓解深度网络中的梯度消失问题，使得训练更深的模型成为可能。
*   **层归一化**: 对每个样本（在这里是序列中的每个位置）的特征进行归一化，使其均值为 0，方差为 1，然后再进行缩放和平移。这有助于稳定训练过程，加速收敛。

**结构**: $LayerNorm(x + \text{Sublayer}(x))$

所以，一个完整的编码器层流程是：
1.  输入 x
2.  多头自注意力: $attn\_output = \text{MultiHeadSelfAttention}(x)$
3.  Add & Norm: $norm1\_output = LayerNorm(x + attn\_output)$
4.  前馈网络: $ffn\_output = FFN(norm1\_output)$
5.  Add & Norm: $output = LayerNorm(norm1\_output + ffn\_output)$

这个 output 就是该编码器层的最终输出，并作为下一层的输入。

### 4. 解码器 (Decoder) 

Transformer 的解码器负责根据编码器对输入序列的理解（上下文向量）以及已经生成的部分目标序列，来逐步构建完整的输出序列。它与编码器共享一些相似的组件（如多头注意力、前馈网络、残差连接和层归一化），但其独特的设计使其能够胜任序列生成任务。

解码器通常由 N 个相同的层堆叠而成。每一层都包含以下三个核心子层和一个最终的线性变换及 Softmax 层（在最后一个解码器层之后）。

#### a. 掩码多头自注意力 (Masked Multi-Head Self-Attention)

这是解码器内部处理其自身生成序列的部分，是解码器与编码器自注意力的关键区别所在。

*   **核心目的**: 实现 **自回归 (Autoregressive)** 特性。在生成目标序列的第 $t$ 个词时，模型必须仅基于之前已经生成的 $1$ 到 $t-1$ 个词以及当前正在预测的第 $t$ 个位置的信息。它绝不能接触到 $t+1$ 及之后位置的“未来”信息。想象一下，如果翻译模型在翻译第 3 个词时就能看到完整的标准答案，那它就不需要学习如何预测了。掩码机制就是为了防止这种“作弊”行为。

*   **工作机制**:
    1.  **输入**: 与编码器自注意力类似，该层的输入是来自 **前一个解码器层** 的输出（对于第一个解码器层，则是目标序列的词嵌入（Word Embedding）与位置编码（Positional Encoding）之和）。
    2.  **Q, K, V 计算**: 基于这个输入，计算出查询（Query, Q）、键（Key, K）、值（Value, V）向量。这与标准自注意力相同，允许序列中的每个位置关注序列中的其他位置（包括自身）。
    3.  **注意力得分计算**: 计算 Q 和 K 的点积，然后进行缩放（除以 $\sqrt{d_k}$）。 $Score = \frac{Q K^T}{\sqrt{d_k}}$
    4.  **应用掩码 (Masking)**: 这是关键步骤。在将注意力得分送入 Softmax 函数之前，会应用一个“后续掩码”（look-ahead mask）。这个掩码通常是一个上三角矩阵（不包括对角线），或者是一个在对应未来位置填充了 $-\infty$ 的矩阵。例如，对于第 $i$ 个查询向量 $Q_i$，它只能关注第 $j \le i$ 个键向量 $K_j$。所有 $j > i$ 的位置对应的得分会被设置为 $-\infty$。
        $MaskedScore = Score + \text{Mask}$ (其中 Mask 矩阵在 $j > i$ 的位置为 $-\infty$，其他位置为 0)
    5.  **Softmax**: 对掩码后的得分应用 Softmax 函数。由于 $e^{-\infty} \approx 0$，那些被掩码的位置的注意力权重将趋近于零。
        $AttentionWeights = \text{softmax}(MaskedScore)$
    6.  **加权求和**: 使用得到的注意力权重对值（Value, V）向量进行加权求和，得到该位置的自注意力输出。
        $Output = AttentionWeights \cdot V$
    7.  **多头机制**: 这个过程会并行地在多个“头”中进行，每个头学习不同的注意力模式。最后将所有头的输出拼接起来并通过一个线性层进行整合。

*   **效果**: 确保了每个位置的输出只依赖于当前及之前位置的输入，完美契合了序列生成任务从左到右（或按时间顺序）生成的自然流程。

#### b. 编码器-解码器注意力 (Encoder-Decoder Attention / Cross-Attention)

这个子层是解码器获取输入序列信息的核心环节，是连接编码器和解码器的桥梁。

*   **核心目的**: 让解码器在生成目标序列的特定位置时，能够**聚焦于输入序列中最相关的部分**。例如，在将英文 "Hello world" 翻译成中文“你好 世界”时，当解码器准备生成“你好”时，它需要特别关注编码器输出中对应 "Hello" 的表示；生成“世界”时，则需要关注对应 "world" 的表示。

*   **工作机制**:
    1.  **输入来源**: 这个注意力机制的输入比较特殊：
        *   **查询 (Query, Q)**: 来自解码器 **前一个子层**（即掩码自注意力层 + Add & Norm 之后）的输出。这代表了解码器当前时刻的状态和需求。
        *   **键 (Key, K) 和 值 (Value, V)**: **均来自编码器栈顶的最终输出**。这包含了整个输入序列经过编码器深度处理后的丰富上下文信息。
    2.  **计算过程**: 使用解码器的状态 Q 去查询编码器的输出 K，计算相关性得分。然后用这些得分对编码器的输出 V 进行加权求和。这个过程与标准的多头注意力类似，只是 Q、K、V 的来源不同。
        $Score = \frac{Q_{decoder} K_{encoder}^T}{\sqrt{d_k}}$
        $AttentionWeights = \text{softmax}(Score)$
        $Output = AttentionWeights \cdot V_{encoder}$
    3.  **多头机制**: 同样采用多头机制并行计算，捕捉不同方面的对应关系。

*   **效果**: 使得解码器能够动态地、有选择地利用输入序列的信息来指导输出序列的生成，极大地提高了生成质量，特别是在处理长距离依赖和对齐问题时（如机器翻译、文本摘要）。**注意：这一层不需要掩码**，因为解码器查询任何输入序列的位置都是合理的，它需要看到完整的输入上下文。

#### c. 位置前馈网络 (Position-wise Feed-Forward Network, FFN)

紧跟在编码器-解码器注意力子层之后。

*   **核心目的**: 对每个位置的表示进行进一步的非线性变换，增加模型的拟合能力和复杂度。它可以被看作是对注意力层整合后的信息进行更深层次的加工和提炼。
*   **工作机制**: 这个 FFN 与编码器中的 FFN **结构完全相同**。它包含两个线性变换（通常第一个将维度从 $d_{model}$ 扩展到 $d_{ff}$，第二个再缩减回 $d_{model}$）和一个非线性激活函数（通常是 ReLU）。
    $FFN(x) = \text{ReLU}(x W_1 + b_1) W_2 + b_2$
    或者写作：
    $FFN(x) = \max(0, x W_1 + b_1) W_2 + b_2$
    这个网络独立地应用于序列中的每一个位置，但所有位置共享相同的权重 $W_1, b_1, W_2, b_2$。

#### d. 残差连接与层归一化 (Add & Norm)

这是 Transformer 结构中的“粘合剂”，确保信息流畅传递和训练稳定。

*   **应用位置**: 在解码器的 **每一个** 子层（掩码自注意力、编码器-解码器注意力、FFN）的输出端，都会应用残差连接，然后进行层归一化。
*   **残差连接 (Add)**: 将子层的输入 $x$ 直接加到子层的输出 $Sublayer(x)$ 上，即 $x + Sublayer(x)$。
    *   **目的**: 允许梯度直接流过网络层，有效缓解深度网络训练中的梯度消失问题。同时，它使得模型更容易学习恒等映射，即如果某个子层不是必需的，模型可以通过学习让 $Sublayer(x)$ 接近于零，从而保留原始输入 $x$。这有助于构建非常深的网络。
*   **层归一化 (Norm)**: 对残差连接后的结果进行归一化，即 $LayerNorm(x + Sublayer(x))$。
    *   **目的**: 稳定每一层输入的分布，减少内部协变量偏移，使得模型对参数初始化和学习率的选择不那么敏感，从而加速训练收敛并提高模型性能。它在每个样本内部、沿着特征维度进行归一化。

#### 完整解码器层流程与输出

一个完整的解码器层接收两个主要输入：① 来自前一解码器层的输出 $y_{prev}$（或目标嵌入+位置编码），② 编码器栈的最终输出 $encoder\_output$。其内部处理流程如下：

1.  $y_1 = \text{MaskedMultiHeadSelfAttention}(y_{prev}, y_{prev}, y_{prev})$
2.  $y_2 = LayerNorm(y_{prev} + y_1)$  (Add & Norm 1)
3.  $y_3 = EncoderDecoderAttention(y_2, encoder\_output, encoder\_output)$
4.  $y_4 = LayerNorm(y_2 + y_3)$  (Add & Norm 2)
5.  $y_5 = FFN(y_4)$
6.  $y_{output} = LayerNorm(y_4 + y_5)$ (Add & Norm 3)

这个 $y_{output}$ 就是当前解码器层的输出，它将作为下一个解码器层的输入 $y_{prev}$。

#### 最终输出层

当数据通过所有 N 个解码器层后，最后一个解码器层的输出会经过一个最终的线性变换层（Linear Layer）和一个 Softmax 层。

*   **线性层**: 将解码器输出的维度 ($d_{model}$) 映射到目标词汇表的大小 (Vocabulary Size)。
*   **Softmax 层**: 将线性层的输出（称为 logits）转换为概率分布，表示词汇表中每个词作为下一个输出词的概率。

在**训练**时，解码器接收完整的带掩码的目标序列，并试图预测每个位置的下一个词。在**推理（生成）**时，解码器通常以一个特殊的起始符 `<SOS>` (Start Of Sentence) 作为初始输入，预测第一个词；然后将预测出的词（或概率最高的词）作为下一个时间步的输入，再次预测第二个词，如此循环，直到预测出结束符 `<EOS>` (End Of Sentence) 或达到最大长度限制。

### 5. 最终输出层 (Final Output Layer)

解码器栈的最终输出是一系列向量。为了将其转换为每个词的概率，需要经过最后两个步骤：

1.  **线性层 (Linear Layer)**: 一个简单的全连接层，将解码器输出的向量投影到词汇表的大小。输出维度为 V (词汇表大小)。
2.  **Softmax 层**: 将线性层的输出转换为概率分布，表示词汇表中每个词是下一个词的概率。

### 6. 训练 (Training)

*   **损失函数**: 通常使用交叉熵损失 (Cross-Entropy Loss) 来比较模型预测的概率分布和真实的目标词（One-hot 编码）。
*   **优化器**: 论文中使用了 Adam 优化器，并配合特定的学习率调度策略（先线性增加，然后按平方根倒数衰减）。
*   **正则化**: 使用了 Dropout 和 Label Smoothing。

### 总结

Transformer 的核心优势在于：

1.  **并行计算**: 与 RNN 不同，Transformer 中的计算（尤其是自注意力）可以在序列维度上高度并行化，大大加快了训练速度。
2.  **长距离依赖**: 自注意力机制直接计算序列中任意两个位置之间的依赖关系，路径长度为 O(1)，有效解决了 RNN 中的长距离依赖问题。
3.  **模型性能**: 在机器翻译等多种 NLP 任务上取得了当时的最佳效果。

其关键组件包括：
*   输入嵌入 + 位置编码
*   多头自注意力机制（编码器和解码器）
*   掩码多头自注意力机制（解码器）
*   编码器-解码器注意力机制（解码器）
*   位置前馈网络
*   残差连接和层归一化
