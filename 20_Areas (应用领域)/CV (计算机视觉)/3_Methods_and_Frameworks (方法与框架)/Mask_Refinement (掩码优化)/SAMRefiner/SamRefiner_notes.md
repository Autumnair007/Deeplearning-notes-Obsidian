---
type: concept-note
tags:
  - cv
  - image-segmentation
  - mask-refinement
  - sam
  - full-supervision
  - semi-supervision
  - weakly-supervised
  - instance-segmentation
  - semantic-segmentation
  - tfs
status: in-progress
model: SAMRefiner
year: 2025
---
## 1. 论文概述与核心思想

SAMRefiner 旨在解决现有粗糙分割掩码（coarse masks）的质量问题，将其提升为高质量的分割结果，从而降低分割模型训练的标注成本。与以往针对特定模型或任务的细化方法不同，SAMRefiner 提出了一种通用且高效的方法，通过调整 SAM (Segment Anything Model) 来执行掩码细化任务。

核心思想：**噪声容忍的提示方案 (noise-tolerant prompting scheme)**。论文认为，直接将粗糙掩码作为 SAM 的输入提示会导致性能下降，因为粗糙掩码中的缺陷（如假阴性、假阳性）会误导提示的提取。因此，SAMRefiner 引入了一套多提示挖掘策略 (multi-prompt excavation strategy)，从初始粗糙掩码中提取多样化的、相互协作的提示：

1.  **距离引导点 (distance-guided points)**：用于提供前景和背景的位置信息。
2.  **上下文感知弹性边界框 (context-aware elastic bounding boxes, CEBox)**：用于调整初始边界框以适应上下文。
3.  **高斯风格掩码 (Gaussian-style masks)**：将粗糙掩码转换为与 SAM 兼容的连续值提示。

此外，为了应对语义分割中多对象的情况，论文引入了 **分而治之 (split-then-merge, STM) 管道**。对于通用 SAMRefiner 性能的进一步提升，论文提出了 **IoU 自适应 (IoU adaption) 步骤**，将其扩展为 SAMRefiner++，通过学习数据集中特定的 IoU 预测偏好，实现性能自提升，且无需额外标注。

**SAMRefiner 的优势**：

*   **通用性 (Universal)**：对多种分割模型、任务和数据集都适用。
*   **高效性 (Efficient)**：相比传统方法，批量处理能力使其推理速度更快。
*   **鲁棒性 (Robust)**：多提示挖掘策略对粗糙掩码的噪声具有更强的鲁棒性。
*   **无需额外标注 (No additional annotation)**：IoU 自适应步骤采用自提升方式，无需新的标注数据。

## 2. 背景回顾：Segment Anything Model (SAM)

在深入SAMRefiner的技术细节之前，我们先回顾一下 SAM 的架构及其工作原理。SAM 是一个可提示的图像分割模型，由以下三个核心组件构成：

![](../../../../../99_Assets%20(资源文件)/images/image-20250710094804023.png)

### 2.1 SAM 架构组件

1.  **图像编码器 (Image Encoder)**：
    *   **作用**：将输入图像编码成图像嵌入 (image embedding)。
    *   **实现**：基于 Vision Transformer (ViT) 模型，并使用 MAE (Masked Autoencoders) 进行预训练。
    *   **输出**：对于输入图像 $I \in \mathbb{R}^{H \times W \times 3}$，图像编码器会生成一个 $16 \times$ 下采样的图像嵌入 $F_{im} \in \mathbb{R}^{h \times w \times c}$，其中 $(h, w) = (H/16, W/16)$， $c$ 是特征维度。

2.  **提示编码器 (Prompt Encoder)**：
    *   **作用**：将不同类型的提示（points, boxes, masks, text）编码为可供解码器使用的提示嵌入。
    *   **实现**：
        *   **稀疏提示 (Sparse Prompts)**：包括点 (points)、框 (boxes)、文本 (text)。
            *   点和框：被表示为位置编码 (positional encodings) 与学习到的嵌入 (learned embeddings) 的总和。
            *   文本提示：通过 CLIP 的文本编码器处理。
        *   **密集提示 (Dense Prompts)**：即掩码 (masks)。
            *   直接与图像嵌入进行卷积，然后进行逐元素相加。

3.  **掩码解码器 (Mask Decoder)**：
    *   **作用**：根据图像嵌入和提示嵌入生成最终的分割掩码。
    *   **实现**：采用基于提示的自注意力 (prompt-based self-attention) 和双向交叉注意力 (two-way cross-attention)。
        *   这种注意力机制允许提示到图像以及图像到提示嵌入之间的交互，从而同时更新编码后的图像和提示特征。
        *   经过两个解码器层后，输出的掩码 token 经过一个 3 层的 MLP 处理，然后与上采样的图像嵌入进行空间上的逐点乘积 (spatially point-wise product)，生成目标掩码。
    *   **多掩码模式 (Multi-mask Mode)**：SAM 能够为每个输入提示生成单个掩码或多个掩码（通常是三个掩码），以解决歧义问题。它通过一个额外的 IoU token 来预测每个生成掩码的置信度，该置信度反映了预测掩码与目标对象之间的 IoU (Intersection over Union)。实验表明，选择 IoU 预测最佳的掩码通常优于单掩码模式。

### 2.2 SAM 在掩码细化任务中的挑战

虽然 SAM 在许多图像分割任务中表现出色，但将其直接应用于掩码细化任务并非易事，主要挑战在于：

*   **噪声敏感性**：粗糙掩码中存在的缺陷（假阴性、假阳性误差）会扭曲提示的提取，从而误导 SAM 的预测结果。例如，粗糙掩码的紧密边界框 (tight box) 对假阴性和假阳性错误非常敏感。
*   **掩码提示的局限性**：SAM 无法单独将原始粗糙掩码作为有效的输入提示。它在预训练时，掩码提示仅仅作为点和框提示在级联细化过程中的辅助，将前一迭代的预测 logits 作为输入来指导下一迭代。这意味着 SAM 需要连续值、logit 形式的掩码作为输入，而原始的二值粗糙掩码不符合这个要求。

## 3. SAMRefiner 技术模型细节

SAMRefiner 的核心在于其 **提示挖掘 (Prompt Excavation)** 策略，以及为了适应语义分割任务和提升性能而引入的 **IoU 自适应 (IoU Adaption)** 步骤。

### 3.1 提示挖掘 (Prompt Excavation)

SAMRefiner 的提示挖掘策略旨在从初始粗糙掩码中自动生成多样化且噪声容忍的提示，包括距离引导点、上下文感知弹性边界框和高斯风格掩码。这些提示协同工作，以减轻粗糙掩码中缺陷的影响。

#### 3.1.1 距离引导点 (Distance-Guided Points)

**目的**：提供对象的精确前景和背景点，克服二值粗糙掩码难以确定重要点的问题。

**原理**：利用目标中心往往是前景且特征可区分，而边界处通常不确定性较高这一先验知识。

**数据流与张量变化**：

1.  **输入**：
    *   初始粗糙掩码 $M_{coarse} \in \{0, 1\}^{H \times W}$。
2.  **前景点 (Positive Prompt) 提取**：
    *   计算粗糙掩码中所有前景像素（值为1的像素）到最近背景像素（值为0的像素）的欧氏距离，得到一个距离图 $D_{fg} \in \mathbb{R}^{H \times W}$。
    *   选择 $D_{fg}$ 中值最大的前景像素点 $(x_p, y_p)$ 作为正提示。这个点代表了前景区域的“中心”或最内部的点，对边界噪声不敏感。
    *   **张量变化**：$M_{coarse} \to D_{fg}$ (Distance Transform 运算) $\to (x_p, y_p)$ (argmax 运算)。

3.  **背景点 (Negative Prompt) 提取**：
    *   负提示的原则：
        1.  距离前景区域最远。
        2.  位于前景区域的边界框内。
    *   **实践工程**：
        *   首先，计算粗糙掩码的最小外接矩形（bounding box）$B_{tight}$。
        *   在 $B_{tight}$ 内部，排除前景区域，形成一个潜在的背景区域。
        *   在这个背景区域内，寻找距离所有前景像素最远的点 $(x_n, y_n)$ 作为负提示。这通常通过计算到前景区域的距离图，并选择背景区域内距离最大的点来实现。
    *   **张量变化**：$M_{coarse} \to B_{tight}$ (min/max 坐标提取) $\to D'_{bg}$ (Distance Transform，前景区域外采样) $\to (x_n, y_n)$ (argmax 运算)。

4.  **SAM 输入**：
    *   将点 $(x_p, y_p)$ 编码为正提示，点 $(x_n, y_n)$ 编码为负提示。这些点通常由其坐标和对应的类别（正/负）表示，并转换为 SAM 提示编码器可接受的格式，即位置编码与学习嵌入的和。

**分析**：这种策略避免了直接使用几何中心点作为提示，因为几何中心点容易受到粗糙掩码中假阳性或假阴性区域的影响而偏离真实目标中心。距离引导的方法更具鲁棒性，能够有效定位目标的核心区域。

#### 3.1.2 上下文感知弹性边界框 (Context-Aware Elastic Bounding Boxes, CEBox)

**目的**：动态调整粗糙掩码的紧密边界框，以适应其周围的上下文，解决紧密边界框对粗糙掩码中假阴性像素敏感的问题。

**原理**：通过分析图像特征嵌入与粗糙掩码特征嵌入的相似性，判断边界框是否需要向外扩展。

**数据流与张量变化**：

1.  **输入**：
    *   原始图像 $I \in \mathbb{R}^{H \times W \times 3}$。
    *   初始粗糙掩码 $M_{coarse} \in \{0, 1\}^{H \times W}$。
    *   初始紧密边界框 $B_{tight}$（由 $M_{coarse}$ 的最大/最小坐标确定）。

2.  **图像特征编码**：
    *   图像 $I$ 经过 SAM 的图像编码器，生成特征嵌入 $F_{im} \in \mathbb{R}^{h \times w \times c}$。(回顾 $h=H/16, w=W/16$)。
    *   **张量变化**：$I \to F_{im}$。

3.  **粗糙掩码与特征对齐**：
    *   将粗糙掩码 $M_{coarse}$ 调整大小 (resize) 到与 $F_{im}$ 相同的空间分辨率，得到 $\hat{M} \in \{0, 1\}^{h \times w}$。
    *   **张量变化**：$M_{coarse} \to \hat{M}$ (resize 运算)。

4.  **查询嵌入 (Query Embedding) 计算**：
    *   计算 $\hat{M}$ 中前景区域（$\hat{M}>0$ 的区域）对应的图像特征 $F_{im}$ 的平均值，作为查询嵌入 $F_{query} \in \mathbb{R}^{c}$。
    *   **公式推导**：
        $$F_{query} = \frac{1}{|\mathbb{1}_{\hat{M}>0}|} \sum_{(i,j) \text{ s.t. } \hat{M}_{i,j}>0} (F_{im})_{i,j}$$
        其中 $\mathbb{1}_{\hat{M}>0}$ 是指示函数，表示 $\hat{M}$ 中前景像素的位置，$|\cdot|$ 表示元素数量。
    *   **张量变化**：$(\hat{M}, F_{im}) \to F_{query}$ (masking, sum, mean 运算)。

5.  **相似度图 (Similarity Map) 生成**：
    *   计算 $F_{query}$ 与下采样图像特征 $\hat{F}_{im} \in \mathbb{R}^{H \times W \times c}$（这是为了与原始图像尺寸保持一致，但通常这里指的是 $F_{im}$ upscale 回到 $H \times W$ 尺寸或者直接在 $h \times w$ 尺寸上进行相似度计算。论文中 $F_{im} \in \mathbb{R}^{h \times w \times c}$，然后 $\hat{F}_{im} \in \mathbb{R}^{H \times W \times c}$，这可能表示对 $F_{im}$ 进行了某种上采样）。这里我们假设相似度是在 $H \times W$ 尺度上计算的。
    *   计算 $F_{query}$ 和 $\hat{F}_{im}$ 之间每个空间位置的余弦相似度 (或点积相似度)，得到相似度图 $Sim \in \mathbb{R}^{H \times W}$。
    *   将 $Sim$ 二值化，阈值为 0.5：$Sim = [F_{query} \cdot \hat{F}_{im}]_{>=0.5}$。这将识别出与粗糙掩码核心特征相似的图像区域。
    *   **张量变化**：$(F_{query}, \hat{F}_{im}) \to Sim$ (dot product, threshold 运算)。

6.  **迭代式边界框扩展**：
    *   对每个方向（左、右、上、下）进行迭代：
        *   将当前边界框 $B$ 沿当前方向**扩大 10% 对应边长**，得到扩展区域 $S_{context}$。
        *   在 $S_{context}$ 中计算二值化相似度图 $Sim$ 的**正比例 (positive ratio)**，记为 $Ratio_{pos}$。
        *   如果 $Ratio_{pos}$ 超过阈值 $\lambda$（论文中 $\lambda=0.1$），则认为该方向需要扩展。
        *   为了避免过度扩展，限制每次扩展的最大像素数量，并进行多次迭代以实现渐进式扩展。
    *   **张量变化**：循环迭代，每次迭代 $B \to B'$ (expand 运算)；裁剪 $Sim$ 得到 $Sim_{context}$ (crop 运算)；数前景像素 $Sim_{context} \to Ratio_{pos}$ (count 运算)；比较 $Ratio_{pos}$ 和 $\lambda$ 决定是否更新 $B$。

7.  **SAM 输入**：
    *   最终得到的弹性边界框 CEBox 编码为 SAM 提示编码器可接受的格式，即位置编码与学习嵌入的和。

**分析**：CEBox 解决了粗糙掩码中假阴性导致边界框过小的问题。通过上下文感知，它能智能地扩展边界框，确保目标区域被充分覆盖，同时避免引入过多的无关背景。图4a 的顶部两行展示了 CEBox 的效果。

#### 3.1.3 高斯风格掩码 (Gaussian-Style Masks)

**目的**：解决原始 SAM 不支持二值掩码作为初始提示的问题，并利用对象中心先验知识。

**原理**：将二值粗糙掩码转换为连续值的高斯分布掩码，其峰值位于距离背景最远的前景点，并沿径向衰减。

**数据流与张量变化**：

1.  **输入**：
    *   初始粗糙掩码 $M_{coarse} \in \{0, 1\}^{H \times W}$。
    *   掩码中心点 $(x_0, y_0)$，如同距离引导点策略中确定的距离背景最远的前景点。

2.  **高斯掩码生成**：
    *   对于掩码中的每个像素 $(x, y)$，计算其与中心点 $(x_0, y_0)$ 的距离。
    *   根据以下公式生成高斯风格掩码 $GM(x, y)$：
        $$GM(x,y) = \omega \cdot \exp\left(-\frac{(x-x_0)^2 + (y-y_0)^2}{|\mathbb{1}_{M_{coarse}>0}| \cdot \gamma}\right)$$
        *   $\omega$ 是控制幅度 (amplitude) 的因子，默认为 15。
        *   $\gamma$ 是控制分布跨度 (span) 的因子，默认为 4。
        *   $|\mathbb{1}_{M_{coarse}>0}|$ 是粗糙掩码中前景像素的数量，用于归一化分母。
    *   **张量变化**：$M_{coarse} \to (x_0, y_0)$ (argmax 运算) $\to GM \in \mathbb{R}^{H \times W}$ (逐像素计算)。

3.  **SAM 输入**：
    *   将 $GM$ 作为密集提示送入 SAM 的提示编码器。由于 $GM$ 是连续值，它能够与 SAM 掩码解码器的 logits-based 输入兼容。

**分析**：
*   **兼容性**：解决了 SAM 对二值掩码输入的不兼容问题。
*   **对象中心先验**：与距离引导点策略一致，强调目标中心区域的重要性，同时平滑边缘区域，使其更具鲁棒性。图4a 的最后两行展示了掩码提示的效果。
*   **参数影响**：$\omega$ 和 $\gamma$ 的选择影响高斯掩码的强度和范围。附录 C.2 和图9a 中的消融实验表明，中等偏高的 $\omega$ 值可以显著提升掩码提示的效果，且对较高的 $\omega$ 和 $\gamma$ 值不敏感。

### 3.1.4 语义分割中的应用：分而治之 (Split-Then-Merge, STM) 管道

**目的**：解决 SAM 在语义分割中处理多对象（尤其是对象跨度大或类别混杂时）的困难。

**原理**：将语义掩码中的多个连通区域拆分，然后根据一定的准则将有意义的区域合并，将其转化为更适合 SAM 处理的实例级别掩码。

**数据流与张量变化**：

1.  **输入**：
    *   初始语义粗糙掩码 $M_{coarse\_semantic} \in \{0, 1, \dots, K\}^{H \times W}$，其中 $K$ 为类别数。

2.  **拆分 (Split)**：
    *   对于 $M_{coarse\_semantic}$ 中的每个类别或整个图像，识别所有的连通区域 (connected components)。
    *   每个连通区域被赋予一个唯一的区域标签，得到 $M_{label} \in \{1, \dots, R\}^{H \times W}$，其中 $R$ 是初始连通区域的数量。
    *   **张量变化**：$M_{coarse\_semantic} \to M_{label}$ (connected components 算法)。这通常涉及图像处理库中的函数，例如 OpenCV 的 `connectedComponents`。

3.  **合并 (Merge)**：
    *   目标是将逻辑上属于同一对象但被拆分的片段合并，或排除噪声区域。
    *   **算法 1 (Region Merging Strategy)** 详细描述了合并过程：
        *   **初始化**：$M_{merge} = M_{label}$，记录当前合并状态；$M_{stm} = \emptyset$，存储最终的 STM 掩码列表。
        *   **循环迭代每个区域对 $(i, j)$**：
            *   提取区域 $i$ 和 $j$ 的最小边界框 $B_i, B_j$ 及其各自的面积 $a_{box\_i}, a_{mask\_i}$ 和 $a_{box\_j}, a_{mask\_j}$。
            *   计算合并区域 $i$ 和 $j$ 后的边界框 $\bar{B}$ 及其面积 $\bar{a}_{box}$。
            *   **合并条件判断**：
                *   如果 $(a_{box\_i} + a_{box\_j}) > \mu \cdot \bar{a}_{box}$ **AND** $(a_{mask\_i} + a_{mask\_j}) > \mu \cdot \bar{a}_{box}$：
                    *   **含义**：合并前两个区域的边界框总面积（或掩码总面积）显著大于合并后新边界框的面积（$\mu$ 是一个超参数，默认为 0.5）。这表示合并一个远距离的或不相关的区域会导致边界框过度膨胀，或者掩码区域稀疏。只有当合并不会导致边界框过度膨胀且合并区域的掩码面积足够稠密时才进行合并。
                    *   **操作**：合并区域 $i$ 和区域 $j$，更新 $M_{merge}$。
        *   从最终的 $M_{merge}$ 中提取所有独特的合并区域标签 $G$。
        *   对于 $G$ 中的每个合并区域 $k$，将其掩码添加到 $M_{stm}$ 列表中。
    *   **张量变化**：$M_{label} \to \{B_i, a_{box\_i}, a_{mask\_i}\}$ (min/max 坐标提取，count 运算) $\to (\bar{B}, \bar{a}_{box})$ (union/min/max 坐标提取) $\to M_{merge}$ (label assignment) $\to M_{stm}$ (mask extraction)。

4.  **SAMRefiner 处理**：
    *   $M_{stm}$ 中的每个实例掩码现在可以作为独立的粗糙掩码输入到 SAMRefiner 的提示挖掘模块中，生成相应的点、框和高斯掩码提示。
    *   SAM 对每个实例生成细化掩码。
    *   最后，将所有细化的实例掩码组合回原始的语义分割图。

**实践工程**：附录 C.1 进一步阐述了 STM。图4b 和图10 展示了 STM 的效果，它能有效提高多对象语义分割的质量。

### 3.2 IoU 自适应 (IoU Adaption) - SAMRefiner++

**目的**：提升 SAM 的 IoU 预测准确性，使其更好地选择最佳掩码，尤其是在目标数据集上。

**原理**：利用粗糙掩码作为先验知识，通过少量的适配训练，提高 SAM 内部 IoU 头的 top-1 预测准确率，而不会影响 SAM 自身的掩码生成能力。

**数据流与张量变化**：

1.  **基础 SAMRefiner 流程**：
    *   给定图像 $I$ 和粗糙掩码 $M_{coarse}$。
    *   通过提示挖掘生成多组提示 (点、框、高斯掩码)。
    *   SAM 接收提示和图像嵌入，输出多套预测掩码 $M_1, M_2, M_3$ 及对应的 IoU 预测值 $x_1, x_2, x_3$ (即 $IoU_{pred}$)。
    *   默认情况下，SAMRefiner 会选择具有最高 $IoU_{pred}$ 值的掩码作为最终结果。

2.  **IoU 预测的局限性**：
    *   SAM 原始的 IoU 头并非针对特定下游任务或数据集进行训练，导致其 $IoU_{pred}$ 可能不准确，无法总是选出与真实 IoU ($IoU_{GT}$) 最接近的掩码。
    *   实验发现，对于点和框提示，粗糙掩码与 SAM 输出掩码的 IoU ($IoU_{coarse}$) 具有更好的 top-1 准确率，但在多提示情况下 $IoU_{coarse}$ 表现不佳。

3.  **自适应策略**：
    *   **目标**：提升 SAM IoU 预测的 top-1 准确率。
    *   **方法**：在 SAM 的 IoU 头中引入一个 **LoRA-style 适配器 (adaptor)** 进行微调。
    *   **模型固定**：固定预训练 SAM 的原始模型参数，只训练 LoRA 适配器，以保留 SAM 的零样本迁移能力和掩码生成能力。
    *   **LoRA 适配器位置**：放置在 IoU MLP 的最后一层。

4.  **训练过程 (Ranking Loss)**：
    *   **无需 GT 掩码**：利用粗糙掩码作为监督信号。
    *   **损失函数**：采用**排序损失 (ranking loss)**，而非回归损失，以应对 $IoU_{coarse}$ 的不精确性。
    *   **数据流**：
        *   对于每次推理，SAM 会输出 $n$ ($n=3$) 个预测掩码 $M_i$ 及其对应的 IoU 预测值 $x_i$。
        *   **计算粗糙 IoU**：对于每个预测掩码 $M_i$，计算它与输入的**粗糙掩码** $M_{coarse}$ 之间的 IoU，得到 $IoU_{coarse\_i}$。
        *   **确定最佳粗糙掩码索引**：找到 $IoU_{coarse\_i}$ 中值最大的掩码索引 $j$，即 $j = \arg\max_i (IoU_{coarse\_i})$。
        *   **排序损失计算**：
            $$\text{loss} = \sum_{i=1, i \neq j}^{n} \max(0, x_i - x_j + m)$$
            *   $n$ 是总掩码数量（SAM 通常为 3）。
            *   $m$ 是用于控制最小差异的边距 (margin)，默认为 0.02。
            *   **含义**：该损失鼓励由 SAM 的 IoU 头预测出的最佳掩码 $x_j$ 的 IoU 值高于其他掩码 $x_i$，且至少保持 $m$ 的裕度。
    *   **训练数据**：仅在**单提示情况**下（例如，只使用点提示或框提示）进行适配器训练，因为此时粗糙 IoU 对选择目标掩码的指导作用更强 (如图 5c 所示)。
    *   **推理应用**：训练好的适配器在推理时用于多提示情况。

**实践工程**：
*   优化器：SGD，学习率 0.01。
*   批大小：5。
*   训练 Epoch：1 个 epoch，学习率在 60 和 100 步时降至十分之一。
*   LoRA Rank：4。
*   **IoU Adaption 是可选步骤**，只在 DAVIS-585 上进行。
*   图 3b 详细展示了 IoU Adaption 的架构。

**分析**：
*   **自提升**：无需额外标注数据，利用现有粗糙掩码的“先验”信息进行自适应。
*   **模型零样本能力保留**：只修改 IoU 头，不影响 SAM 的特征表示和掩码生成能力。
*   **效果**：显著提升了最佳掩码选择的 top-1 准确率 (表1，IoU Adaption 前后对比)，进而提升最终的 IoU 性能。

## 4. 总结

SAMRefiner 成功地将 SAM 适应于通用的掩码细化任务，克服了 SAM 在处理粗糙掩码中的缺陷。其核心贡献在于：

1.  **多提示挖掘策略**：结合距离引导点、上下文感知弹性边界框和高斯风格掩码，有效地从粗糙掩码中提取噪声容忍的、协作的提示。
2.  **分而治之 (STM) 管道**：解决了 SAM 在语义分割中处理多对象场景的难题。
3.  **IoU 自适应 (SAMRefiner++)**：通过 LoRA 适配器和排序损失，无需额外标注即可提升 SAM 的 IoU 预测准确性，实现性能自提升。

SAMRefiner 在广泛的基准测试和不同设置下都展现出卓越的准确性和效率，特别是在处理不精确的人工标注和高分辨率图像方面具有巨大潜力。论文也讨论了其局限性，如对极端粗糙掩码的敏感性，以及语义模糊性问题，并为未来的研究提供了方向。