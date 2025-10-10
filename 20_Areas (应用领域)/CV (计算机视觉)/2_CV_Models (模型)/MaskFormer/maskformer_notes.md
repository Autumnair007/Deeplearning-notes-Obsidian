---
type: concept-note
tags:
  - cv
  - image-segmentation
  - panoptic-segmentation
  - instance-segmentation
  - semantic-segmentation
  - full-supervision
  - transformer
  - detr
  - maskformer
  - mask-classification
status: done
model: MaskFormer
year: 2021
---
参考资料：[[2107.06278v2\] Per-Pixel Classification is Not All You Need for Semantic Segmentation](https://arxiv.org/abs/2107.06278v2)

[maskformer论文笔记](maskformer_paper_notes.md)

------

在讲解 MaskFormer 之前，我们首先需要理解它试图解决的问题。

传统的语义分割（Semantic Segmentation）模型，如 FCN、U-Net 等，其核心思想是进行“逐像素分类”（Per-Pixel Classification）。也就是说，模型需要为图像中的每一个像素预测一个类别标签（比如：人、车、天空）。这种方法虽然直观，但在处理实例分割（Instance Segmentation）和全景分割（Panoptic Segmentation）等更复杂的任务时，会变得非常复杂和低效，因为“每像素分类假设输出数量是静态的，不能返回可变数量的预测区域/段，而这对于实例级任务是必需的”。

MaskFormer 的核心贡献在于，它提出了一个全新的视角：**分割任务的本质不是为每个像素分类，而是预测一组掩码（mask），并为每一个掩码预测一个类别标签。** 这个思想的转变，就是 MaskFormer 的精髓所在。它将分割任务从一个“逐像素分类”问题，转化为了一个“掩码分类”（Mask Classification）问题。

作者的关键观察是：“掩码分类具有足够的通用性，可以解决语义和实例级分割任务”。基于此观测，MaskFormer 能够以统一的模型、损失和训练过程同时解决语义分割和实例分割任务，甚至全景分割（Panoptic Segmentation）。在 MaskFormer 之前，现有掩码分类方法（如 Mask R-CNN 和 DETR）通常需要预测边界框，这限制了它们在纯语义分割中的应用。而 MaskFormer 则“消除了对边界框预测的依赖”，并通过一个简化的框架实现了一致性。

接下来，我们将深入模型的内部，详细剖析其结构和原理。

## 1.先验知识讲解

> 以下内容论文笔记中有详细解释。

### 1.1. 每像素分类形式

对于每像素分类，分割模型旨在预测 $H \times W$ 图像中每个像素所有 $K$ 个可能类别的概率分布：
$$
y = \{p_i | p_i \in \Delta_K\}_{i=1}^{H \cdot W}
$$
其中 $\Delta_K$ 是 $K$ 维概率单纯形。这意味着模型为图像的每个像素输出一个 $K$ 维向量，该向量中的元素表示该像素属于 $K$ 个预定义类别中每一个的可能性。

训练每像素分类模型是直接的：给定每个像素的真实类别标签 $y^{gt} = \{y^{gt}_i | y^{gt}_i \in \{1,...,K\}\}_{i=1}^{H \cdot W}$，通常应用每像素交叉熵（负对数似然）损失：
$$
L_{pixel-cls}(y, y^{gt}) = \sum_{i=1}^{H \cdot W} -\log p_i(y^{gt}_i)
$$

### 1.2. 掩码分类形式

掩码分类将分割任务拆分为两个主要步骤：

1) 将图像划分为 $N$ 个区域，表示为二值掩码 $\{m_i | m_i \in [0,1]^{H \times W}\}_{i=1}^N$；
2) 将每个区域作为一个整体关联到 $K$ 个类别上的某种分布。与对每个像素进行分类不同，基于掩码分类的方法预测一组二值掩码，每个掩码关联一个单一的类别预测。

为了联合对分割区域进行分组和分类（即进行掩码分类），将期望的输出 $z$ 定义为 $N$ 对概率-掩码对的集合，即 $z = \{(p_i, m_i)\}_{i=1}^N$。与每像素类别概率预测不同，对于掩码分类，概率分布 $p_i \in \Delta_{K+1}$ 除了 $K$ 个类别标签外，还包含一个辅助的“无目标”（∅）标签。∅ 标签预测给那些不对应于任何 $K$ 个类别的掩码。值得注意的是，掩码分类允许预测多个具有相同关联类别的掩码，使其适用于语义和实例级分割任务。

为了训练掩码分类模型，需要预测集 $z$ 和 $N_{gt}$ 个真实分割区域 $z^{gt} = \{(c^{gt}_i, m^{gt}_i) | c^{gt}_i \in \{1,...,K\}, m^{gt}_i \in \{0,1\}^{H \times W}\}_{i=1}^{N_{gt}}$ 之间的匹配 $\sigma$。这里 $c^{gt}_i$ 是第 $i$ 个真实分割区域的真实类别。由于预测集 $|z|=N$ 和真实集 $|z^{gt}|=N_{gt}$ 的大小通常不同，假设 $N \ge N_{gt}$ 并用“无目标”标记 ∅ 填充真实标签集，以允许一对一匹配。

MaskFormer 采用**二分匹配（Bipartite Matching）** 进行预测与真实值之间的匹配，而不是固定的匹配方式。二分匹配基于匹配成本（Matching Cost）进行，该成本结合了类别预测的准确性（使用 Focal Loss）和掩码预测的相似性（使用 Dice Loss）。匹配成本定义为：
$$
\mathcal{L}_{match}(y_i, \hat{y}_{\sigma(i)}) = -\hat{p}_{\sigma(i)}(c_i) + \mathcal{L}_{mask}(m_i, \hat{m}_{\sigma(i)})
$$
其中 $y_i=(c_i,m_i)$ 是第 $i$ 个真实物体，包含类别 $c_i$ 和掩码 $m_i$。$\hat{y}_{\sigma(i)}=(\hat{p}_{\sigma(i)},\hat{m}_{\sigma(i)})$ 是与第 $i$ 个真实物体相匹配的预测。$\hat{p}_{\sigma(i)}(c_i)$ 是预测的类别概率，$\mathcal{L}_{mask}$ 是二值掩码损失。

给定匹配，主要的掩码分类损失 $L_{mask-cls}$ 由交叉熵分类损失和每个预测分割的二值掩码损失 $L_{mask}$ 组成：
$$
L_{mask-cls}(z, z^{gt}) = \sum_{j=1}^N \left[ -\log p_{\sigma(j)}(c^{gt}_j) + \mathbb{1}_{c^{gt}_j \neq \emptyset} L_{mask}(m_{\sigma(j)}, m^{gt}_j) \right]
$$
其中 $\mathbb{1}_{c^{gt}_j \neq \emptyset}$ 是指示函数，表示只有当真实目标不是“无目标”（∅）时，才计算掩码损失。作者通常使用焦点损失 (Focal Loss) 和 Dice Loss 的线性组合作为 $L_{mask}$。

## 2. MaskFormer 模型架构

![](../../../../99_Assets%20(资源文件)/images/b42d82c5b1805be4ab73fac5f6b8f5e4.png)

<div style="text-align: center;">
<b>MaskFormer 概述。我们使用骨干网络提取图像特征 F。像素解码器逐步上采样图像特征以提取每像素嵌入 E_pixel。Transformer 解码器关注图像特征并生成 N 个每段嵌入 Q。这些嵌入独立生成 N 个类别预测以及 N 个相应的掩码嵌入 E_mask。然后，模型通过像素嵌入 E_pixel 和掩码嵌入 E_mask 之间的点积，再经过 sigmoid 激活，预测 N 个可能重叠的二值掩码预测。对于语义分割任务，我们可以通过简单矩阵乘法（见 3.4 节）组合 N 个二值掩码及其类别预测来获得最终预测。注意，乘法的维度以灰色显示。<b>
</div>

MaskFormer 的核心思想是将分割任务重新定义为一个**掩码分类（Mask Classification）**问题。它不再对每个像素进行分类，而是直接预测一组$N$个掩码，并为每个掩码分配一个类别标签。其整体架构可以分为三个核心部分：
1.  **像素级模块 (Pixel-level Module)**：由一个骨干网络（Backbone）和一个像素解码器（Pixel Decoder）组成，用于提取高分辨率的、包含丰富空间信息的逐像素嵌入特征。
2.  **Transformer 模块 (Transformer Module)**：利用一个标准的 Transformer 解码器，结合图像特征和一组可学习的查询，生成$N$个“区域提议”或者说“每段嵌入”（Per-Segment Embeddings），这些嵌入编码了每个预测分割区域的全局信息。
3.  **分割模块 (Segmentation Module)**：最终将 Transformer 的输出解码为一组$N$对（类别概率，二值掩码）的预测结果，即$\{(p_i, m_i)\}_{i=1}^N$。

### 模型整体的数据流过程：

1.  **输入图像**：输入一张RGB图像，$IMG \in \mathbb{R}^{H \times W \times 3}$ (H:高度, W:宽度, 3:通道数)。
2.  **像素编码器 (Backbone)**：将$IMG$作为输入，通过一个标准的卷积神经网络（如ResNet）提取多尺度特征图。这一步的输出是相对低分辨率的图像特征$F \in \mathbb{R}^{C_F \times \frac{H}{S} \times \frac{W}{S}}$，其中$S$是特征图的总步长（论文中通常为$S=32$）。
3.  **像素解码器 (Pixel Decoder)**：对骨干网络提取的特征图$F$进行融合和逐步上采样（例如使用FPN结构），生成一个高分辨率的**逐像素嵌入**（Per-pixel Embeddings）。这个嵌入包含了精细的局部细节信息。
    *   输入：骨干网络的特征图 $F$。
    *   输出：$E_{pixel} \in \mathbb{R}^{C_E \times \frac{H}{4} \times \frac{W}{4}}$。$H_{pixel}$和$W_{pixel}$通常是输入图像下采样4倍后的尺寸，即步长为4。
4.  **Transformer 模块 (Transformer Decoder)**：接收骨干网络输出的特征$F$和一组可学习的查询，通过多层注意力机制生成每段嵌入。
    *   输入：图像特征$F$和$N$个可学习的位置嵌入（Object Queries）$Q_{learn} \in \mathbb{R}^{N \times C_Q}$。
    *   输出：$Q_{seg\_emb} \in \mathbb{R}^{N \times C_Q}$ (经过上下文聚合的每段嵌入)。这些嵌入编码了模型预测的每个分割区域的全局信息。
5.  **分割模块 (Segmentation Module)**：将$Q_{seg\_emb}$和$E_{pixel}$解码为最终的类别和掩码预测。这一步分为两个并行的分支：
    *   **类别预测分支**：
        *   输入：$Q_{seg\_emb} \in \mathbb{R}^{N \times C_Q}$。
        *   通过一个线性分类器和Softmax激活函数，为每个段嵌入预测其类别。
        *   输出：$P_{class} \in \mathbb{R}^{N \times (K+1)}$ (每个查询的类别概率分布，其中K是数据集的类别数，+1表示“无目标”类别 $\emptyset$ )。
    *   **掩码预测分支**：
        *   输入：$Q_{seg\_emb} \in \mathbb{R}^{N \times C_Q}$ 和 $E_{pixel} \in \mathbb{R}^{C_E \times \frac{H}{4} \times \frac{W}{4}}$。
        *   首先，一个多层感知机(MLP)将$Q_{seg\_emb}$转换为$N$个掩码嵌入$E_{mask} \in \mathbb{R}^{N \times C_E}$。
        *   然后，通过计算$E_{mask}$和$E_{pixel}$的点积，为每个查询生成一个低分辨率的二值掩码预测。
        *   输出：$M_{pred} \in \mathbb{R}^{N \times \frac{H}{4} \times \frac{W}{4}}$ (每个查询的低分辨率二值掩码预测)。
6.  **Sigmoid 激活和上采样**：对$M_{pred}$中的每个掩码应用Sigmoid函数，使其值域变为$[0, 1]$，然后通过双线性插值等方式上采样至原始图像分辨率。
    *   输出：$M_{final} \in \mathbb{R}^{N \times H \times W}$ (最终的$N$个二值掩码预测)。
7.  **推理阶段（根据任务类型）**：将$P_{class}$和$M_{final}$结合，生成最终的全景/语义分割结果。例如，在语义分割中，通过将每个掩码与其类别概率相乘并求和，得到每个类别的最终分割图。

### 详细数据流与张量维度变化

为了更清晰地理解模型内部的运作，我们来追踪一下张量在网络中传递时的维度变化。假设输入图像尺寸为$H \times W \times 3$，查询数量为$N$，类别数为$K$。

1.  **Backbone**
    *   输入: $IMG \in \mathbb{R}^{3 \times H \times W}$ (Pytorch中通道在前)
    *   经过一个例如ResNet-50的骨干网络，通常会产生一个步长为32的特征图。
    *   输出: $F \in \mathbb{R}^{C_F \times \frac{H}{32} \times \frac{W}{32}}$。其中$C_F=2048$。

2.  **Pixel Decoder**
    *   输入: $F \in \mathbb{R}^{2048 \times \frac{H}{32} \times \frac{W}{32}}$。
    *   像素解码器（如FPN）通过一系列的上采样和卷积操作，将特征图的分辨率提升，同时降低通道数，以生成包含丰富细节的像素级嵌入。
    *   输出: $E_{pixel} \in \mathbb{R}^{C_E \times \frac{H}{4} \times \frac{W}{4}}$。其中$C_E$是嵌入维度，通常设置为256。

3.  **Transformer Decoder**
    *   **输入1 (Queries)**: 一组可学习的嵌入作为对象查询，$Q_{learn} \in \mathbb{R}^{N \times C_Q}$。$N$是预设的查询数量（例如100），$C_Q$是Transformer内部的隐藏层维度（通常为256）。
    *   **输入2 (Image Features)**: 来自Backbone的特征图$F$。在使用前，通常会先通过一个1x1卷积将其通道数从$C_F$降至$C_Q$，并展平空间维度。
        *   $F \rightarrow F' \in \mathbb{R}^{C_Q \times \frac{H}{32} \times \frac{W}{32}}$
        *   $F' \rightarrow F'' \in \mathbb{R}^{(\frac{H}{32} \times \frac{W}{32}) \times C_Q}$ (展平后)
    *   在Transformer解码器内部，Queries通过自注意力（Self-Attention）和交叉注意力（Cross-Attention，关注$F''$）机制不断更新。
    *   输出: $Q_{seg\_emb} \in \mathbb{R}^{N \times C_Q}$。这$N$个向量现在是融合了图像全局信息的“每段嵌入”。

4.  **Segmentation Module**
    *   **类别预测 (Classification Head)**:
        *   输入: $Q_{seg\_emb} \in \mathbb{R}^{N \times C_Q}$。
        *   经过一个线性层 (Linear Layer)，将$C_Q$维的嵌入映射到$K+1$维。
        $$
        \text{Linear}: \mathbb{R}^{N \times C_Q} \rightarrow \mathbb{R}^{N \times (K+1)}
        $$
        *   输出: $P_{class} \in \mathbb{R}^{N \times (K+1)}$ (经过Softmax激活后)。
    *   **掩码预测 (Mask Head)**:
        *   首先，一个MLP将$Q_{seg\_emb}$转换为掩码嵌入。
        $$
        \text{MLP}: \mathbb{R}^{N \times C_Q} \rightarrow \mathbb{R}^{N \times C_E}
        $$
        *   得到掩码嵌入 $E_{mask} \in \mathbb{R}^{N \times C_E}$。
        *   然后，计算$E_{mask}$与$E_{pixel}$的点积。为了维度匹配，我们将$E_{mask}$和$E_{pixel}$进行矩阵乘法。
        $$
        M_{pred}[i, h, w] = \sum_{c=1}^{C_E} E_{mask}[i, c] \cdot E_{pixel}[c, h, w]
        $$
        *   这个操作可以高效地通过一次矩阵乘法完成：
        $$
        (E_{mask} \in \mathbb{R}^{N \times C_E}) \times (E_{pixel} \in \mathbb{R}^{C_E \times (\frac{H}{4} \cdot \frac{W}{4})}) \rightarrow M_{pred} \in \mathbb{R}^{N \times (\frac{H}{4} \cdot \frac{W}{4})}
        $$
        *   最后将$M_{pred}$ reshape回二维空间。
        *   输出: $M_{pred} \in \mathbb{R}^{N \times \frac{H}{4} \times \frac{W}{4}}$。

5.  **Final Prediction**
    
    *   输入: $M_{pred} \in \mathbb{R}^{N \times \frac{H}{4} \times \frac{W}{4}}$。
    *   对每个掩码应用Sigmoid激活，然后上采样到$H \times W$。
    *   输出: $M_{final} \in \mathbb{R}^{N \times H \times W}$。这些掩码与$P_{class}$一起，构成了模型的最终输出，可用于后续的损失计算或推理。

现在，我们来详细拆解每一个模块。

***

### 2.1 Pixel-level module (像素级模块)

![](../../../../99_Assets%20(资源文件)/images/7fb95e5b2dac1327c6044cd181dc2448.png)

**模块目标**：此模块是 MaskFormer 的感知基础，其核心目标有两个：1) 提取一个包含丰富上下文信息的低分辨率图像特征，用于后续 Transformer 模块的全局分析；2) 生成一个包含精细空间细节的高分辨率逐像素嵌入，用于最终精确的掩码生成。
**模块构成**：它由一个**骨干网络 (Backbone)** 和一个**像素解码器 (Pixel Decoder)** 组成。

### 1. 骨干网络 (Backbone)
**功能**：作为特征提取器，负责将输入图像转换为多层次的特征表示。

*   **输入**：一张标准尺寸的图像 $IMG \in \mathbb{R}^{H \times W \times 3}$ (例如，`224x224x3` 或 `512x512x3`)。
*   **实现**：通常使用强大的卷积神经网络（CNN）如 `ResNet` (R50, R101) 或视觉 Transformer (Vision Transformer) 如 `Swin Transformer` 作为骨干网络。根据论文，`Swin-Transformer` 骨干网络（尤其是 `Swin-L`）在 ADE20K 语义分割上取得了 SOTA (55.6 mIoU)，证明了强大骨干网络的重要性。
*   **骨干网络输出**：骨干网络会输出多个不同尺度的特征图，代表不同层次的语义信息。
    *   例如，对于 `ResNet-50`，其四个阶段会分别输出步长为 $S=4, 8, 16, 32$ 的特征图，我们称之为 $Res_2, Res_3, Res_4, Res_5$。它们的维度分别为：
        *   $Res_2 \in \mathbb{R}^{256 \times H/4 \times W/4}$
        *   $Res_3 \in \mathbb{R}^{512 \times H/8 \times W/8}$
        *   $Res_4 \in \mathbb{R}^{1024 \times H/16 \times W/16}$
        *   $Res_5 \in \mathbb{R}^{2048 \times H/32 \times W/32}$
    *   **关键输出 (Image Features $F$)**：其中，**分辨率最低、但语义信息最丰富**的特征图，即 $Res_5$，被定义为论文中的图像特征 $F$。这个 $F \in \mathbb{R}^{C_F \times \frac{H}{S} \times \frac{W}{S}}$ (此处 $C_F=2048, S=32$) 将被直接送入接下来的 **Transformer 模块**，作为其交叉注意力机制的 Key 和 Value，为对象查询（Object Queries）提供全局图像上下文。

### 2. 像素解码器 (Pixel Decoder)
**功能**：这是一个类似特征金字塔网络 (FPN) 的轻量级解码器架构，其任务是融合骨干网络输出的多尺度特征，并逐步上采样，最终生成一个高分辨率的、用于生成掩码的逐像素嵌入。

*   **输入**：来自骨干网络的多个尺度的特征图（$Res_2, Res_3, Res_4, Res_5$）。
*   **输出**：一个融合了多尺度信息的**逐像素嵌入（Per-Pixel Embeddings）** $E_{pixel} \in \mathbb{R}^{C_E \times H/4 \times W/4}$。
    *   $C_E$ 是统一的嵌入特征维度（论文中为 `256`）。
    *   $H/4, W/4$ 是最终输出的特征图尺寸，它保留了较为精细的空间细节，是生成高质量掩码的基础。

### 详细数据流与张量变化
下面我们以 `ResNet-50` 为例，详细追踪像素解码器内部的数据流和张量维度变化，假设 $C_E=256$。
*   **Step 1: 处理最深层特征**
    
    *   输入：$Res_5 \in \mathbb{R}^{2048 \times H/32 \times W/32}$。
    *   操作：通过一个 $1 \times 1$ 卷积将其通道数从 2048 降至 $C_E=256$。
    *   输出：$P_5 \in \mathbb{R}^{256 \times H/32 \times W/32}$。
*   **Step 2: 第一次融合与上采样**
    
    *   输入1：$Res_4 \in \mathbb{R}^{1024 \times H/16 \times W/16}$。通过 $1 \times 1$ 卷积降维得到 $L_4 \in \mathbb{R}^{256 \times H/16 \times W/16}$。
    *   输入2：$P_5$。通过 $2 \times$ 双线性插值上采样得到 $U_5 \in \mathbb{R}^{256 \times H/16 \times W/16}$。
    *   操作：将两者逐元素相加：$P_4 = L_4 + U_5$。然后对结果应用一个 $3 \times 3$ 卷积、`GroupNorm (GN)` 和 `ReLU` 激活，以平滑和提炼特征。
    *   输出：$P_4 \in \mathbb{R}^{256 \times H/16 \times W/16}$。
*   **Step 3: 第二次融合与上采样**
    *   输入1：$Res_3 \in \mathbb{R}^{512 \times H/8 \times W/8}$。通过 $1 \times 1$ 卷积降维得到 $L_3 \in \mathbb{R}^{256 \times H/8 \times W/8}$。
    *   输入2：$P_4$。通过 $2 \times$ 上采样得到 $U_4 \in \mathbb{R}^{256 \times H/8 \times W/8}$。
    *   操作：$P_3 = L_3 + U_4$，同样后接 $3 \times 3$ 卷积、GN 和 ReLU。
    *   输出：$P_3 \in \mathbb{R}^{256 \times H/8 \times W/8}$。
*   **Step 4: 第三次融合与上采样**
    *   输入1：$Res_2 \in \mathbb{R}^{256 \times H/4 \times W/4}$。通过 $1 \times 1$ 卷积降维（如果需要，此处通道数已为256，可直接使用）得到 $L_2 \in \mathbb{R}^{256 \times H/4 \times W/4}$。
    *   输入2：$P_3$。通过 $2 \times$ 上采样得到 $U_3 \in \mathbb{R}^{256 \times H/4 \times W/4}$。
    *   操作：$P_2 = L_2 + U_3$，后接 $3 \times 3$ 卷积、GN 和 ReLU。
    *   输出：$P_2 \in \mathbb{R}^{256 \times H/4 \times W/4}$。
*   **Step 5: 生成最终逐像素嵌入**
    *   最终的 $P_2$ 就是我们需要的逐像素嵌入 $E_{pixel}$。这个高分辨率、富含多尺度信息的特征图，将与分割模块生成的掩码嵌入（Mask Embeddings）进行点积运算，以预测出最终的二值掩码。
    $$
    E_{pixel} = P_2 \in \mathbb{R}^{256 \times H/4 \times W/4}
    $$

总结来说，像素级模块通过“分工合作”的方式，同时为后续的 Transformer 模块和分割模块提供了两种不同但都至关重要的特征表示。

***

### 2.2 Transformer Module (Transformer Decoder)

![](../../../../99_Assets%20(资源文件)/images/67f6196f683e2cab64dbb640e44be995.png)

**模块目标**：此模块是 MaskFormer 的“大脑”，其核心目标是将一组通用的、可学习的“对象查询”（Object Queries）转化为$N$个与图像内容紧密相关的“每段嵌入”（Per-Segment Embeddings）。它通过多层注意力机制，聚合全局图像上下文，让每个查询“认领”并编码图像中的一个特定分割区域（如一个人、一辆车或一片天空）的信息。

**模块构成**：该模块采用一个标准的 Transformer 解码器结构，由多个（论文中默认为6个）相同的解码器层堆叠而成。

**数据流过程**：

*   **输入**：
    1.  **图像特征 (Image Features)** $F \in \mathbb{R}^{C_F \times \frac{H}{S} \times \frac{W}{S}}$：这是来自**骨干网络 (Backbone)** 的低分辨率、高层语义特征图（例如，对于 ResNet-50，维度为 $2048 \times H/32 \times W/32$）。**注意：这里的输入不是来自像素解码器的 $E_{pixel}$**。
    2.  **N 个可学习的查询 (Learnable Queries)**：$Q_{learn} \in \mathbb{R}^{N \times C_Q}$ (例如，`100x256`)。这些查询是模型的可学习参数，**在训练开始时随机初始化。**它们是与具体图像内容无关的“槽位”（slots），在训练过程中，模型会学会利用这些槽位去捕捉图像中的不同对象或区域。每个查询还会加上一个可学习的位置编码，以引入顺序和区分性。
    3.  **N值设定解析：** N是一个预设超参数，定义了模型可预测的最大实例数量（如COCO数据集设为100）。其值通常略大于图像中常见目标总数，以确保足够“槽位”，并通过匈牙利算法与真实目标进行最优匹配。
*   **Transformer Decoder 内部**：每个解码器层都包含三个关键子层：自注意力、交叉注意力和前馈网络。
    
    1.  **自注意力（Self-Attention）**：
        
        *   **作用**：在查询之间建立联系，允许它们相互通信。这有助于模型理解对象间的关系（例如，避免两个查询预测同一个物体），从而进行全局的推理和去重。
        *   **过程**：将输入的$N$个查询作为该层的 Query, Key, 和 Value 进行自注意力计算。
        $$
        Q_{sa\_out} = \text{SelfAttention}(Q_{in})
        $$
        *   输出的查询 $Q_{sa\_out}$ 融合了其他查询的信息。
    2.  **交叉注意力（Cross-Attention）**：
        *   **作用**：这是将查询与图像内容进行关联的核心步骤。它允许每个查询“审视”整个图像特征图 $F$，并提取与其最相关的信息。
        *   **过程**：
            *   `Query (Q)`：来自上一步自注意力层输出的 $N$ 个查询向量。
            *   `Key (K) 和 Value (V)`：均来自于骨干网络输出的图像特征 $F$。在使用前，$F$ 需要经过处理以匹配维度并引入空间信息。
        *   通过交叉注意力，这 $N$ 个查询向量会“查询”图像特征，聚合与自己最相关的区域信息。例如，如果一个查询的目标是预测“人”，它会对特征图 $F$ 中属于“人”的区域产生更高的注意力权重。
    3.  **前馈网络（Feed-Forward Network, FFN）**：
        *   **作用**：对经过注意力层处理后的每个查询进行独立的非线性变换，增加模型的表达能力。它通常由两个线性层和一个 ReLU 激活函数组成。
*   **输出**：经过6层解码器层的迭代处理后，初始的通用查询 $Q_{learn}$ 被转化为最终的**每段嵌入（Per-Segment Embeddings）** $Q_{seg\_emb} \in \mathbb{R}^{N \times C_Q}$ (例如，`100x256`)。此时，每一个嵌入向量都高度特化，浓缩了图像中某一个特定实例或区域的类别和位置信息，准备好被送入最终的分割模块。

### 详细数据流与张量变化
我们以一个解码器层为例，追踪其内部详细的张量流动。假设 $N=100$, $C_Q=256$。
*   **Step 0: 输入准备**
    
    *   **查询输入 (Query Input)**: $Q_{in} \in \mathbb{R}^{100 \times 256}$ (上一层解码器或初始查询)。
    *   **图像特征准备 (Image Feature Preparation)**:
        *   输入: $F \in \mathbb{R}^{2048 \times \frac{H}{32} \times \frac{W}{32}}$。
        *   1) **通道降维**: 通过一个 $1 \times 1$ 卷积将通道数从 2048 降至 $C_Q=256$。得到 $F' \in \mathbb{R}^{256 \times \frac{H}{32} \times \frac{W}{32}}$。
        *   2) **空间展平**: 将空间维度展平，并调换维度顺序以符合 Transformer 输入格式。得到 $F_{flat} \in \mathbb{R}^{(\frac{H}{32} \cdot \frac{W}{32}) \times 256}$。
        *   3) **添加位置编码**: 为 $F_{flat}$ 的每个空间位置添加一个固定的（如正弦）或可学习的位置编码，以注入空间信息。
*   **Step 1: 自注意力 (Self-Attention)**
    *   输入: $Q_{in} \in \mathbb{R}^{100 \times 256}$。
    *   操作: $Q, K, V$ 均由 $Q_{in}$ 经过线性变换生成。
    $$
    Q_{sa\_out} = \text{LayerNorm}(Q_{in} + \text{SelfAttention}(Q_{in}))
    $$
    *   输出: $Q_{sa\_out} \in \mathbb{R}^{100 \times 256}$。注意这里的残差连接和层归一化（LayerNorm）。
*   **Step 2: 交叉注意力 (Cross-Attention)**
    *   **Query**: $Q_{sa\_out} \in \mathbb{R}^{100 \times 256}$。
    *   **Key & Value**: 均由准备好的图像特征 $F_{flat} \in \mathbb{R}^{(\frac{H}{32} \cdot \frac{W}{32}) \times 256}$ 经过线性变换生成。
    *   操作:
    $$
    Q_{ca\_out} = \text{LayerNorm}(Q_{sa\_out} + \text{CrossAttention}(Q=Q_{sa\_out}, K=F_{flat}, V=F_{flat}))
    $$
    *   输出: $Q_{ca\_out} \in \mathbb{R}^{100 \times 256}$。查询向量吸收了图像信息。
*   **Step 3: 前馈网络 (FFN)**
    
    *   输入: $Q_{ca\_out} \in \mathbb{R}^{100 \times 256}$。
    *   操作:
    $$
    Q_{out} = \text{LayerNorm}(Q_{ca\_out} + \text{FFN}(Q_{ca\_out}))
    $$
    *   输出: $Q_{out} \in \mathbb{R}^{100 \times 256}$。这是该解码器层的最终输出。

这个 $Q_{out}$ 将作为下一个解码器层的输入 $Q_{in}$，整个过程重复6次。最后一层解码器的输出即为最终的每段嵌入 $Q_{seg\_emb}$。

***

### 2.3 Segmentation Module (分割模块)

![](../../../../99_Assets%20(资源文件)/images/be2710b4b55a13e49ae022037445d93f.png)

**模块目标**：作为模型的最终“解码器”，此模块负责将 Transformer 输出的 $N$ 个高度抽象的“每段嵌入”翻译成具体的、可解释的分割结果，即为每个嵌入生成一对匹配的**类别概率**和**二值掩码**。

**模块构成**：该模块由两个并行且独立的预测头（Prediction Heads）组成：一个用于类别预测的**线性分类器 (Classification Head)**，和一个用于生成掩码的**多层感知机 (Mask Head)**。

**数据流过程**：两个分支都接收来自 Transformer 模块的相同的每段嵌入作为输入。

#### a) 类别预测分支 (Classification Head)

*   **作用**：为 $N$ 个每段嵌入中的每一个预测其对应的类别。
*   **输入**：$N$ 个每段嵌入向量 $Q_{seg\_emb} \in \mathbb{R}^{N \times C_Q}$ (例如，`100x256`)。
*   **操作**：通过一个简单的线性分类器（Linear Layer）后接一个 Softmax 函数，将每个嵌入向量映射到一个类别概率分布上。
*   **输出**：一个维度为 $N \times (K+1)$ 的概率分布矩阵 $P_{class} \in \mathbb{R}^{N \times (K+1)}$。其中 $K$ 是数据集中物体的类别总数，“+1”代表“无目标”（no object, $\emptyset$）这个特殊的类别，用于处理那些没有匹配到任何真实物体的查询。$P_{class}[i,j]$ 表示第 $i$ 个查询预测出的物体属于第 $j$ 个类别的概率。

#### b) 掩码预测分支 (Mask Head)

*   **作用**：为 $N$ 个每段嵌入中的每一个生成一个对应的二值分割掩码。
*   **输入**：
    
    1.  $N$ 个每段嵌入向量 $Q_{seg\_emb} \in \mathbb{R}^{N \times C_Q}$ (例如，`100x256`)。
    2.  来自**像素级模块**输出的逐像素嵌入特征图 $E_{pixel} \in \mathbb{R}^{C_E \times H/4 \times W/4}$。
*   **操作**：这是一个两步过程。
    
    1.  **生成掩码嵌入**：首先，一个 MLP 将高维的每段嵌入 $Q_{seg\_emb}$ 转换为一个同样维度的掩码嵌入 $E_{mask}$。
    2.  **点积生成掩码**：然后，通过计算每个查询的掩码嵌入与图像上每个像素的嵌入之间的**点积**（相似度），来生成最终的掩码。
    
    >2. 点积 / 内积 (Dot Product / Inner Product)
    >
    >- **操作**：两个**向量**对应位置相乘，然后**将所有乘积结果求和，得到一个标量（一个数字）**。这是衡量两个向量相似度的核心运算。
    >- **符号**：`·` 或 `<, >`
    >- **例子**：
    >  $\vec{a} = [1, 2, 3], \quad \vec{b} = [4, 5, 6]$
    >   $\vec{a} \cdot \vec{b} = (1*4) + (2*5) + (3*6) = 4 + 10 + 18 = 32$
*   **输出**：$N$ 个二值掩码预测 $M_{final} \in \mathbb{R}^{N \times H \times W}$。论文指出，使用 Sigmoid 而非 Softmax 激活是关键，因为它允许掩码之间重叠，这对于处理实例分割中的遮挡至关重要。

### 详细数据流与张量变化

我们来追踪此模块中详细的张量流动。假设 $N=100$, $C_Q=256$, $C_E=256$, 输入图像经过下采样后特征图尺寸为 $H' \times W'$ (即 $H/4 \times W/4$)。

*   **Step 0: 输入 (Inputs)**
    
    *   **每段嵌入**: $Q_{seg\_emb} \in \mathbb{R}^{100 \times 256}$
    *   **逐像素嵌入**: $E_{pixel} \in \mathbb{R}^{256 \times H' \times W'}$
    
*   **Step 1: 类别预测 (Classification Prediction)**
    
    *   输入: $Q_{seg\_emb} \in \mathbb{R}^{100 \times 256}$
    *   操作: 通过一个线性层 `nn.Linear(256, K+1)` 进行变换。
    $$
    P_{logits} = \text{Linear}(Q_{seg\_emb})
    $$
    *   张量变化: $\mathbb{R}^{100 \times 256} \rightarrow \mathbb{R}^{100 \times (K+1)}$
    *   经过 Softmax 激活后，得到最终的类别概率。
    *   输出: $P_{class} \in \mathbb{R}^{100 \times (K+1)}$
    
*   **Step 2: 掩码嵌入生成 (Mask Embedding Generation)**
    *   输入: $Q_{seg\_emb} \in \mathbb{R}^{100 \times 256}$
    *   操作: 通过一个 MLP（论文中明确为2个隐藏层，256通道）进行变换。这个 MLP 的作用是学习一个从“类别-位置”的抽象嵌入到“如何形成掩码”的嵌入的映射。
    $$
    E_{mask} = \text{MLP}(Q_{seg\_emb})
    $$
    *   张量变化: $\mathbb{R}^{100 \times 256} \rightarrow \mathbb{R}^{100 \times 256}$ (维度保持不变，因为 $C_Q = C_E = 256$)
    *   输出: $E_{mask} \in \mathbb{R}^{100 \times 256}$

*   **Step 3: 掩码生成与上采样 (Mask Generation & Upsampling)**
    
    *   输入1: $E_{mask} \in \mathbb{R}^{100 \times 256}$
    *   输入2: $E_{pixel} \in \mathbb{R}^{256 \times H' \times W'}$
    *   操作 (点积): 通过矩阵乘法高效计算所有查询与所有像素的点积。
    $$
    M_{logits} = E_{mask} \cdot E_{pixel}
    $$
    *   张量变化:
        1.  将 $E_{pixel}$ 展平: $\mathbb{R}^{256 \times H' \times W'} \rightarrow \mathbb{R}^{256 \times (H' \cdot W')}$
        2.  矩阵乘法: $(\mathbb{R}^{100 \times 256}) \times (\mathbb{R}^{256 \times (H' \cdot W')}) \rightarrow \mathbb{R}^{100 \times (H' \cdot W')}$
        3.  将结果 Reshape 回空间维度: $\mathbb{R}^{100 \times (H' \cdot W')} \rightarrow \mathbb{R}^{100 \times H' \times W'}$
    *   输出: $M_{logits} \in \mathbb{R}^{100 \times H' \times W'}$
    *   操作 (激活与上采样):
        1.  对 $M_{logits}$ 应用 Sigmoid 函数，得到 $M_{sigmoid} \in [0,1]^{100 \times H' \times W'}$。
        2.  通过双线性插值将 $M_{sigmoid}$ 上采样4倍，恢复到原始图像尺寸。
    *   最终输出: $M_{final} \in \mathbb{R}^{100 \times H \times W}$

### 2.4 训练过程

在训练时，模型会输出 $N$ 组（类别概率，掩码）对。为了计算损失，需要将这 $N$ 个预测与真实标注（Ground Truth）进行匹配。这个过程使用了**匈牙利算法进行二分图匹配（Bipartite Matching）**，与 DETR 中的方法完全相同。
*   **匹配代价（Matching Cost）**：简单来说，对于每一个真实物体（ground truth mask），算法会在 $N$ 个预测中找到一个与之“最匹配”的预测。匹配的依据是匹配代价，它综合了类别预测的准确性和掩码预测的相似性。
    $$
    \mathcal{C}_{\text{match}}(y_i, \hat{y}_{\sigma(i)}) = -\hat{p}_{\sigma(i)}(c_i) + \mathcal{L}_{\text{mask}}(m_i, \hat{m}_{\sigma(i)})
    $$
    其中：
    *   $y_i = (c_i, m_i)$ 是第 $i$ 个真实物体，包含类别 $c_i$ 和掩码 $m_i$。
    *   $\hat{y}_{\sigma(i)} = (\hat{p}_{\sigma(i)}, \hat{m}_{\sigma(i)})$ 是与第 $i$ 个真实物体相匹配的预测。
    *   $\hat{p}_{\sigma(i)}(c_i)$ 是预测的类别概率（使用 Focal Loss 形式计算代价）。
    *   $\mathcal{L}_{\text{mask}}$ 是掩码损失，由 Focal Loss 和 Dice Loss 的线性组合构成。
*   **最终损失计算**：找到代价最低的最优匹配 $\sigma$ 后，就可以计算最终的损失函数。损失由所有匹配上的对的损失组成，包括分类损失和掩码损失。
    $$
    \mathcal{L}_{\text{total}} = \sum_{i=1}^{M} \left[ -\log\hat{p}_{\sigma(i)}(c_i) + \mathcal{L}_{\text{mask}}(m_i, \hat{m}_{\sigma(i)}) \right]
    $$
    其中 $M$ 是真实物体的数量。对于分类损失，还会对未匹配的查询计算其类别为“无目标”$\emptyset$的损失。

## 3. 掩码分类推理
MaskFormer 支持两种推理策略，具体选择取决于评估指标和任务需求。
*   **通用推理（General inference）**：
    *   适用于全景分割和实例分割任务。
    *   将每个像素 $[h,w]$ 分配给 $N$ 个预测的概率-掩码对中的一个，根据以下规则：$\arg \max_{i:c_i \neq \emptyset} p_i(c_i) \cdot m_i[h,w]$。
    *   其中 $c_i = \arg \max_{c \in \{1,...,K,\emptyset\}} p_i(c)$ 是每个概率-掩码对 $i$ 最可能的类标签。这个公式直观地确保了只有当最可能的类别概率 $p_i(c_i)$ 和掩码预测概率 $m_i[h,w]$ 都很高时，像素才会被分配给该对。
    *   分配给相同概率-掩码对 $i$ 的像素形成一个片段，并被标记为 $c_i$。对于实例分割，索引 $i$ 用于区分同一类的不同实例。
    *   为了降低全景分割中的假阳性率，通常会过滤掉低置信度预测，并去除被其他预测大部分遮挡（如重叠区域超过设定阈值）的二值掩码。
*   **语义推理（Semantic inference）**：
    *   专门为语义分割设计，通过简单的矩阵乘法完成，避免了通用推理中的硬性分配，效果更好。
    *   计算每个像素属于各个语义类别的概率，然后取概率最大的类别：
    $$
    \text{prediction}[h,w] = \text{argmax}_{c \in \{1,...,K\}} \sum_{i=1}^N p_i(c) \cdot m_i[h,w]
    $$
    *   这里通过对所有 $N$ 个预测进行加权求和（权重为类别概率），实现了对概率-掩码对的边缘化。
    *   此方法不包括“无目标”类别（∅），因为标准语义分割要求每个输出像素都有一个明确的类别标签。
    *   作者观察到直接最大化这个每像素类别似然（即以此为目标进行端到端训练）会导致性能不佳，推测是因为梯度被均匀地分布到每个查询，使得训练复杂化。因此，这种方法仅用于推理。

### 3.1 实现细节
*   **骨干网络（Backbone）**：MaskFormer 兼容任何骨干架构。论文中使用了标准的基于卷积的 `ResNet` (R50, R101) 和基于 Transformer 的 `Swin-Transformer`。其中，`Swin-Transformer` 在大型数据集上表现出了显著优势。
*   **像素解码器（Pixel decoder）**：设计为轻量级的 `FPN` 架构。在解码器中，低分辨率特征图被2倍上采样并与来自骨干网络的相应分辨率的投影特征图相加。所有特征图通道维度为256。
*   **Transformer 解码器（Transformer decoder）**：使用与 DETR 相同的 `Transformer` 解码器设计。默认使用6个 `Transformer` 解码器层和100个查询，并且在每个解码器层之后都应用相同的损失（辅助损失，Auxiliary Losses），这有助于模型更快收敛。论文指出，即使只有一个解码器层，MaskFormer 在语义分割方面也具有竞争力，但在实例级分割中需要多层才能有效去除重复预测。
*   **分割模块（Segmentation module）**：用于预测掩码嵌入的 `MLP` 有2个隐藏层，每层256个通道，设计类似于 DETR 中的框预测头。像素嵌入 $E_{pixel}$ 和掩码嵌入 $E_{mask}$ 的维度均为256。
*   **损失权重（Loss weights）**：掩码损失 $\mathcal{L}_{\text{mask}}$ 是焦点损失和 Dice Loss 的线性组合：
    $$
    \mathcal{L}_{\text{mask}}(m, m^{gt}) = \lambda_{focal}\mathcal{L}_{focal}(m, m^{gt}) + \lambda_{dice}\mathcal{L}_{dice}(m, m^{gt})
    $$
    其中超参数根据实验设置为 $\lambda_{focal} = 20.0$ 和 $\lambda_{dice} = 1.0$。遵循 DETR 的实践，分类损失中“无目标”（∅）类别的权重被设置为0.1，以平衡正负样本。

## 4. 讨论

MaskFormer 旨在证明掩码分类是通用的分割范式，是每像素分类的有力替代品。它被视为 `DETR` 的“无框”版本，并通过以下方式改进了性能和效率：

*   **使用掩码匹配优于使用框匹配**：MaskFormer 直接与掩码预测进行匹配，而非像 `DETR` 那样依赖边界框。实验显示，基于掩码的匹配具有明显优势 [<sup>1</sup>](https://arxiv.org/abs/2107.06278v2)，特别是对于“stuff”类别，因为这些区域通常占据大面积，基于框的匹配可能会模糊“stuff”区域。这揭示了掩码分类在处理像天空、草地等非实例性区域时更具优势，因为它不会被边界框的限制所束缚。
*   **MaskFormer 的掩码头计算量更小**：MaskFormer 的掩码头设计比 `DETR` 更高效，前者直接生成高分辨率的二值掩码预测，并让像素解码器中的逐像素嵌入被所有查询共享。而后者对每个查询应用独立的上采样模块，导致计算成本高出 `N` 倍 (N为查询数量) [<sup>1</sup>](https://arxiv.org/abs/2107.06278v2)。这使得 MaskFormer 在保持性能的同时，具有更高的效率。

## 5. MaskFormer 的优势总结

1.  **统一框架**：它提供了一个简洁而强大的统一框架，可以同时处理语义分割、实例分割和全景分割这三个任务，而不需要对模型结构做大的改动或使用复杂的辅助损失。只需要根据不同任务调整最终的输出和损失函数即可。
2.  **效率和性能**：相比于逐像素分类，MaskFormer 的“掩码分类”方法更加高效。它不需要对每个像素都进行密集的分类计算，而是生成固定数量（N个）的掩码。在多个分割基准测试中，它都取得了顶尖的性能，尤其在类别数量多的数据集上表现优异。
3.  **思想革新**：它最重要的贡献是挑战了“逐像素分类”这一传统范式，为分割领域的研究开辟了新的方向。后续的许多工作，如 `Mask2Former`、`K-Net` 等，以及专注于“Transformer-based Instance Segmentation”的方法都是在 MaskFormer 的思想上进行的改进和延伸。 `OneFormer` 更是将 MaskFormer 的概念提升到“通用图像分割”，以一个Transformer模型处理多种分割任务。
4.  **开-世界（Open-World）能力**：未来，MaskFormer的范式有望被扩展到开放世界（Open-World）场景，即模型能够处理在训练集中未见过的“未知类”（unknown classes）。例如，`Open-World Panoptic Segmentation` 领域的新基准 `PANIC` 就是针对自动驾驶场景中“未见过的对象”进行像素级和实例级标注，这与MaskFormer的掩码分类思想高度契合，预示着其在更复杂、更现实场景中的应用潜力。

