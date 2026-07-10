---
type: paper-note
tags:
  - cv
  - transformer
  - semantic-segmentation
  - weakly-supervised
  - mctformer
  - vit
  - wsss
  - attention
  - cam
status: done
model: Multi-Class Token Transformer
venue: CVPR2022
---
论文网址：[Multi-Class Token Transformer for Weakly Supervised Semantic Segmentation](https://openaccess.thecvf.com/content/CVPR2022/papers/Xu_Multi-Class_Token_Transformer_for_Weakly_Supervised_Semantic_Segmentation_CVPR_2022_paper.pdf)

本地PDF文件：[[../../../../../../99_Assets (资源文件)/papers/Multi-Class_Token_Transformer_for_Weakly_Supervised_Semantic_Segmentation_CVPR_2022_paper.pdf]]

***
## 摘要 (Abstract)

This paper proposes a new transformer-based frame work to learn class-specific object localization maps as pseudo labels for weakly supervised semantic segmenta tion (WSSS). Inspired by the fact that the attended regions of the one-class token in the standard vision transformer can be leveraged to form a class-agnostic localization map, we investigate if the transformer model can also effectively capture class-specific attention for more discriminative ob ject localization by learning multiple class tokens within the transformer. To this end, we propose a Multi-class To ken Transformer, termed as MCTformer, which uses multi ple class tokens to learn interactions between the class to kens and the patch tokens. The proposed MCTformer can successfully produce class-discriminative object localiza tion maps from the class-to-patch attentions corresponding to different class tokens. We also propose to use a patch level pairwise affinity, which is extracted from the patch to-patch transformer attention, to further refine the local ization maps. Moreover, the proposed framework is shown to fully complement the Class Activation Mapping (CAM) method, leading to remarkably superior WSSS results on the PASCAL VOC and MS COCO datasets. These results underline the importance of the class token for WSSS.

本文提出了一种新的基于Transformer的框架，旨在学习特定类别的目标定位图，作为弱监督语义分割（WSSS）的伪标签。受标准视觉Transformer中单个类别令牌（one-class token）的关注区域可用于形成类别无关定位图这一事实的启发，我们探究了通过在Transformer中学习多个类别令牌，模型是否也能有效捕捉特定类别的注意力，从而实现更具判别性的目标定位。

为此，我们提出了一种多类别令牌Transformer，称为**MCTformer**，它利用多个类别令牌来学习类别令牌与图块令牌（patch tokens）之间的交互。所提出的MCTformer能够根据对应于不同类别令牌的“类别-图块”注意力（class-to-patch attentions），成功生成具有类别判别性的目标定位图。我们还提出利用从“图块-图块”Transformer注意力（patch-to-patch transformer attention）中提取的图块级成对亲和度（patch-level pairwise affinity），来进一步优化定位图。

此外，该框架被证明能与类激活映射（CAM）方法形成充分互补，从而在PASCAL VOC和MS COCO数据集上取得了显著优异的WSSS结果。这些结果凸显了类别令牌在弱监督语义分割中的重要性。

## 1. 引言 (Introduction)

弱监督语义分割（WSSS）旨在通过使用弱监督来减少对像素级真值标签的依赖。此任务的关键一步是利用弱标签生成高质量的伪分割真值标签。图像级标签可以提供简单的弱标签，它们只指示某些类别的存在与否，而没有任何真值定位信息。

传统的WSSS方法通常依赖于类激活映射（CAM）从卷积神经网络（CNNs）中提取目标定位图。尽管使用了复杂的CAM扩展策略或多步训练，现有方法在定位对象的完整性和准确性方面仍然表现出局限性。

Vision Transformer（ViT）作为第一个专为计算机视觉设计的Transformer模型，最近在多项视觉任务上取得了性能突破。特别地，ViT凭借其强大的建模长距离上下文的能力，在大规模图像识别方面取得了SOTA性能。ViT将输入图像分割成不重叠的图像块（patches），并将其转换为一个向量序列。ViT还使用一个额外的类别令牌（class token）来聚合整个图像块令牌序列的信息。尽管在一些最新的Transformer方法中移除了类别令牌，但这项工作将强调其对于弱监督语义分割的重要性。

近期的一项工作DINO揭示了自监督ViT特征中包含图像语义分割的显式信息。更具体地说，观察到可以从类别令牌的注意力图中发现语义场景布局。这些注意力图在无监督分割任务中取得了有希望的结果。尽管已证明Transformer注意力中的不同头（heads）可以关注图像的不同语义区域，但如何将一个头与正确的语义类别关联起来仍不清楚。也就是说，这些注意力图仍然是类别无关的（参见图1）。

![](../../../../../../99_Assets%20(资源文件)/images/63c60de0821bec0bd1faf4e3630e81ca.png)

**图1**展示了标准Vision Transformer（(a) ViT）和提出的多类令牌Transformer（(b) MCTformer）之间的关键区别。(a) 中，只有一个红色的类别令牌，它与所有蓝色的图像块令牌进行交互，生成的注意力图是类别无关的。这意味着它可能无法区分图像中多个不同类别的对象。(b) 中，提出了多个类别令牌（例如，红色、绿色、蓝色方块代表不同类别的令牌），每个令牌与图像块令牌交互，生成的注意力图是类别特定的。这样，每个类别令牌能够捕获特定类别的目标定位信息。

从Transformer中利用类别特异性注意力是具有挑战性的。作者认为，现有基于Transformer的工作有一个共同问题，即仅使用一个类别令牌，这使得在单张图像上准确地定位不同对象变得困难。这主要有两个原因：
首先，一个类令牌的设计本质上不可避免地会捕获来自其他对象类别和背景的上下文信息。换句话说，由于只考虑一个类别令牌，它自然会学习不同对象类别的类别特异性和通用表示，这导致了相对非判别性和噪声较多的对象定位。
其次，模型使用这唯一的类别令牌来学习与数据集中多个不同对象类别的图像块令牌进行交互。因此，模型容量不足以达到目标判别性定位性能。

为了解决这些问题，一个直接的想法是利用多个类别令牌，每个令牌将负责学习不同对象类别的表示。为此，论文提出了一个多类令牌Transformer（MCTformer），其中采用多个类别特定令牌来利用类别特定的Transformer注意力。作者拥有类别特定令牌的目标不能通过简单地增加ViT中类别令牌的数量来实现，因为这些类别令牌仍然没有特定的含义。为了确保每个类别令牌能够有效地学习特定对象类别的高级判别性表示，他们提出了一种类别感知训练策略（class-aware training strategy）用于多个类别令牌。

更具体地说，通过对Transformer编码器输出的类别令牌沿嵌入维度进行平均池化，生成类别分数，并直接通过真值类别标签进行监督。这就在每个类别令牌和相应的类别标签之间建立了**一对一的强连接**。通过这种设计，一个显著的优点是，所学习的不同类别的“类别到图像块注意力”可以直接用作类别特定的定位图。

值得注意的是，学习到的“图像块到图像块注意力”（patch-to-patch attention），作为训练的副产品且无需额外计算，可以作为图像块级成对**亲和性（patch-level pairwise affinity）**。这可以进一步优化类别特定的Transformer注意力图，显著提高定位性能。此外，作者还展示了所提出的Transformer框架在应用于图像块令牌时，能够完全补充CAM方法（通过同时学习基于类别令牌和图像块令牌表示的分类）。这导致类别令牌和图像块令牌之间高度一致，从而大大增强了其派生目标定位图的判别能力。

>在 WSSS 中，利用亲和性，就是利用像素之间“长得像不像”的关系，把确定的标签（猫头）扩散到不确定的区域（猫身子），从而得到一张完整的猫的分割图。

总而言之，论文的主要贡献有三方面：
* 提出利用类别特定的Transformer注意力进行弱监督语义分割。
* 提出一种有效的Transformer框架，包括一个新颖的多类令牌Transformer（MCTformer）耦合类别感知训练策略，以从不同类别令牌的“类别到图像块注意力”中学习类别特定的定位图。
* 提出利用“图像块到图像块Transformer注意力”作为图像块级成对亲和性，这可以显著优化类别特定的Transformer注意力。此外，所提出的MCTformer可以完全补充CAM机制，从而生成高质量的目标定位图。

所提出的方法可以为WSSS生成高质量的类别特定的多标签定位图，并在PASCAL VOC（测试集mIoU为71.6%）和MS COCO（mIoU为42.0%）上取得了新的SOTA结果。

## 2. 相关工作 (Related works)

### 2.1. 弱监督语义分割 (Weakly supervised semantic segmentation)

大多数现有的WSSS方法依赖于类激活映射（CAM）[51]从CNN中提取目标定位图。原始CAM图通常不完整，边界粗糙，因此无法为语义分割网络的学习提供足够的监督。为了解决这个问题，研究者提出了特定的分割损失来弥补不足的分割监督，包括SEC损失[19]、CRF损失[35, 46]和对比损失[17]。此外，许多研究专注于改进从CAM图获得的伪分割标签。这些方法可分为以下几类：

**生成高质量CAM图。** 一些方法开发了启发式策略，例如“Hide & Seek”[31]和Erase[40]，应用于图像[24, 48]或特征图[16, 21]，以驱动网络学习新颖的对象模式。先前的工作还利用子类别[4]和跨图像语义[13, 25, 33]来定位更准确的对象区域。为了解决标准图像分类目标损失函数的局限性，提出了正则化损失[39, 49]来指导网络发现更多的对象区域。此外，其他一些工作[41]通过引入空洞卷积解决了标准图像分类CNN感受野有限的问题，以鼓励判别性激活传播到周围区域。

**通过亲和性学习优化CAM图。** 几项工作专注于学习成对语义亲和性以优化CAM图。Ahn等[1]提出AffinityNet来从原始CAM图的可靠种子（reliable seeds）中学习相邻像素之间的亲和性。学习到的AffinityNet可以通过随机游走传播CAM图来预测亲和性矩阵。类似地，Wang等[38]也使用分割结果中的置信像素学习了一个成对亲和性网络。在[39, 48]中，亲和性直接从分类网络的特征图中学习以优化CAM图。此外，Xu等[44]提出了一种交叉任务亲和性，它从弱监督多任务框架中的显著性（saliency）和分割表示中学习。
与之前所有基于CNN的WSSS方法不同，本文提出了一种基于Transformer的模型来提取类别特定的目标定位图。作者利用自注意力机制中的Transformer注意力图来生成目标定位图。

#### 详细分析解释：
**WSSS 的“三步走”流程**（1.训练分类器 -> 2.精炼伪标签 -> 3.训练分割模型）
#### 1. 生成高质量 CAM 图（魔改训练过程）
**位置：** 处于 **第一步（训练分类器）** 阶段。
**核心逻辑：** 标准的分类模型很“懒”，只看猫头就能认出猫，导致生成的 CAM 只有猫头。为了解决这个问题，这类方法（如 Hide & Seek、擦除法、正则化损失）必须**重新训练分类器**。
**怎么做：** 在训练时故意“刁难”模型，比如随机遮挡图片的一部分，或者在 Loss 函数里加限制。这样逼着模型为了维持高准确率，不得去寻找猫身子、猫尾巴等次要特征。
**一句话总结：** 这不是拿来即用的，而是通过**修改训练策略**，强迫分类器学会看物体的全貌。
#### 2. 通过亲和性学习优化 CAM（额外训练辅助网）
**位置：** 处于 **第一步结束后，第二步开始前** 的中间阶段。
**核心逻辑：** CAM 生成的图边界很糊，计算机不知道哪个像素和哪个像素是一伙的。这类方法（如 AffinityNet）认为光靠原本的分类器不够，需要**额外训练一个专门的网络**来学习“像素关系”。
**怎么做：** 拿 CAM 里最确定的部分（比如猫头中心）做正样本，训练这个辅助网络去判断：旁边的像素跟猫头是不是同类？学会了纹理和颜色的关联后，再用它把 CAM 的热力图扩散开来（Refinement）。
**一句话总结：** 这通常需要**额外造一个网络并单独训练**，专门用来把粗糙的 CAM 修补成精细的 Mask。
#### 3. 本文 MCTformer 的做法（自带亲和性的 Transformer）
**位置：** 回归到 **第一步（训练分类器）**，但在模型结构上换成了 Transformer。
**核心逻辑：** 虽然它也需要针对数据集（如 VOC）进行训练，但它最大的优势是**“一鱼两吃”**。
1.  **分类：** 它的 Class Token 负责找物体（替代了传统的 CAM）。
2.  **亲和性：** 它的 Patch-to-Patch Attention（图块间的注意力）天然就记录了谁和谁长得像。
**怎么做：** 在训练分类任务的同时，Transformer 内部自动就计算好了“亲和性矩阵”。到了测试（Inference）阶段，直接把中间层的 Attention 拿出来当亲和性用，不需要像 AffinityNet 那样再跑一个额外的网络。
**一句话总结：** 虽然也需要训练，但它把“分类”和“亲和性计算”**合二为一**了，训练完分类器，亲和性也就免费得到了（无需额外计算）。

### 2.2. 用于视觉任务的Transformer (Transformers for visual tasks)

Transformer [37]最初是为自然语言处理（NLP）领域中建模长序列的长距离依赖而设计的。最近，Transformer模型已被应用于各种视觉任务[18]，如图像分类[10]、显著性检测[27]和语义分割[30]，并取得了有前景的性能。第一个基于Transformer的视觉模型ViT [10]将图像分割成图像块并将其转换为令牌序列。这些令牌随后被送入多个堆叠的基于自注意力[37]层，使每个图像块具有全局感受野。

Caron等[3]将自监督方法应用于ViT，并观察到类别令牌对图像块的注意力包含了场景语义布局的信息。然而，[3]中并未建立注意力与类别之间的一一映射。此外，他们在Transformer注意力上的发现尚未扩展到弱监督学习。

另一项相关工作是TS-CAM [14]，它将CAM模块适配到ViT中。然而，TS-CAM只利用了ViT的类别无关注意力图，而本文提出的方法利用了Transformer注意力中类别特定的定位图。此外，所提出的多类令牌Transformer框架被证明比原始ViT更好地补充了CAM机制，生成了比TS-CAM更好的目标定位图（见表5）。

## 3. 多类令牌Transformer (Multi-class Token Transformer)

### 3.1. 概述 (Overview)

论文提出了一种新颖的纯基于Transformer的框架（MCTformer-V1），利用Transformer注意力中的类别特定的目标定位图。MCTformer-V1的整体架构如图2所示。

![](../../../../../../99_Assets%20(资源文件)/images/6a37b2edac3ac16b0ecabc77556392db.png)
### **Figure 2：MCTformer-V1 架构与运作流程深度解析**
这张图展示了 **MCTformer-V1** 的完整工作流，核心在于如何利用**多类别Token（Multi-class Tokens）** 在Transformer内部同时完成“分类学习”和“定位图生成”。我们将流程分为三个主要阶段：
#### 1. 输入构建与编码 (Input Construction & Encoding)
*   **输入处理**：给定一张 RGB 输入图像 $I$，首先将其切分为 $N \times N$ 个不重叠的 Patch（例如 $14 \times 14$ 个）。这些 Patches 被展平并映射为 **Patch Tokens**（图中黄色方块序列，数量为 $N^2$）。
*   **多类别 Token 引入**：与传统 ViT 仅使用 1 个 Class Token 不同，MCTformer 初始化了 **$C$ 个额外的可学习 Class Tokens**（图中红色方块序列），其中 $C$ 对应数据集的类别数（例如 PASCAL VOC 为 20）。
*   **序列拼接**：将 $C$ 个 Class Tokens 与 $N^2$ 个 Patch Tokens 拼接，并加上位置编码（**PE**），形成长度为 $C + N^2$ 的输入序列 $\mathbf{T}_{in}$。
*   **Transformer 编码**：序列进入包含 $L$ 层堆叠的 Transformer Encoder。在每一层中，通过 **Multi-Head Attention (MHA)** 模块，Class Tokens 与 Patch Tokens 进行全局交互（Message Passing），学习图像的语义特征。
#### **2. 类别感知训练 (Class-aware Training)**
*   **分类预测**：经过 $L$ 层编码后，提取输出的 $C$ 个 Class Tokens。
*   **平均池化 (Average Pooling)**：**关键细节**——模型对每个 Class Token 的输出特征维度进行**平均池化**（而不是由 MLP 映射或最大池化），得到该类的预测分数（Class scores）。
    *   *论文分析*：论文中的消融实验证明，平均池化能迫使 Class Token 关注物体所有相关的 Patch，而不仅仅是局部最具判别力的区域，从而生成更完整的定位图。
#### **3. 关注度提取与细化 (Attention Extraction & Refinement)**
这是图中虚线框内部（下半部分）的核心流程，展示了推理阶段如何生成定位图：
*   **注意力聚合 (Fusion)**：
    *   从 Transformer 的最后 $K$ 层（图中 $K$ transformer self-attention maps）提取注意力图。
    *   对这些层的 Attention Maps 进行融合（通常是取平均），得到一个全局注意力矩阵 $\mathbf{A}_{t2t}$。
*   **矩阵切片与重组 (Matrix Slicing & Reshaping)**：
    *   **Class-specific Attention (橙色部分)**：提取矩阵的前 $C$ 行、后 $N^2$ 列。这代表了 $C$ 个类别分别关注哪些图像 Patch。将这一行向量 Reshape 为 $N \times N$ 的 2D 空间热力图，即**原始的类别特定定位图 ($\mathbf{A}_{c2p}$)**。
    *   **Patch-level Pairwise Affinity (蓝色部分)**：提取矩阵的后 $N^2$ 行、后 $N^2$ 列。这代表了所有图像 Patch 两两之间的相似度/亲和力。这无需额外计算，直接构成了**亲和度矩阵 ($\mathbf{A}_{p2p}$)**。
*   **亲和度细化 (Refinement)**：
    *   利用提取出的 **Patch 级亲和度矩阵**（蓝色）对 **原始类别定位图**（橙色）进行加权优化。
    *   **数学逻辑**：如果 Patch A 和 Patch B 的亲和度高（视觉特征相似），那么 Patch A 的激活值就会传播给 Patch B。
    *   **结果**：生成了右侧展示的 **Refined class-specific transformer attention maps**。可以看到，细化后的热力图（Refined）比原始图（Reshape后的）覆盖更完整，且边界更贴合物体轮廓。

输入RGB图像首先被分割成非重叠的图像块，然后转换为图像块令牌序列。与传统Transformer只使用一个类别令牌不同，论文提出使用**多个类别令牌**。这些类别令牌与图像块令牌（加入位置嵌入信息）拼接起来，形成Transformer编码器的输入令牌。Transformer编码器使用多个Transformer块来提取图像块令牌和类别令牌的特征。在最后一层，对输出类别令牌进行平均池化以生成类别分数，而不是像传统Transformer那样使用多层感知机（MLP）进行分类预测。

在训练时，为了确保不同的类别令牌能够学习到不同的类别判别性表示，作者采用了第3.2节详细介绍的**类别感知训练策略**。计算由类别令牌直接产生的类别分数与真值类别标签之间的分类损失。这从而在每个类别令牌与相应的类别标签之间建立了强连接。在测试时，可以从Transformer中的“类别到图像块注意力”中提取类别特定的定位图。为了利用从不同Transformer层学习到的互补信息，进一步聚合来自多个层的注意力图。此外，可以从“图像块到图像块注意力”中提取图像块级成对亲和性，以进一步优化“类别到图像块注意力”，从而显著改进类别特定的定位图。这些类别特定的定位图被用作生成伪标签的种子，以监督分割模型。

### 3.2. 类别特定Transformer注意力学习 (Class-Specific Transformer Attention Learning)

#### 多类令牌结构设计 (Multi-class token structure design)

考虑一张输入图像，它被分割成 $N \times N$ 个图像块，然后转换为一个图像块令牌序列 $T_p \in \mathbb{R}^{M \times D}$，其中 $D$ 是嵌入维度，$M = N^2$。论文提出学习 $C$ 个类别令牌 $T_c \in \mathbb{R}^{C \times D}$，其中 $C$ 是类别的数量。这 $C$ 个类别令牌与图像块令牌拼接，并加入位置嵌入，形成Transformer编码器的输入令牌 $T_{in} \in \mathbb{R}^{(C+M) \times D}$。Transformer编码器有 $L$ 个连续的编码层，每一层都由一个多头注意力（MHA）模块、一个MLP和在MHA和MLP之前应用的两个LayerNorm层组成。

#### 类别特定的多类令牌注意力 (Class-specific multi-class token attention)

使用标准的自注意力层来捕获令牌之间的长距离依赖。更具体地说，首先对输入令牌序列进行归一化，并通过线性层将其转换为 $Q \in \mathbb{R}^{(C+M) \times D}$，$K \in \mathbb{R}^{(C+M) \times D}$ 和 $V \in \mathbb{R}^{(C+M) \times D}$。然后采用Scaled Dot-Product Attention机制计算查询和键之间的注意力值。每个输出令牌都是所有令牌的加权和，使用注意力值作为权重，公式如下：

$$
\text{Attention}(Q,K,V) = \text{softmax}(QK^\top / \sqrt{D})V
$$

这里可以获得一个令牌到令牌的注意力图 $A_{t2t} \in \mathbb{R}^{(C+M) \times (C+M)}$，其中 $A_{t2t} = \text{softmax}(QK^\top / \sqrt{D})$。

从全局成对注意力图 $A_{t2t}$ 中，可以提取类别对图像块的注意力 $A_{c2p} \in \mathbb{R}^{C \times M}$，即类别到图像块注意力，其中 $A_{c2p} = A_{t2t}[1:C, C+1:C+M]$，如图2中带有黄点的矩阵所示。每一行代表特定类别对所有图像块的注意力分数。利用这些注意力向量，结合所有图像块的原始空间位置，可以生成 $C$ 个类别相关的定位图。

可以从每个Transformer编码层中提取类别相关的定位图。考虑到较高层学习到更高级别的判别性表示（同时早期层捕获更多通用且低级别的视觉信息），作者提出融合来自最后 $K$ 个Transformer编码层的“类别到图像块注意力”，以在生成的对象定位图的精确度（precision）和召回率（recall）之间找到一个好的权衡。
这个过程的公式表示为：

$$
\hat{A}_{mct} = \frac{1}{K} \sum_{l}^{K} \hat{A}_{l}^{mct}
$$

其中 $\hat{A}_{l}^{mct}$ 是从所提出的MCTformer-V1的第 $l$ 个Transformer编码层提取的类别特定的Transformer注意力。融合后的图 $\hat{A}_{mct}$ 进一步通过沿两个空间维度的min-max归一化方法进行归一化，以生成最终的类别特定对象定位图 $A_{mct} \in \mathbb{R}^{C \times N \times N}$。关于如何选择 $K$ 的详细结果可以在图6中找到。

#### 类别特定注意力优化 (Class-specific attention refinement)

成对亲和性常用于先前的工作[1, 38, 44]中以优化目标定位图。它通常需要一个额外的网络或额外的层来学习亲和性图。与此不同，论文提出从所提出的MCTformer的图像块到图像块注意力中提取成对亲和性图，无需额外的计算和监督。这是通过提取图像块到图像块注意力 $A_{p2p} \in \mathbb{R}^{M \times M}$ 实现的，其中 $A_{p2p} = A_{t2t}[C+1:C+M, C+1:C+M]$，如图2中带有蓝点的矩阵所示。图像块到图像块注意力被重塑为一个 $4D$ 张量 $\hat{A}_{p2p} \in \mathbb{R}^{N \times N \times N \times N}$。提取的亲和性用于进一步优化类别特定的Transformer注意力。这个过程的公式为：

$$
A_{mct}^{\text{ref}}(c, i, j) = \sum_{k=1}^{N} \sum_{l=1}^{N} \hat{A}_{p2p}(i, j, k, l) \cdot A_{mct}(c, k, l) \tag 3
$$

其中 $A_{mct}^{\text{ref}} \in \mathbb{R}^{C \times N \times N}$ 是优化后的类别特定定位图。如表5和图5所示，使用图像块级成对亲和性进行优化可以生成具有改进外观连续性的更好的目标定位图。这在先前的工作[14]中没有观察到。

#### 类别感知训练 (Class-aware training)

与传统Transformer通过MLP对最后一层中单个类别令牌进行分类预测不同，我们有多个类别令牌 $T_{cls} \in \mathbb{R}^{C \times D}$，并且需要确保不同的类别令牌能够学习到不同的类别判别性信息。为此，对输出的类别令牌进行平均池化以生成类别分数：

$$
y(c) = \frac{1}{D} \sum_{j=1}^{D} T_{cls}(c, j)  \tag 4
$$

其中 $y \in \mathbb{R}^C$ 是类别预测， $c \in \{1, 2, ..., C\}$。$T_{cls}(c, j)$ 表示 $T_{cls}$ 中的元素，即第 $c$ 个类别令牌的第 $j$ 个特征。最后，计算类别 $c$ 的类别分数 $y(c)$ 和其真值标签之间的多标签软边际损失（multi-label soft margin loss）。这为每个类别令牌提供了强大且直接的类别感知监督，使得每个类别令牌能够捕获类别特定的信息。

>**为什么论文里叫“第 j 个特征”？**
  这只是学术论文里一种比较咬文嚼字的表达方式。在深度学习里，向量的每一个维度通常被称为一个“Feature Channel（特征通道）”或者直接叫“Feature”。所以“第 $j$ 个维度”就被写成了“第 $j$ 个特征”。

**总结：**
对于第 $c$ 个类别来说，它就是一个 $D$ 维的特征向量。论文的操作就是把这个向量里的所有数字**求了个平均**，直接拿来当做分类的分数去算 Loss。这种极简的操作去掉了传统的全连接层，直接逼迫这个 Token 本身去表征类别信息。

### 3.3. 对图像块令牌CAM的补充 (Complementarity to Patch-Token CAM)

如图3所示，论文将CAM模块[14, 50, 51]整合到所提出的多类令牌Transformer框架中，构建了一个扩展模型，命名为MCTformer-V2。

![](../../../../../../99_Assets%20(资源文件)/images/fefafed4df86c673b643373161b6dbce.png)

**图3**展示了MCTformer-V2的整体架构与工作流程。输入图像首先被转化为序列化的**图像块令牌（Patch tokens）**，与$C$个可学习的**类别令牌（Class tokens）** 拼接后送入Transformer编码器。输出端分为两条支路进行训练：上方支路直接对**类别令牌** 进行平均池化以计算分类损失 ($\mathcal{L}_{cls-class}$)；下方支路引入了一个CAM模块，将**图像块令牌**重塑为特征图，通过卷积层（CONV）和全局平均池化（GAP）计算另一路分类损失 ($\mathcal{L}_{cls-patch}$)。在推理阶段，模型融合了来自类别令牌的**MCT Attention**与来自下方支路的**PatchCAM**，并利用无需额外计算的**图像块亲和性（Patch Affinity）** 对融合图进行后处理优化，最终生成高质量的类别特定目标定位图（Refined fusion maps）。

更具体地说，给定Transformer编码器输出的令牌序列 $T_{out} \in \mathbb{R}^{(C+M) \times D}$，将其分为输出类别令牌 $T_{out}^{cls} \in \mathbb{R}^{C \times D}$ 和输出图像块令牌 $T_{out}^{pat} \in \mathbb{R}^{M \times D}$。然后将图像块令牌重塑并送入一个具有 $C$ 个输出通道的卷积层，生成一个 $2D$ 特征图 $F_{out}^{pat} \in \mathbb{R}^{N \times N \times C}$。最后，$F_{out}^{pat}$ 通过全局平均池化（GAP）层转换为类别预测。此外，还使用输出类别令牌生成类别分数（参见方程(4)）。总损失是分别通过类别令牌和图像块令牌的类别预测与图像级真值标签计算的两个多标签软边际损失之和，如下所示：

$$
L_{\text{total}} = L_{\text{cls-class}} + L_{\text{cls-patch}}
$$
>这实际上是一种 **“双重监督”策略** ：虽然两个分支使用相同的分类标签计算损失，但它们的训练目的互补。**$L_{cls-class}$** 作用于类别令牌，强迫模型通过**注意力机制**去精准“寻找”物体（优化注意力图）；**$L_{cls-patch}$** 作用于图像块令牌，强迫每个 Patch 提取出**更具判别力**的局部特征（优化特征图/PatchCAM）。两者结合，既确保了类别令牌能“看对位置”，又确保了图像块本身“特征清晰”，从而在推理时融合出既有全局语义又有局部细节的高质量定位图。
>类别令牌提供了 **“去看哪里”** 的线索（注意力），图像块令牌提供了 **“哪里有特征”** 的底图。MCTformer 将两者结合，定位才最准确。

**结合PatchCAM和类别特定的Transformer注意力。** 在测试时，可以从最后一个卷积层中提取基于图像块令牌的CAM（此后称为PatchCAM）图。通过对特征图 $F_{out}^{pat}$ 应用min-max归一化，提取PatchCAM图 $A_{pCAM}$，其中 $A_{pCAM} \in \mathbb{R}^{N \times N \times C}$。提取的PatchCAM图随后与所提出的类别特定的Transformer注意力图通过==**逐元素乘法**==操作融合，以生成融合的目标定位图 $A$：

$$
A = A_{pCAM} \circ A_{mct}
$$

其中 $\circ$ 表示Hadamard积。

>Hadamard 积在这里就是让两张图 **“互相过滤”**，只有 **两个模型都认为重要** 的区域才会被保留下来，从而得到更精准的定位图。

**类别特定对象定位图优化。** 类似于MCTformer-V1中提出的注意力优化机制（参见方程3），也可以从MCTformer-V2中提取图像块到图像块注意力图作为图像块级成对亲和性，以优化融合后的目标定位图，如下所示：

$$
A_{\text{ref}}(c, i, j) = \sum_{k=1}^{N} \sum_{l=1}^{N} \hat{A}_{p2p}(i, j, k, l) \cdot A(c, k, l)
$$

MCTformer-V2提供了一个有效的基于Transformer的框架，其中CAM方法可以灵活稳健地适应多标签图像。通过对类别令牌和图像块令牌的类别预测都应用分类损失，可以强制这两种类型令牌之间的高度一致性，以改进模型学习。这种直觉主要有两方面：首先，这种一致性约束可以被视为辅助监督，以指导学习更有效的图像块表示。其次，图像块令牌和多个类别令牌之间强大的成对交互（即信息传递）也可以导致更具代表性的图像块令牌，从而产生比仅使用一个类别令牌（如TS-CAM [14]）更具类别判别性的PatchCAM图。

## 4. 实验 (Experiments)

### 4.1. 实验设置 (Experimental Settings)

**数据集。** 论文在两个数据集上评估了所提出的方法：PASCAL VOC 2012 [11] 和 MS COCO 2014 [26]。
PASCAL VOC包含三个子集：训练集（train）、验证集（val）和测试集（test），分别包含1,464、1,449和1,456张图像。它有20个目标类别和一个背景类别用于语义分割任务。遵循先前的工作[4, 22, 32, 39, 44, 48]，使用了包含[15]中额外数据的10,582张图像的增强集进行训练。
MS COCO使用80个目标类别和一个背景类别进行语义分割。其训练集和验证集分别包含8万和4万张图像。需要注意的是，在训练期间仅使用了这些数据集的图像级真值标签。

**评估指标。** 遵循先前的工作[22]，使用平均交并比（mIoU）来评估两个基准测试集上验证集的语义分割性能。在PASCAL VOC测试集上的语义分割结果是通过官方PASCAL VOC在线评估服务器获得的。

**实施细节。** 所提出的MCTformer是使用DeiT-S骨干网络[14, 36]构建的，该网络预训练于ImageNet [9]。更具体地说，使用了DeiT-S中预训练的类别令牌来初始化所提出的多个类别令牌。遵循[14, 36]中提供的数据增强和默认训练参数。训练图像被调整为256x256，然后裁剪为224x224。对于语义分割，遵循先前的工作[1, 44, 46, 48] 使用基于ResNet38 [43]的Deeplab V1。在测试时，使用多尺度测试和CRF [6]（采用建议的超参数）进行后处理。

### 4.2. 与SOTA方法的比较 (Comparison with State-of-the-arts)

**PASCAL VOC。** 遵循常见做法[4, 22, 32, 39, 48]，在所提出的目标定位图（种子）上应用PSA [1]以在训练集上生成伪语义分割真值标签（掩膜）。如表1所示，所提出的方法在初始种子和伪真值掩膜上都比现有工作有显著的提升，比最佳初始种子[48]提高了4.3%。

![](../../../../../../99_Assets%20(资源文件)/images/0acc7b6fbdc455337bed6e9618777765.png)

**表1**显示了在PASCAL VOC训练集上，初始种子（Seed）和相应伪分割真值掩膜（Mask）的mIoU（%）评估。越高的mIoU表示越好的性能。
本方法MCTformer在Seed和Mask两项上都优于其他方法，表明其生成的初始定位图和伪标签质量更高。

表2显示所提出的MCTformer在验证集和测试集上分别取得了71.9%和71.6%的语义分割mIoU。所提出的MCTformer在仅使用图像级标签的情况下，比现有所有方法表现显著更好。特别是，MCTformer甚至可以与使用额外显著图的方法取得可比或更好的结果。

![](../../../../../../99_Assets%20(资源文件)/images/607775e9783f7e15669e794d2fd02db7.png)

**表2**比较了PASCAL VOC 2012验证集（val）和测试集（test）上WSSS方法的性能（mIoU %），使用不同的分割骨干网络。Sup.: 监督类型。I: 图像级真值标签。S: 现成的显著图（off-the-shelf saliency maps）。
MCTformer（ours）在仅使用图像级标签（I）的情况下，在验证集和测试集上都取得了最高的mIoU（71.9%和71.6%），甚至超过了许多使用了额外显著图（I+S）的方法，展现了其卓越的性能。

图4（左）显示，使用作者的伪标签训练的分割模型可以在各种具有挑战性的场景中生成准确完整的对象轮廓。

![](../../../../../../99_Assets%20(资源文件)/images/2f513a10349398144f76504ec9e2615f.png)

**图4**展示了在PASCAL VOC和MS COCO验证集上的定性分割结果。(a) 输入图像，(b) 真值分割（Ground-truth），(c) 本文方法（Ours）产生的分割结果。
从图中可以看出，本文方法（c）生成的分割结果与真值（b）非常接近，能够准确地勾勒出对象的轮廓和区域，证明了其伪标签的有效性。

**MS COCO。** 表3显示所提出的方法取得了42.0%的分割mIoU，大幅超越了最近的方法。值得注意的是，从表3可以看出，一些使用额外显著图的方法性能不如仅使用图像级标签的最新方法。这揭示了依赖预训练显著性模型的局限性，这些模型在具有挑战性的数据集上可能表现不佳。图4（右）展示了一些定性分割结果。

![](../../../../../../99_Assets%20(资源文件)/images/a4f70a0b315fb2341dd4728bfb2d9291.png)

**表3**比较了MS COCO验证集（val）上WSSS方法的性能（mIoU %）。
MCTformer（ours）在仅使用图像级标签（I）的情况下，取得了最高的mIoU（42.0%），比其他方法有显著提升，包括那些使用了额外显著图（I+S）的方法，再次验证了其有效性。

**模型复杂度。** 比较了所提出的MCTformer与常用CNN模型（ResNet38 [43]）在生成目标定位图时的模型复杂度（参数数量和乘加运算MACs）。表4显示，基于DeiT-S [36]的MCTformer方法的复杂度显著低于基于ResNet38的方法。

![](../../../../../../99_Assets%20(资源文件)/images/660f5eebaa936aeac044f0dad16136a4.png)

**表4**展示了生成目标定位图模型的复杂度。所提出的MCTformer基于DeiT-S [36]骨干网络。
MCTformer-V1和MCTformer-V2的参数量（Params）和MACs远低于ResNet38，表明了Transformer模型在保持高性能的同时，能有更低的计算和参数开销。

### 4.3. 消融研究 (Ablation Studies)

**多类特定令牌的效果。** 在传统的ViT中，类别令牌注意力只表示类别无关的定位图。TS-CAM [14] 将CAM应用于ViT的输出图像块令牌以获得类别特定的定位图。遵循其官方实现，TS-CAM在PASCAL VOC训练集上生成的对象定位图mIoU为29.9%，如表5所示。简单地在他们的PatchCAM图上添加一个ReLU层（即TS-CAM*），mIoU大幅提升了11.4%。相比之下，所提出的基线方法，即MCTformer-V1中多个类别特定令牌的类别特定的Transformer注意力图，达到了47.2%的mIoU，显著优于TS-CAM* 5.9%。这表明了所提出的基于Transformer注意力的类别特定定位图的有效性。

![](../../../../../../99_Assets%20(资源文件)/images/11517ad543716e11d36d96d820312f6d.png)

**表5**评估了在PASCAL VOC训练集上不同目标定位图的mIoU（%）。
MCTformer-V1（Attention）的mIoU为47.2%，远高于TS-CAM及其改进版本（TS-CAM*），说明多类令牌结构直接学习的类别特定注意力非常有效。
当加入图像块亲和性（PatchAffinity）和PatchCAM融合后（MCTformer-V2），mIoU进一步提高，尤其在融合所有机制后，达到了61.7%的最高性能。

**PatchCAM与所提出的类别特定Transformer注意力的互补性。** 表5显示，MCTformer-V2结合标准CAM模块生成的目标定位图mIoU为58.2%。通过使用图像块级成对亲和性进行优化，可以进一步提高到61.7%。如图5e所示，类别特定的Transformer注意力可以有效定位对象，但响应较低且有噪声。相比之下，PatchCAM图（图5f）在对象区域显示高响应，但也激活了对象周围更多的背景像素。这两者的融合产生了明显改进的定位图，只激活对象区域，显著减少了背景噪声（图5g）。这些类别特定的定位图证实了所提出的模型优于TS-CAM [14]（图5b）的显著优势，后者在大多数情况下显示稀疏且低的对象响应。

![](../../../../../../99_Assets%20(资源文件)/images/a1b9458ec7033ba89b837860bda7203f.png)

**图5**展示了不同方法生成的目标定位图可视化示例。(a) 输入图像。 (b) TS-CAM [14] 生成的定位图。(c) V1-attn（MCTformer-V1中类别特定的Transformer注意力）。 (d) V1-attn-refined（MCTformer-V1中经图像块亲和性优化的类别特定的Transformer注意力）。 (e) V2-attn（MCTformer-V2中类别特定的Transformer注意力）。 (f) V2-PatchCAM（MCTformer-V2的PatchCAM图）。(g) V2-fused（MCTformer-V2中类别特定的Transformer注意力与PatchCAM图的融合）。 (h) V2-fused-refined（MCTformer-V2中经图像块亲和性优化的融合图）。 (i) 真值（Ground-truth）。

观察图像可以发现，TS-CAM (b) 提供的定位图较为稀疏和不完整。V1-attn (c) 能够捕获到物体，但是激活区域可能不够集中或存在散点。经过图像块亲和性优化的V1-attn-refined (d) 看起来更平滑完整。V2-PatchCAM (f) 显示了更广的激活区域，但可能包含更多背景噪声。V2-fused (g) 通过结合V2-attn (e) 和V2-PatchCAM (f) 提高了定位准确性和减少了噪声。最终的V2-fused-refined (h) 在物体轮廓的完整性和平滑度上表现最佳，与Ground-truth (i) 最接近，这直观地展示了各组件的有效性及其组合的优越性。

**图像块亲和性的效果。** 如表5和表6所示，通过应用学习到的图像块到图像块注意力作为图像块级成对亲和性来优化MCTformer-V1生成的对象定位图，伪分割标签图可以提高8%，相应地，分割性能也提高了3.2%。MCTformer-V2在生成的伪标签质量和分割性能方面均取得了持续改进，优于不使用图像块亲和性的变体。图5（d）和（h）中的可视化结果显示，优化后的对象定位图看起来更完整，具有更平滑的对象轮廓。这进一步证明了作者的方法在无需额外计算的情况下生成有效图像块亲和性的巨大优势。

![](../../../../../../99_Assets%20(资源文件)/images/e0b5c331de710bce7af266331686ab40%201.png)

**表6**显示了使用不同目标定位图在PASCAL VOC验证集上的分割结果mIoU（%）。
这张表与表5的结果趋势一致，强调了图像块亲和性（PatchAffinity）和PatchCAM融合（MCTformer-V2）对最终分割性能的积极影响。

**不同类别预测方法。** 评估了用于生成类别特定的Transformer注意力图中类别分数的不同策略的效果。如表7所示，最大池化（max pooling）对于类别特定定位的性能最差，mIoU仅为26.8%，而使用全连接层进行线性投影可将mIoU提高到41.5%。平均池化（average pooling）表现最佳，mIoU为47.2%。这些结果证实了作者最初的设计动机。具体来说，全连接层中涉及额外的参数可能会增加学习判别性定位模型的复杂度。与只需关注最相关图像块的max pooling相比，平均池化可以鼓励类别令牌关注更多相关图像块，这有利于学习更好的空间上下文进行定位。

![](../../../../../../99_Assets%20(资源文件)/images/477a8dd9bfa6c0bd6b8029f27629e3d3%201.png)

**表7**比较了MCTformer-V1中用于类别预测的不同方法在生成的类别特定Transformer注意力上的mIoU（%）。
结果显示，平均池化（Average-pooling）取得了最好的性能（47.2%），这表明它在鼓励类别令牌关注更多相关图像块，从而学习更好的空间上下文方面具有优势。

**注意力融合的层数。** 评估了在所提出的MCTformer-V1中融合来自多个Transformer编码层中不同类别令牌注意力图的目标定位图的质量。遵循[39]，使用了三个评估指标：假阳性率（FP）、假阴性率（FN）和mIoU，其中较大的FP和FN值分别表示过度激活和欠激活区域的增加。

![](../../../../../../99_Assets%20(资源文件)/images/24c48a3f4581b417c6af8a49335d5622.png)

**图6**评估了通过融合来自最后K个Transformer层的类别令牌注意力生成的目标定位图的质量，使用假阳性率（FP）、假阴性率（FN）和mIoU进行衡量。
图中横轴表示融合的Transformer层数 $K$，纵轴表示性能百分比。
观察趋势，随着 $K$ 的增加（即融合更多层），FP（假阳性率）和FN（假阴性率）的变化。mIoU曲线显示在 $K=3$ 附近达到峰值（47.2%）。这表明融合最后3个Transformer层的注意力可以得到最佳的伪分割真值标签。

如图6所示，聚合来自更多层的注意力会生成倾向于过度激活的对象定位图。这表明早期层产生更多通用的低级表示，可能对高级语义定位帮助不大。通过减少层数，生成的对象定位图变得更具判别性，但代价是激活覆盖率降低。图6中报告的结果表明，融合最后三层的注意力可以产生最佳的伪分割真值标签（mIoU为47.2%）。

## 5. 结论 (Conclusions)

这篇论文提出了MCTformer，一个简单而有效的基于Transformer的框架，用于生成类别特定的目标定位图，并在WSSS上取得了SOTA结果。该研究表明，不同类别令牌的“类别到图像块注意力”可以发现类别特定的定位信息，而“图像块到图像块注意力”也可以学习有效的成对亲和性以优化定位图。此外，作者证明了所提出的框架可以无缝地补充CAM机制，从而为WSSS生成高质量的伪真值标签。未来的工作将把所提出的方法扩展到更多的下游任务，例如弱监督目标检测和实例分割。
