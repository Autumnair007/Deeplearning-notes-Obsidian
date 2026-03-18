---
type: paper-note
tags:
  - cv
  - semantic-segmentation
  - weakly-supervised
  - clip
  - diffusion-model
  - vit
status: done
model: ComCD (Complementary synergy of CLIP and Diffusion models)
year: 2026
---
论文网址：[Unveiling the complementary synergy of CLIP and diffusion models for weakly supervised semantic segmentation - ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0957417426007979?fr=RR-2&ref=pdf_download&rr=9d81f7f869955251)

本地PDF文件：ComCD [[../../../../../../99_Assets (资源文件)/papers/Unveiling the complementary synergy of CLIP and diffusion models for weakly supervised semantic segmentation.pdf]]
***
ComCD（Complementary synergy of CLIP and Diffusion models）这篇论文提出了一种结合CLIP和扩散模型在弱监督语义分割（WSSS）任务中提升CAM质量的方法。WSSS仅依赖图像级标签，通过生成类激活图（CAMs）作为像素级种子，并将其转换为伪标签进行分割。CLIP模型擅长类定位，而扩散模型擅长保持空间一致性。ComCD旨在利用这两种模型的互补优势。

### 摘要

弱监督语义分割（WSSS）仅依赖图像级标签，通过生成类激活图（CAMs）作为像素级种子并将其转化为伪标签进行分割。最近，一些方法利用对比语言-图像预训练（CLIP）或扩散模型来生成WSSS流程中的CAM。然而，如何将这两种范式整合到一个框架中仍未得到充分探索。本作提出了ComCD，它整合了这两种范式并利用它们的互补性来提高CAM质量。首先，从CLIP分支和扩散模型分支中推导出类特定的CAM。其次，设计了一种基于熵的融合方法，将两个CAM之间的熵差异映射为可靠性权重，将其融合为一个精化的CAM，并将其转换为伪掩码。第三，一个带有Logit Gating Module的可训练分割网络预测权重以融合两个分支并生成最终分割。实验结果表明，所提出的ComCD在WSSS和开放词汇语义分割方面均优于最新的SOTA方法。

### 1. 引言

弱监督语义分割（WSSS）旨在平衡标注成本和像素级预测。与需要像素级标注的全监督方法不同，WSSS使用更经济的监督（例如点标注、涂鸦、边界框和图像级标签）来训练密集分割模型，从而降低数据收集和整理的开销。在这些形式中，图像级标签仅指示类别的存在而没有空间定位，因此最具挑战性。本文采用图像级标签进行语义分割。

在图像级监督下，典型的多阶段流程首先训练一个图像分类器来生成类激活图（CAMs）。然后训练一个精化网络来进一步改善CAMs。最后，使用从CAMs导出的伪标签训练一个分割网络。为了减少流程开销，单阶段方法将这些步骤整合到一个模型中，该模型同时生成伪标签并学习像素级掩码。然而，由于监督不足，CAMs往往只关注判别性区域，导致覆盖不完整和噪声伪标签，最终降低WSSS性能。

最近，基于对比语言-图像预训练（CLIP）和扩散模型的两种范式在WSSS中获得了广泛关注。CLIP-ES是一种免训练、文本驱动的Grad-CAM，直接定位类判别区域，而ExCEL利用补丁-文本对齐来增强类定位和边界锐度。扩散模型的DiG结合预训练扩散嵌入以促进区域级连续性，iSeg则迭代地精化扩散模型的交叉注意力以产生更具空间一致性的掩码。

![](../../../../../../99_Assets%20(资源文件)/images/eaf270a51c05f69c8418bab1749c5878.png)

通过可视化CLIP和扩散模型生成的CAMs（图1(a)），我们观察到它们具有不同的行为：
*   **CLIP-based CAMs**: 强调类别定位，对类别判别性区域激活强烈。
*   **Diffusion-based CAMs**: 偏好空间一致性，显示属于同一语义区域的像素具有相似的激活。

图1(a)展示了CLIP-based CAMs和Diffusion-based CAMs的对比。CLIP-based CAMs突出类别定位（\*），而Diffusion-based CAMs倾向于空间一致性（▴）。

图1(b)对比了CLIP-based和Diffusion-based标签的准确性。准确性随到GT边界的内部距离（像素）而变化。
实验结果（图1(b)）显示，靠近GT边界100像素内，CLIP方法更准确；超过100像素，扩散模型方法更优。这直观地表明，可以在边界处使用CLIP标签，内部使用扩散模型标签。然而，由于物体形状不规则且多尺度，使用硬性距离阈值作为融合标准并不可靠。因此，我们用基于不确定性的标准取代距离阈值，通过计算每个像素类别概率分布的香农熵来量化不确定性。熵值越低表示置信度越高，像素级分割精度也越高。因此，在每个像素处，我们比较来自CLIP和扩散模型CAMs的熵值，并赋予较低熵的CAM更高的权重，从而避免了基于距离阈值的脆弱性，产生了更可靠的融合。

本文提出了ComCD (Complementary synergy of CLIP and Diffusion models)。ComCD在一个双分支WSSS流程（CLIP-based CAMs和diffusion-based CAMs）中采用了简单有效的基于熵的融合（EBF）策略。对于每个像素，ComCD计算每个CAM的熵，并将熵差异映射为逐像素的置信权重，指导融合。将此权重图应用于融合两个CAM，生成一个精化的CAM，然后转换为伪掩码。此外，ComCD引入了Feature Aligned Decoder (FAD)，这是一个在伪掩码监督下训练的分割网络。具体来说，来自CLIP和扩散模型的图像嵌入通过Feature Aligner进行对齐，使其具有共同的空间分辨率和通道宽度。对齐后的嵌入随后输入解码器，生成两个分支的分割预测。Logit Gating Module (LGM)以这两个预测为输入，预测逐像素权重，并将它们组合成融合预测。FAD在伪掩码监督下进行训练，以生成最终的分割掩码。实验评估了所提出的方法在WSSS和开放词汇语义分割设置下的性能。

### 主要贡献：
*   提出了一种简单有效的**基于熵的融合**（Entropy-Based Fusion），将CLIP和扩散模型CAMs之间的逐像素熵差异转换为可靠性权重，从而生成精炼的CAM和伪掩码。
*   开发了一个**Feature Aligned Decoder**，包含一个Feature Aligner和一个Logit Gating Module，并在伪掩码的监督下进行训练。
*   大量实验证明在WSSS基准测试（PASCAL VOC 2012，MS COCO 2014）上具有竞争性能，并在PASCAL-Context，MS COCO-Object和PASCAL VOC 2012上取得了强大的开放词汇分割结果。

### 2. 相关工作

#### 弱监督语义分割 (WSSS)
图像级标签的WSSS通常依赖于类激活图 (CAMs) 为学习分割模型提供密集监督。一个关键限制是CAMs通常只强调高度判别性区域，导致对象覆盖不完整，进一步放大了伪掩码中的噪声和共现偏差。为解决此问题，主流方法采用多阶段精化，通过亲和传播、正则化和迭代自训练逐步扩展和去噪初始线索。为降低此类流程的复杂性，最近的努力日益转向单阶段WSSS，其更紧密地耦合CAM生成和在线精化，并直接使用精化后的伪标签训练分割头，从而减少重复训练和提高效率。

最近，基础模型的先验知识进一步重塑了WSSS：CLIP带来了强大的类感知语义，而扩散模型提供了结构和形状一致的线索。现有研究大多独立利用这些先验知识。相比之下，本作首次将扩散模型导出的结构先验和CLIP导出的类感知先验统一并联合利用到单个WSSS框架中。

#### CLIP 在 WSSS 中的应用
CLIP通过大规模的对比预训练对图像和文本进行对齐，提供了类感知语义，可在仅有图像级标签的WSSS中指导伪标签生成。这促使了一系列基于CLIP的WSSS方法，提高了定位质量并减少了背景混淆。CLIP-ES将冻结的CLIP转换为一个基本免训练的分割器，通过用基于概率的Grad-CAM替换基于logit的Grad-CAM，并使用CLIP的ViT注意力精化掩码。ExCEL从全局图像-文本对齐转向密集补丁-文本对齐，并通过文本语义丰富和视觉校准来挖掘CLIP的细粒度先验。然而，基于CLIP的流程在非判别性区域表现不佳，而扩散模型则提供空间一致性。因此，ComCD结合了CLIP的类定位和扩散模型导出的结构先验，通过自适应融合。

#### 扩散模型 在 WSSS 中的应用
扩散模型学习从噪声到图像的去噪过程，并暴露出编码对象形状和语义的丰富注意力。在WSSS中，这些注意力可以转换为CAMs或用于仅使用图像级标签精化伪标签，这激发了几种基于扩散模型的流程。DiG提出了局部性融合交叉注意力，将预训练的扩散模型嵌入与基于ViT的WSSS模型融合。SeeDiff使用交叉注意力作为种子，并通过多尺度自注意力扩展它们，无需额外训练即可生成高质量掩码。DiffSegmenter表明自注意力捕捉对象形状，而交叉注意力指示语义。iSeg引入了迭代精化和熵减自注意力模块，逐步改善交叉注意力图，实现免训练分割。相反，纯扩散模型流程倾向于空间一致性，但类别特异性有限。ComCD通过CLIP的类别定位补充纯扩散模型流程，并使用基于熵的加权融合这两个分支。

### 3. 方法

ComCD利用预训练的CLIP和文本条件扩散模型获取CAMs，将它们融合为伪掩码，然后训练一个解码器在这些伪掩码的监督下预测最终的分割掩码。

### 3.1. 预备知识

#### 3.1.1. 基于扩散模型的类激活图

ComCD使用一个预训练并冻结的文本条件扩散模型（Stable Diffusion）。给定一张图像及其类别提示，ComCD将图像编码到潜在空间，在固定时间步添加高斯噪声，并执行一个单次反向去噪步骤，使用固定的采样配置来提取自注意力（self-attention）和交叉注意力（cross-attention）图。这些扩散超参数遵循iSeg的公开配置，并在所有实验中保持固定。由于扩散骨干是冻结的，且CAM生成过程中时间步和采样配置是固定的，因此提取的注意力图和由此产生的基于扩散模型的CAM对于给定输入图像是确定性的。

**交叉注意力图 (Cross-attention map)**：
在预训练并冻结的文本条件扩散模型中，给定图像和对应的类别提示，在时间步$t$执行单次反向去噪步骤，得到图像嵌入$\mathbf{E}_{\text{img}} \in \mathbb{R}^{HW \times C}$，其中$H, W$是空间高度和宽度，$C$是通道维度。提示被编码为文本嵌入$\mathbf{E}_{\text{txt}} \in \mathbb{R}^{L \times d}$，其中$L$是文本标记的数量，$d$是标记嵌入维度。学习的投影$\mathbf{W}_{\text{ca}}^q, \mathbf{W}_{\text{ca}}^k, \mathbf{W}_{\text{ca}}^v$生成查询（queries）、键（keys）和值（values）。查询来自图像嵌入$\mathbf{Q}_{\text{ca}} = \mathbf{E}_{\text{img}}\mathbf{W}_{\text{ca}}^q \in \mathbb{R}^{HW \times d}$。键和值来自文本$\mathbf{K}_{\text{ca}} = \mathbf{E}_{\text{txt}}\mathbf{W}_{\text{ca}}^k \in \mathbb{R}^{L \times d}$和$\mathbf{V}_{\text{ca}} = \mathbf{E}_{\text{txt}}\mathbf{W}_{\text{ca}}^v \in \mathbb{R}^{L \times d}$。交叉注意力图计算如下：

$$
\mathbf{A}_{\text{ca}} = \text{Softmax}\left(\frac{\mathbf{Q}_{\text{ca}}\mathbf{K}_{\text{ca}}^\top}{\sqrt{d}}\right) \in \mathbb{R}^{HW \times L}
$$

遵循iSeg的方法，我们从多个空间尺度收集交叉注意力图，沿标记维度应用softmax，以固定权重组合它们，并双线性上采样到共同的低分辨率网格；所有后续的精化步骤都在这个聚合的交叉注意力图上操作。对于每个语义类别，我们预计算其类别名称对应的标记索引，对于类别$c$，平均选定列的聚合交叉注意力并将其重塑为类别感知的空间图$\mathbf{a}_{\text{ca}}^c \in \mathbb{R}^{H_{\text{low}} \times W_{\text{low}}}$。

**自注意力图 (Self-attention map)**：
对于图像嵌入的自注意力，学习的投影$\mathbf{W}_{\text{sa}}^q, \mathbf{W}_{\text{sa}}^k, \mathbf{W}_{\text{sa}}^v$将$\mathbf{E}_{\text{img}}$映射到查询、键和值：$\mathbf{Q}_{\text{sa}} = \mathbf{E}_{\text{img}}\mathbf{W}_{\text{sa}}^q$, $\mathbf{K}_{\text{sa}} = \mathbf{E}_{\text{img}}\mathbf{W}_{\text{sa}}^k$, $\mathbf{V}_{\text{sa}} = \mathbf{E}_{\text{img}}\mathbf{W}_{\text{sa}}^v$，其中$\mathbf{Q}_{\text{sa}}, \mathbf{K}_{\text{sa}}, \mathbf{V}_{\text{sa}} \in \mathbb{R}^{HW \times d}$。自注意力图计算如下：

$$
\mathbf{A}_{\text{sa}} = \text{Softmax}\left(\frac{\mathbf{Q}_{\text{sa}}\mathbf{K}_{\text{sa}}^\top}{\sqrt{d}}\right) \in \mathbb{R}^{HW \times HW}
$$

实际上，我们采用$H_{\text{low}} \times W_{\text{low}}$的潜在分辨率，使得$\mathbf{A}_{\text{sa}}$的空间索引与$\mathbf{a}_{\text{ca}}^c$对齐；这种注意力捕捉长距离空间亲和性并有利于区域连通性。

**自注意力的迭代精化 (Iterative refinement with self-attention)**：
对于类别标记$c$，上述聚合的交叉注意力图提供了最初的类别特定图$\mathbf{a}_{\text{ca}}^c$，它突出了64×64网格上与类别$c$最相关的空间区域。遵循iSeg的方法，我们随后使用自注意力作为空间亲和力在一个传播步骤中精化此图。具体地，$\mathbf{a}_{\text{ca}}^c$被矢量化为$\mathbf{m}_c \in \mathbb{R}^{H_{\text{low}}W_{\text{low}}}$，并且相同分辨率的自注意力被重塑为空间-空间亲和矩阵$\mathbf{A}_{\text{sa}} \in \mathbb{R}^{H_{\text{low}}W_{\text{low}} \times H_{\text{low}}W_{\text{low}}}$。然后我们计算：

$$
\tilde{\mathbf{m}}_c = \mathbf{A}_{\text{sa}}\mathbf{m}_c
$$

这会将类别分数沿高亲和度的空间邻居扩散。最后，$\tilde{\mathbf{m}}_c$被重塑回$\mathbb{R}^{H_{\text{low}} \times W_{\text{low}}}$，双线性上采样到$(H, W)$，并归一化以获得精化后的基于扩散模型的CAM $\mathcal{H}_{\text{diff}}^c \in \mathbb{R}^{H \times W}$。这一精化步骤提高了空间一致性并抑制了虚假隔离响应，而无需引入额外的可训练参数。

#### 3.1.2. 基于 CLIP 的类激活图

CLIP通过大规模对比预训练对图像和文本进行对齐。它由一个图像编码器（通常是Vision Transformer）和一个文本编码器组成，两者都将输入投射到一个共享的嵌入空间中，其中余弦相似度反映了语义对齐。

输入图像被送入CLIP图像编码器，从选定层获取补丁嵌入$\mathbf{E}_{\text{img}} \in \mathbb{R}^{HW \times D}$。线性投影$\mathbf{W}_Q, \mathbf{W}_K, \mathbf{W}_V$生成$\mathbf{Q}=\mathbf{E}_{\text{img}}\mathbf{W}_Q$, $\mathbf{K}=\mathbf{E}_{\text{img}}\mathbf{W}_K$, $\mathbf{V}=\mathbf{E}_{\text{img}}\mathbf{W}_V$。为简化符号，我们省略注意力头维度并写作$\mathbf{Q}, \mathbf{K}, \mathbf{V} \in \mathbb{R}^{HW \times D}$，其中$D$是投影后的通道宽度。

与标准自注意力不同（其中$\mathbf{Q}$和$\mathbf{K}$来自不同的投影$\mathbf{W}_Q$和$\mathbf{W}_K$），这里$\mathbf{Q}$被同时用于查询和键，即$\mathbf{Q}_{\text{sa}}=\mathbf{K}_{\text{sa}}=\mathbf{Q}$，注意力形式为$\mathbf{A}_{QQ} = \text{Softmax}\left(\mathbf{QQ}^\top/\sqrt{D}\right) \in \mathbb{R}^{HW \times HW}$。类似地，替换$(\mathbf{Q}_{\text{sa}}, \mathbf{K}_{\text{sa}})$为$(\mathbf{K}, \mathbf{K})$和$(\mathbf{V}, \mathbf{V})$可得到$\mathbf{A}_{KK}$和$\mathbf{A}_{VV}$。ComCD将空间亲和力定义为$\mathcal{A}=\frac{1}{3}(\mathbf{A}_{QQ}+\mathbf{A}_{KK}+\mathbf{A}_{VV})$。将$\mathcal{A}$应用于值$\mathbf{V}$得到$\tilde{\mathbf{E}}_{\text{img}}=\mathcal{A}\mathbf{V}$。ComCD随后将$\tilde{\mathbf{E}}_{\text{img}}$重塑为$\hat{\mathbf{E}}_{\text{img}} \in \mathbb{R}^{H \times W \times D}$以获得逐像素嵌入。最后，给定类别$c$的文本嵌入$\mathbf{E}_{\text{txt}}^c \in \mathbb{R}^D$，CAM通过余弦相似度计算：

$$
\mathcal{H}_{\text{clip}}^c = \cos\left(\frac{\hat{\mathbf{E}}_{\text{img}}}{\|\hat{\mathbf{E}}_{\text{img}}\|_2}, \frac{\mathbf{E}_{\text{txt}}^c}{\|\mathbf{E}_{\text{txt}}^c\|_2}\right) \in \mathbb{R}^{H \times W}
$$

其中$\cos(\cdot,\cdot)$表示余弦相似度。

#### 3.1.3. 基于扩散模型的 CAM 与 基于 CLIP 的 CAM

CLIP-based CAMs 通常通过将图像-文本相似度分数归因于空间标记来获得，而扩散模型 CAMs 通常从文本条件扩散模型的 U-Net 注意力中提取。

**图1(a)中的CAM可视化**：
*   **CLIP-based CAMs**: 能够清晰地识别和突出图像中对象的类判别性区域（例如，人脸、汽车的特定部分）。它们在对象边界处响应更锐利，但可能在非判别性区域（如大面积的衣服、车辆的均匀表面）显示稀疏激活或覆盖不完整。
*   **Diffusion-based CAMs**: 呈现出更平滑、空间上更连贯的激活。它们在整个对象区域提供更完整的覆盖，更好地保留了对象结构和区域完整性。然而，它们可能在对象边界处不如CLIP-based CAMs精确，容易出现模糊或扩张。

这些观察证实了它们在类定位和空间一致性方面的互补优势。CLIP更侧重于**类定位**，而扩散模型更侧重于**空间一致性**。

### 3.2. 基于熵的融合

ComCD基于 CLIP-based CAMs $\{\mathcal{H}_{\text{clip}}^c\}_{c \in \mathcal{C}_{\text{img}}}$ 和 Diffusion-based CAMs $\{\mathcal{H}_{\text{diff}}^c\}_{c \in \mathcal{C}_{\text{img}}}$ 构建基于熵的融合（Entropy-Based Fusion）。如图1(b)所示，CLIP-based CAMs倾向于在对象边界附近更精确，而 Diffusion-based CAMs 在对象内部更具空间一致性；因此，EBF 被设计为逐像素规则，以决定在每个位置哪个分支更可靠，并相应地加权它们的贡献。

给定输入图像，令$\mathcal{C}_{\text{img}}$表示图像中存在的类别集合（$|\mathcal{C}_{\text{img}}|$是类别数量）。堆叠每个类别的CAM，得到每个分支$b \in \{\text{clip}, \text{diff}\}$的$\mathcal{H}_b \in \mathbb{R}^{|\mathcal{C}_{\text{img}}| \times H \times W}$。第一步，每个分支通过沿类别维度应用softmax转换为像素级类别分布：

$$
\mathcal{P}_b^c = \frac{\exp(\mathcal{H}_b^c)}{\sum_{c' \in \mathcal{C}_{\text{img}}} \exp(\mathcal{H}_b^{c'})}
$$

在每个空间位置$(h, w)$，熵被定义为由分支$b \in \{\text{clip}, \text{diff}\}$诱导的逐像素类别概率分布$\{\mathcal{P}_b^{c,h,w}\}_{c \in \mathcal{C}_{\text{img}}}$的香农熵：

$$
e_b^{h,w} = - \sum_{c \in \mathcal{C}_{\text{img}}} \mathcal{P}_b^{c,h,w} \log \mathcal{P}_b^{c,h,w}
$$

其中$\sum_{c \in \mathcal{C}_{\text{img}}} \mathcal{P}_b^{c,h,w} = 1$对每个$(h, w)$成立。预测熵是基于softmax分类器的标准不确定性度量：对于固定的标签集，较低的熵对应于更尖锐的分布，其中一个类别主导其他类别。实际上，这种低熵预测在分类和分割任务中都与更高的准确性经验相关。在ComCD中，我们以相对的方式利用这一特性：当在同一像素上比较CLIP和扩散分支时，熵较低的分支被视为更可靠。这一设计与图1(b)中的边界-内部曲线一致，其中CLIP分支在对象边界附近更自信也更准确，而扩散分支在对象内部更自信也更准确。与最大概率、边距或方差等替代置信度代理相比，熵的实际优势在于它直接从单个逐像素softmax分布计算，并提供分布锐度的单调标量摘要，无论类别数量如何。在我们的伪掩码生成阶段，由于冻结了CLIP和扩散骨干网络，这使得熵成为一个特别方便和轻量级的选择。

基于这种相对置信度视图，计算差异$\Delta e_{h,w} = e_{\text{clip}}^{h,w} - e_{\text{diff}}^{h,w}$作为逐像素可靠性标准。较低的熵表示较高的置信度。因此，$\Delta e_{h,w} < 0$表示在$(h, w)$处对CLIP赋予更大的置信度，而$\Delta e_{h,w} > 0$则偏向扩散模型分支。将$\Delta e$通过sigmoid函数得到逐像素权重图$\mathbf{W} = \sigma(\Delta e)$，其中$\sigma(\cdot)$表示sigmoid函数。融合后的CAM通过元素乘法定义为：

$$
\mathcal{H}_{\text{fuse}}^c = \mathbf{W} \odot \mathcal{P}_{\text{diff}}^c + (\mathbf{1} - \mathbf{W}) \odot \mathcal{P}_{\text{clip}}^c
$$

其中$\odot$表示元素乘法，$\mathbf{1} \in \mathbb{R}^{H \times W}$是全1图（即每个条目等于1）。此表达式给出类别$c$在所有像素上的融合图。

Image source: Fig. 2.

**图2：ComCD 概述。**
给定图像和类别提示，CLIP生成CLIP-based CAMs，而Stable Diffusion模型生成Diffusion-based CAMs。基于熵的融合（Entropy-Based Fusion）计算逐像素权重，形成融合CAM，并将其转换为伪掩码。伪掩码监督Feature Aligned Decoder (FAD)。FAD使用预训练骨干提取的特征，通过Feature Aligner将两个分支对齐到共同的空间分辨率和通道宽度，使用共享解码器预测两个分支的逐像素对数，并应用Logit Gating Module通过学习的逐像素权重生成最终的融合预测。

最终的掩码$\mathcal{M}$通过像素级argmax在图像特定类别集$\mathcal{C}_{\text{img}}$上获得：

$$
\mathcal{M} = \arg \max_{c \in \mathcal{C}_{\text{img}}} \mathcal{H}_{\text{fuse}}^c
$$

除了直接作为伪掩码进行评估外，最终掩码$\mathcal{M}$还可以作为监督来训练分割网络。

### 3.3. 特征对齐解码器 (Feature Aligned Decoder)

为了进一步探索 CLIP-diffusion 融合的潜力，我们训练了一个分割网络，该网络在第3.2节中通过基于熵的 CAM 融合获得的伪掩码 $\mathcal{M}$ 的监督下进行训练。对于每个分支，从其各自模型的不同块中收集嵌入，并通过 Feature Aligner (FA) 模块进行对齐，以便可以共享单个解码器。

令 $\{ \mathbf{F}_b^i \}_{i=1}^{T_b}$ 表示从分支 $b \in \{\text{clip}, \text{diff}\}$ 的不同块中收集的嵌入。这里 $T_b$ 是从分支 $b$ 中选择的块的数量。FA 将每个 $\mathbf{F}_b^i$ 上采样到 $(H_d, W_d)$ 空间分辨率，将通道投射到共享宽度 $D$，沿通道连接它们，并应用 1x1 卷积以形成解码器输入：

$$
\mathbf{X}_b = \text{Conv}_{1 \times 1}\left(\text{Cat}_{i=1}^{T_b}\left(\text{Up}(\mathbf{F}_b^i)\right)\right) \in \mathbb{R}^{B \times D \times H_d \times W_d}
$$

其中 $\text{Up}(\cdot)$ 将每个嵌入上采样到 $(H_d, W_d)$，$\text{Cat}(\cdot)$ 表示通道级联，$\text{Conv}_{1 \times 1}(\cdot)$ 是一个 1x1 卷积，将连接的嵌入投射到 $D$ 个通道。然后，共享解码器以 $\mathbf{X}_b$ 作为输入，并为每个分支 $b \in \{\text{clip}, \text{diff}\}$ 生成逐像素类别预测 $\mathbf{S}_b \in \mathbb{R}^{B \times |\mathcal{C}| \times H_d \times W_d}$。这里 $\mathcal{C}$ 表示数据集中所有类别的集合（$|\mathcal{C}|$ 是类别数量）。

随后，Logit Gating Module (LGM) 从预测的 logits $\mathbf{S}_b$ 中预测逐像素权重 $\mathbf{W}'$:

$$
\mathbf{W}' = \sigma\left(\text{Conv}_{1 \times 1}\left(\text{Cat}(\mathbf{S}_{\text{clip}}, \mathbf{S}_{\text{diff}})\right)\right) \in \mathbb{R}^{B \times 1 \times H_d \times W_d}
$$

其中 $\text{Conv}_{1 \times 1}(\cdot)$ 是一个 1x1 卷积，将 $2|\mathcal{C}|$ 个通道映射到一个通道，$\sigma$ 是 sigmoid 函数。最后，与 Eq. (7) 中的像素级融合规则一致，融合 logits 定义为：

$$
\mathbf{S}_{\text{fuse}} = \mathbf{W}' \odot \mathbf{S}_{\text{diff}} + (\mathbf{1} - \mathbf{W}') \odot \mathbf{S}_{\text{clip}}
$$

其中 $\odot$ 表示元素乘法，$\mathbf{1} \in \mathbb{R}^{H \times W}$ 是全一的图。

### 3.4. 训练目标

融合的 logits 通过使用逐像素交叉熵损失函数，在伪掩码 $\mathcal{M}$ 的监督下进行训练：

$$
\mathcal{L}_{\text{fuse}} = \text{CE}(\mathbf{S}_{\text{fuse}}, \mathcal{M})
$$

其中 $\text{CE}(\cdot,\cdot)$ 表示标准的逐像素交叉熵。同样，ComCD 使用相同的交叉熵损失来监督每个分支：

$$
\mathcal{L}_{\text{clip}} = \text{CE}(\mathbf{S}_{\text{clip}}, \mathcal{M}), \quad \mathcal{L}_{\text{diff}} = \text{CE}(\mathbf{S}_{\text{diff}}, \mathcal{M})
$$

此外，遵循 ExCEL 的方法，ComCD 采用多样性损失 $\mathcal{L}_{\text{div}}$，以避免冗余预测，并鼓励两个分支之间提供互补的特征。总的训练目标是：

$$
\mathcal{L}_{\text{total}} = \lambda_1 \mathcal{L}_{\text{fuse}} + \lambda_2 \mathcal{L}_{\text{clip}} + \lambda_3 \mathcal{L}_{\text{diff}} + \lambda_4 \mathcal{L}_{\text{div}}
$$

其中 $\lambda_1, \lambda_2, \lambda_3, \lambda_4 \ge 0$ 是重新缩放每个损失项贡献的权重。

### 4. 实验

### 4.1. 实验设置

**数据集和评估指标。** ComCD 在四个数据集上进行评估：PASCAL VOC 2012、MS COCO 2014、PASCAL-Context 和 MS COCO-Object。
*   **PASCAL VOC 2012**: 包含21个语义类别，使用扩展版本（SBD数据集），包括10,582张训练图像，1,449张验证图像和1,456张测试图像。
*   **MS COCO 2014**: 包含81个类别，分为82,081张训练图像和40,137张验证图像。
*   **PASCAL-Context**: 包含60个类别，数据集包括4,998张训练图像和5,105张验证图像。
*   **MS COCO-Object**: 包含81个类别，在COCO 2014验证集的4,952张验证图像上进行评估。
主要评估指标是**平均交并比（mIoU）**。

**实现细节。** ComCD 使用带有 ViT-B/16 图像编码器的 CLIP 和预训练的 Stable Diffusion v2.1。两个骨干网络在整个训练过程中都保持冻结。
*   **CLIP 方面**：遵循 ExCEL，从 ViT-B/16 的所有12个变换器层获取补丁-标记特征，将其投影到256维度，并与 SegFormer 风格的头部融合到一个统一的256通道特征图中。类别名称使用固定提示模板“a clean origami {}”进行渲染，然后传递给文本编码器。
*   **扩散模型方面**：遵循 iSeg，扩散调度器设置为1000个去噪步骤，在训练期间均匀采样一个时间步$t$添加高斯噪声，在推理时固定 $t=100$，并执行单个反向去噪步骤以提取交叉注意力和自注意力图。这些图被聚合到共享的64x64潜在网格（即 $H_{\text{low}}=W_{\text{low}}=64$）用于 CAM 生成，自注意力精化迭代配置与 Sun et al. (2024a) 相同。
两个分支被对齐到共同的256通道、共享分辨率的特征空间，然后输入轻量级基于 Transformer 的解码器头部。遵循以往工作，AdamW 优化所有可训练组件，学习率为 $1 \times 10^{-4}$，权重衰减为 $1 \times 10^{-2}$。损失权重设置为 $\lambda_1 = 1$, $\lambda_2 = \lambda_3 = \lambda_4 = 0.1$。训练在 PASCAL VOC 2012 上进行30,000次迭代，在 MS COCO 2014 上进行100,000次迭代。在推理阶段，ComCD 采用 WSSS 中使用的标准方法，通过多尺度测试和密集 CRF 进行后处理。

**评估协议。**
*   **WSSS (PASCAL VOC 2012 和 MS COCO 2014)**：首先使用冻结的 CLIP 和扩散模型骨干评估 ComCD。在训练集上生成 CLIP-based 和 diffusion-based CAMs，通过基于熵的融合将其融合以获得伪掩码，并计算在训练集上相对于 GT 掩码的 mIoU。此步骤不训练任何额外参数，结果报告在表1中。第二步，这些伪掩码用于监督单阶段 WSSS 框架的分割解码器，该解码器在训练集上训练，并在验证集上评估；相应的 mIoU 总结在表2中，并与最近的有训练需求的 WSSS 方法在相同分割和指标下进行比较。
*   **开放词汇语义分割 (OVSS)**：ComCD 同样采用冻结骨干在 PASCAL VOC 2012、PASCAL-Context 和 MS COCO-Object 的验证集上进行伪掩码评估：冻结的 CLIP 和扩散模型骨干加上基于熵的融合直接生成掩码，并在验证集上计算 mIoU，如表3所示。使用预训练的 CLIP 模型，TagCLIP 生成的图像级类别标签作为文本提示。除非另有说明，所有其他设置与 WSSS 相同。

### 4.2. 实验结果

#### 融合的 CAM 可视化
**图3** 展示了 ComCD 在 PASCAL VOC 2012 上的结果与代表性基线的对比。
*   **图3(b) CLIP-based CAMs**: 提供类别的先验知识进行定位，勾勒出边界并突出判别区域。
*   **图3(c) Diffusion-based CAMs**: 贡献结构先验，促进区域连续性并产生更完整的激活。
*   融合的 CAM: 基于熵的逐像素权重通过降低不太可靠分支的权重，实现了错误激活的相互校正。
*   **图3(d) ComCD**: 基于熵的融合将这些优势整合到单个图中，该图通过抑制边缘附近的背景泄漏和弥补内部间隙，与图3(i)的真实标注更紧密地对齐。

与 CLIP-ES (图3(e)), DuPL (图3(f)), SeCo (图3(g)), 和 WeCLIP (图3(h)) 相比，ComCD 在对象边界和非判别区域产生了更清晰的响应。

#### WSSS 伪掩码评估
**表1** 报告了在 PASCAL VOC 2012 训练集和 MS COCO 2014 训练集上生成的伪掩码的 mIoU。
*   ComCD (使用 ViT-B/16) 在 VOC 上达到 **82.1%**，在 MS COCO 2014 上达到 **51.3%**。
*   与 PRCE 相比，在 VOC 上提高了 +4.5% (82.1% vs 77.6%)。
*   与 IRN 相比，在 MS COCO 2014 上提高了 +8.9% (51.3% vs 42.4%)。
*   相较于 iSeg，在 VOC 和 MS COCO 2014 上分别提高了 +6.9% 和 +5.8%。
*   相较于 T2M，分别提高了 +9.4% 和 +7.6%。
*   相较于 CLIP-ES，分别提高了 +11.3% 和 +11.6%。
*   相较于 ExCEL，分别提高了 +7.5% 和 +7.7%。
*   相较于 DiffSegmenter，在 VOC 上提高了 +11.6%。

**图4** 提供了与 ExCEL、iSeg 和 CLIP-ES 的定性比较。ComCD 生成的伪掩码具有更完整的对象覆盖和更锐利的边界，同时减少了两个数据集上的假阳性、背景泄漏和碎片区域。
**图6** 中的每类别 mIoU 雷达图进一步显示，ComCD 在所有类别上都优于 CLIP-ES、ExCEL 和 iSeg。

#### WSSS 最终分割结果
**表2** 总结了在 PASCAL VOC 2012 验证集和 MS COCO 2014 验证集上多阶段和单阶段有训练需求的 WSSS 方法的分割 mIoU。
*   ComCD (使用 $\mathcal{I}+\mathcal{T}$ 监督和 ViT-B/16 骨干) 在 VOC 上达到 **79.5%**，在 MS COCO 2014 上达到 **52.1%**。
*   在单阶段方法中，它超越了 ExCEL (+2.3% 和 +2.8%)。
*   相对于领先的多阶段方法，它高于 CPAL (在 VOC 上 +5.0%) 和 PSDPM (在 MS COCO 2014 上 +4.9%)。
*   相对于其他代表性单阶段方法，在 VOC 上绝对改进范围为 +1.1% 到 +13.5%，在 MS COCO 2014 上为 +1.8% 到 +13.2%。

**图5** 提供了与 ExCEL、DuPL 和 ToCo 的定性比较，视觉结果与表2中报告的定量增益一致。

#### OVSS 伪掩码评估
**表3** 报告了在 PASCAL VOC 2012、PASCAL-Context 和 MS COCO-Object 验证集上生成的伪掩码的 mIoU。
*   ComCD (使用 ViT-B/16) 在 VOC 上达到 **74.2%**，在 Context 上达到 **54.8%**，在 MS COCO-Object 上达到 **39.3%**，在所有三个基准测试中均排名第一。
*   与 LPOSS 相比，分别提高了 +11.8%、+20.5% 和 +3.9%。
*   与 iSeg 相比，分别提高了 +6.0%、+23.9% 和 +0.9%。
*   与 CASS 相比，分别提高了 +8.4%、+18.1% 和 +1.5%。
PASCAL-Context 上的改进尤为显著，VOC 表现出明显的优势，而 COCO-Object 则显示出较小但一致的优势。
**图7** 展示了与 DiffSegmenter 和 iSeg 的伪掩码定性比较。

#### 与全监督方法的比较
**表4** 比较了 PASCAL VOC 2012 验证集上弱监督方法与全监督方法。
*   ComCD 的有训练需求方法 (使用 ViT-B\*) 达到 **79.5%** mIoU，相当于相同骨干网络的完全监督结果（WeCLIP-Full, 81.6%）的 **97.4%**。
*   在单阶段 WSSS 中，它高于 ExCEL (77.2%，94.6% 比例) 和 WeCLIP (76.4%，93.6% 比例)。
*   相对于 ResNet101 上的强大多阶段方法，它高于 CPAL (74.5%，95.9% 比例)。
与 ViT-B\* 上完全监督上限的剩余差距为 2.1%，表明 ComCD 在使用弱监督的情况下弥合了大部分差距。

### 4.3. 消融研究与分析

#### 模块有效性
在我们的 WSSS 框架中，**表5** 评估了所提出的特征对齐解码器中使用的损失函数。
*   仅监督融合 logits $\mathcal{L}_{\text{fuse}}$ 得到 **74.1%** 的分数。
*   添加 CLIP 分支监督 $\mathcal{L}_{\text{clip}}$ 将分数提高到 **77.4%** (+3.3%)。
*   监督扩散分支 $\mathcal{L}_{\text{diff}}$ 得到 **76.5%** (+2.4%)。
*   在 $\mathcal{L}_{\text{fuse}}$ 旁边引入多样性损失 $\mathcal{L}_{\text{div}}$ 达到 **76.0%** (+1.9%)。
*   将 $\mathcal{L}_{\text{clip}}$ 与 $\mathcal{L}_{\text{div}}$ 结合达到 **78.5%**，这比单独使用 $\mathcal{L}_{\text{clip}}$ 高出 +1.1%。
*   将 $\mathcal{L}_{\text{diff}}$ 与 $\mathcal{L}_{\text{div}}$ 结合达到 **77.4%**，这比单独使用 $\mathcal{L}_{\text{diff}}$ 高出 +0.9%。
*   所有组件一起使用时，性能最佳达 **79.5%**，超过最强的两种损失配置 $\mathcal{L}_{\text{clip}}+\mathcal{L}_{\text{div}}$ 1.0%。
这些结果表明，监督两个分支可以改善融合 logits，并且多样性损失持续鼓励 CLIP 和扩散分支之间的互补性。

#### 基于熵的权重分析
**图8** 可视化了 Eq. (7) 中结合 CLIP 及其 CAMs 的逐像素权重 $\mathbf{W}$。我们分别报告了前景 (FG) 和背景 (BG) 区域的权重。
*   白色像素表示选择置信度更高的分支（CLIP 或扩散）。
*   较低的熵意味着较高的置信度，因此对融合 CAM 在该位置的贡献更大。
*   较暗的像素表示相反。
可视化的权重与两个分支的预期作用一致。权重图倾向于在需要类别判别性定位时偏向 CLIP，而在空间一致性有益时偏向扩散。这一观察与先前的发现一致，即 CLIP 方法强调类别定位，而扩散方法增强空间一致性。融合分割掩码中观察到的改进进一步支持了基于熵加权的有效性。

#### 融合策略分析
**表6** 比较了训练期间的两种融合方案：第3.2节中基于熵的融合 (EBF) 和 Logit Gating Module (LGM)。
*   **EBF**: 首先将 CLIP 和扩散 CAMs 转换为逐像素类别分布，然后通过 sigmoid 将两个分支之间的熵差异映射为标量权重，使得熵较低（置信度较高）的分支在融合预测中获得更大的贡献。
*   **LGM**: 通过一个应用于连接 logits $[\mathbf{S}_{\text{clip}}, \mathbf{S}_{\text{diff}}]$ 的 1x1 卷积，接着是一个 sigmoid，来预测逐像素权重图。
分割分数非常接近 (EBF 79.1%, LGM 79.5%)，我们注意到这些结果是通过多次独立运行（三次不同随机种子运行）平均得到的。这表明 LGM 学习到的融合规则与 EBF 中使用的基于不确定性的加权基本一致，而不是利用两个分支之间完全不同的偏好模式；LGM 的微小但一致的增益表明，轻量级的学习门控可以在模糊区域稍微校准基于熵的规则，而 EBF 本身仍然是一个强大且无参数的基线融合机制。

#### 融合策略的影响
为了进一步量化在伪标签生成阶段基于熵的融合 (EBF) 的重要性，我们将其与四种更简单的融合方案在 PASCAL VOC 2012 上进行了比较：(1) 仅使用 CLIP 的 CAMs 作为伪标签，(2) 仅使用扩散模型的 CAMs，(3) 平均两个分支的逐像素类别概率（等权平均），和 (4) 取两个分支概率的逐元素最大值（最大融合）。对于每种方案，我们首先评估训练集上的伪掩码，然后使用这些掩码训练 ComCD，并报告验证集上的最终分割性能。结果总结在 **表7** 中。
*   仅使用 CLIP、仅使用扩散模型或简单融合（平均/最大）都在约 75-76% 的伪掩码质量和 76-77% 的最终分割性能附近。
*   **EBF 将它们分别提升到 82.1% 和 79.5%**。
这一明显的优势表明，我们基于熵的融合在伪标签阶段至关重要，它自动选择每个像素处更可信的分支，并提供了比简单组合两个分支更强大的监督。

#### 类别行为分析
除了整体 mIoU，我们还使用 **图6** 中的雷达图从类别层面检查伪掩码的质量。
*   总体而言，ComCD 在大多数 21 个类别上往往优于单分支基线，如 CLIP-ES、ExCEL 和 iSeg，在一些代表性类别上优势更明显。
*   特别是，对于具有细小结构或丰富局部细节的类别（例如椅子、瓶子、摩托车、人物）以及经常与复杂共现背景一起出现的类别（例如船、沙发），ComCD 在图6中显示出明显更大的半径，表明伪掩码更稳定，噪声更少。
一个合理的解释是，两个分支强调不同的方面：CLIP 分支对类别判别性线索更敏感，而扩散分支在保持区域连通性方面更有效。基于熵的融合在这些“精细结构+强上下文”的情况下似乎特别有帮助，在这些情况下，平衡判别性和空间一致性对于单一分支来说并非易事。

#### 对象尺寸和形状的鲁棒性
为了评估 ComCD 在不同对象尺度和形状下的表现，我们将 PASCAL VOC 2012 图像分成三个子集，根据前景对象占据图像面积的比例，并在 **表8** 中报告 ExCEL、iSeg 和 ComCD 的伪掩码 mIoU。
*   在小型、中型和大型子集上，ComCD 都取得了比两个基线更高的 mIoU，在小型和中型对象上增益相对更大。
这些子集通常包含部分被遮挡的实例、多个相邻对象和不规则轮廓，这些情况下单分支方法通常面临在保留局部细节和保持全局区域一致性之间的权衡。基于熵的融合通过自适应地结合两个分支在一定程度上缓解了这种紧张。对于占据图像大部分区域的大型对象，ComCD 也保持了持续优势，这表明所提出的融合策略能够适应对象范围和形状复杂性的变化，而不是局限于狭窄的尺度范围。

#### 边界准确度分析
我们进一步引入了一个以边界为重点的指标，以单独评估对象轮廓周围的分割质量。从真实掩码开始，我们通过将掩码膨胀和腐蚀两个像素，构建每个对象边界周围的窄带，并仅在此带内计算 mIoU。**表9** 报告了 ExCEL、iSeg 和 ComCD 的边界 mIoU。
*   ComCD 在三种方法中取得了最高分，表明在对象边缘处局部间隙更少，背景泄漏更不明显。
这一观察结果与 **图8** 中可视化的高度相关的基于熵的权重分布一致：在权重图中，靠近边界的像素通常对 CLIP 分支获得更高的权重，而内部像素倾向于偏向扩散分支。两个分支之间这种角色划分在类别判别性边界和区域级一致性之间提供了更稳定的平衡，这反过来有助于产生更平滑、更连续的轮廓预测。

#### 失败案例和局限性
尽管取得了上述改进，ComCD 仍然表现出一些典型的失败案例。
*   **室内场景中的密集布置对象**：例如餐桌和椅子。当桌子和椅子密集排列且背景混乱时，扩散分支可能会过度平滑桌面和周围椅子之间的过渡，而 CLIP 分支可能主要关注桌面而低估细长的桌腿和支撑。
*   **自行车等细长结构**：车轮辐条和车架结构非常精细；在 challenging 视角或遮挡下，两个分支在这些区域都可能置信度较低，融合预测仍然可能错过车轮的某些部分或将其与附近的背景合并。
*   **小型或视觉模糊的瓶子实例**：例如透明或高反光的瓶子。CLIP 分支有时会强调标志或高对比度纹理，而扩散分支可能会将瓶子连接到具有相似颜色或纹理的背景区域。

这些观察表明，当前的设计依赖于两个冻结骨干网络的逐像素熵和相对简单的融合规则，并未明确建模实例级结构或更丰富的几何先验。我们认为，将实例感知精化和更具表达力的不确定性建模纳入未来工作是很有前景的方向。

#### 效率分析
**表10** 总结了 PASCAL VOC 2012 上的运行时、内存使用和分割性能。
*   与其他单阶段基于 ViT 的基线相比，ComCD 由于双分支设计和解码器中的特征对齐，带来了更高的计算成本 (3.77 FPS 和 3.98 GB 峰值 GPU 内存)，同时实现了 79.5% 的最佳 mIoU。
*   值得注意的是，我们的吞吐量与现有方法保持在同一数量级 (并且高于 DuPL)，内存占用在现代 GPU 上仍然可以接受。
在优先考虑分割准确性的场景中，这种准确性-效率权衡是合理的，而进一步降低开销（例如，通过更轻的扩散骨干网络或特征缓存）则留作未来有前景的工作。

### 5. 结论

本文提出了 ComCD，一个结合 CLIP 局部化能力和扩散模型结构连贯性的 WSSS 框架。首先，通过基于熵的加权将两个分支的 CAMs 融合，生成融合 CAM，然后转换为伪掩码。此伪掩码监督 Feature Aligned Decoder，该解码器对齐特征以供共享解码器使用，从两个分支生成 logits，并应用 Logit Gating Module2 产生最终的融合预测。通过利用两个分支的互补优势，所提出的融合抑制了虚假激活，减少了背景泄漏，并恢复了具有更锐利边界的更完整对象。