---
type: paper-note
tags:
  - cv
  - semantic-segmentation
  - weakly-supervised
  - clip
  - vit
  - dino
  - contrastive-learning
  - wsss
status: todo
model: "SSR: Semantic and Spatial Rectification"
year: 2025
---
论文网址：[SSR: Semantic and Spatial Rectification for CLIP-based Weakly Supervised Segmentation](https://arxiv.org/pdf/2512.01701

本地PDF文件：[SSR](../../../../../../99_Assets%20(资源文件)/papers/SSR%20Semantic%20and%20Spatial%20Rectification%20for%20CLIP-based%20Weakly%20Supervised%20Segmentation.pdf)

***
### 摘要 (Abstract)

近年，对比语言-图像预训练（CLIP）因其强大的跨模态语义理解能力，被广泛应用于弱监督语义分割（WSSS）任务。本文提出了一种新颖的语义和空间校正（SSR）方法，以解决现有基于CLIP的弱监督语义分割方法的局限性：非目标前景区域和背景区域的过激活问题。具体来说，在语义层面，**跨模态原型对齐（CMPA）** 建立了一种对比学习机制，以强制跨模态特征空间对齐，从而减少类间重叠并增强语义相关性，有效纠正非目标前景区域的过激活。在空间层面，**超像素引导校正（SGC）** 利用基于超像素的空间先验，在亲和传播过程中精确过滤非目标区域的干扰，显著纠正背景过激活。在PASCAL VOC和MS COCO数据集上进行的大量实验表明，我们的方法超越了所有单阶段方法以及更复杂的多阶段方法，分别取得了79.5%和50.6%的mIoU分数。

### 引言 (Introduction)

WSSS的目标是利用点、涂鸦、边界框或图像级标签等弱监督信息生成高质量的伪标签，以解决全监督语义分割中像素级标注成本高、耗时长的问题。其中，图像级标签是最流行但最具挑战性的标注形式。本文重点研究基于图像级标签的WSSS。

当前的WSSS方法通常遵循三阶段流程：
1. 训练分类网络生成初始类别激活图（CAM）。
2. 优化这些CAM。
3. 生成伪标签用于训练分割模型。

传统的基于CNN的CAM方法（如(Guo et al. 2022; He et al. 2024)）由于感受野有限，存在激活不足（只关注最具区分度的对象部分）的问题；而最近的基于Vision Transformer (ViT) 的方法（如(Qin et al. 2022)）通过多头自注意力（MHSA）捕获全局上下文，展现出优越的性能。然而，ViT架构引入了新的挑战，即过激活（错误激活背景区域），当前研究（如(Yoon et al. 2024b,a)）通过改进位置编码或CNN-ViT混合架构来解决这一问题。

近年，研究人员开始探索将CLIP（Radford et al. 2021）整合到WSSS中（Yang and Gong 2024; Yang et al. 2025a）。他们使用基于梯度的方法（Selvaraju et al. 2017）生成初始CAM，这些CAM的性能显著优于传统CNN和ViT解决方案，因此在WSSS应用中得到广泛采用。

CLIP作为一个多模态模型，将图像和文本投影到共享表示空间中，使匹配的图像-文本嵌入更接近，不匹配的则推开。然而，在下游应用中，CLIP仍然面临非目标前景区域过激活的问题，这源于图像和文本模态之间语义对齐不足，也就是其固有的模态鸿沟（modality gap）（Liang et al. 2022），如图1所示。视觉特征主要关注低级模式（如颜色和形状），而文本特征则强调更高级、更抽象的语义。为解决语义错位问题，现有改进方法主要集中在优化文本提示的质量。但这些方法（Yang and Gong 2024; Yang et al. 2025a）仅修改了文本侧表示，并未从根本上弥合视觉和文本模态之间的跨模态表示鸿沟。因此，它们未能完全解决由语义错位引起的激活错误。

在CAM特征细化过程中，背景区域与目标区域之间异常高的亲和值常常导致背景区域的错误激活，进而引发虚假背景响应问题（如图1所示）。为解决这一问题，现有研究主要采用两种方法：一是采用多阶段迭代优化策略（Zhang et al. 2021），通过逐步交替训练机制消除背景噪声；二是实现亲和矩阵约束（Ru et al. 2022），通过应用阈值截断或引入约束损失函数来增强监督信号的可靠性。然而，这些方法仍然受到低级特征的干扰和全局上下文混淆问题的影响。

**图1：我们的动机。**
(a) 先前方法存在固有限制。
(b) 初始CAM阶段面临模态鸿沟问题，视觉特征空间中存在类内离散和类间重叠的双重挑战。
(c) 优化CAM阶段受到虚假背景响应的困扰，因为亲和估计被背景噪声污染。
(d) 为解决这些问题，我们提出了SSR。

本研究提出了语义和空间校正方法，以解决由模态鸿沟引起的语义错位以及由虚假背景响应引起的亲和噪声。在语义层面，我们的方法旨在：
1. 保持特征的跨模态语义一致性。
2. 确保相同类别的像素在特征空间中表现出相似的表示。
为实现此目标，我们设计了**跨模态原型对齐（Cross-Modal Prototype Alignment）**，它增强了模态间对齐和特征的区分能力，为伪标签生成提供了更可靠的基础。为了在空间层面进一步提高初始CAM的质量，从而生成更精确的伪标签，我们提出了**超像素引导校正（Superpixel Guided Correction）**。SGC通过引入超像素指导，精确筛选目标和背景区域之间的潜在错误关联，从而有效增强前景语义一致性，同时抑制背景噪声干扰。

本文的主要贡献如下：
* 提出了语义和空间校正方法，以解决基于CLIP的弱监督语义分割中非目标前景区域和背景区域的过激活问题。
* 跨模态原型对齐在语义层面建立了对比学习机制，以对齐特征空间中不同模态的特征表示。
* 超像素引导校正通过利用超像素分割来校正ViT的亲和矩阵，从而实现局部亲和优化。
* 在PASCAL VOC和MS COCO上的大量实验表明，我们的方法显著优于最新的SOTA方法。

### 相关工作 (Related Work)

#### 弱监督语义分割 (Weakly Supervised Semantic Segmentation)

为解决CAM仅突出最具区分度对象区域的局限性，研究人员提出了各种创新解决方案。擦除（erasing）（Kweon et al. 2021）、跨图像挖掘（cross-image mining）（Sun et al. 2020; Li et al. 2021）、自监督学习（self-supervised learning）（Wang et al. 2020; Chen et al. 2022）和对抗攻击（adversarial attack）（Lee, Kim, and Yoon 2021）等方法从不同角度增强了CAM的覆盖范围。
一些工作采用了原型学习框架：PSDPM（Zhao et al. 2024）使用类别原型激活更多次要区分性像素，而FPR（Chen et al. 2023）构建了类别特定的正负原型。
ViT的出现展现出有前景的定位能力。A2GNN（Zhang et al. 2021）通过亲和卷积构建语义图结构，而AFA（Ru et al. 2022）利用MHSA机制探索像素级语义关系。其他方法侧重于位置编码：CTI（Yoon et al. 2024b）从图像中注入类别特定 tokens，而MCTformer（Xu et al. 2022）通过块间成对亲和计算优化类别特定注意力图。

#### 对比语言-图像预训练 (Contrastive Language-Image Pre-training)

在WSSS领域，CLIP因其卓越的性能而备受关注。CLIMS（Xie et al. 2022）在前景和背景区域采用对比学习进行跨语言图像匹配。CLIP-ES（Lin et al. 2023）采用精心设计的手动文本提示并利用softmax函数生成GradCAM。WeakCLIP（Zhu et al. 2025）将WSSS任务转化为连续的文本-图像匹配问题，有效利用了视觉-语言预训练知识。WeCLIP（Zhang et al. 2024）直接采用CLIP视觉编码器完成分割任务。然而，这些方法都面临多模态模型中固有的模态鸿沟（Liang et al. 2022）问题，导致视觉和文本特征之间的语义对齐不一致。为了解决这一挑战，研究人员提出了各种创新解决方案。FMA（Yang and Gong 2024）方法通过为分类和分割任务分别设计可学习提示来优化文本特征表示。ExCEL（Yang et al. 2025a）通过大型语言模型生成细粒度类别描述来丰富文本提示信息。VPL（Xu et al. 2025）采用梯度下降方法在视觉空间中学习类别特定的视觉原型，取代传统文本原型，更准确地捕获语义目标区域的特征。

### 方法 (Methodology)

#### 框架概述 (Framework Overview)

由于文本和视觉模态之间存在显著差异，传统的文本特征优化对齐方法往往难以建立精确的像素级语义对应关系，导致非目标前景区域的错误激活。在CAM细化过程中，亲和矩阵中的噪声经常导致背景过激活。为了系统地解决这些挑战，本文提出的SSR框架创新性地从语义理解和空间关系双维度进行协同建模和联合优化，从而实现更准确的跨模态像素级语义对齐。

**图2：我们的SSR概述。** 我们提出了两个新颖的组件来解决模态鸿沟和错误激活的关键挑战：CMPA和SGC。
(a) CMPA利用跨模态原型对比学习，在共享嵌入空间中建立视觉特征和文本原型之间的精确匹配关系，从而有效缓解类别混淆。
(b) SGC利用超像素聚类导出的局部空间一致性先验，选择性地过滤特征亲和矩阵，消除错误的跨区域传播，引导特征细化过程朝着语义一致的方向发展，从而显著抑制背景过激活现象。

图2展示了SSR的整体框架。跨模态输入由图像模态 $I$ 和文本模态 $T$ 组成，其中 $T$ 包含 $K$ 个前景类别（类别标签 $Y = 1, 2, ..., C$）和 $M$ 个从CLIP-ES（Lin et al. 2023）派生出的背景类别。在语义层面，**跨模态原型对齐（Cross-Modal Prototype Alignment, CMPA）** 通过图像和文本原型之间的对比学习显著减小模态鸿沟，实现紧凑的跨模态特征对齐。在空间层面，**超像素引导校正（Superpixel-Guided Correction, SGC）** 引入了噪声过滤机制，有效抑制亲和建模中错误信息的传播。

#### 跨模态原型对齐 (Cross-Modal Prototype Alignment)

##### 多模态原型生成 (Multimodal Prototype Generation)

为解决视觉-语言模态之间固有的语义鸿沟，我们提出了一种双分支对齐解决方案：对于一批 $N$ 个图像-文本对 $(I_i, T_i)_{i=1}^N$，由结构相同但参数独立的图像语义对齐（Image Semantic Alignment, ISA）和文本语义对齐（Text Semantic Alignment, TSA）模块分别优化从CLIP中提取的视觉特征 $v'_i \in R^{1 \times d_1}$ 和文本特征 $t'_i \in R^{1 \times d_1}$。通过定制的损失函数约束，该解决方案显著改善了跨模态特征的细粒度语义对齐。获得更高语义对齐的图像和文本表示如下：

$v'_i = ISA(v_i), \quad t'_i = TSA(t_i) \quad (1)$

其中 $v'_i \in R^{1 \times d_2}$ 和 $t'_i \in R^{c_f \times d_2}$ 表示通过ISA模块和TSA模块获得的投影图像特征和文本特征，其中 $c_f$ 表示当前图像的类别数量。

对于每张图像，我们使用GradCAM（Selvaraju et al. 2017）生成CAM $CAM_c^{ij}$，并计算图像/文本特定原型。这些原型在ISA和TSA的投影空间中构建，主要基于两个原因：(1) 通过将原型区分转移到投影空间来保留CLIP固有的实例区分能力，以及 (2) 通过这些模块中的投影头进行降维，显著降低原型构建成本。具体来说，包含前景目标信息的图像特征 $f_{image}$ 和文本特征 $f_{text}$ 计算如下：

$f_{image} = MAP(CAM_c \odot v'_i), \quad f_{text} = t'_i[index] \quad (2)$

其中 $MAP(\cdot)$ 表示掩码平均池化。由于文本特征 $t'_i$ 的维度为 $R^{c_f \times d_2}$，因此可以使用索引直接检索目标类别的文本特征。

然后，我们从数据集中所有图像-文本对中收集前景感知的图像特征 $f_{image}$ 和文本特征 $f_{text}$，并执行K-means聚类以获得图像原型 $P_I \in R^{K \times d_2} = [P_I^1, P_I^2, ..., P_I^K]$ 和文本原型 $P_T \in R^{K \times d_2} = [P_T^1, P_T^2, ..., P_T^K]$。K-means聚类后，可以根据每个样本表示与其对应原型之间的接近度生成聚类伪标签。

##### 原型对比学习 (Prototype Contrastive Learning)

本研究提出了一种原型对比学习，通过三重约束实现细粒度语义对齐：
1. 视觉特征与相同类别的文本原型匹配。
2. 文本原型与相同类别的视觉原型聚合。
3. 不同类别的跨模态原型分离。

如图2(a)所示，这种设计引导图像特征趋向语义内容，同时使文本特征能够关注视觉可对齐的属性，直接缩小了模态鸿沟，增强了共享嵌入空间中的类间区分能力，从而有效解决了类别混淆问题。核心创新在于构建跨模态正负样本对，通过对比学习同步优化模态对齐和分类边界。为了建立对比学习，使用视觉特征 $v'_i$ 和文本原型 $P_T$ 构建的正负样本对定义如下：

$p_I^i = \frac{v'_i \cdot P_T}{\tau_{proto}} \quad (3)$

$S_{i}^{pos} = \{p_{i,j} | j = pos_{idx}\} \quad (4)$

$S_{i}^{neg} = \{p_{i,j} | j \neq pos_{idx}\} \quad (5)$

其中，温度超参数 $\tau_{proto}$ 被设置为可学习参数以优化模型性能。为了构建对比学习样本，$pos_{idx}$ 表示引导正负样本对形成的聚类生成的伪标签，其中 $S_{i}^{pos}$ 表示正样本对，$S_{i}^{neg,k}$ 表示第 $k$ 个负样本对。在此基础上，我们采用交叉熵损失函数计算所有批次内样本的平均损失，原型对比损失 $L_{proto}$ 由下式实现：

$L_{proto} = - \frac{1}{N} \sum_{i=1}^N \log \frac{\exp(S_i^{pos})}{\exp(S_i^{pos}) + \sum_{k=1}^K \exp(S_i^{neg,k})} \quad (6)$

#### 超像素引导校正 (Superpixel-Guided Correction)

##### 超像素聚类 (Superpixel Clustering)

为了解决注意力机制中错误的亲和传播导致背景区域误激活的问题，我们提出了超像素引导校正。如图2(b)所示，该模块利用超像素结构信息构建二值掩码，选择性地掩码与非目标区域相关的亲和矩阵中的列向量，从而有效抑制背景区域中错误语义的传播。通过用结构化先验约束注意力传播范围，只保留目标区域内的语义相关性。具体来说，我们定义一个二值掩码矩阵 $Mask$，其元素定义如下：

$Mask = \begin{cases} a_{ij} = 1 & \text{if } j \in \text{target regions}, \\ a_{ij} = 0 & \text{if } j \notin \text{target regions}, \end{cases} \quad (7)$

其中 $i$ 和 $j$ 分别表示亲和矩阵的行索引和列索引。

为了更准确地提取输入图像 $I_i$ 的目标区域，我们采用SLIC（Achanta et al. 2012）算法进行超像素分割。该方法基于特征相似性对像素进行聚类，以减少超像素数量的方式有效地表示图像，同时保持对象边界的完整性。随后，基于颜色空间信息对超像素区域进行聚类，得到目标区域 $C$ 为：

$C = \text{K-means}(\text{SLIC}(I_i)) \quad (8)$

我们计算每个聚类区域内高置信度像素激活的总和与总激活值之比。只有比率高于预定义阈值的区域才被分类为目标区域。此外，SLIC的轻量级设计使其相较于SAM等复杂模型更适合作为SGC中空间先验的角色。

##### 亲和矩阵校正 (Affinity Matrix Correction)

CLIP的MHSA擅长提取全局语义特征，但其空间细节捕获不足导致CAM边界模糊。相比之下，DINO的MHSA通过自监督训练得到增强，强化了局部到全局的一致性。我们使用DINO的局部结构来改进CAM空间先验。为了解决这个问题，我们整合了它们的MHSA特征并对其进行归一化以获得亲和矩阵 $A$：CLIP提供高级语义指导，而DINO补充细粒度空间关系，从而生成既保持类别区分性又实现精确空间定位的CAM，融合后的亲和矩阵计算如下：

$A = \text{Concat}(\text{MHSA}_{\text{CLIP}}, \text{MHSA}_{\text{DINO}}) \quad (9)$

其中 $\text{MHSA}_{\text{CLIP}}$ 代表来自CLIP的MHSA，$\text{MHSA}_{\text{DINO}}$ 表示来自DINO的MHSA，亲和矩阵 $A$ 由它们的 $\text{Concat}$ 操作归一化后获得，有效结合了两种注意力机制用于跨模态特征融合。

随后，我们使用获得的 $Mask$ 对亲和矩阵 $A$ 进行细化，以得到更新后的矩阵 $A^*$，其中冗余的非目标列元素被消除。这个细化后的亲和矩阵 $A^*$ 随后被用于通过空间传播增强初始CAM。校正后的亲和矩阵 $A^*$ 和最终的CAM计算如下：

$A^* = A \odot Mask \quad (10)$

$CAM_c^{refine} = A^* \otimes CAM_c \quad (11)$

其中 $CAM_c^{refine}$ 表示与前景目标类别 $c$ 对应的细化CAM。

#### 训练目标 (Training Objectives)

SSR的总体损失由两部分组成：原型对比损失 $L_{proto}$ 和分割损失 $L_{seg}$。$L_{proto}$ 鼓励样本靠近其对应的同类跨模态原型，同时远离不同类的原型。$L_{seg}$ 使用在线生成的伪掩码，并采用交叉熵公式进行端到端训练以完成分割任务。我们方法的客观函数可以表示为：

$L_{SSR} = L_{proto} + \gamma L_{seg} \quad (12)$

### 实验 (Experiments)

#### 实验设置 (Experimental Settings)

##### 数据集和指标 (Datasets and Metrics)

所提出的SSR方法在PASCAL VOC 2012（Everingham et al. 2015）和MS COCO 2014（Lin et al. 2014）数据集上进行评估。VOC包含21个类别（包括1个背景类别），并遵循既定协议（Li et al. 2021; Du et al. 2022），使用10,582张图像的增强训练集、1,449张验证图像和1,456张测试图像。COCO包含81个类别，拥有82,081张训练图像和40,137张验证图像。我们采用mIoU作为主要评估指标，辅以次要指标，包括混淆比（confusion ratio）、精确率（P）和召回率（R），以进行全面的性能评估。

##### 实现细节 (Implementation Details)

CLIP采用ViT-B/16，DINO使用DINOv1的ViT-S/16架构。我们使用AdamW优化器，学习率为1e-5，权重衰减为2e-3。对于PASCAL VOC 2012，批量大小设置为128，最大迭代次数为30,000次；而MS COCO-2014使用更大的批量大小256和80,000次迭代。损失权重 $\gamma$ 设置为0.1，原型温度系数 $\tau_{Proto}$ 设置为0.05。ISA和TSA均由堆叠的线性层、批归一化和ReLU组成。原型每5,000次迭代更新一次。在SGC中，CLIP:DINO的权重比为0.4:0.6。

#### 与SOTA方法的比较 (Comparisons with State-of-the-art Methods)

##### 语义分割性能 (Performance of Semantic Segmentation)

**表1：PASCAL VOC和COCO数据集上的分割性能比较 (mIoU%)，显示了VOC val/test集和COCO val集的结果。**
| Method | VOC Val | VOC Test | COCO Val |
| --- | --- | --- | --- |
| **Multi-stage.** | | | |
| SIPE (Chen et al. 2022) | 68.8 | 69.7 | 43.6 |
| CLIMS (Xie et al. 2022) | 70.4 | 70.0 | - |
| WeakTr (Zhu et al. 2023) | 78.4 | 79.0 | 50.3 |
| CLIP-ES (Lin et al. 2023) | 73.8 | 73.9 | 45.4 |
| PSDPM (Zhao et al. 2024) | 74.1 | 74.9 | 47.2 |
| CPAL (Tang et al. 2024) | 74.5 | 74.7 | 46.8 |
| CTI (Yoon et al. 2024b) | 74.1 | 73.2 | 45.4 |
| WeakCLIP (Zhu et al. 2025) | 74.0 | 73.8 | 47.4 |
| VPL (Xu et al. 2025) | 79.3 | 79.0 | 49.8 |
| **Single-stage.** | | | |
| DIAL (Jang et al. 2024) | 74.5 | 74.9 | 44.4 |
| DuPL (Wu et al. 2024) | 73.3 | 72.8 | 44.6 |
| WeCLIP (Zhang et al. 2024) | 76.4 | 77.2 | 47.1 |
| MoRe (Yang et al. 2025b) | 76.4 | 75.0 | 47.4 |
| ExCEL (Yang et al. 2025a) | 78.4 | 78.5 | 50.3 |
| Ours w/o CRF | 78.2 | 78.1 | 49.2 |
| Ours | **79.5** | **79.6** | **50.6** |

表1展示了SSR与VOC和COCO数据集上最新方法的分割性能比较。SSR在VOC和COCO上都取得了新的SOTA性能，超越多阶段方法高达0.6%，超出基于CLIP的ExCEL方法0.3%。如图3所示，定性结果表明了卓越的分割质量：(1) 通过增强跨模态对比学习实现精确的类别预测；(2) 改善区域完整性和边界清晰度；(3) 在多类别场景中具有更强的类间区分能力。

##### CAM种子评估 (Evaluation of CAM Seeds)

**表2：VOC训练集上的CAM种子比较。D:DINO。N:网络。R:ResNet。D:Deit。V:Vit-B。**
| Method | Sup. | N. | VOC Train |
| --- | --- | --- | --- |
| MCTformer CVPR’2022 | I +DR | 61.7 |
| CLIMS (Xie et al. 2022) | I +LR | 56.6 |
| WeakTr (Zhu et al. 2023) | ID | 66.2 |
| POLE WACV’2023 | I +LR | 59.0 |
| CLIP-ES (Lin et al. 2023) | I +LV | 70.8 |
| ToCo (Ru et al. 2023) | IV | 71.6 |
| CPAL (Tang et al. 2024) | I +LR | 71.9 |
| CTI (Yoon et al. 2024b) | ID | 69.5 |
| SeCo (Yang et al. 2024) | IV | 74.8 |
| DuPL (Wu et al. 2024) | IV | 75.0 |
| WeCLIP (Zhang et al. 2024) | I +LV | 75.4 |
| DIAL (Jang et al. 2024) | I +LV | 75.2 |
| WeakCLIP (Zhu et al. 2025) | I +LV | 61.7 |
| VPL (Xu et al. 2025) | I +LV | 77.8 |
| MoRe (Yang et al. 2025b) | IV | 77.0 |
| ExCEL (Yang et al. 2025a) | I +LV | 78.0 |
| Ours | I +L +DV | **78.7** |

表2报告了VOC训练集上CAM种子的质量。与最新方法相比，SSR将CAM质量进一步提高到78.7%，至少超越SOTA 0.7%。如图4所示，CAM可视化结果表明，CMPA生成的初始CAM相对于CLIP具有显著优势，表现出卓越的完整性。以“沙发”类别为例，CLIP方法错误地激活了相邻的“椅子”区域，并且此错误在后续处理中被放大。SSR不仅实现了对目标区域更精确的聚焦，而且表现出更强的背景噪声抑制能力。

**图3：SeCo、DUPL、WeCLIP、MoRe和Ours在VOC和COCO上的分割可视化。** 列1-4：PASCAL VOC数据集上的结果。列5-7：MS COCO数据集上的结果。SSR更精确地分割对象。

**图4：VOC验证集上的CAM可视化。** 我们对CLIP生成的初始CAM和CMPA生成的CAM进行了比较分析，并评估了WeCLIP结果和我们最终优化输出之间的性能差距。

#### 消融研究 (Ablation Study)

##### SGC的有效性 (Effectiveness of the SGC)

**表3：SGC在PASCAL VOC训练集上的组件消融研究。At 表示注意力图。**
| CPMAt | CLIP At | DINO At | SGC | PRmIoU |
| --- | --- | --- | --- | --- |
| ✓ | | | | 63.3 |
| ✓ | ✓ | | | 74.6 |
| ✓ | ✓ | ✓ | | 76.3 |
| ✓ | ✓ | ✓ | ✓ | **78.7** |

表3展示了SGC在PASCAL VOC上的组件消融研究：CPMA生成的初始CAM实现了63.3%的mIoU，加入CLIP的多头注意力后提升到74.6%。进一步与DINO的注意力机制集成后，mIoU达到76.3%，表明其在增强块间语义关系方面的互补作用。我们完整的SGC模块通过在CAM细化过程中有效校正注意力图中的融合错误，最终实现了78.7%的mIoU。

##### 训练损失的有效性 (Effectiveness of the training loss)

**表4：使用三种不同损失函数对SSR的性能评估。**
| Conditions | $L_{feature}$ | $L_{in\_modal}$ | $L_{crossmodal}$ | mIoU |
| --- | --- | --- | --- | --- |
| CLIP | | | | 58.6 |
| w $L_{feature}$ | ✓ | | | 53.5 |
| w $L_{in\_modal}$ | | ✓ | | 57.8 |
| w $L_{crossmodal}$ | | | ✓ | **63.3** |

我们评估了PASCAL VOC数据集上的三种不同损失函数。如表4所示，基线CLIP实现了58.6%的mIoU，而直接微调 ($L_{feature}$) 性能下降了5.1%；模内对比学习 ($L_{in\_modal}$) 仅下降了0.8%，表明单模态原型比较效果有限；相比之下，我们的跨模态对比损失 ($L_{crossmodal}$) 通过对齐同类别的跨模态表示同时分离不同类别，显著提升mIoU 4.7%，缓解了模态语义鸿沟。

##### 关键指标上的模型性能 (Model Performance Across Key Metrics)

**表5：与近期方法在PASCAL VOC验证集上四个指标的比较。**
| Method | mIoU | Precision | Recall | Confusion |
| --- | --- | --- | --- | --- |
| SeCo | 0.740 | 0.840 | 0.849 | 0.232 |
| WeCLIP | 0.764 | 0.844 | 0.861 | 0.237 |
| MoRe | 0.764 | 0.837 | 0.847 | 0.239 |
| Ours | **0.795** | **0.879** | **0.891** | **0.198** |

表5比较了我们的方法与SeCo、MoRe和WeCLIP在PASCAL VOC验证集上四个关键指标的性能。我们的方法在所有指标上均表现优越：mIoU超越最佳基线MoRe 3.1%，精确率高出3.5%，显示出增强的检测准确性。召回率提高3%表明分割更完整，假阳性更少，而混淆比降低3.4%则证实了更好的类间区分能力。

##### 全监督对应方法 (Fully-supervised Counterparts)

**表6：与VOC验证集上全监督方法的比较。F: 全监督。ViT-B*: 从CLIP预训练。**
| Methods | Sup. | Net. | Val | Ratio |
| --- | --- | --- | --- | --- |
| DeepLabV2 | F | RN101 | 77.7 | - |
| DeepLabV2 | F | ViT-B | 82.3 | - |
| CLIMS CVPR’2022 | I +LR | RN101 | 70.4 | 90.6% |
| CLIP-ES CVPR’2023 | I +LR | RN101 | 72.2 | 92.9% |
| CPAL CVPR’2024 | I +LR | RN101 | 74.5 | 95.9% |
| ToCo CVPR’2024 | I | ViT-B | 71.1 | 86.4% |
| DuPL CVPR’2024 | I | ViT-B | 73.3 | 89.1% |
| SeCo CVPR’2024 | I | ViT-B | 74.0 | 89.9% |
| DIAL ECCV’2024 | I +L | ViT-B | 74.5 | 90.5% |
| WeCLIP CVPR’2024 | I +L | ViT-B* | 76.4 | 93.6% |
| ExCEL CVPR’2025 | I +L | ViT-B* | 78.4 | 96.1% |
| Ours | I +L +D | ViT-B* | **79.5** | **97.4%** |

表6系统地比较了SSR与全监督方法的性能。较小的性能差距表明了我们方法的有效性。SSR在VOC 2012验证集上取得了79.5%的mIoU，达到了全监督性能的97.4%。结果表明，与现有WSSS方法相比，SSR具有显著优势。

##### CAM细化可视化 (Visualization of CAM refinement)

**图5：CAM细化可视化：** (c) 初始CAM带有背景伪影；(d) 经过SGC处理后的细化CAM，显示出更干净的背景抑制和更清晰的目标聚焦。

图5展示了SGC对初始CAM的优化效果。经过超像素细化后，背景区域的错误激活被显著抑制。这种改进主要得益于精心设计的亲和矩阵，它有效消除了非目标区域亲和关系的干扰，从而实现了激活区域更精确的定位。

##### 特征对齐的有效性 (Effectiveness of feature alignment)

**图6：Pascal VOC 2012验证集上特征嵌入的t-SNE（Van der Maaten and Hinton 2008）可视化显示：** (a) 原始CLIP特征空间分布，以及 (b) 优化后的特征分布。

为了验证CPMA的有效性，图6比较了CLIP特征和CPMA增强后的特征在PASCAL VOC 2012上的表现。虽然CLIP特征展现出良好的迁移能力，但其分布稀疏且存在类别重叠，而CPMA则产生了更紧凑的类内聚类和更清晰的类间边界（例如，餐桌/瓶子的明显分离）。

### 结论 (Conclusion)

我们提出了SSR方法，以解决基于CLIP的WSSS中存在的模态鸿沟和虚假背景响应问题，有效抑制了非目标前景和背景区域的错误激活。具体而言，我们提出的CMPA在特征和跨模态原型之间建立对比关系，实现类内聚合和类间分离。同时，SGC动态调整特征亲和的传播方向，有效抑制亲和矩阵中非目标区域的冗余关联。大量实验表明，我们的方法在分割精度和错误抑制方面展现出显著优势，充分验证了所提解决方案的有效性。
