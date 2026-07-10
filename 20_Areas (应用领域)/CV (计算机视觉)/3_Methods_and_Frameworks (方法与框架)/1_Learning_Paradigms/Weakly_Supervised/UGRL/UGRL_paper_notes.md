---
type: paper-note
tags:
  - cv
  - semantic-segmentation
  - weakly-supervised
  - vit
  - transformer
  - contrastive-learning
  - wsss
status: todo
model: Uncertainty-Guided Reliable Learning (UGRL)
venue: AAAI2026
---
AAAI网址：[Escaping the CAM Shadow: Uncertainty-Guided Reliable Learning for Weakly Supervised Semantic Segmentation | Proceedings of the AAAI Conference on Artificial Intelligence](https://ojs.aaai.org/index.php/AAAI/article/view/37262)

本地PDF: [UGRL](../../../../../../99_Assets%20(资源文件)/images/Escaping%20the%20CAM%20Shadow%20Uncertainty-Guided%20Reliable%20Learning%20for%20Weakly%20Supervised%20Semantic%20Segmentation.pdf)

***
# 逃离CAM的阴影：弱监督语义分割中不确定性引导的可靠学习

## 摘要

弱监督语义分割（WSSS）受苦于粗糙的图像级注释和密集的像素级预测之间固有的不匹配。为了弥合这一差距，现有方法主要关注于生成精炼的类激活图（CAM）作为伪标签。然而，我们认为这种关注是不充分的，因为它忽略了一个关键组件：分割解码器。解码器通常通过在逻辑空间中将预测与伪标签进行表面对齐来训练。鉴于此类标签的嘈杂性质，这种天真的监督会导致误差累积并限制性能。为了解决这个问题，我们提出了一个不确定性引导的可靠学习（UGRL）框架，该框架施加双重控制来重塑学习过程，实现逃离CAM阴影的鲁棒监督。UGRL的基石是一个原型驱动的不确定性建模模块，用于估计类级监督的可靠性。建模出的不确定性实现了两种协同控制机制。首先，它自适应地调制分类和分割损失，鼓励模型从更值得信赖的信号中学习。其次，它指导解码器特征空间的结构化。UGRL不仅依赖于表面对齐，而是通过在可靠像素上应用对比学习来强制进行更深层次的表示对齐。这使得丰富的语义能够转移到细粒度的分割细节中。在PASCAL VOC和MS COCO上的广泛实验表明，我们的方法超越了其他最先进的WSSS方法。

## 方法论

我们的UGRL框架的流程如图2所示。它由三个关键模块组成：原型驱动的不确定性建模（PUM）、不确定性引导的损失调制（ULM）以及可靠语义增强（RSE）。

![](../../../../../../99_Assets%20(资源文件)/images/a348b194c825adc0804a0b1655d2c543.png)

**图2：我们UGRL的概述。该框架通过三个关键阶段对学习过程施加双重控制：（1）PUM：多头注意力加权的类别表示被投影到一个超维空间中，然后捆绑成原型以量化类级不确定性。（2）ULM：类级不确定性直接调制分类损失，并被传播到像素级以自适应地重新加权分割损失。（3）RSE：解码器的像素嵌入被投影到一个单独的超球面空间中，并通过不确定性引导的对比学习目标进行结构化。**

【图2内容说明】：整个网络分为三大区块：上方是原型驱动的不确定性建模（PUM）模块，图像经过编码器输出注意力和分类特征，通过投影融合后产生表示，随后运用全局EMA更新的池子计算类别不确定性。中间是负责调控的不确定性引导损失调制（ULM）模块，它将不确定性不仅应用到分类损失上，还映射到伪标签上用以调制分割损失。下方是可靠语义增强（RSE）模块，解码器输出进行非线性映射并加以L2归一化，挑选低不确定性（可靠）的像素应用引导式对比学习，从而实现特征空间的聚合与分离。

**原型驱动的不确定性建模**

为了挑战所有图像级标签都提供同等可信指导的假设，我们提出了原型驱动的不确定性建模（PUM）模块。PUM的核心功能是在一个超维空间内构建一组稳定的、数据集范围的类原型。这些原型随后作为语义锚点，使我们能够通过测量其特征与这些锚点的相似性来量化每张图像的类级不确定性。

**超维原型表示学习。**
我们利用Transformer主干网络提取特征来构建类级原型。标准的全局池化会导致语义稀释，对于这项任务是次优的。为了解决这个问题，我们引入了一个语义亲和力矩阵 $A \in \mathbb{R}^{hw \times hw}$ 来指导池化，其中 $hw$ 是展平后的空间尺寸。受（Ru等人 2022）的启发，$A$ 衍生自跨多个Transformer层的自注意力图中所嵌入的丰富关系信息。具体来说，对于 $L$ 层中的每一层，我们提取 $n$ 个注意力图并将它们堆叠起来，形成一个特定于层的空间张量 $S^{(l)} \in \mathbb{R}^{hw \times hw \times n}$。然后将这些张量连接起来，创建一个融合的多级表示：$S_{\mathrm{attn}} = \mathrm{Concat}(S^{(1)}, \dots, S^{(L)})$。基于 $S_{\mathrm{attn}}$，我们使用一个轻量级的多层感知机（MLP）$g_{\mathrm{cls}}$ 计算语义亲和力矩阵 $A$ 如下：
$$
A = g_{\mathrm{cls}}(S_{\mathrm{attn}} + S_{\mathrm{attn}}^{\top}), \tag{1}
$$
其中 $S_{\mathrm{attn}}^{\top}$ 是 $S_{\mathrm{attn}}$ 的转置，以确保对称性。

随后，这个语义亲和力矩阵 $A$ 会指导特定类特征的聚合。首先，将分类头应用于主干网络的最终特征图以生成类激活图 $f_{\mathrm{cls}} \in \mathbb{R}^{hw \times hw \times C}$，其中 $C$ 是语义类的数量。然后，使用矩阵 $A$ 在 $f_{\mathrm{cls}}$ 上执行加权空间池化，产生类表示向量 $f_{\mathrm{cls}}^{\mathrm{pooled}} \in \mathbb{R}^{C}$。为了赋予这些表示鲁棒的几何结构（Chen等人 2025），我们使用一个矩阵 $\Phi$ 将它们投影到一个维度为 $d_{\mathrm{cls}}$ 的空间 $\mathbf{H}_{\mathrm{cls}}$ 中。此过程产生了按类别分类的超向量表示 $H_{\mathrm{cls}}^{f} \in \mathbb{R}^{C \times d_{\mathrm{cls}}}$ 如下：
$$
\phi(f_{\mathrm{cls}}) = \Phi^{(C \times d_{\mathrm{cls}})} \otimes f_{\mathrm{cls}}^{\mathrm{pooled}(C)} = H_{\mathrm{cls}}^{f(C \times d_{\mathrm{cls}})}, \tag{2}
$$
其中 $\otimes$ 表示保留通道维度的爱因斯坦求和约定。对于给定的微批次 $B$，我们随后计算每个类别 $c$ 的原型 $\hat{\mathcal{P}}_{\mathrm{batch}}^{c}$，这是通过将该批次中包含该类别所有图像中对应的超向量捆绑在一起来实现的。受（Levy和Gayler 2008）的启发，这种捆绑实现为一个简单而有效的逐元素求和，正式表达为：
$$
\hat{\mathcal{P}}_{\mathrm{batch}}^{c} = \bigoplus_{x \in B, y_{x}^{c} = 1}^{\mathrm{dim}=d_{\mathrm{cls}}} \phi(f_{\mathrm{cls}}^{x}) \tag{3}
$$
其中 $y_{x}^{c} \in \{0, 1\}$ 是表示类别 $c$ 是否存在于图像 $x$ 中的真实指示器，而 $\bigoplus$ 表示捆绑操作。

**不确定性估计。**
单纯依赖当前批次的幼稚聚合可能会产生嘈杂且不稳定的原型，对批次间的差异非常敏感。为了提高稳定性并促进跨数据集良好泛化表示的学习，我们采用指数移动平均（EMA）更新策略。类别 $c$ 的全局原型更新为：
$$
\mathcal{P}_{\mathrm{global}}^{c} \leftarrow \eta \cdot \mathcal{P}_{\mathrm{global}}^{c} + (1 - \eta) \cdot \hat{\mathcal{P}}_{\mathrm{batch}}^{c}, \tag{4}
$$
其中 $\eta \in [0, 1)$ 是一个控制更新速率动量的超参数。此更新在整个训练过程中应用，以积累跨数据集范围的语义知识。

有了这些提炼的原型 $\mathcal{P}_{\mathrm{global}}$ 作为稳定的语义锚点，我们可以估计每个样本的类级不确定性。核心直觉是，可靠的预测应该在几何上接近其相应的类原型，而较大的距离意味着较高的不确定性，暗示该类别在视觉上是非典型的或模糊的。在空间 $\mathbf{H}_{\mathrm{cls}}$ 中，这种几何邻近性可以通过余弦相似度很好地衡量。因此，我们定义函数 $U(\cdot, \cdot)$ 来测量类别 $c$ 的不确定性分数，即超向量 $H_{\mathrm{cls}}^{c}$ 与其相应的全局原型 $\mathcal{P}_{\mathrm{global}}^{c}$ 之间的余弦距离：
$$
u_{\mathrm{cls}}^{c} = U(H_{\mathrm{cls}}^{c}, \mathcal{P}_{\mathrm{global}}^{c}) = 1 - \frac{\langle H_{\mathrm{cls}}^{c}, \mathcal{P}_{\mathrm{global}}^{c} \rangle}{\|H_{\mathrm{cls}}^{c}\|_{2} \cdot \|\mathcal{P}_{\mathrm{global}}^{c}\|_{2}}. \tag{5}
$$

**不确定性引导的损失调制**

传统分类损失中固有的统一加权使得学习过程变得次优，因为它对可靠监督信号和来自模糊或非典型样本的监督信号赋予同等重要性。为了纠正这一点，我们根据估计的不确定性分数动态地调制损失，促进从可靠监督中学习。具体而言，我们采用简单的指数衰减函数将 $u_{\mathrm{cls}}$ 转换为置信权重，然后将其应用于多标签软间隔损失：
$$
\mathcal{L}_{\mathrm{cls}} = \frac{1}{C} \sum_{c=1}^{C} e^{\frac{1}{\alpha \cdot u_{\mathrm{cls}}^{c}}} (y^{c} \log(\hat{y}^{c}) + (1 - y^{c}) \log(1 - \hat{y}^{c})), \tag{6}
$$
其中 $\alpha$ 是温度超参数，$\hat{y}^{c}$ 表示类别 $c$ 的预测概率。

类似地，我们将这种不确定性引导的原则扩展到分割损失。按照标准做法，我们基于CAM生成像素级伪标签 $Y \in \mathbb{R}^{H \times W}$，其中 $H \times W$ 表示图像大小。然而，使用这些伪标签进行分割训练容易受到继承自原始CAM的噪声和空间不准确性的影响。

为了缓解这种情况，我们根据建模出的不确定性调制标准的逐像素交叉熵损失。首先，我们构建一个逐像素的不确定性图 $u_{\mathrm{seg}} \in \mathbb{R}^{H \times W}$。每个像素 $(i, j)$ 的不确定性被分配为其指定伪标签的类级不确定性，从向量 $u_{\mathrm{cls}} \in \mathbb{R}^{C}$ 中检索得到：
$$
u_{\mathrm{seg}}^{i,j} = u_{\mathrm{cls}}[Y_{i,j}]. \tag{7}
$$
该过程有效地将类级不确定性传播到像素级，从而直接指导分割损失。随后，我们将 $u_{\mathrm{seg}}$ 合并到分割目标中，动态地将训练重点放在更值得信赖的伪标签区域。最终的分割损失定义为：
$$
\mathcal{L}_{\mathrm{seg}} = \frac{1}{H \cdot W} \sum_{i=1}^{H} \sum_{j=1}^{W} e^{\frac{1}{\beta \cdot u_{\mathrm{seg}}^{i,j}}} \cdot Y_{i,j} \log(p_{\mathrm{seg}}^{i,j}), \tag{8}
$$
其中 $\beta$ 是温度超参数，$p_{\mathrm{seg}}^{i,j}$ 表示解码器在像素 $(i, j)$ 处的预测概率。

**可靠语义增强**

尽管损失 $\mathcal{L}_{\mathrm{seg}}$ 通过结合不确定性提高了鲁棒性，但它的操作域仅限于逻辑空间。此类监督专注于表面对齐，迫使预测逼近伪标签，而未对特征表示施加任何明确的结构约束。然而，这些特征的质量对高保真分割至关重要。一个具有高类内紧凑性和类间可分离性的良好结构化特征空间，对于克服WSSS中基于CAM监督的局限性至关重要。为了注入这种更深层次的语义结构，我们提出了可靠语义增强（RSE）模块。该模块超越了表面的逻辑对齐，从而学习更鲁棒的特征流形，这对于生成完整对象掩码和细化锐利分割边界至关重要。

**语义度量空间构建。** 为了获得兼具语义上下文和细粒度空间细节的特征表示，我们的解码器汇集了来自编码器主干各个阶段的多尺度特征图。这些特征被融合以产生一个统一的特征图 $z \in \mathbb{R}^{H \times W \times D_{2}}$，其中 $D_{2}$ 是特征维度。为了促进有效的度量学习，我们采用一个非线性投影头 $g_{\mathrm{seg}}$ 将 $z$ 映射到一个专用的超维空间 $\mathbf{H}_{\mathrm{seg}}$ 中。这产生了嵌入 $H_{\mathrm{seg}}^{z} = g_{\mathrm{seg}}(z)$，其中 $H_{\mathrm{seg}}^{z} \in \mathbb{R}^{N \times d_{\mathrm{seg}}}$ 并且 $N = H \times W$ 表示空间位置的数量。关键的是，这种设计解耦了分割任务和度量学习的表示需求。为了使 $\mathbf{H}_{\mathrm{seg}}$ 成为进行语义测量的有效空间，我们对所有像素嵌入应用L2归一化：
$$
H_{\mathrm{seg}}^{i} \leftarrow \frac{H_{\mathrm{seg}}^{i}}{\|H_{\mathrm{seg}}^{i}\|_{2} + \epsilon}, \quad \text{for } i = 1, \dots, N. \tag{9}
$$
其中 $\epsilon$ 是一个小常数。这种归一化将嵌入约束在单位超球面上，使得余弦相似度可以作为高维空间中语义接近度的可靠代理。

**不确定性引导的对比学习。** 理想情况下，属于同一语义类别的像素级嵌入在 $\mathbf{H}_{\mathrm{seg}}$ 中应该更靠近，而来自不同类别的嵌入应该保持良好的分离。伪标签 $Y \in \mathbb{R}^{H \times W}$ 提供了实现这一点的基础监督。然而，全盘接受这些标签是有问题的，因为 $Y$ 继承了CAM的噪声和空间模糊性，不可避免地包含错误标记的像素。在这种情况下不加区分地应用对比损失可能会强化错误的关系并降低学习到的特征空间质量。为了缓解这一问题，我们提出了不确定性引导的对比学习机制，旨在通过专门从最可靠的子集中学习来选择性地构造特征空间。对于每张图像，我们通过将其相应的不确定性分数 $u_{\mathrm{seg}}$ 按升序对所有像素进行排序，并选择得分最低的Top-$K$个像素，来构建一个可靠池 $\mathcal{R}$。对于可靠池 $\mathcal{R}$ 中的每个锚点像素 $i$，我们定义其正样本集 $\mathcal{R}^{+}$ 和负样本集 $\mathcal{R}^{-}$ 如下：
*   $\mathcal{R}^{+}$ 包含 $\mathcal{R}$ 中共享相同伪标签的所有其他像素 $j$，即 $Y_{j} = Y_{i}$。
*   $\mathcal{R}^{-}$ 包含 $\mathcal{R}$ 中具有不同伪标签的所有像素 $j$，即 $Y_{j} \neq Y_{i}$。

对于可靠集 $\mathcal{R}$ 中的每个锚点 $i$，损失定义为：
$$
\ell_{i} = - \log \frac{\sum_{j \in \mathcal{R}^{+}} \exp(s(H_{\mathrm{seg}}^{i}, H_{\mathrm{seg}}^{j}) / \tau)}{\sum_{j \in \mathcal{R}^{+}} \exp(s(H_{\mathrm{seg}}^{i}, H_{\mathrm{seg}}^{j}) / \tau) + \sum_{j \in \mathcal{R}^{-}} \exp(s(H_{\mathrm{seg}}^{i}, H_{\mathrm{seg}}^{j}) / \tau)}, \tag{10}
$$
其中 $s(\cdot)$ 表示余弦相似度，$\tau$ 是一个温度超参数。通过迭代可靠池 $\mathcal{R}$ 中的所有锚点，最终的不确定性引导对比损失表示为：
$$
\mathcal{L}_{\mathrm{ucl}} = \frac{1}{|\mathcal{R}|} \sum_{i \in \mathcal{R}} \ell_{i}, \tag{11}
$$
其中 $|\mathcal{R}|$ 是可靠锚点的数量。

**整体训练目标**

为了进一步增强性能，我们采用了（Ru等人 2022）中的亲和力损失，但将亲和力矩阵 $A$ 的计算限制在可靠池 $\mathcal{R}$ 内的像素上，公式如下：
$$
\mathcal{L}_{\mathrm{aux}} = \frac{1}{|\mathcal{R}^{+}|} \sum_{i \in \mathcal{R}^{+}} \mathrm{sigmoid}(A^{i}) + \frac{1}{|\mathcal{R}^{-}|} \sum_{j \in \mathcal{R}^{-}} (1 - \mathrm{sigmoid}(A^{j})) \tag{12}
$$
我们的UGRL整体损失被表述为：
$$
\mathcal{L} = \mathcal{L}_{\mathrm{cls}} + \lambda_{1} \mathcal{L}_{\mathrm{seg}} + \lambda_{2} \mathcal{L}_{\mathrm{ucl}} + \lambda_{3} \mathcal{L}_{\mathrm{aux}}, \tag{13}
$$
其中 $\lambda_{1}$、$\lambda_{2}$ 和 $\lambda_{3}$ 是权重因子。
