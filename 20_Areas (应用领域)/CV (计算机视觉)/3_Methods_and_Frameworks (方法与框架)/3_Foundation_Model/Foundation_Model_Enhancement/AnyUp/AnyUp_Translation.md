# AnyUp: Universal Feature Upsampling

AnyUp：通用特征上采样

[项目主页](https://wimmerth.github.io/anyup)

![[AnyUp_Fig01.png]]

**图 1｜** **AnyUp** 是一种通用特征上采样模型，可以把任意特征从任意分辨率上采样到任意分辨率，既能泛化到未见过的特征，又能达到当前最优性能。

**图片内容解释。** 图中展示了 ResNet、CLIP、SigLIP 2、DINOv2 和 DINOv3 的特征可视化。每组箭头从低分辨率特征图指向对应的上采样结果。

## 摘要

我们提出 AnyUp，一种可用于任意分辨率下任意视觉特征的上采样方法，无需针对各个编码器单独训练。现有用于 DINO 或 CLIP 等特征的学习式上采样器需要针对每个特征提取器重新训练，因此在推理时无法泛化到不同类型的特征。本文提出一种在*推理时*与特征无关的上采样架构，以缓解这一局限并提高上采样质量。实验中，AnyUp 在特征上采样方面达到新的最优水平，能够泛化到不同的特征类型并保留特征语义，同时高效且易于应用到各类下游任务。

## 1. 引言

DINO（Caron et al., 2021；Oquab et al., 2023；Xue et al., 2025）、CLIP（Radford et al., 2021）、SigLIP（Tschannen et al., 2025）和 MAE（He et al., 2022）等通用图像特征提取器已成为现代计算机视觉的基础组件，为多种下游应用提供语义信息或语言对齐等能力。这些应用包括深度估计或三维重建（Feng et al., 2024；Wang et al., 2025）、开放词汇语义分割（Engelmann et al., 2024；Wysoczańska et al., 2024）以及生成模型的基准评估（Asim et al., 2025）。这类预训练模型通常基于 Transformer，其一个重要局限是输出特征图的分辨率受 Transformer 的 token 数量限制，因而无法预测像素级特征。为此，FeatUp（Fu et al., 2024）、LoftUp（Huang et al., 2025）和 JAFAR（Couairon et al., 2025）等近期工作提出了学习式特征上采样方法。

这些特征上采样方法与训练时配套的视觉编码器一起使用时表现良好，但在推理时通常仍依赖于该编码器，更换特征提取器时需要重新训练。重新训练的成本可能很高，对于最新的大型视觉模型（Xue et al., 2025），在计算资源有限的情况下甚至不可行。这是因为训练上采样器时，每个训练样本都需要多次调用视觉编码器，通常还需要处理更高分辨率的图像。本文为缓解这一局限提出 *AnyUp*。它是一种学习式特征上采样方法，能够泛化到*任意维度*、*任意分辨率*的特征，同时在多种下游任务上达到当前最优性能。如表1所示，AnyUp 是首个具备这些能力的方法。

**表 1｜特征上采样方法分类。** AnyUp 是首个能在*推理时*泛化到任意输入特征的可学习方法。它可以在任意输入分辨率和任意目标分辨率之间进行上采样，并且与任务无关。

| 方法 | 任意编码器 | 任意分辨率 | 任意任务 | 可训练 |
| --- | :---: | :---: | :---: | :---: |
| 双线性上采样 | ✓ | ✓ | ✓ | ✗ |
| 最近邻上采样 | ✓ | ✓ | ✓ | ✗ |
| 增大输入图像分辨率 | ✓ | ✓ | ✓ | ✗ |
| 双边滤波（Tomasi & Manduchi, 1998） | ✓ | ✗ | ✓ | ✗ |
| 引导滤波（He et al., 2012） | ✓ | ✗ | ✓ | ✗ |
| 更改 ViT 的补丁划分步幅 | ✓ | ✗ | ✓ | ✗ |
| SAPA（Lu et al., 2022） | ✗ | ✗ | ✗ | ✓ |
| CARAFE（Wang et al., 2019） | ✗ | ✗ | ✗ | ✓ |
| DySample（Liu et al., 2023） | ✗ | ✗ | ✗ | ✓ |
| ReSFU（Zhou et al., 2025） | ✗ | ✓ | ✗ | ✓ |
| 缩放卷积（Odena et al., 2016） | ✗ | ✗ | ✓ | ✓ |
| LiFT（Suri et al., 2024） | ✗ | ✗ | ✓ | ✓ |
| FeatUp（Fu et al., 2024） | ✗ | ✗ | ✓ | ✓ |
| FeatSharp（Ranzinger et al., 2025） | ✗ | ✗ | ✓ | ✓ |
| LIIF（Chen et al., 2021） | ✗ | ✓ | ✓ | ✓ |
| LoftUp（Huang et al., 2025） | ✗ | ✓ | ✓ | ✓ |
| JAFAR（Couairon et al., 2025） | ✗ | ✓ | ✓ | ✓ |
| **AnyUp（本文）** | ✓ | ✓ | ✓ | ✓ |

特征上采样器从低分辨率特征图和高分辨率 RGB 引导图像中提取信息，以推断高分辨率图像中的每个像素应分配到什么特征（Couairon et al., 2025）。现有工作的主要局限在于，处理低分辨率特征图的方式依赖于特征的维度和类型。AnyUp 采用特征无关层，能够处理任意特征，也能泛化到新的特征类型。我们还引入了窗口注意力机制和基于裁剪的训练策略，进一步提高特征上采样质量。

实验中，AnyUp 将特征上采样的性能提升到新的最优水平。它的主要优势包括通用性和稳健的泛化能力，既可以在任意类型和分辨率的特征上训练和应用，也能泛化到训练时未见过的特征类型。此外，AnyUp 尽量减少对原有特征语义的扭曲，从而保持高保真度。我们通过消融实验分析了不同的设计选择，并总结了适用于特征上采样任务的一般经验。

综上，本文的贡献如下。

- 我们提出 AnyUp，一种在*推理时*与特征无关的上采样模型。它只需训练一次，就能对任意来源、任意分辨率和任意维度的特征进行上采样。
- 我们提出一种特征无关层，用于从不同类型和维度的特征中提取信息。
- 我们提出一种基于窗口注意力的上采样架构，可以利用基于图像局部区域的损失有效训练，并通过一致性正则化保留输入特征空间。
- 我们证明，AnyUp 的表现优于以往的上采样方法，同时能在测试时泛化到任意视觉编码器。

我们公开了代码和预训练权重，提供一个轻量、无需额外训练且易于使用的特征上采样器。[代码与预训练权重](https://github.com/wimmerth/anyup)

## 2. 相关工作

根据方法的能力和对数据的依赖程度，以往的特征上采样工作可以分为若干类，如表1所示。

无需训练的上采样方法通常适用于任意输入特征和任意任务，可以上采样到任意分辨率，计算开销也相对较小。因此，在很多需要高分辨率特征的应用中，双线性上采样等方法已成为事实上的标准选择。图2显示，引导滤波（He et al., 2012）等方法在适合的样本上也能取得出乎意料的好效果，但在低对比度或更复杂的图像上表现会下降，需要逐样本调整超参数，而且往往仍会造成过度模糊或模糊不足（见图2最后一行）。更改视觉 Transformer 的补丁划分步幅（Dosovitskiy et al., 2021），或直接提高输入图像的分辨率，会增加推理计算量，也可能使特征偏离原有分布（Ranzinger et al., 2025）。

![[AnyUp_Fig02.png]]

**图 2｜与其他方法的可视化对比。** RGB 通道分别对应在所有特征上计算得到的前三个主成分。以往方法会导致过度平滑或产生其他伪影。例如，第一行中 LoftUp 对云层特征的平滑，以及 JAFAR 对山体特征造成的分布偏移，还有第三行中 FeatUp 和引导滤波的过度平滑与光晕伪影。AnyUp（本文）输出清晰的特征图，同时保留输入特征的质量。

**图片内容解释。** 三行分别展示海港、建筑与蔬果摊场景。每行并列给出 RGB 图像、低分辨率特征图，以及 AnyUp、JAFAR、LoftUp、FeatUp、引导滤波和双线性上采样的结果。红圈标出图注所述的差异。

可学习的上采样方法（Suri et al., 2024；Fu et al., 2024；Huang et al., 2025；Couairon et al., 2025）试图在无需学习的方法基础上改善质量和稳健性。但一旦引入可学习参数，上采样模型就会针对特定的特征编码器和输入维度拟合，在推理时失去与特征的独立性。此外，一些方法仅适用于特定的下游任务或模型（Wang et al., 2019；Lu et al., 2022；Liu et al., 2023；Zhou et al., 2025），或者无法使用任意缩放因子（Odena et al., 2016；Wang et al., 2019；Lu et al., 2022；Liu et al., 2023；Suri et al., 2024；Fu et al., 2024；Ranzinger et al., 2025）。两项同期工作 JAFAR（Couairon et al., 2025）和 LoftUp（Huang et al., 2025）代表了当前最优水平。它们分别将特征上采样表述为单层和堆叠的高分辨率到低分辨率注意力。我们采用这种表述，因为它天然与分辨率无关，即可以把特征从任意分辨率上采样到任意分辨率。AnyUp 进一步尝试让这种架构与编码器无关，即上采样模型只需训练一次，之后即可应用于任意视觉编码器产生的特征，即使这些特征的维度各不相同。附录 A 提供了部分相关工作的更多细节。

## 3. 任务定义

给定输入 RGB 图像 $I_{\mathrm{hr}} \in \mathbb{R}^{H \times W \times 3}$，特征上采样方法 $f$ 的目标是将低分辨率输入特征图 $p \in \mathbb{R}^{h \times w \times c} := e(I_{\mathrm{hr}})$ 上采样为高分辨率输出特征图 $q := f(p, I_{\mathrm{hr}}) \in \mathbb{R}^{H \times W \times c}$。这一任务之所以必要，是因为近年来人们将大型预训练视觉编码器 $e(\cdot)$ 用作特征提取器和强有力的语义先验。无论是卷积神经网络（Fukushima, 1969；LeCun et al., 1989），还是视觉 Transformer（Dosovitskiy et al., 2021），几乎所有这类模型都会通过学习执行下采样，将图像映射到 $h \times w < H \times W$ 的潜在分辨率，这对于使用这些大型神经网络高效处理图像是必要的。然而，将这些模型产生的特征用于下游任务时，往往需要高分辨率特征，例如进行密集的逐像素预测，或在三维空间中聚合多个视角的信息（Engelmann et al., 2024；Ye et al., 2024；Asim et al., 2025）。

## 4. 学习与编码器无关的特征上采样

为了避免内存和计算瓶颈，我们希望使用轻量、参数少的模型。为此，我们沿用以往工作中基于注意力的架构（Couairon et al., 2025；Huang et al., 2025），如图3所示。具体来说，我们采用 JAFAR（Couairon et al., 2025）的架构，下面简要介绍。首先，输入图像 $I$ 和低分辨率特征图 $p$ 分别通过带残差连接的卷积块。接着，为图像特征加入位置编码。最后一层注意力直接根据这些像素特征计算查询，并结合下采样后的图像与低分辨率特征图的信息计算键。注意力层的值直接取自输入特征图中未经处理的补丁特征。

![[AnyUp_Fig03.png]]

**图 3｜方法概览。** AnyUp 基于窗口注意力进行上采样（第4.2节）。输入特征由特征无关层处理（第4.1节）。训练时，将从随机采样的图像局部区域计算出的特征作为参考，用于监督上采样特征图中的对应区域（第4.3节）。

**图片内容解释。** （a）展示训练策略。编码器 $e$ 分别提取裁剪区域 $I'$ 的参考特征 $\hat{q}$ 和下采样后整图 $I$ 的低分辨率特征 $p$，通过损失 $L_{\mathrm{cos\text{-}mse}}$ 比较输出 $q$ 中的对应区域 $q'$ 和参考特征。（b）展示模型架构。窗口注意力汇集查询（Q）、键（K）和值（V），输出高分辨率特征 $q$。

我们着手解决这一架构的两个主要局限。首先，原有初始特征处理层只能处理固定维度的输入，且需要为每个视觉骨干网络单独训练。我们用提出的特征无关卷积层替换它（第4.1节）。其次，我们在计算注意力时将来源特征限制在局部窗口内，以简化上采样器的任务（第4.2节）。

我们进一步改进了特征上采样方法的训练流程，引入基于图像局部区域的训练策略（第4.3.1节），并加入一致性正则化，以增强模型对噪声的稳健性并更好地保留输入特征空间（第4.3.2节）。

### 4.1 特征无关的上采样设计

在特征无关层中，我们将来自任意模型、具有任意维度的特征图与一组学习得到的基滤波器做卷积，以将它们表示为统一维度的特征图。对于基于注意力的上采样模型，每个像素的输出都是输入特征的线性组合。我们假设这样的模型主要需要掌握输入特征图中整体的局部结构变化。为提取这种结构信息，同时独立于输入特征及其维度，我们设计了一种卷积层，独立提取各输入通道的结构，再把所得信息聚合起来。图4展示了这一层的结构。

![[AnyUp_Fig04.png]]

**图 4｜特征无关层。** 该层独立处理各输入通道，然后对所有通道在各个基滤波器上的贡献取平均，因而输出通道数不随输入维度而变化。

**图片内容解释。** 图中从左到右依次为输入特征 $\mathbf{p}$、基滤波器 $\boldsymbol{\psi}$、分别对各通道的卷积结果应用 softmax，以及沿输入通道取平均后得到的输出 $\mathbf{f}$。输入有 $N$ 个通道，输出有 $M$ 个通道。

在我们提出的特征无关层中，每个输入通道 $p_i$ 都与学习得到的卷积核基 $\{\psi_j \in \mathbb{R}^{k \times k}\}_{j=1,\ldots,M}$ 进行卷积，随后沿基滤波器所在的维度对激活值执行 softmax 操作。接着，在所有输入通道上对每个滤波器的贡献取平均。写成公式，特征无关卷积层的输出特征 $f_j$ 按下式计算：

$$
f_j = \frac{1}{N} \sum_{i \in \{1,\ldots,N\}} \frac{\exp(p_i \ast \psi_j)}{\sum_{j' \in \{1,\ldots,M\}} \exp(p_i \ast \psi_{j'})},
\tag{1}
$$

其中，$N$ 是可变的输入通道数，$M$ 是统一的输出通道数。为简化式（1）的记号，我们省略了卷积运算的空间结构。

### 4.2 局部窗口注意力

分析 JAFAR（Couairon et al., 2025）的注意力模式后，我们发现，在全局注意力机制中，任意像素查询都可以关注输入特征图中的任意特征补丁，因此上采样有时会参考距离很远、几乎毫不相关的图像区域。我们认为，将注意力计算限制在查询点周围的局部窗口内（Ramachandran et al., 2019），可以避免这种情况并简化上采样问题。相较于全局注意力，局部窗口注意力只需对少得多的粗粒度特征做线性组合即可得到一个高分辨率特征，从而使上采样器的目标更容易优化，也提高了效率。上述异常注意力的可视化见附录 E。

### 4.3 训练流程

训练时，难以通过输入分辨率极高的图像来计算高分辨率特征，进而获得用于比较的参考“真值”。除了计算开销太大而难以扩展规模之外，极高分辨率的输入实际上也会使大多数模型超出原来的数据分布范围，Ranzinger et al.（2024）也指出了这一点。

为绕开这一问题，已有工作提出了不同的技术来利用低分辨率特征。FeatUp（Fu et al., 2024）将上采样器的训练表述为“多视图重建”，优化模型对图像空间微小扰动的等变性。这要求仔细设计数据增强，以免增强后的图像偏离视觉骨干网络原有的输入分布。LoftUp（Huang et al., 2025）将分割掩码用作高分辨率引导信号，代价是每一步训练都要调用大型分割模型。在最简单的情况下，例如 JAFAR（Couairon et al., 2025），训练只在低分辨率下进行，即学习如何将 $16 \times 16$ 的特征图上采样到 $32 \times 32$。两种特征图分别从按相应比例缩放的输入图像中计算得到（Suri et al., 2024；Couairon et al., 2025）。Couairon et al.（2025）认为，这种低分辨率训练足以训练基于注意力的上采样机制。我们提出了一种改进的训练策略，利用图像的局部裁剪区域，以更快的速度和更少的内存训练出更强的特征上采样器。

#### 4.3.1 数据采样

我们利用了上采样无需在整张特征图上接受监督这一点，只在较小的局部裁剪区域上监督模型。图3展示了训练流程。具体来说，我们从高分辨率图像 $I \in \mathbb{R}^{H \times W}$ 中随机采样一个较小的局部裁剪区域 $I' \in \mathbb{R}^{h \times w}$。然后，将 $I$ 下采样到相同的分辨率 $h \times w$。我们计算特征 $p = e(I)$ 和 $\hat{q} = e(I')$，再将 $p$ 上采样为高分辨率特征图 $q = f(I, p) \in \mathbb{R}^{h' \times w'}$。选择 $h'$ 和 $w'$ 时，使与 $I'$ 对应的裁剪部分 $q'$ 和 $\hat{q}$ 具有相同分辨率。LoftUp（Huang et al., 2025，第4.2节）也在第二阶段训练中提出了类似策略，不过其损失在高得多的分辨率下计算，用其方法的指数移动平均（EMA）所给出的特征代替真实特征作为参照，因此训练的计算开销大得多。与第4.3节介绍的 JAFAR（Couairon et al., 2025）的简单策略相比，我们的采样方法更轻量，因为无需在更高的分辨率（如 $448 \times 448$）下计算参考特征。

#### 4.3.2 目标函数

我们遵循以往的特征去噪和特征上采样工作（Yang et al., 2024；Couairon et al., 2025），最小化预测特征 $q'$ 与目标特征 $\hat{q}$ 之间的距离：

$$
L_{\mathrm{cos\text{-}mse}}(q', \hat{q}) = 1 - \cos(q', \hat{q}) + L^2(q', \hat{q}).
\tag{2}
$$

此外，我们加入了自一致性正则项 $L_{\mathrm{self\text{-}consistency}}$（详见附录 C.1），以及输入一致性正则项 $L_{\mathrm{input\text{-}consistency}}$。后者直接用输入特征 $p$ 和按相应比例下采样的预测输出特征 $q$ 计算 $L_{\mathrm{cos\text{-}mse}}$。这样做既能进一步增强上采样特征的局部性，也能保留输入特征空间。局部性对于表面法线估计等物体内部层级的任务很重要。

## 5. 实验

我们在 ImageNet 数据集（Krizhevsky et al., 2017）上训练 AnyUp，并在附录 C 中提供了更多实现细节。第 5.1 节将我们的方法与以往工作进行比较，评估探测质量、特征空间保持情况，以及在不同分辨率变化下的表现。第 5.2 节提供更多结果，展示 AnyUp 在测试时对其他特征的泛化能力。第 5.3 节以消融研究结束实验部分。

### 5.1 与以往方法的比较

我们主要将所提方法与 FeatUp（Fu et al., 2024），以及同期提出的 LoftUp（Huang et al., 2025）和 JAFAR（Couairon et al., 2025）进行比较。后两者都是基于注意力的通用特征上采样方法，能够从任意分辨率上采样到任意分辨率。[^anyup-1] 然而，这两个模型都有一个显著缺点：它们在测试时依然绑定各自训练时采用的特征提取器，需要针对每个视觉编码器重新训练。AnyUp 没有这个问题，因此适用范围更广，可以直接用于任何新特征。除非另有说明，实验中训练和测试所用的特征提取器均为 DINOv2 ViT-S（Oquab et al., 2023），因为其他工作也提供了用于该模型的预训练权重。

**定性分析。** 我们在图 2 中对上采样特征的 PCA 投影进行可视化，直观比较了所提方法和以往方法。我们观察到，AnyUp 在保持输入特征质量的同时，生成了边界清晰的输出特征图。相比之下，以往方法的特征图通常仍会过度平滑，或出现特征分布偏移，如图中红圈所示。

**语义分割。** 我们遵循以往工作提出的评估方案（Couairon et al., 2025，第 4.3.1 节），训练一个线性探针（即一个 $1\times1$ 卷积层），让它根据高分辨率特征图预测语义分割标签。输入图像被调整为 $448\times448$ 分辨率，因此特征分辨率为 $28\times28$ 或 $32\times32$，具体取决于用于特征提取的视觉 Transformer 的补丁大小。除非另有说明，通常上采样回输入图像分辨率，即进行 14 倍或 16 倍上采样。我们在 COCO-Stuff（Lin et al., 2014）、ADE20k（Zhou et al., 2018）和 PASCAL VOC（Everingham et al., 2014）数据集上评估性能。

结果用平均交并比（mIoU）和逐像素准确率表示。如表 2 所示，AnyUp 在下游语义分割任务中提供了当前最先进的上采样性能。

**表 2｜语义分割。** 突出显示排名第一、第二和第三的分数，依次以粗体、实线下划线和虚线下划线标记。

| 方法 | COCO mIoU (↑) | COCO Acc. (↑) | PASCAL-VOC mIoU (↑) | PASCAL-VOC Acc. (↑) | ADE20k mIoU (↑) | ADE20k Acc. (↑) |
|---|---:|---:|---:|---:|---:|---:|
| 双线性插值 | 59.48 | 79.32 | 81.43 | 95.38 | 40.54 | 74.12 |
| FeatUp | <span style="text-decoration-line:underline;text-decoration-style:dashed">61.95</span> | <span style="text-decoration-line:underline;text-decoration-style:dashed">81.14</span> | 83.37 | 96.04 | <u>42.19</u> | <span style="text-decoration-line:underline;text-decoration-style:dashed">75.57</span> |
| LoftUp | <u>62.15</u> | <u>81.32</u> | <span style="text-decoration-line:underline;text-decoration-style:dashed">83.69</span> | <span style="text-decoration-line:underline;text-decoration-style:dashed">96.11</span> | 42.02 | <u>75.72</u> |
| JAFAR | 61.82 | 81.07 | **84.36** | **96.22** | <span style="text-decoration-line:underline;text-decoration-style:dashed">42.06</span> | 75.48 |
| **AnyUp** | **62.16** | **81.37** | <u>84.00</u> | <u>96.19</u> | **42.43** | **75.85** |

**深度与表面法线估计。** 我们遵循 Probe3D（Banani et al., 2024）评估深度估计和表面法线估计的下游性能。具体而言，除非另有说明，我们将输入图像、目标深度图和目标表面法线图均调整为 $224\times224$ 分辨率。我们以均方根误差（RMSE）报告结果。对于法线估计，我们还报告预测在指定角度阈值下的准确率。对于深度估计，我们另外报告 $\delta_1$ 分数，对应预测值与真实值之比小于 1.25 的像素比例。有关基准测试和所用指标的更多细节，请参阅 Banani et al.（2024，附录 A.3）。

如表 3 所示，AnyUp 在这些任务中优于所有竞争方法，体现出它在保持上采样特征局部性方面的优势。相比之下，LoftUp 的训练目标使用了物体分割掩码，导致特征在各物体内部被过度平滑，因此在表面法线估计等任务上的表现欠佳。

**表 3｜表面法线与单目深度估计。** 在几何任务上，AnyUp 优于以往的上采样器。评估使用 NYUv2 数据集（Silberman et al., 2012）。

| 方法 | 表面法线 RMSE (↓) | 表面法线 11.25° (↑) | 表面法线 22.5° (↑) | 表面法线 30° (↑) | 绝对深度 RMSE (↓) | 绝对深度 $\delta_1$ (↑) | 相对深度 RMSE (↓) | 相对深度 $\delta_1$ (↑) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 双线性插值 | 32.70 | <span style="text-decoration-line:underline;text-decoration-style:dashed">0.26</span> | <span style="text-decoration-line:underline;text-decoration-style:dashed">0.53</span> | <span style="text-decoration-line:underline;text-decoration-style:dashed">0.66</span> | 0.4925 | 0.8081 | 0.3582 | 0.9112 |
| FeatUp | <span style="text-decoration-line:underline;text-decoration-style:dashed">32.69</span> | 0.25 | <span style="text-decoration-line:underline;text-decoration-style:dashed">0.53</span> | <span style="text-decoration-line:underline;text-decoration-style:dashed">0.66</span> | <u>0.4816</u> | <u>0.8156</u> | <u>0.3413</u> | <u>0.9193</u> |
| LoftUp | 33.94 | <span style="text-decoration-line:underline;text-decoration-style:dashed">0.26</span> | 0.51 | 0.64 | <span style="text-decoration-line:underline;text-decoration-style:dashed">0.4847</span> | <span style="text-decoration-line:underline;text-decoration-style:dashed">0.8127</span> | <span style="text-decoration-line:underline;text-decoration-style:dashed">0.3478</span> | 0.9166 |
| JAFAR | <u>31.54</u> | <u>0.28</u> | <u>0.56</u> | <u>0.68</u> | 0.4906 | 0.8052 | 0.3481 | <span style="text-decoration-line:underline;text-decoration-style:dashed">0.9180</span> |
| **AnyUp** | **31.17** | **0.29** | **0.57** | **0.69** | **0.4755** | **0.8216** | **0.3378** | **0.9233** |

![[AnyUp_Fig05.png]]

**图 5｜定性探测结果。** 对 NYUv2 上的单目深度估计和 ADE20k 上的语义分割线性探测结果进行可视化。更多可视化结果见附录 E。

**图片内容解释。** 图中上排对比不同上采样方法的语义分割结果，下排对比深度估计结果。两排从左到右均为输入图像、双线性插值、FeatUp、LoftUp、JAFAR、AnyUp 和真实标注。

**任意分辨率之间的上采样。** 我们沿用前述评估流程，但改变输入特征的分辨率和上采样的目标尺寸，后者也就是提供给上采样方法的引导图像尺寸。语义分割结果在 COCO 数据集上获得。由于 FeatUp（Fu et al., 2024）不支持改变上采样倍率，我们始终先进行 16 倍上采样，再用双线性插值下采样到所需的输出分辨率。结果见表 4。AnyUp 在大多数分辨率变化下优于竞争方法，仅在从 16 像素上采样到 112 像素时略逊于表现最好的竞争方法。

**表 4｜任意分辨率之间的上采样。** 改变特征图分辨率及输出分辨率时，语义分割（COCO）与深度估计的线性探测结果。

| 方法 | 分割 16 → 112 mIoU (↑) | 分割 16 → 112 Acc. (↑) | 分割 32 → 224 mIoU (↑) | 分割 32 → 224 Acc. (↑) | 分割 32 → 112 mIoU (↑) | 分割 32 → 112 Acc. (↑) | 深度 16 → 112 RMSE（绝对）(↓) | 深度 16 → 112 RMSE（相对）(↓) | 深度 32 → 224 RMSE（绝对）(↓) | 深度 32 → 224 RMSE（相对）(↓) | 深度 32 → 112 RMSE（绝对）(↓) | 深度 32 → 112 RMSE（相对）(↓) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 双线性插值 | 56.38 | 77.17 | 59.42 | 79.28 | 59.40 | 79.27 | 0.4927 | 0.3586 | 0.4606 | 0.3274 | <span style="text-decoration-line:underline;text-decoration-style:dashed">0.4600</span> | <span style="text-decoration-line:underline;text-decoration-style:dashed">0.3273</span> |
| FeatUp | 58.88 | 79.15 | <u>61.92</u> | <span style="text-decoration-line:underline;text-decoration-style:dashed">81.10</span> | <u>61.76</u> | <u>80.99</u> | **0.4357** | **0.3231** | <u>0.4507</u> | <u>0.3145</u> | <u>0.4513</u> | <u>0.3160</u> |
| LoftUp | <span style="text-decoration-line:underline;text-decoration-style:dashed">58.97</span> | <span style="text-decoration-line:underline;text-decoration-style:dashed">79.37</span> | 61.68 | 81.06 | 61.20 | 80.69 | 0.4896 | 0.3533 | <span style="text-decoration-line:underline;text-decoration-style:dashed">0.4591</span> | <span style="text-decoration-line:underline;text-decoration-style:dashed">0.3264</span> | 0.4636 | 0.3296 |
| JAFAR | **59.79** | **79.87** | <span style="text-decoration-line:underline;text-decoration-style:dashed">61.91</span> | <u>81.14</u> | <span style="text-decoration-line:underline;text-decoration-style:dashed">61.66</span> | <span style="text-decoration-line:underline;text-decoration-style:dashed">80.94</span> | <span style="text-decoration-line:underline;text-decoration-style:dashed">0.4871</span> | <span style="text-decoration-line:underline;text-decoration-style:dashed">0.3458</span> | 0.4825 | 0.3489 | 0.4812 | 0.3498 |
| **AnyUp** | <u>59.63</u> | <u>79.75</u> | **62.25** | **81.41** | **62.07** | **81.26** | <u>0.4746</u> | <u>0.3364</u> | **0.4441** | **0.3079** | **0.4455** | **0.3073** |

**特征空间保持。** 理想情况下，上采样后的特征应与原始低分辨率特征处于同一空间，并且服从同一分布。满足这个条件时，对同一个输入特征提取器，在原始特征空间上训练的线性探针应当能够直接用于某个上采样器输出的高分辨率特征，无需任何微调。我们使用基于原始 DINOv2（ViT-S）特征提取器（Oquab et al., 2023）预训练的线性探针，测试不同特征上采样方法在多大程度上保持了特征分布。[^anyup-2] 在表 5 中，我们对两种不同任务进行这项评估，分别是 NYUv2 数据集（Silberman et al., 2012）上的深度估计和 ADE20k 数据集（Zhou et al., 2018）上的语义分割。定性结果见附录 E。

**表 5｜特征空间保持。** 使用基于低分辨率 DINOv2 特征预训练的线性探针进行语义分割（ADE20k）和深度估计（NYUv2）。AnyUp 在提高上采样质量的同时保持了输入特征分布。LoftUp 未能保持输入特征分布，因此结果严重退化。引导滤波需要逐样本调节超参数。

| 方法 | 语义分割 mIoU (↑) | 语义分割 Acc. (↑) | 深度估计 RMSE (↓) | 深度估计 $\delta_1$ (↑) |
|---|---:|---:|---:|---:|
| 双线性插值 | <span style="text-decoration-line:underline;text-decoration-style:dashed">39.73</span> | 73.32 | 0.506 | <span style="text-decoration-line:underline;text-decoration-style:dashed">0.816</span> |
| 引导滤波 | 37.54 | 72.25 | 0.518 | 0.813 |
| FeatUp | <u>40.19</u> | <u>74.05</u> | <span style="text-decoration-line:underline;text-decoration-style:dashed">0.504</span> | <u>0.818</u> |
| LoftUp | 4.27 | 46.58 | 0.765 | 0.802 |
| JAFAR | 39.06 | <span style="text-decoration-line:underline;text-decoration-style:dashed">73.75</span> | <u>0.503</u> | 0.815 |
| **AnyUp** | **40.83** | **74.94** | **0.498** | **0.822** |

与可学习方法和启发式方法相比，AnyUp 对输入特征的保持效果最好，同时通过改善上采样提高了更高分辨率下的预测质量。LoftUp 的预测性能则严重退化，这可以由其训练采用的亲和矩阵损失解释：该损失对齐同一图像内部的特征相似度，没有直接用目标特征监督预测（Huang et al., 2025，第 4.2 节）。FeatUp（Fu et al., 2024）的隐式特征上采样版本计算成本很高，同样无法为这个上采样器专门训练探针。因此，我们在附录 B.2 中将这个版本纳入一项使用较低分辨率图像的类似实验。

**计算性能。** 我们在主要上采样操作中采用的窗口注意力可以高效实现。与以往方法 JAFAR 和 LoftUp 相比，运行时间和内存需求均减少了 50% 以上，但效率略低于 FeatUp。有关参数量、FLOPs 和运行时间，以及前向和反向传播的内存占用，详见附录 B.1。

### 5.2 特征无关的上采样

在图 1 中，我们可视化了特征的 PCA 投影。这里使用同一个特征上采样模型，即只用 DINOv2 ViT-S 特征训练的模型，来处理从 ResNet（He et al., 2016）到 DINOv3（Xue et al., 2025）等图像编码器提取的特征。在表 6 中，我们展示了 AnyUp 出色的泛化能力。使用 DINOv2 训练、在 SigLIP 2 上测试的模型，性能优于或接近直接在 SigLIP 2 上训练的模型。此外，基于 DINOv2 训练的 AnyUp 模型也很好地泛化到了 DINOv3 特征。最后，如图 6 所示，AnyUp 在不同的 DINOv2 架构之间也具有良好的泛化能力。

**表 6｜对其他模型的泛化。** 即便使用本质上不同的特征提取器进行训练，AnyUp 的性能也能达到或超过以往的上采样方法。训练与测试使用同一个特征提取器时，性能还能小幅提升。

| 方法 | 训练模型 | 测试模型 | mIoU (Acc) | RMSE（绝对 / 相对） |
|---|---|---|---:|---:|
| AnyUp | DINOv2 (ViT-S) | SigLIP 2<sup>LoftUp</sup> | <u>51.68 (73.35)</u> | <u>0.59 / 0.48</u> |
| AnyUp | SigLIP 2<sup>LoftUp</sup> | SigLIP 2<sup>LoftUp</sup> | **54.45 (75.49)** | **0.57 / 0.46** |
| LoftUp | SigLIP 2<sup>LoftUp</sup> | SigLIP 2<sup>LoftUp</sup> | 40.73 (64.87) | 0.72 / 0.60 |
| AnyUp | DINOv2 (ViT-S) | SigLIP 2<sup>JAFAR</sup> | 58.51 (78.36) | <u>0.91 / 0.58</u> |
| AnyUp | SigLIP 2<sup>JAFAR</sup> | SigLIP 2<sup>JAFAR</sup> | **60.32 (79.57)** | **0.90 / 0.57** |
| JAFAR | SigLIP 2<sup>JAFAR</sup> | SigLIP 2<sup>JAFAR</sup> | <u>60.10 (79.40)</u> | 0.93 / 0.60 |
| AnyUp | DINOv2 (ViT-S) | DINOv3 (ViT-S<sup>+</sup>) | 62.96 (81.82) | 0.51 / 0.37 |
| AnyUp | DINOv3 (ViT-S<sup>+</sup>) | DINOv3 (ViT-S<sup>+</sup>) | 62.99 (81.84) | 0.51 / 0.37 |

![[AnyUp_Fig06.png]]

**图 6｜对其他模型规模的泛化。** 对 AnyUp，我们改变上采样器训练时使用的模型，以及用于线性探测的模型（测试模型）。我们观察到：（1）无论训练时使用哪个模型，线性探测都符合 $\mathrm{ViT\text{-}L}\geq\mathrm{ViT\text{-}B}\geq\mathrm{ViT\text{-}S}$ 的总体趋势。（2）使用规模更小、能力更弱的 ViT 进行训练时，上采样质量没有显著下降。

**图片内容解释。** 四幅柱状图分别展示语义分割的 mIoU 和准确率，以及深度估计的绝对 RMSE 和相对 RMSE。横轴按训练模型分组，颜色区分测试模型。

**在多个特征提取器上训练。** 我们的特征无关上采样器架构支持使用多个视觉主干网络进行训练。我们利用这一特性开展实验，检验这样的训练能否提升模型在测试时对完全未见特征的泛化性能。实验中，我们使用以下模型的特征训练 AnyUp：DINOv2（ViT-S）、CLIP（ViT-B）、SigLIP（ViT-B）、带寄存器的 DINOv2（ViT-S），以及在 ImageNet 分类任务上训练的 ViT-B。表 7 表明，这种训练策略可以进一步提升模型对未见测试特征（即 DeiT（ViT-B）特征）的泛化能力，提高它在各个测试特征提取器上的整体表现。我们注意到，与针对该编码器训练的上采样器相比，这种策略在 DINOv2 特征上的质量略有下降。我们认为这是因为后者对该编码器特有的位置编码伪影发生了过拟合。

**表 7｜多主干网络训练。** 使用多个特征提取器训练，在已见主干网络上获得了相近的性能，同时提高了对未见主干网络的泛化能力，在跨主干网络评估中优于针对特定编码器的训练。

| AnyUp 训练所用模型 | DINOv2 mIoU (↑) | DINOv2 Acc (↑) | DINOv2 RMSE（绝对）(↓) | DINOv2 RMSE（相对）(↓) | SigLIP mIoU (↑) | SigLIP Acc (↑) | SigLIP RMSE（绝对）(↓) | SigLIP RMSE（相对）(↓) | DeiT mIoU (↑) | DeiT Acc (↑) | DeiT RMSE（绝对）(↓) | DeiT RMSE（相对）(↓) | 平均 mIoU (↑) | 平均 Acc (↑) | 平均 RMSE（绝对）(↓) | 平均 RMSE（相对）(↓) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| DINOv2 | **62.16** | **81.37** | **0.4755** | **0.3378** | 58.51 | 78.36 | 0.9111 | 0.5780 | 53.93 | 75.90 | 0.6467 | 0.497 | 58.20 | 78.54 | 0.6778 | 0.4709 |
| 多主干网络 | 62.04 | 81.30 | 0.4767 | 0.3392 | **59.52** | **79.07** | **0.9088** | **0.5776** | **54.83** | **76.56** | **0.6418** | **0.4886** | **58.80** | **78.98** | **0.6758** | **0.4685** |

### 5.3 消融研究

在表 8 中，我们发现 AnyUp 的表现优于所有消融版本。结果还表明，提出的每个组件都带来了明显影响。对数据采样进行消融时，我们用 JAFAR（Couairon et al., 2025）更简单的训练方法替换了我们的方法。

**表 8｜消融。** 移除特定模型组件或训练组件的影响。

| 变体 | mIoU (Acc.) (↑) | RMSE（绝对 / 相对）(↓) |
|---|---:|---:|
| **AnyUp** | **62.16** (**81.37**) | **0.4755** / <u>0.3378</u> |
| 移除窗口注意力（4.2） | <u>62.12</u> (<u>81.34</u>) | 0.4854 / 0.3449 |
| 移除我们的数据采样（4.3.1） | 62.03 (81.28) | <span style="text-decoration-line:underline;text-decoration-style:dashed">0.4773</span> / <span style="text-decoration-line:underline;text-decoration-style:dashed">0.3387</span> |
| 移除 $L_{\text{self-consistency}}$（4.3.2） | <span style="text-decoration-line:underline;text-decoration-style:dashed">62.09</span> (<span style="text-decoration-line:underline;text-decoration-style:dashed">81.33</span>) | <u>0.4763</u> / **0.3363** |
| 移除所有正则化（4.3.2） | 61.90 (81.23) | 0.4786 / 0.3401 |
| 移除通往键计算的特征路径 | 61.97 (81.23) | 0.4791 / 0.3441 |

值得注意的是，当我们切断上采样模型中从输入特征图到键计算的信息流（见图 3）时，仍能取得与以往特征上采样方法相当的结果，例如表 3 中绝对深度估计的 RMSE。这表明，在采用我们的训练流程时，仅依靠高分辨率图像与其下采样版本之间的位置和颜色匹配，上采样策略就已能为特征上采样提供较强的先验。

为了进一步探究这一点，并量化我们提出的特征无关层（第 4.1 节）的有效性，我们在额外实验中改变了基滤波器的数量。如表 9 所示，特征无关层中的滤波器基越大，上采样性能就越好，结果印证了我们的假设：要区分颜色相似但语义不同的图像区域，有效的上采样器还需要能够读取低分辨率特征图。我们在实现中选用的滤波器基大小为 $M=128$。

**表 9｜滤波器基大小对性能的影响。** 分数按从最好（灰色）到最差（白色）的顺序突出显示。我们单独列出此前的最佳结果作比较。

| $M$ | mIoU (↑) | Acc (↑) | RMSE（绝对）(↓) | RMSE（相对）(↓) |
|---|---:|---:|---:|---:|
| 256 | <span style="background-color:#b7b7b7;color:#000">62.19</span> | <span style="background-color:#b7b7b7;color:#000">81.38</span> | <span style="background-color:#cacaca;color:#000">0.4765</span> | <span style="background-color:#b7b7b7;color:#000">0.3371</span> |
| **128** | <span style="background-color:#c1c1c1;color:#000">62.16</span> | <span style="background-color:#bcbcbc;color:#000">81.37</span> | <span style="background-color:#b7b7b7;color:#000">0.4755</span> | <span style="background-color:#bebebe;color:#000">0.3378</span> |
| 64 | <span style="background-color:#d8d8d8;color:#000">62.09</span> | <span style="background-color:#cbcbcb;color:#000">81.34</span> | <span style="background-color:#bebebe;color:#000">0.4759</span> | <span style="background-color:#c0c0c0;color:#000">0.3380</span> |
| 32 | <span style="background-color:#dfdfdf;color:#000">62.07</span> | <span style="background-color:#d9d9d9;color:#000">81.31</span> | <span style="background-color:#e5e5e5;color:#000">0.4779</span> | <span style="background-color:#d6d6d6;color:#000">0.3402</span> |
| 16 | <span style="background-color:#d8d8d8;color:#000">62.09</span> | <span style="background-color:#d4d4d4;color:#000">81.32</span> | <span style="background-color:#dadada;color:#000">0.4773</span> | <span style="background-color:#d8d8d8;color:#000">0.3404</span> |
| 8 | <span style="background-color:#f6f6f6;color:#000">62.00</span> | <span style="background-color:#ececec;color:#000">81.27</span> | <span style="background-color:#ffffff;color:#000">0.4792</span> | <span style="background-color:#e4e4e4;color:#000">0.3415</span> |
| 0 | <span style="background-color:#ffffff;color:#000">61.97</span> | <span style="background-color:#ffffff;color:#000">81.23</span> | <span style="background-color:#fdfdfd;color:#000">0.4791</span> | <span style="background-color:#ffffff;color:#000">0.3441</span> |
| SOTA | 62.15 | 81.32 | 0.4816 | 0.3413 |

## 6. 局限性

FeatSharp（Ranzinger et al., 2025）将特征上采样与针对输入特征的位置编码伪影进行的显式去噪结合起来（Yang et al., 2024）。这种针对各个模型学习得到的输入特征去噪步骤，可以方便地加到我们流程的最前面。但截至投稿时，学习得到的去噪权重还未公开，因此我们的实验没有包含这项分析。此外，通过在局部图像上训练的策略，我们的模型实际上也学会了基本忽略这类位置编码伪影。

此外，我们基于单次图像到特征注意力的上采样方法具有较高的计算效率，性能优于以往工作，适用范围也更广。不过，它依赖一个简化假设，即上采样特征可以由低分辨率输入特征的线性组合计算出来。一个补丁特征几乎肯定会在其高维通道中编码补丁内部的空间信息，例如，可参阅尝试从补丁特征重建 RGB 图像的工作（Bordes et al., 2022）。使用更大、更复杂的上采样模型，有望提取这些信息并将其用于特征上采样。

## 7. 结论

我们提出了 AnyUp，这是首个能从任意分辨率上采样到任意分辨率，并泛化到训练时未见特征表示的方法。主要技术创新包括特征无关层、窗口注意力和一套训练策略。这些技术共同实现了当前最先进的上采样质量。我们已公开代码和模型，见 [AnyUp GitHub 仓库](https://github.com/wimmerth/anyup)。

[^anyup-1]: 根据 ICLR 指南，LoftUp 和 JAFAR 均被视为同期工作。我们仍选择主要将 AnyUp 与这两种方法及 FeatUp 比较，因为它们是当时最强的竞争方法。

[^anyup-2]: 我们注意到，较新的 DINOv3 方法（Xue et al., 2025）没有可用的预训练线性探针和其他特征上采样方法，因此我们继续使用 DINOv2。

## 附录 A 相关工作的补充细节

**FeatUp（Fu et al., 2024）。** Fu et al.（2024）提出在“多视图重建”设定下训练上采样器。借助一个学习得到的特征下采样操作，特征计算和学习得到的上采样都应当对轻微的图像增强保持等变性。FeatUp 由多个堆叠的联合双边上采样（JBU）层组成，因此只能进行整数倍上采样。

**LoftUp（Huang et al., 2025）。** Huang et al.（2025）提出堆叠的高分辨率到低分辨率注意力。其训练策略包含两个阶段。首先，使用 SAM（Kirillov et al., 2023）生成物体掩码，然后采用双三次上采样，按掩码混合低分辨率特征。在第二阶段，使用学生模型和教师模型进行自蒸馏。Huang et al.（2025）使用亲和矩阵损失，匹配同一图像内部的像素特征相似度矩阵，以代替用余弦相似度损失比较上采样预测和真实特征。这项损失在高分辨率图像上计算。

**JAFAR（Couairon et al., 2025）。** Couairon et al.（2025）用单次高分辨率到低分辨率注意力实现特征上采样。训练在低分辨率和较小的尺度变化下进行，最高为 4 倍。JAFAR 使用空间语义特征调制（Wang et al., 2018）融合下采样图像和输入特征图中的信息。我们发现这没有带来明显改善，因此改用简单的特征拼接，再接一个标准 ResNet 块。

**FeatSharp（Ranzinger et al., 2025）。** FeatSharp 建立在 FeatUp（Fu et al., 2024）之上，将输入图像切成多个图块，并对编码器模型执行多次前向传播，以获得更高分辨率的输出。消除位置编码伪影造成的偏差这一过程通过学习实现。该上采样器的权重没有公开，因此我们未将它纳入比较。

## 附录 B 补充实验

### B.1 性能分析

我们在表 10 中汇总了以运行时间和峰值内存占用衡量的性能（使用单块 NVIDIA A100 GPU 测得）。全局注意力需要 $H\times W\times h\times w$ 次计算，窗口注意力使用相对于输入大小为 $\sigma$ 的窗口，将所需计算量显著降低到仅 $H\times W\times \sigma h\times \sigma w$ 次。实际实现基于 NATTEN 库（Hassani et al., 2025）。该库针对多种架构实现了利用窗口注意力稀疏性的 CUDA 内核，我们设定 $\sigma=0.2$。我们还扩展了代码，通过分块计算注意力在内存占用和计算速度之间取舍，从而支持更大的输入图像和特征图。

**表 10｜性能分析。** AnyUp 受益于窗口注意力的稀疏性。

**（a）不同分辨率下的参数量、推理时间（ms）与 FLOPs（GFLOPs）对比。**

| 模型 | 参数量（百万） | 112² 时间（ms） | 112² FLOPs（GFLOPs） | 224² 时间（ms） | 224² FLOPs（GFLOPs） | 448² 时间（ms） | 448² FLOPs（GFLOPs） |
|---|---:|---:|---:|---:|---:|---:|---:|
| FeatUp | 0.2 | 8.6 | 2.6 | 20.6 | 10.4 | 73.7 | 41.6 |
| LoftUp | 4.3 | 8.2 | 47.1 | 33.5 | 204.1 | 201.2 | 1065.3 |
| JAFAR | 0.7 | 4.3 | 5.5 | 19.6 | 26.8 | 113.9 | 186.1 |
| **AnyUp** | 0.8 | 2.7 | 5.2 | 12.8 | 20.6 | 92.4 | 82.4 |

**（b）不同分辨率下前向和反向传播的内存占用（GB）。**

| 模型 | 112² 前向 | 112² 反向 | 224² 前向 | 224² 反向 | 448² 前向 | 448² 反向 |
|---|---:|---:|---:|---:|---:|---:|
| FeatUp | 0.3 | 0.4 | 0.9 | 1.4 | 3.4 | 5.6 |
| LoftUp | 0.2 | 0.7 | 0.8 | 2.8 | 7.9 | 21.1 |
| JAFAR | 0.2 | 0.5 | 0.6 | 2.7 | 6.9 | 22.3 |
| **AnyUp** | 0.2 | 0.8 | 0.8 | 3.3 | 3.3 | 12.9 |

### B.2 与 FeatUp 隐式上采样的比较

FeatUp 提出的隐式上采样在推理时以类似 NeRF 的方式优化高分辨率特征。计算负担过重，因此无法像第 5.1 节那样训练用于语义分割、深度估计或法线估计的线性探针。我们改为采用表 5 中的方式，分析其使用基于 DINOv2 预训练的线性探针时的性能。结果见表 11。需要注意的是，我们在 224 px 的图像上评估，而表 5 使用的是 448 px 图像，因为 FeatUp 在推理时的优化计算成本太高，难以在更高分辨率下运行。与表 5 一样，在使用预训练探针时，AnyUp 优于所有以往方法。

**表 11｜与 FeatUp（隐式）比较。** 上采样至 224 px 时，使用基于 DINOv2 ViT-S 特征预训练的线性探针比较特征空间保持效果。

| 方法 | 语义分割 mIoU (↑) | 语义分割 Acc. (↑) | 深度估计 RMSE (↓) | 深度估计 $\delta_1$ (↑) |
|---|---:|---:|---:|---:|
| 双线性插值 | 35.54 | 70.20 | <span style="text-decoration-line:underline;text-decoration-style:dashed">0.510</span> | <span style="text-decoration-line:underline;text-decoration-style:dashed">0.818</span> |
| 引导滤波 | 32.17 | 68.25 | 0.537 | 0.802 |
| FeatUp | 35.60 | <span style="text-decoration-line:underline;text-decoration-style:dashed">71.40</span> | <span style="text-decoration-line:underline;text-decoration-style:dashed">0.510</span> | 0.816 |
| → 隐式 | <span style="text-decoration-line:underline;text-decoration-style:dashed">36.70</span> | 70.95 | 0.513 | <span style="text-decoration-line:underline;text-decoration-style:dashed">0.818</span> |
| LoftUp | 3.82 | 44.35 | 0.534 | 0.803 |
| JAFAR | <u>37.52</u> | <u>72.64</u> | <u>0.505</u> | **0.822** |
| **AnyUp** | **37.91** | **72.78** | **0.502** | **0.822** |

## 附录 C 实现细节

我们在 ImageNet 数据集（Krizhevsky et al., 2017）上训练所提方法，共进行 100,000 个训练步。所选批量大小为 4，每张训练图像取 4 个随机局部裁剪。在单块 NVIDIA H100 GPU 上，训练约需 5 小时。我们使用 AdamW 优化器（Loshchilov & Hutter, 2019），学习率为 $2\times10^{-4}$，批量大小为 4。

我们注意到，第 5.1 节所报告的语义分割线性探测结果与 Couairon et al.（2025）报告的结果有所不同。这是因为与他们的实现相比，我们修复了探针训练中的一个错误。针对同期工作，我们在所有实验中均使用其官方发布的权重。

### C.1 自一致性正则化与数据增强

我们加入了先前工作（Couairon et al., 2025）也提出过的自一致性正则项 $L_{\text{self-consistency}}$，并提高数据增强的强度和多样性，以增强上采样方法的鲁棒性。图 7 展示了一批随机采样得到的增强图像。自一致性正则项的计算方式为

$$
L_{\text{self-consistency}}
= d_{\text{cos-mse}}\big(f(p,I_{\mathrm{hr}}), f(p,I'_{\mathrm{hr}})\big).
\tag{3}
$$

其中，$I'_{\mathrm{hr}}$ 是经过加噪或增强的高分辨率图像。自一致性正则项无需使用真实特征 $\hat{q}$，因此在更高的分辨率下计算，即 $224\times224$。

![[AnyUp_Fig07.png]]

**图 7｜对测试图像（左上角）应用随机采样得到的增强。**

**图片内容解释。** 左上角的绿色边框标出鹦鹉原图，其余格子显示同一图像在随机增强后的颜色、亮度和清晰度变化，其中有一幅灰度图。

## 附录 D 关于大语言模型使用情况的必需声明

除借助大语言模型调整图 6 的视觉样式之外，本文的构思和撰写均未使用大语言模型。

## 附录 E 补充可视化结果

![[AnyUp_Fig08.png]]

**图 8｜注意力伪影及其通过局部窗口注意力消除的可视化。** 不受约束的全局注意力会让上采样特征依赖距离很远且不相关的物体所提供的信息。我们给出一个简单的修正，将注意力限制在根据特征图大小确定的局部窗口内，这也简化了模型需要解决的上采样问题。

**图片内容解释。** 图中每行分别给出输入图像、输入特征图，以及不使用和使用局部窗口时的注意力图。两行对应不同像素查询，箭头标出无窗口注意力在远处产生的响应。

![[AnyUp_Fig09.png]]

**图 9｜使用预训练 DINOv2（ViT-S）探针进行语义分割的线性探测结果。** AnyUp 在上采样时保持输入特征空间，有时甚至优于真实标注，例如第 5 行的塔体分割。FeatUp 的预测通常与低分辨率预测一致，但它过度平滑了物体边界，无法得到边缘清晰的分割结果。LoftUp 在上采样时改变了特征分布，导致预训练探针产生错误预测。JAFAR 的预测会丢失细长物体的细节，例如第 2 行的灯具或第 4 行的围栏。

**图片内容解释。** 各行对比同一张输入图像的低分辨率预测、不同上采样方法的预测和真实语义标注。每行下方的颜色图例对应语义类别。

![[AnyUp_Fig10.png]]

**图 10｜使用预训练 DINOv2（ViT-S）探针进行深度估计的线性探测结果。** AnyUp 在保留引导图像清晰边缘的同时，也保留了平滑深度图预测所需的特征局部性，并保留了完整物体，例如第 4 行的白板。值得注意的是，LoftUp 在深度估计中的特征分布偏移较不明显，这在一定程度上可以归因于预测后进行的尺度与偏移对齐。

**图片内容解释。** 各行从输入图像出发，依次展示低分辨率预测、各上采样方法的深度图和真实深度。右侧色条以米为单位。

![[AnyUp_Fig11.png]]

**图 11｜使用在上采样后的 DINOv2（ViT-S）特征上训练的探针所得到的更多语义分割线性探测结果。** AnyUp 也能分割细微的结构，并给出更干净的分割结果，因此优于以往的上采样方法。

**图片内容解释。** 每行依次展示输入图像、双线性插值和各上采样方法的分割结果，以及真实语义标注。颜色条标示该图像的语义类别。

![[AnyUp_Fig12.png]]

**图 12｜使用在上采样后的 DINOv2（ViT-S）特征上训练的探针所得到的更多深度估计线性探测结果。** AnyUp 始终能给出边界清晰的物体轮廓，同时最贴近真实深度。

**图片内容解释。** 每行对比同一场景中不同上采样方法生成的深度图。左侧为输入图像，右侧为真实深度，行下方的色条对应深度数值。
