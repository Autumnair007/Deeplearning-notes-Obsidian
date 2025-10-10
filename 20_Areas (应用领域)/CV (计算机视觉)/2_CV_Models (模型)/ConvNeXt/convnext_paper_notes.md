---
type: paper-note
tags:
  - cv
  - image-classification
  - object-detection
  - instance-segmentation
  - semantic-segmentation
  - training-techniques
  - convnext
status: done
model: ConvNeXt
year: 2022
key_insight: A modern ConvNet can outperform Swin Transformer by adopting its design philosophy and training techniques.
---
论文原文：[[2201.03545\] A ConvNet for the 2020s](https://arxiv.org/abs/2201.03545)

本地pdf文件：[ConvNeXt](../../../../99_Assets%20(资源文件)/papers/ConvNeXt.pdf)

------
## 摘要

视觉识别的“咆哮的20年代”始于Vision Transformers（ViTs）的引入，它们迅速取代了ConvNets成为最先进的图像分类模型。然而，通用ViT在应用于通用计算机视觉任务（如目标检测和语义分割）时面临困难。分层Transformer（如Swin Transformer）重新引入了几个ConvNet先验，使得Transformer实际成为通用的视觉骨干网络，并在各种视觉任务中表现出色。然而，这些混合方法的有效性主要归因于Transformer内在的优越性，而非卷积固有的归纳偏置。在这项工作中，我们重新审视了设计空间，并测试了纯ConvNet所能达到的极限。我们逐步将标准ResNet“现代化”为视觉Transformer的设计，并在此过程中发现了导致性能差异的几个关键组件。这项探索的结果是一个名为ConvNeXt的纯ConvNet模型家族。ConvNeXts完全由标准ConvNet模块构建，在准确性和可扩展性方面与Transformer竞争激烈，在ImageNet上达到了87.8%的top-1准确率，并在COCO目标检测和ADE20K分割任务上优于Swin Transformer，同时保持了标准ConvNets的简洁和高效。

## 1. 引言

2010年代是深度学习取得巨大进步和影响的十年。主要驱动力是神经网络的复兴，特别是卷积神经网络（ConvNets）。在这一十年中，视觉识别领域成功地从特征工程转向了（ConvNet）架构设计。尽管反向传播训练的ConvNets早在1980年代就已经发明，但直到2012年末，我们才看到它在视觉特征学习方面的真正潜力。AlexNet的引入引发了“ImageNet时刻”，开创了计算机视觉的新时代。此后，该领域迅速发展。VGGNet、Inception、ResNe(X)t、DenseNet、MobileNet和EfficientNet等代表性ConvNets关注精度、效率和可扩展性的不同方面，并推广了许多有用的设计原则。

ConvNets在计算机视觉中的完全主导地位并非偶然：在许多应用场景中，“滑动窗口”策略是视觉处理固有的，特别是在处理高分辨率图像时。ConvNets具有几个内置的归纳偏置，使其非常适合各种计算机视觉应用。其中最重要的一个是平移等变性，这对于目标检测等任务是理想的属性。ConvNets本身也因其在滑动窗口方式下共享计算而具有固有的高效性。几十年来，这一直是ConvNets的默认用法，通常用于有限的对象类别，如数字、人脸和行人。进入2010年代，基于区域的检测器进一步将ConvNets提升为视觉识别系统中的基本构建块。

大约在同一时间，自然语言处理（NLP）的神经网络设计路径却大相径庭，Transformer取代了循环神经网络，成为主导的骨干架构。尽管语言和视觉领域之间感兴趣的任务存在差异，但这两个领域在2020年出人意料地实现了融合，Vision Transformer（ViT）的引入彻底改变了网络架构设计的格局。除了最初的“patchify”层（将图像分割成一系列图像块）外，ViT没有引入任何图像特定的归纳偏置，并且对原始NLP Transformer进行了极小的更改。ViT的一个主要焦点是缩放行为：借助更大的模型和数据集，Transformer可以显着优于标准ResNet。这些在图像分类任务上的结果令人鼓舞，但计算机视觉不仅仅局限于图像分类。如前所述，过去十年中许多计算机视觉任务的解决方案都严重依赖于滑动窗口、全卷积范式。缺乏ConvNet归纳偏置的通用ViT模型在被采纳为通用视觉骨干网络时面临许多挑战。最大的挑战是ViT的全局注意力设计，其复杂度与输入大小呈二次关系。这对于ImageNet分类可能可以接受，但对于更高分辨率的输入很快就变得难以处理。

分层Transformer采用混合方法来弥补这一差距。例如，“滑动窗口”策略（如局部窗口内的注意力）被重新引入Transformer，使其行为更类似于ConvNets。Swin Transformer是这一方向上的里程碑式工作，首次证明Transformer可以被采纳为通用视觉骨干网络，并在图像分类以外的一系列计算机视觉任务中达到最先进的性能。Swin Transformer的成功和快速普及也揭示了一点：卷积的本质并未过时；相反，它仍然备受青睐，从未褪色。

从这个角度来看，计算机视觉领域许多Transformer的进步都旨在重新引入卷积。然而，这些尝试付出了代价：滑动窗口自注意力的一种朴素实现可能成本高昂；通过高级方法（如循环移位），速度可以优化，但系统设计变得更加复杂。另一方面，具有讽l刺意味的是，ConvNet已经以直接、朴素的方式满足了许多这些期望的属性。ConvNets之所以看似失去动力，唯一的原因是（分层）Transformer在许多视觉任务中超越了它们，而性能差异通常归因于Transformer卓越的缩放行为，其中多头自注意力是关键组成部分。

与过去十年中逐步改进的ConvNets不同，Vision Transformer的采用是一个阶跃变化。在最近的文献中，系统级比较（例如Swin Transformer与ResNet）通常用于比较两者。ConvNets和分层Vision Transformer同时变得不同又相似：它们都配备了相似的归纳偏置，但在训练过程和宏/微观架构设计上存在显着差异。在这项工作中，我们研究了ConvNets和Transformer之间的架构差异，并试图找出比较网络性能时的混淆变量。我们的研究旨在弥合ConvNets在ViT前时代和ViT后时代之间的差距，并测试纯ConvNet所能达到的极限。

为此，我们从一个使用改进训练过程训练的标准ResNet（例如ResNet-50）开始。我们逐步将该架构“现代化”为分层Vision Transformer（例如Swin-T）的构造。我们的探索围绕一个关键问题：Transformer中的设计决策如何影响ConvNets的性能？在此过程中，我们发现了导致性能差异的几个关键组件。结果，我们提出了一个名为ConvNeXt的纯ConvNet模型家族。我们在ImageNet分类、COCO目标检测/分割和ADE20K语义分割等各种视觉任务上评估了ConvNeXt。令人惊讶的是，ConvNeXt完全由标准ConvNet模块构建，在准确性、可扩展性和在所有主要基准测试中的鲁棒性方面与Transformer竞争激烈。ConvNeXt保持了标准ConvNets的效率，并且训练和测试都具有完全卷积的特性，使其实现起来极其简单。

我们希望新的观察和讨论能够挑战一些普遍的观念，并鼓励人们重新思考卷积在计算机视觉中的重要性。

## 2. 卷积神经网络的现代化：路线图

在本节中，我们提供了一个从ResNet到类似Transformer的ConvNet的演变轨迹。我们考虑两种FLOPS的模型大小，一种是ResNet-50 / Swin-T体制，FLOPS约为$4.5 \times 10^9$，另一种是ResNet-200 / Swin-B体制，FLOPS约为$15.0 \times 10^9$。为简洁起见，我们将以ResNet-50 / Swin-T复杂度的模型来展示结果。更高容量模型的结论是一致的，结果可以在附录C中找到。

从高层次来看，我们的探索旨在调查和遵循Swin Transformer不同级别的设计，同时保持网络作为标准ConvNet的简洁性。我们的探索路线图如下。我们的起点是ResNet-50模型。我们首先使用与训练Vision Transformer相同的训练技术对其进行训练，并获得了比原始ResNet-50更好的结果。这将成为我们的基线。然后，我们研究了一系列设计决策，我们将其总结为：1）宏观设计，2）ResNeXt化，3）倒置瓶颈，4）大核尺寸，以及5）各种层级微观设计。图2显示了“网络现代化”的每个步骤所能达到的过程和结果。由于网络复杂度与最终性能密切相关，因此在探索过程中FLOPS大致受到控制，尽管在中间步骤中FLOPS可能高于或低于参考模型。所有模型都在ImageNet-1K上进行训练和评估。

### 2.1. 训练技术

除了网络架构的设计，训练过程也影响最终的性能。Vision Transformers不仅带来了一套新的模块和架构设计决策，还引入了不同的训练技术（例如，AdamW优化器）到计算机视觉领域。这主要涉及优化策略和相关的超参数设置。因此，我们探索的第一步是使用Vision Transformer的训练过程（在本例中为ResNet-50/200）训练一个基线模型。最近的研究表明，一套现代训练技术可以显著提升简单ResNet-50模型的性能。在我们的研究中，我们使用了一种接近DeiT和Swin Transformer的训练配方。训练周期从ResNets原始的90个epoch延长到300个epoch。我们使用AdamW优化器，数据增强技术如Mixup、Cutmix、RandAugment、Random Erasing，以及正则化方案包括Stochastic Depth和Label Smoothing。我们使用的完整超参数集可以在附录A.1中找到。

仅凭这种增强的训练配方，ResNet-50模型的性能从76.1%提高到78.8%（+2.7%），这意味着传统ConvNets和Vision Transformers之间性能差异的很大一部分可能归因于训练技术。在整个“现代化”过程中，我们将使用这个固定的训练配方和相同的超参数。ResNet-50体制下报告的每个准确率都是从使用三个不同随机种子训练的结果中获得的平均值。

### 2.2. 宏观设计

现在我们分析Swin Transformer的宏观网络设计。Swin Transformer遵循ConvNets使用多阶段设计，每个阶段具有不同的特征图分辨率。有两个有趣的设计考虑：阶段计算比和“stem cell”结构。

#### 改变阶段计算比（Changing stage compute ratio）
ResNet中计算分布在各个阶段的原始设计在很大程度上是经验性的。“res4”阶段的重型设计是为了与下游任务（如目标检测）兼容，其中检测头在14x14的特征平面上操作。Swin-T则遵循相同的原则，但阶段计算比略有不同，为1:1:3:1。对于更大的Swin Transformer，该比率是1:1:9:1。根据此设计，我们将ResNet-50中每个阶段的块数从（3，4，6，3）调整为（3，3，9，3），这也使得FLOPS与Swin-T对齐。这使得模型准确率从78.8%提高到79.4%。值得注意的是，研究人员已经彻底研究了计算分布，并且很可能存在更优化的设计。

从现在开始，我们将使用这个阶段计算比。

#### 将stem改为“Patchify”（Changing stem to “Patchify”）
通常，stem cell设计关注的是输入图像在网络开始时如何被处理。由于自然图像固有的冗余性，一个常见的stem cell会积极地对输入图像进行下采样，以在标准ConvNets和Vision Transformers中获得合适的特征图大小。标准ResNet中的stem cell包含一个步长为2的7x7卷积层，之后是一个最大池化层，这导致输入图像的下采样率为4倍。在Vision Transformers中，使用了一种更具侵略性的“patchify”策略作为stem cell，它对应于一个大核尺寸（例如，核尺寸=14或16）和非重叠卷积。Swin Transformer 使用类似的“patchify”层，但为了适应架构的多阶段设计，其patch尺寸较小，为4。我们将ResNet风格的stem cell替换为一个使用4x4、步长为4的卷积层实现的patchify层。准确率从79.4%变为79.5%。这表明ResNet中的stem cell可以用一个更简单的ViT风格的“patchify”层代替，这将带来类似的性能。

我们将在网络中使用“patchify stem”（4x4非重叠卷积）。

### 2.3. ResNeXt化

在这一部分，我们尝试采用ResNeXt的思想，它比普通的ResNet具有更好的FLOPS/准确率权衡。其核心组件是分组卷积，其中卷积核被分成不同的组。从高层次来看，ResNeXt的指导原则是“使用更多组，扩展宽度”。更精确地说，ResNeXt在瓶颈块中的3x3卷积层中使用了分组卷积。由于这显著减少了FLOPS，因此网络宽度得以扩展以弥补容量损失。

在本例中，我们使用深度可分离卷积，它是分组卷积的一种特殊情况，其中组数等于通道数。深度可分离卷积在MobileNet和Xception中得到了普及。我们注意到，深度可分离卷积类似于自注意力中的加权求和操作，它在每个通道的基础上操作，即只在空间维度上混合信息。深度可分离卷积和1x1卷积的组合导致空间和通道混合的分离，这是Vision Transformers所共有的一个特性，其中每个操作要么在空间维度上混合信息，要么在通道维度上混合信息，但不同时混合。使用深度可分离卷积有效地降低了网络的FLOPS，并且正如预期，降低了准确率。按照ResNeXt中提出的策略，我们将网络宽度扩展到与Swin-T相同的通道数（从64到96）。这使得网络性能达到80.5%，但FLOPS增加（5.3G）。

我们现在将采用ResNeXt设计。

### 2.4. 倒置瓶颈

Transformer块中一个重要的设计是它创建了一个**倒置瓶颈**，即MLP块的隐藏维度比输入维度宽四倍（参阅图4）。有趣的是，这种Transformer设计与ConvNet中使用的扩展比为4的倒置瓶颈设计有关。这个思想由MobileNetV2推广，随后在几种高级ConvNet架构中获得关注。

这里我们探讨倒置瓶颈设计。图3(a)到(b)展示了配置。尽管深度可分离卷积层的FLOPS增加，但由于下采样残差块的快捷连接中1x1卷积层FLOPS显著减少，这一改变将整个网络FLOPS降低到4.6G。有趣的是，这导致性能略有提高（80.5%到80.6%）。在ResNet-200 / Swin-B方案中，这一步带来了更大的收益（81.9%到82.6%），同时FLOPS也有所减少。

我们现在将使用倒置瓶颈。

### 2.5. 大核尺寸

在本次探索中，我们关注大卷积核的行为。Vision Transformers最显著的特点之一是其非局部的自注意力，这使得每一层都具有全局感受野。虽然大核尺寸过去曾用于ConvNets，但金标准（由VGGNet普及）是堆叠小核尺寸（3x3）的卷积层，它们在现代GPU上具有高效的硬件实现。尽管Swin Transformer重新引入了局部窗口到自注意力块中，但窗口大小至少为7x7，远大于ResNe(X)t的3x3核尺寸。这里我们重新审视在ConvNets中使用大核尺寸卷积。

#### 向上移动深度可分离卷积层（Moving up depthwise conv layer）
为了探索大核，一个先决条件是向上移动深度可分离卷积层的位置（图3(b)到(c)）。这在Transformer中也是一个显而易见的设计决策：MSA块放置在MLP层之前。由于我们有一个倒置瓶颈块，这是一个自然的设计选择——复杂/低效的模块（MSA，大核卷积）将具有更少的通道，而高效的密集1x1层将承担主要计算任务。这个中间步骤将FLOPS减少到4.1G，导致性能暂时下降到79.9%。

#### 增加核尺寸（Increasing the kernel size）
经过所有这些准备，采用更大核尺寸卷积的优势是显著的。我们尝试了几种核尺寸，包括3、5、7、9和11。网络性能从79.9%（3x3）提高到80.6%（7x7），而网络FLOPS基本保持不变。此外，我们观察到更大核尺寸的益处在7x7处达到饱和点。我们在大容量模型中也验证了这种行为：ResNet-200体制模型在核尺寸超过7x7时没有进一步的增益。

我们将在每个块中使用7x7深度可分离卷积。

至此，我们已经完成了对网络架构宏观尺度的检查。令人着迷的是，Vision Transformer中许多设计选择都可以映射到ConvNet实例。

### 2.6. 微观设计

在本节中，我们研究了几个其他微观层级的架构差异——这里的大多数探索都在层级完成，重点关注激活函数和归一化层的特定选择。

#### 将ReLU替换为GELU（Replacing ReLU with GELU）
NLP和视觉架构之间的一个小差异是使用哪种激活函数的具体性。随着时间的推移，已经开发出许多激活函数，但由于其简单性和效率，整流线性单元（ReLU）在ConvNets中仍然被广泛使用。ReLU在原始Transformer论文中也被用作激活函数。高斯误差线性单元（GELU）可以被认为是ReLU的一个更平滑的变体，它被用于最先进的Transformer，包括Google的BERT和OpenAI的GPT-2，以及最近的ViT。我们发现ReLU也可以在我们的ConvNet中被GELU替换，尽管准确率保持不变（80.6%）。

#### 更少的激活函数（Fewer activation functions）
Transformer和ResNet块之间一个微小的区别是Transformer拥有更少的激活函数。考虑一个Transformer块，它包含键/查询/值线性嵌入层、投影层以及MLP块中的两个线性层。MLP块中只有一个激活函数。相比之下，通常的做法是将激活函数附加到每个卷积层，包括1x1卷积。这里我们检查当我们坚持相同的策略时，性能如何变化。如图4所示，我们删除了残差块中除了两个1x1层之间的一个GELU层之外的所有GELU层，从而复制了Transformer块的风格。这个过程将结果提高了0.7%达到81.3%，实际上与Swin-T的性能相匹配。

我们现在将在每个块中使用一个GELU激活函数。

#### 更少的归一化层（Fewer normalization layers）
Transformer 块通常也会使用更少的归一化层。这里我们移除两个 BatchNorm (BN) 层，只保留一个 BN 层在 1x1 卷积层之前。这进一步将性能提升到 81.4%，已经超过了 Swin-T 的结果。值得注意的是，我们每个块中使用的归一化层甚至比 Transformers 更少，因为经验上我们发现，在块的开始处添加一个额外的 BN 层并不能提升性能。

#### 将BN替换为LN（Substituting BN with LN）
BatchNorm是ConvNets中一个重要的组成部分，因为它改善了收敛性并减少了过拟合。然而，BN也有许多复杂之处，可能会对模型的性能产生不利影响。尽管有许多尝试开发替代的归一化技术，但BN仍然是大多数视觉任务的首选。另一方面，更简单的Layer Normalization（LN）已被用于Transformer，在不同的应用场景中取得了良好的性能。

直接用LN替换原始ResNet中的BN会导致次优性能。通过网络架构和训练技术的修改，我们在此重新审视使用LN代替BN的影响。我们观察到我们的ConvNet模型在LN的训练中没有任何困难；事实上，性能略有提高，达到了81.5%的准确率。

从现在开始，我们将把一个LayerNorm作为我们每个残差块的归一化选择。

#### 独立下采样层（Separate downsampling layers）
在ResNet中，空间下采样是通过每个阶段开始的残差块实现的，使用步长为2的3x3卷积（以及快捷连接中的步长为2的1x1卷积）。在Swin Transformer中，在阶段之间添加了一个独立的下采样层。我们探索了类似策略，其中我们使用步长为2的2x2卷积层进行空间下采样。这种修改令人惊讶地导致了训练的发散。进一步研究表明，在空间分辨率改变的地方添加归一化层可以帮助稳定训练。这些包括Swin Transformer中也使用的几个LN层：每个下采样层之前一个，stem之后一个，以及最终全局平均池化之后一个。我们可以将准确率提高到82.0%，显著超过Swin-T的81.3%。

我们将使用独立的下采样层。这使得我们最终的模型被称为ConvNeXt。ResNet、Swin和ConvNeXt块结构的比较可以在图4中找到。ResNet-50、Swin-T和ConvNeXt-T的详细架构规范的比较可以在表9中找到。

**总结**：我们完成了第一次“演练”，并发现了ConvNeXt，一个纯ConvNet模型，在这个计算体制下，其在ImageNet-1K分类上的性能可以超越Swin Transformer。值得注意的是，目前讨论的所有设计选择都来源于Vision Transformers。此外，这些设计在ConvNet文献中也并非新颖——它们在过去十年中都被单独研究过，但没有集体研究。我们的ConvNeXt模型在FLOPs、参数数量、吞吐量和内存使用方面与Swin Transformer大致相同，但不需要专门的模块，例如移位窗口注意力或相对位置偏置。

这些发现令人鼓舞，但尚未完全令人信服——我们目前的探索仅限于小规模，但Vision Transformer的缩放行为才是真正使其与众不同之处。此外，ConvNet是否能在目标检测和语义分割等下游任务上与Swin Transformer竞争，是计算机视觉从业者关注的核心问题。在下一节中，我们将从数据和模型大小两个方面扩展我们的ConvNeXt模型，并在各种视觉识别任务上对其进行评估。

## 3. ImageNet上的实证评估

我们构建了不同的ConvNeXt变体，ConvNeXt-T/S/B/L，使其复杂性与Swin-T/S/B/L相似。ConvNeXt-T/B分别是ResNet-50/200方案中“现代化”过程的最终产物。此外，我们还构建了一个更大的ConvNeXt-XL来进一步测试ConvNeXt的可扩展性。这些变体仅在通道数C和每个阶段的块数B上有所不同。遵循ResNets和Swin Transformer，每个新阶段的通道数都会翻倍。我们总结配置如下：

* ConvNeXt-T: $C = (96, 192, 384, 768)$, $B = (3, 3, 9, 3)$
* ConvNeXt-S: $C = (96, 192, 384, 768)$, $B = (3, 3, 27, 3)$
* ConvNeXt-B: $C = (128, 256, 512, 1024)$, $B = (3, 3, 27, 3)$
* ConvNeXt-L: $C = (192, 384, 768, 1536)$, $B = (3, 3, 27, 3)$
* ConvNeXt-XL: $C = (256, 512, 1024, 2048)$, $B = (3, 3, 27, 3)$

### 3.1. 设置

ImageNet-1K数据集包含1000个对象类别，拥有1.2M张训练图像。我们报告ImageNet-1K验证集上的top-1准确率。我们还在ImageNet-22K上进行预训练，这是一个更大的数据集，包含21841个类别（ImageNet-1K的超集），ImageNet-22K包含约14M张图像。然后我们根据ImageNet-1K对预训练模型进行微调以进行评估。我们总结了我们的训练设置如下。更多细节可以在附录A中找到。

#### 在ImageNet-1K上训练
我们使用AdamW优化器，学习率为4e-3，对ConvNeXt进行300个epoch的训练。训练包含20个epoch的线性预热和随后的余弦衰减学习率调度。我们使用4096的批次大小和0.05的权重衰减。对于数据增强，我们采用常用方案，包括Mixup、Cutmix、RandAugment和Random Erasing。我们使用Stochastic Depth和Label Smoothing对网络进行正则化。应用初始值为1e-6的Layer Scale。我们使用指数移动平均（EMA），因为我们发现它能缓解大型模型的过拟合。

#### 在ImageNet-22K上预训练
我们在ImageNet-22K上对ConvNeXt进行90个epoch的预训练，并进行5个epoch的预热。我们不使用EMA。其他设置遵循ImageNet-1K。

#### 在ImageNet-1K上微调
我们将ImageNet-22K预训练的模型在ImageNet-1K上微调30个epoch。我们使用AdamW，学习率为5e-5，余弦学习率调度，分层学习率衰减，无预热，批次大小为512，权重衰减为1e-8。默认的预训练、微调和测试分辨率为$224^2$。此外，我们还在$384^2$的更高分辨率下进行微调，包括ImageNet-22K和ImageNet-1K预训练的模型。

与ViTs/Swin Transformer相比，ConvNeXts在不同分辨率下更容易微调，因为网络是全卷积的，无需调整输入patch大小或插值绝对/相对位置偏置。

### 3.2. 结果

#### ImageNet-1K
表1（上部）展示了与两个最近的Transformer变体DeiT和Swin Transformer，以及两个通过架构搜索得到的ConvNets（RegNet和EfficientNet以及EfficientNetV2）的结果比较。ConvNeXt在准确率-计算量权衡以及推理吞吐量方面与两个强大的ConvNet基线（RegNet和EfficientNet）竞争激烈。ConvNeXt也超越了复杂度相似的Swin Transformer，在整个模型系列中，有时甚至有显著的优势（例如，ConvNeXt-T高出0.8%）。由于没有使用诸如移位窗口或相对位置偏置之类的专用模块，ConvNeXt也享受了比Swin Transformer更高的吞吐量。

结果的一个亮点是384x384分辨率下的ConvNeXt-B：它比Swin-B高出0.6%（85.1% vs. 84.5%），但推理吞吐量高出12.5%（95.7 vs. 85.1 images/s）。我们注意到，当分辨率从224x224增加到384x384时，ConvNeXt-B相对于Swin-B的FLOPs/吞吐量优势变得更大。此外，当进一步扩展到ConvNeXt-L时，我们观察到结果有所改善，达到85.5%。

#### ImageNet-22K
表1（下部）展示了从ImageNet-22K预训练模型微调后的结果。这些实验很重要，因为普遍认为Vision Transformer具有较少的归纳偏置，因此在更大规模的预训练下比ConvNets表现更好。我们的结果表明，在用大型数据集预训练时，设计得当的ConvNets并不劣于Vision Transformer——ConvNeXts的性能仍然与尺寸相似的Swin Transformer相当或更好，并且吞吐量略高。此外，我们的ConvNeXt-XL模型达到了87.8%的准确率——相比于384x384分辨率下的ConvNeXt-L有显著提升，表明ConvNeXts是可扩展的架构。

在ImageNet-1K上，EfficientNetV2-L，一个配备先进模块（如Squeeze-and-Excitation）和渐进式训练过程的搜索架构，取得了最佳性能。然而，在ImageNet-22K预训练下，ConvNeXt能够超越EfficientNetV2，进一步证明了大规模训练的重要性。

在附录B中，我们讨论了ConvNeXt的鲁棒性和域外泛化结果。

### 3.3. 各向同性ConvNeXt 与 ViT

在这个消融实验中，我们检查了ConvNeXt块设计是否可以推广到ViT风格的各向同性架构，这种架构没有下采样层，并在所有深度处保持相同的特征分辨率（例如14x14）。我们使用与ViT-S/B/L相同的特征维度（384/768/1024）构建了各向同性ConvNeXt-S/B/L。深度设置为18/18/36，以匹配参数数量和FLOPs。块结构保持不变（图4）。我们使用DeiT的监督训练结果来表示ViT-S/B，使用MAE来表示ViT-L，因为它们都采用了比原始ViT改进的训练过程。ConvNeXt模型使用与以前相同的设置进行训练，但预热周期更长。ImageNet-1K上224x224分辨率的结果见表2。我们观察到ConvNeXt通常可以与ViT表现相当，这表明我们的ConvNeXt块设计在非分层模型中使用时具有竞争力。

```markdown
$$
\text{表2. 各向同性ConvNeXt与ViT的比较。训练内存在V100 GPU上测量，每GPU批处理大小为32。}
$$
```

| model                      | #param. | FLOPs | throughput (image / s) | training mem. (GB) | IN-1K acc. |
| :------------------------- | :------ | :---- | :--------------------- | :----------------- | :--------- |
| $\circ$ViT-S               | 22M     | 4.6G  | 978.5                  | 4.9                | 79.8       |
| $\bullet$ConvNeXt-S (iso.) | 22M     | 4.3G  | 1038.7                 | 4.2                | 79.7       |
| $\circ$ViT-B               | 87M     | 17.6G | 302.1                  | 9.1                | 81.8       |
| $\bullet$ConvNeXt-B (iso.) | 87M     | 16.9G | 320.1                  | 7.7                | 82.0       |
| $\circ$ViT-L               | 304M    | 61.6G | 93.1                   | 22.5               | 82.6       |
| $\bullet$ConvNeXt-L (iso.) | 306M    | 59.7G | 94.4                   | 20.4               | 82.6       |

## 4. 下游任务上的实证评估

### COCO上的目标检测和分割

我们使用ConvNeXt骨干网络在COCO数据集上微调Mask R-CNN和Cascade Mask R-CNN。遵循Swin Transformer，我们使用多尺度训练、AdamW优化器和3倍调度。更多细节和超参数设置可以在附录A.3中找到。

表3显示了Swin Transformer、ConvNeXt和传统ConvNet（如ResNeXt）在目标检测和实例分割任务上的结果比较。在不同的模型复杂度下，ConvNeXt取得了与Swin Transformer相当或更好的性能。当扩展到在ImageNet-22K上预训练的更大模型（ConvNeXt-B/L/XL）时，在许多情况下，ConvNeXt在box和mask AP方面显著优于Swin Transformer（例如，+1.0 AP）。

### ADE20K上的语义分割

我们还在ADE20K语义分割任务上使用UperNet评估了ConvNeXt骨干网络。所有模型变体都训练了16万次迭代，批处理大小为16。其他实验设置遵循BEiT（更多细节请参见附录A.3）。

在表4中，我们报告了使用多尺度测试的验证mIoU。ConvNeXt模型可以在不同模型容量下达到有竞争力的性能，进一步验证了我们架构设计的有效性。

### 模型效率备注
在相似的FLOPs下，已知使用深度可分离卷积的模型比仅使用密集卷积的ConvNets更慢且消耗更多内存。自然会有人问ConvNeXt的设计是否会使其在实践中效率低下。正如本文中所示，ConvNeXt的推理吞吐量与Swin Transformer相当或更高。这对于分类和其他需要更高分辨率输入任务都适用（参见表1和表3的吞吐量/FPS比较）。此外，我们注意到训练ConvNeXt所需的内存比训练Swin Transformer更少。例如，使用ConvNeXt-B骨干训练Cascade Mask-RCNN，在每GPU批处理大小为2的情况下，峰值内存占用为17.4GB，而Swin-B的参考数据为18.5GB。与纯ViT相比，ConvNeXt和Swin Transformer都表现出更优的准确性-FLOPs权衡，这归因于局部计算。值得注意的是，这种改进的效率是**ConvNet归纳偏置**的结果，与Vision Transformers中的自注意力机制没有直接关系。

## 5. 相关工作

### 混合模型
在ViT出现之前和之后，结合卷积和自注意力/非局部模块的混合模型一直被积极研究。ViT之前的工作主要集中于通过自注意力/非局部模块增强ConvNet以捕捉长程依赖。原始的ViT首先研究了一种混合配置，随后大量的后续工作致力于将卷积先验重新引入ViT，无论是显式地还是隐式地。

### 最近基于卷积的方法
Han et al. [25] 表明局部Transformer注意力等效于非均匀动态深度可分离卷积。Swin中的MSA块随后被替换为动态或常规深度可分离卷积，实现了与Swin相当的性能。同时期工作ConvMixer [4] 表明，在小规模设置中，深度可分离卷积可以用作一种有前途的混合策略。ConvMixer使用较小的patch尺寸以取得最佳结果，但这使吞吐量远低于其他基线。GFNet [56] 采用快速傅里叶变换（FFT）进行token混合。FFT也是一种卷积形式，但具有全局核尺寸和循环填充。

与许多近期的Transformer或ConvNet设计不同，我们研究的一个主要目标是深入探讨如何将标准ResNet现代化以实现最先进的性能。

## 6. 结论

在2020年代，Vision Transformer，特别是像Swin Transformer这样的分层模型，开始超越ConvNet成为通用视觉骨干的首选。普遍认为Vision Transformer比ConvNet更准确、更高效、更具可扩展性。我们提出了ConvNeXt，一个纯ConvNet模型，它能够在多个计算机视觉基准测试中与最先进的分层Vision Transformer竞争，同时保留了标准ConvNet的简洁性和效率。在某些方面，我们的观察结果令人惊讶，而ConvNeXt模型本身并非完全新颖——许多设计选择在过去十年中都已被单独研究过，但没有被集体研究。我们希望本研究中报告的新结果能挑战一些普遍持有的观点，并促使人们重新思考卷积在计算机视觉中的重要性。


