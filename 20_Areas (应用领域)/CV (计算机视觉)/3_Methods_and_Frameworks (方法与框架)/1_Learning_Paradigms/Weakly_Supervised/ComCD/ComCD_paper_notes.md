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
## **揭示 CLIP 和扩散模型在弱监督语义分割中的互补协同作用**

**关键词：** 弱监督语义分割 类激活图 扩散模型 CLIP

### **摘要**

弱监督语义分割 (WSSS) 仅依赖图像级标签，将类激活图 (CAM) 作为像素级种子并将其转换为伪标签进行分割。最近，一些方法利用对比语言-图像预训练 (CLIP) 或扩散模型在 WSSS 流程中生成 CAM。然而，如何将这两种范式整合到一个统一的框架中仍未得到充分探索。在这项工作中，我们提出了 ComCD (CLIP 和扩散模型的互补协同作用)，它集成了这两种范式并利用它们的互补性来提高 CAM 质量。首先，从 CLIP 分支和扩散模型分支中导出类别特定的 CAM。其次，我们设计了一种基于熵的融合，将两个 CAM 之间的熵差映射到可靠性权重，将它们融合成一个精炼的 CAM，并将其转换为伪掩膜。第三，一个带有逻辑门控模块的可训练分割网络预测权重以融合两个分支并生成最终分割。实验结果表明，所提出的 ComCD 在 WSSS 和开放词汇语义分割方面均优于最近的最新方法。

### 1.**引言**

弱监督语义分割 (WSSS) 旨在平衡标注成本与像素级预测。与需要像素级标注的完全监督方法 (Chen et al., 2017; Fu et al., 2025) 不同，WSSS 使用更经济的监督（包括点标注 (Bearman et al., 2016)、涂鸦 (Lin et al., 2016; Vernaza & Chandraker, 2017)、边界框 (Lee et al., 2021b; Oh et al., 2021) 和图像级标签 (Wu et al., 2024a,b; Yang et al., 2025b)）训练密集分割模型，从而降低数据收集和整理的开销。在这些形式中，图像级标签仅指示类别的存在而没有空间定位，因此最具挑战性。在这项工作中，我们采用图像级标签进行语义分割。

在图像级监督下，典型的多阶段流程首先训练一个图像分类器来生成类激活图 (CAM) (Zhou et al., 2016)。然后训练一个细化网络以进一步改进 CAM (Ahn et al., 2019; Ahn & Kwak, 2018)。最后，在从 CAM 导出的伪标签上训练一个分割网络 (Kweon et al., 2023; Xie et al., 2022; Yoon et al., 2024b)。为了减少流程开销，单阶段方法将这些步骤整合到一个模型中，该模型同时生成伪标签并学习像素级掩膜 (Ru et al., 2023; Wu et al., 2024b; Yang et al., 2024)。然而，由于监督不足，CAM 往往只关注判别区域，导致覆盖不完整和伪标签噪声，最终降低 WSSS 性能。最近，两种基于对比语言-图像预训练 (CLIP) 和扩散模型的新范式在 WSSS 中获得了突出地位 (Lin et al., 2023; Sun et al., 2024a; Yang et al., 2025b; Yoon et al., 2024a; Zhang et al., 2024)。每种范式的代表性方法如下。在基于 CLIP 的方法中，CLIP-ES (Lin et al., 2023) 是一种免训练、文本驱动的 Grad-CAM，可以直接定位类别判别区域，而 ExCEL (Yang et al., 2025b) 采用补丁-文本对齐来增强类别定位和边界锐度。在基于扩散的方法中，DiG (Yoon et al., 2024a) 引入预训练扩散嵌入以促进区域级连续性，而 iSeg (Sun et al., 2024a) 迭代细化扩散模型交叉注意力以产生更具空间连贯性的掩膜。

通过可视化基于 CLIP 和基于扩散的 CAM (图1(a))，我们观察到不同的行为：基于 CLIP 的方法强调类别定位，而基于扩散的方法增强空间连贯性。这些观察结果表明了潜在的互补性。为了进一步评估这一点，我们进行了一项实验 (图1(b))，比较了两个分支的预测分割掩膜，并根据与地面真实边界的内部距离描绘了准确性。具体来说，

![](../../../../../../99_Assets%20(资源文件)/images/7985f165c86a28c418b1fa39b958fbc8.png)
图1. 融合基于 CLIP 和基于扩散的 CAM 的动机。(a) 基于 CLIP 与基于扩散的 CAM。基于 CLIP 的 CAM 通过强烈激活类别判别区域来突出类别定位。基于扩散的 CAM 倾向于空间连贯性，在属于同一语义区域的像素上显示出相似的激活。符号：★ 类别定位，▲ 空间连贯性。(b) 基于 CLIP 与基于扩散的标签准确性。准确性随与 GT 边界的内部距离（像素）而变化。

令 $\partial\Omega$ 表示地面真实对象边界，$d(p)$ 是像素 $p$ 到 $\partial\Omega$ 的最短欧几里得距离；对于距离区间 $I$ 和分支 $b \in \{\text{clip, diff}\}$，我们定义内部距离准确性为 $Acc_b(I) = \frac{1}{|\Omega_I|} \sum_{p \in \Omega_I} \mathbf{1}[\hat{y}_b(p) = y^{\star}(p)]$，其中 $\Omega_I = \{p \mid d(p) \in I\}$，$y^{\star}(p)$ 是地面真实标签，$\hat{y}_b(p)$ 是分支 $b$ 的预测。曲线揭示了一个清晰的模式：对于从地面真实边界测量的内部像素，CLIP 派生标签在 100 像素内更准确，而扩散派生标签在 100 像素以上更好。直观地，这表明了一个简单的几何启发式：在边界处使用 CLIP 派生标签，在内部使用扩散派生标签。一种朴素的基于距离的方案是设置一个边界距离阈值，并通过在边界附近选择一个分支并在内部选择另一个分支来进行融合。然而，物体形状不规则且多尺度，因此使用像素级边界距离的硬阈值作为融合标准是不可靠的。因此，我们用基于不确定性的标准替换距离阈值。不确定性通过每个像素的类别概率分布的 Shannon 熵进行量化。当预测分布急剧集中时，熵低；当它接近均匀时，熵高，因此较低的熵表示较高的置信度，进而表示较高的像素级分割准确性。因此，在每个像素处，我们比较来自 CLIP 派生和扩散派生 CAM 的熵值，并给予较低熵的 CAM 更高的权重，从而避免了基于距离阈值的脆弱性并产生了更可靠的融合。
***
### 核心公式
$$Acc_b(I) = \frac{1}{|\Omega_I|} \sum_{p \in \Omega_I} \mathbf{1}[\hat{y}_b(p) = y^{\star}(p)]$$

### 1. 变量与符号拆解

#### **(1) 距离与像素集合的定义（公式的大前提）**
*   **$\partial\Omega$**：代表**真实物体的边界 (Ground Truth Boundary)**。你可以把它想象成物体的完美轮廓线。
*   **$d(p)$**：代表图像中任意一个**像素点 $p$ 到边界 $\partial\Omega$ 的最短欧几里得距离**。简单来说，就是这个像素离物体边缘有多远。如果是边缘上的点，距离为0；越往物体中心走，距离越大。
*   **$I$**：代表一个**距离区间 (Distance Interval)**。例如 $I =[20, 21)$ 像素，指的是“距离边界20到21个像素远”的这个范围。图 1(b) 的横坐标就是这个 $I$ 的变化。
*   **$\Omega_I$**：这是一个**像素集合**。数学定义是 $\Omega_I = \{p \mid d(p) \in I\}$。
    *   **大白话解释**：把所有距离物体边缘正好落在区间 $I$ 内的像素点全部挑出来，放在一起组成一个“集合”。你可以把它想象成物体内部一圈一圈的“等高线”或“环状带”。
*   **$|\Omega_I|$**：集合外面的绝对值符号代表**集合的元素个数 (基数)**。
    *   **大白话解释**：就是处于这个特定距离带（集合 $\Omega_I$）里面的**像素总数**。
#### **(2) 公式左边：所求的结果**
*   **$Acc_b(I)$**：代表在距离区间 $I$ 内，分支 $b$ 的**平均准确率 (Average Accuracy)**。
*   **$b$**：代表当前评估的是哪个**分支 (branch)** 或模型。文本中说 $b \in \{\text{clip, diff}\}$，意味着 $b$ 要么是基于 CLIP 的模型，要么是基于 Diffusion（扩散）的模型。
#### **(3) 公式右边：如何计算准确率**
*   **$y^{\star}(p)$**：代表像素 $p$ 的**真实标签 (Ground Truth Label)**。星号 ($^\star$) 通常在数学中用来表示完美的、真实的目标值。比如这个像素真的属于“人”还是“背景”。
*   **$\hat{y}_b(p)$**：代表分支 $b$ 对像素 $p$ 预测出的**分类结果 (Prediction)**。字母上面的“帽子” ($\hat{}$) 在统计和机器学习中通常代表“预测值”。
*   **$\mathbf{1}[\cdot]$**：这是一个**指示函数 (Indicator Function)**。它的规则非常简单：
    *   如果括号里的条件成立（为真），它就输出 **$1$**。
    *   如果括号里的条件不成立（为假），它就输出 **$0$**。
*   **$\mathbf{1}[\hat{y}_b(p) = y^{\star}(p)]$**：
    *   **大白话解释**：判断模型有没有预测对。如果模型预测的标签 $\hat{y}_b(p)$ 和真实的标签 $y^{\star}(p)$ 是一样的（预测正确），记为 $1$ 分；如果不一样（预测错误），记为 $0$ 分。
*   **$\sum_{p \in \Omega_I}$**：这是**求和符号**。意思是遍历集合 $\Omega_I$ 中的每一个像素 $p$。
    *   **大白话解释**：把这个“环状带”里所有预测正确的像素个数加起来（因为预测对的是1，错的是0，加起来的总和就是预测正确的总像素数）。
*   **$\frac{1}{|\Omega_I|} \times \sum (...)$**：将求和的结果除以这个区域内的总像素数。
    *   **大白话解释**：**正确的像素数 $\div$ 总像素数 = 准确率 (百分比)**。
### 2. 总结：这个公式在描述什么故事？

如果你要把这个公式翻译成计算机执行的步骤，它是这样的：

1.  **选定距离：** 比如我们现在想考察距离物体边缘“40个像素” ($I=40$) 的地方，模型表现怎么样。
2.  **圈定像素：** 找出图像中所有距离真实物体边缘刚好是40个像素的点（这就构成了集合 $\Omega_I$），并数一数一共有多少个这样的点（得到 $|\Omega_I|$）。
3.  **统计正确数：** 看 CLIP 模型（或 Diffusion 模型）对这批挑出来的像素预测得准不准。准的记1，不准的记0，把分数全加起来（得到求和项）。
4.  **计算准确率：** 把算出来的总得分除以挑出来的总像素数。

**结合论文图表 (图1b)：**
作者把距离 $I$ 从 0（最边缘）慢慢增加到 160（最内部），每变动一次距离，就用这个公式算一次 CLIP 的准确率和 Diffusion 的准确率。
*   算完后画在图上，就得出了图 1(b) 的两条曲线。
*   通过公式算出来的数据确凿地证明了：**在距离边缘 0~100 像素的外部/边缘区域，CLIP 算得准（橙线高）；而在大于 100 像素的内部核心区域，Diffusion 算得准（绿线高）。** 这为后续作者提出“结合两者优势”（基于不确定性的融合）提供了严谨的数学和实验依据。
***
在这项工作中，我们提出了 ComCD (CLIP 和扩散模型的互补协同作用)。ComCD 在两分支 WSSS 流程（基于 CLIP 的 CAM 和基于扩散的 CAM）中采用了简单有效的基于熵的融合 (EBF) 策略。对于每个像素，ComCD 计算每个 CAM 的熵，并将熵差映射到逐像素置信度权重，以指导融合。将此权重图应用于融合两个 CAM 会产生一个精炼的 CAM，该 CAM 随后被转换为伪掩膜。此外，ComCD 引入了特征对齐解码器 (FAD)，这是一个在伪掩膜下训练的分割网络。具体而言，来自 CLIP 和扩散模型的图像嵌入通过特征对齐器进行对齐，使其达到共同的空间分辨率和通道宽度。然后将对齐的嵌入输入解码器，解码器为两个分支生成分割逻辑值。逻辑门控模块 (LGM) 将这两个逻辑值作为输入，预测逐像素权重，并将它们组合成一个融合预测。FAD 在伪掩膜监督下进行训练，以生成最终的分割掩膜。实验评估了在 WSSS 和开放词汇语义分割设置下的提议方法。在 PASCAL VOC 2012 和 MS COCO 2014 的图像级监督下，伪掩膜提供了比任何一个分支单独使用更高质量的监督，并且在此掩膜上训练的解码器获得了具有竞争力的性能。在 PASCAL-Context、MS COCO-Object 和 PASCAL VOC 2012 的开放词汇设置下，该方法取得了显著成果。

主要贡献有三方面：
* 我们提出了一种简单有效的基于熵的融合，它将基于 CLIP 和基于扩散的 CAM 之间的逐像素熵差转换为可靠性权重，从而生成精炼的 CAM 和伪掩膜。
* 我们开发了带有特征对齐器和逻辑门控模块的特征对齐解码器，并在伪掩膜的监督下进行训练。
* 大量实验证明在 WSSS 基准（PASCAL VOC 2012，MS COCO 2014）上具有竞争性性能，并在 PASCAL-Context，MS COCO-Object 和 PASCAL VOC 2012 上取得了强大的开放词汇结果。

### 2.**相关工作**

**弱监督语义分割。** 带有图像级标签的弱监督语义分割 (WSSS) 通常依赖于类激活图 (CAM) 来为学习分割模型提供密集监督 (Lee et al., 2022a; Selvaraju et al., 2017; Wang et al., 2020; Wu et al., 2021; Zhou et al., 2016)。一个关键的限制是 CAM 通常强调高度判别性的区域，因此导致不完整的对象覆盖，这进一步放大了伪掩膜中的噪声和共现偏差 (Jiang et al., 2022; Lee et al., 2022a,b; Wang et al., 2020; Wu et al., 2024a; Yang et al., 2024)。为了解决这个问题，主流方法采用多阶段细化，通过亲和传播、正则化和迭代自训练逐步扩展和去噪初始线索 (Ahn et al., 2019; Ahn & Kwak, 2018; Wei et al., 2018)。为了降低此类流程的复杂性，最近的努力越来越多地转向单阶段 WSSS，它更紧密地耦合了 CAM 生成和在线细化，并直接使用细化的伪标签训练分割头，从而减少重复训练并提高效率 (Ru et al., 2023; Wu et al., 2024b; Yao et al., 2026)。最近，基础模型先验进一步重塑了 WSSS：CLIP 带来了强大的类别感知语义 (Radford et al., 2021)，而扩散模型提供了结构和形状一致的线索 (Ho et al., 2020; Rombach et al., 2022)。大多数现有研究分别利用这些先验，如下面两个小节所述。相比之下，我们的工作首次在一个 WSSS 框架中统一并共同利用扩散派生结构先验和 CLIP 派生类别感知先验。

**CLIP 用于 WSSS。** CLIP (Radford et al., 2021) 通过大规模对比预训练对齐图像和文本，这提供了类别感知语义，可以在仅有图像级标签的 WSSS 中指导伪标签生成。这种对齐激发了一系列基于 CLIP 的 WSSS 方法 (Jang et al., 2024; Lin et al., 2023; Tang et al., 2024; Xie et al., 2022; Yang et al., 2025b; Zhang et al., 2024; Zhao et al., 2024)，这些方法提高了定位质量并减少了背景混淆。CLIMS (Xie et al., 2022) 引入了语言引导的 CAM 校准，通过前景和背景提示来抑制共现 (Yang et al., 2024) 并清理伪标签。CLIP-ES (Lin et al., 2023) 通过用基于概率的 Grad-CAM 替换基于逻辑值的 Grad-CAM (Selvaraju et al., 2017) 并使用 CLIP 的 ViT 注意力细化掩膜，将冻结的 CLIP 转化为一个基本免训练的分割器。QA-CLIMS (Deng et al., 2023) 通过问题-答案提示构建自适应前景和背景文本，以扩展真实对象区域。WeCLIP (Zhang et al., 2024) 将冻结的 CLIP 视为一个强大的单阶段骨干网络，并带有一个轻量级解码器，用于高效的 WSSS。ExCEL (Yang et al., 2025b) 从全局图像-文本对齐转向密集补丁-文本对齐，并通过文本语义丰富和视觉校准来挖掘 CLIP 的细粒度先验。然而，基于 CLIP 的流水线在非判别区域中表现不佳，而扩散模型提供了空间连贯性。因此，ComCD 将 CLIP 的类别定位与扩散派生的结构先验通过自适应融合相结合。

**扩散模型用于 WSSS。** 扩散模型 (Ho et al., 2020; Rombach et al., 2022) 学习从噪声到图像的去噪过程，并暴露出丰富的注意力，编码了物体形状和语义。在 WSSS 中，这些注意力可以转换为 CAM 或用于仅使用图像级标签细化伪标签，这激发了几种基于扩散的流水线。DiG (Yoon et al., 2024a) 提出了局部性融合交叉注意力，它将预训练的扩散模型嵌入与基于 ViT 的 WSSS 模型融合，仅使用图像级标签。SeeDiff (Park et al., 2025) 使用交叉注意力作为种子，并使用多尺度自注意力对其进行扩展，以无需额外训练、提示调整或预训练分割网络即可生成高质量掩码。DiffSegmenter (Wang et al., 2025) 揭示了自注意力捕捉物体形状，而交叉注意力指示语义。iSeg (Sun et al., 2024a) 引入了迭代细化，带有一个熵减少的自注意力模块，逐步改进交叉注意力图以实现免训练分割。相反，纯扩散流水线倾向于空间连贯性，但提供的类别特异性有限。ComCD 通过 CLIP 的类别定位补充了纯扩散流水线，并使用基于熵的权重融合了这两个分支。

### 3.**方法论**

ComCD 利用预训练的 CLIP 和文本条件扩散模型获得 CAM，将它们融合成伪掩膜，然后在此伪掩膜下训练解码器以预测最终分割掩膜。3.1 节回顾了如何使用 CLIP 和扩散模型获得 CAM。3.2 节介绍了基于熵的融合，它将熵差映射到权重，将两个 CAM 融合成伪掩膜（图2，基于熵的融合）。3.3 节介绍了特征对齐解码器，这是一个在伪掩膜上训练的共享解码器，它使用逻辑门控模块来加权两个分支的逻辑值并生成最终分割（图2，特征对齐解码器）。

![[../../../../../../99_Assets (资源文件)/images/c22db2eaea2f5895218d2d98ca3ff66d.png]]

图2. ComCD 概述。给定图像和类别提示，CLIP 生成基于 CLIP 的 CAM。同时，稳定扩散模型生成基于扩散的 CAM。通过基于熵的融合计算逐像素权重，形成融合 CAM，并将其转换为伪掩膜。伪掩膜监督特征对齐解码器 (FAD)。FAD 使用预训练主干提取的特征，采用特征对齐器将两个分支对齐到共同的空间分辨率和通道宽度，使用共享解码器预测两个分支的逐像素逻辑值，并应用逻辑门控模块通过学习的逐像素权重生成最终的融合预测。

3.1. **预备知识**
3.1.1. **基于扩散的类激活图**
ComCD 使用预训练和冻结的文本条件扩散模型 (Stable Diffusion Rombach et al., 2022)。给定图像及其类别提示，ComCD 将图像编码到潜在空间，在固定时间步添加高斯噪声，并以固定采样配置执行单个反向去噪步骤以提取自注意力和交叉注意力图。这些扩散超参数遵循公开的 iSeg (Sun et al., 2024a) 配置，并在所有实验中保持固定。由于扩散骨干被冻结，并且在 CAM 生成期间时间步和采样配置都固定，因此提取的注意力图和生成的基于扩散的 CAM 对于给定输入图像是确定性的。在此步骤中，潜在图像嵌入和文本嵌入之间的交叉注意力产生了初始 CAM。此外，来自同一去噪步骤的自注意力提供了空间亲和力，并用于迭代传播和细化这些图，从而获得精炼的基于扩散的 CAM。

**交叉注意力图。** 给定图像和相应的类别提示，在预训练和冻结的文本条件扩散模型中，时间步 $t$ 的单个反向去噪步骤产生图像嵌入 $E_{\text{img}} \in \mathbb{R}^{HW \times C}$，其中 $H$ 和 $W$ 表示空间高度和宽度，$C$ 是通道维度。提示被编码成文本嵌入 $E_{\text{txt}} \in \mathbb{R}^{L \times d}$，其中 $L$ 是文本标记的数量，$d$ 是标记嵌入维度。学习的投影 $W_{\text{ca}}^q, W_{\text{ca}}^k, W_{\text{ca}}^v$ 产生查询、键和值。查询来自图像嵌入 $Q_{\text{ca}} = E_{\text{img}} W_{\text{ca}}^q \in \mathbb{R}^{HW \times d}$。键和值来自文本 $K_{\text{ca}} = E_{\text{txt}} W_{\text{ca}}^k \in \mathbb{R}^{L \times d}$ 和 $V_{\text{ca}} = E_{\text{txt}} W_{\text{ca}}^v \in \mathbb{R}^{L \times d}$。交叉注意力图为

$$
A_{\text{ca}} = \text{Softmax}\left(\frac{Q_{\text{ca}} K_{\text{ca}}^\top}{\sqrt{d}}\right) \in \mathbb{R}^{HW \times L}.
$$

遵循 iSeg (Sun et al., 2024a)，我们从几个空间尺度收集交叉注意力图，沿标记维度应用 softmax，将它们与固定权重组合，并双线性上采样到共同的低分辨率网格；所有后续细化步骤都在此聚合交叉注意力图上操作。对于每个语义类别，我们预计算对应于其类别名称的标记索引，对于类别 $c$，平均聚合交叉注意力的选定列并将结果重塑为类别感知空间图 $a_{\text{ca}}^c \in \mathbb{R}^{H_{\text{low}} \times W_{\text{low}}}$。

![](../../../../../../99_Assets%20(资源文件)/images/186cf178bfb3813f92a4615e7f8dc8ed.png)

图3. CAM 的定性比较。对于每张图像，CAM 激活所有存在的类别。(a) 图像。(b) 基于 CLIP 的 CAM。(c) 基于扩散的 CAM。(d) ComCD。(e) CLIP-ES (Lin et al., 2023)。(f) DuPL (Wu et al., 2024b)。(g) SeCo (Yang et al., 2024)。(h) WeCLIP (Zhang et al., 2024)。(i) 地面真实。

**自注意力图。** 对于图像嵌入上的自注意力，学习的投影 $W_{\text{sa}}^q, W_{\text{sa}}^k, W_{\text{sa}}^v$ 将 $E_{\text{img}}$ 映射到查询、键和值：$Q_{\text{sa}} = E_{\text{img}} W_{\text{sa}}^q$, $K_{\text{sa}} = E_{\text{img}} W_{\text{sa}}^k$, $V_{\text{sa}} = E_{\text{img}} W_{\text{sa}}^v$， $Q_{\text{sa}}, K_{\text{sa}}, V_{\text{sa}} \in \mathbb{R}^{HW \times d}$。自注意力图为

$$
A_{\text{sa}} = \text{Softmax}\left(\frac{Q_{\text{sa}} K_{\text{sa}}^\top}{\sqrt{d}}\right) \in \mathbb{R}^{HW \times HW}.
$$

在实践中，我们采用 $H_{\text{low}} \times W_{\text{low}}$ 潜在分辨率，使得 $A_{\text{sa}}$ 的空间索引与 $a_{\text{ca}}^c$ 的空间索引对齐；这种注意力捕获长程空间亲和力并有利于区域连接性。

**自注意力的迭代细化。** 对于类别标记 $c$，上面描述的聚合交叉注意力图提供了初始的类别特定图 $a_{\text{ca}}^c$，它在 64×64 网格上突出显示与类别 $c$ 最相关的空间区域。遵循 iSeg (Sun et al., 2024a)，我们然后使用自注意力作为空间亲和力在一个传播步骤中细化此图。具体来说，$a_{\text{ca}}^c$ 被向量化为 $m_c \in \mathbb{R}^{H_{\text{low}}W_{\text{low}}}$，并且相同分辨率的自注意力被重塑为空间-空间亲和矩阵 $A_{\text{sa}} \in \mathbb{R}^{H_{\text{low}}W_{\text{low}} \times H_{\text{low}}W_{\text{low}}}$。然后我们计算

$$
\tilde{m}_c = A_{\text{sa}} m_c,
$$

它沿着高亲和力空间邻居传播类别分数。最后，$\tilde{m}_c$ 被重塑回 $\mathbb{R}^{H_{\text{low}} \times W_{\text{low}}}$，双线性上采样到 $(H, W)$，并归一化以获得精炼的基于扩散的 CAM $\mathcal{H}_{\text{diff}}^c \in \mathbb{R}^{H \times W}$。这个细化步骤提高了空间连贯性并抑制了虚假孤立响应，而无需引入额外的可训练参数。

3.1.2. **基于 CLIP 的类激活图**
CLIP (Radford et al., 2021) 通过在大型图像-文本对上进行对比预训练来对齐图像和文本。它由图像编码器（通常是 Vision Transformer Dosovitskiy et al., 2020）和文本编码器组成，两者都将输入投影到共享嵌入空间中，其中余弦相似度反映语义对齐。

输入图像被馈送到 CLIP 图像编码器以从选定层获得补丁嵌入 $E_{\text{img}} \in \mathbb{R}^{HW \times D}$。线性投影 $W_Q, W_K, W_V$ 产生 $Q = E_{\text{img}} W_Q$, $K = E_{\text{img}} W_K$, 和 $V = E_{\text{img}} W_V$。为了符号简洁，我们省略了注意力头维度，并写成 $Q, K, V \in \mathbb{R}^{HW \times D}$，其中 $D$ 是投影通道宽度。

与标准自注意力（其中 $Q$ 和 $K$ 来自不同的投影 $W_Q$ 和 $W_K$）不同，$Q$ 被用于查询和键，即 $Q_{\text{sa}} = K_{\text{sa}} = Q$，并且注意力采用 $A_{QQ} = \text{Softmax}\left(Q Q^\top / \sqrt{D}\right) \in \mathbb{R}^{HW \times HW}$ 的形式。类似地，用 $(K, K)$ 和 $(V, V)$ 替换 $(Q_{\text{sa}}, K_{\text{sa}})$ 产生 $A_{KK}$ 和 $A_{VV}$。ComCD 将空间亲和力定义为 $\mathcal{A} = \frac{1}{3}(A_{QQ} + A_{KK} + A_{VV})$。将 $\mathcal{A}$ 应用于值 $V$ 产生 $\tilde{E}_{\text{img}} = \mathcal{A} V$。然后 ComCD 将 $\tilde{E}_{\text{img}}$ 重塑为 $\hat{E}_{\text{img}} \in \mathbb{R}^{H \times W \times D}$ 以获得逐像素嵌入。最后，给定类别 $c$ 的文本嵌入 $E_{\text{txt}}^c \in \mathbb{R}^D$，CAM 通过余弦相似度计算：

$$
\mathcal{H}_{\text{clip}}^c = \cos\left(\frac{\hat{E}_{\text{img}}}{\|\hat{E}_{\text{img}}\|_2}, \frac{E_{\text{txt}}^c}{\|E_{\text{txt}}^c\|_2}\right) \in \mathbb{R}^{H \times W},
$$

其中 $\cos(\cdot,\cdot)$ 表示余弦相似度。

***
### 1. 为什么不用标准的 $QK^\top$？
在标准的 Transformer 自注意力中，注意力分数是 $QK^\top$：
* **Query (Q)** 代表“我需要什么信息”。
* **Key (K)** 代表“我包含什么信息”。
* $QK^\top$ 是一种**非对称**的匹配，为了找出信息该如何“流动”。

**但在 CLIP 的场景下，直接用 $QK^\top$ 提取掩码有个致命缺陷**：
CLIP 原本是为了“图像分类”（整张图匹配一段文本）预训练的。它的标准注意力网络习惯于把所有注意力集中在图像中最具判别性的一个点（比如只看狗的鼻子），而忽略其他部分。如果你直接拿 $QK^\top$ 出来的特征去生成 CAM，得到的掩码会非常破碎，也就是你之前担心的 **“内部空洞”** 和 **“不完整”**。

### 2. $QQ^\top$ 到底在算什么？（公式拆解）

为了解决上述问题，作者抛弃了标准的注意力流动，转而**测量图像 Patch 之间的物理/语义相似度**。

**① 投影到子空间**
图像特征 $E_{\text{img}}$ 分别乘上权重矩阵，得到了 $Q, K, V$。你可以把这看作是从三个不同的角度（特征子空间）去观察这批图像 Patch。

**② 计算自相似度矩阵**
公式：$A_{QQ} = \text{Softmax}\left(\frac{Q Q^\top}{\sqrt{D}}\right)$
* 这里的 $Q \in \mathbb{R}^{HW \times D}$，$Q^\top \in \mathbb{R}^{D \times HW}$。
* $Q Q^\top$ 得到的是一个 $HW \times HW$ 的矩阵。
* **物理含义**：矩阵的第 $(i, j)$ 个元素，就是第 $i$ 个 Patch 和第 $j$ 个 Patch 在 $Q$ 空间下的**点积（即余弦相似度的大小）**。它回答的问题是：“在这个特征空间里，Patch $i$ 和 Patch $j$ 长得有多像？”
* 同理，$A_{KK}$ 是在 $K$ 空间下看它们有多像，$A_{VV}$ 是在 $V$ 空间下看它们有多像。

**③ 融合亲和度矩阵**
公式：$\mathcal{A} = \frac{1}{3}(A_{QQ} + A_{KK} + A_{VV})$
* 作者认为，单看某一个空间可能不准。于是把三个空间算出来的相似度加起来取平均，得到一个极其鲁棒的**全局亲和度矩阵 $\mathcal{A}$**。
* $\mathcal{A}$ 记录了图像中任意两个 Patch 之间综合的“长相相似度”。（比如：属于“狗身”的 Patch 和属于“狗腿”的 Patch 相似度会很高，但它们和“草地”的相似度很低）。

**④ 特征平滑/传播（填补空洞）**
公式：$\tilde{E}_{\text{img}} = \mathcal{A} V$
* 这是最关键的一步！这本质上是一个**图卷积（Graph Convolution）** 或**信息平滑**过程。
* 它让每一个 Patch 的特征，都去融合那些跟它“长得像”（$\mathcal{A}$ 值高）的 Patch 的特征。
* **结果**：原本 CLIP 可能只在“狗头”处有强烈的特征响应。经过这一步相乘，狗头的高质量特征会顺着亲和度矩阵 $\mathcal{A}$，流向相似的“狗身”和“狗腿”。**这完美地解决了 CLIP 固有激活破碎、不完整的问题。**

### 3. 最后一步：生成基于 CLIP 的 CAM
公式：
$$
\mathcal{H}_{\text{clip}}^c = \cos\left(\frac{\hat{E}_{\text{img}}}{\|\hat{E}_{\text{img}}\|_2}, \frac{E_{\text{txt}}^c}{\|E_{\text{txt}}^c\|_2}\right)
$$
经过上面的平滑操作后，我们得到了“内部特征一致”且“饱满”的图像像素级特征 $\hat{E}_{\text{img}}$。
最后，只需要让每个像素的特征，去和文本提示（比如“狗”的文本向量 $E_{\text{txt}}^c$）算一下最简单的余弦相似度（$\cos$）。相似度高的像素亮起，就得到了一张完整、高质量的 CLIP CAM（对应图 2 里的左边那张热力图）。

### 总结
你直觉非常准，它**不是**在做注意力机制。它是在借用注意力里的 $Q,K,V$ 投影矩阵，去强行算一个 **“图像内部各部位的相似度矩阵”**，从而把 CLIP 集中在局部的散碎特征，平滑（涂抹）成一个完整的物体形状。这是目前无监督/弱监督分割领域非常经典的一种“驯服 CLIP”的手段。

***

3.1.3. **基于扩散的 CAM 与基于 CLIP 的 CAM**
前两段描述了我们的框架中如何获得基于扩散的 CAM 和基于 CLIP 的 CAM。通过在图 1(a) 中可视化它们，我们观察到明显的行为：基于扩散的 CAM 倾向于空间连贯性，产生更平滑、更完整的区域激活，更好地保留了对象结构 (Sun et al., 2024a)，而基于 CLIP 的 CAM 倾向于类别定位，强烈突出类别判别部分，并在对象边界周围产生更锐利的响应，同时在非判别区域保持稀疏 (Yang et al., 2025b)。

关于常用的 CAM 获取方法，基于 CLIP 的 CAM 通常通过将图像-文本相似度分数归因于空间标记来获得，而基于扩散的 CAM 通常从文本条件扩散 U-Net 注意力中提取。

3.2. **基于熵的融合**
在第3.1节的基础上，ComCD 构建了一个基于熵的融合，将基于 CLIP 的 CAM $\left\{\mathcal{H}_{\text{clip}}^c\right\}_{c \in \mathcal{C}_{\text{img}}}$ 和基于扩散的 CAM $\left\{\mathcal{H}_{\text{diff}}^c\right\}_{c \in \mathcal{C}_{\text{img}}}$ 结合起来。正如在图 1(b) 中观察到的，基于 CLIP 的 CAM 倾向于在对象边界周围更精确，而基于扩散的 CAM 在对象内部具有更高的空间连贯性；因此，EBF 被设计为一种逐像素规则，它决定哪个分支在每个位置更可靠，并相应地加权它们的贡献。给定输入图像，令 $\mathcal{C}_{\text{img}}$ 表示图像中存在的类别的集合（$|\mathcal{C}_{\text{img}}|$ 是类别的数量）。堆叠每个类别的 CAM 产生 $\mathcal{H}_b \in \mathbb{R}^{|\mathcal{C}_{\text{img}}| \times H \times W}$，其中 $b \in \{\text{clip, diff}\}$。第一步，每个分支通过沿类别维度应用 softmax 转换为像素级类别分布：

$$
\mathcal{P}_b^c = \frac{\exp(\mathcal{H}_b^c)}{\sum_{c' \in \mathcal{C}_{\text{img}}} \exp(\mathcal{H}_b^{c'})}.
$$

在每个空间位置 $(h, w)$ 处，熵定义为分支 $b \in \{\text{clip, diff}\}$ 引起的逐像素类别概率分布 $\left\{\mathcal{P}_b^{c,h,w}\right\}_{c \in \mathcal{C}_{\text{img}}}$ 的 Shannon 熵：

$$
e_b^{h,w} = - \sum_{c \in \mathcal{C}_{\text{img}}} \mathcal{P}_b^{c,h,w} \log \mathcal{P}_b^{c,h,w}.
$$

其中 $\sum_{c \in \mathcal{C}_{\text{img}}} \mathcal{P}_b^{c,h,w} = 1$ 对于每个 $(h, w)$。预测熵是基于 softmax 的分类器的标准不确定性度量：对于固定的标签集，较低的熵对应于更尖锐的分布，其中一个类别主导其他类别。在实践中，这种低熵预测在分类和分割任务中经验性地与更高的准确性相关联。在 ComCD 中，我们从相对意义上利用这一特性：当在同一像素上比较 CLIP 和扩散分支时，熵较低的分支被视为更可靠。这种设计与图 1(b) 中的边界-内部曲线一致，其中 CLIP 分支在对象边界附近更自信且更准确，而扩散分支在对象内部更自信且更准确。与最大概率、裕度或方差等替代置信度代理相比，熵的实际优势在于它直接从单个逐像素 softmax 分布计算，并提供了一个分布锐度的单调标量摘要，而不管类别的数量。在我们伪掩膜生成阶段，使用冻结的 CLIP 和扩散骨干，这使得熵成为一个特别方便和轻量级的选择。

基于这种相对置信度观点，计算差异 $\Delta e_{h,w} = e_{\text{clip}}^{h,w} - e_{\text{diff}}^{h,w}$ 并将其用作逐像素可靠性标准。较低的熵表示较高的置信度。因此，$\Delta e_{h,w} < 0$ 在 $(h, w)$ 处赋予 CLIP 更大的置信度，而 $\Delta e_{h,w} > 0$ 倾向于扩散模型分支。将 $\Delta e$ 通过 sigmoid 得到逐像素权重图 $W = \sigma(\Delta e)$，其中 $\sigma(\cdot)$ 表示 sigmoid 函数。融合 CAM 通过元素级乘法定义为

$$
\mathcal{H}_{\text{fuse}}^c = W \odot \mathcal{P}_{\text{diff}}^c + (\mathbf{1} - W) \odot \mathcal{P}_{\text{clip}}^c.
$$

其中 $\odot$ 表示元素级乘法，$\mathbf{1} \in \mathbb{R}^{H \times W}$ 是全一图，即每个元素都等于 1。该表达式给出了类别 $c$ 在所有像素上的融合图。

最终掩膜 $\mathcal{M}$ 在图像特定类别集 $\mathcal{C}_{\text{img}}$ 上通过逐像素 argmax 获得：

$$
\mathcal{M} = \underset{c \in \mathcal{C}_{\text{img}}}{\text{arg max}} \, \mathcal{H}_{\text{fuse}}^c.
$$

除了作为伪掩膜直接评估外，最终掩膜 $\mathcal{M}$ 还可以作为监督来训练分割网络。

***
这一节的核心思想非常直观：**“谁更自信，就听谁的”**。前面提到，CLIP 生成的特征图（CAM）在**物体边缘**比较准，而 Diffusion（扩散模型）生成的特征图在**物体内部**比较准。那么对于图像上的每一个像素，我们到底该用 CLIP 的结果，还是用 Diffusion 的结果呢？作者引入了信息论中的 **“熵（Entropy）”** 来作为衡量“自信程度”的指标。下面我们逐个拆解这些公式：

### 1. 将分数转化为概率 (Softmax)
$$
\mathcal{P}_b^c = \frac{\exp(\mathcal{H}_b^c)}{\sum_{c' \in \mathcal{C}_{\text{img}}} \exp(\mathcal{H}_b^{c'})}
$$
*   **背景知识**：模型输出的原始特征图（$\mathcal{H}$）里面的数值大小是不统一的（比如有的是 5.2，有的是 -1.3），这种值叫做 Logit。为了方便比较，我们需要把它变成**概率**。
*   **公式解释**：这就是经典的 Softmax 公式。它把某个像素点在所有类别上的原始分数，压缩到了 $0$ 到 $1$ 之间，并且保证所有类别的概率加起来等于 $1$。
*   **物理含义**：算完之后，$\mathcal{P}_b^c$ 就代表了模型分支 $b$（CLIP 或 Diffusion）认为当前这个像素属于类别 $c$ 的概率是多少。

### 2. 计算“混乱度” (香农熵 Shannon Entropy)
$$
e_b^{h,w} = - \sum_{c \in \mathcal{C}_{\text{img}}} \mathcal{P}_b^{c,h,w} \log \mathcal{P}_b^{c,h,w}
$$
*   **背景知识**：在信息论中，“熵”用来衡量一个系统的不确定性（混乱度）。
    *   **高熵（非常混乱）**：假设一个像素，模型觉得它 50% 是狗，50% 是猫。这个时候模型是**懵逼（不自信）** 的，算出来的熵会很**大**。
    *   **低熵（非常确定）**：假设模型觉得它 99% 是狗，1% 是猫。这个时候模型非常**笃定（自信）**，算出来的熵会很**小**。
*   **公式解释**：对于位置 $(h,w)$ 上的像素，我们把刚才算出的概率套入香农熵公式。算出的 $e_b$ 越小，说明这个模型分支在这个像素上越自信。

### 3. 比较谁更自信 (计算差异与权重)
这部分没有单独的公式大块，但逻辑非常关键：
*   **计算差异**：$\Delta e = e_{\text{clip}} - e_{\text{diff}}$
    *   如果 $\Delta e < 0$：说明 CLIP 的熵更小，**CLIP 更自信**（说明这个像素可能在边缘）。
    *   如果 $\Delta e > 0$：说明 Diffusion 的熵更小，**Diffusion 更自信**（说明这个像素可能在物体内部）。
*   **生成权重图**：$W = \sigma(\Delta e)$
    *   **数学知识**：$\sigma$ 是 Sigmoid 函数，它可以把任何实数映射到 $0$ 到 $1$ 之间。$\Delta e$ 越大（Diffusion 越好），$W$ 越接近 $1$；$\Delta e$ 越小（CLIP 越好），$W$ 越接近 $0$。

### 4. 融合两个模型的结果
$$
\mathcal{H}_{\text{fuse}}^c = W \odot \mathcal{P}_{\text{diff}}^c + (\mathbf{1} - W) \odot \mathcal{P}_{\text{clip}}^c
$$
*   **公式解释**：这是一个典型的**加权平均**公式。$\odot$ 代表逐像素相乘。
    *   如果 Diffusion 更自信（$W$ 接近 $1$），那么 $1-W$ 就接近 $0$。此时等式主要由前面一半主导，最终结果倾向于 Diffusion。
    *   如果 CLIP 更自信（$W$ 接近 $0$），那么 $1-W$ 接近 $1$。此时等式主要由后面一半主导，最终结果倾向于 CLIP。
*   **物理含义**：通过这种动态权重分配，完美结合了 CLIP 懂边缘、Diffusion 懂内部的优势。

### 5. 生成最终的伪掩膜 (Argmax)
$$
\mathcal{M} = \underset{c \in \mathcal{C}_{\text{img}}}{\text{arg max}} \, \mathcal{H}_{\text{fuse}}^c
$$
*   **公式解释**：`arg max` 的意思是“寻找最大值对应的索引（类别）”。
*   **物理含义**：在融合了两个模型的概率之后，对于每一个像素，我们看看它在哪个类别上的融合概率最高，就把这个像素最终划分为那个类别。这就生成了一张包含具体类别标签的图像，也就是我们用来训练下游网络的 **“伪掩膜（Pseudo-mask）”**。
***
### 具体带张量变化例子

为了方便计算，我们假设一张极小的图片，只看其中**1个特定的像素**（比如位置在 $h=1, w=1$，假设这个像素在“狗”的**边缘**）。
我们假设这张图片里一共只有 **2 个类别**（$|\mathcal{C}_{\text{img}}| = 2$）：
*   类别 0：狗（Dog）
*   类别 1：背景（Background）
### 第一步：获取前置数据 (原始 Logits)
**数据从哪来？** 
这正是 **3.1 节**里算出来的结果。CLIP 分支（3.1.2节）和 Diffusion 分支（3.1.1节）分别输出了两张原始的热力图（CAM），符号是 $\mathcal{H}_{\text{clip}}$ 和 $\mathcal{H}_{\text{diff}}$。
**张量形状**：$\mathbb{R}^{2 \times H \times W}$（2个类别）。

在这个像素 $(1,1)$ 上，假设前置步骤算出的原始分数（Logits）是：
*   **CLIP 分支（懂边缘）** 发现这是狗的边缘，很自信：
    *   狗的分数：$\mathcal{H}_{\text{clip}}^{\text{dog}} = 2.0$
    *   背景分数：$\mathcal{H}_{\text{clip}}^{\text{bg}} = 0.0$
*   **Diffusion 分支（不懂边缘）** 在边缘处比较模糊，很犹豫：
    *   狗的分数：$\mathcal{H}_{\text{diff}}^{\text{dog}} = 1.0$
    *   背景分数：$\mathcal{H}_{\text{diff}}^{\text{bg}} = 0.8$
### 第二步：将原始分数转化为概率 (Softmax)
**公式**：$\mathcal{P}_b^c = \frac{\exp(\mathcal{H}_b^c)}{\sum_{c'} \exp(\mathcal{H}_b^{c'})}$
**张量变化**：形状不变，依然是 $\mathbb{R}^{2 \times H \times W}$，但数值从任意实数变成了 $0 \sim 1$ 之间的概率，且沿类别维度相加为 $1$。

**数值计算**：
*   **对于 CLIP 分支**：
    *   狗的概率：$\mathcal{P}_{\text{clip}}^{\text{dog}} = \frac{e^{2.0}}{e^{2.0} + e^{0.0}} = \frac{7.389}{7.389 + 1} \approx \mathbf{0.88}$ (88%)
    *   背景概率：$\mathcal{P}_{\text{clip}}^{\text{bg}} = \frac{e^{0.0}}{e^{2.0} + e^{0.0}} = \frac{1}{8.389} \approx \mathbf{0.12}$ (12%)
*   **对于 Diffusion 分支**：
    *   狗的概率：$\mathcal{P}_{\text{diff}}^{\text{dog}} = \frac{e^{1.0}}{e^{1.0} + e^{0.8}} = \frac{2.718}{2.718 + 2.225} \approx \mathbf{0.55}$ (55%)
    *   背景概率：$\mathcal{P}_{\text{diff}}^{\text{bg}} = \frac{e^{0.8}}{e^{1.0} + e^{0.8}} = \frac{2.225}{4.943} \approx \mathbf{0.45}$ (45%)
### 第三步：计算每个分支的“香农熵”（混乱度）
**公式**：$e_b = - \sum \mathcal{P}_b^c \log \mathcal{P}_b^c$
**张量变化**：因为把类别维度求和消掉了，张量从 $\mathbb{R}^{2 \times H \times W}$ 塌缩成了 $\mathbb{R}^{H \times W}$（每个像素只剩下一个代表混乱度的标量）。

**数值计算**（我们这里用自然对数 $\ln$）：
*   **CLIP 的熵**（很确信，所以熵应该很小）：
    $e_{\text{clip}} = -(0.88 \times \ln 0.88 + 0.12 \times \ln 0.12) = -(-0.112 - 0.254) = \mathbf{0.366}$
*   **Diffusion 的熵**（很犹豫，55开，所以熵应该很大）：
    $e_{\text{diff}} = -(0.55 \times \ln 0.55 + 0.45 \times \ln 0.45) = -(-0.328 - 0.359) = \mathbf{0.687}$
### 第四步：计算熵差与权重图
**过程**：先算差值 $\Delta e = e_{\text{clip}} - e_{\text{diff}}$，再过 Sigmoid 函数 $W = \frac{1}{1 + e^{-\Delta e}}$。
**张量变化**：保持 $\mathbb{R}^{H \times W}$，代表每个像素上，Diffusion 模型应该占多大的权重（$0 \sim 1$之间）。

**数值计算**：
1.  **算差值**：$\Delta e = 0.366 - 0.687 = \mathbf{-0.321}$ 
    *(因为算出来是负数，代表 CLIP 的熵更小，CLIP 更自信)*
2.  **算权重**：$W = \frac{1}{1 + e^{-(-0.321)}} = \frac{1}{1 + 1.378} \approx \mathbf{0.42}$
    *(注意：公式里的 $W$ 是给 Diffusion 的权重，因为 $\Delta e$ 是 CLIP减去Diff。$W=0.42$ 意味着在这个像素上，最终结果将由 $42\%$ 的 Diffusion 和 $58\%$ 的 CLIP 组成。)*
### 第五步：生成融合概率 (加权平均)
**公式**：$\mathcal{H}_{\text{fuse}}^c = W \odot \mathcal{P}_{\text{diff}}^c + (\mathbf{1} - W) \odot \mathcal{P}_{\text{clip}}^c$
**张量变化**：重新回到 $\mathbb{R}^{2 \times H \times W}$，我们得到了融合两个模型优点的最终概率。

**数值计算**（带入前面的权重和概率）：
*   **融合后“狗”的概率**：
    $0.42 \times 0.55$ (Diff的贡献) $+ (1 - 0.42) \times 0.88$ (CLIP的贡献) 
    $= 0.231 + 0.510 = \mathbf{0.741}$ (74.1%)
*   **融合后“背景”的概率**：
    $0.42 \times 0.45$ (Diff的贡献) $+ (1 - 0.42) \times 0.12$ (CLIP的贡献) 
    $= 0.189 + 0.070 = \mathbf{0.259}$ (25.9%)
*(注：0.741 + 0.259 = 1.0，概率守恒)*
### 第六步：生成最终伪掩膜 (Argmax)
**公式**：$\mathcal{M} = \text{arg max} \, \mathcal{H}_{\text{fuse}}^c$
**张量变化**：从 $\mathbb{R}^{2 \times H \times W}$ 变成 $\mathbb{R}^{H \times W}$。里面的数值不再是概率，而是**整数类别的索引**（0 或 1）。

**数值计算**：
比较融合后的概率，狗($0.741$) > 背景($0.259$)。
**最终结果**：在这个像素 $(1,1)$ 上，$\mathcal{M} = \mathbf{0}$（代表判定为“狗”）。

### 总结
这就是论文里整个 3.2 节所做的事情。原本在这个边缘像素上，Diffusion 分支快要搞错了（狗 55%，背景 45%）。但是通过**计算熵**，系统发现 CLIP 在这里更自信（熵只有 0.366），于是自动分配了更多的权重（58%）给 CLIP。最终成功纠正了偏差，在伪掩膜 $\mathcal{M}$ 里给出了正确的类别 0。
***

3.3. **特征对齐解码器**
为了进一步探索 CLIP 扩散融合的潜力，在伪掩膜 $\mathcal{M}$ 的监督下训练一个分割网络，该伪掩膜 $\mathcal{M}$ 是通过第 3.2 节中基于熵的 CAM 融合获得的。对于每个分支，从其各自模型的不同块中收集嵌入，并通过特征对齐器 (FA) 模块对齐，以便可以共享一个解码器。

令 $\left\{F_b^i\right\}_{i=1}^{T_b}$ 表示从分支 $b \in \{\text{clip, diff}\}$ 的不同块中收集的嵌入。这里 $T_b$ 是从分支 $b$ 中选择的块的数量。FA 将每个 $F_b^i$ 上采样到 $(H_d, W_d)$，将通道投影到共享宽度 $D$，沿通道连接它们，并应用 1×1 卷积形成解码器输入：

$$
X_b = \text{Conv}_{1 \times 1}\left(\text{Cat}_{i=1}^{T_b}\left(\text{Up}(F_b^i)\right)\right) \in \mathbb{R}^{B \times D \times H_d \times W_d},
$$

其中 $\text{Up}(\cdot)$ 将每个嵌入上采样到 $(H_d, W_d)$，$\text{Cat}(\cdot)$ 表示通道级联，$\text{Conv}_{1 \times 1}(\cdot)$ 是将级联嵌入投影到 $D$ 通道的 1×1 卷积。然后共享解码器将 $X_b$ 作为输入，并为每个分支 $b \in \{\text{clip, diff}\}$ 生成逐像素类别逻辑值 $S_b \in \mathbb{R}^{B \times |\mathcal{C}| \times H_d \times W_d}$。这里 $\mathcal{C}$ 表示数据集中所有类别的集合（$|\mathcal{C}|$ 是类别的数量）。

随后，逻辑门控模块根据预测的逻辑值 $S_b$ 预测逐像素权重 $W'$：

$$
W' = \sigma\left(\text{Conv}_{1 \times 1}\left(\text{Cat}(S_{\text{clip}}, S_{\text{diff}})\right)\right) \in \mathbb{R}^{B \times 1 \times H_d \times W_d},
$$

其中 $\text{Conv}_{1 \times 1}(\cdot)$ 是将 $2|\mathcal{C}|$ 通道映射到一个通道的 1×1 卷积，$\sigma$ 是 sigmoid 函数。最后，与式(7)中的像素级融合规则一致，定义融合逻辑值为

$$
S_{\text{fuse}} = W' \odot S_{\text{diff}} + (\mathbf{1} - W') \odot S_{\text{clip}},
$$

其中 $\odot$ 表示元素级乘法，$\mathbf{1} \in \mathbb{R}^{H \times W}$ 是全一图。

***

如果说 **3.2 节是在“制作标准答案（伪标签）”**，那么 **3.3 节就是在“教一个学生网络如何又快又好地考出这个分数”**。

在实际应用中，如果每次分割都要把 CLIP 和 Diffusion 跑一遍，然后再去算一遍像素级的熵来融合，计算量会极其恐怖。所以，作者设计了这样一个**特征对齐解码器（FAD）**，让网络在训练时直接学习如何融合。

为了让你彻底明白，我们继续使用上一节的例子，并加入具体的张量变化和数值推导。

假设我们的数据集有 **3 个类别**（$|\mathcal{C}| = 3$：狗、猫、背景），批次大小 $B=1$。解码器处理的分辨率设定为 $H_d=4, W_d=4$。

### 第一步：特征对齐器 (Feature Aligner, FA) —— 统一语言
**背景**：CLIP 和 Diffusion 是两个完全不同的模型，它们提取出的中间特征（Feature Maps）在空间大小和通道数上都是不一样的。不能直接放到同一个解码器里。

**前置数据从哪来？**
在输入一张图片时，我们分别截取 CLIP 和 Diffusion **中间某几层**的输出。
*   假设我们从 CLIP 截取了 $T_{\text{clip}}=2$ 层特征：
    *   第 1 层：$F_{\text{clip}}^1$，形状是 $2 \times 2$，通道数 10。
    *   第 2 层：$F_{\text{clip}}^2$，形状是 $2 \times 2$，通道数 20。

**公式**：
$$
X_b = \text{Conv}_{1 \times 1}\left(\text{Cat}_{i=1}^{T_b}\left(\text{Up}(F_b^i)\right)\right)
$$
**张量变化与计算**：
1.  **$\text{Up}(\cdot)$ (上采样)**：把所有的特征图拉伸到统一的分辨率 $4 \times 4$。比如 $2 \times 2$ 通过双线性插值变成 $4 \times 4$。
2.  **$\text{Cat}(\cdot)$ (通道拼接)**：把上采样后的两层特征像汉堡一样叠起来。
    *   通道数变成：$10 + 20 = 30$。
    *   此时张量形状为：$1 \times 30 \times 4 \times 4$。
3.  **$\text{Conv}_{1 \times 1}(\cdot)$ (降维/统一通道)**：用 $1 \times 1$ 卷积，把 30 个通道映射到一个**固定的共享维度 $D$**（比如 $D=64$）。
    *   **最终对齐后的特征** $X_{\text{clip}} \in \mathbb{R}^{1 \times 64 \times 4 \times 4}$。
    *   同理，Diffusion 的特征也经过类似操作，变成 $X_{\text{diff}} \in \mathbb{R}^{1 \times 64 \times 4 \times 4}$。
### 第二步：共享解码器 (Shared Decoder) —— 各自打分
**过程**：既然两边的特征都被强行转换成了统一的形状 ($1 \times 64 \times 4 \times 4$)，我们就可以用**同一个**神经网络（共享权重的解码器）来处理它们。

**张量变化与计算**：
将 $X_{\text{clip}}$ 和 $X_{\text{diff}}$ 分别输入解码器，得到它们各自的预测分数（Logits），符号为 $S_{\text{clip}}$ 和 $S_{\text{diff}}$。
*   **形状**：$\mathbb{R}^{B \times |\mathcal{C}| \times H_d \times W_d}$，即 $\mathbb{R}^{1 \times 3 \times 4 \times 4}$。

**具体数值（看某 1 个像素）**：
假设在像素 $(1,1)$ 上，解码器给出的**原始分数 (Logits)** 是：
*   $S_{\text{clip}}$：[5.0 (狗), 1.0 (猫), 0.1 (背景)]
*   $S_{\text{diff}}$：[4.0 (狗), 2.0 (猫), 0.5 (背景)]
### 第三步：逻辑门控模块 (Logit Gating Module) —— 学习“谁更自信”
**背景**：在 3.2 节中，我们是通过算“熵”来决定谁的。但在这里，作者让网络**自己去学习**应该分配多少权重。这也是本节最精妙的地方之一：把人工制定的规则，变成了可通过梯度下降优化的参数！

**公式**：
$$
W' = \sigma\left(\text{Conv}_{1 \times 1}\left(\text{Cat}(S_{\text{clip}}, S_{\text{diff}})\right)\right)
$$
**张量变化与计算**：
1.  **$\text{Cat}$**：把两边的分数在通道维度拼起来。
    *   像素 $(1,1)$ 的向量变成：`[5.0, 1.0, 0.1, 4.0, 2.0, 0.5]`。
    *   形状变成：$\mathbb{R}^{1 \times 6 \times 4 \times 4}$ ($3+3=6$个通道)。
2.  **$\text{Conv}_{1 \times 1}$**：这是一个可以学习的线性层，它的作用是把这 6 个数字乘以一套权重（网络自己学的），然后加起来变成 1 个数字。
    *   假设网络学到的权重是 `[0.1, -0.1, 0, 0.2, 0, 0]`。
    *   乘积相加 = $5.0(0.1) + 1.0(-0.1) + \dots + 4.0(0.2) = 0.5 - 0.1 + 0.8 = 1.2$。
3.  **$\sigma$ (Sigmoid)**：把这个数字压缩到 0~1 之间。
    *   $W' = \frac{1}{1 + e^{-1.2}} \approx \mathbf{0.77}$。
    *   **最终形状**：$\mathbb{R}^{1 \times 1 \times 4 \times 4}$。（每个像素都有了一个 0~1 的权重）。
### 第四步：最终融合与监督训练 (Fusion & Supervision)
**公式**：
$$
S_{\text{fuse}} = W' \odot S_{\text{diff}} + (\mathbf{1} - W') \odot S_{\text{clip}}
$$
**张量变化与计算**：
这和 3.2 节的融合公式一模一样，只是现在融合的是网络自己预测的分数。
*   $W' = 0.77$（代表听 Diffusion 的比例是 77%）。
*   $1 - W' = 0.23$（代表听 CLIP 的比例是 23%）。

在这个像素上，最终的融合分数 $S_{\text{fuse}}$ 等于：
*   狗的分数：$0.77 \times 4.0 + 0.23 \times 5.0 = 3.08 + 1.15 = \mathbf{4.23}$
*   猫的分数：$0.77 \times 2.0 + 0.23 \times 1.0 = 1.54 + 0.23 = \mathbf{1.77}$
*   背景分数：$0.77 \times 0.5 + 0.23 \times 0.1 = 0.385 + 0.023 = \mathbf{0.408}$
*   融合后的特征形状依然是 $\mathbb{R}^{1 \times 3 \times 4 \times 4}$。

**最关键的一步（闭环）：**
算出的这个最终分数 $S_{\text{fuse}}$ 有什么用呢？
还记得 **3.2 节**我们辛辛苦苦算出来的**伪掩膜 $\mathcal{M}$** 吗？那个伪掩膜就是这里的 **“标准答案 (Ground Truth)”**！
模型会把 $S_{\text{fuse}}$ 输出的结果和 $\mathcal{M}$ 放在一起，计算**交叉熵损失 (Cross-Entropy Loss)**。
如果 $S_{\text{fuse}}$ 算出来的狗的分数不够高（对比 $\mathcal{M}$），损失函数就会反向传播，去调整前面 $\text{Conv}_{1 \times 1}$ 的权重，以及解码器里所有的参数。

**总结：**
3.3 节就是构建了一个双分支网络，利用 3.2 节的答案作为监督，让网络自己学会了如何提取特征、如何打分、以及如何动态加权（$W'$）。这样在模型部署时，就不需要复杂的熵计算了！
***
3.4. **训练目标**
融合逻辑值通过逐像素交叉熵损失由伪掩膜 $\mathcal{M}$ 监督：

$$
\mathcal{L}_{\text{fuse}} = \text{CE}(S_{\text{fuse}}, \mathcal{M}),
$$

其中 $\text{CE}(\cdot,\cdot)$ 表示标准的逐像素交叉熵损失。类似地，ComCD 用相同的交叉熵损失监督每个分支：

$$
\mathcal{L}_{\text{clip}} = \text{CE}(S_{\text{clip}}, \mathcal{M}), \quad \mathcal{L}_{\text{diff}} = \text{CE}(S_{\text{diff}}, \mathcal{M}).
$$

此外，遵循 ExCEL (Yang et al., 2025b)，ComCD 采用多样性损失 $\mathcal{L}_{\text{div}}$，以 discouraging 冗余预测并鼓励两个分支之间互补的线索。总训练目标是

$$
\mathcal{L}_{\text{total}} = \lambda_1 \mathcal{L}_{\text{fuse}} + \lambda_2 \mathcal{L}_{\text{clip}} + \lambda_3 \mathcal{L}_{\text{diff}} + \lambda_4 \mathcal{L}_{\text{div}},
$$

其中 $\lambda_1, \lambda_2, \lambda_3, \lambda_4 \geq 0$ 是重新缩放每个损失项贡献的权重。

### 4.**实验**

4.1. **实验设置**

**数据集和评估指标。** 我们在四个数据集上评估了我们提出的方法：PASCAL VOC 2012 (Everingham et al., 2010)、MS COCO 2014 (Lin et al., 2014)、PASCAL-Context (Mottaghi et al., 2014) 和 MS COCO-Object (Lin et al., 2014)。对于包含 21 个语义类别的 PASCAL VOC 2012，ComCD 沿用常见做法 (Long et al., 2015; Ru et al., 2023; Wu et al., 2024b)，使用扩展版本，包含 SBD 数据集，由 10,582 张训练图像、1,449 张验证图像和 1,456 张测试图像组成。对于包含 81 个类别的 MS COCO 2014，数据集分为 82,081 张训练图像和 40,137 张验证图像。对于包含 60 个类别的 PASCAL-Context，数据集包含 4,998 张训练图像和 5,105 张验证图像。对于包含 81 个类别的 MS COCO-Object，ComCD 在 COCO 2014 验证集中的 4,952 张验证图像上进行评估。主要评估指标是平均交并比 (mIoU)。

**实施细节。** ComCD 使用带有 ViT-B/16 (Dosovitskiy et al., 2020) 图像编码器的 CLIP 和预训练的 Stable Diffusion v2.1 (Rombach et al., 2022)。在整个训练过程中，两个骨干网络保持冻结。在 CLIP 方面，我们遵循 ExCEL (Yang et al., 2025b)：来自 ViT-B/16 所有 12 个 Transformer 层的补丁-标记特征被投影到 256 维度，并与 SegFormer 风格的头部融合到一个统一的 256 通道特征图中，并且类别名称在传递给文本编码器之前以固定的提示模板“a clean origami {}”呈现。在扩散方面，我们遵循 iSeg (Sun et al., 2024a)：扩散调度器设置为 1000 个去噪步骤，我们在训练期间均匀采样的时间步 $t$ 添加高斯噪声，并在推理时固定 $t=100$，并执行单个反向去噪步骤以提取交叉注意力和自注意力图，这些图被聚合到共享的 64×64 潜在网格（即 $H_{\text{low}}=W_{\text{low}}=64$）上，用于 CAM 生成，自注意力细化迭代按照 Sun et al. (2024a) 中的配置。两个分支被对齐到共同的 256 通道、共享分辨率特征空间，然后馈送到轻量级基于 Transformer 的解码器头部 (Zhang et al., 2024)。遵循先前的工作 (Wu et al., 2024b; Yang et al., 2024, 2025b)，AdamW 优化所有可训练组件，学习率为 1 × 10$^{-4}$，权重衰减为 1 × 10$^{-2}$。损失权重设置为 $\lambda_1=1$， $\lambda_2=\lambda_3=\lambda_4=0.1$。训练在 PASCAL VOC 2012 上进行 30,000 次迭代，在 MS COCO 2014 上进行 100,000 次迭代。在推理过程中，ComCD 采用 WSSS 中使用的标准方法，通过多尺度测试和使用密集 CRF 后处理。

![](../../../../../../99_Assets%20(资源文件)/images/276838941235e674b002fddc4283d070.png)

图4. WSSS 伪掩膜的可视化。我们比较了 ComCD 与 ExCEL (Yang et al., 2025b)、iSeg (Sun et al., 2024a) 和 CLIP-ES (Lin et al., 2023) 在 PASCAL VOC 2012 和 MS COCO 2014 数据集上的结果。

表1
伪掩膜评估。PASCAL VOC 2012 训练集和 MS COCO 2014 训练集上的分割 mIoU。

| 方法                                | 出版物                 | 类型  | 网络        | VOC  | COCO |
| --------------------------------- | ------------------- | --- | --------- | ---- | ---- |
| **需要训练的方法。**                      |                     |     |           |      |      |
| IRN (Ahn et al., 2019)            | CVPR’2019           | 多阶段 | ResNet50  | 66.5 | 42.4 |
| AdvCAM (Lee et al., 2021a)        | CVPR’2021           | 多阶段 | ResNet101 | 55.6 | 35.8 |
| CLIMS (Xie et al., 2022)          | CVPR’2022           | 多阶段 | ResNet50  | 56.6 | –    |
| HSC (Wu et al., 2023)             | IJCAI’2023          | 多阶段 | ResNet101 | 71.8 | –    |
| ToCo (Ru et al., 2023)            | CVPR’2023           | 单阶段 | ViT-B/16  | 72.2 | –    |
| BAS (Zhai et al., 2024)           | IJCV’2024           | 多阶段 | ResNet50  | 57.7 | 36.9 |
| WeCLIP (Zhang et al., 2024)       | CVPR’2024           | 单阶段 | ViT-B/16  | 75.4 | –    |
| MCTformer+ (Xu et al., 2024)      | TIPAMI’2024         | 多阶段 | ResNet38  | 68.8 | –    |
| DiG (Yoon et al., 2024a)          | ECCV’2024           | 多阶段 | DeiT-S    | 74.3 | –    |
| S2C (Kweon & Yoon, 2024)          | CVPR’2024           | 多阶段 | WResNet38 | 81.7 | –    |
| PRCE (Xu et al., 2025)            | CVPR’2025           | 单阶段 | ResNet50  | 77.6 | –    |
| **免训练方法。**                        |                     |     |           |      |      |
| DiffSegmenter (Wang et al., 2025) | TIP’2025            | 单阶段 | U-Net     | 70.5 | –    |
| T2M (Xiao et al., 2024)           | Neurocomputing’2024 | 单阶段 | U-Net     | 72.7 | 43.7 |
| CLIP-ES (Lin et al., 2023)        | CVPR’2023           | 多阶段 | ViT-B/16  | 70.8 | 39.7 |
| iSeg (Sun et al., 2024a)          | arXiv’2024          | 单阶段 | U-Net     | 75.2 | 45.5 |
| ExCEL (Yang et al., 2025b)        | CVPR’2025           | 单阶段 | ViT-B/16  | 74.6 | 43.6 |
| Ours                              | –                   | 单阶段 | ViT-B/16  | 82.1 | 51.3 |

**评估协议。** 对于 PASCAL VOC 2012 和 MS COCO 2014 上的 WSSS，我们首先使用冻结的 CLIP 和扩散骨干评估 ComCD：我们在训练集上生成基于 CLIP 和基于扩散的 CAM，通过基于熵的融合将它们融合成伪掩膜，并计算在训练集上与GT真实掩膜的 mIoU。在此步骤中，不训练任何额外的参数，包括解码器，结果报告在表 1 中。在第二步中，这些伪掩膜用于监督我们单阶段 WSSS 框架的分割解码器，该解码器在训练集上训练并在验证集上评估；相应的 mIoU 总结在表 2 中，并与最近的需要训练的 WSSS 方法在相同的分割和指标下进行比较。

对于开放词汇语义分割，ComCD 同样在 PASCAL VOC 2012、PASCAL-Context 和 MS COCO-Object 的验证集上采用冻结骨干的伪掩膜评估：冻结的 CLIP 和扩散骨干以及基于熵的融合直接生成掩膜，我们计算在验证集上的 mIoU，如表 3 所示。使用预训练的 CLIP 模型，并使用 TagCLIP (Lin et al., 2024) 生成的图像级类别标签作为文本提示。除非另有说明，所有其他设置均遵循 WSSS 中使用的设置。

4.2. **实验结果**
**融合的 CAM。** 图 3 展示了 ComCD 在 PASCAL VOC 2012 上的效果，以及代表性的基线方法。图 3(b) 中的基于 CLIP 的 CAM 为定位提供了类别先验，勾勒出边界并突出判别区域，而图 3(c) 中的基于扩散的 CAM 贡献了促进区域连续性并产生更完整激活的结构先验。此外，基于熵的逐像素权重对不可靠分支进行降权，从而实现了错误激活的相互校正。图 3(d) 中我们基于熵的融合将这些优势整合到单个图中，通过抑制边缘附近的背景泄漏和弥合内部间隙，使其与图 3(i) 中的地面真实更紧密地对齐。与图 3(e)-(h) 中的 CLIP-ES (Lin et al., 2023)、DuPL (Wu et al., 2024b)、SeCo (Yang et al., 2024) 和 WeCLIP (Zhang et al., 2024) 相比，ComCD 沿对象边界和在非判别区域产生更清晰的响应。

**WSSS 伪掩码评估。** 表 1 报告了在 PASCAL VOC 2012 和 MS COCO 2014 上生成的伪掩码的 mIoU。使用 ViT-B/16，ComCD 在 VOC 上达到 82.1%，在 MS COCO 2014 上达到 51.3%。与表 1 中最近的强基线方法相比，ComCD 在 VOC 上超过 PRCE (77.6%) +4.5%，在 MS COCO 2014 上超过 IRN (42.4%) +8.9%。相对于 iSeg，在 VOC 上增益 +6.9%，在 MS COCO 2014 上增益 +5.8%；相对于 T2M，增益 +9.4% 和 +7.6%；相对于 CLIP-ES，增益 +11.3% 和 +11.6%；相对于 ExCEL，增益 +7.5% 和 +7.7%；以及相对于 DiffSegmenter，在 VOC 上增益 +11.6%。图 4 提供了与 ExCEL (Yang et al., 2025b)、iSeg (Sun et al., 2024a) 和 CLIP-ES (Lin et al., 2023) 的定性比较，其中 ComCD 产生的伪掩膜在两个数据集上都具有更完整的对象覆盖和更锐利的边界，同时减少了假阳性、背景泄漏和碎片化区域。所有差异均直接从表 1 计算。没有 MS COCO 2014 结果的方法仅在 VOC 上进行比较。图 6 中的每类别 mIoU 雷达图进一步显示，ComCD 在所有类别上均优于 CLIP-ES、ExCEL 和 iSeg。

**WSSS 最终分割结果。** 表 2 总结了在 PASCAL VOC 2012 验证集和 MS COCO 2014 验证集上多阶段和单阶段需要训练的 WSSS 的分割 mIoU。使用 $\mathcal{I}+\mathcal{T}$ 监督和 ViT-B/16 骨干，我们的方法在 VOC 上达到 79.5%，在 MS COCO 2014 上达到 52.1%。在单阶段方法中，它超过了 ExCEL (Yang et al., 2025b) (77.2% 和 49.3%) +2.3% 和 +2.8%。相对于领先的多阶段方法，它高于 CPAL (Tang et al., 2024) 在 VOC 上的 74.5% (+5.0%)，并高于 PSDPM (Zhao et al., 2024) 在 MS COCO 2014 上的 47.2% (+4.9%)。在其他代表性单阶段方法中，绝对改进范围在 VOC 上从 +1.1% 到 +13.5%，在 MS COCO 2014 上从 +1.8% 到 +13.2%。图 5 提供了与 ExCEL (Yang et al., 2025b)、DuPL (Wu et al., 2024b) 和 ToCo (Ru et al., 2023) 的定性比较，视觉结果与表 2 中报告的定量增益一致。

表2 
弱监督语义分割结果。PASCAL VOC 2012 验证集和 MS COCO 2014 验证集上的分割 mIoU。Sup. 表示监督类型（$\mathcal{I}$ 图像级，$\mathcal{S}$ 显著性， $\mathcal{T}$ 文本）。'–' 表示未报告。

| 方法                           | 出版物         | 监督                          | 网络        | VOC  | COCO |
| ---------------------------- | ----------- | --------------------------- | --------- | ---- | ---- |
| **多阶段 WSSS 方法。**             |             |                             |           |      |      |
| L2G (Jiang et al., 2022)     | CVPR’2022   | $\mathcal{I}+\mathcal{S}$   | ResNet101 | 72.1 | 44.2 |
| RCA (Zhou et al., 2022b)     | CVPR’2022   | $\mathcal{I}+\mathcal{S}$   | ResNet38  | 72.2 | 36.8 |
| OCR (Cheng et al., 2023)     | CVPR’2023   | $\mathcal{I}$               | ResNet38  | 72.7 | 42.5 |
| BECO (Rong et al., 2023)     | CVPR’2023   | $\mathcal{I}$               | ResNet101 | 73.7 | 45.1 |
| MCTformer+ (Xu et al., 2024) | TIPAMI’2024 | $\mathcal{I}$               | ResNet38  | 74.0 | 45.2 |
| CTI (Yoon et al., 2024b)     | CVPR’2024   | $\mathcal{I}$               | ResNet101 | 74.1 | 45.4 |
| MuP-VSS (Duan et al., 2025)  | CVPR’2025   | $\mathcal{I}$               | ResNet38  | 73.6 | 46.6 |
| CLIMS (Xie et al., 2022)     | CVPR’2022   | $\mathcal{I} + \mathcal{T}$ | ResNet50  | 70.4 | –    |
| CLIP-ES (Lin et al., 2023)   | CVPR’2023   | $\mathcal{I} + \mathcal{T}$ | ResNet101 | 72.2 | 45.4 |
| PSDPM (Zhao et al., 2024)    | CVPR’2024   | $\mathcal{I} + \mathcal{T}$ | ResNet101 | 74.1 | 47.2 |
| CPAL (Tang et al., 2024)     | CVPR’2024   | $\mathcal{I} + \mathcal{T}$ | ResNet101 | 74.5 | 46.8 |
| DiG (Yoon et al., 2024a)     | ECCV’2024   | $\mathcal{I}$               | WResNet34 | 73.9 | 45.5 |
| S2C (Kweon & Yoon, 2024)     | CVPR’2024   | $\mathcal{I}$               | WResNet34 | 78.2 | 49.8 |
| **单阶段 WSSS 方法。**             |             |                             |           |      |      |
| AFA (Ru et al., 2022)        | CVPR’2022   | $\mathcal{I}$               | MiT-B/16  | 66.0 | 38.9 |
| TSCD (Xu et al., 2023b)      | AAAI’2023   | $\mathcal{I}$               | MiT-B/16  | 67.3 | 40.1 |
| ToCo (Ru et al., 2023)       | CVPR’2023   | $\mathcal{I}$               | ViT-B/16  | 71.1 | 42.3 |
| DuPL (Wu et al., 2024b)      | CVPR’2024   | $\mathcal{I}$               | ViT-B/16  | 73.3 | 44.6 |
| SeCo (Yang et al., 2024)     | CVPR’2024   | $\mathcal{I}$               | ViT-B/16  | 74.0 | 46.7 |
| PRCE (Xu et al., 2025)       | CVPR’2025   | $\mathcal{I}$               | VIT-B/16  | 75.5 | 47.2 |
| DIAL (Jang et al., 2024)     | ECCV’2024   | $\mathcal{I} + \mathcal{T}$ | ViT-B/16  | 74.5 | 44.4 |
| WeCLIP (Zhang et al., 2024)  | CVPR’2024   | $\mathcal{I} + \mathcal{T}$ | ViT-B/16  | 76.4 | 47.1 |
| ExCEL (Yang et al., 2025b)   | CVPR’2025   | $\mathcal{I} + \mathcal{T}$ | ViT-B/16  | 77.2 | 49.3 |
| Ours                         | –           | $\mathcal{I} + \mathcal{T}$ | ViT-B/16  | 79.5 | 52.1 |

表3
与开放词汇分割方法的比较。我们报告了 PASCAL VOC 2012 验证集、PASCAL-VOC Context 验证集和 MS COCO-Object 验证集上的 mIoU。'–' 表示未报告。

| 方法                                | 出版物        | VOC  | Context | Object |
| --------------------------------- | ---------- | ---- | ------- | ------ |
| **需要训练的方法。**                      |            |      |         |        |
| ReCo (Shin et al., 2022)          | NeurIPS’22 | 25.1 | 19.9    | 15.7   |
| MaskCLIP (Zhou et al., 2022a)     | ECCV’22    | 38.8 | 23.6    | 20.6   |
| SegCLIP (Luo et al., 2023)        | ICML’23    | 52.6 | 24.7    | 26.5   |
| CLIPpy (Ranasinghe et al., 2023)  | ICCV’23    | 52.2 | –       | 32.0   |
| ViewCo (Ren et al., 2026)         | ICLR’23    | 52.4 | 23.0    | 23.5   |
| OVSegmenter (Xu et al., 2023a)    | CVPR’23    | 53.8 | 20.4    | 25.1   |
| TCL (Cha et al., 2023)            | CVPR’23    | 51.2 | 24.3    | 30.4   |
| SAM-CLIP (Wang et al., 2024)      | CVPRW’24   | 60.6 | 29.2    | 31.5   |
| LPOSS (Stojnic et al., 2025)      | CVPR’25    | 62.4 | 34.3    | 35.4   |
| **免训练方法。**                        |            |      |         |        |
| TagCLIP (Lin et al., 2024)        | AAAI’24    | 64.8 | –       | –      |
| CaR (Sun et al., 2024b)           | CVPR’24    | 67.6 | 30.5    | 36.6   |
| DiffSegmenter (Wang et al., 2025) | TIP’25     | 60.1 | 27.5    | 37.9   |
| iSeg (Sun et al., 2024a)          | arXiv’24   | 68.2 | 30.9    | 38.4   |
| ProxyCLIP (Lan et al., 2024)      | ECCV’24    | 61.3 | 35.3    | 37.5   |
| ResCLIP (Yang et al., 2025a)      | CVPR’25    | 61.1 | 33.5    | 35.0   |
| CASS (Kim et al., 2025)           | CVPR’25    | 65.8 | 36.7    | 37.8   |
| Ours                              | –          | 74.2 | 54.8    | 39.3   |

**OVSS 伪掩膜评估。** 表 3 报告了在 PASCAL VOC 2012、PASCAL-Context 和 MS COCO-Object 上生成的伪掩膜的 mIoU。使用 ViT-B/16，ComCD 在 VOC 上达到 74.2%，在 Context 上达到 54.8%，在 MS COCO-Object 上达到 39.3%，在所有三个基准上均排名第一。与 LPOSS (Stojnić et al., 2025) (62.4%, 34.3%, 35.4%) 相比，增益为 +11.8%, +20.5% 和 +3.9%。相对于 iSeg (Sun et al., 2024a) (68.2%, 30.9%, 38.4%)，改进为 +6.0%, +23.9% 和 +0.9%，并且相对于 CASS (Kim et al., 2025) (65.8%, 36.7%, 37.8%)，改进为 +8.4%, +18.1% 和 +1.5%。PASCAL-Context 上的改进尤为显著，而 VOC 表现出明显的优势，COCO-Object 则表现出较小但一致的优势。图 7 展示了生成的伪掩膜与 DiffSegmenter (Wang et al., 2025) 和 iSeg (Sun et al., 2024a) 的定性比较。所有差异均直接从表 3 计算。

**与全监督方法的比较。** 表 4 比较了弱监督方法与它们在 PASCAL VOC 2012 验证集上的全监督对应方法。我们的需要训练的方法，使用

表4
与 VOC 验证集上全监督/弱监督方法的比较。$\mathcal{F}$：全监督。ViT-B*：从 CLIP 预训练。Ratio 是相对于相同骨干的全监督对应方法的比率。

| 方法                               | 出版物        | 监督                        | 网络        | 验证集  | 比例    |
| -------------------------------- | ---------- | ------------------------- | --------- | ---- | ----- |
| **全监督方法**                        |            |                           |           |      |       |
| DeepLabV2 (Chen et al., 2017)    | TPAMI’2017 | $\mathcal{F}$             | ResNet101 | 77.7 | –     |
| DeepLabV2 (Chen et al., 2017)    | TPAMI’2017 | $\mathcal{F}$             | ViT-B/16  | 82.3 | –     |
| WeCLIP-Full (Zhang et al., 2024) | CVPR’2024  | $\mathcal{F}$             | ViT-B*    | 81.6 | –     |
| **多阶段 WSSS 方法**                  |            |                           |           |      |       |
| CLIMS (Xie et al., 2022)         | CVPR’2022  | $\mathcal{I}+\mathcal{T}$ | ResNet101 | 70.4 | 90.6% |
| CLIP-ES (Lin et al., 2023)       | CVPR’2023  | $\mathcal{I}+\mathcal{T}$ | ResNet101 | 72.2 | 92.9% |
| CPAL (Tang et al., 2024)         | CVPR’2024  | $\mathcal{I}+\mathcal{T}$ | ResNet101 | 74.5 | 95.9% |
| **单阶段 WSSS 方法**                  |            |                           |           |      |       |
| ToCo (Ru et al., 2023)           | CVPR’2023  | $\mathcal{I}$             | ViT-B/16  | 71.1 | 86.4% |
| DuPL (Wu et al., 2024b)          | CVPR’2024  | $\mathcal{I}$             | ViT-B/16  | 73.3 | 89.1% |
| SeCo (Yang et al., 2024)         | CVPR’2024  | $\mathcal{I}$             | ViT-B/16  | 74.0 | 89.9% |
| DIAL (Jang et al., 2024)         | ECCV’2024  | $\mathcal{I}+\mathcal{T}$ | ViT-B/16  | 74.5 | 90.5% |
| WeCLIP (Zhang et al., 2024)      | CVPR’2024  | $\mathcal{I}+\mathcal{T}$ | ViT-B*    | 76.4 | 93.6% |
| ExCEL (Yang et al., 2025b)       | CVPR’2025  | $\mathcal{I}+\mathcal{T}$ | ViT-B*    | 77.2 | 94.6% |
| Ours                             | –          | $\mathcal{I}+\mathcal{T}$ | ViT-B*    | 79.5 | 97.4% |

![](../../../../../../99_Assets%20(资源文件)/images/242164f8a033ad865752b94434728873.png)
图5. 需要训练的弱监督分割结果的可视化。我们比较了 ComCD 与 ExCEL (Yang et al., 2025b)、DuPL (Wu et al., 2024b) 和 ToCo (Ru et al., 2023) 的结果。

表5
消融研究。“Seg.” 表示分割性能。

|$\mathcal{L}_{\text{fuse}}$|$\mathcal{L}_{\text{clip}}$|$\mathcal{L}_{\text{diff}}$|$\mathcal{L}_{\text{div}}$|Seg.|
|---|---|---|---|---|
|✓||||74.1|
|✓|✓|||77.4|
|✓||✓||76.5|
|✓|||✓|76.0|
|✓|✓|✓||78.5|
|✓|✓||✓|77.4|
|✓|✓|✓|✓|79.5|

ViT-B* 实现了 79.5% 的 mIoU，这相当于使用相同骨干的全监督结果 (WeCLIP-Full (Zhang et al., 2024)，81.6%) 的 97.4%。在单阶段 WSSS 中，它高于 ExCEL (Yang et al., 2025b) 的 77.2% (94.6%)，并高于 WeCLIP (Zhang et al., 2024) 的 76.4% (93.6%)。相对于 ResNet101 上的强大多阶段方法，它高于 CPAL (Tang et al., 2024) 的 74.5% (95.9%)。与 ViT-B* 上的全监督上限的剩余差距为 2.1%，这表明我们的方法在弱监督的情况下弥合了大部分差距。

4.3. **消融研究与分析**
**组件的有效性。** 在我们的 WSSS 框架中，表 5 评估了所提出的 Feature Aligned Decoder 中使用的损失。仅监督融合逻辑值 $\mathcal{L}_{\text{fuse}}$ 产生 74.1%。添加 CLIP 分支监督 $\mathcal{L}_{\text{clip}}$ 将分数提高到 77.4% (+3.3%)。监督扩散分支 $\mathcal{L}_{\text{diff}}$ 得到 76.5% (+2.4%)。引入多样性损失 (Yang et al., 2025b) $\mathcal{L}_{\text{div}}$ 和 $\mathcal{L}_{\text{fuse}}$ 达到 76.0% (+1.9%)。将 $\mathcal{L}_{\text{clip}}$ 与 $\mathcal{L}_{\text{div}}$ 结合达到 78.5%，比单独使用 $\mathcal{L}_{\text{clip}}$ 提高了 +1.1%，而将 $\mathcal{L}_{\text{diff}}$ 与 $\mathcal{L}_{\text{div}}$ 结合达到 77.4%，比单独使用 $\mathcal{L}_{\text{diff}}$ 提高了 +0.9%。所有组件一起使用时性能最佳，达到 79.5%，比最强的双损失配置 $\mathcal{L}_{\text{clip}}+\mathcal{L}_{\text{div}}$ 高出 +1.0%。这些结果表明，监督两个分支可以改善融合逻辑值，并且多样性损失始终鼓励 CLIP 和扩散分支之间的互补性。

**基于熵的权重分析。** 图 8 可视化了式 (7) 中组合基于 CLIP 和基于扩散 CAM 的逐像素权重 $W$。我们分别报告了前景 (FG) 和背景 (BG) 区域的权重。

![](../../../../../../99_Assets%20(资源文件)/images/ea80f3c491afa976ffb983ed0641a18b.png)

图6. PASCAL VOC 2012 上每类别 mIoU。PASCAL VOC 2012 训练集上每类别伪标签 mIoU 的雷达图，比较了 ComCD 与 CLIP-ES (Lin et al., 2023)、ExCEL (Yang et al., 2025b) 和 iSeg (Sun et al., 2024a)。半径越大表示 mIoU 越高。ComCD 在所有 21 个类别上均超越了所有三个基线，表明在所有类别上都取得了持续的增益。

![](../../../../../../99_Assets%20(资源文件)/images/e4ac77aa1a186b22250dd14f59ad866d.png)

图7. OVSS 伪掩膜的可视化。我们将 ComCD 的结果与 DiffSegmenter (DiffSeg) (Wang et al., 2025) 和 iSeg (Sun et al., 2024a) 进行了比较。

![](../../../../../../99_Assets%20(资源文件)/images/e09e91572f95950400edee2fa58b7c53.png)

图8. 融合权重可视化。式 (7) 中使用的逐像素基于熵的权重 $W$。白色像素表示选择更自信的分支（基于 CLIP 或基于扩散）。“FG”和“BG”分别表示在前景对象区域和背景区域计算的权重。

表6
融合策略消融。“Seg.” 表示分割性能。

| EBF | LGM | Seg. |
| :-: | :-: | :--- |
| ✓   |     | 79.1 |
|     | ✓   | 79.5 |

表7
融合策略对 PASCAL VOC 2012 的影响。“Pseudo”和“Seg.”分别表示训练集上的伪掩膜 mIoU 和验证集上的最终分割。

| 融合策略        | Pseudo | Seg. |
| :-------------- | :----- | :--- |
| 仅 CLIP         | 74.9   | 76.4 |
| 仅 Diffusion    | 75.0   | 75.9 |
| 等权重平均      | 76.3   | 77.1 |
| 元素级最大值    | 75.6   | 76.1 |
| EBF (ours)      | 82.1   | 79.5 |

白色像素表示选择了更自信的分支（基于 CLIP 的 CAM 或基于扩散的 CAM）。较低的熵意味着较高的置信度，因此对该位置的融合 CAM 贡献更大。较暗的像素表示相反。可视化权重与两个分支的预期作用一致。权重图倾向于在需要类别判别定位时偏爱 CLIP，在空间连贯性有利时偏爱扩散。这一观察结果与先前的发现一致，即基于 CLIP 的方法强调类别定位 (Lin et al., 2023; Xie et al., 2022; Yang et al., 2025b)，而基于扩散的方法增强空间连贯性 (Sun et al., 2024a; Wang et al., 2025; Yoon et al., 2024a)。在融合分割掩膜中观察到的改进进一步支持了基于熵的加权的有效性。

**融合策略分析。** 表 6 比较了训练时的两种融合方案：第 3.2 节中的基于熵的融合 (EBF) 和逻辑门控模块 (LGM)。EBF 首先将基于 CLIP 和基于扩散的 CAM 转换为逐像素类别分布，然后通过 sigmoid 将两个分支之间的熵差映射到标量权重，使得熵较低（置信度较高）的分支在融合预测中获得更大的贡献，而 LGM 通过将 1×1 卷积应用于连接的逻辑值 $[S_{\text{clip}}, S_{\text{diff}}]$ 后接 sigmoid，预测逐像素权重图。分割分数非常接近 (EBF 79.1%，LGM 79.5%)，我们注意到这些结果是通过多次独立运行（三次不同随机种子运行）平均得到的。这表明 LGM 学习的融合规则与 EBF 中使用的基于不确定性的加权在很大程度上是一致的，而不是利用两个分支之间完全不同的偏好模式；LGM 的微小但一致的增益表明，轻量级的学习门控可以在模糊区域中适度校准基于熵的规则，而 EBF 本身仍然是一个强大且无参数的基线融合机制。

**融合策略的影响。** 为了进一步量化在伪标签生成阶段基于熵的融合 (EBF) 的重要性，我们将其与 PASCAL VOC 2012 上的四种更简单的融合方案进行了比较：(1) 仅使用基于 CLIP 的 CAM 作为伪标签，(2) 仅使用基于扩散的 CAM，(3) 平均两个分支的逐像素类别概率（等权重平均），以及 (4) 取两个分支概率的元素级最大值（最大融合）。对于每种方案，我们首先在训练集上评估伪掩膜，然后在此掩膜上训练 ComCD，并报告在验证集上的最终分割性能。结果总结在表 7 中。仅使用 CLIP、仅使用扩散或简单融合（平均/最大）都在 75-76% 的伪掩膜质量和 76-77% 的最终分割范围内，而 EBF 将它们分别提升到 82.1% 和 79.5%。这个明显的优势表明，我们伪标签阶段的基于熵的融合，它自动地在每个像素选择更值得信任的分支，是至关重要的，并且提供了比两个分支的朴素组合更强的监督。

表8
物体大小对伪掩膜质量的影响。我们将 VOC 图像划分为三个子集，根据前景物体占据图像区域的比例，并报告每个子集上的 mIoU。

| 子集（面积比） | ExCEL | iSeg | ComCD |
| :------------- | :---- | :--- | :---- |
| Small (<20%)   | 65.1  | 65.7 | 74.9  |
| Medium (20%−50%) | 77.3  | 77.0 | 84.0  |
| Large (>50%)   | 73.5  | 73.1 | 79.0  |

表9
VOC 上的边界准确性。我们计算了地面真实边界周围 2 像素带宽内的 mIoU，以评估边缘质量。

| 方法        | 边界 mIoU |
| :---------- | :-------- |
| ExCEL       | 72.3      |
| iSeg        | 66.9      |
| ComCD       | 75.6      |

**按类别行为。** 除了总体 mIoU，我们通过图 6 中的雷达图进一步检查了类别级别的伪掩膜质量。总体而言，ComCD 在 21 个类别的大多数上都倾向于优于单分支基线，如 CLIP-ES (Lin et al., 2023)、ExCEL (Yang et al., 2025b) 和 iSeg (Sun et al., 2024a)，在几个代表性类别上具有更明显的优势。特别是，具有细长结构或丰富局部细节的类别（例如，椅子、瓶子、摩托车、人）以及经常出现在复杂共现背景下的类别（例如，船、沙发）在图 6 中显示出 ComCD 更大的半径，表明伪掩膜更稳定、噪声更少。一个合理的解释是这两个分支强调不同方面：CLIP 分支对类别判别线索更敏感，而扩散分支在保持区域连接性方面更有效。基于熵的融合在这种“精细结构 + 强上下文”场景中似乎特别有帮助，在这种场景中，平衡判别性和空间连贯性对于单个分支来说并非易事。

**对物体尺寸和形状的鲁棒性。** 为了评估 ComCD 在不同物体尺度和形状下的行为，我们根据前景物体占据图像区域的比例，将 PASCAL VOC 2012 图像 (Everingham et al., 2010) 划分为三个子集，并报告 ExCEL (Yang et al., 2025b)、iSeg (Sun et al., 2024a) 和 ComCD 的伪掩膜 mIoU（见表 8）。在小型、中型和大型子集中，ComCD 的 mIoU 均高于两个基线，在小型和中型物体上增益相对较大。这些子集通常包含部分被遮挡的实例、多个相邻物体和不规则的轮廓，在这些情况下，单分支方法常常面临在保留局部细节和保持全局区域一致性之间的权衡。基于熵的融合通过自适应地结合两个分支在一定程度上缓解了这种紧张关系。对于占据图像很大一部分的大型物体，ComCD 也保持了持续的优势，这表明所提出的融合策略可以适应物体范围和形状复杂性的变化，而不是局限于狭窄的尺度范围。

**边界准确性分析。** 我们进一步引入了一个以边界为中心的指标，以单独评估对象轮廓周围的分割质量。从地面真实掩膜开始，我们通过膨胀和腐蚀掩膜两个像素来构建每个对象边界周围的窄带，并仅在此带宽内计算 mIoU。表 9 报告了 ExCEL (Yang et al., 2025b)、iSeg (Sun et al., 2024a) 和 ComCD 的边界 mIoU。ComCD 在三个方法中取得了最高分，表明沿对象边缘的局部间隙更少，背景泄漏更不明显。这一观察结果与图 8 中可视化基于熵的权重分布一致：在权重图中，边界附近的像素通常对 CLIP 分支具有更高的权重，而内部像素倾向于扩散分支。两个分支之间的这种角色分工在类别判别边界和区域级连贯性之间提供了更稳定的平衡，这反过来有助于产生更平滑、更连续的轮廓预测。

**失败案例和局限性。** 尽管观察到上述改进，ComCD 仍然表现出一些典型的失败案例。一个典型的场景出现在包含餐桌和椅子 (图 6) 的室内场景中：当桌子和椅子紧密排列且背景杂乱时，扩散分支可能会过度平滑桌面和周围椅子之间的过渡，而 CLIP 分支可能主要关注桌面而低估细长的腿和支撑。对于自行车，车轮辐条和车架结构非常精细；在具有挑战性的视角或遮挡下，两个分支在这些区域都可能置信度较低，融合预测仍可能遗漏部分车轮或将其与附近的背景合并。对于小型或视觉模糊的瓶子实例（例如透明或高度反光的瓶子），CLIP 分支有时会强调徽标或高对比度纹理，而扩散分支可能会将瓶子连接到具有相似颜色或纹理的背景区域。这些观察结果表明，当前的设计，依赖于来自两个冻结骨干的逐像素熵和相对简单的融合规则，并未明确建模实例级结构或更丰富的几何先验。我们认为将实例感知细化和更具表现力的不确定性建模作为未来工作的有前景方向。

表10
运行时、内存和 VOC 上的分割性能。我们报告了 PASCAL VOC 2012 (val) 上的 FPS、峰值 GPU 内存和推理时间。

| 方法                         | FPS  | 内存 (GB) | 时间 (s) | Seg. |
| :--------------------------- | :--- | :-------- | :------- | :--- |
| ToCo (Ru et al., 2023) (CVPR’23) | 5.93 | 1.87      | 244.22   | 71.1 |
| DuPL (Wu et al., 2024b) (CVPR’24) | 2.98 | 2.25      | 486.75   | 73.3 |
| SeCo (Yang et al., 2024) (CVPR’24) | 4.94 | 1.85      | 293.18   | 74.0 |
| ExCEL (Yang et al., 2025b) (CVPR’25) | 7.75 | 1.41      | 186.94   | 77.2 |
| Ours                         | 3.77 | 3.98      | 384.68   | 79.5 |


**效率分析。** 表 10 总结了 PASCAL VOC 2012 上的运行时、内存使用和分割性能。与其他单阶段基于 ViT 的基线方法相比，ComCD 由于双分支设计和解码器中的特征对齐而产生了更高的计算成本（3.77 FPS 和 3.98 GB 峰值 GPU 内存），同时实现了最佳的 79.5% mIoU。特别是，我们的吞吐量仍与先前方法处于同一数量级（并且高于 DuPL Wu et al., 2024b），并且内存占用在现代 GPU 上仍然可接受。在优先考虑分割准确性的场景中，这种准确性-效率权衡是合理的，并且进一步降低开销（例如，通过更轻的扩散骨干或特征缓存）作为有前景的未来工作。

### 5.**结论**
在这项工作中，我们提出了 ComCD，一个结合了基于 CLIP 的定位和基于扩散的结构连贯性的 WSSS 框架。首先，通过基于熵的加权融合两个分支的 CAM，生成融合 CAM，然后将其转换为伪掩膜。该伪掩膜监督特征对齐解码器，该解码器对齐共享解码器的特征，从两个分支生成逻辑值，并应用逻辑门控模块生成最终的融合预测。通过利用两个分支的互补优势，所提出的融合抑制了虚假激活，减少了背景泄漏，并恢复了具有更锐利边界的更完整对象。