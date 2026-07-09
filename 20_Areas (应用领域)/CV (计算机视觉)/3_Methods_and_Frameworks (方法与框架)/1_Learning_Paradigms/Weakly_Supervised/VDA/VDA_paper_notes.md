CVPR网址：[CVPR 2026 Open Access Repository](https://openaccess.thecvf.com/content/CVPR2026/html/Qiu_Beyond_Text_Visual_Description_Assembly_by_Probabilistic_Model_for_CLIP-based_CVPR_2026_paper.html)

本地PDF：[VDA](../../../../../../99_Assets%20(资源文件)/papers/Qiu_Beyond_Text_Visual_Description_Assembly_by_Probabilistic_Model_for_CLIP-based_CVPR_2026_paper.pdf)
***
# Beyond Text: Visual Description Assembly by Probabilistic Model forCLIP-based Weakly Supervised Semantic Segmentation 

超越文本：用于基于 CLIP 的弱监督语义分割的概率模型视觉描述组装
## 摘要

对比语言-图像预训练 (CLIP) 通过从文本-图像对齐中生成类激活图 (CAMs)，为弱监督语义分割 (WSSS) 提供了一种新范式。现有方法主要依赖于手工制作的模板或由大型语言模型生成的通用属性描述来构建文本原型，以此作为查询视觉特征的依据。然而，这些策略面临两个主要局限性：CLIP 中固有的模态差距阻碍了文本原型与视觉特征实现紧密对齐；并且它们静态的文本原型无法自适应地响应展现出多样化视觉属性的目标实例。为了应对这些挑战，我们的核心见解是直接构建特定于实例的视觉描述原型作为查询，从而绕过次优的静态文本描述优化。为此，我们提出了视觉描述组装 (VDA) 框架。它采用概率模型将复杂的 CLIP 视觉特征映射到一个结构化的潜在空间中。这个潜在空间允许我们显式地解耦和聚合不同的视觉属性，然后将它们动态地组装成特定于实例的视觉原型。此外，为了增强该原型的鲁棒性，我们自适应地将语义稳定的文本原型融入其中，作为生成卓越 CAMs 的最终查询。实验结果表明，我们的方法优于现有的基线，在 WSSS 基准测试上实现了最先进的性能。

## 3. 方法论

### 3.1. 预备知识

**CAM 生成。** 在基于 CLIP 的 WSSS 中，CAMs 是通过计算视觉块特征和类别文本嵌入之间的余弦相似度得出的，如 [42] 中所述。给定一张图像 $I \in \mathbb{R}^{3 \times H \times W}$，令 $V \in \mathbb{R}^{h \times w \times D}$ 为其 CLIP 视觉特征，且 $T \in \mathbb{R}^{K \times D}$ 为 $K$ 个类别的文本嵌入。类别 $k$ 的 CAM $M_k$ 计算如下：

$$
M_k = \mathrm{Norm}(\cos\langle V, T_{k}^{\top} \rangle), \tag{1}
$$

其中 $\mathrm{Norm}$ 表示最小-最大归一化，$\cos\langle\cdot, \cdot\rangle$ 表示其中两项之间的余弦相似度。显然，在视觉嵌入 $V$ 保持不变的情况下，CAM $M_k$ 的准确性完全取决于文本嵌入 $T$ 的质量及其与 $V$ 实现细粒度、紧密对齐的能力。不幸的是，这正是被先前方法的静态、不灵活的文本以及 CLIP 的固有模态差异所阻碍的。我们的工作重点是构建一个查询原型，通过直接整合来自视觉空间的动态属性以更好地对齐 $V$，从而取代静态文本原型。

**框架概览。** 我们的 VDA 的整体框架如图 2 所示。它将视觉空间中的属性组装成视觉描述原型，将其作为查询以避免由于仅仅使用文本原型而造成的模态差距。为了从视觉空间中分离视觉属性，我们首先提出了视觉属性建模与解耦 (VAMD)，它使用可逆神经网络 (INN) [1] 将 CLIP 视觉特征建模为结构化的 GMM，在此我们可以显式地解耦类内属性。然后，为了将这些属性组装成视觉原型，我们提出了视觉描述组装与融合 (VDAF)。它根据目标对象对 GMM 中类别属性组件的响应动态地组装属性，并自适应地融合文本原型以生成最终的鲁棒查询原型。这个包含丰富、特定目标的视觉属性描述的查询原型，被用于激活视觉特征以获得卓越的 CAMs。此外，为了增强解码器特征的语义一致性，我们设计了解码器语义增强 (DSE) 模块，该模块使用来自 GMM 的类别原型作为语义锚点，在解码器中进行对比学习。

![](../../../../../../99_Assets%20(资源文件)/images/9797a2ea56505b4768ff09ff6df54062.png)

图2. 我们框架的概览。首先，从解码器预测中提取实例原型，并通过(a)中所示的零样本过滤器收集到可靠批次中。然后，VAMD (b) 使用 INN 将可靠的原型映射到一个层次化 GMM (H-GMM) 上，该模型建模了类间关系并解耦了类内属性。随后，VDAF (c) 根据属性响应构建特定于实例的视觉原型，将其与文本原型动态融合以创建查询原型，并生成用于解码器监督的 CAM。此外，DSE (d) 从 H-GMM 中提取全局类别原型，并将它们用作对比学习的锚点，以增强适配器嵌入的语义一致性。

图片内容解释：图2详细展示了VDA方法的四个主要模块的处理流程。(a) 整体框架 (Overall Framework) 显示输入图像经过冻结的CLIP视觉编码器，一方面送到适配器和解码器产生预测和实例原型，另一方面送到基于零样本的过滤器 (Zero-shot Filter) 中停止梯度并收集可靠的实例原型批次。(b) 视觉属性建模与解耦 (VAMD) 模块接收这些批次，使用可逆神经网络 (INN) 将特征映射到潜在空间中，通过构建类间 (Inter-Class) 和类内 (Intra-Class) 的高斯混合模型，解耦出诸如颜色、形状等潜在的属性原型。(c) 视觉描述组装与融合 (VDAF) 模块将新的实例原型输入 INN 进行密度估计获得各个属性的响应强度，以此组装成特定实例的视觉原型，然后和CLIP文本编码器得到的文本原型进行融合，产生最终的查询原型 (Query Prototype) 以生成CAM。(d) 解码器语义增强 (DSE) 模块提取高斯混合模型中的全局类别原型作为锚点 (Class Anchors)，通过对比损失函数对适配后的实例原型进行推拉操作 (Pull/Push)，以增强语义一致性。

> [!note] 我的理解：框架训练流与梯度流
> 理解这个 single-stage WSSS 框架时，不能只看图中的模块连线，而要把“谁生成监督信号、谁接受监督、梯度回到哪里、推理时还保留什么”分开看。需要特别注意的是，图 2 中并没有画出一条从 CAM 直接连到 decoder 或分割头的箭头；CAM 是 VDAF 右侧生成的输出结果。论文文字里说它 “generates a CAM for decoder supervision”，更精确的意思是：VDAF 生成 CAM，CAM 再被精细化为密集伪标签 $\hat{P}$，然后 $\hat{P}$ 通过 $\mathcal{L}_{\mathrm{ce}}=\mathrm{CE}(P,\hat{P})$ 监督 decoder 的预测 $P$。因此，这里的监督关系是“CAM/refined pseudo-label 作为训练目标”，而不是“CAM 特征流入 decoder”。
>
> 具体来看，分割网络这条训练流是：冻结的 CLIP 图像编码器产生视觉特征 $V$，adapter 将其转为 adapted visual features，decoder 基于这些特征输出当前分割预测 $P$；同时，当前预测 $P$ 会作为 mask，在原始 CLIP visual features 上做 mask average pooling，得到 instance prototype。这个 instance prototype 一方面进入 zero-shot filter，经过筛选后形成 reliable prototype batch，用于训练 VAMD；另一方面也进入 VDAF，和文本原型一起生成 query prototype。query prototype 再和 CLIP visual features 计算余弦相似度得到 CAM。随后，论文在文字中补充说明：这些 CAM 会被 refined 成密集伪标签 $\hat{P}$，用来监督 decoder prediction $P$。所以从 CE loss 的角度看，$\hat{P}$ 更像在线生成的标签，交叉熵梯度主要回到 adapter/decoder，而不是穿过 CAM、VDAF、VAMD 一路反传。
>
> VAMD 是另一条相对独立的概率模型训练流。它使用 reliable prototype batch 训练 INN/H-GMM，通过 $\mathcal{L}_{\mathrm{inn}}=\mathcal{L}_{\mathrm{inter}}+\mathcal{L}_{\mathrm{dis}}+\mathcal{L}_{\mathrm{intra}}$ 来学习类间中心和类内属性组件。这里 VAMD 确实是可学习的，但它的反向传播主要更新 INN 和 GMM 相关参数；论文在 3.5 中明确说明 INN 和分割网络的优化是完全独立的，来自 INN 的梯度不会反向传播到 segmentation network。图中 zero-shot filter 到 reliable prototype batch 再到 VAMD 的路径旁边也标了 Stop Gradient，强调 reliable prototypes 被当成训练概率模型的样本，而不是让 VAMD loss 反向拖动前面的 decoder。
>
> VDAF 可以理解为“使用 VAMD 学到的视觉属性词典来组装当前实例的查询”。当输入一个当前 instance prototype 时，VDAF 先通过 INN 和 intra-class GMM 计算它对多个属性组件的响应 $\omega_i^k$，再把 latent attribute prototypes 经由 INN 逆映射回 CLIP 视觉空间，得到 attribute prototypes，并按响应强度加权求和得到 visual prototype $A_k^{\mathrm{vis}}$。由于当前实例原型来自预测 mask，可能不完整或有噪声，作者没有完全丢掉文本原型 $T_k$，而是用 density estimation 得到的 $\alpha_k$ 在 $T_k$ 和 $A_k^{\mathrm{vis}}$ 之间动态融合，生成最终的 query prototype $Q_k$。这个 $Q_k$ 替代公式 (1) 中的静态文本原型 $T_k$ 来生成 CAM，这是 VDAF 的核心作用。
>
> 因此，这个框架里确实存在一种“左脚踩右脚”的自训练关系：当前 decoder prediction 参与产生 instance prototype，instance prototype 影响 VDAF 的 query prototype，query prototype 影响 CAM，CAM refined 之后形成的伪标签又监督 decoder prediction。但这个闭环不是毫无约束的端到端互相拉扯，而是被几个机制稳住了：zero-shot filter 只收集较可靠的 prototypes，VAMD 用独立的概率建模损失训练且 stop-gradient，VDAF 中视觉原型还会通过文本原型兜底，DSE 则通过 class anchors 对 adapted instance prototype 做对比学习来增强 adapter embedding 的语义一致性。推理阶段论文明确说明 INN-related components 会被移除，这也说明 VAMD/VDAF/INN 主要是训练时用于构造更好 CAM 监督和语义锚点的 teacher machinery，最终真正用于输出分割结果的是训练好的 adapter 与 decoder。

### 3.2. 视觉属性建模与解耦

**可靠实例原型过滤。** 我们的目标是在 CLIP 空间内解耦视觉属性并将它们组装成视觉描述。考虑到不同图像中的对象实例携带多样的类属性，我们采用实例原型来捕获这种属性信息。具体来说，对于给定具有冻结的 CLIP 视觉特征 $V$ 的图像 $I$，类别 $k$ 的实例原型 $P_k \in \mathbb{R}^{1 * D}$ 可以通过掩码平均池化计算得出：

$$
P_k = \frac{\sum_{x=1,y=1}^{h,w} P_k(x, y) * V(x, y)}{\sum_{x=1,y=1}^{h,w} P_k(x, y)}, \tag{2}
$$

其中 $P_k$ 是解码器预测掩码，且 $P_k(x, y) = \mathbb{I}[P(x, y) = k]$。$\mathbb{I}$ 是指示函数。然而，这些原型是基于解码器预测得出的，在 WSSS 中可能不可靠且嘈杂。直接使用所有这些原型将严重破坏我们的概率模型学习。因此，为了滤除嘈杂的原型并确保只有可靠、语义纯粹的输入被用于后续建模，我们采用了一种基于 CLIP 零样本分类能力的鲁棒过滤机制。我们首先定义类别文本集合为 $\mathcal{T}_{\mathrm{zs}} = \mathcal{T}_{\mathrm{fg}} \cup \mathcal{T}_{\mathrm{bg}}$，它包含 $K$ 个前景类别提示和 $N$ 个背景类别提示。我们对 $\mathcal{T}_{\mathrm{zs}}$ 中的所有文本进行编码，得到文本嵌入 $T_{\mathrm{zs}}$。背景提示遵循 [17, 48] 中的设置。然后我们为原型 $P_k$ 计算如下的零样本分类得分 $s \in \mathbb{R}^{K+N}$：

$$
s = \mathrm{Softmax}\left(\frac{P_k T_{\mathrm{zs}}^{\top}}{\tau}\right), \tag{3}
$$

其中 $\tau$ 是 CLIP [25] 中的温度参数。然后我们可以收集一批可靠的原型 $\mathcal{B} = \{P_k \mid s_k > \eta\}$，其中 $s_k$ 是类别 $k$ 的零样本分类得分，$\eta$ 是置信度阈值。这个过滤步骤对于构建一个用于我们的属性建模和解耦的可靠且语义纯粹的潜在空间至关重要。

> [!note] 我的理解：为什么 zero-shot filter 能过滤原型噪声
> 这里用到的是冻结 CLIP 的 zero-shot classification 能力，而不是重新训练 CLIP。可以把 $T_{\mathrm{zs}}$ 理解成一个候选概念表：它不仅包含 $K$ 个前景类别文本，例如 “a photo of a dog”、“a photo of a train”，还包含 $N$ 个背景文本，例如 sky、grass、water、wall 等。所有这些文本经过 CLIP text encoder 后堆叠成 $T_{\mathrm{zs}} \in \mathbb{R}^{(K+N)\times D}$。当某个由预测 mask 池化出来的 instance prototype $P_k$ 与 $T_{\mathrm{zs}}^\top$ 相乘时，本质上就是在问：这个视觉原型更像前景类别 $k$，还是更像其他前景类或背景概念？
>
> 因此，$s_k>\eta$ 的含义不是简单地说“这个区域有响应”，而是说“这个由当前预测 mask 提取出来的视觉原型，在 CLIP 的图文对齐空间里确实很像类别 $k$”。如果 decoder 的 mask 错把草地、天空等背景区域当成了火车的一部分，那么池化得到的 $P_k$ 很可能会和 grass、sky 这类背景 prompt 更相似，而不是和 train prompt 更相似，此时类别 $k$ 的分数 $s_k$ 就会低于阈值，被过滤掉。背景 prompt 在这里相当于给噪声区域提供了一个“出口”，避免所有背景特征都被迫归到某个前景类别里。
>
> 这个过滤器的重要性在于：后续 VAMD 要用这些 prototypes 去训练 H-GMM，如果把大量错误或混杂的 prototype 放进去，GMM 学到的类中心和类内属性组件就会被污染。zero-shot filter 的作用就是先借助 CLIP 已有的语义判断能力，尽量只保留语义纯度较高的 instance prototypes，再让 VAMD 在这些较干净的样本上做概率建模。

**层次化高斯混合建模。** 我们的核心洞见是用视觉属性的组装来替代文本描述。然而，正如我们之前讨论的，视觉属性在原始 CLIP 视觉空间中高度分散和纠缠。为了将这种复杂的视觉特征分布解耦为显式属性并进行聚合，我们提出使用 INN 将其映射到一个分层高斯混合模型 (H-GMM) 上。GMM 天然假设一个复杂的分布（目标类）是由多个简单的 GMM 组件（类属性）的混合体构成的。我们选择 INN 是因为它的双射性质，这允许精确的概率估计，并确保我们可以将潜在属性映射回 CLIP 视觉空间。

> [!note] 我的理解：这一段到底在建什么模型
> 这里的 H-GMM 可以理解为两层“整理空间”的机制：第一层 inter-class GMM 先把不同类别的 reliable instance prototypes 整理成 $K$ 个类别中心；第二层 intra-class GMM 再在每个类别内部拆出 $M$ 个属性组件。需要区分的是，GMM/INN/likelihood 这些是前置数学工具，作者的改造在于把它们用到 WSSS 的 reliable prototypes 上，并把 GMM 组件解释为“类别中心”和“类内视觉属性原型”。

然而，同时优化所有类别及其所有细粒度属性的单一 GMM 是一个复杂且不稳定的优化问题。因此，为了将这一复杂任务分解成更稳定的子问题，我们采用了渐进式的学习策略。我们首先通过建立稳定的类中心来构建类间 GMM，然后才在这个稳定的基础上建立细粒度的类内 GMMs。

我们首先使用 INN 来建模一个类间 GMM，以为潜在空间中的 $K$ 个类别建立稳定的类中心。具体而言，INN 学习一个双射映射 $f_\theta : X \to Z$，表示为 $z = f_\theta(x)$，以将原始特征 $X$ 映射到潜在空间 $Z$ 中，并且它通过最小化标准的负对数似然 (NLL) 损失 $\mathcal{L}_{\mathrm{nll}}$ [9, 14] 来进行训练：

$$
\mathcal{L}_{\mathrm{nll}} = \mathbb{E}_{x \sim X}[-\log p_Z(f_\theta(x)) - \log |\det J|], \tag{4}
$$

> [!note] 前置数学：为什么 INN 会出现 NLL 和 Jacobian
> 公式 (4) 属于 normalizing flow / INN 的基础数学。$x$ 是原始空间中的输入特征，这里对应 reliable instance prototype；$z=f_\theta(x)$ 是 INN 映射后的潜在变量；$p_Z$ 是作者希望潜在空间服从的先验分布；$J$ 是映射 $f_\theta$ 对 $x$ 的雅可比矩阵。由于 INN 是双射映射，概率密度可以通过变量替换公式计算，所以损失里会出现 $-\log p_Z(f_\theta(x))$ 和 $-\log|\det J|$。前者要求映射后的 $z$ 落在高概率区域，后者修正空间变换造成的体积变化。这里的 NLL 不是分割任务特有的，而是可逆网络做密度估计时的标准形式。

其中 $p_Z(z) = \sum_{k=1}^K \pi_k \cdot \mathcal{N}(z|\mu_k, \Sigma_k)$ 意味着将潜在分布 $p_Z(z)$ 建模为具有 $K$ 个组件的高斯混合模型 (GMM)，而 $J$ 是 $f_\theta(x)$ 的雅可比矩阵。为了提高优化的稳定性并简化计算，我们将所有协方差矩阵设置为单位矩阵，即 $\Sigma_k = \mathbb{I}$。此外，混合权重 $\pi_k$ 被参数化为可学习的参数。这是通过对一个可学习的对数几率向量 $\psi \in \mathbb{R}^K$ 应用 Softmax 函数来实现的，使得 $\pi_k = \mathrm{Softmax}(\psi)_k$。将这个 GMM 先验代入 NLL 损失中（详细推导见附录），我们获得了如下的类间最大对数似然损失 $\mathcal{L}_{\mathrm{inter}}$：

$$
\mathcal{L}_{\mathrm{inter}} = \mathbb{E}_{x \sim \mathcal{B}}[-\underset{k}{\mathrm{LSE}}(c_k - E_k(f_\theta(x), \mu_k)) - \log |\det J|], \tag{5}
$$

其中 $E_k(z, \mu_k) = \frac{1}{2} ||z - \mu_k||_2^2$ 是第 $k$ 个组件的负对数似然，$\mathrm{LSE}(\cdot)$ 表示 logsumexp 操作。可学习的参数是对数权重 $c_k = \log(\pi_k)$ 和均值中心 $\mu_k$。此外，为了将由 $\mathcal{L}_{\mathrm{inter}}$ 学习的 GMM 组件与它们对应的类别关联起来，我们引入了一个类判别损失：

$$
\mathcal{L}_{\mathrm{dis}} = -\mathbb{E}_{(x,k) \sim \mathcal{B}}[-\underset{k}{\mathrm{LSM}}(c_{k'} - E_k(f_\theta(x), \mu_{k'})) - c_k], \tag{6}
$$

其中 $k'$ 表示除类别 $k$ 之外的其他类别，且 $\mathrm{LSM}(\cdot)$ 是 logsoftmax 操作。它将 $x$ 拉向与其对应的第 $k$ 个组件关联，并将其推离其他组件。

> [!note] 作者改造：从普通 GMM 到“类别中心”
> 公式 (5) 和 (6) 是作者把 flow/GMM 用到分割原型建模里的关键改造。$\mathcal{B}$ 是前面 zero-shot filter 筛出来的 reliable prototype batch；$K$ 是数据集前景类别数；$\mu_k$ 是潜在空间中第 $k$ 类的中心；$\pi_k$ 或 $c_k$ 控制第 $k$ 个高斯组件的权重。$\mathcal{L}_{\mathrm{inter}}$ 只要求样本在整个 GMM 下概率高，但 GMM 组件天然有“编号可交换”的问题，因此还需要 $\mathcal{L}_{\mathrm{dis}}$ 把类别标签 $k$ 和组件 $\mu_k$ 对齐。直觉上，前者让样本落进某个合理的团，后者规定 bird 的样本应该落到 bird 的团，而不是落到 dog 或 train 的团。

类间 GMM 的优化稳定了主要的类中心。随后，在这个稳定的基础之上，我们通过将每个类间组件扩展为其自己的包含 $M$ 个组件的类内 GMM 来进行构建，即 $p(Z|k) = \sum_{i=1}^M \pi_i(k)\mathcal{N}(\mu_{i}^{k}, \mathbb{I})$。每个 $\mu_{i}^{k}$ 代表类别 $k$ 的一个独特潜在视觉属性，而 $\pi_i(k)$ 是类别 $k$ 中第 $i$ 个组件的混合权重。然而，从头开始独立学习所有 $M \times K$ 个属性中心是不稳定的。为了确保这些属性的优化不会灾难性地偏移已经稳定的类中心 $\mu_k$（现在表示为 $\mu_{1}^{k}$），我们将其他 $M - 1$ 个属性中心参数化为相对于这个主中心的偏移量 $\{\Delta\mu_{i}^{k}\}$：

$$
\mu_{i}^{k} = \mu_{1}^{k} + \Delta\mu_{i}^{k}, \quad \text{其中 } \Delta\mu_{1}^{k} = 0. \tag{7}
$$

> [!note] 我的理解：为什么要用偏移量学属性中心
> 公式 (7) 是稳定训练的设计，不是 GMM 的基础必然形式。作者不是让所有 $M\times K$ 个属性中心从零开始乱跑，而是先把类中心 $\mu_1^k$ 当成锚点，再学习其他属性中心相对它的偏移 $\Delta\mu_i^k$。这样可以理解为：先确定“这是 bird 这个大区域”，再在 bird 区域内部学习颜色、姿态、局部结构等变化方向。这里的“属性”没有人工语义标签，并不一定真的对应人类命名的颜色或部件，而是类内视觉分布中被模型拆出来的潜在组件。

这种参数化使得优化能够只专注于学习代表属性变化的偏移量。这些偏移向量然后通过以下类内损失 $\mathcal{L}_{\mathrm{intra}}$ 进行优化：

$$
\mathcal{L}_{\mathrm{intra}} = \mathbb{E}_{(x,k) \sim \mathcal{B}}[-\underset{i}{\mathrm{LSE}}(c_{i}^{k} - E(f_\theta(x), \mu_{i}^{k})) - \log |\det J|], \tag{8}
$$

这里 $c_{i}^{k} = \log(\pi_i(k))$。最后，我们对这些目标采用渐进式训练策略。INN 首先仅使用 $\mathcal{L}_{\mathrm{inter}}$ 和 $\mathcal{L}_{\mathrm{dis}}$ 进行几次迭代预热，以建模类间 GMM。在预热之后，我们引入 $\mathcal{L}_{\mathrm{intra}}$ 并使用组合损失来优化 INN：

$$
\mathcal{L}_{\mathrm{inn}} = \mathcal{L}_{\mathrm{inter}} + \mathcal{L}_{\mathrm{dis}} + \mathcal{L}_{\mathrm{intra}}. \tag{9}
$$

该过程产生了一个显式结构化的 H-GMM，它建模了稳定的类间关系并解耦了细粒度的类内视觉属性。

> [!note] 总结：公式 (8)(9) 在训练什么
> 公式 (8) 是类内 GMM 的 likelihood 目标：给定一个属于类别 $k$ 的 prototype，模型希望它在类别 $k$ 的 $M$ 个属性组件中至少能被某些组件很好解释。$\mathrm{LSE}$ 仍然来自混合模型的 log-sum-exp 计算，$c_i^k$ 是类别 $k$ 内第 $i$ 个属性组件的 log mixture weight。公式 (9) 则说明训练是渐进式合并目标：先用 $\mathcal{L}_{\mathrm{inter}}+\mathcal{L}_{\mathrm{dis}}$ 学稳类别中心，再加入 $\mathcal{L}_{\mathrm{intra}}$ 拆类内属性。最终得到的 H-GMM 不是直接输出分割 mask，而是为下一节 VDAF 提供一个“视觉属性词典”。

### 3.3. 视觉描述组装与融合

一旦 INN 训练完成，它就有效地为每个类别学习了一个“视觉属性词汇表”。类内 GMMs 中的组件充当特定潜在视觉属性的分布，这些可以被认为是构成该类的核心视觉构建块。对于任何给定的对象，我们不再依赖静态的、一刀切的文本锚点 $T_k$，而是通过量化对象在多大程度上展现了这些学习到的视觉属性，来动态地组装一个特定目标的视觉描述。这个组装过程包括以下三个步骤：

**视觉属性原型检索。** 学习到的属性原型 $\{\mu_{i}^{k}\}$ 存在于潜在空间 $Z$ 中。然而，要被用作查询，它们最终必须与位于视觉空间中的原始 CLIP 视觉特征 $V$ 进行比较。因此，我们利用 INN 的可逆性 $g_\theta = f_\theta^{-1}$ 将这些潜在属性中心映射回 CLIP 视觉空间：

$$
a_{i}^{k} = g_\theta(\mu_{i}^{k}). \tag{10}
$$

这 $M$ 个原型代表了 CLIP 视觉空间中类别 $k$ 的某些抽象属性的压缩，例如颜色、形状和动作等。

**属性响应强度计算。** 上一步检索了视觉词汇表 $\{a_{i}^{k}\}$。然而，图像中的某个特定实例仅会展现出这些通用属性的一个子集。例如，该对象可能与某种特定属性强烈匹配，但与另一种属性不匹配。因此，在我们可以组装描述之前，我们必须首先量化当前对象原型 $P_k$ 对这 $M$ 个属性中每一个属性的响应强度。这些响应强度将作为精确的混合权重，允许我们构建一个仅包含对象实际拥有的属性的复合原型。

具体而言，对于给定的实例原型 $P_k$。为了确定它与发现的属性之间的关系，我们首先将其映射到潜在空间：$z_k = f_\theta(P_k)$。接下来，我们通过计算潜在原型 $z_k$ 属于 $M$ 个类内 GMM 组件中每一个组件的后验概率来计算属性响应强度。这个后验概率 $\omega_{i}^{k}$ 的计算方式如下：

$$
\omega_{i}^{k}(z_k) = \mathrm{Softmax}_i \left(-\frac{||z_k - \mu_{i}^{k}||_2^2}{2} + c_{i}^{k}\right). \tag{11}
$$

每个 $\omega_{i}^{k}(z_k)$ 代表目标原型 $P_k$ 对类别 $k$ 内第 $i$ 个视觉属性的响应。现在既然这些特定实例的响应强度被量化了，我们就拥有了构建最终复合视觉描述的所有必要组件。

**复合视觉属性组装。** 现在我们可以组装最终的复合视觉描述 $A_{k}^{\mathrm{vis}}$，它用作专门针对对象的特定视觉表现量身定制的动态查询原型。这个复合原型被计算为所有视觉属性原型 $\{a_{i}^{k}\}$ 的加权总和，使用它们对应的属性响应强度 $\omega_{i}^{k}(z_k)$ 作为权重：

$$
A_{k}^{\mathrm{vis}}(P_k) = \sum_{i=1}^M \omega_{i}^{k}(z_k) \cdot a_{i}^{k}. \tag{12}
$$

结果向量 $A_{k}^{\mathrm{vis}}(P_k)$ 是一个动态的、特定于实例的视觉描述，纯粹从学习到的视觉属性空间中组装而来。

**基于密度的自适应原型融合。** 理想情况下，高质量的 $A_{k}^{\mathrm{vis}}$ 总会优于文本。然而，我们必须面对 WSSS 的一个核心挑战：$A_{k}^{\mathrm{vis}}$ 的质量完全取决于输入原型 $P_k$ 的质量，而该输入原型派生自具有潜在噪声的预测掩码。如果 $P_k$ 是不完整或错误的，$A_{k}^{\mathrm{vis}}$ 将是一个有缺陷、不可靠的描述。尽管模板文本‘一个干净的 [CLASS] 折纸’只能提供通用的语义表示，但它在语义上是稳定的，可以作为一个语义锚点。因此，我们认为不应简单地丢弃可靠的 $T_k$，而是应将其动态融合到视觉原型中。

具体而言，我们定义了一个自适应的、基于密度的权重 $\alpha_k(P_k)$。这个权重衡量了潜在原型 $z_k = f_\theta(P_k)$ 在其对应的类间组件 $\mathcal{N}(\mu_k, \mathbb{I})$ 下的“典型”程度。我们通过获取 $z_k$ 的概率密度，然后将其除以最大概率密度（当 $z_k = \mu_k$ 时）进行归一化来计算它：

$$
\alpha_k(P_k) = \exp \left( -\frac{1}{2} ||z_k - \mu_k||_2^2 \right). \tag{13}
$$

一个较高的 $\alpha_k$ 表明实例原型 $P_k$ 满足其代表性的类分布。因此，融合会对其组装的视觉属性 $A_{k}^{\mathrm{vis}}$ 施加更多的信任。一个较低的 $\alpha_k$ 表明 $P_k$ 远离其代表性类分布。这表示样本质量低或不典型。融合机制随即会回退到稳定的、通用的文本锚点 $T_k$。最终融合的原型 $Q_k$ 是由该分数控制的线性插值：

$$
Q_k = (1 - \alpha_k(P_k))T_k + \alpha_k(P_k)A_{k}^{\mathrm{vis}}(P_k). \tag{14}
$$

这个 $Q_k$ 取代了公式 (1) 计算 CAM 中的静态 $T_k$ 作为查询原型，从而提供了一个动态平衡文本语义普适性与特定于实例的视觉描述的查询原型。这些 CAMs 然后被精细化以生成密集的伪标签 $\hat{P}$ 作为解码器预测 $P$ 的监督。

### 3.4. 解码器语义增强

为了进一步增强解码器嵌入的语义一致性，我们使用一个可学习的适配器将冻结的 CLIP 特征转移到解码器，并引入对比学习，该学习将自适应的特定于实例的特征与来自我们的类间 H-GMM 的稳定的全局类别原型对齐。具体而言，适配器 $f_{\mathrm{adapt}}$ 将冻结的 CLIP 特征 $V$ 映射到 $V_{\mathrm{dec}}$，然后我们可以像公式 (2) 一样通过掩码平均池化得到适配器实例原型 $P_{\mathrm{dec},k}$。全局语义锚点 $G_k$ 是从类间 GMM 的组件中心 $\{\mu_k\}_{k=1}^K$ 派生的，并将它们映射回视觉空间 $V_{k}^{g} = g_\theta(\mu_k)$。然后，我们使用 InfoNCE 损失将适配器实例原型 $P_{\mathrm{dec},k}$ 拉向其对应的全局锚点 $G_k$，同时将其推离其他类的锚点：

$$
\mathcal{L}_{\mathrm{con}} = -\log \frac{\exp(\mathrm{sim}(P_{\mathrm{dec},k}, G_k) / \tau)}{\sum_{j=1}^K \exp(\mathrm{sim}(P_{\mathrm{dec},k}, G_j) / \tau)}, \tag{15}
$$

其中 $\mathrm{sim}(\cdot, \cdot)$ 是余弦相似度，而 $\tau$ 是温度。这个过程将适配器的表示与我们的类间 GMM 所学习到的全局语义关系对齐。这些增强了语义一致性的适配器特征 $V_{\mathrm{dec}}$ 然后被送入解码器以生成最终的分割预测 $P$。

### 3.5. 整体框架训练

我们的框架包括两个主要独立的训练目标。一个是分割网络训练，它使用交叉熵损失 $\mathcal{L}_{\mathrm{ce}} = \mathrm{CE}(P, \hat{P})$ 和我们的对比增强损失 $\mathcal{L}_{\mathrm{con}}$。总损失为：

$$
\mathcal{L}_{\mathrm{seg}} = \mathcal{L}_{\mathrm{ce}} + \lambda \mathcal{L}_{\mathrm{con}}, \tag{16}
$$

其中 $\lambda$ 是损失平衡项。另一个目标是 INN 训练，它使用公式 (9) 中的损失 $\mathcal{L}_{\mathrm{inn}}$ 来训练我们的概率模型。在我们的训练流水线中，INN 和分割网络的优化是完全独立的，即来自 INN 的梯度不会反向传播到分割网络。并且在推理过程中，与 INN 相关的组件被移除。
