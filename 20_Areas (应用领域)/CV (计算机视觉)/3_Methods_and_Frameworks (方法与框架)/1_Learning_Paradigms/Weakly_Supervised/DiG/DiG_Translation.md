# Diffusion-Guided Weakly Supervised Semantic Segmentation

扩散引导的弱监督语义分割

**摘要。** 使用分类标签的弱监督语义分割（WSSS）通常使用类激活图（CAM）基于卷积神经网络（CNN）定位目标。由于感受野有限，基于 CNN 的 CAM 往往无法定位完整目标。视觉 Transformer（ViT）的出现凭借其优越性能缓解了这一问题，但 ViT 缺乏局部性又带来了新的挑战。受去噪扩散概率模型（DDPM）能够捕获高层语义信息这一能力的启发，我们将扩散模型引入 WSSS 来解决该问题。首先，为融合并在语义上对齐 DDPM 与 ViT 之间的信息，我们设计了局部性融合交叉注意力（Locality Fusion Cross Attention，LFCA）模块。LFCA 利用从预训练 DDPM 去噪过程中聚合的特征，生成扩散 CAM（Diffusion-CAM），以向来自 ViT 的 CAM（ViT-CAM）提供局部性信息。其次，通过向原始图像添加噪声并用 DDPM 去噪，我们获得可作为增强样本使用的去噪图像。为了有效引导 ViT 挖掘图块之间的关系，我们在原始图像与去噪图像的输出之间设计了图块亲和力一致性（Patch Affinity Consistency，PAC）。大量消融研究支持所提方法的优越性。我们的方法在 WSSS 中广泛使用的两个数据集 PASCAL VOC 2012 和 MS-COCO 2014 上取得了新的最先进性能。代码见 https://github.com/yoon307/DiG。

**关键词：** 去噪扩散概率模型 · 弱监督语义分割

## 1 引言

为缓解全监督语义分割模型中劳动密集且成本高昂的标注过程，研究者提出使用弱标签训练语义分割。借助图像级标签[25,32,42,59,63–65]、涂鸦[33,50]和边界框[24,28]等低廉且易得的标签，弱监督语义分割（WSSS）方法得到积极探索，目标是达到全监督性能。

![](../../../../../../99_Assets%20%28资源文件%29/images/dig_fig1.png)

**图 1：** (a) 来自预训练扩散模型特征的 K-Means 聚类结果。右图中相同颜色表示位于对应像素的特征属于同一聚类。(b) 所提框架的简化流程。为解决 ViT 的局部性问题，局部性融合交叉注意力（LFCA）模块提供语义上定位良好的扩散 CAM，以改善 ViT-CAM 的局部性。

**图片内容解释：** (a)以火车和摩托车图像显示扩散特征在空间上形成连贯语义区域；(b)显示冻结的扩散编码器经 K-Means 提供局部性信息，ViT 提供语义对齐信息，二者在 LFCA 中融合。

尽管使用图像级标签训练语义分割模型是最实用的设置，提取定位信息仍很困难，因为标签只告知某些目标是否存在。为从分类器捕获空间信息并生成伪像素级标签，大多数 WSSS 工作采用基于 CNN 的类激活图（CAM）[67]。然而，基于 CNN 的 CAM 由于其固有特性，不仅聚焦于最具判别性的区域（即激活不足），还具有不精确的目标边界。为使获得的 CAM 能作为语义分割的伪标签，各项研究致力于提高 CAM 的质量。

转而使用视觉 Transformer（ViT）提取 CAM，借助多头自注意力（MHSA）机制获得全局特征，能够有效减轻 CNN-CAM 的激活不足问题并显示出可喜结果。然而，用 ViT 提取 CAM 也带来另一挑战：相较于 CNN，ViT 可能不擅长捕获细粒度局部模式。一些方法通过改进位置编码[54]或采用结合 ViT 与 CNN 的混合模型[19]，缓解 ViT 的这种局部性不足[19,31,48,54]。不过在 WSSS 中，对于解决 ViT 局部性缺失所致 CAM 性能下降的关注相对较少。

与此同时，去噪扩散概率模型（DDPM）[21]已在图像生成、修复和超分辨率等多种生成建模任务中展现出良好结果。已有研究表明，DDPM 的特征表示能够捕获对语义分割有价值的高层语义信息[3]。考虑到扩散模型也可以以无标签方式训练，它适合用于 WSSS。基于这一动机，我们提出一个新颖的 WSSS 框架：将语义上聚类良好的 DDPM 特征与 ViT 融合，以提高 CAM 质量；还提出一种自监督学习技术，利用 DDPM 加噪、去噪过程恢复的图像进一步改善 CAM 质量。训练 DDPM 时，我们不使用外部数据集和目标数据集图像之外的额外监督。

如图 1 所示，我们以 ViT 为分类骨干、以基于 U-Net 的网络为 DDPM，且只使用图像级监督。尽管 DDPM 如图 1(a)所示能够以较高局部性捕获语义信息，它仍需与 ViT 结构的语义对齐。因此，我们提出 LFCA 模块实现这种对齐，并用它改善 ViT-CAM 的局部性。具体地，LFCA 从 ViT 中类别对齐的图块取查询 token，从扩散特征取键/值 token，并以交叉注意力融合不同特征表示。为引导 LFCA 的输出 token，我们施加分类损失，并以 Kullback-Leibler 散度（KLD）从 ViT 模型蒸馏知识，从而连续地将 LFCA 语义对齐到 ViT。充分对齐后，由 LFCA 输出 token 生成的扩散 CAM 用于训练 ViT-CAM。此外，我们引入 PAC，旨在训练 ViT 鲁棒地学习图块间关系。当图像由 DDPM 加噪并去噪时，图像发生人类难以察觉但显著的退化，同时语义特征得以保留。因局部语义在图像退化期间不变，去噪图像与原图的图块亲和力应当相似。PAC 对原图与退化图像的图块亲和力施加相似性约束，以有效引导仅靠分类损失难以正则化的图块亲和力。

本文主要贡献如下：

- 提出首个成功将扩散模型融合至基于 ViT 的 WSSS 模型的框架；鉴于任务性质，方法也只利用图像级监督。
- 提出 LFCA 模块，在将预训练扩散特征对齐至 ViT 语义类别的同时，挖掘其潜力。
- 设计简单而有效的 PAC：将去噪图像作为空间与语义一致的增强样本，更好地引导用于细化 CAM 的图块亲和力。

PASCAL VOC 2012[15]和 MS COCO[34]的实验结果，通过建立新的最先进性能验证了所提方法的优越性。

## 2 相关工作

### 2.1 扩散模型

扩散模型[21,43]是旨在逼近真实图像分布的概率生成模型。给定图像被连续高斯噪声破坏时，模型学习恢复该图像。DDPM[21]建模前向扩散过程，并提出以简化方式逆转扩散过程的训练目标。为加快图像生成推理，DDIM[44]提出具有非马尔可夫链的隐式概率模型。Dhariwal 等人[13]受 GAN 模型架构取得进展而扩散模型未获同等关注的观察启发，提出面向扩散模型的新架构；凭借强大的模型和修改后的目标，扩散模型首次在样本质量与多样性上超过 GAN。本文遵循 DDPM 过程，采用[13]提出的基于 U-Net 的扩散模型。

### 2.2 基于生成模型的图像分割

与 WSSS 旨在减轻语义分割标注负担的目标相似，已有大量工作利用生成模型做语义分割。DDPM 出现前，多数研究专注于利用 GAN 表示进行语义分割。Tritrong 等人[49]提出利用 GAN 表示的一次学习语义部件分割框架。Xu 等人[58]和 Galeev 等人[17]表明 GAN 以简单方式编码图像语义，施加线性变换或构建轻量解码器即可将 GAN 特征投影为语义分割图。随着在图像合成上超越 GAN 的扩散模型出现[13]，许多研究[3,4,47,53]致力于将扩散模型用于语义分割。Baranchuk 等人[3]表明扩散模型能够用于语义分割，同时捕获对其他任务有价值的高层语义；Rahman 等人[38,53]提出基于扩散模型的医学图像分割框架。

### 2.3 弱监督语义分割

**CAM 改进。** 大多数使用图像级标签的 WSSS 方法用 CAM[67]定位图像中的目标。然而，未经细化的 CAM 无法发现较不具判别性的区域，并且目标边界不精确。为改善 CAM，研究提出：利用跨图像语义关系[16,30,45]、注意力机制[37,52,55]、将图像拆分为互补图块[65]、局部—全局视图图像间的一致性[22]、引入额外类别[5,60]，以及重建器与分类器间的对抗学习[26]。除这些方法外，还引入了基于“擦除并寻找”机制的对抗擦除（AE）方法[25,29,46,62,66]。除扩展 CAM 外，基于对比学习的方法[8,57,68]引导 CAM 获得更准确目标边界。随着 ViT 显示出有希望的定位能力，许多 WSSS 工作[18,35,40–42,59,60]采用 ViT 而非 CNN 作为定位骨干。为提取注意力图，MCTformer[59]将单一类别 token 扩展为多个类别 token，并提出在 ViT 内以类别特定方式提取和细化 CAM 的框架。Xu 等人[60]和 Lin 等人[35]引入视觉—语言（VL）模型以丰富 ViT 的类别表示能力；但这些方法使用的语言模型在大型外部数据集上训练、引入额外语言监督，故不宜与仅图像级监督方法比较。尽管已有许多基于 ViT 的 WSSS 方法，据我们所知，尚无研究处理 ViT 的局部性缺失。

**CAM 细化。** 为进一步细化 CAM 质量以作为语义分割标签，[1,2]利用邻近像素间语义亲和力，[7]使用边界信息。MARS[23]利用无监督语义分割模型的特征生成去偏伪标签。Mat-Label[51]将图像抠图带入 WSSS 伪标签生成过程。BECO[39]提出通过标签混合改善语义边界处伪标签的协同训练框架。由于这些后处理技术依赖 CAM 质量并可与 CAM 改进方法整合，本研究重点放在 **CAM 改进阶段**，而非 CAM 细化阶段。

## 3 方法

### 3.1 预备知识

**去噪扩散概率模型。** 本文采用扩散模型[21,43]为提取 CAM 的分类器提供结构良好的高层语义信息。DDPM 包括前向过程 $q(x_t|x_{t-1})$：在 $T$ 步中逐步将输入数据 $x_0$ 破坏为高斯噪声 $x_T\sim\mathcal{N}(0,1)$；以及反向（即去噪）过程 $p_\theta(x_{t-1}|x_t)$：预测前向扩散过程中的噪声，获得更干净的样本 $x_t$。

Ho 等人[21]经验发现，网络直接预测噪声 $\epsilon_\theta(x_t,t)$ 而非预测均值 $\mu_\theta(x_t,t)$，能生成更多高频细节，故本文遵循这一先前工作。Ho 等人[21]的简化训练目标为：

$$
\mathcal{L}_{\mathrm{simple}}=\mathbb{E}_{t,x_0,\epsilon}[\lVert\epsilon-\epsilon_\theta(x_t,t)\rVert^2]. \tag{1}
$$

因此，在扩散模型训练过程中，除图像本身外不使用任何标签。关于 DDPM 的额外数学细节见补充材料。

**用于 WSSS 的视觉 Transformer。** 为把输入图像 $x_0$ 输入 ViT，先将图像分为 $N\times N$ 个图块，继而嵌入为图块 token $T_p\in\mathbb{R}^{N^2\times D}$，其中 $D$ 为嵌入维度。随后，将 $C$ 个类别 token $T_c\in\mathbb{R}^{C\times D}$ 与图块 token 拼接并加入位置嵌入。拼接 token 输入 $T_{\mathrm{in}}\in\mathbb{R}^{(C+N^2)\times D}$ 按[59]送入 ViT；输出 token $T_{\mathrm{out}}\in\mathbb{R}^{(C+N^2)\times D}$ 被拆分为类别 token $T^c_{\mathrm{out}}\in\mathbb{R}^{C\times D}$ 与图块 token $T^p_{\mathrm{out}}\in\mathbb{R}^{N^2\times D}$。对 $T^c_{\mathrm{out}}$ 池化得到类别预测 $\hat y_c$。将图块 token $T^p_{\mathrm{out}}$ 重塑并施加具有 $C$ 个通道的卷积层，可得到 CAM $A\in\mathbb{R}^{N\times N\times C}$ 和 ViT 的类别预测 $\hat y_p$。其中：

$$
\hat y_p=\operatorname{TopK}(\operatorname{ReLU}(\dot A),K)-\operatorname{TopK}(\operatorname{ReLU}(-\dot A),K). \tag{2}
$$

其中，$\dot A$ 和 $K$ 分别为生成 $A$ 前、施加 ReLU 前的图块级特征以及选择数量。$\operatorname{TopK}(\cdot)$ 是 Top-K 池化操作：沿每个通道的空间维选择并平均 $K$ 个最高值。虽然基线[59]用全局平均池化计算类别预测，本文用式(2)修改池化过程。分类损失 $\mathcal{L}_{\mathrm{cls-vit}}=\mathcal{L}_{\mathrm{cls}}(\hat y_c,y)+\mathcal{L}_{\mathrm{cls}}(\hat y_p,y)$ 用于监督 ViT 的类别预测 $\hat y_c,\hat y_p$；其中 $\mathcal{L}_{\mathrm{cls}}(\hat y,y)$ 表示预测 $\hat y$ 与标签 $y$ 之间的多标签 soft margin 损失。

此外，可从 ViT 提取 token 对 token 注意力图 $A_{t2t}\in\mathbb{R}^{(C+N^2)\times(C+N^2)}$。由此获得类别到图块注意力 $A_{c2p}\in\mathbb{R}^{C\times N^2}$ 及图块到图块注意力 $A_{p2p}\in\mathbb{R}^{N^2\times N^2}$。为聚合全局—局部信息，融合 ViT 的 $L$ 层类别到图块注意力，得到类别特定注意力图 $A_{\mathrm{att}}=\sum_{l=1}^{L}A_{c2p}^{l}$；将其转置、重塑为 $\mathbb{R}^{N\times N\times C}$。类似地，融合并重塑 $L$ 层图块到图块注意力，得到 $A_{\mathrm{aff}}\in\mathbb{R}^{N\times N\times N\times N}$，表示图块 token 间的亲和力：$A_{\mathrm{aff}}=\sum_{l=1}^{L}A_{p2p}^{l}$。最后，利用 $A_{\mathrm{att}}$ 与 $A_{\mathrm{aff}}$ 细化 CAM $A$，生成 $A_{\mathrm{ref}}\in\mathbb{R}^{N\times N\times C}$：

$$
A_{\mathrm{ref}}(i,j,c)=\sum_m^N\sum_n^N A_{\mathrm{aff}}(i,j,m,n)\cdot(A\odot A_{\mathrm{att}})(m,n,c), \tag{3}
$$

其中 $\odot$ 表示逐元素乘法，$\cdot$ 表示乘法。

### 3.2 概述

![](../../../../../../99_Assets%20%28资源文件%29/images/dig_fig2.png)

**图 2：** 所提框架的可视化。图像 $x_0$ 与 $C$ 个类别 token 一同输入 ViT 和在不同时间步 $t$ 下冻结的 DDPM。键（$K_f$）和值（$V_f$）由聚合扩散特征构成；查询（$Q$）从 ViT 后部层的图块 token 中提取。$Q,K_f,V_f$ 被用于 LFCA，生成语义对齐特征，进而产生扩散 CAM $A_f$。同时，由 ViT 输出图块 token 创建 CAM $A$。在训练若干 epoch、语义对齐 LFCA 模块后，在 $A$ 与 $A_f$ 间施加 $L_1$ 损失，以传播基于扩散的语义局部性。为简洁起见，图中省略分类损失。

**图片内容解释：** 上方是原图和去噪图共同经过共享 ViT、以 PAC 约束细化 CAM；下方是多时间步噪声图经锁定的扩散 U-Net 生成聚合特征，LFCA 输出扩散 CAM 并以停止梯度方式监督 ViT-CAM。

如图 2 所示，我们提出两种将 DDPM 能力传播至 CAM 的方法。由于使用预训练 DDPM 而不微调，先以式(1)目标训练基于 U-Net 的 DDPM。为将 DDPM 语义和局部性均良好聚类的特征对齐到语义类别，我们设计图 3 所示 LFCA。另将原图 $x_0$ 迭代加噪、再用 DDPM 去噪回原始步骤，得到去噪图像 $\tilde{x}_0$ 作为增强样本；据此提出 PAC，检查干净图像 $x_0$ 与去噪图像 $\tilde{x}_0$ 的图块 token 间亲和力是否一致。

### 3.3 局部性融合交叉注意力

在 LFCA 中，我们聚合将不同时间步 $t$ 的噪声图像 $x_t$ 送入扩散编码器（瓶颈特征）得到的扩散特征 $F_t\in\mathbb{R}^{H_f\times W_f\times D_f}$，其中 $H_f,W_f$ 是扩散特征的高和宽，$D_f$ 是特征维度。不同噪声图像的扩散特征被拼接，再通过一系列卷积和层归一化操作降维，获得聚合扩散特征 $F_{\mathrm{diff}}\in\mathbb{R}^{H_f\times W_f\times D_f}$。尽管 $F_{\mathrm{diff}}$ 能向 ViT 提供有意义的局部性信息，它尚未与 ViT 在语义上对齐。为有效将扩散特征语义与 ViT 特征语义对齐，我们以交叉注意力融合 ViT 层的查询 token 与聚合扩散特征。为计算交叉注意力，先从 ViT 层提取查询 $Q\in\mathbb{R}^{(C+N^2)\times D}$，提供类别对齐语义信息。为从扩散特征提取局部对齐信息，先将 $F_{\mathrm{diff}}$ 重塑为 $T_f\in\mathbb{R}^{N_f^2\times D_f}$，其中 $N_f^2=H_f\times W_f$；从 $F_{\mathrm{diff}}$ 提取键 $K_f\in\mathbb{R}^{N_f^2\times D}$ 和值 $V_f\in\mathbb{R}^{N_f^2\times D_f}$。随后按交叉注意力机制融合特征：

$$
\operatorname{LFCA}(Q,K_f,V_f)=\operatorname{softmax}\left(\frac{QK_f^{\top}}{\sqrt D}\right)V_f. \tag{4}
$$

![](../../../../../../99_Assets%20%28资源文件%29/images/dig_fig3.png)

**图 3：** 局部性融合交叉注意力模块的可视化。该模块使用来自 ViT 层的查询 $Q$ token，通过交叉注意力对扩散特征进行语义对齐。对类别 token 和图块 token 的类别预测都计算 KD 损失 $\mathcal{L}_{\mathrm{kd}}$。扩散 CAM $A_{\mathrm{diff}}$ 用于引导 ViT-CAM，以提供局部性信息。

**图片内容解释：** 多个扩散编码器中间特征先汇聚为 $F_{\mathrm{diff}}$ 并投影为键和值；ViT token 提供查询。交叉注意力输出同时产生类别/图块预测与扩散 CAM，后者经残差与卷积得到。

将融合 token $T_{\mathrm{lfca}}\in\mathbb{R}^{(C+N^2)\times D_f}$ 拆成类别 token $T_{c-\mathrm{diff}}\in\mathbb{R}^{C\times D_f}$ 和图块 token $T_{p-\mathrm{diff}}\in\mathbb{R}^{N^2\times D_f}$，以施加分类损失并生成 CAM。沿 $D_f$ 维池化 $T_{c-\mathrm{diff}}$ 得类别预测 $\hat y_{c-\mathrm{diff}}$；$\hat y_{p-\mathrm{diff}}$ 可按式(2)得到。同时，将 $T_{p-\mathrm{diff}}$ 重塑为 $N\times N\times D_f$ 特征图并插值为 $F_{p-\mathrm{diff}}\in\mathbb{R}^{H_f\times W_f\times D_f}$。$F_{p-\mathrm{diff}}$ 与 $F_{\mathrm{diff}}$ 的残差相加，送入卷积层，获得扩散 CAM $A_{\mathrm{diff}}\in\mathbb{R}^{H_f\times W_f\times C}$ 和类别预测 $\hat y_{p-\mathrm{diff}}$。

为监督扩散特征的类别预测 $\hat y_{c-\mathrm{diff}}$ 和 $\hat y_{p-\mathrm{diff}}$，使用两种损失。首先，对图像级标签 $y$ 施加多标签 soft margin 损失 $\mathcal{L}_{\mathrm{cls}}(\cdot)$。此外，在类别预测间施加 Kullback-Leibler（KL）散度，使扩散特征理解并对齐 ViT 的概率分布。先以温度 $T$ 缩放 $\hat y_c,\hat y_{c-\mathrm{diff}}$，并施加 softmax 函数 $\sigma$ 计算用于 KL 散度的概率分布；KL 散度损失定义为：

$$
\mathcal{L}_{\mathrm{kl}}(\hat y,\hat y_{\mathrm{diff}})=D\bigl(\sigma(\hat y/T)\Vert\sigma(\hat y_{\mathrm{diff}}/T)\bigr), \tag{5}
$$

其中，对于概率分布 $P,Q$，$D(P\Vert Q)=\sum_xP(x)\log\frac{P(x)}{Q(x)}$。最终，为平衡监督 $\hat y_{\mathrm{diff}}$ 的多标签 soft margin 损失和 KL 散度损失，采用平衡参数 $\alpha$：

$$
\mathcal{L}_{\mathrm{kd}}(\hat y,\hat y_{\mathrm{diff}},y)=\alpha\mathcal{L}_{\mathrm{cls}}(\hat y_{\mathrm{diff}},y)+(1-\alpha)T^2\mathcal{L}_{\mathrm{kl}}(\hat y,\hat y_{\mathrm{diff}}). \tag{6}
$$

这里引入 $T^2$ 校正温度缩放。上述方程可同时用于类别 token 与图块 token 的类别预测。训练 LFCA 的整体分类损失为：

$$
\mathcal{L}_{\mathrm{cls-diff}}=\mathcal{L}_{\mathrm{kd}}(\hat y_c,\hat y_{c-\mathrm{diff}},y)+\mathcal{L}_{\mathrm{kd}}(\hat y_p,\hat y_{p-\mathrm{diff}},y). \tag{7}
$$

LFCA 的最终目的是将 DDPM 的局部性信息提供给 ViT。然而在训练早期 LFCA 尚未语义对齐，因此仅在经过设定数量的 epoch 后，使用下式以扩散 CAM $A_{\mathrm{diff}}$ 监督 ViT-CAM $A$：

$$
\mathcal{L}_{\mathrm{lfca}}=\lvert A-A_{\mathrm{diff}}\rvert_1. \tag{8}
$$

其中 $\lvert\cdot\rvert_1$ 表示 $L_1$ 损失。在式(8)中，$\mathcal{L}_{\mathrm{lfca}}$ 不会向 $A_{\mathrm{diff}}$ 反向传播，因为扩散 CAM 仅用作监督；此处通过插值重设 $A_{\mathrm{diff}}$ 的大小以匹配 $A$。

### 3.4 图块亲和力一致性

尽管图块亲和力 $A_{\mathrm{aff}}$ 可用作细化而显著提高 ViT-CAM $A$ 质量，但图块亲和力自身由于没有空间约束、只受分类损失监督，容易提高假阳性激活。为改善图块亲和力，我们将扩散模型用作数据增强形式：它保持空间一致性，同时确保图块间亲和力相对相似。不同于传统 DDPM 从随机噪声 $x_T$ 生成新图像的做法，我们从原图 $x_0$ 经 $t$ 步得到最小但充分加噪的版本 $x_t$，再用 DDPM 恢复，获得 $\tilde{x}_0$。与会改变空间一致性并影响逐图块亲和力的常规增强（如变换）不同，扩散模型以空间一致方式去噪，同时保留语义内容：输出的模糊图像仍具有与原图相似的图块亲和力。虽然还存在其他空间一致的数据增强技术（如高斯模糊），它们未必保留语义一致性；扩散不同之处在于，它只在共享语义对齐的区域内选择性去噪。该增强保证不同语义区域之间的信息仍然隔离。因此，经扩散的图像 $\tilde{x}_0$ 在保留语义细节时只产生很小的亲和力变化，适合作为亲和力学习候选。

为以扩散增强正则化图块亲和力，原图 $x_0$ 与去噪图像 $\tilde{x}_0$ 都经过 ViT。$x_0$ 经 ViT 后按式(3)生成细化 CAM $A_{\mathrm{ref}}$，而 $\tilde{x}_0$ 经 ViT 得到细化 CAM $\tilde A_{\mathrm{ref}}$。如式(3)，图块亲和力 $A_{\mathrm{aff}}$ 与 $\tilde A_{\mathrm{aff}}$ 分别施加于 $A$ 和 $\tilde A$。因此，PAC 以下式训练这些亲和力，使其对空间与语义一致的数据增强保持鲁棒：

$$
\mathcal{L}_{\mathrm{pac}}=\lvert A_{\mathrm{ref}}-\tilde A_{\mathrm{ref}}\rvert_1. \tag{9}
$$

其中 $\lvert\cdot\rvert_1$ 表示 $L_1$ 损失。PAC 是增强图块亲和力韧性的简单有效方法，而该参数原本难以正则化。所提框架最终损失为：

$$
\mathcal{L}_{\mathrm{total}}=\mathcal{L}_{\mathrm{cls-vit}}+\mathcal{L}_{\mathrm{cls-diff}}+\mathcal{L}_{\mathrm{lfca}}+\lambda\mathcal{L}_{\mathrm{pac}}, \tag{10}
$$

其中 $\lambda$ 是平衡各损失项的超参数。

## 4 实验

### 4.1 实验设置

**数据集。** 我们在 WSSS 中广泛使用的两个多标签数据集 PASCAL VOC 2012[15]和 MS-COCO 2014[34]上评估方法。PASCAL VOC 2012 含 20 个前景类别和 1 个背景类别，包含 train、val、test 三个子集，分别有 1,464、1,449、1,456 张图像。按以往工作[26,59,60,62]，用 SBD 数据集[20]将 VOC 2012 的 train 集扩增至 10,582 张图像。MS-COCO 2014 有 80 个前景类别和 1 个背景类别，含两个子集（82K train 图像和 40K val 图像）。

**评估指标。** 与先前工作[26,59]一致，使用平均交并比（mIoU）评估 CAM 质量和语义分割性能。评估 WSSS 中 CAM 质量时，报告最优阈值下的 mIoU。CAM 质量通常在 train 集评估，语义分割模型性能在 val 集评估。两个数据集 val 集的像素级真值标签都可获得并在本地计算；PASCAL VOC 2012 test 集评估则通过官方网站完成。

**实现细节。** 为公平比较，遵循以往基于 ViT 的 WSSS 方法[18,26,59,60]，使用在 ImageNet[12]上预训练的 DeiT-S 作为分类骨干。分类网络用 Adam 优化器训练 60 个 epoch，初始学习率为 $5\mathrm{e}{-4}$。数据增强与 MCTformer[59]一致，唯不使用颜色抖动。$K_f,V_f$ 的特征 $F_t$ 从基于 U-Net 的 DDPM 中间层获得，$t\in\{0,1,2,3,4\}$。受[61]启发，联合采用正、负预测以有效引导 $\mathcal{L}_{\mathrm{kl}}$。$\mathcal{L}_{\mathrm{kl}}$ 的温度参数 $T$、$\mathcal{L}_{\mathrm{kd}}$ 的平衡参数 $\alpha$ 经实验设为 5 和 0.2。$\mathcal{L}_{\mathrm{lfca}}$ 从第 10 个 epoch 开始施加，该时刻由分类准确率选定。还对去噪图像 $\tilde x_0$ 导出的类别预测施加多标签 soft margin 损失。为平衡损失尺度，$\lambda=10$。为公平比较 CAM 改进方法，生成伪标签时也与既有工作一样使用后处理模型 IRN[1]；也报告不同后处理（PSA[2]）的性能。语义分割模型使用两种常用模型：以 WideResNet38 为骨干的 Deeplab V1，以及以 ResNet 101 为骨干的 Deeplab V2。扩散模型架构细节见补充材料。

### 4.2 消融研究

**组件分析。** 为展示所提框架每个元素的重要性，进行了表 1 的消融研究。本文使用 Min-Max K 池化而非全局平均池化（GAP）作为基线来计算类别预测 $\hat y_p$。与 MCTformer[59]显示 max pooling 显著降低 CAM $A$ 性能的实验结果相反，在相同网络结构下，Min-Max K 池化相较 GAP 带来 2.4% mIoU 增益；该差异可解释为使用了 negativity[62]。关于 Min-Max K 池化的更多讨论见补充材料。利用所提 LFCA 获得的扩散 CAM $A_{\mathrm{diff}}$ 训练 ViT-CAM $A$，相比基线提升 1.7%（表 1-(b)）；表中将此记为 $\mathcal{L}_{\mathrm{cls-diff}}$，因为 LFCA 损失 $\mathcal{L}_{\mathrm{lfca}}$ 仅在 $A_{\mathrm{diff}}$ 被 $\mathcal{L}_{\mathrm{cls-diff}}$ 充分训练后才有效。用 PAC 引导 ViT 学习图块间关系，结果(c)相对基线提升 1.6%。最后同时以 $\mathcal{L}_{\mathrm{lfca}}$ 和 $\mathcal{L}_{\mathrm{pac}}$ 训练，较基线显著提升 3.6%，说明 LFCA 和 PAC 从扩散模型带来的收益独立且不冗余。如图 4，定性上可观察到局部性相对基线改善。尽管不使用 $\mathcal{L}_{\mathrm{kl}}$ 时方法已有高性能，如(d)所示，通过比较(d)和(f)可知，蒸馏 ViT 知识有助于将扩散的语义信息与 ViT 对齐。最后，若用 GAP 计算类别预测，如(e)所示，mIoU 仍有 4.6% 的显著增益。

![](../../../../../../99_Assets%20%28资源文件%29/images/dig_fig4.png)

**图 4：** 细化 CAM 的定性结果。红框表示与具有相同语义内容的区域对应、被均匀激活的区域。

**图片内容解释：** 左侧给出图像和真值掩码，右侧比较基线与所提方法对“餐桌”和“人”的热图；红框突出所提方法补全的同语义区域。

**表 1：** PASCAL VOC 2012 train 集消融研究。$\mathcal{L}^{\dagger}_{\mathrm{cls-vit}}$ 指用 GAP 得到的类别预测与类别标签之间计算的多标签 soft margin 损失。$\mathcal{L}^{-}_{\mathrm{cls-diff}}$ 表示不含 KL 散度损失 $\mathcal{L}_{\mathrm{kl}}$ 的损失。

| 设置 | $\mathcal{L}^{\dagger}_{\mathrm{cls-vit}}$ | $\mathcal{L}_{\mathrm{cls-vit}}$ | $\mathcal{L}^{-}_{\mathrm{cls-diff}}+\mathcal{L}_{\mathrm{lfca}}$ | $\mathcal{L}_{\mathrm{cls-diff}}+\mathcal{L}_{\mathrm{lfca}}$ | $\mathcal{L}_{\mathrm{pac}}$ | mIoU |
|---|---:|---:|---:|---:|---:|---:|
| MCTformer[59] | ✓ |  |  |  |  | 63.3 |
| Baseline |  | ✓ |  |  |  | 65.7 |
| (a) |  | ✓ | ✓ |  |  | 66.8 |
| (b) |  | ✓ |  | ✓ |  | 67.4 |
| (c) |  | ✓ |  |  | ✓ | 67.3 |
| (d) |  | ✓ | ✓ |  | ✓ | 68.2 |
| (e) | ✓ |  |  | ✓ | ✓ | 67.9 |
| (f) |  | ✓ |  | ✓ | ✓ | **69.3** |

**扩散时间步的影响。** 图 5 显示不同步数 $t\in\{30,60,300\}$ 的去噪图像 $\tilde x_0$。PAC 使用 $t=60$，此时 $\tilde x_{0\leftarrow60}$ 在感知上与原图难以区分，却已充分增强以供网络辨别。当 $t=300$ 时，$\tilde x_{0\leftarrow300}$ 与 $x_0$ 明显不同，图块亲和力可能偏离原图过多。实验上，使用 $t<150$ 的去噪图像显示相近 mIoU（$67.3\pm0.3\%$），更高步数则使性能退化；该消融见补充材料。

![](../../../../../../99_Assets%20%28资源文件%29/images/dig_fig5.png)

**图 5：** 不同时间步 $t$ 的加噪图像（下）与扩散去噪图像（上）。红框表示导致亲和力变化的显著变形，蓝框表示具有不同语义信息的区域间清晰边界。

**图片内容解释：** 随 $t$ 增大，去噪图像逐渐模糊、细节丢失；图中用蓝框和红框分别标明语义区域边界仍清晰与产生明显形变的位置。

### 4.3 与最先进方法比较

表 2 报告 CAM（Seed）和用于语义分割的伪真值（Mask）性能，评估在 train 集进行。相较当前最先进方法，所提方法在 seed 和 mask 质量上均显著更优。表 3 显示，用该高质量伪真值训练的语义分割模型，在 PASCAL VOC val、test 集分别达到 73.9% 和 73.7%。考虑到伪标签质量为 73.3%，语义分割模型比标签高 0.6%，说明方法生成的标签为分割模型编码了有意义的信息。BECO[39]认为伪标签与分割模型之间相关性偏低，通常源于假阳性背景或不完整目标；其分析认为伪标签噪声主要发生在语义边界。由此可推知，本方法获得的伪标签在这些语义边界上更准确，相应地模型比基线有 2.3% 更高精度。除 PASCAL VOC 2012 外，表 3 还列出 MS COCO 2014 结果：模型 CAM 在 80K train 集达到 43.0%，施加 IRN[1]得到 46.1% 的伪标签；MS COCO 语义分割模型达到 46.6%，超过最先进方法。关于 CAM 和语义分割预测的更多可视化见补充材料。

![](../../../../../../99_Assets%20%28资源文件%29/images/dig_fig6.png)

**图 6：** (a) VOC 2012 和 (b) COCO 2014 的语义分割可视化结果。从上至下：图像、本文方法、GT。

**图片内容解释：** 图中按列给出多个 VOC 与 COCO 场景，并在每个场景下对比本文预测掩码与真值掩码。

**表 2：** 与最先进 WSSS（多阶段）方法比较。CAM 和 Mask 层面的 mIoU 在 PASCAL VOC 2012 train 集评估；列出每种方法骨干以便公平比较。粗体表示最佳结果。为公平比较，表中也列出各方法使用的后处理（PSA[2]/IRN[1]）。(W)RN 指（宽）ResNet。

| 方法 | 骨干 | Seed | 后处理 | Mask |
|---|---|---:|---|---:|
| OC-CSE[25] ICCV21 | WRN38 | 56.0 | PSA | 66.9 |
| CPN[65] ICCV21 | WRN38 | 57.4 | PSA | 67.8 |
| PPC[14] CVPR22 | WRN38 | 61.5 | IRN | 70.1 |
| ReCAM[10] CVPR22 | RN50 | 54.8 | IRN | 70.5 |
| RIB[27] NeurIPS21 | RN50 | 56.5 | IRN | 70.6 |
| AEFT[62] ECCV22 | WRN38 | 56.0 | PSA | 71.0 |
| ACR[26] CVPR23 | WRN38 | 60.3 | IRN | 72.3 |
| Mat-Label[51] ICCV23 | RN50 | 62.3 | IRN | 72.9 |
| MCTformer[59] CVPR22 | DeiT-S | 61.7 | PSA | 69.1 |
| LPCAM[9] CVPR23 | DeiT-S | 63.5 | PSA | 70.8 |
| FPR[6] ICCV23 | DeiT-S | 63.8 | – | – |
| USAGE[36] ICCV23 | DeiT-S | 67.7 | PSA | 72.8 |
| 本文 | DeiT-S | **69.3** | IRN | 73.3 |
| 本文 | DeiT-S | **69.3** | PSA | **74.3** |

**表 3：** 所提方法与现有 WSSS 工作的 mIoU（%）比较。评估在 PASCAL VOC 2012 和 MS-COCO 2014 上进行；为公平比较，表中仅列使用图像级分类标签的方法。粗体表示最佳结果。

| 方法 | 骨干 | VOC val | VOC test | COCO val |
|---|---|---:|---:|---:|
| SIPE[8] CVPR22 | RN101 | 68.8 | 69.7 | – |
| RIB[27] NeurIPS21 | RN101 | 68.3 | 68.6 | 43.8 |
| FPR[6] ICCV23 | RN101 | 70.3 | 70.1 | 43.9 |
| ReCAM[10] CVPR22 | RN101 | 68.5 | 68.4 | 42.9 |
| USAGE[36] ICCV23 | RN101 | – | – | 44.3 |
| 本文 | RN101 | 71.8 | 72.4 | **46.6** |
| OC-CSE[25] ICCV21 | WRN38 | 68.4 | 68.2 | 36.4 |
| CPN[65] ICCV21 | WRN38 | 67.8 | 68.5 | – |
| SIPE[8] CVPR22 | WRN38 | – | – | 43.6 |
| AEFT[62] ECCV22 | WRN38 | 70.9 | 71.7 | 44.8 |
| ACR[26] CVPR23 | WRN38 | 71.9 | 71.9 | 45.3 |
| OCR[11] CVPR23 | WRN38 | 72.7 | 72.0 | 42.5 |
| MCT[59] CVPR22 | WRN38 | 71.9 | 71.6 | 42.0 |
| LPCAM[9] CVPR23 | WRN38 | 72.6 | 72.4 | 42.8 |
| USAGE[36] ICCV23 | WRN38 | 71.9 | 72.8 | 42.7 |
| 本文 | WRN38 | **73.9** | **73.7** | 45.5 |

## 5 结论

本文受扩散模型捕获高层语义信息能力的启发，设计了一个利用预训练扩散模型处理 ViT 局部性缺失问题的框架。通过成功将扩散特征投影到其对应语义类别，所提局部性融合交叉注意力（LFCA）模块生成扩散 CAM，为 ViT-CAM 提供局部性引导。此外，PAC 中基于扩散的数据增强为原本难以正则化的参数提供一致性，从而改善 CAM 细化。所提模块显示出优越性，并在 VOC 和 COCO 数据集上取得最先进性能。框架的局限与潜在未来工作在于，仍存在设计选择空间。例如，尽管发现 $t\in\{0,1,2,3,4\}$ 合适，仍可能存在更佳时间步组合；研究这些方面将为更好的基于扩散的 WSSS 性能打开新可能。
