---
type: paper-note
tags:
  - cv
  - semantic-segmentation
  - weakly-supervised
  - wsss
  - diffusion-model
  - ddpm
  - vision-transformer
  - cam
  - attention
status: done
model: DiG
venue: ECCV2024
---

论文网址：[Diffusion-Guided Weakly Supervised Semantic Segmentation](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/04703.pdf)

本地PDF文件：[Diffusion-Guided Weakly Supervised Semantic Segmentation](../../../../../../99_Assets%20(资源文件)/papers/Diffusion-Guided%20Weakly%20Supervised%20Semantic%20Segmentation.pdf)

***

# Diffusion-Guided Weakly Supervised Semantic Segmentation

扩散引导的弱监督语义分割

**摘要。** 使用分类标签的弱监督语义分割（WSSS）通常使用类激活图（CAM）基于卷积神经网络（CNN）定位目标。由于感受野有限，基于 CNN 的 CAM 往往无法定位完整目标。视觉 Transformer（ViT）的出现凭借其优越性能缓解了这一问题，但 ViT 缺乏局部性又带来了新的挑战。受去噪扩散概率模型（DDPM）能够捕获高层语义信息这一能力的启发，我们将扩散模型引入 WSSS 来解决该问题。首先，为融合并在语义上对齐 DDPM 与 ViT 之间的信息，我们设计了局部性融合交叉注意力（Locality Fusion Cross Attention，LFCA）模块。LFCA 利用从预训练 DDPM 去噪过程中聚合的特征，生成扩散 CAM（Diffusion-CAM），以向来自 ViT 的 CAM（ViT-CAM）提供局部性信息。其次，通过向原始图像添加噪声并用 DDPM 去噪，我们获得可作为增强样本使用的去噪图像。为了有效引导 ViT 挖掘图块之间的关系，我们在原始图像与去噪图像的输出之间设计了图块亲和力一致性（Patch Affinity Consistency，PAC）。大量消融研究支持所提方法的优越性。我们的方法在 WSSS 中广泛使用的两个数据集 PASCAL VOC 2012 和 MS-COCO 2014 上取得了新的最先进性能。代码见 https://github.com/yoon307/DiG。

**关键词：** 去噪扩散概率模型 · 弱监督语义分割

## 1 引言

为缓解全监督语义分割模型中劳动密集且成本高昂的标注过程，研究者提出使用弱标签训练语义分割。借助图像级标签[25,32,42,59,63–65]、涂鸦[33,50]和边界框[24,28]等低廉且易得的标签，弱监督语义分割（WSSS）方法得到积极探索，目标是达到全监督性能。

![](../../../../../../99_Assets%20%28资源文件%29/images/dig_fig1.png)

**图 1：** (a) 来自预训练扩散模型特征的 K-Means 聚类结果。右图中相同颜色表示位于对应像素的特征属于同一聚类。(b) 所提框架的简化流程。为解决 ViT 的局部性问题，局部性融合交叉注意力（LFCA）模块提供语义上定位良好的扩散 CAM，以改善 ViT-CAM 的局部性。

**图片内容解释：** (a)以火车和摩托车图像显示扩散特征在空间上形成连贯语义区域；(b)显示冻结的扩散编码器经 K-Means 提供局部性信息，ViT 提供语义对齐信息，二者在 LFCA 中融合。

> [!note] 图 1 解读｜DiG 如何把“知道是什么”和“知道在哪里”结合起来？
>
> **(a) K-Means 展示的是什么？** 对预训练 DDPM 的每个空间位置的中间特征做 K-Means；右图中同色块表示这些位置的特征被分到同一簇。可以看到火车、轨道、树木，以及摩托车、人、背景等区域大多形成空间连续的簇。这不是分割真值，而是一个直观证据：扩散特征已将局部外观相近且语义相关的位置聚在特征空间中。图中较粗、锯齿状的边界也提醒我们，这种局部性来自较低分辨率的特征图，不能直接等同于精细的像素级掩码。
>
> **K-Means 不在训练主干中。** 它在图 1(a) 中用于诊断和可视化 DDPM 特征的聚类性；DiG 实际送入 LFCA 的是冻结 DDPM U-Net 的中间特征，而不是 K-Means 的离散簇标签。因此，方法并不依赖预先得到“火车簇”或“人簇”。
>
> **(b) 两路信息各补什么？** ViT 的图块 token 经图像级分类训练，通常较擅长判断图中有哪些类别，故提供“语义对齐”（*what*）；但其 CAM 可能只激活判别性部位，且相邻图块的关系不够稳定。冻结的扩散编码器则提供“语义局部性”（*where locally*）：同一物体或背景区域在特征空间往往更连贯，却没有直接对应 WSSS 类别的标签语义。
>
> **LFCA 如何融合？** 将 ViT 中已带类别语义的 token 作为查询（Query），将扩散特征作为键和值（Key/Value）。交叉注意力让每个 ViT token 从与其最相关的扩散空间位置读取信息，得到同时具有类别指向性和局部连续性的输出 token；据此生成的 diffusion CAM 可视为“局部性更好、但已被类别语义校准”的 CAM。
>
> **为何还需要对齐与 PAC？** 仅有交叉注意力不能保证扩散特征的类别含义与 ViT 一致，所以 DiG 用图像级分类损失和来自 ViT 的 KLD 蒸馏来校准 LFCA。对齐充分后，diffusion CAM 以停止梯度的监督信号约束 ViT-CAM。另一路 PAC 则比较原图与“加噪后再去噪”图像的图块亲和力：若语义和空间结构未变，它们的图块—图块关系也应相近，从而抑制只靠分类损失难以约束的零散、误激活区域。
>
> 可以把整张图概括为：**DDPM 给 ViT 提供局部结构证据，ViT 给 DDPM 特征提供类别语义锚点；LFCA 负责融合，diffusion CAM 与 PAC 共同把这种互补性传回 ViT-CAM。**

尽管使用图像级标签训练语义分割模型是最实用的设置，提取定位信息仍很困难，因为标签只告知某些目标是否存在。为从分类器捕获空间信息并生成伪像素级标签，大多数 WSSS 工作采用基于 CNN 的类激活图（CAM）[67]。然而，基于 CNN 的 CAM 由于其固有特性，不仅聚焦于最具判别性的区域（即激活不足），还具有不精确的目标边界。为使获得的 CAM 能作为语义分割的伪标签，各项研究致力于提高 CAM 的质量。

转而使用视觉 Transformer（ViT）提取 CAM，借助多头自注意力（MHSA）机制获得全局特征，能够有效减轻 CNN-CAM 的激活不足问题并显示出可喜结果。然而，用 ViT 提取 CAM 也带来另一挑战：相较于 CNN，ViT 可能不擅长捕获细粒度局部模式。一些方法通过改进位置编码[54]或采用结合 ViT 与 CNN 的混合模型[19]，缓解 ViT 的这种局部性不足[19,31,48,54]。不过在 WSSS 中，对于解决 ViT 局部性缺失所致 CAM 性能下降的关注相对较少。

> [!note] 我的理解｜本文要补的是 ViT 的哪块短板？
>
> CNN 的局部感很强但容易只盯住最有辨识度的局部；ViT 更容易覆盖物体，却没有足够强的“邻近图块属于同一物体”的偏好。DiG 不把扩散模型当作分割器，而是借它已有的局部语义结构来约束 ViT：LFCA 给 CAM 补局部性，PAC 给图块—图块关系补稳定性。

与此同时，去噪扩散概率模型（DDPM）[21]已在图像生成、修复和超分辨率等多种生成建模任务中展现出良好结果。已有研究表明，DDPM 的特征表示能够捕获对语义分割有价值的高层语义信息[3]。考虑到扩散模型也可以以无标签方式训练，它适合用于 WSSS。基于这一动机，我们提出一个新颖的 WSSS 框架：将语义上聚类良好的 DDPM 特征与 ViT 融合，以提高 CAM 质量；还提出一种自监督学习技术，利用 DDPM 加噪、去噪过程恢复的图像进一步改善 CAM 质量。训练 DDPM 时，我们不使用外部数据集和目标数据集图像之外的额外监督。

> [!note] 文献 [3]｜DDPM 特征为什么能帮助分割？
>
> [3] 是 Baranchuk 等人的 *[Label-Efficient Semantic Segmentation with Diffusion Models](https://arxiv.org/abs/2112.03126)*（ICLR 2022）。该工作冻结预训练 DDPM，考察其反向去噪网络中的中间激活，发现这些特征能有效编码输入图像的语义，并可作为像素级分割表征；即使只有少量带像素标注的训练图像，配合一个简单的分割方法也能取得很强的结果。
>
> 它为这里的论断提供了直接依据：DDPM 的 U-Net 不只是生成噪声预测，还在中间层保留了可区分物体区域的语义信息。与 [3] 不同，DiG 不使用像素级标注来训练分割头，而是把这些特征经 LFCA 对齐到 ViT 的类别语义，再用扩散 CAM 和 PAC 反哺仅有图像级标签的 WSSS。

如图 1 所示，我们以 ViT 为分类骨干、以基于 U-Net 的网络为 DDPM，且只使用图像级监督。尽管 DDPM 如图 1(a)所示能够以较高局部性捕获语义信息，它仍需与 ViT 结构的语义对齐。因此，我们提出 LFCA 模块实现这种对齐，并用它改善 ViT-CAM 的局部性。具体地，LFCA 从 ViT 中类别对齐的图块取查询 token，从扩散特征取键/值 token，并以交叉注意力融合不同特征表示。为引导 LFCA 的输出 token，我们施加分类损失，并以 Kullback-Leibler 散度（KLD）从 ViT 模型蒸馏知识，从而连续地将 LFCA 语义对齐到 ViT。充分对齐后，由 LFCA 输出 token 生成的扩散 CAM 用于训练 ViT-CAM。此外，我们引入 PAC，旨在训练 ViT 鲁棒地学习图块间关系。当图像由 DDPM 加噪并去噪时，图像发生人类难以察觉但显著的退化，同时语义特征得以保留。因局部语义在图像退化期间不变，去噪图像与原图的图块亲和力应当相似。PAC 对原图与退化图像的图块亲和力施加相似性约束，以有效引导仅靠分类损失难以正则化的图块亲和力。

> [!note] 概念补充｜DiG 中的 KL 散度与知识蒸馏
>
> **大白话。** 图像级标签只告诉模型“这张图有火车”，却不告诉它“火车与轨道、树木之间的相对可能性”。ViT 已从分类训练中形成一套较可靠的类别判断；KL 散度要求 LFCA 的扩散分支给出相近的判断。也就是说，ViT 不只传递最终答案，还传递“最像什么、次像什么、分别有多像”的软排序。
>
> **数学上。** 对两个类别概率分布 $P,Q$，KL 散度定义为
> $$
> D_{\mathrm{KL}}(P\Vert Q)=\sum_iP_i\log\frac{P_i}{Q_i}\geq0.
> $$
> 在式 (5) 中，$P=\sigma(\hat y/T)$ 是 ViT 的温度概率分布，$Q=\sigma(\hat y_{\mathrm{diff}}/T)$ 是 LFCA/扩散分支的分布；最小化 $D_{\mathrm{KL}}(P\Vert Q)$ 会尤其惩罚“ViT 很确信、扩散分支却给出很低概率”的类别。它是有方向的：$D_{\mathrm{KL}}(P\Vert Q)$ 一般不等于 $D_{\mathrm{KL}}(Q\Vert P)$，此处的方向表达的是“扩散分支向 ViT 学习”。
>
> **它和分类损失有什么不同？** 图像标签产生的 $\mathcal L_{\mathrm{cls}}$ 是硬监督：类别出现与否应符合标注；KL 是软监督：即使两个分支都预测“火车存在”，也进一步要求它们的整组类别置信度相近。固定 $P$ 时，有 $D_{\mathrm{KL}}(P\Vert Q)=H(P,Q)-H(P)$；由于熵 $H(P)$ 是常数，最小化 KL 等价于以 ViT 的软概率为目标最小化交叉熵。
>
> **温度 $T$ 与 $T^2$。** 先除以 $T>1$ 再做 Softmax 会把概率分布变平，使“第二像什么”等弱信息也能参与蒸馏；乘回 $T^2$ 是标准的梯度尺度补偿，避免升温后 KL 项的梯度过小。故式 (6) 中 $\alpha$ 平衡硬标签分类与软蒸馏，而不是让扩散分支无条件复制 ViT。
>
> 延伸阅读：[[Distillation#3.1 像素级软 logit 蒸馏|知识蒸馏算子库：软 logit 蒸馏]]（含张量维度、温度 Softmax 和实现层面的说明）。

本文主要贡献如下：

- 提出首个成功将扩散模型融合至基于 ViT 的 WSSS 模型的框架；鉴于任务性质，方法也只利用图像级监督。
- 提出 LFCA 模块，在将预训练扩散特征对齐至 ViT 语义类别的同时，挖掘其潜力。
- 设计简单而有效的 PAC：将去噪图像作为空间与语义一致的增强样本，更好地引导用于细化 CAM 的图块亲和力。

PASCAL VOC 2012[15]和 MS COCO[34]的实验结果，通过建立新的最先进性能验证了所提方法的优越性。

## 2 相关工作

### 2.1 扩散模型

扩散模型[21,43]是旨在逼近真实图像分布的概率生成模型。给定图像被连续高斯噪声破坏时，模型学习恢复该图像。DDPM[21]建模前向扩散过程，并提出以简化方式逆转扩散过程的训练目标。为加快图像生成推理，DDIM[44]提出具有非马尔可夫链的隐式概率模型。Dhariwal 等人[13]受 GAN 模型架构取得进展而扩散模型未获同等关注的观察启发，提出面向扩散模型的新架构；凭借强大的模型和修改后的目标，扩散模型首次在样本质量与多样性上超过 GAN。本文遵循 DDPM 过程，采用[13]提出的基于 U-Net 的扩散模型。

> [!note] 从 DDPM 到 DiG：这段话到底在说什么？
> **DDPM 的基本机制。** 从真实图像 $x_0$ 出发，前向过程在 $T$ 个时间步中逐步加入很小的高斯噪声：$q(x_t\mid x_{t-1})=\mathcal N(\sqrt{1-\beta_t}x_{t-1},\beta_tI)$。当 $t$ 足够大时，$x_T$ 近似标准高斯噪声；并且可直接写成 $x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon$，其中 $\epsilon\sim\mathcal N(0,I)$。因此训练时随机抽取 $t$ 和噪声 $\epsilon$，令网络 $\epsilon_\theta(x_t,t)$ 预测该噪声，并最小化 $\lVert\epsilon-\epsilon_\theta(x_t,t)\rVert^2$。DiG 使用的是 DDPM U-Net 的中间特征，而不是让 DDPM 直接输出分割图。[Ho et al., 2020](https://arxiv.org/abs/2006.11239)
>
> **DDIM 为什么更快。** DDIM 保留 DDPM 的训练目标，却构造一族共享边缘分布的非马尔可夫过程；采样时可跳过大量时间步，在 $\eta=0$ 时反向轨迹还是确定性的。DiG 不使用 DDIM 来生成分割结果。[Song et al., 2020](https://arxiv.org/abs/2010.02502)
>
> **[13] 的贡献及其与 DiG 的关系。** Dhariwal 与 Nichol 通过系统消融改进扩散 U-Net，并以 classifier guidance 改进条件生成。DiG 采用这一类 DDPM 式 U-Net 的中间特征作为局部结构来源。[Dhariwal & Nichol, 2021](https://arxiv.org/abs/2105.05233)

### 2.2 基于生成模型的图像分割

与 WSSS 旨在减轻语义分割标注负担的目标相似，已有大量工作利用生成模型做语义分割。DDPM 出现前，多数研究专注于利用 GAN 表示进行语义分割。Tritrong 等人[49]提出利用 GAN 表示的一次学习语义部件分割框架。Xu 等人[58]和 Galeev 等人[17]表明 GAN 以简单方式编码图像语义，施加线性变换或构建轻量解码器即可将 GAN 特征投影为语义分割图。随着在图像合成上超越 GAN 的扩散模型出现[13]，许多研究[3,4,47,53]致力于将扩散模型用于语义分割。Baranchuk 等人[3]表明扩散模型能够用于语义分割，同时捕获对其他任务有价值的高层语义；Rahman 等人[38,53]提出基于扩散模型的医学图像分割框架。

### 2.3 弱监督语义分割

**CAM 改进。** 大多数使用图像级标签的 WSSS 方法用 CAM[67]定位图像中的目标。然而，未经细化的 CAM 无法发现较不具判别性的区域，并且目标边界不精确。为改善 CAM，研究提出：利用跨图像语义关系[16,30,45]、注意力机制[37,52,55]、将图像拆分为互补图块[65]、局部—全局视图图像间的一致性[22]、引入额外类别[5,60]，以及重建器与分类器间的对抗学习[26]。除这些方法外，还引入了基于“擦除并寻找”机制的对抗擦除（AE）方法[25,29,46,62,66]。除扩展 CAM 外，基于对比学习的方法[8,57,68]引导 CAM 获得更准确目标边界。随着 ViT 显示出有希望的定位能力，许多 WSSS 工作[18,35,40–42,59,60]采用 ViT 而非 CNN 作为定位骨干。为提取注意力图，MCTformer[59]将单一类别 token 扩展为多个类别 token，并提出在 ViT 内以类别特定方式提取和细化 CAM 的框架。Xu 等人[60]和 Lin 等人[35]引入视觉—语言（VL）模型以丰富 ViT 的类别表示能力；但这些方法使用的语言模型在大型外部数据集上训练、引入额外语言监督，故不宜与仅图像级监督方法比较。尽管已有许多基于 ViT 的 WSSS 方法，据我们所知，尚无研究处理 ViT 的局部性缺失。

**CAM 细化。** 为进一步细化 CAM 质量以作为语义分割标签，[1,2]利用邻近像素间语义亲和力，[7]使用边界信息。MARS[23]利用无监督语义分割模型的特征生成去偏伪标签。Mat-Label[51]将图像抠图带入 WSSS 伪标签生成过程。BECO[39]提出通过标签混合改善语义边界处伪标签的协同训练框架。由于这些后处理技术依赖 CAM 质量并可与 CAM 改进方法整合，本研究重点放在 **CAM 改进阶段**，而非 CAM 细化阶段。

> [!note] 我的理解｜DiG和普通CAM后处理的区别
>
> IRN、PSA一类方法拿到CAM后再沿像素关系或边界传播，本质上是在**已有种子上做后处理**；DiG的LFCA和PAC则直接进入分类网络的训练过程，试图让ViT一开始就产生局部性更好的CAM。两者并不冲突：DiG先改善种子，实验中仍可把IRN或PSA接在后面生成伪标签。换句话说，DiG解决“分类器给出的证据不够局部、亲和力不够可靠”，后处理解决“如何把已有证据进一步铺成稠密掩码”。

## 3 方法

### 3.1 预备知识

**去噪扩散概率模型。** 本文采用扩散模型[21,43]为提取 CAM 的分类器提供结构良好的高层语义信息。DDPM 包括前向过程 $q(x_t|x_{t-1})$：在 $T$ 步中逐步将输入数据 $x_0$ 破坏为高斯噪声 $x_T\sim\mathcal{N}(0,1)$；以及反向（即去噪）过程 $p_\theta(x_{t-1}|x_t)$：预测前向扩散过程中的噪声，获得更干净的样本 $x_t$。

Ho 等人[21]经验发现，网络直接预测噪声 $\epsilon_\theta(x_t,t)$ 而非预测均值 $\mu_\theta(x_t,t)$，能生成更多高频细节，故本文遵循这一先前工作。Ho 等人[21]的简化训练目标为：

$$
\mathcal{L}_{\mathrm{simple}}=\mathbb{E}_{t,x_0,\epsilon}[\lVert\epsilon-\epsilon_\theta(x_t,t)\rVert^2]. \tag{1}
$$

因此，在扩散模型训练过程中，除图像本身外不使用任何标签。关于 DDPM 的额外数学细节见补充材料。

> [!note] 我的理解｜公式1：DDPM在本文中先学会“从噪声还原结构”
>
> 公式整体做的是噪声回归：先随机取一张训练图像 $x_0$、一个时间步 $t$ 和一份高斯噪声 $\epsilon$，按前向扩散规则得到 $x_t$；网络看到 $x_t$ 与 $t$ 后预测其中的噪声 $\epsilon_\theta(x_t,t)$，再用均方误差逼近真实噪声。预测准确以后，就能从较脏的 $x_t$ 逐步恢复出更干净的图像，也会在U-Net中间层形成有关物体结构与区域一致性的特征。
>
> 这里的DDPM训练目标是通用背景，并不是DiG新提出的损失。DiG的新设计在于如何复用两类产物：一类是不同轻微噪声时间步的**中间扩散特征**，交给LFCA；另一类是从原图加噪后恢复出的**去噪图像**，交给PAC。因为式(1)只需要知道人为加入的噪声，不需要类别或像素标签，所以扩散模型仍符合图像级弱监督设置。不过它需要在目标数据集图像上预训练，这部分计算成本不能理解为“免费”。

**用于 WSSS 的视觉 Transformer。** 为把输入图像 $x_0$ 输入 ViT，先将图像分为 $N\times N$ 个图块，继而嵌入为图块 token $T_p\in\mathbb{R}^{N^2\times D}$，其中 $D$ 为嵌入维度。随后，将 $C$ 个类别 token $T_c\in\mathbb{R}^{C\times D}$ 与图块 token 拼接并加入位置嵌入。拼接 token 输入 $T_{\mathrm{in}}\in\mathbb{R}^{(C+N^2)\times D}$ 按[59]送入 ViT；输出 token $T_{\mathrm{out}}\in\mathbb{R}^{(C+N^2)\times D}$ 被拆分为类别 token $T^c_{\mathrm{out}}\in\mathbb{R}^{C\times D}$ 与图块 token $T^p_{\mathrm{out}}\in\mathbb{R}^{N^2\times D}$。对 $T^c_{\mathrm{out}}$ 池化得到类别预测 $\hat y_c$。将图块 token $T^p_{\mathrm{out}}$ 重塑并施加具有 $C$ 个通道的卷积层，可得到 CAM $A\in\mathbb{R}^{N\times N\times C}$ 和 ViT 的类别预测 $\hat y_p$。其中：

$$
\hat y_p=\operatorname{TopK}(\operatorname{ReLU}(\dot A),K)-\operatorname{TopK}(\operatorname{ReLU}(-\dot A),K). \tag{2}
$$

其中，$\dot A$ 和 $K$ 分别为生成 $A$ 前、施加 ReLU 前的图块级特征以及选择数量。

> [!note] 我的理解｜公式2：Min-Max K池化同时收集正证据和负证据
>
> 先看目的：CAM $A$ 是空间图，但分类损失需要每个类别一个分数，式(2)负责把整张响应图压成类别预测 $\hat y_p$。第一项取该类别最强的 $K$ 个正响应并求平均，表示“哪些图块支持该类存在”；第二项对 $-\dot A$ 做同样操作，等价于寻找最强的负响应，再从正证据中减掉它。这样分类分数不只由最显著的一个图块决定，也显式利用了反对该类别的区域。
>
> 例如某类别的四个响应为 $[3,2,-1,-4]$，若 $K=1$，正分支取 $3$，负分支在 $[{-3},{-2},1,4]$ 中取 $4$，最后得到 $3-4=-1$；强负证据会压低类别置信度。若整张图只有少量高正响应而其余区域强烈反对，该类别不会仅靠一个峰值轻易通过。作者把MCTformer的GAP替换成这一操作，消融中仅更换池化就把CAM mIoU从63.3提高到65.7；这是基线改动，不是扩散模块带来的增益。$\operatorname{TopK}(\cdot)$ 是 Top-K 池化操作：沿每个通道的空间维选择并平均 $K$ 个最高值。虽然基线[59]用全局平均池化计算类别预测，本文用式(2)修改池化过程。分类损失 $\mathcal{L}_{\mathrm{cls-vit}}=\mathcal{L}_{\mathrm{cls}}(\hat y_c,y)+\mathcal{L}_{\mathrm{cls}}(\hat y_p,y)$ 用于监督 ViT 的类别预测 $\hat y_c,\hat y_p$；其中 $\mathcal{L}_{\mathrm{cls}}(\hat y,y)$ 表示预测 $\hat y$ 与标签 $y$ 之间的多标签 soft margin 损失。

> [!note] $C$ 个类别 token 从哪里来？它们不是 CLIP 文本 token
>
> MCTformer 将普通 ViT 的单个可学习 $[\mathrm{CLS}]$ token 扩展为 $C$ 个随机初始化的视觉 token。第 $c$ 个 token 通过图像级标签的第 $c$ 维监督获得类别身份；$C$ 是数据集预设类别数，不会按图像动态产生。类别 token 只进入 ViT，DDPM 只接收加噪图像。

此外，可从 ViT 提取 token 对 token 注意力图 $A_{t2t}\in\mathbb{R}^{(C+N^2)\times(C+N^2)}$。由此获得类别到图块注意力 $A_{c2p}\in\mathbb{R}^{C\times N^2}$ 及图块到图块注意力 $A_{p2p}\in\mathbb{R}^{N^2\times N^2}$。为聚合全局—局部信息，融合 ViT 的 $L$ 层类别到图块注意力，得到类别特定注意力图 $A_{\mathrm{att}}=\sum_{l=1}^{L}A_{c2p}^{l}$；将其转置、重塑为 $\mathbb{R}^{N\times N\times C}$。类似地，融合并重塑 $L$ 层图块到图块注意力，得到 $A_{\mathrm{aff}}\in\mathbb{R}^{N\times N\times N\times N}$，表示图块 token 间的亲和力：$A_{\mathrm{aff}}=\sum_{l=1}^{L}A_{p2p}^{l}$。最后，利用 $A_{\mathrm{att}}$ 与 $A_{\mathrm{aff}}$ 细化 CAM $A$，生成 $A_{\mathrm{ref}}\in\mathbb{R}^{N\times N\times C}$：

$$
A_{\mathrm{ref}}(i,j,c)=\sum_m^N\sum_n^N A_{\mathrm{aff}}(i,j,m,n)\cdot(A\odot A_{\mathrm{att}})(m,n,c), \tag{3}
$$

其中 $\odot$ 表示逐元素乘法，$\cdot$ 表示乘法。

> [!note] 我的理解｜CAM 细化前已经有两种关系
>
> **先看式(3)做什么：**它先用类别注意力筛选原始CAM，再让每个目标图块从与自己相似的所有源图块汇总类别证据，得到细化CAM。
>
> $A_{\mathrm{att}}\in\mathbb R^{N\times N\times C}$ 是“类别 token 觉得哪些图块像该类”的类别证据；$A_{\mathrm{aff}}\in\mathbb R^{N\times N\times N\times N}$ 是“一个图块应从哪些图块借信息”的图块关系。$(A\odot A_{\mathrm{att}})(m,n,c)$ 先要求位置 $(m,n)$ 同时有CAM响应和类别注意力支持；随后对固定目标位置 $(i,j)$，用 $A_{\mathrm{aff}}(i,j,m,n)$ 加权汇总所有源位置 $(m,n)$ 的第 $c$ 类证据。
>
> 可以用一个小例子理解：若“狗头”图块对dog响应很高，而同属狗身体的“狗腿”图块响应低，但二者亲和力高，那么狗头证据会沿 $A_{\mathrm{aff}}$ 传播到狗腿。反过来，如果草地图块与狗头亲和力也被错误估高，dog响应也会传播到草地，产生假阳性。这正是作者说图块亲和力“有用但难以约束”的原因，也是PAC要解决的问题。
>
> 注意PAC没有给 $A_{\mathrm{aff}}$ 人工真值，而是要求原图与语义保持的去噪图经过这套传播后得到一致结果。它是对亲和力的间接监督，不保证每一个亲和力元素都具有明确类别含义。

### 3.2 概述

![](../../../../../../99_Assets%20%28资源文件%29/images/dig_fig2.png)

**图 2：** 所提框架的可视化。图像 $x_0$ 与 $C$ 个类别 token 一同输入 ViT 和在不同时间步 $t$ 下冻结的 DDPM。键（$K_f$）和值（$V_f$）由聚合扩散特征构成；查询（$Q$）从 ViT 后部层的图块 token 中提取。$Q,K_f,V_f$ 被用于 LFCA，生成语义对齐特征，进而产生扩散 CAM $A_f$。同时，由 ViT 输出图块 token 创建 CAM $A$。在训练若干 epoch、语义对齐 LFCA 模块后，在 $A$ 与 $A_f$ 间施加 $L_1$ 损失，以传播基于扩散的语义局部性。为简洁起见，图中省略分类损失。

**图片内容解释：** 上方是原图和去噪图共同经过共享 ViT、以 PAC 约束细化 CAM；下方是多时间步噪声图经锁定的扩散 U-Net 生成聚合特征，LFCA 输出扩散 CAM 并以停止梯度方式监督 ViT-CAM。

如图 2 所示，我们提出两种将 DDPM 能力传播至 CAM 的方法。由于使用预训练 DDPM 而不微调，先以式(1)目标训练基于 U-Net 的 DDPM。为将 DDPM 语义和局部性均良好聚类的特征对齐到语义类别，我们设计图 3 所示 LFCA。另将原图 $x_0$ 迭代加噪、再用 DDPM 去噪回原始步骤，得到去噪图像 $\tilde{x}_0$ 作为增强样本；据此提出 PAC，检查干净图像 $x_0$ 与去噪图像 $\tilde{x}_0$ 的图块 token 间亲和力是否一致。

> [!note] 我的理解｜整体流程
>
> **输入与监督。** 输入是图像 $x_0$、其图像级多标签 $y$，以及已经预训练并冻结的DDPM。没有人工像素标注；最终分割网络使用的像素监督来自分类网络生成并经IRN/PSA处理的伪标签。
>
> **按前向顺序看。** ①原图经ViT得到类别预测、原始CAM $A$、类别到图块注意力 $A_{\mathrm{att}}$ 和图块亲和力 $A_{\mathrm{aff}}$，式(3)产生细化CAM $A_{\mathrm{ref}}$；②同一原图被加到多个很浅的时间步 $t\in\{0,1,2,3,4\}$，分别送进冻结扩散U-Net，聚合其中间特征；③LFCA以ViT token为查询、扩散特征为键和值，产生带类别语义的扩散CAM $A_{\mathrm{diff}}$；④LFCA先用图像标签和ViT预测做语义对齐，第10个epoch之后才让停止梯度的 $A_{\mathrm{diff}}$ 监督ViT-CAM；⑤另一条支路把原图加噪到 $t=60$ 再去噪得到 $\tilde x_0$，共享ViT分别处理 $x_0,\tilde x_0$，PAC要求两者细化CAM一致；⑥训练完分类网络后，其CAM经后处理生成伪标签，再按标准多阶段WSSS流程训练分割网络。
>
> **两条扩散支路不要混在一起。** LFCA读取的是若干轻微噪声图像在U-Net中的**特征**，目的是给CAM提供局部性教师；PAC读取的是一次加噪—去噪后得到的**图像**，目的是构造语义和空间近似不变的增强样本。前者解决“哪里应当连成一片”，后者解决“这种图块关系对轻微退化是否稳定”。
>
> **训练与推理。** 图2描述的是CAM生成网络的训练阶段。扩散U-Net冻结，LFCA和ViT会学习；生成伪标签后，语义分割模型另行训练。最终语义分割推理运行的是分割模型，不需要再次执行DDPM、LFCA或PAC。具体分割网络与伪标签生成的实现细节留到代码分析时再展开。

> [!note] 我的理解｜模块按“可学习、冻结、固定操作”怎么分？
>
> **冻结模型：**预训练DDPM在DiG训练时锁定，只提供中间特征和去噪结果。**可学习模块：**ViT分类骨干继续更新；聚合扩散特征的卷积/LayerNorm、生成 $K_f,V_f$ 的投影、LFCA及产生扩散CAM的卷积也需要学习。后续Deeplab分割模型在伪标签阶段之后单独训练。**无新增参数的固定操作：**图块划分、注意力求和、Top-K选择、式(3)亲和力传播和各损失的数值运算。**停止梯度边界：**式(8)明确不向 $A_{\mathrm{diff}}$ 反传，但这不等于LFCA完全冻结；LFCA仍由 $\mathcal L_{\mathrm{cls-diff}}$ 学习。去噪图像是DDPM产生的数据，不会被PAC反向修改。

### 3.3 局部性融合交叉注意力

在 LFCA 中，我们聚合将不同时间步 $t$ 的噪声图像 $x_t$ 送入扩散编码器（瓶颈特征）得到的扩散特征 $F_t\in\mathbb{R}^{H_f\times W_f\times D_f}$，其中 $H_f,W_f$ 是扩散特征的高和宽，$D_f$ 是特征维度。不同噪声图像的扩散特征被拼接，再通过一系列卷积和层归一化操作降维，获得聚合扩散特征 $F_{\mathrm{diff}}\in\mathbb{R}^{H_f\times W_f\times D_f}$。尽管 $F_{\mathrm{diff}}$ 能向 ViT 提供有意义的局部性信息，它尚未与 ViT 在语义上对齐。为有效将扩散特征语义与 ViT 特征语义对齐，我们以交叉注意力融合 ViT 层的查询 token 与聚合扩散特征。为计算交叉注意力，先从 ViT 层提取查询 $Q\in\mathbb{R}^{(C+N^2)\times D}$，提供类别对齐语义信息。为从扩散特征提取局部对齐信息，先将 $F_{\mathrm{diff}}$ 重塑为 $T_f\in\mathbb{R}^{N_f^2\times D_f}$，其中 $N_f^2=H_f\times W_f$；从 $F_{\mathrm{diff}}$ 提取键 $K_f\in\mathbb{R}^{N_f^2\times D}$ 和值 $V_f\in\mathbb{R}^{N_f^2\times D_f}$。随后按交叉注意力机制融合特征：

$$
\operatorname{LFCA}(Q,K_f,V_f)=\operatorname{softmax}\left(\frac{QK_f^{\top}}{\sqrt D}\right)V_f. \tag{4}
$$

![](../../../../../../99_Assets%20%28资源文件%29/images/dig_fig3.png)

**图 3：** 局部性融合交叉注意力模块的可视化。该模块使用来自 ViT 层的查询 $Q$ token，通过交叉注意力对扩散特征进行语义对齐。对类别 token 和图块 token 的类别预测都计算 KD 损失 $\mathcal{L}_{\mathrm{kd}}$。扩散 CAM $A_{\mathrm{diff}}$ 用于引导 ViT-CAM，以提供局部性信息。

**图片内容解释：** 多个扩散编码器中间特征先汇聚为 $F_{\mathrm{diff}}$ 并投影为键和值；ViT token 提供查询。交叉注意力输出同时产生类别/图块预测与扩散 CAM，后者经残差与卷积得到。

> [!note] 图 3 全流程｜从冻结 DDPM 特征到 $\mathcal L_{\mathrm{lfca}}$（按箭头顺序）
>
> **1. 取多时刻的局部扩散特征。** 对同一图像的浅层加噪版本 $x_0,\ldots,x_4$，冻结的 Diffusion Encoder 分别输出 $F_0,\ldots,F_4\in\mathbb R^{H_f\times W_f\times D_f}$。它们先拼接、再经卷积和归一化，得到聚合特征 $F_{\mathrm{diff}}$。此时每个位置有较强的局部结构信息，但它还不知道应对应 VOC/COCO 的哪个类别。
>
> **2. 将扩散特征准备成可被查询的“局部记忆”。** 把 $F_{\mathrm{diff}}$ 展平成 $N_f^2=H_fW_f$ 个位置 token。两条独立的 $1\times1$ 投影分别生成键 $K_f$ 与值 $V_f$：$K_f$ 用于和 ViT 查询计算“该读哪个扩散位置”，$V_f$ 保存“读到该位置后返回什么局部内容”。图左侧两个 $1\times1$ 方块正是这两件事。
>
> **3. ViT 提出问题，LFCA 从扩散特征中检索答案。** ViT 的查询 $Q$ 包含 $C$ 个类别 token 和 $N^2$ 个图块 token。LFCA 先算 $QK_f^\top$，每一行都得到“这个 ViT token 对所有扩散位置的相关度”；softmax 后作为权重，再对 $V_f$ 加权求和。输出 $T_{\mathrm{lfca}}\in\mathbb R^{(C+N^2)\times D_f}$ 的关键含义是：**token 的类别语义来自 ViT，token 读取的局部证据来自 DDPM。**
>
> **4. 右侧第一分叉：类别 token 做整图语义校准。** $T_{\mathrm{lfca}}$ 顶部的 $C$ 个 token 被拆为 $T_{c-\mathrm{diff}}$，沿特征维池化得到整图类别预测 $\hat y_{c-\mathrm{diff}}$。它与 ViT 的类别预测 $\hat y_c$、以及图像级标签 $y$ 一起进入上方第一个 $\mathcal L_{\mathrm{kd}}$：标签保证“图中确实有哪些类”，KL 蒸馏保证扩散分支的置信度结构与 ViT 一致。
>
> **5. 右侧第二分叉：图块 token 有两件并行的事。** 剩下的 $N^2$ 个 token 是 $T_{p-\mathrm{diff}}$。其图块级类别响应先汇总为 $\hat y_{p-\mathrm{diff}}$，再与 ViT 的图块汇总预测 $\hat y_p$、图像标签 $y$ 进入第二个 $\mathcal L_{\mathrm{kd}}$。两路 KD 不是重复计算：前者保证全局类别 token 会认类，后者迫使**空间图块**也携带可用于 CAM 的类别信息；与此同时，同一批图块 token 还会走下方的空间恢复路径。
>
> **6. 与分类汇总并行，图块 token 还要变回二维特征图。** $T_{p-\mathrm{diff}}$ 被重排为 $N\times N\times D_f$，再插值到扩散特征的空间大小，得到 $F_{p-\mathrm{diff}}\in\mathbb R^{H_f\times W_f\times D_f}$。图中的 `Interp.` 只是在对齐空间分辨率，并不产生新的语义。
>
> **7. 残差融合后生成 diffusion CAM。** 将 $F_{p-\mathrm{diff}}$ 与原始 $F_{\mathrm{diff}}$ 相加（图中的 $\oplus$），再经卷积得到 $A_{\mathrm{diff}}\in\mathbb R^{H_f\times W_f\times C}$。残差的作用是同时保留：LFCA 已对齐的类别信息，以及 DDPM 原有的局部细节；每个空间位置的 $C$ 个通道就是该位置属于各类的响应。图最右上方的 $\hat y_{p-\mathrm{diff}}$ 是该图块/CAM 路径汇总出的分类预测，故也接收第二个 KD 损失。
>
> **8. 最终才用 $A_{\mathrm{diff}}$ 教 ViT-CAM。** 两个 $\mathcal L_{\mathrm{kd}}$ 先让 LFCA 学会“这些局部特征在类别上是什么意思”。对齐稳定后，才将 $A_{\mathrm{diff}}$ 调整到与 ViT-CAM $A$ 相同的大小，并以 $\mathcal L_{\mathrm{lfca}}=\lVert A-A_{\mathrm{diff}}\rVert_1$ 监督 $A$。这条损失对 $A_{\mathrm{diff}}$ 停止梯度：它更新 ViT-CAM，不反过来把未对齐的 ViT 噪声写回扩散分支。
>
> **一句话串起来：**多时刻 DDPM 特征 $\rightarrow F_{\mathrm{diff}}\rightarrow(K_f,V_f)$，ViT token $\rightarrow Q$，交叉注意力 $\rightarrow T_{c-\mathrm{diff}}/T_{p-\mathrm{diff}}$；前者走整图 KD，后者一边走图块 KD、一边经插值、残差和卷积生成 $A_{\mathrm{diff}}$；待两路 KD 完成语义校准后，再由 $A_{\mathrm{diff}}$ 通过 $\mathcal L_{\mathrm{lfca}}$ 改善 ViT-CAM。

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

> [!note] 公式5–7：为什么先让 ViT 教 LFCA“认类别”？
>
> ViT 给出 $(\hat y_c,\hat y_p,A)$，LFCA 从 DDPM 特征得到 $(\hat y_{c-\mathrm{diff}},\hat y_{p-\mathrm{diff}},A_{\mathrm{diff}})$。DDPM 特征有局部结构却没有类别坐标，因此 LFCA 在此是学生、ViT 是教师。硬标签告诉 LFCA 有哪些类，KL 要求它复现 ViT 的相对置信度；类别 token 与图块汇总预测都要对齐，才使 $A_{\mathrm{diff}}$ 的类别通道可用。

LFCA 的最终目的是将 DDPM 的局部性信息提供给 ViT。然而在训练早期 LFCA 尚未语义对齐，因此仅在经过设定数量的 epoch 后，使用下式以扩散 CAM $A_{\mathrm{diff}}$ 监督 ViT-CAM $A$：

$$
\mathcal{L}_{\mathrm{lfca}}=\lvert A-A_{\mathrm{diff}}\rvert_1. \tag{8}
$$

其中 $\lvert\cdot\rvert_1$ 表示 $L_1$ 损失。在式(8)中，$\mathcal{L}_{\mathrm{lfca}}$ 不会向 $A_{\mathrm{diff}}$ 反向传播，因为扩散 CAM 仅用作监督；此处通过插值重设 $A_{\mathrm{diff}}$ 的大小以匹配 $A$。

> [!note] 公式8：校准过的 LFCA 如何反过来教 ViT？
>
> 前 10 个 epoch 不开启 $\mathcal L_{\mathrm{lfca}}$；LFCA 先通过式(5)–(7)完成类别对齐。之后 $A_{\mathrm{diff}}$ 作为停止梯度的 target，以 $L_1$ 损失更新 ViT-CAM；它不接收这项损失的梯度，LFCA 仍通过 $\mathcal L_{\mathrm{cls-diff}}$ 学习，DDPM 始终冻结。

### 3.4 图块亲和力一致性

尽管图块亲和力 $A_{\mathrm{aff}}$ 可用作细化而显著提高 ViT-CAM $A$ 质量，但图块亲和力自身由于没有空间约束、只受分类损失监督，容易提高假阳性激活。为改善图块亲和力，我们将扩散模型用作数据增强形式：它保持空间一致性，同时确保图块间亲和力相对相似。不同于传统 DDPM 从随机噪声 $x_T$ 生成新图像的做法，我们从原图 $x_0$ 经 $t$ 步得到最小但充分加噪的版本 $x_t$，再用 DDPM 恢复，获得 $\tilde{x}_0$。与会改变空间一致性并影响逐图块亲和力的常规增强（如变换）不同，扩散模型以空间一致方式去噪，同时保留语义内容：输出的模糊图像仍具有与原图相似的图块亲和力。虽然还存在其他空间一致的数据增强技术（如高斯模糊），它们未必保留语义一致性；扩散不同之处在于，它只在共享语义对齐的区域内选择性去噪。该增强保证不同语义区域之间的信息仍然隔离。因此，经扩散的图像 $\tilde{x}_0$ 在保留语义细节时只产生很小的亲和力变化，适合作为亲和力学习候选。

> [!note] PAC 流程｜图 2 上方绿色区域到底在做什么？
>
> 这里没有再使用 LFCA 的 $F_{\mathrm{diff}}$、$K_f$、$V_f$ 或 $A_{\mathrm{diff}}$；那些属于图 2 下方的橙色 LFCA 路径。PAC 只使用图 2 上方的两条共享 ViT 路径：
>
> 1. 原图 $x_0$ 经过 ViT，得到普通 CAM $A$，以及类别到图块注意力 $A_{\mathrm{att}}$ 和图块到图块亲和力 $A_{\mathrm{aff}}$；按式(3)将它们组合为细化 CAM $A_{\mathrm{ref}}$。
> 2. 同一张图经“加噪 $\rightarrow$ 冻结 DDPM 去噪”得到 $\tilde{x}_0$。这不是从纯噪声生成的新图，而是原图的去噪版本；论文实验使用 $t=60$。
> 3. $\tilde{x}_0$ 经**同一个** ViT（图中 Shared Weights），同样得到 $\tilde A,\tilde A_{\mathrm{att}},\tilde A_{\mathrm{aff}}$，再按式(3)得到 $\tilde A_{\mathrm{ref}}$。
> 4. PAC 只要求两份细化 CAM 接近。于是模型若想减小该损失，就要让两张外观略有变化但语义相同的图具有稳定的 CAM 与图块亲和力。
>
> 图中的圆圈叉号就是式(3)的简写：它把 CAM 和 $A_{\mathrm{att}},A_{\mathrm{aff}}$ 组合为 $A_{\mathrm{ref}}$。亲和力矩阵本身是图块—图块的 $N^2\times N^2$ 关系，因此图中没有把它画成一张“局部特征图”，而是以回接箭头标为 $A_{\mathrm{aff}}$。

为以扩散增强正则化图块亲和力，原图 $x_0$ 与去噪图像 $\tilde{x}_0$ 都经过 ViT。$x_0$ 经 ViT 后按式(3)生成细化 CAM $A_{\mathrm{ref}}$，而 $\tilde{x}_0$ 经 ViT 得到细化 CAM $\tilde A_{\mathrm{ref}}$。如式(3)，图块亲和力 $A_{\mathrm{aff}}$ 与 $\tilde A_{\mathrm{aff}}$ 分别施加于 $A$ 和 $\tilde A$。因此，PAC 以下式训练这些亲和力，使其对空间与语义一致的数据增强保持鲁棒：

$$
\mathcal{L}_{\mathrm{pac}}=\lvert A_{\mathrm{ref}}-\tilde A_{\mathrm{ref}}\rvert_1. \tag{9}
$$

其中 $\lvert\cdot\rvert_1$ 表示 $L_1$ 损失。PAC 是增强图块亲和力韧性的简单有效方法，而该参数原本难以正则化。所提框架最终损失为：

$$
\mathcal{L}_{\mathrm{total}}=\mathcal{L}_{\mathrm{cls-vit}}+\mathcal{L}_{\mathrm{cls-diff}}+\mathcal{L}_{\mathrm{lfca}}+\lambda\mathcal{L}_{\mathrm{pac}}, \tag{10}
$$

其中 $\lambda$ 是平衡各损失项的超参数。

> [!note] 式(9) 与式(10)｜PAC 在总训练中负责什么？
>
> 式(9)比较两份细化 CAM，而非直接比较 $A_{\mathrm{aff}}$；由于细化 CAM 经式(3)产生，损失会间接约束 ViT 的亲和力。PAC 更新共享 ViT，不更新冻结 DDPM。

## 4 实验结果阅读

> [!note] 表 2–3 解读｜骨干网络、后处理和分数应如何分开看？
>
> **先区分两个阶段。** 表 2 的 `Seed` 衡量分类网络直接产出的初始 CAM，`Mask` 衡量 CAM 经后处理扩展、再生成伪标签后的质量；这里的 `DeiT-S` 是 DiG 用来产生 CAM 的**分类/定位骨干**。表 3 则报告用伪标签另行训练的最终语义分割模型；其中 `RN101` 和 `WRN38` 是该分割模型（论文中分别配合 DeepLab V2、DeepLab V1）的**分割骨干**。因此，不能把表 2 的 DeiT-S 与表 3 的 RN101/WRN38 当作同一个网络的不同大小版本，也不应跨骨干把 mIoU 的微小差异完全归因于方法本身。
>
> **DeiT-S（Data-efficient Image Transformer, Small）。** 它是较小的 ViT：将图像划成 patch、通过多头自注意力让远距离 patch 直接交互，并采用适合较少数据预训练的 DeiT 训练策略。它的全局建模有利于 CAM 覆盖同一物体的远距离部分，却天然缺少 CNN 那样的邻域归纳偏置；DiG 的 DDPM 特征、LFCA 与 PAC 正是在补这一局部性短板。这里的 DeiT-S 不等于最终部署时输出像素标签的 DeepLab。
>
> **RN101 与 WRN38。** RN101（ResNet-101）是 101 层残差 CNN，依靠残差连接训练较深网络；WRN38（Wide ResNet-38）层数较少但通道更宽，用较大的每层容量换取表达能力。二者都是成熟的密集预测特征提取器，但容量、预训练、DeepLab 版本和训练配方均可能影响结果。故表 3 最稳妥的读法是：优先在**同一骨干组内**比较 DiG 与其他方法；DiG 在 WRN38 组的 VOC val/test 最好，在 RN101 组的 COCO val 最好。
>
> **PSA 做什么？** PSA（Pixel Semantic Affinity）从初始 CAM 的高置信种子出发，学习像素对是否语义相近，再将类别响应沿高亲和力路径传播。它的目标是补全 CAM 没有激活的同类区域，同时尽量不跨越语义边界；输出是更稠密的伪分割标签，而非改变 DiG 的 DDPM 或 ViT 参数。
>
> **IRN 做什么？** IRN（Inter-pixel Relation Network）学习像素间关系与边界感知信息，用这些关系把 CAM 的可靠激活向同一区域扩散、并在物体边缘处抑制传播，进而生成伪标签。它同样位于 DiG 分类网络训练完成之后，属于多阶段 WSSS 的伪标签细化步骤。可继续参阅 [[多阶段弱监督语义分割详细流程#Q6: IRN（Inter-pixel Relation Network）在某些方法中的作用？|IRN 说明]]。
>
> **为何同一个 Seed 会有两个 Mask？** DiG 的 Seed 都是 69.3，但 IRN 后为 73.3、PSA 后为 74.3；这一 1.0 mIoU 的差异说明后处理本身会显著影响最终伪标签质量。因而表 2 的公平比较应同时看 `Seed`（DiG 是否真的改善了 CAM）和**相同后处理下**的 `Mask`，而非只比较最终 Mask。图 6 则提供定性补充：除了看大区域有没有找全，还应观察预测是否越过真实边界、是否漏掉低判别性的物体部分。
