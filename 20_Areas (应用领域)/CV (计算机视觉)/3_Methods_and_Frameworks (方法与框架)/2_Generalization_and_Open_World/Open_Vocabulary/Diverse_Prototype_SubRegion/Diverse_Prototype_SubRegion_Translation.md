# Training-free Open-Vocabulary Semantic Segmentation via Diverse Prototype Construction and Sub-region Matching

通过多样化原型构建与子区域匹配实现免训练开放词汇语义分割

## 摘要

开放词汇语义分割（OVSS）旨在对由类别标签指定的任意类别图像进行分割。虽然先前的方法依赖于大量图像-文本对或密集语义标注，但近期的免训练方法试图通过在构建阶段构造语义原型以及在测试期间进行图像到图像匹配（即原型匹配）来克服这些局限。然而，这些方法往往难以有效捕捉类别的视觉特征，并且在原型匹配期间未能利用局部特征。为了解决这些问题，我们提出了一种用于 OVSS 的新型免训练框架，该框架构建多样化原型并执行细粒度子区域匹配。具体而言，我们的方法在构建阶段利用大语言模型（LLMs）通过关于类别不同属性的描述来引导支持图像生成，并采用粗-细聚类来获得多样且稳健的部件级原型。在测试期间，我们提出了一种子区域匹配方法，利用最优传输将部件级原型分配给子区域，从而充分利用部件级原型之间的局部图像特征。大量实验证明了我们方法的有效性，并表明我们的方法达到了最先进的性能，在五个数据集上超越了先前方法。

## 引言

开放词汇语义分割（OVSS）（Ghiasi et al. 2022）将图像中的像素分类到由文本输入指定的一组任意类别中。这项任务具有挑战性，因为它需要从庞大的词汇表中将像素分配到正确的语义标签。为了解决这一具有挑战性的任务，一些工作（Xu et al. 2022b; Liang et al. 2023; Xu et al. 2023b; Han et al. 2023）首先生成类别无关的掩码提议，然后使用视觉-语言模型，即 CLIP（Radford et al. 2021）和 ALIGN（Jia et al. 2021）对提议进行分类。然而，这些方法需要密集语义标注，并面临将视觉-语言模型的图像分类能力迁移到区域分类的问题（Zhou et al. 2023; Liu et al. 2024; Jiao et al. 2023）。另一些工作通过来自图像-文本对的弱监督进行对比学习（Xu et al. 2022a; Luo et al. 2023; Xu et al. 2023a）来克服这一局限。这些方法需要大量图像-文本对，但由于缺乏细粒度监督，仍然难以输出高质量的掩码提议。

近期，一些免训练方式（Karazija et al. 2023; Wang et al. 2024; Barsellotti et al. 2024）遵循两阶段框架取得了有前景的性能，即在构建阶段构造语义原型，在测试阶段进行相似度比较。具体来说，OVDiff 首次提出通过稳定扩散模型（Rombach et al. 2022）构造语义原型，并比较测试图像与所构建原型之间的相似度。RIM（Wang et al. 2024）在该框架中引入了基于排序的匹配过程，而 FreeDA（Barsellotti et al. 2024）提出从大量图像描述中收集原型。

我们认为，要在 OVSS 中取得良好性能有两个关键点：1）用于捕捉视觉特征的多样化原型，以及 2）用于更好利用原型的恰当图像到图像匹配方法。然而，上述方法（Karazija et al. 2023; Wang et al. 2024; Barsellotti et al. 2024）未能满足这两个关键点。它们要么依赖分割数据集的带标注描述，要么使用诸如“A good photo of a cat.”这样的朴素提示来生成图像。前者需要大量人工标注，且所构建的原型无法很好地表示这些描述中未包含的类别。后者导致所生成原型的多样性不足，因而无法概括真实数据的特征，而真实数据通常是多变的。此外，区域的类别仅由与区域特征最相似的整体级原型（即表示生成图像前景区域内全局特征的原型）来决定。这丢失了大量局部信息，也没有利用不同原型之间的信息。具体而言，对于某个待分类的汽车区域（如图 1 所示），该区域的不同子区域与属于汽车的不同支持图像的部件具有很高的相似度（例如，该区域中的车门与某张支持图像中的车门相似，而该区域中的车轮与另一张支持图像中的车轮相似）。然而，由于平均池化，整个区域与支持图像整个前景区域之间的相似度可能并不高。此外，可能有多于一张支持图像与该区域具有高相似度，但只有最高的相似度被使用，其余的被丢弃，从而导致对原型的利用不充分。

![](../../../../../../99_Assets%20(资源文件)/images/Diverse_Prototype_SubRegion/Diverse_Prototype_SubRegion_fig1.png)

**图 1：本工作的动机。**（a）我们在 LLM 的引导下构建多样化原型。（b）我们将不同的部件级原型分配给各个子区域，以实现细粒度匹配和互补信息的利用。

**图片内容解释：** 图 1 包含上下两个子图。子图（a）“Diverse Prototype Construction.”左右对比了两种支持图像生成方式：左侧虚线框内将单一提示“A photo of a car.”与噪声输入 Stable Diffusion，生成的 8 张汽车图像在视角和场景上相近，标注红色哭脸与“Low Diversity”；右侧虚线框内由 LLM 生成“Car parked in front of house.”、“Car abandoned In desert.”、“……”等多条描述来引导（guide）Stable Diffusion，生成的 8 张汽车图像涵盖停靠、车轮特写、废弃锈车、内饰等多种外观与场景，标注绿色笑脸与“High Diversity”。子图（b）“Sub-region Matching”中间展示一张红色汽车测试图像及其待分类区域与多个绿色子区域（风挡、车门区域、车轮）；左侧三张汽车支持图像分别用绿色标出风挡、车门、车轮部件，并以实线箭头连接到测试图的对应子区域，标注相似度 0.52、0.47、0.55，下方绿色方框标注笑脸与“Sub-region Matching P=0.14”；右侧三张支持图像用蓝色标出整车前景区域，以箭头连接到测试车整体的蓝色圆点，分别标注 0.58（实线）、0.41（虚线，右下角带红叉）、0.13（虚线，右下角带红叉），下方蓝色方框标注哭脸与“Image-to-image Matching P=0.05”。

为了解决上述第一个问题，我们用 LLM（Touvron et al. 2023a,b; Meta 2024）生成的关于不同属性的各种描述来扩展朴素提示，从而向图像合成中注入多样性。为了保留合成图像的局部特征，我们将它们映射为部件级原型。进一步地，我们通过粗-细聚类方法进行原型融合，以获得更具代表性的部件级原型，并滤除由一些低质量部件带来的噪声。通过对上一段第二个问题的分析，我们认为将待分类提议与来自所有支持图像的部件进行比较，比仅与单张支持图像的部件进行比较更加合适。因此，我们提出通过最优传输（OT）（Rubner, Tomasi, and Guibas 2000）将所有部件级原型分配给提议的子区域特征以及一个垃圾桶（dustbin）。这一过程会激活来自不同支持图像、且与子区域具有高相似度的部件级原型。我们的方法实现了细粒度相似度比较，并避免了先前方法中存在的原型利用不足问题。综上所述，本文的贡献如下：

- 我们利用 LLM 引导的图像合成与粗-细原型融合方法来构建多样且稳健的原型。
- 我们提出了一种利用 OT 的子区域匹配方法，以实现测试图像与支持图像之间的细粒度相似度比较，并充分利用所有原型之间的信息。
- 实验表明，我们的方法在六个数据集上达到了最先进的性能，且不需要训练和额外数据集。

## 相关工作

### 开放词汇语义分割

SimSeg（Xu et al. 2022b）提出了一种简单的两阶段框架，使用 CLIP 对掩码提议进行开放词汇分类。后续工作（Xu et al. 2023b; Liang et al. 2023; Xu et al. 2023b）尝试对齐提议生成器与 CLIP。然而，这些方法受限于对昂贵密集标注的依赖，在难以获取此类标注的场景下构成了挑战。另一些工作尝试基于大规模图像-文本对进行对比学习（Xu et al. 2022a; Luo et al. 2023; Xu et al. 2023a）。尽管不使用密集标注，这些工作在密集预测任务中往往会遇到空间混淆问题（Yi et al. 2023）。

### 免训练 OVSS

近期一些工作（Karazija et al. 2023; Wang et al. 2024; Barsellotti et al. 2024）通过支持图像合成与相似度比较来解决 OVSS。其中，OVDiff 首次提出通过稳定扩散模型（Rombach et al. 2022）构造语义原型，并比较测试图像与原型之间的相似度。RIM（Wang et al. 2024）在该框架中引入了基于排序的匹配过程，而 FreeDA（Barsellotti et al. 2024）提出从大量描述中收集原型。尽管它们在免训练 OVSS 中取得了有前景的结果，但它们要么仅使用朴素提示，要么需要大量人工标注的描述来进行支持图像合成。此外，它们未能执行图像之间的细粒度匹配，也未能充分利用原型之间的信息。

### 最优传输

最优传输（OT）是一个经典的优化问题，可以形象地理解为以最小代价将一堆土搬运以匹配另一堆土（Rubner, Tomasi, and Guibas 2000）。它被广泛应用于许多计算机视觉任务中。在领域自适应中，（Flamary et al. 2016）提出使用 OT 对齐源分布和目标分布。在局部特征匹配中，（Ni et al. 2023）开发了 PATS，利用 OT 来匹配图像块。语义对应也受益于 OT 建模。（Liu et al. 2020）将语义对应建模为一个 OT 问题，从而在外观变化显著的图像之间实现更稳健的匹配。在全景分割中，（Li et al. 2023）提出了 Point2Mask，将点监督分割表述为一个 OT 问题，从而能够从稀疏点标注生成高质量的分割掩码。据我们所知，我们是首个在开放词汇语义分割领域引入 OT 的工作。

## 方法

### 概览

我们在图 2 的第一行展示了我们方法的框架。在多样化原型构建中，我们使用 LLM（Touvron et al. 2023a,b; Meta 2024）引导扩散模型（Rombach et al. 2022）生成支持图像，同时伴随用于定位前景区域的注意力图（Hertz et al. 2022; Tang et al. 2022）。由特征提取器提取的这些支持图像的特征以及注意力图，经由区域池化和粗-细聚类映射为融合部件级原型（在图 2 中记为“Fused P-Level Prototypes”）。在测试阶段，采用 SAM 和超像素（在图 2 中记为“Supixel”）为测试图像生成掩码提议，并通过相同的特征提取器提取每个提议的子区域特征。随后在融合部件级原型与子区域特征之间进行子区域匹配，以确定每个提议的类别。具体细节如下。

![](../../../../../../99_Assets%20(资源文件)/images/Diverse_Prototype_SubRegion/Diverse_Prototype_SubRegion_fig2.png)

**图 2：** 第一行展示了我们方法的概览。在**多样化原型构建（Diverse Prototype Construction）**中，我们采用 LLM 和粗-细聚类来获得多样且稳健的语义原型。在**子区域匹配（Sub-Region Matching）**中，测试期间在子区域特征与融合部件级原型之间执行相似度比较。第二行展示了区域池化（Region Pooling）与粗-细聚类（Coarse-Fine Clustering）的细节。

**图片内容解释：** 图 2 分为上下两行。第一行左侧蓝色底框为“Diverse Prototype Construction”：LLM 生成关于 plane、truck、TV 等类别的文本描述，与噪声一起送入 Stable Diffusion，产出各类别的支持图像与对应的注意力图（Support Images & Attention maps）；支持图像输入视觉编码器（Visual Encoder）得到特征图，与注意力图一起经过区域池化（Region Pooling）和粗-细聚类（Coarse-Fine Clustering），输出按类别组织的融合部件级原型（Fused P-Level Prototypes）。第一行右侧粉色底框为“Sub-Region Matching”：测试图像（图中以显示器/电视为例）一方面经过 SAM 与超像素（Supixel），另一方面经过视觉编码器，得到子区域原型/特征（Sub-Region Proto），随后与融合部件级原型共同输入最优传输（Optimal Transport）模块，输出类别概率分布（Probability）。第二行左侧绿色虚线框为“Region Pooling”细节：上方分支将注意力图经阈值阶跃符号二值化为前景掩码，并与特征图做掩码平均池化（MAP）得到整体级特征（Whole-Level Feature）；下方分支将支持图像做超像素分割（Supixel），与二值前景掩码结合得到彩色前景部件区域，再与特征图做 MAP 得到部件级原型（Part-Level Prototypes）。第二行右侧浅蓝虚线框为“Coarse-Fine Clustering”细节：先利用支持图像的整体级特征进行聚类，将同一类别的多张支持图像划分为 Coarse Cluster 1、Coarse Cluster 2 至 Coarse Cluster N；再在每个粗聚类内部对其包含的部件级原型做第二次细粒度聚类，输出每组的融合部件级原型（Fused Part-Level Prototypes）。

### 多样化原型构建

**LLM 引导的图像合成。** 为了向稳定扩散模型合成的图像注入多样性，我们借助大语言模型为全部 $C$ 个候选类别，针对以下四个属性中的每一个各生成 $K$ 条不同的描述：状态（state）、特征（feature）、共现物体（co-occurrence object）和位置（location）。这些描述可以表示为 $T_c = \{t_{c,a}^1, t_{c,a}^2, \cdots, t_{c,a}^K \mid a \in (\text{state}, \text{feature}, \text{object}, \text{location}),\ c \in C\}$。例如，在为 car 生成关于“location”属性的描述时，我们用如下方式询问 LLM：“List different locations in which a car may appear in an image. Generate $K$ short descriptions about car in these locations. Each description should start as A photo of car”（列出汽车在图像中可能出现的不同位置。生成 $K$ 条关于处于这些位置的汽车的简短描述。每条描述应以 A photo of car 开头）。我们用每条描述生成一张支持图像，从而得到某一类别的一组支持图像 $S_c = \{S_c^1, S_c^2, \cdots, S_c^N \mid c \in C\},\ N = 4K$。

接下来，我们在支持图像 $S_c$ 上定位类别 $c$ 的前景区域，并将该区域划分为多个部件，以同时构建整体级和部件级原型。具体而言，对于类别 $c$ 的一张支持图像 $S_c^i,\ i \in \{1, \cdots, N\}$，我们通过对交叉注意力图取平均来计算得分图（Hertz et al. 2022; Tang et al. 2022）：

$$
A(S_c^i, c) = \frac{1}{TLH} \sum_{t,l,h} A(S_c^i, c)_{t,l,h},
\tag{1}
$$

其中 $t$、$l$ 和 $h$ 分别索引扩散时间步、去噪层和交叉注意力头。随后我们将该得分图归一化到 $[0, 1]$ 并以 $\gamma$ 为阈值进行二值化，得到用于前景区域的二值掩码 $M_{fg}(S_c^i, c) \in \{0, 1\}^{H \times W}$。最后，$S_c^i$ 的前景区域被映射为整体级特征：

$$
\omega_c^i = MAP(\varphi_1(S_c^i), \zeta_1(M_{fg})),
\tag{2}
$$

其中 $\varphi_1$ 是稳定扩散模型的 UNet，$\zeta_1(\cdot)$ 表示将二值掩码调整到 $\varphi_1$ 所提取特征图尺寸的双线性插值，$MAP$ 表示掩码平均池化（mask average pooling）。进一步地，我们利用超像素算法（Felzenszwalb and Huttenlocher 2004）通过将像素分组为互不重叠的部件来划分 $S_c^i$，这些部件可以表示为 $R_1, R_2, \cdots, R_F \in \{0, 1\}^{H \times W}$，其中 $F$ 是超像素的数量。$S_c^i$ 的这些部件被映射为部件级原型：

$$
p_c^{i,f} = MAP(\varphi_2(S_c^i), \zeta_2(M_{fg} \odot R_f)), \quad f \in F,
\tag{3}
$$

其中 $\varphi_2$ 是 DINOv2 的特征提取器，$\zeta_2(\cdot)$ 表示将二值掩码调整到 DINOv2 特征图尺寸的双线性插值。我们将所有类别的整体级特征和部件级原型分别记为 $\Omega = \{\omega_c^i \mid i = 1, 2, \cdots, N\}_c^C$ 和 $P = \{p_c^{i;1,\cdots,F} \mid i = 1, 2, \cdots, N\}_c^C$。

**粗-细聚类。** 为了使构建的原型更加多样且稳健，我们将原型的规模扩大 $\kappa$ 倍，并使用我们的粗-细聚类方法将这些原型融合回其原始规模。具体来说，扩大规模后共有支持图像 $S = \{S_c^i \mid i = 1, 2, \cdots, \kappa N\}_c^C$、整体级特征 $\Omega = \{\omega_c^i \mid i = 1, 2, \cdots, \kappa N\}_c^C$ 以及部件级原型 $P = \{p_c^{i;1,\cdots,F} \mid i = 1, 2, \cdots, \kappa N\}_c^C$。以类别 $c$ 为例，我们首先使用 K-means 算法根据支持图像对应的整体级特征，将支持图像 $\{S_c^i \mid i = 1, 2, \cdots, \kappa N\}$ 分组为 $N$ 个粗聚类 $\{Clu_c^1, Clu_c^2, \cdots, Clu_c^N\}$。在一个粗聚类（即 $Clu_c^i$）内部，我们再次使用 K-means 算法将部件级原型 $\{p_c^{j;1,\cdots,F} \mid S_c^j \in Clu_c^i\}$ 划分为细聚类，并取各聚类中心作为融合部件级原型 $\hat{p}_c^{i;1,\cdots,F}$，其中 $i$ 表示这些部件级原型属于哪一个粗聚类，$F$ 是粗聚类 $Clu_c^i$ 中融合部件或细聚类的数量。最后，我们将所有融合部件级原型记为 $\hat{P} = \{\hat{p}_c^{i,1,\cdots,F} \mid i = 1, 2, \cdots, N\}_c^C$。

### 子区域匹配

**回顾最优传输。** 假设一组供应者 $\mathcal{S} = \{s_i \mid i = 1, 2, \cdots, m\}$ 需要将货物运输给另一组消费者 $\mathcal{D} = \{d_i \mid i = 1, 2, \cdots, n\}$，其中 $s_i$ 表示供应者 $i$ 的供应单位，$d_j$ 表示需求者 $j$ 的需求量。同时，总供应单位应等于总需求单位，即 $\sum_i s_i = \sum_j d_j$。代价函数 $c_{ij}$ 指定了将一单位货物从供应者 $i$ 运送到消费者 $j$ 的代价。OT（Rachev 1985）的目标是找到一个使总运输代价最小化的传输方案 $X = \{x_{ij} \mid i = 1, 2, \cdots, m,\ j = 1, 2, \cdots, n\}$：

$$
\begin{aligned}
\min_{X} \quad & \sum_{i=1}^{m} \sum_{j=1}^{n} c_{ij} x_{ij} \\
\text{s.t.} \quad & \sum_{j=1}^{n} x_{ij} = s_i, \quad i = 1, \cdots, m \\
& \sum_{i=1}^{m} x_{ij} = d_j, \quad j = 1, \cdots, n \\
& x_{ij} \ge 0, \quad i = 1, \cdots, m,\ j = 1, \cdots, n.
\end{aligned}
\tag{4}
$$

该优化问题可以通过 Sinkhorn 迭代（Cuturi 2013）高效求解。

**用于子区域匹配的最优传输。** 为清晰起见，我们首先解释 $\kappa = 1$ 的情况（此时不执行粗-细聚类）。

对于一张测试图像，我们分别使用 SAM 和超像素算法生成类别无关的掩码提议和子区域。类似于公式 3 中的部件原型构建，我们获得一个掩码提议内各子区域的特征：

$$
q^f = MAP(\varphi_2(I_{test}), \zeta_2(M_p \odot R_f)), \quad f \in F.
\tag{5}
$$

在公式 5 中，$M_p$、$I_{test}$ 和 $F$ 分别表示掩码提议、测试图像和子区域的数量，而其他符号的含义与公式 3 相同。

基于我们之前的分析，我们提出根据 $Q = \{q^f,\ f = 1, \dots, F\}$ 与先前构建的 $P$ 之间的相似度，激活并将一部分部件级原型分配给 $Q$，同时抑制并丢弃其余原型。一旦该匹配过程完成，我们就可以计算不同类别中分配给 $Q$ 的原型数量，并依据该数量决定提议的类别。为了实现这一设计，我们将 $P$ 中的全部部件级原型定义为 $m$ 个供应者，将 $Q$ 中的子区域特征定义为 $n$ 个需求者，并将它们之间的一减去余弦相似度作为代价：

$$
c_{ij} = 1 - \frac{\mathcal{S}_i^T \cdot \mathcal{D}_j}{\|\mathcal{S}_i\|_2 \cdot \|\mathcal{D}_j\|_2}.
\tag{6}
$$

直观上，如果一个供应者（需求者）拥有更多的供应单位，它将在最优传输问题中发挥更重要的作用。因此，如果某个部件级原型（子区域）对应的面积在前景（掩码提议）中所占比例更大，我们就为其分配更多的供应（需求）单位。因为我们认为面积在某种程度上能够反映某一特定部件在整个前景中的重要性。由此，我们推导出第 $i$ 个供应者（对应于 $P$ 中的 $p_c^{n;f}$）的供应单位以及第 $j$ 个需求者（对应于 $Q$ 中的 $q^f$）的需求单位：

$$
s_i = \frac{1}{N} \cdot \frac{SUM(M_{fg} \odot R_f)}{\sum_{f \in F} SUM(M_{fg} \odot R_f)}
\tag{7}
$$

$$
d_j = \frac{SUM(M_p \odot R_f)}{\sum_{f \in F} SUM(M_p \odot R_f)},
\tag{8}
$$

其中 $SUM$ 表示对矩阵中所有元素求和，公式 7 和公式 8 中的 $R_f$ 分别表示支持图像 $S_c^n$ 和测试图像 $I_{test}$ 的超像素。借助这些供应单位、需求单位和代价，我们将优化问题表述如下：

$$
\begin{aligned}
\min_{X} \quad & \sum_{i=1}^{m} \sum_{j=1}^{n} c_{ij} x_{ij} \\
\text{s.t.} \quad & \sum_{j=1}^{n} x_{ij} \le s_i, \quad i = 1, \dots, m \\
& \sum_{i=1}^{m} x_{ij} = d_j, \quad j = 1, \dots, n \\
& x_{ij} \ge 0, \quad i = 1, \dots, m,\ j = 1, \dots, n.
\end{aligned}
\tag{9}
$$

需要注意的是，供应与需求之间存在不平衡（$\sum_i s_i = C \sum_j d_j = C$），这意味着只有一部分供应单位会被运送给需求者。这种不平衡是必要的，因为供应者可以选择只传输具有较小传输代价的货物（即与子区域特征具有高相似度的部件级原型），而不是全部货物。在后一种情况下，我们无法获得任何有用信息，因为所有类别都会选择传输一单位货物。为了求解问题 9，我们添加一个垃圾桶作为第 $(n + 1)$ 个需求者，其需求量为 $(C - 1)$ 单位货物，并将所有供应者到垃圾桶需求者的代价设为零，从而平衡供应与需求，推导出与问题 4 形式相同的最终问题：

$$
\begin{aligned}
\min_{X} \quad & \sum_{i=1}^{m} \sum_{j=1}^{n} c_{ij} x_{ij} + \sum_{i=1}^{m} 0 \cdot x_{i(n+1)} \\
\text{s.t.} \quad & \sum_{j=1}^{n} x_{ij} + x_{i(n+1)} = s_i, \quad i = 1, \cdots, m \\
& \sum_{i=1}^{m} x_{ij} = d_j, \quad j = 1, \cdots, n \\
& \sum_{i=1}^{m} x_{i(n+1)} = C - 1 \\
& x_{ij} \ge 0, \quad i = 1, \cdots, m,\ j = 1, \cdots, n, n + 1.
\end{aligned}
\tag{10}
$$

问题 10 等价于问题 9，是一个标准的最优传输问题，可以使用 Sinkhorn 迭代高效求解。我们在图 3 左侧以包含三个类别的示例来说明问题 10。如图 3 右侧所示，我们从传输方案 $X$ 中获得提议类别的概率分布：

$$
P(c \mid X) = \sum_{i \in \Gamma_c} \sum_{j=1}^{n} x_{ij}
\tag{11}
$$

其中 $\Gamma_c = \{i \mid \mathcal{S}_i \in P_c\}$。

![](../../../../../../99_Assets%20(资源文件)/images/Diverse_Prototype_SubRegion/Diverse_Prototype_SubRegion_fig3.png)

**图 3：子区域匹配示意图。** 为清晰起见，我们仅展示一个包含三个类别的示例。

**图片内容解释：** 图 3 分为左右两部分。左侧圆角框内展示包含 plane、truck、TV 三个类别的二分图匹配模型：底部为 8 个部件级原型供应者及其供应量柱状条 $s_1, s_2, s_3$（plane）、$s_4, s_5$（truck）与 $s_6, s_7, s_8$（TV）；上方为 3 个测试子区域需求者（标号 1.、2.、3.，对应需求量柱状条 $d_1, d_2, d_3$）以及第 4 个黄绿色圆形垃圾桶图标需求者（对应较高的需求量柱状条 $d_4$），线条连接底部供应者与上方需求者或垃圾桶。右侧浅蓝色底框展示对应的 $8 \times 4$ 最优传输矩阵（OT Matrix）：行按 plane（3 行）、truck（2 行）、TV（3 行）分组，列为 1.、2.、3. 和垃圾桶列；格子的灰度深浅代表传输量大小，其中 plane 与 truck 行的深色主要集中在垃圾桶列，TV 行的深色主要集中在 1.、2.、3. 列；图中用蓝色框框住 truck 行在前 3 列的子矩阵并标注红色“sum”箭头指向右侧的类别概率（Probability）柱状图，可见 TV 类别对应的概率条最长。

当情况变为 $\kappa > 1$（即执行了粗-细聚类）时，仅有少数几处变化。部件级原型 $P$ 应替换为融合部件级原型 $\hat{P}$，而第 $i$ 个供应者（对应于 $\hat{p}_c^{i';f}$）的供应单位应替换为如下求和：

$$
\hat{s}_i = \frac{1}{\kappa} \sum_{j \in \Psi_c^{i';f}} s_j,
\tag{12}
$$

其中 $\Psi_c^{i';f}$ 表示用于融合 $\hat{q}_c^{i';f}$ 的部件级原型的索引集合。

## 实验

### 数据集与评估指标

我们在传统语义分割基准的验证集划分上评估我们的方法，即 Pascal VOC 2012（Everingham et al. 2010）、Pascal Context（Mottaghi et al. 2014）、ADE20K（Zhou et al. 2017, 2019）和 COCO Stuff（Caesar, Uijlings, and Ferrari 2018）。我们也在考虑额外“background”（背景）类的情况下验证了我们的方法。对于这种情况，我们在带有额外“background”类的 Pascal Context（记为 Context-60）和 COCO Object（Lin et al. 2014）验证集划分上评估我们的方法。我们使用按类别平均的交并比（mIoU）来衡量性能。

### 实现细节

我们使用 Llama3（Meta 2024）作为我们的 LLM，为每个类别生成多样化描述。在支持图像生成方面，我们采用 Stable Diffusion v2-base（Rombach et al. 2022），扩散步数为 50。我们采用基于 ViT-B（Dosovitskiy et al. 2020）的 DINOv2（Oquab et al. 2023），输入图像尺寸为 $518 \times 518$。这会产生尺寸对应于 $37 \times 37$ 的密集特征。采用基于 ViT-B 的 SAM（Kirillov et al. 2023）作为测试图像的掩码提议生成器。我们以网格方式采集 $32 \times 32$ 个提示点来为测试图像生成掩码提议。我们还使用 Felzenszwalb 算法（Felzenszwalb and Huttenlocher 2004）来提取超像素。为了获得多样化的支持图像，我们将 $K$ 和 $\kappa$ 分别设为 16 和 8。对于前景掩码，我们将阈值设为 $\gamma = 0.4$，这可以滤除大部分背景区域。

### 与最先进方法的比较

我们首先将我们的方法与近期用于免训练开放词汇语义分割的最先进方法进行比较。具体而言，我们包括利用图像-文本相似度的 DiffSeg（Wang et al. 2023）、SCLIP（Wang, Mei, and Yuille 2023）、GEM（Bousselham et al. 2024）、ClearClIP（Lan et al. 2024a）、ProxyCLIP（Lan et al. 2025）和 LaVG（Kang and Cho 2025），以及同样利用任意输入类别来获取一组视觉参考的 ReCo（Shin, Xie, and Albanie 2022）、OVDiff（Karazija et al. 2023）、RIM（Wang et al. 2024）和 FreeDA（Barsellotti et al. 2024）。我们还与 MaskCLIP（Zhou, Loy, and Dai 2022）进行了比较，该方法对 CLIP 架构引入了一些修改以利用其多模态嵌入空间；并与依赖在大规模图像-文本对上进行大量对比训练来学习文本-视觉对齐的 GroupViT（Xu et al. 2022a）和 TCL（Cha, Mun, and Roh 2023）进行了比较。此外，我们列出了 OpenSeg（Ghiasi et al. 2022）、SimSeg（Xu et al. 2022b）和 OvSeg（Liang et al. 2023）的性能作为参考，它们在密集标注分割掩码的监督下训练提议网络并微调 VLM。

我们在表 1 中报告了在四个基准上的结果。如表所示，我们的方案在所有数据集上都取得了最佳结果，以一致的幅度超越了所有竞争者。具体而言，与采用和我们相同设定的 OVDiff 相比，我们取得了平均 5.6 个 mIoU 点的提升。我们也超越了使用额外数据集来进行支持图像合成的 FreeDA。令人惊讶的是，我们甚至取得了与 SimSeg 相当的性能，这表明基于扩散模型生成视觉参考特征的方法具有广阔的前景。

**表 1：与最先进无监督开放词汇语义分割模型的比较。** 标记 $\star$ 指仅用于支持的数据集。对于监督类型，“SM”表示分割掩码（segmentation mask），“IT”表示图像-文本对（image-text pairs），“Cap”表示仅使用描述（captions only）。（注：原表中 OpenSeg、SimSeg、OvSeg 三行数值以浅灰色列示作为参考，Ours 一行加粗并带浅灰底色。）

| Method | Training Dataset | Supervision | Training-free | VOC-20 (mIoU) | Context-59 (mIoU) | Stuff (mIoU) | ADE (mIoU) | Context-60 (mIoU) | Object (mIoU) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| OpenSeg (Ghiasi et al. 2022) | COCO Stuff | SM | ✗ | 60.0 | 36.9 | 15.3 | - | - | - |
| SimSeg (Xu et al. 2022b) | COCO Stuff | SM | ✗ | 88.4 | 47.7 | 20.5 | - | - | - |
| OvSeg (Liang et al. 2023) | COCO Stuff | SM | ✗ | 92.6 | 53.3 | 24.8 | - | - | - |
| ReCo (Shin, Xie, and Albanie 2022) | ImageNet1k$^\star$ | - | ✓ | 57.7 | 22.3 | 16.3 | 11.2 | 19.9 | 15.7 |
| MaskCLIP (Zhou, Loy, and Dai 2022) | LAION | IT | ✗ | 74.9 | 26.4 | 16.4 | 9.8 | 23.6 | 20.6 |
| GroupViT (Xu et al. 2022a) | CC12M | IT | ✗ | 81.5 | 23.8 | 15.4 | 9.2 | 18.7 | 27.9 |
| TCL (Cha, Mun, and Roh 2023) | CC3M+CC12M | IT | ✗ | 83.2 | 33.9 | 22.4 | 17.1 | 30.4 | 30.4 |
| DiffSeg (Wang et al. 2023) | - | - | ✓ | - | - | - | - | 27.5 | 37.9 |
| SCLIP (Wang, Mei, and Yuille 2023) | - | - | ✓ | 80.4 | 33.0 | 22.4 | 14.6 | 30.4 | 30.5 |
| GEM (Bousselham et al. 2024) | - | - | ✓ | - | 34.5 | - | 17.1 | - | - |
| ClearCLIP (Lan et al. 2024a) | - | - | ✓ | 80.9 | 35.9 | 23.9 | 16.7 | 32.6 | 33.0 |
| ProxyCLIP (Lan et al. 2024b) | - | - | ✓ | 80.3 | 39.1 | 26.5 | 20.2 | 35.3 | 37.5 |
| LaVG (Kang and Cho 2024) | - | - | ✓ | 82.5 | 34.7 | 23.2 | 15.8 | 31.6 | 34.2 |
| OVDiff (Karazija et al. 2023) | - | - | ✓ | 81.7 | 33.7 | - | 14.9 | 30.1 | 34.8 |
| RIM (Wang et al. 2024) | - | - | ✓ | 77.8 | 34.3 | - | 17.0 | - | - |
| FreeDA (Barsellotti et al. 2024) | COCO Captions$^\star$ | Cap | ✓ | 85.6 | 43.1 | 27.8 | 22.4 | 38.3 | 37.4 |
| **Ours** | **-** | **-** | **✓** | **86.5** | **43.9** | **28.5** | **24.2** | **39.4** | **38.4** |

### 消融研究

**不同组件的消融研究。** 我们首先在 Context-59 和 ADE 数据集上进行了一系列消融研究，以深入探究所提出方法各个组件的影响。如表 2 所示，第 1 至第 3 行是不含子区域匹配的消融研究，第 4 至第 6 行是包含子区域匹配的消融研究。在第 1 行和第 2 行中，我们沿用先前工作，通过与整体级原型（区别于仅用于引导粗-细聚类的整体级特征）的余弦相似度进行图像匹配。整体级原型的构建方式与整体级特征类似，只是其特征图来自 DINOv2。在第 3 行中，我们使用提议的子区域特征与来自单张支持图像的部件级原型之间余弦相似度的加权组合。结果表明，LLM 描述、粗-细聚类和子区域匹配都是有用的。有趣的是，在第 3 行中，当不使用 LLM 时，子区域匹配算法造成了轻微的性能下降，我们将此归因于部件级原型缺乏多样性。

**表 2：不同组件的消融研究。**

| Proto Construction — LLM | Proto Construction — Clustering | Sub-Region Matching | Context-59 (mIoU) | ADE (mIoU) |
| :---: | :---: | :---: | :---: | :---: |
| ✗ | ✗ | ✗ | 37.46 | 21.87 |
| ✓ | ✗ | ✗ | 40.66 | 22.33 |
| ✓ | ✓ | ✗ | 41.16 | 22.40 |
| ✗ | ✗ | ✓ | 37.30 | 21.22 |
| ✓ | ✗ | ✓ | 42.76 | 23.28 |
| ✓ | ✓ | ✓ | **43.88** | **24.18** |

在图 4 中，我们展示了由朴素提示构建的整体级原型（黄点）与由 LLM 描述构建的整体级原型的 T-SNE 可视化（每个子图代表一个类别）。可以观察到，由 LLM 描述生成的原型在特征空间中覆盖了更广泛的区域。我们分别统计了使用图像到图像匹配和子区域匹配时真实标签类别的预测概率，并在图 5 中展示了它们的分布。在使用图像到图像匹配时，真实类别的预测概率主要集中在 0.01 到 0.13 之间，这说明了模型的混淆。然而，在使用子区域匹配之后，概率扩展到了更大的数值，表明模型在分类中变得更加确定。

![](../../../../../../99_Assets%20(资源文件)/images/Diverse_Prototype_SubRegion/Diverse_Prototype_SubRegion_fig4.png)

**图 4：** 由朴素提示构建的整体级原型（黄点）与由 LLM 描述构建的整体级原型（蓝点）的 T-SNE 可视化。

**图片内容解释：** 图 4 为 $2 \times 3$ 排列的六个类别散点图，标题分别为 food、keyboard、sign、table、tree、truck。在全部六个子图中，黄色圆点（朴素提示构建的原型）相对集中地聚集在局部核心区域，而蓝色圆点（LLM 描述构建的原型）分布范围明显更广，向外围多个方向延伸。

![](../../../../../../99_Assets%20(资源文件)/images/Diverse_Prototype_SubRegion/Diverse_Prototype_SubRegion_fig5.png)

**图 5：** 真实标签类别预测概率的分布。

**图片内容解释：** 图 5 为直方图及拟合虚线曲线，纵轴为“Probability density”（范围 0.00 至 0.14），横轴为“GT probability”（原图中横轴六个刻度标签均印刷为 0.00）。蓝色柱状图与深蓝虚线表示图像到图像匹配得到的 GT 概率分布（GT probability from image-to-image matching），在靠近左侧低概率区呈现高达约 0.14 的尖峰并迅速衰减；绿色柱状图与深绿虚线表示子区域匹配得到的 GT 概率分布（GT probability from sub-region matching），左侧峰值明显降低（约 0.06），并向右方高概率区拖出更长的尾部。

**匹配策略的消融研究。** 我们还在 Context-59 上比较了不同匹配方法的性能。在表 3 的第 1 行中，我们计算掩码提议特征与所有整体级原型之间的余弦相似度。如果最相似的原型属于类别 $c$，我们就将该提议归类为类别 $c$。这是先前工作所采用的简单图像到图像匹配策略（对应于表 2 中的第 2 行）。在该策略基础上，我们将掩码提议与原型之间的余弦相似度替换为提议的子区域特征与单张支持图像的部件级原型之间的 EMD 距离（Rubner, Tomasi, and Guibas 2000）（第 2 行和第 5 行），从而实现更细粒度的匹配。此时，我们进一步发问：将一个提议的不同子区域匹配到来自不同支持图像的原型，是否比仅匹配到来自单张支持图像的原型更好。为了验证这一点，我们计算了提议的子区域特征与每个类别的全部部件级原型之间的 EMD 距离，并将距离最近的类别作为预测类别（第 3 行和第 6 行）。在此基础上，我们提出了最终的子区域匹配方法（第 4 行和第 7 行）。

如表 3 所示，将余弦距离替换为 EMD 距离可以带来一定的性能提升。在将子区域特征与全部部件级原型进行匹配之后，性能进一步提升，证明了互补信息在匹配中非常有用。

**表 3：不同匹配方法的消融研究。**

| Proto Construction — LLM | Proto Construction — Clustering | Matching Method — single | Matching Method — img-to-cls | Matching Method — all | Matching Method — img-to-img | mIoU |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| ✓ | ✗ | | | | ✓ | 40.66 |
| ✓ | ✗ | ✓ | | | | 41.25 |
| ✓ | ✗ | | ✓ | | | 41.72 |
| ✓ | ✗ | | | ✓ | | 42.76 |
| ✓ | ✓ | ✓ | | | | 41.05 |
| ✓ | ✓ | | ✓ | | | 42.56 |
| ✓ | ✓ | | | ✓ | | **43.88** |

**粗-细聚类的消融研究。** 在本节中，我们首先将我们的粗-细聚类与普通聚类（plain clustering）方法进行比较，后者直接使用 K-means 算法融合部件级原型（表 4 中的第 3 行和第 4 行）。结果表明，我们的粗-细聚类优于普通聚类。接下来我们舍弃聚类，在扩大规模后直接使用原型（表 4 中的第 1 行）。可以观察到，在没有聚类的情况下性能出现了显著下降。我们将此归因于当生成的支持图像数量扩大规模时，部件级原型中引入了噪声。这表明在增加多样性的同时，聚类对于保证部件级原型的质量是必要的。

**表 4：粗-细聚类的消融研究。**（注：原表第四列表头印刷为“corse-fine”，最后一行加粗并带浅灰底色。）

| Clustering Method — no clustering | Clustering Method — no scaling up | Clustering Method — plain | Clustering Method — corse-fine | mIoU |
| :---: | :---: | :---: | :---: | :---: |
| ✓ | | | | 42.23 |
| | ✓ | | | 42.76 |
| | | ✓ | | 43.34 |
| | | | ✓ | **43.88** |

## 结论

在这项工作中，我们提出了一种用于解决 OVSS 的免训练框架。我们在 LLM 和粗-细聚类的引导下构建多样化原型。随后，我们在子区域特征与所构建的原型之间执行子区域匹配。实验上，我们在六个数据集上取得了最先进的结果。
