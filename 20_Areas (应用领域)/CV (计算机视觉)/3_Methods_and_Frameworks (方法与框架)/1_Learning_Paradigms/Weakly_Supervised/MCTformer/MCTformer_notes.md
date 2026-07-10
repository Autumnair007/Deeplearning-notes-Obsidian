---
type: concept-note
tags:
  - cv
  - transformer
  - semantic-segmentation
  - weakly-supervised
  - mctformer
  - vit
  - wsss
  - attention
  - cam
status: done
model: Multi-Class Token Transformer
venue: CVPR2022
---
## 1. 核心设计理念与动机
**MCTformer (Multi-class Token Transformer)** 旨在解决弱监督语义分割（WSSS）中，利用Transformer生成高质量伪标签（Pseudo Labels）的问题。
**核心痛点：** 标准的Vision Transformer (ViT) 仅使用**一个**Class Token。这个Token聚合了全图信息，其对应的Attention Map往往是类别无关（Class-agnostic）的，倾向于激活所有前景物体，无法区分具体类别。
**解决方案：** 引入**多个**Class Tokens（每个类别一个）。让每个Class Token专门负责学习特定类别的语义特征，通过Class Token与Patch Tokens之间的Attention交互，直接生成类别判别性（Class-discriminative）的定位图。
**创新点总结：**
1.  **多类别Token结构**：替代单一Token，学习类别特定的交互。
2.  **Patch级亲和度细化**：利用Transformer自带的Patch-to-Patch注意力作为亲和度（Affinity），无需额外的亲和度网络（如AffinityNet）。
3.  **MCTformer-V2互补架构**：结合PatchCAM（基于CNN风格的CAM）与Transformer Attention，利用双流互补性提升性能。

## 2. 模型架构详解 (Architecture)
![](../../../../../../99_Assets%20(资源文件)/images/6a37b2edac3ac16b0ecabc77556392db.png)
### 关注度提取与细化部分详细解释，模型架构解释看paper_notes
#### 1. 核心前提：Token 是怎么排队的？
在 Transformer 里，所有数据都要排成一列长队（Sequence）。
假设我们有：
*   **$C$ 个类别 Token**（比如 20 个，代表猫、狗、车...）
*   **$M$ 个图像 Patch Token**（比如图片切成 $16 \times 16$，就有 $M = 256$ 个方块）。
**它们进入 Transformer 时的排队顺序是固定的：**
$$ \text{Input} = [\underbrace{\text{Class}_1, \dots, \text{Class}_C}_{\text{前 } C \text{ 个位置}}, \underbrace{\text{Patch}_1, \dots, \text{Patch}_M}_{\text{后 } M \text{ 个位置}}] $$
总长度 $L = C + M$。
#### 2. Attention 矩阵长什么样？（全员互动表）

Self-Attention 的本质是 **“每两个 Token 都要打个招呼，算算关系好不好”**。所以生成的 Attention 矩阵 $\mathbf{A}$ 是一个 $L \times L$ 的大方块（$(C+M) \times (C+M)$）。这个矩阵的**每一行**代表“**观察者（Query）**”，**每一列**代表“**被观察对象（Key）**”。

我们可以把这个大矩阵切成 **4 个区域**：

| | **列：前 $C$ 个 (Class)** | **列：后 $M$ 个 (Patch)** |
| :--- | :--- | :--- |
| **行：前 $C$ 个 (Class)** | 1. 类别看类别 (忽略) | **2. 类别看图片 (关键!)** |
| **行：后 $M$ 个 (Patch)** | 3. 图片看类别 (忽略) | **4. 图片看图片 (关键!)** |
#### 3. 为什么要切那两个位置？（对应你的困惑）
现在我们对应你提到的两个部分：
#### **A. Class-specific Attention (橙色部分)**
*   **位置：** 提取**前 $C$ 行**，**后 $M$ 列**。
*   **物理含义：**
    *   **行 (观察者)：** 第 $i$ 行是第 $i$ 个**类别 Token**（比如“猫 Token”）。
    *   **列 (被观察)：** 所有的**图片 Patch**。
    *   **数值含义：** “猫 Token”觉得“这张图片的右上角那个方块”有多重要？
*   **为什么要 Reshape？**
    *   因为列是平铺的 ($1, 2, \dots, 256$)。我们需要把它还原回 $16 \times 16$ 的二维网格，才能看出哪里亮了。
    *   **这就是我们要的“原始定位图”**（猫 Token 盯着哪里，哪里就是猫）。
#### **B. Patch-level Pairwise Affinity (蓝色部分)**
*   **位置：** 提取**后 $M$ 行**，**后 $M$ 列**。
*   **物理含义：**
    *   **行 (观察者)：** 图片上的 Patch A。
    *   **列 (被观察)：** 图片上的 Patch B。
    *   **数值含义：** Patch A 觉得 Patch B 跟我长得像吗？（纹理、颜色、语义相似度）。
*   **这也就是“亲和度矩阵”：** 它天然记录了图像内部纹理的相似性。无需额外计算，Attention 机制自动算好了。
#### 4. 为什么要 Refinement（细化）？
**直观理解：**
*   **原始图 (橙色)** 说：“我觉得 Patch A 是猫（因为有猫耳朵）。” —— **但是它没认出 Patch B（猫肚子），因为它只盯着耳朵看。**
*   **亲和图 (蓝色)** 说：“我不认识猫，但我知道 Patch A（耳朵）和 Patch B（肚子）的毛色纹理是一模一样的，它俩肯定是亲戚。”
**Refinement 操作 (矩阵乘法)：**
$$ \text{Refined Map} = \text{原始图} \times \text{亲和图} $$
**逻辑推演：**
1.  既然 Patch A 是猫（原始图确信）。
2.  既然 Patch A 和 Patch B 长得一样（亲和图确信）。
3.  **推论：** Patch B 肯定也是猫！把 Patch A 的分数传给 Patch B。
### 总结你的疑问

**“凭什么就选这些位置？”**
因为输入序列是硬拼起来的（Class在前，Patch在后）。这是人为定义的物理位置，矩阵的坐标直接对应输入序列的索引。
 
 **“这些位置指的是一个特征图里面的patch的相对位置吗？”**
矩阵的**索引 (Index)** 对应的是 Patch 在展平序列中的 ID（第1个, 第2个... 第256个）。
通过 **Reshape** 操作，第 $k$ 个 Patch 就会回到图片上第 $(row, col)$ 的几何位置。
*   **蓝色区域 (Patch-to-Patch)** 的每一个点 $A_{ij}$，不仅是矩阵里的数值，更是图片上**位置 $i$ 的方块**和**位置 $j$ 的方块**之间的相似度。

***
### 2.1 基础骨干与输入处理
模型基于 **DeiT-S (Data-efficient Image Transformers)** 骨干网。
**输入处理：**
给定输入图像 $I \in \mathbb{R}^{H \times W \times 3}$。
图像被切分为 $N \times N$ 个不重叠的 Patch。每个Patch被展平并线性映射到嵌入维度 $D$。
Patch Tokens 表示为 $\mathbf{T}_{p} \in \mathbb{R}^{M \times D}$，其中 $M = N^2$ 是Patch的总数。

### 2.2 多类别Token设计 (Multi-class Token Structure)
这是MCTformer最关键的改动。
定义 $C$ 个可学习的 Class Tokens，记为 $\mathbf{T}_{c} \in \mathbb{R}^{C \times D}$，其中 $C$ 是数据集的类别数（例如 PASCAL VOC 为 20）。
**输入序列拼接：**
将 Class Tokens 与 Patch Tokens 拼接，并加上位置编码（Position Embeddings, PE）。
输入 Transformer Encoder 的整体序列 $\mathbf{T}_{in}$ 为：
$$\mathbf{T}_{in} = \text{Concat}(\mathbf{T}_{c}, \mathbf{T}_{p}) + \text{PE} \in \mathbb{R}^{(C+M) \times D}$$
这个设计保证了 $C$ 个Class Tokens 能够与所有Patch Tokens进行全局交互。

### 2.3 类别感知训练策略 (Class-aware Training)
为了强迫每个Class Token学习特定类别的特征，必须设计对应的监督信号。
**不同于ViT的做法：** 标准ViT使用MLP头进行分类。
**MCTformer的做法：**
1.  取出最后一层输出的 Class Tokens。
2.  对这些 Tokens 进行**平均池化 (Average Pooling)** 或直接对应类别索引（论文中提到建立一对一映射）。
3.  计算类别分数 $\mathbf{y}(c)$。
4.  使用**多标签软间隔损失 (Multi-label Soft Margin Loss)** 与图像级标签（Image-level Ground Truth）计算损失。
这种一对一的强监督（One-to-one strong connection）确保了第 $c$ 个Token只关注第 $c$ 类的语义区域。

## 3. 核心机制：类别特定注意力学习 (Class-Specific Attention Learning)

### 3.0 CAM相关内容的简要讲解

#### 第一部分：什么是 CAM？（直观理解）

**核心定义：**
CAM 是一种可视化技术，它能告诉我们卷积神经网络 (CNN) 在识别一张图片时，到底在**看哪儿**。

**通俗类比：**
假设你教一个小孩（模型）识别“猫”。
*   你给一张照片，问：“这是什么？”
*   小孩说：“是猫。”
*   CAM 就是你紧接着问的一个问题：“**你是凭什么说是猫的？指给我看。**”
*   小孩（模型）指着图片上的**猫头**和**猫耳朵**说：“因为我有看到这个。”
*   **那个被指出来的区域（热力图），就是 CAM。**

在 WSSS 任务中，我们只有类别标签（知道图里有猫），没有位置标签（不知道猫在哪）。CAM 就像是一个“作弊器”，让我们通过分类任务，**白嫖**到了物体的位置信息。
#### 第二部分：标准 CAM 的工作原理（硬核数学）

标准的 CAM 是由 Bolei Zhou 等人在 CVPR 2016 提出的。它的实现非常依赖于一种特殊的网络结构：**全局平均池化 (GAP)**。
#### 1. 结构要求
传统的 CNN（如 VGG 早期版本）最后通常接的是全连接层 (Fully Connected Layer) 展平数据。
为了做 CAM，必须把网络改成全卷积形式：
*   **最后一次卷积层输出：** 得到 $K$ 张特征图 (Feature Maps)，每张大小为 $H \times W$。记为 $A_k (x,y)$。
*   **GAP 层：** 对每张特征图求平均值，得到 $K$ 个数值。
*   **Softmax 层：** 这 $K$ 个数值通过全连接层权重 $w_k^c$ 映射到类别 $c$ 的分数。
#### 2. 数学公式
对于类别 $c$（比如“猫”），其 CAM 图 $M_c$ 的计算公式为：
$$ M_c(x,y) = \sum_{k} w_k^c \cdot A_k(x,y) $$
**逐项解析：**
*   $A_k(x,y)$：**第 $k$ 张特征图**。这可以理解为“某种纹理检测器”，比如第 5 张图专门检测“毛茸茸”，第 10 张图专门检测“尖耳朵”。
*   $w_k^c$：**权重**。这是模型学出来的。如果识别“猫”，那么“尖耳朵”对应的权重 $w_{10}^{cat}$ 就会很大；而“轮子”对应的权重就会很小。
*   **求和**：把所有特征图叠加起来。如果是重要的特征（权重大的），就在图上把那一块加亮。

**结果：**
最终得到的 $M_c$ 是一个二维热力图，红色的地方代表模型认为那里是属于类别 $c$ 的证据。
#### 第三部分：CAM 的致命弱点（为什么 WSSS 很难做？）

这部分直接关联到你正在读的论文（MCTformer）以及上一条关于“亲和性”的问题。
CAM 的设计初衷是**分类**，不是**分割**。
*   **分类模型很懒：** 只要看到**猫头**，它就敢确信这是猫。它没必要去费劲把猫身子、猫尾巴都找出来。
*   **Discriminative Region (最具判别力的区域)：** CAM 通常只高亮物体最显著的特征（比如头、嘴巴）。
*   **WSSS 的困境：** 我们拿 CAM 当伪标签（Pseudo Label）训练分割模型。如果 CAM 只覆盖了猫头，分割模型学出来的就只是“所有的猫都只有一个头，没有身体”。
**解决方案：**
这就是为什么需要 **Affinity (亲和性)** 或 **MCTformer**。它们的作用就是把 CAM 的高亮区域从“猫头”**扩散**到“全身”。
#### 第四部分：CAM 家族的进化（相关变体）

标准 CAM 有个大缺点：它要求必须改网络结构（必须有 GAP 层）。为了解决这个问题，学术界搞出了一整套 CAM 家族。
#### 1. Grad-CAM (Gradient-weighted CAM) —— 最常用
*   **原理：** 不再依赖最后的全连接层权重 $w$，而是利用**梯度**。
*   **逻辑：** 计算目标类别的分数 $y^c$ 对特征图 $A_k$ 的偏导数（梯度）。如果某张特征图稍微变一点，分类分数就剧烈变化，说明这张图很重要。
*   **优点：** 适用于任何 CNN 结构（ResNet, VGG, DenseNet），**无需修改网络，无需重新训练**。
#### 2. Grad-CAM++
*   **改进：** 针对一张图里有**多个同类物体**（比如图里有三只猫）的情况。Grad-CAM 有时只能定到一只最明显的猫，Grad-CAM++ 通过更复杂的梯度公式，能把三只猫都找出来。
#### 3. Score-CAM
*   **原理：** 抛弃梯度（梯度有时会有噪声）。
*   **逻辑：** 直接把特征图 $A_k$ 当作掩码（Mask）盖在原图上，再次输入模型看分数。如果盖上某张图，分数特别高，说明这张图对应的区域很重要。
*   **优点：** 结果更干净，更平滑，但计算量稍微大一点（因为要多次前向传播）。
#### 4. LayerCAM, X-Grad-CAM, Eigen-CAM...
还有很多变体，但万变不离其宗：都是为了找到**“哪些特征图对分类结果贡献最大”**。
#### 第五部分：Transformer 时代的 CAM（MCTformer 的背景）

你读的论文是关于 Transformer 的。在 ViT (Vision Transformer) 中，概念发生了一点变化。

*   **传统 CAM：** 针对 CNN 的 Feature Maps。
*   **Transformer：** 没有 Feature Maps，只有 **Tokens** 和 **Attention Maps**。

在 ViT 中，虽然没有 GAP 层那种结构，但可以利用 **Self-Attention Map**。
*   `[CLS]` Token 会通过 Attention 机制从所有 Patch Tokens 中聚合信息。
*   **Visualize `[CLS]` Attention:** 直接把 `[CLS]` 对所有 Patch 的注意力权重画出来，天然就是一张热力图。
*   **MCTformer 的贡献：** 以前只有一个 `[CLS]`（只能看个大概），现在搞了 $N$ 个 Class Tokens，每个 Token 专门负责生成对应类别的 Attention Map（即 Transformer 版的 CAM）。

### 总结 WSSS 的标准流水线

理解了 CAM，你就理解了 WSSS 的半壁江山：

1.  **训练分类器：** 用只有 Image-level 标签的数据训练一个 CNN 或 Transformer。
2.  **提取 CAM：** 生成粗糙的物体定位热力图（只包含最具判别力的部分，如猫头）。
3.  **精炼 (Refinement)：** 利用 **Affinity (亲和性)**、CRF 或 MCTformer 中的 Patch Attention，把热力图从猫头扩散到猫全身。
4.  **生成伪标签 (Pseudo Mask)：** 把精炼后的热力图切个阈值，变成 0/1 的黑白掩码。
5.  **全监督训练：** 把这个伪标签当作 Ground Truth，去训练一个标准的语义分割模型（如 DeepLabV3+）。

这篇论文（MCTformer）主要就是在优化第 2 和第 3 步：生成更好的初始图，并利用自带的 Attention 做更好的扩散。

### 3.1 原始注意力提取
Transformer 的 Self-Attention 机制计算如下：
$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{D}})\mathbf{V}$$
其中得到的 Attention Matrix $\mathbf{A}_{t2t} \in \mathbb{R}^{(C+M) \times (C+M)}$包含了所有Token之间的交互。
我们需要从中提取两部分关键信息：
1.  **Class-to-Patch Attention ($\mathbf{A}_{c2p}$)**：表示类别Token关注哪些图像Patch。
    提取位置：矩阵的前 $C$ 行，后 $M$ 列。
    $$\mathbf{A}_{c2p} = \mathbf{A}_{t2t}[1:C, \, C+1:C+M] \in \mathbb{R}^{C \times M}$$
    reshape 后得到 $C \times N \times N$ 的空间注意力图。
2.  **Patch-to-Patch Attention ($\mathbf{A}_{p2p}$)**：表示Patch之间的相似性/亲和度。
    提取位置：矩阵的后 $M$ 行，后 $M$ 列。
    $$\mathbf{A}_{p2p} = \mathbf{A}_{t2t}[C+1:C+M, \, C+1:C+M] \in \mathbb{R}^{M \times M}$$
    这自然构成了一个不需要额外计算的亲和度矩阵。

### 3.2 多层聚合 (Layer Aggregation)
深层网络虽然语义丰富，但可能丢失细节；浅层网络细节多但语义弱。为了平衡精度和召回率，MCTformer 聚合最后 $K$ 层的 Attention Maps。
聚合公式：
$$\hat{\mathbf{A}}_{mct} = \frac{1}{K} \sum_{l=L-K+1}^{L} \hat{\mathbf{A}}_{mct}^{l}$$
其中 $\hat{\mathbf{A}}_{mct}^{l}$ 是第 $l$ 层归一化后的注意力图。实验表明融合最后 **3层** ($K=3$) 效果最佳。

### 3.3 亲和度细化 (Affinity Refinement) - V1 核心
利用提取出的 $\mathbf{A}_{p2p}$ 对 $\mathbf{A}_{c2p}$ 进行随机游走（Random Walk）式的传播细化，使激活区域更完整并贴合物体边界。
细化公式：
$$\mathbf{A}_{mct\_ref}(c, i, j) = \sum_{k}^{N}\sum_{l}^{N} \hat{\mathbf{A}}_{p2p}(i, j, k, l) \cdot \mathbf{A}_{mct}(c, k, l)$$
这里 $(i, j)$ 和 $(k, l)$ 代表Patch的空间坐标。本质上是用 Patch 之间的相似度作为权重，对原始定位图进行加权平滑。

#### 公式解释：

**下标与上标的含义**
*   **$A$**: Attention Map（注意力图/定位图），就是一个存数值的矩阵。
*   **$c$**: **Class（类别）**。比如 $c=1$ 代表“猫”，$c=2$ 代表“狗”。
*   **$mct$**: 指的是 **MCTformer 生成的原始图**（还没被优化的）。这是由你在上一步里提到的那 $C$ 个类别 Token 算出来的图（即 $A_{c2p}$）。
*   **$ref$**: **Refined（优化后的）**。这就是我们想要计算出来的、边缘更清晰的好图。
*   **$p2p$**: **Patch-to-Patch（图块对图块）**。这就是亲和性矩阵，代表图块之间的相似度。
**坐标的含义 (关键！)**
假设我们的特征图大小是 $N \times N$（比如 $14 \times 14$）：
*   **$(i, j)$**: **“目标位置”**。也就是我们现在正要计算、要更新的那个像素格子的坐标。
*   **$(k, l)$**: **“源位置”**。也就是图像上**其他所有**像素格子的坐标。我们要遍历整张图去询问它们。
 
 **公式的物理过程：全图投票**

让我们用 **“猫的定位”** ($c = \text{Cat}$) 来举例。

我们要计算：**坐标 $(i, j)$ 这个格子，它是猫的概率是多少？** ($A_{mct}^{\text{ref}}(c, i, j)$)
公式右边的求和 $\sum_{k} \sum_{l}$ 意味着我们要**遍历图上所有的点 $(k, l)$**，让每个点都来对 $(i, j)$ 投一票。
每一票的权重由两部分组成：
1.  **$A_{mct}(c, k, l)$ —— “源点是猫吗？”**
    *   这是**原始定位图**在 $(k, l)$ 处的值。
    *   如果这个值很大，说明模型（Class Token）非常确信位置 $(k, l)$ 是猫头。
2.  **$\hat{A}_{p2p}(i, j, k, l)$ —— “源点和目标点像吗？”**
    *   这是**亲和性矩阵**。它记录了位置 $(i, j)$ 和位置 $(k, l)$ 在纹理、颜色上有多相似。
    *   如果 $(i, j)$ 是黑色毛发，$(k, l)$ 也是黑色毛发，这个值就很大（接近 1）。
    *   如果 $(i, j)$ 是草地，$(k, l)$ 是黑色毛发，这个值就很小（接近 0）。

**总结**
*   **$A_{mct}$ (原始图)**：像是一个**“近视眼”**，只看清了猫头，猫肚子看成了黑影。
*   **$A_{p2p}$ (亲和性)**：像是一张**“纹理关系网”**，它知道猫头和猫肚子其实是一类材质。
*   **公式的操作**：利用关系网，把“近视眼”看清的那个点（猫头）的信息，**顺着纹理扩散**到没看清的点（猫肚子），从而得到一张清晰完整的图 ($A^{ref}$)。
这个过程不需要额外训练网络，因为 $A_{p2p}$ 也就是 Transformer 里那个巨大的 Attention 矩阵的一部分（Patch 对 Patch 的那部分），是现成的。
***
#### 亲和度的详细解释：
##### 1. 小白解释：就像“油漆桶”工具

想象一下你在用 Windows 画图或者 Photoshop。

**背景：**
WSSS 任务中，一开始生成的伪标签（通常来自 CAM）往往很粗糙。比如一张猫的照片，模型可能只高亮了**猫头**，但是**猫身子**和**猫尾巴**没有被激活。这时候我们知道“猫头是猫”，但不知道身子是不是。

**亲和性（Affinity）是什么？**
亲和性就是用来衡量**两个像素（或两个块）之间“像不像”** 的指标。
*   **高亲和性：** 两个像素颜色一样（都是黑毛）、纹理一样，那它们大概率是同一个物体。
*   **低亲和性：** 一个像素是黑毛，旁边一个是绿草，那它们大概率有一个是边界，不属于同一个物体。

**它怎么工作？**
这就好比 Photoshop 里的**魔棒**或**油漆桶工具**。
1.  **种子点（Seed）：** CAM 告诉你“这里有个点肯定是猫（比如猫头）”。
2.  **扩散（Propagation）：** 算法会问旁边的像素：“嘿，你跟猫头长得像吗（亲和性高吗）？”
    *   如果是（像猫身子），算法就说：“那你也是猫，我要把你涂上颜色。”
    *   如果不是（像背景草地），算法就说：“停，到此为止，你是背景。”

**总结：**
在 WSSS 中，利用亲和性，就是利用像素之间“长得像不像”的关系，把确定的标签（猫头）扩散到不确定的区域（猫身子），从而得到一张完整的猫的分割图。

##### 2. 专业解释：成对关系矩阵与标签传播

在学术定义中，**Affinity（亲和性）** 指的是特征空间中两个单元（Pixel 或 Patch）之间的相似度。

 **(1) 定义**
假设图像上有 $N$ 个 Patch。我们构建一个 $N \times N$ 的矩阵 $A$，其中 $A_{ij}$ 表示第 $i$ 个 Patch 和第 $j$ 个 Patch 的相似程度。
通常的计算方式是基于特征的余弦相似度或欧氏距离，例如：
$$ A_{ij} = \exp\left(-\frac{||f_i - f_j||^2}{\sigma^2}\right) $$
其中 $f$ 是颜色特征（RGB）或深度特征。

##### (2) WSSS 中的痛点：CAM 的局限性
传统的 Class Activation Mapping (CAM) 只能定位到物体**最具判别力**的部分（Discriminative Parts）。
*   比如识别“人”，模型看到“人脸”就足够判断这是人了，它懒得去看手和脚。
*   **结果：** 初始的 Heatmap 只有人脸是红的，其他部分是蓝的。这做不了分割。

##### (3) Affinity 的作用：Refinement（精炼）
为了找回丢失的手和脚，我们引入 Affinity。
*   **假设：** 属于同一个物体的像素，在特征空间中应该是紧密相连的（Affinity 高）。
*   **操作：** 使用 **Random Walk（随机游走）** 或 **CRF（条件随机场）** 等算法。
    *   我们将 CAM 的高响应区域作为“源”。
    *   让能量沿着 Affinity 矩阵流动。
    *   因为人脸（源）和手（未知区域）的纹理、颜色特征相似（Affinity 高），能量会流过去，把手也点亮。
    *   因为人脸和背景特征差异大（Affinity 低），能量流不过去，从而保护了边界。

##### (4) 本文（MCTformer）的创新点在哪里？
通常计算 Affinity 需要：
1.  **基于 RGB 颜色：** 这种太低级，容易受光照影响（比如阴影下的猫毛和阳光下的猫毛颜色不同，会被误判为不相似）。
2.  **基于 CNN 特征：** 需要额外计算。

**MCTformer 说：我这里有“免费”且“高级”的 Affinity！**
*   **Transformer 的 Self-Attention 机制 ($Q \times K^T$) 本身就是在计算相似度！**
*   Attention Map 中的每一个数值，代表了 Patch A 有多关注 Patch B。如果 A 强烈关注 B，说明它们在语义上高度相关。
*   **Patch-to-Patch Attention**：这就是作者提取出来的 $N \times N$ 矩阵。
*   **无需额外计算：** 这是 Transformer 前向传播时必须要算的中间结果，作者直接拿来当 Affinity 矩阵用，去修正那个不完整的 CAM，效果出奇的好。
#### 总结对照
*   **原始 CAM：** 像是一个近视眼画的圈，只圈出了猫头。
*   **亲和性（Affinity）：** 是一张详细的“关系网”，记录了谁和谁是亲戚。
*   **操作：** 拿着猫头的标签，顺着“关系网”把猫身子、猫尾巴都认领回来。
*   **MCTformer 的贡献：** 指出 Transformer 内部自带一张非常高级的“语义关系网”（Attention Map），直接拿来用就能把分割做得很好。

### 3.4 注意力图相关知识讲解
#### 1. 简单回答你的三个核心疑问

*   **Q1: 是每一块（Block）之后出现一个注意力图吗？**
    *   **是的。** Transformer 由 $L$ 个层（Layer/Block）堆叠而成（比如 ViT-Base 有 12 层）。**每一层**内部都有一个 Self-Attention 模块，因此**每一层都会计算出一个全新的注意力图**。
    *   这篇论文为了效果好，只取了**最后 $K$ 层**的图拿来平均，而不是只用最后一层。
*   **Q2: 它的大小一般是咋样的？**
    *   它的原始形状是一个**正方形矩阵**。
    *   大小是：**(所有Token的数量) $\times$ (所有Token的数量)**。
    *   在 MCTformer 中，Token 数量 = $C$（类别 Token）+ $M$（图片 Patch Token）。
    *   所以大小是 $(C+M) \times (C+M)$。
*   **Q3: H、W、C 分别和什么有关？**
    *   这里的 $H$ 和 $W$ 通常指**Patch 的行数和列数**（不是原图像素的高宽）。
    *   这里的 $C$ 在这篇论文里特指**类别数量 (Classes)**，而不是通道数 (Channels)。
    *   详细换算见下文。
#### 2. 详细拆解：从输入图片到注意力矩阵
#### A. 准备工作：切块 (Patching)
假设输入图片大小是 $224 \times 224$ 像素。
ViT 把图片切成 $16 \times 16$ 的小方块。
*   行方向有：$224 / 16 = 14$ 个块。
*   列方向有：$224 / 16 = 14$ 个块。
*   总 Patch 数量 $M = 14 \times 14 = 196$ 个。
#### B. 加入 Class Tokens
MCTformer 说：“我要识别 20 个类别（比如 VOC 数据集）”。
*   它插入了 20 个 Class Tokens。
*   **总 Token 序列长度** = 20 (类别) + 196 (图片) = **216**。
#### C. 计算 Attention 矩阵 ($A_{t2t}$)
Attention 的本质是计算**所有 Token 两两之间的关系**。
公式 $QK^T$ 实际上是做点积。
*   $Q$ (Query) 有 216 个向量。
*   $K$ (Key) 有 216 个向量。
*   结果矩阵的大小 = **$216 \times 216$**。
这个 $216 \times 216$ 的矩阵，就是你问题里的**“全局成对注意力图 ($A_{t2t}$)”**。
*   第 1 行代表：第 1 个类别 Token 对全员（包括自己和图片）的关注度。
*   第 50 行代表：第 30 个图片 Patch 对全员的关注度。
#### 3. 维度变换：如何把矩阵变成“图片”？
你在论文图里看到的热力图（Heatmap），是经过 **Reshape（重塑）** 操作的。这一步最容易让人困惑。
我们只关注**类别 Token 对图片 Patch 的关注**（即论文中的 $A_{c2p}$）。
1.  **切片 (Slicing)：**
    *   从 $216 \times 216$ 的大矩阵里，取出**前 20 行**（对应 20 个类别）。
    *   再取出这 20 行里的**后 196 列**（对应 196 个图片 Patch）。
    *   现在的矩阵大小是：**$20 \times 196$**。
    *   物理含义：20 个类别分别给 196 个小方块打分。
2.  **重塑 (Reshaping)：**
    *   对于第 1 个类别（比如“猫”），它有一个长度为 196 的向量。
    *   这 196 个数值对应图片上的 196 个位置。
    *   我们把这 196 个数，按顺序填回到 $14 \times 14$ 的网格里。
    *   **现在的形状是：$14 \times 14$。这就是一张低分辨率的“猫的定位图”！**
3.  **对所有类别操作：**
    *   对 20 个类别都这么做。
    *   最终得到的数据形状是：**$20 \times 14 \times 14$**。
    *   对应论文里的符号：$C \times N \times N$（论文里的 $N$ 指的是 Patch 的边长数量，即这里的 14）。
### 4. 补充：多头注意力 (Multi-Head) 的影响
还有一个细节论文没展开讲，但你需要知道：
Transformer 通常是 **Multi-Head** 的（比如 12 个头）。
这意味着在每一层，其实计算了 **12 个** $216 \times 216$ 的矩阵。
*   **常规操作：** 为了简化，通常会把这 12 个头的注意力图取**平均值**，压缩成一个图，然后再进行上面的 Reshape 操作。
*   **物理含义：** 不同的头可能关注不同的特征（有的头看纹理，有的头看形状），取平均是为了获得更鲁棒的全局注意力。
### 总结
当你看到 $A_{mct} \in \mathbb{R}^{C \times N \times N}$ 时：
*   **$C$ (20)**：你有 20 个分类任务，所以有 20 张图。
*   **$N \times N$ ($14 \times 14$)**：这是一张低分辨率的特征图，每个点代表原图中 $16 \times 16$ 像素区域的重要性。
*   **来源**：它来自 Transformer 内部那个巨大的 $(C+M) \times (C+M)$ 矩阵的**切片**和**变形**。
这也是为什么 Transformer 做分割很方便：它内部天生就算好了这些“谁关注谁”的图，不需要像 CNN 那样必须等到最后才通过 CAM 算出来。
## 4. 进阶模型：MCTformer-V2 与 PatchCAM 互补

### 4.1 PatchCAM 模块
为了弥补 Transformer 注意力可能存在的过度平滑或局部性不足，V2版本引入了基于 Patch Token 的 CAM 分支。
**流程：**
1.  取出Transformer编码器输出的 Patch Tokens $\mathbf{T}_{out\_pat} \in \mathbb{R}^{M \times D}$。
2.  Reshape 为 2D 特征图 $N \times N \times D$。
3.  通过一个卷积层（$1 \times 1$ Conv）将通道数降为 $C$（类别数）。
4.  得到特征图 $\mathbf{F}_{out\_pat} \in \mathbb{R}^{N \times N \times C}$。
5.  应用全局平均池化 (GAP) 得到分类分数，计算损失 $\mathcal{L}_{cls-patch}$。
6.  训练时，总损失为：$\mathcal{L}_{total} = \mathcal{L}_{cls-class} + \mathcal{L}_{cls-patch}$。

### 4.2 融合策略 (Fusion Strategy)
推理阶段，将 Transformer 的注意力图与 PatchCAM 的激活图进行**元素级乘法 (Element-wise Multiplication)** 融合：
$$\mathbf{A} = \mathbf{A}_{pCAM} \circ \mathbf{A}_{mct}$$
**原理：** Transformer Attention 提供了高质量的全局上下文关联（去除背景噪音能力强），而 PatchCAM 提供了基于局部特征的强激活。两者相乘（交集操作）可以有效抑制背景噪音并强化物体区域。融合后的结果再经过 Patch-to-Patch Affinity 进行细化。

## 5. 数据流与张量维度深度解析 (Data Flow Analysis)
本节以具体的数字为例进行追踪。
**假设设置：**
*   输入图像大小：$224 \times 224$
*   Patch Size：$16 \times 16$
*   类别数 $C$：20 (PASCAL VOC)
*   Embedding Dim $D$：384 (DeiT-S)
*   Patch 数量 $M = (224/16)^2 = 14 \times 14 = 196$
*   Head 数量 $H$：6 (DeiT-S config)

### Step 1: 输入构建
*   **Image**: `[1, 3, 224, 224]`
*   **Patch Projection**: 展平并映射 -> `[1, 196, 384]` (即 $\mathbf{T}_{p}$)
*   **Class Tokens**: 初始化 -> `[1, 20, 384]` (即 $\mathbf{T}_{c}$)
*   **Concat**: 拼接 $\mathbf{T}_{c}$ 和 $\mathbf{T}_{p}$ -> `[1, 216, 384]` (即 $\mathbf{T}_{in}$)
*   **Position Embedding**: 加法操作，维度不变 -> `[1, 216, 384]`

### Step 2: Transformer Encoder 内部 (单层为例)
*   **Input**: $\mathbf{X} \in \mathbb{R}^{216 \times 384}$
*   **Q, K, V 生成**: 通过Linear层投影，并拆分为 $H=6$ 个头。
    *   $Q, K, V$ shape per head: `[1, 6, 216, 64]` (384/6 = 64)
*   **Attention Score 计算**: $\text{MatMul}(Q, K^T)$
    *   计算结果维度：`[1, 6, 216, 216]` (Token-to-Token 矩阵)
    *   这就是 $\mathbf{A}_{t2t}$ (在Head维度平均后)。
*   **矩阵切片 (Slicing)**:
    *   **Class-to-Patch ($\mathbf{A}_{c2p}$)**: 取前20行，后196列 -> `[1, 20, 196]`。
        *   物理含义：20个类别分别对196个Patch的关注度。
        *   Reshape -> `[1, 20, 14, 14]` (空间定位图)。
    *   **Patch-to-Patch ($\mathbf{A}_{p2p}$)**: 取后196行，后196列 -> `[1, 196, 196]`。
        *   物理含义：196个Patch两两之间的相似度。

### Step 3: MCTformer-V1 输出与细化
*   **Aggregation**: 将最后3层的 $\mathbf{A}_{c2p}$ (`[20, 14, 14]`) 平均 -> $\hat{\mathbf{A}}_{mct}$。
*   **Refinement**:
    *   输入 $\hat{\mathbf{A}}_{mct}$: `[20, 196]` (Flattened)
    *   输入 $\hat{\mathbf{A}}_{p2p}$: `[196, 196]`
    *   矩阵乘法: `[20, 196] x [196, 196]^T` -> `[20, 196]`
    *   结果：利用Patch相似性传播后的注意力图。

### Step 4: MCTformer-V2 融合流程
*   **PatchCAM 分支**:
    *   输入：最后一层 Patch Tokens `[1, 196, 384]`
    *   Reshape -> `[1, 384, 14, 14]`
    *   Conv $1\times1$ ($384 \to 20$) -> `[1, 20, 14, 14]` ($\mathbf{F}_{out\_pat}$)
    *   GAP -> `[1, 20]` (用于分类 Loss)
    *   归一化 (Min-Max) -> $\mathbf{A}_{pCAM}$ `[20, 14, 14]`
*   **Fusion**:
    *   $\mathbf{A}_{mct}$ (`[20, 14, 14]`) $\circ$ $\mathbf{A}_{pCAM}$ (`[20, 14, 14]`) -> `[20, 14, 14]`
    *   再经过 Step 3 的 Affinity Refinement。

## 6. 工程实践与实现细节 (Implementation Details)

### 6.1 训练设置
*   **预训练权重**：使用在 ImageNet 上预训练的 DeiT-S (Distilled) 权重。
*   **Class Token 初始化**：用 DeiT-S 原始的一个 Class Token 复制 $C$ 份来初始化 $C$ 个 Class Tokens，保证初始化分布一致。
*   **数据增强**：
    *   Resize: $256 \times 256$
    *   Random Crop: $224 \times 224$
    *   水平翻转 (Horizontal Flip)
*   **优化器**：AdamW
    *   Learning Rate: $5 \times 10^{-4}$ (Backbone), $5 \times 10^{-3}$ (Head)
    *   Weight Decay: $0.01$
*   **Batch Size**: 依据显存调整，通常较大。
*   **迭代次数**: PASCAL VOC 上训练，使得模型收敛。

### 6.2 伪标签生成 (Pseudo Label Generation)
1.  **多尺度测试 (Multi-scale Testing)**：推理时使用不同的图像尺度，聚合结果以获得更鲁棒的定位图。
2.  **阈值分割**：生成的定位图是连续值的热力图，需要设定阈值（背景/前景）生成 Mask。
3.  **CRF 后处理**：虽然Transformer自带了 Affinity Refinement，但在生成最终伪标签 Mask 之前，工程上通常还会接一个标准的 DenseCRF (Conditional Random Field) 来进一步锐化边界。
    *   CRF 参数通常参考 DeepLab 或 PSA (Pixel Semantic Affinity) 的默认设置。

### 6.3 实验中的关键发现 (Ablation Insights)
*   **Pooling 方式的选择**：
    *   Max Pooling (26.8% mIoU): 效果最差，倾向于只激活物体最具判别力的局部（比如猫头），导致覆盖不全。
    *   MLP (41.5% mIoU): 引入额外参数，增加了学习难度，容易过拟合。
    *   **Average Pooling (47.2% mIoU)**: 效果最好。迫使 Token 关注物体所有相关 Patch 以最大化平均得分，从而提升了物体的完整性（Completeness）。
*   **层融合 (Layer Fusion)**：
    *   单独使用最后一层：语义最强，但空间细节可能丢失。
    *   融合所有层：包含太多低级纹理信息，导致背景噪音增加。
    *   融合最后3层：最佳的 Precision/Recall 平衡。
*   **V1 vs V2**:
    *   V1 (Attention Only): mIoU 47.2%
    *   V1 (Attn + Affinity): mIoU 55.2% (证明了Patch-to-Patch Affinity的有效性)
    *   V2 (Attn + PatchCAM): mIoU 58.2%
    *   V2 (Full): mIoU 61.7% (在 PASCAL VOC train set 上的 Seed 质量)。

## 7. 总结与分析
MCTformer 通过“多类别Token”这一简单而高效的改动，从根本上解决了ViT在弱监督定位任务中“身份不明”的问题。
**技术核心逻辑链：**
多Token -> 一对一分类监督 -> 强迫Token学习特定类别Attention -> Attention即定位图 -> Patch间Self-Attention即亲和度 -> 互补PatchCAM提升鲁棒性。
**工程价值：**
该方法无需像传统CNN方法那样训练多个阶段或复杂的擦除网络（Erasing networks），是一个端到端的纯Transformer解决方案，且能无缝集成到现有的CAM框架中，并在 PASCAL VOC 和 MS COCO 上刷新了当时的 SOTA 结果（PASCAL VOC test mIoU 71.6%）。
