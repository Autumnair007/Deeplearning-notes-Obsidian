---
type: concept-note
tags:
  - cv
  - dense-prediction
  - monocular-depth-estimation
  - semantic-segmentation
  - vit
  - transformer
  - dpt
  - encoder-decoder
  - self-attention
status: done
model: Dense Prediction Transformer
year: 2021
---
参考资料：[[2107.14467\] DPT: Deformable Patch-based Transformer for Visual Recognition](https://arxiv.org/abs/2107.14467)

***
### 概述

DPT，全称为 Dense Prediction Transformer，是由 René Ranftl 等人在2021年提出的一个深度学习模型。它的核心贡献在于**首次成功地将纯粹的 Vision Transformer (ViT) 架构用作骨干网络，并高效地解决了密集预测（Dense Prediction）任务**。
- **密集预测任务是什么？** 这类任务要求模型对输入图像的**每一个像素**都生成一个或多个预测值。典型的例子包括：
	- **单目深度估计 (Monocular Depth Estimation):** 预测每个像素离相机的距离。
	- **语义分割 (Semantic Segmentation):** 为每个像素分配一个类别标签（如“人”、“车”、“天空”）。
- **DPT的意义何在？** 在DPT出现之前，密集预测任务的主流模型大多基于全卷积网络（FCNs），例如U-Net。这些模型通过卷积操作来提取局部特征，虽然有效，但其感受野（receptive field）相对有限，难以捕捉图像的全局上下文信息。而Vision Transformer (ViT) 通过其自注意力机制（Self-Attention）能够有效捕捉全局信息，但在设计上是为图像分类任务服务的，输出的是一个单一的类别标签，无法直接用于像素级的密集预测。DPT巧妙地将ViT强大的全局特征提取能力与卷积网络的空间重建能力结合起来，解决了这一难题。

### DPT 模型架构详解

![](../../../../99_Assets%20(资源文件)/images/31645224d0f32b5312872b68df133e59.png)

#### 整体架构（左图）

架构概览。输入图像被转换为标记（橙色），其方法有两种：一是通过提取不重叠的图像块，然后对其展平的表示进行线性投影（DPT-Base和DPT-Large）；二是通过应用一个ResNet-50特征提取器（DPT-Hybrid）。图像嵌入被一个位置嵌入和一个与图像块无关的读取标记（红色）所增强。这些标记通过多个Transformer阶段。我们将不同阶段的标记重新组合成多种分辨率的类图像表示。融合模块（紫色）逐步融合和上采样这些表示，以生成一个细粒度的预测。

模型的输入是一张图片，然后通过两种不同的方式将其转换为“tokens”（橙色小块）：

1. **DPT-Base 和 DPT-Large**：这两种变体通过提取图像中不重叠的图像块（patches），然后对这些块进行线性投影，将其转换为 tokens。
2. **DPT-Hybrid**：这种变体则使用一个 ResNet-50 特征提取器来生成 tokens。

生成的 tokens 就像是图像的压缩表示。为了让模型知道每个 token 在图像中的位置，模型会给它们添加一个位置编码（positional embedding）。此外，还会加入一个特殊的、与图像块无关的“readout” token（红色小块），这个 token 负责汇总所有信息，以便进行最终的预测。

这些 tokens 随后被送入多个 **Transformer** 编码器层进行处理。在处理过程中，模型会从不同阶段的 Transformer 输出中提取 tokens，并通过 **Reassemble** 操作将它们重组成图像的表示（image-like representation），不过分辨率会逐渐降低（从 1/32 到 1/4）。

接下来，这些不同分辨率的图像表示会进入 **Fusion** 模块。Fusion 模块会逐步融合和上采样这些表示，最终生成一个高分辨率的深度预测图（即右下角的黑白图像）。

#### Reassemble 操作（中图）

Reassemble操作的概览。标记被组装成特征图，其空间分辨率是输入图像的$\frac{1}{s}$。

**Reassemble** 操作是模型中的一个关键步骤，其目的是将 Transformer 处理后的 tokens 重新组织成一个具有空间结构（类似图像）的特征图。

如图所示，输入的 tokens 是一组一维的数据。Reassemble 模块首先将这些 tokens 按照它们在原始图像中的位置进行排列，形成一个二维的特征图。这个特征图的空间分辨率是原始图像的 $\frac{1}{s}$，其中 s 是一个上采样系数。

具体来说，Reassemble 模块包含以下步骤：

- **Read**：从 Transformer 的输出中读取 tokens。
- **Resample**：对 tokens 进行重采样，将它们排列成一个网格。
- **Project**：将重排后的 tokens 投影到新的特征空间，得到一个具有 $\frac{1}{s}$ 空间分辨率的特征图。

#### Fusion 模块（右图）

 融合块使用残差卷积单元组合特征并上采样特征图。

**Fusion** 模块的作用是整合来自不同分辨率的特征图，并将其上采样以生成最终的深度预测。

如图所示，Fusion 模块主要由以下部分组成：

- **Residual Conv Unit**：残差卷积单元，它包含一系列卷积层和残差连接，用于提取和整合特征。使用残差连接可以帮助模型更好地学习深层特征，并避免梯度消失问题。
- **Resample**：该操作将输入的特征图上采样，使其分辨率翻倍（例如，从 1/8 变为 1/4）。
- **Project**：将上采样后的特征图投影到正确的特征维度。

Fusion 模块会从最低分辨率（例如 1/32）的特征图开始，逐步与其他更高分辨率的特征图（例如 1/16, 1/8, 1/4）进行融合，同时不断进行上采样，最终生成与输入图像大小相同的精细深度预测图。

总结来说，这个模型通过 **Transformer** 强大的全局上下文理解能力，结合 **Reassemble** 操作将一维 tokens 重构为空间特征，再通过 **Fusion** 模块对多尺度特征进行融合和上采样，最终实现精确的深度估计。

### DPT模型结构示意图

#### 1. 编码器 (Encoder)：Vision Transformer (ViT)

DPT直接采用了Vision Transformer (ViT) 作为其编码器部分，用于从输入图像中提取特征。从高层次来看，ViT对图像的“词袋（bag-of-words）”表示进行操作。在这种范式中，图像块（image patches）被单独嵌入到特征空间，扮演着“词语”的角色。在DPT中，这些嵌入的“词语”被称为**tokens**。这个过程可以分解为以下几步：

![](../../../../99_Assets%20(资源文件)/images/7ea3464f418ab83c8190441ae0815220%201.png)

**a. 图像分块与嵌入 (Image Patching and Embedding)**

* 首先，模型会将输入的2D图像 $x \in \mathbb{R}^{H \times W \times C}$ (H是高, W是宽, C是通道数) 分割成一系列固定大小、非重叠的正方形图像块（patches）。在所有实验中，图像块的大小 $p=16$，即每个块为 $16 \times 16$ 像素。然后，每个图像块被展平（flatten）成一个一维向量，并通过一个可学习的线性投射层（Linear Projection）将其单独嵌入，映射到一个固定维度的嵌入向量（embedding）。这个过程产生了我们所说的“tokens”。

* ViT的另一種更具样本效率的变体，即 **ViT-Hybrid**，采用了不同的嵌入策略：它首先通过一个卷积网络（如ResNet50）来提取图像的深层特征图，然后将特征图的像素特征作为tokens。

无论采用哪种方式，为了保留空间位置信息（因为Transformer本身是集合到集合的函数，不感知位置），图像嵌入会与一个可学习的位置嵌入（Positional Embedding）进行拼接。

最后，遵循自然语言处理（NLP）中的工作，ViT还额外添加了一个不基于输入图像的特殊token，我们称之为**readout token**。它参与后续所有的计算，并作为最终的全局图像表示，用于分类等下游任务。

经过这一步，一张完整的2D图像就被转换成了一个1D的token序列。这个过程可以用下面的公式更精确地表示：

$$
z_0 = [t_0^0; E(p_1); E(p_2); \dots; E(p_N)] + E_{pos}
$$

让我们来解释这个公式和相关术语：

-   $p_i$ 是第 $i$ 个展平后的图像块向量。
-   $E$ 是线性投射层，它本质上是一个可学习的权重矩阵。对于ViT-Base和ViT-Large，这个投影分别将展平的图像块投射到维度 $D=768$ 和 $D=1024$。由于这些特征维度都大于输入图像块中的像素数量（$16 \times 16 \times 3 = 768$），这意味着嵌入过程可以学习保留全部信息，如果这对任务有利，原则上甚至可以解析出像素级的精度。
-   $E(p_i)$ 就是每个图像块经过投射后得到的“token”或嵌入向量。
-   $t_0^0$ 指的是 **readout token**。
-   最终，我们得到一组 $N_p+1$ 个tokens $t^0 = \{t_0^0, ..., t_{N_p}^0\}$，其中 $t_n^0 \in \mathbb{R}^D$，$N_p = \frac{HW}{p^2}$ 是图像块的总数，$D$ 是每个token的特征维度。
-   $E_{pos}$ 是位置嵌入（Positional Embedding）。因为Transformer的自注意力机制本身不包含位置信息，所以必须显式地为每个token添加一个位置编码，告诉模型每个图像块的原始空间位置。

**b. Transformer编码层 (Transformer Encoder Layers)**

这个token序列 $t^0$ 接着被送入 $L$ 个标准的Transformer编码层进行转换，生成新的表示 $t^l$，其中 $l$ 指代第 $l$ 个Transformer层的输出。DPT主要使用了ViT的三种变体：
*   **ViT-Base**：使用基于图像块的嵌入过程，并具有12个Transformer层。
*   **ViT-Large**：使用相同的嵌入过程，但有24个Transformer层和更宽的特征尺寸 $D=1024$。
*   **ViT-Hybrid**：采用ResNet50计算图像嵌入，后接12个Transformer层。

每个编码层主要由两个子模块构成：

-   **多头自注意力模块 (Multi-Head Self-Attention, MHSA):** 这是Transformer的核心。它允许模型中的每一个token关注到序列中的所有其他token，并计算它们之间的相关性权重。对于DPT的应用而言，关键在于MHSA本质上是一种**全局操作**，因为每个token都可以关注（attend to）并因此影响其他所有token。因此，Transformer在初始嵌入后的每个阶段都具有**全局感受野**。这与卷积网络形成鲜明对比，后者随着特征通过连续的卷积和下采样层而逐渐增加其感受野。
    其核心计算公式为：
    $$
    \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
    $$
    -   $Q$ (Query), $K$ (Key), $V$ (Value) 是从输入token序列通过线性变换得到的三個矩阵。
    -   “多头”则意味着这个过程会并行地进行多次，每一“头”学习不同的注意力模式，从而捕捉更丰富的特征关系。

-   **前馈网络 (Feed-Forward Network, FFN):** 这是一个简单的多层感知机（MLP），通常由两个线性层和一个激活函数（如GELU）组成，用于对自注意力模块的输出进行非线性变换和特征增强。

这些编码层会堆叠多次（例如，ViT-Base有12层），每一层都会对token序列进行更深层次的特征提炼。一个至关重要的特性是，Transformer在所有计算过程中都**保持token的数量不变**。由于tokens与图像块之间存在一对一的对应关系，这意味着ViT编码器在所有Transformer阶段都保持了初始嵌入的空间分辨率。

---

### 编码器部分的数据流过程

下面是编码器部分数据处理流程的文字和箭头表示：

**输入图像**
(尺寸: $H \times W \times C$)
` | `
` | `
` V `
**1. 图像分块 (Image Patching)**
`将图像分割成 Np 个 p x p 的非重叠块 (p=16)`
` | `
` | `
` V `
**展平的图像块**
(数量: $N_p$, 每个块尺寸: $p^2 \times C$)
` | `
` | `
` V `
**2. 线性投射 (Linear Projection)**
`每个展平的块通过一个线性层 E，映射到 D 维`
` | `
` | `
` V `
**图像Tokens**
(数量: $N_p$, 每个Token尺寸: $D$)
` | `
` | `
` V `
**3. 添加特殊Tokens和位置编码**
`a. 在序列最前面加入一个可学习的 [readout token] (D维)`
`b. 为序列中的每一个Token (包括readout token) 添加一个可学习的 [位置嵌入]`
` | `
` | `
` V `
**初始Token序列 $t^0$**
(尺寸: $(N_p + 1) \times D$)
` | `
` | `
` V `
**4. Transformer 编码层 1...L (循环 L 次)**
`--->` **多头自注意力 (MHSA)**
      `每个Token计算与所有其他Token的注意力权重，实现全局信息交互，更新自身表示`
      ` | `
      ` V `
      **前馈网络 (FFN)**
      `对MHSA的输出进行非线性变换和特征增强`
` | `
` | `
` V `
**最终输出的Token序列 $t^L$**
(尺寸: $(N_p + 1) \times D$, 空间分辨率保持不变，但特征被深度提炼)

-----
#### 2. 解码器 (Decoder)：卷积与上采样

编码器的输出是一个经过充分信息交互和提炼的token序列。然而，对于密集预测任务（如深度估计），我们需要的是一个与原图分辨率相同（或接近）的像素级预测图。解码器的作用就是将这个一维的token序列**重新组装**并**上采样**回一个2D的特征图。

DPT的解码器设计是其成功的关键，它巧妙地融合了来自Transformer不同层级的特征。它将tokens集合组装成不同分辨率的图像状特征表示，然后被逐步融合，形成最终的稠密预测。

![](../../../../99_Assets%20(资源文件)/images/496c76481f14d0db65fcc2e1f24d899d.png)

**a. 重组操作 (Reassemble Operation)**

解码器的第一步是将编码器输出的token序列重新排列组合成一个2D的特征图。论文中将这个过程定义为一个由三部分组成的 **Reassemble** 操作，用于从Transformer编码器任意层的输出tokens中恢复图像状表示：
$$
\mathrm{Reassemble}_{\hat{D}}^s(t) = (\mathrm{Resample}_s \circ \mathrm{Concatenate} \circ \mathrm{Read})(t)
$$
其中，$s$ 表示恢复后的特征图相对于输入图像的尺寸比例（例如，$s=16$ 意味着特征图分辨率是原图的 1/16），$\hat{D}$ 表示输出特征维度。

这个操作可以分解为以下三个核心步骤：

1. **Read 操作 (处理Readout Token):**
   此操作首先将 $N_p+1$ 个输入token映射到 $N_p$ 个适合空间重组的token。它的主要职责是恰当地处理 **readout token**。虽然这个token对稠密预测任务没有直接用途，但它在编码器中聚合了全局信息。因此，如何利用或舍弃它的信息至关重要。论文评估了三种策略：

   *   **Read_ignore**: 简单地忽略readout token，只保留图像块token。
   *   **Read_add**: 将readout token的表示加到所有其他图像块token上，将全局信息分发出去。
   *   **Read_proj**: 将readout token与每个图像块token拼接，然后通过一个MLP（线性层+GELU激活）将表示投影回原始维度D。这是默认架构采用的方式，能更灵活地融合全局信息。

2. **Concatenate 操作 (空间重组):**
   在Read操作之后，生成的 $N_p$ 个token（维度为D）被重新塑形为一个图像状的表示。具体来说，就是将这 $N$ 个维度为 $D$ 的token，按照它们原先在图像中的空间位置，重新排列成一个 $\frac{H}{p} \times \frac{W}{p} \times D$ 的三维特征图。例如，对于 $224 \times 224$ 的输入和 $16 \times 16$ 的patch大小，我们会得到一个 $14 \times 14 \times D$ 的特征图。
   $$
   \mathrm{Concatenate}: \mathbb{R}^{N_p \times D} \to \mathbb{R}^{\frac{H}{p} \times \frac{W}{p} \times D} \tag{5}
   $$
   ==注意：这里虽然叫Concatenate操作，但本质上是一个重塑操作，将token转换为特征图的形式。==

3. **Resample 操作 (尺寸与维度调整):**
   最后，Concatenate产生的特征图会经过一个空间重采样层，将其尺寸调整为 $\frac{H}{s} \times \frac{W}{s}$，并将特征维度（通道数）调整为 $\hat{D}$（默认为256）。此操作通过以下方式实现：

   *   首先使用一个 $1 \times 1$ 卷积将特征维度从 $D$ 投影到 $\hat{D}$。
   *   接着，根据需要进行空间尺寸的调整。

**b. 逐层上采样与融合 (Progressive Upsampling and Fusion)**

DPT的解码器包含多个上采样阶段，逐步将低分辨率的特征图恢复到原始尺寸。其精妙之处在于，它不仅仅使用了Transformer最后一层的输出，而是**融合了来自编码器多个中间层的特征**。

具体来说，DPT会从编码器的四个不同阶段（层级）和四种不同分辨率（1/32, 1/16, 1/8, 1/4）进行Reassemble操作。
*   使用 **ViT-Large** 时，从第 `l={5, 12, 18, 24}` 层提取token。
*   使用 **ViT-Base** 时，从第 `l={3, 6, 9, 12}` 层提取token。
*   使用 **ViT-Hybrid** 时，使用来自嵌入网络（ResNet）的两个特征，以及Transformer的第 `l={9, 12}` 层。

然后，DPT使用一个基于 **RefineNet** 的**特征融合模块 (Fusion Module)** 来组合这些来自连续阶段的特征图。一个典型的融合阶段包含以下步骤：

1.  **特征图处理:** 对从上一阶段传入的特征图（或初始的Reassemble特征图）应用一个**残差卷积单元 (Residual Conv Unit)** 进行处理，以提取和整合特征。
2.  **特征融合 (Fusion):** 将处理后的特征图与**来自ViT编码器相应层级、经过Reassemble操作后的特征图**在通道维度上进行拼接（Concatenate）或相加。
3.  **上采样 (Resample):** 使用`Resample`操作（通常是双线性插值）将融合后的特征图尺寸**放大2倍** (即 `Resample_0.5`)。
4.  **卷积处理 (Project):** 再通过一个卷积层（Project）对上采样后的特征进行进一步的提炼和维度调整。

这个“融合-上采样”的过程会重复进行，从最低分辨率（1/32）开始，逐步融合更高分辨率的特征，直到特征图的分辨率恢复到输入图像的**一半**。最后，再连接一个任务特定的输出头（Head）生成最终的像素级预测。

**c. 为什么需要融合多层特征？**

-   **深层特征:** 来自ViT编码器后几层的特征（如第18、24层），经过了多次自注意力计算，包含了丰富的**全局语义信息**（知道图像里“有什么”）。这些特征经过Reassemble后形成低分辨率（如1/32, 1/16）的特征图。
-   **浅层特征:** 来自编码器前几层的特征（如第5、12层），更多地保留了**局部细节和空间信息**（知道物体的边缘、纹理“在哪里”）。这些特征经过Reassemble后形成较高分辨率（如1/8, 1/4）的特征图。

通过将不同层级的特征融合在一起，DPT的解码器能够同时利用全局的语义理解和局部的空间细节，这对于生成高质量、边缘清晰的密集预测结果至关重要。

**d. 处理可变图像尺寸**

DPT可以处理不同尺寸的输入图像。当输入图像尺寸变化时，图像块的数量 $N_p$ 会改变，但Transformer编码器作为一种集合到集合的架构，可以轻松处理可变数量的tokens。唯一需要调整的是**位置嵌入**，DPT采用的方法是对预训练的位置嵌入进行2D线性插值，以适应新的图像尺寸。解码器的Reassemble和Fusion模块本身也是全卷积的，因此也能自然地处理不同分辨率的特征图。

---

### 解码器部分的数据流过程图

下面是解码器部分数据处理流程的文字和箭头表示，整合了您提供的所有信息：

**来自ViT编码器不同层的Token序列**
(例如, 来自第5, 12, 18, 24层的4组Token序列, 每组尺寸为 $(N_p+1) \times D$)
` | `
` | `
` V `
**1. 并行的Reassemble操作 (在4个尺度上)**
`--->` **Reassemble_32 (使用最深层, 如L24的Token)**
      ` a. Read_proj: 处理readout token并融合其信息`
      ` b. Concatenate: 重组为 1/16 分辨率的特征图`
      ` c. Resample: 调整为 1/32 分辨率, 维度为 D_hat=256`
      ` | `
`--->` **Reassemble_16 (使用较深层, 如L18的Token)**
      ` a. Read_proj`
      ` b. Concatenate`
      ` c. Resample -> 1/16 分辨率, D_hat=256`
      ` | `
`--->` **Reassemble_8 (使用较浅层, 如L12的Token)**
      ` ... -> 1/8 分辨率, D_hat=256`
      ` | `
`--->` **Reassemble_4 (使用最浅层, 如L5的Token)**
      ` ... -> 1/4 分辨率, D_hat=256`
` | `
` | `
` V `
**2. 逐级的特征融合与上采样 (Fusion Modules)**

**[阶段1: Fusion]**
`1/32 特征图 -> [Residual Conv Unit] -> 与 1/16 特征图融合 -> [Resample_0.5 (上采样x2)] -> [Project]`
` | `
` V `
**输出: 1/8 分辨率的融合特征图**

**[阶段2: Fusion]**
`1/8 融合特征图 -> [Residual Conv Unit] -> 与 1/8 Reassemble特征图融合 -> [Resample_0.5 (上采样x2)] -> [Project]`
` | `
` V `
**输出: 1/4 分辨率的融合特征图**

**[阶段3: Fusion]**
`1/4 融合特征图 -> [Residual Conv Unit] -> 与 1/4 Reassemble特征图融合 -> [Resample_0.5 (上采样x2)] -> [Project]`
` | `
` V `
**最终解码器输出: 1/2 分辨率的最终特征图**
` | `
` | `
` V `
**3. 任务输出头 (Task Head)**
`通过一个最终的卷积层，将 1/2 分辨率的特征图转换为最终的像素级预测（例如深度图）`
` | `
` V `
**最终预测结果**
(尺寸: $H/2 \times W/2 \times 1$)

***

### 附录中的具体架构细节

![](../../../../99_Assets%20(资源文件)/images/21f8bdb3208e3929296b4ccdec90360e.png)

- **混合编码器 (Hybrid encoder)**：混合编码器基于具有组归一化和权重标准化 [57] 的预激活ResNet50。它在初始stem后定义了四个阶段，每个阶段在应用多个ResNet块之前对表示进行下采样。RN指的是第N个阶段的输出。DPT-Hybrid因此在第一（R0）和第二阶段（R1）后连接了跳跃连接。

- **残差卷积单元 (Residual convolutional units)**：图A1(a)展示了解码器中使用的残差卷积单元 [23] 的示意图。语义分割使用批归一化，但单目深度估计禁用批归一化。使用批归一化时，禁用前一个卷积层中的偏差。

- **单目深度估计头 (Monocular depth estimation head)**：单目深度估计的输出头如图A1(b)所示。初始卷积将特征维度减半，而第二个卷积的输出维度为32。最终的线性层将此表示投影为一个非负标量，表示每个像素的逆深度预测。使用双线性插值对表示进行上采样。

- **语义分割头 (Semantic segmentation head)**：语义分割的输出头如图A1(c)所示。第一个卷积块保留了特征维度，而最终的线性层将表示投影到输出类别数。使用丢弃层 (Dropout)，比率为0.1。最终的上采样操作使用双线性插值。因此，预测表示每个像素的类别logit。

### DPT模型的优势总结

1. **强大的全局上下文建模能力:** 凭借Transformer的自注意力机制，DPT能够捕捉图像中任意像素之间的长距离依赖关系，这对于理解大的场景和物体间的关系非常有帮助，远超传统CNN有限的感受野。
2. **灵活的骨干网络:** DPT证明了ViT作为一种通用的特征提取器，可以被成功应用于密集预测任务，打破了CNN在该领域的垄断地位。
3. **高效的多尺度特征融合:** 其精心设计的解码器能够有效地融合ViT编码器不同阶段的特征，兼顾了高级语义信息和低级空间细节，从而实现了卓越的性能。

总而言之，DPT模型的核心思想是“**用Transformer的全局视野进行编码，用卷积的局部精细操作进行解码**”，通过一个巧妙的解码器将这两者的优势结合起来，为密集预测领域开辟了一个新的、高效的范式。
