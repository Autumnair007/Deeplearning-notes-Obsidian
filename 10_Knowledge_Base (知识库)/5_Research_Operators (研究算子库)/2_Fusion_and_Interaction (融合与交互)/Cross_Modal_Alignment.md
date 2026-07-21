---
type: operator-note
aliases:
  - Cross-Modal Alignment
  - 跨模态对齐
tags:
  - research-operator
  - cross-modal-alignment
  - vision-language
  - semantic-segmentation
  - weakly-supervised
  - open-vocabulary
status: in-progress
---

# Cross-Modal Alignment（跨模态对齐）

> [!abstract] 本页定位
> 本页不是重复介绍某一篇论文，而是整理**如何让视觉单元与文本语义建立可用于密集预测的对应关系**。当前版本只从 `3_Methods_and_Frameworks` 中弱监督语义分割和开放词汇语义分割相关笔记提炼，后续再补源码位置和其他领域中的实现。

## 1. 这个操作解决什么问题？

视觉—语言模型通常能判断“整张图是否与一句文本匹配”，但分割任务需要回答更细的问题：

> 图像中的**哪个像素、补丁或区域**对应文本类别 $c$？

这中间至少存在三种错位：

1. **粒度错位**：CLIP预训练对齐的是全局图像和整句文本，分割需要局部视觉单元和类别文本对齐。
2. **模态错位**：视觉特征更容易保留颜色、纹理和形状，文本特征更强调抽象语义；即使维度相同，也不代表两者在同一个语义坐标系中。
3. **空间—语义错位**：CLIP语义识别较强但空间定位可能过平滑；DINO、SAM或扩散模型能提供更好的结构，却不一定知道区域属于哪个开放词汇类别。

因此，跨模态对齐并不只是“计算一次余弦相似度”，而是一条完整的数据流：

$$
\text{选择视觉单元}
\rightarrow
\text{统一特征空间}
\rightarrow
\text{计算视觉—文本对应}
\rightarrow
\text{恢复空间预测}.
$$

> [!note] 我的理解｜先判断问题发生在哪一步
> 如果CAM能识别正确类别但边界模糊，问题未必出在跨模态语义对齐，而可能出在视觉特征的空间结构；如果模型把其他前景物体也激活成目标类别，才更像是视觉—文本语义间隔没有拉开。不要看到分割图不好就统一归因于“CLIP对齐不足”。

## 2. 统一输入输出张量

设批量大小为 $B$，视觉特征维度为 $D_v$，文本特征维度为 $D_t$，类别数为 $C$。

### 2.1 常见视觉单元

| 对齐粒度 | 视觉张量 | 含义 |
|---|---|---|
| 全局图像 | $V_g\in\mathbb{R}^{B\times D_v}$ | 每张图一个向量 |
| 补丁/像素 | $P\in\mathbb{R}^{B\times N\times D_v}$ | $N=H'W'$，保留空间位置 |
| 区域/掩码 | $Z\in\mathbb{R}^{B\times R\times D_v}$ | 每个候选区域一个向量 |
| 视觉原型 | $P_v\in\mathbb{R}^{K\times D_v}$ | 类别或簇的代表性视觉中心 |

文本类别通常编码为：

$$
T\in\mathbb{R}^{C\times D_t}.
$$

如果两个编码器不共享特征空间，需要先学习映射：

$$
\psi:T\in\mathbb{R}^{C\times D_t}
\rightarrow
T'\in\mathbb{R}^{C\times D_v}.
$$

### 2.2 最常见的密集输出

对补丁特征和文本特征做L2归一化后：

$$
\hat P=\frac{P}{\|P\|_2},\qquad
\hat T=\frac{T}{\|T\|_2},
$$

这里的归一化是在最后一个特征维度 $D$ 上进行。它把每个视觉补丁和每个类别文本都缩放成长度为1的向量，因此后续点积只比较两个向量的方向，不再受向量模长影响。方向越接近，通常表示视觉内容和类别文本在共享空间中越匹配。

密集类别响应为：

$$
S=\hat P\hat T^T
\in\mathbb{R}^{B\times N\times C}.
$$

这一步可以理解为：**让每个视觉补丁依次和全部类别文本做一次余弦相似度比较。** $\hat P$ 的形状是 $[B,N,D]$，$\hat T^T$ 的形状是 $[D,C]$；矩阵乘法会消去共同的特征维 $D$，留下 $[B,N,C]$。因此，$S[b,n,c]$ 表示第 $b$ 张图中第 $n$ 个补丁与第 $c$ 个类别文本的匹配分数。

例如，若 $B=2$、CLIP特征网格为 $20\times20$、类别数 $C=20$，那么：

$$
\hat P:[2,400,512],\qquad
\hat T^T:[512,20],\qquad
S:[2,400,20].
$$

此时 $S$ 已经保存了密集类别响应，但空间维仍被展平成长度为400的序列，还不能直接当作二维分割图显示。

第一步，把类别维移动到PyTorch图像张量常用的通道位置，并将 $N=H'W'$ 还原为空间网格：

$$
S_{\mathrm{map}}
=\mathrm{Reshape}\bigl(\mathrm{Permute}(S)\bigr)
\in\mathbb{R}^{B\times C\times H'\times W'}.
$$

对应的张量变化是：

```text
[B, N, C]
→ permute(0, 2, 1)
→ [B, C, N]
→ reshape(B, C, H', W')
→ [B, C, H', W']
```

这里没有产生新的预测值，只是把“第几个补丁”重新放回它在二维网格中的位置。例如长度为400的补丁序列会恢复成 $20\times20$ 的低分辨率类别响应图。

第二步，将低分辨率响应图恢复到输入图像大小：

$$
M=\mathrm{Upsample}(S_{\mathrm{map}})
\in\mathbb{R}^{B\times C\times H\times W}.
$$

最常见的是双线性插值，例如 `F.interpolate(..., mode="bilinear")`。它没有可学习参数，只根据周围位置平滑地补出新像素；如果使用转置卷积，则上采样核可以通过训练学习，但会增加参数，并可能引入棋盘格伪影。更完整的上采样方法可参考 [[downsampling_and_upsampling(下采样与上采样)]]。

在带解码器的分割模型中，上采样后还可能与浅层高分辨率特征进行相加或拼接，以补充边缘细节；如果只是把CLIP相似度图放大成CAM，通常直接插值即可，不一定存在跳跃连接。

对应的基础代码是：

```python
scores = scores.permute(0, 2, 1)              # [B, C, N]
scores = scores.reshape(B, C, H_patch, W_patch)
scores = F.interpolate(
    scores,
    size=(H_image, W_image),
    mode="bilinear",
    align_corners=False,
)
```

> [!warning] 同维度不等于已经对齐
> 两个张量都是 $D=512$ 只能说明矩阵乘法可以执行，不能说明它们的坐标轴具有相同语义。CLIP图像与文本编码器通过对比预训练建立了共享空间；CLIP文本与DINO视觉特征则需要额外映射、原型桥接或检索机制。

## 3. 先从论文建立直观认识

下面先不急着抽象分类，而是具体看当前笔记中的论文分别发现了什么问题、怎样解决，以及最后得到什么。

| 论文 | 任务与起点 | 它认为原方法哪里不够 | 具体做法 | 与跨模态对齐的关系 |
|---|---|---|---|---|
| [[CLIP-ES_paper_notes]] | 图像级标签WSSS；使用冻结CLIP | CLIP只有全局图文匹配分数，没有直接的像素类别 | 设计前景/背景文本提示，用概率形式的图文分数生成Grad-CAM，再利用ViT注意力细化 | 保留全局图文对齐，通过梯度间接寻找空间位置 |
| [[WeCLIP_paper_notes]] | 单阶段WSSS；冻结CLIP作为骨干 | 初始CLIP CAM不完整，单独作为伪标签噪声较大 | 从CLIP得到初始CAM，同时训练轻量解码器；再用冻结CLIP注意力关系和解码器结果互相细化 | 跨模态匹配主要负责类别种子，空间补全交给视觉关系与解码器 |
| [[ExCEL_paper_notes]] | 基于CLIP的WSSS；补丁—文本匹配 | 单一类别文本语义太少，CLIP patch又过度平滑 | 用大语言模型产生细粒度属性描述以增强文本；同时用静态和可学习视觉校准改善patch关系 | 同时修改对齐两端：文本更具体，视觉patch更可分 |
| [[ComCD_paper_notes]] | CLIP与扩散模型协同的WSSS | CLIP类别定位较强但物体内部不完整；扩散模型空间连续但类别特异性弱 | 分别生成CLIP CAM和扩散CAM，根据每个像素的预测熵动态决定更相信哪一支 | 把“语义对齐”和“空间结构”作为互补证据融合 |
| [[DiCLIP_paper_notes]] | 补丁—文本WSSS；冻结CLIP和SD | CLIP视觉patch空间感知不足，单一文本向量也不能覆盖类内外观 | VCE把SD空间亲和力注入CLIP注意力；TSA用SD生成图建立视觉键值缓存，让patch既匹配文本又检索视觉原型 | 把直接patch—text匹配扩展为“视觉校准 + 文本CAM + 视觉知识检索” |
| [[SSR_paper_notes]] | 基于CLIP的WSSS | 图像与文本存在模态间隙，导致非目标前景被激活；亲和力传播还会污染背景 | ISA/TSA分别投影视觉和文本特征，再用跨模态原型对比学习拉近同类；之后用超像素限制空间传播 | 先在新的共享子空间中校正语义对应，再单独处理空间噪声 |
| [[OpenSeg_paper_notes]] | 开放词汇分割；图像—标题训练 | 整图或逐像素对齐难以获得完整对象，标题中的词也没有区域标注 | 先预测类别无关掩码，把掩码内像素池化成区域特征，再让区域和标题词语进行对比式接地 | 把对齐粒度从单个patch提升到完整候选区域 |
| [[Talk2DINO_paper_notes]] | 无监督开放词汇分割；DINOv2视觉特征 + CLIP文本 | DINOv2有优秀空间结构却没有语言接口；CLIP文本不能直接和DINO patch比较 | 学习一个轻量非线性映射，把CLIP文本嵌入变换到DINOv2视觉空间，并用图像—标题对训练 | 不强迫视觉特征迁移到CLIP空间，而是把文本主动送入更适合定位的视觉空间 |
| [[CorrCLIP_paper_notes]] | 免训练开放词汇分割；CLIP分类器 | CLIP patch之间存在错误类间相关性，导致即使文本分类器正确也会输入糟糕的视觉特征 | 用SAM限制patch交互范围，用DINO重建相似度值，再与CLIP Value聚合后和文本分类 | 主要改造对齐前的视觉表示，说明最终点积之外的视觉相关性同样关键 |
| [[Trident_paper_notes]] | 高分辨率开放词汇分割；CLIP+DINO+SAM | 滑窗“先分割后拼接”破坏全局感受野并产生窗口接缝 | 先拼接子图特征再统一分割；DINO负责局部对象关系，SAM负责全局聚合，CLIP负责类别语义 | 保留CLIP文本分类接口，但重新构造进入接口的高分辨率视觉特征 |
| [[ReME_paper_notes]] | 参考集驱动的开放词汇分割 | 直接跨模态相似度容易受模态差异影响，未知标签关系难以可靠迁移 | 分别计算测试片段与参考视觉片段、测试标签与参考标签的模态内相似度，再通过参考片段—标签关系聚合 | 用参考集作桥梁，把困难的直接跨模态匹配拆成两个相对稳定的模态内检索 |

> [!note] 我的理解｜这些论文并不是在解决同一个小问题
> CLIP-ES和WeCLIP关心“怎样从全局CLIP得到可用CAM”；ExCEL、ComCD和DiCLIP关心“直接patch—text匹配还缺少什么”；SSR关心“模态间隙怎样通过训练缩小”；OpenSeg关心“对齐单元是否应该从patch变成区域”；Talk2DINO关心“两个本来不共享空间的模型怎样连接”；CorrCLIP和Trident则提醒我们，最终文本分类之前的视觉特征质量可能才是瓶颈。

### 3.1 从论文中归纳出的实现形式

| 实现形式 | 对齐单元 | 是否需要训练 | 优点 | 主要限制 | 代表工作 |
|---|---|---:|---|---|---|
| 全局图文匹配 + 梯度定位 | 图像—文本 | 可选 | 直接复用CLIP最可靠的全局能力 | 空间响应间接，易集中于判别区域 | [[CLIP-ES_paper_notes]]、[[WeCLIP_paper_notes]] |
| 补丁—文本相似度 | patch—类别文本 | 否 | 路径直接，天然输出密集响应 | CLIP patch语义和边界不一定可靠 | [[DiCLIP_paper_notes]]、[[ExCEL_paper_notes]]、[[ComCD_paper_notes]] |
| 区域—文本对齐 | mask/region—词语 | 通常需要 | 区域内部一致，边界更完整 | 依赖区域提议质量 | [[OpenSeg_paper_notes]] |
| 跨模态原型对齐 | 特征—类别原型 | 通常需要 | 用稳定锚点减少样本噪声和模态间隙 | 原型会受伪掩码、聚类和长尾分布影响 | [[SSR_paper_notes]] |
| 学习空间映射 | 文本—另一视觉空间 | 需要轻量训练 | 可组合CLIP语义与DINO空间结构 | 映射质量依赖训练数据和区域选择 | [[Talk2DINO_paper_notes]] |
| 检索式对齐 | patch—视觉键值记忆 | 静态或轻量训练 | 文本类别可借助多种视觉外观 | 缓存覆盖率、合成偏差和内存开销 | [[DiCLIP_paper_notes]]、[[ReME_paper_notes]] |

这些形式不是完全互斥的。一篇论文可以先做补丁—文本相似度，再用视觉原型、空间亲和力或外部缓存修正结果。它们的共同点是最终都要建立“视觉单元—语义类别”的关系；主要差异在于视觉单元的粒度、两端是否共享空间，以及这个关系是固定计算还是通过训练学习。

## 4. 各种形式怎样工作？

### 4.1 全局图文匹配 + 梯度定位

**核心思路**：先利用CLIP可靠的全局图文相似度得到类别分数，再通过Grad-CAM等方法把类别分数对视觉特征的梯度还原为空间热图。

数据流可以写成：

$$
I\rightarrow V_g,\qquad
\{t_c\}_{c=1}^{C}\rightarrow T,\qquad
s_c=\cos(V_g,t_c)
\rightarrow \mathrm{GradCAM}(s_c).
$$

大白话来看，这条公式分成三步：图像编码器先把整张图压缩成全局向量 $V_g$；文本编码器把第 $c$ 个类别名称编码成 $t_c$；二者的余弦相似度 $s_c$ 回答“整张图像不像类别 $c$”。这个分数只有一个数，没有空间位置，因此Grad-CAM继续计算 $s_c$ 对某层二维视觉特征的梯度：哪些位置稍微变化就会明显影响类别分数，哪些位置就更可能与该类别有关。

它利用的是CLIP最擅长的全局对齐，但空间位置来自梯度，而不是视觉补丁与文本的直接匹配。最终热图的空间大小取决于被选中的视觉层，通常还要插值回输入图像大小。

**适合**：希望几乎不训练模型，或者需要一个初始CAM种子。

**局限**：容易只覆盖最有判别力的局部；梯度定位质量还会受到所选视觉层和注意力结构影响。

在现有笔记中，[[WeCLIP_paper_notes]] 使用冻结CLIP产生初始CAM，随后再通过注意力和解码器形成伪标签闭环。这说明“生成语义种子”和“补全空间区域”可以由不同模块负责。

### 4.2 补丁—文本直接对齐

这是密集预测中最直接的形式。令：

$$
P\in\mathbb{R}^{B\times H'\times W'\times D},\qquad
T\in\mathbb{R}^{C\times D},
$$

则每个patch对每个类别的响应为：

$$
M_t=\mathrm{Norm}(\cos(P,T))
\in\mathbb{R}^{B\times H'\times W'\times C}.
$$

这里的 $P[b,i,j,:]$ 是图像网格位置 $(i,j)$ 的patch向量，$T[c,:]$ 是类别 $c$ 的文本向量。对每个空间位置分别计算它与全部 $C$ 个文本向量的余弦相似度，就得到该位置的类别响应。`Norm` 通常把同一类别在全部空间位置上的分数缩放到 $[0,1]$，方便阈值化或生成伪标签；不同论文的归一化轴可能不同，所以必须结合代码确认。

与上一节的全局方法相比，这里不需要通过梯度寻找位置，因为每个patch本来就带着自己的空间索引。代价是：CLIP预训练并没有直接监督每个patch应该对应哪个类别，所以“有位置”不等于“定位准确”。

[[DiCLIP_paper_notes]] 将这种形式作为基础CAM，但明确指出：CLIP按整图目标训练，patch特征虽然保留位置，却未必具有可靠的局部边界和物体完整性。DiCLIP因此从两个方向补强：

1. **VCE（Visual Correlation Enhancement，视觉相关性增强）**用扩散模型的空间关系修正CLIP视觉注意力；它主要改善视觉patch之间“哪些位置应该互相联系”。
2. **TSA（Text Semantic Augmentation，文本语义增强）**用视觉键值缓存补充单一文本向量无法覆盖的类内外观变化；它并不是继续写更多文本，而是借助带类别信息的视觉原型丰富类别语义。

[[ComCD_paper_notes]] 同样在视觉侧先通过patch亲和力聚合特征，再让逐像素特征与类别文本计算余弦相似度。[[ExCEL_paper_notes]] 则同时丰富文本语义并校准视觉关系。

> [!note] 我的理解｜补丁—文本对齐只是最后一步
> $P T^T$ 很容易写，但真正决定CAM质量的是乘法之前的 $P$ 和 $T$：视觉patch是否仍然过平滑，文本是否只表达了类别名称，二者是否覆盖背景和类内变化。很多论文表面上在提出新的“对齐方法”，实际创新主要发生在视觉校准、文本增强或相似度后的空间传播。

### 4.3 区域—文本对齐

补丁粒度容易受到纹理和边界噪声干扰，因此可以先得到类别无关掩码，再对掩码内部的视觉特征进行池化。

[[OpenSeg_paper_notes]] 中，预测掩码为：

$$
s\in\mathbb{R}^{B\times R\times H'\times W'},
$$

像素特征为：

$$
F\in\mathbb{R}^{B\times H'\times W'\times D}.
$$

第 $r$ 个区域的特征常见归一化写法为：

$$
z_r=
\frac{\sum_{i,j}s_{r,i,j}F_{i,j}}
{\sum_{i,j}s_{r,i,j}+\epsilon}
\in\mathbb{R}^{D}.
$$

OpenSeg原文公式直接写成掩码加权求和；这里补上分母，是为了展示更常见的掩码平均池化形式。实际阅读代码时应确认掩码是否已经归一化，避免重复除以区域面积。

这条公式的直觉是：先用第 $r$ 个软掩码 $s_r$ 给区域内的像素特征加权，再把它们汇总成一个向量 $z_r$。掩码值越大的位置，对区域向量贡献越大；分母相当于除以区域的有效面积，避免大区域仅仅因为像素多就得到更大的特征模长。

然后计算区域和词语嵌入的相似度：

$$
S_{r,c}=\cos(z_r,t_c).
$$

$S_{r,c}$ 是“第 $r$ 个候选区域像不像类别 $c$”的分数。此时分类发生在区域层面：区域内部不再让每个像素独立和文本竞争，而是共享同一个语义判断。

推理时再将区域类别分数投回像素：

$$
Y_{c,i,j}=\sum_{r=1}^{R}S_{r,c}s_{r,i,j}.
$$

最后需要把区域判断送回每个像素。若像素 $(i,j)$ 高度属于区域 $r$，那么该区域的类别分数 $S_{r,c}$ 就会较多地贡献给像素输出 $Y_{c,i,j}$。多个软掩码可以重叠，所以同一个像素也可能同时接收多个区域的加权投票。

**优点**：同一区域共享语义，通常比逐patch分类更连贯。

**风险**：掩码漏掉目标时，文本对齐无法补回该区域；掩码合并多个物体时，区域向量会混入多个类别。

### 4.4 跨模态原型对齐

直接对齐单个视觉样本和文本容易受噪声影响，可以先构造类别或语义簇的代表性中心。

[[SSR_paper_notes]] 在CLIP编码器后加入两个结构相同但参数独立的轻量投影模块：

- **ISA（Image Semantic Alignment，图像语义对齐）**：接收CLIP视觉特征，通过 `Linear → BatchNorm → LeakyReLU` 一类的小型MLP，把视觉表示投影到新的对齐空间。
- **TSA（Text Semantic Alignment，文本语义对齐）**：用另一套MLP处理CLIP文本特征，使文本表示也进入同一个新空间。

两者不是新的大型编码器，也不是注意力模块。它们更像两个“翻译器”：CLIP视觉编码器和文本编码器保持冻结，只训练ISA/TSA学习怎样把两种已有特征翻译成更适合当前WSSS数据的表达。

整体思路可以概括为：

1. 用轻量ISA/TSA分别投影视觉和文本特征；
2. 用CAM掩码平均池化提取前景视觉表示；
3. 对视觉和文本表示聚类，获得图像原型 $P^I$ 和文本原型 $P^T$；
4. 通过原型对比学习拉近同类跨模态表示，推开异类表示。

其视觉前景特征可抽象为：

$$
f_{\mathrm{image}}
=\mathrm{MAP}(M_c\odot F_I),
$$

$M_c$ 是类别 $c$ 的CAM，$F_I$ 是ISA输出的空间视觉特征。逐元素相乘先压低背景位置，再通过掩码平均池化（Masked Average Pooling, MAP）把剩余前景汇总成一个向量。这个向量不是单个像素，而是“当前图像中类别 $c$ 的前景摘要”。收集许多图像的前景摘要并做K-means，聚类中心就形成视觉原型；文本特征经过TSA后也可以构建文本原型。

再计算视觉表示与文本原型的相似度：

$$
p_i=\frac{f_i(P^T)^T}{\tau}.
$$

这里 $f_i$ 是一个视觉表示，$P^T$ 包含所有文本原型。$f_i(P^T)^T$ 会一次算出它与全部文本原型的相似度，除以温度 $\tau$ 用来控制分数分布的尖锐程度。训练时把同类文本原型作为正目标、其他类别作为负目标，反向传播只更新ISA/TSA，使同类跨模态特征靠近、异类分开。

**优点**：原型充当跨样本的稳定锚点，不必让每个带噪样本直接决定对齐方向。

**风险**：CAM不完整会污染视觉原型；K-means可能把罕见外观当成异常；单原型还可能把多峰类内分布压成一个中心。

### 4.5 学习不同特征空间之间的映射

当文本和视觉特征来自不同模型时，需要显式学习坐标变换。

[[Talk2DINO_paper_notes]] 保留DINOv2的密集视觉特征，并学习从CLIP文本空间到DINOv2视觉空间的非线性映射：

$$
\psi(t)=W_b^T\tanh(W_a^Tt+b_a)+b_b.
$$

这就是一个两层MLP：第一层 $W_a$ 把CLIP文本特征送到中间空间，`tanh` 提供非线性，第二层 $W_b$ 再输出与DINOv2 patch相同维度的向量。它不是把文本内容改写成另一句话，而是在数值特征空间中改变坐标，使映射后的文本可以和DINO视觉特征直接计算相似度。

DINO不同注意力头对应不同候选区域。将区域加权视觉特征 $v_{A_i}$ 与映射后的文本 $\psi(t)$ 比较，并选取最相关的区域：

$$
\operatorname{score}(I,t)
=\max_i\cos(v_{A_i},\psi(t)).
$$

DINO的不同注意力头可能分别关注前景主体、局部部件或背景区域。$v_{A_i}$ 是根据第 $i$ 个注意力图加权汇总出的区域视觉向量。公式先分别比较文本与所有候选区域，再取最大值：只要其中一个区域与文本高度匹配，这对图像—文本样本就可以被视为正对应。这样训练映射时不必事先知道标题中的词具体落在哪个像素区域。

通过图像—文本对上的对比损失学习 $\psi$，最终可以直接计算DINO patch与任意类别文本的密集相似度。

这种方法适合以下结构性矛盾：

- CLIP文本语义开放，但CLIP patch定位不足；
- DINO patch定位较好，但没有天然语言接口。

### 4.6 检索式对齐

一个类别文本向量很难描述同一类别的姿态、颜色和视角变化。检索式方法不要求patch只和文本向量匹配，而是让它查询带类别信息的视觉记忆。

设patch查询为：

$$
Q\in\mathbb{R}^{B\times N\times D},
$$

缓存键和值为：

$$
K\in\mathbb{R}^{U\times D},\qquad
V\in\mathbb{R}^{U\times C},
$$

则密集检索为：

$$
M_{\mathrm{cache}}
=\mathrm{Norm}\left(\sigma(QK^T)V\right)
\in\mathbb{R}^{B\times N\times C}.
$$

这条公式可以拆成三步：

1. $QK^T:[B,N,D]\times[D,U]\rightarrow[B,N,U]$：每个patch分别和 $U$ 个视觉键比较，得到“它像哪些缓存原型”。
2. $\sigma(\cdot)$：把原始相似度变成非负或更稳定的检索权重，具体激活函数由实现决定。
3. $[B,N,U]V:[U,C]\rightarrow[B,N,C]$：每个缓存键都绑定一个类别值，矩阵乘法把“像哪些原型”转换为“像哪些类别”。

因此，文本对齐提供的是“patch像类别词的程度”，缓存检索提供的是“patch像哪些已有视觉外观，而这些外观属于什么类别”。两条路径可以相加，互相补充。

[[DiCLIP_paper_notes]] 用扩散模型生成的单类图像建立前景/背景视觉缓存，把patch—文本匹配扩展成patch—视觉知识检索。静态缓存提供不训练的参考CAM，动态适配器再适应真实训练数据。

**优点**：一个类别可以对应多个视觉键，能够表达多种类内外观。

**风险**：缓存中的合成图像可能与真实数据存在域差异；键太少覆盖不足，键太多则增加检索和存储成本。

## 5. 工程实例：DiCLIP官方代码中的对齐与检索

本节只追踪一个仓库：[zwyang6/DiCLIP](https://github.com/zwyang6/DiCLIP)。代码基于提交 [`1c3f6ff`](https://github.com/zwyang6/DiCLIP/tree/1c3f6ff7d4fde2afff32d527d78b28d119583602) 阅读，不要求把整个项目运行起来。

### 5.1 先看主调用链

在 [`DiCLIP_model.forward`](https://github.com/zwyang6/DiCLIP/blob/1c3f6ff7d4fde2afff32d527d78b28d119583602/model/model_diclip.py#L148-L185) 中，与本页最相关的数据流是：

```text
输入图像 img
  ↓
generate_clip_fts
  └─ 得到 image_features [B, 1+N, 512]
       ├─ 第0个token：CLS全局特征
       └─ 后N个token：空间patch特征
  ↓
clip_feature_surgery(image_features, text_features)
  └─ 得到基础patch—text CAM attr_maps_raw [B, N, C_fg]
  ↓
diff_knowledge_inject_cam(kv_cache, image_features)
  └─ 得到静态缓存CAM diff_maps [B, N, C_fg]
  ↓
attr_maps_raw + 0.5 × diff_maps
  └─ 形成静态增强CAM

image_features[:, 1:, :]
  ↓
dynamic_adapter
  └─ 得到动态缓存响应 dynamic_maps [B, N, C]
  ↓
permute + reshape
  └─ dynamic_maps_pred [B, C, H', W']
```

对应的主干代码非常短：

```python
image_features, attn_weights, all_feats = clip.generate_clip_fts(
    img, self.encoder, return_weights=True, ex_feats=diff_attn
)

attr_maps_raw = clip.clip_feature_surgery(
    image_features, self.integral_text_features
)[:, 1:, :self.num_classes - 1]

diff_maps = self.diff_knowledge_inject_cam(
    self.kv_cache, image_features
)

fuse = 0.5 * diff_maps + attr_maps_raw

dynamic_maps = self.dynamic_adapter(image_features[:, 1:, :])
dynamic_maps_pred = dynamic_maps.permute(0, 2, 1).reshape(
    b, self.num_classes, f_h, f_w
)
```

这几行正好对应论文中的三种信号：基础文本CAM、固定视觉缓存CAM、可学习动态缓存CAM。代码没有把它们包装成很神秘的算子，本质上仍然是特征提取、相似度、加权相加和张量变形。

### 5.2 文本特征怎样得到？

模型初始化时，代码先构造前景类别和一组背景词，然后调用 [`encode_text_with_prompt_ensemble`](https://github.com/zwyang6/DiCLIP/blob/1c3f6ff7d4fde2afff32d527d78b28d119583602/clip/clip.py#L252-L269)：

```python
text_prompts = new_class_names + BACKGROUND_CATEGORY
self.integral_text_features = clip.encode_text_with_prompt_ensemble(
    self.encoder,
    text_prompts,
    device,
    prompt_templates=["a clean origami {}."],
)
```

函数内部的流程是：

```python
class_embeddings = model.encode_text(prompted_t)
class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
class_embedding = class_embeddings.mean(dim=0)
class_embedding /= class_embedding.norm()
```

先把类别名称放入模板，例如 `a clean origami dog.`；文本编码器得到每个模板的向量后做L2归一化；如果同一类别使用多个模板，就先求平均再归一化。最终得到的 `integral_text_features` 可以理解为一个文本分类器权重表，每一行代表一个前景或背景语义。

> [!note] 代码阅读点｜“prompt ensemble”不一定真的用了多个模板
> 函数支持多个prompt模板并对它们求平均，但DiCLIP主模型当前传入的列表只有一个模板。因此接口具有prompt ensemble能力，当前配置却相当于单模板编码。阅读代码时应区分“函数能做什么”和“实验配置实际做了什么”。

### 5.3 基础patch—text CAM不是简单的一行矩阵乘法

基础CAM来自 [`clip_feature_surgery`](https://github.com/zwyang6/DiCLIP/blob/1c3f6ff7d4fde2afff32d527d78b28d119583602/clip/clip.py#L288-L310)。为了看懂下面的张量，先设：

```text
image_features: [B, 1+N, D]
text_features:  [C, D]
```

官方代码的核心部分是：

```python
prob = image_features[:, :1, :] @ text_features.t()  # [B, 1, C]
prob = (prob * 2).softmax(-1)
weight = prob / prob.mean(-1, keepdim=True)

features = (
    image_features.reshape(B, 1 + N, 1, D)
    * text_features.reshape(1, 1, C, D)
)                                                       # [B, 1+N, C, D]
features *= weight.unsqueeze(-1)

redundant = features.mean(2, keepdim=True)              # [B, 1+N, 1, D]
features = features - redundant
similarity = features.sum(-1)                           # [B, 1+N, C]

attr_maps = (similarity - similarity.min(1, keepdim=True)[0]) / (
    similarity.max(1, keepdim=True)[0]
    - similarity.min(1, keepdim=True)[0]
)
```

逐步解释如下：

1. **CLS token先估计整图类别权重。** `image_features[:, :1, :]` 取全局CLS特征，与所有文本类别比较，得到 `[B,1,C]`。它不是最终CAM，而是用于调节各类别后续patch响应的权重。
2. **每个patch与每个文本逐通道相乘。** reshape后广播相乘，得到 `[B,1+N,C,D]`。对最后的 $D$ 个通道求和，本来就等价于点积。
3. **减去跨类别公共成分。** `features.mean(2)` 在类别维求平均，作者把这部分视为不同类别共有的冗余响应；减掉它可以强化类别之间的差异。
4. **沿空间token做min-max归一化。** 对每个类别，把所有位置的最小响应变成0、最大响应变成1，得到便于生成CAM的范围。
5. **主模型再去掉CLS和背景类别。** `[:, 1:, :self.num_classes-1]` 最终留下 `[B,N,C_fg]` 的前景patch CAM。

所以论文里的 $\cos(P,T)$ 在实际代码中被扩展成了“全局类别加权 + patch与文本逐通道匹配 + 去公共响应 + 空间归一化”。这比普通的 `normalize(P) @ normalize(T).T` 多了一个类别去冗余过程。

### 5.4 低分辨率相似度图怎样恢复空间？

仓库中的 [`get_similarity_map`](https://github.com/zwyang6/DiCLIP/blob/1c3f6ff7d4fde2afff32d527d78b28d119583602/clip/clip.py#L272-L285) 给出了完整实现：

```python
side = int(sm.shape[1] ** 0.5)
sm = sm.reshape(sm.shape[0], side, side, -1)
sm = sm.permute(0, 3, 1, 2)             # [B, C, H', W']
sm = F.interpolate(sm, shape, mode="bilinear")
sm = sm.permute(0, 2, 3, 1)             # [B, H, W, C]
```

`side = sqrt(N)` 隐含假设patch网格是正方形。如果输入图像或特征网格不是正方形，只保存 $N$ 就不能唯一恢复 $H'$ 和 $W'$，工程上应直接传入二者。这里使用双线性插值，没有转置卷积，也没有跳跃连接；它只是把CAM平滑放大，不会凭空恢复被patch化丢失的真实边缘。

### 5.5 视觉键值缓存怎样离线建立？

缓存构建位于 [`generate_kv_cache.py`](https://github.com/zwyang6/DiCLIP/blob/1c3f6ff7d4fde2afff32d527d78b28d119583602/maintain_kv_cache/both_fore_bkg/generate_kv_cache.py#L89-L160)。关键流程是：

```python
features = model(inputs, return_feats=True)     # [B, 1+N, D]
masks_binary = masks > 0                       # [B, N, 1]

foreground = features[:, 1:, :] * masks_binary
foreground_key = foreground.sum(1) / masks_binary.sum(1)

background = features[:, 1:, :] * (~masks_binary)
background_key = background.sum(1) / (~masks_binary).sum(1)
```

这就是掩码平均池化：先去掉CLS token，再分别用前景和背景掩码筛选patch，最后除以有效patch数量。每张生成图像因此产生一个前景视觉向量和一个背景视觉向量。

同一类别会生成许多向量，代码随后使用K-means压缩：

```python
kmeans = KMeans(n_clusters=num_cluster, random_state=0).fit(class_keys)
cluster_embedding = kmeans.cluster_centers_

cache_keys = torch.cat(cluster_keys)       # [U, D]
cache_values = torch.cat(cluster_values)   # [U, C]
```

聚类中心成为缓存键 $K$；类别one-hot成为缓存值 $V$。这里前景和背景分开处理，并给背景使用更多聚类中心，以覆盖更复杂的背景外观。

### 5.6 静态检索怎样把视觉相似度变成类别CAM？

[`diff_knowledge_inject_cam`](https://github.com/zwyang6/DiCLIP/blob/1c3f6ff7d4fde2afff32d527d78b28d119583602/model/model_diclip.py#L114-L146) 的参数名 `text_features` 容易让人误解；它实际接收的是缓存中的视觉键：

```python
cache_keys, cache_values = kv_cache

# 先得到每个patch与每个缓存键的相似度
similarity = patch_key_similarity(image_features, cache_keys)  # [B, 1+N, U]

# 再把U个缓存响应转换成C个类别响应
similarity = similarity[:, 1:, :] @ values.unsqueeze(0)       # [B, N, C]

foreground_maps = diff_maps[:, :, 1:]
background_maps = diff_maps[:, :, 0].unsqueeze(-1)
fused_maps = foreground_maps * (1 - background_maps)
```

第一步回答“当前patch像哪个视觉原型”；第二步通过值矩阵回答“这个视觉原型对应什么类别”。代码最后显式取出背景通道，并用 $1-\text{background}$ 抑制前景：一个位置越像背景，它的所有前景类别响应就会被压得越低。

值得注意的是，代码没有直接使用加载进来的 `cache_values` 完成最终映射，而是又比较缓存键和完整文本类别嵌入，重新构造并softmax归一化 `values`。因此，代码中的值不是纯粹固定的one-hot标签，而是进一步加入了视觉键与类别文本之间的相似性权重。

### 5.7 动态检索为什么等价于两层MLP？

[`KV_Adapter`](https://github.com/zwyang6/DiCLIP/blob/1c3f6ff7d4fde2afff32d527d78b28d119583602/model/model_diclip.py#L18-L42) 包含两个线性层：

```python
self.fc1 = nn.Linear(in_features, hidden_features)
self.fc1.weight[:num_keys] = cache_key

self.fc2 = nn.Linear(hidden_features, out_features)
self.fc2.weight = nn.Parameter(value_prompt.t())
```

输入patch为 $[B,N,D]$：

```text
[B, N, D]
→ fc1（由缓存键初始化）
→ [B, N, U']
→ ReLU + Dropout
→ fc2（由缓存值初始化）
→ [B, N, C]
```

第一层的每个神经元可以理解成一个视觉键，它检测patch和某个缓存原型的匹配程度；第二层把这些原型响应汇总成类别分数。与静态检索不同，这两个矩阵随后参与反向传播，可以逐渐适应真实训练图像。`adapter_size` 大于真实缓存键数量时，剩余行由截断正态分布初始化，相当于额外加入可学习提示槽位。

最后：

```python
dynamic_maps_pred = dynamic_maps.permute(0, 2, 1).reshape(
    B, C, H_patch, W_patch
)
```

这正是前面解释过的 `[B,N,C] → [B,C,H',W']`。如果训练损失使用原图大小的伪标签，还需要在损失计算前继续插值到匹配尺寸。

### 5.8 论文公式和代码之间值得注意的地方

| 观察 | 代码中的实际情况 | 阅读或修改时的影响 |
|---|---|---|
| 相似度 | 基础CAM使用feature surgery，不是单纯的余弦矩阵乘法 | 复现论文时只写 `P @ T.T` 可能得不到相同结果 |
| 归一化维度 | `generate_clip_fts` 对 `image_features.norm(dim=1)` 归一化，而常见逐patch归一化写法是 `dim=-1` | 这里需要结合上游特征布局核对，不能只凭注释认定为标准余弦相似度 |
| min-max稳定性 | 多处 `(x-min)/(max-min)` 没有显式加入epsilon | 若某类所有patch响应相同，理论上可能产生除零，需要考虑 `clamp_min` |
| 背景处理 | 代码使用 `foreground × (1-background)` | 阅读论文中的正负缓存公式时，要先确认负分支表示“背景概率”还是“非背景置信度” |
| 适配器激活 | 构造函数接收 `act_layer=nn.GELU`，但forward实际调用 `F.relu` | 修改激活函数时只改构造参数并不会生效 |
| 梯度边界 | 返回静态增强CAM时使用 `diff_maps.detach()`，动态适配器和分割头则在参数组中训练 | 静态CAM是监督来源，动态分支才是被优化的学生 |

> [!note] 从这份代码中真正应该学什么？
> 不需要记住所有文件。最值得保留的是一种模块化思路：先让patch直接与文本产生一个低成本基线；当单个文本向量覆盖不了视觉变化时，再建立“视觉键—类别值”缓存；固定缓存可以直接检索，也可以被改写成两层线性层的初始化，从而自然过渡到可学习适配器。公式、静态检索和MLP在这里不是三套无关方法，而是同一组矩阵运算的三种解释。

## 6. 怎样选择？

| 当前问题 | 优先检查或尝试 |
|---|---|
| 只有全局图文分数，需要快速获得初始CAM | 全局匹配 + Grad-CAM |
| CLIP patch已经可用，希望直接生成密集响应 | 补丁—文本相似度 |
| 逐patch预测破碎，但已有较可靠区域提议 | 区域—文本对齐 |
| 非目标前景经常被激活为目标类别 | 投影层 + 跨模态原型对齐 |
| 想使用DINO等纯视觉模型的空间特征 | 学习文本到视觉空间的映射 |
| 类别名称不足以覆盖多种视觉外观 | 多原型或检索式对齐 |
| 类别判断正确但边界和物体完整性差 | 先检查视觉相关性与空间细化，而不是继续堆文本提示 |

可以用四个问题快速判断一个新方法的真实创新点：

1. 它把什么当作视觉单元：整图、patch、区域还是原型？
2. 视觉和文本是否天然共享空间；如果不是，桥接方式是什么？
3. 相似度只是生成结果，还是还参与训练损失？
4. 最终提升来自对齐本身，还是来自对齐前后的视觉校准与空间传播？

## 7. 当前论文证据索引

### 弱监督语义分割

- [[DiCLIP_paper_notes]]：补丁—文本CAM、视觉键值缓存、静态与动态密集检索。
- [[SSR_paper_notes]]：视觉/文本投影、跨模态原型和原型对比学习。
- [[ExCEL_paper_notes]]：文本语义增强与视觉校准共同改善密集补丁—文本匹配。
- [[ComCD_paper_notes]]：先利用视觉亲和力聚合patch，再与类别文本计算密集相似度。
- [[WeCLIP_paper_notes]]：冻结CLIP生成初始CAM，并把语义种子与空间细化解耦。

### 开放词汇语义分割

- [[OpenSeg_paper_notes]]：类别无关掩码、区域—词语对齐及区域分数回投像素。
- [[Talk2DINO_paper_notes]]：学习CLIP文本到DINOv2视觉空间的非线性映射。
- [[CorrCLIP_paper_notes]]：在文本分类前重建CLIP patch相关性，说明视觉特征质量是密集对齐的前提。
- [[Trident_paper_notes]]：CLIP提供开放类别语义，DINO/SAM补充局部与全局空间关系。
- [[ReME_paper_notes]]：通过参考片段和标签关系进行相似度检索，为检索式语义迁移提供另一种视角。

## 8. DiCLIP源码定位索引

| 想重新查看的问题 | 文件与函数 |
|---|---|
| 文本模板怎样编码和归一化 | [`clip/clip.py::encode_text_with_prompt_ensemble`](https://github.com/zwyang6/DiCLIP/blob/1c3f6ff7d4fde2afff32d527d78b28d119583602/clip/clip.py#L252-L269) |
| patch—text基础CAM怎样计算 | [`clip/clip.py::clip_feature_surgery`](https://github.com/zwyang6/DiCLIP/blob/1c3f6ff7d4fde2afff32d527d78b28d119583602/clip/clip.py#L288-L310) |
| 展平响应怎样恢复并上采样 | [`clip/clip.py::get_similarity_map`](https://github.com/zwyang6/DiCLIP/blob/1c3f6ff7d4fde2afff32d527d78b28d119583602/clip/clip.py#L272-L285) |
| 前景/背景视觉键怎样构建 | [`maintain_kv_cache/.../generate_kv_cache.py`](https://github.com/zwyang6/DiCLIP/blob/1c3f6ff7d4fde2afff32d527d78b28d119583602/maintain_kv_cache/both_fore_bkg/generate_kv_cache.py#L89-L160) |
| 静态缓存怎样生成CAM | [`model/model_diclip.py::diff_knowledge_inject_cam`](https://github.com/zwyang6/DiCLIP/blob/1c3f6ff7d4fde2afff32d527d78b28d119583602/model/model_diclip.py#L114-L146) |
| 缓存怎样初始化动态适配器 | [`model/model_diclip.py::KV_Adapter`](https://github.com/zwyang6/DiCLIP/blob/1c3f6ff7d4fde2afff32d527d78b28d119583602/model/model_diclip.py#L18-L42) |
| 三种CAM在主模型中怎样连接 | [`model/model_diclip.py::forward`](https://github.com/zwyang6/DiCLIP/blob/1c3f6ff7d4fde2afff32d527d78b28d119583602/model/model_diclip.py#L148-L185) |

这张表的作用是防止以后重新进入仓库时又从根目录开始找。正文只保留理解算子所需的代码，完整上下文以对应提交中的源码为准。

## 9. 当前整理结论

跨模态对齐的核心选择不是“使用哪一种相似度”，而是决定：

$$
\boxed{\text{用什么视觉粒度，在什么特征空间，与什么语义锚点建立对应}}
$$

从当前弱监督和开放词汇笔记中，可以得到五条较稳定的判断：

1. 全局图文对齐适合类别判断，但不能自然保证密集定位。
2. patch或区域粒度更适合分割，但更依赖视觉空间结构和边界质量。
3. 当视觉空间与文本空间能力互补但不兼容时，可以通过投影、原型或检索建立桥梁。
4. 对齐的输出可能只是推理时的类别分数，也可能被转换成CAM、伪标签或训练损失；阅读论文时要明确它在整条监督链中的位置。
5. 很多方法的提升并不来自新的相似度公式，而来自相似度之前的视觉校准、文本增强，以及相似度之后的空间传播和背景抑制。

因此，面对一个新的跨模态分割方法时，可以先用一句话概括：

> 它选择了什么视觉单元，通过什么桥梁进入哪一个语义空间，最后把对应关系用在了哪里？

只要这四件事能够说清楚，论文中的模块名称、公式和源码实现就能被放回同一条数据流中理解。
