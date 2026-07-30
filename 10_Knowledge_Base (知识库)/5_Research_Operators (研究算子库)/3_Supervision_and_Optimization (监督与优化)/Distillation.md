---
type: operator-note
aliases:
  - Knowledge Distillation
  - Self-Distillation
  - 知识蒸馏
tags:
  - research-operator
  - distillation
  - teacher-student
  - weakly-supervised-segmentation
  - open-vocabulary-segmentation
status: in-progress
---

# Distillation（知识蒸馏）

## 1. 本页定位

本页整理经典模型、弱监督语义分割（Weakly Supervised Semantic Segmentation，**WSSS**）和开放词汇分割（Open-Vocabulary Segmentation，**OVS**）中“一个模型或分支怎样把知识转给另一个模型”的通用操作。这里既包含经典软 logit 蒸馏，也包含特征、关系、硬伪标签和自蒸馏；同时明确哪些方法只是“像蒸馏的教师—学生结构”，并不是标准的同构 logit Kullback–Leibler 散度（Kullback–Leibler divergence，**KL**）蒸馏。

完整方法仍回到 [[dinov2_paper_notes]]、[[S2C_paper_notes]]、[[WeCLIP_paper_notes]]、[[ExCEL_paper_notes]]、[[OpenSeg_paper_notes]] 和 [[DiCLIP_paper_notes]] 阅读。本页的目标是方法手册与源码导航：教师传递什么、学生在哪里接收、哪一侧有梯度、教师是否参与最终部署。

> [!abstract] 一句话直觉
> 教师不一定直接告诉学生“答案是第几类”。它还可以告诉学生类别之间的软关系、哪些像素属于同一区域、哪些 token 应该相似，或哪些位置值得信任。蒸馏的关键是信息流，而不是是否出现 `teacher` 这个变量名。

## 2. 这个算子解决什么问题

### 2.1 大白话解释

大型基础模型常有学生缺少的能力：

- CLIP 有跨模态类别语义，但密集空间定位不一定精确；
- 分割一切模型（Segment Anything Model，**SAM**）有区域和边界先验，但没有目标数据集类别；
- DINOv2 有稳定的 patch 空间结构，但没有天然文本接口；
- 扩散模型有空间亲和力和生成外观先验，但训练/推理很重；
- 大型视觉 Transformer 教师表现强，但不适合低成本部署。

蒸馏在训练时借用这些能力，把结果写进较轻学生的参数；理想状态下最终只部署学生。

### 2.2 专业表述

给定教师 $f_t$、学生 $f_s$ 和同一输入 $x$，选择知识表征 $K_t(x)$ 与 $K_s(x)$，再最小化距离：

$$
\mathcal L_{distill}=d\bigl(\operatorname{sg}(K_t(x)),K_s(x)\bigr).
$$

**公式解释：** $f_t,f_s$ 分别是教师和学生，$K_t(x),K_s(x)$ 是它们从同一输入 $x$ 提取的待匹配知识；这些知识可能是向量、空间特征图或关系矩阵，但在送入距离函数 $d$ 前必须具有可比较的 shape。$d$ 对知识张量的全部待比较维进行归约，输出标量损失 $\mathcal L_{distill}$；$\operatorname{sg}$ 表示 stop-gradient，使教师侧不接收反向梯度，梯度只更新学生及允许训练的投影/解码器。

### 2.3 哪些相似结构不一定叫蒸馏

- 冻结主干只提供输入特征，学生没有显式模仿目标：更接近“冻结特征迁移”或 adapter 学习。
- 当前学生自己生成伪标签再学习：同时属于自训练；若有不同视图、时间平均或结构变换，也可视为自蒸馏。
- 仅从预训练权重初始化后正常微调：这是迁移学习，不是持续的教师监督。
- 教师和学生都被同一真值独立监督，没有互相匹配：不是蒸馏。

> [!note] 我的理解｜用信息流来命名最稳妥
> 先写“谁产生目标 → 目标是什么 → 是否离散化 → 是否停止梯度 → 谁被更新”，再决定叫软蒸馏、硬伪标签、自训练还是关系迁移。像 S2C 这类方法可同时属于多个范畴，不必强行只贴一个标签。

## 3. 统一输入输出张量

### 3.1 像素级软 logit 蒸馏

教师和学生输出：

$$
Z_t,Z_s\in\mathbb R^{B\times C\times H\times W}.
$$

**公式解释：** $Z_t,Z_s$ 分别是教师和学生的像素级未归一化 logit，shape 都为 `[B,C,H,W]`；$B,C,H,W$ 分别表示批量、共同类别数和共同空间网格。该式只声明输入，没有发生维度运算；$Z[b,c,h,w]$ 是位置 $(h,w)$ 属于类别 $c$ 的原始分数。

- $B$：batch 大小；
- $C$：共同类别数；
- $H,W$：共同输出网格；
- $Z[b,c,h,w]$：像素 $(h,w)$ 对类别 $c$ 的未归一化分数。

温度概率和蒸馏损失为：

$$
p_t=\operatorname{softmax}_{c}(Z_t/T),\qquad
p_s=\operatorname{softmax}_{c}(Z_s/T),
$$

**公式解释：** 教师和学生每个像素的 $C$ 类 logit 先除以标量温度 $T$，再只沿类别维做 softmax；分母对全部 $C$ 类指数分数求和，但输出仍保留类别维，`p_t,p_s=[B,C,H,W]`。$p_t[b,c,h,w]$ 和 $p_s[b,c,h,w]$ 分别表示教师、学生在该像素分给类别 $c$ 的温度概率。

$$
\mathcal L_{KD}=
\frac{T^2}{BHW}\sum_{b,h,w}
\operatorname{KL}\left(p_{t,b,:,h,w}\|p_{s,b,:,h,w}\right).
$$

**公式解释：** KL 在每个 $(b,h,w)$ 上比较教师和学生两个长度为 $C$ 的概率分布，内部沿类别维求和并消去 $C$，得到 `[B,H,W]` 逐像素损失；外层再沿 $b,h,w$ 求和并除以 $BHW$，消去批量和空间维，最终输出标量 $\mathcal L_{KD}$。$T^2$ 用来补偿高温度导致的梯度缩小。

数字例子：`Z_t=Z_s=[2,21,64,64]`，softmax 后仍为 `[2,21,64,64]`；类别维 KL 后是 `[2,64,64]`，最终归约为一个标量。

> [!note] KL 散度｜它究竟在惩罚什么？
>
> 把教师分布 $P$ 看成“教师认为各类别有多像”，学生分布 $Q$ 看成“学生的同一份判断”。KL 散度 $D_{\mathrm{KL}}(P\Vert Q)=\sum_iP_i\log(P_i/Q_i)$ 会重点惩罚这样的错误：**教师很确信某一类（$P_i$ 大），学生却几乎不给该类概率（$Q_i$ 很小）**。因此它保留了第一名以外的类别相对关系，常被称为软目标或“暗知识”。
>
> KL 不是对称距离：$D_{\mathrm{KL}}(P\Vert Q)\neq D_{\mathrm{KL}}(Q\Vert P)$。蒸馏通常写成“教师 $\Vert$ 学生”，以明确学生要覆盖教师的高置信判断。固定教师 $P$ 时，$D_{\mathrm{KL}}(P\Vert Q)=H(P,Q)-H(P)$；其中 $H(P)$ 不随学生改变，所以最小化 KL 等价于用教师的**软概率**作为目标最小化交叉熵。温度 $T>1$ 让分布更平缓、露出次要类别的相对信息；外面的 $T^2$ 则补偿升温后缩小的梯度。
>
> 具体例子见 [[DiG_paper_notes|DiG 论文笔记]]：ViT 的类别预测是参考分布，LFCA 的扩散分支向它做 KL 对齐；这与图像级标签提供的硬分类监督互补。

### 3.2 分辨率不一致

若教师输出 `[B,C,32,32]`、学生输出 `[B,C,64,64]`，可先对教师连续 logit 或概率做双线性插值：

```text
teacher logits [B,C,32,32]
→ bilinear interpolate(size=(64,64), align_corners=False)
→ [B,C,64,64]
→ softmax / KL with student [B,C,64,64]
```

离散硬标签应使用最近邻插值，避免类别编号被平均。转置卷积有可学习参数，适合学生解码器内部恢复空间，不适合作为教师标签的无参数对齐。基础见 [[downsampling_and_upsampling(下采样与上采样)]]。

### 3.3 特征蒸馏

教师和学生 token 为：

$$
F_t\in\mathbb R^{B\times N_t\times D_t},\qquad
F_s\in\mathbb R^{B\times N_s\times D_s}.
$$

**公式解释：** $F_t$ 与 $F_s$ 分别保存教师和学生 token；$N_t,N_s$ 是各自 token 数，$D_t,D_s$ 是各自特征维。该式只声明输入，尚未消去任何维度；$F_t[b,n,:]$ 是教师第 $n$ 个位置的 $D_t$ 维表示。若 $N$ 或 $D$ 不同，两者不能直接逐元素相减，必须先做空间与通道对齐。

通道不同需学习投影 $g:\mathbb R^{D_s}\to\mathbb R^{D_t}$；token 数不同需先按真实二维网格恢复和插值，不能直接截断：

```text
student [B,N_s,D_s]
→ reshape [B,H_s',W_s',D_s]
→ permute [B,D_s,H_s',W_s']
→ interpolate to [B,D_s,H_t',W_t']
→ projection 1×1 conv to [B,D_t,H_t',W_t']
→ flatten/permute [B,N_t,D_t]
```

余弦特征损失可写：

$$
\mathcal L_{feat}=\frac1{BN_t}
\sum_{b,n}\left(1-\cos(g(f^s_{b,n}),\operatorname{sg}(f^t_{b,n}))\right).
$$

**公式解释：** 学生 token $f^s_{b,n}\in\mathbb R^{D_s}$ 先经投影 $g$ 变为 $D_t$ 维，再与教师 token $f^t_{b,n}\in\mathbb R^{D_t}$ 计算余弦相似度；余弦运算沿特征维 $D_t$ 点乘和求范数，因此消去特征维，为每个 $(b,n)$ 输出一个标量。$1-\cos$ 把相似度变成距离，外层再沿 batch $b$ 和 token $n$ 求和并平均，消去这两个维度，最终得到标量 $\mathcal L_{feat}$；教师 token 被 stop-gradient。

### 3.4 关系蒸馏

若绝对特征空间差异太大，可匹配 token 两两关系：

$$
A_t=\hat F_t\hat F_t^T,\qquad
A_s=\hat F_s\hat F_s^T,\qquad
A_t,A_s\in\mathbb R^{B\times N\times N},
$$

**公式解释：** 对教师和学生分别用 `F_hat=[B,N,D]` 与其转置 `[B,D,N]` 做批量矩阵乘法，共同的特征维 $D$ 被消去，得到 `A_t,A_s=[B,N,N]`；$A[b,i,j]$ 是同一图中 token $i$ 与 $j$ 的相似度。两个 token 维均被保留，所以关系矩阵能够描述全部位置对。

$$
\mathcal L_{rel}=\|A_s-\operatorname{sg}(A_t)\|_F^2.
$$

**公式解释：** 该式先逐元素比较两张 `[B,N,N]` 关系矩阵，教师矩阵由 $\operatorname{sg}$ 停止梯度；Frobenius 范数再对 batch 和两个 token 维的误差平方求和，把全部维度归约成标量 $\mathcal L_{rel}$。学生关系接收梯度，教师关系不接收；显式关系矩阵的计算与存储复杂度是 $O(N^2)$。

### 3.5 硬伪标签蒸馏

教师概率离散化为：

$$
\tilde Y_{b,h,w}=\arg\max_c p_{t,b,c,h,w},
\qquad
\mathcal L_{hard}=\operatorname{CE}(Z_s,\tilde Y).
$$

**公式解释：** 第一式对教师概率 `p_t=[B,C,H,W]` 的类别维 $C$ 取最大值索引，消去类别维，得到硬标签 `Y_tilde=[B,H,W]`；$\tilde Y[b,h,w]$ 是教师在该像素选择的类别编号。第二式让学生 `Z_s=[B,C,H,W]` 在每个像素的 $C$ 类中预测该编号，交叉熵先消去类别维，再按实现对 batch 和空间维平均，输出标量 $\mathcal L_{hard}$。离散化会丢失教师对其他类别的相对偏好。

## 4. 常见蒸馏形式归纳

| 形式 | 教师输出 | 学生匹配位置 | 优点 | 局限 |
|---|---|---|---|---|
| 软 logit 蒸馏 | `[B,C,H,W]` 概率分布 | 分割 logit | 保留类间软关系 | 类别集合和输出网格必须可比 |
| 硬标签蒸馏 | `[B,H,W]` 类别索引 | 最终预测 | 简单，可直接复用 CE | 丢失暗知识，错误类别被当真值 |
| 特征蒸馏 | `[B,N,D_t]` 或特征图 | 中间层/adapter | 可传递细粒度表示 | 需投影与空间对应 |
| 关系蒸馏 | `[B,N,N]` 亲和矩阵 | 注意力或特征关系 | 不要求绝对特征同空间 | 二次复杂度，关系也会含噪声 |
| 区域/结构蒸馏 | 区域索引、边界、掩码 | 特征或伪标签分支 | 教师不需要目标类别词表 | 区域无语义，需学生补类别 |
| 自蒸馏 | EMA 教师、历史学生或不同视图 | 同一模型另一分支 | 不需外部教师 | 容易确认偏差，需要防塌缩 |

## 5. 代表论文逐篇说明

| 论文 | 教师/知识来源 | 学生或接受者 | 具体传递方式 | 与本算子的关系 |
|---|---|---|---|---|
| [[dinov2_paper_notes]] | 训练中的指数移动平均（Exponential Moving Average，**EMA**）教师，以及蒸馏小模型时冻结的 ViT-g 教师 | 学生 ViT 与更小 ViT 架构 | 图像级用不同裁剪的 `[CLS]` 原型分布做交叉熵；patch 级用 iBOT 对被遮挡学生 token 和教师对应可见 token 的分布做匹配；小模型蒸馏复用训练循环，并评估学生 EMA | 经典自蒸馏与大模型到小模型蒸馏参考；教师概率还需中心化/温度处理以防塌缩 |
| [[S2C_paper_notes]] | 冻结 SAM 的区域分组与点提示掩码 | 主 CAM 网络的特征和类别 CAM | 一条路径用 SAM 区域索引构造区域原型对比（SSC）；另一条路径用多尺度 CAM 点提示 SAM，所得类别伪掩码通过 CAM-based Prompting Module（**CPM**）交叉熵监督主网络 | 不是同构 soft-logit KL；教师传的是“区域关系 + 硬掩码”，并同时带有自训练闭环 |
| [[WeCLIP_paper_notes]] | 冻结 CLIP 的多层特征、注意力与静态初始 CAM | 轻量解码器和冻结 CLIP CAM 细化模块（Refinement Module，**RFM**） | 解码器读取冻结 CLIP 多层特征；RFM 将冻结注意力与解码器动态亲和力结合，细化静态 CAM 为在线伪标签，再反过来监督解码器预测与亲和力 | 是冻结教师先验与学生动态关系共同生成目标，教师不更新；比单向蒸馏更接近相互促进的在线伪标签学习 |
| [[ExCEL_paper_notes]] | 静态视觉校准生成的固定 CAM/伪标签，以及冻结 CLIP 多层特征 | 可学习视觉校准（Learnable Visual Calibration，**LVC**）adapter 与分割头 | 静态伪标签把像素对是否同类转为关系目标，监督 adapter 动态特征的自相似矩阵；动态 CAM 再生成伪标签训练分割头 | 传递的主要是关系而非同构 logit；“静态分支是教师”是机制解释，论文主体也可视为关系正则与伪标签训练 |
| [[OpenSeg_paper_notes]] | 先在有完整分割标注的 COCO 上训练的分割教师 | 在大规模图像—标题数据上训练的 OpenSeg | 教师给 Localized Narratives 图像生成类别无关伪分割掩码；OpenSeg 混合人工掩码和伪掩码，联合区域分割损失与区域—词语接地损失 | 典型离线硬标签蒸馏/伪标注：教师扩展数据规模，学生再学习开放词汇区域语义 |
| [[DiCLIP_paper_notes]] | 冻结扩散模型的空间亲和力与生成单类图像缓存 | CLIP 注意力、动态 adapter 和分割头 | 视觉相关性增强把扩散亲和力作为偏置注入 CLIP 后层注意力；文本语义增强让真实 patch 查询前景/背景视觉键值缓存，静态结果再初始化或监督动态分支 | 更接近外部先验注入与缓存迁移，不是传统教师输出匹配；仍可按“空间关系知识 + 视觉外观知识”两条传递链分析 |
| [[CLIP-ES_paper_notes]] | 冻结 CLIP 产生并细化的伪掩码 | 独立最终分割模型 | softmax-Grad-CAM、类别相关背景提示和类感知注意力亲和力生成伪掩码；置信度引导损失忽略低可靠位置 | 属于硬伪标签式知识迁移；低置信位置不参与学生梯度，详见 [[Confidence_Reweighting]] |

> [!note] 我的理解｜“冻结基础模型”不自动等于知识蒸馏
> WeCLIP、ExCEL 和 DiCLIP 中，冻结模型有时只是特征源，有时产生明确目标，有时提供关系偏置。应分别记录。若学生只消费冻结特征但没有匹配目标，就是特征迁移；若静态分支的输出约束可学习分支，才具有明确蒸馏含义。

## 6. 各种实现怎样工作

### 6.1 软 logit 蒸馏

**直觉**：教师不只给第一名，还告诉学生“猫比狗更像，和飞机差得很远”。

**数据流**：教师/学生 logit → 同一温度 softmax → 教师 stop-gradient → 逐像素 KL → 与真值/伪标签损失加权求和。

**适用场景**：教师和学生具有相同类别词表，且教师输出经过合理校准。

**局限**：OVS 中词表可动态变化；只在 seen 类封闭 softmax 上蒸馏，可能把教师连续文本空间压成封闭分类器，损害未见类迁移。

### 6.2 特征蒸馏

**直觉**：不要求学生最后答案逐项相同，而是让中间表示学会教师看图的方式。

**数据流**：选择对应层 → 去掉/保留 `[CLS]` 等特殊 token → 恢复二维网格 → 空间插值 → 通道投影 → 归一化 → L1/L2/余弦损失。

**适用场景**：学生类别头不同，但希望继承通用视觉表示或局部结构。

**局限**：层号相同不代表语义深度相同；小学生容量不足时，强迫匹配全部中间层会和主任务冲突。

### 6.3 关系蒸馏

**直觉**：不要求两种语言的向量坐标相同，只要求它们都认为“狗头和狗身相近，狗和天空较远”。

**数据流**：教师/学生 token 归一化 → 自相似或注意力矩阵 `[B,N,N]` → 采样/掩码 → 矩阵距离。

**适用场景**：教师和学生特征维不同，或 SAM/扩散模型主要提供空间结构。

**局限**：关系矩阵中的错误边会传播到许多 token；`N=H'W'` 大时显存迅速增长。

### 6.4 硬伪标签与自训练

**直觉**：教师先批改出一份答案，学生按普通监督学习；答案不确定处可以 ignore。

**数据流**：教师预测 → 多尺度/后处理 → argmax/阈值 → `[B,H,W]` 硬标签 → 学生交叉熵。

**适用场景**：教师和学生结构差异很大，无法直接对齐内部特征；希望缓存标签降低训练成本。

**局限**：离线伪标签不会随学生改进；在线自训练会形成确认偏差。应记录标签是否每轮更新、是否使用 `no_grad`、背景和 ignore 的编号。

## 7. 冻结教师、EMA 教师与同伴分支

EMA 教师更新为：

$$
\theta_t\leftarrow\mu\theta_t+(1-\mu)\theta_s.
$$

**公式解释：** $\theta_t$ 和 $\theta_s$ 是形状逐项对应的教师、学生参数集合，标量 $\mu\in[0,1)$ 是动量系数。公式对每个参数元素做加权和，没有矩阵乘法或维度消去，更新后的教师参数 shape 完全不变；单个元素是“旧教师值的 $\mu$ 倍 + 当前学生值的 $1-\mu$ 倍”。这不是损失，也不经过反向传播，通常在优化器 step 后用 `no_grad` 执行。

| 教师形式 | 参数来源 | 是否随训练变化 | 主要价值 | 主要风险 |
|---|---|---:|---|---|
| 冻结基础模型 | 外部预训练 CLIP/SAM/DINOv2/扩散模型 | 否 | 固定外部先验，不随学生漂移 | 目标域偏差长期不变 |
| EMA 教师 | 学生历史参数的平滑 | 是 | 目标更稳定，适合自蒸馏 | 学生错误会进入教师 |
| 在线同伴分支 | 同一网络另一头/视图 | 是 | 低额外存储，可互补 | 两分支可能一起塌缩 |
| 离线伪标签教师 | 预计算结果 | 否 | 训练快，可去掉教师前向 | 无法在线纠错，存储大 |

[[dinov2_paper_notes]] 同时提供两种参考：预训练阶段的 EMA 自蒸馏，以及用冻结 ViT-g 蒸馏较小模型。[[S2C_paper_notes]] 则使用冻结 SAM，但主网络产生 CAM 点提示，形成“固定结构教师 + 当前学生语义”的混合目标。

## 8. 官方仓库静态分析：S2C 怎样把 SAM 知识转给 CAM 网络

- 官方仓库：[sangrockEG/S2C](https://github.com/sangrockEG/S2C)
- 阅读 commit：[`102e14c690c8e3bce3d5ccd1ae7832145ce10b27`](https://github.com/sangrockEG/S2C/tree/102e14c690c8e3bce3d5ccd1ae7832145ce10b27)
- 关键文件：[`models/model_s2c.py`](https://github.com/sangrockEG/S2C/blob/102e14c690c8e3bce3d5ccd1ae7832145ce10b27/models/model_s2c.py)
- 训练入口：`update(self, epo, iter)`

### 8.1 总调用链

```text
冻结 SAM 提供预生成区域图 self.se
→ SSC：区域图分组主网络特征，形成区域原型对比目标

当前主网络在四个尺度产生 CAM（eval + no_grad）
→ CAM 局部峰值作为 SAM 正点提示
→ 冻结 SAM encoder/decoder 产生候选掩码与置信度
→ 与 CAM 区域均值融合为硬伪标签 pgt_sam
→ 主网络切回 train 再前向
→ CPM：主 CAM + 背景通道与 pgt_sam 做交叉熵
→ 分类损失 + SSC + CPM 一起 backward，只更新主网络
```

这条链中 SAM 传递两种知识：`self.se` 表示“哪些像素属于同一片段”，用于 SSC 特征/关系约束；点提示掩码表示“这个类别的候选区域”，用于 CPM 硬标签监督。

### 8.2 教师冻结与延迟启用

[`L189-L230`](https://github.com/sangrockEG/S2C/blob/102e14c690c8e3bce3d5ccd1ae7832145ce10b27/models/model_s2c.py#L189-L230) 的关键代码：

```python
self.net_sam.eval()
use_cpm = epo > self.args.sstart - 1

if use_cpm:
    with torch.no_grad():
        self.net_main.eval()
        # 0.5 / 1.0 / 1.5 / 2.0 multi-scale CAM
        ...

    with torch.no_grad():
        features_sam = self.net_sam(run_encoder_only=True, ...)
```

- `self.net_sam.eval()` 固定 dropout、归一化等训练行为；真正阻断梯度的是后续 `torch.no_grad()`。
- `use_cpm = epo > sstart - 1`。若 epoch 是整数，它等价于 `epo >= sstart`：CPM 延迟到指定 epoch 才启用，避免训练早期错误 CAM 立即成为教师提示。
- 主网络先切到 `eval()` 在无梯度状态产生多尺度 CAM；这份 CAM 来自当前学生参数，却作为当前 step 的固定目标构造材料，因此同时具有自训练性质。
- SAM encoder 在 `no_grad` 中提取并缓存图像特征，随后同一块 `no_grad` 还包住点采样和 decoder 候选生成；SAM 不接收任何优化梯度。

### 8.3 从 CAM 点提示到 SAM 硬标签

[`L233-L299`](https://github.com/sangrockEG/S2C/blob/102e14c690c8e3bce3d5ccd1ae7832145ce10b27/models/model_s2c.py#L233-L299) 先从每个已知类别 CAM 采样全局最大点和局部峰值，再调用 SAM decoder：

```python
mask = output_sam[0]  # (1,3,H,W)
conf = output_sam[2]  # (1,3,H,W)
idx_max_sam = 2
target_mask = mask[0,idx_max_sam]
target_conf = conf[0,idx_max_sam]
sam_conf[i,k][target_mask] = (
    target_conf[target_mask] * cam_ms[i,k][target_mask].mean()
)
temp = sam_conf.max(dim=1)
pgt_sam = temp[1]
pgt_sam[temp[0] < 0] = 20
```

这里不是软 logit 蒸馏：

1. SAM 返回 3 个候选，但代码固定 `idx_max_sam=2`，并非动态比较候选后取最大；
2. `target_conf` 保留掩码内逐像素差异，CAM 在掩码内先求均值成为类别语义标量；
3. `sam_conf=[B,C,H,W]` 在类别维 `max(dim=1)`，消去 $C$，`indices` 得到硬标签 `[B,H,W]`；
4. 未覆盖位置设为 VOC 背景索引 20，不是 255 ignore；
5. 这些步骤全部无梯度，学生不能通过操纵提示链直接减小本 step 的损失。

### 8.4 SSC：区域关系知识进入学生特征

[`L327-L343`](https://github.com/sangrockEG/S2C/blob/102e14c690c8e3bce3d5ccd1ae7832145ce10b27/models/model_s2c.py#L327-L343) 为：

```python
feat_main = F.interpolate(feat_main, size=(H,W),
                          mode='bilinear', align_corners=False)
feat_main = F.normalize(feat_main, dim=1)
feat_main_ = feat_main.view(B,D,-1)             # [B,D,HW]
index_ = self.se.view(B,1,-1).long()            # [B,1,HW]
pt = torch_scatter.scatter_mean(feat_main_.detach(), index_)
pt = F.normalize(pt, dim=1)                     # [B,D,Nseg]
pred_ssc = torch.bmm(pt.permute(0,2,1), feat_main_)
self.loss_ssc = F.cross_entropy(pred_ssc*self.T,
                                index_.squeeze(1), ignore_index=0)
```

`scatter_mean` 按 SAM 区域索引把 `[B,D,HW]` 聚合为 `[B,D,Nseg]` 区域原型；转成 `[B,Nseg,D]` 后与像素特征相乘，消去 $D$，得到 `[B,Nseg,HW]` 区域分类 logit。原型由 `feat_main_.detach()` 构造，原型分支停止梯度；相似度中的像素特征未 detach，仍由 SSC 更新。`ignore_index=0` 忽略区域图中 0 号位置。

这条路径不是让学生模仿 SAM embedding。SAM 只给区域分组，原型本身来自学生当前特征；更准确地说，它是“结构教师定义关系，学生在自己的空间满足关系”。详见 [[Contrastive_Regularization]]。

### 8.5 CPM：硬掩码怎样真正更新主网络

主网络在 [`L304-L317`](https://github.com/sangrockEG/S2C/blob/102e14c690c8e3bce3d5ccd1ae7832145ce10b27/models/model_s2c.py#L304-L317) 切回 `train()` 并重新前向，这一次计算图被保留。CPM 位于 [`L345-L360`](https://github.com/sangrockEG/S2C/blob/102e14c690c8e3bce3d5ccd1ae7832145ce10b27/models/model_s2c.py#L345-L360)：

```python
cam_bg = 1 - cam_main.max(dim=1, keepdims=True)[0]
cam_main = torch.cat((cam_main, cam_bg), dim=1)
self.loss_cpm = F.cross_entropy(cam_main, pgt_sam, ignore_index=255)
...
loss.backward()
self.opt_main.step()
```

- `cam_main=[B,C,H,W]`；在类别维取最大值后变 `[B,1,H,W]`，`1-max` 构造背景通道；
- `concat` 沿类别维拼接，得到 `[B,C+1,H,W]`，背景位于最后一个通道；
- `pgt_sam=[B,H,W]` 是前景类别索引或背景 20，交叉熵消去类别维并对空间/batch 汇总；
- 虽然损失设置 `ignore_index=255`，当前伪标签构造对未覆盖位置使用 20，因此这些背景位置仍参与梯度；
- `backward/step` 只更新 `opt_main` 管理的主网络参数，SAM 教师不在优化路径上。

### 8.6 训练成本与部署路径

训练时需要四尺度主网络前向、SAM encoder/decoder、SSC 全分辨率特征和主网络有梯度前向，成本并不等于最终学生的推理成本。固定 commit 的推理函数 [`infer_multi`](https://github.com/sangrockEG/S2C/blob/102e14c690c8e3bce3d5ccd1ae7832145ce10b27/models/model_s2c.py#L367-L422) 只调用 `self.net_main_replicas` 生成多尺度翻转 CAM，没有调用 SAM。这说明教师在最终 CAM 导出路径中被移除；报告效率时应分别写训练成本与部署成本。

### 8.7 论文叙事、公式与代码的边界

- S2C 不是经典“教师与学生同类别 logit 做 KL”；它组合了区域关系约束和硬伪标签 CE。
- CPM 伪标签由**当前主网络 CAM + 冻结 SAM**共同产生，既是知识迁移也是在线自训练。
- SSC 原型来自学生特征且 detach，不是 SAM 特征；SAM 只确定哪些像素一起平均。
- 代码固定取 SAM 第三个候选，论文中若只写“按置信度选择/聚合”，不能自动推断为动态候选 argmax。
- 背景通道是 `1-max(foreground)` 的启发式分数，不是 SAM 单独输出的背景概率。

## 9. 置信掩码蒸馏

教师并非每个像素都可靠，可加入权重：

$$
\mathcal L_{KD}=
\frac{\sum_iw_iT^2\operatorname{KL}(p_{t,i}\|p_{s,i})}
{\sum_iw_i+\varepsilon}.
$$

**公式解释：** 对像素 $i$，$p_{t,i}$ 与 $p_{s,i}$ 都是长度为 $C$ 的类别分布；KL 先沿类别维求和并消去 $C$，得到单像素标量。可靠性 $w_i$ 再逐像素缩放该损失，分子沿像素索引 $i$ 求和，分母用同一批权重归一化并加 $\varepsilon$ 防止除零，因此最终输出标量 $\mathcal L_{KD}$。$w_i=0$ 表示该像素没有蒸馏梯度，权重通常也应停止梯度。

## 10. OVS 蒸馏的特殊问题

OVS 的教师知识不应只剩固定 seen 类概率。更适合保留：

- patch/区域与文本嵌入的余弦相似度结构；
- 区域—词语对应关系；
- 包含额外文本类别的扩展词表分布；
- 类别无关掩码和独立语义分类两条接口。

如果只在训练类 $C_{seen}$ 上做 softmax KL，学生可能学成封闭分类器：seen 类 logit 很像教师，但新增文本没有可用接口。评估应分别报告 seen、unseen 和调和均值，并测试扩大候选词表后是否仍稳定。

## 11. 选型指南

| 当前症状 | 优先考虑 | 不值得或需先检查 |
|---|---|---|
| 大教师与小学生类别完全相同 | 软 logit + 主任务标签 | 若教师过度自信，先做温度和置信校准 |
| 类别头不同但希望迁移局部结构 | 特征或关系蒸馏 | 先确认 token 坐标、特殊 token 和通道投影 |
| SAM 边界好但没有类别 | 区域关系 + 学生语义提示生成硬标签 | 不直接把 SAM 区域编号当数据集类别 |
| WSSS 只有噪声 CAM | 高置信硬标签、EMA/多视图稳定目标 | 不让早期在线伪标签立即强监督，考虑延迟启用 |
| OVS 需要保持未见类 | 区域/patch—文本相似度蒸馏 | 不只蒸馏 seen 类封闭 softmax |
| 教师训练前向太贵 | 离线缓存 logit/特征/伪掩码 | 缓存会失去随机增强对应并增加存储 |
| 学生容量远小于教师 | 选择少数层、关系或分阶段蒸馏 | 不强迫每层和每头逐项完全相同 |
| 目标域与教师预训练域差异大 | 置信筛选、适配器、学生真值损失纠偏 | 不把教师输出当绝对真值 |

最简单基线应包含：只用主任务损失、硬伪标签 CE、软 logit KL，以及相同训练预算下的冻结特征学生。只有当最终部署确实移除教师，才应把“推理更轻”作为蒸馏收益。

## 12. 论文与源码索引

- [[dinov2_paper_notes]]：EMA 自蒸馏、图像级 DINO 目标、patch 级 iBOT 目标和大模型到小模型蒸馏。
- [[S2C_paper_notes]]：SAM 区域关系与点提示硬掩码怎样进入 WSSS 主网络。
- [[WeCLIP_paper_notes]]：冻结 CLIP 静态先验与可学习解码器/RFM 的在线闭环。
- [[ExCEL_paper_notes]]：静态伪标签对动态自相似关系的监督。
- [[OpenSeg_paper_notes]]：分割教师离线标注大规模图像—标题数据。
- [[DiCLIP_paper_notes]]：扩散空间关系和生成视觉缓存的外部知识注入。
- [S2C 教师目标生成](https://github.com/sangrockEG/S2C/blob/102e14c690c8e3bce3d5ccd1ae7832145ce10b27/models/model_s2c.py#L189-L299)：延迟 CPM、多尺度 CAM、SAM encoder/decoder 与硬标签。
- [S2C 学生损失](https://github.com/sangrockEG/S2C/blob/102e14c690c8e3bce3d5ccd1ae7832145ce10b27/models/model_s2c.py#L304-L360)：分类、SSC、CPM 和真实梯度路径。
- [S2C 最终 CAM 推理](https://github.com/sangrockEG/S2C/blob/102e14c690c8e3bce3d5ccd1ae7832145ce10b27/models/model_s2c.py#L362-L422)：确认部署路径只运行主网络。

## 13. 当前整理结论

蒸馏的核心选择是“教师拥有哪一种学生缺少且值得迁移的知识”，而不是先选 KL、L2 还是交叉熵。阅读新论文时应追问教师目标是否软化、是否离散化、是否随训练更新、哪条路径有梯度、教师是否参与推理；修改模型时先检查空间/通道/词表对齐、背景与 ignore 编号、温度、延迟启用和学生容量。不同论文都可放回同一条数据流：教师产生知识表征 → 对齐或离散化 → 停止教师梯度 → 学生在 logit、特征、关系或标签层接收 → 与主任务损失共同更新 → 部署时确认保留哪些分支。
