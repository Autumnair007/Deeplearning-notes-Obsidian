---
type: operator-note
aliases:
  - Contrastive Regularization
  - Contrastive Learning for Segmentation
  - 对比正则
tags:
  - research-operator
  - contrastive-learning
  - prototype
  - weakly-supervised-segmentation
  - open-vocabulary-segmentation
status: in-progress
---

# Contrastive Regularization（对比正则）

## 1. 本页定位

本页整理弱监督语义分割（Weakly Supervised Semantic Segmentation，**WSSS**）和开放词汇分割（Open-Vocabulary Segmentation，**OVS**）中的像素、区域、原型及图文对比约束。它不是对 CLIP 或 DINOv2 整篇预训练方法的总结；基础图文对比入口见 [[clip_paper_notes]]，自蒸馏与 patch 表示见 [[dinov2_notes]]、[[dinov2_paper_notes]]。这里更关心下游分割如何构造可靠正负样本、怎样从 `[B,D,H,W]` 生成对比 logit、伪标签噪声如何进入特征空间，以及代码中的梯度究竟流向哪一侧。

> [!abstract] 一句话直觉
> 像素交叉熵只问“这个位置分成哪一类”；对比正则还问“预测之前的特征空间是否把同类放近、异类分开”。它通常是主任务损失之外的结构约束，不直接替代最终分割头。

## 2. 这个算子解决什么问题

### 2.1 大白话解释

两个像素可能都被分类为“狗”，但它们的中间特征仍相距很远；一个“狗”像素也可能和“猫”像素非常接近。分类头暂时能把它们分开，不代表特征空间稳定。数据域变化、伪标签波动或开放词表扩展后，这些纠缠特征很容易越过决策边界。

对比正则在特征层增加几何要求：

- 同类像素、同一区域像素或匹配图文对是正样本，应靠近；
- 异类原型、其他区域或不匹配文本是负样本，应分开；
- 不可靠伪标签不应轻易进入正样本集合，否则损失会非常有效地“学错”。

### 2.2 专业表述

设查询 $q_i\in\mathbb R^D$、正键 $k_i^+\in\mathbb R^D$、负键集合 $\{k_{i,j}^-\}_{j=1}^{K}$。信息噪声对比估计（Info Noise-Contrastive Estimation，**InfoNCE**）可写为：

$$
\mathcal L_i=-\log
\frac{\exp(\operatorname{sim}(q_i,k_i^+)/\tau)}
{\exp(\operatorname{sim}(q_i,k_i^+)/\tau)+
\sum_{j=1}^{K}\exp(\operatorname{sim}(q_i,k_{i,j}^-)/\tau)}.
$$

**公式解释：** 该式让查询 $q_i$ 在一个正键 $k_i^+$ 和 $K$ 个负键 $k_{i,j}^-$ 中识别正确配对。`sim` 把两个 $D$ 维向量变成标量相似度，$\tau$ 调节 logit 尖锐程度；分母沿候选索引 $j$ 汇总，消去候选维，最终每个查询输出标量 $\mathcal L_i$。分式表示正键在全部候选中的 softmax 概率，负对数越小，说明正对相对负对越相似。

### 2.3 它不负责什么

- **空间恢复**：对比损失不会自动把 token 变回像素图，仍需明确 reshape、permute 和上采样。
- **正负标签发现**：损失只执行给定关系，不能保证伪标签、SAM 区域或文本配对正确。
- **边界细化**：类内紧凑可能让区域更一致，但不等于边界一定更锐利。
- **跨模态尺寸对齐**：视觉维 $D_v$ 与文本维 $D_t$ 不同，必须先投影到公共维度，见 [[Cross_Modal_Alignment]]。

> [!note] 我的理解｜最重要的不是损失名字
> 同一个交叉熵既可以是普通分类，也可以实现原型对比；决定它是不是对比学习的是 logit 如何构造、候选代表什么、target 指向哪一个候选。读源码时先画出 query、key、候选维和 target，而不是只搜索 `contrastive_loss`。

## 3. 统一输入输出张量

### 3.1 像素特征与类别原型

分割特征常为：

$$
F\in\mathbb R^{B\times D\times H'\times W'}.
$$

**公式解释：** $F$ 是分割网络的连续特征图；$B$ 是批量大小，$D$ 是通道或嵌入维，$H',W'$ 是低分辨率空间尺寸。该式只声明输入 shape，没有进行乘法或消去维度；$F[b,:,h,w]$ 是第 $b$ 张图在位置 $(h,w)$ 的一个 $D$ 维像素特征。

若需要逐像素对比，先统一到目标尺寸并展平：

```text
[B,D,H',W']
→ bilinear interpolate(size=(H,W))
→ [B,D,H,W]
→ view(B,D,H×W)
→ [B,D,N]，N=H×W
→ permute(0,2,1)
→ [B,N,D]
```

双线性插值适合连续特征；若同步调整离散伪标签 $Y\in\{0,\ldots,C\}^{B\times H'\times W'}$，应使用最近邻插值，防止类别编号被平均。`view` 只把空间维展平，不改变像素顺序；`permute` 把通道移到最后，便于与 `[B,K,D]` 原型做批量矩阵乘法。

设归一化原型：

$$
P\in\mathbb R^{B\times K\times D},\qquad
\hat F=\frac{F}{\|F\|_2},\quad
\hat P=\frac{P}{\|P\|_2}.
$$

**公式解释：** $P$ 为每张图的 $K$ 个 $D$ 维候选原型，shape 是 `[B,K,D]`；展平后的像素特征按最后一维保存时为 `[B,N,D]`。两个除法都沿特征维 $D$ 计算 L2 范数并广播回原 shape，因此没有删掉 token 或原型维；$\hat F$ 与 $\hat P$ 中每个向量长度约为 1，后续点积可解释为余弦相似度。

相似度为：

$$
S=\hat P\hat F^T\in\mathbb R^{B\times K\times N}.
$$

**公式解释：** `P_hat=[B,K,D]` 与 `F_hat^T=[B,D,N]` 做批量矩阵乘法，参与运算的是原型维 $K$、特征维 $D$ 和像素维 $N$；共同的特征维 $D$ 被乘加消去，输出 `S=[B,K,N]`。$S[b,k,n]$ 表示第 $b$ 张图的第 $n$ 个像素与第 $k$ 个原型的余弦相似度。

数字例子：`F=[2,256,32,32] → [2,256,1024]`，`P=[2,21,256]`，相乘得到 `S=[2,21,1024]`。若作为分割 logit 恢复二维：

```text
[B,K,N]
→ reshape(B,K,H,W)
→ [2,21,32,32]
→ bilinear upsample
→ [2,21,512,512]
```

### 3.2 区域原型

若区域索引为 $R\in\{0,\ldots,K-1\}^{B\times H\times W}$，第 $k$ 个区域原型为：

$$
p_{b,k}=\frac{\sum_{n=1}^{N}\mathbf1[R_{b,n}=k]f_{b,n}}
{\sum_{n=1}^{N}\mathbf1[R_{b,n}=k]+\varepsilon}.
$$

**公式解释：** 指示函数 $\mathbf1[R_{b,n}=k]$ 只保留属于区域 $k$ 的像素特征 $f_{b,n}\in\mathbb R^D$。分子沿像素维 $N$ 求和，分母沿相同维度统计区域像素数，因而位置索引 $n$ 被消去；输出 $p_{b,k}\in\mathbb R^D$，其第 $d$ 个元素是区域 $k$ 在特征通道 $d$ 上的均值，$\varepsilon$ 防止空区域除零。代码常用 `scatter_mean` 一次完成分组与求均值。

### 3.3 图文批内对比

图像和文本全局嵌入为 $V,T\in\mathbb R^{B\times D}$：

$$
S_{it}=\hat V\hat T^T/\tau\in\mathbb R^{B\times B}.
$$

**公式解释：** `V_hat=[B,D]` 与 `T_hat^T=[D,B]` 做矩阵乘法，特征维 $D$ 被乘加消去，两个 batch 维分别保留为图像行和文本列，再由标量温度 $\tau$ 缩放，得到 `S_it=[B,B]`。$S_{it}[i,j]$ 是图像 $i$ 与文本 $j$ 的匹配 logit；对角线是配对样本，其余 batch 元素通常作为负样本。多标签或语义重复 caption 中，非对角元素可能仍描述同一概念，形成假负样本。

## 4. 正负样本怎样定义

| 粒度 | 正样本依据 | 负样本依据 | 典型风险 |
|---|---|---|---|
| 像素—像素 | 伪标签相同、同一增强对应位置 | 类别不同 | $O(N^2)$ 内存，伪标签错误形成假正样本 |
| 像素—类别原型 | 像素标签与原型类别相同 | 其他类原型 | 单原型压扁类内多样性 |
| 像素—区域原型 | 属于同一 SAM/超像素区域 | 其他区域 | 区域编号不等于语义类别 |
| 区域—文本 | 掩码区域与标题词/类别配对 | 其他词或 caption | 弱标题未给出精确区域对应 |
| 图像—文本 | 原始图文对 | batch 内其他文本 | 多标签数据中假负样本常见 |
| 跨模态原型 | 同类视觉与文本原型 | 异类原型 | 原型构造依赖筛选和聚类质量 |

正样本错误通常比漏掉一部分正样本更危险：漏样本只是少学；把异类当同类会主动把决策边界两侧拉到一起。WSSS 中应优先从高置信、区域内部和跨增强稳定位置构造正对，边界和冲突区可以不参与。

## 5. 代表论文逐篇说明

| 论文 | 任务与起点 | 原方法存在的问题 | 具体做法 | 与本算子的关系 |
|---|---|---|---|---|
| [[SSR_paper_notes]] | CLIP 驱动 WSSS；视觉与文本原型生成 CAM | CLIP 全局预训练留下模态差距，视觉特征类内分散、类间重叠，导致非目标前景过激活 | 跨模态原型对齐（Cross-Modal Prototype Alignment，**CMPA**）先用 CAM 感知的前景池化收集视觉/文本特征，在投影空间聚类为两类原型；对比约束让视觉特征靠近同类文本原型、同类视觉与文本原型聚合、异类跨模态原型分离 | 典型“跨模态原型对比”；正负关系由聚类伪标签决定，图中用于 CAM 的余弦 logit 与用于原型损失的 logit 需区分 |
| [[S2C_paper_notes]] | WSSS；SAM 提供无类别区域，主网络产生 CAM 特征 | 只有图像级分类损失时，像素特征缺少区域结构；SAM 无语义类标签，不能直接做类别交叉熵 | SAM-Segment Contrasting（**SSC**，SAM 区域对比）把分割一切模型（Segment Anything Model，**SAM**）的区域索引当作自监督 target；对每个区域平均学生特征形成原型，再让每个像素在本图区域原型中识别自己的区域编号 | 不是类别级 InfoNCE，而是图内“区域原型分类”；0 号区域被忽略，区域编号跨图像没有共享语义 |
| [[UGRL_paper_notes]] | 单阶段 WSSS；解码器由噪声 CAM 伪标签监督 | 只在 logit 层模仿伪标签，解码器特征仍缺少类内紧凑和类间分离；全部像素参与会放大噪声 | 原型驱动不确定性建模先估计类/像素可靠性；可靠语义增强（Reliable Semantic Enhancement，**RSE**）把解码器像素映射并 L2 归一化，只选低不确定性的 Top-K 像素，同伪类为正、异类为负 | [[Confidence_Reweighting]] 与对比学习串联：可靠性决定哪些样本有资格进入正负池 |
| [[OpenSeg_paper_notes]] | OVS；从类别无关掩码和图像标题学习开放词汇 | 全局图文向量丢失位置；逐像素与 caption 弱对齐噪声过大 | 先用掩码查询产生少量区域提议，再做掩码池化得到 `[N,D]` 区域嵌入；用区域—词语接地损失对齐标题中的名词/形容词，降低从弱标题学习密集语义的难度 | 区域化改变了对比/接地的输入单位；关键不是更多负样本，而是先把像素聚合为更可靠语义实体 |
| [[Talk2DINO_paper_notes]] | 无监督 OVS；DINOv2 有空间感知但无文本接口 | CLIP 全局语义强而定位弱，DINOv2 patch 特征定位强但未与文本对齐 | 学习投影 $\psi$ 把 CLIP 文本嵌入映射到 DINOv2 空间；每个 DINOv2 注意力头对 patch 特征加权池化成一个视觉候选，取与 caption 最匹配的头，用真实图文对为正、batch 内其余对为负训练 | 图文对比的“图像表示”不是固定全局池化，而是从多注意力头中动态选择；训练后可用 patch—文本相似度做分割 |
| [[POT_paper_notes]] | WSSS；图内原型和最优传输（Optimal Transport，**OT**）扩展 CAM | 单一全局类中心无法表达一张图内不同部位和外观；硬伪标签不稳定 | 从高置信区域按类聚类多个图内原型，利用原型与分类器权重关系设置非均匀容量，再用 OT 软分配像素并做一致性学习 | 不是标准 InfoNCE，但与原型对比共享“像素—候选原型相似度矩阵”；OT 额外约束候选容量，而 softmax 对比通常没有容量约束 |
| [[VDA_paper_notes]] | CLIP WSSS；动态视觉属性原型生成 CAM | 静态文本原型难适配实例的颜色、姿态和局部属性，解码器语义不一致 | 视觉属性建模与解耦（Visual Attribute Modeling and Disentanglement，**VAMD**）用层次高斯混合模型学习类别/属性原型；解码器语义增强（Decoder Semantic Enhancement，**DSE**）以全局类别原型为锚，对比约束解码器适配器嵌入 | 展示多原型不仅服务 CAM 查询，也能作为对比锚点；属性组件没有人工属性名称，不能把潜在簇直接解释成具体部件 |

## 6. 常见实现形式归纳

| 实现形式 | 输入单元 | 是否训练 | 优点 | 局限 | 代表论文 |
|---|---|---:|---|---|---|
| 像素两两对比 | 采样像素 `[M,D]` | 是 | 保留细粒度局部差异 | 配对复杂度高，伪标签噪声敏感 | [[UGRL_paper_notes]] |
| 类别原型对比 | 像素/区域与 `[C,D]` 原型 | 是 | 计算从 $O(N^2)$ 降到 $O(NC)$ | 单原型忽略类内多峰 | [[SSR_paper_notes]]、[[VDA_paper_notes]] |
| 区域原型分类 | 像素与图内 `[K,D]` 区域原型 | 是 | 不需区域语义标签，利用结构分组 | 区域编号不跨图共享 | [[S2C_paper_notes]] |
| 区域—文本对齐 | 区域嵌入与词嵌入 | 是 | 适合弱 caption 和 OVS | 区域—词对应仍是弱监督 | [[OpenSeg_paper_notes]] |
| 图文批内对比 | 全局/区域池化视觉向量与 caption | 是 | 可利用海量图文对 | 假负样本和全局定位不足 | [[Talk2DINO_paper_notes]] |
| 关系矩阵匹配 | 学生/教师 token 关系矩阵 | 是 | 无需明确负类标签 | $O(N^2)$，只保相对结构 | [[Distillation]] |

这些形式可以组合：用区域提议降低像素噪声，再用多原型表达类内多样性；或者先由 [[Confidence_Reweighting]] 选可靠像素，再做像素—原型对比。[[S2C_paper_notes]] 同时使用区域原型对比和 SAM 伪标签交叉熵；[[SSR_paper_notes]] 同时使用跨模态原型对比与超像素空间校正。

## 7. 各种实现怎样工作

### 7.1 像素—类别原型对比

**直觉**：不让每个像素和海量像素比较，而是让它回答“我最像哪个类别中心”。

**数据流**：特征上采样 → 伪标签筛选 → 按类平均或 EMA 更新原型 → L2 归一化 → 像素与所有类原型点积 → 对比交叉熵。

相似度为：

$$
s_{i,c}=\frac{\hat f_i^T\hat p_c}{\tau},
\qquad
\mathcal L_i=-\log\frac{e^{s_{i,y_i}}}{\sum_{c'=1}^{C}e^{s_{i,c'}}}.
$$

**公式解释：** $f_i,p_c\in\mathbb R^D$，第一式对两个归一化 $D$ 维向量做点积，消去特征维 $D$，再除以温度 $\tau$，得到像素 $i$ 对类别 $c$ 的标量 logit $s_{i,c}$。对全部 $C$ 类计算后形成长度为 $C$ 的候选向量；第二式在类别索引 $c'$ 上求和并消去类别维，取真实/伪标签 $y_i$ 对应项的负对数概率，最终输出单像素标量损失 $\mathcal L_i$。

**适用场景**：类别数远小于像素数、类别语义稳定、显存有限。

**局限**：背景和外观多样类别不适合只压成一个中心。可用每类多个原型 `[C,K,D]`，再对同类 $K$ 个原型取最大值或 log-sum-exp，但必须处理候选数增加造成的类别偏置。

### 7.2 区域原型对比

**直觉**：先相信“这些像素属于同一个视觉片段”，暂时不要求知道片段叫什么。

**数据流**：SAM/超像素区域图 → 区域内平均特征 → 像素查询本图全部区域原型 → 区域编号为 target。

**适用场景**：外部分组模型边界可靠，但没有任务类别；WSSS 缺少像素真值。

**容易误解**：不同图像里的区域 1 没有共同语义，不能直接跨图把相同索引当正样本；0 号区域是否表示背景、无效区或第一个实例必须从代码确认。

### 7.3 区域—文本与图文对比

**直觉**：caption 只说图里有什么，不说在哪。把像素先组成少量区域，可把“几万个像素找几个词”的难题变成“几十个区域找几个词”。

**数据流**：mask proposal → mask pooling → 区域嵌入 `[B,R,D]` → 文本编码 `[B,L,D]` → 余弦相似度 `[B,R,L]` → 区域—词接地或跨图对比。

若做：

$$
S=RT^T\in\mathbb R^{B\times R\times L},
$$

**公式解释：** 区域张量 `R=[B,R,D]` 与文本张量转置 `T^T=[B,D,L]` 做批量矩阵乘法，共同的特征维 $D$ 被消去，区域数 $R$ 和文本 token 数 $L$ 被保留，输出 `S=[B,R,L]`。$S[b,r,l]$ 表示第 $b$ 张图的区域 $r$ 与词 $l$ 的匹配分数；之后沿区域维 max 表示“图中是否有某词”，沿词维 max 则给区域选择最匹配词，两种归约含义不同。

**局限**：标题没提到的对象不应被当成负类；两个 caption 共享词时，batch 内非配对不一定是真负样本。

## 8. 温度、归一化与采样

### 8.1 L2 归一化

$$
\hat f=f/(\|f\|_2+\varepsilon).
$$

**公式解释：** 对每个 $D$ 维向量 $f$，分母先把其 $D$ 个分量平方求和并开根得到标量 L2 范数，再加 $\varepsilon$ 防止零向量除零；该标量广播到全部 $D$ 个分量，输出同 shape 的单位向量 $\hat f$。归一化不会删除 token 或空间维，点积因而只比较方向，避免模型仅增大模长来提高 logit。对 `[B,D,H,W]` 应使用 `dim=1`；对 `[B,N,D]` 应使用 `dim=-1`。

### 8.2 温度

常见写法是 `logit / tau`。$\tau$ 越小，分布越尖，困难负样本梯度越强；过小可能导致不稳定。也有代码把一个“逆温度”尺度 $s$ 直接乘 logit，此时 $s=1/\tau$。看到变量名 `T` 不能推断它一定是温度，必须看乘除号和默认值。

### 8.3 类别平衡与像素采样

密集特征数量很大，通常按类采样 $m_c$ 个可靠像素，并按类平均：

$$
\mathcal L=\frac1{|\mathcal C_I|}\sum_{c\in\mathcal C_I}
\frac1{m_c}\sum_{i=1}^{m_c}\ell(q_i^c).
$$

**公式解释：** $\mathcal C_I$ 是当前图像中参与训练的类别集合，$m_c$ 是类别 $c$ 采到的像素数，$q_i^c$ 是其第 $i$ 个查询，$\ell$ 输出单查询标量损失。内层沿样本索引 $i$ 求均值并消去采样维，得到每类损失；外层沿类别 $c$ 再求均值并消去类别维，最终输出标量 $\mathcal L$。这种两级平均可防止背景和大物体仅凭像素数量主导损失。

## 9. 官方仓库静态分析：S2C 的 SSC

- 官方仓库：[sangrockEG/S2C](https://github.com/sangrockEG/S2C)
- 阅读 commit：[`102e14c690c8e3bce3d5ccd1ae7832145ce10b27`](https://github.com/sangrockEG/S2C/tree/102e14c690c8e3bce3d5ccd1ae7832145ce10b27)
- 关键文件与函数：[`models/model_s2c.py::update`](https://github.com/sangrockEG/S2C/blob/102e14c690c8e3bce3d5ccd1ae7832145ce10b27/models/model_s2c.py#L180-L360)
- SSC 核心：[`L327-L343`](https://github.com/sangrockEG/S2C/blob/102e14c690c8e3bce3d5ccd1ae7832145ce10b27/models/model_s2c.py#L327-L343)

### 9.1 调用链

```text
主网络中间特征 feat_main [B,D,h,w]
→ 双线性插值到输入尺寸 [B,D,H,W]
→ 沿通道维 L2 归一化
→ view 为 [B,D,HW]
SAM 预生成区域图 self.se [B,H,W]
→ view 为区域索引 [B,1,HW]
→ scatter_mean 构造每个区域原型 [B,D,Nseg]
→ 原型归一化
→ 原型^T @ 像素特征，得到 [B,Nseg,HW]
→ 以区域索引为 target 做交叉熵
```

### 9.2 逐行解释

代码为：

```python
feat_main = F.interpolate(feat_main, size=(H,W),
                          mode='bilinear', align_corners=False)
feat_main = F.normalize(feat_main, dim=1)
feat_main_ = feat_main.view(B,D,-1)             # (B,D,HW)
index_ = self.se.view(B,1,-1).long()            # (B,1,HW)

pt = torch_scatter.scatter_mean(feat_main_.detach(), index_)
pt = F.normalize(pt, dim=1)
index_ = index_.squeeze(1)
pred_ssc = torch.bmm(pt.permute(0,2,1), feat_main_)  # (B,N,HW)

self.loss_ssc = F.cross_entropy(pred_ssc*self.T,
                                index_, ignore_index=0)
```

1. [`L328`](https://github.com/sangrockEG/S2C/blob/102e14c690c8e3bce3d5ccd1ae7832145ce10b27/models/model_s2c.py#L328) 把连续特征插值到 `(H,W)`；这里正确使用 `align_corners=False`，保证后续每个特征位置与 `self.se` 区域图像素一一对应。
2. [`L329`](https://github.com/sangrockEG/S2C/blob/102e14c690c8e3bce3d5ccd1ae7832145ce10b27/models/model_s2c.py#L329) 在 `dim=1` 通道维归一化。输入 `[B,D,H,W]`，因此每个像素的 $D$ 维向量单位化。
3. `view(B,D,-1)` 把 $H,W$ 合并为 $HW$；`self.se` 使用相同展平顺序，因此索引和特征仍对应。
4. [`scatter_mean`](https://github.com/sangrockEG/S2C/blob/102e14c690c8e3bce3d5ccd1ae7832145ce10b27/models/model_s2c.py#L333) 依据 `index_` 在像素维分组平均，`[B,D,HW] → [B,D,Nseg]`。$Nseg$ 由当前 batch 中区域索引最大值决定。
5. `pt.permute(0,2,1)` 得到 `[B,Nseg,D]`，与 `feat_main_=[B,D,HW]` 批量相乘，消去 $D$，输出 `[B,Nseg,HW]`。`pred_ssc[b,k,n]` 是像素 $n$ 与区域原型 $k$ 的相似度。
6. `index_.squeeze(1)` 得到交叉熵 target `[B,HW]`。交叉熵把 `Nseg` 当作候选类别维，让每个像素识别所属区域。
7. `ignore_index=0` 使 `self.se` 中索引为 0 的位置不产生 SSC 梯度。0 在这里是被忽略区域编号，不应自动解释成普通语义背景类。

### 9.3 梯度边界

`feat_main_.detach()` 只出现在构造原型的分支：

```text
feat_main_ ──detach──> scatter_mean ──> pt ──┐
feat_main_ ──────────────────────────────────×──> pred_ssc ──> loss
```

因此：

- 区域原型 `pt` 被视为当前特征的固定目标，不接收梯度；
- 查询像素特征 `feat_main_` 未 detach，仍由 `loss_ssc` 更新；
- 这不是完全停止 SSC 梯度，而是单边停止，避免原型和像素同时移动形成不稳定目标；
- `self.se` 是离散区域索引，本来也不可导。

### 9.4 温度实现与常见公式不同

代码使用：

```python
F.cross_entropy(pred_ssc * self.T, index_, ignore_index=0)
```

常见 InfoNCE 写 `pred_ssc / tau`。如果代码变量 `self.T=0.5`，乘法会把 logit 缩小为原来一半，使分布更平；常见的“除以温度 0.5”则会把 logit 放大两倍，使分布更尖。除非作者把 `T` 定义成逆温度，否则两者含义相反。阅读这份实现时应记录“代码乘 `T`”，不能把论文中的除温度公式直接代入。

### 9.5 代码与常规类别对比的区别

- 候选是**本图 SAM 区域**，不是数据集共享语义类别；同类别若被 SAM 切成两个区域，SSC 会暂时把它们当不同候选。
- 原型来自主网络当前特征的区域均值，并非冻结 SAM 的视觉 embedding；SAM 只提供 `self.se` 分组索引。
- 全分辨率 `[B,D,H,W]` 对比会增加显存和计算，代码没有再采样像素。
- 区域原型分支 detach，使这种约束更接近“固定当前区域中心，移动像素查询”，而不是对称 InfoNCE。

## 10. 关系约束不一定需要显式负样本

如果负样本不可靠，可让学生匹配教师关系矩阵：

$$
\mathcal L_{rel}=
\left\|\hat F_s\hat F_s^T-\operatorname{sg}(\hat F_t\hat F_t^T)\right\|_F^2.
$$

**公式解释：** $F_s,F_t\in\mathbb R^{B\times N\times D}$ 分别是学生和教师 token。每个张量与自身转置 `[B,D,N]` 批量相乘，特征维 $D$ 被消去，得到 `[B,N,N]` 关系矩阵；其中元素 $(i,j)$ 是两个 token 的相似度。$\operatorname{sg}$ 表示 stop-gradient，使教师关系不接收梯度；相减后，Frobenius 范数对两个 token 维及 batch 中的全部元素平方求和，输出标量 $\mathcal L_{rel}$。它保留“谁和谁相似”，不要求给每对显式正负标签，但计算量为 $O(N^2)$。

## 11. 选型指南

| 当前症状 | 优先考虑 | 先别做什么 |
|---|---|---|
| CAM/分割 logit 尚可，但解码器特征纠缠 | 可靠像素—类别原型对比 | 不对所有伪标签像素无筛选地两两对比 |
| 有可靠 SAM/超像素区域但无语义标签 | 图内区域原型分类 | 不把跨图相同区域编号当作同类 |
| 背景与前景外观高度多样 | 每类多原型、背景多簇 | 不强行压成单一背景原型 |
| caption 有图级语义但无位置 | 先区域提议，再区域—文本对齐 | 不把未出现于 caption 的区域直接当负样本 |
| OVS 需要保留未见类迁移 | 匹配区域/patch 与连续文本嵌入结构 | 不只在 seen 类封闭 softmax 上做蒸馏式对比 |
| batch 小导致负样本不足 | 原型队列或动量编码器 | 先检查旧特征漂移与跨卡索引 |
| 伪标签噪声高 | [[Confidence_Reweighting]]、增强一致性筛选 | 不因对比损失数值下降就认为语义更好 |

最简单基线通常是“L2 归一化像素特征 + 每类一个 detach 原型 + 交叉熵”，然后再依次验证多原型、可靠采样和困难负样本。应同时看线性/最近原型分类、类内/类间相似度、最终 mIoU 和边界指标；更漂亮的特征可视化不保证分割更准。

## 12. 论文与源码索引

- [[SSR_paper_notes]]：跨模态原型的正负关系、投影空间和 CAM logit 的区别。
- [[S2C_paper_notes]]：SAM 区域索引如何成为自监督 target。
- [[UGRL_paper_notes]]：不确定性筛选如何保护对比学习免受伪标签噪声影响。
- [[OpenSeg_paper_notes]]：先分组再做区域—词语弱监督对齐。
- [[Talk2DINO_paper_notes]]：DINOv2 注意力头选择与图文批内对比。
- [[VDA_paper_notes]]：多属性原型和解码器语义增强。
- [S2C SSC 源码](https://github.com/sangrockEG/S2C/blob/102e14c690c8e3bce3d5ccd1ae7832145ce10b27/models/model_s2c.py#L327-L343)：区域原型、相似度矩阵、停止梯度和温度实现。
- [S2C 主分支反向传播](https://github.com/sangrockEG/S2C/blob/102e14c690c8e3bce3d5ccd1ae7832145ce10b27/models/model_s2c.py#L304-L360)：SSC 与分类/CPM 损失最终如何一起更新主网络。

## 13. 当前整理结论

对比正则的核心选择是“比较什么”和“凭什么把它们判成正负”，不是套用哪一个公式。阅读新论文时应追问候选维代表类别、区域还是 batch 样本，正负关系由真值、伪标签、区域分组还是图文配对提供；修改模型时先检查归一化维、原型更新方式、温度乘除、背景定义与 detach 边界。把不同方法放回统一数据流，就是“形成 query/key → 构造或筛选正负关系 → 计算相似度矩阵 → 在候选维分类 → 将梯度送回允许更新的分支”。
