---
type: operator-note
aliases:
  - Attention Refinement
  - Affinity Refinement
  - 注意力与亲和力细化
tags:
  - research-operator
  - attention
  - affinity
  - refinement
  - semantic-segmentation
  - weakly-supervised
  - open-vocabulary
status: in-progress
---

# Attention and Affinity Refinement（注意力与亲和力细化）

> [!abstract] 本页定位
> 本页整理**怎样构造、校准并使用“位置—位置”关系矩阵来修正密集特征或类别响应**。证据来自经典Transformer/分割模型、弱监督语义分割（Weakly-Supervised Semantic Segmentation, WSSS）与开放词汇语义分割（Open-Vocabulary Segmentation, OVS）。单篇论文笔记保存完整方法；本页把注意力、特征亲和力、掩码约束和转移矩阵放进统一张量接口，并静态追踪一个官方实现。

> [!tip] 基础机制入口
> $QK^T$、多头注意力（Multi-Head Self-Attention, MHSA）与残差块先看 [[vision_transformer_notes]]、[[vision_transformer_code_notes]]；无标签自蒸馏第二版（self-DIstillation with NO labels v2, DINOv2）注意力的代码解释见 [[dinov2_code_notes_detailed]]。后文还会用到对比语言—图像预训练（Contrastive Language-Image Pre-training, CLIP）和分割一切模型（Segment Anything Model, SAM）。本页只关心这些关系怎样服务于密集预测。

## 1. 这个算子解决什么问题？

大白话说，类别激活图（Class Activation Map, CAM）回答“每个位置像什么类别”，亲和力回答“两个位置是否应该一起变化”。如果鸟头被可靠激活、鸟身没有激活，一个好的关系矩阵可以把鸟头证据传给鸟身；如果关系矩阵把鸟和天空连在一起，细化反而会扩大错误。

常见现象包括：

- CAM只覆盖最有判别力的一小块；
- 同一物体内部响应断裂；
- CLIP patch过度相似，类别边界被抹平；
- 自注意力包含全局关系，但其中既有同物体连接，也有跨类别噪声；
- 类别无关的DINO、SAM或扩散关系结构好，却不能单独给出类别。

专业表述是：给定位置特征或初始类别响应，构造关系矩阵 $A$，再将其转成适合消息聚合的权重 $T$：

$$
F\longrightarrow A\longrightarrow T,
\qquad
M'=\Psi(T,M).
$$

**公式解释：** $F$ 是视觉特征，先由它计算位置关系 $A$，再把 $A$ 归一化为可聚合的转移矩阵 $T$；$M$ 是原始类别响应，$\Psi$ 表示用 $T$ 重组 $M$，得到 $M'$。这条式子只规定处理顺序，本身不消去维度；真正的矩阵乘法在后面的 $T@M$ 中发生。

本算子主要决定**边是否存在、边有多强、边是否有方向、怎样归一化**。至于同一关系重复传播多少步、怎样重启和停止，重点见 [[Spatial_Propagation]]。

> [!note] 我的理解｜先判断语义错还是关系错
> 如果初始CAM把“狗”认成“猫”，关系传播没有新类别信息，通常只会让错误变完整；如果类别正确但区域断裂，亲和力细化才是对症操作。OVS中DINO/SAM只能修结构，最终类别仍应来自文本或开放分类器。

### 1.1 哪些对象容易被混淆？

| 对象 | 常见形状 | 是否已归一化 | 含义 |
|---|---|---:|---|
| attention logit | `[B,H,N,N]` | 否 | $QK^T/\sqrt d$ 的原始交互证据 |
| attention weight | `[B,H,N,N]` | 通常逐行softmax | query位置从哪些Key/Value读取 |
| feature affinity | `[B,N,N]` | 可选 | 两位置特征方向或距离是否相近 |
| binary adjacency | `[B,N,N]` | 否 | 是否允许两位置连边 |
| transition matrix | `[B,N,N]` | 必须说明方向 | 一步消息传播时权重怎样流动 |

形状相同不代表可以互换。注意力权重是特定 $Q/K$ 投影与softmax的结果；特征亲和力可能对称；转移矩阵必须满足明确的归一化约束。

## 2. 统一输入输出张量

设视觉特征为：

$$
F\in\mathbb{R}^{B\times N\times D},
$$

**公式解释：** $B$ 是批量大小，$N=H'W'$ 是展平后的 patch/像素位置数，$D$ 是每个位置的特征维。$F[b,n,d]$ 表示第 $b$ 张图第 $n$ 个位置的第 $d$ 个特征分量。这只是输入 shape 声明，没有发生运算或维度消去。初始类别响应为：

$$
M\in\mathbb{R}^{B\times N\times C},
$$

**公式解释：** $C$ 是类别数，$M[b,n,c]$ 表示第 $b$ 张图第 $n$ 个位置对类别 $c$ 的响应。它与 $F$ 共享 $B,N$，但最后一维从特征坐标 $D$ 变为类别坐标 $C$；这里只声明接口，没有求和或维度消去。

关系矩阵为：

$$
A\in\mathbb{R}^{B\times N\times N}.
$$

**公式解释：** $A$ 的两个 $N$ 分别对应接收位置 $i$ 和来源位置 $j$。$A[b,i,j]$ 描述第 $b$ 张图中位置 $i$ 与位置 $j$ 的关系强度。它没有类别维，只说明位置关系；若行 $i$ 是接收位置，则归一化后的 $T[b,i,j]$ 表示位置 $i$ 从位置 $j$ 读取多少证据。

一次细化为：

$$
M'=TM\in\mathbb{R}^{B\times N\times C}.
$$

**公式解释：** $T=[B,N,N]$、$M=[B,N,C]$。批量矩阵乘法把 $T$ 的来源位置维与 $M$ 的位置维相乘求和，消去一个 $N$；接收位置 $N$ 和类别维 $C$ 保留，输出仍是 `[B,N,C]`。$M'[b,i,c]$ 表示位置 $i$ 聚合邻居后对类别 $c$ 的新响应。具体地：

$$
M'[b,i,c]=\sum_{j=1}^{N}T[b,i,j]M[b,j,c].
$$

**公式解释：** 固定 $b,i,c$，公式遍历所有来源位置 $j$。$T[b,i,j]$ 是边权，$M[b,j,c]$ 是来源位置对同一类别的证据；求和消去 $j$，得到标量 $M'[b,i,c]$。类别维没有被混合，所以亲和力本身不会把猫分数变成狗分数。

数字例子：

```text
F: [2,400,512]
F^T: [2,512,400]
A = F @ F^T: [2,400,400]
M: [2,400,20]
M' = A @ M: [2,400,20]
```

若恢复空间图：

```text
[B,N,C]
→ permute(0,2,1)
→ [B,C,N]
→ reshape(B,C,H',W')
→ [B,C,H',W']
→ bilinear interpolate
→ [B,C,H,W]
```

`permute` 与 `reshape` 只恢复索引布局，不生成新语义；双线性插值只平滑放大，不能代替关系细化。完整上采样比较见 [[downsampling_and_upsampling(下采样与上采样)]]。

## 3. 代表模型与论文

| 论文/模型 | 任务与起点 | 原方法存在的问题 | 具体做法 | 与本算子的关系 |
|---|---|---|---|---|
| [[vision_transformer_notes]] | 图像分类；patch token | CNN局部感受野难直接建立全局依赖 | 用MHSA计算每个query对全部Key的权重，再聚合Value并做残差更新 | 提供最常用的位置关系来源，但注意力不是分割真值 |
| [[mask2former_notes]] | 全监督通用分割；对象query | query对整图交叉注意力成本高、关注分散 | 用上一解码层预测掩码将无关像素的attention logit设为 $-\infty$，让query只读候选区域 | 说明预测掩码可以反向约束下一层交互范围 |
| [[CLIP-ES_paper_notes]] | WSSS；冻结CLIP与梯度加权类别激活映射（Gradient-weighted Class Activation Mapping, Grad-CAM） | 原始MHSA类别无关，直接传播会放大非目标区域 | 类感知基于注意力的亲和力（Class-aware Attention-based Affinity, CAA）用Sinkhorn交替归一化并对称化注意力，再用当前类别CAM的连通框限制传播 | 类别种子决定允许使用哪些通用注意力边 |
| [[WeCLIP_paper_notes]] | 单阶段WSSS；冻结CLIP骨干 | 静态CAM训练中无法改进，多层CLIP关系含噪 | 冻结CLIP CAM细化模块（Refinement Module for Frozen CLIP CAM, RFM）用可学习解码器亲和力筛选CLIP多层注意力，再细化CAM | 用动态任务特征当裁判，选择可靠的冻结关系 |
| [[DiCLIP_paper_notes]] | WSSS；CLIP + 稳定扩散 | CLIP注意力过平滑、空间多样性不足 | 视觉相关性增强（Visual Correlation Enhancement, VCE）把经注意力聚类细化（Attention Clustering Refinement, ACR）的扩散关系加到CLIP后层注意力中 | 在特征生成阶段注入外部空间关系，而非只后处理CAM |
| [[SSR_paper_notes]] | WSSS；CLIP CAM | 图文模态间隙与背景传播同时存在 | 先做图文子空间/原型对齐，再用超像素约束亲和力随机游走 | 明确把语义校准和空间关系校准拆成两步 |
| [[CorrCLIP_paper_notes]] | 免训练OVS；CLIP分类器 | CLIP patch存在错误类间相关性 | 用分割一切模型（Segment Anything Model, SAM）限制patch交互区域，以DINO相似度重建关系值，再聚合CLIP Value | 类别无关模型修正文本分类前的视觉关系 |
| [[Trident_paper_notes]] | 高分辨率OVS；CLIP/DINO/SAM | 滑窗特征缺少跨窗关系 | DINO补局部对象关系、SAM补全局区域聚合、CLIP保留语义分类 | 展示多来源关系在不同空间范围内分工 |

## 4. 常见实现形式

| 实现形式 | 关系来源 | 是否训练 | 优点 | 局限 | 代表论文 |
|---|---|---:|---|---|---|
| 复用Transformer注意力 | $QK^T$ 或softmax权重 | 否/随骨干 | 无需额外关系网络 | 受预训练目标与投影影响 | [[CLIP-ES_paper_notes]]、[[WeCLIP_paper_notes]] |
| 特征自相似 | $\hat F\hat F^T$ | 取决于 $F$ | 简单、对称语义直观 | 特征错误会直接变成错误边 | [[ExCEL_paper_notes]] |
| 学习式亲和力头 | 解码特征/像素对监督 | 是 | 可适配目标任务 | 依赖伪标签，可能自我确认 | [[WeCLIP_paper_notes]] |
| 外部结构注入 | DINO、SAM、扩散关系 | 通常冻结 | 补充CLIP空间结构 | 尺度与token对应复杂 | [[DiCLIP_paper_notes]]、[[CorrCLIP_paper_notes]] |
| 区域/边界掩码 | CAM框、超像素、SAM掩码 | 否 | 直接抑制越界传播 | 错误区域会阻断召回 | [[CLIP-ES_paper_notes]]、[[SSR_paper_notes]] |
| 多关系融合 | 多头、多层、多模型矩阵 | 可选 | 利用互补关系 | 归一化和方向必须一致 | [[WeCLIP_paper_notes]]、[[Trident_paper_notes]] |

## 5. 各种实现怎样工作？

### 5.1 特征自相似

先在最后一维做L2归一化：

$$
\hat F_{b,n,:}=\frac{F_{b,n,:}}{\|F_{b,n,:}\|_2+\varepsilon},
$$

**公式解释：** 对每张图、每个位置分别计算其 $D$ 维特征向量的 L2 范数，再让所有分量除以同一个标量。归一化发生在特征维 $D$，输出 $\hat F$ 仍是 `[B,N,D]`；$\hat F[b,n,:]$ 是只保留方向、模长约为 1 的位置特征。$\varepsilon$ 防止零向量除零。

再计算：

$$
A=\hat F\hat F^T\in\mathbb{R}^{B\times N\times N}.
$$

**公式解释：** `[B,N,D] @ [B,D,N]` 逐特征维点积并消去 $D$，得到 `[B,N,N]`。$A[b,i,j]$ 是位置 $i,j$ 的余弦相似度；它天然对称，但含负值，尚不是转移概率。

常见处理为：

$$
A_+=\max(A,0),\qquad
\tilde A=A_++\gamma I,
$$

**公式解释：** $A_+=\max(A,0)$ 对 `[B,N,N]` 每个元素做 ReLU，把负相关边置零；$I$ 是 $N\times N$ 单位矩阵，$\gamma I$ 给每个位置增加自环。输出 $\tilde A$ shape 不变，$\tilde A[b,i,j]$ 是去负边并补自环后的非负边权。

$$
T_{ij}=\frac{\tilde A_{ij}}{\sum_k\tilde A_{ik}+\varepsilon}.
$$

**公式解释：** 对固定接收位置 $i$，分母沿所有来源位置 $k$ 求和，得到该行总边权；每个 $\tilde A_{ij}$ 再除以该标量。求和临时消去来源维，但广播相除后 $T$ 仍为 `[B,N,N]`，且每行和为 1。$T_{ij}$ 可解释为位置 $i$ 从位置 $j$ 读取证据的比例。若任务需要有向信息流，不应为了“看起来稳定”盲目对称化。

**适用**：已有空间可分的视觉特征。**局限**：全连接关系显存为 $O(N^2)$；$N=4096$ 时单个32位浮点（32-bit floating point, FP32）矩阵约64 MiB，还未计batch、多头与梯度。

### 5.2 复用多头注意力

单个注意力头：

$$
A_h=\operatorname{softmax}\left(\frac{Q_hK_h^T}{\sqrt{d_h}}\right)
\in\mathbb{R}^{B\times N\times N}.
$$

**公式解释：** $Q_h,K_h$ 都是 `[B,N,d_h]`。`Q_h @ K_h^T` 在头内特征维 $d_h$ 做点积并消去它，得到 `[B,N,N]`；除以 $\sqrt{d_h}$ 控制数值尺度，softmax 沿最后的来源位置维归一化。$A_h[b,i,j]$ 表示 query 位置 $i$ 从 key 位置 $j$ 读取的权重。完整 Transformer 输出还包括 $A_hV_h$、多头拼接、输出投影、残差与 MLP，因此“注意力图”不是模型最终贡献度的严格解释。

多头/多层可平均：

$$
\bar A=\sum_{l,h}\omega_{l,h}A_{l,h},
\qquad \sum_{l,h}\omega_{l,h}=1.
$$

**公式解释：** $A_{l,h}$ 是第 $l$ 层第 $h$ 个头的 `[B,N,N]` 关系矩阵，$\omega_{l,h}$ 是对应标量权重。对层和头索引求和会消去这两个来源维，输出 $\bar A$ 仍是 `[B,N,N]`；$\bar A[b,i,j]$ 是各层各头对边 $i\leftarrow j$ 的加权平均。平均前要确认各层 token 顺序、是否含 CLS 以及归一化方向一致。CLS 行列通常应去掉，否则 `[1+N,1+N]` 不能直接对应 $H'\times W'$。

### 5.3 掩码限制关系范围

设 $G\in\{0,1\}^{B\times N\times N}$ 表示允许的边。最稳妥的是在softmax前屏蔽：

$$
L'_{ij}=\begin{cases}
L_{ij},&G_{ij}=1,\\
-\infty,&G_{ij}=0,
\end{cases}
\qquad T=\operatorname{softmax}(L';\text{dim}=-1).
$$

**公式解释：** $L,G$ 都是 `[B,N,N]`，$G_{ij}$ 只决定边是否允许。禁止边的 logit 被设为 $-\infty$，随后 softmax 沿来源位置 $j$ 归一化，所以禁止边概率严格为 0，允许边重新分配到总和 1。shape 始终不变，$T[b,i,j]$ 是掩码约束后的读取概率。若先 softmax 再做 $T\odot G$，必须再次行归一化，否则每行总质量会随被遮挡比例变化。

$G$ 可来自当前类别CAM的框、超像素同区关系、SAM区域或边界预测。框比像素阈值掩码更宽松，有利于从不完整CAM向外补全；超像素/SAM边界更严格，更能防泄漏但可能挡住缺失区域。

### 5.4 外部关系注入

DiCLIP式加法可以抽象为：

$$
A_{en}^{l}=A_{clip}^{l}+\alpha A_{ext}.
$$

**公式解释：** $A_{clip}^{l}$ 与 $A_{ext}$ 必须是同一 token 网格上的 `[B,N,N]` 关系矩阵，$\alpha$ 是控制外部关系强度的标量。逐元素相加不消去维度；$A_{en}^{l}[b,i,j]$ 表示 CLIP 原关系与外部结构关系共同给出的边权。两者相加前还必须方向和数值范围一致；若一个已 softmax、一个未归一化，相同 $\alpha$ 没有稳定含义。

CorrCLIP更接近“用SAM决定允许连接谁，用DINO决定连接多强，再用CLIP Value聚合”。这说明关系矩阵的**拓扑**与**边权**可以来自不同模型。

### 5.5 残差式细化

直接用 $TM$ 可能完全覆盖原始种子，因此常写成：

$$
M'=(1-\lambda)M+\lambda TM,
\qquad 0\le\lambda\le1.
$$

**公式解释：** $M$ 是 `[B,N,C]` 原响应，`T @ M` 先消去来源位置维并得到同 shape 的传播响应，$\lambda$ 是标量门控。两个 `[B,N,C]` 张量逐元素加权相加，不再消去维度；$M'[b,n,c]$ 是原证据与邻域证据折中的类别分数。$\lambda=0$ 不细化，$\lambda=1$ 完全依赖关系矩阵。残差只能减缓错误传播，不能纠正系统性错误类别。

## 6. 官方仓库静态分析：CLIP-ES的CAA细化

### 6.1 仓库与固定版本

- 官方仓库：[linyq2117/CLIP-ES](https://github.com/linyq2117/CLIP-ES)
- commit：[`3893f817be359c5ee1dbf8111cad381a532c7acc`](https://github.com/linyq2117/CLIP-ES/tree/3893f817be359c5ee1dbf8111cad381a532c7acc)
- 关键文件：[generate_cams_voc12.py#L151-L196](https://github.com/linyq2117/CLIP-ES/blob/3893f817be359c5ee1dbf8111cad381a532c7acc/generate_cams_voc12.py#L151-L196)

### 6.2 调用链与shape

```text
输入图像
→ Softmax-GradCAM grayscale_cam [H',W']
→ CLIP最后注意力，去掉CLS [N,N]
→ 交替列/行归一化
→ 对称化并平方一次
→ CAM连通框生成 aff_mask [1,N]
→ trans_mat * aff_mask
→ [N,N] @ [N,1]
→ reshape [H',W']
→ 放大到原图
```

源码关键段：

```python
attn_weight = [aw[:, 1:, 1:] for aw in attn_weight_list]
attn_weight = torch.stack(attn_weight, dim=0)[-8:]
attn_weight = torch.mean(attn_weight, dim=0)
attn_weight = attn_weight[0].cpu().detach()

trans_mat = aff_mat / torch.sum(aff_mat, dim=0, keepdim=True)
trans_mat = trans_mat / torch.sum(trans_mat, dim=1, keepdim=True)
for _ in range(2):
    trans_mat = trans_mat / torch.sum(trans_mat, dim=0, keepdim=True)
    trans_mat = trans_mat / torch.sum(trans_mat, dim=1, keepdim=True)
trans_mat = (trans_mat + trans_mat.transpose(1, 0)) / 2
trans_mat = torch.matmul(trans_mat, trans_mat)
trans_mat = trans_mat * aff_mask
cam_refined = torch.matmul(trans_mat, cam_to_refine).reshape(h // 16, w // 16)
```

逐步解释：

1. `[:,1:,1:]` 同时去掉类别标记（class token, CLS token）对应的query行与key列，把 `[1+N,1+N]` 变成 `[N,N]`。
2. `detach().cpu()` 明确切断注意力到CLIP的梯度；CAA生成伪CAM是免训练路径。
3. 代码先列归一化、再行归一化，共做3轮交替归一化，近似论文所说的Sinkhorn处理。
4. `(T + T.T) / 2` 把有向注意力变成无向亲和力；随后 `T @ T` 把两跳路径压成一次矩阵。
5. `cam_to_refine` 从 `[H',W']` 展平为 `[N,1]`；`[N,N] @ [N,1]` 消去来源位置维，得到每个接收位置的新CAM。

### 6.3 值得注意的工程细节

- 归一化除法没有 `clamp_min(eps)`；若某行/列和为0，存在NaN风险。
- `aff_mask` 的shape是 `[1,N]`，与 `[N,N]` 相乘时沿行广播，因此屏蔽的是**列/来源位置**，不是构造完整的 `[N,N]` 成对框掩码。
- 掩码乘法后没有重新归一化，最终 `trans_mat` 不再保证行和或列和为1；它更准确地是“受框限制的亲和权重”，不应再严格解释成转移概率。
- `attn_weight_list` 在当前循环中只在 `idx == 0` 时追加一次；按这段脚本本身，`stack(...)[-8:]` 不会形成8层平均。变量名与切片意图需要结合CAM对象返回值进一步核对，不能仅凭 `[-8:]` 声称融合了8层。
- `torch.matmul(T,T)` 扩大关系感受范围，再用框掩码裁剪；这不是卷积，也不是双线性上采样。
- CAM放大使用后续 `scale_cam_image`，空间插值发生在关系传播之后。

> [!note] 我的理解｜论文公式与代码要分开读
> 论文的CAA强调“双随机、对称、类别框限制”；代码确实执行交替归一化与对称化，但掩码是列广播且没有再归一化。复现或改写时应先决定自己要的是成对区域掩码、来源掩码，还是严格随机游走矩阵。

## 7. 选型指南

| 当前问题 | 优先考虑 |
|---|---|
| CAM类别正确但物体内部断裂 | 特征自相似或注意力亲和力 + 残差细化 |
| 传播跨越明显边界 | 超像素/SAM/边界掩码，并在屏蔽后重归一化 |
| CLIP patch过度同质化 | 在分类前用DINO/扩散关系重建视觉交互 |
| 冻结骨干多层注意力质量不一 | 用任务特征筛层，或先做逐层消融 |
| 小物体被关系平均抹掉 | 局部稀疏边、自环和较小 $\lambda$ |
| 显存不足 | 局部窗口、Top-k kNN图或分块矩阵乘法 |
| 类别本身错误 | 先修语义对齐/提示/原型，不先传播 |
| 免训练OVS需要边界结构 | 类别无关DINO/SAM关系 + CLIP文本分类 |

## 8. 调试与正确消融

1. 固定同一初始CAM，只替换关系来源，避免把更好种子误归因于亲和力。
2. 分别保存原始logit、softmax注意力、掩码后矩阵、归一化后矩阵与最终CAM。
3. 检查行和、列和、对称误差、对角线、自环比例、非零边比例和NaN。
4. 可视化若干query行，确认高权重邻居落在同一物体而非只看矩阵均值。
5. 报告细化前后CAM mIoU、前景召回、背景误激活与边界F-score；响应面积变大不等于变好。
6. 比较完整 $N^2$、局部窗口、Top-k图，区分长距离关系收益与额外计算收益。
7. 做“oracle affinity”或固定关系源实验，估计上限来自种子还是关系。

## 9. 论文与源码索引

- [[CLIP-ES_paper_notes]]：CAA、Sinkhorn归一化与类别框限制。
- [[WeCLIP_paper_notes]]：用动态解码器亲和力筛选冻结CLIP注意力。
- [[DiCLIP_paper_notes]]：扩散关系经ACR后注入CLIP注意力。
- [[SSR_paper_notes]]：超像素限制WSSS随机游走。
- [[CorrCLIP_paper_notes]]：SAM定拓扑、DINO定边权、CLIP聚合Value。
- [[Trident_paper_notes]]：局部与全局关系由不同骨干负责。
- [CLIP-ES注意力提取与CLS移除](https://github.com/linyq2117/CLIP-ES/blob/3893f817be359c5ee1dbf8111cad381a532c7acc/generate_cams_voc12.py#L151-L166)：确认关系矩阵来源与detach。
- [CLIP-ES归一化、对称化和传播](https://github.com/linyq2117/CLIP-ES/blob/3893f817be359c5ee1dbf8111cad381a532c7acc/generate_cams_voc12.py#L168-L196)：确认广播、平方与reshape。

## 10. 当前整理结论

注意力与亲和力细化的核心不是“拿一个 `[N,N]` 矩阵去乘CAM”，而是：

$$
\boxed{
\text{关系来源}
\rightarrow
\text{方向与归一化}
\rightarrow
\text{类别/边界约束}
\rightarrow
\text{一次受控聚合}
}.
$$

**公式解释：** 这不是数值计算式，而是阅读亲和力模块的四步检查顺序：先确认关系来自什么特征，再确认边的方向与归一化维，然后确认类别或边界怎样屏蔽边，最后检查一次聚合后的输出 shape 与语义。

阅读新论文时，应追问矩阵是logit、注意力、相似度还是转移概率，softmax在哪一维，CLS是否移除，掩码在归一化前还是后应用。修改模型时先确认初始语义是否正确，再检查边的拓扑和权重；只有关系可靠，空间证据传播才有意义。
