---
type: operator-note
aliases: [Pooling, Region Aggregation, 池化与区域聚合]
tags: [research-operator, pooling, region-feature]
status: todo
---

# Pooling and Region Aggregation（池化与区域聚合）

> [!abstract] 核心直觉
> 池化是在回答：“一组位置怎样压成一个代表向量？”它减少数据量并提高区域一致性，但被压掉的空间差异通常无法恢复。

> [!tip] 基础机制入口
> 全局池化与多尺度池化的模型级背景可参考 [[deeplabv3+_notes]] 和 [[upernet_notes]]；本页重点是掩码、片段和原型内部怎样做区域聚合。

## 1. 输入与输出

设密集特征 $F\in\mathbb{R}^{B\times N\times D}$，区域掩码 $M\in\{0,1\}^{B\times R\times N}$，则区域池化输出：

$$Z\in\mathbb{R}^{B\times R\times D}.$$

从 `[B,N,D]` 到 `[B,R,D]`，被消去的是区域内部的位置维。若全局池化，则 $R=1$，输出 `[B,D]`。

## 2. 常见实现形式

| 形式 | 计算 | 优点 | 局限 |
|---|---|---|---|
| 平均池化 | 区域内求均值 | 稳定、无参数 | 小目标会被背景稀释 |
| 最大池化 | 每通道取最大值 | 保留最强证据 | 梯度集中、忽略其余位置 |
| 加权池化 | 按注意力/置信度加权 | 可强调可靠位置 | 权重错误会放大噪声 |
| 掩码池化 | 仅聚合候选区域 | 区域语义更完整 | 依赖掩码质量 |
| 原型/簇内池化 | 每个簇分别聚合 | 保留类内多样性 | 需要聚类或分配 |

## 3. 公式与大白话解释

### 3.1 掩码平均池化

$$
z_r=\frac{\sum_{n=1}^{N}M_{r,n}F_n}
{\sum_{n=1}^{N}M_{r,n}+\varepsilon}.
$$

$M_{r,n}=1$ 表示第 $n$ 个patch属于区域 $r$。分子把区域内特征相加，分母除以位置数，得到区域平均外观。$\varepsilon$ 防止空掩码除零。

数值例子：一个区域包含三个一维特征 `[2,4,9]`，掩码只选择前两个，则输出为 $(2+4)/2=3$，而不是把区域外的9也平均进去。

### 3.2 置信度加权池化

$$
\alpha_{r,n}=\frac{\exp(a_{r,n})}{\sum_{j\in r}\exp(a_{r,j})},\qquad
z_r=\sum_n\alpha_{r,n}F_n.
$$

softmax把区域内权重变成和为1的比例。得分更高的位置贡献更大；如果所有分数相同，它会退化成平均池化。

### 3.3 最大池化

$$z[d]=\max_n F[n,d].$$

它是逐通道取最大值，不一定所有通道都来自同一个位置。因此输出可以是多个位置的“拼合证据”，不一定对应真实存在的某个patch。

## 4. 论文中的用法

| 论文 | 聚合单元 | 目的 |
|---|---|---|
| [[OpenSeg_paper_notes]] | 类别无关掩码内的像素 | 得到区域嵌入，再和标题词语对齐 |
| [[POT_paper_notes]] | 类别种子内部的多个簇 | 为同一类别构造多个局部原型 |
| [[SSR_paper_notes]] | 可靠前景特征簇 | 构造跨模态原型并做对比学习 |
| [[S2C_paper_notes]] | SAM片段内特征 | 形成片段原型，约束像素靠近所属片段 |
| [[MCTformer_paper_notes]] | 类别token对patch的注意力 | 把多个注意力头/层汇成类别定位图 |

## 5. 基础代码骨架

```python
# features: [B, N, D], masks: [B, R, N]
weights = masks.float()
denom = weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
regions = torch.einsum("brn,bnd->brd", weights, features) / denom
```

`einsum` 中的 $n$ 同时出现在输入但不出现在输出，因此它被求和消掉；$b,r,d$ 被保留，结果正是 `[B,R,D]`。

## 6. 工程观察：S2C的片段原型

S2C固定版本 [`102e14c`](https://github.com/sangrockEG/S2C/tree/102e14c690c8e3bce3d5ccd1ae7832145ce10b27) 在 [`models/model_s2c.py`](https://github.com/sangrockEG/S2C/blob/102e14c690c8e3bce3d5ccd1ae7832145ce10b27/models/model_s2c.py#L327-L349) 中先把主分支特征插值到图像大小并归一化，再用片段索引收集区域。代码重点不是某一行平均，而是“片段编号充当分组键”：同一编号的像素被聚合成片段原型，随后用于SAM-Segment Contrasting（SAM片段对比，SSC）。

## 7. 怎样选择？

- 区域边界大致可靠：掩码平均池化是最稳的起点。
- 只有少量判别位置可靠：最大池化或高温度加权池化更合适。
- 类内外观差异大：不要把整类压成一个向量，改用多原型。
- 边界非常粗：池化前先做 [[Region_Grouping_and_Proposal]] 或 [[Attention_and_Affinity_Refinement]]。
- 需要恢复像素预测：必须保留区域到像素的映射，之后才能把区域分数回投。

## 8. 与经典分割的连接

池化在不同架构中压缩的维度并不相同：

| 经典机制 | 被聚合的范围 | 保留下来的信息 | 对当前算子的启示 |
|---|---|---|---|
| 分类网络GAP | 整张特征图 | 每通道全局均值 | 适合“有没有”，不适合直接恢复“在哪里” |
| [[deeplabv3+_notes]] 的ASPP图像级分支 | 全局 + 多种感受野 | 多尺度上下文 | 上下文池化与区域池化目的不同，不能混称 |
| [[upernet_notes]] 的PPM | 多个空间网格 | 分层上下文 | 池化后仍可上采样回特征图并参与解码 |
| [[maskformer_notes]] / [[mask2former_notes]] | 掩码覆盖的像素 | 一个query对应的区域表示 | 先分组后分类，接近开放词汇区域对齐 |

弱监督中的池化通常从噪声CAM或片段中构造监督锚点；开放词汇中的池化则把像素/patch变成更完整的区域嵌入，减少局部token与文本直接匹配的不稳定性。这两者都依赖区域纯度，但前者更关心伪标签噪声，后者更关心区域是否漏掉标题中的对象。

## 9. 软掩码、重叠区域与回投

硬掩码并非唯一选择。若 $M_{r,n}\in[0,1]$，仍可计算：

$$
z_r=\frac{\sum_n M_{r,n}F_n}{\sum_n M_{r,n}+\varepsilon}.
$$

它允许边界位置部分属于某区域，但掩码logit应先经过sigmoid或其他明确变换；直接用任意正负logit做权重会造成分母抵消。

区域可能重叠。把区域分数 $s_{r,c}$ 回投像素时，常见选择包括：

$$
y_{n,c}=\max_{r:M_{r,n}>0}s_{r,c},
\qquad
y_{n,c}=\frac{\sum_rM_{r,n}s_{r,c}}{\sum_rM_{r,n}+\varepsilon}.
$$

最大值保留最自信提议，平均更平滑但会让大量低质量提议稀释正确区域。[[OpenSeg_paper_notes]] 的关键不只是生成区域向量，还包括将区域分类结果重新组合为像素预测；因此必须记录区域索引和空间掩码，不能只缓存 $Z$。

## 10. 数值与数据边界

- **空区域**：分母截断只能避免NaN，不能赋予空区域语义；应丢弃并在loss中mask掉。
- **区域大小偏差**：大区域在训练样本中可能占据更多权重，应明确按区域平均还是按像素平均loss。
- **背景污染**：小目标掩码中混入少量背景就可能显著改变均值，可先腐蚀边界或按置信度加权。
- **特征插值顺序**：先插值特征再池化与先池化再插值一般不等价；前者改变区域内样本，后者已经丢失空间维。
- **归一化顺序**：`mean(normalize(F))` 与 `normalize(mean(F))` 不等价。用于余弦分类时通常至少要对最终区域向量再归一化。

## 11. 应报告什么

除了最终mIoU，还应报告区域覆盖率（真实物体是否被至少一个区域覆盖）、区域纯度、平均提议数、空掩码比例，以及像素级与区域级损失各自的权重。这样才能区分“区域没提出来”和“区域提出来但文本分类错了”。

## 12. 池化到底在聚合哪一维？

同一个 `mean()` 可能对应完全不同的操作。先把常见张量写清楚：

| 输入 | 被聚合的维度 | 输出 | 含义 |
|---|---|---|---|
| $F:[B,D,H,W]$ | $H,W$ | $[B,D]$ | 每张图一个全局向量 |
| $F:[B,N,D]$ | $N$ | $[B,D]$ | 对全部patch做全局池化 |
| $F:[B,N,D],M:[B,R,N]$ | 每个mask内的 $N$ | $[B,R,D]$ | 每个区域一个向量 |
| $F:[B,R,D]$ | $R$ | $[B,D]$ | 把所有区域再压成整图表示 |
| $A:[B,H_{attn},N,D_h]$ | 注意力头 $H_{attn}$ | $[B,N,D_h]$ | 多头聚合，不是空间池化 |
| $S:[B,R,C]$ | 区域 $R$ | $[B,C]$ | multiple-instance式图像分类 |

阅读代码时不要只记录“average pooling”，而应写成“沿哪个轴、按什么权重、输出交给谁”。一旦聚合掉位置维，就不能仅靠reshape恢复原来的内部空间结构。

## 13. 全局池化：从局部证据得到图像级判断

### 13.1 Global Average Pooling（GAP）

$$
z_d=\frac1N\sum_{n=1}^{N}F_{n,d}.
$$

若分类器为 $y_c=w_c^Tz$，则：

$$
y_c=\frac1N\sum_nw_c^TF_n.
$$

这正是经典CAM能把分类权重 $w_c$ 重新投到每个空间位置的原因：全局logit可分解为所有位置的类别证据平均。[[S2C_paper_notes]] 先生成CAM，再沿空间轴GAP得到图像级分类logit，也利用了同一结构。

GAP让所有位置梯度相对均匀，但小目标只占很少位置时，其信号会被大面积背景稀释。它适合稳定的图像级存在性监督，不保证模型学到完整目标。

### 13.2 Global Max Pooling（GMP）

$$z_d=\max_nF_{n,d}.$$

最大池化只让每个通道的赢家位置接收梯度，特别符合“至少一个位置出现该模式”的multiple-instance假设。它能保留小而强的证据，却容易让分类器长期依赖最判别部位，造成WSSS CAM更不完整。

### 13.3 平均与最大之间的连续折中

LogSumExp pooling：

$$
z_d=\frac1\beta\log\left(\frac1N\sum_n\exp(\beta F_{n,d})\right).
$$

$\beta\rightarrow0$ 时接近平均（差一个可忽略的常数极限处理），$\beta\rightarrow\infty$ 时接近最大。它让多个高响应位置都有梯度，又比平均更重视强证据。

Generalized Mean（GeM）在非负特征上定义为：

$$
z_d=\left(\frac1N\sum_n(F_{n,d}+\varepsilon)^p\right)^{1/p}.
$$

$p=1$ 是平均，$p$ 增大后更接近最大。若特征含负值，应先明确非负变换，不能直接对负数取非整数幂。

## 14. 上下文池化与对象区域池化不是一回事

### 14.1 Pyramid Pooling Module（PPM）

[[upernet_notes]] 的PPM把最高层特征自适应池化到多个网格，例如 $1\times1,2\times2,3\times3,6\times6$，分别投影并上采样回原网格后拼接：

$$
F_{ppm}=\operatorname{Concat}left(
F,operatorname{Up}(P_1(F)),\ldots,\operatorname{Up}(P_K(F))
\right).
$$

它的目的是给**每个位置**补充不同范围的上下文，最终仍输出二维特征图；不是把一个真实对象mask压成一个区域向量。

### 14.2 ASPP中的图像池化分支

[[deeplabv3+_notes]] 的ASPP同时使用多种空洞率和一个全局图像池化分支。全局分支先把整张特征图压到 $1\times1$，投影后广播/上采样回各位置。它提供场景先验，例如室内/室外上下文，但不会保留目标内部结构。

### 14.3 对象区域池化

mask pooling只聚合某个候选对象/片段覆盖的像素：

$$z_r=\operatorname{Pool}(F\mid M_r).$$

它输出区域集合 `[B,R,D]`，用于区域命名、原型或检索。上下文池化与区域池化可以组合：先用PPM增强 $F$，再在 $M_r$ 内聚合；顺序相反则区域已经失去像素网格，无法再执行空间金字塔。

## 15. Box pooling、RoI Align与mask pooling

### 15.1 Box pooling

给定边界框 $b_r=(x_1,y_1,x_2,y_2)$，最简单做法是在框内平均。问题是框包含背景，尤其细长或非矩形对象会被严重污染。

### 15.2 RoI Pooling与RoI Align

检测/实例分割常把不同大小box变为固定 $K\times K$ 特征：

$$F_r=\operatorname{RoIAlign}(F,b_r)\in\mathbb R^{D\times K\times K}.$$

RoI Pooling对边界量化并在bin内最大池化；RoI Align使用双线性采样减少坐标量化误差。二者保留固定网格内部结构，而普通区域平均只输出 `[D]`。如果后续还需预测mask形状，应优先保留 $K\times K$ RoI特征；如果只做开放词汇区域分类，单向量更省计算。

### 15.3 mask pooling

mask比box更贴合对象：

$$z_r=\frac{\sum_{h,w}M_{r,h,w}F_{h,w}}
{\sum_{h,w}M_{r,h,w}+\varepsilon}.$$

但mask与特征图通常分辨率不同。可将mask插值到特征网格，或把特征插值到mask网格。前者更省计算，后者更保留高分辨率边界；二者数值不等价，应在实现中明确。

## 16. 硬mask、软mask和前景纯度

### 16.1 硬mask

$M\in\{0,1\}$ 直观、便于索引，但阈值附近的梯度被截断。如果mask由另一个可训练分支产生，硬二值化通常阻断从区域分类损失到mask预测器的梯度。

### 16.2 软mask

$M\in[0,1]$ 允许区域分类loss反向影响mask：

$$
\frac{\partial z_r}{\partial M_{r,n}}
=\frac{F_n-z_r}{\sum_jM_{r,j}+\varepsilon}.
$$

当某像素特征比当前区域均值更有利于loss时，其mask权重可能增大。这样可以联合学习区域与语义，但也可能出现捷径：mask只保留少量最判别位置以提高分类分数，而不是覆盖完整对象。需要分割loss、面积约束或类别无关mask监督共同约束。

### 16.3 内核区域与边界区域分开池化

弱监督mask边界噪声大时，可腐蚀mask得到高纯度内核 $M^{core}$，再把边界作为另一组：

$$M^{boundary}=M-M^{core}.$$

用内核构造语义原型，用边界只参与低权重一致性或细化，可以减少背景进入原型。代价是小目标可能被腐蚀为空，需要按区域大小自适应核半径。

## 17. 加权区域聚合的权重从哪里来

设未归一化可靠度为 $a_{r,n}$：

$$
\alpha_{r,n}=\frac{M_{r,n}\exp(a_{r,n}/\tau)}
{\sum_jM_{r,j}\exp(a_{r,j}/\tau)+\varepsilon},
\qquad
z_r=\sum_n\alpha_{r,n}F_n.
$$

可用权重包括：

| 权重来源 | 强调什么 | 主要风险 |
|---|---|---|
| CAM/文本相似度 | 类别相关位置 | 再次强化最判别局部 |
| mask概率 | 区域内部稳定位置 | mask分支错误会被放大 |
| 边界距离 | 远离边界的纯净内核 | 忽略真实边界外观 |
| attention/query分数 | 与区域query最相关位置 | attention不等于类别贡献 |
| 不确定性反权重 | 低熵/高margin位置 | 过度自信错误仍会占优 |

温度 $\tau$ 小会趋向单个位置，区域聚合退化成近似max；$\tau$ 大则趋近mask平均。应查看有效样本数：

$$N_{eff}=\frac1{\sum_n\alpha_{r,n}^2}.$$
$$

$N_{eff}\approx1$ 表示几乎只有一个位置贡献，即使mask面积很大。

## 18. 多头注意力池化与query聚合

给定区域query $q_r$ 和像素键值 $K,V$：

$$
a_{r,n}=\operatorname{softmax}_n(q_r^Tk_n/\sqrt D),
\qquad
z_r=\sum_na_{r,n}v_n.
$$

这比固定mask平均更灵活：query可根据内容选择区域内部的不同部位。MaskFormer/Mask2Former的query通过cross-attention聚合图像信息，得到per-segment embedding；上一层mask还可限制下一层注意范围。需要注意：

- query特征 $z_r$ 是通过多层交互得到的区域槽位表示，不必等于对最终二值mask做简单平均；
- query注意力权重与最终mask预测也不是同一个张量；
- 若要与文本做余弦相似度，通常还需类别投影与L2归一化。

## 19. OpenSeg：为什么先分组再聚合再对齐

[[OpenSeg_paper_notes]] 把三种图像表示区分得很清楚：

```text
全局图文模型：1 × D —— 有语义，丢失位置
逐像素模型：H × W × D —— 保留位置，像素监督昂贵
区域集合模型：R × D —— 用mask保持对象组织，再与词语对齐
```

其核心数据流可概括为：

$$
F_s\xrightarrow{\text{mask queries}}M\in[0,1]^{R\times H\times W},
$$

$$
(F_z,M)\xrightarrow{\text{mask pooling}}Z\in\mathbb R^{R\times D},
$$

$$
Z\xrightarrow{\text{region-word grounding}}S\in\mathbb R^{R\times C}.
$$

这里生成mask的特征 $F_s$ 与用于语义聚合的特征 $F_z$ 可以承担不同职责。区域池化是连接视觉分组和文本对齐的接口，而不是独立完成开放词汇分类。

## 20. 弱监督中的区域聚合实例

### 20.1 S2C片段原型

[[S2C_paper_notes]] 用SAM自动片段给像素分组，在每个片段内平均分类器特征形成原型；特征和原型在平均前后沿通道归一化，再进行片段对比。这里片段标签不是语义类别，而是“属于同一空间区域”的自监督关系。

### 20.2 POT类别原型

[[POT_paper_notes]] 中，初始CAM筛选类别相关特征，mask/分配后的特征经平均形成多个聚类原型。池化样本的纯度直接影响原型CAM；若背景进入同一聚类，之后的余弦激活会系统性扩散错误。

### 20.3 SSR跨模态原型

[[SSR_paper_notes]] 先筛可靠前景，再聚合视觉原型并与文本原型对比。此时池化既要去背景，又要保留类内多样性；“把所有高CAM像素平均成一个向量”未必足够。

## 21. 变量区域数与批处理实现

不同图像区域数 $R_b$ 不同，常见实现有三种：

1. **padding + valid mask**：补到批内 $R_{max}$，同时保存 `region_valid:[B,Rmax]`；适合GPU批处理。
2. **扁平化区域列表**：把所有区域合成 $[\sum_bR_b,D]$，另存 `batch_index`；适合聚类和检索。
3. **逐图循环**：最直观但吞吐较差，可用于SAM提议数差异很大的原型构建。

padding区域必须从softmax、对比负样本和loss中排除。仅把其特征设为0仍可能产生一个合法的零向量logit或进入分母。

一个支持软mask的批量实现：

```python
def masked_average(features, masks, region_valid=None, eps=1e-6):
    # features: [B, N, D], masks: [B, R, N]
    weights = masks.float().clamp_min(0)
    mass = weights.sum(dim=-1, keepdim=True)          # [B, R, 1]
    regions = torch.einsum("brn,bnd->brd", weights, features)
    regions = regions / mass.clamp_min(eps)

    nonempty = mass.squeeze(-1) > eps
    valid = nonempty if region_valid is None else nonempty & region_valid
    regions = regions.masked_fill(~valid.unsqueeze(-1), 0)
    return regions, valid, mass.squeeze(-1)
```

函数同时返回区域质量 `mass` 和有效标记，避免“数值上防除零”被误解为“空区域已经得到可用特征”。

## 22. 梯度与停止梯度的选择

区域特征 $z(F,M)$ 同时依赖特征和mask。不同训练目的对应不同梯度路径：

| 目的 | 对 $F$ 梯度 | 对 $M$ 梯度 |
|---|---:|---:|
| 用固定SAM片段监督分类器 | 是 | 否，SAM mask固定 |
| 联合学习类别无关mask与区域语义 | 是 | 是，但需mask监督防捷径 |
| 用伪mask构造稳定原型 | 通常是 | 通常detach |
| 离线建立检索缓存 | 否 | 否 |

若mask来自当前CAM，并用池化后的原型反过来监督同一CAM，完整梯度闭环可能让模型通过收缩mask获得更纯的原型。常见做法是对选择/二值mask停止梯度，只让被选中的视觉特征学习。

## 23. 失败诊断矩阵

| 现象 | 可能原因 | 优先检查 |
|---|---|---|
| 区域类别总被背景主导 | box/mask含大量背景 | 区域纯度、内核池化、背景原型 |
| 小目标区域向量接近零 | 下采样后仅1–2个位置或空mask | 特征分辨率、mass、RoI Align |
| 多个同类实例难区分 | 全类池化把实例合并 | 实例/片段级区域而非类别级mask |
| 区域分类好、回投像素差 | 重叠mask冲突或覆盖不足 | proposal recall、回投规则 |
| 软mask只剩判别小块 | 分类梯度驱动mask收缩 | 分割loss、面积/完整性约束 |
| 区域特征模长差异大 | 面积、归一化顺序或空区域 | 最终L2归一化、mass统计 |
| 加权池化退化为max | 温度过低/置信度过尖 | $N_{eff}$ 与权重直方图 |

## 24. 池化实验应如何拆解

- 固定区域提议，比较mean/max/加权/attention pooling，隔离聚合规则收益。
- 固定聚合规则，替换真实mask、SAM、超像素和CAM区域，估计区域质量上限。
- 同时报区域分类准确率、oracle proposal IoU、最终像素mIoU，定位失败阶段。
- 按区域面积分桶；平均池化和RoI Align对小区域的表现可能完全不同。
- 对soft mask报告梯度是否进入mask分支，以及加入语义loss后mask面积和覆盖率如何变化。
- 比较 `normalize→pool→normalize`、`pool→normalize` 等顺序，不能只写“使用余弦相似度”。

## 25. 阅读论文时的区域聚合记录模板

```text
输入特征来源、层号和形状：
区域来源（真值/SAM/query/CAM/超像素）：
区域是box、硬mask还是软mask：
mask与特征的分辨率对齐方式：
池化轴与公式：
是否类别条件化/注意力加权：
池化前后归一化顺序：
空区域和小区域回退策略：
重叠区域如何处理：
区域分数怎样回投像素：
mask/特征两侧梯度是否保留：
评价区域覆盖、纯度和分类的指标：
```

## 26. 当前整理结论

池化不是免费的压缩：它用区域稳定性换取空间细节。每次池化都应明确“哪一维被消掉”以及“以后是否还需要把结果放回原位置”。
