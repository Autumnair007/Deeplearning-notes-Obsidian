---
type: operator-note
aliases:
  - Confidence Reweighting
  - Uncertainty Reweighting
  - 置信度重加权
tags:
  - research-operator
  - confidence
  - uncertainty
  - weakly-supervised-segmentation
  - open-vocabulary-segmentation
status: in-progress
---

# Confidence Reweighting（置信度重加权）

## 1. 本页定位

本页整理弱监督语义分割（Weakly Supervised Semantic Segmentation，**WSSS**）和开放词汇分割（Open-Vocabulary Segmentation，**OVS**）中“不同像素、类别、样本或分支不应同等可信”这一类操作。重点不是复述某篇论文的置信度模块，而是回答：可靠性从哪里来、在哪个维度产生权重、权重改变预测还是梯度、什么时候需要停止梯度，以及如何检查模型是否只是把困难样本藏起来。完整方法仍回到 [[ComCD_paper_notes]]、[[UGRL_paper_notes]]、[[POT_paper_notes]]、[[S2C_paper_notes]] 和 [[CLIP-ES_paper_notes]] 阅读。

> [!abstract] 一句话直觉
> 两个分支、两种伪标签或两个像素证据并不总是一样可靠。置信度重加权就是先估计“这一份证据有多值得信”，再决定它对融合结果或训练梯度贡献多少。

## 2. 这个算子解决什么问题

### 2.1 大白话解释

弱监督模型常从不完整的类激活图（Class Activation Map，**CAM**）、冻结基础模型或自动生成的伪掩码学习。它们在容易区域可能正确，在边界、遮挡、小目标和共现背景处可能犯错。如果把所有像素都当作真值，错误监督和正确监督会以相同强度更新模型。

置信度重加权通常放在三处：

1. **预测融合前**：判断当前位置更信任 CLIP、扩散模型、SAM 或学生分支中的哪一个；
2. **伪标签生成时**：低置信像素设为 ignore，或只让高置信区域成为监督；
3. **损失汇总时**：不删除像素，但缩小不可靠样本产生的梯度。

### 2.2 专业表述

给定证据 $x_i$、预测 $p_i$ 和单点损失 $\ell_i$，可靠性估计器产生 $w_i\in[0,1]$，随后用 $w_i$ 调节融合、标签采纳或经验风险：

$$
\mathcal L=\frac{\sum_i w_i\ell_i}{\sum_iw_i+\varepsilon}.
$$

**公式解释：** 分子把每项标量损失 $\ell_i$ 乘可靠性 $w_i$ 后沿 $i$ 求和，分母是有效权重总量。索引 $i$（可表示像素、区域、类别或图像）被消去，输出标量 $\mathcal L$；$\varepsilon$ 防止全零权重除零，$w_i$ 越大，该项梯度贡献越大。

### 2.3 哪些相似问题不由它解决

- CAM 只覆盖判别性部位：置信度只能选择或缩放已有证据，不能凭空补出缺失区域；需要亲和传播、区域扩张或 [[Pseudo_Label_Refinement]]。
- 两个分支处在不同通道数、分辨率或语义空间：先做 [[Cross_Modal_Alignment]] 和尺寸对齐，不能直接比较未对齐 logit。
- 错误预测非常自信：这是校准或分布偏移问题，单次 softmax 最大值无法识别。
- 伪标签边界粗糙：降低边界权重可少学错，但不会自动变出更准确的边界。

> [!note] 我的理解｜置信度不是“真值概率”
> 它只是由当前规则或模型给出的可靠性代理。低熵、分支一致和增强稳定都可能与正确性相关，但都不保证正确。真正需要验证的是：权重高的子集是否确实更准，以及这种相关性在新词表、新类别和新数据域上是否保持。

## 3. 统一输入输出张量

### 3.1 双分支逐像素门控

设两个分支输出：

$$
Z^{(1)},Z^{(2)}\in\mathbb R^{B\times C\times H\times W},
\qquad G\in[0,1]^{B\times1\times H\times W}.
$$

**公式解释：** $Z^{(1)},Z^{(2)}$ 是两个分支的 `[B,C,H,W]` 类别 logit，$G$ 是 `[B,1,H,W]` 门控图。$B,C,H,W$ 分别是批量、类别和空间维；$G[b,0,h,w]$ 是该位置对分支 1 的信任。这里只声明输入，没有维度消去。

- $B$：batch 大小；
- $C$：类别数；
- $H,W$：输出空间尺寸；
- $Z^{(m)}[b,c,h,w]$：分支 $m$ 对类别 $c$ 的 logit；
- $G[b,0,h,w]$：位置 $(h,w)$ 对分支 1 的信任程度。

融合为：

$$
Z=G\odot Z^{(1)}+(1-G)\odot Z^{(2)}.
$$

**公式解释：** $G$ 的类别维是 1，乘法时广播到 $C$ 类；两项都是逐元素乘法和加法，没有矩阵乘法或维度消去，输出仍是 `[B,C,H,W]`。$Z[b,c,h,w]$ 是两个分支在同一像素、同一类别上的加权联合分数。

数字例子：两个分支都是 `[2,20,64,64]`，门控图为 `[2,1,64,64]`；广播后输出 `[2,20,64,64]`。如果门控为 `[2,20,64,64]`，则每个类别可拥有不同权重，但参数更多，也更容易学到类别偏置。

### 3.2 多分支门控

若有 $M$ 个分支，将 logit 堆叠为：

$$
Z_{all}\in\mathbb R^{B\times M\times C\times H\times W},
\qquad A\in\mathbb R^{B\times M\times H\times W}.
$$

**公式解释：** $Z_{all}$ 在第二维堆叠 $M$ 个 `[B,C,H,W]` 分支，$A$ 为同位置的 $M$ 个未归一化门控 logit。这里只增加分支维，没有求和；$Z_{all}[b,m,c,h,w]$ 是分支 $m$ 的类别分数，$A[b,m,h,w]$ 是其门控证据。

先在分支维归一化：

$$
\tilde G_{b,m,h,w}=\operatorname{softmax}_{m}(A)_{b,m,h,w},
\qquad
Z_{b,c,h,w}=\sum_{m=1}^{M}\tilde G_{b,m,h,w}Z_{all,b,m,c,h,w}.
$$

**公式解释：** softmax 只在分支维 $M$ 进行，使每个像素的 $\tilde G$ 权重和为 1，shape 仍是 `[B,M,H,W]`。第二式把权重在类别维广播，与 $Z_{all}$ 逐元素相乘，再沿 $m$ 求和并消去分支维；输出 `[B,C,H,W]`。$Z[b,c,h,w]$ 是 $M$ 个分支的动态投票。

### 3.3 不同分辨率必须先对齐

若一个分支为 `[B,C,32,32]`，另一个为 `[B,C,64,64]`，需先插值到共同网格：

```text
[B,C,32,32]
→ bilinear interpolate(size=(64,64))
→ [B,C,64,64]
→ 与另一分支计算置信度和融合
```

双线性插值适合连续 logit、概率和置信图；最近邻插值适合离散类别索引，避免产生不存在的类别编号；转置卷积有可学习参数，适合解码器内恢复结构，不适合仅为比较两个固定置信图临时引入。基础机制见 [[downsampling_and_upsampling(下采样与上采样)]]。

## 4. 常见置信度信号

| 信号 | 计算单位 | 直觉 | 主要风险 |
|---|---|---|---|
| 最大概率 | 像素/类别 | 第一名越高越自信 | 不看第二名有多接近，受校准影响 |
| top-1 margin | 像素 | 第一名与第二名差距越大越可靠 | 类别数和长尾分布改变尺度 |
| Shannon 熵 | 像素的类别分布 | 分布越平越犹豫 | softmax 词表变化会改变熵 |
| 分支一致性 | 像素或区域 | 两个独立来源同意时更可信 | 两分支可能继承同一偏差 |
| 增强稳定性 | 像素/样本 | 缩放、翻转或扰动后仍一致 | 需要多次前向，坐标必须反变换 |
| 原型距离 | 像素/类别 | 更靠近可靠类原型更可信 | 原型本身可能被伪标签污染 |
| 学习式门控 | 像素/分支 | 小网络根据上下文预测权重 | 可能塌缩为永远选择强分支 |

### 4.1 熵权重怎样计算

对分支 $m$ 的类别概率：

$$
p_c^{(m)}=\operatorname{softmax}_{c}(Z^{(m)})_c,
\qquad
H^{(m)}=-\sum_{c=1}^{C}p_c^{(m)}\log(p_c^{(m)}+\varepsilon).
$$

**公式解释：** 第一式在类别维 $C$ 把 `Z^(m)=[B,C,H,W]` 变成同 shape 的概率 `p^(m)`。第二式逐类计算 $-p\log p$ 并沿 $c$ 求和，消去类别维，得到 `[B,1,H,W]` 熵图；$H^{(m)}[b,0,h,w]$ 表示分支 $m$ 在该像素的类别分布有多分散。

双分支权重可写为：

$$
G=\sigma\bigl(\beta(H^{(2)}-H^{(1)})\bigr).
$$

**公式解释：** $H^{(1)},H^{(2)}$ 都是 `[B,1,H,W]`，逐元素相减后 shape 不变；标量 $\beta$ 缩放熵差，sigmoid 再映射到 `[0,1]`，输出同 shape 门控 $G$。当分支 1 熵更低时差值为正，$G>0.5$；$G[b,0,h,w]$ 是该像素给分支 1 的权重。

## 5. 代表论文逐篇说明

| 论文 | 任务与起点 | 原方法存在的问题 | 具体做法 | 与本算子的关系 |
|---|---|---|---|---|
| [[ComCD_paper_notes]] | WSSS 与 OVS；分别生成 CLIP CAM 和扩散 CAM | CLIP 擅长类别定位但非判别区域稀疏；扩散响应连续但类别特异性较弱，固定选一个会浪费互补性 | 先计算两份逐像素类别分布的 Shannon 熵，再把熵差映射成像素权重，生成融合 CAM 与伪掩码；后续特征对齐解码器还用 logit 门控模块（Logit Gating Module，**LGM**）学习融合两分支分割 logit | 同时展示“固定熵规则”和“可学习门控”两种层级；权重先用于监督生成，后用于最终预测 |
| [[UGRL_paper_notes]] | 单阶段 WSSS；CAM 伪标签监督解码器 | 所有类别和像素被同等对待，低显著性类别会带来噪声；只在 logit 层模仿伪标签，特征仍纠缠 | 用原型驱动的不确定性估计类级监督可靠性；一方面调制分类与分割损失，另一方面只在可靠像素上施加对比约束，使解码器特征类内紧凑、类间可分 | 权重不仅筛标签，还直接控制任务损失与 [[Contrastive_Regularization]] 的样本集合 |
| [[POT_paper_notes]] | WSSS；用原型最优传输（Prototypical Optimal Transport，**POT**）扩展 CAM | 图内不同原型质量和像素分配难度不同，均匀监督会让噪声原型过度影响一致性学习 | 从高置信特征聚类出图像内原型，以分类器权重与原型关系决定非均匀传输容量；再用二元信息熵构造自适应权重，调节最优传输（Optimal Transport，**OT**）细化结果与原 CAM 的一致性损失 | 置信度作用于损失项；需区分“OT 传输计划是当前迭代计算结果”和“网络参数通过一致性损失更新” |
| [[S2C_paper_notes]] | WSSS；用分割一切模型（Segment Anything Model，**SAM**）给 CAM 提供区域结构 | SAM 有边界但无目标任务类别；CAM 有类别但区域不完整 | 多尺度 CAM 的局部峰值作为 SAM 正点提示；固定取 SAM 的第 3 个候选，将其逐像素置信度乘以该掩码内对应类别 CAM 的均值，再按类别取最大分数得到硬伪标签 | 是“结构置信度 × 语义置信度”的保守乘法门；权重用于类别竞争和伪标签选择，不是两个 logit 的线性融合 |
| [[CLIP-ES_paper_notes]] | 免训练生成 CAM，再训练最终分割器 | 阈值化 CAM 会把不确定区域错误地当成前景或背景，噪声伪掩码误导训练 | 置信度引导损失（Confidence-Guided Loss，**CGL**）从 CAM 构造置信图，忽略低置信位置，只在更可靠像素上计算分割交叉熵 | 属于硬筛选式权重：低置信像素权重直接为 0；应同时报告有效监督覆盖率 |
| [[DiCLIP_paper_notes]] | WSSS；扩散空间先验增强 CLIP 密集知识 | CLIP patch 表示过平滑，扩散注意力和生成缓存也含噪声 | 注意力聚类细化先阈值过滤扩散注意力的低值关系，再递归强化同组关系；前景/背景缓存共同参与 patch 级检索 | 它主要做可靠关系筛选和固定比例融合，并非标准概率置信重加权；适合作为“先筛可靠关系再传播”的邻近形式 |
| [[TokenMasking_paper_notes]] | WSSS；每类一个分类 token | 多个类别 token 容易学到相同注意区域，类别归属混乱 | 依据图像级已知类别，对不存在类别的分类 token 最终输出随机置零；伪掩码合并时让高 logit 类别覆盖低 logit 类别 | 图像级标签是硬可靠性先验，logit 排序决定冲突优先级；它不是校准后的连续置信度 |

> [!note] 我的理解｜不要把所有“有阈值”的模块都叫置信度重加权
> DiCLIP 的关系阈值、TokenMasking 的类别存在掩码和 S2C 的候选选择都在丢弃不可靠证据，但只有当信号被明确映射为权重并改变融合或梯度时，才是本页的核心形式。记录新论文时应写清权重到底乘在哪里。

## 6. 常见实现形式归纳

| 实现形式 | 权重 shape | 是否训练 | 优点 | 局限 | 代表论文 |
|---|---:|---:|---|---|---|
| 熵差分支融合 | `[B,1,H,W]` 或 `[B,M,H,W]` | 否 | 无额外参数，可直接检验分支互补 | 依赖分支校准和候选词表 | [[ComCD_paper_notes]] |
| 学习式门控 | `[B,M,H,W]` | 是 | 可利用局部特征和上下文 | 易塌缩，需监控门控分布 | [[ComCD_paper_notes]] |
| 置信阈值筛选 | `[B,H,W]` 布尔掩码 | 否 | 简单，能阻断明显噪声 | 大量 ignore 会降低覆盖率 | [[CLIP-ES_paper_notes]] |
| 损失连续加权 | 与单点损失同 shape | 可选 | 保留中等置信样本，不是一刀切 | 学生可通过降低权重逃避困难样本 | [[UGRL_paper_notes]]、[[POT_paper_notes]] |
| 多源乘法打分 | `[B,C,H,W]` | 否 | 任一证据差都会被抑制，较保守 | 一项尺度过小会压扁全部分数 | [[S2C_paper_notes]] |
| 原型/距离可靠性 | `[B,C]`、`[B,N]` 或 `[B,C,H,W]` | 依实现而定 | 利用特征结构而不只看 softmax | 原型噪声会系统性传播 | [[UGRL_paper_notes]]、[[POT_paper_notes]] |

这些形式可以组合。例如先用熵门控融合两份 CAM，再对融合 CAM 做阈值筛选，最后以可靠像素训练模型。但同一未经校准的最大概率若同时控制融合、筛选和损失，会把一次判断放大三遍，使错误高置信区域占据绝大多数梯度。

## 7. 各种实现怎样工作

### 7.1 预测融合：谁更可靠就多听谁

**直觉**：两个分支都给答案，权重决定当前位置的投票比例。

**数据流**：分支 logit → 各自转概率 → 估计熵/稳定性 → 分支维 softmax → 加权求和 → 融合 logit。

**适用场景**：两个分支确实互补，并且输出能对齐到相同类别和空间网格。

**容易误解**：直接平均概率和平均 logit 不等价；若两分支温度不同，熵较低的分支可能只是更过度自信，而非更准确。

### 7.2 伪标签筛选：不知道就暂时不教

给定类别概率 $P\in[0,1]^{B\times C\times H\times W}$，可计算：

$$
q_{b,h,w}=\max_cP_{b,c,h,w},\qquad
\tilde Y_{b,h,w}=\begin{cases}
\arg\max_cP_{b,c,h,w},&q_{b,h,w}\ge\tau,\\
255,&q_{b,h,w}<\tau.
\end{cases}
$$

**公式解释：** 第一式在类别维 $C$ 取最大概率，消去类别维，得到 `q=[B,H,W]`；$q[b,h,w]$ 是该像素第一名类别的置信度。第二式在同一类别维执行 $\arg\max$：当 $q\ge\tau$ 时输出获胜类别索引，否则输出 255。最终 `Y_tilde=[B,H,W]`，每个元素表示一个类别编号或“暂不监督”；255 通常作为 `ignore_index`，不会产生交叉熵梯度。

**适用场景**：伪标签噪声显著，训练损失支持 ignore。

**局限**：阈值升高通常提高被保留像素的精度，却降低覆盖率。只报告伪标签精度而不报告有效像素比例，会掩盖这种取舍。

### 7.3 损失加权：保留样本，但让它少说一点

```python
weight = confidence.detach()
loss_map = F.cross_entropy(logit, target, reduction="none")
loss = (weight * loss_map).sum() / (weight.sum() + 1e-6)
```

若 `logit=[B,C,H,W]`，逐像素交叉熵 `loss_map=[B,H,W]`，置信图也应为 `[B,H,W]`。乘法逐元素进行，最后对 $B,H,W$ 求和得到标量。`detach()` 使学生不能通过主动降低权重来减小损失；梯度只沿 `loss_map → logit → 学生参数` 回传。

**适用场景**：希望中等置信像素仍提供少量监督，而不是全部 ignore。

**局限**：权重来源若与学生共享同一错误，停止梯度也不能纠正偏差，只能防止投机。

## 8. 官方仓库静态分析：S2C 的乘法置信度

- 官方仓库：[sangrockEG/S2C](https://github.com/sangrockEG/S2C)
- 阅读 commit：[`102e14c690c8e3bce3d5ccd1ae7832145ce10b27`](https://github.com/sangrockEG/S2C/tree/102e14c690c8e3bce3d5ccd1ae7832145ce10b27)
- 关键文件：[`models/model_s2c.py`](https://github.com/sangrockEG/S2C/blob/102e14c690c8e3bce3d5ccd1ae7832145ce10b27/models/model_s2c.py)
- 入口函数：`update(self, epo, iter)`

### 8.1 调用链

```text
输入图像 [B,3,H,W]
→ 主网络在 0.5/1.0/1.5/2.0 倍尺度生成 CAM
→ ReLU、插值到 [H,W]、图像级标签过滤、求和并逐类归一化
→ 从每个已知类别 CAM 采样全局/局部峰值
→ 峰值作为 SAM 正点提示，SAM 返回 3 个候选掩码与置信图
→ 固定选择候选索引 2
→ SAM 像素置信 × 候选掩码内该类 CAM 均值
→ 在类别维 max，得到硬伪标签 [B,H,W]
→ 后续 CPM 交叉熵监督主网络
```

### 8.2 多尺度语义置信度

[`L195-L218`](https://github.com/sangrockEG/S2C/blob/102e14c690c8e3bce3d5ccd1ae7832145ce10b27/models/model_s2c.py#L195-L218) 在 `torch.no_grad()` 中计算多尺度 CAM：

```python
img_ms = [img_05, img_10, img_15, img_20]
cam_temp = F.relu(F.interpolate(out['cam'], size=(H,W),
                               mode='bilinear', align_corners=False))
cam_temp *= self.label.view(B,C,1,1)
cam_ms += cam_temp
cam_max = F.adaptive_max_pool2d(cam_ms, (1, 1))
cam_ms = cam_ms / (cam_max + 1e-5)
```

分步骤看：

1. 每个尺度的 `out['cam']` 先双线性插值到 `[B,C,H,W]`，因此四个尺度可以逐元素相加；
2. `self.label=[B,C]` reshape 为 `[B,C,1,1]`，在空间维广播，不存在的类别整张 CAM 被置零；
3. `adaptive_max_pool2d(...,(1,1))` 得到 `[B,C,1,1]` 的逐图逐类最大值；
4. 除法在空间维广播，使每个类别 CAM 的最大值接近 1，代码加入 `1e-5` 防止除零；
5. 整段无梯度，主网络不能通过改变这条伪标签构造链直接降低当前损失。

### 8.3 结构置信度与语义置信度相乘

[`L266-L298`](https://github.com/sangrockEG/S2C/blob/102e14c690c8e3bce3d5ccd1ae7832145ce10b27/models/model_s2c.py#L266-L298) 是核心：

```python
sam_conf = -1e5*torch.ones_like(cam_ms)
mask = output_sam[0]  # (1,3,H,W)
conf = output_sam[2]  # (1,3,H,W)
idx_max_sam = 2
target_mask = mask[0,idx_max_sam]
target_conf = conf[0,idx_max_sam].unsqueeze(0).unsqueeze(0)
target_conf = F.interpolate(target_conf, (H,W), mode='bilinear',
                            align_corners=False)[0,0]
sam_conf[i,k][target_mask] = (
    target_conf[target_mask] * cam_ms[i,k][target_mask].mean()
)
temp = sam_conf.max(dim=1)
pgt_sam = temp[1]
pgt_score = temp[0]
pgt_sam[pgt_score<0] = 20
```

- `mask/output_sam[0]` 是三个候选掩码，`conf/output_sam[2]` 是相应置信图；注释声称 shape 均为 `(1,3,H,W)`。
- `idx_max_sam = 2` **不是动态选最大置信候选**，而是固定取第三个候选；变量名容易让人误以为做了 argmax。
- `target_conf` 经两次 `unsqueeze` 变成 `[1,1,h_s,w_s]`，插值到 `[1,1,H,W]` 后再索引回 `[H,W]`。
- `cam_ms[i,k][target_mask].mean()` 是一个标量，表示候选区域对类别 $k$ 的平均 CAM 语义证据；它乘到掩码内每个像素，所以 SAM 置信保留像素差异，CAM 只对整个候选区域做统一缩放。
- `sam_conf=[B,C,H,W]` 初始为 `-1e5`。未被任何候选覆盖的位置保持大负数，避免在类别竞争中被误选。
- `max(dim=1)` 消去类别维 $C$：返回的 `values` 是 `[B,H,W]` 最大分数，`indices` 是 `[B,H,W]` 类别索引。
- 最大分数仍小于 0 的位置被设为类别 20。对 PASCAL VOC 来说这里是**背景索引**，不是 255 ignore。

### 8.4 论文叙事与代码真实实现的差异

- 论文层面可概括为“按 SAM 置信度聚合候选”，代码却固定选择第 3 个候选，并未逐样本比较三个候选的置信度。
- 乘法分数不是两张完整概率图逐像素相乘：CAM 先在掩码内求均值，变成区域标量；只有 SAM 置信度保留像素级变化。
- 伪标签链位于 `torch.no_grad()`，等价于显式停止梯度；后续学生只能学习目标，不能操纵置信分数。
- 未覆盖位置是背景 20，因此会参与后续监督；若设计意图是“不确定就忽略”，应改为 255 并检查交叉熵的 `ignore_index`，两者语义不同。

## 9. 概率校准与开放词表问题

温度缩放为：

$$
p_c=\operatorname{softmax}(z_c/T).
$$

**公式解释：** 对一个位置的类别 logit 向量 $z\in\mathbb R^C$，先把每个 $z_c$ 除以标量温度 $T$，再只沿类别维 $C$ 做 softmax；输出 $p\in[0,1]^C$ 且 $\sum_cp_c=1$。softmax 的分母会对全部 $C$ 类求和并消去求和索引，$p_c$ 表示校准后分给类别 $c$ 的概率。$T>1$ 让分布更平，$T<1$ 让分布更尖；类别排序不变，但最大概率和熵会改变。温度应在独立验证集拟合，不同分支可有不同 $T_m$。

常用诊断包括期望校准误差（Expected Calibration Error，**ECE**）、可靠性图和负对数似然。ECE 将样本按置信度分箱，比较每箱平均置信度与真实准确率；它依赖分箱方式，只能说明校准程度，不能证明预测本身正确。

OVS 中候选类别集合会改变 softmax 分母。即使原始余弦相似度不变，加入更多类别后最大概率和熵也会变化。因此阈值需在相同词表规模、seen/unseen 组成和背景提示设置下校准，不能从 20 类配置直接搬到几百类词表。

## 10. 选型指南

| 当前症状 | 优先考虑 | 不值得或需先检查 |
|---|---|---|
| 两分支在边界和物体内部明显互补 | 先做等权平均与熵差门控，再考虑学习式门控 | 若单分支几乎处处更好，门控只会增加复杂度 |
| 伪标签少量区域明显错误 | ignore 低置信像素或连续损失加权 | 先报告覆盖率，避免靠忽略大多数像素“提纯” |
| 学生把困难样本权重压到零 | 对置信权重 `detach`，或改由冻结教师估计 | 不要仅增加权重正则而忽略梯度路径 |
| SAM 边界好但类别不可靠 | 结构置信 × CAM 区域语义分数 | 检查 SAM 候选选择是否真是动态 argmax |
| OVS 更换词表后阈值失效 | 重新做温度与阈值校准，分 seen/unseen 统计 | 不直接比较不同词表下的 softmax 熵 |
| 背景占绝大多数 | 前景/背景分别归一化或按类加权 | 全局最大概率会让高置信背景淹没前景 |
| 不确定性来自边界模糊 | 增强稳定性、区域一致性或边界专用权重 | 单次熵无法区分数据不确定性与模型过度自信 |

最低成本基线应包含：单分支、等权平均、最大概率门控、熵门控，以及 oracle 逐像素选择上界。若 oracle 相比最强单分支也没有明显空间，说明“互补融合”本身并不是主要矛盾。

## 11. 论文与源码索引

- [[ComCD_paper_notes]]：熵差融合与学习式逻辑门控的两级设计。
- [[UGRL_paper_notes]]：不确定性如何同时控制任务损失和可靠像素对比学习。
- [[POT_paper_notes]]：熵权重如何调节原型最优传输一致性。
- [[S2C_paper_notes]]：SAM 结构置信与 CAM 区域语义置信的乘法聚合。
- [[CLIP-ES_paper_notes]]：低置信伪标签的硬筛选式训练。
- [S2C `update` 的多尺度 CAM 与 SAM 路径](https://github.com/sangrockEG/S2C/blob/102e14c690c8e3bce3d5ccd1ae7832145ce10b27/models/model_s2c.py#L189-L299)：从 CAM 生成带置信度的硬伪标签。
- [S2C CPM 损失](https://github.com/sangrockEG/S2C/blob/102e14c690c8e3bce3d5ccd1ae7832145ce10b27/models/model_s2c.py#L345-L360)：检查这些伪标签最终怎样产生学生梯度。

## 12. 当前整理结论

置信度重加权的核心选择不是“用最大概率还是熵”，而是四个更基础的问题：可靠性由谁估计、在哪个维度生效、它控制预测还是梯度、学生能否反向操纵它。阅读新论文时先标出权重 shape 和归一化维；修改模型时先检查分支校准、有效监督覆盖率、背景占比与 `detach/no_grad` 边界；所有方法都可放回“产生证据 → 估计可靠性 → 融合/筛选/加权 → 形成输出或损失”这一条数据流中理解。
