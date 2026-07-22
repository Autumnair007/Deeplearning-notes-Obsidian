---
type: operator-note
aliases:
  - Retrieval and Memory
  - 检索与记忆
tags:
  - research-operator
  - retrieval
  - memory-bank
  - cache
  - semantic-segmentation
  - weakly-supervised
  - open-vocabulary
status: in-progress
---

# Retrieval and Memory（检索与记忆）

> [!abstract] 本页定位
> 本页整理**如何让当前图像的patch、区域或query从外部视觉/语义记忆中检索邻居，再把邻居携带的类别或关系读回密集预测**。资料来自经典视觉表征，以及弱监督语义分割（Weakly-Supervised Semantic Segmentation, WSSS）与开放词汇语义分割（Open-Vocabulary Segmentation, OVS）论文。单篇论文笔记保存完整方法；本页关注可复用的键值接口、缓存生命周期、检索形式、工程风险和源码导航。

> [!tip] 基础机制入口
> 对比语言—图像预训练（Contrastive Language-Image Pre-training, CLIP）的共享空间、二范数（L2 norm）归一化和温度见 [[clip_notes]]；无标签自蒸馏第二版（self-DIstillation with NO labels v2, DINOv2）视觉表征见 [[dinov2_notes]]；区域池化与原型的相邻概念见 [[Pooling_and_Region_Aggregation]]、[[Prototype_Construction]]。检索结果怎样与文本类别激活图（Class Activation Map, CAM）组合见 [[Multi_Level_Fusion]]。

## 1. 这个算子解决什么问题？

大白话说，一个类别文本只有一个或少量向量，很难覆盖真实世界中同类物体的颜色、姿态、材质和视角。例如“dog”文本向量不一定能稳定匹配一只被遮挡、背对镜头或卡通风格的狗。检索方法提前保存多种视觉外观；当前patch先问“我像哪些记忆”，再根据这些记忆绑定的类别信息投票。

常见症状包括：

- 直接patch—text相似度只能激活典型外观；
- 训练类别内变化大，单一类别原型覆盖不足；
- 测试词与训练/参考词不完全相同，需要借助同义词或参考关系迁移；
- 类别无关分割能得到完整区域，却不知道区域叫什么；
- 希望冻结大骨干，只训练很小的读写模块。

专业地说，检索把封闭的一次分类改成：

$$
\text{query}\rightarrow\text{key matching}
\rightarrow\text{value readout}\rightarrow\text{dense prediction}.
$$

**公式解释：** 这是一条检索流程式，不是矩阵运算。query 决定“拿什么去查”，key matching 产生对记忆条目的权重，value readout 把条目权重转换成类别或属性证据，最后恢复为密集预测；此式本身没有求和或维度消去。

这里的核心选择是：查询粒度、键的特征空间、值携带的语义、记忆是否更新，以及如何拒绝没有可靠邻居的查询。

> [!note] 我的理解｜检索不是自动支持开放词汇
> 如果值矩阵只保存固定训练类别one-hot，缓存最多只能输出这些类别；只有当值能由测试文本动态重算，或通过参考标签关系映射到新词表时，检索才真正支持新增类别。

### 1.1 哪些相似操作不属于本算子？

- `self-attention` 的Key/Value只来自当前样本，通常是当前前向中的交互，不等于持久记忆库。
- 类别分类头的权重可被解释成原型，但若没有邻居选择、外部状态或显式读出，不必都叫检索。
- 多层特征拼接属于 [[Multi_Level_Fusion]]。
- 同图位置传播属于 [[Spatial_Propagation]]；它查询的是空间邻居，而非外部样本记忆。

## 2. 统一输入输出张量

查询特征：

$$
Q\in\mathbb{R}^{B\times N\times D},
$$

**公式解释：** $B$ 是批量大小，$N$ 是 patch/区域数，$D$ 是查询特征维；$Q[b,n,d]$ 是第 $b$ 张图第 $n$ 个查询的第 $d$ 个分量。这里只声明输入 shape，没有运算或维度消去。键和值：

$$
K\in\mathbb{R}^{U\times D},
\qquad
V\in\mathbb{R}^{U\times C}.
$$

**公式解释：** $U$ 是记忆条目数，$K[u,:]\in\mathbb R^D$ 是第 $u$ 个视觉/语义键，$V[u,:]\in\mathbb R^C$ 是该条目携带的类别证据。键与查询共享特征维 $D$，值把每条记忆映射到 $C$ 类；此式只声明接口。

先沿特征维做L2归一化：

$$
\hat Q=\frac{Q}{\|Q\|_2+\varepsilon},
\qquad
\hat K=\frac{K}{\|K\|_2+\varepsilon}.
$$

**公式解释：** 对 $Q$ 的每个 `[D]` 查询向量和 $K$ 的每个 `[D]` 键向量分别沿特征维做 L2 归一化。输出 `hat Q=[B,N,D]`、`hat K=[U,D]`，shape 不变且向量模长约为 1；$\varepsilon$ 防止零向量除零。这样后续点积只比较方向。

再检索：

$$
A=\hat Q\hat K^T
\in\mathbb{R}^{B\times N\times U}.
$$

**公式解释：** `[B,N,D] @ [D,U]` 在特征维点积并消去 $D$，输出 `[B,N,U]`。$A[b,n,u]$ 表示第 $b$ 张图第 $n$ 个查询与第 $u$ 条记忆键的余弦相似度；记忆维 $U$ 被保留，供后续选择或加权。

将相似度变成权重并读出：

$$
W=\operatorname{softmax}(A/\tau;\text{dim}=U),
$$

**公式解释：** $A=[B,N,U]$ 除以温度 $\tau$ 后，softmax 沿记忆维 $U$ 归一化，输出 $W$ shape 不变。$W[b,n,u]$ 是查询 $n$ 分给第 $u$ 条记忆的读取比例，并满足对 $u$ 求和为 1；这里没有消去维度。

$$
S=WV\in\mathbb{R}^{B\times N\times C}.
$$

**公式解释：** `[B,N,U] @ [U,C]` 沿记忆条目维相乘求和并消去 $U$，得到 `[B,N,C]`。$S[b,n,c]$ 是全部记忆条目对查询 $n$ 属于类别 $c$ 的加权投票；具体命中了哪条记忆不再保留，因此调试时应另存 Top-k 索引与权重。

数字例子：

```text
Q:   [2,400,512]
K^T: [512,312]
A:   [2,400,312]
V:   [312,21]
S:   [2,400,21]
```

若输出要恢复成空间CAM：

```text
[B,N,C]
→ permute(0,2,1)
→ [B,C,N]
→ reshape(B,C,H',W')
→ [B,C,H',W']
→ bilinear interpolate
→ [B,C,H,W]
```

这一步只恢复空间布局；记忆检索发生在展平的patch/区域层面。

## 3. 代表模型与论文

| 论文/模型 | 任务与起点 | 原方法存在的问题 | 具体做法 | 与检索/记忆的关系 |
|---|---|---|---|---|
| [[clip_notes]] | 图文对比预训练；全局图像—文本 | 类别文本难覆盖全部视觉外观 | 图像与文本编码到共享空间，通过归一化相似度匹配 | 提供检索空间，但本身没有密集外部视觉缓存 |
| [[dinov2_notes]] | 自监督视觉表征 | 缺少开放文本接口 | 学习空间结构清晰、可迁移的视觉特征 | 常作为视觉键或区域检索编码器，类别需由别处提供 |
| [[maskformer_notes]] | 掩码分类；对象query | 像素分类缺少对象级聚合 | query与像素嵌入交互并输出掩码和类别 | query可视作当前图像内的可学习槽位，不是持久外部记忆 |
| [[OpenSeg_paper_notes]] | OVS；图像—标题监督 | patch级对齐难获得完整对象 | 先产生类别无关掩码并池化区域，再与标题词对比接地 | 区域嵌入是更稳定的检索单元，可作为参考片段 |
| [[DiCLIP_paper_notes]] | WSSS；冻结CLIP与稳定扩散 | 单一文本向量不能覆盖类内外观，CLIP patch空间感知不足 | 用稳定扩散生成单类图；前景/背景区域平均池化并按类聚类成视觉键，值保存类别证据；真实patch同时走静态缓存和由缓存初始化的动态适配器 | 完整展示离线建库、静态读出、可学习检索与伪标签监督 |
| [[ReME_paper_notes]] | 免训练OVS；类别无关片段 | 直接跨模态匹配受模态错位与低质量参考集限制 | 从真实图像构造片段—文本参考集，用视觉模态内相似度过滤错配；测试片段先检索参考片段，再通过参考片段—标签关系和标签—测试词相似度完成两阶段读出 | 用参考集桥接视觉与文本，支持测试词表变化 |
| [[Talk2DINO_paper_notes]] | OVS；DINOv2视觉空间 + CLIP文本 | 两个编码器不共享空间 | 学习轻量非线性映射，把文本嵌入送入DINO视觉空间，并用语义锚点与区域对应训练 | 更接近“可学习语义锚点”，可与视觉近邻/原型检索组合 |
| [[CorrCLIP_paper_notes]] | 免训练OVS；CLIP分类 | CLIP patch关系错误 | 以DINO相似度和SAM范围重建视觉特征，再与文本分类 | 提醒先保证query空间可靠，再谈外部近邻质量 |

## 4. 常见实现形式

| 形式 | 查询/记忆 | 更新方式 | 优点 | 局限 | 代表工作 |
|---|---|---|---|---|---|
| 静态键值缓存 | patch → 离线视觉原型 | 不更新 | 稳定、可复现、无训练读出 | 覆盖不足与域偏差 | [[DiCLIP_paper_notes]] |
| 缓存初始化适配器 | patch → 线性层隐单元 | 梯度更新 | 保留先验并适配目标数据 | 可能遗忘缓存语义 | [[DiCLIP_paper_notes]] |
| 先进先出/指数移动平均记忆库（First-In First-Out, FIFO / Exponential Moving Average, EMA） | 当前特征 → 历史样本 | 队列或指数移动平均 | 训练中持续扩充负样本/原型 | 陈旧特征、分布式同步复杂 |
| 类别原型库 | 像素/区域 → 类中心 | 均值、聚类或可学习 | 容量小、解释直接 | 多模态类别被压成少量中心 | [[Prototype_Construction]] |
| 参考集关系检索 | 测试片段/词 → 参考片段/词 | 通常固定 | 可通过关系桥接新词表 | 数据质量与索引成本决定上限 | [[ReME_paper_notes]] |
| Top-k稀疏检索 | 任意query → 最近k项 | 每次查询筛选 | 降噪、减少读出成本 | 早期错检后无法恢复 |

## 5. 各种实现怎样工作？

### 5.1 静态键值检索

直觉：提前做一本“视觉词典”。键回答“像什么外观”，值回答“这个外观支持什么类别”。

数据流：

```text
离线图像/区域
→ 特征提取
→ 掩码平均池化
→ 每类聚类/去重
→ K视觉键 + V类别值

当前patch Q
→ 与K相似度
→ softmax/Top-k
→ 加权读取V
→ 类别CAM
```

若每类有 $E$ 个前景原型、共有 $C_f$ 个前景类，再有 $E_b$ 个背景原型，则：

$$
U=C_fE+E_b.
$$

**公式解释：** $C_f$ 是前景类别数，$E$ 是每类保存的前景原型数，$E_b$ 是额外背景原型数。乘法得到全部前景条目 $C_fE$，再加背景条目，输出标量缓存容量 $U$；这不是张量运算，只计算条目数量。

每类多个键保留类内多样性；背景也需要多个键，因为“非目标”通常比任何单一前景类别更复杂。

**适用**：冻结骨干、外部视觉样本可靠、希望低成本补充文本。**局限**：固定one-hot值不能输出建库时没有的类；合成图缓存可能与真实域存在偏差。

### 5.2 Top-k、温度与拒识

先取邻居集合 $\mathcal N_k(q)$，再在集合内归一化：

$$
w_u=\frac{\exp(\cos(q,k_u)/\tau)}
{\sum_{v\in\mathcal N_k(q)}\exp(\cos(q,k_v)/\tau)},
\quad u\in\mathcal N_k(q),
$$

**公式解释：** $q,k_u\in\mathbb R^D$，余弦相似度先在特征维 $D$ 上计算并消去它；只保留 Top-k 邻居集合 $\mathcal N_k(q)$。分母沿邻居索引 $v$ 求和，$w_u$ 是查询分给邻居 $u$ 的归一化权重，所有保留邻居权重和为 1；$\tau$ 控制分布尖锐度。

$$
s_c=\sum_{u\in\mathcal N_k(q)}w_uv_{u,c}.
$$

**公式解释：** 对固定类别 $c$，把每个保留邻居的读取权重 $w_u$ 乘以该条目的类别值 $v_{u,c}$，再沿邻居维 $u$ 求和并消去它。输出标量 $s_c$ 是当前查询对类别 $c$ 的缓存投票。温度 $\tau$ 小，权重集中到最近邻；温度大，更多记忆参与。

仅有softmax会强迫每个query从某条记忆读出类别，即使所有相似度都很低。可增加拒识条件：

$$
\max_u\cos(q,k_u)<\delta
\Rightarrow \text{unknown/background/不使用缓存}.
$$

**公式解释：** 先对所有缓存条目 $u$ 的余弦相似度取最大值，消去记忆维，只保留“最近邻有多近”的标量。若该值仍小于阈值 $\delta$，说明没有可靠邻居，输出应转为 unknown、background 或跳过缓存；该式不产生类别分数。

开放世界场景尤其要报告域外query的最大相似度分布，而不只看已知类准确率。

### 5.3 类别平衡与前景/背景双分支

若某类有更多缓存条目，它更容易占据Top-k，并在softmax分母中形成数量先验。常见处理：

- 每类固定相同原型数；
- 先做类内Top-k，再做类间竞争；
- 对值按类频率校正；
- 前景与背景分开检索、分别校准容量与温度。

背景抑制可写成：

$$
S_{fg}'=S_{fg}\odot(1-S_{bg}).
$$

**公式解释：** `S_fg=[B,N,C_f]`，`S_bg=[B,N,1]`；$1-S_{bg}$ 在前景类别维广播，与每类前景分数逐元素相乘，不消去维度。$S'_{fg}[b,n,c]$ 是经同位置背景置信度抑制后的前景类 $c$ 分数。该式要求背景值已在 `[0,1]` 且确实表示背景概率。

### 5.4 缓存初始化的可学习适配器

静态检索：

$$
S=\sigma(QK^T)V.
$$

**公式解释：** `Q @ K^T` 先消去特征维 $D$，得到查询对 $U$ 个固定键的响应；$\sigma$ 逐元素激活，随后乘 `V=[U,C]` 消去原型维 $U$，输出类别分数 `S=[B,N,C]`。$S[b,n,c]$ 是静态缓存对 patch $n$ 的类别 $c$ 投票。

可改写为两层网络：

$$
S_d=\sigma(QW_1^T+b_1)W_2+b_2,
$$

**公式解释：** 若 `W_1=[U,D]`，第一层用 `[B,N,D] @ [D,U]` 消去特征维，输出 $U$ 个原型隐单元；$b_1$ 广播后经 $\sigma$，再乘 `W_2=[U,C]` 消去隐单元维，得到 `S_d=[B,N,C]`，最后加类别偏置 $b_2$。以 $W_1\leftarrow K$、$W_2\leftarrow V$ 初始化时等价于缓存读出；训练后权重会偏离严格键值解释。

**适用**：合成/外部缓存与目标域有偏差，但有训练信号可适配。**局限**：偏置、Dropout、非线性和权重更新都会使它偏离严格最近邻解释；需要保留静态分支做消融。

### 5.5 参考集桥接检索

ReME把一次困难的直接跨模态比较拆成两种模态内检索。设：

- 测试片段 $S_{test}\in\mathbb{R}^{K\times D_v}$；
- 参考片段 $S_{ref}\in\mathbb{R}^{U\times D_v}$；
- 参考标签 $L_{ref}\in\mathbb{R}^{R\times D_t}$；
- 测试类别 $L_{test}\in\mathbb{R}^{C\times D_t}$；
- 参考片段—标签关系 $O_{ref}\in\mathbb{R}^{U\times R}$。

视觉侧：

$$
A_1=\operatorname{softmax}(S_{test}S_{ref}^T)O_{ref}
\in\mathbb{R}^{K\times R}.
$$

**公式解释：** `S_test=[K,D_v]` 与 `S_ref^T=[D_v,U]` 相乘，消去视觉维 $D_v$，得到每个测试片段对 $U$ 个参考片段的相似度；softmax 沿参考片段维归一化。再乘 `O_ref=[U,R]` 时消去 $U$，输出 `A_1=[K,R]`；$A_1[k,r]$ 是测试片段 $k$ 对参考标签 $r$ 的视觉侧证据。

文本侧：

$$
A_2=\operatorname{softmax}(L_{ref}L_{test}^T)
\in\mathbb{R}^{R\times C}.
$$

**公式解释：** $L_{ref}=[R,D_t]$ 与 $L_{test}^T=[D_t,C]$ 相乘，在文本特征维 $D_t$ 上点积并消去它，得到 `[R,C]`；softmax 按实现指定的目标类别维归一化。$A_2[r,c]$ 表示参考标签 $r$ 与测试类别 $c$ 的文本关系。

最终：

$$
P_{seg}=A_1A_2\in\mathbb{R}^{K\times C}.
$$

**公式解释：** $A_1=[K,R]$ 与 $A_2=[R,C]$ 相乘，沿参考标签维 $R$ 求和并消去它，输出 `[K,C]`。$P_{seg}[k,c]$ 表示测试片段 $k$ 通过“相似参考片段 → 参考标签 → 测试类别”两跳桥接后对类别 $c$ 的分数。

若类别无关掩码为 $M_{seg}\in\mathbb{R}^{K\times H\times W}$，回填像素为：

$$
P[x,y,c]=\sum_{k=1}^{K}P_{seg}[k,c]M_{seg}[k,x,y].
$$

**公式解释：** 对固定像素 $(x,y)$ 和类别 $c$，把每个片段的类别分数 $P_{seg}[k,c]$ 乘以该片段在此像素的掩码权重 $M_{seg}[k,x,y]$，再沿片段维 $K$ 求和并消去它。输出 `P=[H,W,C]`；$P[x,y,c]$ 是所有重叠区域对该像素类别 $c$ 的加权投票。

## 6. 记忆库生命周期

```text
确定数据、骨干、预处理与版本
→ 提取候选键
→ 质量筛选、去重、类平衡
→ 构造值与元数据
→ 固定缓存并验证查询兼容性
→ 静态检索或训练中更新
→ 版本变化后重编码/失效
```

至少保存：来源图像/生成提示、区域掩码来源、类别与背景定义、特征编码器commit、输入归一化、键是否L2归一化、聚类参数和随机种子。

骨干微调后，旧键会变成“陈旧特征”。安全策略只有两类：冻结query编码器保持空间不变，或按版本周期性重编码记忆。测试图像及其真值不得进入建库，否则会产生数据泄漏。

## 7. 官方仓库静态分析：DiCLIP静态与动态缓存

### 7.1 仓库与固定版本

- 官方仓库：[zwyang6/DiCLIP](https://github.com/zwyang6/DiCLIP)
- commit：[`1c3f6ff7d4fde2afff32d527d78b28d119583602`](https://github.com/zwyang6/DiCLIP/tree/1c3f6ff7d4fde2afff32d527d78b28d119583602)
- 缓存生成：[maintain_kv_cache/both_fore_bkg/generate_kv_cache.py#L80-L160](https://github.com/zwyang6/DiCLIP/blob/1c3f6ff7d4fde2afff32d527d78b28d119583602/maintain_kv_cache/both_fore_bkg/generate_kv_cache.py#L80-L160)
- 静态检索：[model/model_diclip.py#L114-L146](https://github.com/zwyang6/DiCLIP/blob/1c3f6ff7d4fde2afff32d527d78b28d119583602/model/model_diclip.py#L114-L146)
- 动态适配器：[model/model_diclip.py#L18-L42](https://github.com/zwyang6/DiCLIP/blob/1c3f6ff7d4fde2afff32d527d78b28d119583602/model/model_diclip.py#L18-L42)
- 在线调用与detach：[model/model_diclip.py#L148-L185](https://github.com/zwyang6/DiCLIP/blob/1c3f6ff7d4fde2afff32d527d78b28d119583602/model/model_diclip.py#L148-L185)

### 7.2 离线建库调用链

```text
生成/缓存数据图像 + 区域labels
→ CLIP/视觉相关性增强（Visual Correlation Enhancement, VCE）patch特征（去类别标记，即class token/CLS token）
→ labels最近邻缩放到patch网格
→ 前景/背景分别掩码平均池化
→ 每类KMeans聚类
→ 拼接 cache_keys [U,512]
→ 拼接 cache_values [U,C]
→ torch.save
```

关键代码：

```python
labels_down = F.interpolate(
    labels.unsqueeze(0),
    size=[args.resize_size // 16, args.resize_size // 16],
    mode='nearest'
)[0]
masks_binary = labels_down.reshape(1, -1).unsqueeze(-1) > 0

key = feats[:, 1:, :] * masks_binary
key_norm = key.sum(1) / masks_binary.sum(1)

key_bkg = feats[:, 1:, :] * (~masks_binary)
key_bkg_norm = key_bkg.sum(1) / (~masks_binary).sum(1)
```

逐步解释：

1. 离散区域标签使用最近邻缩放，避免双线性产生非整数类别。
2. `reshape(1,-1).unsqueeze(-1)` 把二维掩码变为 `[1,N,1]`，可广播乘 `[1,N,D]` patch特征。
3. `sum(1) / count` 在位置维 $N$ 做掩码平均池化，输出 `[1,D]` 区域键。
4. 前景与补集背景分别建键；代码仅显式检查前景非空，背景分母若为0仍可能出问题。
5. 本段把变量命名为 `key_norm`，实际执行的是区域平均，并没有在保存前显式做L2归一化。

随后每类执行KMeans，聚类中心拼成 `cache_keys`，对应one-hot类别值拼成 `cache_values`。背景类在代码中使用20个中心，前景类使用参数 `args.num_cluster`。

### 7.3 静态检索不是简单的 `softmax(QK^T)V`

代码先用CLS对全部键产生全局权重：

```python
prob = image_features[:, :1, :] @ text_features.t()
prob = (prob * 2).softmax(-1)
w = prob / prob.mean(-1, keepdim=True)
```

这里变量 `text_features` 实际来自 `kv_cache[0]`，即视觉缓存键。`image_features[:,:1,:]` 只取CLS，得到 `[B,1,U]`；因此 `w` 是每张图共享给所有patch的全局原型先验，不是逐patch检索权重。

接着做逐元素特征交互：

```python
feats = image_features.reshape(b, n_i, 1, c) \
      * text_features.reshape(1, 1, n_t, c)
feats *= w.unsqueeze(-1)
redundant_feats = feats.mean(2, keepdim=True)
feats = feats - redundant_feats
similarity = F.relu(feats.sum(-1))
```

形状变化：

```text
image_features: [B,N_i,D] → [B,N_i,1,D]
cache keys:     [U,D]     → [1,1,U,D]
广播逐元素乘                 [B,N_i,U,D]
沿U求均值并相减              [B,N_i,U,D]
沿D求和                       [B,N_i,U]
```

减去原型维均值是“特征手术”式冗余抑制；它已经超出标准余弦检索。

更关键的是，函数开头读入的 `kv_cache[1]` 没有直接用于静态读出。代码重新用缓存键与整体文本特征计算值：

```python
values = text_features @ text_features_21.transpose(1, 0)
values[values == 0] = float('-inf')
value = values.softmax(0)
similarity = similarity[:, 1:, :] @ value.unsqueeze(0)
```

`values` 是 `[U,C]`；`softmax(0)` 沿记忆条目维 $U$ 做归一化，即每个类别在全部缓存键之间分配权重，而不是每条键在类别维归一化。随后去掉CLS，`[B,N,U] @ [U,C] → [B,N,C]`。

### 7.4 背景抑制与数值风险

```python
diff_maps = (similarity - similarity.min(1, keepdim=True)[0]) / (
    similarity.max(1, keepdim=True)[0]
    - similarity.min(1, keepdim=True)[0]
)
fore_maps = diff_maps[:, :, 1:]
bkg_maps = diff_maps[:, :, 0].unsqueeze(-1)
fuse = F.relu(fore_maps * (1 - bkg_maps))
```

min-max沿patch维 `dim=1`，所以每张图的每个类别独立把空间最小/最大值拉到0/1。分母没有epsilon；若某类所有patch分数相同，会除零。背景通道 `[B,N,1]` 广播抑制全部前景类。

### 7.5 动态适配器怎样由缓存初始化？

```python
self.fc1 = nn.Linear(in_features, hidden_features)
k_prompt = trunc_normal_(torch.zeros(hidden_features, in_features))
k_prompt[:idx] = cache_key
self.fc1.weight = nn.Parameter(k_prompt.clone())

self.fc2 = nn.Linear(hidden_features, out_features)
v_prompt = trunc_normal_(torch.zeros(hidden_features, out_features))
v_prompt[:idx] = cache_value
self.fc2.weight = nn.Parameter(v_prompt.t().clone())
```

若 `K:[U,512]`、`V:[U,C]` 且隐藏宽度为312，则第一层权重 `[312,512]` 的前 $U$ 行由键初始化；第二层权重 `[C,312]` 的前 $U$ 列由值初始化。剩余提示槽随机截断正态初始化。

前向实际为：

```python
x = self.fc1(x)
x = F.relu(x)
x = self.drop(x)
x = self.fc2(x)
x = self.drop(x)
```

- 构造函数传入的高斯误差线性单元（Gaussian Error Linear Unit, GELU）`act_layer=nn.GELU` 没有在前向使用，实际固定为线性整流单元（Rectified Linear Unit, ReLU）。
- 两个 `nn.Linear` 的bias保持默认可学习初始化，不来自缓存。
- 两层后都用Dropout；这与严格确定性的键值读出不同。
- `get_param_groups()` 明确把适配器参数加入优化器，因此键/值初始化会被训练更新。
- 在线函数最终返回 `diff_maps.detach()` 作为伪标签路径，而动态适配器与分割头本身参与训练。

> [!note] 我的理解｜变量名会误导，shape更可靠
> 静态函数里的 `text_features` 实际是视觉缓存键，加载的缓存值没有直接使用；代码重新根据键—文本相似度构造值。动态分支才把保存的 `cache_value` 用来初始化第二层。理解这段实现必须跟shape和赋值走，不能只看变量名或论文中的标准KV公式。

## 8. 选型指南

| 当前问题 | 优先考虑 | 不值得先做 |
|---|---|---|
| 文本CAM只覆盖典型外观 | 每类多视觉原型的静态缓存 | 只增加同义提示数量 |
| 缓存与真实域有差异且有训练数据 | 缓存初始化适配器 + 静态教师 | 完全丢弃缓存随机训练分类头 |
| 测试词表会变化 | 参考集关系桥接或动态文本值 | 固定训练类one-hot缓存 |
| 类别长尾、背景条目过多 | 类平衡采样、类内Top-k、前/背景分支 | 全库一次softmax不校正数量 |
| 没有可靠近邻 | 最大相似度拒识/unknown | 强制softmax输出某个类 |
| 只想减少特征噪声 | 先改 [[Attention_and_Affinity_Refinement]] | 引入大型外部库 |
| 推理延迟敏感 | 聚类原型、小缓存、分块Top-k | 生成完整 `[B,N,U]` 大矩阵 |
| 只有同图邻域需要补全 | [[Spatial_Propagation]] | 建跨样本记忆库 |

## 9. 工程检查与消融

1. 报告缓存大小 $U$、每类条目数、Top-k、温度、查询延迟、峰值显存和分块策略。
2. 检查query与key是否来自同一编码器、预处理、checkpoint和归一化方式。
3. 保存Top-k来源、类别、相似度和区域可视化，不能只看最终mIoU。
4. 比较随机缓存、同容量类平衡缓存、完整缓存，证明收益来自知识质量而非条目数量。
5. 分别报告静态缓存、随机初始化适配器、缓存初始化适配器和冻结适配器。
6. 检查近邻纯度、长尾召回、前/背景检索比例和域外最大相似度。
7. 做“移除正确类别记忆”压力测试，观察系统是否拒识还是从错误邻居给出高置信预测。
8. 任何min-max或L2归一化都要有epsilon，并明确归一化维度。
9. 建库数据、生成提示、聚类和模型版本必须纳入实验配置；缓存不是无关紧要的中间文件。

## 10. 论文与源码索引

### 论文双链

- [[DiCLIP_paper_notes]]：扩散生成图、前/背景视觉缓存、静态与动态检索。
- [[ReME_paper_notes]]：真实片段—文本参考集、模态内清洗和两阶段检索。
- [[Talk2DINO_paper_notes]]：CLIP文本到DINO视觉空间的轻量映射与语义锚点。
- [[OpenSeg_paper_notes]]：区域级视觉—文本接地，适合理解检索单元粒度。
- [[CorrCLIP_paper_notes]]：检索前query空间的结构校准。
- [[Prototype_Construction]]：类别原型构造、更新与多中心表示。

### DiCLIP源码入口

- [前景/背景键提取](https://github.com/zwyang6/DiCLIP/blob/1c3f6ff7d4fde2afff32d527d78b28d119583602/maintain_kv_cache/both_fore_bkg/generate_kv_cache.py#L92-L123)：掩码缩放和区域平均池化。
- [按类KMeans与缓存保存](https://github.com/zwyang6/DiCLIP/blob/1c3f6ff7d4fde2afff32d527d78b28d119583602/maintain_kv_cache/both_fore_bkg/generate_kv_cache.py#L132-L160)：键值容量和背景中心数。
- [静态检索真实实现](https://github.com/zwyang6/DiCLIP/blob/1c3f6ff7d4fde2afff32d527d78b28d119583602/model/model_diclip.py#L114-L146)：CLS全局权重、特征手术、重算值和背景抑制。
- [KV适配器初始化与前向](https://github.com/zwyang6/DiCLIP/blob/1c3f6ff7d4fde2afff32d527d78b28d119583602/model/model_diclip.py#L18-L42)：键值如何进入两层线性权重。
- [训练参数分组](https://github.com/zwyang6/DiCLIP/blob/1c3f6ff7d4fde2afff32d527d78b28d119583602/model/model_diclip.py#L73-L83)：确认动态适配器会训练。
- [静态/动态分支在线调用](https://github.com/zwyang6/DiCLIP/blob/1c3f6ff7d4fde2afff32d527d78b28d119583602/model/model_diclip.py#L148-L185)：CAM融合、reshape与detach边界。

## 11. 当前整理结论

检索与记忆的核心选择可以统一为：

$$
\boxed{
\text{查询粒度}
\rightarrow
\text{同空间的键}
\rightarrow
\text{可解释的值}
\rightarrow
\text{受控读出与拒识}
\rightarrow
\text{缓存版本管理}
}.
$$

**公式解释：** 这不是张量运算式，而是检索模块的五步检查顺序：先确定查询单位，再确认 query/key 共享特征空间，随后确认 value 的语义，检查 Top-k、归一化与拒识，最后记录缓存来源和版本。任何一步不清楚，最终类别分数都难以解释。

阅读新论文时，应追问键来自真实图、生成图还是训练队列，值是固定one-hot、软标签还是可随测试词表重算的关系；修改模型时先检查query/key空间兼容、类平衡、背景容量、归一化维度和无邻居时的行为。静态缓存、动态适配器与参考集桥接都能写成“匹配键—读取值”，但它们的开放词汇能力、训练边界和数据风险并不相同。
