---
type: operator-note
aliases:
  - Background Handling
  - Unknown Handling
  - 背景与未知类处理
tags:
  - research-operator
  - background
  - unknown
  - open-world
  - weakly-supervised-segmentation
  - open-vocabulary-segmentation
status: in-progress
---

# Background and Unknown Handling（背景与未知类处理）

## 1. 本页定位

本页整理闭集语义分割、弱监督语义分割（Weakly Supervised Semantic Segmentation，**WSSS**）、开放词汇分割（Open-Vocabulary Segmentation，**OVS**）和开放集分割中，怎样定义并实现 background、unknown、ignore、void 和 no-object。重点不是把所有“非前景”统一成第 0 类，而是回答：背景分数从哪里来、未知物体是否需要单独拒识、伪标签不确定区是否参与梯度、类别集合变化后阈值为什么失效，以及代码中的整数索引到底代表什么。

完整方法回到 [[CLIP-ES_paper_notes]]、[[Talk2DINO_paper_notes]]、[[SSR_paper_notes]]、[[ComCD_paper_notes]]、[[POT_paper_notes]]、[[DiCLIP_paper_notes]] 和 [[S2C_paper_notes]] 阅读。本页负责统一语义和数据流；伪标签筛选见 [[Pseudo_Label_Refinement]]，置信度估计见 [[Confidence_Reweighting]]，背景文本设计见 [[Prompt_Construction]]。

> [!abstract] 一句话直觉
> “不属于当前前景类”至少有三种原因：它真是任务背景、它是词表外但有意义的物体、或者模型暂时不确定。三者若都压成背景，模型会把未知物体当负样本，也会把不确定伪标签学成确定错误。

## 2. 这个算子解决什么问题

### 2.1 大白话解释

前景类可以枚举为人、车、狗；背景却可能是天空、墙、路、树影、未标注物体以及当前词表没覆盖的新类别。闭集数据集允许把这些统一成 background，但 OVS/开放集任务可能希望模型以后还能认出其中的未知物体。下文中的 CAM 首次出现时指类别激活图（Class Activation Map，**CAM**）。

该算子通常位于两处：

```text
视觉/文本前景分数
→ 构造或预测背景分数
→ 前景、背景、unknown 竞争
→ 输出类别/拒识结果

CAM/教师概率
→ 前景阈值、背景阈值、ignore 区间
→ 生成训练伪标签
→ 只让可靠位置进入损失
```

第一条是推理语义决策，第二条是训练监督策略。`ignore` 只表示“这次不学”，不是一种真实世界语义；把 ignore 当 background 会使漏检前景产生错误负梯度。

### 2.2 专业表述

给定 $C$ 个前景类别分数：

$$
S_{fg}\in\mathbb R^{B\times C\times H\times W}.
$$

**公式解释：** $B,C,H,W$ 分别表示 batch、前景类别和空间尺寸；$S_{fg}[b,c,h,w]$ 是第 $b$ 张图位置 $(h,w)$ 对前景类别 $c$ 的分数。该式只声明输入，没有求和或维度消去；这些分数可以是 logit、余弦相似度、CAM 或概率，但后续背景公式必须知道它们属于哪一种尺度。

若背景分数为 $S_{bg}\in\mathbb R^{B\times1\times H\times W}$，可沿类别维拼接：

$$
S=\operatorname{Concat}_{C}(S_{fg},S_{bg})
\in\mathbb R^{B\times(C+1)\times H\times W}.
$$

**公式解释：** `S_fg=[B,C,H,W]` 与 `S_bg=[B,1,H,W]` 必须具有相同 batch 和空间网格；Concat 只沿类别通道并排放置，不做数值相加，也不消去维度，输出 `S=[B,C+1,H,W]`。$S[b,c,h,w]$ 的通道语义取决于拼接顺序：背景可位于第 0 通道，也可位于最后一通道，target 索引必须与此一致。

若还显式区分 unknown，可增加独立 unknown logit，或让拒识函数在 $C+1$ 个已知类别之外输出特殊标签。二者不同：独立 logit 通常需要 unknown 监督；后处理拒识只需要可校准的不确定性分数。

### 2.3 哪些问题不是它单独负责的

- CAM 只激活目标的一小部分：背景规则不能补全缺失前景，需要传播、区域或伪标签细化。
- 前景与背景分数不在同一尺度：先做校准和特征/空间对齐，不能直接 concat 后 softmax。
- 未知类没有训练样本：显式 unknown 头未必能学到开放世界拒识，只会学提供过的代理异常。
- 模型高置信地错分未知物体：最大概率阈值无法保证识别，需要能量、特征距离、外部分布数据或开放集训练。

> [!note] 我的理解｜先写语义，再写整数
> `0`、`20`、`255` 本身没有固定含义。先写这个位置是背景、未知、void、ignore 还是 no-object，再检查数据集、拼接顺序和损失函数怎样编码它。很多训练错误不是公式错，而是两个模块对同一个整数的语义约定不同。

## 3. 四种任务设定必须区分

| 设定 | 测试候选类别 | 词表外物体要求 | background 含义 |
|---|---|---|---|
| 闭集语义分割 | 与训练类别相同 | 不要求单独识别 | 数据集定义中非前景区域的统一类 |
| 零样本/开放词汇分割 | 可加入训练未见但测试时给出名称的类 | 给出名称后应能分类 | 候选词表外区域常仍并入背景 |
| 开放集分割 | 存在未提供名称的物体 | 应拒识为 unknown | background 与 unknown 应分开评价 |
| 开放世界分割 | unknown 后续可能被命名和增量学习 | 发现、拒识并持续学习 | 随任务扩展而变化，需保存历史语义 |

OVS 的“unseen class”通常在测试时已经给出文本名称，因此它不等同于开放集中的“完全未知”。一个模型能通过文本分割 unseen 类，不代表它能把任意词表外物体拒识成 unknown。

## 4. 统一实现形式

### 4.1 前景补集

最简单背景分数为：

$$
S_{bg}=1-\max_{c=1,\ldots,C}S_{fg,c}.
$$

**公式解释：** 输入 `S_fg=[B,C,H,W]`，$\max_c$ 在每个像素沿前景类别维 $C$ 取最大值并消去该维，若保留单通道则结果为 `[B,1,H,W]`；再逐元素用 1 相减，shape 不变。$S_{bg}[b,0,h,w]$ 表示“该像素没有任何强前景响应”的补集分数。该式要求前景值已处于可解释的 `[0,1]` 范围；对任意 logit 或逐类独立 min-max CAM 直接使用时，概率含义不成立。

数字例子：某像素 3 个前景分数为 `[0.9,0.2,0.1]`，最大值 0.9，背景为 0.1；若为 `[0.2,0.15,0.1]`，背景为 0.8。类别维被消去，所以该规则只记住第一名前景，不关心第二名是否接近。

### 4.2 显式背景 logit

模型也可以直接预测背景，并在完整类别维归一化：

$$
P_{b,:,h,w}=\operatorname{softmax}_{c}
\bigl([Z_{fg},Z_{bg}]_{b,:,h,w}\bigr).
$$

**公式解释：** `Z_fg=[B,C,H,W]` 与 `Z_bg=[B,1,H,W]` 先拼成 `[B,C+1,H,W]`；softmax 对每个 $(b,h,w)$ 沿 $C+1$ 个类别求指数和并在分母中归约候选索引，但输出仍保留完整类别维。$P[b,c,h,w]$ 是前景或背景通道 $c$ 的归一化概率。背景头可学习上下文，但需要可靠背景监督，并可能把训练中未见的真实物体吸入背景。

### 4.3 多背景文本或视觉原型

设 $M$ 个背景原型 $B\in\mathbb R^{M\times D}$、视觉 token $P\in\mathbb R^{B_s\times N\times D}$，可取最大背景匹配：

$$
S_{bg}[b,n]=\max_{m=1,\ldots,M}
\hat P[b,n,:]^T\hat B[m,:].
$$

**公式解释：** 每个视觉 token 与每个背景原型都在特征维 $D$ 点积，特征维被消去，先得到 `[B_s,N,M]` 相似度；再沿背景原型索引 $m$ 取最大值并消去 $M$，输出 `S_bg=[B_s,N]`。$S_{bg}[b,n]$ 表示该 token 与所有背景外观中最相似的一种。最大值适合“命中任一背景即可”，但单个错误原型也可能强烈抑制前景。

背景原型可以是 `sky/road/wall` 等文本，也可以来自视觉缓存。前景与背景原型必须使用相同归一化和温度；某数据集中的 `wall` 或 `road` 若是合法类别，不能同时被写成背景负类。

### 4.4 双阈值与 ignore

伪标签训练常采用：

$$
\tilde Y_{b,h,w}=
\begin{cases}
\arg\max_c P_{b,c,h,w}, & q_{b,h,w}\ge\tau_{fg},\\
\text{background}, & q_{b,h,w}\le\tau_{bg},\\
255, & \tau_{bg}<q_{b,h,w}<\tau_{fg},
\end{cases}
\qquad
q_{b,h,w}=\max_cP_{b,c,h,w}.
$$

**公式解释：** `P=[B,C,H,W]` 先沿类别维 $C$ 取最大概率和最大值索引，类别维被消去，得到 `q=[B,H,W]` 与获胜类别 `[B,H,W]`。分段规则逐像素输出一个整数：高置信位置用前景获胜类，足够低的前景置信位置设背景，中间区设 255；输出 `Y_tilde=[B,H,W]`。255 通常是 `ignore_index`，交叉熵不会从这些位置产生梯度；它不是 unknown 语义标签。

### 4.5 拒识 unknown

结合最大概率和 Shannon 熵可写：

$$
\hat y=
\begin{cases}
\arg\max_c p_c, & \max_cp_c\ge\tau_p\ \text{且}\ H(p)\le\tau_H,\\
\text{unknown}, & \text{otherwise},
\end{cases}
\qquad
H(p)=-\sum_{c=1}^{C}p_c\log(p_c+\varepsilon).
$$

**公式解释：** 对一个像素的 $C$ 类概率向量 $p$，最大值与 $\arg\max$ 都沿类别维操作并消去该维；熵也沿 $c$ 求和并消去类别维，得到标量不确定性。只有第一名概率足够高且整个分布足够尖锐时才输出已知类别，否则输出 unknown。批量应用后标签 shape 为 `[B,H,W]`；$\varepsilon$ 防止 $\log0$。两个阈值必须用独立的已知/未知验证数据校准。

### 4.6 能量分数

对已知类别 logit $z_c(x)$，常见能量为：

$$
E(x)=-T\log\sum_{c=1}^{C}\exp\bigl(z_c(x)/T\bigr).
$$

**公式解释：** 每个类别 logit 先除以标量温度 $T$ 并指数化，再沿类别索引 $c$ 求和并消去类别维，`log` 和乘 $-T$ 后输出单个样本/像素的标量能量 $E(x)$。它保留全部 logit 的整体幅值，而不仅是归一化后的第一名概率；使用上述符号约定时，强已知证据通常产生更低（更负）的能量，但最终 unknown 阈值方向必须以实际实现和验证集为准。

余弦分类器的 logit scale、温度和候选词表数量都会改变能量。增加大量相近类别后，求和项自然变大，因此不同词表配置不能共用未经校准的固定阈值。

## 5. 常见实现形式归纳

| 实现形式 | 背景/未知信号 | 是否训练 | 优点 | 局限 | 代表论文 |
|---|---|---:|---|---|---|
| 前景补集 | $1-\max S_{fg}$ | 否 | 极简，无额外模型 | 依赖 `[0,1]` 尺度，只看第一名 | [[S2C_paper_notes]] |
| 固定前景阈值 | 最大概率/相似度 | 否 | 易实现和控制覆盖率 | 跨词表、跨域不稳定 | [[Talk2DINO_paper_notes]] |
| 类别相关背景文本 | 水、铁轨等负语义 | 否 | 利用开放文本接口抑制共现 | 背景列表与数据集绑定 | [[CLIP-ES_paper_notes]] |
| 多背景视觉缓存 | 前景/背景视觉键值 | 可选 | 覆盖文本难描述的外观 | 未知前景可能被吸入背景 | [[DiCLIP_paper_notes]] |
| 学习式背景通道 | 独立 logit `[B,1,H,W]` | 是 | 可利用上下文和边界 | 需要可靠监督，易封闭化 | 经典分割头 |
| 熵/能量拒识 | 分布锐度或 logit 幅值 | 否或校准 | 允许模型说“不知道” | 高置信错误仍可能通过 | 开放集基线 |
| 结构引导背景清除 | 自注意力/区域前景性 | 否或轻量 | 不只依赖类别概率 | 结构模型也会漏小目标 | [[Talk2DINO_paper_notes]] |

## 6. 代表论文逐篇说明

| 论文 | 任务与起点 | 原方法存在的问题 | 具体做法 | 与本算子的关系 |
|---|---|---|---|---|
| [[CLIP-ES_paper_notes]] | 免训练对比语言—图像预训练（Contrastive Language-Image Pre-training，**CLIP**）WSSS；softmax-Grad-CAM | 目标类别与非目标前景、船—水、火车—铁轨等共现背景混淆 | 把数据集真实类别和为它们定制的类别相关背景集一起编码，让目标概率在同一 softmax 候选集合中接受负语义抑制；再用类感知注意力亲和力细化 CAM，并在最终训练中忽略低置信位置 | 背景不是补集，而是显式文本竞争者；背景词依赖数据集语义，换任务必须复核 |
| [[Talk2DINO_paper_notes]] | 无监督 OVS；DINOv2 patch 与映射后文本比较 | 与所有给定类别相似度都低的区域需要被识别为背景，单一阈值忽略 DINO 的结构前景性 | 利用 DINOv2 不同自注意力头产生的连贯区域，对类别相似度图进行背景清除/塑形，再把所有语义类别分数都低于阈值的位置标为背景；可选像素自适应掩码细化（Pixel-Adaptive Mask Refinement，**PAMR**）继续细化 | 属于“类别证据 + 类别无关结构”联合背景判断；它识别的是基准词表外背景，不等同于开放集 unknown |
| [[SSR_paper_notes]] | CLIP WSSS；跨模态原型生成 CAM | 错误视觉—文本对齐和无约束传播会把前景扩散进背景 | 先用跨模态原型对齐修正类别语义，再用超像素约束传播边，阻止响应跨越明显区域边界 | 不显式新增背景概率，而是减少语义错误和传播越界；属于背景污染抑制的结构路线 |
| [[ComCD_paper_notes]] | CLIP CAM 与扩散 CAM 融合 | 两个分支在物体内部、边界和背景上的可靠性不同，固定平均会保留各自误报 | 对同一像素两路类别分布计算熵，以相对低熵分支获得更高融合权重；伪掩码再监督双分支特征对齐解码器与可学习 logit 门控 | 背景/模糊区由相对可靠性处理，但低熵不保证正确；需检查两分支校准和背景占比 |
| [[POT_paper_notes]] | WSSS；高置信 CAM 种子与多原型最优传输 | 前景覆盖不足，但原型和传输若吸入背景会系统扩散错误 | 从可靠区域聚类图内原型，用分类器权重设置非均匀传输容量，让像素软分配到原型；熵权重调节细化结果与原 CAM 的一致性 | 背景处理体现在种子筛选、原型容量和一致性权重，不是一个独立背景头 |
| [[DiCLIP_paper_notes]] | CLIP 文本、扩散关系和生成视觉缓存 | 单个 `background` 文本无法覆盖多样外观，静态 patch—text 又会误激活共现区域 | 同时构造背景文本和由生成图像 patch 聚合得到的背景视觉键值；缓存读出中前景分数再乘 $1-S_{bg}$ 抑制；动态适配器学习目标域读出 | 文本背景与视觉背景原型组合；代码需检查背景通道排序、归一化是否有 epsilon、分数是否真在 `[0,1]` |
| [[S2C_paper_notes]] | WSSS；主网络 CAM 与分割一切模型（Segment Anything Model，**SAM**）点提示伪掩码 | 主 CAM 只有前景通道，SAM 区域需要一个可供交叉熵监督的背景类别 | 用 $1-\max$ 前景 CAM 构造最后一个背景通道；SAM 未覆盖位置在伪标签聚合时设为 VOC 背景索引 20，基于 CAM 的提示模块（CAM-based Prompting Module，**CPM**）交叉熵训练主网络 | 典型补集背景；背景 20 会产生梯度，不是 ignore 255，且输入更像归一化 CAM 分数而非标准未归一化 logit |

## 7. 各种实现怎样工作

### 7.1 补集背景：没有前景就算背景

**直觉**：把“最像前景的程度”取反。

**数据流**：前景 CAM `[B,C,H,W]` → 类别维 max `[B,1,H,W]` → `1-max` → 拼到前景通道 → 交叉熵/argmax。

**适用场景**：前景 CAM 已稳定归一化到 `[0,1]`，需要一个最低成本背景基线。

**局限**：逐类 min-max 会让每类都在某处达到 1，补集只在当前图内部有相对意义；若输入是负到正的 logit，`1-max` 更没有概率解释。补集背景还会把任何低前景未知物体直接当背景。

### 7.2 背景提示/原型：让背景参与竞争

**直觉**：给天空、道路、水面等常见非目标外观自己的语义方向。

**数据流**：背景词/视觉区域 → 编码或聚类 `[M,D]` → patch 与前景、背景同时相似度计算 → 类别竞争/前景抑制。

**适用场景**：错误集中在可枚举的共现背景，且有开放文本或可靠视觉缓存。

**局限**：背景集合越大不一定越好；多个背景原型会增加在任一方向偶然高匹配的机会。应统计每个背景原型实际抑制了哪些真实类别，并检查跨数据集语义冲突。

### 7.3 ignore：暂时不知道就不产生梯度

**直觉**：不确定像素先不教，而不是强行教成背景。

**数据流**：类别概率 → 双阈值/一致性 → 高置信前景、可靠背景、ignore → `cross_entropy(ignore_index=255)`。

**适用场景**：WSSS 伪标签边界和未激活区噪声高。

**局限**：阈值过严会留下极少监督。必须同时报告前景、背景和 ignore 比例；只报告保留像素精度会掩盖覆盖率下降。

### 7.4 unknown：判断是否超出已知世界

**直觉**：即使必须在已知类中选第一名，也允许模型说“这些类都不像”。

**数据流**：已知类 logit/嵌入距离 → 最大概率、熵、能量或原型距离 → 校准阈值 → 已知类别或 unknown。

**适用场景**：测试确实包含词表外物体，评价协议也提供 unknown 标注。

**局限**：softmax 最大概率随候选类数量变化；未知类可能与某个已知类高度相似而高置信误判。开放集评价需要独立 unknown 数据，不能只看闭集平均交并比（mean Intersection over Union，**mIoU**）。

## 8. no-object、void、ignore 与 background

| 名称 | 粒度 | 是否是语义类别 | 是否产生训练梯度 | 常见位置 |
|---|---|---:|---:|---|
| background | 像素/区域 | 是，任务明确要求 | 通常是 | 闭集分割最后/第 0 通道 |
| unknown | 像素/区域 | 是，表示词表外对象 | 依训练设定 | 开放集输出 |
| ignore | 像素 | 否，只是不采用监督 | 否 | 噪声伪标签、void 映射 |
| void/unlabeled | 数据集像素 | 通常不评价 | 通常否 | 标注缺失或边界 |
| no-object | query | 是匹配层面的“未匹配” | 通常是 | [[maskformer_notes]] 的 query 分类 |

MaskFormer/Mask2Former 的 no-object 用于告诉某个对象 query 没有匹配区域，不代表对应像素都是语义背景。反过来，WSSS 的 255 ignore 只阻断损失，不应被当成一个可预测通道。

## 9. 官方仓库静态分析：S2C 的补集背景

- 官方仓库：[sangrockEG/S2C](https://github.com/sangrockEG/S2C)
- 阅读 commit：[`102e14c690c8e3bce3d5ccd1ae7832145ce10b27`](https://github.com/sangrockEG/S2C/tree/102e14c690c8e3bce3d5ccd1ae7832145ce10b27)
- 背景与 CPM 损失：[`models/model_s2c.py#L345-L360`](https://github.com/sangrockEG/S2C/blob/102e14c690c8e3bce3d5ccd1ae7832145ce10b27/models/model_s2c.py#L345-L360)
- SAM 类别聚合与背景赋值：[`model_s2c.py#L266-L299`](https://github.com/sangrockEG/S2C/blob/102e14c690c8e3bce3d5ccd1ae7832145ce10b27/models/model_s2c.py#L266-L299)

### 9.1 完整调用链

```text
主网络前景 CAM [B,20,H,W]
→ 多尺度融合、ReLU、逐类空间归一化
→ CAM 峰值提示冻结 SAM
→ SAM 掩码置信 × 掩码内类别 CAM 均值
→ 类别维 max 得 pgt_sam [B,H,W]
→ 未覆盖位置设为类别索引 20

主网络有梯度前向得到 cam_main [B,20,H,W]
→ 类别维 max: [B,1,H,W]
→ 1-max 得 cam_bg [B,1,H,W]
→ cat 前景和背景: [B,21,H,W]
→ cross_entropy(target=pgt_sam, ignore_index=255)
→ backward，只更新主网络
```

### 9.2 核心代码逐行解释

```python
cam_bg = 1 - cam_main.max(dim=1, keepdims=True)[0]
cam_main = torch.cat((cam_main, cam_bg), dim=1)
self.loss_cpm = F.cross_entropy(
    cam_main, pgt_sam, ignore_index=255
)
```

1. `max(dim=1, keepdims=True)` 沿 20 个前景类别取最大值，类别维由 20 归约为 1，输出 `[B,1,H,W]`；`[0]` 取 `values`，不取获胜类别索引。
2. `1 - max` 逐元素构造背景，shape 不变。代码没有额外 sigmoid/softmax，因此它假设 `cam_main` 的数值范围适合做补集。
3. `torch.cat(..., dim=1)` 把背景追加到最后，得到 `[B,21,H,W]`；VOC 背景索引因此是 20，不是常见的第 0 通道。
4. `pgt_sam=[B,H,W]` 必须使用同一索引约定。交叉熵在类别维选择 target 对应通道并归约为空间/batch 标量损失。
5. `ignore_index=255` 只会忽略 target 真正等于 255 的位置；前面 SAM 聚合把未覆盖像素设为 20，因此这些位置作为背景参与梯度。

### 9.3 伪标签背景怎样产生

S2C 先初始化 `sam_conf=[B,C,H,W]` 为大负数，只在 SAM 候选掩码内写入“逐像素 SAM 置信 × 掩码内该类 CAM 均值”。随后 `max(dim=1)` 消去前景类别维，返回最大分数和类别索引；最大分数仍小于 0 的位置被设成 20。这里的 20 是训练背景，不是不确定 ignore。

因此背景有两个来源：学生预测端用前景补集构造背景通道，教师目标端把任何未被候选覆盖的位置设背景。若 CAM 漏掉真实前景，后者会形成明确负监督；改成 255 会更保守，但也降低监督覆盖率，必须作为独立设计选择和消融。

### 9.4 论文公式与代码实现的边界

- `F.cross_entropy` 通常接收未归一化 logit；这里输入由前景 CAM 与启发式补集组成，更像有限范围分数。函数在数值上可运行，但不能直接把 softmax 输出解释成经过概率校准的后验。
- 背景拼在最后一个通道，与许多数据加载器使用背景 0 的约定相反；迁移代码时必须同步 remap target。
- `keepdims=True` 在当前 PyTorch 中可作为 `keepdim` 的别名使用，但常见 API 写法是 `keepdim=True`；复用到不同版本时应确认。
- CPM 延迟到指定 epoch 才启用，减少早期错误 CAM 直接制造背景负监督；背景策略不能脱离伪标签课程单独理解。
- SAM 与伪标签生成位于无梯度路径，CPM 梯度只经过当前主网络 `cam_main`，不会更新 SAM 或回穿离散候选选择。

## 10. 调试与评价

至少分开统计：

- 已知前景 → background：漏检率；
- background → 已知前景：背景误激活；
- unknown → 已知前景：开放集误接收；
- 已知前景 → unknown：过度拒识；
- 伪标签中前景、背景、ignore 的比例；
- 更换候选词表前后背景阈值和预测面积变化。

闭集报告前景 mIoU、背景 IoU 和边界指标；OVS 报告 seen、unseen 及调和平均；开放集还应报告 unknown IoU、接收者操作特征曲线下面积（Area Under the Receiver Operating Characteristic，**AUROC**）、精确率—召回率曲线下面积（Area Under the Precision-Recall Curve，**AUPR**）和真阳性率 95% 时的假阳性率（False Positive Rate at 95% True Positive Rate，**FPR@95TPR**），并说明阈值来自哪个验证集。

## 11. 选型指南

| 当前症状 | 优先考虑 | 不值得或需先检查 |
|---|---|---|
| 前景 CAM 已归一化，只缺背景基线 | $1-\max$ 补集 | 先确认输入不是任意 logit/逐类不可比响应 |
| 固定共现背景导致误激活 | 类别相关背景文本或视觉原型 | 检查背景词是否在新数据集属于合法前景 |
| WSSS 未激活区真假不明 | 双阈值 + ignore | 不把全部未覆盖像素直接当背景 |
| OVS 只需给定词表内分类 | 前景词表 + 背景阈值/结构清除 | 不把它宣称为完整 unknown 拒识 |
| 测试有词表外真实物体 | 独立 unknown 评价 + 能量/距离校准 | 最大 softmax 概率不足以保证开放集能力 |
| 背景外观非常多样 | 多背景原型、类无关 objectness | 单一 `background` 文本通常过宽 |
| 更换词表后背景面积突变 | 重新校准温度和阈值 | 不跨词表复用 softmax/能量阈值 |
| 小目标频繁被当背景 | 降低结构清除强度、保留 ignore | 前景性注意力也会漏小目标 |

最低成本实验应包含：无背景通道/固定阈值、前景补集、显式背景提示或通道，以及 ignore 与强制背景两种伪标签策略。只有在明确存在词表外物体的评估集上，才讨论 unknown 拒识；否则应准确称为背景判定或低置信过滤。

## 12. 论文与源码索引

- [[CLIP-ES_paper_notes]]：类别相关背景文本如何进入 softmax 竞争并抑制共现背景。
- [[Talk2DINO_paper_notes]]：DINOv2 注意力结构怎样用于 OVS 背景清除。
- [[SSR_paper_notes]]：跨模态语义修正与超像素边界怎样减少背景污染。
- [[ComCD_paper_notes]]：双分支像素熵如何决定模糊/背景位置更信任谁。
- [[DiCLIP_paper_notes]]：前景/背景视觉缓存和文本背景锚点的组合。
- [[S2C_paper_notes]]：补集背景、背景索引 20 与 SAM 伪标签的实际训练语义。
- [S2C 补集背景和 CPM](https://github.com/sangrockEG/S2C/blob/102e14c690c8e3bce3d5ccd1ae7832145ce10b27/models/model_s2c.py#L345-L360)：背景通道位置、交叉熵和反向更新入口。
- [S2C SAM 类别聚合](https://github.com/sangrockEG/S2C/blob/102e14c690c8e3bce3d5ccd1ae7832145ce10b27/models/model_s2c.py#L266-L299)：未覆盖位置为何成为背景 20 而非 ignore 255。

## 13. 当前整理结论

背景与未知处理的核心选择不是“阈值设多少”，而是先定义非前景区域的语义：任务背景、词表外对象还是暂不监督。阅读新论文时应追问背景分数来自补集、文本、视觉原型还是学习头，unknown 是否有独立评价，候选词表变化是否重新校准；修改模型时先核对通道顺序、target 编码、`ignore_index` 和数值尺度，再决定是否增加复杂拒识分数。所有方法都可放回“前景证据 → 背景/未知证据 → 校准竞争或拒识 → 输出标签/训练掩码”这一条数据流中理解。
