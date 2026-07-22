---
type: operator-note
aliases:
  - Prompt Construction
  - Prompt Engineering
  - 提示构建
tags:
  - research-operator
  - prompt
  - text-embedding
  - vision-language
  - weakly-supervised-segmentation
  - open-vocabulary-segmentation
status: in-progress
---

# Prompt Construction（提示构建）

## 1. 本页定位

本页整理弱监督语义分割（Weakly Supervised Semantic Segmentation，**WSSS**）和开放词汇分割（Open-Vocabulary Segmentation，**OVS**）中，怎样把数据集类别名、同义词、视觉属性、背景词和可学习上下文变成可用于密集预测的文本锚点。它不是单篇提示学习论文的总结，也不把“换一句更好听的话”当作机制；真正关心的是提示在哪个维度聚合、它改变了类别语义还是负类竞争、文本特征能否离线缓存，以及提示在整图分类上变好时是否真的改善 patch/region 定位。

完整方法回到 [[clip_paper_notes]]、[[CLIP-ES_paper_notes]]、[[ExCEL_paper_notes]]、[[VDA_paper_notes]]、[[DiCLIP_paper_notes]] 和 [[WeCLIP+_paper_notes]] 阅读。对比语言—图像预训练（Contrastive Language-Image Pre-training，**CLIP**）文本编码器和零样本分类的基础接口见 [[clip_notes]]；本页只提炼可复用的数据流、张量和工程检查项。

> [!abstract] 一句话直觉
> 类别名只是一个短标签，不一定等于模型在预训练时学到的最佳视觉描述。提示构建决定“让文本编码器怎样理解这个类别”，提示集成决定“多个理解最终怎样变成一个或多个分类锚点”。

## 2. 这个算子解决什么问题

### 2.1 大白话解释

直接输入 `dog` 可能过于抽象；输入 `a photo of a dog` 会把词放进 CLIP 熟悉的图文语境；加入 `four-legged`、`fur` 等属性可能更接近局部外观。但提示并非越长越好：动作、场景和共现物虽然能帮助整图判断，却可能让类别激活图（Class Activation Map，**CAM**）激活到草地、铁轨或水面上。

提示构建通常位于数据流最前端：

```text
数据集类别与语义定义
→ 模板、同义词、属性或负提示展开
→ tokenizer
→ 冻结或可训练文本编码器
→ 每个提示的文本向量
→ 提示维聚合/选择
→ 类别文本锚点
→ 图像、patch 或 region 与文本计算相似度
```

它主要解决三类错位：类别字符串有歧义、预训练文本分布与目标域表述不同、单个类别名没有提供足够的局部可见属性。它不直接解决视觉 token 过平滑、边界粗糙或空间分辨率不足；这些问题应分别回到 [[Attention_and_Affinity_Refinement]]、[[Spatial_Propagation]] 和 [[Multi_Level_Fusion]]。

### 2.2 专业表述

对类别 $c$ 构造 $K$ 个文本提示：

$$
\mathcal T_c=\{t_{c,1},\ldots,t_{c,K}\}.
$$

**公式解释：** $\mathcal T_c$ 是类别 $c$ 的字符串集合，$K$ 是模板、同义词或属性描述数量；$t_{c,k}$ 表示第 $k$ 条具体文本。该式只定义离散输入集合，没有张量乘法或维度消去；真正的数值张量要经过 tokenizer 和文本编码器后才产生。

文本编码器 $f_t$ 输出：

$$
E_c=\begin{bmatrix}f_t(t_{c,1})\\ \vdots\\ f_t(t_{c,K})\end{bmatrix}
\in\mathbb R^{K\times D}.
$$

**公式解释：** 每条提示被编码成一个 $D$ 维向量，沿提示维堆叠后得到 `E_c=[K,D]`。$E_c[k,d]$ 是第 $k$ 条提示在文本特征通道 $d$ 上的数值；这里只做堆叠，不对提示求和，因此提示维 $K$ 和特征维 $D$ 都保留。

若有 $C$ 个类别，聚合后形成文本锚点矩阵 $T\in\mathbb R^{C\times D}$，再与视觉 token $P\in\mathbb R^{B\times N\times D}$ 比较，详见 [[Cross_Modal_Alignment]]。

### 2.3 哪些相似现象不是提示构建负责的

- 文本与视觉维度分别是 $D_t,D_v$：需要学习投影或公共空间对齐，不是多写几个模板就能解决。
- 类别文本正确但 patch 响应粘在一起：属于视觉关系与空间传播问题。
- 背景词覆盖不了所有非前景区域：背景本身不是稳定类别，需结合 [[Background_and_Unknown_Handling]]。
- 更换词表后 softmax 置信度变化：属于候选集合和校准问题，不等于提示本身失效。

> [!note] 我的理解｜提示是在定义决策边界，不只是润色文本
> 文本向量最终充当分类器权重或检索键。增加一句属性描述，相当于移动类别锚点；加入背景词，相当于新增竞争方向；提示平均，相当于决定多个语义方向如何压成一个中心。因此应像调模型模块一样记录输入、聚合轴和评估协议。

## 3. 统一输入输出张量

### 3.1 固定模板集成

最常见做法是先逐提示 L2 归一化，再沿提示维平均，最后再次归一化：

$$
\bar e_c=\operatorname{Norm}\left(
\frac{1}{K}\sum_{k=1}^{K}\operatorname{Norm}(e_{c,k})
\right)\in\mathbb R^D.
$$

**公式解释：** 输入 `E_c=[K,D]`，内层 `Norm` 分别沿每条提示的特征维 $D$ 归一化，shape 仍为 `[K,D]`；随后沿提示索引 $k$ 求和并除以 $K$，提示维被消去，得到一个 `[D]` 类别均值向量；最外层再次沿 $D$ 维归一化，输出 $\bar e_c\in\mathbb R^D$。$\bar e_c[d]$ 表示类别 $c$ 集成后在文本通道 $d$ 上的方向分量。

数字例子：`E_c=[80,512]` 表示一个类别有 80 条模板、每条为 512 维；沿模板维平均后得到 `[512]`。20 个类别分别处理并堆叠，最终文本矩阵为 `[20,512]`。

### 3.2 加权集成

若提示质量不等，可引入权重：

$$
e_c=\sum_{k=1}^{K}\alpha_{c,k}e_{c,k},
\qquad
\alpha_{c,k}\ge 0,
\qquad
\sum_{k=1}^{K}\alpha_{c,k}=1.
$$

**公式解释：** $e_{c,k}\in\mathbb R^D$ 是第 $k$ 条提示向量，$\alpha_{c,k}$ 是对应标量权重。每个权重在 $D$ 个特征通道上广播，随后沿提示维 $K$ 加权求和并消去该维，输出 $e_c\in\mathbb R^D$；$e_c[d]$ 是所有提示在通道 $d$ 上的加权平均。权重和为 1 让输出尺度更易比较，但若后面还做 L2 归一化，主要保留的是方向差异。

权重可以由验证集固定、由 CAM 锐度选择，也可以随图像动态预测。动态权重若为 `[B,C,K]`，与提示张量 `[C,K,D]` 相乘后输出 `[B,C,D]`，这意味着同一类别在不同图像中拥有不同文本锚点；此时文本特征不能再完全离线压成 `[C,D]`。

### 3.3 从文本锚点到密集相似度

设归一化视觉 token 和类别文本为：

$$
\hat P\in\mathbb R^{B\times N\times D},
\qquad
\hat T\in\mathbb R^{C\times D},
\qquad
S=\hat P\hat T^T\in\mathbb R^{B\times N\times C}.
$$

**公式解释：** `P_hat=[B,N,D]` 与 `T_hat^T=[D,C]` 做矩阵乘法，共同特征维 $D$ 被乘加消去，patch 数 $N$ 和类别数 $C$ 被保留，输出 `S=[B,N,C]`。$S[b,n,c]$ 表示第 $b$ 张图第 $n$ 个视觉 token 与类别 $c$ 提示锚点的余弦相似度；提示构建最终就是通过改变 $T[c,:]$ 来改变这些密集分数。

若 `N=400,D=512,C=20`，则 `[2,400,512] @ [512,20] → [2,400,20]`。恢复二维空间时：

```text
[B,N,C]
→ permute(0,2,1)
→ [B,C,N]
→ reshape(B,C,H',W')，N=H'W'
→ bilinear interpolate
→ [B,C,H,W]
```

`permute` 只改变维度顺序，`reshape` 按原 patch 顺序拆回二维网格；双线性插值适合连续相似度图。离散类别索引只能用最近邻插值，基础见 [[downsampling_and_upsampling(下采样与上采样)]]。

## 4. 代表论文逐篇说明

| 论文 | 任务与起点 | 原方法存在的问题 | 具体做法 | 与本算子的关系 |
|---|---|---|---|---|
| [[CLIP-ES_paper_notes]] | 免训练 CLIP WSSS；从文本类别和 Grad-CAM 生成初始定位 | ImageNet 分类常用的模板集成会突出少数目标类，在多标签图像中反而压低其他共存类；目标类还会与水、铁轨等相关背景混淆 | 为每个候选模板统计图像中真实目标类别分数的锐度，选择分布更均衡的 `a clean origami {}.`；再为数据集类别引入同义词，并把类别相关背景词加入同一 softmax 竞争集合 | 展示“分类最佳提示不一定定位最佳”；提示选择目标是多标签 CAM，而不是整图 top-1 |
| [[ExCEL_paper_notes]] | CLIP WSSS；类别名难覆盖细粒度视觉外观 | 全局文本概念不足以描述颜色、部件、纹理等局部属性，直接 patch—类别名匹配容易漏掉非判别区域 | 借助大语言模型（Large Language Model，**LLM**）生成类别的细粒度视觉属性候选，再结合文本语义丰富与视觉校准，挖掘适合密集 patch 对齐的属性先验 | 属于属性提示/语义扩展；需区分 LLM 给出的文字知识与图像中真正可见的属性 |
| [[VDA_paper_notes]] | CLIP WSSS；静态类别文本原型生成 CAM | 单个类别锚点难表达实例颜色、姿态和部件变化，简单属性列表又可能混入共享或不可见描述 | 视觉属性建模与解耦（Visual Attribute Modeling and Disentanglement，**VAMD**）用层次概率模型形成类别及属性原型，再按图像证据组装动态视觉描述；解码器语义增强继续用类别原型约束特征 | 不只是把属性句子平均，而是把类别语义拆成可组合的潜在视觉成分 |
| [[DiCLIP_paper_notes]] | 扩散知识增强 CLIP WSSS | 前景文本无法覆盖背景外观；静态文本 CAM 与视觉缓存各有噪声 | 把前景类别名和多组背景类别送入固定模板编码，形成文本锚点；生成的单类图像进一步构造前景/背景视觉键值缓存，让真实 patch 同时查询文本和视觉先验 | 提示定义文本分类坐标，缓存补充文本难表达的视觉外观；两者可组合但不能混为同一个模块 |
| [[WeCLIP_paper_notes]]、[[WeCLIP+_paper_notes]] | 冻结 CLIP/DINO 的单阶段 WSSS | 冻结骨干提供稳定语义，但初始文本 CAM 是静态目标且会持续保留同一错误 | 用前景和背景类别构造文本提示，经冻结文本编码器生成分类锚点；解码器与冻结注意力关系再动态细化这些 CAM | 提示只负责初始类别语义，后续提升主要来自关系细化，不能把全部收益归因于文本输入 |
| [[Talk2DINO_paper_notes]] | 无监督 OVS；DINOv2 patch 定位强但没有文本接口 | CLIP 文本向量与 DINOv2 视觉空间不同，原始类别字符串不能直接和 DINO patch 点积 | 保留 CLIP 文本编码器生成的文本语义，再学习非线性映射把文本向量送入 DINOv2 空间；训练时用 caption 图文对比，推理时与开放类别文本匹配 | 提示构建与 [[Cross_Modal_Alignment]] 串联：先决定文本内容，再解决两个预训练空间的坐标差异 |

> [!note] 我的理解｜论文中的“文本增强”至少要拆成三问
> 文本内容是否改变？多个向量怎样聚合？聚合后的向量在哪个视觉空间使用？CLIP-ES 主要改前两项，Talk2DINO 主要解决第三项，VDA/ExCEL 则试图让文本内容更贴近局部可见属性。

## 5. 常见实现形式归纳

| 实现形式 | 输入单位 | 是否训练 | 优点 | 局限 | 代表论文 |
|---|---|---:|---|---|---|
| 单固定模板 | 类别名 + 1 个句式 | 否 | 最简单，可完全缓存 | 对域和措辞敏感 | [[CLIP-ES_paper_notes]]、[[DiCLIP_paper_notes]] |
| 多模板嵌入平均 | 每类 $K$ 个模板 | 否 | 降低单模板偶然偏差 | 多标签定位中可能放大强类 | [[clip_paper_notes]] |
| 验证集/锐度选择 | 模板集合 + 图像级标签 | 否 | 直接针对定位指标选模板 | 与数据集绑定，可能过拟合 | [[CLIP-ES_paper_notes]] |
| 同义词融合 | 类别别名集合 | 否 | 扩大词义覆盖 | 别名可能语境或层级不同 | [[CLIP-ES_paper_notes]] |
| 属性语义扩展 | 类别 + 属性短语 | 可选 | 更接近 patch/region 外观 | 共享、不可见或虚假属性会带偏 | [[ExCEL_paper_notes]]、[[VDA_paper_notes]] |
| 前景—背景竞争提示 | 前景词与背景词集合 | 否 | 显式压制共现背景 | 背景无法穷举，跨数据集冲突 | [[CLIP-ES_paper_notes]]、[[DiCLIP_paper_notes]] |
| 可学习上下文 | 连续 prompt token + 类别名 | 是 | 可适配目标域 | 易过拟合 seen 类，需额外训练 | [[WeCLIP+_paper_notes]] 中讨论的相关路线 |

这些形式可以组合，例如“每类同义词 × 多模板”后做嵌入平均，或“类别名 + 属性原型”动态组装。但组合会使提示数从 $K_1+K_2$ 变成 $K_1K_2$，需要明确先在哪个维度聚合，避免无意中让别名更多的类别获得更多投票。

## 6. 各种实现怎样工作

### 6.1 单模板与多模板集成

**直觉**：同一个类别换几种 CLIP 熟悉的句式描述，再把结果平均，降低某一句话的偶然偏差。

**数据流**：类别名 → 模板展开 `[C,K]` 字符串 → tokenize `[CK,L]` → 文本编码 `[CK,D]` → reshape `[C,K,D]` → 沿 $K$ 平均 → `[C,D]`。

**适用场景**：不训练文本侧、类别集合固定、希望离线缓存文本锚点。

**局限**：整图分类的模板集成目标是突出唯一正确类；WSSS 图像可能同时有多个目标，过尖的类别分布会让弱类的 Grad-CAM 梯度变小。应分别验证图像级分类和初始 CAM，而不是只复用 ImageNet 默认模板集。

### 6.2 同义词与类别本体

**直觉**：模型可能更熟悉 `sofa` 而不是 `couch`，把合法别名都提供给它。

**数据流**：数据集标签 → 人工/词典别名 → 逐别名编码 → 类内平均、最大值或 log-sum-exp → 类别分数。

**适用场景**：类别名存在地域差异、缩写或常用别名。

**容易误解**：近义词不一定是数据集定义下的同一类，例如 `person`、`rider` 和 `pedestrian` 有层级与场景差异；复数、冠词和多义词也会改变 tokenization。别名表必须和数据集标签本体一起版本化。

### 6.3 视觉属性提示

**直觉**：类别名告诉模型“它是什么”，属性告诉模型“局部看起来像什么”。

**数据流**：类别 → 属性候选 → 去重与可见性筛选 → 编码 `[C,A,D]` → 与类别锚点融合或作为多原型 → patch/region 检索。

**适用场景**：同一类别外观多样，单一文本中心只激活最典型部位。

**局限**：`used for transportation` 是功能知识，不是稳定可见属性；`on the road` 是场景共现；`four-legged` 被多个类共享。大语言模型生成的属性必须检查可见性、区分度和事实性，不能因文字更丰富就默认更适合密集定位。

### 6.4 背景与负提示

**直觉**：只告诉模型“这是船”不够，还要让它知道“水不是船”。

**数据流**：前景类别 + 类别相关背景词 → 同一文本编码器 → 拼成候选矩阵 → softmax 竞争或前景分数减背景分数。

**适用场景**：目标与固定共现背景混淆明显，如船—水、火车—铁轨。

**局限**：某数据集的背景词可能是另一数据集合法前景；一个统一 `background` 向量也无法表达所有 stuff 与未知物体。更完整处理见 [[Background_and_Unknown_Handling]]。

## 7. CLIP-ES 的提示锐度

CLIP-ES 用图像级标签衡量一个提示是否让同图目标类别分数过度失衡。可概括为：

$$
\operatorname{Sharpness}(t)=
\frac{\sum_{i=1}^{n}\operatorname{Var}\bigl(s_{i,1}^{(t)},\ldots,s_{i,k_i}^{(t)}\bigr)}
{\sum_{i=1}^{n}\operatorname{Mean}\bigl(s_{i,1}^{(t)},\ldots,s_{i,k_i}^{(t)}\bigr)+\varepsilon}.
$$

**公式解释：** $n$ 是评估图像数，$k_i$ 是第 $i$ 张图真实存在的目标类别数，$s_{i,j}^{(t)}$ 是使用提示 $t$ 时第 $j$ 个真实类别的图像级分数。每张图先在其目标类别集合上计算方差和均值，类别索引 $j$ 被归约；再沿图像索引 $i$ 求和并消去图像维，分子衡量目标类之间有多不均衡，分母校正总体分数尺度，最终输出单个提示的标量锐度。$\varepsilon$ 防止平均分数接近零时除零。

低锐度并不保证 CAM 一定更好，但它针对 WSSS 的多标签症状：同图多个真实类别应都保有足够分数，不能只让一个强类压倒其余目标类。提示选择只能使用训练/验证图像级标签，不能根据测试集 CAM 反复调句式。

## 8. 两种集成方式为什么不等价

先平均嵌入再计算余弦：

$$
s_c^{\mathrm{embed}}=
\cos\left(v,\operatorname{Norm}\left(\sum_{k=1}^{K}\hat e_{c,k}\right)\right).
$$

**公式解释：** $v,\hat e_{c,k}\in\mathbb R^D$。先沿提示维 $K$ 把归一化文本向量相加，提示维被消去，得到一个 $D$ 维类别方向；归一化后与视觉向量 $v$ 在特征维 $D$ 点积，特征维也被消去，输出类别 $c$ 的标量分数。该分数受提示向量彼此是否一致影响：方向互相抵消时，重新归一化会改变类别间尺度。

逐提示计算余弦再平均：

$$
s_c^{\mathrm{score}}=
\frac{1}{K}\sum_{k=1}^{K}\cos(v,\hat e_{c,k}).
$$

**公式解释：** 每个提示先与 $v$ 沿特征维 $D$ 点积，得到 $K$ 个标量相似度；再沿提示索引 $k$ 求平均并消去提示维，输出标量 $s_c^{\mathrm{score}}$。它保留各提示独立打分后再投票，不会因先形成一个归一化中心而额外改变类别尺度。

若用 `max`，语义是“任一属性命中即可”；若用 log-sum-exp，则是平滑最大值。两者适合多外观原型，但也更容易被一个错误高响应属性支配，需报告每个提示的命中频率。

## 9. 官方仓库静态分析：DiCLIP 的提示集成

- 官方仓库：[zwyang6/DiCLIP](https://github.com/zwyang6/DiCLIP)
- 阅读 commit：[`1c3f6ff7d4fde2afff32d527d78b28d119583602`](https://github.com/zwyang6/DiCLIP/tree/1c3f6ff7d4fde2afff32d527d78b28d119583602)
- 提示函数：[`clip/clip.py::encode_text_with_prompt_ensemble`](https://github.com/zwyang6/DiCLIP/blob/1c3f6ff7d4fde2afff32d527d78b28d119583602/clip/clip.py#L252-L269)
- 主模型调用：[`model/model_diclip.py#L63-L67`](https://github.com/zwyang6/DiCLIP/blob/1c3f6ff7d4fde2afff32d527d78b28d119583602/model/model_diclip.py#L63-L67)

### 9.1 调用链

```text
new_class_names + BACKGROUND_CATEGORY
→ 每个类别代入 prompt_templates
→ tokenize: [K,L]
→ CLIP encode_text: [K,D]
→ 每条提示沿 D 归一化
→ 沿 K 平均: [D]
→ 再沿 D 归一化
→ C 个类别堆叠并转置: [C,D]
→ 保存为 integral_text_features
→ 与全局/patch视觉特征计算相似度
```

### 9.2 核心代码与 shape

固定 commit 的实现是：

```python
for t in texts:
    prompted_t = [template.format(t) for template in prompt_templates]
    prompted_t = tokenize(prompted_t).to(device)
    class_embeddings = model.encode_text(prompted_t)
    class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
    class_embedding = class_embeddings.mean(dim=0)
    class_embedding /= class_embedding.norm()
    text_features.append(class_embedding)
text_features = torch.stack(text_features, dim=1).to(device).t()
```

分步骤看：

1. 对单个类别，`prompted_t` 有 $K$ 个字符串，tokenize 后通常是 `[K,L]`，$L$ 是 CLIP 固定文本长度；
2. `encode_text` 输出 `[K,D]`，`norm(dim=-1, keepdim=True)` 得到 `[K,1]` 并在 $D$ 维广播相除；
3. `mean(dim=0)` 消去模板维 $K$，得到 `[D]`；第二次 `norm()` 对整个 $D$ 维向量求标量范数；
4. $C$ 个 `[D]` 向量以 `dim=1` 堆成 `[D,C]`，最后 `.t()` 得到 `[C,D]`；
5. 这段过程只经过冻结 CLIP 文本编码器，没有新的可学习提示参数，文本锚点可在模型初始化时一次计算。

### 9.3 函数能力与实际配置不同

函数在 `prompt_templates is None` 时内置一组较大的 ImageNet 模板，确实支持 prompt ensemble；但 DiCLIP 主模型实际调用为：

```python
prompt_templates=['a clean origami {}.']
```

因此该实验路径中 $K=1$，`mean(dim=0)` 并没有融合多个模板，只保留统一接口。`text_prompts` 同时包含前景类别和 `BACKGROUND_CATEGORY`，输出类别维也包含多条背景语义；不能仅根据 `num_classes=21` 推断 `integral_text_features` 恰好只有 21 行，真实数量取决于背景列表。

### 9.4 值得注意的工程细节

- 归一化除法没有显式 epsilon。CLIP 文本向量通常非零，但若复用到其他编码器，应考虑 `F.normalize(..., eps=...)`。
- 提示字符串、类别顺序、背景列表、tokenizer 和 CLIP checkpoint 共同决定 `[C,D]` 缓存；只保存张量而不保存这些元数据，后续无法确认每一行对应哪个语义。
- 文本特征在 `__init__` 中直接计算并保存为普通属性；检查 checkpoint/device 迁移时，应确认它是否作为 buffer 注册，以及模型迁移设备后该张量是否跟随。
- 代码先逐提示归一化、平均后再归一化，和“逐提示相似度平均”不是同一实现。
- 主模型随后还用视觉缓存和 feature surgery 生成 CAM，因此最终结果不能被解释为单模板余弦相似度的直接输出。

## 10. 选型指南

| 当前症状 | 优先考虑 | 先检查或不值得做的情况 |
|---|---|---|
| 类别名有明显多义词 | 数据集释义 + 人工核验同义词 | 不把上位词、下位词和共现词都算同义词 |
| 整图分类好但 CAM 只亮场景 | 删除动作/场景描述，保留局部可见属性 | 不继续堆更长 caption |
| 同图强类压制弱类 | 单模板/低锐度模板，检查 softmax 候选集合 | 不默认 ImageNet 多模板集成最优 |
| 类内外观差异大 | 多属性原型或动态属性组装 | 若属性高度共享，先做单类别名基线 |
| 船—水、火车—铁轨混淆 | 类别相关背景提示参与竞争 | 背景词在目标数据集中若是合法前景，不能直接作为负类 |
| OVS 未见类性能下降 | 类别无关共享模板、冻结文本编码器 | 不为每个 seen 类学习完全独立上下文 |
| 推理类别频繁变化 | 保留在线文本编码接口或按词表缓存 | 不把最终头固定成训练类别数 |
| 视觉与文本空间不同 | 先做文本到视觉投影 | 多模板无法替代空间对齐 |

最低成本基线应包含：裸类别名、`a photo of a {class}`、一个经验证的单模板、标准多模板平均，以及同义词/属性分别加入的消融。若提示只改善图像级分数却不改善初始 CAM、区域检索或最终平均交并比（mean Intersection over Union，**mIoU**），就不应把它保留为密集预测模块。

## 11. 论文与源码索引

- [[CLIP-ES_paper_notes]]：多标签 WSSS 中提示锐度、单模板选择、同义词与类别相关背景集。
- [[ExCEL_paper_notes]]：细粒度视觉属性如何服务 patch—文本语义丰富。
- [[VDA_paper_notes]]：属性原型的概率建模与动态视觉描述组装。
- [[DiCLIP_paper_notes]]：前景/背景文本锚点和视觉键值缓存怎样协同。
- [[Talk2DINO_paper_notes]]：文本内容确定后，怎样映射到 DINOv2 patch 空间。
- [DiCLIP 提示展开与集成](https://github.com/zwyang6/DiCLIP/blob/1c3f6ff7d4fde2afff32d527d78b28d119583602/clip/clip.py#L252-L269)：模板展开、两次归一化、提示平均和 `[C,D]` 输出。
- [DiCLIP 实际单模板配置](https://github.com/zwyang6/DiCLIP/blob/1c3f6ff7d4fde2afff32d527d78b28d119583602/model/model_diclip.py#L63-L67)：确认实验路径使用一个模板，并同时编码背景类别。

## 12. 当前整理结论

提示构建的核心选择是：文本要补充类别名、局部属性还是负语义；多个提示是在嵌入维聚成单中心，还是保留多原型分别打分；最终文本锚点在哪个视觉空间使用。阅读新论文时应追问模板和别名的确切字符串、聚合轴、归一化顺序、提示是否随图像变化，以及选择提示时用了什么数据。修改模型时先做词义和类别本体检查，再比较单模板与集成，最后才考虑 LLM 属性或可学习 prompt；否则很容易用复杂文本掩盖视觉侧真正的空间问题。
