---
type: operator-note
aliases: [Region Grouping, Region Proposal, 区域分组与提议]
tags: [research-operator, grouping, mask-proposal, region]
status: todo
---

# Region Grouping and Proposal（区域分组与提议）

> [!abstract] 核心直觉
> 先回答“哪些像素属于同一个物体或片段”，再回答“它是什么类别”。这样可以把空间边界与开放词汇语义拆开处理。

> [!tip] 基础机制入口
> 像素分类与mask分类的完整区别优先看 [[maskformer_notes]]，掩码查询与masked attention看 [[mask2former_notes]]。本页关注类别无关区域怎样接入文本对齐和弱监督种子。

## 1. 解决什么问题？

逐像素或逐patch分类容易出现椒盐噪声，也难保证同一物体内部预测一致。区域分组利用颜色、边缘、自监督特征或基础分割模型，把 $N$ 个位置压成 $R$ 个候选区域，通常 $R\ll N$。

它不一定知道区域类别，所以“分组正确”与“命名正确”是两个不同问题。

## 2. 输入输出张量

输入密集特征：

$$F\in\mathbb{R}^{B\times N\times D}.$$

输出可以是硬掩码：

$$M\in\{0,1\}^{B\times R\times H\times W},$$

也可以是软分配：

$$A\in[0,1]^{B\times N\times R},\qquad \sum_r A_{n,r}=1.$$

利用 [[Pooling_and_Region_Aggregation]] 后得到区域特征 $Z\in\mathbb{R}^{B\times R\times D}$。再把区域类别分数 $S\in\mathbb{R}^{B\times R\times C}$ 回投像素：

$$Y_{n,c}=\sum_r A_{n,r}S_{r,c}.$$

这里 $r$ 被求和消掉，结果恢复为每个位置的类别分数 `[B,N,C]`。

## 3. 常见形式

| 形式 | 分组依据 | 优点 | 局限 | 论文 |
|---|---|---|---|---|
| 超像素 | 颜色、纹理与局部边缘 | 快、无需训练、边界较细 | 语义弱且可能过分割 | [[SSR_paper_notes]] |
| K-means聚类 | 特征距离 | 实现简单 | 忽略空间连续性 | [[POT_paper_notes]] |
| 类别无关mask proposal | 学习到的掩码头 | 可得到完整物体候选 | 需要训练或预训练模型 | [[OpenSeg_paper_notes]] |
| SAM自动掩码/提示掩码 | Segment Anything Model | 边界与通用物体性强 | 可能拆分或合并语义类 | [[S2C_paper_notes]]、[[Trident_paper_notes]] |
| 注意力图聚类 | token亲和力 | 可直接复用Transformer内部关系 | 注意力不等于真实边界 | [[DiCLIP_paper_notes]] |

## 4. 硬分组与软分组

硬分组用 `argmax`：

$$g_n=\arg\max_r A_{n,r}.$$

每个位置只属于一个区域，便于索引和池化，但边界附近没有模糊余地。软分组保留比例，例如一个边界patch可对区域1和区域2分别贡献0.6和0.4，梯度也更连续，但显存为 $O(NR)$。

## 5. 论文之间的差异

| 论文 | 区域从哪里来 | 语义怎样加入 | 最终用途 |
|---|---|---|---|
| [[OpenSeg_paper_notes]] | 模型预测类别无关掩码 | 区域特征与标题/类别文本对齐 | 开放词汇区域分类并回投像素 |
| [[SSR_paper_notes]] | 传统超像素 | 初始CAM给类别种子 | 限制随机游走不要跨越物理边界 |
| [[S2C_paper_notes]] | SAM自动片段与CAM提示 | CAM确定类别，SAM补边界 | 片段对比和类别伪标签 |
| [[Trident_paper_notes]] | CLIP粗分割转为SAM提示 | CLIP保留开放类别语义 | 推理时细化高分辨率掩码 |
| [[DiCLIP_paper_notes]] | 扩散亲和力聚类 | 聚类本身无类别，后续注入CLIP | 形成多样的空间相关性偏置 |

## 6. 工程实例：S2C怎样把CAM变成SAM提示

固定阅读版本为 [`102e14c`](https://github.com/sangrockEG/S2C/tree/102e14c690c8e3bce3d5ccd1ae7832145ce10b27)。在 [`models/model_s2c.py`](https://github.com/sangrockEG/S2C/blob/102e14c690c8e3bce3d5ccd1ae7832145ce10b27/models/model_s2c.py#L266-L300) 中，代码按类别准备点提示，然后只运行SAM解码器：

```python
output_sam = self.net_sam(
    run_decoder_only=True,
    features_sam=features_sam[i].unsqueeze(0),
    point_coords=points,
    point_labels=points_label,
)
mask = output_sam[0]
conf = output_sam[2]
```

这说明工程上可以缓存昂贵的SAM图像特征，再为不同类别重复运行较轻的提示/掩码解码器。返回的多个候选掩码并没有全部使用；代码固定选择索引2，并结合SAM置信度与掩码内CAM均值进行聚合。论文中的“区域提议”落到代码后，往往还包含候选选择与冲突消解。

## 7. 选择与失败模式

- 小物体被漏掉：提高提议数量或使用高分辨率特征。
- 同类物体被切碎：允许多个区域共享类别，最后再合并。
- 不同语义类被合并：区域内再做细粒度分类，或保留软分组。
- CAM提示本身错误：SAM通常只会把错误位置分得更完整，不会自动纠正类别。
- 需要开放词汇：区域生成应尽量类别无关，把类别判断留给 [[Cross_Modal_Alignment]]。

## 8. 从经典像素分类到mask分类

[[fcn_notes]]、[[deeplabv3+_notes]] 等经典语义分割通常直接为每个像素预测固定类别；[[maskformer_notes]] 则把输出拆成一组mask和每个mask的类别分布，[[mask2former_notes]] 再用上一层预测mask限制cross-attention范围。区域提议页关注的是前半部分“mask怎样形成及是否覆盖物体”，而不是把query、mask和类别预测混成一个步骤。

三种常见组合可写成：

```text
像素分类：F → 每像素固定C类
mask分类：F → R个可学习mask/query → 每个mask固定C类
开放词汇mask分类：F → R个类别无关mask → 区域特征与运行时文本词表比较
```

弱监督方法还多一层约束：图像级标签只能告诉模型哪些类别可能出现，不能直接监督mask边界；因此CAM常被用作SAM提示、区域命名或候选过滤，而不是可靠真值。

## 9. 提议质量与分类质量要分开测

给定真实掩码 $G$ 和候选集合 $\{M_r\}$，可用最佳重叠衡量提议覆盖：

$$
\operatorname{oracleIoU}(G)=\max_r\operatorname{IoU}(G,M_r).
$$

如果oracle IoU已经很低，后续再强的文本分类器也无法恢复缺失区域；如果oracle IoU高而最终mIoU低，问题更可能在区域特征池化、类别对齐或冲突消解。应同时报告proposal recall@IoU、平均候选数与最终分类性能。

## 10. 重叠mask的组合与冲突

对重叠区域，可将mask概率 $m_r(n)$、区域objectness $o_r$ 与类别概率 $p_r(c)$ 组合：

$$
s_{n,c}=\sum_r m_r(n)\,o_r\,p_r(c).
$$

求和适合多个提议共同支持同一像素；逐像素取最大值更保守。若候选数量随图像变化，未经归一化的求和会偏向提议更多的位置。还应设置no-object或低objectness过滤，避免大量空mask进入竞争。

## 11. SAM与超像素的能力边界

- SAM提供的是提示条件下的通用mask，不天然带数据集语义；错误类别提示可得到边界精致但语义错误的掩码。
- 自动mask往往过分割，一个语义类可对应多个区域；语义分割允许它们共享类别后合并。
- 超像素重视低级边缘，可能把纹理变化当边界；它适合限制传播，却不保证实例完整。
- 类别无关提议更利于开放词汇迁移，但若训练数据只覆盖固定类别，提议器仍可能对未见物体产生选择偏差。

## 12. 先区分 partition、cover 与 instance set

区域输出并不总是同一种数学对象：

### 12.1 Partition（互斥划分）

每个位置只属于一个区域：

$$A_{n,r}\in\{0,1\},\qquad\sum_rA_{n,r}=1.$$

超像素标签图和普通语义分割argmax常属于这种形式。优点是回投简单，没有重叠冲突；局限是一个像素无法同时属于粗粒度“person”和细粒度“shirt”等层级区域。

### 12.2 Cover（可重叠覆盖）

$$M_r\subseteq\Omega,\qquad\sum_rM_r(n)\text{ 可大于 }1.$$

SAM自动mask、候选proposal和层级区域经常重叠。它能同时表达对象、部件与子区域，但需要去重、排序和像素冲突消解。

### 12.3 Instance set（实例集合）

多个mask可以拥有相同语义类别但对应不同实例。语义分割最终会合并同类实例；实例分割必须保留它们的独立ID；全景分割还要求thing实例互斥并为stuff区域合并。使用区域算子前必须明确最终任务，否则合并规则会相反。

## 13. 区域分组可拆成四个子问题

```text
候选产生：哪些像素可能组成一个区域？
  → 区域打分：这个候选像不像完整、稳定的对象/片段？
  → 去重与冲突：重叠候选保留、合并还是互斥？
  → 语义命名：区域属于哪个固定类或文本查询？
```

很多错误来自把最后一步提前。例如先用不可靠类别CAM定义唯一区域，后续SAM只能围绕错误种子细化；更稳的开放词汇路线是先生成类别无关候选，再用区域特征与文本命名。反过来，完全类别无关的提议可能遗漏没有“对象性”的stuff区域，因此也未必适合所有语义分割数据集。

## 14. 超像素：低级边界驱动的局部划分

超像素把相邻且颜色/纹理相似的像素聚成小块。可抽象为带空间约束的聚类：

$$
d(i,k)=d_{appearance}(x_i,\mu_k)+lambda_s d_{spatial}(u_i,v_k).
$$

$x_i$ 是颜色或局部特征，$u_i$ 是坐标；$\lambda_s$ 控制规则紧凑度。只有外观距离会产生空间上离散的簇，只有坐标距离又会忽略真实边缘。

### 优点

- 不需要类别标签和训练；
- 以较低成本给出局部边界；
- 可把 $N$ 个像素压成 $R\ll N$ 个图节点；
- 适合限制随机游走不跨明显边界。

### 局限

- 通常过分割，一个物体被切成许多片；
- 纹理丰富区域会产生大量碎片；
- 弱边界处可能把不同类别合并；
- 超像素编号没有跨图像语义，不能直接当类别。

[[SSR_paper_notes]] 用超像素空间先验过滤亲和传播中的跨区域噪声，其作用是“建墙”，不是独立识别类别。

## 15. 连通组件：从种子响应得到候选区域

对类别响应 $M_c$ 阈值化：

$$B_c(n)=\mathbb I[M_c(n)\ge\tau_c].$$

在4邻域或8邻域上求connected components，得到每个类别的若干种子区域。4邻域更保守，对角接触不会合并；8邻域更容易保持斜线/细结构连续，也更可能误合并相邻对象。

常见后处理：

- 移除面积小于 $a_{min}$ 的噪声组件；
- 对孔洞做fill；
- 形态学开运算去小噪声、闭运算连断裂区域；
- 每个组件取峰值点、box或粗mask作为SAM提示。

这些操作都依赖像素尺度。固定 `a_min=100` 在不同输入分辨率上语义不同，更稳妥的是使用图像面积或预计目标尺度的比例。

## 16. 特征聚类：外观分组必须加空间约束

仅在特征空间执行K-means：

$$g_n=\arg\min_k\|f_n-\mu_k\|^2$$

可能把图像中相隔很远但外观相似的区域分到同一簇。这对类别原型也许有用，却未必是连续mask。可拼接归一化坐标：

$$\tilde f_n=[\hat f_n;\lambda_x x_n/H;\lambda_y y_n/W].$$

也可先构建局部KNN图，再做图聚类/连通组件。坐标权重大时区域更紧凑但可能切碎大对象；权重小时更语义化却可能出现离散岛。

如果最终只需要类别内多个外观原型，离散簇可以接受；如果要产生可显示的区域mask，则应明确空间连通性约束。这也是[[Prototype_Construction]]与本页的边界。

## 17. 经典mask classification：query怎样成为区域槽位

[[maskformer_notes]] 把分割统一成 $R$ 个mask与 $R$ 个类别预测：

$$
q_r\xrightarrow{\text{Transformer decoder}}z_r,
$$

$$
m_r(n)=\sigma(e_n^Tg(z_r)),\qquad
p_r=\operatorname{softmax}(h(z_r)).
$$

$e_n$ 是像素嵌入，$z_r$ 是per-segment embedding。mask与类别从同一query表示分叉，但它们承担不同任务：$m_r$ 决定区域，$p_r$ 决定命名。

### Hungarian matching

训练时预测集合无固定顺序，需要把真值mask $y_j$ 与query $r$ 一一匹配：

$$
\hat\sigma=\arg\min_{\sigma}
\sum_j\mathcal C(y_j,\hat y_{\sigma(j)}).
$$

代价常综合类别、mask BCE和Dice。未匹配query学习no-object。这个一一匹配适合实例/段集合；语义分割中同类stuff如何合并由具体实现决定。

[[mask2former_notes]] 用上一层预测mask限制下一层cross-attention，使query只在相关区域读取像素特征。这是“预测区域 → 限制交互 → 更新区域表示”的迭代，不是一次静态mask池化。

## 18. 类别无关提议为什么有利于开放词汇

固定类mask头学习：

$$F\rightarrow Y\in\mathbb R^{C_{train}\times H\times W},$$

输出通道绑定训练类别。类别无关提议则拆成：

$$F\rightarrow\{M_r,z_r\}_{r=1}^{R},$$

$$s_{r,c}=\cos(z_r,t_c),\qquad c\in\mathcal C_{test}.$$

测试时替换 $\mathcal C_{test}$ 即可重算命名，不必重新生成mask。[[OpenSeg_paper_notes]] 的核心论点就是视觉—语义对齐应发生在视觉分组之后，区域集合是全局向量和逐像素表示之间的中间层。

但“类别无关”是训练目标层面的描述，不保证数据无偏。若提议器训练标注主要覆盖thing、大物体或常见域，它仍可能漏掉stuff、细线结构和域外对象。开放词汇评估应单独测proposal oracle recall，避免把提议偏差误归给文本对齐。

## 19. OpenSeg的数据流拆解

[[OpenSeg_paper_notes]] 的区域路径可写为：

```text
FPN多尺度特征
  → 融合到P2分辨率
  → 添加位置编码
  → N个可学习query进行region-to-image cross-attention
  → query与空间特征点积，产生N个类别无关mask
  → 在另一视觉特征上做mask-based pooling
  → 区域特征与标题词语/测试类别对齐
  → 区域分数回投像素
```

几个关键边界：

- query初始可学习，不代表预先固定类别；
- 类别无关mask监督负责教视觉分组；
- 标题只有词语，没有词—mask真值对应，因此区域—词语接地仍是弱监督；
- 推理时可加入更多区域proposal改善覆盖，但提议数增加也会加重冲突与计算。

## 20. SAM：提示条件下的mask生成器

SAM由图像编码器、提示编码器和轻量mask解码器组成。图像嵌入可缓存，之后重复输入不同提示：

| 提示 | 表达的信息 | 优点 | 风险 |
|---|---|---|---|
| 正点 | 目标内部位置 | 从CAM峰值容易提取 | 点落错类会生成完整错误区域 |
| 负点 | 不应包含的位置 | 可拆相邻对象 | 负点选择困难 |
| box | 目标大致范围 | 提供强空间范围 | box包含多物体时语义不明确 |
| mask | 粗轮廓/上一轮logit | 信息最完整 | 粗mask错误会被继承 |
| 自动点网格 | 覆盖整图候选 | 无需类别种子 | 提议多、重叠与去重成本高 |

SAM输出多个候选mask与质量预测时，不应默认固定索引在所有数据上都最佳。应结合预测IoU、稳定性、与输入CAM的重叠/区域内均值等选择，并记录选择规则。

> [!warning] SAM的“zero-shot”不是开放词汇命名
> SAM可在新域用提示产生mask，但训练没有显式类别监督。它提供的是提示条件下的区域边界泛化；要得到语义分割类别，仍需CLIP文本、图像标签、分类器CAM或其他命名机制。

## 21. 从CAM到点、框与mask提示

### 21.1 点提示

对每类CAM寻找局部峰值：

$$x_c^*=\arg\max_xM_c(x).$$

单峰只覆盖一个实例时，可做non-maximum suppression后取多个峰。峰值间最小距离要随分辨率缩放。图像标签只允许存在类别生成正点；不存在类的峰值应过滤。

### 21.2 框提示

阈值化后对连通组件取最小外接框。框太紧会截断低响应但真实的对象部分，太松会包含邻近对象。可按框宽高扩张比例 $\delta$，而不是固定像素。

### 21.3 mask提示

粗CAM归一化/阈值后作为低分辨率mask logit。相比点，它携带更多形状；相比硬mask，保留连续置信度通常更适合提示编码器。需要按SAM接口要求处理尺寸、padding和logit尺度。

[[S2C_paper_notes]] 的CPM从类别CAM提取局部峰值作为点提示，并用SAM与CAM联合可靠度聚合类别mask；[[Trident_paper_notes]] 比较点、框和mask提示进行开放词汇粗预测细化。两者共同依赖初始语义种子，但一个在训练中把SAM知识转给CAM分类器，一个主要在推理路径细化结果。

## 22. 自动mask生成与去重

自动mask通常从多尺度点网格产生大量候选。基本筛选链：

```text
生成多候选mask
  → 预测质量阈值
  → 稳定性阈值
  → 移除极小区域/孔洞
  → mask NMS或包含关系去重
  → 保留mask、score与来源提示
```

Mask IoU：

$$
\operatorname{IoU}(M_i,M_j)=\frac{|M_i\cap M_j|}{|M_i\cup M_j|}.
$$

普通NMS按分数保留一个、抑制高IoU候选。对层级mask，父对象与部件可能真实重叠，过强NMS会丢掉细粒度区域；可使用包含率：

$$
\operatorname{Contain}(M_i,M_j)=\frac{|M_i\cap M_j|}{\min(|M_i|,|M_j|)}
$$

识别近似包含关系，并根据任务决定保留父、子或二者。

## 23. 区域质量分数应分成三部分

一个候选是否值得保留至少涉及：

1. **objectness/segmentness**：它像不像完整、内部一致的区域？
2. **mask quality**：边界和预测稳定性如何？
3. **semantic confidence**：它像哪个类别，类别分数有多可靠？

可组合为：

$$q_{r,c}=q_r^{obj}\cdot q_r^{mask}\cdot q_{r,c}^{sem}.$$

乘法保守，但三个分数必须大致可比。开放词汇文本相似度可能为负或未经概率校准，不能直接与 `[0,1]` 的SAM预测IoU相乘；应先做温度/softmax、sigmoid或验证集校准。

同一mask在不同类别下共享结构分数，语义分数随候选词表变化。缓存区域时最好分别保存结构与语义，而不是只存最终乘积分数。

## 24. 重叠区域怎样恢复像素预测

给定区域mask概率 $m_r(n)$ 和类别分布 $p_r(c)$：

### 加权求和

$$s_{n,c}=\sum_rq_r m_r(n)p_r(c).$$

提议多的位置总分更高，可再除以 $\sum_rq_rm_r(n)$。

### 逐像素最大

$$s_{n,c}=\max_rq_rm_r(n)p_r(c).$$

保留最强候选，不被大量弱提议稀释；但梯度/解释只落在赢家。

### 区域先NMS再回投

先按区域类别/质量去重，再组合，适合实例式候选；但语义分割中两个同类重叠区域可能应合并而非互相抑制。

### query mask classification

MaskFormer式语义推理常聚合类别概率与mask概率：

$$s_{n,c}=\sum_rp_r(c)m_r(n),$$

并排除no-object通道。若开放词汇分类器没有no-object，应通过objectness或背景/unknown规则过滤空query。

## 25. 冲突图与区域合并

可把每个区域视为图节点，重叠或相邻关系为边：

$$G=(V,E),\qquad e_{ij}=\operatorname{IoU}(M_i,M_j)\text{ or }\operatorname{Adj}(M_i,M_j).$$

合并规则可同时考虑：

- 空间重叠/边界接触；
- 区域特征余弦相似度；
- 文本类别是否相同或层级兼容；
- 合并后区域内部一致性是否提高。

同类碎片合并能改善语义分割，但不同实例同类区域在实例任务中不能合并。仅按类别相同就合并会把画面中相隔很远的两个人变成一个实例。

## 26. 坐标与分辨率：最常见的工程错误

原图可能经历：

```text
原始尺寸
  → 等比例缩放
  → padding到模型输入
  → backbone下采样到patch网格
  → mask解码器上采样
```

每个区域应保存其坐标空间标识。SAM常在resize后padding的坐标上编码图像，CLIP/DINO窗口又可能使用不同crop。把CAM峰值送入SAM前，必须做与图像预处理完全相同的点坐标变换；输出mask回原图时先去padding再缩放。

对低分辨率mask插值：

- logit/软mask通常用bilinear；
- 离散区域ID图用nearest，避免生成不存在的混合ID；
- 二值mask可先插值logit再阈值，而不是对0/1结果线性插值后直接当真值。

## 27. 提议数量、复杂度与缓存

若区域数为 $R$、像素/patch数为 $N$、维度为 $D$：

- mask存储约 $O(RN)$；
- mask pooling约 $O(RND)$（稀疏mask可降成本）；
- 区域—文本分类约 $O(RCD)$；
- query cross-attention约 $O(RND)$；
- 区域两两NMS最坏约 $O(R^2N)$。

自动SAM候选很多时，可先用结构质量和面积过滤，再提取昂贵区域特征/文本相似度。缓存mask应同时保存图像预处理元数据、mask RLE/压缩形式、质量分数、提示来源与模型版本；只保存PNG会丢失浮点logit和候选关系。

## 28. 一个类别无关提议到开放分类的代码骨架

```python
def classify_regions(pixel_features, masks, text_features, eps=1e-6):
    # pixel_features: [B, N, D]
    # masks: [B, R, N] in [0, 1]
    # text_features: [C, D]，必须与视觉特征处于同一对齐空间
    mass = masks.sum(dim=-1, keepdim=True)
    valid = mass.squeeze(-1) > eps

    region_features = torch.einsum("brn,bnd->brd", masks, pixel_features)
    region_features = region_features / mass.clamp_min(eps)
    region_features = torch.nn.functional.normalize(region_features, dim=-1)
    text_features = torch.nn.functional.normalize(text_features, dim=-1)

    raw_region_scores = torch.einsum("brd,cd->brc", region_features, text_features)
    region_scores = raw_region_scores.masked_fill(~valid.unsqueeze(-1), -torch.inf)

    # 简单回投；真实系统还应加入objectness、背景和重叠校准
    safe_scores = raw_region_scores.masked_fill(~valid.unsqueeze(-1), 0.0)
    pixel_scores = torch.einsum("brn,brc->bnc", masks, safe_scores)
    normalizer = masks.sum(dim=1).clamp_min(eps).unsqueeze(-1)
    pixel_scores = pixel_scores / normalizer
    return region_scores, pixel_scores, valid
```

这段代码只展示接口，不能替代提议质量、背景处理和类别校准。若 `pixel_features` 来自DINO而 `text_features` 来自CLIP，直接运行虽形状合法，语义仍错误。

## 29. 论文链条逐一放回区域框架

### OpenSeg

可学习query产生类别无关mask，mask pooling形成区域嵌入，标题词语/测试文本给区域命名。它把分组和语义对齐显式拆开。

### S2C

SAM自动mask用于片段对比，传递类别无关分组；CAM峰值转点提示产生类别相关SAM mask，再作为训练CAM的自监督信号。SAM区域通过CAM获得语义。

### SSR

超像素不是最终mask proposal，而是亲和传播的局部边界约束。它强调区域先验也可只用于限制信息通路。

### CorrCLIP

SAM mask定义CLIP patch交互范围，并用于最终地图校正；DINO/自监督相似度重建区域内部的相关性值。区域既是attention scope，也是输出一致性约束。

### Trident

CLIP给粗开放词汇语义，DINO提供空间协变特征，SAM关联矩阵负责跨窗口全局聚合，粗预测再转SAM提示细化。区域提议处于高分辨率推理链而非训练伪标签链。

## 30. 区域方法的错误分解

| 错误类型 | 定义 | 典型表现 |
|---|---|---|
| miss | 没有候选覆盖真实对象 | 后续分类器无论多强都失败 |
| fragmentation | 一个真实对象被切成很多片 | 语义分割可合并，实例ID容易碎裂 |
| merge | 一个候选包含多个真实类别/实例 | 区域池化混合语义，无法唯一命名 |
| duplicate | 多个候选重复同一对象 | 回投分数偏置、计算增加 |
| boundary error | 类别正确但轮廓偏移 | boundary F-score低 |
| semantic error | mask覆盖正确但类别错误 | oracle proposal高、最终mIoU低 |
| hierarchy conflict | 对象与部件/父类同时覆盖 | 取决于标签本体与任务定义 |

这套分解比只看最终mIoU更有诊断力。特别是开放词汇系统应先测类别无关proposal的oracle上限，再测区域文本分类。

## 31. 区域提议的评价指标

### Proposal recall

对每个真值区域 $G_j$：

$$
\operatorname{Recall}@\tau=
\frac1J\sum_j\mathbb I\left[\max_r\operatorname{IoU}(G_j,M_r)\ge\tau\right].
$$

同时画候选数 $R$ 与Recall曲线；仅提高 $R$ 带来的覆盖提升必须与计算成本一起看。

### Average Recall

在多个IoU阈值上平均，兼顾是否找到对象与边界质量。对小/中/大对象分别报告，可定位分辨率偏差。

### Fragmentation与merge

- 每个真值对象匹配到的高重叠候选数；
- 每个候选与多少不同类别/实例真值显著重叠；
- 区域纯度：候选内占比最高的真值类别比例。

### 最终任务指标

语义mIoU、实例AP、全景PQ、边界F-score以及开放词汇seen/unseen指标不能互相替代。区域算子页应说明它为哪一种最终任务服务。

## 32. 实验应如何拆解

- 固定文本/类别分类器，只替换超像素、query mask、SAM和真值oracle区域。
- 固定区域集合，只替换平均池化、query表示与文本对齐，隔离命名误差。
- 报告提议数—召回—延迟曲线，不只报告一个阈值配置。
- 分别消融结构质量、语义置信和融合分数，防止三者乘积掩盖问题。
- 对SAM比较点/框/mask提示及提示错误扰动，量化对初始种子的敏感性。
- 开放词汇中按seen/unseen、thing/stuff、大小目标分析proposal recall。
- 检查测试时额外proposal是否使用了测试标注或标签先验，避免隐性数据泄漏。

## 33. 阅读论文时的区域分组记录模板

```text
输出是partition、可重叠cover还是实例集合：
区域来源（超像素/聚类/query/SAM/CAM组件）：
是否类别无关：
区域数固定还是可变：
硬mask、软mask、box或query表示：
训练区域监督从哪里来：
是否需要Hungarian matching/no-object：
SAM提示类型、坐标变换与候选选择：
objectness、mask quality与semantic score如何定义：
去重/NMS/包含关系规则：
重叠像素如何组合：
区域如何池化、如何与文本/类别对齐：
区域分数如何回投像素：
训练和推理是否都需要提议器：
proposal recall、fragmentation、merge如何评价：
缓存格式、预处理与模型版本：
```

## 34. 当前整理结论

区域分组的价值在于把问题拆成：

$$\boxed{\text{先找完整区域，再为区域命名}}.$$

它能显著改善空间一致性，但不会凭空提供正确语义。
