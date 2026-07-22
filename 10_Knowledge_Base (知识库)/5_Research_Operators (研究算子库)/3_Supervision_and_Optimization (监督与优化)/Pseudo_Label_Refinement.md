---
type: operator-note
aliases:
  - Pseudo-Label Refinement
  - 伪标签细化
tags:
  - research-operator
  - pseudo-label
  - refinement
  - weakly-supervised
  - open-vocabulary
  - semantic-segmentation
status: in-progress
---

# Pseudo-Label Refinement（伪标签细化）

> [!abstract] 本页定位
> 伪标签细化不是单个算法，而是把类别激活图（Class Activation Map, CAM）、教师概率或区域提议转换成“足够可靠、可以监督分割器”的目标，并显式放弃不确定位置。本页整理弱监督语义分割（Weakly-Supervised Semantic Segmentation, WSSS）和开放词汇语义分割（Open-Vocabulary Segmentation, OVS）中常见的阈值、背景、区域、传播、互教和课程更新接口；原论文笔记保存完整方法，本页关注监督目标如何真正进入损失。

## 1. 输入输出

输入类别概率：

$$P\in[0,1]^{B\times C\times H\times W},\qquad\sum_cP_{c,h,w}=1.$$

**公式解释：** $P$ 是像素级类别概率，shape 为 `[B,C,H,W]`；$B,C,H,W$ 分别是批量、类别和空间维。对固定的 $(b,h,w)$，求和只沿类别索引 $c$ 进行并消去类别维，结果为标量 1；$P[b,c,h,w]$ 表示该像素分给类别 $c$ 的概率。严格写 batch 索引时，归一化条件是 $\sum_cP_{b,c,h,w}=1$。

输出硬标签：

$$\tilde Y\in\{0,1,\ldots,C-1,255\}^{B\times H\times W}.$$

**公式解释：** $\tilde Y$ 是离散硬伪标签，shape 为 `[B,H,W]`，已不再保留类别通道；每个 $\tilde Y[b,h,w]$ 是 $0$ 到 $C-1$ 的一个类别编号，或特殊值 255。这里没有矩阵运算，类别维是在此前的 $\arg\max$ 或阈值决策中被消去的；255 通常表示该像素不参与交叉熵。

`255` 常用作ignore index，表示该像素不参与交叉熵。细化的目标不是强迫每个像素都有答案，而是平衡“覆盖更多像素”和“减少错误监督”。

如果模型原始输出是logit：

$$
Z\in\mathbb{R}^{B\times C\times H'\times W'},
$$

**公式解释：** $Z$ 是模型在低分辨率网格上的未归一化 logit，shape 为 `[B,C,H',W']`；$Z[b,c,h',w']$ 是位置 $(h',w')$ 对类别 $c$ 的原始分数。该式只声明输入，没有归约或消去维度；$H',W'$ 往往小于最终监督尺寸 $H,W$。

通常先上采样到监督网格，再沿类别维归一化：

$$
P=\operatorname{softmax}(\operatorname{Interpolate}(Z);\text{dim}=1)
\in[0,1]^{B\times C\times H\times W}.
$$

**公式解释：** `Interpolate` 先只改变空间维，把 `Z=[B,C,H',W']` 上采样为 `[B,C,H,W]`，批量维和类别维不变；随后 softmax 在 `dim=1` 的类别维 $C$ 上计算，每个输出类别都除以同一像素全部 $C$ 类指数分数之和。求和索引在分母中被消去，但输出仍保留完整类别维，得到 `P=[B,C,H,W]`；$P[b,c,h,w]$ 是上采样后该像素属于类别 $c$ 的概率。

例如 `[2,21,20,20] → bilinear → [2,21,320,320] → softmax(dim=1)`。双线性插值处理连续logit；已有离散伪标签只能用最近邻插值，否则会产生不存在的类别ID。

## 2. 一条常见流程

```text
连续CAM/概率
  → 类别过滤（只保留图像标签中出现的类）
  → 空间传播或区域细化
  → 加入背景分数
  → 置信度阈值与冲突消解
  → argmax得到类别
  → 低置信位置设为ignore
```

## 3. 常见实现形式

| 形式 | 主要作用 | 优点 | 风险 | 论文 |
|---|---|---|---|---|
| 双阈值 | 高分前景、低分背景、中间忽略 | 简单且保守 | 阈值依赖数据集 | [[UGRL_paper_notes]] |
| 亲和力传播 | 补全同一物体内部 | 提高覆盖 | 错误扩散 | [[CLIP-ES_paper_notes]]、[[SSR_paper_notes]] |
| 解码器—CAM互教 | 两个预测互相修正 | 能随训练更新 | 可能确认偏差 | [[WeCLIP_paper_notes]] |
| 多模型置信融合 | 选择更可靠分支 | 利用互补模型 | 概率需校准 | [[ComCD_paper_notes]] |
| SAM/区域细化 | 用通用边界把种子变完整掩码 | 边界好 | 错误提示会生成完整错误区域 | [[S2C_paper_notes]]、[[Trident_paper_notes]] |
| CRF | 图像颜色与边缘后处理 | 不训练即可细边界 | 速度和超参数成本 | [[CLIP-ES_paper_notes]] |

## 4. 阈值化怎样工作？

设最高类别概率与类别为：

$$q=\max_cP_c,\qquad c^*=\arg\max_cP_c.$$

**公式解释：** 对每个像素的长度为 $C$ 的概率向量，$\max_c$ 返回最大概率 $q$，$\arg\max_c$ 返回达到最大值的类别索引 $c^*$；两者都沿类别维 $C$ 操作并将其消去。若输入是 `[B,C,H,W]`，则 $q$ 和 $c^*$ 都是 `[B,H,W]`；$q[b,h,w]$ 表示第一名有多可信，$c^*[b,h,w]$ 表示第一名是谁。

可定义：

$$
\tilde Y=\begin{cases}
c^*,&q\ge\tau_{fg},\\
\text{background},&q\le\tau_{bg},\\
255,&\text{otherwise}.
\end{cases}
$$

**公式解释：** 该式逐像素读取标量置信度 $q$ 和获胜类别 $c^*$，没有新的矩阵乘法。若 $q\ge\tau_{fg}$，输出前景类别 $c^*$；若 $q\le\tau_{bg}$，输出预先约定的背景索引；中间区输出 255。输入中的类别维已由上一式消去，所以结果 `Y_tilde=[B,H,W]`，每个元素表示前景类、背景或“不监督”三种决策之一。

若 $\tau_{fg}=0.7,\tau_{bg}=0.2$，概率0.85的像素成为前景，0.1成为背景，0.45不确定而被忽略。中间区不产生梯度，避免把模糊边界当成确定真值。

## 5. 代表论文逐篇对比

| 论文 | 任务与初始监督 | 原伪标签问题 | 具体细化流程 | 谁接受监督 |
|---|---|---|---|---|
| [[CLIP-ES_paper_notes]] | 图像级标签WSSS；CLIP Softmax-GradCAM | 初始CAM只覆盖判别区域，原始注意力类别无关 | 用当前类别CAM连通框限制CLIP注意力亲和力，交替归一化、对称化并传播；再用DenseCRF贴合颜色边界 | 最终分割模型，并通过置信度引导损失忽略噪声区域 |
| [[WeCLIP_paper_notes]] | 单阶段WSSS；冻结CLIP CAM | 冻结伪标签在训练中不能改善，固定错误持续监督 | 解码器特征形成可学习亲和力，用它筛选冻结CLIP多层注意力，构造RFM细化图并作用于初始CAM | 可训练轻量解码器；解码器反过来继续改善RFM关系 |
| [[UGRL_paper_notes]] | 图像级标签WSSS；初始CAM与不确定区域 | CAM阴影、边界与未激活区域可靠性不同 | 通过可靠性估计区分可靠/不可靠区域，并让可靠区域学习链逐步扩展监督，而非一次性给所有像素硬标签 | 分类与分割主干；不确定像素受到更保守处理 |
| [[SSR_paper_notes]] | CLIP式WSSS | 图文模态间隙使类别种子错误，随机游走又会污染背景 | 先用图文投影和跨模态原型对比校正语义，再用超像素约束亲和力传播范围 | 单阶段分割模型；语义校正和空间补全分开执行 |
| [[S2C_paper_notes]] | ResNet38 CAM + 图像级标签 | CAM峰值提示可能错误，SAM输出多个重叠且无类别掩码 | 从多尺度CAM提多个峰值作SAM点提示；将SAM逐像素置信图乘以掩码内类别CAM均值，跨类argmax并设置背景 | 同一个CAM分类器通过CPM交叉熵学习区域级伪标签 |
| [[DiCLIP_paper_notes]] | CLIP CAM、扩散关系与视觉缓存 | 直接patch—text响应空间平滑且类内外观覆盖有限 | 静态缓存CAM先监督动态适配器；动态CAM与文本CAM融合、经PAR细化为伪掩码，再监督分割头 | 动态适配器与读取冻结CLIP特征的分割头，两级伪监督链 |
| [[ComCD_paper_notes]] | CLIP与扩散双CAM | 两路在不同像素可靠性不同，整图固定融合会保留各自错误 | 计算两路像素类别分布熵，按局部不确定性动态分配权重，再构造训练目标 | 下游WSSS分割分支；融合前需保证概率可比 |
| [[TokenMasking_paper_notes]] | 类别CLS注意图 | 低分辨率注意图存在空洞与重叠，非目标类也可能响应 | 先用图像标签筛选正类注意图，恢复patch网格、阈值化并填补未分配位置 | 后续全监督分割模型，注意图本身来自冻结推理 |

## 6. 工程实例：S2C的类别伪标签聚合

固定版本 [`102e14c`](https://github.com/sangrockEG/S2C/tree/102e14c690c8e3bce3d5ccd1ae7832145ce10b27)。[`models/model_s2c.py`](https://github.com/sangrockEG/S2C/blob/102e14c690c8e3bce3d5ccd1ae7832145ce10b27/models/model_s2c.py#L281-L300) 中，每个SAM掩码的像素分数由两部分相乘：

```python
sam_conf[i, k][target_mask] = (
    target_conf[target_mask] * cam_ms[i, k][target_mask].mean()
)
temp = sam_conf.max(dim=1)
pgt_sam = temp[1]
pgt_score = temp[0]
```

SAM置信度回答“这个掩码像不像一个完整区域”，掩码内CAM均值回答“这个区域像不像类别 $k$”。随后在类别维取最大值，得到每个像素的伪类别。主分支在[同一文件的CPM损失](https://github.com/sangrockEG/S2C/blob/102e14c690c8e3bce3d5ccd1ae7832145ce10b27/models/model_s2c.py#L345-L352) 中用交叉熵学习该结果。

### 6.1 固定版本与完整调用链

- 官方仓库：[sangrockEG/S2C](https://github.com/sangrockEG/S2C)
- commit：[`102e14c690c8e3bce3d5ccd1ae7832145ce10b27`](https://github.com/sangrockEG/S2C/tree/102e14c690c8e3bce3d5ccd1ae7832145ce10b27)
- 多尺度CAM与SAM编码：[model_s2c.py#L189-L230](https://github.com/sangrockEG/S2C/blob/102e14c690c8e3bce3d5ccd1ae7832145ce10b27/models/model_s2c.py#L189-L230)
- CAM峰值提示：[model_s2c.py#L233-L264](https://github.com/sangrockEG/S2C/blob/102e14c690c8e3bce3d5ccd1ae7832145ce10b27/models/model_s2c.py#L233-L264)
- SAM解码与置信聚合：[model_s2c.py#L266-L299](https://github.com/sangrockEG/S2C/blob/102e14c690c8e3bce3d5ccd1ae7832145ce10b27/models/model_s2c.py#L266-L299)
- CPM监督：[model_s2c.py#L345-L360](https://github.com/sangrockEG/S2C/blob/102e14c690c8e3bce3d5ccd1ae7832145ce10b27/models/model_s2c.py#L345-L360)

```text
当前主网络（eval + no_grad）
→ 0.5/1.0/1.5/2.0倍图像CAM求和
→ ReLU + 每类空间最大值归一化
→ 只保留图像标签存在的类别
→ 每类全局峰值 + 局部峰值
→ 坐标缩放后作为SAM正点提示
→ SAM编码器每图只运行一次，按类别重复解码
→ 取固定候选mask 2及其置信图
→ SAM置信 × 掩码内CAM均值
→ 跨类别max得到 pgt_sam
→ 主网络CAM + 背景通道
→ cross_entropy(cam_main, pgt_sam)
```

### 6.2 CAM峰值怎样变成SAM提示？

源码先保存全局最大值，再用3×3 maximum filter和 `peak_local_max(min_distance=20)` 找局部峰值，低于 `self.th_multi` 的峰被删除。展平索引通过整除/取余还原行列，之后 `np.flip` 变成SAM需要的 $(x,y)$ 顺序，并按 `self.size_sam/H` 缩放坐标。

因此一个类别可产生多个正点，适合同图多个实例；但点全部来自当前CAM，没有负点，若CAM在背景上有高峰，SAM会生成结构完整但语义错误的候选。

### 6.3 重叠类别怎样消解？

初始化：

```python
sam_conf = -1e5 * torch.ones_like(cam_ms)  # [B,C,H,W]
```

只有SAM目标掩码内的位置被赋值：

```python
sam_conf[i, k][target_mask] = (
    target_conf[target_mask]
    * cam_ms[i, k][target_mask].mean()
)
```

`target_conf[target_mask]` 是逐像素结构置信度，`cam_ms[...,target_mask].mean()` 是该掩码对类别 $k$ 的单个语义标量；二者广播相乘后仍是一组像素分数。随后 `sam_conf.max(dim=1)` 消去类别维，返回每个像素最高分及其类别索引。

### 6.4 代码与论文叙述之间的重要细节

- `idx_max_sam = 2` 是固定候选索引，不是运行时按SAM置信度选择三个候选中的最大项。
- 没有任何类别成功覆盖的位置保留负哨兵；代码将 `pgt_score < 0` 的像素设为类别20。在VOC的20前景类设置中，背景实际位于**最后一个通道索引20**。
- 主分支构造 `cam_bg = 1 - cam_main.max(dim=1)` 后用 `torch.cat((cam_main, cam_bg), dim=1)`，同样把背景追加到末尾；这与一些公式习惯把背景写成索引0不同。
- CPM损失传入 `ignore_index=255`，但这段聚合把未覆盖像素设为背景20，并没有在这里生成255 ignore。若想采用保守ignore策略，需要显式改目标构造。
- 生成多尺度CAM、提峰和SAM提示都在 `torch.no_grad()` 中；伪标签不会从CPM损失反向穿过SAM或峰值选择。真正更新的是后面的主网络CAM分支。
- CPM通过 `use_cpm = epo > sstart - 1` 延迟启用，避免训练初期CAM过差时立即产生错误提示。

## 7. 防止确认偏差

- 教师或伪标签分支通常 `detach`，避免目标随学生梯度一起移动。
- 只监督可靠位置，而不是追求伪标签100%覆盖。
- 分别统计前景、背景、ignore比例，防止阈值让模型只学背景。
- 多分支融合前检查置信度是否可比；logit尺度不同会让softmax熵失真。
- 保存细化前后可视化，确认提升来自补全而不是目标膨胀。

## 8. 硬标签、软标签与部分标签

并非所有细化结果都必须立刻argmax：

| 监督形式 | 保存的信息 | 优点 | 主要风险 |
|---|---|---|---|
| 硬标签 | 单一类别/ignore | 接口简单，可直接交叉熵 | 丢失不确定性 |
| 软标签 | 完整类别分布 | 保留类间关系 | 教师错误概率也被学习 |
| 部分标签 | 候选类别集合 | 不强迫模糊像素选唯一类 | 需要专门损失 |
| 区域标签 | mask + 区域类别分布 | 保留区域一致性 | 依赖提议覆盖和冲突处理 |

经典全监督分割直接使用人工硬标签；弱监督方法的核心是从不完整证据构造上述目标；开放词汇方法还要处理训练词表与测试词表不一致，因此伪标签最好保留可重新分类的区域，而不是只保存固定类ID。

## 9. 迭代更新与课程策略

单轮伪标签固定后训练分割器最稳，但无法利用学生逐渐改善的预测；在线更新更灵活，也更容易确认偏差。常见折中是：

```text
高精度、低召回种子
  → 训练若干轮/EMA教师
  → 只扩大满足一致性与边界约束的区域
  → 逐步降低阈值或增加监督覆盖
```

阈值变化应由预定schedule或验证集确定，不能根据测试结果反复选择。每轮记录前景、背景、ignore比例和与上一轮标签的变化率；标签大面积突变通常意味着训练不稳定。

## 10. 区域冲突与背景

同一像素被多个类别mask覆盖时，可比较“mask质量 × 区域内CAM/文本分数”，也可保留为ignore。将所有未覆盖位置直接设为背景会把漏检前景变成错误负样本；WSSS通常更适合保留一部分unknown/ignore。详细区分见 [[Background_and_Unknown_Handling]]。

CRF、超像素和SAM主要改善边界或区域一致性，不负责验证类别名。使用它们之后仍应检查：类别是否正确、目标是否被合并、其他同类实例是否遗漏。

## 11. 伪标签质量与最终模型要分开报告

- 伪标签mIoU/precision/recall及ignore比例；
- 前景与背景分别的错误率；
- 细化前后的边界F-score和连通组件统计；
- 用同一个分割器分别训练原CAM标签与细化标签，隔离标签质量收益；
- 最终模型是否仍需在线SAM/CRF/教师，明确训练和推理成本。

## 12. 选型与论文/源码索引

| 当前问题 | 优先细化方式 | 主要风险 |
|---|---|---|
| 种子精度高、召回低 | 亲和力/超像素传播 | 越界传播 |
| 边界粗但类别可靠 | SAM、CRF或区域提议 | 完整地放大错误类别 |
| 两路模型局部互补 | 校准后的置信融合 | 熵/最大概率不可比 |
| 在线伪标签随学生更新 | EMA教师或互教 + 可靠掩码 | 确认偏差 |
| 背景与未知前景混淆 | 前景/背景双阈值 + ignore | 把漏检前景当负样本 |
| 开放词表会变化 | 保留区域与连续文本分数 | 固定类ID伪标签失去开放性 |

导航：[[CLIP-ES_paper_notes]]、[[WeCLIP_paper_notes]]、[[UGRL_paper_notes]]、[[SSR_paper_notes]]、[[S2C_paper_notes]]、[[DiCLIP_paper_notes]]、[[ComCD_paper_notes]]。

源码入口：

- [S2C多尺度CAM](https://github.com/sangrockEG/S2C/blob/102e14c690c8e3bce3d5ccd1ae7832145ce10b27/models/model_s2c.py#L197-L218)：尺度融合、类别过滤和归一化。
- [S2C峰值到点提示](https://github.com/sangrockEG/S2C/blob/102e14c690c8e3bce3d5ccd1ae7832145ce10b27/models/model_s2c.py#L233-L264)：全局/局部峰值与坐标转换。
- [S2C类别聚合](https://github.com/sangrockEG/S2C/blob/102e14c690c8e3bce3d5ccd1ae7832145ce10b27/models/model_s2c.py#L266-L299)：SAM候选、置信乘法和背景赋值。
- [S2C伪标签损失](https://github.com/sangrockEG/S2C/blob/102e14c690c8e3bce3d5ccd1ae7832145ce10b27/models/model_s2c.py#L345-L360)：背景通道、交叉熵与主网络更新。

## 13. 当前整理结论

伪标签细化的核心不是把图“变好看”，而是控制监督噪声：

$$\boxed{\text{可信的就教，不确定的先忽略，空间补全必须有边界约束}}.$$

**公式解释：** 这是对本页决策规则的文字化总结，不是数值计算公式，因此没有输入张量、矩阵乘法或维度消去。它对应三步工程含义：高置信位置进入监督，低置信位置设为 ignore，扩大空间覆盖时必须借助亲和力、超像素、CRF 或 SAM 等边界约束防止越界传播。
