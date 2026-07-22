---
type: operator-note
aliases: [Contrastive Regularization, 对比正则]
tags: [research-operator, contrastive-learning, prototype]
status: todo
---

# Contrastive Regularization（对比正则）

> [!abstract] 核心直觉
> 对比学习不直接规定每个像素的类别logit，而是在特征空间要求“应该相似的靠近，不应该相似的分开”。它常作为伪标签监督之外的结构约束。

> [!tip] 基础机制入口
> 图文对比预训练、温度和对称损失看 [[clip_paper_notes]]；自蒸馏与patch级目标看 [[dinov2_notes]]。本页只整理分割任务如何构造像素、区域和原型级正负样本。

## 1. 输入输出

查询特征 $Q\in\mathbb{R}^{M\times D}$，正样本/原型 $P^+\in\mathbb{R}^{M\times D}$，负样本集合 $P^-\in\mathbb{R}^{K\times D}$。输出通常是标量损失。

InfoNCE形式：

$$
\mathcal L_i=-\log
\frac{\exp(\operatorname{sim}(q_i,p_i^+)/\tau)}
{\exp(\operatorname{sim}(q_i,p_i^+)/\tau)+\sum_k\exp(\operatorname{sim}(q_i,p_k^-)/\tau)}.
$$

分子是正确配对，分母加入所有候选。最小化损失等价于提高正样本相似度，同时压低负样本相似度。

## 2. 正负样本怎样定义？

| 粒度 | 正样本 | 负样本 | 论文 |
|---|---|---|---|
| 类别原型 | 同类视觉/文本原型 | 其他类别原型 | [[SSR_paper_notes]]、[[UGRL_paper_notes]] |
| 区域/片段 | 同一SAM片段内像素 | 其他片段像素/原型 | [[S2C_paper_notes]] |
| 图文 | 配对图像与标题 | batch内其他标题 | [[OpenSeg_paper_notes]]、[[Talk2DINO_paper_notes]] |
| 属性/描述 | 同类属性与视觉特征 | 不相关属性 | [[VDA_paper_notes]] |

## 3. 温度和归一化

通常先做：

$$\hat q=q/\|q\|_2,\qquad\hat p=p/\|p\|_2.$$

此时点积就是余弦相似度。$\tau$ 越小，模型越强烈关注最难区分的候选，但梯度也更尖锐。若忘记归一化，模型可能只增大向量模长来降低损失，而不是学到方向上的语义结构。

## 4. 原型对比与像素对比

像素两两对比需要 $O(N^2)$ 配对；原型对比把比较对象压到 $C$ 或 $CK$ 个：

$$S=\hat F\hat P^T\in\mathbb{R}^{B\times N\times C}.$$

它更省内存，也对单个伪标签错误更稳，但原型平均会损失细粒度类内结构。可结合 [[Prototype_Construction]] 的多原型折中。

## 5. 论文对比

| 论文 | 对齐对象 | 主要目的 | 噪声控制 |
|---|---|---|---|
| [[SSR_paper_notes]] | 投影视觉/文本特征与跨模态原型 | 缩小CLIP密集模态间隙 | 先筛纯净前景并聚类 |
| [[S2C_paper_notes]] | 像素特征与SAM片段原型 | 向分类器传递SAM区域结构 | 片段作为自监督分组 |
| [[UGRL_paper_notes]] | 可靠特征与类别锚点 | 避免不确定区域主导训练 | 不确定性筛选 |
| [[OpenSeg_paper_notes]] | 区域嵌入与标题词语 | 学习开放词汇区域语义 | 区域级聚合减少像素噪声 |
| [[Talk2DINO_paper_notes]] | DINO图像与映射后的CLIP文本 | 给自监督特征增加语言接口 | 图像—标题配对监督 |

## 6. 工程实例：S2C的SSC数据准备

S2C固定版本 [`102e14c`](https://github.com/sangrockEG/S2C/blob/102e14c690c8e3bce3d5ccd1ae7832145ce10b27/models/model_s2c.py#L327-L343) 先做：

```python
feat_main = F.interpolate(feat_main, size=(H, W), mode="bilinear")
feat_main = F.normalize(feat_main, dim=1)
feat_main_ = feat_main.view(B, D, -1)
index_ = self.se.view(B, 1, -1).long()
```

这几行分别统一空间尺寸、沿通道维归一化、展平像素、准备SAM片段索引。对比损失之前的数据整理非常关键：如果在错误维度归一化，或片段索引与插值后的像素顺序不一致，公式正确也无法工作。

## 7. 常见失败模式

- 伪标签错误变成“错误正样本”，把不同类别强行拉近。
- batch太小导致负样本不足；可使用队列，但要处理陈旧特征。
- 背景内部高度多样，强行压成一个背景原型可能不合理。
- 简单随机负样本太容易，损失很快饱和；困难负样本又可能是假负样本。
- 对比损失权重过大，会改善特征聚类却损害最终像素分类校准。

## 8. 三种监督范式如何构造正负样本

| 范式 | 正样本依据 | 负样本依据 | 可靠性来源 |
|---|---|---|---|
| 全监督分割 | 像素真值相同 | 真值类别不同 | 人工像素标注 |
| 弱监督分割 | CAM/伪标签/同一片段 | 伪类别不同或其他原型 | 阈值、超像素、SAM、教师 |
| 开放词汇分割 | 区域与配对词语/标题 | batch内其他文本或类别 | 图文配对、文本过滤、区域提议 |

弱监督中最危险的是假正样本：错误伪标签会主动把异类特征拉近。开放词汇中还存在假负样本，例如两条标题都描述“dog”，却因来自不同图像被当作batch负样本。多标签图文数据不能机械套用单标签实例判别假设。

## 9. 像素采样与类别平衡

密集特征数量很大，通常不对全部像素做两两对比。可按类别从高置信区域采样 $m_c$ 个query，并优先加入边界或相似类别困难样本。损失按类平均：

$$
\mathcal L=\frac1{|\mathcal C_I|}\sum_{c\in\mathcal C_I}
\frac1{m_c}\sum_{i=1}^{m_c}\ell(q_i^c),
$$

避免大物体和背景仅凭像素数量主导优化。背景高度多峰，更适合多个背景原型或采样多种背景簇，而不是压成一个中心。

## 10. 关系约束不一定需要InfoNCE

如果只有正对，没有可信负样本，可用余弦回归、BYOL式停止梯度目标或区域一致性；如果目标是保持教师关系，可匹配相似度矩阵：

$$
\mathcal L_{rel}=\left\|\hat F_s\hat F_s^T-operatorname{sg}(\hat F_t\hat F_t^T)\right\|_F^2.
$$

`sg` 表示stop-gradient。关系损失复杂度为 $O(N^2)$，可在采样token、局部窗口或区域原型上计算。

## 11. 训练与评价检查

- 明确正负样本是否跨图像、跨类别、跨模态以及是否含背景。
- 分布式训练若使用全局负样本，应正确all-gather并处理本样本索引。
- memory queue中的旧特征来自过去编码器；动量编码器可减小表示漂移。
- 同时看线性/最近原型分类、类内类间相似度和最终mIoU；更漂亮的聚类图不保证更好的边界。
- 消融温度、采样数量和损失权重时固定伪标签，避免监督质量同时变化。

## 12. 当前整理结论

对比正则的关键不是套用InfoNCE，而是定义可靠的正负关系。样本构造错了，损失会非常有效地学错。
