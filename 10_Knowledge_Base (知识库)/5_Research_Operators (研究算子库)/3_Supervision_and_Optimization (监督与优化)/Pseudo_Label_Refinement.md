---
type: operator-note
aliases: [Pseudo-Label Refinement, 伪标签细化]
tags: [research-operator, pseudo-label, refinement, weakly-supervised]
status: todo
---

# Pseudo-Label Refinement（伪标签细化）

> [!abstract] 核心直觉
> 伪标签细化不是单个算法，而是把连续响应图转换成“足够可靠、可以监督分割器”的离散标签，并显式放弃不确定位置。

## 1. 输入输出

输入类别概率：

$$P\in[0,1]^{B\times C\times H\times W},\qquad\sum_cP_{c,h,w}=1.$$

输出硬标签：

$$\tilde Y\in\{0,1,\ldots,C-1,255\}^{B\times H\times W}.$$

`255` 常用作ignore index，表示该像素不参与交叉熵。细化的目标不是强迫每个像素都有答案，而是平衡“覆盖更多像素”和“减少错误监督”。

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

可定义：

$$
\tilde Y=\begin{cases}
c^*,&q\ge\tau_{fg},\\
\text{background},&q\le\tau_{bg},\\
255,&\text{otherwise}.
\end{cases}
$$

若 $\tau_{fg}=0.7,\tau_{bg}=0.2$，概率0.85的像素成为前景，0.1成为背景，0.45不确定而被忽略。中间区不产生梯度，避免把模糊边界当成确定真值。

## 5. 论文对比

| 论文 | 初始噪声来源 | 细化信号 | 谁最终接受监督 |
|---|---|---|---|
| [[WeCLIP_paper_notes]] | 冻结CLIP CAM不完整 | 多层注意力、RFM与解码器预测 | 可训练解码器 |
| [[UGRL_paper_notes]] | CAM阴影与不确定区域 | PUM/ULM/RSE可靠性链 | 分类/分割主干 |
| [[SSR_paper_notes]] | 模态间隙和背景污染 | 原型对齐 + 超像素约束传播 | 单阶段模型 |
| [[S2C_paper_notes]] | CAM噪声提示 | SAM置信度和CAM区域均值 | 主分类器CAM |
| [[DiCLIP_paper_notes]] | CLIP密集知识有限 | 扩散亲和力、视觉缓存和分割头 | 动态适配器与分割头 |

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

## 12. 当前整理结论

伪标签细化的核心不是把图“变好看”，而是控制监督噪声：

$$\boxed{\text{可信的就教，不确定的先忽略，空间补全必须有边界约束}}.$$
