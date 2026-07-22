---
type: operator-note
aliases:
  - Multi-Level Fusion
  - 多层级融合
tags:
  - research-operator
  - fusion
  - multi-scale
  - multi-level
status: todo
---

# Multi-Level Fusion（多层级融合）

> [!abstract] 核心直觉
> 最终层语义强但空间细节不足；浅层细节多却不容易区分类别。多层融合的目的，是把互补信息变成一个可供预测头使用的统一张量。

> [!tip] 基础机制入口
> 编码器—解码器、上采样和跳跃连接看 [[fcn_notes]]，Transformer多层重组看 [[dpt_notes]]，FPN/PSP式层级融合看 [[upernet_notes]]。本页比较这些基础操作在当前论文中如何被改造成多骨干、多分支或动态融合。

## 1. 融合前必须先统一什么？

设第 $l$ 层特征：

$$F_l\in\mathbb{R}^{B\times D_l\times H_l\times W_l}.$$

不同层通常通道数和分辨率都不同。先投影通道：

$$\bar F_l=\phi_l(F_l)\in\mathbb{R}^{B\times D\times H_l\times W_l},$$

再插值到共同网格：

$$\tilde F_l=\operatorname{Resize}(\bar F_l,H_*,W_*)\in\mathbb{R}^{B\times D\times H_*\times W_*}.$$

$\phi_l$ 常用 $1\times1$ 卷积或线性层。它只混合通道，不扩大空间感受野。完成这两步后，张量才可以安全相加或拼接。

## 2. 常见实现形式

| 形式 | 输出 | 优点 | 代价/风险 |
|---|---|---|---|
| 加权求和 | $\sum_l\alpha_l\tilde F_l$ | 参数少、输出维度不变 | 必须维度一致；信息混在一起 |
| concat + projection | $\phi([F_1;\cdots;F_L])$ | 信息保留较多 | 通道、显存和参数增加 |
| cross-attention | Query与多层Key/Value交互 | 能按内容选择信息 | 计算通常为二次复杂度 |
| layer selection | 直接选若干层 | 简单可解释 | 依赖人工经验，不能逐图变化 |
| dynamic gating | 每层/每位置学习门控 | 灵活、可样本自适应 | 需训练并防止权重塌缩 |

## 3. 公式要怎样读？

### 3.1 加权求和

$$F_{out}=\sum_{l=1}^{L}\alpha_l\tilde F_l,\qquad
\alpha_l=\frac{e^{a_l}}{\sum_j e^{a_j}}.$$

softmax保证各层权重非负且总和为1。若三层得分为 `[0,0,0]`，权重就是 `[1/3,1/3,1/3]`；如果一层得分长期远大于其余层，融合会退化为单层选择，这就是权重塌缩。

### 3.2 拼接后投影

$$F_{cat}=\operatorname{Concat}(\tilde F_1,\ldots,\tilde F_L)\in
\mathbb{R}^{B\times LD\times H_*\times W_*},$$

$$F_{out}=\operatorname{Conv}_{1\times1}(F_{cat})\in
\mathbb{R}^{B\times D_o\times H_*\times W_*}.$$

拼接发生在通道维，空间尺寸不变。若4层各256通道，拼接后是1024通道；投影层再压到256通道。

### 3.3 动态门控

$$G=\sigma(g(F_1,\ldots,F_L)),\qquad
F_{out}=G\odot F_a+(1-G)\odot F_b.$$

$G$ 可以是每张图一个数、每通道一个数，也可以是 `[B,1,H,W]` 的位置权重。粒度越细越灵活，也越容易过拟合。

## 4. 论文中的具体实例

| 论文 | 融合对象 | 融合方式 | 解决的问题 |
|---|---|---|---|
| [[WeCLIP_paper_notes]] | 冻结CLIP多个中间层 | 解码器逐级解释与汇聚 | 最终层空间细节不足 |
| [[MCTformer_paper_notes]] | class-token注意力与patch-token CAM | 互补相加/细化 | 类别token定位与分类CAM各有缺失 |
| [[ComCD_paper_notes]] | CLIP CAM与扩散CAM | 熵差形成像素级动态权重 | 两个分支在不同位置可靠性不同 |
| [[Trident_paper_notes]] | 子图CLIP/DINO特征与SAM亲和力 | 先拼接局部特征，再全局聚合 | 滑窗预测缺少跨窗口上下文 |
| [[DiCLIP_paper_notes]] | 文本CAM、静态缓存CAM、动态缓存CAM | 固定加权并提供多路监督 | 单一patch-text响应覆盖不足 |

## 5. 工程实例：DiCLIP中的最小融合

固定版本 [`1c3f6ff`](https://github.com/zwyang6/DiCLIP/tree/1c3f6ff7d4fde2afff32d527d78b28d119583602) 的 [`forward`](https://github.com/zwyang6/DiCLIP/blob/1c3f6ff7d4fde2afff32d527d78b28d119583602/model/model_diclip.py#L148-L185) 中：

```python
fuse = 0.5 * diff_maps + attr_maps_raw
```

两者都是 `[B,N,C]`，所以可以逐元素相加。这不是concat，也没有可学习门控：缓存分支固定乘0.5，文本CAM系数相当于1。它简单、稳定，但不能针对不同图像或位置调整信任度。若换成门控，首先要决定 $G$ 是 `[B,1,1]`、`[B,N,1]` 还是 `[B,N,C]`。

## 6. 怎样选择？

- 两个信号同形状、希望先验证互补性：加权求和。
- 不确定哪些通道有用：concat + projection。
- 不同位置可靠性明显不同：位置级动态门控。
- 两组token需要显式交换信息：cross-attention，但先评估 $N_qN_k$ 成本。
- 层数很多：先用消融筛掉无效层，再训练门控，避免把所有层无差别堆入。

## 7. 经典分割提供的四种基线

| 基线 | 融合方向 | 主要解决的问题 | 对弱监督/开放词汇的启示 |
|---|---|---|---|
| [[fcn_notes]] 跳跃连接 | 深层上采样后与浅层预测融合 | 恢复轮廓 | 浅层细节不能替代类别语义 |
| [[deeplabv3+_notes]] ASPP + 低层解码 | 同层多感受野 + 浅深层融合 | 多尺度上下文与边界 | “多尺度”既可指输入尺度，也可指感受野，需分清 |
| [[upernet_notes]] PPM/FPN | 自顶向下逐级融合 | 统一多层特征金字塔 | 适合作为冻结骨干的通用消费者 |
| [[segformer_notes]] All-MLP decoder | 四级特征统一通道/尺度后融合 | 用轻量头解释分层Transformer | 投影与插值本身已是关键对齐操作 |

[[mask2former_notes]] 的多尺度策略又增加了query与像素特征的交互，它不是简单把四层特征一次拼接。阅读论文时应先判断融合发生在“特征图—特征图”“query—特征图”还是“类别响应—类别响应”之间。

## 8. 特征级融合与分数级融合

特征级融合发生在分类前：

$$F=\Phi(F^{(1)},F^{(2)}),\qquad Z=h(F).$$

它允许下游头学习跨来源组合，但要求空间和通道兼容。分数级融合发生在各分支已经输出同一类别集合后：

$$Z=\sum_m\alpha_mZ^{(m)}.$$

它更容易解释，却要求不同分支的类别顺序、背景定义和logit尺度一致。CLIP相似度、扩散CAM与分割头logit数值范围通常不同，融合前应做温度或逐分支归一化；不能因形状相同就直接相加并把系数解释为概率。

开放词汇场景还要检查文本词表是否共享。若一个分支只输出训练类、另一个可输出任意文本类，则应先在共同候选集合上重新计算或映射，而不是用零填充假装兼容。

## 9. 位置级动态融合

对语义强但边界粗的分支 $Z_s$ 与结构强但类别弱的分支 $Z_g$，可用不确定性门控：

$$
G_{h,w}=\operatorname{softmax}\bigl(g(H_s,H_g,F)\bigr),
\qquad
Z_{h,w}=G_1Z_{s,h,w}+G_2Z_{g,h,w}.
$$

若结构分支没有类别logit，应先把它作为亲和矩阵或区域约束作用于语义分支，而不是强行做分数加权。[[ComCD_paper_notes]] 属于两路CAM置信融合；[[Trident_paper_notes]] 更接近多骨干结构关系融合，二者不能仅用“动态融合”概括。

## 10. 诊断与消融

- 分别保存每个分支和融合输出，报告单分支、等权、固定权重与学习门控。
- 对门控权重做直方图和空间图，检查是否长期饱和到0或1。
- 逐项验证通道投影、空间插值和融合规则，避免把对齐收益误归给门控。
- 报告大/中/小目标以及边界指标；多尺度融合的提升应在这些子集上有可解释变化。
- 多模型融合要同时报告训练与推理依赖，不能只报告可训练参数而忽略冻结大骨干的FLOPs和显存。

## 11. 当前整理结论

多层融合不是“把特征放在一起”这么简单。实际顺序应是：

$$\boxed{\text{统一语义维度}\rightarrow\text{统一空间坐标}\rightarrow\text{选择融合规则}\rightarrow\text{检查是否塌缩}}.$$
