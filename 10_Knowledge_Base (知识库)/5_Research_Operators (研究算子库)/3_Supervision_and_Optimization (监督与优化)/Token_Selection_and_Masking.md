---
type: operator-note
aliases: [Token Selection, Token Masking, 令牌选择与掩码]
tags: [research-operator, token, masking, transformer]
status: todo
---

# Token Selection and Masking（令牌选择与掩码）

> [!abstract] 核心直觉
> token操作既可以筛掉不可靠视觉位置，也可以约束特殊token的职责。当前论文中的“Class-specific Token Masking”主要掩蔽部分**非目标类别CLS token的最终输出**，目的是促进CLS—类别分配；它不是普通的随机patch遮挡。

> [!tip] 基础机制入口
> token、CLS和多头自注意力的通用结构看 [[vision_transformer_notes]]。本页关注类别特定CLS掩码、注意力头稀疏化和可靠patch筛选。

## 1. 输入输出

token序列：

$$X\in\mathbb{R}^{B\times N\times D},$$

二值掩码：

$$m\in\{0,1\}^{B\times N}.$$

硬掩码输出：

$$X'=X\odot m[:,:,None].$$

形状仍是 `[B,N,D]`，被遮token变成零或可学习的mask token。若直接删除token，则输出长度变为 $N'<N$，速度更快，但必须同步更新位置编码和注意力掩码。

## 2. 常见形式

| 形式 | 选择规则 | 用途 | 风险 |
|---|---|---|---|
| 随机patch mask | Bernoulli或固定比例 | 迫使模型利用未遮挡视觉证据 | 比例过高导致语义丢失 |
| Top-k保留 | 按注意力/相似度 | 降噪与省计算 | 早期选择错误不可恢复 |
| 低置信剔除 | 按CAM/熵 | 构造可靠原型或伪标签 | 只剩最容易区域 |
| 类别token输出mask | 从非目标类别CLS中随机选择 | 促进CLS与指定类别的硬分配 | 需使用图像级标签区分目标/非目标 |
| 稀疏注意力mask | 限定允许交互的token对 | 控制空间传播 | mask构建本身有成本 |

## 3. 两种容易混淆的随机掩码

### 3.1 通用patch随机掩码

$$m_n\sim\operatorname{Bernoulli}(1-r),$$

$r$ 是遮挡率。若 $N=100,r=0.3$，期望保留70个token，但每次实际数量会波动。固定数量掩码则每张图严格遮30个，训练统计更稳定。

为保持期望幅值，dropout式掩码会使用：

$$X'=\frac{m\odot X}{1-r}.$$

但如果mask token代表“缺失位置”而不是普通dropout，未必应该做这个缩放。

### 3.2 Class-specific Token Masking的CLS输出掩码

设图像标签集合为 $Y$，第 $i$ 个CLS token固定对应类别 $i$。论文只从 $i\notin Y$ 的非目标CLS token中随机选择一部分：

$$
m(i)=\begin{cases}
0,&i\in Y,\\
1,&i\notin Y\text{ 且本轮被随机选中},\\
0,&\text{其他情况}.
\end{cases}
$$

随后掩蔽最终输出：

$$z_{[CLS]_i}^{L}\leftarrow z_{[CLS]_i}^{L}(1-m(i)).$$

因此，当前图像真正存在的类别CLS不会被该规则遮挡；被选中的非目标CLS输出变成零。论文还为每个注意力头学习门控，并用Hard Concrete分布近似 $L_0$ 正则来剪除冗余头。CLS输出掩码负责类别分配，注意力头稀疏化负责减少噪声，两者不是同一个操作。

## 4. 论文中的做法

| 论文 | 被操作的token | 目的 |
|---|---|---|
| [[TokenMasking_paper_notes]] | 多个类别特定CLS token的最终输出；另对注意力头加门控 | 随机遮挡部分非目标CLS以促进类别分配；稀疏注意力头以得到更清晰的类别注意图 |
| [[MCTformer_paper_notes]] | 多类别class tokens与patch tokens | 每类token学习类别特定注意力 |
| [[ExCEL_paper_notes]] | CLIP patch关系中的噪声响应 | 通过静态/动态视觉校准抑制不可靠关系 |
| [[DiCLIP_paper_notes]] | 缓存生成时的前景/背景patch | 阈值筛选后写入视觉知识缓存 |

## 5. 选择与掩码不是同一件事

- **选择**通常保留子集并缩短序列，可能真正减少计算。
- **值置零**仍保留序列长度，标准注意力复杂度没有下降。
- **attention mask**阻止被遮位置参与Q/K交互，但若仍计算完整矩阵，理论FLOPs未必下降。
- **稀疏内核**只有配合相应实现，才把逻辑稀疏转成实际加速。

## 6. 两种掩码的代码骨架

```python
# 通用patch masking；不是论文的CLS输出掩码
keep = torch.rand(x.shape[:2], device=x.device) > mask_ratio
x_masked = torch.where(keep.unsqueeze(-1), x, mask_token)

# 类别特定CLS输出掩码
# cls_out: [B, C, D], labels: [B, C]
absent = ~labels.bool()
sampled = torch.rand_like(labels.float()) < mask_ratio
mask_cls = absent & sampled
cls_out = cls_out.masked_fill(mask_cls.unsqueeze(-1), 0.0)
```

第一段改变视觉patch；第二段不删除token，也不减少前面Transformer编码的计算，只在最终输出处让部分非目标CLS不参与预测。实现时必须先确认论文的mask发生在输入、注意力矩阵还是最终输出。

## 7. 调试指标

- 实际保留率与设定值是否一致；
- 被掩CLS中目标类别与非目标类别的比例；目标CLS按论文规则应始终保留；
- 不同mask率下CLS—类别分配、伪掩码质量与分类准确率的权衡；
- 训练时mask、推理时不mask是否造成性能落差；
- 置零后被遮token是否仍通过残差或位置编码泄露信息。

## 8. 四个操作层级必须区分

| 操作位置 | 例子 | 是否改变序列长度 | 是否天然省计算 |
|---|---|---:|---:|
| 输入像素/patch | MAE式随机遮挡、Cutout | 可选 | 仅删除token时可能 |
| token值 | 置零或替换mask token | 否 | 否 |
| attention logit | 禁止某些Q-K对 | 否 | 标准稠密实现中否 |
| 输出token/损失 | 遮CLS输出、忽略低置信token | 否 | 否 |

论文使用“masking”时应定位到具体层级。[[TokenMasking_paper_notes]] 的类别CLS最终输出掩码主要改变监督分配，并不是MAE式patch重建，也不是推理token pruning。

## 9. attention mask的数学形式

对允许矩阵 $G\in\{0,1\}^{N_q\times N_k}$：

$$
A=\operatorname{softmax}\left(\frac{QK^T}{\sqrt D}+B_G\right),\qquad
(B_G)_{ij}=\begin{cases}0,&G_{ij}=1\\-\infty,&G_{ij}=0.\end{cases}
$$

掩码应加在softmax前，才能让禁止位置概率严格为0。softmax后再乘0会使每行权重和小于1，除非再次归一化。混合精度可用框架提供的布尔attention mask，减少手写极小数导致NaN的风险。

## 10. 三类分割中的不同用途

- **经典分割**：token pruning/稀疏attention主要追求高分辨率效率，选择标准应尽量不损失小物体与边界。
- **弱监督分割**：类别token、可靠patch和随机遮挡用于发现更完整证据或减少噪声，但图像级标签可参与选择。
- **开放词汇分割**：视觉token选择不能依赖固定训练类，否则会在测试时漏掉未见类；更适合类别无关objectness、区域提议或对所有候选文本动态打分。

## 11. 选择偏差与真实加速

Top-k选择形成不可逆瓶颈：被删的小目标、背景中的未见物体无法被后续文本分类恢复。训练早期可用软门控或较高保留率，推理再根据验证消融剪枝。若声称效率提升，应报告端到端延迟和峰值显存；稀疏率本身不等于硬件加速，尤其当实现仍构造完整 $N\times N$ attention时。

额外应检查每类/每尺度token保留率、小物体召回，以及保留token是否集中在已有CAM高响应区域。后者过强说明mask只强化已有判别区域，没有发现新证据。

## 12. 当前整理结论

令牌掩码既可以是数据增强，也可以是信息路由。实现前先明确目标是“发现更多证据”“排除噪声”还是“真正减少计算”，三者需要的mask方式并不相同。
