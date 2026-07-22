---
type: operator-note
aliases: [Spatial Propagation, 空间传播]
tags: [research-operator, propagation, random-walk, segmentation]
status: todo
---

# Spatial Propagation（空间传播）

> [!abstract] 核心直觉
> 初始CAM常只覆盖物体最有判别力的一小块。空间传播把这块“可靠种子”沿相似位置扩散，但需要边界或置信度阻止它蔓延到背景。

## 1. 输入与输出

种子响应 $M^{(0)}\in\mathbb{R}^{B\times N\times C}$，传播矩阵 $T\in\mathbb{R}^{B\times N\times N}$，迭代：

$$M^{(t+1)}=T M^{(t)}.$$

输出形状不变。$T[i,j]$ 表示位置 $j$ 对位置 $i$ 的贡献。若每行和为1，它可看作随机游走的转移概率。

## 2. 与注意力细化有什么区别？

[[Attention_and_Affinity_Refinement]] 更关注“关系矩阵怎样得到和修正”；本页更关注“给定关系后，证据传播几次、怎样停止、怎样防止越界”。两者通常前后相连。

## 3. 常见形式

| 形式 | 更新 | 优点 | 风险 |
|---|---|---|---|
| 单步传播 | $TM$ | 快、较少过平滑 | 覆盖范围有限 |
| 多步随机游走 | $T^kM$ | 可补全远处同类区域 | 错误随步数扩大 |
| 残差传播 | $(1-\lambda)M+\lambda TM$ | 保留原种子 | 需调节比例 |
| 边界约束传播 | $(T\odot G)M$ | 减少跨物体泄漏 | 边界错误会阻断补全 |
| 双分支传播 | 两种关系分别传播后融合 | 利用互补结构 | 成本与冲突处理增加 |

## 4. 一个小例子

三个位置的单类种子为 $M=[1,0,0]^T$，转移矩阵为：

$$T=\begin{bmatrix}0.6&0.4&0\\0.4&0.4&0.2\\0&0.2&0.8\end{bmatrix}.$$

一次传播后 $TM=[0.6,0.4,0]^T$：第2个位置获得0.4响应，第3个位置仍未激活。再传播一次才可能到达第3个位置。这解释了步数为什么控制扩散半径。

## 5. 论文中的传播路径

| 论文 | 种子 | 传播关系 | 防泄漏机制 |
|---|---|---|---|
| [[CLIP-ES_paper_notes]] | Softmax-GradCAM | CLIP注意力亲和力 | CAM候选框掩码 |
| [[SSR_paper_notes]] | CLIP初始CAM | 修正亲和矩阵随机游走 | 超像素同区域约束 |
| [[ComCD_paper_notes]] | CLIP/扩散CAM | 各分支空间关系和后续融合 | 像素熵决定更可信分支 |
| [[UGRL_paper_notes]] | 可靠伪标签 | 可靠区域学习/扩张 | 不确定性筛选 |
| [[MCTformer_paper_notes]] | class-token注意力与CAM | Transformer patch关系 | 类别感知训练与互补细化 |

## 6. 基础代码骨架

```python
# affinity: [B, N, N], cams: [B, N, C]
transition = affinity / affinity.sum(dim=-1, keepdim=True).clamp_min(1e-6)
refined = cams
for _ in range(steps):
    refined = torch.bmm(transition, refined)
refined = (1 - lam) * cams + lam * refined
```

`torch.bmm` 按batch做矩阵乘法。不要把 `[B,N,C]` 错写成 `[B,C,N]` 后直接相乘；先明确哪一维代表被传播的位置。

## 7. 工程实例：CLIP-ES的一步传播

CLIP-ES固定版本 [`3893f81`](https://github.com/linyq2117/CLIP-ES/tree/3893f817be359c5ee1dbf8111cad381a532c7acc) 的 [`generate_cams_voc12.py`](https://github.com/linyq2117/CLIP-ES/blob/3893f817be359c5ee1dbf8111cad381a532c7acc/generate_cams_voc12.py#L168-L196) 将CAM展平成 `[N,1]`，执行 `trans_mat @ cam_to_refine`，再reshape回 $H'\times W'$ 并放大到原图。源码清楚地表明：先空间传播，后二维恢复和插值；插值本身不负责语义补全。

## 8. 怎样判断该不该传播？

- 类别对、区域小：适合传播补全。
- 类别本身错：先修正 [[Cross_Modal_Alignment]] 或背景建模。
- 边界被明显越过：加入超像素/SAM限制，或减少步数与 $\lambda$。
- 小物体被抹掉：保留残差、使用局部稀疏邻接，避免过多迭代。
- 显存不足：使用局部窗口、K近邻稀疏矩阵，而不是完整 $N^2$ 关系。

## 9. 与CRF、解码器和亲和力学习的区别

| 方法 | 传播依据 | 是否学习 | 主要输出 |
|---|---|---:|---|
| 解码器 | 多层特征与卷积/注意力 | 通常是 | 直接预测分割logit |
| DenseCRF | 颜色、位置与边缘势函数 | 通常否 | 后处理后的标签分布 |
| 随机游走 | 给定图邻接/转移矩阵 | 可选 | 扩散后的种子响应 |
| 亲和力学习 | 标签或伪标签监督的像素对关系 | 是 | 供随机游走使用的边 |

经典全监督模型可依靠像素真值训练解码器；弱监督方法没有完整真值，因而常先从CAM构造可靠种子，再借空间关系扩张。开放词汇方法也会传播文本类别响应，但关系通常来自类别无关模型；传播只能补结构，不能创造新的类别锚点。

## 10. 带重启的传播更稳

纯多步传播 $T^kM^{(0)}$ 容易忘掉初始证据。可每步拉回种子：

$$
M^{(t+1)}=\alpha TM^{(t)}+(1-\alpha)M^{(0)},\qquad0\le\alpha<1.
$$

固定点满足：

$$M^*=(1-\alpha)(I-\alpha T)^{-1}M^{(0)}.$$

工程上通常迭代若干步，不显式求逆。$\alpha$ 小更信任种子，$\alpha$ 大传播更远。若 $T$ 行随机且 $\alpha<1$，迭代通常比无重启的反复乘法稳定。

还可以先加入边界门控 $G$ 并重新归一化：

$$
\tilde T_{ij}=\frac{T_{ij}G_{ij}}{\sum_kT_{ik}G_{ik}+\varepsilon}.
$$

只做逐元素相乘而不重归一化会让某些行总质量显著变小，响应强度随迭代衰减。

## 11. 传播半径和过平滑

步数不是抽象超参数：在局部邻接图中，它近似限制图上的最远传播距离；在含全局KNN边的图中，一步就可能跨越很远空间。多步后各位置分布逐渐相似，是图平滑的自然结果。可以监控平均像素熵、边界两侧分数差和类别响应方差，判断是否过平滑。

## 12. 正确的消融方式

- 固定种子，替换关系源与传播步数，避免初始CAM不同造成混淆。
- 同时报种子精度/召回与传播后精度/召回；理想传播提高召回且精度下降有限。
- 分别分析同物体补全、跨同类实例传播和跨类别泄漏。
- 将传播结果与简单双线性上采样、形态学膨胀和CRF比较，确认复杂图传播确有额外价值。

## 13. 当前整理结论

空间传播主要决定“正确证据能走多远”。关系质量、传播步数和边界约束三者缺一不可。
