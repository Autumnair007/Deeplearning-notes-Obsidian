---
type: operator-note
aliases: [Attention Refinement, Affinity Refinement, 注意力与亲和力细化]
tags: [research-operator, attention, affinity, refinement]
status: todo
---

# Attention and Affinity Refinement（注意力与亲和力细化）

> [!abstract] 核心直觉
> CAM回答“每个位置像什么类别”，亲和力回答“两个位置是否应该一起变化”。细化算子利用后一种关系修正前一种响应。

> [!tip] 基础机制入口
> $QK^T$、多头注意力和残差块的基础实现看 [[vision_transformer_notes]] 与 [[vision_transformer_code_notes]]；DINOv2注意力的代码级解释可看 [[dinov2_code_notes_detailed]]。本页关注如何把关系矩阵用于CAM细化。

## 1. 输入输出张量

初始类别响应：

$$M\in\mathbb{R}^{B\times N\times C}.$$

位置亲和矩阵：

$$A\in\mathbb{R}^{B\times N\times N}.$$

细化后仍为：

$$M'=\bar A M\in\mathbb{R}^{B\times N\times C}.$$

矩阵乘法消去第二个位置维。$M'[b,i,c]$ 是其他位置 $j$ 的类别分数按 $A[b,i,j]$ 加权汇总后的结果。

## 2. 亲和力从哪里来？

| 来源 | 形式 | 优点 | 风险 | 论文 |
|---|---|---|---|---|
| Transformer自注意力 | $QK^T$或注意力权重 | 可直接复用骨干 | 可能过平滑或含类间错误相关 | [[WeCLIP_paper_notes]]、[[CLIP-ES_paper_notes]] |
| 特征自相似 | $FF^T$ | 简单且不依赖注意力头 | 受特征质量影响 | [[ExCEL_paper_notes]] |
| DINO相关性 | 自监督patch相似度 | 局部结构通常较好 | 没有类别语义 | [[CorrCLIP_paper_notes]] |
| 扩散模型亲和力 | 去噪网络中间关系 | 空间一致性丰富 | 提取成本与尺度对应复杂 | [[DiCLIP_paper_notes]]、[[ComCD_paper_notes]] |
| 超像素/SAM掩码 | 同区域为1、跨区域为0 | 边界约束直观 | 依赖区域提议 | [[SSR_paper_notes]]、[[Trident_paper_notes]] |

## 3. 基础计算

### 3.1 特征自相似

$$A=\operatorname{softmax}\left(\frac{\hat F\hat F^T}{\tau}\right).$$

$F:[B,N,D]$ 与 $F^T:[B,D,N]$ 相乘得到 `[B,N,N]`。第 $i$ 行描述位置 $i$ 应该从哪些位置收集信息；softmax通常沿最后一维做，使每行和为1。温度 $\tau$ 小，权重更尖锐；太小会只保留少数邻居。

### 3.2 掩码限制交互范围

$$A'_{ij}=\begin{cases}A_{ij},&G_{ij}=1\\-\infty,&G_{ij}=0\end{cases},\qquad
\bar A=\operatorname{softmax}(A').$$

$G$ 可来自SAM或超像素。把禁止位置设为 $-\infty$ 后，softmax概率变为0。工程上常用一个很小的有限数，混合精度下要检查是否产生NaN。

### 3.3 残差式细化

$$M'=(1-\lambda)M+\lambda\bar A M.$$

直接用 $\bar AM$ 可能把错误关系放大；残差保留原始CAM。$\lambda=0$ 完全不传播，$\lambda=1$ 完全依赖亲和力。

## 4. 论文具体差异

| 论文 | 修正了什么 | 关系来源 | 关键限制 |
|---|---|---|---|
| [[CLIP-ES_paper_notes]] | Grad-CAM | CLIP多头自注意力 | 用CAM候选框限制传播范围 |
| [[WeCLIP_paper_notes]] | 初始CAM与解码器关系 | 多层冻结CLIP注意力 | 学习筛选哪些注意力层可靠 |
| [[CorrCLIP_paper_notes]] | CLIP patch相关性 | SAM范围 + DINO值重建 | 在文本分类前修视觉相关结构 |
| [[SSR_paper_notes]] | CAM随机游走 | CLIP亲和力 + 超像素 | 防止跨越区域边界污染背景 |
| [[DiCLIP_paper_notes]] | CLIP注意力 | 扩散亲和力聚类后形成偏置 | 递归注入多种空间关系 |

## 5. 工程实例：CLIP-ES的注意力传播

固定版本 [`3893f81`](https://github.com/linyq2117/CLIP-ES/tree/3893f817be359c5ee1dbf8111cad381a532c7acc)。在 [`generate_cams_voc12.py`](https://github.com/linyq2117/CLIP-ES/blob/3893f817be359c5ee1dbf8111cad381a532c7acc/generate_cams_voc12.py#L162-L196) 中：

```python
trans_mat = aff_mat / torch.sum(aff_mat, dim=0, keepdim=True)
trans_mat = trans_mat / torch.sum(trans_mat, dim=1, keepdim=True)
trans_mat = trans_mat * aff_mask
cam_refined = torch.matmul(trans_mat, cam_to_refine)
```

代码先对亲和矩阵做两次归一化，再用候选框掩码限制传播，最后执行 `[N,N] @ [N,1] → [N,1]`。注意这里是一次矩阵传播，不是卷积。候选框来自当前CAM，因此它能减少远距离泄漏，但也可能限制CAM向真实框外缺失区域扩展。

## 6. 计算与调试

- 完整亲和矩阵显存为 $O(N^2)$；$N=4096$ 时单张float32矩阵约64 MB，还未计算梯度与多头开销。
- 检查归一化方向：行归一化表示“每个接收位置从谁收集”；列归一化语义不同。
- 可视化若干行而不是只看矩阵均值，确认邻居是否集中在同一物体。
- 多次传播可能过平滑；观察迭代后边界和类别熵。
- 亲和力只有结构，没有类别，不能单独替代CAM。

## 7. 注意力、相似度和转移矩阵不是同一个对象

这三个矩阵都可能是 `[B,N,N]`，但语义不同：

| 名称 | 常见计算 | 是否归一化 | 典型含义 |
|---|---|---|---|
| attention logit | $QK^T/\sqrt D$ | 否 | softmax前的交互证据 |
| attention weight | $\operatorname{softmax}(QK^T/\sqrt D)$ | 通常逐行 | 当前token从哪些token读取Value |
| feature affinity | $\hat F\hat F^T$ | 可选 | 两位置表征是否相似 |
| transition matrix | $D^{-1}A$ | 必须明确方向 | 传播一步时证据如何流动 |

因此不能仅因形状相同就互换。Transformer实际输出还经过 $AV$、输出投影、残差和MLP；注意力权重不是模型最终贡献度的严格解释。把它用于WSSS是一个有效结构先验，但需要用消融验证，而不是把它当成像素真值。

## 8. 构造稳定亲和力的常见处理

原始相似度可先去负值、对称化、加入自环并归一化：

$$
A_+=\max(A,0),\qquad
A_s=\frac{A_++A_+^T}{2},\qquad
\tilde A=A_s+\gamma I,
$$

$$
T=D^{-1}\tilde A,\qquad D_{ii}=\sum_j\tilde A_{ij}.
$$

对称化表达无向邻接，行归一化后才得到随机游走转移。若任务需要有向信息流，就不应盲目对称化。自环 $\gamma I$ 保留当前位置证据，也能避免孤立节点行和为0。

完整 $N^2$ 图并非必要。可只连接空间窗口或特征Top-k邻居：

$$
E=E_{\text{local}}\cup E_{\text{knn}},
$$

前者守住边界附近的局部连续性，后者允许同一物体相隔较远部分互相补全。稀疏图只有配合稀疏存储/算子才真正节省显存。

## 9. 三类分割中的角色

- **经典全监督分割**：non-local/context attention直接改善特征，错误可由像素真值纠正。
- **弱监督分割**：亲和力主要负责把稀疏CAM种子扩为完整物体，监督噪声使越界传播风险更高。
- **开放词汇分割**：DINO/SAM等类别无关关系修复CLIP的空间结构，但最终类别仍必须来自文本或开放分类器。

这解释了[[CorrCLIP_paper_notes]]与[[SSR_paper_notes]]的共同点和差异：二者都限制错误相关性，前者在开放词汇分类前重建视觉交互，后者在弱监督CAM后用超像素限制随机游走。

## 10. 评价细化是否真的有效

同时比较细化前后CAM mIoU、边界F-score、前景召回和背景误激活。只看响应面积变大可能只是目标膨胀。还应做“oracle affinity”或“固定CAM替换关系源”的消融，以区分收益来自更好的初始语义还是更好的关系矩阵。

## 11. 当前整理结论

注意力细化的实质是：

$$\boxed{\text{先定义允许的信息通路，再沿通路重分配类别证据}}.$$

它最适合补全已有正确种子；若初始类别完全错误，传播往往只会扩大错误。
