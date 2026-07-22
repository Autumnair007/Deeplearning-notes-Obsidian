---
type: operator-note
aliases:
  - Token Selection
  - Token Masking
  - 令牌选择与掩码
tags:
  - research-operator
  - token
  - masking
  - transformer
  - weakly-supervised
  - semantic-segmentation
status: in-progress
---

# Token Selection and Masking（令牌选择与掩码）

> [!abstract] 本页定位
> token操作既可以筛掉不可靠视觉位置，也可以约束特殊token的职责。本页区分输入patch遮挡、token值替换、注意力边掩码、最终类别token输出掩码和真实token删除，并结合经典视觉Transformer（Vision Transformer, ViT）、弱监督语义分割（Weakly-Supervised Semantic Segmentation, WSSS）与开放词汇分割说明选型。重点论文“Class-specific Token Masking”主要掩蔽部分**非目标类别CLS token的最终输出**，目的是促进CLS—类别分配；它不是普通随机patch遮挡。

> [!tip] 基础机制入口
> token、CLS和多头自注意力的通用结构看 [[vision_transformer_notes]]。本页关注类别特定CLS掩码、注意力头稀疏化和可靠patch筛选。

## 1. 输入输出

token序列：

$$X\in\mathbb{R}^{B\times N\times D},$$

**公式解释：** $X$ 是 token 序列，shape 为 `[B,N,D]`；$B$ 是批量大小，$N$ 是序列中的 token 数，$D$ 是每个 token 的特征维。该式只声明输入，没有发生矩阵运算或维度消去；$X[b,n,:]$ 是第 $b$ 张图第 $n$ 个 token 的 $D$ 维表示。

二值掩码：

$$m\in\{0,1\}^{B\times N}.$$

**公式解释：** $m$ 是与 token 一一对应的二值掩码，shape 为 `[B,N]`；$m[b,n]=1$ 表示保留该 token，0 表示遮挡。它不含特征维 $D$，应用到 $X$ 时需要新增一个长度为 1 的末维并广播。

硬掩码输出：

$$X'=X\odot m[:,:,None].$$

**公式解释：** `m[:,:,None]` 把 `m=[B,N]` reshape 为 `[B,N,1]`，随后在特征维 $D$ 广播，与 `X=[B,N,D]` 逐元素相乘；这里没有求和，也不消去任何维度，输出 `X'=[B,N,D]`。$X'[b,n,d]=X[b,n,d]m[b,n]$，所以同一 token 的全部 $D$ 个分量会一起保留或归零。

形状仍是 `[B,N,D]`，被遮token变成零或可学习的mask token。若直接删除token，则输出长度变为 $N'<N$，速度更快，但必须同步更新位置编码和注意力掩码。

对于类别特定token方法，序列更准确地写成：

$$
X_0=[X_{cls}^{(1)},\ldots,X_{cls}^{(C)},X_{patch}^{(1)},\ldots,X_{patch}^{(N)},X_{reg}]
\in\mathbb{R}^{B\times(C+N+1)\times D}.
$$

**公式解释：** 该式沿序列维拼接 $C$ 个类别 token、$N$ 个 patch token 和 1 个寄存器 token，每个 token 都是 $D$ 维；拼接不做加和，因此没有维度被消去，输出序列长度是 $C+N+1$，shape 为 `[B,C+N+1,D]`。$X_0[b,i,:]$ 的含义由位置决定：前 $C$ 个负责类别，中间 $N$ 个对应空间 patch，最后一个保存通用上下文。

$C$ 个类别标记（class token, CLS token）各自对应一个类别，$N$ 个patch token保存空间单元，最后的寄存器token（register token, REG token）吸收通用上下文。若 $B=2,C=20,N=196,D=384$，完整序列是 `[2,217,384]`。

最终类别token输出：

$$
Z_{cls}\in\mathbb{R}^{B\times C\times D},
$$

**公式解释：** $Z_{cls}$ 是 Transformer 最后一层的类别 token 输出，shape 为 `[B,C,D]`；它从完整序列中切出前 $C$ 个位置，序列中的 $N$ 个 patch 和 REG 位置不再出现在该张量中。$Z_{cls}[b,c,:]$ 是类别 $c$ 的 $D$ 维全局表示，这里尚未把特征维归约成分类 logit。

类别存在性标签：

$$
Y\in\{0,1\}^{B\times C}.
$$

**公式解释：** $Y$ 是图像级多标签矩阵，shape 为 `[B,C]`；$Y[b,c]=1$ 表示第 $b$ 张图存在类别 $c$，0 表示不存在。该式只声明离散监督接口，没有消去维度；它按类别维决定哪些 CLS token 属于目标类、哪些可参与随机掩码。

类别特定输出掩码只在 $C$ 维选择哪些CLS被置零，输出仍是 `[B,C,D]`；它发生在全部Transformer块计算完成之后，因此不减少前面自注意力的序列长度或计算量。

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

**公式解释：** 对每个 token 索引 $n$，从保留概率为 $1-r$ 的 Bernoulli 分布独立采样标量 $m_n\in\{0,1\}$；$r$ 是遮挡率。该式没有矩阵乘法或维度归约，重复采样 $N$ 次才组成长度为 $N$ 的掩码；因此保留数量的期望是 $N(1-r)$，但单次实际数量会波动。

$r$ 是遮挡率。若 $N=100,r=0.3$，期望保留70个token，但每次实际数量会波动。固定数量掩码则每张图严格遮30个，训练统计更稳定。

为保持期望幅值，dropout式掩码会使用：

$$X'=\frac{m\odot X}{1-r}.$$

**公式解释：** 掩码 $m$ 先在特征维广播，与 $X$ 逐元素相乘，未发生求和，输出 shape 与 $X$ 相同；再除以标量保留率 $1-r$，把幸存 token 的幅值放大。对任一元素，$X'_{n,d}=m_nX_{n,d}/(1-r)$，从随机采样的期望看可保持训练前后的平均幅值接近不变。

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

**公式解释：** $i$ 遍历 $C$ 个类别 token，$Y$ 表示当前图像真实存在的类别集合。该分段规则逐类输出标量 $m(i)$：目标类 $i\in Y$ 必为 0，非目标类只有被本轮随机选中时才为 1，其余为 0；没有矩阵乘法或维度消去。全部类别组合后得到长度为 $C$ 的输出掩码，其中 1 的语义是“把该非目标 CLS 输出置零”。

随后掩蔽最终输出：

$$z_{[CLS]_i}^{L}\leftarrow z_{[CLS]_i}^{L}(1-m(i)).$$

**公式解释：** $z_{[CLS]_i}^{L}\in\mathbb R^D$ 是最后一层第 $i$ 个类别 token，$m(i)$ 是标量，因此 $(1-m(i))$ 会广播到它的全部 $D$ 个分量。运算不求和、不改变 shape：$m(i)=1$ 时输出整个 $D$ 维零向量，$m(i)=0$ 时保持原值；这发生在编码结束后，所以没有缩短此前的序列。

因此，当前图像真正存在的类别CLS不会被该规则遮挡；被选中的非目标CLS输出变成零。论文还为每个注意力头学习门控，并用Hard Concrete分布近似 $L_0$ 正则来剪除冗余头。CLS输出掩码负责类别分配，注意力头稀疏化负责减少噪声，两者不是同一个操作。

## 4. 代表模型与论文

| 论文/模型 | 任务与起点 | 原方法存在的问题 | 具体操作 | 与本算子的关系 |
|---|---|---|---|---|
| [[vision_transformer_notes]] | 图像分类；单CLS + patch序列 | 全局分类token没有类别拆分，所有patch均参与稠密注意力 | CLS与全部patch共同经过多层自注意力，最终只取CLS分类 | 提供token、位置编码和注意力mask的基础接口 |
| [[MCTformer_paper_notes]] | WSSS；多标签图像分类 | 单CLS难为同图多个类别分别定位 | 为每类引入class token，从class-to-patch注意力得到类别图，并与patch分类CAM互补 | 奠定“每类一个CLS”的类别特定token路线 |
| [[TokenMasking_paper_notes]] | WSSS；多类别token ViT | 多个CLS没有硬职责约束，冗余注意力头使类别图含噪 | 训练时从图像不存在的类别CLS中随机置零部分最终输出；同时用Hard Concrete近似 $L_0$ 正则门控注意力头 | 类别输出掩码解决职责分配，头门控解决关系噪声，两者发生在不同层级 |
| [[mask2former_notes]] | 全监督通用分割；对象query | query若读取全图，计算大且注意分散 | 用上一解码层预测掩码把无关交叉注意力logit设为 $-\infty$ | 典型attention-edge masking，而非token置零或删除 |
| [[TokenMasking_paper_notes]] | 推理伪掩码 | 类别注意图仍是低分辨率连续响应 | 聚合最后若干层类别CLS到patch注意力，与patch CAM相乘或单独使用，再恢复网格和阈值化 | 展示token注意如何变成空间监督 |
| [[ExCEL_paper_notes]] | CLIP式WSSS；patch响应 | CLIP patch关系过平滑且含不可靠响应 | 静态视觉校准与动态可学习校准抑制噪声关系 | 更接近软权重/关系校准，不真正删除token |
| [[DiCLIP_paper_notes]] | WSSS；视觉缓存构建 | 全部patch写库会混入背景和低置信区域 | 用前景/背景掩码分别选择patch并做区域平均，再聚类成缓存键 | 可靠patch选择发生在建库阶段，不改变主干token计算 |

> [!note] 我的理解｜“mask”必须带上作用位置
> TokenMasking论文同时有“CLS最终输出置零”和“注意力头门控”；Mask2Former有“注意力边屏蔽”；MAE式方法是“输入patch缺失”；token pruning才会“缩短序列”。四者都可能被写作mask，但梯度、信息流和加速效果完全不同。

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

## 7. 官方仓库静态分析：TokenMasking-WSSS

### 7.1 仓库与固定版本

- 官方仓库：[HSG-AIML/TokenMasking-WSSS](https://github.com/HSG-AIML/TokenMasking-WSSS)
- 阅读commit：[`3daaec734700a4c9578dd8ce7bedef7f917aed66`](https://github.com/HSG-AIML/TokenMasking-WSSS/tree/3daaec734700a4c9578dd8ce7bedef7f917aed66)
- 多类别token与CAM：[model.py#L581-L739](https://github.com/HSG-AIML/TokenMasking-WSSS/blob/3daaec734700a4c9578dd8ce7bedef7f917aed66/model.py#L581-L739)
- 类别特定token dropout：[model.py#L744-L816](https://github.com/HSG-AIML/TokenMasking-WSSS/blob/3daaec734700a4c9578dd8ce7bedef7f917aed66/model.py#L744-L816)
- Concrete注意力头门控：[model.py#L16-L188](https://github.com/HSG-AIML/TokenMasking-WSSS/blob/3daaec734700a4c9578dd8ce7bedef7f917aed66/model.py#L16-L188)
- 注意力内应用门控：[model.py#L222-L290](https://github.com/HSG-AIML/TokenMasking-WSSS/blob/3daaec734700a4c9578dd8ce7bedef7f917aed66/model.py#L222-L290)
- 伪掩码生成：[generate_pseudomasks.py#L394-L439](https://github.com/HSG-AIML/TokenMasking-WSSS/blob/3daaec734700a4c9578dd8ce7bedef7f917aed66/generate_pseudomasks.py#L394-L439)

### 7.2 多类别token的真实序列与输出

[`forward_features`](https://github.com/HSG-AIML/TokenMasking-WSSS/blob/3daaec734700a4c9578dd8ce7bedef7f917aed66/model.py#L660-L687) 构造：

```python
cls_tokens = self.cls_token.expand(B, -1, -1)
register_token = self.register_token.expand(B, -1, -1)
x = torch.cat((cls_tokens, x, register_token), dim=1)
x = x + self.interpolate_pos_encoding(x, w, h)

for i, blk in enumerate(self.blocks):
    x, weights_i = blk(x)
    attn_weights.append(weights_i)

return (
    x[:, 0:self.num_classes],
    x[:, self.num_classes:-1],
    attn_weights,
)
```

若输入patch为 `[B,N,D]`，拼接后是 `[B,C+N+1,D]`。切片前 $C$ 个位置得到类别token `[B,C,D]`，中间得到patch token `[B,N,D]`，最后一个REG不进入这两个返回张量。

patch token随后恢复网格并经卷积分类头：

```text
[B,N,D]
→ reshape [B,H',W',D]
→ permute [B,D,H',W']
→ Conv2d head
→ feature_map [B,C,H',W']
```

同时，所有层注意力堆成 `[L,B,H,C+N+1,C+N+1]`，先沿注意力头 $H$ 求均值，再取最后若干层的“类别token行—patch列”：

$$
M_{att}=\sum_{l=L-r+1}^{L}\operatorname{MeanHead}(A_l)[:,0:C,C:C+N].
$$

**公式解释：** $A_l$ 是第 $l$ 层的多头注意力矩阵；`MeanHead` 先沿注意力头维求均值并消去头维，再用 `[:,0:C,C:C+N]` 选择 $C$ 个类别 token 作为 query、$N$ 个 patch token 作为 key，得到每层 `[B,C,N]`。外层沿最后 $r$ 个层索引 $l$ 求和并消去层维，输出 `M_att=[B,C,N]`；$M_{att}[b,c,n]$ 表示类别 $c$ 对 patch $n$ 的跨层累计注意力。

输出是 `[B,C,N]`，再reshape成 `[B,C,H',W']`。代码的 `fused` 模式做：

$$
M_{cam}=M_{att}\odot\operatorname{ReLU}(M_{patch}),
$$

**公式解释：** $M_{att}$ 与 $M_{patch}$ 都是 `[B,C,N]`，ReLU 先把 patch 分类响应中的负值截为 0，再与类别注意力逐元素相乘；没有矩阵乘法或求和，输出 shape 仍为 `[B,C,N]`。$M_{cam}[b,c,n]$ 只有在“类别 token 注意到 patch”且“patch 分类头也支持该类”时才保持较大。

即类别注意力和patch分类响应逐元素相乘，shape不变。

### 7.3 类别特定CLS掩码发生在哪里？

[`ViTWithTokenDropout.forward`](https://github.com/HSG-AIML/TokenMasking-WSSS/blob/3daaec734700a4c9578dd8ce7bedef7f917aed66/model.py#L794-L816) 的真实代码：

```python
features, mtatt, patch_attn, attn_weights = self.mct(x)
token_features = features

if labels is not None:
    for b in range(labels.shape[0]):
        for i in range(self.num_classes):
            if labels[b, i] == 0 and random.random() < self.rate:
                token_features[b, i, :] = torch.zeros_like(
                    features[b, i, :]
                )

x_cls_logits = token_features.view(token_features.shape[0], -1)
x_cls_logits = self.classifier_head_test(x_cls_logits)
```

逐步解释：

1. `self.mct(x)` 已经完成全部Transformer块，因此掩码不影响本次前向中的Q/K/V计算。
2. 只有 `labels[b,i]==0` 的非目标类别才有概率被置零；目标类别CLS始终保留。
3. `token_features = features` 没有clone，两者指向同一张量；后续原地写零会直接修改 `features` 的对应位置。
4. 所有类别token `[B,C,D]` 被展平成 `[B,CD]`，再由一个线性层整体映射到 `[B,C]`。因此代码并非“每个CLS只经过独立标量分类器”，线性头仍可跨类别token组合信息。
5. Python双循环和 `random.random()` 简单直观，但不是向量化实现；分布式复现时还需控制Python随机种子，而不只设置PyTorch种子。
6. 置零发生在最终输出，序列长度不变，所以这一机制服务于类别职责学习，不提供前向加速。

### 7.4 注意力头门控怎样工作？

注意力先正常计算：

```python
attn = (q @ k.transpose(-2, -1)) * self.scale
attn = self.attend(attn)
weights = attn
if self.prune:
    attn = self.gate(attn)
x = attn @ v
```

门控shape是 `[1,H,1,1]`，所以每个注意力头共享一个标量门，广播到该头的全部query—key位置。训练时Concrete分布采样连续门，再用straight-through方式硬化到0/1；$L_0$惩罚鼓励关闭更多头。

值得注意的是，`weights = attn` 发生在 `self.gate(attn)` 之前，返回并用于构造 `mtatt` 的注意力权重是**门控前**的矩阵。门控会影响 `attn @ v` 和后续token表示，但不会在这一返回张量上直接把被剪头清零；阅读“剪枝后的注意图”时必须区分这两条路径。

### 7.5 伪掩码脚本真实使用什么？

[`generate_pseudomasks.py`](https://github.com/HSG-AIML/TokenMasking-WSSS/blob/3daaec734700a4c9578dd8ce7bedef7f917aed66/generate_pseudomasks.py#L394-L439) 在 `torch.no_grad()` 下调用模型，随后取：

```python
out, x, *_ = model(img)
attn_cls = x[0].cpu().numpy()
```

这里 `x` 是模型返回的 `mtatt`，`x[0]` 取batch中的第一张图，shape为 `[C,N]`。脚本再根据图像标签选择正类注意图、reshape到patch网格并用OpenCV上采样。代码还读入像素真值 `gt_mask` 以取得输出shape并保存评估，但注释明确伪掩码生成不应把真值内容作为类别监督；核对新数据集适配时要确保真值没有进入阈值或填充决策。

## 8. 调试指标

- 实际保留率与设定值是否一致；
- 被掩CLS中目标类别与非目标类别的比例；目标CLS按论文规则应始终保留；
- 不同mask率下CLS—类别分配、伪掩码质量与分类准确率的权衡；
- 训练时mask、推理时不mask是否造成性能落差；
- 置零后被遮token是否仍通过残差或位置编码泄露信息。

## 9. 四个操作层级必须区分

| 操作位置 | 例子 | 是否改变序列长度 | 是否天然省计算 |
|---|---|---:|---:|
| 输入像素/patch | MAE式随机遮挡、Cutout | 可选 | 仅删除token时可能 |
| token值 | 置零或替换mask token | 否 | 否 |
| attention logit | 禁止某些Q-K对 | 否 | 标准稠密实现中否 |
| 输出token/损失 | 遮CLS输出、忽略低置信token | 否 | 否 |

论文使用“masking”时应定位到具体层级。[[TokenMasking_paper_notes]] 的类别CLS最终输出掩码主要改变监督分配，并不是MAE式patch重建，也不是推理token pruning。

## 10. attention mask的数学形式

对允许矩阵 $G\in\{0,1\}^{N_q\times N_k}$：

$$
A=\operatorname{softmax}\left(\frac{QK^T}{\sqrt D}+B_G\right),\qquad
(B_G)_{ij}=\begin{cases}0,&G_{ij}=1\\-\infty,&G_{ij}=0.\end{cases}
$$

**公式解释：** 若 `Q=[B,H,N_q,D]`、`K=[B,H,N_k,D]`，则 $QK^T$ 在每个 batch 和注意力头内做矩阵乘法，特征维 $D$ 被消去，得到 `[B,H,N_q,N_k]` 的 query—key logit；除以 $\sqrt D$ 控制尺度。`B_G=[N_q,N_k]` 按 batch 和头广播：允许边加 0，禁止边加 $-\infty$。softmax 只沿 key 维 $N_k$ 归一化并在分母求和，输出 $A$ shape 不变；$A[b,h,i,j]$ 是 query $i$ 分给 key $j$ 的注意力概率，禁止位置严格为 0。

## 11. 三类分割中的不同用途

- **经典分割**：token pruning/稀疏attention主要追求高分辨率效率，选择标准应尽量不损失小物体与边界。
- **弱监督分割**：类别token、可靠patch和随机遮挡用于发现更完整证据或减少噪声，但图像级标签可参与选择。
- **开放词汇分割**：视觉token选择不能依赖固定训练类，否则会在测试时漏掉未见类；更适合类别无关objectness、区域提议或对所有候选文本动态打分。

## 12. 选择偏差与真实加速

Top-k选择形成不可逆瓶颈：被删的小目标、背景中的未见物体无法被后续文本分类恢复。训练早期可用软门控或较高保留率，推理再根据验证消融剪枝。若声称效率提升，应报告端到端延迟和峰值显存；稀疏率本身不等于硬件加速，尤其当实现仍构造完整 $N\times N$ attention时。

额外应检查每类/每尺度token保留率、小物体召回，以及保留token是否集中在已有CAM高响应区域。后者过强说明mask只强化已有判别区域，没有发现新证据。

## 13. 选型、论文与源码索引

| 当前目标 | 优先操作 | 不应混淆为 |
|---|---|---|
| 让每个类别token职责更清晰 | 非目标CLS最终输出掩码 | patch遮挡或token pruning |
| 减少噪声注意力头 | Concrete门控 + 稀疏正则 | 删除空间token |
| 阻止query读取某些位置 | softmax前attention mask | softmax后简单乘零 |
| 真正减少高分辨率计算 | 物理删除token + 稀疏/变长算子 | 值置零 |
| WSSS只保留可靠patch | CAM/不确定性筛选，保留ignore | 把未选位置全部当背景 |
| OVS避免漏掉未见类 | 类别无关objectness/区域提议 | 固定训练类Top-k选择 |

导航：

- [[TokenMasking_paper_notes]]：类别CLS掩码、Hard Concrete头剪枝和伪掩码生成。
- [[MCTformer_paper_notes]]：多类别token与patch CAM的基础结构。
- [[vision_transformer_notes]]：标准CLS、patch token与自注意力数据流。
- [[mask2former_notes]]：掩码交叉注意力与对象query。
- [多类别token布局与CAM](https://github.com/HSG-AIML/TokenMasking-WSSS/blob/3daaec734700a4c9578dd8ce7bedef7f917aed66/model.py#L660-L739)：`[CLS×C, patch×N, REG]` 到类别注意图。
- [类别特定输出置零](https://github.com/HSG-AIML/TokenMasking-WSSS/blob/3daaec734700a4c9578dd8ce7bedef7f917aed66/model.py#L794-L816)：标签条件、随机率与分类头。
- [注意力头Concrete门控](https://github.com/HSG-AIML/TokenMasking-WSSS/blob/3daaec734700a4c9578dd8ce7bedef7f917aed66/model.py#L250-L290)：门控位置与返回权重。
- [伪掩码读取类别注意](https://github.com/HSG-AIML/TokenMasking-WSSS/blob/3daaec734700a4c9578dd8ce7bedef7f917aed66/generate_pseudomasks.py#L394-L439)：标签筛选、网格恢复与上采样入口。

## 14. 当前整理结论

令牌掩码既可以是数据增强，也可以是信息路由。实现前先明确目标是“发现更多证据”“排除噪声”还是“真正减少计算”，三者需要的mask方式并不相同。
