---
type: operator-note
aliases:
  - CAM Generation
  - Class Activation Map Generation
  - 类激活图生成
tags:
  - research-operator
  - CAM
  - weakly-supervised
  - open-vocabulary
  - semantic-segmentation
status: in-progress
---

# CAM Generation（类激活图生成）

> [!abstract] 本页定位
> 类别激活图（Class Activation Map, CAM）把图像级类别判断重新投回空间位置，回答“模型凭图中哪里判断为类别 $c$”。本页从经典分类器CAM、梯度加权CAM、类别token注意力和patch—text相似度中提炼统一接口，并说明它们在弱监督语义分割（Weakly-Supervised Semantic Segmentation, WSSS）和开放词汇语义分割（Open-Vocabulary Segmentation, OVS）中的不同监督含义。单篇论文笔记保存完整方法，本页关注生成算子、shape、选型和源码入口。

> [!note] 我的理解｜CAM是“证据图”，不是“物体图”
> 分类器只需找到足以判断类别的区域，因此狗脸、车轮或鸟头已经可能完成分类。CAM没有被训练成覆盖完整实例，也没有天然背景通道；后续的传播、区域细化和伪标签构造不是可有可无的美化步骤，而是在补足分类监督没有提供的信息。

## 1. 输入与输出

卷积特征：

$$F\in\mathbb{R}^{B\times D\times H'\times W'},$$

**公式解释：** $B$ 是批量大小，$D$ 是视觉通道数，$H',W'$ 是低分辨率网格；$F[b,d,h,w]$ 是第 $b$ 张图位置 $(h,w)$ 的第 $d$ 个特征分量。这里只声明输入 shape，没有求和或维度消去。

分类器权重：

$$W\in\mathbb{R}^{C\times D}.$$

**公式解释：** $C$ 是类别数，$W[c,d]$ 是类别 $c$ 分类器对第 $d$ 个特征通道的权重。它与 $F$ 共享通道维 $D$，为后续通道加权做准备；此式仍只是参数接口。

普通CAM：

$$M_{b,c,h,w}=\operatorname{ReLU}\left(\sum_dW_{c,d}F_{b,d,h,w}\right),$$

**公式解释：** 固定 $b,c,h,w$，将 $W[c,:]$ 与 $F[b,:,h,w]$ 在通道维 $D$ 上逐项相乘求和，因此通道维 $d$ 被消去，得到一个标量；ReLU 再去掉负证据。输出 $M=[B,C,H',W']$，$M[b,c,h,w]$ 表示位置 $(h,w)$ 对类别 $c$ 的正向分类证据。

例如：

```text
F: [2,512,20,20]
W: [20,512]
把每个空间位置的512维特征与20个类别权重做点积
M: [2,20,20,20]
bilinear interpolate
M_high: [2,20,320,320]
```

这里被消去的是通道维 $D=512$；$M[b,c,h,w]$ 表示第 $b$ 张图低分辨率位置 $(h,w)$ 对类别 $c$ 的正证据。双线性插值只把 $20\times20$ 平滑放大到 $320\times320$，不会创造新的类别信息或真实边界。

ViT或patch—text路线通常先得到 `[B,N,C]`：

```text
[B,N,C]
→ permute(0,2,1)
→ [B,C,N]
→ reshape(B,C,H',W')
→ [B,C,H',W']
→ interpolate
→ [B,C,H,W]
```

其中 $N=H'W'$。`permute` 把类别维放到图像通道位置，`reshape` 按原patch顺序恢复二维网格；更完整的上采样差异见 [[downsampling_and_upsampling(下采样与上采样)]]。

## 2. 常见形式

| 形式 | 空间证据来源 | 是否需反向梯度 | 优点 | 局限 |
|---|---|---:|---|---|
| 线性分类器CAM | 分类权重 × 特征 | 否 | 快、直接 | 要求GAP+线性分类结构 |
| Grad-CAM | 类别分数对特征的梯度 | 是 | 可用于一般网络层 | 常只覆盖判别区域 |
| Softmax-GradCAM | 类别概率梯度 | 是 | 引入类间竞争 | 受候选类别集合影响 |
| class-token注意力 | 类别token到patch的注意力 | 否 | Transformer内生定位 | 注意力不一定等于贡献 |
| patch-text相似度CAM | patch与类别文本相似度 | 否/可选 | 开放词汇自然 | patch语义与边界可能粗 |
| 原型CAM | patch与视觉原型相似度 | 否/可选 | 能覆盖多种视觉外观 | 原型纯度决定上限 |

## 3. Grad-CAM大白话流程

设类别分数为 $y_c$，某层特征为 $F^d$：

$$\alpha_d^c=\frac{1}{H'W'}\sum_{h,w}\frac{\partial y_c}{\partial F^d_{h,w}},$$

**公式解释：** $y_c$ 是类别 $c$ 的标量分数，偏导表示位置 $(h,w)$ 的通道 $d$ 对该分数的敏感度。沿 $H',W'$ 两个空间维求和并除以位置数，空间维被消去，得到每个通道一个标量 $\alpha_d^c$；它表示通道 $d$ 对类别 $c$ 的平均梯度贡献。

$$M_c=\operatorname{ReLU}\left(\sum_d\alpha_d^cF^d\right).$$

**公式解释：** 每个通道特征图 $F^d=[H',W']$ 乘以标量 $\alpha_d^c$，再沿通道维 $d$ 求和并消去它，得到二维热图；ReLU 只保留正贡献。输出 $M_c[h,w]$ 表示该位置对提高类别 $c$ 分数的梯度加权证据。Grad-CAM 需要反向传播拿梯度，但不等于更新参数。

## 4. patch-text CAM

若 $P:[B,N,D]$，文本 $T:[C,D]$：

$$S=\hat P\hat T^T\in\mathbb{R}^{B\times N\times C}.$$

**公式解释：** $\hat P=[B,N,D]$ 是归一化 patch 特征，$\hat T^T=[D,C]$ 是归一化类别文本转置。矩阵乘法沿共同特征维 $D$ 点积并消去它，输出 `[B,N,C]`；$S[b,n,c]$ 是第 $b$ 张图第 $n$ 个 patch 与类别 $c$ 文本的余弦相似度。之后再 `permute → reshape → interpolate` 得到 `[B,C,H,W]`。

## 5. 代表论文逐篇对比

| 论文 | 任务与起点 | 原方法存在的问题 | CAM怎样具体产生 | 后续怎样使用 |
|---|---|---|---|---|
| [[MCTformer_paper_notes]] | WSSS；多类别token Transformer | 普通分类CAM只覆盖判别区域，单CLS无法为多类分别定位 | 为每类设置class token，读取类别token到patch的注意力；同时用patch特征经分类头生成patch CAM，再融合两种证据 | 类别token提供类别特异注意，patch CAM补充局部分类响应，之后生成伪掩码 |
| [[TokenMasking_paper_notes]] | WSSS；多个类别CLS token | 类别token可能职责混淆，冗余注意力头产生噪声 | 随机遮部分非目标CLS最终输出以促进类别分配，并用Hard Concrete门控稀疏注意力头；推理聚合类别CLS到patch的注意力 | 注意图直接阈值化、上采样并生成伪分割标签，不依赖传统GAP分类器CAM |
| [[CLIP-ES_paper_notes]] | WSSS；冻结CLIP ViT | 原始Grad-CAM存在目标—背景和目标—非目标类别混淆 | 将候选前景与类别相关背景文本一起送入CLIP，用softmax后的类别概率作为梯度目标，对最后Transformer块前的归一化层做Grad-CAM | 用类感知注意力亲和力（Class-aware Attention-based Affinity, CAA）补全，再经CRF形成伪掩码 |
| [[WeCLIP_paper_notes]] | 单阶段WSSS；冻结CLIP骨干 | 冻结CLIP产生的CAM训练中不更新，错误伪标签会固定存在 | 图像—文本分类分数经Grad-CAM产生初始 $M_{init}$；另一路解码CLIP多层特征 | 冻结CLIP CAM细化模块（Refinement Module for Frozen CLIP CAM, RFM）用可学习解码器亲和力筛选CLIP注意力，形成动态伪标签监督解码器 |
| [[POT_paper_notes]] | WSSS；分类器CAM与原型 | 单类别原型无法覆盖明显类内变化 | 先由分类器CAM选可靠类别种子，再把特征分为多个簇，使用各簇原型重新激活对应外观 | 相似度感知最优传输分配特征—原型关系，并以一致性损失改善表示 |
| [[S2C_paper_notes]] | WSSS；ResNet38分类CAM + SAM | 推理时用SAM后处理会放大错误种子，而且SAM掩码没有语义 | $1\times1$ 分类头输出CAM并做多尺度聚合；从每类CAM采局部峰值作为SAM点提示 | SAM掩码稳定性与掩码内CAM均值相乘产生类别伪标签，再反向训练CAM分类器 |
| [[DiCLIP_paper_notes]] | WSSS；冻结CLIP与稳定扩散 | 直接patch—text匹配缺少空间多样性和类内视觉外观 | 基础patch—text CAM、静态视觉缓存CAM和动态适配器CAM分别生成，再按固定系数增强 | 动态CAM经PAR变成伪掩码，监督读取CLIP多层特征的分割头 |
| [[ComCD_paper_notes]] | WSSS；CLIP与扩散模型双分支 | CLIP语义强但不完整，扩散响应空间完整却类别特异性弱 | 两个模型分别输出同一类别空间的CAM | 依据每个像素两分支类别熵动态分配信任，而不是整图固定选一支 |

> [!note] 我的理解｜这些“CAM”并非完全同一种数值
> 分类器CAM是分类权重对特征的线性投影；Grad-CAM权重来自梯度；token注意图表示信息读取；patch—text CAM是跨模态相似度；原型CAM是视觉近邻相似度。它们都能画成热图，但归一化、背景定义和高值含义不同，不能未经校准直接相加。

## 6. 工程实例：CLIP-ES官方代码中的Softmax-GradCAM

固定版本 [`3893f81`](https://github.com/linyq2117/CLIP-ES/tree/3893f817be359c5ee1dbf8111cad381a532c7acc)：

- [`reshape_transform`](https://github.com/linyq2117/CLIP-ES/blob/3893f817be359c5ee1dbf8111cad381a532c7acc/generate_cams_voc12.py#L28-L35) 去掉CLS token并把ViT序列恢复为二维特征。
- [`perform`](https://github.com/linyq2117/CLIP-ES/blob/3893f817be359c5ee1dbf8111cad381a532c7acc/generate_cams_voc12.py#L134-L158) 编码图像、逐类别调用GradCAM并插值到原图。
- [`GradCAM`初始化](https://github.com/linyq2117/CLIP-ES/blob/3893f817be359c5ee1dbf8111cad381a532c7acc/generate_cams_voc12.py#L235-L240) 把CLIP最后一个Transformer block的归一化层选为目标层。

### 6.1 主调用链

```text
输入图像
→ CLIP image encoder 得 image_features 与 attention
→ 当前图像标签对应的前景文本 + 类别相关背景文本
→ 逐前景类别构造 ClipOutputTarget
→ GradCAM 对目标层求梯度
→ grayscale_cam [H',W']
→ cv2.resize 到原图
→ CAA注意力细化（属于后续算子）
```

代码在 [`perform`](https://github.com/linyq2117/CLIP-ES/blob/3893f817be359c5ee1dbf8111cad381a532c7acc/generate_cams_voc12.py#L125-L158) 中先拼接当前图像标签对应的前景文本和背景文本：

```python
fg_features_temp = fg_text_features[label_id_list].to(device_id)
text_features_temp = torch.cat([fg_features_temp, bg_features_temp], dim=0)
input_tensor = [image_features, text_features_temp, h, w]

for idx, label in enumerate(label_list):
    targets = [ClipOutputTarget(label_list.index(label))]
    grayscale_cam, logits_per_image, attn_weight_last = cam(
        input_tensor=input_tensor,
        targets=targets,
        target_size=None,
    )
```

这不是一次前向同时生成全部类别CAM，而是对图像中每个已知前景类别逐次调用Grad-CAM。候选文本集合参与softmax，因此改变背景词或前景候选集合，会改变类别概率及其梯度。

### 6.2 ViT token怎样交给Grad-CAM？

[`reshape_transform`](https://github.com/linyq2117/CLIP-ES/blob/3893f817be359c5ee1dbf8111cad381a532c7acc/generate_cams_voc12.py#L28-L35) 的真实变化是：

```python
tensor = tensor.permute(1, 0, 2)
result = tensor[:, 1:, :].reshape(
    tensor.size(0), height, width, tensor.size(2)
)
result = result.transpose(2, 3).transpose(1, 2)
```

若目标层输出布局是 `[1+N,B,D]`：

```text
[1+N,B,D]
→ permute
→ [B,1+N,D]
→ 去掉CLS
→ [B,N,D]
→ reshape
→ [B,H',W',D]
→ 两次transpose
→ [B,D,H',W']
```

这一步没有产生CAM，只把Transformer序列改成Grad-CAM库期望的CNN式特征图。默认 `height=width=28` 是固定假设；若改变输入/patch网格而不传入匹配尺寸，`reshape` 会失败或空间对应错误。

### 6.3 目标层、梯度和空间恢复

初始化代码固定在 [`generate_cams_voc12.py#L235-L240`](https://github.com/linyq2117/CLIP-ES/blob/3893f817be359c5ee1dbf8111cad381a532c7acc/generate_cams_voc12.py#L235-L240)：

```python
target_layers = [model.visual.transformer.resblocks[-1].ln_1]
cam = GradCAM(
    model=model,
    target_layers=target_layers,
    reshape_transform=reshape_transform,
)
```

目标是最后一个Transformer块的第一层归一化 `ln_1`，不是最终分类logit，也不是最后注意力矩阵本身。Grad-CAM需要对目标类别分数求梯度，但这里的反向只为读取梯度权重，并不调用优化器更新CLIP。低分辨率CAM最后通过 `cv2.resize(grayscale_cam, (ori_width, ori_height))` 放大；OpenCV默认线性插值只改变显示/监督分辨率。

### 6.4 代码阅读中值得注意的点

- 前景文本只取图像级标签中存在的类，这利用了WSSS训练时已知的图像标签；不能把同一设置直接称为无标签开放词汇推理。
- 提示模板固定为 `a clean origami {}.`，所以CAM质量同时依赖提示和候选背景集合。
- `reshape_transform` 固定28×28，输入尺寸变化必须同步检查patch数。
- CAM逐类别生成，类别数增多时梯度调用成本近似线性增加。
- Grad-CAM输出与后续CAA注意力传播是两步：前者给类别种子，后者补空间覆盖，不应把全部提升归因于CAM生成。

## 7. 调试与选型

- CAM类别对但太小：处理覆盖问题，优先看 [[Spatial_Propagation]] 与多原型。
- CAM边界糊：看特征分辨率和区域细化，不要只调阈值。
- 多类混淆：加入类别竞争、文本/视觉原型或背景词。
- 输出全零：检查ReLU前是否全负、分类器权重与特征通道是否对应。
- 上采样前保存低分辨率CAM，区分“原响应粗”与“插值显示模糊”。

## 8. CAM与经典分割输出的根本差异

全监督分割头直接用像素真值学习 $C$ 类logit；CAM则由图像级分类目标间接产生空间证据。两者虽然最终都能写成 `[B,C,H,W]`，监督含义不同：

| 输出 | 训练目标 | 高响应意味着什么 | 能否直接当真值 |
|---|---|---|---|
| 分割logit | 每像素类别 | 该像素属于类别 $c$ | 经softmax后用于预测 |
| 分类CAM | 图像级多标签 | 此处支持整图类别判断 | 否，常只覆盖判别部位 |
| attention map | token交互 | 模型从该位置读取信息 | 否，不等价于类别贡献 |
| patch-text相似度 | 图文预训练/对齐 | 局部特征接近文本锚点 | 否，仍需背景和空间校正 |

[[fcn_notes]] 的像素交叉熵拥有完整空间监督；WSSS缺失的正是这一环，所以需要 [[Pseudo_Label_Refinement]] 把CAM变成带ignore区域的训练目标。开放词汇分割则可在运行时更换文本类别，但“开放类别”不自动改善边界。

## 9. 多标签分类与类间竞争

WSSS图像可能同时含多个类别，分类训练常对每类独立使用sigmoid/BCE；像素分割通常用softmax让类别在同一位置竞争。二者不可混用：独立CAM可能在同一像素同时很高，转换伪标签时才需要冲突消解。

对CLIP提示集合做softmax还要明确候选集合：只在图像已知标签中竞争会利用图像级监督；在完整词表中竞争更接近开放词汇推理。改变候选类别数会改变softmax概率和熵，因此跨数据集阈值不能直接复用。

## 10. 归一化与多尺度集成

每类min-max归一化：

$$
\tilde M_c=\frac{M_c-\min M_c}{\max M_c-\min M_c+\varepsilon}
$$

**公式解释：** $\min M_c$ 与 $\max M_c$ 沿类别 $c$ 的全部空间位置求极值，分别得到标量；它们广播回 `[H',W']`，把每个位置线性缩放到近似 `[0,1]`，shape 不变。$\tilde M_c[h,w]$ 是该位置在本类别空间最小—最大响应范围中的相对强度；$\varepsilon$ 防止所有位置相等时除零。它适合类内阈值，却不能比较不同类别的绝对置信度。

## 11. 如何评价初始CAM

在有验证像素标注时，至少报告：

- **seed precision**：高置信前景中有多少正确，决定伪监督纯度；
- **seed recall**：真实目标有多少被激活，决定后续传播需要补多少；
- **CAM mIoU**：阈值/背景策略后的整体质量；
- **boundary F-score**：区分覆盖完整但边界粗与语义错误；
- **分类性能**：确认CAM变化没有以破坏图像级识别为代价。

调阈值时应用验证集，不应根据测试集最高mIoU选择oracle阈值后当作可部署结果。

## 12. 选型与论文/源码索引

| 当前需求 | 优先形式 | 首先检查 |
|---|---|---|
| CNN已有GAP + 线性分类器 | 经典线性CAM | 分类权重和特征通道是否对应 |
| 任意网络层、只需类别解释 | Grad-CAM | 目标层分辨率与梯度目标 |
| 希望CLIP免训练产生WSSS种子 | Softmax-GradCAM | 候选文本集合、背景词和reshape |
| 需要运行时任意文本类别 | patch—text CAM | 图文空间是否共享、背景和词表规模 |
| 类内外观变化很大 | 多原型/检索CAM | 原型纯度、覆盖率和类别平衡 |
| 类别正确但边界/覆盖差 | 保留CAM生成，增加空间细化 | 不要反复更换分类头掩盖空间问题 |

导航：

- [[CLIP-ES_paper_notes]]：Softmax-GradCAM、背景文本和CAA。
- [[WeCLIP_paper_notes]]：冻结CAM与可学习解码器闭环。
- [[MCTformer_paper_notes]]、[[TokenMasking_paper_notes]]：类别token注意图路线。
- [[POT_paper_notes]]：多原型CAM与最优传输。
- [[S2C_paper_notes]]：CAM峰值提示SAM并反向改善分类器。
- [[DiCLIP_paper_notes]]、[[ComCD_paper_notes]]：多路CAM与跨模型融合。
- [CLIP-ES reshape](https://github.com/linyq2117/CLIP-ES/blob/3893f817be359c5ee1dbf8111cad381a532c7acc/generate_cams_voc12.py#L28-L35)：ViT序列恢复网格。
- [CLIP-ES逐类CAM](https://github.com/linyq2117/CLIP-ES/blob/3893f817be359c5ee1dbf8111cad381a532c7acc/generate_cams_voc12.py#L134-L158)：候选文本与梯度目标。
- [CLIP-ES目标层](https://github.com/linyq2117/CLIP-ES/blob/3893f817be359c5ee1dbf8111cad381a532c7acc/generate_cams_voc12.py#L235-L240)：Grad-CAM初始化。

## 13. 当前整理结论

CAM只提供类别相关的空间证据。评价它时要分开看类别正确性、目标覆盖率和边界质量；这三者通常由不同算子负责。
