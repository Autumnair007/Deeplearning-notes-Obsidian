---
type: operator-note
aliases: [CAM Generation, 类激活图生成]
tags: [research-operator, CAM, weakly-supervised]
status: todo
---

# CAM Generation（类激活图生成）

> [!abstract] 核心直觉
> Class Activation Map（类激活图，CAM）把图像级类别判断重新投回空间位置，得到“模型凭图中哪里判断为类别 $c$”。它是弱监督分割的语义种子，不等于完整分割掩码。

## 1. 输入与输出

卷积特征：

$$F\in\mathbb{R}^{B\times D\times H'\times W'},$$

分类器权重：

$$W\in\mathbb{R}^{C\times D}.$$

普通CAM：

$$M_{b,c,h,w}=\operatorname{ReLU}\left(\sum_dW_{c,d}F_{b,d,h,w}\right),$$

输出 $M\in\mathbb{R}^{B\times C\times H'\times W'}$。对每个位置，用类别 $c$ 的分类器权重给各通道加权；ReLU只保留正证据。

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

$$M_c=\operatorname{ReLU}\left(\sum_d\alpha_d^cF^d\right).$$

第一式先问“第 $d$ 个通道稍微变化，会多大程度影响类别 $c$ 的分数”，再把整张图上的梯度平均成通道权重。第二式用这些权重组合特征图。因此Grad-CAM需要一次反向传播来拿梯度，但不等于更新模型参数。

## 4. patch-text CAM

若 $P:[B,N,D]$，文本 $T:[C,D]$：

$$S=\hat P\hat T^T\in\mathbb{R}^{B\times N\times C}.$$

之后 `permute → reshape → interpolate` 得到 `[B,C,H,W]`。完整张量解释见 [[Cross_Modal_Alignment]]。这种方式直接，但容易把所有与类别文字相似的区域都激活，未必区分目标实例与上下文。

## 5. 论文对比

| 论文 | 初始CAM | 后续处理 |
|---|---|---|
| [[CLIP-ES_paper_notes]] | CLIP Softmax-GradCAM | CLIP注意力亲和力细化 |
| [[WeCLIP_paper_notes]] | 冻结CLIP CAM | RFM与可训练解码器形成闭环 |
| [[MCTformer_paper_notes]] | class-token注意力 + patch-token CAM | 利用两者互补生成伪掩码 |
| [[POT_paper_notes]] | 分类器CAM确定类别种子 | 多原型与最优传输补全 |
| [[DiCLIP_paper_notes]] | patch-text、静态检索、动态检索三路CAM | 固定融合与分割监督 |
| [[ComCD_paper_notes]] | CLIP与扩散模型各自产生CAM | 按像素熵动态选择 |

## 6. 工程实例：CLIP-ES

固定版本 [`3893f81`](https://github.com/linyq2117/CLIP-ES/tree/3893f817be359c5ee1dbf8111cad381a532c7acc)：

- [`reshape_transform`](https://github.com/linyq2117/CLIP-ES/blob/3893f817be359c5ee1dbf8111cad381a532c7acc/generate_cams_voc12.py#L28-L35) 去掉CLS token并把ViT序列恢复为二维特征。
- [`perform`](https://github.com/linyq2117/CLIP-ES/blob/3893f817be359c5ee1dbf8111cad381a532c7acc/generate_cams_voc12.py#L134-L158) 编码图像、逐类别调用GradCAM并插值到原图。
- [`GradCAM`初始化](https://github.com/linyq2117/CLIP-ES/blob/3893f817be359c5ee1dbf8111cad381a532c7acc/generate_cams_voc12.py#L235-L240) 把CLIP最后一个Transformer block的归一化层选为目标层。

这一调用链表明，使用ViT做Grad-CAM时必须提供 `reshape_transform`；CNN特征本来就是 `[B,D,H',W']`，ViT token则需要显式丢CLS和恢复网格。

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

适合可视化和类内阈值，却会让每个类别至少有一个位置达到1，即使该类不在图中。因此必须先用图像标签或开放词汇存在性分数过滤类别。多尺度/翻转CAM应先逆变换回原图坐标，再平均或取最大；对各尺度分别min-max再融合会抹掉尺度间置信差异。

## 11. 如何评价初始CAM

在有验证像素标注时，至少报告：

- **seed precision**：高置信前景中有多少正确，决定伪监督纯度；
- **seed recall**：真实目标有多少被激活，决定后续传播需要补多少；
- **CAM mIoU**：阈值/背景策略后的整体质量；
- **boundary F-score**：区分覆盖完整但边界粗与语义错误；
- **分类性能**：确认CAM变化没有以破坏图像级识别为代价。

调阈值时应用验证集，不应根据测试集最高mIoU选择oracle阈值后当作可部署结果。

## 12. 当前整理结论

CAM只提供类别相关的空间证据。评价它时要分开看类别正确性、目标覆盖率和边界质量；这三者通常由不同算子负责。
