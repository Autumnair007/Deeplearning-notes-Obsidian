---
type: map-of-content
aliases:
  - Research Operators
  - 研究算子库
tags:
  - research-operator
  - MOC
status: todo
---

# Research Operators（研究算子库）

> [!abstract] 这套笔记怎样使用
> 这里整理的不是“某篇论文得到了什么结果”，而是“模型对张量做了什么操作、为什么这样做、还能怎样替换”。证据主线来自 `3_Methods_and_Frameworks` 下的弱监督与开放词汇笔记，并用FCN、DeepLabV3+、UPerNet、SegFormer、MaskFormer/Mask2Former和SAM等经典/基础分割笔记建立对照；以论文、已有笔记和静态代码阅读为主，不把未复现实验写成已验证结论。

## 1. 从问题进入，而不是从论文进入

```text
定位不完整/细节丢失
  → 特征提取、多层融合、空间传播、注意力细化

类内变化大/单一类别向量不够
  → 原型构建、区域池化、检索与记忆

CAM噪声大/伪标签不可靠
  → CAM生成、置信度重加权、伪标签细化、背景处理

视觉与文本空间不匹配
  → 跨模态对齐、提示构建、对比正则

不想全量微调大模型
  → 高效适配、蒸馏、令牌选择与掩码
```

## 2. 算子索引

### 1_Feature_and_Representation（特征与表征）

| 页面 | 主要回答的问题 |
|---|---|
| [[Feature_Extraction]] | 从骨干的哪一层、以什么空间粒度取出特征？ |
| [[Pooling_and_Region_Aggregation]] | 怎样把多个像素或patch压成一个区域/图像向量？ |
| [[Prototype_Construction]] | 怎样从一组样本构造稳定且可复用的代表向量？ |
| [[Region_Grouping_and_Proposal]] | 怎样先获得类别无关区域，再做语义判断？ |

### 2_Fusion_and_Interaction（融合与交互）

| 页面 | 主要回答的问题 |
|---|---|
| [[Cross_Modal_Alignment]] | 视觉单元怎样和文本、类别或另一视觉空间建立对应？ |
| [[Multi_Level_Fusion]] | 多层、多尺度或多分支信息怎样合并？ |
| [[Attention_and_Affinity_Refinement]] | 怎样修正token之间“谁应该和谁交互”的关系？ |
| [[Spatial_Propagation]] | 怎样把可靠种子沿空间关系扩散到完整物体？ |
| [[Retrieval_and_Memory]] | 怎样用外部样本、视觉键值或参考集补充当前预测？ |

### 3_Supervision_and_Optimization（监督与优化）

| 页面 | 主要回答的问题 |
|---|---|
| [[CAM_Generation]] | 怎样从分类模型得到带类别的空间热图？ |
| [[Pseudo_Label_Refinement]] | 怎样把粗糙响应变成可训练分割器的标签？ |
| [[Confidence_Reweighting]] | 多个分支冲突时怎样按可靠性加权？ |
| [[Contrastive_Regularization]] | 怎样在特征空间拉近同类、推远异类？ |
| [[Token_Selection_and_Masking]] | 怎样主动遮挡或筛选token以减少局部依赖？ |
| [[Distillation]] | 怎样把大模型或辅助分支的知识转给目标模型？ |

### 4_Adaptation_and_Efficiency（适配与效率）

| 页面 | 主要回答的问题 |
|---|---|
| [[Efficient_Adaptation]] | 冻结大骨干时，只训练哪些轻量参数？ |
| [[Prompt_Construction]] | 类别名称怎样扩写、集成并变成更好的文本锚点？ |
| [[Background_and_Unknown_Handling]] | 怎样显式建模背景、未知类和拒识区域？ |

## 3. 阅读一页时固定检查什么

1. **问题位置**：错误发生在特征、交互、监督还是输出阶段？
2. **张量契约**：输入和输出的形状是什么，哪一维被聚合或扩展？
3. **信息代价**：算子保留了什么，又不可逆地丢掉了什么？
4. **训练要求**：无参数、静态统计、轻量训练还是全量训练？
5. **替代关系**：它和相邻算子是并列候选、前后流水线，还是可以组合？
6. **代码入口**：以后重读仓库时从哪个函数开始，而不是从根目录盲找？

## 4. 一条常见的弱监督/开放词汇流水线

$$
I
\xrightarrow{\text{特征提取}}F
\xrightarrow{\text{CAM或跨模态匹配}}M_0
\xrightarrow{\text{亲和力/区域传播}}M_1
\xrightarrow{\text{置信度与背景处理}}\tilde Y
\xrightarrow{\text{分割监督}}P.
$$

大白话来说：先把图像变成仍保留空间位置的特征；再生成“哪些位置像哪个类别”的初始响应；随后补全物体、压制噪声；把可靠部分转成伪标签；最后训练真正的分割输出。不是每篇论文都包含全部步骤，但把方法放回这条链，通常就能看清创新究竟改了哪里。

## 5. 三种分割范式的共同坐标

| 问题 | 经典全监督分割 | 图像级弱监督分割 | 开放词汇分割 |
|---|---|---|---|
| 类别从哪里来 | 固定像素标签 | 图像级多标签 | 运行时文本词表/图文预训练 |
| 空间监督从哪里来 | 人工像素或实例mask | CAM、亲和力、超像素、SAM、伪标签 | 类别无关mask、CLIP patch、DINO/SAM结构 |
| 典型预测单元 | 像素或mask query | CAM像素/patch，随后生成伪mask | patch、region或mask query |
| 核心瓶颈 | 多尺度语义与边界恢复 | 语义种子不完整且监督有噪声 | 文本语义强但密集定位/边界弱 |
| 背景含义 | 固定第 $C+1$ 类 | 低响应区但可能含漏检前景 | 给定词表外区域，未必等于unknown |
| 主要评估 | mIoU、边界、速度 | CAM/伪标签质量 + 最终mIoU | seen/unseen、跨数据集、词表鲁棒性 |

这张表的用途是避免“换了监督来源就忘了分割基本问题”。例如开放词汇方法仍要解决经典解码器的多尺度与边界问题；弱监督方法即使使用SAM，也仍需用图像标签或跨模态分数给区域命名。

## 6. 三条典型工作流

### 6.1 经典密集预测

$$
I\xrightarrow{\text{backbone}}\{F_l\}
\xrightarrow{\text{decoder/fusion}}Z
\xrightarrow{\text{pixel or mask loss}}Y.
$$

参考 [[fcn_notes]]、[[deeplabv3+_notes]]、[[upernet_notes]]、[[segformer_notes]]、[[maskformer_notes]] 与 [[mask2former_notes]]。这条线提供空间恢复、特征金字塔和mask分类的基础接口。

### 6.2 图像级弱监督分割

$$
(I,y_{img})\rightarrow M_{cam}
\rightarrow M_{refined}
\rightarrow \tilde Y_{pixel/region}
\rightarrow \text{segmentation model}.
$$

关键问题依次是CAM种子精度/召回、传播是否越界、背景和ignore如何设置、伪标签是否产生确认偏差。单阶段方法会把这些步骤放在同一个训练图中，但信息流仍可按此拆解。

### 6.3 开放词汇分割

$$
I\rightarrow\{P\text{ or }R\},\qquad
\mathcal C_{test}\rightarrow T,
$$

$$
(P/R,T)\xrightarrow{\text{alignment}}S
\xrightarrow{\text{spatial recovery + background}}Y.
$$

区域可以来自类别无关mask，语义可以来自CLIP文本，空间结构可由DINO/SAM补充。真正的开放性来自测试时能够重算 $T$，而不是文件名中出现“CLIP”。

## 7. 算子间接口：谁在前，谁在后

```text
Feature_Extraction
  ├─→ Multi_Level_Fusion ─→ segmentation head
  ├─→ CAM_Generation ─→ Attention/Affinity ─→ Spatial_Propagation
  │                                  └─→ Pseudo_Label_Refinement
  ├─→ Region_Grouping ─→ Pooling ─→ Cross_Modal_Alignment
  │                                  └─→ region score back-projection
  └─→ Prototype_Construction / Retrieval_and_Memory
                         └─→ Contrastive_Regularization or dense classification

Prompt_Construction ─→ Cross_Modal_Alignment
Background_and_Unknown_Handling ─→ CAM / pseudo label / inference
Confidence_Reweighting ─→ fusion / pseudo label / loss
Efficient_Adaptation & Distillation ─→ constrain how the whole chain is trained/deployed
```

同一个模块可以出现在多个位置。例如置信度既可融合两个推理分支，也可选择伪标签，还可加权训练loss；笔记中必须写明它作用在哪个张量、梯度是否通过。

## 8. 基础机制入口：不在算子页重复展开

| 基础问题 | 优先回看已有笔记 | 算子库只保留什么 |
|---|---|---|
| ViT的patch、CLS、自注意力与张量布局 | [[vision_transformer_notes]]、[[vision_transformer_code_notes]] | 与当前算子直接相关的输入输出变化 |
| CLIP图文编码、归一化与对比学习 | [[clip_notes]]、[[clip_paper_notes]]、[[clip_code_notes]] | 密集patch—text、提示与开放类别用法 |
| DINOv2自监督特征与patch级表征 | [[dinov2_notes]]、[[dinov2_paper_notes]] | 它怎样给分割提供空间关系 |
| 上采样、跳跃连接与多层特征 | [[fcn_notes]]、[[dpt_notes]]、[[upernet_notes]] | 算子怎样接入弱监督/开放词汇流程 |
| 像素分类与mask分类 | [[maskformer_notes]]、[[mask2former_notes]] | 区域提议怎样承载文本语义与伪监督 |
| 多尺度上下文与池化 | [[deeplabv3+_notes]]、[[upernet_notes]] | 当前论文为何选择某种聚合方式 |

若基础笔记已经完整解释通用机制，算子页不再重新写一遍教材；但为了让公式可独立读懂，仍保留最短的张量契约和一两句直观说明。

## 9. 当前范围说明

- 弱监督：[[CLIP-ES_paper_notes]]、[[ComCD_paper_notes]]、[[DiCLIP_paper_notes]]、[[ExCEL_paper_notes]]、[[MCTformer_paper_notes]]、[[POT_paper_notes]]、[[S2C_paper_notes]]、[[SSR_paper_notes]]、[[TokenMasking_paper_notes]]、[[UGRL_paper_notes]]、[[VDA_paper_notes]]、[[WeCLIP_paper_notes]]、[[WeCLIP+_paper_notes]]。
- 开放词汇：[[CorrCLIP_paper_notes]]、[[OpenSeg_paper_notes]]、[[ReME_paper_notes]]、[[Talk2DINO_paper_notes]]、[[Trident_paper_notes]]。
- 经典与基础分割对照：[[fcn_notes]]、[[deeplabv3+_notes]]、[[upernet_notes]]、[[segformer_notes]]、[[maskformer_notes]]、[[mask2former_notes]]、[[sam_notes]]。
- 本库优先整理可迁移的操作，不追求覆盖这些论文的每一个模块。

## 10. 证据强度与写作规则

每页内容按以下层级理解：

1. **论文/官方代码明确给出**：可以陈述具体模块、公式和实现，但需保留论文笔记或固定提交链接。
2. **从代码静态阅读得到**：写明张量流和调用位置，不宣称已复现实验数值。
3. **跨论文归纳**：用“可理解为”“共同点是”等措辞，避免把归纳说成作者原话。
4. **工程建议/诊断**：说明它是检查方法或可替代方案，不暗示原论文已经采用。

所有算子页至少应回答：输入输出张量、信息来源、是否训练、梯度路径、与相邻算子的边界、三种范式中的角色、失败模式和可验证指标。

## 11. 在线核对入口

论文特有结论优先以作者论文页和官方代码为准，而不是只依据二手总结。当前已交叉核对的入口包括：

- [CLIP-ES｜arXiv 2212.09506](https://arxiv.org/abs/2212.09506) 与 [官方代码](https://github.com/linyq2117/CLIP-ES/tree/3893f817be359c5ee1dbf8111cad381a532c7acc)。
- [POT｜CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Wang_POT_Prototypical_Optimal_Transport_for_Weakly_Supervised_Semantic_Segmentation_CVPR_2025_paper.html) 与 [官方代码](https://github.com/jianwang91/POT/tree/60fd4ce4934c07744d0afe7426fd8aae94860f56)。
- [S2C｜CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Kweon_From_SAM_to_CAMs_Exploring_Segment_Anything_Model_for_Weakly_CVPR_2024_paper.html) 与 [官方代码](https://github.com/sangrockEG/S2C/tree/102e14c690c8e3bce3d5ccd1ae7832145ce10b27)。
- [WeCLIP｜CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Zhang_Frozen_CLIP_A_Strong_Backbone_for_Weakly_Supervised_Semantic_Segmentation_CVPR_2024_paper.html)。
- [ExCEL｜arXiv 2503.20826](https://arxiv.org/abs/2503.20826)、[Token Masking｜arXiv 2507.06848](https://arxiv.org/abs/2507.06848)、[DiCLIP｜arXiv 2605.04593](https://arxiv.org/abs/2605.04593)。
- [OpenSeg｜arXiv 2112.12143](https://arxiv.org/abs/2112.12143)、[Talk2DINO｜arXiv 2411.19331](https://arxiv.org/abs/2411.19331)、[Trident｜arXiv 2411.09219](https://arxiv.org/abs/2411.09219)、[ReME｜arXiv 2506.21233](https://arxiv.org/abs/2506.21233)。

代码链接固定到提交哈希；论文若后续更新版本，算子页的结论仍需重新核查，所以所有页面统一标记为 `todo`。
