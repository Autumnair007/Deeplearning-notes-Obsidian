# Deeplearning-notes-Obsidian

> 最后更新：2026-07-09

这是一个使用 [Obsidian](https://obsidian.md/) 维护的深度学习与计算机视觉研究笔记库。

仓库主要记录学习过程中的基础概念、论文阅读、代码实践和项目复现。当前内容以计算机视觉为主，尤其关注图像分割、弱监督语义分割，以及 `CLIP`、`SAM`、`DINOv2` 等视觉基础模型在分割任务中的应用。

## 🗂️ 内容结构

这个仓库整体采用类似 PARA 的组织方式，把基础知识、研究方向、工程实践和资源文件分开管理。

### `00_Meta (元数据)`

用于维护知识库本身，包括：

- 论文笔记模板、通用笔记模板和写作提示词。
- 目录说明、归档规则和用于生成目录结构的脚本。
- 面向 Obsidian 工作流的辅助配置。

### `10_Knowledge_Base (知识库)`

沉淀相对通用、可复用的基础概念，主要包括：

- 数学基础：范数、余弦相似度、极大似然估计、凸函数等。
- 机器学习概念：监督学习、聚类方法等。
- 深度学习核心模块：注意力机制、归一化、激活函数、正则化、上下采样等。
- 经典模型机制：Transformer、RNN、Seq2Seq、U-Net、GAN 等。

### `20_Areas (应用领域)`

按研究方向组织的主体内容，目前主要集中在 CV：

- CV 理论：图像处理、分割指标、类别不平衡、Transformer 在 CV 中的基本概念。
- 分割模型：`FCN`、`DeepLabV3+`、`SETR`、`SegFormer`、`MaskFormer`、`Mask2Former`、`UPerNet` 等。
- 学习范式：弱监督分割、半监督分割、自监督/多模态方法。
- 开放词汇与泛化：CLIP-based segmentation、open-vocabulary segmentation、foundation model adaptation。
- NLP 相关内容目前较少，主要作为基础补充。

### `30_Projects (项目实践)`

记录代码实践和实验过程，包括：

- `PyTorch` 基础练习和经典模型复现。
- `MMSegmentation`、`MMPretrain` 等框架教程。
- SegFormer、UPerNet 等模型训练与实验记录。
- `TorchServe` 部署实践，包括 MNIST、ResNet18 等示例。

### `99_Assets (资源文件)`

集中存放笔记中引用的材料：

- 论文 PDF、综述文献和补充材料。
- 笔记插图、论文截图、实验可视化结果。
- 一些模型运行结果和辅助文件。

## 🔎 当前关注方向

- 语义分割模型: `FCN`、`DeepLabV3+`、`SETR`、`SegFormer`、`MaskFormer`、`Mask2Former`、`UPerNet` 等。
- 弱监督语义分割: CAM 生成、伪标签构建、CLIP-based WSSS、原型学习、最优传输和掩码细化。
- 视觉基础模型: `CLIP`、`DINOv2`、`SAM` 及其在 dense prediction、open-vocabulary segmentation 中的使用。
- 工程实践: `PyTorch`、`MMSegmentation`、`MMPretrain`、`TorchServe` 等框架和工具链。

## 📌 个人相关工作

- `ModuSeg`: ECCV 2026 相关工作，面向 training-free weakly supervised semantic segmentation。该方法将 object discovery 与 semantic retrieval 解耦，结合通用 mask proposer、视觉语义基础模型和非参数特征检索，减少传统 WSSS 中伪标签噪声与多阶段训练带来的复杂度。相关代码见 [Autumnair007/ModuSeg](https://github.com/Autumnair007/ModuSeg)，论文见 [arXiv:2604.07021](https://arxiv.org/abs/2604.07021)。

## 🧭 笔记组织方式

笔记整体结合 Obsidian 的双链、标签和 Dataview 进行索引。多数论文和模型笔记会通过 YAML 元数据标记主题、状态、年份和模型名称，便于在 `_Hub.md` 页面中自动聚合。

常见笔记类型包括：

- `paper-note`: 论文阅读笔记，通常包含方法动机、核心模块、实验结果和个人理解。
- `code-note`: 代码阅读或复现笔记，更关注运行流程、张量变化和实现细节。
- `hub-note`: 索引页，用于按主题自动聚合相关笔记。
- `concept-note`: 概念解释，服务于后续论文阅读和方法对比。

这个仓库更偏向个人研究过程记录，而不是完整教程或论文复现集合。部分笔记会比较详细地拆解论文方法、张量流和实现细节，也会保留一些尚未整理完成的阅读痕迹。

## 📝 说明

内容会随着后续阅读和实验持续更新。目前的主线是围绕弱监督语义分割梳理近年的方法演进，并逐步把相关论文从“读懂”整理到“可比较、可复现、可用于选题判断”的状态。
