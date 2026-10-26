# Deeplearning-notes-Obsidian

> 最后更新：2026-07-09

这是一个使用 [Obsidian](https://obsidian.md/) 维护的深度学习与计算机视觉研究笔记库。

仓库主要记录学习过程中的基础概念、论文阅读、代码实践和项目复现。当前内容以计算机视觉为主，尤其关注图像分割、弱监督语义分割，以及 `CLIP`、`SAM`、`DINOv2` 等视觉基础模型在分割任务中的应用。

## 🗂️ 内容结构与入口

知识、领域、实践、探索和资源按职责分层，组织规则放在元数据目录。静态入口可直接在 GitHub 或 Obsidian 中使用，Hub 中的动态列表用于进一步检索。

| 内容 | 入口与职责 |
|---|---|
| 元数据与规则 | [00_Meta](00_Meta%20(元数据)/) · [项目框架](00_Meta%20(元数据)/笔记项目框架.md) · [标签规范](00_Meta%20(元数据)/标签规范.md) |
| 知识层 | [知识库 Hub](10_Knowledge_Base%20(知识库)/_Knowledge_Base_Hub.md)：数学、机器学习概念、深度学习机制与模型 |
| 领域层 | [应用领域 Hub](20_Areas%20(应用领域)/_Areas_Hub.md) · [CV](20_Areas%20(应用领域)/CV%20(计算机视觉)/_CV_Hub.md) · [NLP](20_Areas%20(应用领域)/NLP%20(自然语言处理)/_NLP_Hub.md) |
| 实践层 | [项目实践 Hub](30_Projects%20(项目实践)/_Projects_Hub.md)：实现、框架教程、训练实验与部署 |
| 探索层 | [知识探索 Hub](40_Exploration%20(知识探索)/_Exploration_Hub.md)：新方向初识、概览与长期问题 |
| 资源层 | [资源目录](99_Assets%20(资源文件)/)：[论文](99_Assets%20(资源文件)/papers/) · [图片](99_Assets%20(资源文件)/images/) |

研究与写作的常用入口：

- [Research Operators（研究算子库）](10_Knowledge_Base%20(知识库)/5_Research_Operators%20(研究算子库)/_Research_Operators_Hub.md)
- [方法与框架归档规则](20_Areas%20(应用领域)/CV%20(计算机视觉)/3_Methods_and_Frameworks%20(方法与框架)/README.md)
- [弱监督语义分割专题](20_Areas%20(应用领域)/CV%20(计算机视觉)/4_CV_Topics%20(专题研究)/_WSSS_Hub.md) · [免训练分割专题](20_Areas%20(应用领域)/CV%20(计算机视觉)/4_CV_Topics%20(专题研究)/_TFS_Hub.md)
- [通用笔记模板](00_Meta%20(元数据)/note_template.md) · [论文翻译与精读模板](00_Meta%20(元数据)/paper_notes_template.md) · [探索概览模板](00_Meta%20(元数据)/overview_note_template.md)
- [论文工作流 PromptV4](00_Meta%20(元数据)/PromptV4.txt) · [元数据提示词 V2](00_Meta%20(元数据)/metadata_promptV2.txt)

## 🔎 当前关注方向

- 语义分割模型: `FCN`、`DeepLabV3+`、`SETR`、`SegFormer`、`MaskFormer`、`Mask2Former`、`UPerNet` 等。
- 弱监督语义分割: CAM 生成、伪标签构建、CLIP-based WSSS、原型学习、最优传输和掩码细化。
- 视觉基础模型: `CLIP`、`DINOv2`、`SAM` 及其在 dense prediction、open-vocabulary segmentation 中的使用。
- 工程实践: `PyTorch`、`MMSegmentation`、`MMPretrain`、`TorchServe` 等框架和工具链。

## 📌 个人相关工作

- `ModuSeg`: ECCV 2026 相关工作，面向 training-free weakly supervised semantic segmentation。该方法将 object discovery 与 semantic retrieval 解耦，结合通用 mask proposer、视觉语义基础模型和非参数特征检索，减少传统 WSSS 中伪标签噪声与多阶段训练带来的复杂度。相关代码见 [Autumnair007/ModuSeg](https://github.com/Autumnair007/ModuSeg)，论文见 [arXiv:2604.07021](https://arxiv.org/abs/2604.07021)。

## 🧭 笔记组织方式

笔记通过内部链接、标签和 Dataview 建立交叉索引。核心知识笔记以开头 YAML 的 `type`、`tags`、`status` 描述用途、主题和整理进度；新论文的发表来源使用 `venue` 字符串，已有属性保留。标签平铺、英文小写，按内容选择，`cv` / `nlp` 非必填。

常见类型包括 `concept-note`、`paper-note`、`operator-note`、`concept-comparison`、`hub-note` 和 `map-of-content`。探索笔记可选 `overview-note`，也可以不加元数据；译文、附录翻译、随手记、资料占位、教程和代码笔记不强制补卡片。`status` 只反映整理进度，不按正文篇幅判断。

探索层可长期保存初识和问题，不要求成熟后迁出。完整组织约定见 [项目框架](00_Meta%20(元数据)/笔记项目框架.md)，具体标签见 [标签规范](00_Meta%20(元数据)/标签规范.md)。

这个仓库更偏向个人研究过程记录，而不是完整教程或论文复现集合。部分笔记会比较详细地拆解论文方法、张量流和实现细节，也会保留一些尚未整理完成的阅读痕迹。

## 📝 说明

内容会随着后续阅读和实验持续更新。目前的主线是围绕弱监督语义分割梳理近年的方法演进，并逐步把相关论文从“读懂”整理到“可比较、可复现、可用于选题判断”的状态。
