# 方法与框架目录指南

按论文主要解决的任务和问题归档，技术手段作为辅助标签。一篇论文只放一个主位置，其他主题通过双链和 Hub 连接。

## 现有目录职责

| 目录 | 归档条件与现有子目录 |
|---|---|
| `1_Learning_Paradigms` | 核心问题是标注来源、标注成本或学习范式。现有 `Weakly_Supervised`、`Semi_Supervised_Segmentation(半监督分割)`。 |
| `2_Generalization_and_Open_World` | 核心任务是未见类别、开放词汇或泛化；现有 `Open_Vocabulary`。不要仅因用了 CLIP 就归入开放词汇。 |
| `3_Foundation_Model` | 基础模型本体、能力分析、增强或应用；四个子类见下表。 |
| `4_Core_Mechanisms_and_Modules` | 核心贡献是可复用机制或子问题处理；现有 `Background_and_Unseen_Class_Handling (背景与未知类别处理)`、`Entity_Segmentation_Frameworks`、`Mask_Refinement (掩码优化)`、`Prototype-Based Methods(基于原型的方法)`。 |

### `3_Foundation_Model` 的四个子类

| 子目录 | 内容与判断 |
|---|---|
| `Models (基础模型)` | 模型本体、结构和通用能力，如 CLIP、DINOv2、Depth Anything、SenseNova-Vision。 |
| `Foundation_Model_Analysis` | 对已有基础模型表征、空间结构、行为与局限的分析，如 Geosubprobe。 |
| `Foundation_Model_Enhancement` | 以改善基础模型能力或推理行为为核心的增强方法，如 SteerViT；适配、提示等按实际问题判断，可参考本目录已有 README。 |
| `Applications (应用)` | 将基础模型用于具体应用，且该应用没有更优先的现有任务归档位置，如 ColPali。 |

## 任务优先的决策顺序

1. 先问论文明确解决什么任务。若主线是开放词汇/未见类别识别，优先进入泛化与开放世界；使用何种监督和基础模型用标签补充。
2. 若核心问题是固定任务下的弱监督、半监督等标注设定，优先进入学习范式。基于 SAM、DINO 或 CLIP 的 WSSS 论文仍按 WSSS 主线归档。
3. 若主题是基础模型本体、能力分析、通用增强或应用，再按 `3_Foundation_Model` 的四个实际子类选择；不能把所有使用基础模型的论文都放在这里。
4. 若主要讨论通用模块、掩码细化、背景处理、实体分割框架或原型机制，按 `4_Core_Mechanisms_and_Modules` 的实际子目录归档。某个模块服务于明确的 WSSS/开放词汇任务时，仍先看论文主问题。

任务需要目录中尚无的分类时，先复用最贴近的现有位置，并用标签、双链说明交叉关系；只有实际内容形成持续主题时才增加目录，不创建规划中的空目录。

## 文件与关联

沿用所在目录的 `{方法名}_paper_notes.md`、`{方法名}_Translation.md`、`{方法名}_code_notes.md` 命名习惯，既有文件保留名称与位置。通用原理链接到知识库，复用操作链接到研究算子库，实验和框架教程链接到项目实践；不复制多份正文。

组织依据为 [笔记项目框架](../../../00_Meta%20(元数据)/笔记项目框架.md) 与 [标签规范](../../../00_Meta%20(元数据)/标签规范.md)。元数据按内容用途确定，译文和代码笔记不强制补卡片；领域标签不强制添加。
