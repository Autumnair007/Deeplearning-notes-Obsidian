# 方法与框架 (Methods and Frameworks) 目录结构指南
本文档用于指导本目录下论文笔记与概念笔记的归档规则。为了解决深度学习领域“方法交叉”（如：基于原型的少样本弱监督分割）导致的归档困难，本目录采用 **“核心问题定义优先，技术手段为辅”** 的分类逻辑。
## 1. 核心文件夹分类标准
目前我们将所有方法划分为以下四大类，请按照优先级顺序判断归档位置：
### 1_Learning_Paradigms (学习范式)
**定义**：在**标准闭集 (Closed-set)** 类别下，致力于**降低标注成本**或**利用非标准数据**的方法。
**收录规则**：只要论文的核心贡献在于使用了非全监督的数据设置，且不涉及新类别的发现，均放于此。
**子文件夹示例**：
*   `Weakly_Supervised` (弱监督)：仅使用级标签 (image-level)、点标签 (point)、框标签 (box) 进行训练。
*   `Semi_Supervised` (半监督)：混合使用少量有标签数据和大量无标签数据。
*   `Self_Supervised` (自监督/无监督)：无标签预训练或聚类。
### 2_Generalization_and_Open_World (泛化与开放世界)
**定义**：致力于**打破固定类别限制**，解决**新类别 (Novel Class)**、**新领域 (New Domain)** 识别问题的方法。这是当前最前沿的分类。
**收录规则**：如果论文涉及“未见类别”、“少样本”、“零样本”、“开放词汇”，**优先级高于学习范式**，一律放这里。
**子文件夹示例**：
*   `Open_Vocabulary` (开放词汇)：利用 CLIP 等图文对齐模型识别任意文本描述的类别 (如 ITACLIP)。
*   `Few_Shot` (少样本)：利用少量 Support Set 图像分割 Query Set (如 FS-DINO-4D)。
*   `Domain_Adaptation` (域适应)：解决训练集与测试集分布不一致的问题。
### 3_Foundation_Model_Adaptation (大模型适配)
**定义**：核心贡献在于**如何微调或适配**通用的基础模型 (Foundation Models, 如 SAM, DINO, CLIP) 到下游任务，且不局限于特定的少样本或弱监督场景。
**收录规则**：如果论文纯粹讨论 Prompt Engineering (提示工程)、Adapter 设计或从大模型蒸馏知识，放这里。
**注意**：如果论文是用 DINO 解决 Few-Shot 问题，优先放入 `2_Generalization_and_Open_World/Few_Shot`，并在笔记 Tag 中标记 `#foundation-model`。
**子文件夹规划**：
*   `Prompting_and_Adapters` (提示与适配器)：视觉提示、LoRA 微调等。
*   `Distillation_from_VLM` (大模型蒸馏)：将 SAM/DINO 能力转移到小模型。
### 4_Core_Mechanisms_and_Modules (核心机制与组件)
**定义**：通用的**技术改进模块**或**特定子问题处理**，通常可以即插即用到上述任何范式中。
**收录规则**：如果论文不改变监督方式（范式），也不改变任务设定（开放世界），只是提出了一个新的 Loss、一个新的注意力模块、或者一种后处理方法，放这里。
**子文件夹示例**：
*   `Prototype-Based_Methods` (基于原型的方法)：单纯研究原型特征提取与更新机制。
*   `Mask_Refinement` (掩码优化)：边缘细化、后处理模块 (如 SAMRefiner)。
*   `Background_and_Unseen_Class_Handling` (背景处理)：专门解决背景偏移或长尾分布的模块。
*   `Attention_and_CAMs` (注意力与类激活图)：可解释性分析或注意力机制改进。
## 2. 归档决策流程 (Decision Logic)
当你拿到一篇新论文时，请按以下流程提问：
1.  **这篇论文是否试图识别训练集里没有的类别？**
    *   是 $\rightarrow$ `2_Generalization_and_Open_World` (Few-Shot / Open-Vocab)
2.  **如果类别是固定的，它的核心卖点是否是节省标注？**
    *   是 $\rightarrow$ `1_Learning_Paradigms` (Weakly / Semi)
3.  **它的核心贡献是否是关于如何写 Prompt 或设计 Adapter 来微调大模型？**
    *   是 $\rightarrow$ `3_Foundation_Model_Adaptation`
4.  **以上都不是，它是否只是改进了网络中的某一个模块（如边缘细化、原型更新、注意力计算）？**
    *   是 $\rightarrow$ `4_Core_Mechanisms_and_Modules`
## 3. 创建文件夹与命名规范
*   **文件夹命名**：使用英语，单词首字母大写，单词间用下划线 `_` 连接 (Snake_Case)。例如：`Context_Label_Learning`。
*   **文件命名**：保持当前风格，即 `Paper_Title_paper_notes.md` 或 `Method_Name_notes.md`。