---
type: "hub-note"
tags: [hub, knowledge-base]
status: "done"
---
# 知识层 (10_Knowledge_Base)

> **核心原则**: 此处存放的是“放之四海而皆准”的知识原子。它们是构成你所有上层建筑（应用领域、项目实践）的最基础、最纯粹的砖块。

---

## ⚡ 动态索引 (Dynamic Indexes)

利用 Dataview 插件，这里会自动追踪你的学习进度。

### 📝 待办知识点 (To-Do Notes)
这里列出了所有你标记为 `status: "todo"` 的笔记，提醒你接下来要攻克的知识点。

```dataview
LIST
FROM "10_Knowledge_Base (知识库)"
WHERE status = "todo"
SORT file.name ASC
```

### ✨ 最近更新 (Recently Modified)
你最近在 `10_Knowledge_Base` 目录中编辑过的 10 篇笔记。

```dataview
TABLE WITHOUT ID
	file.link AS "笔记名称",
	file.mtime AS "修改日期"
FROM "10_Knowledge_Base (知识库)"
WHERE file.name != "_Knowledge_Base_Hub"
SORT file.mtime DESC
LIMIT 10
```

---

## 📚 知识库手动索引 (Manual Index)

这是根据你的文件夹结构创建的完整索引。你可以随时点击链接跳转。

### 1. 数学基础 (1_Math_Foundations)
- [[10_Knowledge_Base/1_Math_Foundations/convex_and_concave_functions(凸函数和凹函数)|convex_and_concave_functions(凸函数和凹函数)]]
- [[10_Knowledge_Base/1_Math_Foundations/maximum_likelihood_estimation(极大似然估计)|maximum_likelihood_estimation(极大似然估计)]]
- [[10_Knowledge_Base/1_Math_Foundations/norms_and_cosine_similarity(范数和余弦相似度)|norms_and_cosine_similarity(范数和余弦相似度)]]
- [[10_Knowledge_Base/1_Math_Foundations/vector_space_and_orthogonality(空间和正交性)|vector_space_and_orthogonality(空间和正交性)]]

### 2. 机器学习核心 (2_ML_Core_Concepts)
- [[10_Knowledge_Base/2_ML_Core_Concepts/supervised_learning_concepts(监督学习概念)|supervised_learning_concepts(监督学习概念)]]
- **聚类 (Clustering)**
    - [[10_Knowledge_Base/2_ML_Core_Concepts/Clustering/Birch(平衡迭代规约和聚类)/birch_notes|Birch(平衡迭代规约和聚类)]]
    - [[10_Knowledge_Base/2_ML_Core_Concepts/Clustering/Optics(OPTICS算法)/optics_notes|Optics(OPTICS算法)]]

### 3. 深度学习核心 (3_DL_Core_Concepts)
- [[10_Knowledge_Base/3_DL_Core_Concepts/activation_functions(激活函数)|activation_functions(激活函数)]]
- [[10_Knowledge_Base/3_DL_Core_Concepts/attention_mechanism(注意力机制)|attention_mechanism(注意力机制)]]
- [[10_Knowledge_Base/3_DL_Core_Concepts/cross_entropy_loss_for_language_models[交叉熵损失(语言模型)]|cross_entropy_loss_for_language_models[交叉熵损失(语言模型)]]]
- [[10_Knowledge_Base/3_DL_Core_Concepts/data_flow_of_attention_mechanism_in_nlp[注意力机制的数据流过程(NLP)]|data_flow_of_attention_mechanism_in_nlp[注意力机制的数据流过程(NLP)]]]
- [[10_Knowledge_Base/3_DL_Core_Concepts/downsampling_and_upsampling(下采样与上采样)|downsampling_and_upsampling(下采样与上采样)]]
- [[10_Knowledge_Base/3_DL_Core_Concepts/multi_head_attention_self_attention_and_positional_encoding_notes(多头注意力、自注意力与位置编码笔记)|multi_head_attention_self_attention_and_positional_encoding_notes(多头注意力、自注意力与位置编码笔记)]]
- [[10_Knowledge_Base/3_DL_Core_Concepts/n-gram|n-gram]]
- [[10_Knowledge_Base/3_DL_Core_Concepts/normalization(归一化)|normalization(归一化)]]
- [[10_Knowledge_Base/3_DL_Core_Concepts/regularization(正则化)|regularization(正则化)]]
- [[10_Knowledge_Base/3_DL_Core_Concepts/深度学习思考|深度学习思考]]

### 4. 模型与机制 (4_DL_Models_And_Mechanisms)
- **生成模型 (Generative Models)**
    - [[10_Knowledge_Base/4_DL_Models_And_Mechanisms/Generative_Adversarial_Network(生成对抗网络)/gan_notes|Generative Adversarial Network (GAN)]]
- **序列模型 (Sequence Models)**
    - [[10_Knowledge_Base/4_DL_Models_And_Mechanisms/Recurrent_Neural_Network (循环神经网络)/rnn_notes|Recurrent Neural Network (RNN)]]
    - [[10_Knowledge_Base/4_DL_Models_And_Mechanisms/Seq2Seq(序列到序列模型)/seq2seq_notes|Seq2Seq (序列到序列模型)]]
- **核心机制 (Core Mechanisms)**
    - [[10_Knowledge_Base/4_DL_Models_And_Mechanisms/Transformer/transformer_notes|Transformer]]
        - [[10_Knowledge_Base/4_DL_Models_And_Mechanisms/Transformer/transformer_code_notes|Transformer (代码笔记)]]
    - [[10_Knowledge_Base/4_DL_Models_And_Mechanisms/U-Net/u-net_notes|U-Net]]
- **词表示 (Word Representation)**
    - [[10_Knowledge_Base/4_DL_Models_And_Mechanisms/Word2Vec(词向量)/word2vec_notes|Word2Vec]]

---
## 🗺️ 知识原子全景图 (All Knowledge Atoms Overview)

下面是 `10_Knowledge_Base` 文件夹下所有知识原子的完整列表，以及它们的元数据。

```dataview
TABLE
    type AS "类型",
    tags AS "标签",
    status AS "状态"
FROM "10_Knowledge_Base (知识库)"
WHERE file.name != "_Knowledge_Base_Hub"
SORT file.folder ASC, file.name ASC
```