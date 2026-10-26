---
type: "hub-note"
tags: [hub, nlp, areas]
status: "done"

---
# 自然语言处理领域 (NLP)

> **核心原则**: 此处汇集了所有与自然语言处理（NLP）相关的理论、模型和实践笔记。它作为该领域的知识中心，连接基础概念与具体应用，如文本预处理、语言模型和机器翻译等。

## 常用入口

- [应用领域总览](../_Areas_Hub.md)
- [注意力机制](../../10_Knowledge_Base%20%28%E7%9F%A5%E8%AF%86%E5%BA%93%29/3_DL_Core_Concepts%20%28%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E6%A0%B8%E5%BF%83%E6%A6%82%E5%BF%B5%29/attention_mechanism%28%E6%B3%A8%E6%84%8F%E5%8A%9B%E6%9C%BA%E5%88%B6%29.md)
- [Transformer](../../10_Knowledge_Base%20%28%E7%9F%A5%E8%AF%86%E5%BA%93%29/4_DL_Models_And_Mechanisms%20%28%E6%A8%A1%E5%9E%8B%E4%B8%8E%E6%9C%BA%E5%88%B6%29/Transformer/transformer_notes.md)
- [Word2Vec](../../10_Knowledge_Base%20%28%E7%9F%A5%E8%AF%86%E5%BA%93%29/4_DL_Models_And_Mechanisms%20%28%E6%A8%A1%E5%9E%8B%E4%B8%8E%E6%9C%BA%E5%88%B6%29/Word2Vec%28%E8%AF%8D%E5%90%91%E9%87%8F%29/word2vec_notes.md)

---

## ⚡ 动态索引 (Dynamic Indexes)

利用 Dataview 插件，这里会自动追踪你在此领域的探索进度。

### 📝 待办领域知识 (To-do NLP Topics)

这里列出了所有你标记为 `status: "todo"` 和 `status: "in-progress"` 的NLP笔记，提醒你接下来要深入研究的方向。

```dataview
LIST
FROM "20_Areas (应用领域)/NLP (自然语言处理)"
WHERE (contains(["todo", "in-progress"], status)) AND file.path != this.file.path AND !contains(["hub-note", "map-of-content"], default(type, ""))
SORT file.name ASC
```

### ✨ 最近更新 (Recently Modified)

你最近在 `20_Areas (应用领域)/NLP (自然语言处理)` 目录中编辑过的 10 篇笔记。

```dataview
TABLE WITHOUT ID
	file.link AS "笔记名称",
	file.mtime AS "修改日期"
FROM "20_Areas (应用领域)/NLP (自然语言处理)"
WHERE file.path != this.file.path AND !contains(["hub-note", "map-of-content"], default(type, ""))
SORT file.mtime DESC
LIMIT 10
```

## 🗺️ 自然语言处理全景图 (All NLP Notes Overview)

下面是 `20_Areas (应用领域)/NLP (自然语言处理)` 文件夹下所有笔记的完整列表，以及它们的元数据。

```dataview
TABLE
    type AS "类型",
    tags AS "标签",
    status AS "状态"
FROM "20_Areas (应用领域)/NLP (自然语言处理)"
WHERE file.path != this.file.path AND !contains(["hub-note", "map-of-content"], default(type, ""))
SORT file.folder ASC, file.name ASC
```