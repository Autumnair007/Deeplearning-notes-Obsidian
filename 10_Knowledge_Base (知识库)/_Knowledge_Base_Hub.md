---
type: "hub-note"
tags: [hub, knowledge-base]
status: "done"
---
# 知识层 (10_Knowledge_Base)

> **核心原则**: 此处存放的是“放之四海而皆准”的知识原子。它们是构成你所有上层建筑（应用领域、项目实践）的最基础、最纯粹的砖块。

## 常用入口

- [仓库首页](../README.md)
- [监督与学习范式](2_ML_Concepts%20%28%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E6%A6%82%E5%BF%B5%29/supervised_learning_concepts%28%E7%9B%91%E7%9D%A3%E5%AD%A6%E4%B9%A0%E6%A6%82%E5%BF%B5%29.md)
- [Transformer机制](4_DL_Models_And_Mechanisms%20%28%E6%A8%A1%E5%9E%8B%E4%B8%8E%E6%9C%BA%E5%88%B6%29/Transformer/transformer_notes.md)
- [研究算子库](5_Research_Operators%20%28%E7%A0%94%E7%A9%B6%E7%AE%97%E5%AD%90%E5%BA%93%29/_Research_Operators_Hub.md)

---

## ⚡ 动态索引 (Dynamic Indexes)

利用 Dataview 插件，这里会自动追踪你的学习进度。

### 📝 待办知识点 
这里列出了所有你标记为 `status: "todo"` 和 `status: "in-progress"`的笔记，提醒你接下来要攻克的知识点。

```dataview
LIST
FROM "10_Knowledge_Base (知识库)"
WHERE (contains(["todo", "in-progress"], status)) AND file.path != this.file.path AND !contains(["hub-note", "map-of-content"], default(type, ""))
SORT file.name ASC
```

### ✨ 最近更新 (Recently Modified)
你最近在 `10_Knowledge_Base` 目录中编辑过的 10 篇笔记。

```dataview
TABLE WITHOUT ID
	file.link AS "笔记名称",
	file.mtime AS "修改日期"
FROM "10_Knowledge_Base (知识库)"
WHERE file.path != this.file.path AND !contains(["hub-note", "map-of-content"], default(type, ""))
SORT file.mtime DESC
LIMIT 10
```
---
## 🗺️ 知识原子全景图 (All Knowledge Atoms Overview)

下面是 `10_Knowledge_Base` 文件夹下所有知识原子的完整列表，以及它们的元数据。

```dataview
TABLE
    type AS "类型",
    tags AS "标签",
    status AS "状态"
FROM "10_Knowledge_Base (知识库)"
WHERE file.path != this.file.path AND !contains(["hub-note", "map-of-content"], default(type, ""))
SORT file.folder ASC, file.name ASC
```