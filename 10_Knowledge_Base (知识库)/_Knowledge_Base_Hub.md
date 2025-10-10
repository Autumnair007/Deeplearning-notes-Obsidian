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

### 📝 待办知识点 
这里列出了所有你标记为 `status: "todo"` 和`status:"in-progess"`的笔记，提醒你接下来要攻克的知识点。

```dataview
LIST
FROM "10_Knowledge_Base (知识库)"
WHERE contains(["todo", "in-progress"], status)
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