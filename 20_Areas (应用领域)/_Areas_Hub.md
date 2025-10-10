---
type: "hub-note"
tags: [hub, areas]
status: "done"

---
# 应用领域层 (20_Areas)

> **核心原则**: 此处存放的是将基础知识（知识原子）应用于特定领域的笔记。它们是理论与实践之间的桥梁，展示了深度学习在不同场景下的具体应用，例如计算机视觉（CV）和自然语言处理（NLP）。

---

## ⚡ 动态索引 (Dynamic Indexes)

利用 Dataview 插件，这里会自动追踪你在此领域的探索进度。

### 📝 待办领域知识 (To-do Areas)

这里列出了所有你标记为 `status: "todo"` 和 `status: "in-progress"` 的领域笔记，提醒你接下来要深入研究的方向。

```dataview
LIST
FROM "20_Areas (应用领域)"
WHERE contains(["todo", "in-progress"], status)
SORT file.name ASC
```

### ✨ 最近更新 (Recently Modified)

你最近在 `20_Areas (应用领域)` 目录中编辑过的 10 篇笔记。

```dataview
TABLE WITHOUT ID
	file.link AS "笔记名称",
	file.mtime AS "修改日期"
FROM "20_Areas (应用领域)"
WHERE file.name != "_Areas_Hub"
SORT file.mtime DESC
LIMIT 10
```

## 🗺️ 应用领域全景图 (All Areas Overview)

下面是 `20_Areas (应用领域)` 文件夹下所有领域知识的完整列表，以及它们的元数据。

```dataview
TABLE
    type AS "类型",
    tags AS "标签",
    status AS "状态"
FROM "20_Areas (应用领域)"
WHERE file.name != "_Areas_Hub"
SORT file.folder ASC, file.name ASC
```