---
type: "hub-note"
tags: [hub, areas]
status: "done"

---
# 应用领域层 (20_Areas)

> **核心原则**: 此处存放的是将基础知识（知识原子）应用于特定领域的笔记。它们是理论与实践之间的桥梁，展示了深度学习在不同场景下的具体应用，例如计算机视觉（CV）和自然语言处理（NLP）。

## 常用入口

- [仓库首页](../README.md)
- [计算机视觉](CV%20%28%E8%AE%A1%E7%AE%97%E6%9C%BA%E8%A7%86%E8%A7%89%29/_CV_Hub.md)
- [自然语言处理](NLP%20%28%E8%87%AA%E7%84%B6%E8%AF%AD%E8%A8%80%E5%A4%84%E7%90%86%29/_NLP_Hub.md)
- [项目实践](../30_Projects%20%28%E9%A1%B9%E7%9B%AE%E5%AE%9E%E8%B7%B5%29/_Projects_Hub.md)

---

## ⚡ 动态索引 (Dynamic Indexes)

利用 Dataview 插件，这里会自动追踪你在此领域的探索进度。

### 📝 待办领域知识 (To-do Areas)

这里列出了所有你标记为 `status: "todo"` 和 `status: "in-progress"` 的领域笔记，提醒你接下来要深入研究的方向。

```dataview
LIST
FROM "20_Areas (应用领域)"
WHERE (contains(["todo", "in-progress"], status)) AND file.path != this.file.path AND !contains(["hub-note", "map-of-content"], default(type, ""))
SORT file.name ASC
```

### ✨ 最近更新 (Recently Modified)

你最近在 `20_Areas (应用领域)` 目录中编辑过的 10 篇笔记。

```dataview
TABLE WITHOUT ID
	file.link AS "笔记名称",
	file.mtime AS "修改日期"
FROM "20_Areas (应用领域)"
WHERE file.path != this.file.path AND !contains(["hub-note", "map-of-content"], default(type, ""))
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
WHERE file.path != this.file.path AND !contains(["hub-note", "map-of-content"], default(type, ""))
SORT file.folder ASC, file.name ASC
```