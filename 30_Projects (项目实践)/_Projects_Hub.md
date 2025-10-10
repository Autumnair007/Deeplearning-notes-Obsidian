---
type: "hub-note"
tags: [hub, projects]
status: "done"

---
# 项目实践层 (30_Projects)

> **核心原则**: 此处是你将理论付诸实践的地方。存放的是具体的代码实现、项目教程、实验报告和可部署的应用。每个子文件夹代表一个独立的项目或一次技术探索。

---

## ⚡ 动态索引 (Dynamic Indexes)

### 📝 进行中的项目文档 (In-Progress Projects)

这里列出了所有你标记为 `status: "todo"` 或 `status: "in-progress"` 的项目文档或教程，方便你跟踪写作和学习进度。

```dataview
LIST
FROM "30_Projects (项目实践)"
WHERE file.ext = "md" AND contains(["todo", "in-progress"], status)
SORT file.name ASC
```

### ✨ 最近活动 (Recent Activity)

你最近在 `30_Projects (项目实践)` 目录中编辑过的 10 个文件（包括代码和笔记）。

```dataview
TABLE WITHOUT ID
	file.link AS "文件名称",
	file.mtime AS "修改日期"
FROM "30_Projects (项目实践)"
WHERE file.name != "_Projects_Hub"
SORT file.mtime DESC
LIMIT 10
```

## 🗺️ 项目全景图 (All Projects Overview)

### 🚀 项目代码文件 (Project Code Files)

列出所有项目中的代码实现文件。

```dataview
LIST
FROM "30_Projects (项目实践)"
WHERE !contains(file.ext, "md") 
SORT file.folder ASC, file.name ASC
```

### 📚 项目文档与笔记 (Project Documentation & Notes)

下面是 `30_Projects (项目实践)` 文件夹下所有项目文档（Markdown 文件）的完整列表及其元数据。

```dataview
TABLE
    type AS "类型",
    tags AS "标签",
    status AS "状态"
FROM "30_Projects (项目实践)"
WHERE file.ext = "md" AND file.name != "_Projects_Hub"
SORT file.folder ASC, file.name ASC
```