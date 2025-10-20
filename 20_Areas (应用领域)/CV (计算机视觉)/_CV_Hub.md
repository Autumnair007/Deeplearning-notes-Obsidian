---
type: hub-note
tags:
  - hub
  - cv
  - areas
status: done
---
# 计算机视觉领域 (CV)

> **核心原则**: 此处汇集了所有与计算机视觉（CV）相关的理论、模型和实践笔记。它作为该领域的知识中心，连接基础概念与具体应用，如图像分割、分类和目标检测等。

---

## ⚡ 动态索引 (Dynamic Indexes)

利用 Dataview 插件，这里会自动追踪你在此领域的探索进度。

### 📝 待办领域知识 (To-do CV Topics)

这里列出了所有你标记为 `status: "todo"` 和 `status: "in-progress"` 的CV笔记，提醒你接下来要深入研究的方向。

```dataview
LIST
FROM "20_Areas (应用领域)/CV (计算机视觉)"
WHERE contains(["todo", "in-progress"], status)
SORT file.name ASC
```

### ✨ 最近更新 (Recently Modified)

你最近在 `20_Areas (应用领域)/CV (计算机视觉)` 目录中编辑过的 10 篇笔记。

```dataview
TABLE WITHOUT ID
	file.link AS "笔记名称",
	file.mtime AS "修改日期"
FROM "20_Areas (应用领域)/CV (计算机视觉)"
WHERE file.name != "_CV_Hub"
SORT file.mtime DESC
LIMIT 10
```

## 🗺️ 计算机视觉全景图 (All CV Notes Overview)

下面是 `20_Areas (应用领域)/CV (计算机视觉)` 文件夹下所有笔记的完整列表，以及它们的元数据。

```dataview
TABLE
    type AS "类型",
    tags AS "标签",
    status AS "状态"
FROM "20_Areas (应用领域)/CV (计算机视觉)"
WHERE file.name != "_CV_Hub"
SORT file.folder ASC, file.name ASC
```