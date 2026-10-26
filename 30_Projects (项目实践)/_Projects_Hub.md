---
type: "hub-note"
tags: [hub, projects]
status: "done"

---
# 项目实践层 (30_Projects)

> **核心原则**: 此处是你将理论付诸实践的地方。存放的是具体的代码实现、项目教程、实验报告和可部署的应用。每个子文件夹代表一个独立的项目或一次技术探索。

## 常用入口

- [仓库首页](../README.md)
- [TorchServe / ResNet18项目说明](TorchServe_deployments/ResNet18%E9%83%A8%E7%BD%B2/README.md)
- [UPerNet实验总结](MMSegmentation_tutorials/UPerNet_tutorial/upernet_conclusion.md)
- [SegFormer实验总结](MMSegmentation_tutorials/SegFormer_tutorial/segformer_conclusion1.md)

---

## ⚡ 动态索引 (Dynamic Indexes)

### 📝 进行中的项目文档 (In-Progress Projects)

这里列出了所有你标记为 `status: "todo"` 或 `status: "in-progress"` 的项目文档与笔记，方便你跟踪写作和学习进度。

```dataview
LIST
FROM "30_Projects (项目实践)"
WHERE (file.ext = "md" AND contains(["todo", "in-progress"], status)) AND file.path != this.file.path AND !contains(["hub-note", "map-of-content"], default(type, ""))
SORT file.name ASC
```

### ✨ 最近活动 (Recent Activity)

你最近在 `30_Projects (项目实践)` 目录中编辑过的 10 篇项目文档与笔记。

```dataview
TABLE WITHOUT ID
	file.link AS "笔记名称",
	file.mtime AS "修改日期"
FROM "30_Projects (项目实践)"
WHERE file.path != this.file.path AND !contains(["hub-note", "map-of-content"], default(type, ""))
SORT file.mtime DESC
LIMIT 10
```

## 🗺️ 项目全景图 (All Projects Overview)

### 🚀 项目代码文件 (Project Code Files)

列出所有项目中的代码实现文件。

```dataviewjs
const folderPath = "30_Projects (项目实践)";
// 获取保险库中所有文件的路径
const allFiles = app.vault.getFiles().map(f => f.path);
// 过滤出指定文件夹下的、以 .py 结尾的文件
const pyFiles = allFiles.filter(p => p.startsWith(folderPath) && p.endsWith(".py"));

// 如果找到了文件，则创建一个列表
if (pyFiles.length > 0) {
    dv.list(pyFiles.map(p => dv.fileLink(p)));
} else {
    dv.paragraph("在此文件夹中未找到任何 .py 文件。");
}
```

### 📚 项目文档与笔记 (Project Documentation & Notes)

下面是 `30_Projects (项目实践)` 文件夹下所有项目文档（Markdown 文件）的完整列表及其元数据。

```dataview
TABLE
    type AS "类型",
    tags AS "标签",
    status AS "状态"
FROM "30_Projects (项目实践)"
WHERE file.ext = "md" AND file.path != this.file.path AND !contains(["hub-note", "map-of-content"], default(type, ""))
SORT file.folder ASC, file.name ASC
```