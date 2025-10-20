---
type: hub-note
tags:
  - hub
  - cv
  - tfs
status: done

---
# 🎨 免训练分割专题研究 (Training-Free Segmentation)

> **核心原则**: 此处汇集了所有与“免训练分割”（TFS）相关的模型、方法和理论笔记。TFS旨在利用预训练的通用大模型（如SAM, DINOv2）的能力，在不进行额外训练或微调的情况下，实现对下游任务的分割。

---

## ✨ 如何将笔记添加到这里？

非常简单！你只需要在任何你认为与“免训练分割”相关的笔记的 **元数据区域（Frontmatter）** 中，添加一个标签 `tfs` 即可。这个页面会自动将它们收录进来。

**示例**：
在 `sam_notes.md` 文件的开头添加 `tags: [tfs, model, sam]`。

---

## ⚡ 核心模型与方法 (Core Models & Methods)

这里会自动列出所有被你标记为 `tfs` 的笔记，方便你快速查阅。

```dataview
TABLE
    type AS "类型",
    file.folder AS "所属文件夹",
    status AS "状态"
FROM #tfs AND !"99_Assets (资源文件)" 
WHERE file.name != "_TFS_Hub"
SORT file.name ASC
```

## 🗺️ 相关笔记全景图 (All Related Notes Overview)

下面是所有与TFS相关的笔记，按照文件夹进行分类。

```dataview
LIST
FROM #tfs AND !"99_Assets (资源文件)"
WHERE file.name != "_TFS_Hub"
GROUP BY file.folder
SORT rows.file.name ASC
```

### 📝 最近更新 (Recently Modified)

你最近编辑过的10篇TFS相关笔记。

```dataview
TABLE WITHOUT ID
	file.link AS "笔记名称",
	file.mtime AS "修改日期"
FROM #tfs AND !"99_Assets (资源文件)"
WHERE file.name != "_TFS_Hub"
SORT file.mtime DESC
LIMIT 10
```