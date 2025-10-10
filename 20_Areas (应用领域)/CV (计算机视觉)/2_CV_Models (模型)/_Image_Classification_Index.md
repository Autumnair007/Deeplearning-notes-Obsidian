---
type: "hub-note"
tags: [hub, cv, index, image-classification]
status: "done"
---
# 图像分类模型索引 (Image Classification Models Index)

> **核心原则**: 这里是所有与**图像分类**相关的模型笔记的动态聚合页面。它会自动扫描你的知识库，将所有被标记为 `image-classification` 的模型和论文笔记汇总于此。

---

## 🗺️ 图像分类模型全景图

利用 Dataview，下面会自动列出所有与图像分类相关的模型笔记，并展示其核心元数据。请确保你的模型笔记拥有 `tags: [..., image-classification, ...]` 标签。

```dataview
TABLE
    type AS "类型",
    model as "模型",
    year as "年份",
    status AS "状态",
    file.folder AS "所在文件夹"
FROM "10_Knowledge_Base (知识库)" OR "20_Areas (应用领域)"
WHERE contains(tags, "image-classification")
SORT year DESC, file.name ASC
```

## ✨ 最近更新的分类模型笔记

你最近编辑过的图像分类相关笔记。

```dataview
TABLE WITHOUT ID
	file.link AS "笔记名称",
	file.mtime AS "修改日期"
FROM "10_Knowledge_Base (知识库)" OR "20_Areas (应用领域)"
WHERE contains(tags, "image-classification")
SORT file.mtime DESC
LIMIT 10
```