---
type: "hub-note"
tags: [hub, cv, index, semantic-segmentation]
status: "done"
---
# 语义分割模型索引 (Semantic Segmentation Models Index)

> **核心原则**: 这里是所有与**语义分割**相关的模型笔记的动态聚合页面。它会自动扫描你的知识库，将所有被标记为 `semantic-segmentation` 的模型和论文笔记汇总于此，方便你进行查阅和对比。

---

## 🗺️ 语义分割模型全景图

利用 Dataview，下面会自动列出所有与语义分割相关的模型笔记，并展示其核心元数据。请确保你的模型笔记拥有 `tags: [..., semantic-segmentation, ...]` 标签。

```dataview
TABLE
    type AS "类型",
    model as "模型",
    year as "年份",
    choice(contains(tags, "panoptic-segmentation"), "是", "否") as "全景分割",
    choice(contains(tags, "instance-segmentation"), "是", "否") as "实例分割",
    status AS "状态"
FROM "10_Knowledge_Base (知识库)" OR "20_Areas (应用领域)"
WHERE contains(tags, "semantic-segmentation") AND file.name != "_Semantic_Segmentation_Index"
SORT year DESC, file.name ASC
```

## ✨ 最近更新的分割模型笔记

你最近编辑过的语义分割相关笔记。

```dataview
TABLE WITHOUT ID
	file.link AS "笔记名称",
	file.mtime AS "修改日期"
FROM "10_Knowledge_Base (知识库)" OR "20_Areas (应用领域)"
WHERE contains(tags, "semantic-segmentation") AND file.name != "_Semantic_Segmentation_Index"
SORT file.mtime DESC
LIMIT 10
```