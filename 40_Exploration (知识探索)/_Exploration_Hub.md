---
type: hub-note
tags: [hub, exploration]
status: done
---
# 知识探索 (40_Exploration)

这里长期保存新概念的速览、跨领域线索和暂时没有固定归属的主题概览。先记录足够帮助理解与回看的内容，再根据实际用途决定是否深入。

## 常用入口

- [仓库首页](../README.md)
- [速览笔记模板](../00_Meta%20%28%E5%85%83%E6%95%B0%E6%8D%AE%29/overview_note_template.md)
- [知识探索提示词](../00_Meta%20%28%E5%85%83%E6%95%B0%E6%8D%AE%29/exploration_prompt.txt)：可直接提供主题或粘贴材料，生成带前置知识与配图的探索笔记。
- [知识库总览](../10_Knowledge_Base%20%28%E7%9F%A5%E8%AF%86%E5%BA%93%29/_Knowledge_Base_Hub.md)
- [应用领域总览](../20_Areas%20%28%E5%BA%94%E7%94%A8%E9%A2%86%E5%9F%9F%29/_Areas_Hub.md)

## 主题笔记

- [3DGS：从前置知识理解三维高斯泼溅](3DGS_overview.md)：从坐标、投影和高斯出发，配合示意图理解表示、渲染、学习及其与三维分割的联系。

## 记录与升级

- 初期保持本目录平铺，不预建细分主题目录。
- 元数据可选；需要时使用 `type: overview-note`、主题标签与实际整理状态。无元数据的笔记也会显示在下方目录全景。
- 速览可以长期保留，不因停留时间变长而自动成为待办。只有主动设置 `status: todo` 或 `status: in-progress` 的笔记才进入待整理列表。
- 一个主题持续增长时，先在本页增加静态导航分组；形成稳定的知识原子、领域专题或项目文档后，再通过链接接入对应Hub。升级以内容成熟度和使用需要为准。

## 目录全景

```dataview
TABLE type AS "类型", tags AS "标签", status AS "整理状态"
FROM "40_Exploration (知识探索)"
WHERE file.path != this.file.path AND !contains(["hub-note", "map-of-content"], default(type, ""))
SORT file.name ASC
```

## 最近更新

```dataview
TABLE WITHOUT ID file.link AS "笔记", file.mtime AS "修改日期"
FROM "40_Exploration (知识探索)"
WHERE file.path != this.file.path AND !contains(["hub-note", "map-of-content"], default(type, ""))
SORT file.mtime DESC
LIMIT 10
```

## 主动标记的待整理笔记

```dataview
LIST
FROM "40_Exploration (知识探索)"
WHERE contains(["todo", "in-progress"], status) AND file.path != this.file.path AND !contains(["hub-note", "map-of-content"], default(type, ""))
SORT file.name ASC
```
