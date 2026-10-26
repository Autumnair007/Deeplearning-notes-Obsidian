---
type: hub-note
tags:
  - hub
  - cv
  - wsss
status: done
---
# 📚 弱监督语义分割专题研究 (Weakly Supervised Semantic Segmentation)

> **核心原则**: 此处汇集了所有与“弱监督语义分割”（WSSS）相关的模型、方法和理论笔记。WSSS旨在仅使用弱标签（如图像级分类标签、边界框、涂鸦点等）来训练模型，以生成像素级的分割结果，从而大幅降低数据标注成本。

## 常用入口

- [计算机视觉总览](../_CV_Hub.md)
- [多阶段弱监督流程](../3_Methods_and_Frameworks%20%28%E6%96%B9%E6%B3%95%E4%B8%8E%E6%A1%86%E6%9E%B6%29/1_Learning_Paradigms/Weakly_Supervised/Analysis/%E5%A4%9A%E9%98%B6%E6%AE%B5%E5%BC%B1%E7%9B%91%E7%9D%A3%E8%AF%AD%E4%B9%89%E5%88%86%E5%89%B2%E8%AF%A6%E7%BB%86%E6%B5%81%E7%A8%8B.md)
- [CAM与注意力图](../3_Methods_and_Frameworks%20%28%E6%96%B9%E6%B3%95%E4%B8%8E%E6%A1%86%E6%9E%B6%29/1_Learning_Paradigms/Weakly_Supervised/Analysis/%E6%B7%B1%E5%85%A5%E7%90%86%E8%A7%A3CAM%E4%B8%8E%E6%B3%A8%E6%84%8F%E5%8A%9B%E5%9B%BE.md)
- [CLIP-ES论文](../3_Methods_and_Frameworks%20%28%E6%96%B9%E6%B3%95%E4%B8%8E%E6%A1%86%E6%9E%B6%29/1_Learning_Paradigms/Weakly_Supervised/CLIP-ES/CLIP-ES_paper_notes.md)

---

## ✨ 如何将笔记添加到这里？

非常简单！你只需要在任何你认为与“弱监督语义分割”相关的笔记的 **元数据区域（Frontmatter）** 中，添加标签 `wsss` 即可。这个页面会自动将它们收录进来。

**示例**：
在 `KnowYour_Attention_Maps_for_WSSS_paper_notes.md` 文件的开头添加 `tags: [wsss, paper, vit]`。

```yaml
---
tags:
  - wsss
  - paper
---
```

---

## 待整理笔记

只按显式的整理状态列出；`wsss` 与更宽泛的学习范式标签分别保留。

```dataview
LIST
FROM #wsss OR "20_Areas (应用领域)/CV (计算机视觉)/3_Methods_and_Frameworks (方法与框架)/1_Learning_Paradigms/Weakly_Supervised/Analysis"
WHERE (contains(map(default(tags, []), (tag) => lower(tag)), "wsss") OR file.folder = "20_Areas (应用领域)/CV (计算机视觉)/3_Methods_and_Frameworks (方法与框架)/1_Learning_Paradigms/Weakly_Supervised/Analysis") AND contains(["todo", "in-progress"], status) AND file.path != this.file.path AND !contains(["hub-note", "map-of-content"], default(type, ""))
SORT file.name ASC
```

## ⚡ 核心模型与方法 (Core Models & Methods)

这里汇集标记为 `wsss` 的笔记，以及弱监督专题 `Analysis` 目录中的分析笔记，方便你快速查阅。

```dataview
TABLE
    type AS "类型",
    file.folder AS "所属文件夹",
    status AS "状态"
FROM #wsss OR "20_Areas (应用领域)/CV (计算机视觉)/3_Methods_and_Frameworks (方法与框架)/1_Learning_Paradigms/Weakly_Supervised/Analysis"
WHERE (contains(map(default(tags, []), (tag) => lower(tag)), "wsss") OR file.folder = "20_Areas (应用领域)/CV (计算机视觉)/3_Methods_and_Frameworks (方法与框架)/1_Learning_Paradigms/Weakly_Supervised/Analysis") AND file.path != this.file.path AND !contains(["hub-note", "map-of-content"], default(type, ""))
SORT file.name ASC
```

## 🗺️ 相关笔记全景图 (All Related Notes Overview)

下面按文件夹列出 `wsss` 笔记与专题分析笔记。

```dataview
LIST
FROM #wsss OR "20_Areas (应用领域)/CV (计算机视觉)/3_Methods_and_Frameworks (方法与框架)/1_Learning_Paradigms/Weakly_Supervised/Analysis"
WHERE (contains(map(default(tags, []), (tag) => lower(tag)), "wsss") OR file.folder = "20_Areas (应用领域)/CV (计算机视觉)/3_Methods_and_Frameworks (方法与框架)/1_Learning_Paradigms/Weakly_Supervised/Analysis") AND file.path != this.file.path AND !contains(["hub-note", "map-of-content"], default(type, ""))
GROUP BY file.folder
SORT rows.file.name ASC
```

### 📝 最近更新 (Recently Modified)

你最近编辑过的15篇WSSS相关笔记。

```dataview
TABLE WITHOUT ID
	file.link AS "笔记名称",
	file.mtime AS "修改日期"
FROM #wsss OR "20_Areas (应用领域)/CV (计算机视觉)/3_Methods_and_Frameworks (方法与框架)/1_Learning_Paradigms/Weakly_Supervised/Analysis"
WHERE (contains(map(default(tags, []), (tag) => lower(tag)), "wsss") OR file.folder = "20_Areas (应用领域)/CV (计算机视觉)/3_Methods_and_Frameworks (方法与框架)/1_Learning_Paradigms/Weakly_Supervised/Analysis") AND file.path != this.file.path AND !contains(["hub-note", "map-of-content"], default(type, ""))
SORT file.mtime DESC
LIMIT 15
```