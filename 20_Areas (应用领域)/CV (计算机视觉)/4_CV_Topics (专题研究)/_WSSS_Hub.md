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

## ⚡ 核心模型与方法 (Core Models & Methods)

这里会自动列出所有被你标记为 `wsss` 的笔记，方便你快速查阅。

```dataview
TABLE
    type AS "类型",
    file.folder AS "所属文件夹",
    status AS "状态"
FROM #wsss AND !"99_Assets (资源文件)" 
WHERE file.name != "_WSSS_Hub"
SORT file.name ASC
```

## 🗺️ 相关笔记全景图 (All Related Notes Overview)

下面是所有与WSSS相关的笔记，按照文件夹进行分类。

```dataview
LIST
FROM #wsss AND !"99_Assets (资源文件)"
WHERE file.name != "_WSSS_Hub"
GROUP BY file.folder
SORT rows.file.name ASC
```

### 📝 最近更新 (Recently Modified)

你最近编辑过的15篇WSSS相关笔记。

```dataview
TABLE WITHOUT ID
	file.link AS "笔记名称",
	file.mtime AS "修改日期"
FROM #wsss AND !"99_Assets (资源文件)"
WHERE file.name != "_WSSS_Hub"
SORT file.mtime DESC
LIMIT 15
```