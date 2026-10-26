---
type: "hub-note"
tags: [hub, cv, index, semantic-segmentation]
status: "done"
---
# 语义分割模型索引 (Semantic Segmentation Models Index)

> **核心原则**: 这里是所有与**语义分割**相关的模型笔记的动态聚合页面。它会自动扫描你的知识库，将所有被标记为 `semantic-segmentation` 的模型和论文笔记汇总于此，方便你进行查阅和对比。

## 常用入口

- [计算机视觉总览](../_CV_Hub.md)
- [FCN](2_Segmentation%20%28%E5%9B%BE%E5%83%8F%E5%88%86%E5%89%B2%29/FCN/fcn_notes.md)
- [DeepLabV3+](2_Segmentation%20%28%E5%9B%BE%E5%83%8F%E5%88%86%E5%89%B2%29/DeepLabV3+/deeplabv3+_notes.md)
- [UPerNet](2_Segmentation%20%28%E5%9B%BE%E5%83%8F%E5%88%86%E5%89%B2%29/UPerNet/upernet_notes.md)
- [弱监督专题](../4_CV_Topics%20%28%E4%B8%93%E9%A2%98%E7%A0%94%E7%A9%B6%29/_WSSS_Hub.md)

---

## 🗺️ 语义分割模型全景图

利用 Dataview，下面会自动列出所有与语义分割相关的模型笔记，并展示其核心元数据。请确保你的模型笔记拥有 `tags: [..., semantic-segmentation, ...]` 标签。

年份依次读取有效四位整数 `year`、`paper_year`，再从 `venue` 识别唯一明确年份或合法月份的 `arXiv:YYMM.NNNNN`；仅 `venue` 出现多个不同年份或无法确定时留空，显式 `year` / `paper_year` 仍按上述优先级读取。此索引只展示信息，不补写论文年份。

```dataviewjs
function noteYear(page) {
    const fourDigitYear = value => {
        const text = String(value ?? "").trim();
        return /^[1-9]\d{3}$/.test(text) ? Number(text) : null;
    };
    const explicit = fourDigitYear(page.year) ?? fourDigitYear(page.paper_year);
    if (explicit !== null) return explicit;
    const venue = typeof page.venue === "string" ? page.venue : "";
    const venueText = venue.replace(/\barxiv\s*:\s*\d{4}\.\d{4,5}(?:v\d+)?\b/gi, "")
        .replace(/\b\d{4}\.\d{4,5}(?:v\d+)?\b/g, "");
    const years = new Set(Array.from(venueText.matchAll(/(?<!\d)[1-9]\d{3}(?!\d)/g), m => Number(m[0])));
    for (const match of venue.matchAll(/\barxiv\s*:\s*(\d{2})(0[1-9]|1[0-2])\.\d{4,5}(?:v\d+)?\b/gi)) {
        years.add(2000 + Number(match[1]));
    }
    return years.size === 1 ? [...years][0] : null;
}
const hasTag = (page, tag) => Array.from(page.tags ?? []).some(t => String(t).toLowerCase() === tag);
const rows = Array.from(dv.pages('"10_Knowledge_Base (知识库)" OR "20_Areas (应用领域)"'))
    .filter(p => p.file.path !== dv.current().file.path && !["hub-note", "map-of-content"].includes(p.type) && hasTag(p, "semantic-segmentation"))
    .map(p => ({ p, year: noteYear(p) }))
    .sort((a, b) => (b.year ?? 0) - (a.year ?? 0) || a.p.file.name.localeCompare(b.p.file.name));
dv.table(["笔记", "类型", "模型", "年份", "全景分割", "实例分割", "状态"], rows.map(({ p, year }) => [p.file.link, p.type ?? "", p.model ?? "", year ?? "", hasTag(p, "panoptic-segmentation") ? "是" : "否", hasTag(p, "instance-segmentation") ? "是" : "否", p.status ?? ""]));
```

## ✨ 最近更新的分割模型笔记

你最近编辑过的语义分割相关笔记。

```dataview
TABLE WITHOUT ID
	file.link AS "笔记名称",
	file.mtime AS "修改日期"
FROM "10_Knowledge_Base (知识库)" OR "20_Areas (应用领域)"
WHERE contains(map(default(tags, []), (tag) => lower(tag)), "semantic-segmentation") AND file.path != this.file.path AND !contains(["hub-note", "map-of-content"], default(type, ""))
SORT file.mtime DESC
LIMIT 10
```

## 待整理模型笔记

```dataview
LIST
FROM "10_Knowledge_Base (知识库)" OR "20_Areas (应用领域)"
WHERE contains(map(default(tags, []), (tag) => lower(tag)), "semantic-segmentation") AND contains(["todo", "in-progress"], status) AND file.path != this.file.path AND !contains(["hub-note", "map-of-content"], default(type, ""))
SORT file.name ASC
```
