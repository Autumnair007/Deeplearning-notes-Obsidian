---
type: map-of-content
tags: [wsss, method-review]
status: done
source_checked: 2026-11-11
---

# WSSS 论文 Method 回顾：阅读顺序

这组文档用于逐篇回顾本仓库 Weakly_Supervised 专题中的论文，重点是方法、实际数据流和代码实现。每篇由独立研究任务完成，同一工作的会议版与期刊扩展版集中在一篇讲解中，先会议、后期刊。

## 怎么读

- 按下面顺序逐篇阅读；先看整体流程图，再顺着数据流理解模块和损失，最后对照源码。
- 有可核实的官方代码时，讲解按所引用版本的实际执行流程展开；论文描述和实现不同的地方，以该实现为主。
- 代码版本、期刊扩展关系与资料访问情况，以各篇末尾的“资料与实现版本”说明为准。未公开或无法访问的代码不会被写成已验证实现。
- 各文档中的代码阅读不代表训练、复现或性能验证。
- 原有的专题分析笔记位于相邻的 [Analysis 目录](../Analysis/)。

## 阅读顺序

| 顺序 | 讲解 |
|---|---|
| 01 | [MCTformer与MCTformer+](01_MCTformer与MCTformer+_Method回顾.md) |
| 02 | [BECO](02_BECO_Method回顾.md) |
| 03 | [CLIP-ES](03_CLIP-ES_Method回顾.md) |
| 04 | [WeCLIP与WeCLIP+](04_WeCLIP与WeCLIP+_Method回顾.md) |
| 05 | [S2C](05_S2C_Method回顾.md) |
| 06 | [DiG](06_DiG_Method回顾.md) |
| 07 | [ExCEL](07_ExCEL_Method回顾.md) |
| 08 | [POT](08_POT_Method回顾.md) |
| 09 | [TokenMasking](09_TokenMasking_Method回顾.md) |
| 10 | [SSR](10_SSR_Method回顾.md) |
| 11 | [UGRL](11_UGRL_Method回顾.md) |
| 12 | [CD-CLIP](12_CD-CLIP_Method回顾.md) |
| 13 | [VDA](13_VDA_Method回顾.md) |
| 14 | [DiCLIP](14_DiCLIP_Method回顾.md) |
| 15 | [ComCD](15_ComCD_Method回顾.md) |
| 16 | [ModuSeg](16_ModuSeg_Method回顾.md) |

## 覆盖范围

以弱监督专题中的 17 个论文目录为主线：MCTformer、BECO、CLIP-ES、WeCLIP、WeCLIP+、S2C、DiG、ExCEL、POT、TokenMasking、SSR、UGRL、CD-CLIP、VDA、DiCLIP、ComCD、ModuSeg。

WeCLIP 与 WeCLIP+ 合并为同一篇版本演进讲解，因此主线为 16 篇回顾文档。其他工作的期刊扩展版，在核实后纳入对应文档。年份排序兼顾同一论文家族的连续阅读。

仓库其他位置的开放词汇、全监督、半监督分割基础论文不因带有相关标签而自动计入这 17 个 WSSS 论文目录。

## 会议版与期刊扩展版

- 第 01 篇依次讲 MCTformer（CVPR 2022）和 MCTformer+（TPAMI 2024）。
- 第 04 篇依次讲 WeCLIP（CVPR 2024）和 WeCLIP+（TPAMI 2025）。
- 其他篇目的扩展关系以各篇核查结果为准。作者重合、代码承接或方法相关，本身不作为“期刊扩展版”的确认依据。

## 公开实现的资料边界

以下状态对应本次资料核查，不代表代码今后不会公开：

| 论文 | 本次获取情况 |
|---|---|
| SSR | 未找到可确认的官方实现，采用原论文方法讲解。 |
| UGRL | 论文所列官方仓库返回 404，未能读取源码。 |
| CD-CLIP | 核查了官方仓库与版本记录，但仓库未提供模型实现。 |
| VDA | 核查了官方仓库与版本记录，但算法实现尚未公开。 |
| ComCD | 未定位到可确认归属的官方实现，采用原论文方法讲解。 |

其他文档同样记录具体的源码覆盖范围；仅能读取部分实现时，不把缺失环节写成已经验证的代码流程。

