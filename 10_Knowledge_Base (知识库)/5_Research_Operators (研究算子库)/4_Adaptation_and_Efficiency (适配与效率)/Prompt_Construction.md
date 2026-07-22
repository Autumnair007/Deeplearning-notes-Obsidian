---
type: operator-note
aliases: [Prompt Construction, Prompt Engineering, 提示构建]
tags: [research-operator, prompt, text-embedding, vision-language]
status: todo
---

# Prompt Construction（提示构建）

> [!abstract] 核心直觉
> 类别名不是唯一的文本表示。“dog”可以写成句子、同义词或视觉属性。提示构建决定文本编码器看到什么，提示集成决定多个文本向量怎样变成类别锚点。

> [!tip] 基础机制入口
> CLIP的文本编码器、零样本分类与模板集成背景看 [[clip_notes]] 和 [[clip_paper_notes]]。本页只比较弱监督分割中的前景/背景提示、同义词和视觉属性增强。

## 1. 输入与输出

类别 $c$ 的 $K$ 个提示：

$$\{t_{c,1},\ldots,t_{c,K}\}.$$

文本编码后：

$$E_c\in\mathbb{R}^{K\times D}.$$

最常见集成：

$$\bar e_c=\operatorname{Norm}\left(\frac1K\sum_k\operatorname{Norm}(e_{c,k})\right)\in\mathbb{R}^{D}.$$

先逐提示归一化，避免某个向量因模长大而主导平均；平均后再归一化，使最终类别锚点可用于余弦相似度。

## 2. 常见形式

| 形式 | 示例 | 优点 | 风险 |
|---|---|---|---|
| 固定模板 | `a photo of a {class}` | 简单、零训练 | 域描述可能不合适 |
| 多模板集成 | photo/rendering/close-up等 | 降低单模板偏差 | 编码和存储增加 |
| 同义词融合 | dog/canine等 | 扩大词汇表达 | 同义词可能语境不同 |
| 视觉属性描述 | 颜色、部件、形状 | 更贴近局部外观 | 属性噪声或类间共享 |
| 前景/背景对比提示 | object vs background words | 显式提供负语义 | 背景开放且难枚举 |
| 可学习上下文token | 连续向量 + 类别名 | 数据自适应 | 需训练、可能损害泛化 |

## 3. 为什么简单平均有时不够？

若一组提示中只有少数适合当前数据域，等权平均会把好提示和差提示混在一起。可按权重：

$$e_c=\sum_k\alpha_{c,k}e_{c,k},\qquad\sum_k\alpha_{c,k}=1.$$

权重可以按验证集性能、CAM锐度或学习得到。按图像动态选择提示更灵活，但会把原本可离线计算的文本特征变成输入相关计算。

## 4. 论文中的具体策略

| 论文 | 文本内容 | 聚合/使用方式 |
|---|---|---|
| [[CLIP-ES_paper_notes]] | 前景、背景模板和同义词 | 用CAM锐度选提示，再融合同义词 |
| [[ExCEL_paper_notes]] | 大语言模型生成的细粒度视觉属性 | 搜索隐式属性并与类别语义融合 |
| [[VDA_paper_notes]] | 解耦的视觉属性描述 | 概率建模后组装成类别视觉描述 |
| [[DiCLIP_paper_notes]] | 前景类别与一组背景词 | 模板编码后形成文本分类锚点 |
| [[WeCLIP+_paper_notes]] | 增强文本语义/提示 | 改善冻结CLIP的类别定位 |

## 5. 工程实例：DiCLIP提示集成函数

固定版本 [`1c3f6ff`](https://github.com/zwyang6/DiCLIP/tree/1c3f6ff7d4fde2afff32d527d78b28d119583602)。[`encode_text_with_prompt_ensemble`](https://github.com/zwyang6/DiCLIP/blob/1c3f6ff7d4fde2afff32d527d78b28d119583602/clip/clip.py#L252-L269) 内部流程是：

```python
class_embeddings = model.encode_text(prompted_t)
class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
class_embedding = class_embeddings.mean(dim=0)
class_embedding /= class_embedding.norm()
```

但主模型传入的 `prompt_templates` 当前只有 `"a clean origami {}."` 一个模板。函数名支持ensemble，实验配置却退化为单模板；以后复用代码时不要仅凭函数名判断实际提示数量。

## 6. 背景提示为什么特殊？

前景类别通常有明确名字，背景却包含天空、道路、墙面和未知物体等大量内容。一个“background”向量难覆盖全部背景外观，因此常用多个背景词，或直接在视觉侧建立背景原型。详细比较见 [[Background_and_Unknown_Handling]]。

## 7. 实验与调试

- 保存每个模板的单独性能，不只报告集成结果。
- 检查同义词是否真同义，尤其在数据集标签语境中。
- 属性提示可能共享，如“four-legged”同时适合多类，需观察类间混淆。
- 文本特征可离线缓存；模板、类别顺序和模型版本必须一起保存。
- 集成前后都做L2归一化，并明确平均发生在提示维还是类别维。

## 8. 密集预测为什么比整图分类更依赖词义清洁

整图图文匹配可利用场景上下文，例如“train”常与轨道共同出现；像素/区域分类却应尽量描述目标本身。包含场景、动作或共现物的提示可能提高整图分类，却把CAM激活到上下文。开放词汇分割的提示应优先使用局部可见、具区分度的名词与属性，并单独验证定位质量。

数据集标签还存在多义词与层级冲突：`crane` 可指鸟或起重机，`wall` 在一个数据集是背景、在另一个数据集是前景，`person` 与 `rider` 可能层级重叠。提示模板必须绑定数据集语义定义，而不是只把标签字符串塞入句子。

## 9. 两种集成方式并不等价

嵌入平均后打分：

$$s_c=\cos\left(v,\operatorname{Norm}\left(\sum_k e_{c,k}\right)\right),$$

与逐提示打分后平均：

$$s_c=\frac1K\sum_k\cos(v,e_{c,k})$$

在所有向量已归一化且不再归一化均值时关系较近，但通常不严格相同；前者构造单一类别锚点，后者保留每个提示的独立匹配再聚合。属性差异较大时还可用 `max` 或log-sum-exp，但max更容易被一个偶然高分提示主导。

## 10. 提示的功能分类

| 功能 | 示例内容 | 适用阶段 | 主要风险 |
|---|---|---|---|
| 类别命名 | 类别名、同义词、释义 | 开放分类 | 多义词与标签本体不一致 |
| 域模板 | photo、street scene、medical image | 全局/区域编码 | 把域偏差写进提示 |
| 视觉属性 | 颜色、部件、材质、形状 | patch/region定位 | 属性共享导致类间混淆 |
| 负/背景提示 | background、stuff、非目标描述 | 背景竞争 | 背景无法穷举 |
| 可学习上下文 | 连续prompt token | 目标域适配 | 过拟合seen类别 |

LLM生成属性后应做去重、事实检查和区分度筛选。并非描述越长越好：CLIP文本长度有限，且不可见属性、功能性知识或典型场景会把局部对齐带偏。

## 11. 开放词汇评估规范

- 提示选择和权重只使用训练/验证数据，不能按测试类别逐类试出最好模板。
- 报告单模板、模板集成、同义词和属性增强的独立增益。
- 在seen/unseen、不同数据集和不同词表规模上评估，检查是否只适配固定标签集合。
- 保存准确的模板字符串、冠词/复数处理、类别别名顺序、tokenizer和文本编码器版本。
- 同时看图像级分类与初始密集CAM；前者提高、后者下降说明提示更依赖场景而非目标外观。

## 12. 当前整理结论

提示构建不是修辞润色，而是在定义类别的语义锚点。最有用的提示应补充图像局部可见属性，同时避免引入多个类别共有的模糊描述。
