---
type: operator-note
aliases: [Background Handling, Unknown Handling, 背景与未知类处理]
tags: [research-operator, background, unknown, open-world]
status: todo
---

# Background and Unknown Handling（背景与未知类处理）

> [!abstract] 核心直觉
> 前景类别可以用名称枚举，背景却不是一个稳定视觉类别；开放词汇场景还要区分“已知前景之外的真实物体”和“无意义背景”。因此背景不能总被当成普通的第 $C+1$ 类。

## 1. 输入输出

前景分数：

$$S_{fg}\in\mathbb{R}^{B\times C\times H\times W}.$$

背景分数 $S_{bg}\in\mathbb{R}^{B\times1\times H\times W}$，拼接后：

$$S=\operatorname{Concat}(S_{fg},S_{bg})\in
\mathbb{R}^{B\times(C+1)\times H\times W}.$$

如果还显式区分unknown，则再增加一个通道，或通过拒识阈值将低可靠位置标成未知。

## 2. 常见背景分数

| 形式 | 公式/来源 | 优点 | 风险 |
|---|---|---|---|
| 前景补集 | $S_{bg}=1-\max_cS_c$ | 极简、无需背景模型 | 前景分数未校准时不可靠 |
| 固定阈值 | 所有前景低于阈值则背景 | 可控 | 阈值跨数据集不稳定 |
| 背景文本 | background/stuff/场景词向量 | 保留开放文本接口 | “背景”语义过宽 |
| 多背景原型 | 多个视觉/文本背景中心 | 覆盖多样外观 | 与未知前景易混淆 |
| 显式背景分支 | 网络预测独立背景logit | 可学习上下文 | 需要监督或可靠伪标签 |
| 拒识/unknown | 低最大概率、高熵或能量阈值 | 允许模型说“不知道” | 置信度需校准 |

## 3. 前景补集怎样理解？

$$S_{bg}=1-\max_cS_{fg,c}.$$

若一个像素最高前景响应为0.9，背景分数0.1；最高前景只有0.2，背景分数0.8。它假设前景响应已经位于 `[0,1]` 且不同图像可比。若CAM只是每类独立min-max归一化，这个假设往往不成立：每个类都可能在某处达到1。

## 4. unknown与background的区别

- **background**：任务定义中不需要命名的区域，例如VOC中的道路或天空。
- **unknown object**：确实是一个物体，但不属于当前已知类别集合。
- **ignore**：训练时暂时不相信其标签，不代表它在语义上属于某类。

把三者都编码成0会让模型学不到拒识能力，也会把未见物体压进背景。

## 5. 论文中的处理

| 论文 | 背景/未知信号 | 目的 |
|---|---|---|
| [[CLIP-ES_paper_notes]] | 定制背景文本与前景/背景softmax竞争 | 减少非目标类和背景混淆 |
| [[Talk2DINO_paper_notes]] | 识别与所有文本类别都不匹配的区域 | 在开放词汇定位中抑制背景 |
| [[SSR_paper_notes]] | 原型校正 + 超像素传播限制 | 避免亲和力把前景扩散到背景 |
| [[ComCD_paper_notes]] | CLIP/扩散分支不确定性 | 在背景和模糊位置选择可靠预测 |
| [[POT_paper_notes]] | CAM种子和多原型 | 补全前景但需避免原型吸入背景 |
| [[DiCLIP_paper_notes]] | 背景文本 + 前景/背景视觉缓存 | 用多种背景外观参与检索竞争 |

## 6. 工程实例：S2C的补集背景

固定版本 [`102e14c`](https://github.com/sangrockEG/S2C/blob/102e14c690c8e3bce3d5ccd1ae7832145ce10b27/models/model_s2c.py#L345-L349) 中：

```python
cam_bg = 1 - cam_main.max(dim=1, keepdims=True)[0]
cam_main = torch.cat((cam_main, cam_bg), dim=1)
self.loss_cpm = F.cross_entropy(cam_main, pgt_sam, ignore_index=255)
```

这里先从最高前景CAM构造背景，再拼到类别维，用SAM伪标签监督。注意 `cross_entropy` 接收的通常应是未归一化logit；而这里CAM已经过ReLU、最大值归一化和背景补集，它更像分数而非标准logit。阅读仓库时应记录这种“数学上可运行但概率解释不完全标准”的实现差异。

## 7. 一个简单拒识规则

$$
\hat y=\begin{cases}
\arg\max_c p_c,&\max_cp_c\ge\tau_p\text{ 且 }H(p)\le\tau_H,\\
\text{unknown},&\text{otherwise}.
\end{cases}
$$

最大概率防止所有类别都低，熵阈值防止多个类别接近。两者结合比单一阈值更保守，但仍需在未知类验证集上校准。

## 8. 调试与评价

- 分别统计已知前景、背景、未知物体的混淆，不只看总mIoU。
- 可视化“被预测成背景的真实前景”和“被预测成已知类的未知物体”。
- 背景原型应保持多样性，避免被数量最多的天空/道路主导。
- 弱监督训练中保留ignore区，不要过早强制成背景。
- 开放词汇类别集合变化后重新校准阈值；softmax概率会随候选类数量变化。

## 9. 四个容易混淆的任务设定

| 设定 | 测试类别 | 未知类要求 | 背景含义 |
|---|---|---|---|
| 闭集语义分割 | 与训练类相同 | 不要求拒识 | 数据集未标注为前景的统一类 |
| 零样本/开放词汇分割 | 可包含训练未见文本类 | 要能命名给定词表中的未见类 | 词表外区域通常仍并入背景 |
| 开放集分割 | 含词表外物体 | 要识别为unknown | background与unknown应分开 |
| 开放世界分割 | unknown后续可能被增量命名 | 要发现、拒识并继续学习 | 背景定义随任务扩展 |

因此“开放词汇”不必然意味着能够拒识任何未知物体：只要测试时给出了类别名称，模型仍是在给定候选集合内分类。评估或笔记中应明确词表是否预先给定、词表外物体如何计分。

## 10. no-object、void和ignore

- **no-object**：mask分类中某个query没有匹配真实区域，是查询级训练类别。
- **void/unlabeled**：数据集不提供语义标注的像素，评价时通常排除。
- **ignore**：训练算法暂时不使用的像素，可来自不确定伪标签。
- **background**：任务明确要求预测的背景类。

它们在代码中都可能被编码为0、255或最后一个类别，但不能据整数值判断语义。[[maskformer_notes]] 的no-object服务于query匹配；WSSS的ignore用于控制伪监督噪声，二者不是同一概念。

## 11. 开放集拒识分数

除最大概率和熵，还可使用能量：

$$
E(x)=-T\log\sum_{c=1}^{C}\exp(z_c(x)/T).
$$

它保留logit整体幅值信息，但阈值仍需已知/未知验证数据校准。对余弦分类器还要固定或记录logit scale；scale变化会直接改变能量。增加大量相近文本类别也会改变 $E$，因此跨词表比较要谨慎。

背景文本可使用多个负提示或视觉原型，但“wall/road/sky”等词在其他数据集可能是合法前景类。更稳妥的实现应把任务标签本体和背景提示配置一起保存，防止语义冲突。

## 12. 更完整的评估

- 闭集：前景mIoU、背景IoU和边界指标。
- 开放词汇：seen mIoU、unseen mIoU及调和平均，避免只优化一侧。
- 开放集：unknown IoU、AUROC、AUPR、FPR@95TPR，并报告阈值来源。
- 错误分解：已知→背景、未知→已知、背景→前景分别统计。
- 词表敏感性：加入同义词、相近类和无关类后，检查已知预测是否稳定。

## 13. 当前整理结论

背景处理的关键是承认它不是单一语义类；开放世界还要允许模型区分“这是任务背景”和“这是我尚未命名的物体”。
