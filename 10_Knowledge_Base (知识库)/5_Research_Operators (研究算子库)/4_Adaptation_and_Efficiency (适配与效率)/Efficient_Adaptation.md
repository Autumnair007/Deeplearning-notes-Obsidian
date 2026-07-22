---
type: operator-note
aliases: [Efficient Adaptation, Parameter-Efficient Adaptation, 高效适配]
tags: [research-operator, adapter, parameter-efficient, frozen-backbone]
status: todo
---

# Efficient Adaptation（高效适配）

> [!abstract] 核心直觉
> 大模型已有通用知识，但直接全量微调成本高且可能破坏预训练能力。高效适配冻结大部分骨干，只学习小型投影、门控、适配器或任务头。

## 1. 输入输出与参数范围

骨干输出：

$$X\in\mathbb{R}^{B\times N\times D}.$$

瓶颈适配器：

$$\operatorname{Adapter}(X)=X+s\,W_{up}\sigma(W_{down}X),$$

其中 $W_{down}:D\rightarrow d$，$W_{up}:d\rightarrow D$，且 $d\ll D$。输出形状仍是 `[B,N,D]`；可训练参数约为 $2Dd$，远小于重训整个Transformer。

例如 $D=512,d=64$，两层无偏置参数约65,536个，而单个 $512\times512$ 线性层就有262,144个参数。

## 2. 常见形式

| 形式 | 学习什么 | 优点 | 局限 |
|---|---|---|---|
| 线性投影 | 一个 $D_{in}\to D_{out}$ 映射 | 最简单、易分析 | 表达力有限 |
| 瓶颈Adapter | down→非线性→up + 残差 | 参数少且保留原特征 | 瓶颈维需选择 |
| 动态门控 | 分支/层/位置权重 | 能按输入自适应 | 可能塌缩为单一路径 |
| 缓存适配器 | 用视觉键值初始化线性层 | 兼具外部知识与学习能力 | 依赖缓存质量 |
| 轻量解码头 | 冻结骨干，只训分割头 | 推理路径清楚 | 骨干表征不足时难补救 |
| Prompt tuning | 只学习提示token | 不改骨干主体 | 密集视觉问题未必仅靠文本解决 |

## 3. 冻结并不等于没有梯度

骨干参数可设 `requires_grad=False`，但若需要对输入或中间特征计算Grad-CAM，仍可能需要建立部分计算图。三件事要分开：

- 参数是否更新；
- 前向是否记录梯度；
- 模型处于 `train()` 还是 `eval()` 模式。

冻结参数后仍应考虑BatchNorm/Dropout状态；`eval()` 会改变它们的行为。

## 4. 论文中的适配位置

| 论文 | 冻结部分 | 可训练部分 | 适配目标 |
|---|---|---|---|
| [[WeCLIP_paper_notes]] | CLIP骨干 | 解码器与RFM相关轻量模块 | 把冻结多层特征解释成分割输出 |
| [[ExCEL_paper_notes]] | CLIP主体 | 可学习视觉校准器 | 修正过平滑patch相关性 |
| [[DiCLIP_paper_notes]] | CLIP与扩散骨干 | 动态KV适配器、分割头 | 适配视觉缓存并蒸馏密集知识 |
| [[Talk2DINO_paper_notes]] | DINOv2与CLIP编码器 | 非线性文本映射 | 把文本送入DINO视觉空间 |
| [[VDA_paper_notes]] | CLIP主体 | 属性建模/组装与解码器增强 | 扩展类别视觉描述 |

## 5. 工程实例：DiCLIP的缓存适配器

固定版本 [`1c3f6ff`](https://github.com/zwyang6/DiCLIP/tree/1c3f6ff7d4fde2afff32d527d78b28d119583602)。[`KV_Adapter`](https://github.com/zwyang6/DiCLIP/blob/1c3f6ff7d4fde2afff32d527d78b28d119583602/model/model_diclip.py#L18-L42) 把静态缓存变成两层映射；[`forward`](https://github.com/zwyang6/DiCLIP/blob/1c3f6ff7d4fde2afff32d527d78b28d119583602/model/model_diclip.py#L148-L185) 将最终patch送入动态适配器：

```python
dynamic_maps = self.dynamic_adapter(image_features[:, 1:, :])
dynamic_maps_pred = dynamic_maps.permute(0, 2, 1).reshape(
    b, self.num_classes, f_h, f_w
)
```

输入 `[B,N,D]`，输出 `[B,N,C]`，再恢复为 `[B,C,H',W']`。适配器没有改变patch数量，主要学习视觉特征到类别缓存响应的映射。

## 6. 训练和验证清单

- 列出可训练参数名与数量，确认骨干没有意外解冻。
- 比较“随机初始化Adapter”和“知识初始化Adapter”，区分结构收益与先验收益。
- 给残差分支使用小尺度或零初始化，避免训练初期破坏原表征。
- 记录训练显存、吞吐与推理是否仍需辅助大模型。
- 适配层输入输出都要标形状，尤其注意token-first与batch-first转换。

## 7. 什么时候不够？

若任务与预训练域差异极大，固定特征根本不包含所需边界或类别信息，小适配器只能重新组合已有信息，不能凭空创造表征。此时可逐步解冻后几层，而不是从“全冻结”直接跳到“全量微调”。

## 8. 适配方法的完整坐标

| 方法 | 改动位置 | 参数随类别数增长吗 | 开放词汇风险 |
|---|---|---:|---|
| linear probe/分割头 | 骨干输出之后 | 固定类头会 | 容易退化成封闭分类器 |
| Adapter | Transformer块内/后 | 通常否 | 过拟合训练域会改变通用表征 |
| LoRA | 注意力或MLP权重的低秩增量 | 否 | 需检查未见类保持 |
| prompt tuning | 文本/视觉输入token | 通常可共享 | 类特定prompt可能难迁移 |
| 部分解冻 | 后若干block/Norm | 否 | 参数更多，灾难性遗忘风险上升 |
| 全量微调 | 全骨干 | 否 | 成本最高，开放能力可能下降 |

LoRA常写成：

$$W'=W+\frac{\alpha}{r}BA,qquad A\in\mathbb R^{r\times D_{in}},\ B\in\mathbb R^{D_{out}\times r},$$

其中 $r$ 是秩。它减少可训练参数，但前向仍使用原大矩阵 $W$，所以不会按参数比例降低基础骨干FLOPs。

## 9. 密集任务特有的适配位置

图像分类适配只需改善全局token，分割还要求局部token和边界关系不被破坏。适配器可放在：

- 每个patch的通道投影：不直接混合空间，成本低；
- attention的Q/K/V：会改变token关系，应监控边界和过平滑；
- 多层解码器：保留骨干，专门学习空间恢复；
- 文本到视觉空间映射：如[[Talk2DINO_paper_notes]]，保持DINO空间结构；
- 缓存/原型读出：如[[DiCLIP_paper_notes]]，不必全量修改CLIP。

选择位置应对应实际错位：语义类名不适配优先改文本/投影，边界差优先改空间解码或关系，而不是默认给所有block加Adapter。

## 10. 参数效率、训练效率和推理效率

这三者必须分别报告：

1. **参数效率**：可训练参数/总参数，优化器状态大小。
2. **训练效率**：峰值显存、每步时间、是否需冻结教师多次前向。
3. **推理效率**：部署保留的全部模型、FLOPs、延迟和缓存大小。

冻结参数减少梯度和优化器内存，但不会消除其前向计算；训练时依赖SAM/扩散教师、推理时移除它们的方案，训练和部署成本应分两行写。

## 11. 保持开放能力的验证

- 比较适配前后seen、unseen和跨数据集性能，而不只看训练集类别。
- 在扩大/替换文本词表后测试，确认输出头没有写死类别数。
- 检查图文相似度排序和文本模板鲁棒性是否退化。
- 做从head-only到逐层解冻的曲线，找到收益与遗忘的拐点。
- 保存可训练参数清单、初始权重和冻结状态，保证参数量统计可复现。

## 12. 当前整理结论

高效适配的本质是选择最小的可学习接口：

$$\boxed{\text{尽量保留预训练知识，只在任务真正错位的位置增加自由度}}.$$
