---
type: operator-note
aliases: [Feature Extraction, 特征提取]
tags: [research-operator, feature-extraction, weakly-supervised, open-vocabulary]
status: todo
---

# Feature Extraction（特征提取）

> [!abstract] 本页定位
> 特征提取不是一句“使用CLIP/DINO作为骨干”就结束，而是决定从哪个模型、哪一层、以全局还是密集形式取出什么张量。后续所有融合和监督都受这个选择约束。

> [!tip] 基础机制入口
> ViT的patch embedding、CLS token和编码器结构优先看 [[vision_transformer_notes]]；CLIP双编码器看 [[clip_notes]]；DINOv2的patch级自监督表征看 [[dinov2_notes]]。本页只讨论这些特征怎样被弱监督与开放词汇方法取出和消费。

## 1. 解决什么问题？

同一骨干通常同时包含不同性质的信息：浅层分辨率高、边缘与纹理清楚；深层语义强，却可能把同类区域过度平滑。分类只需要判断“有没有”，分割还要知道“在哪里”。因此要明确：

- 取全局特征还是patch特征；
- 只取最终层还是保留多个中间层；
- 骨干冻结还是参与训练；
- 是否同时调用CLIP、DINO、SAM或扩散模型等互补骨干。

## 2. 输入与输出张量

输入图像为：

$$I\in\mathbb{R}^{B\times3\times H\times W}.$$

卷积骨干常输出：

$$F_l\in\mathbb{R}^{B\times D_l\times H_l\times W_l}.$$

Vision Transformer（视觉Transformer，ViT）则常输出：

$$X_l=[x_{cls};P_l]\in\mathbb{R}^{B\times(1+N_l)\times D_l},\qquad N_l=H_lW_l.$$

`CLS` token是整图摘要，$P_l$ 是保留空间顺序的patch序列。若 $B=2$、输入为 $320\times320$、patch大小为16，则 $H_l=W_l=20$、$N_l=400$，输出可能是 `[2,401,512]`。去掉第一个token后才得到 `[2,400,512]` 的密集特征。

## 3. 常见实现形式

| 形式 | 输出 | 优点 | 主要风险 | 代表笔记 |
|---|---|---|---|---|
| 最终层密集特征 | $[B,N,D]$ | 语义最强、接口简单 | 边界与局部差异可能被平滑 | [[DiCLIP_paper_notes]]、[[ExCEL_paper_notes]] |
| 多个中间层 | $\{F_l\}$ | 同时保留细节与语义 | 维度/分辨率不一致 | [[WeCLIP_paper_notes]]、[[WeCLIP+_paper_notes]] |
| 多尺度输入 | 多组 $[B,N_s,D]$ | 改善大小物体覆盖 | 推理计算随尺度数增长 | [[S2C_paper_notes]] |
| 多骨干并行 | $F^{clip},F^{dino},F^{sam}$ | 语义、局部关系、边界互补 | 对齐和显存成本高 | [[Trident_paper_notes]]、[[CorrCLIP_paper_notes]] |
| 冻结骨干 + 轻量头 | 固定特征 + 可训练预测 | 训练稳定、成本低 | 上限受固定表征限制 | [[WeCLIP_paper_notes]]、[[DiCLIP_paper_notes]] |

## 4. 关键操作怎样理解？

### 4.1 选择最终层

$$P=X_L[:,1:,:]\in\mathbb{R}^{B\times N\times D}.$$

这里的 `1:` 是丢掉CLS，只保留空间patch。它没有改变特征值，只改变选择范围。若误把CLS一起reshape为二维网格，就会因为 $1+N$ 通常不是平方数而出错。

### 4.2 收集中间层

$$\mathcal F=\{P_{l_1},P_{l_2},\ldots,P_{l_m}\}.$$

这一步暂时不融合，只把候选层保存下来。后续应先用投影统一通道维，再用插值统一空间分辨率，详见 [[Multi_Level_Fusion]]。

### 4.3 多尺度提取

对尺度集合 $\mathcal S$：

$$P^{(s)}=E(\operatorname{Resize}(I,s)),\quad s\in\mathcal S.$$

例如对原图、0.5倍和1.5倍图像分别编码。大尺度更容易保留小物体，小尺度带来更大感受野；最终必须把响应图恢复到同一 $H\times W$ 后才能求和。

## 5. 论文之间的具体差异

| 论文 | 提取什么 | 后续怎样使用 |
|---|---|---|
| [[WeCLIP_paper_notes]] | 冻结CLIP的多层视觉特征和注意力 | 多层特征送入解码器，注意力用于CAM细化 |
| [[ExCEL_paper_notes]] | CLIP patch特征 | 静态和可学习视觉校准先修正patch关系，再与文本匹配 |
| [[DiCLIP_paper_notes]] | CLIP token、各层特征和注意力 | 扩散亲和力注入注意力；patch用于文本CAM与视觉缓存检索 |
| [[CorrCLIP_paper_notes]] | CLIP的Q/K/V与DINO相关性 | 重建交互范围和值，再执行文本分类 |
| [[Trident_paper_notes]] | 子图的CLIP/DINO特征与SAM全局关系 | 先拼接局部特征，再用SAM关联矩阵全局聚合 |

## 6. 工程实例：DiCLIP怎样暴露多种特征

固定阅读版本为 [`1c3f6ff`](https://github.com/zwyang6/DiCLIP/tree/1c3f6ff7d4fde2afff32d527d78b28d119583602)。主调用位于 [`DiCLIP_model.forward`](https://github.com/zwyang6/DiCLIP/blob/1c3f6ff7d4fde2afff32d527d78b28d119583602/model/model_diclip.py#L148-L185)：

```python
image_features, attn_weights, all_feats = clip.generate_clip_fts(
    img, self.encoder, return_weights=True, ex_feats=diff_attn
)
```

三个返回值分工不同：`image_features` 是最终token，用于CAM和检索；`attn_weights` 是token关系；`all_feats` 是中间层特征，交给分割头。阅读复杂模型时不要只记录“backbone输出”，最好把每个返回值的消费者也记下来。

## 7. 选型检查

- 只做图像分类：全局向量通常足够。
- 要直接生成密集响应：必须保留patch或二维特征图。
- 最终层定位过粗：优先尝试中间层或多尺度，而不是立即更换损失函数。
- 多骨干互补：先检查空间分辨率、通道维和坐标顺序，再谈融合。
- 冻结骨干：显存省，但要确保轻量头有能力把固定特征解释成目标任务输出。

## 8. 放回经典分割、弱监督与开放词汇三条主线

| 范式 | 特征必须保留什么 | 典型读取位置 | 后续消费者 | 最常见的错误归因 |
|---|---|---|---|---|
| 全监督经典分割 | 多尺度空间细节与任务语义 | CNN各stage、分层Transformer四级输出 | FCN/DeepLab/UPerNet/SegFormer解码头 | 把解码器不足误认为骨干不够强 |
| 图像级弱监督分割 | 分类语义、可反传类别分数、patch关系 | 分类器前特征、ViT patch与注意力 | CAM、亲和传播、伪标签生成 | 只取最深层后要求它同时给出精细边界 |
| 开放词汇分割 | 可与文本比较的语义 + 可定位的空间结构 | CLIP密集token，或DINO/SAM与CLIP的组合 | patch/region-text分类器、mask分类器 | 把DINO与CLIP同维特征直接点积 |

经典架构给出了几个重要基线：[[fcn_notes]] 说明跳跃连接用浅层位置细节补深层语义；[[deeplabv3+_notes]] 通过空洞卷积和ASPP在不过度降采样时扩大上下文；[[upernet_notes]] 用FPN式自顶向下路径统一多级特征；[[segformer_notes]] 则把四级Transformer特征投影到共同通道后由轻量MLP解码。它们提醒我们，弱监督或开放词汇论文即使更换了监督来源，也没有取消分割对“语义—分辨率折中”的基本要求。

## 9. 取层、归一化和空间坐标的细节

### 9.1 取层不是越深越好

可把每层的用途分开验证：用全局池化后的线性分类检查语义可分性，用冻结特征上的轻量分割头检查空间可用性。分类准确率最高的层不一定产生最好的CAM边界。对ViT还要记录特征来自block输入、block输出、最终LayerNorm之前还是之后；这些张量形状可能相同，数值分布和局部结构却不同。

### 9.2 token必须能恢复到原图坐标

若输入经过缩放、裁剪或padding，patch索引对应的是**预处理后的图像坐标**。密集预测回原图时应逆序执行：先恢复patch网格，再去padding，再逆缩放/逆裁剪。仅调用一次 `interpolate` 不能自动解决坐标偏移。多尺度和翻转测试也应先把每个响应逆变换到同一坐标系后再融合。

### 9.3 不同骨干的数值尺度不可直接比较

CNN、CLIP、DINO和SAM特征可能有不同LayerNorm、通道方差和模长。拼接前可使用独立投影与归一化：

$$
\tilde F^{(m)}=\operatorname{Norm}\bigl(\phi_m(F^{(m)})\bigr),
$$

其中每个模型使用自己的 $\phi_m$。这一步只让表示适合融合，不代表跨模态语义已建立；需要文本接口时仍应交给 [[Cross_Modal_Alignment]]。

## 10. 可复现实验与诊断

- **取层消融**：逐层报告图像分类、初始CAM mIoU和最终分割mIoU，分清语义收益与空间收益。
- **分辨率消融**：固定骨干与损失，只改变输入尺度、patch大小或output stride；同时报告显存和吞吐。
- **冻结消融**：比较全冻结、只解冻后几层和全量微调，并检查未见类性能是否下降。
- **特征可视化**：不要只看PCA彩图；同时检查最近邻、位置自相似和边界两侧余弦相似度。
- **消费者追踪**：为每个返回张量记录去向。没有被任何损失或预测头使用的“多层特征”不会自动产生收益。

## 11. 先建立一张完整的“特征类型地图”

“特征”不是单一张量。分割论文至少会使用下面六类表示：

| 特征类型 | 常见形状 | 空间粒度 | 最适合回答的问题 | 典型消费者 |
|---|---|---|---|---|
| 全局图像特征 | $[B,D]$ | 整图 | 图中是否存在类别 $c$？ | 多标签分类、图文对比、图像检索 |
| 稠密像素/patch特征 | $[B,N,D]$ 或 $[B,D,H',W']$ | 规则网格 | 哪个位置属于类别 $c$？ | 像素分类、CAM、patch-text匹配 |
| 多层特征金字塔 | $\{[B,D_l,H_l,W_l]\}_{l=1}^L$ | 多种分辨率 | 如何兼顾语义与边界？ | FPN、UPerNet、SegFormer解码器 |
| 区域特征 | $[B,R,D]$ | mask/box/片段 | 这个完整区域是什么？ | 区域分类、开放词汇mask分类 |
| 关系特征 | $[B,N,N]$ | 位置对 | 哪些位置应该一起变化？ | 亲和传播、相关性重建、随机游走 |
| 类别/文本特征 | $[C,D]$ | 类别 | 视觉单元与哪个语义锚点匹配？ | 开放词汇分类、原型对比 |

同一编码器可能同时返回其中多种。例如ViT最终输出既含一个全局CLS，也含 $N$ 个patch；其注意力又产生 $N\times N$ 关系。论文写“使用CLIP特征”时，必须进一步问：使用哪个token、哪个block、LayerNorm前后哪个版本、是否保留空间顺序，以及该张量最终送到哪里。

## 12. CNN特征：分辨率、output stride与感受野

### 12.1 下采样后的空间尺寸

对二维卷积或池化，单个方向的输出尺寸为：

$$
H_{out}=\left\lfloor\frac{H_{in}+2p-d(k-1)-1}{s}+1\right\rfloor,
$$

其中 $k$ 是核大小、$s$ 是步长、$p$ 是padding、$d$ 是dilation。连续层的总下采样倍率称为output stride（OS）。输入 $512\times512$，若OS为32、16、8，则最终网格分别约为 $16\times16$、$32\times32$、$64\times64$。

更小的OS保留更多位置，但注意力、卷积和激活显存随 $H'W'$ 增长。[[deeplabv3+_notes]] 使用空洞卷积在不继续降采样的情况下扩大感受野，其核心不是“让图变大”，而是让稠密网格上的每个位置看到更广上下文。

### 12.2 层级特征的语义分工

以四级CNN或分层Transformer为例：

$$
F_1:[B,D_1,H/4,W/4],\quad
F_2:[B,D_2,H/8,W/8],
$$

$$
F_3:[B,D_3,H/16,W/16],\quad
F_4:[B,D_4,H/32,W/32].
$$

- $F_1/F_2$：边缘、纹理、小结构明显，但容易把相似纹理的不同类别混在一起。
- $F_3/F_4$：类别和上下文更稳定，但小目标可能已消失，边界也更平滑。
- [[fcn_notes]] 的skip connection、[[upernet_notes]] 的FPN、[[segformer_notes]] 的多级MLP投影，本质上都在重组这组张量。

“浅层负责边界、深层负责语义”只是统计倾向，不是硬规则。应通过线性探针、CAM质量或轻量解码器消融验证具体骨干，而不是按层号直接下结论。

## 13. ViT特征：token序列怎样变回二维空间

### 13.1 patch embedding

若patch大小为 $P_h\times P_w$，输入经过必要padding后得到：

$$
H_p=\left\lceil\frac{H}{P_h}\right\rceil,\qquad
W_p=\left\lceil\frac{W}{P_w}\right\rceil,\qquad
N=H_pW_p.
$$

标准非重叠patch embedding可视为核大小和步长都为 $P$ 的卷积。SegFormer的overlapping patch embedding则让核大小大于步长，使相邻token共享像素，通常更利于局部连续性。

### 13.2 特殊token不能一律假设只有一个

常见序列可能是：

```text
[CLS] + patch tokens
[CLS] + [DIST] + patch tokens
[CLS] + register tokens + patch tokens
pure patch tokens（部分密集视觉编码器）
```

因此不要硬编码 `x[:, 1:, :]` 后就假设剩余长度可reshape。更稳妥的做法是从模型配置读取 `num_prefix_tokens`、patch grid或接口返回的 `x_norm_patchtokens`。SAM面向密集mask生成，其图像编码接口通常直接提供二维图像嵌入，而不是依赖分类CLS。

### 13.3 序列恢复

对patch序列 $P\in\mathbb R^{B\times N\times D}$：

$$
P_{2d}=\operatorname{Permute}\left(
\operatorname{Reshape}(P,B,H_p,W_p,D)
\right)
\in\mathbb R^{B\times D\times H_p\times W_p}.
$$

必须使用真实的 $H_p,W_p$，不能用 $\sqrt N$ 猜正方形；长宽不等、动态padding或滑动窗口都会让这个假设失败。

### 13.4 位置编码插值不等于特征插值

ViT改变输入分辨率时，常先对预训练位置编码插值，再运行Transformer；预测结束后还会对输出响应上采样。这是两个不同步骤：前者让编码器接受新token网格，后者把低分辨率预测恢复到原图。漏掉前者可能导致形状错误或位置先验失配，漏掉后者则无法得到像素输出。

## 14. 从Transformer block的哪里取？

一个pre-norm block可概括为：

$$
X'=X+\operatorname{MSA}(\operatorname{LN}_1(X)),
$$

$$
X_{out}=X'+\operatorname{MLP}(\operatorname{LN}_2(X')).
$$

可选特征包括：

| 位置 | 保留的信息 | 使用时要注意什么 |
|---|---|---|
| block输入 $X$ | 上一层完整表示 | 语义相对较浅 |
| LN后的Q/K/V输入 | 归一化后用于注意力 | 与block最终输出不是同一空间 |
| attention输出 $AV$ | token关系聚合结果 | 尚未经过输出投影/残差/MLP |
| 第一次残差后 $X'$ | attention更新 + 原表示 | 常用于分析注意力影响 |
| block最终输出 $X_{out}$ | attention、残差、MLP完整结果 | 可能最语义化，也可能更空间不变 |
| 最终LayerNorm后 | 稳定的下游接口 | 数值分布与LN前明显不同 |

[[CorrCLIP_paper_notes]] 关注CLIP最后层patch相关性及其范围和值，说明开放词汇分割的瓶颈可能发生在文本点积**之前**。从最终block简单取出patch并不保证它们仍具有充分位置判别力。

## 15. CLIP、DINO、SAM分别提供什么

### 15.1 CLIP：开放语义接口强，密集定位不是预训练主目标

CLIP图像—文本对比主要约束全局图像向量与整句文本。用于分割时常取patch token：

$$S_{patch-text}=\hat P_{clip}\hat T_{clip}^T.$$

优点是类别可以在测试时更换；风险是patch特征经过全局token交互后可能空间不变、边界粗或混入上下文。[[WeCLIP_paper_notes]] 用冻结CLIP多层特征训练解码器，说明“冻结”并不妨碍下游头解释其密集表示；[[CorrCLIP_paper_notes]] 则直接修复patch相关结构。

### 15.2 DINO/DINOv2：空间协变和视觉对应强，没有天然文本分类器

DINOv2通过图像级与patch级自监督目标学习视觉表示。其patch特征常用于对象对应、聚类、KNN与空间亲和力，但不能因为维度碰巧与CLIP文本相同就直接点积。常见组合是：

```text
CLIP文本/语义 → 决定“是什么”
DINO patch/affinity → 决定“哪些位置属于一起”
学习映射或受限聚合 → 连接两种空间
```

### 15.3 SAM：类别无关mask结构强，输出不是语义类别

SAM图像编码器产生供提示解码器重复使用的图像嵌入；点、框或mask提示决定要解码哪个区域。SAM特征或mask可用于边界、区域分组、全局关联，但“SAM分出了一个完整区域”不等于它知道该区域是 `cat` 还是 `dog`。

### 15.4 三模型协同

[[Trident_paper_notes]] 将高分辨率子图的CLIP/DINO特征拼回全图，再借SAM关系扩展跨窗口感受野。这类方案首先要解决三件工程问题：预处理坐标一致、特征通道/归一化兼容、窗口重叠区域如何融合。模型名字的并列不等于特征已自动对齐。

## 16. 多尺度、滑动窗口与高分辨率特征

### 16.1 多尺度输入

对尺度集合 $\mathcal S$，每个尺度产生响应：

$$M^{(s)}=h(E(\operatorname{Resize}(I,s))).$$

正确融合顺序是：

```text
缩放/翻转图像
  → 分别提取特征和预测
  → 恢复各自patch网格
  → 逆翻转、逆缩放到原图坐标
  → 再融合响应
```

在特征层直接平均不同尺度token，需要它们已被投影到同一通道并插值到同一网格；否则应在类别响应层融合。

### 16.2 滑动窗口

高分辨率图像可切成重叠窗口 $I_k$。每个窗口的预测回填全图时，用权重窗 $W_k$ 减少接缝：

$$
M(x)=\frac{\sum_kW_k(x)M_k(x)}{\sum_kW_k(x)+\varepsilon}.
$$

中心权重大、边缘权重小通常比硬拼接稳定。但窗口编码器只能看到局部上下文；[[Trident_paper_notes]] 指出的正是“分别分割后再拼接”可能丢失跨窗口关系。若任务包含超大物体，还应保留低分辨率全图分支。

### 16.3 缓存编码结果

SAM图像嵌入、冻结CLIP/DINO特征或教师输出可离线缓存，但缓存键必须包含：图像版本、resize/crop参数、骨干权重、层号、归一化方式与dtype。任何数据增强改变坐标后，旧的密集缓存都不能直接复用。

## 17. 冻结、`eval()`、`no_grad()`与`detach()`

四个概念经常被混为一谈：

| 操作 | 参数更新 | 是否建计算图 | BN/Dropout状态 | 典型用途 |
|---|---:|---:|---|---|
| `requires_grad_(False)` | 否 | 仍可能为输入/下游建图 | 不改变 | 冻结骨干参数 |
| `model.eval()` | 取决于参数 | 是 | 切换到推理行为 | 固定BN统计、关闭Dropout |
| `torch.no_grad()` | 否 | 否 | 不改变 | 纯特征缓存/推理 |
| `tensor.detach()` | 上游无梯度 | 下游可重新建图 | 不涉及 | 固定教师目标/伪标签 |

需要Grad-CAM时，即使骨干参数不更新，也必须保留目标分数到目标特征的梯度路径，不能把整段前向包在 `no_grad()` 中。只训练解码器时则可对冻结骨干使用 `no_grad()` 节省激活显存，但要确认没有关系损失或输入梯度依赖这些激活。

## 18. 一个稳健的实现骨架

```python
def extract_vit_features(model, images, num_prefix_tokens, grid_hw):
    outputs = model.forward_features(images)

    # 不同仓库应优先使用明确命名的patch接口
    if isinstance(outputs, dict) and "x_norm_patchtokens" in outputs:
        patches = outputs["x_norm_patchtokens"]       # [B, N, D]
        global_feat = outputs.get("x_norm_clstoken")
    else:
        tokens = outputs                               # [B, P+N, D]
        global_feat = (
            tokens[:, 0] if num_prefix_tokens > 0
            else tokens.mean(dim=1)
        )
        patches = tokens[:, num_prefix_tokens:]

    hp, wp = grid_hw
    if patches.shape[1] != hp * wp:
        raise ValueError("patch数量与真实网格不一致")

    dense = patches.reshape(patches.shape[0], hp, wp, -1)
    dense = dense.permute(0, 3, 1, 2).contiguous()    # [B, D, Hp, Wp]
    return global_feat, patches, dense
```

这段代码故意不使用 $\sqrt N$，并把prefix token数量作为显式输入。真实工程还应从patch embedding输出或预处理元数据得到 `grid_hw`，而不是仅由原图尺寸猜测。

多层hook要注意及时移除，避免重复注册：

```python
captured = {}
handles = []

def save_output(name):
    def hook(_, __, output):
        captured[name] = output
    return hook

for name, block in selected_blocks.items():
    handles.append(block.register_forward_hook(save_output(name)))

_ = model(images)
for handle in handles:
    handle.remove()
```

若只用于分析，可在hook内 `output.detach()`；若中间层还要参与训练loss，则不能detach。

## 19. 复杂度与显存账本

### 19.1 稠密token数量

输入边长扩大2倍、patch大小不变时，token数量约扩大4倍。全局自注意力矩阵大小为 $N^2$，因此注意力权重内存可近似扩大16倍。以float32单头为例：

| patch网格 | $N$ | 一个 $N\times N$ 矩阵 |
|---:|---:|---:|
| $20\times20$ | 400 | 约0.61 MB |
| $40\times40$ | 1600 | 约9.77 MB |
| $64\times64$ | 4096 | 64 MB |

真实训练还要乘batch、层数、头数、梯度与临时张量。仅报告输入分辨率不足以说明成本，应同时记录token数与是否保存注意力。

### 19.2 多层特征存储

保存 $L$ 层 `[B,N,D]` 特征的激活量约为 $B\sum_lN_lD_l$。冻结骨干并detach中间层可降低反向图开销；若每层都参与可训练融合，显存会显著增加。先用消融筛选层，再保留必要hook，通常比无差别保存所有block更可靠。

## 20. 失败现象到原因的定位表

| 现象 | 更可能的原因 | 优先检查 |
|---|---|---|
| 类别正确但目标只激活一小块 | 深层分类特征偏判别区域 | 中间层、多尺度、空间传播 |
| 目标完整但边界糊 | 网格过粗或关系过平滑 | patch大小、OS、浅层/区域特征 |
| 相邻不同类互相污染 | patch相关性含类间聚合 | Q/K/V来源、亲和范围、SAM/超像素约束 |
| 小目标完全消失 | 下采样或窗口分辨率不足 | 浅层网格、输入尺度、proposal recall |
| DINO与文本点积结果随机 | 两者不在同一语义空间 | 映射、原型或检索桥梁 |
| 多骨干融合后性能下降 | 坐标、尺度、归一化或通道不一致 | 逐分支可视化与单分支消融 |
| 训练显存异常高 | 冻结骨干仍保留计算图/所有层hook | `no_grad`范围、detach与hook数量 |
| 滑窗出现棋盘接缝 | 窗口边缘硬拼接、上下文不足 | 重叠权重、全图分支、先拼特征后预测 |

## 21. 阅读论文时的特征提取记录模板

```text
视觉编码器：
预训练目标与权重版本：
输入预处理/分辨率/patch大小：
取出的层与block内部位置：
特殊token数量：
原始张量形状：
二维网格与原图坐标恢复方式：
是否L2/LayerNorm/投影：
骨干冻结、eval、no_grad状态：
该特征的直接消费者：
是否参与损失、梯度流向哪里：
多尺度/滑窗/缓存策略：
推理时是否仍需要该骨干：
```

只要这份记录完整，后续的[[Multi_Level_Fusion]]、[[Pooling_and_Region_Aggregation]]、[[Cross_Modal_Alignment]]和[[Attention_and_Affinity_Refinement]]就能基于明确接口讨论，而不是停留在“用了某基础模型”的描述。

## 22. 当前整理结论

特征提取的核心不是“选一个更强骨干”，而是：

$$\boxed{\text{选择后续算子真正需要的粒度、层级、空间结构和训练状态}}.$$
