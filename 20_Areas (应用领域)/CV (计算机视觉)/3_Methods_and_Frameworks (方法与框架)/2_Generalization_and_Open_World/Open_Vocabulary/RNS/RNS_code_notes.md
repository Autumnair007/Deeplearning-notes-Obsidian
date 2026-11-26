---
type: code-note
tags: [semantic-segmentation, open-vocabulary-segmentation, retrieval, frozen-backbone, pytorch]
model: RNS
status: in-progress
---

# RNS 源码精读｜为每张测试图挑选训练样本，再拟合一个分割头

假设测试图中有一辆靠在树旁的自行车。车轮所在的 patch 可以从支持库中找到自行车的例子，树叶所在的 patch 又能找到树的例子。RNS 把两边找到的带标签向量放进同一个小训练集，再为入选的类补上融合了类名和视觉外观的向量。这个训练集只服务于当前图像，学出一个线性分类器就能给整张图做分割。下一张图使用同一个支持库，重新检索、重新拟合。相关背景见 [[RNS_paper_notes]] 和 [[RNS_Translation]]。[CVPR 论文](https://openaccess.thecvf.com/content/CVPR2026/html/Aravanis_Retrieve_and_Segment_Are_a_Few_Examples_Enough_to_Bridge_CVPR_2026_paper.html) · [第一作者官方仓库](https://github.com/TilemahosAravanis/Retrieve-and-Segment)

| 阅读入口 | 在 Method 中跟踪什么 |
|---|---|
| [main_eval.py](https://github.com/TilemahosAravanis/Retrieve-and-Segment/blob/f5adb7ec2a02368a314f8c38a1664e932839fdd3/main_eval.py#L32-L334) | `create_support_set → slide_feat_extract → gt_mask_pooling` 怎样得到单图类别向量，`set_class_embeddings` 怎样做全库汇总和图文融合 |
| [dinov3_txt_ovss.py](https://github.com/TilemahosAravanis/Retrieve-and-Segment/blob/f5adb7ec2a02368a314f8c38a1664e932839fdd3/models/dinov3_txt_ovss/dinov3_txt_ovss.py#L24-L62) · [clip_ovss.py](https://github.com/TilemahosAravanis/Retrieve-and-Segment/blob/f5adb7ec2a02368a314f8c38a1664e932839fdd3/models/clip_ovss/clip_ovss.py#L17-L34) · [maskclip.py](https://github.com/TilemahosAravanis/Retrieve-and-Segment/blob/f5adb7ec2a02368a314f8c38a1664e932839fdd3/models/maskclip/maskclip.py#L50-L96) | 两种冻结骨干怎样交出可以与文本比较的稠密特征 |
| [mmseg_utils/apis/test.py](https://github.com/TilemahosAravanis/Retrieve-and-Segment/blob/f5adb7ec2a02368a314f8c38a1664e932839fdd3/mmseg_utils/apis/test.py#L10-L110) | `slide_inference` 在特征提取与分类之间插入可选的区域池化，在分类之后恢复像素图 |
| [core/rns.py](https://github.com/TilemahosAravanis/Retrieve-and-Segment/blob/f5adb7ec2a02368a314f8c38a1664e932839fdd3/core/rns.py#L14-L261) · [core/TTA_models/Linear.py](https://github.com/TilemahosAravanis/Retrieve-and-Segment/blob/f5adb7ec2a02368a314f8c38a1664e932839fdd3/core/TTA_models/Linear.py#L5-L44) | `propose_masks`、`retrieve_similar_feats`、`RNS_Inference` 和真正被优化的 `linear_layer` |

## 1. 先把长期保存的量和每张图重算的量分开

```text
支持图像 + 像素真值 → 冻结视觉骨干 → 每张图中每个出现类一个向量 → 视觉支持库
                                                    └→ 按类汇总 ┐
类别名 + 提示模板 → 冻结文本编码器 → 文本类别向量 ────────────────┴→ 各比例融合特征

测试图像 → 冻结视觉骨干 → patch 向量（可用 SAM 转成区域向量）→ 查询向量 Q
Q 逐行近邻检索 → 支持行号并集 → 检索到的视觉样本 + 对应类的融合样本
Q 遇到缺视觉支持的类 → 按文本预测池化 Q → 融合后生成软标签样本
三组样本 + 本图类别权重 → 优化一个新线性层 → 对 Q 分类 → 恢复像素类别
```

下文用 $D$ 表示图文特征维，$C$ 表示数据集类别数，$S$ 表示入库支持图数量。单张测试图的 patch 网格为 $h\times w$，$N=hw$。把真正送入检索和分类的行数记作 $J$，走 patch 分支时 $J=N$，走区域分支时 $J=R$（区域数）。后面按每类一个文本向量解释，便于看清主路径。作者在适配阶段多次使用 `squeeze(0)`，每次调用针对一张测试图。

## 2. 两种骨干交出同一个接口

**OpenCLIP。** 普通 CLIP 的整图向量无法区分车轮和树叶。这里的 MaskCLIP 路径把倒数第二个 Transformer block 的输出留在 forward hook 中，取出后送入最后一层的 `ln_1` 与打包的 QKV 线性投影。`extract_v` 拆出 Q、K、V 中的 V 分支，保留输出投影、输入残差以及逐 token 的 MLP 残差。最后这个分支没有用 QK 在空间位置间重新加权，之前各层提供的上下文仍保留。去掉 CLS、排回网格后，`MaskClipHead.proj` 用预训练的 `visual.proj` 构成 $1\times1$ 卷积，对每个网格位置施加相同的视觉到文本空间映射。以仓库的 ViT-B/16 为例，`[1,768,h,w] → [1,512,h,w]`。这是 RNS 借用的稠密表征提取方式，后续按图检索、训练的逻辑在 `core/rns.py`。[特征提取与 V 分支](https://github.com/TilemahosAravanis/Retrieve-and-Segment/blob/f5adb7ec2a02368a314f8c38a1664e932839fdd3/models/maskclip/maskclip.py#L43-L96) · [投影与文本编码](https://github.com/TilemahosAravanis/Retrieve-and-Segment/blob/f5adb7ec2a02368a314f8c38a1664e932839fdd3/models/maskclip/maskclip.py#L145-L212)

**DINOv3.txt。** `get_vlm_features` 解包 `encode_image_with_patch_tokens` 的三个返回值，选第三个 `backbone_patch_tokens`，经 `permute(0,2,1).reshape(B,-1,h,w)` 从 `[B,N,D]` 回到 `[B,D,h,w]`。文本端对每个类名套用多种模板，按通道切取文本输出的后半段，先逐条做 L2 归一化，再对模板求均值并归一化，得到与 patch 匹配的方向。ViT-L/16 路径的稠密特征维为 1024。替换骨干时，要核对模型原生的图文对齐关系以及选取哪组 token。光让最后一维相等并不能建立语义对应。[视觉/文本输出与调用](https://github.com/TilemahosAravanis/Retrieve-and-Segment/blob/f5adb7ec2a02368a314f8c38a1664e932839fdd3/models/dinov3_txt_ovss/dinov3_txt_ovss.py#L36-L62) · [patch 网格恢复](https://github.com/TilemahosAravanis/Retrieve-and-Segment/blob/f5adb7ec2a02368a314f8c38a1664e932839fdd3/models/dinov3_txt_ovss/dinov3_txt_ovss.py#L93-L136) · [维度配置](https://github.com/TilemahosAravanis/Retrieve-and-Segment/blob/f5adb7ec2a02368a314f8c38a1664e932839fdd3/configs/dinov3txt.yaml#L14-L28)

两条路径都支持滑窗提特征。对每个 crop 先沿通道归一化，把局部网格放回全图对应位置并累加。`count_mat` 的形状是 `[1,1,h,w]`，记录每个位置被多少窗口覆盖。除法时这个单通道计数会广播给 $D$ 个通道。重叠区由多个视野的特征平均得到，这一步解决了不同窗口输出怎样拼成同一套空间坐标的问题。[支持图滑窗](https://github.com/TilemahosAravanis/Retrieve-and-Segment/blob/f5adb7ec2a02368a314f8c38a1664e932839fdd3/main_eval.py#L32-L70) · [测试图滑窗](https://github.com/TilemahosAravanis/Retrieve-and-Segment/blob/f5adb7ec2a02368a314f8c38a1664e932839fdd3/mmseg_utils/apis/test.py#L10-L55)

## 3. 用像素数给 patch 分配权重，一张图的一个类只占一行

`gt_mask_pooling` 的输入有两个分辨率。`feats` 是 `[1,D,N]` 的 patch 特征，`ann` 里每个原图像素有一个类别 ID。沿高、宽各做一次 `unfold`，窗口和步长都设为 patch 边长 $P$，每个 patch 所含的 $P^2$ 个像素就成为一行。以下为两个核心计算的源码节选。[main_eval.py · gt_mask_pooling](https://github.com/TilemahosAravanis/Retrieve-and-Segment/blob/f5adb7ec2a02368a314f8c38a1664e932839fdd3/main_eval.py#L72-L90)

```python
ann_downsampled = ann.unfold(0, patch_size, patch_size).unfold(1, patch_size, patch_size)
ann_downsampled = ann_downsampled.reshape(dim_patches_h*dim_patches_w, -1)
# 中间按类统计各 patch 的像素数，拼成 masks
instance_embeddings = torch.matmul(feats, masks.T).squeeze()
```

设 $n_{cj}$ 为第 $j$ 个 patch 内类别 $c$ 的像素数，源码的 `masks[c,j]` 等于 $n_{cj}/\sum_t n_{ct}$。分母统计**整张图中属于当前类的像素**。例如同一类在两个 patch 内分别占 3 和 1 个像素，那么对该类的池化权重是 $0.75,0.25$，即使第二个 patch 的多数像素属于别的类，它也贡献四分之一。`[1,D,N] @ [N,C] → [1,D,C]`，再转置得到 `[C,D]`，对应式（2）。同一张图里的两辆自行车若同属一个语义类，就被池化为同一行，实例身份在此消失。

对一个视图聚合后的行向量 $u$，发布代码还做了 `vlm_feats += vlm_feats / vlm_feats.norm(dim=-1, keepdim=True)`。有效行变成 $u+u/\|u\|$。单视图下最终再归一化，方向不变，多视图下则先累加这些向量，再做行归一化。因此，读多视图分支时要留意每个视图的相对权重。每个支持图始终占 $C$ 行，缺失类别会因计数为零产生无效值，建库流程用 `+inf` 占位。这让全库可以按 `[S,C,D] ↔ [S*C,D]` 重排，也使 `row_id % C` 恰好等于类别 ID。对空库行的处理见第 5 节。[逐图汇总与存储](https://github.com/TilemahosAravanis/Retrieve-and-Segment/blob/f5adb7ec2a02368a314f8c38a1664e932839fdd3/main_eval.py#L207-L276)

## 4. 同一类别为何同时要保留许多向量

一张支持图中的“自行车”有自己的姿态和背景。RNS 保留这个具体实例来源的向量 $v_c^i$ 供局部检索，也计算跨图的类别代表 $v_c$。`set_class_embeddings` 复制库、把 `+inf` 替成零，恢复 `[S,C,D]` 后沿 $S$ 求均值，再沿 $D$ 单位化。对至少出现一次的类，如果有效方向和不为零，用总图数 $S$ 作分母或只用实际出现次数作分母，最后的单位方向相同。随后把这个方向与归一化文本 $t_c$ 混合。[main_eval.py · set_class_embeddings](https://github.com/TilemahosAravanis/Retrieve-and-Segment/blob/f5adb7ec2a02368a314f8c38a1664e932839fdd3/main_eval.py#L279-L334)

```python
feats = lambda_ * visual + (1 - lambda_) * textual
feats = feats / feats.norm(dim=1, keepdim=True)
```

式（4）的 $\lambda$ 乘**文本**，源码 `lambda_` 乘**视觉**。源码系数 `[0.9, 0.8, 0.6, 0.4, 0.2, 0.0]` 对应论文记号中的文本权重 `[0.1, 0.2, 0.4, 0.6, 0.8, 1.0]`。因此 `lambda_=0` 给出纯文本样本。举例设视觉方向为 $(1,0)$，文本方向为 $(0,1)$，源码系数 $0.8$ 产生 $(0.8,0.2)$，归一化后约为 $(0.970,0.243)$。这些方向共享同一个真类标签，小头会同时看到“更像这批图像”和“更像类名”的样本。

`torch.cat(mixed, dim=0)` 按比例成块堆叠。每个块都是一轮完整类别序列，所以 `mixed_labels.repeat(比例数)` 才和特征逐行对应，例如两类、三个比例对应 `[0,1,0,1,0,1]`。`repeat_interleave` 在这里会得到不同的行序。这个融合库在支持集确定后可供多个查询复用，每张图另行挑选需要的类别。论文的增量场景在这里也容易理解：新增支持图只需提取其图内类别向量并更新受影响的类统计，发布入口 `create_support_set` 展示的是整批建库。[特征堆叠](https://github.com/TilemahosAravanis/Retrieve-and-Segment/blob/f5adb7ec2a02368a314f8c38a1664e932839fdd3/main_eval.py#L306-L334) · [标签匹配](https://github.com/TilemahosAravanis/Retrieve-and-Segment/blob/f5adb7ec2a02368a314f8c38a1664e932839fdd3/core/rns.py#L186-L194)

## 5. 每个局部查询取近邻，再取整张图的并集

把测试特征排成 `Q=[J,D]`。作者用 FAISS 的 `GpuIndexFlatIP` 在 GPU 上做精确内积检索。当键和查询都是有限的单位向量时，内积和余弦相似度的排序相同。`.contiguous()` 把转置过的张量整理成适合外部库读取的连续内存，不会改变行列所表达的对象。程序在每次调用里建索引、加入支持行并执行 `search`。[core/rns.py · 检索](https://github.com/TilemahosAravanis/Retrieve-and-Segment/blob/f5adb7ec2a02368a314f8c38a1664e932839fdd3/core/rns.py#L81-L89)

```python
unique_supp_indices, inverse_supp_indices = torch.unique(supp_indices, sorted=True, return_inverse=True)
unique_supp_indices = unique_supp_indices[unique_supp_indices != -1]
support_feats = vlm.support_vlm_feats[unique_supp_indices].cuda()
support_labels = unique_supp_indices % vlm.num_classes
```

`supp_indices` 是 `[J,k]`。假设有 3 类，两块 patch 分别找到 `[3,4]`、`[4,8]`，并集就是 `[3,4,8]`。这三行的类 ID 是 `[0,1,2]`，编号 4 只入选一次。距离决定是否入选。返回的 `supp_sims` 及反向映射 `inverse_supp_indices` 均没有参与后面的损失。若另一个库条目也属于类别 1，它仍可作为独立样本与第 4 行同时入选。于是**同一行去重，同一类可以保留多个外观**。这一机制对应式（6）。[core/rns.py · 行索引并集](https://github.com/TilemahosAravanis/Retrieve-and-Segment/blob/f5adb7ec2a02368a314f8c38a1664e932839fdd3/core/rns.py#L123-L148)

设最终留下 $U$ 行，`support_labels.unique()` 就是本图检索命中的类别集合 $\mathcal C_r$。用 `torch.isin` 从整张融合表选出这些类别的全部比例，再用 `torch.isfinite(...).all(dim=1)` 排除无效的融合行。全名目里的其他类别不会自动各得六个训练样本。注意代码把含 `+inf` 占位的原支持表直接交给 FAISS，之后明确删除 `-1` 行号和人为丢弃的类别，但没有对检索到的原视觉行做相同的有限性检查。自己实现时可在建索引前压紧有效行并随行存储 `class_id`，这样不会依赖检索后端对非有限值的行为。压紧后必须通过保存的标签解码。[检索与融合类筛选](https://github.com/TilemahosAravanis/Retrieve-and-Segment/blob/f5adb7ec2a02368a314f8c38a1664e932839fdd3/core/rns.py#L125-L146) · [融合行有效性检查](https://github.com/TilemahosAravanis/Retrieve-and-Segment/blob/f5adb7ec2a02368a314f8c38a1664e932839fdd3/core/rns.py#L184-L198) · [[Retrieval_and_Memory]]

## 6. 缺少某类的支持图时，用当前图构造软监督

设某些类别有名字，但没有像素标注例子。实现中 `drop_classes` 明确记录在实验里移除视觉支持的类别。先用 `Q @ textual_embeddings.T` 得到 `[J,C]` 的局部图文相似度，沿类别轴做 `softmax(100 * scores)`，再 `argmax(dim=1)` 给每个 patch 或区域指派一个临时类。这第一次 softmax 的结果只用于取 `argmax`。在精确运算下，乘正数不会改变分组。下方给融合向量生成教师分布时，代码会重新计算另一组相似度；第二次 softmax 中的系数才控制软目标的尖锐程度。[core/rns.py · 文本预测](https://github.com/TilemahosAravanis/Retrieve-and-Segment/blob/f5adb7ec2a02368a314f8c38a1664e932839fdd3/core/rns.py#L116-L123)

对于每个确实被预测到的类，`Q[test_labels == i]` 取出属于它的一组行。沿 `dim=0` 求均值再单位化，得到一条当前图的视觉代表，正好实现了式（11）将同类位置等权汇总的操作。随后只保留同时位于 `drop_classes` 的类：一个仅仅没被本图近邻检索命中的类，并不会因此触发伪监督。对每条保留下来的视觉代表，与该类文本按六个比例逐一混合、归一化。若 3 个缺视觉类中只有 1 个在图上被预测到，就只追加 6 个软监督样本。[伪实例池化、筛选与融合](https://github.com/TilemahosAravanis/Retrieve-and-Segment/blob/f5adb7ec2a02368a314f8c38a1664e932839fdd3/core/rns.py#L150-L184)

```python
text_pseudo_aug_labels = torch.matmul(text_pseudo_aug_feats, vlm.textual_embeddings.T)
text_pseudo_aug_labels = F.softmax(text_pseudo_aug_labels * 100, dim=1)
```

最初的硬预测回答“哪些位置可以放在一起”。最终监督来自**每个融合向量重新与全部类名比较**的结果。假设某条由“自行车”组构造的向量在“自行车”和“摩托车”上的软目标为 $0.55,0.45$，它教给分类器的是这种不确定性。把第一个 `argmax` 直接拿来做这个阶段的交叉熵，会抹去论文式（12）刻意保留的信息。完整视觉支持时 `drop_classes` 为空，这组样本自然为零行。[软目标的构造](https://github.com/TilemahosAravanis/Retrieve-and-Segment/blob/f5adb7ec2a02368a314f8c38a1664e932839fdd3/core/rns.py#L196-L210)

反过来，部分**类名**缺失时，`set_class_embeddings` 用有文本的类向量的均值替代缺失行，再做单位化。`RNS_Inference` 在 `drop_text_fraction > 0` 时对所有类使用均匀的全图权重。完全无文本的论文描述让式（4）退到纯视觉端点。发布实现的 `use_text=False` 会直接构造空的融合表，在所有类均有视觉支持的设定下靠检索到的视觉行训练，这一分支仍计算文本匹配和初始分组，随后因为没有缺视觉支持的类而筛掉全部伪样本。实际参与优化的是检索到的视觉行。将 `drop_text_fraction` 直接设为 `1.0` 会使 `textual[keep_mask]` 为空，对它求均值将产生 NaN；源码用 `use_text=False` 表达无文本基线。[文本替代与融合开关](https://github.com/TilemahosAravanis/Retrieve-and-Segment/blob/f5adb7ec2a02368a314f8c38a1664e932839fdd3/main_eval.py#L279-L334) · [图像级权重开关](https://github.com/TilemahosAravanis/Retrieve-and-Segment/blob/f5adb7ec2a02368a314f8c38a1664e932839fdd3/core/rns.py#L98-L115)

## 7. 为一张测试图训练什么，谁给梯度

先计算本图中的“类相关性”。`glob_emb` 是 $J$ 个归一化查询向量的均值，再除以其 L2 范数。它和各类文本向量点积、乘 `class_score_Temp`、沿类别 `softmax`，得到 `[C]` 的 `global_textual_class_scores`，对应式（8）和（9）。这是作用于**整张图**的类别权重，与上一节每个样本自己的软目标有不同的统计范围。启用 SAM 时 $J=R$，每个区域在这次全局平均里占一票，大区不会因面积更大自动加权。[全图类别先验](https://github.com/TilemahosAravanis/Retrieve-and-Segment/blob/f5adb7ec2a02368a314f8c38a1664e932839fdd3/core/rns.py#L91-L115)

三组样本按固定顺序拼成 `support_feats=[M,D]` 和 `support_labels=[M,C]`。设入选视觉行 $U$ 条、有真类的融合行 $F$ 条、伪标签融合行 $A$ 条，则 $M=U+F+A$。前两组暂存为 one-hot，在交叉熵处再还原整数类标。第三组保存完整的教师概率分布。

| 训练集切片 | 向量来自哪里 | 监督及用途 |
|---|---|---|
| `[:U]` | 已检索的逐图视觉类向量 | 来自支持图真值的硬标签，用 CE 学习该图所需的视觉边界 |
| `[U:U+F]` | 命中类的全库视觉代表与文本的插值 | 构造时已知的硬标签，用 CE 让同一类的不同模态比例指向同一输出 |
| `[U+F:]` | 测试图伪分组代表与文本的插值 | 教师软标签，用 KL 蒸馏给当前头，仅产生于缺视觉支持且被预测到的类 |

`support_labels_weights = global_textual_class_scores[support_labels.argmax(dim=1)]` 从类别权重中为每个样本查出一个数，再对**所有入选样本**归一化到总和为 1。真实样本取真类的权重。伪样本取软目标最大项的类权重，而论文式（12）写的是构造该向量的类别 $c$ 的权重。在这里讨论的每类一个单位文本向量的情形下，两种查法通常一致。因为每个被分到类 $c$ 的查询都满足 $q^\top(t_c-t_d)\geq 0$（对任意其他类 $d$），组内取均值并单位化仍保留这个不等式。再加入同类文本方向也会保留它，因为 $t_c^\top(t_c-t_d)=1-t_c^\top t_d\geq 0$。所以软标签即使接近均匀，构造类仍是最大项之一；完全并列、数值精度或额外的类别映射需单独考虑。若权重为 $[0.8,0.2]$，样本的最大标签为 $[0,0,1]$，那么归一化后的样本权重为 $[0.8,0.8,0.2]/1.8$，所以每个样本的份额同时取决于本图入选了哪些其他样本。[支持集组装与权重](https://github.com/TilemahosAravanis/Retrieve-and-Segment/blob/f5adb7ec2a02368a314f8c38a1664e932839fdd3/core/rns.py#L196-L218)

```python
output = model(X[HW:])
output_log = F.log_softmax(output, dim=-1)
# 前两组取 CE，最后一组取 KL，并逐样本乘图像相关性权重
loss = torch.cat([loss_retrieved, vlm.beta_mixed * loss_mixed, vlm.beta_pseudo * loss_pseudo]).mean()
```

`X` 的前 $J$ 行是查询，后 $M$ 行是训练样本，变量名 `HW` 在区域分支中实际也是 $J$。训练使用 `X[HW:]`，预测才使用 `X[:HW]`。代码对前两组计算 `CrossEntropyLoss(reduction='none')`，每行得到一个数。最后一组调用 `KLDivLoss`，输入是模型的 **log** 概率，目标是教师的普通概率，方向为 $D_{\mathrm{KL}}(p_{\text{teacher}}\|p_{\text{head}})$。`weight.unsqueeze(1)` 从 `[A]` 变成 `[A,1]`，沿类别广播，`sum(dim=-1)` 后每个软样本也只剩一个数。最后才乘各分支系数、拼接并取均值。对固定的本图训练集，这相当于在论文加权求和外多了共同的缩放。不能把三组各自取 mean 再相加，否则会改变组间相对贡献。[实际损失及优化](https://github.com/TilemahosAravanis/Retrieve-and-Segment/blob/f5adb7ec2a02368a314f8c38a1664e932839fdd3/core/rns.py#L220-L259)

每张图新建 `linear_layer(D,C,dropout=0.0,init_weights=None,bias=True)`，它只有随机初始化的 $W\in\mathbb R^{C\times D}$ 和从零开始的 $b\in\mathbb R^C$。训练时返回 `X @ W.T + b` 的 logits，评估模式才沿类轴 softmax。外面的验证循环在 `torch.no_grad()` 中，内部用 `torch.enable_grad()` 只对小头的训练过程打开梯度，AdamW 接收的也只有这个头的参数。进入下一图时会重建分类头及优化器。因此可复用的实现方式是：冻结昂贵的特征提取，先明确哪些向量提供监督，再围绕固定的小矩阵做优化。若将调用端改成 `torch.inference_mode()`，需要相应调整梯度作用域，单靠内层 `enable_grad` 不能恢复 inference tensor 所需的反向传播行为。[线性头](https://github.com/TilemahosAravanis/Retrieve-and-Segment/blob/f5adb7ec2a02368a314f8c38a1664e932839fdd3/core/TTA_models/Linear.py#L17-L44) · [梯度作用域与优化器](https://github.com/TilemahosAravanis/Retrieve-and-Segment/blob/f5adb7ec2a02368a314f8c38a1664e932839fdd3/core/rns.py#L218-L259) · [外层验证](https://github.com/TilemahosAravanis/Retrieve-and-Segment/blob/f5adb7ec2a02368a314f8c38a1664e932839fdd3/mmseg_utils/apis/test.py#L186-L195)

## 8. 同一个分类头怎样给每个像素交答案

**按 patch 分类。** $J=hw$，`[J,C] → [1,C,h,w]` 后双线性插值到补齐后的图像尺寸，裁掉右边与下边的填充，再按原图尺寸恢复。空间上的边缘精度受 patch 网格约束。[mmseg_utils/apis/test.py · patch 回填](https://github.com/TilemahosAravanis/Retrieve-and-Segment/blob/f5adb7ec2a02368a314f8c38a1664e932839fdd3/mmseg_utils/apis/test.py#L68-L94)

**按区域分类。** SAM 2.1 为当前测试图给出类别无关的候选掩码。`propose_masks` 将候选按面积从小到大依次写入一幅像素 ID 图，重叠部分以后写的区域覆盖先写的区域。仍未被覆盖的像素保留初始 ID 0（若存在），这个区域照样交给分类器判断。代码随后只对实际留下的 ID 连续重编号。这里的 0 是区域表的行号，不能直接理解为最终语义类别“背景”。[core/rns.py · 区域划分与重编号](https://github.com/TilemahosAravanis/Retrieve-and-Segment/blob/f5adb7ec2a02368a314f8c38a1664e932839fdd3/core/rns.py#L35-L64)

得到 `region_id[H,W]` 后，沿用第 3 节的像素计数思路，只是把真类 $c$ 换成区域 $r$。取 $R$ 个区域在 $N$ 个 patch 中的计数并按区域归一化，得到 $A\in\mathbb R^{R\times N}$。计算 `[D,N] @ [N,R] → [D,R]`，再单位化，便得到式（13）的区域表征。用它做检索、算先验、拟合分类头、预测，得到 `output[R,C]`。现在 `output[proposed_masks]` 用二维整型索引一次性查表：像素 $(y,x)$ 的值若为 7，输出就是 `output[7,:]`。得到 `[H,W,C]`，置换为 `[1,C,H,W]`。一个细长物体可以按掩码边缘得到完整预测，但若同一区域把两类粘在了一起，区域内部便共享同一分布。池化发生在 patch 网格上，回填的边界则来自像素级 ID 图。[区域池化](https://github.com/TilemahosAravanis/Retrieve-and-Segment/blob/f5adb7ec2a02368a314f8c38a1664e932839fdd3/core/rns.py#L62-L79) · [像素查表](https://github.com/TilemahosAravanis/Retrieve-and-Segment/blob/f5adb7ec2a02368a314f8c38a1664e932839fdd3/mmseg_utils/apis/test.py#L55-L92) · [[Pooling_and_Region_Aggregation]]

一个容易忽略的输出细节：线性头 `eval()` 已返回概率，外层 `inference` 又对恢复后的通道做一次 softmax。此处第二次 softmax 保持每个像素的类别排序，所以 `argmax` 标签相同，但会改变概率的数值。如果复用它做阈值筛选或置信度校准，需要先决定读取哪一级输出。[线性头输出](https://github.com/TilemahosAravanis/Retrieve-and-Segment/blob/f5adb7ec2a02368a314f8c38a1664e932839fdd3/core/TTA_models/Linear.py#L34-L44) · [最终预测](https://github.com/TilemahosAravanis/Retrieve-and-Segment/blob/f5adb7ec2a02368a314f8c38a1664e932839fdd3/mmseg_utils/apis/test.py#L97-L110)

## 下次写模型时可直接带走什么

- **让检索直接决定训练集。** 有带标签的外部记忆、查询图里又存在多种对象时，用局部查询的近邻并集保留多种外观。移植时把有效支持行、原始行号和类别标签显式绑定，使去重之后仍能找到真标签。[retrieve_similar_feats 与索引选择](https://github.com/TilemahosAravanis/Retrieve-and-Segment/blob/f5adb7ec2a02368a314f8c38a1664e932839fdd3/core/rns.py#L81-L89) · [[Retrieval_and_Memory]]
- **把像素标签和候选区域写成同一套池化接口。** 两者都能提供 `[组数,N]` 的行归一化权重矩阵。替换分组依据只需替换权重的构造，`特征 @ 权重.T` 和后续分类都可以复用。需要原尺寸分割时，一起保留像素级索引图。[gt_mask_pooling](https://github.com/TilemahosAravanis/Retrieve-and-Segment/blob/f5adb7ec2a02368a314f8c38a1664e932839fdd3/main_eval.py#L72-L90) · [propose_masks](https://github.com/TilemahosAravanis/Retrieve-and-Segment/blob/f5adb7ec2a02368a314f8c38a1664e932839fdd3/core/rns.py#L14-L79) · [[Pooling_and_Region_Aggregation]]
- **为单个查询重置小头。** 一组严格对齐的 `features / targets / weights` 可以同时容纳检索真值、跨模态插值和软监督。组合损失前先保持逐样本形状，最后再约减，就能精确控制谁在监督谁、哪批样本对这张图的头更重要。[RNS_Inference](https://github.com/TilemahosAravanis/Retrieve-and-Segment/blob/f5adb7ec2a02368a314f8c38a1664e932839fdd3/core/rns.py#L196-L261)
