# INSID3：从冻结特征到参考条件分割的源码学习

论文：[INSID3: Training-Free In-Context Segmentation with DINOv3](https://openaccess.thecvf.com/content/CVPR2026/html/Cuttano_INSID3_Training-Free_In-Context_Segmentation_with_DINOv3_CVPR_2026_paper.html) · [官方项目](https://visinf.github.io/INSID3/) · [官方仓库](https://github.com/visinf/INSID3)。配套阅读：[[INSID3_Translation]]、[[INSID3_paper_notes]]。

本文实际阅读官方实现，重点是方法计算，不是复现指南。以下源码链接固定到同一阅读版本；原样节选标为“官方源码节选”，重新组织的教学代码标为“可复用改写”。结论来自论文与静态源码核对，没有运行 DINOv3 权重或复现分割指标。

主要入口：

| 文件与函数 | 阅读目的 |
|---|---|
| [`models/insid3.py::predict_mask`](https://github.com/visinf/INSID3/blob/0c165a10cf52ab91f335883d06260de86854adbe/models/insid3.py#L122-L189) | 分割方法主线，以及原始/去偏特征的分工 |
| [`_build_positional_basis` / `_debias_features`](https://github.com/visinf/INSID3/blob/0c165a10cf52ab91f335883d06260de86854adbe/models/insid3.py#L245-L271) | 固定位置子空间与正交补投影 |
| [`_locate_candidates` / `_seed_and_aggregate`](https://github.com/visinf/INSID3/blob/0c165a10cf52ab91f335883d06260de86854adbe/models/insid3.py#L275-L364) | 参考匹配、种子选择与区域合并 |
| [`utils/clustering.py`](https://github.com/visinf/INSID3/blob/0c165a10cf52ab91f335883d06260de86854adbe/utils/clustering.py) | 凝聚聚类与归一化簇原型 |
| [`utils/data.py::downsample_mask`](https://github.com/visinf/INSID3/blob/0c165a10cf52ab91f335883d06260de86854adbe/utils/data.py#L102-L112) | 标注如何对应到 patch 网格 |
| [`utils/refinement.py`](https://github.com/visinf/INSID3/blob/0c165a10cf52ab91f335883d06260de86854adbe/utils/refinement.py) | 二值输出的插值与可选边界细化 |

## 1. 先把完整方法在代码里走一遍

一句话理解：先把目标图切成一组由视觉特征决定的小区域，再用参考图及其 mask 判断“我要的是哪些区域”。参考与目标之间的比较容易受绝对位置干扰，所以用去偏特征；目标自身的分组又需要空间结构，所以保留原始特征。两套特征来自同一次骨干前向，没有第二个视觉模型。

```text
构建模型时：低语义内容图 → 冻结 DINOv3 → 通道子空间基 basis

每次分割：S 张参考图 + 1 张目标图 → 冻结 DINOv3 → 原始单位特征
                                                   ├─ 目标凝聚聚类 → labels
                                                   └─ 去偏并再次归一化
                                                        ├─ 参考 mask 池化 → 参考原型
                                                        └─ 双向信息筛选 → 候选 patch

候选 patch + labels → 候选簇
候选簇的去偏原型与参考原型比较 → seed cluster
跨图语义分数 × 同图种子相似度 × 候选覆盖率 → 整簇保留/舍弃
patch 二值 mask → 双线性插值 → 可选 CRF → 输出二值 mask
```

上图概括数据依赖。实际 `predict_mask` 先算候选，再执行聚类；若候选为空，就直接返回空 mask，省掉聚类。论文叙述先聚类后匹配，两者依赖不冲突。

`set_reference` 累积参考图，`set_target` 设置目标图，`segment` 调用 `predict_mask`，结束后清空输入状态。这里的“in-context”意味着通过当前参考样例改变分割目标，**没有把参考 mask 编码为 Transformer 提示 token，也没有通过梯度把新类别学进模型**。mask 只参与特征筛选、原型池化和最近邻标签检查。

## 2. 张量约定：shot 维不是 Transformer 序列维

本文用 $S$ 表示参考张数，$C$ 表示特征通道数，$h,w$ 表示 patch 网格，$N=hw$，$K$ 表示当前目标图的簇数。源码 `_extract_features` 另有 $B,T$，分别表示 episode 维和每个 episode 内图像数；当前分割入口实际使用 $B=1,T=S+1$，不是任意 batch 的完整实现。

官方源码节选（[`predict_mask`](https://github.com/visinf/INSID3/blob/0c165a10cf52ab91f335883d06260de86854adbe/models/insid3.py#L135-L146)）：

```python
S = ref_images.shape[0]
imgs = torch.cat([ref_images, tgt_image], dim=0).unsqueeze(0)
fmaps = self._extract_features(imgs)
fmaps_norm = F.normalize(fmaps, p=2, dim=2)
_, _, C, h, w = fmaps_norm.shape
ref_masks = ref_masks.unsqueeze(1)
feat_tgt = fmaps_norm[:, S]
```

参考 `[S,3,H,W]` 和目标 `[1,3,H,W]` 沿图像维拼接，再变为 `[1,S+1,3,H,W]`。`_extract_features` 用 `einops.rearrange('b t c h w -> (b t) c h w')` 暂时合并前两维，把所有图像作为普通图像 batch 送入编码器，输出后恢复 `[B,T,C,h,w]`。**参考和目标不会因此在骨干 Attention 内跨图交互**；真正的跨图关系由后面的 `einsum` 单独计算。

编码器调用是 `get_intermediate_layers(x, n=1, reshape=True)[0]`，即最后一个 block 的密集表示。额外核对了 [DINOv3 的接口实现](https://github.com/facebookresearch/dinov3/blob/6876159a11b4df116f30f667f8c9888617df0751/dinov3/models/vision_transformer.py#L269-L316)：默认 `norm=True`；先处理归一化，再去掉 CLS 和 storage tokens，最后按 patch 网格 reshape。这里无需手动写 `tokens[:,1:]`，否则容易误留特殊 token。DINOv3 的当前接口版本只用于核对该 API，不能视为作者实验锁定的依赖版本。

以代码默认 Large/16、输入 `1024×1024` 为例，$C=1024$、$h=w=64$。单参考得到 `[1,2,1024,64,64]`；`F.normalize(..., dim=2)` 对每个位置的 1024 维特征做 L2 归一化，不是沿图像维或空间维归一化。取 `[:, S]` 后，目标是 `[1,1024,64,64]`。这些是接口推导的形状，不是本机实测日志。

[`models/__init__.py::build_insid3`](https://github.com/visinf/INSID3/blob/0c165a10cf52ab91f335883d06260de86854adbe/models/__init__.py#L29-L59) 将参数 `requires_grad=False`，`predict_mask`、特征提取和子空间估计又处于 `no_grad`；官方推理脚本调用 `model.eval()`。前两者控制梯度，后者控制模块的推理行为。这里没有分割 loss、优化器或可训练 decoder；SVD 和聚类是推理计算，不是训练新模型。

## 3. 位置去偏：从特征里扣掉一组固定方向

### 3.1 基底来自“低语义图”，不是人工指定坐标轴

直觉是给骨干一张几乎没有对象语义的图，观察其不同 patch 仍然怎样变化。这些变化提供一种位置相关子空间的近似，再从真实图像特征中减去落在该子空间内的分量。

官方源码节选（[`_build_positional_basis`](https://github.com/visinf/INSID3/blob/0c165a10cf52ab91f335883d06260de86854adbe/models/insid3.py#L249-L261)）：

```python
noise_img = normalize(
    torch.zeros(1, 3, self.image_size, self.image_size),
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],
).to(device)
noise_fmaps = self.encoder.to(device).get_intermediate_layers(
    noise_img, n=1, reshape=True
)[0]
noise_fmaps = F.normalize(noise_fmaps, p=2, dim=1)
E = einops.rearrange(noise_fmaps, 'b c h w -> c (b h w)')
E = E - E.mean(dim=1, keepdim=True)
U, _, _ = torch.linalg.svd(E, full_matrices=False)
return U[:, :self.svd_components].contiguous()
```

关键顺序不能只记成“SVD 去偏”：

1. `torch.zeros` 构造的是全零 RGB 图；经 ImageNet 标准化后，每个颜色通道是空间上恒定的负值。变量叫 `noise_img`，**实际并没有采样高斯噪声**。论文式 (3) 则写 $I_{noise}\sim\mathcal N(0,1)$。
2. 提取结果 `[1,C,h,w]`，沿通道归一化后排成 `E:[C,N]`。一列是一个 patch，一行是某个通道在全部位置的响应。
3. `E.mean(dim=1, keepdim=True)` 是 `[C,1]`，广播到 `[C,N]`，每个通道减去自己的空间均值。减的是用于估计基底的图；真实输入特征后面并没有执行同样的中心化。
4. `torch.linalg.svd(E, full_matrices=False)` 返回经济型分解。若 `E:[1024,4096]`，`U:[1024,1024]`，前 $s=500$ 列组成 `basis:[1024,500]`。这 500 列是通道空间的正交方向，不是保留 500 个 patch。

论文把特征写成 `[N,C]`，所以取右奇异向量；代码转置成 `[C,N]`，取左奇异向量 `U`。**左右名称不同是布局造成的，不是方法矛盾**。真正的差异是输入图、预先 L2 归一化和中心化步骤。

### 3.2 正交补投影到底乘了什么

官方源码节选（[`_debias_features`](https://github.com/visinf/INSID3/blob/0c165a10cf52ab91f335883d06260de86854adbe/models/insid3.py#L265-L271)）：

```python
B, T, C, H, W = fmaps_norm.shape
X = fmaps_norm.reshape(B * T, C, H * W)
basis = self.positional_basis.to(device=X.device, dtype=X.dtype)
P_perp = torch.eye(C, device=X.device, dtype=X.dtype) - basis @ basis.T
X_deb = torch.matmul(P_perp.unsqueeze(0), X).reshape(B, T, C, H, W)
return F.normalize(X_deb, p=2, dim=2)
```

这里的局部变量 `H,W` 是特征图尺寸。`basis @ basis.T` 将 `[C,s] @ [s,C]` 变成 `[C,C]`，消去基向量索引 $s$，得到位置子空间的投影矩阵。`P_perp` 是其正交补。`unsqueeze(0)` 产生 `[1,C,C]`，`matmul` 的 batch 广播使同一矩阵作用于全部 $BT$ 张图的 `[C,N]` 特征。

$$
X_{deb}=(I_C-B_pB_p^\top)X=X-B_p(B_p^\top X).
$$

**公式解释：** 此处 $B_p:[C,s]$ 是位置基，与 batch 符号区分；$X:[BT,C,N]$ 是原始单位特征。第一种写法用 `[C,C]` 左乘每张图，消去输入通道索引，输出仍为 `[BT,C,N]`；第二种先算每个 patch 在 $s$ 个基方向上的系数 `[BT,s,N]`，再重建位置分量 `[BT,C,N]` 并逐元素相减。两式在实数精度下等价，输出每列是该 patch 的残余特征。

若某特征为 $(3,4)$，位置基是 $(1,0)^\top$，投影后得到 $(0,4)$；随后 L2 归一化得到 $(0,1)$。因此这是**改变比较所用的方向信息**，不是简单把余弦分数乘小。投影会减小特征模长，所以代码重新归一化，否则残余能量大的 patch 会获得额外点积优势。

去偏分支只供跨图匹配；`feat_tgt` 仍保留投影前的原始单位特征，用于聚类和种子到其他簇的同图相似度。不要为了代码简洁把所有消费者都替换为去偏特征。

### 3.3 可迁移写法：避免显式建立方阵

```python
# 可复用改写：不是作者原函数；输入采用 [B,N,C] 布局。
# basis 的列必须正交，且来自与 x 相同的编码器/取层/预处理。
def remove_subspace(x, basis, eps=1e-6):
    x32 = torch.nn.functional.normalize(x.float(), dim=-1, eps=eps)
    b32 = basis.to(device=x.device, dtype=torch.float32)  # [C,s]
    coefficients = x32 @ b32                           # [B,N,s]
    residual = x32 - coefficients @ b32.T              # [B,N,C]
    return torch.nn.functional.normalize(residual, dim=-1, eps=eps)
```

这份改写用 $O(NCs)$ 的低秩乘法代替显式 `[C,C]` 投影的主体计算；当 $s\ll C$ 时尤其有价值，不保证所有尺寸都更快。FP32 是教学实现的数值选择，官方代码把基转换为特征 dtype。若投影后向量接近零，归一化并不能恢复已丢失的语义，应单独检查残余模长；$s$ 也不能不加判断地设置到全部通道。

适合迁移到“某种已知干扰占据稳定线性子空间”的密集对应或检索任务。它不保证只去掉位置而绝不损失语义；换骨干、输入尺度或取层后，需要重新估计并验证基底。算子归纳见 [[Feature_Extraction#23. INSID3案例：同一骨干保留结构空间与去偏匹配空间]]。

## 4. 参考原型：先把每张图的前景压成一个向量

### 4.1 mask 怎么对上 patch

参考标注先进入模型输入尺寸，再通过 `downsample_mask` 缩至 `[h,w]`。该函数先双线性插值二值浮点 mask 后用 `>0.5` 判定；若变为空，再尝试最近邻；仍为空则取原前景坐标均值映射到 patch 网格，强制保留一个位置。

这说明“小部件完全被缩没”会影响方法本身：后面连参考原型都无法构造。最后的质心回退假定**原始标注非空，且采用代码的方形等尺度输入**；不应把它当成任意空 mask 或非等比尺寸的通用处理。迁移时应明确检查空标注，并分别用高、宽缩放比例映射坐标。质心也未必仍在非凸物体内部，这是保留最小提示信号的启发式。

### 4.2 布尔索引与多 shot 的权重

官方源码节选（[`predict_mask`](https://github.com/visinf/INSID3/blob/0c165a10cf52ab91f335883d06260de86854adbe/models/insid3.py#L151-L161)）：

```python
ref_prototypes = []
for s in range(S):
    mask_s = downsample_mask(ref_masks[s:s+1], h, w)
    fg = feat_refs_deb[0, s, :, mask_s]
    if fg.shape[1] > 0:
        ref_prototypes.append(fg.mean(dim=1))
ref_prototype = F.normalize(
    torch.stack(ref_prototypes).mean(dim=0), p=2, dim=0
).unsqueeze(1)
```

`feat_refs_deb[0,s]` 是 `[C,h,w]`，`mask_s:[h,w]` 同时索引两个空间轴，结果 `fg:[C,n_s]`，其中 $n_s$ 是本参考前景 patch 数。`mean(dim=1)` 消去前景位置，输出 `[C]`。`torch.stack` 得到 `[S_valid,C]`，再沿参考图维平均，最后沿通道归一化，增加末维得到 `[C,1]`。

这里是“每张参考图先平均，然后参考图之间等权平均”。例如参考 A 有 100 个前景 patch，B 只有 10 个；直接合并所有 patch 会让 A 权重为 $100/110$，源码两张图各占 $1/2$。不过每张图的均值在融合前没有单独归一化，所以内部方向更一致、均值模长更大的参考图，在最终方向合成中仍会有更大影响。“图级等权”不等于“各图单位方向等权”。

这个原型代表当前 mask 定义的对象/部件/指定实例，不是固定数据集类别权重。下一次更换参考样例，就重新构造，不需要更新神经网络。

## 5. 候选定位：先问像不像，再问最像参考图里的哪里

### 5.1 全 patch 对全 patch 的相关性

官方源码关键行（[`predict_mask`](https://github.com/visinf/INSID3/blob/0c165a10cf52ab91f335883d06260de86854adbe/models/insid3.py#L163-L169)）：

```python
sim_m = torch.einsum('bchw,bcxy->bhwxy', feat_ref_m, feat_tgt_deb)
```

输入分别是参考 `[1,C,h_r,w_r]`、目标 `[1,C,h_t,w_t]`。`c` 不出现在输出，下标求和消去通道；其余四个空间索引全部保留，输出 `[1,h_r,w_r,h_t,w_t]`。每个元素表示某个参考 patch 与某个目标 patch 的余弦相似度，数值来自二者已归一化的去偏特征。

这是全对全相似度，不是 Transformer 的多头 Q/K/V Attention：没有可训练 Q/K/V 投影、没有 softmax、没有 value 加权汇聚。后面只使用最大相似位置的索引。若网格都是 `64×64`，单参考相关矩阵含 $4096^2$ 个元素；FP32 的纯矩阵存储约 64 MiB，实际内存还包括其他张量。

### 5.2 backward：目标 patch 用参考 mask 给自己投票

官方源码节选（[`_locate_candidates`](https://github.com/visinf/INSID3/blob/0c165a10cf52ab91f335883d06260de86854adbe/models/insid3.py#L294-L307)）：

```python
for m, sim_m in enumerate(sim_maps):
    sim0 = sim_m[0]
    Hs, Ws = sim0.shape[:2]
    sim_t_to_r = sim0.permute(2, 3, 0, 1)
    best_idx = sim_t_to_r.reshape(h, w, -1).argmax(dim=2)
    rows = best_idx // Ws
    cols = best_idx % Ws
    ref_mask_m = downsample_mask(ref_masks[m:m+1], Hs, Ws).squeeze(0)
    votes += ref_mask_m[rows, cols].to(torch.int32)
majority_thresh = math.ceil(k / 2)
backward_mask = votes >= majority_thresh
return forward_mask & backward_mask
```

`permute` 只换轴，把目标位置放前面，参考位置放末尾；随后 `reshape(h,w,-1)` 将参考坐标展平为长度 $N_r$ 的检索库。`argmax(dim=2)` 为每个目标位置找参考图中的最佳 patch，结果 `[h_t,w_t]`。若参考宽度为 4、最佳编号为 6，则行 `6//4=1`、列 `6%4=2`。`ref_mask_m[rows,cols]` 是逐目标位置的成对高级索引，返回 `[h_t,w_t]`，而不是两组坐标的笛卡尔积。

例如参考 mask 只标了狗耳朵：目标“狗头”虽可能接近耳朵原型，但其最近参考 patch 若落在未标注的狗脸上，就会被排除。**负证据来自参考图全部 mask 外 patch 的竞争，不是另建一个背景均值原型**。mask 外只表示“当前提示不选”，可能仍是同一只狗的其他部位；不能等同于通用背景类或 unknown 类。

多参考时分别得到 0/1 票，使用 `ceil(S/2)`。$S=5$ 至少三票；$S=2$ 一票即可，因此源码所称 majority 包含偶数时的平票，不是严格过半。标准单参考设定就是一次最近邻 mask 检查。

### 5.3 forward 并非“正向最近邻”，也不是循环一致性

源码的 `sim_fwd = einsum('bchw,cd->bhw', ...)` 用 `[1,C,h,w]` 与参考原型 `[C,1]` 得到 `[1,h,w]`。表达式中的 `c,d` 都被消去，`d` 只有长度 1；可等价理解为每个目标特征与单个向量点积。若扩成多个类别原型 `[C,L]`，不能原封不动沿用这个 `einsum`，否则它会把 $L$ 个原型分数相加而不是保留类别轴。

`forward_mask = sim_fwd > 0`；若没有正响应，改为严格大于整图 90% 分位数的位置。它是原型正响应过滤，不是 reference→target 最近邻。最终 `candidate_mask = forward_mask & backward_mask`。因此这是两种互补判据的交集，**不是先去目标再返回参考原位置的 cycle consistency**。论文式 (6)–(8)主要定义 backward 候选，正向过滤及其回退属于此公开实现的额外细节。

### 5.4 可迁移写法：只保留近邻索引时可以分块

```python
# 可复用改写：单参考，二维展平接口；只实现 backward mask。
# query:[Nt,C]、reference:[Nr,C] 已沿 C 归一化；mask:[Nr] 为 bool。
def backward_mask_chunked(query, reference, mask, chunk_size=512):
    pieces = []
    for q in query.split(chunk_size, dim=0):
        similarity = q @ reference.T       # [chunk,Nr]
        nearest = similarity.argmax(dim=1) # [chunk]
        pieces.append(mask[nearest])
    return torch.cat(pieces, dim=0)         # [Nt]
```

它改变内存组织而非检索定义：每个查询块仍与全部参考 patch 比较，峰值相似度矩阵降为 `[chunk,Nr]`。前提是 query 的展平顺序与后续目标 mask 恢复一致；多参考仍需外层投票，forward 过滤也需另算。这是单独可复用的检索写法，不代表官方已使用分块。

## 6. 凝聚聚类：区域来自原始特征，不来自候选 mask

直觉是先让特征本身决定“哪些 patch 可作为一个单元”，之后只对整个区域作决定。候选 mask 不负责生成区域边界，也不会把一个簇切成若干候选碎片。

官方源码节选（[`agglomerative_clustering`](https://github.com/visinf/INSID3/blob/0c165a10cf52ab91f335883d06260de86854adbe/utils/clustering.py#L18-L25)）：

```python
S = (X @ X.T).clamp(-1, 1)
D = (1.0 - S).cpu().numpy()
ac = AgglomerativeClustering(
    n_clusters=None, metric='precomputed',
    linkage='average', distance_threshold=float(1.0 - tau),
)
labels = ac.fit_predict(D)
return torch.from_numpy(labels).long().to(X.device)
```

`X` 来自原始 `feat_tgt[0].reshape(C,-1).permute(1,0)`，形状 `[N,C]`。`X @ X.T` 消去通道，得到 `[N,N]` 的 patch 余弦相似度。`clamp(-1,1)` 控制浮点误差造成的越界，`1-S` 转成距离。随后计算移到 CPU，交给 scikit-learn；返回整数簇编号 `[N]`，再移回原设备并恢复 `[h,w]`。

`n_clusters=None` 意味着不预设 $K$；`distance_threshold=1-tau` 控制停止合并的尺度。$\tau$ 越高，允许距离越低，通常得到越细的簇。`linkage='average'` 使用两簇所有跨簇点对的平均距离，因此不是“只要存在一个相似点就合并”，也不是“要求所有点对都相似”。

没有传 `connectivity` 网格图，也没有加入坐标距离。论文说区域通常空间连贯，是对 DINOv3 表示的利用；**源码并不保证一个簇在二维图像上连通**。如果自己的任务要求每个实例必须连通，需要额外约束或连通组件处理，不能把这个性质归给 API。

## 7. 从候选簇到 seed：簇编号索引与原型归一化

### 7.1 一个分组，两个空间，各自池化

`compute_cluster_prototypes` 对每个 `k` 取 `X[labels==k]`，沿 patch 维平均得到 `[C]`，再沿通道归一化，最终拼成 `[K,C]`。同一个目标分组先作用于去偏特征，得到跨图匹配原型；之后又作用于原始特征，得到同图结构原型。

这一步要记住：**cluster labels 可以共享，特征值不能混用**。它们均是当前图像的区域统计量，不是额外可学习参数。

官方函数对空簇有随机向量回退。但标准凝聚聚类输出连续编号、每个簇非空，主路径通常不会用到它；迁移到含 ignore 或稀疏编号的分组时，应显式处理空簇和编号映射，不能用随机原型掩盖接口错误。

### 7.2 种子选的是哪个编号

官方源码节选（[`_seed_and_aggregate`](https://github.com/visinf/INSID3/blob/0c165a10cf52ab91f335883d06260de86854adbe/models/insid3.py#L324-L340)）：

```python
matched_mask = candidate_mask & (cluster_labels >= 0)
matched_ids, n_pixels = cluster_labels[matched_mask].unique(return_counts=True)
all_areas = cluster_labels[cluster_labels >= 0].unique(return_counts=True)[1]
per_cluster = torch.zeros(K, device=cluster_labels.device)
per_cluster[matched_ids] = n_pixels.float()
area_weights = per_cluster / all_areas
protos_matched = cluster_protos[matched_ids]
cross_sim_matched = (ref_prototype.T @ protos_matched.T).squeeze(0)
seed_idx = int(torch.argmax(cross_sim_matched).item())
seed_id = matched_ids[seed_idx].item()
```

假设 label map 含 0–7 号簇，候选只覆盖 2、5、7 号，`matched_ids:[3]` 内容为 `[2,5,7]`。`cluster_protos[matched_ids]` 取出 `[3,C]`，参考原型转置为 `[1,C]`，矩阵乘法得到 `[1,3]`。如果最大分数在第 1 项，`seed_idx=1`，但真正 seed label 是 `matched_ids[1]=5`。将“候选列表位置”误当“全图簇编号”会选错区域，这是索引迁移时最常见的陷阱之一。

`unique(return_counts=True)` 同时计算每个候选簇的候选 patch 数。`all_areas` 是每个簇的总 patch 数，因此 `area_weights` 表示**候选在簇内部的覆盖比例**，不是簇占整图比例。代码依赖聚类编号连续且排序一致，使 `all_areas[k]` 对应第 $k$ 簇。

## 8. 聚合最容易被看漏的地方：三种分数不完全同义

### 8.1 同图项：原始空间中与种子簇的方向一致性

官方关键行（[`_seed_and_aggregate`](https://github.com/visinf/INSID3/blob/0c165a10cf52ab91f335883d06260de86854adbe/models/insid3.py#L343-L347)）：

```python
orig_protos = compute_cluster_prototypes(
    feat_tgt_flat, cluster_labels.view(-1), K
)
intra_sim = torch.einsum('c,kc->k', orig_protos[seed_id], orig_protos)
```

seed `[C]` 与全部簇 `[K,C]` 点积，消去通道，返回 `[K]`。由于原型已单位化，这是同图原始空间的余弦相似度；种子与自身的分数在非零原型和理想精度下为 1。它没有进行像素空间传播，也不要求候选与种子边界相邻。

### 8.2 跨图项：选 seed 用方向，合并用 patch 分数均值

官方源码节选（[`_seed_and_aggregate`](https://github.com/visinf/INSID3/blob/0c165a10cf52ab91f335883d06260de86854adbe/models/insid3.py#L349-L364)）：

```python
fg_sim = torch.einsum('bchw,cd->bhw', feat_tgt_deb, ref_prototype).squeeze(0)
cross_sim = torch.empty(K, device=fg_sim.device, dtype=fg_sim.dtype)
for k in range(K):
    idx = (cluster_labels == k)
    cross_sim[k] = fg_sim[idx].mean() if idx.any() else 0.0
combined = cross_sim * intra_sim
area_weights[seed_id] = 1.0
combined *= area_weights
final_mask = torch.zeros(h, w, dtype=torch.bool, device=cluster_labels.device)
valid = cluster_labels >= 0
final_mask[valid] = combined[cluster_labels[valid]] > self.merge_threshold
```

这里 `fg_sim:[h,w]` 是每个目标 patch 与参考原型的相似度。`fg_sim[idx].mean()` 对**整个簇**的 patch 分数求均值，不只是簇内被 candidate mask 命中的部分。记去偏单位 patch 为 $\tilde f_i$，簇均值为 $\mu_k$，参考单位原型为 $\hat p_r$：

$$
\mu_k=\frac1{|G_k|}\sum_{i\in G_k}\tilde f_i,
\qquad a_k^{seed}=\left\langle\frac{\mu_k}{\|\mu_k\|_2},\hat p_r\right\rangle,
\qquad a_k^{merge}=\frac1{|G_k|}\sum_{i\in G_k}\langle\tilde f_i,\hat p_r\rangle
=\|\mu_k\|_2\,a_k^{seed}.
$$

**公式解释：** $\tilde f_i,\mu_k,\hat p_r$ 都是 `[C]` 向量；簇内均值消去 patch 维，点积消去通道维，两个 $a_k$ 均为标量。等式假定 $\mu_k$ 非零，先逐 patch 点积再平均，等价于未单位化均值与参考点积。若输入 patch 均为单位向量，$\|\mu_k\|_2\le1$，会反映方向一致程度；选择 seed 时重新归一化去除了这部分模长，合并时却保留。

例子：参考方向 $(1,0)$，簇 A 的两个单位特征为 $(1,0),(0,1)$。均值 $(0.5,0.5)$，seed 分数约 0.707，merge 分数 0.5。簇 B 的所有 patch 都是 $(0.6,0.8)$，两个分数均为 0.6。因此选 seed 时 A 胜过 B，而仅比较 merge 跨图项时 B 胜过 A。不能把源码两处 `cross_sim` 当作完全相同的变量。

这也是值得迁移的统计区别：单位化均值强调方向，未单位化均值保留内部一致性。在自己的模型里两者都合理，但阈值意义不同，不能一边替换一边保持旧阈值并声称等价。对应归纳见 [[Prototype_Construction#30. INSID3案例：均值原型的方向与一致性应分开使用]]。

### 8.3 面积项不是“小簇奖励”，而是候选覆盖惩罚

$$
w_k=\frac{|G_k\cap C_{candidate}|}{|G_k|},\qquad
w_{seed}\leftarrow1,\qquad
q_k=a_k^{merge}\,a_k^{intra}\,w_k.
$$

**公式解释：** 集合计数对 patch 维归约，得到每簇标量；整组 $w,a^{merge},a^{intra},q$ 均是 `[K]`。除法产生候选覆盖率，乘法逐簇进行，没有通道矩阵乘法。$q_k$ 是第 $k$ 簇最终阈值分数；seed 的覆盖率被设为 1，免去该项惩罚。

例如一个簇共 20 个 patch，只命中 4 个候选，$w=0.2$；跨图项 0.8、同图项 0.9，最终为 0.144，不能超过默认 0.2。若全部 20 个都命中，则得到 0.72。说明“簇里偶然有一个匹配点”虽足以进入候选集合，还不一定足以把整簇加进最终 mask。

最后 `combined[cluster_labels[valid]]` 是把 `[K]` 的簇分数查表恢复到每个 patch：label 为 5 的所有位置都取 `combined[5]`。先比较 `>merge_threshold` 再写入 `[h,w]` 布尔 mask，所以一个簇整体接受或拒绝，不会只留下候选部分。

### 8.4 论文式 (13)–(14) 与源码应怎样对应

论文强调跨图项乘同图项，并显式把种子并入最终 mask。公开实现额外加入覆盖权重；选种子与合并使用的跨图分数又有上述归一化差异。源码只令 `area_weights[seed_id]=1`，**并未写 `final_mask |= cluster_labels==seed_id`**。seed 的同图项虽约为 1，但跨图均值仍可能不超过阈值，因而仍可能被丢掉。论文用 $\ge\alpha$，源码用严格 `>self.merge_threshold`。

这些是静态代码层面可确认的行为差异，不能据此猜测作者实验实际用了哪个历史版本。笔记保留论文解释，同时明确当前公开实现的执行语义。

可复用的区域分数回投可以很短：

```python
# 可复用改写：labels:[H,W]，scores:[K]；假定合法编号是 0..K-1。
valid = (labels >= 0) & (labels < scores.numel())
prediction = torch.zeros_like(labels, dtype=torch.bool)
prediction[valid] = scores[labels[valid]] > threshold
# 若自己的定义要求 seed 必留，可显式增加下面一行；官方源码没有这一行。
# prediction |= labels == seed_id
```

## 9. 输出为何还要插值，以及 CRF 真正改了哪里

### 9.1 二值 mask 的插值

[`upsample_mask`](https://github.com/visinf/INSID3/blob/0c165a10cf52ab91f335883d06260de86854adbe/utils/refinement.py#L7-L12) 将 `[h,w]` 布尔 mask 增加 batch、channel 轴，转成 `[1,1,h,w]` 浮点数，使用双线性插值，最后 `>0.5` 恢复二值输出。不是转置卷积，也没有可学习参数。

这里插值的是已经阈值化的区域 mask，不是所有簇的连续 `combined` 得分图。它平滑的是二值边界的几何过渡，不会重新利用被阈值舍弃的语义置信度。离散多类 label map 则不应直接这样双线性插值，否则会产生无意义的混合类别编号。

### 9.2 可选 CRF 只允许改边界带

CRF 即 Conditional Random Field（条件随机场）。代码在独立分辨率上先计算一个窄边界带，再调用外部 GPU CRF：

1. 用圆形 0/1 卷积核与初始 mask 做 `conv2d`；窗口内计数 `>0` 对应膨胀，计数等于核面积对应腐蚀。
2. `band = dil ^ ero` 得到边界带；腐蚀内核确定前景，膨胀外部确定背景。
3. 前景概率初始化为内核 0.95、外部 0.05、带内 0.5，构造成两通道输入；原图反标准化到 `[0,1]` 供颜色关系使用。
4. CRF 返回后，只有边界带采用细化结果，内核强制前景、外部强制背景；因此不能靠此步骤发现远离初始预测的新目标。

一个容易误读的数值细节：源码构造 `logit_fg = torch.logit(prob_fg)` 后使用 `[-logit_fg, logit_fg]`，不是 `[log(1-p),log(p)]`。若按二分类 softmax 理解，其前景概率为 `sigmoid(2*logit(p))`，内核 $p=0.95$ 对应约 0.9972，而不是仍然 0.95。这段作为能量/分数初始化应按实际代码理解；本文没有展开或运行外部 CRF 求解器。

论文实验说明采用 CRF；公开构建函数默认 `mask_refiner='bilinear'`，需显式选 CRF 才走细化分支。`resize_to_orig_size=True` 时恢复原图尺寸；官方命令行构建入口则传 `False`，输出到模型输入尺寸。这些选择影响输出几何，但不改变前面区域匹配算法。

## 10. 方法与实现核对表

| 环节 | 论文表达 | 当前公开实现 |
|---|---|---|
| 位置基输入 | 式 (3) 标准高斯噪声 | 全零 RGB 图经标准化 |
| SVD | `[N,C]` 的右奇异向量 | 归一化并中心化的 `[C,N]` 的左奇异向量；左右差异来自转置 |
| 特征空间 | 去偏用于跨图，原始用于同图 | 与此分工一致，投影前后都显式 L2 归一化 |
| 候选定位 | backward NN 落在参考前景 | 还与原型正响应相交；空 forward 用 90% 分位数回退 |
| 参考样例 | 主文以 one-shot 为核心 | 支持多参考，图级均值与至少半数投票 |
| 区域聚类 | 凝聚聚类、阈值控制粒度 | average linkage、余弦距离、CPU 执行；没有网格连通约束 |
| 跨图簇分数 | 用簇原型与参考比较 | seed 用单位簇均值；merge 用逐 patch 分数均值 |
| 合并 | 跨图 × 同图，seed 显式保留 | 多候选覆盖率项；seed 只豁免覆盖惩罚，仍过阈值 |
| 阈值 | 式 (14) 为 $\ge\alpha$ | 严格 `>merge_threshold` |
| 边界恢复 | 实验使用 CRF | API 默认仅双线性插值，CRF 可选 |

作为独立的去偏迁移验证，仓库还有 [`predict_keypoint`](https://github.com/visinf/INSID3/blob/0c165a10cf52ab91f335883d06260de86854adbe/models/insid3.py#L193-L228)。它在参考关键点处取描述子，与目标所有 patch 点积得到 `[1,L,h,w]`，再调用 [`utils/keypoints.py`](https://github.com/visinf/INSID3/blob/0c165a10cf52ab91f335883d06260de86854adbe/utils/keypoints.py) 的 Gaussian-suppressed soft-argmax 输出坐标。主文将评估匹配概括为最大相似位置，而此实现用最大值位置为高斯中心、乘相似图、温度 softmax 后求坐标期望；不要把它写成纯 `argmax`。这条辅助路径不参与分割候选定位。

## 11. 读完最值得带走的实现思路

| 学会的操作 | 源码入口 | 迁移时要保留的条件 |
|---|---|---|
| 一次骨干输出派生两个任务空间 | `_extract_features`、`_debias_features` | 各消费者的语义要明确；不要全局替换原始表示 |
| 用固定正交基消去干扰分量 | `_build_positional_basis` | 基底必须与骨干、预处理、层和分辨率匹配；检查残余能量 |
| mask 内池化与参考图级聚合 | `predict_mask` 的 reference prototype | 区分 patch 等权、图像等权、均值后归一化的不同意义 |
| 用参考 mask 外样本提供负证据 | `_locate_candidates` | 最近邻必须在完整参考图里检索，不能预先只保留前景 |
| 分组和语义选择解耦 | `agglomerative_clustering`、`_seed_and_aggregate` | 分组编号与所有空间中的 patch 顺序一致 |
| 用区域索引完成分数回投 | `combined[cluster_labels[valid]]` | label 是查表索引，不能把候选列表位置当全局编号 |
| 区分方向分数与区域一致性 | `compute_cluster_prototypes`、`fg_sim[idx].mean()` | 归一化位置会改变排序和阈值，不只是数值稳定技巧 |

对我自己写模型最有价值的是：先明确每个张量为谁服务，再选计算。INSID3 没有复杂 decoder，关键恰好集中在几个很普通的操作——投影、归一化、布尔索引、最近邻、均值、查表——以及这些操作的先后顺序和所处特征空间。相关通用知识已补入 [[Feature_Extraction]] 与 [[Prototype_Construction]]；完整方法动机仍回到 [[INSID3_paper_notes]]。
