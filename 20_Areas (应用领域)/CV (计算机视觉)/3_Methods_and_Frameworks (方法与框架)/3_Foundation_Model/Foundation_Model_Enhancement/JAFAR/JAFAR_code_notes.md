# JAFAR 源码学习笔记

把一张粗特征图铺到像素网格上时，物体边缘附近最难填。同一块 patch 里可能同时有鸟和树枝。JAFAR 让每个输出位置拿着图像里的颜色、纹理和位置，去所有粗 patch 中找来源。粗 patch 用自身对应的图像外观和骨干语义回答“我适合被谁读取”。最后填回去的是这些 patch 原有的特征向量。一幅图只需构造一轮跨网格注意力，就能指定输出高和宽。[NeurIPS 2025 论文](https://papers.nips.cc/paper_files/paper/2025/hash/8eed150084dc3534f01ba63f9b7d32d2-Abstract-Conference.html) · [项目页](https://jafar-upsampler.github.io/) · [官方仓库](https://github.com/PaulCouairon/JAFAR) · [[JAFAR_Translation]] · [[JAFAR_paper_notes]]

| 从哪个问题读起 | 官方源码入口 |
|---|---|
| 图像、粗特征和目标尺寸怎样进入同一次前向？ | [JAFAR.forward / upsample][jafar] · [推理示例][demo] |
| 图像如何带上位置，又怎样变成两个网格？ | [create_coordinate][grid] · [RoPE][rope] · [图像、Q/K 及条件编码器][enc] |
| 骨干特征在哪一步改变空间关系？ | [SFTModulation.forward][sft] · [CrossAttentionBlock / CrossAttention][attn] |
| 什么数据教会模型分配权重？ | [构造两级骨干特征][teacher] · [调用上采样器并反传][train] · [Cosine_MSE][loss] |

## 1. 先走通一次推理

`PretrainedViTWrapper.forward` 向 timm 请求 `NCHW` 空间特征，将前缀 token 单独处理。在演示中，`lr_feats, _ = backbone(image_batch)` 取出 patch 网格，同一张图再作上采样器的引导。`model(image_batch, lr_feats, (Hq, Wq))` 的输出仍在骨干的特征空间，线性分割头或深度头可以在这个较细的网格上继续预测。[骨干的空间输出][vit] · [官方 Colab 中的 run_model_demo][demo]

```text
引导图 I → 图像编码器 → 在引导图网格上加入 RoPE → 共享图像表征 E
E → 查询编码器 → 池化到 (Hq,Wq) → 归一化和局部卷积 → Q
E → 键编码器 → 池化到 (h,w) → 初步 K
骨干特征 F → 归一化和条件编码器 → γ、β → 调制初步 K → K
Q、K → 各头的跨网格注意力 → 头间平均 → A
骨干特征 F 直接作 V → 按 A 加权 → 稠密特征
```

设 `B` 是批量，`Hg,Wg` 是引导图的高宽，`h,w` 是骨干空间高宽，`Hq,Wq` 是调用者要求的输出高宽。`D` 表示 **Q/K 的匹配通道数**，`C` 表示**骨干描述子的通道数**。默认配置中 `D=128`、4 个头。以 DINOv2-S/14 为例，`C=384`。从 `[1,3,448,448]` 得到骨干特征 `[1,384,32,32]`，如请求输出 `112×112`，就需要给 12,544 个输出位置各产生一行长度为 1,024 的权重。每一行生成一个 **384** 维向量。把输出尺寸换成非方形时，两维也是独立传入的。[模型配置][cfg] · [输出还原][jafar]

## 2. 共享图像表征怎样同时表达外观和位置

### 2.1 两个匹配分支先看同一幅图

`make_encoder` 先做一次卷积把输入通道映射到 `D`，接两个同宽的残差块。每块中依次做两次 GroupNorm、SiLU、`1×1` 卷积的组合，再加回输入。模型文件的注释提到图像卷积可以用 `3×3`。构造函数和发布的训练配置给 `image_encoder` 的 `kernel_size` 均为 **1**。`key_encoder`、`query_encoder`、`key_features_encoder` 各自有参数。真正共享的是前端一次算出的 `encoded_image`，后两个图像分支会各自学习“查询要看什么”和“来源要提供哪些匹配线索”。后面还有一次明确用于 Q 的 `3×3` 卷积。[编码器定义][enc] · [残差块][res] · [模型配置][cfg]

### 2.2 RoPE 放在两个分支分开之前

从图像取出的特征先排成 `[B,Hg*Wg,D]`，和同序坐标做位置编码，再还原到二维送往两个分支。`create_coordinate` 在高、宽上分别用 `linspace(0,1,...)` 取点，`meshgrid(indexing="ij")` 生成所有配对。变量虽然叫 `x,y`，返回的第一分量实际沿**行**变，第二分量沿**列**变。比如 `2×3` 网格前四个 token 是 `(0,0), (0,0.5), (0,1), (1,0)`。矩形网格能直接检验高宽顺序。[网格函数][grid] · [展平、编码和还原][jafar]

```python
angle = coords @ self.freqs
return x * angle.cos() + rotate_half(x) * angle.sin()
```

`coords=[1,Hg*Wg,2]` 乘 `self.freqs=[2,D]`，沿两个坐标分量求和，得到 `angle=[1,Hg*Wg,D]`，同一组相位沿 batch 广播。频率是可学习的 `nn.Parameter`，初始化时一半通道使用行坐标，另一半使用列坐标，各轴以多组频率起步。`rotate_half` 把 **整个**末维切成等长的两半，将 `(a,b)` 重排为 `(-b,a)`，再按上面两项逐元素组合。例如一个位置处的输入为 `[a,b,c,d]`，重排结果就是 `[-c,-d,a,b]`。同一对中的前、后半通道初始化时分别取行、列相位，两者通常不同。因此不能直接套用标准 RoPE 的保范数推导，也不能据此断言最终 Q/K 的点积只依赖相对位移。复用这段实现时，应同时保留坐标顺序、频率矩阵布局和通道配对方式。论文说 Q/K 含位置信息，在这里具体指它们都从这张带位置的图派生，没有在池化后给两套 token 各调用一次 RoPE。[位置编码实现][rope]

迁移到不同分辨率时，归一化坐标让“靠近图像右边缘”的数值语义仍然相近。使用同一套函数还能避免高低网格带着各自的整数像素单位去匹配。原函数默认在 `cuda` 建网格，移到 CPU 或不同 GPU 时应传入实际输入的设备。修改 `D` 时还需满足此频率初始化的四等分、GroupNorm 的 8 组以及多头拆分。[建网格的默认设备][grid] · [频率初始化][rope] · [编码器参数][enc]

## 3. 用两个池化尺寸定义“在哪里问”和“向谁问”

```python
queries = self.query_encoder(encoded_image)
queries = F.adaptive_avg_pool2d(queries, output_size=output_size)
queries = self.norm(queries)
```

这段来自 `JAFAR.upsample`。`query_encoder` 在引导图网格上处理 `[B,D,Hg,Wg]`。池化保持批量和通道，直接得到 `[B,D,Hq,Wq]`。`self.norm` 是 8 组、无可学习仿射参数的 GroupNorm。另一条路用 `key_encoder` 处理同一张带位置的图，池化目标取自 `features.shape[-2:]`，输出 `[B,D,h,w]`。前者的网格决定整张结果有多少行查询，后者的每个格子将与一个来源 patch 一一对应。骨干特征本身在此处保留原值。[两条池化分支][jafar]

自适应平均池化按**指定的输出尺寸**决定每个输出格覆盖哪些输入点，各通道独立求平均。若引导图为 `224×224`，训练目标 `32×32`，每个 Q 在池化时恰好汇总 `7×7` 个输入格。若粗骨干网格为 `16×16`，每个初步 K 则汇总 `14×14` 个。非整除时相邻分桶可以重叠，例如长度 5 池化为 3，三段分别取输入下标 `[0,1]`、`[1,2,3]`、`[3,4]`。这种操作聚合局部证据，随后才有语义比较。不能用“对骨干特征池化”代替两路顺序。[两条分支的输入输出][jafar] · [[Pooling_and_Region_Aggregation#28. 用自适应平均池化指定跨尺度匹配的网格]]

论文描述训练时压低 Q 的分辨率，源码始终调用同一个池化函数，尺寸由外部指定。当 `output_size == (Hg,Wg)` 时，每个窗口只有一个元素，池化本身等价于原样传递。若请求大于引导图的网格，API 也能返回指定大小，但更多槽位仍然共享已有采样信息。官方演示在固定图像与粗特征上依次请求 64、128、256、448 和 512 的方形网格。同一套参数可以改变查询数量，全局注意力的算量也随之改变。[官方任意分辨率示例][demo]

## 4. 同一份骨干特征一份用来调 K，一份原样作 V

从图像池化出的初步键可以告诉像素“这里颜色很像”，容易把外观相近的不同物体混起来。JAFAR 将该粗格对应的骨干向量送入条件分支，让它逐位置、逐通道调整 K，随后再和 Q 比较。[构造条件和赋值 V][jafar]

```python
keys = self.sft_key(keys, self.key_features_encoder(F.normalize(features, dim=1)))
```

先在 `features=[B,C,h,w]` 的类别无关特征维 `C` 上做 L2 归一化，每个粗格单独处理。`key_features_encoder` 将这份条件映射为 `[B,D,h,w]`。`SFTModulation` 用两个独立、无偏置的 `1×1` 卷积从条件得到 `gamma` 和 `beta`，形状都与初步键相同。最后一行的 `image` 在这次调用里是初步键张量。[条件编码器][enc] · [SFT 的前向][sft]

```python
gamma = self.gamma(features)
beta = self.beta(features)
return gamma * self.norm(image) + beta  # Spatial modulation
```

对应论文式 (2)，实现额外显式地对初步键做无仿射 GroupNorm。对每个样本，GroupNorm 在每组通道及其全部空间位置上计算统计量。之后的乘加按相同的 `[b,d,y,x]` 取值。若某个位置、某个通道归一化后为 `0.8`，条件给 `gamma=0.5, beta=-0.1`，该键分量成为 `0.3`。相邻粗格可给出另一组系数，卷积本身则在所有位置共享参数。这里没有先给 `gamma` 加 1，初始化也无需假定为恒等调制。[SFTModulation][sft]

`values = features` 则保留了归一化之前的骨干向量。更换骨干时，条件编码器的输入宽度和学到的语义统计需要匹配新骨干。如果已经算好某幅图的关系矩阵，同位置排列的另一个连续信号即使通道数不同，也可以用同一矩阵汇聚。前提是细格从何处取值的关系对新信号同样有用。[K/V 分路][jafar]

## 5. 多头注意力怎样成为可复用的插值核

`CrossAttentionBlock` 先单独给目标网格的 Q 做有填充、步幅 1 的 `3×3` 卷积，空间大小保持不变，每个查询此时能利用邻近查询格的内容。然后 Q/K/V 各自把二维空间按行展平，`batch_first=True` 使接口统一为 `[B,token,channel]`。就 token 数量而言，Q 可以有自己的长度，而 K 和 V 的长度及空间顺序必须一致。三路还需共享 batch 大小，Q/K 的通道均为 D，V 的通道 C 要与构造 MHA 时的 `vdim` 一致。[卷积与展平][attn]

```python
query = self.norm_q(query)
key = self.norm_k(key)

_, attn_scores = self.attention(query, key, self.norm_v(value), average_attn_weights=True)
attn_output = einsum("b i j, b j d -> b i d", attn_scores, value)
```

若把直接返回的第一个 MHA 输出交给下游，会得到另一种模块。两个返回值各自的作用可以沿下面四步读清。

1. **按头比较。** Q、K 先分别沿末维做 RMSNorm，再由 `nn.MultiheadAttention` 内部各自投影、分头。设 `Nq=Hq*Wq`、`Nk=h*w`、头数为 `M`，每个头的 Q 与 K 末维都是 `D/M`。每个头分别计算 `[B,Nq,Nk]` 的点积，除以 `sqrt(D/M)`，沿**来源 patch 轴** softmax。`D=128,M=4` 时每个头的缩放分母是 $\sqrt{32}$。这具体化了论文式 (1) 的单头简写。
2. **先归一化，再在头之间平均。** `average_attn_weights=True` 得到 `A=[B,Nq,Nk]`。`A[b,i,j]` 表示第 `b` 张图的第 `i` 个目标位置给第 `j` 个粗 patch 的权重。先平均各头 logits 再做一次 softmax，一般会给出不同结果。
3. **用原 V 汇聚。** MHA 收到了 `norm_v(value)`，也按常规执行内部投影，但赋给 `_` 的常规输出被舍弃。`attn_scores` 由 Q、K 决定，随后 `einsum` 读取的 `value=[B,Nk,C]` 仍是原骨干向量。下标 `j` 只留在乘积里，因而沿所有来源求和。`i` 和骨干通道 `d` 被保留，结果 `[B,Nq,C]`。从最终特征损失回溯，Q/K 的投影有梯度路径。仅供 MHA 第一个输出使用的 V 投影、V 的 RMSNorm 参数和输出投影没有来自这条损失的有效梯度路径。
4. **还原二维。** `rearrange(..., "b (h w) c -> b c h w", h=output_size[0])` 用请求的**目标高度**拆回位置轴，得到 `[B,C,Hq,Wq]`。整个过程在每个目标位置用相同的空间权重处理全部 `C` 个通道。[多头返回值与汇聚][attn] · [输出还原][jafar]

例如把问题缩成 3 个粗 patch 的二维值，分别为 `(2,0)`、`(0,4)`、`(2,2)`。两头给某个细位置的权重分别为 `[0.8,0.1,0.1]` 与 `[0.4,0.3,0.3]`，平均为 `[0.6,0.2,0.2]`，得到 `(1.6,1.2)`。即使匹配空间有 128 维、值只有 2 维，算术也一样。这里 `dropout=0`，各行权重非负、和为 1，因此每个输出是来源向量的**凸组合**。在粗特征全相等时，任意引导图都应输出那个常量。想保留骨干特征的原始尺度和坐标系，这一性质比“输出形状也叫 C”更具体。[注意力层构造与读出][attn] · [[Attention_and_Affinity_Refinement#6.6 用全局跨网格注意力重采样原始值]]

全局候选让一个目标点可以读到远处相似 patch，也意味着关系图有 `B*Nq*Nk` 个数。`448×448` 查询看 `32×32` 个来源时，即使 `B=1`，**仅头均值矩阵**也有 205,520,896 个元素，以两字节存储约 392 MiB。训练的缓存、逐头中间量及特征图另计。迁移时若按 Q 分块降低推理峰值，应先在**完整网格**依次完成 Q 的自适应池化、GroupNorm、`3×3` 卷积和展平，再切查询序列，每块仍对全部 K 做 softmax。直接把图像裁块会改变归一化统计和边界上下文。[原始计算顺序][jafar] · [Q 卷积与 MHA][attn]

## 6. 没有像素级真值，谁教会模型分配权重？

训练时同一张图经过同一冻结骨干两次。一份较大的图提供目标，另一份缩小的图提供上采样器能读取的值。对每个目标格，只要分给粗 patch 的权重不合适，混出来的向量就会偏离教师。模型通过这个误差学习如何从图像构造 Q、K 与调制参数，训练过程中始终有明确的**特征监督**。[训练所用的骨干输出][teacher] · [JAFAR 参数与训练调用][train]

| 同一张样本中的角色 | DINOv2-S/14 训练例 | 怎样得到 |
|---|---|---|
| 教师目标 `hr_feats` | 图像 `448×448` → 特征 `32×32` | 骨干在较大图上编码，作为损失目标 |
| 待上采样的 `lr_feats` | 如图像 `168×168` → 特征 `12×12` | 缩小原图后**重新运行骨干**，保留每格原始 384 维向量 |
| 图像引导 `image_batch` | 原图双线性缩为 `224×224` | 专供 JAFAR 图像分支使用，对应论文第 4.1 节 |
| JAFAR 的训练输出 | `12×12` → `32×32` | `model(引导图, lr_feats, hr_feats.shape[-2:])` |

源码的随机分支均匀抽缩图系数 `s∈[0.25,0.5]`，将 `sH,sW` 分别取整到最近的 patch 大小倍数，然后 `F.interpolate(..., mode="bilinear")`。论文第 3.2 节用放大倍数 $\delta\in[2,4]$ 表述。二者在此区间互为倒数，**均匀抽取倒数与均匀抽倍率的概率分布不同**，取整还会让实际比例离散。`torch.no_grad()` 包住两次骨干调用。优化器只收集 JAFAR 的参数。推理演示只有一次骨干调用，指定目标网格后直接输出，不需要再用大图生成教师。[构造两级特征][teacher] · [训练参数及前向][train] · [推理入口][demo]

论文式 (3) 含余弦项与 L2 距离。官方 `Cosine_MSE` 具体比较的是每个目标位置的方向和带尺度的数值误差。先把预测、目标均排成 `[B*Hq*Wq,C]`。余弦嵌入损失的标签全部是 `+1`，因此比较每一对原向量的 `1−cos`。对 MSE，`min_val`、`max_val` **只从该位置的目标向量**沿 `C` 取，形状 `[B*Hq*Wq,1]`。同一对上下界广播给预测和目标。展开某位置的平方差，相当于将各通道误差平方都除以 `(max_target−min_target+1e−6)²`，再对位置和通道取平均。预测超出目标范围时不会被截断，两个损失直接相加。因而只用未缩放的 `torch.norm(pred-target)` 无法重现这段训练目标。[Cosine_MSE 实现][loss]

## 7. 下次写模型时怎样借用

| 自己遇到的问题 | 可以抽出来的写法 | 首先核对什么 |
|---|---|---|
| 让一份稠密引导支持不同输出尺寸 | 共享编码图分两路，用两个目标尺寸的自适应平均池化定义查询与来源网格 | 两张图覆盖相同视野，裁剪、翻转和空间轴对应一致 |
| 匹配需要语义，输出还要交给原有特征头 | 将语义输入分作 SFT 条件和原始 V。Q/K 负责学习匹配 | K 与 V 的 patch 顺序逐个一致，条件编码器与骨干通道相容 |
| 想把相同的空间匹配用到多种连续预测上 | 保存头均值权重，分别计算 `A @ V1`、`A @ V2` | 值在同一个粗网格上。更换信号时核实当前任务是否适合同一组空间权重 |
| 缺少像素级特征标签 | 用同一冻结骨干处理原图和缩图，在较细的教师网格监督重建 | 训练中的教师尺寸、引导图尺寸和最终部署尺寸要分别记录 |

回看时可以从 `JAFAR.upsample` 的两次池化出发，一路沿初步 K 追到 `SFTModulation`，另一路从 `values = features` 追到 `einsum`。前一路适合学习怎样按目标大小构造带语义条件的匹配网格，后一路适合学习怎样保留既有特征头所依赖的通道含义。[网格与分路][jafar] · [条件调制][sft] · [原始值读出][attn]

[jafar]: https://github.com/PaulCouairon/JAFAR/blob/25321a2def9b00d06a8a8794ca0a799dcfd18d86/src/upsampler/jafar.py#L68-L103
[enc]: https://github.com/PaulCouairon/JAFAR/blob/25321a2def9b00d06a8a8794ca0a799dcfd18d86/src/upsampler/jafar.py#L13-L66
[res]: https://github.com/PaulCouairon/JAFAR/blob/25321a2def9b00d06a8a8794ca0a799dcfd18d86/src/layers/convolutions.py#L23-L64
[grid]: https://github.com/PaulCouairon/JAFAR/blob/25321a2def9b00d06a8a8794ca0a799dcfd18d86/utils/img.py#L7-L16
[rope]: https://github.com/PaulCouairon/JAFAR/blob/25321a2def9b00d06a8a8794ca0a799dcfd18d86/src/layers/positional_encoding.py#L6-L36
[sft]: https://github.com/PaulCouairon/JAFAR/blob/25321a2def9b00d06a8a8794ca0a799dcfd18d86/src/layers/other.py#L4-L14
[attn]: https://github.com/PaulCouairon/JAFAR/blob/25321a2def9b00d06a8a8794ca0a799dcfd18d86/src/layers/attentions.py#L8-L52
[vit]: https://github.com/PaulCouairon/JAFAR/blob/25321a2def9b00d06a8a8794ca0a799dcfd18d86/src/backbone/vit_wrapper.py#L136-L166
[cfg]: https://github.com/PaulCouairon/JAFAR/blob/25321a2def9b00d06a8a8794ca0a799dcfd18d86/config/model/jafar.yaml#L4-L11
[teacher]: https://github.com/PaulCouairon/JAFAR/blob/25321a2def9b00d06a8a8794ca0a799dcfd18d86/train.py#L33-L50
[train]: https://github.com/PaulCouairon/JAFAR/blob/25321a2def9b00d06a8a8794ca0a799dcfd18d86/train.py#L104-L159
[loss]: https://github.com/PaulCouairon/JAFAR/blob/25321a2def9b00d06a8a8794ca0a799dcfd18d86/src/loss.py#L127-L145
[demo]: https://github.com/PaulCouairon/JAFAR/blob/25321a2def9b00d06a8a8794ca0a799dcfd18d86/colab_demo.ipynb
