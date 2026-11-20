# UPLiFT 源码学习笔记

给定一张图和一张很粗的骨干特征图，想给物体轮廓两边的像素分配不同的语义特征。UPLiFT 把这件事拆成两个问题。**轮廓在哪里？** 看图像的浅层卷积特征，也看当前的语义特征。**轮廓两侧分别填什么？** 从粗网格邻居已有的特征向量中按比例取。这样一个小解码器只需学会放大两倍，多次调用就能得到稠密特征。

[CVPR 2026 论文](https://openaccess.thecvf.com/content/CVPR2026/html/Walmer_UPLiFT_Efficient_Pixel-Dense_Feature_Upsampling_with_Local_Attenders_CVPR_2026_paper.html) · [作者代码仓库](https://github.com/mwalmer-umd/UPLiFT) · [[UPLiFT_Translation]] · [[UPLiFT_paper_notes]]

| 想跟踪的计算 | 对应源码 |
|---|---|
| 从一张图到骨干网格，再交给上采样器 | [UPLiFTExtractor.forward](https://github.com/mwalmer-umd/UPLiFT/blob/e58d213d79c125d5cceaa7af0fefb6a94677bf55/uplift/uplift_extractor.py#L142-L185) · [ViT 包装器](https://github.com/mwalmer-umd/UPLiFT/blob/e58d213d79c125d5cceaa7af0fefb6a94677bf55/uplift/extractors/vit_wrapper.py#L136-L166) |
| 一次图像编码、共享解码器、多轮迭代 | [UPLiFT.forward 与 upsample](https://github.com/mwalmer-umd/UPLiFT/blob/e58d213d79c125d5cceaa7af0fefb6a94677bf55/uplift/uplift.py#L598-L643) · [编码器与解码器](https://github.com/mwalmer-umd/UPLiFT/blob/e58d213d79c125d5cceaa7af0fefb6a94677bf55/uplift/uplift.py#L373-L531) |
| 把一个粗格的邻居分给多个子像素 | [LocalAttender](https://github.com/mwalmer-umd/UPLiFT/blob/e58d213d79c125d5cceaa7af0fefb6a94677bf55/uplift/uplift.py#L740-L847) · [DINOv2-S/14 的模型定义](https://github.com/mwalmer-umd/UPLiFT/blob/e58d213d79c125d5cceaa7af0fefb6a94677bf55/uplift/configs/uplift_dinov2-s14.yaml#L12-L100) |
| 冻结骨干构造各尺度目标、多个输出共同受监督 | [train_uplift.py 的特征提取与损失](https://github.com/mwalmer-umd/UPLiFT/blob/e58d213d79c125d5cceaa7af0fefb6a94677bf55/uplift/train_uplift.py#L241-L334) |
| 换到生成任务后，哪些约束发生变化 | [Stable Diffusion 1.5 的模型定义](https://github.com/mwalmer-umd/UPLiFT/blob/e58d213d79c125d5cceaa7af0fefb6a94677bf55/uplift/configs/uplift_sd1.5vae.yaml#L13-L117) · [变分自编码器的编码与解码](https://github.com/mwalmer-umd/UPLiFT/blob/e58d213d79c125d5cceaa7af0fefb6a94677bf55/uplift/extractors/diff_extractor.py#L62-L91) |

## 1. 跟一次推理，看清三种不同的特征

以 DINOv2-S/14 这个视觉 Transformer（ViT）骨干为例。`PretrainedViTWrapper.forward` 向 timm 请求最后一层的空间特征（`output_fmt="NCHW"`），把前缀 token 另行返回。推理包装器取返回元组的第 0 项，所以此处已经是 `[B,384,h,w]` 的 patch 网格。每个格子有自己的图像位置，全局分类 token 不参与后续的空间混合。

```text
归一化图像 ──→ ViT 骨干 ──→ V₀（粗网格，每格一个 384 维向量）
      └────→ UPLiFTEncoder ──→ E（缓存图像的稠密浅层特征）
每轮 t：
  Vₜ、按所需尺寸采样的 E ──→ 共享 Decoder ──→ Gₜ（高分辨率引导特征）
  Gₜ ──→ 每个子像素的邻居权重 ┐
  Vₜ ──→ 本轮粗网格上的邻居值 ┴──→ Local Attender 输出 Vₜ₊₁
Vₜ₊₁ ──→ 下一轮；最后一轮的特征交给分割或深度预测头
```

**E 保存图像细节，G 是本轮如何取值的依据，V 承载实际要输出的语义。** E 只编码一次，G 和 V 每轮更新。解码器里有一个可学习的转置卷积，它先把引导表示放大。最后的 384 维向量出自 Local Attender 的值分支。把这两个“放大”分清，才能解释为什么解码器即使只输出 64 个通道，也能产生 384 通道的最终特征。

## 2. 解码器怎样同时读到轮廓和语义

在 DINOv2-S/14 模型中，`UPLiFTEncoder.forward` 将五个连续卷积组的结果依次放进列表。配置名 `down_steps: 4` 表示遍历四个可选下采样位置，实际四处 `down_op` 都为 `none`，各组都保留同一空间网格，通道数依次为 16、32、64、128、128。共享的是这份列表。解码器只有一次上采样，直接用到列表的最后两项，前面各组通过卷积链间接参与。对照 [编码器前向](https://github.com/mwalmer-umd/UPLiFT/blob/e58d213d79c125d5cceaa7af0fefb6a94677bf55/uplift/uplift.py#L417-L442) 和 [DINOv2 配置](https://github.com/mwalmer-umd/UPLiFT/blob/e58d213d79c125d5cceaa7af0fefb6a94677bf55/uplift/configs/uplift_dinov2-s14.yaml#L23-L100)。

下面是 `UPLiFTDecoder.forward` 实际走到的语句，省略配置开关。两个 `noise_and_combine` 在此配置下的噪声通道数均为 0。[解码器](https://github.com/mwalmer-umd/UPLiFT/blob/e58d213d79c125d5cceaa7af0fefb6a94677bf55/uplift/uplift.py#L502-L527) · [拼接函数](https://github.com/mwalmer-umd/UPLiFT/blob/e58d213d79c125d5cceaa7af0fefb6a94677bf55/uplift/uplift.py#L354-L367)

```python
x = noise_and_combine(x, enc_feats[-1], self.ncat_c, self.ncat_s, self.enc_resample)
x = self.block_0(x)
x = self.up_1(x)
x = noise_and_combine(x, enc_feats[-2], self.ncat_c, self.ncat_s, self.enc_resample)
x = self.block_1(x)
```

取一张 `448×448` 图，骨干步长为 14，第一轮从 `32×32` 出发。

| 位置 | 第一轮的实际形状 | 这一行学什么 |
|---|---|---|
| 粗尺度融合 | `[B,384,32,32]` 与最近邻缩至 `[B,128,32,32]` 的 `enc_feats[-1]` 沿通道拼接，得 `[B,512,32,32]` | 同一位置的语义和外观共同进入卷积 |
| `block_0 → up_1` | 两次 $3\times3$ 卷积等组成的块先输出 64 通道。`ConvTranspose2d` 的核与步长均为 2，输出 `[B,64,64,64]`，加上同尺寸双线性上采样的残差 | 从一个粗格产生四个可区分的子位置，残差提供平滑的初始信号 |
| 细尺度融合 | 按新尺寸取 `enc_feats[-2]`，通道拼接后是 `[B,192,64,64]`，`block_1` 产出 `G=[B,64,64,64]` | 让四个子位置各自看到图像线索，用于预报各自的权重 |
| Local Attender | `G` 和保留的 `V=[B,384,32,32]` 输入，得到 `[B,384,64,64]` | 输出空间随 G，通道随 V |

`torch.cat(..., dim=1)` 要求两份图的批量、高、宽一致，通道可以不同。源码在每个调用位置将缓存的图像特征最近邻缩放到**当前** `x.shape[2:]`。这也说明了 `UPLiFT.upsample` 为什么返回 `(x, enc_feats)`，下一轮沿用高密度的缓存重新采样，避免拿已经缩到 `32×32` 的引导图再逐轮插大而丢失纹理。[上采样算子及双线性残差](https://github.com/mwalmer-umd/UPLiFT/blob/e58d213d79c125d5cceaa7af0fefb6a94677bf55/uplift/uplift.py#L319-L343) · [缓存的读写](https://github.com/mwalmer-umd/UPLiFT/blob/e58d213d79c125d5cceaa7af0fefb6a94677bf55/uplift/uplift.py#L599-L635)

**步长 14 需要额外对齐。** 上例的骨干看 `448×448`。`upsample` 只在提取浅层特征前按整数计算 `448×16//14=512`，这条 RGB 引导分支的内部网格因而是 `512×512`。四轮两倍上采样给出 `32→64→128→256→512`。外层 `UPLiFTExtractor` 默认先把非整除的 PIL 图像宽、高分别四舍五入至最近的 patch 整数倍，用双线性插值调整。包装器构造参数名为 `out_size`，它向内层 `UPLiFT.forward` 传递这个值。若内层使用 `outsize=[448,448]`，`forward` 最后才对 512 网格做双线性调整，此分支直接返回调整后的结果。论文评测要求的输出尺寸、骨干输入尺寸与迭代自然产生的尺寸是三个不同的量。迁移到新骨干时先算清这三者，再确定迭代次数及最终是否需要插值。[步长适配与输出调整](https://github.com/mwalmer-umd/UPLiFT/blob/e58d213d79c125d5cceaa7af0fefb6a94677bf55/uplift/uplift.py#L598-L643) · [包装器输入对齐](https://github.com/mwalmer-umd/UPLiFT/blob/e58d213d79c125d5cceaa7af0fefb6a94677bf55/uplift/uplift_extractor.py#L224-L235)

## 3. Local Attender 怎样让同一粗格长出不同的子像素

### 3.1 每个子像素给固定方向打分

把粗网格上的“中心、左、右、上、下……”先排成固定顺序。引导图在每个细位置只回答一件事：这个位置应分别从这些方向各取多少。论文 第 3.2 节 的邻域 $\mathcal N$ 对应 `self.offsets`。源码支持 5、9、13、17、25 个偏移。DINOv2 配置的 17 点是完整 $3\times3$ 内圈，加上距离两格的四个轴向点与四个角点，有效覆盖 $5\times5$ 范围。偏移量按**粗网格**的行、列计数。[邻域列表与可学习投影](https://github.com/mwalmer-umd/UPLiFT/blob/e58d213d79c125d5cceaa7af0fefb6a94677bf55/uplift/uplift.py#L740-L770)

```python
att = self.conv1(guide)
if self.conv_res:
    att = residual_auto(guide, att)
```

$1\times1$ 投影让同一位置的 64 个引导通道变成 17 个方向分数。`conv_res=True` 是公开实现的默认值，`residual_auto` 在 64→17 这条路径上又加回 `guide[:, :17]`。若输入通道较少则在通道末尾补零。这个捷径没有参数，**加在 softmax 前的方向分数上**。稍后的 `F.softmax(att, dim=1)` 沿 17 个邻居归一化，既不沿空间求和，也不跨图像求和。同一粗格中的四个子位置可以学到四套不同权重。[残差的通道处理](https://github.com/mwalmer-umd/UPLiFT/blob/e58d213d79c125d5cceaa7af0fefb6a94677bf55/uplift/uplift.py#L24-L39) · [分数生成与归一化](https://github.com/mwalmer-umd/UPLiFT/blob/e58d213d79c125d5cceaa7af0fefb6a94677bf55/uplift/uplift.py#L801-L830)

这里没有 $QK^\top$。高分辨率的 G 担任选择邻居的角色，固定方向给出可访问的位置，V 直接提供原特征向量。G 的生成已经看过 V 和 E，所以权重仍然随图像内容改变。把它接到自己的模型上，G 和 V 的通道数可以不同，两者的裁剪范围、方向、空间原点和倍率则必须匹配。

### 3.2 一组偏移切片怎样变成所有粗格的邻居

`make_offsets` 先用 `ReplicationPad2d` 复制边缘，再对每个方向各切一张与输入等大的图。设 V 为 `[B,C,H,W]`，`pn` 是最大偏移半径。[make_offsets](https://github.com/mwalmer-umd/UPLiFT/blob/e58d213d79c125d5cceaa7af0fefb6a94677bf55/uplift/uplift.py#L784-L798)

```python
x = self.pad(x)
x_all = []
for off in self.offsets:
    off_0, off_1 = off
    x_off = x[:, :, self.pn+off_0:self.pn+off_0+H, self.pn+off_1:self.pn+off_1+W]
    x_all.append(x_off)
x = torch.stack(x_all, dim=2)
```

`off=(0,1)` 对应那张切片在坐标 $(h,w)$ 的元素，读到输入 $(h,w+1)$ —— 整张结果的每一格都知道自己的右邻居。最右一列会再次读到本列，所分到的权重仍然落在有效语义向量上。`stack(dim=2)` 新建邻居轴，得到 `[B,C,D,H,W]`，$D=|\mathcal N|$。各偏移的顺序就是分数通道的含义。调换 `offsets` 的顺序但沿用已训练的权重，会让“向右取”的分数改去取别的方向；即使 shape 全部正确，边界也可能错位。

### 3.3 两个单例维为什么恰好把子像素送回正确位置

用 $I$ 表示行列共同的整数倍率，源码分别断言 `H_out==I*H`、`W_out==I*W`。以 $I=2$ 为例，细位置 $(2h+u,2w+v)$ 属于粗位置 $(h,w)$，$u,v$ 各取 0 或 1。四个细位置共享同一份可选邻居，各自的取样比例仍然独立。源码的普通并行分支如下，保留运算顺序，省去了中间注释。[LocalAttender.forward](https://github.com/mwalmer-umd/UPLiFT/blob/e58d213d79c125d5cceaa7af0fefb6a94677bf55/uplift/uplift.py#L817-L837)

```python
x = self.make_offsets(x)
x = torch.unsqueeze(x, dim=4)
x = torch.unsqueeze(x, dim=6)

att = F.softmax(att, dim=1)
att = torch.reshape(att, [B, D, H, I, W, I])
att = torch.unsqueeze(att, dim=1)

res = x * att
res = torch.sum(res, dim=2)
res = res.reshape([B, C, H_out, W_out])
```

逐个对齐轴更容易读懂这段广播。

```text
轴           batch  通道  邻居  粗行  子行  粗列  子列
x             B      C     D     H     1     W     1
att           B      1     D     H     I     W     I
逐元素乘积    B      C     D     H     I     W     I
沿邻居求和    B      C           H     I     W     I
恢复图像      B      C            H × I       W × I
```

`unsqueeze` 只插入长度为 1 的轴。乘法时，一个邻居向量被同一粗格的 $I^2$ 个子像素读取；反过来，每个子像素的一套邻居权重广播给全部 $C$ 个特征通道。`sum(dim=2)` 只消去邻居轴，逐通道做相同的空间加权。`reshape` 中 `H,I,W,I` 的交错顺序对应“粗行和子行合成细行，粗列和子列合成细列”。如果写成 `H,W,I,I` 再直接压回图像，元素个数仍对，但权重会跑到别的格子。

具体看一个标量通道。设中心与右邻分别为 10、30，教学例只保留这两个方向，偏中心的子像素给出 `[0.8,0.2]`，输出 14；偏右的子像素给出 `[0.2,0.8]`，输出 26。对完整 384 维向量，两种配比逐分量应用。若邻居全是同一个向量，任意位置权重和都为 1，得到的仍是该向量。常量图可以检查求和后是否保持常量；要分辨边界策略，令值图每一行都为 `[0,1,2,3]`，手工把权重全部给右邻，最右父格的所有子像素应取 3。镜像填充此时取 2，循环填充取 0。

**迁移写法：把预测权重和聚合值拆成两个函数。** 如果新模型已有自己的引导分支，只需让它输出每个方向的 logits，就可以单独复用下面的聚合部分。例如空间标签较稀时，可让高分辨率边缘分支预报权重，值仍取自原有的语义特征。下面是教学改写，采用 `einsum` 直接写出消去哪个维度，假设两张图对齐、偏移列表非空。

```python
import torch
import torch.nn.functional as F

def mix_local_values(value, logits, offsets):
    batch, channels, height, width = value.shape
    high_h, high_w = logits.shape[-2:]
    scale = high_h // height
    assert scale > 0 and (high_h, high_w) == (height * scale, width * scale)
    assert logits.shape[:2] == (batch, len(offsets))

    radius = max(max(abs(dy), abs(dx)) for dy, dx in offsets)
    padded = F.pad(value, (radius,) * 4, mode="replicate")
    views = []
    for dy, dx in offsets:
        y, x = radius + dy, radius + dx
        views.append(padded[:, :, y:y + height, x:x + width])
    neighbors = torch.stack(views, dim=2)
    weights = logits.softmax(dim=1).reshape(batch, len(offsets), height, scale, width, scale)
    result = torch.einsum("bckhw,bkhuwv->bchuwv", neighbors, weights)
    return result.reshape(batch, channels, high_h, high_w)
```

这里的 `k` 是邻居，`h,w` 是父格，`u,v` 是子格。`einsum` 的输出保留 `b,c,h,u,w,v`，所以只对 `k` 求和。`logits` 与 `value` 应在同一设备、使用可共同参与乘加的相同浮点类型。仅更换值的通道数不影响这个函数；整套 UPLiFT 的解码器输入层仍绑定训练时的骨干通道数。用 `[2,3,3,5]` 的非方形值图、5 个方向和两倍放大，对照逐坐标读取参考实现，可以同时发现高宽颠倒、边界填充和子像素顺序错误。

这段改写在 PyTorch 2.1.2 的 CPU 上做过小张量核对。输入为 `2×3×3×5`，交叉取五种官方邻域及 `1/2/3` 倍率。15 组中，官方并行分支、逐邻居累加分支和上面的改写均与逐坐标参考结果在 `torch.testing.assert_close` 容差内一致；改写对 G、V 的梯度也与官方并行分支一致。

### 3.4 反复混合怎样约束特征，成本又落在哪里

`UPLiFT.upsample` 一进门就保留 `x_in = x`，解码器预测 G，`self.attender(x, x_in)` 才输出本轮结果。第一轮的 `x_in` 是原始骨干图，第二轮就是第一轮结果。于是预测分支在每一步都取**本轮局部候选的凸组合**，展开多步后仍可表示为初始骨干向量的非负加权和。这是固定一次前向的权重后对值的描述。由于生成权重的 G 也依赖 V，整个网络对输入仍是非线性的。凸组合能约束取值范围，却不能保证恰好保留原分布、方差或单位范数。邻居都没表达出的语义，单靠这一步也无法凭空补出。[迭代与实参顺序](https://github.com/mwalmer-umd/UPLiFT/blob/e58d213d79c125d5cceaa7af0fefb6a94677bf55/uplift/uplift.py#L599-L635)

若输出位置数 $T=(IH)(IW)$，权重图有 $BDT$ 个元素，直接聚合需约 $BCDT$ 次标量乘加。固定邻域和通道数时，成本随输出位置数线性增加。普通 PyTorch 路径仍会物化乘积 `[B,C,D,H,I,W,I]`，大通道、高分辨率时它可以非常大。`low_mem=True` 保留低分辨率的 `stack`，将 `D` 次加权逐项累加，省掉同时展开全部方向的高分辨率乘积，代价是更多串行操作。普通低显存分支用 `torch.zeros(...).to(x.device)` 建累加器，默认浮点类型通常是 FP32，并未显式继承输入的 dtype。移植时可用 `x.new_zeros(...)` 保持一致，或显式指定 FP32 累加后按接口需要转换。论文的渐近复杂度与实际峰值内存解释的是两个不同问题。[并行与逐邻居分支](https://github.com/mwalmer-umd/UPLiFT/blob/e58d213d79c125d5cceaa7af0fefb6a94677bf55/uplift/uplift.py#L822-L847)

## 4. 一个批次怎样监督同一解码器的多轮输出

先看 DINOv2 的预测任务配置（`lat_aug=0`）。训练时，高分辨率图像给出可比较的目标。源码先将同一批图调整为若干尺寸，分别在 `torch.no_grad()` 下提取骨干特征，存进以图像边长为键的 `feat_scales`。这样每个目标都来自**骨干在对应尺度重新看图的结果**。把最粗特征插值放大只会产生平滑版本，提供不了同样的细节监督。模型训练时打开 `return_steps(True)` 返回所有轮次的预测，优化器只接收 `uplift.parameters()`。[模型与损失定义](https://github.com/mwalmer-umd/UPLiFT/blob/e58d213d79c125d5cceaa7af0fefb6a94677bf55/uplift/train_uplift.py#L104-L108) · [优化器、MSE](https://github.com/mwalmer-umd/UPLiFT/blob/e58d213d79c125d5cceaa7af0fefb6a94677bf55/uplift/train_uplift.py#L181-L196) · [多尺度骨干目标](https://github.com/mwalmer-umd/UPLiFT/blob/e58d213d79c125d5cceaa7af0fefb6a94677bf55/uplift/train_uplift.py#L241-L275)

以 DINOv2、训练图最大边长 448、`down_steps=3` 为例。下表只写边长，所有网格均为正方形。

| 训练深度 | 送入上采样器的图像 / 骨干网格 | 依次预测的网格边长 | 分别与哪个图像尺寸的骨干结果比较 |
|---|---|---|---|
| 1 | 224 / 16 | 32 | 448 |
| 2 | 112 / 8 | 16、32 | 224、448 |
| 3 | 56 / 4 | 8、16、32 | 112、224、448 |

三行各自从不同分辨率的真实骨干特征起步，同一行内才把上一轮预测送进下一轮。比如表中同样预测 `32×32`，第一行从 `16×16` 读一次，第三行从 `4×4` 连续读三次，累积误差不同。多个起点把这些情况都交给同一套参数学习。实现中 `scale_dst = feat_out.shape[2] * cfg.backbone.patch` 用预测网格的高度找目标尺寸，这依赖训练图是方形且与步长对齐。若改成矩形或不规则步长的数据，应按真实的高宽和网格契约匹配目标。[训练深度与目标查找](https://github.com/mwalmer-umd/UPLiFT/blob/e58d213d79c125d5cceaa7af0fefb6a94677bf55/uplift/train_uplift.py#L286-L334)

论文式 (1) 描述最深输出的重建，式 (2)–(4) 将中间输出和同尺度目标配对，式 (5) 再跨训练深度相加。配置中的 `l2` 对应默认 `torch.nn.MSELoss()`，即每个尺度先对批、通道、空间元素的**平方误差求均值**，最后加上 1+2+3 共六个标量。没有对六项再求平均，也没有对平方和开根号。`loss.backward()` 穿过解码器每次使用的位置，循环中没有把上轮输出 `detach`。末轮的误差同样能训练较早轮次的共享参数。推理时只需输入图像及其一次骨干提取，不再准备高分辨率教师图或计算任何重建损失。[前向循环](https://github.com/mwalmer-umd/UPLiFT/blob/e58d213d79c125d5cceaa7af0fefb6a94677bf55/uplift/uplift.py#L624-L643) · [损失累计](https://github.com/mwalmer-umd/UPLiFT/blob/e58d213d79c125d5cceaa7af0fefb6a94677bf55/uplift/train_uplift.py#L310-L334)

## 5. 换成生成用的潜变量，为什么多了一个细化器

Stable Diffusion 1.5 的变分自编码器（VAE）每隔 8 个像素产生一个 4 维潜变量。官方 `DiffExtractor.encode` 调用 VAE 后取 `latent_dist.mode()`，上采样所得的 4 通道特征可以送回 `vae.decode` 得到 RGB 图。例：一张 `256×256` 图编码成 `[B,4,32,32]`，两轮放大到 `[B,4,128,128]`，解码后为 `1024×1024` 图像。这条路径没有把图像先提升到 1024 再送进 VAE 编码。[潜变量的取得与解码](https://github.com/mwalmer-umd/UPLiFT/blob/e58d213d79c125d5cceaa7af0fefb6a94677bf55/uplift/extractors/diff_extractor.py#L32-L38) · [encode/decode](https://github.com/mwalmer-umd/UPLiFT/blob/e58d213d79c125d5cceaa7af0fefb6a94677bf55/uplift/extractors/diff_extractor.py#L74-L91)

关键变化发生在每一轮的 Local Attender 之后。SD1.5 配置保留 17 邻居，解码器两次拼接时各生成 4 个新的高斯噪声通道。注意力输出的 4 通道潜变量再与最后一张 512 通道浅层图、4 个新噪声通道拼接成 520 通道；12 个 `SingleConv` 块在 512 通道上细化，随后 `ref_outc` 用 $1\times1$ 投影回 4 通道。`noise_and_combine` 内的 `torch.randn` 由 `ncat_c>0` 触发，推理也会采样，与训练专用的输入潜变量加噪 `lat_aug` 是两件事。这里局部凸组合描述的是 **Attender 的中间输出**，后置卷积可以产生超出候选潜变量凸包的值。阅读第 5 节的生成结果时应把这条额外路径计入模型能力。[生成模型配置](https://github.com/mwalmer-umd/UPLiFT/blob/e58d213d79c125d5cceaa7af0fefb6a94677bf55/uplift/configs/uplift_sd1.5vae.yaml#L79-L117) · [噪声拼接](https://github.com/mwalmer-umd/UPLiFT/blob/e58d213d79c125d5cceaa7af0fefb6a94677bf55/uplift/uplift.py#L354-L367) · [refiner 的输入和输出](https://github.com/mwalmer-umd/UPLiFT/blob/e58d213d79c125d5cceaa7af0fefb6a94677bf55/uplift/uplift.py#L565-L577) · [每轮调用](https://github.com/mwalmer-umd/UPLiFT/blob/e58d213d79c125d5cceaa7af0fefb6a94677bf55/uplift/uplift.py#L611-L621)

**训练时的潜变量加噪还会影响后续中间目标。** SD1.5 配置设 `lat_aug: 1`。训练循环按深度从浅到深读取 `feat_in = feat_scales[scale_in]`，随后用 `feat_in += noise_aug` 原地更新，字典中的张量也随之改变。例如最高图像边长 1024 时，深度 1 对图像边长 512 的潜变量（网格 `64×64`）加噪，预测 1024 图像对应的干净目标。接着深度 2 以边长 256 图像的加噪潜变量为起点，其第一轮预测 `64×64`，按 `64×8=512` 查表，取得的正是刚刚改过的张量。最终 1024 尺度从未作为低分辨率输入，目标保持原始提取值。这个共享引用只涉及当批内存中的 `feat_scales`。迁移到需要始终保留干净教师的实现时，可用 `feat_in = feat_in + noise_aug` 产生新张量；这会改变公开训练代码的行为。[VAE 训练设置](https://github.com/mwalmer-umd/UPLiFT/blob/e58d213d79c125d5cceaa7af0fefb6a94677bf55/uplift/configs/uplift_sd1.5vae.yaml#L119-L136) · [原地加噪与目标读取](https://github.com/mwalmer-umd/UPLiFT/blob/e58d213d79c125d5cceaa7af0fefb6a94677bf55/uplift/train_uplift.py#L286-L328)

## 6. 在自己的模型里可以带走什么

| 手头的问题 | 可复用的写法 | 先核对什么 |
|---|---|---|
| 想把已有粗语义图提升到轮廓清晰的稠密特征 | 让高分辨率分支预测局部空间权重，直接重组原特征向量 | 两条分支必须是同一裁剪和原点；若部署时只有粗特征，模型就缺少高分辨率引导 |
| 倍率固定，但多轮推理不断重复相同的图像编码 | 像 `UPLiFT.forward` 一样让图像缓存跨轮复用，每轮在实际拼接位置采样 | 只缓存与当前轮次无关的图像特征；G 包含本轮 V 的信息，需要重算 |
| 写了自定义 attention，形状正确却出现棋盘或位移 | 把轴显式写为“邻居、粗行、子行、粗列、子列”，用单一方向权重及带坐标的网格核对 | 只选中心时应等于按整数倍率复制；只选右邻时最右列应复制边界，整幅常量图应保持常量 |
| 同一模型要用一步，也要连续用很多步 | 返回各轮结果，以同一冻结骨干在相应图像尺度的输出做深监督 | 各项损失的归约方式决定权重；中间预测若被截断梯度，末步监督就无法穿过前一步 |

最值得重读的三处分别是 `UPLiFT.upsample` 的 `x_in`、`LocalAttender.forward` 的两个单例维，以及训练循环的 `scale_dst`。它们分别说明了“数值从哪里来”“哪个像素向哪些邻居取值”和“每轮在拟合谁”。把局部重采样视作通用操作时，可直接回看 [[Attention_and_Affinity_Refinement#6.5 用高分辨率引导生成固定邻域的边权]]。
