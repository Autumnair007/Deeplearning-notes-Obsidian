---
type: concept-note
tags:
  - cv
  - nlp
  - text-to-image
  - generative-ai
  - diffusion-model
  - ldm
  - vae
  - unet
  - transformer
  - stable-diffusion
status: done
model: Stable Diffusion
venue: CVPR2022
paper_year: 2021
last_verified: 2026-07-17
---
# Stable Diffusion：从像素到隐空间扩散

详细版旧笔记：[[stable_diffusion_notes]]

> [!important] 本文的适用范围
> 除非特别说明，本文讨论的是经典的 **Stable Diffusion v1.5（SD 1.x）**。它采用冻结的 AutoencoderKL、冻结的 OpenAI CLIP ViT-L/14 文本编码器，以及在隐空间中预测噪声的条件 U-Net。
>
> SD 2.x、SDXL、SD3 并不与它完全相同。例如，SD 2.x 更换了文本编码器且部分模型采用 $v$-prediction；SDXL 扩大了 U-Net 并使用双文本编码器；SD3 则采用 Rectified Flow 与 MMDiT。不能把本文所有维度和训练目标直接套到这些版本上。

Stable Diffusion 是 **Latent Diffusion Model（LDM）** 的一个公开模型系列。LDM 论文在 2021 年首次提交，并发表于 CVPR 2022；Stable Diffusion 模型在 2022 年公开，因此元数据中的 `year` 记为 2022。

本文用一个具体训练样本说明 SD v1.5 的主线：

- 输入图像：一张 $512\times512$ 的柯基犬照片；
- 配对文本：为贴合 SD v1.5 的训练分布，使用英文 caption `a cute corgi`；
- Batch Size：$B=1$。

一句话概括训练目标：

> 把真实图像压缩到隐空间，在随机时间步加入已知高斯噪声，再训练 U-Net 根据带噪隐变量、时间步和文本条件预测这份噪声。

![[../../../../../99_Assets (资源文件)/images/b422453087751a51f60cdec4d131aa93.png]]

## 一、训练阶段：五个核心步骤

### 第一步：用 VAE 编码到隐空间

直接在 $512\times512$ 像素上运行扩散模型代价很高。LDM 先训练一个带感知压缩能力的自编码器，再在它的隐空间中训练扩散模型。

SD v1.5 的典型张量变化为：

- 输入图像 $x$：`[1, 3, 512, 512]`；
- VAE 编码器输出后验分布的参数 $\mu(x),\sigma(x)$；
- 从后验中采样原始 latent：

$$
z_{\mathrm{raw}}=\mu(x)+\sigma(x)\odot\eta,\qquad \eta\sim\mathcal N(0,I);
$$

- 空间尺寸缩小 8 倍、通道数变为 4：`[1, 4, 64, 64]`；
- SD 1.x 实现还会使用缩放系数 $s=0.18215$：

$$
z_0=s\,z_{\mathrm{raw}}.
$$

这里的四通道 latent 不是一张“低清 RGB 图”或简单马赛克，而是经过学习得到的连续特征表示。它需要配套的 VAE 解码器才能还原成人类可见图像。

> [!note] 哪些参数在这一阶段更新？
> 要区分两个训练阶段：
>
> 1. **训练第一阶段自编码器时**，VAE 编码器和解码器都参与训练；
> 2. **训练 SD 的扩散 U-Net 时**，已经预训练好的 VAE 被冻结，只负责把图像编码成 latent。此时 VAE 解码器通常不进入扩散损失的计算图。

### 第二步：在随机时间步一次性构造带噪 latent

设训练噪声日程为 $\{\beta_t\}_{t=1}^{T}$，并定义：

$$
\alpha_t=1-\beta_t,\qquad \bar\alpha_t=\prod_{i=1}^{t}\alpha_i.
$$

训练时随机采样时间步 $t$ 和高斯噪声 $\epsilon$：

$$
t\sim\mathrm{Uniform}\{0,\ldots,T-1\},\qquad
\epsilon\sim\mathcal N(0,I),
$$

然后直接构造：

$$
z_t=\sqrt{\bar\alpha_t}\,z_0+\sqrt{1-\bar\alpha_t}\,\epsilon.
$$

当 $B=1$ 时：

- $z_0$：`[1, 4, 64, 64]`；
- $\epsilon$：`[1, 4, 64, 64]`；
- $z_t$：`[1, 4, 64, 64]`。

训练时不需要真的从 $z_0$ 逐步执行 $t$ 次加噪；闭式公式可以一步得到任意 $z_t$。SD v1.x 通常定义 $T=1000$ 个训练时间步，这也不意味着生成一张图必须运行 1000 次 U-Net。推理采样器通常只选取其中一小段离散轨迹。

### 第三步：把文本编码为 token 序列

SD v1.x 使用冻结的 **OpenAI CLIP ViT-L/14 文本编码器**。文本先经过 CLIP tokenizer，再得到逐 token 的上下文表示：

$$
c=\tau(y)\in\mathbb R^{B\times77\times768}.
$$

本例中 $c$ 的形状是 `[1, 77, 768]`：

- 77 是最大序列长度，包含起止符、正文 token 以及 padding；过长文本会被截断；
- 768 是每个文本 token 的隐藏维度；
- U-Net 使用的是整段 token 序列，而不是只使用一个 pooled sentence vector。

这更准确地称为“**token 序列条件下的交叉注意力**”，而不是严格意义上的 Sequence-to-Sequence 模型。

> [!note] 为什么示例改成英文 caption？
> SD v1.5 的训练数据和 CLIP 文本编码器主要面向英文。中文字符串也能被 tokenizer 编码，但其语义控制能力通常明显弱于自然英文提示词。这不影响张量形状，却会影响实际生成质量。

### 第四步：条件 U-Net 预测噪声

U-Net 接收三个主要输入：

1. 带噪 latent $z_t$：`[1, 4, 64, 64]`；
2. 时间步 $t$ 的嵌入；
3. 文本上下文 $c$：`[1, 77, 768]`。

输出为：

$$
\epsilon_\theta(z_t,t,c)\in\mathbb R^{1\times4\times64\times64}.
$$

它是一个与真实噪声同形状的**噪声预测张量（点估计）**，不应称为完整的“噪声概率分布”。

SD v1.5 的去噪器整体仍是卷积 U-Net：它包含下采样路径、中间块、上采样路径和跳跃连接；但若干分辨率层中嵌入了 Spatial Transformer 模块，其中既可以有自注意力，也可以有文本交叉注意力。因此，它既不是纯卷积 U-Net，也不是纯 Transformer。

#### 交叉注意力如何注入文本？

假设某层图像特征为：

```text
h: [B, C, H, W]
```

展平空间维度后：

```text
h_flat: [B, N, C],  N = H × W
```

多头交叉注意力中，图像特征产生 Query，文本序列产生 Key 和 Value：

$$
Q=h_{\mathrm{flat}}W_Q,\qquad K=cW_K,\qquad V=cW_V,
$$

$$
A=\mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d}}\right),\qquad
\mathrm{CrossAttn}(h,c)=AV.
$$

若当前空间分辨率是 $64\times64$，则 $N=4096$。忽略 batch 和 head 维度时，注意力权重可理解为一个 `4096 × 77` 的矩阵：每个空间位置对 77 个文本 token 分配权重。多头实现中的实际形状更接近 `[B, heads, N, 77]`。

需要注意：Q、K 都经过可学习投影。较大的注意力权重表示模型学到的条件路由更强，不等价于“原始视觉向量与某个单词在 CLIP 空间中的余弦相似度”，也不能把单个注意力值直接解释成可靠的目标边界。

### 第五步：计算噪声预测损失

SD v1.x 的简化训练目标是：

$$
\mathcal L_{\mathrm{simple}}
=\mathbb E_{z_0,y,t,\epsilon}
\left[
\left\|\epsilon-\epsilon_\theta(z_t,t,\tau(y))\right\|_2^2
\right].
$$

直观上：

- $\epsilon$ 是本轮真正加入的噪声，即监督信号；
- $\epsilon_\theta$ 是 U-Net 的预测；
- 两者通常以 MSE 比较；
- 反向传播主要更新 U-Net，包括其中的卷积、时间步模块、自注意力和交叉注意力参数；
- 经典 SD v1.5 训练中，VAE 与 CLIP 文本编码器保持冻结。

模型训练时还可以随机丢弃一部分文本条件，让同一个 U-Net 同时学到条件预测和空条件预测，为推理时的 Classifier-Free Guidance（CFG）提供基础。

## 二、推理阶段：从纯噪声生成图像

训练完成后不再输入真实图像。推理从高斯噪声开始：

$$
z_T\sim\mathcal N(0,I),\qquad z_T\in\mathbb R^{1\times4\times64\times64}.
$$

在每个采样时间步，U-Net 分别得到空条件和文本条件下的预测。CFG 将二者组合为：

$$
\hat\epsilon_{mathrm{cfg}}
=\epsilon_\theta(z_t,t,c_{\varnothing})
+w\left[
\epsilon_\theta(z_t,t,c)-\epsilon_\theta(z_t,t,c_{\varnothing})
\right],
$$

其中 $w$ 是 guidance scale。增大 $w$ 往往增强提示词一致性，但过大可能导致颜色过饱和、细节僵硬或多样性降低。

调度器或采样器根据 $\hat\epsilon_{\mathrm{cfg}}$ 把 $z_t$ 更新为 $z_{t-1}$。DDIM、Euler、DPM-Solver 等采样器的区别主要在于如何离散化并求解这条反向生成轨迹；它们不是重新训练了一套 U-Net。

得到最终 latent 后，撤销 SD 1.x 的缩放并通过 VAE 解码器：

$$
\hat x=\mathcal D\left(z_0/s\right),\qquad s=0.18215,
$$

最终输出约为 `[1, 3, 512, 512]` 的图像。

## 三、训练与推理的区别

| 环节 | 扩散训练 | 文生图推理 |
|---|---|---|
| 起点 | 真实图像 $x$ | 纯高斯噪声 $z_T$ |
| VAE 编码器 | 使用，把 $x$ 编码为 $z_0$ | 不使用 |
| 加噪 | 随机抽一个 $t$，一次构造 $z_t$ | 不执行正向加噪 |
| U-Net 调用 | 每个样本通常针对一个随机 $t$ | 沿采样轨迹重复调用多次 |
| 文本编码器 | 提供条件；通常冻结 | 提供条件和空条件 |
| 损失与反向传播 | 有 | 无 |
| VAE 解码器 | 扩散训练中通常不使用 | 最后使用一次 |

因此，“训练时从真实图像恢复图像”和“推理时从纯噪声生图”不能混为同一个过程。训练只要求模型在各种噪声强度下学会局部的去噪方向；采样器把许多这样的局部预测串起来，才形成完整生成轨迹。

## 四、几个常见混淆

### 1. Stable Diffusion 的目标永远都是预测噪声吗？

不是。对本文讨论的 **SD v1.5**，写成 $\epsilon$-prediction 是准确的；但其他扩散模型还可以预测干净样本 $x_0$、速度变量 $v$，或者学习 score。SD3 采用的 Rectified Flow 又是另一种参数化。因此，“模型的目标是找噪声”只能作为 SD v1.x 的入门主线，不能定义整个扩散模型家族。

### 2. Stable Diffusion 中的 VAE 和“扩散模型像 VAE”是一回事吗？

不是。

- SD 的 AutoencoderKL 是一个真实存在的第一阶段压缩器，负责像素空间与隐空间之间的映射；
- DDPM 本身又可以写成具有许多随机隐变量的生成模型，并通过变分下界推导训练目标。

第二点说明扩散模型与层次化潜变量模型/变分推断关系紧密，但不宜简化成“扩散模型数学上就是普通 VAE”。DDPM 还可以从去噪分数匹配、SDE/ODE 等角度理解。

### 3. 训练时间步和推理步数是一回事吗？

不是。训练可以定义 1000 个噪声时间步，推理时则由采样器选择几十步甚至更少的轨迹。推理步数越少速度越快，但误差通常也更难控制。

### 4. 自注意力和交叉注意力分别在做什么？

- **自注意力**：Q、K、V 都来自图像 latent 特征，主要建模图像不同空间位置之间的关系；
- **交叉注意力**：Q 来自图像 latent，K、V 来自文本 token，主要把文本条件注入图像特征。

DiCLIP 等工作利用的正是 SD U-Net 自注意力中包含的空间相关性，而不是简单把文本交叉注意力直接当成分割结果。参见：[[DiCLIP_paper_notes]]。

## 五、进一步思考：哪些直觉成立，哪些需要加限定？

### 1. 能否用 DINO 一类表示编码器替换 VAE？

不能把现成 DINO 编码器直接塞进 SD，然后期待原 VAE 解码器无缝还原图像：编码空间、缩放、分布和解码器都不匹配，而且表示模型通常更强调语义不变性，而不是像素级可逆性。

但“DINO 类编码器不能用于扩散生成”已经不正确。**Representation Autoencoders（RAE）** 的工作把冻结的 DINO、SigLIP、MAE 等表示编码器与专门训练的解码器配对，在语义更强、维度更高的 latent 空间中训练 DiT，并取得很强的生成结果。真正的技术难点包括：

- 为表示空间训练高质量解码器；
- 处理高维 latent 的噪声尺度和优化问题；
- 让扩散 Transformer 有效建模这些 latent。

所以更准确的结论是：**可以替换，但必须把“编码器、解码器、latent 分布和去噪器”作为一个系统重新设计。**

### 2. U-Net 可以被 Transformer 替换吗？

可以。DiT 用处理 latent patch 的 Transformer 替代常见 U-Net，并观察到增加模型深度、宽度或 token 数带来的 Gflops 增长通常伴随更低 FID。这说明 DiT 具有良好的可扩展性，但不能把实验结论写成“生成质量必然线性增长”。

SD3 的去噪网络采用 MMDiT：图像 token 与文本 token 使用不同权重流，并允许双向交换信息。Sora 的官方技术说明也明确描述了作用于时空 patch 的 diffusion transformer。不过，这并不表示所有新模型都已淘汰 U-Net，也不能说 Sora 原封不动采用了最初的 class-conditional DiT。

### 3. 扩散模型是不是一种“另类 VAE”？

可以把 DDPM 看成带有长马尔可夫隐变量链的生成模型，并从变分下界推导训练目标；从这个角度，它与层次化 VAE 有亲缘关系。但扩散过程通常不依靠一个低维空间瓶颈来完成信息压缩，而且其常用训练解释还包括去噪分数匹配。

因此推荐表述为：

> 扩散模型是潜变量生成模型的一类；DDPM 的目标可由变分推断推导，并与去噪分数匹配建立联系。它与层次化 VAE 密切相关，但二者不应在所有语境下直接画等号。

### 4. 为什么保留文本 token 序列？ControlNet 又解决什么问题？

保留逐 token 表示，使不同图像位置能够通过交叉注意力选择不同文本成分；但这并不保证模型一定正确处理数量、属性绑定和复杂空间关系。

ControlNet 处理的是另一类条件：边缘、深度、人体姿态、分割图等空间结构。原始 ControlNet 冻结预训练 Stable Diffusion 主干，复用其编码块和中间块建立可训练分支，并用零初始化卷积连接两条路径。它不是一个简单的小型文本编码器，也不是把控制图“揉进提示词”。通常不同控制类型需要相应训练数据和专门权重。

### 5. 去噪是否天然“从粗到细”？

标准图像扩散采样经常呈现这样的经验趋势：高噪声阶段首先确定构图、主体位置和大色块，低噪声阶段再完善纹理与边缘。这与自然图像的频谱统计、不同噪声水平下各频率成分的有效信噪比以及网络归纳偏置有关。

但需要加三点限定：

1. 这是一种常见经验规律，不是对任意数据、噪声日程和模型架构都成立的定理；
2. “低频较早出现”不等价于“语义最显著的概念一定最早出现”；
3. 专门的 blur diffusion、级联模型或多尺度采样方法会显式改变不同频率的生成顺序，不能用它们的结果直接证明标准 SD 的每一步都严格遵循固定的由粗到细规律。

## 六、核心总结

以 SD v1.5 为例，完整链路是：

```text
训练：
真实图像 x
  -> 冻结 VAE 编码与缩放得到 z0
  -> 随机抽 t 和 ε，一步构造 zt
  -> 冻结 CLIP 编码文本得到 token 序列 c
  -> 条件 U-Net 预测 εθ(zt, t, c)
  -> 与真实 ε 计算 MSE，只更新 U-Net

推理：
随机噪声 zT
  -> U-Net + CFG + 采样器反复更新
  -> 得到 z0
  -> 撤销缩放并用冻结 VAE 解码
  -> 输出图像
```

Stable Diffusion 的关键不只是“会去噪”，而是把三个组件组合起来：

1. **VAE**：把高成本像素空间转换为较小的可重建隐空间；
2. **文本编码器与交叉注意力**：把 token 级语言条件注入空间特征；
3. **扩散 U-Net 与采样器**：学习局部去噪方向，并在推理时把这些方向串成从噪声到图像的轨迹。

## 参考资料（优先一手来源）

1. Rombach et al., [High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752), CVPR 2022.
2. Ho et al., [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239), NeurIPS 2020.
3. Song et al., [Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502), ICLR 2021.
4. Ho and Salimans, [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598), 2022.
5. Radford et al., [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020), ICML 2021.
6. Peebles and Xie, [Scalable Diffusion Models with Transformers](https://arxiv.org/abs/2212.09748), ICCV 2023.
7. Esser et al., [Scaling Rectified Flow Transformers for High-Resolution Image Synthesis](https://arxiv.org/abs/2403.03206), 2024（SD3 / MMDiT）.
8. Zhang et al., [Adding Conditional Control to Text-to-Image Diffusion Models](https://arxiv.org/abs/2302.05543), ICCV 2023（ControlNet）.
9. Zheng et al., [Diffusion Transformers with Representation Autoencoders](https://arxiv.org/abs/2510.11690), 2025（RAE）.
10. [Stable Diffusion v1.5 Model Card](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5).
11. OpenAI, [Video generation models as world simulators](https://openai.com/index/video-generation-models-as-world-simulators/), 2024（Sora 技术说明）.
