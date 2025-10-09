# GAN笔记（Deepseek生成）

学习资料：[图解 生成对抗网络GAN 原理 超详解_gan原理图-CSDN博客](https://blog.csdn.net/DFCED/article/details/105175097)

***
# 生成式对抗网络（GAN）全面笔记

## 1. GAN基本概念

### 1.1 定义与组成
生成式对抗网络（GAN）由Ian Goodfellow等人在2014年提出，其灵感来源于博弈论中的二人零和博弈。可以将其类比为伪钞制造者（生成器）和警察（判别器）的游戏：伪钞制造者试图制造出能骗过警察的假钞，而警察则不断提升自己的鉴别能力。二者在对抗中共同进步，最终使得伪钞制造者能造出与真钞几乎无异的钞票。
- **生成器（Generator, G）：**
  其任务是学习真实数据的分布。它输入一个从简单先验分布（如高斯分布或均匀分布）中采样的随机噪声向量 $z$（也称为潜在向量），输出一个与真实数据维度相同的生成数据 $G(z)$。生成器的目标是让其输出 $G(z)$ 的分布 $p_g$ 尽可能地接近真实数据的分布 $p_{data}$，从而“欺骗”判别器。
- **判别器（Discriminator, D）：**
  其本质是一个二分类器，用于判断输入的数据是真实的还是生成的。它输入一个数据样本 $x$（可能来自真实数据集，也可能来自生成器），输出一个标量 $D(x)$，该值在 $[0, 1]$ 区间内，表示 $x$ 来自真实数据分布 $p_{data}$ 的概率。判别器的目标是尽可能准确地将真实数据（$D(x) \to 1$）与生成数据（$D(G(z)) \to 0$）区分开。

### 1.2 核心思想
- **对抗性训练（Adversarial Training）：**
  G和D构成一个动态的“极小极大博弈”（Minimax Game）。训练过程是交替进行的，一方的参数在更新时，另一方的参数保持固定。
- **D的目标：** 判别器的目标是最大化其判别准确率，即最大化目标函数 $V(G, D)$。它希望给真实数据高分（接近1），给生成数据低分（接近0）。
  $$ \max_D V(D, G) = \mathbb{E}_{x \sim p_{data}(x)}[\log D(x)] + \mathbb{E}_{z \sim p_z(z)}[\log(1 - D(G(z)))] $$
- **G的目标：** 生成器的目标与判别器相反，它希望最小化判别器将其识破的概率，即最小化目标函数 $V(G, D)$。它希望自己生成的假数据被判别器给予高分（$D(G(z)) \to 1$），这样 $\log(1 - D(G(z)))$ 就会变得非常小。
  $$ \min_G V(D, G) = \mathbb{E}_{x \sim p_{data}(x)}[\log D(x)] + \mathbb{E}_{z \sim p_z(z)}[\log(1 - D(G(z)))] $$

## 2. 数学推导与关键公式

### 2.1 最优判别器 $D^*(x)$
对于一个固定的生成器G，我们可以推导出最优的判别器 $D^*(x)$。目标函数可以写成积分形式：
$$ V(D, G) = \int_x p_{data}(x) \log(D(x)) dx + \int_z p_z(z) \log(1 - D(G(z))) dz $$
通过变量替换 $x = G(z)$，右半部分可以改写为关于 $p_g(x)$ 的积分：
$$ V(D, G) = \int_x [p_{data}(x) \log(D(x)) + p_g(x) \log(1 - D(x))] dx $$
为了求得使上式最大化的 $D(x)$，我们可以对被积函数 $f(y) = a \log(y) + b \log(1-y)$ （其中 $a=p_{data}(x), b=p_g(x), y=D(x)$）求导，并令导数为0。
$$ \frac{df}{dy} = \frac{a}{y} - \frac{b}{1-y} = 0 \implies a(1-y) = by \implies y = \frac{a}{a+b} $$
因此，对于固定G，最优判别器为：
$$ D^*(x) = \frac{p_{data}(x)}{p_{data}(x) + p_g(x)} $$
这个公式直观地表明，当一个样本 $x$ 处的真实数据密度 $p_{data}(x)$ 远大于生成数据密度 $p_g(x)$ 时，最优判别器会给出接近1的概率，反之则给出接近0的概率。

### 2.2 生成器的全局最优条件
当且仅当生成分布与真实分布完全一致，即 $p_g = p_{data}$ 时，整个系统达到纳什均衡。此时，对于任何样本 $x$，都有 $p_{data}(x) = p_g(x)$，代入最优判别器的公式：
$$ D^*(x) = \frac{p_{data}(x)}{p_{data}(x) + p_{data}(x)} = \frac{1}{2} \quad \forall x $$
这意味着判别器无法从任何样本中区分其来源，只能随机猜测，其输出恒为0.5。此时生成器达到了最优状态。

### 2.3 目标函数的散度解释
将最优判别器 $D^*(x)$ 代入生成器的目标函数 $C(G) = \max_D V(D, G)$ 中：
$$ C(G) = \mathbb{E}_{x \sim p_{data}}\left[\log \frac{p_{data}(x)}{p_{data}(x) + p_g(x)}\right] + \mathbb{E}_{x \sim p_g}\left[\log \frac{p_g(x)}{p_{data}(x) + p_g(x)}\right] $$
通过一些变换，可以将其与KL散度和JS散度联系起来。令 $M = \frac{p_{data}+p_g}{2}$，则上式可以写为：
$$ C(G) = KL\left(p_{data} \left\| \frac{p_{data}+p_g}{2}\right.\right) + KL\left(p_g \left\| \frac{p_{data}+p_g}{2}\right.\right) - \log(4) $$
这正好是两倍的**Jensen-Shannon散度 (JSD)** 减去一个常数 $\log(4)$：
$$ C(G) = 2 \cdot JSD(p_{data}\|p_g) - \log(4) $$
因此，原始GAN的训练目标，当判别器最优时，等价于最小化真实分布 $p_{data}$ 与生成分布 $p_g$ 之间的JSD。

## 3. 训练过程详解

### 3.1 训练步骤
实际训练中，我们无法得到完美的 $D^*$，而是通过有限步的梯度下降来交替优化G和D。
1. **初始化**: 随机初始化生成器G和判别器D的权重参数 $\theta_g$ 和 $\theta_d$。
2. **交替优化**: 在每个训练迭代中：
   - **训练D（固定G）**: 通常会训练D多步（k步，k≥1），以确保它接近最优状态。
     - 从真实数据集中采样一个大小为 $m$ 的小批量（mini-batch）样本 $\{x^{(1)}, ..., x^{(m)}\}$。
     - 从噪声分布 $p_z$ 中采样一个同样大小的小批量噪声 $\{z^{(1)}, ..., z^{(m)}\}$。
     - 通过生成器生成假数据 $\{G(z^{(1)}), ..., G(z^{(m)})\}$。
     - 计算D的损失函数，并通过梯度上升（或最小化负损失）来更新D的参数 $\theta_d$：
     $$ \nabla_{\theta_d} \frac{1}{m} \sum_{i=1}^m \left[ \log D(x^{(i)}) + \log(1-D(G(z^{(i)}))) \right] $$
   - **训练G（固定D）**:
     - 重新从噪声分布 $p_z$ 中采样一个小批量噪声 $\{z^{(1)}, ..., z^{(m)}\}$。
     - 计算G的损失。原始的损失函数是 $\mathcal{L}_G = \mathbb{E}_{z \sim p_z}[\log(1-D(G(z)))]$。但在训练初期，G生成的样本质量很差，$D(G(z))$ 接近0，导致 $\log(1-D(G(z)))$ 梯度很小（梯度消失），G难以学习。因此，实践中通常使用一个改进的、非饱和的损失函数：
     $$ \mathcal{L}_G = -\frac{1}{m} \sum_{i=1}^m \log(D(G(z^{(i)}))) $$
     这个损失函数在 $D(G(z))$ 接近0时能提供更强的梯度信号。
     - 通过梯度下降更新G的参数 $\theta_g$：
     $$ \nabla_{\theta_g} \frac{1}{m} \sum_{i=1}^m \log(D(G(z^{(i)}))) $$
     注意这里是最小化损失，所以是梯度下降，等价于最大化 $D(G(z))$。

### 3.2 训练动态
- **初期**: G是随机的，生成的样本毫无意义。D很容易学会区分真实和虚假数据，其损失迅速下降，准确率接近100%。G的梯度信号较强，开始学习基本的数据结构。
- **中期**: 随着G的进步，它生成的样本开始模仿真实数据的某些特征。D的任务变得更加困难，其准确率会从100%下降。G和D在此阶段进行激烈的对抗学习。
- **理想收敛**: G完全捕捉到了真实数据的分布（$p_g = p_{data}$）。D再也无法区分真假样本，其输出对所有输入都趋于0.5，准确率降至50%。此时系统达到纳什均衡。然而在实践中，达到完美的均衡非常困难。

## 4. 数据流与张量维度详解
我们以生成 $64 \times 64$ 的3通道彩色图片为例，假设批处理大小（batch_size）为 $B$，噪声向量维度为 $Z$。
1.  **生成器（G）输入**:
    - 从标准正态分布中采样一个噪声批次 `noise`。
    - **张量维度**: `(B, Z, 1, 1)`。通常将噪声向量表示为 $1 \times 1$ 的“图像”，以便于使用转置卷积网络进行上采样。
2.  **生成器（G）内部（以DCGAN为例）**:
    - G通常由一系列转置卷积层（`ConvTranspose2d`）构成，每层后面跟着批归一化（`BatchNorm`）和ReLU激活函数。
    - `(B, Z, 1, 1)` -> `ConvT1` -> `(B, C1, 4, 4)` -> `ConvT2` -> `(B, C2, 8, 8)` -> `ConvT3` -> `(B, C3, 16, 16)` -> `ConvT4` -> `(B, C4, 32, 32)` -> `ConvT5` -> `(B, 3, 64, 64)`。
    - 最后一层通常使用 `Tanh` 激活函数，将输出像素值归一化到 $[-1, 1]$ 范围。
3.  **生成器（G）输出**:
    - 生成的假图片批次 `fake_data`。
    - **张量维度**: `(B, 3, 64, 64)`。
4.  **判别器（D）输入**:
    - 训练D时，其输入是两个数据源的拼接或交替：
      - **真实数据**: 从数据加载器中获取的真实图片 `real_data`。**张量维度**: `(B, 3, 64, 64)`。
      - **虚假数据**: 来自生成器的 `fake_data`。**张量维度**: `(B, 3, 64, 64)`。
5.  **判别器（D）内部（以DCGAN为例）**:
    - D通常由一系列标准卷积层（`Conv2d`）构成，每层后面跟着批归一化（`BatchNorm`）和 `LeakyReLU` 激活函数。LeakyReLU可以防止稀疏梯度。
    - `(B, 3, 64, 64)` -> `Conv1` -> `(B, C1, 32, 32)` -> `Conv2` -> `(B, C2, 16, 16)` -> `Conv3` -> `(B, C3, 8, 8)` -> `Conv4` -> `(B, C4, 4, 4)` -> `Conv5` -> `(B, 1, 1, 1)`。
    - 最后一层卷积将特征图压缩成一个单一的值，然后通过 `Sigmoid` 激活函数将其映射到 `[0, 1]` 区间，代表“真实”的概率。
6.  **判别器（D）输出**:
    - 对于 `real_data` 的输出 `d_real_output`。**张量维度**: `(B, 1, 1, 1)`，可被压缩为 `(B, 1)`。
    - 对于 `fake_data` 的输出 `d_fake_output`。**张量维度**: `(B, 1, 1, 1)`，可被压缩为 `(B, 1)`。
7.  **损失计算**:
    - **D的损失**:
      - `loss_real = BCELoss(d_real_output, real_labels)`，其中 `real_labels` 是一个全为1的张量，维度为 `(B, 1)`。
      - `loss_fake = BCELoss(d_fake_output, fake_labels)`，其中 `fake_labels` 是一个全为0的张量，维度为 `(B, 1)`。
      - `loss_D = loss_real + loss_fake`。
    - **G的损失**:
      - `loss_G = BCELoss(d_fake_output, real_labels)`。这里我们欺骗G，让它以“真实标签”（全1）为目标来优化，这等价于非饱和损失 $-\log(D(G(z)))$。

## 5. 关键散度分析

### 5.1 KL散度（Kullback-Leibler Divergence）
- 定义：衡量两个概率分布 $P$ 和 $Q$ 之间差异的非对称度量。
$$ KL(P\|Q) = \int P(x) \log \frac{P(x)}{Q(x)} dx = \mathbb{E}_{x \sim P} \left[ \log \frac{P(x)}{Q(x)} \right] $$
- 特点：
  - 非对称性：$KL(P\|Q) \neq KL(Q\|P)$。
  - $KL(P\|Q) \ge 0$，当且仅当 $P=Q$ 时取等号。
- 缺陷：当 $P(x) > 0$ 而 $Q(x) \to 0$ 的区域，$\log \frac{P(x)}{Q(x)}$ 会趋于无穷大，导致KL散度值爆炸。更严重的是，如果两个分布的支撑集（support）没有重叠或者重叠部分测度为零，KL散度是无定义的或为无穷大，这在GAN训练初期非常常见。

### 5.2 JS散度（Jensen-Shannon Divergence）
- 定义：KL散度的一个对称、平滑版本。
$$ JSD(P\|Q) = \frac{1}{2} KL \left( P\| M \right) + \frac{1}{2} KL \left( Q \left\| M \right) \right) $$
其中 $M = \frac{P+Q}{2}$ 是 $P$ 和 $Q$ 的混合分布。
- 特点：
  - 对称性：$JSD(P\|Q) = JSD(Q\|P)$。
  - 有界性：取值范围为 $[0, \log 2]$。
  - 解决了KL散度的非对称性和无穷大问题。
- 局限性：当两个分布 $P$ 和 $Q$ 的支撑集几乎没有重叠时，JSD会饱和到一个常数 $\log 2$。此时，其梯度会趋近于0，导致生成器无法接收到有效的学习信号，即**梯度消失**。这正是原始GAN训练不稳定的一个核心理论原因。

### 5.3 Wasserstein距离（又称Earth-Mover's Distance）
- 定义：衡量将一个分布 $P$ “搬运”成另一个分布 $Q$ 所需的最小“成本”。
$$ W(P, Q) = \inf_{\gamma \in \Pi(P, Q)} \mathbb{E}_{(x, y) \sim \gamma} [\| x - y \|] $$
其中 $\Pi(P, Q)$ 是所有以 $P$ 和 $Q$ 为边缘分布的联合分布 $\gamma(x, y)$ 的集合。直观上，$\gamma(x, y)$ 表示从 $x$ 点搬运多少“泥土”到 $y$ 点，而 $\|x-y\|$ 是搬运单位泥土的距离。
- 特点：
  - 即使两个分布没有重叠，Wasserstein距离仍然能提供一个有意义的、平滑的度量，反映了它们之间的距离。
  - 这意味着即使在训练初期，G也能获得平滑且有效的梯度，从而大大提高了训练的稳定性。
- WGAN改进：
  - WGAN的作者利用Kantorovich-Rubinstein对偶性将难以直接计算的Wasserstein距离转化为一个更易于优化的形式：
  $$ W(p_{data}, p_g) = \sup_{\|f\|_L \le 1} \mathbb{E}_{x \sim p_{data}}[f(x)] - \mathbb{E}_{x \sim p_g}[f(x)] $$
  - 这里的 $f$ 是一个1-Lipschitz函数。WGAN用一个神经网络（称为**评论家Critic**，取代了判别器）来近似这个 $f$。
  - WGAN不再有Sigmoid输出和对数损失，其目标函数变为：
    - Critic Loss: $\mathcal{L}_C = \mathbb{E}_{x \sim p_g}[C(x)] - \mathbb{E}_{x \sim p_{data}}[C(x)]$
    - Generator Loss: $\mathcal{L}_G = - \mathbb{E}_{x \sim p_g}[C(x)]$
  - 为了强制满足Lipschitz约束，原版WGAN采用了简单的权重裁剪（weight clipping），但这可能导致问题。后续的WGAN-GP则引入了更优的梯度惩罚。

## 6. 实际挑战与解决方案

### 6.1 模式坍塌（Mode Collapse）
- **现象**：生成器G发现只要生成少数几个高质量、高多样性的样本就能很好地骗过当前的判别器D，于是它就停止探索其他可能性，导致生成样本的多样性急剧下降，例如，在MNIST数据集上只生成数字“1”。
- **原因**：GAN的优化目标是让 $p_g$ 逼近 $p_{data}$，但如果G找到了一个局部最优解（能骗过当前D的捷径），它就没有动力去覆盖 $p_{data}$ 的所有模式。
- **解决方案**：
  - **小批量判别（Mini-batch Discrimination）**: 让判别器在判断一个样本时，可以参考同一小批量中的其他样本。如果生成器产出的样本彼此之间过于相似，判别器就能轻易发现它们是假的，从而迫使生成器提高多样性。
  - **Unrolled GANs**: 在更新生成器时，向前“展开”判别器的几步更新，让生成器预判到判别器接下来的动向，从而避免陷入只生成单一模式的陷阱。
  - **使用Wasserstein距离**: WGAN在理论上和实践中都能有效缓解模式坍塌问题。

### 6.2 训练不稳定
- **表现**：D过强，导致G的梯度消失，学习停滞；或者G的更新破坏了D的学习，导致损失函数剧烈震荡，无法收敛。
- **改进方法**：
  - **Wasserstein GAN (WGAN)**：如前所述，使用Wasserstein距离作为损失函数，提供了更平滑的梯度。
  - **梯度惩罚（Gradient Penalty, WGAN-GP）**: 这是对WGAN的改进，用梯度惩罚替代了权重裁剪来强制Lipschitz约束。它通过惩罚评论家（Critic）梯度范数偏离1的情况来实现。具体做法是：
    - 在真实样本 $x_{data}$ 和生成样本 $x_g$ 之间进行线性插值，得到新的样本点 $\hat{x} = \epsilon x_{data} + (1-\epsilon) x_g$，其中 $\epsilon \sim U[0, 1]$。
    - 计算评论家在这些插值点上的梯度范数，并将其加入到评论家的损失函数中：
    $$ \text{Penalty} = \lambda \mathbb{E}_{\hat{x} \sim p_{\hat{x}}} [(\|\nabla_{\hat{x}} C(\hat{x})\|_2 - 1)^2] $$
    其中 $\lambda$ 是惩罚系数（通常设为10）。这使得评论家在整个样本空间中都近似满足1-Lipschitz约束，极大地稳定了训练。

### 6.3 评估指标
评估GAN的生成质量是一个开放性问题，因为“好”是主观的。常用的量化指标有：
- **Inception Score (IS)**:
  $$ IS = \exp\left(\mathbb{E}_{x \sim p_g} KL(p(y|x) \| p(y))\right) $$
  - $p(y|x)$ 是给定生成图像 $x$，一个预训练的Inception-v3网络输出的类别概率分布。如果图像质量高、内容清晰，这个分布应该很“尖锐”（低熵）。
  - $p(y) = \int p(y|x) dx$ 是所有生成图像的平均类别分布。如果生成样本多样性好，这个分布应该很“平坦”（高熵）。
  - IS同时衡量生成图像的**清晰度**和**多样性**，分数越高越好。
- **Frechet Inception Distance (FID)**:
  $$ FID = \|\mu_{data} - \mu_g\|^2 + \text{Tr}(\Sigma_{data} + \Sigma_g - 2(\Sigma_{data}\Sigma_g)^{1/2}) $$
  - FID通过比较真实图像和生成图像在Inception-v3网络某一层的激活特征的统计数据（均值 $\mu$ 和协方差 $\Sigma$）来衡量相似度。
  - 它被认为比IS更稳健，更能反映人眼的感知相似度。FID分数越低，表示生成分布与真实分布越接近，效果越好。

## 7. 实例与伪代码

### 7.1 硬币分布的JSD计算
- **真实分布** $P$: 公平硬币， $P(\text{正面})=0.5, P(\text{反面})=0.5$。
- **生成分布** $Q$: 偏置硬币， $Q(\text{正面})=0.9, Q(\text{反面})=0.1$。
- **中间分布** $M$: $M(\text{正面}) = \frac{0.5+0.9}{2}=0.7, M(\text{反面}) = \frac{0.5+0.1}{2}=0.3$。
- **计算KL散度**:
  - $KL(P\|M) = 0.5 \log(\frac{0.5}{0.7}) + 0.5 \log(\frac{0.5}{0.3}) \approx 0.087$
  - $KL(Q\|M) = 0.9 \log(\frac{0.9}{0.7}) + 0.1 \log(\frac{0.1}{0.3}) \approx 0.149$
- **JSD结果**：
  $$ JSD(P\|Q) = \frac{1}{2}(KL(P\|M) + KL(Q\|M)) \approx \frac{1}{2}(0.087 + 0.149) \approx 0.118 $$

### 7.2 伪代码示例（基于PyTorch）
```python
# G和D是神经网络模型, optimizer_G和optimizer_D是优化器
# z_dim是噪声向量维度, k_steps是每次迭代中D的训练步数
for epoch in range(epochs):
    for real_data in dataloader:
        batch_size = real_data.size(0)

        # ---------------------
        #  训练判别器 D
        # ---------------------
        for _ in range(k_steps):
            optimizer_D.zero_grad()

            # 真实数据的损失
            real_labels = torch.ones(batch_size, 1)
            d_real_output = D(real_data)
            loss_D_real = binary_cross_entropy(d_real_output, real_labels)

            # 虚假数据的损失
            noise = torch.randn(batch_size, z_dim, 1, 1)
            fake_data = G(noise)
            fake_labels = torch.zeros(batch_size, 1)
            # 使用.detach()防止梯度回传到G
            d_fake_output = D(fake_data.detach())
            loss_D_fake = binary_cross_entropy(d_fake_output, fake_labels)

            # 总损失
            loss_D = loss_D_real + loss_D_fake
            loss_D.backward()
            optimizer_D.step()

        # -----------------
        #  训练生成器 G
        # -----------------
        optimizer_G.zero_grad()

        # 我们希望D将G生成的假数据判断为真
        # 因此使用真实标签(全1)计算G的损失
        noise = torch.randn(batch_size, z_dim, 1, 1)
        fake_data = G(noise)
        d_fake_output_for_G = D(fake_data)
        loss_G = binary_cross_entropy(d_fake_output_for_G, real_labels)

        loss_G.backward()
        optimizer_G.step()
```

## 8. 总结与对比

### 8.1 散度对比

| 散度类型 | 对称性 | 梯度特性 | 适用场景与问题 |
| :--- | :--- | :--- | :--- |
| **KL散度** | 非对称 | 对模式遗漏敏感（zero-forcing） | 变分自编码器（VAE）。若分布不重叠则梯度无穷大。 |
| **JS散度** | 对称 | 分布不重叠时梯度消失 | 原始GAN的理论基础。导致训练不稳定和模式坍塌。 |
| **Wasserstein距离** | 对称 | 即使分布不重叠，梯度依然平滑有效 | WGAN、WGAN-GP。显著提高训练稳定性，缓解模式坍塌。 |

### 8.2 GAN的优缺点
- **优点：**
  - 无需显式建模概率密度函数 $p(x)$，可以学习非常复杂和高维的分布。
  - 能够生成质量极高、非常逼真的样本，在图像生成领域达到SOTA。
  - 属于半监督学习的强大框架，判别器学到的特征可用于其他任务。
- **缺点：**
  - **训练不稳定**：需要精心设计网络架构和调整超参数，否则G和D的博弈容易崩溃。
  - **模式坍塌（Mode Collapse）**：生成器可能只学习到数据分布的少数几个模式，缺乏多样性。
  - **评估困难**：缺乏客观、统一的评估指标来衡量生成样本的质量和多样性。

**结语：** GAN通过开创性的对抗博弈思想，在数据生成领域取得了革命性突破。其核心在于生成器和判别器之间的动态平衡，而这一平衡的数学本质是最小化两个概率分布之间的散度。理解从JS散度到Wasserstein距离的演进，以及梯度惩罚等工程技巧，是掌握和应用现代GAN、克服其固有挑战的关键所在。