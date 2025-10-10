---
type: concept-note
tags:
  - cv
  - image-classification
  - resnet
  - dl-architecture
  - residual-learning
  - backbone
status: done
model: ResNet
year: 2015
---
学习资料：[你必须要知道CNN模型：ResNet - 知乎](https://zhuanlan.zhihu.com/p/31852747)

[【深度学习】ResNet网络讲解-CSDN博客](https://blog.csdn.net/weixin_44001371/article/details/134192776?ops_request_misc=%7B%22request%5Fid%22%3A%22f039634669097fe290b450e22d7f0012%22%2C%22scm%22%3A%2220140713.130102334.pc%5Fall.%22%7D&request_id=f039634669097fe290b450e22d7f0012&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~first_rank_ecpm_v1~hot_rank-4-134192776-null-null.142^v102^pc_search_result_base3&utm_term=ResNet&spm=1018.2226.3001.4187)

[深度学习笔记（七）--ResNet（残差网络）-CSDN博客](https://blog.csdn.net/qq_29893385/article/details/81207203?ops_request_misc=%7B%22request%5Fid%22%3A%22f039634669097fe290b450e22d7f0012%22%2C%22scm%22%3A%2220140713.130102334.pc%5Fall.%22%7D&request_id=f039634669097fe290b450e22d7f0012&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~first_rank_ecpm_v1~hot_rank-3-81207203-null-null.142^v102^pc_search_result_base3&utm_term=ResNet&spm=1018.2226.3001.4187)

[深入浅出读懂ResNet原理与实现_resnet-18原理-CSDN博客](https://blog.csdn.net/zhoumoon/article/details/105103557)

[7.6. 残差网络（ResNet） — 动手学深度学习 2.0.0 documentation](https://zh-v2.d2l.ai/chapter_convolutional-modern/resnet.html#fig-functionclasses)

------

ResNet（残差网络）是由何恺明等人在2015年提出的深度卷积神经网络架构，它通过引入**残差学习**（Residual Learning）解决了深层网络训练中的梯度消失和网络退化问题，使得训练数百甚至上千层的网络成为可能。以下是对ResNet的详细解析：

### **1. 背景与核心问题**
- **深层网络的困境**：传统神经网络随着层数加深，容易出现梯度消失/爆炸，导致训练困难。尽管通过归一化（如BatchNorm）缓解了梯度问题，但实验表明，深层网络的性能反而可能比浅层更差（即“网络退化”）。
- **残差学习的提出**：ResNet提出让网络直接学习**残差函数**（目标输出与输入的差值），而非直接学习原始目标函数，使深层网络更容易优化。

---

### **2. 残差块（Residual Block）**
ResNet的核心是**残差块**，结构如下：

#### **基本残差块（Basic Block）**
- 包含两个3x3卷积层，每个卷积后接BatchNorm和ReLU激活。
- **跳跃连接（Shortcut Connection）**：将输入直接绕过卷积层，与输出相加。
- 数学表达：  
  $$
  \mathbf{y} = \mathcal{F}(\mathbf{x}, \{W_i\}) + \mathbf{x}
  $$
  其中，\($x$\)为输入，\($F$\)为残差函数，\($y$\)为输出。

#### **瓶颈残差块（Bottleneck Block）**
- 针对更深的网络（如ResNet-50/101/152），减少计算量。
- 结构：1x1卷积（降维）→ 3x3卷积 → 1x1卷积（升维）。
- 跳跃连接可能包含1x1卷积调整维度。

---

### **3. ResNet网络架构**
ResNet通过堆叠残差块构建不同深度的网络，常见版本如下：

| 网络名称   | 层数 | 残差块类型       | 阶段配置（层数） |
| ---------- | ---- | ---------------- | ---------------- |
| ResNet-18  | 18   | Basic Block      | [2, 2, 2, 2]     |
| ResNet-34  | 34   | Basic Block      | [3, 4, 6, 3]     |
| ResNet-50  | 50   | Bottleneck Block | [3, 4, 6, 3]     |
| ResNet-101 | 101  | Bottleneck Block | [3, 4, 23, 3]    |
| ResNet-152 | 152  | Bottleneck Block | [3, 8, 36, 3]    |

- **整体流程**：
  1. **输入层**：7x7卷积（步长2） + 3x3最大池化，快速降采样。
  2. **四个阶段**：每个阶段由多个残差块组成，逐步缩小空间尺寸、增加通道数。
  3. **全局平均池化 + 全连接层**：输出分类结果。

---

### **4. 关键技术细节**
#### **跳跃连接处理**
- **维度匹配**：当残差块的输入和输出维度不一致时（如通道数增加或空间尺寸缩小），跳跃连接通过**1x1卷积 + BatchNorm**调整维度。
- **恒等映射（Identity Mapping）**：若输入输出维度相同，直接相加。

#### **下采样方式**
- 每个阶段的第一个残差块通过**步长为2的卷积**降低特征图尺寸。

#### **初始化与激活函数**
- 权重初始化采用He初始化（适应ReLU）。
- 每个卷积层后接BatchNorm和ReLU，**最后一个ReLU在相加后执行**。

---

### **5. 为什么ResNet有效？**
1. **缓解梯度消失**：跳跃连接允许梯度直接回传到浅层，改善反向传播。
2. **学习残差更简单**：若恒等映射是最优解，残差函数只需逼近零，比直接学习非线性映射更容易。
3. **隐式深度监督**：残差结构使网络在多个路径上同时学习，类似集成学习。

------

# 疑问解答

---

### **1. 残差块的基本公式详解**

#### **公式表达**
残差块的核心公式为：  
$$
\mathbf{y} = \mathcal{F}(\mathbf{x}, \{W_i\}) + \mathbf{x}
$$

- **符号解释**：
  - $f(x)$：输入特征图（残差块的输入）。
  - $\mathcal{F}(\mathbf{x}, \{W_i\}$：经过多层卷积和非线性激活后的输出（残差函数）。
  - $\{W_i\}$：卷积层的权重参数。
  - $\mathbf{y}$：最终输出（输入与残差的和）。

#### **物理意义**
- **直接映射**：跳跃连接$\mathbf{x}$将原始输入直接传递到输出端。
- **残差学习**：网络只需学习输出与输入的差值 $\mathcal{F} = \mathbf{y} - \mathbf{x}$，而非直接拟合复杂的 $\mathbf{y}$。
- **梯度流动**：反向传播时，梯度可以通过跳跃连接直接传递到浅层，避免因链式求导导致的梯度消失。

#### **公式的代码实现**
```python
# 以PyTorch为例的残差块前向传播
def forward(x):
    identity = x  # 保存输入
    out = conv1(x)
    out = bn1(out)
    out = relu(out)
    out = conv2(out)
    out = bn2(out)
    out += identity  # 跳跃连接相加
    out = relu(out)
    return out
```

---

### **2. 瓶颈残差块（Bottleneck Block）中1x1卷积的作用**

#### **结构分析**
瓶颈块的结构为：  
**1x1卷积（降维） → 3x3卷积（特征提取） → 1x1卷积（升维）**  

1. **降维（第一个1x1卷积）**：
   - 输入通道数为\(C\)，通过1x1卷积将通道数压缩为\(C/4\)（例如ResNet-50中）。
   - **作用**：减少后续3x3卷积的计算量（计算量与通道数平方相关）。
   - **为什么能这么做**：1x1卷积可在不改变空间尺寸的前提下调整通道数，相当于“通道间的线性组合”。

2. **升维（最后一个1x1卷积）**：
   - 将通道数从\(C/4\)恢复为\(C\)，以便与跳跃连接的输入相加。
   - **作用**：恢复通道数，保证维度匹配。

#### **计算量对比**
假设输入为$(C \times H \times W)$，3x3卷积的计算量为：
- **Basic Block**：$2 \times (3^2 C^2 H W)$。
- **Bottleneck Block**：$(1^2 C (C/4) H W + 3^2 (C/4)^2 H W + 1^2 (C/4) C H W = \frac{3}{4} C^2 H W)$。  
**结论**：Bottleneck Block的计算量仅为Basic Block的约1/3，适合更深的网络。

---

### **3. 下采样时为何增加通道数？如何实现通道数匹配？**

#### **为什么下采样要增加通道数？**
1. **信息密度补偿**：
   - 下采样（如步长2的卷积）会减少特征图尺寸（如从$(H \times W)$变为$(H/2 \times W/2)$，空间信息减少。
   - 增加通道数（如从64到128）可保留更多抽象特征，维持网络的表达能力。
   
2. **深层网络的特性**：
   - 深层网络需要更高级的语义特征（如物体的部件、整体类别），增加通道数能容纳更多特征检测器。

#### **如何实现通道数匹配？**
当残差块的输入和输出维度不一致时（如下采样导致尺寸变化或通道数增加），通过两种方式调整跳跃连接：

1. **1x1卷积调整**：
   - 在跳跃连接中加入1x1卷积层，将输入通道数和空间尺寸调整为与输出一致。
   - 例如：输入为$(C \times H \times W)$，输出为$(2C \times H/2 \times W/2)$，跳跃连接的1x1卷积使用步长2，输出通道数设为\(2C\)。

2. **代码示例**：
```python
class BottleneckBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        # 主路径
        self.conv1 = nn.Conv2d(in_channels, out_channels//4, kernel_size=1, stride=stride)
        self.conv2 = nn.Conv2d(out_channels//4, out_channels//4, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(out_channels//4, out_channels, kernel_size=1)
        # 跳跃连接
        self.shortcut = nn.Sequential()
        if stride !=1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        identity = self.shortcut(x)  # 调整维度
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = self.conv3(out)
        out += identity
        return F.relu(out)
```

#### **关键步骤**
- **步长为2的卷积**：主路径的第一个卷积（或池化）使用步长2，降低空间尺寸。
- **通道数倍增**：每个阶段（如ResNet的Stage 1到Stage 4）的第一个残差块会将通道数翻倍（如64→128→256→512）。
- **跳跃连接的1x1卷积**：同时调整通道数和空间尺寸，确保与主路径输出一致。

---

### 4. **1x1卷积的工作原理**

1x1卷积虽然核尺寸极小（仅覆盖单个像素），但其核心作用在于**通道维度的变换**，本质是**跨通道的线性组合**。以下是其实现机制和作用的详细解释：

---

#### 基本操作解析
假设输入特征图的尺寸为 **$(C_{in} \times H \times W)$**（通道数 × 高度 × 宽度）：
- **卷积核参数**：1x1卷积的每个核大小为 **$(1 \times 1 \times C_{in})$**，输出通道数为 **$(C_{out})$**。
- **计算过程**：
  - 对输入特征图的**每个空间位置（H×W中的每个点）**，1x1卷积核会遍历所有输入通道（$(C_{in})$个），进行加权求和。
  - 每个输出通道对应一个独立的1x1卷积核，最终生成 **$(C_{out} \times H \times W)$** 的输出。

**数学表达**：  
对于输出通道\(j\)的某个位置\((h,w)\)：  
$$
\text{Output}[j, h, w] = \sum_{i=1}^{C_{in}} W_{j,i} \cdot \text{Input}[i, h, w] + b_j
$$

- $$(W_{j,i})$$：第\(j\)个卷积核在输入通道\(i\)上的权重。
- $$(b_j)$$：偏置项（可选）。

---

#### **为什么是“通道间的线性组合”？**
- **通道融合**：1x1卷积的每个输出通道的值，是所有输入通道值的线性加权和。例如：
  - 若输入有3个通道（R, G, B），1x1卷积可将它们组合成新的通道（如 R+2G-B）。
- **参数意义**：权重矩阵 **$$(W \in \mathbb{R}^{C_{out} \times C_{in}}$$)** 决定了如何混合输入通道的信息。
- **类比全连接层**：1x1卷积等效于在每个空间位置（共$$(H \times W)$$个位置）独立执行一个全连接层，将$$(C_{in})$$维向量映射到$$(C_{out})$$维。

---

#### **如何调整通道数？**

通道数本质是“特征图的数量”，而***卷积核***的数量决定了提取多少种新特征。网络设计者可以根据需要增加（如64→128）或减少通道数。

- **增加通道数（升维）**：设置 $$(C_{out} > C_{in})$$，例如从256通道扩展到512通道。
- **减少通道数（降维）**：设置 $$(C_{out} < C_{in})$$，例如从512通道压缩到64通道。
- **保持通道数**：设置 $$(C_{out} = C_{in})$$，仅混合通道信息（类似通道注意力机制）。

**示例**：  
输入特征图尺寸为 **$$(256 \times 32 \times 32)$$**，使用 $$(C_{out} = 64)$$ 的1x1卷积：

- 输出尺寸为 **$$(64 \times 32 \times 32)$$**（通道数减少，空间尺寸不变）。
- 每个输出通道是256个输入通道的线性组合。

---

#### **为什么选择1x1卷积？**
1. **计算高效**：
   - 1x1卷积的参数量为 $$(C_{in} \times C_{out})$$，远小于3x3卷积（参数量 $$(3 \times 3 \times C_{in} \times C_{out})$$）。
   - 在ResNet的瓶颈块中，先用1x1卷积降维（例如从256→64），再进行3x3卷积，计算量大幅降低。

2. **灵活调整通道数**：
   - 无需改变特征图的空间尺寸（H和W），仅通过改变$$(C_{out})$$即可控制通道数。
   - 在跨层跳跃连接中，可快速对齐输入输出通道数（如ResNet的维度匹配）。

3. **引入非线性**：
   - 1x1卷积后通常接ReLU激活函数，增加非线性表达能力。

------

### 总结

1. **残差公式**：通过跳跃连接让网络学习残差，而非直接映射，优化更简单。
2. **1x1卷积**：1x1卷积通过**跨通道的加权求和**，实现了在不改变空间尺寸的前提下调整通道数。其本质是**对每个空间位置的所有通道进行线性组合**，权重由卷积核参数决定。这种操作既高效（参数量少）又灵活（可升维、降维或混合通道），成为现代深度网络设计的核心组件之一。
3. **下采样与通道数**：通过1x1卷积和步长调整，在减少空间尺寸的同时增加通道数，保持信息密度，确保输入输出维度匹配。

这些设计使得ResNet既能训练极深的网络，又避免了计算量爆炸和维度不匹配的问题。
