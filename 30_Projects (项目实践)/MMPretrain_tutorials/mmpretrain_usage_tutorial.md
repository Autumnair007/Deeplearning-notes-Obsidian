---
type: tutorial
tags:
  - cv
  - image-classification
  - mmpretrain
  - convnext
  - convnext-v2
  - full-supervision
  - finetuning
  - code-note
status: done
model: ConvNeXt-Tiny
year: 2023
---
学习资料：[学习配置文件 — MMPretrain 1.2.0 文档](https://mmpretrain.readthedocs.io/zh-cn/latest/user_guides/config.html)

[open-mmlab/mmpretrain: OpenMMLab Pre-training Toolbox and Benchmark](https://github.com/open-mmlab/mmpretrain/tree/main)

------

### MMPretrain 配置系统核心思想：积木与蓝图

想象一下，你要搭建一个复杂的乐高模型，比如一艘千年隼号。你不会从零开始制造每一块积木，而是会拿到一盒预制好的标准积木（轮子、方块、连接件等）和一张详细的**蓝图**。

在 `mmpretrain` 中：
*   **积木 (Components)**：就是深度学习中那些可复用的模块，比如一个 ResNet50 模型、AdamW 优化器、CIFAR10 数据集加载器等。这些都是预先写好、功能明确的代码。
*   **蓝图 (Config Files)**：就是我们一直在写的 `.py` 配置文件。它不包含复杂的逻辑代码，只负责一件事：**告诉程序该用哪些积木，以及如何把这些积木组装起来**。

这种设计的巨大优势是**解耦**和**复用**。模型代码的开发者不需要关心训练细节，而做实验的用户（比如我们）也不需要去修改复杂的模型源码，只需要修改“蓝图”即可。

---

### 配置文件结构深度解读：四大金刚

如教程所说，一个完整的深度学习实验被拆分成了四个核心部分，我称之为“四大金刚”。它们都住在 `configs/_base_/` 这个“积木仓库”里。

#### 1. 模型 (Model)：实验的“大脑”

这是整个实验的核心，定义了“我们要用什么网络，以及它如何计算对错”。

*   `model = dict(...)`：这是模型的总定义。
    *   `type='ImageClassifier'`：指定了任务类型。这不仅仅是一个名字，它对应着 `mmpretrain.models.classifiers.ImageClassifier` 这个类。程序看到这个 `type`，就会去实例化这个类，把 `model` 字典里其他的键值对作为参数传进去。这就是**注册器 (Registry)** 机制在起作用，它是 OpenMMLab 框架的灵魂。
    *   `backbone`：**主干网络**，负责从图片中提取特征。可以把它想象成动物的“眼睛和初级视觉皮层”。
        *   `type='ResNet'` 或 `type='ConvNeXt'`：同样，这会实例化对应的模型类。
        *   `depth=50`，`arch='tiny'`：这些是传给模型类的参数，用来控制模型的具体规格。
        *   `init_cfg=dict(type='Pretrained', ...)`：这是我们实践过的**初始化配置**。它告诉模型在实例化之后，要执行一个“预加载权重”的操作。
    *   `neck`：**颈部网络**，连接主干和头的中间件。它负责对主干网络输出的特征图进行“再加工”，使其更适合分类任务。
        *   `type='GlobalAveragePooling'`：这是一个非常常见的 `neck`，它把一个二维的特征图（比如 7x7x2048）通过取平均值，压缩成一个一维的特征向量（比如 1x2048）。这样，无论输入图片多大，最后得到的特征向量长度都是固定的。
    *   `head`：**头部网络**，实验的“决策者”。它接收来自 `neck` 的特征向量，并做出最终的分类判断。
        *   `type='LinearClsHead'`：表示这是一个简单的线性分类头，通常就是一个全连接层。
        *   `num_classes=10`：定义了输出的类别数，这是我们必须根据自己任务修改的地方。
        *   `in_channels=768`：定义了输入这个头的特征向量的维度。**这个值必须和 `neck` (或 `backbone`) 的输出维度严格匹配**，否则会报错。
        *   `loss=dict(type='CrossEntropyLoss', ...)`：定义了损失函数，即“如何衡量预测的对错”。交叉熵损失是分类任务最常用的损失函数。

#### 2. 数据 (Data)：实验的“食物”

这部分定义了模型训练所需的“原材料”从哪里来，以及如何进行预处理。

*   `train_dataloader`, `val_dataloader`, `test_dataloader`：这三个字典分别定义了训练、验证和测试时的数据加载器。
    *   `batch_size=128`：**批量大小**。表示每次给模型“喂”多少张图片。这个值越大，梯度计算越稳定，但对显存的消耗也越大。
    *   `num_workers=2`：**工作线程数**。表示用多少个子进程来预读取数据。这个值大于0可以显著加速数据加载，避免 GPU 等待。对于 Windows 系统，通常设为 2 或 4 比较稳定。
    *   `dataset=dict(...)`：这是数据加载器的核心，定义了数据集本身。
        *   `type='CIFAR10'`：指定了数据集的类型，程序会去找对应的 `CIFAR10Dataset` 类来处理。
        *   `data_prefix='data/cifar10'`：指定了数据集存放的路径。
        *   `pipeline=train_pipeline`：**数据处理流水线**，这是数据部分最重要、最灵活的地方。它是一个列表，定义了对每一张图片要执行的一系列操作。
            *   `dict(type='RandomCrop', ...)`：随机裁剪。
            *   `dict(type='RandomFlip', ...)`：随机翻转。
            *   这些操作都是为了**数据增强 (Data Augmentation)**，通过对原始图片进行微小的随机改动，来增加数据的多样性，从而提高模型的泛化能力，防止过拟合。
            *   `dict(type='PackInputs')`：这是流水线的**最后一环**，负责把处理好的图片数据和标签等信息打包成模型能够接受的格式。
*   `val_evaluator=dict(type='Accuracy', topk=(1, ))`：**评估器**。定义了在验证集上用什么指标来评价模型的好坏。`Accuracy` 表示计算分类准确率，`topk=(1, )` 表示我们只关心 Top-1 准确率。

#### 3. 训练策略 (Schedule)：实验的“教练”

这部分定义了如何“指导”模型进行学习，主要是关于优化器和学习率的配置。

*   `optim_wrapper=dict(...)`：**优化器包装器**。它对 PyTorch 的优化器做了一层封装，增加了更多功能，比如梯度裁剪、混合精度训练等。
    *   `optimizer=dict(type='AdamW', lr=4e-3, ...)`：定义了使用的**优化器**。`AdamW` 是目前训练 Transformer 类模型（包括 ConvNeXt）的标配。`lr` 是**学习率**，是训练中最重要的超参数之一，它控制了模型参数更新的步长。
*   `param_scheduler=[...]`：**参数调度器**。这是一个列表，可以组合多种学习率调整策略。
    *   `dict(type='LinearLR', ...)`：线性预热 (Warmup)。在训练刚开始的几个 epoch，让学习率从一个很小的值慢慢增长到我们设定的初始值（比如 `4e-3`）。这可以帮助模型在训练初期更稳定地收敛。
    *   `dict(type='CosineAnnealingLR', ...)`：余弦退火。在预热结束后，让学习率按照余弦函数的形状平滑地下降。这被证明是一种非常有效的学习率调整策略。
*   `train_cfg=dict(by_epoch=True, max_epochs=100, val_interval=1)`：**训练循环配置**。
    *   `max_epochs=100`：定义了总共训练多少个周期。
    *   `val_interval=1`：定义了每训练 1 个 epoch，就进行一次验证。

#### 4. 运行设置 (Runtime)：实验的“后勤保障”

这部分定义了所有与训练过程本身无关，但又必不可少的“杂项”，比如日志、断点续传、环境设置等。

*   `default_hooks=dict(...)`：**钩子 (Hooks)**。这是 MMEngine 的一个强大机制。你可以把 Hook 想象成一个个可以挂在训练流程中特定“挂点”（如“每个 epoch 开始前”、“每次迭代结束后”）的小插件，用来执行特定任务。
    *   `logger=dict(type='LoggerHook', ...)`：日志钩子，负责在控制台打印日志。
    *   `checkpoint=dict(type='CheckpointHook', ...)`：检查点钩子，负责保存模型权重。`save_best='auto'` 会自动根据验证集上的指标保存表现最好的模型。
    *   `timer=dict(type='IterTimerHook')`：计时器钩子，负责计算训练速度。
*   `load_from=None`：如果这里指定一个模型权重文件的路径，程序会在训练开始前加载它。这通常用于测试或微调。
*   `resume=False`：如果设为 `True`，程序会尝试从上一次中断的地方恢复训练，它不仅会加载模型权重，还会加载优化器状态、当前 epoch 数等。

### 继承与修改：配置文件的“魔法”

`_base_` 是 `mmpretrain` 配置系统最核心的特性。

*   **继承**：`_base_ = ['path/to/a.py', 'path/to/b.py']` 这行代码的意思是：“先把我需要的所有基础配置（积木）从 a.py 和 b.py 里加载进来。”
*   **修改/覆盖**：在 `_base_` 下面定义的任何变量，都会**覆盖**掉从 `_base_` 文件中继承来的同名变量。
    *   在我们的实践中，我们继承了 `convnext_v2/tiny.py`，但我们又在下面重新定义了 `model` 字典。这并不是把两个 `model` 合并，而是用我们自己的 `model` **完全替换**掉了基础配置里的 `model`。
    *   这种“深层覆盖”非常强大。比如，`_base_` 里的 `model` 定义了 `head=dict(num_classes=1000)`，我们在自己的文件里写 `model=dict(head=dict(num_classes=10))`，最终生效的就是 `num_classes=10`，而 `model` 里的其他部分（如 `backbone`）则保持不变。

通过这种“继承+覆盖”的模式，我们可以用极少的代码，灵活地组合和修改各种复杂的实验配置，就像我们刚刚做的那样。

希望这份超详细的解读能让你对 `mmpretrain` 的架构有一个更深刻、更全面的理解！这套设计理念是整个 OpenMMLab 系列项目的基石，一旦掌握，你就可以触类旁通，轻松上手其他工具库（如 MMDetection, MMSegmentation 等）。

---

### **实战演练：在 CIFAR-10 上训练 ConvNeXt-Tiny 模型**

#### **第零步：环境准备 (一次性操作)**

在开始之前，请确保你已经完成了这些基础设置。

1.  **安装 MMPretrain**：按照官方指南安装。
    
    ```bash
    conda create -n open-mmlab python=3.8 pytorch==1.10.1 torchvision==0.11.2 cudatoolkit=11.3 -c pytorch -y
    conda activate open-mmlab
    pip install openmim
    git clone https://github.com/open-mmlab/mmpretrain.git
    cd mmpretrain
    mim install -e .
    ```
    
    *(之后的所有命令，都默认你在此 `mmpretrain` 根目录下运行)*

#### **第一步：创建你的专属“蓝图” (配置文件)**

这是整个实验的核心。我们将创建一个新的配置文件，告诉程序我们要如何组合“积木”。

1.  **定位到配置文件夹**：
    进入 `configs/convnext/` 目录。这是存放所有 ConvNeXt 相关配置的地方。

2.  **创建新配置文件**：
    在此目录下，创建一个新的 Python 文件。为了清晰明了，我们给它起一个有意义的名字，比如：
    `my-convnext-tiny-cifar10.py`

3.  **编写配置文件内容**：
    用你的代码编辑器打开这个新文件，然后将下面这份我们之前已经调试好的、最终版本的代码**完整地复制**进去。

    ````python name=configs/convnext/convnext-tiny_cifar10_finetune.py
    # -------------------------------------------------------------------
    # 配置文件元信息 (Meta Information)
    # -------------------------------------------------------------------
    
    
    # --- 配置文件核心：继承 (Inheritance) ---
    # `_base_` 是一个列表，定义了此配置文件要继承的基础“蓝图”。
    # 程序会首先加载这些基础文件，然后用我们下面定义的配置去覆盖和修改它们。
    _base_ = [
        # 继承 ConvNeXt-V2 Tiny 版本的基础模型结构。
        # 这个文件里定义了模型的层、通道数等固有属性。
        '../_base_/models/convnext_v2/tiny.py',
    
        # 继承默认的运行设置。
        # 这个文件里包含了日志、钩子(hooks)、环境等通用的后勤保障配置。
        '../_base_/default_runtime.py'
    ]
    
    
    # --- 1. 模型修改 (Model) ---
    # `model` 是一个字典(dict)，包含了所有关于模型架构、权重和行为的指令。
    # 我们在这里定义的 `model` 会深度覆盖掉从 `_base_` 继承来的同名 `model` 配置。
    model = dict(
        # `backbone` 字典定义了模型的“主干网络”，即特征提取部分。
        backbone=dict(
            # `type` 指定了要实例化的主干网络类，这里是 'ConvNeXt'。
            type='ConvNeXt',
    
            # `arch` 是传给 ConvNeXt 类的一个参数，指定了模型的具体尺寸，这里是 'tiny'。
            arch='tiny',
    
            # `init_cfg` (Initialization Configuration) 字典定义了模型权重如何初始化。
            # 这是实现“迁移学习”的关键。
            init_cfg=dict(
                # `type` 指定初始化方式为 'Pretrained'，即加载预训练权重。
                type='Pretrained',
    
                # `checkpoint` 提供了预训练模型权重的下载链接。
                # 程序在启动时会自动从这个URL下载模型。
                checkpoint='https://download.openmmlab.com/mmclassification/v0/convnext-v2/convnext-v2-tiny_3rdparty-fcmae_in1k_20230104-80513adc.pth',
    
                # `prefix` 告诉程序，只加载 `checkpoint` 中键名以 'backbone' 开头的权重。
                # 这可以确保我们只初始化主干网络，而不会影响我们自定义的分类头。
                prefix='backbone',
            )
        ),
    
        # `head` 字典定义了模型的“分类头”，即决策部分。
        head=dict(
            # `num_classes` 指定了分类任务的类别数量，对于 CIFAR-10 来说是 10。
            num_classes=10,
    
            # `in_channels` 指定了输入到这个头的特征向量的维度。
            # 这个值必须与 `backbone` 的输出维度(ConvNeXt-Tiny是768)严格匹配。
            in_channels=768,
        ),
    
        # `train_cfg` (Training Configuration) 字典定义了模型在训练时的特定行为。
        train_cfg=dict(
            # `augments` 用于配置 MixUp, CutMix 等在批数据(batch)上进行的复杂数据增强。
            # 设置为 `None` 表示不使用它们，这对于在小数据集上微调是常见的做法。
            augments=None,
        ),
    )
    
    
    # --- 2. 数据集修改 (Dataset) ---
    # `dataset_type` 是一个字符串变量，方便我们在后面的配置中复用。
    dataset_type = 'CIFAR10'
    
    # `data_preprocessor` 字典定义了数据预处理器。
    # 它负责在数据进入模型前，进行归一化等批处理操作。
    data_preprocessor = dict(
        # 再次指定类别数，预处理器内部会用到。
        num_classes=10,
    
        # `mean` 是一个列表，包含了 CIFAR-10 数据集在R,G,B三个通道上的像素均值。
        # 用于图像归一化：(pixel - mean) / std。
        mean=[125.307, 122.961, 113.8575],
    
        # `std` 是一个列表，包含了 CIFAR-10 数据集在R,G,B三个通道上的像素标准差。
        std=[51.5865, 50.847, 51.255],
    
        # `to_rgb` 设置为 False，因为 CIFAR-10 数据集本身就是 RGB 顺序，不需要通道转换。
        to_rgb=False,
    )
    
    # `train_pipeline` 是一个列表(list)，定义了训练数据的处理流水线（包含数据增强）。
    # 列表中的每一个 `dict` 都是一个处理步骤。
    train_pipeline = [
        # 第一个步骤：随机裁剪。
        # `type` 指定了操作类型为 'RandomCrop'。
        dict(type='RandomCrop', crop_size=32, padding=4),
    
        # 第二个步骤：随机水平翻转。
        # `prob=0.5` 表示有 50% 的概率执行翻转。
        dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    
        # 最后一个步骤：打包输入。
        # 它会将处理好的图像和标签等信息打包成模型所需的格式。
        dict(type='PackInputs'),
    ]
    
    # `test_pipeline` 定义了验证和测试数据的处理流水线。
    # 通常不包含随机的数据增强操作，以保证评估结果的一致性。
    test_pipeline = [
        # 只进行打包操作。
        dict(type='PackInputs'),
    ]
    
    # `train_dataloader` 字典定义了训练数据加载器。
    train_dataloader = dict(
        # `batch_size` 定义了每个批次加载的样本数量。
        batch_size=128,
    
        # `num_workers` 定义了用于预读取数据的子进程数量，可以加速数据加载。
        num_workers=2,
    
        # `dataset` 字典定义了要加载的数据集。
        dataset=dict(
            # `type` 指定数据集类型，使用我们之前定义的 `dataset_type` 变量。
            type=dataset_type,
    
            # `data_prefix` 指定了数据集存放的路径。如果不存在，程序会自动下载到这里。
            data_prefix='data/cifar10',
    
            # `split` 指定使用数据集的哪一部分，这里是 'train'（训练集）。
            split='train',
    
            # `pipeline` 指定了要应用在这个数据集上的处理流水线。
            pipeline=train_pipeline,
        ),
    
        # `sampler` 字典定义了数据采样器。
        sampler=dict(
            # `type` 指定使用默认的采样器。
            type='DefaultSampler',
    
            # `shuffle=True` 表示在每个 epoch 开始前，都打乱训练数据的顺序。
            shuffle=True
        ),
    )
    
    # `val_dataloader` 字典定义了验证数据加载器。
    val_dataloader = dict(
        # 验证时没有反向传播，显存占用较小，可以使用和训练时相同的 `batch_size`。
        batch_size=128,
    
        # 工作线程数。
        num_workers=2,
    
        # 数据集配置。
        dataset=dict(
            # 数据集类型。
            type=dataset_type,
    
            # 数据集路径。
            data_prefix='data/cifar10',
    
            # `split` 指定使用 'test' 部分，CIFAR-10 官方将其作为验证集/测试集。
            split='test',
    
            # 应用测试流水线。
            pipeline=test_pipeline,
        ),
    
        # 采样器配置。
        sampler=dict(
            # 默认采样器。
            type='DefaultSampler',
    
            # `shuffle=False` 表示在验证时不打乱数据顺序，保证每次验证结果可比。
            shuffle=False
        ),
    )
    
    # `test_dataloader` 是一个快捷方式，让测试时直接使用和验证时完全相同的配置。
    test_dataloader = val_dataloader
    
    # `val_evaluator` 字典定义了验证评估器。
    val_evaluator = dict(
        # `type` 指定评估指标为 'Accuracy'（准确率）。
        type='Accuracy',
        
        # `topk` 是一个元组，指定计算 Top-k 准确率。`(1, )` 表示只计算 Top-1 准确率。
        topk=(1, )
    )
    
    # `test_evaluator` 让测试时也使用和验证时相同的评估器。
    test_evaluator = val_evaluator
    
    
    # --- 3. 训练策略修改 (Schedule) ---
    # `optim_wrapper` (Optimizer Wrapper) 字典定义了优化器及其包装器的配置。
    optim_wrapper = dict(
        # `optimizer` 字典定义了要使用的优化器。
        optimizer=dict(
            # `type` 指定使用 'AdamW' 优化器，它对 Transformer 架构模型效果很好。
            type='AdamW',
    
            # `lr` (Learning Rate) 是学习率，是训练中最重要的超参数之一。
            lr=4e-3,
    
            # `weight_decay` 是权重衰减，一种防止过拟合的正则化技术。
            weight_decay=0.05,
        ),
    
        # `paramwise_cfg` (Parameter-wise Configuration) 可以为不同类型的参数设置不同的优化策略。
        paramwise_cfg=dict(
            # `norm_decay_mult=0.0` 表示不对归一化层(Normalization)的参数进行权重衰减。
            norm_decay_mult=0.0,
    
            # `bias_decay_mult=0.0` 表示不对偏置(bias)参数进行权重衰减。
            bias_decay_mult=0.0,
        ),
        
        # `clip_grad` 字典配置了梯度裁剪，可以防止梯度爆炸，让训练更稳定。
        clip_grad=dict(max_norm=5.0),
    )
    
    # `param_scheduler` (Parameter Scheduler) 是一个列表，定义了学习率等参数的调整策略。
    param_scheduler = [
        # 第一个策略：线性预热 (Warmup)。
        dict(
            # `type` 指定策略类型为 'LinearLR'。
            type='LinearLR',
    
            # `start_factor` 表示学习率从 `lr * start_factor` 开始。
            start_factor=1e-3,
    
            # `by_epoch=True` 表示调度器的计数单位是 epoch。
            by_epoch=True,
    
            # `begin=0` 表示这个策略从第 0 个 epoch 开始。
            begin=0,
    
            # `end=10` 表示这个策略在第 10 个 epoch 结束。
            end=10,
        ),
    
        # 第二个策略：余弦退火。
        dict(
            # `type` 指定策略类型为 'CosineAnnealingLR'。
            type='CosineAnnealingLR',
    
            # 按 epoch 计数。
            by_epoch=True,
    
            # `begin=10` 表示这个策略紧接着上一个策略，在第 10 个 epoch 开始。
            begin=10,
    
            # `end=100` 表示这个策略在第 100 个 epoch 结束。
            end=100,
    
            # `eta_min` 表示学习率在退火结束时达到的最小值。
            eta_min=0.0,
        )
    ]
    
    # `train_cfg` 字典定义了训练循环的配置。
    train_cfg = dict(
        # `by_epoch=True` 表示训练的主循环是以 epoch 为单位的。
        by_epoch=True,
    
        # `max_epochs=100` 定义了总共训练 100 个 epoch。
        max_epochs=100,
    
        # `val_interval=1` 定义了每训练 1 个 epoch，就进行一次验证。
        val_interval=1
    )
    
    # `val_cfg` 定义了验证循环的配置，为空表示使用默认设置。
    val_cfg = dict()
    
    # `test_cfg` 定义了测试循环的配置，为空表示使用默认设置。
    test_cfg = dict()
    
    
    # --- 4. 运行设置修改 (Runtime) ---
    # `default_hooks` 字典定义了在训练过程中要使用的“钩子”(Hook)。
    # 我们在这里覆盖掉 `_base_` 中 `default_runtime.py` 的同名设置。
    default_hooks = dict(
        # `checkpoint` 钩子负责保存模型权重。
        checkpoint=dict(
            # `type` 指定钩子类型。
            type='CheckpointHook',
    
            # `interval=10` 表示每 10 个 epoch 保存一次模型。
            interval=10,
    
            # `save_best='auto'` 会自动监测验证集上的主要指标(如此处的Accuracy)，并保存最佳模型。
            save_best='auto'
        ),
    
        # `logger` 钩子负责在终端打印日志。
        logger=dict(
            # 钩子类型。
            type='LoggerHook',
    
            # `interval=50` 表示每 50 次迭代(iteration)打印一次日志。
            interval=50
        ),
    )
    
    # `randomness` 字典用于配置随机性，以保证实验的可复现性。
    randomness = dict(
        # `seed=0` 设置了全局的随机种子。
        seed=0
    )

#### **第二步：启动训练**

现在，你的“蓝图”已经准备就绪。回到你的终端（命令行），确保你仍然在 `mmpretrain` 的根目录下。

1.  **执行训练命令**：
    运行 `tools/train.py` 脚本，并将你的配置文件的路径作为参数传给它。

    ```bash
    # 使用单张 GPU 进行训练
    CUDA_VISIBLE_DEVICES=7 python tools/train.py configs/convnext_v2/my-convnext-tiny-cifar10.py
    ```

    *   如果你有多张 GPU，并且想使用它们加速训练，可以使用 `tools/dist_train.sh` 脚本 (Linux/macOS) 或 `tools/dist_train.ps1` (Windows PowerShell)。

2.  **观察初次运行的输出**：
    
    *   **自动下载数据集**：程序会检查 `data/cifar10` 目录，如果发现是空的，会自动从网上下载 CIFAR-10 数据集。
    *   **自动下载预训练模型**：程序会从配置文件中指定的 `checkpoint` 链接下载预训练的 ConvNeXt-Tiny 模型权重，并缓存到你的用户目录下。
    *   **开始训练**：下载完成后，你会看到熟悉的训练日志开始滚动，显示 `Epoch`, `loss`, `lr` 等信息。

#### **第三步：监控与结果**

训练开始后，你可以做以下事情：

1.  **实时监控**：观察终端输出的日志。
    *   **Loss**：你应该会看到 `loss` 值在第一个 epoch 内快速下降，之后缓慢下降。
    *   **Accuracy**：每个 epoch 结束后，程序会进行一次验证，并打印出类似 `val  Accuracy/top1: 85.34` 这样的日志，显示模型在验证集上的准确率。这个值应该会稳步提升。

2.  **找到训练产物**：
    所有的训练结果，包括保存的模型权重（`.pth` 文件）、日志文件（`.log` 文件）和配置文件快照，都会被保存在 `work_dirs` 目录下，一个以你配置文件名命名的文件夹里。
    *   **路径**：`mmpretrain/work_dirs/convnext-tiny_cifar10_finetune/`
    *   **最佳模型**：在该目录下，你会找到一个名为 `best_accuracy_top1_xxx.pth` 的文件。这就是整个训练过程中，在验证集上准确率最高的那个模型权重，也是你最终需要的成果。

通过这个流程，你不仅成功地跑通了一个实验，更重要的是，你亲手实践了 `mmpretrain` “配置驱动”的核心思想。
