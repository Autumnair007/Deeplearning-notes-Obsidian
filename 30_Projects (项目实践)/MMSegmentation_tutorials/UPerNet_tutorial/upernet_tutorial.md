---
type: tutorial
tags:
  - cv
  - semantic-segmentation
  - full-supervision
  - upernet
  - mmsegmentation
  - guide
  - cityscapes
  - pascal-voc
  - resnet
status: done
model: UPerNet
year: 2018
related_backbone: ResNet-50
summary: MMSegmentation框架下UPerNet模型的快速上手推理和基于PASCAL VOC 2012数据集的训练评估教程。重点讲解了如何从基于迭代(Iteration-based)的训练切换到基于周期(Epoch-based)的训练，以及如何实现断点恢复和多GPU训练。
---
使用 MMSegmentation 框架来复现和测试 UPerNet 是一个非常好的选择。MMSegmentation 是一个功能强大、标准化的语义分割工具箱，它让模型的测试、训练和部署变得更加简单和规范。

相比于原版仓库，MMSegmentation 的优势在于：

*   **环境更新**：支持更新的 PyTorch 和 CUDA 版本，环境配置更友好。
*   **标准化**：所有模型都遵循统一的配置和代码结构，方便切换和对比。
*   **功能强大**：内置了大量数据集支持、数据增强方法和评估脚本。

下面，我将为你提供一个使用 MMSegmentation 测试 UPerNet 的“傻瓜式”教程。我们将以 **Cityscapes** 数据集和一个预训练好的 **ResNet-50 UPerNet** 模型为例。

### 重要概念：MMSegmentation 与原版 UPerNet 的区别

请注意，MMSegmentation 中实现的 UPerNet 主要用于**语义分割**任务，即将图像中的每个像素分配给一个类别（如“道路”、“汽车”、“天空”）。

而原版论文中的“统一感知解析 (Unified Perceptual Parsing)”是一个更广义的任务，它同时解析物体的**部分 (parts)**、**材质 (materials)** 等多个层级。MMSegmentation 的实现并未包含这部分多任务解析，而是专注于其在标准语义分割数据集上的强大性能。

---

### Part 1: 环境准备

前面已经安装好环境`open-mmlab`了，所以这一部分内容跳过。

---

### Part 2: 快速上手 (对单张图片进行推理)

这是最快看到效果的一步。我们将下载一个预训练模型，并对一张你自己的图片或示例图片进行分割。

#### 步骤 1: 准备工作

1.  **激活 Conda 环境** (如果已关闭终端)
    ```bash
    conda activate open-mmlab
    cd path/to/your/mmsegmentation # 进入你克隆的 mmsegmentation 目录
    ```

2.  **创建目录**
    MMSegmentation 有标准的文件夹结构，我们先创建好。
    
    ```bash
    mkdir checkpoints
    mkdir outputs
    ```
    *   `checkpoints`: 用于存放下载的预训练模型文件 (`.pth`)。
    *   `outputs`: 用于存放模型的分割结果。

#### 步骤 2: 下载模型和配置文件

从你提供的 `README` 中，我们选择一个模型。以 `UPerNet-ResNet50` 在 `Cityscapes` 上的模型为例：

1.  **下载模型文件**：
    在终端中运行以下命令，将模型下载到 `checkpoints` 文件夹。
    
    ```bash
    wget --no-check-certificate -P checkpoints https://download.openmmlab.com/mmsegmentation/v0.5/upernet/upernet_r50_512x1024_40k_cityscapes/upernet_r50_512x1024_40k_cityscapes_20200605_094827-aa54cb54.pth
    ```
    
    我们需要在原来的 `wget` 命令后面加上 `--no-check-certificate` 参数，就可以告诉 `wget` “我知道这个网站的证书有问题，但我信任它，请继续下载”。
    
2.  **找到对应的配置文件**：
    这个模型对应的配置文件是 `configs/upernet/upernet_r50_4xb2-40k_cityscapes-512x1024.py`。你不需要下载它，因为它已经在 `mmsegmentation` 的代码中了。

#### 步骤 3: 准备测试图片

MMSegmentation 仓库里自带了一张 `demo.png`，我们可以直接用它。你也可以将自己喜欢的任何图片（比如一张街景图）放到 `demo` 文件夹下。

#### 步骤 4: 运行推理脚本

现在，使用 `MMSegmentation` 提供的 `image_demo.py` 脚本进行推理。

```bash
python demo/image_demo.py \
    demo/demo.png \
    configs/upernet/upernet_r50_4xb2-40k_cityscapes-512x1024.py \
    checkpoints/upernet_r50_512x1024_40k_cityscapes_20200605_094827-aa54cb54.pth \
    --out-file outputs/result.jpg \
    --device cuda:7
```

**命令解释**：
*   `demo/demo.png`: 你的输入图片路径。
*   `configs/.../upernet_...py`: 使用的模型配置文件。
*   `checkpoints/upernet_...pth`: 下载的预训练模型权重。
*   `--out-file outputs/result.jpg`: 指定输出结果的保存路径。
*   `--device cuda:0`: 指定使用第一张 GPU 进行计算。如果想用 CPU，可以改成 `cpu`。

运行成功后，你会在 `outputs` 文件夹里找到一张 `result.jpg` 图片。这张图上会用不同颜色标注出模型识别出的不同类别（如道路、汽车、行人等）。

---

### Part 3：训练和评估 UPerNet (PASCAL VOC 2012)

**目标**：使用 PASCAL VOC 2012 标准数据集，在 MMSegmentation 框架下，从头开始训练并评估一个 UPerNet 模型。

**前提**：你当前位于 `mmsegmentation` 项目的根目录下，并已配置好 Conda 环境。

#### 步骤 1: 数据准备 (下载与解压)

我们使用的 PASCAL VOC 2012 数据集版本已经为语义分割任务做好了预处理，因此数据准备步骤异常简单。

1.  **创建 `data` 目录**
    如果这个目录不存在，请先创建它。
    ```bash
    mkdir -p data
    ```

2.  **下载数据集文件**
    使用以下稳定链接将数据集下载到 `data` 目录中。
    
    ```bash
    wget -P data http://data.brainchip.com/dataset-mirror/voc/VOCtrainval_11-May-2012.tar
    ```
    
3.  **解压数据集**
    进入 `data` 目录，解压文件，然后返回项目根目录。
    ```bash
    cd data
    tar -xf VOCtrainval_11-May-2012.tar
    cd ..
    ```

4.  **验证 (可选但推荐)**
    执行以下命令，确认训练所需的列表文件已就位。
    ```bash
    ls data/VOCdevkit/VOC2012/ImageSets/Segmentation
    ```
    你应该能在输出中看到 `train.txt`, `val.txt` 和 `trainval.txt`。

**至此，数据准备工作已全部完成！无需任何转换或复制操作。**

### 步骤 2：创建并修改配置文件

我们将采用 MMSegmentation 最优雅的配置方式——通过修改 `_base_` 继承列表，从根本上切换数据集和训练策略。

#### 1. 复制配置文件

为了不污染官方配置，我们创建一个副本进行操作。

```bash
cp configs/upernet/upernet_r50_4xb4-40k_voc12aug-512x512.py configs/upernet/my_upernet_voc12.py
```

#### 2. 修改 `_base_` 列表并添加自定义配置

使用文本编辑器（如 `nano` 或 `vim`）打开我们刚刚创建的文件。

```bash
nano configs/upernet/my_upernet_voc12.py
```

在文件的最上方，找到 `_base_` 列表。它的作用就像积木一样，组合了模型、数据集和训练策略。

**原始内容：**

```python
_base_ = [
    '../_base_/models/upernet_r50.py',
    '../_base_/datasets/pascal_voc12_aug.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_40k.py'
]
```

**修改操作：**

1. 将指向增强数据集的 `pascal_voc12_aug.py` 替换为指向标准数据集的 **`pascal_voc12.py`**。
2. 将训练策略从基于迭代的 `schedule_40k.py` 切换为基于 Epoch 的自定义配置。为此，我们**删除** `'../_base_/schedules/schedule_40k.py'`。

**修改后内容：**

```python
_base_ = [
    '../_base_/models/upernet_r50.py',
    '../_base_/datasets/pascal_voc12.py',
    '../_base_/default_runtime.py',
]

crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor,
    decode_head=dict(num_classes=21),
    auxiliary_head=dict(num_classes=21))

# 优化器配置
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer, clip_grad=None)

# 学习率调度器
param_scheduler = [
    dict(
        type='PolyLR',
        eta_min=1e-4,
        power=0.9,
        begin=0,
        end=200,
        by_epoch=True)
]

# 默认钩子
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=True),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=True, interval=10),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'))

# --- 新增修改：重写训练数据加载器配置 ---
# 将采样器从默认的 InfiniteSampler 修改为 DefaultSampler，以兼容 EpochBasedTrainLoop
train_dataloader = dict(
    sampler=dict(type='DefaultSampler', shuffle=True))

# 训练、验证和测试循环配置
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=200, val_begin=1, val_interval=10)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
```

保存并退出：在 `nano` 中，按 `Ctrl+X`，然后按 `Y`，最后按 `Enter`。

**为什么这样修改？**

- **解耦配置**：`_base_` 的作用是将模型、数据集和训练策略等核心配置分离。通过删除 `'../_base_/schedules/schedule_40k.py'`，我们明确表示不再使用其默认的**基于迭代**的训练策略。
- ==**从迭代到 Epoch 的转换**：==原有的 `schedule_40k.py` 文件定义了**基于 40,000 次迭代**的训练。为了实现**基于 Epoch** 的训练，我们手动复制了优化器、学习率和钩子等配置，然后将所有与“迭代”相关的参数（如 `by_epoch=False`、`log_metric_by_epoch=False`）都改为了以 **`True`** 或 **`epoch`** 为单位。
- 除此之外，我们需要在配置文件中显式地重写（override）`train_dataloader`的配置，将采样器从 `InfiniteSampler` 改为 `DefaultSampler`。这样，训练循环才能在每个epoch结束后正确停止，从而触发checkpoint的保存。
- **提高恢复效率**：这种修改能够从根本上解决断点恢复时需要“快进”数千次空迭代的问题，从而大大加快训练启动速度。

#### 步骤 3: 开始训练

现在，所有准备工作都已就绪。确保使用你修改过的配置文件启动训练。

**单 GPU 训练 (指定卡号)：**
假设你想在 ID 为 1 的 GPU 上进行训练，而不是默认的 GPU 0。你只需要在命令前加上 CUDA_VISIBLE_DEVICES=1。

```bash
CUDA_VISIBLE_DEVICES=7 python tools/train.py configs/upernet/my_upernet_voc12.py
```
注意： GPU 的 ID 通常从 0 开始。

想要选择特定的 GPU 进行训练，而不是让系统自动分配，是深度学习中非常常见的需求。这可以避免你的任务与服务器上其他正在运行的任务发生冲突。

**多 GPU 训练 (指定多块卡号)**

如果你想用多块 GPU 进行训练，比如使用 ID 为 `0` 和 `1` 的两块显卡，只需要用逗号将它们隔开。

```bash
CUDA_VISIBLE_DEVICES=6,7 ./tools/dist_train.sh configs/upernet/my_upernet_voc12.py 2
```

这个命令会确保你的多 GPU 训练脚本（`dist_train.sh`）只使用 GPU `0` 和 `1`，并且启动两个进程来分别管理它们。

如果你想使用的 GPU ID 不是连续的，比如 `0` 和 `2`，命令也同样适用：

```bash
CUDA_VISIBLE_DEVICES=0,2 ./tools/dist_train.sh configs/upernet/my_upernet_voc12.py 2
```

**注意：** 命令最后的数字 `2` 仍然表示你将使用**两块** GPU，这与前面指定的 GPU 数量相匹配。

终端会开始打印训练日志，就像你看到的那样。日志和模型权重文件将会被保存在 `work_dirs/my_upernet_voc12/` 目录下。

#### 步骤 3.1: 从断点恢复训练 (Resume Training)

在长时间的训练过程中，可能会因为服务器维护、意外断电或手动停止而中断。幸运的是，MMSegmentation 提供了强大的断点恢复功能，可以让你从上次保存的状态（包括模型权重、优化器状态和迭代次数）无缝地继续训练，而不会丢失任何进度。

**场景**：假设你的训练在第 8000 次迭代后停止了，现在你想从 `iter_8000.pth` 这个断点文件继续跑完剩下的迭代。

**操作命令**：
你只需要在原来的训练命令后面加上 `--resume` 参数即可。

**单 GPU 从断点恢复**:
```bash
CUDA_VISIBLE_DEVICES=7 python tools/train.py configs/upernet/my_upernet_voc12.py --resume
```

**多 GPU 从断点恢复**:
```bash
CUDA_VISIBLE_DEVICES=6,7 ./tools/dist_train.sh configs/upernet/my_upernet_voc12.py 2 --resume
```

**工作原理**：
当使用 `--resume` 参数时，MMSegmentation 会自动执行以下操作：

1.  查找配置文件 `my_upernet_voc12.py` 中指定的 `work_dir`（即 `work_dirs/my_upernet_voc12/`）。
2.  在该目录中找到**最新的**一个检查点文件（例如 `iter_8000.pth`）。
3.  加载这个文件的所有训练状态。
4.  从第 8001 次迭代开始，继续执行训练，学习率等也会根据训练计划正确衔接。

#### 步骤 4: 评估你的模型

当训练完成后（或者中途想测试某个保存的 checkpoint），你可以使用 `tools/test.py` 脚本来评估其性能。

```bash
# 将下面的 CHECKPOINT_FILE 路径替换为你实际训练得到的模型权重文件
# 通常在 work_dirs/my_upernet_voc12/ 目录下，文件名类似 epoch_200.pth
CHECKPOINT_FILE="work_dirs/my_upernet_voc12/epoch_200.pth"
CONFIG_FILE="configs/upernet/my_upernet_voc12.py"

# 使用单 GPU 评估
python tools/test.py $CONFIG_FILE $CHECKPOINT_FILE
```
脚本运行结束后，你会在终端看到一个表格，其中包含详细的评估结果，最重要的指标是 `mIoU` (平均交并比)，它衡量了模型在所有类别上的平均分割精度。

#### 步骤 5: 使用你训练好的模型进行推理

最后，你可以像我们最开始那样，使用 `demo/image_demo.py` 来可视化模型的分割效果，只不过这次用的是**你自己亲手训练的模型**！

```bash
# 运行推理
python demo/image_demo.py \
    demo/demo.png \
    configs/upernet/my_upernet_voc12.py \
    work_dirs/my_upernet_voc12/epoch_200.pth \
    --out-file outputs/upernet_result.jpg \
    --device cuda:5
```
`work_dirs/my_upernet_voc12/iter_8000.pth \`：这里需要修改加载的权重文件的名字。

生成的结果如下：

![](../../../99_Assets%20(资源文件)/images/upernet_result.jpg)

测试结果2：

```bash
# 运行推理
python demo/image_demo.py \
    demo/PASCAL_VOC_2012_test.jpg \
    configs/upernet/my_upernet_voc12.py \
    work_dirs/my_upernet_voc12/epoch_200.pth \
    --out-file outputs/upernet_no_enhanced_200_result.jpg \
    --device cuda:5
```

![](../../../99_Assets%20(资源文件)/images/upernet_no_enhanced_200_result.jpg)
