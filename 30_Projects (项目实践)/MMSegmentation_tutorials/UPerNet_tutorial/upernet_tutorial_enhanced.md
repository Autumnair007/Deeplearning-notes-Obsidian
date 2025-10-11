---
type: tutorial
tags:
  - cv
  - semantic-segmentation
  - full-supervision
  - upernet
  - mmsegmentation
  - guide
  - pascal-voc
  - data-augmentation
status: done
model: UPerNet
year: 2018
related_backbone: ResNet-50
summary: UPerNet模型精度提升的完整指南，核心策略包括：集成SBD增强数据集(VOC+SBD, 10582张图)以解决数据量不足，并创建了一个高效的Epoch-based训练配置，解决了官方迭代式训练恢复慢的问题。配置中包含了学习率线性缩放、ResNet50预训练权重加载等最佳实践。
---
参考资料：[【数据集】——SBD数据集下载链接_sbd dataset-CSDN博客](https://blog.csdn.net/u011622208/article/details/131774571)
***
我们已经成功搭建了 MMSegmentation 环境并完成了初步训练，但对结果不满意。核心问题在于**训练数据量不足**和**训练恢复效率低下**。本指南将一步步带你解决这两个问题，从而显著提升模型性能。

#### **核心优化策略**

1.  **数据为王**：将训练数据从 PASCAL VOC 2012 的 **1,464** 张图片，扩充为结合了 SBD (Semantic Boundaries Dataset) 的 **10,582** 张增强数据集。这是提升精度的最关键一步。
2.  **高效训练**：保留你改进的 **Epoch-based (基于周期)** 训练模式，它能实现断点续训时的**秒级恢复**，避免官方迭代式训练恢复慢的问题。同时，我们会将训练总长度和学习率对齐官方的高效设定。

---

### **Part 1: 准备增强数据集 (解决数据量不足问题)**

**目标**：创建包含 10,582 张图片的 `trainaug.txt` 训练列表，供 MMSegmentation 使用。

**原因解释**：MMSegmentation 不会自动下载或转换 SBD 增强数据集。你提供的 `benchmark.tgz` 是正确的源文件，但需要一个官方脚本来处理它，将其格式（`.mat` 文件）和图片列表转换为 MMSegmentation 可识别的形式。你当前版本的 `mmsegmentation` 使用 `voc_aug.py` 脚本来完成此任务。

#### **步骤 1: 下载并解压 SBD 数据集**

如果还未操作，请先执行此步骤。在 `mmsegmentation` 根目录下，进入 `data` 文件夹执行：

```bash
# 进入数据目录
cd data

# 使用找到的有效链接下载 SBD 数据集
wget https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark.tgz

# 解压
tar -xvf benchmark.tgz

# 返回项目根目录
cd ..
```
操作完成后，你的 `data` 文件夹下应该有一个 `benchmark_RELEASE` 目录。

#### **步骤 2: 运行转换脚本生成训练列表**

这是将 SBD 数据集集成到项目中的关键一步。在 `mmsegmentation` **根目录**下运行以下命令：

```bash
python tools/dataset_converters/voc_aug.py \
    data/VOCdevkit \
    data/benchmark_RELEASE
```

**此命令为何这样写？**
*   `tools/dataset_converters/voc_aug.py`：这是你当前环境下正确的转换脚本。
*   `data/VOCdevkit`：这是脚本需要的第一个参数 `devkit_path`，它指向你存放 PASCAL VOC 2012 的位置。脚本将在这里生成最终的 `.txt` 列表文件。
*   `data/benchmark_RELEASE`：这是脚本需要的第二个参数 `aug_path`，它指向 SBD 数据集的位置。脚本会从这里读取 `.mat` 标注文件和原始的图片列表。

**执行后会发生什么？**

1.  **转换标注**：脚本会在 `data/VOCdevkit/VOC2012/` 下创建 `SegmentationClassAug` 目录，并将 SBD 数据集的 `.mat` 标注文件全部转换为 `.png` 格式的分割掩码保存在此。
2.  **生成列表**：脚本会在 `data/VOCdevkit/VOC2012/ImageSets/Segmentation/` 目录下生成 `trainaug.txt` 文件。此文件包含了用于训练的全部 **10,582** 个样本名。

至此，数据准备工作已 **圆满完成**。

---

### **Part 2: 配置高效的训练任务 (解决恢复慢并优化训练)**

**目标**：创建一个新的配置文件，它将使用我们刚准备好的增强数据集，并采用高效、可快速恢复的 Epoch-based 训练模式。

**原因解释**：直接使用官方的迭代式训练计划（如 `schedule_40k.py`）会导致从断点恢复训练时速度极慢。你之前将其改为 Epoch-based 训练是完全正确的方向。现在我们要做的是，在保持这个优点的基础上，将训练的总量和学习率等核心参数与经过验证的官方设置对齐。

#### **步骤 1: 创建新的配置文件**

为了不污染官方文件，我们复制一份配置进行修改。

```bash
cp configs/upernet/upernet_r50_4xb4-40k_voc12aug-512x512.py configs/upernet/my_upernet_final_voc.py
```

#### **步骤 2: 修改配置文件**

用你的文本编辑器打开 `configs/upernet/my_upernet_final_voc.py`，将其内容替换为以下精心调整过的配置：

````python name=configs/upernet/my_upernet_final_voc.py
# =========================================================================
#
#           UPerNet-ResNet50 在 PASCAL VOC 2012 + SBD (增强集) 上的
#                         正式训练配置文件
#
# 作者: Autumnair007 & Copilot
# 日期: 2025-08-21
#
# 目的: 本配置文件旨在提供一个健壮、可复现的训练流程。
#      它使用 PASCAL VOC 增强数据集 (10,582 张图片)，并采用
#      基于 Epoch 的训练循环，以便于从断点快速恢复训练。
#      所有超参数均经过计算，以对齐官方的最佳实践。
#
# =========================================================================

# --- 第 1 部分: 继承基础配置 ---
# 我们从 _base_ 目录继承模型架构、数据集特性和默认运行时行为
# (如日志格式等) 的基础设置。
# 我们特意省略了 `schedule_*.py` 文件，以便自定义基于 Epoch 的训练策略。
_base_ = [
    '../_base_/models/upernet_r50.py',
    '../_base_/datasets/pascal_voc12_aug.py',
    '../_base_/default_runtime.py'
]


# --- 第 2 部分: 用户自定义变量与硬件设置 ---
# 将关键参数集中在此处，方便快速修改。
# 当你需要将此配置适配到新硬件环境时，这里应该是你第一个检查和修改的地方。

# 你的硬件配置
gpu_count = 2         # 你用于训练的 GPU 数量。
samples_per_gpu = 4   # 每块 GPU 的批量大小 (Batch Size)。请根据你的显存大小调整。

# 训练超参数
num_workers = 4       # 用于数据加载的 CPU 线程数。

# 官方的经典配方通常基于总批量大小为 16。
# 我们使用“学习率线性缩放规则” (Linear Scaling Rule) 来适配你的硬件。
# 公式: 新学习率 = 基础学习率 * (你的总批量大小 / 官方总批量大小)
total_batch_size = gpu_count * samples_per_gpu
base_lr = 0.01
learning_rate = base_lr * (total_batch_size / 16)

# (可选) 计算与官方 40k 次迭代等价的 Epoch 总数，以确保训练量对齐。
# 公式: 总Epoch数 = 总迭代次数 / (数据集大小 / 总批量大小)
dataset_size = 10582  # PASCAL VOC 增强训练集的大小。
total_iterations = 40000
# 我们对计算结果进行四舍五入，以得到一个干净的整数。
# calculated_epochs = round(total_iterations / (dataset_size / total_batch_size)) # 约等于 30

# 根据你的要求，我们将训练周期设置为 200 个 Epochs 以进行更充分的训练。
max_epochs = 200


# --- 第 3 部分: 模型配置 ---
# 在这里，我们对从 _base_ 继承来的模型定义进行微调。
crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)
model = dict(
    # 关键修复: 显式地设置 `pretrained=None`。
    # 这解决了 "AssertionError: init_cfg and pretrained cannot be setting at the same time" 错误。
    # 它禁用了旧版的权重加载方式，确保只有新版的 `init_cfg` 方法生效。
    pretrained=None,
    data_preprocessor=data_preprocessor,
    backbone=dict(
        # `init_cfg` 是 MMEngine 推荐的权重初始化方式。
        # 这里，我们从 TorchVision 的模型库中加载在 ImageNet 上预训练好的 ResNet-50 权重。
        # 这是保证模型良好性能的至关重要的一步。
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    # 确保输出类别数与 PASCAL VOC 数据集匹配 (20个物体类别 + 1个背景类别)。
    decode_head=dict(num_classes=21),
    auxiliary_head=dict(num_classes=21))


# --- 第 4 部分: 数据加载器 (Dataloader) 配置 ---
# 定义在训练、验证和测试过程中，数据如何被送入模型。
train_dataloader = dict(
    batch_size=samples_per_gpu,  # 使用上方定义的变量。
    num_workers=num_workers,     # 使用上方定义的变量。
    # `DefaultSampler` 是实现基于 Epoch 训练的关键。它确保每个 Epoch 中，
    # 数据集里的每张图片只被采样一次。`shuffle=True` 是训练时的标准做法。
    sampler=dict(type='DefaultSampler', shuffle=True))

val_dataloader = dict(
    batch_size=1,                # 验证时，通常一次只处理一张图片。
    num_workers=num_workers,
    sampler=dict(type='DefaultSampler', shuffle=False)) # 验证时无需打乱数据顺序。

# 在本配置中，测试数据加载器与验证数据加载器完全相同。
test_dataloader = val_dataloader


# --- 第 5 部分: 优化器与学习率策略 ---
# 本部分定义了优化器 (如 SGD, Adam) 以及学习率在训练过程中的变化方式。
optimizer = dict(type='SGD', lr=learning_rate, momentum=0.9, weight_decay=0.0005)
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer, clip_grad=None)

# 学习率调度器 (Learning Rate Scheduler) 在训练中动态调整学习率。
param_scheduler = [
    dict(
        type='PolyLR',  # “多项式衰减”是一种常见且非常有效的学习率策略。
        eta_min=1e-4,   # 训练结束时学习率的最小值。
        power=0.9,      # 多项式函数的指数。
        begin=0,
        end=max_epochs, # 调度器将在整个训练周期 (0 到 max_epochs) 内生效。
        by_epoch=True)  # 关键参数: 必须为 True，以确保学习率按 Epoch 更新。
]


# --- 第 6 部分: 训练、验证与测试循环配置 ---
# 配置训练流程的整体控制逻辑。
train_cfg = dict(
    type='EpochBasedTrainLoop', # 指定训练由 Epoch 控制，而非迭代次数。
    max_epochs=max_epochs,      # 训练的总 Epoch 数。
    val_interval=1)             # 每训练 1 个 Epoch，进行一次验证。

val_cfg = dict(type='ValLoop')  # 使用标准的验证循环。
test_cfg = dict(type='TestLoop')# 使用标准的测试循环。


# --- 第 7 部分: 钩子 (Hooks) 配置 ---
# “钩子”是在训练循环的特定节点（如Epoch结束后、迭代前）执行的插件，用于执行特定操作。
default_hooks = dict(
    # 记录迭代耗时。
    timer=dict(type='IterTimerHook'),
    # 向控制台打印训练日志。`log_metric_by_epoch=True` 使指标（如mIoU）按 Epoch 聚合显示。
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=True),
    # 应用 `param_scheduler` 中定义的学习率变化。
    param_scheduler=dict(type='ParamSchedulerHook'),
    # 保存模型权重文件（checkpoint）。`by_epoch=True` 确保在 Epoch 结束时保存，与我们的设置保持一致。
    checkpoint=dict(type='CheckpointHook', by_epoch=True, interval=10, max_keep_ckpts=3), # 每10个epoch保存一次，最多保留最近3个
    # 为分布式训练中的每个进程设置不同的随机种子。
    sampler_seed=dict(type='DistSamplerSeedHook'),
    # (可选) 在训练过程中可视化分割结果，方便调试。
    visualization=dict(type='SegVisualizationHook'))
````

### **Part 3: 开始最终训练**

现在，使用你刚刚创建的最终版配置文件来启动训练。

在你的 VS Code 终端里，输入以下命令：

```bash
tmux kill-session -t my_training
tmux new -s my_training
conda activate open-mmlab
```

**单 GPU 训练**:

```bash
# 假设使用 GPU 7
CUDA_VISIBLE_DEVICES=7 python tools/train.py configs/upernet/my_upernet_final_voc.py
```

**多 GPU 训练**:
```bash
# 假设使用 GPU 6 和 7 (共2块)
CUDA_VISIBLE_DEVICES=6,7 ./tools/dist_train.sh configs/upernet/my_upernet_final_voc.py 2
```

**从断点恢复训练**:
由于我们采用了 Epoch-based 训练，恢复将非常迅速。只需在训练命令后加上 `--resume` 参数。

```bash
# 多 GPU 恢复示例
CUDA_VISIBLE_DEVICES=6,7 ./tools/dist_train.sh configs/upernet/my_upernet_final_voc.py 2 --resume
```

如果要连接回你指定的会话，可以输入命令：

```bash
tmux attach -t my_training
```

### Part 4 测试

输入命令测试训练结果：

```bash
python demo/image_demo.py \
    demo/PASCAL_VOC_2012_test.jpg \
    configs/upernet/my_upernet_voc12.py \
    work_dirs/my_upernet_final_voc/epoch_100.pth \
    --out-file outputs/upernet_enhanced_result.jpg \
    --device cuda:5
```

测试epoch_100.pth的结果如下：

![](../../../99_Assets%20(资源文件)/images/upernet_enhanced_result.jpg)

测试200 epochs的命令和结果如下：

```bash
python demo/image_demo.py \
    demo/PASCAL_VOC_2012_test.jpg \
    configs/upernet/my_upernet_voc12.py \
    work_dirs/my_upernet_final_voc/epoch_200.pth \
    --out-file outputs/upernet_enhanced_200_result.jpg \
    --device cuda:5
```

![](../../../99_Assets%20(资源文件)/images/upernet_enhanced_200_result.jpg)

测试demo图像的命令和结果：

```bash
# 运行推理
python demo/image_demo.py \
    demo/demo.png \
    configs/upernet/my_upernet_voc12.py \
    work_dirs/my_upernet_final_voc/epoch_200.pth \
    --out-file outputs/upernet_enhanced_demo_200_result.jpg \
    --device cuda:5
```

![](../../../99_Assets%20(资源文件)/images/upernet_enhanced_demo_200_result.jpg)

### **总结**

通过以上三个部分的系统性优化，你已经：
1.  将训练数据量扩大了 **7 倍**以上，为模型性能提供了坚实的基础。
2.  构建了一个**高效、可快速恢复**的训练配置，解决了断点续训的痛点。
3.  将学习率、训练总长等关键超参数与官方验证过的**最优实践对齐**。

