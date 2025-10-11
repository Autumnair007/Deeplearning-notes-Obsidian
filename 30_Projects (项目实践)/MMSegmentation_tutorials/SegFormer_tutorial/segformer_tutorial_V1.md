---
type: tutorial
tags:
  - cv
  - semantic-segmentation
  - full-supervision
  - transformer
  - segformer
  - mit-b2
  - pascal-voc
  - mmsegmentation
  - code-note
status: done
model: SegFormer
year: 2021
---
参考资料：[Python AttributeError: module ‘distutils‘ has no attribute ‘version‘_attributeerror: module 'distutils' has no attribut-CSDN博客](https://blog.csdn.net/Alexa_/article/details/132686602)

------

本指南将引导你完成一个完整、高效且专业的训练流程，使用 MMSegmentation 框架在 **PASCAL VOC 2012 增强数据集**上训练一个高性能的 **SegFormer-B2** 模型。

**核心优势:**

*   **前沿模型**: 采用强大的 `SegFormer` Transformer 架构，以其高效和高精度著称。
*   **海量数据**: 使用包含 **10,582** 张图片的增强数据集，为模型性能提供坚实保障。
*   **【性能优化】高效训练**: 通过**最大化批量大小 (Batch Size)**，充分压榨 GPU 硬件性能，显著缩短训练时间。并坚持使用基于 **Epoch** 的训练模式，实现断点续训的**秒级恢复**。
*   **专业实践**: 配置文件经过精心设计，遵循社区公认的最佳实践，并采用**最优模型保存策略**，确保最佳结果永不丢失。
*   **全程监控**: 无缝集成 `tmux` 进行稳定的会话管理，并通过 `TensorBoard` 提供从损失函数到评估指标的全方位可视化监控。

### Part 1: 环境与数据集准备

这部分将确保你的环境和数据已准备就绪。

1. **激活 Conda 环境**

   ```bash
   conda activate open-mmlab
   ```

2. **【重要】环境排错与预检**
   在开始之前，我们根据之前的经验，一次性修复所有已知的环境依赖问题，避免训练中断。

   *   **问题 1 (`setuptools`)**: 较新版本的 `setuptools` 与 PyTorch (1.10.1) 的 `distutils` 存在兼容性问题。
   *   **问题 2 (`protobuf`)**: `tensorboard` 与过高版本的 `protobuf` 存在 API 冲突。
   *   **解决方案**: 卸载任何已知的冲突包 (如 `openxlab`)，然后强制安装与你环境兼容的 `setuptools` 和 `protobuf` 的“黄金版本”。

   ```bash
   # 卸载已知冲突源
   pip uninstall openxlab
   # 强制安装兼容版本的 setuptools 和 protobuf
   pip install --force-reinstall "setuptools==59.5.0" "protobuf==3.20.1"
   ```

3. **进入项目目录**

   ```bash
   # 请将下面的路径替换为你自己的 mmsegmentation 仓库路径
   cd ~/projects/mmsegmentation
   ```

4. **确认目录与数据集**
   请确保标准目录（`data`, `checkpoints`, `outputs`, `work_dirs`）均已存在，并且你的增强数据集关键文件 `data/VOCdevkit/VOC2012/ImageSets/Segmentation/trainaug.txt` 已准备就绪。

### Part 2: 创建终极训练配置文件

我们将创建一个文件名能精确反映你实际训练配置的专属文件，并采用最稳妥的方式获取预训练权重。

#### 步骤 1: 【性能优化】创建高效率配置文件

基于我们对 GPU 利用率的观察（例如，之前仅有 30%），我们决定将每张卡的批量大小 (`samples_per_gpu`) 提升至 `6`，以充分利用硬件资源。我们将这个优化直接体现在文件名中。

```bash
# 文件名中的 '3xb6' 精确地反映了你的“3卡 x 每卡6样本”的高效配置
touch configs/segformer/my_segformer_mit-b2_3xb6-200e_voc12aug-512x512.py
```
**文件名解析**:

*   `3xb6`: 3张显卡 x 每张卡**6个样本**。总批量大小提升至18，显著提升训练效率和稳定性。

#### 步骤 1.5: 【核心】从 Hugging Face 获取并转换预训练权重

*   **问题**: 直接从 OpenMMLab 官方链接下载权重文件可能会因为网络环境的 SSL 证书问题 (`ssl.SSLCertVerificationError`) 而失败。
*   **最佳解决方案**: 从 Hugging Face 下载官方原始权重，然后转换为 MMSegmentation 兼容的格式。

1. **在本地电脑下载**:
   访问 NVIDIA 官方的 `mit-b2` 模型页面：[**https://huggingface.co/nvidia/mit-b2**](https://huggingface.co/nvidia/mit-b2)。点击 "Files and versions" 标签页，下载名为 `pytorch_model.bin` 的文件到你的本地电脑。

2. **上传文件到远程服务器**:
   这是将本地文件上传到 VSCode 连接的远程服务器的最快方法。
   a. 在 VS Code 左侧的文件资源管理器中，导航到服务器的 `~/projects/mmsegmentation/` 目录，并创建一个新文件夹 `hf_models`。
   b. 在你**本地电脑**的文件管理器中，找到下载好的 `pytorch_model.bin`。
   c. **直接用鼠标将 `pytorch_model.bin` 文件拖拽到 VS Code 左侧的 `hf_models` 文件夹上**，松开鼠标即可自动上传。

3. **在服务器上转换权重格式**:
   运行 MMSegmentation 自带的转换脚本，生成我们最终要用的权重文件。

   ```bash
   # 确保 checkpoints 目录存在
   mkdir -p checkpoints
   
   # 运行转换脚本
   python tools/model_converters/mit2mmseg.py \
     hf_models/pytorch_model.bin \
     checkpoints/mit_b2_converted_from_hf.pth
   ```

#### 步骤 2: 填入性能优化版配置
将以下经过**性能优化**的配置内容，完整地复制并粘贴到你刚刚创建的 `..._3xb6_...` 文件中。

````python name=configs/segformer/my_segformer_mit-b2_3xb6-200e_voc12aug-512x512.py
# =========================================================================
#
#        SegFormer-MiT-B2 在 PASCAL VOC 2012 增强数据集上的
#          终极训练配置文件 (v2.1 - 学习率与结束点双重修正版)
#
# 作者: Autumnair007 & Copilot
# 日期: 2025-08-27 (采纳用户关于 scheduler end point 的优化建议)
#
# =========================================================================

# --- 第 1 部分: 继承基础配置 ---
_base_ = [
    '../_base_/models/segformer_mit-b0.py',
    '../_base_/datasets/pascal_voc12_aug.py',
    '../_base_/default_runtime.py'
]

# --- 第 2 部分: 硬件与训练超参数 (性能优化) ---
gpu_count = 3
samples_per_gpu = 6
num_workers = 8
learning_rate = 0.00006
checkpoint_epoch = 10 # 每隔多少 epoch 保存一次模型
val_epoch = 10    # 每隔多少 epoch 进行一次验证
max_epochs = 200 
warmup_epochs = 15

# --- 第 3 部分: 模型配置 ---
crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)
checkpoint = 'checkpoints/mit_b2_converted_from_hf.pth'

model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint),
        embed_dims=64,
        num_layers=[3, 4, 6, 3]),
    decode_head=dict(
        in_channels=[64, 128, 320, 512],
        num_classes=21),
)

# --- 第 4 部分: 数据加载器 (Dataloader) 配置 ---
train_dataloader = dict(
    batch_size=samples_per_gpu,
    num_workers=num_workers,
    sampler=dict(type='DefaultSampler', shuffle=True))

val_dataloader = dict(
    batch_size=1,
    num_workers=num_workers,
    sampler=dict(type='DefaultSampler', shuffle=False))
test_dataloader = val_dataloader

# --- 第 5 部分: 优化器与学习率策略 ---
optimizer = dict(
    type='AdamW', lr=learning_rate, betas=(0.9, 0.999), weight_decay=0.01)
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=optimizer,
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=10.)
        }))

# =========================================================================
# 【关键修正】学习率调度器现在使用更合理的 Warmup 周期和更严谨的结束点
# =========================================================================
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1e-6,
        by_epoch=True,
        begin=0,
        # Warmup 阶段延长至 15 个 Epochs，让模型充分预热
        end=warmup_epochs,
    ),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        by_epoch=True,
        # 在 Warmup 结束后 (第 15 个 epoch 之后) 再开始学习率衰减
        begin=warmup_epochs,
        # 【采纳建议】设置为 max_epochs + 1，确保最后一个 epoch 的衰减逻辑严谨无误
        end=max_epochs + 1,
    )
]

# --- 第 6 部分: 训练、验证与测试循环配置 ---
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=max_epochs,
    val_interval=val_epoch)

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# --- 第 7 部分: 钩子 (Hooks) 与最终可视化配置 ---
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=True),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=True,
        interval= checkpoint_epoch,
        max_keep_ckpts=3,
        save_best='mIoU',
        rule='greater'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'))

# 可视化后端配置
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend')
]
visualizer = dict(
    type='SegLocalVisualizer', 
    vis_backends=vis_backends, 
    name='visualizer'
)
````

### Part 3: 开始训练与监控 (Tmux 与 TensorBoard 整合流程)

这个流程将确保你的训练任务在一个稳定的后台会话中运行，并能从你自己的电脑方便地监控。

#### 步骤 1: 创建并进入 Tmux 会话
```bash
tmux new -s seg_train_b6
```

#### 步骤 2: 在 Tmux 会话中启动训练
在 `seg_train_b6` 会话中，激活环境并使用**性能优化后**的配置文件启动训练。
```bash
conda activate open-mmlab
# 使用更新后的高效配置文件
CUDA_VISIBLE_DEVICES=5,6,7 ./tools/dist_train.sh configs/segformer/my_segformer_mit-b2_3xb6-200e_voc12aug-512x512.py 3
```

#### 步骤 3: 启动并访问 TensorBoard (VS Code 远程终极方案)
此步骤不变，TensorBoard 会自动找到新训练任务生成的新日志文件夹。
1. **分离会话**: `Ctrl + B`, then `D`.

2. **启动 TensorBoard 服务**: 

   ```bash
   tensorboard --logdir work_dirs/
   ```

3. **在 VS Code 中转发端口**: `PORTS` -> `Forward a Port` -> `6006`.

4. **从你的电脑访问仪表盘**: 点击“在浏览器中打开”图标或访问 `http://localhost:6006/`。

#### 步骤 4: 管理你的训练会话

* **重新连接 (查看进度)**: `tmux attach -t seg_train_b6`
* **从断点恢复训练**:
  ```bash
  tmux attach -t seg_train_b6
  # 确保使用正确的优化版配置文件
  CUDA_VISIBLE_DEVICES=5,6,7 ./tools/dist_train.sh configs/segformer/my_segformer_mit-b2_3xb6-200e_voc12aug-512x512.py 3 --resume
  ```
* **删除旧会话**: `tmux kill-session -t seg_train_b6`

### Part 4: 评估与测试

训练完成后，对你的模型进行最终的检验。所有命令都已更新为使用新的配置文件和工作目录。

#### 步骤 1: 评估模型性能

**可能会发生的错误**

从你提供的日志最后部分的 `Traceback` 来看，错误的核心在于这一行：

```
AttributeError: module 'PIL.Image' has no attribute 'ANTIALIAS'
```

**错误原因**：

这个错误表示在 `PIL`（或其替代品 `Pillow`）库的 `Image` 模块中找不到 `ANTIALIAS` 这个属性。

1.  **Pillow 版本过高**：`Image.ANTIALIAS` 在 `Pillow` 10.0.0 版本中被废弃并移除。取而代之的是 `Image.Resampling.LANCZOS` 或者 `Image.LANCZOS`。
2.  **依赖冲突**：你环境中安装的 `torch` 或 `torchvision` 版本可能依赖于一个旧的 `Pillow` 版本，但现在你环境中的 `Pillow` 版本太新了，导致 `torchvision` 或 `tensorboard` 内部的代码调用了已经被移除的属性。

从错误堆栈中可以看到，问题发生在 `torch/utils/tensorboard/summary.py` 文件调用 `Image.ANTIALIAS` 时。这说明你环境中的 `tensorboard`（作为 `PyTorch` 的一部分）期望使用一个较旧版本的 `Pillow`。

**解决方案**

最直接和推荐的解决方案是**降级 Pillow 库的版本**，使其与你环境中其他库（特别是 `PyTorch 1.10.1`）兼容。请在你的 `open-mmlab` conda 环境中执行以下命令：

```bash
pip install Pillow==9.5.0
```

我们优先使用**最优模型**进行评估。
```bash
CONFIG_FILE="configs/segformer/my_segformer_mit-b2_3xb6-200e_voc12aug-512x512.py"
# 【注意】工作目录名会根据配置文件名自动改变
CHECKPOINT_FILE="work_dirs/my_segformer_mit-b2_3xb6-200e_voc12aug-512x512/best_mIoU_epoch_200.pth"

CUDA_VISIBLE_DEVICES=4 python tools/test.py $CONFIG_FILE $CHECKPOINT_FILE --show-dir outputs/test_results_b6
```

评估结果如下：

```bash
+-------------+-------+-------+
|    Class    |  IoU  |  Acc  |
+-------------+-------+-------+
|  background | 90.23 | 95.79 |
|  aeroplane  | 74.78 | 83.51 |
|   bicycle   | 32.35 | 76.72 |
|     bird    | 67.08 | 80.48 |
|     boat    | 51.21 | 64.39 |
|    bottle   | 42.44 | 48.06 |
|     bus     | 80.75 | 85.15 |
|     car     | 70.96 | 85.67 |
|     cat     | 70.84 | 80.85 |
|    chair    | 25.53 | 38.86 |
|     cow     | 61.07 |  71.8 |
| diningtable | 37.51 | 42.36 |
|     dog     | 60.29 |  81.7 |
|    horse    | 59.67 | 72.09 |
|  motorbike  | 67.64 | 78.69 |
|    person   | 70.57 | 85.21 |
| pottedplant | 41.16 | 48.25 |
|    sheep    | 73.58 | 81.71 |
|     sofa    | 29.73 | 36.48 |
|    train    | 74.72 | 83.38 |
|  tvmonitor  | 48.46 | 58.68 |
+-------------+-------+-------+
2025/08/28 11:27:37 - mmengine - INFO - Iter(test) [1449/1449]    aAcc: 89.9100  mIoU: 58.6000  mAcc: 70.4700  data_time: 0.8346  time: 0.9140
```

#### 步骤 2: 对单张图片进行可视化推理

同样使用最优模型，在**第 4 号 GPU** 上对任意一张图片进行测试。
```bash
python demo/image_demo.py \
    demo/demo.png \
    configs/segformer/my_segformer_mit-b2_3xb6-200e_voc12aug-512x512.py \
    work_dirs/my_segformer_mit-b2_3xb6-200e_voc12aug-512x512/best_mIoU_epoch_200.pth \
    --out-file outputs/my_segformer_best_result_b6.jpg \
    --device cuda:4
```

这是训练结果：

![](../../../99_Assets%20(资源文件)/images/my_segformer_best_result_b6.jpg)

```bash
python demo/image_demo.py \
    demo/PASCAL_VOC_2012_test.jpg \
    configs/segformer/my_segformer_mit-b2_3xb6-200e_voc12aug-512x512.py \
    work_dirs/my_segformer_mit-b2_3xb6-200e_voc12aug-512x512/best_mIoU_epoch_200.pth \
    --out-file outputs/my_segformer_best_result_b6_PASCAL_VOC.jpg \
    --device cuda:4
```

这是测试的结果

![](../../../99_Assets%20(资源文件)/images/my_segformer_best_result_b6_PASCAL_VOC.jpg)
