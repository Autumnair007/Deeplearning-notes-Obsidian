# Segformer ADE数据集 V1

从PASCAL VOC 2012 增强数据集V3版本继承继续研究。

参考资料：[(一)ADE20K数据集-CSDN博客](https://blog.csdn.net/lx_ros/article/details/125650685)

ADEchallenge 2016 的下载地址为： http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip

官网下载地址为：[ADE20K dataset](https://ade20k.csail.mit.edu/)，要注册。GitHub仓库为：[CSAILVision/ADE20K: ADE20K Dataset](https://github.com/CSAILVision/ADE20K)

***

### 官方下载的数据集和其他地方数据集的差别

简单来说，**你下载的 `ADE20K_2021_17_01` 和你在其他地方看到的 `ADEChallengeData2016`，很可能指向的是同一个核心数据集，即包含150个语义类别、超过2万张训练图像和2千张验证图像的那个版本。** 它们本质内容相同，只是官方在不同时期可能用了不同的打包命名方式。

下面是一个表格，帮你快速了解这两个版本标识的关系和区别：

| 特性维度              | ADEChallengeData2016 (常见于论文和代码库)                    | ADE20K_2021_17_01 (你从官网下载的版本)                       |
| :-------------------- | :----------------------------------------------------------- | :----------------------------------------------------------- |
| **常见名称**          | ADE20K (Scene Parsing), ADE20K-150                           | 可能为数据包内部版本标识或特定发布版名称                     |
| **核心数据内容**      | **150个语义类别**，**20,210张训练图**，**2,000张验证图**     | 应包含相同的150个类别及图像数据（需确认`objects.txt`内容）   |
| **主要用途**          | **语义分割**、场景解析（如SegFormer等模型训练）              | 同左                                                         |
| **来源与引用**        | MIT发布，论文和框架（如MMSegmentation）中常用此名称          | 应源自同一官方渠道，可能为更新或重新打包的版本标识           |
| **文件结构参考**      | `ADEChallengeData2016`<br>  ├── `annotations`<br>  │   ├── `training`<br>  │   └── `validation`<br>  └── `images`<br>      ├── `training`<br>      └── `validation` | `ADE20K_2021_17_01`<br>  ├── `images`<br>  │   └── `ADE`<br>  ├── `training` (子目录按场景分类)<br>  ├── `validation`<br>  ├── `index_ade20k.mat`<br>  ├── `index_ade20k.pkl`<br>  └── `objects.txt` |
| **与SegFormer的关联** | SegFormer论文及MMSegmentation等框架**均使用此版本名称**      | 核心数据应与前者一致，可供SegFormer训练                      |

🧐 **SegFormer论文与MMSegmentation用的是哪个？**

根据你提供的MMSegmentation中SegFormer的README，以及搜索结果，**SegFormer原始论文和MMSegmentation等主流框架中提到的ADE20K数据集，指的就是 `ADEChallengeData2016` 这个版本**（包含150个类别，20210张训练图像和2000张验证图像）。你提供的MMSegmentation的配置文件中也多次出现了 `ADE20K` 的字段，这指向的同样是这个150个类别的数据集。

你从官网下载的 `ADE20K_2021_17_01`，其名称中的“2021”可能表示数据打包或发布的年份，而“17_01”可能是一个内部版本号或批次标识。**关键在于其核心内容（图像和150个类别的标注）应与 `ADEChallengeData2016` 一致。**

💡 **给你的建议**

1.  **确认数据内容**：打开你下载的 `ADE20K_2021_17_01` 中的 `objects.txt` 文件，检查它是否包含了150个类别的名称列表。这是确认它是否是150类别版本的最直接方法。
2.  **用于训练**：如果你想在MMSegmentation等框架中训练SegFormer或其他模型，**核心是确保你的数据路径设置正确**。虽然文件夹名称不同，你只需要在配置文件中将数据路径指向你解压后的 `ADE20K_2021_17_01` 文件夹即可，或者将其重命名为框架期望的名称（如 `ADEChallengeData2016`），但更推荐**修改配置文件中的路径**以避免混淆。
3.  **理解差异**：`ADE20K_2021_17_01` 的 `training` 目录下包含了按场景类型（如cultural, urban等）划分的子目录，这可能是官方提供的另一种数据组织方式，便于用户按场景查找图像，但**最终的语义分割标注（annotations）应该还是那150个类别**。

📌 **总结一下**
你可以认为 `ADEChallengeData2016` 是学术论文和代码框架中**约定俗成的名称**，指向那个经典的150类别数据集。而你下载的 `ADE20K_2021_17_01` 很可能是**官方后续提供的一个数据包版本**，其核心内容与前者一致，只是打包方式和顶层目录结构略有不同。

因此，**不必过于纠结名称差异**，只要你的数据集包含150个类别、超过2万张训练图像和2千张验证图像，就可以放心地把它当作“那个”ADE20K数据集来使用，并用于训练像SegFormer这样的模型。

***

代码的主要修改点如下：

1. **数据集配置**：
   - `dataset_type` 已从 `PascalVOCDataset` 更改为 `ADE20KDataset`。
   - `data_root` 已更新为 `data/ADEChallengeData2016`。
   - `data_prefix` 和 `ann_file` 已根据 `ADE20K` 的目录结构进行了调整。`ADE20K` 不需要 `ann_file`，并且训练/验证集分别在不同的子目录中。
   - 由于 `ADE20K` 没有像 `VOC` 那样的 `aug.txt` 增强集，因此我移除了 `ConcatDataset`，现在只使用标准的训练集。
2. **模型配置**：
   - 解码头（`decode_head`）的 `num_classes` 已从 `21`（PASCAL VOC）更改为 `150`（ADE20K）。
   - `ignore_index` 在 `ADE20K` 中通常是 `255`，这与你的损失函数设置一致，予以保留。
   - 将FocalLoss参数权重下调，提高DiceLoss权重
   - 在训练流水线中，使用 RandomCutOut 替代 RandomErasing。
3. **预训练权重（重要修正）**：
   - `mit_b2_converted_from_hf.pth` 应该作为 **backbone 的预训练权重**，而不是使用 `load_from` 来加载整个模型的 checkpoint。`load_from` 会尝试加载所有组件（包括解码头），这在数据集类别数不同时会引发错误。
   - 因此，我**移除了全局的 `load_from` 配置**。
   - 并在 `model` -> `backbone` 配置中，**重新启用了 `init_cfg`**，并将其 `checkpoint` 路径指向你提供的 `checkpoints/mit_b2_converted_from_hf.pth`。
   - 这样，MMSegmentation 会在初始化模型时，只加载主干网络（Backbone）的权重，而解码头（Decode Head）等其他部分将进行随机初始化，这正是从头开始训练新数据集（如 ADE20K）的标准做法。

其他所有配置，包括学习率、优化器、数据增强流水线 (`train_pipeline`)、训练周期 (`max_epochs`) 等，都保持了你原有的设置。

```python name=my_segformer_mit-b2_512x512_200e_ade20k.py
# =========================================================================
#
#        SegFormer-MiT-B2 在 ADE20K 数据集上的
#       训练配置文件 (v1.0 - 基于 PASCAL VOC v3.1 脚本修改)
#
# 作者: Autumnair007
# 日期: 2025-09-02
#
# =========================================================================

# --- 第 1 部分: 继承基础配置 ---
_base_ = [
    '../_base_/models/segformer_mit-b0.py', # 基础模型结构，后续会覆盖
    '../_base_/default_runtime.py'
]

checkpoint = 'checkpoints/mit_b2_converted_from_hf.pth'

# --- 第 2 部分: 硬件与训练超参数 ---
gpu_count = 3
samples_per_gpu = 2
num_workers = 8
learning_rate = 0.000006
checkpoint_epoch = 10
val_epoch = 10
max_epochs = 200
warmup_epochs = 20

# --- 第 3 部分: 数据集配置 (修改为 ADE20K) ---
dataset_type = 'ADE20KDataset'
data_root = 'data/ADEChallengeData2016' # <-- 修改: 数据集根目录
crop_size = (512, 512)

# 训练数据处理流水线
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(
        type='RandomResize',
        scale=(2048, 512),
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    # 使用 RandomCutOut 替代 RandomErasing
    dict(type='RandomCutOut', 
         prob=0.5, 
         n_holes=(1, 3),
         cutout_ratio=[(0.02, 0.02), (0.2, 0.2)],
         fill_in=(0, 0, 0), 
         seg_fill_in=255),
    dict(type='Pad', size=crop_size),
    dict(type='PackSegInputs')
]

# 测试数据处理流水线
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(2048, 512), keep_ratio=True),
    dict(type='LoadAnnotations', reduce_zero_label=True), # ADE20K 标签从 1 开始，0 是背景，需要减一
    dict(type='PackSegInputs')
]

# --- 第 4 部分: 数据加载器配置 (修改为 ADE20K) ---
train_dataloader = dict(
    batch_size=samples_per_gpu,
    num_workers=num_workers,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='images/training', seg_map_path='annotations/training'), # <-- 修改: ADE20K 路径
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=1,
    num_workers=num_workers,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='images/validation', seg_map_path='annotations/validation'), # <-- 修改: ADE20K 路径
        pipeline=test_pipeline))

test_dataloader = val_dataloader

# 评估器配置
val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = val_evaluator

# --- 第 5 部分: 模型配置 ---
data_preprocessor = dict(
    size=crop_size
)

model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(
        # 保持 MiT-B2 的配置
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint),
        embed_dims=64,
        num_layers=[3, 4, 6, 3]),
    decode_head=dict(
        in_channels=[64, 128, 320, 512],
        num_classes=150, # <-- 修改: ADE20K 类别数为 150
        loss_decode=[
            dict(
                type='FocalLoss',
                use_sigmoid=True,
                loss_weight=0.7),
            dict(
                type='DiceLoss',
                loss_weight=1.3,
                ignore_index=255) # ADE20K 的 ignore_index 也是 255
        ]),
)

# --- 第 6 部分: 优化器与学习率策略 ---
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

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1e-7,
        by_epoch=True,
        begin=0,
        end=warmup_epochs,
    ),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        by_epoch=True,
        begin=warmup_epochs,
        end=max_epochs + 1,
    )
]

# --- 第 7 部分: 训练、验证与测试循环配置 ---
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=max_epochs,
    val_interval=val_epoch)

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# --- 第 8 部分: 钩子与可视化配置 ---
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=True),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=True,
        interval=checkpoint_epoch,
        max_keep_ckpts=3,
        save_best='mIoU',
        rule='greater'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'))

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend')
]
visualizer = dict(
    type='SegLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer'
)
```

### 运行命令：

```bash
CUDA_VISIBLE_DEVICES=5,6,7 ./tools/dist_train.sh configs/segformer/my_segformer_mit-b2_512x512_200e_ade20k.py 3
```

#### tmux的命令如下：

* **创建会话**: 

  ```bash
  tmux new-session -s seg_ade_v1
  conda activate open-mmlab
  ```

* **恢复会话：**

  ```bash
  tmux attach -t seg_ade_v1
  ```

* **删除旧会话**: 

  ```bash
  tmux kill-session -t seg_ade_v1
  ```

***

### V1第一次训练出现的问题和分析：

在初次将模型从 PASCAL VOC 迁移到 ADE20K 数据集时，直接沿用了之前的配置，遇到了训练效率低下的问题。

**初始配置:**

*   **数据集**: ADE20K (150类)
*   **学习率**: `6e-6` (非常保守)
*   **Batch Size**: `samples_per_gpu=2` (受显存限制)
*   **Crop Size**: `(512, 512)`

**问题分析:**

1.  **任务复杂度剧增**: 从21类到150类，场景更复杂，模型需要学习的特征和边界也更精细。
2.  **Batch Size 过小**: `total_batch_size = 2 * 3 = 6`，这会导致梯度估计噪声大，训练不稳定，收敛速度慢。
3.  **学习率严重不匹配**: 对于 ADE20K 这样的大型、复杂数据集，`6e-6` 的学习率过小，导致模型在初期探索不足，陷入了“龟速”学习的困境，mIoU 增长极其缓慢。

#### V1.1: 错误的优化尝试——“灾难性遗忘”

为了解决 V1 的问题，我进行了一次优化尝试，但采用了错误的方法，即**在已经训练了一段时间的模型上直接修改关键超参数**。

**错误的操作流程:**

1.  加载了一个已经训练了120个epoch的检查点 (`epoch_120.pth`)。
2.  为了增大 Batch Size，将 `crop_size` 从 `(512, 512)` 减小到 `(480, 480)`。
3.  为了“加速训练”，将学习率从一个已经衰减后很小的值，突然大幅提高回 `6e-5`。

**结果：** 模型性能瞬间崩溃，mIoU 几乎归零。

**问题根源——灾难性遗忘 (Catastrophic Forgetting):**
一个已经经过充分训练、接近收敛的模型，其权重已经处在损失函数的“最优谷底”。此时突然引入一个巨大的学习率，相当于用一个巨大的步长将权重“踢”出了这个最优区域。模型之前学到的所有知识瞬间丢失，一切归零。

> **核心教训**: 严禁在训练后期或对一个已收敛的模型，将学习率从小突然调大。这是一种“开倒车”行为，会毁掉已有的训练成果。

#### V1.2: 正确的优化策略 (当前方案)

吸取了 V1.1 的教训后，我制定了全新的、从零开始的训练策略。该策略旨在从一开始就平衡好**训练效率**和**模型性能**。

**最终配置 (基于 `v1.1` 代码):**
*   **训练起点**: 从官方提供的 **ImageNet 预训练权重** (`mit_b2_converted_from_hf.pth`) 开始，确保一个良好的起点。
*   **学习率**: `learning_rate = 6e-5`。这是一个经过社区和论文验证的、适合 ADE20K 的标准初始学习率。
*   **Batch Size**: `samples_per_gpu = 4`。
*   **Crop Size**: **保持 `(512, 512)` 不变**，以保证模型能看到足够大的感受野和上下文信息。
*   **损失函数权重**: 调整 `FocalLoss` 和 `DiceLoss` 的权重为 `1.0:1.0`，进行平衡性实验。

**关键优化手段:**

1.  **【核心】启用混合精度训练 (AMP)**
    
    ```python
    optim_wrapper = dict(
        type='AmpOptimWrapper',  # 启用混合精度训练
        optimizer=optimizer,
        clip_grad=dict(max_norm=35, norm_type=2),
        loss_scale='dynamic',
        ...
    )
    ```
    **原理**: 这是本次优化的**关键**。通过启用 AMP，利用现代 GPU 的 Tensor Core 进行 FP16 计算，在不牺牲太多精度的情况下，**大幅降低了显存占用**。这使得我们可以在**不减小 `crop_size` 的前提下，将 `samples_per_gpu` 从 2 提升到 4**，从而获得更稳定、更高效的梯度更新。
    
2.  **【稳定性】梯度裁剪与动态损失缩放**
    *   `clip_grad`: 防止在训练初期因学习率较大可能导致的梯度爆炸问题。
    *   `loss_scale='dynamic'`: 在混合精度训练中，自动调整损失的缩放因子，防止因 FP16 数值范围过小导致的梯度下溢（变为0）。

3.  **【加速收敛】为解码头设置更高学习率**
    
    ```python
    paramwise_cfg=dict(
        ...
        'head': dict(lr_mult=10.)  # 解码头使用10倍学习率
    )
    ```
    **原理**: Backbone 部分使用了 ImageNet 预训练权重，已经学到了很好的通用特征，需要较小的学习率进行微调。而解码头（Decode Head）是随机初始化的，需要一个更大的学习率来快速学习如何将这些特征映射到150个分割类别上。
    
4.  **【稳定性】充足的 Warmup**
    ```python
    param_scheduler = [
        dict(
            type='LinearLR',
            start_factor=1e-6, # 从一个极小值开始
            by_epoch=True,
            begin=0,
            end=warmup_epochs, # warmup_epochs = 20
        ),
        ...
    ]
    ```
    **原理**: 在训练初期，通过 20 个 epoch 的线性学习率预热，让模型和优化器状态（如 AdamW 的动量）逐渐适应数据和较大的学习率，有效防止训练初期的震荡和崩溃。

完整代码：
```python
# =========================================================================
#
#      SegFormer-MiT-B2 在 ADE20K(ADEChallengeData2016) 数据集上的
#       训练配置文件 (v1.2 - 提高学习率并启用混合精度训练)
#
# 作者: Autumnair007
# 日期: 2025-09-04
#
# =========================================================================

# --- 第 1 部分: 继承基础配置 ---
_base_ = [
    '../_base_/models/segformer_mit-b0.py', # 基础模型结构，后续会覆盖
    '../_base_/default_runtime.py'
]

checkpoint = 'checkpoints/mit_b2_converted_from_hf.pth'

# --- 第 2 部分: 硬件与训练超参数 ---
gpu_count = 3
samples_per_gpu = 2
num_workers = 8
learning_rate = 3e-05 # <-- 修改: 提高学习率以适应更大的 batch size
checkpoint_epoch = 10
val_epoch = 10
max_epochs = 200
warmup_epochs = 20

# --- 第 3 部分: 数据集配置 (ADE20K) ---
dataset_type = 'ADE20KDataset'
data_root = 'data/ADEChallengeData2016'
crop_size = (512, 512)

# 训练数据处理流水线
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(
        type='RandomResize',
        scale=(2048, 512),
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='RandomCutOut', 
         prob=0.5, 
         n_holes=(1, 3),
         cutout_ratio=[(0.02, 0.02), (0.2, 0.2)],
         fill_in=(0, 0, 0), 
         seg_fill_in=255),
    dict(type='Pad', size=crop_size),
    dict(type='PackSegInputs')
]

# 测试数据处理流水线
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(2048, 512), keep_ratio=True),
    dict(type='LoadAnnotations', reduce_zero_label=True), # ADE20K 标签从 1 开始，0 是背景，需要减一
    dict(type='PackSegInputs')
]

# --- 第 4 部分: 数据加载器配置 (ADE20K) ---
train_dataloader = dict(
    batch_size=samples_per_gpu,
    num_workers=num_workers,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='images/training', seg_map_path='annotations/training'), # <-- 修改: ADE20K 路径
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=1,
    num_workers=num_workers,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='images/validation', seg_map_path='annotations/validation'), # <-- 修改: ADE20K 路径
        pipeline=test_pipeline))

test_dataloader = val_dataloader

# 评估器配置
val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = val_evaluator

# --- 第 5 部分: 模型配置 ---
data_preprocessor = dict(
    size=crop_size
)

model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(
        # 保持 MiT-B2 的配置
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint),
        embed_dims=64,
        num_layers=[3, 4, 6, 3]),
    decode_head=dict(
        in_channels=[64, 128, 320, 512],
        num_classes=150, # <-- 修改: ADE20K 类别数为 150
        loss_decode=[
            dict(
                type='FocalLoss',
                use_sigmoid=True,
                loss_weight=0.5,),
            dict(
                type='DiceLoss',
                loss_weight=1.0,
                ignore_index=255) # ADE20K 的 ignore_index 也是 255
        ]),
)

# --- 第 6 部分: 优化器与学习率策略 ---
optimizer = dict(
    type='AdamW', 
    lr=learning_rate, 
    betas=(0.9, 0.999), 
    weight_decay=0.01
)

optim_wrapper = dict(
    type='AmpOptimWrapper',  # 启用混合精度训练
    optimizer=optimizer,
    clip_grad=dict(max_norm=10, norm_type=2),  # 添加梯度裁剪稳定训练
    loss_scale='dynamic',  # 动态损失缩放:
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=10.)  # 解码头使用更高学习率
        }
    )
)

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1e-6,
        by_epoch=True,
        begin=0,
        end=warmup_epochs,
    ),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        by_epoch=True,
        begin=warmup_epochs,
        end=max_epochs,
    )
]

# --- 第 7 部分: 训练、验证与测试循环配置 ---
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=max_epochs,
    val_interval=val_epoch)

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# --- 第 8 部分: 钩子与可视化配置 ---
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=True),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=True,
        interval=checkpoint_epoch,
        max_keep_ckpts=3,
        save_best='mIoU',
        rule='greater'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'))

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend')
]
visualizer = dict(
    type='SegLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer'
)
```

#### 最终总结与训练建议

1.  **训练策略应该是“先快后慢”**: 用一个相对较大的学习率开始（如 `6e-5`），配合充足的 warmup，然后通过学习率调度器（如 `PolyLR`）平滑地降低它。
2.  **发现训练慢，果断“从头再来”**: 当处于训练**早期或中期**，发现收敛不理想时，不要犹豫。调整初始超参数（如学习率、Batch Size、数据增强策略），并从一个干净的预训练权重重新开始。这远比在一条错误的道路上“缝缝补补”要高效。
3.  **严禁“开倒车”**: 绝对不要在训练后期或对已收敛模型，将一个已经很小的学习率突然调大。
4.  **优先考虑 AMP**: 当显存不足时，**启用混合精度训练（AMP）是提升 Batch Size 的首选方案**，其次才是考虑减小 `crop_size`，因为后者会牺牲模型的感受野和性能。

### V1.3 出现`NaN` Loss：

**我建议的行动计划如下：**

1. **停止并删除当前的训练任务和输出。**
2. 修改配置文件：
   - 将 `learning_rate` 修改为 `2e-5`。
   - 将 `FocalLoss` 的 `loss_weight` 修改为 `0.5`。
   - ```python
     optim_wrapper = dict(
         type='AmpOptimWrapper',  # 启用混合精度训练
         optimizer=optimizer,
         clip_grad=dict(max_norm=2, norm_type=2),  # max_norm 从10降为2
         loss_scale='dynamic',  # 动态损失缩放:
         paramwise_cfg=dict(
             custom_keys={
                 'pos_block': dict(decay_mult=0.),
                 'norm': dict(decay_mult=0.),
                 'head': dict(lr_mult=3.)  # 解码头使用更低学习率
             }
         )
     )
     param_scheduler = [
         dict(
             type='LinearLR',
             start_factor=5e-7, # 从更小的学习率开始
             by_epoch=True,
             begin=0,
             end=warmup_epochs,
         ),
         dict(
             type='PolyLR',
             eta_min=0.0,
             power=1.0,
             by_epoch=True,
             begin=warmup_epochs,
             end=max_epochs,
         )
     ]
     ```
   - 
3. **从零开始训练**：确保你加载的是原始的 ImageNet 预训练权重 (`mit_b2_converted_from_hf.pth`)，而不是中途出错的检查点。
4. **密切观察**：在新训练开始后，密切关注前几个 epoch 的 `loss` 和 `grad_norm`。它们应该是有值的、平稳的，并且 `grad_norm` 不应出现 `nan` 或 `inf`。

### V1.4 修改

重新返回V1版本，只修改超参数提高学习效率。

***

#### 评估模型代码：

```bash
CONFIG_FILE="configs/segformer/my_segformer_mit-b2_512x512_200e_ade20k.py"
# 【注意】工作目录名会根据配置文件名自动改变
CHECKPOINT_FILE="work_dirs/my_segformer_mit-b2_512x512_200e_ade20k/best_mIoU_epoch_200.pth"

CUDA_VISIBLE_DEVICES=7 python tools/test.py $CONFIG_FILE $CHECKPOINT_FILE --show-dir outputs/my_segformer_mit-b2_512x512_200e_ade20k
```

评估结果如下：

```bash
+---------------------+-------+-------+
|        Class        |  IoU  |  Acc  |
+---------------------+-------+-------+
|         wall        | 67.15 | 84.24 |
|       building      | 76.19 | 89.75 |
|         sky         |  92.8 | 96.28 |
|        floor        | 70.86 | 85.45 |
|         tree        | 66.91 | 82.55 |
|       ceiling       | 76.86 | 87.59 |
|         road        | 74.37 |  83.9 |
|         bed         | 74.93 | 89.79 |
|      windowpane     | 49.28 | 66.96 |
|        grass        | 58.97 | 73.65 |
|       cabinet       |  45.1 | 64.43 |
|       sidewalk      | 51.46 | 73.26 |
|        person       | 60.44 | 82.75 |
|        earth        | 28.62 | 40.68 |
|         door        | 23.29 | 34.51 |
|        table        | 35.22 |  51.5 |
|       mountain      | 45.44 | 61.54 |
|        plant        | 43.97 | 56.34 |
|       curtain       | 54.96 | 72.73 |
|        chair        | 34.94 | 48.77 |
|         car         |  66.4 | 85.14 |
|        water        | 41.64 | 52.21 |
|       painting      | 53.86 | 73.37 |
|         sofa        | 47.81 | 69.64 |
|        shelf        | 30.92 | 44.93 |
|        house        | 34.87 | 53.68 |
|         sea         | 43.52 |  71.0 |
|        mirror       | 39.08 | 50.79 |
|         rug         | 36.33 | 43.27 |
|        field        | 18.19 |  33.2 |
|       armchair      | 23.23 |  36.2 |
|         seat        | 43.28 | 61.57 |
|        fence        | 17.94 | 25.43 |
|         desk        | 26.63 | 40.17 |
|         rock        | 32.33 | 56.22 |
|       wardrobe      | 40.92 | 58.96 |
|         lamp        | 32.78 | 43.36 |
|       bathtub       | 42.77 | 55.14 |
|       railing       | 19.11 |  25.0 |
|       cushion       | 26.29 | 37.42 |
|         base        |  8.38 | 13.93 |
|         box         |  6.13 |  7.55 |
|        column       |  27.1 | 34.03 |
|      signboard      | 15.69 | 19.81 |
|   chest of drawers  | 25.94 | 39.62 |
|       counter       | 22.39 | 31.81 |
|         sand        | 28.88 | 35.97 |
|         sink        | 35.17 | 54.04 |
|      skyscraper     |  60.2 | 71.27 |
|      fireplace      | 47.15 | 70.97 |
|     refrigerator    | 43.39 | 60.99 |
|      grandstand     | 23.44 | 55.99 |
|         path        | 10.98 | 15.29 |
|        stairs       |  20.8 |  25.2 |
|        runway       | 65.27 |  87.8 |
|         case        | 26.78 | 54.86 |
|      pool table     | 75.74 | 91.52 |
|        pillow       | 33.76 | 43.19 |
|     screen door     | 41.41 | 47.61 |
|       stairway      | 19.44 | 24.21 |
|        river        |  7.54 | 20.81 |
|        bridge       | 16.91 | 23.89 |
|       bookcase      | 26.62 | 37.06 |
|        blind        | 13.34 | 14.58 |
|     coffee table    | 34.64 | 65.62 |
|        toilet       | 54.05 | 73.42 |
|        flower       | 23.57 | 36.22 |
|         book        | 24.32 | 30.24 |
|         hill        |  8.17 | 15.48 |
|        bench        | 31.21 | 37.15 |
|      countertop     | 30.52 | 42.78 |
|        stove        | 42.08 | 59.97 |
|         palm        | 27.38 | 31.67 |
|    kitchen island   | 14.08 | 30.61 |
|       computer      | 35.85 | 50.55 |
|     swivel chair    |  22.1 | 36.86 |
|         boat        | 29.31 | 44.59 |
|         bar         | 15.11 | 16.17 |
|    arcade machine   | 10.51 | 14.39 |
|        hovel        | 17.82 | 34.53 |
|         bus         | 44.67 | 70.25 |
|        towel        | 18.72 | 24.73 |
|        light        | 34.31 | 37.72 |
|        truck        |  0.46 |  0.59 |
|        tower        | 15.07 | 19.57 |
|      chandelier     | 43.98 | 59.05 |
|        awning       | 10.41 |  12.2 |
|     streetlight     |  1.71 |  1.76 |
|        booth        |  9.97 | 15.91 |
| television receiver | 34.98 | 50.55 |
|       airplane      | 31.28 | 54.78 |
|      dirt track     |  7.6  | 20.71 |
|       apparel       | 21.06 | 31.08 |
|         pole        |  1.72 |  1.83 |
|         land        |  0.26 |  0.34 |
|      bannister      |  0.01 |  0.01 |
|      escalator      |  2.13 |  3.35 |
|       ottoman       |  4.73 |  5.18 |
|        bottle       |  1.32 |  1.55 |
|        buffet       | 22.07 | 29.64 |
|        poster       |  0.03 |  0.03 |
|        stage        |  3.83 |  7.08 |
|         van         |  6.63 |  7.71 |
|         ship        |  3.33 |  4.19 |
|       fountain      |  0.21 |  0.25 |
|    conveyer belt    |  28.7 | 52.95 |
|        canopy       |  5.9  |  7.13 |
|        washer       | 33.19 | 39.06 |
|      plaything      |  6.08 | 13.78 |
|    swimming pool    | 43.32 | 49.34 |
|        stool        |  1.47 |  1.56 |
|        barrel       |  0.58 |  5.67 |
|        basket       |  2.12 |  2.18 |
|      waterfall      | 54.48 | 70.31 |
|         tent        | 71.38 | 97.19 |
|         bag         |  0.0  |  0.0  |
|       minibike      |  19.2 | 29.55 |
|        cradle       | 53.21 | 75.67 |
|         oven        |  1.37 |  1.63 |
|         ball        |  8.95 |  24.2 |
|         food        | 28.69 | 30.77 |
|         step        |  0.0  |  0.0  |
|         tank        |  5.53 |  5.82 |
|      trade name     |  7.88 |  8.0  |
|      microwave      | 21.53 | 24.85 |
|         pot         |  6.59 |  7.65 |
|        animal       | 30.88 | 32.84 |
|       bicycle       | 14.78 | 23.88 |
|         lake        | 48.57 | 65.44 |
|      dishwasher     | 21.98 |  35.7 |
|        screen       | 60.27 | 80.29 |
|       blanket       |  0.0  |  0.0  |
|      sculpture      |  6.67 |  9.7  |
|         hood        | 17.46 | 22.01 |
|        sconce       |  7.73 |  8.11 |
|         vase        |  4.17 |  4.68 |
|    traffic light    |  3.43 |  3.62 |
|         tray        |  0.0  |  0.0  |
|        ashcan       |  7.76 |  9.75 |
|         fan         | 24.87 | 32.14 |
|         pier        |  9.29 | 15.58 |
|      crt screen     |  0.03 |  0.04 |
|        plate        | 10.12 | 10.83 |
|       monitor       |  0.8  |  0.81 |
|    bulletin board   |  6.82 |  7.3  |
|        shower       |  0.0  |  0.0  |
|       radiator      | 19.68 | 19.92 |
|        glass        |  0.08 |  0.08 |
|        clock        |  0.17 |  0.18 |
|         flag        |  1.56 |  1.6  |
+---------------------+-------+-------+
09/08 11:11:31 - mmengine - INFO - Iter(test) [2000/2000]    aAcc: 74.0900  mIoU: 26.6000  mAcc: 36.2500  data_time: 1.1006  time: 1.1699
```

