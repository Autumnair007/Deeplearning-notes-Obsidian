# Segformer 实验增强版 V3

### 实验流程分析与总结

你所执行的整个 V2 流程，从“激进策略”的失败到“集大成者”的成功，是一个非常宝贵的实践案例。它完美地展示了从“知道要用什么”到“知道该怎么用”的进阶过程。

1.  **科学的迭代闭环**: 你完整地走了一个 `提出假设 -> 实验 -> 分析失败 -> 修正方案 -> 再次实验 -> 获得提升` 的闭环。这是解决复杂问题的最有效路径。特别是从 V2 初版的失败中吸取教训，定位到 `gamma` 参数问题的分析，非常精准。

2.  **对“提升幅度小”的看法**: 首先，我要强调，在像 PASCAL VOC 这样成熟的数据集上，当模型达到一定瓶颈后，mIoU 从 58.6% 提升到 59.33% **并不算是一个微小的进步**。这 0.73 个点的提升是你在正确分析问题后，通过引入更复杂的混合损失函数和数据增强策略“硬啃”下来的，含金量很高。在学术界，很多论文的改进也就集中在 1-2 个百分点。所以，请先肯定自己的成果！

3.  **成功的核心要素**:
    *   **混合损失函数是灵魂**: 你最终的成功，关键在于从单一的 `FocalLoss` 转向了 `FocalLoss` + `DiceLoss` 的组合。这体现了一个深刻的理解：
        *   `FocalLoss` 解决了**像素层面**的类别不平衡（易分/难分样本）。
        *   `DiceLoss` 解决了**区域/结构层面**的类别不平衡（关注预测掩码与真实掩码的重叠度，对小物体更友好）。
        *   二者结合，让模型既能看清“树木”（像素细节），又能看见“森林”（物体结构），是 1+1 > 2 的典范。
    *   **数据增强的协同作用**: `PhotoMetricDistortion` 在一个鲁棒的损失函数（混合损失）的指导下，终于发挥了它应有的正面作用——增强模型泛化能力，而不是加速梯度消失。

**总结**: 你的 V2 实验是一次高质量的优化实践。你不仅提升了模型的性能，更重要的是，你通过亲手实践，深刻理解了高级算法（如 Focal Loss）的超参数敏感性以及不同损失函数之间的协同机制。

---

### **下一步改进方案建议 (V3 探索方向)**

现在，模型性能已经站上了一个新的台阶。要继续突破，我们需要从更多维度进行精细化调优和“压榨”模型的潜力。以下是一些你可以尝试的方向，我将它们从**低成本、易于尝试**到**高成本、更复杂**的顺序列出：

#### **方向一：超参数与训练策略微调 (低成本)**

你的 V2 最终版配置已经非常完善，但仍有微调空间。这些改动不需要大的代码重写，只需修改配置文件。

1. **调整学习率 (Learning Rate)**:

   *   **尝试更低的学习率**: 你的 `learning_rate` 是 `6e-5`。既然模型已经比较成熟，可以尝试用当前最好的模型 (`best_mIoU_epoch_200.pth`) 作为预训练权重，进行一个“微调（fine-tuning）”阶段的训练。在这个阶段，使用一个更小的学习率，例如 `1e-5` 或 `6e-6`，再训练 50-100 个 epochs。这有助于模型在当前的位置进行更精细的搜索，可能会找到一个更好的局部最优点。
   *   **调整学习率调度器**: 你使用了 `LinearLR` (warmup) + `PolyLR`。这是一个非常标准且强大的组合。可以尝试延长 `warmup_epochs` 到 `20` 或 `25`，让模型在初始阶段有更平滑的启动过程，尤其是在使用了复杂数据增强和损失函数时。


下面是具体的操作步骤和出现的问题：

***

### 为什么修改的参数不会生效？

当你使用 `--resume` 标志时，`mmengine` 不仅仅是加载了模型的权重 (`state_dict`)，它还会加载一个完整的训练状态快照，其中包含：

1.  **模型权重**：这部分当然会加载。
2.  **优化器状态 (Optimizer State)**：对于像 AdamW 这样的优化器，它会保存每个参数的动量（momentums）和方差（variances）。这些状态是基于*旧的*学习率计算的。
3.  **学习率调度器状态 (Scheduler State)**：这是最关键的一点。调度器会记录它自己已经执行到了哪一步。在你的情况中，它记录着：“我已经完成了 20 个 epoch 的 warmup，并且已经执行 `PolyLR` 策略到了第 200 个 epoch”。
4.  **当前的 Epoch 和 Iteration 数**：日志明确显示 `resumed epoch: 200`。

因此，当你恢复训练时，会发生以下情况：

*   **`max_epochs = 400`**: 这个参数**会生效**。训练循环会从 epoch 200 开始，一直跑到 400。
*   **`learning_rate = 0.000006`**: 这个新的基础学习率**不会生效**。优化器和调度器会从检查点中恢复它们的状态，继续使用基于*旧的学习率（0.00006）* 计算出的衰减曲线。
*   **`warmup_epochs = 20`** 和 **`start_factor=1e-7`**: 这些参数**完全不会生效**。因为恢复的 epoch 是 200，已经远远超过了 warmup 阶段（0-20 epochs）。调度器会直接跳过 `LinearLR` 部分，继续执行它在第 200 个 epoch 时的 `PolyLR` 状态。

**总结一下：`--resume` 的设计哲学是“精确地从中断的地方继续”，它会忽略掉那些可能与已保存状态冲突的新配置。**

### 如果想让新参数生效，应该怎么做？

如果你想在一个已经训练好的模型基础上，用*全新的*超参数（比如更小的学习率）来微调（fine-tune）模型，你不应该使用 `--resume`。

你应该使用 `load_from`。

**解决方案：**

1. **修改配置文件**:

   *   确保你的配置文件中设置了所有**新**的超参数（`learning_rate`, `max_epochs`, `warmup_epochs`, `param_scheduler` 等）。
   *   在配置文件中，添加 `load_from` 字段，指向你的预训练模型。
   *   **删除或注释掉**模型 backbone 中的 `init_cfg`，因为 `load_from` 会在更高层级上加载权重，避免冲突。

   你的配置文件应该看起来像这样：

   ```python name=my_segformer_mit-b2_3xb6-400e_finetune.py
   # =========================================================================
   #
   #        SegFormer-MiT-B2 在 PASCAL VOC 2012 增强数据集上的
   #       训练配置文件 (v3 - 修改超参数进行微调训练)
   #
   # 作者: Autumnair007
   # 日期: 2025-09-01 (修改学习率，warmup和参数调度器的超参数进行微调训练)
   #
   # =========================================================================
   
   # --- 第 1 部分: 继承基础配置 (只继承模型和运行时) ---
   _base_ = [
       '../_base_/models/segformer_mit-b0.py',
       '../_base_/default_runtime.py'
   ]
   
   # --- 第 2 部分: 硬件与训练超参数 ---
   gpu_count = 3
   samples_per_gpu = 6
   num_workers = 8
   learning_rate = 0.000006 # 新的学习率
   checkpoint_epoch = 10
   val_epoch = 10
   max_epochs = 200
   warmup_epochs = 20 # 新的 warmup
   
   # --- 第 3 部分: 数据集配置 ---
   dataset_type = 'PascalVOCDataset'
   data_root = 'data/VOCdevkit/VOC2012'
   crop_size = (512, 512)
   
   # 训练数据处理流水线 (包含光度畸变)
   train_pipeline = [
       dict(type='LoadImageFromFile'),
       dict(type='LoadAnnotations'),
       dict(
           type='RandomResize',
           scale=(2048, 512),
           ratio_range=(0.5, 2.0),
           keep_ratio=True),
       dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
       dict(type='RandomFlip', prob=0.5),
       dict(type='PhotoMetricDistortion'),  # 光度畸变
       dict(type='Pad', size=crop_size),
       dict(type='PackSegInputs')
   ]
   
   # 测试数据处理流水线
   test_pipeline = [
       dict(type='LoadImageFromFile'),
       dict(type='Resize', scale=(2048, 512), keep_ratio=True),
       dict(type='LoadAnnotations'),
       dict(type='PackSegInputs')
   ]
   
   # 定义训练数据集
   dataset_train = dict(
       type=dataset_type,
       data_root=data_root,
       data_prefix=dict(img_path='JPEGImages', seg_map_path='SegmentationClass'),
       ann_file='ImageSets/Segmentation/train.txt',
       pipeline=train_pipeline)
   
   # 定义增强数据集
   dataset_aug = dict(
       type=dataset_type,
       data_root=data_root,
       data_prefix=dict(
           img_path='JPEGImages', seg_map_path='SegmentationClassAug'),
       ann_file='ImageSets/Segmentation/aug.txt',
       pipeline=train_pipeline)
   
   # --- 第 4 部分: 数据加载器配置 ---
   train_dataloader = dict(
       batch_size=samples_per_gpu,
       num_workers=num_workers,
       persistent_workers=True,
       # 将 InfiniteSampler 换回 DefaultSampler 以支持按 Epoch 训练
       sampler=dict(type='DefaultSampler', shuffle=True),
       dataset=dict(type='ConcatDataset', datasets=[dataset_train, dataset_aug]))
   
   val_dataloader = dict(
       batch_size=1,
       num_workers=num_workers,
       persistent_workers=True,
       sampler=dict(type='DefaultSampler', shuffle=False),
       dataset=dict(
           type=dataset_type,
           data_root=data_root,
           data_prefix=dict(
               img_path='JPEGImages', seg_map_path='SegmentationClass'),
           ann_file='ImageSets/Segmentation/val.txt',
           pipeline=test_pipeline))
   
   test_dataloader = val_dataloader
   
   # 评估器配置
   val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
   test_evaluator = val_evaluator
   
   # --- 第 5 部分: 模型配置 ---
   data_preprocessor = dict(size=crop_size)
   # 不再需要这个，因为 load_from 会处理
   # checkpoint = 'checkpoints/mit_b2_converted_from_hf.pth' 
   
   model = dict(
       data_preprocessor=data_preprocessor,
       backbone=dict(
           # 【重要】注释或删除这里的 init_cfg，避免和 load_from 冲突
           # init_cfg=dict(type='Pretrained', checkpoint=checkpoint),
           embed_dims=64,
           num_layers=[3, 4, 6, 3]),
       decode_head=dict(
           in_channels=[64, 128, 320, 512],
           num_classes=21,
           loss_decode=[
               dict(
                   type='FocalLoss',
                   use_sigmoid=True,
                   loss_weight=1.0),
               dict(
                   type='DiceLoss',
                   loss_weight=1.0,
                   ignore_index=255)
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
           start_factor=1e-7, # 从 1e-6 变成 1e-7
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
   
   # 【关键】在这里添加 load_from
   load_from = '/home/qz/projects/mmsegmentation/work_dirs/my_segformer_mit-b2_3xb6-200e_voc12aug_v2-advanced-training/epoch_200.pth'
   
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

2. **修改并运行命令**:

   *   **去掉 `--resume` 标志！**
   *   运行新的训练命令。

   ```bash
   # 注意，没有 --resume
   CUDA_VISIBLE_DEVICES=5,6,7 ./tools/dist_train.sh configs/segformer/my_segformer_mit-b2_3xb6-400e_finetune.py 3
   ```

**这样做会发生什么？**

*   训练会从 **Epoch 1** 开始。
*   `load_from` 会将 `epoch_200.pth` 文件中模型的**权重**加载到你的新模型中。
*   **但是**，它不会加载优化器和学习率调度器的状态。
*   优化器和调度器会根据你配置文件中的**新设置**（`lr=0.000006`, `warmup_epochs=20` 等）进行全新的初始化。
*   训练会从一个新的、非常低的学习率开始，并按照你新设定的策略（20个epoch的warmup，然后Poly衰减）进行训练，直到新的200个epoch。

***

#### tmux的命令如下：

* **创建会话**: 

  ```bash
  tmux new-session -s seg_train_b6_v3
  conda activate open-mmlab
  ```

* **恢复会话：**

  ```bash
  tmux attach -t seg_train_b6_v3
  ```

* **删除旧会话**: 

  ```bash
  tmux kill-session -t seg_train_b6_v3
  ```

#### 评估模型代码：

```bash
CONFIG_FILE="configs/segformer/my_segformer_mit-b2_3xb6-400e_finetune.py"
# 【注意】工作目录名会根据配置文件名自动改变
CHECKPOINT_FILE="work_dirs/my_segformer_mit-b2_3xb6-400e_finetune/best_mIoU_epoch_190.pth"

CUDA_VISIBLE_DEVICES=5 python tools/test.py $CONFIG_FILE $CHECKPOINT_FILE --show-dir outputs/segformer_400e_finetune
```

评估结果如下：

```bash
这里是150 epoch的结果：
+-------------+-------+-------+
|    Class    |  IoU  |  Acc  |
+-------------+-------+-------+
|  background | 90.94 | 95.65 |
|  aeroplane  | 76.71 | 85.64 |
|   bicycle   | 33.96 | 79.44 |
|     bird    | 62.99 | 76.09 |
|     boat    | 50.77 | 67.97 |
|    bottle   | 48.41 | 54.84 |
|     bus     | 82.59 | 86.42 |
|     car     | 73.05 | 87.85 |
|     cat     | 72.23 | 86.61 |
|    chair    | 22.33 | 33.55 |
|     cow     | 61.88 | 72.02 |
| diningtable | 37.73 | 45.04 |
|     dog     | 61.02 | 78.91 |
|    horse    |  61.2 | 76.94 |
|  motorbike  | 69.35 | 82.36 |
|    person   | 73.44 | 87.52 |
| pottedplant |  41.6 | 52.78 |
|    sheep    | 68.72 | 77.02 |
|     sofa    | 34.79 | 46.26 |
|    train    | 76.77 | 84.26 |
|  tvmonitor  | 56.26 | 67.13 |
+-------------+-------+-------+
09/02 12:23:44 - mmengine - INFO - Iter(test) [1449/1449]    aAcc: 90.4000  mIoU: 59.8400  mAcc: 72.5900  data_time: 0.9260  time: 0.9995
下面是190 epoch的结果：
+-------------+-------+-------+
|    Class    |  IoU  |  Acc  |
+-------------+-------+-------+
|  background | 91.01 | 95.78 |
|  aeroplane  | 77.28 |  85.0 |
|   bicycle   | 34.56 | 77.75 |
|     bird    | 63.54 |  77.7 |
|     boat    | 51.38 | 67.89 |
|    bottle   | 49.63 |  56.3 |
|     bus     | 84.15 | 88.05 |
|     car     | 75.09 | 87.86 |
|     cat     | 72.86 | 86.85 |
|    chair    | 22.06 | 35.15 |
|     cow     | 61.15 | 72.73 |
| diningtable | 37.56 | 44.56 |
|     dog     | 61.43 | 79.06 |
|    horse    | 59.17 | 73.48 |
|  motorbike  | 69.51 |  82.2 |
|    person   | 73.76 | 87.53 |
| pottedplant | 41.01 | 52.06 |
|    sheep    | 68.71 | 77.45 |
|     sofa    | 33.33 | 43.04 |
|    train    | 77.95 | 84.85 |
|  tvmonitor  | 56.12 | 66.49 |
+-------------+-------+-------+
09/02 16:09:23 - mmengine - INFO - Iter(test) [1449/1449]    aAcc: 90.4900  mIoU: 60.0600  mAcc: 72.4700  data_time: 0.8988  time: 0.9732
```

### **方向二：数据增强“军火库”升级 (中等成本)**

在完成了初步的微调后，我们进入了实验的第二阶段。此阶段的目标是通过引入更高级的数据增强技术和对损失函数进行精细调整，来深度挖掘模型的潜力。这些方法的计算成本适中，但往往能带来显著的性能提升。

#### **（1）损失权重调整：从“均衡”到“精准打击”**

你的模型目前同时使用 `FocalLoss` 和 `DiceLoss`，权重均为 `1.0`。这是一个非常稳健的起点，但为了突破性能瓶颈，我们需要根据模型的具体表现进行针对性调整。

**A. 现状分析 (基于你的 mIoU: 59.84 测试结果)**

*   **核心问题诊断**: 你的模型在分割**大块、连续**的物体（如 `background`, `bus`, `aeroplane`）时表现出色，但在处理**小物体**（如 `pottedplant`, `bottle`）和**结构复杂、有细长部分**的物体（如 `chair`, `bicycle`, `sofa`）时，IoU 明显偏低。这说明模型对物体的整体结构和精细边界的感知能力是当前的短板。

**B. 损失函数特性与调整策略**

*   **`FocalLoss` (像素级)**: 它关注的是像素分类的准确性，擅长处理难分类的边界像素，但可能会忽略物体的整体结构，对小目标的损失贡献也较小。
*   **`DiceLoss` (区域级)**: 它关注的是预测区域与真实区域的重合度，对物体的**轮廓和结构**非常敏感，即使对很小的物体也能产生有效的惩罚梯度。

**C. 推荐方案：强化对结构和小物体的关注**

根据上述分析，我建议**加大 `DiceLoss` 的影响力**，迫使模型在训练中更加关注它不擅长的方面。

*   **推荐权重配比**:
    *   `loss_weight` for `FocalLoss`: **0.7**
    *   `loss_weight` for `DiceLoss`: **1.3**

*   **调整理由**:
    1.  **强化结构感知**: 提高 `DiceLoss` 的权重，会驱动模型更努力地学习 `bicycle`、`chair` 等物体的复杂结构。
    2.  **提升小目标性能**: `DiceLoss` 对小目标的惩罚更有效，增加其权重能让模型在训练中更加“在意”那些容易被忽略的小物体。
    3.  **保留像素级优化**: 同时保留权重为 `0.7` 的 `FocalLoss`，可以确保模型在优化整体结构的同时，不会丢失像素级别的精度。

#### **（2）引入更多“猛料”：高级数据增强**

你的流水线中已包含核心的几何与色彩增强。现在，我们将引入两种更强大的增强策略，它们通过创造更具挑战性的训练样本来提升模型的鲁棒性和泛化能力。

**1. CutOut / Random Erasing**

*   **是什么**: 在图像上随机挖掉一个或多个矩形区域，并用固定的值填充。
*   **为什么有效**: 这种“遮挡”强迫模型利用物体的**上下文信息**进行预测（例如，根据桌子和地板来判断被遮挡的椅子部分），而不是仅仅依赖于物体某个孤立的局部特征。这能极大地提高模型在真实世界中应对遮挡情况的鲁棒性。

**2. MixUp / CutMix**

*   **是什么**: 这是两种更高级的“混合”类增强策略。`MixUp` 将两张图片按一定比例进行像素级线性混合。`CutMix` 则是将一张图的一部分裁切下来，直接粘贴覆盖到另一张图的随机位置上。两种方法的标签也会进行相应的混合。
*   **为什么有效**: 它们创造了在真实世界中不存在、但极具挑战性的虚拟样本，极大地扩展了训练数据的分布。这是一种非常强大的正则化手段，能有效抑制模型过拟合，显著提升其泛化能力。

***

#### **方向三：模型与数据策略的变革 (高成本)**

这些是更根本性的改变，可能需要修改更多代码或进行更复杂的数据处理。

1.  **OHEM (Online Hard Example Mining)**:
    *   **是什么**: 不再平等地对待所有像素，而是在线（每个 mini-batch）地只选择那些损失值最高的像素（即最难分的像素）来回传梯度。
    *   **为什么有效**: 这是解决类别不平衡的另一种经典思路。它能让模型更专注于学习那些模棱两可、难以区分的边界或小物体。
    *   **如何实现**: 在 `decode_head` 的 `loss_decode` 中，除了损失类型，还可以配置 `sampler`。你可以将 `CrossEntropyLoss` 或 `FocalLoss` 与 `OHEMPixelSampler` 结合使用。这通常需要你重写 `loss_decode` 部分，将一个损失函数包裹在 OHEM 采样器逻辑中。

2.  **引入额外数据 (External Data)**:
    *   **是什么**: PASCAL VOC 2012 的 `aug` 数据集（SBD）是一个很好的补充。但你还可以考虑引入 COCO 数据集的部分数据进行预训练。
    *   **为什么有效**: 更多、更多样化的数据是提升模型性能上限最朴素也最有效的方法。在 COCO 上预训练过的模型，其特征提取能力通常会比只在 ImageNet 上预训练的模型更强，尤其对于分割任务。
    *   **如何实现**: 这会是一个更复杂的流程。你需要在 COCO 上先训练你的 Segformer，然后将得到的权重作为预训练模型，再到你的 PASCAL VOC 数据集上进行微调。
3.  **升级模型主干 (Backbone)**
    - **是什么**：将当前使用的 SegFormer-MiT-B2 主干网络替换为更大、更强的版本，例如 **MiT-B3** 或 **MiT-B5**。
    - **为什么有效**：更大的主干网络拥有更多的参数和更深/更宽的结构，这意味着它具有更强大的特征提取和表示能力。这种“能力”的提升，可以直接转化为对图像中更复杂上下文关系和更精细细节的理解能力，尤其是在处理像 `chair`、`bicycle` 这类结构复杂的类别时，效果提升可能更明显。这是一种通过增加模型容量来换取性能上限的直接方法。
    - **如何实现**：你需要在模型配置中，修改 `backbone` 字典里的参数，如 `embed_dims` 和 `num_layers`，使之与目标型号（如 MiT-B5）的官方规格匹配。最关键的是，你需要下载并使用对应更大主干网络的**预训练权重**（`load_from`），这是保证其强大性能的关键。
4.  **引入知识蒸馏 (Knowledge Distillation)**
    - **是什么**：这是一种“教师-学生”训练模式。我们选择一个性能非常强大但可能过于庞大而不适合直接部署的“教师模型”（例如，在 ImageNet 或其他大型分割数据集上表现优异的 SegFormer-MiT-B5 或其他SOTA模型），用它来指导我们当前的“学生模型”（SegFormer-MiT-B2）的训练。学生模型不仅从真实的标签中学习，还从教师模型输出的“软知识”（如预测的概率分布）中学习。
    - **为什么有效**：真实的标签是“硬”的（例如，这个像素是“猫”或“不是猫”），而教师模型的输出是“软”的（例如，这个像素有95%的可能是“猫”，3%的可能是“狗”，因为它们有相似的皮毛纹理）。这种软知识包含了类别之间的相似性等更丰富的信息，能引导学生模型学到更鲁棒、泛化能力更强的特征，从而达到超越单独训练的效果。
    - **如何实现**：这通常需要修改训练框架。你需要在配置中同时定义教师模型和学生模型，并额外添加一个“蒸馏损失函数”。这个损失函数用来计算学生模型输出与教师模型输出之间的差异，并将其加入到总的训练损失中。
5.  **利用伪标签进行半监督学习 (Pseudo-Labeling)**
    - **是什么**：这是一种挖掘额外数据潜力的方法。首先，你使用当前训练好的最优模型，对一批**没有标注**的新图像进行预测。然后，你筛选出那些模型预测置信度非常高的预测结果（例如，模型非常有把握地认为某块区域是“汽车”），将这些高置信度的预测结果当作“伪标签” (Pseudo Labels)。最后，将这些带有伪标签的新数据与你已有的手动标注数据合并在一起，共同用于下一轮的模型训练。
    - **为什么有效**：PASCAL VOC 数据集相对较小，模型的学习容易达到瓶颈。通过伪标签，你可以低成本地引入海量、多样的未标注数据，极大地扩充了训练集，让模型见到更广泛的场景，从而提升其泛化能力和鲁棒性。这相当于让模型进行“自我学习和成长”。
    - **如何实现**：这个过程分为几个步骤：(1) 准备一批无标签的图像数据；(2) 使用你最好的模型进行推理预测并保存结果；(3) 编写脚本，根据置信度阈值筛选预测结果，生成伪标签掩码图；(4) 将这些新的伪标签数据整合到你的数据加载流程中，与真实数据一同参与训练。这需要更多的数据处理和流程管理工作。

### **行动建议**

我建议你按照以下顺序，循序渐进地尝试：

1.  **首先尝试方向一**: 进行学习率微调。这是成本最低、见效可能最快的方法。用你现有的 `59.33%` 的模型作为起点，用小 10 倍的学习率再跑 50 个 epoch 看看。
2.  **如果微调效果不明显，再尝试方向二**: 在你的 V2 最终版配置中，加入 `RandomErasing`。这是一个非常实用的增强技术，与你现有的策略能很好地互补。
3.  **如果想追求极致性能，再考虑方向三**: OHEM 或引入外部数据是冲击更高分数的“大招”，但它们需要更多的时间和精力投入。

祝你的 V3 实验顺利，期待你的模型能再次突破性能记录！



