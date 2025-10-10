# Mask2Former ADE20K 实践笔记 V1.1

**作者**: Autumnair007
**日期**: 2025-09-09

本文档详细记录了将 `mmsegmentation` 官方提供的 `Mask2Former-Swin-T` 配置文件，从默认的 `IterBased` (基于迭代次数) 训练模式，迁移和重构为更符合个人习惯的 `EpochBased` (基于训练轮次) 模式的全过程。笔记涵盖了配置文件的关键修改点，以及在此过程中遇到的两个典型环境依赖问题及其解决方案。

***

### V1.1 核心目标与修改要点

在成功运行了官方的 `IterBased` 配置后，为了统一实验管理风格（使其与之前的 `SegFormer` 项目对齐），并增强训练过程的可观测性，我决定进行以下重构。

**核心目标：**

1.  **切换训练范式**：将训练循环从按 `iteration` 计数切换为按 `epoch` 计数。
2.  **增强可视化**：集成 `TensorBoard`，以便实时监控损失、mIoU等关键指标。
3.  **提升易用性**：将所有常用超参数集中到文件顶部，方便快速调整和迭代实验。

**配置文件主要修改点总结：**

1.  **训练循环切换 (Epoch-Based)**
    *   `train_cfg` 中的 `type` 从 `IterBasedTrainLoop` 更改为 `EpochBasedTrainLoop`。
    *   `max_iters` 和 `val_interval` (迭代数) 被替换为 `max_epochs` 和 `val_interval` (轮次数)。
    *   数据加载器 `train_dataloader` 中的采样器 `sampler` 从 `InfiniteSampler` 更改为 `DefaultSampler`，这是 Epoch-Based 训练的标准配置。
    *   所有相关的钩子（`hooks`）如 `CheckpointHook` 和 `LoggerHook`，均设置为 `by_epoch=True`。
    *   日志处理器 `log_processor` 也设置为 `by_epoch=True`。

2.  **学习率策略调整**
    *   为对齐 Epoch-Based 训练，并参考 `SegFormer` 的成功经验，引入了 `LinearLR` 学习率预热（Warmup）阶段。
    *   学习率调度器 `param_scheduler` 中的所有配置项均设置为 `by_epoch=True`，并根据 `max_epochs` 和 `warmup_epochs` 调整 `begin` 和 `end` 参数。

3.  **TensorBoard 可视化集成**
    *   在 `vis_backends` 列表中，除了默认的 `LocalVisBackend`，新增了 `dict(type='TensorboardVisBackend')`。

4.  **超参数集中化**
    *   在文件顶部创建了一个专门的配置区域，用于存放 `max_epochs`, `warmup_epochs`, `learning_rate`, `samples_per_gpu` 等所有可调超参数，并添加了详细注释。

***

### 调试历程：从环境依赖到成功运行

在修改完配置后，我尝试启动训练，但连续遇到了两个由环境依赖引起的问题。

#### 问题一：`ModuleNotFoundError: No module named 'mmdet'`

**1. 触发操作与现象**

首次尝试使用分布式训练脚本启动任务。

```bash
# 激活环境
conda activate open-mmlab

# 切换到项目目录
cd ~/projects/mmsegmentation

# 启动分布式训练
CUDA_VISIBLE_DEVICES=5,6,7 ./tools/dist_train.sh configs/mask2former/my_mask2former_swin-t-160k_ade20k-512x512_v1.1.py 3
```

训练立即失败，日志中充满了 `ModuleNotFoundError: No module named 'mmdet'` 以及 `ImportError: Failed to import mmdet.models` 的错误。

**2. 问题分析与根源**

`Mask2Former` 模型虽然属于 `mmsegmentation` 框架，但其设计哲学使其大量复用了来自目标检测框架 `mmdetection` 的先进组件。例如，它的像素解码器 `MSDeformAttnPixelDecoder`、匈牙利匹配器 `HungarianAssigner` 以及多种损失函数（`CrossEntropyLoss`, `DiceLoss` 等）都是通过 `custom_imports = dict(imports='mmdet.models', ...)` 从 `mmdet` 动态导入的。

错误日志明确指出，当前的 `open-mmlab` Python 环境中并未安装 `mmdetection` 库，导致导入失败。

**3. 解决方案**

解决方案非常直接：在当前环境中安装 `mmdetection`。使用 OpenMMLab 官方提供的 `mim` 工具是最佳选择，它能自动处理复杂的依赖关系。

```bash
# 确保 conda 环境已激活
conda activate open-mmlab

# （可选）更新 mim 到最新版本
pip install -U openmim

# 使用 mim 安装 mmdetection
mim install mmdet
```

安装完成后，`mmdet` 依赖问题得到解决。

#### 问题二：`AttributeError: module 'setuptools._distutils' has no attribute 'version'`

**1. 触发操作与现象**

解决了 `mmdet` 的问题后，我再次运行了相同的训练命令。这次程序启动了一小会，但在初始化 `Runner` 和 `Visualizer` 的阶段再次崩溃。错误日志指向了 `torch/utils/tensorboard` 内部。

```
File ".../torch/utils/tensorboard/__init__.py", line 4, in <module>
    LooseVersion = distutils.version.LooseVersion
AttributeError: module 'setuptools._distutils' has no attribute 'version'
```

**2. 问题分析与根源**

这个错误与我的配置文件代码完全无关，是一个典型的 **Python 环境兼容性问题**。

*   当我添加 `TensorboardVisBackend` 后，`mmengine` 会尝试导入 PyTorch 内置的 TensorBoard 工具。
*   该工具为了处理版本号，依赖于一个名为 `distutils` 的旧模块。
*   然而，在较新版本的 `setuptools` 包（通常是 `v60.0.0` 及以上）中，`distutils` 被重构，其内部结构发生了改变。
*   这导致 PyTorch 无法通过旧路径找到 `distutils.version`，从而引发 `AttributeError`。

**3. 解决方案**

解决方法是将 `setuptools` 包降级到一个与当前 PyTorch 版本兼容的旧版本。根据社区经验，`59.6.0` 以下的版本通常是安全的。

```bash
# 确保 conda 环境已激活
conda activate open-mmlab

# 将 setuptools 降级到指定版本
pip install setuptools==59.5.0
```

执行降级后，环境兼容性问题被彻底解决。

***

### 代码部分

`my_mask2former_swin-t-160k_ade20k-512x512_V1.py `代码如下：

```python
# =========================================================================
#
#        Mask2Former-Swin-T 在 ADE20K 数据集上的
#       训练配置文件 (v1.1 - Epoch-Based & TensorBoard)
#
# 作者: Autumnair007 & Copilot
# 日期: 2025-09-09
#
# --- v1.1 更新内容 ---
# 1. 训练循环由 IterBased 切换为 EpochBased，方便按轮次管理训练。
# 2. 引入学习率预热 (Warmup) 策略，与 Epoch-Based 训练对齐。
# 3. 添加 TensorBoard 可视化后端，便于监控训练过程。
# 4. 集中管理所有超参数，并添加详细注释，方便快速调整。
# 5. 调整数据加载器采样器为 DefaultSampler，以适应 Epoch-Based 训练。
#
# =========================================================================

# --- 第 1 部分: 超参数与可修改配置 ---
# -- 硬件与训练超参数
samples_per_gpu = 4       # 每张 GPU 的批大小 (Batch Size)
num_workers = 8           # 数据加载器的工作线程数
max_epochs = 200          # 最大训练轮次
warmup_epochs = 20         # 学习率预热的轮次
val_interval = 10          # 每隔多少轮进行一次验证
checkpoint_interval = 10   # 每隔多少轮保存一次检查点

# -- 模型与优化器超参数
pretrained = 'checkpoints/swin_tiny_patch4_window7_224_20220317-1cdeb081.pth' # 预训练模型路径
learning_rate = 0.0001    # 基础学习率
weight_decay = 0.05       # AdamW 优化器的权重衰减

# -- 数据集配置
data_root = 'data/ADEChallengeData2016' # 数据集根目录
dataset_type = 'ADE20KDataset'
num_classes = 150         # ADE20K 类别数
crop_size = (512, 512)    # 训练时随机裁剪的尺寸
# 数据归一化参数
mean = [123.675, 116.28, 103.53]
std = [58.395, 57.12, 57.375]

# =================================================================

# --- 第 2 部分: 基础配置与导入 ---
# -- 自定义导入 (依赖 mmdetection)
custom_imports = dict(imports='mmdet.models', allow_failed_imports=False)

# -- 数据预处理器
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=mean,
    std=std,
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size=crop_size,
    test_cfg=dict(size_divisor=32))

# =================================================================

# --- 第 3 部分: 模型定义 ---
# -- Swin Transformer 骨干网络参数
depths = [2, 2, 6, 2]

# -- 完整模型结构
model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='SwinTransformer',
        embed_dims=96,
        depths=depths,
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.3,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        with_cp=False,
        frozen_stages=-1,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    decode_head=dict(
        type='Mask2FormerHead',
        in_channels=[96, 192, 384, 768], # 对应 Swin-T
        strides=[4, 8, 16, 32],
        feat_channels=256,
        out_channels=256,
        num_classes=num_classes,
        num_queries=100,
        num_transformer_feat_level=3,
        align_corners=False,
        pixel_decoder=dict(
            type='mmdet.MSDeformAttnPixelDecoder',
            num_outs=3,
            norm_cfg=dict(type='GN', num_groups=32),
            act_cfg=dict(type='ReLU'),
            encoder=dict(
                num_layers=6,
                layer_cfg=dict(
                    self_attn_cfg=dict(
                        embed_dims=256, num_heads=8, num_levels=3, num_points=4,
                        im2col_step=64, dropout=0.0, batch_first=True),
                    ffn_cfg=dict(
                        embed_dims=256, feedforward_channels=1024, num_fcs=2,
                        ffn_drop=0.0, act_cfg=dict(type='ReLU', inplace=True)))),
            positional_encoding=dict(num_feats=128, normalize=True)),
        enforce_decoder_input_project=False,
        positional_encoding=dict(num_feats=128, normalize=True),
        transformer_decoder=dict(
            return_intermediate=True,
            num_layers=9,
            layer_cfg=dict(
                self_attn_cfg=dict(
                    embed_dims=256, num_heads=8, attn_drop=0.0, proj_drop=0.0, batch_first=True),
                cross_attn_cfg=dict(
                    embed_dims=256, num_heads=8, attn_drop=0.0, proj_drop=0.0, batch_first=True),
                ffn_cfg=dict(
                    embed_dims=256, feedforward_channels=2048, num_fcs=2,
                    act_cfg=dict(type='ReLU', inplace=True), ffn_drop=0.0, add_identity=True))),
        loss_cls=dict(
            type='mmdet.CrossEntropyLoss', use_sigmoid=False, loss_weight=2.0,
            class_weight=[1.0] * num_classes + [0.1]),
        loss_mask=dict(
            type='mmdet.CrossEntropyLoss', use_sigmoid=True, loss_weight=5.0),
        loss_dice=dict(
            type='mmdet.DiceLoss', use_sigmoid=True, activate=True, eps=1.0, loss_weight=5.0),
        train_cfg=dict(
            num_points=12544, oversample_ratio=3.0, importance_sample_ratio=0.75,
            assigner=dict(
                type='mmdet.HungarianAssigner',
                match_costs=[
                    dict(type='mmdet.ClassificationCost', weight=2.0),
                    dict(type='mmdet.CrossEntropyLossCost', weight=5.0, use_sigmoid=True),
                    dict(type='mmdet.DiceCost', weight=5.0, pred_act=True, eps=1.0)]),
            sampler=dict(type='mmdet.MaskPseudoSampler'))),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

# =================================================================

# --- 第 4 部分: 数据集与数据加载器 ---
# 训练数据处理流水线
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(
        type='RandomChoiceResize',
        scales=[int(512 * x * 0.1) for x in range(5, 21)],
        resize_type='ResizeShortestEdge',
        max_size=2048),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]

# 测试数据处理流水线
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(2048, 512), keep_ratio=True),
    dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(type='PackSegInputs')
]

# 数据加载器
train_dataloader = dict(
    batch_size=samples_per_gpu,
    num_workers=num_workers,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True), # <-- 修改: 适应 Epoch-Based
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='images/training', seg_map_path='annotations/training'),
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
            img_path='images/validation',
            seg_map_path='annotations/validation'),
        pipeline=test_pipeline))

test_dataloader = val_dataloader

# 评估器
val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = val_evaluator

# =================================================================

# --- 第 5 部分: 优化器与学习率策略 ---
# 优化器封装 (Swin Transformer 定制化学习率)
backbone_norm_multi = dict(lr_mult=0.1, decay_mult=0.0)
backbone_embed_multi = dict(lr_mult=0.1, decay_mult=0.0)
embed_multi = dict(lr_mult=1.0, decay_mult=0.0)
custom_keys = {
    'backbone': dict(lr_mult=0.1, decay_mult=1.0),
    'backbone.patch_embed.norm': backbone_norm_multi,
    'backbone.norm': backbone_norm_multi,
    'absolute_pos_embed': backbone_embed_multi,
    'relative_position_bias_table': backbone_embed_multi,
    'query_embed': embed_multi,
    'query_feat': embed_multi,
    'level_embed': embed_multi
}
custom_keys.update({
    f'backbone.stages.{stage_id}.blocks.{block_id}.norm': backbone_norm_multi
    for stage_id, num_blocks in enumerate(depths)
    for block_id in range(num_blocks)
})
custom_keys.update({
    f'backbone.stages.{stage_id}.downsample.norm': backbone_norm_multi
    for stage_id in range(len(depths) - 1)
})

optimizer = dict(
    type='AdamW', lr=learning_rate, weight_decay=weight_decay, eps=1e-8, betas=(0.9, 0.999))
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=optimizer,
    clip_grad=dict(max_norm=0.01, norm_type=2),
    paramwise_cfg=dict(custom_keys=custom_keys, norm_decay_mult=0.0))

# 学习率调度器 (Warmup + Poly)
param_scheduler = [
    # 线性预热
    dict(
        type='LinearLR',
        start_factor=1e-6,
        by_epoch=True, # <-- 修改: 按 Epoch
        begin=0,
        end=warmup_epochs),
    # 多项式衰减
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=0.9,
        by_epoch=True, # <-- 修改: 按 Epoch
        begin=warmup_epochs,
        end=max_epochs)
]

# =================================================================

# --- 第 6 部分: 训练、验证与测试循环 ---
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=val_interval)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# =================================================================

# --- 第 7 部分: 钩子 (Hooks) 与可视化 ---
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=True), # <-- 修改: 按 Epoch 记录
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=True, # <-- 修改: 按 Epoch 保存
        interval=checkpoint_interval,
        save_best='mIoU',
        max_keep_ckpts=3,
        rule='greater'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'))

# 可视化后端 (本地 + TensorBoard)
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend') # <-- 新增: TensorBoard
]
visualizer = dict(
    type='SegLocalVisualizer', vis_backends=vis_backends, name='visualizer')

# =================================================================

# --- 第 8 部分: 默认运行时设置 ---
default_scope = 'mmseg'
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)
log_processor = dict(by_epoch=True) # <-- 修改: 按 Epoch 处理日志
log_level = 'INFO'
load_from = None
resume = False
tta_model = dict(type='SegTTAModel')
auto_scale_lr = dict(enable=False, base_batch_size=16)
```

### 最终运行与监控命令

在解决了所有环境问题后，训练终于可以顺利启动。

#### 1. 启动训练会话 (使用 tmux)

为了确保训练过程在终端关闭后也能持续运行，我使用了 `tmux` 进行会话管理。

```bash
# 创建一个新的 tmux 会话，命名为 mask2former_ade_v1_1
tmux new-session -s mask2former_ade_v1_1

# 在 tmux 会话中，激活 conda 环境
conda activate open-mmlab

# 切换到项目目录
cd ~/projects/mmsegmentation

# 运行分布式训练脚本
CUDA_VISIBLE_DEVICES=5,6,7 ./tools/dist_train.sh configs/mask2former/my_mask2former_swin-t-160k_ade20k-512x512_v1.1.py 3
```

现在可以安全地分离会话（快捷键 `Ctrl+b` 然后按 `d`），训练将在后台继续。

#### 2. 监控训练过程 (TensorBoard)

打开一个新的终端，用于启动 TensorBoard 前端。

```bash
# 激活环境
conda activate open-mmlab

# 启动 TensorBoard，并将其指向 mmsegmentation 的工作目录
tensorboard --logdir ~/projects/mmsegmentation/work_dirs
```

在浏览器中打开 `http://localhost:6006/`，即可实时查看 `mIoU`、`loss`、学习率变化等图表，对训练状态一目了然。

#### 3. 恢复与管理 tmux 会话

```bash
# 恢复（附加到）之前的会话
tmux attach -t mask2former_ade_v1_1

# 训练结束后，删除会话
tmux kill-session -t mask2former_ade_v1_1
```

通过以上步骤，我成功地将 `Mask2Former` 的训练流程整合到了自己熟悉的 `Epoch-Based` 工作流中，并解决了所有环境障碍，为后续的实验和调优打下了坚实的基础。
