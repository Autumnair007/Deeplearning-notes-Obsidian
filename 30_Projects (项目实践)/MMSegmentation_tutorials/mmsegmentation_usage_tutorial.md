#  MMSegmentation使用教程

学习资料：[(7 封私信 / 22 条消息) 超详细！带你轻松掌握 MMSegmentation 整体构建流程 - 知乎](https://zhuanlan.zhihu.com/p/520397255)

[(7 封私信 / 22 条消息) MMSegmentation的保姆级使用（PyTorch版 . 20240412） - 知乎](https://zhuanlan.zhihu.com/p/692128992)

GitHub仓库：[open-mmlab/mmsegmentation: OpenMMLab Semantic Segmentation Toolbox and Benchmark.](https://github.com/open-mmlab/mmsegmentation)

安装教程：[mmsegmentation/docs/zh_cn/get_started.md at main · open-mmlab/mmsegmentation](https://github.com/open-mmlab/mmsegmentation/blob/main/docs/zh_cn/get_started.md#installation)

------

### 整体流程概览

整个过程可以分为五个主要阶段：

1.  **连接服务器与环境准备**: 使用 PyCharm 连接到你的服务器，并准备好代码和 Python 环境。
2.  **数据准备**: 将你的数据集上传到服务器的正确位置。
3.  **配置训练**: 创建并修改一个配置文件，告诉 `mmpretrain` 你想训练什么模型、用什么数据、以及如何训练。
4.  **开始训练**: 运行一条简单的命令来启动训练过程。
5.  **查看结果**: 找到训练日志和保存的模型文件。

---

###  第零步：确认框架 `mmpretrain` vs `mmsegmentation`

在你开始之前，有一个非常重要的点需要确认：

*   **`mmpretrain`**: 主要用于**图像分类**和**自监督学习预训练**。比如你之前训练的 ConvNeXt 就是一个典型的分类模型。
*   **`mmsegmentation`**: 专门用于**图像语义分割**。DeepLabV3+ 是一个标准的语义分割模型。

虽然 `mmpretrain` 理论上也可以通过自定义来支持分割任务，但这非常复杂且不推荐。**训练 DeepLabV3+ 模型，你应该使用 `mmsegmentation` 框架。**

接下来的教程将以使用更合适的 **`mmsegmentation`** 框架为例，它的操作流程和 `mmpretrain` **几乎完全一样**，只是文件夹名字和一些配置文件里的细节不同。这样可以确保你一次成功！

---

### 第一步：连接服务器与环境准备

这个阶段的目标是在服务器上搭建好“舞台”。

1.  **用 PyCharm 连接服务器**:
    
    *   打开 PyCharm，选择 `File` -> `Remote Development` -> `SSH` -> `New Connection...`。
    *   输入你的服务器登录信息，例如 `ssh username@server_ip_address`，然后输入密码或设置 SSH 密钥。
    *   PyCharm 会在服务器上部署一个轻量级的客户端，稍等片刻后，你就可以像在本地一样打开服务器上的项目了。
    
2. **安装环境并下载 `mmsegmentation` 框架**:

   ```
   步骤 0. 使用 MIM 安装 MMCV
   激活虚拟环境，在虚拟环境里面使用：
   conda activate open-mmlab
   pip install -U openmim
   mim install mmengine
   mim install "mmcv<2.2.0,>=2.0.0rc4"
   步骤 1. 安装 MMSegmentation
   
   情况 a: 如果您想立刻开发和运行 mmsegmentation，您可通过源码安装：
   
   git clone -b main https://github.com/open-mmlab/mmsegmentation.git
   cd mmsegmentation
   pip install -v -e .
   pip install ftfy regex
   # '-v' 表示详细模式，更多的输出
   # '-e' 表示以可编辑模式安装工程，
   # 因此对代码所做的任何修改都生效，无需重新安装
   ```

---

### 第二步：准备你的数据集

这个阶段是把你要用来训练的“食材”放到指定位置。

1.  **创建 `data` 文件夹**:
    *   在 `mmsegmentation` 的项目根目录里，手动创建一个名为 `data` 的文件夹。这是一个约定俗成的习惯，用来存放所有数据。

2.  **上传并组织数据集**:
    
    *   假设你的数据集叫做 `MyDataset`。在 `data` 文件夹里，再创建一个 `MyDataset` 文件夹。
    *   将你的图片和标注文件上传到服务器。对于语义分割，通常包含两个子文件夹：
        *   `img_dir`: 存放所有原始训练图片（比如 `.jpg`, `.png`）。
        *   `ann_dir`: 存放所有对应的标注图片（通常是单通道的 `.png` 文件，每个像素的颜色值代表一个类别）。
    *   最终的目录结构看起来像这样：
        ```
        mmsegmentation/
        ├── data/
        │   └── MyDataset/
        │       ├── img_dir/
        │       │   ├── train/
        │       │   │   ├── 0001.jpg
        │       │   │   └── 0002.jpg
        │       │   └── val/
        │       │       ├── 0003.jpg
        │       │       └── 0004.jpg
        │       ├── ann_dir/
        │       │   ├── train/
        │       │   │   ├── 0001.png
        │       │   │   └── 0002.png
        │       │   └── val/
        │       │       ├── 0003.png
        │       │       └── 0004.png
        ```
    *   你可以直接通过 PyCharm 的文件浏览器把本地文件拖拽到服务器的目录里，非常方便。

---

**前面是准备操作，下面是具体的一个微调模型的操作步骤，可参考。**

### 全实战终极教程：快速测试 DeepLabV3+ 训练流程

本教程将指导你如何通过**自创的虚拟数据集**，来快速、无痛地验证 `mmsegmentation` 的完整训练流程。这个方法无需下载庞大的数据集，并能帮你规避所有我们曾遇到过的“坑”。

**最终目标**：成功运行训练命令，并看到迭代日志（`Epoch(train) [1][10/200] ...`）开始滚动。

假设你已位于 `mmsegmentation` 项目根目录，并激活了你的 conda 环境。

### 第一步：创建虚拟数据集（我们自己的脚本）

我们首先创建一个 Python 脚本来凭空生成测试数据，摆脱对任何外部数据集或官方脚本的依赖。

1.  **新建 `create_dummy_data.py` 文件**:
    在 `mmsegmentation` 根目录下，我们创建一个`myfiles`文件夹，然后再创建一个名为 `create_dummy_data.py` 的文件。

2.  **粘贴代码**:
    将以下代码完整复制到该文件中。

    ```python name=create_dummy_data.py
    import os
    import numpy as np
    from PIL import Image
    
    # --- 配置参数 ---
    # 数据集要创建在哪个目录下
    base_dir = 'data/dummy_dataset'
    # 训练集和验证集的图片数量
    num_train_images = 50
    num_val_images = 10
    # 图片尺寸
    img_width = 256
    img_height = 256
    
    # --- 主函数 ---
    def generate_dataset():
        """生成虚拟的图像分割数据集"""
        print("开始生成虚拟数据集...")
    
        # 创建训练和验证集目录
        train_img_dir = os.path.join(base_dir, 'img_dir', 'train')
        train_ann_dir = os.path.join(base_dir, 'ann_dir', 'train')
        val_img_dir = os.path.join(base_dir, 'img_dir', 'val')
        val_ann_dir = os.path.join(base_dir, 'ann_dir', 'val')
    
        for path in [train_img_dir, train_ann_dir, val_img_dir, val_ann_dir]:
            os.makedirs(path, exist_ok=True)
    
        # --- 生成训练数据 ---
        print(f"正在生成 {num_train_images} 张训练图片...")
        for i in range(num_train_images):
            random_image_array = np.random.randint(0, 256, (img_height, img_width, 3), dtype=np.uint8)
            img = Image.fromarray(random_image_array)
            img.save(os.path.join(train_img_dir, f'train_{i}.jpg'))
    
            mask = np.zeros((img_height, img_width), dtype=np.uint8)
            x1, y1 = np.random.randint(0, img_width // 2), np.random.randint(0, img_height // 2)
            x2, y2 = np.random.randint(img_width // 2, img_width), np.random.randint(img_height // 2, img_height)
            mask[y1:y2, x1:x2] = 1
            ann = Image.fromarray(mask)
            ann.save(os.path.join(train_ann_dir, f'train_{i}.png'))
    
        # --- 生成验证数据 ---
        print(f"正在生成 {num_val_images} 张验证图片...")
        for i in range(num_val_images):
            random_image_array = np.random.randint(0, 256, (img_height, img_width, 3), dtype=np.uint8)
            img = Image.fromarray(random_image_array)
            img.save(os.path.join(val_img_dir, f'val_{i}.jpg'))
    
            mask = np.zeros((img_height, img_width), dtype=np.uint8)
            x1, y1 = np.random.randint(0, img_width // 2), np.random.randint(0, img_height // 2)
            x2, y2 = np.random.randint(img_width // 2, img_width), np.random.randint(img_height // 2, img_height)
            mask[y1:y2, x1:x2] = 1
            ann = Image.fromarray(mask)
            ann.save(os.path.join(val_ann_dir, f'val_{i}.png'))
    
        print(f"\n虚拟数据集生成完毕！位置: {base_dir}")
    
    if __name__ == '__main__':
        generate_dataset()
    ```

3.  **运行脚本**:

    ```bash
    cd myfiles
    python create_dummy_data.py
    cd ..
    ```

### 第二步：下载预训练模型

为了模拟真实的微调（Fine-tuning）过程，我们下载一个官方模型权重。

```bash
# 创建一个存放预训练模型的文件夹
mkdir -p checkpoints

# 下载在 Cityscapes 数据集上预训练的 DeepLabV3+ 模型
wget -P checkpoints https://download.openmmlab.com/mmsegmentation/v0.5/deeplabv3plus/deeplabv3plus_r50-d8_512x1024_40k_cityscapes/deeplabv3plus_r50-d8_512x1024_40k_cityscapes_20200605_094610-d222ffcd.pth
```

### 第三步：创建最终的配置文件

这是整个流程的核心。下面的这份配置文件，是我们经历了所有调试后得到的**最终正确版本**。

1.  **新建配置文件**:
    在 `configs/deeplabv3plus/` 目录下，创建新文件 `deeplabv3plus_r50-d8_dummy-test.py`。

2.  **粘贴最终配置代码**:
    将以下**全部代码**复制粘贴到新文件中。

    ````python name=configs/deeplabv3plus/deeplabv3plus_r50-d8_dummy-test.py
    # 基础模型配置
    _base_ = [
        '../_base_/models/deeplabv3plus_r50-d8.py',
        '../_base_/datasets/pascal_voc12.py', # 仅借用其数据处理流程结构
        '../_base_/default_runtime.py',
        '../_base_/schedules/schedule_20k.py'
    ]
    
    # 定义图像裁切大小，方便复用
    crop_size = (256, 256)
    
    # --- 1. 修改模型，适应我们的2分类虚拟数据 ---
    model = dict(
        # 关键修正4: 为数据预处理器明确指定打包尺寸
        data_preprocessor=dict(
            type='SegDataPreProcessor',
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            bgr_to_rgb=True,
            pad_val=0,
            seg_pad_val=255,
            size=crop_size), # 明确指定对齐尺寸，解决 size_divisor 错误
        decode_head=dict(num_classes=2),
        auxiliary_head=dict(num_classes=2)
    )
    
    # --- 2. 修改数据集信息 ---
    data_root = 'myfiles/data/dummy_dataset'
    
    train_pipeline = [
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations'),
        dict(type='RandomResize', scale=(512, 256), ratio_range=(0.5, 2.0), keep_ratio=True),
        dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
        dict(type='RandomFlip', prob=0.5),
        dict(type='PackSegInputs')
    ]
    
    test_pipeline = [
        dict(type='LoadImageFromFile'),
        dict(type='Resize', scale=(512, 256), keep_ratio=True),
        dict(type='LoadAnnotations'),
        dict(type='PackSegInputs')
    ]
    
    # 修改数据加载器
    train_dataloader = dict(
        batch_size=4,
        dataset=dict(
            # 关键修正2: 使用通用基类，避免类别名冲突
            type='BaseSegDataset',
            data_root=data_root,
            # 关键修正3: 明确指定无清单文件，让程序扫描文件夹
            ann_file='',
            data_prefix=dict(img_path='img_dir/train', seg_map_path='ann_dir/train'),
            pipeline=train_pipeline,
            metainfo=dict(classes=('background', 'foreground'), palette=[[120, 120, 120], [6, 230, 230]])
        )
    )
    
    val_dataloader = dict(
        batch_size=1,
        dataset=dict(
            type='BaseSegDataset',
            data_root=data_root,
            ann_file='',
            data_prefix=dict(img_path='img_dir/val', seg_map_path='ann_dir/val'),
            pipeline=test_pipeline,
            metainfo=dict(classes=('background', 'foreground'), palette=[[120, 120, 120], [6, 230, 230]])
        )
    )
    
    test_dataloader = val_dataloader
    
    # --- 3. 加载我们下载的预训练模型 ---
    load_from = './checkpoints/deeplabv3plus_r50-d8_512x1024_40k_cityscapes_20200605_094610-d222ffcd.pth'
    
    # --- 4. 修改训练循环配置 ---
    train_cfg = dict(type='IterBasedTrainLoop', max_iters=200, val_interval=50)
    val_cfg = dict(type='ValLoop')
    test_cfg = dict(type='TestLoop')
    
    # --- 5. 修改优化器学习率 ---
    optim_wrapper = dict(optimizer=dict(lr=0.001))
    
    # --- 6. 修改默认钩子(Hook)配置 ---
    default_hooks = dict(
        logger=dict(type='LoggerHook', interval=10),
        checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=50)
    )
    ````

### 第四步：环境依赖的最终确认（关键修正1）

这是我们遇到的最核心的环境问题。如果你的训练因 `MMCV` 版本不兼容而失败，请执行以下命令来安装一个与 `mmsegmentation 1.2.2` 完美匹配的版本。

```bash
# 1. (如果需要) 卸载掉当前不兼容的版本
pip uninstall mmcv

# 2. 安装一个精确范围内的兼容版本
mim install "mmcv<2.2.0,>=2.0.0rc4"
```

### 第五步：开始训练！

万事俱备，只欠东风。执行下面的命令，你将看到训练成功的曙光。

```bash
python tools/train.py configs/deeplabv3plus/deeplabv3plus_r50-d8_dummy-test.py
```

**预期成功输出**:
你会看到一系列的日志，包括加载模型时关于 `size mismatch` 的**正常警告**（这是微调的标志），最后，训练日志会开始滚动：
```
...
07/29 17:30:01 - mmengine - INFO - Epoch(train) [1][10/200]  lr: 1.0000e-03 ... loss: 0.5123 ...
07/29 17:30:02 - mmengine - INFO - Epoch(train) [1][20/200]  lr: 1.0000e-03 ... loss: 0.4321 ...
...
```
