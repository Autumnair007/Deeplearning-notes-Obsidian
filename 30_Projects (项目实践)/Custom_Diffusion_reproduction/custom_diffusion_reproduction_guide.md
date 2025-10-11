---
type: tutorial
tags:
  - cv
  - diffusion-model
  - text-to-image
  - finetuning
  - cd
  - stable-diffusion
  - pytorch
  - code-note
status: done
related_models:
  - Stable Diffusion
  - CLIP
summary: Custom Diffusion（多概念定制T2I）模型的详细复现指南，包含单概念和联合微调的步骤、环境配置、代码修改和OOM等常见问题的解决方案。
---
本项目是 Custom Diffusion 的一个复现实现，详细记录了完整的复现步骤，并包含了大部分所需文件。复现过程主要在一张3060（12G）和一张4060（16G）显卡上进行单卡训练，用户可根据自身硬件情况调整脚本。考虑到网络限制，项目采用了本地下载文件后上传至训练服务器的方式，若网络通畅则可直接从Hugging Face及GitHub获取资源。

**重要文件准备：**

*   `sd-v1-4.ckpt` 需自行下载并放置于 `stable-diffusion/` 目录下。
*   `pytorch_model.bin` (Safety Checker) 需自行下载并放置于 `stable-diffusion/CompVis/stable-diffusion-safety-checker/` 目录下。
*   `pytorch_model.bin` (CLIP ViT-L/14) 需自行下载并放置于 `stable-diffusion/openai/clip-vit-large-patch14/` 目录下。
*   `checkpoint_liberty_with_aug.pth` 需根据指南内容自行放置。

其他指南中提及的下载内容大多已包含在项目中。如需从零开始，可遵循复现指南操作。

**项目结构说明：**

*   **`logs/`**: 存放单概念微调和多概念微调的相关文件，包括 checkpoints、生成的图像等。
*   **`data/`**: 存放官方提供的数据文件。
*   **`gen_reg/`**: 存放由脚本自动生成的正则化数据。
*   **`samples/`**: 存放微调后生成的图像。

**参考资料：**

*   Stable-diffusion复现笔记: [https://blog.csdn.net/qq_45791526/article/details/134757194?spm=1001.2014.3001.5506](https://blog.csdn.net/qq_45791526/article/details/134757194?spm=1001.2014.3001.5506)
*   论文网址 (Multi-Concept Customization of Text-to-Image Diffusion): [Multi-Concept Customization of Text-to-Image Diffusion](https://www.cs.cmu.edu/~custom-diffusion/)
*   复现资料：[Autumnair007/Custom-Diffusion-Replication: CustomDiffusion模型的复现项目](https://github.com/Autumnair007/Custom-Diffusion-Replication)

------
## 前期准备 

1.  #### **硬件要求**:
    
    *   **GPU**: 微调 Custom Diffusion 对显存有较高要求。
        *   官方提及在两块 A100 (每块约30GB显存) 上使用 batchsize=8 进行测试，可以根据自己的情况适当较小batchsize。
    
2.  #### **软件要求**:
    
    *   **Python**: 推荐版本 3.8。
    *   **Anaconda/Miniconda**: 用于创建和管理独立的 Python 环境。
    *   **CUDA 和 cuDNN**: 根据您的 NVIDIA GPU 型号和驱动程序，安装匹配的 CUDA Toolkit 和 cuDNN。这是 PyTorch 进行 GPU 加速的关键。
    *   **Git**: 用于克隆代码仓库。

## **项目搭建与环境配置**

*   #### **操作目录**:
    
    打开您的终端 (Terminal/Shell)。首先，使用 `cd` 命令进入您希望存放项目代码的父文件夹。例如：
    
    ```bash
    # mkdir ~/projects # 如果 projects 文件夹不存在则创建
    # cd ~/projects
    # 为方便演示，后续假设您的项目父文件夹为 ~/projects
    # 您可以替换为您的实际路径，例如 /data/coding/
    mkdir -p ~/projects 
    cd ~/projects
    ```

1.  #### **克隆 Custom Diffusion 仓库**:
    
    在 `~/projects/` 文件夹内执行：
    
    ```bash
    git clone https://github.com/adobe-research/custom-diffusion.git
    ```
    
2.  #### **进入 Custom Diffusion 目录**:
    
    ```bash
    cd custom-diffusion
    ```
    *   **当前所在文件夹**: `~/projects/custom-diffusion`
    
3.  #### **克隆 Stable Diffusion 仓库 (作为 Custom Diffusion 的依赖)**:
    
    在 `~/projects/custom-diffusion/` 文件夹内执行：
    
    ```bash
    git clone https://github.com/CompVis/stable-diffusion.git
    ```
    *   这会在 `custom-diffusion` 文件夹内创建一个名为 `stable-diffusion` 的子文件夹。
    *   **当前目录结构**:
        
        ```
        projects/
        └── custom-diffusion/
            ├── stable-diffusion/
            └── ... (custom-diffusion的其他文件)
        ```
    
4.  #### **进入 Stable Diffusion 目录以创建 Conda 环境**:
    
    ```bash
    cd stable-diffusion
    ```
    *   **当前所在文件夹**: `~/projects/custom-diffusion/stable-diffusion`
    
5.  #### **创建并激活 Conda 环境 (`ldm`)**:
    
    使用官方提供的 `environment.yaml` 文件创建环境。
    
    ```bash
    conda env create -f environment.yaml
    conda activate ldm
    ```
    
    *   **Conda 环境创建问题排查 (来自您的笔记)**:
        
        *   **`clip` 和 `taming-transformers` 下载或安装失败**:
            `environment.yaml` 中包含通过 git 直接安装这两个库的指令:
            ```yaml
            # - -e git+https://github.com/CompVis/taming-transformers.git@master#egg=taming-transformers
            # - -e git+https://github.com/openai/CLIP.git@main#egg=clip
            ```
            如果这些自动下载失败 (例如由于网络问题或git相关错误)，您可以手动安装：
            1.  **手动下载源码**:
                
                *   [CompVis/taming-transformers](https://github.com/CompVis/taming-transformers)
                *   [openai/CLIP](https://github.com/openai/CLIP)
            2.  **放置源码**:
                下载并解压后，在 `~/projects/custom-diffusion/stable-diffusion/` 目录下创建一个 `src` 文件夹 (如果尚不存在)。
                
                ```bash
                # 确保当前在 ~/projects/custom-diffusion/stable-diffusion/ 目录下
                mkdir -p src
                ```
                将解压后的 `taming-transformers` 和 `CLIP` 项目文件夹放入此 `src` 目录。
                **目录结构应为**:
                
                ```
                custom-diffusion/
                └── stable-diffusion/
                    ├── src/
                    │   ├── taming-transformers/  (包含 setup.py 等)
                    │   └── CLIP/                 (包含 setup.py 等)
                    ├── environment.yaml
                    └── ...
                ```
            3.  **以可编辑模式安装**:
                确保 `ldm` 环境已激活 (`conda activate ldm`)。然后从 `~/projects/custom-diffusion/stable-diffusion/` 目录执行 (注意路径)：
                
                ```bash
                pip install -e ./src/taming-transformers/
                pip install -e ./src/CLIP/
                ```
                如果从 `custom-diffusion` 根目录执行的命令，路径会相应调整为：
                ```bash
                # 假设当前在 ~/projects/custom-diffusion/ 目录
                # pip install -e ./stable-diffusion/src/taming-transformers/
                # pip install -e ./stable-diffusion/src/CLIP/
                ```
                选择与您当前操作目录匹配的命令。
            4.  **验证安装**:
                
                ```bash
                python -c "import taming; print('taming-transformers imported successfully')"
                python -c "import clip; print('CLIP imported successfully')"
                ```
                如果无报错，则表示 Python 可以找到这些本地包。
            
        *   **`conda env create` 失败的其他原因**:
            
            *   尝试解决 `environment.yaml` 中指出的具体包版本冲突，一定注意==**完全按照**==`environment.yaml` 里的要求安装环境。
            *   或尝试更新环境：`conda env update -f environment.yaml --prune`。
    
6.  #### **安装额外依赖 (在 `ldm` 环境中)**:
    
    确保 `ldm` Conda 环境已激活 (`conda activate ldm`)。
    整合后的 `pip` 安装列表如下 (部分使用了清华镜像源加速)：
    
    ```bash
    # 来自Custom Diffusion的依赖 
    pip install wandb clip-retrieval tqdm -i https://pypi.tuna.tsinghua.edu.cn/simple
    # 来自Stable Diffusion environment.yaml中的pip部分 (如果前面conda env create不完整，可手动补装)
    # 建议优先确保 conda env create -f environment.yaml 成功执行
    pip install -i https://pypi.tuna.tsinghua.edu.cn/simple albumentations==0.4.3
    pip install -i https://pypi.tuna.tsinghua.edu.cn/simple diffusers
    pip install -i https://pypi.tuna.tsinghua.edu.cn/simple opencv-python==4.1.2.30
    pip install -i https://pypi.tuna.tsinghua.edu.cn/simple pudb==2019.2
    pip install -i https://pypi.tuna.tsinghua.edu.cn/simple invisible-watermark
    pip install -i https://pypi.tuna.tsinghua.edu.cn/simple imageio==2.9.0
    pip install -i https://pypi.tuna.tsinghua.edu.cn/simple imageio-ffmpeg==0.4.2
    pip install -i https://pypi.tuna.tsinghua.edu.cn/simple pytorch-lightning==1.4.2
    pip install -i https://pypi.tuna.tsinghua.edu.cn/simple omegaconf==2.1.1
    pip install -i https://pypi.tuna.tsinghua.edu.cn/simple "test-tube>=0.7.5"
    pip install -i https://pypi.tuna.tsinghua.edu.cn/simple "streamlit>=0.73.1"
    pip install -i https://pypi.tuna.tsinghua.edu.cn/simple einops==0.3.0 
    pip install -i https://pypi.tuna.tsinghua.edu.cn/simple torch-fidelity==0.3.0
    pip install -i https://pypi.tuna.tsinghua.edu.cn/simple transformers==4.19.2
    pip install -i https://pypi.tuna.tsinghua.edu.cn/simple torchmetrics==0.6.0 
    pip install -i https://pypi.tuna.tsinghua.edu.cn/simple kornia==0.6 
    ```
    **注意**: `environment.yaml` 中通常已包含这些依赖。此步骤主要用于补充安装或确认特定版本。优先确保 `conda env create` 成功。
    
7.  #### **返回 Custom Diffusion 主目录**:
    
    ```bash
    cd ..
    ```
    *   **当前所在文件夹**: `~/projects/custom-diffusion` (之后的大部分操作将在此目录下进行)

## **下载所需文件**

1.  #### **下载 Stable Diffusion v1.4 预训练模型**:
    
    *   在 `~/projects/custom-diffusion/stable-diffusion` 目录下执行：
    ```bash
    wget https://huggingface.co/CompVis/stable-diffusion-v-1-4-original/resolve/main/sd-v1-4.ckpt
    ```
    *    `finetune_gen.sh` 示例使用的是 `stable-diffusion/sd-v1-4.ckpt`，这意味着它期望模型在 `custom-diffusion/stable-diffusion/` 目录下。请根据实际脚本的路径要求放置或修改脚本中的路径。
    
2.  #### **下载 Custom Diffusion 示例数据集**:
    
    *   在 `~/projects/custom-diffusion` 目录下执行：
    ```bash
    wget https://www.cs.cmu.edu/~custom-diffusion/assets/data.zip
    unzip data.zip
    ```
    *   这会在 `custom-diffusion` 目录下创建一个 `data` 文件夹，其中包含如 `cat`, `wooden_pot` 等子文件夹，每个子文件夹里有对应概念的训练图片。

3.  #### **手动下载clip-vit-large-patch14和stable-diffusion-safety-checker文件**:
    
    这部分内容用于解决某些自动下载失败或特定项目结构的需求。对于 Custom Diffusion，通常其依赖的 `stable-diffusion` 子项目会通过 `environment.yaml` 或脚本运行时自动处理 CLIP 等模型的下载和缓存。
    
    ==**如果 Custom Diffusion 脚本运行时没有提示缺失这些，则可能不需要此手动步骤。**==
    
    *   **`clip-vit-large-patch14` 文件**:
        
        *   从 [openai/clip-vit-large-patch14 at main (huggingface.co)](https://huggingface.co/openai/clip-vit-large-patch14/tree/main) 下载以下文件：
            *   `pytorch_model.bin`
            *   `config.json`
            *   `merges.txt`
            *   `preprocessor_config.json`
            *   `special_tokens_map.json`
            *   `tokenizer.json`
            *   `tokenizer_config.json`
            *   `vocab.json`
        *   下载完成后，在项目根目录 (例如 `~/projects/custom-diffusion/`) 创建文件夹 `openai/clip-vit-large-patch14`，并将下载的内容放入其中。
            ```bash
            # 在 ~/projects/custom-diffusion/ 目录下
            mkdir -p openai/clip-vit-large-patch14
            # mv <下载的文件> openai/clip-vit-large-patch14/
            ```
        
    *   **`stable-diffusion-safety-checker` 文件**:
        
        *   从 [CompVis/stable-diffusion-v1-4 at main (huggingface.co)](https://huggingface.co/CompVis/stable-diffusion-v1-4/tree/main) 下载：
            *   `pytorch_model.bin` (安全检查器模型)
            *   `config.json` (安全检查器配置)
        *   以及特征提取器的配置文件 (来自同一链接，具体文件名为 `preprocessor_config.json`，在 `feature_extractor` 子目录中)。
        *   下载完成后，在项目根目录 (例如 `~/projects/custom-diffusion/`) 创建文件夹 `CompVis/stable-diffusion-safety-checker`，并将这三个文件放入其中。
            ```bash
            # 在 ~/projects/custom-diffusion/ 目录下
            mkdir -p CompVis/stable-diffusion-safety-checker
            # mv <下载的三个文件> CompVis/stable-diffusion-safety-checker/
            ```
    *   **注意**: 这种手动放置文件的方式通常是为了确保在无网络或自动下载失败时，脚本能找到所需资源。请确认 Custom Diffusion 的脚本是否会查找这些特定路径下的文件。通常，Hugging Face Transformers 或 Diffusers 库会将模型缓存到 `~/.cache/huggingface/hub/`。

## **环境配置中常见问题与解决方案 **

1.  #### **模型/文件下载网络连接问题 (`ConnectionResetError`)**:
    
    *   **症状**: 脚本在运行时尝试从 URL (如 GitHub, Hugging Face) 下载某些依赖文件 (例如 `kornia` 依赖的 `checkpoint_liberty_with_aug.pth`) 时，出现 `ConnectionResetError: [Errno 104] Connection reset by peer`。
        
        错误示例: 
        
        ```bash
        Downloading:"https://github.com/DagnyT/hardnet/raw/master/pretrained/train_liberty_with_aug/checkpoint_liberty_with_aug.pth" to /root/.cache/torch/hub/checkpoints/checkpoint_liberty_with_aug.pth
        ```
        
    *   **解决方案：手动下载并放置文件**:
        
        1. **本地下载**: 在您的本地电脑 (如 Windows) 使用浏览器或下载工具下载目标 URL 的文件 (`checkpoint_liberty_with_aug.pth`)。https://github.com/DagnyT/hardnet/raw/master/pretrained/train_liberty_with_aug/checkpoint_liberty_with_aug.pth
        
        2.  **创建服务器目标目录**: 在云服务器上，确保脚本期望的缓存路径存在。
            
            ```bash
            # 示例路径，根据实际错误日志调整
            # 如果以 root 用户运行，路径可能是 /root/.cache/...
            # 如果以普通用户运行，路径可能是 ~/.cache/...
            mkdir -p /root/.cache/torch/hub/checkpoints/
            ```
            
        3. **上传文件**: 使用 `scp`、FileZilla、WinSCP 等工具将下载的文件上传到云服务器的对应目标目录中。
        
        4.  **重新运行脚本**: 文件上传后，再次运行之前失败的 Python 脚本。相关库应能检测到本地缓存的文件并跳过下载。
    
2.  #### **脚本执行 `\r: command not found` 或类似错误 (Windows 到 Linux 换行符问题)**:
    
    * **原因**: 当在 Windows 环境下编辑或克隆 Git 仓库（如果 Git 配置为自动将 LF 转换为 CRLF）后，再在 Linux 环境下运行这些脚本时，Windows 的换行符 (`\r\n` 或 CRLF) 中的回车符 (`\r`) 可能被 Linux shell 误解。
    
    *   **解决方案：转换文件的换行符**:
        假设出问题的脚本是 `scripts/finetune_gen.sh` (位于 `custom-diffusion` 目录下)。
        
        *   **使用 `dos2unix` 工具**:
            
            1.  安装 `dos2unix` (如果系统未安装):
                ```bash
                # Debian/Ubuntu 系统
                sudo apt-get update
                sudo apt-get install dos2unix
                ```
            2.  转换脚本文件:
                ```bash
                # 在 custom-diffusion 目录下执行
                dos2unix scripts/finetune_gen.sh
                ```
            
        * 转换完成后，重新运行您的脚本命令。

---

## Custom Diffusion 单概念微调与图像生成步骤

本指南将引导您完成使用 Custom Diffusion 进行单概念微调，并使用微调后的模型生成图像的完整流程。

**核心流程概览：**

1.  **环境与代码准备**：确保 Conda 环境激活，并对项目代码进行必要的修改。
2.  **数据准备**：准备概念图像和正则化图像。
3.  **配置文件调整**：修改 YAML 配置文件以适应您的硬件和需求。
4.  **执行微调**：运行微调脚本。
5.  **提取 Delta 权重**：从微调后的检查点中提取增量权重。
6.  **使用微调模型生成图像**：利用原始模型、Delta 权重和特殊 Token 生成新图像。

---

### **一、环境与代码准备**

1.  #### **激活 Conda 环境**：
    
    确保已激活包含所有必要依赖（PyTorch, Pillow, etc.）的 Conda 环境。例如：
    
    ```bash
    conda activate ldm
    ```
    
2.  #### **项目代码修改**：
    
    *   **初始脚本运行失败，主要错误包括：**`OSError: Can't load tokenizer for 'openai/clip-vit-large-patch14'`** (在 `sample.py` 和 `train.py` 中均出现)
        
        **问题描述**：Hugging Face `transformers` 库无法从网络下载或在本地找到 `openai/clip-vit-large-patch14` 模型对应的分词器 (tokenizer) 和模型文件。
        
        **解决方案 (直接修改Python源代码 - 成功)**：
        
        1. 我们将模型文件下载到本地。参考[手动下载clip-vit-large-patch14和stable-diffusion-safety-checker文件](#手动下载clip-vit-large-patch14和stable-diffusion-safety-checker文件)。
        2. 确认问题根源：脚本中的 `CLIPTokenizer.from_pretrained()` 和 `CLIPTextModel.from_pretrained()` 仍然尝试从网络ID `"openai/clip-vit-large-patch14"` 加载。
        3. **定位修改点**：
           *   当前在文件夹`custom-diffusion`下面。
           *   对于 `sample.py` 相关的加载（通常在 `stable-diffusion/ldm/modules/encoders/modules.py` 中的 `CLIPEmbedder` 类）。
           *   对于 `train.py` 相关的加载（通常在 `src/custom_modules.py` 中的 `FrozenCLIPEmbedder` 类）。
        4. **具体操作**：直接编辑上述 Python 文件，将所有 `CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")` 或 `CLIPTokenizer.from_pretrained(version)` 以及 `CLIPTextModel.from_pretrained(...)` 中的模型名称/路径参数，**硬编码为你本地存放 `openai/clip-vit-large-patch14` 完整模型文件（包括分词器文件、`config.json`、`pytorch_model.bin`等）的绝对路径**。例如：
        
        ```python
        # 参考代码示例
        CLIPTokenizer.from_pretrained("/path/to/your/local/clip-vit-large-patch14/")
        # 具体路径
        CLIPTokenizer.from_pretrained("/data/coding/upload-data/data/custom-diffusion/stable-diffusion/openai/clip-vit-large-patch14/")
        ```
        
    *   **`RuntimeError: CUDA out of memory.`** (在 `sample.py` 运行时)
        
        *   **问题描述**：GPU 显存不足。日志显示尝试分配大量显存（如 `Tried to allocate 5.00 GiB`），而你的 GPU 总共约 11.76 GiB。
        *   **原因分析**：`sample.py` 脚本（用于生成正则化图片）一次性生成的图片数量（批处理大小，`--n_samples`）对于当前显存来说过大。日志中 `Data shape for DDIM sampling is (10, ...)` 暗示了这一点。
        *   **解决方案**：编辑 `scripts/finetune_gen.sh` 文件，找到调用 `python sample.py ...` 的命令行，将其中的 `--n_samples` 参数值改小（例如从 `10` 改为 `1` 或 `2`）。
        
    *   **`pytorch_lightning.utilities.exceptions.MisconfigurationException: You requested GPUs: [0, 1] But your machine only has: [0]`** (在 `train.py` 运行时)
        
        *   **问题描述**：PyTorch Lightning 检测到脚本请求使用两个GPU（GPU 0 和 GPU 1），但系统实际上只有一个GPU（GPU 0）。
        *   **原因分析**：`scripts/finetune_gen.sh` 文件中调用 `python train.py ...` 时，传递的 `--gpus` 参数被设置为了类似 `"0,1"`。
        *   **解决方案**：编辑 `scripts/finetune_gen.sh` 文件，修改传递给 `train.py` 的 `--gpus` 参数，使其仅指定使用一个GPU（例如改为 `--gpus 1`）。
        
    *   **修改 `scripts/finetune_gen.sh` (使用自定义正则化图像)**：
        如果您希望==**使用自己准备的正则化图像**==，而不是让脚本自动生成，请注释掉调用 `sample.py` 生成正则化图像的部分。
        
        如果想让脚本自动生成就不需要注释（默认情况）。
        打开 `scripts/finetune_gen.sh`，找到类似以下代码块并注释掉：
        
        ```bash
        # python -u sample.py \
        #        --n_samples 1 \
        #        --n_iter 200 \
        #        --scale 6 \
        #        --ddim_steps 50  \
        #        --ckpt ${ARRAY[5]} \
        #        --ddim_eta 1. \
        #        --outdir "${ARRAY[2]}" \
        #        --prompt "photo of a ${ARRAY[0]}"
        ```
        
    *   **修改 `src/finetune_data.py` (解决 Pillow `Image.LINEAR` 错误)**：
        较新版本的 Pillow 库可能不包含 `Image.LINEAR`。将其替换为 `Image.Resampling.BILINEAR`。
        打开 `src/finetune_data.py`，找到约第 159 行，修改如下：
        
        ```python
        # from PIL import Image # 确保 Image 被导入
        # ...
        # self.interpolation = {"linear": Image.LINEAR, # 旧代码
        self.interpolation = {"linear": Image.Resampling.BILINEAR, # 修改后的代码
        # ...
        ```
        
    *   **修改 `train.py` (解决 `AttributeError: 'int' object has no attribute 'strip'` for `gpus` 参数)**：
        使 `train.py` 能够正确处理不同类型的 `gpus` 参数。打开 `train.py`，找到处理 `lightning_config.trainer.gpus` 的部分 (约第 926 行附近)，替换为以下健壮的代码：
        
    *   ```bash
        if not cpu:
            gpus_config = lightning_config.trainer.gpus
            if isinstance(gpus_config, str):
                items = [item for item in gpus_config.split(',') if item.strip()]
                ngpu = len(items)
            elif isinstance(gpus_config, int):
                if gpus_config >= 0:
                    ngpu = gpus_config
                else: # gpus_config == -1 (all GPUs)
                    # Simplified: assume at least 1 if not cpu and -1 is specified.
                    # Accurate count would require torch.cuda.device_count()
                    ngpu = 1
            elif isinstance(gpus_config, list):
                ngpu = len(gpus_config)
            else: # None or other unexpected type
                ngpu = 0
        else: # if cpu:
            ngpu = 1 # Original logic for CPU
        ```
        
    *   **主要改动说明：**
        
        1.  **类型检查**：在尝试对 `gpus_config` 使用字符串方法（如 `.strip()` 或 `.split()`）之前，代码现在会检查其类型。
        2.  **处理字符串**：如果 `gpus_config` 是字符串，它会通过列表推导式 `[item for item in gpus_config.split(',') if item.strip()]` 来分割并计算有效的 GPU ID 数量，这能更好地处理各种格式（如 `"0,1"`, `",0,"`, `""`）。
        3.  **处理整数**：
            *   如果 `gpus_config` 是一个非负整数（例如 `0`, `1`, `2`），`ngpu` 会直接设为该值。
            *   如果 `gpus_config` 是 `-1`（通常表示使用所有可用的 GPU），`ngpu` 暂时设为 `1`。这是一个占位符，因为准确的数量需要通过 `torch.cuda.device_count()` 获取，而此代码片段不直接调用它。鉴于 `if not cpu` 为真，假定至少有1个GPU是合理的简化。
        4.  **处理列表**：如果 `gpus_config` 是一个列表（例如 `[0, 1]`），`ngpu` 会设为列表的长度。
        5.  **默认情况**：如果 `gpus_config` 是 `None` 或其他非预期的类型，`ngpu` 会默认为 `0`。
        6.  **CPU 情况**：当 `cpu` 为 `True` 时，`ngpu = 1` 的逻辑保持不变，因为这是您原始代码的一部分。
        
        这个修改后的片段应该能更稳健地处理不同类型的 `gpus` 配置。

### **二、数据准备**

1.  #### **概念图像 (Concept Images)**：
    
    *   准备一组代表要微调的新概念的图像（例如，特定风格的猫的图片）。
    *   将这些图像放置在一个文件夹中。
    *   **示例路径**：`data/cat/`
    
2.  #### **正则化图像 (Regularization Images)**：
    
    *   这些图像用于防止模型遗忘原始概念。
    *   如果选择使用自定义的正则化图像（已修改 `finetune_gen.sh`）：
        *   准备与概念类别相同但风格泛化的图像（例如，普通的猫的照片）。
        *   将这些图像放置在 `finetune_gen.sh` 脚本参数 `${ARRAY[2]}/samples/` 所指向的目录。
        *   **示例路径**：`gen_reg/samples_cat/samples/` (注意，脚本中 `${ARRAY[2]}` 是 `gen_reg/samples_cat`，然后 `train.py` 会查找其下的 `samples` 子目录)。

### **三、配置文件调整 (`.yaml`)**

微调的行为由一个 YAML 配置文件控制。您使用的是 `finetune_addtoken.yaml`。

*   **文件路径**：`configs/custom-diffusion/finetune_addtoken.yaml` (或您指定的其他配置文件)
*   **关键参数调整 (防止显存不足 `CUDA out of memory`)**：

    *   **`batch_size`**: 减小批处理大小。
        
        ```yaml
        data:
          params:
            batch_size: 1 # 从 4 调整为 1 
        ```
    *   **`image_size`**: 减小训练图像的尺寸。
        ```yaml
        data:
          params:
            train:
              params:
                size: 256 # 从 512 调整为 256 或其他较小值
            validation:
              params:
                size: 256 # 从 512 调整为 256 或其他较小值
        ```
    *   **`precision`**: 启用混合精度训练。
        
        ```yaml
        lightning:
          trainer:
            precision: 16 # 添加此行，使用16位混合精度
        ```
*   **`modifier_token`**: 确认或设置您的特殊 Token。
    
    ```yaml
    model:
      params:
        modifier_token: "<new1>" # 这是您的特殊 Token
    ```
    这个 Token 将在后续生成图像时用于调用您的新概念。

### **四、执行微调**

1.  #### **切换到项目根目录**（参考命令）：
    
    ```bash
    # 根据实际的存放路径返回到custom-diffusion文件夹中
    cd /data/coding/upload-data/data/custom-diffusion/
    ```
    
2.  #### **运行微调脚本**：
    
    使用 `scripts/finetune_gen.sh` 脚本，并根据您的实际路径和参数进行调整。
    
    *   **命令结构**：
        
        ```bash
        bash scripts/finetune_gen.sh "concept_name_or_caption" /path/to/concept_images /path/to/reg_samples_parent_dir output_dir_name config_file.yaml /path/to/base_sd_model.ckpt
        ```
    *   **示例命令**：
        
        ```bash
        bash scripts/finetune_gen.sh "cat" data/cat gen_reg/samples_cat cat finetune_addtoken.yaml stable-diffusion/sd-v1-4.ckpt
        ```
        *   `"cat"`: 概念的描述或类别。
        *   `data/cat`: 您的概念图像路径。
        *   `gen_reg/samples_cat`: 正则化图像的父目录 (脚本内部会查找 `gen_reg/samples_cat/samples/`)。
        *   `cat`: 实验输出目录名称的一部分。
        *   `finetune_addtoken.yaml`: 使用的配置文件名 (位于 `configs/custom-diffusion/` 下)。
        *   `stable-diffusion/sd-v1-4.ckpt`: 基础 Stable Diffusion 模型路径。
    
    训练完成后，日志和检查点将保存在类似 `logs/YYYY-MM-DDTHH-MM-SS_output_dir_name-sdv4/` 的目录中。
    **示例日志文件夹**：`logs/2025-05-10T15-52-41_cat-sdv4/`

### **五、提取 Delta 权重**

Delta 权重代表模型在微调过程中学习到的变化。图像生成脚本通常需要这个 Delta 文件和原始预训练模型。

1.  #### **命令结构**：
    
    ```bash
    python src/get_deltas.py --path <path_to_your_log_folder_containing_checkpoints> --newtoken 1
    ```
    `--newtoken 1` 通常表示您在微调时使用了1个新的 modifier token。
    
2.  #### **示例命令** (假设当前位于项目根目录)：
    
    ```bash
    python src/get_deltas.py --path logs/2025-05-10T15-52-41_cat-sdv4 --newtoken 1
    ```
    
3.  #### **输出**：
    
    执行后，Delta 权重文件（例如 `delta_epoch=last.ckpt`）会保存在日志文件夹的 `checkpoints` 子目录中。
    **示例 Delta 权重文件路径**：`logs/2025-05-10T15-52-41_cat-sdv4/checkpoints/delta_epoch=last.ckpt`

### **六、使用微调模型生成图像**

现在，结合原始预训练模型、提取的 Delta 权重和特殊 Token 来生成图像。

1.  #### **确认所需信息**（参考）：
    
    *   **Delta 权重文件路径**：`logs/2025-05-10T15-52-41_cat-sdv4/checkpoints/delta_epoch=last.ckpt`
    *   **原始 SD 模型路径**：`/data/coding/upload-data/data/custom-diffusion/stable-diffusion/sd-v1-4.ckpt`
    *   **您的特殊 Token**：`<new1>` (来自 YAML 文件，默认情况是这样)
    
2. #### **运行图像生成脚本 (`sample.py`)**：

   *   **命令结构**：
       
       ```bash
       python sample.py \
           --prompt "YOUR_SPECIAL_TOKEN description of the image" \
           --delta_ckpt <path_to_your_delta_weights_file.ckpt> \
           --ckpt <path_to_your_original_sd_model.ckpt> \
           [--n_samples N] \
           [--scale S] \
           [--ddim_steps D] \
           [other_optional_parameters]
       ```
   *   **完整示例命令** (假设当前位于custom-diffusion目录)：
       
       ```bash
       python sample.py \
           --prompt "<new1> cat wearing a tiny wizard hat, fantasy art style" \
           --delta_ckpt logs/2025-05-10T15-52-41_cat-sdv4/checkpoints/delta_epoch=last.ckpt \
           --ckpt /data/coding/upload-data/data/custom-diffusion/stable-diffusion/sd-v1-4.ckpt \
           --n_samples 2 \
           --scale 7.5 \
           --ddim_steps 50
       ```

3.  #### **可调整参数**：
    
    *   `--prompt`: **必须包含您的特殊 Token (`<new1>`)**，后跟图像描述。
    *   `--n_samples`: 生成图片的数量。
    *   `--scale`: CFG scale，影响提示遵循程度。
    *   `--ddim_steps`: 采样步数。
    
4.  #### **输出**：
    
    生成的图像通常保存在项目根目录下的 `outputs/txt2img-samples/` 文件夹中，在一个以提示或时间戳命名的子文件夹内。

---

**重要提示：**

*   **路径准确性**：务必确保所有命令中的文件路径和名称完全正确。
*   **Token 使用**：在生成图像的提示中正确使用您在微调时定义的 `modifier_token` 是关键。
*   **逐步调试**：如果遇到问题，仔细检查每一步的输出和错误信息。

------

## **Custom Diffusion 多概念联合微调详细步骤 **

### **一、 环境与配置细节**

在开始操作前，请确保您手头有以下关键信息、文件和已完成的准备工作：

1.  #### **项目与环境**：
    
    *   **项目根目录（参考）**：`/data/coding/upload-data/data/custom-diffusion/` (所有命令均在此目录下执行)。
    *   **Conda 环境**：`ldm`。激活命令：`conda activate ldm`。
    *   **操作系统/Shell**:  Linux 环境。

2.  #### **概念相关文件与信息**：
    
    *   **第一个概念 (例如 "cat")**：
        *   **训练图像路径**：`data/cat/`
        *   **正则化图像路径** (假设已在单概念微调步骤中生成)：`gen_reg/samples_cat/samples/`
        *   **指定 Token** (联合微调脚本中)：`<new1>`
    *   **第二个概念 (例如 "wooden\_pot")**：
        *   **训练图像路径**：`data/wooden_pot/`
        *   **指定 Token** (联合微调脚本中)：`<new2>`
        *   **正则化图像路径** (将在步骤1中创建)：目标为 `gen_reg/samples_wooden_pot/samples/`
    
3.  #### **预训练模型路径（参考）**：
    
    *   原始 Stable Diffusion `.ckpt` 文件路径：`/data/coding/upload-data/data/custom-diffusion/stable-diffusion/sd-v1-4.ckpt` (后续简称 `<pretrained-model-path>`)
    
4.  #### **配置文件与脚本**：
    
    *   **联合微调配置文件 (.yaml)**：`configs/custom-diffusion/finetune_joint.yaml`。
        *   思考过程 (Batch Size 调整)：您在后续步骤中因显存不足和 `batch_size=0` 错误，对此文件中的 `data.params.batch_size` 进行了调整，最终确定为 `2`。
        *   关键内容:
            ```yaml
            # finetune_joint.yaml (部分内容)
            model:
              # ... (model params)
            data:
              target: train.DataModuleFromConfig
              params:
                batch_size: 4 # <--- 初始值，后调整为 2
                num_workers: 4
                # ... (train, train2 dataloader configs)
            lightning:
              # ... (callbacks)
              trainer:
                max_steps: 550
            ```
    *   **联合微调脚本 (.sh)**：您将使用一个定制的 `scripts/finetune_joint_gen.sh` 脚本。
    
5.  #### **GPU 配置**：
    
    *   **可用 GPU**: 从日志判断，您至少有一块 GPU (GPU 0)，总显存约 `15.70 GiB`。
    *   **脚本配置**: `finetune_joint_gen.sh` 脚本中需正确设置 `--gpus` 参数。
        *   思考过程 (GPU 数量)：为解决显存问题，建议在联合训练时使用单 GPU (例如 `--gpus 1`)。

### **二、 详细操作步骤与问题排查历程**

#### **步骤 1：为第二个概念生成正则化图像 (例如 "wooden\_pot")**

* **目的**：为第二个概念（"wooden\_pot"）创建通用图像。

*   **初始尝试与问题 **：
    
    * **参考命令结构** ：
    
    * ```bash
      python -u sample.py \
          --n_samples <reduced_num_samples_per_iteration> \
          --n_iter <increased_num_iterations> \
          --scale <cfg_scale> \
          --ddim_steps <sampling_steps>  \
          --ckpt <pretrained-model-path> \
          --ddim_eta 1.0 \
          --outdir "gen_reg/<name_of_second_concept_regex_folder>" \
          --prompt "photo of a <generic_class_name_of_second_concept>"
      ```
    
    * **具体命令**:
    
      ```bash
      python -u sample.py --n_samples 5 --n_iter 40 --scale 6 --ddim_steps 50 --ckpt <pretrained-model-path> --ddim_eta 1.0 --outdir "gen_reg/samples_wooden_pot" --prompt "photo of a wooden_pot"
      ```
    
    * **特定错误信息**: `RuntimeError: CUDA out of memory. Tried to allocate 2.50 GiB (GPU 0; 15.70 GiB total capacity; ...)`
    
    * **思考过程**：错误明确指向显存不足。`--n_samples 5` 尝试并行生成5张图，对显存消耗较大。
    
*   **解决方案与采纳的命令**：减少单次并行样本数 (`--n_samples`)，增加迭代次数 (`--n_iter`)，保持总样本数（目标200张）。
    
    *   **命令 (用于 "wooden\_pot"，在项目根目录执行)**：
        
        ```bash
        # 推荐命令 (n_samples=2, n_iter=100 => 200张)
        python -u sample.py \
            --n_samples 2 \
            --n_iter 100 \
            --scale 6 \
            --ddim_steps 50  \
            --ckpt /data/coding/upload-data/data/custom-diffusion/stable-diffusion/sd-v1-4.ckpt \
            --ddim_eta 1.0 \
            --outdir "gen_reg/samples_wooden_pot" \
            --prompt "photo of a wooden_pot"
        ```
        (备选：如果 `--n_samples 2` 仍OOM, 则 `--n_samples 1 --n_iter 200`)
    
* **预期结果**：图像生成在 `gen_reg/samples_wooden_pot/samples/` 目录下。

#### **步骤 2：创建并配置联合微调脚本 (`scripts/finetune_joint_gen.sh`)**

*   确保您的 `finetune_joint_gen.sh` 脚本配置正确，可以接收多个概念的参数并启动联合训练。该定制脚本对于处理双概念、生成图正则化及特定 token (`<new1>`, `<new2>`) 至关重要。

* **操作**：

  1. **创建文件**：

     ```bash
     nano scripts/finetune_joint_gen.sh
     ```

  2.  **粘贴内容**：
      
      ```bash name=scripts/finetune_joint_gen.sh
      #!/usr/bin/env bash
      #### command to run with generated images as regularization for two concepts
      # 1st arg: target caption1 (e.g., "wooden pot")
      # 2nd arg: path to target images1 (e.g., data/wooden_pot)
      # 3rd arg: path where generated regex images1 are saved (e.g., gen_reg/samples_wooden_pot)
      # 4rth arg: target caption2 (e.g., "cat")
      # 5th arg: path to target images2 (e.g., data/cat)
      # 6th arg: path where generated regex images2 are saved (e.g., gen_reg/samples_cat)
      # 7th arg: name of the experiment (e.g., wooden_pot+cat)
      # 8th arg: config name (e.g., finetune_joint.yaml)
      # 9th arg: pretrained model path
      
      ARRAY=()
      
      for i in "$@"
      do
          echo $i
          ARRAY+=("${i}")
      done
      
      # 注意：确保你的 GPU ID (如 --gpus 0,1) 是正确的
      # 注意：你的笔记中 --gpus 6,7，这里我用 0,1 作为示例，请根据你的环境修改
      # ****** 请务必根据您的实际可用GPU修改下面的 --gpus 参数 ******
      python -u train.py \
              --base configs/custom-diffusion/${ARRAY[7]}  \
              -t --gpus 1 \
              --resume-from-checkpoint-custom ${ARRAY[8]} \
              --caption "<new1> ${ARRAY[0]}" \
              --datapath ${ARRAY[1]} \
              --reg_datapath "${ARRAY[2]}/samples" \
              --reg_caption "${ARRAY[0]}" \
              --caption2 "<new2> ${ARRAY[3]}" \
              --datapath2 ${ARRAY[4]} \
              --reg_datapath2 "${ARRAY[5]}/samples" \
              --reg_caption2 "${ARRAY[3]}" \
              --modifier_token "<new1>+<new2>" \
              --name "${ARRAY[6]}-sdv4"
      ```
      *   **关键配置细节**：
          
          *   `--gpus 1`: 明确指定使用单GPU，以配合后续 `batch_size` 调整解决OOM。
          *   `--no-test`: 此前您遇到过与 `trainer.test()` 或 `outputs` 目录相关的 `CUDACallback` 或 `MisconfigurationException`，添加此参数是为避免这些问题。
          *   脚本通过 `${ARRAY[index]}` 接收命令行参数，并将正则化路径硬编码为添加 `/samples` 后缀。
          
      * **换行符问题 (潜在)**：如果从Windows环境复制脚本，可能存在 `\r\n` 换行符问题，导致脚本无法正确执行。使用下面命令可解决。
      
        ```bash
        dos2unix scripts/finetune_joint_gen.sh
        ```
      
  3. **保存并退出**。

  4.  **添加执行权限**：`chmod +x scripts/finetune_joint_gen.sh`

#### **步骤 3：运行联合微调脚本与问题排查**

* **目的**：启动对 "wooden pot" 和 "cat" 两个概念的联合微调。

* **参考命令结构** 

* ```bash
  bash scripts/finetune_joint_gen.sh \
      "<generic_class_name_of_concept1>" \
      <path_to_training_images_concept1> \
      <path_to_regex_images_folder_concept1_NO_samples_suffix> \
      "<generic_class_name_of_concept2>" \
      <path_to_training_images_concept2> \
      <path_to_regex_images_folder_concept2_NO_samples_suffix> \
      "<experiment_name_concept1+concept2>" \
      <config_filename.yaml> \
      <pretrained-model-path>
  ```

*   **命令 (在项目根目录执行)**：
    
    ```bash
    bash scripts/finetune_joint_gen.sh \
        "wooden pot" \
        data/wooden_pot \
        gen_reg/samples_wooden_pot \
        "cat" \
        data/cat \
        gen_reg/samples_cat \
        "wooden_pot+cat" \
        finetune_joint.yaml \
        /data/coding/upload-data/data/custom-diffusion/stable-diffusion/sd-v1-4.ckpt
    ```
    
*   **问题1: `CUDA out of memory` **
    
    *   **特定错误信息**:
        
        ```
        RuntimeError: CUDA out of memory. Tried to allocate 1024.00 MiB (GPU 0; 15.70 GiB total capacity; 12.59 GiB already allocated; 826.94 MiB free; 13.26 GiB reserved in total by PyTorch)
        ```
    *   **思考过程与解决方案**：
        
        1.  检查 `finetune_joint.yaml` 中的 `data.params.batch_size`。如果大于1，尝试降为 `1`。
        2.  确认 `finetune_joint_gen.sh` 中 `--gpus` 设置为单GPU (如 `--gpus 1`)。
        3.  若上述仍OOM，考虑在 YAML 的 `lightning.trainer` 中启用梯度累积 (`accumulate_grad_batches: 2` 或 `4`)。
    *   **您的操作**：您将 `finetune_joint.yaml` 中的 `data.params.batch_size` 修改为了 `1`。
    
*   **问题2: `ValueError: batch_size should be a positive integer value, but got batch_size=0`**
    
    *   **特定错误信息 (Traceback)**:
        
        ```
        Traceback (most recent call last):
          File "train.py", line 988, in <module>
            trainer.fit(model, data)
          # ... (中间调用栈) ...
          File "/data/coding/upload-data/data/custom-diffusion/train.py", line 402, in _train_dataloader
            return DataLoader(concat_dataset, batch_size=self.batch_size // 2, # <--- 问题根源
          # ...
          File "/data/miniconda/envs/ldm/lib/python3.8/site-packages/torch/utils/data/sampler.py", line 215, in __init__
            raise ValueError("batch_size should be a positive integer value, "
        ValueError: batch_size should be a positive integer value, but got batch_size=0
        ```
    *   **思考过程与解决方案**：
        
        *   YAML 中 `data.params.batch_size` 为 `1`。
        *   `train.py` (line 402) 中实际传递给 `DataLoader` 的是 `self.batch_size // 2`。
        *   `1 // 2` (整数除法) 结果为 `0`，导致错误。
        *   **解决方案**：修改 `finetune_joint.yaml`，将 `data.params.batch_size` 从 `1` 改为 `2`。这样 `2 // 2 = 1`，为有效值。同时保持单GPU训练。
    
*   **预期结果**：训练完成，最终检查点保存在 `logs/2025-05-10T17-31-09_wooden_pot+cat-sdv4/checkpoints/last.ckpt`。您已确认此文件存在。

#### **步骤 4：为多概念模型提取 Delta 权重**

* **目的**：从 `last.ckpt` 中提取代表两个新概念变化的 Delta 权重。

* **参考命令结构**：

* ```bash
  python src/get_deltas.py --path logs/<folder-name_from_joint_training> --newtoken <number_of_new_concepts>
  ```

*   **命令 (在项目根目录执行)**：
    
    ```bash
    python src/get_deltas.py \
        --path /data/coding/upload-data/data/custom-diffusion/logs/2025-05-10T17-31-09_wooden_pot+cat-sdv4 \
        --newtoken 2 \
        --ckpt_name last.ckpt
    ```
    
*   **预期结果**：在 `--path` 指定目录的 `checkpoints/` 子目录下 (或直接在 `--path` 目录，取决于脚本实现)，生成 Delta 权重文件 (例如 `delta_last.ckpt` 或 `delta_wooden_pot+cat-sdv4.bin`)。

#### **步骤 5：使用多概念微调模型生成图像 (推理/采样)**

* **目的**：使用学习的 Delta 权重和原始模型，生成复合概念图像。

* **参考命令结构**：

* ```bash
  python sample.py \
      --prompt "description using <token_for_concept1> and <token_for_concept2>" \
      --delta_ckpt logs/<folder-name_from_joint_training>/checkpoints/<delta_weights_filename.ckpt> \
      --ckpt <pretrained-model-path> \
      [other_optional_parameters]
  ```

*   **命令 (在项目根目录执行，替换 `<delta_weights_filename>` 为实际生成的文件名和路径)**：
    
    ```bash
    # 假设 delta 文件名为 delta_last.ckpt 位于 checkpoints/ 子目录
    python sample.py \
        --prompt "the <new2> cat sculpture in the style of a <new1> wooden pot" \
        --delta_ckpt logs/2025-05-10T17-31-09_wooden_pot+cat-sdv4/checkpoints/delta_epoch=last.ckpt \
        --ckpt /data/coding/upload-data/data/custom-diffusion/stable-diffusion/sd-v1-4.ckpt \
        --n_samples 2 \
        --scale 7.5 \
        --ddim_steps 50
    ```
    
*   **预期结果**：生成包含 "wooden pot" 和 "cat" 特征的图像，保存在 `outputs/txt2img-samples/` 目录下。
