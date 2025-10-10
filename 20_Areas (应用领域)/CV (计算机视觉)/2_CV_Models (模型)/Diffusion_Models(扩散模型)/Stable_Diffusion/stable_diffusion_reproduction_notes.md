---
type: "project-summary"
tags: [cv, nlp, text-to-image, generative-ai, stable-diffusion, setup, bash, python]
status: "done"
related_models: [Stable Diffusion]
summary: "Stable Diffusion模型（LDM）的复现环境搭建与运行脚本指南，包括conda环境创建、依赖安装、模型权重下载及常见问题解决方案。"
---
参考资料：[Stable-diffusion复现笔记_stable diffusion复现-CSDN博客](https://blog.csdn.net/qq_45791526/article/details/134757194?spm=1001.2014.3001.5506)

------
**A. 前期准备 (在开始之前)**

1.  **软件环境**：
    *   **Python**：推荐版本 3.8, 3.9, 或 3.10。
    *   **Anaconda/Miniconda**：用于创建和管理独立的 Python 环境。
    *   **CUDA 和 cuDNN**：根据您的 NVIDIA GPU 型号和驱动程序，安装匹配的 CUDA Toolkit 和 cuDNN。这是 PyTorch 进行 GPU 加速的关键。
    *   **Git**：用于克隆代码仓库。

---

**B. 环境搭建 (Project Setup)**

*   **操作目录**：
    打开您的终端 (Terminal/Shell)。首先，使用 `cd` 命令进入您希望存放项目代码的父文件夹。例如：
    
    ```bash
    # mkdir ~/projects # 如果projects文件夹不存在
    mkdir projects 
    # cd ~/projects
    cd projects
    ```

1.  **克隆 Stable Diffusion 仓库**：
    在 `~/projects/` 文件夹内执行：
    
    ```bash
    git clone https://github.com/CompVis/stable-diffusion.git
    ```
    *   这会在 `~/projects/` 文件夹内创建一个名为 `stable-diffusion` 的子文件夹。
    *   **当前所在文件夹**：`~/projects/stable-diffusion`
    
2.  **进入 Stable Diffusion 目录并创建 Conda 环境**：
    
    ```bash
    cd stable-diffusion
    ```
    *   **当前所在文件夹**：`~/projects/stable-diffusion`
    
3.  **创建并激活 Conda 环境 (`ldm`)**：
    使用官方提供的 `environment.yaml` 文件创建环境。
    
    ```bash
    conda env create -f environment.yaml
    conda activate ldm
    ```
    
    如果这里关于
    
    ```bash
        - -e git+https://github.com/CompVis/taming-transformers.git@master#egg=taming-transformers
        - -e git+https://github.com/openai/CLIP.git@main#egg=clip
    ```
    
    的git下载有问题的话，删除上面代码并手动安装clip和taming-transformers两个库，手动安装步骤在下面。
    
4. **处理 Conda 环境创建可能遇到的问题**：

   *   **软件包冲突或下载失败 (特别是 `clip` 和 `taming-transformers`)**：
       
       *   **解决方法**：如果自动下载失败，可以从 GitHub 手动下载这两个包：
           
           *   [CompVis/taming-transformers](https://github.com/CompVis/taming-transformers)
           *   [openai/CLIP](https://github.com/openai/CLIP)
       *   下载后解压。在 `~/projects/stable-diffusion/` 项目中创建一个 `src` 文件夹 (如果不存在)。
           ```bash
           mkdir -p src # 在 stable-diffusion 目录下创建 src
           ```
       *   将解压后的 `taming-transformers` 和 `CLIP` 项目文件夹放入 `src` 中。
           *   **目录结构应为**：
               
               ```
               stable-diffusion/
               ├── src/
               │   ├── taming-transformers/  (包含 setup.py 等)
               │   └── CLIP/                 (包含 setup.py 等)
               └── ...
               ```
       *   然后从 `~/projects/stable-diffusion/` 目录（确保 `ldm` 环境已激活）使用 pip 以可编辑模式安装它们：
           ```bash
           pip install -e ./src/taming-transformers/
           pip install -e ./src/CLIP/
           ```
           *   **验证安装**：
               
               ```bash
               python -c "import taming; print('taming-transformers imported successfully')"
               python -c "import clip; print('CLIP imported successfully')"
               ```
               如果无报错，则表示 Python 可以找到这些本地包。
       
   *   **PyTorch Lightning 版本问题**：
       
       *   错误信息如：`CUDACallback.on_train_epoch_end() missing 1 required positional argument: 'outputs'`。
       *   这通常是 `pytorch-lightning` 版本不兼容导致的。`environment.yaml` 文件会指定版本。
       *   **解决方法1**：确保 `pytorch-lightning` 版本正确安装（见下方额外依赖）。
       *   **解决方法2**：如果问题依旧，可能需要调整 `train.py` 中的代码（具体调整未在笔记中详述，但通常是更新回调函数签名以匹配新版本API）。
       
   *   **`conda env create` 失败**：
       
       *   尝试解决 `environment.yaml` 中指出的具体包版本问题。
       *   或尝试更新环境：`conda env update -f environment.yaml --prune`。

5.  **安装额外依赖**：
    确保 `ldm` Conda 环境已激活。
    
    ```bash
    # 下面是environment.yaml里面pip需要安装的库以及对应版本，如果前面能正常安装就不需要管
    # 使用清华镜像源加速下载
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
    
6.  **验证 `torchmetrics` 版本兼容性问题**：
    
    ```bash
    python -c "import pytorch_lightning; print(f'pytorch-lightning version: {pytorch_lightning.__version__}'); import torchmetrics; print(f'torchmetrics version: {torchmetrics.__version__}')"
    ```
    确保能打印出版本且无报错。

---

**C. 下载所需文件 (模型权重、CLIP、安全检查器)**

*   **基础目录**：这些文件通常放置在 `stable-diffusion` 项目目录中，具体路径取决于脚本配置和官方指引。

1.  **Stable Diffusion 预训练权重模型**：
    
    *   **来源**：[CompVis on Hugging Face](https://huggingface.co/CompVis)
    *   **选择**：例如 `CompVis/stable-diffusion-v1-4` -> `sd-v1-4.ckpt` (文件名可能为 `v1-4.ckpt` 或 `sd-v1-4.ckpt`，通常约4GB)。
    *   **下载文件**：`sd-v1-4.ckpt` (或类似名称)。
    *   **放置**：官方通常建议放在 `models/ldm/stable-diffusion-v1/` 目录下，并可能需要重命名为 `model.ckpt`。
        
        ```bash
        # 在 stable-diffusion 目录下
        mkdir -p models/ldm/stable-diffusion-v1/
        # 将下载的 .ckpt 文件移动到这里并重命名 (如果需要)
        # mv path/to/your/sd-v1-4.ckpt models/ldm/stable-diffusion-v1/model.ckpt
        ```
        或者，您可以将其放在 `stable-diffusion` 根目录下，并在运行时通过 `--ckpt` 参数指定其路径。根据您的 `txt2img.py` 脚本参数 `sd-v1-4.ckpt`，它期望在脚本执行的当前目录或可访问路径下找到此文件。
    
2.  **CLIP 模型 (`clip-vit-large-patch14`)**：
    *   Stable Diffusion 的 `environment.yaml` 通常会通过 `pip install git+https://github.com/openai/CLIP.git@main#egg=clip` 来安装 CLIP。如果此步骤成功，则无需手动下载其权重。
    *   如果脚本明确要求或自动下载失败，CLIP 的权重和配置文件通常由 PyTorch Hub 或 Transformers 库在首次使用时自动下载和缓存到用户目录 (如 `~/.cache/clip` 或 Hugging Face cache)。
    *   笔记中提到的手动下载 CLIP 文件并放置到 `openai/clip-vit-large-patch14` 的步骤，更多是针对特定项目结构或解决自动下载问题。对于标准 Stable Diffusion 复现，通常不需要这一步，除非遇到特定导入或加载错误。

3.  **Stable Diffusion 安全性检查器**：
    
    *   与 CLIP 类似，安全性检查器相关文件也通常由 Stable Diffusion 脚本在需要时通过 Hugging Face Hub 自动下载和缓存。
    *   手动下载并放置到 `CompVis/stable-diffusion-safety-checker` 的步骤，同样是针对特定情况。标准流程依赖于自动下载。
    
    *   **注意**：对于第2和第3点，建议首先尝试让 Stable Diffusion 脚本自动处理这些依赖项的下载。仅当自动下载因网络问题或配置问题失败时，再考虑手动下载并放置到脚本期望的缓存路径（通常在 `~/.cache/huggingface/hub/` 或类似位置，具体路径需根据错误信息确定）。

---

**D. 常见问题与解决方案**

1. **模型下载网络连接问题 (`ConnectionResetError`)**：

   *   **症状**：脚本在运行时尝试从 URL (如 GitHub, Hugging Face) 下载预训练模型文件 (例如 `checkpoint_liberty_with_aug.pth` 或其他依赖) 时，出现 `ConnectionResetError: [Errno 104] Connection reset by peer`。
   *   **场景示例**：下载 `kornia` 依赖的 `"https://github.com/DagnyT/hardnet/raw/master/pretrained/train_liberty_with_aug/checkpoint_liberty_with_aug.pth"` 到 `/root/.cache/torch/hub/checkpoints/checkpoint_liberty_with_aug.pth` 失败。
   *   **解决方案：手动下载并放置文件**：
       
       1.  **本地下载**：在您的本地电脑 (如 Windows) 使用浏览器或下载工具下载目标 URL 的文件，确保文件名与脚本期望的一致。
       2.  **创建服务器目标目录**：在云服务器上，确保脚本期望的缓存路径存在。
           
           ```bash
           # 示例路径，根据实际错误日志调整
           mkdir -p /root/.cache/torch/hub/checkpoints/
           # 或者 Hugging Face 缓存路径，例如：
           # mkdir -p /root/.cache/huggingface/hub/models--CompVis--stable-diffusion-v1-4/blobs/
           ```
           (注意：`/root/` 路径表示您是以 root 用户运行。如果以其他用户运行，路径可能是 `~/.cache/...`)
       3.  **上传文件**：使用 `scp`、FileZilla、WinSCP 等工具将下载的文件上传到云服务器的对应目标目录中。
       4.  **重新运行脚本**：文件上传后，再次运行之前失败的 Python 脚本。相关库应能检测到本地缓存的文件并跳过下载。

---

**E. 运行 Stable Diffusion 文本到图像生成脚本 (示例)**

*   确保您在 `stable-diffusion` 目录下，并且 `ldm` 环境已激活。
*   **文本到图像生成脚本 (来自笔记)**：
    
    ```bash
    python txt2img.py --prompt "a photograph of an astronaut riding a horse" \
                              --ckpt sd-v1-4.ckpt \
                              --config configs/stable-diffusion/v1-inference.yaml \
                              --H 384 --W 384 \
                              --plms
    ```
    *   **参数说明**：
        *   `--prompt`: 您想要生成的图像的文本描述。
        *   `--ckpt`: 指向您下载的 Stable Diffusion 模型权重文件 (`.ckpt`) 的路径。这里假设您已按C部分建议放置并命名为 `model.ckpt`。如果放在根目录且名为 `sd-v1-4.ckpt`，则应为 `--ckpt sd-v1-4.ckpt`。
        *   `--config`: 指向适用于所用模型的配置文件。对于 v1.4/v1.5 模型，通常是 `configs/stable-diffusion/v1-inference.yaml`。
        *   `--H`, `--W`: 生成图像的高度和宽度。512x512 是 v1 模型的标准尺寸，这里由于显存原因我们生成384x384的图像。
        *   `--plms` (或 `--ddim`): 使用的采样器类型。PLMS 和 DDIM 是常见的选项。
    *   **输出**：生成的图像通常保存在 `outputs/txt2img-samples/` 目录中。

