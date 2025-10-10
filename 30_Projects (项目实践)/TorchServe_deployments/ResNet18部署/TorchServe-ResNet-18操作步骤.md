## TorchServe ResNet-18 操作步骤

本指南将指导你完成使用预训练的 ResNet-18 模型部署 TorchServe 的过程。

参考文档：[serve/examples/image_classifier/resnet_18 at master · pytorch/serve](https://github.com/pytorch/serve/tree/master/examples/image_classifier/resnet_18)，里面文件仅供参考，直接按上面运行无法使用。

**假设:**

*   你已通过终端登录到云主机（如 魔搭社区的PAI-DSW、华为云或其他 Linux 服务器）。我自己使用的是PAI-DSW，属于PaaS，所以环境搭建非常快。如果环境搭建存在问题可能需要寻找对应的解决方案。
*   云主机是基于 Linux 的 (如 Ubuntu)。
*   你拥有 `sudo` 权限。
*   你的用户主目录是 `~` (即 `/home/<username>/`)。
*   你的云主机可以访问互联网 (特别是 GitHub 和 PyTorch 资源)。
*   当前日期和时间 (UTC): 2025-04-29 06:13:07
*   当前用户: username

---

**步骤 1：安装基础依赖**

1.  **更新系统包列表:**
    打开你的终端。

    ```bash
    sudo apt update
    ```

2.  **安装 Java (推荐 OpenJDK 11 或 17):**

    ```bash
    sudo apt install -y openjdk-11-jdk
    java -version # 验证安装，应显示 Java 版本信息
    ```

3.  **安装 Python 3 和 pip:**

    ```bash
    sudo apt install -y python3 python3-pip git wget
    python3 --version # 验证安装
    pip3 --version    # 验证安装
    ```

4.  **升级 pip (使用国内镜像):**
    为了提高下载速度，建议使用国内镜像。

    ```bash
    # pip3 install --upgrade pip
    pip3 install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple
    ```

---

**步骤 2：使用国内镜像安装 Python 库**

1.  **安装 PyTorch, TorchVision (CPU 版本示例):**
    ResNet-18 是 TorchVision 提供的模型，因此需要安装 TorchVision。这方面有其他问题再搜资料。

    ```bash
    # 如果你的环境支持 GPU 并且已正确配置 CUDA，可以选择安装 GPU 版本的 PyTorch
    # CPU 版本命令：
    # pip3 install torch torchvision torchaudio
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu -i https://pypi.tuna.tsinghua.edu.cn/simple
    ```
    
2.  **安装 TorchServe 相关库:**
    安装 `torchserve`, `torch-model-archiver`。ResNet-18 图像分类需要 `Pillow` 来处理图像，`numpy` 也通常需要。

    ```bash
    # pip3 install torchserve torch-model-archiver Pillow numpy
    pip3 install torchserve torch-model-archiver Pillow numpy -i https://pypi.tuna.tsinghua.edu.cn/simple
    ```

==上面安装的步骤（步骤1，步骤2）根据自己的实际情况调整即可。==

------

**步骤 3：创建工作区并获取 ResNet-18 模型文件**

1.  **创建工作目录:**
    在你的 Home 目录下创建一个新的工作区。

    ```bash
    mkdir ~/my_resnet_deployment
    # mkdir my_resnet_deployment
    ```
    
2.  **进入工作目录:**
    后续所有操作都在这个目录下进行。

    ```bash
    cd ~/my_resnet_deployment
    # cd my_resnet_deployment
    ```
    
3.  **确认当前位置 (重要!):**

    ```bash
    pwd
    # 应该输出类似 /home/username/my_resnet_deployment 的路径
    ```

4.  **克隆 PyTorch Serve 仓库以获取示例文件:**
    我们需要 ResNet-18 的模型定义文件 (`model.py`)、类别映射文件 (`index_to_name.json`) 和测试图片 (`kitten.jpg`)。

    ```bash
    git clone https://github.com/pytorch/serve.git
    ```

    假如从其他文件夹里面复制serve文件夹可以参考下面代码：
    
    ```bash
    cp -r my_mnist_deployment/serve my_resnet_deployment/
    ```
    
5.  **准备 ResNet-18 相关文件:**

    * **重写模型定义文件:**

      创建`model.py`文件:
    
      ```bash
      nano model.py
      ```

      如果没有nano文本编辑器可以用下面命令安装：
    
      ```bash
      sudo apt-get update && apt-get install -y nano
      ```

      官方的原文件是直接继承的。为了方便学习，我重新写了ResNet-18的模型文件，代码如下：
    
      ```python
      import torch
      import torch.nn as nn
      import torchvision.models as models
      
      class ResNet18Classifier(nn.Module):
          """
          ResNet-18 model for image classification, adapted for TorchServe
          """
          def __init__(self, num_classes=1000):
              super(ResNet18Classifier, self).__init__()
              # Load pre-trained ResNet-18 model
              self.resnet18 = models.resnet18(pretrained=False)
              # Replace the final fully connected layer if needed
              if num_classes != 1000:
                  self.resnet18.fc = nn.Linear(self.resnet18.fc.in_features, num_classes)
          
          def forward(self, x):
              return self.resnet18(x)
      ```
    
      **保存并退出:** 按 `Ctrl+X`，然后按 `Y`，最后按 `Enter`。
      文件位于 `/home/username/my_resnet_deployment/config.properties`
    
    *   **复制类别映射文件:**
    
        ```bash
        cp serve/examples/image_classifier/index_to_name.json .
        ```
    
    *   **创建测试数据目录并复制测试图片:**
    
        ```bash
        mkdir test_data
        cp serve/examples/image_classifier/kitten.jpg test_data/
        ```
    
    *   **下载预训练权重:**
    
        ```bash
        wget https://download.pytorch.org/models/resnet18-f37072fd.pth
        ```
        
    *   **重写`handler`代码：**
    
        创建`resnet18_handler.py`文件:
        
        ```bash
        nano resnet18_handler.py
        ```
        
        关于ResNet-18 模型部署问题排查与解决报告可以看 [TorchServe-ResNet-18补充](TorchServe-ResNet-18补充.md)。
        
        具体代码在 **resnet18_handler.py**。
    
    现在你的 `~/my_resnet_deployment` 目录下应该有:
    
    ![image-20250429200908395](TorchServe-ResNet-18操作步骤.assets/image-20250429200908395.png)
    
    *   `model.py` (自己写的ResNet-18 模型结构定义)
    *   `resnet18-f37072fd.pth` (下载的预训练权重)
    *   `index_to_name.json` (类别索引到名称的映射)
    *   `serve/` (克隆的仓库目录)
    *   `test_data/` (子目录，内含 `kitten.jpg`)
    *   `resnet18_handler.py` (自己写的handler文件)

---

**步骤 4：检查 `torch-model-archiver` 并修复路径 (关键排错步骤)**

**确保你仍然在 `~/my_resnet_deployment` 目录下执行以下检查。**如果不在，请先执行下面命令：

 ```bash
 cd ~/my_resnet_deployment
 # cd my_resnet_deployment
 ```

1.  **检查 `torch-model-archiver` 是否已安装:**

    ```bash
    pip3 list | grep torch-model-archiver
    ```

    *   如果看到类似 `torch-model-archiver 0.x.x` 的输出，请继续第 2 步。
    *   如果**没有输出**，回到**步骤 2.2** 重新安装，然后再回到这里继续。

2.  **检查 PATH 环境变量:**

    ```bash
    echo $PATH
    ```

    仔细查看输出，是否包含 `/home/user/.local/bin` 。

3.  **如果 PATH 中 *不包含* `~/.local/bin`:**

    *   **永久添加 (推荐):**

        ```bash
        echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
        source ~/.bashrc
        # 可能需要关闭当前终端并重新打开一个才能完全生效
        ```

4.  **最终验证:**
    打开一个新的终端（如果执行了上一步的 `source` 可能不需要，但新终端更保险），或者在当前终端尝试：

    ```bash
    torch-model-archiver --version
    ```

    如果看到用法信息（提示缺少参数）而不是 "command not found"，说明 `torch-model-archiver` 已在你的 PATH 中，可以正常使用了。

---

**步骤 5：打包 ResNet-18 模型为 .mar 文件**

**确保你仍然在 `~/my_resnet_deployment` 目录下。**

1.  **创建用于存放 .mar 文件的目录:**

    ```bash
    mkdir model_store
    ```

    目录位于 `/home/username/my_resnet_deployment/model_store/`

2.  **执行打包命令 (使用 ResNet-18 的文件和内置处理器):**

    ```bash
    # --model-name: 指定模型在 TorchServe 中的名称
    # --version: 模型版本
    # --model-file: 指定模型定义的 Python 文件 (我们之前复制的 model.py)
    # --serialized-file: 指定预训练权重文件 (我们下载的 .pth 文件)
    # --handler: 指定处理器。'image_classifier' 是 TorchServe 内置的处理器，适用于常见的图像分类任务，会自动处理图像预处理和结果格式化。
    # --extra-files: 指定需要一起打包的额外文件，这里是类别映射 JSON 文件。
    # --export-path: 指定 .mar 文件输出的目录。
    # -f: 强制覆盖同名文件。
    torch-model-archiver --model-name resnet-18 \
                         --version 1.0 \
                         --model-file model.py \
                         --serialized-file resnet18-f37072fd.pth \
                         --handler resnet18_handler.py \
                         --extra-files index_to_name.json \
                         --export-path model_store \
                         -f
    ```

    打包成功后，检查 `model_store` 目录确认 `resnet-18.mar` 文件已生成:

    ```bash
    ls -l model_store
    # 应能看到 resnet-18.mar 文件
    ```

---

#### ==注意，下面的版本是不使用Token的版本！后面还有使用Token的操作步骤！==

[使用Token的步骤](#下面是使用Token的步骤：从步骤六开始) 

**==关于 Token 授权== **

参考资料： [TorchServe-Token](TorchServe-Token.md)

*   **默认启用:** 如果你没有在 `config.properties` 或启动命令中添加 `disable_token_authorization=true` / `--disable-token-auth`，则 Token 授权是**默认开启**的。
*   **`key_file.json`:** 启动时会在**当前工作目录** (`~/my_resnet_deployment`) 生成 `key_file.json`。
*   **获取 Keys:** 打开 `key_file.json` 文件 (例如使用 `cat key_file.json`)，复制 `management` 和 `inference` 对应的 key 值。
*   **使用 Keys:** 在 `curl` 或其他 API 请求中添加 HTTP Header:
    *   管理 API (端口 8081): `-H "Authorization: Bearer <management_key>"`
    *   推理 API (端口 8080): `-H "Authorization: Bearer <inference_key>"`
*   **有效期:** Key 默认 60 分钟有效。
*   **禁用:** 最简单的方法是在启动时使用 `--disable-token-auth` 参数或在 `config.properties` 中设置 `disable_token_authorization=true`。

------

**步骤 6：创建 TorchServe 配置文件 (可选但推荐)**

**确保你仍然在 `~/my_resnet_deployment` 目录下。**

1. **创建并编辑配置文件:**

   ```bash
   nano config.properties
   ```

   **粘贴以下内容:**
   (使用 `Ctrl+Shift+V` 或鼠标右键粘贴到 nano 编辑器中)

   ```properties name=config.properties
   inference_address=http://0.0.0.0:8080
   management_address=http://0.0.0.0:8081
   metrics_address=http://127.0.0.1:8085
   # model_store=model_store # 我们将在启动命令中指定，这里注释掉
   # 允许通过环境变量覆盖配置 (可选)
   enable_envvars_config=true
   # 明确禁用 Token 授权，方便测试 (如果需要 Token，请注释或删除此行)
   disable_token_authorization=true
   # 配置 Job Queue 大小 (可选，根据需要调整)
   # default_workers_per_model=1
   ```

3.  **保存并退出:** 按 `Ctrl+X`，然后按 `Y`，最后按 `Enter`。
    文件位于 `/home/username/my_resnet_deployment/config.properties`

---

**步骤 7：启动 TorchServe 并加载 ResNet-18 模型**

**确保你仍然在 `~/my_resnet_deployment` 目录下。**

```bash
# 使用当前目录的 config.properties 和 model_store 目录
# 启动时加载 resnet-18.mar 模型，并将其命名为 "resnet-18" (等号左边是在服务中注册的名字，右边是mar文件名)
# --ncs 禁用快照，推荐用于稳定版本
torchserve --start --ncs --ts-config config.properties --model-store model_store --models resnet-18=resnet-18.mar

# 如果没有创建 config.properties 或想覆盖其中的设置，可以直接在命令行指定：
# torchserve --start --ncs --model-store model_store --models resnet-18=resnet-18.mar --disable-token-auth --foreground --inference-address http://0.0.0.0:8080 --management-address http://0.0.0.0:8081

```

*   `--start`: 启动服务。
*   `--ncs`: 禁用快照功能。
*   `--ts-config`: 指定配置文件路径。
*   `--model-store`: 指定包含 `.mar` 文件的目录。
*   `--models resnet-18=resnet-18.mar`: 启动时加载模型。`resnet-18` 是 API 调用时使用的模型名称，`resnet-18.mar` 是实际的模型归档文件。
*   `--disable-token-auth` (如果未在 `config.properties` 中设置): 明确禁用令牌认证。
*   `--foreground` (可选): 在前台运行并显示日志，方便调试。按 `Ctrl+C` 停止。如果不用此选项，服务将在后台运行。

*   **查看后台日志**（如果未使用 `--foreground`）:
    日志文件通常位于启动命令执行的目录下的 `logs/` 子目录中。你可以打开一个新的终端来查看。

    ```bash
    # 确保你在 ~/my_resnet_deployment 目录下
    tail -f logs/ts_log.log # 按 Ctrl+C 退出查看
    ```
    你应该能看到类似 "Model resnet-18 loaded" 的信息。

---

**步骤 8：测试服务**

1. **从云主机内部测试:**
   **打开一个新的终端窗口** (保持 TorchServe 在第一个终端运行，特别是如果用了 `--foreground`)。在新终端中，**首先进入工作目录**：

   ```bash
   # 在新终端中执行
   cd ~/my_resnet_deployment
   # cd my_resnet_deployment
   ```

*   **健康检查:**

    ```bash
    # 如果启用了 Token 授权，此命令需要 inference token:
    # curl http://localhost:8080/ping -H "Authorization: Bearer <inference_key>"
    # 如果禁用了 Token 授权：
    curl http://localhost:8080/ping
    # 预期输出: {"status": "Healthy"}
    ```

*   **模型列表:**

    ```bash
    # 如果启用了 Token 授权，需要 management token:
    # curl http://localhost:8081/models -H "Authorization: Bearer <management_key>"
    # 如果禁用了 Token 授权：
    curl http://localhost:8081/models
    # 预期输出应包含 "modelName": "resnet-18" 的信息
    ```

*   **推理测试 (使用之前复制的 `kitten.jpg`):**

    ```bash
    # 如果启用了 Token 授权，需要 inference token:
    # curl -X POST http://localhost:8080/predictions/resnet-18 -T test_data/kitten.jpg -H "Authorization: Bearer <inference_key>"
    # 如果禁用了 Token 授权：
    curl -X POST http://localhost:8080/predictions/resnet-18 -T test_data/kitten.jpg
    ```

    **预期输出:** 你应该会看到一个 JSON 响应，列出可能性最高的几个类别及其置信度，类似于：

    ```json
    {
      "tabby": 0.40966343879699707,
      "tiger_cat": 0.346704363822937,
      "Egyptian_cat": 0.13002890348434448,
      "lynx": 0.023919545114040375,
      "bucket": 0.011532172560691833
    }
    ```
    ![image-20250429201453197](TorchServe-ResNet-18操作步骤.assets/image-20250429201453197.png)
    

**步骤 9：停止 TorchServe**

当你完成测试或不再需要服务时，可以在**任何目录下**执行停止命令：

```bash
torchserve --stop
```

这将停止在后台运行的 TorchServe 进程。如果之前使用了 `--foreground` 选项，直接在运行的终端按 `Ctrl+C` 即可。

------

#### ==下面是使用Token的步骤：从步骤六开始==

[返回不使用Token的步骤](#注意，下面的版本是不使用Token的版本！后面还有使用Token的操作步骤！) 

------

**步骤 6 (修改)：创建 TorchServe 配置文件 (启用 Token 授权)**

**确保你在工作目录下，例如 `~/my_resnet_deployment`。**

1.  **创建并编辑配置文件:**

    ```bash
    nano config.properties
    ```

2.  **粘贴以下内容:**
    (确保 `disable_token_authorization` 要么被注释掉，要么设置为 `false`，或者干脆不包含这行，因为默认就是启用 Token)

    ```properties name=config.properties
    inference_address=http://0.0.0.0:8080
    management_address=http://0.0.0.0:8081
    metrics_address=http://127.0.0.1:8085
    # model_store=model_store # 我们将在启动命令中指定
    # 允许通过环境变量覆盖配置
    enable_envvars_config=true
    # Token 授权默认启用，以下任一方式都可以显式启用或保持默认启用：
    # 方式一：明确设置为 false (启用)
    # disable_token_authorization=false
    # 方式二：注释掉或删除此行 (默认启用)
    # disable_token_authorization=true # <--- 确保这一行不存在或被注释掉
    
    # (可选) 设置 Token 过期时间 (默认为 60 分钟)
    # token_expiration_min=30
    
    # 配置 Job Queue 大小 (可选)
    # default_workers_per_model=1
    ```

3.  **保存并退出:** 按 `Ctrl+X`，然后按 `Y`，最后按 `Enter`。

---

**步骤 7 (修改)：启动 TorchServe (启用 Token 授权)**

**确保你仍然在 `~/my_resnet_deployment` 目录下。**

```bash
# 使用当前目录的 config.properties 和 model_store 目录
# 启动时加载 resnet-18.mar 模型
# 确保没有使用 --disable-token-auth 标志
torchserve --start --ncs --ts-config config.properties --model-store model_store --models resnet-18=resnet-18.mar

# 如果不使用 config.properties，命令行如下 (注意没有 --disable-token-auth):
# torchserve --start --ncs --model-store model_store --models resnet-18=resnet-18.mar --inference-address http://0.0.0.0:8080 --management-address http://0.0.0.0:8081
```

*   `--start`, `--ncs`, `--ts-config`, `--model-store`, `--models`: 参数含义同之前的示例。
*   **关键:** 我们**没有**使用 `--disable-token-auth` 命令行标志，并且 `config.properties` 也**没有**设置 `disable_token_authorization=true`。因此，Token 授权将**默认启用**。

*   **查看日志:**
    
    ```bash
    # 确保你在 ~/my_resnet_deployment 目录下
    tail -f logs/ts_log.log # 按 Ctrl+C 退出查看
    ```
    你应该能看到类似 "Token Authorization Enabled" 和 "Model resnet-18 loaded" 的信息。

---

**步骤 7.5：获取访问 Tokens**

**因为 Token 授权已启用，TorchServe 会在启动时生成密钥文件。**

1.  **定位密钥文件:**
    该文件名为 `key_file.json`，它会被创建在**你运行 `torchserve --start` 命令的那个目录下** (也就是 `~/my_resnet_deployment`)。

2. **查看密钥文件内容:**

   ```bash
   # 确保你在 ~/my_resnet_deployment 目录下
   cat key_file.json
   ```

   你会看到类似下面的 JSON 内容：

   ```json
   {
     "management": {
       "key": "SOME_RANDOM_MGMT_KEY",  # <--- 你需要这个 Key
       "expiration time": "..."
     },
     "inference": {
       "key": "SOME_RANDOM_INFER_KEY", # <--- 你需要这个 Key
       "expiration time": "..."
     },
     "API": {
       "key": "SOME_RANDOM_API_KEY"      # <--- 这个用于生成新 Key
     }
   }
   ```

   ![image-20250429214757444](TorchServe-ResNet-18操作步骤.assets/image-20250429214757444.png)

3.  **复制 Keys:**
    
    *   复制 `management` 下的 `key` 值 (例如 `SOME_RANDOM_MGMT_KEY`)。这是你的**管理 Token**。
    *   复制 `inference` 下的 `key` 值 (例如 `SOME_RANDOM_INFER_KEY`)。这是你的**推理 Token**。
    *   **注意:** 这些 Key 是随机生成的，每次启动服务都会不同。它们有默认 60 分钟的有效期（除非你在 `config.properties` 中修改了 `token_expiration_min`）。

---

**步骤 8 (修改)：使用 Tokens 测试服务**

**现在所有的 API 调用都需要在 HTTP Header 中包含正确的 Token。**

1.  **从云主机内部测试:**
    **打开一个新的终端窗口**，并**首先进入工作目录**：

    ```bash
    # 在新终端中执行
    cd ~/my_resnet_deployment
    ```

*   **健康检查 (需要推理 Token):**
    将 `<inference_key>` 替换为你从 `key_file.json` 复制的 `inference` key。

    ```bash
    # 将 SOME_RANDOM_INFER_KEY 替换为你的实际 inference key
    curl http://localhost:8080/ping -H "Authorization: Bearer SOME_RANDOM_INFER_KEY"
    # 例如我上面的例子就是：
    curl http://localhost:8080/ping -H "Authorization: Bearer -30lK-9j"
    # 预期输出: {"status": "Healthy"}
    
    # 如果不加 Token，会收到 401 Unauthorized 错误
    # curl http://localhost:8080/ping 
    ```
    
*   **模型列表 (需要管理 Token):**
    将 `<management_key>` 替换为你从 `key_file.json` 复制的 `management` key。

    ```bash
    # 将 SOME_RANDOM_MGMT_KEY 替换为你的实际 management key
    curl http://localhost:8081/models -H "Authorization: Bearer SOME_RANDOM_MGMT_KEY"
    # 预期输出应包含 "modelName": "resnet-18" 的信息
    # 例如我上面的例子就是：
    curl http://localhost:8081/models -H "Authorization: Bearer aqViSuIC"
    # 如果不加 Token，会收到 401 Unauthorized 错误
    # curl http://localhost:8081/models
    ```
    
*   **推理测试 (需要推理 Token):**
    将 `<inference_key>` 替换为你从 `key_file.json` 复制的 `inference` key。

    ```bash
    # 将 SOME_RANDOM_INFER_KEY 替换为你的实际 inference key
    curl -X POST http://localhost:8080/predictions/resnet-18 -T test_data/kitten.jpg -H "Authorization: Bearer SOME_RANDOM_INFER_KEY"
    # 例如我上面的例子就是：
    curl -X POST http://localhost:8080/predictions/resnet-18 -T test_data/kitten.jpg -H "Authorization: Bearer -30lK-9j"
    ```

    **预期输出:** 你应该会看到与之前一样的 JSON 推理结果。
    
    ```json
    {
      "tabby": 0.40966343879699707,
      "tiger_cat": 0.346704363822937,
      // ... 其他类别 ...
    }
    ```
    
    ![image-20250429215148358](TorchServe-ResNet-18操作步骤.assets/image-20250429215148358.png)
    
    **如果你不提供正确的 `Authorization` Header，所有这些请求都会失败并返回 `401 Unauthorized` 错误。**

2.  **从外部机器测试:**
    
    *   获取云主机**公网 IP 地址**。
    *   确保**安全组/防火墙**允许外部访问 TCP 端口 8080 和 8081。
    *   在本地电脑终端执行 (替换 `<云主机公网IP>`、`<inference_key>` 或 `<management_key>` 以及本地图片路径):
    
    ```bash
    # 健康检查 (需要推理 Token)
    curl http://<云主机公网IP>:8080/ping -H "Authorization: Bearer <inference_key>"
    
    # 模型列表 (需要管理 Token)
    curl http://<云主机公网IP>:8081/models -H "Authorization: Bearer <management_key>"
    
    # 推理请求 (需要推理 Token)
    # 确保你的本地电脑上有图片文件
    curl -X POST http://<云主机公网IP>:8080/predictions/resnet-18 -T /path/on/your/local/machine/kitten.jpg -H "Authorization: Bearer <inference_key>"
    ```

我这里通过了ngrok来测试，结果也是没有问题的。具体步骤参考：[ngrok连接的操作步骤](ngrok连接的操作步骤.md)

![image-20250429225038291](TorchServe-ResNet-18操作步骤.assets/image-20250429225038291.png)

------

**步骤 9：停止 TorchServe**

停止服务的命令不变：

```bash
torchserve --stop
```

停止服务后，`key_file.json` 文件会被自动删除。下次启动时会生成新的密钥文件。
