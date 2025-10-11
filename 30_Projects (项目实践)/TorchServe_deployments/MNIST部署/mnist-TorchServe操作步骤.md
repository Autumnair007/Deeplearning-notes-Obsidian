---
type: "tutorial"
tags: [pytorch, torchserve, model-deployment, mnist, linux, machine-learning, tutorial]
status: "done"
summary: "TorchServe部署MNIST模型的完整操作步骤。本教程指导用户在基于Linux的云主机上完成依赖安装（包括Java和TorchServe）、工作区设置、TorchServe示例文件的获取、模型打包（.mar文件创建）、TorchServe配置文件的创建与服务启动，以及最终的内部和外部服务测试。同时，详细说明了新版TorchServe中Token授权的启用和禁用方法。"
---
我自己是在魔搭社区的PAI-DSW（内置了Ubuntu系统和pytorch）的开发环境测试了下面的步骤，没有问题。前面的安装依赖等步骤我在另一台华为云的云主机里面测试了，TorchServe能正常使用，但是由于网络问题我换了一个云电脑测试。

参考文档：[serve/examples/image_classifier/mnist at master · pytorch/serve](https://github.com/pytorch/serve/tree/master/examples/image_classifier/mnist)

**假设:**

*   你已通过终端登录到云主机。
*   云主机是基于 Linux 的 (如 Ubuntu)。
*   你拥有 `sudo` 权限。
*   你的用户主目录是 `~` (即 `/home/username/`)。
*   你的云主机可以访问互联网 (特别是 GitHub 和 PyTorch 资源)。

---

**步骤 1：安装基础依赖**

1.  **更新系统包列表:**

    ```bash
    sudo apt update
    ```

2.  **安装 Java (推荐 OpenJDK 11 或 17):**

    ```bash
    sudo apt install -y openjdk-11-jdk
    java -version # 验证安装
    ```

3.  **安装 Python 3 和 pip:**

    ```bash
    sudo apt install -y python3 python3-pip git wget
    python3 --version # 验证安装
    pip3 --version    # 验证安装
    ```

4.  **升级 pip (使用国内镜像):**

    ```bash
    pip3 install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple
    ```

---

**步骤 2：使用国内镜像安装 Python 库**

如果能直接连接到外网也可以不使用清华镜像源。使用清华镜像源的网址为：(`https://pypi.tuna.tsinghua.edu.cn/simple`)。

1.  **安装 PyTorch, TorchVision (CPU 版本示例):**

    ```bash
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu -i https://pypi.tuna.tsinghua.edu.cn/simple
    ```

2.  **安装 TorchServe 相关库:**

    ```bash
    # 安装 torchserve, torch-model-archiver 及 MNIST 示例可能需要的库 (Pillow 用于图像处理)
    pip3 install torchserve torch-model-archiver captum numpy Pillow -i https://pypi.tuna.tsinghua.edu.cn/simple
    ```

---

**步骤 3：创建工作区并获取 MNIST 模型文件**

1.  **创建工作目录: **这个命令在 Home 目录下创建工作区。

    ```bash
    mkdir ~/my_mnist_deployment
    ```

2.  **进入工作目录:**

    ```bash
    cd ~/my_mnist_deployment
    ```

3.  **确认当前位置 (重要! 确保后续命令在此目录下执行):**

    ```bash
    pwd
    # 应该输出 /home/username/my_mnist_deployment
    ```

4.  **克隆 PyTorch Serve 仓库以获取示例文件:**
    (我们只需要其中的 `examples` 目录，但克隆整个仓库通常更简单)

    ```bash
    git clone https://github.com/pytorch/serve.git
    ```

5.  **将所需的 MNIST 文件复制到当前工作目录:**

    ```bash
    # 复制模型定义文件
    cp serve/examples/image_classifier/mnist/mnist.py .
    
    # 复制预训练权重文件
    cp serve/examples/image_classifier/mnist/mnist_cnn.pt .
    
    # 复制自定义处理器文件
    cp serve/examples/image_classifier/mnist/mnist_handler.py .
    
    # 创建测试数据目录并复制测试图片
    mkdir test_data
    cp serve/examples/image_classifier/mnist/test_data/0.png test_data/
    ```

    现在你的 `/home/username/my_mnist_deployment` 目录下应该有 `mnist.py`, `mnist_cnn.pt`, `mnist_handler.py` 和一个 `test_data` 子目录（内含 `0.png`）。

---

**步骤 4：检查 `torch-model-archiver` 并修复路径 (关键排错步骤)**

**确保你仍然在 `~/my_mnist_deployment` 目录下执行以下检查。**不在的话输入`cd ~/my_mnist_deployment`进入文件夹里面。

1. **检查 `torch-model-archiver` 是否已安装:**

   ```bash
   pip3 list | grep torch-model-archiver
   ```

   *   如果看到类似 `torch-model-archiver 0.x.x` 的输出，请继续第 2 步。
   *   ![](../../../99_Assets%20(资源文件)/images/image-20250428195518649.png)
   *   如果**没有输出**，回到**步骤 2.2** 重新安装，然后再回到这里继续。

2.  **检查 PATH 环境变量:**

    ```bash
    echo $PATH
    ```

    仔细查看输出，是否包含 `/home/username/.local/bin`。

3.  **如果 PATH 中 *不包含* `~/.local/bin`:**

    * **永久添加 (推荐):**

      ```bash
      echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
      source ~/.bashrc
      ```

4.  **最终验证:**

    ```bash
    torch-model-archiver --version
    ```

    如果是下面的的情况说明能正常运行：
    
    ```bash
    (base) developer@developer:/$ torch-model-archiver --version
    usage: torch-model-archiver [-h] --model-name MODEL_NAME
                                [--serialized-file SERIALIZED_FILE]
                                [--model-file MODEL_FILE] --handler HANDLER
                                [--extra-files EXTRA_FILES]
                                [--runtime {python,python3,LSP}]
                                [--export-path EXPORT_PATH]
                                [--archive-format {tgz,no-archive,zip-store,default}]
                                [-f] -v VERSION [-r REQUIREMENTS_FILE]
                                [-c CONFIG_FILE]
    torch-model-archiver: error: argument -v/--version: expected one argument

---

**步骤 5：打包 MNIST 模型为 .mar 文件**

**确保你仍然在 `~/my_mnist_deployment` 目录下。**

1.  **创建用于存放 .mar 文件的目录:**

    ```bash
    mkdir model_store
    ```

    *目录位于* `/home/username/my_mnist_deployment/model_store/`

2.  **执行打包命令 (使用 MNIST 特定的文件和处理器):**

    ```bash
    # 确保 torch-model-archiver 命令现在可用
    # 读取当前目录下的 .py, .pt, handler.py 文件
    # 将输出 mnist.mar 文件到 model_store 子目录
    torch-model-archiver --model-name mnist \
                         --version 1.0 \
                         --model-file mnist.py \
                         --serialized-file mnist_cnn.pt \
                         --handler mnist_handler.py \
                         --export-path model_store \
                         -f
    ```

    这里面的`mnist.py`,`mnist_cnn.pt`,`mnist_handler.py`,都是官方示例给的文件，如果我们自己部署应该要自己去弄。
    
    打包成功后，文件位于 `/home/username/my_mnist_deployment/model_store/mnist.mar`
    检查 `model_store` 目录确认文件已生成:
    
    ```bash
    ls -l model_store
    ```

---

**步骤 6：创建 TorchServe 配置文件 (可选但推荐)**

**确保你仍然在 `~/my_mnist_deployment` 目录下。**

1.  **创建并编辑配置文件:**

    ```bash
    nano config.properties
    ```

2.  **粘贴以下内容:**

    ```properties name=config.properties
    inference_address=http://0.0.0.0:8080
    management_address=http://0.0.0.0:8081
    metrics_address=http://127.0.0.1:8085
    # model_store=model_store # 我们将在启动命令中指定
    # 允许通过环境变量覆盖配置 (可选)
    enable_envvars_config=true
    # 如果想默认禁用 Token 授权 (不推荐)，可以取消下面这行的注释
    disable_token_authorization=true
    ```

3.  **保存并退出:** 按 `Ctrl+X`，然后按 `Y`，再按 `Enter`。
    文件位于 `/home/username/my_mnist_deployment/config.properties`

---

**步骤 7：启动 TorchServe 并加载 MNIST 模型**

**确保你仍然在 `~/my_mnist_deployment` 目录下。**

```bash
# 使用当前目录的 config.properties 和 model_store 目录
# 启动时加载 mnist.mar 模型，并将其命名为 "mnist"
# 添加 --disable-token-auth 明确禁用令牌认证（如果需要）
torchserve --start --ncs --ts-config config.properties --model-store model_store --models mnist=mnist.mar --disable-token-auth

# 如果不使用 config.properties 文件：
# torchserve --start --ncs --model-store model_store --models mnist=mnist.mar --disable-token-auth --foreground
```

*   `--ncs`: 禁用快照功能 (通常推荐用于生产)。
*   `--disable-token-auth`: 禁用令牌认证，方便测试。**如果移除此选项，请参考下面的 Token 授权说明。**
*   `--foreground`: (可选) 在前台运行并显示日志，方便调试。按 `Ctrl+C` 停止。如果不用此选项，服务将在后台运行。
*   查看后台日志（如果未使用 `--foreground`）:

    ```bash
    tail -f logs/ts_log.log # 按 Ctrl+C 退出
    ```

    日志文件位于 `/home/username/my_mnist_deployment/logs/`

---

**==关于 Token 授权== (如果未加 `--disable-token-auth`)**

关于这个Token的相关资料我上网找到了官方的说明：[serve/docs/token_authorization_api.md at master · pytorch/serve](https://github.com/pytorch/serve/blob/master/docs/token_authorization_api.md)，我测试的时候就直接禁用了这个功能，先方便测试，后面可以再探讨。

*   **默认启用:** 新版 TorchServe 默认开启 Token 授权。
*   **`key_file.json`:** 启动时会在**当前工作目录** (`~/my_mnist_deployment`) 生成 `key_file.json`。
*   **获取 Keys:** 从 `key_file.json` 中复制 `management` 和 `inference` 的 key。
*   **使用 Keys:** 在 `curl` 或其他 API 请求中添加 Header:
    *   管理 API: `-H "Authorization: Bearer <management_key>"`
    *   推理 API: `-H "Authorization: Bearer <inference_key>"`
*   **有效期:** Key 默认 60 分钟有效。
*   **禁用:** 使用 `--disable-token-auth` 启动参数（如上例所示）或在 `config.properties` 中设置 `disable_token_authorization=true`。
*   **警告：** 不要手动修改 `key_file.json`。

详细内容在这里：[TorchServe-Token](TorchServe-Token.md)

---

**步骤 8：测试服务**

1. **从云主机内部测试 (仍在 `~/my_mnist_deployment` 目录):**

   打开另一个终端，进入指定的目录：

   ```bash
   cd ~/my_mnist_deployment
   ```

   *   健康检查:

       ```bash
       # 如果启用了 Token 授权，此命令可能失败或需要 Token
       curl http://localhost:8080/ping
       ```

   *   模型列表:

       ```bash
       # 如果启用了 Token 授权，需要添加 -H "Authorization: Bearer <management_key>"
       curl http://localhost:8081/models
       ```

   *   推理测试 (使用之前复制的 `0.png`):

       ```bash
       # 如果启用了 Token 授权，需要添加 -H "Authorization: Bearer <inference_key>"
       curl -X POST http://localhost:8080/predictions/mnist -T test_data/0.png
       ```

       预期输出应该是一个表示识别出的数字（这个例子中可能是 `0`）的 JSON 响应。参考界面如下：

   ![](../../../99_Assets%20(资源文件)/images/image-20250428195942663.png)

2.  **从外部机器测试 (你的本地电脑):**

    ==这边暂时还没有测试，因为我的免费的环境公网IP不太好找==
    
    *   获取云主机**公网 IP 地址**。
    *   确保**安全组/防火墙**允许外部访问 TCP 端口 8080 (推理) 和 8081 (管理)。
    *   在本地电脑终端执行 (替换 `<云主机公网IP>` 和本地的 `0.png` 图片路径):
    
        ```bash
        # 健康检查 (如果启用了 Token，可能失败)
        curl http://<云主机公网IP>:8080/ping
        
        # 推理请求 (如果启用了 Token，需要 -H "Authorization: Bearer <inference_key>")
        # 确保你的本地电脑上有 0.png 文件，或者换成其他 MNIST 数字图片
        curl -X POST http://<云主机公网IP>:8080/predictions/mnist -T /path/on/your/local/machine/0.png
        ```

---

**步骤 9：停止 TorchServe**

当你不需要服务时，可以在任何目录下执行：

```bash
torchserve --stop
```

---

如果在任何步骤遇到问题，请检查命令输出和 TorchServe 日志 (`logs/ts_log.log`) 获取详细信息。
