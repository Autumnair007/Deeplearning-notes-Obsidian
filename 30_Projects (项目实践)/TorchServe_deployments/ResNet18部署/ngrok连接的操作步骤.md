---
type: "tutorial"
tags: [ngrok, torchserve, model-deployment, remote-access, linux, windows, tutorial]
status: "done"
summary: "使用ngrok从Windows本地电脑访问Ubuntu云主机（PAI-DSW）上TorchServe服务的详细操作指南。流程分为两部分：在云主机上下载、配置Authtoken并启动ngrok隧道（转发本地8080端口），以及在本地Windows终端上使用curl命令和ngrok生成的临时公共URL，携带TorchServe的推理Token，进行远程模型推理测试。强调了ngrok URL的临时性和保持ngrok进程运行的重要性。"
---
下面是 **ngrok** 的完整步骤整理，用于从 **Windows 11 本地电脑** 访问运行在 **Ubuntu 云主机 (PAI-DSW)** 上的 TorchServe 服务（监听 8080 端口）：

**第一部分：在 Ubuntu 云主机 (PAI-DSW) 上操作**

1.  **下载 ngrok:**
    
    *   在你的工作目录下（例如 `/mnt/workspace/my_resnet_deployment`），使用 `wget` 下载 ngrok 的 Linux 可执行文件压缩包。
        ```bash
        wget https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-linux-amd64.tgz
        ```
    
2.  **解压缩 ngrok:**
    
    *   使用 `tar` 命令解压刚刚下载的文件，得到 `ngrok` 可执行文件。
        ```bash
        tar xvzf ngrok-v3-stable-linux-amd64.tgz
        ```
    
3.  **获取并配置 ngrok Authtoken (身份验证令牌):**
    
    *   访问 ngrok 官网 ([https://ngrok.com/](https://ngrok.com/)) 并注册/登录你的账户。
    *   在你的 ngrok Dashboard找到你的个人 Authtoken。它会显示在一个类似 `ngrok config add-authtoken <你的令牌>` 的命令里。
    *   **复制** 这条完整的命令。
    *   回到你的 Ubuntu 终端，**粘贴并运行** 这条命令（注意，因为 `ngrok` 在当前目录，所以命令前要加 `./`）：
        ```bash
        ./ngrok config add-authtoken <粘贴你从ngrok网站复制的那个长长的令牌> 
        ```
    *   这会将你的令牌保存到 ngrok 的配置文件中，以后就不需要再输了。
    
4.  **启动 ngrok 隧道:**
    
    *   运行 `ngrok`，让它将公网的 HTTP/HTTPS 请求转发到你本地正在监听 `8080` 端口的 TorchServe 推理服务。
        ```bash
        ./ngrok http 8080
        ```
        
    *   ngrok 启动后，会显示类似下面的信息：
        ```
        Forwarding                    https://<随机字符>.ngrok-free.app -> http://localhost:8080 
        ```
        
    * ![](../../../99_Assets%20(资源文件)/images/image-20250429225941145.png)
    
    * **记下** 这个 `https://<随机字符>.ngrok-free.app` 的 URL (我们称之为 `<你的ngrok推理URL>`)，这是你从外部访问服务的入口。
    
    *   **保持这个 ngrok 进程在前台运行**，不要关闭这个终端窗口或按 `Ctrl+C`。

**第二部分：在 Windows 11 本地电脑上操作**

5.  **准备测试数据 (如果需要):**
    
    * 打开 Windows 终端 (Windows Terminal, PowerShell, 或 cmd)。
    
    * 使用 `cd /d "你的目标目录"` 命令切换到包含测试图片（例如 `kitten.jpg`）的目录。
    
    * (如果还没有图片) 使用 `curl -o kitten.jpg <图片URL>` 下载测试图片到当前目录。例如：
    
    * ```bash
      curl -o kitten.jpg https://raw.githubusercontent.com/pytorch/serve/master/examples/image_classifier/kitten.jpg
      ```
    
6.  **执行测试命令:**
    
    *   在**已经切换到包含测试图片的目录**的 Windows 终端中，使用 `curl` 命令，通过第 4 步获取的 `<你的ngrok推理URL>` 来向 TorchServe 发送请求。
    *   **关键：** 使用**相对路径** (`kitten.jpg`) 来指定上传的文件，以避免因路径中包含中文或其他特殊字符导致的编码问题。
    *   记得包含 TorchServe 需要的 `Authorization` 头，并替换 `<你的推理Token>`。
        ```bash
        # 确保当前目录是包含 kitten.jpg 的目录
        curl -X POST <你的ngrok推理URL>/predictions/resnet-18 -T kitten.jpg -H "Authorization: Bearer <你的推理Token>" 
        # 例如
        # curl -X POST https://dff4-120-27-137-238.ngrok-free.app/predictions/resnet-18 -T kitten.jpg -H "Authorization: Bearer FHV-WPeF"
        ```
        (将 `<你的ngrok推理URL>` 和 `<你的推理Token>` 替换成你实际的值)
    
7.  **查看结果:**
    
    *   如果一切正常，终端会输出 TorchServe 返回的 JSON 格式的推理结果。

**重要提示:**

*   **保持 ngrok 运行:** 只要你想从外部访问服务，Ubuntu 云主机上的 ngrok 进程就必须保持运行。
*   **URL 时效性:** ngrok 免费版提供的 `Forwarding` URL 是**临时的**。每次你停止并重新启动 ngrok 进程，这个 URL 都会改变。
*   **推理 vs 管理端口:** 这个流程是针对推理端口 (8080) 的。如果你需要访问管理端口 (8081)，你需要在 Ubuntu 上停止当前的 ngrok (`Ctrl+C`)，然后运行 `./ngrok http 8081`，获取一个新的 URL，并在 Windows 上使用这个新 URL 和**管理 Token** 进行测试。
