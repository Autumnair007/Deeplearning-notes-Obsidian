# TorchServe Token详细解释

官方的说明：[serve/docs/token_authorization_api.md at master · pytorch/serve](https://github.com/pytorch/serve/blob/master/docs/token_authorization_api.md)，下面是中文翻译版：

---

**TorchServe 令牌授权 API**

TorchServe 现在默认强制执行令牌授权。

TorchServe 默认强制执行令牌授权，这意味着在调用 API 时需要提供正确的令牌。这是一项安全功能，旨在解决未经授权的 API 调用问题。这适用于未经授权的用户试图访问正在运行的 TorchServe 实例的场景。默认行为是启用此功能，它会创建一个包含用于 API 调用的适当令牌的密钥文件。用户可以禁用此功能以阻止 API 调用需要令牌授权（如何禁用），但请注意，这样做会使 TorchServe 面临潜在的未经授权的 API 调用风险。

**如何设置和禁用令牌授权**

1.  **全局环境变量：** 使用 `TS_DISABLE_TOKEN_AUTHORIZATION`，将其设置为 `true` 可禁用令牌授权，设置为 `false` 可启用。请注意，必须在 `config.properties` 中设置 `enable_envvars_config=true` 才能使用全局环境变量。
2.  **命令行：** 命令行只能通过添加 `--disable-token-auth` 标志来*禁用*令牌授权。
3.  **配置文件 (`config.properties`)：** 使用 `disable_token_authorization`，将其设置为 `true` 可禁用令牌授权，设置为 `false` 可启用。
4.  **优先级：** 环境变量、命令行参数和配置文件之间的优先级遵循以下 TorchServe 标准（命令行 > 环境变量 > 配置文件，但具体行为见下例）：

**示例 1：**

*   配置文件：`disable_token_authorization=false` （启用）
*   命令行：`torchserve --start --ncs --model-store model_store --disable-token-auth` （禁用）
*   **结果：** 令牌授权被禁用。尽管配置文件尝试启用它，但命令行具有更高的优先级并禁用了它。

**示例 2：**

*   配置文件：`disable_token_authorization=true` （禁用）
*   命令行：`torchserve --start --ncs --model-store model_store` （未配置）
*   **结果：** 令牌授权被禁用。配置文件禁用了它，并且命令行没有覆盖该设置。

**配置**

*   TorchServe 将默认启用令牌授权。预期的日志语句为：`main org.pytorch.serve.http.TokenAuthorizationHandler - Token Authorization Enabled`
*   在当前工作目录下将生成一个名为 `key_file.json` 的文件。
*   **示例密钥文件 (`key_file.json`)：**
    ```json
    {
      "management": {
        "key": "B-E5KSRM",
        "expiration time": "2024-02-16T21:12:24.801167Z"
      },
      "inference": {
        "key": "gNRuA7dS",
        "expiration time": "2024-02-16T21:12:24.801148Z"
      },
      "API": {
        "key": "yv9uQajP"
      }
    }
    ```
*   文件中有 3 个密钥，各有不同用途：
    *   **Management key (管理密钥):** 用于管理 API。
        示例：`curl http://localhost:8081/models/densenet161 -H "Authorization: Bearer I_J_ItMb"`
    *   **Inference key (推理密钥):** 用于推理 API。
        示例：`curl http://127.0.0.1:8080/predictions/densenet161 -T examples/image_classifier/kitten.jpg -H "Authorization: Bearer FINhR1fj"`
    *   **API key (API 密钥):** 用于令牌授权 API 本身，以生成新的管理或推理密钥（见下文第 4 节 API 用途）。

**API 用途：生成新密钥**

可以使用 API 密钥调用特定端点来生成新的管理密钥或推理密钥，以替换当前的密钥。

*   **管理密钥示例：**
    `curl localhost:8081/token?type=management -H "Authorization: Bearer m4M-5IBY"`
    此命令将使用新的密钥替换 `key_file.json` 中的当前管理密钥，并更新其过期时间。
*   **推理密钥示例：**
    `curl localhost:8081/token?type=inference -H "Authorization: Bearer m4M-5IBY"`
    用户将需要使用上述 API 中的一个来更新密钥。

当用户关闭服务器时，`key_file.json` 文件将被删除。

**注意**

*   **请勿修改密钥文件 (`key_file.json`)。** 修改该文件可能会影响文件的读写，从而阻止新密钥在文件中正确显示。
*   **过期时间：** 默认设置为 60 分钟，但可以在 `config.properties` 文件中通过添加 `token_expiration_min` 来更改。例如：`token_expiration_min=30`。
*   **灵活性：** 提供三个不同的令牌是为了让所有者在使用上拥有最大的灵活性，并使他们能够根据自己的用途调整令牌。服务器所有者可以向只需要对已加载模型运行推理的用户提供推理令牌。如果所有者希望用户能够添加和删除模型，则可以向他们提供管理密钥。
