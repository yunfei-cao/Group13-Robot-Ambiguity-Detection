# 模型客户端：离线开发与后续 Qwen 配置

本模块仅使用 Python 标准库（建议 Python 3.10+），无需安装依赖。
不负责数据读取、提示词构造、实验调度或评估，不修改其他组员模块。

## Mock 使用

从项目根目录运行：

```python
from src.model_client import ModelClient, ModelConfig

client = ModelClient(ModelConfig.from_file("configs/model.mock.json"))
result = client.classify("在这里传入提示词模块构建的完整 prompt")
print(result)
```

默认 `ModelClient()` 也使用 Mock，无需密钥，不创建网络传输。
`mock_response` 是可配置的固定响应，可设为 `CLEAR`、
`NEEDS_CLARIFICATION` 或任意非法输出用于测试；不是实际歧义判断，不能作为实验结果。
## 输入与响应接口

`classify(prompt)` 接受字符串或标准 OpenAI messages 列表。字符串转换为单条
user 消息；列表保持角色、顺序、content 和额外字段，不修改调用方数据。例如：

```python
from dataclasses import asdict

result = client.classify([
    {"role": "system", "content": "只输出 CLEAR 或 NEEDS_CLARIFICATION"},
    {"role": "user", "content": "拿起那个杯子"},
])
record = asdict(result)  # 可序列化为 JSON；error 也转换为字典
if result.status == "error":
    print(result.error.code, result.error.http_status, result.error.retryable)
```

支持 system/developer/user/assistant/tool/function 角色及结构化 content parts，
保留 assistant 工具调用字段；具体模型是否支持这些消息类型由服务端决定。
空列表、无效角色、缺失 content 等本地输入错误抛出 `ValueError`，不发送请求。

| 字段 | 类型与含义 |
| --- | --- |
| raw_response | `str`，assistant 原始输出文本，不是完整 HTTP JSON；无输出时为空字符串 |
| status | `success`（有效标签）、`invalid_output`（非法标签）、`error`（请求/响应错误） |
| latency_ms | `float`，通过两次 `time.perf_counter()` 差值乘 1000；包含请求、读取和解析，Mock 与失败路径也计时，不含输入校验 |
| token_usage | 字典，包含 `prompt_tokens`、`completion_tokens`、`total_tokens`；读取 API usage，缺失或非法计数为 `None`，不自行推算 total；API 明确返回的 0 保留 |
| error | 成功或非法标签时为 `None`；错误时为 `ModelError(code, message, retryable, http_status)` |
| model_id | 优先记录 API 返回的 model，否则使用配置模型 ID；Mock 为 `mock` |
| label / is_valid | 严格解析的标签（或 `None`）及其有效性 |
| raw_text / model / mode | 兼容旧字段；前两项分别等于 raw_response/model_id，mode 为 mock/qwen |

解析仅去除首尾空白，严格接受两类标签。非法输出保留原文，`label=None`，
`is_valid=False`，方便统计有效输出率。Mock 的三个 Token 计数均为 `None`，
耗时随实际执行变化；Mock 固定的是输出内容，不是整个响应对象。

## 错误分类与调度边界

| 场景 | error.code | retryable |
| --- | --- | --- |
| HTTP 401 | authentication_error | false |
| HTTP 403 | permission_denied | false |
| HTTP 429 | rate_limit | true |
| HTTP 5xx | server_error | true |
| HTTP 408 / 网络超时（含 URLError 包装的 TimeoutError） | timeout | true |
| 其他网络/OSError | network_error | true |
| 其他 HTTP 状态（含禁止的重定向） | http_error | false |
| 非 JSON / 响应结构错误 | invalid_response | false |
| finish_reason 非 stop | incomplete_response | false |

HTTP 错误包含数值 `http_status`，其他错误为 `None`。异常信息使用固定文本，
不回显服务端错误正文、密钥或底层异常原文。响应截断时保留已取得的文本和 usage，
但不判为有效标签。`invalid_output` 是解析结果，不属于传输错误，error 为 None。

**本版本将原先的传输异常改为返回 `status="error"` 的 ModelResponse**；
调用方应检查 status/error，而非依赖捕获 ModelClientError。配置和输入错误仍抛出
`ValueError`（配置文件 IO/JSON 错误按 Python 原异常传播）。
`retryable` 仅提示后续 run_experiment.py 是否可安排重试，本模块每次调用最多发出
一次请求，不实现重试、退避或实验调度。重试次数与最终失败记录由后续调度器负责。

## 执行手册核对范围

本次依据用户提供的第 3/5 页相关要求实现六个新增字段、messages、耗时、usage
和错误分类。当前仓库未提供 Group13 执行手册原文，无法宣称已逐页核对；
以上 status 枚举、error 结构及 raw_response 文本定义仍须与手册原文复核。

## 离线测试

```powershell
python -B -m unittest discover -s tests -v
```

测试替换 Qwen HTTP 传输，并阻断 socket 连接；不使用真实密钥、真实 API 或付费请求。

## 下一阶段：真实 Qwen API

1. 在阿里云 Model Studio 控制台确认所在地域、工作空间、API Key 和可用的
   **固定版本 Qwen Plus 模型 ID**。不要直接用滚动别名代替论文实验的固定版本。
2. 以 `configs/model.qwen.example.json` 为模板准备配置，填写 `model` 与
   `base_url`（OpenAI 兼容接口的基础 URL，以 `/compatible-mode/v1` 结尾，
   不包含 `/chat/completions`）。具体地址从当前控制台/官方文档复制，须与密钥地域对应。
3. 仅在准备实际调用时，将 `allow_real_requests` 改为 JSON 布尔值 `true`。
   模板默认 `false` 且地址、模型为空，不能直接发出请求。
4. 在当前 PowerShell 会话设置 `$env:DASHSCOPE_API_KEY = "你的密钥"`。
   不把真实密钥写进 JSON、源代码或提交文件。本模块不自动加载 `.env`。
5. 使用 `ModelConfig.from_file(...)` 创建客户端；随后执行 `classify(prompt)`
   才会发送实际请求。导入模块和构造客户端均不会联网。

请求固定 `enable_thinking=false`、`stream=false`；默认温度为 0，最大输出 64 tokens，
超时 30 秒。未正常结束的 completion 视为错误。禁止自动 HTTP 重定向。
Qwen 分支仅完成离线协议测试，真实服务的模型权限和兼容性仍需下一阶段验证。
建议下一阶段先做单样本联通检查，再接入实验调度；本阶段没有运行 600 次实验。

官方参考（实际地址和模型可用性以控制台为准）：
- https://www.alibabacloud.com/help/en/model-studio/deep-thinking
- https://www.alibabacloud.com/help/en/model-studio/get-api-key
