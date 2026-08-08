# LLM Reverse Deployer

> **把本地 vLLM 变成一个公网 OpenAI-compatible API。**

将部署在局域网 / 内网 GPU 服务器上的 LLM 推理服务（vLLM），通过 Worker 主动建立的反向连接映射到公网 Gateway，使公网用户能够通过标准 OpenAI-compatible API 访问内网 LLM。

三个组件：

- **Gateway** — 运行在公网 VPS，对外提供 OpenAI API，负责模型路由与流式转发。
- **Worker** — 运行在内网 GPU 服务器，主动拨出 WebSocket 反向隧道，连接本地 vLLM。
- **Backend** — Worker 侧的抽象层，v0.1 实现 `VLLMBackend`。

> **端口约定**：开发阶段不使用 80 / 443 等标准端口。Gateway 对外提供 LLM API 服务的端口为自定义的 **17171**（可用 `--port` 覆盖）。

---

## 1. 安装

```bash
# 从源码安装（当前未发布到 PyPI，本地部署用）
pip install -e .

# 发布后可直接：
# pip install llm-reverse-deployer
```

提供两个命令：`llm-gateway`、`llm-worker`。

---

## 2. Gateway（公网 VPS）

配置环境变量：

```bash
export LLM_GATEWAY_API_KEY="your-api-key"     # 公网 API Key，Client → Gateway
export LLM_WORKER_TOKEN="your-worker-token"   # Worker 认证 Token，Worker → Gateway
```

启动：

```bash
llm-gateway start \
    --host 0.0.0.0 \
    --port 17171
```

建议在前面加一层 TLS（Nginx / Caddy / Traefik），将 `https://your-domain.com:17171/` 反代到 `127.0.0.1:17171`。

---

## 3. vLLM（内网 GPU 服务器）

```bash
vllm serve Qwen/Qwen3-32B \
    --host 127.0.0.1 \
    --port 8000
```

---

## 4. Worker（内网 GPU 服务器）

配置环境变量：

```bash
export LLM_GATEWAY_URL="wss://your-domain.com:17171/worker/connect"
export LLM_WORKER_TOKEN="your-worker-token"   # 与 Gateway 配置一致
export LLM_WORKER_ID="gpu-01"
export VLLM_BASE_URL="http://127.0.0.1:8000"  # 本地 vLLM 地址，公网不可见
# export VLLM_API_KEY=""                       # 可选，vLLM 自身的 API Key
```

启动：

```bash
llm-worker start \
    --gateway wss://your-domain.com:17171/worker/connect \
    --token your-worker-token \
    --worker-id gpu-01 \
    --backend vllm \
    --vllm http://127.0.0.1:8000
```

Worker 启动后：连接 Gateway → 检查 Backend 健康 → 发现模型 → 注册 → 心跳 → 等待请求。连接断开会自动按 `1s → 2s → 4s → … → 30s` 退避重连。

---

## 5. 验证

### 列出模型

```bash
curl https://your-domain.com:17171/v1/models \
    -H "Authorization: Bearer your-api-key"
```

### Streaming 请求

```bash
curl https://your-domain.com:17171/v1/chat/completions \
    -H "Authorization: Bearer your-api-key" \
    -H "Content-Type: application/json" \
    -d '{
      "model": "Qwen/Qwen3-32B",
      "messages": [
        {
          "role": "user",
          "content": "Hello"
        }
      ],
      "stream": true
    }'
```

### Python（OpenAI SDK）

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://your-domain.com:17171/v1",
    api_key="your-api-key",
)

stream = client.chat.completions.create(
    model="Qwen/Qwen3-32B",
    messages=[{"role": "user", "content": "你好"}],
    stream=True,
)

for chunk in stream:
    content = chunk.choices[0].delta.content
    if content:
        print(content, end="")
```

---

## 环境变量参考

| 组件 | 环境变量 | 说明 |
|------|----------|------|
| Gateway | `LLM_GATEWAY_API_KEY` | 公网 API Key，Client → Gateway |
| Gateway | `LLM_WORKER_TOKEN` | Worker 认证 Token，Worker → Gateway |
| Worker | `LLM_GATEWAY_URL` | Gateway 地址（`wss://…:17171/worker/connect`） |
| Worker | `LLM_WORKER_TOKEN` | Worker 认证 Token，需与 Gateway 一致 |
| Worker | `LLM_WORKER_ID` | Worker 唯一标识 |
| Worker | `VLLM_BASE_URL` | 本地 vLLM 地址（Worker 本地配置，公网不可见） |
| Worker | `VLLM_API_KEY` | 可选，vLLM 自身的 API Key |

## 安全说明

- 两个 Token 相互独立，不可混用；均不会写入日志。
- Client 无法通过 API 指定后端地址，杜绝 SSRF。
- Gateway 从不主动访问内网，仅由 Worker 拨出反向隧道。
