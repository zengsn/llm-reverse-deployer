"""End-to-end streaming integration test.

Spins up a fake vLLM, a real Gateway (uvicorn) and a real Worker, then drives
the full chain exactly as a public client would:

    Client -> Gateway -> Worker -> VLLMBackend -> Fake vLLM

Verifies chunk-by-chunk streaming, non-streaming completion, auth, and error
paths.
"""

from __future__ import annotations

import asyncio
import json
import socket
import threading
import time
from typing import Iterator

import httpx
import pytest
import uvicorn

from llm_reverse_deployer.backend.factory import create_backend
from llm_reverse_deployer.config import (
    BackendConfig,
    GatewayConfig,
    VLLMConfig,
    WorkerConfig,
)
from llm_reverse_deployer.gateway.app import create_app
from llm_reverse_deployer.logging import setup_logging
from llm_reverse_deployer.worker.manager import WorkerManager

MODEL = "Qwen/Qwen3-32B"
API_KEY = "test-api-key"
WORKER_TOKEN = "test-worker-token"
PIECES = ["你", "好", "，", "世界"]


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _make_fake_vllm_app():
    from fastapi import FastAPI
    from fastapi.responses import StreamingResponse

    app = FastAPI()

    @app.get("/v1/models")
    async def models():
        return {"object": "list", "data": [{"id": MODEL, "object": "model"}]}

    @app.post("/v1/chat/completions")
    async def chat(body: dict):
        if body.get("stream"):

            async def gen():
                for piece in PIECES:
                    payload = {
                        "id": "chatcmpl-test",
                        "object": "chat.completion.chunk",
                        "model": MODEL,
                        "choices": [{"index": 0, "delta": {"content": piece}, "finish_reason": None}],
                    }
                    yield f"data: {json.dumps(payload)}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(gen(), media_type="text/event-stream")
        return {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "model": MODEL,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "".join(PIECES)}, "finish_reason": "stop"}],
        }

    return app


def _run_uvicorn(app, port: int) -> uvicorn.Server:
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
    server = uvicorn.Server(config)
    threading.Thread(target=server.run, daemon=True).start()
    return server


def _wait_http(port: int, path: str = "/health", timeout: float = 15.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            resp = httpx.get(f"http://127.0.0.1:{port}{path}")
            if resp.status_code < 500:
                return
        except httpx.HTTPError:
            pass
        time.sleep(0.05)
    raise TimeoutError(f"HTTP server on {port}{path} did not become ready")


@pytest.fixture(scope="module")
def services() -> Iterator[dict]:
    setup_logging("WARNING")
    vllm_port = _free_port()
    gateway_port = _free_port()

    vllm_server = _run_uvicorn(_make_fake_vllm_app(), vllm_port)
    _wait_http(vllm_port, "/v1/models")

    gateway_config = GatewayConfig(
        host="127.0.0.1",
        port=gateway_port,
        api_key=API_KEY,
        worker_token=WORKER_TOKEN,
    )
    gateway_server = _run_uvicorn(create_app(gateway_config), gateway_port)
    _wait_http(gateway_port, "/health")

    worker_config = WorkerConfig(
        gateway_url=f"ws://127.0.0.1:{gateway_port}/worker/connect",
        worker_id="gpu-01",
        worker_token=WORKER_TOKEN,
        backend=BackendConfig(type="vllm", vllm=VLLMConfig(base_url=f"http://127.0.0.1:{vllm_port}")),
    )
    manager = WorkerManager(worker_config, create_backend(worker_config.backend))

    def _run_worker() -> None:
        asyncio.run(manager.run())

    worker_thread = threading.Thread(target=_run_worker, daemon=True)
    worker_thread.start()

    env = {
        "gateway_port": gateway_port,
        "vllm_port": vllm_port,
        "manager": manager,
        "worker_thread": worker_thread,
    }

    # Wait until the Worker registers the model on the Gateway.
    deadline = time.time() + 15.0
    while time.time() < deadline:
        resp = httpx.get(
            f"http://127.0.0.1:{gateway_port}/v1/models",
            headers={"Authorization": f"Bearer {API_KEY}"},
        )
        if resp.status_code == 200 and any(m["id"] == MODEL for m in resp.json()["data"]):
            break
        time.sleep(0.1)
    else:
        manager.stop()
        raise RuntimeError("Worker never registered its model on the Gateway")

    try:
        yield env
    finally:
        manager.stop()
        worker_thread.join(timeout=5)
        gateway_server.should_exit = True
        vllm_server.should_exit = True


@pytest.fixture()
def base_url(services: dict) -> str:
    return f"http://127.0.0.1:{services['gateway_port']}"


def _auth_headers() -> dict:
    return {"Authorization": f"Bearer {API_KEY}"}


def test_health_is_open(base_url: str):
    resp = httpx.get(f"{base_url}/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_models_require_auth(base_url: str):
    resp = httpx.get(f"{base_url}/v1/models")
    assert resp.status_code == 401


def test_models_after_registration(base_url: str):
    resp = httpx.get(f"{base_url}/v1/models", headers=_auth_headers())
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert {"id": MODEL, "object": "model", "owned_by": "vllm"} in data


def test_streaming_chat_full_chain(base_url: str):
    with httpx.Client(timeout=30) as client:
        with client.stream(
            "POST",
            f"{base_url}/v1/chat/completions",
            headers={**_auth_headers(), "Content-Type": "application/json"},
            json={
                "model": MODEL,
                "messages": [{"role": "user", "content": "你好"}],
                "stream": True,
            },
        ) as resp:
            assert resp.status_code == 200
            assert resp.headers["content-type"].startswith("text/event-stream")
            pieces: list[str] = []
            saw_done = False
            for line in resp.iter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[len("data: "):]
                if data == "[DONE]":
                    saw_done = True
                    break
                chunk = json.loads(data)
                content = chunk["choices"][0]["delta"].get("content")
                if content:
                    pieces.append(content)
    assert pieces == PIECES
    assert saw_done


def test_non_streaming_chat_full_chain(base_url: str):
    resp = httpx.post(
        f"{base_url}/v1/chat/completions",
        headers={**_auth_headers(), "Content-Type": "application/json"},
        json={
            "model": MODEL,
            "messages": [{"role": "user", "content": "你好"}],
            "stream": False,
        },
        timeout=30,
    )
    assert resp.status_code == 200
    assert resp.json()["choices"][0]["message"]["content"] == "".join(PIECES)


def test_model_not_found(base_url: str):
    resp = httpx.post(
        f"{base_url}/v1/chat/completions",
        headers={**_auth_headers(), "Content-Type": "application/json"},
        json={"model": "does/not-exist", "messages": [], "stream": False},
    )
    assert resp.status_code == 404
    assert resp.json()["error"]["type"] == "model_not_found"


def test_invalid_api_key_rejected(base_url: str):
    resp = httpx.get(
        f"{base_url}/v1/models", headers={"Authorization": "Bearer wrong-key"}
    )
    assert resp.status_code == 401


def test_missing_model_field(base_url: str):
    resp = httpx.post(
        f"{base_url}/v1/chat/completions",
        headers={**_auth_headers(), "Content-Type": "application/json"},
        json={"messages": []},
    )
    assert resp.status_code == 400
    assert resp.json()["error"]["type"] == "invalid_request"


def test_invalid_json_body(base_url: str):
    resp = httpx.post(
        f"{base_url}/v1/chat/completions",
        headers={**_auth_headers(), "Content-Type": "application/json"},
        content=b"{not json",
    )
    assert resp.status_code == 400
