"""VLLMBackend tests with a mocked HTTP layer (httpx.MockTransport).

Covers list_models(), health(), chat_completions() for both streaming and
non-streaming requests.
"""

from __future__ import annotations

import json

import httpx
import pytest

from llm_reverse_deployer.backend.base import ChatCompletionRequest
from llm_reverse_deployer.backend.vllm import VLLMBackend

MODELS_RESPONSE = {"object": "list", "data": [{"id": "Qwen/Qwen3-32B", "object": "model"}]}
COMPLETION_RESPONSE = {
    "id": "cmpl-test",
    "object": "chat.completion",
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
}
def _sse(text: str) -> bytes:
    return f'data: {{"choices":[{{"delta":{{"content":"{text}"}}}}]}}\n\n'.encode("utf-8")


SSE_CHUNKS = [
    _sse("你"),
    _sse("好"),
    b"data: [DONE]\n\n",
]


class _ChunkStream(httpx.AsyncByteStream):
    """Splits a response into the same byte chunks a real SSE server sends."""

    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks

    async def __aiter__(self):
        for chunk in self._chunks:
            yield chunk

    async def aclose(self) -> None:
        pass


def _make_backend() -> tuple[VLLMBackend, list[httpx.Request]]:
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        if request.url.path == "/v1/models":
            return httpx.Response(200, json=MODELS_RESPONSE)
        if request.url.path == "/v1/chat/completions":
            body = json.loads(request.content)
            if body.get("stream"):
                return httpx.Response(200, stream=_ChunkStream(SSE_CHUNKS))
            return httpx.Response(200, json=COMPLETION_RESPONSE)
        return httpx.Response(404)

    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(base_url="http://vllm.local", transport=transport)
    backend = VLLMBackend(base_url="http://vllm.local", client=client)
    return backend, captured


@pytest.mark.asyncio
async def test_health_ok():
    backend, _ = _make_backend()
    assert await backend.health() is True
    await backend.close()


@pytest.mark.asyncio
async def test_health_fails_when_down():
    transport = httpx.MockTransport(lambda req: httpx.Response(500))
    client = httpx.AsyncClient(base_url="http://vllm.local", transport=transport)
    backend = VLLMBackend(base_url="http://vllm.local", client=client)
    assert await backend.health() is False
    await backend.close()


@pytest.mark.asyncio
async def test_list_models():
    backend, captured = _make_backend()
    models = await backend.list_models()
    assert [m.id for m in models] == ["Qwen/Qwen3-32B"]
    assert captured[0].url.path == "/v1/models"
    await backend.close()


@pytest.mark.asyncio
async def test_chat_completions_non_streaming():
    backend, captured = _make_backend()
    request = ChatCompletionRequest(
        model="Qwen/Qwen3-32B",
        messages=[{"role": "user", "content": "hi"}],
        stream=False,
    )
    chunks = [c async for c in backend.chat_completions(request)]
    assert json.loads(b"".join(chunks)) == COMPLETION_RESPONSE
    assert captured[0].url.path == "/v1/chat/completions"
    await backend.close()


@pytest.mark.asyncio
async def test_chat_completions_streaming_passthrough():
    backend, _ = _make_backend()
    request = ChatCompletionRequest(
        model="Qwen/Qwen3-32B",
        messages=[{"role": "user", "content": "hi"}],
        stream=True,
    )
    chunks = [c async for c in backend.chat_completions(request)]
    assert chunks == list(SSE_CHUNKS)
    await backend.close()


@pytest.mark.asyncio
async def test_backend_error_raises():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="vLLM overloaded")

    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(base_url="http://vllm.local", transport=transport)
    backend = VLLMBackend(base_url="http://vllm.local", client=client)
    request = ChatCompletionRequest(
        model="Qwen/Qwen3-32B", messages=[], stream=False
    )
    from llm_reverse_deployer.backend.base import BackendError

    with pytest.raises(BackendError):
        _ = [c async for c in backend.chat_completions(request)]
    await backend.close()
