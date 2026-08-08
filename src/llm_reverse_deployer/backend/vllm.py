"""vLLM backend.

Talks to a local vLLM OpenAI-compatible API:

    GET  /v1/models
    POST /v1/chat/completions
"""

from __future__ import annotations

import logging
from typing import AsyncIterator, Dict, List, Optional

import httpx

from .base import Backend, BackendError, ChatCompletionRequest, ModelInfo

logger = logging.getLogger(__name__)

_VLLM_TIMEOUT = httpx.Timeout(connect=10.0, read=120.0, write=30.0, pool=10.0)


class VLLMBackend(Backend):
    name = "vllm"

    def __init__(
        self,
        base_url: str,
        api_key: str = "",
        client: Optional[httpx.AsyncClient] = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._client = client or httpx.AsyncClient(
            base_url=self._base_url,
            timeout=_VLLM_TIMEOUT,
        )

    @property
    def endpoint(self) -> str:
        """The vLLM base URL (local configuration, never sent to the Gateway)."""
        return self._base_url

    async def close(self) -> None:
        await self._client.aclose()

    def _headers(self, content_type: bool = False) -> Dict[str, str]:
        headers: Dict[str, str] = {}
        if content_type:
            headers["Content-Type"] = "application/json"
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    async def health(self) -> bool:
        try:
            resp = await self._client.get("/v1/models", headers=self._headers())
            return resp.status_code == 200
        except httpx.HTTPError:
            return False

    async def list_models(self) -> List[ModelInfo]:
        resp = await self._client.get("/v1/models", headers=self._headers())
        if resp.status_code != 200:
            raise BackendError(f"vLLM /v1/models returned HTTP {resp.status_code}")
        data = resp.json().get("data", [])
        return [ModelInfo(id=item["id"], owned_by="vllm") for item in data]

    async def chat_completions(self, request: ChatCompletionRequest) -> AsyncIterator[bytes]:
        payload = request.model_dump()
        url = "/v1/chat/completions"
        if request.stream:
            async with self._client.stream(
                "POST", url, json=payload, headers=self._headers(content_type=True)
            ) as resp:
                if resp.status_code >= 400:
                    body = (await resp.aread()).decode("utf-8", errors="replace")
                    raise BackendError(f"vLLM returned HTTP {resp.status_code}: {body}")
                async for chunk in resp.aiter_bytes():
                    yield chunk
        else:
            resp = await self._client.post(
                url, json=payload, headers=self._headers(content_type=True)
            )
            if resp.status_code >= 400:
                raise BackendError(f"vLLM returned HTTP {resp.status_code}: {resp.text}")
            yield resp.content
