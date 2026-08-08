"""Backend abstraction.

A Backend talks to a *local* LLM inference server. The Worker only ever calls
this interface; the Gateway knows nothing about concrete backends (vLLM,
Ollama, SGLang, ...). Adding a new backend in the future must not change the
Tunnel Protocol.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Dict, List

from pydantic import BaseModel, ConfigDict, Field


class ModelInfo(BaseModel):
    """A model served by a backend."""

    id: str
    owned_by: str = "unknown"


class ChatCompletionRequest(BaseModel):
    """An OpenAI-style chat completion request.

    ``extra="allow"`` keeps unknown fields (temperature, max_tokens, tools,
    ...) intact so the request body can be passed through to the backend
    unmodified.
    """

    model: str
    messages: List[Dict[str, Any]] = Field(default_factory=list)
    stream: bool = False
    model_config = ConfigDict(extra="allow")


class BackendError(RuntimeError):
    """Raised when a backend request fails in a way the tunnel can report."""


class Backend(ABC):
    """Interface implemented by concrete LLM backends."""

    #: Human-readable backend name, e.g. ``"vllm"``.
    name: str = "backend"

    @abstractmethod
    async def list_models(self) -> List[ModelInfo]:
        """Return the models the backend currently serves."""

    @abstractmethod
    async def chat_completions(self, request: ChatCompletionRequest) -> AsyncIterator[bytes]:
        """Run a chat completion.

        Yields raw response bytes. For streaming requests these are SSE bytes
        as received from the inference server; for non-streaming requests the
        full JSON response body is yielded as a single item. The caller never
        has to parse or reconstruct the payload.
        """

    @abstractmethod
    async def health(self) -> bool:
        """Return whether the backend is reachable and healthy."""
