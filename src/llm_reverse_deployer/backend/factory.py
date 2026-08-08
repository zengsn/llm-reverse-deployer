"""Backend factory.

Kept deliberately simple: v0.1 only knows ``vllm``. Future backends (Ollama,
SGLang, ...) extend this factory with a single ``if`` branch — no plugin
registry, no entry points, no DI framework.
"""

from __future__ import annotations

from ..config import BackendConfig, VLLMConfig
from .base import Backend
from .vllm import VLLMBackend


def create_backend(config: BackendConfig) -> Backend:
    if config.type == "vllm":
        vllm = config.vllm or VLLMConfig(base_url="http://127.0.0.1:8000")
        return VLLMBackend(base_url=vllm.base_url, api_key=vllm.api_key)
    raise ValueError(f"Unknown backend type: {config.type!r}")
