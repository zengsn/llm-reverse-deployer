"""Configuration models.

Configuration is explicit and data-driven: each component receives a config
object. Tokens and API keys live only in these objects (or the environment)
and are never written to logs.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class VLLMConfig(BaseModel):
    base_url: str = "http://127.0.0.1:8000"
    api_key: str = ""


class BackendConfig(BaseModel):
    type: str = "vllm"
    vllm: Optional[VLLMConfig] = None


class WorkerConfig(BaseModel):
    gateway_url: str
    worker_id: str
    worker_token: str
    backend: BackendConfig = Field(default_factory=BackendConfig)
    heartbeat_interval: float = 15.0
    connect_timeout: float = 10.0


class GatewayConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 17171
    api_key: str = ""
    worker_token: str = ""
    request_body_limit: int = 1_000_000
    request_timeout: float = 300.0
    stream_idle_timeout: float = 120.0
    heartbeat_timeout: float = 90.0
