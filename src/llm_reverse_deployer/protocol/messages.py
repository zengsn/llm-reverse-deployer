"""Tunnel Protocol messages exchanged between Gateway and Worker.

The tunnel carries JSON control messages. Streaming payload bytes are
base64-encoded inside the ``data`` field of :class:`ChunkMessage` so that the
passthrough is byte-exact regardless of how the upstream backend chunks data.

Message types
-------------

``register``    Worker -> Gateway   announce identity + backend + models
``registered``  Gateway -> Worker   acknowledge a registration
``heartbeat``   Worker -> Gateway   keep-alive
``request``     Gateway -> Worker   forward an HTTP request to the backend
``chunk``       Worker -> Gateway   streaming payload bytes
``end``         Worker -> Gateway   request finished successfully
``error``       either             a request failed
``cancel``      Gateway -> Worker   ask the worker to abort an in-flight request
"""

from __future__ import annotations

import base64
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

MESSAGE_TYPES = ("register", "registered", "heartbeat", "request", "chunk", "end", "error", "cancel")


class Message(BaseModel):
    """Base class for all tunnel messages."""

    type: str
    model_config = ConfigDict(extra="forbid")


class RegisterMessage(Message):
    """Worker announces itself to the Gateway."""

    type: Literal["register"] = "register"
    worker_id: str
    backend: str
    models: list[str] = Field(default_factory=list)


class RegisteredMessage(Message):
    """Gateway acknowledges a Worker registration."""

    type: Literal["registered"] = "registered"
    worker_id: str
    models: list[str] = Field(default_factory=list)


class HeartbeatMessage(Message):
    """Worker keep-alive."""

    type: Literal["heartbeat"] = "heartbeat"
    worker_id: str
    timestamp: float


class RequestMessage(Message):
    """Gateway forwards an HTTP request to a Worker."""

    type: Literal["request"] = "request"
    request_id: str
    method: str = "POST"
    path: str
    headers: dict[str, str] = Field(default_factory=dict)
    body: Any = None


class ChunkMessage(Message):
    """Streaming payload bytes (base64-encoded ``data``)."""

    type: Literal["chunk"] = "chunk"
    request_id: str
    data: str

    def payload_bytes(self) -> bytes:
        return base64.b64decode(self.data)

    @staticmethod
    def encode_bytes(request_id: str, data: bytes) -> "ChunkMessage":
        return ChunkMessage(request_id=request_id, data=base64.b64encode(data).decode("ascii"))


class EndMessage(Message):
    """A request completed successfully; no more chunks follow."""

    type: Literal["end"] = "end"
    request_id: str


class ErrorDetail(BaseModel):
    type: str
    message: str


class ErrorMessage(Message):
    """A request failed; carries an OpenAI-style error detail."""

    type: Literal["error"] = "error"
    request_id: str
    error: ErrorDetail


class CancelMessage(Message):
    """Gateway asks the Worker to abort an in-flight request."""

    type: Literal["cancel"] = "cancel"
    request_id: str


def decode_message(raw: dict[str, Any]) -> Message:
    """Decode a raw JSON dict into the matching message type."""
    msg_type = raw.get("type")
    if msg_type == "register":
        return RegisterMessage.model_validate(raw)
    if msg_type == "registered":
        return RegisteredMessage.model_validate(raw)
    if msg_type == "heartbeat":
        return HeartbeatMessage.model_validate(raw)
    if msg_type == "request":
        return RequestMessage.model_validate(raw)
    if msg_type == "chunk":
        return ChunkMessage.model_validate(raw)
    if msg_type == "end":
        return EndMessage.model_validate(raw)
    if msg_type == "error":
        return ErrorMessage.model_validate(raw)
    if msg_type == "cancel":
        return CancelMessage.model_validate(raw)
    raise ValueError(f"Unknown tunnel message type: {msg_type!r}")
