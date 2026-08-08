"""Tunnel Protocol message tests: register / registered / heartbeat / request /
chunk / end / error / cancel."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from llm_reverse_deployer.protocol import decode_message
from llm_reverse_deployer.protocol.messages import (
    CancelMessage,
    ChunkMessage,
    EndMessage,
    ErrorMessage,
    HeartbeatMessage,
    RegisteredMessage,
    RegisterMessage,
    RequestMessage,
)


def test_register_roundtrip():
    raw = {
        "type": "register",
        "worker_id": "gpu-01",
        "backend": "vllm",
        "models": ["Qwen/Qwen3-32B"],
    }
    msg = decode_message(raw)
    assert isinstance(msg, RegisterMessage)
    assert msg.worker_id == "gpu-01"
    assert msg.backend == "vllm"
    assert msg.models == ["Qwen/Qwen3-32B"]
    assert msg.model_dump() == raw


def test_registered_roundtrip():
    raw = {"type": "registered", "worker_id": "gpu-01", "models": ["Qwen/Qwen3-32B"]}
    assert isinstance(decode_message(raw), RegisteredMessage)


def test_heartbeat_roundtrip():
    raw = {"type": "heartbeat", "worker_id": "gpu-01", "timestamp": 1234567890.0}
    msg = decode_message(raw)
    assert isinstance(msg, HeartbeatMessage)
    assert msg.timestamp == 1234567890.0


def test_request_roundtrip():
    raw = {
        "type": "request",
        "request_id": "req_123",
        "method": "POST",
        "path": "/v1/chat/completions",
        "headers": {},
        "body": {"model": "Qwen/Qwen3-32B", "messages": [], "stream": True},
    }
    msg = decode_message(raw)
    assert isinstance(msg, RequestMessage)
    assert msg.request_id == "req_123"
    assert msg.body["model"] == "Qwen/Qwen3-32B"


def test_chunk_roundtrip_bytes_exact():
    """Base64 payload must survive a round trip byte-for-byte."""
    original = b"data: {\"content\": \"hi\"}\n\n"
    encoded = ChunkMessage.encode_bytes("req_123", original)
    assert isinstance(encoded, ChunkMessage)
    decoded = decode_message(encoded.model_dump())
    assert isinstance(decoded, ChunkMessage)
    assert decoded.payload_bytes() == original


def test_chunk_base64_splits_safely():
    """Non-UTF-8 byte boundaries must not corrupt payloads."""
    original = b"\x00\xff\x80 data\n\n"
    encoded = ChunkMessage.encode_bytes("req_1", original)
    assert decode_message(encoded.model_dump()).payload_bytes() == original


def test_end_roundtrip():
    assert isinstance(decode_message({"type": "end", "request_id": "req_123"}), EndMessage)


def test_error_roundtrip():
    raw = {
        "type": "error",
        "request_id": "req_123",
        "error": {"type": "backend_error", "message": "vLLM backend unavailable"},
    }
    msg = decode_message(raw)
    assert isinstance(msg, ErrorMessage)
    assert msg.error.type == "backend_error"
    assert msg.error.message == "vLLM backend unavailable"


def test_cancel_roundtrip():
    assert isinstance(decode_message({"type": "cancel", "request_id": "req_123"}), CancelMessage)


def test_unknown_type_rejected():
    with pytest.raises(ValueError):
        decode_message({"type": "nope"})


def test_missing_required_field_rejected():
    with pytest.raises(ValidationError):
        decode_message({"type": "register", "backend": "vllm"})
