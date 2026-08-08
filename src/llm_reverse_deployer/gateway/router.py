"""Gateway request routing.

Routes OpenAI-compatible chat completions to the Worker that serves the
requested model, then pipes the backend stream back to the client. The Gateway
never talks to a concrete backend; it only knows Worker -> Model -> Request.
"""

from __future__ import annotations

import asyncio
import json
import logging
import secrets
import time
from typing import Any, Dict, List, Optional

from fastapi import Response
from fastapi.responses import JSONResponse, StreamingResponse

from ..config import GatewayConfig
from ..protocol import (
    ChunkMessage,
    EndMessage,
    ErrorDetail,
    ErrorMessage,
    RequestMessage,
)
from .errors import GatewayError, status_for_error
from .registry import WorkerInfo, WorkerRegistry

logger = logging.getLogger(__name__)


class PendingRequest:
    """Bookkeeping for one in-flight tunneled request."""

    def __init__(self, request_id: str, worker_id: str) -> None:
        self.request_id = request_id
        self.worker_id = worker_id
        self.queue: asyncio.Queue[Any] = asyncio.Queue()
        self.finished = False


def _sse_error(error: ErrorDetail) -> bytes:
    payload = json.dumps({"error": {"type": error.type, "message": error.message}})
    return f"data: {payload}\n\n".encode("utf-8")


class RequestRouter:
    """Routes chat-completion requests to workers and back."""

    def __init__(self, config: GatewayConfig, registry: WorkerRegistry) -> None:
        self.config = config
        self.registry = registry
        self._pending: Dict[str, PendingRequest] = {}
        self._lock = asyncio.Lock()

    async def send_request(
        self,
        worker: WorkerInfo,
        request_id: str,
        method: str,
        path: str,
        body: Any,
    ) -> PendingRequest:
        pr = PendingRequest(request_id=request_id, worker_id=worker.worker_id)
        async with self._lock:
            self._pending[request_id] = pr
        message = RequestMessage(
            request_id=request_id, method=method, path=path, body=body
        )
        try:
            await worker.connection.send_json(message.model_dump())
        except Exception:
            await self.release(request_id)
            raise
        return pr

    async def push(self, message: Any) -> None:
        """Deliver a chunk/end/error message to the waiting request handler."""
        request_id = getattr(message, "request_id", None)
        pr = self._pending.get(request_id)
        if pr is None or pr.finished:
            return
        if isinstance(message, (EndMessage, ErrorMessage)):
            pr.finished = True
        await pr.queue.put(message)

    async def release(self, request_id: str) -> None:
        async with self._lock:
            pr = self._pending.pop(request_id, None)
            if pr is not None:
                pr.finished = True

    async def fail_requests_for_worker(
        self, worker_id: str, message: str = "Worker disconnected"
    ) -> None:
        """Fail every in-flight request routed to a disconnected Worker."""
        to_fail = [
            pr for pr in self._pending.values() if pr.worker_id == worker_id
        ]
        for pr in to_fail:
            pr.finished = True
            await pr.queue.put(
                ErrorMessage(
                    request_id=pr.request_id,
                    error=ErrorDetail(type="worker_unavailable", message=message),
                )
            )

    async def _cancel(self, worker: WorkerInfo, request_id: str) -> None:
        try:
            await worker.connection.send_json(
                {"type": "cancel", "request_id": request_id}
            )
        except Exception:
            logger.warning("Failed to send cancel for %s", request_id)

    async def _iter_stream(
        self,
        pr: PendingRequest,
        worker: WorkerInfo,
        request_id: str,
    ) -> Any:
        try:
            while True:
                message = await asyncio.wait_for(
                    pr.queue.get(), timeout=self.config.stream_idle_timeout
                )
                if isinstance(message, ChunkMessage):
                    yield message.payload_bytes()
                elif isinstance(message, EndMessage):
                    break
                elif isinstance(message, ErrorMessage):
                    yield _sse_error(message.error)
                    break
        except asyncio.TimeoutError:
            logger.warning("Stream idle timeout: %s", request_id)
            await self._cancel(worker, request_id)
        finally:
            await self.release(request_id)
            if not pr.finished:
                await self._cancel(worker, request_id)

    async def _collect(
        self,
        pr: PendingRequest,
        worker: WorkerInfo,
        request_id: str,
    ) -> "tuple[bytes, Optional[ErrorDetail]]":
        buf = bytearray()
        deadline = time.monotonic() + self.config.request_timeout
        try:
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise asyncio.TimeoutError
                message = await asyncio.wait_for(
                    pr.queue.get(), timeout=remaining
                )
                if isinstance(message, ChunkMessage):
                    buf.extend(message.payload_bytes())
                elif isinstance(message, EndMessage):
                    return bytes(buf), None
                elif isinstance(message, ErrorMessage):
                    return bytes(buf), message.error
        except asyncio.TimeoutError:
            logger.warning("Request timeout: %s", request_id)
            await self._cancel(worker, request_id)
            raise GatewayError(
                504, "gateway_timeout", "Worker did not respond in time"
            )
        finally:
            await self.release(request_id)
            if not pr.finished:
                await self._cancel(worker, request_id)

    async def handle_chat_completion(self, body: Dict[str, Any]) -> Response:
        model = body.get("model")
        if not isinstance(model, str) or not model:
            raise GatewayError(400, "invalid_request", "Missing required field: model")

        worker = self.registry.find_worker_for_model(model)
        if worker is None:
            if self.registry.has_model(model):
                raise GatewayError(
                    503, "worker_unavailable", f"No available worker for model: {model}"
                )
            raise GatewayError(404, "model_not_found", f"Model not available: {model}")

        request_id = f"req_{secrets.token_hex(6)}"
        logger.info("Request received: %s model=%s", request_id, model)
        logger.info("Routing request: %s -> %s", request_id, worker.worker_id)

        try:
            pr = await self.send_request(
                worker, request_id, "POST", "/v1/chat/completions", body
            )
        except Exception:
            raise GatewayError(
                502, "backend_error", "Failed to reach worker: " + worker.worker_id
            )

        if body.get("stream", False):
            return StreamingResponse(
                self._iter_stream(pr, worker, request_id),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                },
            )

        buf, error = await self._collect(pr, worker, request_id)
        if error is not None:
            raise GatewayError(status_for_error(error.type), error.type, error.message)
        try:
            content = json.loads(buf.decode("utf-8") or "{}")
        except json.JSONDecodeError:
            raise GatewayError(502, "backend_error", "Backend returned invalid JSON")
        return JSONResponse(content=content)
