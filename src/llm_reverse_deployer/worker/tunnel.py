"""Worker tunnel client.

Connects to the Gateway over WebSocket (``wss://<gateway>/worker/connect``),
registers with real backend model discovery, keeps the connection alive with
heartbeats, and serves forwarded requests through the Backend interface.
Auto-reconnects with exponential backoff.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Dict, Optional

import websockets

from ..backend.base import Backend, ChatCompletionRequest
from ..config import WorkerConfig
from ..protocol import (
    CancelMessage,
    ChunkMessage,
    EndMessage,
    ErrorDetail,
    ErrorMessage,
    HeartbeatMessage,
    RegisterMessage,
    RequestMessage,
    decode_message,
)

logger = logging.getLogger(__name__)

#: Reconnect backoff (seconds); 30s is the cap.
_BACKOFF = (1, 2, 4, 8, 16, 30)


class TunnelClient:
    def __init__(
        self,
        config: WorkerConfig,
        backend: Backend,
        stop_event: Optional[asyncio.Event] = None,
    ) -> None:
        self.config = config
        self.backend = backend
        self._stop = stop_event or asyncio.Event()
        self._tasks: Dict[str, asyncio.Task] = {}
        self._send_lock = asyncio.Lock()
        self._ws: Optional[websockets.WebSocketClientProtocol] = None

    async def run(self) -> None:
        attempt = 0
        while not self._stop.is_set():
            try:
                await self._connect_once()
                attempt = 0
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("Gateway connection error: %s", exc)

            if self._stop.is_set():
                break

            delay = _BACKOFF[min(attempt, len(_BACKOFF) - 1)]
            attempt += 1
            logger.info("Reconnecting to gateway in %ss", delay)
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=delay)
            except asyncio.TimeoutError:
                pass

    async def _connect_once(self) -> None:
        headers = {"Authorization": f"Bearer {self.config.worker_token}"}
        async with websockets.connect(
            self.config.gateway_url,
            additional_headers=headers,
            open_timeout=self.config.connect_timeout,
            ping_interval=20.0,
            ping_timeout=20.0,
        ) as ws:
            self._ws = ws
            logger.info("Connected to gateway")

            models = await self._discover()
            await self._send(
                RegisterMessage(
                    worker_id=self.config.worker_id,
                    backend=self.backend.name,
                    models=models,
                ).model_dump()
            )
            logger.info("Registered worker: %s backend=%s", self.config.worker_id, self.backend.name)
            logger.info("Models discovered: %s", models)

            heartbeat_task = asyncio.create_task(self._heartbeat_loop())
            try:
                async for raw in ws:
                    if isinstance(raw, bytes):
                        raw = raw.decode("utf-8")
                    try:
                        message = decode_message(json.loads(raw))
                    except Exception:
                        logger.warning("Dropping malformed gateway message")
                        continue
                    await self._handle_message(message)
            finally:
                heartbeat_task.cancel()
                for task in list(self._tasks.values()):
                    task.cancel()
                self._tasks.clear()
                self._ws = None

    async def _discover(self) -> list:
        if await self.backend.health():
            try:
                return [model.id for model in await self.backend.list_models()]
            except Exception as exc:
                logger.warning("Model discovery failed: %s", exc)
                return []
        logger.warning("Backend: OFFLINE")
        return []

    async def _heartbeat_loop(self) -> None:
        while True:
            await asyncio.sleep(self.config.heartbeat_interval)
            await self._send(
                HeartbeatMessage(
                    worker_id=self.config.worker_id, timestamp=time.time()
                ).model_dump()
            )

    async def _send(self, data: dict) -> None:
        async with self._send_lock:
            await self._ws.send(json.dumps(data))

    async def _handle_message(self, message) -> None:
        if message.type == "registered":
            logger.info("Worker registered (ack): %s", message.worker_id)
        elif message.type == "request":
            task = asyncio.create_task(self._handle_request(message))
            self._tasks[message.request_id] = task
            task.add_done_callback(
                lambda t, rid=message.request_id: self._tasks.pop(rid, None)
            )
        elif message.type == "cancel":
            task = self._tasks.get(message.request_id)
            if task is not None:
                logger.info("Request cancelled by gateway: %s", message.request_id)
                task.cancel()

    async def _handle_request(self, message: RequestMessage) -> None:
        request_id = message.request_id
        model = message.body.get("model") if isinstance(message.body, dict) else None
        logger.info("Request received: %s model=%s", request_id, model)
        try:
            if message.path == "/v1/chat/completions":
                request = ChatCompletionRequest.model_validate(message.body or {})
                async for chunk in self.backend.chat_completions(request):
                    await self._send(
                        ChunkMessage.encode_bytes(request_id, chunk).model_dump()
                    )
                await self._send(EndMessage(request_id=request_id).model_dump())
                logger.info("Stream completed: %s", request_id)
            else:
                await self._send_error(
                    request_id,
                    "unsupported_path",
                    f"Unsupported path: {message.path}",
                )
        except asyncio.CancelledError:
            logger.info("Request cancelled: %s", request_id)
            raise
        except Exception as exc:
            logger.exception("Request failed: %s", request_id)
            await self._send_error(request_id, "backend_error", str(exc))

    async def _send_error(self, request_id: str, error_type: str, message: str) -> None:
        try:
            await self._send(
                ErrorMessage(
                    request_id=request_id,
                    error=ErrorDetail(type=error_type, message=message),
                ).model_dump()
            )
        except Exception:
            logger.warning("Failed to send error for %s", request_id)
