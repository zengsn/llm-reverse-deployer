"""Gateway WebSocket endpoint for Worker reverse tunnels.

Workers dial in (``wss://<gateway>/worker/connect``) and keep a long-lived
connection. The Gateway never initiates a connection to a Worker.
"""

from __future__ import annotations

import asyncio
import logging
import secrets
from typing import TYPE_CHECKING, Optional

from fastapi import WebSocket, WebSocketDisconnect

from ..protocol import decode_message
from ..protocol.messages import RegisteredMessage
from .errors import GatewayError
from .registry import WorkerInfo

if TYPE_CHECKING:
    from .app import GatewayState

logger = logging.getLogger(__name__)


class WorkerConnection:
    """Serialized access to a Worker's WebSocket (shared senders)."""

    def __init__(self, websocket: WebSocket) -> None:
        self._ws = websocket
        self._lock = asyncio.Lock()

    async def send_json(self, data: dict) -> None:
        async with self._lock:
            await self._ws.send_json(data)

    async def close(self, code: int = 1000) -> None:
        async with self._lock:
            await self._ws.close(code=code)


def _extract_bearer(authorization: str) -> str:
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer":
        return ""
    return token.strip()


def _authenticate(state: "GatewayState", websocket: WebSocket) -> None:
    token = _extract_bearer(websocket.headers.get("authorization", ""))
    if not token or not secrets.compare_digest(token, state.config.worker_token):
        raise GatewayError(401, "unauthorized", "Invalid worker token")


async def handle_worker_connection(
    websocket: WebSocket, state: "GatewayState"
) -> None:
    _authenticate(state, websocket)
    await websocket.accept()

    connection = WorkerConnection(websocket)
    worker_id: Optional[str] = None
    try:
        while True:
            raw = await websocket.receive_json()
            try:
                message = decode_message(raw)
            except Exception:
                logger.warning("Dropping malformed tunnel message")
                continue

            if message.type == "register":
                worker_id = message.worker_id
                info = WorkerInfo(
                    worker_id=message.worker_id,
                    backend=message.backend,
                    models=message.models,
                    connection=connection,
                )
                await state.registry.register(info)
                await connection.send_json(
                    RegisteredMessage(
                        worker_id=message.worker_id, models=message.models
                    ).model_dump()
                )
                logger.info(
                    "Worker registered: %s backend=%s",
                    message.worker_id,
                    message.backend,
                )
                logger.info("Models discovered: %s", message.models)
            elif message.type == "heartbeat":
                await state.registry.update_heartbeat(message.worker_id)
            elif message.type in ("chunk", "end", "error"):
                await state.router.push(message)
    except WebSocketDisconnect:
        logger.info("Worker disconnected: %s", worker_id)
    except Exception:
        logger.exception("Worker connection error: %s", worker_id)
    finally:
        if worker_id is not None:
            await state.registry.set_status(worker_id, "offline")
            await state.router.fail_requests_for_worker(worker_id)
