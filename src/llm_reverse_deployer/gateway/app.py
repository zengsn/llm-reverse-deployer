"""Gateway FastAPI application.

Public surface (v0.1):

    GET  /health
    GET  /v1/models
    POST /v1/chat/completions
    WS   /worker/connect
"""

from __future__ import annotations

import asyncio
import json
import logging
import secrets
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import Depends, FastAPI, Request, Response, WebSocket
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from ..config import GatewayConfig
from .errors import GatewayError
from .registry import WorkerRegistry
from .router import RequestRouter
from .websocket import handle_worker_connection

logger = logging.getLogger(__name__)

_bearer = HTTPBearer(auto_error=False)


class GatewayState:
    """Shared state for one Gateway process."""

    def __init__(self, config: GatewayConfig) -> None:
        self.config = config
        self.registry = WorkerRegistry()
        self.router = RequestRouter(config, self.registry)


def _require_api_key(config: GatewayConfig):
    async def dependency(
        credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
    ) -> None:
        if not config.api_key:
            return
        if credentials is None or not secrets.compare_digest(
            credentials.credentials, config.api_key
        ):
            raise GatewayError(401, "unauthorized", "Invalid API key")

    return dependency


async def _heartbeat_sweeper(state: GatewayState) -> None:
    """Mark silent Workers offline and drop their stale connections."""
    while True:
        await asyncio.sleep(15)
        now = time.time()
        for worker in state.registry.list_workers():
            heartbeat = worker.last_heartbeat
            if (
                heartbeat is not None
                and now - heartbeat > state.config.heartbeat_timeout
            ):
                logger.warning("Worker heartbeat timeout: %s", worker.worker_id)
                await state.registry.set_status(worker.worker_id, "offline")
                if worker.connection is not None:
                    await worker.connection.close(code=4000)


@asynccontextmanager
async def _lifespan(app: FastAPI):
    state: GatewayState = app.state.gateway
    sweeper = asyncio.create_task(_heartbeat_sweeper(state))
    try:
        yield
    finally:
        sweeper.cancel()


def create_app(config: GatewayConfig) -> FastAPI:
    state = GatewayState(config)

    app = FastAPI(title="LLM Reverse Deployer Gateway", lifespan=_lifespan)
    app.state.gateway = state

    api_key_dep = Depends(_require_api_key(config))

    @app.exception_handler(GatewayError)
    async def _gateway_error_handler(request: Request, exc: GatewayError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": {"type": exc.error_type, "message": exc.message}},
        )

    @app.exception_handler(Exception)
    async def _unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled gateway error")
        return JSONResponse(
            status_code=500,
            content={"error": {"type": "internal_error", "message": "Internal server error"}},
        )

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok"}

    @app.get("/v1/models", dependencies=[api_key_dep])
    async def list_models() -> dict:
        return {"object": "list", "data": state.registry.list_models()}

    @app.post("/v1/chat/completions", dependencies=[api_key_dep])
    async def chat_completions(request: Request) -> Response:
        raw = await request.body()
        if len(raw) > state.config.request_body_limit:
            raise GatewayError(
                413,
                "request_too_large",
                f"Request body exceeds limit of {state.config.request_body_limit} bytes",
            )
        try:
            body = json.loads(raw)
        except json.JSONDecodeError:
            raise GatewayError(400, "invalid_request", "Request body must be valid JSON")
        return await state.router.handle_chat_completion(body)

    @app.websocket("/worker/connect")
    async def worker_connect(websocket: WebSocket) -> None:
        await handle_worker_connection(websocket, state)

    return app
