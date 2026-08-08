"""CLI entry points: ``llm-gateway`` and ``llm-worker``.

Configuration comes from CLI flags with environment-variable fallbacks so the
same binary works in scripts and containers.
"""

from __future__ import annotations

import asyncio
import logging

import typer

from .backend.factory import create_backend
from .config import BackendConfig, GatewayConfig, VLLMConfig, WorkerConfig
from .logging import setup_logging
from .worker.manager import WorkerManager

logger = logging.getLogger(__name__)

gateway_app = typer.Typer(help="LLM Reverse Deployer Gateway", no_args_is_help=True)
worker_app = typer.Typer(help="LLM Reverse Deployer Worker", no_args_is_help=True)


@gateway_app.callback()
def _gateway_callback() -> None:
    """Expose the intranet LLM via a public OpenAI-compatible API."""


@worker_app.callback()
def _worker_callback() -> None:
    """Connect a local LLM backend to a Gateway over a reverse tunnel."""


@gateway_app.command("start")
def gateway_start(
    host: str = typer.Option("0.0.0.0", "--host", help="Bind host"),
    port: int = typer.Option(17171, "--port", help="Bind port"),
    api_key: str = typer.Option(
        "", "--api-key", envvar="LLM_GATEWAY_API_KEY", help="Public API key for /v1/*"
    ),
    worker_token: str = typer.Option(
        "", "--worker-token", envvar="LLM_WORKER_TOKEN", help="Token for /worker/connect"
    ),
    log_level: str = typer.Option("INFO", "--log-level"),
) -> None:
    setup_logging(log_level)
    config = GatewayConfig(
        host=host, port=port, api_key=api_key, worker_token=worker_token
    )

    from .gateway.app import create_app
    import uvicorn

    app = create_app(config)
    logger.info("Starting gateway on %s:%s", host, port)
    uvicorn.run(app, host=host, port=port, log_level=log_level.lower())


@worker_app.command("start")
def worker_start(
    gateway: str = typer.Option(
        ..., "--gateway", envvar="LLM_GATEWAY_URL", help="Gateway URL, e.g. wss://llm.example.com/worker/connect"
    ),
    token: str = typer.Option(
        ..., "--token", envvar="LLM_WORKER_TOKEN", help="Worker token for Gateway auth"
    ),
    worker_id: str = typer.Option(
        ..., "--worker-id", envvar="LLM_WORKER_ID", help="Unique worker identifier"
    ),
    backend: str = typer.Option("vllm", "--backend", help="Backend type (v0.1: vllm)"),
    vllm: str = typer.Option(
        "http://127.0.0.1:8000", "--vllm", envvar="VLLM_BASE_URL", help="Local vLLM base URL"
    ),
    vllm_api_key: str = typer.Option(
        "", "--vllm-api-key", envvar="VLLM_API_KEY", help="Optional vLLM API key"
    ),
    heartbeat_interval: float = typer.Option(15.0, "--heartbeat-interval"),
    log_level: str = typer.Option("INFO", "--log-level"),
) -> None:
    if backend != "vllm":
        raise typer.BadParameter(
            f"Unknown backend: {backend!r}. v0.1 supports only 'vllm'."
        )
    setup_logging(log_level)
    config = WorkerConfig(
        gateway_url=gateway,
        worker_id=worker_id,
        worker_token=token,
        heartbeat_interval=heartbeat_interval,
        backend=BackendConfig(type="vllm", vllm=VLLMConfig(base_url=vllm, api_key=vllm_api_key)),
    )
    manager = WorkerManager(config, create_backend(config.backend))
    asyncio.run(manager.run())


if __name__ == "__main__":
    worker_app()
