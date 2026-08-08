"""Worker lifecycle manager.

    load config -> create Backend -> run tunnel (connect, health, discover,
    register, heartbeat, serve requests) with auto-reconnect.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

from ..backend.base import Backend
from ..config import WorkerConfig
from .tunnel import TunnelClient

logger = logging.getLogger(__name__)


class WorkerManager:
    def __init__(self, config: WorkerConfig, backend: Backend) -> None:
        self.config = config
        self.backend = backend
        self._stop = asyncio.Event()
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    async def run(self) -> None:
        self._loop = asyncio.get_running_loop()
        logger.info("Starting worker: %s", self.config.worker_id)
        logger.info("Backend: %s", self.backend.name)
        endpoint = getattr(self.backend, "endpoint", None)
        if endpoint:
            logger.info("%s endpoint: %s", self.backend.name, endpoint)

        client = TunnelClient(self.config, self.backend, self._stop)
        try:
            await client.run()
        finally:
            close = getattr(self.backend, "close", None)
            if close is not None:
                await close()

    def stop(self) -> None:
        """Request a graceful shutdown (safe to call from another thread)."""
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._stop.set)
