"""Gateway Worker Registry.

An in-memory map of connected Workers. v0.1 keeps everything in memory; a
database is explicitly out of scope.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class WorkerInfo(BaseModel):
    worker_id: str
    backend: str
    models: List[str] = Field(default_factory=list)
    status: str = "online"  # "online" | "offline"
    #: Connection wrapper used to push tunnel messages to the Worker.
    connection: Optional[Any] = None
    last_heartbeat: Optional[float] = None
    connected_at: float = Field(default_factory=time.time)


class WorkerRegistry:
    """Thread-safe (asyncio) registry of Workers + model routing."""

    def __init__(self) -> None:
        self._workers: Dict[str, WorkerInfo] = {}
        self._lock = asyncio.Lock()

    async def register(self, worker: WorkerInfo) -> None:
        async with self._lock:
            self._workers[worker.worker_id] = worker

    async def update_heartbeat(self, worker_id: str) -> bool:
        async with self._lock:
            worker = self._workers.get(worker_id)
            if worker is None:
                return False
            worker.last_heartbeat = time.time()
            return True

    async def set_status(self, worker_id: str, status: str) -> None:
        async with self._lock:
            worker = self._workers.get(worker_id)
            if worker is not None:
                worker.status = status

    async def remove(self, worker_id: str) -> Optional[WorkerInfo]:
        async with self._lock:
            return self._workers.pop(worker_id, None)

    def get(self, worker_id: str) -> Optional[WorkerInfo]:
        return self._workers.get(worker_id)

    def find_worker_for_model(self, model: str) -> Optional[WorkerInfo]:
        for worker in list(self._workers.values()):
            if worker.status == "online" and model in worker.models:
                return worker
        return None

    def has_model(self, model: str) -> bool:
        return any(model in worker.models for worker in self._workers.values())

    def list_models(self) -> List[Dict[str, str]]:
        """Aggregated OpenAI-style model list from all online Workers."""
        models: Dict[str, str] = {}
        for worker in list(self._workers.values()):
            if worker.status != "online":
                continue
            for model in worker.models:
                # First worker wins when multiple Workers serve the same model.
                models.setdefault(model, worker.backend)
        return [
            {"id": model, "object": "model", "owned_by": owned_by}
            for model, owned_by in models.items()
        ]

    def list_workers(self) -> List[WorkerInfo]:
        return list(self._workers.values())

    def snapshot(self) -> Dict[str, Dict[str, Any]]:
        return {
            worker_id: {
                "worker_id": worker.worker_id,
                "backend": worker.backend,
                "models": worker.models,
                "status": worker.status,
                "last_heartbeat": worker.last_heartbeat,
            }
            for worker_id, worker in self._workers.items()
        }
