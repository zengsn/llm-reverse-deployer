"""Worker Registry tests: register / find worker / find model / remove / offline."""

from __future__ import annotations

import pytest

from llm_reverse_deployer.gateway.registry import WorkerInfo, WorkerRegistry


@pytest.mark.asyncio
async def test_register_and_find():
    registry = WorkerRegistry()
    worker = WorkerInfo(
        worker_id="gpu-01",
        backend="vllm",
        models=["Qwen/Qwen3-32B"],
    )
    await registry.register(worker)
    assert registry.get("gpu-01") is worker
    assert registry.find_worker_for_model("Qwen/Qwen3-32B") is worker


@pytest.mark.asyncio
async def test_model_routing_ignores_offline_workers():
    registry = WorkerRegistry()
    online = WorkerInfo(worker_id="gpu-01", backend="vllm", models=["A"])
    offline = WorkerInfo(worker_id="gpu-02", backend="vllm", models=["A"], status="offline")
    await registry.register(online)
    await registry.register(offline)
    assert registry.find_worker_for_model("A") is online


@pytest.mark.asyncio
async def test_model_not_found_returns_none():
    registry = WorkerRegistry()
    await registry.register(WorkerInfo(worker_id="gpu-01", backend="vllm", models=["A"]))
    assert registry.find_worker_for_model("B") is None
    assert not registry.has_model("B")
    assert registry.has_model("A")


@pytest.mark.asyncio
async def test_remove_worker():
    registry = WorkerRegistry()
    await registry.register(WorkerInfo(worker_id="gpu-01", backend="vllm", models=["A"]))
    removed = await registry.remove("gpu-01")
    assert removed is not None
    assert registry.get("gpu-01") is None
    assert registry.find_worker_for_model("A") is None


@pytest.mark.asyncio
async def test_offline_worker_removed_from_models():
    registry = WorkerRegistry()
    await registry.register(WorkerInfo(worker_id="gpu-01", backend="vllm", models=["A"]))
    await registry.set_status("gpu-01", "offline")
    assert registry.find_worker_for_model("A") is None
    assert registry.list_models() == []


@pytest.mark.asyncio
async def test_aggregated_models_with_owned_by():
    registry = WorkerRegistry()
    await registry.register(
        WorkerInfo(worker_id="gpu-01", backend="vllm", models=["Qwen/Qwen3-32B"])
    )
    models = registry.list_models()
    assert models == [
        {"id": "Qwen/Qwen3-32B", "object": "model", "owned_by": "vllm"}
    ]


@pytest.mark.asyncio
async def test_heartbeat_updates_last_heartbeat():
    registry = WorkerRegistry()
    await registry.register(WorkerInfo(worker_id="gpu-01", backend="vllm", models=[]))
    assert registry.get("gpu-01").last_heartbeat is None
    assert await registry.update_heartbeat("gpu-01")
    assert registry.get("gpu-01").last_heartbeat is not None
    assert not await registry.update_heartbeat("missing")
