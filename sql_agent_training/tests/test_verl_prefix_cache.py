from __future__ import annotations

import asyncio
import sys
from types import ModuleType
from typing import Any

from sql_agent_training.train.verl_prefix_cache import (
    _force_vllm_config_prefix_cache,
    patch_verl_prefix_cache_config,
)


def test_force_vllm_config_prefix_cache_updates_cache_config() -> None:
    cache_config = type("FakeCacheConfig", (), {"enable_prefix_caching": True})()
    vllm_config = type("FakeVllmConfig", (), {"cache_config": cache_config})()

    before, after = _force_vllm_config_prefix_cache(vllm_config, False)

    assert before is True
    assert after is False
    assert cache_config.enable_prefix_caching is False


def test_patch_run_server_forces_created_engine_config(monkeypatch: Any) -> None:
    class FakeCacheConfig:
        enable_prefix_caching = True

    class FakeVllmConfig:
        cache_config = FakeCacheConfig()

    class FakeConfig:
        enable_prefix_caching = False

    class FakeServer:
        config = FakeConfig()

    class FakeAsyncEngineArgs:
        @staticmethod
        def create_engine_config(_engine_args: Any, *_args: Any, **_kwargs: Any) -> FakeVllmConfig:
            return FakeVllmConfig()

    async def fake_run_server(self: Any, _args: Any) -> bool:
        vllm_config = FakeAsyncEngineArgs.create_engine_config(object(), usage_context=object())
        return vllm_config.cache_config.enable_prefix_caching

    class FakeVllmHttpServer:
        run_server = fake_run_server

    vllm_module = ModuleType("vllm")
    vllm_engine_module = ModuleType("vllm.engine")
    vllm_arg_utils_module = ModuleType("vllm.engine.arg_utils")
    vllm_arg_utils_module.AsyncEngineArgs = FakeAsyncEngineArgs
    server_module = ModuleType("verl.workers.rollout.vllm_rollout.vllm_async_server")
    server_module.vLLMHttpServer = FakeVllmHttpServer
    monkeypatch.setitem(sys.modules, "vllm", vllm_module)
    monkeypatch.setitem(sys.modules, "vllm.engine", vllm_engine_module)
    monkeypatch.setitem(sys.modules, "vllm.engine.arg_utils", vllm_arg_utils_module)
    monkeypatch.setitem(sys.modules, "verl.workers.rollout.vllm_rollout.vllm_async_server", server_module)
    monkeypatch.setenv("SQL_AGENT_FIX_PREFIX_CACHE_CONFIG", "1")

    assert patch_verl_prefix_cache_config()
    result = asyncio.run(server_module.vLLMHttpServer.run_server(FakeServer(), object()))
    assert result is False
