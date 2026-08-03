"""Compatibility patch for verl async vLLM prefix-cache settings."""

from __future__ import annotations

import functools
import importlib
import os
from typing import Any

PATCH_ENV = "SQL_AGENT_FIX_PREFIX_CACHE_CONFIG"
LOG_PREFIX = "PREFIX_CACHE_PATCH"


def _env_flag(name: str) -> bool:
    value = os.environ.get(name)
    return value is not None and value.strip().lower() in {"1", "true", "yes", "on"}


def _rollout_prefix_cache_value(config: Any) -> bool:
    value = getattr(config, "enable_prefix_caching", None)
    if value is None and hasattr(config, "get"):
        value = config.get("enable_prefix_caching", True)
    return bool(value)


def _force_vllm_config_prefix_cache(vllm_config: Any, enabled: bool) -> tuple[Any, Any]:
    cache_config = getattr(vllm_config, "cache_config", None)
    if cache_config is None or not hasattr(cache_config, "enable_prefix_caching"):
        return None, None

    before = cache_config.enable_prefix_caching
    cache_config.enable_prefix_caching = enabled
    return before, cache_config.enable_prefix_caching


def patch_verl_prefix_cache_config() -> bool:
    """Force async vLLM engine config to honor verl RolloutConfig."""

    if not _env_flag(PATCH_ENV):
        return False

    from vllm.engine.arg_utils import AsyncEngineArgs

    server_module = importlib.import_module("verl.workers.rollout.vllm_rollout.vllm_async_server")
    server_cls = server_module.vLLMHttpServer
    original_run_server = getattr(server_cls, "run_server", None)
    if original_run_server is None or getattr(original_run_server, "_sql_agent_prefix_cache_patched", False):
        return False

    @functools.wraps(original_run_server)
    async def run_server_with_prefix_cache_config(self: Any, args: Any, *run_args: Any, **run_kwargs: Any) -> Any:
        original_create_engine_config = AsyncEngineArgs.create_engine_config
        enabled = _rollout_prefix_cache_value(self.config)

        @functools.wraps(original_create_engine_config)
        def create_engine_config_with_prefix_cache(engine_args: Any, *args: Any, **kwargs: Any) -> Any:
            vllm_config = original_create_engine_config(engine_args, *args, **kwargs)
            before, after = _force_vllm_config_prefix_cache(vllm_config, enabled)
            print(
                f"{LOG_PREFIX} applied rollout_enable_prefix_caching={enabled} "
                f"engine_before={before} engine_after={after}",
                flush=True,
            )
            return vllm_config

        AsyncEngineArgs.create_engine_config = create_engine_config_with_prefix_cache
        try:
            return await original_run_server(self, args, *run_args, **run_kwargs)
        finally:
            AsyncEngineArgs.create_engine_config = original_create_engine_config

    run_server_with_prefix_cache_config._sql_agent_prefix_cache_patched = True  # type: ignore[attr-defined]
    server_cls.run_server = run_server_with_prefix_cache_config
    print(f"{LOG_PREFIX}_ENABLED server={server_cls.__module__}.{server_cls.__name__}", flush=True)
    return True
