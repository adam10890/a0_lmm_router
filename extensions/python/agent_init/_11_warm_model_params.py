"""Warm llama.cpp parameter cache on first agent init (Router Mode fleet).

Runs once per container lifetime: calls the host helper to plan or load cached
per-model ctx/KV settings and regenerate ``models_preset.ini`` without blocking
agent construction.
"""
from __future__ import annotations

import logging
import os

from helpers.extension import Extension
from helpers import files

log = logging.getLogger("a0_lmm_router.warm_params")

_WARMED = False


class WarmModelParamsExtension(Extension):
    """Deferred warm of model_params_cache.json + preset render."""

    def execute(self, **kwargs) -> None:
        global _WARMED
        if _WARMED:
            return
        _WARMED = True

        if os.environ.get("A0_LMM_SKIP_PARAMS_WARM", "").strip().lower() in ("1", "true", "yes"):
            return

        try:
            plugin_conf = files.get_abs_path(
                "usr/plugins/a0_lmm_router/conf/llama_cpp_servers.yaml"
            )
            root_conf = files.get_abs_path("conf/llama_cpp_servers.yaml")
            env_conf = os.environ.get("A0_LMM_ROUTER_CONFIG", "")
            config_path = env_conf if env_conf and os.path.exists(env_conf) else (
                root_conf if os.path.exists(root_conf) else plugin_conf
            )
            if not os.path.exists(config_path):
                return

            from usr.plugins.a0_lmm_router.helpers import fleet_models

            def _warm() -> None:
                try:
                    result = fleet_models.warm_params_cache(force_refresh=False, restart=False)
                    if result.get("ok"):
                        stats = result.get("stats") or {}
                        log.info(
                            "Model params cache warm: cached=%s computed=%s seeded=%s",
                            stats.get("cached"),
                            stats.get("computed"),
                            stats.get("seeded_from_last_success"),
                        )
                    else:
                        log.debug("Model params warm skipped: %s", result.get("error"))
                except Exception as exc:
                    log.debug("Model params warm failed: %s", exc)

            try:
                from helpers.defer import DeferredTask
                DeferredTask("LmmParamsWarm").start_task(_warm)
            except Exception:
                pass
        except ImportError:
            return
