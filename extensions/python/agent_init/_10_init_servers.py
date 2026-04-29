"""llama.cpp Server Initialization Extension.

Runs on every `Agent.__init__` (agent_init extension point). The extension
point is invoked synchronously, so this handler MUST be a sync `execute`.
The actual start-up work is async, so we schedule it once per container
lifetime via `DeferredTask`.

Placement: extensions/python/agent_init/_10_init_servers.py
"""
from __future__ import annotations

import os

from helpers.extension import Extension
from helpers import files


# Module-level flag — one ignition attempt per container lifetime.
# Re-running the A0 container resets this; restarting the fleet by hand
# is covered by the /plugins/a0_lmm_router/lmm_fleet_ignite API.
_IGNITED = False


class LlamaCppInitExtension(Extension):
    """Kick off llama.cpp fleet startup the first time an agent is created."""

    def execute(self, **kwargs) -> None:  # SYNC — agent_init is called sync
        global _IGNITED
        if _IGNITED:
            return
        _IGNITED = True

        try:
            plugin_conf = files.get_abs_path(
                "usr/plugins/a0_lmm_router/conf/llama_cpp_servers.yaml"
            )
            root_conf = files.get_abs_path("conf/llama_cpp_servers.yaml")
            config_path = plugin_conf if os.path.exists(plugin_conf) else root_conf
            if not os.path.exists(config_path):
                return  # plugin disabled / no config

            from usr.plugins.a0_lmm_router.helpers.llama_cpp_manager import (
                LlamaCppManager,
            )

            manager = LlamaCppManager.get_instance(config_path)
            # Expose the manager on the agent for downstream tools.
            if self.agent is not None:
                self.agent.llama_cpp_manager = manager

            global_cfg = manager.global_config or {}
            if not global_cfg.get("auto_start", False):
                # Respect the master switch; nothing to do.
                return

            enabled = [n for n, s in manager.servers.items() if s.config.enabled]
            if not enabled:
                return

            # Defer the async start so we don't block agent construction
            # and so we don't poison AgentContext on load errors.
            try:
                from helpers.defer import DeferredTask
                DeferredTask("LlamaCppIgnite").start_task(manager.start_all)
            except Exception:
                # Deferred scheduler not available yet — silently skip.
                # The UI "Ignite Fleet" button and `launcher.py start`
                # remain available for manual ignition.
                pass

            if self.agent is not None and getattr(self.agent, "context", None):
                try:
                    self.agent.context.log.log(
                        type="info",
                        heading="llama.cpp",
                        content=(
                            f"Scheduling llama.cpp fleet start for: "
                            f"{', '.join(enabled)}"
                        ),
                    )
                except Exception:
                    pass

        except ImportError:
            return
        except Exception as exc:  # defensive — never break agent_init
            if self.agent is not None and getattr(self.agent, "context", None):
                try:
                    self.agent.context.log.log(
                        type="error",
                        heading="llama.cpp",
                        content=f"init error: {type(exc).__name__}: {exc}",
                    )
                except Exception:
                    pass
