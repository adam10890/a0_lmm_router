"""llama.cpp Server + MCP Initialization Extension.

Runs on every `Agent.__init__` (agent_init extension point). The extension
point is invoked synchronously, so this handler MUST be a sync `execute`.
The actual start-up work is async, so we schedule it once per container
lifetime via `DeferredTask`.

Auto-starts:
  1. llama.cpp fleet slots (if global.auto_start is true)
  2. MCP server on port 8095 (if mcp_server.enabled is true)

Placement: extensions/python/agent_init/_10_init_servers.py
"""
from __future__ import annotations

import logging
import os
import socket
import subprocess
import sys
import yaml

from helpers.extension import Extension
from helpers import files

log = logging.getLogger("a0_lmm_router.init")

# Module-level flag — one ignition attempt per container lifetime.
# Re-running the A0 container resets this; restarting the fleet by hand
# is covered by the /plugins/a0_lmm_router/lmm_fleet_ignite API.
_IGNITED = False


def _mcp_already_running(port: int = 8095) -> bool:
    """Check if something is already listening on the MCP port."""
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.5):
            return True
    except (OSError, ConnectionRefusedError):
        return False


def _start_mcp_server(config_path: str) -> None:
    """Launch the MCP server as a detached background subprocess."""
    try:
        with open(config_path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
    except Exception:
        return

    mcp_cfg = data.get("mcp_server", {}) if isinstance(data, dict) else {}
    if not mcp_cfg.get("enabled", True):
        log.info("MCP server disabled in config — skipping auto-start.")
        return

    port = int(mcp_cfg.get("port", 8095))
    if _mcp_already_running(port):
        log.info("MCP server already running on port %d — skipping.", port)
        return

    launcher = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "launcher.py",
    )
    log.info("Starting MCP server on port %d via %s ...", port, launcher)

    subprocess.Popen(
        [sys.executable, launcher, "mcp"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
    )


class LlamaCppInitExtension(Extension):
    """Kick off llama.cpp fleet + MCP server the first time an agent is created."""

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
            env_conf = os.environ.get("A0_LMM_ROUTER_CONFIG", "")
            config_path = env_conf if env_conf and os.path.exists(env_conf) else (
                root_conf if os.path.exists(root_conf) else plugin_conf
            )
            if not os.path.exists(config_path):
                return  # plugin disabled / no config

            # ── MCP server auto-start (always, independent of auto_start) ──
            _start_mcp_server(config_path)

            from usr.plugins.a0_lmm_router.helpers.llama_cpp_manager import BackendManager

            manager = BackendManager.get_instance(config_path)
            # Expose the manager on the agent for downstream tools.
            if self.agent is not None:
                self.agent.llama_cpp_manager = manager
                self.agent.lmm_backend_manager = manager

            global_cfg = getattr(manager, "global_config", {}) or {}
            if not global_cfg.get("auto_start", False):
                # Respect the master switch; nothing to do.
                return

            enabled = [
                n for n, s in getattr(manager, "_slot_configs", {}).items()
                if s.get("enabled", True)
            ]
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
