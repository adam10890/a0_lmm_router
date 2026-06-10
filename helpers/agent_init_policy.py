"""Agent-init activity policy for the Agent Zero plugin surface.

Decides what the plugin may run inside the Agent Zero process when an
agent initializes. In ``quiet`` mode (the default) the agent process
stays free of plugin background work: no MCP subprocess spawn, no
BackendManager construction, no model-params warm. The fleet, the MCP
server, and the standalone provider are operated outside the agent
process (host compose, ``launcher.py``, ``scripts/run_provider.*``).
``full`` mode preserves the legacy bootstrap behavior.

Pure-policy module: no Agent Zero imports, no I/O beyond the mappings
passed in. Extensions fetch the plugin config themselves and delegate
every go/no-go decision here.
"""
from __future__ import annotations

import os
from typing import Any, Mapping

MODE_QUIET = "quiet"
MODE_FULL = "full"
DEFAULT_MODE = MODE_QUIET

ENV_MODE = "A0_LMM_AGENT_INIT_MODE"

_VALID_MODES = frozenset({MODE_QUIET, MODE_FULL})


def _normalized(value: Any) -> str:
    return str(value or "").strip().lower()


def resolve_mode(
    config: Mapping[str, Any] | None,
    env: Mapping[str, str] | None = None,
) -> str:
    """Resolve the agent-init mode: env override > plugin config > default."""
    env_map: Mapping[str, str] = os.environ if env is None else env

    env_mode = _normalized(env_map.get(ENV_MODE))
    if env_mode in _VALID_MODES:
        return env_mode

    section = config.get("agent_init") if isinstance(config, Mapping) else None
    if isinstance(section, Mapping):
        cfg_mode = _normalized(section.get("mode"))
        if cfg_mode in _VALID_MODES:
            return cfg_mode

    return DEFAULT_MODE


def allow_mcp_autostart(
    config: Mapping[str, Any] | None,
    env: Mapping[str, str] | None = None,
) -> bool:
    """May agent init spawn the MCP server subprocess?"""
    return resolve_mode(config, env) == MODE_FULL


def allow_backend_manager_init(
    config: Mapping[str, Any] | None,
    env: Mapping[str, str] | None = None,
) -> bool:
    """May agent init construct the BackendManager singleton?"""
    return resolve_mode(config, env) == MODE_FULL


def allow_params_warm(
    config: Mapping[str, Any] | None,
    env: Mapping[str, str] | None = None,
) -> bool:
    """May agent init warm the model-params cache via the host helper?"""
    return resolve_mode(config, env) == MODE_FULL
