"""Tests for the quiet/full agent-init activity policy."""
from __future__ import annotations

import sys
from pathlib import Path

PLUGIN_ROOT = Path(__file__).resolve().parents[1]
HELPERS_ROOT = PLUGIN_ROOT / "helpers"
if str(HELPERS_ROOT) not in sys.path:
    sys.path.insert(0, str(HELPERS_ROOT))

from agent_init_policy import (  # noqa: E402
    DEFAULT_MODE,
    ENV_MODE,
    MODE_FULL,
    MODE_QUIET,
    allow_backend_manager_init,
    allow_mcp_autostart,
    allow_params_warm,
    resolve_mode,
)

_ALL_GATES = (allow_mcp_autostart, allow_backend_manager_init, allow_params_warm)


def test_default_mode_is_quiet():
    assert DEFAULT_MODE == MODE_QUIET
    assert resolve_mode({}, env={}) == MODE_QUIET
    assert resolve_mode(None, env={}) == MODE_QUIET


def test_config_full_enables_bootstrap():
    config = {"agent_init": {"mode": "full"}}
    assert resolve_mode(config, env={}) == MODE_FULL
    for gate in _ALL_GATES:
        assert gate(config, env={}) is True


def test_config_quiet_disables_bootstrap():
    config = {"agent_init": {"mode": "quiet"}}
    assert resolve_mode(config, env={}) == MODE_QUIET
    for gate in _ALL_GATES:
        assert gate(config, env={}) is False


def test_env_override_beats_config():
    config = {"agent_init": {"mode": "quiet"}}
    assert resolve_mode(config, env={ENV_MODE: "full"}) == MODE_FULL

    config = {"agent_init": {"mode": "full"}}
    assert resolve_mode(config, env={ENV_MODE: "quiet"}) == MODE_QUIET


def test_values_are_normalized():
    config = {"agent_init": {"mode": "  FULL "}}
    assert resolve_mode(config, env={}) == MODE_FULL
    assert resolve_mode({}, env={ENV_MODE: " Quiet  "}) == MODE_QUIET


def test_invalid_values_fall_through():
    # Invalid env falls through to config; invalid config falls to default.
    config = {"agent_init": {"mode": "full"}}
    assert resolve_mode(config, env={ENV_MODE: "loud"}) == MODE_FULL
    assert resolve_mode({"agent_init": {"mode": "banana"}}, env={}) == MODE_QUIET
    assert resolve_mode({"agent_init": "not-a-mapping"}, env={}) == MODE_QUIET


def test_gates_default_closed_without_config():
    for gate in _ALL_GATES:
        assert gate({}, env={}) is False
        assert gate(None, env={}) is False
