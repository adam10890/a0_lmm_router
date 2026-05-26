"""Tests for router context budgeting."""
from __future__ import annotations

import sys
from pathlib import Path

PLUGIN_ROOT = Path(__file__).resolve().parents[1]
HELPERS_ROOT = PLUGIN_ROOT / "helpers"
if str(HELPERS_ROOT) not in sys.path:
    sys.path.insert(0, str(HELPERS_ROOT))

from router_context import (  # noqa: E402
    EXTRAS_TEMPLATE_RESERVE,
    RESPONSE_TOKEN_RESERVE,
    _chat_signature,
    _normalize_api_base,
    history_token_budget,
    is_local_fleet_chat_active,
    resolve_router_ctx_limit,
)


def test_history_budget_reserves_system_completion_and_extras():
    cfg = {
        "provider": "lmm_router",
        "name": "chat",
        "api_base": "http://localhost:8080/v1",
        "ctx_length": 65536,
    }
    budget = history_token_budget(cfg, system_tokens=12000, extras_tokens=3000)
    # int(65536 * 0.9) - 12000 - 3000 - 8192 = 35790
    assert budget == 35790


def test_history_budget_minimum_floor():
    cfg = {"ctx_length": 8192, "name": "chat", "api_base": "http://127.0.0.1:9/v1"}
    budget = history_token_budget(cfg, system_tokens=70000, extras_tokens=0)
    assert budget == 4096


def test_resolve_router_ctx_falls_back_to_cfg():
    cfg = {"ctx_length": 32768, "name": "chat", "api_base": "http://127.0.0.1:9/v1"}
    assert resolve_router_ctx_limit(cfg) == 32768


def test_constants_sane():
    assert RESPONSE_TOKEN_RESERVE > 0
    assert EXTRAS_TEMPLATE_RESERVE > 0


def test_normalize_api_base_aliases():
    assert _normalize_api_base("http://127.0.0.1:8080/v1") == _normalize_api_base(
        "http://host.docker.internal:8080/v1"
    )


def test_chat_signature_local_fleet_shape():
    sig = _chat_signature(
        {
            "provider": "lmm_router",
            "name": "chat",
            "api_base": "http://host.docker.internal:8080/v1",
        }
    )
    assert sig == ("lmm_router", "chat", "http://host.docker.internal:8080/v1")


def test_is_local_fleet_inactive_without_agent():
    assert is_local_fleet_chat_active(None) is False
