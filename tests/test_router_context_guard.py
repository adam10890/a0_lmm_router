"""Tests for router context budgeting."""
from __future__ import annotations

import sys
from pathlib import Path

PLUGIN_ROOT = Path(__file__).resolve().parents[1]
if str(PLUGIN_ROOT) not in sys.path:
    sys.path.insert(0, str(PLUGIN_ROOT))

from helpers.router_context import (
    EXTRAS_TEMPLATE_RESERVE,
    RESPONSE_TOKEN_RESERVE,
    history_token_budget,
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
