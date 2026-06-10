"""Tests for router context budgeting."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

PLUGIN_ROOT = Path(__file__).resolve().parents[1]
HELPERS_ROOT = PLUGIN_ROOT / "helpers"
if str(HELPERS_ROOT) not in sys.path:
    sys.path.insert(0, str(HELPERS_ROOT))

import router_context  # noqa: E402
from router_context import (  # noqa: E402
    EXTRAS_TEMPLATE_RESERVE,
    RESPONSE_TOKEN_RESERVE,
    _chat_signature,
    _normalize_api_base,
    context_budget_details,
    history_token_budget,
    is_local_fleet_chat_active,
    resolve_router_ctx_limit,
)


@pytest.fixture
def offline_ctx_sources(monkeypatch):
    """Force resolve_router_ctx_limit() onto the cfg ctx_length path.

    Without this, fetch_router_model_ctx() issues a real HTTP GET to the
    cfg api_base (a live Router Mode fleet on :8080 reports its actual
    n_ctx and overrides the test value), and read_slot_context_size()
    can leak the real llama_cpp_servers.yaml via the BackendManager
    singleton when other tests in the session have initialized it.
    """
    monkeypatch.setattr(router_context, "fetch_router_model_ctx", lambda *a, **k: None)
    monkeypatch.setattr(router_context, "read_slot_context_size", lambda: None)


def test_history_budget_reserves_system_completion_and_extras(offline_ctx_sources):
    cfg = {
        "provider": "lmm_router",
        "name": "chat",
        "api_base": "http://localhost:8080/v1",
        "ctx_length": 65536,
    }
    budget = history_token_budget(cfg, system_tokens=12000, extras_tokens=3000)
    # int(65536 * 0.70) - 12000 - 3000 - 8192 = 22683
    assert budget == 22683


def test_context_budget_details_reports_effective_window(offline_ctx_sources):
    cfg = {
        "provider": "lmm_router",
        "name": "chat",
        "api_base": "http://127.0.0.1:9/v1",
        "ctx_length": 65536,
    }
    details = context_budget_details(cfg, 1000, extras_tokens=500, history_tokens=20000)
    assert details["hard_ctx"] == 65536
    assert details["effective_ratio"] == 0.70
    assert details["effective_ctx"] == 45875
    assert details["projected_occupancy"] > details["occupancy"]


def test_history_budget_minimum_floor(offline_ctx_sources):
    cfg = {"ctx_length": 8192, "name": "chat", "api_base": "http://127.0.0.1:9/v1"}
    budget = history_token_budget(cfg, system_tokens=70000, extras_tokens=0)
    assert budget == 4096


def test_resolve_router_ctx_falls_back_to_cfg(offline_ctx_sources):
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
