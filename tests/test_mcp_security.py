from __future__ import annotations

import asyncio
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class DummyMCP:
    def __init__(self) -> None:
        self.tools: dict[str, object] = {}

    def tool(self):
        def _decorator(fn):
            self.tools[fn.__name__] = fn
            return fn

        return _decorator


def test_mcp_host_defaults_to_loopback_when_config_is_public(monkeypatch):
    from usr.plugins.a0_lmm_router.mcp_server import server

    monkeypatch.delenv("MCP_HOST", raising=False)
    monkeypatch.delenv("MCP_BIND_PUBLIC", raising=False)

    assert server._resolve_mcp_host({"host": "0.0.0.0"}) == "127.0.0.1"


def test_mcp_public_bind_requires_explicit_opt_in(monkeypatch):
    from usr.plugins.a0_lmm_router.mcp_server import server

    monkeypatch.setenv("MCP_BIND_PUBLIC", "1")

    assert server._resolve_mcp_host({"host": "0.0.0.0"}) == "0.0.0.0"


def test_static_token_verifier_accepts_matching_bearer_token(tmp_path):
    from usr.plugins.a0_lmm_router.mcp_server.server import StaticFileTokenVerifier

    token_file = tmp_path / "mcp-token"
    token_file.write_text("secret-token\n", encoding="utf-8")
    verifier = StaticFileTokenVerifier([str(token_file)])

    access = asyncio.run(verifier.verify_token("secret-token"))

    assert access is not None
    assert access.client_id == "a0_lmm_router"
    assert access.scopes == ["mcp"]


def test_static_token_verifier_rejects_wrong_token(tmp_path):
    from usr.plugins.a0_lmm_router.mcp_server.server import StaticFileTokenVerifier

    token_file = tmp_path / "mcp-token"
    token_file.write_text("secret-token", encoding="utf-8")
    verifier = StaticFileTokenVerifier([str(token_file)])

    assert asyncio.run(verifier.verify_token("wrong-token")) is None


def test_register_tools_hides_mutating_tools_by_default(monkeypatch):
    from usr.plugins.a0_lmm_router.mcp_server import tools

    monkeypatch.setattr(tools.bridge, "chat_complete", object())
    monkeypatch.setattr(tools.bridge, "get_embeddings", object())
    monkeypatch.setattr(tools.bridge, "fleet_status", object())
    monkeypatch.setattr(tools.bridge, "slot_configs", lambda: {})

    mcp = DummyMCP()
    tools.register_tools(mcp, allow_mutating_tools=False)

    assert "fleet_status" in mcp.tools
    assert "list_slots" in mcp.tools
    assert "start_fleet" not in mcp.tools
    assert "start_slot" not in mcp.tools
    assert "stop_slot" not in mcp.tools


def test_register_tools_includes_mutating_tools_when_enabled(monkeypatch):
    from usr.plugins.a0_lmm_router.mcp_server import tools

    monkeypatch.setattr(tools.bridge, "chat_complete", object())
    monkeypatch.setattr(tools.bridge, "get_embeddings", object())
    monkeypatch.setattr(tools.bridge, "fleet_status", object())
    monkeypatch.setattr(tools.bridge, "start_fleet", object())
    monkeypatch.setattr(tools.bridge, "start_slot", object())
    monkeypatch.setattr(tools.bridge, "stop_slot", object())
    monkeypatch.setattr(tools.bridge, "slot_configs", lambda: {})

    mcp = DummyMCP()
    tools.register_tools(mcp, allow_mutating_tools=True)

    assert "start_fleet" in mcp.tools
    assert "start_slot" in mcp.tools
    assert "stop_slot" in mcp.tools
