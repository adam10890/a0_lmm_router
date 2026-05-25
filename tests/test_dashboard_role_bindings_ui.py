from __future__ import annotations

from pathlib import Path

PLUGIN_ROOT = Path(__file__).resolve().parents[1]


def test_dashboard_store_wires_role_bindings_endpoint():
    js = (PLUGIN_ROOT / "webui" / "js" / "dashboard-store.js").read_text(encoding="utf-8")

    assert "routerAliases" in js
    assert "setRouterAliasModel" in js
    assert "roleBindings" in js
    assert "_fetchRoleBindings" in js
    assert "setRoleAliasModel" in js


def test_dashboard_has_role_bindings_card_and_preload_label():
    html = (PLUGIN_ROOT / "webui" / "dashboard.html").read_text(encoding="utf-8")

    assert "Role Bindings" in html
    assert "role-binding-card" in html
    assert "Pre-load on startup" in html
