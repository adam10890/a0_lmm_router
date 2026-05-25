from __future__ import annotations

from pathlib import Path

PLUGIN_ROOT = Path(__file__).resolve().parents[1]


def test_dashboard_store_has_ux_helpers():
    js = (PLUGIN_ROOT / "webui" / "js" / "dashboard-store.js").read_text(encoding="utf-8")

    assert "normalizeModelPath" in js
    assert "effectiveFleetMode" in js
    assert "showToast" in js
    assert "roleBindingLoadedLabel" in js
    assert "fleetBannerNote" in js
    assert "isRouterPrimaryUI" in js


def test_dashboard_html_ux_layout():
    html = (PLUGIN_ROOT / "webui" / "dashboard.html").read_text(encoding="utf-8")

    assert "role-binding-card" in html
    assert "lmm-toast-stack" in html
    assert "fleet-mode-banner--router" in html or "fleetModeBannerClass" in html
    assert "section-collapsible" in html
    assert "Ignite Router" in html
    assert "role-binding-row" not in html
    # Fleet banner should appear before Compute
    assert html.index("Fleet Mode Banner (priority)") < html.index("Compute section")


def test_router_aliases_applies_default_port():
    from api.router_aliases import _apply_slot_defaults

    bindings = {
        "chat": {"alias": "chat", "loaded": True, "port": None},
        "utility": {"alias": "utility", "loaded": False, "port": None},
    }
    _apply_slot_defaults(bindings, {"port": 8080})
    assert bindings["chat"]["port"] == 8080
    assert bindings["utility"]["port"] is None
