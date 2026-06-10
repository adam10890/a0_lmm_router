from __future__ import annotations

from pathlib import Path

PLUGIN_ROOT = Path(__file__).resolve().parents[1]


def test_dashboard_store_wires_read_only_fit_summary_endpoint():
    js = (PLUGIN_ROOT / "webui" / "js" / "dashboard-store.js").read_text(encoding="utf-8")

    assert "fitSummary" in js
    assert "lmm_fit_summary" in js
    assert "_fetchFitSummary" in js
    assert "fitSummary: {" in js
    assert "read_only" in js


def test_dashboard_html_contains_fit_summary_panel():
    html = (PLUGIN_ROOT / "webui" / "dashboard.html").read_text(encoding="utf-8")

    assert "Hardware / Model Fit Summary" in html
    assert "Read-only mode" in html
    assert "fit-summary-table" in html
    assert "fitSummary.slots" in html
    assert "fitSummary.recommendations" in html
    assert "fitSummary.warnings" in html
