from __future__ import annotations

from pathlib import Path


CONFIG_HTML = Path(__file__).resolve().parents[1] / "webui" / "config.html"


def test_config_panel_registers_alpine_destroy_cleanup():
    html = CONFIG_HTML.read_text(encoding="utf-8")

    assert 'x-destroy="cleanup()"' in html


def test_config_panel_clears_poll_jobs_interval():
    html = CONFIG_HTML.read_text(encoding="utf-8")

    assert "_pollJobsTimer" in html
    assert "clearInterval(this._pollJobsTimer)" in html
    assert "this._pollJobsTimer = setInterval" in html
