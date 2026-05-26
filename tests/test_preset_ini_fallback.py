from __future__ import annotations

from pathlib import Path

from helpers import preset_ini


def test_rewrite_alias_model_updates_only_target_section(tmp_path: Path):
    preset = tmp_path / "models_preset.ini"
    preset.write_text(
        "[chat]\nalias = chat\nmodel = /models/old.gguf\n\n"
        "[utility]\nalias = utility\nmodel = /models/util.gguf\n",
        encoding="utf-8",
    )
    result = preset_ini.write_alias_model("chat", "/models/new-chat.gguf", preset)
    assert result["ok"] is True
    text = preset.read_text(encoding="utf-8")
    assert "/models/new-chat.gguf" in text
    assert "/models/util.gguf" in text
    assert "/models/old.gguf" not in text
    assert (tmp_path / "models_preset.ini.bak").is_file()


def test_fleet_models_falls_back_when_helper_unknown_endpoint(monkeypatch):
    from helpers import fleet_models

    monkeypatch.setattr(fleet_models, "_helper_router_capabilities", lambda: set())
    monkeypatch.setattr(
        fleet_models,
        "_helper_request",
        lambda *a, **k: {"ok": False, "error": "unknown endpoint: /router/write_preset_ini"},
    )

    called = {}

    def fake_local(alias, model_path, preset_path=None):
        called["alias"] = alias
        return {"ok": True, "via": "local", "preset_path": "/x", "backup_path": "/x.bak", "snippet": ""}

    monkeypatch.setattr(fleet_models, "write_alias_model", fake_local, raising=False)
    # Import path used inside write_preset_ini
    import helpers.preset_ini as pi

    monkeypatch.setattr(pi, "write_alias_model", fake_local)

    out = fleet_models.write_preset_ini("chat", "/models/x.gguf")
    assert out.get("ok") is True
    assert out.get("via") == "local"
    assert called.get("alias") == "chat"
