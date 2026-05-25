from __future__ import annotations

from pathlib import Path

import yaml

PLUGIN_ROOT = Path(__file__).resolve().parents[1]


def test_plugin_manifest_uses_standard_external_settings_section():
    manifest = yaml.safe_load((PLUGIN_ROOT / "plugin.yaml").read_text(encoding="utf-8"))

    assert manifest["settings_sections"] == ["external"]
    assert manifest["always_enabled"] is False


def test_default_config_declares_security_knobs():
    default_config = yaml.safe_load((PLUGIN_ROOT / "default_config.yaml").read_text(encoding="utf-8"))

    assert default_config["mcp"]["allow_mutating_tools"] is False
    assert default_config["host_helper"]["bind"] == "127.0.0.1"
    assert "compose_allowlist" in default_config["host_helper"]
    assert default_config["fleet"]["preferred_mode"] == "router"
