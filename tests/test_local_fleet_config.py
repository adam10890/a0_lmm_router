from __future__ import annotations

import configparser
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _read_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def test_local_fleet_utility_defaults_follow_chat_alias():
    env = _read_env(ROOT / "docker" / "docker-compose.lmm.env")

    assert env["UTILITY_MODEL_PATH"] == env["CHAT_MODEL_PATH"]
    assert env["UTILITY_CTX_SIZE"] == env["CHAT_CTX_SIZE"] == "131072"
    assert env["A0_LMM_UTILITY_FOLLOWS_CHAT"] == "1"


def test_router_preset_utility_is_direct_call_fallback_for_chat():
    parser = configparser.ConfigParser()
    parser.read(ROOT / "conf" / "models_preset.ini", encoding="utf-8")

    assert parser["utility"]["model"] == parser["chat"]["model"]
    assert parser["utility"]["ctx-size"] == parser["chat"]["ctx-size"] == "131072"
    assert parser["utility"]["cache-type-k"] == parser["chat"]["cache-type-k"]
    assert parser["utility"]["cache-type-v"] == parser["chat"]["cache-type-v"]


def test_plugin_config_routes_utility_role_to_chat_alias_by_default():
    config_text = (ROOT / "default_config.yaml").read_text(encoding="utf-8")

    assert "route_utility_to_chat_alias: true" in config_text
    assert "utility_chat_alias: chat" in config_text
