"""
api/set_router_default.py — Set the default (pre-loaded) model for a router slot.

Persists the choice to conf/router_state.json so it survives restarts.
The change takes effect on the next slot start (restart required).
"""
from __future__ import annotations

import configparser
import json
import os
from pathlib import Path

from flask import Request
from helpers.api import ApiHandler

try:
    from usr.plugins.a0_lmm_router.helpers.llama_cpp_manager import BackendManager
except ImportError:
    import sys
    _here = os.path.dirname(os.path.abspath(__file__))
    _plugin_root = os.path.dirname(_here)
    if _plugin_root not in sys.path:
        sys.path.insert(0, _plugin_root)
    from helpers.llama_cpp_manager import BackendManager


def _state_file() -> str:
    """Path to the persistent router state file."""
    for candidate in [
        "/a0/conf/router_state.json",
        os.path.join(os.path.dirname(__file__), "..", "conf", "router_state.json"),
    ]:
        if os.path.isdir(os.path.dirname(os.path.abspath(candidate))):
            return os.path.abspath(candidate)
    return "/a0/conf/router_state.json"


def _load_state() -> dict:
    p = _state_file()
    if os.path.exists(p):
        try:
            with open(p, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _save_state(state: dict) -> None:
    p = _state_file()
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def _resolve_model_path(slot_cfg: dict, alias: str) -> str:
    """Find the full model path for an alias in the preset.ini."""
    preset = slot_cfg.get("router_models_preset", "")
    mdir   = slot_cfg.get("router_models_dir", "")
    if not preset or not os.path.exists(preset):
        return ""
    cp = configparser.ConfigParser()
    cp.read(preset, encoding="utf-8")
    for section in cp.sections():
        a = cp.get(section, "alias", fallback=section)
        if a == alias:
            rel = cp.get(section, "model", fallback="")
            return os.path.join(mdir, rel) if (mdir and rel) else rel
    return ""


class SetRouterDefault(ApiHandler):
    async def process(self, input: dict, request: Request) -> dict:
        slot_id     = input.get("slot_id", "")
        model_alias = input.get("model_alias", "")

        if not slot_id or not model_alias:
            return {"ok": False, "error": "slot_id and model_alias are required"}

        mgr = BackendManager.get_instance()
        slot_cfg = mgr._slot_configs.get(slot_id)
        if not slot_cfg:
            return {"ok": False, "error": f"Slot '{slot_id}' not found"}
        if not slot_cfg.get("router_mode"):
            return {"ok": False, "error": f"Slot '{slot_id}' is not in router mode"}

        model_path = _resolve_model_path(slot_cfg, model_alias)

        # Update in-memory config
        mgr._slot_configs[slot_id]["router_default_model"] = model_alias

        # Persist to router_state.json
        state = _load_state()
        state.setdefault(slot_id, {})["router_default_model"] = model_alias
        _save_state(state)

        return {
            "ok": True,
            "slot_id": slot_id,
            "model_alias": model_alias,
            "model_path": model_path,
            "restart_required": True,
            "message": f"Default set to '{model_alias}'. Restart slot to apply.",
        }
