"""api/set_router_alias_model.py - Rewrite a router alias and restart router."""
from __future__ import annotations

import os
import sys

_PLUGIN_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PLUGIN_ROOT not in sys.path:
    sys.path.insert(0, _PLUGIN_ROOT)

from flask import Request
try:
    from helpers.api import ApiHandler
except ImportError:
    class ApiHandler:  # type: ignore[no-redef]
        pass

try:
    from usr.plugins.a0_lmm_router.helpers.llama_cpp_manager import BackendManager
    from usr.plugins.a0_lmm_router.helpers import fleet_models
except ImportError:
    _here = os.path.dirname(os.path.abspath(__file__))
    _plugin_root = os.path.dirname(_here)
    if _plugin_root not in sys.path:
        sys.path.insert(0, _plugin_root)
    from helpers.llama_cpp_manager import BackendManager
    from helpers import fleet_models

ROLES = {"chat", "utility", "embedding"}


class SetRouterAliasModel(ApiHandler):
    async def process(self, input: dict, request: Request) -> dict:
        slot_id = input.get("slot_id", "slot_router")
        alias = str(input.get("alias") or "").strip()
        model_path = str(input.get("model_path") or "").strip()

        if alias not in ROLES:
            return {"ok": False, "error": "alias must be chat, utility, or embedding"}
        if not model_path:
            return {"ok": False, "error": "model_path is required"}

        mgr = BackendManager.get_instance()
        slot_cfg = mgr._slot_configs.get(slot_id)
        if not slot_cfg:
            return {"ok": False, "error": f"Slot '{slot_id}' not found"}
        if not slot_cfg.get("router_mode"):
            return {"ok": False, "error": f"Slot '{slot_id}' is not in router mode"}

        write_result = fleet_models.write_preset_ini(alias=alias, model_path=model_path)
        if not write_result.get("ok"):
            return {
                "ok": False,
                "stage": "write_preset_ini",
                "error": write_result.get("error", "failed to write preset"),
                "details": write_result,
            }

        restart_result = fleet_models.restart_router()
        return {
            "ok": bool(restart_result.get("ok")),
            "slot_id": slot_id,
            "alias": alias,
            "model_path": model_path,
            "snippet": write_result.get("snippet", ""),
            "backup_path": write_result.get("backup_path", ""),
            "restart": restart_result,
            "restarting": True,
            "error": restart_result.get("error") if not restart_result.get("ok") else "",
        }
