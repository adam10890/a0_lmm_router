"""
api/assign_model.py — Assign model to slot endpoint.
Proxies through fleet_models.assign_model().
"""
from __future__ import annotations

from flask import Request
from helpers.api import ApiHandler

try:
    from usr.plugins.a0_lmm_router.helpers import fleet_models
except ImportError:
    import sys, os
    _here = os.path.dirname(os.path.abspath(__file__))
    _plugin_root = os.path.dirname(os.path.dirname(_here))
    if _plugin_root not in sys.path:
        sys.path.insert(0, _plugin_root)
    from helpers import fleet_models


class AssignModel(ApiHandler):
    async def process(self, input: dict, request: Request) -> dict:
        """Assign a model to a slot (chat/utility/embedding)."""
        try:
            slot = input.get("slot", "").strip()
            model_id = input.get("model_id", "").strip()
            apply_now = input.get("apply_now", True)

            if not slot:
                return {"ok": False, "error": "slot is required"}
            if not model_id:
                return {"ok": False, "error": "model_id is required"}

            result = fleet_models.assign_model(slot, model_id, apply_now)
            if result.get("ok"):
                return {"ok": True, "message": "Model assigned"}
            return {"ok": False, "error": result.get("error", "Unknown error")}
        except Exception as e:
            return {"ok": False, "error": str(e)}
