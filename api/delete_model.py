"""
api/delete_model.py — Delete model endpoint.
Proxies through fleet_models.delete_model().
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


class DeleteModel(ApiHandler):
    async def process(self, input: dict, request: Request) -> dict:
        """Delete a model from the host volume."""
        try:
            model_id = input.get("model_id", "").strip()
            if not model_id:
                return {"ok": False, "error": "model_id is required"}

            result = fleet_models.delete_model(model_id)
            if result.get("ok"):
                return {"ok": True, "message": "Model deleted"}
            return {"ok": False, "error": result.get("error", "Unknown error")}
        except Exception as e:
            return {"ok": False, "error": str(e)}
