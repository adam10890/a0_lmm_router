"""
api/load_model.py — Load model into slot (combined assign + context calc + restart).

Proxies through fleet_models.load_model().
Inspired by lmstudio-js client.llm.load() — one call does everything.
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


class LoadModel(ApiHandler):
    async def process(self, input: dict, request: Request) -> dict:
        """Load a model into a slot with auto-calculated context window.

        Body: { slot, model_id, ctx_size? }
        """
        try:
            slot = input.get("slot", "").strip()
            model_id = input.get("model_id", "").strip()
            ctx_size = input.get("ctx_size")  # optional override

            if not slot:
                return {"ok": False, "error": "slot is required"}
            if not model_id:
                return {"ok": False, "error": "model_id is required"}

            result = fleet_models.load_model(slot, model_id, ctx_size)
            return result
        except Exception as e:
            return {"ok": False, "error": str(e)}
