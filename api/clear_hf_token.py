"""
api/clear_hf_token.py — Clear HuggingFace token endpoint.
Proxies through fleet_models.clear_hf_token().
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


class ClearHfToken(ApiHandler):
    async def process(self, input: dict, request: Request) -> dict:
        """Clear the HF token on the host helper."""
        try:
            result = fleet_models.clear_hf_token()
            if result.get("ok"):
                return {"ok": True, "message": "HF token cleared"}
            return {"ok": False, "error": result.get("error", "Unknown error")}
        except Exception as e:
            return {"ok": False, "error": str(e)}
