"""
api/set_hf_token.py — Set HuggingFace token endpoint.
Proxies through fleet_models.set_hf_token().
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


class SetHfToken(ApiHandler):
    async def process(self, input: dict, request: Request) -> dict:
        """Set the HF token on the host helper."""
        try:
            token = input.get("token", "").strip()
            if not token:
                return {"ok": False, "error": "token is required"}

            result = fleet_models.set_hf_token(token)
            if result.get("ok"):
                return {"ok": True, "message": "HF token set successfully"}
            return {"ok": False, "error": result.get("error", "Unknown error")}
        except Exception as e:
            return {"ok": False, "error": str(e)}
