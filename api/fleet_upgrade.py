"""
api/fleet_upgrade.py — Fleet upgrade endpoint.
Proxies through fleet_models.fleet_upgrade().
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


class FleetUpgrade(ApiHandler):
    async def process(self, input: dict, request: Request) -> dict:
        """Pull latest llama.cpp image and restart fleet."""
        try:
            result = fleet_models.fleet_upgrade()
            if result.get("ok"):
                return {"ok": True, "message": "Fleet upgrade started"}
            return {"ok": False, "error": result.get("error", "Unknown error")}
        except Exception as e:
            return {"ok": False, "error": str(e)}
