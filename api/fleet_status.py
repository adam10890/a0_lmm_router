"""
api/fleet_status.py — Fleet status endpoint.
Proxies through fleet_models.fleet_status() and hf_token_status().
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


class FleetStatus(ApiHandler):
    async def process(self, input: dict, request: Request) -> dict:
        """Get fleet status including slots, health, and HF token state."""
        try:
            result = fleet_models.fleet_status()
            if result.get("ok"):
                hf_result = fleet_models.hf_token_status()
                return {
                    "ok": True,
                    "fleet_status": {
                        "host_helper": result.get("stdout", ""),
                        "image_version": result.get("llama_cpp_image", {}).get("digest", "unknown"),
                    },
                    "hf_token_configured": hf_result.get("hf_token_present", False),
                }
            return {"ok": False, "error": result.get("error", "Unknown error")}
        except Exception as e:
            return {"ok": False, "error": str(e)}
