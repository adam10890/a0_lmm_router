"""API: warm llama.cpp per-model parameter cache and regenerate preset."""
from __future__ import annotations

from flask import Request
from helpers.api import ApiHandler

try:
    from usr.plugins.a0_lmm_router.helpers import fleet_models
except ImportError:
    from helpers import fleet_models  # type: ignore


class LmmWarmParams(ApiHandler):
    async def process(self, input: dict, request: Request) -> dict:
        force_refresh = bool(input.get("force_refresh", False))
        restart = bool(input.get("restart", True))
        result = fleet_models.warm_params_cache(force_refresh=force_refresh, restart=restart)
        if result.get("_router_unreachable"):
            return {
                "ok": False,
                "error": "Host helper unreachable. Is lmm_host_helper.py running on the host?",
                "detail": result.get("error"),
            }
        return result
