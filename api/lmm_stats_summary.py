"""
API endpoint: /plugins/a0_lmm_router/lmm_stats_summary

Returns usage statistics, savings estimates, and failover tracking for the LMM Router.

Query params:
    window: "24h" (default), "7d", or "30d"
"""
from flask import Request
from helpers.api import ApiHandler


class LmmStatsSummary(ApiHandler):
    async def process(self, input: dict, request: Request) -> dict:
        try:
            from usr.plugins.a0_lmm_router.helpers.stats_tracker import get_stats_summary, get_slot_stats
            
            window = input.get("window", "24h")
            if window not in ("24h", "7d", "30d"):
                window = "24h"
            
            summary = get_stats_summary(window=window)
            return {"ok": True, "stats": summary}
        except Exception as e:
            return {"ok": False, "error": str(e)}
