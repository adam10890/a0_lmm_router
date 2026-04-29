"""
API endpoint: /plugins/a0_lmm_router/lmm_compute_stats

Returns a real-time snapshot of GPU, CPU, RAM, and LMM slot status.
"""
from flask import Request
from helpers.api import ApiHandler


class LmmComputeStats(ApiHandler):
    async def process(self, input: dict, request: Request) -> dict:
        try:
            from usr.plugins.a0_lmm_router.helpers.compute_monitor import get_compute_snapshot
            snapshot = get_compute_snapshot()
            return {"ok": True, **snapshot}
        except Exception as e:
            return {"ok": False, "error": str(e)}
