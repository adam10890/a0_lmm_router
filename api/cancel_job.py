"""
api/cancel_job.py — Cancel job endpoint.
Proxies through fleet_models.cancel_job().
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


class CancelJob(ApiHandler):
    async def process(self, input: dict, request: Request) -> dict:
        """Cancel an in-flight download job."""
        try:
            job_id = input.get("job_id", "").strip()
            if not job_id:
                return {"ok": False, "error": "job_id is required"}

            result = fleet_models.cancel_job(job_id)
            if result.get("ok"):
                return {"ok": True, "message": "Job cancelled"}
            return {"ok": False, "error": result.get("error", "Unknown error")}
        except Exception as e:
            return {"ok": False, "error": str(e)}
