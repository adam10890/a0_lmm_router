"""
api/job_status.py — Job status endpoint.
Proxies through fleet_models.job_status().
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


class JobStatus(ApiHandler):
    async def process(self, input: dict, request: Request) -> dict:
        """Poll a download job's progress."""
        try:
            job_id = input.get("job_id", "").strip()
            if not job_id:
                return {"ok": False, "error": "job_id is required"}

            result = fleet_models.job_status(job_id)
            if result.get("ok"):
                return {
                    "ok": True,
                    "job": {
                        "id": result.get("job_id", job_id),
                        "status": result.get("status", "unknown"),
                        "progress": result.get("percent", 0),
                        "downloaded_bytes": result.get("downloaded_bytes", 0),
                        "total_bytes": result.get("total_bytes", 0),
                        "local_path": result.get("local_path", ""),
                        "model_id": result.get("model_id", ""),
                        "error": result.get("error", ""),
                    },
                }
            return {"ok": False, "error": result.get("error", "Unknown error")}
        except Exception as e:
            return {"ok": False, "error": str(e)}
