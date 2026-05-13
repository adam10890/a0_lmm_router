from flask import Request
from helpers.api import ApiHandler


class LmmModelInstall(ApiHandler):
    async def process(self, input: dict, request: Request) -> dict:
        try:
            from usr.plugins.a0_lmm_router.helpers.fleet_models import install_model
        except ImportError:
            import sys, os
            _here = os.path.dirname(os.path.abspath(__file__))
            _plugin_root = os.path.dirname(os.path.dirname(_here))
            if _plugin_root not in sys.path:
                sys.path.insert(0, _plugin_root)
            from helpers.fleet_models import install_model

        repo_id = input.get("repo_id", "").strip()
        filename = input.get("filename", "").strip()
        role = input.get("role", "").strip() or None

        if not repo_id or not filename:
            return {"ok": False, "error": "repo_id and filename are required"}

        result = install_model(repo_id=repo_id, filename=filename, role=role)
        if result.get("_router_unreachable"):
            return {"ok": False, "error": "Host helper unreachable. Is lmm_host_helper.py running?", "detail": result.get("error")}
        # Return job_id if available for GUI tracking
        if result.get("ok") and result.get("job_id"):
            return {"ok": True, "job_id": result["job_id"], "message": "Installation started"}
        return result
