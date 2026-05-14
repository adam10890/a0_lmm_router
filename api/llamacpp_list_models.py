from flask import Request
from helpers.api import ApiHandler


class LlamacppListModels(ApiHandler):
    async def process(self, input: dict, request: Request) -> dict:
        try:
            from usr.plugins.a0_lmm_router.helpers.fleet_models import list_models
        except ImportError:
            import sys, os
            _here = os.path.dirname(os.path.abspath(__file__))
            _plugin_root = os.path.dirname(os.path.dirname(_here))
            if _plugin_root not in sys.path:
                sys.path.insert(0, _plugin_root)
            from helpers.fleet_models import list_models

        result = list_models()
        if result.get("_router_unreachable"):
            return {"ok": True, "models_dir": "", "models": [], "message": "Host helper unreachable. Models are managed in the external LMM fleet."}
        return result
