import json
import time

from flask import Request
from helpers.api import ApiHandler

_DEBUG_LOG = "/a0/usr/plugins/a0_lmm_router/data/debug-e401df.log"


def _dbg(hypothesis_id: str, location: str, message: str, data: dict | None = None) -> None:
    # #region agent log
    try:
        payload = {
            "sessionId": "e401df",
            "hypothesisId": hypothesis_id,
            "location": location,
            "message": message,
            "data": data or {},
            "timestamp": int(time.time() * 1000),
        }
        with open(_DEBUG_LOG, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        pass
    # #endregion


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
        models = result.get("models")
        model_count = len(models) if isinstance(models, dict) else (len(models) if isinstance(models, list) else 0)
        _dbg(
            "H1-H3",
            "llamacpp_list_models.py:process",
            "list_models result",
            {
                "unreachable": bool(result.get("_router_unreachable")),
                "host_helper_unreachable": bool(result.get("host_helper_unreachable")),
                "source": result.get("source", "host_helper"),
                "ok": result.get("ok"),
                "model_count": model_count,
                "models_type": type(models).__name__ if models is not None else "none",
                "models_dir": result.get("models_dir", ""),
                "error": result.get("error", ""),
            },
        )
        return result
