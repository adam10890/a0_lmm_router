"""
api/fleet_reconnect.py — HTTP-based fleet detection / reconnect.

Probes the llama.cpp fleet over HTTP (no Docker socket required, so it works
from inside the Agent Zero container) and reports what is ACTUALLY running:
a native Router, a 3-slot fixed fleet, or nothing.

This backs the dashboard "Reconnect" button. Calling it forces a fresh probe
and resets the BackendManager singleton so the next status/slot query reflects
reality rather than a stale cached view.

URL: POST /plugins/a0_lmm_router/fleet_reconnect
Returns: {"ok": True, "mode": "router"|"three_slot"|"idle", "router": {...},
          "slots": {...}, "reset": bool}
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from flask import Request
from helpers.api import ApiHandler

try:
    from usr.plugins.a0_lmm_router.helpers.router_probe import detect_fleet_http
except ImportError:
    _here = os.path.dirname(os.path.abspath(__file__))
    _plugin_root = os.path.dirname(_here)
    if _plugin_root not in sys.path:
        sys.path.insert(0, _plugin_root)
    from helpers.router_probe import detect_fleet_http


def _resolve_conf_path() -> str:
    env_conf = os.environ.get("A0_LMM_ROUTER_CONFIG", "").strip()
    if env_conf and os.path.exists(env_conf):
        return env_conf
    here = Path(__file__).resolve()
    plugin_conf = str(here.parents[1] / "conf" / "llama_cpp_servers.yaml")
    root_conf = str(here.parents[4] / "conf" / "llama_cpp_servers.yaml")
    return root_conf if os.path.exists(root_conf) else plugin_conf


def _read_lmm_hosts() -> dict:
    try:
        import yaml
        with open(_resolve_conf_path(), "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return (data.get("global", {}) or {}).get("lmm_hosts", {}) or {}
    except Exception:
        return {}


class FleetReconnect(ApiHandler):
    async def process(self, input: dict, request: Request) -> dict:
        try:
            lmm_hosts = _read_lmm_hosts()
            detected = detect_fleet_http(lmm_hosts or None)

            # Optionally reset the BackendManager singleton so the next slot
            # query re-reads config + re-probes. Off by default to avoid
            # disturbing in-flight requests; the dashboard sets reset=true.
            reset = bool(input.get("reset", False))
            did_reset = False
            if reset:
                try:
                    from usr.plugins.a0_lmm_router.helpers.llama_cpp_manager import BackendManager
                    BackendManager._instance = None
                    did_reset = True
                except Exception:
                    did_reset = False

            router = detected.get("router")
            summary = None
            if detected.get("mode") == "router" and router:
                summary = {
                    "port": router.get("port"),
                    "healthy": router.get("healthy"),
                    "max_instances": router.get("max_instances"),
                    "model_count": router.get("model_count"),
                    "models": [m.get("display") or m.get("id") for m in router.get("models", [])],
                    "loaded": [m.get("display") or m.get("id")
                               for m in router.get("models", []) if m.get("loaded")],
                    "build_info": router.get("build_info"),
                }

            return {
                "ok": True,
                "mode": detected.get("mode", "unknown"),
                "router": summary,
                "slots": {
                    role: {
                        "port": p.get("port"),
                        "reachable": p.get("reachable"),
                        "healthy": p.get("healthy"),
                    }
                    for role, p in (detected.get("slots") or {}).items()
                },
                "reset": did_reset,
            }
        except Exception as exc:
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
