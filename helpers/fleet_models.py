"""
helpers/fleet_models.py — Fleet model management abstraction.

All model operations (list, install, delete, assign, status, verify) route
through a backend adapter. MVP uses the host helper adapter; a future
Variant B adapter can be swapped in without touching the GUI or router APIs.

Security: every call to the host helper reuses the existing X-Token auth.
"""
from __future__ import annotations

import json
import logging
import os
import urllib.request
import urllib.error
from pathlib import Path
from typing import Optional

log = logging.getLogger("a0_lmm_router.fleet_models")

# ---------------------------------------------------------------------------
# Host-helper adapter (MVP)
# ---------------------------------------------------------------------------

_HELPER_BASE = os.environ.get("A0_LMM_HELPER_URL", "http://host.docker.internal:55501")
_TOKEN_CANDIDATES = ("/host/a0_lmm_host.key", "/a0/tmp/lmm_host_token")


def _read_token() -> str:
    env_path = os.environ.get("A0_LMM_HOST_TOKEN_PATH", "").strip()
    for path in ((env_path,) if env_path else ()) + _TOKEN_CANDIDATES:
        try:
            p = Path(path)
            if p.is_file():
                return p.read_text(encoding="utf-8").strip()
        except Exception:
            continue
    return ""


def _helper_request(method: str, path: str, body: Optional[dict] = None, timeout: int = 30) -> dict:
    """Send an HTTP request to the host helper, return parsed JSON."""
    token = _read_token()
    url = f"{_HELPER_BASE}{path}"
    data = json.dumps(body).encode("utf-8") if body else None

    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Content-Type", "application/json")
    req.add_header("X-Token", token)

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        try:
            return json.loads(e.read().decode("utf-8"))
        except Exception:
            return {"ok": False, "error": f"HTTP {e.code}", "_router_unreachable": True}
    except urllib.error.URLError as e:
        return {"ok": False, "error": str(e.reason), "_router_unreachable": True}
    except Exception as e:
        return {"ok": False, "error": str(e), "_router_unreachable": True}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def list_models() -> dict:
    """Return all models known to the fleet (from host manifest)."""
    return _helper_request("GET", "/models/list")


def install_model(repo_id: str, filename: str, role: Optional[str] = None) -> dict:
    """Start a model download job on the host helper.

    Returns { job_id, status: "queued" } on success.
    """
    body = {"repo_id": repo_id, "filename": filename}
    if role:
        body["role"] = role
    return _helper_request("POST", "/models/install", body, timeout=10)


def job_status(job_id: str) -> dict:
    """Poll a download job's progress."""
    return _helper_request("GET", f"/models/jobs/{job_id}")


def cancel_job(job_id: str) -> dict:
    """Cancel an in-flight download job."""
    return _helper_request("POST", f"/models/jobs/{job_id}/cancel")


def delete_model(model_id: str) -> dict:
    """Delete a model from the host volume."""
    return _helper_request("POST", "/models/delete", {"model_id": model_id})


def assign_model(slot: str, model_id: str, apply_now: bool = True) -> dict:
    """Assign a model to a slot (chat/utility/embed)."""
    return _helper_request("POST", "/models/assign", {
        "slot": slot,
        "model_id": model_id,
        "apply_now": apply_now,
    }, timeout=180)


def load_model(slot: str, model_id: str, ctx_size: int | None = None) -> dict:
    """Load a model into a slot with auto-calculated context window.

    Combined endpoint: assign + context calculation + container restart.
    Inspired by lmstudio-js client.llm.load() — one call does everything.

    Args:
        slot: chat, utility, or embed
        model_id: ID from the model manifest
        ctx_size: Optional context window override (auto-calculated if None)
    """
    body: dict = {"slot": slot, "model_id": model_id}
    if ctx_size is not None:
        body["ctx_size"] = ctx_size
    return _helper_request("POST", "/models/load", body, timeout=180)


def start_slot(slot: str) -> dict:
    """Start a slot's container via docker compose."""
    return _helper_request("POST", "/models/start", {"slot": slot}, timeout=60)


def stop_slot(slot: str) -> dict:
    """Stop a slot's container via docker compose."""
    return _helper_request("POST", "/models/stop", {"slot": slot}, timeout=60)


def fleet_status() -> dict:
    """Get fleet status including slots, health, and image version."""
    return _helper_request("GET", "/status")


def verify_model(model_id: str) -> dict:
    """Recompute sha256 for a model file."""
    return _helper_request("POST", "/models/verify", {"model_id": model_id})


def hf_token_status() -> dict:
    """Check if HF token is configured on the host."""
    return _helper_request("GET", "/tokens/hf")


def set_hf_token(token: str) -> dict:
    """Set the HF token on the host helper."""
    return _helper_request("POST", "/tokens/hf", {"token": token})


def clear_hf_token() -> dict:
    """Clear the HF token on the host helper."""
    return _helper_request("DELETE", "/tokens/hf")


def _helper_unknown_endpoint(result: dict, path: str) -> bool:
    err = str(result.get("error") or "")
    return "unknown endpoint" in err and path in err


def _helper_router_capabilities() -> set[str]:
    """Probe host helper /health for router endpoint support (new helpers only)."""
    try:
        url = f"{_HELPER_BASE}/health"
        with urllib.request.urlopen(url, timeout=3.0) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        caps = data.get("capabilities") or []
        return {str(c) for c in caps}
    except Exception:
        return set()


def write_preset_ini(alias: str, model_path: str, preset_path: str | None = None) -> dict:
    """Rewrite one alias model line in the router preset on the host."""
    body = {"alias": alias, "model_path": model_path}
    if preset_path:
        body["preset_path"] = preset_path

    caps = _helper_router_capabilities()
    if not caps or "router/write_preset_ini" in caps:
        result = _helper_request("POST", "/router/write_preset_ini", body, timeout=30)
        if result.get("ok") or not _helper_unknown_endpoint(result, "/router/write_preset_ini"):
            if result.get("ok"):
                result["via"] = "host_helper"
            return result

    try:
        from helpers.preset_ini import write_alias_model
    except ImportError:
        from usr.plugins.a0_lmm_router.helpers.preset_ini import write_alias_model  # type: ignore

    local = write_alias_model(alias=alias, model_path=model_path, preset_path=preset_path)
    if local.get("ok"):
        local["host_helper_stale"] = "router/write_preset_ini" not in caps
    return local


def restart_router() -> dict:
    """Restart the single llama.cpp router container."""
    caps = _helper_router_capabilities()
    if not caps or "router/restart" in caps:
        result = _helper_request("POST", "/router/restart", timeout=90)
        if result.get("ok") or not _helper_unknown_endpoint(result, "/router/restart"):
            return result

    return {
        "ok": False,
        "error": "host helper is outdated (missing /router/restart). Re-run start_agent_zero.bat to restart the helper, then run: docker restart a0-llama-router",
        "host_helper_stale": True,
    }


def fleet_upgrade() -> dict:
    """Pull latest llama.cpp image and restart fleet."""
    return _helper_request("POST", "/fleet/upgrade", timeout=300)


def fleet_upgrade_rollback() -> dict:
    """Rollback to previous llama.cpp image."""
    return _helper_request("POST", "/fleet/upgrade/rollback", timeout=300)
