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


def _debug_log(hypothesis_id: str, location: str, message: str, data: Optional[dict] = None) -> None:
    # #region agent log
    import time
    try:
        log_path = "/a0/usr/plugins/a0_lmm_router/data/debug-e401df.log"
        payload = {
            "sessionId": "e401df",
            "hypothesisId": hypothesis_id,
            "location": location,
            "message": message,
            "data": data or {},
            "timestamp": int(time.time() * 1000),
        }
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        pass
    # #endregion


def _helper_request(method: str, path: str, body: Optional[dict] = None, timeout: int = 30) -> dict:
    """Send an HTTP request to the host helper, return parsed JSON."""
    token = _read_token()
    url = f"{_HELPER_BASE}{path}"
    data = json.dumps(body).encode("utf-8") if body else None

    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Content-Type", "application/json")
    req.add_header("X-Token", token)

    _debug_log(
        "H1-H2",
        "fleet_models.py:_helper_request",
        "host helper request",
        {"method": method, "url": url, "has_token": bool(token), "path": path},
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            parsed = json.loads(resp.read().decode("utf-8"))
            models = parsed.get("models") if isinstance(parsed, dict) else None
            model_count = len(models) if isinstance(models, (dict, list)) else 0
            _debug_log(
                "H2-H3",
                "fleet_models.py:_helper_request",
                "host helper response",
                {"path": path, "ok": parsed.get("ok") if isinstance(parsed, dict) else None, "model_count": model_count},
            )
            return parsed
    except urllib.error.HTTPError as e:
        try:
            parsed = json.loads(e.read().decode("utf-8"))
            _debug_log("H1", "fleet_models.py:_helper_request", "HTTP error", {"path": path, "code": e.code, "body_ok": parsed.get("ok")})
            return parsed
        except Exception:
            _debug_log("H1", "fleet_models.py:_helper_request", "HTTP error unreadable", {"path": path, "code": e.code})
            return {"ok": False, "error": f"HTTP {e.code}", "_router_unreachable": True}
    except urllib.error.URLError as e:
        _debug_log("H1", "fleet_models.py:_helper_request", "URL error", {"path": path, "reason": str(e.reason)})
        return {"ok": False, "error": str(e.reason), "_router_unreachable": True}
    except Exception as e:
        _debug_log("H1", "fleet_models.py:_helper_request", "request exception", {"path": path, "error": str(e)})
        return {"ok": False, "error": str(e), "_router_unreachable": True}


# ---------------------------------------------------------------------------
# HTTP fleet fallback (when host helper is down)
# ---------------------------------------------------------------------------

def _read_lmm_hosts() -> dict:
    try:
        import yaml
        from usr.plugins.a0_lmm_router.helpers.conf_resolver import resolve_conf_path
    except ImportError:
        import yaml
        from conf_resolver import resolve_conf_path
    try:
        with open(resolve_conf_path(__file__), "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        return (data.get("global", {}) or {}).get("lmm_hosts", {}) or {}
    except Exception:
        return {}


def _model_path_from_status(status: dict) -> str:
    args = status.get("args") if isinstance(status, dict) else None
    if not isinstance(args, list):
        return ""
    for idx, arg in enumerate(args):
        if arg == "--model" and idx + 1 < len(args):
            return str(args[idx + 1])
    return ""


def _entry_from_v1_model(item: dict, *, role_hint: str = "") -> tuple[str, dict]:
    alias = str(item.get("id") or item.get("alias") or "")
    status = item.get("status") if isinstance(item.get("status"), dict) else {}
    model_path = _model_path_from_status(status)
    meta = item.get("meta") if isinstance(item.get("meta"), dict) else {}

    file_name = Path(model_path).name if model_path else f"{alias}.gguf"
    rel_parent = ""
    if model_path.startswith("/models/"):
        rel = model_path[len("/models/"):]
        rel_parent = str(Path(rel).parent) if "/" in rel else ""

    stem = Path(file_name).stem if file_name else alias
    model_id = stem or alias

    size_bytes = meta.get("size") or 0
    try:
        size_gb = round(int(size_bytes) / (1024 ** 3), 2)
    except (TypeError, ValueError):
        size_gb = 0.0

    hint = role_hint or alias
    if hint not in ("chat", "utility", "embedding", "vision", "reasoning"):
        hint = "utility"

    loaded = str(status.get("value") or "").lower() == "loaded"

    return model_id, {
        "file": file_name,
        "path": rel_parent,
        "repo_id": "router",
        "size_gb": size_gb,
        "role_hint": hint,
        "model_path": model_path,
        "n_ctx_train": meta.get("n_ctx_train"),
        "n_layer": meta.get("n_layer"),
        "n_embd": meta.get("n_embd"),
        "assigned_slot": alias,
        "loaded": loaded,
        "source": "router_http",
    }


def _list_models_from_http_fleet() -> dict:
    """Build installed-models map from live llama.cpp HTTP /v1/models."""
    try:
        from usr.plugins.a0_lmm_router.helpers.router_probe import detect_fleet_http, _http_get_json
    except ImportError:
        from router_probe import detect_fleet_http, _http_get_json

    lmm_hosts = _read_lmm_hosts()
    detected = detect_fleet_http(lmm_hosts or None)
    models: dict = {}

    if detected.get("mode") == "router":
        router = detected.get("router") or {}
        host = str(router.get("host") or "host.docker.internal")
        port = int(router.get("port") or detected.get("primary_port") or 8080)
        payload = _http_get_json(f"http://{host}:{port}/v1/models")
        if isinstance(payload, dict):
            for item in payload.get("data", []) or []:
                if not isinstance(item, dict):
                    continue
                mid, entry = _entry_from_v1_model(item)
                models[mid] = entry
    elif detected.get("mode") == "three_slot":
        for role, probe in (detected.get("slots") or {}).items():
            if not isinstance(probe, dict) or not probe.get("reachable"):
                continue
            host = str(probe.get("host") or "host.docker.internal")
            port = int(probe.get("port") or 8080)
            payload = _http_get_json(f"http://{host}:{port}/v1/models")
            if not isinstance(payload, dict):
                continue
            for item in payload.get("data", []) or []:
                if not isinstance(item, dict):
                    continue
                mid, entry = _entry_from_v1_model(item, role_hint=role)
                models[mid] = entry

    _debug_log(
        "FIX",
        "fleet_models.py:_list_models_from_http_fleet",
        "router http fallback",
        {"mode": detected.get("mode"), "model_count": len(models)},
    )
    return {"models": models, "mode": detected.get("mode", "idle")}


def list_models() -> dict:
    """Return all models known to the fleet (host manifest, with HTTP fallback)."""
    result = _helper_request("GET", "/models/list")
    models = result.get("models") if isinstance(result, dict) else None
    has_models = bool(models) if isinstance(models, dict) else bool(models)

    if result.get("ok") and has_models and not result.get("_router_unreachable"):
        return result

    if not result.get("_router_unreachable") and result.get("ok"):
        # Helper reachable but manifest empty — still return as-is.
        return result

    fallback = _list_models_from_http_fleet()
    fb_models = fallback.get("models") or {}
    if fb_models:
        return {
            "ok": True,
            "models": fb_models,
            "models_dir": result.get("models_dir", ""),
            "source": "router_http",
            "host_helper_unreachable": True,
            "message": (
                "Host helper unreachable — showing models from live router /v1/models. "
                "Start lmm_host_helper.py on the host for install/assign controls."
            ),
        }

    if result.get("_router_unreachable"):
        result.setdefault(
            "message",
            "Host helper unreachable and no live llama.cpp fleet answered over HTTP.",
        )
    return result


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


def warm_params_cache(force_refresh: bool = False, restart: bool = True) -> dict:
    """Plan/cache llama.cpp params for all fleet slots and refresh models_preset.ini."""
    return _helper_request(
        "POST",
        "/router/warm_params",
        {"force_refresh": force_refresh, "restart": restart},
        timeout=180,
    )


def record_params_success(role: str = "", model_path: str = "", all_roles: bool = False) -> dict:
    """Mark the last successful model+params for a role (seeds future planning)."""
    body: dict = {}
    if all_roles:
        body["all"] = True
    else:
        body["role"] = role
        body["model_path"] = model_path
    return _helper_request("POST", "/router/record_success", body, timeout=30)


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
