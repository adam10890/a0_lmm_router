"""HTTP-based fleet probe — detect a running llama.cpp Router or fixed slots
WITHOUT a Docker socket.

Why this exists
---------------
`fleet_mode.detect_fleet_mode()` shells out to `docker ps`. That works on the
host but NOT from inside the Agent Zero container, which has no Docker socket.
When the plugin runs as an A0 plugin it can still reach the llama.cpp fleet
over HTTP (host.docker.internal:<port>), so this module detects reality purely
over HTTP:

  * GET /props        — a llama.cpp router answers with {"role": "router", ...};
                        a fixed-model server answers with a different role.
  * GET /health       — {"status": "ok"} when the server is up.
  * GET /v1/models    — the registered models (router: many; fixed slot: one).

This is the single source of truth used by the dashboard so it shows what is
ACTUALLY running, not just what conf/llama_cpp_servers.yaml describes.

No third-party dependencies (urllib only) so it is safe to import from both
api/ handlers and helpers/ inside the A0 container.
"""
from __future__ import annotations

import json
import urllib.request
from typing import Any, Dict, List, Optional

DEFAULT_TIMEOUT = 2.5


def _http_get_json(url: str, timeout: float = DEFAULT_TIMEOUT) -> Optional[dict]:
    try:
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            if 200 <= resp.status < 300:
                return json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None
    return None


def _clean_model_id(raw: str) -> str:
    """Strip a /models/... path and .gguf suffix to a display-friendly id."""
    mid = (raw or "").replace(".gguf", "")
    if "/" in mid:
        mid = mid.rsplit("/", 1)[-1]
    return mid


def probe_endpoint(host: str, port: int, timeout: float = DEFAULT_TIMEOUT) -> Dict[str, Any]:
    """Probe a single llama.cpp HTTP endpoint.

    Returns a dict that is always shaped the same way:

        {
          "host": str, "port": int,
          "reachable": bool,          # /health or /props answered
          "healthy": bool,            # /health == ok
          "is_router": bool,          # /props.role == "router"
          "role": str,                # raw role from /props ("" if unknown)
          "max_instances": int|None,  # router only
          "models_autoload": bool|None,
          "build_info": str,
          "models": [ {"id","aliases","loaded"} ... ],
          "model_count": int,
        }
    """
    base = f"http://{host}:{port}"
    out: Dict[str, Any] = {
        "host": host,
        "port": port,
        "reachable": False,
        "healthy": False,
        "is_router": False,
        "role": "",
        "max_instances": None,
        "models_autoload": None,
        "build_info": "",
        "models": [],
        "model_count": 0,
    }

    health = _http_get_json(f"{base}/health", timeout)
    if health is not None:
        out["reachable"] = True
        out["healthy"] = str(health.get("status", "")).lower() == "ok"

    props = _http_get_json(f"{base}/props", timeout)
    if props is not None:
        out["reachable"] = True
        role = str(props.get("role", ""))
        out["role"] = role
        out["is_router"] = role == "router"
        if "max_instances" in props:
            try:
                out["max_instances"] = int(props["max_instances"])
            except (TypeError, ValueError):
                pass
        if "models_autoload" in props:
            out["models_autoload"] = bool(props["models_autoload"])
        out["build_info"] = str(props.get("build_info", ""))

    models_payload = _http_get_json(f"{base}/v1/models", timeout)
    if models_payload is not None:
        out["reachable"] = True
        data = models_payload.get("data", []) if isinstance(models_payload, dict) else []
        models: List[dict] = []
        for item in data:
            if not isinstance(item, dict):
                continue
            raw_id = str(item.get("id") or item.get("alias") or "")
            status = item.get("status") or {}
            models.append({
                "id": raw_id,
                "display": _clean_model_id(raw_id),
                "loaded": str(status.get("value") or "").lower() == "loaded"
                if isinstance(status, dict) else False,
            })
        out["models"] = models
        out["model_count"] = len(models)

    return out


def detect_fleet_http(
    lmm_hosts: Optional[Dict[str, str]] = None,
    timeout: float = DEFAULT_TIMEOUT,
) -> Dict[str, Any]:
    """Detect fleet state over HTTP from a role->host:port map.

    `lmm_hosts` mirrors conf/llama_cpp_servers.yaml `global.lmm_hosts`, e.g.
        {"chat": "host.docker.internal:8080",
         "utility": "host.docker.internal:8088",
         "embedding": "host.docker.internal:8082"}
    Defaults to the standard host.docker.internal layout when omitted.

    Returns:
        {
          "mode": "router" | "three_slot" | "idle",
          "router": <probe dict> | None,   # the router endpoint if found
          "slots": { role: <probe dict> },  # per-role probes (3-slot mode)
          "primary_port": int,
        }
    """
    hosts = lmm_hosts or {
        "chat": "host.docker.internal:8080",
        "utility": "host.docker.internal:8088",
        "embedding": "host.docker.internal:8082",
    }

    def split(hp: str, default_port: int) -> tuple[str, int]:
        hp = (hp or "").replace("http://", "").replace("https://", "").split("/", 1)[0]
        if ":" in hp:
            h, p = hp.rsplit(":", 1)
            return h, int(p) if p.isdigit() else default_port
        return hp or "host.docker.internal", default_port

    # The router (if any) shares the chat port — that is the compose contract.
    chat_host, chat_port = split(hosts.get("chat", "host.docker.internal:8080"), 8080)
    primary = probe_endpoint(chat_host, chat_port, timeout)

    if primary.get("is_router"):
        return {
            "mode": "router",
            "router": primary,
            "slots": {},
            "primary_port": chat_port,
        }

    # Not a router on the chat port — probe each role as a fixed slot.
    slots: Dict[str, Any] = {}
    any_up = False
    default_ports = {"chat": 8080, "utility": 8088, "embedding": 8082}
    for role, hp in hosts.items():
        h, p = split(hp, default_ports.get(role, 8080))
        probe = probe_endpoint(h, p, timeout) if (role != "chat") else primary
        slots[role] = probe
        any_up = any_up or probe.get("reachable", False)

    return {
        "mode": "three_slot" if any_up else "idle",
        "router": None,
        "slots": slots,
        "primary_port": chat_port,
    }
